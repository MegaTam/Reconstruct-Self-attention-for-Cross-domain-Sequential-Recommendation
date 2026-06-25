from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

from autocdsr_bert4rec import AutoCDSRBERT4Rec
from dataloader import build_all_dataloaders
from masking import build_eval_batch_next_token
from remap import IDMapper


@dataclass
class TestConfig:
    data_root: str = "data"
    mapper_dir: str = "artifacts/remap"
    checkpoint_path: str = "artifacts/checkpoints/autocdsr_full_best.pt"

    batch_size: int = 16
    num_workers: int = 0
    max_length: int | None = None
    max_records_per_split: int | None = None

    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_seq_len: int = 800

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    ks: tuple[int, ...] = (5, 10, 20)

    # sampled ranking
    num_negative_samples: int = 5000
    exclude_seen_items: bool = True
    random_seed: int = 42

    # special item ids after remap
    pad_token: int = 0
    mask_token: int = 1
    unk_token: int = 2

    # paper-defined type ids after remap
    # raw "9" -> 2, raw "4" -> 4
    type_a_event_id: int = 2
    type_b_event_id: int = 4


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def build_model(cfg: TestConfig, mapper: IDMapper, device: torch.device) -> AutoCDSRBERT4Rec:
    model = AutoCDSRBERT4Rec(
        num_items=mapper.num_items,
        num_events=mapper.num_events,
        max_seq_len=cfg.max_seq_len,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        item_pad_idx=mapper.config.item_pad,
        event_pad_idx=mapper.config.event_pad,
    ).to(device)
    return model


def load_checkpoint(model: AutoCDSRBERT4Rec, checkpoint_path: str | Path, device: torch.device) -> dict:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def build_forbidden_mask_for_row(
    model_input_item_seq_row: torch.Tensor,
    true_label: int,
    num_items: int,
    pad_token: int,
    mask_token: int,
    unk_token: int,
    exclude_seen_items: bool = True,
) -> torch.Tensor:
    """
    Returns a bool mask [num_items], True means 'forbidden as negative candidate'.
    """
    forbidden = torch.zeros(num_items, dtype=torch.bool, device=model_input_item_seq_row.device)

    # special tokens are always forbidden
    for sid in (pad_token, mask_token, unk_token):
        if 0 <= sid < num_items:
            forbidden[sid] = True

    if exclude_seen_items:
        seen = model_input_item_seq_row
        seen = seen[(seen != pad_token) & (seen != mask_token) & (seen != unk_token)]
        if seen.numel() > 0:
            forbidden[torch.unique(seen)] = True

    # true label must be allowed
    forbidden[true_label] = False
    return forbidden


def sample_negative_candidates(
    true_label: int,
    model_input_item_seq_row: torch.Tensor,
    num_items: int,
    num_negative_samples: int,
    pad_token: int,
    mask_token: int,
    unk_token: int,
    generator: torch.Generator,
    exclude_seen_items: bool = True,
) -> torch.Tensor:
    """
    Returns candidate ids of shape [1 + num_negative_samples]:
        [true_label, neg1, neg2, ...]
    """
    device = model_input_item_seq_row.device

    forbidden = build_forbidden_mask_for_row(
        model_input_item_seq_row=model_input_item_seq_row,
        true_label=true_label,
        num_items=num_items,
        pad_token=pad_token,
        mask_token=mask_token,
        unk_token=unk_token,
        exclude_seen_items=exclude_seen_items,
    )

    allowed_ids = (~forbidden).nonzero(as_tuple=False).squeeze(1)

    if allowed_ids.numel() == 0:
        raise RuntimeError("No valid negative candidates available.")

    if allowed_ids.numel() >= num_negative_samples:
        perm = torch.randperm(allowed_ids.numel(), generator=generator, device=device)
        negatives = allowed_ids[perm[:num_negative_samples]]
    else:
        idx = torch.randint(
            low=0,
            high=allowed_ids.numel(),
            size=(num_negative_samples,),
            generator=generator,
            device=device,
        )
        negatives = allowed_ids[idx]

    candidates = torch.cat(
        [torch.tensor([true_label], device=device, dtype=torch.long), negatives],
        dim=0,
    )
    return candidates


def compute_ranks_in_sampled_candidates(
    logits: torch.Tensor,                # [B, num_items]
    labels: torch.Tensor,                # [B]
    model_input_item_seq: torch.Tensor,  # [B, L]
    num_items: int,
    num_negative_samples: int,
    pad_token: int,
    mask_token: int,
    unk_token: int,
    generator: torch.Generator,
    exclude_seen_items: bool = True,
) -> torch.Tensor:
    """
    For each row:
    - build sampled candidate set [true + negatives]
    - rank true label inside that candidate set

    Returns
    -------
    ranks: [B], 1-based
    """
    batch_size = logits.size(0)
    ranks = []

    for b in range(batch_size):
        true_label = int(labels[b].item())

        if true_label in (pad_token, mask_token, unk_token):
            ranks.append(torch.tensor(num_negative_samples + 1, device=logits.device))
            continue

        candidates = sample_negative_candidates(
            true_label=true_label,
            model_input_item_seq_row=model_input_item_seq[b],
            num_items=num_items,
            num_negative_samples=num_negative_samples,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            generator=generator,
            exclude_seen_items=exclude_seen_items,
        )

        candidate_scores = logits[b, candidates]
        true_score = candidate_scores[0]

        rank = 1 + (candidate_scores[1:] > true_score).sum()
        ranks.append(rank)

    return torch.stack(ranks, dim=0)


def classify_type_by_target_event(
    event_seq: torch.Tensor,          # [B, L]
    label_location: torch.Tensor,     # [B, 2]
    type_a_event_id: int,
    type_b_event_id: int,
) -> List[str]:
    """
    Paper-defined types:
    - Type A: target event type == remapped raw '9'
    - Type B: target event type == remapped raw '4'
    - Other: all other event types
    """
    batch_size = event_seq.size(0)
    out: List[str] = []

    for b in range(batch_size):
        target_pos = int(label_location[b, 1].item())
        target_event = int(event_seq[b, target_pos].item())

        if target_event == type_a_event_id:
            out.append("A")
        elif target_event == type_b_event_id:
            out.append("B")
        else:
            out.append("Other")

    return out


def recall_at_k_from_ranks(ranks: torch.Tensor, k: int) -> float:
    return float((ranks <= k).float().mean().item())


def hits_at_k_from_ranks(ranks: torch.Tensor, k: int) -> float:
    return float((ranks <= k).float().mean().item())


def ndcg_at_k_from_ranks(ranks: torch.Tensor, k: int) -> float:
    gains = torch.where(
        ranks <= k,
        1.0 / torch.log2(ranks.float() + 1.0),
        torch.zeros_like(ranks, dtype=torch.float),
    )
    return float(gains.mean().item())


def summarize_ranks(ranks: torch.Tensor, ks: tuple[int, ...], prefix: str = "") -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    if ranks.numel() == 0:
        metrics[f"{prefix}num_samples"] = 0.0
        metrics[f"{prefix}mean_rank"] = 0.0
        metrics[f"{prefix}median_rank"] = 0.0
        for k in ks:
            metrics[f"{prefix}Recall@{k}"] = 0.0
            metrics[f"{prefix}Hits@{k}"] = 0.0
            metrics[f"{prefix}NDCG@{k}"] = 0.0
        return metrics

    metrics[f"{prefix}num_samples"] = float(ranks.numel())
    metrics[f"{prefix}mean_rank"] = float(ranks.float().mean().item())
    metrics[f"{prefix}median_rank"] = float(ranks.float().median().item())

    for k in ks:
        metrics[f"{prefix}Recall@{k}"] = recall_at_k_from_ranks(ranks, k)
        metrics[f"{prefix}Hits@{k}"] = hits_at_k_from_ranks(ranks, k)
        metrics[f"{prefix}NDCG@{k}"] = ndcg_at_k_from_ranks(ranks, k)

    return metrics


@torch.no_grad()
def run_test_epoch_sampled(
    model: AutoCDSRBERT4Rec,
    test_loader,
    device: torch.device,
    num_items: int,
    pad_token: int,
    mask_token: int,
    unk_token: int,
    type_a_event_id: int,
    type_b_event_id: int,
    ks: tuple[int, ...],
    num_negative_samples: int,
    random_seed: int,
    exclude_seen_items: bool = True,
) -> Dict[str, float]:
    model.eval()

    all_ranks: List[torch.Tensor] = []
    all_ranks_A: List[torch.Tensor] = []
    all_ranks_B: List[torch.Tensor] = []
    all_ranks_other: List[torch.Tensor] = []

    generator = torch.Generator(device=device)
    generator.manual_seed(random_seed)

    printed_debug = False

    for batch in test_loader:
        batch = move_batch_to_device(batch, device)

        test_batch = build_eval_batch_next_token(
            batch=batch,
            padding_token=pad_token,
            masking_token=mask_token,
            is_for_inference=False,
        )
        test_batch = move_batch_to_device(test_batch, device)

        output = model(
            item_seq=test_batch["model_input_item_seq"],
            event_seq=test_batch["event_seq"],
            padding_mask=test_batch["padding_mask"],
            label_location=test_batch["label_location"],
            labels=test_batch["labels"],
            return_cd_attn_score=False,
        )

        logits = output.logits
        labels = test_batch["labels"]

        ranks = compute_ranks_in_sampled_candidates(
            logits=logits,
            labels=labels,
            model_input_item_seq=test_batch["model_input_item_seq"],
            num_items=num_items,
            num_negative_samples=num_negative_samples,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            generator=generator,
            exclude_seen_items=exclude_seen_items,
        )

        sample_types = classify_type_by_target_event(
            event_seq=test_batch["event_seq"],
            label_location=test_batch["label_location"],
            type_a_event_id=type_a_event_id,
            type_b_event_id=type_b_event_id,
        )

        if not printed_debug:
            topk = torch.topk(logits, k=10, dim=1).indices
            for i in range(min(5, logits.size(0))):
                print(f"\nSample {i}")
                print("label:", labels[i].item())
                print("sampled rank:", ranks[i].item())
                print("target_type:", sample_types[i])
                print("global top10:", topk[i].tolist())
            printed_debug = True

        all_ranks.append(ranks.cpu())

        for i, t in enumerate(sample_types):
            if t == "A":
                all_ranks_A.append(ranks[i:i+1].cpu())
            elif t == "B":
                all_ranks_B.append(ranks[i:i+1].cpu())
            else:
                all_ranks_other.append(ranks[i:i+1].cpu())

    all_ranks_t = torch.cat(all_ranks, dim=0) if len(all_ranks) > 0 else torch.empty(0, dtype=torch.long)
    all_ranks_A_t = torch.cat(all_ranks_A, dim=0) if len(all_ranks_A) > 0 else torch.empty(0, dtype=torch.long)
    all_ranks_B_t = torch.cat(all_ranks_B, dim=0) if len(all_ranks_B) > 0 else torch.empty(0, dtype=torch.long)
    all_ranks_other_t = torch.cat(all_ranks_other, dim=0) if len(all_ranks_other) > 0 else torch.empty(0, dtype=torch.long)

    metrics: Dict[str, float] = {}
    metrics.update(summarize_ranks(all_ranks_t, ks, prefix="Overall_"))
    metrics.update(summarize_ranks(all_ranks_A_t, ks, prefix="TypeA_"))
    metrics.update(summarize_ranks(all_ranks_B_t, ks, prefix="TypeB_"))
    metrics.update(summarize_ranks(all_ranks_other_t, ks, prefix="Other_"))

    return metrics


def print_metric_group(metrics: Dict[str, float], prefix: str, ks: tuple[int, ...]) -> None:
    num_samples = int(metrics[f"{prefix}num_samples"])
    mean_rank = metrics[f"{prefix}mean_rank"]
    median_rank = metrics[f"{prefix}median_rank"]

    print(f"{prefix[:-1]} num_samples: {num_samples}")
    print(f"{prefix[:-1]} mean_rank: {mean_rank:.6f}")
    print(f"{prefix[:-1]} median_rank: {median_rank:.6f}")

    for k in ks:
        print(f"{prefix[:-1]} Recall@{k}: {metrics[f'{prefix}Recall@{k}']:.6f}")
        print(f"{prefix[:-1]} Hits@{k}: {metrics[f'{prefix}Hits@{k}']:.6f}")
        print(f"{prefix[:-1]} NDCG@{k}: {metrics[f'{prefix}NDCG@{k}']:.6f}")


def main() -> None:
    cfg = TestConfig()
    device = torch.device(cfg.device)

    print("Using device:", device)

    mapper = IDMapper.load(cfg.mapper_dir)
    cfg.pad_token = mapper.config.item_pad
    cfg.mask_token = mapper.config.item_mask
    cfg.unk_token = mapper.config.item_unk

    print("Loaded mapper:")
    print("num_items :", mapper.num_items)
    print("num_events:", mapper.num_events)
    print("item_pad  :", cfg.pad_token)
    print("item_mask :", cfg.mask_token)
    print("item_unk  :", cfg.unk_token)
    print("num_negative_samples:", cfg.num_negative_samples)
    print("exclude_seen_items :", cfg.exclude_seen_items)
    print("Type A event id (raw 9):", cfg.type_a_event_id)
    print("Type B event id (raw 4):", cfg.type_b_event_id)

    outputs = build_all_dataloaders(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        padding_token=cfg.pad_token,
        max_length=cfg.max_length,
        max_records_per_split=cfg.max_records_per_split,
        num_workers=cfg.num_workers,
        mapper=mapper,
    )

    test_loader = outputs["test_loader"]
    test_dataset = outputs["test_dataset"]

    print("Dataset sizes:")
    print("test:", len(test_dataset))

    model = build_model(cfg, mapper, device)
    ckpt = load_checkpoint(model, cfg.checkpoint_path, device)

    print(f"Loaded checkpoint from: {cfg.checkpoint_path}")
    if "epoch" in ckpt:
        print("Checkpoint epoch:", ckpt["epoch"])

    metrics = run_test_epoch_sampled(
        model=model,
        test_loader=test_loader,
        device=device,
        num_items=mapper.num_items,
        pad_token=cfg.pad_token,
        mask_token=cfg.mask_token,
        unk_token=cfg.unk_token,
        type_a_event_id=cfg.type_a_event_id,
        type_b_event_id=cfg.type_b_event_id,
        ks=cfg.ks,
        num_negative_samples=cfg.num_negative_samples,
        random_seed=cfg.random_seed,
        exclude_seen_items=cfg.exclude_seen_items,
    )

    print("\n===== Overall Sampled Test Metrics =====")
    print_metric_group(metrics, "Overall_", cfg.ks)

    print("\n===== Type A Metrics =====")
    print_metric_group(metrics, "TypeA_", cfg.ks)

    print("\n===== Type B Metrics =====")
    print_metric_group(metrics, "TypeB_", cfg.ks)

    print("\n===== Other Type Metrics =====")
    print_metric_group(metrics, "Other_", cfg.ks)


if __name__ == "__main__":
    main()