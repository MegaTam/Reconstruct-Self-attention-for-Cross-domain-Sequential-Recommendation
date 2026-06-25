from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import math

import torch
from torch.optim import AdamW

from autocdsr_bert4rec import AutoCDSRBERT4Rec
from dataloader import build_all_dataloaders
from masking import (
    build_train_batch_random_masking,
    build_eval_batch_next_token,
)
from remap import IDMapper


@dataclass
class TrainConfig:
    data_root: str = "data"
    mapper_dir: str = "artifacts/remap"
    save_dir: str = "artifacts/checkpoints"

    batch_size: int = 4
    num_workers: int = 0
    max_length: int | None = None
    max_records_per_split: int | None = None

    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_seq_len: int = 800

    masking_probability: float = 0.15

    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 3

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every_steps: int = 10
    eps: float = 1e-12

    # preference-aware / full version
    pref_K: int = 10
    target_pref_index: int = 1

    # Frank-Wolfe
    fw_max_iter: int = 30
    fw_tol: float = 1e-6


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def save_checkpoint(
    model: AutoCDSRBERT4Rec,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_total_loss: float,
    eval_rec_loss: float,
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_total_loss": train_total_loss,
            "eval_rec_loss": eval_rec_loss,
        },
        save_path,
    )


def _flatten_grads_with_params(
    grads: Iterable[torch.Tensor | None],
    params: Iterable[torch.nn.Parameter],
) -> torch.Tensor:
    vecs: List[torch.Tensor] = []

    for g, p in zip(grads, params):
        if g is None:
            vecs.append(torch.zeros_like(p).reshape(-1))
        else:
            vecs.append(g.reshape(-1))

    if len(vecs) == 0:
        raise RuntimeError("No gradients were collected.")
    return torch.cat(vecs)


def _get_shared_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("Model has no trainable parameters.")
    return params


def build_preference_vectors(K: int = 10, device: torch.device | None = None) -> torch.Tensor:
    """
    p_k = (cos(k*pi/(2K)), sin(k*pi/(2K))), k=0,...,K
    shape: [K+1, 2]
    """
    prefs = []
    for k in range(K + 1):
        angle = k * math.pi / (2 * K)
        prefs.append([math.cos(angle), math.sin(angle)])
    return torch.tensor(prefs, dtype=torch.float32, device=device)


def get_active_constraints(
    rec_loss: torch.Tensor,
    cd_loss: torch.Tensor,
    preference_vectors: torch.Tensor,   # [K+1, 2]
    target_pref_index: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    S = {p_k | p_k^T L(theta) - p_1^T L(theta) >= 0}

    Returns
    -------
    active_prefs: [m, 2]
    active_indices: [m]
    """
    loss_vec = torch.stack([rec_loss.detach(), cd_loss.detach()])   # [2]
    target_pref = preference_vectors[target_pref_index]             # [2]
    target_score = torch.dot(target_pref, loss_vec)

    active_indices: List[int] = []
    for i, p in enumerate(preference_vectors):
        score = torch.dot(p, loss_vec) - target_score
        if score >= 0:
            active_indices.append(i)

    if len(active_indices) == 0:
        active_indices = [target_pref_index]

    active_indices_t = torch.tensor(active_indices, device=loss_vec.device, dtype=torch.long)
    active_prefs = preference_vectors[active_indices_t]
    return active_prefs, active_indices_t


def constraint_grad_vector(
    rec_loss: torch.Tensor,
    cd_loss: torch.Tensor,
    p_k: torch.Tensor,         # [2]
    p_target: torch.Tensor,    # [2]
    shared_params: List[torch.nn.Parameter],
) -> torch.Tensor:
    """
    gradient of:
        (p_k^T L - p_target^T L)
    """
    coeff_rec = p_k[0] - p_target[0]
    coeff_cd = p_k[1] - p_target[1]

    objective = coeff_rec * rec_loss + coeff_cd * cd_loss

    grads = torch.autograd.grad(
        objective,
        shared_params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    return _flatten_grads_with_params(grads, shared_params)


def frank_wolfe_on_active_constraints(
    grad_list: List[torch.Tensor],   # each [P]
    max_iter: int = 30,
    tol: float = 1e-6,
) -> torch.Tensor:
    """
    Solve beta on simplex using Frank-Wolfe.

    Returns
    -------
    beta: [m]
    """
    m = len(grad_list)
    if m == 1:
        return torch.ones(1, device=grad_list[0].device)

    G = torch.stack(grad_list, dim=0)   # [m, P]
    beta = torch.full((m,), 1.0 / m, device=G.device)

    for _ in range(max_iter):
        g_mix = (beta.unsqueeze(1) * G).sum(dim=0)   # [P]

        # choose t = argmin <g_mix, g_t>
        correlations = torch.mv(G, g_mix)   # [m]
        t = torch.argmin(correlations).item()

        e_t = torch.zeros_like(beta)
        e_t[t] = 1.0

        g_t = G[t]
        diff = g_mix - g_t

        denom = torch.dot(diff, diff)
        if denom.item() < tol:
            break

        eta = torch.dot(g_mix, diff) / denom
        eta = torch.clamp(eta, 0.0, 1.0)

        beta = (1.0 - eta) * beta + eta * e_t

        if eta.item() < tol:
            break

    return beta


def derive_task_weights_from_beta(
    active_prefs: torch.Tensor,   # [m, 2]
    beta: torch.Tensor,           # [m]
    target_pref: torch.Tensor,    # [2]
    eps: float = 1e-12,
) -> tuple[float, float]:
    """
    Engineering approximation:
    - combine active preference vectors by beta
    - then blend with target preference to keep the "main task preferred" spirit

    Returns
    -------
    alpha_rec, alpha_cd
    """
    pref_mix = (beta.unsqueeze(1) * active_prefs).sum(dim=0)   # [2]

    # blend target preference and active preference mixture
    blended = 0.5 * target_pref + 0.5 * pref_mix

    alpha_rec = float(blended[0].item())
    alpha_cd = float(blended[1].item())

    s = alpha_rec + alpha_cd
    if s <= eps:
        alpha_rec, alpha_cd = 0.9, 0.1
    else:
        alpha_rec /= s
        alpha_cd /= s

    return alpha_rec, alpha_cd


def run_train_epoch(
    model: AutoCDSRBERT4Rec,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    padding_token: int,
    masking_token: int,
    masking_probability: float,
    pref_K: int,
    target_pref_index: int,
    fw_max_iter: int,
    fw_tol: float,
    log_every_steps: int = 10,
    eps: float = 1e-12,
) -> tuple[float, float, float, float, float, float, float]:
    model.train()

    total_total_loss = 0.0
    total_rec_loss = 0.0
    total_cd_loss = 0.0
    total_alpha_rec = 0.0
    total_alpha_cd = 0.0
    total_active_count = 0.0
    total_beta_entropy = 0.0
    total_steps = 0

    shared_params = _get_shared_params(model)

    for step, batch in enumerate(train_loader, start=1):
        batch = move_batch_to_device(batch, device)

        train_batch = build_train_batch_random_masking(
            batch=batch,
            padding_token=padding_token,
            masking_token=masking_token,
            masking_probability=masking_probability,
        )
        train_batch = move_batch_to_device(train_batch, device)

        optimizer.zero_grad()

        output = model(
            item_seq=train_batch["model_input_item_seq"],
            event_seq=train_batch["event_seq"],
            padding_mask=train_batch["padding_mask"],
            label_location=train_batch["label_location"],
            labels=train_batch["labels"],
            return_cd_attn_score=True,
        )

        rec_loss = output.loss
        cd_loss = output.cd_attn_score

        if rec_loss is None:
            raise RuntimeError("Training rec_loss is None.")
        if cd_loss is None:
            raise RuntimeError("Training cd_attn_score is None.")

        # build preference vectors
        preference_vectors = build_preference_vectors(K=pref_K, device=rec_loss.device)
        target_pref = preference_vectors[target_pref_index]

        # active constraints
        active_prefs, active_indices = get_active_constraints(
            rec_loss=rec_loss,
            cd_loss=cd_loss,
            preference_vectors=preference_vectors,
            target_pref_index=target_pref_index,
        )

        # gradient vectors for active constraints
        grad_list: List[torch.Tensor] = []
        for p_k in active_prefs:
            g_k = constraint_grad_vector(
                rec_loss=rec_loss,
                cd_loss=cd_loss,
                p_k=p_k,
                p_target=target_pref,
                shared_params=shared_params,
            )
            grad_list.append(g_k)

        # Frank-Wolfe on active constraints
        beta = frank_wolfe_on_active_constraints(
            grad_list=grad_list,
            max_iter=fw_max_iter,
            tol=fw_tol,
        )

        alpha_rec, alpha_cd = derive_task_weights_from_beta(
            active_prefs=active_prefs,
            beta=beta,
            target_pref=target_pref,
            eps=eps,
        )

        total_loss = alpha_rec * rec_loss + alpha_cd * cd_loss
        total_loss.backward()
        optimizer.step()

        total_total_loss += float(total_loss.item())
        total_rec_loss += float(rec_loss.item())
        total_cd_loss += float(cd_loss.item())
        total_alpha_rec += alpha_rec
        total_alpha_cd += alpha_cd
        total_active_count += float(len(active_indices))

        # beta entropy for monitoring whether FW collapses to a single active constraint
        beta_clamped = beta.clamp_min(1e-12)
        beta_entropy = float((-(beta_clamped * beta_clamped.log()).sum()).item())
        total_beta_entropy += beta_entropy

        total_steps += 1

        if step % log_every_steps == 0:
            print(
                f"[Train] step={step:04d} "
                f"total_loss={total_loss.item():.4f} "
                f"rec_loss={rec_loss.item():.4f} "
                f"cd_loss={cd_loss.item():.6f} "
                f"alpha_rec={alpha_rec:.4f} "
                f"alpha_cd={alpha_cd:.4f} "
                f"active={len(active_indices)} "
                f"beta_entropy={beta_entropy:.4f} "
                f"avg_total={total_total_loss / total_steps:.4f}"
            )

    avg_total_loss = total_total_loss / max(total_steps, 1)
    avg_rec_loss = total_rec_loss / max(total_steps, 1)
    avg_cd_loss = total_cd_loss / max(total_steps, 1)
    avg_alpha_rec = total_alpha_rec / max(total_steps, 1)
    avg_alpha_cd = total_alpha_cd / max(total_steps, 1)
    avg_active_count = total_active_count / max(total_steps, 1)
    avg_beta_entropy = total_beta_entropy / max(total_steps, 1)

    return (
        avg_total_loss,
        avg_rec_loss,
        avg_cd_loss,
        avg_alpha_rec,
        avg_alpha_cd,
        avg_active_count,
        avg_beta_entropy,
    )


@torch.no_grad()
def run_eval_epoch(
    model: AutoCDSRBERT4Rec,
    eval_loader,
    device: torch.device,
    padding_token: int,
    masking_token: int,
) -> tuple[float, float]:
    model.eval()

    total_rec_loss = 0.0
    total_cd_loss = 0.0
    total_steps = 0

    for batch in eval_loader:
        batch = move_batch_to_device(batch, device)

        eval_batch = build_eval_batch_next_token(
            batch=batch,
            padding_token=padding_token,
            masking_token=masking_token,
            is_for_inference=False,
        )
        eval_batch = move_batch_to_device(eval_batch, device)

        output = model(
            item_seq=eval_batch["model_input_item_seq"],
            event_seq=eval_batch["event_seq"],
            padding_mask=eval_batch["padding_mask"],
            label_location=eval_batch["label_location"],
            labels=eval_batch["labels"],
            return_cd_attn_score=True,
        )

        rec_loss = output.loss
        cd_loss = output.cd_attn_score

        if rec_loss is None:
            raise RuntimeError("Eval rec_loss is None.")
        if cd_loss is None:
            raise RuntimeError("Eval cd_attn_score is None.")

        total_rec_loss += float(rec_loss.item())
        total_cd_loss += float(cd_loss.item())
        total_steps += 1

    avg_rec_loss = total_rec_loss / max(total_steps, 1)
    avg_cd_loss = total_cd_loss / max(total_steps, 1)
    return avg_rec_loss, avg_cd_loss


def main() -> None:
    cfg = TrainConfig()
    device = torch.device(cfg.device)

    print("Using device:", device)

    mapper = IDMapper.load(cfg.mapper_dir)
    padding_token = mapper.config.item_pad
    masking_token = mapper.config.item_mask

    print("Loaded mapper:")
    print("num_items :", mapper.num_items)
    print("num_events:", mapper.num_events)
    print("item_pad  :", mapper.config.item_pad)
    print("item_mask :", mapper.config.item_mask)
    print("item_unk  :", mapper.config.item_unk)
    print("pref_K    :", cfg.pref_K)
    print("target_pref_index:", cfg.target_pref_index)

    outputs = build_all_dataloaders(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        padding_token=padding_token,
        max_length=cfg.max_length,
        max_records_per_split=cfg.max_records_per_split,
        num_workers=cfg.num_workers,
        mapper=mapper,
    )

    train_loader = outputs["train_loader"]
    eval_loader = outputs["eval_loader"]

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

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_eval_rec_loss = float("inf")

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\n========== Epoch {epoch}/{cfg.num_epochs} ==========")

        (
            train_total,
            train_rec,
            train_cd,
            train_alpha_rec,
            train_alpha_cd,
            train_active_count,
            train_beta_entropy,
        ) = run_train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            padding_token=padding_token,
            masking_token=masking_token,
            masking_probability=cfg.masking_probability,
            pref_K=cfg.pref_K,
            target_pref_index=cfg.target_pref_index,
            fw_max_iter=cfg.fw_max_iter,
            fw_tol=cfg.fw_tol,
            log_every_steps=cfg.log_every_steps,
            eps=cfg.eps,
        )

        eval_rec, eval_cd = run_eval_epoch(
            model=model,
            eval_loader=eval_loader,
            device=device,
            padding_token=padding_token,
            masking_token=masking_token,
        )

        print(
            f"[Epoch {epoch}] "
            f"train_total={train_total:.4f} "
            f"train_rec={train_rec:.4f} "
            f"train_cd={train_cd:.6f} "
            f"alpha_rec={train_alpha_rec:.4f} "
            f"alpha_cd={train_alpha_cd:.4f} "
            f"active={train_active_count:.2f} "
            f"beta_entropy={train_beta_entropy:.4f} | "
            f"eval_rec={eval_rec:.4f} "
            f"eval_cd={eval_cd:.6f}"
        )

        latest_ckpt = Path(cfg.save_dir) / "autocdsr_full_latest.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_total_loss=train_total,
            eval_rec_loss=eval_rec,
            save_path=latest_ckpt,
        )

        if eval_rec < best_eval_rec_loss:
            best_eval_rec_loss = eval_rec
            best_ckpt = Path(cfg.save_dir) / "autocdsr_full_best.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_total_loss=train_total,
                eval_rec_loss=eval_rec,
                save_path=best_ckpt,
            )
            print(f"Saved new best checkpoint to: {best_ckpt}")

    print("\nTraining finished.")
    print("Best eval rec loss:", best_eval_rec_loss)


if __name__ == "__main__":
    main()