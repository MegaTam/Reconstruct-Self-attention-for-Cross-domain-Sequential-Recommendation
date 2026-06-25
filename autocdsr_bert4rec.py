from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass
class AutoCDSRBERT4RecOutput:
    logits: torch.Tensor                    # [N, num_items]
    hidden_states: torch.Tensor             # [B, L, D]
    pred_hidden: torch.Tensor               # [N, D]
    attention_weights: List[torch.Tensor]   # list of [B, H, L, L]
    cd_attn_score: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


class CustomTransformerBlock(nn.Module):
    """
    A minimal Transformer block that returns attention weights.
    self-attention block
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,                    # [B, L, D]
        padding_mask: torch.Tensor,         # [B, L], True=valid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x: updated hidden states, [B, L, D]
        attn_weights: [B, H, L, L]
        """
        key_padding_mask = ~padding_mask  # MultiheadAttention expects True=ignore

        attn_out, attn_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,   # keep per-head attention
        )
        # attn_weights: [B, H, L, L]

        x = self.attn_norm(x + self.attn_dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.ffn_dropout(ffn_out))

        return x, attn_weights # return updated hidden states and attention weights


class AutoCDSRBERT4Rec(nn.Module):
    """
    Stage A + Stage B:
    BERT4Rec-style model that returns attention weights and computes
    cross-domain attention score.

    Stage B idea:
        Use event_seq to build a cross-domain token-pair mask,
        then measure how much attention mass is assigned to those pairs.
    """

    def __init__(
        self,
        num_items: int,
        num_events: int,
        max_seq_len: int = 800,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        item_pad_idx: int = 0,
        event_pad_idx: int = 0,
    ) -> None:
        super().__init__()

        self.num_items = num_items
        self.num_events = num_events
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.item_pad_idx = item_pad_idx
        self.event_pad_idx = event_pad_idx

        self.item_embedding = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=d_model,
            padding_idx=item_pad_idx,
        )
        self.event_embedding = nn.Embedding(
            num_embeddings=num_events,
            embedding_dim=d_model,
            padding_idx=event_pad_idx,
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=d_model,
        )

        self.input_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                CustomTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_layer_norm = nn.LayerNorm(d_model)
        self.loss_fn = nn.CrossEntropyLoss()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.event_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        if self.item_pad_idx is not None:
            with torch.no_grad():
                self.item_embedding.weight[self.item_pad_idx].fill_(0)

        if self.event_pad_idx is not None:
            with torch.no_grad():
                self.event_embedding.weight[self.event_pad_idx].fill_(0)

    def _build_position_ids(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

    def _gather_label_hidden(
        self,
        hidden_states: torch.Tensor,    # [B, L, D]
        label_location: torch.Tensor,   # [N, 2]
    ) -> torch.Tensor:
        row_idx = label_location[:, 0]
        col_idx = label_location[:, 1]
        return hidden_states[row_idx, col_idx]   # [N, D]

    def _compute_logits(self, pred_hidden: torch.Tensor) -> torch.Tensor:
        item_weight = self.item_embedding.weight   # [num_items, D]
        return pred_hidden @ item_weight.T         # [N, num_items]

    def _build_cross_domain_mask(
        self,
        event_seq: torch.Tensor,        # [B, L]
        padding_mask: torch.Tensor,     # [B, L], True=valid
    ) -> torch.Tensor:
        """
        Returns
        -------
        cross_domain_mask: [B, L, L], bool

        True means:
        - both positions are valid tokens
        - event/domain ids are different
        """
        # [B, L, 1] and [B, 1, L]
        event_i = event_seq.unsqueeze(2)
        event_j = event_seq.unsqueeze(1)

        # domain different
        domain_diff = event_i != event_j   # [B, L, L]

        # both valid
        valid_i = padding_mask.unsqueeze(2)
        valid_j = padding_mask.unsqueeze(1)
        valid_pair = valid_i & valid_j     # [B, L, L]

        cross_domain_mask = domain_diff & valid_pair
        return cross_domain_mask

    def _compute_cd_attention_score(
        self,
        attention_weights: List[torch.Tensor],   # list of [B, H, L, L]
        event_seq: torch.Tensor,                 # [B, L]
        padding_mask: torch.Tensor,              # [B, L]
    ) -> torch.Tensor:
        """
        Compute cross-domain attention score.

        First version:
        - build cross-domain valid-pair mask
        - for each layer, average attention on those pairs across all heads
        - then average across layers

        Returns
        -------
        cd_attn_score: scalar tensor
        """
        cross_domain_mask = self._build_cross_domain_mask(
            event_seq=event_seq,
            padding_mask=padding_mask,
        )  # [B, L, L]

        layer_scores: List[torch.Tensor] = []

        for attn in attention_weights:
            # attn: [B, H, L, L]
            bsz, num_heads, seq_len_1, seq_len_2 = attn.shape

            layer_mask = cross_domain_mask.unsqueeze(1).expand(
                bsz, num_heads, seq_len_1, seq_len_2
            )  # [B, H, L, L]

            layer_mask_float = layer_mask.float()
            masked_sum = (attn * layer_mask_float).sum()
            masked_count = layer_mask_float.sum().clamp_min(1.0)

            layer_score = masked_sum / masked_count
            layer_scores.append(layer_score)

        if len(layer_scores) == 0:
            return torch.tensor(0.0, device=event_seq.device)

        cd_attn_score = torch.stack(layer_scores).mean()
        return cd_attn_score

    def forward(
        self,
        item_seq: torch.Tensor,           # [B, L]
        event_seq: torch.Tensor,          # [B, L]
        padding_mask: torch.Tensor,       # [B, L], True=valid
        label_location: torch.Tensor,     # [N, 2]
        labels: Optional[torch.Tensor] = None,
        return_cd_attn_score: bool = True,
    ) -> AutoCDSRBERT4RecOutput:
        device = item_seq.device
        batch_size, seq_len = item_seq.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        position_ids = self._build_position_ids(batch_size, seq_len, device)

        item_emb = self.item_embedding(item_seq)      # [B, L, D]
        event_emb = self.event_embedding(event_seq)   # [B, L, D]
        pos_emb = self.position_embedding(position_ids)

        x = item_emb + event_emb + pos_emb
        x = self.input_dropout(x)

        all_attention_weights: List[torch.Tensor] = []

        for block in self.blocks:
            x, attn_weights = block(x, padding_mask)
            all_attention_weights.append(attn_weights)

        hidden_states = self.output_layer_norm(x)  # [B, L, D]

        pred_hidden = self._gather_label_hidden(hidden_states, label_location)
        logits = self._compute_logits(pred_hidden)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        cd_attn_score = None
        if return_cd_attn_score:
            cd_attn_score = self._compute_cd_attention_score(
                attention_weights=all_attention_weights,
                event_seq=event_seq,
                padding_mask=padding_mask,
            )

        return AutoCDSRBERT4RecOutput(
            logits=logits,
            hidden_states=hidden_states,
            pred_hidden=pred_hidden,
            attention_weights=all_attention_weights,
            cd_attn_score=cd_attn_score,
            loss=loss,
        )


if __name__ == "__main__":
    from dataloader import build_all_dataloaders
    from masking import build_train_batch_random_masking
    from remap import IDMapper

    mapper = IDMapper.load("artifacts/remap")

    PADDING_TOKEN = mapper.config.item_pad   # 0
    MASKING_TOKEN = mapper.config.item_mask  # 1

    outputs = build_all_dataloaders(
        data_root="data",
        batch_size=2,
        padding_token=PADDING_TOKEN,
        max_length=None,
        max_records_per_split=None,
        num_workers=0,
        mapper=mapper,
    )

    train_loader = outputs["train_loader"]
    train_batch = next(iter(train_loader))

    masked_train_batch = build_train_batch_random_masking(
        batch=train_batch,
        padding_token=PADDING_TOKEN,
        masking_token=MASKING_TOKEN,
        masking_probability=0.15,
    )

    model = AutoCDSRBERT4Rec(
        num_items=mapper.num_items,
        num_events=mapper.num_events,
        max_seq_len=800,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        item_pad_idx=mapper.config.item_pad,
        event_pad_idx=mapper.config.event_pad,
    )

    output = model(
        item_seq=masked_train_batch["model_input_item_seq"],
        event_seq=masked_train_batch["event_seq"],
        padding_mask=masked_train_batch["padding_mask"],
        label_location=masked_train_batch["label_location"],
        labels=masked_train_batch["labels"],
        return_cd_attn_score=True,
    )

    print("==== AutoCDSR Stage B Forward Test ====")
    print("hidden_states shape:", tuple(output.hidden_states.shape))
    print("pred_hidden shape  :", tuple(output.pred_hidden.shape))
    print("logits shape       :", tuple(output.logits.shape))
    print("loss               :", output.loss.item() if output.loss is not None else None)

    print("\nNumber of attention layers:", len(output.attention_weights))
    for i, attn in enumerate(output.attention_weights):
        print(f"Layer {i} attention shape:", tuple(attn.shape))

    print("\nCross-domain attention score:")
    print(output.cd_attn_score.item() if output.cd_attn_score is not None else None)