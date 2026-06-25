from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class LabelFunctionOutput:
    """
    Output of a masking / label transformation.
    """
    sequence: torch.Tensor         # [B, L] or [B, L+1]
    labels: torch.Tensor           # [N]
    label_location: torch.Tensor   # [N, 2]


class RandomMasking:
    """
    BERT4Rec-style random masking.

    Only masks valid item positions (i.e. item != padding_token).
    Ensures each sequence has at least one masked token.
    """

    def __init__(
        self,
        masking_probability: float = 0.15,
        masking_tolerance: float = 0.05,
        max_trials_for_masking: int = 5,
    ) -> None:
        self.masking_probability = masking_probability
        self.masking_tolerance = masking_tolerance
        self.max_trials_for_masking = max_trials_for_masking

    def transform_label(
        self,
        sequence: torch.Tensor,
        padding_token: int,
        masking_token: int,
    ) -> LabelFunctionOutput:
        """
        Parameters
        ----------
        sequence:
            Tensor[B, L], item sequence.
        padding_token:
            Invalid / empty position marker, e.g. -1.
        masking_token:
            Special [MASK] token id.

        Returns
        -------
        LabelFunctionOutput
        """
        sequence = sequence.clone()
        content_mask = sequence != padding_token

        # Remove rows that are fully padding, just in case.
        valid_rows = content_mask.sum(dim=1) > 0
        if not valid_rows.all():
            sequence = sequence[valid_rows]
            content_mask = sequence != padding_token

        row_ids = torch.arange(sequence.size(0), device=sequence.device)

        masking_proportion = 0.0
        counter = 0

        while not (
            self.masking_probability - self.masking_tolerance
            < masking_proportion
            < self.masking_probability + self.masking_tolerance
        ):
            masking_mask = torch.rand(sequence.shape, device=sequence.device) < self.masking_probability
            masking_mask = masking_mask & content_mask

            # average masking ratio over valid tokens
            masking_proportion = (
                (masking_mask.sum(dim=1).float() / content_mask.sum(dim=1).float())
                .mean()
                .item()
            )

            counter += 1
            if counter > self.max_trials_for_masking:
                raise ValueError(
                    f"Cannot get desired masking proportion after {counter} trials."
                )

        # Ensure at least one masked token per row
        one_random_index_by_row = torch.multinomial(
            content_mask.float(), num_samples=1
        ).squeeze(1)
        masking_mask[row_ids, one_random_index_by_row] = True

        labels = sequence[masking_mask]
        sequence[masking_mask] = masking_token
        label_location = masking_mask.nonzero(as_tuple=False)

        return LabelFunctionOutput(
            sequence=sequence,
            labels=labels,
            label_location=label_location,
        )


class NextTokenMasking:
    """
    Mask the last valid item of each sequence.

    If is_for_inference=False:
        replace the last valid item with masking_token and use the original item as label.

    If is_for_inference=True:
        append one extra position at the end and place masking_token there,
        so the model predicts the next item after the observed sequence.
    """

    def __init__(self, is_for_inference: bool = False) -> None:
        self.is_for_inference = is_for_inference

    def transform_label(
        self,
        sequence: torch.Tensor,
        padding_token: int = 0,
        masking_token: int = 1,
    ) -> LabelFunctionOutput:
        sequence = sequence.clone()

        content_mask = sequence != padding_token
        valid_rows = content_mask.sum(dim=1) > 0
        if not valid_rows.all():
            sequence = sequence[valid_rows]
            content_mask = sequence != padding_token

        row_ids = torch.arange(sequence.size(0), device=sequence.device)

        reversed_mask = torch.flip(content_mask, dims=[1])  # [B, L]
        last_offset_from_right = reversed_mask.float().argmax(dim=1)  # [B]
        last_indices = sequence.size(1) - 1 - last_offset_from_right  # [B]

        if self.is_for_inference:
            pad_col = torch.full(
                (sequence.size(0), 1),
                fill_value=padding_token,
                dtype=sequence.dtype,
                device=sequence.device,
            )
            sequence = torch.cat([sequence, pad_col], dim=1)

            new_mask_pos = torch.full(
                (sequence.size(0),),
                fill_value=sequence.size(1) - 1,
                dtype=torch.long,
                device=sequence.device,
            )

            sequence[row_ids, new_mask_pos] = masking_token

            labels = torch.full(
                (sequence.size(0),),
                fill_value=masking_token,
                dtype=sequence.dtype,
                device=sequence.device,
            )
            label_location = torch.stack([row_ids, new_mask_pos], dim=1)

        else:
            labels = sequence[row_ids, last_indices]
            sequence[row_ids, last_indices] = masking_token
            label_location = torch.stack([row_ids, last_indices], dim=1)

        return LabelFunctionOutput(
            sequence=sequence,
            labels=labels,
            label_location=label_location,
        )


def build_train_batch_random_masking(
    batch: Dict[str, torch.Tensor],
    padding_token: int = 0,
    masking_token: int = 1,
    masking_probability: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """
    Convert dataloader output into BERT4Rec-style training batch.
    """
    masker = RandomMasking(masking_probability=masking_probability)

    out = masker.transform_label(
        sequence=batch["item_seq"],
        padding_token=padding_token,
        masking_token=masking_token,
    )

    return {
        "item_seq": batch["item_seq"],                     # original
        "event_seq": batch["event_seq"],                   # original aligned event seq
        "padding_mask": batch["padding_mask"],             # original valid positions
        "lengths": batch["lengths"],

        "model_input_item_seq": out.sequence,              # masked sequence for model
        "labels": out.labels,                              # original masked item ids
        "label_location": out.label_location,              # [N, 2]
    }


def build_eval_batch_next_token(
    batch: Dict[str, torch.Tensor],
    padding_token: int = 0,
    masking_token: int = 1,
    is_for_inference: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Convert dataloader output into next-item prediction batch.
    """
    masker = NextTokenMasking(is_for_inference=is_for_inference)

    out = masker.transform_label(
        sequence=batch["item_seq"],
        padding_token=padding_token,
        masking_token=masking_token,
    )

    # If inference=True, sequence length may become L+1, so event_seq / padding_mask
    # also need one extra padding column to keep shapes aligned.
    event_seq = batch["event_seq"]
    padding_mask = batch["padding_mask"]

    if out.sequence.size(1) != event_seq.size(1):
        extra_cols = out.sequence.size(1) - event_seq.size(1)

        event_pad = torch.full(
            (event_seq.size(0), extra_cols),
            fill_value=padding_token,
            dtype=event_seq.dtype,
            device=event_seq.device,
        )
        mask_pad = torch.zeros(
            (padding_mask.size(0), extra_cols),
            dtype=padding_mask.dtype,
            device=padding_mask.device,
        )

        event_seq = torch.cat([event_seq, event_pad], dim=1)
        padding_mask = torch.cat([padding_mask, mask_pad], dim=1)

    return {
        "item_seq": batch["item_seq"],
        "event_seq": event_seq,
        "padding_mask": padding_mask,
        "lengths": batch["lengths"],

        "model_input_item_seq": out.sequence,
        "labels": out.labels,
        "label_location": out.label_location,
    }


if __name__ == "__main__":
    from dataloader import build_all_dataloaders
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
    eval_loader = outputs["eval_loader"]

    train_batch = next(iter(train_loader))
    eval_batch = next(iter(eval_loader))

    print("==== Original Train Batch ====")
    print("item_seq shape      :", tuple(train_batch["item_seq"].shape))
    print("event_seq shape     :", tuple(train_batch["event_seq"].shape))
    print("padding_mask shape  :", tuple(train_batch["padding_mask"].shape))
    print("lengths             :", train_batch["lengths"])

    masked_train_batch = build_train_batch_random_masking(
        batch=train_batch,
        padding_token=PADDING_TOKEN,
        masking_token=MASKING_TOKEN,
        masking_probability=0.15,
    )

    print("\n==== Random Masking Train Batch ====")
    print("model_input_item_seq shape:", tuple(masked_train_batch["model_input_item_seq"].shape))
    print("labels shape              :", tuple(masked_train_batch["labels"].shape))
    print("label_location shape      :", tuple(masked_train_batch["label_location"].shape))

    print("\nFirst row original item_seq[:20]:")
    print(train_batch["item_seq"][0, :20])

    print("\nFirst row masked item_seq[:20]:")
    print(masked_train_batch["model_input_item_seq"][0, :20])

    print("\nFirst 10 labels:")
    print(masked_train_batch["labels"][:10])

    print("\nFirst 10 label locations:")
    print(masked_train_batch["label_location"][:10])

    next_token_eval_batch = build_eval_batch_next_token(
        batch=eval_batch,
        padding_token=PADDING_TOKEN,
        masking_token=MASKING_TOKEN,
        is_for_inference=False,
    )

    print("\n==== Next Token Eval Batch ====")
    print("model_input_item_seq shape:", tuple(next_token_eval_batch["model_input_item_seq"].shape))
    print("labels shape              :", tuple(next_token_eval_batch["labels"].shape))
    print("label_location shape      :", tuple(next_token_eval_batch["label_location"].shape))

    print("\nEval first row original tail:")
    print(eval_batch["item_seq"][0, -10:])

    print("\nEval first row masked tail:")
    print(next_token_eval_batch["model_input_item_seq"][0, -10:])

    print("\nEval labels:")
    print(next_token_eval_batch["labels"])