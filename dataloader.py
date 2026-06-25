from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

import torch
from torch.utils.data import Dataset, DataLoader

try:
    import tensorflow as tf
except ImportError as e:
    raise ImportError(
        "This dataloader needs TensorFlow to read TFRecord files. "
        "Please install tensorflow first."
    ) from e


if TYPE_CHECKING:
    from remap import IDMapper


@dataclass
class SequenceSample:
    """
    One user sequence sample.

    Attributes
    ----------
    item_seq:
        Sequence from `sequence_data` (already remapped if mapper is provided).
    event_seq:
        Sequence from `sequence_event_type` (already remapped if mapper is provided).
    """
    item_seq: List[int]
    event_seq: List[int]


class AutoCDSRTFRecordDataset(Dataset):
    """
    A simple in-memory map-style dataset for TFRecord(.gz) files.

    This version loads all records into memory, which is convenient for:
    - debugging
    - small to medium scale experiments
    - early-stage reproduction

    Later, if your full dataset becomes very large, you can replace it
    with an IterableDataset / streaming version.
    """

    def __init__(
        self,
        tfrecord_files: Sequence[str | Path],
        compression_type: str = "GZIP",
        max_records: Optional[int] = None,
        validate_alignment: bool = True,
        mapper: Optional["IDMapper"] = None,
    ) -> None:
        if len(tfrecord_files) == 0:
            raise ValueError("No TFRecord files were provided.")

        self.tfrecord_files = [str(Path(f)) for f in tfrecord_files]
        self.compression_type = compression_type
        self.max_records = max_records
        self.validate_alignment = validate_alignment
        self.mapper = mapper

        self.samples: List[SequenceSample] = []
        self._load_all_records()

    def _parse_example(self, raw_record: bytes) -> SequenceSample:
        example = tf.train.Example()
        example.ParseFromString(raw_record)

        features = example.features.feature

        if "sequence_data" not in features:
            raise KeyError("Missing required feature: 'sequence_data'")
        if "sequence_event_type" not in features:
            raise KeyError("Missing required feature: 'sequence_event_type'")

        item_seq = list(features["sequence_data"].int64_list.value)
        event_seq = list(features["sequence_event_type"].int64_list.value)

        if self.validate_alignment and len(item_seq) != len(event_seq):
            raise ValueError(
                f"Length mismatch: sequence_data has length {len(item_seq)}, "
                f"but sequence_event_type has length {len(event_seq)}."
            )

        if self.mapper is not None:
            item_seq, event_seq = self.mapper.transform_sequence(item_seq, event_seq)

        return SequenceSample(item_seq=item_seq, event_seq=event_seq)

    def _load_all_records(self) -> None:
        loaded = 0

        for file_path in self.tfrecord_files:
            dataset = tf.data.TFRecordDataset(
                file_path,
                compression_type=self.compression_type,
            )

            for raw_record in dataset:
                sample = self._parse_example(raw_record.numpy())
                self.samples.append(sample)

                loaded += 1
                if self.max_records is not None and loaded >= self.max_records:
                    return

        if not self.samples:
            raise ValueError("No valid records were loaded from the TFRecord files.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        sample = self.samples[idx]
        return {
            "item_seq": sample.item_seq,
            "event_seq": sample.event_seq,
        }


def pad_1d_sequence(
    seq: Sequence[int],
    target_length: int,
    padding_value: int,
) -> List[int]:
    if len(seq) > target_length:
        return list(seq[:target_length])
    return list(seq) + [padding_value] * (target_length - len(seq))


def autocdsr_collate_fn(
    batch: List[Dict[str, List[int]]],
    padding_token: int = 0,
    max_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Output
    ------
    item_seq: Tensor[B, L]
    event_seq: Tensor[B, L]
    padding_mask: Tensor[B, L]   (True means valid token)
    lengths: Tensor[B]
    """
    if len(batch) == 0:
        raise ValueError("Received an empty batch in collate_fn.")

    item_seqs = [sample["item_seq"] for sample in batch]
    event_seqs = [sample["event_seq"] for sample in batch]

    raw_lengths = [len(seq) for seq in item_seqs]
    target_length = max(raw_lengths) if max_length is None else max_length

    padded_items = [
        pad_1d_sequence(seq, target_length, padding_token) for seq in item_seqs
    ]
    padded_events = [
        pad_1d_sequence(seq, target_length, padding_token) for seq in event_seqs
    ]

    item_tensor = torch.tensor(padded_items, dtype=torch.long)
    event_tensor = torch.tensor(padded_events, dtype=torch.long)

    event_tensor[item_tensor == padding_token] = padding_token

    padding_mask = item_tensor != padding_token
    lengths_tensor = padding_mask.sum(dim=1)

    return {
        "item_seq": item_tensor,         # [B, L]
        "event_seq": event_tensor,       # [B, L]
        "padding_mask": padding_mask,    # [B, L]
        "lengths": lengths_tensor,       # [B]
    }


def list_tfrecord_files(folder: str | Path) -> List[str]:
    """
    Find all .tfrecord.gz files under a folder and sort them by filename.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    files = sorted(folder.glob("*.tfrecord.gz"))
    if len(files) == 0:
        raise ValueError(f"No .tfrecord.gz files found in folder: {folder}")

    return [str(f) for f in files]


def build_dataset_from_folder(
    folder: str | Path,
    max_records: Optional[int] = None,
    mapper: Optional["IDMapper"] = None,
) -> AutoCDSRTFRecordDataset:
    tfrecord_files = list_tfrecord_files(folder)
    return AutoCDSRTFRecordDataset(
        tfrecord_files=tfrecord_files,
        compression_type="GZIP",
        max_records=max_records,
        validate_alignment=True,
        mapper=mapper,
    )


def build_dataloader_from_folder(
    folder: str | Path,
    batch_size: int = 4,
    shuffle: bool = False,
    padding_token: int = 0,
    max_length: Optional[int] = None,
    max_records: Optional[int] = None,
    num_workers: int = 0,
    mapper: Optional["IDMapper"] = None,
) -> tuple[AutoCDSRTFRecordDataset, DataLoader]:
    dataset = build_dataset_from_folder(
        folder=folder,
        max_records=max_records,
        mapper=mapper,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: autocdsr_collate_fn(
            batch=batch,
            padding_token=padding_token,
            max_length=max_length,
        ),
    )
    return dataset, loader


def build_all_dataloaders(
    data_root: str | Path,
    batch_size: int = 4,
    padding_token: int = 0,
    max_length: Optional[int] = None,
    max_records_per_split: Optional[int] = None,
    num_workers: int = 0,
    mapper: Optional["IDMapper"] = None,
) -> Dict[str, object]:
    """
    Build train / eval / test datasets and dataloaders from:

    data_root/
        training/
        evaluation/
        testing/
    """
    data_root = Path(data_root)

    train_dir = data_root / "training"
    eval_dir = data_root / "evaluation"
    test_dir = data_root / "testing"

    train_dataset, train_loader = build_dataloader_from_folder(
        folder=train_dir,
        batch_size=batch_size,
        shuffle=True,
        padding_token=padding_token,
        max_length=max_length,
        max_records=max_records_per_split,
        num_workers=num_workers,
        mapper=mapper,
    )

    eval_dataset, eval_loader = build_dataloader_from_folder(
        folder=eval_dir,
        batch_size=batch_size,
        shuffle=False,
        padding_token=padding_token,
        max_length=max_length,
        max_records=max_records_per_split,
        num_workers=num_workers,
        mapper=mapper,
    )

    test_dataset, test_loader = build_dataloader_from_folder(
        folder=test_dir,
        batch_size=batch_size,
        shuffle=False,
        padding_token=padding_token,
        max_length=max_length,
        max_records=max_records_per_split,
        num_workers=num_workers,
        mapper=mapper,
    )

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "eval_loader": eval_loader,
        "test_loader": test_loader,
    }


if __name__ == "__main__":
    from remap import IDMapper

    # 这里加载你前面已经生成好的 mapper
    mapper = IDMapper.load("artifacts/remap")

    outputs = build_all_dataloaders(
        data_root="data",
        batch_size=2,
        padding_token=mapper.config.item_pad,   # remap item_pad = 0
        max_length=None,                        # 800
        max_records_per_split=None,
        num_workers=0,
        mapper=mapper,
    )

    train_dataset = outputs["train_dataset"]
    eval_dataset = outputs["eval_dataset"]
    test_dataset = outputs["test_dataset"]

    train_loader = outputs["train_loader"]
    eval_loader = outputs["eval_loader"]
    test_loader = outputs["test_loader"]

    print("Dataset sizes:")
    print("train:", len(train_dataset))
    print("eval :", len(eval_dataset))
    print("test :", len(test_dataset))

    print("\nMapper info:")
    print("item_pad :", mapper.config.item_pad)
    print("item_mask:", mapper.config.item_mask)
    print("item_unk :", mapper.config.item_unk)
    print("event_pad:", mapper.config.event_pad)
    print("event_unk:", mapper.config.event_unk)
    print("num_items :", mapper.num_items)
    print("num_events:", mapper.num_events)

    first_train_sample = train_dataset[0]
    print("\nFirst train sample:")
    print("item_seq[:20] :", first_train_sample["item_seq"][:20])
    print("event_seq[:20]:", first_train_sample["event_seq"][:20])

    first_train_batch = next(iter(train_loader))
    print("\nTrain batch shapes:")
    for k, v in first_train_batch.items():
        print(k, tuple(v.shape))

    print("\nTrain batch item_seq:")
    print(first_train_batch["item_seq"])

    print("\nTrain batch event_seq:")
    print(first_train_batch["event_seq"])

    print("\nTrain batch padding_mask:")
    print(first_train_batch["padding_mask"])

    print("\nTrain batch lengths:")
    print(first_train_batch["lengths"])

    first_eval_sample = eval_dataset[0]
    print("\nFirst eval sample:")
    print("item_seq[:20] :", first_eval_sample["item_seq"][:20])
    print("event_seq[:20]:", first_eval_sample["event_seq"][:20])

    first_eval_batch = next(iter(eval_loader))
    print("\nEval batch shapes:")
    for k, v in first_eval_batch.items():
        print(k, tuple(v.shape))

    print("\nEval batch item_seq:")
    print(first_eval_batch["item_seq"])

    print("\nEval batch event_seq:")
    print(first_eval_batch["event_seq"])

    print("\nEval batch padding_mask:")
    print(first_eval_batch["padding_mask"])

    print("\nEval batch lengths:")
    print(first_eval_batch["lengths"])