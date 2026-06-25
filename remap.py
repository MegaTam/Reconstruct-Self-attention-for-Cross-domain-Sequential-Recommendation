from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json

try:
    import tensorflow as tf
except ImportError as e:
    raise ImportError(
        "This script needs TensorFlow to read TFRecord files. Please install tensorflow."
    ) from e


@dataclass
class VocabConfig:
    # item vocab special ids
    item_pad: int = 0
    item_mask: int = 1
    item_unk: int = 2

    # event vocab special ids
    event_pad: int = 0
    event_unk: int = 1

    # raw padding token in original TFRecord
    raw_padding_token: int = -1


class IDMapper:
    def __init__(
        self,
        item2idx: Dict[int, int],
        event2idx: Dict[int, int],
        config: Optional[VocabConfig] = None,
    ) -> None:
        self.config = config or VocabConfig()
        self.item2idx = item2idx
        self.event2idx = event2idx

        self.idx2item = {v: k for k, v in item2idx.items()}
        self.idx2event = {v: k for k, v in event2idx.items()}

    @property
    def num_items(self) -> int:
        return max(self.item2idx.values()) + 1 if self.item2idx else 3

    @property
    def num_events(self) -> int:
        return max(self.event2idx.values()) + 1 if self.event2idx else 2

    def map_item(self, x: int) -> int:
        if x == self.config.raw_padding_token:
            return self.config.item_pad
        return self.item2idx.get(x, self.config.item_unk)

    def map_event(self, x: int, item_is_padding: bool = False) -> int:
        if item_is_padding or x == self.config.raw_padding_token:
            return self.config.event_pad
        return self.event2idx.get(x, self.config.event_unk)

    def transform_sequence(
        self,
        item_seq: Sequence[int],
        event_seq: Sequence[int],
    ) -> Tuple[List[int], List[int]]:
        if len(item_seq) != len(event_seq):
            raise ValueError(
                f"Length mismatch: item_seq={len(item_seq)}, event_seq={len(event_seq)}"
            )

        mapped_items: List[int] = []
        mapped_events: List[int] = []

        for item_id, event_id in zip(item_seq, event_seq):
            item_is_padding = item_id == self.config.raw_padding_token
            new_item = self.map_item(item_id)
            new_event = self.map_event(event_id, item_is_padding=item_is_padding)

            mapped_items.append(new_item)
            mapped_events.append(new_event)

        return mapped_items, mapped_events

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": {
                "item_pad": self.config.item_pad,
                "item_mask": self.config.item_mask,
                "item_unk": self.config.item_unk,
                "event_pad": self.config.event_pad,
                "event_unk": self.config.event_unk,
                "raw_padding_token": self.config.raw_padding_token,
            },
            "item2idx": {str(k): v for k, v in self.item2idx.items()},
            "event2idx": {str(k): v for k, v in self.event2idx.items()},
        }

        with open(save_dir / "mapper.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, save_dir: str | Path) -> "IDMapper":
        save_dir = Path(save_dir)
        with open(save_dir / "mapper.json", "r", encoding="utf-8") as f:
            payload = json.load(f)

        cfg = payload["config"]
        config = VocabConfig(
            item_pad=cfg["item_pad"],
            item_mask=cfg["item_mask"],
            item_unk=cfg["item_unk"],
            event_pad=cfg["event_pad"],
            event_unk=cfg["event_unk"],
            raw_padding_token=cfg["raw_padding_token"],
        )

        item2idx = {int(k): int(v) for k, v in payload["item2idx"].items()}
        event2idx = {int(k): int(v) for k, v in payload["event2idx"].items()}

        return cls(item2idx=item2idx, event2idx=event2idx, config=config)


def list_tfrecord_files(folder: str | Path) -> List[str]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    files = sorted(folder.glob("*.tfrecord.gz"))
    if len(files) == 0:
        raise ValueError(f"No .tfrecord.gz files found in folder: {folder}")

    return [str(f) for f in files]


def iter_raw_samples(
    tfrecord_files: Sequence[str | Path],
    compression_type: str = "GZIP",
) -> Iterable[Tuple[List[int], List[int]]]:
    for file_path in tfrecord_files:
        dataset = tf.data.TFRecordDataset(
            str(file_path),
            compression_type=compression_type,
        )
        for raw_record in dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            features = example.features.feature

            item_seq = list(features["sequence_data"].int64_list.value)
            event_seq = list(features["sequence_event_type"].int64_list.value)

            yield item_seq, event_seq


def build_mapper_from_folders(
    folders: Sequence[str | Path],
    config: Optional[VocabConfig] = None,
) -> IDMapper:
    """
    Build mapper from multiple folders together, e.g.
    ['data/training', 'data/evaluation', 'data/testing'].
    """
    config = config or VocabConfig()

    all_files: List[str] = []
    for folder in folders:
        all_files.extend(list_tfrecord_files(folder))

    if len(all_files) == 0:
        raise ValueError("No TFRecord files found across the given folders.")

    item2idx: Dict[int, int] = {}
    event2idx: Dict[int, int] = {}

    next_item_idx = 3  # 0=PAD, 1=MASK, 2=UNK
    next_event_idx = 2  # 0=PAD, 1=UNK

    for item_seq, event_seq in iter_raw_samples(all_files):
        if len(item_seq) != len(event_seq):
            raise ValueError(
                f"Length mismatch found while scanning data: "
                f"{len(item_seq)} vs {len(event_seq)}"
            )

        for item_id, event_id in zip(item_seq, event_seq):
            item_is_padding = item_id == config.raw_padding_token

            if not item_is_padding and item_id not in item2idx:
                item2idx[item_id] = next_item_idx
                next_item_idx += 1

            # 只有 item 有效时才记录 event
            if (not item_is_padding) and (event_id != config.raw_padding_token):
                if event_id not in event2idx:
                    event2idx[event_id] = next_event_idx
                    next_event_idx += 1

    return IDMapper(item2idx=item2idx, event2idx=event2idx, config=config)


def build_mapper_from_data_root(
    data_root: str | Path,
    config: Optional[VocabConfig] = None,
) -> IDMapper:
    """
    Convenience wrapper for:
        data_root/training
        data_root/evaluation
        data_root/testing
    """
    data_root = Path(data_root)
    folders = [
        data_root / "training",
        data_root / "evaluation",
        data_root / "testing",
    ]
    return build_mapper_from_folders(folders=folders, config=config)


if __name__ == "__main__":
    mapper = build_mapper_from_data_root("data")

    print("num_items :", mapper.num_items)
    print("num_events:", mapper.num_events)

    save_dir = "artifacts/remap"
    mapper.save(save_dir)

    loaded_mapper = IDMapper.load(save_dir)
    print("loaded num_items :", loaded_mapper.num_items)
    print("loaded num_events:", loaded_mapper.num_events)

    train_files = list_tfrecord_files("data/training")
    first_item_seq, first_event_seq = next(iter_raw_samples(train_files))
    mapped_item_seq, mapped_event_seq = loaded_mapper.transform_sequence(
        first_item_seq, first_event_seq
    )

    print("\noriginal item[:20]:", first_item_seq[:20])
    print("mapped   item[:20]:", mapped_item_seq[:20])

    print("\noriginal event[:20]:", first_event_seq[:20])
    print("mapped   event[:20]:", mapped_event_seq[:20])

    eval_files = list_tfrecord_files("data/evaluation")
    eval_item_seq, eval_event_seq = next(iter_raw_samples(eval_files))
    mapped_eval_item_seq, mapped_eval_event_seq = loaded_mapper.transform_sequence(
        eval_item_seq, eval_event_seq
    )

    print("\noriginal eval item[:20]:", eval_item_seq[:20])
    print("mapped   eval item[:20]:", mapped_eval_item_seq[:20])

    print("\noriginal eval event[:20]:", eval_event_seq[:20])
    print("mapped   eval event[:20]:", mapped_eval_event_seq[:20])

    test_files = list_tfrecord_files("data/testing")
    test_item_seq, test_event_seq = next(iter_raw_samples(test_files))
    mapped_test_item_seq, mapped_test_event_seq = loaded_mapper.transform_sequence(
        test_item_seq, test_event_seq
    )

    print("\noriginal test item[:20]:", test_item_seq[:20])
    print("mapped   test item[:20]:", mapped_test_item_seq[:20])

    print("\noriginal test event[:20]:", test_event_seq[:20])
    print("mapped   test event[:20]:", mapped_test_event_seq[:20])