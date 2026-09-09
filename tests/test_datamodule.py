import pytest
import pandas as pd
import torch
from torch.utils.data import Dataset

from semg_fm.data.datamodule import CanonicalDataModule, task_homogeneous_collate
from semg_fm.data.samplers import TaskBalancedBatchSampler, TaskEvaluationBatchSampler


def _item(task_id, length, value=1.0):
    return {
        "emg": torch.full((length, 16), value, dtype=torch.float32),
        "sample_mask": torch.ones(length, dtype=torch.bool),
        "task_id": task_id,
        "record_id": f"{task_id}-{length}",
    }


def test_collate_rejects_mixed_task_ids():
    with pytest.raises(ValueError, match="one task_id"):
        task_homogeneous_collate([_item("pose", 2), _item("typing", 3)])


def test_collate_pads_time_and_returns_sample_mask_and_metadata():
    batch = task_homogeneous_collate([_item("pose", 2, 1), _item("pose", 4, 2)])

    assert batch["emg"].shape == (2, 4, 16)
    assert batch["sample_mask"].dtype == torch.bool
    assert batch["sample_mask"].tolist() == [[True, True, False, False], [True, True, True, True]]
    assert batch["task_id"] == "pose"
    assert [metadata["record_id"] for metadata in batch["metadata"]] == ["pose-2", "pose-4"]


def test_collate_returns_zero_weight_for_padded_evaluation_examples():
    real_item = _item("pose", 2)
    padded_item = _item("pose", 2)
    real_item["is_padding"] = False
    padded_item["is_padding"] = True

    batch = task_homogeneous_collate([real_item, padded_item])

    assert batch["is_padding"].dtype == torch.bool
    assert batch["is_padding"].tolist() == [False, True]
    assert batch["example_weight"].tolist() == [1.0, 0.0]


class _TaskDataset(Dataset):
    def __init__(self):
        self.items = [_item(task, 2) for task in ("a", "a", "a", "a", "b", "b", "b", "b")]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def test_task_balanced_sampler_set_epoch_is_deterministic_and_changes_order():
    dataset = _TaskDataset()
    sampler = TaskBalancedBatchSampler(dataset, batch_size=2, seed=17)

    sampler.set_epoch(0)
    epoch_zero = list(sampler)
    sampler.set_epoch(1)
    epoch_one = list(sampler)
    sampler.set_epoch(0)
    repeated_epoch_zero = list(sampler)

    assert epoch_zero == repeated_epoch_zero
    assert epoch_zero != epoch_one
    assert [dataset[batch[0]]["task_id"] for batch in epoch_zero] == ["a", "b", "a", "b"]


def test_task_balanced_sampler_rank_partitions_do_not_overlap():
    dataset = _TaskDataset()
    first_rank = TaskBalancedBatchSampler(dataset, batch_size=1, seed=3, rank=0, world_size=2)
    second_rank = TaskBalancedBatchSampler(dataset, batch_size=1, seed=3, rank=1, world_size=2)
    first_rank.set_epoch(2)
    second_rank.set_epoch(2)

    first_indices = {index for batch in first_rank for index in batch}
    second_indices = {index for batch in second_rank for index in batch}

    assert not first_indices.intersection(second_indices)
    assert len(first_indices) == sum(len(batch) for batch in first_rank)
    assert len(second_indices) == sum(len(batch) for batch in second_rank)


def test_task_balanced_sampler_keeps_distributed_rank_batch_counts_equal():
    class UnevenTaskDataset(Dataset):
        def __init__(self):
            self.items = [_item(task, 2) for task in ("a",) * 7 + ("b",) * 7]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    dataset = UnevenTaskDataset()
    first_rank = TaskBalancedBatchSampler(dataset, batch_size=2, seed=3, rank=0, world_size=2)
    second_rank = TaskBalancedBatchSampler(dataset, batch_size=2, seed=3, rank=1, world_size=2)

    assert len(list(first_rank)) == len(list(second_rank)) == 2


def test_task_balanced_sampler_reads_epoch_when_its_loader_starts():
    epoch = [0]
    sampler = TaskBalancedBatchSampler(
        _TaskDataset(), batch_size=2, seed=3, epoch_getter=lambda: epoch[0]
    )

    list(sampler)
    epoch[0] = 5
    list(sampler)

    assert sampler.epoch == 5


def test_datamodule_yields_task_homogeneous_train_batch(canonical_meta_root):
    module = CanonicalDataModule(
        manifest_path=canonical_meta_root / "manifest.csv",
        root=canonical_meta_root,
        window_ms=100,
        stride_ms=100,
        batch_size=2,
        num_workers=0,
        sampler_seed=5,
    )
    module.setup("fit")

    batch = next(iter(module.train_dataloader()))

    assert batch["emg"].shape[0] == 2
    assert batch["emg"].shape[2] == 16
    assert batch["sample_mask"].dtype == torch.bool
    assert all(metadata["task_id"] == batch["task_id"] for metadata in batch["metadata"])


def test_datamodule_uses_attached_trainer_rank_when_building_loader(canonical_meta_root):
    module = CanonicalDataModule(
        manifest_path=canonical_meta_root / "manifest.csv",
        root=canonical_meta_root,
        window_ms=100,
        stride_ms=100,
        batch_size=1,
        num_workers=0,
        sampler_seed=5,
    )
    module.setup("fit")

    class FakeTrainer:
        global_rank = 1
        world_size = 2
        current_epoch = 0

    module.trainer = FakeTrainer()
    rank_one_sampler = module.train_dataloader().batch_sampler
    rank_zero_sampler = TaskBalancedBatchSampler(
        module.train_dataset, batch_size=1, seed=5, rank=0, world_size=2
    )

    assert rank_one_sampler.rank == 1
    assert rank_one_sampler.world_size == 2
    assert list(rank_one_sampler) != list(rank_zero_sampler)


def _manifest_for_split(canonical_meta_root, split):
    manifest = pd.read_csv(canonical_meta_root / "manifest.csv", keep_default_na=False)
    manifest["official_split"] = split
    path = canonical_meta_root / f"{split}_only.csv"
    manifest.to_csv(path, index=False)
    return path


def test_setup_initializes_only_the_requested_split_lifecycle(canonical_meta_root):
    val_module = CanonicalDataModule(
        split_paths={"val": _manifest_for_split(canonical_meta_root, "val")},
        root=canonical_meta_root, window_ms=100, stride_ms=100, batch_size=2, num_workers=0,
    )
    val_module.setup("validate")
    assert val_module.val_dataset is not None
    assert val_module.train_dataset is None
    assert val_module.test_dataset is None

    test_module = CanonicalDataModule(
        split_paths={"test": _manifest_for_split(canonical_meta_root, "test")},
        root=canonical_meta_root, window_ms=100, stride_ms=100, batch_size=2, num_workers=0,
    )
    test_module.setup("test")
    assert test_module.test_dataset is not None
    assert test_module.train_dataset is None
    assert test_module.val_dataset is None

    fit_module = CanonicalDataModule(
        split_paths={
            "train": _manifest_for_split(canonical_meta_root, "train"),
            "val": _manifest_for_split(canonical_meta_root, "val"),
        },
        root=canonical_meta_root, window_ms=100, stride_ms=100, batch_size=2, num_workers=0,
    )
    fit_module.setup("fit")
    assert fit_module.train_dataset is not None
    assert fit_module.val_dataset is not None
    assert fit_module.test_dataset is None


class _ImbalancedTaskDataset(Dataset):
    def __init__(self):
        self.items = [_item("major", 2) for _ in range(20)] + [_item("minor", 2) for _ in range(2)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def test_eval_loaders_cover_each_imbalanced_task_index_once_without_epoch_variation(canonical_meta_root):
    module = CanonicalDataModule(
        manifest_path=canonical_meta_root / "manifest.csv",
        root=canonical_meta_root, window_ms=100, stride_ms=100, batch_size=2, num_workers=0,
    )
    dataset = _ImbalancedTaskDataset()
    module.val_dataset = dataset
    module.test_dataset = dataset

    for loader in (module.val_dataloader(), module.test_dataloader()):
        sampler = loader.batch_sampler
        first_epoch = [index for batch in sampler for index in batch]
        module.set_epoch(9)
        second_epoch = [index for batch in sampler for index in batch]
        assert sorted(first_epoch) == list(range(22))
        assert len(first_epoch) == 22
        assert second_epoch == first_epoch


def test_distributed_evaluation_padding_keeps_rank_lengths_equal_and_marks_duplicates():
    class UnevenEvaluationDataset(Dataset):
        def __init__(self):
            self.items = [_item("major", 2) for _ in range(5)] + [_item("minor", 2) for _ in range(2)]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    dataset = UnevenEvaluationDataset()
    first_rank = TaskEvaluationBatchSampler(dataset, batch_size=2, rank=0, world_size=2)
    second_rank = TaskEvaluationBatchSampler(dataset, batch_size=2, rank=1, world_size=2)

    first_batches = list(first_rank)
    second_batches = list(second_rank)
    entries = [entry for batch in first_batches + second_batches for entry in batch]
    nonpadding = [int(entry) for entry in entries if not entry.is_padding]
    padding = [entry for entry in entries if entry.is_padding]

    assert len(first_rank) == len(second_rank)
    assert sorted(nonpadding) == list(range(7))
    assert len(nonpadding) == 7
    assert padding
    assert all(entry.is_padding for entry in padding)
