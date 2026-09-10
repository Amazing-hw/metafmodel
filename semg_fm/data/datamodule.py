"""Lightning-compatible canonical sEMG data loading."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from lightning.pytorch import LightningDataModule
except ImportError:  # pragma: no cover - supports older Lightning installs.
    try:
        from pytorch_lightning import LightningDataModule
    except ImportError:  # pragma: no cover - permits dataset-only environments.
        class LightningDataModule:  # type: ignore[no-redef]
            pass

from .samplers import TaskBalancedBatchSampler, TaskEvaluationBatchSampler
from .window_dataset import CanonicalWindowDataset


class _EvaluationDataset(Dataset):
    """Attach the sampler's padding marker without changing canonical records."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> dict[str, Any]:
        item = dict(self.dataset[int(index)])
        item["is_padding"] = bool(getattr(index, "is_padding", False))
        return item


def task_homogeneous_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length windows after enforcing one task per batch."""
    if not items:
        raise ValueError("cannot collate an empty batch")
    task_ids = {str(item["task_id"]) for item in items}
    if len(task_ids) != 1:
        raise ValueError("a batch must contain one task_id")
    max_time = max(item["emg"].shape[0] for item in items)
    channels = items[0]["emg"].shape[-1]
    if channels != 16 or any(item["emg"].ndim != 2 or item["emg"].shape[-1] != channels for item in items):
        raise ValueError("each emg item must have shape [T, 16]")
    dtype, device = items[0]["emg"].dtype, items[0]["emg"].device
    emg = torch.zeros((len(items), max_time, channels), dtype=dtype, device=device)
    sample_mask = torch.zeros((len(items), max_time), dtype=torch.bool, device=device)
    is_padding = torch.tensor(
        [bool(item.get("is_padding", False)) for item in items], dtype=torch.bool, device=device
    )
    metadata = []
    for index, item in enumerate(items):
        window = item["emg"]
        if window.dtype != dtype or window.device != device:
            raise ValueError("all emg items must share dtype and device")
        length = window.shape[0]
        emg[index, :length] = window
        sample_mask[index, :length] = item["sample_mask"].to(dtype=torch.bool, device=device)
        metadata.append({
            key: value for key, value in item.items()
            if key not in {"emg", "sample_mask", "is_padding"}
        })
    batch = {
        "emg": emg, "sample_mask": sample_mask, "task_id": task_ids.pop(), "metadata": metadata,
        "is_padding": is_padding, "example_weight": (~is_padding).to(dtype=torch.float32),
    }
    if all(isinstance(item.get("target"), torch.Tensor) and isinstance(item.get("target_mask"), torch.Tensor) for item in items):
        targets = [item["target"] for item in items]
        target_masks = [item["target_mask"] for item in items]
        target = targets[0]
        target_mask = target_masks[0]
        if target.ndim < 1 or target_mask.ndim < 1:
            raise ValueError("kinematic target and target_mask must have a time dimension")
        if target_mask.dtype != torch.bool:
            raise ValueError("kinematic target_mask must be boolean")
        if target_mask.shape[1:] not in ((), target.shape[1:]):
            raise ValueError("kinematic target_mask must be [T] or match the target shape")
        if any(
            value.ndim < 1
            or value.shape[0] != item["emg"].shape[0]
            or value.shape[1:] != target.shape[1:]
            or value.dtype != target.dtype
            or value.device != target.device
            for item, value in zip(items, targets)
        ):
            raise ValueError("kinematic targets must share time-aligned trailing shape, dtype, and device")
        if any(
            value.ndim < 1
            or value.shape[0] != item["emg"].shape[0]
            or value.shape[1:] != target_mask.shape[1:]
            or value.dtype != torch.bool
            or value.device != target_mask.device
            for item, value in zip(items, target_masks)
        ):
            raise ValueError("kinematic target masks must share time-aligned trailing shape and device")
        padded_target = torch.zeros((len(items), max_time, *target.shape[1:]), dtype=target.dtype, device=target.device)
        padded_target_mask = torch.zeros((len(items), max_time, *target_mask.shape[1:]), dtype=torch.bool, device=target_mask.device)
        for index, (value, mask) in enumerate(zip(targets, target_masks)):
            length = value.shape[0]
            padded_target[index, :length] = value
            padded_target_mask[index, :length] = mask
        batch["target"] = padded_target
        batch["target_mask"] = padded_target_mask
    return batch


class CanonicalDataModule(LightningDataModule):
    """Build canonical split datasets and deterministic task-balanced loaders."""

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        *,
        split_paths: Mapping[str, str | Path] | None = None,
        root: str | Path | None = None,
        window_ms: float,
        stride_ms: float,
        batch_size: int,
        num_workers: int = 0,
        sampler_seed: int = 0,
    ) -> None:
        super().__init__()
        if manifest_path is None and not split_paths:
            raise ValueError("manifest_path or split_paths is required")
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.split_paths = {name: Path(path) for name, path in (split_paths or {}).items()}
        self.root = Path(root) if root is not None else None
        self.window_ms = float(window_ms)
        self.stride_ms = float(stride_ms)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.sampler_seed = int(sampler_seed)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_sampler = None
        self.val_sampler = None
        self.test_sampler = None

    def _manifest_for(self, split: str) -> Path:
        path = self.split_paths.get(split, self.manifest_path)
        if path is None:
            raise ValueError(f"no manifest path configured for {split}")
        return path

    def _dataset_for(self, split: str) -> CanonicalWindowDataset:
        manifest_path = self._manifest_for(split)
        return CanonicalWindowDataset.from_manifest(
            manifest_path, split=split, window_ms=self.window_ms, stride_ms=self.stride_ms,
            root=self.root if self.root is not None else manifest_path.parent,
        )

    def _sampler_for(self, dataset) -> TaskBalancedBatchSampler | None:
        if not len(dataset):
            return None
        rank, world_size = self._distributed_context()
        return TaskBalancedBatchSampler(
            dataset, self.batch_size, seed=self.sampler_seed, rank=rank, world_size=world_size,
            epoch_getter=self._trainer_epoch,
        )

    def _distributed_context(self) -> tuple[int, int]:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return 0, 1
        return int(trainer.global_rank), int(trainer.world_size)

    def _trainer_epoch(self) -> int:
        trainer = getattr(self, "trainer", None)
        return int(getattr(trainer, "current_epoch", 0))

    def _evaluation_sampler_for(self, dataset) -> TaskEvaluationBatchSampler | None:
        if not len(dataset):
            return None
        rank, world_size = self._distributed_context()
        return TaskEvaluationBatchSampler(
            dataset, self.batch_size, rank=rank, world_size=world_size,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            if self.train_dataset is None:
                self.train_dataset = self._dataset_for("train")
            if self.val_dataset is None:
                self.val_dataset = self._dataset_for("val")
        elif stage == "validate" and self.val_dataset is None:
            self.val_dataset = self._dataset_for("val")
        if stage in (None, "test") and self.test_dataset is None:
            self.test_dataset = self._dataset_for("test")

    def _loader(self, dataset, sampler) -> DataLoader:
        loader_dataset = _EvaluationDataset(dataset) if isinstance(sampler, TaskEvaluationBatchSampler) else dataset
        if sampler is None:
            return DataLoader(
                loader_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                collate_fn=task_homogeneous_collate,
            )
        return DataLoader(
            loader_dataset, batch_sampler=sampler, num_workers=self.num_workers,
            collate_fn=task_homogeneous_collate,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        self.train_sampler = self._sampler_for(self.train_dataset)
        return self._loader(self.train_dataset, self.train_sampler)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("validate")
        self.val_sampler = self._evaluation_sampler_for(self.val_dataset)
        return self._loader(self.val_dataset, self.val_sampler)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        self.test_sampler = self._evaluation_sampler_for(self.test_dataset)
        return self._loader(self.test_dataset, self.test_sampler)

    def set_epoch(self, epoch: int) -> None:
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
