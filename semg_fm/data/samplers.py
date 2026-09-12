"""Deterministic task-balanced samplers for heterogeneous sEMG training data."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator

import torch
from torch.utils.data import Sampler


def _task_ids(dataset) -> list[str]:
    """Read task IDs without opening HDF5 windows when using the canonical dataset."""
    if hasattr(dataset, "task_ids"):
        task_ids = list(dataset.task_ids)
        if len(task_ids) == len(dataset):
            return [str(task_id) for task_id in task_ids]
    if hasattr(dataset, "_index") and hasattr(dataset, "_records"):
        return [str(dataset._records[path]["task_id"]) for path, _, _ in dataset._index]
    return [str(dataset[index]["task_id"]) for index in range(len(dataset))]


def _task_groups(dataset) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, task_id in enumerate(_task_ids(dataset)):
        groups[task_id].append(index)
    if not groups:
        raise ValueError("dataset must contain at least one task")
    return {task_id: indices for task_id, indices in sorted(groups.items())}


class EvaluationIndex(int):
    """An integer dataset index that identifies a deterministic padded duplicate."""

    def __new__(cls, value: int, is_padding: bool = False):
        instance = int.__new__(cls, value)
        instance.is_padding = bool(is_padding)
        return instance


class TaskBalancedBatchSampler(Sampler[list[int]]):
    """Yield homogeneous batches, interleaving each task equally within an epoch."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        epoch_getter: Callable[[], int] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._epoch_getter = epoch_getter
        self.epoch = 0
        self._groups = _task_groups(dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def state_dict(self) -> dict[str, int]:
        """Persist the shuffle epoch so DataLoader resume replays deterministically."""
        return {"epoch": self.epoch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = int(state.get("epoch", 0))

    def _rank_batches(self) -> dict[str, list[list[int]]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batches: dict[str, list[list[int]]] = {}
        for task_id, indices in self._groups.items():
            order = torch.randperm(len(indices), generator=generator).tolist()
            usable_samples = len(indices) // (self.batch_size * self.world_size)
            samples_per_rank = usable_samples * self.batch_size
            ordered_indices = [indices[position] for position in order]
            rank_start = self.rank * samples_per_rank
            rank_indices = ordered_indices[rank_start : rank_start + samples_per_rank]
            batches[task_id] = [
                rank_indices[start : start + self.batch_size]
                for start in range(0, len(rank_indices) - self.batch_size + 1, self.batch_size)
            ]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        if self._epoch_getter is not None:
            self.set_epoch(self._epoch_getter())
        batches = self._rank_batches()
        # Interleave one batch per task per round until every task is exhausted.
        # A min() cutoff would starve the whole epoch whenever one task has
        # fewer windows than a batch; skip exhausted tasks instead.
        longest = max((len(task_batches) for task_batches in batches.values()), default=0)
        for batch_index in range(longest):
            for task_id in self._groups:
                task_batches = batches[task_id]
                if batch_index < len(task_batches):
                    yield task_batches[batch_index]

    def __len__(self) -> int:
        batches = self._rank_batches()
        return sum(len(task_batches) for task_batches in batches.values())


class TaskEvaluationBatchSampler(Sampler[list[int]]):
    """Yield every evaluation index exactly once, while keeping each batch task-homogeneous."""

    def __init__(self, dataset, batch_size: int, *, rank: int = 0, world_size: int = 1) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._groups = _task_groups(dataset)

    def _batches(self) -> list[list[int]]:
        batches: list[list[int]] = []
        for indices in self._groups.values():
            actual_batches = [
                indices[start : start + self.batch_size]
                for start in range(0, len(indices), self.batch_size)
            ]
            padded_count = (-len(actual_batches)) % self.world_size
            assigned_batches = actual_batches + [
                actual_batches[offset % len(actual_batches)] for offset in range(padded_count)
            ]
            for batch_index in range(self.rank, len(assigned_batches), self.world_size):
                is_padding = batch_index >= len(actual_batches)
                batches.append([
                    EvaluationIndex(index, is_padding) for index in assigned_batches[batch_index]
                ])
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())
