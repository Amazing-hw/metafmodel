"""Lazy windowed access to canonical sEMG HDF5 records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import load_manifest
from .task_labels import parse_target


OFFICIAL_SPLITS = frozenset({"train", "val", "test"})
REQUIRED_ROOT_ATTRIBUTES = (
    "dataset_id", "task_id", "subject_id", "session_id", "side", "pair_id",
    "official_split", "sample_rate_hz",
)
MANIFEST_ATTRIBUTE_FIELDS = REQUIRED_ROOT_ATTRIBUTES[:-1]


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return str(value.item())
    return str(value)


def _canonical_signal_datasets(h5: h5py.File, record: dict[str, Any], path: Path):
    missing_attrs = [name for name in REQUIRED_ROOT_ATTRIBUTES if name not in h5.attrs]
    if missing_attrs:
        raise ValueError(f"{path} lacks required root attributes: {', '.join(missing_attrs)}")
    for name in MANIFEST_ATTRIBUTE_FIELDS:
        if _text(h5.attrs[name]) != record[name]:
            raise ValueError(f"{path} root attribute {name} does not match manifest")
    try:
        sample_rate_hz = float(h5.attrs["sample_rate_hz"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} sample_rate_hz must be numeric") from exc
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError(f"{path} has invalid sample_rate_hz")
    if not np.isclose(sample_rate_hz, record["sample_rate_hz"]):
        raise ValueError(f"{path} root attribute sample_rate_hz does not match manifest")
    emg_dataset = h5.get("signal/emg")
    time_dataset = h5.get("signal/time_s")
    if not isinstance(emg_dataset, h5py.Dataset) or not isinstance(time_dataset, h5py.Dataset):
        raise ValueError(f"{path} requires signal/emg and signal/time_s datasets")
    if emg_dataset.dtype != np.dtype(np.float32):
        raise ValueError(f"{path} signal/emg must have dtype float32")
    if time_dataset.dtype != np.dtype(np.float64):
        raise ValueError(f"{path} signal/time_s must have dtype float64")
    if emg_dataset.ndim != 2 or emg_dataset.shape[1] != 16:
        raise ValueError(f"{path} signal/emg must have shape [T, 16]")
    if time_dataset.ndim != 1 or time_dataset.shape[0] != emg_dataset.shape[0]:
        raise ValueError(f"{path} signal/time_s must have shape [T] aligned with signal/emg")
    return emg_dataset, time_dataset, sample_rate_hz


class CanonicalWindowDataset(Dataset):
    """Index full, fixed-duration windows without keeping HDF5 files open."""

    def __init__(self, records: list[dict[str, Any]], window_ms: float, stride_ms: float):
        if window_ms <= 0 or stride_ms <= 0:
            raise ValueError("window_ms and stride_ms must be positive")
        self.window_ms = float(window_ms)
        self.stride_ms = float(stride_ms)
        self._records = {Path(record["path"]): record for record in records}
        self._index: list[tuple[Path, int, int]] = []
        for record in records:
            rate_hz = float(record["sample_rate_hz"])
            window_samples = int(round(rate_hz * self.window_ms / 1000.0))
            stride_samples = int(round(rate_hz * self.stride_ms / 1000.0))
            if window_samples <= 0 or stride_samples <= 0:
                raise ValueError("window_ms and stride_ms must produce at least one sample")
            sample_count = int(record["sample_count"])
            for start_index in range(0, sample_count - window_samples + 1, stride_samples):
                self._index.append((Path(record["path"]), start_index, start_index + window_samples))

    @classmethod
    def from_manifest(cls, manifest_path, split: str, window_ms: float, stride_ms: float, root=None):
        if split not in OFFICIAL_SPLITS:
            raise ValueError(f"split must be one of {sorted(OFFICIAL_SPLITS)}")
        manifest_path = Path(manifest_path)
        root = Path(root) if root is not None else manifest_path.parent
        frame = load_manifest(manifest_path, root)
        selected = frame.loc[frame["official_split"] == split]
        records = []
        for _, row in selected.iterrows():
            sample_count, duration_s = int(row["sample_count"]), float(row["duration_s"])
            if "sample_rate_hz" in row and row["sample_rate_hz"] != "":
                rate_hz = float(row["sample_rate_hz"])
            elif sample_count > 1 and duration_s > 0:
                rate_hz = (sample_count - 1) / duration_s
            else:
                raise ValueError(f"cannot infer sample rate for {row['path']} without sample_rate_hz")
            if not np.isfinite(rate_hz) or rate_hz <= 0:
                raise ValueError(f"invalid sample rate for {row['path']}")
            records.append({
                "path": Path(row["resolved_path"]), "dataset_id": _text(row["dataset_id"]),
                "task_id": _text(row["task_id"]), "record_id": _text(row.get("record_id", Path(row["path"]).stem)),
                "subject_id": _text(row["subject_id"]), "session_id": _text(row["session_id"]),
                "side": _text(row["side"]), "pair_id": _text(row["pair_id"]),
                "official_split": _text(row["official_split"]),
                "sample_count": sample_count, "sample_rate_hz": rate_hz,
            })
        return cls(records, window_ms, stride_ms)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, start_index, end_index = self._index[index]
        record = self._records[path]
        with h5py.File(path, "r") as h5:
            emg_dataset, time_dataset, sample_rate_hz = _canonical_signal_datasets(h5, record, path)
            emg = np.asarray(emg_dataset[start_index:end_index])
            # The interval is [first selected timestamp, midpoint after its final sample).
            # A following stored timestamp defines that midpoint; at record end, the prior
            # selected interval provides the local extrapolation for the open boundary.
            time_with_boundary = np.asarray(time_dataset[start_index:min(end_index + 1, time_dataset.shape[0])])
            time_s = time_with_boundary[: end_index - start_index]
            if emg.shape[0] != end_index - start_index:
                raise ValueError(f"{path} selected EMG slice has an unexpected length")
            if not np.isfinite(emg).all():
                raise ValueError(f"{path} selected EMG slice must be finite numeric values")
            if time_s.ndim != 1 or time_s.shape[0] != emg.shape[0]:
                raise ValueError(f"{path} selected time_s slice must align with EMG")
            if not np.isfinite(time_with_boundary).all():
                raise ValueError(f"{path} selected time_s slice must be finite numeric values")
            if time_with_boundary.shape[0] > 1 and np.any(np.diff(time_with_boundary) < 0):
                raise ValueError(f"{path} selected time_s slice must be monotonic")
            start_s = float(time_s[0])
            if time_with_boundary.shape[0] > time_s.shape[0]:
                end_s = float((time_s[-1] + time_with_boundary[-1]) / 2)
            elif time_s.shape[0] > 1:
                end_s = float(time_s[-1] + (time_s[-1] - time_s[-2]) / 2)
            else:
                end_s = float(time_s[-1] + 0.5 / sample_rate_hz)
            item = {
                "emg": torch.as_tensor(emg, dtype=torch.float32),
                "sample_mask": torch.ones(emg.shape[0], dtype=torch.bool),
                "dataset_id": record["dataset_id"], "task_id": record["task_id"],
                "record_id": _text(h5.attrs.get("record_id", record["record_id"])),
                "subject_id": record["subject_id"], "session_id": record["session_id"],
                "side": record["side"], "pair_id": record["pair_id"], "start_s": start_s,
                "sample_rate_hz": sample_rate_hz,
            }
            item.update(parse_target(record["task_id"], h5, start_s, end_s, start_index, end_index))
        return item
