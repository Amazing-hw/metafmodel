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


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return str(value.item())
    return str(value)


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
                "sample_count": sample_count, "sample_rate_hz": rate_hz,
            })
        return cls(records, window_ms, stride_ms)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, start_index, end_index = self._index[index]
        record = self._records[path]
        with h5py.File(path, "r") as h5:
            emg = np.asarray(h5["signal/emg"][start_index:end_index])
            if "signal/time_s" not in h5:
                raise ValueError(f"{path} lacks signal/time_s needed for label alignment")
            time_s = np.asarray(h5["signal/time_s"][start_index:end_index])
            if emg.ndim != 2 or emg.shape[1] != 16:
                raise ValueError(f"{path} signal/emg must have shape [T, 16]")
            if emg.shape[0] != end_index - start_index:
                raise ValueError(f"{path} selected EMG slice has an unexpected length")
            if not np.issubdtype(emg.dtype, np.number) or not np.isfinite(emg).all():
                raise ValueError(f"{path} selected EMG slice must be finite numeric values")
            if time_s.ndim != 1 or time_s.shape[0] != emg.shape[0]:
                raise ValueError(f"{path} selected time_s slice must align with EMG")
            if not np.issubdtype(time_s.dtype, np.number) or not np.isfinite(time_s).all():
                raise ValueError(f"{path} selected time_s slice must be finite numeric values")
            if time_s.shape[0] > 1 and np.any(np.diff(time_s) < 0):
                raise ValueError(f"{path} selected time_s slice must be monotonic")
            sample_rate_hz = float(h5.attrs.get("sample_rate_hz", record["sample_rate_hz"]))
            if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
                raise ValueError(f"{path} has invalid sample_rate_hz")
            start_s = float(time_s[0])
            end_s = start_s + (end_index - start_index) / sample_rate_hz
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
