"""Shared record contract for canonical, per-wrist sEMG HDF5 files."""

from dataclasses import dataclass
from typing import Any

import numpy as np


SCHEMA_VERSION = "1.0"
SAMPLE_RATE_HZ = 2000


@dataclass
class CanonicalRecord:
    dataset_id: str
    task_id: str
    subject_id: str
    session_id: str
    side: str
    pair_id: str
    official_split: str
    source_file: str
    emg: np.ndarray
    time_s: np.ndarray
    labels: dict[str, Any]
    source_meta: dict[str, Any]
    preprocessing: dict[str, Any]

    @classmethod
    def minimal(cls, emg: np.ndarray, time_s: np.ndarray) -> "CanonicalRecord":
        return cls(
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "",
            "unknown",
            "",
            emg,
            time_s,
            {},
            {},
            {"converter_transformations": []},
        )


def validate_record(record: CanonicalRecord) -> None:
    """Raise ValueError unless a record satisfies the signal-level contract."""
    if record.emg.ndim != 2 or record.emg.shape[1] != 16:
        raise ValueError("EMG must have shape [T, 16] with exactly 16 channels")
    if record.time_s.ndim != 1 or len(record.time_s) != len(record.emg):
        raise ValueError("time_s must be a length-T vector matching EMG")
    if not np.isfinite(record.emg).all() or not np.isfinite(record.time_s).all():
        raise ValueError("EMG and time_s must be finite")
    if len(record.time_s) > 1 and np.any(np.diff(record.time_s) < 0):
        raise ValueError("time_s must be monotonic")
