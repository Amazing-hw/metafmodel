"""Shared record contract for canonical, per-wrist sEMG HDF5 files."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import h5py
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


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _write_labels(labels_group: h5py.Group, labels: dict[str, Any]) -> None:
    """Write task-specific tables while retaining source-compatible label JSON."""
    labels_group.create_dataset(
        "source_json",
        data=json.dumps(labels, default=_json_default, sort_keys=True),
        dtype=h5py.string_dtype("utf-8"),
    )


def _structured_table(
    rows: list[dict[str, Any]], fields: list[tuple[str, Any]],
) -> np.ndarray:
    dtype = np.dtype(fields)
    result = np.empty(len(rows), dtype=dtype)
    for index, row in enumerate(rows):
        result[index] = tuple(row[field] for field, _ in fields)
    return result


def _write_utf8_scalar(group: h5py.Group, name: str, value: Any) -> None:
    group.create_dataset(name, data=str(value), dtype=h5py.string_dtype("utf-8"))


def _write_task_labels(
    labels_group: h5py.Group, task_id: str, labels: dict[str, Any]
) -> None:
    text = h5py.string_dtype("utf-8")
    if task_id == "typing":
        labels_group.create_dataset(
            "keystrokes",
            data=_structured_table(
                labels.get("keystrokes", []),
                [("key", text), ("press_s", np.float64), ("release_s", np.float64)],
            ),
        )
        labels_group.create_dataset(
            "prompts",
            data=_structured_table(
                labels.get("prompts", []),
                [("text", text), ("start_s", np.float64), ("end_s", np.float64)],
            ),
        )
    elif task_id == "discrete_gesture":
        labels_group.create_dataset(
            "events",
            data=_structured_table(
                labels.get("events", []), [("name", text), ("time_s", np.float64)]
            ),
        )
        if "stage" in labels:
            _write_utf8_scalar(labels_group, "stage", labels["stage"])
    elif task_id == "handwriting":
        labels_group.create_dataset(
            "prompts",
            data=_structured_table(
                labels.get("prompts", []),
                [("text", text), ("start_s", np.float64), ("end_s", np.float64)],
            ),
        )
        if "stage" in labels:
            _write_utf8_scalar(labels_group, "stage", labels["stage"])
    elif task_id == "pose":
        if "joint_angles" not in labels:
            raise ValueError("Pose records require labels['joint_angles']")
        labels_group.create_dataset("joint_angles", data=np.asarray(labels["joint_angles"]))
        if "no_ik_failure" in labels:
            labels_group.create_dataset(
                "no_ik_failure", data=np.asarray(labels["no_ik_failure"])
            )
        if "stage" in labels:
            _write_utf8_scalar(labels_group, "stage", labels["stage"])
    elif task_id == "wrist_angle":
        targets = labels.get("targets", {})
        if not isinstance(targets, dict) or not targets:
            raise ValueError("Wrist-angle records require non-empty labels['targets']")
        target_group = labels_group.create_group("targets")
        for name, values in targets.items():
            target_group.create_dataset(str(name), data=np.asarray(values))
        if "stage" in labels:
            _write_utf8_scalar(labels_group, "stage", labels["stage"])


def write_record(
    path: str | Path, record: CanonicalRecord, *, overwrite: bool = False
) -> Path:
    """Write a validated canonical record without transforming its EMG samples."""
    validate_record(record)
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing canonical record: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    relative_time_s = record.time_s.astype(np.float64, copy=False) - record.time_s[0]
    emg = record.emg.astype(np.float32, copy=False)
    with h5py.File(output, "w") as h5:
        h5.attrs.update(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": record.dataset_id,
                "task_id": record.task_id,
                "subject_id": record.subject_id,
                "session_id": record.session_id,
                "side": record.side,
                "pair_id": record.pair_id,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "official_split": record.official_split,
                "source_file": record.source_file,
                "preprocessing_json": json.dumps(
                    record.preprocessing, default=_json_default, sort_keys=True
                ),
            }
        )
        signal = h5.create_group("signal")
        signal.create_dataset("emg", data=emg, compression="gzip")
        signal.create_dataset("time_s", data=relative_time_s)

        labels = h5.create_group("labels")
        _write_labels(labels, record.labels)
        _write_task_labels(labels, record.task_id, record.labels)

        meta = h5.create_group("meta")
        meta.create_dataset(
            "source_json",
            data=json.dumps(record.source_meta, default=_json_default, sort_keys=True),
            dtype=h5py.string_dtype("utf-8"),
        )
        quality = meta.create_group("quality")
        quality.create_dataset("sample_count", data=len(emg))
        quality.create_dataset("duration_s", data=float(relative_time_s[-1]))
        quality.create_dataset("nan_count", data=int(np.isnan(emg).sum()))
        quality.create_dataset("channel_rms", data=np.sqrt(np.mean(emg**2, axis=0)))
    return output


def read_canonical_header(path: str | Path) -> dict[str, Any]:
    """Read enough canonical metadata for lightweight test and tree validation."""
    with h5py.File(path, "r") as h5:
        return {
            "schema_version": str(h5.attrs["schema_version"]),
            "emg_shape": tuple(h5["signal/emg"].shape),
            "time_s": h5["signal/time_s"][:],
        }
