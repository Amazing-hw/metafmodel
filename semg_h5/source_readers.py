"""Source-layout readers and provenance helpers for Meta public datasets."""

import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .canonical import CanonicalRecord


def decode_value(value: Any) -> Any:
    """Convert HDF5 scalar values into JSON-serializable Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def read_attributes(source: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    return {str(key): decode_value(value) for key, value in source.attrs.items()}


def source_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_component(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    return safe.strip(".") or "unknown"


def output_relative_path(record: CanonicalRecord) -> Path:
    """Return a portable, task-namespaced record location below output_root."""
    return (
        Path(safe_component(record.dataset_id))
        / safe_component(record.task_id)
        / safe_component(record.subject_id.split(":", 1)[-1])
        / f"{safe_component(record.session_id.split(':', 1)[-1])}_{safe_component(record.side)}.h5"
    )


def build_manifest_row(record: CanonicalRecord, relative_path: Path) -> dict[str, Any]:
    return {
        "path": relative_path.as_posix(),
        "dataset_id": record.dataset_id,
        "task_id": record.task_id,
        "subject_id": record.subject_id,
        "session_id": record.session_id,
        "side": record.side,
        "pair_id": record.pair_id,
        "official_split": record.official_split,
        "sample_count": int(len(record.emg)),
        "duration_s": float(record.time_s[-1] - record.time_s[0]),
        "source_file": record.source_file,
        "source_sha256": source_sha256(record.source_file),
    }


def read_emg2pose(path: str | Path, official_split: str = "unknown") -> CanonicalRecord:
    path = Path(path)
    with h5py.File(path, "r") as h5:
        if "emg2pose/timeseries" not in h5:
            raise ValueError(f"{path} lacks documented emg2pose/timeseries dataset")
        group = h5["emg2pose"]
        timeseries = h5["emg2pose/timeseries"]
        required = {"emg", "joint_angles", "time"}
        fields = set(timeseries.dtype.names or ())
        missing = required - fields
        if missing:
            raise ValueError(f"{path} missing emg2pose timeseries fields: {sorted(missing)}")
        data = timeseries[:]
        metadata = read_attributes(group)
        subject_raw = str(metadata.get("subject_id", path.parent.name))
        session_raw = str(metadata.get("session_id", path.stem))
        labels: dict[str, Any] = {"joint_angles": data["joint_angles"]}
        if "no_ik_failure" in fields:
            labels["no_ik_failure"] = data["no_ik_failure"]
        if "stage" in metadata:
            labels["stage"] = metadata["stage"]
        return CanonicalRecord(
            dataset_id="emg2pose",
            task_id="pose",
            subject_id=f"emg2pose:{subject_raw}",
            session_id=f"emg2pose:{session_raw}",
            side=str(metadata.get("side", "unknown")),
            pair_id="",
            official_split=official_split,
            source_file=str(path.resolve()),
            emg=np.asarray(data["emg"]),
            time_s=np.asarray(data["time"]),
            labels=labels,
            source_meta={
                "source_group": "emg2pose",
                "source_timeseries": "emg2pose/timeseries",
                "attributes": metadata,
            },
            preprocessing={"converter_transformations": ["time_s shifted to record-relative origin"]},
        )
