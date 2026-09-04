"""Source-layout readers and provenance helpers for Meta public datasets."""

import hashlib
import json
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


def _json_rows(value: Any, label_name: str) -> list[dict[str, Any]]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON for {label_name}: {error}") from error
    if isinstance(value, dict):
        value = value.get(label_name, value.get("rows", []))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{label_name} must decode to a list of objects")
    return [{str(key): decode_value(item) for key, item in row.items()} for row in value]


def _dataset_rows(group: h5py.Group, names: tuple[str, ...]) -> list[dict[str, Any]] | None:
    for name in names:
        if name not in group:
            continue
        dataset = group[name]
        if not isinstance(dataset, h5py.Dataset):
            continue
        if dataset.dtype.names is None:
            return _json_rows(dataset.asstr()[()], name)
        return [
            {field: decode_value(row[field]) for field in dataset.dtype.names}
            for row in dataset[:]
        ]
    return None


def _metadata_rows(
    group: h5py.Group, metadata: dict[str, Any], names: tuple[str, ...]
) -> list[dict[str, Any]]:
    rows = _dataset_rows(group, names)
    if rows is not None:
        return rows
    for name in names:
        for candidate in (name, f"{name}_json"):
            if candidate in metadata:
                return _json_rows(metadata[candidate], name)
    return []


def _required_value(row: dict[str, Any], names: tuple[str, ...], label: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise ValueError(f"{label} row lacks one of {names}")


def _relative_qwerty_labels(
    group: h5py.Group, metadata: dict[str, Any], origin_s: float
) -> dict[str, Any]:
    keystrokes = []
    for row in _metadata_rows(group, metadata, ("keystrokes", "keylogger")):
        keystrokes.append(
            {
                "key": str(_required_value(row, ("key", "key_name", "name"), "keystroke")),
                "press_s": float(
                    _required_value(row, ("press_s", "press", "down_s", "down"), "keystroke")
                )
                - origin_s,
                "release_s": float(
                    _required_value(
                        row, ("release_s", "release", "up_s", "up"), "keystroke"
                    )
                )
                - origin_s,
            }
        )
    prompts = []
    for row in _metadata_rows(group, metadata, ("prompts", "prompt")):
        prompts.append(
            {
                "text": str(_required_value(row, ("text", "prompt"), "prompt")),
                "start_s": float(_required_value(row, ("start_s", "start"), "prompt")) - origin_s,
                "end_s": float(_required_value(row, ("end_s", "end"), "prompt")) - origin_s,
            }
        )
    return {"keystrokes": keystrokes, "prompts": prompts}


def read_emg2qwerty(
    path: str | Path, official_split: str = "unknown"
) -> list[CanonicalRecord]:
    """Read one bilateral qwerty session as two linked canonical wrist records."""
    path = Path(path)
    with h5py.File(path, "r") as h5:
        if "emg2qwerty/timeseries" not in h5:
            raise ValueError(f"{path} lacks documented emg2qwerty/timeseries dataset")
        group = h5["emg2qwerty"]
        timeseries = h5["emg2qwerty/timeseries"]
        required = {"emg_left", "emg_right", "time"}
        fields = set(timeseries.dtype.names or ())
        missing = required - fields
        if missing:
            raise ValueError(f"{path} missing qwerty timeseries fields: {sorted(missing)}")
        data = timeseries[:]
        left, right, time_s = data["emg_left"], data["emg_right"], data["time"]
        if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape or left.shape[1] != 16:
            raise ValueError("qwerty EMG sides must be aligned [T,16] arrays")
        if time_s.ndim != 1 or len(time_s) != len(left):
            raise ValueError("qwerty time must align with both EMG sides")
        metadata = read_attributes(group)
        subject_raw = str(metadata.get("subject_id", path.parent.name))
        session_raw = str(metadata.get("session_id", path.stem))
        pair_id = f"emg2qwerty:{session_raw}"
        labels = _relative_qwerty_labels(group, metadata, float(time_s[0]))
        common = {
            "dataset_id": "emg2qwerty",
            "task_id": "typing",
            "subject_id": f"emg2qwerty:{subject_raw}",
            "session_id": pair_id,
            "pair_id": pair_id,
            "official_split": official_split,
            "source_file": str(path.resolve()),
            "time_s": np.asarray(time_s),
            "labels": labels,
            "source_meta": {
                "source_group": "emg2qwerty",
                "source_timeseries": "emg2qwerty/timeseries",
                "attributes": metadata,
            },
            "preprocessing": {"converter_transformations": ["time_s shifted to record-relative origin"]},
        }
        return [
            CanonicalRecord(side="left", emg=np.asarray(left), **common),
            CanonicalRecord(side="right", emg=np.asarray(right), **common),
        ]
