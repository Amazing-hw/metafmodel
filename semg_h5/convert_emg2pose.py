"""Convert emg2pose source HDF5 records into canonical per-wrist HDF5 files."""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCHEMA_VERSION = "1.0"
SAMPLE_RATE_HZ = 2000
MANIFEST_COLUMNS = [
    "path", "dataset_id", "task_id", "subject_id", "session_id", "side", "pair_id",
    "official_split", "sample_count", "duration_s", "source_file", "source_sha256",
]


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


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _decode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _validate_record(record: CanonicalRecord) -> None:
    if record.emg.ndim != 2 or record.emg.shape[1] != 16:
        raise ValueError("EMG must have shape [T, 16] with exactly 16 channels")
    if record.time_s.ndim != 1 or len(record.time_s) != len(record.emg):
        raise ValueError("time_s must be a length-T vector matching EMG")
    if not np.isfinite(record.emg).all() or not np.isfinite(record.time_s).all():
        raise ValueError("EMG and time_s must be finite")
    if len(record.time_s) > 1 and np.any(np.diff(record.time_s) < 0):
        raise ValueError("time_s must be monotonic")


def _write_record(path: str | Path, record: CanonicalRecord, *, overwrite: bool = False) -> Path:
    _validate_record(record)
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing canonical record: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    relative_time_s = record.time_s.astype(np.float64, copy=False) - record.time_s[0]
    emg = record.emg.astype(np.float32, copy=False)
    with h5py.File(output, "w") as h5:
        h5.attrs.update({
            "schema_version": SCHEMA_VERSION, "dataset_id": record.dataset_id,
            "task_id": record.task_id, "subject_id": record.subject_id,
            "session_id": record.session_id, "side": record.side, "pair_id": record.pair_id,
            "sample_rate_hz": SAMPLE_RATE_HZ, "official_split": record.official_split,
            "source_file": record.source_file,
            "preprocessing_json": json.dumps(record.preprocessing, default=_json_default, sort_keys=True),
        })
        signal = h5.create_group("signal")
        signal.create_dataset("emg", data=emg, compression="gzip")
        signal.create_dataset("time_s", data=relative_time_s)
        labels = h5.create_group("labels")
        labels.create_dataset("source_json", data=json.dumps(record.labels, default=_json_default, sort_keys=True), dtype=h5py.string_dtype("utf-8"))
        if "joint_angles" not in record.labels:
            raise ValueError("Pose records require labels['joint_angles']")
        labels.create_dataset("joint_angles", data=np.asarray(record.labels["joint_angles"]))
        if "no_ik_failure" in record.labels:
            labels.create_dataset("no_ik_failure", data=np.asarray(record.labels["no_ik_failure"]))
        if "stage" in record.labels:
            labels.create_dataset("stage", data=str(record.labels["stage"]), dtype=h5py.string_dtype("utf-8"))
        meta = h5.create_group("meta")
        meta.create_dataset("source_json", data=json.dumps(record.source_meta, default=_json_default, sort_keys=True), dtype=h5py.string_dtype("utf-8"))
        quality = meta.create_group("quality")
        quality.create_dataset("sample_count", data=len(emg))
        quality.create_dataset("duration_s", data=float(relative_time_s[-1]))
        quality.create_dataset("nan_count", data=int(np.isnan(emg).sum()))
        quality.create_dataset("channel_rms", data=np.sqrt(np.mean(emg**2, axis=0)))
    return output


def _source_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_component(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    return safe.strip(".") or "unknown"


def _output_relative_path(record: CanonicalRecord) -> Path:
    return (
        Path(_safe_component(record.dataset_id)) / _safe_component(record.task_id)
        / _safe_component(record.subject_id.split(":", 1)[-1])
        / f"{_safe_component(record.session_id.split(':', 1)[-1])}_{_safe_component(record.side)}.h5"
    )


def _build_manifest_row(record: CanonicalRecord, relative_path: Path) -> dict[str, Any]:
    return {
        "path": relative_path.as_posix(), "dataset_id": record.dataset_id, "task_id": record.task_id,
        "subject_id": record.subject_id, "session_id": record.session_id, "side": record.side,
        "pair_id": record.pair_id, "official_split": record.official_split,
        "sample_count": int(len(record.emg)), "duration_s": float(record.time_s[-1] - record.time_s[0]),
        "source_file": record.source_file, "source_sha256": _source_sha256(record.source_file),
    }


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append_manifest_row(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    missing = set(MANIFEST_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"Manifest row missing required fields: {sorted(missing)}")
    normalized = {column: row[column] for column in MANIFEST_COLUMNS}
    rows = [existing for existing in _read_manifest(path) if existing["path"] != normalized["path"]]
    rows.append(normalized)
    rows.sort(key=lambda item: str(item["path"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_emg2pose(path: str | Path, official_split: str = "unknown") -> CanonicalRecord:
    path = Path(path)
    with h5py.File(path, "r") as h5:
        if "emg2pose/timeseries" not in h5:
            raise ValueError(f"{path} lacks documented emg2pose/timeseries dataset")
        group, timeseries = h5["emg2pose"], h5["emg2pose/timeseries"]
        required = {"emg", "joint_angles", "time"}
        fields = set(timeseries.dtype.names or ())
        missing = required - fields
        if missing:
            raise ValueError(f"{path} missing emg2pose timeseries fields: {sorted(missing)}")
        data = timeseries[:]
        metadata = {str(key): _decode_value(value) for key, value in group.attrs.items()}
        subject_raw = str(metadata.get("subject_id", path.parent.name))
        session_raw = str(metadata.get("session_id", path.stem))
        labels: dict[str, Any] = {"joint_angles": data["joint_angles"]}
        if "no_ik_failure" in fields:
            labels["no_ik_failure"] = data["no_ik_failure"]
        if "stage" in metadata:
            labels["stage"] = metadata["stage"]
        return CanonicalRecord(
            dataset_id="emg2pose", task_id="pose", subject_id=f"emg2pose:{subject_raw}",
            session_id=f"emg2pose:{session_raw}", side=str(metadata.get("side", "unknown")), pair_id="",
            official_split=official_split, source_file=str(path.resolve()), emg=np.asarray(data["emg"]),
            time_s=np.asarray(data["time"]), labels=labels,
            source_meta={"source_group": "emg2pose", "source_timeseries": "emg2pose/timeseries", "attributes": metadata},
            preprocessing={"converter_transformations": ["time_s shifted to record-relative origin"]},
        )


def convert_file(source_path: str | Path, output_root: str | Path, *, official_split: str = "unknown", overwrite: bool = False) -> dict[str, object]:
    record = _read_emg2pose(source_path, official_split)
    relative_path = _output_relative_path(record)
    _write_record(Path(output_root) / relative_path, record, overwrite=overwrite)
    return _build_manifest_row(record, relative_path)


def _split_lookup(csv_path: str | Path | None) -> dict[str, str]:
    if csv_path is None:
        return {}
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or "split" not in rows.fieldnames:
            raise ValueError("Split CSV must include a 'split' column")
        key_column = next((name for name in ("path", "source_file", "file") if name in rows.fieldnames), None)
        if key_column is None:
            raise ValueError("Split CSV must include one of: path, source_file, file")
        return {str(row[key_column]).replace("\\", "/"): str(row["split"]) for row in rows}


def run_conversion(input_root: str | Path, output_root: str | Path, *, split: str | None = None, splits_csv: str | Path | None = None, overwrite: bool = False) -> list[dict[str, object]]:
    input_root, output_root = Path(input_root), Path(output_root)
    lookup = _split_lookup(splits_csv)
    rows: list[dict[str, object]] = []
    for source_path in sorted(input_root.rglob("*.h5")):
        relative_source = source_path.relative_to(input_root).as_posix()
        official_split = lookup.get(relative_source, lookup.get(source_path.name, "unknown"))
        if split is not None and official_split != split:
            continue
        row = convert_file(source_path, output_root, official_split=official_split, overwrite=overwrite)
        _append_manifest_row(output_root / "manifest.csv", row)
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", help="Convert only this official split")
    parser.add_argument("--splits-csv", help="CSV mapping source path or file to official split")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    rows = run_conversion(args.input_root, args.output_root, split=args.split, splits_csv=args.splits_csv, overwrite=args.overwrite)
    print(f"Converted {len(rows)} emg2pose records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
