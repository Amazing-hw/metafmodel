"""Convert GNI wrist-angle recordings to canonical HDF5."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

import h5py
import numpy as np
import pandas as pd


SCHEMA_VERSION, SAMPLE_RATE_HZ = "1.0", 2000
EXPECTED_SOURCE_TASK, DATASET_ID, TASK_ID = "wrist", "gni_wrist", "wrist_angle"
MANIFEST_COLUMNS = ["path", "dataset_id", "task_id", "subject_id", "session_id", "side", "pair_id", "official_split", "sample_count", "duration_s", "source_file", "source_sha256"]


def _decode(value: Any) -> Any:
    if isinstance(value, bytes): return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    return value


def _json_default(value: Any) -> Any:
    return _decode(value) if isinstance(value, (bytes, np.generic, np.ndarray)) else str(value)


def _rows(path: Path, key: str) -> list[dict[str, Any]]:
    try:
        frame = pd.read_hdf(path, key=key)
    except (FileNotFoundError, KeyError, OSError, ValueError, TypeError):
        with h5py.File(path, "r") as h5:
            if key not in h5 or not isinstance(h5[key], h5py.Dataset) or h5[key].dtype.names is None: return []
            return [{field: _decode(row[field]) for field in h5[key].dtype.names} for row in h5[key][:]]
    return [{str(column): _decode(value) for column, value in row.items()} for row in frame.to_dict("records")]


def _safe(value: str) -> str:
    return ("".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip(".") or "unknown")


def _read_gni(path: str | Path, official_split: str) -> dict[str, Any]:
    path = Path(path)
    with h5py.File(path, "r") as h5:
        if "data" not in h5 or not isinstance(h5["data"], h5py.Dataset): raise ValueError(f"{path} lacks documented GNI 'data' dataset")
        source = h5["data"]; metadata = {str(key): _decode(value) for key, value in source.attrs.items()}
        if str(metadata.get("task", "")) != EXPECTED_SOURCE_TASK: raise ValueError(f"{path} has task {metadata.get('task', '')!r}; {EXPECTED_SOURCE_TASK!r} was required")
        fields, data = list(source.dtype.names or ()), source[:]
        if "time" not in fields: raise ValueError(f"{path} GNI data lacks required 'time' field")
        candidates = [field for field in fields if np.asarray(data[field]).ndim == 2 and np.asarray(data[field]).shape[1] == 16]
        emg_field = "emg" if "emg" in candidates else candidates[0] if len(candidates) == 1 else None
        if emg_field is None: raise ValueError(f"{path} must contain exactly one 16-channel GNI EMG field")
        emg, time_s = np.asarray(data[emg_field]), np.asarray(data["time"])
        if time_s.ndim != 1 or len(time_s) != len(emg): raise ValueError("GNI time must be a T-length vector aligned with EMG")
        if len(emg) == 0: raise ValueError("GNI records must contain at least one sample")
        targets: dict[str, np.ndarray] = {}
        target_fields: dict[str, str] = {}
        for field in fields:
            values = np.asarray(data[field])
            if field in {"time", emg_field} or not np.issubdtype(values.dtype, np.number): continue
            if values.shape[0] != len(emg): raise ValueError(f"GNI wrist target {field!r} does not align with EMG")
            target_name = _safe(field)
            if target_name in targets:
                raise ValueError(f"GNI wrist target sanitization collision: {target_fields[target_name]!r} and {field!r} both map to {target_name!r}")
            target_fields[target_name] = field
            targets[target_name] = values
    stages = _rows(path, "stages"); labels: dict[str, Any] = {"targets": targets}
    if len(stages) == 1 and "stage" in stages[0]: labels["stage"] = stages[0]["stage"]
    return {"dataset_id": DATASET_ID, "task_id": TASK_ID, "subject_id": f"{DATASET_ID}:{metadata.get('subject_id', path.parent.name)}", "session_id": f"{DATASET_ID}:{metadata.get('session_id', path.stem)}", "side": str(metadata.get("side", "dominant")), "pair_id": "", "official_split": official_split, "source_file": str(path.resolve()), "emg": emg, "time_s": time_s, "labels": labels, "source_meta": {"source_dataset": "data", "emg_field": emg_field, "attributes": metadata, "stages": stages}, "preprocessing": {"converter_transformations": ["time_s shifted to record-relative origin"]}}


def _relative(record: dict[str, Any]) -> Path:
    return Path(_safe(record["dataset_id"])) / _safe(record["task_id"]) / _safe(record["subject_id"].split(":", 1)[-1]) / f"{_safe(record['session_id'].split(':', 1)[-1])}_{_safe(record['side'])}.h5"


def _validate(record: dict[str, Any]) -> None:
    emg, time_s = record["emg"], record["time_s"]
    if emg.ndim != 2 or emg.shape[1] != 16: raise ValueError("EMG must have shape [T, 16] with exactly 16 channels")
    if time_s.ndim != 1 or len(time_s) != len(emg): raise ValueError("time_s must be a length-T vector matching EMG")
    if len(emg) == 0: raise ValueError("GNI records must contain at least one sample")
    if not np.isfinite(emg).all() or not np.isfinite(time_s).all(): raise ValueError("EMG and time_s must be finite")
    if len(time_s) > 1 and np.any(np.diff(time_s) < 0): raise ValueError("time_s must be monotonic")


def _write(path: str | Path, record: dict[str, Any], *, overwrite: bool) -> None:
    _validate(record); path = Path(path)
    if path.exists() and not overwrite: raise FileExistsError(f"Refusing to overwrite existing canonical record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True); relative_time = record["time_s"].astype(np.float64, copy=False) - record["time_s"][0]
    if not record["labels"]["targets"]: raise ValueError("Wrist-angle records require non-empty labels['targets']")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with h5py.File(temporary, "w") as h5:
        h5.attrs.update({"schema_version": SCHEMA_VERSION, **{key: record[key] for key in ("dataset_id", "task_id", "subject_id", "session_id", "side", "pair_id", "official_split", "source_file")}, "sample_rate_hz": SAMPLE_RATE_HZ, "preprocessing_json": json.dumps(record["preprocessing"], default=_json_default, sort_keys=True)})
        signal = h5.create_group("signal"); signal.create_dataset("emg", data=record["emg"].astype(np.float32, copy=False), compression="gzip"); signal.create_dataset("time_s", data=relative_time)
        text = h5py.string_dtype("utf-8"); labels = h5.create_group("labels"); labels.create_dataset("source_json", data=json.dumps(record["labels"], default=_json_default, sort_keys=True), dtype=text)
        targets = labels.create_group("targets")
        for name, values in record["labels"]["targets"].items(): targets.create_dataset(str(name), data=np.asarray(values))
        if "stage" in record["labels"]: labels.create_dataset("stage", data=str(record["labels"]["stage"]), dtype=text)
        meta = h5.create_group("meta"); meta.create_dataset("source_json", data=json.dumps(record["source_meta"], default=_json_default, sort_keys=True), dtype=text)
        quality = meta.create_group("quality"); quality.create_dataset("sample_count", data=len(record["emg"])); quality.create_dataset("duration_s", data=float(relative_time[-1])); quality.create_dataset("nan_count", data=int(np.isnan(record["emg"]).sum())); quality.create_dataset("channel_rms", data=np.sqrt(np.mean(record["emg"].astype(np.float32, copy=False) ** 2, axis=0)))
    os.replace(temporary, path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _manifest(record: dict[str, Any], relative: Path) -> dict[str, Any]:
    return {"path": relative.as_posix(), **{key: record[key] for key in ("dataset_id", "task_id", "subject_id", "session_id", "side", "pair_id", "official_split", "source_file")}, "sample_count": int(len(record["emg"])), "duration_s": float(record["time_s"][-1] - record["time_s"][0]), "source_sha256": _sha256(record["source_file"])}


def _append(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); lock = path.with_name(f".{path.name}.lock"); deadline = time.monotonic() + 10.0
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(descriptor); break
        except FileExistsError:
            if time.monotonic() >= deadline: raise TimeoutError(f"Timed out waiting for manifest lock: {lock}")
            time.sleep(0.05)
    try:
        rows = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
        rows = [item for item in rows if item["path"] != row["path"]] + [{column: row[column] for column in MANIFEST_COLUMNS}]; rows.sort(key=lambda item: item["path"])
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS); writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        lock.unlink(missing_ok=True)


def _splits(path: str | Path | None) -> dict[str, str]:
    if path is None: return {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or "split" not in rows.fieldnames: raise ValueError("Split CSV must include a 'split' column")
        column = next((name for name in ("path", "source_file", "file") if name in rows.fieldnames), None)
        if column is None: raise ValueError("Split CSV must include one of: path, source_file, file")
        return {str(row[column]).replace("\\", "/"): str(row["split"]) for row in rows}


def convert_file(source_path: str | Path, output_root: str | Path, *, official_split: str = "unknown", overwrite: bool = False) -> dict[str, object]:
    record = _read_gni(source_path, official_split); relative = _relative(record); _write(Path(output_root) / relative, record, overwrite=overwrite); return _manifest(record, relative)


def run_conversion(input_root: str | Path, output_root: str | Path, *, split: str | None = None, splits_csv: str | Path | None = None, overwrite: bool = False) -> list[dict[str, object]]:
    input_root, output_root, lookup = Path(input_root), Path(output_root), _splits(splits_csv); results = []
    try:
        output_root.resolve().relative_to(input_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("output-root must not be nested under input-root")
    for source in sorted(input_root.rglob("*.h5")):
        official_split = lookup.get(source.relative_to(input_root).as_posix(), lookup.get(source.name, "unknown"))
        if split is None or official_split == split:
            row = convert_file(source, output_root, official_split=official_split, overwrite=overwrite); _append(output_root / "manifest.csv", row); results.append(row)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--input-root", required=True); parser.add_argument("--output-root", required=True); parser.add_argument("--split"); parser.add_argument("--splits-csv"); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv); rows = run_conversion(args.input_root, args.output_root, split=args.split, splits_csv=args.splits_csv, overwrite=args.overwrite); print(f"Converted {len(rows)} {DATASET_ID} records."); return 0


if __name__ == "__main__": raise SystemExit(main())
