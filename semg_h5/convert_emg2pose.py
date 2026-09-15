"""将 emg2pose 源 HDF5 记录转换为 canonical 的单腕 HDF5 文件。"""

import argparse
import csv
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os

if os.name == "nt":
    import msvcrt

    def _try_lock(handle) -> None:
        """Take an OS-released advisory lock, raising OSError while it is held."""
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(handle) -> None:
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _try_lock(handle) -> None:
        """Take an OS-released advisory lock, raising OSError while it is held."""
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

from pathlib import Path
import time
from typing import Any
import uuid

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
    source_path: str
    emg: np.ndarray
    time_s: np.ndarray
    labels: dict[str, Any]
    source_meta: dict[str, Any]
    preprocessing: dict[str, Any]


@contextmanager
def _replace_after_success(temporary: Path, destination: Path):
    try:
        yield
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    try:
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _acquire_manifest_lock(path: Path):
    lock = path.with_name(f".{path.name}.lock")
    handle = lock.open("a+b")
    try:
        if lock.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + 10.0
        while True:
            try:
                handle.seek(0)
                _try_lock(handle)
                return handle
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for manifest lock: {lock}")
                time.sleep(0.05)
    except BaseException:
        handle.close()
        raise


def _release_manifest_lock(handle) -> None:
    try:
        handle.seek(0)
        _unlock(handle)
    finally:
        handle.close()


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
    if len(record.emg) == 0:
        raise ValueError("emg2pose records must contain at least one sample")
    if not np.isfinite(record.emg).all() or not np.isfinite(record.time_s).all():
        raise ValueError("EMG and time_s must be finite")
    if len(record.time_s) > 1 and np.any(np.diff(record.time_s) < 0):
        raise ValueError("time_s must be monotonic")
    if record.task_id == "pose":
        if "joint_angles" not in record.labels:
            raise ValueError("Pose records require labels['joint_angles']")
        joint_angles = np.asarray(record.labels["joint_angles"])
        if joint_angles.ndim == 0 or joint_angles.shape[0] != len(record.emg):
            raise ValueError("joint_angles must align with EMG samples")
        if not np.issubdtype(joint_angles.dtype, np.number) or not np.isfinite(joint_angles).all():
            raise ValueError("joint_angles must be finite numeric values")
        if "no_ik_failure" in record.labels:
            no_ik_failure = np.asarray(record.labels["no_ik_failure"])
            if no_ik_failure.ndim == 0 or no_ik_failure.shape[0] != len(record.emg):
                raise ValueError("no_ik_failure must align with EMG samples")


def _write_record(path: str | Path, record: CanonicalRecord, *, overwrite: bool = False) -> Path:
    _validate_record(record)
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing canonical record: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    relative_time_s = record.time_s.astype(np.float64, copy=False) - record.time_s[0]
    emg = record.emg.astype(np.float32, copy=False)
    with _replace_after_success(temporary, output), h5py.File(temporary, "w") as h5:
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
        "source_file": record.source_file, "source_sha256": _source_sha256(record.source_path),
    }


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append_manifest_rows(path: str | Path, new_rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    for row in new_rows:
        missing = set(MANIFEST_COLUMNS) - set(row)
        if missing:
            raise ValueError(f"Manifest row missing required fields: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = _acquire_manifest_lock(path)
    try:
        normalized = [{column: row[column] for column in MANIFEST_COLUMNS} for row in new_rows]
        replaced_paths = {row["path"] for row in normalized}
        rows = [existing for existing in _read_manifest(path) if existing["path"] not in replaced_paths]
        rows.extend(normalized)
        rows.sort(key=lambda item: str(item["path"]))
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with _replace_after_success(temporary, path), temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    finally:
        _release_manifest_lock(lock_handle)


def _append_manifest_row(path: str | Path, row: dict[str, Any]) -> None:
    _append_manifest_rows(path, [row])


def _read_emg2pose(path: str | Path, official_split: str = "unknown", source_file: str | None = None) -> CanonicalRecord:
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
        emg, time_s = np.asarray(data["emg"]), np.asarray(data["time"])
        if len(emg) == 0:
            raise ValueError("emg2pose records must contain at least one sample")
        joint_angles = np.asarray(data["joint_angles"])
        if joint_angles.ndim == 0 or joint_angles.shape[0] != len(emg):
            raise ValueError("joint_angles must align with EMG samples")
        if not np.issubdtype(joint_angles.dtype, np.number) or not np.isfinite(joint_angles).all():
            raise ValueError("joint_angles must be finite numeric values")
        if "no_ik_failure" in fields:
            no_ik_failure = np.asarray(data["no_ik_failure"])
            if no_ik_failure.ndim == 0 or no_ik_failure.shape[0] != len(emg):
                raise ValueError("no_ik_failure must align with EMG samples")
        metadata = {str(key): _decode_value(value) for key, value in group.attrs.items()}
        # The release carries the subject as `user` and the session as `session`;
        # `subject_id`/`session_id` are absent.  Falling back to the parent
        # directory would give every subject the same id and collapse
        # leakage-safe subject grouping into one group.
        subject_raw = str(metadata.get("subject_id") or metadata.get("user") or path.parent.name)
        session_raw = str(metadata.get("session_id") or metadata.get("session") or path.stem)
        labels: dict[str, Any] = {"joint_angles": joint_angles}
        if "no_ik_failure" in fields:
            labels["no_ik_failure"] = data["no_ik_failure"]
        if "stage" in metadata:
            labels["stage"] = metadata["stage"]
        return CanonicalRecord(
            dataset_id="emg2pose", task_id="pose", subject_id=f"emg2pose:{subject_raw}",
            session_id=f"emg2pose:{session_raw}", side=str(metadata.get("side", "unknown")), pair_id="",
            official_split=official_split, source_file=source_file or path.name, source_path=str(path.resolve()), emg=emg,
            time_s=time_s, labels=labels,
            source_meta={"source_group": "emg2pose", "source_timeseries": "emg2pose/timeseries", "attributes": metadata, "source_time_origin_s": float(time_s[0])},
            preprocessing={"converter_transformations": ["time_s shifted to record-relative origin"]},
        )


def convert_file(source_path: str | Path, output_root: str | Path, *, official_split: str = "unknown", overwrite: bool = False, source_file: str | None = None) -> dict[str, object]:
    record = _read_emg2pose(source_path, official_split, source_file)
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


def _source_files(input_root: Path) -> list[Path]:
    """List convertible source files under ``input_root``.

    The released corpora use both ``.h5`` (emg2pose, emg2qwerty) and ``.hdf5``
    (the GNI drop).  Globbing only ``*.h5`` matched nothing on the GNI corpora
    and the run reported success after converting zero records.
    """
    sources = sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix in {".h5", ".hdf5"}
    )
    if not sources:
        raise ValueError(f"no .h5 or .hdf5 source files found under {input_root}")
    return sources


def run_conversion(input_root: str | Path, output_root: str | Path, *, split: str | None = None, splits_csv: str | Path | None = None, overwrite: bool = False) -> list[dict[str, object]]:
    input_root, output_root = Path(input_root), Path(output_root)
    try:
        output_root.resolve().relative_to(input_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("output-root must not be nested under input-root")
    lookup = _split_lookup(splits_csv)
    rows: list[dict[str, object]] = []
    for source_path in _source_files(input_root):
        relative_source = source_path.relative_to(input_root).as_posix()
        official_split = lookup.get(relative_source, lookup.get(source_path.name, "unknown"))
        if split is not None and official_split != split:
            continue
        row = convert_file(source_path, output_root, official_split=official_split, overwrite=overwrite, source_file=relative_source)
        _append_manifest_rows(output_root / "manifest.csv", [row])
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
