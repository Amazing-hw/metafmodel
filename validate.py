"""Validate canonical H5 records and their optional provenance manifest."""

import argparse
from pathlib import Path

import h5py
import numpy as np

from .canonical import SCHEMA_VERSION
from .manifest import read_manifest


REQUIRED_ATTRIBUTES = {
    "schema_version",
    "dataset_id",
    "task_id",
    "subject_id",
    "session_id",
    "side",
    "pair_id",
    "sample_rate_hz",
    "official_split",
    "source_file",
    "preprocessing_json",
}


def validate_file(path: str | Path) -> list[str]:
    path = Path(path)
    errors: list[str] = []
    try:
        with h5py.File(path, "r") as h5:
            missing_attributes = REQUIRED_ATTRIBUTES - set(h5.attrs)
            if missing_attributes:
                errors.append(f"missing attributes {sorted(missing_attributes)}")
            elif str(h5.attrs["schema_version"]) != SCHEMA_VERSION:
                errors.append("unexpected schema_version")
            if "signal/emg" not in h5 or "signal/time_s" not in h5:
                errors.append("missing /signal/emg or /signal/time_s")
                return errors
            emg = h5["signal/emg"]
            time_s = h5["signal/time_s"][:]
            if emg.ndim != 2 or emg.shape[1] != 16:
                errors.append("/signal/emg must have shape [T,16]")
            if time_s.ndim != 1 or len(time_s) != emg.shape[0]:
                errors.append("/signal/time_s must have length T")
            elif not np.isfinite(time_s).all() or (len(time_s) > 1 and np.any(np.diff(time_s) < 0)):
                errors.append("/signal/time_s must be finite and monotonic")
            if "labels" not in h5 or "meta" not in h5:
                errors.append("missing /labels or /meta")
    except (OSError, KeyError, TypeError) as error:
        errors.append(str(error))
    return errors


def validate_tree(input_path: str | Path) -> dict[Path, list[str]]:
    root = Path(input_path)
    h5_files = sorted(root.rglob("*.h5")) if root.is_dir() else [root]
    failures = {path: errors for path in h5_files if (errors := validate_file(path))}
    manifest_path = root / "manifest.csv" if root.is_dir() else root.parent / "manifest.csv"
    if manifest_path.exists():
        for row in read_manifest(manifest_path).to_dict("records"):
            record_path = root / row["path"] if root.is_dir() else root.parent / row["path"]
            if not record_path.is_file():
                failures.setdefault(record_path, []).append("listed by manifest but missing")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate canonical Meta sEMG HDF5 files.")
    parser.add_argument("--input", required=True, help="Canonical H5 file or directory")
    args = parser.parse_args(argv)
    failures = validate_tree(args.input)
    if failures:
        for path, errors in failures.items():
            print(f"{path}: {'; '.join(errors)}")
        return 1
    print("Canonical H5 validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
