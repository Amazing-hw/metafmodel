"""Validation and path resolution for canonical training manifests."""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "path",
    "dataset_id",
    "task_id",
    "subject_id",
    "session_id",
    "side",
    "pair_id",
    "official_split",
    "sample_count",
    "duration_s",
    "source_file",
    "source_sha256",
}


def load_manifest(manifest_path, root):
    """Load a canonical manifest, validate its schema, and resolve record paths."""
    manifest_path = Path(manifest_path)
    root = Path(root)
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")

    frame["resolved_path"] = frame["path"].map(lambda value: root / value)
    absent = [str(path) for path in frame["resolved_path"] if not path.exists()]
    if absent:
        preview = ", ".join(absent[:3])
        raise FileNotFoundError(f"missing record paths: {preview}")
    return frame
