"""Read-only validation and summary reporting for canonical sEMG corpora."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import pandas as pd


REQUIRED_COLUMNS = {
    "path",
    "dataset_id",
    "task_id",
    "subject_id",
    "session_id",
    "duration_s",
    "official_split",
}
ALLOWED_TASKS = {
    "pose",
    "wrist_angle",
    "discrete_gesture",
    "handwriting",
    "typing",
}
ALLOWED_SPLITS = {"", "unknown", "train", "val", "test"}


def _resolve_record_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def audit_canonical_corpus(
    root: Path, manifest_path: Path, h5_samples: int = 0
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one canonical manifest and summarize it without modifying data."""
    if h5_samples < 0:
        raise ValueError("h5_samples must be non-negative")

    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"missing required manifest columns: {', '.join(missing_columns)}"
        )

    tasks = frame["task_id"].astype(str).str.strip()
    official = frame["official_split"].astype(str).str.strip()
    invalid_tasks = sorted(set(tasks) - ALLOWED_TASKS)
    invalid_splits = sorted(set(official) - ALLOWED_SPLITS)
    resolved = [
        _resolve_record_path(root, value)
        for value in frame["path"].astype(str)
    ]
    existing = [path for path in resolved if path.is_file()]
    missing_paths = [str(path) for path in resolved if not path.is_file()]

    sampled = 0
    schema_errors: list[str] = []
    for path in existing[:h5_samples]:
        try:
            with h5py.File(path, "r") as handle:
                if "signal/emg" not in handle:
                    schema_errors.append(f"{path}: missing signal/emg dataset")
                else:
                    emg = handle["signal/emg"]
                    if emg.ndim != 2 or emg.shape[1] != 16:
                        schema_errors.append(
                            f"{path}: signal/emg must have shape [time, 16]"
                        )
            sampled += 1
        except OSError as exc:
            schema_errors.append(f"{path}: {exc}")

    durations = pd.to_numeric(frame["duration_s"], errors="raise")
    summary_frame = frame.assign(duration_s=durations)
    grouped = (
        summary_frame.groupby(["dataset_id", "task_id"], dropna=False)
        .agg(
            records=("path", "size"),
            subjects=("subject_id", "nunique"),
            sessions=("session_id", "nunique"),
            duration_s=("duration_s", "sum"),
        )
        .reset_index()
    )
    grouped["duration_hours"] = grouped["duration_s"] / 3600.0
    rows = grouped.drop(columns=["duration_s"]).to_dict(orient="records")
    report: dict[str, Any] = {
        "root": str(root),
        "manifest": str(manifest_path),
        "total_records": int(len(frame)),
        "total_subjects": int(frame["subject_id"].nunique()),
        "total_sessions": int(frame["session_id"].nunique()),
        "duration_hours": float(durations.sum() / 3600.0),
        "missing_files": len(missing_paths),
        "missing_paths": missing_paths,
        "invalid_tasks": invalid_tasks,
        "invalid_splits": invalid_splits,
        "sampled_h5_files": sampled,
        "h5_schema_errors": schema_errors,
        "official_split_counts": {
            str(key): int(value) for key, value in official.value_counts().items()
        },
    }
    return report, rows


def write_audit_reports(
    output_stem: Path, report: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[Path, Path]:
    """Write one JSON summary and one dataset/task CSV table."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_stem.with_suffix(".json")
    csv_path = output_stem.with_suffix(".csv")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(csv_path, index=False, lineterminator="\n")
    return json_path, csv_path
