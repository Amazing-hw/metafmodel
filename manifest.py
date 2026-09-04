"""Stable provenance manifest for canonical sEMG HDF5 records."""

import csv
from pathlib import Path
from typing import Any

import pandas as pd


MANIFEST_COLUMNS = [
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
]


def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    return pd.read_csv(path, dtype=str, keep_default_na=False)[MANIFEST_COLUMNS]


def append_manifest_row(path: str | Path, row: dict[str, Any]) -> None:
    """Upsert one row, sort it by relative output path, then replace atomically."""
    path = Path(path)
    missing = set(MANIFEST_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"Manifest row missing required fields: {sorted(missing)}")
    rows = read_manifest(path).to_dict("records")
    normalized = {column: row[column] for column in MANIFEST_COLUMNS}
    rows = [existing for existing in rows if existing["path"] != normalized["path"]]
    rows.append(normalized)
    rows.sort(key=lambda item: str(item["path"]))

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
