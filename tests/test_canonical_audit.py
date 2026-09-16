import json
from pathlib import Path

import h5py
import pandas as pd
import pytest

from semg_fm.data.audit import audit_canonical_corpus, write_audit_reports


def make_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "canonical"
    sample = root / "demo" / "subject-001" / "session-001" / "sample.h5"
    sample.parent.mkdir(parents=True)
    with h5py.File(sample, "w") as handle:
        signal = handle.create_group("signal")
        signal.create_dataset("emg", data=[[0.0] * 16] * 20)
    pd.DataFrame(
        [
            {
                "path": sample.relative_to(root).as_posix(),
                "dataset_id": "demo",
                "task_id": "pose",
                "subject_id": "subject-001",
                "session_id": "session-001",
                "duration_s": 0.01,
                "official_split": "train",
            }
        ]
    ).to_csv(root / "manifest.csv", index=False)
    return root


def test_audit_reports_counts_and_zero_missing_files(tmp_path):
    root = make_corpus(tmp_path)

    report, rows = audit_canonical_corpus(
        root, root / "manifest.csv", h5_samples=1
    )

    assert report["total_records"] == 1
    assert report["missing_files"] == 0
    assert report["sampled_h5_files"] == 1
    assert report["h5_schema_errors"] == []
    assert rows[0]["task_id"] == "pose"


def test_audit_rejects_missing_required_columns(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"path": "x.h5"}]).to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="missing required manifest columns"):
        audit_canonical_corpus(tmp_path, manifest, h5_samples=0)


def test_write_audit_reports_creates_json_and_csv(tmp_path):
    root = make_corpus(tmp_path)
    report, rows = audit_canonical_corpus(
        root, root / "manifest.csv", h5_samples=0
    )

    json_path, csv_path = write_audit_reports(
        tmp_path / "reports" / "audit", report, rows
    )

    assert json.loads(json_path.read_text(encoding="utf-8"))["total_records"] == 1
    assert pd.read_csv(csv_path).iloc[0]["records"] == 1
