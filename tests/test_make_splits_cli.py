import subprocess
import sys

import pandas as pd
import pytest

from semg_fm.data.make_splits import main


def test_root_make_splits_assigns_unknown_and_preserves_official(tmp_path):
    record = tmp_path / "record.h5"
    record.write_bytes(b"x")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "path": ["record.h5"], "dataset_id": ["d"], "task_id": ["t"],
            "subject_id": ["s"], "session_id": ["x"], "side": ["right"],
            "pair_id": [""], "official_split": ["unknown"], "sample_count": [1],
            "duration_s": [0.0], "source_file": ["record.h5"], "source_sha256": ["0" * 64],
        }
    ).to_csv(manifest, index=False)
    output = tmp_path / "split.csv"
    subprocess.run(
        [sys.executable, "make_splits.py", "--manifest", str(manifest), "--root", str(tmp_path),
         "--output", str(output), "--seed", "17"], check=True,
    )
    assert pd.read_csv(output).split.iloc[0] in {"train", "val", "test"}


def test_invalid_official_split_is_rejected(tmp_path):
    (tmp_path / "record.h5").write_bytes(b"x")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "path": ["record.h5"], "dataset_id": ["d"], "task_id": ["t"], "subject_id": ["s"],
            "session_id": ["x"], "side": ["right"], "pair_id": [""], "official_split": ["holdout"],
            "sample_count": [1], "duration_s": [0.0], "source_file": ["record.h5"], "source_sha256": ["0" * 64],
        }
    ).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="invalid official_split"):
        main(["--manifest", str(manifest), "--root", str(tmp_path), "--output", str(tmp_path / "out.csv"), "--seed", "1"])
