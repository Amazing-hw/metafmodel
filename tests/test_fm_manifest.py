import pandas as pd
import pytest

from semg_fm.data.manifest import load_manifest


def test_manifest_resolves_records_and_preserves_task_ids(canonical_meta_root):
    frame = load_manifest(canonical_meta_root / "manifest.csv", canonical_meta_root)
    assert set(frame.task_id) == {"pose", "typing", "discrete_gesture", "handwriting", "wrist_angle"}
    assert frame.resolved_path.map(lambda p: p.exists()).all()


def test_manifest_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame({"path": ["x.h5"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_manifest(path, tmp_path)
