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


def test_manifest_rejects_directory_record_path(tmp_path):
    path = tmp_path / "manifest.csv"
    columns = {
        "path": ["record.h5"],
        "dataset_id": ["canonical"], "task_id": ["pose"], "subject_id": ["s"],
        "session_id": ["session"], "side": ["right"], "pair_id": [""],
        "official_split": ["train"], "sample_count": [1], "duration_s": [0.0],
        "source_file": ["record.h5"], "source_sha256": ["0" * 64],
    }
    (tmp_path / "record.h5").mkdir()
    pd.DataFrame(columns).to_csv(path, index=False)
    with pytest.raises(FileNotFoundError, match="missing record paths"):
        load_manifest(path, tmp_path)


def test_canonical_fixture_has_root_schema_attributes(canonical_meta_root):
    import h5py

    with h5py.File(canonical_meta_root / "pose" / "pose_right.h5", "r") as h5:
        assert h5.attrs["schema_version"]
        assert h5.attrs["source_file"]
        assert h5.attrs["preprocessing_json"]
