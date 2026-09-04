from semg_h5.manifest import MANIFEST_COLUMNS, append_manifest_row, read_manifest


def test_manifest_row_exposes_task_selection_and_pairing(tmp_path):
    row = {
        "path": "typing/user_1/session_1_left.h5",
        "dataset_id": "emg2qwerty",
        "task_id": "typing",
        "subject_id": "emg2qwerty:user_1",
        "session_id": "emg2qwerty:session_1",
        "side": "left",
        "pair_id": "pair_1",
        "official_split": "train",
        "sample_count": 4,
        "duration_s": 0.0015,
        "source_file": "source.h5",
        "source_sha256": "abc",
    }
    manifest_path = tmp_path / "manifest.csv"
    append_manifest_row(manifest_path, row)

    manifest = read_manifest(manifest_path)
    assert list(manifest.columns) == MANIFEST_COLUMNS
    assert manifest.loc[0, "task_id"] == "typing"
    assert manifest.loc[0, "pair_id"] == "pair_1"
