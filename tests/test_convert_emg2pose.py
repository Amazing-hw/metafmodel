import h5py


def test_convert_emg2pose_preserves_pose_labels_and_source_split(tmp_path, emg2pose_h5):
    from semg_h5.convert_emg2pose import convert_file

    manifest_row = convert_file(emg2pose_h5, tmp_path / "out", official_split="val")
    with h5py.File(tmp_path / "out" / manifest_row["path"]) as h5:
        assert h5.attrs["dataset_id"] == "emg2pose"
        assert h5.attrs["task_id"] == "pose"
        assert h5.attrs["official_split"] == "val"
        assert h5["signal/emg"].shape == (4, 16)
        assert h5["labels/joint_angles"].shape == (4, 3)
        assert h5["labels/no_ik_failure"].shape == (4,)
