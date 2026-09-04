import importlib

import h5py
import pytest


@pytest.mark.parametrize(
    ("converter_module", "task_id", "label_path"),
    [
        ("convert_gni_discrete_gestures", "discrete_gesture", "labels/events"),
        ("convert_gni_handwriting", "handwriting", "labels/prompts"),
        ("convert_gni_wrist", "wrist_angle", "labels/targets/flexion_extension"),
    ],
)
def test_convert_gni_task_preserves_16ch_stream_and_its_labels(
    tmp_path, gni_h5_by_task, converter_module, task_id, label_path
):
    module = importlib.import_module(f"semg_h5.{converter_module}")
    row = module.convert_file(gni_h5_by_task[task_id], tmp_path / "out", official_split="test")
    with h5py.File(tmp_path / "out" / row["path"]) as h5:
        assert h5.attrs["task_id"] == task_id
        assert h5["signal/emg"].shape == (4, 16)
        assert label_path in h5
