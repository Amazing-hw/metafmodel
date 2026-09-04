import h5py
import json
import numpy as np
import pytest


@pytest.fixture
def emg2pose_h5(tmp_path):
    source_path = tmp_path / "user_1" / "session_1.h5"
    source_path.parent.mkdir()
    dtype = np.dtype(
        [
            ("emg", np.float32, (16,)),
            ("joint_angles", np.float32, (3,)),
            ("time", np.float64),
            ("no_ik_failure", np.bool_),
        ]
    )
    rows = np.zeros(4, dtype=dtype)
    rows["time"] = np.arange(4) / 2000 + 10.0
    rows["joint_angles"] = np.arange(12, dtype=np.float32).reshape(4, 3)
    rows["no_ik_failure"] = [True, True, False, True]
    with h5py.File(source_path, "w") as h5:
        group = h5.create_group("emg2pose")
        group.attrs["subject_id"] = "user_1"
        group.attrs["session_id"] = "session_1"
        group.attrs["side"] = "left"
        group.attrs["stage"] = "capture"
        group.create_dataset("timeseries", data=rows)
    return source_path


@pytest.fixture
def qwerty_h5(tmp_path):
    source_path = tmp_path / "user_2" / "typing_1.h5"
    source_path.parent.mkdir()
    dtype = np.dtype(
        [("emg_left", np.float32, (16,)), ("emg_right", np.float32, (16,)), ("time", np.float64)]
    )
    rows = np.zeros(4, dtype=dtype)
    rows["emg_left"] = 1
    rows["emg_right"] = 2
    rows["time"] = np.arange(4) / 2000 + 20.0
    with h5py.File(source_path, "w") as h5:
        group = h5.create_group("emg2qwerty")
        group.attrs["subject_id"] = "user_2"
        group.attrs["session_id"] = "typing_1"
        group.attrs["keystrokes_json"] = json.dumps(
            [{"key": "a", "press_s": 20.0005, "release_s": 20.001}]
        )
        group.attrs["prompts_json"] = json.dumps(
            [{"text": "a", "start_s": 20.0, "end_s": 20.0015}]
        )
        group.create_dataset("timeseries", data=rows)
    return source_path
