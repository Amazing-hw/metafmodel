import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from semg_fm.data.window_dataset import CanonicalWindowDataset


def _task_manifest(canonical_meta_root, task_id):
    manifest = pd.read_csv(canonical_meta_root / "manifest.csv", keep_default_na=False)
    path = canonical_meta_root / f"{task_id}_manifest.csv"
    manifest.loc[manifest.task_id == task_id].to_csv(path, index=False)
    return path


@pytest.fixture
def pose_dataset(canonical_meta_root):
    return CanonicalWindowDataset.from_manifest(
        _task_manifest(canonical_meta_root, "pose"), split="train", window_ms=200,
        stride_ms=100, root=canonical_meta_root,
    )


@pytest.fixture
def handwriting_dataset(canonical_meta_root):
    return CanonicalWindowDataset.from_manifest(
        _task_manifest(canonical_meta_root, "handwriting"), split="train", window_ms=200,
        stride_ms=100, root=canonical_meta_root,
    )


def test_window_uses_milliseconds_and_returns_time_major_emg(canonical_meta_root):
    ds = CanonicalWindowDataset.from_manifest(
        canonical_meta_root / "manifest.csv", split="train", window_ms=200, stride_ms=100
    )
    item = ds[0]
    assert item["emg"].ndim == 2
    assert item["emg"].shape[1] == 16
    assert item["emg"].shape[0] == 400
    assert item["emg"].dtype == torch.float32
    assert item["sample_mask"].dtype == torch.bool
    assert item["sample_mask"].all()


def test_dataset_construction_does_not_open_h5_files(canonical_meta_root, monkeypatch):
    real_file = h5py.File
    calls = []

    def observed_file(*args, **kwargs):
        calls.append(args[0])
        return real_file(*args, **kwargs)

    monkeypatch.setattr("semg_fm.data.window_dataset.h5py.File", observed_file)
    ds = CanonicalWindowDataset.from_manifest(
        canonical_meta_root / "manifest.csv", split="train", window_ms=200, stride_ms=100
    )
    assert calls == []
    ds[0]
    assert len(calls) == 1


def test_pose_invalid_ik_frames_become_target_mask(pose_dataset):
    item = pose_dataset[0]
    assert item["target"].shape[0] == item["emg"].shape[0]
    assert item["target_mask"].dtype == torch.bool
    assert not item["target_mask"][3]


def test_handwriting_returns_prompt_text_without_character_times(handwriting_dataset):
    item = handwriting_dataset[0]
    assert item["target_text"] == "hello"
    assert "character_times" not in item


def test_gesture_events_are_relative_to_window_start(canonical_meta_root):
    ds = CanonicalWindowDataset.from_manifest(
        _task_manifest(canonical_meta_root, "discrete_gesture"), split="train", window_ms=100,
        stride_ms=100, root=canonical_meta_root,
    )
    assert ds[1]["events"] == [("fist", pytest.approx(0.05))]


def test_label_times_follow_the_record_clock_at_a_window_boundary(canonical_meta_root):
    path = canonical_meta_root / "discrete_gesture" / "discrete_gesture_right.h5"
    with h5py.File(path, "r+") as h5:
        time_s = h5["signal/time_s"][:]
        time_s[200:] += 0.01
        h5["signal/time_s"][:] = time_s
    ds = CanonicalWindowDataset.from_manifest(
        _task_manifest(canonical_meta_root, "discrete_gesture"), split="train", window_ms=100,
        stride_ms=100, root=canonical_meta_root,
    )
    item = ds[1]
    assert item["start_s"] == pytest.approx(0.11)
    assert item["events"] == [("fist", pytest.approx(0.04))]


@pytest.mark.parametrize(
    ("dataset_name", "dtype", "expected"),
    [("emg", np.float64, "float32"), ("time_s", np.float32, "float64")],
)
def test_dataset_rejects_noncanonical_signal_dtypes(canonical_meta_root, dataset_name, dtype, expected):
    path = canonical_meta_root / "pose" / "pose_right.h5"
    with h5py.File(path, "r+") as h5:
        values = h5[f"signal/{dataset_name}"][:].astype(dtype)
        del h5[f"signal/{dataset_name}"]
        h5["signal"].create_dataset(dataset_name, data=values)
    ds = CanonicalWindowDataset.from_manifest(
        _task_manifest(canonical_meta_root, "pose"), split="train", window_ms=200,
        stride_ms=100, root=canonical_meta_root,
    )
    with pytest.raises(ValueError, match=expected):
        ds[0]


def test_dataset_rejects_missing_or_mismatched_canonical_root_attributes(canonical_meta_root):
    path = canonical_meta_root / "pose" / "pose_right.h5"
    with h5py.File(path, "r+") as h5:
        del h5.attrs["dataset_id"]
    ds = CanonicalWindowDataset.from_manifest(
        _task_manifest(canonical_meta_root, "pose"), split="train", window_ms=200,
        stride_ms=100, root=canonical_meta_root,
    )
    with pytest.raises(ValueError, match="dataset_id"):
        ds[0]
    with h5py.File(path, "r+") as h5:
        h5.attrs["dataset_id"] = "not-canonical"
    with pytest.raises(ValueError, match="does not match manifest"):
        ds[0]


def test_gapped_record_clock_controls_the_open_window_end(canonical_meta_root):
    path = canonical_meta_root / "discrete_gesture" / "discrete_gesture_right.h5"
    with h5py.File(path, "r+") as h5:
        time_s = h5["signal/time_s"][:]
        time_s[201:] += np.arange(time_s.size - 201) * 0.0005
        h5["signal/time_s"][:] = time_s
        events = h5["labels/events"][:]
        events["time_s"] = 0.22
        h5["labels/events"][:] = events
    ds = CanonicalWindowDataset.from_manifest(
        _task_manifest(canonical_meta_root, "discrete_gesture"), split="train", window_ms=100,
        stride_ms=100, root=canonical_meta_root,
    )
    assert ds[1]["events"] == [("fist", pytest.approx(0.12))]


def test_typing_preserves_pair_id_for_each_wrist(canonical_meta_root):
    ds = CanonicalWindowDataset.from_manifest(
        _task_manifest(canonical_meta_root, "typing"), split="train", window_ms=100,
        stride_ms=100, root=canonical_meta_root,
    )
    items = [ds[0], ds[3]]
    assert {item["side"] for item in items} == {"left", "right"}
    assert {item["pair_id"] for item in items} == {"pair-1"}
    assert all("pair_positive" not in item for item in items)
    window = ds[1]
    assert window["keystrokes"] == [("a", pytest.approx(0.05), pytest.approx(0.051))]
    assert window["prompt_start_s"] == pytest.approx(-0.1)
    assert window["prompt_end_s"] == pytest.approx(0.1995)


def test_wrist_targets_are_name_keyed_and_aligned(canonical_meta_root):
    ds = CanonicalWindowDataset.from_manifest(
        _task_manifest(canonical_meta_root, "wrist_angle"), split="train", window_ms=200,
        stride_ms=100, root=canonical_meta_root,
    )
    item = ds[0]
    assert set(item["target"]) == {"flexion_extension", "pronation"}
    assert all(value.shape == (400,) for value in item["target"].values())
    assert all(value.dtype == torch.bool for value in item["target_mask"].values())
