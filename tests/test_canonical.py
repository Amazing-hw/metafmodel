import numpy as np
import pytest
import h5py

from semg_h5.canonical import CanonicalRecord, validate_record


def test_validate_record_accepts_a_16_channel_monotonic_stream():
    record = CanonicalRecord(
        dataset_id="emg2pose",
        task_id="pose",
        subject_id="emg2pose:user_1",
        session_id="emg2pose:session_1",
        side="left",
        pair_id="",
        official_split="train",
        source_file="source.h5",
        emg=np.zeros((4, 16)),
        time_s=np.array([0.0, 0.0005, 0.001, 0.0015]),
        labels={},
        source_meta={},
        preprocessing={"converter_transformations": []},
    )
    validate_record(record)


def test_validate_record_rejects_a_non_16_channel_stream():
    record = CanonicalRecord.minimal(
        emg=np.zeros((4, 15)), time_s=np.arange(4) / 2000
    )
    with pytest.raises(ValueError, match="16 channels"):
        validate_record(record)


def test_write_record_creates_required_groups_and_relative_clock(tmp_path):
    from semg_h5.canonical import read_canonical_header, write_record

    record = CanonicalRecord.minimal(
        emg=np.arange(64, dtype=np.float64).reshape(4, 16),
        time_s=np.array([10.0, 10.0005, 10.001, 10.0015]),
    )
    output = tmp_path / "record.h5"
    write_record(output, record)

    header = read_canonical_header(output)
    assert header["schema_version"] == "1.0"
    assert header["emg_shape"] == (4, 16)
    assert np.allclose(header["time_s"], [0.0, 0.0005, 0.001, 0.0015])


def test_typing_labels_round_trip_as_structured_tables(tmp_path):
    from semg_h5.canonical import write_record

    record = CanonicalRecord.minimal(np.zeros((4, 16)), np.arange(4) / 2000)
    record.task_id = "typing"
    record.labels = {
        "keystrokes": [{"key": "a", "press_s": 0.0005, "release_s": 0.001}],
        "prompts": [{"text": "a", "start_s": 0.0, "end_s": 0.0015}],
    }
    output = tmp_path / "typing.h5"
    write_record(output, record)

    with h5py.File(output) as h5:
        assert h5["labels/keystrokes"][0]["key"].decode() == "a"
        assert h5["labels/prompts"][0]["text"].decode() == "a"


def test_tree_validator_rejects_nan_signal_values(tmp_path):
    from semg_h5.canonical import write_record
    from semg_h5.validate import validate_tree

    output = tmp_path / "record.h5"
    write_record(
        output,
        CanonicalRecord.minimal(np.zeros((4, 16)), np.arange(4) / 2000),
    )
    with h5py.File(output, "r+") as h5:
        h5["signal/emg"][0, 0] = np.nan

    assert any("finite" in error for error in validate_tree(output)[output])
