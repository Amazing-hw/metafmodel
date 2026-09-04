import numpy as np
import pytest

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
