import h5py
import pytest


def test_convert_qwerty_emits_one_linked_record_per_wrist(tmp_path, qwerty_h5):
    from semg_h5.convert_emg2qwerty import convert_file

    rows = convert_file(qwerty_h5, tmp_path / "out", official_split="train")
    assert len(rows) == 2
    assert {row["side"] for row in rows} == {"left", "right"}
    assert rows[0]["pair_id"] == rows[1]["pair_id"]
    for row in rows:
        with h5py.File(tmp_path / "out" / row["path"]) as h5:
            assert h5["signal/emg"].shape == (4, 16)
            assert h5["labels/keystrokes"].shape[0] == 1
            assert h5["labels/keystrokes"][0]["press_s"] == pytest.approx(0.0005)
