"""独立 Meta 转换器安全防护与数据溯源的回归测试。"""

import csv
import importlib.util
import inspect
import json
from pathlib import Path

import h5py
import numpy as np
import pytest


SEMG_H5_DIRECTORY = Path(__file__).resolve().parents[1] / "semg_h5"
META_CONVERTERS = ["convert_emg2pose", "convert_emg2qwerty"]


def _load_converter(name):
    path = (SEMG_H5_DIRECTORY / f"{name}.py").resolve()
    spec = importlib.util.spec_from_file_location(f"standalone_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_pose(path, *, sample_count=2, joint_angles=None, no_ik_failure=None):
    dtype = [("emg", np.float32, (16,)), ("joint_angles", np.float32, (3,)), ("time", np.float64)]
    if no_ik_failure is not None:
        dtype.append(("no_ik_failure", np.bool_))
    rows = np.zeros(sample_count, dtype=np.dtype(dtype))
    rows["time"] = np.arange(sample_count) / 2000 + 11.0
    if joint_angles is not None:
        rows["joint_angles"] = joint_angles
    if no_ik_failure is not None:
        rows["no_ik_failure"] = no_ik_failure
    with h5py.File(path, "w") as h5:
        group = h5.create_group("emg2pose")
        group.attrs.update({"subject_id": "user", "session_id": "pose", "side": "left"})
        group.create_dataset("timeseries", data=rows)


def _write_qwerty(path, *, sample_count=2, keystrokes=None, prompts=None):
    rows = np.zeros(sample_count, dtype=np.dtype([
        ("emg_left", np.float32, (16,)), ("emg_right", np.float32, (16,)), ("time", np.float64),
    ]))
    rows["emg_left"] = 1
    rows["emg_right"] = 2
    rows["time"] = np.arange(sample_count) / 2000 + 21.0
    with h5py.File(path, "w") as h5:
        group = h5.create_group("emg2qwerty")
        group.attrs.update({"subject_id": "user", "session_id": "typing"})
        if keystrokes is not None:
            group.attrs["keystrokes_json"] = json.dumps(keystrokes)
        if prompts is not None:
            group.attrs["prompts_json"] = json.dumps(prompts)
        group.create_dataset("timeseries", data=rows)


@pytest.mark.parametrize("name", META_CONVERTERS)
def test_meta_converters_reject_empty_records_before_time_endpoint_access(tmp_path, name):
    source = tmp_path / "empty.h5"
    if name == "convert_emg2pose":
        _write_pose(source, sample_count=0)
    else:
        _write_qwerty(source, sample_count=0, keystrokes=[], prompts=[])
    converter = _load_converter(name)

    with pytest.raises(ValueError, match="at least one sample"):
        converter.convert_file(source, tmp_path / "out")


@pytest.mark.parametrize("name", META_CONVERTERS)
def test_meta_converters_reject_output_root_nested_in_input_root(tmp_path, name):
    input_root = tmp_path / "input"
    input_root.mkdir()
    converter = _load_converter(name)

    with pytest.raises(ValueError, match="output-root.*input-root"):
        converter.run_conversion(input_root, input_root / "out")


@pytest.mark.parametrize("name", META_CONVERTERS)
def test_meta_converters_use_os_released_manifest_locks(name):
    source = inspect.getsource(_load_converter(name))
    assert "msvcrt.locking" in source
    assert "O_EXCL" not in source


@pytest.mark.parametrize("name", META_CONVERTERS)
def test_meta_h5_replace_failure_preserves_existing_output_and_cleans_temp(tmp_path, monkeypatch, name):
    source = tmp_path / "source.h5"
    if name == "convert_emg2pose":
        _write_pose(source)
    else:
        _write_qwerty(source, keystrokes=[], prompts=[])
    converter = _load_converter(name)
    output_root = tmp_path / "out"
    rows = converter.convert_file(source, output_root)
    row = rows[0] if isinstance(rows, list) else rows
    output = output_root / row["path"]
    original = output.read_bytes()

    monkeypatch.setattr(converter.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("replace failed")))
    with pytest.raises(OSError, match="replace failed"):
        converter.convert_file(source, output_root, overwrite=True)

    assert output.read_bytes() == original
    assert not list(output.parent.glob("*.tmp"))


@pytest.mark.parametrize("name", META_CONVERTERS)
def test_meta_manifest_replace_failure_cleans_unique_temp(tmp_path, monkeypatch, name):
    converter = _load_converter(name)
    row = {column: f"value-{column}" for column in converter.MANIFEST_COLUMNS}
    row["path"] = "record.h5"
    monkeypatch.setattr(converter.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("manifest failed")))

    with pytest.raises(OSError, match="manifest failed"):
        converter._append_manifest_rows(tmp_path / "manifest.csv", [row])

    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("name", META_CONVERTERS)
def test_run_conversion_records_input_relative_source_and_time_origin(tmp_path, name):
    input_root = tmp_path / "input"
    source = input_root / "nested" / "session.h5"
    source.parent.mkdir(parents=True)
    if name == "convert_emg2pose":
        _write_pose(source)
    else:
        _write_qwerty(source, keystrokes=[{"key": "a", "press_s": 21.0, "release_s": 21.001}], prompts=[{"text": "a", "start_s": 21.0, "end_s": 21.001}])
    converter = _load_converter(name)
    output_root = tmp_path / "out"
    rows = converter.run_conversion(input_root, output_root)
    assert rows
    assert {row["source_file"] for row in rows} == {"nested/session.h5"}
    with (output_root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        assert {row["source_file"] for row in csv.DictReader(handle)} == {"nested/session.h5"}
    for row in rows:
        with h5py.File(output_root / row["path"], "r") as h5:
            assert h5.attrs["source_file"] == "nested/session.h5"
            meta = json.loads(h5["meta/source_json"].asstr()[()])
            assert meta["source_time_origin_s"] == pytest.approx(11.0 if name == "convert_emg2pose" else 21.0)
            if name == "convert_emg2qwerty":
                assert meta["source_label_rows"]["keystrokes"][0]["press_s"] == pytest.approx(21.0)


def test_pose_rejects_nonfinite_joint_angles_and_misaligned_optional_label(tmp_path):
    converter = _load_converter("convert_emg2pose")
    source = tmp_path / "pose.h5"
    _write_pose(source, joint_angles=np.array([[np.nan, 0, 0], [0, 0, 0]], dtype=np.float32))
    with pytest.raises(ValueError, match="joint_angles.*finite"):
        converter.convert_file(source, tmp_path / "out")

    record = converter.CanonicalRecord(
        "emg2pose", "pose", "emg2pose:user", "emg2pose:pose", "left", "", "unknown",
        "pose.h5", str(source), np.zeros((2, 16), dtype=np.float32), np.array([0.0, 0.1]),
        {"joint_angles": np.zeros((2, 3), dtype=np.float32), "no_ik_failure": np.array([True])},
        {"source_time_origin_s": 0.0}, {"converter_transformations": []},
    )
    with pytest.raises(ValueError, match="no_ik_failure.*align"):
        converter._write_record(tmp_path / "out" / "bad.h5", record)


@pytest.mark.parametrize("keystrokes,prompts,match", [
    (None, [], "keystrokes"),
    ([{"key": "a", "press_s": 21.002, "release_s": 21.001}], [], "press.*release"),
    ([{"key": "a", "press_s": float("nan"), "release_s": 21.001}], [], "finite"),
    ([], [{"text": "a", "start_s": 21.002, "end_s": 21.001}], "start.*end"),
])
def test_qwerty_requires_and_validates_typing_labels(tmp_path, keystrokes, prompts, match):
    source = tmp_path / "typing.h5"
    _write_qwerty(source, keystrokes=keystrokes, prompts=prompts)
    converter = _load_converter("convert_emg2qwerty")

    with pytest.raises(ValueError, match=match):
        converter.convert_file(source, tmp_path / "out")


def test_qwerty_pair_commit_rolls_back_first_replacement_on_second_failure(tmp_path, monkeypatch):
    source = tmp_path / "typing.h5"
    _write_qwerty(source, keystrokes=[], prompts=[])
    converter = _load_converter("convert_emg2qwerty")
    output_root = tmp_path / "out"
    rows = converter.convert_file(source, output_root)
    outputs = [output_root / row["path"] for row in rows]
    originals = [output.read_bytes() for output in outputs]
    original_replace = converter.os.replace
    final_replacements = 0

    def fail_second_final_replace(source_path, destination_path):
        nonlocal final_replacements
        if Path(destination_path) in outputs and Path(source_path).suffix == ".tmp":
            final_replacements += 1
            if final_replacements == 2:
                raise OSError("second pair replacement failed")
        return original_replace(source_path, destination_path)

    monkeypatch.setattr(converter.os, "replace", fail_second_final_replace)
    with pytest.raises(OSError, match="second pair replacement failed"):
        converter.convert_file(source, output_root, overwrite=True)

    assert [output.read_bytes() for output in outputs] == originals
    assert not list(outputs[0].parent.glob("*.tmp"))


def test_qwerty_reader_accepts_the_released_session_layout(tmp_path):
    """The release spells identity as user/session_name and key times as start/end.

    The converter was written against the design document's schema
    (subject_id/session_id, press_s/release_s, a plain-text prompts table), so
    every real session failed to read and the typing target was unavailable.
    """
    source = tmp_path / "2020-08-13-1597354281-keystrokes.hdf5"
    dtype = np.dtype(
        [("emg_left", np.float32, (16,)), ("emg_right", np.float32, (16,)), ("time", np.float64)]
    )
    rows = np.zeros(4, dtype=dtype)
    rows["time"] = np.arange(4) / 2000 + 100.0
    with h5py.File(source, "w") as h5:
        group = h5.create_group("emg2qwerty")
        group.attrs["user"] = "09456349"
        group.attrs["session_name"] = "2020-08-13-1597354281-keystrokes"
        group.attrs["keystrokes"] = json.dumps(
            [{"ascii": 97, "key": "a", "start": 100.0005, "end": 100.001}]
        )
        group.attrs["prompts"] = json.dumps(
            [
                {"name": "emg_left", "start": 100.0, "end": 100.004,
                 "payload": None, "payload_type": None},
                {"name": "text_prompt", "start": 100.0, "end": 100.004,
                 "payload": {"prompt_num": 0, "text": "a"}, "payload_type": "text_prompt"},
            ]
        )
        group.create_dataset("timeseries", data=rows)
    converter = _load_converter("convert_emg2qwerty")

    records = converter.read_emg2qwerty(source)

    assert len(records) == 2
    assert {record.side for record in records} == {"left", "right"}
    left = next(record for record in records if record.side == "left")
    assert left.subject_id == "emg2qwerty:09456349"
    assert left.session_id == "emg2qwerty:2020-08-13-1597354281-keystrokes"
    assert left.labels["keystrokes"] == [
        {"key": "a", "press_s": pytest.approx(0.0005), "release_s": pytest.approx(0.001)}
    ]
    # Only the text_prompt segment becomes a prompt; emg_left stays in provenance.
    assert [prompt["text"] for prompt in left.labels["prompts"]] == ["a"]


def test_qwerty_reader_rejects_a_session_with_no_text_prompts(tmp_path):
    source = tmp_path / "no-text.hdf5"
    dtype = np.dtype(
        [("emg_left", np.float32, (16,)), ("emg_right", np.float32, (16,)), ("time", np.float64)]
    )
    rows = np.zeros(4, dtype=dtype)
    with h5py.File(source, "w") as h5:
        group = h5.create_group("emg2qwerty")
        group.attrs["user"] = "u1"
        group.attrs["keystrokes"] = json.dumps([{"key": "a", "start": 0.0005, "end": 0.001}])
        group.attrs["prompts"] = json.dumps(
            [{"name": "emg_left", "start": 0.0, "end": 0.004, "payload": None, "payload_type": None}]
        )
        group.create_dataset("timeseries", data=rows)
    converter = _load_converter("convert_emg2qwerty")

    with pytest.raises(ValueError, match="none carry text"):
        converter.read_emg2qwerty(source)
