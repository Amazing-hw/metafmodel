"""Regression coverage for standalone GNI converter safety guards."""

import importlib.util
import inspect
from pathlib import Path

import h5py
import numpy as np
import pytest


GNI_CONVERTERS = [
    "convert_gni_discrete_gestures",
    "convert_gni_handwriting",
    "convert_gni_wrist",
]

SEMG_H5_DIRECTORY = Path(__file__).resolve().parents[1] / "semg_h5"


def _load_converter(converter_name):
    script_path = (SEMG_H5_DIRECTORY / f"{converter_name}.py").resolve()
    spec = importlib.util.spec_from_file_location(f"standalone_{converter_name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("converter_name", GNI_CONVERTERS)
def test_gni_manifest_lock_is_os_released_advisory_lock(converter_name):
    converter = _load_converter(converter_name)
    source = inspect.getsource(converter)

    assert "msvcrt.locking" in source
    assert "O_EXCL" not in source


def _write_gni_source(path, task, *, target_fields=(), sample_count=1):
    fields = [("emg", np.float32, (16,)), ("time", np.float64)] + list(target_fields)
    rows = np.zeros(sample_count, dtype=np.dtype(fields))
    with h5py.File(path, "w") as h5:
        data = h5.create_dataset("data", data=rows)
        data.attrs["task"] = task


@pytest.mark.parametrize("converter_name", GNI_CONVERTERS)
def test_gni_converter_rejects_empty_record_before_time_endpoint_access(tmp_path, converter_name):
    source = tmp_path / "empty.h5"
    task = {"convert_gni_discrete_gestures": "discrete_gestures", "convert_gni_handwriting": "handwriting", "convert_gni_wrist": "wrist"}[converter_name]
    _write_gni_source(source, task, target_fields=[("flexion_extension", np.float32)] if task == "wrist" else (), sample_count=0)

    converter = _load_converter(converter_name)
    with pytest.raises(ValueError, match="at least one sample"):
        converter.convert_file(source, tmp_path / "out")


@pytest.mark.parametrize("converter_name", GNI_CONVERTERS)
def test_gni_converter_rejects_output_root_nested_in_input_root(tmp_path, converter_name):
    input_root = tmp_path / "input"
    input_root.mkdir()
    converter = _load_converter(converter_name)

    with pytest.raises(ValueError, match="output-root.*input-root"):
        converter.run_conversion(input_root, input_root / "converted")


def test_wrist_converter_rejects_sanitized_target_name_collision(tmp_path):
    source = tmp_path / "collision.h5"
    _write_gni_source(source, "wrist", target_fields=[("a/b", np.float32), ("a?b", np.float32)])

    converter = _load_converter("convert_gni_wrist")
    with pytest.raises(ValueError, match="collision"):
        converter.convert_file(source, tmp_path / "out")


@pytest.mark.parametrize("converter_name", GNI_CONVERTERS)
def test_gni_h5_replace_failure_preserves_existing_output_and_cleans_temp(tmp_path, monkeypatch, converter_name):
    source = tmp_path / "source.h5"
    task = {"convert_gni_discrete_gestures": "discrete_gestures", "convert_gni_handwriting": "handwriting", "convert_gni_wrist": "wrist"}[converter_name]
    _write_gni_source(source, task, target_fields=[("flexion_extension", np.float32)] if task == "wrist" else ())
    converter = _load_converter(converter_name)
    output_root = tmp_path / "out"
    row = converter.convert_file(source, output_root)
    output = output_root / row["path"]
    original = output.read_bytes()

    def fail_replace(source_path, destination_path):
        raise OSError("injected replace failure")

    monkeypatch.setattr(converter.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        converter.convert_file(source, output_root, overwrite=True)

    assert output.read_bytes() == original
    assert not list(output.parent.glob("*.tmp"))


@pytest.mark.parametrize("converter_name", GNI_CONVERTERS)
def test_gni_manifest_replace_failure_cleans_temp(tmp_path, monkeypatch, converter_name):
    converter = _load_converter(converter_name)
    manifest = tmp_path / "manifest.csv"
    row = {column: f"value-{column}" for column in converter.MANIFEST_COLUMNS}
    row["path"] = "record.h5"
    append = getattr(converter, "_append_manifest_row", None) or converter._append

    def fail_replace(source_path, destination_path):
        raise OSError("injected manifest replace failure")

    monkeypatch.setattr(converter.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected manifest replace failure"):
        append(manifest, row)

    assert not list(tmp_path.glob("*.tmp"))
