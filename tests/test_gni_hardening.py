"""Regression coverage for standalone GNI converter safety guards."""

import importlib

import h5py
import numpy as np
import pytest


GNI_CONVERTERS = [
    "convert_gni_discrete_gestures",
    "convert_gni_handwriting",
    "convert_gni_wrist",
]


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

    converter = importlib.import_module(f"semg_h5.{converter_name}")
    with pytest.raises(ValueError, match="at least one sample"):
        converter.convert_file(source, tmp_path / "out")


@pytest.mark.parametrize("converter_name", GNI_CONVERTERS)
def test_gni_converter_rejects_output_root_nested_in_input_root(tmp_path, converter_name):
    input_root = tmp_path / "input"
    input_root.mkdir()
    converter = importlib.import_module(f"semg_h5.{converter_name}")

    with pytest.raises(ValueError, match="output-root.*input-root"):
        converter.run_conversion(input_root, input_root / "converted")


def test_wrist_converter_rejects_sanitized_target_name_collision(tmp_path):
    source = tmp_path / "collision.h5"
    _write_gni_source(source, "wrist", target_fields=[("a/b", np.float32), ("a?b", np.float32)])

    converter = importlib.import_module("semg_h5.convert_gni_wrist")
    with pytest.raises(ValueError, match="collision"):
        converter.convert_file(source, tmp_path / "out")
