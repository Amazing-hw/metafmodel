"""Verify converter scripts can run independently of the ``semg_h5`` package."""

import ast
import importlib.util
from pathlib import Path

import h5py
import pytest


REPOSITORY_ROOT = Path("D:/Fmodel")
SEMG_H5_DIRECTORY = REPOSITORY_ROOT / "semg_h5"
CONVERTER_NAMES = [
    "convert_emg2pose.py",
    "convert_emg2qwerty.py",
    "convert_gni_discrete_gestures.py",
    "convert_gni_handwriting.py",
    "convert_gni_wrist.py",
]

CONVERTER_CASES = [
    ("convert_emg2pose.py", "emg2pose_h5", "pose", "labels/joint_angles"),
    ("convert_emg2qwerty.py", "qwerty_h5", "typing", "labels/keystrokes"),
    (
        "convert_gni_discrete_gestures.py",
        "discrete_gesture",
        "discrete_gesture",
        "labels/events",
    ),
    ("convert_gni_handwriting.py", "handwriting", "handwriting", "labels/prompts"),
    (
        "convert_gni_wrist.py",
        "wrist_angle",
        "wrist_angle",
        "labels/targets/flexion_extension",
    ),
]


def test_converter_scripts_are_the_expected_standalone_files_without_package_imports():
    converter_paths = sorted(SEMG_H5_DIRECTORY.glob("convert_*.py"))
    assert [path.name for path in converter_paths] == CONVERTER_NAMES

    violations = []
    for script_path in converter_paths:
        tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                node.level > 0 or (node.module or "").startswith("semg_h5")
            ):
                violations.append(f"{script_path.name}:{node.lineno}")
            elif isinstance(node, ast.Import) and any(
                alias.name.startswith("semg_h5") for alias in node.names
            ):
                violations.append(f"{script_path.name}:{node.lineno}")

    assert not violations, f"converter scripts must not import semg_h5 packages: {violations}"


def _source_path(request, fixture_name):
    if fixture_name in {"emg2pose_h5", "qwerty_h5"}:
        return request.getfixturevalue(fixture_name)
    return request.getfixturevalue("gni_h5_by_task")[fixture_name]


@pytest.mark.parametrize(
    ("converter_name", "fixture_name", "task_id", "label_path"), CONVERTER_CASES
)
def test_converter_script_executes_against_existing_fixtures(
    tmp_path, request, monkeypatch, converter_name, fixture_name, task_id, label_path
):
    script_path = (SEMG_H5_DIRECTORY / converter_name).resolve()
    monkeypatch.syspath_prepend(str(SEMG_H5_DIRECTORY))
    spec = importlib.util.spec_from_file_location(f"standalone_{script_path.stem}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.convert_file(_source_path(request, fixture_name), tmp_path / "out")
    manifest_rows = result if isinstance(result, list) else [result]
    for manifest_row in manifest_rows:
        with h5py.File(tmp_path / "out" / manifest_row["path"]) as h5:
            assert h5.attrs["task_id"] == task_id
            assert h5["signal/emg"].shape == (4, 16)
            assert label_path in h5
