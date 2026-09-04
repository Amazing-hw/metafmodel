import subprocess
import sys

import pandas as pd


def test_converter_cli_writes_a_valid_tree_and_manifest(tmp_path, qwerty_h5):
    output_root = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "semg_h5.convert_emg2qwerty",
            "--input-root",
            str(qwerty_h5.parent),
            "--output-root",
            str(output_root),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    validation = subprocess.run(
        [sys.executable, "-m", "semg_h5.validate", "--input", str(output_root)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert validation.returncode == 0, validation.stderr
    assert len(pd.read_csv(output_root / "manifest.csv")) == 2
