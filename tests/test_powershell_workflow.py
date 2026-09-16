from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "scripts" / "powershell"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")

pytestmark = pytest.mark.skipif(POWERSHELL is None, reason="PowerShell is unavailable")


def run_script(name: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT_ROOT / name),
            *arguments,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_preflight_script_inventory():
    required = [
        "Common.ps1",
        "00_Preflight.ps1",
        "01_AuditCanonical.ps1",
        "02_PrepareSplits.ps1",
    ]
    assert [name for name in required if not (SCRIPT_ROOT / name).is_file()] == []


def test_all_powershell_scripts_parse():
    escaped = str(SCRIPT_ROOT).replace("'", "''")
    command = (
        f"$files=Get-ChildItem -LiteralPath '{escaped}' -Filter '*.ps1';"
        "$failed=$false;foreach($file in $files){"
        "$tokens=$null;$errors=$null;"
        "[System.Management.Automation.Language.Parser]::ParseFile("
        "$file.FullName,[ref]$tokens,[ref]$errors)>$null;"
        "if($errors.Count){$failed=$true;$errors|ForEach-Object Message}};"
        "if($failed){exit 1}"
    )
    result = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-Command", command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_preflight_dry_run_uses_only_d_drive_defaults():
    result = run_script("00_Preflight.ps1", "-DryRun")
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert r"D:\Fmodel\metafmodel" in output
    assert r"D:\Fmodel\emg_data_canonical" in output
    assert r"D:\Fmodel\runs" in output
    assert r"Documents\F_model" not in output
    assert "import torch, lightning, hydra" in output


def test_audit_dry_run_calls_read_only_python_auditor():
    result = run_script("01_AuditCanonical.ps1", "-DryRun", "-H5Samples", "5")
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "audit_canonical.py" in output
    assert "--h5-samples 5" in output
    assert r"D:\Fmodel\runs\audits" in output


@pytest.mark.parametrize(
    ("protocol", "expected_manifest", "expected_output"),
    [
        ("Official", "manifest.csv", "splits.official.seed-20260916.csv"),
        (
            "StrictSubject",
            "manifest.strict-subject.csv",
            "splits.strict-subject.seed-20260916.csv",
        ),
    ],
)
def test_split_dry_run_keeps_protocol_outputs_separate(
    protocol: str, expected_manifest: str, expected_output: str
):
    result = run_script(
        "02_PrepareSplits.ps1",
        "-DryRun",
        "-Protocol",
        protocol,
        "-Seed",
        "20260916",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert expected_manifest in output
    assert expected_output in output
    assert "make_splits.py" in output
