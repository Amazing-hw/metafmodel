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
    required = ["Common.ps1", "00_Preflight.ps1"]
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
