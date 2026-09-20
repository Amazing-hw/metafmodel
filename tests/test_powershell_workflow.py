from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "scripts" / "powershell"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")

pytestmark = pytest.mark.skipif(POWERSHELL is None, reason="PowerShell is unavailable")


def run_script(
    name: str, *arguments: str, include_split: bool = True
) -> subprocess.CompletedProcess[str]:
    split_arguments = []
    if include_split and name in {
        "03_SmokeTest.ps1", "04_TrainBaseline.ps1", "05_Pretrain.ps1",
        "06_Finetune.ps1", "07_Evaluate.ps1", "08_Robustness.ps1",
    }:
        split_arguments = [
            "-SplitsPath", r"D:\Fmodel\emg_data_canonical\splits.csv"
        ]
    return subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT_ROOT / name),
            *arguments,
            *split_arguments,
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )


def test_preflight_script_inventory():
    required = [
        "Common.ps1",
        "00_Preflight.ps1",
        "01_AuditCanonical.ps1",
        "02_PrepareSplits.ps1",
        "03_SmokeTest.ps1",
        "04_TrainBaseline.ps1",
        "05_Pretrain.ps1",
        "06_Finetune.ps1",
        "07_Evaluate.ps1",
        "08_Robustness.ps1",
        "RunPipeline.ps1",
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
        encoding="utf-8",
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


def test_all_task_smoke_dry_run_is_bounded_and_windows_safe():
    result = run_script("03_SmokeTest.ps1", "-Task", "All", "-DryRun")
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count("finetune.py") == 5
    for task in ["pose", "wrist_angle", "discrete_gesture", "handwriting", "typing"]:
        assert f"task={task}" in output
    assert output.count("trainer=cpu_smoke") == 5
    assert output.count("data.num_workers=0") == 5


@pytest.mark.parametrize("model", ["supervised_tcn", "bendr", "cbramod", "aemg"])
def test_baseline_finetune_dry_run_supports_declared_models(model: str):
    result = run_script(
        "04_TrainBaseline.ps1",
        "-Model",
        model,
        "-Stage",
        "finetune",
        "-Task",
        "pose",
        "-DryRun",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "run_baseline.py" in output
    assert f"model={model}" in output
    assert "stage=finetune" in output


def test_supervised_tcn_pretrain_is_rejected():
    result = run_script(
        "04_TrainBaseline.ps1",
        "-Model",
        "supervised_tcn",
        "-Stage",
        "pretrain",
        "-DryRun",
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "supervised_tcn" in output
    assert "pretrain" in output


@pytest.mark.parametrize(
    ("mode", "extra", "expected_steps"),
    [
        ("Pilot", [], "optimizer.max_steps=200"),
        ("Full", ["-MaxSteps", "100000"], "optimizer.max_steps=100000"),
        (
            "Resume",
            [
                "-MaxSteps",
                "100000",
                "-ResumeFrom",
                r"D:\Fmodel\runs\pretrain\checkpoints\last.ckpt",
            ],
            "optimizer.max_steps=100000",
        ),
    ],
)
def test_pretrain_modes_build_bounded_commands(
    mode: str, extra: list[str], expected_steps: str
):
    result = run_script(
        "05_Pretrain.ps1",
        "-Mode",
        mode,
        "-Devices",
        "1",
        "-AccumulateGradBatches",
        "2",
        "-BatchSize",
        "8",
        *extra,
        "-DryRun",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "pretrain.py" in output
    assert "global_batch_size=16" in output
    assert expected_steps in output
    if mode == "Resume":
        assert "resume_from=" in output


def test_full_pretrain_requires_explicit_max_steps():
    result = run_script("05_Pretrain.ps1", "-Mode", "Full", "-DryRun")
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "MaxSteps" in output


def test_resume_pretrain_requires_checkpoint():
    result = run_script(
        "05_Pretrain.ps1",
        "-Mode",
        "Resume",
        "-MaxSteps",
        "100000",
        "-DryRun",
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "ResumeFrom" in output


def test_finetune_dry_run_supports_probe_and_label_budget():
    result = run_script(
        "06_Finetune.ps1", "-Task", "pose", "-FinetuneMode", "probe",
        "-LabelBudget", "0.1", "-PretrainedCheckpoint",
        r"D:\Fmodel\runs\pretrain\checkpoints\last.ckpt", "-DryRun",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "finetune.py" in output
    assert "finetune_mode=probe" in output
    assert "label_budget=0.1" in output


def test_evaluation_dry_run_requires_checkpoint_and_keeps_output_explicit():
    result = run_script(
        "07_Evaluate.ps1", "-Task", "pose", "-CheckpointPath",
        r"D:\Fmodel\runs\finetune\pose\checkpoints\last.ckpt", "-DryRun",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "evaluate.py" in output
    assert "checkpoint_path=" in output
    assert "output_dir=" in output


def test_robustness_all_dry_run_uses_separate_condition_outputs():
    result = run_script(
        "08_Robustness.ps1", "-Task", "pose", "-Condition", "All",
        "-CheckpointPath", r"D:\Fmodel\runs\finetune\pose\checkpoints\last.ckpt",
        "-DryRun",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    for condition in [
        "none", "channel_rotation", "channel_dropout", "gain_drift",
        "additive_noise", "temporal_loss",
    ]:
        assert f"robustness={condition}" in output
        assert rf"robustness\custom-" in output
        assert rf"\pose\{condition}" in output


def test_pipeline_requires_explicit_safe_stage_list_and_forwards_dry_run():
    result = run_script(
        "RunPipeline.ps1", "-Stages", "Preflight,Audit,Splits,Smoke",
        "-Protocol", "Official", "-SplitSeed", "20260916", "-DryRun",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count("Stage:") == 4
    assert "pretrain.py" not in output
    assert "run_baseline.py" not in output
    assert "splits.official.seed-20260916.csv" in output


def test_protocol_selected_split_reaches_every_smoke_task(tmp_path: Path):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    selected = canonical / "splits.strict-subject.seed-17.csv"
    selected.write_text("path\n", encoding="utf-8")
    result = run_script(
        "03_SmokeTest.ps1", "-CanonicalRoot", str(canonical),
        "-Protocol", "StrictSubject", "-SplitSeed", "17", "-DryRun",
        include_split=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count(selected.name) == 5


def test_training_script_rejects_missing_split_selection():
    result = run_script("03_SmokeTest.ps1", "-DryRun", include_split=False)
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "split selection" in output


def test_pretrain_protocol_selected_split_is_forwarded(tmp_path: Path):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    selected = canonical / "splits.official.seed-17.csv"
    selected.write_text("path\n", encoding="utf-8")
    result = run_script(
        "05_Pretrain.ps1", "-CanonicalRoot", str(canonical),
        "-Protocol", "Official", "-SplitSeed", "17", "-DryRun",
        include_split=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert selected.name in output


def test_existing_output_requires_force(tmp_path: Path):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    selected = canonical / "splits.official.seed-17.csv"
    selected.write_text("path\n", encoding="utf-8")
    runs = tmp_path / "runs"
    (runs / "smoke" / "official-seed-17" / "pose").mkdir(parents=True)

    result = run_script(
        "03_SmokeTest.ps1", "-CanonicalRoot", str(canonical),
        "-RunsRoot", str(runs), "-Task", "pose", "-Protocol", "Official",
        "-SplitSeed", "17", "-DryRun", include_split=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "output directory already exists" in output

    forced = run_script(
        "03_SmokeTest.ps1", "-CanonicalRoot", str(canonical),
        "-RunsRoot", str(runs), "-Task", "pose", "-Protocol", "Official",
        "-SplitSeed", "17", "-Force", "-DryRun", include_split=False,
    )
    assert forced.returncode == 0, forced.stdout + forced.stderr


def test_pipeline_forwards_selected_split_from_splits_stage_in_dry_run(tmp_path: Path):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "manifest.csv").write_text(
        "dataset_id,subject_id,official_split\nexample,subject,train\n",
        encoding="utf-8",
    )
    result = run_script(
        "RunPipeline.ps1", "-CanonicalRoot", str(canonical),
        "-RunsRoot", str(tmp_path / "runs"), "-Stages", "Splits,Smoke",
        "-Protocol", "StrictSubject", "-SplitSeed", "17", "-DryRun",
        include_split=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count("splits.strict-subject.seed-17.csv") >= 2
