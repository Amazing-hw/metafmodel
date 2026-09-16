"""The standalone public documents must name every public entry point."""

from __future__ import annotations

from pathlib import Path


def test_public_documents_include_all_entry_points():
    root = Path(__file__).parents[1]
    text = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in [
            "README.md",
            "docs/META16_TRAINING_WORKFLOW.md",
            "docs/POWERSHELL_SCRIPT_REFERENCE.md",
        ]
    )
    for command in [
        "pretrain.py", "finetune.py", "evaluate.py", "run_baseline.py",
        "make_splits.py", "audit_canonical.py",
    ]:
        assert command in text


def test_training_stack_documents_exist():
    root = Path(__file__).parents[1]
    for name in [
        "README.md",
        "docs/META16_TRAINING_WORKFLOW.md",
        "docs/POWERSHELL_SCRIPT_REFERENCE.md",
    ]:
        assert (root / name).is_file(), f"missing {name}"


def test_workflow_documents_all_powershell_stages():
    root = Path(__file__).parents[1]
    text = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in [
            "docs/META16_TRAINING_WORKFLOW.md",
            "docs/POWERSHELL_SCRIPT_REFERENCE.md",
        ]
    )
    for script in [
        "00_Preflight.ps1", "01_AuditCanonical.ps1", "02_PrepareSplits.ps1",
        "03_SmokeTest.ps1", "04_TrainBaseline.ps1", "05_Pretrain.ps1",
        "06_Finetune.ps1", "07_Evaluate.ps1", "08_Robustness.ps1",
        "RunPipeline.ps1",
    ]:
        assert script in text


def test_workflow_documents_state_protocol_and_completion_limits():
    root = Path(__file__).parents[1]
    text = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in [
            "docs/META16_TRAINING_WORKFLOW.md",
            "docs/POWERSHELL_SCRIPT_REFERENCE.md",
        ]
    )
    for expected in ["Official", "StrictSubject", "Pilot", "100,000", "not run"]:
        assert expected in text
