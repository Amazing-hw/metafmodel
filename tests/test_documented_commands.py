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
            "docs/POST_CONVERSION_EXECUTION_GUIDE.md",
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
        "docs/POST_CONVERSION_EXECUTION_GUIDE.md",
    ]:
        assert (root / name).is_file(), f"missing {name}"


def test_workflow_documents_all_powershell_stages():
    root = Path(__file__).parents[1]
    text = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in [
            "docs/META16_TRAINING_WORKFLOW.md",
            "docs/POWERSHELL_SCRIPT_REFERENCE.md",
            "docs/POST_CONVERSION_EXECUTION_GUIDE.md",
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
    for expected in ["Official", "StrictSubject", "Pilot", "100,000", "不会自动运行"]:
        assert expected in text


def test_public_workflow_documents_are_chinese():
    root = Path(__file__).parents[1]
    expected = {
        "README.md": "Meta 16 通道 sEMG 基础模型训练仓库",
        "docs/META16_TRAINING_WORKFLOW.md": "Meta16 完整训练流程",
        "docs/POWERSHELL_SCRIPT_REFERENCE.md": "PowerShell 脚本参考",
        "docs/POST_CONVERSION_EXECUTION_GUIDE.md": "五数据集转换完成后的完整执行指南",
    }
    for name, title in expected.items():
        assert title in (root / name).read_text(encoding="utf-8")


def test_chinese_execution_guide_covers_every_operational_stage():
    root = Path(__file__).parents[1]
    text = (root / "docs/POST_CONVERSION_EXECUTION_GUIDE.md").read_text(
        encoding="utf-8"
    )
    for section in [
        "环境预检", "数据审计", "生成切分", "CPU 烟雾测试", "预训练",
        "微调", "基线", "评估", "鲁棒性", "失败恢复",
    ]:
        assert section in text
