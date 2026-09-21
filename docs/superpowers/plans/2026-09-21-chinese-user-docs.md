# 用户文档全面中文化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将仓库全部面向使用者的入口与训练流程文档统一改写为中文，同时保持所有命令和程序标识可直接复制执行。

**Architecture:** 四份用户文档按“入口概览、阶段流程、脚本参考、完整运行手册”分工，不修改历史规格与计划。测试通过关键词、脚本清单、协议名和中文标题检查覆盖范围，现有 PowerShell 与全仓测试负责验证命令契约没有被文档改写破坏。

**Tech Stack:** Markdown、PowerShell 命令示例、pytest 文档契约测试、Git。

---

### Task 1: 增加中文文档契约

**Files:**
- Modify: `tests/test_documented_commands.py`

- [ ] **Step 1: 写入中文覆盖测试**

新增测试，读取四份用户文档并断言每份包含其中文主标题，同时断言完整指南包含全部阶段名称：

```python
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
```

再断言指南包含“环境预检、数据审计、生成切分、CPU 烟雾测试、预训练、微调、基线、评估、鲁棒性、失败恢复”。

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_documented_commands.py::test_public_workflow_documents_are_chinese -q`

Expected: FAIL，因为 README、训练流程和脚本参考尚未使用中文标题。

### Task 2: 中文化入口和阶段流程

**Files:**
- Modify: `README.md`
- Modify: `docs/META16_TRAINING_WORKFLOW.md`

- [ ] **Step 1: 改写 README**

使用以下中文结构：项目定位、默认目录、安装、推荐首次执行、详细文档入口。保留 `D:\Fmodel` 路径以及所有 PowerShell/Python 命令原文。

- [ ] **Step 2: 改写完整训练流程**

使用以下中文章节：语料与切分协议、阶段门槛、环境预检、数据审计、生成切分、五任务烟雾测试、比较基线、CIRCE-Myo 预训练、下游微调与评估、鲁棒性评估。写入当前审计数据：28,609 条记录、601 个 subject、14,893 个 session、1,396.474 小时。

- [ ] **Step 3: 运行中文契约与路径契约测试**

Run: `python -m pytest tests/test_documented_commands.py tests/test_repository_contract.py -q`

Expected: README 与训练流程相关断言通过；不存在退役工作目录。

### Task 3: 中文化脚本参考与完整执行指南

**Files:**
- Modify: `docs/POWERSHELL_SCRIPT_REFERENCE.md`
- Modify: `docs/POST_CONVERSION_EXECUTION_GUIDE.md`

- [ ] **Step 1: 改写脚本参考**

表格列名统一为“脚本、用途、关键参数、输出与失败行为”。逐项覆盖 `00_Preflight.ps1` 至 `08_Robustness.ps1` 和 `RunPipeline.ps1`，保留参数名与枚举值原文。

- [ ] **Step 2: 清理并补强完整执行指南**

删除“emg2pose 仍在转换”的过时描述。按照当前状态写明两套 split 已生成，并给出以下固定路径：

```text
D:\Fmodel\emg_data_canonical\splits.official.seed-20260916.csv
D:\Fmodel\emg_data_canonical\splits.strict-subject.seed-20260916.csv
```

每个阶段都包含：用途、命令、输出目录、通过标准和失败恢复。正式 GPU 阶段明确要求 CUDA；CPU-only 环境只能运行 `cpu_smoke`。

- [ ] **Step 3: 运行文档契约测试**

Run: `python -m pytest tests/test_documented_commands.py -q`

Expected: PASS，公开入口、全部 PowerShell 阶段、`Official`、`StrictSubject` 和中文标题均存在。

### Task 4: 完整验证与提交推送

**Files:**
- Verify: `README.md`
- Verify: `docs/META16_TRAINING_WORKFLOW.md`
- Verify: `docs/POWERSHELL_SCRIPT_REFERENCE.md`
- Verify: `docs/POST_CONVERSION_EXECUTION_GUIDE.md`
- Verify: `tests/test_documented_commands.py`

- [ ] **Step 1: 检查英文说明段落和占位符**

Run: `rg -n "^# (Meta16 training workflow|PowerShell script reference)|TBD|TODO|emg2pose 仍在转换" README.md docs/META16_TRAINING_WORKFLOW.md docs/POWERSHELL_SCRIPT_REFERENCE.md docs/POST_CONVERSION_EXECUTION_GUIDE.md`

Expected: 无匹配。

- [ ] **Step 2: 运行 PowerShell 工作流和全仓测试**

Run: `python -m pytest tests/test_powershell_workflow.py tests/test_documented_commands.py tests/test_repository_contract.py -q`

Expected: PASS。

Run: `python -m pytest -q`

Expected: 全部通过，仅允许既有显式 skip。

- [ ] **Step 3: 检查差异**

Run: `git diff --check`

Expected: 无错误。

- [ ] **Step 4: 提交文档与测试**

```powershell
git add README.md docs\META16_TRAINING_WORKFLOW.md docs\POWERSHELL_SCRIPT_REFERENCE.md docs\POST_CONVERSION_EXECUTION_GUIDE.md tests\test_documented_commands.py
git commit -m "docs: publish Chinese training workflow"
```

- [ ] **Step 5: 推送当前分支**

Run: `git push origin codex/standalone-training-workflow`

Expected: 远端分支更新到新提交；若 GitHub 网络不可达，保留本地提交并报告具体错误。
