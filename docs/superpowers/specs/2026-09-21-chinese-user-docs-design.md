# 用户文档全面中文化设计

## 目标

将独立训练仓库中面向使用者的入口说明、训练流程、PowerShell 脚本参考和转换后执行指南统一为中文，使研究人员能够直接按照文档完成数据审计、切分、烟雾测试、预训练、基线、微调、评估和鲁棒性实验。

## 修改范围

本次修改以下四份用户文档：

- `README.md`
- `docs/META16_TRAINING_WORKFLOW.md`
- `docs/POWERSHELL_SCRIPT_REFERENCE.md`
- `docs/POST_CONVERSION_EXECUTION_GUIDE.md`

历史设计与实施计划保留原貌，避免把历史记录改写成当前状态。Python 与 PowerShell 代码不因文档语言调整而改变行为。

## 语言规则

标题、说明、表格列名、通过条件、失败处理、恢复步骤和注意事项全部使用中文。以下内容保持原始英文或代码形式，确保命令可以复制执行并方便检索错误：

- 文件名、路径、Python 模块名和 PowerShell 参数名；
- `Official`、`StrictSubject`、`Pilot`、`Full`、`Resume` 等枚举值；
- `pose`、`typing`、`discrete_gesture`、`handwriting`、`wrist_angle` 等任务 ID；
- 命令行、配置键、环境变量、指标名及程序实际输出关键字。

不采用逐段中英双语，避免文档过长和两种语言内容失去同步。

## 内容结构

`README.md` 提供中文项目定位、默认目录、安装方式、安全起步命令和三份详细文档入口。

训练流程文档按阶段描述语料与协议、环境预检、全量审计、split 生成、五任务烟雾测试、基线、预训练、微调、评估和鲁棒性实验。每个阶段给出命令、输出位置与通过标准。

脚本参考文档逐项说明 `00` 至 `08` 以及 `RunPipeline.ps1` 的用途、关键参数、输出目录和失败保护。

完整执行指南保留最详细的逐步命令，更新为五个数据集已经转换完成的当前状态，并记录 2026-09-21 审计得到的 28,609 条记录、两套 split 文件及其使用规则。

## 安全与一致性

所有示例路径仅使用 `D:\Fmodel`。从 `03_SmokeTest.ps1` 开始，每条训练或评估命令必须明确使用 `-SplitsPath`，或者同时使用 `-Protocol` 与 `-SplitSeed`，不得混用两种选择方式。

文档必须说明当前机器为 CPU-only 时只能执行 `cpu_smoke`，正式 `a100_ddp` 阶段需要 CUDA 主机。输出目录说明必须与脚本当前的 split、训练 seed、标签预算、微调模式、checkpoint 哈希和预训练配置隔离规则一致。

## 验证标准

完成后执行：

1. 文档契约测试，确认所有公开入口、脚本阶段和协议仍被完整列出；
2. 仓库路径契约测试，确认没有退役工作目录；
3. PowerShell 工作流测试，确认文档示例对应的参数契约未被破坏；
4. 全仓 `pytest`；
5. `git diff --check`；
6. 检查四份用户文档的普通说明文字不再以英文段落为主体。

验证通过后，在 `codex/standalone-training-workflow` 分支提交并推送到 `https://github.com/Amazing-hw/metafmodel.git`。网络不可用时保留本地提交并明确报告未推送状态。
