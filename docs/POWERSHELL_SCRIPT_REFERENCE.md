# PowerShell 脚本参考

所有脚本默认使用 `D:\Fmodel\metafmodel`、`D:\Fmodel\emg_data_canonical` 和 `D:\Fmodel\runs`。凡是可能调用外部命令的脚本都支持 `-DryRun`；干跑会验证输入并打印准确的 Python 命令，但不会训练或创建输出。

| 脚本 | 用途 | 关键参数 | 输出与失败行为 |
| --- | --- | --- | --- |
| `00_Preflight.ps1` | 环境预检 | `ProjectRoot`、`CanonicalRoot`、`RunsRoot`、`Python` | 缺少依赖、入口、数据根或 manifest 时失败 |
| `01_AuditCanonical.ps1` | 只读 H5/manifest 审计 | `ManifestPath`、`H5Samples` | 在 `runs\audits` 下写入带时间戳的 JSON/CSV；审计非零退出即失败 |
| `02_PrepareSplits.ps1` | 生成 split manifest | `Protocol`（`Official` 或 `StrictSubject`）、`Seed`、`Force` | 默认拒绝覆盖；StrictSubject 会检查泄漏 |
| `03_SmokeTest.ps1` | 五任务 CPU 门槛 | `Task`、`SplitsPath` 或 `Protocol` + `SplitSeed`、`Seed`、`Force` | 输出到 `runs\smoke\<split-id>\task\seed-<seed>`；使用 `trainer=cpu_smoke` 和零 worker |
| `04_TrainBaseline.ps1` | 对比基线 | split 选择、`Model`、`Stage`、`Task`、`LabelBudget`、`FinetuneMode`、`ResumeFrom`、`Force` | 拒绝 `supervised_tcn` 预训练；输出按 split、模式、预算、seed 和初始化 checkpoint 隔离 |
| `05_Pretrain.ps1` | CIRCE-Myo 预训练 | split 选择、`Mode`、`MaxSteps`、`Devices`、`AccumulateGradBatches`、`BatchSize`、`ResumeFrom`、`Force` | 目录包含步数、设备数、梯度累积、batch 和 seed；`Full` 必须提供步数，`Resume` 必须提供 checkpoint |
| `06_Finetune.ps1` | 下游适配 | split 选择、`Task`、`FinetuneMode`、`LabelBudget`、`PretrainedCheckpoint`、`ResumeFrom`、`Force` | 输出按 split、任务、模式、预算、seed 和初始化 checkpoint 隔离 |
| `07_Evaluate.ps1` | 测试集独立评估 | split 选择、`Task`、必需的 `CheckpointPath`、`Seed`、`SavePredictions`、`Smoke`、`Force` | 输出按 split、seed 和 checkpoint 隔离；非干跑时 checkpoint 不存在即失败 |
| `08_Robustness.ps1` | 扰动条件评估 | split 选择、`Task`、`Condition` 或 `All`、必需的 `CheckpointPath`、`Seed`、`Smoke`、`Force` | 每个条件、seed 和 checkpoint 使用独立目录 |
| `RunPipeline.ps1` | 显式阶段编排 | 必需的逗号分隔 `Stages`；split 阶段使用 `SplitsPath` 或 `Protocol` + `SplitSeed`；可选 `CheckpointPath`、`Force` | 只运行明确列出的阶段并传递同一 split；评估与鲁棒性阶段要求 checkpoint |

## 示例

```powershell
powershell -File scripts\powershell\00_Preflight.ps1 -DryRun
powershell -File scripts\powershell\02_PrepareSplits.ps1 -Protocol StrictSubject -Seed 20260916 -DryRun
powershell -File scripts\powershell\04_TrainBaseline.ps1 -Model aemg -Stage finetune -Task typing -Protocol Official -SplitSeed 20260916 -DryRun
powershell -File scripts\powershell\05_Pretrain.ps1 -Mode Resume -MaxSteps 100000 -ResumeFrom D:\Fmodel\runs\pretrain\official-seed-20260916\full\steps-100000\devices-1\accum-1\batch-8\seed-0\checkpoints\last.ckpt -Protocol Official -SplitSeed 20260916 -DryRun
powershell -File scripts\powershell\RunPipeline.ps1 -Stages Preflight,Audit,Splits,Smoke -Protocol Official -SplitSeed 20260916 -DryRun
```

对应的 Python 入口为 `audit_canonical.py`、`make_splits.py`、`pretrain.py`、`run_baseline.py`、`finetune.py` 和 `evaluate.py`。由于 Hydra 任务会改变工作目录，所有传入路径都使用绝对路径。
