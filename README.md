# Meta 16 通道 sEMG 基础模型训练仓库

本仓库是面向五个已转换 Meta 16 通道 sEMG 数据集的独立训练工程。仓库保留五个 `convert_*.py` 转换入口，并提供规范数据审计、切分生成、CIRCE-Myo 预训练、基线对比、下游微调、独立评估和鲁棒性实验所需的完整代码。

默认 Windows 项目目录、规范数据目录和运行输出目录分别为 `D:\Fmodel\metafmodel`、`D:\Fmodel\emg_data_canonical` 和 `D:\Fmodel\runs`。所有脚本均支持通过参数覆盖默认路径。

## 安装与测试

```powershell
cd D:\Fmodel\metafmodel
python -m pip install -r requirements-train.txt
python -m pytest -q
```

## 推荐首次执行顺序

```powershell
powershell -File scripts\powershell\00_Preflight.ps1
powershell -File scripts\powershell\01_AuditCanonical.ps1 -H5Samples 20
powershell -File scripts\powershell\02_PrepareSplits.ps1 -Protocol Official -Seed 20260916
powershell -File scripts\powershell\03_SmokeTest.ps1 -Task All -Protocol Official -SplitSeed 20260916
powershell -File scripts\powershell\05_Pretrain.ps1 -Mode Pilot -MaxSteps 200 -Devices 1 -Protocol Official -SplitSeed 20260916
```

正式实验前请先阅读[转换完成后的详细执行指南](docs/POST_CONVERSION_EXECUTION_GUIDE.md)，通过[完整训练流程](docs/META16_TRAINING_WORKFLOW.md)了解协议和阶段门槛，并在修改默认参数前查阅[PowerShell 脚本参考](docs/POWERSHELL_SCRIPT_REFERENCE.md)。主要 Python 入口为 `audit_canonical.py`、`make_splits.py`、`pretrain.py`、`run_baseline.py`、`finetune.py` 和 `evaluate.py`。
