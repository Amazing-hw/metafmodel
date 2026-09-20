# Meta 16-Channel sEMG Foundation-Model Training

This repository is a standalone training stack for the five converted Meta 16-channel sEMG datasets. It preserves the five `convert_*.py` entry points and adds data auditing, split preparation, CIRCE-Myo pretraining, baseline comparison, downstream fine-tuning, evaluation, and robustness experiments.

Default Windows paths are `D:\Fmodel\metafmodel`, `D:\Fmodel\emg_data_canonical`, and `D:\Fmodel\runs`. All scripts accept path overrides.

## Install

```powershell
cd D:\Fmodel\metafmodel
python -m pip install -r requirements-train.txt
python -m pytest -q
```

## Safe first run

```powershell
powershell -File scripts\powershell\00_Preflight.ps1
powershell -File scripts\powershell\01_AuditCanonical.ps1 -H5Samples 20
powershell -File scripts\powershell\02_PrepareSplits.ps1 -Protocol Official -Seed 20260916
powershell -File scripts\powershell\03_SmokeTest.ps1 -Task All -Protocol Official -SplitSeed 20260916
powershell -File scripts\powershell\05_Pretrain.ps1 -Mode Pilot -MaxSteps 200 -Devices 1 -Protocol Official -SplitSeed 20260916
```

Read [the detailed post-conversion execution guide](docs/POST_CONVERSION_EXECUTION_GUIDE.md) before a real experiment, [the workflow overview](docs/META16_TRAINING_WORKFLOW.md) for protocol rules, and [the PowerShell reference](docs/POWERSHELL_SCRIPT_REFERENCE.md) before changing defaults. Python entry points are `audit_canonical.py`, `make_splits.py`, `pretrain.py`, `run_baseline.py`, `finetune.py`, and `evaluate.py`.
