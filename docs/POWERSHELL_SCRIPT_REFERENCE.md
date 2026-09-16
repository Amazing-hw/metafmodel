# PowerShell script reference

All scripts default to `D:\Fmodel\metafmodel`, `D:\Fmodel\emg_data_canonical`, and `D:\Fmodel\runs`; all support `-DryRun` where an external command could run. Dry-run validates inputs and prints the exact Python command without training or writing outputs.

| Script | Purpose | Key parameters | Output/failure behavior |
| --- | --- | --- | --- |
| `00_Preflight.ps1` | verify environment | `ProjectRoot`, `CanonicalRoot`, `RunsRoot`, `Python` | fails on missing package, entry point, root, or manifest |
| `01_AuditCanonical.ps1` | read-only H5/manifest audit | `ManifestPath`, `H5Samples` | timestamped JSON/CSV under `runs\audits`; nonzero audit exits fail |
| `02_PrepareSplits.ps1` | create split manifest | `Protocol` (`Official` or `StrictSubject`), `Seed`, `Force` | refuses output collisions unless `Force`; strict mode checks leakage |
| `03_SmokeTest.ps1` | five-task CPU gate | `Task` (`All` or task), `SplitsPath`, `Seed` | `runs\smoke\task`; uses `trainer=cpu_smoke` and zero workers |
| `04_TrainBaseline.ps1` | comparison baseline | `Model`, `Stage`, `Task`, `LabelBudget`, `FinetuneMode`, `ResumeFrom` | rejects `supervised_tcn` pretrain; `runs\baselines` |
| `05_Pretrain.ps1` | CIRCE-Myo training | `Mode` (`Pilot`, `Full`, `Resume`), `MaxSteps`, `Devices`, `BatchSize`, `ResumeFrom` | prints global batch; Full requires MaxSteps; Resume requires checkpoint |
| `06_Finetune.ps1` | downstream adaptation | `Task`, `FinetuneMode`, `LabelBudget`, `PretrainedCheckpoint`, `ResumeFrom` | `runs\finetune` |
| `07_Evaluate.ps1` | held-out metrics | `Task`, mandatory `CheckpointPath`, `SavePredictions`, `Smoke` | `runs\evaluation`; missing checkpoint fails outside dry-run |
| `08_Robustness.ps1` | corrupted-condition metrics | `Task`, `Condition` or `All`, mandatory `CheckpointPath`, `Smoke` | one `runs\robustness\task\condition` per condition |
| `RunPipeline.ps1` | explicit orchestration | mandatory comma-separated `Stages`, optional `CheckpointPath` | stages are only those named; evaluation/robustness require checkpoint |

Examples:

```powershell
powershell -File scripts\powershell\00_Preflight.ps1 -DryRun
powershell -File scripts\powershell\02_PrepareSplits.ps1 -Protocol StrictSubject -Seed 20260916 -DryRun
powershell -File scripts\powershell\04_TrainBaseline.ps1 -Model aemg -Stage finetune -Task typing -DryRun
powershell -File scripts\powershell\05_Pretrain.ps1 -Mode Resume -MaxSteps 100000 -ResumeFrom D:\Fmodel\runs\pretrain\full\seed-0\checkpoints\last.ckpt -DryRun
powershell -File scripts\powershell\RunPipeline.ps1 -Stages Preflight,Audit,Splits,Smoke -DryRun
```

Python equivalents are `audit_canonical.py`, `make_splits.py`, `pretrain.py`, `run_baseline.py`, `finetune.py`, and `evaluate.py`. Hydra paths are always absolute because its jobs change working directory.
