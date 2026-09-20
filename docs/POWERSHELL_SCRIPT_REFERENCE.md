# PowerShell script reference

All scripts default to `D:\Fmodel\metafmodel`, `D:\Fmodel\emg_data_canonical`, and `D:\Fmodel\runs`; all support `-DryRun` where an external command could run. Dry-run validates inputs and prints the exact Python command without training or writing outputs.

| Script | Purpose | Key parameters | Output/failure behavior |
| --- | --- | --- | --- |
| `00_Preflight.ps1` | verify environment | `ProjectRoot`, `CanonicalRoot`, `RunsRoot`, `Python` | fails on missing package, entry point, root, or manifest |
| `01_AuditCanonical.ps1` | read-only H5/manifest audit | `ManifestPath`, `H5Samples` | timestamped JSON/CSV under `runs\audits`; nonzero audit exits fail |
| `02_PrepareSplits.ps1` | create split manifest | `Protocol` (`Official` or `StrictSubject`), `Seed`, `Force` | refuses output collisions unless `Force`; strict mode checks leakage |
| `03_SmokeTest.ps1` | five-task CPU gate | `Task` (`All` or task), required split selection: `SplitsPath` **or** `Protocol` + `SplitSeed`, `Force` | `runs\smoke\<split-id>\task`; uses `trainer=cpu_smoke` and zero workers |
| `04_TrainBaseline.ps1` | comparison baseline | required split selection, `Model`, `Stage`, `Task`, `LabelBudget`, `FinetuneMode`, `ResumeFrom`, `Force` | rejects `supervised_tcn` pretrain; split-scoped `runs\baselines` |
| `05_Pretrain.ps1` | CIRCE-Myo training | required split selection, `Mode` (`Pilot`, `Full`, `Resume`), `MaxSteps`, `Devices`, `BatchSize`, `ResumeFrom`, `Force` | prints global batch; Full requires MaxSteps; Resume requires checkpoint |
| `06_Finetune.ps1` | downstream adaptation | required split selection, `Task`, `FinetuneMode`, `LabelBudget`, `PretrainedCheckpoint`, `ResumeFrom`, `Force` | split-scoped `runs\finetune` |
| `07_Evaluate.ps1` | held-out metrics | required split selection, `Task`, mandatory `CheckpointPath`, `SavePredictions`, `Smoke`, `Force` | split-scoped `runs\evaluation`; missing checkpoint fails outside dry-run |
| `08_Robustness.ps1` | corrupted-condition metrics | required split selection, `Task`, `Condition` or `All`, mandatory `CheckpointPath`, `Smoke`, `Force` | one split-scoped `runs\robustness\task\condition` per condition |
| `RunPipeline.ps1` | explicit orchestration | mandatory comma-separated `Stages`; split stages require `SplitsPath` or `Protocol` + `SplitSeed`; optional `CheckpointPath`, `Force` | stages are only those named; forwards the selected split; evaluation/robustness require checkpoint |

Examples:

```powershell
powershell -File scripts\powershell\00_Preflight.ps1 -DryRun
powershell -File scripts\powershell\02_PrepareSplits.ps1 -Protocol StrictSubject -Seed 20260916 -DryRun
powershell -File scripts\powershell\04_TrainBaseline.ps1 -Model aemg -Stage finetune -Task typing -Protocol Official -SplitSeed 20260916 -DryRun
powershell -File scripts\powershell\05_Pretrain.ps1 -Mode Resume -MaxSteps 100000 -ResumeFrom D:\Fmodel\runs\pretrain\official-seed-20260916\full\seed-0\checkpoints\last.ckpt -Protocol Official -SplitSeed 20260916 -DryRun
powershell -File scripts\powershell\RunPipeline.ps1 -Stages Preflight,Audit,Splits,Smoke -Protocol Official -SplitSeed 20260916 -DryRun
```

Python equivalents are `audit_canonical.py`, `make_splits.py`, `pretrain.py`, `run_baseline.py`, `finetune.py`, and `evaluate.py`. Hydra paths are always absolute because its jobs change working directory.
