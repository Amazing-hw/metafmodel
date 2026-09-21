# Meta16 training workflow

## Corpus and protocol

The canonical root is `D:\Fmodel\emg_data_canonical`. The completed manifest audited on 2026-09-21 contains 28,609 records, 601 subjects, 14,893 sessions, and 1,396.474 hours. Do not commit data, manifests, checkpoints, predictions, or `D:\Fmodel\runs` artifacts.

| Task | Dataset records | Primary target |
| --- | ---: | --- |
| pose | 25,253 | joint-angle regression |
| typing | 2,270 | text decoding |
| discrete_gesture | 100 | gesture class |
| handwriting | 804 | handwriting trace/event |
| wrist_angle | 182 | wrist-angle regression |

`Official` preserves release `official_split` labels; it can legitimately span a subject across splits. `StrictSubject` copies the manifest, replaces every official assignment with `unknown`, then deterministically splits dataset/subject/pair groups. Use one protocol per reported result and state it in every result table.

For every stage after `02_PrepareSplits.ps1`, select the exact split with either `-SplitsPath` or `-Protocol` plus `-SplitSeed`. The examples below use the latter. Output directories include that selection and refuse accidental reuse unless `-Force` is explicitly supplied. See [the detailed execution guide](POST_CONVERSION_EXECUTION_GUIDE.md) for the complete command sequence and recovery steps.

## Stage gates

1. Preflight checks Python packages, entry points, the canonical root, and `manifest.csv`.

```powershell
powershell -File scripts\powershell\00_Preflight.ps1
```

Pass gate: command exits zero. Output: package/CUDA probe only.

2. Audit verifies paths, allowed task/split vocabulary, corpus totals, and sampled `signal/emg` arrays shaped `[time, 16]`.

```powershell
powershell -File scripts\powershell\01_AuditCanonical.ps1 -H5Samples 20
```

Pass gate: zero missing files and schema errors. Outputs: `D:\Fmodel\runs\audits\audit-*.json` and CSV.

3. Prepare the protocol-specific split file. Use `-DryRun` before `-Force`; neither protocol overwrites `manifest.csv`.

```powershell
powershell -File scripts\powershell\02_PrepareSplits.ps1 -Protocol Official -Seed 20260916
powershell -File scripts\powershell\02_PrepareSplits.ps1 -Protocol StrictSubject -Seed 20260916
```

Pass gate: task-by-split counts print; StrictSubject also prints leakage-check PASS. Outputs: `splits.official.seed-20260916.csv` or `splits.strict-subject.seed-20260916.csv`.

4. Run real-data fast-dev smoke tests for all five tasks before GPU work.

```powershell
powershell -File scripts\powershell\03_SmokeTest.ps1 -Task All -Protocol Official -SplitSeed 20260916
```

Pass gate: five Lightning CPU runs finish with `data.num_workers=0`. Outputs: `runs\smoke\official-seed-20260916\<task>\seed-0`.

5. Establish comparison baselines with `supervised_tcn`, `bendr`, `cbramod`, or `aemg`.

```powershell
powershell -File scripts\powershell\04_TrainBaseline.ps1 -Model bendr -Stage finetune -Task pose -LabelBudget 1.0 -Protocol Official -SplitSeed 20260916
```

6. Start CIRCE-Myo with a bounded pilot, inspect loss/config/manifest metadata, then request compute for Full mode. Full 100,000-step training is not run by this delivery.

```powershell
powershell -File scripts\powershell\05_Pretrain.ps1 -Mode Pilot -MaxSteps 200 -Devices 1 -Protocol Official -SplitSeed 20260916
powershell -File scripts\powershell\05_Pretrain.ps1 -Mode Full -MaxSteps 100000 -Devices 1 -Protocol Official -SplitSeed 20260916
```

7. Fine-tune each task at label budgets 0.01, 0.05, 0.10, and 1.0 with `probe`, `partial`, and `full` modes. Then evaluate only trusted checkpoints produced by this repository.

```powershell
$PretrainCkpt = 'D:\Fmodel\runs\pretrain\official-seed-20260916\full\steps-100000\devices-1\accum-1\batch-8\seed-0\checkpoints\last.ckpt'
powershell -File scripts\powershell\06_Finetune.ps1 -Task pose -FinetuneMode probe -LabelBudget 0.1 -PretrainedCheckpoint $PretrainCkpt -Protocol Official -SplitSeed 20260916
$FinetuneCandidates = @(Get-ChildItem 'D:\Fmodel\runs\finetune\official-seed-20260916\pose\probe\budget-0.1\seed-0\checkpoint-*\checkpoints\last.ckpt')
if ($FinetuneCandidates.Count -ne 1) { throw "expected exactly one checkpoint, found $($FinetuneCandidates.Count)" }
$FinetuneCkpt = $FinetuneCandidates[0].FullName
powershell -File scripts\powershell\07_Evaluate.ps1 -Task pose -CheckpointPath $FinetuneCkpt -Protocol Official -SplitSeed 20260916
```

Evaluation writes resolved configuration, manifest digest, environment metadata, overall/per-subject metrics, and optional predictions.

8. Evaluate robustness independently under clean, channel rotation, channel dropout, gain drift, additive noise, and temporal loss.

```powershell
powershell -File scripts\powershell\08_Robustness.ps1 -Task pose -Condition All -CheckpointPath $FinetuneCkpt -Protocol Official -SplitSeed 20260916
```

Each condition gets its own output directory. The complete multi-seed robustness grid is not run by this delivery.

Use `RunPipeline.ps1 -Stages Preflight,Audit,Splits,Smoke -Protocol Official -SplitSeed 20260916 -DryRun` to inspect a safe ordered sequence. It never adds expensive stages automatically.
