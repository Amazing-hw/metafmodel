# Standalone Meta 16-Channel sEMG Training Repository Design

## 1. Goal

Turn `Amazing-hw/metafmodel` into a standalone repository that can validate the
converted Meta 16-channel sEMG corpus, prepare evaluation splits, pretrain and
fine-tune foundation models, run comparison baselines, evaluate held-out data,
and execute robustness experiments without depending on another local source
checkout.

The repository will preserve the five existing converter entry points while
adding the complete runtime code, configuration, tests, operational
documentation, and PowerShell orchestration required after conversion.

## 2. Publication Boundary

Development is based on the target repository's existing `main` commit
`bf8f64f`. The work is published on
`codex/standalone-training-workflow`. It does not import the commit history of
`Neuromuscular-model`; required runtime files are added as a reviewed snapshot
in new commits on the target repository.

The following are never committed:

- raw or canonical datasets;
- checkpoints, TensorBoard logs, predictions, or generated result tables;
- machine-specific credentials or GitHub tokens;
- absolute paths under the former `C:\Users\User\Documents\F_model` root.

## 3. Repository Structure

```text
metafmodel/
  README.md
  requirements.txt
  requirements-train.txt
  pytest.ini
  make_splits.py
  pretrain.py
  run_baseline.py
  finetune.py
  evaluate.py
  semg_h5/
  semg_fm/
  configs/
  scripts/powershell/
    Common.ps1
    00_Preflight.ps1
    01_AuditCanonical.ps1
    02_PrepareSplits.ps1
    03_SmokeTest.ps1
    04_TrainBaseline.ps1
    05_Pretrain.ps1
    06_Finetune.ps1
    07_Evaluate.ps1
    08_Robustness.ps1
    RunPipeline.ps1
  docs/
    META16_TRAINING_WORKFLOW.md
    POWERSHELL_SCRIPT_REFERENCE.md
  tests/
```

Each PowerShell file has one responsibility. `Common.ps1` owns shared path
resolution, command invocation, dry-run output, directory checks, and common
validation. Stage scripts expose typed parameters and call the repository's
Python entry points. `RunPipeline.ps1` selects stages but does not silently
start a full-cost training run.

## 4. Default Paths and Portability

The Windows defaults are:

```text
Repository: D:\Fmodel\metafmodel
Canonical data: D:\Fmodel\emg_data_canonical
Run outputs: D:\Fmodel\runs
```

Every operational script accepts `-ProjectRoot`, `-CanonicalRoot`, and
`-RunsRoot` overrides where relevant. Paths passed to Hydra are absolute
because Hydra changes into the run directory. No script depends on the
`codex-ring-criss-cross-semg-fm` worktree after publication.

## 5. Runtime Components

The standalone snapshot includes:

- `semg_h5`: the five canonical H5 converters;
- `semg_fm.data`: manifest loading, leakage-aware split assignment, windowed
  H5 loading, task-balanced distributed sampling, and task label parsing;
- `semg_fm.models`: Ring Criss-Cross, Supervised TCN, BENDR, CBraMod, and
  AEMG backbones;
- `semg_fm.objectives`: masking, EMA-teacher latent prediction, optional
  kinematic anchoring, ring equivariance diagnostics, and vector quantization;
- `semg_fm.lightning`: pretraining, baseline, and downstream Lightning
  modules;
- `semg_fm.metrics`: regression, event, and text metrics;
- Hydra configurations for data, models, tasks, trainers, and robustness;
- entry points for split generation, pretraining, baselines, fine-tuning, and
  evaluation.

## 6. Operational Flow

### 6.1 Preflight

`00_Preflight.ps1` verifies Python, required packages, repository entry points,
the canonical data root, `manifest.csv`, optional CUDA visibility, and writable
run output directories. It fails before training when a prerequisite is
missing.

### 6.2 Canonical data audit

`01_AuditCanonical.ps1` reports record, subject, session, and duration counts
by dataset/task; validates required manifest columns and file existence; checks
the allowed task and split vocabulary; and optionally performs a bounded H5
schema sample audit. It writes a timestamped JSON/CSV audit report under the
run root without modifying canonical H5 files.

### 6.3 Split preparation

`02_PrepareSplits.ps1` supports two explicit protocols:

- `Official`: preserve valid `official_split` assignments and generate only
  missing assignments;
- `StrictSubject`: create a derived manifest whose official split is
  `unknown`, then assign deterministic dataset/subject/pair components to
  train/validation/test.

The two protocols use different output filenames and cannot overwrite the
source manifest unless `-Force` is explicitly passed. The script prints the
effective task-by-split table and verifies pair and subject leakage under the
selected protocol.

### 6.4 Smoke tests

`03_SmokeTest.ps1` runs the real-data CPU smoke profile for all tasks or a
selected task. It uses `data.num_workers=0` by default on Windows. Success
requires resolved configuration, manifest provenance, metadata, and a
completed Lightning fast-dev run.

### 6.5 Baselines and foundation-model pretraining

`04_TrainBaseline.ps1` supports `supervised_tcn`, `bendr`, `cbramod`, and
`aemg`, with only valid pretrain/fine-tune stage combinations.

`05_Pretrain.ps1` supports pilot, full, and resume modes for CIRCE-Myo. Pilot
mode has a bounded trainer step count. Full mode requires explicit device,
accumulation, and maximum-step parameters. The script prints the effective
global batch size before execution.

### 6.6 Fine-tuning, evaluation, and robustness

`06_Finetune.ps1` supports all five downstream tasks, probe/partial/full
fine-tuning, deterministic label budgets, pretrained initialization, and
resume.

`07_Evaluate.ps1` loads a trusted repository-produced checkpoint and writes
metrics, per-subject metrics, split provenance, environment information, and
optional predictions.

`08_Robustness.ps1` evaluates the same checkpoint under clean, channel
rotation, channel dropout, gain drift, additive noise, and temporal-loss
conditions. Every condition receives a separate output directory.

## 7. Dry-Run and Failure Handling

All stage scripts implement `-DryRun`. Dry-run mode validates arguments and
prints the exact command but does not launch Python training. Non-zero exit
codes propagate to the caller. Required inputs use `Test-Path -LiteralPath`;
output collisions stop unless the caller explicitly supplies `-Force` where
that operation is safe.

`RunPipeline.ps1` accepts an explicit stage list. Expensive stages such as
full pretraining, full baseline sweeps, and robustness grids are never enabled
by default.

## 8. Documentation

`docs/META16_TRAINING_WORKFLOW.md` is the end-to-end research runbook. It
explains the current canonical corpus, official versus strict-subject
protocols, gates between stages, exact commands, expected outputs, experiment
matrix, and reporting rules.

`docs/POWERSHELL_SCRIPT_REFERENCE.md` documents every script parameter,
default, example, output, failure mode, and resume behavior. `README.md`
provides installation and quick-start links to both documents.

## 9. Verification

Completion requires all of the following:

1. The complete Python test suite passes from the target repository.
2. Every PowerShell file parses successfully.
3. Every stage script completes `-DryRun` with representative arguments.
4. The canonical data audit accepts `D:\Fmodel\emg_data_canonical` and finds
   no missing manifest paths.
5. The five-task smoke workflow completes against real canonical data, unless
   unavailable hardware makes a GPU-only stage inapplicable.
6. `rg` finds no obsolete project-root references or committed data/checkpoint
   artifacts.
7. Git status contains only intended files before commit and is clean after
   commit.
8. The feature branch is pushed to
   `https://github.com/Amazing-hw/metafmodel.git`.

## 10. Non-Goals

This delivery does not run the full 100,000-step pretraining experiment, the
complete three-seed downstream matrix, or every robustness severity. It
provides and verifies the repository and scripts required to run those jobs.
It also does not rewrite canonical H5 data or change the scientific model
architecture beyond what is required for standalone execution.
