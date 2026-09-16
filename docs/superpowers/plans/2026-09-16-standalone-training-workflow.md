# Standalone Meta 16-Channel sEMG Training Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a self-contained `metafmodel` repository that audits the five converted Meta sEMG datasets and provides reproducible entry points for splitting, smoke testing, baseline training, foundation-model pretraining, fine-tuning, evaluation, and robustness experiments.

**Architecture:** Preserve the target repository's five converter entry points, then add a reviewed snapshot of the tested `semg_h5`, `semg_fm`, Hydra configuration, and test stack from the existing `D:\Fmodel` worktree. Thin PowerShell stage scripts share validation and command execution through `Common.ps1`; Python remains the owner of data, model, objective, and metric behavior. A dedicated audit CLI produces machine-readable reports without modifying the canonical corpus.

**Tech Stack:** Python 3.10+, PyTorch, Lightning, Hydra/OmegaConf, pandas, NumPy, h5py, pytest, PowerShell 7 or Windows PowerShell 5.1, Git.

---

## File map

- `.gitignore`: excludes datasets, checkpoints, run outputs, caches, and local environments.
- `README.md`: standalone installation and quick-start landing page.
- `requirements.txt`, `requirements-train.txt`, `pytest.ini`: environment and test configuration.
- `make_splits.py`, `pretrain.py`, `run_baseline.py`, `finetune.py`, `evaluate.py`: public Python entry points.
- `semg_h5/`: canonical conversion implementation.
- `semg_fm/`: data, model, objective, Lightning, metrics, evaluation, and reproducibility runtime.
- `configs/`: Hydra data/model/task/trainer/robustness configurations.
- `audit_canonical.py`, `semg_fm/data/audit.py`: read-only corpus audit CLI and implementation.
- `scripts/powershell/Common.ps1`: shared path, validation, dry-run, and exit-code helpers.
- `scripts/powershell/00_Preflight.ps1` through `08_Robustness.ps1`: one operational stage per script.
- `scripts/powershell/RunPipeline.ps1`: explicit stage selection without automatic expensive runs.
- `docs/META16_TRAINING_WORKFLOW.md`: end-to-end research runbook.
- `docs/POWERSHELL_SCRIPT_REFERENCE.md`: parameter, command, output, and failure reference.
- `tests/test_repository_contract.py`: publication boundary and inventory tests.
- `tests/test_canonical_audit.py`: audit correctness and error tests.
- `tests/test_powershell_workflow.py`: script parsing, dry-run, and safety tests.
- `tests/`: existing model/data/config/CLI regression suite copied with the runtime.

### Task 1: Lock the publication boundary and import the tested runtime snapshot

**Files:**
- Create: `.gitignore`
- Create: `tests/test_repository_contract.py`
- Create: `requirements.txt`
- Create: `requirements-train.txt`
- Create: `pytest.ini`
- Create: `make_splits.py`
- Create: `pretrain.py`
- Create: `run_baseline.py`
- Create: `finetune.py`
- Create: `evaluate.py`
- Create: `semg_h5/`
- Create: `semg_fm/`
- Create: `configs/`
- Create: `tests/test_augment.py`, `tests/test_baseline_cbramod_aemg.py`, `tests/test_baseline_runner.py`, `tests/test_baseline_tcn_bendr.py`, `tests/test_checkpoint_resume.py`, `tests/test_cli_integration.py`, `tests/test_datamodule.py`, `tests/test_documented_commands.py`, `tests/test_downstream_entry.py`, `tests/test_downstream_module.py`, `tests/test_downstream_primitives.py`, `tests/test_downstream_protocols.py`, `tests/test_evaluation_outputs.py`, `tests/test_fm_manifest.py`, `tests/test_fm_splits.py`, `tests/test_gni_hardening.py`, `tests/test_hydra_config.py`, `tests/test_make_splits_cli.py`, `tests/test_meta_hardening.py`, `tests/test_patch_encoder.py`, `tests/test_pretrain_entry.py`, `tests/test_pretrain_module.py`, `tests/test_pretrain_objectives.py`, `tests/test_pretrain_smoke.py`, `tests/test_reproducibility.py`, `tests/test_ring_attention.py`, `tests/test_ring_criss_cross.py`, `tests/test_standalone_scripts.py`, `tests/test_task_labels.py`, `tests/test_window_dataset.py`, and `tests/conftest.py`
- Preserve: all five root `convert_*.py` files

- [ ] **Step 1: Write the failing repository contract test**

Create `tests/test_repository_contract.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_standalone_runtime_inventory():
    required = [
        "make_splits.py", "pretrain.py", "run_baseline.py", "finetune.py",
        "evaluate.py", "semg_h5", "semg_fm", "configs", "requirements.txt",
        "requirements-train.txt", "pytest.ini",
    ]
    assert [name for name in required if not (ROOT / name).exists()] == []


def test_original_converter_entry_points_are_preserved():
    stems = [
        "emg2pose", "emg2qwerty", "gni_discrete_gestures",
        "gni_handwriting", "gni_wrist",
    ]
    assert [stem for stem in stems if not (ROOT / f"convert_{stem}.py").is_file()] == []


def test_repository_text_has_no_retired_project_root():
    retired = "C:" + "\\Users\\User\\Documents\\F_model"
    suffixes = {".py", ".yaml", ".yml", ".md", ".ps1", ".txt", ".ini"}
    offenders = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if retired.lower() in text.lower():
                offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []
```

- [ ] **Step 2: Verify the inventory test fails**

Run from `D:\Fmodel\metafmodel`:

```powershell
python -m pytest tests\test_repository_contract.py::test_standalone_runtime_inventory -q
```

Expected: failure listing at least `make_splits.py`, `semg_fm`, and `configs`.

- [ ] **Step 3: Add the publication ignore policy**

Create `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.coverage
htmlcov/
.venv/
venv/
.idea/
.vscode/
outputs/
runs/
lightning_logs/
checkpoints/
*.ckpt
*.pt
*.pth
*.h5
*.hdf5
manifest.csv
splits.csv
audit-*.json
audit-*.csv
```

- [ ] **Step 4: Copy the reviewed runtime snapshot without Git history or generated artifacts**

```powershell
$source = 'D:\Fmodel\.worktrees\codex-ring-criss-cross-semg-fm'
Copy-Item -LiteralPath "$source\requirements.txt", "$source\requirements-train.txt", "$source\pytest.ini", "$source\make_splits.py", "$source\pretrain.py", "$source\run_baseline.py", "$source\finetune.py", "$source\evaluate.py" -Destination .
Copy-Item -Recurse -LiteralPath "$source\semg_h5", "$source\semg_fm", "$source\configs" -Destination .
New-Item -ItemType Directory -Force -Path .\tests | Out-Null
Get-ChildItem -LiteralPath "$source\tests" -Filter 'test_*.py' | Copy-Item -Destination .\tests
Copy-Item -LiteralPath "$source\tests\conftest.py" -Destination .\tests
```

The root converter files are excluded from the copy command and remain at the target repository versions.

- [ ] **Step 5: Verify the snapshot**

```powershell
python -m pytest tests\test_repository_contract.py -q
python -c "import semg_fm, semg_h5; import pretrain, run_baseline, finetune, evaluate"
```

Expected: repository contract passes and imports exit with code 0.

- [ ] **Step 6: Commit the runtime snapshot**

```powershell
git add .gitignore requirements.txt requirements-train.txt pytest.ini make_splits.py pretrain.py run_baseline.py finetune.py evaluate.py semg_h5 semg_fm configs tests
git commit -m "feat: add standalone training runtime"
```

### Task 2: Add a read-only canonical corpus audit CLI

**Files:**
- Create: `semg_fm/data/audit.py`
- Create: `audit_canonical.py`
- Create: `tests/test_canonical_audit.py`

- [ ] **Step 1: Write failing audit tests**

Create `tests/test_canonical_audit.py`:

```python
import json
from pathlib import Path

import h5py
import pandas as pd
import pytest

from semg_fm.data.audit import audit_canonical_corpus, write_audit_reports


def make_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "canonical"
    sample = root / "demo" / "subject-001" / "session-001" / "sample.h5"
    sample.parent.mkdir(parents=True)
    with h5py.File(sample, "w") as handle:
        handle.create_dataset("emg", data=[[0.0] * 16] * 20)
    pd.DataFrame([{
        "path": sample.relative_to(root).as_posix(), "dataset_id": "demo",
        "task_id": "pose", "subject_id": "subject-001",
        "session_id": "session-001", "duration_s": 0.01,
        "official_split": "train",
    }]).to_csv(root / "manifest.csv", index=False)
    return root


def test_audit_reports_counts_and_zero_missing_files(tmp_path):
    root = make_corpus(tmp_path)
    report, rows = audit_canonical_corpus(root, root / "manifest.csv", h5_samples=1)
    assert report["total_records"] == 1
    assert report["missing_files"] == 0
    assert report["sampled_h5_files"] == 1
    assert rows[0]["task_id"] == "pose"


def test_audit_rejects_missing_required_columns(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"path": "x.h5"}]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="missing required manifest columns"):
        audit_canonical_corpus(tmp_path, manifest, h5_samples=0)


def test_write_audit_reports_creates_json_and_csv(tmp_path):
    root = make_corpus(tmp_path)
    report, rows = audit_canonical_corpus(root, root / "manifest.csv", h5_samples=0)
    json_path, csv_path = write_audit_reports(tmp_path / "reports" / "audit", report, rows)
    assert json.loads(json_path.read_text(encoding="utf-8"))["total_records"] == 1
    assert pd.read_csv(csv_path).iloc[0]["records"] == 1
```

- [ ] **Step 2: Verify the missing-module failure**

```powershell
python -m pytest tests\test_canonical_audit.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'semg_fm.data.audit'`.

- [ ] **Step 3: Implement the audit module**

Create `semg_fm/data/audit.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import h5py
import pandas as pd

REQUIRED_COLUMNS = {
    "path", "dataset_id", "task_id", "subject_id", "session_id",
    "duration_s", "official_split",
}
ALLOWED_TASKS = {"pose", "wrist_angle", "discrete_gesture", "handwriting", "typing"}
ALLOWED_SPLITS = {"", "unknown", "train", "val", "test"}


def audit_canonical_corpus(root: Path, manifest_path: Path, h5_samples: int = 0):
    root = root.resolve()
    frame = pd.read_csv(manifest_path.resolve(), keep_default_na=False)
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"missing required manifest columns: {', '.join(missing_columns)}")
    invalid_tasks = sorted(set(frame["task_id"].astype(str)) - ALLOWED_TASKS)
    official = frame["official_split"].astype(str).str.strip()
    invalid_splits = sorted(set(official) - ALLOWED_SPLITS)
    resolved = [root / Path(value) for value in frame["path"].astype(str)]
    missing_paths = [str(path) for path in resolved if not path.is_file()]
    sampled = 0
    schema_errors = []
    for path in [path for path in resolved if path.is_file()][:h5_samples]:
        try:
            with h5py.File(path, "r") as handle:
                if "signal/emg" not in handle:
                    schema_errors.append(f"{path}: missing signal/emg dataset")
                elif handle["signal/emg"].ndim != 2 or handle["signal/emg"].shape[1] != 16:
                    schema_errors.append(f"{path}: signal/emg must have shape [time, 16]")
            sampled += 1
        except OSError as exc:
            schema_errors.append(f"{path}: {exc}")
    grouped = frame.groupby(["dataset_id", "task_id"], dropna=False).agg(
        records=("path", "size"), subjects=("subject_id", "nunique"),
        sessions=("session_id", "nunique"), duration_s=("duration_s", "sum"),
    ).reset_index()
    rows = grouped.to_dict(orient="records")
    report = {
        "root": str(root), "manifest": str(manifest_path.resolve()),
        "total_records": int(len(frame)),
        "total_subjects": int(frame["subject_id"].nunique()),
        "total_sessions": int(frame["session_id"].nunique()),
        "duration_hours": float(pd.to_numeric(frame["duration_s"]).sum() / 3600.0),
        "missing_files": len(missing_paths), "missing_paths": missing_paths,
        "invalid_tasks": invalid_tasks, "invalid_splits": invalid_splits,
        "sampled_h5_files": sampled, "h5_schema_errors": schema_errors,
    }
    return report, rows


def write_audit_reports(output_stem: Path, report: dict, rows: list[dict]):
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_stem.with_suffix(".json")
    csv_path = output_stem.with_suffix(".csv")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False, lineterminator="\n")
    return json_path, csv_path
```

- [ ] **Step 4: Implement the audit CLI**

Create `audit_canonical.py` with `argparse` options `--root`, `--manifest`, `--output-stem`, and `--h5-samples`. Call both audit functions, print the JSON summary, and return code 2 when missing files, invalid tasks, invalid splits, or H5 schema errors are present. End with `raise SystemExit(main())`.

- [ ] **Step 5: Run focused tests and a real-corpus audit**

```powershell
python -m pytest tests\test_canonical_audit.py -q
python audit_canonical.py --root D:\Fmodel\emg_data_canonical --manifest D:\Fmodel\emg_data_canonical\manifest.csv --output-stem D:\Fmodel\runs\audit-plan-check --h5-samples 5
```

Expected: three tests pass; the real audit reports 14,498 records and zero missing files.

- [ ] **Step 6: Commit the audit CLI**

```powershell
git add audit_canonical.py semg_fm/data/audit.py tests/test_canonical_audit.py
git commit -m "feat: add canonical corpus audit"
```

### Task 3: Build shared PowerShell primitives and preflight checks

**Files:**
- Create: `scripts/powershell/Common.ps1`
- Create: `scripts/powershell/00_Preflight.ps1`
- Create: `tests/test_powershell_workflow.py`

- [ ] **Step 1: Write failing PowerShell contract tests**

Create `tests/test_powershell_workflow.py`. Select `pwsh` first and fall back to `powershell`; parse every `.ps1` file with the parser API; run `00_Preflight.ps1 -DryRun`; assert that output contains the three D-drive defaults and never contains the retired C-drive project root. Use this parser command:

```python
command = [
    shell, "-NoProfile", "-Command",
    "$e=$null;$t=$null;[System.Management.Automation.Language.Parser]::ParseFile($args[0],[ref]$t,[ref]$e)>$null;if($e.Count){$e|% Message;exit 1}",
    str(script),
]
```

- [ ] **Step 2: Verify missing-script failures**

```powershell
python -m pytest tests\test_powershell_workflow.py -q
```

Expected: failure because `scripts/powershell` is absent.

- [ ] **Step 3: Implement `Common.ps1`**

```powershell
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-FullPath([string]$Path, [string]$BasePath = (Get-Location).Path) {
    if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
    return [IO.Path]::GetFullPath((Join-Path $BasePath $Path))
}
function Assert-File([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$Label not found: $Path" }
}
function Assert-Directory([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { throw "$Label not found: $Path" }
}
function Format-Command([string]$Executable, [string[]]$Arguments) {
    $quoted = $Arguments | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }
    return (($Executable, $quoted) -join ' ')
}
function Invoke-Checked([string]$Executable, [string[]]$Arguments, [switch]$DryRun) {
    Write-Host (Format-Command $Executable $Arguments)
    if ($DryRun) { return }
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable exited with code $LASTEXITCODE" }
}
function Ensure-OutputDirectory([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { New-Item -ItemType Directory -Path $Path | Out-Null }
    return (Resolve-Path -LiteralPath $Path).Path
}
```

- [ ] **Step 4: Implement preflight**

`00_Preflight.ps1` accepts typed `ProjectRoot`, `CanonicalRoot`, `RunsRoot`, `Python`, and `DryRun` parameters with D-drive defaults. Dot-source `Common.ps1`; verify the five Python entry points and `manifest.csv`; create the run root only outside dry-run; then invoke:

```powershell
@('-c', 'import torch, lightning, hydra, omegaconf, pandas, h5py; print(torch.__version__); print(torch.cuda.is_available())')
```

- [ ] **Step 5: Run focused tests and dry-run**

```powershell
python -m pytest tests\test_powershell_workflow.py -q
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\powershell\00_Preflight.ps1 -DryRun
```

Expected: tests pass and only validated D-drive paths are printed.

- [ ] **Step 6: Commit shared workflow primitives**

```powershell
git add scripts/powershell/Common.ps1 scripts/powershell/00_Preflight.ps1 tests/test_powershell_workflow.py
git commit -m "feat: add PowerShell preflight workflow"
```

### Task 4: Add canonical audit and split-preparation wrappers

**Files:**
- Create: `scripts/powershell/01_AuditCanonical.ps1`
- Create: `scripts/powershell/02_PrepareSplits.ps1`
- Modify: `tests/test_powershell_workflow.py`

- [ ] **Step 1: Add failing dry-run cases**

Add cases for these commands and assert distinct official/strict output filenames:

```text
01_AuditCanonical.ps1 -DryRun -H5Samples 5
02_PrepareSplits.ps1 -DryRun -Protocol Official -Seed 20260916
02_PrepareSplits.ps1 -DryRun -Protocol StrictSubject -Seed 20260916
```

- [ ] **Step 2: Verify only the new cases fail**

```powershell
python -m pytest tests\test_powershell_workflow.py -q
```

- [ ] **Step 3: Implement the audit wrapper**

`01_AuditCanonical.ps1` accepts common paths plus `ManifestPath`, `H5Samples`, `Python`, and `DryRun`. Validate non-negative sampling, use timestamp `yyyyMMdd-HHmmss`, and call:

```text
python audit_canonical.py --root D:\Fmodel\emg_data_canonical --manifest D:\Fmodel\emg_data_canonical\manifest.csv --output-stem D:\Fmodel\runs\audits\audit-20260916-120000 --h5-samples 5
```

- [ ] **Step 4: Implement protocol-safe split preparation**

`02_PrepareSplits.ps1` validates `Protocol` to `Official` or `StrictSubject`. Official mode passes `manifest.csv` to `make_splits.py` and writes `splits.official.seed-$Seed.csv`. Strict-subject mode exports a derived `manifest.strict-subject.csv` after setting every `official_split` value to `unknown`, then writes `splits.strict-subject.seed-$Seed.csv`. Existing outputs raise unless `-Force` is supplied. Dry-run writes nothing. After a real run, import the result, print the `task_id` by `split` count table, and call the Python leakage assertion for the selected protocol.

- [ ] **Step 5: Run focused and real-data checks**

```powershell
python -m pytest tests\test_powershell_workflow.py -q
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\powershell\01_AuditCanonical.ps1 -H5Samples 5
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\powershell\02_PrepareSplits.ps1 -Protocol Official -Seed 20260916 -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\powershell\02_PrepareSplits.ps1 -Protocol StrictSubject -Seed 20260916 -DryRun
```

Expected: audit exits 0; split commands show separate destinations and do not alter `manifest.csv`.

- [ ] **Step 6: Commit the wrappers**

```powershell
git add scripts/powershell/01_AuditCanonical.ps1 scripts/powershell/02_PrepareSplits.ps1 tests/test_powershell_workflow.py
git commit -m "feat: add audit and split workflow scripts"
```

### Task 5: Add bounded smoke and baseline training scripts

**Files:**
- Create: `scripts/powershell/03_SmokeTest.ps1`
- Create: `scripts/powershell/04_TrainBaseline.ps1`
- Modify: `tests/test_powershell_workflow.py`

- [ ] **Step 1: Add failing command-construction tests**

Cover all five tasks and four baseline models. Assert smoke commands include `trainer=cpu_smoke` and `data.num_workers=0`. Limit models to `supervised_tcn`, `bendr`, `cbramod`, and `aemg`; limit stages to `pretrain` and `finetune`; reject supervised TCN pretraining.

- [ ] **Step 2: Verify missing-script failures**

```powershell
python -m pytest tests\test_powershell_workflow.py -q
```

- [ ] **Step 3: Implement `03_SmokeTest.ps1`**

Support `Task` values `All`, `pose`, `wrist_angle`, `discrete_gesture`, `handwriting`, and `typing`. Each selected task invokes:

```text
python finetune.py trainer=cpu_smoke task=pose data.root=D:\Fmodel\emg_data_canonical data.manifest_path=D:\Fmodel\emg_data_canonical\splits.official.seed-20260916.csv data.num_workers=0 hydra.run.dir=D:\Fmodel\runs\smoke\pose
```

- [ ] **Step 4: Implement `04_TrainBaseline.ps1`**

Expose `Model`, `Stage`, `Task`, `Seed`, `LabelBudget`, `FinetuneMode`, `UnfrozenBlocks`, `ResumeFrom`, and common paths. Invoke `run_baseline.py` and construct the output as `Join-Path $RunsRoot "baselines\$Model\$Stage\$Task\seed-$Seed"`. Add checkpoint overrides only when provided.

- [ ] **Step 5: Run focused tests and representative dry-runs**

```powershell
python -m pytest tests\test_powershell_workflow.py -q
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\powershell\03_SmokeTest.ps1 -Task All -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\powershell\04_TrainBaseline.ps1 -Model bendr -Stage finetune -Task pose -DryRun
```

Expected: five smoke commands and one baseline command print without launching Trainer.

- [ ] **Step 6: Commit smoke and baseline scripts**

```powershell
git add scripts/powershell/03_SmokeTest.ps1 scripts/powershell/04_TrainBaseline.ps1 tests/test_powershell_workflow.py
git commit -m "feat: add smoke and baseline scripts"
```

### Task 6: Add pilot, full, and resume pretraining

**Files:**
- Create: `scripts/powershell/05_Pretrain.ps1`
- Modify: `tests/test_powershell_workflow.py`

- [ ] **Step 1: Add failing mode and batch-size tests**

Test `Pilot`, `Full`, and `Resume`. Require positive `MaxSteps`, `Devices`, `AccumulateGradBatches`, and `BatchSize`. Assert the printed global batch size equals their product; pilot is bounded; resume requires a checkpoint; full does not use CPU-smoke settings.

- [ ] **Step 2: Verify the new cases fail**

```powershell
python -m pytest tests\test_powershell_workflow.py -q
```

- [ ] **Step 3: Implement `05_Pretrain.ps1`**

Invoke `pretrain.py` with absolute data, split, output, and resume paths. Pilot defaults to 200 steps, one device, and batch size 8. Full requires caller-supplied maximum steps. Resume adds the concrete PowerShell value `"resume_from=$ResumeFrom"`. Print the global batch size and run directory before invocation.

- [ ] **Step 4: Verify command construction**

```powershell
python -m pytest tests\test_powershell_workflow.py -q
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\powershell\05_Pretrain.ps1 -Mode Pilot -MaxSteps 20 -Devices 1 -AccumulateGradBatches 2 -BatchSize 8 -DryRun
```

Expected: output contains `global_batch_size=16`, `optimizer.max_steps=20`, and an absolute run directory.

- [ ] **Step 5: Commit pretraining orchestration**

```powershell
git add scripts/powershell/05_Pretrain.ps1 tests/test_powershell_workflow.py
git commit -m "feat: add pretraining orchestration"
```

### Task 7: Add fine-tuning, evaluation, and robustness scripts

**Files:**
- Create: `scripts/powershell/06_Finetune.ps1`
- Create: `scripts/powershell/07_Evaluate.ps1`
- Create: `scripts/powershell/08_Robustness.ps1`
- Modify: `tests/test_powershell_workflow.py`

- [ ] **Step 1: Add failing downstream dry-run tests**

Cover all five tasks; modes `probe`, `partial`, `full`; budgets `0.01`, `0.05`, `0.10`, `1.0`; prediction toggling; and conditions `none`, `channel_rotation`, `channel_dropout`, `gain_drift`, `additive_noise`, `temporal_loss`. Outside dry-run, checkpoint consumers reject missing files and state that only repository-produced checkpoints are trusted.

- [ ] **Step 2: Verify the new cases fail**

```powershell
python -m pytest tests\test_powershell_workflow.py -q
```

- [ ] **Step 3: Implement fine-tuning**

`06_Finetune.ps1` invokes `finetune.py` with task, seed, label budget, mode, unfrozen blocks, data paths, and optional pretraining/resume checkpoints. Construct the output as `Join-Path $RunsRoot "finetune\$Task\$FinetuneMode\budget-$LabelBudget\seed-$Seed"`.

- [ ] **Step 4: Implement held-out evaluation**

`07_Evaluate.ps1` requires `CheckpointPath` and invokes `evaluate.py` with checkpoint, output, task, data paths, and optional `save_predictions=true`. `trainer=cpu_smoke` is used only for explicit `-Smoke`.

- [ ] **Step 5: Implement isolated robustness conditions**

`08_Robustness.ps1` accepts one condition or `All`. Invoke `evaluate.py` once per selected condition with the concrete override `"robustness=$item"` and output `Join-Path $RunsRoot "robustness\$Task\$item"`.

- [ ] **Step 6: Verify tests and dry-runs**

```powershell
python -m pytest tests\test_powershell_workflow.py -q
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\powershell\06_Finetune.ps1 -Task pose -FinetuneMode probe -LabelBudget 0.1 -PretrainedCheckpoint D:\Fmodel\runs\pretrain-pilot\checkpoints\last.ckpt -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\powershell\07_Evaluate.ps1 -Task pose -CheckpointPath D:\Fmodel\runs\finetune\pose\full\budget-1\seed-0\checkpoints\last.ckpt -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\powershell\08_Robustness.ps1 -Task pose -Condition All -CheckpointPath D:\Fmodel\runs\finetune\pose\full\budget-1\seed-0\checkpoints\last.ckpt -DryRun
```

Expected: commands are valid and robustness prints six distinct output directories.

- [ ] **Step 7: Commit downstream scripts**

```powershell
git add scripts/powershell/06_Finetune.ps1 scripts/powershell/07_Evaluate.ps1 scripts/powershell/08_Robustness.ps1 tests/test_powershell_workflow.py
git commit -m "feat: add downstream evaluation scripts"
```

### Task 8: Add explicit pipeline orchestration

**Files:**
- Create: `scripts/powershell/RunPipeline.ps1`
- Modify: `tests/test_powershell_workflow.py`

- [ ] **Step 1: Add failing pipeline-selection tests**

Require a non-empty `Stages` array from `Preflight`, `Audit`, `Splits`, `Smoke`, `Baseline`, `Pretrain`, `Finetune`, `Evaluate`, `Robustness`. Assert an omitted list performs no work and fails clearly. Assert `-DryRun` is forwarded to every child.

- [ ] **Step 2: Verify the missing pipeline failure**

```powershell
python -m pytest tests\test_powershell_workflow.py -q
```

- [ ] **Step 3: Implement `RunPipeline.ps1`**

Use a fixed stage-to-script mapping. Resolve each script under the repository, forward common paths and `-DryRun`, stop on the first non-zero result, and never add an expensive stage automatically.

- [ ] **Step 4: Verify safe stage selection**

```powershell
python -m pytest tests\test_powershell_workflow.py -q
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\powershell\RunPipeline.ps1 -Stages Preflight,Audit,Splits,Smoke -DryRun
```

Expected: exactly four child stages print in the requested order.

- [ ] **Step 5: Commit the pipeline**

```powershell
git add scripts/powershell/RunPipeline.ps1 tests/test_powershell_workflow.py
git commit -m "feat: add explicit training pipeline"
```

### Task 9: Write workflow and script-reference documentation

**Files:**
- Create or replace: `README.md`
- Create: `docs/META16_TRAINING_WORKFLOW.md`
- Create: `docs/POWERSHELL_SCRIPT_REFERENCE.md`
- Create: `PRETRAINING.md`
- Create: `DOWNSTREAM_EVALUATION.md`
- Create: `BASELINES.md`
- Create: `EXPERIMENT_PROTOCOL.md`
- Create: `H5_DATA_CONTRACT.md`
- Create: `SERVER_A100.md`
- Modify: `tests/test_documented_commands.py`

- [ ] **Step 1: Extend documentation contract tests**

Require both new documents and every PowerShell script basename to appear in the runbook or reference. Assert D-drive examples, official versus strict-subject behavior, pilot before full pretraining, stage gates, and an explicit statement that the full experiment has not been executed by this delivery.

- [ ] **Step 2: Verify documentation failures**

```powershell
python -m pytest tests\test_documented_commands.py tests\test_repository_contract.py -q
```

Expected: failures name the missing workflow and script-reference documents.

- [ ] **Step 3: Copy detailed runtime references**

```powershell
$source = 'D:\Fmodel\.worktrees\codex-ring-criss-cross-semg-fm'
Copy-Item -LiteralPath "$source\PRETRAINING.md", "$source\DOWNSTREAM_EVALUATION.md", "$source\BASELINES.md", "$source\EXPERIMENT_PROTOCOL.md", "$source\H5_DATA_CONTRACT.md", "$source\SERVER_A100.md" -Destination .
```

- [ ] **Step 4: Write the end-to-end runbook**

`docs/META16_TRAINING_WORKFLOW.md` contains, in order: corpus assumptions and the measured five-dataset table; environment creation; preflight; audit; both split protocols; all-task smoke gate; baseline pilot; CIRCE-Myo pilot; full pretraining; five-task fine-tuning matrix; held-out evaluation; robustness grid; artifact layout; failure recovery; reproducibility checklist; and manuscript reporting checklist. Every stage includes one exact command, expected output files, a pass gate, and the next permitted stage.

- [ ] **Step 5: Write the script reference**

`docs/POWERSHELL_SCRIPT_REFERENCE.md` documents every common parameter and each script's purpose, syntax, defaults, validation, output paths, dry-run behavior, resume behavior, and exact example. Include task/model/robustness compatibility tables and explain that the pipeline never opts into expensive stages.

- [ ] **Step 6: Write the root quick start**

`README.md` links both documents, preserves converter usage, lists prerequisites, installs `requirements-train.txt`, and gives this safe sequence:

```powershell
powershell -File scripts\powershell\00_Preflight.ps1
powershell -File scripts\powershell\01_AuditCanonical.ps1 -H5Samples 20
powershell -File scripts\powershell\02_PrepareSplits.ps1 -Protocol Official -Seed 20260916
powershell -File scripts\powershell\03_SmokeTest.ps1 -Task All
powershell -File scripts\powershell\05_Pretrain.ps1 -Mode Pilot -MaxSteps 200 -Devices 1
```

- [ ] **Step 7: Verify documentation contracts**

```powershell
python -m pytest tests\test_documented_commands.py tests\test_repository_contract.py -q
rg -n -i "C:\\Users\\User\\Documents\\F_model" README.md docs scripts
```

Expected: tests pass and the path scan prints no matches.

- [ ] **Step 8: Commit documentation**

```powershell
git add README.md PRETRAINING.md DOWNSTREAM_EVALUATION.md BASELINES.md EXPERIMENT_PROTOCOL.md H5_DATA_CONTRACT.md SERVER_A100.md docs/META16_TRAINING_WORKFLOW.md docs/POWERSHELL_SCRIPT_REFERENCE.md tests/test_documented_commands.py
git commit -m "docs: add Meta16 training runbook"
```

### Task 10: Run full verification, review the publication diff, and push

**Files:**
- Modify only files required to correct observed verification failures.

- [ ] **Step 1: Run the full Python suite**

```powershell
python -m pytest -q
```

Expected: all tests pass. The source baseline is 257 passed and 1 skipped; the target count must exceed 257 because new workflow tests are added.

- [ ] **Step 2: Parse every PowerShell script**

```powershell
Get-ChildItem scripts\powershell\*.ps1 | ForEach-Object { $tokens=$null; $errors=$null; [System.Management.Automation.Language.Parser]::ParseFile($_.FullName,[ref]$tokens,[ref]$errors) | Out-Null; if ($errors.Count) { $errors | ForEach-Object Message; throw "PowerShell parse failure: $($_.FullName)" } }
```

Expected: no parser errors.

- [ ] **Step 3: Run representative dry-runs for every stage**

Run the documented dry-run matrix: both split protocols, every smoke task, all baseline models, all three pretraining modes, all fine-tuning modes, evaluation, all robustness conditions, and the safe pipeline. Every command must exit 0 and dry-run must create no training output.

- [ ] **Step 4: Audit the actual canonical corpus**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\powershell\01_AuditCanonical.ps1 -CanonicalRoot D:\Fmodel\emg_data_canonical -RunsRoot D:\Fmodel\runs -H5Samples 20
```

Expected: 14,498 rows, zero missing files, valid vocabularies, and no sampled H5 schema errors.

- [ ] **Step 5: Run the five-task real-data CPU smoke gate**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\powershell\03_SmokeTest.ps1 -Task All -CanonicalRoot D:\Fmodel\emg_data_canonical -RunsRoot D:\Fmodel\runs
```

Expected: five Lightning fast-dev runs complete with `data.num_workers=0`.

- [ ] **Step 6: Scan the publication boundary**

```powershell
rg -n -i "C:\\Users\\User\\Documents\\F_model|codex-ring-criss-cross-semg-fm" . -g '!docs/superpowers/**'
git status --short
git diff --check main..HEAD
git diff --stat main..HEAD
```

Expected: no retired path or source-worktree dependency outside implementation-history documents; no whitespace errors; no dataset, checkpoint, run-output, or credential file in the diff.

- [ ] **Step 7: Commit verification corrections only when needed**

```powershell
git add README.md docs scripts semg_fm tests
git commit -m "test: verify standalone training workflow"
```

- [ ] **Step 8: Push only the target feature branch**

```powershell
git remote -v
git branch --show-current
git push -u origin codex/standalone-training-workflow
```

Expected: `origin` is `https://github.com/Amazing-hw/metafmodel.git`, branch is `codex/standalone-training-workflow`, and Git reports the new upstream. Do not push any branch from the older `D:\Fmodel` repository.

- [ ] **Step 9: Report the immutable handoff**

Report the pushed branch, HEAD commit, full test result, real audit result, smoke result, and GitHub branch URL. State that full 100,000-step training and the complete multi-seed robustness grid were prepared but not run.
