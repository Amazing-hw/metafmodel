# Split Provenance and Safe Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure every post-conversion run uses an explicit, provenance-preserving split manifest and cannot silently reuse an existing output directory.

**Architecture:** Add shared PowerShell helpers for split resolution and output-directory protection. Each stage script consumes the same split contract, while the pipeline resolves it once and forwards the exact path. Documentation describes only commands that select a protocol/seed or an explicit split path.

**Tech Stack:** PowerShell 7 or Windows PowerShell 5.1, Python pytest, Hydra, Lightning.

---

### Task 1: Test and implement shared split resolution

**Files:**
- Modify: `scripts/powershell/Common.ps1`
- Modify: `scripts/powershell/03_SmokeTest.ps1`
- Modify: `scripts/powershell/04_TrainBaseline.ps1`
- Modify: `scripts/powershell/05_Pretrain.ps1`
- Modify: `scripts/powershell/06_Finetune.ps1`
- Modify: `scripts/powershell/07_Evaluate.ps1`
- Modify: `scripts/powershell/08_Robustness.ps1`
- Test: `tests/test_powershell_workflow.py`

- [ ] **Step 1: Add failing protocol-selection tests**

Add one test that runs `03_SmokeTest.ps1 -Protocol StrictSubject -Seed 17 -DryRun` and asserts `splits.strict-subject.seed-17.csv` appears in every command. Add one test that calls the same script with neither `SplitsPath` nor protocol/seed and asserts a non-zero exit with `split selection` in output. Repeat one protocol assertion for `05_Pretrain.ps1`.

- [ ] **Step 2: Run the focused tests**

```powershell
python -m pytest tests\test_powershell_workflow.py -q
```

Expected: the new tests fail because stage scripts still default to `splits.csv`.

- [ ] **Step 3: Add the shared helper**

Add `Resolve-SplitManifest` to `Common.ps1` with parameters `CanonicalRoot`, `SplitsPath`, `Protocol`, and `Seed`. It returns an absolute existing file. Its behavior is:

```powershell
if ($SplitsPath) { return (Resolve-FullPath $SplitsPath) }
if (-not $Protocol -or $Seed -lt 0) { throw 'split selection requires -SplitsPath or both -Protocol and -Seed' }
$slug = if ($Protocol -eq 'Official') { 'official' } else { 'strict-subject' }
$candidate = Join-Path $CanonicalRoot "splits.$slug.seed-$Seed.csv"
Assert-File $candidate 'selected split manifest'
return $candidate
```

- [ ] **Step 4: Apply the shared parameters to all seven stage scripts**

Use this parameter shape in every listed stage script:

```powershell
[string]$SplitsPath = '',
[ValidateSet('Official', 'StrictSubject')][string]$Protocol = '',
[Nullable[int]]$SplitSeed = $null
```

Replace every fallback to `splits.csv` with `Resolve-SplitManifest`. Pass the returned path through `data.manifest_path` unchanged.

- [ ] **Step 5: Run focused green tests**

```powershell
python -m pytest tests\test_powershell_workflow.py -q
```

Expected: protocol-selected dry-runs name the seed-qualified split, and missing selection fails before Python launches.

- [ ] **Step 6: Commit shared split provenance**

```powershell
git add scripts/powershell/Common.ps1 scripts/powershell/03_SmokeTest.ps1 scripts/powershell/04_TrainBaseline.ps1 scripts/powershell/05_Pretrain.ps1 scripts/powershell/06_Finetune.ps1 scripts/powershell/07_Evaluate.ps1 scripts/powershell/08_Robustness.ps1 tests/test_powershell_workflow.py
git commit -m "fix: require explicit training split selection"
```

### Task 2: Protect stage output directories

**Files:**
- Modify: `scripts/powershell/Common.ps1`
- Modify: `scripts/powershell/03_SmokeTest.ps1`
- Modify: `scripts/powershell/04_TrainBaseline.ps1`
- Modify: `scripts/powershell/05_Pretrain.ps1`
- Modify: `scripts/powershell/06_Finetune.ps1`
- Modify: `scripts/powershell/07_Evaluate.ps1`
- Modify: `scripts/powershell/08_Robustness.ps1`
- Test: `tests/test_powershell_workflow.py`

- [ ] **Step 1: Add failing collision tests**

Create a temporary output directory, call `03_SmokeTest.ps1` with that directory as `RunsRoot` and a protocol-selected split, and assert non-dry execution stops before Python launch. Repeat with `-Force` and assert the command reaches its Python invocation.

- [ ] **Step 2: Implement the guard**

Add `Assert-OutputDirectoryAvailable` to `Common.ps1`:

```powershell
function Assert-OutputDirectoryAvailable([string]$Path, [switch]$Force, [switch]$DryRun) {
    if ((Test-Path -LiteralPath $Path) -and -not $Force) { throw "output directory already exists; pass -Force to reuse it: $Path" }
    if (-not $DryRun -and -not (Test-Path -LiteralPath $Path)) { New-Item -ItemType Directory -Path $Path | Out-Null }
}
```

Add `Force` to every stage script and call this helper for each final run directory before Python starts. Include a split slug in the directory path: protocol plus seed, or `custom-` plus the first twelve characters of the SHA-256 split-path digest.

- [ ] **Step 3: Run collision and parser tests**

```powershell
python -m pytest tests\test_powershell_workflow.py -q
```

Expected: existing output fails without Force; dry-run does not create a directory; Force permits Python command construction.

- [ ] **Step 4: Commit output safety**

```powershell
git add scripts/powershell/Common.ps1 scripts/powershell/03_SmokeTest.ps1 scripts/powershell/04_TrainBaseline.ps1 scripts/powershell/05_Pretrain.ps1 scripts/powershell/06_Finetune.ps1 scripts/powershell/07_Evaluate.ps1 scripts/powershell/08_Robustness.ps1 tests/test_powershell_workflow.py
git commit -m "fix: guard training output directories"
```

### Task 3: Forward one split through pipeline stages

**Files:**
- Modify: `scripts/powershell/02_PrepareSplits.ps1`
- Modify: `scripts/powershell/RunPipeline.ps1`
- Test: `tests/test_powershell_workflow.py`

- [ ] **Step 1: Add a failing pipeline test**

Run `RunPipeline.ps1 -Stages Splits,Smoke -Protocol StrictSubject -Seed 17 -DryRun`. Assert both printed child commands contain `splits.strict-subject.seed-17.csv`; assert no command contains an unqualified `splits.csv`.

- [ ] **Step 2: Implement pipeline forwarding**

Add `Protocol`, `Seed`, `SplitsPath`, and `Force` parameters to `RunPipeline.ps1`. For `Splits`, forward protocol and seed. For every later data-consuming stage, compute the exact path using `Resolve-SplitManifest` and pass it as `SplitsPath`. Reject a custom `SplitsPath` combined with the `Splits` stage because a custom file is already prepared.

- [ ] **Step 3: Run focused tests**

```powershell
python -m pytest tests\test_powershell_workflow.py -q
```

Expected: one pipeline invocation uses one exact split path at every stage.

- [ ] **Step 4: Commit pipeline propagation**

```powershell
git add scripts/powershell/02_PrepareSplits.ps1 scripts/powershell/RunPipeline.ps1 tests/test_powershell_workflow.py
git commit -m "fix: propagate selected split through pipeline"
```

### Task 4: Publish the detailed post-conversion runbook

**Files:**
- Modify: `README.md`
- Modify: `docs/META16_TRAINING_WORKFLOW.md`
- Modify: `docs/POWERSHELL_SCRIPT_REFERENCE.md`
- Modify: `tests/test_documented_commands.py`

- [ ] **Step 1: Add failing documentation assertions**

Require every run command in the two documents to include either `SplitsPath` or protocol plus seed. Require documentation of `Force`, collision behavior, resume output selection, Official versus StrictSubject interpretation, and all nine operational stages.

- [ ] **Step 2: Write the complete runbook**

Document the exact ordered commands: conversion completion gate, audit, Official or StrictSubject preparation, five-task smoke, baseline matrix, pilot then full pretraining, fine-tuning budgets, evaluation, robustness, resume, artifact collection, and reporting. Use `D:\Fmodel` absolute examples and name inputs, outputs, pass gates, and next stage for each command.

- [ ] **Step 3: Run documentation and PowerShell checks**

```powershell
python -m pytest tests\test_documented_commands.py tests\test_powershell_workflow.py -q
Get-ChildItem scripts\powershell\*.ps1 | ForEach-Object { $t=$null;$e=$null;[System.Management.Automation.Language.Parser]::ParseFile($_.FullName,[ref]$t,[ref]$e)>$null;if($e.Count){throw $_.FullName} }
```

Expected: tests pass and no script parser error prints.

- [ ] **Step 4: Commit documentation**

```powershell
git add README.md docs/META16_TRAINING_WORKFLOW.md docs/POWERSHELL_SCRIPT_REFERENCE.md tests/test_documented_commands.py
git commit -m "docs: add safe post-conversion runbook"
```

### Task 5: Verify the branch and push the fixes

**Files:**
- Modify only files required by observed verification failures.

- [ ] **Step 1: Run complete verification**

```powershell
python -m pytest -q
git diff --check main..HEAD
rg -n -i "C:\\Users\\User\\Documents\\F_model" README.md docs scripts
git status --short
```

Expected: all tests pass, no whitespace or retired-root match prints, and only intended files appear before the final commit.

- [ ] **Step 2: Push the feature branch**

```powershell
git push origin codex/standalone-training-workflow
```

Expected: GitHub receives only the target repository feature branch.
