# Five Standalone sEMG H5 Converters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the package-style converter implementation with exactly five directly executable, standalone Python scripts in `semg_h5`.

**Architecture:** Each converter contains all code required to read its source layout, validate `[T,16]` sEMG, write the canonical H5 record, and upsert `manifest.csv`. The common mechanics are intentionally duplicated so the five scripts have no local imports. The public H5 schema and task-specific labels remain unchanged.

**Tech Stack:** Python 3.10+, `h5py`, `numpy`, `pandas`, `pytest`, standard-library `argparse`, `csv`, `hashlib`, `json`, `pathlib`, `subprocess`.

---

## File structure

```text
semg_h5/
  convert_emg2pose.py
  convert_emg2qwerty.py
  convert_gni_discrete_gestures.py
  convert_gni_handwriting.py
  convert_gni_wrist.py
tests/
  conftest.py
  test_standalone_scripts.py
README.md
```

Every remaining `semg_h5/*.py` file is a user-facing converter. Test code loads a converter by absolute script path using `importlib.util.spec_from_file_location`, so passing tests prove no package import is required.

### Task 1: Direct-execution regression contract

**Files:**
- Create: `tests/test_standalone_scripts.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Write failing tests that import each converter by file path**

```python
SCRIPT_NAMES = (
    "convert_emg2pose.py",
    "convert_emg2qwerty.py",
    "convert_gni_discrete_gestures.py",
    "convert_gni_handwriting.py",
    "convert_gni_wrist.py",
)

def load_script(name):
    path = ROOT / "semg_h5" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_semg_h5_contains_exactly_five_converter_scripts():
    assert sorted(path.name for path in (ROOT / "semg_h5").glob("*.py")) == list(SCRIPT_NAMES)

def test_each_converter_has_no_local_semg_h5_imports():
    for name in SCRIPT_NAMES:
        source = (ROOT / "semg_h5" / name).read_text(encoding="utf-8")
        assert "from ." not in source
        assert "from semg_h5" not in source
```

- [ ] **Step 2: Run the direct-execution tests to verify RED**

Run: `python -m pytest tests/test_standalone_scripts.py -v`

Expected: FAIL because the directory contains shared modules and the five current converters use relative imports.

- [ ] **Step 3: Add direct script conversion tests using existing synthetic H5 fixtures**

```python
@pytest.mark.parametrize(
    ("script", "fixture_name", "expected_task", "expected_label"),
    [
        ("convert_emg2pose.py", "emg2pose_h5", "pose", "labels/joint_angles"),
        ("convert_emg2qwerty.py", "qwerty_h5", "typing", "labels/keystrokes"),
        ("convert_gni_discrete_gestures.py", "gni_discrete_gestures_h5", "discrete_gesture", "labels/events"),
        ("convert_gni_handwriting.py", "gni_handwriting_h5", "handwriting", "labels/prompts"),
        ("convert_gni_wrist.py", "gni_wrist_h5", "wrist_angle", "labels/targets/flexion_extension"),
    ],
)
def test_standalone_converter_writes_task_contract(script, fixture_name, expected_task, expected_label, request, tmp_path):
    module = load_script(script)
    source = request.getfixturevalue(fixture_name)
    rows = module.convert_file(source, tmp_path / "out", official_split="test")
    rows = rows if isinstance(rows, list) else [rows]
    for row in rows:
        with h5py.File(tmp_path / "out" / row["path"]) as h5:
            assert h5.attrs["task_id"] == expected_task
            assert h5["signal/emg"].shape[1] == 16
            assert expected_label in h5
```

### Task 2: Standalone emg2pose and qwerty scripts

**Files:**
- Modify: `semg_h5/convert_emg2pose.py`
- Modify: `semg_h5/convert_emg2qwerty.py`

- [ ] **Step 1: Replace all relative imports in `convert_emg2pose.py` with local helpers**

Embed `decode_value`, attribute reading, `[T,16]`/monotonic-time validation, canonical writer, pose label writer, SHA-256 calculation, manifest upsert, split lookup, output-path construction, `convert_file`, `run_conversion`, and `main` in the script. Retain the documented source requirement `emg2pose/timeseries` compound fields `emg`, `joint_angles`, `time`, with optional `no_ik_failure` and `stage`.

- [ ] **Step 2: Run the pose standalone test to verify GREEN**

Run: `python -m pytest tests/test_standalone_scripts.py -k emg2pose -v`

Expected: PASS with `/signal/emg [T,16]`, `/labels/joint_angles`, optional `/labels/no_ik_failure`, and a manifest row.

- [ ] **Step 3: Replace all relative imports in `convert_emg2qwerty.py` with local helpers**

Embed the same common writer/manifest mechanics plus qwerty-specific handling for `emg2qwerty/timeseries` fields `emg_left`, `emg_right`, `time`; JSON or structured keystroke/prompt labels; conversion of label timestamps to record-relative seconds; and creation of exactly two output rows with the same `pair_id`.

- [ ] **Step 4: Run the qwerty standalone test to verify GREEN**

Run: `python -m pytest tests/test_standalone_scripts.py -k emg2qwerty -v`

Expected: PASS with exactly left and right records, matching `pair_id`, and `/labels/keystrokes` in both files.

- [ ] **Step 5: Commit the two self-contained Meta converters**

```bash
git add semg_h5/convert_emg2pose.py semg_h5/convert_emg2qwerty.py tests/test_standalone_scripts.py tests/conftest.py
git commit -m "refactor: make Meta converters standalone"
```

### Task 3: Three standalone GNI scripts

**Files:**
- Modify: `semg_h5/convert_gni_discrete_gestures.py`
- Modify: `semg_h5/convert_gni_handwriting.py`
- Modify: `semg_h5/convert_gni_wrist.py`

- [ ] **Step 1: Embed the GNI common mechanics independently in every GNI script**

Each script must include direct imports only from the Python standard library, `h5py`, `numpy`, and `pandas`; read `data.attrs['task']`; locate `time` and a single 16-channel EMG field; write canonical H5/manifest output; and implement its own CLI. Each script must reject a source task other than its fixed expected task.

- [ ] **Step 2: Implement the three fixed label mappings**

```text
convert_gni_discrete_gestures.py: data.attrs['task'] == 'discrete_gestures'; prompts name/time -> labels/events(name, time_s)
convert_gni_handwriting.py:       data.attrs['task'] == 'handwriting'; prompts prompt/start/end -> labels/prompts(text, start_s, end_s)
convert_gni_wrist.py:             data.attrs['task'] == 'wrist'; numeric fields except time/emg -> labels/targets/<field>
```

Pandas HDF tables `prompts` and `stages` are attempted first; a structured HDF5 dataset is the fallback. Stage metadata is stored as a scalar `labels/stage` only when one row unambiguously supplies it; complete stage rows remain in source metadata.

- [ ] **Step 3: Run all GNI standalone tests to verify GREEN**

Run: `python -m pytest tests/test_standalone_scripts.py -k gni -v`

Expected: three PASS results with their distinct task ID and label paths.

- [ ] **Step 4: Commit the three self-contained GNI converters**

```bash
git add semg_h5/convert_gni_discrete_gestures.py semg_h5/convert_gni_handwriting.py semg_h5/convert_gni_wrist.py tests/test_standalone_scripts.py tests/conftest.py
git commit -m "refactor: make GNI converters standalone"
```

### Task 4: Remove package support code and revise documentation

**Files:**
- Delete: `semg_h5/__init__.py`
- Delete: `semg_h5/canonical.py`
- Delete: `semg_h5/manifest.py`
- Delete: `semg_h5/source_readers.py`
- Delete: `semg_h5/gni_conversion.py`
- Delete: `semg_h5/validate.py`
- Modify: `README.md`
- Modify: `tests/test_cli_integration.py`

- [ ] **Step 1: Write a failing CLI test that invokes a script by absolute file path**

```python
result = subprocess.run(
    [sys.executable, str(ROOT / "semg_h5" / "convert_emg2qwerty.py"),
     "--input-root", str(qwerty_h5.parent), "--output-root", str(output_root)],
    text=True, capture_output=True, check=False,
)
assert result.returncode == 0, result.stderr
assert len(pd.read_csv(output_root / "manifest.csv")) == 2
```

- [ ] **Step 2: Run the CLI test to verify GREEN before removal**

Run: `python -m pytest tests/test_cli_integration.py -v`

Expected: PASS because the standalone qwerty script has no package dependency.

- [ ] **Step 3: Delete only the six named support modules**

Use explicit paths and verify the target list before deletion. Do not remove the five converter scripts, tests, README, requirements, research report, or documentation.

- [ ] **Step 4: Replace README module commands with direct paths**

```powershell
python semg_h5\convert_emg2pose.py --input-root RAW_EMG2POSE --output-root CANONICAL_META
python semg_h5\convert_emg2qwerty.py --input-root RAW_EMG2QWERTY --output-root CANONICAL_META
python semg_h5\convert_gni_discrete_gestures.py --input-root RAW_GNI_DISCRETE --output-root CANONICAL_META
python semg_h5\convert_gni_handwriting.py --input-root RAW_GNI_HANDWRITING --output-root CANONICAL_META
python semg_h5\convert_gni_wrist.py --input-root RAW_GNI_WRIST --output-root CANONICAL_META
```

Remove the separate `python -m semg_h5.validate` documentation and state that each converter validates its source before writing.

- [ ] **Step 5: Commit the exact-five-script directory**

```bash
git add -A semg_h5 README.md tests/test_cli_integration.py tests/test_standalone_scripts.py tests/conftest.py
git commit -m "refactor: retain five standalone sEMG converters"
```

### Task 5: Final verification

**Files:**
- Verify: `semg_h5/*.py`, test suite, README

- [ ] **Step 1: Run the entire suite**

Run: `python -m pytest -q`

Expected: exit code 0.

- [ ] **Step 2: Verify exact file count and absence of package imports**

Run: `rg --files semg_h5 -g '*.py'; rg -n 'from \.|from semg_h5' semg_h5`

Expected: exactly the five named files; second command prints no matches.

- [ ] **Step 3: Inspect all CLI help screens**

Run: `python semg_h5/convert_emg2pose.py --help` and repeat for the other four scripts.

Expected: each exits 0 and exposes `--input-root`, `--output-root`, `--split`, `--splits-csv`, and `--overwrite`.

- [ ] **Step 4: Confirm no generated test artifacts are tracked**

Run: `git status --short`

Expected: no `.pytest_tmp`, `__pycache__`, converted `.h5`, or generated `manifest.csv` files are tracked.

## Plan self-review

- Spec coverage: Tasks 2 and 3 replace every converter with a standalone implementation while retaining every task label; Task 4 removes exactly the six shared modules; Tasks 1, 4, and 5 prove direct execution and the five-file boundary.
- Placeholder scan: the plan contains no deferred or unspecified implementation behavior.
- Type consistency: all scripts expose `convert_file(source_path, output_root, official_split='unknown', overwrite=False)`; qwerty alone returns a list of rows, while the other four return one row.
