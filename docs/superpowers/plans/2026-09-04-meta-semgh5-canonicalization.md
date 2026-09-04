# Meta sEMG Canonical H5 Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create five tested CLI converters that transform the Meta/CTRL-labs public sEMG HDF5 corpora into a single per-wrist canonical H5 schema plus a provenance manifest.

**Architecture:** A shared `semg_h5.canonical` module owns schema validation and all canonical H5 writes. Source-specific readers only extract documented source arrays and labels into `CanonicalRecord` values. Each output file contains one `[T,16]` wrist stream; qwerty creates two linked outputs using `pair_id`. A CSV manifest mirrors selection/provenance fields so dataloaders can filter without opening every H5.

**Tech Stack:** Python 3.10+, `h5py`, `numpy`, `pandas`, `pytest`; standard library `argparse`, `json`, `hashlib`, `csv`, `pathlib`.

---

## File structure

```text
semg_h5/
  __init__.py
  canonical.py
  manifest.py
  source_readers.py
  convert_emg2pose.py
  convert_emg2qwerty.py
  convert_gni_discrete_gestures.py
  convert_gni_handwriting.py
  convert_gni_wrist.py
  validate.py
tests/
  conftest.py
  test_canonical.py
  test_manifest.py
  test_convert_emg2pose.py
  test_convert_emg2qwerty.py
  test_convert_gni.py
README.md
requirements.txt
```

`canonical.py` has the only code path allowed to create `/signal`, `/labels`, and `/meta` in a canonical file. `manifest.py` serializes a stable row per output H5. `source_readers.py` supplies input-layout functions used by the five short CLI modules. Tests construct minimal H5 fixtures that mimic the declared layouts, so no public dataset download is necessary to run the suite.

### Task 1: Package skeleton and schema contract tests

**Files:**
- Create: `requirements.txt`
- Create: `semg_h5/__init__.py`
- Create: `semg_h5/canonical.py`
- Create: `tests/conftest.py`
- Create: `tests/test_canonical.py`

- [ ] **Step 1: Write the failing canonical-record test**

```python
# tests/test_canonical.py
import numpy as np
import pytest

from semg_h5.canonical import CanonicalRecord, validate_record


def test_validate_record_accepts_a_16_channel_monotonic_stream():
    record = CanonicalRecord(
        dataset_id="emg2pose", task_id="pose", subject_id="emg2pose:user_1",
        session_id="emg2pose:session_1", side="left", pair_id="",
        official_split="train", source_file="source.h5", emg=np.zeros((4, 16)),
        time_s=np.array([0.0, 0.0005, 0.001, 0.0015]), labels={}, source_meta={},
        preprocessing={"converter_transformations": []},
    )
    validate_record(record)


def test_validate_record_rejects_a_non_16_channel_stream():
    record = CanonicalRecord.minimal(emg=np.zeros((4, 15)), time_s=np.arange(4) / 2000)
    with pytest.raises(ValueError, match="16 channels"):
        validate_record(record)
```

- [ ] **Step 2: Run the test to verify RED**

Run: `pytest tests/test_canonical.py -v`

Expected: collection fails because `semg_h5.canonical` does not exist.

- [ ] **Step 3: Create the minimal package and validation implementation**

```python
# semg_h5/canonical.py
from dataclasses import dataclass
from typing import Any
import numpy as np

SCHEMA_VERSION = "1.0"
SAMPLE_RATE_HZ = 2000

@dataclass
class CanonicalRecord:
    dataset_id: str
    task_id: str
    subject_id: str
    session_id: str
    side: str
    pair_id: str
    official_split: str
    source_file: str
    emg: np.ndarray
    time_s: np.ndarray
    labels: dict[str, Any]
    source_meta: dict[str, Any]
    preprocessing: dict[str, Any]

    @classmethod
    def minimal(cls, emg: np.ndarray, time_s: np.ndarray) -> "CanonicalRecord":
        return cls("unknown", "unknown", "unknown", "unknown", "unknown", "", "unknown",
                   "", emg, time_s, {}, {}, {"converter_transformations": []})

def validate_record(record: CanonicalRecord) -> None:
    if record.emg.ndim != 2 or record.emg.shape[1] != 16:
        raise ValueError("EMG must have shape [T, 16] with exactly 16 channels")
    if record.time_s.ndim != 1 or len(record.time_s) != len(record.emg):
        raise ValueError("time_s must be a length-T vector matching EMG")
    if not np.isfinite(record.emg).all() or not np.isfinite(record.time_s).all():
        raise ValueError("EMG and time_s must be finite")
    if len(record.time_s) > 1 and np.any(np.diff(record.time_s) < 0):
        raise ValueError("time_s must be monotonic")
```

```text
# requirements.txt
h5py>=3.8
numpy>=1.24
pandas>=1.5
pytest>=7.0
```

- [ ] **Step 4: Run the canonical tests to verify GREEN**

Run: `pytest tests/test_canonical.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit the first tested contract**

```bash
git add requirements.txt semg_h5/__init__.py semg_h5/canonical.py tests/test_canonical.py
git commit -m "feat: define canonical sEMG record contract"
```

### Task 2: Canonical H5 writer and reader validation

**Files:**
- Modify: `semg_h5/canonical.py`
- Modify: `tests/test_canonical.py`

- [ ] **Step 1: Write the failing write/read round-trip test**

```python
def test_write_record_creates_required_groups_and_relative_clock(tmp_path):
    from semg_h5.canonical import read_canonical_header, write_record
    record = CanonicalRecord.minimal(
        emg=np.arange(64, dtype=np.float64).reshape(4, 16),
        time_s=np.array([10.0, 10.0005, 10.001, 10.0015]),
    )
    output = tmp_path / "record.h5"
    write_record(output, record)
    header = read_canonical_header(output)
    assert header["schema_version"] == "1.0"
    assert header["emg_shape"] == (4, 16)
    assert np.allclose(header["time_s"], [0.0, 0.0005, 0.001, 0.0015])
```

- [ ] **Step 2: Run the test to verify RED**

Run: `pytest tests/test_canonical.py::test_write_record_creates_required_groups_and_relative_clock -v`

Expected: FAIL because `write_record` and `read_canonical_header` are not defined.

- [ ] **Step 3: Implement the writer without physiological transformations**

Implement `write_record(path, record, overwrite=False)` with these exact behaviors:

```python
with h5py.File(path, "w") as h5:
    h5.attrs.update({
        "schema_version": SCHEMA_VERSION,
        "dataset_id": record.dataset_id,
        "task_id": record.task_id,
        "subject_id": record.subject_id,
        "session_id": record.session_id,
        "side": record.side,
        "pair_id": record.pair_id,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "official_split": record.official_split,
        "source_file": record.source_file,
        "preprocessing_json": json.dumps(record.preprocessing, sort_keys=True),
    })
    signal = h5.create_group("signal")
    signal.create_dataset("emg", data=record.emg.astype(np.float32), compression="gzip")
    signal.create_dataset("time_s", data=record.time_s.astype(np.float64) - record.time_s[0])
```

Write `source_json` as a UTF-8 scalar dataset, write `/meta/quality/sample_count`, `/meta/quality/duration_s`, `/meta/quality/nan_count`, and `/meta/quality/channel_rms`, then call a label-writer dispatch function. Reject existing paths unless `overwrite=True`.

- [ ] **Step 4: Run the writer tests to verify GREEN**

Run: `pytest tests/test_canonical.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit the H5 writer**

```bash
git add semg_h5/canonical.py tests/test_canonical.py
git commit -m "feat: write validated canonical sEMG H5 records"
```

### Task 3: Label serialization, manifest, and tree validator

**Files:**
- Create: `semg_h5/manifest.py`
- Create: `semg_h5/validate.py`
- Create: `tests/test_manifest.py`
- Modify: `semg_h5/canonical.py`

- [ ] **Step 1: Write failing manifest and task-label tests**

```python
def test_manifest_row_exposes_task_selection_and_pairing(tmp_path):
    from semg_h5.manifest import append_manifest_row, read_manifest
    row = {"path": "typing/user_1/session_1_left.h5", "dataset_id": "emg2qwerty",
           "task_id": "typing", "subject_id": "emg2qwerty:user_1",
           "session_id": "emg2qwerty:session_1", "side": "left", "pair_id": "pair_1",
           "official_split": "train", "sample_count": 4, "source_sha256": "abc"}
    append_manifest_row(tmp_path / "manifest.csv", row)
    assert read_manifest(tmp_path / "manifest.csv").to_dict("records") == [row]
```

```python
def test_typing_labels_round_trip_as_structured_tables(tmp_path):
    from semg_h5.canonical import CanonicalRecord, write_record
    record = CanonicalRecord.minimal(np.zeros((4, 16)), np.arange(4) / 2000)
    record.task_id = "typing"
    record.labels = {"keystrokes": [{"key": "a", "press_s": 0.0005, "release_s": 0.001}],
                     "prompts": [{"text": "a", "start_s": 0.0, "end_s": 0.0015}]}
    write_record(tmp_path / "typing.h5", record)
    with h5py.File(tmp_path / "typing.h5") as h5:
        assert h5["labels/keystrokes"][0]["key"].decode() == "a"
        assert h5["labels/prompts"][0]["text"].decode() == "a"
```

- [ ] **Step 2: Run the tests to verify RED**

Run: `pytest tests/test_manifest.py tests/test_canonical.py::test_typing_labels_round_trip_as_structured_tables -v`

Expected: FAIL because manifest and label serialization do not exist.

- [ ] **Step 3: Implement stable manifest and label writers**

Implement `append_manifest_row` by reading existing CSV, replacing an existing row with the same `path`, sorting by `path`, and writing atomically via `manifest.csv.tmp`. Fix the manifest field order to:

```python
MANIFEST_COLUMNS = ["path", "dataset_id", "task_id", "subject_id", "session_id", "side",
                    "pair_id", "official_split", "sample_count", "duration_s",
                    "source_file", "source_sha256"]
```

Implement H5 label table datasets with UTF-8 variable-length strings and named numeric fields. Support exactly these label mappings:

```text
typing:             keystrokes(key, press_s, release_s), prompts(text, start_s, end_s)
discrete_gesture:   events(name, time_s), stage(optional UTF-8 scalar)
handwriting:        prompts(text, start_s, end_s)
pose:               joint_angles, no_ik_failure(optional), stage(optional UTF-8 scalar)
wrist_angle:        target arrays under labels/targets/<source field>, stage(optional UTF-8 scalar)
```

Implement `python -m semg_h5.validate --input PATH` to recursively validate all `*.h5`, check the manifest paths, and exit non-zero with every invalid path listed.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `pytest tests/test_manifest.py tests/test_canonical.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit manifest and label support**

```bash
git add semg_h5 tests/test_manifest.py tests/test_canonical.py
git commit -m "feat: add task labels manifest and canonical validator"
```

### Task 4: emg2pose source reader and converter

**Files:**
- Create: `semg_h5/source_readers.py`
- Create: `semg_h5/convert_emg2pose.py`
- Create: `tests/test_convert_emg2pose.py`

- [ ] **Step 1: Write the failing emg2pose conversion test**

```python
def test_convert_emg2pose_preserves_pose_labels_and_source_split(tmp_path, emg2pose_h5):
    from semg_h5.convert_emg2pose import convert_file
    manifest_row = convert_file(emg2pose_h5, tmp_path / "out", official_split="val")
    with h5py.File(tmp_path / "out" / manifest_row["path"]) as h5:
        assert h5.attrs["dataset_id"] == "emg2pose"
        assert h5.attrs["task_id"] == "pose"
        assert h5.attrs["official_split"] == "val"
        assert h5["signal/emg"].shape == (4, 16)
        assert h5["labels/joint_angles"].shape == (4, 3)
        assert h5["labels/no_ik_failure"].shape == (4,)
```

- [ ] **Step 2: Run the test to verify RED**

Run: `pytest tests/test_convert_emg2pose.py -v`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement documented emg2pose extraction**

Read group `emg2pose/timeseries` as a compound dataset with fields `emg`, `joint_angles`, and `time`. Read metadata from the source group/attributes if present, otherwise derive namespaced IDs from the source relative path. For each source H5, produce exactly one canonical record. Preserve optional source `no_ik_failure` and `stage`; set `side` from source metadata when available. The CLI must enumerate input `*.h5`, join official split from a supplied metadata CSV when present, write each record, append one manifest row, and support `--split` filtering.

- [ ] **Step 4: Run emg2pose tests to verify GREEN**

Run: `pytest tests/test_convert_emg2pose.py -v`

Expected: 1 passed.

- [ ] **Step 5: Commit the emg2pose converter**

```bash
git add semg_h5/source_readers.py semg_h5/convert_emg2pose.py tests/test_convert_emg2pose.py
git commit -m "feat: convert emg2pose into canonical H5"
```

### Task 5: emg2qwerty bilateral converter

**Files:**
- Create: `semg_h5/convert_emg2qwerty.py`
- Create: `tests/test_convert_emg2qwerty.py`
- Modify: `semg_h5/source_readers.py`

- [ ] **Step 1: Write the failing bilateral-pair test**

```python
def test_convert_qwerty_emits_one_linked_record_per_wrist(tmp_path, qwerty_h5):
    from semg_h5.convert_emg2qwerty import convert_file
    rows = convert_file(qwerty_h5, tmp_path / "out", official_split="train")
    assert len(rows) == 2
    assert {row["side"] for row in rows} == {"left", "right"}
    assert rows[0]["pair_id"] == rows[1]["pair_id"]
    for row in rows:
        with h5py.File(tmp_path / "out" / row["path"]) as h5:
            assert h5["signal/emg"].shape == (4, 16)
            assert h5["labels/keystrokes"].shape[0] == 1
```

- [ ] **Step 2: Run the test to verify RED**

Run: `pytest tests/test_convert_emg2qwerty.py -v`

Expected: FAIL because the qwerty module does not exist.

- [ ] **Step 3: Implement documented qwerty extraction**

Read `emg2qwerty/timeseries` compound fields `emg_left`, `emg_right`, and `time`. Read keylogger and prompt metadata from the source group metadata JSON/attributes. Convert keylogger absolute timestamps and prompt times to the same record-relative reference as `time_s`. Validate that both sides have `[T,16]`, identical lengths, and aligned time vectors before writing outputs. Give both records the deterministic identifier `emg2qwerty:<source-session>` and `pair_id` `emg2qwerty:<source-session>`. Write labels into both output files so either paired typing record is self-describing.

- [ ] **Step 4: Run qwerty tests to verify GREEN**

Run: `pytest tests/test_convert_emg2qwerty.py -v`

Expected: 1 passed.

- [ ] **Step 5: Commit the qwerty converter**

```bash
git add semg_h5/convert_emg2qwerty.py semg_h5/source_readers.py tests/test_convert_emg2qwerty.py
git commit -m "feat: convert bilateral emg2qwerty sessions"
```

### Task 6: Generic-neuromotor-interface source reader and three task CLIs

**Files:**
- Create: `semg_h5/convert_gni_discrete_gestures.py`
- Create: `semg_h5/convert_gni_handwriting.py`
- Create: `semg_h5/convert_gni_wrist.py`
- Create: `tests/test_convert_gni.py`
- Modify: `semg_h5/source_readers.py`

- [ ] **Step 1: Write failing parametrized tests for each GNI task**

```python
@pytest.mark.parametrize(
    ("converter_module", "task_id", "label_path"),
    [
        ("convert_gni_discrete_gestures", "discrete_gesture", "labels/events"),
        ("convert_gni_handwriting", "handwriting", "labels/prompts"),
        ("convert_gni_wrist", "wrist_angle", "labels/targets/flexion_extension"),
    ],
)
def test_convert_gni_task_preserves_16ch_stream_and_its_labels(
    tmp_path, gni_h5_by_task, converter_module, task_id, label_path
):
    module = importlib.import_module(f"semg_h5.{converter_module}")
    row = module.convert_file(gni_h5_by_task[task_id], tmp_path / "out", official_split="test")
    with h5py.File(tmp_path / "out" / row["path"]) as h5:
        assert h5.attrs["task_id"] == task_id
        assert h5["signal/emg"].shape == (4, 16)
        assert label_path in h5
```

- [ ] **Step 2: Run the test to verify RED**

Run: `pytest tests/test_convert_gni.py -v`

Expected: FAIL because no GNI converter modules exist.

- [ ] **Step 3: Implement the shared GNI reader**

Read source `data` as a compound dataset and extract `time` and its one 16-channel EMG field. Read task name from `data.attrs["task"]`. Read a `prompts` pandas HDF table when present and a `stages` table when present. Dispatch labels as follows:

```text
discrete_gestures: prompts columns name/time -> labels/events(name, time_s)
handwriting:       prompts columns prompt/start/end -> labels/prompts(text, start_s, end_s)
wrist:             every numeric data field other than time/emg -> labels/targets/<field>
```

For the wrist fixture and output contract, normalize the source field name `flexion_extension` to `flexion_extension`; preserve all other source numeric target fields under their safe ASCII field name. Require target arrays to have length `T`. Add `stage` as a UTF-8 scalar when a single matching stage is available, otherwise preserve complete source stage rows in `source_json`.

- [ ] **Step 4: Implement three thin CLIs**

Each module must call a shared `run_gni_conversion(task_name, input_root, output_root, split, overwrite)` with its fixed task string. The generic runner must reject a source H5 whose `data.attrs["task"]` differs from the invoked task. Write `dataset_id` values `gni_discrete_gestures`, `gni_handwriting`, and `gni_wrist`; write sides as `dominant` unless an explicit source side exists.

- [ ] **Step 5: Run GNI tests to verify GREEN**

Run: `pytest tests/test_convert_gni.py -v`

Expected: 3 passed.

- [ ] **Step 6: Commit all three GNI converters**

```bash
git add semg_h5/convert_gni_discrete_gestures.py semg_h5/convert_gni_handwriting.py semg_h5/convert_gni_wrist.py semg_h5/source_readers.py tests/test_convert_gni.py
git commit -m "feat: convert generic neuromotor interface tasks"
```

### Task 7: CLI integration tests and user documentation

**Files:**
- Modify: `tests/test_canonical.py`
- Create: `tests/test_cli_integration.py`
- Create: `README.md`

- [ ] **Step 1: Write a failing end-to-end CLI test**

```python
def test_converter_cli_writes_a_valid_tree_and_manifest(tmp_path, qwerty_h5):
    result = subprocess.run(
        [sys.executable, "-m", "semg_h5.convert_emg2qwerty", "--input-root", str(qwerty_h5.parent),
         "--output-root", str(tmp_path / "out")], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    validation = subprocess.run(
        [sys.executable, "-m", "semg_h5.validate", "--input", str(tmp_path / "out")],
        text=True, capture_output=True, check=False
    )
    assert validation.returncode == 0, validation.stderr
    assert len(pd.read_csv(tmp_path / "out" / "manifest.csv")) == 2
```

- [ ] **Step 2: Run the test to verify RED**

Run: `pytest tests/test_cli_integration.py -v`

Expected: FAIL until all converters expose their CLI parser and write a manifest.

- [ ] **Step 3: Implement consistent argparse entrypoints and README**

Document these exact commands:

```bash
python -m semg_h5.convert_emg2pose --input-root RAW_EMG2POSE --output-root CANONICAL_META
python -m semg_h5.convert_emg2qwerty --input-root RAW_EMG2QWERTY --output-root CANONICAL_META
python -m semg_h5.convert_gni_discrete_gestures --input-root RAW_GNI_DISCRETE --output-root CANONICAL_META
python -m semg_h5.convert_gni_handwriting --input-root RAW_GNI_HANDWRITING --output-root CANONICAL_META
python -m semg_h5.convert_gni_wrist --input-root RAW_GNI_WRIST --output-root CANONICAL_META
python -m semg_h5.validate --input CANONICAL_META
```

README must include canonical tree layout, manifest columns, the qwerty `pair_id` reconstruction example, no-refilter/no-rescale policy, `--split`, and `--overwrite` semantics.

- [ ] **Step 4: Run integration and full test suites to verify GREEN**

Run: `pytest -q`

Expected: all tests pass with no failures.

- [ ] **Step 5: Commit documentation and integration coverage**

```bash
git add README.md tests/test_cli_integration.py
git commit -m "docs: document canonical Meta sEMG conversion workflow"
```

### Task 8: Final verification and acceptance audit

**Files:**
- Verify: all files created above

- [ ] **Step 1: Run fresh full verification**

Run: `pytest -q`

Expected: exit code 0 and all tests passed.

- [ ] **Step 2: Inspect converter command help**

Run:

```bash
python -m semg_h5.convert_emg2pose --help
python -m semg_h5.convert_emg2qwerty --help
python -m semg_h5.convert_gni_discrete_gestures --help
python -m semg_h5.convert_gni_handwriting --help
python -m semg_h5.convert_gni_wrist --help
python -m semg_h5.validate --help
```

Expected: every command exits 0 and exposes `--input-root`, `--output-root`, `--split`, and `--overwrite` where applicable.

- [ ] **Step 3: Audit acceptance criteria**

Verify against the approved design:

```text
[ ] all five converter modules exist
[ ] all emitted records have /signal/emg [T,16] and /signal/time_s [T]
[ ] qwerty emits paired left/right rows
[ ] every task preserves its required labels
[ ] manifest contains task, split, IDs, side, pair_id and provenance hash
[ ] validator catches malformed canonical H5 and manifest inconsistency
[ ] full pytest suite passes
```

- [ ] **Step 4: Commit any final audit-only documentation change, if needed**

```bash
git status --short
```

Expected: only intentionally untracked research artifacts remain; no generated test H5 files or output manifests are tracked.

## Plan self-review

- Spec coverage: Tasks 1–3 implement the shared contract, labels, manifest and validator; Tasks 4–6 cover every approved source dataset; Task 7 documents and exercises the public interfaces; Task 8 maps directly to every acceptance criterion.
- Placeholder scan: no incomplete or deferred behavior remains.
- Type consistency: every converter produces `CanonicalRecord`; all canonical writes go through `write_record`; manifest rows use the fixed `MANIFEST_COLUMNS` list.
