# Meta sEMG Canonical H5 Conversion Design

## Goal

Convert the five public Meta/CTRL-labs sEMG task corpora into a common, loss-aware per-wrist HDF5 format that can be consumed by one generic dataloader while retaining task-specific labels and the source benchmark partitions needed for downstream evaluation.

## Scope

The converter accepts already-downloaded source data and creates canonical output. It does not download source archives, train a model, create fixed training windows, alter source files, or infer missing labels.

The supported inputs are `emg2pose`, `emg2qwerty`, and the three `generic-neuromotor-interface` tasks: `discrete_gestures`, `handwriting`, and `wrist`.

## Chosen architecture

Each output H5 represents one physical wrist stream. Its common waveform arrays are always `/signal/emg` with shape `[T, 16]` and `/signal/time_s` with shape `[T]`. Single-wrist input corpora therefore create one output per source recording. `emg2qwerty` creates two output files from each bilateral session, one for the left wrist and one for the right wrist, and gives both a shared `pair_id`.

This makes raw self-supervised sampling exact and uniform. A typing dataloader reconstructs bilateral input by joining left and right output rows on `pair_id`; it must never concatenate arbitrary records merely because their timestamps overlap.

All labels remain in `/labels`, with one task-specific layout per `task_id`. All mutable provenance and task-selection information is duplicated in a corpus-level CSV manifest. The generic dataloader need only read common root attributes and `/signal`; task heads read their corresponding label group.

## Canonical H5 contract

Every output file must contain these root attributes:

```text
schema_version = "1.0"
dataset_id     = one of {emg2pose, emg2qwerty, gni_discrete_gestures,
                         gni_handwriting, gni_wrist}
task_id        = one of {pose, typing, discrete_gesture, handwriting, wrist_angle}
subject_id     = source-namespaced anonymous identifier
session_id     = source-namespaced session/recording identifier
side           = "left", "right", "dominant", or "unknown"
pair_id        = non-empty only for bilateral qwerty records
sample_rate_hz = 2000
official_split = source train/val/test value, or "unknown"
source_file    = input-relative path
preprocessing_json = a JSON object describing known source processing and
                     transformations applied by this converter
```

Every output file must contain:

```text
/signal/emg       float32 [T, 16]
/signal/time_s    float64 [T]
/meta/source_json UTF-8 JSON scalar
/meta/quality     float64 or integer scalar datasets for sample count,
                  duration, NaN count, and per-channel RMS
/labels/...       task-specific datasets listed below
```

The converter must preserve timestamps after translating them to seconds relative to the first sample in that output record. The original absolute reference, when available, stays in `source_json`. Signals are cast to `float32` but are not refiltered, re-scaled, sign-flipped, rotated, normalized, resampled, or clipped. `preprocessing_json` reports source-declared processing and `converter_transformations: []`.

## Task label contracts

| task_id | Required `/labels` content |
|---|---|
| `pose` | `joint_angles [T,D]`, optional `no_ik_failure [T]`, plus `stage` UTF-8 scalar |
| `typing` | `keystrokes` table with key, press time and release time; `prompts` table with text, start and end time |
| `discrete_gesture` | event table with gesture name and event time; stage metadata when supplied |
| `handwriting` | prompt table with text, start and end time |
| `wrist_angle` | every source-provided continuous wrist angle array aligned to `time_s`; stage metadata when supplied |

Event and prompt times are expressed in the same relative seconds clock as `/signal/time_s`. Unknown/missing optional source fields are represented by an absent optional dataset, never fabricated values.

## Files

```text
semg_h5/
  __init__.py
  canonical.py                    # schema constants, H5 writer, validation helpers
  source_readers.py               # source-shape readers and source metadata extraction
  manifest.py                     # manifest row creation and atomic CSV merge
  convert_emg2pose.py
  convert_emg2qwerty.py
  convert_gni_discrete_gestures.py
  convert_gni_handwriting.py
  convert_gni_wrist.py
  validate.py                     # CLI validation of canonical file/tree
tests/
  test_canonical.py
  test_convert_emg2pose.py
  test_convert_emg2qwerty.py
  test_convert_gni.py
README.md
```

`canonical.py` is the only module permitted to write canonical H5 paths. The five command modules only adapt source records into a `CanonicalRecord` and task-label payload. This prevents drift in common fields.

## CLI contract

Every converter exposes:

```text
python -m semg_h5.convert_<dataset> \
  --input-root INPUT_ROOT --output-root OUTPUT_ROOT \
  [--split train|val|test] [--overwrite]
```

The script writes deterministic file names based on the namespaced source IDs and emits/updates `OUTPUT_ROOT/manifest.csv`. It refuses to overwrite an existing canonical H5 without `--overwrite`. It reports unreadable source files, malformed time vectors, non-16-channel input, and unknown input layouts with an actionable path-specific error.

## Error handling and data integrity

- Require EMG to have exactly 16 channels after source field extraction.
- Require finite, monotonic timestamps and matching signal/time length.
- Preserve source benchmark split and subject identity at record level; no window-level random split is created.
- Convert qwerty sides independently but validate their shared length/timestamp alignment before assigning a `pair_id`.
- Preserve all known source preprocessing in metadata and make no unrecorded physiological transformation.
- Store source-relative input paths and a source file SHA-256 in the manifest for provenance.

## Test strategy

Tests use small synthetic H5 fixtures matching each public reader's documented layout. Before implementation, tests will assert that the shared writer rejects invalid `[T,C]` waveforms and emits a valid canonical file. Per-source tests will then assert waveform shape, `task_id`, source split/IDs, task-label conversion, and qwerty left/right pairing. A validation-CLI test will check a converted output tree and manifest consistency.

## Acceptance criteria

1. Five converter CLIs exist and do not require source-code modification.
2. Every emitted H5 meets the common contract and validates with one command.
3. `emg2qwerty` emits two related 16-channel outputs per bilateral source session.
4. Labels sufficient for the official downstream task are preserved without label invention.
5. The manifest supports filtering by dataset, task, split, subject, session, side, and `pair_id`.
6. Tests create synthetic source files, demonstrate red-green behavior during implementation, and pass as a suite.

