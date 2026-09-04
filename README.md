# Canonical Meta 16-channel sEMG H5 conversion

This repository converts five public Meta/CTRL-labs 16-channel wrist sEMG corpora into one per-wrist HDF5 contract.  The converters are deliberately **not** physiological preprocessing pipelines: no channel reordering, filtering, rescaling, polarity correction, clipping, or resampling is performed.  The only representation change is storing `time_s` relative to the first sample; the source path and preprocessing record remain in each output.

## Install

```bash
python -m pip install -r requirements.txt
```

## Convert each corpus

Run each command with the directory containing only the corresponding raw source H5 files.  Use a separate output root from the raw input directory.

```bash
python semg_h5\convert_emg2pose.py --input-root RAW_EMG2POSE --output-root CANONICAL_META
python semg_h5\convert_emg2qwerty.py --input-root RAW_EMG2QWERTY --output-root CANONICAL_META
python semg_h5\convert_gni_discrete_gestures.py --input-root RAW_GNI_DISCRETE --output-root CANONICAL_META
python semg_h5\convert_gni_handwriting.py --input-root RAW_GNI_HANDWRITING --output-root CANONICAL_META
python semg_h5\convert_gni_wrist.py --input-root RAW_GNI_WRIST --output-root CANONICAL_META
```

Each standalone converter validates its source layout, the 16-channel signal shape, sample timestamps, and task-specific labels before writing output; no separate validator command is required.  When an official split CSV is available, supply `--splits-csv SPLITS.csv`; it must contain `split` and one of `path`, `source_file`, or `file`.  Its paths are relative to `--input-root` (or may be bare filenames).  `--split train` filters before conversion.  Do not generate random window-level splits: preserve official session/participant splits for downstream evaluation.  Existing output H5 files are refused by default; pass `--overwrite` only when intentionally regenerating them.

## Output contract

```text
CANONICAL_META/
  manifest.csv
  emg2pose/pose/<subject>/<session>_<side>.h5
  emg2qwerty/typing/<subject>/<session>_left.h5
  emg2qwerty/typing/<subject>/<session>_right.h5
  gni_discrete_gestures/discrete_gesture/...
  gni_handwriting/handwriting/...
  gni_wrist/wrist_angle/...
```

Every H5 includes root provenance attributes (`schema_version`, IDs, `official_split`, `source_file`, `preprocessing_json`) and:

```text
/signal/emg       float32 [T, 16]
/signal/time_s    float64 [T], starts at 0
/labels           task-specific labels plus source_json
/meta/source_json original source attributes and uncollapsed metadata
/meta/quality     sample_count, duration_s, nan_count, channel_rms
```

Task labels are intentionally distinct: `pose` has `joint_angles` and optional `no_ik_failure`; `typing` has `keystrokes` and `prompts`; `discrete_gesture` has `events`; `handwriting` has prompt intervals; and `wrist_angle` has one or more continuous arrays under `/labels/targets`.

`manifest.csv` has one row per canonical H5 with the fixed columns `path`, `dataset_id`, `task_id`, `subject_id`, `session_id`, `side`, `pair_id`, `official_split`, `sample_count`, `duration_s`, `source_file`, and `source_sha256`.  It is the recommended first-stage filter for a dataloader.

## Bilateral typing reconstruction

`emg2qwerty` is the only bilateral corpus.  A source session produces two canonical files with the same `pair_id`, one `side=left` and one `side=right`; both contain the session's typing labels.  For a bilateral model, group manifest rows by `pair_id`, require exactly the left/right pair, then align their shared relative clocks.  For a single-wrist pretraining model, treat them as ordinary independent canonical records but retain `pair_id` to avoid participant/session leakage across splits.

## Source-layout expectations

The emg2pose converter expects `emg2pose/timeseries` compound fields `emg`, `joint_angles`, and `time` (plus optional `no_ik_failure`).  The emg2qwerty converter expects `emg2qwerty/timeseries` fields `emg_left`, `emg_right`, and `time`; it accepts typing metadata as JSON attributes or structured `keystrokes`/`keylogger` and `prompts` datasets.  The three GNI converters expect a `data` compound dataset with `data.attrs['task']`, a `time` field, and one 16-channel EMG field; `prompts` and `stages` are read as pandas HDF tables when available.

Unknown source layouts fail fast with the missing field/task in the error message.  This is intentional: it prevents silently producing misaligned foundation-model data from a release variant that needs an explicit reader update.
