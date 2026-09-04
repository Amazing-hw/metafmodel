# Five standalone sEMG H5 converters

## Goal

Reduce `D:\Fmodel\semg_h5` to exactly five independently executable Python scripts, one for each supported public Meta/CTRL-labs dataset task:

1. `convert_emg2pose.py`
2. `convert_emg2qwerty.py`
3. `convert_gni_discrete_gestures.py`
4. `convert_gni_handwriting.py`
5. `convert_gni_wrist.py`

No script may import another local module from `semg_h5`.

## Design

Every retained script embeds the small common functions it needs: source-layout checks, `[T,16]` sEMG validation, canonical H5 writing, task-label serialization, output-path generation, SHA-256 provenance, CSV manifest upsert, optional official-split lookup, and an argparse CLI. It may import third-party dependencies declared in `requirements.txt` (`h5py`, `numpy`, `pandas`) and the Python standard library.

The five scripts preserve the existing output contract:

- `/signal/emg`: `float32 [T,16]`
- `/signal/time_s`: `float64 [T]`, shifted only to a record-relative origin
- root provenance attributes and `/meta/quality`
- per-task labels: pose angles; qwerty keystrokes/prompts; discrete gesture events; handwriting prompt intervals; wrist continuous targets
- `manifest.csv` fields including task ID, official split, IDs, side, pair ID, and source hash

`convert_emg2qwerty.py` emits one H5 per wrist and keeps a shared `pair_id`. The three GNI scripts each validate their own source task value rather than sharing a runner.

## Removal and compatibility

Delete all other Python files under `semg_h5`: `__init__.py`, `canonical.py`, `manifest.py`, `source_readers.py`, `gni_conversion.py`, and `validate.py`. The standalone scripts must run directly, for example:

```powershell
python D:\Fmodel\semg_h5\convert_emg2pose.py --input-root RAW --output-root CANONICAL
```

The previous package-module invocation and separate validator command are intentionally removed. Validation is performed before each script writes a record. README examples will change to direct script execution.

## Verification

Tests will invoke every script directly by file path against synthetic source H5 fixtures. They will verify that each output has `[T,16]` sEMG and its task-specific labels, that qwerty produces a linked left/right pair, and that each script has no imports from `semg_h5`.
