# Split Provenance and Safe Run Scripts Design

## Goal

Make every post-conversion experiment explicitly select one split manifest and prevent accidental reuse of an existing run directory. This applies only to split preparation and training, baseline, pretraining, fine-tuning, evaluation, robustness, and pipeline PowerShell scripts. It does not alter H5 conversion or scientific model behavior.

## Split contract

`Common.ps1` will provide one resolver accepting `CanonicalRoot`, optional `SplitsPath`, `Protocol`, and `Seed`.

- `SplitsPath`, when supplied, resolves to an existing absolute file and takes precedence.
- Otherwise, `Protocol` plus `Seed` are required and resolve to `splits.official.seed-<seed>.csv` or `splits.strict-subject.seed-<seed>.csv` under the canonical root.
- Legacy `splits.csv` remains usable only through an explicit `SplitsPath`.

All scripts pass the resolved path as Hydra `data.manifest_path`, preserving split provenance in the existing resolved configuration and manifest metadata.

## Script behavior

`02_PrepareSplits.ps1` keeps `Official` and `StrictSubject`, produces the matching seed-qualified filename, and prints the exact downstream selection. `03_SmokeTest.ps1` through `08_Robustness.ps1` expose `SplitsPath`, `Protocol`, and `Seed`, rejecting calls that provide neither a path nor a protocol/seed pair. `RunPipeline.ps1` exposes the same contract, forwards protocol and seed to split preparation, and forwards the same resolved split path to every selected child stage.

## Output safety

`Common.ps1` will provide an output-directory guard. Before non-dry execution, every stage rejects an existing final output directory unless `Force` is supplied. Dry-run reports the split and output directory but creates nothing. Force permits reuse but never deletes existing data. Run locations include protocol and seed when those values resolved the split; an explicit split path receives a stable path digest in the run location.

## Documentation and verification

The workflow and script reference will become a post-conversion runbook for setup, audit, both split protocols, smoke validation, baseline comparison, pilot/full pretraining, label-budget fine-tuning, evaluation, robustness, resume, and artifact reporting. Every command states its split selection and overwrite behavior.

Tests first demonstrate split forwarding, failure when selection is absent, pipeline consistency, and output collision rejection. They then validate the minimal PowerShell implementation, parser validity, and dry-run construction.
