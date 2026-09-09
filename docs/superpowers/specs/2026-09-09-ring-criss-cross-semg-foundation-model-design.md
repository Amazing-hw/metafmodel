# Ring Criss-Cross sEMG Foundation Model Design

**Date:** 2026-09-09  
**Status:** Approved for implementation planning  
**Repository:** `D:\Fmodel`

## 1. Objective

Build a reproducible PyTorch Lightning and Hydra research stack on top of the five existing Meta 16-channel wrist-sEMG H5 converters. The stack will support:

- topology-aware self-supervised pretraining for a uniformly spaced 16-electrode wrist ring;
- downstream evaluation on emg2pose, emg2qwerty, GNI discrete gestures, GNI handwriting, and GNI wrist;
- supervised TCN, BENDR-style, CBraMod-style, and AEMG-style comparisons;
- fair subject/session/pair splits, low-label transfer, and channel-perturbation robustness evaluation;
- training on one to eight A100 GPUs without changing code.

The primary method is named **CIRCE-Myo** in documentation. Its backbone is a **Ring Criss-Cross Transformer** that combines temporal attention with circular spatial attention. The research claim is limited to transfer across tasks, people, sessions, and electrode-phase perturbations within the Meta wristband family. The project will not claim cross-device generalization without external-device data.

## 2. Scope

### 2.1 Included

- Lazy loading from the canonical H5 contract produced by the five converter scripts.
- Deterministic subject/session/pair-level split management.
- Time-based windowing with labels aligned from `time_s`.
- A multi-scale time-domain and spectral patch encoder.
- Ring Criss-Cross attention with circular relative-position bias.
- Continuous masked teacher pretraining, circular equivariance, and optional kinematic anchoring.
- Five task adapters and task-appropriate metrics.
- Four comparison methods using the same data and downstream protocols.
- Hydra configurations for CPU smoke tests and one to eight GPU DDP runs.
- Unit and integration tests using synthetic canonical H5 files.

### 2.2 Excluded from the first implementation

- An external language model for qwerty decoding.
- Claims of official reproduction for BENDR, CBraMod, or AEMG.
- Multi-node training. The code may work with Lightning multi-node settings, but the acceptance suite targets a single node with up to eight GPUs.
- Online serving, mobile deployment, and quantization.
- Cross-manufacturer device evaluation.
- Arbitrary channel permutation invariance. Only transformations supported by the wrist-ring geometry are allowed.

## 3. Repository Layout

The five converter scripts in `semg_h5/` remain standalone and unchanged. New training code is isolated in `semg_fm/`.

```text
D:\Fmodel
├── semg_h5/
├── semg_fm/
│   ├── data/
│   │   ├── manifest.py
│   │   ├── splits.py
│   │   ├── window_dataset.py
│   │   ├── task_labels.py
│   │   ├── datamodule.py
│   │   └── samplers.py
│   ├── models/
│   │   ├── patch_encoder.py
│   │   ├── ring_attention.py
│   │   ├── ring_criss_cross.py
│   │   ├── bendr.py
│   │   ├── cbramod.py
│   │   ├── aemg.py
│   │   ├── supervised_tcn.py
│   │   └── heads.py
│   ├── objectives/
│   │   ├── masked_latent.py
│   │   ├── ring_equivariance.py
│   │   ├── kinematic.py
│   │   └── vector_quantization.py
│   ├── lightning/
│   │   ├── pretrain_module.py
│   │   ├── downstream_module.py
│   │   └── baseline_module.py
│   ├── metrics/
│   └── utils/
├── configs/
│   ├── pretrain.yaml
│   ├── finetune.yaml
│   ├── evaluate.yaml
│   ├── baseline.yaml
│   ├── data/
│   ├── model/
│   ├── task/
│   └── trainer/
├── pretrain.py
├── finetune.py
├── evaluate.py
├── run_baseline.py
├── tests/
├── requirements-train.txt
├── PRETRAINING.md
├── DOWNSTREAM_EVALUATION.md
├── BASELINES.md
├── EXPERIMENT_PROTOCOL.md
├── H5_DATA_CONTRACT.md
└── SERVER_A100.md
```

## 4. Canonical Data Contract

Each record must contain:

- `/signal/emg`: finite `float32 [T, 16]`;
- `/signal/time_s`: monotonic `float64 [T]`;
- root attributes including `dataset_id`, `task_id`, `subject_id`, `session_id`, `side`, `pair_id`, `official_split`, and `sample_rate_hz`;
- task-specific datasets under `/labels`.

The loader validates the contract before yielding a window. It opens H5 files lazily per worker and does not retain live `h5py.File` handles across process serialization. Window and patch durations are configured in milliseconds, then converted using each record's `sample_rate_hz`.

The dataloader returns a common dictionary:

```python
{
    "emg": FloatTensor[T, 16],
    "sample_mask": BoolTensor[T],
    "dataset_id": str,
    "task_id": str,
    "record_id": str,
    "subject_id": str,
    "session_id": str,
    "side": str,
    "pair_id": str,
    "start_s": float,
    "sample_rate_hz": float,
    "target": task_specific_value,
}
```

Task-specific collation is selected by `task_id`. A pretraining batch is homogeneous in source task so that optional labels have a stable schema. A task-balanced sampler controls the long-run contribution of each dataset.

## 5. Splits and Leakage Prevention

Splits occur before windowing. The loader prefers non-empty `official_split` values. If a corpus lacks official splits, a dedicated command creates a deterministic split CSV using subject groups. That file becomes an immutable input to all later runs.

Rules:

- no subject may occur in more than one split unless an official protocol explicitly requires subject adaptation;
- no session may cross splits;
- left and right qwerty records sharing `pair_id` must remain in the same split;
- low-label subsets select complete subject/session groups rather than random windows;
- normalization statistics are estimated from training records only;
- every run copies the resolved split CSV and stores a SHA-256 digest of the source manifest.

## 6. Signal Normalization and Augmentation

The H5 converters preserve source values. Training preprocessing is explicit and reproducible in Hydra configuration.

The default normalization uses a robust scalar estimated from the training corpus or record, shared across the 16 channels. Independent per-channel normalization is disabled by default because it destroys spatial amplitude ratios. The resolved configuration records filtering, normalization, and augmentation parameters.

Physically supported augmentations include:

- circular channel shift;
- contiguous ring-sector dropout;
- independent channel dropout;
- scalar or slowly varying gain drift;
- additive noise;
- short time-block dropout.

Arbitrary channel shuffling is prohibited. Left/right reflection and polarity inversion remain disabled until converter metadata proves the required physical transform for a corpus.

## 7. CIRCE-Myo Backbone

### 7.1 Multi-scale patch encoder

Input has shape `[B, T, 16]`. A short time-domain branch captures approximately 20–50 ms morphology. A second branch summarizes approximately 100–250 ms envelope or log-power information. Configuration specifies durations in milliseconds, allowing records with different sampling rates to share one model contract.

The fused tensor has shape `[B, C=16, N, D]`. The implementation preserves the channel axis throughout the backbone.

### 7.2 Ring Criss-Cross block

Each block computes temporal and spatial attention in parallel:

- temporal attention reshapes to `[B*C, N, D]` and attends across time for each electrode;
- ring attention reshapes to `[B*N, C, D]` and attends across electrodes at each time token;
- a learned gate fuses the two residual updates.

Ring attention uses a learnable signed circular relative-position bias indexed by `(c_i - c_j) mod 16`. Consequently, channels 15 and 0 are adjacent. No absolute anatomical meaning is assigned to a fixed channel number.

The backbone returns:

- `z_eq [B, 16, N, D]`, an electrode-field representation that should transform equivariantly under circular shifts;
- `z_motion [B, N, D]`, an orbit-pooled motion representation designed to be stable under circular shifts.

### 7.3 Mask generation

Masking operates on the `[16, N]` patch grid and always emits an explicit boolean mask. Available mask types are temporal spans, contiguous electrode arcs, complete channel dropout, local spatial-temporal blocks, and sparse independent patches. The default configuration samples a mixture so that reconstruction requires both temporal context and cross-electrode context.

## 8. Pretraining Objectives

The default method uses three objectives:

```text
L = L_masked_latent + lambda_ring * L_ring + lambda_kin * L_kinematic
```

### 8.1 Continuous masked teacher objective

An EMA teacher receives the unmasked view. The student receives a masked view. The loss is computed only at masked positions and combines normalized Smooth-L1 and cosine distance between continuous teacher and student latents. The teacher is stop-gradient and updated after optimizer steps. All DDP ranks start from the same parameters and perform identical EMA updates.

### 8.2 Circular equivariance objective

For a randomly sampled circular shift `S_k`, the electrode-field target is equivariant and the pooled motion target is invariant:

```text
z_eq(S_k x) ~= S_k z_eq(x)
z_motion(S_k x) ~= z_motion(x)
```

The implementation aligns the shifted electrode axis before computing the first loss. This objective does not treat arbitrary samples or left/right wrists as positive pairs.

### 8.3 Kinematic objective

The objective is active only for training records from emg2pose and GNI wrist. A small adapter predicts normalized continuous targets from `z_motion`. emg2pose frames with invalid inverse-kinematics status are masked from the loss. Kinematic normalization statistics come from the training split and evaluation restores original units.

Envelope reconstruction, spectral reconstruction, future prediction, and VQ targets are optional ablations rather than default losses.

## 9. Downstream Tasks

### 9.1 emg2pose

- Source: `/labels/joint_angles` and optional `/labels/no_ik_failure`.
- Head: per-frame continuous regression.
- Metrics: per-joint and aggregate MAE, RMSE, Pearson correlation, and velocity MAE.
- Invalid IK frames are excluded from both loss and metrics.

### 9.2 GNI wrist

- Source: numeric datasets under `/labels/targets`.
- Head: configurable multi-target regression.
- Metrics: original-unit MAE, RMSE, and correlation for every target and their macro average.

### 9.3 GNI discrete gestures

- Source: `/labels/events` with `name` and `time_s`.
- Head: frame-level classification including a background class.
- Training targets: event-centered tolerance intervals configured in milliseconds.
- Metrics: macro F1, balanced accuracy, and event F1 under declared time tolerances.

### 9.4 GNI handwriting

- Source: `/labels/prompts` with `text`, `start_s`, and `end_s`.
- Head: character CTC decoder over each prompt interval.
- Metrics: character error rate, normalized edit distance, and exact prompt accuracy.
- The implementation does not invent per-character timing labels.

### 9.5 emg2qwerty

- Source: `/labels/keystrokes` and `/labels/prompts`.
- Head: character CTC decoder, with an optional event output for key timing.
- Metrics: character error rate, key-event F1, and detection latency.
- Main encoder comparisons do not use an external language model.

## 10. Fine-tuning Protocols

Every task supports:

- linear probing with the backbone frozen;
- partial fine-tuning of the last configured number of backbone blocks;
- full fine-tuning.

Label budgets are 1%, 5%, 10%, and 100% at the subject/session-group level. All methods use the same task head, optimizer family, maximum update budget, split file, and preprocessing. Each reported configuration supports at least three seeds.

## 11. Comparison Methods

### 11.1 Supervised TCN

A temporal convolutional encoder trained directly on each downstream task without self-supervised pretraining. It establishes whether pretraining improves over a competitive supervised model.

### 11.2 BENDR-style baseline

A convolutional latent encoder and temporal Transformer use masked continuous contrastive prediction. Negatives come from other positions in the same record. It tests whether gains arise from ordinary continuous masked pretraining without explicit ring topology.

### 11.3 CBraMod-style baseline

A time-domain and FFT patch encoder feeds parallel temporal and ordinary spatial attention. Conditional position encoding and masked waveform reconstruction follow the core CBraMod mechanism. It uses the fixed channel order without circular relative-position bias or equivariance loss.

### 11.4 AEMG-style baseline

A contraction-aware encoder, vector-quantization codebook, and masked token classifier reproduce the core AEMG hypothesis. Logs include codebook perplexity, code utilization, and dead-code fraction. The documentation states all deviations from the paper and does not label this as an official implementation.

All methods provide small and base configurations. The experiment table reports parameter count, training tokens, steps, global batch size, and approximate compute so model-size or data-budget differences remain visible.

## 12. Robustness Evaluation

Evaluation-time transforms operate in memory and never modify H5 files. Required tests are:

- circular shifts of 1, 2, 4, and 8 channels;
- one or more adjacent missing electrodes;
- random single-channel dropout;
- gain drift;
- additive noise at configured signal-to-noise ratios;
- short temporal signal loss.

Metrics are reported separately for every dataset, subject, perturbation type, and severity. Synthetic circular shift experiments establish algorithmic channel-phase robustness. They are not described as evidence of real re-wearing robustness unless the source data contain corresponding physical re-wearing metadata.

## 13. PyTorch Lightning and Hydra Runtime

Top-level scripts are thin Hydra entry points. Configuration groups select data, model, task, robustness transform, and trainer profile. Resolved configuration is saved with each run.

Default server behavior:

- BF16 mixed precision on A100;
- `strategy=auto` for one GPU and DDP for multiple GPUs;
- `devices=auto`, overridable as `trainer.devices=1`, `2`, `4`, or `8`;
- gradient accumulation used to preserve a configured effective global batch size;
- deterministic seeding where supported;
- checkpointing of model, optimizer, scheduler, EMA teacher, epoch, and global step;
- restart from `last.ckpt` without resetting the EMA teacher or schedule.

The run metadata records Git commit, manifest digest, split digest, dependency versions, visible GPU count, world size, per-device batch size, effective global batch size, precision, and seed.

## 14. Outputs

Every run writes to a Hydra output directory containing:

```text
config.yaml
metrics.json
per_subject_metrics.csv
split_manifest.csv
environment.txt
checkpoints/
```

Evaluation additionally writes `predictions.h5` when configured. Metrics for heterogeneous tasks remain separate; the framework does not manufacture a single aggregate score across regression, classification, and sequence recognition.

## 15. Testing Strategy

Tests create compact canonical H5 fixtures with 16 channels and deterministic labels. Production code follows test-driven development.

Required unit tests cover:

- manifest validation and lazy H5 access;
- time-to-sample window conversion and label alignment;
- split leakage checks, including qwerty `pair_id`;
- task-balanced sampling under DDP rank partitioning;
- circular distance and channel-rolling behavior;
- Ring Attention boundary behavior for channels 15 and 0;
- numerical equivariance of the Ring Criss-Cross backbone in evaluation mode;
- masking semantics and loss application only at masked positions;
- all five task collators, losses, and metrics;
- checkpoint state for the EMA teacher;
- Hydra composition and command-line overrides.

Integration tests run one CPU `fast_dev_run` for CIRCE-Myo and each pretraining baseline, then one short fine-tune/evaluation path for each task. The local suite does not claim to validate NCCL. `SERVER_A100.md` defines a separate one-batch DDP smoke test for 1, 2, 4, and 8 GPUs.

## 16. Documentation

- `README.md`: installation, project map, and five-minute commands.
- `PRETRAINING.md`: CIRCE-Myo architecture, losses, data mixing, and resume behavior.
- `DOWNSTREAM_EVALUATION.md`: task labels, metrics, split rules, low-label protocols, and robustness commands.
- `BASELINES.md`: baseline mechanisms, fidelity boundaries, and fair-comparison rules.
- `EXPERIMENT_PROTOCOL.md`: main tables, ablations, seeds, budgets, and reporting checklist.
- `H5_DATA_CONTRACT.md`: fields required from each converter output and validation commands.
- `SERVER_A100.md`: environment setup and one-to-eight-GPU launch, resume, and troubleshooting commands.

## 17. Acceptance Criteria

The implementation is complete when:

1. Existing five converter tests still pass and converter scripts remain standalone.
2. A synthetic canonical manifest can run one CPU pretraining step for CIRCE-Myo, BENDR-style, CBraMod-style, and AEMG-style methods.
3. Synthetic data can run one train/evaluate step for all five downstream tasks.
4. Split tests prove that subjects, sessions, and qwerty pairs do not leak across partitions.
5. Ring equivariance tests pass within declared floating-point tolerance.
6. Hydra commands select models, tasks, trainer devices, label budgets, and robustness settings without source edits.
7. Checkpoint resume restores the EMA teacher and training step.
8. Every public command in the Markdown documentation is covered by configuration composition or a smoke test.
9. The full local test suite passes.
10. The A100 server smoke-test instructions cover 1, 2, 4, and 8 GPUs and clearly distinguish unverified multi-GPU behavior from locally verified behavior.

