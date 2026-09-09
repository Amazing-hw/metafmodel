# Ring Criss-Cross sEMG Foundation Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PyTorch Lightning and Hydra research stack that pretrains CIRCE-Myo on the five canonical Meta 16-channel H5 corpora, evaluates five downstream tasks, and runs four fair comparison methods.

**Architecture:** Keep the five standalone converters unchanged. Add a `semg_fm` package with a lazy canonical-H5 data layer, a Ring Criss-Cross backbone, pretraining objectives, comparison backbones, downstream heads, metrics, and thin Hydra entry points. Every behavior is first specified by a failing pytest, then implemented minimally and verified before the next commit.

**Tech Stack:** Python 3.10+, PyTorch 2.2+, Lightning 2.2+, Hydra 1.3+, h5py, NumPy, pandas, torchmetrics, pytest.

---

## File Map

- `semg_fm/data`: manifest validation, group splits, lazy H5 windows, labels, augmentation, balanced sampling, Lightning DataModule.
- `semg_fm/models`: shared patch encoder, circular attention, CIRCE-Myo, and four comparison encoders.
- `semg_fm/objectives`: masking, continuous teacher loss, ring equivariance, kinematic loss, and VQ diagnostics.
- `semg_fm/lightning`: pretraining, downstream, and baseline LightningModules.
- `semg_fm/metrics`: regression, classification/event, edit-distance, and robustness aggregation.
- `configs`: Hydra composition for data, model, task, trainer, and robustness settings.
- `pretrain.py`, `finetune.py`, `evaluate.py`, `run_baseline.py`: public commands.
- `tests`: synthetic canonical H5 fixtures, unit tests, and CPU fast-development integration tests.
- Markdown files at repository root: installation, data contract, pretraining, downstream evaluation, baselines, experiment protocol, and A100 operation.

### Task 1: Training dependencies and Hydra composition

**Files:**
- Create: `requirements-train.txt`
- Create: `semg_fm/__init__.py`
- Create: `semg_fm/config.py`
- Create: `configs/pretrain.yaml`
- Create: `configs/finetune.yaml`
- Create: `configs/evaluate.yaml`
- Create: `configs/baseline.yaml`
- Create: `configs/data/canonical_meta.yaml`
- Create: `configs/model/ring_criss_cross.yaml`
- Create: `configs/trainer/cpu_smoke.yaml`
- Create: `configs/trainer/a100_ddp.yaml`
- Test: `tests/test_hydra_config.py`

- [ ] **Step 1: Write the failing configuration tests**

```python
from hydra import compose, initialize_config_dir
from pathlib import Path


CONFIG_DIR = str((Path(__file__).parents[1] / "configs").resolve())


def test_pretrain_config_composes_with_four_devices():
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="pretrain", overrides=["trainer.devices=4"])
    assert cfg.trainer.devices == 4
    assert cfg.model.name == "ring_criss_cross"
    assert cfg.data.channels == 16


def test_cpu_smoke_profile_disables_distributed_training():
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="pretrain", overrides=["trainer=cpu_smoke"])
    assert cfg.trainer.accelerator == "cpu"
    assert cfg.trainer.devices == 1
    assert cfg.trainer.strategy == "auto"
```

- [ ] **Step 2: Run the test and verify the missing configuration failure**

Run: `python -m pytest tests/test_hydra_config.py -q`  
Expected: FAIL because `configs/pretrain.yaml` and `semg_fm.config` do not exist.

- [ ] **Step 3: Implement the dependency and configuration contracts**

`requirements-train.txt` must pin compatible major/minor floors:

```text
-r requirements.txt
torch>=2.2
lightning>=2.2
hydra-core>=1.3,<1.4
omegaconf>=2.3,<2.4
torchmetrics>=1.3
pytest>=7.0
```

`semg_fm/config.py` defines a small validator used by entry points:

```python
def validate_devices(devices: int | str) -> None:
    if devices == "auto":
        return
    if not isinstance(devices, int) or devices < 1 or devices > 8:
        raise ValueError("trainer.devices must be 'auto' or an integer from 1 to 8")
```

The pretraining defaults must compose `data=canonical_meta`, `model=ring_criss_cross`, and `trainer=a100_ddp`; CPU smoke uses `accelerator: cpu`, `devices: 1`, `strategy: auto`, `precision: 32-true`, and `fast_dev_run: true`. A100 uses `accelerator: gpu`, `devices: auto`, `strategy: auto`, and `precision: bf16-mixed`.

- [ ] **Step 4: Run the focused test**

Run: `python -m pytest tests/test_hydra_config.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit the configuration foundation**

```powershell
git add requirements-train.txt semg_fm/__init__.py semg_fm/config.py configs tests/test_hydra_config.py
git commit -m "build: add lightning hydra training configuration"
```

### Task 2: Synthetic canonical H5 fixtures and manifest validation

**Files:**
- Modify: `tests/conftest.py`
- Create: `semg_fm/data/__init__.py`
- Create: `semg_fm/data/manifest.py`
- Test: `tests/test_fm_manifest.py`

- [ ] **Step 1: Write the failing manifest tests**

```python
import pandas as pd
import pytest
from semg_fm.data.manifest import load_manifest


def test_manifest_resolves_records_and_preserves_task_ids(canonical_meta_root):
    frame = load_manifest(canonical_meta_root / "manifest.csv", canonical_meta_root)
    assert set(frame.task_id) == {
        "pose", "typing", "discrete_gesture", "handwriting", "wrist_angle"
    }
    assert frame.resolved_path.map(lambda p: p.exists()).all()


def test_manifest_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame({"path": ["x.h5"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_manifest(path, tmp_path)
```

`canonical_meta_root` creates one small, finite `[T,16]` H5 for every task and a qwerty left/right pair with the same `pair_id`. It writes each real task-specific label structure used by the converters.

- [ ] **Step 2: Verify the tests fail because the data package is absent**

Run: `python -m pytest tests/test_fm_manifest.py -q`  
Expected: FAIL with `ModuleNotFoundError: semg_fm.data`.

- [ ] **Step 3: Implement manifest validation**

```python
REQUIRED_COLUMNS = {
    "path", "dataset_id", "task_id", "subject_id", "session_id", "side",
    "pair_id", "official_split", "sample_count", "duration_s", "source_file",
    "source_sha256",
}


def load_manifest(path, root):
    frame = pd.read_csv(path, keep_default_na=False)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["resolved_path"] = frame["path"].map(lambda value: Path(root) / value)
    absent = [str(p) for p in frame.resolved_path if not p.is_file()]
    if absent:
        raise FileNotFoundError(f"manifest records not found: {absent[:3]}")
    return frame
```

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest tests/test_fm_manifest.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit fixtures and manifest loader**

```powershell
git add tests/conftest.py tests/test_fm_manifest.py semg_fm/data
git commit -m "feat: validate canonical training manifests"
```

### Task 3: Deterministic group splits without leakage

**Files:**
- Create: `semg_fm/data/splits.py`
- Create: `make_splits.py`
- Test: `tests/test_fm_splits.py`

- [ ] **Step 1: Write failing split tests**

```python
from semg_fm.data.splits import assign_group_splits, assert_no_leakage


def test_qwerty_pair_never_crosses_splits(manifest_frame):
    assigned = assign_group_splits(manifest_frame, seed=17, ratios=(0.7, 0.15, 0.15))
    pair_rows = assigned[assigned.pair_id == "pair-1"]
    assert pair_rows.split.nunique() == 1
    assert_no_leakage(assigned)


def test_assignment_is_deterministic(manifest_frame):
    a = assign_group_splits(manifest_frame, seed=17, ratios=(0.7, 0.15, 0.15))
    b = assign_group_splits(manifest_frame, seed=17, ratios=(0.7, 0.15, 0.15))
    assert a[["path", "split"]].equals(b[["path", "split"]])
```

- [ ] **Step 2: Verify the missing split implementation failure**

Run: `python -m pytest tests/test_fm_splits.py -q`  
Expected: FAIL with import error for `semg_fm.data.splits`.

- [ ] **Step 3: Implement group assignment and CLI**

Use `pair_id` when non-empty; otherwise use the tuple `(dataset_id, subject_id)`. Hash `f"{seed}:{group_key}"` with SHA-256, map the first eight bytes to `[0,1)`, and assign train/val/test by cumulative ratios. `assert_no_leakage` must reject repeated subject groups or pair IDs in different splits. `make_splits.py` reads a manifest, preserves non-empty official splits, assigns only missing rows, validates leakage, and writes `split_manifest.csv`.

- [ ] **Step 4: Run split tests and a fixture CLI check**

Run: `python -m pytest tests/test_fm_splits.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit split management**

```powershell
git add semg_fm/data/splits.py make_splits.py tests/test_fm_splits.py
git commit -m "feat: add leakage-safe subject splits"
```

### Task 4: Lazy H5 window dataset and task-label alignment

**Files:**
- Create: `semg_fm/data/task_labels.py`
- Create: `semg_fm/data/window_dataset.py`
- Test: `tests/test_window_dataset.py`
- Test: `tests/test_task_labels.py`

- [ ] **Step 1: Write failing dataset and alignment tests**

```python
import torch
from semg_fm.data.window_dataset import CanonicalWindowDataset


def test_window_uses_milliseconds_and_returns_time_major_emg(canonical_meta_root):
    ds = CanonicalWindowDataset.from_manifest(
        canonical_meta_root / "manifest.csv", split="train",
        window_ms=200, stride_ms=100,
    )
    item = ds[0]
    assert item["emg"].ndim == 2
    assert item["emg"].shape[1] == 16
    assert item["emg"].shape[0] == 400


def test_pose_invalid_ik_frames_become_target_mask(pose_dataset):
    item = pose_dataset[0]
    assert item["target"].shape[0] == item["emg"].shape[0]
    assert item["target_mask"].dtype == torch.bool
    assert not item["target_mask"][3]


def test_handwriting_returns_prompt_text_without_character_times(handwriting_dataset):
    item = handwriting_dataset[0]
    assert item["target_text"] == "hello"
    assert "character_times" not in item
```

- [ ] **Step 2: Verify failure before dataset implementation**

Run: `python -m pytest tests/test_window_dataset.py tests/test_task_labels.py -q`  
Expected: FAIL because `CanonicalWindowDataset` and label adapters are missing.

- [ ] **Step 3: Implement lazy access and five adapters**

`CanonicalWindowDataset` stores only record paths and integer sample bounds. `__getitem__` opens the H5 file in a context manager, reads only the selected EMG slice, validates `[window_samples,16]`, and calls `parse_target(task_id, h5, start_s, end_s, start_index, end_index)`. Implement adapters for pose, wrist targets, gesture events, handwriting prompts, and qwerty keystrokes/prompts. Text fields decode both bytes and UTF-8 strings. Empty target windows carry an explicit empty target rather than a fabricated label.

- [ ] **Step 4: Run focused dataset tests**

Run: `python -m pytest tests/test_window_dataset.py tests/test_task_labels.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit the canonical dataset layer**

```powershell
git add semg_fm/data/task_labels.py semg_fm/data/window_dataset.py tests/test_window_dataset.py tests/test_task_labels.py
git commit -m "feat: load aligned canonical semg windows"
```

### Task 5: Physical augmentation, collation, sampling, and DataModule

**Files:**
- Create: `semg_fm/data/augment.py`
- Create: `semg_fm/data/samplers.py`
- Create: `semg_fm/data/datamodule.py`
- Test: `tests/test_augment.py`
- Test: `tests/test_datamodule.py`

- [ ] **Step 1: Write failing physical-augmentation tests**

```python
import torch
from semg_fm.data.augment import circular_shift, contiguous_channel_dropout


def test_circular_shift_rolls_only_channel_axis():
    x = torch.arange(2 * 5 * 16).reshape(2, 5, 16)
    shifted = circular_shift(x, 1)
    assert torch.equal(shifted[..., 0], x[..., 15])
    assert torch.equal(shifted[:, 0, :], torch.roll(x[:, 0, :], 1, -1))


def test_contiguous_dropout_wraps_across_ring_boundary():
    x = torch.ones(4, 16)
    y, missing = contiguous_channel_dropout(x, start=15, count=2)
    assert missing.tolist() == [0, 15]
    assert y[:, [0, 15]].count_nonzero() == 0
```

- [ ] **Step 2: Verify augmentation tests fail**

Run: `python -m pytest tests/test_augment.py tests/test_datamodule.py -q`  
Expected: FAIL because augmentation and DataModule modules are absent.

- [ ] **Step 3: Implement augmentation and batching**

Implement channel rolling with `torch.roll`, modular contiguous indices, gain drift, additive noise, and time-block dropout. Return a boolean observation mask for every destructive transform. Implement task-homogeneous collation and a deterministic task-balanced batch sampler whose `set_epoch` changes ordering. `CanonicalDataModule` builds train/validation/test datasets from one split manifest and passes Lightning rank/world-size information to the sampler.

- [ ] **Step 4: Run focused data pipeline tests**

Run: `python -m pytest tests/test_augment.py tests/test_datamodule.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit the distributed data pipeline**

```powershell
git add semg_fm/data tests/test_augment.py tests/test_datamodule.py
git commit -m "feat: add physical augmentation and task-balanced loading"
```

### Task 6: Multi-scale patch encoder and circular attention

**Files:**
- Create: `semg_fm/models/__init__.py`
- Create: `semg_fm/models/patch_encoder.py`
- Create: `semg_fm/models/ring_attention.py`
- Test: `tests/test_patch_encoder.py`
- Test: `tests/test_ring_attention.py`

- [ ] **Step 1: Write failing shape and boundary tests**

```python
import torch
from semg_fm.models.patch_encoder import MultiScalePatchEncoder
from semg_fm.models.ring_attention import signed_ring_offsets


def test_patch_encoder_preserves_sixteen_channel_axis():
    encoder = MultiScalePatchEncoder(channels=16, d_model=32, patch_samples=40)
    out = encoder(torch.randn(2, 400, 16))
    assert out.shape == (2, 16, 10, 32)


def test_ring_offsets_make_endpoints_neighbors():
    offsets = signed_ring_offsets(16)
    assert abs(int(offsets[0, 15])) == 1
    assert abs(int(offsets[15, 0])) == 1
    assert int(offsets[0, 8]) in {-8, 8}
```

- [ ] **Step 2: Verify missing-model failures**

Run: `python -m pytest tests/test_patch_encoder.py tests/test_ring_attention.py -q`  
Expected: FAIL with import errors.

- [ ] **Step 3: Implement encoder and ring-relative attention**

The time branch applies grouped temporal convolution to each channel. The spectral branch computes `log1p(abs(rfft(patch)))` and projects a fixed number of frequency bins. Concatenate and linearly project both branches to `d_model`. `RingMultiheadAttention` computes scaled dot-product attention and adds a learnable bias table indexed by signed circular offsets before softmax. It accepts an electrode-observation mask and never attends to missing electrodes.

- [ ] **Step 4: Run focused model tests**

Run: `python -m pytest tests/test_patch_encoder.py tests/test_ring_attention.py -q`  
Expected: PASS, including finite outputs with one missing channel.

- [ ] **Step 5: Commit patch and ring primitives**

```powershell
git add semg_fm/models tests/test_patch_encoder.py tests/test_ring_attention.py
git commit -m "feat: encode multiscale circular semg patches"
```

### Task 7: Ring Criss-Cross backbone and equivariance

**Files:**
- Create: `semg_fm/models/ring_criss_cross.py`
- Create: `configs/model/ring_criss_cross.yaml`
- Test: `tests/test_ring_criss_cross.py`

- [ ] **Step 1: Write the failing backbone tests**

```python
import torch
from semg_fm.models.ring_criss_cross import RingCrissCrossEncoder


def test_backbone_returns_field_and_motion_tokens():
    model = RingCrissCrossEncoder(d_model=32, depth=2, heads=4, patch_samples=40)
    outputs = model(torch.randn(2, 400, 16))
    assert outputs["z_eq"].shape == (2, 16, 10, 32)
    assert outputs["z_motion"].shape == (2, 10, 32)


def test_field_rolls_and_motion_stays_stable_under_channel_shift():
    torch.manual_seed(7)
    model = RingCrissCrossEncoder(d_model=16, depth=1, heads=4, patch_samples=40).eval()
    x = torch.randn(1, 400, 16)
    a = model(x)
    b = model(torch.roll(x, 3, dims=-1))
    assert torch.allclose(torch.roll(a["z_eq"], 3, dims=1), b["z_eq"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(a["z_motion"], b["z_motion"], atol=1e-5, rtol=1e-4)
```

- [ ] **Step 2: Verify the backbone tests fail**

Run: `python -m pytest tests/test_ring_criss_cross.py -q`  
Expected: FAIL because `RingCrissCrossEncoder` is absent.

- [ ] **Step 3: Implement parallel temporal/ring blocks**

Each block normalizes once, reshapes for `nn.MultiheadAttention` over time and `RingMultiheadAttention` over channels, restores `[B,16,N,D]`, and fuses residuals using `softmax(gate_logits)` weights. Feed-forward layers operate pointwise. Avoid absolute channel embeddings. Compute `z_motion` with mean pooling over the channel orbit, respecting the observation mask.

- [ ] **Step 4: Run backbone and primitive tests**

Run: `python -m pytest tests/test_ring_criss_cross.py tests/test_ring_attention.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit CIRCE-Myo backbone**

```powershell
git add semg_fm/models/ring_criss_cross.py configs/model/ring_criss_cross.yaml tests/test_ring_criss_cross.py
git commit -m "feat: add ring criss-cross semg encoder"
```

### Task 8: Masking, continuous teacher, equivariance, and kinematic losses

**Files:**
- Create: `semg_fm/objectives/__init__.py`
- Create: `semg_fm/objectives/masking.py`
- Create: `semg_fm/objectives/masked_latent.py`
- Create: `semg_fm/objectives/ring_equivariance.py`
- Create: `semg_fm/objectives/kinematic.py`
- Test: `tests/test_pretrain_objectives.py`

- [ ] **Step 1: Write failing objective tests**

```python
import torch
from semg_fm.objectives.masked_latent import masked_latent_loss
from semg_fm.objectives.ring_equivariance import ring_equivariance_loss


def test_masked_latent_loss_ignores_visible_positions():
    student = torch.zeros(1, 2, 3, 4)
    teacher = student.clone()
    teacher[:, :, 0] = 1
    mask = torch.tensor([[[True, False, False], [True, False, False]]])
    first = masked_latent_loss(student, teacher, mask)
    teacher[:, :, 1:] = 1000
    second = masked_latent_loss(student, teacher, mask)
    assert torch.allclose(first, second)


def test_equivariance_loss_aligns_shifted_field():
    field = torch.randn(2, 16, 3, 8)
    motion = field.mean(1)
    shifted = torch.roll(field, 4, dims=1)
    loss = ring_equivariance_loss(field, motion, shifted, motion, shift=4)
    assert loss.item() < 1e-7
```

- [ ] **Step 2: Verify objective tests fail**

Run: `python -m pytest tests/test_pretrain_objectives.py -q`  
Expected: FAIL with import errors.

- [ ] **Step 3: Implement masks and losses**

`MaskGenerator` returns a boolean `[B,16,N]` grid using configured temporal-span, ring-arc, channel-drop, block, and random-patch probabilities. `masked_latent_loss` combines Smooth-L1 and cosine distance only where mask is true. `ring_equivariance_loss` inverse-rolls the shifted field before comparison and directly compares pooled motion. `kinematic_loss` applies Smooth-L1 only where the target mask is true and returns a differentiable zero when a batch has no kinematic target.

- [ ] **Step 4: Run objective tests**

Run: `python -m pytest tests/test_pretrain_objectives.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit pretraining objectives**

```powershell
git add semg_fm/objectives tests/test_pretrain_objectives.py
git commit -m "feat: add circe myo pretraining objectives"
```

### Task 9: Lightning pretraining module, EMA state, and public entry point

**Files:**
- Create: `semg_fm/lightning/__init__.py`
- Create: `semg_fm/lightning/pretrain_module.py`
- Create: `semg_fm/registry.py`
- Create: `pretrain.py`
- Test: `tests/test_pretrain_module.py`
- Test: `tests/test_pretrain_smoke.py`

- [ ] **Step 1: Write failing EMA and smoke tests**

```python
import torch
from lightning import Trainer
from semg_fm.lightning.pretrain_module import PretrainModule


def test_ema_teacher_is_saved_and_restored(tiny_pretrain_module, tiny_datamodule, tmp_path):
    trainer = Trainer(accelerator="cpu", devices=1, fast_dev_run=True, logger=False, enable_checkpointing=False)
    trainer.fit(tiny_pretrain_module, datamodule=tiny_datamodule)
    before = {k: v.clone() for k, v in tiny_pretrain_module.teacher.state_dict().items()}
    path = tmp_path / "state.ckpt"
    trainer.save_checkpoint(path)
    restored = PretrainModule.load_from_checkpoint(path)
    for key, value in before.items():
        assert torch.equal(value, restored.teacher.state_dict()[key])


def test_circe_pretraining_fast_dev_run(tiny_pretrain_module, tiny_datamodule):
    trainer = Trainer(accelerator="cpu", devices=1, fast_dev_run=True, logger=False, enable_checkpointing=False)
    trainer.fit(tiny_pretrain_module, datamodule=tiny_datamodule)
    assert trainer.global_step == 1
```

- [ ] **Step 2: Verify pretraining tests fail**

Run: `python -m pytest tests/test_pretrain_module.py tests/test_pretrain_smoke.py -q`  
Expected: FAIL because the Lightning module is absent.

- [ ] **Step 3: Implement pretraining and Hydra entry point**

`PretrainModule` owns student, frozen teacher, mask generator, kinematic adapter, optimizer, and scheduler. It creates an unmasked teacher view under `torch.no_grad()`, creates student and shifted views, computes the three losses, and logs each loss plus task-specific validation values with `sync_dist=True`. `optimizer_step` performs EMA only after the optimizer changes student weights. All teacher parameters live in `state_dict`. `pretrain.py` validates devices, seeds through Lightning, instantiates the registry-selected model and DataModule, snapshots the resolved config and manifest digest, and calls `Trainer.fit(ckpt_path=cfg.resume_from)`.

- [ ] **Step 4: Run pretraining tests**

Run: `python -m pytest tests/test_pretrain_module.py tests/test_pretrain_smoke.py -q`  
Expected: PASS on CPU.

- [ ] **Step 5: Commit the public pretraining path**

```powershell
git add semg_fm/lightning semg_fm/registry.py pretrain.py tests/test_pretrain_module.py tests/test_pretrain_smoke.py
git commit -m "feat: train circe myo with lightning"
```

### Task 10: Supervised TCN and BENDR-style comparison methods

**Files:**
- Create: `semg_fm/models/supervised_tcn.py`
- Create: `semg_fm/models/bendr.py`
- Create: `configs/model/supervised_tcn.yaml`
- Create: `configs/model/bendr.yaml`
- Test: `tests/test_baseline_tcn_bendr.py`

- [ ] **Step 1: Write failing comparison tests**

```python
import torch
from semg_fm.models.supervised_tcn import SupervisedTCN
from semg_fm.models.bendr import BendrEncoder


def test_supervised_tcn_returns_temporal_features():
    y = SupervisedTCN(in_channels=16, d_model=32)(torch.randn(2, 400, 16))
    assert y.ndim == 3 and y.shape[0] == 2 and y.shape[-1] == 32


def test_bendr_returns_continuous_context_and_targets():
    out = BendrEncoder(in_channels=16, d_model=32)(torch.randn(2, 400, 16))
    assert out["context"].shape == out["target"].shape
    assert out["context"].ndim == 3
```

- [ ] **Step 2: Verify comparison tests fail**

Run: `python -m pytest tests/test_baseline_tcn_bendr.py -q`  
Expected: FAIL with missing model modules.

- [ ] **Step 3: Implement the two continuous baselines**

The TCN uses causal-capable residual Conv1d blocks and returns `[B,N,D]`. BENDR uses a Conv1d feature encoder followed by a temporal Transformer and exposes uncontextualized continuous targets plus masked contextual features. Its contrastive loss samples negatives from other positions of the same record and excludes the positive index.

- [ ] **Step 4: Run focused baseline tests**

Run: `python -m pytest tests/test_baseline_tcn_bendr.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit TCN and BENDR baselines**

```powershell
git add semg_fm/models/supervised_tcn.py semg_fm/models/bendr.py configs/model tests/test_baseline_tcn_bendr.py
git commit -m "feat: add supervised tcn and bendr baselines"
```

### Task 11: CBraMod-style and AEMG-style comparison methods

**Files:**
- Create: `semg_fm/models/cbramod.py`
- Create: `semg_fm/models/aemg.py`
- Create: `semg_fm/objectives/vector_quantization.py`
- Create: `configs/model/cbramod.yaml`
- Create: `configs/model/aemg.yaml`
- Test: `tests/test_baseline_cbramod_aemg.py`

- [ ] **Step 1: Write failing structural and VQ tests**

```python
import torch
from semg_fm.models.cbramod import CBraModEncoder
from semg_fm.models.aemg import AEMGEncoder


def test_cbramod_preserves_channel_time_grid():
    out = CBraModEncoder(d_model=32, depth=1, heads=4, patch_samples=40)(torch.randn(2, 400, 16))
    assert out["tokens"].shape == (2, 16, 10, 32)
    assert out["reconstruction"].shape[:3] == (2, 16, 10)


def test_aemg_reports_codebook_health():
    out = AEMGEncoder(d_model=32, codebook_size=64, patch_samples=40)(torch.randn(2, 400, 16))
    assert out["token_ids"].dtype == torch.long
    assert 0.0 <= float(out["dead_code_fraction"]) <= 1.0
    assert float(out["perplexity"]) >= 1.0
```

- [ ] **Step 2: Verify missing-baseline failures**

Run: `python -m pytest tests/test_baseline_cbramod_aemg.py -q`  
Expected: FAIL with import errors.

- [ ] **Step 3: Implement both core-mechanism reproductions**

CBraMod uses the shared time/FFT patch encoder, ordinary channel-order conditional position convolution, parallel temporal and full spatial attention, and a waveform patch reconstruction head. It does not use ring-relative bias. AEMG detects contraction-weighted patches without dropping low-energy patches, applies nearest-neighbor VQ with commitment loss and EMA code updates, and predicts masked token IDs. `codebook_statistics` computes usage counts, perplexity, and the exact fraction of entries unused during an epoch.

- [ ] **Step 4: Run focused baseline tests**

Run: `python -m pytest tests/test_baseline_cbramod_aemg.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit CBraMod and AEMG baselines**

```powershell
git add semg_fm/models/cbramod.py semg_fm/models/aemg.py semg_fm/objectives/vector_quantization.py configs/model tests/test_baseline_cbramod_aemg.py
git commit -m "feat: add cbramod and aemg baselines"
```

### Task 12: Downstream heads, losses, and metrics

**Files:**
- Create: `semg_fm/models/heads.py`
- Create: `semg_fm/metrics/__init__.py`
- Create: `semg_fm/metrics/regression.py`
- Create: `semg_fm/metrics/events.py`
- Create: `semg_fm/metrics/text.py`
- Create: `semg_fm/tasks.py`
- Test: `tests/test_downstream_tasks.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write failing metric tests**

```python
from semg_fm.metrics.text import character_error_rate
from semg_fm.metrics.events import event_f1


def test_character_error_rate_uses_edit_distance():
    assert character_error_rate(["cat"], ["cut"]) == 1 / 3


def test_event_f1_matches_with_declared_tolerance():
    result = event_f1(pred=[("tap", 1.04)], target=[("tap", 1.0)], tolerance_s=0.05)
    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
```

Add one head/loss shape test for each of pose, wrist, gesture, handwriting CTC, and qwerty CTC.

- [ ] **Step 2: Verify task and metric tests fail**

Run: `python -m pytest tests/test_downstream_tasks.py tests/test_metrics.py -q`  
Expected: FAIL with missing modules.

- [ ] **Step 3: Implement task registry and metrics**

Regression heads return per-frame outputs and honor boolean target masks. Gesture uses frame logits with an explicit background class and deterministic event decoding. Text heads return time-major CTC log probabilities and use a vocabulary persisted with the checkpoint. Implement original-unit MAE/RMSE/Pearson, macro F1, balanced accuracy, tolerance-based one-to-one event matching, Levenshtein CER, normalized edit distance, and exact prompt accuracy.

- [ ] **Step 4: Run downstream unit tests**

Run: `python -m pytest tests/test_downstream_tasks.py tests/test_metrics.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit downstream contracts**

```powershell
git add semg_fm/models/heads.py semg_fm/metrics semg_fm/tasks.py tests/test_downstream_tasks.py tests/test_metrics.py
git commit -m "feat: add five semg downstream task contracts"
```

### Task 13: Fine-tuning, evaluation, and robustness entry points

**Files:**
- Create: `semg_fm/lightning/downstream_module.py`
- Create: `semg_fm/evaluation.py`
- Create: `finetune.py`
- Create: `evaluate.py`
- Create: `configs/task/emg2pose.yaml`
- Create: `configs/task/gni_wrist.yaml`
- Create: `configs/task/gni_gesture.yaml`
- Create: `configs/task/gni_handwriting.yaml`
- Create: `configs/task/emg2qwerty.yaml`
- Create: `configs/robustness/none.yaml`
- Create: `configs/robustness/channel_rotation.yaml`
- Create: `configs/robustness/channel_dropout.yaml`
- Create: `configs/robustness/gain_drift.yaml`
- Create: `configs/robustness/additive_noise.yaml`
- Test: `tests/test_downstream_smoke.py`
- Test: `tests/test_evaluation_outputs.py`

- [ ] **Step 1: Write failing five-task smoke and output tests**

```python
import json
from semg_fm.evaluation import write_evaluation_outputs


def test_evaluation_writes_machine_readable_outputs(tmp_path):
    write_evaluation_outputs(
        tmp_path,
        metrics={"cer": 0.25},
        per_subject=[{"subject_id": "s1", "cer": 0.25}],
        predictions={"record_id": ["r1"], "text": ["test"]},
    )
    assert json.loads((tmp_path / "metrics.json").read_text())["cer"] == 0.25
    assert (tmp_path / "per_subject_metrics.csv").is_file()
    assert (tmp_path / "predictions.h5").is_file()
```

Parameterize a Lightning `fast_dev_run` test over all five task configurations and the three fine-tuning modes: linear probe, partial fine-tune, and full fine-tune.

- [ ] **Step 2: Verify downstream integration tests fail**

Run: `python -m pytest tests/test_downstream_smoke.py tests/test_evaluation_outputs.py -q`  
Expected: FAIL because public downstream modules and writers are missing.

- [ ] **Step 3: Implement fine-tune modes and evaluation outputs**

`DownstreamModule` instantiates the task head, freezes backbone parameters according to mode, computes task loss, accumulates per-record predictions, and logs synchronized validation metrics. `evaluate.py` loads a checkpoint, applies one configured in-memory robustness transform, evaluates, and writes `metrics.json`, `per_subject_metrics.csv`, optional `predictions.h5`, copied split CSV, resolved config, and environment details. Rotation values are integer channel shifts; documentation maps 1, 2, 4, and 8 to 22.5, 45, 90, and 180 degrees.

- [ ] **Step 4: Run all downstream integration tests**

Run: `python -m pytest tests/test_downstream_smoke.py tests/test_evaluation_outputs.py -q`  
Expected: PASS on CPU.

- [ ] **Step 5: Commit downstream commands**

```powershell
git add semg_fm/lightning/downstream_module.py semg_fm/evaluation.py finetune.py evaluate.py configs/task configs/robustness tests/test_downstream_smoke.py tests/test_evaluation_outputs.py
git commit -m "feat: fine tune and evaluate five semg tasks"
```

### Task 14: Unified baseline Lightning command and CPU smoke suite

**Files:**
- Create: `semg_fm/lightning/baseline_module.py`
- Create: `run_baseline.py`
- Test: `tests/test_baseline_smoke.py`

- [ ] **Step 1: Write failing parameterized baseline smoke test**

```python
import pytest
from lightning import Trainer
from semg_fm.registry import build_baseline_module


@pytest.mark.parametrize("name", ["supervised_tcn", "bendr", "cbramod", "aemg"])
def test_baseline_completes_one_cpu_step(name, tiny_datamodule):
    module = build_baseline_module(name=name, d_model=16, depth=1)
    trainer = Trainer(accelerator="cpu", devices=1, fast_dev_run=True, logger=False, enable_checkpointing=False)
    trainer.fit(module, datamodule=tiny_datamodule)
    assert trainer.global_step == 1
```

- [ ] **Step 2: Verify baseline integration fails**

Run: `python -m pytest tests/test_baseline_smoke.py -q`  
Expected: FAIL because `build_baseline_module` and `BaselineModule` are incomplete.

- [ ] **Step 3: Implement stage-aware baseline training**

`run_baseline.py` accepts `model=<name>` and `stage=pretrain|finetune`. Supervised TCN rejects `stage=pretrain` with a clear error. BENDR logs continuous contrastive loss, CBraMod logs masked reconstruction loss, and AEMG logs token classification, commitment loss, perplexity, utilization, and dead-code fraction. Fine-tuning uses the same `DownstreamModule` heads and protocols as CIRCE-Myo.

- [ ] **Step 4: Run baseline smoke tests**

Run: `python -m pytest tests/test_baseline_smoke.py -q`  
Expected: PASS for four methods.

- [ ] **Step 5: Commit the baseline command**

```powershell
git add semg_fm/lightning/baseline_module.py semg_fm/registry.py run_baseline.py tests/test_baseline_smoke.py
git commit -m "feat: run fair semg comparison methods"
```

### Task 15: Reproducibility metadata and checkpoint resume

**Files:**
- Create: `semg_fm/utils/__init__.py`
- Create: `semg_fm/utils/reproducibility.py`
- Modify: `semg_fm/lightning/pretrain_module.py`
- Modify: `pretrain.py`
- Modify: `finetune.py`
- Modify: `evaluate.py`
- Modify: `run_baseline.py`
- Test: `tests/test_reproducibility.py`
- Test: `tests/test_checkpoint_resume.py`

- [ ] **Step 1: Write failing digest and resume tests**

```python
from semg_fm.utils.reproducibility import sha256_file, effective_global_batch


def test_effective_batch_records_world_size_and_accumulation():
    assert effective_global_batch(per_device=8, devices=4, accumulate=2) == 64


def test_manifest_digest_changes_when_contents_change(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("a", encoding="utf-8")
    first = sha256_file(path)
    path.write_text("b", encoding="utf-8")
    assert sha256_file(path) != first
```

The resume test trains one step, saves a checkpoint, loads it, and asserts equality for `global_step`, teacher state, and scheduler state before training a second step.

- [ ] **Step 2: Verify reproducibility tests fail**

Run: `python -m pytest tests/test_reproducibility.py tests/test_checkpoint_resume.py -q`  
Expected: FAIL with missing utilities or missing EMA/scheduler state.

- [ ] **Step 3: Implement run snapshots and exact resume**

Write resolved Hydra YAML, Git commit, manifest and split digests, Python/PyTorch/Lightning versions, CUDA visibility, world size, precision, seed, and effective global batch to each output directory. Keep EMA teacher and scheduler under the Lightning checkpoint state. Pass `ckpt_path` directly to `Trainer.fit` so resumed training retains optimizer and global step.

- [ ] **Step 4: Run reproducibility and resume tests**

Run: `python -m pytest tests/test_reproducibility.py tests/test_checkpoint_resume.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit reproducibility support**

```powershell
git add semg_fm/utils semg_fm/lightning/pretrain_module.py pretrain.py finetune.py evaluate.py run_baseline.py tests/test_reproducibility.py tests/test_checkpoint_resume.py
git commit -m "feat: record and resume reproducible semg runs"
```

### Task 16: User documentation and experiment protocol

**Files:**
- Modify: `README.md`
- Create: `PRETRAINING.md`
- Create: `DOWNSTREAM_EVALUATION.md`
- Create: `BASELINES.md`
- Create: `EXPERIMENT_PROTOCOL.md`
- Create: `H5_DATA_CONTRACT.md`
- Create: `SERVER_A100.md`
- Test: `tests/test_documented_commands.py`

- [ ] **Step 1: Write the failing documentation-command test**

```python
from pathlib import Path


def test_public_documents_include_all_entry_points():
    root = Path(__file__).parents[1]
    text = "\n".join((root / name).read_text(encoding="utf-8") for name in [
        "README.md", "PRETRAINING.md", "DOWNSTREAM_EVALUATION.md", "BASELINES.md", "SERVER_A100.md"
    ])
    for command in ["pretrain.py", "finetune.py", "evaluate.py", "run_baseline.py", "make_splits.py"]:
        assert command in text
```

- [ ] **Step 2: Verify documentation tests fail**

Run: `python -m pytest tests/test_documented_commands.py -q`  
Expected: FAIL because the documents do not exist.

- [ ] **Step 3: Write operational documentation**

`README.md` contains installation and a five-minute CPU smoke command. `PRETRAINING.md` explains shapes, masks, losses, and resume. `DOWNSTREAM_EVALUATION.md` specifies every label path, split rule, metric, and robustness command. `BASELINES.md` names each implementation as a core-mechanism reproduction and lists deviations. `EXPERIMENT_PROTOCOL.md` fixes main tables, ablations, seeds, label budgets, parameter reporting, and the no-language-model qwerty comparison. `H5_DATA_CONTRACT.md` mirrors the exact loader checks. `SERVER_A100.md` gives commands for 1, 2, 4, and 8 devices, Slurm-free launch, resume, NCCL diagnostics, and out-of-memory reduction while preserving effective batch size.

- [ ] **Step 4: Run documentation and Hydra composition tests**

Run: `python -m pytest tests/test_documented_commands.py tests/test_hydra_config.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit documentation**

```powershell
git add README.md PRETRAINING.md DOWNSTREAM_EVALUATION.md BASELINES.md EXPERIMENT_PROTOCOL.md H5_DATA_CONTRACT.md SERVER_A100.md tests/test_documented_commands.py
git commit -m "docs: document semg foundation model experiments"
```

### Task 17: Full verification and acceptance audit

**Files:**
- Modify: `.gitignore`
- Modify: `pytest.ini` if the existing temporary-directory setting is incompatible with the final suite

- [ ] **Step 1: Verify converter independence before changing ignore rules**

Run: `python -m pytest tests/test_standalone_scripts.py -q`. If the test file is unavailable in a later checkout, run these exact commands instead:

```powershell
python semg_h5/convert_emg2pose.py --help
python semg_h5/convert_emg2qwerty.py --help
python semg_h5/convert_gni_discrete_gestures.py --help
python semg_h5/convert_gni_handwriting.py --help
python semg_h5/convert_gni_wrist.py --help
```

Expected: all standalone converter checks PASS without importing `semg_fm`.

- [ ] **Step 2: Run the complete test suite**

Run: `python -m pytest -q`  
Expected: all tests PASS with no warnings promoted to errors.

- [ ] **Step 3: Run public CPU smoke commands**

```powershell
python pretrain.py trainer=cpu_smoke data.root=tests/fixtures/canonical_meta
python run_baseline.py model=bendr trainer=cpu_smoke data.root=tests/fixtures/canonical_meta
python finetune.py task=emg2pose trainer=cpu_smoke data.root=tests/fixtures/canonical_meta
python evaluate.py task=emg2pose trainer=cpu_smoke data.root=tests/fixtures/canonical_meta
```

Expected: every command exits with code 0 and writes a resolved config plus metrics in its Hydra output directory.

- [ ] **Step 4: Audit repository state and generated artifacts**

Run: `git status --short`  
Expected: only intended source, config, tests, and documentation changes appear. Dataset files, checkpoints, Hydra outputs, `.pytest_cache`, and Python bytecode are ignored. Existing untracked `emg_data/` and `meta_semg_16ch_datasets.md` remain untouched unless the user separately requests them.

- [ ] **Step 5: Commit final integration fixes**

```powershell
git add .gitignore pytest.ini semg_fm configs tests pretrain.py finetune.py evaluate.py run_baseline.py make_splits.py requirements-train.txt README.md PRETRAINING.md DOWNSTREAM_EVALUATION.md BASELINES.md EXPERIMENT_PROTOCOL.md H5_DATA_CONTRACT.md SERVER_A100.md
git commit -m "test: verify semg foundation model research stack"
```

## Completion Evidence

Before reporting completion, capture:

- the final `python -m pytest -q` summary;
- the four pretraining smoke-test outcomes;
- the five downstream smoke-test outcomes;
- the Ring Criss-Cross equivariance tolerance result;
- `git status --short` showing no unintended tracked changes;
- a note that NCCL and multi-GPU behavior still require the documented A100 server smoke test if the current machine has no multi-GPU CUDA environment.
