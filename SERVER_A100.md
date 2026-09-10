# A100 Server Guide

Launch, resume, and troubleshoot multi-GPU runs. All commands assume the
repository root as working directory and absolute data paths
(`hydra.job.chdir=true`).

## Setup

```bash
python -m pip install -r requirements-train.txt
# Sanity: one CPU step with synthetic fixtures
python -m pytest tests -q
```

## Launch by device count

`strategy=auto` selects DDP for multiple GPUs. Keep the **effective global
batch size** fixed when changing device count: with the base
`batch_size=8`:

| devices | accumulate_grad_batches | effective global batch |
|---|---|---|
| 1 | 8 | 64 |
| 2 | 4 | 64 |
| 4 | 2 | 64 |
| 8 | 1 | 64 |

```bash
# Pretraining on 1 / 2 / 4 / 8 devices
python pretrain.py trainer=a100_ddp trainer.devices=1 trainer.accumulate_grad_batches=8 \
  data.root=CANONICAL_META data.manifest_path=CANONICAL_META/splits.csv seed=0
python pretrain.py trainer=a100_ddp trainer.devices=2 trainer.accumulate_grad_batches=4 ...
python pretrain.py trainer=a100_ddp trainer.devices=4 trainer.accumulate_grad_batches=2 ...
python pretrain.py trainer=a100_ddp trainer.devices=8 trainer.accumulate_grad_batches=1 ...

# Downstream fine-tune and evaluate use the same trainer group
python finetune.py trainer=a100_ddp trainer.devices=4 task=pose \
  data.root=... data.manifest_path=... finetune_mode=full label_budget=1.0
python evaluate.py trainer=a100_ddp trainer.devices=4 task=pose \
  checkpoint_path=outputs/<run>/checkpoints/last.ckpt data.root=... data.manifest_path=...
```

No Slurm integration is required; launch from a plain shell on the node.

## Resume

Checkpoints retain optimizer, scheduler, EMA teacher, and global step; resume
with the same effective batch size as the original run. For deterministic
continuation, use `last.ckpt` created after a completed epoch (an
epoch-boundary checkpoint):

```bash
python pretrain.py trainer=a100_ddp resume_from=outputs/<run>/checkpoints/last.ckpt ...
python finetune.py trainer=a100_ddp resume_from=outputs/<run>/checkpoints/last.ckpt ...
```

Generic PyTorch dataloaders cannot reproduce an arbitrary in-epoch position.
If a run stops during an epoch, use the most recent completed epoch's
`last.ckpt`; do not expect a mid-epoch resume to replay the interrupted
position deterministically.

## DDP smoke test (one batch)

Before long runs, verify NCCL across N devices:

```bash
python run_baseline.py stage=pretrain model=bendr trainer=a100_ddp trainer.devices=4 \
  trainer.fast_dev_run=true data.root=... data.manifest_path=... data.num_workers=4
```

The test suite does not claim NCCL coverage; run this smoke on the server.

## Diagnostics

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET python pretrain.py ...
nvidia-smi                          # device health and memory
```

For recovery, confirm that the selected `last.ckpt` was written after a
completed epoch before relying on deterministic continuation.

## Out-of-memory reductions (preserving effective batch)

1. `data.batch_size=4` with `trainer.accumulate_grad_batches=16` (or the
   matching multiple for your device count).
2. `trainer.precision=bf16-mixed` is the default; do not switch to fp32
   before trying the above.
3. Reduce model size: `model.d_model=64 model.depth=4`.
4. `data.num_workers=4` (or 8) with `trainer.devices>1` to keep dataloading
   off the critical path.

Every run writes `run_metadata.json` with the effective global batch size,
manifest digest, git commit, and environment — attach it to any reported
number.
