"""Behavioral contracts for Lightning pretraining and EMA persistence."""

from __future__ import annotations

import torch
from lightning import Trainer
from torch.utils.data import DataLoader, Dataset

import semg_fm.lightning.pretrain_module as pretrain_module
from semg_fm.models import RingCrissCrossEncoder
from semg_fm.lightning.pretrain_module import PretrainModule


class _TinyPretrainDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        generator = torch.Generator().manual_seed(index)
        return {
            "emg": torch.randn(80, 16, generator=generator),
            "sample_mask": torch.ones(80, dtype=torch.bool),
            "task_id": "pose",
        }


def _collate(examples):
    return {
        "emg": torch.stack([example["emg"] for example in examples]),
        "sample_mask": torch.stack([example["sample_mask"] for example in examples]),
        "task_id": examples[0]["task_id"],
    }


def _module() -> PretrainModule:
    return PretrainModule(
        model_config={"d_model": 8, "depth": 1, "heads": 2, "patch_samples": 40},
        ema_decay=0.5,
        learning_rate=1e-2,
        warmup_steps=0,
        max_steps=1,
    )


def _trainer() -> Trainer:
    return Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )


def test_teacher_starts_as_frozen_copy_of_student():
    module = _module()
    for student, teacher in zip(module.student.parameters(), module.teacher.parameters()):
        assert torch.equal(student, teacher)
        assert not teacher.requires_grad


def test_ring_equivariance_is_a_detached_diagnostic_not_an_optimized_loss(monkeypatch):
    module = _module()
    module.latent_weight = 2.0
    module.ring_weight = 3.0
    module.kinematic_weight = 5.0
    monkeypatch.setattr(
        pretrain_module,
        "ring_equivariance_loss",
        lambda field, motion, shifted_field, shifted_motion, shift: field.new_tensor(7.0),
    )
    batch = _collate([_TinyPretrainDataset()[0], _TinyPretrainDataset()[1]])

    total, losses = module._losses(batch)

    assert "ring_equivariance_error" in losses
    assert not losses["ring_equivariance_error"].requires_grad
    assert torch.equal(total, module.latent_weight * losses["latent"] + module.kinematic_weight * losses["kinematic"])


def test_log_records_ring_equivariance_as_a_metric_without_loss_suffix(monkeypatch):
    module = _module()
    names: list[str] = []
    monkeypatch.setattr(module, "log", lambda name, value, **kwargs: names.append(name))

    module._log(
        "train",
        torch.tensor(1.0),
        {"latent": torch.tensor(2.0), "ring_equivariance_error": torch.tensor(3.0), "kinematic": torch.tensor(4.0)},
        batch_size=2,
    )

    assert "train/ring_equivariance_error" in names
    assert "train/ring_equivariance_error_loss" not in names


def test_ema_updates_only_after_optimizer_step_and_teacher_has_no_grads():
    module = _module()
    before = {key: value.clone() for key, value in module.teacher.state_dict().items()}
    loader = DataLoader(_TinyPretrainDataset(), batch_size=2, collate_fn=_collate)

    _trainer().fit(module, train_dataloaders=loader)

    assert module.ema_updates == 1
    for key, value in module.teacher.state_dict().items():
        expected = before[key] * 0.5 + module.student.state_dict()[key] * 0.5
        assert torch.equal(value, expected)
    assert all(parameter.grad is None for parameter in module.teacher.parameters())


def test_ema_teacher_is_saved_and_restored_exactly(tmp_path):
    module = _module()
    trainer = _trainer()
    trainer.fit(module, train_dataloaders=DataLoader(_TinyPretrainDataset(), batch_size=2, collate_fn=_collate))
    before = {key: value.clone() for key, value in module.teacher.state_dict().items()}
    path = tmp_path / "state.ckpt"
    trainer.save_checkpoint(path)

    restored = PretrainModule.load_from_checkpoint(path)

    assert restored.ema_updates == 1
    for key, value in before.items():
        assert torch.equal(value, restored.teacher.state_dict()[key])


def test_direct_student_round_trip_serializes_complete_architecture(tmp_path):
    student = RingCrissCrossEncoder(d_model=8, depth=1, heads=2, patch_samples=40)
    module = PretrainModule(student=student, model_config={"d_model": 8})
    trainer = _trainer()
    trainer.fit(module, train_dataloaders=DataLoader(_TinyPretrainDataset(), batch_size=2, collate_fn=_collate))
    path = tmp_path / "direct-student.ckpt"
    trainer.save_checkpoint(path)

    restored = PretrainModule.load_from_checkpoint(path)

    assert restored.model_config == {
        "d_model": 8, "depth": 1, "heads": 2, "patch_samples": 40,
        "channels": 16, "dropout": 0.0,
    }
    assert restored.student.patch_samples == 40
