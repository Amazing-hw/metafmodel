"""EMA teacher-student Lightning module for CIRCE-Myo pretraining."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import torch
from lightning import LightningModule
from torch import nn

from semg_fm.models import RingCrissCrossEncoder
from semg_fm.objectives import MaskGenerator, kinematic_loss, masked_latent_loss, ring_equivariance_loss
from semg_fm.registry import build_model


class PretrainModule(LightningModule):
    """CIRCE-Myo student with a frozen teacher updated after optimizer steps."""

    def __init__(self, student: RingCrissCrossEncoder | None = None, *, model_name: str = "ring_criss_cross", model_config: Mapping[str, Any] | None = None, mask_config: Mapping[str, Any] | None = None, kinematic_dim: int | None = None, latent_weight: float = 1.0, ring_weight: float = 1.0, kinematic_weight: float = 1.0, ema_decay: float = 0.996, learning_rate: float = 1e-3, weight_decay: float = 0.05, warmup_steps: int = 1_000, max_steps: int = 100_000) -> None:
        super().__init__()
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if warmup_steps < 0 or max_steps < 1:
            raise ValueError("warmup_steps must be non-negative and max_steps must be positive")
        if kinematic_dim is not None and kinematic_dim < 1:
            raise ValueError("kinematic_dim must be positive when provided")
        config = dict(model_config or {})
        config.pop("name", None)
        if student is None:
            student = build_model(model_name, **config)
        else:
            # A supplied student is authoritative: never serialize a partial
            # caller mapping that would make checkpoint reconstruction fail.
            config = _model_config(student)
        self.student = student
        self.teacher = copy.deepcopy(student)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.model_name, self.model_config = model_name, config
        self.mask_config = dict(mask_config or {})
        self.mask_generator = MaskGenerator(**self.mask_config)
        self.kinematic_adapter = nn.Linear(student.d_model, kinematic_dim) if kinematic_dim else None
        self.latent_weight, self.ring_weight = float(latent_weight), float(ring_weight)
        self.kinematic_weight, self.ema_decay = float(kinematic_weight), float(ema_decay)
        self.learning_rate, self.weight_decay = float(learning_rate), float(weight_decay)
        self.warmup_steps, self.max_steps = int(warmup_steps), int(max_steps)
        self.register_buffer("_ema_updates", torch.zeros((), dtype=torch.long), persistent=True)
        self.save_hyperparameters({
            "model_name": self.model_name,
            "model_config": self.model_config,
            "mask_config": self.mask_config,
            "kinematic_dim": kinematic_dim,
            "latent_weight": self.latent_weight,
            "ring_weight": self.ring_weight,
            "kinematic_weight": self.kinematic_weight,
            "ema_decay": self.ema_decay,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
        })

    @property
    def ema_updates(self) -> int:
        return int(self._ema_updates.item())

    def forward(self, emg: torch.Tensor, observation_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self.student(emg, observation_mask=observation_mask)

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        total, losses = self._losses(batch)
        self._log("train", total, losses, int(batch["emg"].shape[0]))
        return total

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        total, losses = self._losses(batch)
        batch_size = int(batch["emg"].shape[0])
        self._log("val", total, losses, batch_size)
        task = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(batch.get("task_id", "unknown")))
        self.log(f"val/{task}_loss", total, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return total

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW((parameter for parameter in self.parameters() if parameter.requires_grad), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._lr_scale)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_closure=None) -> None:
        """Advance teacher state strictly after the optimizer update."""
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        with torch.no_grad():
            teacher_state = self.teacher.state_dict(keep_vars=True)
            for name, student_value in self.student.state_dict(keep_vars=True).items():
                teacher_value = teacher_state[name]
                if teacher_value.is_floating_point():
                    teacher_value.mul_(self.ema_decay).add_(student_value.detach(), alpha=1 - self.ema_decay)
                else:
                    teacher_value.copy_(student_value.detach())
            self._ema_updates.add_(1)

    def _losses(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        emg = batch["emg"]
        if not isinstance(emg, torch.Tensor) or emg.ndim != 3:
            raise ValueError("batch emg must be [B, T, C]")
        batch_size, samples, channels = emg.shape
        if channels != self.student.channels or samples % self.student.patch_samples:
            raise ValueError("emg channels and sample count must match the backbone")
        observed = _observed_samples(batch.get("sample_mask"), emg)
        patches = samples // self.student.patch_samples
        token_observed = observed.reshape(batch_size, patches, self.student.patch_samples, channels).any(2).permute(0, 2, 1)
        token_mask = self.mask_generator(batch_size, patches, device=emg.device) & token_observed
        if not token_mask.any():
            token_mask = token_observed
        student_observed = observed & _expand_grid(~token_mask, self.student.patch_samples)
        self.teacher.eval()
        with torch.no_grad():
            teacher = self.teacher(emg * observed.to(emg.dtype), observation_mask=observed)
        student_input = emg * student_observed.to(emg.dtype)
        student = self.student(student_input, observation_mask=student_observed)
        shifted = self.student(torch.roll(student_input, 1, -1), observation_mask=torch.roll(student_observed, 1, -1))
        latent = masked_latent_loss(student["z_eq"], teacher["z_eq"], token_mask)
        ring = ring_equivariance_loss(student["z_eq"], student["z_motion"], shifted["z_eq"], shifted["z_motion"], 1)
        kinematic = self._kinematic(student["z_motion"], batch)
        total = self.latent_weight * latent + self.ring_weight * ring + self.kinematic_weight * kinematic
        return total, {"latent": latent, "ring": ring, "kinematic": kinematic}

    def _kinematic(self, motion: torch.Tensor, batch: Mapping[str, Any]) -> torch.Tensor:
        if self.kinematic_adapter is None:
            return motion.reshape(-1)[:0].sum()
        prediction = self.kinematic_adapter(motion)
        target, target_mask = batch.get("target"), batch.get("target_mask")
        if not isinstance(target, torch.Tensor) or not isinstance(target_mask, torch.Tensor):
            return kinematic_loss(prediction, None, None)
        if target.ndim != 3:
            raise ValueError("kinematic target must be [B, T, D]")
        if target.shape[1] != prediction.shape[1]:
            if target.shape[1] % prediction.shape[1]:
                raise ValueError("kinematic target must align to patches")
            width = target.shape[1] // prediction.shape[1]
            target = target.reshape(target.shape[0], prediction.shape[1], width, target.shape[2]).mean(2)
            target_mask = target_mask.reshape(target.shape[0], prediction.shape[1], width, *target_mask.shape[2:]).all(2)
        return kinematic_loss(prediction, target.to(prediction), target_mask.to(prediction.device))

    def _log(self, stage: str, total: torch.Tensor, losses: Mapping[str, torch.Tensor], batch_size: int) -> None:
        on_step = stage == "train"
        self.log(f"{stage}/loss", total, on_step=on_step, on_epoch=True, sync_dist=True, batch_size=batch_size)
        for name, value in losses.items():
            self.log(f"{stage}/{name}_loss", value, on_step=on_step, on_epoch=True, sync_dist=True, batch_size=batch_size)

    def _lr_scale(self, step: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        span = max(1, self.max_steps - self.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(max(step - self.warmup_steps, 0), span) / span))


def _model_config(student: RingCrissCrossEncoder) -> dict[str, int | float]:
    if not isinstance(student, RingCrissCrossEncoder):
        raise TypeError("student must be RingCrissCrossEncoder when model_config is omitted")
    return {"d_model": student.d_model, "depth": len(student.blocks), "heads": student.blocks[0].temporal_attention.num_heads, "patch_samples": student.patch_samples, "channels": student.channels, "dropout": 0.0}


def _observed_samples(mask: Any, emg: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(emg, dtype=torch.bool)
    if not isinstance(mask, torch.Tensor):
        raise ValueError("sample_mask must be a tensor")
    if mask.shape == emg.shape[:2]:
        return mask.to(device=emg.device, dtype=torch.bool).unsqueeze(-1).expand_as(emg)
    if mask.shape == emg.shape:
        return mask.to(device=emg.device, dtype=torch.bool)
    raise ValueError("sample_mask must be [B, T] or [B, T, C]")


def _expand_grid(grid: torch.Tensor, patch_samples: int) -> torch.Tensor:
    return grid.permute(0, 2, 1).repeat_interleave(patch_samples, dim=1)
