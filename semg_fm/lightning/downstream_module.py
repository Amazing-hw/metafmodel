"""Lightning fine-tuning module for the five canonical downstream tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F

from semg_fm.metrics.regression import masked_regression_metrics
from semg_fm.models.heads import CTCTextHead, GestureEventHead, TemporalRegressionHead
from semg_fm.tasks import task_kind, task_spec


class DownstreamModule(LightningModule):
    """Fine-tune a backbone using task-specific labels from canonical metadata.

    Canonical loaders keep heterogeneous labels in ``batch['metadata']`` so an
    HDF5 window stays faithful to its source format.  Batches are task
    homogeneous; this module validates that invariant before constructing the
    corresponding supervised target.  DDP evaluation duplicates are ignored
    through ``example_weight`` rather than silently counted twice.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        task_id: str,
        output_dim: int | None = None,
        target_name: str | None = None,
        class_names: Sequence[str] = (),
        vocabulary: Sequence[str] = (),
        learning_rate: float = 1e-3,
        weight_decay: float = 0.05,
    ) -> None:
        super().__init__()
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not hasattr(backbone, "d_model") or not isinstance(backbone.d_model, int):
            raise ValueError("backbone must expose integer d_model")
        self.backbone = backbone
        self.task_id = task_spec(task_id).task_id
        self.target_name = target_name
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        kind = task_kind(task_id)
        if kind == "regression":
            if output_dim is None or output_dim < 1:
                raise ValueError("regression tasks require positive output_dim")
            self.head: nn.Module = TemporalRegressionHead(backbone.d_model, output_dim)
        elif kind == "events":
            self.head = GestureEventHead(backbone.d_model, class_names)
        else:
            self.head = CTCTextHead(backbone.d_model, vocabulary)
        # Avoid serializing the potentially large backbone object in hparams;
        # entry points separately snapshot its config and source checkpoint.
        self.save_hyperparameters(ignore=["backbone", "head"])

    def forward(self, emg: torch.Tensor, sample_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return temporal backbone features plus a token-validity mask."""
        if emg.ndim != 3:
            raise ValueError("emg must have shape [B,T,C]")
        if sample_mask is None:
            sample_mask = torch.ones(emg.shape[:2], dtype=torch.bool, device=emg.device)
        if sample_mask.shape != emg.shape[:2] or sample_mask.dtype is not torch.bool:
            raise ValueError("sample_mask must be boolean [B,T]")
        observed = sample_mask.to(emg.device).unsqueeze(-1).expand_as(emg)
        outputs = self.backbone(emg, observation_mask=observed)
        if not isinstance(outputs, Mapping) or "z_motion" not in outputs:
            raise ValueError("backbone must return mapping with z_motion [B,N,D]")
        features = outputs["z_motion"]
        if features.ndim != 3:
            raise ValueError("backbone z_motion must have shape [B,N,D]")
        patch_samples = getattr(self.backbone, "patch_samples", None)
        if not isinstance(patch_samples, int) or patch_samples < 1 or emg.shape[1] % patch_samples:
            raise ValueError("backbone must expose patch_samples dividing input length")
        token_mask = sample_mask.reshape(emg.shape[0], -1, patch_samples).any(dim=-1)
        if features.shape[:2] != token_mask.shape:
            raise ValueError("backbone z_motion shape does not align with patch_samples")
        return {"features": features, "token_mask": token_mask, "backbone_outputs": outputs}

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        """Compute and log one task-homogeneous supervised loss."""
        loss, metrics = self._loss_and_metrics(batch)
        self._log("train", loss, metrics, int(batch["emg"].shape[0]))
        return loss

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        """Compute validation loss while preserving per-task metric names."""
        loss, metrics = self._loss_and_metrics(batch)
        self._log("val", loss, metrics, int(batch["emg"].shape[0]))
        return loss

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        """Compute held-out loss with the same task and padding protections."""
        loss, metrics = self._loss_and_metrics(batch)
        self._log("test", loss, metrics, int(batch["emg"].shape[0]))
        return loss

    def configure_optimizers(self):
        """Optimize the selected backbone and its downstream head together."""
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def load_backbone_checkpoint(self, checkpoint_path: str) -> None:
        """Load a pretraining ``student.*`` state dict into this module's backbone."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint must contain a state_dict mapping")
        student_state = {
            name.removeprefix("student."): value for name, value in state.items() if name.startswith("student.")
        }
        if not student_state:
            student_state = {name.removeprefix("backbone."): value for name, value in state.items() if name.startswith("backbone.")}
        if not student_state:
            student_state = {name.removeprefix("model.encoder."): value for name, value in state.items() if name.startswith("model.encoder.")}
        if not student_state:
            raise ValueError("checkpoint has neither student.*, backbone.*, nor model.encoder.* weights")
        self.backbone.load_state_dict(student_state, strict=True)

    def _loss_and_metrics(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if str(batch.get("task_id")) != self.task_id:
            raise ValueError(f"batch task_id {batch.get('task_id')!r} does not match module task_id {self.task_id!r}")
        emg, sample_mask = batch.get("emg"), batch.get("sample_mask")
        if not isinstance(emg, torch.Tensor) or not isinstance(sample_mask, torch.Tensor):
            raise ValueError("batch requires tensor emg and sample_mask")
        metadata = batch.get("metadata")
        if not isinstance(metadata, list) or len(metadata) != emg.shape[0]:
            raise ValueError("batch metadata must be one mapping per example")
        weights = batch.get("example_weight")
        if weights is None:
            weights = torch.ones(emg.shape[0], device=emg.device, dtype=emg.dtype)
        if not isinstance(weights, torch.Tensor) or weights.shape != (emg.shape[0],):
            raise ValueError("example_weight must be [B]")
        encoded = self(emg, sample_mask.to(device=emg.device, dtype=torch.bool))
        kind = task_kind(self.task_id)
        if kind == "regression":
            return self._regression_loss(encoded["features"], metadata, weights)
        if kind == "events":
            return self._event_loss(encoded["features"], encoded["token_mask"], metadata, weights, emg.shape[1])
        return self._ctc_loss(encoded["features"], encoded["token_mask"], metadata, weights)

    def _regression_loss(
        self, features: torch.Tensor, metadata: list[Mapping[str, Any]], example_weight: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert isinstance(self.head, TemporalRegressionHead)
        targets, masks = self._regression_targets(metadata, features.device)
        prediction = self.head(features)
        prediction = F.interpolate(prediction.transpose(1, 2), size=targets.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        metrics = masked_regression_metrics(prediction, targets, masks, example_weight=example_weight.to(features))
        loss = metrics["sum_squared_error"] / metrics["count"].clamp_min(1.0)
        return loss, {"mae": metrics["mae"], "rmse": metrics["rmse"], "count": metrics["count"]}

    def _regression_targets(
        self, metadata: list[Mapping[str, Any]], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values, masks = [], []
        for item in metadata:
            target, target_mask = item.get("target"), item.get("target_mask")
            if self.task_id == "wrist_angle":
                if not self.target_name or not isinstance(target, Mapping) or not isinstance(target_mask, Mapping):
                    raise ValueError("wrist_angle requires target_name present in target and target_mask mappings")
                target, target_mask = target.get(self.target_name), target_mask.get(self.target_name)
            if not isinstance(target, torch.Tensor) or not isinstance(target_mask, torch.Tensor):
                raise ValueError("regression metadata requires tensor target and target_mask")
            if target.ndim == 1:
                target = target.unsqueeze(-1)
            if target.ndim != 2 or target_mask.shape[0] != target.shape[0]:
                raise ValueError("regression targets must be [T,D] with aligned mask")
            values.append(target.to(device=device, dtype=torch.float32))
            masks.append(target_mask.to(device=device, dtype=torch.bool))
        targets = torch.stack(values)
        if targets.shape[-1] != getattr(self.head, "output_dim"):
            raise ValueError("configured output_dim does not match canonical target dimension")
        return targets, torch.stack(masks)

    def _event_loss(
        self, features: torch.Tensor, token_mask: torch.Tensor, metadata: list[Mapping[str, Any],], example_weight: torch.Tensor,
        input_samples: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert isinstance(self.head, GestureEventHead)
        logits = self.head(features)
        targets = torch.zeros(token_mask.shape, dtype=torch.long, device=features.device)
        for batch, item in enumerate(metadata):
            rate = float(item.get("sample_rate_hz", 0.0))
            if rate <= 0:
                raise ValueError("event metadata requires positive sample_rate_hz")
            for label, relative_time in item.get("events", []):
                if label not in self.head.class_names:
                    continue
                index = min(token_mask.shape[1] - 1, max(0, int(float(relative_time) * rate / input_samples * token_mask.shape[1])))
                if token_mask[batch, index]:
                    targets[batch, index] = self.head.class_names.index(label) + 1
        per_token = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
        weights = token_mask.to(per_token.dtype) * example_weight.to(per_token).unsqueeze(-1)
        loss = (per_token * weights).sum() / weights.sum().clamp_min(1.0)
        return loss, {"event_loss": loss.detach(), "event_tokens": weights.sum()}

    def _ctc_loss(
        self, features: torch.Tensor, token_mask: torch.Tensor, metadata: list[Mapping[str, Any]], example_weight: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert isinstance(self.head, CTCTextHead)
        encoded = []
        for item in metadata:
            text = item.get("target_text", "")
            if not isinstance(text, str):
                raise ValueError("CTC metadata target_text must be a string")
            try:
                encoded.append([self.head.vocabulary.index(character) + 1 for character in text])
            except ValueError as error:
                raise ValueError("target_text contains a character absent from configured vocabulary") from error
        max_length = max((len(row) for row in encoded), default=0)
        targets = torch.zeros((len(encoded), max_length), dtype=torch.long, device=features.device)
        for index, row in enumerate(encoded):
            if row:
                targets[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=features.device)
        target_lengths = torch.tensor([len(row) for row in encoded], dtype=torch.long, device=features.device)
        per_example = self.head.loss(
            self.head(features), targets=targets, input_lengths=token_mask.sum(dim=-1), target_lengths=target_lengths, reduction="none"
        )
        loss = (per_example * example_weight.to(per_example)).sum() / example_weight.sum().clamp_min(1.0)
        return loss, {"ctc_loss": loss.detach(), "characters": target_lengths.sum().to(features.dtype)}

    def _log(self, stage: str, loss: torch.Tensor, metrics: Mapping[str, torch.Tensor], batch_size: int) -> None:
        # Unit tests and library callers may invoke a step before Lightning has
        # attached a Trainer; the loss remains usable but there is nowhere safe
        # to emit framework metrics yet.
        if self._trainer is None:
            return
        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=True, sync_dist=True, batch_size=batch_size)
        for name, value in metrics.items():
            self.log(f"{stage}/{self.task_id}_{name}", value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)


__all__ = ["DownstreamModule"]
