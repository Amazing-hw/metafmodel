"""Smoke contracts for Lightning downstream fine-tuning on canonical batches."""

from __future__ import annotations

import torch
from torch import nn

from semg_fm.lightning.downstream_module import DownstreamModule


class _TinyBackbone(nn.Module):
    """Small deterministic backbone exposing the shared temporal interface."""

    d_model = 6
    patch_samples = 20
    channels = 16

    def forward(self, emg: torch.Tensor, observation_mask: torch.Tensor | None = None):
        batch, samples, _ = emg.shape
        tokens = emg.reshape(batch, samples // 20, 20, 16).mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        return {"z_motion": tokens.expand(-1, -1, self.d_model)}


def test_downstream_regression_uses_metadata_targets_and_excludes_padding_weight():
    module = DownstreamModule(backbone=_TinyBackbone(), task_id="pose", output_dim=2, learning_rate=1e-3)
    batch = {
        "emg": torch.randn(2, 100, 16),
        "sample_mask": torch.ones(2, 100, dtype=torch.bool),
        "task_id": "pose",
        "example_weight": torch.tensor([1.0, 0.0]),
        "metadata": [
            {"target": torch.zeros(100, 2), "target_mask": torch.ones(100, dtype=torch.bool)},
            {"target": torch.full((100, 2), 10_000.0), "target_mask": torch.ones(100, dtype=torch.bool)},
        ],
    }

    loss = module.training_step(batch, 0)
    loss.backward()

    assert torch.isfinite(loss)
    assert module.head.projection.weight.grad is not None


def test_downstream_module_rejects_a_batch_for_a_different_task():
    module = DownstreamModule(backbone=_TinyBackbone(), task_id="pose", output_dim=2)
    batch = {"emg": torch.randn(1, 100, 16), "sample_mask": torch.ones(1, 100, dtype=torch.bool), "task_id": "typing", "metadata": [], "example_weight": torch.ones(1)}

    try:
        module.training_step(batch, 0)
    except ValueError as error:
        assert "task_id" in str(error)
    else:
        raise AssertionError("expected task mismatch to be rejected")


def test_downstream_ctc_uses_per_example_loss_before_padding_weighting():
    module = DownstreamModule(
        backbone=_TinyBackbone(), task_id="handwriting", vocabulary=("a", "b"), learning_rate=1e-3
    )
    batch = {
        "emg": torch.randn(2, 100, 16),
        "sample_mask": torch.ones(2, 100, dtype=torch.bool),
        "task_id": "handwriting",
        "example_weight": torch.tensor([1.0, 0.0]),
        "metadata": [{"target_text": "ab"}, {"target_text": "ba"}],
    }

    loss = module.training_step(batch, 0)
    loss.backward()

    assert torch.isfinite(loss)
    assert module.head.classifier.weight.grad is not None
