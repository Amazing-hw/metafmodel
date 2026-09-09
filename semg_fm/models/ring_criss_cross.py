"""Criss-cross temporal and circular-electrode encoding for sEMG."""

from __future__ import annotations

import torch
from torch import nn

from .patch_encoder import MultiScalePatchEncoder
from .ring_attention import RingMultiheadAttention


class RingCrissCrossBlock(nn.Module):
    """Fuse temporal and ring attention computed from one normalized token grid."""

    def __init__(self, d_model: int, heads: int, channels: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.temporal_attention = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.ring_attention = RingMultiheadAttention(d_model, heads, channels=channels)
        self.gate = nn.Linear(d_model, 2)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def fusion_weights(self, normalized_tokens: torch.Tensor) -> torch.Tensor:
        """Return the temporal/ring convex-combination weights for each token."""
        return torch.softmax(self.gate(normalized_tokens), dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        electrode_observed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply a pre-norm attention and pointwise-FFN residual pair."""
        batch, channels, patches, width = tokens.shape
        normalized = self.attention_norm(tokens)
        temporal_input = normalized.reshape(batch * channels, patches, width)
        temporal_output, _ = self.temporal_attention(
            temporal_input, temporal_input, temporal_input, need_weights=False
        )
        temporal_output = temporal_output.reshape(batch, channels, patches, width)
        ring_output = self.ring_attention(normalized, electrode_observed=electrode_observed)
        weights = self.fusion_weights(normalized)
        fused = weights[..., :1] * temporal_output + weights[..., 1:] * ring_output
        tokens = tokens + self.attention_dropout(fused)
        return tokens + self.ffn(self.ffn_norm(tokens))


class RingCrissCrossEncoder(nn.Module):
    """Encode a circular electrode ring into equivariant and pooled motion tokens."""

    def __init__(
        self,
        d_model: int,
        depth: int,
        heads: int,
        patch_samples: int,
        channels: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if channels < 2:
            raise ValueError("channels must be at least 2")

        self.d_model = d_model
        self.channels = channels
        self.patch_samples = patch_samples
        # Encode every electrode through the same single-channel encoder.  This
        # avoids any channel-index parameters before circular attention.
        self.patch_encoder = MultiScalePatchEncoder(
            channels=1, d_model=d_model, patch_samples=patch_samples
        )
        self.blocks = nn.ModuleList(
            RingCrissCrossBlock(d_model, heads, channels=channels, dropout=dropout)
            for _ in range(depth)
        )

    def forward(
        self,
        emg: torch.Tensor,
        electrode_observed: torch.Tensor | None = None,
        *,
        observation_mask: torch.Tensor | None = None,
        electrode_observation_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return ``z_eq`` [B, C, N, D] and masked-mean ``z_motion`` [B, N, D].

        Observation masks may describe electrodes as ``[B, C]``, patches as
        ``[B, C, N]``, or individual input samples as ``[B, T, C]``.
        """
        masks = [
            mask
            for mask in (electrode_observed, observation_mask, electrode_observation_mask)
            if mask is not None
        ]
        if len(masks) > 1:
            raise ValueError("pass only one electrode observation mask")
        if emg.ndim != 3:
            raise ValueError("emg must have shape [batch, time, channels]")
        batch, samples, channels = emg.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")
        if samples < self.patch_samples or samples % self.patch_samples:
            raise ValueError("time length must be a positive multiple of patch_samples")

        patches = samples // self.patch_samples
        observed = self._patch_observation_mask(
            masks[0] if masks else None, batch, samples, patches, emg.device
        )
        channel_major = emg.transpose(1, 2).reshape(batch * channels, samples, 1)
        tokens = self.patch_encoder(channel_major).reshape(batch, channels, patches, self.d_model)
        for block in self.blocks:
            tokens = block(tokens, electrode_observed=observed)

        pooled_weights = observed.unsqueeze(-1).to(tokens.dtype)
        z_motion = (tokens * pooled_weights).sum(dim=1) / pooled_weights.sum(dim=1).clamp_min(1.0)
        return {"z_eq": tokens, "z_motion": z_motion}

    def _patch_observation_mask(
        self,
        mask: torch.Tensor | None,
        batch: int,
        samples: int,
        patches: int,
        device: torch.device,
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones(batch, self.channels, patches, dtype=torch.bool, device=device)
        if mask.shape == (batch, self.channels):
            return mask.to(device=device, dtype=torch.bool).unsqueeze(-1).expand(-1, -1, patches)
        if mask.shape == (batch, self.channels, patches):
            return mask.to(device=device, dtype=torch.bool)
        if mask.shape == (batch, samples, self.channels):
            sample_mask = mask.to(device=device, dtype=torch.bool)
            return sample_mask.reshape(batch, patches, self.patch_samples, self.channels).all(dim=2).permute(0, 2, 1)
        raise ValueError("electrode observation mask must have shape [B, C], [B, C, N], or [B, T, C]")
