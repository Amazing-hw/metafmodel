"""Neural building blocks for circular 16-electrode sEMG models and baselines."""

from .bendr import BendrEncoder, BendrLikeContrastiveLoss, BendrPretrainingModel
from .patch_encoder import MultiScalePatchEncoder
from .ring_criss_cross import RingCrissCrossBlock, RingCrissCrossEncoder
from .ring_attention import RingMultiheadAttention, signed_ring_offsets
from .supervised_tcn import SupervisedTCN

__all__ = [
    "BendrEncoder",
    "BendrLikeContrastiveLoss",
    "BendrPretrainingModel",
    "MultiScalePatchEncoder",
    "RingCrissCrossBlock",
    "RingCrissCrossEncoder",
    "RingMultiheadAttention",
    "signed_ring_offsets",
    "SupervisedTCN",
]
