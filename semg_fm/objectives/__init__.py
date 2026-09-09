"""Self-supervised objectives for circular sEMG pre-training."""

from .kinematic import kinematic_loss
from .masked_latent import masked_latent_loss
from .masking import MaskGenerator
from .ring_equivariance import ring_equivariance_loss

__all__ = ["MaskGenerator", "kinematic_loss", "masked_latent_loss", "ring_equivariance_loss"]
