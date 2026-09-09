"""Validation helpers for training configuration."""


def validate_devices(devices: int | str) -> None:
    """Validate a Lightning device count, allowing automatic selection."""
    if devices == "auto":
        return
    if isinstance(devices, bool) or not isinstance(devices, int) or not 1 <= devices <= 8:
        raise ValueError("devices must be 'auto' or an integer from 1 through 8")


def validate_seed(seed: int) -> None:
    """Require an explicit non-negative integral random seed."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
