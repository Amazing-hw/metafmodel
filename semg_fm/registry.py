"""Explicit model registry used by public entry points."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import RingCrissCrossEncoder

MODEL_REGISTRY = {"ring_criss_cross": RingCrissCrossEncoder}


def build_model(name: str, /, **kwargs: Any) -> RingCrissCrossEncoder:
    """Build a registered backbone."""
    try:
        return MODEL_REGISTRY[name](**kwargs)
    except KeyError as error:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}") from error


def model_from_config(config: Mapping[str, Any]) -> RingCrissCrossEncoder:
    """Build a model from a Hydra-compatible mapping with a ``name`` field."""
    values = dict(config)
    try:
        name = str(values.pop("name"))
    except KeyError as error:
        raise ValueError("model config must include name") from error
    return build_model(name, **values)
