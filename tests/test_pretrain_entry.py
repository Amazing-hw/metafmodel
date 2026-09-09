"""Configuration validation contracts for the public pretraining entry point."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from pretrain import build_datamodule


def test_pretrain_requires_explicit_canonical_data_root():
    with pytest.raises(ValueError, match="data.root must be provided"):
        build_datamodule(OmegaConf.create({"root": None}))


def test_pretrain_accepts_documented_data_root_override(tmp_path):
    module = build_datamodule(OmegaConf.create({"root": str(tmp_path)}))
    assert module.root == tmp_path
    assert module.manifest_path == tmp_path / "manifest.csv"
