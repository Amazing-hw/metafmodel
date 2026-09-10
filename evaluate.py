"""Hydra entry point for held-out canonical sEMG task evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from pretrain import build_datamodule, manifest_digest
from semg_fm.config import validate_devices, validate_seed
from semg_fm.lightning.downstream_module import DownstreamModule
from semg_fm.registry import model_from_config


def _module_from_config(cfg: DictConfig) -> DownstreamModule:
    """Reconstruct a downstream module before loading its checkpoint state."""
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    task_config = OmegaConf.to_container(cfg.task, resolve=True)
    assert isinstance(model_config, dict) and isinstance(task_config, dict)
    return DownstreamModule(
        backbone=model_from_config(model_config),
        task_id=str(task_config["name"]),
        output_dim=task_config.get("output_dim"),
        target_name=task_config.get("target_name"),
        class_names=task_config.get("class_names", ()),
        vocabulary=task_config.get("vocabulary", ()),
    )


@hydra.main(version_base="1.3", config_path="configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    """Run test split evaluation and save configuration/metric provenance."""
    validate_devices(cfg.trainer.devices)
    validate_seed(cfg.seed)
    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path must be provided")
    seed_everything(cfg.seed, workers=True)
    datamodule = build_datamodule(cfg.data)
    module = _module_from_config(cfg)
    checkpoint = torch.load(str(cfg.checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a state_dict mapping")
    module.load_state_dict(state, strict=True)
    output = Path(cfg.output_dir) if cfg.output_dir else Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
    trainer_options = OmegaConf.to_container(cfg.trainer, resolve=True)
    assert isinstance(trainer_options, dict)
    results = Trainer(**trainer_options).test(module, datamodule=datamodule)
    metrics = results[0] if results else {}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(
            {"path": str(datamodule.manifest_path.resolve()), "sha256": manifest_digest(datamodule.manifest_path)},
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
