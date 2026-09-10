"""Hydra entry point for task-specific fine-tuning of canonical sEMG models."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from pretrain import build_datamodule, manifest_digest
from semg_fm.config import validate_devices, validate_seed
from semg_fm.lightning.downstream_module import DownstreamModule
from semg_fm.registry import model_from_config


def _module_from_config(cfg: DictConfig) -> DownstreamModule:
    """Construct a task head while keeping evaluation-only settings explicit."""
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    task_config = OmegaConf.to_container(cfg.task, resolve=True)
    assert isinstance(model_config, dict) and isinstance(task_config, dict)
    task_config.pop("event_tolerance_s", None)  # Used by held-out event F1, not the training head.
    module = DownstreamModule(
        backbone=model_from_config(model_config),
        task_id=str(task_config.pop("name")),
        output_dim=task_config.pop("output_dim", None),
        target_name=task_config.pop("target_name", None),
        class_names=task_config.pop("class_names", ()),
        vocabulary=task_config.pop("vocabulary", ()),
        learning_rate=float(cfg.optimizer.learning_rate),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    unsupported = sorted(key for key, value in task_config.items() if value is not None)
    if unsupported:
        raise ValueError(f"unsupported finetune task settings: {unsupported}")
    return module


@hydra.main(version_base="1.3", config_path="configs", config_name="finetune")
def main(cfg: DictConfig) -> None:
    """Build a backbone/head pair, optionally initialize it, then fine-tune."""
    validate_devices(cfg.trainer.devices)
    validate_seed(cfg.seed)
    seed_everything(cfg.seed, workers=True)
    datamodule = build_datamodule(cfg.data)
    module = _module_from_config(cfg)
    if cfg.pretrained_checkpoint:
        module.load_backbone_checkpoint(str(cfg.pretrained_checkpoint))
    output = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
    (output / "manifest.json").write_text(
        json.dumps(
            {"path": str(datamodule.manifest_path.resolve()), "sha256": manifest_digest(datamodule.manifest_path)},
            indent=2,
        ),
        encoding="utf-8",
    )
    trainer_options = OmegaConf.to_container(cfg.trainer, resolve=True)
    assert isinstance(trainer_options, dict)
    Trainer(**trainer_options).fit(module, datamodule=datamodule, ckpt_path=cfg.resume_from)


if __name__ == "__main__":
    main()
