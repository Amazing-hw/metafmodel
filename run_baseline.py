"""Run declared comparison baselines through the same data and Trainer stack."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from finetune import _module_from_config as downstream_module_from_config
from pretrain import build_datamodule, manifest_digest
from semg_fm.config import validate_devices, validate_seed
from semg_fm.lightning.baseline_module import BaselinePretrainModule


def build_baseline_module(cfg: DictConfig):
    """Dispatch only the baseline stage/model combinations defined by protocol."""
    if cfg.stage == "finetune":
        return downstream_module_from_config(cfg)
    if cfg.stage == "pretrain":
        model_config = OmegaConf.to_container(cfg.model, resolve=True)
        assert isinstance(model_config, dict)
        name = str(model_config.pop("name"))
        return BaselinePretrainModule(
            name,
            learning_rate=float(cfg.optimizer.learning_rate),
            weight_decay=float(cfg.optimizer.weight_decay),
            **model_config,
        )
    raise ValueError("stage must be 'pretrain' or 'finetune'")


@hydra.main(version_base="1.3", config_path="configs", config_name="baseline")
def main(cfg: DictConfig) -> None:
    """Train one baseline stage and snapshot its data/config provenance."""
    validate_devices(cfg.trainer.devices)
    validate_seed(cfg.seed)
    seed_everything(cfg.seed, workers=True)
    datamodule = build_datamodule(cfg.data)
    module = build_baseline_module(cfg)
    if cfg.stage == "finetune" and cfg.pretrained_checkpoint:
        module.load_backbone_checkpoint(str(cfg.pretrained_checkpoint))
    output = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
    (output / "manifest.json").write_text(
        json.dumps({"path": str(datamodule.manifest_path.resolve()), "sha256": manifest_digest(datamodule.manifest_path)}, indent=2),
        encoding="utf-8",
    )
    trainer_options = OmegaConf.to_container(cfg.trainer, resolve=True)
    assert isinstance(trainer_options, dict)
    Trainer(**trainer_options).fit(module, datamodule=datamodule, ckpt_path=cfg.resume_from)


if __name__ == "__main__":
    main()
