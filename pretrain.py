"""Hydra entry point for CIRCE-Myo pretraining."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import hydra
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from semg_fm.config import validate_devices
from semg_fm.data.datamodule import CanonicalDataModule
from semg_fm.lightning import PretrainModule
from semg_fm.registry import model_from_config


def manifest_digest(path: str | Path) -> str:
    """Hash the exact manifest consumed by a training run."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_datamodule(data: DictConfig) -> CanonicalDataModule:
    root = Path(str(data.root))
    manifest = Path(str(data.get("manifest_path", root / "manifest.csv")))
    return CanonicalDataModule(manifest_path=manifest, root=root, window_ms=float(data.get("window_ms", 200)), stride_ms=float(data.get("stride_ms", 100)), batch_size=int(data.get("batch_size", 8)), num_workers=int(data.get("num_workers", 0)), sampler_seed=int(data.get("sampler_seed", 0)))


@hydra.main(version_base="1.3", config_path="configs", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    validate_devices(cfg.trainer.devices)
    seed_everything(int(cfg.get("seed", 0)), workers=True)
    datamodule = build_datamodule(cfg.data)
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_config, dict)
    module = PretrainModule(student=model_from_config(model_config), model_name=str(cfg.model.name), model_config=model_config)
    output = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
    (output / "manifest.json").write_text(json.dumps({"path": str(datamodule.manifest_path.resolve()), "sha256": manifest_digest(datamodule.manifest_path)}, indent=2), encoding="utf-8")
    trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
    assert isinstance(trainer_config, dict)
    Trainer(**trainer_config).fit(module, datamodule=datamodule, ckpt_path=cfg.get("resume_from"))


if __name__ == "__main__":
    main()
