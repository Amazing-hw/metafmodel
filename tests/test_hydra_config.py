from hydra import compose, initialize_config_dir
from pathlib import Path
import pytest

from semg_fm.config import validate_seed

from semg_fm.config import validate_devices


CONFIG_DIR = str((Path(__file__).parents[1] / "configs").resolve())


def test_pretrain_config_composes_with_four_devices():
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="pretrain", overrides=["trainer.devices=4"])
    assert cfg.trainer.devices == 4
    assert cfg.model.name == "ring_criss_cross"
    assert cfg.data.channels == 16


def test_cpu_smoke_profile_disables_distributed_training():
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="pretrain", overrides=["trainer=cpu_smoke"])
    assert cfg.trainer.accelerator == "cpu"
    assert cfg.trainer.devices == 1
    assert cfg.trainer.strategy == "auto"


def test_pretrain_default_declares_a_null_data_root_that_accepts_an_override():
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        default = compose(config_name="pretrain")
        configured = compose(config_name="pretrain", overrides=["data.root=tests/fixtures/canonical_meta"])
    assert default.data.root is None
    assert configured.data.root == "tests/fixtures/canonical_meta"


@pytest.mark.parametrize("seed", [True, -1, 1.2, "4"])
def test_seed_validation_rejects_invalid_values(seed):
    with pytest.raises(ValueError, match="seed"):
        validate_seed(seed)


def test_seed_validation_accepts_nonnegative_integer():
    validate_seed(0)
    validate_seed(17)


def test_validate_devices_accepts_auto_and_supported_integer_range():
    validate_devices("auto")
    for devices in range(1, 9):
        validate_devices(devices)


def test_validate_devices_rejects_invalid_values():
    for devices in (0, 9, "0", "gpu"):
        try:
            validate_devices(devices)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {devices!r}")
