from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_standalone_runtime_inventory():
    required = [
        "make_splits.py",
        "pretrain.py",
        "run_baseline.py",
        "finetune.py",
        "evaluate.py",
        "semg_h5",
        "semg_fm",
        "configs",
        "requirements.txt",
        "requirements-train.txt",
        "pytest.ini",
    ]
    assert [name for name in required if not (ROOT / name).exists()] == []


def test_original_converter_entry_points_are_preserved():
    stems = [
        "emg2pose",
        "emg2qwerty",
        "gni_discrete_gestures",
        "gni_handwriting",
        "gni_wrist",
    ]
    assert [stem for stem in stems if not (ROOT / f"convert_{stem}.py").is_file()] == []


def test_repository_text_has_no_retired_project_root():
    retired = "C:" + "\\Users\\User\\Documents\\F_model"
    suffixes = {".py", ".yaml", ".yml", ".md", ".ps1", ".txt", ".ini"}
    offenders = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if retired.lower() in text.lower():
                offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []
