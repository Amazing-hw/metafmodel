"""The public documents must name every public entry point."""

from __future__ import annotations

from pathlib import Path


def test_public_documents_include_all_entry_points():
    root = Path(__file__).parents[1]
    text = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in [
            "README.md",
            "PRETRAINING.md",
            "DOWNSTREAM_EVALUATION.md",
            "BASELINES.md",
            "SERVER_A100.md",
        ]
    )
    for command in ["pretrain.py", "finetune.py", "evaluate.py", "run_baseline.py", "make_splits.py"]:
        assert command in text


def test_training_stack_documents_exist():
    root = Path(__file__).parents[1]
    for name in [
        "README.md",
        "PRETRAINING.md",
        "DOWNSTREAM_EVALUATION.md",
        "BASELINES.md",
        "EXPERIMENT_PROTOCOL.md",
        "H5_DATA_CONTRACT.md",
        "SERVER_A100.md",
    ]:
        assert (root / name).is_file(), f"missing {name}"


def test_pretraining_docs_limit_deterministic_resume_to_epoch_boundaries():
    root = Path(__file__).parents[1]
    text = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in ["PRETRAINING.md", "SERVER_A100.md"]
    )
    assert "epoch-boundary" in text
    assert "mid-epoch resume replays deterministically" not in text


def test_pretraining_commands_use_reproducible_absolute_path_variables():
    root = Path(__file__).parents[1]
    text = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in ["PRETRAINING.md", "SERVER_A100.md"]
    )

    for expected in [
        "epoch-boundary",
        "CANONICAL_META=/absolute/path/to/canonical_meta",
        "RUN_DIR=/absolute/path/to/outputs/run",
        'data.root="${CANONICAL_META}"',
        'data.manifest_path="${CANONICAL_META}/splits.csv"',
        'resume_from="${RUN_DIR}/checkpoints/last.ckpt"',
        "+data.task_id=pose objective.kinematic_dim=20",
    ]:
        assert expected in text
    for prohibited in [
        "data.root=...",
        "data.manifest_path=...",
        "outputs/<run>",
        "supplied no independent gradient",
        "...",
    ]:
        assert prohibited not in text
