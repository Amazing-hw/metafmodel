"""Deterministic, group-aware train/validation/test assignment."""

from __future__ import annotations

import hashlib

import pandas as pd


RATIO_TOLERANCE = 1e-9
_REQUIRED_COLUMNS = {"dataset_id", "subject_id", "pair_id"}
_SPLITS = ("train", "val", "test")


def _validate_ratios(ratios):
    try:
        values = tuple(float(value) for value in ratios)
    except (TypeError, ValueError) as exc:
        raise ValueError("ratios must contain exactly three positive values summing to 1.0") from exc
    if len(values) != 3 or any(value <= 0 for value in values) or abs(sum(values) - 1.0) > RATIO_TOLERANCE:
        raise ValueError("ratios must contain exactly three positive values summing to 1.0")
    return values


def _is_present(value) -> bool:
    if value is None:
        return False
    if value is pd.NA:
        return False
    missing = pd.isna(value)
    try:
        if bool(missing):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def _group_key(row):
    if _is_present(row["pair_id"]):
        return f"pair:{row['pair_id']}"
    return f"subject:{row['dataset_id']}\x1f{row['subject_id']}"


def assign_group_splits(frame: pd.DataFrame, seed: int, ratios=(0.8, 0.1, 0.1)) -> pd.DataFrame:
    """Return a copy with deterministic group-level ``split`` assignments."""
    missing = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    values = _validate_ratios(ratios)
    result = frame.copy()
    keys = result.apply(_group_key, axis=1)
    assignments = {}
    cumulative = (values[0], values[0] + values[1], 1.0)
    for key in pd.unique(keys):
        digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
        point = int.from_bytes(digest[:8], "big") / 2**64
        assignments[key] = _SPLITS[next(index for index, bound in enumerate(cumulative) if point < bound)]
    result["split"] = keys.map(assignments).astype("string")
    return result


def assert_no_leakage(frame: pd.DataFrame) -> None:
    """Raise ``ValueError`` when a group is assigned to multiple splits."""
    required = _REQUIRED_COLUMNS | {"split"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    for key_name, mask in (("pair_id", frame["pair_id"].map(_is_present)),):
        grouped = frame.loc[mask].groupby(key_name, dropna=False)["split"].nunique(dropna=False)
        if (grouped > 1).any():
            raise ValueError(f"leakage detected for {key_name}")
    unpaired = frame.loc[~frame["pair_id"].map(_is_present)].copy()
    grouped = unpaired.groupby(["dataset_id", "subject_id"], dropna=False)["split"].nunique(dropna=False)
    if (grouped > 1).any():
        raise ValueError("leakage detected for dataset/subject group")
