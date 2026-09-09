"""CLI for adding deterministic leakage-safe splits to a manifest."""

import argparse

from .manifest import load_manifest
from .splits import assert_no_leakage, assign_group_splits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args(argv)

    frame = load_manifest(args.manifest, args.root)
    official = frame["official_split"].fillna("").astype(str).str.strip()
    assigned = frame.copy()
    assigned["split"] = official
    empty = official == ""
    if empty.any():
        assigned.loc[empty, "split"] = assign_group_splits(
            frame.loc[empty], seed=args.seed, ratios=(0.8, 0.1, 0.1)
        )["split"].to_numpy()
    assert_no_leakage(assigned)
    assigned.to_csv(args.output, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
