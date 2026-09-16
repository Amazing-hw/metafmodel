"""Audit a canonical Meta sEMG corpus and write JSON/CSV reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semg_fm.data.audit import audit_canonical_corpus, write_audit_reports


def _non_negative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-stem", required=True, type=Path)
    parser.add_argument("--h5-samples", type=_non_negative, default=0)
    args = parser.parse_args(argv)

    report, rows = audit_canonical_corpus(
        args.root, args.manifest, h5_samples=args.h5_samples
    )
    json_path, csv_path = write_audit_reports(args.output_stem, report, rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")
    failures = (
        report["missing_files"],
        report["invalid_tasks"],
        report["invalid_splits"],
        report["h5_schema_errors"],
    )
    return 2 if any(failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
