"""Convert emg2pose source HDF5 records into canonical per-wrist HDF5 files."""

import argparse
import csv
from pathlib import Path

from .canonical import write_record
from .manifest import append_manifest_row
from .source_readers import build_manifest_row, output_relative_path, read_emg2pose


def convert_file(
    source_path: str | Path,
    output_root: str | Path,
    *,
    official_split: str = "unknown",
    overwrite: bool = False,
) -> dict[str, object]:
    record = read_emg2pose(source_path, official_split)
    relative_path = output_relative_path(record)
    write_record(Path(output_root) / relative_path, record, overwrite=overwrite)
    return build_manifest_row(record, relative_path)


def _split_lookup(csv_path: str | Path | None) -> dict[str, str]:
    if csv_path is None:
        return {}
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or "split" not in rows.fieldnames:
            raise ValueError("Split CSV must include a 'split' column")
        key_column = next((name for name in ("path", "source_file", "file") if name in rows.fieldnames), None)
        if key_column is None:
            raise ValueError("Split CSV must include one of: path, source_file, file")
        return {str(row[key_column]).replace("\\", "/"): str(row["split"]) for row in rows}


def run_conversion(
    input_root: str | Path,
    output_root: str | Path,
    *,
    split: str | None = None,
    splits_csv: str | Path | None = None,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    lookup = _split_lookup(splits_csv)
    rows: list[dict[str, object]] = []
    for source_path in sorted(input_root.rglob("*.h5")):
        relative_source = source_path.relative_to(input_root).as_posix()
        official_split = lookup.get(relative_source, lookup.get(source_path.name, "unknown"))
        if split is not None and official_split != split:
            continue
        row = convert_file(source_path, output_root, official_split=official_split, overwrite=overwrite)
        append_manifest_row(output_root / "manifest.csv", row)
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", help="Convert only this official split")
    parser.add_argument("--splits-csv", help="CSV mapping source path or file to official split")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    rows = run_conversion(
        args.input_root,
        args.output_root,
        split=args.split,
        splits_csv=args.splits_csv,
        overwrite=args.overwrite,
    )
    print(f"Converted {len(rows)} emg2pose records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
