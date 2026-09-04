"""Convert bilateral emg2qwerty HDF5 sessions into linked canonical wrist records."""

import argparse
from pathlib import Path

from .canonical import write_record
from .convert_emg2pose import _split_lookup
from .manifest import append_manifest_row
from .source_readers import build_manifest_row, output_relative_path, read_emg2qwerty


def convert_file(
    source_path: str | Path,
    output_root: str | Path,
    *,
    official_split: str = "unknown",
    overwrite: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in read_emg2qwerty(source_path, official_split):
        relative_path = output_relative_path(record)
        write_record(Path(output_root) / relative_path, record, overwrite=overwrite)
        rows.append(build_manifest_row(record, relative_path))
    return rows


def run_conversion(
    input_root: str | Path,
    output_root: str | Path,
    *,
    split: str | None = None,
    splits_csv: str | Path | None = None,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    input_root, output_root = Path(input_root), Path(output_root)
    lookup = _split_lookup(splits_csv)
    rows: list[dict[str, object]] = []
    for source_path in sorted(input_root.rglob("*.h5")):
        relative_source = source_path.relative_to(input_root).as_posix()
        official_split = lookup.get(relative_source, lookup.get(source_path.name, "unknown"))
        if split is not None and official_split != split:
            continue
        for row in convert_file(source_path, output_root, official_split=official_split, overwrite=overwrite):
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
    print(f"Converted {len(rows)} canonical wrist records from emg2qwerty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
