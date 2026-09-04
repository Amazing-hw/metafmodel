"""Shared execution layer for the three Generic Neuromotor Interface converters."""

import argparse
from pathlib import Path

from .canonical import write_record
from .convert_emg2pose import _split_lookup
from .manifest import append_manifest_row
from .source_readers import build_manifest_row, output_relative_path, read_gni


def convert_gni_file(
    source_path: str | Path,
    output_root: str | Path,
    *,
    expected_source_task: str,
    dataset_id: str,
    task_id: str,
    official_split: str = "unknown",
    overwrite: bool = False,
) -> dict[str, object]:
    record = read_gni(
        source_path,
        expected_source_task=expected_source_task,
        dataset_id=dataset_id,
        task_id=task_id,
        official_split=official_split,
    )
    relative_path = output_relative_path(record)
    write_record(Path(output_root) / relative_path, record, overwrite=overwrite)
    return build_manifest_row(record, relative_path)


def run_gni_conversion(
    input_root: str | Path,
    output_root: str | Path,
    *,
    expected_source_task: str,
    dataset_id: str,
    task_id: str,
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
        row = convert_gni_file(
            source_path,
            output_root,
            expected_source_task=expected_source_task,
            dataset_id=dataset_id,
            task_id=task_id,
            official_split=official_split,
            overwrite=overwrite,
        )
        append_manifest_row(output_root / "manifest.csv", row)
        rows.append(row)
    return rows


def cli_main(
    argv: list[str] | None,
    *,
    description: str,
    expected_source_task: str,
    dataset_id: str,
    task_id: str,
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", help="Convert only this official split")
    parser.add_argument("--splits-csv", help="CSV mapping source path or file to official split")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    rows = run_gni_conversion(
        args.input_root,
        args.output_root,
        expected_source_task=expected_source_task,
        dataset_id=dataset_id,
        task_id=task_id,
        split=args.split,
        splits_csv=args.splits_csv,
        overwrite=args.overwrite,
    )
    print(f"Converted {len(rows)} {dataset_id} records.")
    return 0
