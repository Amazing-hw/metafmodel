"""Convert GNI wrist-angle recordings to canonical HDF5."""

from .gni_conversion import cli_main, convert_gni_file


def convert_file(source_path, output_root, *, official_split="unknown", overwrite=False):
    return convert_gni_file(
        source_path, output_root, expected_source_task="wrist",
        dataset_id="gni_wrist", task_id="wrist_angle",
        official_split=official_split, overwrite=overwrite,
    )


def main(argv=None):
    return cli_main(argv, description=__doc__, expected_source_task="wrist",
                    dataset_id="gni_wrist", task_id="wrist_angle")


if __name__ == "__main__":
    raise SystemExit(main())
