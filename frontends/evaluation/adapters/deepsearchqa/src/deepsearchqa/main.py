"""CLI entry point for generating DeepSearchQA Harbor tasks."""

import argparse
from pathlib import Path

from .adapter import DEFAULT_DATA_URL
from .adapter import DeepSearchQAAdapter

# Default output dir: <repo>/datasets/<adapter_id>
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[4] / "datasets" / "deepsearchqa"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write generated tasks",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tasks",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional local path to DSQA-full.csv. Defaults to downloading from Hugging Face.",
    )
    parser.add_argument(
        "--data-url",
        default=DEFAULT_DATA_URL,
        help="URL to DSQA-full.csv when --csv-path is not provided.",
    )
    parser.add_argument(
        "--org",
        default="aiq",
        help="Harbor organization namespace for generated task names.",
    )
    parser.add_argument(
        "--user",
        default=None,
        help="Optional UID, username, or UID:GID to set for agent and verifier execution.",
    )
    args = parser.parse_args()

    adapter = DeepSearchQAAdapter(
        args.output_dir,
        overwrite=args.overwrite,
        csv_path=args.csv_path,
        data_url=args.data_url,
        org=args.org,
        user=args.user,
    )

    adapter.run()


if __name__ == "__main__":
    main()
