"""CLI entry point for generating DeepResearch Bench II Harbor tasks."""

import argparse
from pathlib import Path

from .adapter import DEFAULT_DATA_URL
from .adapter import DeepResearchBenchIIAdapter

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[4] / "datasets" / "deepresearch-bench-ii"


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
        "--tasks-jsonl",
        type=Path,
        default=None,
        help="Optional local tasks_and_rubrics.jsonl file. Defaults to the pinned upstream source.",
    )
    parser.add_argument(
        "--data-url",
        default=DEFAULT_DATA_URL,
        help="URL to tasks_and_rubrics.jsonl when --tasks-jsonl is not provided.",
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

    DeepResearchBenchIIAdapter(
        args.output_dir,
        overwrite=args.overwrite,
        tasks_jsonl=args.tasks_jsonl,
        data_url=args.data_url,
        org=args.org,
        user=args.user,
    ).run()


if __name__ == "__main__":
    main()
