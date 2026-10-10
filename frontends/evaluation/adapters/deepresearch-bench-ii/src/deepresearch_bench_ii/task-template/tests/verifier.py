#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

OFFICIAL_DIR = Path(__file__).resolve().parent / "official"
if str(OFFICIAL_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_DIR))

import aggregate_scores  # type: ignore  # noqa: E402
import run_evaluation  # type: ignore  # noqa: E402
from gemini_client import GeminiClient  # type: ignore  # noqa: E402

GRADER_SETUP_ERROR_MARKERS = (
    "Missing configuration: GEMINI_API_KEY",
    "Missing configuration: GEMINI_MODEL",
)


def finite_score(value: Any) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def official_config() -> dict[str, Any]:
    config = run_evaluation.get_default_config()
    return {
        "chunk_size": int(os.environ.get("CHUNK_SIZE", config["chunk_size"])),
        "max_retries": int(os.environ.get("MAX_RETRIES", config["max_retries"])),
        "max_paper_chars": int(os.environ.get("MAX_PAPER_CHARS", config["max_paper_chars"])),
    }


def official_content(metadata: dict[str, Any]) -> dict[str, Any]:
    task = metadata.get("task")
    rubric = metadata.get("rubric")
    blocked = metadata.get("blocked", {})
    if not isinstance(task, str) or not task.strip():
        raise ValueError("metadata.json must contain a non-empty task string.")
    if not isinstance(rubric, dict):
        raise ValueError("metadata.json must contain a rubric object.")
    if not isinstance(blocked, dict):
        raise ValueError("metadata.json blocked field must be an object.")
    return {"task": task, "rubric": rubric, "blocked": blocked}


def evaluate(metadata: dict[str, Any], report_path: Path) -> dict[str, Any]:
    config = official_config()
    run_evaluation.client = GeminiClient(verbose=False)
    source_id = metadata.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("metadata.json must contain a non-empty source_id.")

    _, result_payload, grader_tokens = run_evaluation.process_one_with_chunking(
        0,
        str(report_path),
        official_content(metadata),
        config["chunk_size"],
        config["max_paper_chars"],
        config["max_retries"],
    )
    if not isinstance(result_payload, dict) or result_payload.get("error"):
        error = result_payload.get("error") if isinstance(result_payload, dict) else "invalid official result"
        return {
            "reward": 0.0,
            "info_recall": 0.0,
            "analysis": 0.0,
            "presentation": 0.0,
            "blocked_rate": 0.0,
            "rubric_item_count": 0.0,
            "grader_valid": 0.0,
            "source_id": source_id,
            "error": error,
            "official_config": config,
            "official_result": result_payload,
        }

    dimensions = aggregate_scores.compute_dimension_averages(result_payload)
    rubric_item_count = 0
    scores = result_payload.get("scores", {})
    if isinstance(scores, dict):
        for dimension in ("info_recall", "analysis", "presentation"):
            dimension_scores = scores.get(dimension)
            if isinstance(dimension_scores, dict):
                rubric_item_count += len(dimension_scores)

    return {
        "reward": finite_score(dimensions.get("total")),
        "info_recall": finite_score(dimensions.get("inforecall")),
        "analysis": finite_score(dimensions.get("analysis")),
        "presentation": finite_score(dimensions.get("presentation")),
        "blocked_rate": finite_score(dimensions.get("blocked_rate")),
        "rubric_item_count": float(rubric_item_count),
        "grader_valid": 1.0,
        "source_id": source_id,
        "grader_tokens": float(grader_tokens or 0),
        "official_config": config,
        "official_result": result_payload,
    }


def write_outputs(result: dict[str, Any], verifier_dir: Path) -> None:
    verifier_dir.mkdir(parents=True, exist_ok=True)
    reward_payload = {
        "reward": finite_score(result.get("reward")),
        "info_recall": finite_score(result.get("info_recall")),
        "analysis": finite_score(result.get("analysis")),
        "presentation": finite_score(result.get("presentation")),
        "blocked_rate": finite_score(result.get("blocked_rate")),
        "rubric_item_count": finite_score(result.get("rubric_item_count")),
        "grader_valid": finite_score(result.get("grader_valid")),
    }
    if "grader_tokens" in result:
        reward_payload["grader_tokens"] = finite_score(result.get("grader_tokens"))
    (verifier_dir / "reward.json").write_text(
        json.dumps(reward_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (verifier_dir / "grading.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def is_grader_setup_error(error: str) -> bool:
    return any(marker in error for marker in GRADER_SETUP_ERROR_MARKERS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    verifier_dir = Path("/logs/verifier")
    try:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        if not args.report.exists() or not args.report.read_text(encoding="utf-8").strip():
            raise ValueError("Agent report file is missing or empty.")
        result = evaluate(metadata, args.report)
    except Exception as exc:
        result = {
            "reward": 0.0,
            "info_recall": 0.0,
            "analysis": 0.0,
            "presentation": 0.0,
            "blocked_rate": 0.0,
            "rubric_item_count": 0.0,
            "grader_valid": 0.0,
            "error": str(exc),
        }
        if is_grader_setup_error(result["error"]):
            verifier_dir.mkdir(parents=True, exist_ok=True)
            (verifier_dir / "grading.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(result["error"], file=sys.stderr)
            raise SystemExit(2) from exc
    write_outputs(result, verifier_dir)


if __name__ == "__main__":
    main()
