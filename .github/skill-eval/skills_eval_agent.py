#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deep Researcher Agent skill-eval coordinator.

This script validates Agent Skill eval specs, generates Harbor-style datasets,
and can optionally run Harbor when the target environment is available.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("DEEP_RESEARCHER_SKILL_EVAL_OUTPUT_DIR", "/tmp/deep-researcher-skill-eval/datasets")
)
REQUIRED_SPEC_KEYS = ("skills", "resources", "env", "expects")

# Argv elements whose key portion ends in any of these suffixes are redacted in
# printed commands so credentials passed via `harbor --ae KEY=VALUE` (and similar
# inline KEY=VALUE forms) do not leak into stdout or captured workflow logs.
# The real values are still passed to subprocess.run unchanged.
_SECRET_KEY_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")


def _is_secret_key(name: str) -> bool:
    upper = name.upper()
    return any(upper.endswith(suffix) for suffix in _SECRET_KEY_SUFFIXES)


_SUFFIX_REVEAL_CHARS = 4
_SUFFIX_REVEAL_MIN_VALUE_LEN = 9


def _mask_value(value: str) -> str:
    """Mask a secret value, optionally revealing a short suffix for identification.

    Short values are fully masked (`***`); longer values keep the last few
    characters to help operators confirm which key was loaded without
    exposing meaningful entropy. The reveal length and minimum value
    length are constants so they can be tuned in one place.
    """
    if len(value) < _SUFFIX_REVEAL_MIN_VALUE_LEN:
        return "***"
    return f"***{value[-_SUFFIX_REVEAL_CHARS:]}"


def _mask_kv(arg: str) -> str:
    """Return KEY=***xxxx if KEY looks secret-shaped, else `arg` unchanged."""
    if "=" not in arg:
        return arg
    key, value = arg.split("=", 1)
    return f"{key}={_mask_value(value)}" if _is_secret_key(key) else arg


def _redact_for_log(cmd: list[str]) -> list[str]:
    """Return a copy of `cmd` with secret-shaped values masked for safe printing."""
    redacted: list[str] = []
    iterator = iter(cmd)
    for arg in iterator:
        if arg == "--ae":
            # The next argv element is the KEY=VALUE pair; mask its value if secret-shaped.
            redacted.append(arg)
            nxt = next(iterator, None)
            if nxt is not None:
                redacted.append(_mask_kv(nxt))
            continue
        # Inline KEY=VALUE arguments (e.g. exported through `KEY=VAL command` patterns
        # that leak into argv) are also masked defensively.
        redacted.append(_mask_kv(arg))
    return redacted


def _run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(_redact_for_log(cmd)), flush=True)
    return subprocess.run(cmd, cwd=cwd, text=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _git_changed_files(base: str) -> list[str]:
    result = _run(["git", "diff", "--name-only", f"{base}...HEAD"])
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _changed_skill_names(base: str | None) -> set[str]:
    if not base:
        return set()
    changed: set[str] = set()
    for path in _git_changed_files(base):
        parts = Path(path).parts
        if len(parts) >= 2 and parts[0] == "skills":
            changed.add(parts[1])
    return changed


def _discover_specs(*, all_specs: bool, base: str | None) -> list[Path]:
    changed = _changed_skill_names(base)
    specs = sorted(REPO_ROOT.glob("skills/*/evals/*-product.json"))
    if all_specs or not changed:
        return specs
    return [spec for spec in specs if spec.parts[-3] in changed]


def _validate_spec(spec_path: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text())
    missing = [key for key in REQUIRED_SPEC_KEYS if key not in spec]
    if missing:
        raise ValueError(f"{spec_path} missing required keys: {', '.join(missing)}")
    if not isinstance(spec["skills"], list) or not spec["skills"]:
        raise ValueError(f"{spec_path} must define non-empty skills list")
    platforms = spec.get("resources", {}).get("platforms")
    if not isinstance(platforms, dict) or not platforms:
        raise ValueError(f"{spec_path} must define resources.platforms")
    if not isinstance(spec["expects"], list) or not spec["expects"]:
        raise ValueError(f"{spec_path} must define non-empty expects list")
    for idx, expect in enumerate(spec["expects"], start=1):
        if "query" not in expect or not expect.get("checks"):
            raise ValueError(f"{spec_path} expects[{idx}] must define query and non-empty checks")
    return spec


def _adapter_for(skill: str) -> Path:
    return REPO_ROOT / ".github" / "skill-eval" / "adapters" / skill / "generate.py"


def _task_roots(dataset_root: Path) -> list[Path]:
    roots: list[Path] = []
    for task_toml in dataset_root.rglob("task.toml"):
        parent = task_toml.parent
        if parent.name.startswith("step-"):
            roots.append(parent.parent)
        else:
            roots.append(parent)
    return sorted(set(roots))


def _run_harbor(task_root: Path, out_dir: Path) -> tuple[int, Path | None]:
    agent = os.environ.get("DEEP_RESEARCHER_SKILL_EVAL_AGENT", "claude-code")
    model = os.environ.get("DEEP_RESEARCHER_SKILL_EVAL_MODEL") or os.environ.get("ANTHROPIC_MODEL")
    max_retries = os.environ.get("DEEP_RESEARCHER_SKILL_EVAL_MAX_RETRIES", "1")
    if agent != "oracle" and not model:
        print(
            "BLOCKED: DEEP_RESEARCHER_SKILL_EVAL_MODEL or ANTHROPIC_MODEL is required for Harbor execution",
            file=sys.stderr,
        )
        return 2, None

    cmd = [
        "uvx",
        "harbor",
        "run",
        "-p",
        str(task_root),
        "-a",
        agent,
        "--max-retries",
        max_retries,
        "-n",
        "1",
        "--yes",
        "-o",
        str(out_dir),
    ]
    if model:
        cmd.extend(["--model", model])
    if agent == "claude-code":
        cmd.extend(["--ae", "CLAUDE_CODE_DISABLE_THINKING=1"])
        for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"):
            value = os.environ.get(key)
            if value:
                cmd.extend(["--ae", f"{key}={value}"])
    if agent == "codex":
        for key in ("OPENAI_API_KEY", "CODEX_FORCE_AUTH_JSON", "CODEX_AUTH_JSON_PATH"):
            value = os.environ.get(key)
            if value:
                cmd.extend(["--ae", f"{key}={value}"])
    result = _run(cmd)
    print(result.stdout)
    # harbor writes <out_dir>/<timestamp>/result.json; out_dir is unique per run,
    # so read the result directly from disk instead of scraping stdout.
    candidates = sorted(out_dir.glob("*/result.json"))
    return result.returncode, (candidates[-1] if candidates else None)


def _gate(result_paths: list[Path | None], threshold: float) -> tuple[bool, list[str]]:
    """Aggregate a pass/fail verdict across every trial of every spec/skill in the run.

    Reads each Harbor job-level result.json. A trial that errored (exception)
    ALWAYS fails the run. A trial whose reward is below `threshold` fails the run
    unless `threshold <= 0` (report-only: every reward >= 0 passes). Returns
    (ok, summary_lines) where ok=False means the job should exit non-zero.
    """
    ok = True
    exceptions = 0
    lines: list[str] = []
    for path in result_paths:
        if path is None or not path.exists():
            ok = False
            lines.append("  MISSING result.json (harbor produced no parseable result) -> FAIL")
            continue
        stats = json.loads(path.read_text()).get("stats", {})
        exceptions += int(stats.get("n_errored_trials", 0) or 0)
        for name, ev in (stats.get("evals") or {}).items():
            buckets = ((ev or {}).get("reward_stats") or {}).get("reward") or {}
            for reward_str, trials in buckets.items():
                reward = float(reward_str)
                for trial in trials:
                    below = threshold > 0 and reward < threshold
                    if below:
                        ok = False
                    mark = f"BELOW {threshold}" if below else "ok"
                    lines.append(f"  {name} {trial}: reward {reward} [{mark}]")
    if exceptions > 0:
        ok = False
        lines.append(f"  exceptions: {exceptions} errored trial(s) -> FAIL")
    return ok, lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Generate every checked-in spec")
    parser.add_argument("--base", default=os.environ.get("PR_BASE") or os.environ.get("GITHUB_BASE_REF"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-harbor", action="store_true", help="Run Harbor after generating datasets")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = _discover_specs(all_specs=args.all, base=args.base)
    if not specs:
        print("BLOCKED: no Deep Researcher Agent skill eval specs found")
        return 0

    generated_roots: list[Path] = []
    for spec_path in specs:
        spec = _validate_spec(spec_path)
        skill = spec["skills"][0]
        skill_dir = REPO_ROOT / "skills" / skill
        adapter = _adapter_for(skill)
        if not skill_dir.exists():
            raise FileNotFoundError(f"skill directory not found: {skill_dir}")
        if not adapter.exists():
            raise FileNotFoundError(f"adapter not found: {adapter}")

        result = _run(
            [
                sys.executable,
                str(adapter),
                "--output-dir",
                str(output_dir / skill),
                "--skill-dir",
                str(skill_dir),
                "--spec",
                str(spec_path),
                "--repo-root",
                str(REPO_ROOT),
            ]
        )
        print(result.stdout)
        if result.returncode != 0:
            return result.returncode
        generated_roots.extend(_task_roots(output_dir / skill / spec_path.stem))

    summary = {
        "specs": [str(path.relative_to(REPO_ROOT)) for path in specs],
        "datasets": [str(path) for path in sorted(set(generated_roots))],
    }
    print(json.dumps(summary, indent=2))

    if args.run_harbor:
        results_root = Path(
            os.environ.get("DEEP_RESEARCHER_SKILL_EVAL_RESULTS_DIR", "/tmp/deep-researcher-skill-eval/results")
        )
        # Isolate this job's results so the gate can never read a previous job's
        # output on the long-lived self-hosted runner (the results root persists
        # between jobs). Keyed by GitHub run id + attempt in CI; random locally.
        run_id = os.environ.get("GITHUB_RUN_ID", "")
        attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
        job_key = "-".join(p for p in (run_id, attempt) if p) or f"local-{uuid.uuid4().hex[:12]}"
        job_results_root = results_root / job_key
        result_paths: list[Path | None] = []
        for index, task_root in enumerate(sorted(set(generated_roots))):
            # Unique output dir per run so the result.json location is deterministic
            # (harbor writes <out_dir>/<timestamp>/result.json).
            out_dir = job_results_root / f"run-{index:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            rc, results_path = _run_harbor(task_root, out_dir)
            if rc != 0:
                return rc  # infrastructure failure (Harbor itself errored)
            result_paths.append(results_path)

        # Post-Harbor verdict. Harbor exits 0 even at low reward, so gate here so a
        # green check actually means the eval passed (catches PR regressions).
        threshold = float(os.environ.get("DEEP_RESEARCHER_SKILL_EVAL_REWARD_THRESHOLD", "1.0"))
        ok, summary = _gate(result_paths, threshold)
        gate_desc = "report-only" if threshold <= 0 else f"threshold {threshold}"
        print(f"=== Skill-eval verdict ({gate_desc}) ===")
        for line in summary:
            print(line)
        if not ok:
            print(
                "EVAL FAILED: a trial errored or scored below threshold "
                "(set DEEP_RESEARCHER_SKILL_EVAL_REWARD_THRESHOLD=0 for report-only)"
            )
            return 1
        print("EVAL PASSED")

    print(f"DONE: generated {len(set(generated_roots))} Deep Researcher Agent skill-eval dataset(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
