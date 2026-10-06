#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic verifier for Deep Researcher Agent skill eval tasks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

TRAJECTORY_CANDIDATES = (
    "/logs/agent/trajectory.jsonl",
    "/logs/agent/trajectory.json",
    "/logs/agent/claude-code.txt",
    "/logs/agent/agent.log",
)


def locate_trajectory() -> Path | None:
    override = os.environ.get("DEEP_RESEARCHER_EVAL_TRAJECTORY")
    if override:
        path = Path(override)
        return path if path.exists() else None
    for candidate in TRAJECTORY_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def read_trajectory() -> str:
    path = locate_trajectory()
    if path is None:
        return ""
    return path.read_text(errors="replace")


def run_shell(command: str) -> tuple[bool, str]:
    skills_dir = os.environ.get("DEEP_RESEARCHER_EVAL_SKILLS_DIR")
    if skills_dir:
        command = command.replace("/skills/", f"{skills_dir.rstrip('/')}/")
    result = subprocess.run(
        command,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(os.environ.get("DEEP_RESEARCHER_EVAL_CHECK_TIMEOUT", "120")),
        check=False,
    )
    output = (result.stdout or "").strip()
    return result.returncode == 0, output[-4000:]


def run_json_command(command: str) -> tuple[bool, str]:
    passed, output = run_shell(command)
    if not passed:
        return False, output
    if not output:
        return False, "stdout was empty; expected JSON output"
    try:
        json.loads(output)
    except json.JSONDecodeError as exc:
        return False, f"stdout was not valid JSON: {exc}: {output[:1000]}"
    return True, output[:1000]


def evaluate_check(check: str, trajectory: str) -> dict[str, Any]:
    if check.startswith("shell:"):
        command = check.removeprefix("shell:").strip()
        passed, output = run_shell(command)
        return {"check": check, "pass": passed, "route": "shell", "evidence": output}

    if check.startswith("json_command:"):
        command = check.removeprefix("json_command:").strip()
        passed, output = run_json_command(command)
        return {"check": check, "pass": passed, "route": "json_command", "evidence": output}

    if check.startswith("trajectory_contains:"):
        needle = check.removeprefix("trajectory_contains:").strip()
        passed = needle in trajectory
        return {
            "check": check,
            "pass": passed,
            "route": "trajectory_contains",
            "evidence": needle if passed else "substring not found",
        }

    if check.startswith("trajectory_not_contains:"):
        needle = check.removeprefix("trajectory_not_contains:").strip()
        passed = needle not in trajectory
        return {
            "check": check,
            "pass": passed,
            "route": "trajectory_not_contains",
            "evidence": "absent" if passed else needle,
        }

    return {
        "check": check,
        "pass": False,
        "route": "unsupported",
        "evidence": "check must start with shell:, json_command:, trajectory_contains:, or trajectory_not_contains:",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    expects = spec.get("expects") or []
    if args.step < 1 or args.step > len(expects):
        raise SystemExit(f"step {args.step} out of range for {len(expects)} expects")

    checks = expects[args.step - 1].get("checks") or []
    trajectory = read_trajectory()
    results = [evaluate_check(check, trajectory) for check in checks]
    passed = sum(1 for result in results if result["pass"])
    total = len(results)
    # A step with no checks verifies nothing; score it 0.0 rather than a silent
    # perfect 1.0 (validation also rejects empty `checks`, this is defense-in-depth).
    reward = 0.0 if total == 0 else passed / total

    log_dir = Path(os.environ.get("DEEP_RESEARCHER_EVAL_VERIFIER_LOG_DIR", "/logs/verifier"))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "reward.txt").write_text(f"{reward:.6f}\n")
    (log_dir / "deep-researcher-checks.json").write_text(json.dumps(results, indent=2) + "\n")

    for result in results:
        status = "PASS" if result["pass"] else "FAIL"
        print(f"{status}: {result['check']}")
        if result["evidence"]:
            print(f"  {result['evidence']}")
    print(f"=== Results: {passed} passed, {total - passed} failed (of {total}) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
