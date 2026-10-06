#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the pytest-owned live OpenShell acceptance suite.

The assertions, fixtures, resource ownership, and verified teardown live in
``tests/deep_researcher_agent/agents/deep_researcher/sandbox/test_openshell_live.py``.
See ``docs/source/deployment/openshell.md`` for the operator contract.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIVE_TEST = Path("tests/deep_researcher_agent/agents/deep_researcher/sandbox/test_openshell_live.py")


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gateway",
        default=os.getenv("DEEP_RESEARCHER_OPENSHELL_GATEWAY_NAME"),
        help="Registered OpenShell gateway name; default uses the active gateway",
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("DEEP_RESEARCHER_OPENSHELL_WORKSPACE") or "default",
        help="OpenShell workspace that scopes live-test sandbox operations",
    )
    parser.add_argument(
        "--policy",
        default=os.getenv(
            "DEEP_RESEARCHER_OPENSHELL_POLICY_FILE",
            "configs/openshell/generated/deep-researcher-openshell-policy.yaml",
        ),
        help="Policy submitted and attested by the live suite",
    )
    parser.add_argument(
        "--image",
        default=os.getenv("DEEP_RESEARCHER_OPENSHELL_IMAGE", "deep-researcher-openshell-demo:latest"),
        help="Prebuilt OpenShell sandbox image",
    )
    parser.add_argument(
        "--expected-gateway-version",
        default=os.getenv("DEEP_RESEARCHER_OPENSHELL_EXPECTED_GATEWAY_VERSION"),
        help="Optional exact gateway version; default requires it to match the installed SDK",
    )
    parser.add_argument(
        "--allow-best-effort-landlock",
        action="store_true",
        help="Permit a local-demo best_effort policy; never use this for production acceptance",
    )
    return parser.parse_args(argv)


def _environment(args: argparse.Namespace, source: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if source is None else source)
    env["DEEP_RESEARCHER_OPENSHELL_LIVE_TESTS"] = "1"
    if args.gateway:
        env["DEEP_RESEARCHER_OPENSHELL_GATEWAY_NAME"] = args.gateway
    else:
        env.pop("DEEP_RESEARCHER_OPENSHELL_GATEWAY_NAME", None)
    env["DEEP_RESEARCHER_OPENSHELL_WORKSPACE"] = args.workspace
    env["DEEP_RESEARCHER_OPENSHELL_POLICY_FILE"] = args.policy
    env["DEEP_RESEARCHER_OPENSHELL_IMAGE"] = args.image
    if args.expected_gateway_version:
        env["DEEP_RESEARCHER_OPENSHELL_EXPECTED_GATEWAY_VERSION"] = args.expected_gateway_version
    else:
        env.pop("DEEP_RESEARCHER_OPENSHELL_EXPECTED_GATEWAY_VERSION", None)
    if args.allow_best_effort_landlock:
        env["DEEP_RESEARCHER_OPENSHELL_LIVE_ALLOW_BEST_EFFORT"] = "1"
    return env


def _command() -> list[str]:
    return [sys.executable, "-m", "pytest", "-m", "integration", "-vv", str(_LIVE_TEST)]


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell, operator-selected interpreter
        _command(),
        cwd=_REPO_ROOT,
        env=_environment(args),
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
