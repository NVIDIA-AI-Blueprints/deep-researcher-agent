#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Harbor tasks for the deep-researcher-research skill."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from string import Template

REPO_ROOT = Path(__file__).resolve().parents[4]
VERIFIER = Path(__file__).resolve().parents[2] / "verifiers" / "deep_researcher_checks.py"
DEFAULT_SERVER_URL = "http://localhost:8000"
PREAMBLE = (
    "You are running inside a non-interactive evaluation harness. "
    "Use the installed Deep Researcher Agent Skills autonomously. Do not ask the user "
    "to run commands that you can run yourself."
)

PLATFORMS = {
    "local": {
        "short_name": "local",
        "description": "Runner-local or externally reachable Deep Researcher Agent server",
    }
}


def _copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _render(value: str, *, platform: str, mode: str, repo_root: Path) -> str:
    rendered = value
    server_url = os.environ.get("DEEP_RESEARCHER_SERVER_URL", DEFAULT_SERVER_URL)
    replacements = {
        "{{platform}}": platform,
        "{{mode}}": mode,
        "{{repo_root}}": str(repo_root),
        "{{deep_researcher_server_url}}": server_url,
    }
    for key, replacement in replacements.items():
        rendered = rendered.replace(key, replacement)
    return Template(rendered).safe_substitute(
        platform=platform,
        mode=mode,
        repo_root=str(repo_root),
        deep_researcher_server_url=server_url,
    )


def _test_script(spec_name: str, step: int) -> str:
    return (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'TEST_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        'LOCAL_SKILLS_DIR="$(cd "$TEST_DIR/.." && pwd)/skills"\n'
        'if [ -d "$LOCAL_SKILLS_DIR" ]; then\n'
        '  export DEEP_RESEARCHER_EVAL_SKILLS_DIR="${DEEP_RESEARCHER_EVAL_SKILLS_DIR:-$LOCAL_SKILLS_DIR}"\n'
        "fi\n"
        "# Exit status propagates from the verifier: 0 means it wrote reward.txt\n"
        "# (for any reward); non-zero means it crashed, which Harbor must see as\n"
        "# an error rather than a silent clean exit with no reward.txt.\n"
        f'python3 "$TEST_DIR/deep_researcher_checks.py" --spec "$TEST_DIR/{spec_name}" --step {step}\n'
    )


def _solution_script() -> str:
    return (
        "#!/bin/bash\nset -euo pipefail\npython3 /skills/deep-researcher-research/scripts/deep_researcher.py health\n"
    )


def _environment_compose() -> str:
    return (
        "services:\n"
        "  main:\n"
        "    extra_hosts:\n"
        '      - "host.docker.internal:host-gateway"\n'
        "    environment:\n"
        "      DEEP_RESEARCHER_SERVER_URL: ${DEEP_RESEARCHER_SERVER_URL}\n"
        "    volumes:\n"
        "      - type: bind\n"
        "        source: ${CONTEXT_DIR}/../skills\n"
        "        target: /skills\n"
        "        read_only: true\n"
    )


def _task_toml(skill: str, spec_stem: str, platform: str, mode: str, step: int, total_steps: int) -> str:
    platform_short = PLATFORMS[platform]["short_name"]
    step_suffix = f"-step-{step}" if total_steps > 1 else ""
    server_url = os.environ.get("DEEP_RESEARCHER_SERVER_URL", DEFAULT_SERVER_URL)
    return "\n".join(
        [
            "[task]",
            f'name = "nvidia-deep-researcher/{skill}-{spec_stem}-{platform_short}-{mode}{step_suffix}"',
            f'description = "{skill} {spec_stem} eval on {platform}/{mode} step {step} of {total_steps}"',
            f'keywords = ["deep-researcher", "{skill}", "{spec_stem}", "{platform}", "{mode}"]',
            "",
            "[environment]",
            'skills_dir = "/skills"',
            "",
            "[environment.env]",
            f'DEEP_RESEARCHER_SERVER_URL = "{server_url}"',
            "",
            "[verifier.env]",
            f'DEEP_RESEARCHER_SERVER_URL = "{server_url}"',
            "",
            "[metadata]",
            f'skill = "{skill}"',
            f'spec = "{spec_stem}"',
            f'platform = "{platform}"',
            f'mode = "{mode}"',
            f"step = {step}",
            f"total_steps = {total_steps}",
            "",
        ]
    )


def generate(spec_path: Path, skill_dir: Path, output_dir: Path, repo_root: Path) -> list[Path]:
    spec = json.loads(spec_path.read_text())
    skill = skill_dir.name
    spec_stem = spec_path.stem
    expects = spec.get("expects") or []
    if not expects:
        raise ValueError(f"{spec_path} has no expects entries")

    generated: list[Path] = []
    platforms = spec.get("resources", {}).get("platforms", {})
    for platform, platform_cfg in platforms.items():
        if platform not in PLATFORMS:
            raise ValueError(f"unsupported platform {platform!r}; supported: {sorted(PLATFORMS)}")
        for mode in platform_cfg.get("modes", []):
            mode_slug = mode.lower().replace("_", "-")
            base_dir = output_dir / spec_stem / f"{PLATFORMS[platform]['short_name']}-{mode_slug}"
            for idx, expect in enumerate(expects, start=1):
                step_dir = base_dir / f"step-{idx}" if len(expects) > 1 else base_dir
                tests_dir = step_dir / "tests"
                solution_dir = step_dir / "solution"
                skills_dir = step_dir / "skills"
                env_dir = step_dir / "environment"
                tests_dir.mkdir(parents=True, exist_ok=True)
                solution_dir.mkdir(parents=True, exist_ok=True)
                skills_dir.mkdir(parents=True, exist_ok=True)
                env_dir.mkdir(parents=True, exist_ok=True)

                instruction = "\n".join(
                    [
                        PREAMBLE,
                        "",
                        f"Deep Researcher Agent server URL: "
                        f"`{os.environ.get('DEEP_RESEARCHER_SERVER_URL', DEFAULT_SERVER_URL)}`.",
                        "Use the `/deep-researcher-research` skill for this task.",
                        "If the server is unavailable and `/deep-researcher-deploy` is installed, "
                        "use it to start or verify Deep Researcher Agent first.",
                        "",
                        f"## Query {idx} of {len(expects)}",
                        "",
                        _render(expect.get("query", ""), platform=platform, mode=mode, repo_root=repo_root),
                        "",
                        "## Environment Notes",
                        "",
                        _render(spec.get("env", ""), platform=platform, mode=mode, repo_root=repo_root),
                        "",
                    ]
                )
                (step_dir / "instruction.md").write_text(instruction + "\n")
                (step_dir / "task.toml").write_text(
                    _task_toml(skill, spec_stem, platform, mode_slug, idx, len(expects))
                )
                (env_dir / "Dockerfile").write_text("FROM python:3.12-slim\n")
                (env_dir / "docker-compose.yaml").write_text(_environment_compose())
                (tests_dir / spec_path.name).write_text(json.dumps(spec, indent=2) + "\n")
                shutil.copy2(VERIFIER, tests_dir / "deep_researcher_checks.py")
                test_sh = tests_dir / "test.sh"
                test_sh.write_text(_test_script(spec_path.name, idx))
                test_sh.chmod(0o755)
                solve_sh = solution_dir / "solve.sh"
                solve_sh.write_text(_solution_script())
                solve_sh.chmod(0o755)

                _copytree(skill_dir, skills_dir / skill)
                deploy_skill = repo_root / "skills" / "deep-researcher-deploy"
                if deploy_skill.exists():
                    _copytree(deploy_skill, skills_dir / "deep-researcher-deploy")
            generated.append(base_dir)
    return generated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skill-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    args = parser.parse_args()

    generated = generate(
        spec_path=Path(args.spec).resolve(),
        skill_dir=Path(args.skill_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        repo_root=Path(args.repo_root).resolve(),
    )
    print(json.dumps({"generated": [str(path) for path in generated]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
