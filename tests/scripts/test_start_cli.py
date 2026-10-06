# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the local CLI launcher."""

import os
import shutil
import subprocess
from pathlib import Path


def test_start_cli_forwards_verbose_flag(tmp_path: Path) -> None:
    """The shell launcher forwards verbose mode to the Python CLI."""
    scripts_dir = tmp_path / "scripts"
    bin_dir = tmp_path / ".venv" / "bin"
    scripts_dir.mkdir()
    bin_dir.mkdir(parents=True)
    launcher = scripts_dir / "start_cli.sh"
    shutil.copy2(Path("scripts/start_cli.sh"), launcher)
    (bin_dir / "activate").write_text("", encoding="utf-8")
    arguments_path = tmp_path / "arguments.txt"
    fake_cli = bin_dir / "deep-researcher-research"
    fake_cli.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$DEEP_RESEARCHER_TEST_ARGUMENTS_PATH"\n', encoding="utf-8")
    fake_cli.chmod(0o755)
    env = {**os.environ, "DEEP_RESEARCHER_TEST_ARGUMENTS_PATH": str(arguments_path)}

    subprocess.run([launcher, "--verbose"], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)

    assert arguments_path.read_text(encoding="utf-8").splitlines() == [
        "--config_file",
        "configs/config_cli_default.yml",
        "--verbose",
    ]
