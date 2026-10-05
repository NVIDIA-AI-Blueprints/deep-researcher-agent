# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the maintainer workflow-config validator."""

import importlib.util
from pathlib import Path

import pytest
import yaml

_VALIDATOR_PATH = (
    Path(__file__).parents[2] / ".agents" / "skills" / "aiq-configure-workflow" / "scripts" / "validate_config.py"
)
_SPEC = importlib.util.spec_from_file_location("aiq_validate_config", _VALIDATOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_VALIDATOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_VALIDATOR)


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_direct_data_science_workflow_is_valid(tmp_path, capsys):
    path = _write_config(
        tmp_path,
        {
            "llms": {"model": {"_type": "nim", "model_name": "test"}},
            "functions": {"data_science_agent": {"_type": "data_science_agent", "llm": "model"}},
            "workflow": {"_type": "data_science_workflow"},
        },
    )

    assert _VALIDATOR.validate(str(path)) == 0
    assert "RESULT: no errors" in capsys.readouterr().out


def test_direct_data_science_workflow_requires_agent_function(tmp_path, capsys):
    path = _write_config(
        tmp_path,
        {
            "llms": {"model": {"_type": "nim", "model_name": "test"}},
            "functions": {},
            "workflow": {"_type": "data_science_workflow"},
        },
    )

    assert _VALIDATOR.validate(str(path)) == 1
    assert "workflow requires function 'data_science_agent'" in capsys.readouterr().out


def test_chat_workflow_requires_declared_hybrid_adapter(tmp_path, capsys):
    path = _write_config(
        tmp_path,
        {
            "functions": {
                "intent_classifier": {"_type": "intent_classifier"},
                "shallow_research_agent": {"_type": "shallow_research_agent"},
                "deep_research_agent": {"_type": "deep_research_agent"},
            },
            "workflow": {
                "_type": "chat_deepresearcher_agent",
                "hybrid_research_agent": "misspelled_hybrid_adapter",
            },
        },
    )

    assert _VALIDATOR.validate(str(path)) == 1
    output = capsys.readouterr().out
    assert "workflow.hybrid_research_agent references function 'misspelled_hybrid_adapter'" in output
    assert "configure a data_science_hybrid_adapter function" in output


@pytest.mark.parametrize(
    "hybrid_research_agent",
    [
        ["data_science_hybrid_adapter"],
        {"name": "data_science_hybrid_adapter"},
    ],
)
def test_hybrid_research_agent_must_be_a_string(tmp_path, capsys, hybrid_research_agent):
    path = _write_config(
        tmp_path,
        {
            "functions": {},
            "workflow": {
                "_type": "chat_deepresearcher_agent",
                "hybrid_research_agent": hybrid_research_agent,
            },
        },
    )

    assert _VALIDATOR.validate(str(path)) == 1
    assert "workflow.hybrid_research_agent must be a string function name" in capsys.readouterr().out


@pytest.mark.parametrize(
    "workflow_type",
    [
        ["data_science_workflow"],
        {"name": "data_science_workflow"},
    ],
)
def test_workflow_type_must_be_a_string(tmp_path, capsys, workflow_type):
    path = _write_config(
        tmp_path,
        {
            "functions": {},
            "workflow": {"_type": workflow_type},
        },
    )

    assert _VALIDATOR.validate(str(path)) == 1
    assert "workflow._type must be one of" in capsys.readouterr().out
