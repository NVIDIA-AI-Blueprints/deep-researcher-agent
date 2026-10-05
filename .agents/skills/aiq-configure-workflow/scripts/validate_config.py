#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Static, read-only validator for an AI-Q workflow YAML config.

Checks cross-references and workflow shape before a config reaches a running
backend. Dependency-light: PyYAML + stdlib only.

Usage:
    uv run python validate_config.py path/to/config.yml
"""

from __future__ import annotations

import os
import re
import sys

try:
    import yaml
except ImportError:  # pragma: no cover
    print("ERROR: PyYAML is required. Install with: uv sync", file=sys.stderr)
    sys.exit(2)

CHAT_WORKFLOW_TYPE = "chat_deepresearcher_agent"
DATA_SCIENCE_WORKFLOW_TYPE = "data_science_workflow"
WORKFLOW_TYPES = {CHAT_WORKFLOW_TYPE, DATA_SCIENCE_WORKFLOW_TYPE}
FRONT_END_TYPE = "aiq_api"
LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}
TRACING_TYPES = {"langsmith", "otelcollector_redaction", "phoenix", "weave"}
EXPIRY_SECONDS_MIN = 600
EXPIRY_SECONDS_MAX = 604800

LLM_REF_FIELDS = (
    "llm",
    "orchestrator_llm",
    "planner_llm",
    "researcher_llm",
    "writer_llm",
    "source_router_llm",
    "summary_llm",
    "intent_llm",
    "summary_model",
)

REQUIRED_WORKFLOW_FUNCTIONS = {
    CHAT_WORKFLOW_TYPE: ("intent_classifier", "shallow_research_agent", "deep_research_agent"),
    DATA_SCIENCE_WORKFLOW_TYPE: ("data_science_agent",),
}

ENV_REF = re.compile(r"\$\{([A-Z0-9_]+)(?::-[^}]*)?\}")


def _load(path: str) -> tuple[dict, str]:
    """Load a YAML file and return its parsed mapping plus raw text."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        msg = f"Top-level YAML must be a mapping (got {type(data).__name__})."
        raise ValueError(msg)
    return data, text


def _iter_refs(node: object):
    """Yield (field, alias) for every LLM-alias reference found recursively."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in LLM_REF_FIELDS and isinstance(value, str):
                yield key, value
            else:
                yield from _iter_refs(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_refs(item)


def _general_block(data: dict) -> dict:
    """Return the top-level general block when it is a mapping."""
    general = data.get("general") or {}
    return general if isinstance(general, dict) else {}


def _validate_registry(registry: dict, declared_functions: set[str], errors: list[str], warnings: list[str]) -> None:
    """Validate data source registry shape and tool references."""
    sources = registry.get("sources")
    if not isinstance(sources, list):
        errors.append("data_source_registry.sources must be a list.")
        return

    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            errors.append(f"data_source_registry.sources[{index}] must be a mapping.")
            continue
        sid = source.get("id")
        label = sid or f"<source {index}>"
        if not sid:
            errors.append(f"data_source_registry source at index {index} is missing id.")
        if not source.get("name"):
            errors.append(f"source '{label}' is missing name.")
        tools = source.get("tools")
        if tools is None:
            errors.append(f"source '{label}' is missing tools:.")
            tools = []
        elif not isinstance(tools, list):
            errors.append(f"source '{label}' tools: must be a list.")
            tools = []
        for tool in tools:
            if tool not in declared_functions:
                errors.append(
                    f"source '{label}' lists tool '{tool}' in its tools:, but '{tool}' is not declared under "
                    "functions: or function_groups:"
                )
        for bool_field in ("default_enabled", "requires_auth"):
            if bool_field in source and not isinstance(source[bool_field], bool):
                errors.append(f"source '{label}' {bool_field}: must be true or false.")
        if source.get("requires_auth") is True:
            warnings.append(
                f"source '{label}' has requires_auth: true — confirm auth/token wiring "
                "(see config_web_frag_mcp_auth.yml)."
            )


def _validate_telemetry(general: dict, errors: list[str], warnings: list[str]) -> None:
    """Validate telemetry logging and tracing configuration shape."""
    telemetry = general.get("telemetry")
    if telemetry is None:
        return
    if not isinstance(telemetry, dict):
        errors.append("general.telemetry must be a mapping.")
        return

    logging = telemetry.get("logging")
    if logging is not None:
        if not isinstance(logging, dict):
            errors.append("general.telemetry.logging must be a mapping.")
        else:
            console = logging.get("console")
            if console is not None:
                if not isinstance(console, dict):
                    errors.append("general.telemetry.logging.console must be a mapping.")
                else:
                    if console.get("_type") != "console":
                        errors.append("general.telemetry.logging.console._type must be 'console'.")
                    level = console.get("level")
                    if level is not None and str(level).upper() not in LOG_LEVELS:
                        errors.append(
                            f"general.telemetry.logging.console.level must be one of {', '.join(sorted(LOG_LEVELS))}."
                        )

    tracing = telemetry.get("tracing")
    if tracing is not None:
        if not isinstance(tracing, dict):
            errors.append("general.telemetry.tracing must be a mapping.")
            return
        for name, exporter in tracing.items():
            if not isinstance(exporter, dict):
                errors.append(f"general.telemetry.tracing.{name} must be a mapping.")
                continue
            exporter_type = exporter.get("_type")
            if exporter_type not in TRACING_TYPES:
                errors.append(
                    f"general.telemetry.tracing.{name}._type must be one of {', '.join(sorted(TRACING_TYPES))}."
                )
            if exporter_type in {"otelcollector_redaction", "phoenix"} and not exporter.get("endpoint"):
                warnings.append(f"tracing exporter '{name}' usually needs an endpoint.")
            if exporter_type == "langsmith" and not os.environ.get("LANGCHAIN_API_KEY"):
                warnings.append("LangSmith tracing is configured; confirm LANGCHAIN_API_KEY is set.")
            if exporter_type == "weave" and not os.environ.get("WANDB_API_KEY"):
                warnings.append("Weave tracing is configured; confirm WANDB_API_KEY is set.")


def _validate_front_end(general: dict, errors: list[str]) -> None:
    """Validate AI-Q API front-end settings."""
    front_end = general.get("front_end")
    if front_end is None:
        return
    if not isinstance(front_end, dict):
        errors.append("`general.front_end` must be a mapping.")
        return
    if front_end.get("_type") != FRONT_END_TYPE:
        errors.append(f"general.front_end._type must be '{FRONT_END_TYPE}' for the AI-Q web API.")
    db_url = front_end.get("db_url")
    if db_url is not None and not isinstance(db_url, str):
        errors.append("general.front_end.db_url must be a string.")
    expiry = front_end.get("expiry_seconds")
    if expiry is not None:
        if not isinstance(expiry, int):
            errors.append("general.front_end.expiry_seconds must be an integer.")
        elif expiry < EXPIRY_SECONDS_MIN or expiry > EXPIRY_SECONDS_MAX:
            errors.append(
                f"general.front_end.expiry_seconds must be between {EXPIRY_SECONDS_MIN} and {EXPIRY_SECONDS_MAX}."
            )
    cors = front_end.get("cors")
    if cors is not None and not isinstance(cors, dict):
        errors.append("general.front_end.cors must be a mapping.")


def validate(path: str) -> int:
    """Validate one AI-Q workflow config and print a human-readable report."""
    errors: list[str] = []
    warnings: list[str] = []

    try:
        data, raw = _load(path)
    except FileNotFoundError:
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2
    except (yaml.YAMLError, ValueError) as exc:
        print(f"ERROR: could not parse YAML: {exc}", file=sys.stderr)
        return 2

    llms = data.get("llms") or {}
    defined_aliases = set(llms.keys()) if isinstance(llms, dict) else set()

    functions = data.get("functions") or {}
    if not isinstance(functions, dict):
        functions = {}
    declared_functions = set(functions.keys())

    function_groups = data.get("function_groups", {})
    if not isinstance(function_groups, dict):
        errors.append("`function_groups:` must be a mapping.")
        function_groups = {}

    for field, alias in _iter_refs(functions):
        if alias not in defined_aliases:
            defined = ", ".join(sorted(defined_aliases)) or "none"
            errors.append(
                f"llm alias '{alias}' referenced by a '{field}' field is not defined under llms: (defined: {defined})"
            )

    if not defined_aliases:
        warnings.append("no `llms:` block found — the config defines no LLM aliases.")

    registry = None
    for block in functions.values():
        if isinstance(block, dict) and block.get("_type") == "data_source_registry":
            registry = block
            break

    if registry is None:
        warnings.append("no data_source_registry function found (fine for minimal configs).")
    else:
        _validate_registry(registry, declared_functions | set(function_groups), errors, warnings)

    workflow = data.get("workflow")
    if workflow is None:
        errors.append("`workflow:` is required for an AI-Q workflow config.")
    elif not isinstance(workflow, dict):
        errors.append("`workflow:` must be a mapping.")
    else:
        wf_type = workflow.get("_type")
        if not isinstance(wf_type, str) or wf_type not in WORKFLOW_TYPES:
            allowed = ", ".join(sorted(WORKFLOW_TYPES))
            errors.append(f"workflow._type must be one of {allowed} (got {wf_type!r}).")
        if isinstance(wf_type, str):
            for function_name in REQUIRED_WORKFLOW_FUNCTIONS.get(wf_type, ()):
                if function_name not in declared_functions:
                    errors.append(f"workflow requires function '{function_name}' under functions: (missing).")
        hybrid_research_agent = workflow.get("hybrid_research_agent")
        if hybrid_research_agent is not None and not isinstance(hybrid_research_agent, str):
            errors.append("workflow.hybrid_research_agent must be a string function name when configured.")
        elif hybrid_research_agent is not None and hybrid_research_agent not in declared_functions:
            errors.append(
                "workflow.hybrid_research_agent references function "
                f"'{hybrid_research_agent}' but it is missing under functions: "
                "(configure a data_science_hybrid_adapter function)."
            )
        if (
            wf_type == CHAT_WORKFLOW_TYPE
            and workflow.get("enable_clarifier") is True
            and "clarifier_agent" not in declared_functions
        ):
            errors.append("workflow.enable_clarifier is true but 'clarifier_agent' is missing under functions:.")
        if (
            wf_type == CHAT_WORKFLOW_TYPE
            and workflow.get("use_async_deep_research") is True
            and not _general_block(data).get("front_end")
        ):
            warnings.append(
                "use_async_deep_research is true but general.front_end is missing "
                "(web/aiq_api mode expected for async jobs)."
            )

    general = _general_block(data)
    _validate_telemetry(general, errors, warnings)
    _validate_front_end(general, errors)

    env_vars = sorted(set(ENV_REF.findall(raw)))

    print(f"AI-Q config validation: {path}")
    print("-" * 60)
    for err in errors:
        print(f"ERROR: {err}")
    for warn in warnings:
        print(f"WARN:  {warn}")

    if env_vars:
        print("\nEnvironment variables referenced (set these in deploy/.env):")
        for var in env_vars:
            present = "set" if os.environ.get(var) else "NOT set in this shell"
            print(f"  - {var} ({present})")

    print("-" * 60)
    if errors:
        print(f"RESULT: {len(errors)} error(s), {len(warnings)} warning(s). Fix errors before deploying.")
        return 1
    print(f"RESULT: no errors, {len(warnings)} warning(s). Config is structurally valid.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and run validation."""
    args = argv if argv is not None else sys.argv
    if len(args) != 2:
        print("Usage: validate_config.py path/to/config.yml", file=sys.stderr)
        return 2
    return validate(args[1])


if __name__ == "__main__":
    raise SystemExit(main())
