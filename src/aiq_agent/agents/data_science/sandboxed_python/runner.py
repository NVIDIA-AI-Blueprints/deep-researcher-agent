# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-shot scientific Python runner with request-scoped structured-data helpers."""

from __future__ import annotations

import ast
import contextlib
import io
import itertools
import json
import math
import re
import statistics
import sys
import traceback
from collections import Counter
from collections import defaultdict
from collections.abc import Callable
from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels.api as sm
from scipy import stats


class _CappedStringIO(io.StringIO):
    """Capture text without retaining more than one configured output budget."""

    def __init__(self, max_chars: int) -> None:
        super().__init__()
        self.max_chars = max_chars
        self.captured_chars = 0
        self.truncated = False

    def write(self, value: str) -> int:
        """Retain only the remaining prefix while reporting the full write length."""

        text = str(value)
        remaining = max(0, self.max_chars - self.captured_chars)
        if len(text) > remaining:
            self.truncated = True
        if remaining:
            super().write(text[:remaining])
            self.captured_chars += min(len(text), remaining)
        return len(text)


def _combined_output(stdout: _CappedStringIO, stderr: _CappedStringIO, max_output_chars: int) -> str:
    printed = stdout.getvalue()
    warnings = stderr.getvalue()
    combined = printed + (("\n" if printed and warnings else "") + warnings if warnings else "")
    truncated = stdout.truncated or stderr.truncated or len(combined) > max_output_chars
    combined = combined[:max_output_chars]
    if truncated:
        combined += "\n... output truncated ..."
    return combined


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("invalid structured-data evidence manifest")
    return payload


def _analysis_helpers(manifest_path: Path) -> dict[str, Callable[..., Any]]:
    """Build trusted helpers that reload the authoritative receipt manifest."""

    def list_analysis_results() -> pd.DataFrame:
        entries = _read_manifest(manifest_path).get("results") or []
        columns = [
            "ref",
            "provider",
            "tool_name",
            "question",
            "database_name",
            "request_id",
            "row_count",
            "columns",
            "truncated",
        ]
        return pd.DataFrame(entries).reindex(columns=columns)

    def resolve(reference: str | None = None) -> dict[str, Any]:
        entries = _read_manifest(manifest_path).get("results") or []
        if not entries:
            raise LookupError("No successful structured-data results are registered for this request.")
        if reference in {None, "latest"}:
            return entries[-1]
        for entry in entries:
            if entry.get("ref") == reference:
                return entry
        available = ", ".join(str(entry.get("ref")) for entry in entries)
        raise KeyError(f"Unknown structured-data result reference {reference!r}. Available references: {available}")

    def analysis_result(reference: str | None = None) -> dict[str, Any]:
        entry = resolve(reference)
        return json.loads(Path(entry["path"]).read_text(encoding="utf-8"))

    def analysis_rows(reference: str | None = None) -> pd.DataFrame:
        return pd.DataFrame(analysis_result(reference).get("rows") or [])

    def analysis_sql(reference: str | None = None) -> str:
        return str(analysis_result(reference).get("sql") or "")

    def analysis_latest() -> pd.DataFrame:
        return analysis_rows("latest")

    return {
        "analysis_latest": analysis_latest,
        "analysis_result": analysis_result,
        "analysis_rows": analysis_rows,
        "analysis_sql": analysis_sql,
        "list_analysis_results": list_analysis_results,
    }


def _compile_script(code: str) -> tuple[Any | None, Any | None]:
    tree = ast.parse(code, mode="exec")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        statements = ast.Module(body=tree.body[:-1], type_ignores=[])
        expression = ast.Expression(body=tree.body[-1].value)
        return compile(statements, "<aiq-python>", "exec"), compile(expression, "<aiq-python>", "eval")
    return compile(tree, "<aiq-python>", "exec"), None


def _display(value: Any, max_output_chars: int) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.DataFrame):
        rendered = value.to_string(max_rows=60, max_cols=50, line_width=160)
    elif isinstance(value, pd.Series):
        rendered = value.to_string(max_rows=100)
    elif isinstance(value, np.ndarray):
        rendered = np.array2string(value, threshold=500, edgeitems=20)
    else:
        rendered = repr(value)
    if len(rendered) <= max_output_chars:
        return rendered
    return rendered[:max_output_chars] + "\n... output truncated ..."


def _visible_variables(namespace: dict[str, Any]) -> list[str]:
    hidden = {
        "Counter",
        "Path",
        "date",
        "datetime",
        "defaultdict",
        "analysis_latest",
        "analysis_result",
        "analysis_rows",
        "analysis_sql",
        "itertools",
        "json",
        "list_analysis_results",
        "math",
        "np",
        "pd",
        "re",
        "scipy",
        "sklearn",
        "sm",
        "statistics",
        "stats",
        "timedelta",
        "timezone",
    }
    return sorted(name for name in namespace if not name.startswith("_") and name not in hidden)


def _new_namespace(manifest_path: Path) -> dict[str, Any]:
    return {
        "__name__": "__aiq_analysis__",
        "Counter": Counter,
        "Path": Path,
        "date": date,
        "datetime": datetime,
        "defaultdict": defaultdict,
        "itertools": itertools,
        "json": json,
        "math": math,
        "np": np,
        "pd": pd,
        "re": re,
        "scipy": scipy,
        "sklearn": sklearn,
        "sm": sm,
        "statistics": statistics,
        "stats": stats,
        "timedelta": timedelta,
        "timezone": timezone,
        **_analysis_helpers(manifest_path),
    }


def execute(code: str, manifest_path: Path, max_output_chars: int) -> dict[str, Any]:
    """Execute one self-contained script in a fresh scientific namespace."""

    namespace = _new_namespace(manifest_path)
    stdout = _CappedStringIO(max_output_chars)
    stderr = _CappedStringIO(max_output_chars)
    try:
        statements, expression = _compile_script(code)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if statements is not None:
                exec(statements, namespace)
            value = eval(expression, namespace) if expression is not None else None
        return {
            "status": "ok",
            "output": _combined_output(stdout, stderr, max_output_chars),
            "result": _display(value, max_output_chars),
            "result_type": type(value).__name__ if value is not None else None,
            "variables": _visible_variables(namespace),
        }
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        return {
            "status": "error",
            "error": type(exc).__name__,
            "detail": str(exc)[:2_000],
            "traceback": "".join(traceback.format_exception(exc))[-4_000:],
            "output": _combined_output(stdout, stderr, max_output_chars),
            "variables": _visible_variables(namespace),
        }


def main() -> None:
    if len(sys.argv) != 5:
        raise ValueError("Python runner requires <manifest> <request> <max-output-chars> <response>")
    manifest_path = Path(sys.argv[1])
    request = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    if not isinstance(request, dict) or not isinstance(request.get("code"), str):
        raise ValueError("invalid Python execution request")
    response = execute(request["code"], manifest_path, int(sys.argv[3]))
    Path(sys.argv[4]).write_text(
        json.dumps(response, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
