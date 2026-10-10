"""Generate Harbor tasks for DeepResearch Bench II."""

from __future__ import annotations

import json
import re
import shutil
import urllib.request
from pathlib import Path
from string import Template
from typing import Any

UPSTREAM_REVISION = "5451c9884baa61e172f968a58ad461edde4e411a"  # pragma: allowlist secret
DEFAULT_DATA_URL = (
    f"https://raw.githubusercontent.com/imlrz/DeepResearch-Bench-II/{UPSTREAM_REVISION}/tasks_and_rubrics.jsonl"
)
DATASET_ID = "DeepResearch-Bench-II"
EXPECTED_TASK_COUNT = 132
DIMENSIONS = ("info_recall", "analysis", "presentation")


class _TaskTomlTemplate(Template):
    delimiter = "@"


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _keyword(value: str) -> str:
    normalized = value.lower().replace("&", "and")
    return "-".join(part for part in re.split(r"[^a-z0-9]+", normalized) if part)


def _task_id_for_index(index: int) -> str:
    return f"deepresearch-bench-ii-{index:03d}"


class DeepResearchBenchIIAdapter:
    def __init__(
        self,
        output_dir: Path,
        overwrite: bool = False,
        tasks_jsonl: Path | None = None,
        data_url: str = DEFAULT_DATA_URL,
        org: str = "aiq",
        user: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.tasks_jsonl = tasks_jsonl
        self.data_url = data_url
        self.org = org
        self.user = user
        self.template_dir = Path(__file__).parent / "task-template"
        self.task_toml_template = _TaskTomlTemplate((self.template_dir / "task.toml").read_text(encoding="utf-8"))

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows = self._load_rows()
        for row in rows:
            index = row["source_index"]
            self._write_task(index=index, task_id=_task_id_for_index(index), row=row)
        print(f"Generated {len(rows)} DeepResearch Bench II task(s) in {self.output_dir}")

    def _load_rows(self) -> list[dict[str, Any]]:
        if self.tasks_jsonl is not None:
            source_text = self.tasks_jsonl.read_text(encoding="utf-8")
        else:
            with urllib.request.urlopen(self.data_url, timeout=60) as response:
                source_text = response.read().decode("utf-8")

        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(source_text.splitlines(), start=1):
            if not line.strip():
                continue
            raw_row = json.loads(line)
            if not isinstance(raw_row, dict):
                raise ValueError(f"DeepResearch Bench II line {line_number} is not an object.")
            rows.append(self._normalize_row(raw_row, line_number))

        rows.sort(key=lambda row: row["source_index"])
        source_ids = [row["source_id"] for row in rows]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("DeepResearch Bench II source contains duplicate IDs.")

        source_indices = [row["source_index"] for row in rows]
        if source_indices != list(range(1, len(rows) + 1)):
            raise ValueError("DeepResearch Bench II source indices must be consecutive and start at 1.")
        if self.tasks_jsonl is None and len(rows) != EXPECTED_TASK_COUNT:
            raise ValueError(
                f"Pinned DeepResearch Bench II source must contain {EXPECTED_TASK_COUNT} tasks, found {len(rows)}."
            )
        return rows

    def _normalize_row(self, raw_row: dict[str, Any], line_number: int) -> dict[str, Any]:
        source_id = raw_row.get("id")
        source_index = raw_row.get("idx")
        language = raw_row.get("language")
        theme = raw_row.get("theme")
        prompt = raw_row.get("prompt")
        content = raw_row.get("content")

        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"DeepResearch Bench II line {line_number} has an empty id.")
        if not isinstance(source_index, int) or source_index <= 0:
            raise ValueError(f"DeepResearch Bench II line {line_number} has malformed idx: {source_index!r}.")
        for field_name, value in {"language": language, "theme": theme, "prompt": prompt}.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"DeepResearch Bench II line {line_number} has empty {field_name}.")
        if not isinstance(content, dict):
            raise ValueError(f"DeepResearch Bench II line {line_number} has malformed content.")

        task = content.get("task")
        rubric = content.get("rubric")
        blocked = content.get("blocked", {})
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"DeepResearch Bench II line {line_number} has empty content.task.")
        if not isinstance(rubric, dict):
            raise ValueError(f"DeepResearch Bench II line {line_number} has malformed content.rubric.")
        if not isinstance(blocked, dict):
            raise ValueError(f"DeepResearch Bench II line {line_number} has malformed content.blocked.")

        normalized_rubric: dict[str, list[str]] = {}
        for dimension in DIMENSIONS:
            items = rubric.get(dimension)
            if not isinstance(items, list) or not items:
                raise ValueError(f"DeepResearch Bench II line {line_number} must contain non-empty rubric.{dimension}.")
            if not all(isinstance(item, str) and item.strip() for item in items):
                raise ValueError(
                    f"DeepResearch Bench II line {line_number} rubric.{dimension} contains an invalid item."
                )
            normalized_rubric[dimension] = [item.strip() for item in items]

        description = raw_row.get("description")
        if not isinstance(description, str) or not description.strip():
            description = next((line.strip() for line in task.splitlines() if line.strip()), "Deep research task")[:120]

        return {
            "source_id": source_id.strip(),
            "source_index": source_index,
            "language": language.strip(),
            "theme": theme.strip(),
            "description": description.strip(),
            "license": str(raw_row.get("license") or "CC-BY-4.0 / CC-BY-4.0-NC"),
            "prompt": prompt.strip(),
            "task": task.strip(),
            "rubric": normalized_rubric,
            "blocked": blocked,
        }

    def _write_task(self, index: int, task_id: str, row: dict[str, Any]) -> None:
        task_dir = self.output_dir / task_id
        if task_dir.exists():
            if not self.overwrite:
                raise FileExistsError(f"{task_dir} already exists. Re-run with --overwrite to replace it.")
            shutil.rmtree(task_dir)

        shutil.copytree(self.template_dir, task_dir)
        (task_dir / "instruction.md").write_text(row["prompt"], encoding="utf-8")
        (task_dir / "task.toml").write_text(
            self._render_task_toml(index=index, task_id=task_id, row=row),
            encoding="utf-8",
        )
        (task_dir / "tests" / "metadata.json").write_text(
            json.dumps(
                {
                    "example_id": task_id,
                    "source_id": row["source_id"],
                    "source_index": index,
                    "source_dataset": DATASET_ID,
                    "source_repo": "https://github.com/imlrz/DeepResearch-Bench-II",
                    "source_revision": UPSTREAM_REVISION,
                    "source_url": self.data_url,
                    "language": row["language"],
                    "theme": row["theme"],
                    "description": row["description"],
                    "task": row["task"],
                    "prompt": row["prompt"],
                    "rubric": row["rubric"],
                    "blocked": row["blocked"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _render_task_toml(self, index: int, task_id: str, row: dict[str, Any]) -> str:
        user_line = "" if self.user is None else f"user = {_toml_string(self.user)}"
        description = f"DeepResearch Bench II example {row['source_id']}: {row['description']}"
        return self.task_toml_template.substitute(
            TASK_NAME=_toml_string(f"{self.org}/{task_id}"),
            DESCRIPTION=_toml_string(description),
            LANGUAGE_KEYWORD=_toml_string(f"language-{_keyword(row['language'])}"),
            THEME_KEYWORD=_toml_string(_keyword(row["theme"]) or "uncategorized"),
            SOURCE_DATASET=_toml_string(DATASET_ID),
            SOURCE_ID=_toml_string(row["source_id"]),
            SOURCE_INDEX=str(index),
            LANGUAGE=_toml_string(row["language"]),
            THEME=_toml_string(row["theme"]),
            LICENSE=_toml_string(row["license"]),
            VERIFIER_USER=user_line,
            AGENT_USER=user_line,
        )
