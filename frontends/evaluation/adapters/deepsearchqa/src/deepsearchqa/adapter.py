"""Generate Harbor tasks for DeepSearchQA."""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import urllib.request
from pathlib import Path
from string import Template

DEFAULT_DATA_URL = "https://huggingface.co/datasets/google/deepsearchqa/resolve/main/DSQA-full.csv"
DATASET_ID = "google/deepsearchqa"
DATASET_FILE = "DSQA-full.csv"
ANSWER_TYPES = {"Single Answer", "Set Answer"}


class _TaskTomlTemplate(Template):
    delimiter = "@"


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _keyword(value: str) -> str:
    normalized = value.lower().replace("&", "and")
    return "-".join(part for part in re.split(r"[^a-z0-9]+", normalized) if part)


def _task_id_for_index(index: int) -> str:
    return f"deepsearchqa-{index:04d}"


class DeepSearchQAAdapter:
    def __init__(
        self,
        output_dir: Path,
        overwrite: bool = False,
        csv_path: Path | None = None,
        data_url: str = DEFAULT_DATA_URL,
        org: str = "aiq",
        user: str | None = None,
    ):
        """Initialize the DeepSearchQA task generator."""
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.csv_path = csv_path
        self.data_url = data_url
        self.org = org
        self.user = user
        self.template_dir = Path(__file__).parent / "task-template"
        self.task_toml_template = _TaskTomlTemplate((self.template_dir / "task.toml").read_text(encoding="utf-8"))

    def run(self) -> None:
        """Iterate over DeepSearchQA rows and generate Harbor task directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows = self._load_rows()
        seen_source_ids: set[str] = set()

        for index, row in enumerate(rows, start=1):
            task_id = _task_id_for_index(index)
            source_id = row["source_id"]
            if source_id in seen_source_ids:
                raise ValueError(f"Duplicate DeepSearchQA source ID: {source_id}")
            seen_source_ids.add(source_id)

            self._write_task(index=index, task_id=task_id, row=row)
        print(f"Generated {len(rows)} DeepSearchQA task(s) in {self.output_dir}")

    def _load_rows(self) -> list[dict[str, str]]:
        if self.csv_path is not None:
            csv_text = self.csv_path.read_text(encoding="utf-8-sig")
        else:
            with urllib.request.urlopen(self.data_url, timeout=60) as response:
                csv_text = response.read().decode("utf-8-sig")

        reader = csv.DictReader(io.StringIO(csv_text))
        required_columns = {"problem", "problem_category", "answer", "answer_type"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"DeepSearchQA CSV is missing columns: {sorted(missing_columns)}")

        return [self._normalize_row(dict(row), line_number) for line_number, row in enumerate(reader, start=2)]

    def _normalize_row(self, row: dict[str, str], line_number: int) -> dict[str, str]:
        normalized = {key: (value or "").strip() for key, value in row.items()}
        for field in ("problem", "problem_category", "answer", "answer_type"):
            if not normalized.get(field):
                raise ValueError(f"DeepSearchQA CSV line {line_number} has empty {field}.")

        answer_type = normalized["answer_type"]
        if answer_type not in ANSWER_TYPES:
            raise ValueError(f"DeepSearchQA CSV line {line_number} has unsupported answer_type: {answer_type!r}.")

        source_index = line_number - 1
        source_id = normalized.get("example_id") or normalized.get("id") or str(source_index)
        normalized["source_id"] = str(source_id).strip()
        if not normalized["source_id"]:
            raise ValueError(f"DeepSearchQA CSV line {line_number} has empty source ID.")
        normalized["source_index"] = str(source_index)
        return normalized

    def _write_task(self, index: int, task_id: str, row: dict[str, str]) -> None:
        task_dir = self.output_dir / task_id
        if task_dir.exists():
            if not self.overwrite:
                raise FileExistsError(f"{task_dir} already exists. Re-run with --overwrite to replace it.")
            shutil.rmtree(task_dir)

        shutil.copytree(
            self.template_dir,
            task_dir,
            ignore=shutil.ignore_patterns("solution"),
        )

        problem = row["problem"].strip()
        category = row["problem_category"].strip()
        answer = row["answer"].strip()
        answer_type = row["answer_type"].strip()
        source_id = row["source_id"].strip()

        (task_dir / "instruction.md").write_text(
            self._render_instruction(problem),
            encoding="utf-8",
        )
        (task_dir / "task.toml").write_text(
            self._render_task_toml(
                index=index,
                task_id=task_id,
                source_id=source_id,
                category=category,
            ),
            encoding="utf-8",
        )
        (task_dir / "tests" / "metadata.json").write_text(
            json.dumps(
                {
                    "example_id": task_id,
                    "source_id": source_id,
                    "source_index": index,
                    "source_dataset": DATASET_ID,
                    "source_file": DATASET_FILE,
                    "source_url": self.data_url,
                    "problem": problem,
                    "problem_category": category,
                    "answer": answer,
                    "answer_type": answer_type,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _render_instruction(self, problem: str) -> str:
        return problem

    def _render_task_toml(
        self,
        index: int,
        task_id: str,
        category: str,
        source_id: str | None = None,
    ) -> str:
        source_id = source_id or str(index)
        user_line = "" if self.user is None else f"user = {_toml_string(self.user)}"
        return self.task_toml_template.substitute(
            TASK_NAME=_toml_string(f"{self.org}/{task_id}"),
            DESCRIPTION=_toml_string(f"DeepSearchQA example {index}: {category}"),
            CATEGORY_KEYWORD=_toml_string(_keyword(category) or "uncategorized"),
            CATEGORY=_toml_string(category),
            SOURCE_DATASET=_toml_string(DATASET_ID),
            SOURCE_ID=_toml_string(source_id),
            SOURCE_INDEX=str(index),
            VERIFIER_USER=user_line,
            AGENT_USER=user_line,
        )
