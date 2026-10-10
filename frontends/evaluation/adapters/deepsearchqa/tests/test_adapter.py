import json
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest
from deepsearchqa.adapter import DeepSearchQAAdapter


def _render_task_toml(tmp_path: Path, user: str | None = None) -> str:
    adapter = DeepSearchQAAdapter(output_dir=tmp_path, user=user)
    return adapter._render_task_toml(
        index=7,
        task_id="deepsearchqa-0007",
        category='Arts & "Culture"',
        source_id="source-7",
    )


def test_task_toml_template_renders_valid_toml(tmp_path: Path) -> None:
    rendered = _render_task_toml(tmp_path)
    parsed = tomllib.loads(rendered)

    assert "@{" not in rendered
    assert parsed["task"]["name"] == "aiq/deepsearchqa-0007"
    assert parsed["task"]["description"] == 'DeepSearchQA example 7: Arts & "Culture"'
    assert parsed["task"]["keywords"][-1] == "arts-and-culture"
    assert len(parsed["task"]["authors"]) == 12
    assert parsed["metadata"] == {
        "category": 'Arts & "Culture"',
        "source_dataset": "google/deepsearchqa",
        "source_id": "source-7",
        "source_index": 7,
    }
    assert "user" not in parsed["verifier"]
    assert "user" not in parsed["agent"]


def test_task_toml_template_renders_optional_user(tmp_path: Path) -> None:
    parsed = tomllib.loads(_render_task_toml(tmp_path, user="1000:1000"))

    assert parsed["verifier"]["user"] == "1000:1000"
    assert parsed["agent"]["user"] == "1000:1000"


def test_adapter_generates_every_source_row(tmp_path: Path) -> None:
    csv_path = tmp_path / "DSQA-full.csv"
    csv_path.write_text(
        "example_id,problem,problem_category,answer,answer_type\n"
        'source-a,"Question A?",Arts,Answer A,Single Answer\n'
        'source-b,"Question B?",Science,Answer B,Set Answer\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "tasks"

    DeepSearchQAAdapter(output_dir=output_dir, csv_path=csv_path).run()

    task_dirs = sorted(path.name for path in output_dir.iterdir())
    assert task_dirs == ["deepsearchqa-0001", "deepsearchqa-0002"]
    second_metadata = json.loads((output_dir / "deepsearchqa-0002" / "tests" / "metadata.json").read_text())
    assert second_metadata["source_id"] == "source-b"
    assert second_metadata["source_index"] == 2


def test_rendered_task_toml_matches_harbor_schema(tmp_path: Path) -> None:
    harbor_config = pytest.importorskip("harbor.models.task.config")
    task_config = harbor_config.TaskConfig.model_validate_toml(_render_task_toml(tmp_path, user="1000:1000"))

    assert version("harbor") == "0.22.0"
    assert task_config.schema_version == "1.4"
    assert task_config.task is not None
    assert task_config.task.name == "aiq/deepsearchqa-0007"
