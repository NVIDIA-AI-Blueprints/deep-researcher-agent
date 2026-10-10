import importlib.util
import json
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest
from deepresearch_bench_ii.adapter import DeepResearchBenchIIAdapter


def _source_row(index: int) -> dict:
    return {
        "id": f"task-{index}",
        "idx": index,
        "language": "English",
        "theme": "Science & Technology",
        "description": f"Research task {index}",
        "license": "CC-BY-4.0",
        "prompt": f"Write report {index}.",
        "content": {
            "task": f"Evaluate topic {index}.",
            "rubric": {
                "info_recall": ["Recall one fact."],
                "analysis": ["Analyze the fact."],
                "presentation": ["Present the result clearly."],
            },
            "blocked": {"references": []},
        },
    }


def _render_task_toml(tmp_path: Path, user: str | None = None) -> str:
    adapter = DeepResearchBenchIIAdapter(output_dir=tmp_path, user=user)
    row = adapter._normalize_row(_source_row(7), line_number=1)
    return adapter._render_task_toml(index=7, task_id="deepresearch-bench-ii-007", row=row)


def test_task_toml_template_renders_valid_toml(tmp_path: Path) -> None:
    rendered = _render_task_toml(tmp_path)
    parsed = tomllib.loads(rendered)

    assert "@{" not in rendered
    assert parsed["schema_version"] == "1.4"
    assert parsed["task"]["name"] == "aiq/deepresearch-bench-ii-007"
    assert parsed["metadata"]["source_id"] == "task-7"
    assert parsed["metadata"]["source_index"] == 7
    assert parsed["agent"]["timeout_sec"] == 5400.0
    assert parsed["environment"]["network_mode"] == "public"
    assert len(parsed["task"]["authors"]) == 6
    assert "user" not in parsed["verifier"]
    assert "user" not in parsed["agent"]


def test_task_toml_template_renders_optional_user(tmp_path: Path) -> None:
    parsed = tomllib.loads(_render_task_toml(tmp_path, user="1000:1000"))

    assert parsed["verifier"]["user"] == "1000:1000"
    assert parsed["agent"]["user"] == "1000:1000"


def test_adapter_generates_every_local_source_row(tmp_path: Path) -> None:
    source_path = tmp_path / "tasks_and_rubrics.jsonl"
    source_path.write_text(
        "\n".join(json.dumps(_source_row(index)) for index in (1, 2)) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "tasks"

    DeepResearchBenchIIAdapter(output_dir=output_dir, tasks_jsonl=source_path).run()

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "deepresearch-bench-ii-001",
        "deepresearch-bench-ii-002",
    ]
    second_task = output_dir / "deepresearch-bench-ii-002"
    assert (second_task / "instruction.md").read_text(encoding="utf-8") == "Write report 2."
    metadata = json.loads((second_task / "tests" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_id"] == "task-2"
    assert metadata["rubric"]["analysis"] == ["Analyze the fact."]
    assert (second_task / "tests" / "official" / "run_evaluation.py").is_file()


def test_rendered_task_toml_matches_harbor_schema(tmp_path: Path) -> None:
    harbor_config = pytest.importorskip("harbor.models.task.config")
    task_config = harbor_config.TaskConfig.model_validate_toml(_render_task_toml(tmp_path, user="1000:1000"))

    assert version("harbor") == "0.22.0"
    assert task_config.schema_version == "1.4"
    assert task_config.task is not None
    assert task_config.task.name == "aiq/deepresearch-bench-ii-007"


def test_verifier_exports_grader_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    verifier_path = Path(__file__).parents[1] / "src/deepresearch_bench_ii/task-template/tests/verifier.py"
    spec = importlib.util.spec_from_file_location("deepresearch_bench_ii_verifier", verifier_path)
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)

    monkeypatch.setattr(verifier, "GeminiClient", lambda *, verbose: object())
    monkeypatch.setattr(
        verifier.run_evaluation,
        "process_one_with_chunking",
        lambda *_args: (
            0,
            {
                "scores": {
                    "info_recall": {"fact": {"score": 1}},
                    "analysis": {"analysis": {"score": 0}},
                    "presentation": {"style": {"score": -1}},
                }
            },
            123,
        ),
    )
    report_path = tmp_path / "report.md"
    report_path.write_text("Report", encoding="utf-8")

    result = verifier.evaluate(
        {
            "source_id": "task-1",
            "task": "Research task",
            "rubric": {},
            "blocked": {},
        },
        report_path,
    )

    assert result["grader_tokens"] == 123.0
    assert "total_tokens" not in result


def test_gemini_client_uses_google_api_when_url_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    client_path = Path(__file__).parents[1] / "src/deepresearch_bench_ii/task-template/tests/official/gemini_client.py"
    spec = importlib.util.spec_from_file_location("deepresearch_bench_ii_gemini_client", client_path)
    assert spec is not None and spec.loader is not None
    gemini_client = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gemini_client
    spec.loader.exec_module(gemini_client)

    captured: dict = {}

    class Response:
        status_code = 200

        def json(self) -> dict:
            return {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                "usageMetadata": {"totalTokenCount": 4},
            }

    def post(url: str, **kwargs: object) -> Response:
        captured["url"] = url
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr(gemini_client.requests, "post", post)
    client = gemini_client.GeminiClient(
        api_url="",
        api_token="test-key",
        model="gcp/google/gemini-2.5-pro",
        verbose=False,
    )

    result = client.query(gemini_client.GeminiInput(text="prompt"))

    assert captured["url"] == ("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent")
    assert captured["headers"] == {
        "Content-Type": "application/json",
        "x-goog-api-key": "test-key",
    }
    assert captured["json"] == {
        "contents": [{"role": "user", "parts": [{"text": "prompt"}]}],
    }
    assert result.text == "ok"
    assert result.usage_metadata["totalTokenCount"] == 4
