#!/usr/bin/env python3
# ruff: noqa: E501
"""DeepSearchQA verifier adapted from the official starter notebook."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any

DEEPSEARCH_QA_PROMPT = """\
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a string), `"Correctness Details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and `"Excessive Answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.


"""


GRADER_RATING_OUTPUT_EXAMPLE = r"""**Example (Partial):**

"```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true,
    }},
    "Excessive Answers": [ "Italy" ]
  }}
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""


REWARD_KEYS = {
    "reward",
    "fully_correct",
    "fully_incorrect",
    "correct_with_excessive_answers",
    "precision",
    "recall",
    "f1_score",
    "grader_valid",
}


def parse_json_response(raw_response: str) -> Any:
    json_str = raw_response.strip()
    start_marker = "```json"
    start_idx = json_str.find(start_marker)
    if start_idx != -1:
        json_str = json_str[start_idx + len(start_marker) :].strip()
        end_idx = json_str.rfind("```")
        if end_idx != -1:
            json_str = json_str[:end_idx].strip()
    return json.loads(json_str)


def get_correctness_details(json_response: Any) -> dict[str, bool] | None:
    try:
        details = json_response["Answer Correctness"]["Correctness Details"]
    except (KeyError, TypeError):
        return None
    if not isinstance(details, dict):
        return None
    if not all(isinstance(key, str) for key in details):
        return None
    if not all(isinstance(value, bool) for value in details.values()):
        return None
    return details


def get_excessive_answers(json_response: Any) -> list[str] | None:
    try:
        excessive_answers = json_response["Answer Correctness"]["Excessive Answers"]
    except (KeyError, TypeError):
        return []
    if not isinstance(excessive_answers, list):
        return None
    if not all(isinstance(item, str) for item in excessive_answers):
        return None
    return excessive_answers


def calculate_metric(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> dict[str, float]:
    precision = 0.0
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)

    recall = 0.0
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)

    f1_score = 0.0
    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def build_rating_prompt(metadata: dict[str, str], response: str) -> str:
    return DEEPSEARCH_QA_PROMPT + GRADER_RATING_OUTPUT_EXAMPLE.format(
        prompt=metadata["problem"].strip(),
        prompt_type=metadata["answer_type"].strip(),
        answer=metadata["answer"].strip(),
        response=response.strip(),
    )


def _native_gemini_model(model: str) -> str:
    prefix = "gcp/google/"
    if model.startswith(prefix):
        return model[len(prefix) :]
    return model


def _is_openai_chat_completions_url(url: str) -> bool:
    return "chat/completions" in url.lower()


def _call_openai_compatible(prompt: str) -> str:
    import requests

    api_url = os.environ.get("GEMINI_API_URL", "").strip()
    api_key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("DSQA_GRADER_MODEL", "gcp/google/gemini-2.5-flash")
    if not api_url:
        raise RuntimeError("Set GEMINI_API_URL for the OpenAI-compatible grader.")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY for the verifier.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(api_url, json=payload, headers=headers, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Empty response from grader.")
    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Empty grader content.")
    return content.strip()


def _call_google_genai(prompt: str) -> str:
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY for the verifier.")

    model = _native_gemini_model(os.environ.get("DSQA_GRADER_MODEL", "gcp/google/gemini-2.5-flash"))
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty Gemini response.")
    return text.strip()


def call_gemini(prompt: str) -> str:
    api_url = os.environ.get("GEMINI_API_URL", "").strip()
    max_retries = int(os.environ.get("DSQA_GRADER_MAX_RETRIES", "5"))
    if api_url and _is_openai_chat_completions_url(api_url):
        caller = _call_openai_compatible
    else:
        caller = _call_google_genai

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return caller(prompt)
        except Exception as exc:  # pragma: no cover - depends on remote API
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(1 + (2 ** (attempt + random.random())))

    raise RuntimeError(f"Gemini grader failed after {max_retries} attempts: {last_error}")


def evaluate(metadata: dict[str, str], answer_text: str) -> dict[str, Any]:
    rating_prompt = build_rating_prompt(metadata, answer_text)
    grader_response = call_gemini(rating_prompt)
    parsed = parse_json_response(grader_response)

    answer_correctness = parsed.get("Answer Correctness") if isinstance(parsed, dict) else None
    if not isinstance(answer_correctness, dict):
        raise ValueError("Missing or malformed 'Answer Correctness' node.")

    explanation = answer_correctness.get("Explanation")
    if not isinstance(explanation, str):
        raise ValueError("Missing or malformed 'Explanation'.")

    details = get_correctness_details(parsed)
    if details is None:
        raise ValueError("Missing or malformed 'Correctness Details'.")

    excessive_answers = get_excessive_answers(parsed)
    if excessive_answers is None:
        raise ValueError("Missing or malformed 'Excessive Answers'.")

    ratings = list(details.values())
    true_positives = sum(1 for rating in ratings if rating)
    false_negatives = len(ratings) - true_positives
    false_positives = len(excessive_answers)
    metrics = calculate_metric(true_positives, false_positives, false_negatives)

    has_expected_answers = bool(ratings)
    all_expected_answers_correct = has_expected_answers and true_positives == len(ratings)
    fully_incorrect = has_expected_answers and true_positives == 0
    correct_with_excessive_answers = bool(excessive_answers) and (
        all_expected_answers_correct or not has_expected_answers
    )
    all_correct_no_excessive = (all_expected_answers_correct or not has_expected_answers) and not excessive_answers

    return {
        "reward": metrics["f1_score"],
        "fully_correct": 1.0 if all_correct_no_excessive else 0.0,
        "fully_incorrect": 1.0 if fully_incorrect else 0.0,
        "correct_with_excessive_answers": (1.0 if correct_with_excessive_answers else 0.0),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "grader_valid": 1.0,
        "expected_answer_count": len(ratings),
        "correct_answer_count": true_positives,
        "excessive_answer_count": false_positives,
        "answer_correctness_explanation": explanation,
        "correctness_details": details,
        "excessive_answers": excessive_answers,
        "rating_prompt": rating_prompt,
        "rating_response": grader_response,
    }


def write_outputs(result: dict[str, Any], verifier_dir: Path) -> None:
    verifier_dir.mkdir(parents=True, exist_ok=True)

    reward_payload = {
        key: value
        for key, value in result.items()
        if key in REWARD_KEYS and isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    (verifier_dir / "reward.json").write_text(
        json.dumps(reward_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (verifier_dir / "grading.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_path", type=Path)
    parser.add_argument("answer_path", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    verifier_dir = Path("/logs/verifier")

    try:
        metadata = json.loads(args.metadata_path.read_text(encoding="utf-8"))
        answer_text = args.answer_path.read_text(encoding="utf-8").strip()
        if not answer_text:
            raise ValueError("Agent answer file is empty.")
        result = evaluate(metadata, answer_text)
    except Exception as exc:
        result = {
            "reward": 0.0,
            "fully_correct": 0.0,
            "fully_incorrect": 0.0,
            "correct_with_excessive_answers": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "grader_valid": 0.0,
            "error": str(exc),
        }

    write_outputs(result, verifier_dir)


if __name__ == "__main__":
    main()
