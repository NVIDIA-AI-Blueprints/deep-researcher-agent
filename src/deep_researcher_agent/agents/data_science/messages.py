# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Message conversion helpers for the data-science agent."""

import json
import re
from typing import Any

_CLARIFICATION_CUE = re.compile(
    r"\b(?:could|can|would) you (?:please )?(?:clarify|confirm|provide|select|specify|tell me)\b"
    r"|\bplease (?:clarify|confirm|provide|select|specify)\b"
    r"|\b(?:do you want|would you like|should i)\b",
    flags=re.IGNORECASE,
)
_QUESTION_OPENING = re.compile(
    r"^(?:clarification (?:needed|required)\s*:?\s*)?"
    r"(?:which|what|when|where|who|whose|how|is|are|do|does|should|would|could|can)\b",
    flags=re.IGNORECASE,
)


def message_text(message: Any) -> str:
    """Return a compact text representation of a LangChain message."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def is_clarification_request(message: Any) -> bool:
    """Conservatively identify a response that waits for missing user input."""
    text = message_text(message).strip()
    if "?" not in text:
        return False
    if _CLARIFICATION_CUE.search(text):
        return True
    return len(text) <= 1600 and text.endswith("?") and _QUESTION_OPENING.match(text) is not None


__all__ = ["is_clarification_request", "message_text"]
