# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Message utilities for extracting content from conversation history."""

from __future__ import annotations

import logging

from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


def get_latest_user_query(messages: list[BaseMessage]) -> str:
    """Return the most recent user-authored message content.

    Iterates through messages in reverse order to find the latest
    HumanMessage, which represents the user's most recent query.

    Args:
        messages: List of conversation messages.

    Returns:
        The content of the latest user message, or empty string if none found.
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content

    if messages:
        last_message = messages[-1]
        if hasattr(last_message, "content"):
            return last_message.content

    logger.warning("No user message found in conversation history, returning empty string")
    return ""
