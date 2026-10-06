# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from pydantic import Field

# ---------------------------------------------------------------------------
# Config models (Pydantic) — deserialised from YAML
# ---------------------------------------------------------------------------


class ModelPriceConfig(BaseModel):
    """Per-token prices for one model, in USD per 1 M tokens."""

    input_per_1m_tokens: float
    output_per_1m_tokens: float
    # Optional — if omitted, cached tokens are billed at the full input rate
    # (i.e. no caching discount).
    cached_input_per_1m_tokens: float | None = None


class ToolPriceConfig(BaseModel):
    """Per-call price for one tool (e.g. a search API)."""

    cost_per_call: float = 0.0


class PricingRegistryConfig(BaseModel):
    """
    Pricing table read from the ``tokenomics.pricing`` section of the eval
    config YAML.  ``models`` is keyed by the exact model name that appears in
    Relay traces (e.g. ``"nvidia/nemotron-3-ultra-550b-a55b"``). ``default`` is used as a
    fallback when no model key matches.  ``tools`` is keyed by tool name as it
    appears in the trace.
    """

    models: dict[str, ModelPriceConfig] = Field(default_factory=dict)
    tools: dict[str, ToolPriceConfig] = Field(default_factory=dict)
    default: ModelPriceConfig | None = None


# ---------------------------------------------------------------------------
# Runtime objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPrice:
    """Resolved per-token prices for a single model."""

    input_per_1m_tokens: float
    cached_input_per_1m_tokens: float
    output_per_1m_tokens: float

    def cost(self, prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> float:
        """Return USD cost for one LLM call."""
        uncached = max(0, prompt_tokens - cached_tokens)
        return (
            uncached * self.input_per_1m_tokens
            + cached_tokens * self.cached_input_per_1m_tokens
            + completion_tokens * self.output_per_1m_tokens
        ) / 1_000_000

    def cache_savings(self, cached_tokens: int) -> float:
        """USD saved vs. paying full input price for cached tokens."""
        return cached_tokens * (self.input_per_1m_tokens - self.cached_input_per_1m_tokens) / 1_000_000


@dataclass(frozen=True)
class ToolPrice:
    """Resolved per-call price for a single tool."""

    cost_per_call: float = 0.0


class PricingRegistry:
    """
    Maps model names to :class:`ModelPrice` objects and tool names to
    :class:`ToolPrice` objects.

    Model lookup order:
    1. Exact match on ``model_name``.
    2. Substring match — useful for versioned or provider-prefixed names
       (e.g. ``"nvidia/nemotron-3-ultra-550b-a55b"`` matches key ``"nemotron-3-ultra"``).
    3. ``default`` price, if configured.
    4. :class:`KeyError`.

    Tool lookup order:
    1. Exact match on ``tool_name``.
    2. Substring match (key in name, or name in key).
    3. Zero-cost default (tool costs are optional — no KeyError raised).
    """

    def __init__(
        self,
        prices: dict[str, ModelPrice],
        default: ModelPrice | None = None,
        tools: dict[str, ToolPrice] | None = None,
    ):
        self._prices = prices
        self._default = default
        self._tools: dict[str, ToolPrice] = tools or {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: PricingRegistryConfig) -> PricingRegistry:
        prices = {}
        for name, cfg in config.models.items():
            cached = cfg.cached_input_per_1m_tokens
            if cached is None:
                cached = cfg.input_per_1m_tokens
            prices[name] = ModelPrice(
                input_per_1m_tokens=cfg.input_per_1m_tokens,
                cached_input_per_1m_tokens=cached,
                output_per_1m_tokens=cfg.output_per_1m_tokens,
            )

        default = None
        if config.default is not None:
            cached = config.default.cached_input_per_1m_tokens
            if cached is None:
                cached = config.default.input_per_1m_tokens
            default = ModelPrice(
                input_per_1m_tokens=config.default.input_per_1m_tokens,
                cached_input_per_1m_tokens=cached,
                output_per_1m_tokens=config.default.output_per_1m_tokens,
            )

        tools = {name: ToolPrice(cost_per_call=cfg.cost_per_call) for name, cfg in config.tools.items()}

        return cls(prices, default, tools)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PricingRegistry:
        """Convenience constructor: pass the raw ``tokenomics.pricing`` dict."""
        return cls.from_config(PricingRegistryConfig(**raw))

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, model_name: str) -> ModelPrice:
        if model_name in self._prices:
            return self._prices[model_name]
        for key, price in self._prices.items():
            if key in model_name or model_name in key:
                return price
        if self._default is not None:
            return self._default
        raise KeyError(
            f"No price configured for model {model_name!r}. "
            "Add it to tokenomics.pricing.models in the config file, "
            "or set tokenomics.pricing.default."
        )

    def get_tool(self, tool_name: str) -> ToolPrice:
        """Return the :class:`ToolPrice` for ``tool_name``.

        Never raises — returns a zero-cost :class:`ToolPrice` if no match is
        found, so unconfigured tools simply contribute $0.
        """
        if tool_name in self._tools:
            return self._tools[tool_name]
        for key, price in self._tools.items():
            if key in tool_name or tool_name in key:
                return price
        return ToolPrice(cost_per_call=0.0)

    def known_models(self) -> list[str]:
        return list(self._prices)

    def known_tools(self) -> list[str]:
        return list(self._tools)
