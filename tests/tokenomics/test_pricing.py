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

"""Tests for tokenomics pricing (ModelPrice, PricingRegistry).

Module under test: src/deep_researcher_agent/tokenomics/pricing.py
"""

import pytest

from deep_researcher_agent.tokenomics.pricing import ModelPrice
from deep_researcher_agent.tokenomics.pricing import ModelPriceConfig
from deep_researcher_agent.tokenomics.pricing import PricingRegistry
from deep_researcher_agent.tokenomics.pricing import PricingRegistryConfig
from deep_researcher_agent.tokenomics.pricing import ToolPriceConfig


def test_model_price_cost_uncached_and_completion():
    mp = ModelPrice(
        input_per_1m_tokens=1.0,
        cached_input_per_1m_tokens=0.5,
        output_per_1m_tokens=4.0,
    )
    # 1M prompt (all uncached) + 500k completion
    assert mp.cost(1_000_000, 0, 500_000) == pytest.approx(1.0 + 2.0)


def test_model_price_cost_with_cache_split():
    mp = ModelPrice(
        input_per_1m_tokens=2.0,
        cached_input_per_1m_tokens=0.5,
        output_per_1m_tokens=1.0,
    )
    # 800k prompt, 300k cached -> 500k uncached
    assert mp.cost(800_000, 300_000, 100_000) == pytest.approx(
        (500_000 * 2.0 + 300_000 * 0.5 + 100_000 * 1.0) / 1_000_000
    )


def test_model_price_cache_savings():
    mp = ModelPrice(
        input_per_1m_tokens=2.0,
        cached_input_per_1m_tokens=0.5,
        output_per_1m_tokens=1.0,
    )
    assert mp.cache_savings(400_000) == pytest.approx(400_000 * (2.0 - 0.5) / 1_000_000)


def test_pricing_registry_from_config_cached_defaults_to_input():
    reg = PricingRegistry.from_config(
        PricingRegistryConfig(
            models={
                "m": ModelPriceConfig(
                    input_per_1m_tokens=1.0,
                    output_per_1m_tokens=2.0,
                ),
            },
        )
    )
    mp = reg.get("m")
    assert mp.cached_input_per_1m_tokens == 1.0


def test_pricing_registry_get_exact_match():
    reg = PricingRegistry.from_dict(
        {
            "models": {
                "azure/openai/gpt-5.2": {
                    "input_per_1m_tokens": 1.0,
                    "output_per_1m_tokens": 2.0,
                },
            },
            "default": {"input_per_1m_tokens": 9.0, "output_per_1m_tokens": 9.0},
        }
    )
    assert reg.get("azure/openai/gpt-5.2").input_per_1m_tokens == 1.0


def test_pricing_registry_get_substring_match():
    reg = PricingRegistry.from_dict(
        {
            "models": {
                "gpt-5.2": {"input_per_1m_tokens": 1.5, "output_per_1m_tokens": 3.0},
            },
            "default": {"input_per_1m_tokens": 9.0, "output_per_1m_tokens": 9.0},
        }
    )
    p = reg.get("azure/openai/gpt-5.2")
    assert p.input_per_1m_tokens == 1.5


def test_pricing_registry_get_fallback_default():
    reg = PricingRegistry.from_dict(
        {
            "models": {},
            "default": {"input_per_1m_tokens": 0.5, "output_per_1m_tokens": 1.5},
        }
    )
    assert reg.get("any/model").output_per_1m_tokens == 1.5


def test_pricing_registry_get_missing_raises():
    reg = PricingRegistry.from_dict({"models": {}})
    with pytest.raises(KeyError, match="No price configured"):
        reg.get("unknown/model")


def test_pricing_registry_get_tool_exact_and_substring():
    reg = PricingRegistry.from_dict(
        {
            "models": {"m": {"input_per_1m_tokens": 1.0, "output_per_1m_tokens": 1.0}},
            "tools": {
                "paper_search": ToolPriceConfig(cost_per_call=0.0003),
            },
        }
    )
    assert reg.get_tool("paper_search").cost_per_call == pytest.approx(0.0003)
    assert reg.get_tool("my_paper_search_tool").cost_per_call == pytest.approx(0.0003)


def test_pricing_registry_get_tool_unknown_is_zero():
    reg = PricingRegistry.from_dict(
        {"models": {"m": {"input_per_1m_tokens": 1.0, "output_per_1m_tokens": 1.0}}, "tools": {}}
    )
    assert reg.get_tool("no_such_tool").cost_per_call == 0.0


def test_pricing_registry_known_models_and_tools():
    reg = PricingRegistry.from_dict(
        {
            "models": {"a": {"input_per_1m_tokens": 1.0, "output_per_1m_tokens": 1.0}},
            "tools": {"t": {"cost_per_call": 0.01}},
        }
    )
    assert reg.known_models() == ["a"]
    assert reg.known_tools() == ["t"]
