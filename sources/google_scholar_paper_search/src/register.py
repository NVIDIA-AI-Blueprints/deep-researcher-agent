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

"""NAT register function for Google Scholar paper search tool."""

import logging
import os

from pydantic import AliasChoices
from pydantic import Field
from pydantic import SecretStr

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from .paper_search import PaperSearchProvider
from .paper_search import PaperSearchTool

logger = logging.getLogger(__name__)

_missing_key_warned = False


class PaperSearchToolConfig(FunctionBaseConfig, name="paper_search"):
    """Configuration for the paper search tool.

    Tool that searches for academic papers using Google Scholar. The
    ``provider`` field selects the backend API: ``serper`` (default),
    ``serpapi``, ``searchapi``, or ``serply``. Each provider requires its own
    API key.
    """

    provider: PaperSearchProvider = Field(
        default=PaperSearchProvider.SERPER,
        description="Google Scholar backend: 'serper', 'serpapi', 'searchapi', or 'serply'",
    )
    timeout: int = Field(
        default=30,
        description="Timeout in seconds for the search requests",
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of search results to return",
    )
    serper_api_key: SecretStr | None = Field(
        default=None,
        description="API key for Serper (required when provider='serper')",
    )
    serpapi_api_key: SecretStr | None = Field(
        default=None,
        description="API key for SerpAPI (required when provider='serpapi')",
    )
    searchapi_api_key: SecretStr | None = Field(
        default=None,
        description="API key for SearchAPI (required when provider='searchapi')",
    )
    serply_api_key: SecretStr | None = Field(
        default=None,
        description="API key for Serply (required when provider='serply')",
    )


# Maps each provider to (env var name, config attr name, sign-up URL)
_PROVIDER_KEY_INFO = {
    PaperSearchProvider.SERPER: ("SERPER_API_KEY", "serper_api_key", "https://serper.dev/"),
    PaperSearchProvider.SERPAPI: ("SERPAPI_API_KEY", "serpapi_api_key", "https://serpapi.com/"),
    PaperSearchProvider.SEARCHAPI: ("SEARCHAPI_API_KEY", "searchapi_api_key", "https://www.searchapi.io/"),
    PaperSearchProvider.SERPLY: ("SERPLY_API_KEY", "serply_api_key", "https://serply.io/"),
}


def _resolve_api_key(provider: PaperSearchProvider, tool_config: PaperSearchToolConfig) -> str | None:
    env_var, config_attr, _ = _PROVIDER_KEY_INFO[provider]
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value
    config_value = getattr(tool_config, config_attr)
    if config_value:
        return config_value.get_secret_value()
    return None


@register_function(config_type=PaperSearchToolConfig)
async def paper_search(tool_config: PaperSearchToolConfig, builder: Builder):
    provider = tool_config.provider
    api_key = _resolve_api_key(provider, tool_config)

    if not api_key:
        env_var, _, signup_url = _PROVIDER_KEY_INFO[provider]
        global _missing_key_warned
        if not _missing_key_warned:
            logger.warning(
                "%s not found for provider '%s'. The paper search tool will be registered but "
                "will return an error when called. To enable: set %s in your environment, .env "
                "file, or specify the API key in your workflow config.",
                env_var,
                provider.value,
                env_var,
            )
            _missing_key_warned = True

        async def _paper_search_stub(
            query: str = Field(..., validation_alias=AliasChoices("query", "question")),
            year: str | int | None = None,
        ) -> str:
            return (
                f"Error: Paper search is unavailable because {env_var} is not set "
                f"(provider='{provider.value}').\n"
                "To enable this tool:\n"
                f"1. Get an API key from {signup_url}\n"
                f"2. Set the API key in your environment or .env file as {env_var}\n"
                "   (alternatively, specify the API key in your workflow config)\n"
                "3. Restart the application"
            )

        yield FunctionInfo.from_fn(
            _paper_search_stub,
            description=(
                f"Search for academic papers and peer-reviewed scientific publications "
                f"on Google Scholar via the {provider.value} backend. This tool is "
                f"registered in a degraded state because {env_var} is not configured; "
                f"calling it returns setup instructions instead of search results."
            ),
        )
        return

    tool = PaperSearchTool(
        provider=provider,
        serper_api_key=api_key if provider is PaperSearchProvider.SERPER else None,
        serpapi_api_key=api_key if provider is PaperSearchProvider.SERPAPI else None,
        searchapi_api_key=api_key if provider is PaperSearchProvider.SEARCHAPI else None,
        serply_api_key=api_key if provider is PaperSearchProvider.SERPLY else None,
        timeout=tool_config.timeout,
        max_results=tool_config.max_results,
    )

    async def _paper_search(
        query: str,
        year: str | int | None = None,
    ) -> str:
        """Searches for peer-reviewed academic papers and scientific publications.

        This tool returns papers from Google Scholar with citations, abstracts,
        and links for research queries requiring authoritative, scholarly sources
        including: scientific concepts, algorithms, methodologies, technical
        foundations, theoretical frameworks, empirical studies, and peer-reviewed
        evidence.

        Args:
            query (str): The search query string.
            year (str | int | None): Optional year or year range (e.g., "2023" or "2020-2023").

        Returns:
            str: Formatted string with search results.
        """
        return await tool.search(query, year)

    yield FunctionInfo.from_fn(
        _paper_search,
        description=_paper_search.__doc__,
    )
