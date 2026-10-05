# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral ontology capability assignments."""

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from nat.data_models.component_ref import FunctionRef


class OntologyProviderConfig(BaseModel):
    """Assign one ontology provider's tools to catalog, analytical, and predictive roles."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    catalog_tools: list[FunctionRef] = Field(min_length=1)
    analytical_tools: list[FunctionRef] = Field(min_length=1)
    predictive_tools: list[FunctionRef] = Field(default_factory=list)

    @field_validator("provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        """Normalize and reject blank provider identifiers."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("ontology provider identifier must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_tool_roles(self) -> "OntologyProviderConfig":
        """Reject duplicate or ambiguous role assignments."""

        role_names = {
            "catalog_tools": [str(tool) for tool in self.catalog_tools],
            "analytical_tools": [str(tool) for tool in self.analytical_tools],
            "predictive_tools": [str(tool) for tool in self.predictive_tools],
        }
        for role, names in role_names.items():
            if len(names) != len(set(names)):
                raise ValueError(f"ontology provider {role} must not contain duplicates")

        assigned_roles: dict[str, str] = {}
        overlap: set[str] = set()
        for role, names in role_names.items():
            for name in names:
                if name in assigned_roles:
                    overlap.add(name)
                assigned_roles[name] = role
        if overlap:
            raise ValueError(f"ontology provider tool roles overlap: {', '.join(sorted(overlap))}")
        return self

    @property
    def catalog_tool_names(self) -> frozenset[str]:
        """Return exact catalog tool names."""

        return frozenset(str(tool) for tool in self.catalog_tools)

    @property
    def analytical_tool_names(self) -> frozenset[str]:
        """Return exact analytical tool names."""

        return frozenset(str(tool) for tool in self.analytical_tools)

    @property
    def predictive_tool_names(self) -> frozenset[str]:
        """Return exact predictive tool names."""

        return frozenset(str(tool) for tool in self.predictive_tools)

    @property
    def tool_names(self) -> frozenset[str]:
        """Return every assigned provider tool."""

        return self.catalog_tool_names | self.analytical_tool_names | self.predictive_tool_names


__all__ = ["OntologyProviderConfig"]
