# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Typed Deep Researcher Agent configuration for NeMo Relay plugins."""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import AnyHttpUrl
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class _RelayBaseConfig(BaseModel):
    """Strict base for operator-facing Relay YAML."""

    model_config = ConfigDict(extra="forbid")


class RelayAtofConfig(_RelayBaseConfig):
    """Raw Relay event export configuration."""

    enabled: bool = True
    output_directory: str = "./relay"
    filename: str = "deep-researcher-relay.atof.jsonl"
    mode: Literal["append", "overwrite"] = "append"


class RelayOpenTelemetryEndpointConfig(_RelayBaseConfig):
    """One Relay 0.7 typed OTLP trace destination."""

    type: Literal["full", "gen_ai", "openinference"] = "openinference"
    endpoint: AnyHttpUrl = AnyHttpUrl("http://localhost:6006/v1/traces")
    transport: Literal["http_binary", "grpc"] = "http_binary"
    service_name: str = "deep-researcher-relay"
    service_namespace: str | None = None
    service_version: str | None = None
    instrumentation_scope: str = "nemo-relay"
    timeout_millis: int = Field(default=3000, gt=0)
    header_env: dict[str, str] = Field(default_factory=dict)
    resource_attributes: dict[str, str] = Field(
        default_factory=lambda: {"openinference.project.name": "deep-researcher-relay"}
    )


class RelayOpenTelemetryConfig(_RelayBaseConfig):
    """Relay OpenTelemetry fan-out configuration."""

    enabled: bool = False
    endpoints: list[RelayOpenTelemetryEndpointConfig] = Field(
        default_factory=lambda: [RelayOpenTelemetryEndpointConfig()]
    )


class RelayObservabilityConfig(_RelayBaseConfig):
    """Relay Observability plugin version 3 configuration."""

    enable_full_payloads: bool = True
    atof: RelayAtofConfig = Field(default_factory=RelayAtofConfig)
    opentelemetry: RelayOpenTelemetryConfig = Field(default_factory=RelayOpenTelemetryConfig)


class RelayRedactionConfig(_RelayBaseConfig):
    """Config-driven sanitization applied before subscribers and exporters."""

    enabled: bool = True
    request_privacy_attributes: list[Literal["data", "category_profile"]] = Field(
        default_factory=lambda: ["data", "category_profile"]
    )
    detectors: list[
        Literal[
            "email",
            "phone",
            "api_key",
            "bearer_token",
            "jwt",
            "credit_card",
            "aws_access_key_id",
            "aws_secret_access_key",
            "gcp_api_key",
            "azure_storage_account_key",
        ]
    ] = Field(
        default_factory=lambda: [
            "email",
            "phone",
            "api_key",
            "bearer_token",
            "jwt",
            "credit_card",
            "aws_access_key_id",
            "aws_secret_access_key",
            "gcp_api_key",
            "azure_storage_account_key",
        ]
    )


class RelayPricingConfig(_RelayBaseConfig):
    """Model-pricing enrichment for managed LLM responses."""

    enabled: bool = True
    sources: list[dict[str, Any]] = Field(default_factory=list)


class RelayConfig(_RelayBaseConfig):
    """Deep Researcher Agent-owned Relay configuration; Relay itself is always instrumented."""

    logging: bool = True
    observability: RelayObservabilityConfig = Field(default_factory=RelayObservabilityConfig)
    redaction: RelayRedactionConfig = Field(default_factory=RelayRedactionConfig)
    pricing: RelayPricingConfig = Field(default_factory=RelayPricingConfig)

    def to_plugin_config(self) -> dict[str, Any]:
        """Translate Deep Researcher Agent YAML into Relay's supported plugin document shape."""
        observability = self.observability
        atof = observability.atof
        otel = observability.opentelemetry
        components: list[dict[str, Any]] = [
            {
                "kind": "observability",
                "enabled": True,
                "config": {
                    "version": 3,
                    "enable_full_payloads": observability.enable_full_payloads,
                    "atof": {
                        "enabled": atof.enabled,
                        "sinks": [
                            {
                                "type": "file",
                                "output_directory": atof.output_directory,
                                "filename": atof.filename,
                                "mode": atof.mode,
                            }
                        ]
                        if atof.enabled
                        else [],
                    },
                    "opentelemetry": {
                        "enabled": otel.enabled,
                        "endpoints": [
                            endpoint.model_dump(mode="json", exclude_none=True) for endpoint in otel.endpoints
                        ]
                        if otel.enabled
                        else [],
                    },
                },
            }
        ]
        if self.redaction.enabled:
            components.append(
                {
                    "kind": "pii_redaction",
                    "enabled": True,
                    "config": {
                        "version": 1,
                        "profiles": [
                            {
                                "mode": "builtin",
                                "priority": 80 + index,
                                "builtin": {"action": "redact", "detector": detector},
                            }
                            for index, detector in enumerate(self.redaction.detectors)
                        ],
                    },
                }
            )
        if self.pricing.enabled:
            components.append(
                {
                    "kind": "pricing",
                    "enabled": True,
                    "config": {"sources": self.pricing.sources},
                }
            )
        return {"version": 1, "components": components}
