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

import logging
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_telemetry_exporter

logger = logging.getLogger(__name__)


def ensure_registered() -> None:
    return None


def _should_redact_from_headers(headers: dict[str, Any]) -> bool:
    for value in headers.values():
        if value is None:
            continue
        if str(value).strip().lower() == "true":
            return True
    return False


try:
    from nat.observability.mixin.redaction_config_mixin import HeaderRedactionConfigMixin  # type: ignore
except Exception:

    class HeaderRedactionConfigMixin(BaseModel):
        redaction_enabled: bool = Field(default=False, description="Whether to enable redaction processing.")
        redaction_value: str = Field(default="[REDACTED]", description="Value to replace redacted attributes with.")
        redaction_attributes: list[str] = Field(
            default_factory=lambda: ["input.value", "output.value", "nat.metadata"],
            description="Attributes to redact when redaction is triggered.",
        )
        force_redaction: bool = Field(default=False, description="Always redact regardless of other conditions.")
        redaction_tag: str | None = Field(default=None, description="Tag to add to spans when redaction is triggered.")
        redaction_headers: list[str] = Field(default_factory=list, description="Headers to check for redaction.")


try:
    from nat.observability.mixin.tagging_config_mixin import PrivacyTaggingConfigMixin  # type: ignore
except Exception:

    class PrivacyTaggingConfigMixin(BaseModel):
        tags: dict[str, str] | None = Field(default=None, description="Tags to add to spans.")


try:
    from nat.plugins.opentelemetry.register import OtelCollectorTelemetryExporter  # type: ignore
except Exception as e:
    raise RuntimeError(
        "OpenTelemetry telemetry plugin is required for otelcollector_redaction. "
        "Install NeMo Agent toolkit with telemetry extras."
    ) from e


class OtelCollectorRedactionTelemetryExporter(
    HeaderRedactionConfigMixin,
    PrivacyTaggingConfigMixin,
    OtelCollectorTelemetryExporter,
    name="otelcollector_redaction",
):
    resource_attributes: dict[str, str] = Field(default_factory=dict, description="Resource attributes to attach.")


@register_telemetry_exporter(config_type=OtelCollectorRedactionTelemetryExporter)
async def otelcollector_redaction_telemetry_exporter(
    config: OtelCollectorRedactionTelemetryExporter, _builder: Builder
):
    from nat.plugins.opentelemetry.otel_span_exporter import get_opentelemetry_sdk_version
    from nat.plugins.opentelemetry.otlp_span_redaction_adapter_exporter import OTLPSpanHeaderRedactionAdapterExporter

    default_resource_attributes = {
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.version": get_opentelemetry_sdk_version(),
        "service.name": config.project,
    }
    merged_resource_attributes = {**default_resource_attributes, **config.resource_attributes}

    yield OTLPSpanHeaderRedactionAdapterExporter(
        endpoint=config.endpoint,
        resource_attributes=merged_resource_attributes,
        batch_size=config.batch_size,
        flush_interval=config.flush_interval,
        max_queue_size=config.max_queue_size,
        drop_on_overflow=config.drop_on_overflow,
        shutdown_timeout=config.shutdown_timeout,
        redaction_attributes=config.redaction_attributes,
        redaction_headers=config.redaction_headers,
        redaction_callback=_should_redact_from_headers,
        redaction_enabled=config.redaction_enabled,
        force_redaction=config.force_redaction,
        redaction_tag=config.redaction_tag,
        redaction_value=getattr(config, "redaction_value", "[REDACTED]"),
        tags=config.tags,
    )
