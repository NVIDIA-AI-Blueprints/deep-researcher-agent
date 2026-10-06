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

"""Shared Guardrails middleware behavior."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from collections.abc import Callable
from typing import Any

from nemoguardrails.rails.llm.options import GenerationLogOptions
from nemoguardrails.rails.llm.options import GenerationOptions
from nemoguardrails.rails.llm.options import GenerationResponse

from deep_researcher_agent.guardrails.dynamic_field_selection import DynamicFieldSelectionMixin
from nat.builder.builder import Builder
from nat.middleware.function_middleware import CallNextStream
from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.middleware import InvocationContext
from nat.plugins.security.middleware.guardrails.nemo_guardrails_middleware import GuardrailsMiddleware
from nat.plugins.security.middleware.guardrails.nemo_guardrails_middleware_config import GuardrailsMiddlewareConfig

logger = logging.getLogger(__name__)

_GUARDRAILS_FAILURE_REFUSAL = "I'm sorry, I can't help with that."
_NEMO_GUARDRAILS_RUNTIME_LOGGER = "nemoguardrails.colang.v1_0.runtime.runtime"


def _configure_nemo_guardrails_runtime_logging() -> None:
    """Prevent NeMo Guardrails runtime context from being logged below WARNING."""
    runtime_logger = logging.getLogger(_NEMO_GUARDRAILS_RUNTIME_LOGGER)
    if runtime_logger.getEffectiveLevel() < logging.WARNING:
        runtime_logger.setLevel(logging.WARNING)


class GuardrailsMixin(DynamicFieldSelectionMixin, GuardrailsMiddleware):
    """Provide shared Guardrails behavior for boundary-specific middleware.

    This mixin adds dynamic field-selection traversal and block-response
    adaptation hooks so concrete middleware can guard selected boundary fields
    while preserving the intercepted function's expected return schema.
    """

    def __init__(self, config: GuardrailsMiddlewareConfig, builder: Builder) -> None:
        """Initialize Guardrails with a safe default for dependency runtime logs."""
        # NeMo Guardrails logs complete event/context payloads at INFO. Set the
        # dependency logger before LLMRails construction so request content and
        # credentials cannot be emitted during initialization or evaluation.
        _configure_nemo_guardrails_runtime_logging()
        super().__init__(config=config, builder=builder)

    async def pre_invoke(self, context: InvocationContext) -> InvocationContext | None:
        """Run input rails and adapt blocked outputs for the intercepted boundary."""
        try:
            result = await super().pre_invoke(context)
        except Exception as exc:
            logger.error(
                "Input Guardrails failed while evaluating selected fields; refusing request; error_type=%s",
                type(exc).__name__,
            )
            context.output = self._on_pre_invoke_blocked(context, _GUARDRAILS_FAILURE_REFUSAL)
            return context

        current_context = result or context
        # NAT returns a context for both input rewrites and blocks; only blocks populate output with a refusal.
        blocked = result is not None and current_context.output is not None
        if blocked and isinstance(current_context.output, str):
            current_context.output = self._on_pre_invoke_blocked(current_context, current_context.output)
        return result

    def _on_pre_invoke_blocked(self, context: InvocationContext, block_message: str) -> object:
        """Adapt input-rail block output for the intercepted boundary."""
        return block_message

    async def post_invoke(self, context: InvocationContext) -> InvocationContext | None:
        """Run output rails and adapt blocked outputs for the intercepted boundary."""
        try:
            result = await super().post_invoke(context)
            if result is not None:
                self._synchronize_terminal_output(context)
            return result
        except Exception as exc:
            logger.error(
                "Output Guardrails failed while evaluating selected fields for function %s; "
                "refusing response; error_type=%s",
                context.function_context.name,
                type(exc).__name__,
            )
            context.output = self._refuse_output_safely(context, context.output)
            return context

    async def function_middleware_stream(
        self,
        *args: Any,
        call_next: CallNextStream,
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Run Guardrails around a stream without stringifying structured chunks."""
        ctx = InvocationContext(
            function_context=context,
            original_args=args,
            original_kwargs=dict(kwargs),
            modified_args=args,
            modified_kwargs=dict(kwargs),
            output=None,
        )

        result = await self.pre_invoke(ctx)
        if result is not None:
            ctx = result
        if ctx.output is not None:
            yield ctx.output
            return

        # Final-output rails evaluate completed output, so match NAT's buffering
        # semantics while preserving structured chunks instead of stringifying them.
        buffered = [chunk async for chunk in call_next(*ctx.modified_args, **ctx.modified_kwargs)]
        if not buffered:
            return

        if all(isinstance(chunk, str) for chunk in buffered):
            ctx.output = "".join(buffered)
            result = await self.post_invoke(ctx)
            if result is not None:
                ctx = result
            yield ctx.output
            return

        try:
            output, blocked = await self._apply_output_rails_to_structured_stream(ctx, buffered)
        except Exception as exc:
            logger.error(
                "Output Guardrails failed while evaluating buffered structured output for function %s; "
                "refusing response; error_type=%s",
                context.name,
                type(exc).__name__,
            )
            ctx.output = self._refuse_output_safely(ctx, buffered[0])
            yield ctx.output
            return
        if blocked:
            yield output
            return

        for chunk in buffered:
            yield chunk

    async def _apply_output_rails_to_structured_stream(
        self,
        context: InvocationContext,
        buffered: list[object],
    ) -> tuple[object, bool]:
        """Evaluate output rails against buffered structured assistant output.

        Structured stream chunks are already buffered before emission. Evaluate
        the logical assistant text reconstructed from selected fields so rails
        can catch content split across chunk boundaries.
        """
        await self.bind_llms_to_rail()

        input_text = ""
        if context.original_args:
            raw: Any = context.original_args[0]
            input_text = getattr(raw, "input_message", None) or (raw if isinstance(raw, str) else str(raw))

        paths = self._resolve_guarded_targets_for_phase(context.function_context.name, "post_invoke")
        selections = self._stream_output_selections(buffered, paths)
        if not selections:
            return buffered[0], False

        text = "".join(selection[0] for selection in selections)
        messages = [{"role": "user", "content": input_text}] if input_text else []
        messages.append({"role": "assistant", "content": text})
        response: GenerationResponse = await self._llm_rails.generate_async(
            messages=messages,
            options=GenerationOptions(
                rails=["output"],
                log=GenerationLogOptions(activated_rails=True),
                output_vars=["bot_message", "user_message"],
            ),
        )

        if self._rail_blocked(response):
            context.output = buffered[0]
            block_message = self._handle_blocked_rail_response(response)
            context.output = self._on_post_invoke_blocked(context, block_message, context.output)
            return context.output, True

        result_text = self._handle_modified_rail_response(response, fallback=text)
        if result_text != text:
            selections[0][1](result_text)
            for _text, apply_to_field in selections[1:]:
                apply_to_field("")
            self._synchronize_buffered_outputs(buffered, paths)

        return buffered[0], False

    def _stream_output_selections(
        self,
        buffered: list[object],
        paths: list[str],
    ) -> list[tuple[str, Callable[[str], None]]]:
        """Return writable text selections from string and structured stream chunks."""
        selections: list[tuple[str, Callable[[str], None]]] = []
        for index, chunk in enumerate(buffered):
            if isinstance(chunk, str):
                selections.append((chunk, lambda value, index=index: buffered.__setitem__(index, value)))
            else:
                selections.extend(self._gather_guardrail_inputs(chunk, paths, lambda _value: None))
        return selections

    def _synchronize_buffered_outputs(self, buffered: list[object], paths: list[str]) -> None:
        """Let a concrete boundary align auxiliary fields with rewritten public output."""

    def _synchronize_terminal_output(self, context: InvocationContext) -> None:
        """Let a concrete boundary align auxiliary fields with guarded public output."""

    def on_post_invoke_blocked(self, context: InvocationContext, block_message: str) -> object:
        """Adapt blocked output before the intercepted result is returned."""
        return self._on_post_invoke_blocked(context, block_message, context.output)

    def _on_post_invoke_blocked(
        self,
        context: InvocationContext,
        block_message: str,
        original_output: object,
    ) -> object:
        """Adapt output-rail block output for the intercepted boundary."""
        if not isinstance(original_output, str):
            paths = self._resolve_guarded_targets_for_phase(context.function_context.name, "post_invoke")
            modified = False
            for _text, apply_to_field in self._gather_guardrail_inputs(original_output, paths, lambda _value: None):
                apply_to_field(block_message)
                modified = True
            if modified:
                self._synchronize_blocked_output(original_output, block_message)
                return original_output
            return self._build_emergency_output_refusal(context, original_output)
        return block_message

    def _synchronize_blocked_output(self, output: object, block_message: str) -> None:
        """Let a concrete boundary align auxiliary fields with a blocked public output."""

    def _refuse_output_safely(self, context: InvocationContext, original_output: object) -> object:
        """Adapt a refusal without allowing a failing target traversal to escape."""
        try:
            return self._on_post_invoke_blocked(context, _GUARDRAILS_FAILURE_REFUSAL, original_output)
        except Exception as exc:
            logger.error(
                "Output Guardrails failed while adapting refusal for function %s; "
                "using emergency refusal; error_type=%s",
                context.function_context.name,
                type(exc).__name__,
            )
            return self._build_emergency_output_refusal(context, original_output)

    def _build_emergency_output_refusal(self, context: InvocationContext, original_output: object) -> object:
        """Return the traversal-independent refusal for this intercepted boundary."""
        return _GUARDRAILS_FAILURE_REFUSAL
