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

"""Field-selection extensions for guardrails middleware."""

from __future__ import annotations

import types
import typing
from collections.abc import Callable
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import RootModel

from nat.plugins.security.middleware.guardrails.nemo_guardrails_middleware import GenerationLogOptions
from nat.plugins.security.middleware.guardrails.nemo_guardrails_middleware import GenerationOptions


class PhaseFieldSelectionConfig(BaseModel):
    """Phase-specific selections for a dynamically intercepted function."""

    model_config = ConfigDict(extra="forbid")

    pre_invoke: dict[str, list[str] | dict[str, list[str]]] = Field(default_factory=dict)
    post_invoke: dict[str, list[str] | dict[str, list[str]]] = Field(default_factory=dict)


class FunctionFieldSelection(RootModel[PhaseFieldSelectionConfig | dict[str, list[str] | dict[str, list[str]]]]):
    """Field selection for one dynamically intercepted function."""

    root: PhaseFieldSelectionConfig | dict[str, list[str] | dict[str, list[str]]] = Field(default_factory=dict)


class DynamicFieldSelectionConfigMixin(BaseModel):
    """Config extension for model-member field selections on dynamic middleware."""

    workflow_functions: list[str] | dict[str, FunctionFieldSelection] | None = Field(
        default=None,
        description="Workflow functions to wrap and optional field or model-member field selections.",
    )


class DynamicFieldSelectionMixin:
    """Traversal extension for dynamic middleware field selections."""

    async def pre_invoke(self, context: Any) -> Any:
        """Run input rails over the configured pre-invoke field selections."""
        await self.bind_llms_to_rail()

        if not context.modified_args or context.modified_args[0] is None:
            return None

        value: Any = context.modified_args[0]
        paths = self._resolve_guarded_targets_for_phase(context.function_context.name, "pre_invoke")

        def apply_to_value(new_value: str) -> None:
            self._apply_modified_input(context, new_value)

        modified = False

        for text, apply_to_field in self._gather_guardrail_inputs(value, paths, apply_to_value):
            response = await self._llm_rails.generate_async(
                prompt=text,
                options=GenerationOptions(
                    rails=["input"],
                    log=GenerationLogOptions(activated_rails=True),
                    output_vars=["user_message", "bot_message"],
                ),
            )

            if self._rail_blocked(response):
                context.output = self._handle_blocked_rail_response(response)
                return context

            result_text = self._handle_modified_rail_response(response, fallback=text)

            if result_text != text:
                apply_to_field(result_text)
                modified = True

        return context if modified else None

    async def post_invoke(self, context: Any) -> Any:
        """Run output rails over the configured post-invoke field selections."""
        await self.bind_llms_to_rail()

        if context.output is None:
            return None

        input_text = ""

        if context.original_args:
            raw: Any = context.original_args[0]
            input_text = getattr(raw, "input_message", None) or (raw if isinstance(raw, str) else str(raw))

        value: Any = context.output
        paths = self._resolve_guarded_targets_for_phase(context.function_context.name, "post_invoke")

        def apply_to_value(new_value: str) -> None:
            context.output = new_value

        modified = False

        for text, apply_to_field in self._gather_guardrail_inputs(value, paths, apply_to_value):
            messages = [{"role": "user", "content": input_text}] if input_text else []
            messages.append({"role": "assistant", "content": text})
            response = await self._llm_rails.generate_async(
                messages=messages,
                options=GenerationOptions(
                    rails=["output"],
                    log=GenerationLogOptions(activated_rails=True),
                    output_vars=["bot_message", "user_message"],
                ),
            )

            if self._rail_blocked(response):
                context.output = self.on_post_invoke_blocked(context, self._handle_blocked_rail_response(response))
                return context

            result_text = self._handle_modified_rail_response(response, fallback=text)

            if result_text != text:
                apply_to_field(result_text)
                modified = True

        return context if modified else None

    def _path_resolves_to_string(self, schema: type[BaseModel], path: str) -> bool:
        """Return whether a dotted path resolves to a string-compatible leaf on a schema."""
        *prefix, last = path.split(".")
        current_schemas: list[type[BaseModel]] = [schema]

        for segment in prefix:
            next_schemas: list[type[BaseModel]] = []
            for current_schema in current_schemas:
                field: Any = current_schema.model_fields.get(segment)
                if field is None:
                    if current_schema.__name__ == segment:
                        next_schemas.append(current_schema)
                    continue

                resolved_schemas = self._annotation_model_choices(field.annotation)
                if not resolved_schemas:
                    return False
                next_schemas.extend(resolved_schemas)
            if not next_schemas:
                return False
            current_schemas = next_schemas

        return bool(current_schemas) and all(
            self._schema_field_is_string_compatible(current_schema, last) for current_schema in current_schemas
        )

    def _schema_field_is_string_compatible(self, schema: type[BaseModel], field_name: str) -> bool:
        """Return whether a schema field can provide string content to middleware."""
        field: Any = schema.model_fields.get(field_name)
        return field is not None and self._annotation_is_string_compatible(field.annotation)

    def _annotation_model_choices(self, annotation: Any) -> list[type[BaseModel]]:
        """Resolve every concrete model choice represented by an annotation."""
        annotation = self._strip_annotated(annotation)
        origin: Any = typing.get_origin(annotation)

        if origin in (typing.Union, types.UnionType):
            model_choices: list[type[BaseModel]] = []
            for arg in typing.get_args(annotation):
                if arg is type(None):
                    continue
                resolved = self._annotation_model_choices(arg)
                if not resolved:
                    return []
                model_choices.extend(resolved)
            return model_choices

        if origin is list:
            element_args = typing.get_args(annotation)
            if not element_args:
                return []
            return self._annotation_model_choices(element_args[0])

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return [annotation]
        return []

    def _annotation_is_string_compatible(self, annotation: Any) -> bool:
        """Return whether an annotation can expose a string value at runtime."""
        annotation = self._strip_annotated(annotation)
        origin: Any = typing.get_origin(annotation)

        if annotation is str:
            return True
        if origin in (typing.Union, types.UnionType):
            return any(
                arg is not type(None) and self._annotation_is_string_compatible(arg)
                for arg in typing.get_args(annotation)
            )
        if origin is list:
            element_args = typing.get_args(annotation)
            return bool(element_args) and self._annotation_is_string_compatible(element_args[0])
        return False

    def _strip_annotated(self, annotation: Any) -> Any:
        """Return the base annotation under any Annotated metadata."""
        while typing.get_origin(annotation) is typing.Annotated:
            annotation = typing.get_args(annotation)[0]
        return annotation

    def _resolve_guarded_targets(self, name: str) -> list[str]:
        """Expand configured field selections into traversal paths."""
        return self._resolve_guarded_targets_for_phase(name, None)

    def _resolve_guarded_targets_for_phase(self, name: str, phase: str | None) -> list[str]:
        """Expand field selections for the current middleware phase."""
        config = getattr(self, "_config", None) or getattr(self, "_guardrails_config", None)
        if config is None or not isinstance(config.workflow_functions, dict):
            return []

        selection: Any = config.workflow_functions.get(name)
        if selection is None:
            return []

        paths: list[str] = []
        seen: set[str] = set()
        for selection_root in self._selection_roots_for_phase(selection, phase):
            for path in self._expand_selection_root(selection_root):
                if path not in seen:
                    paths.append(path)
                    seen.add(path)
        return paths

    def _selection_roots_for_phase(
        self,
        selection: FunctionFieldSelection,
        phase: str | None,
    ) -> list[dict[str, list[str] | dict[str, list[str]]]]:
        """Return the configured selection for one phase."""
        if isinstance(selection.root, PhaseFieldSelectionConfig):
            if phase is not None:
                return [getattr(selection.root, phase, {})]
            return [getattr(selection.root, field_name) for field_name in type(selection.root).model_fields]
        return [selection.root]

    def _expand_selection_root(self, selection_root: dict[str, list[str] | dict[str, list[str]]]) -> list[str]:
        """Expand one field-selection mapping into traversal paths."""
        paths: list[str] = []
        for field, subpaths in selection_root.items():
            if isinstance(subpaths, dict):
                for model_name, model_subpaths in subpaths.items():
                    paths.extend(
                        [f"{field}.{model_name}"]
                        if not model_subpaths
                        else [f"{field}.{model_name}.{subpath}" for subpath in model_subpaths]
                    )
            else:
                paths.extend([field] if not subpaths else [f"{field}.{subpath}" for subpath in subpaths])
        return paths

    def _iter_targets_at_path(self, value: Any, path: str) -> Iterator[tuple[str, Callable[[str], None]]]:
        """Yield each string reached by a dotted path, including model-member selectors."""
        *prefix, last = path.split(".")
        parents: list[Any] = list(value) if isinstance(value, list) else [value]
        for segment in prefix:
            next_parents: list[Any] = []
            for node in parents:
                attr: Any = getattr(node, segment, None)
                if attr is not None:
                    next_parents.extend(attr if isinstance(attr, list) else [attr])
                elif node.__class__.__name__ == segment:
                    next_parents.append(node)
            parents = next_parents

        for parent in parents:
            leaf: Any = getattr(parent, last, None)
            if isinstance(leaf, str):
                yield leaf, self._set_modified_rail_value(parent, last)
            elif isinstance(leaf, list):
                for index, item in enumerate(leaf):
                    if isinstance(item, str):
                        yield item, self._set_modified_rail_value_in_list(leaf, index)
