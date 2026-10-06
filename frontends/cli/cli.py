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

import argparse
import asyncio
import logging
import os
import re
import sys
import uuid
import warnings
from pathlib import Path

import nemo_relay
import yaml
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from nat.builder.context import ContextState
from nat.data_models.interactive import HumanPromptModelType
from nat.data_models.interactive import HumanResponse
from nat.data_models.interactive import HumanResponseText
from nat.data_models.interactive import InteractionPrompt
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepType
from nat.runtime.loader import load_workflow
from nat.runtime.session import SessionManager

# Suppress warnings by default; re-enabled in main() if --verbose is passed
if not os.environ.get("PYTHONWARNINGS"):
    warnings.filterwarnings("ignore")
logging.getLogger("nat.builder.function_info").setLevel(logging.ERROR)
logging.getLogger("langgraph.checkpoint").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
console = Console()

# Setup prompt_toolkit with persistent history for arrow key navigation
_history_file = Path.home() / ".deep-researcher" / "cli_history"
_history_file.parent.mkdir(parents=True, exist_ok=True)
prompt_session: PromptSession[str] = PromptSession(history=FileHistory(str(_history_file)))

_ANSI_ESCAPE = re.compile(r"\033\[[0-9;]*m")
_BOLD_MAGENTA = "\033[1;35m"
_BOLD_CYAN = "\033[1;36m"
_RESET = "\033[0m"

# Keep references to the real stdout/stderr before anything can redirect them
_real_stdout = sys.stdout
_real_stderr = sys.stderr


class _BlankLineFilter:
    """Wraps a file object and suppresses writes that are only whitespace.

    Errors, warnings, and any real text pass through unchanged.
    Only bare newlines / blank lines are suppressed.
    """

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, s):
        if s.strip():
            return self._wrapped.write(s)
        return len(s)

    def flush(self):
        self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


class _StderrTracker:
    """Wraps stderr and counts newlines so the spinner can clean up."""

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self.newlines = 0

    def write(self, s):
        self.newlines += s.count("\n")
        return self._wrapped.write(s)

    def flush(self):
        self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


class _Spinner:
    """Async spinner that writes directly to the real stdout.

    Uses \\r to overwrite a single line — no Rich Live display, so no ANSI
    cursor-control artifacts on stop/start.  While running, installs a
    blank-line filter on sys.stdout and a tracker on sys.stderr.  If
    anything writes to stderr (e.g. log warnings), the spinner detects
    the newlines and clears orphaned spinner text from above.
    """

    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, text: str = ""):
        self._text = text
        self._task: asyncio.Task | None = None
        self._idx = 0
        self._last_visible_len = 0
        self._stderr_tracker: _StderrTracker | None = None

    def update(self, text: str) -> None:
        self._text = text

    def start(self) -> None:
        sys.stdout = _BlankLineFilter(_real_stdout)
        self._stderr_tracker = _StderrTracker(_real_stderr)
        sys.stderr = self._stderr_tracker
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self._spin())

    def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
        # Clear the spinner line
        _real_stdout.write(f"\r{' ' * self._last_visible_len}\r")
        _real_stdout.flush()
        # Restore stdout and stderr
        sys.stdout = _real_stdout
        sys.stderr = _real_stderr
        self._stderr_tracker = None

    async def _spin(self) -> None:
        try:
            while True:
                # If stderr output pushed the cursor down, go back up
                # and erase the orphaned spinner text
                if self._stderr_tracker and self._stderr_tracker.newlines > 0:
                    n = self._stderr_tracker.newlines
                    self._stderr_tracker.newlines = 0
                    # Move up n lines, clear the old spinner line, move back down
                    _real_stdout.write(f"\033[{n}A\033[2K\033[{n}B")

                frame = self._FRAMES[self._idx % len(self._FRAMES)]
                line = f"{frame} {self._text}"
                visible_len = len(_ANSI_ESCAPE.sub("", line))
                pad = max(0, self._last_visible_len - visible_len)
                _real_stdout.write(f"\033[2K\r{line}{' ' * pad}")
                _real_stdout.flush()
                self._last_visible_len = visible_len
                self._idx += 1
                await asyncio.sleep(0.08)
        except asyncio.CancelledError:
            pass


# Tracks the active spinner so the HITL callback can pause/resume it
_active_spinner: _Spinner | None = None

ASCII_DEEP_RESEARCHER = r"""
              █████╗  ██╗         ██████╗
             ██╔══██╗ ██║        ██╔═══██╗
             ███████║ ██║ █████║ ██║   ██║
             ██╔══██║ ██║        ██║▄▄ ██║
             ██║  ██║ ██║        ╚██████╔╝
             ╚═╝  ╚═╝ ╚═╝         ╚══▀▀═╝
   NVIDIA Deep Researcher Agent Blueprint powered by NeMo Agent Toolkit
"""


def parse_and_display_response(response: str, verbose: bool = False) -> None:
    """
    Parse and display the response with colored output.
    Handles both ReAct-style formatting and <think> tags.

    Args:
        response: Complete response string
        verbose: Whether to show all sections or just final answer
    """
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    response_without_think = think_pattern.sub("", response).strip()

    if response_without_think:
        console.print()
        console.print(
            Panel(
                Markdown(response_without_think),
                title="[bold bright_white]✨ Answer[/bold bright_white]",
                border_style="bright_white",
            )
        )


async def cli_user_input_callback(prompt: InteractionPrompt) -> HumanResponse:
    """
    NAT-native HITL callback handler for CLI mode.

    This function follows NAT's console front-end pattern for handling
    Human-in-the-Loop interactions in terminal environments.

    Args:
        prompt: The HITL interaction prompt from the workflow

    Returns:
        HumanResponse with the user's input

    Raises:
        ValueError: If the prompt type is not supported in CLI mode
    """
    if prompt.content.input_type == HumanPromptModelType.TEXT:
        was_spinning = _active_spinner is not None
        if was_spinning:
            _active_spinner.stop()

        console.print(
            Panel(
                Markdown(prompt.content.text), title="[bold yellow]🤔 Input Needed[/bold yellow]", border_style="yellow"
            )
        )

        placeholder = getattr(prompt.content, "placeholder", None) or "Enter your response..."
        user_response = (
            await prompt_session.prompt_async(HTML(f"<b><ansicyan>Your Response:</ansicyan></b> ({placeholder}) "))
        ).strip()
        console.print()

        if was_spinning:
            _active_spinner.start()

        return HumanResponseText(text=user_response)

    raise ValueError(
        f"Unsupported human prompt input type: {prompt.content.input_type}. "
        "The CLI only supports 'HumanPromptText' input type. "
        "Please use 'nat serve' for full HITL support."
    )


def _check_interactive_auth(config_file: str) -> bool:
    """Check if interactive_auth is enabled in the config file."""
    try:
        config_path = Path(config_file)
        if not config_path.exists():
            return False

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        workflow_config = config.get("workflow", {})
        return workflow_config.get("interactive_auth", False)
    except Exception as e:
        logging.debug(f"Failed to read config for interactive_auth: {e}")
        return False


async def _initialize_auth() -> None:
    """Initialize authentication for CLI mode."""
    try:
        from deep_researcher_research_cli.auth import initialize_workflow_auth

        console.print("[dim]Initializing authentication...[/dim]")
        await initialize_workflow_auth(interactive_auth=True)
        console.print("[green]✓ Authentication successful[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠ Authentication failed: {e}[/yellow]")
        console.print("[dim]Continuing without authentication...[/dim]")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="deep-researcher-agent",
        description="Deep Researcher Agent Blueprint - Interactive CLI for research and data analysis",
    )
    parser.add_argument(
        "--config_file",
        default="configs/config_cli.yml",
        help="Path to NAT workflow config file (default: configs/config_cli.yml)",
    )
    parser.add_argument(
        "--env_file",
        default="deploy/.env",
        help="Path to .env file containing API keys (default: deploy/.env)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


async def interactive_loop(session_manager: SessionManager, verbose: bool = False):
    """Run the interactive chat loop.

    Args:
        session_manager: NAT session manager instance.
        verbose: Whether to show verbose output (all tool calls and thinking).
    """
    console.print(ASCII_DEEP_RESEARCHER, style="bold cyan")
    console.print("Deep Researcher Agent initialized!", style="bold green")
    console.print("Type 'exit', 'quit', or 'q' to quit.", style="dim")
    if verbose:
        console.print("Verbose mode: ON - Showing all tool calls and thinking", style="bold yellow")
    else:
        console.print("Verbose mode: OFF - Showing final answers only (use -v for verbose)", style="dim")
    console.print()

    cli_session_id = str(uuid.uuid4())
    console.print(f"Session ID: {cli_session_id}", style="dim")

    async with session_manager.session(user_input_callback=cli_user_input_callback) as session:
        while True:
            try:
                user_input = (await prompt_session.prompt_async(HTML("<b><ansiblue>You:</ansiblue></b> "))).strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    console.print("\n[bold green]Goodbye! Happy researching![/bold green]")
                    break

                if not user_input:
                    continue

                console.print()
                if verbose:
                    console.print("[bold magenta]Assistant (Verbose Mode):[/bold magenta]")
                    console.print()

                context_state = ContextState.get()
                context_state.conversation_id.set(cli_session_id)

                async with session.run(user_input) as runner:
                    if not verbose:
                        global _active_spinner
                        prefix = f"{_BOLD_MAGENTA}Assistant:{_RESET}"
                        thinking = f"{_BOLD_CYAN}Thinking...{_RESET}"
                        spinner = _Spinner(f"{prefix} {thinking}")
                        _active_spinner = spinner
                        spinner.start()

                        def _on_step(step: IntermediateStep) -> None:
                            if step.event_type == IntermediateStepType.TOOL_START and step.name:
                                tool_label = f"{_BOLD_CYAN}Using tool:{_RESET} {step.name}"
                                spinner.update(f"{prefix} {tool_label}")
                            elif step.event_type == IntermediateStepType.TOOL_END:
                                spinner.update(f"{prefix} {thinking}")

                        subscription = runner.context.intermediate_step_manager.subscribe(on_next=_on_step)
                        try:
                            result = await runner.result(to_type=str)
                        finally:
                            subscription.unsubscribe()
                            spinner.stop()
                            _active_spinner = None
                    else:
                        result = await runner.result(to_type=str)

                # Relay subscriber delivery is asynchronous. Wait until the run context has
                # closed its outer scopes before displaying the answer and opening the next
                # prompt, otherwise late lifecycle logs can be painted after ``You:``.
                try:
                    await nemo_relay.subscribers.flush_async()
                except Exception as error:  # noqa: BLE001 - telemetry must not suppress the answer
                    logger.warning("NeMo Relay flush failed (error_type=%s)", type(error).__name__)

                parse_and_display_response(result, verbose=verbose)

                # Check if the response indicates a critical error (e.g., missing API key)
                # This is a fallback in case validation didn't catch it earlier
                if "Missing Required API Keys" in result or "Missing keys:" in result:
                    console.print("[bold red]Cannot continue without required API keys. Exiting.[/bold red]")
                    break

            except (EOFError, KeyboardInterrupt):
                console.print("\n\n[bold green]Goodbye! Happy researching![/bold green]")
                break
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")


def main() -> None:
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s", handlers=[logging.StreamHandler()]
        )
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(name)s - %(message)s")

    env_file = Path(args.env_file)
    if env_file.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path=env_file)
        except ImportError:
            print("Warning: python-dotenv not installed. Environment variables must be set manually.")
        except Exception as e:
            print(f"Warning: Failed to load .env file: {e}")

    # Validate LLM API keys based on config
    try:
        config_path = Path(args.config_file)
        if config_path.exists():
            import yaml

            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            from deep_researcher_agent.common.config_validation import validate_llm_configs

            is_valid, missing_keys = validate_llm_configs(config)
            if not is_valid:
                console.print(
                    f"\n[bold red]❌ Error: Missing required API keys ({', '.join(missing_keys)})[/bold red]\n"
                    "[yellow]Please set these keys in your .env file or environment variables "
                    "and restart the application.[/yellow]\n"
                )
                os._exit(1)
    except Exception as e:
        logger.debug(f"Failed to validate LLM config: {e}")

    interactive_auth = _check_interactive_auth(args.config_file)

    async def _run():
        if interactive_auth:
            await _initialize_auth()

        async with load_workflow(args.config_file) as session_manager:
            await interactive_loop(session_manager, verbose=args.verbose)

    try:
        asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_run())
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
