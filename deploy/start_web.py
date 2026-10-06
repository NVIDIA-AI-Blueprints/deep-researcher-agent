# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Custom web server startup script for the NVIDIA NeMo Agent toolkit
research assistant.

=============================================================================
WHY THIS FILE EXISTS - THE ASYNCIO/ANYIO EVENT LOOP CONFLICT
=============================================================================

The standard NAT CLI command `nat serve` uses asyncio.run() internally to start
the FastAPI server. This creates a fundamental conflict with how anyio (used by
FastAPI/Starlette) manages event loops:

    Problem Chain:
    1. `nat serve` calls asyncio.run() which creates a NEW event loop
    2. FastAPI/Starlette uses anyio for async operations
    3. anyio detects there's already a running loop and refuses to create
       another
    4. Result: "RuntimeError: This event loop is already running" or
               "Runner.run() cannot be called from a running event loop"

    The Error You'd See:
    ```
    RuntimeError: asyncio.run() cannot be called from a running event loop
    ```
    or
    ```
    RuntimeError: Runner.run() cannot be called from a running event loop
    ```

=============================================================================
THE SOLUTION
=============================================================================

Instead of using `nat serve`, we bypass the asyncio.run() wrapper by:

1. Loading the NAT configuration ourselves using the same loader NAT uses
2. Setting up the required environment variables that NAT's FastAPI app expects
3. Running uvicorn DIRECTLY with loop="asyncio"

This allows uvicorn to manage its own event loop without conflict.

Key insight: uvicorn's `loop="asyncio"` parameter tells it to create and manage
the event loop itself, rather than expecting one to already exist or creating
nested loops.

=============================================================================
ENVIRONMENT VARIABLES
=============================================================================

This script reads:
- CONFIG_FILE: Path to the NAT config YAML (default:
  /app/configs/config_web_frag.yml)
- HOST: Bind address (default: 0.0.0.0)
- PORT: Bind port (default: 8000)

This script sets (for NAT's FastAPI app):
- NAT_CONFIG_FILE: Path to the config file (passed directly to NAT)
- NAT_FRONT_END_WORKER: The worker class name for the FastAPI frontend

=============================================================================
ALTERNATIVE APPROACHES CONSIDERED
=============================================================================

1. nest_asyncio: Patches asyncio to allow nested event loops
   - Rejected: Can cause subtle bugs and is not recommended for production

2. Running nat serve in a subprocess
   - Rejected: Adds complexity, harder to manage signals/shutdown

3. Modifying NAT source code
   - Rejected: We want to use NAT as a dependency, not fork it

4. Using uvicorn programmatically (THIS APPROACH)
   - Chosen: Clean, no patches, uvicorn handles the loop correctly

=============================================================================
USAGE
=============================================================================

This script is the main entry point for the Docker container:

    # In Dockerfile
    CMD ["python", "/app/deploy/start_web.py"]

Or run directly for local development:

    export CONFIG_FILE=/app/configs/config_web_frag.yml
    python deploy/start_web.py

=============================================================================
"""

import logging
import os
import sys
import warnings

# Suppress warnings unless PYTHONWARNINGS is explicitly set
if not os.environ.get("PYTHONWARNINGS"):
    warnings.filterwarnings("ignore")


def configure_logging():
    """Configure logging to match nat serve behavior."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=("%(asctime)s - %(levelname)-8s - %(name)s:%(lineno)d - %(message)s"),
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    # Set specific loggers to appropriate levels
    # These match NAT's default logging configuration
    logger_names = ("nat", "deep_researcher_agent", "deep_researcher_api", "knowledge_layer")
    for logger_name in logger_names:
        logging.getLogger(logger_name).setLevel(getattr(logging, log_level, logging.INFO))

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    return log_level


def main():
    """
    Main entry point for the web server.

    This function:
    1. Reads configuration from environment variables
    2. Loads and validates the NAT config file
    3. Sets up environment variables required by NAT's FastAPI app
    4. Starts uvicorn directly (bypassing nat serve's asyncio.run wrapper)
    """
    # Configure logging first (before any imports that might log)
    log_level = configure_logging()

    # Read configuration from environment (set by Docker or defaults)
    config_file = os.environ.get(
        "CONFIG_FILE",
        "/app/configs/config_web_frag.yml",
    )
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    print("=" * 60)
    print("NeMo Agent toolkit research assistant - Web Server")
    print("=" * 60)
    print(f"Config: {config_file}")
    print(f"Host: {host}:{port}")
    print(f"Log Level: {log_level}")
    print("Using custom startup script to avoid asyncio conflicts")
    print("=" * 60)
    print()

    # -------------------------------------------------------------------------
    # STEP 1: Load and validate the NAT configuration
    # -------------------------------------------------------------------------
    # NAT's load_config() reads the YAML file and returns a validated Config
    # object using Pydantic. This ensures the config is valid before we start.
    from nat.runtime.loader import load_config

    print("Loading configuration...")
    config = load_config(config_file)
    print("✓ Configuration loaded and validated")

    # -------------------------------------------------------------------------
    # STEP 2: Set NAT_CONFIG_FILE for NAT's FastAPI app
    # -------------------------------------------------------------------------
    # NAT's FastAPI app (nat.front_ends.fastapi.main:get_app) expects the
    # config to be available via the NAT_CONFIG_FILE environment variable.
    #
    # We pass the original config file path since load_config() has already
    # validated it. NAT will re-load it internally when the app starts.
    print("Setting NAT_CONFIG_FILE...")
    os.environ["NAT_CONFIG_FILE"] = config_file
    print(f"✓ NAT_CONFIG_FILE set to {config_file}")

    # -------------------------------------------------------------------------
    # STEP 3: Set up the worker class for NAT's FastAPI frontend
    # -------------------------------------------------------------------------
    # NAT's FastAPI app needs to know which worker class to use for handling
    # requests. This is normally set by nat serve, but since we're bypassing
    # that, we need to set it ourselves.
    #
    # IMPORTANT: We need to use the runner_class from the config file, not the
    # default NAT worker. The config specifies deep_researcher_async.plugin.AsyncAPIWorker
    # which provides async job infrastructure with SSE streaming.
    print("Setting up FastAPI worker...")
    from nat.front_ends.fastapi.utils import get_class_name

    # Get the runner_class from the loaded config
    try:
        runner_class_name = getattr(
            config.general.front_end,
            "runner_class",
            None,
        )
        if runner_class_name:
            # The runner_class in config is already a string like
            # "deep_researcher_async.plugin.AsyncAPIWorker".
            os.environ["NAT_FRONT_END_WORKER"] = runner_class_name
            print(f"✓ Worker class (from config): {runner_class_name}")
        else:
            # Fallback to default NAT worker if not specified in config
            from nat.front_ends.fastapi.fastapi_front_end_plugin import FastApiFrontEndPlugin

            worker_class = FastApiFrontEndPlugin.get_worker_class(None)
            os.environ["NAT_FRONT_END_WORKER"] = get_class_name(worker_class)
            print(f"✓ Worker class (default): {os.environ['NAT_FRONT_END_WORKER']}")
    except AttributeError as e:
        # If config structure is different, fall back to default
        print(f"⚠ Could not read runner_class from config: {e}")
        print("  Using default NAT worker")
        from nat.front_ends.fastapi.fastapi_front_end_plugin import FastApiFrontEndPlugin

        worker_class = FastApiFrontEndPlugin.get_worker_class(None)
        os.environ["NAT_FRONT_END_WORKER"] = get_class_name(worker_class)
        print(f"✓ Worker class (default): {os.environ['NAT_FRONT_END_WORKER']}")

    # -------------------------------------------------------------------------
    # STEP 4: Run uvicorn directly
    # -------------------------------------------------------------------------
    # This is the key part that avoids the asyncio conflict.
    #
    # uvicorn.run() parameters:
    # - "nat.front_ends.fastapi.main:get_app": The ASGI app factory path
    # - factory=True: Tells uvicorn that get_app is a factory function
    # - loop="asyncio": Let uvicorn create and manage the event loop
    #
    # By using loop="asyncio", uvicorn creates a fresh event loop without
    # conflicting with any existing loops. This is the standard way to run
    # ASGI apps and avoids the nested loop issues that nat serve causes.
    print()
    print("Starting uvicorn server...")
    print(f"Server will be available at: http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"Health Check: http://{host}:{port}/health")
    print()

    import uvicorn

    uvicorn.run(
        "nat.front_ends.fastapi.main:get_app",
        host=host,
        port=port,
        factory=True,
        loop="asyncio",
    )


if __name__ == "__main__":
    main()
