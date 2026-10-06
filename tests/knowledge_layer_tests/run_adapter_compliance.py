#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Adapter Compliance Test

This script validates that an adapter correctly implements BaseRetriever/BaseIngestor:
1. Quick mode: Registration and instantiation checks (no external services)
2. Full mode: Complete ingestion and retrieval pipeline

Prerequisites:
    - NVIDIA_API_KEY environment variable set (for full mode)
    - Backend dependencies installed

Usage:
    export NVIDIA_API_KEY=nvapi-your-key

    # Quick mode - registration check only (no files/services needed)
    python tests/knowledge_layer_tests/run_adapter_compliance.py --backend llamaindex --quick
    python tests/knowledge_layer_tests/run_adapter_compliance.py --backend foundational_rag --quick
    python tests/knowledge_layer_tests/run_adapter_compliance.py --backend opensearch --quick
    python tests/knowledge_layer_tests/run_adapter_compliance.py --backend azure_ai_search --quick \
      --config '{"endpoint":"https://example.search.windows.net","start_ttl_cleanup":false}'

    # Full mode - complete ingestion + retrieval test
    python tests/knowledge_layer_tests/run_adapter_compliance.py --backend llamaindex
    python tests/knowledge_layer_tests/run_adapter_compliance.py --backend foundational_rag
    python tests/knowledge_layer_tests/run_adapter_compliance.py --backend opensearch

    export AZURE_SEARCH_ENDPOINT=https://your-service.search.windows.net
    export AZURE_SEARCH_API_KEY=your-search-admin-key
    export DEEP_RESEARCHER_AZURE_SEARCH_INDEX_PREFIX="deep-researcher-${USER}"
    python tests/knowledge_layer_tests/run_adapter_compliance.py --backend azure_ai_search \
      --config '{"start_ttl_cleanup":false}'

    # For managed identity, omit AZURE_SEARCH_API_KEY. Set AZURE_CLIENT_ID for a
    # user-assigned identity. NVIDIA_API_KEY remains required for embeddings.

Exit codes:
    0 - All tests passed
    1 - Some tests failed
    2 - Configuration error
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class TestResult:
    """Result of a single test."""

    name: str
    passed: bool
    message: str
    duration_ms: float = 0.0

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message} ({self.duration_ms:.0f}ms)"


class AdapterComplianceTest:
    """
    Generic test suite for Knowledge Layer adapters.

    Tests all required methods from BaseRetriever and BaseIngestor.
    """

    def __init__(
        self,
        backend: str,
        config: dict[str, Any],
        test_file: str,
        collection_name: str = "compliance_test",
        cleanup: bool = True,
        timeout: int = 120,
        quick: bool = False,
    ):
        self.backend = backend
        self.config = config
        self.test_file = test_file
        self.collection_name = f"{collection_name}_{int(time.time())}"  # Unique per run
        self.cleanup = cleanup
        self.timeout = timeout
        self.quick = quick
        self.results: list[TestResult] = []

        self.ingestor = None
        self.retriever = None
        self.job_id = None

    def _import_backend(self):
        """Import the backend module to trigger registration."""
        backend_imports = {
            "llamaindex": "knowledge_layer.llamaindex",
            "foundational_rag": "knowledge_layer.foundational_rag",
            "opensearch": "knowledge_layer.opensearch",
            "azure_ai_search": "knowledge_layer.azure_ai_search",
        }

        module_name = backend_imports.get(self.backend.lower())
        if not module_name:
            # Try generic import pattern
            module_name = f"knowledge_layer.{self.backend}"

        try:
            __import__(module_name)
            return True, f"Imported {module_name}"
        except ImportError as e:
            return False, f"Failed to import {module_name}: {e}"

    def _run_test(self, name: str, test_fn) -> TestResult:
        """Run a single test and capture result."""
        start = time.time()
        try:
            result = test_fn()
            duration = (time.time() - start) * 1000

            if asyncio.iscoroutine(result):
                result = asyncio.get_event_loop().run_until_complete(result)

            if isinstance(result, tuple):
                passed, message = result
            else:
                passed, message = bool(result), str(result)

            return TestResult(name, passed, message, duration)

        except Exception as e:
            duration = (time.time() - start) * 1000
            return TestResult(name, False, f"Exception: {e}", duration)

    def run_all_tests(self) -> bool:
        """Run all compliance tests."""
        mode = "QUICK (registration only)" if self.quick else "FULL"

        print(f"\n{'=' * 70}")
        print("Knowledge Layer Adapter Compliance Test")
        print(f"{'=' * 70}")
        print(f"Backend: {self.backend}")
        print(f"Mode: {mode}")
        if not self.quick:
            print(f"Config: {json.dumps(self.config, indent=2)}")
            print(f"Test file: {self.test_file}")
            print(f"Collection: {self.collection_name}")
        print(f"{'=' * 70}\n")

        # Phase 1: Setup & Registration (always run)
        print("--- Phase 1: Setup & Registration ---")
        self.results.append(self._run_test("Import backend module", self._import_backend))

        self.results.append(self._run_test("Get ingestor instance", self._test_get_ingestor))

        self.results.append(self._run_test("Get retriever instance", self._test_get_retriever))

        self.results.append(self._run_test("Verify backend_name property", self._test_backend_name))

        # Quick mode stops here
        if self.quick:
            self._print_summary()
            return all(r.passed for r in self.results)

        # Phase 2: Collection Management
        print("\n--- Phase 2: Collection Management ---")
        self.results.append(self._run_test("Create collection", self._test_create_collection))

        self.results.append(self._run_test("Get collection", self._test_get_collection))

        self.results.append(self._run_test("List collections (includes new)", self._test_list_collections))

        # Phase 3: File Management
        print("\n--- Phase 3: File Management ---")
        self.results.append(self._run_test("Upload file", self._test_upload_file))

        self.results.append(self._run_test("Get file status (polling)", self._test_get_file_status))

        self.results.append(self._run_test("List files in collection", self._test_list_files))

        # Phase 4: Retrieval
        print("\n--- Phase 4: Retrieval ---")
        self.results.append(self._run_test("Retrieve documents", self._test_retrieve))

        self.results.append(self._run_test("Verify Chunk schema", self._test_chunk_schema))

        # Phase 5: Cleanup
        print("\n--- Phase 5: Cleanup ---")
        self.results.append(self._run_test("Delete file", self._test_delete_file))

        if self.cleanup:
            self.results.append(self._run_test("Delete collection", self._test_delete_collection))
        else:
            self.results.append(TestResult("Delete collection", True, "Skipped (--keep flag)", 0))

        # Summary
        self._print_summary()

        return all(r.passed for r in self.results)

    # =========================================================================
    # Individual Tests
    # =========================================================================

    def _test_get_ingestor(self):
        from deep_researcher_agent.knowledge.factory import get_ingestor

        self.ingestor = get_ingestor(self.backend, self.config)
        return True, f"Got {type(self.ingestor).__name__}"

    def _test_get_retriever(self):
        from deep_researcher_agent.knowledge.factory import get_retriever

        self.retriever = get_retriever(self.backend, self.config)
        return True, f"Got {type(self.retriever).__name__}"

    def _test_backend_name(self):
        ingestor_name = self.ingestor.backend_name
        retriever_name = self.retriever.backend_name

        if not ingestor_name:
            return False, "Ingestor.backend_name is empty"
        if not retriever_name:
            return False, "Retriever.backend_name is empty"

        return True, f"Ingestor: '{ingestor_name}', Retriever: '{retriever_name}'"

    def _test_create_collection(self):
        from deep_researcher_agent.knowledge.schema import CollectionInfo

        result = self.ingestor.create_collection(name=self.collection_name, description="Compliance test collection")

        if not isinstance(result, CollectionInfo):
            return False, f"Expected CollectionInfo, got {type(result)}"
        if result.name != self.collection_name:
            return False, f"Name mismatch: expected '{self.collection_name}', got '{result.name}'"

        return True, f"Created '{result.name}'"

    def _test_get_collection(self):
        from deep_researcher_agent.knowledge.schema import CollectionInfo

        result = self.ingestor.get_collection(self.collection_name)

        if result is None:
            return False, "Collection not found after creation"
        if not isinstance(result, CollectionInfo):
            return False, f"Expected CollectionInfo, got {type(result)}"

        return True, f"Found collection with {result.chunk_count} chunks"

    def _test_list_collections(self):
        collections = self.ingestor.list_collections()

        if not isinstance(collections, list):
            return False, f"Expected list, got {type(collections)}"

        names = [c.name for c in collections]
        if self.collection_name not in names:
            return False, f"New collection not in list: {names}"

        return True, f"Found {len(collections)} collections"

    def _test_upload_file(self):
        from deep_researcher_agent.knowledge.schema import FileInfo

        if not Path(self.test_file).exists():
            return False, f"Test file not found: {self.test_file}"

        result = self.ingestor.upload_file(file_path=self.test_file, collection_name=self.collection_name)

        # Result can be FileInfo or job_id string
        if isinstance(result, FileInfo):
            self.job_id = result.file_id
            return True, f"Job ID: {self.job_id}"
        elif isinstance(result, str):
            self.job_id = result
            return True, f"Job ID: {self.job_id}"
        else:
            return False, f"Expected FileInfo or str, got {type(result)}"

    def _test_get_file_status(self):
        from deep_researcher_agent.knowledge.schema import FileInfo
        from deep_researcher_agent.knowledge.schema import FileStatus

        if not self.job_id:
            return False, "No job_id from upload_file"

        start = time.time()
        last_status = None

        while time.time() - start < self.timeout:
            status = self.ingestor.get_file_status(self.job_id, self.collection_name)

            if status is None:
                # Some backends return None initially
                time.sleep(2)
                continue

            if not isinstance(status, FileInfo):
                return False, f"Expected FileInfo, got {type(status)}"

            last_status = status
            print(f"    Status: {status.status.value}")

            # Check for terminal states
            if status.status == FileStatus.SUCCESS:
                return True, f"Completed with {status.chunk_count} chunks"
            elif status.status == FileStatus.FAILED:
                return False, f"Ingestion failed: {status.error_message}"

            time.sleep(3)

        return (
            False,
            f"Timeout after {self.timeout}s, last status: {last_status.status.value if last_status else 'None'}",
        )

    def _test_list_files(self):
        from deep_researcher_agent.knowledge.schema import FileInfo

        files = self.ingestor.list_files(self.collection_name)

        if not isinstance(files, list):
            return False, f"Expected list, got {type(files)}"

        if len(files) == 0:
            return False, "No files found after upload"

        # Verify FileInfo schema
        for f in files:
            if not isinstance(f, FileInfo):
                return False, f"Expected FileInfo in list, got {type(f)}"

        filenames = [f.file_name for f in files]
        return True, f"Found {len(files)} files: {filenames}"

    def _test_retrieve(self):
        from deep_researcher_agent.knowledge.schema import RetrievalResult

        async def do_retrieve():
            result = await self.retriever.retrieve(
                query="What is this document about?", collection_name=self.collection_name, top_k=5
            )
            return result

        result = asyncio.get_event_loop().run_until_complete(do_retrieve())

        if not isinstance(result, RetrievalResult):
            return False, f"Expected RetrievalResult, got {type(result)}"

        if len(result.chunks) == 0:
            return False, "No chunks retrieved"

        self._last_retrieval = result
        return True, f"Retrieved {len(result.chunks)} chunks"

    def _test_chunk_schema(self):
        from deep_researcher_agent.knowledge.schema import Chunk
        from deep_researcher_agent.knowledge.schema import ContentType

        if not hasattr(self, "_last_retrieval") or not self._last_retrieval.chunks:
            return False, "No retrieval results to validate"

        chunk = self._last_retrieval.chunks[0]

        if not isinstance(chunk, Chunk):
            return False, f"Expected Chunk, got {type(chunk)}"

        # Validate required fields
        errors = []

        if not chunk.chunk_id:
            errors.append("chunk_id is empty")
        if chunk.content is None:  # Empty string is OK for images
            errors.append("content is None")
        if not isinstance(chunk.content_type, ContentType):
            errors.append(f"content_type not a ContentType enum: {chunk.content_type}")
        if not chunk.file_name:
            errors.append("file_name is empty")
        if not chunk.display_citation:
            errors.append("display_citation is empty (REQUIRED)")
        if not (0.0 <= chunk.score <= 1.0):
            errors.append(f"score out of range [0,1]: {chunk.score}")

        if errors:
            return False, f"Schema violations: {', '.join(errors)}"

        return True, f"Chunk valid: type={chunk.content_type.value}, score={chunk.score:.2f}"

    def _test_delete_file(self):
        files = self.ingestor.list_files(self.collection_name)
        if not files:
            return True, "No files to delete (already empty)"

        file_id = files[0].file_id
        filename = files[0].file_name
        result = self.ingestor.delete_file(file_id, self.collection_name)

        if not result:
            return False, f"delete_file returned False for '{filename}'"

        # Verify deletion
        remaining = self.ingestor.list_files(self.collection_name)
        remaining_names = [f.file_name for f in remaining]

        if filename in remaining_names:
            return False, f"File '{filename}' still exists after deletion"

        return True, f"Deleted '{filename}'"

    def _test_delete_collection(self):
        result = self.ingestor.delete_collection(self.collection_name)

        if not result:
            return False, "delete_collection returned False"

        # Verify deletion
        remaining = self.ingestor.get_collection(self.collection_name)
        if remaining is not None:
            return False, "Collection still exists after deletion"

        return True, f"Deleted '{self.collection_name}'"

    # =========================================================================
    # Summary
    # =========================================================================

    def _print_summary(self):
        print(f"\n{'=' * 70}")
        print("TEST RESULTS")
        print(f"{'=' * 70}")

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name}")
            if not r.passed:
                print(f"         {r.message}")

        print(f"\n{'=' * 70}")
        print(f"Summary: {passed}/{total} passed, {failed} failed")

        if failed == 0:
            print("All tests PASSED - Adapter is compliant!")
        else:
            print("Some tests FAILED - Review implementation")
        print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test Knowledge Layer adapter compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--backend",
        "-b",
        required=True,
        help="Backend name (e.g., llamaindex, foundational_rag, opensearch, azure_ai_search)",
    )

    parser.add_argument("--config", "-c", default="{}", help="Backend config as JSON string (default: {})")

    parser.add_argument(
        "--test-file",
        "-f",
        default=str(Path(__file__).parent / "data" / "multimodal_test.pdf"),
        help="PDF file to use for testing (default: tests/knowledge_layer_tests/data/multimodal_test.pdf)",
    )

    parser.add_argument(
        "--collection", default="compliance_test", help="Base name for test collection (timestamp appended)"
    )

    parser.add_argument("--keep", action="store_true", help="Don't delete test collection after tests")

    parser.add_argument(
        "--timeout", "-t", type=int, default=120, help="Timeout for ingestion polling in seconds (default: 120)"
    )

    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Quick mode: only test registration (import, get_ingestor, get_retriever, backend_name)",
    )

    parser.add_argument("--full", action="store_true", help="Full mode: all tests including file operations (default)")

    args = parser.parse_args()

    # Determine mode (--quick takes precedence)
    quick_mode = args.quick and not args.full

    # Parse config
    try:
        config = json.loads(args.config)
    except json.JSONDecodeError as e:
        print(f"Error parsing config JSON: {e}")
        sys.exit(2)

    # Verify test file exists (only required for full mode)
    if not quick_mode and not Path(args.test_file).exists():
        print(f"Error: Test file not found: {args.test_file}")
        print("Hint: Use --quick to skip file operations, or provide a valid test file.")
        sys.exit(2)

    # Run tests
    tester = AdapterComplianceTest(
        backend=args.backend,
        config=config,
        test_file=args.test_file,
        collection_name=args.collection,
        cleanup=not args.keep,
        timeout=args.timeout,
        quick=quick_mode,
    )

    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
