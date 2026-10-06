#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
LlamaIndex Backend - Collection & File Management API Test

This script tests all abstract methods implemented in the LlamaIndex adapter:
1. Collection Management: create, list, get, delete
2. File Management: upload, list, get_status, delete

Prerequisites:
    - NVIDIA_API_KEY environment variable (via export or .env file)
    - LlamaIndex dependencies: uv pip install -e "sources/knowledge_layer[llamaindex]"

Usage:
    # Option 1: Export env var
    export NVIDIA_API_KEY=nvapi-your-key

    # Option 2: Use .env file (auto-loaded, won't override existing vars)
    echo 'NVIDIA_API_KEY=nvapi-your-key' >> .env

    # Run with default test file
    python tests/knowledge_layer_tests/run_llamaindex_api.py

    # Keep collection after test (don't cleanup)
    python tests/knowledge_layer_tests/run_llamaindex_api.py --no-cleanup

    # Text-only mode (faster, no VLM)
    python tests/knowledge_layer_tests/run_llamaindex_api.py --text-only
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file (won't override existing vars)
load_dotenv()

from deep_researcher_agent.knowledge.factory import get_ingestor  # noqa: E402
from deep_researcher_agent.knowledge.schema import CollectionInfo  # noqa: E402
from deep_researcher_agent.knowledge.schema import FileInfo  # noqa: E402
from deep_researcher_agent.knowledge.schema import FileStatus  # noqa: E402


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(label: str, value, indent: int = 2):
    """Print a formatted result."""
    prefix = " " * indent
    print(f"{prefix}{label}: {value}")


def print_collection_info(col: CollectionInfo, indent: int = 4):
    """Print CollectionInfo details."""
    prefix = " " * indent
    print(f"{prefix}Name: {col.name}")
    print(f"{prefix}Backend: {col.backend}")
    print(f"{prefix}File Count: {col.file_count}")
    print(f"{prefix}Chunk Count: {col.chunk_count}")
    print(f"{prefix}Description: {col.description or '(none)'}")
    if col.metadata:
        print(f"{prefix}Metadata: {col.metadata}")


def print_file_info(file: FileInfo, indent: int = 4):
    """Print FileInfo details."""
    prefix = " " * indent
    print(f"{prefix}File ID: {file.file_id}")
    print(f"{prefix}File Name: {file.file_name}")
    print(f"{prefix}Collection: {file.collection_name}")
    print(f"{prefix}Status: {file.status.value}")
    print(f"{prefix}Chunk Count: {file.chunk_count}")
    if file.file_size:
        print(f"{prefix}File Size: {file.file_size:,} bytes")
    if file.error_message:
        print(f"{prefix}Error: {file.error_message}")
    if file.metadata:
        print(f"{prefix}Metadata: {file.metadata}")


def test_collection_management(ingestor, collection_name: str) -> bool:
    """Test collection CRUD operations."""
    print_header("TEST 1: Collection Management")

    # 1.1 Create Collection
    print("\n[1.1] Creating collection...")
    try:
        col_info = ingestor.create_collection(
            name=collection_name,
            description="Test collection for API validation",
            metadata={"test_run": True, "purpose": "e2e_testing"},
        )
        print_result("Result", "SUCCESS")
        print_collection_info(col_info)

        assert col_info.name == collection_name, "Collection name mismatch"
        assert col_info.backend == "llamaindex", "Backend mismatch"
        print_result("Assertions", "PASSED")
    except Exception as e:
        print_result("Result", f"FAILED - {e}")
        return False

    # 1.2 List Collections
    print("\n[1.2] Listing collections...")
    try:
        collections = ingestor.list_collections()
        print_result("Result", f"SUCCESS - Found {len(collections)} collection(s)")

        found = False
        for col in collections:
            if col.name == collection_name:
                found = True
                print_collection_info(col)
                break

        assert found, f"Created collection '{collection_name}' not found in list"
        print_result("Assertions", "PASSED")
    except Exception as e:
        print_result("Result", f"FAILED - {e}")
        return False

    # 1.3 Get Collection
    print("\n[1.3] Getting collection by name...")
    try:
        col_info = ingestor.get_collection(collection_name)
        print_result("Result", "SUCCESS")

        assert col_info is not None, "Collection not found"
        assert col_info.name == collection_name, "Collection name mismatch"
        print_collection_info(col_info)
        print_result("Assertions", "PASSED")
    except Exception as e:
        print_result("Result", f"FAILED - {e}")
        return False

    # 1.4 Get Non-existent Collection
    print("\n[1.4] Getting non-existent collection...")
    try:
        col_info = ingestor.get_collection("non_existent_collection_xyz")
        print_result("Result", f"SUCCESS - Returned: {col_info}")

        assert col_info is None, "Should return None for non-existent collection"
        print_result("Assertions", "PASSED")
    except Exception as e:
        print_result("Result", f"FAILED - {e}")
        return False

    return True


def test_file_management(ingestor, collection_name: str, test_file: str) -> tuple[bool, str]:
    """Test file CRUD operations. Returns (success, file_id)."""
    print_header("TEST 2: File Management")

    file_id = None

    # 2.1 Upload File
    print("\n[2.1] Uploading file...")
    print_result("File", test_file)
    try:
        file_info = ingestor.upload_file(
            file_path=test_file,
            collection_name=collection_name,
            metadata={"test_file": True, "source": "e2e_test"},
        )
        file_id = file_info.file_id
        print_result("Result", "SUCCESS")
        print_file_info(file_info)

        assert file_info.file_name == Path(test_file).name, "File name mismatch"
        assert file_info.collection_name == collection_name, "Collection name mismatch"
        assert file_info.status in [FileStatus.UPLOADING, FileStatus.INGESTING], (
            f"Unexpected status: {file_info.status}"
        )
        print_result("Assertions", "PASSED")
    except Exception as e:
        print_result("Result", f"FAILED - {e}")
        return False, ""

    # 2.2 Wait for Ingestion & Check Status
    print("\n[2.2] Waiting for ingestion to complete...")
    max_wait = 120  # 2 minutes max
    poll_interval = 2
    elapsed = 0

    while elapsed < max_wait:
        try:
            file_info = ingestor.get_file_status(file_id, collection_name)
            if file_info is None:
                print_result("Status", f"[{elapsed}s] File not found!")
                break

            status = file_info.status.value
            chunks = file_info.chunk_count
            print_result("Status", f"[{elapsed}s] {status} (chunks: {chunks})")

            if file_info.status == FileStatus.SUCCESS:
                print_result("Result", "SUCCESS - Ingestion completed!")
                print_file_info(file_info)
                break
            elif file_info.status == FileStatus.FAILED:
                print_result("Result", f"FAILED - {file_info.error_message}")
                break

            time.sleep(poll_interval)
            elapsed += poll_interval

        except Exception as e:
            print_result("Error", str(e))
            time.sleep(poll_interval)
            elapsed += poll_interval

    if elapsed >= max_wait:
        print_result("Result", "TIMEOUT - Ingestion did not complete in time")
        return False, file_id

    # 2.3 List Files
    print("\n[2.3] Listing files in collection...")
    try:
        files = ingestor.list_files(collection_name)
        print_result("Result", f"SUCCESS - Found {len(files)} file(s)")

        found = False
        for f in files:
            print(f"    - {f.file_name} (id={f.file_id[:8]}..., chunks={f.chunk_count})")
            if f.file_id == file_id:
                found = True

        assert found, "Uploaded file not found in list"
        print_result("Assertions", "PASSED")
    except Exception as e:
        print_result("Result", f"FAILED - {e}")
        return False, file_id

    # 2.4 Get File Status (final)
    print("\n[2.4] Getting final file status...")
    try:
        file_info = ingestor.get_file_status(file_id, collection_name)
        print_result("Result", "SUCCESS")
        print_file_info(file_info)

        assert file_info is not None, "File not found"
        assert file_info.status == FileStatus.SUCCESS, f"Expected SUCCESS, got {file_info.status}"
        assert file_info.chunk_count > 0, "Expected chunks to be created"
        print_result("Assertions", "PASSED")
    except Exception as e:
        print_result("Result", f"FAILED - {e}")
        return False, file_id

    return True, file_id


def test_file_deletion(ingestor, collection_name: str, file_id: str) -> bool:
    """Test file deletion."""
    print_header("TEST 3: File Deletion")

    # 3.1 Delete File
    print("\n[3.1] Deleting file...")
    print_result("File ID", file_id)
    try:
        success = ingestor.delete_file(file_id, collection_name)
        print_result("Result", "SUCCESS" if success else "FAILED")

        assert success, "File deletion failed"
        print_result("Assertions", "PASSED")
    except Exception as e:
        print_result("Result", f"FAILED - {e}")
        return False

    # 3.2 Verify File Deleted
    print("\n[3.2] Verifying file was deleted...")
    try:
        file_info = ingestor.get_file_status(file_id, collection_name)
        print_result("Result", f"File status: {file_info}")

        # File should either be None or not in the list
        files = ingestor.list_files(collection_name)
        found = any(f.file_id == file_id for f in files)

        assert not found, "File still found in collection after deletion"
        print_result("Assertions", "PASSED - File successfully removed")
    except Exception as e:
        print_result("Result", f"FAILED - {e}")
        return False

    return True


def test_collection_deletion(ingestor, collection_name: str) -> bool:
    """Test collection deletion."""
    print_header("TEST 4: Collection Deletion")

    # 4.1 Delete Collection
    print("\n[4.1] Deleting collection...")
    print_result("Collection", collection_name)
    try:
        success = ingestor.delete_collection(collection_name)
        print_result("Result", "SUCCESS" if success else "FAILED")

        assert success, "Collection deletion failed"
        print_result("Assertions", "PASSED")
    except Exception as e:
        print_result("Result", f"FAILED - {e}")
        return False

    # 4.2 Verify Collection Deleted
    print("\n[4.2] Verifying collection was deleted...")
    try:
        col_info = ingestor.get_collection(collection_name)
        print_result("Result", f"Collection info: {col_info}")

        assert col_info is None, "Collection still exists after deletion"
        print_result("Assertions", "PASSED - Collection successfully removed")
    except Exception as e:
        print_result("Result", f"FAILED - {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="End-to-end test for Knowledge Layer Collection & File Management API")
    parser.add_argument(
        "--file",
        "-f",
        default=str(Path(__file__).parent / "data" / "multimodal_test.pdf"),
        help="Path to test PDF file (default: tests/knowledge_layer_tests/data/multimodal_test.pdf)",
    )
    parser.add_argument(
        "--collection",
        "-c",
        default="test_collection",
        help="Collection name for testing (default: test_collection)",
    )
    parser.add_argument(
        "--chroma-dir",
        default="/tmp/chroma_e2e_test",
        help="ChromaDB persistence directory (default: /tmp/chroma_e2e_test)",
    )
    parser.add_argument("--no-cleanup", action="store_true", help="Don't delete collection after test")
    parser.add_argument("--text-only", action="store_true", help="Skip VLM-based extraction (faster, text only)")
    args = parser.parse_args()

    # Validate environment
    if not os.environ.get("NVIDIA_API_KEY"):
        print("ERROR: NVIDIA_API_KEY environment variable not set")
        print("Export your key: export NVIDIA_API_KEY='nvapi-...'")
        sys.exit(1)

    # Validate test file
    test_file = Path(args.file)
    if not test_file.exists():
        print(f"ERROR: Test file not found: {test_file}")
        sys.exit(1)

    print("=" * 70)
    print("  Knowledge Layer - Collection & File API End-to-End Test")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Test File:    {test_file}")
    print(f"  Collection:   {args.collection}")
    print(f"  ChromaDB Dir: {args.chroma_dir}")
    print(f"  Text Only:    {args.text_only}")
    print(f"  Cleanup:      {not args.no_cleanup}")

    # Initialize ingestor
    print("\nInitializing LlamaIndex ingestor...")
    config = {
        "persist_dir": args.chroma_dir,
        "extract_tables": not args.text_only,
        "extract_images": not args.text_only,
        "extract_charts": not args.text_only,
    }

    try:
        # Import adapter to register it
        import knowledge_layer.llamaindex.adapter  # noqa: F401

        ingestor = get_ingestor("llamaindex", config)
        print(f"  Backend: {ingestor.backend_name}")
        print(f"  Persist Dir: {ingestor.persist_dir}")
    except Exception as e:
        print(f"ERROR: Failed to initialize ingestor: {e}")
        sys.exit(1)

    # Run tests
    results = {}
    file_id = None

    # Test 1: Collection Management
    results["collection_management"] = test_collection_management(ingestor, args.collection)

    # Test 2: File Management (only if Test 1 passed)
    if results["collection_management"]:
        results["file_management"], file_id = test_file_management(ingestor, args.collection, str(test_file))
    else:
        results["file_management"] = False
        print("\nSkipping file management tests (collection management failed)")

    # Test 3: File Deletion (only if we have a file_id)
    if file_id and results["file_management"]:
        results["file_deletion"] = test_file_deletion(ingestor, args.collection, file_id)
    else:
        results["file_deletion"] = False
        if results["file_management"]:
            print("\nSkipping file deletion test (no file_id)")

    # Test 4: Collection Deletion (cleanup)
    if not args.no_cleanup:
        results["collection_deletion"] = test_collection_deletion(ingestor, args.collection)
    else:
        results["collection_deletion"] = True  # Skipped
        print("\n[Skipping collection deletion - --no-cleanup flag set]")

    # Summary
    print_header("TEST SUMMARY")

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for test_name, passed_test in results.items():
        status = "PASSED" if passed_test else "FAILED"
        print(f"  {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests PASSED!")
        sys.exit(0)
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
