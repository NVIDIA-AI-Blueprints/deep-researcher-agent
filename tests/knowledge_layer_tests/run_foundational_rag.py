#!/usr/bin/env python3
"""
Foundational RAG Backend - End-to-End Test

This script tests the complete knowledge layer workflow with Foundational RAG:
1. Create a collection
2. Upload PDFs and poll for ingestion status
3. Test retrieval via semantic search
4. Cleanup (optional)

Prerequisites:
    - Deploy the NVIDIA RAG Blueprint first:
      https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/deploy-docker-self-hosted.md
    - Set environment variables for server URLs (via export or .env file)

Usage:
    # Option 1: Export env vars
    export RAG_SERVER_URL=http://your-server:8081/v1
    export RAG_INGEST_URL=http://your-server:8082/v1

    # Option 2: Use .env file (auto-loaded, won't override existing vars)
    echo 'RAG_SERVER_URL=http://your-server:8081/v1' >> .env
    echo 'RAG_INGEST_URL=http://your-server:8082/v1' >> .env

    # Run with default test files (all PDFs in data/)
    python tests/knowledge_layer_tests/run_foundational_rag.py

    # Run with custom data directory
    python tests/knowledge_layer_tests/run_foundational_rag.py --data-dir /path/to/pdfs

    # Delete collection after test (default: keep)
    python tests/knowledge_layer_tests/run_foundational_rag.py --delete

    # Skip retrieval test
    python tests/knowledge_layer_tests/run_foundational_rag.py --skip-retrieval

    # Generate client-side summaries during upload
    python tests/knowledge_layer_tests/run_foundational_rag.py --generate-summary

NVIDIA RAG Blueprint has TWO servers:
    - Ingestion Server (port 8082): /documents, /collections, /status
    - Query Server (port 8081): /search (retrieval)
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file (won't override existing vars)
load_dotenv()

# Import adapter to register it
import knowledge_layer.foundational_rag.adapter  # noqa: E402, F401

from deep_researcher_agent.knowledge.factory import get_ingestor  # noqa: E402
from deep_researcher_agent.knowledge.factory import get_retriever  # noqa: E402
from deep_researcher_agent.knowledge.schema import FileStatus  # noqa: E402

# Polling configuration
POLL_INTERVAL_SECONDS = 5
MAX_POLL_DURATION_SECONDS = 300  # 5 minutes


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end test for RAG Server adapter")
    parser.add_argument(
        "--ingest-url",
        default=os.environ.get("RAG_INGEST_URL", "http://localhost:8082/v1"),
        help="RAG Ingestion server URL (port 8082)",
    )
    parser.add_argument(
        "--query-url",
        default=os.environ.get("RAG_SERVER_URL", "http://localhost:8081/v1"),
        help="RAG Query server URL (port 8081)",
    )
    parser.add_argument(
        "--collection",
        default="test_collection",
        help="Collection name (default: test_collection)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).parent / "data"),
        help="Directory containing test files",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete collection after test (default: keep)",
    )
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Skip retrieval test",
    )
    parser.add_argument(
        "--generate-summary",
        "-s",
        action="store_true",
        help="Generate client-side one-sentence summary for each PDF during upload",
    )
    return parser.parse_args()


def print_header(msg: str):
    print(f"\n{'=' * 70}")
    print(f"  {msg}")
    print(f"{'=' * 70}")


def print_step(step: int, msg: str):
    print(f"\n[Step {step}] {msg}")
    print("-" * 50)


async def main():
    args = parse_args()

    print_header("Foundational RAG Adapter - End-to-End Test")
    print(f"Ingest URL: {args.ingest_url}")
    print(f"Query URL:  {args.query_url}")
    print(f"Collection: {args.collection}")
    print(f"Data dir: {args.data_dir}")
    print(f"Generate summary: {args.generate_summary}")

    # Initialize adapters with separate URLs
    ingest_config = {
        "rag_url": args.ingest_url,  # Ingestor uses port 8082
        "timeout": 300,
        "chunk_size": 512,
        "chunk_overlap": 150,
        "generate_summary": args.generate_summary,
    }

    query_config = {
        "rag_url": args.query_url,  # Retriever uses port 8081
        "timeout": 60,
    }

    ingestor = get_ingestor("foundational_rag", ingest_config)
    retriever = get_retriever("foundational_rag", query_config)

    # =========================================================================
    # Step 1: Health Check
    # =========================================================================
    print_step(1, "Health Check")

    healthy = await ingestor.health_check()
    print(f"RAG Server healthy: {healthy}")

    if not healthy:
        print("ERROR: RAG server is not healthy. Aborting.")
        sys.exit(1)

    # =========================================================================
    # Step 2: Create Collection
    # =========================================================================
    print_step(2, f"Create Collection: {args.collection}")

    try:
        collection = ingestor.create_collection(
            name=args.collection,
            description="End-to-end test collection",
            metadata={"embedding_dimension": 2048},
        )
        print(f"✓ Created collection: {collection.name}")
    except Exception as e:
        print(f"Collection creation: {e}")
        # Collection might already exist, continue

    # =========================================================================
    # Step 3: Upload Files
    # =========================================================================
    print_step(3, "Upload Files")

    data_dir = Path(args.data_dir)
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF files to upload:")
    for f in pdf_files:
        print(f"  - {f.name}")

    uploaded_files = []
    task_ids = []

    for pdf_file in pdf_files:
        print(f"\nUploading: {pdf_file.name}...")
        try:
            file_info = ingestor.upload_file(
                file_path=str(pdf_file),
                collection_name=args.collection,
            )
            uploaded_files.append(file_info)
            task_id = file_info.metadata.get("task_id", "")
            if task_id:
                task_ids.append(task_id)
            print(f"  ✓ Uploaded, task_id: {task_id}")
            print(f"    Status: {file_info.status.value}")

            # Show client-side summary if generated
            summary = file_info.metadata.get("summary")
            if summary:
                print(f"    Summary: {summary}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # =========================================================================
    # Step 4: Poll for Ingestion Status
    # =========================================================================
    print_step(4, "Poll Ingestion Status")

    if task_ids:
        print(f"Polling {len(task_ids)} task(s) every {POLL_INTERVAL_SECONDS} seconds...")
        print("(RAG server uses Celery states: PENDING → STARTED → SUCCESS/FAILURE)\n")

        max_polls = MAX_POLL_DURATION_SECONDS // POLL_INTERVAL_SECONDS
        poll_count = 0
        poll_interval = POLL_INTERVAL_SECONDS

        # Track task completion
        pending_tasks = set(task_ids)
        completed_tasks = {}  # task_id -> final status

        while poll_count < max_polls and pending_tasks:
            print(f"Poll {poll_count + 1}/{max_polls}:")

            for task_id in list(pending_tasks):
                status = ingestor.get_file_status(task_id, args.collection)
                if status:
                    # Get raw Celery state from metadata
                    raw_state = status.metadata.get("raw_state", "UNKNOWN")
                    file_status = status.status.value

                    # Check if terminal state
                    if status.status in (FileStatus.SUCCESS, FileStatus.FAILED):
                        pending_tasks.discard(task_id)
                        completed_tasks[task_id] = status
                        print(f"  ✓ {task_id[:24]}... : {file_status} (raw: {raw_state})")
                    else:
                        print(f"  ⏳ {task_id[:24]}... : {file_status} (raw: {raw_state})")
                else:
                    print(f"  ? {task_id[:24]}... : No status returned")

            if not pending_tasks:
                print(f"\n✓ All {len(task_ids)} files processed!")
                break

            poll_count += 1
            remaining = len(pending_tasks)
            print(f"\n  {remaining} task(s) still processing, waiting {poll_interval}s...")
            await asyncio.sleep(poll_interval)

        if pending_tasks:
            print(f"\n⚠ Polling timeout - {len(pending_tasks)} file(s) may still be processing")

        # Summary of completed tasks
        if completed_tasks:
            success_count = sum(1 for s in completed_tasks.values() if s.status == FileStatus.SUCCESS)
            failed_count = sum(1 for s in completed_tasks.values() if s.status == FileStatus.FAILED)
            print(f"\nCompleted: {success_count} success, {failed_count} failed")
    else:
        print("No task IDs to poll - waiting 10 seconds for server processing...")
        await asyncio.sleep(10)

    # =========================================================================
    # Step 5: List Files in Collection
    # =========================================================================
    print_step(5, "List Files in Collection")

    files = ingestor.list_files(args.collection)
    print(f"Found {len(files)} file(s) in collection:")
    for f in files:
        print(f"  - {f.file_name} (status: {f.status.value})")

    # =========================================================================
    # Step 6: Get Collection Info
    # =========================================================================
    print_step(6, "Get Collection Info")

    coll_info = ingestor.get_collection(args.collection)
    if coll_info:
        print(f"Collection: {coll_info.name}")
        print(f"  Chunks: {coll_info.chunk_count}")
        print(f"  Backend: {coll_info.backend}")
    else:
        print("Collection info not available")

    # =========================================================================
    # Step 7: Test Retrieval (Optional)
    # =========================================================================
    if not args.skip_retrieval:
        print_step(7, "Test Retrieval")

        test_queries = [
            "What is cystic fibrosis?",
            "What treatments are available for cystic fibrosis?",
            "What are the key findings about CFTR?",
        ]

        for query in test_queries:
            print(f"\nQuery: '{query}'")
            try:
                result = await retriever.retrieve(
                    query=query,
                    collection_name=args.collection,
                    top_k=3,
                )
                print(f"  Found {len(result.chunks)} chunks:")
                for chunk in result.chunks[:2]:  # Show first 2
                    content_preview = chunk.content[:100].replace("\n", " ")
                    print(f"    - {chunk.display_citation}: {content_preview}...")
            except Exception as e:
                print(f"  Retrieval error: {e}")

    # =========================================================================
    # Step 8: Cleanup
    # =========================================================================
    if args.delete:
        print_step(8, "Cleanup - Delete Collection")

        success = ingestor.delete_collection(args.collection)
        if success:
            print(f"✓ Deleted collection: {args.collection}")
        else:
            print(f"⚠ Failed to delete collection: {args.collection}")
    else:
        print_step(8, "Cleanup - Skipped (default behavior)")
        print(f"Collection '{args.collection}' preserved for inspection")

    # =========================================================================
    # Summary
    # =========================================================================
    print_header("Test Summary")
    print(f"Ingest URL: {args.ingest_url}")
    print(f"Query URL:  {args.query_url}")
    print(f"Collection: {args.collection}")
    print(f"Files uploaded: {len(uploaded_files)}")
    print(f"Files in collection: {len(files)}")
    if coll_info:
        print(f"Total chunks: {coll_info.chunk_count}")
    print("\n✓ End-to-end test completed!")


if __name__ == "__main__":
    asyncio.run(main())
