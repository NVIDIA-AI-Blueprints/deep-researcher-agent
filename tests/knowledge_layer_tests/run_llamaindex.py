#!/usr/bin/env python3
"""
LlamaIndex Backend - End-to-End Test

This script tests the complete knowledge layer workflow with LlamaIndex:
1. Ingest PDFs with multimodal extraction (text, tables, charts, images)
2. Store embeddings in ChromaDB
3. Test retrieval via semantic search

Prerequisites:
    - NVIDIA_API_KEY environment variable (via export or .env file)
    - LlamaIndex dependencies: uv pip install -e "sources/knowledge_layer[llamaindex]"

Usage:
    # Option 1: Export env var
    export NVIDIA_API_KEY=nvapi-your-key

    # Option 2: Use .env file (auto-loaded, won't override existing vars)
    echo 'NVIDIA_API_KEY=nvapi-your-key' >> .env

    # Run with default test file
    python tests/knowledge_layer_tests/run_llamaindex.py

    # Run with ALL PDFs in data directory
    python tests/knowledge_layer_tests/run_llamaindex.py tests/knowledge_layer_tests/data/*.pdf

    # Run with specific files
    python tests/knowledge_layer_tests/run_llamaindex.py doc1.pdf doc2.pdf

    # Text-only mode (faster, no VLM)
    python tests/knowledge_layer_tests/run_llamaindex.py --text-only

    # Generate one-sentence summary for each file
    python tests/knowledge_layer_tests/run_llamaindex.py --generate-summary

    # Skip retrieval test
    python tests/knowledge_layer_tests/run_llamaindex.py --ingest-only
"""

import argparse
import asyncio
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Knowledge Layer E2E Test - Ingest and query documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest default test file
  python tests/knowledge_layer_tests/run_llamaindex.py

  # Ingest ALL PDFs in data directory (recommended)
  python tests/knowledge_layer_tests/run_llamaindex.py tests/knowledge_layer_tests/data/*.pdf --collection my_test

  # Ingest specific PDF files
  python tests/knowledge_layer_tests/run_llamaindex.py doc1.pdf doc2.pdf doc3.pdf

  # Text-only mode (faster, no VLM API calls)
  python tests/knowledge_layer_tests/run_llamaindex.py data/*.pdf --text-only --collection fast_test

  # Custom queries after ingestion
  python tests/knowledge_layer_tests/run_llamaindex.py --query "What is CFTR?" --query "What treatments exist?"

  # Ingest only (skip retrieval test)
  python tests/knowledge_layer_tests/run_llamaindex.py --ingest-only file.pdf
        """,
    )
    parser.add_argument("files", nargs="*", help="PDF files to ingest. If not specified, uses default test file.")
    parser.add_argument(
        "--collection", "-c", default="test_collection", help="Collection name (default: test_collection)"
    )
    parser.add_argument(
        "--chroma-dir",
        "-d",
        default="/tmp/chroma_data",
        help="ChromaDB persistence directory (default: /tmp/chroma_data)",
    )
    parser.add_argument("--ingest-only", "-i", action="store_true", help="Only ingest, skip retrieval test")
    parser.add_argument(
        "--query", "-q", action="append", help="Custom query to run after ingestion (can specify multiple)"
    )
    parser.add_argument(
        "--text-only",
        "-t",
        action="store_true",
        help="Text extraction only (skip tables, charts, images - avoids VLM API calls)",
    )
    parser.add_argument(
        "--generate-summary",
        "-s",
        action="store_true",
        help="Generate one-sentence summary for each file during ingestion",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    print("=" * 70)
    print("Knowledge Layer E2E Test - LlamaIndex with Multimodal Extraction")
    print("=" * 70)

    # Import adapter to register it with the factory
    from knowledge_layer.llamaindex.adapter import LlamaIndexIngestor  # noqa: F401
    from knowledge_layer.llamaindex.adapter import LlamaIndexRetriever  # noqa: F401

    from deep_researcher_agent.knowledge import get_ingestor
    from deep_researcher_agent.knowledge import get_retriever

    # Configuration
    persist_dir = args.chroma_dir
    collection_name = args.collection

    # Determine files to ingest
    if args.files:
        file_paths = [Path(f) for f in args.files]
    else:
        # Default test file (relative to tests/knowledge_layer_tests/)
        file_paths = [Path(__file__).parent / "data" / "multimodal_test.pdf"]

    print(f"\nFiles to ingest: {len(file_paths)}")
    for f in file_paths:
        print(f"  - {f}")
    print(f"ChromaDB: {persist_dir}")
    print(f"Collection: {collection_name}")

    # Validate files exist
    missing = [f for f in file_paths if not f.exists()]
    if missing:
        print("\nERROR: Files not found:")
        for f in missing:
            print(f"  - {f}")
        return 1

    # Check API key
    if not os.environ.get("NVIDIA_API_KEY"):
        print("\nERROR: NVIDIA_API_KEY not set. Run:")
        print('  export NVIDIA_API_KEY="nvapi-your-key"')
        return 1

    # =========================================================================
    # STEP 1: Ingestion
    # =========================================================================
    print("\n" + "-" * 70)
    print(f"STEP 1: Ingesting {len(file_paths)} PDF(s)")
    print("-" * 70)

    # Configure extraction based on flags
    if args.text_only:
        ingest_config = {
            "persist_dir": persist_dir,
            "extract_tables": False,
            "extract_charts": False,
            "extract_images": False,
            "generate_summary": args.generate_summary,
        }
        print("Extraction mode: TEXT ONLY (no VLM)")
    else:
        ingest_config = {
            "persist_dir": persist_dir,
            "extract_tables": True,  # Extract tables via pdfplumber
            "extract_charts": True,  # Extract charts with VLM analysis
            "extract_images": True,  # Extract images with VLM captioning
            "generate_summary": args.generate_summary,
        }
        print("Extraction mode: text + tables + charts + images (VLM-enabled)")

    if args.generate_summary:
        print("Summary generation: ENABLED")

    ingestor = get_ingestor("llamaindex", ingest_config)

    # Submit ingestion job with all files
    job_id = ingestor.submit_job(
        file_paths=[str(f) for f in file_paths],
        collection_name=collection_name,
    )
    print(f"Job ID: {job_id}")

    # Poll for completion
    start_time = time.time()
    while True:
        status = ingestor.get_job_status(job_id)
        elapsed = time.time() - start_time
        print(f"  [{elapsed:.1f}s] {status.status.value}: {status.processed_files}/{status.total_files} files")

        if status.is_terminal:
            break
        time.sleep(1)

    # Display results
    if status.is_success:
        total_chunks = sum(f.chunks_created for f in status.file_details)
        metadata = status.metadata or {}

        print("\nIngestion Complete!")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Text chunks: {metadata.get('text_chunks', 'N/A')}")
        print(f"  Tables extracted: {metadata.get('tables_extracted', 0)}")
        print(f"  Charts extracted: {metadata.get('charts_extracted', 0)}")
        print(f"  Images captioned: {metadata.get('images_captioned', 0)}")
        print(f"  Time: {elapsed:.1f}s")

        # Display summaries if generated
        if args.generate_summary:
            print("\nGenerated Summaries:")
            files = ingestor.list_files(collection_name)
            for f in files:
                summary = f.metadata.get("summary")
                if summary:
                    print(f"  [{f.file_name}]: {summary}")
                else:
                    print(f"  [{f.file_name}]: No summary generated")
    else:
        print(f"\nIngestion FAILED: {status.error_message}")
        return 1

    # =========================================================================
    # STEP 2: Retrieval (Semantic Search)
    # =========================================================================
    if args.ingest_only:
        print("\n" + "-" * 70)
        print("Skipping retrieval test (--ingest-only)")
        print("-" * 70)
    else:
        print("\n" + "-" * 70)
        print("STEP 2: Testing Semantic Search")
        print("-" * 70)

        retriever = get_retriever("llamaindex", {"persist_dir": persist_dir})

        # Use custom queries if provided, otherwise use defaults
        if args.query:
            queries = args.query
        else:
            queries = [
                "What is this document about?",
                "Are there any tables or charts?",
                "What is the conclusion?",
            ]

        for query in queries:
            print(f'\nQuery: "{query}"')

            result = await retriever.retrieve(
                query=query,
                collection_name=collection_name,
                top_k=3,
            )

            if not result.chunks:
                print("  No results found.")
                continue

            print(f"  Found {len(result.chunks)} results:")
            for i, chunk in enumerate(result.chunks, 1):
                content_preview = chunk.content[:120].replace("\n", " ")
                print(f"    [{i}] score={chunk.score:.3f} | type={chunk.content_type.value}")
                print(f"        {chunk.display_citation}")
                print(f"        {content_preview}...")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
    print(f"\nData persisted to: {persist_dir}")
    print(f"Collection: {collection_name}")
    print(f"Files ingested: {len(file_paths)}")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
