# Foundational RAG Adapter

HTTP client adapter for deployed [NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag) endpoints.

> **Tested with:** NVIDIA RAG Blueprint **`v2.4.0`** (Helm chart `nvidia-blueprint-rag`). Other versions may work but have not been validated.

## Prerequisites: Deploy the RAG Blueprint

**This adapter requires a deployed NVIDIA RAG Blueprint server.**

Before using this adapter, you must deploy the NVIDIA RAG Blueprint. Follow the official deployment guide:

**[Deploy Docker Self-Hosted Guide](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/deploy-docker-self-hosted.md)**

**[Deploy Docker NVIDIA-Hosted Guide](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/deploy-docker-nvidia-hosted.md)**

The deployment creates two servers:
- **ingestor-server** (port 8082): Document upload, collection management
- **rag-server** (port 8081): Search and retrieval

## Overview

This adapter enables the Knowledge Layer to work with a deployed NVIDIA RAG Blueprint server. Instead of running ingestion and retrieval locally, all operations are performed via HTTP calls to the remote RAG server.

## When to Use

| Use Case | Recommended Adapter |
|----------|---------------------|
| Development/Testing | `llamaindex` or `nvingest_inprocess` |
| Production with dedicated RAG server | **`foundational_rag`** |

## Architecture

The NVIDIA RAG Blueprint has **two separate servers**:

```
┌─────────────────────┐                ┌─────────────────────────┐
│   Knowledge Layer   │                │  NVIDIA RAG Blueprint   │
│ (foundational_rag)  │                │                         │
├─────────────────────┤                ├─────────────────────────┤
│                     │     HTTP       │  ingestor-server :8082  │
│ FoundationalRag     │ ────────────>  │  ├── POST /documents    │
│ Ingestor            │                │  ├── GET /documents     │
│                     │                │  ├── DELETE /documents  │
│                     │                │  ├── GET /collections   │
│                     │                │  ├── POST /collection   │
│                     │                │  ├── DELETE /collections│
│                     │                │  ├── GET /status        │
│                     │                │  └── GET /health        │
├─────────────────────┤                ├─────────────────────────┤
│                     │     HTTP       │  rag-server :8081       │
│ FoundationalRag     │ ────────────>  │  ├── POST /search       │
│ Retriever           │                │  └── GET /health        │
└─────────────────────┘                └─────────────────────────┘
```

## Configuration

### Environment Variables

```bash
# rag-server URL (for retrieval) - port 8081
export RAG_SERVER_URL="http://rag-server:8081/v1"

# ingestor-server URL (for uploads) - port 8082
export RAG_INGEST_URL="http://ingestor-server:8082/v1"

# Optional: API key for authentication
export RAG_API_KEY="your-api-key"
```

### Python Usage

```python
import knowledge_layer.foundational_rag.adapter  # Register the adapter
from deep_researcher_agent.knowledge.factory import get_ingestor, get_retriever

# Get ingestor (uses ingestor-server - port 8082)
ingestor = get_ingestor("foundational_rag", {
    "rag_url": "http://ingestor-server:8082/v1",
    "api_key": None,  # Optional
    "timeout": 300,
    "chunk_size": 512,
    "chunk_overlap": 150,
})

# Get retriever (uses rag-server - port 8081)
retriever = get_retriever("foundational_rag", {
    "rag_url": "http://rag-server:8081/v1",
})
```

### Config File (YAML)

```yaml
# config_cli_hosted.yml
functions:
  knowledge_search:
    _type: knowledge_retrieval
    backend: foundational_rag
    rag_url: ${RAG_SERVER_URL:-http://rag-server:8081/v1}
    ingest_url: ${RAG_INGEST_URL:-http://ingestor-server:8082/v1}
    collection_name: ${SESSION_ID}
    api_key: ${RAG_API_KEY}  # Optional
```

## API Methods

### Collection Management

```python
# Create a collection (typically using session_id as name)
collection = ingestor.create_collection(
    name="session_abc123",
    description="User session documents",
    metadata={"embedding_dimension": 2048}
)

# List all collections
collections = ingestor.list_collections()
for c in collections:
    print(f"{c.name}: {c.chunk_count} chunks")

# Get specific collection
collection = ingestor.get_collection("session_abc123")

# Delete collection (cleanup on session end)
success = ingestor.delete_collection("session_abc123")
```

### File Management

```python
# Upload a file
file_info = ingestor.upload_file(
    file_path="/path/to/document.pdf",
    collection_name="session_abc123",
    metadata={"chunk_size": 512}
)
print(f"Task ID: {file_info.metadata.get('task_id')}")

# Check file status (poll until complete)
status = ingestor.get_file_status(
    file_id=file_info.metadata.get('task_id'),
    collection_name="session_abc123"
)
print(f"Status: {status.status}")  # UPLOADING -> INGESTING -> SUCCESS

# List files in collection
files = ingestor.list_files("session_abc123")
for f in files:
    print(f"{f.file_name}: {f.status}")

# Delete a file
success = ingestor.delete_file("document.pdf", "session_abc123")

# Batch delete files
result = ingestor.delete_files(
    document_names=["doc1.pdf", "doc2.pdf"],
    collection_name="session_abc123"
)
print(f"Deleted: {result['total_deleted']}")
```

### Retrieval

```python
import asyncio

async def search():
    result = await retriever.retrieve(
        query="What are the key findings?",
        collection_name="session_abc123",
        top_k=10,
    )
    for chunk in result.chunks:
        print(f"[{chunk.display_citation}] {chunk.content[:100]}...")

asyncio.run(search())
```

### Async Ingestion with Status Polling

The RAG server uses Celery for async task processing:

```
Upload → PENDING → STARTED → SUCCESS/FAILURE
```

```python
import time
from deep_researcher_agent.knowledge.schema import FileStatus

# Upload file (returns immediately with task_id)
file_info = ingestor.upload_file(
    file_path="/path/to/doc.pdf",
    collection_name="session_abc123",
)
task_id = file_info.metadata.get("task_id")
print(f"Task ID: {task_id}")

# Poll for completion (every 5 seconds like frontend)
while True:
    status = ingestor.get_file_status(task_id, "session_abc123")
    if status:
        raw_state = status.metadata.get("raw_state", "UNKNOWN")
        print(f"Status: {status.status.value} (Celery: {raw_state})")

        # Check for terminal states
        if status.status in (FileStatus.SUCCESS, FileStatus.FAILED):
            break

    time.sleep(5)

if status.status == FileStatus.SUCCESS:
    print(f"✓ Ingestion complete!")
else:
    print(f"✗ Failed: {status.error_message}")
```

## RAG Blueprint API Reference

The adapter calls these NVIDIA RAG Blueprint endpoints:

**ingestor-server (8082)**:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/documents` | POST | Upload document(s) for ingestion |
| `/documents` | GET | List documents in a collection |
| `/documents` | DELETE | Delete documents by name |
| `/status` | GET | Get ingestion task status |
| `/collections` | GET | List all collections |
| `/collection` | POST | Create a new collection |
| `/collections` | DELETE | Delete collections |

**rag-server (8081)**:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/search` | POST | Search/retrieve chunks from collections |

## Session Management Pattern

The recommended pattern is to use `session_id` as the `collection_name`:

```python
import uuid

# When session starts
session_id = str(uuid.uuid4())
ingestor.create_collection(name=session_id)

# During session - upload files
ingestor.upload_file("/path/to/doc.pdf", session_id)

# Query documents
result = await retriever.retrieve("query", session_id)

# When session ends - cleanup
ingestor.delete_collection(session_id)
```

This ensures:
- **Isolation**: Each session has its own document store
- **Simplicity**: No user-facing naming, auto-generated
- **Easy cleanup**: Session ends = delete collection

## Troubleshooting

### Connection Refused
```
requests.exceptions.ConnectionError: Connection refused
```
- Verify RAG server is running: `curl http://RAG_SERVER_URL/health`
- Check firewall/network access
- See [deployment guide](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/deploy-docker-self-hosted.md) for setup

### 422 Validation Error
```
requests.exceptions.HTTPError: 422 Client Error
```
- Collection may already exist
- Check request payload format

### Timeout
```
requests.exceptions.ReadTimeout
```
- Increase `timeout` config (default: 300s)
- Large files may take longer to ingest

### Task Not Found
```
Task {task_id} not found
```
- Task IDs expire after completion
- Use `list_files()` instead for completed files

## Differences from Local Adapters

| Feature | Local Adapters | Foundational RAG Adapter |
|---------|---------------|--------------------------|
| Storage | Local (Milvus-Lite, ChromaDB) | Remote (RAG server's Milvus) |
| Processing | In-process | Remote server |
| Latency | Low | Network dependent |
| Scaling | Single machine | Server-side scaling |
| State | Local job store | Server-side task tracking |

## Requirements

```
pip install requests urllib3
```

No heavy ML dependencies needed - all processing happens on the RAG server.

## Related Links

- [NVIDIA RAG Blueprint Repository](https://github.com/NVIDIA-AI-Blueprints/rag)
- [Deployment Guide](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/deploy-docker-self-hosted.md)
- [RAG Blueprint Documentation](https://github.com/NVIDIA-AI-Blueprints/rag/tree/main/docs)
