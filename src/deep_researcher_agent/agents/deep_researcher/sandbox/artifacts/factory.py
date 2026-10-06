# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configured construction for the application-level artifact store."""

from __future__ import annotations

import os
from functools import lru_cache

from .blob_store import S3ArtifactBlobStore
from .store import ArtifactStore
from .store import SqlArtifactStore


def build_artifact_store(db_url: str) -> ArtifactStore:
    """Build the SQL metadata store with the configured byte provider."""
    provider = os.environ.get("DEEP_RESEARCHER_ARTIFACT_BLOB_PROVIDER", "sql").strip().lower()
    return _build_artifact_store(
        db_url,
        provider,
        os.environ.get("DEEP_RESEARCHER_ARTIFACT_S3_BUCKET", ""),
        os.environ.get("DEEP_RESEARCHER_ARTIFACT_S3_ENDPOINT_URL"),
        os.environ.get("DEEP_RESEARCHER_ARTIFACT_S3_REGION"),
        os.environ.get("DEEP_RESEARCHER_ARTIFACT_S3_PREFIX", "artifacts/v1"),
    )


@lru_cache(maxsize=16)
def _build_artifact_store(
    db_url: str,
    provider: str,
    bucket: str,
    endpoint_url: str | None,
    region: str | None,
    prefix: str,
) -> ArtifactStore:
    """Construct and cache one configured store per process and configuration."""
    if provider == "sql":
        return SqlArtifactStore(db_url)
    if provider == "s3":
        blob_store = S3ArtifactBlobStore(
            bucket=bucket,
            endpoint_url=endpoint_url,
            region=region,
            prefix=prefix,
        )
        return SqlArtifactStore(db_url, blob_store=blob_store)
    raise ValueError(f"Unsupported DEEP_RESEARCHER_ARTIFACT_BLOB_PROVIDER: {provider!r}; expected 'sql' or 's3'")
