# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte storage adapters for durable sandbox artifacts."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

from .models import Artifact

_READ_CHUNK_BYTES = 1 << 20


class ArtifactBlobStore(ABC):
    """Provider interface for artifact bytes; metadata remains in SQL."""

    @property
    @abstractmethod
    def scheme(self) -> str:
        """URI scheme handled by this provider."""

    @abstractmethod
    def make_uri(self, artifact: Artifact) -> str:
        """Return the durable URI to persist in artifact metadata."""

    @abstractmethod
    def put(self, artifact: Artifact, data: bytes) -> bytes | None:
        """Write bytes externally, or return bytes stored with SQL metadata."""

    @abstractmethod
    def open_bytes(self, artifact: Artifact) -> Iterator[bytes]:
        """Stream artifact bytes from the provider."""

    @abstractmethod
    def delete(self, artifact: Artifact) -> None:
        """Delete artifact bytes. Missing bytes must be treated as success."""

    @abstractmethod
    def validate(self) -> None:
        """Raise when the configured byte storage is unavailable."""


class SqlArtifactBlobStore(ArtifactBlobStore):
    """Store artifact bytes in the existing SQL ``content`` column."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    @property
    def scheme(self) -> str:
        return "db"

    def make_uri(self, artifact: Artifact) -> str:
        return f"db://artifacts/{artifact.artifact_id}"

    def put(self, artifact: Artifact, data: bytes) -> bytes:
        return data

    def open_bytes(self, artifact: Artifact) -> Iterator[bytes]:
        from sqlalchemy import text

        with self._engine.connect() as conn:
            row = conn.execute(
                text("SELECT content FROM artifacts WHERE job_id = :job_id AND artifact_id = :artifact_id"),
                {"job_id": artifact.job_id, "artifact_id": artifact.artifact_id},
            ).fetchone()
        if row is None or row[0] is None:
            return
        data: bytes = row[0]
        for start in range(0, len(data), _READ_CHUNK_BYTES):
            yield data[start : start + _READ_CHUNK_BYTES]

    def delete(self, artifact: Artifact) -> None:
        # Inline bytes are deleted with their SQL metadata row.
        pass

    def validate(self) -> None:
        pass


class S3ArtifactBlobStore(ArtifactBlobStore):
    """Store bytes in AWS S3 or an S3-compatible service such as MinIO."""

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "artifacts/v1",
        endpoint_url: str | None = None,
        region: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not bucket:
            raise ValueError("DEEP_RESEARCHER_ARTIFACT_S3_BUCKET is required when the artifact blob provider is s3")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                raise RuntimeError(
                    "S3 artifact storage requires the optional 's3' dependency: install deep-researcher-agent[s3]"
                ) from exc
            client = boto3.client(
                "s3",
                endpoint_url=endpoint_url or None,
                region_name=region or None,
                config=Config(
                    connect_timeout=10,
                    read_timeout=30,
                    retries={"max_attempts": 3, "mode": "standard"},
                    # MinIO and other custom endpoints do not necessarily provide
                    # bucket-name DNS. Path style works consistently for them.
                    s3={"addressing_style": "path"} if endpoint_url else {},
                ),
            )
        self._client = client

    @property
    def scheme(self) -> str:
        return "s3"

    def make_uri(self, artifact: Artifact) -> str:
        key = "/".join(part for part in (self.prefix, artifact.job_id, artifact.artifact_id) if part)
        return f"s3://{self.bucket}/{key}"

    def put(self, artifact: Artifact, data: bytes) -> None:
        bucket, key = self._location(artifact.storage_uri)
        self._client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=artifact.mime_type,
            Metadata={"sha256": artifact.sha256},
        )

    def open_bytes(self, artifact: Artifact) -> Iterator[bytes]:
        bucket, key = self._location(artifact.storage_uri)
        response = self._client.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        try:
            yield from body.iter_chunks(chunk_size=_READ_CHUNK_BYTES)
        finally:
            body.close()

    def delete(self, artifact: Artifact) -> None:
        bucket, key = self._location(artifact.storage_uri)
        self._client.delete_object(Bucket=bucket, Key=key)

    def validate(self) -> None:
        self._client.head_bucket(Bucket=self.bucket)

    @staticmethod
    def _location(storage_uri: str) -> tuple[str, str]:
        parsed = urlparse(storage_uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
            raise ValueError(f"Invalid S3 artifact URI: {storage_uri}")
        return parsed.netloc, parsed.path.lstrip("/")
