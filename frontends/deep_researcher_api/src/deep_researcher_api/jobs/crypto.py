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

"""Content encryption for Deep Researcher Agent async jobs.

This module covers ``job_info.output`` and selected sensitive fields in
``job_events.event_data`` for the Deep Researcher Agent async API. Checkpoint state, summaries,
and errors remain plaintext.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import random
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel

logger = logging.getLogger(__name__)

ENVELOPE_PREFIX = "aiqenc:"
ENCRYPTED_FIELD_MARKER = "_deep_researcher_encrypted"
ENCRYPTED_FIELD_VALUE = "value"
ENVELOPE_VERSION = 1
CONTENT_ALGORITHM = "AES-256-GCM"
CONTENT_ENCRYPTION_POLICY_VERSION = 1
DEK_BYTES = 32
GCM_NONCE_BYTES = 12
GCM_TAG_BYTES = 16
DEFAULT_READINESS_TTL_SECONDS = 60.0
DEFAULT_DEK_CACHE_TTL_SECONDS = 900.0
DEFAULT_DEK_CACHE_MAX_ENTRIES = 1024
DEFAULT_VAULT_TRANSIT_MOUNT = "transit"
DEFAULT_VAULT_TIMEOUT_SECONDS = 5.0
_VAULT_ATTEMPTS = 2
_VAULT_RETRY_BASE_SECONDS = 0.05
_VAULT_RETRY_JITTER_SECONDS = 0.025
_VAULT_AUTH_RETRY_EXCEPTION_NAMES = {"Unauthorized"}
_VAULT_RETRYABLE_EXCEPTION_NAMES = {
    "BadGateway",
    "InternalServerError",
    "RateLimitExceeded",
    "VaultDown",
}
_VAULT_NON_RETRYABLE_EXCEPTION_NAMES = {
    "Forbidden",
    "InvalidPath",
    "InvalidRequest",
    "ParamValidationError",
    "PreconditionFailed",
    "UnsupportedOperation",
    "VaultNotInitialized",
}


class ContentEncryptionError(Exception):
    """Base class for content-encryption failures."""


class ContentEncryptionConfigError(ContentEncryptionError, ValueError):
    """Invalid encryption configuration. This is a startup-hard failure."""


class ContentEncryptionUnavailable(ContentEncryptionError):
    """Encryption is configured but operationally unavailable."""


class ContentEncryptionInvalidData(ContentEncryptionError):
    """Persisted encrypted data is malformed, plaintext, or undecryptable."""


class ContentEncryptionPlaintextViolation(ContentEncryptionInvalidData):
    """Encrypted mode encountered plaintext where an envelope is required."""


class ContentEncryptionPolicyMismatch(ContentEncryptionError):
    """API and worker content-encryption policies do not match."""


@dataclass(frozen=True)
class ContentEncryptionPolicyIdentity:
    """Non-secret encryption identity propagated from the API to a worker."""

    version: int
    mode: str
    key_id: str | None = None
    static_key_fingerprint: str | None = field(default=None, repr=False)
    vault_addr: str | None = None
    vault_namespace: str | None = None
    vault_transit_mount: str | None = None
    vault_transit_key: str | None = None


@dataclass(frozen=True)
class ContentEncryptionConfig:
    """Process-local content-encryption configuration parsed from env."""

    mode: str
    key_id: str | None = None
    static_key: bytes | None = field(default=None, repr=False)
    vault_addr: str | None = None
    vault_namespace: str | None = None
    vault_transit_mount: str = DEFAULT_VAULT_TRANSIT_MOUNT
    vault_transit_key: str | None = None
    vault_role_id: str | None = field(default=None, repr=False)
    vault_secret_id: str | None = field(default=None, repr=False)
    vault_timeout_seconds: float = DEFAULT_VAULT_TIMEOUT_SECONDS
    readiness_ttl_seconds: float = DEFAULT_READINESS_TTL_SECONDS
    dek_cache_ttl_seconds: float = DEFAULT_DEK_CACHE_TTL_SECONDS
    dek_cache_max_entries: int = DEFAULT_DEK_CACHE_MAX_ENTRIES

    @property
    def encrypted(self) -> bool:
        return self.mode in {"key", "vault"}

    @property
    def effective_key_id(self) -> str:
        if self.key_id:
            return self.key_id
        if self.mode == "key":
            return "static-key"
        if self.mode == "vault":
            return f"{self.vault_transit_mount}/{self.vault_transit_key}"
        return "off"

    @property
    def signature(self) -> tuple[Any, ...]:
        return (
            self.mode,
            self.key_id,
            _secret_fingerprint(self.static_key),
            self.vault_addr,
            self.vault_namespace,
            self.vault_transit_mount,
            self.vault_transit_key,
            _secret_fingerprint(self.vault_role_id),
            _secret_fingerprint(self.vault_secret_id),
            self.vault_timeout_seconds,
            self.readiness_ttl_seconds,
            self.dek_cache_ttl_seconds,
            self.dek_cache_max_entries,
        )

    @property
    def policy_identity(self) -> ContentEncryptionPolicyIdentity:
        """Return the non-secret identity workers must match for submitted jobs."""

        if self.mode == "key":
            return ContentEncryptionPolicyIdentity(
                version=CONTENT_ENCRYPTION_POLICY_VERSION,
                mode=self.mode,
                key_id=self.effective_key_id,
                static_key_fingerprint=_secret_fingerprint(self.static_key),
            )
        if self.mode == "vault":
            return ContentEncryptionPolicyIdentity(
                version=CONTENT_ENCRYPTION_POLICY_VERSION,
                mode=self.mode,
                key_id=self.effective_key_id,
                vault_addr=self.vault_addr,
                vault_namespace=self.vault_namespace,
                vault_transit_mount=self.vault_transit_mount,
                vault_transit_key=self.vault_transit_key,
            )
        return ContentEncryptionPolicyIdentity(
            version=CONTENT_ENCRYPTION_POLICY_VERSION,
            mode="off",
        )


@dataclass(frozen=True)
class ContentEncryptionReadiness:
    """Cached readiness state safe to expose through health responses."""

    mode: str
    ready: bool
    checked_at: float | None = None
    reason: str | None = None
    exception_type: str | None = None
    encrypt_ready: bool | None = None
    decrypt_ready: bool | None = None

    def to_health_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"mode": self.mode, "ready": self.ready}
        if self.encrypt_ready is not None:
            result["encrypt_ready"] = self.encrypt_ready
        if self.decrypt_ready is not None:
            result["decrypt_ready"] = self.decrypt_ready
        if self.reason:
            result["reason"] = self.reason
        if self.exception_type:
            result["exception_type"] = self.exception_type
        return result


@dataclass(frozen=True)
class WrappedDEK:
    wrap: str
    kid: str
    wrapped_dek: str
    wrap_nonce: str | None = None
    wrap_tag: str | None = None


@dataclass
class JobContentCipher:
    """Per-job encryption context used by the worker final-output write."""

    manager: ContentEncryptionManager
    job_id: str
    dek: bytes | None = None
    wrapped_dek: WrappedDEK | None = None

    def encrypt_output_json(self, output_json: str) -> str:
        if not self.manager.config.encrypted:
            return output_json
        if self.dek is None or self.wrapped_dek is None:
            raise ContentEncryptionUnavailable("encrypted job output cipher is not initialized")
        return self.manager.encrypt_job_output_text(self.job_id, output_json, self.dek, self.wrapped_dek)

    def encrypt_event_field_json(self, field_path: str, value_json: str) -> str:
        if not self.manager.config.encrypted:
            return value_json
        if self.dek is None or self.wrapped_dek is None:
            raise ContentEncryptionUnavailable("encrypted job event cipher is not initialized")
        return self.manager.encrypt_job_event_field_text(
            self.job_id,
            field_path,
            value_json,
            self.dek,
            self.wrapped_dek,
        )


@dataclass
class _DEKCacheEntry:
    dek: bytes
    expires_at: float


@dataclass
class _DEKCache:
    ttl_seconds: float
    max_entries: int
    _entries: OrderedDict[str, _DEKCacheEntry] = field(default_factory=OrderedDict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def get(self, cache_key: str) -> bytes | None:
        if self.ttl_seconds <= 0:
            return None
        with self._lock:
            now = time.monotonic()
            entry = self._entries.get(cache_key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._entries.pop(cache_key, None)
                return None
            self._entries.move_to_end(cache_key)
            return entry.dek

    def put(self, cache_key: str, dek: bytes) -> None:
        if self.ttl_seconds <= 0:
            return
        with self._lock:
            now = time.monotonic()
            self._entries[cache_key] = _DEKCacheEntry(dek=dek, expires_at=now + self.ttl_seconds)
            self._entries.move_to_end(cache_key)
            self._evict_locked(now)

    def _evict(self, now: float) -> None:
        with self._lock:
            self._evict_locked(now)

    def _evict_locked(self, now: float) -> None:
        expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired:
            self._entries.pop(key, None)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


class ContentEncryptionManager:
    """Encrypt/decrypt async final report payloads for one process config."""

    def __init__(self, config: ContentEncryptionConfig):
        self.config = config
        self._lock = threading.RLock()
        self._dek_cache = _DEKCache(
            ttl_seconds=config.dek_cache_ttl_seconds,
            max_entries=config.dek_cache_max_entries,
        )
        self._readiness = ContentEncryptionReadiness(mode=config.mode, ready=not config.encrypted)
        self._vault_client: _VaultTransitClient | None = None

    def get_readiness(self) -> ContentEncryptionReadiness:
        with self._lock:
            if self.config.mode == "off":
                return ContentEncryptionReadiness(mode="off", ready=True)
            return self._readiness

    def check_readiness(self, *, force: bool = False, operation: str = "readiness") -> ContentEncryptionReadiness:
        with self._lock:
            if self.config.mode == "off":
                self._readiness = ContentEncryptionReadiness(mode="off", ready=True)
                return self._readiness
            if self.config.mode == "key":
                self._readiness = ContentEncryptionReadiness(mode="key", ready=True, checked_at=time.monotonic())
                return self._readiness
            if not force and self._readiness.checked_at is not None:
                age = time.monotonic() - self._readiness.checked_at
                if age < self.config.readiness_ttl_seconds:
                    return self._readiness

            self._readiness = self._check_vault_readiness(operation=operation)
            return self._readiness

    def _check_vault_readiness(self, *, operation: str) -> ContentEncryptionReadiness:
        dek: bytes | None = None
        unwrapped_dek: bytes | None = None
        try:
            dek, wrapped = self._vault().generate_data_key(operation=f"{operation}_readiness_generate")
            try:
                unwrapped_dek = self._vault().unwrap_dek(
                    wrapped.wrapped_dek,
                    operation=f"{operation}_readiness_decrypt",
                )
            except ContentEncryptionUnavailable as exc:
                return self._failed_vault_readiness(
                    operation=operation,
                    reason="vault_decrypt_unavailable",
                    exception=exc,
                    encrypt_ready=True,
                    decrypt_ready=False,
                )
            if unwrapped_dek != dek:
                return self._failed_vault_readiness(
                    operation=operation,
                    reason="vault_readiness_dek_mismatch",
                    exception=None,
                    encrypt_ready=True,
                    decrypt_ready=False,
                )
            return ContentEncryptionReadiness(
                mode=self.config.mode,
                ready=True,
                checked_at=time.monotonic(),
                encrypt_ready=True,
                decrypt_ready=True,
            )
        except ContentEncryptionUnavailable as exc:
            return self._failed_vault_readiness(
                operation=operation,
                reason="vault_generate_unavailable",
                exception=exc,
                encrypt_ready=False,
                decrypt_ready=False,
            )
        finally:
            if dek is not None:
                _zero_bytes(dek)
            if unwrapped_dek is not None:
                _zero_bytes(unwrapped_dek)

    def _failed_vault_readiness(
        self,
        *,
        operation: str,
        reason: str,
        exception: Exception | None,
        encrypt_ready: bool,
        decrypt_ready: bool,
    ) -> ContentEncryptionReadiness:
        readiness = ContentEncryptionReadiness(
            mode=self.config.mode,
            ready=False,
            checked_at=time.monotonic(),
            reason=reason,
            exception_type=exception.__class__.__name__ if exception is not None else None,
            encrypt_ready=encrypt_ready,
            decrypt_ready=decrypt_ready,
        )
        logger.warning(
            "Content encryption readiness failed mode=%s operation=%s reason=%s exception=%s",
            self.config.mode,
            operation,
            readiness.reason,
            readiness.exception_type,
        )
        return readiness

    def require_ready(self, *, operation: str) -> None:
        readiness = self.check_readiness(force=False, operation=operation)
        if self.config.encrypted and not readiness.ready:
            raise ContentEncryptionUnavailable(readiness.reason or "content_encryption_unready")

    def create_job_cipher(self, job_id: str) -> JobContentCipher:
        if self.config.mode == "off":
            return JobContentCipher(manager=self, job_id=job_id)
        if self.config.mode == "key":
            dek = os.urandom(DEK_BYTES)
            wrapped = self._wrap_dek_with_static_key(dek)
            return JobContentCipher(manager=self, job_id=job_id, dek=dek, wrapped_dek=wrapped)
        if self.config.mode == "vault":
            dek, wrapped = self._vault().generate_data_key(operation="worker_job_cipher")
            return JobContentCipher(manager=self, job_id=job_id, dek=dek, wrapped_dek=wrapped)
        raise ContentEncryptionConfigError(f"Unsupported content encryption mode: {self.config.mode}")

    def encrypt_job_output_text(self, job_id: str, output_json: str, dek: bytes, wrapped_dek: WrappedDEK) -> str:
        aad = job_output_aad(job_id)
        return self._encrypt_text(output_json, dek, wrapped_dek, aad)

    def encrypt_job_event_field_text(
        self,
        job_id: str,
        field_path: str,
        value_json: str,
        dek: bytes,
        wrapped_dek: WrappedDEK,
    ) -> str:
        aad = job_event_field_aad(job_id, field_path)
        return self._encrypt_text(value_json, dek, wrapped_dek, aad)

    def _encrypt_text(self, plaintext: str, dek: bytes, wrapped_dek: WrappedDEK, aad: str) -> str:
        nonce = os.urandom(GCM_NONCE_BYTES)
        encrypted = AESGCM(dek).encrypt(nonce, plaintext.encode("utf-8"), aad.encode("utf-8"))
        ciphertext, tag = encrypted[:-GCM_TAG_BYTES], encrypted[-GCM_TAG_BYTES:]
        envelope = {
            "v": ENVELOPE_VERSION,
            "alg": CONTENT_ALGORITHM,
            "wrap": wrapped_dek.wrap,
            "kid": wrapped_dek.kid,
            "aad_hint": aad,
            "wrapped_dek": wrapped_dek.wrapped_dek,
            "nonce": _b64url_encode(nonce),
            "ciphertext": _b64url_encode(ciphertext),
            "tag": _b64url_encode(tag),
        }
        if wrapped_dek.wrap_nonce is not None:
            envelope["wrap_nonce"] = wrapped_dek.wrap_nonce
        if wrapped_dek.wrap_tag is not None:
            envelope["wrap_tag"] = wrapped_dek.wrap_tag
        return encode_envelope(envelope)

    def decrypt_job_output_text(self, job_id: str, stored_output: str) -> str:
        aad = job_output_aad(job_id)
        return self._decrypt_text(stored_output, aad)

    def decrypt_job_event_field_text(self, job_id: str, field_path: str, stored_value: str) -> str:
        aad = job_event_field_aad(job_id, field_path)
        return self._decrypt_text(stored_value, aad)

    def _decrypt_text(self, stored_value: str, aad: str) -> str:
        envelope = decode_envelope(stored_value)
        dek = self._unwrap_dek(envelope)
        try:
            nonce = _required_b64url(envelope, "nonce")
            ciphertext = _required_b64url(envelope, "ciphertext")
            tag = _required_b64url(envelope, "tag")
            plaintext = AESGCM(dek).decrypt(nonce, ciphertext + tag, aad.encode("utf-8"))
        except InvalidTag as exc:
            raise ContentEncryptionInvalidData("encrypted job output failed authentication") from exc
        except ValueError as exc:
            raise ContentEncryptionInvalidData("encrypted job output is malformed") from exc
        return plaintext.decode("utf-8")

    def _unwrap_dek(self, envelope: dict[str, Any]) -> bytes:
        self._validate_envelope_metadata(envelope)
        cache_key = _wrapped_dek_cache_key(envelope)
        cached = self._dek_cache.get(cache_key)
        if cached is not None:
            return cached

        wrap = envelope["wrap"]
        if wrap == "key":
            dek = self._unwrap_dek_with_static_key(envelope)
        elif wrap == "vault":
            dek = self._vault().unwrap_dek(str(envelope["wrapped_dek"]), operation="decrypt_job_output")
        else:
            raise ContentEncryptionInvalidData("encrypted job output uses an unsupported wrap type")

        self._dek_cache.put(cache_key, dek)
        return dek

    def _validate_envelope_metadata(self, envelope: dict[str, Any]) -> None:
        if envelope.get("v") != ENVELOPE_VERSION:
            raise ContentEncryptionInvalidData("encrypted job output uses an unsupported envelope version")
        if envelope.get("alg") != CONTENT_ALGORITHM:
            raise ContentEncryptionInvalidData("encrypted job output uses an unsupported content algorithm")
        if envelope.get("wrap") not in {"key", "vault"}:
            raise ContentEncryptionInvalidData("encrypted job output uses an unsupported wrap type")
        if envelope.get("wrap") != self.config.mode:
            raise ContentEncryptionInvalidData("encrypted job output wrap does not match configured mode")
        if not isinstance(envelope.get("kid"), str) or not envelope["kid"]:
            raise ContentEncryptionInvalidData("encrypted job output is missing a key id")
        if not isinstance(envelope.get("wrapped_dek"), str) or not envelope["wrapped_dek"]:
            raise ContentEncryptionInvalidData("encrypted job output is missing a wrapped DEK")

    def _wrap_dek_with_static_key(self, dek: bytes) -> WrappedDEK:
        key = self.config.static_key
        if key is None:
            raise ContentEncryptionConfigError("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY is required in key mode")
        kid = self.config.effective_key_id
        nonce = os.urandom(GCM_NONCE_BYTES)
        encrypted = AESGCM(key).encrypt(nonce, dek, _kek_wrap_aad(kid))
        ciphertext, tag = encrypted[:-GCM_TAG_BYTES], encrypted[-GCM_TAG_BYTES:]
        return WrappedDEK(
            wrap="key",
            kid=kid,
            wrapped_dek=_b64url_encode(ciphertext),
            wrap_nonce=_b64url_encode(nonce),
            wrap_tag=_b64url_encode(tag),
        )

    def _unwrap_dek_with_static_key(self, envelope: dict[str, Any]) -> bytes:
        key = self.config.static_key
        if key is None:
            raise ContentEncryptionConfigError("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY is required in key mode")
        if envelope.get("kid") != self.config.effective_key_id:
            raise ContentEncryptionInvalidData("encrypted job output key id does not match configured key")
        try:
            nonce = _required_b64url(envelope, "wrap_nonce")
            ciphertext = _required_b64url(envelope, "wrapped_dek")
            tag = _required_b64url(envelope, "wrap_tag")
            dek = AESGCM(key).decrypt(nonce, ciphertext + tag, _kek_wrap_aad(self.config.effective_key_id))
        except InvalidTag as exc:
            raise ContentEncryptionInvalidData("encrypted job output DEK unwrap failed") from exc
        except ValueError as exc:
            raise ContentEncryptionInvalidData("encrypted job output DEK wrapper is malformed") from exc
        if len(dek) != DEK_BYTES:
            raise ContentEncryptionInvalidData("encrypted job output DEK has invalid length")
        return dek

    def _vault(self) -> _VaultTransitClient:
        with self._lock:
            if self._vault_client is None:
                self._vault_client = _VaultTransitClient(self.config)
            return self._vault_client


class _VaultTransitClient:
    """Small synchronous Vault Transit client with bounded retry."""

    def __init__(self, config: ContentEncryptionConfig):
        self._config = config
        self._lock = threading.RLock()
        self._client: Any | None = None

    def generate_data_key(self, *, operation: str) -> tuple[bytes, WrappedDEK]:
        data = self._with_retry(lambda: self._generate_data_key_once(), operation=operation)
        try:
            plaintext = data["data"]["plaintext"]
            ciphertext = data["data"]["ciphertext"]
        except (KeyError, TypeError) as exc:
            raise ContentEncryptionUnavailable("vault_datakey_response_invalid") from exc

        dek = _decode_vault_key(plaintext, field_name="vault plaintext data key")
        if len(dek) != DEK_BYTES:
            raise ContentEncryptionUnavailable("vault_datakey_invalid_length")
        if not isinstance(ciphertext, str) or not ciphertext:
            raise ContentEncryptionUnavailable("vault_datakey_missing_ciphertext")
        return dek, WrappedDEK(
            wrap="vault",
            kid=self._config.effective_key_id,
            wrapped_dek=ciphertext,
        )

    def unwrap_dek(self, wrapped_dek: str, *, operation: str) -> bytes:
        data = self._with_retry(lambda: self._unwrap_dek_once(wrapped_dek), operation=operation)
        try:
            plaintext = data["data"]["plaintext"]
        except (KeyError, TypeError) as exc:
            raise ContentEncryptionUnavailable("vault_decrypt_response_invalid") from exc
        dek = _decode_vault_key(plaintext, field_name="vault plaintext data key")
        if len(dek) != DEK_BYTES:
            raise ContentEncryptionUnavailable("vault_decrypt_invalid_length")
        return dek

    def _generate_data_key_once(self) -> dict[str, Any]:
        return self._authenticated_client().secrets.transit.generate_data_key(
            name=self._required(self._config.vault_transit_key, "DEEP_RESEARCHER_ENCRYPTION_TRANSIT_KEY"),
            key_type="plaintext",
            bits=DEK_BYTES * 8,
            mount_point=self._config.vault_transit_mount,
        )

    def _unwrap_dek_once(self, wrapped_dek: str) -> dict[str, Any]:
        return self._authenticated_client().secrets.transit.decrypt_data(
            name=self._required(self._config.vault_transit_key, "DEEP_RESEARCHER_ENCRYPTION_TRANSIT_KEY"),
            ciphertext=wrapped_dek,
            mount_point=self._config.vault_transit_mount,
        )

    def _with_retry(self, fn, *, operation: str) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(_VAULT_ATTEMPTS):
            try:
                with self._lock:
                    return fn()
            except Exception as exc:  # hvac exposes several operational exception classes.
                if isinstance(exc, ContentEncryptionConfigError):
                    raise
                last_exc = exc
                if not _should_retry_vault_failure(exc):
                    break
                if attempt + 1 < _VAULT_ATTEMPTS:
                    if _is_auth_failure(exc):
                        try:
                            self._login(force=True)
                        except Exception as login_exc:  # AppRole login can fail transiently too.
                            if isinstance(login_exc, ContentEncryptionConfigError):
                                raise
                            last_exc = login_exc
                            if not _should_retry_vault_failure(login_exc):
                                break
                    _sleep_before_vault_retry(attempt)
                    continue

        assert last_exc is not None
        logger.warning(
            "Vault transit operation failed mode=vault operation=%s retryable=%s exception=%s",
            operation,
            _should_retry_vault_failure(last_exc),
            last_exc.__class__.__name__,
        )
        raise ContentEncryptionUnavailable(f"vault_{operation}_failed") from last_exc

    def _authenticated_client(self) -> Any:
        with self._lock:
            client = self._get_client()
            if not client.is_authenticated():
                self._login(force=True)
            return client

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is not None:
                return self._client
            try:
                import hvac
            except ImportError as exc:
                raise ContentEncryptionConfigError("Vault mode requires the hvac package") from exc

            self._client = hvac.Client(
                url=self._required(self._config.vault_addr, "VAULT_ADDR"),
                namespace=self._config.vault_namespace,
                timeout=self._config.vault_timeout_seconds,
            )
            self._login(force=True)
            return self._client

    def _login(self, *, force: bool) -> None:
        with self._lock:
            client = self._get_client_without_login()
            if not force and client.is_authenticated():
                return
            try:
                client.auth.approle.login(
                    role_id=self._required(self._config.vault_role_id, "VAULT_ROLE_ID"),
                    secret_id=self._required(self._config.vault_secret_id, "VAULT_SECRET_ID"),
                )
            except Exception as exc:
                logger.warning("Vault AppRole login failed mode=vault exception=%s", exc.__class__.__name__)
                raise ContentEncryptionUnavailable("vault_approle_login_failed") from exc

    def _get_client_without_login(self) -> Any:
        with self._lock:
            if self._client is not None:
                return self._client
            try:
                import hvac
            except ImportError as exc:
                raise ContentEncryptionConfigError("Vault mode requires the hvac package") from exc
            self._client = hvac.Client(
                url=self._required(self._config.vault_addr, "VAULT_ADDR"),
                namespace=self._config.vault_namespace,
                timeout=self._config.vault_timeout_seconds,
            )
            return self._client

    @staticmethod
    def _required(value: str | None, name: str) -> str:
        if not value:
            raise ContentEncryptionConfigError(f"{name} is required in vault mode")
        return value


_manager: ContentEncryptionManager | None = None
_manager_signature: tuple[Any, ...] | None = None
_manager_lock = threading.RLock()


def reset_content_encryption_manager_for_tests() -> None:
    global _manager
    global _manager_signature
    with _manager_lock:
        _manager = None
        _manager_signature = None


def get_content_encryption_config() -> ContentEncryptionConfig:
    mode = os.environ.get("DEEP_RESEARCHER_CONTENT_ENCRYPTION", "off").strip().lower()
    if mode not in {"off", "key", "vault"}:
        raise ContentEncryptionConfigError("DEEP_RESEARCHER_CONTENT_ENCRYPTION must be one of: off, key, vault")

    readiness_ttl = _parse_non_negative_float(
        os.environ.get("DEEP_RESEARCHER_CONTENT_ENCRYPTION_READINESS_TTL_SECONDS"),
        default=DEFAULT_READINESS_TTL_SECONDS,
        name="DEEP_RESEARCHER_CONTENT_ENCRYPTION_READINESS_TTL_SECONDS",
    )
    dek_cache_ttl = _parse_non_negative_float(
        os.environ.get("DEEP_RESEARCHER_CONTENT_ENCRYPTION_DEK_CACHE_TTL_SECONDS"),
        default=DEFAULT_DEK_CACHE_TTL_SECONDS,
        name="DEEP_RESEARCHER_CONTENT_ENCRYPTION_DEK_CACHE_TTL_SECONDS",
    )

    key_id = _empty_to_none(os.environ.get("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY_ID"))
    if mode == "off":
        return ContentEncryptionConfig(
            mode="off",
            key_id=key_id,
            readiness_ttl_seconds=readiness_ttl,
            dek_cache_ttl_seconds=dek_cache_ttl,
        )
    if mode == "key":
        raw_key = os.environ.get("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY")
        if not raw_key:
            raise ContentEncryptionConfigError("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY is required in key mode")
        key = _strict_base64_decode(raw_key, field_name="DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY")
        if len(key) != DEK_BYTES:
            raise ContentEncryptionConfigError("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY must decode to exactly 32 bytes")
        return ContentEncryptionConfig(
            mode="key",
            key_id=key_id,
            static_key=key,
            readiness_ttl_seconds=readiness_ttl,
            dek_cache_ttl_seconds=dek_cache_ttl,
        )

    vault_addr = _empty_to_none(os.environ.get("VAULT_ADDR"))
    vault_transit_key = _empty_to_none(os.environ.get("DEEP_RESEARCHER_ENCRYPTION_TRANSIT_KEY"))
    vault_role_id = _empty_to_none(os.environ.get("VAULT_ROLE_ID"))
    vault_secret_id = _empty_to_none(os.environ.get("VAULT_SECRET_ID"))
    if vault_addr is None:
        raise ContentEncryptionConfigError("VAULT_ADDR is required in vault mode")
    if vault_transit_key is None:
        raise ContentEncryptionConfigError("DEEP_RESEARCHER_ENCRYPTION_TRANSIT_KEY is required in vault mode")
    if vault_role_id is None:
        raise ContentEncryptionConfigError("VAULT_ROLE_ID is required in vault mode")
    if vault_secret_id is None:
        raise ContentEncryptionConfigError("VAULT_SECRET_ID is required in vault mode")

    return ContentEncryptionConfig(
        mode="vault",
        key_id=key_id,
        vault_addr=vault_addr,
        vault_namespace=_empty_to_none(os.environ.get("VAULT_NAMESPACE")),
        vault_transit_mount=os.environ.get("VAULT_TRANSIT_MOUNT", DEFAULT_VAULT_TRANSIT_MOUNT).strip()
        or DEFAULT_VAULT_TRANSIT_MOUNT,
        vault_transit_key=vault_transit_key,
        vault_role_id=vault_role_id,
        vault_secret_id=vault_secret_id,
        vault_timeout_seconds=_parse_positive_float(
            os.environ.get("VAULT_TIMEOUT_SECONDS"),
            default=DEFAULT_VAULT_TIMEOUT_SECONDS,
            name="VAULT_TIMEOUT_SECONDS",
        ),
        readiness_ttl_seconds=readiness_ttl,
        dek_cache_ttl_seconds=dek_cache_ttl,
    )


def get_content_encryption_manager() -> ContentEncryptionManager:
    global _manager
    global _manager_signature
    config = get_content_encryption_config()
    with _manager_lock:
        if _manager is None or _manager_signature != config.signature:
            _manager = ContentEncryptionManager(config)
            _manager_signature = config.signature
        return _manager


def validate_content_encryption_startup() -> ContentEncryptionReadiness:
    """Validate config and eagerly check operational readiness.

    Configuration errors propagate and fail startup. Vault operational failures
    are cached as unhealthy so the process can start but reject submissions.
    """

    manager = get_content_encryption_manager()
    return manager.check_readiness(force=True, operation="api_startup")


async def validate_content_encryption_startup_async() -> ContentEncryptionReadiness:
    """Validate startup readiness without blocking the FastAPI event loop."""

    return await asyncio.to_thread(validate_content_encryption_startup)


def get_content_encryption_health() -> ContentEncryptionReadiness:
    return get_content_encryption_manager().check_readiness(force=False, operation="health")


async def get_content_encryption_health_async() -> ContentEncryptionReadiness:
    """Check encryption health without blocking the FastAPI event loop."""

    return await asyncio.to_thread(get_content_encryption_health)


def require_content_encryption_ready_for_submission() -> None:
    get_content_encryption_manager().require_ready(operation="submit")


async def require_content_encryption_ready_for_submission_async() -> None:
    """Check submission readiness without blocking the FastAPI event loop."""

    await asyncio.to_thread(require_content_encryption_ready_for_submission)


def get_content_encryption_policy_identity() -> ContentEncryptionPolicyIdentity:
    """Return the process-local policy identity safe to send to a worker."""

    return get_content_encryption_manager().config.policy_identity


def require_content_encryption_policy(expected: ContentEncryptionPolicyIdentity | None) -> None:
    """Fail closed unless the worker policy exactly matches the submitter policy."""

    if expected is None or get_content_encryption_policy_identity() != expected:
        raise ContentEncryptionPolicyMismatch("worker content encryption policy does not match submission policy")


def create_job_content_cipher(job_id: str) -> JobContentCipher:
    return get_content_encryption_manager().create_job_cipher(job_id)


async def update_job_output(
    job_store: Any,
    job_id: str,
    status: Any,
    *,
    output: BaseModel | dict[str, Any] | list[Any] | str,
    cipher: JobContentCipher,
) -> None:
    """Update NAT JobStore output, encrypting only in encrypted modes."""

    if not cipher.manager.config.encrypted:
        await job_store.update_status(job_id, status, output=output)
        return

    output_json = _serialize_output_json(output)
    encrypted_output = cipher.encrypt_output_json(output_json)
    await job_store.update_status(job_id, status, output=encrypted_output)


def serialize_job_output_for_storage(
    output: BaseModel | dict[str, Any] | list[Any] | str,
    cipher: JobContentCipher,
) -> str:
    """Return the exact string ``update_job_output`` would persist to job_info.output.

    Encrypts in encrypted modes; otherwise mirrors NAT's JSON serialization. Used
    by callers that need to write the output through their own (e.g. conditional)
    UPDATE while keeping serialization identical to the normal path.
    """
    if not cipher.manager.config.encrypted:
        if isinstance(output, BaseModel):
            return output.model_dump_json(round_trip=True)
        if isinstance(output, dict | list):
            return json.dumps(output)
        return output
    return cipher.encrypt_output_json(_serialize_output_json(output))


def read_job_output(job_id: str, stored_output: Any) -> Any:
    """Read job output, decrypting only when encrypted mode is configured."""

    if stored_output is None:
        return None

    manager = get_content_encryption_manager()
    if not manager.config.encrypted:
        if isinstance(stored_output, str):
            try:
                return json.loads(stored_output)
            except json.JSONDecodeError:
                return stored_output
        return stored_output

    if not isinstance(stored_output, str) or not stored_output.startswith(ENVELOPE_PREFIX):
        logger.warning(
            "Plaintext job output encountered in encrypted mode mode=%s job_id=%s operation=read_job_output",
            manager.config.mode,
            job_id,
        )
        raise ContentEncryptionPlaintextViolation("job_info.output is plaintext in encrypted mode")

    plaintext_json = manager.decrypt_job_output_text(job_id, stored_output)
    try:
        return json.loads(plaintext_json)
    except json.JSONDecodeError as exc:
        raise ContentEncryptionInvalidData("decrypted job output is not valid JSON") from exc


async def read_job_output_async(job_id: str, stored_output: Any) -> Any:
    """Read/decrypt job output without blocking the FastAPI event loop."""

    return await asyncio.to_thread(read_job_output, job_id, stored_output)


def encrypt_event_field(job_id: str, field_path: str, value: Any, cipher: JobContentCipher | None) -> Any:
    """Encrypt one event payload field, preserving plaintext behavior in off mode."""

    if cipher is None or not cipher.manager.config.encrypted:
        return value

    value_json = json.dumps(value)
    encrypted_value = cipher.encrypt_event_field_json(field_path, value_json)
    return {
        ENCRYPTED_FIELD_MARKER: True,
        ENCRYPTED_FIELD_VALUE: encrypted_value,
    }


def decrypt_event_field(job_id: str, field_path: str, stored_value: Any) -> Any:
    """Decrypt one event payload field when it uses the encrypted-field marker."""

    if not is_encrypted_event_field(stored_value):
        return stored_value

    manager = get_content_encryption_manager()
    if not manager.config.encrypted:
        return stored_value

    plaintext_json = manager.decrypt_job_event_field_text(job_id, field_path, stored_value[ENCRYPTED_FIELD_VALUE])
    try:
        return json.loads(plaintext_json)
    except json.JSONDecodeError as exc:
        raise ContentEncryptionInvalidData("decrypted event field is not valid JSON") from exc


def is_encrypted_event_field(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get(ENCRYPTED_FIELD_MARKER) is True
        and isinstance(value.get(ENCRYPTED_FIELD_VALUE), str)
        and value[ENCRYPTED_FIELD_VALUE].startswith(ENVELOPE_PREFIX)
    )


def encode_envelope(envelope: dict[str, Any]) -> str:
    data = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return ENVELOPE_PREFIX + _b64url_encode(data)


def decode_envelope(value: str) -> dict[str, Any]:
    if not isinstance(value, str) or not value.startswith(ENVELOPE_PREFIX):
        raise ContentEncryptionPlaintextViolation("job_info.output is not an aiqenc envelope")
    encoded = value[len(ENVELOPE_PREFIX) :]
    try:
        raw = _b64url_decode(encoded)
        envelope = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContentEncryptionInvalidData("encrypted job output envelope is malformed") from exc
    if not isinstance(envelope, dict):
        raise ContentEncryptionInvalidData("encrypted job output envelope is malformed")
    return envelope


def job_output_aad(job_id: str) -> str:
    return f"deep-researcher:v1:job_info:output:{job_id}"


def job_event_field_aad(job_id: str, field_path: str) -> str:
    return f"deep-researcher:v1:job_events:event_data:{job_id}:{field_path}"


def _serialize_output_json(output: BaseModel | dict[str, Any] | list[Any] | str) -> str:
    if isinstance(output, BaseModel):
        return output.model_dump_json(round_trip=True)
    if isinstance(output, str):
        return output
    return json.dumps(output)


def _kek_wrap_aad(key_id: str) -> bytes:
    return f"deep-researcher:kek-wrap:v1:{key_id}".encode()


def _wrapped_dek_cache_key(envelope: dict[str, Any]) -> str:
    pieces = [
        str(envelope.get("wrap", "")),
        str(envelope.get("kid", "")),
        str(envelope.get("wrapped_dek", "")),
        str(envelope.get("wrap_nonce", "")),
        str(envelope.get("wrap_tag", "")),
    ]
    return hashlib.sha256("\0".join(pieces).encode("utf-8")).hexdigest()


def _required_b64url(envelope: dict[str, Any], field_name: str) -> bytes:
    value = envelope.get(field_name)
    if not isinstance(value, str) or not value:
        raise ContentEncryptionInvalidData(f"encrypted job output is missing {field_name}")
    return _b64url_decode(value)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.b64decode((data + padding).encode("ascii"), altchars=b"-_", validate=True)


def _strict_base64_decode(value: str, *, field_name: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ContentEncryptionConfigError(f"{field_name} must be base64 encoded")
    try:
        return _b64url_decode(value.strip())
    except (ValueError, UnicodeEncodeError) as exc:
        raise ContentEncryptionConfigError(f"{field_name} must be base64 encoded") from exc


def _decode_vault_key(value: str, *, field_name: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ContentEncryptionUnavailable(f"{field_name} is missing")
    try:
        return _b64url_decode(value.strip())
    except (ValueError, UnicodeEncodeError) as exc:
        raise ContentEncryptionUnavailable(f"{field_name} is not base64 encoded") from exc


def _parse_non_negative_float(value: str | None, *, default: float, name: str) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ContentEncryptionConfigError(f"{name} must be a non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ContentEncryptionConfigError(f"{name} must be a non-negative number")
    return parsed


def _parse_positive_float(value: str | None, *, default: float, name: str) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ContentEncryptionConfigError(f"{name} must be a positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ContentEncryptionConfigError(f"{name} must be a positive number")
    return parsed


def _secret_fingerprint(value: bytes | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        type_tag = b"bytes\0"
        encoded = value
    else:
        type_tag = b"str\0"
        encoded = value.encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(b"deep-researcher-content-encryption-config\0" + type_tag + encoded).hexdigest()


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _is_auth_failure(exc: Exception) -> bool:
    candidate = _vault_failure_candidate(exc)
    return candidate.__class__.__name__ in _VAULT_AUTH_RETRY_EXCEPTION_NAMES


def _should_retry_vault_failure(exc: Exception) -> bool:
    candidate = _vault_failure_candidate(exc)
    if _is_auth_failure(candidate):
        return True
    if _is_retryable_network_failure(candidate):
        return True

    status_code = _status_code_from_exception(candidate)
    if status_code is not None:
        return status_code == 429 or 500 <= status_code < 600

    exception_name = candidate.__class__.__name__
    if exception_name in _VAULT_NON_RETRYABLE_EXCEPTION_NAMES:
        return False
    return exception_name in _VAULT_RETRYABLE_EXCEPTION_NAMES


def _vault_failure_candidate(exc: Exception) -> Exception:
    if isinstance(exc, ContentEncryptionUnavailable) and isinstance(exc.__cause__, Exception):
        return exc.__cause__
    return exc


def _is_retryable_network_failure(exc: Exception) -> bool:
    try:
        import requests
    except ImportError:
        requests = None
    if requests is not None and isinstance(
        exc,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ),
    ):
        return True

    try:
        import urllib3
    except ImportError:
        urllib3 = None
    if urllib3 is not None and isinstance(
        exc,
        (
            urllib3.exceptions.ConnectTimeoutError,
            urllib3.exceptions.MaxRetryError,
            urllib3.exceptions.NewConnectionError,
            urllib3.exceptions.ReadTimeoutError,
            urllib3.exceptions.TimeoutError,
        ),
    ):
        return True
    return False


def _status_code_from_exception(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)
    response_status_code = getattr(response, "status_code", None)
    if isinstance(response_status_code, int):
        return response_status_code
    return None


def _sleep_before_vault_retry(attempt: int) -> None:
    delay = _VAULT_RETRY_BASE_SECONDS * (2**attempt)
    jitter = random.uniform(0, _VAULT_RETRY_JITTER_SECONDS)
    time.sleep(delay + jitter)


def _zero_bytes(_value: bytes) -> None:
    # Python bytes are immutable, so this is only a lifetime boundary marker.
    return None
