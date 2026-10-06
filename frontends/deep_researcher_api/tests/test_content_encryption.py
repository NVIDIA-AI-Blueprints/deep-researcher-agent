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

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from hvac import exceptions as vault_exceptions
from requests import exceptions as requests_exceptions

from deep_researcher_api.jobs import crypto


def _static_key() -> str:
    return base64.urlsafe_b64encode(bytes(range(32))).decode("ascii")


def _real_vault_env_present() -> bool:
    return all(
        os.environ.get(name)
        for name in (
            "VAULT_ADDR",
            "VAULT_ROLE_ID",
            "VAULT_SECRET_ID",
            "DEEP_RESEARCHER_ENCRYPTION_TRANSIT_KEY",
        )
    )


@pytest.fixture(autouse=True)
def clean_encryption_env(monkeypatch, request):
    if request.node.name == "test_real_vault_transit_round_trip":
        crypto.reset_content_encryption_manager_for_tests()
        yield
        crypto.reset_content_encryption_manager_for_tests()
        return

    for name in (
        "DEEP_RESEARCHER_CONTENT_ENCRYPTION",
        "DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY",
        "DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY_ID",
        "DEEP_RESEARCHER_CONTENT_ENCRYPTION_READINESS_TTL_SECONDS",
        "DEEP_RESEARCHER_CONTENT_ENCRYPTION_DEK_CACHE_TTL_SECONDS",
        "VAULT_ADDR",
        "VAULT_NAMESPACE",
        "VAULT_TRANSIT_MOUNT",
        "VAULT_ROLE_ID",
        "VAULT_SECRET_ID",
        "DEEP_RESEARCHER_ENCRYPTION_TRANSIT_KEY",
        "VAULT_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    crypto.reset_content_encryption_manager_for_tests()
    yield
    crypto.reset_content_encryption_manager_for_tests()


def _enable_static_key(monkeypatch, *, cache_ttl: str | None = None) -> None:
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION", "key")
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY", _static_key())
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY_ID", "test-key")
    if cache_ttl is not None:
        monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_DEK_CACHE_TTL_SECONDS", cache_ttl)
    crypto.reset_content_encryption_manager_for_tests()


def _vault_config() -> crypto.ContentEncryptionConfig:
    return crypto.ContentEncryptionConfig(
        mode="vault",
        vault_addr="https://vault.example.com",
        vault_transit_key="reports",
        vault_role_id="role-id",
        vault_secret_id="secret-id",
    )


def _vault_client_for_tests() -> crypto._VaultTransitClient:
    return crypto._VaultTransitClient(_vault_config())


def _disable_vault_retry_sleep(monkeypatch) -> list[float]:
    sleeps = []
    monkeypatch.setattr(crypto.time, "sleep", sleeps.append)
    monkeypatch.setattr(crypto.random, "uniform", lambda _start, _end: 0)
    return sleeps


def test_content_encryption_defaults_to_off():
    config = crypto.get_content_encryption_config()

    assert config.mode == "off"
    assert config.encrypted is False


def test_config_repr_and_signature_do_not_expose_credentials():
    static_key = b"static-key-credential"
    role_id = "vault-role-credential"
    secret_id = "vault-secret-credential"
    config = crypto.ContentEncryptionConfig(
        mode="vault",
        static_key=static_key,
        vault_role_id=role_id,
        vault_secret_id=secret_id,
    )

    config_repr = repr(config)
    signature_repr = repr(config.signature)

    assert "static_key=" not in config_repr
    assert "vault_role_id=" not in config_repr
    assert "vault_secret_id=" not in config_repr
    for credential in (static_key.decode(), role_id, secret_id):
        assert credential not in config_repr
        assert credential not in signature_repr


@pytest.mark.parametrize(
    ("credential_name", "replacement"),
    [
        ("static_key", b"replacement-static-key"),
        ("vault_role_id", "replacement-role-id"),
        ("vault_secret_id", "replacement-secret-id"),
    ],
)
def test_config_signature_changes_when_credentials_change(credential_name, replacement):
    credentials = {
        "static_key": b"original-static-key",
        "vault_role_id": "original-role-id",
        "vault_secret_id": "original-secret-id",
    }
    original = crypto.ContentEncryptionConfig(mode="vault", **credentials)
    credentials[credential_name] = replacement
    updated = crypto.ContentEncryptionConfig(mode="vault", **credentials)

    assert original.signature != updated.signature


def test_static_key_policy_identity_is_non_secret_and_key_specific():
    static_key = b"a" * crypto.DEK_BYTES
    replacement_key = b"b" * crypto.DEK_BYTES
    original = crypto.ContentEncryptionConfig(mode="key", key_id="reports", static_key=static_key).policy_identity
    replacement = crypto.ContentEncryptionConfig(
        mode="key",
        key_id="reports",
        static_key=replacement_key,
    ).policy_identity

    assert original.mode == "key"
    assert original.key_id == "reports"
    assert original.static_key_fingerprint == crypto._secret_fingerprint(static_key)
    assert original != replacement
    assert static_key.decode() not in repr(original)


def test_vault_policy_identity_changes_with_transit_key_location():
    original = _vault_config().policy_identity
    same_key_with_different_credentials = crypto.ContentEncryptionConfig(
        mode="vault",
        vault_addr="https://vault.example.com",
        vault_transit_key="reports",
        vault_role_id="different-role-id",
        vault_secret_id="different-secret-id",
    ).policy_identity
    replacement = crypto.ContentEncryptionConfig(
        mode="vault",
        vault_addr="https://other-vault.example.com",
        vault_transit_key="reports",
    ).policy_identity

    assert original == same_key_with_different_credentials
    assert original != replacement
    assert not hasattr(original, "vault_role_id")
    assert not hasattr(original, "vault_secret_id")


def test_worker_policy_identity_is_required_and_accepts_an_exact_match():
    expected = crypto.get_content_encryption_policy_identity()

    crypto.require_content_encryption_policy(expected)
    with pytest.raises(crypto.ContentEncryptionPolicyMismatch, match="does not match"):
        crypto.require_content_encryption_policy(None)


def test_secret_fingerprint_distinguishes_bytes_from_strings():
    assert crypto._secret_fingerprint(b"same-value") != crypto._secret_fingerprint("same-value")


def test_secret_fingerprint_supports_surrogate_escaped_strings():
    credential = "vault-role-\udcff"

    fingerprint = crypto._secret_fingerprint(credential)

    assert fingerprint == crypto._secret_fingerprint(credential)
    assert credential not in fingerprint


@pytest.mark.parametrize(
    ("credential_name", "replacement"),
    [
        ("static_key", b"replacement-static-key"),
        ("vault_role_id", "replacement-role-id"),
        ("vault_secret_id", "replacement-secret-id"),
    ],
)
def test_manager_cache_reuses_identical_config_and_recreates_for_changed_credentials(
    monkeypatch, credential_name, replacement
):
    credentials = {
        "static_key": b"original-static-key",
        "vault_role_id": "original-role-id",
        "vault_secret_id": "original-secret-id",
    }
    configs = iter(
        [
            crypto.ContentEncryptionConfig(mode="vault", **credentials),
            crypto.ContentEncryptionConfig(mode="vault", **credentials),
            crypto.ContentEncryptionConfig(mode="vault", **{**credentials, credential_name: replacement}),
        ]
    )
    monkeypatch.setattr(crypto, "get_content_encryption_config", lambda: next(configs))

    original_manager = crypto.get_content_encryption_manager()
    reused_manager = crypto.get_content_encryption_manager()
    replacement_manager = crypto.get_content_encryption_manager()

    assert reused_manager is original_manager
    assert replacement_manager is not original_manager


@pytest.mark.parametrize(
    ("env_name", "mode"),
    [
        ("DEEP_RESEARCHER_CONTENT_ENCRYPTION_READINESS_TTL_SECONDS", "off"),
        ("DEEP_RESEARCHER_CONTENT_ENCRYPTION_DEK_CACHE_TTL_SECONDS", "off"),
        ("VAULT_TIMEOUT_SECONDS", "vault"),
    ],
)
@pytest.mark.parametrize("non_finite_value", ["nan", "inf", "-inf"])
def test_non_finite_numeric_config_is_rejected(monkeypatch, env_name, mode, non_finite_value):
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION", mode)
    monkeypatch.setenv(env_name, non_finite_value)
    if mode == "vault":
        monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
        monkeypatch.setenv("VAULT_ROLE_ID", "role-id")
        monkeypatch.setenv("VAULT_SECRET_ID", "secret-id")
        monkeypatch.setenv("DEEP_RESEARCHER_ENCRYPTION_TRANSIT_KEY", "reports")

    with pytest.raises(crypto.ContentEncryptionConfigError):
        crypto.get_content_encryption_config()


def test_static_key_envelope_round_trip(monkeypatch):
    _enable_static_key(monkeypatch)

    cipher = crypto.create_job_content_cipher("job-1")
    stored = cipher.encrypt_output_json('{"report":"secret report"}')

    assert stored.startswith(crypto.ENVELOPE_PREFIX)
    assert "secret report" not in stored
    assert crypto.read_job_output("job-1", stored) == {"report": "secret report"}


def test_aad_mismatch_fails(monkeypatch):
    _enable_static_key(monkeypatch)

    stored = crypto.create_job_content_cipher("job-1").encrypt_output_json('{"report":"secret"}')

    with pytest.raises(crypto.ContentEncryptionInvalidData):
        crypto.read_job_output("job-2", stored)


def test_tamper_fails(monkeypatch):
    _enable_static_key(monkeypatch)

    stored = crypto.create_job_content_cipher("job-1").encrypt_output_json('{"report":"secret"}')
    envelope = crypto.decode_envelope(stored)
    padding = "=" * (-len(envelope["ciphertext"]) % 4)
    ciphertext = bytearray(base64.urlsafe_b64decode(envelope["ciphertext"] + padding))
    ciphertext[0] ^= 0x01
    envelope["ciphertext"] = base64.urlsafe_b64encode(bytes(ciphertext)).decode("ascii").rstrip("=")
    tampered = crypto.encode_envelope(envelope)

    with pytest.raises(crypto.ContentEncryptionInvalidData):
        crypto.read_job_output("job-1", tampered)


def test_non_ascii_envelope_is_classified_as_invalid_data():
    with pytest.raises(crypto.ContentEncryptionInvalidData, match="envelope is malformed"):
        crypto.decode_envelope(f"{crypto.ENVELOPE_PREFIX}\N{LATIN SMALL LETTER E WITH ACUTE}")


def test_plaintext_job_output_is_rejected_in_encrypted_mode(monkeypatch):
    _enable_static_key(monkeypatch)

    with pytest.raises(crypto.ContentEncryptionPlaintextViolation):
        crypto.read_job_output("job-1", '{"report":"plaintext"}')


def test_off_mode_preserves_current_behavior_and_does_not_decrypt_envelopes(monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION", "off")
    crypto.reset_content_encryption_manager_for_tests()

    assert crypto.read_job_output("job-1", '{"report":"plaintext"}') == {"report": "plaintext"}
    assert crypto.read_job_output("job-1", "aiqenc:not-json") == "aiqenc:not-json"


@pytest.mark.asyncio
async def test_async_content_encryption_wrappers_run_off_event_loop(monkeypatch):
    loop_thread = threading.get_ident()
    call_threads = {}

    def fake_validate_startup():
        call_threads["startup"] = threading.get_ident()
        return crypto.ContentEncryptionReadiness(mode="off", ready=True)

    def fake_health():
        call_threads["health"] = threading.get_ident()
        return crypto.ContentEncryptionReadiness(mode="off", ready=True)

    def fake_require_ready():
        call_threads["readiness"] = threading.get_ident()

    def fake_read_output(job_id, stored_output):
        call_threads["read_output"] = threading.get_ident()
        return {"job_id": job_id, "stored_output": stored_output}

    monkeypatch.setattr(crypto, "validate_content_encryption_startup", fake_validate_startup)
    monkeypatch.setattr(crypto, "get_content_encryption_health", fake_health)
    monkeypatch.setattr(crypto, "require_content_encryption_ready_for_submission", fake_require_ready)
    monkeypatch.setattr(crypto, "read_job_output", fake_read_output)

    assert await crypto.validate_content_encryption_startup_async() == crypto.ContentEncryptionReadiness(
        mode="off", ready=True
    )
    assert await crypto.get_content_encryption_health_async() == crypto.ContentEncryptionReadiness(
        mode="off", ready=True
    )
    await crypto.require_content_encryption_ready_for_submission_async()
    assert await crypto.read_job_output_async("job-1", "stored") == {"job_id": "job-1", "stored_output": "stored"}

    assert set(call_threads) == {"startup", "health", "readiness", "read_output"}
    assert all(thread_id != loop_thread for thread_id in call_threads.values())


@pytest.mark.asyncio
async def test_async_content_encryption_wrapper_does_not_block_event_loop(monkeypatch):
    import time

    def slow_read_output(_job_id, _stored_output):
        time.sleep(0.1)
        return {"report": "secret"}

    monkeypatch.setattr(crypto, "read_job_output", slow_read_output)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while ticks < 3:
            await asyncio.sleep(0.01)
            ticks += 1

    read_task = asyncio.create_task(crypto.read_job_output_async("job-1", "stored"))
    ticker_task = asyncio.create_task(ticker())

    assert await read_task == {"report": "secret"}
    await ticker_task
    assert ticks >= 3


@pytest.mark.asyncio
async def test_update_job_output_encrypts_entire_payload(monkeypatch):
    _enable_static_key(monkeypatch)
    cipher = crypto.create_job_content_cipher("job-1")
    job_store = SimpleNamespace(update_status=AsyncMock())

    await crypto.update_job_output(job_store, "job-1", "success", output={"report": "secret"}, cipher=cipher)

    stored = job_store.update_status.await_args.kwargs["output"]
    assert stored.startswith(crypto.ENVELOPE_PREFIX)
    assert "secret" not in stored
    assert crypto.read_job_output("job-1", stored) == {"report": "secret"}


@pytest.mark.asyncio
async def test_sqlite_job_store_persists_encrypted_output(monkeypatch, tmp_path):
    from nat.front_ends.fastapi.async_jobs.job_store import Base
    from nat.front_ends.fastapi.async_jobs.job_store import JobStatus
    from nat.front_ends.fastapi.async_jobs.job_store import JobStore
    from nat.front_ends.fastapi.async_jobs.job_store import get_db_engine

    _enable_static_key(monkeypatch)
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}"
    engine = get_db_engine(db_url, use_async=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    job_store = JobStore(scheduler_address="tcp://localhost:8786", db_engine=engine)
    await job_store._create_job(job_id="job-1")
    cipher = crypto.create_job_content_cipher("job-1")

    await crypto.update_job_output(job_store, "job-1", JobStatus.SUCCESS, output={"report": "secret"}, cipher=cipher)
    job = await job_store.get_job("job-1")

    assert job.output.startswith(crypto.ENVELOPE_PREFIX)
    assert "secret" not in job.output
    assert crypto.read_job_output("job-1", job.output) == {"report": "secret"}
    await engine.dispose()


def test_static_key_invalid_config_fails_hard(monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION", "key")
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY", "not a 32 byte base64 key")

    with pytest.raises(crypto.ContentEncryptionConfigError):
        crypto.get_content_encryption_config()


def test_large_report_round_trip(monkeypatch):
    _enable_static_key(monkeypatch)
    report = "large report\n" * 100_000

    stored = crypto.create_job_content_cipher("job-large").encrypt_output_json(json.dumps({"report": report}))

    assert crypto.read_job_output("job-large", stored)["report"] == report


def test_dek_cache_reuses_unwrapped_dek(monkeypatch):
    _enable_static_key(monkeypatch)
    manager = crypto.get_content_encryption_manager()
    stored = manager.create_job_cipher("job-1").encrypt_output_json('{"report":"secret"}')
    calls = 0
    original = manager._unwrap_dek_with_static_key

    def counted_unwrap(envelope):
        nonlocal calls
        calls += 1
        return original(envelope)

    monkeypatch.setattr(manager, "_unwrap_dek_with_static_key", counted_unwrap)

    assert manager.decrypt_job_output_text("job-1", stored) == '{"report":"secret"}'
    assert manager.decrypt_job_output_text("job-1", stored) == '{"report":"secret"}'
    assert calls == 1


def test_dek_cache_can_be_disabled(monkeypatch):
    _enable_static_key(monkeypatch, cache_ttl="0")
    manager = crypto.get_content_encryption_manager()
    stored = manager.create_job_cipher("job-1").encrypt_output_json('{"report":"secret"}')
    calls = 0
    original = manager._unwrap_dek_with_static_key

    def counted_unwrap(envelope):
        nonlocal calls
        calls += 1
        return original(envelope)

    monkeypatch.setattr(manager, "_unwrap_dek_with_static_key", counted_unwrap)

    assert manager.decrypt_job_output_text("job-1", stored) == '{"report":"secret"}'
    assert manager.decrypt_job_output_text("job-1", stored) == '{"report":"secret"}'
    assert calls == 2


def test_dek_cache_evicts_lru_when_max_entries_is_exceeded():
    cache = crypto._DEKCache(ttl_seconds=60, max_entries=2)

    cache.put("first", b"1" * 32)
    cache.put("second", b"2" * 32)
    assert cache.get("first") == b"1" * 32
    cache.put("third", b"3" * 32)

    assert cache.get("second") is None
    assert cache.get("first") == b"1" * 32
    assert cache.get("third") == b"3" * 32


def test_dek_cache_is_thread_safe_under_concurrent_access():
    cache = crypto._DEKCache(ttl_seconds=60, max_entries=8)

    def exercise_cache(worker_id: int) -> None:
        for index in range(500):
            cache_key = f"key-{(worker_id + index) % 16}"
            cache.put(cache_key, bytes([index % 256]) * crypto.DEK_BYTES)
            cache.get(cache_key)
            cache.get(f"key-{index % 16}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(exercise_cache, worker_id) for worker_id in range(8)]
        for future in as_completed(futures):
            future.result()

    cache.put("final", b"f" * crypto.DEK_BYTES)
    assert cache.get("final") == b"f" * crypto.DEK_BYTES


def test_vault_client_initialization_is_thread_safe(monkeypatch):
    clients = []
    login_calls = 0
    calls_lock = threading.Lock()

    class FakeAppRole:
        def login(self, *, role_id, secret_id):
            nonlocal login_calls
            with calls_lock:
                login_calls += 1
            assert role_id == "role-id"
            assert secret_id == "secret-id"
            clients[0].authenticated = True

    class FakeAuth:
        def __init__(self):
            self.approle = FakeAppRole()

    class FakeClient:
        def __init__(self, *, url, namespace, timeout):
            assert url == "https://vault.example.com"
            assert namespace is None
            assert timeout == crypto.DEFAULT_VAULT_TIMEOUT_SECONDS
            self.auth = FakeAuth()
            self.authenticated = False
            clients.append(self)

        def is_authenticated(self):
            return self.authenticated

    monkeypatch.setitem(sys.modules, "hvac", SimpleNamespace(Client=FakeClient))
    vault = _vault_client_for_tests()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(vault._get_client) for _ in range(16)]
        results = [future.result() for future in as_completed(futures)]

    assert len(clients) == 1
    assert login_calls == 1
    assert all(result is clients[0] for result in results)


def test_vault_retry_retries_transient_network_error(monkeypatch):
    sleeps = _disable_vault_retry_sleep(monkeypatch)
    vault = _vault_client_for_tests()
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise requests_exceptions.Timeout("request timed out")
        return {"data": {"ok": True}}

    assert vault._with_retry(operation, operation="unit") == {"data": {"ok": True}}
    assert calls == 2
    assert len(sleeps) == 1


def test_vault_retry_retries_transient_vault_error(monkeypatch):
    sleeps = _disable_vault_retry_sleep(monkeypatch)
    vault = _vault_client_for_tests()
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise vault_exceptions.InternalServerError("vault server error")
        return {"data": {"ok": True}}

    assert vault._with_retry(operation, operation="unit") == {"data": {"ok": True}}
    assert calls == 2
    assert len(sleeps) == 1


def test_vault_retry_fails_after_bounded_transient_attempts(monkeypatch):
    sleeps = _disable_vault_retry_sleep(monkeypatch)
    vault = _vault_client_for_tests()
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        raise vault_exceptions.RateLimitExceeded("vault is busy")

    with pytest.raises(crypto.ContentEncryptionUnavailable, match="vault_unit_failed"):
        vault._with_retry(operation, operation="unit")

    assert calls == crypto._VAULT_ATTEMPTS
    assert len(sleeps) == crypto._VAULT_ATTEMPTS - 1


def test_vault_retry_reauthenticates_after_unauthorized(monkeypatch):
    sleeps = _disable_vault_retry_sleep(monkeypatch)
    vault = _vault_client_for_tests()
    calls = 0
    login_forces = []

    def login(*, force):
        login_forces.append(force)

    monkeypatch.setattr(vault, "_login", login)

    def operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise vault_exceptions.Unauthorized("token expired")
        return {"data": {"ok": True}}

    assert vault._with_retry(operation, operation="unit") == {"data": {"ok": True}}
    assert calls == 2
    assert login_forces == [True]
    assert len(sleeps) == 1


@pytest.mark.parametrize("exc_cls", [vault_exceptions.Forbidden, vault_exceptions.InvalidRequest])
def test_vault_retry_does_not_retry_permission_or_invalid_request_errors(monkeypatch, caplog, exc_cls):
    sleeps = _disable_vault_retry_sleep(monkeypatch)
    vault = _vault_client_for_tests()
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        raise exc_cls("secret-value")

    with caplog.at_level("WARNING", logger=crypto.__name__):
        with pytest.raises(crypto.ContentEncryptionUnavailable, match="vault_unit_failed"):
            vault._with_retry(operation, operation="unit")

    assert calls == 1
    assert sleeps == []
    assert exc_cls.__name__ in caplog.text
    assert "secret-value" not in caplog.text


def test_vault_missing_required_config_fails_startup(monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION", "vault")
    monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
    monkeypatch.setenv("VAULT_ROLE_ID", "role-id")
    monkeypatch.setenv("VAULT_SECRET_ID", "secret-id")

    with pytest.raises(crypto.ContentEncryptionConfigError, match="DEEP_RESEARCHER_ENCRYPTION_TRANSIT_KEY"):
        crypto.get_content_encryption_config()


def test_vault_operational_failure_starts_unhealthy_and_uses_readiness_cache(monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION", "vault")
    monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
    monkeypatch.setenv("VAULT_ROLE_ID", "role-id")
    monkeypatch.setenv("VAULT_SECRET_ID", "secret-id")
    monkeypatch.setenv("DEEP_RESEARCHER_ENCRYPTION_TRANSIT_KEY", "reports")
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_READINESS_TTL_SECONDS", "60")
    calls = 0

    class FailingVault:
        def __init__(self, _config):
            pass

        def generate_data_key(self, *, operation):
            nonlocal calls
            calls += 1
            raise crypto.ContentEncryptionUnavailable("vault down")

    monkeypatch.setattr(crypto, "_VaultTransitClient", FailingVault)
    crypto.reset_content_encryption_manager_for_tests()

    startup = crypto.validate_content_encryption_startup()
    health = crypto.get_content_encryption_health()

    assert startup.ready is False
    assert health.ready is False
    assert health.encrypt_ready is False
    assert health.decrypt_ready is False
    assert health.reason == "vault_generate_unavailable"
    assert calls == 1


def test_vault_readiness_requires_generate_and_unwrap(monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION", "vault")
    monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
    monkeypatch.setenv("VAULT_ROLE_ID", "role-id")
    monkeypatch.setenv("VAULT_SECRET_ID", "secret-id")
    monkeypatch.setenv("DEEP_RESEARCHER_ENCRYPTION_TRANSIT_KEY", "reports")
    calls = {"generate": 0, "unwrap": 0}
    dek = b"d" * crypto.DEK_BYTES

    class ReadyVault:
        def __init__(self, _config):
            pass

        def generate_data_key(self, *, operation):
            calls["generate"] += 1
            assert operation == "api_startup_readiness_generate"
            return dek, crypto.WrappedDEK(wrap="vault", kid="transit/reports", wrapped_dek="vault:v1:dek")

        def unwrap_dek(self, wrapped_dek, *, operation):
            calls["unwrap"] += 1
            assert wrapped_dek == "vault:v1:dek"
            assert operation == "api_startup_readiness_decrypt"
            return dek

    monkeypatch.setattr(crypto, "_VaultTransitClient", ReadyVault)
    crypto.reset_content_encryption_manager_for_tests()

    readiness = crypto.validate_content_encryption_startup()

    assert readiness.ready is True
    assert readiness.encrypt_ready is True
    assert readiness.decrypt_ready is True
    assert calls == {"generate": 1, "unwrap": 1}


def test_vault_readiness_fails_when_unwrap_is_denied(monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION", "vault")
    monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
    monkeypatch.setenv("VAULT_ROLE_ID", "role-id")
    monkeypatch.setenv("VAULT_SECRET_ID", "secret-id")
    monkeypatch.setenv("DEEP_RESEARCHER_ENCRYPTION_TRANSIT_KEY", "reports")

    class DecryptDeniedVault:
        def __init__(self, _config):
            pass

        def generate_data_key(self, *, operation):
            return b"d" * crypto.DEK_BYTES, crypto.WrappedDEK(
                wrap="vault", kid="transit/reports", wrapped_dek="vault:v1:dek"
            )

        def unwrap_dek(self, wrapped_dek, *, operation):
            raise crypto.ContentEncryptionUnavailable("decrypt denied")

    monkeypatch.setattr(crypto, "_VaultTransitClient", DecryptDeniedVault)
    crypto.reset_content_encryption_manager_for_tests()

    readiness = crypto.validate_content_encryption_startup()

    assert readiness.ready is False
    assert readiness.encrypt_ready is True
    assert readiness.decrypt_ready is False
    assert readiness.reason == "vault_decrypt_unavailable"
    assert readiness.exception_type == "ContentEncryptionUnavailable"


def test_vault_readiness_fails_when_unwrapped_dek_differs(monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION", "vault")
    monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
    monkeypatch.setenv("VAULT_ROLE_ID", "role-id")
    monkeypatch.setenv("VAULT_SECRET_ID", "secret-id")
    monkeypatch.setenv("DEEP_RESEARCHER_ENCRYPTION_TRANSIT_KEY", "reports")

    class MismatchedVault:
        def __init__(self, _config):
            pass

        def generate_data_key(self, *, operation):
            return b"d" * crypto.DEK_BYTES, crypto.WrappedDEK(
                wrap="vault", kid="transit/reports", wrapped_dek="vault:v1:dek"
            )

        def unwrap_dek(self, wrapped_dek, *, operation):
            return b"e" * crypto.DEK_BYTES

    monkeypatch.setattr(crypto, "_VaultTransitClient", MismatchedVault)
    crypto.reset_content_encryption_manager_for_tests()

    readiness = crypto.validate_content_encryption_startup()

    assert readiness.ready is False
    assert readiness.encrypt_ready is True
    assert readiness.decrypt_ready is False
    assert readiness.reason == "vault_readiness_dek_mismatch"


def test_vault_readiness_cache_avoids_repeated_generate_and_unwrap(monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION", "vault")
    monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
    monkeypatch.setenv("VAULT_ROLE_ID", "role-id")
    monkeypatch.setenv("VAULT_SECRET_ID", "secret-id")
    monkeypatch.setenv("DEEP_RESEARCHER_ENCRYPTION_TRANSIT_KEY", "reports")
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_READINESS_TTL_SECONDS", "60")
    calls = {"generate": 0, "unwrap": 0}
    dek = b"d" * crypto.DEK_BYTES

    class ReadyVault:
        def __init__(self, _config):
            pass

        def generate_data_key(self, *, operation):
            calls["generate"] += 1
            return dek, crypto.WrappedDEK(wrap="vault", kid="transit/reports", wrapped_dek="vault:v1:dek")

        def unwrap_dek(self, wrapped_dek, *, operation):
            calls["unwrap"] += 1
            return dek

    monkeypatch.setattr(crypto, "_VaultTransitClient", ReadyVault)
    crypto.reset_content_encryption_manager_for_tests()

    startup = crypto.validate_content_encryption_startup()
    health = crypto.get_content_encryption_health()

    assert startup.ready is True
    assert health.ready is True
    assert calls == {"generate": 1, "unwrap": 1}


def test_vault_readiness_rechecks_when_cache_is_stale(monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION", "vault")
    monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
    monkeypatch.setenv("VAULT_ROLE_ID", "role-id")
    monkeypatch.setenv("VAULT_SECRET_ID", "secret-id")
    monkeypatch.setenv("DEEP_RESEARCHER_ENCRYPTION_TRANSIT_KEY", "reports")
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_READINESS_TTL_SECONDS", "0")
    calls = {"generate": 0, "unwrap": 0}
    dek = b"d" * crypto.DEK_BYTES

    class ReadyVault:
        def __init__(self, _config):
            pass

        def generate_data_key(self, *, operation):
            calls["generate"] += 1
            return dek, crypto.WrappedDEK(wrap="vault", kid="transit/reports", wrapped_dek="vault:v1:dek")

        def unwrap_dek(self, wrapped_dek, *, operation):
            calls["unwrap"] += 1
            return dek

    monkeypatch.setattr(crypto, "_VaultTransitClient", ReadyVault)
    crypto.reset_content_encryption_manager_for_tests()

    crypto.validate_content_encryption_startup()
    crypto.get_content_encryption_health()

    assert calls == {"generate": 2, "unwrap": 2}


@pytest.mark.skipif(not _real_vault_env_present(), reason="real Vault Transit credentials are not configured")
def test_real_vault_transit_round_trip(monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION", "vault")
    crypto.reset_content_encryption_manager_for_tests()

    readiness = crypto.validate_content_encryption_startup()
    stored = crypto.create_job_content_cipher("job-real-vault").encrypt_output_json('{"report":"secret"}')

    assert readiness.ready is True
    assert readiness.encrypt_ready is True
    assert readiness.decrypt_ready is True
    assert stored.startswith(crypto.ENVELOPE_PREFIX)
    assert "secret" not in stored
    assert crypto.read_job_output("job-real-vault", stored) == {"report": "secret"}
