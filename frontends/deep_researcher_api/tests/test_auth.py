# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for deep_researcher_api.auth (middleware, JWT validator, base types)."""

from __future__ import annotations

import json
import os
import sys
import types
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt import PyJWK
from jwt.algorithms import RSAAlgorithm

from deep_researcher_api.auth import AuthMiddleware
from deep_researcher_api.auth import JWTValidator
from deep_researcher_api.auth import TokenExpiredError
from deep_researcher_api.auth import TokenInvalidError
from deep_researcher_api.auth import TokenValidator
from deep_researcher_api.auth import get_current_user
from deep_researcher_api.auth import middleware as middleware_module
from deep_researcher_api.auth.errors import AuthError

# ---------------------------------------------------------------------------
# TokenValidator (ABC)
# ---------------------------------------------------------------------------


class TestTokenValidator:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            TokenValidator()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# JWTValidator
# ---------------------------------------------------------------------------


def _rsa_private_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _signing_jwk_from_private(priv):
    return PyJWK.from_dict(json.loads(RSAAlgorithm.to_jwk(priv.public_key())))


class TestJWTValidatorInit:
    def test_strips_trailing_slash_on_issuer(self) -> None:
        v = JWTValidator("https://issuer.example/")
        assert v.issuer_url == "https://issuer.example"

    def test_is_token_validator(self) -> None:
        v = JWTValidator("https://issuer.example/")
        assert isinstance(v, TokenValidator)


class TestJWTValidatorCanHandle:
    def test_true_for_compact_jwt_shape(self) -> None:
        v = JWTValidator("https://issuer.example/")
        assert v.can_handle("eyJ.a.b")

    def test_false_for_non_jwt(self) -> None:
        v = JWTValidator("https://issuer.example/")
        assert not v.can_handle("opaque-api-key")
        assert not v.can_handle("a.b")
        assert not v.can_handle("")


class TestJWTValidatorValidate:
    @pytest.mark.asyncio
    async def test_returns_claims_when_signing_key_matches(self) -> None:
        priv = _rsa_private_key()
        jwk = _signing_jwk_from_private(priv)
        issuer = "https://issuer.example"
        now = datetime.now(UTC)
        claims = {
            "sub": "user-1",
            "iss": issuer,
            "aud": "my-api",
            "exp": now + timedelta(hours=1),
            "iat": now,
        }
        token = jwt.encode(claims, priv, algorithm="RS256", headers={"kid": "k1"})

        validator = JWTValidator(issuer, audience="my-api", jwks_uri="https://unused/jwks")
        with patch.object(validator, "_get_signing_key", return_value=jwk):
            out, error = await validator.validate(token)

        assert out is not None
        assert error is None
        assert out["sub"] == "user-1"
        assert out["type"] == "jwt"
        assert out["token"] == token
        assert out["skip_clarifier"] is False
        assert out["iss"] == issuer

    @pytest.mark.asyncio
    async def test_returns_none_when_no_signing_key(self) -> None:
        validator = JWTValidator("https://issuer.example", jwks_uri="https://unused/jwks")
        with patch.object(validator, "_get_signing_key", return_value=None):
            user, error = await validator.validate("any.token.here")
            assert user is None
            assert error == "token_invalid"

    @pytest.mark.asyncio
    async def test_returns_none_on_expired_token(self) -> None:
        priv = _rsa_private_key()
        jwk = _signing_jwk_from_private(priv)
        issuer = "https://issuer.example"
        now = datetime.now(UTC)
        claims = {
            "sub": "user-1",
            "iss": issuer,
            "exp": now - timedelta(hours=1),
            "iat": now - timedelta(hours=2),
        }
        token = jwt.encode(claims, priv, algorithm="RS256")

        validator = JWTValidator(issuer, jwks_uri="https://unused/jwks")
        with patch.object(validator, "_get_signing_key", return_value=jwk):
            user, error = await validator.validate(token)
            assert user is None
            assert error == "token_expired"

    @pytest.mark.asyncio
    async def test_skips_audience_when_not_configured(self) -> None:
        priv = _rsa_private_key()
        jwk = _signing_jwk_from_private(priv)
        issuer = "https://issuer.example"
        now = datetime.now(UTC)
        claims = {
            "sub": "user-2",
            "iss": issuer,
            "exp": now + timedelta(hours=1),
            "iat": now,
        }
        token = jwt.encode(claims, priv, algorithm="RS256")

        validator = JWTValidator(issuer, audience=None, jwks_uri="https://unused/jwks")
        with patch.object(validator, "_get_signing_key", return_value=jwk):
            out, error = await validator.validate(token)
        assert out is not None and out["sub"] == "user-2"
        assert error is None
        assert out["type"] == "jwt"
        assert out["token"] == token
        assert out["skip_clarifier"] is False


class TestJWTValidatorGetSigningKey:
    def test_fetches_oidc_config_when_jwks_uri_missing(self) -> None:
        validator = JWTValidator("https://issuer.example")
        validator._jwks_uri = None
        fake_jwk = MagicMock()
        with patch.object(validator, "_fetch_oidc_config", return_value={"jwks_uri": "https://issuer/jwks"}):
            with patch.object(
                validator,
                "_fetch_jwks_keys",
                return_value=[("kid-a", fake_jwk)],
            ):
                with patch("jwt.get_unverified_header", return_value={}):
                    key = validator._get_signing_key("header.payload.sig")
        assert key is fake_jwk
        assert validator._jwks_uri == "https://issuer/jwks"

    def test_matches_key_by_kid(self) -> None:
        import time

        validator = JWTValidator("https://issuer.example", jwks_uri="https://issuer/jwks")
        k1, k2 = MagicMock(), MagicMock()
        validator._cached_keys = [("kid-1", k1), ("kid-2", k2)]
        validator._jwks_keys_fetched_at = time.monotonic()  # mark as freshly fetched
        validator._jwks_cache_ttl = 999999.0
        with patch("jwt.get_unverified_header", return_value={"kid": "kid-2"}):
            assert validator._get_signing_key("t") is k2

    def test_adds_use_sig_when_key_omits_use(self) -> None:
        validator = JWTValidator("https://issuer.example", jwks_uri="https://issuer/jwks")
        jwks_body = json.dumps({"keys": [{"kty": "RSA", "kid": "x", "n": "abc", "e": "AQAB"}]}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = jwks_body
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_resp
        mock_cm.__exit__.return_value = None
        with patch("urllib.request.urlopen", return_value=mock_cm):
            with patch("jwt.PyJWK.from_dict") as mock_from_dict:
                mock_from_dict.side_effect = Exception("bad key material")
                keys = validator._fetch_jwks_keys()
        assert keys == []
        call_kw = mock_from_dict.call_args[0][0]
        assert call_kw.get("use") == "sig"


# ---------------------------------------------------------------------------
# get_current_user / AuthMiddleware
# ---------------------------------------------------------------------------


def _http_scope(
    path: str,
    *,
    host: bytes = b"internal.local",
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> dict:
    headers: list[tuple[bytes, bytes]] = [(b"host", host)]
    if extra_headers:
        headers.extend(extra_headers)
    return {
        "type": "http",
        "asgi": {"spec_version": "2.0", "version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "scheme": "http",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 80),
    }


@pytest.fixture
def capture_asgi():
    """Inner ASGI app that records scope state and emits a minimal response."""
    state: dict = {}

    async def app(scope, receive, send):
        state["user"] = scope.get("state", {}).get("user")
        state["scope"] = scope
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"text/plain"]],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})

    return app, state


@pytest.fixture
def external_host() -> bytes:
    return b"api.public.example"


class TestGetCurrentUser:
    def test_returns_value_from_contextvar_when_set(self) -> None:
        token = middleware_module._current_user.set({"type": "jwt", "skip_clarifier": False})
        try:
            assert get_current_user() == {"type": "jwt", "skip_clarifier": False}
        finally:
            middleware_module._current_user.reset(token)


class TestAuthMiddlewareExternalPaths:
    @pytest.mark.asyncio
    async def test_disallowed_path_returns_404(self, capture_asgi, external_host):
        app, state = capture_asgi
        mw = AuthMiddleware(app, require_auth=False, external_hostnames={external_host.decode()})
        messages: list[dict] = []

        async def send(msg):
            messages.append(msg)

        scope = _http_scope("/v1/secret/admin", host=external_host)
        await mw(scope, AsyncMock(), send)

        assert state.get("user") is None
        assert messages[0]["type"] == "http.response.start"
        assert messages[0]["status"] == 404
        body = json.loads(messages[1]["body"].decode())
        assert body["detail"] == "Not found"

    @pytest.mark.asyncio
    async def test_allowed_prefix_path(self, capture_asgi, external_host):
        app, state = capture_asgi
        mw = AuthMiddleware(app, require_auth=False, external_hostnames={external_host.decode()})
        messages: list[dict] = []

        async def send(msg):
            messages.append(msg)

        scope = _http_scope("/v1/jobs/async/job/job-99/stream", host=external_host)
        await mw(scope, AsyncMock(), send)

        assert state["user"]["type"] == "anonymous"
        assert messages[0]["status"] == 200


class TestAuthMiddlewareAuthFlow:
    @pytest.mark.asyncio
    async def test_require_auth_missing_token_401(self, capture_asgi, external_host):
        app, _state = capture_asgi
        mw = AuthMiddleware(app, validators=[], require_auth=True, external_hostnames={external_host.decode()})
        messages: list[dict] = []

        async def send(msg):
            messages.append(msg)

        scope = _http_scope("/chat", host=external_host)
        await mw(scope, AsyncMock(), send)

        assert messages[0]["status"] == 401
        assert json.loads(messages[1]["body"].decode())["detail"] == "Missing auth token"

    @pytest.mark.asyncio
    async def test_require_auth_invalid_token_401(self, capture_asgi, external_host):
        app, _state = capture_asgi
        mock_v = MagicMock()
        mock_v.can_handle.return_value = True
        mock_v.validate = AsyncMock(return_value=(None, "token_invalid"))
        mw = AuthMiddleware(
            app,
            validators=[mock_v],
            require_auth=True,
            external_hostnames={external_host.decode()},
        )
        messages: list[dict] = []

        async def send(msg):
            messages.append(msg)

        scope = _http_scope(
            "/chat",
            host=external_host,
            extra_headers=[(b"authorization", b"Bearer badtoken")],
        )
        await mw(scope, AsyncMock(), send)

        assert messages[0]["status"] == 401
        body = json.loads(messages[1]["body"].decode())
        assert body["error"] == "token_invalid"
        assert body["detail"] == "Invalid auth token"

    @pytest.mark.asyncio
    async def test_require_auth_success_sets_user_and_headless_flag(self, capture_asgi, external_host):
        app, state = capture_asgi
        user = {"type": "oidc", "sub": "u1", "token": "t"}
        mock_v = MagicMock()
        mock_v.can_handle.return_value = True
        mock_v.validate = AsyncMock(return_value=(user, None))
        mw = AuthMiddleware(
            app,
            validators=[mock_v],
            require_auth=True,
            external_hostnames={external_host.decode()},
        )
        messages: list[dict] = []

        async def send(msg):
            messages.append(msg)

        scope = _http_scope(
            "/chat",
            host=external_host,
            extra_headers=[
                (b"authorization", b"Bearer good"),
                (b"x-deep-researcher-mode", b"headless"),
            ],
        )
        await mw(scope, AsyncMock(), send)

        assert messages[0]["status"] == 200
        assert state["user"]["skip_clarifier"] is True

    @pytest.mark.asyncio
    async def test_first_validator_wins(self, capture_asgi, external_host):
        app, state = capture_asgi
        v1 = MagicMock()
        v1.can_handle.return_value = False
        v1.validate = AsyncMock()
        v2 = MagicMock()
        v2.can_handle.return_value = True
        v2.validate = AsyncMock(return_value=({"type": "x", "token": "z"}, None))
        mw = AuthMiddleware(
            app,
            validators=[v1, v2],
            require_auth=True,
            external_hostnames={external_host.decode()},
        )

        async def send(msg):
            pass  # not used — success path hits app

        scope = _http_scope(
            "/chat",
            host=external_host,
            extra_headers=[(b"authorization", b"Bearer t")],
        )
        await mw(scope, AsyncMock(), send)

        v1.validate.assert_not_awaited()
        v2.validate.assert_awaited_once()
        assert state["user"]["type"] == "x"


class TestAuthMiddlewareInternal:
    @pytest.mark.asyncio
    async def test_internal_validates_bearer_jwt_when_validator_accepts(self, capture_asgi):
        app, state = capture_asgi
        mock_v = MagicMock()
        mock_v.can_handle.return_value = True
        mock_v.validate = AsyncMock(return_value={"type": "oidc", "sub": "user-1", "token": "good"})
        mw = AuthMiddleware(app, validators=[mock_v], require_auth=True, external_hostnames=set())

        async def send(msg):
            pass

        scope = _http_scope(
            "/any/path",
            extra_headers=[(b"authorization", b"Bearer eyJhbGciOiJIUzI1NiJ9.x.y")],
        )
        await mw(scope, AsyncMock(), send)

        assert state["user"]["type"] == "oidc"
        assert state["user"]["sub"] == "user-1"

    @pytest.mark.asyncio
    async def test_internal_idtoken_cookie_stays_unverified_without_valid_identity(self, capture_asgi):
        app, state = capture_asgi
        mock_v = MagicMock()
        mock_v.can_handle.return_value = True
        mock_v.validate = AsyncMock(return_value=None)
        mw = AuthMiddleware(app, validators=[mock_v], require_auth=False, external_hostnames=set())

        async def send(msg):
            pass

        scope = _http_scope(
            "/x",
            extra_headers=[(b"cookie", b"idToken=cookieval; other=1")],
        )
        await mw(scope, AsyncMock(), send)

        assert state["user"]["type"] == "unverified_jwt"
        assert state["user"]["token"] == "cookieval"

    @pytest.mark.asyncio
    async def test_internal_invalid_bearer_token_falls_back_to_internal_when_auth_not_required(self, capture_asgi):
        app, state = capture_asgi
        mock_v = MagicMock()
        mock_v.can_handle.return_value = True
        mock_v.validate = AsyncMock(return_value=None)
        mw = AuthMiddleware(app, validators=[mock_v], require_auth=False, external_hostnames=set())

        async def send(msg):
            pass

        scope = _http_scope(
            "/any/path",
            extra_headers=[(b"authorization", b"Bearer badtoken")],
        )
        await mw(scope, AsyncMock(), send)

        assert state["user"]["type"] == "unverified_jwt"
        assert state["user"]["token"] == "badtoken"

    @pytest.mark.asyncio
    async def test_verified_user_tags_active_ddtrace_span(self, capture_asgi):
        app, _state = capture_asgi
        span_tags: dict[str, str] = {}

        class FakeSpan:
            def set_tag(self, key: str, value: str) -> None:
                span_tags[key] = value

        ddtrace_module = types.ModuleType("ddtrace")
        ddtrace_module.tracer = types.SimpleNamespace(current_span=lambda: FakeSpan())

        mock_v = MagicMock()
        mock_v.can_handle.return_value = True
        mock_v.validate = AsyncMock(
            return_value={
                "type": "oidc",
                "sub": "user-123",
                "email": "alice@example.com",
                "name": "Alice",
                "token": "good",
            }
        )

        async def send(msg):
            pass

        scope = _http_scope(
            "/chat",
            extra_headers=[(b"authorization", b"Bearer eyJhbGciOiJIUzI1NiJ9.x.y")],
        )
        with (
            patch.dict(
                os.environ,
                {
                    "DEEP_RESEARCHER_TRACE_USER_IDENTITY_MODE": "full",
                    "DEEP_RESEARCHER_TRACE_USER_IDENTITY_HMAC_SECRET": "test-secret",  # pragma: allowlist secret
                },
                clear=False,
            ),
            patch.dict(sys.modules, {"ddtrace": ddtrace_module}, clear=False),
        ):
            mw = AuthMiddleware(app, validators=[mock_v], require_auth=True, external_hostnames=set())
            await mw(scope, AsyncMock(), send)

        expected_id = middleware_module._build_pseudonymous_trace_user_id("oidc", "user-123", "test-secret")
        assert span_tags == {
            "enduser.id": expected_id,
            "deep-researcher.user.id": expected_id,
            "deep-researcher.auth.type": "oidc",
            "deep-researcher.user.email": "alice@example.com",
            "deep-researcher.user.name": "Alice",
            "deep-researcher.caller.type": "oidc",
            "deep-researcher.auth.transport": "bearer",
            "deep-researcher.auth.verified": "true",
            "deep-researcher.access.channel": "api",
        }

    @pytest.mark.asyncio
    async def test_verified_user_tags_active_otel_span(self, capture_asgi):
        app, _state = capture_asgi
        span_attributes: dict[str, str] = {}

        class FakeSpan:
            def set_attribute(self, key: str, value: str) -> None:
                span_attributes[key] = value

        opentelemetry_module = types.ModuleType("opentelemetry")
        trace_module = types.ModuleType("opentelemetry.trace")
        trace_module.get_current_span = lambda: FakeSpan()

        mock_v = MagicMock()
        mock_v.can_handle.return_value = True
        mock_v.validate = AsyncMock(return_value={"type": "service", "sub": "svc-user", "token": "good"})

        async def send(msg):
            pass

        scope = _http_scope(
            "/chat",
            extra_headers=[(b"authorization", b"Bearer eyJhbGciOiJIUzI1NiJ9.x.y")],
        )
        with (
            patch.dict(
                os.environ,
                {
                    "DEEP_RESEARCHER_TRACE_USER_IDENTITY_MODE": "id",
                    "DEEP_RESEARCHER_TRACE_USER_IDENTITY_HMAC_SECRET": "test-secret",  # pragma: allowlist secret
                },
                clear=False,
            ),
            patch.dict(
                sys.modules,
                {"opentelemetry": opentelemetry_module, "opentelemetry.trace": trace_module},
                clear=False,
            ),
        ):
            mw = AuthMiddleware(app, validators=[mock_v], require_auth=True, external_hostnames=set())
            await mw(scope, AsyncMock(), send)

        expected_id = middleware_module._build_pseudonymous_trace_user_id("service", "svc-user", "test-secret")
        assert span_attributes == {
            "enduser.id": expected_id,
            "deep-researcher.user.id": expected_id,
            "deep-researcher.auth.type": "service",
            "deep-researcher.caller.type": "service",
            "deep-researcher.auth.transport": "bearer",
            "deep-researcher.auth.verified": "true",
            "deep-researcher.access.channel": "api",
        }

    @pytest.mark.asyncio
    async def test_none_mode_does_not_tag_verified_user(self, capture_asgi):
        app, _state = capture_asgi
        span_tags: dict[str, str] = {}

        class FakeSpan:
            def set_tag(self, key: str, value: str) -> None:
                span_tags[key] = value

        ddtrace_module = types.ModuleType("ddtrace")
        ddtrace_module.tracer = types.SimpleNamespace(current_span=lambda: FakeSpan())

        mock_v = MagicMock()
        mock_v.can_handle.return_value = True
        mock_v.validate = AsyncMock(return_value={"type": "oidc", "sub": "user-123", "token": "good"})

        async def send(msg):
            pass

        scope = _http_scope(
            "/chat",
            extra_headers=[(b"authorization", b"Bearer eyJhbGciOiJIUzI1NiJ9.x.y")],
        )
        with (
            patch.dict(
                os.environ,
                {"DEEP_RESEARCHER_TRACE_USER_IDENTITY_MODE": "none"},
                clear=False,
            ),
            patch.dict(sys.modules, {"ddtrace": ddtrace_module}, clear=False),
        ):
            mw = AuthMiddleware(app, validators=[mock_v], require_auth=True, external_hostnames=set())
            await mw(scope, AsyncMock(), send)

        assert span_tags == {
            "deep-researcher.caller.type": "oidc",
            "deep-researcher.auth.transport": "bearer",
            "deep-researcher.auth.verified": "true",
            "deep-researcher.access.channel": "api",
        }

    @pytest.mark.asyncio
    async def test_always_on_common_tags_present_without_user_identity(self, capture_asgi):
        app, _state = capture_asgi
        span_tags: dict[str, str] = {}

        class FakeSpan:
            def set_tag(self, key: str, value: str) -> None:
                span_tags[key] = value

        ddtrace_module = types.ModuleType("ddtrace")
        ddtrace_module.tracer = types.SimpleNamespace(current_span=lambda: FakeSpan())

        mock_v = MagicMock()
        mock_v.can_handle.return_value = True
        mock_v.validate = AsyncMock(return_value={"type": "oidc", "sub": "user-123", "token": "good"})

        async def send(msg):
            pass

        scope = _http_scope(
            "/chat",
            extra_headers=[(b"authorization", b"Bearer eyJhbGciOiJIUzI1NiJ9.x.y")],
        )
        with (
            patch.dict(
                os.environ,
                {"DEEP_RESEARCHER_TRACE_USER_IDENTITY_MODE": "none", "DEEP_RESEARCHER_TRACE_CLIENT_ID_MODE": "none"},
                clear=False,
            ),
            patch.dict(sys.modules, {"ddtrace": ddtrace_module}, clear=False),
        ):
            mw = AuthMiddleware(app, validators=[mock_v], require_auth=True, external_hostnames=set())
            await mw(scope, AsyncMock(), send)

        assert span_tags == {
            "deep-researcher.caller.type": "oidc",
            "deep-researcher.auth.transport": "bearer",
            "deep-researcher.auth.verified": "true",
            "deep-researcher.access.channel": "api",
        }

    @pytest.mark.asyncio
    async def test_missing_secret_does_not_tag_verified_user(self, capture_asgi):
        app, _state = capture_asgi
        span_tags: dict[str, str] = {}

        class FakeSpan:
            def set_tag(self, key: str, value: str) -> None:
                span_tags[key] = value

        ddtrace_module = types.ModuleType("ddtrace")
        ddtrace_module.tracer = types.SimpleNamespace(current_span=lambda: FakeSpan())

        mock_v = MagicMock()
        mock_v.can_handle.return_value = True
        mock_v.validate = AsyncMock(return_value={"type": "oidc", "sub": "user-123", "token": "good"})

        async def send(msg):
            pass

        scope = _http_scope(
            "/chat",
            extra_headers=[(b"authorization", b"Bearer eyJhbGciOiJIUzI1NiJ9.x.y")],
        )
        with (
            patch.dict(
                os.environ,
                {
                    "DEEP_RESEARCHER_TRACE_USER_IDENTITY_MODE": "id",
                    "DEEP_RESEARCHER_TRACE_USER_IDENTITY_HMAC_SECRET": "",
                },
                clear=False,
            ),
            patch.dict(sys.modules, {"ddtrace": ddtrace_module}, clear=False),
        ):
            mw = AuthMiddleware(app, validators=[mock_v], require_auth=True, external_hostnames=set())
            await mw(scope, AsyncMock(), send)

        assert span_tags == {
            "deep-researcher.caller.type": "oidc",
            "deep-researcher.auth.transport": "bearer",
            "deep-researcher.auth.verified": "true",
            "deep-researcher.access.channel": "api",
        }

    @pytest.mark.asyncio
    async def test_explicit_access_channel_override_wins(self, capture_asgi):
        app, _state = capture_asgi
        span_tags: dict[str, str] = {}

        class FakeSpan:
            def set_tag(self, key: str, value: str) -> None:
                span_tags[key] = value

        ddtrace_module = types.ModuleType("ddtrace")
        ddtrace_module.tracer = types.SimpleNamespace(current_span=lambda: FakeSpan())

        mock_v = MagicMock()
        mock_v.can_handle.return_value = True
        mock_v.validate = AsyncMock(return_value={"type": "oidc", "sub": "user-123", "token": "good"})

        async def send(msg):
            pass

        scope = _http_scope(
            "/chat",
            extra_headers=[
                (b"authorization", b"Bearer eyJhbGciOiJIUzI1NiJ9.x.y"),
                (b"x-deep-researcher-access-channel", b"headless"),
                (b"x-deep-researcher-mode", b"headless"),
            ],
        )
        with (
            patch.dict(
                os.environ,
                {"DEEP_RESEARCHER_TRACE_USER_IDENTITY_MODE": "none", "DEEP_RESEARCHER_TRACE_CLIENT_ID_MODE": "none"},
                clear=False,
            ),
            patch.dict(sys.modules, {"ddtrace": ddtrace_module}, clear=False),
        ):
            mw = AuthMiddleware(app, validators=[mock_v], require_auth=True, external_hostnames=set())
            await mw(scope, AsyncMock(), send)

        assert span_tags["deep-researcher.access.channel"] == "headless"

    @pytest.mark.asyncio
    async def test_explicit_access_channel_ignored_for_external_anonymous_request(self, capture_asgi, external_host):
        app, _state = capture_asgi
        span_tags: dict[str, str] = {}

        class FakeSpan:
            def set_tag(self, key: str, value: str) -> None:
                span_tags[key] = value

        ddtrace_module = types.ModuleType("ddtrace")
        ddtrace_module.tracer = types.SimpleNamespace(current_span=lambda: FakeSpan())

        async def send(msg):
            pass

        scope = _http_scope(
            "/health",
            host=external_host,
            extra_headers=[(b"x-deep-researcher-access-channel", b"internal")],
        )
        with (
            patch.dict(
                os.environ,
                {"DEEP_RESEARCHER_TRACE_USER_IDENTITY_MODE": "none", "DEEP_RESEARCHER_TRACE_CLIENT_ID_MODE": "none"},
                clear=False,
            ),
            patch.dict(sys.modules, {"ddtrace": ddtrace_module}, clear=False),
        ):
            mw = AuthMiddleware(app, validators=[], require_auth=False, external_hostnames={external_host.decode()})
            await mw(scope, AsyncMock(), send)

        assert span_tags["deep-researcher.access.channel"] == "anonymous"

    @pytest.mark.asyncio
    async def test_client_ip_mode_adds_pseudonymous_client_id(self, capture_asgi):
        app, _state = capture_asgi
        span_tags: dict[str, str] = {}

        class FakeSpan:
            def set_tag(self, key: str, value: str) -> None:
                span_tags[key] = value

        ddtrace_module = types.ModuleType("ddtrace")
        ddtrace_module.tracer = types.SimpleNamespace(current_span=lambda: FakeSpan())

        mock_v = MagicMock()
        mock_v.can_handle.return_value = True
        mock_v.validate = AsyncMock(return_value={"type": "anonymous", "skip_clarifier": True})

        async def send(msg):
            pass

        scope = _http_scope(
            "/health",
            host=b"api.public.example",
            extra_headers=[(b"x-forwarded-for", b"203.0.113.10, 10.0.0.1")],
        )
        with (
            patch.dict(
                os.environ,
                {
                    "DEEP_RESEARCHER_TRACE_USER_IDENTITY_MODE": "none",
                    "DEEP_RESEARCHER_TRACE_CLIENT_ID_MODE": "ip",
                    "DEEP_RESEARCHER_TRACE_CLIENT_ID_HMAC_SECRET": "client-secret",  # pragma: allowlist secret
                },
                clear=False,
            ),
            patch.dict(sys.modules, {"ddtrace": ddtrace_module}, clear=False),
        ):
            mw = AuthMiddleware(app, validators=[mock_v], require_auth=False, external_hostnames={"api.public.example"})
            await mw(scope, AsyncMock(), send)

        expected_client_id = middleware_module._build_pseudonymous_trace_client_id("203.0.113.10", "client-secret")
        assert span_tags == {
            "deep-researcher.caller.type": "anonymous",
            "deep-researcher.auth.transport": "none",
            "deep-researcher.auth.verified": "false",
            "deep-researcher.access.channel": "anonymous",
            "deep-researcher.client.id": expected_client_id,
        }

    @pytest.mark.asyncio
    async def test_unverified_internal_token_does_not_tag_active_span(self, capture_asgi):
        app, _state = capture_asgi
        span_tags: dict[str, str] = {}

        class FakeSpan:
            def set_tag(self, key: str, value: str) -> None:
                span_tags[key] = value

        ddtrace_module = types.ModuleType("ddtrace")
        ddtrace_module.tracer = types.SimpleNamespace(current_span=lambda: FakeSpan())

        mock_v = MagicMock()
        mock_v.can_handle.return_value = True
        mock_v.validate = AsyncMock(return_value=None)

        async def send(msg):
            pass

        scope = _http_scope(
            "/chat",
            extra_headers=[(b"authorization", b"Bearer badtoken")],
        )
        with (
            patch.dict(
                os.environ,
                {
                    "DEEP_RESEARCHER_TRACE_USER_IDENTITY_MODE": "full",
                    "DEEP_RESEARCHER_TRACE_USER_IDENTITY_HMAC_SECRET": "test-secret",  # pragma: allowlist secret
                },
                clear=False,
            ),
            patch.dict(sys.modules, {"ddtrace": ddtrace_module}, clear=False),
        ):
            mw = AuthMiddleware(app, validators=[mock_v], require_auth=False, external_hostnames=set())
            await mw(scope, AsyncMock(), send)

        assert span_tags == {
            "deep-researcher.caller.type": "unverified_jwt",
            "deep-researcher.auth.transport": "bearer",
            "deep-researcher.auth.verified": "false",
            "deep-researcher.access.channel": "api",
        }


class TestAuthMiddlewareHelpers:
    def test_extract_token_bearer(self) -> None:
        mw = AuthMiddleware(MagicMock(), external_hostnames=set())
        h = {b"authorization": b"Bearer abc.def.ghi"}
        assert mw._extract_token(h) == "abc.def.ghi"

    def test_extract_token_from_cookie(self) -> None:
        mw = AuthMiddleware(MagicMock(), external_hostnames=set())
        h = {b"cookie": b"foo=1; idToken=tok123"}
        assert mw._extract_token(h) == "tok123"

    def test_path_allowed_exact_and_prefix(self) -> None:
        mw = AuthMiddleware(MagicMock(), external_hostnames=set())
        assert mw._path_allowed("/live") is True
        assert mw._path_allowed("/health") is True
        assert mw._path_allowed("/v1/jobs/async/job/abc/result") is True
        assert mw._path_allowed("/v1/jobs/async/job") is True
        assert mw._path_allowed("/nope") is False
        # Per-user MCP auth routes must be reachable externally.
        assert mw._path_allowed("/v1/auth/mcp/gdrive/status") is True
        assert mw._path_allowed("/v1/auth/mcp/gdrive/connect") is True
        assert mw._path_allowed("/v1/auth/mcp/gdrive/callback") is True

    def test_mcp_oauth_callback_is_auth_exempt(self) -> None:
        from deep_researcher_api.auth.middleware import _is_oauth_callback_path

        # Only the OAuth redirect callback is exempt (no Deep Researcher Agent token; secured by state).
        assert _is_oauth_callback_path("/v1/auth/mcp/gdrive/callback") is True
        # status/connect must still require auth (they need the principal).
        assert _is_oauth_callback_path("/v1/auth/mcp/gdrive/status") is False
        assert _is_oauth_callback_path("/v1/auth/mcp/gdrive/connect") is False
        assert _is_oauth_callback_path("/v1/data_sources") is False

    @pytest.mark.asyncio
    async def test_non_http_passthrough(self) -> None:
        inner_calls: list[str] = []

        async def app(scope, receive, send):
            inner_calls.append(scope["type"])

        mw = AuthMiddleware(app, external_hostnames=set())
        scope = {"type": "websocket", "headers": []}

        await mw(scope, AsyncMock(), AsyncMock())

        assert inner_calls == ["websocket"]

    @pytest.mark.asyncio
    async def test_contextvar_reset_after_request(self, capture_asgi, external_host):
        app, _ = capture_asgi
        mw = AuthMiddleware(app, require_auth=False, external_hostnames={external_host.decode()})

        async def send(msg):
            pass

        scope = _http_scope("/health", host=external_host)
        await mw(scope, AsyncMock(), send)

        assert get_current_user()["type"] == "internal"


class TestAuthMiddlewareExempt:
    @pytest.mark.asyncio
    async def test_liveness_exempt_without_auth(self, capture_asgi, external_host):
        app, state = capture_asgi
        mw = AuthMiddleware(app, require_auth=True, external_hostnames={external_host.decode()})

        async def send(msg):
            pass

        scope = _http_scope("/live", host=external_host)
        await mw(scope, AsyncMock(), send)

        assert state["user"]["type"] == "anonymous"

    @pytest.mark.asyncio
    async def test_health_exempt_without_auth(self, capture_asgi, external_host):
        app, state = capture_asgi
        mw = AuthMiddleware(app, require_auth=True, external_hostnames={external_host.decode()})

        async def send(msg):
            pass

        scope = _http_scope("/health", host=external_host)
        await mw(scope, AsyncMock(), send)

        assert state["user"]["type"] == "anonymous"


class TestLoadExternalHostnames:
    def test_reads_deep_researcher_external_hostnames_env(self) -> None:
        with patch.dict(os.environ, {"DEEP_RESEARCHER_EXTERNAL_HOSTNAMES": " a.com , b.com "}):
            from deep_researcher_api.auth.middleware import _load_external_hostnames

            assert _load_external_hostnames() == {"a.com", "b.com"}


class TestAuthPackageExports:
    def test_all_exports_importable(self) -> None:
        from deep_researcher_api import auth as auth_pkg

        for name in auth_pkg.__all__:
            assert getattr(auth_pkg, name) is not None


# ---------------------------------------------------------------------------
# Error subclasses
# ---------------------------------------------------------------------------


class TestAuthErrorSubclasses:
    def test_token_expired_is_auth_error(self) -> None:
        assert issubclass(TokenExpiredError, AuthError)
        assert TokenExpiredError.error_code == "token_expired"

    def test_token_invalid_is_auth_error(self) -> None:
        assert issubclass(TokenInvalidError, AuthError)
        assert TokenInvalidError.error_code == "token_invalid"

    def test_base_auth_error_code(self) -> None:
        assert AuthError.error_code == "auth_error"


# ---------------------------------------------------------------------------
# validate() error codes (JWTValidator)
# ---------------------------------------------------------------------------


class TestValidateErrorCodes:
    @pytest.mark.asyncio
    async def test_jwt_validator_expired_returns_token_expired(self) -> None:
        priv = _rsa_private_key()
        jwk = _signing_jwk_from_private(priv)
        issuer = "https://issuer.example"
        now = datetime.now(UTC)
        claims = {
            "sub": "user-1",
            "iss": issuer,
            "exp": now - timedelta(hours=1),
            "iat": now - timedelta(hours=2),
        }
        token = jwt.encode(claims, priv, algorithm="RS256")

        validator = JWTValidator(issuer, jwks_uri="https://unused/jwks")
        with patch.object(validator, "_get_signing_key", return_value=jwk):
            user, error = await validator.validate(token)

        assert user is None
        assert error == "token_expired"

    @pytest.mark.asyncio
    async def test_jwt_validator_invalid_sig_returns_token_invalid(self) -> None:
        priv = _rsa_private_key()
        other_priv = _rsa_private_key()
        jwk = _signing_jwk_from_private(other_priv)  # wrong key
        issuer = "https://issuer.example"
        now = datetime.now(UTC)
        claims = {
            "sub": "user-1",
            "iss": issuer,
            "exp": now + timedelta(hours=1),
            "iat": now,
        }
        token = jwt.encode(claims, priv, algorithm="RS256")

        validator = JWTValidator(issuer, jwks_uri="https://unused/jwks")
        with patch.object(validator, "_get_signing_key", return_value=jwk):
            user, error = await validator.validate(token)

        assert user is None
        assert error == "token_invalid"

    @pytest.mark.asyncio
    async def test_jwt_validator_success_returns_user_and_no_error(self) -> None:
        priv = _rsa_private_key()
        jwk = _signing_jwk_from_private(priv)
        issuer = "https://issuer.example"
        now = datetime.now(UTC)
        claims = {
            "sub": "user-1",
            "iss": issuer,
            "aud": "my-api",
            "exp": now + timedelta(hours=1),
            "iat": now,
        }
        token = jwt.encode(claims, priv, algorithm="RS256", headers={"kid": "k1"})

        validator = JWTValidator(issuer, audience="my-api", jwks_uri="https://unused/jwks")
        with patch.object(validator, "_get_signing_key", return_value=jwk):
            user, error = await validator.validate(token)

        assert user is not None
        assert user["sub"] == "user-1"
        assert user["type"] == "jwt"
        assert error is None


# ---------------------------------------------------------------------------
# Middleware error code propagation
# ---------------------------------------------------------------------------


class TestMiddlewareErrorCodes:
    @pytest.mark.asyncio
    async def test_missing_token_returns_token_missing(self, capture_asgi, external_host):
        app, _ = capture_asgi
        mw = AuthMiddleware(app, validators=[], require_auth=True, external_hostnames={external_host.decode()})
        messages: list[dict] = []

        async def send(msg):
            messages.append(msg)

        scope = _http_scope("/chat", host=external_host)
        await mw(scope, AsyncMock(), send)

        body = json.loads(messages[1]["body"].decode())
        assert body["error"] == "token_missing"
        assert messages[0]["status"] == 401

    @pytest.mark.asyncio
    async def test_expired_token_returns_token_expired(self, capture_asgi, external_host):
        app, _ = capture_asgi
        mock_v = MagicMock()
        mock_v.can_handle.return_value = True
        mock_v.validate = AsyncMock(return_value=(None, "token_expired"))
        mw = AuthMiddleware(
            app,
            validators=[mock_v],
            require_auth=True,
            external_hostnames={external_host.decode()},
        )
        messages: list[dict] = []

        async def send(msg):
            messages.append(msg)

        scope = _http_scope(
            "/chat",
            host=external_host,
            extra_headers=[(b"authorization", b"Bearer expired.token.here")],
        )
        await mw(scope, AsyncMock(), send)

        body = json.loads(messages[1]["body"].decode())
        assert body["error"] == "token_expired"
        assert body["detail"] == "Token has expired"
        assert messages[0]["status"] == 401
