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

"""JWT validation via OIDC discovery and JWKS signature verification.

Uses PyJWT[cryptography] to verify RS256-signed JWTs against the issuer's
published public keys. Keys are cached for 6 hours (default); a kid miss
forces an immediate refresh.

Install: pip install 'PyJWT[cryptography]'
"""

import asyncio
import logging
import threading
import time
from typing import Any

from .base import TokenValidator

logger = logging.getLogger(__name__)

_MISSING_PYJWT = "PyJWT[cryptography] is required for JWT validation. Install with: pip install 'PyJWT[cryptography]'"
_MAX_FETCH_BYTES = 64 << 10  # 64 KB cap on OIDC/JWKS responses


class JWTValidator(TokenValidator):
    """
    Validates OIDC JWTs (any standards-compliant provider).

    Fetches the JWKS URI from the issuer's OIDC discovery document, then
    verifies token signatures using the matching public key.  The JWKS
    client is recreated after `jwks_cache_ttl` seconds.

    Args:
        issuer_url: OIDC issuer base URL (e.g. ``https://accounts.google.com``).
            Discovery document is fetched from ``{issuer_url}/.well-known/openid-configuration``.
        audience: Optional ``aud`` claim to verify.  Pass ``None`` to skip
            audience verification.
        jwks_cache_ttl: Seconds before the JWKS client is recreated.  Defaults
            to 6 hours.
    """

    def __init__(
        self,
        issuer_url: str,
        audience: str | None = None,
        jwks_cache_ttl: int = 6 * 3600,
        jwks_uri: str | None = None,
        verify_iss: bool = True,
        algorithms: list[str] | None = None,
    ) -> None:
        self.issuer_url = issuer_url.rstrip("/")
        self.audience = audience
        self._jwks_cache_ttl = jwks_cache_ttl
        self.verify_iss = verify_iss
        self.algorithms = algorithms or ["RS256", "ES256", "ES512"]

        # Lazily initialised; if jwks_uri is provided, skip OIDC discovery
        self._jwks_uri: str | None = jwks_uri
        # List of (kid_or_None, PyJWK) pairs; refreshed after TTL
        self._cached_keys: list[tuple[str | None, Any]] | None = None
        self._jwks_keys_fetched_at: float = 0.0
        self._jwks_lock = threading.Lock()

    def can_handle(self, token: str) -> bool:
        """Accept any token that looks like a compact JWT (three dot-separated segments)."""
        parts = token.split(".")
        return len(parts) == 3

    # ------------------------------------------------------------------
    # Internal helpers (sync — run via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _fetch_oidc_config(self) -> dict:
        """Fetch the OIDC discovery document and return it as a dict."""
        import json
        import urllib.request

        url = f"{self.issuer_url}/.well-known/openid-configuration"
        logger.debug("Fetching OIDC discovery from %s", url)
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            raw = resp.read(_MAX_FETCH_BYTES + 1)
            if len(raw) > _MAX_FETCH_BYTES:
                logger.warning(
                    "OIDC discovery response from %s exceeded %d-byte cap; truncating",
                    url,
                    _MAX_FETCH_BYTES,
                )
                raw = raw[:_MAX_FETCH_BYTES]
            return json.loads(raw)

    def _fetch_jwks_keys(self) -> list[tuple[str | None, Any]]:
        """Fetch JWKS and return (kid, PyJWK) pairs.

        Adds ``use="sig"`` for keys that omit the field so that
        PyJWT ≥ 2.9 — which silently drops keys without ``use="sig"`` — still
        accepts them.
        """
        import json
        import urllib.request

        import jwt as _jwt

        assert self._jwks_uri is not None  # caller ensures this
        with urllib.request.urlopen(self._jwks_uri, timeout=10) as resp:  # noqa: S310
            raw = resp.read(_MAX_FETCH_BYTES + 1)
            if len(raw) > _MAX_FETCH_BYTES:
                logger.warning(
                    "JWKS response from %s exceeded %d-byte cap; truncating",
                    self._jwks_uri,
                    _MAX_FETCH_BYTES,
                )
                raw = raw[:_MAX_FETCH_BYTES]
            data = json.loads(raw)

        keys: list[tuple[str | None, Any]] = []
        for key_data in data.get("keys", []):
            if "use" not in key_data:
                key_data = {**key_data, "use": "sig"}
            try:
                jwk = _jwt.PyJWK.from_dict(key_data)
                keys.append((key_data.get("kid"), jwk))
            except Exception as e:
                logger.debug("Skipping unparseable JWK: %s", e)

        logger.debug("Loaded %d JWK(s) from %s", len(keys), self._jwks_uri)
        return keys

    def _get_signing_key(self, token: str):
        """Return the signing key for *token*, or ``None`` on failure.

        Cache reads/writes run under ``_jwks_lock`` because this method is
        invoked from thread-pool workers via ``asyncio.to_thread``.
        """
        import jwt as _jwt

        with self._jwks_lock:
            now = time.monotonic()
            if self._cached_keys is None or (now - self._jwks_keys_fetched_at) > self._jwks_cache_ttl:
                if self._jwks_uri is None:
                    config = self._fetch_oidc_config()
                    self._jwks_uri = config["jwks_uri"]
                    logger.debug("JWKS URI: %s", self._jwks_uri)
                try:
                    self._cached_keys = self._fetch_jwks_keys()
                    self._jwks_keys_fetched_at = time.monotonic()
                except Exception as e:
                    logger.debug("JWKS fetch failed: %s", e)
                    self._cached_keys = None
                    return None

            if not self._cached_keys:
                logger.debug("No usable keys in JWKS")
                return None

            # Match by kid if the token carries one
            try:
                header = _jwt.get_unverified_header(token)
                kid = header.get("kid")
            except Exception:
                kid = None

            if kid:
                for key_kid, jwk in self._cached_keys:
                    if key_kid == kid:
                        return jwk
                # kid not in cache — force a refresh once and retry
                logger.debug("kid %s not in cached keys, refreshing JWKS", kid)
                self._cached_keys = None
                try:
                    self._cached_keys = self._fetch_jwks_keys()
                    self._jwks_keys_fetched_at = time.monotonic()
                    for key_kid, jwk in self._cached_keys:
                        if key_kid == kid:
                            return jwk
                except Exception as e:
                    logger.debug("JWKS refresh failed: %s", e)
                return None

            # No kid — return the only key, or the first one if multiple
            return self._cached_keys[0][1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def validate(self, token: str) -> tuple[dict[str, Any] | None, str | None]:
        """Validate *token* and return ``(user_dict, error_code)``.

        Returns ``(user_dict, None)`` on success.  On failure returns
        ``(None, error_code)`` where *error_code* is one of:

        - ``"token_expired"`` — signature valid but ``exp`` claim passed
        - ``"token_invalid"`` — bad signature, malformed, or claim check failure

        Merges verified JWT claims with the required ``type`` / ``token`` /
        ``skip_clarifier`` fields. Contract keys are applied after ``**claims``
        so the bearer token and caller metadata cannot be spoofed by payload
        claims.
        """
        try:
            import jwt as _jwt
        except ImportError:
            logger.error(_MISSING_PYJWT)
            return (None, "token_invalid")

        try:
            signing_key = await asyncio.to_thread(self._get_signing_key, token)
            if signing_key is None:
                return (None, "token_invalid")

            options: dict[str, Any] = {"verify_exp": True}
            if not self.verify_iss:
                options["verify_iss"] = False
            decode_kwargs: dict[str, Any] = {
                "algorithms": self.algorithms,
                "issuer": self.issuer_url,
                "options": options,
            }
            if self.audience:
                decode_kwargs["audience"] = self.audience
            else:
                options["verify_aud"] = False

            claims = _jwt.decode(token, signing_key.key, **decode_kwargs)
            user = {
                **claims,
                "type": "jwt",
                "sub": claims.get("sub"),
                "email": claims.get("email"),
                "name": claims.get("name"),
                "token": token,
                "skip_clarifier": False,
            }
            return (user, None)

        except _jwt.ExpiredSignatureError:
            logger.debug("JWT expired")
            return (None, "token_expired")
        except _jwt.InvalidTokenError as exc:
            logger.debug("JWT invalid: %s", exc)
            return (None, "token_invalid")
        except Exception as exc:
            logger.warning("JWT validation error: %s", exc)
            return (None, "token_invalid")
