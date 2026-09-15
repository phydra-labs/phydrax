#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ._contracts import AuthenticationError, AuthorizationError, ValidatedPrincipal


_MAX_COMPACT_TOKEN_BYTES = 16_384
_MAX_HEADER_SEGMENT_BYTES = 2_048
_MAX_PAYLOAD_SEGMENT_BYTES = 12_288
_MAX_SIGNATURE_SEGMENT_BYTES = 2_048
_MAX_HEADER_DECODED_BYTES = 1_024
_MAX_PAYLOAD_DECODED_BYTES = 8_192
_MAX_SIGNATURE_DECODED_BYTES = 1_024
_MAX_JSON_NESTING = 8
_MAX_JSON_KEYS = 64
_MAX_JSON_KEY_BYTES = 128
_MAX_JSON_ARRAY_ITEMS = 64
_MAX_JSON_STRING_BYTES = 2_048


class Clock(Protocol):
    def now(self) -> int:
        """Return whole Unix seconds."""


class SystemClock:
    def now(self) -> int:
        return int(time.time())


class AccessTokenValidator(Protocol):
    def validate(self, token: str, /) -> ValidatedPrincipal:
        """Authenticate an OAuth2 bearer access token."""


class ResourceAuthorizer(Protocol):
    def authorize(
        self,
        principal: ValidatedPrincipal,
        required_scope: str,
        resource_tenant_id: str,
        /,
    ) -> None:
        """Authorize a tenant-scoped resource action or fail closed."""


@dataclass(frozen=True, slots=True)
class OIDCConfiguration:
    issuer: str
    audience: str
    clock_skew_seconds: int = 30
    maximum_token_lifetime_seconds: int = 3_600

    def __post_init__(self) -> None:
        if not self.issuer or not self.audience:
            raise ValueError("OIDC issuer and audience must be nonempty.")
        if self.clock_skew_seconds < 0:
            raise ValueError("OIDC clock skew must be nonnegative.")
        if self.maximum_token_lifetime_seconds <= 0:
            raise ValueError("OIDC maximum token lifetime must be positive.")


@dataclass(frozen=True, slots=True)
class HMACSigningKey:
    key_id: str
    secret: bytes

    def __post_init__(self) -> None:
        if not self.key_id:
            raise ValueError("OIDC signing key_id must be nonempty.")
        if len(self.secret) < 32:
            raise ValueError("OIDC HS256 signing keys must contain at least 256 bits.")


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _bounded_access_token(token: str, /) -> str:
    if not isinstance(token, str) or not token:
        raise AuthenticationError("Bearer token must be a nonempty string.")
    if len(token) > _MAX_COMPACT_TOKEN_BYTES:
        raise AuthenticationError("Bearer token exceeds the compact-token byte limit.")
    if not token.isascii():
        raise AuthenticationError("Bearer token must contain only ASCII data.")
    return token


def _preflight_segment(
    value: str,
    label: str,
    /,
    *,
    maximum_encoded_bytes: int,
    maximum_decoded_bytes: int,
) -> None:
    if not value:
        raise AuthenticationError(f"Bearer token {label} segment is empty.")
    if len(value) > maximum_encoded_bytes:
        raise AuthenticationError(
            f"Bearer token {label} segment exceeds its encoded-byte limit."
        )
    if len(value) * 3 // 4 > maximum_decoded_bytes:
        raise AuthenticationError(
            f"Bearer token {label} segment exceeds its decoded-byte limit."
        )


def _compact_token_segments(token: str, /) -> tuple[str, str, str]:
    bounded = _bounded_access_token(token)
    parts = bounded.split(".", 3)
    if len(parts) != 3 or any(not part for part in parts):
        raise AuthenticationError("Bearer token must be a compact signed JWT.")
    for value, label, encoded_limit, decoded_limit in (
        (
            parts[0],
            "header",
            _MAX_HEADER_SEGMENT_BYTES,
            _MAX_HEADER_DECODED_BYTES,
        ),
        (
            parts[1],
            "payload",
            _MAX_PAYLOAD_SEGMENT_BYTES,
            _MAX_PAYLOAD_DECODED_BYTES,
        ),
        (
            parts[2],
            "signature",
            _MAX_SIGNATURE_SEGMENT_BYTES,
            _MAX_SIGNATURE_DECODED_BYTES,
        ),
    ):
        _preflight_segment(
            value,
            label,
            maximum_encoded_bytes=encoded_limit,
            maximum_decoded_bytes=decoded_limit,
        )
    return parts[0], parts[1], parts[2]


def _decode_segment(
    value: str,
    label: str,
    /,
    *,
    maximum_encoded_bytes: int,
    maximum_decoded_bytes: int,
) -> bytes:
    _preflight_segment(
        value,
        label,
        maximum_encoded_bytes=maximum_encoded_bytes,
        maximum_decoded_bytes=maximum_decoded_bytes,
    )
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (ValueError, UnicodeEncodeError) as error:
        raise AuthenticationError(
            "Bearer token contains invalid base64url data."
        ) from error
    if len(decoded) > maximum_decoded_bytes:
        raise AuthenticationError(
            f"Bearer token {label} segment exceeds its decoded-byte limit."
        )
    return decoded


def _preflight_json(value: bytes, label: str, /) -> None:
    stack: list[list[int]] = []
    key_count = 0
    index = 0
    last_string_bytes: int | None = None

    def note_array_value() -> None:
        if stack and stack[-1][0] == ord("[") and stack[-1][2]:
            stack[-1][1] += 1
            stack[-1][2] = 0
            if stack[-1][1] > _MAX_JSON_ARRAY_ITEMS:
                raise AuthenticationError(
                    f"Bearer token {label} exceeds the JSON array-item limit."
                )

    while index < len(value):
        byte = value[index]
        if byte in b" \t\r\n":
            index += 1
            continue
        if byte == ord('"'):
            note_array_value()
            start = index + 1
            index = start
            escaped = False
            while index < len(value):
                current = value[index]
                if escaped:
                    escaped = False
                elif current == ord("\\"):
                    escaped = True
                elif current == ord('"'):
                    break
                index += 1
            if index >= len(value):
                raise AuthenticationError(f"Bearer token {label} is not valid JSON.")
            string_bytes = index - start
            if string_bytes > _MAX_JSON_STRING_BYTES:
                raise AuthenticationError(
                    f"Bearer token {label} exceeds the JSON string-byte limit."
                )
            last_string_bytes = string_bytes
            index += 1
            continue
        if byte in (ord("{"), ord("[")):
            note_array_value()
            if len(stack) >= _MAX_JSON_NESTING:
                raise AuthenticationError(
                    f"Bearer token {label} exceeds the JSON nesting limit."
                )
            stack.append([byte, 0, 1])
            last_string_bytes = None
        elif byte in (ord("}"), ord("]")):
            expected = ord("{") if byte == ord("}") else ord("[")
            if not stack or stack[-1][0] != expected:
                raise AuthenticationError(f"Bearer token {label} is not valid JSON.")
            stack.pop()
            last_string_bytes = None
        elif byte == ord(":"):
            key_count += 1
            if key_count > _MAX_JSON_KEYS:
                raise AuthenticationError(
                    f"Bearer token {label} exceeds the JSON key-count limit."
                )
            if last_string_bytes is not None and last_string_bytes > _MAX_JSON_KEY_BYTES:
                raise AuthenticationError(
                    f"Bearer token {label} exceeds the JSON key-byte limit."
                )
            last_string_bytes = None
        elif byte == ord(","):
            if stack and stack[-1][0] == ord("["):
                stack[-1][2] = 1
            last_string_bytes = None
        else:
            note_array_value()
            last_string_bytes = None
        index += 1
    if stack:
        raise AuthenticationError(f"Bearer token {label} is not valid JSON.")


def _json_segment(value: str, kind: str) -> dict[str, Any]:
    if kind == "header":
        encoded_limit = _MAX_HEADER_SEGMENT_BYTES
        decoded_limit = _MAX_HEADER_DECODED_BYTES
    else:
        encoded_limit = _MAX_PAYLOAD_SEGMENT_BYTES
        decoded_limit = _MAX_PAYLOAD_DECODED_BYTES
    raw = _decode_segment(
        value,
        kind,
        maximum_encoded_bytes=encoded_limit,
        maximum_decoded_bytes=decoded_limit,
    )
    _preflight_json(raw, kind)
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthenticationError(f"Bearer token {kind} is not valid JSON.") from error
    if not isinstance(decoded, dict):
        raise AuthenticationError(f"Bearer token {kind} must be a JSON object.")
    return decoded


def _signature_segment(value: str, /) -> bytes:
    return _decode_segment(
        value,
        "signature",
        maximum_encoded_bytes=_MAX_SIGNATURE_SEGMENT_BYTES,
        maximum_decoded_bytes=_MAX_SIGNATURE_DECODED_BYTES,
    )


def _required_string(claims: Mapping[str, Any], name: str) -> str:
    value = claims.get(name)
    if not isinstance(value, str) or not value.strip():
        raise AuthenticationError(f"Bearer token requires a nonempty {name!r} claim.")
    return value


def _required_timestamp(claims: Mapping[str, Any], name: str) -> int:
    value = claims.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AuthenticationError(f"Bearer token requires an integer {name!r} claim.")
    return value


class HMACOIDCTokenValidator:
    """Strict offline validator for RFC 9068-style HS256 access JWTs.

    Production network adapters can supply any ``AccessTokenValidator`` backed by an
    OIDC discovery/JWKS implementation. This reference validator performs signature and
    claim validation itself and never trusts unverified tenant or scope claims.
    """

    def __init__(
        self,
        configuration: OIDCConfiguration,
        keys: tuple[HMACSigningKey, ...],
        /,
        *,
        clock: Clock | None = None,
    ):
        if not keys:
            raise ValueError("At least one OIDC signing key is required.")
        by_id = {key.key_id: key.secret for key in keys}
        if len(by_id) != len(keys):
            raise ValueError("OIDC signing key identifiers must be unique.")
        self._configuration = configuration
        self._keys = by_id
        self._clock = SystemClock() if clock is None else clock

    def validate(self, token: str, /) -> ValidatedPrincipal:
        parts = _compact_token_segments(token)
        header = _json_segment(parts[0], "header")
        claims = _json_segment(parts[1], "payload")
        if header.get("alg") != "HS256" or header.get("typ") != "at+jwt":
            raise AuthenticationError("Bearer token algorithm or type is not accepted.")
        key_id = _required_string(header, "kid")
        key = self._keys.get(key_id)
        if key is None:
            raise AuthenticationError("Bearer token signing key is not trusted.")
        signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
        signature = _signature_segment(parts[2])
        expected = hmac.new(key, signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, signature):
            raise AuthenticationError("Bearer token signature is invalid.")

        issuer = _required_string(claims, "iss")
        if issuer != self._configuration.issuer:
            raise AuthenticationError("Bearer token issuer is not accepted.")
        audiences = claims.get("aud")
        if isinstance(audiences, str):
            audience_values = (audiences,)
        elif isinstance(audiences, list) and all(
            isinstance(value, str) and value for value in audiences
        ):
            audience_values = tuple(audiences)
        else:
            raise AuthenticationError("Bearer token audience claim is invalid.")
        if self._configuration.audience not in audience_values:
            raise AuthenticationError("Bearer token audience is not accepted.")

        issued_at = _required_timestamp(claims, "iat")
        not_before = _required_timestamp(claims, "nbf")
        expires_at = _required_timestamp(claims, "exp")
        now = self._clock.now()
        skew = self._configuration.clock_skew_seconds
        if issued_at > now + skew or not_before > now + skew:
            raise AuthenticationError("Bearer token is not active yet.")
        if expires_at <= now - skew:
            raise AuthenticationError("Bearer token has expired.")
        if expires_at <= issued_at or (
            expires_at - issued_at > self._configuration.maximum_token_lifetime_seconds
        ):
            raise AuthenticationError("Bearer token lifetime is not accepted.")

        scope_claim = claims.get("scope")
        if isinstance(scope_claim, str):
            scopes = frozenset(scope_claim.split())
        elif isinstance(scope_claim, list) and all(
            isinstance(value, str) and value for value in scope_claim
        ):
            scopes = frozenset(scope_claim)
        else:
            raise AuthenticationError("Bearer token scope claim is invalid.")
        if not scopes:
            raise AuthenticationError("Bearer token must grant at least one scope.")
        return ValidatedPrincipal(
            subject=_required_string(claims, "sub"),
            tenant_id=_required_string(claims, "tenant_id"),
            issuer=issuer,
            audience=self._configuration.audience,
            client_id=_required_string(claims, "client_id"),
            token_id=_required_string(claims, "jti"),
            scopes=scopes,
            issued_at=issued_at,
            expires_at=expires_at,
        )


class HMACOIDCTokenIssuer:
    """Reference OIDC access-token issuer paired with the offline validator."""

    def __init__(
        self,
        configuration: OIDCConfiguration,
        signing_key: HMACSigningKey,
        /,
        *,
        clock: Clock | None = None,
    ):
        self._configuration = configuration
        self._signing_key = signing_key
        self._clock = SystemClock() if clock is None else clock

    def issue(
        self,
        *,
        subject: str,
        tenant_id: str,
        client_id: str,
        token_id: str,
        scopes: frozenset[str],
        lifetime_seconds: int = 600,
    ) -> str:
        values = (subject, tenant_id, client_id, token_id)
        if any(not value.strip() for value in values) or not scopes:
            raise ValueError("OIDC token identity and scopes must be nonempty.")
        if any(not value.strip() for value in scopes):
            raise ValueError("OIDC scopes must be nonempty.")
        lifetime = int(lifetime_seconds)
        if not 0 < lifetime <= self._configuration.maximum_token_lifetime_seconds:
            raise ValueError("OIDC token lifetime exceeds configured bounds.")
        issued_at = self._clock.now()
        header = {"alg": "HS256", "kid": self._signing_key.key_id, "typ": "at+jwt"}
        claims = {
            "iss": self._configuration.issuer,
            "aud": self._configuration.audience,
            "sub": subject,
            "tenant_id": tenant_id,
            "client_id": client_id,
            "jti": token_id,
            "scope": " ".join(sorted(scopes)),
            "iat": issued_at,
            "nbf": issued_at,
            "exp": issued_at + lifetime,
        }
        encoded_header = _b64url_encode(
            json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        encoded_claims = _b64url_encode(
            json.dumps(claims, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        signing_input = f"{encoded_header}.{encoded_claims}".encode("ascii")
        signature = hmac.new(
            self._signing_key.secret, signing_input, hashlib.sha256
        ).digest()
        return f"{encoded_header}.{encoded_claims}.{_b64url_encode(signature)}"


class ScopeTenantAuthorizer:
    """Fail-closed OAuth scope and exact-tenant authorization."""

    def authorize(
        self,
        principal: ValidatedPrincipal,
        required_scope: str,
        resource_tenant_id: str,
        /,
    ) -> None:
        if principal.tenant_id != resource_tenant_id:
            raise AuthorizationError(
                "Principal is not authorized for this tenant resource."
            )
        if required_scope not in principal.scopes:
            raise AuthorizationError("Bearer token does not grant the required scope.")


__all__ = [
    "AccessTokenValidator",
    "Clock",
    "HMACOIDCTokenIssuer",
    "HMACOIDCTokenValidator",
    "HMACSigningKey",
    "OIDCConfiguration",
    "ResourceAuthorizer",
    "ScopeTenantAuthorizer",
    "SystemClock",
]
