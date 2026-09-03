#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""OIDC/JWKS and mutually authenticated workload identity policies."""

from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping, Protocol
from urllib.parse import urlparse

from ._auth import Clock, OIDCConfiguration, SystemClock
from ._contracts import AuthenticationError, ValidatedPrincipal
from ._schedulers import HTTPTransport


def _b64url_decode(value: str) -> bytes:
    try:
        return base64.b64decode(
            value + "=" * (-len(value) % 4), altchars=b"-_", validate=True
        )
    except (ValueError, UnicodeEncodeError) as error:
        raise AuthenticationError("JWT contains invalid base64url data.") from error


def _b64url_int(value: str) -> int:
    decoded = _b64url_decode(value)
    if not decoded:
        raise AuthenticationError("JWK integer is empty.")
    return int.from_bytes(decoded, "big")


def _json_object(segment: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_b64url_decode(segment).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthenticationError(f"JWT {label} is not valid JSON.") from error
    if not isinstance(value, dict):
        raise AuthenticationError(f"JWT {label} must be a JSON object.")
    return value


def _required_string(values: Mapping[str, Any], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value.strip():
        raise AuthenticationError(f"JWT requires a nonempty {name!r} claim.")
    return value


def _required_time(values: Mapping[str, Any], name: str) -> int:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AuthenticationError(f"JWT requires an integer {name!r} claim.")
    return value


@dataclass(frozen=True, slots=True)
class JSONWebKeySet:
    keys: tuple[Mapping[str, object], ...]
    fetched_at: int
    expires_at: int
    revoked_key_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.fetched_at < 0 or self.expires_at <= self.fetched_at:
            raise ValueError("JWKS cache validity interval is invalid.")
        normalized: list[Mapping[str, object]] = []
        identifiers: set[str] = set()
        for key in self.keys:
            copied = json.loads(
                json.dumps(dict(key), allow_nan=False, separators=(",", ":"))
            )
            kid = copied.get("kid")
            if not isinstance(kid, str) or not kid or kid in identifiers:
                raise ValueError("JWKS keys require unique nonempty kid values.")
            identifiers.add(kid)
            normalized.append(MappingProxyType(copied))
        if not normalized:
            raise ValueError("JWKS must contain at least one key.")
        object.__setattr__(self, "keys", tuple(normalized))
        object.__setattr__(self, "revoked_key_ids", frozenset(self.revoked_key_ids))

    def key(self, key_id: str, algorithm: str, /) -> Mapping[str, object] | None:
        if key_id in self.revoked_key_ids:
            return None
        for key in self.keys:
            if key.get("kid") != key_id:
                continue
            configured_algorithm = key.get("alg")
            if configured_algorithm is not None and configured_algorithm != algorithm:
                continue
            if key.get("use", "sig") != "sig":
                continue
            operations = key.get("key_ops")
            if operations is not None and (
                not isinstance(operations, list) or "verify" not in operations
            ):
                continue
            return key
        return None


class JWKSProvider(Protocol):
    def get(self, issuer: str, /, *, force_refresh: bool = False) -> JSONWebKeySet: ...


class StaticJWKSProvider:
    """Mutable-injection reference used to model rotation without network effects."""

    def __init__(self, issuer: str, key_set: JSONWebKeySet, /):
        self._issuer = issuer
        self._key_set = key_set
        self._lock = threading.RLock()

    def rotate(self, key_set: JSONWebKeySet, /) -> None:
        with self._lock:
            self._key_set = key_set

    def get(self, issuer: str, /, *, force_refresh: bool = False) -> JSONWebKeySet:
        if issuer != self._issuer:
            raise AuthenticationError("No JWKS is configured for the token issuer.")
        with self._lock:
            return self._key_set


class HTTPSJWKSProvider:
    """Authenticated injected-transport JWKS fetcher with an explicit local cache."""

    def __init__(
        self,
        issuer: str,
        jwks_uri: str,
        transport: HTTPTransport,
        /,
        *,
        clock: Clock | None = None,
        bearer_token: str | None = None,
        default_cache_seconds: int = 300,
    ):
        issuer_url = urlparse(issuer)
        jwks_url = urlparse(jwks_uri)
        for value, label in ((issuer_url, "issuer"), (jwks_url, "JWKS URI")):
            if (
                value.scheme != "https"
                or not value.hostname
                or value.username is not None
                or value.password is not None
                or value.fragment
            ):
                raise ValueError(
                    f"OIDC {label} must be an absolute HTTPS URL without credentials."
                )
        if bearer_token is not None and (
            not bearer_token or "\r" in bearer_token or "\n" in bearer_token
        ):
            raise ValueError("JWKS bearer credential must be nonempty and single-line.")
        if default_cache_seconds <= 0:
            raise ValueError("JWKS default cache duration must be positive.")
        self._issuer = issuer
        self._uri = jwks_uri
        self._transport = transport
        self._clock = SystemClock() if clock is None else clock
        self._bearer_token = bearer_token
        self._default_cache = default_cache_seconds
        self._cached: JSONWebKeySet | None = None
        self._lock = threading.RLock()

    def get(self, issuer: str, /, *, force_refresh: bool = False) -> JSONWebKeySet:
        with self._lock:
            return self._get(issuer, force_refresh=force_refresh)

    def _get(self, issuer: str, /, *, force_refresh: bool) -> JSONWebKeySet:
        if issuer != self._issuer:
            raise AuthenticationError("No JWKS is configured for the token issuer.")
        now = self._clock.now()
        if (
            not force_refresh
            and self._cached is not None
            and now < self._cached.expires_at
        ):
            return self._cached
        headers = {"Accept": "application/json"}
        if self._bearer_token is not None:
            headers["Authorization"] = f"Bearer {self._bearer_token}"
        response = self._transport.request("GET", self._uri, headers=headers)
        if response.status != 200:
            raise AuthenticationError(
                "The issuer JWKS endpoint did not return a trusted key set."
            )
        try:
            decoded = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AuthenticationError(
                "The issuer JWKS response is not valid JSON."
            ) from error
        if not isinstance(decoded, dict) or not isinstance(decoded.get("keys"), list):
            raise AuthenticationError("The issuer JWKS response has no keys collection.")
        lifetime = _cache_max_age(
            response.headers.get("Cache-Control"), self._default_cache
        )
        self._cached = JSONWebKeySet(tuple(decoded["keys"]), now, now + lifetime)
        return self._cached


def _cache_max_age(value: str | None, default: int) -> int:
    if value is None:
        return default
    for directive in value.split(","):
        name, separator, raw = directive.strip().partition("=")
        if separator and name.lower() == "max-age" and raw.isdigit():
            return max(1, int(raw))
    return default


class OIDCJWKSTokenValidator:
    """Strict JWT signature/claim validator with a one-shot rotation refresh."""

    def __init__(
        self,
        configuration: OIDCConfiguration,
        key_provider: JWKSProvider,
        /,
        *,
        clock: Clock | None = None,
        accepted_algorithms: frozenset[str] = frozenset({"RS256", "EdDSA"}),
    ):
        if not accepted_algorithms or not accepted_algorithms <= {"RS256", "EdDSA"}:
            raise ValueError(
                "Accepted OIDC algorithms must be an explicit RS256/EdDSA subset."
            )
        self._configuration = configuration
        self._provider = key_provider
        self._clock = SystemClock() if clock is None else clock
        self._algorithms = accepted_algorithms

    def validate(self, token: str, /) -> ValidatedPrincipal:
        parts = token.split(".")
        if len(parts) != 3 or any(not part for part in parts):
            raise AuthenticationError("Bearer token must be a compact signed JWT.")
        header = _json_object(parts[0], "header")
        claims = _json_object(parts[1], "payload")
        if "crit" in header:
            raise AuthenticationError("JWT critical extensions are not supported.")
        algorithm = _required_string(header, "alg")
        if algorithm not in self._algorithms or algorithm == "none":
            raise AuthenticationError("JWT signing algorithm is not accepted.")
        if header.get("typ") not in ("at+jwt", "application/at+jwt"):
            raise AuthenticationError("JWT type is not an OAuth access token.")
        key_id = _required_string(header, "kid")
        signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
        signature = _b64url_decode(parts[2])
        verified = False
        now = self._clock.now()
        skew = self._configuration.clock_skew_seconds
        for refresh in (False, True):
            key_set = self._provider.get(
                self._configuration.issuer, force_refresh=refresh
            )
            if key_set.fetched_at > now + skew:
                raise AuthenticationError("The issuer JWKS was fetched in the future.")
            if now >= key_set.expires_at:
                if refresh:
                    raise AuthenticationError("The issuer JWKS cache is expired.")
                continue
            key = key_set.key(key_id, algorithm)
            if key is not None and _verify_jwk(key, algorithm, signing_input, signature):
                verified = True
                break
        if not verified:
            raise AuthenticationError("JWT signing key or signature is not trusted.")
        return self._validate_claims(claims)

    def _validate_claims(self, claims: Mapping[str, Any]) -> ValidatedPrincipal:
        issuer = _required_string(claims, "iss")
        if issuer != self._configuration.issuer:
            raise AuthenticationError("JWT issuer is not accepted.")
        raw_audience = claims.get("aud")
        if isinstance(raw_audience, str):
            audiences = (raw_audience,)
        elif (
            isinstance(raw_audience, list)
            and raw_audience
            and all(isinstance(value, str) and value for value in raw_audience)
        ):
            audiences = tuple(raw_audience)
        else:
            raise AuthenticationError("JWT audience claim is invalid.")
        if self._configuration.audience not in audiences:
            raise AuthenticationError("JWT audience is not accepted.")
        issued_at = _required_time(claims, "iat")
        not_before = _required_time(claims, "nbf")
        expires_at = _required_time(claims, "exp")
        now = self._clock.now()
        skew = self._configuration.clock_skew_seconds
        if issued_at > now + skew or not_before > now + skew:
            raise AuthenticationError("JWT is not active yet.")
        if expires_at <= now - skew:
            raise AuthenticationError("JWT has expired.")
        if (
            expires_at <= issued_at
            or expires_at - issued_at > self._configuration.maximum_token_lifetime_seconds
        ):
            raise AuthenticationError("JWT lifetime is not accepted.")
        scope_value = claims.get("scope")
        if not isinstance(scope_value, str):
            raise AuthenticationError("JWT scope claim must be a string.")
        scopes = frozenset(scope_value.split())
        if not scopes:
            raise AuthenticationError("JWT must grant at least one scope.")
        client_id_claim = claims.get("client_id")
        authorized_party = claims.get("azp")
        if len(audiences) > 1:
            if not isinstance(authorized_party, str) or not authorized_party:
                raise AuthenticationError(
                    "JWT with multiple audiences requires an azp claim."
                )
            if client_id_claim is not None and client_id_claim != authorized_party:
                raise AuthenticationError("JWT authorized-party claim is inconsistent.")
        client_id = client_id_claim if client_id_claim is not None else authorized_party
        if not isinstance(client_id, str) or not client_id:
            raise AuthenticationError("JWT requires a client_id or azp claim.")
        return ValidatedPrincipal(
            _required_string(claims, "sub"),
            _required_string(claims, "tenant_id"),
            issuer,
            self._configuration.audience,
            client_id,
            _required_string(claims, "jti"),
            scopes,
            issued_at,
            expires_at,
        )


def _verify_jwk(
    key: Mapping[str, object],
    algorithm: str,
    message: bytes,
    signature: bytes,
) -> bool:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa
    except ImportError as error:
        raise RuntimeError(
            "OIDC asymmetric JWT verification requires the optional "
            "'cryptography' package."
        ) from error
    try:
        if algorithm == "RS256":
            if (
                key.get("kty") != "RSA"
                or not isinstance(key.get("n"), str)
                or not isinstance(key.get("e"), str)
            ):
                return False
            public_key = rsa.RSAPublicNumbers(
                _b64url_int(key["e"]), _b64url_int(key["n"])
            ).public_key()
            public_key.verify(signature, message, padding.PKCS1v15(), hashes.SHA256())
        elif algorithm == "EdDSA":
            if (
                key.get("kty") != "OKP"
                or key.get("crv") != "Ed25519"
                or not isinstance(key.get("x"), str)
            ):
                return False
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(
                _b64url_decode(key["x"])
            )
            public_key.verify(signature, message)
        else:
            return False
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


@dataclass(frozen=True, slots=True)
class WorkloadCertificate:
    san_uris: tuple[str, ...]
    not_before: int
    not_after: int
    sha256_fingerprint: str
    issuer_sha256_fingerprint: str
    client_auth: bool = True

    def __post_init__(self) -> None:
        if not self.san_uris or any(not value for value in self.san_uris):
            raise ValueError("A workload certificate requires URI SAN identities.")
        if self.not_before < 0 or self.not_after <= self.not_before:
            raise ValueError("Workload certificate validity interval is invalid.")
        for digest in (self.sha256_fingerprint, self.issuer_sha256_fingerprint):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(
                    "Certificate fingerprints must be lowercase SHA-256 digests."
                )
        object.__setattr__(self, "san_uris", tuple(self.san_uris))


@dataclass(frozen=True, slots=True)
class WorkloadIdentity:
    spiffe_id: str
    trust_domain: str
    certificate_fingerprint: str
    expires_at: int


@dataclass(frozen=True, slots=True)
class MTLSCertificatePolicy:
    trust_domain: str
    trusted_issuer_fingerprints: frozenset[str]
    maximum_lifetime_seconds: int = 86_400
    clock_skew_seconds: int = 30

    def __post_init__(self) -> None:
        if not self.trust_domain or "/" in self.trust_domain:
            raise ValueError("mTLS trust_domain must be a DNS authority.")
        if not self.trusted_issuer_fingerprints:
            raise ValueError("mTLS policy requires an explicit issuer trust set.")
        for value in self.trusted_issuer_fingerprints:
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(
                    "Trusted issuer fingerprints must be lowercase SHA-256 digests."
                )
        if self.maximum_lifetime_seconds <= 0 or self.clock_skew_seconds < 0:
            raise ValueError("mTLS lifetime and skew policy are invalid.")

    def validate(
        self, certificate: WorkloadCertificate, at_time: int, /
    ) -> WorkloadIdentity:
        if not certificate.client_auth:
            raise AuthenticationError(
                "mTLS certificate is not valid for client authentication."
            )
        if certificate.issuer_sha256_fingerprint not in self.trusted_issuer_fingerprints:
            raise AuthenticationError("mTLS certificate issuer is not trusted.")
        if certificate.not_after - certificate.not_before > self.maximum_lifetime_seconds:
            raise AuthenticationError("mTLS certificate lifetime exceeds policy.")
        if at_time + self.clock_skew_seconds < certificate.not_before:
            raise AuthenticationError("mTLS certificate is not active yet.")
        if at_time - self.clock_skew_seconds >= certificate.not_after:
            raise AuthenticationError("mTLS certificate has expired.")
        if len(certificate.san_uris) != 1:
            raise AuthenticationError(
                "mTLS certificate must contain exactly one URI SAN identity."
            )
        identity = certificate.san_uris[0]
        parsed = urlparse(identity)
        if (
            parsed.scheme != "spiffe"
            or parsed.netloc != self.trust_domain
            or not parsed.path.startswith("/")
            or parsed.path.startswith("//")
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise AuthenticationError(
                "mTLS certificate must contain exactly one accepted SPIFFE URI SAN."
            )
        return WorkloadIdentity(
            identity,
            self.trust_domain,
            certificate.sha256_fingerprint,
            certificate.not_after,
        )


class X509WorkloadCertificateValidator:
    """Parses and verifies one leaf against explicitly supplied issuer certificates."""

    def __init__(
        self,
        policy: MTLSCertificatePolicy,
        issuer_certificates: tuple[bytes, ...],
        /,
        *,
        clock: Clock | None = None,
    ):
        if not issuer_certificates:
            raise ValueError("At least one issuer certificate is required.")
        self._policy = policy
        self._issuers = tuple(bytes(value) for value in issuer_certificates)
        self._clock = SystemClock() if clock is None else clock

    def validate(self, certificate: bytes, /) -> WorkloadIdentity:
        try:
            from cryptography import x509
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import (
                ec,
                ed448,
                ed25519,
                padding,
                rsa,
            )
            from cryptography.x509.oid import ExtendedKeyUsageOID
        except ImportError as error:
            raise RuntimeError(
                "X.509 workload identity validation requires the optional "
                "'cryptography' package."
            ) from error
        try:
            leaf = (
                x509.load_pem_x509_certificate(certificate)
                if certificate.lstrip().startswith(b"-----BEGIN")
                else x509.load_der_x509_certificate(certificate)
            )
            issuer_objects = [
                (
                    x509.load_pem_x509_certificate(value)
                    if value.lstrip().startswith(b"-----BEGIN")
                    else x509.load_der_x509_certificate(value)
                )
                for value in self._issuers
            ]
        except ValueError as error:
            raise AuthenticationError("mTLS certificate encoding is invalid.") from error
        issuer = next(
            (value for value in issuer_objects if value.subject == leaf.issuer),
            None,
        )
        if issuer is None:
            raise AuthenticationError("mTLS leaf issuer is not configured.")
        now = datetime.fromtimestamp(self._clock.now(), tz=timezone.utc)
        if not issuer.not_valid_before_utc <= now < issuer.not_valid_after_utc:
            raise AuthenticationError("mTLS issuer certificate is expired or inactive.")
        try:
            constraints = issuer.extensions.get_extension_for_class(
                x509.BasicConstraints
            ).value
            if not constraints.ca:
                raise AuthenticationError(
                    "mTLS issuer certificate is not a certificate authority."
                )
        except x509.ExtensionNotFound as error:
            raise AuthenticationError(
                "mTLS issuer certificate lacks CA constraints."
            ) from error
        public_key = issuer.public_key()
        try:
            if isinstance(public_key, rsa.RSAPublicKey):
                public_key.verify(
                    leaf.signature,
                    leaf.tbs_certificate_bytes,
                    padding.PKCS1v15(),
                    leaf.signature_hash_algorithm,
                )
            elif isinstance(public_key, ec.EllipticCurvePublicKey):
                public_key.verify(
                    leaf.signature,
                    leaf.tbs_certificate_bytes,
                    ec.ECDSA(leaf.signature_hash_algorithm),
                )
            elif isinstance(
                public_key,
                (ed25519.Ed25519PublicKey, ed448.Ed448PublicKey),
            ):
                public_key.verify(leaf.signature, leaf.tbs_certificate_bytes)
            else:
                raise AuthenticationError("mTLS issuer key type is not supported.")
        except Exception as error:
            if isinstance(error, AuthenticationError):
                raise
            raise AuthenticationError("mTLS certificate signature is invalid.") from error
        try:
            san_uris = tuple(
                leaf.extensions.get_extension_for_class(
                    x509.SubjectAlternativeName
                ).value.get_values_for_type(x509.UniformResourceIdentifier)
            )
        except x509.ExtensionNotFound:
            san_uris = ()
        try:
            usage = leaf.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
            client_auth = ExtendedKeyUsageOID.CLIENT_AUTH in usage
        except x509.ExtensionNotFound:
            client_auth = False
        not_before = int(leaf.not_valid_before_utc.timestamp())
        not_after = int(leaf.not_valid_after_utc.timestamp())
        return self._policy.validate(
            WorkloadCertificate(
                san_uris,
                not_before,
                not_after,
                leaf.fingerprint(hashes.SHA256()).hex(),
                issuer.fingerprint(hashes.SHA256()).hex(),
                client_auth,
            ),
            self._clock.now(),
        )


__all__ = [
    "HTTPSJWKSProvider",
    "JSONWebKeySet",
    "JWKSProvider",
    "MTLSCertificatePolicy",
    "OIDCJWKSTokenValidator",
    "StaticJWKSProvider",
    "WorkloadCertificate",
    "WorkloadIdentity",
    "X509WorkloadCertificateValidator",
]
