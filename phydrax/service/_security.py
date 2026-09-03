#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scoped secret handles and asymmetric signing/trust primitives."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
from dataclasses import dataclass, replace
from typing import Protocol

from ._auth import Clock, SystemClock
from ._contracts import AuthorizationError, IntegrityError, ResourceNotFound


@dataclass(frozen=True, slots=True)
class ScopedSecretHandle:
    handle_id: str
    tenant_id: str
    scopes: frozenset[str]
    created_at: int
    expires_at: int
    key_version: str

    def __post_init__(self) -> None:
        if any(
            not value or value != value.strip()
            for value in (self.handle_id, self.tenant_id, self.key_version)
        ):
            raise ValueError(
                "Secret handle identifiers must be nonempty canonical strings."
            )
        if not self.scopes or any(
            not value or value != value.strip() for value in self.scopes
        ):
            raise ValueError("Secret handles require explicit nonempty scopes.")
        if self.created_at < 0 or self.expires_at <= self.created_at:
            raise ValueError("Secret handle validity interval is invalid.")
        object.__setattr__(self, "scopes", frozenset(self.scopes))


class SecretHandleBroker(Protocol):
    def issue(
        self,
        tenant_id: str,
        secret_value: bytes,
        scopes: frozenset[str],
        /,
        *,
        lifetime_seconds: int,
        key_version: str,
    ) -> ScopedSecretHandle: ...

    def resolve(
        self, handle: ScopedSecretHandle, tenant_id: str, required_scope: str, /
    ) -> bytes: ...
    def revoke(self, handle: ScopedSecretHandle, tenant_id: str, /) -> None: ...


@dataclass(slots=True)
class _SecretEntry:
    handle: ScopedSecretHandle
    value: bytearray


class LocalSecretHandleBroker:
    """Process-local reference vault; handles and reprs never contain secret values."""

    def __init__(
        self,
        /,
        *,
        clock: Clock | None = None,
        maximum_lifetime_seconds: int = 900,
    ):
        if maximum_lifetime_seconds <= 0:
            raise ValueError("Maximum secret-handle lifetime must be positive.")
        self._clock = SystemClock() if clock is None else clock
        self._maximum_lifetime = maximum_lifetime_seconds
        self._entries: dict[tuple[str, str], _SecretEntry] = {}
        self._lock = threading.RLock()

    def issue(
        self,
        tenant_id: str,
        secret_value: bytes,
        scopes: frozenset[str],
        /,
        *,
        lifetime_seconds: int,
        key_version: str,
    ) -> ScopedSecretHandle:
        if not isinstance(secret_value, bytes) or not secret_value:
            raise ValueError("Secret values must be nonempty bytes.")
        if not 0 < lifetime_seconds <= self._maximum_lifetime:
            raise ValueError(
                "Secret-handle lifetime exceeds the configured short-lived bound."
            )
        now = self._clock.now()
        with self._lock:
            while True:
                handle_id = secrets.token_urlsafe(24)
                if (tenant_id, handle_id) not in self._entries:
                    break
            handle = ScopedSecretHandle(
                handle_id,
                tenant_id,
                scopes,
                now,
                now + lifetime_seconds,
                key_version,
            )
            self._entries[(tenant_id, handle.handle_id)] = _SecretEntry(
                handle, bytearray(secret_value)
            )
            return handle

    def resolve(
        self, handle: ScopedSecretHandle, tenant_id: str, required_scope: str, /
    ) -> bytes:
        # Tenant and scope checks intentionally precede the backing-store lookup.
        if handle.tenant_id != tenant_id:
            raise AuthorizationError(
                "Secret handle does not belong to the requesting tenant."
            )
        if required_scope not in handle.scopes:
            raise AuthorizationError("Secret handle does not grant the required scope.")
        if self._clock.now() >= handle.expires_at:
            self.revoke(handle, tenant_id)
            raise AuthorizationError("Secret handle has expired.")
        with self._lock:
            entry = self._entries.get((tenant_id, handle.handle_id))
            if entry is None or entry.handle != handle:
                raise ResourceNotFound("Secret handle does not exist.")
            return bytes(entry.value)

    def revoke(self, handle: ScopedSecretHandle, tenant_id: str, /) -> None:
        if handle.tenant_id != tenant_id:
            raise AuthorizationError(
                "Secret handle does not belong to the requesting tenant."
            )
        with self._lock:
            entry = self._entries.pop((tenant_id, handle.handle_id), None)
            if entry is not None:
                entry.value[:] = b"\x00" * len(entry.value)


@dataclass(frozen=True, slots=True)
class SignatureEnvelope:
    key_id: str
    algorithm: str
    purpose: str
    signed_at: int
    payload_sha256: str
    signature: bytes

    def __post_init__(self) -> None:
        if any(
            not value or value != value.strip()
            for value in (self.key_id, self.algorithm, self.purpose)
        ):
            raise ValueError("Signature identifiers must be nonempty canonical strings.")
        if self.signed_at < 0:
            raise ValueError("Signature timestamp must be nonnegative.")
        if len(self.payload_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.payload_sha256
        ):
            raise ValueError("Signature payload digest must be lowercase SHA-256.")
        if not isinstance(self.signature, bytes) or not self.signature:
            raise ValueError("Signature bytes must be nonempty.")

    @property
    def signature_base64url(self) -> str:
        return base64.urlsafe_b64encode(self.signature).rstrip(b"=").decode("ascii")


class AsymmetricSigner(Protocol):
    @property
    def key_id(self) -> str: ...
    @property
    def algorithm(self) -> str: ...
    def sign(
        self, payload: bytes, /, *, purpose: str, signed_at: int
    ) -> SignatureEnvelope: ...


class AsymmetricVerifier(Protocol):
    @property
    def key_id(self) -> str: ...
    @property
    def algorithm(self) -> str: ...
    def verify(self, payload: bytes, envelope: SignatureEnvelope, /) -> None: ...


def _signing_message(payload: bytes, purpose: str, signed_at: int) -> bytes:
    if not isinstance(payload, bytes):
        raise TypeError("Signed payload must be bytes.")
    if not purpose or purpose != purpose.strip() or "\x00" in purpose:
        raise ValueError("Signature purpose must be a canonical nonempty string.")
    if type(signed_at) is not int or signed_at < 0:
        raise ValueError("Signature timestamp must be a nonnegative integer.")
    return (
        b"phydrax-signature-v1\x00"
        + purpose.encode("utf-8")
        + b"\x00"
        + str(signed_at).encode("ascii")
        + b"\x00"
        + payload
    )


class Ed25519Signer:
    """Optional-cryptography Ed25519 signer; private bytes never leave this object."""

    algorithm = "Ed25519"

    def __init__(self, key_id: str, private_key: bytes, /):
        if not key_id:
            raise ValueError("Ed25519 key_id must be nonempty.")
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
        except ImportError as error:
            raise RuntimeError(
                "Ed25519 signing requires the optional 'cryptography' package."
            ) from error
        if len(private_key) != 32:
            raise ValueError("Ed25519 private keys must be 32-byte seeds.")
        self._key_id = key_id
        self._private_key = Ed25519PrivateKey.from_private_bytes(private_key)

    @property
    def key_id(self) -> str:
        return self._key_id

    @property
    def public_key_bytes(self) -> bytes:
        try:
            from cryptography.hazmat.primitives import serialization
        except (
            ImportError
        ) as error:  # pragma: no cover - constructor already establishes boundary
            raise RuntimeError(
                "Ed25519 requires the optional 'cryptography' package."
            ) from error
        return self._private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )

    def sign(
        self, payload: bytes, /, *, purpose: str, signed_at: int
    ) -> SignatureEnvelope:
        message = _signing_message(payload, purpose, signed_at)
        return SignatureEnvelope(
            self.key_id,
            self.algorithm,
            purpose,
            signed_at,
            hashlib.sha256(payload).hexdigest(),
            self._private_key.sign(message),
        )


class Ed25519Verifier:
    algorithm = "Ed25519"

    def __init__(self, key_id: str, public_key: bytes, /):
        if not key_id:
            raise ValueError("Ed25519 key_id must be nonempty.")
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except ImportError as error:
            raise RuntimeError(
                "Ed25519 verification requires the optional 'cryptography' package."
            ) from error
        if len(public_key) != 32:
            raise ValueError("Ed25519 public keys must contain 32 bytes.")
        self._key_id = key_id
        self._public_key = Ed25519PublicKey.from_public_bytes(public_key)

    @property
    def key_id(self) -> str:
        return self._key_id

    def verify(self, payload: bytes, envelope: SignatureEnvelope, /) -> None:
        if envelope.key_id != self.key_id or envelope.algorithm != self.algorithm:
            raise IntegrityError(
                "Signature key or algorithm does not match the verifier."
            )
        if not hmac.compare_digest(
            hashlib.sha256(payload).hexdigest(), envelope.payload_sha256
        ):
            raise IntegrityError("Signature payload digest does not match.")
        try:
            self._public_key.verify(
                envelope.signature,
                _signing_message(payload, envelope.purpose, envelope.signed_at),
            )
        except Exception as error:
            raise IntegrityError("Ed25519 signature is invalid.") from error


class KMSSigningProvider(Protocol):
    def sign(self, key_id: str, algorithm: str, message: bytes, /) -> bytes: ...


class KMSVerificationProvider(Protocol):
    def verify(
        self, key_id: str, algorithm: str, message: bytes, signature: bytes, /
    ) -> bool: ...


class KMSSigner:
    def __init__(self, key_id: str, algorithm: str, provider: KMSSigningProvider, /):
        if not key_id or not algorithm:
            raise ValueError("KMS key and algorithm must be nonempty.")
        self._key_id = key_id
        self._algorithm = algorithm
        self._provider = provider

    @property
    def key_id(self) -> str:
        return self._key_id

    @property
    def algorithm(self) -> str:
        return self._algorithm

    def sign(
        self, payload: bytes, /, *, purpose: str, signed_at: int
    ) -> SignatureEnvelope:
        message = _signing_message(payload, purpose, signed_at)
        signature = self._provider.sign(self.key_id, self.algorithm, message)
        return SignatureEnvelope(
            self.key_id,
            self.algorithm,
            purpose,
            signed_at,
            hashlib.sha256(payload).hexdigest(),
            signature,
        )


class KMSVerifier:
    def __init__(self, key_id: str, algorithm: str, provider: KMSVerificationProvider, /):
        if not key_id or not algorithm:
            raise ValueError("KMS key and algorithm must be nonempty.")
        self._key_id = key_id
        self._algorithm = algorithm
        self._provider = provider

    @property
    def key_id(self) -> str:
        return self._key_id

    @property
    def algorithm(self) -> str:
        return self._algorithm

    def verify(self, payload: bytes, envelope: SignatureEnvelope, /) -> None:
        if envelope.key_id != self.key_id or envelope.algorithm != self.algorithm:
            raise IntegrityError(
                "Signature key or algorithm does not match the KMS verifier."
            )
        if not hmac.compare_digest(
            hashlib.sha256(payload).hexdigest(), envelope.payload_sha256
        ):
            raise IntegrityError("Signature payload digest does not match.")
        if not self._provider.verify(
            self.key_id,
            self.algorithm,
            _signing_message(payload, envelope.purpose, envelope.signed_at),
            envelope.signature,
        ):
            raise IntegrityError("KMS signature is invalid.")


@dataclass(frozen=True, slots=True)
class SigningKeyTrustRecord:
    key_id: str
    algorithm: str
    activated_at: int
    expires_at: int
    revoked_at: int | None = None
    supersedes_key_id: str | None = None

    def __post_init__(self) -> None:
        if not self.key_id or not self.algorithm:
            raise ValueError("Signing trust identifiers must be nonempty.")
        if self.activated_at < 0 or self.expires_at <= self.activated_at:
            raise ValueError("Signing key validity interval is invalid.")
        if self.revoked_at is not None and self.revoked_at < self.activated_at:
            raise ValueError("Signing key revocation cannot precede activation.")
        if self.supersedes_key_id == self.key_id:
            raise ValueError("A signing key cannot supersede itself.")


class SigningTrustStore:
    """Explicit key rotation/revocation registry with fail-closed verification."""

    def __init__(self):
        self._records: dict[str, SigningKeyTrustRecord] = {}
        self._verifiers: dict[str, AsymmetricVerifier] = {}
        self._lock = threading.RLock()

    def trust(
        self, record: SigningKeyTrustRecord, verifier: AsymmetricVerifier, /
    ) -> None:
        if verifier.key_id != record.key_id or verifier.algorithm != record.algorithm:
            raise ValueError("Trust record does not match its verifier.")
        with self._lock:
            if record.key_id in self._records:
                raise ValueError("Signing key is already registered.")
            if (
                record.supersedes_key_id is not None
                and record.supersedes_key_id not in self._records
            ):
                raise ValueError("Superseded signing key is not registered.")
            self._records[record.key_id] = record
            self._verifiers[record.key_id] = verifier

    def revoke(self, key_id: str, revoked_at: int, /) -> SigningKeyTrustRecord:
        with self._lock:
            record = self._records.get(key_id)
            if record is None:
                raise ResourceNotFound("Signing key is not trusted.")
            if record.revoked_at is not None and record.revoked_at != revoked_at:
                raise IntegrityError("Signing key has a conflicting revocation record.")
            updated = replace(record, revoked_at=revoked_at)
            updated.__post_init__()
            self._records[key_id] = updated
            return updated

    def verify(
        self, payload: bytes, envelope: SignatureEnvelope, /, *, at_time: int
    ) -> None:
        with self._lock:
            record = self._records.get(envelope.key_id)
            verifier = self._verifiers.get(envelope.key_id)
        if record is None or verifier is None:
            raise IntegrityError("Signature key is not trusted.")
        if record.algorithm != envelope.algorithm:
            raise IntegrityError("Signature algorithm is not trusted for this key.")
        if not record.activated_at <= envelope.signed_at < record.expires_at:
            raise IntegrityError(
                "Signature was created outside the key validity interval."
            )
        if at_time < envelope.signed_at or at_time >= record.expires_at:
            raise IntegrityError(
                "Signature verification time is outside its trust interval."
            )
        # Revocation invalidates every signature from the compromised key, including old ones.
        if record.revoked_at is not None:
            raise IntegrityError("Signature key has been revoked.")
        verifier.verify(payload, envelope)

    def records(self) -> tuple[SigningKeyTrustRecord, ...]:
        with self._lock:
            return tuple(
                sorted(
                    self._records.values(),
                    key=lambda value: (value.activated_at, value.key_id),
                )
            )


__all__ = [
    "AsymmetricSigner",
    "AsymmetricVerifier",
    "Ed25519Signer",
    "Ed25519Verifier",
    "KMSSigner",
    "KMSSigningProvider",
    "KMSVerifier",
    "KMSVerificationProvider",
    "LocalSecretHandleBroker",
    "ScopedSecretHandle",
    "SecretHandleBroker",
    "SignatureEnvelope",
    "SigningKeyTrustRecord",
    "SigningTrustStore",
]
