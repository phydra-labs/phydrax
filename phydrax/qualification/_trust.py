#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Purpose-separated qualification trust over the service signing primitives."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

from .._fingerprint import canonical_fingerprint, canonical_json
from ._evidence import SupportDependency
from ._registry import ReleaseGateEvidence, ReleaseIndex


if TYPE_CHECKING:
    from ..service._security import AsymmetricSigner, SignatureEnvelope, SigningTrustStore

QUALIFICATION_ROLES = frozenset(
    (
        "criterion-approver",
        "executor",
        "scientific-reviewer",
        "entry-decision-authority",
        "release-authority",
        "channel-promoter",
    )
)


class CanonicalRecord(Protocol):
    def to_record(self) -> dict[str, object]: ...


def _record(value: Mapping[str, object] | CanonicalRecord) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    return value.to_record()


@dataclass(frozen=True, slots=True)
class SignedQualificationRecord:
    content_json: str
    role: str
    issued_at: int
    expires_at: int
    signature: SignatureEnvelope

    def __post_init__(self) -> None:
        if self.role not in QUALIFICATION_ROLES:
            raise ValueError("Unknown qualification signing role.")
        if (
            type(self.issued_at) is not int
            or type(self.expires_at) is not int
            or not 0 <= self.issued_at < self.expires_at
        ):
            raise ValueError("Signed record requires an ordered time interval.")

    @property
    def signed_payload(self) -> bytes:
        return canonical_json(
            {
                "kind": "signed-qualification-record",
                "content_json": self.content_json,
                "role": self.role,
                "issued_at": self.issued_at,
                "expires_at": self.expires_at,
            }
        ).encode()

    @property
    def attestation_id(self) -> str:
        return canonical_fingerprint(
            {
                "payload": self.signed_payload.decode(),
                "key_id": self.signature.key_id,
                "signature": self.signature.signature.hex(),
            }
        )

    @classmethod
    def sign(
        cls,
        record: object,
        signer: AsymmetricSigner,
        /,
        *,
        role: str,
        issued_at: int,
        expires_at: int,
    ) -> SignedQualificationRecord:
        content = canonical_json(_record(record))
        payload = canonical_json(
            {
                "kind": "signed-qualification-record",
                "content_json": content,
                "role": role,
                "issued_at": issued_at,
                "expires_at": expires_at,
            }
        ).encode()
        signature = signer.sign(
            payload, purpose=f"qualification.{role}", signed_at=issued_at
        )
        return cls(content, role, issued_at, expires_at, signature)

    def to_record(self) -> dict[str, object]:
        signature = self.signature
        return {
            "content_json": self.content_json,
            "role": self.role,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "signature": {
                "key_id": signature.key_id,
                "algorithm": signature.algorithm,
                "purpose": signature.purpose,
                "signed_at": signature.signed_at,
                "payload_sha256": signature.payload_sha256,
                "signature": signature.signature.hex(),
            },
            "attestation_id": self.attestation_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> SignedQualificationRecord:
        from ..service._security import SignatureEnvelope

        raw = dict(record["signature"])
        raw["signature"] = bytes.fromhex(raw["signature"])
        result = cls(
            record["content_json"],
            record["role"],
            record["issued_at"],
            record["expires_at"],
            SignatureEnvelope(**raw),
        )
        if record["attestation_id"] != result.attestation_id:
            raise ValueError("Signed qualification record content address is invalid.")
        return result


class QualificationRoleTrust:
    """Public verifiers plus disjoint role memberships; never installs authority keys."""

    def __init__(self, store: SigningTrustStore, roles: Mapping[str, Sequence[str]], /):
        if set(roles) != QUALIFICATION_ROLES:
            raise ValueError("All six qualification roles must be configured explicitly.")
        normalized = {role: frozenset(keys) for role, keys in roles.items()}
        keys = [key for members in normalized.values() for key in members]
        if any(not members for members in normalized.values()) or len(set(keys)) != len(
            keys
        ):
            raise ValueError(
                "Qualification roles must have distinct nonempty key memberships."
            )
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("Role memberships require canonical key IDs.")
        self.store = store
        self.roles = normalized

    def verify(
        self,
        record: object,
        attestation: SignedQualificationRecord,
        /,
        *,
        role: str,
        at_time: int,
    ) -> str:
        if not isinstance(attestation, SignedQualificationRecord):
            raise TypeError(
                "An authenticated record, not an opaque evidence ID, is required."
            )
        if attestation.signature.algorithm not in (
            "Ed25519",
            "ECDSA_SHA_256",
            "RSASSA_PSS_SHA_256",
        ):
            raise ValueError(
                "Production qualification roles require asymmetric signatures."
            )
        if (
            type(at_time) is not int
            or not attestation.issued_at <= at_time < attestation.expires_at
        ):
            raise ValueError("Qualification attestation is stale or not yet active.")
        if (
            attestation.role != role
            or attestation.signature.purpose != f"qualification.{role}"
        ):
            raise ValueError("Qualification signature purpose or role is mismatched.")
        if attestation.signature.key_id not in self.roles.get(role, ()):
            raise ValueError("Qualification signer is not authorized for this role.")
        if (
            attestation.signature.signed_at != attestation.issued_at
            or attestation.content_json != canonical_json(_record(record))
        ):
            raise ValueError("Qualification content or signing time was tampered with.")
        self.store.verify(
            attestation.signed_payload, attestation.signature, at_time=at_time
        )
        return attestation.signature.key_id

    def expiry(self, attestation: SignedQualificationRecord, /) -> int:
        records = {record.key_id: record for record in self.store.records()}
        record = records.get(attestation.signature.key_id)
        if record is None:
            raise ValueError("Attestation signer has no trust record.")
        return min(attestation.expires_at, record.expires_at)

    @classmethod
    def from_public_config(
        cls, config: Mapping[str, object], /
    ) -> QualificationRoleTrust:
        """Load only public Ed25519 roots; KMS verifiers are injected explicitly."""
        from ..service._security import (
            Ed25519Verifier,
            SigningKeyTrustRecord,
            SigningTrustStore,
        )

        if (
            set(config) != {"kind", "keys", "roles"}
            or config["kind"] != "qualification-public-trust"
        ):
            raise ValueError("Public trust configuration has unknown or missing fields.")
        store = SigningTrustStore()
        pending = list(config["keys"])
        registered: set[str] = set()
        while pending:
            progressed = False
            for item in tuple(pending):
                if set(item) != {"record", "public_key_hex"}:
                    raise ValueError(
                        "Public trust entries contain verification keys only."
                    )
                record = SigningKeyTrustRecord(**item["record"])
                if (
                    record.supersedes_key_id is not None
                    and record.supersedes_key_id not in registered
                ):
                    continue
                if record.algorithm != "Ed25519":
                    raise ValueError("Offline roots require an Ed25519 public verifier.")
                store.trust(
                    record,
                    Ed25519Verifier(record.key_id, bytes.fromhex(item["public_key_hex"])),
                )
                registered.add(record.key_id)
                pending.remove(item)
                progressed = True
            if not progressed:
                raise ValueError("Public trust rotation lineage is missing or cyclic.")
        result = cls(store, config["roles"])
        if set().union(*result.roles.values()) != registered:
            raise ValueError("Every public root must have exactly one declared role.")
        return result


class AsymmetricReleaseSigner:
    """Adapt Ed25519/KMS purpose signing to the existing ReleaseIndex protocol."""

    def __init__(self, signer: AsymmetricSigner, /, *, issued_at: int):
        if signer.algorithm not in ("Ed25519", "ECDSA_SHA_256", "RSASSA_PSS_SHA_256"):
            raise ValueError(
                "Production release signing requires an asymmetric algorithm."
            )
        self.signer = signer
        self.issued_at = issued_at

    @property
    def signer_id(self) -> str:
        return self.signer.key_id

    @property
    def signature_algorithm(self) -> str:
        return self.signer.algorithm

    def sign(self, payload: bytes, /) -> bytes:
        return self.signer.sign(
            payload, purpose="qualification.release-index", signed_at=self.issued_at
        ).signature


class AsymmetricReleaseTrustPolicy:
    """Release trust with live signed typed proof revalidation for every gate."""

    def __init__(
        self,
        roles: QualificationRoleTrust,
        /,
        *,
        proofs: Sequence[object] = (),
        max_index_age: int,
    ):
        if type(max_index_age) is not int or max_index_age <= 0:
            raise ValueError("Index freshness must have a positive finite bound.")
        self.roles = roles
        self.max_index_age = max_index_age
        self.proofs = tuple(proofs)
        self._verification_chain = ContextVar("qualification-proof-chain", default=())

    def verify_index(self, index: ReleaseIndex, at_time: int, /) -> bool:
        from ..service._security import SignatureEnvelope

        try:
            ReleaseIndex.from_record(index.to_record())
            if any(
                not profile.released
                or not profile.required_gates
                or not profile.release_evidence
                or any(
                    not isinstance(dependency, SupportDependency)
                    for dependency in profile.dependencies
                )
                for profile in index.profiles
            ):
                return False
            if (
                type(at_time) is not int
                or not index.issued_at <= at_time < index.issued_at + self.max_index_age
            ):
                return False
            if index.signer_id not in self.roles.roles["release-authority"]:
                return False
            if index.signature_algorithm not in (
                "Ed25519",
                "ECDSA_SHA_256",
                "RSASSA_PSS_SHA_256",
            ):
                return False
            envelope = SignatureEnvelope(
                index.signer_id,
                index.signature_algorithm,
                "qualification.release-index",
                index.issued_at,
                hashlib.sha256(index.signed_payload).hexdigest(),
                bytes.fromhex(index.signature),
            )
            self.roles.store.verify(index.signed_payload, envelope, at_time=at_time)
        except (TypeError, ValueError, KeyError, RuntimeError):
            return False
        return True

    def accepts_evidence(self, evidence: ReleaseGateEvidence, at_time: int, /) -> bool:
        if not evidence.accepted or not evidence.is_current(at_time):
            return False
        matches = [
            proof for proof in self.proofs if proof.gate_id == evidence.evidence_id
        ]
        if len(matches) != 1:
            return False
        chain = self._verification_chain.get()
        if evidence.evidence_id in chain:
            return False
        token = self._verification_chain.set((*chain, evidence.evidence_id))
        try:
            # Proofs are typed release builders' retained inputs, not ID allowlists.
            matches[0].verify(self, at_time=at_time)
        except (TypeError, ValueError, KeyError, RuntimeError):
            return False
        finally:
            self._verification_chain.reset(token)
        return True
