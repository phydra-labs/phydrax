#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fail-closed cardiovascular qualification and commercial release records.

The records in this module describe technical release readiness.  They neither
issue a software licence nor make a medical, clinical, or regulated-device
claim.  Payload storage, execution lifecycle, and release-signing primitives
remain owned by :mod:`phydrax.artifacts`, :mod:`phydrax.lifecycle`, and
:mod:`phydrax.qualification`, respectively.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Protocol

from ..._fingerprint import canonical_fingerprint, canonical_json
from ...artifacts import ArtifactManifest
from ...lifecycle import RunRecord
from ...qualification import (
    CapabilityProfile,
    ReleaseGateEvidence,
    ReleaseSigner,
    ReleaseTrustPolicy,
    SupportTuple,
)
from ._case import CardiovascularCaseManifest
from ._execution import CardiovascularExecutionManifest


_MAX_TIMESTAMP = 2**63 - 1
_CARDIOVASCULAR_CAPABILITY = "cardiovascular.workflow"
_REQUIRED_SUPPORT_COORDINATES = {
    "data_classification": "non-phi",
    "deployment": "local",
    "regulated_device": False,
}
_REQUIRED_NON_CLAIMS = (
    "clinical-decision-support",
    "diagnosis",
    "regulated-medical-device",
    "treatment",
)


def _identifier(value: str, role: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{role} must be a string.")
    normalized = value.strip()
    if (
        not normalized
        or normalized != value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{role} must be non-empty canonical text.")
    return normalized


def _identifiers(
    values: Sequence[str], role: str, /, *, nonempty: bool = False
) -> tuple[str, ...]:
    normalized = tuple(sorted(_identifier(value, role) for value in values))
    if nonempty and not normalized:
        raise ValueError(f"{role} must be non-empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{role} must be unique.")
    return normalized


def _timestamp(value: int, role: str, /) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{role} must be an integer timestamp.")
    normalized = int(value)
    if normalized < 0 or normalized > _MAX_TIMESTAMP:
        raise ValueError(f"{role} must be a non-negative signed 64-bit timestamp.")
    return normalized


def _positive_integer(value: int, role: str, /) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{role} must be a positive integer.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{role} must be positive.")
    return normalized


def _signature_hex(value: str, role: str, /) -> str:
    signature = _identifier(value.lower(), role)
    if len(signature) % 2 or any(
        character not in "0123456789abcdef" for character in signature
    ):
        raise ValueError(f"{role} must be non-empty lowercase hex.")
    return signature


class CardiovascularSignatureVerifier(Protocol):
    """Public-key or trust-service verifier; verification never signs again."""

    @property
    def signer_id(self) -> str: ...

    @property
    def signature_algorithm(self) -> str: ...

    def verify(self, payload: bytes, signature: bytes, /) -> bool: ...


class CardiovascularReleaseGate(IntEnum):
    """The complete ordered cardiovascular technical release gate set."""

    G0_INTENDED_USE = 0
    G1_CODE_VERIFICATION = 1
    G2_SOLUTION_VERIFICATION = 2
    G3_VALIDATION_UQ = 3
    G4_DERIVATIVE_VALIDITY = 4
    G5_PROVENANCE_SUPPLY_CHAIN = 5
    G6_QUALITY_OPERATIONS = 6
    G7_INDEPENDENT_RELEASE_REVIEW = 7

    @property
    def gate_key(self) -> str:
        return self.name.replace("_", "-").lower()


class CardiovascularArtifactKind(StrEnum):
    """Required kinds of externally stored cardiovascular release artifacts."""

    SBOM = "sbom"
    BUILD_PROVENANCE = "build-provenance"
    COMMERCIAL_LICENSE = "commercial-license"
    NOTICE_AUDIT = "notice-audit"
    DATA_RIGHTS = "data-rights"
    SUPPLY_CHAIN_ATTESTATION = "supply-chain-attestation"


_REQUIRED_ARTIFACT_KINDS = tuple(CardiovascularArtifactKind)


class CardiovascularClaimStatus(IntEnum):
    """Technical disposition for one exact support tuple."""

    TECHNICAL_SUPPORT_CANDIDATE = 0
    NOT_SUPPORTED = 1
    PROHIBITED = 2


@dataclass(frozen=True, slots=True)
class CardiovascularClaimDecision:
    """One technical support decision, never a commercial or medical claim."""

    support_tuple: SupportTuple
    status: CardiovascularClaimStatus
    technical_scope: tuple[str, ...]
    excluded_uses: tuple[str, ...]
    rationale: str
    evidence_ids: tuple[str, ...] = ()
    decision_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.support_tuple, SupportTuple):
            raise TypeError("support_tuple must be SupportTuple.")
        status = CardiovascularClaimStatus(self.status)
        scope = _identifiers(self.technical_scope, "technical scope", nonempty=True)
        excluded = _identifiers(self.excluded_uses, "excluded use", nonempty=True)
        rationale = _identifier(self.rationale, "claim-decision rationale")
        evidence = _identifiers(self.evidence_ids, "claim evidence ID")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "technical_scope", scope)
        object.__setattr__(self, "excluded_uses", excluded)
        object.__setattr__(self, "rationale", rationale)
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(
            self,
            "decision_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-claim-decision",
                    "support_tuple": self.support_tuple.support_tuple_id,
                    "status": int(status),
                    "technical_scope": list(scope),
                    "excluded_uses": list(excluded),
                    "rationale": rationale,
                    "evidence_ids": list(evidence),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CardiovascularClaimsMatrix:
    """Exact-tuple technical claims matrix with no wildcard matching."""

    decisions: tuple[CardiovascularClaimDecision, ...]
    matrix_id: str = field(init=False)

    def __post_init__(self) -> None:
        decisions = tuple(self.decisions)
        if not decisions or any(
            not isinstance(decision, CardiovascularClaimDecision)
            for decision in decisions
        ):
            raise TypeError("Claims matrix requires typed, non-empty decisions.")
        support_ids = tuple(
            decision.support_tuple.support_tuple_id for decision in decisions
        )
        if len(set(support_ids)) != len(support_ids):
            raise ValueError(
                "Claims matrix requires exactly one decision per SupportTuple."
            )
        decisions = tuple(
            sorted(
                decisions, key=lambda decision: decision.support_tuple.support_tuple_id
            )
        )
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(
            self,
            "matrix_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-claims-matrix",
                    "decisions": [decision.decision_id for decision in decisions],
                }
            ),
        )

    def decision_for(self, support_tuple: SupportTuple, /) -> CardiovascularClaimDecision:
        if not isinstance(support_tuple, SupportTuple):
            raise TypeError("support_tuple must be SupportTuple.")
        for decision in self.decisions:
            if decision.support_tuple.support_tuple_id == support_tuple.support_tuple_id:
                return decision
        raise KeyError(
            f"No cardiovascular claim decision for {support_tuple.support_tuple_id}."
        )


@dataclass(frozen=True, slots=True)
class CardiovascularResourcePolicy:
    """Hard local resource envelope used by release qualification."""

    maximum_wall_time_seconds: int
    maximum_resident_bytes: int
    maximum_artifact_bytes: int
    maximum_concurrent_runs: int
    policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        wall = _positive_integer(
            self.maximum_wall_time_seconds, "maximum_wall_time_seconds"
        )
        resident = _positive_integer(
            self.maximum_resident_bytes, "maximum_resident_bytes"
        )
        artifact = _positive_integer(
            self.maximum_artifact_bytes, "maximum_artifact_bytes"
        )
        concurrent = _positive_integer(
            self.maximum_concurrent_runs, "maximum_concurrent_runs"
        )
        object.__setattr__(self, "maximum_wall_time_seconds", wall)
        object.__setattr__(self, "maximum_resident_bytes", resident)
        object.__setattr__(self, "maximum_artifact_bytes", artifact)
        object.__setattr__(self, "maximum_concurrent_runs", concurrent)
        object.__setattr__(
            self,
            "policy_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-resource-policy",
                    "maximum_wall_time_seconds": wall,
                    "maximum_resident_bytes": resident,
                    "maximum_artifact_bytes": artifact,
                    "maximum_concurrent_runs": concurrent,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CardiovascularPrivacyPolicy:
    """Privacy boundary for the only supplied commercial-support profile."""

    allowed_data_classifications: tuple[str, ...] = ("non-phi",)
    phi_allowed: bool = False
    external_transfer_allowed: bool = False
    telemetry_allowed: bool = False
    maximum_retention_days: int = 30
    policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        classes = _identifiers(
            self.allowed_data_classifications,
            "allowed data classification",
            nonempty=True,
        )
        retention = _positive_integer(
            self.maximum_retention_days, "maximum_retention_days"
        )
        object.__setattr__(self, "allowed_data_classifications", classes)
        object.__setattr__(self, "phi_allowed", bool(self.phi_allowed))
        object.__setattr__(
            self, "external_transfer_allowed", bool(self.external_transfer_allowed)
        )
        object.__setattr__(self, "telemetry_allowed", bool(self.telemetry_allowed))
        object.__setattr__(self, "maximum_retention_days", retention)
        object.__setattr__(
            self,
            "policy_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-privacy-policy",
                    "allowed_data_classifications": list(classes),
                    "phi_allowed": bool(self.phi_allowed),
                    "external_transfer_allowed": bool(self.external_transfer_allowed),
                    "telemetry_allowed": bool(self.telemetry_allowed),
                    "maximum_retention_days": retention,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CardiovascularSecurityPolicy:
    """Fail-closed security controls and reviewer allow-lists."""

    authorized_reviewer_ids: tuple[str, ...]
    trusted_signer_ids: tuple[str, ...]
    network_access_allowed: bool = False
    isolated_local_execution: bool = True
    dependency_lock_required: bool = True
    signed_evidence_required: bool = True
    policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        reviewers = _identifiers(
            self.authorized_reviewer_ids, "authorized reviewer ID", nonempty=True
        )
        signers = _identifiers(
            self.trusted_signer_ids,
            "trusted evidence signer ID",
            nonempty=True,
        )
        object.__setattr__(self, "authorized_reviewer_ids", reviewers)
        object.__setattr__(self, "trusted_signer_ids", signers)
        object.__setattr__(
            self, "network_access_allowed", bool(self.network_access_allowed)
        )
        object.__setattr__(
            self, "isolated_local_execution", bool(self.isolated_local_execution)
        )
        object.__setattr__(
            self, "dependency_lock_required", bool(self.dependency_lock_required)
        )
        object.__setattr__(
            self, "signed_evidence_required", bool(self.signed_evidence_required)
        )
        object.__setattr__(
            self,
            "policy_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-security-policy",
                    "authorized_reviewer_ids": list(reviewers),
                    "trusted_signer_ids": list(signers),
                    "network_access_allowed": bool(self.network_access_allowed),
                    "isolated_local_execution": bool(self.isolated_local_execution),
                    "dependency_lock_required": bool(self.dependency_lock_required),
                    "signed_evidence_required": bool(self.signed_evidence_required),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CardiovascularUsePolicy:
    """Explicit intended use and categorical regulated-use exclusions."""

    intended_use: str
    prohibited_uses: tuple[str, ...] = _REQUIRED_NON_CLAIMS
    regulated_device_use_allowed: bool = False
    clinical_decision_support_allowed: bool = False
    grants_commercial_license: bool = field(default=False, init=False)
    policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        intended = _identifier(self.intended_use, "intended use")
        prohibited = _identifiers(self.prohibited_uses, "prohibited use", nonempty=True)
        object.__setattr__(self, "intended_use", intended)
        object.__setattr__(self, "prohibited_uses", prohibited)
        object.__setattr__(
            self,
            "regulated_device_use_allowed",
            bool(self.regulated_device_use_allowed),
        )
        object.__setattr__(
            self,
            "clinical_decision_support_allowed",
            bool(self.clinical_decision_support_allowed),
        )
        object.__setattr__(
            self,
            "policy_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-use-policy",
                    "intended_use": intended,
                    "prohibited_uses": list(prohibited),
                    "regulated_device_use_allowed": bool(
                        self.regulated_device_use_allowed
                    ),
                    "clinical_decision_support_allowed": bool(
                        self.clinical_decision_support_allowed
                    ),
                    "grants_commercial_license": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CardiovascularReviewRoles:
    """Named, mutually independent qualification and release roles."""

    author_id: str
    technical_reviewer_id: str
    validation_reviewer_id: str
    security_reviewer_id: str
    release_approver_id: str
    roles_id: str = field(init=False)

    def __post_init__(self) -> None:
        values = tuple(
            _identifier(value, "review role ID")
            for value in (
                self.author_id,
                self.technical_reviewer_id,
                self.validation_reviewer_id,
                self.security_reviewer_id,
                self.release_approver_id,
            )
        )
        if len(set(values)) != len(values):
            raise ValueError("Cardiovascular qualification roles must be independent.")
        (
            author,
            technical,
            validation,
            security,
            approver,
        ) = values
        object.__setattr__(self, "author_id", author)
        object.__setattr__(self, "technical_reviewer_id", technical)
        object.__setattr__(self, "validation_reviewer_id", validation)
        object.__setattr__(self, "security_reviewer_id", security)
        object.__setattr__(self, "release_approver_id", approver)
        object.__setattr__(
            self,
            "roles_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-review-roles",
                    "author": author,
                    "technical_reviewer": technical,
                    "validation_reviewer": validation,
                    "security_reviewer": security,
                    "release_approver": approver,
                }
            ),
        )

    def expected_reviewer(self, gate: CardiovascularReleaseGate, /) -> str:
        gate_ = CardiovascularReleaseGate(gate)
        if gate_ in (
            CardiovascularReleaseGate.G0_INTENDED_USE,
            CardiovascularReleaseGate.G1_CODE_VERIFICATION,
            CardiovascularReleaseGate.G4_DERIVATIVE_VALIDITY,
        ):
            return self.technical_reviewer_id
        if gate_ in (
            CardiovascularReleaseGate.G2_SOLUTION_VERIFICATION,
            CardiovascularReleaseGate.G3_VALIDATION_UQ,
            CardiovascularReleaseGate.G7_INDEPENDENT_RELEASE_REVIEW,
        ):
            return self.validation_reviewer_id
        return self.security_reviewer_id


@dataclass(frozen=True, slots=True)
class CardiovascularSignedNonClaim:
    """Authenticated exclusion statement bound to one exact SupportTuple."""

    support_tuple_id: str
    excluded_use: str
    statement: str
    author_id: str
    issued_at: int
    expires_at: int
    signer_id: str
    signature_algorithm: str
    signature: str
    non_claim_id: str = field(init=False)

    def __post_init__(self) -> None:
        support = _identifier(self.support_tuple_id, "support tuple ID")
        excluded = _identifier(self.excluded_use, "excluded use")
        statement = _identifier(self.statement, "non-claim statement")
        author = _identifier(self.author_id, "non-claim author ID")
        issued = _timestamp(self.issued_at, "issued_at")
        expires = _timestamp(self.expires_at, "expires_at")
        signer = _identifier(self.signer_id, "non-claim signer ID")
        algorithm = _identifier(self.signature_algorithm, "non-claim signature algorithm")
        signature = _signature_hex(self.signature, "non-claim signature")
        object.__setattr__(self, "support_tuple_id", support)
        object.__setattr__(self, "excluded_use", excluded)
        object.__setattr__(self, "statement", statement)
        object.__setattr__(self, "author_id", author)
        object.__setattr__(self, "issued_at", issued)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "signer_id", signer)
        object.__setattr__(self, "signature_algorithm", algorithm)
        object.__setattr__(self, "signature", signature)
        object.__setattr__(
            self,
            "non_claim_id",
            canonical_fingerprint(
                {
                    **self._unsigned_record(),
                    "signature": signature,
                }
            ),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "kind": "cardiovascular-signed-non-claim",
            "support_tuple_id": self.support_tuple_id,
            "excluded_use": self.excluded_use,
            "statement": self.statement,
            "author_id": self.author_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "signer_id": self.signer_id,
            "signature_algorithm": self.signature_algorithm,
            "effect": "non-claim-only",
            "grants_commercial_license": False,
        }

    @property
    def signed_payload(self) -> bytes:
        return canonical_json(self._unsigned_record()).encode("utf-8")

    @classmethod
    def issue(
        cls,
        support_tuple: SupportTuple,
        excluded_use: str,
        statement: str,
        /,
        *,
        author_id: str,
        issued_at: int,
        expires_at: int,
        signer: ReleaseSigner,
    ) -> CardiovascularSignedNonClaim:
        if not isinstance(support_tuple, SupportTuple):
            raise TypeError("support_tuple must be SupportTuple.")
        unsigned = cls(
            support_tuple.support_tuple_id,
            excluded_use,
            statement,
            author_id,
            issued_at,
            expires_at,
            _identifier(signer.signer_id, "non-claim signer ID"),
            _identifier(signer.signature_algorithm, "non-claim signature algorithm"),
            "00",
        )
        signature = signer.sign(unsigned.signed_payload)
        if not isinstance(signature, bytes) or not signature:
            raise TypeError("Release signer must return non-empty bytes.")
        return cls(
            unsigned.support_tuple_id,
            unsigned.excluded_use,
            unsigned.statement,
            unsigned.author_id,
            unsigned.issued_at,
            unsigned.expires_at,
            unsigned.signer_id,
            unsigned.signature_algorithm,
            signature.hex(),
        )

    def is_current(self, at_time: int, /) -> bool:
        timestamp = _timestamp(at_time, "at_time")
        return self.issued_at <= timestamp <= self.expires_at

    def verify(self, verifier: CardiovascularSignatureVerifier, /) -> bool:
        if (
            verifier.signer_id != self.signer_id
            or verifier.signature_algorithm != self.signature_algorithm
        ):
            return False
        return bool(verifier.verify(self.signed_payload, bytes.fromhex(self.signature)))

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "signature": self.signature,
            "non_claim_id": self.non_claim_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> CardiovascularSignedNonClaim:
        if not isinstance(record, Mapping):
            raise TypeError("Signed non-claim record must be a mapping.")
        if (
            record["kind"] != "cardiovascular-signed-non-claim"
            or record["effect"] != "non-claim-only"
            or record["grants_commercial_license"] is not False
        ):
            raise ValueError("Signed non-claim record changes its non-claim effect.")
        value = cls(
            str(record["support_tuple_id"]),
            str(record["excluded_use"]),
            str(record["statement"]),
            str(record["author_id"]),
            int(record["issued_at"]),
            int(record["expires_at"]),
            str(record["signer_id"]),
            str(record["signature_algorithm"]),
            str(record["signature"]),
        )
        if str(record["non_claim_id"]) != value.non_claim_id:
            raise ValueError("Signed non-claim record has an invalid content address.")
        return value


@dataclass(frozen=True, slots=True)
class CardiovascularArtifactReference:
    """Freshness and dependency metadata around a generic artifact manifest."""

    kind: CardiovascularArtifactKind
    manifest: ArtifactManifest
    issued_at: int
    expires_at: int
    dependency_reference_ids: tuple[str, ...] = ()
    reference_id: str = field(init=False)

    def __post_init__(self) -> None:
        kind = CardiovascularArtifactKind(self.kind)
        if not isinstance(self.manifest, ArtifactManifest):
            raise TypeError("manifest must be ArtifactManifest.")
        issued = _timestamp(self.issued_at, "issued_at")
        expires = _timestamp(self.expires_at, "expires_at")
        dependencies = _identifiers(
            self.dependency_reference_ids, "artifact dependency reference ID"
        )
        if expires <= issued:
            raise ValueError("Artifact reference must expire after it is issued.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "issued_at", issued)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "dependency_reference_ids", dependencies)
        reference_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-artifact-reference",
                "artifact_kind": kind.value,
                "manifest": self.manifest.manifest_id,
                "issued_at": issued,
                "expires_at": expires,
                "dependencies": list(dependencies),
            }
        )
        if reference_id in dependencies:
            raise ValueError("Artifact reference cannot depend on itself.")
        object.__setattr__(self, "reference_id", reference_id)

    def is_current(self, at_time: int, /) -> bool:
        timestamp = _timestamp(at_time, "at_time")
        return self.issued_at <= timestamp <= self.expires_at


@dataclass(frozen=True, slots=True)
class CardiovascularArtifactSet:
    """A dependency-addressed set of references; incomplete sets remain auditable."""

    references: tuple[CardiovascularArtifactReference, ...] = ()
    artifact_set_id: str = field(init=False)

    def __post_init__(self) -> None:
        references = tuple(self.references)
        if any(
            not isinstance(reference, CardiovascularArtifactReference)
            for reference in references
        ):
            raise TypeError(
                "Artifact set entries must be CardiovascularArtifactReference."
            )
        kinds = tuple(reference.kind for reference in references)
        identifiers = tuple(reference.reference_id for reference in references)
        manifests = tuple(reference.manifest.manifest_id for reference in references)
        if len(set(kinds)) != len(kinds):
            raise ValueError(
                "Artifact set permits exactly one reference per artifact kind."
            )
        if len(set(identifiers)) != len(identifiers) or len(set(manifests)) != len(
            manifests
        ):
            raise ValueError("Artifact references and manifests must be unique.")
        references = tuple(sorted(references, key=lambda reference: reference.kind.value))
        object.__setattr__(self, "references", references)
        object.__setattr__(
            self,
            "artifact_set_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-artifact-set",
                    "references": [reference.reference_id for reference in references],
                }
            ),
        )

    def reference_for(
        self, kind: CardiovascularArtifactKind, /
    ) -> CardiovascularArtifactReference:
        kind_ = CardiovascularArtifactKind(kind)
        for reference in self.references:
            if reference.kind == kind_:
                return reference
        raise KeyError(f"No cardiovascular {kind_.value} artifact reference.")

    def blockers(
        self,
        required_kinds: Sequence[CardiovascularArtifactKind],
        at_time: int,
        /,
    ) -> tuple[str, ...]:
        timestamp = _timestamp(at_time, "at_time")
        required = tuple(CardiovascularArtifactKind(kind) for kind in required_kinds)
        present_by_kind = {reference.kind: reference for reference in self.references}
        present_by_id = {
            reference.reference_id: reference for reference in self.references
        }
        reasons: list[str] = []
        for kind in required:
            if kind not in present_by_kind:
                reasons.append(f"missing-artifact:{kind.value}")
            elif not present_by_kind[kind].is_current(timestamp):
                reasons.append(f"stale-artifact:{kind.value}")
        for reference in self.references:
            for dependency_id in reference.dependency_reference_ids:
                if dependency_id not in present_by_id:
                    reasons.append(
                        f"missing-artifact-dependency:{reference.kind.value}:{dependency_id}"
                    )
                elif not present_by_id[dependency_id].is_current(timestamp):
                    reasons.append(
                        f"stale-artifact-dependency:{reference.kind.value}:{dependency_id}"
                    )
        return tuple(reasons)


@dataclass(frozen=True, slots=True)
class CardiovascularGateEvidence:
    """Signed dossier record around canonical G0-G7 release evidence."""

    gate: CardiovascularReleaseGate
    release_evidence: ReleaseGateEvidence
    dossier_id: str
    signer_id: str
    signature_algorithm: str
    signature: str
    gate_evidence_id: str = field(init=False)

    def __post_init__(self) -> None:
        gate = CardiovascularReleaseGate(self.gate)
        if not isinstance(self.release_evidence, ReleaseGateEvidence):
            raise TypeError("release_evidence must be ReleaseGateEvidence.")
        if self.release_evidence.gate != gate.gate_key:
            raise ValueError("Release evidence gate does not match its G0-G7 record.")
        dossier = _identifier(self.dossier_id, "gate dossier ID")
        signer = _identifier(self.signer_id, "gate signer ID")
        algorithm = _identifier(self.signature_algorithm, "gate signature algorithm")
        signature = _signature_hex(self.signature, "gate signature")
        if signer != self.release_evidence.reviewer_id:
            raise ValueError("Each G0-G7 record must be signed by its named reviewer.")
        object.__setattr__(self, "gate", gate)
        object.__setattr__(self, "dossier_id", dossier)
        object.__setattr__(self, "signer_id", signer)
        object.__setattr__(self, "signature_algorithm", algorithm)
        object.__setattr__(self, "signature", signature)
        object.__setattr__(
            self,
            "gate_evidence_id",
            canonical_fingerprint({**self._unsigned_record(), "signature": signature}),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "kind": "cardiovascular-signed-gate-evidence",
            "gate": self.gate.gate_key,
            "dossier_id": self.dossier_id,
            "release_evidence": self.release_evidence.to_record(),
            "reviewer_id": self.release_evidence.reviewer_id,
            "evidence_ids": list(self.release_evidence.evidence_ids),
            "signer_id": self.signer_id,
            "signature_algorithm": self.signature_algorithm,
        }

    @property
    def signed_payload(self) -> bytes:
        return canonical_json(self._unsigned_record()).encode("utf-8")

    @classmethod
    def issue(
        cls,
        gate: CardiovascularReleaseGate,
        /,
        *,
        passed: bool,
        evidence_ids: Sequence[str],
        reviewer_id: str,
        dossier_id: str,
        issued_at: int,
        expires_at: int,
        signer: ReleaseSigner,
        deviation_ids: Sequence[str] = (),
    ) -> CardiovascularGateEvidence:
        gate_ = CardiovascularReleaseGate(gate)
        reviewer = _identifier(reviewer_id, "gate reviewer ID")
        if signer.signer_id != reviewer:
            raise ValueError("Gate signer must be the named reviewer.")
        evidence = ReleaseGateEvidence(
            gate_.gate_key,
            passed=passed,
            evidence_ids=evidence_ids,
            reviewer_id=reviewer,
            issued_at=issued_at,
            expires_at=expires_at,
            deviation_ids=deviation_ids,
        )
        unsigned = cls(
            gate_,
            evidence,
            dossier_id,
            signer.signer_id,
            signer.signature_algorithm,
            "00",
        )
        signature = signer.sign(unsigned.signed_payload)
        if not isinstance(signature, bytes) or not signature:
            raise TypeError("Release signer must return non-empty bytes.")
        return cls(
            gate_,
            evidence,
            unsigned.dossier_id,
            unsigned.signer_id,
            unsigned.signature_algorithm,
            signature.hex(),
        )

    def verify(self, verifier: CardiovascularSignatureVerifier, /) -> bool:
        if (
            verifier.signer_id != self.signer_id
            or verifier.signature_algorithm != self.signature_algorithm
        ):
            return False
        return bool(verifier.verify(self.signed_payload, bytes.fromhex(self.signature)))


@dataclass(frozen=True, slots=True)
class CardiovascularCommercialSupportProfile:
    """One exact local, non-PHI, non-regulated technical support profile."""

    name: str
    version: str
    support_tuple: SupportTuple
    claims_matrix: CardiovascularClaimsMatrix
    resource_policy: CardiovascularResourcePolicy
    privacy_policy: CardiovascularPrivacyPolicy
    security_policy: CardiovascularSecurityPolicy
    use_policy: CardiovascularUsePolicy
    dependency_profile_ids: tuple[str, ...] = ()
    required_artifact_kinds: tuple[CardiovascularArtifactKind, ...] = (
        _REQUIRED_ARTIFACT_KINDS
    )
    capability_profile: CapabilityProfile = field(init=False)
    profile_id: str = field(init=False)

    def __post_init__(self) -> None:
        name = _identifier(self.name, "commercial support profile name")
        version = _identifier(self.version, "commercial support profile version")
        if not isinstance(self.support_tuple, SupportTuple):
            raise TypeError("support_tuple must be SupportTuple.")
        if self.support_tuple.capability != _CARDIOVASCULAR_CAPABILITY:
            raise ValueError(
                f"Commercial profile capability must be {_CARDIOVASCULAR_CAPABILITY!r}."
            )
        if not isinstance(self.claims_matrix, CardiovascularClaimsMatrix):
            raise TypeError("claims_matrix must be CardiovascularClaimsMatrix.")
        if not isinstance(self.resource_policy, CardiovascularResourcePolicy):
            raise TypeError("resource_policy must be CardiovascularResourcePolicy.")
        if not isinstance(self.privacy_policy, CardiovascularPrivacyPolicy):
            raise TypeError("privacy_policy must be CardiovascularPrivacyPolicy.")
        if not isinstance(self.security_policy, CardiovascularSecurityPolicy):
            raise TypeError("security_policy must be CardiovascularSecurityPolicy.")
        if not isinstance(self.use_policy, CardiovascularUsePolicy):
            raise TypeError("use_policy must be CardiovascularUsePolicy.")
        coordinates = dict(self.support_tuple.attributes)
        if any(
            coordinates.get(coordinate) != expected
            for coordinate, expected in _REQUIRED_SUPPORT_COORDINATES.items()
        ):
            raise ValueError(
                "Commercial support is limited to the exact local, non-PHI, "
                "non-regulated SupportTuple coordinates."
            )
        decision = self.claims_matrix.decision_for(self.support_tuple)
        if decision.status != CardiovascularClaimStatus.TECHNICAL_SUPPORT_CANDIDATE:
            raise ValueError(
                "Commercial support profile requires a technical candidate decision."
            )
        if not set(_REQUIRED_NON_CLAIMS).issubset(decision.excluded_uses):
            raise ValueError("Claims decision omits a required medical-use exclusion.")
        if (
            self.privacy_policy.allowed_data_classifications != ("non-phi",)
            or self.privacy_policy.phi_allowed
            or self.privacy_policy.external_transfer_allowed
            or self.privacy_policy.telemetry_allowed
        ):
            raise ValueError("Commercial support profile must remain local and non-PHI.")
        if (
            self.security_policy.network_access_allowed
            or not self.security_policy.isolated_local_execution
            or not self.security_policy.dependency_lock_required
            or not self.security_policy.signed_evidence_required
        ):
            raise ValueError(
                "Commercial support profile requires fail-closed local security."
            )
        if (
            self.use_policy.regulated_device_use_allowed
            or self.use_policy.clinical_decision_support_allowed
            or not set(_REQUIRED_NON_CLAIMS).issubset(self.use_policy.prohibited_uses)
        ):
            raise ValueError(
                "Regulated, diagnostic, treatment, and clinical uses are excluded."
            )
        dependencies = _identifiers(self.dependency_profile_ids, "dependency profile ID")
        artifact_kinds = tuple(
            CardiovascularArtifactKind(kind) for kind in self.required_artifact_kinds
        )
        if set(artifact_kinds) != set(_REQUIRED_ARTIFACT_KINDS) or len(
            artifact_kinds
        ) != len(_REQUIRED_ARTIFACT_KINDS):
            raise ValueError(
                "Commercial profile cannot weaken release artifact prerequisites."
            )
        artifact_kinds = tuple(sorted(artifact_kinds, key=lambda kind: kind.value))
        generic = CapabilityProfile(
            name,
            "phydrax",
            version,
            (self.support_tuple,),
            dependencies=dependencies,
            required_gates=tuple(gate.gate_key for gate in CardiovascularReleaseGate),
            released=False,
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "dependency_profile_ids", dependencies)
        object.__setattr__(self, "required_artifact_kinds", artifact_kinds)
        object.__setattr__(self, "capability_profile", generic)
        object.__setattr__(
            self,
            "profile_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-commercial-support-profile",
                    "capability_profile": generic.profile_id,
                    "claims_matrix": self.claims_matrix.matrix_id,
                    "resource_policy": self.resource_policy.policy_id,
                    "privacy_policy": self.privacy_policy.policy_id,
                    "security_policy": self.security_policy.policy_id,
                    "use_policy": self.use_policy.policy_id,
                    "required_artifact_kinds": [kind.value for kind in artifact_kinds],
                    "grants_commercial_license": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CardiovascularQualificationBundle:
    """Candidate evidence without a release decision."""

    support_tuple: SupportTuple
    gates: tuple[CardiovascularGateEvidence, ...]
    artifacts: CardiovascularArtifactSet
    lifecycle_records: tuple[RunRecord, ...]
    non_claims: tuple[CardiovascularSignedNonClaim, ...]
    roles: CardiovascularReviewRoles
    case_manifest: CardiovascularCaseManifest
    execution_manifests: tuple[CardiovascularExecutionManifest, ...]
    bundle_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.support_tuple, SupportTuple):
            raise TypeError("support_tuple must be SupportTuple.")
        gates = tuple(self.gates)
        if any(not isinstance(gate, CardiovascularGateEvidence) for gate in gates):
            raise TypeError("gates must contain CardiovascularGateEvidence records.")
        gate_keys = tuple(gate.gate for gate in gates)
        if len(set(gate_keys)) != len(gate_keys):
            raise ValueError("Qualification bundle contains duplicate G0-G7 gates.")
        dossier_ids = {gate.dossier_id for gate in gates}
        if len(dossier_ids) > 1:
            raise ValueError("All G0-G7 records must bind one exact dossier ID.")
        if not isinstance(self.artifacts, CardiovascularArtifactSet):
            raise TypeError("artifacts must be CardiovascularArtifactSet.")
        lifecycle_records = tuple(self.lifecycle_records)
        if any(not isinstance(record, RunRecord) for record in lifecycle_records):
            raise TypeError("lifecycle_records must contain RunRecord values.")
        record_ids = tuple(record.record_id for record in lifecycle_records)
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("Lifecycle record IDs must be unique.")
        non_claims = tuple(self.non_claims)
        if any(
            not isinstance(record, CardiovascularSignedNonClaim) for record in non_claims
        ):
            raise TypeError("non_claims must contain signed non-claim records.")
        exclusions = tuple(record.excluded_use for record in non_claims)
        if len(set(exclusions)) != len(exclusions):
            raise ValueError(
                "Qualification bundle permits one signed record per exclusion."
            )
        if not isinstance(self.roles, CardiovascularReviewRoles):
            raise TypeError("roles must be CardiovascularReviewRoles.")
        if not isinstance(self.case_manifest, CardiovascularCaseManifest):
            raise TypeError("case_manifest must be CardiovascularCaseManifest.")
        execution_manifests = tuple(self.execution_manifests)
        if any(
            not isinstance(manifest, CardiovascularExecutionManifest)
            for manifest in execution_manifests
        ):
            raise TypeError(
                "execution_manifests must contain CardiovascularExecutionManifest values."
            )
        execution_ids = tuple(manifest.manifest_id for manifest in execution_manifests)
        if len(set(execution_ids)) != len(execution_ids):
            raise ValueError("Cardiovascular execution manifest IDs must be unique.")
        if any(
            manifest.case_manifest_id != self.case_manifest.manifest_id
            for manifest in execution_manifests
        ):
            raise ValueError(
                "Every cardiovascular execution manifest must bind the bundle case."
            )
        gates = tuple(sorted(gates, key=lambda gate: int(gate.gate)))
        non_claims = tuple(sorted(non_claims, key=lambda record: record.excluded_use))
        object.__setattr__(self, "gates", gates)
        object.__setattr__(self, "lifecycle_records", lifecycle_records)
        object.__setattr__(self, "non_claims", non_claims)
        object.__setattr__(self, "execution_manifests", execution_manifests)
        object.__setattr__(
            self,
            "bundle_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-qualification-bundle",
                    "support_tuple": self.support_tuple.support_tuple_id,
                    "gates": [gate.gate_evidence_id for gate in gates],
                    "artifacts": self.artifacts.artifact_set_id,
                    "lifecycle_records": list(record_ids),
                    "non_claims": [record.non_claim_id for record in non_claims],
                    "roles": self.roles.roles_id,
                    "case_manifest": self.case_manifest.manifest_id,
                    "execution_manifests": list(execution_ids),
                }
            ),
        )


def _candidate_content_record(
    support_profile_id: str,
    bundle_id: str,
    release_approver_id: str,
    evaluated_at: int,
    valid_through: int,
    capability_profile: CapabilityProfile,
    trust_evidence: Sequence[ReleaseGateEvidence],
    qualified: bool,
    blockers: Sequence[str],
    /,
) -> dict[str, object]:
    return {
        "kind": "cardiovascular-release-candidate",
        "support_profile": support_profile_id,
        "bundle": bundle_id,
        "release_approver": release_approver_id,
        "evaluated_at": evaluated_at,
        "valid_through": valid_through,
        "capability_profile": capability_profile.profile_id,
        "trust_evidence": [evidence.evidence_id for evidence in trust_evidence],
        "qualified": qualified,
        "blockers": list(blockers),
        "release_decision": None,
    }


@dataclass(frozen=True, slots=True)
class CardiovascularReleaseCandidate:
    """Qualification result before independent release authorization."""

    support_profile_id: str
    bundle_id: str
    release_approver_id: str
    evaluated_at: int
    valid_through: int
    capability_profile: CapabilityProfile
    trust_evidence: tuple[ReleaseGateEvidence, ...]
    qualified: bool
    blockers: tuple[str, ...]
    candidate_id: str

    def __post_init__(self) -> None:
        support_profile = _identifier(
            self.support_profile_id, "commercial support profile ID"
        )
        bundle = _identifier(self.bundle_id, "qualification bundle ID")
        approver = _identifier(self.release_approver_id, "release approver ID")
        evaluated = _timestamp(self.evaluated_at, "evaluated_at")
        valid_through = _timestamp(self.valid_through, "valid_through")
        if not isinstance(self.capability_profile, CapabilityProfile):
            raise TypeError("capability_profile must be CapabilityProfile.")
        if self.capability_profile.released:
            raise ValueError("A release candidate cannot contain a released profile.")
        evidence = tuple(self.trust_evidence)
        if any(not isinstance(item, ReleaseGateEvidence) for item in evidence):
            raise TypeError("trust_evidence must contain ReleaseGateEvidence values.")
        evidence_ids = tuple(item.evidence_id for item in evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("Candidate trust evidence must be unique.")
        blockers = tuple(_identifier(item, "candidate blocker") for item in self.blockers)
        if len(set(blockers)) != len(blockers):
            raise ValueError("Candidate blockers must be unique.")
        qualified = bool(self.qualified)
        if qualified != (not blockers):
            raise ValueError("Candidate qualification must agree with its blockers.")
        if qualified and valid_through < evaluated:
            raise ValueError("A qualified candidate cannot already be expired.")
        content = _candidate_content_record(
            support_profile,
            bundle,
            approver,
            evaluated,
            valid_through,
            self.capability_profile,
            evidence,
            qualified,
            blockers,
        )
        if _identifier(
            self.candidate_id, "release candidate ID"
        ) != canonical_fingerprint(content):
            raise ValueError("Release candidate has an invalid content address.")
        object.__setattr__(self, "support_profile_id", support_profile)
        object.__setattr__(self, "bundle_id", bundle)
        object.__setattr__(self, "release_approver_id", approver)
        object.__setattr__(self, "evaluated_at", evaluated)
        object.__setattr__(self, "valid_through", valid_through)
        object.__setattr__(self, "trust_evidence", evidence)
        object.__setattr__(self, "qualified", qualified)
        object.__setattr__(self, "blockers", blockers)

    @property
    def commercial_ready(self) -> bool:
        """A candidate alone is never an authorization to release."""
        return False


@dataclass(frozen=True, slots=True)
class CardiovascularReleaseDecision:
    """Approver-authenticated authorization separate from qualification."""

    candidate_id: str
    approved: bool
    approver_id: str
    decided_at: int
    rationale: str
    signature_algorithm: str
    signature: str
    decision_id: str = field(init=False)

    def __post_init__(self) -> None:
        candidate = _identifier(self.candidate_id, "release candidate ID")
        approver = _identifier(self.approver_id, "release approver ID")
        decided = _timestamp(self.decided_at, "decided_at")
        rationale = _identifier(self.rationale, "release-decision rationale")
        algorithm = _identifier(
            self.signature_algorithm, "release-decision signature algorithm"
        )
        signature = _signature_hex(self.signature, "release-decision signature")
        object.__setattr__(self, "candidate_id", candidate)
        object.__setattr__(self, "approved", bool(self.approved))
        object.__setattr__(self, "approver_id", approver)
        object.__setattr__(self, "decided_at", decided)
        object.__setattr__(self, "rationale", rationale)
        object.__setattr__(self, "signature_algorithm", algorithm)
        object.__setattr__(self, "signature", signature)
        object.__setattr__(
            self,
            "decision_id",
            canonical_fingerprint({**self._unsigned_record(), "signature": signature}),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "kind": "cardiovascular-signed-release-decision",
            "candidate": self.candidate_id,
            "approved": self.approved,
            "approver": self.approver_id,
            "decided_at": self.decided_at,
            "rationale": self.rationale,
            "signer_id": self.approver_id,
            "signature_algorithm": self.signature_algorithm,
            "grants_commercial_license": False,
            "regulated_device_claim": False,
        }

    @property
    def signed_payload(self) -> bytes:
        return canonical_json(self._unsigned_record()).encode("utf-8")

    def verify(self, verifier: CardiovascularSignatureVerifier, /) -> bool:
        if (
            verifier.signer_id != self.approver_id
            or verifier.signature_algorithm != self.signature_algorithm
        ):
            return False
        return bool(verifier.verify(self.signed_payload, bytes.fromhex(self.signature)))


def _assessment_content_record(
    candidate_id: str,
    decision_id: str,
    capability_profile: CapabilityProfile,
    commercial_ready: bool,
    blockers: Sequence[str],
    /,
) -> dict[str, object]:
    return {
        "kind": "cardiovascular-release-assessment",
        "candidate": candidate_id,
        "decision": decision_id,
        "capability_profile": capability_profile.profile_id,
        "commercial_ready": commercial_ready,
        "blockers": list(blockers),
        "grants_commercial_license": False,
        "regulated_device_claim": False,
    }


@dataclass(frozen=True, slots=True)
class CardiovascularReleaseAssessment:
    """Final fail-closed result after qualification and release authorization."""

    candidate_id: str
    decision_id: str
    capability_profile: CapabilityProfile
    commercial_ready: bool
    blockers: tuple[str, ...]
    assessment_id: str

    def __post_init__(self) -> None:
        candidate = _identifier(self.candidate_id, "release candidate ID")
        decision = _identifier(self.decision_id, "release decision ID")
        if not isinstance(self.capability_profile, CapabilityProfile):
            raise TypeError("capability_profile must be CapabilityProfile.")
        blockers = tuple(
            _identifier(item, "assessment blocker") for item in self.blockers
        )
        if len(set(blockers)) != len(blockers):
            raise ValueError("Assessment blockers must be unique.")
        ready = bool(self.commercial_ready)
        if ready != (not blockers) or self.capability_profile.released != ready:
            raise ValueError("Assessment readiness, blockers, and profile disagree.")
        content = _assessment_content_record(
            candidate,
            decision,
            self.capability_profile,
            ready,
            blockers,
        )
        if _identifier(
            self.assessment_id, "release assessment ID"
        ) != canonical_fingerprint(content):
            raise ValueError("Release assessment has an invalid content address.")
        object.__setattr__(self, "candidate_id", candidate)
        object.__setattr__(self, "decision_id", decision)
        object.__setattr__(self, "commercial_ready", ready)
        object.__setattr__(self, "blockers", blockers)

    def require_commercial_ready(self) -> str:
        if not self.commercial_ready:
            raise ValueError(
                "Cardiovascular profile is not commercial-ready: "
                + "; ".join(self.blockers)
            )
        return self.assessment_id


def _known_evidence_ids(
    profile: CardiovascularCommercialSupportProfile,
    bundle: CardiovascularQualificationBundle,
    /,
) -> set[str]:
    return {
        profile.claims_matrix.matrix_id,
        profile.resource_policy.policy_id,
        profile.privacy_policy.policy_id,
        profile.security_policy.policy_id,
        profile.use_policy.policy_id,
        bundle.roles.roles_id,
        bundle.case_manifest.manifest_id,
        *(reference.reference_id for reference in bundle.artifacts.references),
        *(record.record_id for record in bundle.lifecycle_records),
        *(manifest.manifest_id for manifest in bundle.execution_manifests),
        *(record.non_claim_id for record in bundle.non_claims),
    }


def _dependency_profile_blockers(
    dependency_profile_ids: Sequence[str],
    dependency_profiles: Sequence[CapabilityProfile],
    trust_policy: ReleaseTrustPolicy,
    at_time: int,
    /,
) -> tuple[str, ...]:
    profiles = tuple(dependency_profiles)
    if any(not isinstance(profile, CapabilityProfile) for profile in profiles):
        raise TypeError("dependency_profiles must contain CapabilityProfile values.")
    by_id = {profile.profile_id: profile for profile in profiles}
    if len(by_id) != len(profiles):
        raise ValueError("Dependency capability profiles must be unique.")
    reasons: list[str] = []

    def visit(profile_id: str, ancestry: tuple[str, ...]) -> None:
        if profile_id not in by_id:
            reasons.append(f"missing-dependency-profile:{profile_id}")
            return
        profile = by_id[profile_id]
        if not profile.released:
            reasons.append(f"unreleased-dependency-profile:{profile_id}")
        evidence_by_gate = {
            evidence.gate: evidence for evidence in profile.release_evidence
        }
        for gate in profile.required_gates:
            if gate not in evidence_by_gate:
                reasons.append(f"dependency-missing-evidence:{profile_id}:{gate}")
            elif not trust_policy.accepts_evidence(evidence_by_gate[gate], at_time):
                reasons.append(f"dependency-stale-or-rejected:{profile_id}:{gate}")
        for child_id in profile.dependencies:
            if child_id in ancestry:
                reasons.append(f"dependency-cycle:{child_id}")
            else:
                visit(child_id, ancestry + (child_id,))

    for dependency_id in dependency_profile_ids:
        visit(dependency_id, (dependency_id,))
    return tuple(dict.fromkeys(reasons))


def evaluate_cardiovascular_release_candidate(
    profile: CardiovascularCommercialSupportProfile,
    bundle: CardiovascularQualificationBundle,
    trust_policy: ReleaseTrustPolicy,
    signature_verifiers: Mapping[str, CardiovascularSignatureVerifier],
    /,
    *,
    at_time: int,
    dependency_profiles: Sequence[CapabilityProfile] = (),
) -> CardiovascularReleaseCandidate:
    """Evaluate all G0-G7 clauses without making a release decision."""
    if not isinstance(profile, CardiovascularCommercialSupportProfile):
        raise TypeError("profile must be CardiovascularCommercialSupportProfile.")
    if not isinstance(bundle, CardiovascularQualificationBundle):
        raise TypeError("bundle must be CardiovascularQualificationBundle.")
    if not isinstance(signature_verifiers, Mapping):
        raise TypeError("signature_verifiers must be a verifier mapping.")
    timestamp = _timestamp(at_time, "at_time")
    reasons: list[str] = []
    if bundle.roles.release_approver_id not in profile.security_policy.trusted_signer_ids:
        reasons.append("untrusted-release-approver")
    if bundle.support_tuple.support_tuple_id != profile.support_tuple.support_tuple_id:
        reasons.append(f"support-tuple-mismatch:{bundle.support_tuple.support_tuple_id}")
    if bundle.case_manifest.support_profile_id != profile.profile_id:
        reasons.append(
            f"case-support-profile-mismatch:{bundle.case_manifest.support_profile_id}"
        )
    metadata = bundle.case_manifest.metadata_mapping
    if metadata.get("data_classification") != "non-phi":
        reasons.append("case-data-classification:not-explicitly-non-phi")
    artifact_by_kind = {
        reference.kind: reference for reference in bundle.artifacts.references
    }
    if (
        CardiovascularArtifactKind.BUILD_PROVENANCE in artifact_by_kind
        and bundle.case_manifest.build_id
        != artifact_by_kind[
            CardiovascularArtifactKind.BUILD_PROVENANCE
        ].manifest.artifact_id
    ):
        reasons.append("case-build-artifact-mismatch")
    if (
        CardiovascularArtifactKind.SBOM in artifact_by_kind
        and bundle.case_manifest.sbom_id
        != artifact_by_kind[CardiovascularArtifactKind.SBOM].manifest.artifact_id
    ):
        reasons.append("case-sbom-artifact-mismatch")
    if CardiovascularArtifactKind.COMMERCIAL_LICENSE in artifact_by_kind and (
        artifact_by_kind[
            CardiovascularArtifactKind.COMMERCIAL_LICENSE
        ].manifest.artifact_id
        not in bundle.case_manifest.license_ids
    ):
        reasons.append("case-commercial-license-artifact-missing")
    if CardiovascularArtifactKind.DATA_RIGHTS in artifact_by_kind and (
        artifact_by_kind[CardiovascularArtifactKind.DATA_RIGHTS].manifest.artifact_id
        not in bundle.case_manifest.data_rights_ids
    ):
        reasons.append("case-data-rights-artifact-missing")
    execution_by_id = {
        manifest.manifest_id: manifest for manifest in bundle.execution_manifests
    }
    if not execution_by_id:
        reasons.append("missing-cardiovascular-execution-manifest")
    completed_execution_pairs: set[tuple[str, str]] = set()
    for record in bundle.lifecycle_records:
        if record.execution_plan_id not in execution_by_id:
            reasons.append(f"lifecycle-execution-unbound:{record.record_id}")
            continue
        execution = execution_by_id[record.execution_plan_id]
        if record.analysis_plan_id != execution.analysis_plan_id:
            reasons.append(f"lifecycle-analysis-plan-mismatch:{record.record_id}")
        if record.numeric_revision_id != execution.numeric_revision_id:
            reasons.append(f"lifecycle-numeric-revision-mismatch:{record.record_id}")
        if (
            record.status == "completed"
            and record.analysis_plan_id == execution.analysis_plan_id
            and record.numeric_revision_id == execution.numeric_revision_id
        ):
            completed_execution_pairs.add((record.record_id, execution.manifest_id))

    reasons.extend(bundle.artifacts.blockers(profile.required_artifact_kinds, timestamp))
    reasons.extend(
        _dependency_profile_blockers(
            profile.dependency_profile_ids,
            dependency_profiles,
            trust_policy,
            timestamp,
        )
    )
    gates_by_gate = {gate.gate: gate for gate in bundle.gates}
    known_ids = _known_evidence_ids(profile, bundle)
    for gate in CardiovascularReleaseGate:
        gate_record = gates_by_gate.get(gate)
        if gate_record is None:
            reasons.append(f"missing-gate:{gate.gate_key}")
            continue
        evidence = gate_record.release_evidence
        if evidence.reviewer_id != bundle.roles.expected_reviewer(gate):
            reasons.append(f"independent-reviewer-mismatch:{gate.gate_key}")
        if evidence.reviewer_id not in profile.security_policy.authorized_reviewer_ids:
            reasons.append(f"unauthorized-reviewer:{gate.gate_key}")
        if gate_record.signer_id not in profile.security_policy.trusted_signer_ids:
            reasons.append(f"untrusted-gate-signer:{gate.gate_key}")
        elif gate_record.signer_id not in signature_verifiers:
            reasons.append(f"missing-gate-verifier:{gate.gate_key}")
        elif not gate_record.verify(signature_verifiers[gate_record.signer_id]):
            reasons.append(f"invalid-gate-signature:{gate.gate_key}")
        if not evidence.passed:
            reasons.append(f"failed-gate:{gate.gate_key}")
        elif evidence.deviation_ids:
            reasons.append(f"unapproved-deviation:{gate.gate_key}")
        elif not evidence.is_current(timestamp):
            reasons.append(f"stale-gate:{gate.gate_key}")
        elif not trust_policy.accepts_evidence(evidence, timestamp):
            reasons.append(f"untrusted-gate-evidence:{gate.gate_key}")
        for evidence_id in evidence.evidence_ids:
            if evidence_id not in known_ids:
                reasons.append(
                    f"unresolved-evidence-reference:{gate.gate_key}:{evidence_id}"
                )
        if gate in (
            CardiovascularReleaseGate.G1_CODE_VERIFICATION,
            CardiovascularReleaseGate.G2_SOLUTION_VERIFICATION,
            CardiovascularReleaseGate.G3_VALIDATION_UQ,
            CardiovascularReleaseGate.G4_DERIVATIVE_VALIDITY,
        ) and not any(
            run_id in evidence.evidence_ids and execution_id in evidence.evidence_ids
            for run_id, execution_id in completed_execution_pairs
        ):
            reasons.append(f"missing-exact-run-execution-evidence:{gate.gate_key}")

    g0 = gates_by_gate.get(CardiovascularReleaseGate.G0_INTENDED_USE)
    if g0 is not None:
        required_g0_ids = {
            profile.claims_matrix.matrix_id,
            profile.use_policy.policy_id,
            bundle.case_manifest.manifest_id,
            *(record.non_claim_id for record in bundle.non_claims),
        }
        for missing_id in sorted(
            required_g0_ids.difference(g0.release_evidence.evidence_ids)
        ):
            reasons.append(f"g0-missing-record:{missing_id}")

    g5 = gates_by_gate.get(CardiovascularReleaseGate.G5_PROVENANCE_SUPPLY_CHAIN)
    if g5 is not None:
        required_g5_ids = {
            reference.reference_id
            for reference in bundle.artifacts.references
            if reference.kind in profile.required_artifact_kinds
        }
        for missing_id in sorted(
            required_g5_ids.difference(g5.release_evidence.evidence_ids)
        ):
            reasons.append(f"g5-missing-artifact-reference:{missing_id}")

    g6 = gates_by_gate.get(CardiovascularReleaseGate.G6_QUALITY_OPERATIONS)
    if g6 is not None:
        required_g6_ids = {
            profile.resource_policy.policy_id,
            profile.privacy_policy.policy_id,
            profile.security_policy.policy_id,
        }
        for missing_id in sorted(
            required_g6_ids.difference(g6.release_evidence.evidence_ids)
        ):
            reasons.append(f"g6-missing-policy:{missing_id}")

    g7 = gates_by_gate.get(CardiovascularReleaseGate.G7_INDEPENDENT_RELEASE_REVIEW)
    if g7 is not None and bundle.roles.roles_id not in g7.release_evidence.evidence_ids:
        reasons.append(f"g7-missing-review-roles:{bundle.roles.roles_id}")

    non_claims_by_use = {record.excluded_use: record for record in bundle.non_claims}
    for excluded_use in _REQUIRED_NON_CLAIMS:
        if excluded_use not in non_claims_by_use:
            reasons.append(f"missing-signed-non-claim:{excluded_use}")
    for record in bundle.non_claims:
        if record.support_tuple_id != profile.support_tuple.support_tuple_id:
            reasons.append(f"non-claim-support-mismatch:{record.excluded_use}")
        if record.author_id != bundle.roles.author_id:
            reasons.append(f"non-claim-author-mismatch:{record.excluded_use}")
        if record.signer_id in (
            bundle.roles.author_id,
            bundle.roles.release_approver_id,
        ):
            reasons.append(f"non-claim-signer-not-independent:{record.excluded_use}")
        if record.signer_id not in profile.security_policy.trusted_signer_ids:
            reasons.append(f"untrusted-non-claim-signer:{record.excluded_use}")
        elif record.signer_id not in signature_verifiers:
            reasons.append(f"missing-non-claim-verifier:{record.excluded_use}")
        elif not record.verify(signature_verifiers[record.signer_id]):
            reasons.append(f"invalid-non-claim-signature:{record.excluded_use}")
        if not record.is_current(timestamp):
            reasons.append(f"stale-non-claim:{record.excluded_use}")
    validity_deadlines = (
        *(gate.release_evidence.expires_at for gate in bundle.gates),
        *(reference.expires_at for reference in bundle.artifacts.references),
        *(record.expires_at for record in bundle.non_claims),
        *(
            dependency_evidence.expires_at
            for dependency_profile in dependency_profiles
            for dependency_evidence in dependency_profile.release_evidence
        ),
    )
    valid_through = min(validity_deadlines) if validity_deadlines else timestamp

    blockers = tuple(dict.fromkeys(reasons))
    evidence = tuple(
        gate.release_evidence
        for gate in bundle.gates
        if gate.gate.gate_key in profile.capability_profile.required_gates
    )
    candidate_profile = CapabilityProfile(
        profile.capability_profile.name,
        profile.capability_profile.provider,
        profile.capability_profile.version,
        (profile.support_tuple,),
        dependencies=profile.dependency_profile_ids,
        required_gates=profile.capability_profile.required_gates,
        release_evidence=evidence,
        released=False,
    )
    trust_evidence_by_id = {
        item.evidence_id: item
        for item in (
            *evidence,
            *(
                dependency_evidence
                for dependency_profile in dependency_profiles
                for dependency_evidence in dependency_profile.release_evidence
            ),
        )
    }
    trust_evidence = tuple(
        trust_evidence_by_id[evidence_id] for evidence_id in sorted(trust_evidence_by_id)
    )
    candidate_id = canonical_fingerprint(
        _candidate_content_record(
            profile.profile_id,
            bundle.bundle_id,
            bundle.roles.release_approver_id,
            timestamp,
            valid_through,
            candidate_profile,
            trust_evidence,
            not blockers,
            blockers,
        )
    )
    return CardiovascularReleaseCandidate(
        profile.profile_id,
        bundle.bundle_id,
        bundle.roles.release_approver_id,
        timestamp,
        valid_through,
        candidate_profile,
        trust_evidence,
        not blockers,
        blockers,
        candidate_id,
    )


def make_cardiovascular_release_decision(
    candidate: CardiovascularReleaseCandidate,
    roles: CardiovascularReviewRoles,
    /,
    *,
    approved: bool,
    signer: ReleaseSigner,
    decided_at: int,
    rationale: str,
) -> CardiovascularReleaseDecision:
    """Create a separate decision; approval is impossible for a blocked candidate."""
    if not isinstance(candidate, CardiovascularReleaseCandidate):
        raise TypeError("candidate must be CardiovascularReleaseCandidate.")
    if not isinstance(roles, CardiovascularReviewRoles):
        raise TypeError("roles must be CardiovascularReviewRoles.")
    approver = _identifier(signer.signer_id, "release approver ID")
    decision_time = _timestamp(decided_at, "decided_at")
    if approver != roles.release_approver_id or approver != candidate.release_approver_id:
        raise ValueError(
            "Only the independent named release approver may decide release."
        )
    if bool(approved) and not candidate.qualified:
        raise ValueError(
            "A blocked cardiovascular candidate cannot receive release approval: "
            + "; ".join(candidate.blockers)
        )
    if bool(approved) and not (
        candidate.evaluated_at <= decision_time <= candidate.valid_through
    ):
        raise ValueError(
            "Release approval requires a current candidate evaluation and evidence."
        )
    unsigned = CardiovascularReleaseDecision(
        candidate.candidate_id,
        bool(approved),
        approver,
        decision_time,
        rationale,
        signer.signature_algorithm,
        "00",
    )
    signature = signer.sign(unsigned.signed_payload)
    if not isinstance(signature, bytes) or not signature:
        raise TypeError("Release signer must return non-empty bytes.")
    return CardiovascularReleaseDecision(
        unsigned.candidate_id,
        unsigned.approved,
        unsigned.approver_id,
        unsigned.decided_at,
        unsigned.rationale,
        unsigned.signature_algorithm,
        signature.hex(),
    )


def assess_cardiovascular_release(
    candidate: CardiovascularReleaseCandidate,
    decision: CardiovascularReleaseDecision,
    trust_policy: ReleaseTrustPolicy,
    signature_verifiers: Mapping[str, CardiovascularSignatureVerifier],
    /,
) -> CardiovascularReleaseAssessment:
    """Apply the independent decision without treating it as a licence grant."""
    if not isinstance(candidate, CardiovascularReleaseCandidate):
        raise TypeError("candidate must be CardiovascularReleaseCandidate.")
    if not isinstance(decision, CardiovascularReleaseDecision):
        raise TypeError("decision must be CardiovascularReleaseDecision.")
    if not isinstance(signature_verifiers, Mapping):
        raise TypeError("signature_verifiers must be a verifier mapping.")
    reasons = list(candidate.blockers)
    for evidence in candidate.trust_evidence:
        if not trust_policy.accepts_evidence(evidence, decision.decided_at):
            reasons.append(
                f"release-time-rejected-evidence:{evidence.gate}:{evidence.evidence_id}"
            )
    if decision.candidate_id != candidate.candidate_id:
        reasons.append(f"release-decision-candidate-mismatch:{decision.candidate_id}")
    if decision.approver_id != candidate.release_approver_id:
        reasons.append(f"release-decision-approver-mismatch:{decision.approver_id}")
    if decision.approver_id not in signature_verifiers:
        reasons.append("release-decision-verifier:missing")
    elif not decision.verify(signature_verifiers[decision.approver_id]):
        reasons.append("release-decision-signature:invalid")
    if decision.decided_at < candidate.evaluated_at:
        reasons.append("release-decision-predates-candidate")
    if decision.decided_at > candidate.valid_through:
        reasons.append("release-candidate-evidence-expired")
    if not decision.approved:
        reasons.append("release-decision:not-approved")
    blockers = tuple(dict.fromkeys(reasons))
    ready = not blockers
    released_profile = CapabilityProfile(
        candidate.capability_profile.name,
        candidate.capability_profile.provider,
        candidate.capability_profile.version,
        candidate.capability_profile.support_tuples,
        dependencies=candidate.capability_profile.dependencies,
        required_gates=candidate.capability_profile.required_gates,
        release_evidence=candidate.capability_profile.release_evidence,
        released=ready,
    )
    assessment_id = canonical_fingerprint(
        _assessment_content_record(
            candidate.candidate_id,
            decision.decision_id,
            released_profile,
            ready,
            blockers,
        )
    )
    return CardiovascularReleaseAssessment(
        candidate.candidate_id,
        decision.decision_id,
        released_profile,
        ready,
        blockers,
        assessment_id,
    )


__all__ = [
    "CardiovascularArtifactKind",
    "CardiovascularArtifactReference",
    "CardiovascularArtifactSet",
    "CardiovascularClaimDecision",
    "CardiovascularClaimsMatrix",
    "CardiovascularClaimStatus",
    "CardiovascularCommercialSupportProfile",
    "CardiovascularGateEvidence",
    "CardiovascularPrivacyPolicy",
    "CardiovascularQualificationBundle",
    "CardiovascularReleaseAssessment",
    "CardiovascularReleaseCandidate",
    "CardiovascularReleaseDecision",
    "CardiovascularReleaseGate",
    "CardiovascularResourcePolicy",
    "CardiovascularReviewRoles",
    "CardiovascularSecurityPolicy",
    "CardiovascularSignatureVerifier",
    "CardiovascularSignedNonClaim",
    "CardiovascularUsePolicy",
    "assess_cardiovascular_release",
    "evaluate_cardiovascular_release_candidate",
    "make_cardiovascular_release_decision",
]
