#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Mapping, Sequence
from typing import Protocol

import equinox as eqx

from .._fingerprint import canonical_fingerprint, canonical_json
from .._strict import StrictModule
from .._trainable import NonTrainableState


SupportValue = str | int | bool
_MAX_TIMESTAMP = 2**63 - 1
_HEX_DIGITS = frozenset("0123456789abcdef")
_CAPABILITY_NAME = re.compile(r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*$")


def _identifier(value: str, name: str, /) -> str:
    normalized = str(value)
    if not normalized or normalized != normalized.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return normalized


def _capability_name(value: str, name: str, /) -> str:
    normalized = _identifier(value, name)
    if _CAPABILITY_NAME.fullmatch(normalized) is None:
        raise ValueError(
            f"{name} must use lowercase dotted namespaces and hyphenated compounds."
        )
    return normalized


def _timestamp(value: int, name: str, /) -> int:
    normalized = int(value)
    if normalized < 0 or normalized > _MAX_TIMESTAMP:
        raise ValueError(f"{name} must be a non-negative signed 64-bit timestamp.")
    return normalized


def _support_value(value: object, /) -> SupportValue:
    if type(value) not in (str, int, bool):
        raise TypeError("Support-tuple values must be strings, integers, or booleans.")
    if isinstance(value, str) and not value:
        raise ValueError("Support-tuple string values must be non-empty.")
    return value


class SupportTuple(StrictModule, NonTrainableState):
    """One exact, provider-neutral conjunction of capability coordinates."""

    capability: str = eqx.field(static=True)
    attributes: tuple[tuple[str, SupportValue], ...] = eqx.field(static=True)
    support_tuple_id: str = eqx.field(static=True)

    def __init__(
        self,
        capability: str,
        attributes: Mapping[str, SupportValue],
        /,
    ):
        capability_ = _capability_name(capability, "capability")
        if not isinstance(attributes, Mapping) or not attributes:
            raise TypeError("attributes must be a non-empty mapping.")
        normalized = tuple(
            sorted(
                (
                    _identifier(str(name), "support coordinate"),
                    _support_value(value),
                )
                for name, value in attributes.items()
            )
        )
        names = tuple(name for name, _ in normalized)
        if len(set(names)) != len(names):
            raise ValueError("Support-tuple coordinate names must be unique.")
        self.capability = capability_
        self.attributes = normalized
        self.support_tuple_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "capability-support-tuple",
            "capability": self.capability,
            "attributes": dict(self.attributes),
        }

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-ready record including the content address."""
        return {**self._content_record(), "support_tuple_id": self.support_tuple_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> SupportTuple:
        """Reconstruct and content-verify a serialized support tuple."""
        if not isinstance(record, Mapping):
            raise TypeError("Support-tuple record must be a mapping.")
        attributes = record["attributes"]
        if not isinstance(attributes, Mapping):
            raise TypeError("Serialized support-tuple attributes must be a mapping.")
        value = cls(str(record["capability"]), attributes)
        recorded_id = record.get("support_tuple_id")
        if recorded_id is not None and str(recorded_id) != value.support_tuple_id:
            raise ValueError("Serialized support tuple has an invalid content address.")
        return value


class ReleaseGateEvidence(StrictModule, NonTrainableState):
    """Time-bounded evidence for one provider-neutral release gate."""

    gate: str = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    reviewer_id: str = eqx.field(static=True)
    deviation_ids: tuple[str, ...] = eqx.field(static=True)
    issued_at: int = eqx.field(static=True)
    expires_at: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        gate: str,
        /,
        *,
        passed: bool,
        evidence_ids: Sequence[str],
        reviewer_id: str,
        issued_at: int,
        expires_at: int,
        deviation_ids: Sequence[str] = (),
    ):
        gate_ = _identifier(gate, "gate")
        evidence = tuple(_identifier(value, "evidence ID") for value in evidence_ids)
        reviewer = _identifier(reviewer_id, "reviewer ID")
        deviations = tuple(_identifier(value, "deviation ID") for value in deviation_ids)
        issued = _timestamp(issued_at, "issued_at")
        expires = _timestamp(expires_at, "expires_at")
        if not evidence:
            raise ValueError("Release-gate evidence must cite at least one artifact.")
        if len(set(evidence)) != len(evidence) or len(set(deviations)) != len(deviations):
            raise ValueError("Evidence and deviation IDs must be unique.")
        if expires <= issued:
            raise ValueError("Release-gate evidence must expire after it is issued.")
        self.gate = gate_
        self.passed = bool(passed)
        self.evidence_ids = tuple(sorted(evidence))
        self.reviewer_id = reviewer
        self.deviation_ids = tuple(sorted(deviations))
        self.issued_at = issued
        self.expires_at = expires
        self.evidence_id = canonical_fingerprint(self._content_record())

    @property
    def accepted(self) -> bool:
        return self.passed and not self.deviation_ids

    def is_current(self, at_time: int, /) -> bool:
        timestamp = _timestamp(at_time, "at_time")
        return self.issued_at <= timestamp <= self.expires_at

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "release-gate-evidence",
            "gate": self.gate,
            "passed": self.passed,
            "evidence_ids": list(self.evidence_ids),
            "reviewer_id": self.reviewer_id,
            "deviation_ids": list(self.deviation_ids),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-ready evidence record."""
        return {**self._content_record(), "evidence_id": self.evidence_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> ReleaseGateEvidence:
        """Reconstruct and content-verify serialized release evidence."""
        if not isinstance(record, Mapping):
            raise TypeError("Release-evidence record must be a mapping.")
        evidence_ids = record["evidence_ids"]
        deviation_ids = record["deviation_ids"]
        if not isinstance(evidence_ids, Sequence) or isinstance(evidence_ids, str):
            raise TypeError("Serialized evidence IDs must be a sequence.")
        if not isinstance(deviation_ids, Sequence) or isinstance(deviation_ids, str):
            raise TypeError("Serialized deviation IDs must be a sequence.")
        value = cls(
            str(record["gate"]),
            passed=bool(record["passed"]),
            evidence_ids=tuple(str(item) for item in evidence_ids),
            reviewer_id=str(record["reviewer_id"]),
            issued_at=int(record["issued_at"]),
            expires_at=int(record["expires_at"]),
            deviation_ids=tuple(str(item) for item in deviation_ids),
        )
        recorded_id = record.get("evidence_id")
        if recorded_id is not None and str(recorded_id) != value.evidence_id:
            raise ValueError(
                "Serialized release evidence has an invalid content address."
            )
        return value


class CapabilityProfile(StrictModule, NonTrainableState):
    """Content-addressed declaration of exact support and its release evidence."""

    name: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    version: str = eqx.field(static=True)
    support_tuples: tuple[SupportTuple, ...]
    dependencies: tuple[str, ...] = eqx.field(static=True)
    required_gates: tuple[str, ...] = eqx.field(static=True)
    release_evidence: tuple[ReleaseGateEvidence, ...]
    released: bool = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        provider: str,
        version: str,
        support_tuples: Sequence[SupportTuple],
        /,
        *,
        dependencies: Sequence[str] = (),
        required_gates: Sequence[str] = (),
        release_evidence: Sequence[ReleaseGateEvidence] = (),
        released: bool = False,
    ):
        name_ = _capability_name(name, "profile name")
        provider_ = _identifier(provider, "provider")
        version_ = _identifier(version, "profile version")
        tuples_ = tuple(support_tuples)
        if not tuples_ or any(not isinstance(item, SupportTuple) for item in tuples_):
            raise TypeError("support_tuples must contain typed, non-empty support.")
        tuple_ids = tuple(item.support_tuple_id for item in tuples_)
        if len(set(tuple_ids)) != len(tuple_ids):
            raise ValueError("Capability profile contains duplicate support tuples.")
        capabilities = {item.capability for item in tuples_}
        if len(capabilities) != 1:
            raise ValueError("A capability profile must describe one capability family.")
        dependencies_ = tuple(
            sorted(_identifier(item, "dependency profile ID") for item in dependencies)
        )
        gates = tuple(
            sorted(_identifier(item, "required gate") for item in required_gates)
        )
        evidence = tuple(release_evidence)
        if any(not isinstance(item, ReleaseGateEvidence) for item in evidence):
            raise TypeError("release_evidence must contain ReleaseGateEvidence values.")
        if len(set(dependencies_)) != len(dependencies_):
            raise ValueError("Capability profile dependencies must be unique.")
        if len(set(gates)) != len(gates):
            raise ValueError("Capability profile required gates must be unique.")
        evidence_gates = tuple(item.gate for item in evidence)
        if len(set(evidence_gates)) != len(evidence_gates):
            raise ValueError("Capability profile has duplicate release-gate evidence.")
        released_ = bool(released)
        if released_ and set(evidence_gates) != set(gates):
            raise ValueError("A released profile needs evidence for every required gate.")
        if released_ and any(not item.accepted for item in evidence):
            raise ValueError(
                "A released profile cannot contain failed or deviated evidence."
            )
        self.name = name_
        self.provider = provider_
        self.version = version_
        self.support_tuples = tuple(
            sorted(tuples_, key=lambda item: item.support_tuple_id)
        )
        self.dependencies = dependencies_
        self.required_gates = gates
        self.release_evidence = tuple(
            sorted(evidence, key=lambda item: (item.gate, item.evidence_id))
        )
        self.released = released_
        self.profile_id = canonical_fingerprint(self._content_record())

    @property
    def capability(self) -> str:
        return self.support_tuples[0].capability

    def supports(self, support_tuple: SupportTuple, /) -> bool:
        if not isinstance(support_tuple, SupportTuple):
            raise TypeError("support_tuple must be a SupportTuple.")
        return any(
            item.support_tuple_id == support_tuple.support_tuple_id
            for item in self.support_tuples
        )

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "capability-profile",
            "name": self.name,
            "provider": self.provider,
            "version": self.version,
            "support_tuples": [item.to_record() for item in self.support_tuples],
            "dependencies": list(self.dependencies),
            "required_gates": list(self.required_gates),
            "release_evidence": [item.to_record() for item in self.release_evidence],
            "released": self.released,
        }

    def to_record(self) -> dict[str, object]:
        """Return the complete deterministic JSON-ready producer record."""
        return {**self._content_record(), "profile_id": self.profile_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> CapabilityProfile:
        """Reconstruct and content-verify a serialized capability profile."""
        if not isinstance(record, Mapping):
            raise TypeError("Capability-profile record must be a mapping.")
        tuple_records = record["support_tuples"]
        dependencies = record["dependencies"]
        required_gates = record["required_gates"]
        evidence_records = record["release_evidence"]
        for name, values in (
            ("support_tuples", tuple_records),
            ("dependencies", dependencies),
            ("required_gates", required_gates),
            ("release_evidence", evidence_records),
        ):
            if not isinstance(values, Sequence) or isinstance(values, str):
                raise TypeError(f"Serialized {name} must be a sequence.")
        if any(not isinstance(item, Mapping) for item in tuple_records):
            raise TypeError("Serialized support tuples must be mappings.")
        if any(not isinstance(item, Mapping) for item in evidence_records):
            raise TypeError("Serialized release evidence must be mappings.")
        value = cls(
            str(record["name"]),
            str(record["provider"]),
            str(record["version"]),
            tuple(SupportTuple.from_record(item) for item in tuple_records),
            dependencies=tuple(str(item) for item in dependencies),
            required_gates=tuple(str(item) for item in required_gates),
            release_evidence=tuple(
                ReleaseGateEvidence.from_record(item) for item in evidence_records
            ),
            released=bool(record["released"]),
        )
        recorded_id = record.get("profile_id")
        if recorded_id is not None and str(recorded_id) != value.profile_id:
            raise ValueError(
                "Serialized capability profile has an invalid content address."
            )
        return value


class ReleaseSigner(Protocol):
    """Provider interface used by :meth:`ReleaseIndex.sign`."""

    @property
    def signer_id(self) -> str: ...

    @property
    def signature_algorithm(self) -> str: ...

    def sign(self, payload: bytes, /) -> bytes: ...


class ReleaseTrustPolicy(Protocol):
    """Trust and freshness interface used for registry evaluation."""

    def verify_index(self, index: ReleaseIndex, at_time: int, /) -> bool: ...

    def accepts_evidence(
        self, evidence: ReleaseGateEvidence, at_time: int, /
    ) -> bool: ...


class HMACSHA256ReleaseSigner:
    """Small deterministic signer for local registries and sealed CI secrets."""

    __slots__ = ("_secret", "_signer_id")

    def __init__(self, signer_id: str, secret: bytes, /):
        signer = _identifier(signer_id, "signer ID")
        if not isinstance(secret, bytes) or not secret:
            raise TypeError("HMAC signing secret must be non-empty bytes.")
        self._signer_id = signer
        self._secret = secret

    @property
    def signer_id(self) -> str:
        return self._signer_id

    @property
    def signature_algorithm(self) -> str:
        return "hmac-sha256"

    def sign(self, payload: bytes, /) -> bytes:
        if not isinstance(payload, bytes):
            raise TypeError("Signed release-index payload must be bytes.")
        return hmac.new(self._secret, payload, hashlib.sha256).digest()


class HMACSHA256TrustPolicy:
    """Allow-list trust policy with index and evidence freshness limits."""

    __slots__ = (
        "_maximum_evidence_age",
        "_maximum_index_age",
        "_trusted_keys",
    )

    def __init__(
        self,
        trusted_signers: Mapping[str, bytes],
        /,
        *,
        maximum_index_age: int,
        maximum_evidence_age: int,
    ):
        if not isinstance(trusted_signers, Mapping) or not trusted_signers:
            raise TypeError("trusted_signers must be a non-empty signer-key mapping.")
        keys = tuple(
            sorted(
                (
                    _identifier(signer, "trusted signer ID"),
                    key,
                )
                for signer, key in trusted_signers.items()
            )
        )
        if any(not isinstance(key, bytes) or not key for _, key in keys):
            raise TypeError("Trusted HMAC keys must be non-empty bytes.")
        maximum_index_age_ = int(maximum_index_age)
        maximum_evidence_age_ = int(maximum_evidence_age)
        if maximum_index_age_ <= 0 or maximum_evidence_age_ <= 0:
            raise ValueError("Trust-policy freshness limits must be positive.")
        self._trusted_keys = keys
        self._maximum_index_age = maximum_index_age_
        self._maximum_evidence_age = maximum_evidence_age_

    def verify_index(self, index: ReleaseIndex, at_time: int, /) -> bool:
        if not isinstance(index, ReleaseIndex):
            raise TypeError("index must be a ReleaseIndex.")
        timestamp = _timestamp(at_time, "at_time")
        if (
            index.signature_algorithm != "hmac-sha256"
            or index.issued_at > timestamp
            or timestamp - index.issued_at > self._maximum_index_age
        ):
            return False
        trusted = dict(self._trusted_keys)
        if index.signer_id not in trusted:
            return False
        signature = index.signature
        if len(signature) != 64 or any(
            character not in _HEX_DIGITS for character in signature
        ):
            return False
        expected = hmac.new(
            trusted[index.signer_id], index.signed_payload, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    def accepts_evidence(self, evidence: ReleaseGateEvidence, at_time: int, /) -> bool:
        if not isinstance(evidence, ReleaseGateEvidence):
            raise TypeError("evidence must be ReleaseGateEvidence.")
        timestamp = _timestamp(at_time, "at_time")
        return (
            evidence.accepted
            and evidence.is_current(timestamp)
            and timestamp - evidence.issued_at <= self._maximum_evidence_age
        )


class ReleaseIndex(StrictModule, NonTrainableState):
    """Content-addressed profile set authenticated by an external signer."""

    profiles: tuple[CapabilityProfile, ...]
    issued_at: int = eqx.field(static=True)
    signer_id: str = eqx.field(static=True)
    signature_algorithm: str = eqx.field(static=True)
    index_id: str = eqx.field(static=True)
    signature: str = eqx.field(static=True)

    def __init__(
        self,
        profiles: Sequence[CapabilityProfile],
        /,
        *,
        issued_at: int,
        signer_id: str,
        signature_algorithm: str,
        signature: str,
    ):
        profiles_ = tuple(profiles)
        if not profiles_ or any(
            not isinstance(item, CapabilityProfile) for item in profiles_
        ):
            raise TypeError("Release index requires typed, non-empty profiles.")
        ids = tuple(item.profile_id for item in profiles_)
        coordinates = tuple(
            (item.provider, item.name, item.version) for item in profiles_
        )
        if len(set(ids)) != len(ids) or len(set(coordinates)) != len(coordinates):
            raise ValueError("Release index contains duplicate profiles.")
        self.profiles = tuple(sorted(profiles_, key=lambda item: item.profile_id))
        self.issued_at = _timestamp(issued_at, "issued_at")
        self.signer_id = _identifier(signer_id, "signer ID")
        self.signature_algorithm = _identifier(signature_algorithm, "signature algorithm")
        self.index_id = canonical_fingerprint(self._content_record())
        signature_ = str(signature).lower()
        if not signature_:
            raise ValueError("Release index signature must be non-empty.")
        self.signature = signature_

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "release-index",
            "profiles": [item.profile_id for item in self.profiles],
            "issued_at": self.issued_at,
            "signer_id": self.signer_id,
            "signature_algorithm": self.signature_algorithm,
        }

    @property
    def signed_payload(self) -> bytes:
        return canonical_json(self._content_record()).encode("utf-8")

    @classmethod
    def sign(
        cls,
        profiles: Sequence[CapabilityProfile],
        signer: ReleaseSigner,
        /,
        *,
        issued_at: int,
    ) -> ReleaseIndex:
        """Build and sign a deterministic content-addressed index."""
        unsigned = cls(
            profiles,
            issued_at=issued_at,
            signer_id=signer.signer_id,
            signature_algorithm=signer.signature_algorithm,
            signature="unsigned",
        )
        signature = signer.sign(unsigned.signed_payload)
        if not isinstance(signature, bytes) or not signature:
            raise TypeError("Release signer must return non-empty bytes.")
        return cls(
            unsigned.profiles,
            issued_at=unsigned.issued_at,
            signer_id=unsigned.signer_id,
            signature_algorithm=unsigned.signature_algorithm,
            signature=signature.hex(),
        )

    def to_record(self) -> dict[str, object]:
        """Return a complete deterministic JSON-ready signed index record."""
        return {
            **self._content_record(),
            "profile_records": [item.to_record() for item in self.profiles],
            "index_id": self.index_id,
            "signature": self.signature,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> ReleaseIndex:
        """Reconstruct and content-verify a signed release index."""
        if not isinstance(record, Mapping):
            raise TypeError("Release-index record must be a mapping.")
        profile_records = record["profile_records"]
        if not isinstance(profile_records, Sequence) or isinstance(profile_records, str):
            raise TypeError("Serialized profile records must be a sequence.")
        if any(not isinstance(item, Mapping) for item in profile_records):
            raise TypeError("Serialized profiles must be mappings.")
        value = cls(
            tuple(CapabilityProfile.from_record(item) for item in profile_records),
            issued_at=int(record["issued_at"]),
            signer_id=str(record["signer_id"]),
            signature_algorithm=str(record["signature_algorithm"]),
            signature=str(record["signature"]),
        )
        listed_profiles = record["profiles"]
        if not isinstance(listed_profiles, Sequence) or isinstance(listed_profiles, str):
            raise TypeError("Serialized profile IDs must be a sequence.")
        if tuple(str(item) for item in listed_profiles) != tuple(
            item.profile_id for item in value.profiles
        ):
            raise ValueError("Release index profile records do not match its manifest.")
        if str(record["index_id"]) != value.index_id:
            raise ValueError("Serialized release index has an invalid content address.")
        return value

    def require_trusted(self, policy: ReleaseTrustPolicy, at_time: int, /) -> str:
        if not policy.verify_index(self, at_time):
            raise ValueError(
                "Release index signature, signer trust, or freshness failed."
            )
        return self.index_id


def _profile_rejection_reasons(
    profile: CapabilityProfile,
    profiles_by_id: Mapping[str, CapabilityProfile],
    policy: ReleaseTrustPolicy,
    at_time: int,
    support_tuple: SupportTuple | None,
    ancestry: tuple[str, ...],
    /,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not profile.released:
        reasons.append("profile-unreleased")
    if support_tuple is not None and not profile.supports(support_tuple):
        reasons.append(f"unsupported-tuple:{support_tuple.support_tuple_id}")
    evidence_by_gate = {item.gate: item for item in profile.release_evidence}
    for gate in profile.required_gates:
        if gate not in evidence_by_gate:
            reasons.append(f"missing-evidence:{gate}")
        elif not policy.accepts_evidence(evidence_by_gate[gate], at_time):
            reasons.append(f"rejected-or-stale-evidence:{gate}")
    for dependency_id in profile.dependencies:
        if dependency_id not in profiles_by_id:
            reasons.append(f"missing-dependency:{dependency_id}")
        elif dependency_id in ancestry:
            reasons.append(f"dependency-cycle:{dependency_id}")
        else:
            dependency_reasons = _profile_rejection_reasons(
                profiles_by_id[dependency_id],
                profiles_by_id,
                policy,
                at_time,
                None,
                ancestry + (dependency_id,),
            )
            reasons.extend(
                f"dependency:{dependency_id}:{reason}" for reason in dependency_reasons
            )
    return tuple(reasons)


def discover_profiles(
    index: ReleaseIndex,
    support_tuple: SupportTuple,
    trust_policy: ReleaseTrustPolicy,
    /,
    *,
    at_time: int,
) -> tuple[CapabilityProfile, ...]:
    """Discover released profiles satisfying the exact tuple and every dependency."""
    if not isinstance(index, ReleaseIndex):
        raise TypeError("index must be a ReleaseIndex.")
    if not isinstance(support_tuple, SupportTuple):
        raise TypeError("support_tuple must be a SupportTuple.")
    index.require_trusted(trust_policy, at_time)
    profiles_by_id = {item.profile_id: item for item in index.profiles}
    discovered = tuple(
        profile
        for profile in index.profiles
        if not _profile_rejection_reasons(
            profile,
            profiles_by_id,
            trust_policy,
            at_time,
            support_tuple,
            (profile.profile_id,),
        )
    )
    return tuple(
        sorted(
            discovered,
            key=lambda item: (item.provider, item.name, item.version, item.profile_id),
        )
    )


def require_profile(
    index: ReleaseIndex,
    profile_id: str,
    support_tuple: SupportTuple,
    trust_policy: ReleaseTrustPolicy,
    /,
    *,
    at_time: int,
) -> CapabilityProfile:
    """Require one exact profile, rejecting any failed AND-clause explicitly."""
    if not isinstance(index, ReleaseIndex):
        raise TypeError("index must be a ReleaseIndex.")
    if not isinstance(support_tuple, SupportTuple):
        raise TypeError("support_tuple must be a SupportTuple.")
    index.require_trusted(trust_policy, at_time)
    identifier = _identifier(profile_id, "profile ID")
    profiles_by_id = {item.profile_id: item for item in index.profiles}
    if identifier not in profiles_by_id:
        raise KeyError(
            f"No capability profile {identifier!r} exists in the release index."
        )
    profile = profiles_by_id[identifier]
    reasons = _profile_rejection_reasons(
        profile,
        profiles_by_id,
        trust_policy,
        at_time,
        support_tuple,
        (identifier,),
    )
    if reasons:
        raise ValueError(
            f"Capability profile {identifier} is not admissible: " + "; ".join(reasons)
        )
    return profile


__all__ = [
    "CapabilityProfile",
    "HMACSHA256ReleaseSigner",
    "HMACSHA256TrustPolicy",
    "ReleaseGateEvidence",
    "ReleaseIndex",
    "ReleaseSigner",
    "ReleaseTrustPolicy",
    "SupportTuple",
    "discover_profiles",
    "require_profile",
]
