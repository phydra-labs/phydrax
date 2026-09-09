#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass

from ..._fingerprint import canonical_fingerprint
from ...qualification._evidence import SupportDependency
from ...qualification._registry import (
    CapabilityProfile,
    ReleaseGateEvidence,
    ReleaseIndex,
    require_profile,
)
from ...qualification._trust import (
    AsymmetricReleaseTrustPolicy,
    SignedQualificationRecord,
)
from ._release import BatteryReleaseRecord
from ._release_contracts import CIRCUIT_ECM_SCIENTIFIC_METRICS


SERIES_PACK_ENTRY_CASES = ("charge-rest", "discharge-rest")
SERIES_PACK_ENTRY_METRICS = (
    "maximum-normalized-analytic-error",
    "maximum-kcl-defect-a",
    "maximum-power-defect-w",
    "maximum-thermal-defect-w",
)


@dataclass(frozen=True, slots=True)
class SeriesPackEntryAssessment:
    circuit_release: BatteryReleaseRecord
    authority_signature: SignedQualificationRecord

    def decision_record(self) -> dict[str, object]:
        bundle = self.circuit_release.bundle
        selected = tuple(
            item
            for item in bundle.criteria
            if item.applicability in SERIES_PACK_ENTRY_CASES
            and item.metric in SERIES_PACK_ENTRY_METRICS
        )
        selected_ids = {item.criterion_id for item in selected}
        return {
            "kind": "series-pack-entry-decision",
            "eligible": True,
            "circuit_profile_id": self.circuit_release.profile.profile_id,
            "distribution_id": self.circuit_release.distribution_id,
            "envelope_id": self.circuit_release.envelope_id,
            "criteria": [item.to_record() for item in selected],
            "starts": [
                item.to_record()
                for item in bundle.starts
                if item.criterion_id in selected_ids
            ],
            "observations": [
                item.to_record()
                for item in bundle.observations
                if item.criterion_id in selected_ids
            ],
            "evidence": [
                item.to_record()
                for item in bundle.evidence
                if set(item.criteria_ids) <= selected_ids
            ],
            "gate": self.circuit_release.profile.release_evidence[0].to_record(),
            "coverage": bundle.coverage.to_record(),
        }

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "series-pack-entry-assessment",
            "circuit_release": self.circuit_release.to_record(),
            "authority_signature": self.authority_signature.to_record(),
        }

    @classmethod
    def from_record(cls, record, /):
        if record.get("kind") != "series-pack-entry-assessment":
            raise ValueError("Invalid pack-entry assessment record.")
        return cls(
            BatteryReleaseRecord.from_record(record["circuit_release"]),
            SignedQualificationRecord.from_record(record["authority_signature"]),
        )

    def verify(
        self,
        index: ReleaseIndex,
        policy: AsymmetricReleaseTrustPolicy,
        /,
        *,
        at_time: int,
    ) -> int:
        from ._qualification import CIRCUIT_ECM_SUPPORT

        if not isinstance(self.circuit_release, BatteryReleaseRecord) or not isinstance(
            policy, AsymmetricReleaseTrustPolicy
        ):
            raise TypeError(
                "Pack entry requires a retained typed circuit release and asymmetric role trust."
            )
        profile = require_profile(
            index,
            self.circuit_release.profile.profile_id,
            CIRCUIT_ECM_SUPPORT,
            policy,
            at_time=at_time,
        )
        if profile.profile_id != self.circuit_release.profile.profile_id:
            raise ValueError("Pack entry references a substituted circuit release.")
        self.circuit_release.verify(policy, at_time=at_time)
        selected = [
            item
            for item in self.circuit_release.bundle.criteria
            if item.applicability in SERIES_PACK_ENTRY_CASES
            and item.metric in SERIES_PACK_ENTRY_METRICS
        ]
        expected = {
            (case, metric)
            for case in SERIES_PACK_ENTRY_CASES
            for metric in SERIES_PACK_ENTRY_METRICS
        }
        if (
            len(selected) != len(expected)
            or {(item.applicability, item.metric) for item in selected} != expected
        ):
            raise ValueError(
                "Pack entry requires both current polarities' exact typed sign/KCL/power/thermal criteria."
            )
        definitions = {metric.name: metric for metric in CIRCUIT_ECM_SCIENTIFIC_METRICS}
        for criterion in selected:
            definitions[criterion.metric].validate(criterion)
        policy.roles.verify(
            self.decision_record(),
            self.authority_signature,
            role="entry-decision-authority",
            at_time=at_time,
        )
        if self.authority_signature.issued_at < self.circuit_release.issued_at:
            raise ValueError("Pack-entry authority cannot precede the circuit release.")
        return min(
            self.circuit_release.expires_at, policy.roles.expiry(self.authority_signature)
        )


def evaluate_series_pack_entry_gate(
    *,
    release_index: ReleaseIndex | None,
    profile_id: str | None,
    trust_policy: AsymmetricReleaseTrustPolicy | None,
    assessment: SeriesPackEntryAssessment | None,
    at_time: int,
):
    from ._qualification import BatteryExpansionGateDecision

    try:
        if (
            not isinstance(assessment, SeriesPackEntryAssessment)
            or not isinstance(assessment.circuit_release, BatteryReleaseRecord)
            or profile_id != assessment.circuit_release.profile.profile_id
        ):
            raise ValueError("Exact typed circuit entry assessment is missing.")
        assessment.verify(release_index, trust_policy, at_time=at_time)
    except (TypeError, ValueError, KeyError, RuntimeError):
        return BatteryExpansionGateDecision(
            False, False, "series-pack-entry-inconclusive", ()
        )
    return BatteryExpansionGateDecision(
        True,
        True,
        "series-pack-entry-criteria-satisfied",
        (
            canonical_fingerprint(assessment.decision_record()),
            assessment.authority_signature.attestation_id,
        ),
    )


@dataclass(frozen=True, slots=True)
class SeriesPackEntryReleaseRecord:
    profile: CapabilityProfile
    assessment: SeriesPackEntryAssessment
    prerequisite_index: ReleaseIndex
    issued_at: int
    expires_at: int

    @property
    def gate_id(self) -> str:
        return self.profile.release_evidence[0].evidence_id

    @property
    def distribution_id(self) -> str:
        return self.assessment.circuit_release.distribution_id

    @property
    def envelope_id(self) -> str:
        return self.assessment.circuit_release.envelope_id

    def verify(self, policy, /, *, at_time: int) -> None:
        cap = self.assessment.verify(self.prerequisite_index, policy, at_time=at_time)
        if not self.issued_at <= at_time < self.expires_at or self.expires_at > cap:
            raise ValueError("Pack-entry release is stale or outlives its proof.")
        expected = _entry_profile(self.assessment, self.issued_at, self.expires_at)
        if self.profile.to_record() != expected.to_record():
            raise ValueError("Pack-entry profile was substituted.")

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "series-pack-entry-release-record",
            "profile": self.profile.to_record(),
            "assessment": self.assessment.to_record(),
            "prerequisite_index": self.prerequisite_index.to_record(),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_record(cls, record, /):
        if record.get("kind") != "series-pack-entry-release-record":
            raise ValueError("Invalid pack-entry release record.")
        return cls(
            CapabilityProfile.from_record(record["profile"]),
            SeriesPackEntryAssessment.from_record(record["assessment"]),
            ReleaseIndex.from_record(record["prerequisite_index"]),
            record["issued_at"],
            record["expires_at"],
        )


def _entry_profile(assessment, issued_at, expires_at):
    from ._qualification import CIRCUIT_ECM_SUPPORT, SERIES_PACK_ENTRY_SUPPORT

    gate = ReleaseGateEvidence(
        "battery.series-pack.entry",
        passed=True,
        evidence_ids=(
            canonical_fingerprint(assessment.decision_record()),
            assessment.authority_signature.attestation_id,
        ),
        reviewer_id=assessment.authority_signature.signature.key_id,
        issued_at=issued_at,
        expires_at=expires_at,
    )
    return CapabilityProfile(
        "battery.series-pack.entry",
        "phydrax",
        "eligible",
        (SERIES_PACK_ENTRY_SUPPORT,),
        dependencies=(
            SupportDependency(
                assessment.circuit_release.profile.profile_id,
                CIRCUIT_ECM_SUPPORT.support_tuple_id,
            ),
        ),
        required_gates=(gate.gate,),
        release_evidence=(gate,),
        released=True,
    )


def build_series_pack_entry_release(
    assessment: SeriesPackEntryAssessment,
    prerequisite_index: ReleaseIndex,
    /,
    *,
    trust_policy: AsymmetricReleaseTrustPolicy,
    at_time: int,
    expires_at: int,
) -> SeriesPackEntryReleaseRecord:
    cap = assessment.verify(prerequisite_index, trust_policy, at_time=at_time)
    if type(expires_at) is not int or expires_at <= at_time or cap <= at_time:
        raise ValueError("Pack-entry expiry must follow issuance.")
    expiry = min(expires_at, cap)
    return SeriesPackEntryReleaseRecord(
        _entry_profile(assessment, at_time, expiry),
        assessment,
        prerequisite_index,
        at_time,
        expiry,
    )
