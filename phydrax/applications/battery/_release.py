#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Retained, offline-verifiable numerical release construction; IDs are not proof."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from ..._fingerprint import canonical_fingerprint, canonical_json
from ...qualification._criterion import (
    CampaignObservationRecord,
    CampaignStartRecord,
    QualificationCriterion,
    validate_qualification_causality,
)
from ...qualification._evidence import (
    ObservedResourceRecord,
    QualificationCoverageReport,
    QualificationEvidence,
    QualificationMatrix,
    SupportDependency,
)
from ...qualification._reference import ReferenceArtifactManifest
from ...qualification._registry import (
    CapabilityProfile,
    ReleaseGateEvidence,
    ReleaseIndex,
    require_profile,
)
from ...qualification._runtime_distribution import parse_distribution_manifest
from ...qualification._trust import (
    AsymmetricReleaseTrustPolicy,
    SignedQualificationRecord,
)
from ._release_contracts import battery_release_contract, RESOURCE_METRICS
from ._rights import ReferenceRightsAttestation
from ._validity import BATTERY_NUMERICAL_ENVELOPES, BatteryValidityEnvelope


REQUIRED_RESOURCE_MEASUREMENTS = frozenset(name for name, _, _, _ in RESOURCE_METRICS)


def _unique(items, identity):
    result = {identity(item): item for item in items}
    if len(result) != len(items):
        raise ValueError("Typed release records must have unique identities.")
    return result


def _envelope_from_record(record) -> BatteryValidityEnvelope:
    if record.get("kind") != "battery-validity-envelope":
        raise ValueError("Invalid battery validity envelope record.")
    values = {key: value for key, value in record.items() if key != "kind"}
    for key in (
        "geometry_bounds",
        "operating_bounds",
        "production_parameter_bounds",
        "production_initial_bounds",
        "production_protocol_bounds",
        "production_layout_bounds",
        "resource_limits",
    ):
        values[key] = tuple(tuple(row) for row in values[key])
    values["parameter_ranges"] = tuple(values["parameter_ranges"].items())
    values["property_support_ids"] = tuple(values["property_support_ids"])
    return BatteryValidityEnvelope(**values)


@dataclass(frozen=True, slots=True)
class CleanReplayComparison:
    case_ids: tuple[str, ...]
    allowed_environment_differences: tuple[str, ...]
    observable_limits: tuple[tuple[str, float], ...]
    observed_discrepancies: tuple[tuple[str, float], ...]
    starts: tuple[CampaignStartRecord, ...]
    observations: tuple[CampaignObservationRecord, ...]
    evidence: tuple[QualificationEvidence, ...]
    fresh_install: bool
    reused_raw_bytes: bool
    reused_caches: bool
    reused_checkpoints: bool

    def plan_record(self) -> dict[str, object]:
        return {
            "case_ids": list(self.case_ids),
            "allowed_environment_differences": list(self.allowed_environment_differences),
            "observable_limits": dict(self.observable_limits),
        }

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "battery-clean-replay-comparison",
            **self.plan_record(),
            "observed_discrepancies": dict(self.observed_discrepancies),
            "starts": [item.to_record() for item in self.starts],
            "observations": [item.to_record() for item in self.observations],
            "evidence": [item.to_record() for item in self.evidence],
            "fresh_install": self.fresh_install,
            "reused_raw_bytes": self.reused_raw_bytes,
            "reused_caches": self.reused_caches,
            "reused_checkpoints": self.reused_checkpoints,
        }

    @property
    def comparison_id(self) -> str:
        return canonical_fingerprint(self.to_record())

    @classmethod
    def from_record(cls, record, /):
        if record.get("kind") != "battery-clean-replay-comparison":
            raise ValueError("Invalid clean replay record.")
        return cls(
            tuple(record["case_ids"]),
            tuple(record["allowed_environment_differences"]),
            tuple(record["observable_limits"].items()),
            tuple(record["observed_discrepancies"].items()),
            tuple(CampaignStartRecord.from_record(row) for row in record["starts"]),
            tuple(
                CampaignObservationRecord.from_record(row)
                for row in record["observations"]
            ),
            tuple(QualificationEvidence.from_record(row) for row in record["evidence"]),
            record["fresh_install"],
            record["reused_raw_bytes"],
            record["reused_caches"],
            record["reused_checkpoints"],
        )


@dataclass(frozen=True, slots=True)
class BatteryReleaseBundle:
    envelope: BatteryValidityEnvelope
    distribution_id: str
    distribution_manifest_json: str
    criteria: tuple[QualificationCriterion, ...]
    starts: tuple[CampaignStartRecord, ...]
    observations: tuple[CampaignObservationRecord, ...]
    evidence: tuple[QualificationEvidence, ...]
    references: tuple[ReferenceArtifactManifest, ...]
    reference_contents: tuple[bytes, ...]
    rights: tuple[ReferenceRightsAttestation, ...]
    resources: tuple[ObservedResourceRecord, ...]
    replay: CleanReplayComparison
    matrix: QualificationMatrix
    coverage: QualificationCoverageReport
    raw_records_json: tuple[str, ...]
    attestations: tuple[SignedQualificationRecord, ...]
    dependencies: tuple[SupportDependency, ...] = ()
    prerequisite_index: ReleaseIndex | None = None

    @property
    def source_build_id(self) -> str:
        manifest = parse_distribution_manifest(self.distribution_manifest_json)
        if manifest["distribution_id"] != self.distribution_id:
            raise ValueError(
                "Release distribution differs from its retained source/artifact mapping."
            )
        return manifest["source_build_id"]

    def approval_record(self, support_tuple_id: str) -> dict[str, object]:
        return {
            "kind": "battery-release-matrix-approval",
            "support_tuple_id": support_tuple_id,
            "envelope_id": self.envelope.envelope_id,
            "distribution_id": self.distribution_id,
            "source_build_id": self.source_build_id,
            "distribution_manifest_json": self.distribution_manifest_json,
            "criteria_ids": sorted(item.criterion_id for item in self.criteria),
            "matrix": self.matrix.to_record(),
            "replay_plan": self.replay.plan_record(),
            "required_resource_measurements": sorted(REQUIRED_RESOURCE_MEASUREMENTS),
            "required_contract": battery_release_contract(
                self.envelope.model_id
            ).to_record(),
        }

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "battery-release-bundle",
            "envelope": self.envelope.to_record(),
            "distribution_id": self.distribution_id,
            "distribution_manifest_json": self.distribution_manifest_json,
            **{
                name: [item.to_record() for item in items]
                for name, items in (
                    ("criteria", self.criteria),
                    ("starts", self.starts),
                    ("observations", self.observations),
                    ("evidence", self.evidence),
                    ("references", self.references),
                    ("rights", self.rights),
                    ("resources", self.resources),
                    ("attestations", self.attestations),
                    ("dependencies", self.dependencies),
                )
            },
            "reference_contents_hex": [value.hex() for value in self.reference_contents],
            "replay": self.replay.to_record(),
            "matrix": self.matrix.to_record(),
            "coverage": self.coverage.to_record(),
            "raw_records_json": list(self.raw_records_json),
            "prerequisite_index": None
            if self.prerequisite_index is None
            else self.prerequisite_index.to_record(),
        }

    @property
    def bundle_id(self) -> str:
        return canonical_fingerprint(self.to_record())

    @classmethod
    def from_record(cls, record, /):
        if record.get("kind") != "battery-release-bundle":
            raise ValueError("Invalid battery release bundle.")
        types = {
            "criteria": QualificationCriterion,
            "starts": CampaignStartRecord,
            "observations": CampaignObservationRecord,
            "evidence": QualificationEvidence,
            "references": ReferenceArtifactManifest,
            "rights": ReferenceRightsAttestation,
            "resources": ObservedResourceRecord,
            "attestations": SignedQualificationRecord,
            "dependencies": SupportDependency,
        }
        values = {
            name: tuple(kind.from_record(row) for row in record[name])
            for name, kind in types.items()
        }
        return cls(
            envelope=_envelope_from_record(record["envelope"]),
            distribution_id=record["distribution_id"],
            distribution_manifest_json=record["distribution_manifest_json"],
            reference_contents=tuple(
                bytes.fromhex(value) for value in record["reference_contents_hex"]
            ),
            replay=CleanReplayComparison.from_record(record["replay"]),
            matrix=QualificationMatrix.from_record(record["matrix"]),
            coverage=QualificationCoverageReport.from_record(record["coverage"]),
            raw_records_json=tuple(record["raw_records_json"]),
            prerequisite_index=None
            if record["prerequisite_index"] is None
            else ReleaseIndex.from_record(record["prerequisite_index"]),
            **values,
        )

    def _signature(self, record, role: str) -> SignedQualificationRecord:
        content = canonical_json(
            record if isinstance(record, dict) else record.to_record()
        )
        matches = [
            item
            for item in self.attestations
            if item.role == role and item.content_json == content
        ]
        if len(matches) != 1:
            raise ValueError(f"Exactly one authenticated {role} record is required.")
        return matches[0]

    def verify(
        self,
        candidate: CapabilityProfile,
        policy: AsymmetricReleaseTrustPolicy,
        /,
        *,
        at_time: int,
    ) -> int:
        from ._qualification import (
            CIRCUIT_ECM_SUPPORT,
            DFN_ENTRY_SUPPORT,
            MARQUIS_2019_SPME_SUPPORT,
            NEWMAN_DFN_SUPPORT,
            SERIES_PACK_ENTRY_SUPPORT,
            SERIES_PACK_SUPPORT,
            THERMAL_ECM_SUPPORT,
            validate_battery_candidate_profile,
        )

        if not isinstance(policy, AsymmetricReleaseTrustPolicy):
            raise TypeError("Production release proofs require asymmetric role trust.")
        self.envelope.require_production_bounds()
        support = candidate.support_tuples[0]
        validate_battery_candidate_profile(candidate, support)
        if candidate.released or self.envelope.envelope_id not in {
            item.envelope_id for item in BATTERY_NUMERICAL_ENVELOPES
        }:
            raise ValueError(
                "Release construction requires a known candidate and exact ex-ante envelope."
            )
        attributes = dict(support.attributes)
        if self.envelope.model_id != attributes.get("model_id") or (
            support.support_tuple_id != THERMAL_ECM_SUPPORT.support_tuple_id
            and attributes.get("validity_envelope_id") != self.envelope.envelope_id
        ):
            raise ValueError(
                "Release envelope does not bind the exact numerical support."
            )
        if (
            not isinstance(self.distribution_id, str)
            or not self.distribution_id
            or self.distribution_id != self.distribution_id.strip()
        ):
            raise ValueError("A final immutable distribution identity is mandatory.")
        source_build_id = self.source_build_id
        criteria = _unique(self.criteria, lambda item: item.criterion_id)
        if not criteria:
            raise ValueError("Release criteria cannot be empty.")
        contract = battery_release_contract(self.envelope.model_id)
        contract.validate(
            self.criteria,
            self.matrix,
            self.envelope,
            self.replay,
            support_tuple_id=support.support_tuple_id,
            build_id=source_build_id,
        )
        limits: list[int] = []

        def authenticate(record, role):
            signature = self._signature(record, role)
            key = policy.roles.verify(record, signature, role=role, at_time=at_time)
            limits.append(policy.roles.expiry(signature))
            return key, signature

        approval = self.approval_record(support.support_tuple_id)
        _, approved = authenticate(approval, "criterion-approver")
        for criterion in self.criteria:
            QualificationCriterion.from_record(criterion.to_record())
            if (
                criterion.support_tuple_id != support.support_tuple_id
                or not criterion.is_valid(at_time)
            ):
                raise ValueError("A release criterion is stale or has the wrong support.")
            _, signature = authenticate(criterion, "criterion-approver")
            if signature.issued_at > criterion.issued_at:
                raise ValueError(
                    "Criterion approval must precede its scientific issuance."
                )
            if criterion.valid_until is not None:
                limits.append(criterion.valid_until)
        raw_ids = set()
        raw_by_id = {}
        for raw_json in self.raw_records_json:
            raw = json.loads(raw_json)
            identifier = raw.pop("raw_artifact_id")
            if canonical_fingerprint(raw) != identifier or identifier in raw_ids:
                raise ValueError("Raw output bytes are tampered with or duplicated.")
            raw_ids.add(identifier)
            raw_by_id[identifier] = raw

        def validate_execution(starts, observations, evidence):
            start_by_criterion = _unique(starts, lambda item: item.criterion_id)
            observation_by_criterion = _unique(
                observations, lambda item: item.criterion_id
            )
            if (
                set(start_by_criterion) != set(criteria)
                or set(observation_by_criterion) != set(criteria)
                or len(evidence) != len(criteria)
            ):
                raise ValueError(
                    "Exactly one start, observation and outcome per criterion is required."
                )
            seen = set()
            executors, reviewers = set(), set()
            for item in evidence:
                if len(item.criteria_ids) != 1 or item.criteria_ids[0] in seen:
                    raise ValueError("Evidence must be one outcome per criterion.")
                criterion_id = item.criteria_ids[0]
                if criterion_id not in criteria:
                    raise ValueError("Evidence cites an unapproved criterion.")
                seen.add(criterion_id)
                start, observation = (
                    start_by_criterion[criterion_id],
                    observation_by_criterion[criterion_id],
                )
                validate_qualification_causality(
                    criteria[criterion_id], start, observation, item
                )
                if (
                    approved.issued_at >= start.started_at
                    or not item.passed
                    or not item.is_current(at_time)
                ):
                    raise ValueError(
                        "Release evidence is not precommitted, current and passed."
                    )
                if (
                    item.build_id != source_build_id
                    or item.subject_ids
                    != tuple(sorted((candidate.profile_id, support.support_tuple_id)))
                    or item.topology != self.envelope.model_id
                ):
                    raise ValueError("Evidence build, subject or topology is mismatched.")
                if item.campaign_start_record_ids != (
                    start.start_record_id,
                ) or item.campaign_observation_record_ids != (
                    observation.observation_record_id,
                ):
                    raise ValueError("Evidence borrows another execution boundary.")
                if not set(observation.raw_artifact_ids) <= raw_ids:
                    raise ValueError(
                        "Observation lacks its actual content-addressed raw bytes."
                    )
                criterion = criteria[criterion_id]
                metric_key = f"{criterion.applicability}/{criterion.metric}"
                measurements = [
                    raw_by_id[identifier].get("metrics", {}).get(metric_key)
                    for identifier in observation.raw_artifact_ids
                ]
                measurements = [value for value in measurements if value is not None]
                if len(measurements) != 1 or any(
                    raw_by_id[identifier].get("infrastructure_failures")
                    for identifier in observation.raw_artifact_ids
                ):
                    raise ValueError(
                        "A passed outcome lacks one usable canonical raw metric."
                    )
                measurement = measurements[0]
                value = measurement.get("value")
                if (
                    measurement.get("unavailable_reason") is not None
                    or type(value) not in (int, float)
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        "Unavailable/nonfinite observations cannot support release."
                    )
                passed = {
                    "equal": value == criterion.target,
                    "less-than-or-equal": value <= criterion.target,
                    "greater-than-or-equal": value >= criterion.target,
                }[criterion.comparison]
                if not passed:
                    raise ValueError(
                        "Reviewed outcome contradicts its raw measured threshold."
                    )
                start_key, start_signature = authenticate(start, "executor")
                observation_key, observation_signature = authenticate(
                    observation, "executor"
                )
                evidence_key, evidence_signature = authenticate(item, "executor")
                reviewer_key, review_signature = authenticate(item, "scientific-reviewer")
                if (
                    len({start_key, observation_key, evidence_key}) != 1
                    or item.reviewer_id != reviewer_key
                ):
                    raise ValueError(
                        "Execution/review principal identities are mismatched."
                    )
                if (
                    start_signature.issued_at != start.started_at
                    or observation_signature.issued_at != observation.observed_at
                    or evidence_signature.issued_at != item.issued_at
                    or review_signature.issued_at < item.issued_at
                ):
                    raise ValueError(
                        "Authenticated execution/review causality is invalid."
                    )
                executors.add(start_key)
                reviewers.add(reviewer_key)
                limits.append(item.expires_at)
            return executors, reviewers

        primary_executors, primary_reviewers = validate_execution(
            self.starts, self.observations, self.evidence
        )
        replay_executors, replay_reviewers = validate_execution(
            self.replay.starts, self.replay.observations, self.replay.evidence
        )
        if primary_executors & replay_executors or primary_reviewers & replay_reviewers:
            raise ValueError(
                "Independent replay requires separate executor and reviewer keys."
            )
        replay = self.replay
        if replay.fresh_install is not True or any(
            value is not False
            for value in (
                replay.reused_raw_bytes,
                replay.reused_caches,
                replay.reused_checkpoints,
            )
        ):
            raise ValueError(
                "Replay must use a fresh install without reused raw bytes/caches/checkpoints."
            )
        if not replay.case_ids or len(set(replay.case_ids)) != len(replay.case_ids):
            raise ValueError("Replay must bind finite precommitted case IDs.")
        bounds, errors = (
            dict(replay.observable_limits),
            dict(replay.observed_discrepancies),
        )
        if (
            not bounds
            or set(bounds) != set(errors)
            or len(bounds) != len(replay.observable_limits)
            or len(errors) != len(replay.observed_discrepancies)
        ):
            raise ValueError(
                "Replay must compare each precommitted observable exactly once."
            )
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in (*bounds.values(), *errors.values())
        ) or any(errors[name] > bound for name, bound in bounds.items()):
            raise ValueError("Independent clean replay comparison failed.")
        original_raw = {
            value for item in self.observations for value in item.raw_artifact_ids
        }
        replay_raw = {
            value for item in replay.observations for value in item.raw_artifact_ids
        }
        if original_raw & replay_raw or min(
            item.started_at for item in replay.starts
        ) <= max(item.observed_at for item in self.observations):
            raise ValueError(
                "Independent replay must be a second execution without borrowed bytes."
            )
        baseline_environments = {item.environment_id for item in self.evidence}
        if {
            item.environment_id for item in replay.evidence
        } != baseline_environments and not replay.allowed_environment_differences:
            raise ValueError("Replay environment differences were not precommitted.")
        authenticate(replay, "executor")
        authenticate(replay, "scientific-reviewer")

        QualificationMatrix.from_record(self.matrix.to_record())
        evaluated = self.matrix.evaluate(self.evidence, at_time=at_time)
        QualificationCoverageReport.from_record(self.coverage.to_record())
        if (
            not evaluated.passed
            or self.coverage.matrix_id != self.matrix.matrix_id
            or not self.coverage.passed
            or self.coverage.matched_evidence_ids != evaluated.matched_evidence_ids
            or self.coverage.passed_predicate_ids != evaluated.passed_predicate_ids
        ):
            raise ValueError(
                "Release matrix coverage is incomplete, stale, failed or substituted."
            )
        if set(evaluated.matched_evidence_ids) != {
            item.evidence_id for item in self.evidence
        }:
            raise ValueError("Release matrix must cover every criterion outcome.")
        for _, predicate in self.matrix.predicates:
            constraints = dict(predicate)
            if (
                constraints.get("criterion_id") not in criteria
                or constraints.get("subject_id") != support.support_tuple_id
                or constraints.get("build_id") != source_build_id
            ):
                raise ValueError(
                    "Each release predicate must bind exact criterion/support/build."
                )
        authenticate(self.coverage, "scientific-reviewer")

        reference_count = contract.reference_count
        if (
            len(self.references) != reference_count
            or len(self.reference_contents) != reference_count
            or len(self.rights) != reference_count
        ):
            raise ValueError(
                "Release requires the exact model-specific reference and rights count."
            )
        rights_by_manifest = _unique(self.rights, lambda item: item.manifest_id)
        _unique(self.references, lambda item: item.manifest_id)
        for manifest, content in zip(
            self.references, self.reference_contents, strict=True
        ):
            if (
                len(content) != manifest.size_bytes
                or hashlib.new(manifest.checksum_algorithm, content).hexdigest()
                != manifest.checksum
            ):
                raise ValueError(
                    "Reference content is missing or does not match its immutable manifest."
                )
            rights = rights_by_manifest[manifest.manifest_id]
            signature = self._signature(rights, "scientific-reviewer")
            rights.verify(manifest, signature, policy.roles, at_time=at_time)
            limits.extend((rights.expires_at, policy.roles.expiry(signature)))
        kinds = {item.evidence_kind for item in self.evidence}
        if not {"scientific", "performance", "operational"} <= kinds or (
            reference_count and "reference" not in kinds
        ):
            raise ValueError(
                "Release lacks scientific, absolute-resource, clean-replay or reference criteria."
            )
        resources = _unique(self.resources, lambda item: item.record_id)
        if not resources:
            raise ValueError("Measured absolute resource records are mandatory.")
        cited_resources = {
            value
            for item in (*self.evidence, *replay.evidence)
            for value in item.observed_resource_record_ids
        }
        if set(resources) != cited_resources:
            raise ValueError(
                "Resource records must be exactly cited by authenticated outcomes."
            )
        for resource in resources.values():
            ObservedResourceRecord.from_record(resource.to_record())
            measurements = dict(resource.measurements)
            if (
                resource.build_id != source_build_id
                or resource.subject_id != support.support_tuple_id
                or resource.topology != self.envelope.model_id
                or resource.observed_at > at_time
                or not set(resource.raw_artifact_ids) <= raw_ids
            ):
                raise ValueError(
                    "Resource record scope or causal raw provenance is mismatched."
                )
            if (
                not REQUIRED_RESOURCE_MEASUREMENTS <= measurements.keys()
                or measurements["global-dense-arrays"] != 0
                or measurements["unsuccessful-executions"] != 0
            ):
                raise ValueError(
                    "Required resource measurements are unavailable or a dense production matrix was materialized."
                )
            absolute_limits = dict(self.envelope.resource_limits)
            if not REQUIRED_RESOURCE_MEASUREMENTS <= absolute_limits.keys() or any(
                measurements[name] > absolute_limits[name]
                for name in REQUIRED_RESOURCE_MEASUREMENTS
            ):
                raise ValueError(
                    "Observed workload exceeds an absolute ex-ante resource limit."
                )
            authenticate(resource, "executor")
        required_dependencies = {
            CIRCUIT_ECM_SUPPORT.support_tuple_id: {THERMAL_ECM_SUPPORT.support_tuple_id},
            NEWMAN_DFN_SUPPORT.support_tuple_id: {
                MARQUIS_2019_SPME_SUPPORT.support_tuple_id,
                DFN_ENTRY_SUPPORT.support_tuple_id,
            },
            SERIES_PACK_SUPPORT.support_tuple_id: {
                CIRCUIT_ECM_SUPPORT.support_tuple_id,
                SERIES_PACK_ENTRY_SUPPORT.support_tuple_id,
            },
        }.get(support.support_tuple_id, set())
        if {
            item.support_tuple_id for item in self.dependencies
        } != required_dependencies or len(self.dependencies) != len(
            required_dependencies
        ):
            raise ValueError(
                "Release dependencies must exactly match the model prerequisite DAG."
            )
        if self.dependencies:
            if self.prerequisite_index is None:
                raise ValueError("Authenticated prerequisite staging index is mandatory.")
            for dependency in self.dependencies:
                profile = next(
                    (
                        item
                        for item in self.prerequisite_index.profiles
                        if item.profile_id == dependency.profile_id
                    ),
                    None,
                )
                if profile is None:
                    raise ValueError("Missing exact prerequisite profile.")
                selected = next(
                    (
                        item
                        for item in profile.support_tuples
                        if item.support_tuple_id == dependency.support_tuple_id
                    ),
                    None,
                )
                if selected is None:
                    raise ValueError("Missing exact prerequisite support.")
                require_profile(
                    self.prerequisite_index,
                    profile.profile_id,
                    selected,
                    policy,
                    at_time=at_time,
                )
                for gate in profile.release_evidence:
                    limits.append(gate.expires_at)
                    proofs = [
                        proof
                        for proof in policy.proofs
                        if proof.gate_id == gate.evidence_id
                    ]
                    if (
                        len(proofs) != 1
                        or proofs[0].distribution_id != self.distribution_id
                    ):
                        raise ValueError(
                            "Prerequisite evidence belongs to a different distribution."
                        )
        elif self.prerequisite_index is not None:
            raise ValueError(
                "An independent release cannot cite an unused prerequisite index."
            )
        return min(limits)


@dataclass(frozen=True, slots=True)
class BatteryReleaseRecord:
    candidate: CapabilityProfile
    profile: CapabilityProfile
    bundle: BatteryReleaseBundle
    issued_at: int
    expires_at: int

    @property
    def gate_id(self) -> str:
        return self.profile.release_evidence[0].evidence_id

    @property
    def distribution_id(self) -> str:
        return self.bundle.distribution_id

    @property
    def envelope_id(self) -> str:
        return self.bundle.envelope.envelope_id

    def verify(self, policy: AsymmetricReleaseTrustPolicy, /, *, at_time: int) -> None:
        if not self.issued_at <= at_time < self.expires_at:
            raise ValueError("Typed release is stale or not active.")
        cap = self.bundle.verify(self.candidate, policy, at_time=at_time)
        if (
            self.expires_at > cap
            or self.profile.profile_id
            != _release_profile(
                self.candidate, self.bundle, self.issued_at, self.expires_at
            ).profile_id
        ):
            raise ValueError("Release content, citations or expiry were substituted.")
        CapabilityProfile.from_record(self.profile.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "battery-release-record",
            "candidate": self.candidate.to_record(),
            "profile": self.profile.to_record(),
            "bundle": self.bundle.to_record(),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_record(cls, record, /):
        if record.get("kind") != "battery-release-record":
            raise ValueError("Invalid typed battery release record.")
        return cls(
            CapabilityProfile.from_record(record["candidate"]),
            CapabilityProfile.from_record(record["profile"]),
            BatteryReleaseBundle.from_record(record["bundle"]),
            record["issued_at"],
            record["expires_at"],
        )


def _release_profile(candidate, bundle, issued_at, expires_at):
    from ._qualification import BATTERY_RELEASE_COORDINATES

    name, version = BATTERY_RELEASE_COORDINATES[
        candidate.support_tuples[0].support_tuple_id
    ]
    reviewer = bundle._signature(bundle.coverage, "scientific-reviewer").signature.key_id
    gate = ReleaseGateEvidence(
        f"{name}.scientific",
        passed=True,
        evidence_ids=(
            bundle.bundle_id,
            bundle.coverage.report_id,
            bundle.replay.comparison_id,
            bundle.envelope.envelope_id,
        ),
        reviewer_id=reviewer,
        issued_at=issued_at,
        expires_at=expires_at,
    )
    return CapabilityProfile(
        name,
        "phydrax",
        version,
        candidate.support_tuples,
        dependencies=bundle.dependencies,
        required_gates=(gate.gate,),
        release_evidence=(gate,),
        released=True,
    )


def build_battery_release(
    candidate: CapabilityProfile,
    bundle: BatteryReleaseBundle,
    /,
    *,
    trust_policy: AsymmetricReleaseTrustPolicy,
    at_time: int,
    expires_at: int,
) -> BatteryReleaseRecord:
    if not isinstance(bundle, BatteryReleaseBundle):
        raise TypeError(
            "Release construction requires retained typed inputs, not opaque evidence IDs."
        )
    cap = bundle.verify(candidate, trust_policy, at_time=at_time)
    if type(expires_at) is not int or expires_at <= at_time:
        raise ValueError("Release expiry must follow issuance.")
    expires = min(expires_at, cap)
    if expires <= at_time:
        raise ValueError("A cited proof expires before this release can be issued.")
    profile = _release_profile(candidate, bundle, at_time, expires)
    return BatteryReleaseRecord(candidate, profile, bundle, at_time, expires)
