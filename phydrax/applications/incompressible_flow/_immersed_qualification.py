#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._sharp_measures import QualifiedSharpGeometry
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...qualification._evidence import (
    QualificationCoverageReport,
    QualificationEvidence,
    QualificationMatrix,
)
from ...solver._marker_flow_runtime import HydrodynamicLoadRecord
from ._immersed_profile import (
    IMMERSED_REFERENCE_CASES,
    ImmersedDNSQualificationProfile,
)
from ._immersed_support import ImmersedBodyRegimePlan


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _identifiers(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise TypeError(f"{name} values must be a sequence.")
    normalized = tuple(_identifier(value, name) for value in values)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} values must be non-empty and unique.")
    return tuple(sorted(normalized))


class ImmersedReferenceCaseEvidence(StrictModule, NonTrainableState):
    """A measured reference error tied to one bound regime and raw artifacts."""

    error: Array
    finite: Array
    qualified: Array
    case_id: str = eqx.field(static=True)
    regime_plan_id: str = eqx.field(static=True)
    support_tuple_id: str = eqx.field(static=True)
    subject_ids: tuple[str, ...] = eqx.field(static=True)
    raw_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        case_id: str,
        regime_plan_id: str,
        support_tuple_id: str,
        error: ArrayLike,
        qualified: ArrayLike,
        /,
        *,
        subject_ids: Sequence[str],
        raw_artifact_ids: Sequence[str],
    ):
        case = _identifier(case_id, "case_id")
        if case not in IMMERSED_REFERENCE_CASES:
            raise ValueError(f"Unknown immersed reference case {case!r}.")
        regime = _identifier(regime_plan_id, "regime_plan_id")
        support = _identifier(support_tuple_id, "support_tuple_id")
        subjects = _identifiers(subject_ids, "subject ID")
        artifacts = _identifiers(raw_artifact_ids, "raw artifact ID")
        value = jnp.asarray(error)
        certified = jnp.asarray(qualified, dtype=bool)
        if value.shape != () or certified.shape != ():
            raise ValueError(
                "Reference error and qualification predicate must be scalar."
            )
        finite = jnp.isfinite(value) & (value >= 0.0)
        self.error = value
        self.finite = finite
        self.qualified = certified
        self.case_id = case
        self.regime_plan_id = regime
        self.support_tuple_id = support
        self.subject_ids = subjects
        self.raw_artifact_ids = artifacts
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "immersed-reference-case-evidence",
                "case_id": case,
                "regime_plan_id": regime,
                "support_tuple_id": support,
                "subject_ids": subjects,
                "raw_artifact_ids": artifacts,
            }
        )

    @classmethod
    def manufactured_loads(
        cls,
        regime: ImmersedBodyRegimePlan,
        record: HydrodynamicLoadRecord,
        expected_force: ArrayLike,
        expected_torque: ArrayLike,
        /,
        *,
        raw_artifact_ids: Sequence[str],
    ) -> ImmersedReferenceCaseEvidence:
        if not isinstance(regime, ImmersedBodyRegimePlan):
            raise TypeError("regime must be ImmersedBodyRegimePlan.")
        if not isinstance(record, HydrodynamicLoadRecord):
            raise TypeError("record must be HydrodynamicLoadRecord.")
        force = jnp.asarray(expected_force, dtype=record.force.dtype)
        torque = jnp.asarray(expected_torque, dtype=record.torque.dtype)
        if force.shape != record.force.shape or torque.shape != record.torque.shape:
            raise ValueError("Manufactured load references must match the load record.")
        error = jnp.maximum(
            jnp.max(jnp.abs(record.force - force)),
            jnp.max(jnp.abs(record.torque - torque)),
        )
        provenance = (
            record.marker_set_id == regime.marker_set_id
            and record.geometry_id == regime.geometry_id
            and record.route_id == regime.route_id
            and record.topology_epoch_id == regime.topology_epoch_id
        )
        channels = (
            jnp.all(record.pressure_available | record.viscous_available)
            if regime.regime == "fixed-topology-sharp"
            else jnp.all(record.marker_available)
        )
        qualified = record.successful & channels & provenance
        return cls(
            "manufactured-loads",
            regime.plan_id,
            regime.support_tuple.support_tuple_id,
            error,
            qualified,
            subject_ids=(regime.owner_plan_id, record.record_id),
            raw_artifact_ids=raw_artifact_ids,
        )

    @classmethod
    def sharp_certificate(
        cls,
        regime: ImmersedBodyRegimePlan,
        geometry: QualifiedSharpGeometry,
        /,
        *,
        raw_artifact_ids: Sequence[str],
    ) -> ImmersedReferenceCaseEvidence:
        if not isinstance(regime, ImmersedBodyRegimePlan):
            raise TypeError("regime must be ImmersedBodyRegimePlan.")
        if regime.regime != "fixed-topology-sharp":
            raise ValueError("Sharp certificate evidence requires the sharp regime.")
        if not isinstance(geometry, QualifiedSharpGeometry):
            raise TypeError("geometry must be QualifiedSharpGeometry.")
        widths = (
            geometry.evidence.cell_bound_width,
            *geometry.evidence.face_bound_width,
        )
        error = jnp.max(jnp.stack(tuple(jnp.max(value) for value in widths)))
        qualified = (
            geometry.accepted
            & geometry.qualified
            & geometry.evidence.topology_resolved
            & geometry.evidence.bounds_valid
            & (geometry.realization_id == regime.geometry_id)
        )
        return cls(
            "sharp-certificate",
            regime.plan_id,
            regime.support_tuple.support_tuple_id,
            error,
            qualified,
            subject_ids=(regime.owner_plan_id, geometry.realization_id),
            raw_artifact_ids=raw_artifact_ids,
        )


class ImmersedReferenceCampaignResult(StrictModule):
    errors: Array
    tolerances: Array
    passed: Array
    finite: Array
    qualified: Array
    successful: Array
    case_ids: tuple[str, ...] = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)


_CASES_BY_REGIME: dict[str, tuple[str, ...]] = {
    "prescribed-marker": (
        "manufactured-loads",
        "fixed-cylinder",
        "moving-cylinder",
    ),
    "free-rigid-marker": (
        "manufactured-loads",
        "fixed-sphere",
        "moving-sphere",
        "added-mass",
        "free-settling",
    ),
    "fixed-topology-sharp": (
        "manufactured-loads",
        "fixed-cylinder",
        "fixed-sphere",
        "sharp-certificate",
    ),
    "deformable-contact": (
        "manufactured-loads",
        "flexible-contact-state",
    ),
    "lbm-body": (
        "manufactured-loads",
        "fixed-cylinder",
        "moving-cylinder",
        "fixed-sphere",
        "moving-sphere",
    ),
    "resolved-cfd-dem": (
        "manufactured-loads",
        "free-settling",
        "flexible-contact-state",
    ),
}


class ImmersedReferenceCampaignPlan(StrictModule, NonTrainableState):
    """Qualify measured references from bound owners without executing another solve."""

    profile: ImmersedDNSQualificationProfile
    regimes: tuple[ImmersedBodyRegimePlan, ...]
    qualification_matrix: QualificationMatrix
    required_cases: tuple[str, ...] = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)

    def __init__(
        self,
        profile: ImmersedDNSQualificationProfile,
        regimes: Sequence[ImmersedBodyRegimePlan],
        /,
    ):
        if not isinstance(profile, ImmersedDNSQualificationProfile):
            raise TypeError("profile must be ImmersedDNSQualificationProfile.")
        regimes_ = tuple(regimes)
        if not regimes_ or any(
            not isinstance(regime, ImmersedBodyRegimePlan) for regime in regimes_
        ):
            raise TypeError("regimes must contain bound ImmersedBodyRegimePlan values.")
        plan_ids = tuple(regime.plan_id for regime in regimes_)
        if len(set(plan_ids)) != len(plan_ids):
            raise ValueError("Reference campaign contains duplicate regime plans.")
        if any(not profile.supports(regime.support_tuple) for regime in regimes_):
            raise ValueError("A regime is outside the immersed qualification profile.")
        required = tuple(
            case
            for case in profile.required_reference_cases
            if any(case in _CASES_BY_REGIME[regime.regime] for regime in regimes_)
        )
        matrix = QualificationMatrix(
            {
                f"reference.{case}": {
                    "evidence_kind": "reference",
                    "criterion_id": case,
                }
                for case in required
            }
        )
        self.profile = profile
        self.regimes = regimes_
        self.required_cases = required
        self.qualification_matrix = matrix
        self.campaign_id = canonical_fingerprint(
            {
                "kind": "immersed-reference-campaign-plan",
                "profile": profile.profile_id,
                "regimes": plan_ids,
                "required_cases": required,
                "qualification_matrix": matrix.matrix_id,
            }
        )

    def evaluate(
        self,
        evidence: Sequence[ImmersedReferenceCaseEvidence],
        /,
    ) -> ImmersedReferenceCampaignResult:
        values = tuple(evidence)
        if any(not isinstance(item, ImmersedReferenceCaseEvidence) for item in values):
            raise TypeError("evidence must contain ImmersedReferenceCaseEvidence values.")
        by_case = {item.case_id: item for item in values}
        if len(by_case) != len(values):
            raise ValueError("Reference campaign contains duplicate case evidence.")
        if set(by_case) != set(self.required_cases):
            missing = tuple(case for case in self.required_cases if case not in by_case)
            extra = tuple(case for case in by_case if case not in self.required_cases)
            raise ValueError(
                f"Reference campaign evidence mismatch; missing={missing}, extra={extra}."
            )
        regimes = {regime.plan_id: regime for regime in self.regimes}
        ordered = tuple(by_case[case] for case in self.required_cases)
        for item in ordered:
            if item.regime_plan_id not in regimes:
                raise ValueError("Reference evidence names an unbound regime plan.")
            regime = regimes[item.regime_plan_id]
            if (
                item.support_tuple_id != regime.support_tuple.support_tuple_id
                or item.case_id not in _CASES_BY_REGIME[regime.regime]
            ):
                raise ValueError("Reference evidence and regime support do not match.")
        errors = jnp.stack(tuple(item.error for item in ordered))
        tolerances = jnp.asarray(
            tuple(self.profile.tolerance_for(case) for case in self.required_cases),
            dtype=errors.dtype,
        )
        finite = jnp.stack(tuple(item.finite for item in ordered))
        qualified = jnp.stack(tuple(item.qualified for item in ordered))
        passed = finite & qualified & (errors <= tolerances)
        return ImmersedReferenceCampaignResult(
            errors,
            tolerances,
            passed,
            jnp.all(finite),
            jnp.all(qualified),
            jnp.all(passed),
            self.required_cases,
            tuple(item.evidence_id for item in ordered),
            self.campaign_id,
            self.profile.profile_id,
        )

    def evaluate_governed(
        self,
        evidence: Sequence[QualificationEvidence],
        /,
        *,
        at_time: int,
    ) -> QualificationCoverageReport:
        """Evaluate governed, time-bounded artifacts for this exact campaign."""

        return self.qualification_matrix.evaluate(evidence, at_time=at_time)


__all__ = [
    "ImmersedReferenceCampaignPlan",
    "ImmersedReferenceCampaignResult",
    "ImmersedReferenceCaseEvidence",
]
