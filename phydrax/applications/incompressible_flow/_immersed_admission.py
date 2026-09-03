#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntFlag

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume._distributed_marker_transfer import (
    DistributedMarkerTransferDiagnostics,
)
from ...qualification._evidence import QualificationCoverageReport
from ...solver._marker_flow_runtime import HydrodynamicLoadRecord
from ._immersed_profile import ImmersedDNSQualificationProfile
from ._immersed_support import (
    ImmersedBodyRegimePlan,
    ImmersedDerivativeMode,
    ImmersedNearGapDecision,
)


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


class ImmersedRuntimeAdmissionFailure(IntFlag):
    NONE = 0
    QUALIFICATION_FAILED = 1
    MARKER_RANK_FAILED = 2
    MARKER_CONDITION_FAILED = 4
    RESOURCE_BUDGET_EXCEEDED = 8
    DERIVATIVE_SCOPE_UNSUPPORTED = 16
    OWNER_MISMATCH = 32
    TOPOLOGY_EPOCH_MISMATCH = 64
    MOTION_EPOCH_MISMATCH = 128
    GEOMETRY_EPOCH_MISMATCH = 256
    ROUTE_MISMATCH = 512
    SUPPORT_TRUNCATED = 1024
    DISTRIBUTED_REDUCTION_FAILED = 2048
    GAP_REGIME_INADMISSIBLE = 4096
    TOPOLOGY_CHANGE = 8192
    SHARP_CERTIFICATE_FAILED = 16384
    LOAD_PROVENANCE_FAILED = 32768


class ImmersedRuntimePreflightEvidence(StrictModule, NonTrainableState):
    marker_numerical_rank: Array
    marker_condition: Array
    observed_resource_bytes: int = eqx.field(static=True)
    rank_certified: Array
    campaign_qualified: Array
    owner_plan_id: str = eqx.field(static=True)
    support_tuple_id: str = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        owner_plan_id: str,
        support_tuple_id: str,
        marker_numerical_rank: ArrayLike,
        marker_condition: ArrayLike,
        observed_resource_bytes: int,
        rank_certified: ArrayLike,
        campaign_qualified: ArrayLike,
        /,
        *,
        evidence_ids: Sequence[str],
    ):
        owner = _identifier(owner_plan_id, "owner_plan_id")
        support = _identifier(support_tuple_id, "support_tuple_id")
        rank = jnp.asarray(marker_numerical_rank, dtype=jnp.int32)
        condition = jnp.asarray(marker_condition)
        certified = jnp.asarray(rank_certified, dtype=bool)
        qualified = jnp.asarray(campaign_qualified, dtype=bool)
        if any(value.shape != () for value in (rank, condition, certified, qualified)):
            raise ValueError("Runtime preflight predicates must be scalar.")
        resources = int(observed_resource_bytes)
        if resources < 0:
            raise ValueError("observed_resource_bytes must be non-negative.")
        ids = _identifiers(evidence_ids, "preflight evidence ID")
        self.marker_numerical_rank = rank
        self.marker_condition = condition
        self.observed_resource_bytes = resources
        self.rank_certified = certified
        self.campaign_qualified = qualified
        self.owner_plan_id = owner
        self.support_tuple_id = support
        self.evidence_ids = ids
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "immersed-runtime-preflight-evidence",
                "owner_plan_id": owner,
                "support_tuple_id": support,
                "observed_resource_bytes": resources,
                "evidence_ids": ids,
            }
        )

    @classmethod
    def from_coverage(
        cls,
        regime: ImmersedBodyRegimePlan,
        coverage: QualificationCoverageReport,
        marker_numerical_rank: ArrayLike,
        marker_condition: ArrayLike,
        observed_resource_bytes: int,
        rank_certified: ArrayLike,
        /,
        *,
        additional_evidence_ids: Sequence[str] = (),
    ) -> ImmersedRuntimePreflightEvidence:
        """Bind preflight to a governed campaign report without another solve."""

        if not isinstance(regime, ImmersedBodyRegimePlan):
            raise TypeError("regime must be ImmersedBodyRegimePlan.")
        if not isinstance(coverage, QualificationCoverageReport):
            raise TypeError("coverage must be QualificationCoverageReport.")
        return cls(
            regime.owner_plan_id,
            regime.support_tuple.support_tuple_id,
            marker_numerical_rank,
            marker_condition,
            observed_resource_bytes,
            rank_certified,
            coverage.outcome == "passed",
            evidence_ids=(coverage.report_id, *additional_evidence_ids),
        )


class ImmersedRuntimeEvidence(StrictModule, NonTrainableState):
    support_truncated: Array
    topology_changed: Array
    geometry_refresh_required: Array
    sharp_certificate_valid: Array
    differentiation_routes_frozen: Array
    gap: Array | None
    distributed: DistributedMarkerTransferDiagnostics | None
    load_record: HydrodynamicLoadRecord | None
    owner_plan_id: str = eqx.field(static=True)
    support_tuple_id: str = eqx.field(static=True)
    marker_set_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    topology_epoch_id: str = eqx.field(static=True)
    motion_epoch_id: str = eqx.field(static=True)
    geometry_epoch: int = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        owner_plan_id: str,
        support_tuple_id: str,
        marker_set_id: str,
        geometry_id: str,
        route_id: str,
        topology_epoch_id: str,
        motion_epoch_id: str,
        geometry_epoch: int,
        support_truncated: ArrayLike,
        topology_changed: ArrayLike,
        geometry_refresh_required: ArrayLike,
        sharp_certificate_valid: ArrayLike,
        differentiation_routes_frozen: ArrayLike,
        evidence_ids: Sequence[str],
        gap: ArrayLike | None = None,
        distributed: DistributedMarkerTransferDiagnostics | None = None,
        load_record: HydrodynamicLoadRecord | None = None,
    ):
        predicates = tuple(
            jnp.asarray(value, dtype=bool)
            for value in (
                topology_changed,
                geometry_refresh_required,
                sharp_certificate_valid,
                differentiation_routes_frozen,
            )
        )
        if any(value.shape != () for value in predicates):
            raise ValueError("Runtime epoch/certificate predicates must be scalar.")
        truncated = jnp.asarray(support_truncated, dtype=bool)
        gap_ = None if gap is None else jnp.asarray(gap)
        if gap_ is not None and gap_.size == 0:
            raise ValueError("Runtime gap evidence cannot be empty.")
        if distributed is not None and not isinstance(
            distributed, DistributedMarkerTransferDiagnostics
        ):
            raise TypeError(
                "distributed must be DistributedMarkerTransferDiagnostics or None."
            )
        if load_record is not None and not isinstance(
            load_record, HydrodynamicLoadRecord
        ):
            raise TypeError("load_record must be HydrodynamicLoadRecord or None.")
        epoch = int(geometry_epoch)
        if epoch < 0:
            raise ValueError("geometry_epoch must be non-negative.")
        self.support_truncated = truncated
        (
            self.topology_changed,
            self.geometry_refresh_required,
            self.sharp_certificate_valid,
            self.differentiation_routes_frozen,
        ) = predicates
        self.gap = gap_
        self.distributed = distributed
        self.load_record = load_record
        self.owner_plan_id = _identifier(owner_plan_id, "owner_plan_id")
        self.support_tuple_id = _identifier(support_tuple_id, "support_tuple_id")
        self.marker_set_id = _identifier(marker_set_id, "marker_set_id")
        self.geometry_id = _identifier(geometry_id, "geometry_id")
        self.route_id = _identifier(route_id, "route_id")
        self.topology_epoch_id = _identifier(topology_epoch_id, "topology_epoch_id")
        self.motion_epoch_id = _identifier(motion_epoch_id, "motion_epoch_id")
        self.geometry_epoch = epoch
        self.evidence_ids = _identifiers(evidence_ids, "runtime evidence ID")
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "immersed-runtime-evidence",
                "owner_plan_id": self.owner_plan_id,
                "support_tuple_id": self.support_tuple_id,
                "marker_set_id": self.marker_set_id,
                "geometry_id": self.geometry_id,
                "route_id": self.route_id,
                "topology_epoch_id": self.topology_epoch_id,
                "motion_epoch_id": self.motion_epoch_id,
                "geometry_epoch": epoch,
                "distributed": None if distributed is None else distributed.plan_id,
                "load_record": None if load_record is None else load_record.record_id,
                "evidence_ids": self.evidence_ids,
            }
        )


class PreparedImmersedRuntimeAdmission(StrictModule, NonTrainableState):
    plan: ImmersedRuntimeAdmissionPlan
    preflight: ImmersedRuntimePreflightEvidence
    rank_admitted: Array
    condition_admitted: Array
    resource_admitted: Array
    qualification_admitted: Array
    derivative_admitted: Array
    owner_admitted: bool = eqx.field(static=True)
    status: Array
    prepared: Array
    prepared_id: str = eqx.field(static=True)

    def admit(
        self, evidence: ImmersedRuntimeEvidence, /
    ) -> ImmersedRuntimeAdmissionResult:
        return self.plan._admit(self, evidence)


class ImmersedRuntimeAdmissionResult(StrictModule):
    preflight: PreparedImmersedRuntimeAdmission
    runtime_evidence: ImmersedRuntimeEvidence
    gap_decision: ImmersedNearGapDecision | None
    topology_epoch_admitted: Array
    motion_epoch_admitted: Array
    geometry_epoch_admitted: Array
    route_admitted: Array
    support_admitted: Array
    distributed_admitted: Array
    gap_admitted: Array
    sharp_admitted: Array
    load_admitted: Array
    status: Array
    admitted: Array
    plan_id: str = eqx.field(static=True)


class ImmersedRuntimeAdmissionPlan(StrictModule, NonTrainableState):
    """Two-phase, fail-closed admission over already-produced owner evidence."""

    profile: ImmersedDNSQualificationProfile
    regime: ImmersedBodyRegimePlan
    derivative_mode: ImmersedDerivativeMode = eqx.field(static=True)
    maximum_resource_bytes: int = eqx.field(static=True)
    marker_condition_limit: float = eqx.field(static=True)
    distributed_tolerance: float = eqx.field(static=True)
    require_load_record: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        profile: ImmersedDNSQualificationProfile,
        regime: ImmersedBodyRegimePlan,
        /,
        *,
        maximum_resource_bytes: int,
        derivative_mode: ImmersedDerivativeMode = "none",
        marker_condition_limit: float | None = None,
        distributed_tolerance: float = 1.0e-9,
        require_load_record: bool = False,
    ):
        if not isinstance(profile, ImmersedDNSQualificationProfile):
            raise TypeError("profile must be ImmersedDNSQualificationProfile.")
        if not isinstance(regime, ImmersedBodyRegimePlan):
            raise TypeError("regime must be ImmersedBodyRegimePlan.")
        if not profile.supports(regime.support_tuple):
            raise ValueError("regime is outside the immersed qualification profile.")
        resources = int(maximum_resource_bytes)
        condition = (
            profile.marker_condition_limit
            if marker_condition_limit is None
            else float(marker_condition_limit)
        )
        reduction_tolerance = float(distributed_tolerance)
        if resources <= 0:
            raise ValueError("maximum_resource_bytes must be positive.")
        if not np.isfinite(condition) or condition <= 1.0:
            raise ValueError(
                "marker_condition_limit must be finite and greater than one."
            )
        if not np.isfinite(reduction_tolerance) or reduction_tolerance <= 0.0:
            raise ValueError("distributed_tolerance must be finite and positive.")
        if derivative_mode not in ("none", "jvp", "vjp"):
            raise ValueError("derivative_mode must be 'none', 'jvp', or 'vjp'.")
        self.profile = profile
        self.regime = regime
        self.derivative_mode = derivative_mode
        self.maximum_resource_bytes = resources
        self.marker_condition_limit = condition
        self.distributed_tolerance = reduction_tolerance
        self.require_load_record = bool(require_load_record)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "immersed-runtime-admission-plan",
                "profile": profile.profile_id,
                "regime": regime.plan_id,
                "maximum_resource_bytes": resources,
                "marker_condition_limit": condition,
                "distributed_tolerance": reduction_tolerance,
                "derivative_mode": derivative_mode,
                "require_load_record": bool(require_load_record),
            }
        )

    def _derivative_supported(self) -> bool:
        if self.derivative_mode == "none":
            return True
        if not self.regime.fixed_topology or self.regime.distributed_transfer is not None:
            return False
        if self.regime.regime in (
            "prescribed-marker",
            "free-rigid-marker",
            "fixed-topology-sharp",
        ):
            return not self.regime.contact_capable
        if self.regime.regime == "deformable-contact":
            return not self.regime.contact_capable
        return False

    def prepare(
        self, evidence: ImmersedRuntimePreflightEvidence, /
    ) -> PreparedImmersedRuntimeAdmission:
        if not isinstance(evidence, ImmersedRuntimePreflightEvidence):
            raise TypeError("evidence must be ImmersedRuntimePreflightEvidence.")
        owner = (
            evidence.owner_plan_id == self.regime.owner_plan_id
            and evidence.support_tuple_id == self.regime.support_tuple.support_tuple_id
        )
        constraint_count = self.regime.marker_constraint_count
        if constraint_count == 0:
            rank = jnp.asarray(True)
            condition = jnp.asarray(True)
        else:
            rank = evidence.rank_certified & (
                evidence.marker_numerical_rank == constraint_count
            )
            condition = (
                jnp.isfinite(evidence.marker_condition)
                & (evidence.marker_condition >= 1.0)
                & (evidence.marker_condition <= self.marker_condition_limit)
            )
        required_resources = max(
            self.regime.estimated_marker_resource_bytes,
            evidence.observed_resource_bytes,
        )
        resource = jnp.asarray(required_resources <= self.maximum_resource_bytes)
        qualification = evidence.campaign_qualified
        derivative = jnp.asarray(self._derivative_supported())
        status = jnp.asarray(0, dtype=jnp.int32)
        status = status | jnp.where(
            qualification,
            0,
            int(ImmersedRuntimeAdmissionFailure.QUALIFICATION_FAILED),
        ).astype(jnp.int32)
        status = status | jnp.where(
            rank, 0, int(ImmersedRuntimeAdmissionFailure.MARKER_RANK_FAILED)
        ).astype(jnp.int32)
        status = status | jnp.where(
            condition,
            0,
            int(ImmersedRuntimeAdmissionFailure.MARKER_CONDITION_FAILED),
        ).astype(jnp.int32)
        status = status | jnp.where(
            resource,
            0,
            int(ImmersedRuntimeAdmissionFailure.RESOURCE_BUDGET_EXCEEDED),
        ).astype(jnp.int32)
        status = status | jnp.asarray(
            0 if owner else int(ImmersedRuntimeAdmissionFailure.OWNER_MISMATCH),
            dtype=jnp.int32,
        )
        status = status | jnp.where(
            derivative,
            0,
            int(ImmersedRuntimeAdmissionFailure.DERIVATIVE_SCOPE_UNSUPPORTED),
        ).astype(jnp.int32)
        prepared = status == 0
        return PreparedImmersedRuntimeAdmission(
            self,
            evidence,
            rank,
            condition,
            resource,
            qualification,
            derivative,
            owner,
            status,
            prepared,
            canonical_fingerprint(
                {
                    "kind": "prepared-immersed-runtime-admission",
                    "plan": self.plan_id,
                    "preflight": evidence.evidence_id,
                }
            ),
        )

    def admit(
        self,
        preflight: ImmersedRuntimePreflightEvidence,
        runtime: ImmersedRuntimeEvidence,
        /,
    ) -> ImmersedRuntimeAdmissionResult:
        return self.prepare(preflight).admit(runtime)

    def _admit(
        self,
        prepared: PreparedImmersedRuntimeAdmission,
        evidence: ImmersedRuntimeEvidence,
        /,
    ) -> ImmersedRuntimeAdmissionResult:
        if not isinstance(evidence, ImmersedRuntimeEvidence):
            raise TypeError("evidence must be ImmersedRuntimeEvidence.")
        if prepared.plan.plan_id != self.plan_id:
            raise ValueError("Prepared admission belongs to another plan.")
        regime = self.regime
        owner_match = (
            evidence.owner_plan_id == regime.owner_plan_id
            and evidence.support_tuple_id == regime.support_tuple.support_tuple_id
            and evidence.marker_set_id == regime.marker_set_id
            and evidence.geometry_id == regime.geometry_id
        )
        topology_epoch = jnp.asarray(
            evidence.topology_epoch_id == regime.topology_epoch_id
        )
        motion_epoch = jnp.asarray(
            (not regime.moving) or evidence.motion_epoch_id == regime.motion_epoch_id
        )
        geometry_epoch = jnp.asarray(evidence.geometry_epoch == regime.geometry_epoch)
        route = jnp.asarray(evidence.route_id == regime.route_id)
        support = ~jnp.any(evidence.support_truncated)
        topology_change = evidence.topology_changed | evidence.geometry_refresh_required

        if regime.distributed_transfer is None:
            distributed = jnp.asarray(evidence.distributed is None)
        elif evidence.distributed is None:
            distributed = jnp.asarray(False)
        else:
            diagnostics = evidence.distributed
            distributed = (diagnostics.plan_id == regime.distributed_transfer.plan_id) & (
                diagnostics.successful
                & diagnostics.finite
                & (diagnostics.duplicated_owner_count == 0)
                & (
                    jnp.max(jnp.abs(diagnostics.global_force_residual))
                    <= self.distributed_tolerance
                )
                & (
                    jnp.abs(diagnostics.global_work_residual)
                    <= self.distributed_tolerance
                )
            )

        gap_decision = None if evidence.gap is None else regime.classify_gap(evidence.gap)
        needs_gap = regime.contact_capable or regime.lubrication is not None
        gap = (
            jnp.asarray(not needs_gap)
            if gap_decision is None
            else gap_decision.admissible
        )
        sharp = (
            jnp.asarray(regime.regime != "fixed-topology-sharp")
            | evidence.sharp_certificate_valid
        )
        derivative_routes = (
            jnp.asarray(self.derivative_mode == "none")
            | evidence.differentiation_routes_frozen
        )

        if evidence.load_record is None:
            load = jnp.asarray(not self.require_load_record)
        else:
            record = evidence.load_record
            provenance = (
                record.marker_set_id == regime.marker_set_id
                and record.geometry_id == regime.geometry_id
                and record.route_id == regime.route_id
                and record.topology_epoch_id == regime.topology_epoch_id
            )
            if regime.regime == "fixed-topology-sharp":
                channels = jnp.all(record.pressure_available | record.viscous_available)
            else:
                channels = jnp.all(record.marker_available)
            if regime.lubrication is not None:
                channels = channels & jnp.all(record.lubrication_available)
            if regime.contact_capable:
                channels = channels & jnp.all(record.contact_available)
            load = record.successful & provenance & channels

        status = prepared.status
        status = status | jnp.asarray(
            0 if owner_match else int(ImmersedRuntimeAdmissionFailure.OWNER_MISMATCH),
            dtype=jnp.int32,
        )
        status = status | jnp.where(
            topology_epoch,
            0,
            int(ImmersedRuntimeAdmissionFailure.TOPOLOGY_EPOCH_MISMATCH),
        ).astype(jnp.int32)
        status = status | jnp.where(
            motion_epoch,
            0,
            int(ImmersedRuntimeAdmissionFailure.MOTION_EPOCH_MISMATCH),
        ).astype(jnp.int32)
        status = status | jnp.where(
            geometry_epoch,
            0,
            int(ImmersedRuntimeAdmissionFailure.GEOMETRY_EPOCH_MISMATCH),
        ).astype(jnp.int32)
        status = status | jnp.where(
            route, 0, int(ImmersedRuntimeAdmissionFailure.ROUTE_MISMATCH)
        ).astype(jnp.int32)
        status = status | jnp.where(
            support, 0, int(ImmersedRuntimeAdmissionFailure.SUPPORT_TRUNCATED)
        ).astype(jnp.int32)
        status = status | jnp.where(
            distributed,
            0,
            int(ImmersedRuntimeAdmissionFailure.DISTRIBUTED_REDUCTION_FAILED),
        ).astype(jnp.int32)
        status = status | jnp.where(
            gap, 0, int(ImmersedRuntimeAdmissionFailure.GAP_REGIME_INADMISSIBLE)
        ).astype(jnp.int32)
        status = status | jnp.where(
            ~topology_change,
            0,
            int(ImmersedRuntimeAdmissionFailure.TOPOLOGY_CHANGE),
        ).astype(jnp.int32)
        status = status | jnp.where(
            sharp,
            0,
            int(ImmersedRuntimeAdmissionFailure.SHARP_CERTIFICATE_FAILED),
        ).astype(jnp.int32)
        status = status | jnp.where(
            derivative_routes,
            0,
            int(ImmersedRuntimeAdmissionFailure.DERIVATIVE_SCOPE_UNSUPPORTED),
        ).astype(jnp.int32)
        status = status | jnp.where(
            load,
            0,
            int(ImmersedRuntimeAdmissionFailure.LOAD_PROVENANCE_FAILED),
        ).astype(jnp.int32)
        admitted = status == 0
        return ImmersedRuntimeAdmissionResult(
            prepared,
            evidence,
            gap_decision,
            topology_epoch,
            motion_epoch,
            geometry_epoch,
            route,
            support,
            distributed,
            gap,
            sharp,
            load,
            status,
            admitted,
            self.plan_id,
        )


__all__ = [
    "ImmersedRuntimeAdmissionFailure",
    "ImmersedRuntimeAdmissionPlan",
    "ImmersedRuntimeAdmissionResult",
    "ImmersedRuntimeEvidence",
    "ImmersedRuntimePreflightEvidence",
    "PreparedImmersedRuntimeAdmission",
]
