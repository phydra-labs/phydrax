#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....solver._mac_ale import MACALEGeometryPlan, MACALEResult, MACALEStageGeometry


ALEGapProvider = Callable[[MACALEStageGeometry, Any], ArrayLike]
ALESweptGapProvider = Callable[
    [MACALEStageGeometry, MACALEStageGeometry, Array, Array, Any],
    ArrayLike,
]
FaceVelocity = tuple[Array, ...]


class ALETransitionStatus(IntEnum):
    SUCCESS = 0
    INVALID_MESH = 1
    GCL_FAILURE = 2
    MINIMUM_GAP_FAILURE = 3
    FLOW_SOLVE_FAILURE = 4
    NONFINITE = 5


class ALEMinimumGapRoute(StrictModule, NonTrainableState):
    """Noncontact clearance qualification for a conforming ALE valve mesh.

    This route rejects a window before geometric intersection. It is intentionally
    distinct from the true-contact route in ``_leaflets`` and does not evaluate or
    approximate contact forces.
    """

    gap_provider: ALEGapProvider = eqx.field(static=True)
    swept_gap_provider: ALESweptGapProvider = eqx.field(static=True)
    minimum_gap: float = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        gap_provider: ALEGapProvider,
        swept_gap_provider: ALESweptGapProvider,
        /,
        *,
        minimum_gap: float,
        route_id: str = "cardiovascular-ale-minimum-gap",
    ):
        if not callable(gap_provider) or not callable(swept_gap_provider):
            raise TypeError("gap_provider and swept_gap_provider must be callable.")
        gap = float(minimum_gap)
        identifier = str(route_id)
        if not np.isfinite(gap) or gap < 0.0:
            raise ValueError("minimum_gap must be finite and non-negative.")
        if not identifier:
            raise ValueError("route_id must be non-empty.")
        self.gap_provider = gap_provider
        self.swept_gap_provider = swept_gap_provider
        self.minimum_gap = gap
        self.route_id = identifier


class ALEGapEvidence(StrictModule):
    gap: Array
    minimum_gap: Array
    required_gap: Array
    separated: Array
    finite: Array
    successful: Array
    swept_certified: Array
    route_id: str = eqx.field(static=True)


class ALEMeshEvidence(StrictModule):
    minimum_cell_volume: Array
    minimum_face_measure: Array
    minimum_velocity_dual_measure: Array
    minimum_oriented_dual_distance: Array
    maximum_gcl_residual: Array
    map_velocity_residual: Array
    boundary_kinematic_residual: Array
    free_stream_residual: Array
    mapped_adjoint_residual: Array
    gap: ALEGapEvidence
    admissible: Array
    gcl_certified: Array
    finite: Array
    successful: Array
    topology_id: str = eqx.field(static=True)
    motion_plan_id: str = eqx.field(static=True)


class CardiovascularALEState(StrictModule):
    """Accepted conforming ALE flow state on one immutable connectivity epoch."""

    velocity: FaceVelocity
    pressure: Array


class CardiovascularALETransition(StrictModule):
    """Candidate/evidence/accepted transaction for one ALE time window."""

    candidate_state: CardiovascularALEState
    accepted_state: CardiovascularALEState
    flow: MACALEResult
    geometry: MACALEStageGeometry
    evidence: ALEMeshEvidence
    successful: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class CardiovascularALEPlan(StrictModule, NonTrainableState):
    """Plan conforming noncontact cardiovascular ALE over native MAC GCL motion."""

    motion: MACALEGeometryPlan
    gap_route: ALEMinimumGapRoute
    minimum_cell_volume: float = eqx.field(static=True)
    minimum_face_measure: float = eqx.field(static=True)
    minimum_dual_measure: float = eqx.field(static=True)
    gcl_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        motion: MACALEGeometryPlan,
        gap_route: ALEMinimumGapRoute,
        /,
        *,
        minimum_cell_volume: float = 1.0e-12,
        minimum_face_measure: float = 1.0e-12,
        minimum_dual_measure: float = 1.0e-12,
        gcl_tolerance: float = 1.0e-9,
    ):
        if not isinstance(motion, MACALEGeometryPlan):
            raise TypeError("motion must be MACALEGeometryPlan.")
        if not isinstance(gap_route, ALEMinimumGapRoute):
            raise TypeError(
                "gap_route must be ALEMinimumGapRoute; true contact belongs to the "
                "leaflet contact workflow."
            )
        measures = (
            float(minimum_cell_volume),
            float(minimum_face_measure),
            float(minimum_dual_measure),
        )
        tolerance = float(gcl_tolerance)
        if any(not np.isfinite(value) or value <= 0.0 for value in measures):
            raise ValueError("ALE admissibility measures must be finite and positive.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("gcl_tolerance must be finite and positive.")
        self.motion = motion
        self.gap_route = gap_route
        self.minimum_cell_volume = measures[0]
        self.minimum_face_measure = measures[1]
        self.minimum_dual_measure = measures[2]
        self.gcl_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-conforming-noncontact-ale",
                "motion": motion.plan_id,
                "gap_route": gap_route.route_id,
                "minimum_cell_volume": measures[0],
                "minimum_face_measure": measures[1],
                "minimum_dual_measure": measures[2],
                "gcl_tolerance": tolerance,
                "topology_mutation": False,
            }
        )

    def prepare(self, /) -> PreparedCardiovascularALE:
        return PreparedCardiovascularALE(self)


class PreparedCardiovascularALE(StrictModule, NonTrainableState):
    """Prepared fixed-connectivity ALE evaluator with fail-closed acceptance."""

    plan: CardiovascularALEPlan
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: CardiovascularALEPlan, /):
        if not isinstance(plan, CardiovascularALEPlan):
            raise TypeError("plan must be CardiovascularALEPlan.")
        self.plan = plan
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-cardio-ale", "plan": plan.plan_id}
        )

    def _gap_evidence(
        self,
        gap: ArrayLike,
        dtype: Any,
        /,
        *,
        swept_certified: bool,
    ) -> ALEGapEvidence:
        gap_ = jnp.asarray(gap, dtype=dtype)
        if gap_.size == 0:
            raise ValueError("ALE gap provider must return at least one gap value.")
        minimum_gap = jnp.min(gap_)
        finite_gap = jnp.all(jnp.isfinite(gap_))
        separated = minimum_gap >= self.plan.gap_route.minimum_gap
        swept = jnp.asarray(swept_certified)
        return ALEGapEvidence(
            gap_,
            minimum_gap,
            jnp.asarray(self.plan.gap_route.minimum_gap, dtype=dtype),
            separated,
            finite_gap,
            separated & finite_gap,
            swept,
            self.plan.gap_route.route_id,
        )

    def evidence(
        self,
        geometry: MACALEStageGeometry,
        args: Any = None,
        /,
        *,
        gap_evidence: ALEGapEvidence | None = None,
    ) -> ALEMeshEvidence:
        if not isinstance(geometry, MACALEStageGeometry):
            raise TypeError("geometry must be MACALEStageGeometry.")
        gap_evidence_ = (
            self._gap_evidence(
                self.plan.gap_route.gap_provider(geometry, args),
                geometry.cell_volumes.dtype,
                swept_certified=False,
            )
            if gap_evidence is None
            else gap_evidence
        )
        if not isinstance(gap_evidence_, ALEGapEvidence):
            raise TypeError("gap_evidence must be ALEGapEvidence or None.")
        admissible = (
            geometry.passed
            & (geometry.minimum_cell_volume >= self.plan.minimum_cell_volume)
            & (geometry.minimum_face_measure >= self.plan.minimum_face_measure)
            & (geometry.minimum_velocity_dual_measure >= self.plan.minimum_dual_measure)
            & (geometry.minimum_oriented_dual_distance >= self.plan.minimum_dual_measure)
        )
        gcl = (
            (geometry.maximum_gcl_residual <= self.plan.gcl_tolerance)
            & (geometry.map_velocity_residual <= self.plan.gcl_tolerance)
            & (geometry.boundary_kinematic_residual <= self.plan.gcl_tolerance)
            & (geometry.free_stream_residual <= self.plan.gcl_tolerance)
            & (geometry.mapped_adjoint_residual <= self.plan.gcl_tolerance)
        )
        finite = geometry.finite & gap_evidence_.finite
        successful = admissible & gcl & gap_evidence_.successful & finite
        return ALEMeshEvidence(
            geometry.minimum_cell_volume,
            geometry.minimum_face_measure,
            geometry.minimum_velocity_dual_measure,
            geometry.minimum_oriented_dual_distance,
            geometry.maximum_gcl_residual,
            geometry.map_velocity_residual,
            geometry.boundary_kinematic_residual,
            geometry.free_stream_residual,
            geometry.mapped_adjoint_residual,
            gap_evidence_,
            admissible,
            gcl,
            finite,
            successful,
            geometry.topology_id,
            geometry.motion_plan_id,
        )

    def evaluate_geometry(
        self, time: ArrayLike, args: Any = None, /
    ) -> tuple[MACALEStageGeometry, ALEMeshEvidence]:
        geometry = self.plan.motion.evaluate(time, args)
        return geometry, self.evidence(geometry, args)

    def advance(
        self,
        state: CardiovascularALEState,
        start_time: ArrayLike,
        step_size: ArrayLike,
        args: Any = None,
        /,
        *,
        viscosity: ArrayLike = 0.0,
        density: ArrayLike = 1.0,
        forcing: FaceVelocity | None = None,
    ) -> CardiovascularALETransition:
        if not isinstance(state, CardiovascularALEState):
            raise TypeError("state must be CardiovascularALEState.")
        start = jnp.asarray(start_time)
        step = jnp.asarray(step_size)
        end = start + step
        flow = self.plan.motion.advance(
            state.velocity,
            start,
            step,
            args,
            viscosity=viscosity,
            density=density,
            forcing=forcing,
            pressure=state.pressure,
        )
        start_geometry = self.plan.motion.evaluate(start, args)
        geometry = self.plan.motion.evaluate(end, args)
        swept_gap = self._gap_evidence(
            self.plan.gap_route.swept_gap_provider(
                start_geometry,
                geometry,
                start,
                end,
                args,
            ),
            geometry.cell_volumes.dtype,
            swept_certified=True,
        )
        evidence = self.evidence(geometry, args, gap_evidence=swept_gap)
        candidate = CardiovascularALEState(flow.velocity, flow.pressure)
        finite_state = jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in flow.velocity))
        ) & jnp.all(jnp.isfinite(flow.pressure))
        successful = (
            flow.success
            & evidence.successful
            & evidence.gap.swept_certified
            & finite_state
        )
        accepted = CardiovascularALEState(
            tuple(
                jnp.where(successful, candidate_value, accepted_value)
                for candidate_value, accepted_value in zip(
                    candidate.velocity, state.velocity, strict=True
                )
            ),
            jnp.where(successful, candidate.pressure, state.pressure),
        )
        status = jnp.where(
            successful,
            int(ALETransitionStatus.SUCCESS),
            jnp.where(
                ~finite_state | ~evidence.finite,
                int(ALETransitionStatus.NONFINITE),
                jnp.where(
                    ~evidence.admissible,
                    int(ALETransitionStatus.INVALID_MESH),
                    jnp.where(
                        ~evidence.gcl_certified,
                        int(ALETransitionStatus.GCL_FAILURE),
                        jnp.where(
                            ~evidence.gap.successful,
                            int(ALETransitionStatus.MINIMUM_GAP_FAILURE),
                            int(ALETransitionStatus.FLOW_SOLVE_FAILURE),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return CardiovascularALETransition(
            candidate,
            accepted,
            flow,
            geometry,
            evidence,
            successful,
            status,
            self.prepared_id,
        )


__all__ = [
    "ALEGapEvidence",
    "ALEGapProvider",
    "ALESweptGapProvider",
    "ALEMeshEvidence",
    "ALEMinimumGapRoute",
    "ALETransitionStatus",
    "CardiovascularALEPlan",
    "CardiovascularALEState",
    "CardiovascularALETransition",
    "PreparedCardiovascularALE",
]
