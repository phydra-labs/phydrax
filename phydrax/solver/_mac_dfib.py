#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import LagrangianMarkerKinematics
from ..discretization.finite_volume import (
    FaceVelocity,
    MACBoundaryStageData,
    MACMarkerRelation,
    MACMarkerRouteState,
    MACMarkerTransferDiagnostics,
    PreparedMACMarkerTransfer,
)
from ..linalg import (
    DifferentiationPolicy,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    solve,
    TolerancePolicy,
)
from ._mac_stage_inverse_general import (
    MACOperatorStageInverseMomentum,
    MACVariableDensityStageInverseMomentum,
)
from ._mac_stage_inverse_momentum import (
    MACDiagonalStageInverseMomentum,
    MACHelmholtzStageInverseMomentum,
    MACStageInverseMomentum,
)
from ._structured_incompressible import (
    MACPressureProjectionPlan,
    MACPressureProjectionResult,
)


class MACDivergenceFreeTransferDiagnostics(StrictModule):
    regularized: MACMarkerTransferDiagnostics
    projected_divergence_norm: Array
    interpolation_work: Array
    spreading_work: Array
    work_residual: Array
    finite: Array
    successful: Array
    transfer_id: str = eqx.field(static=True)


class MACDivergenceFreeMarkerTransfer(StrictModule, NonTrainableState):
    """Discrete Helmholtz-projected marker interpolation and its exact adjoint."""

    transfer: PreparedMACMarkerTransfer
    projection: MACPressureProjectionPlan
    periodic: bool = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: PreparedMACMarkerTransfer,
        projection: MACPressureProjectionPlan,
        /,
        *,
        require_periodic: bool = True,
        tolerance: float = 1.0e-9,
    ):
        if transfer.operators.prepared_id != projection.operators.prepared_id:
            raise ValueError("DFIB transfer and pressure projection operators differ.")
        periodic = all(
            axis.periodic
            for axis in transfer.operators.discretization.grid.structured_axes
        )
        if require_periodic and not periodic:
            raise ValueError("Periodic DFIB requires every MAC axis to be periodic.")
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("DFIB tolerance must be positive and finite.")
        self.transfer = transfer
        self.projection = projection
        self.periodic = periodic
        self.tolerance = tolerance_
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "mac-divergence-free-marker-transfer",
                "transfer": transfer.prepared_id,
                "projection": projection.plan_id,
                "periodic": periodic,
                "tolerance": tolerance_,
            }
        )

    def project_velocity(
        self,
        velocity: FaceVelocity,
        /,
        *,
        boundary_stage: MACBoundaryStageData | None = None,
    ) -> MACPressureProjectionResult:
        if not self.periodic and boundary_stage is not None:
            homogeneous = jnp.all(
                jnp.stack(
                    tuple(
                        jnp.all(value == 0.0)
                        for value in boundary_stage.values + boundary_stage.rates
                    )
                )
            )
            checked = tuple(
                eqx.error_if(
                    value,
                    ~homogeneous,
                    "Bounded DFIB correction requires homogeneous boundary data.",
                )
                for value in velocity
            )
        else:
            checked = velocity
        return self.projection.project(
            checked,
            1.0,
            boundary_stage=boundary_stage,
        )

    def gather(
        self,
        relation: MACMarkerRelation,
        velocity: FaceVelocity,
        /,
        *,
        boundary_stage: MACBoundaryStageData | None = None,
    ) -> Array:
        projected = self.project_velocity(velocity, boundary_stage=boundary_stage)
        return self.transfer.gather(relation, projected.velocity)

    def spread(
        self,
        relation: MACMarkerRelation,
        marker_force_density: ArrayLike,
        /,
        *,
        boundary_stage: MACBoundaryStageData | None = None,
    ) -> FaceVelocity:
        raw = self.transfer.spread(relation, marker_force_density)
        projected = self.project_velocity(raw, boundary_stage=boundary_stage)
        return projected.velocity

    def diagnostics(
        self,
        relation: MACMarkerRelation,
        velocity: FaceVelocity,
        marker_force_density: ArrayLike,
        /,
        *,
        boundary_stage: MACBoundaryStageData | None = None,
    ) -> MACDivergenceFreeTransferDiagnostics:
        projected = self.project_velocity(velocity, boundary_stage=boundary_stage)
        gathered = self.transfer.gather(relation, projected.velocity)
        spread = self.spread(
            relation,
            marker_force_density,
            boundary_stage=boundary_stage,
        )
        force = self.transfer.markers.active_velocity_space.validate(
            jnp.asarray(marker_force_density)
        )
        interpolation_work = jnp.real(
            self.transfer.markers.active_velocity_space.inner(gathered, force)
        )
        spreading_work = jnp.real(
            self.transfer.operators.velocity_space.inner(velocity, spread)
        )
        residual = interpolation_work - spreading_work
        divergence = self.transfer.operators.divergence(spread)
        divergence_norm = jnp.sqrt(
            jnp.sum(self.transfer.operators.discretization.cell_volumes * divergence**2)
        )
        regular = self.transfer.diagnostics(relation, projected.velocity, force)
        scale = jnp.maximum(1.0, jnp.abs(interpolation_work) + jnp.abs(spreading_work))
        finite = (
            regular.finite
            & projected.finite
            & jnp.isfinite(divergence_norm)
            & jnp.isfinite(residual)
        )
        successful = (
            finite
            & projected.converged
            & (divergence_norm <= self.tolerance)
            & (jnp.abs(residual) <= self.tolerance * scale)
        )
        return MACDivergenceFreeTransferDiagnostics(
            regular,
            divergence_norm,
            interpolation_work,
            spreading_work,
            residual,
            finite,
            successful,
            self.transfer_id,
        )


class MACDFIBProjectionResult(StrictModule):
    velocity: FaceVelocity
    pressure: Array
    marker_force_density: Array
    relation: MACMarkerRelation
    route_state: MACMarkerRouteState
    linear: LinearSolveResult
    transfer: MACDivergenceFreeTransferDiagnostics
    divergence_norm: Array
    slip_norm: Array
    route_unchanged: Array
    finite: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)


class MACDFIBProjectionPlan(StrictModule, NonTrainableState):
    """Exact prescribed-velocity solve using divergence-free marker transfer."""

    transfer: MACDivergenceFreeMarkerTransfer
    linear_policy: LinearSolvePolicy
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: MACDivergenceFreeMarkerTransfer,
        /,
        *,
        linear_policy: LinearSolvePolicy | None = None,
        tolerance: float = 1.0e-9,
    ):
        if not isinstance(transfer, MACDivergenceFreeMarkerTransfer):
            raise TypeError("transfer must be MACDivergenceFreeMarkerTransfer.")
        tolerance_ = float(tolerance)
        if tolerance_ <= 0.0 or not np.isfinite(tolerance_):
            raise ValueError("DFIB projection tolerance must be positive and finite.")
        self.transfer = transfer
        self.linear_policy = (
            LinearSolvePolicy(
                GMRES(restart=50),
                tolerance=TolerancePolicy(
                    relative=tolerance_, absolute=tolerance_, max_steps=500
                ),
                differentiation=DifferentiationPolicy("mathematical"),
            )
            if linear_policy is None
            else linear_policy
        )
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-dfib-prescribed-projection",
                "transfer": transfer.transfer_id,
                "linear_method": self.linear_policy.method.name,
                "tolerance": tolerance_,
            }
        )

    def project(
        self,
        velocity: FaceVelocity,
        inverse_momentum: MACStageInverseMomentum | ArrayLike,
        kinematics: LagrangianMarkerKinematics,
        /,
        *,
        marker_force_density: ArrayLike | None = None,
        boundary_stage: MACBoundaryStageData | None = None,
        expected_routes: MACMarkerRouteState | None = None,
        allow_route_refresh: bool = False,
    ) -> MACDFIBProjectionResult:
        regular = self.transfer.transfer
        marker_state = regular.markers.validate_kinematics(kinematics)
        relation = regular.relation(marker_state.position)
        routes = regular.route_state(relation)
        route_unchanged = (
            jnp.asarray(True)
            if expected_routes is None
            else regular.routes_match(relation, expected_routes)
        )
        stage = (
            self.transfer.projection.boundaries.evaluate(jnp.asarray(0.0), None)
            if boundary_stage is None
            else self.transfer.projection.boundaries.validate_stage(boundary_stage)
        )
        stage_inverse = (
            inverse_momentum
            if isinstance(
                inverse_momentum,
                (
                    MACDiagonalStageInverseMomentum,
                    MACHelmholtzStageInverseMomentum,
                    MACOperatorStageInverseMomentum,
                    MACVariableDensityStageInverseMomentum,
                ),
            )
            else MACDiagonalStageInverseMomentum(
                regular.operators,
                self.transfer.projection.boundaries,
                stage,
                inverse_momentum,
            )
        )
        projected_base = self.transfer.project_velocity(velocity, boundary_stage=stage)
        target = regular.markers.active_values(marker_state.velocity)
        marker_space = regular.markers.active_velocity_space

        def mobility_action(multiplier):
            spread = self.transfer.spread(relation, multiplier, boundary_stage=stage)
            inverse = stage_inverse.apply_inverse(spread)
            return self.transfer.gather(relation, inverse, boundary_stage=stage)

        mobility = FunctionLinearOperator(
            mobility_action,
            source=marker_space,
            target=marker_space,
            transpose_action=mobility_action,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
            operator_id=f"mac-dfib-mobility/{relation.relation_id}",
        )
        before = self.transfer.gather(
            relation, projected_base.velocity, boundary_stage=stage
        )
        right_hand_side = target - before
        initial = (
            jnp.zeros_like(target)
            if marker_force_density is None
            else marker_space.validate(jnp.asarray(marker_force_density))
        )
        linear = solve(
            LinearSystem(
                mobility,
                problem_id=f"mac-dfib-marker/{relation.relation_id}",
            ),
            right_hand_side,
            policy=self.linear_policy,
            initial_guess=initial,
        )
        spread = self.transfer.spread(relation, linear.value, boundary_stage=stage)
        inverse = stage_inverse.apply_inverse(spread)
        candidate = tuple(
            value + update
            for value, update in zip(projected_base.velocity, inverse, strict=True)
        )
        projected = self.transfer.project_velocity(candidate, boundary_stage=stage)
        slip = (
            self.transfer.gather(relation, projected.velocity, boundary_stage=stage)
            - target
        )
        divergence_norm = jnp.sqrt(
            jnp.sum(
                regular.operators.discretization.cell_volumes
                * projected.divergence_after**2
            )
        )
        slip_norm = jnp.sqrt(jnp.real(marker_space.inner(slip, slip)))
        diagnostics = self.transfer.diagnostics(
            relation,
            projected.velocity,
            linear.value,
            boundary_stage=stage,
        )
        finite = (
            projected.finite
            & diagnostics.finite
            & jnp.isfinite(divergence_norm)
            & jnp.isfinite(slip_norm)
        )
        scale = jnp.maximum(1.0, jnp.sqrt(jnp.real(marker_space.inner(target, target))))
        accepted = (
            linear.successful
            & projected.converged
            & diagnostics.successful
            & finite
            & (route_unchanged | bool(allow_route_refresh))
            & (divergence_norm <= self.tolerance)
            & (slip_norm <= self.tolerance * scale)
        )
        velocity_value = tuple(
            jnp.where(accepted, candidate_, original)
            for candidate_, original in zip(projected.velocity, velocity, strict=True)
        )
        return MACDFIBProjectionResult(
            velocity_value,
            projected.pressure,
            linear.value,
            relation,
            routes,
            linear,
            diagnostics,
            divergence_norm,
            slip_norm,
            route_unchanged,
            finite,
            accepted,
            self.plan_id,
        )


__all__ = [
    "MACDFIBProjectionPlan",
    "MACDFIBProjectionResult",
    "MACDivergenceFreeMarkerTransfer",
    "MACDivergenceFreeTransferDiagnostics",
]
