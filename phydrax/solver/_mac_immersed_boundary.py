#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._lagrangian_marker import LagrangianMarkerKinematics
from ..discretization.finite_volume._incompressible import (
    FaceVelocity,
    PreparedMACOperators,
)
from ..discretization.finite_volume._mac_boundary import (
    MACBoundaryPlan,
    MACBoundaryStageData,
    PreparedMACBoundaryPlan,
)
from ..discretization.finite_volume._mac_marker_transfer import (
    MACMarkerRelation,
    MACMarkerTransferDiagnostics,
    PreparedMACMarkerTransfer,
)
from ..linalg import (
    BlockSpace,
    DifferentiationPolicy,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    RankPolicy,
    solve,
    svd as svd_linalg,
    TolerancePolicy,
)
from ._structured_incompressible import MACPressureClosureReport


MACImmersedBoundarySolveMethod: TypeAlias = Literal["iterative"]


class MACImmersedBoundaryProjectionStatus(IntFlag):
    SUCCESS = 0
    INVALID_COEFFICIENT = 1
    BOUNDARY_FAILED = 2
    TRANSFER_FAILED = 4
    LINEAR_SOLVE_FAILED = 8
    NONFINITE = 16
    PRESSURE_GAUGE_FAILED = 32
    DIVERGENCE_FAILED = 64
    MARKER_SLIP_FAILED = 128
    KKT_RESIDUAL_FAILED = 256
    ROUTE_CHANGED = 512
    MARKER_RANK_FAILED = 1024


class MACImmersedBoundaryProjectionResult(StrictModule):
    """One exact pressure-plus-marker correction with fail-closed evidence."""

    velocity: FaceVelocity
    candidate_velocity: FaceVelocity
    pressure: Array
    pressure_increment: Array
    marker_force_density: Array
    candidate_marker_force_density: Array
    relation: MACMarkerRelation
    transfer_diagnostics: MACMarkerTransferDiagnostics
    marker_velocity_before: Array
    marker_velocity_after: Array
    marker_slip: Array
    divergence_before: Array
    divergence_after: Array
    kkt_residual: tuple[Array, Array]
    kkt_residual_norm: Array
    gauge_defect: Array
    closure: MACPressureClosureReport
    linear: LinearSolveResult
    marker_numerical_rank: Array
    marker_condition: Array
    marker_rank_certified: Array
    status: Array
    finite: Array
    converged: Array
    projection_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.converged


class MACImmersedBoundaryProjectionPlan(StrictModule, NonTrainableState):
    """Closure-aware unit-density projection enforcing divergence and marker velocity."""

    operators: PreparedMACOperators
    boundaries: PreparedMACBoundaryPlan
    transfer: PreparedMACMarkerTransfer
    constraint_length: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    maximum_rank_check_size: int = eqx.field(static=True)
    solve_method: MACImmersedBoundarySolveMethod = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        transfer: PreparedMACMarkerTransfer,
        /,
        *,
        boundaries: PreparedMACBoundaryPlan | MACBoundaryPlan | None = None,
        constraint_length: float | None = None,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 500,
        linear_policy: LinearSolvePolicy | None = None,
        maximum_rank_check_size: int = 256,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        if not isinstance(transfer, PreparedMACMarkerTransfer):
            raise TypeError("transfer must be PreparedMACMarkerTransfer.")
        if transfer.operators.prepared_id != operators.prepared_id:
            raise ValueError("Transfer and projection must share MAC operators.")
        boundaries_ = (
            MACBoundaryPlan(operators).prepare()
            if boundaries is None
            else boundaries.prepare()
            if isinstance(boundaries, MACBoundaryPlan)
            else boundaries
        )
        if not isinstance(boundaries_, PreparedMACBoundaryPlan):
            raise TypeError("boundaries must be a prepared or unprepared MAC boundary plan.")
        if boundaries_.operators.prepared_id != operators.prepared_id:
            raise ValueError("Projection boundaries must share MAC operators.")
        spacings = []
        for axis in operators.discretization.grid.structured_axes:
            widths = np.asarray(axis.interval_widths, dtype=float)
            if not np.allclose(widths, widths[0], rtol=1.0e-10, atol=1.0e-12):
                raise ValueError("Exact immersed projection currently requires uniform axes.")
            spacings.append(float(widths[0]))
        length = min(spacings) if constraint_length is None else float(constraint_length)
        tolerance_ = float(tolerance)
        iterations = int(maximum_iterations)
        if (
            not np.isfinite(length)
            or length <= 0.0
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or iterations <= 0
        ):
            raise ValueError("Constraint length, tolerance, and iterations are invalid.")
        policy = (
            LinearSolvePolicy(
                GMRES(restart=min(50, iterations)),
                tolerance=TolerancePolicy(
                    relative=tolerance_, absolute=tolerance_, max_steps=iterations
                ),
                differentiation=DifferentiationPolicy("mathematical"),
            )
            if linear_policy is None
            else linear_policy
        )
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be LinearSolvePolicy or None.")
        rank_limit = int(maximum_rank_check_size)
        if rank_limit <= 0:
            raise ValueError("maximum_rank_check_size must be positive.")
        closure_id = canonical_fingerprint(
            {
                "kind": "mac-immersed-pressure-closure",
                "boundaries": boundaries_.prepared_id,
                "closure": boundaries_.closure_kind,
                "constraint_length": length,
            }
        )
        problem_id = canonical_fingerprint(
            {
                "kind": "mac-immersed-boundary-linear-system",
                "operators": operators.prepared_id,
                "transfer": transfer.prepared_id,
                "closure": closure_id,
            }
        )
        self.operators = operators
        self.boundaries = boundaries_
        self.transfer = transfer
        self.constraint_length = length
        self.tolerance = tolerance_
        self.linear_policy = policy
        self.maximum_rank_check_size = rank_limit
        self.solve_method = "iterative"
        self.closure_id = closure_id
        self.problem_id = problem_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-immersed-boundary-projection-plan",
                "problem": problem_id,
                "linear_method": policy.method.name,
                "maximum_rank_check_size": rank_limit,
            }
        )

    def _stage(self, stage: MACBoundaryStageData | None, /) -> MACBoundaryStageData:
        return (
            self.boundaries.evaluate(jnp.asarray(0.0), None)
            if stage is None
            else self.boundaries.validate_stage(stage)
        )

    def project(
        self,
        velocity: FaceVelocity,
        inverse_momentum_coefficient: ArrayLike,
        kinematics: LagrangianMarkerKinematics,
        /,
        *,
        pressure: ArrayLike | None = None,
        marker_force_density: ArrayLike | None = None,
        boundary_stage: MACBoundaryStageData | None = None,
    ) -> MACImmersedBoundaryProjectionResult:
        stage = self._stage(boundary_stage)
        values = self.operators.validate_velocity(velocity)
        dtype = self.operators.pressure_space.dtype
        coefficient = jnp.asarray(inverse_momentum_coefficient, dtype=dtype)
        if coefficient.shape != ():
            raise ValueError("inverse_momentum_coefficient must be scalar.")
        valid_coefficient = jnp.isfinite(coefficient) & (coefficient > 0.0)
        safe_coefficient = jnp.where(valid_coefficient, coefficient, 1.0)
        marker_state = self.transfer.markers.validate_kinematics(kinematics)
        relation = self.transfer.relation(marker_state.position)
        target_velocity = self.transfer.markers.active_values(marker_state.velocity)
        bounded = self.boundaries.enforce(values, stage)
        bounded = self.operators.validate_velocity(bounded)
        marker_before = self.transfer.gather(relation, bounded)
        divergence_before = self.operators.divergence(bounded)
        incoming_pressure = (
            jnp.zeros(self.operators.discretization.cell_shape, dtype=dtype)
            if pressure is None
            else self.operators.validate_pressure(pressure)
        )
        if self.boundaries.closure_kind == "neumann":
            incoming_pressure = self.operators.gauge_project(incoming_pressure)
        initial_multiplier = (
            jnp.zeros(
                self.transfer.markers.active_velocity_space.structure().shape,
                dtype=dtype,
            )
            if marker_force_density is None
            else self.transfer.markers.active_velocity_space.validate(
                jnp.asarray(marker_force_density, dtype=dtype)
            )
        )
        ell = jnp.asarray(self.constraint_length, dtype=dtype)
        zero_pressure = jnp.zeros_like(incoming_pressure)
        face_inverse = tuple(
            jnp.full(layout.shape, safe_coefficient, dtype=dtype)
            for layout in self.operators.discretization.face_layouts
        )
        marker_space = self.transfer.markers.active_velocity_space
        marker_rank_certified = marker_space.size <= self.maximum_rank_check_size
        if marker_rank_certified:
            marker_mobility = FunctionLinearOperator(
                lambda multiplier: self.transfer.gather(
                    relation,
                    self.boundaries.homogeneous_rate(
                        tuple(
                            inverse * force
                            for inverse, force in zip(
                                face_inverse,
                                self.transfer.spread(relation, multiplier),
                                strict=True,
                            )
                        )
                    ),
                ),
                source=marker_space,
                target=marker_space,
                properties=OperatorProperties(
                    self_adjoint=True,
                    evidence={"self_adjoint": "construction"},
                ),
                operator_id=f"mac-marker-mobility/{relation.relation_id}",
            )
            marker_rank_result = svd_linalg.svd(
                svd_linalg.SVDProblem(
                    marker_mobility,
                    problem_id=f"mac-marker-rank/{relation.relation_id}",
                ),
                policy=svd_linalg.SVDSolvePolicy(
                    count=marker_space.size,
                    rank=RankPolicy(require_full_rank=True),
                ),
            )
            marker_singular_values = marker_rank_result.singular_values
            marker_smallest = jnp.min(marker_singular_values)
            marker_condition = jnp.max(marker_singular_values) / jnp.maximum(
                marker_smallest,
                jnp.finfo(marker_singular_values.dtype).tiny,
            )
            marker_rank = marker_rank_result.numerical_rank
            marker_rank_valid = marker_rank_result.successful & jnp.isfinite(
                marker_condition
            )
        else:
            marker_rank = jnp.asarray(-1, dtype=jnp.int32)
            marker_condition = jnp.asarray(jnp.nan, dtype=dtype)
            marker_rank_valid = jnp.asarray(True)
        boundary_gradient = self.boundaries.pressure_gradient(
            zero_pressure, stage, homogeneous=False
        )
        boundary_divergence = self.operators.divergence(
            tuple(
                inverse * gradient
                for inverse, gradient in zip(
                    face_inverse, boundary_gradient, strict=True
                )
            )
        )
        pressure_rhs = ell * (-divergence_before + boundary_divergence)
        if self.boundaries.closure_kind == "neumann":
            pressure_rhs = self.operators.compatibility_project(pressure_rhs)
        marker_rhs = target_velocity - marker_before
        dual_space = BlockSpace(
            (self.operators.pressure_space, self.transfer.markers.active_velocity_space),
            names=("pressure", "marker"),
        )

        def correction(dual):
            scaled_pressure, multiplier = dual
            scaled_pressure = self.operators.validate_pressure(scaled_pressure)
            if self.boundaries.closure_kind == "neumann":
                volumes = self.operators.discretization.cell_volumes.astype(dtype)
                mean = jnp.sum(volumes * scaled_pressure) / jnp.sum(volumes)
                projected_pressure = scaled_pressure - mean
            else:
                mean = jnp.asarray(0.0, dtype=dtype)
                projected_pressure = scaled_pressure
            physical_pressure = ell * projected_pressure
            gradient = self.boundaries.pressure_gradient(
                physical_pressure,
                stage,
                homogeneous=self.boundaries.closure_kind == "neumann",
            )
            spread = self.transfer.spread(relation, multiplier)
            raw = tuple(
                inverse * (derivative - force)
                for inverse, derivative, force in zip(
                    face_inverse, gradient, spread, strict=True
                )
            )
            admissible = self.boundaries.homogeneous_rate(raw)
            return admissible, mean

        def dual_action(dual):
            image, mean = correction(dual)
            pressure_image = -ell * self.operators.divergence(image)
            if self.boundaries.closure_kind == "neumann":
                pressure_image = pressure_image + mean
            marker_image = -self.transfer.gather(relation, image)
            return pressure_image, marker_image

        operator = FunctionLinearOperator(
            dual_action,
            source=dual_space,
            target=dual_space,
            properties=OperatorProperties(
                self_adjoint=True,
                evidence={"self_adjoint": "construction"},
            ),
            operator_id=f"mac-immersed-dual/{self.plan_id}",
        )
        problem = LinearSystem(operator, problem_id=self.problem_id)
        initial_guess = (incoming_pressure / ell, initial_multiplier)
        linear = solve(
            problem,
            (pressure_rhs, marker_rhs),
            policy=self.linear_policy,
            initial_guess=initial_guess,
        )
        scaled_pressure_candidate, multiplier_candidate = linear.value
        velocity_correction, _ = correction(linear.value)
        candidate_velocity = tuple(
            original - delta
            for original, delta in zip(bounded, velocity_correction, strict=True)
        )
        physical_increment = ell * scaled_pressure_candidate
        if self.boundaries.closure_kind == "neumann":
            physical_increment = self.operators.gauge_project(physical_increment)
            pressure_candidate = self.operators.gauge_project(
                incoming_pressure + physical_increment
            )
        else:
            pressure_candidate = physical_increment
        divergence_candidate = self.operators.divergence(candidate_velocity)
        marker_after_candidate = self.transfer.gather(relation, candidate_velocity)
        slip_candidate = marker_after_candidate - target_velocity
        residual = operator(linear.value)
        kkt_residual = (
            residual[0] - pressure_rhs,
            residual[1] - marker_rhs,
        )
        kkt_residual_norm = jnp.sqrt(
            jnp.real(dual_space.inner(kkt_residual, kkt_residual))
        )
        volumes = self.operators.discretization.cell_volumes.astype(dtype)
        rhs_norm = jnp.sqrt(
            jnp.sum(volumes * pressure_rhs**2)
            + jnp.real(
                self.transfer.markers.active_velocity_space.inner(
                    marker_rhs, marker_rhs
                )
            )
        )
        divergence_norm = jnp.sqrt(jnp.sum(volumes * divergence_candidate**2))
        slip_norm = jnp.sqrt(
            jnp.real(
                self.transfer.markers.active_velocity_space.inner(
                    slip_candidate, slip_candidate
                )
            )
        )
        gauge_defect = (
            jnp.abs(jnp.sum(volumes * pressure_candidate))
            if self.boundaries.closure_kind == "neumann"
            else jnp.asarray(0.0, dtype=dtype)
        )
        transfer_diagnostics = self.transfer.diagnostics(
            relation, candidate_velocity, multiplier_candidate
        )
        scale = jnp.maximum(rhs_norm, 1.0)
        tolerance = self.tolerance * scale
        finite = (
            stage.finite
            & jnp.all(
                jnp.stack(
                    tuple(jnp.all(jnp.isfinite(item)) for item in candidate_velocity)
                )
            )
            & jnp.all(jnp.isfinite(pressure_candidate))
            & jnp.all(jnp.isfinite(multiplier_candidate))
            & jnp.isfinite(kkt_residual_norm)
            & jnp.isfinite(divergence_norm)
            & jnp.isfinite(slip_norm)
            & jnp.isfinite(gauge_defect)
        )
        converged = (
            valid_coefficient
            & stage.successful
            & relation.successful
            & transfer_diagnostics.successful
            & linear.successful
            & marker_rank_valid
            & finite
            & (kkt_residual_norm <= tolerance)
            & (divergence_norm <= tolerance)
            & (slip_norm <= tolerance)
            & (gauge_defect <= self.tolerance)
        )
        status = jnp.asarray(
            int(MACImmersedBoundaryProjectionStatus.SUCCESS), dtype=jnp.int32
        )
        checks = (
            (valid_coefficient, MACImmersedBoundaryProjectionStatus.INVALID_COEFFICIENT),
            (stage.successful, MACImmersedBoundaryProjectionStatus.BOUNDARY_FAILED),
            (
                relation.successful & transfer_diagnostics.successful,
                MACImmersedBoundaryProjectionStatus.TRANSFER_FAILED,
            ),
            (linear.successful, MACImmersedBoundaryProjectionStatus.LINEAR_SOLVE_FAILED),
            (finite, MACImmersedBoundaryProjectionStatus.NONFINITE),
            (
                gauge_defect <= self.tolerance,
                MACImmersedBoundaryProjectionStatus.PRESSURE_GAUGE_FAILED,
            ),
            (
                divergence_norm <= tolerance,
                MACImmersedBoundaryProjectionStatus.DIVERGENCE_FAILED,
            ),
            (
                slip_norm <= tolerance,
                MACImmersedBoundaryProjectionStatus.MARKER_SLIP_FAILED,
            ),
            (
                kkt_residual_norm <= tolerance,
                MACImmersedBoundaryProjectionStatus.KKT_RESIDUAL_FAILED,
            ),
            (
                marker_rank_valid,
                MACImmersedBoundaryProjectionStatus.MARKER_RANK_FAILED,
            ),
        )
        for passed, flag in checks:
            status = status | jnp.where(passed, 0, int(flag)).astype(jnp.int32)
        accepted_velocity = tuple(
            jnp.where(converged, candidate, original)
            for candidate, original in zip(candidate_velocity, bounded, strict=True)
        )
        accepted_pressure = jnp.where(
            converged, pressure_candidate, incoming_pressure
        )
        accepted_increment = jnp.where(
            converged, physical_increment, jnp.zeros_like(physical_increment)
        )
        accepted_multiplier = jnp.where(
            converged, multiplier_candidate, initial_multiplier
        )
        marker_after = self.transfer.gather(relation, accepted_velocity)
        slip = marker_after - target_velocity
        divergence_after = self.operators.divergence(accepted_velocity)
        closure = MACPressureClosureReport(
            kind=self.boundaries.closure_kind,
            gauge="zero-mean"
            if self.boundaries.closure_kind == "neumann"
            else "none",
            compatibility="projected"
            if self.boundaries.closure_kind == "neumann"
            else "unprojected",
            integrated_mass_flux=jnp.sum(volumes * divergence_after),
            mass_defect=jnp.abs(jnp.sum(volumes * divergence_after)),
            gauge_defect=gauge_defect,
            finite=finite,
            successful=converged,
            closure_id=self.closure_id,
        )
        return MACImmersedBoundaryProjectionResult(
            velocity=accepted_velocity,
            candidate_velocity=candidate_velocity,
            pressure=accepted_pressure,
            pressure_increment=accepted_increment,
            marker_force_density=accepted_multiplier,
            candidate_marker_force_density=multiplier_candidate,
            relation=relation,
            transfer_diagnostics=transfer_diagnostics,
            marker_velocity_before=marker_before,
            marker_velocity_after=marker_after,
            marker_slip=slip,
            divergence_before=divergence_before,
            divergence_after=divergence_after,
            kkt_residual=kkt_residual,
            kkt_residual_norm=kkt_residual_norm,
            gauge_defect=gauge_defect,
            closure=closure,
            linear=linear,
            marker_numerical_rank=marker_rank,
            marker_condition=marker_condition,
            marker_rank_certified=jnp.asarray(marker_rank_certified),
            status=status,
            finite=finite,
            converged=converged,
            projection_id=self.plan_id,
        )


__all__ = [
    "MACImmersedBoundaryProjectionPlan",
    "MACImmersedBoundaryProjectionResult",
    "MACImmersedBoundaryProjectionStatus",
    "MACImmersedBoundarySolveMethod",
]
