#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_difference import (
    diagonalize_fd_laplacian,
    FDLaplacianSolvePlan,
)
from ..discretization.finite_volume import FaceVelocity, PreparedMACOperators
from ..discretization.finite_volume._mac_boundary import (
    MACBoundaryPlan,
    MACBoundaryStageData,
    MACPressureClosureKind,
    PreparedMACBoundaryPlan,
)
from ..linalg import (
    ConjugateGradient,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    prepare,
    PreparedLinearSolve,
    refresh,
    solve,
    TolerancePolicy,
    TransformDiagonalSolveResult,
)


MACPressureSolveMethod: TypeAlias = Literal["auto", "transform", "iterative"]
MACPressureGaugeKind: TypeAlias = Literal["zero-mean", "none"]
MACPressureCompatibilityKind: TypeAlias = Literal["projected", "unprojected"]


class _WeightedMACPressureAction(StrictModule, NonTrainableState):
    operators: PreparedMACOperators
    boundaries: PreparedMACBoundaryPlan
    face_inverse_momentum: FaceVelocity

    def __init__(
        self,
        operators: PreparedMACOperators,
        boundaries: PreparedMACBoundaryPlan,
        face_inverse_momentum: FaceVelocity,
        /,
    ):
        self.operators = operators
        self.boundaries = boundaries
        self.face_inverse_momentum = operators.validate_velocity(face_inverse_momentum)

    def __call__(self, pressure: Array, /) -> Array:
        value = self.operators.validate_pressure(pressure)
        if self.boundaries.closure_kind == "neumann":
            volumes = self.operators.discretization.cell_volumes.astype(value.dtype)
            mean = jnp.sum(volumes * value) / jnp.sum(volumes)
            value = value - mean
        else:
            mean = jnp.asarray(0.0, dtype=value.dtype)
        gradient = self.boundaries.pressure_gradient(value, None, homogeneous=True)
        weighted = tuple(
            coefficient * derivative
            for coefficient, derivative in zip(
                self.face_inverse_momentum, gradient, strict=True
            )
        )
        return -self.operators.divergence(weighted) + mean


class MACPressureClosureReport(StrictModule, NonTrainableState):
    """Closure-dependent gauge, compatibility, and integrated mass evidence."""

    kind: MACPressureClosureKind = eqx.field(static=True)
    gauge: MACPressureGaugeKind = eqx.field(static=True)
    compatibility: MACPressureCompatibilityKind = eqx.field(static=True)
    integrated_mass_flux: Array
    mass_defect: Array
    gauge_defect: Array
    finite: Array
    successful: Array
    closure_id: str = eqx.field(static=True)


class MACPressureProjectionResult(StrictModule):
    velocity: FaceVelocity
    pressure: Array
    pressure_increment: Array
    divergence_before: Array
    divergence_after: Array
    pressure_residual: Array
    compatible_rhs: Array
    face_inverse_momentum: FaceVelocity
    gauge_defect: Array
    closure: MACPressureClosureReport
    solve_method: str = eqx.field(static=True)
    linear: LinearSolveResult | None
    transform: TransformDiagonalSolveResult | None
    finite: Array
    converged: Array
    projection_id: str = eqx.field(static=True)


class MACRateProjectionResult(StrictModule):
    """Projected momentum rate and its pressure Lagrange multiplier."""

    rate: FaceVelocity
    pressure: Array
    divergence_before: Array
    divergence_after: Array
    pressure_residual: Array
    compatible_rhs: Array
    face_inverse_density: FaceVelocity
    gauge_defect: Array
    closure: MACPressureClosureReport
    solve_method: str = eqx.field(static=True)
    linear: LinearSolveResult | None
    transform: TransformDiagonalSolveResult | None
    finite: Array
    converged: Array
    projection_id: str = eqx.field(static=True)


class MACPressureProjectionPlan(StrictModule, NonTrainableState):
    """Prepared compatible MAC projection with closure-aware pressure solves."""

    operators: PreparedMACOperators
    boundaries: PreparedMACBoundaryPlan
    density: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    solve_method: MACPressureSolveMethod = eqx.field(static=True)
    closure_kind: MACPressureClosureKind = eqx.field(static=True)
    gauge_kind: MACPressureGaugeKind = eqx.field(static=True)
    compatibility_kind: MACPressureCompatibilityKind = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    linear_problem: LinearSystem
    prepared_linear: PreparedLinearSolve
    transform_plan: FDLaplacianSolvePlan | None
    operator_id: str = eqx.field(static=True)
    pressure_problem_id: str = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        /,
        *,
        boundaries: PreparedMACBoundaryPlan | MACBoundaryPlan | None = None,
        density: float = 1.0,
        tolerance: float = 1e-9,
        maximum_iterations: int = 500,
        solve_method: MACPressureSolveMethod = "auto",
        linear_policy: LinearSolvePolicy | None = None,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        boundaries_ = (
            MACBoundaryPlan(operators).prepare()
            if boundaries is None
            else boundaries.prepare()
            if isinstance(boundaries, MACBoundaryPlan)
            else boundaries
        )
        if not isinstance(boundaries_, PreparedMACBoundaryPlan):
            raise TypeError(
                "boundaries must be PreparedMACBoundaryPlan, MACBoundaryPlan, or None."
            )
        if boundaries_.operators.prepared_id != operators.prepared_id:
            raise ValueError("MAC projection boundaries must use the same operators.")
        density_ = float(density)
        tolerance_ = float(tolerance)
        iterations = int(maximum_iterations)
        if (
            not np.isfinite(density_)
            or density_ <= 0.0
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or iterations <= 0
        ):
            raise ValueError("Projection density, tolerance, and iterations are invalid.")
        if solve_method not in ("auto", "transform", "iterative"):
            raise ValueError("solve_method must be 'auto', 'transform', or 'iterative'.")
        closure_kind = boundaries_.closure_kind
        gauge_kind: MACPressureGaugeKind = (
            "zero-mean" if closure_kind == "neumann" else "none"
        )
        compatibility_kind: MACPressureCompatibilityKind = (
            "projected" if closure_kind == "neumann" else "unprojected"
        )
        closure_id = canonical_fingerprint(
            {
                "kind": "mac-pressure-closure",
                "boundaries": boundaries_.prepared_id,
                "closure": closure_kind,
                "gauge": gauge_kind,
                "compatibility": compatibility_kind,
            }
        )
        unit_face = tuple(
            jnp.ones(layout.shape, dtype=operators.pressure_space.dtype)
            for layout in operators.discretization.face_layouts
        )
        operator_id = canonical_fingerprint(
            {
                "kind": "mac-closure-pressure-operator",
                "operators": operators.prepared_id,
                "closure": closure_id,
            }
        )
        pressure_operator = FunctionLinearOperator(
            _WeightedMACPressureAction(operators, boundaries_, unit_face),
            source=operators.pressure_space,
            target=operators.pressure_space,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
            operator_id=operator_id,
        )
        pressure_problem_id = canonical_fingerprint(
            {"kind": "mac-pressure-system", "operator": operator_id}
        )
        problem = LinearSystem(pressure_operator, problem_id=pressure_problem_id)
        policy = (
            LinearSolvePolicy(
                ConjugateGradient(),
                tolerance=TolerancePolicy(
                    relative=tolerance_,
                    absolute=tolerance_,
                    max_steps=iterations,
                ),
            )
            if linear_policy is None
            else linear_policy
        )
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be LinearSolvePolicy or None.")
        prepared_linear = prepare(problem, policy)
        transform_plan = self._prepare_transform(operators, boundaries_, tolerance_)
        if solve_method == "transform" and transform_plan is None:
            raise ValueError(
                "Transform MAC projection requires a uniform all-Neumann tensor closure."
            )
        identifier = canonical_fingerprint(
            {
                "kind": "mac-pressure-projection-plan",
                "operators": operators.prepared_id,
                "boundaries": boundaries_.prepared_id,
                "density": density_,
                "tolerance": tolerance_,
                "solve_method": solve_method,
                "closure": closure_id,
                "linear_plan": prepared_linear.plan.plan_id,
                "transform_plan": (
                    None if transform_plan is None else transform_plan.plan_id
                ),
            }
        )
        self.operators = operators
        self.boundaries = boundaries_
        self.density = density_
        self.tolerance = tolerance_
        self.solve_method = solve_method
        self.closure_kind = closure_kind
        self.gauge_kind = gauge_kind
        self.compatibility_kind = compatibility_kind
        self.linear_policy = policy
        self.linear_problem = problem
        self.prepared_linear = prepared_linear
        self.transform_plan = transform_plan
        self.operator_id = operator_id
        self.pressure_problem_id = pressure_problem_id
        self.closure_id = closure_id
        self.plan_id = identifier

    @staticmethod
    def _prepare_transform(
        operators: PreparedMACOperators,
        boundaries: PreparedMACBoundaryPlan,
        tolerance: float,
        /,
    ) -> FDLaplacianSolvePlan | None:
        if (
            not operators.report.transform_eligible
            or boundaries.closure_kind != "neumann"
        ):
            return None
        boundary_kinds = {
            name: (("periodic", "periodic") if axis.periodic else ("neumann", "neumann"))
            for name, axis in zip(
                operators.discretization.grid.axis_names,
                operators.discretization.grid.structured_axes,
                strict=True,
            )
        }
        diagonalization = diagonalize_fd_laplacian(
            operators.discretization.grid, boundary_kinds
        )
        probe = jnp.arange(
            int(np.prod(operators.discretization.cell_shape)),
            dtype=operators.pressure_space.dtype,
        ).reshape(operators.discretization.cell_shape)
        direct_action = -diagonalization.apply(probe)
        mac_action = operators.positive_laplacian(probe)
        defect = float(jnp.max(jnp.abs(direct_action - mac_action)))
        if defect > max(100.0 * tolerance, 5e-10):
            raise RuntimeError(
                "FD transform and MAC pressure operators failed exact-action identity."
            )
        return FDLaplacianSolvePlan(
            diagonalization,
            operator_scale=-1.0,
            compatibility="project_rhs",
            gauge="zero_mean",
            zero_tolerance=tolerance,
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
        step_size: ArrayLike,
        /,
        *,
        pressure: ArrayLike | None = None,
        inverse_momentum_diagonal: ArrayLike | None = None,
        boundary_stage: MACBoundaryStageData | None = None,
    ) -> MACPressureProjectionResult:
        stage = self._stage(boundary_stage)
        values = self.operators.validate_velocity(velocity)
        dtype = self.operators.pressure_space.dtype
        step = jnp.asarray(step_size, dtype=dtype).reshape(())
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Pressure projection step_size must be positive and finite.",
        )
        incoming_pressure = (
            jnp.zeros(self.operators.discretization.cell_shape, dtype=dtype)
            if pressure is None
            else self.operators.validate_pressure(pressure)
        )
        if self.closure_kind == "neumann":
            incoming_pressure = self.operators.gauge_project(incoming_pressure)
        inverse = (
            jnp.full(
                self.operators.discretization.cell_shape,
                step / self.density,
                dtype=dtype,
            )
            if inverse_momentum_diagonal is None
            else self.operators.validate_pressure(inverse_momentum_diagonal)
        )
        inverse = eqx.error_if(
            inverse,
            jnp.any(~jnp.isfinite(inverse) | (inverse <= 0.0)),
            "Inverse momentum diagonal must be positive and finite.",
        )
        face_inverse = self.operators.interpolate_inverse_momentum(inverse)
        divergence_before = self.operators.divergence(values)
        zero_pressure = jnp.zeros_like(divergence_before)
        boundary_gradient = self.boundaries.pressure_gradient(
            zero_pressure, stage, homogeneous=False
        )
        boundary_divergence = self.operators.divergence(
            tuple(
                coefficient * derivative
                for coefficient, derivative in zip(
                    face_inverse, boundary_gradient, strict=True
                )
            )
        )
        raw_rhs = -divergence_before + boundary_divergence
        rhs = (
            self.operators.compatibility_project(raw_rhs)
            if self.closure_kind == "neumann"
            else raw_rhs
        )
        constant_inverse = inverse_momentum_diagonal is None
        use_transform = (
            self.solve_method != "iterative"
            and constant_inverse
            and self.transform_plan is not None
        )
        if self.solve_method == "transform" and not use_transform:
            raise ValueError(
                "Transform projection does not support this closure or variable momentum."
            )
        if use_transform:
            alpha = step / self.density
            transform = self.transform_plan.solve(rhs / alpha)
            solution_candidate = self.operators.gauge_project(transform.value)
            solve_success = transform.converged
            linear = None
            route = "transform"
        else:
            pressure_operator = FunctionLinearOperator(
                _WeightedMACPressureAction(self.operators, self.boundaries, face_inverse),
                source=self.operators.pressure_space,
                target=self.operators.pressure_space,
                properties=OperatorProperties(
                    self_adjoint=True,
                    positive_definite=True,
                    evidence={
                        "self_adjoint": "construction",
                        "positive_definite": "construction",
                    },
                ),
                operator_id=self.operator_id,
            )
            problem = LinearSystem(pressure_operator, problem_id=self.pressure_problem_id)
            prepared = refresh(self.prepared_linear, problem)
            linear = solve(prepared, rhs, initial_guess=incoming_pressure)
            solution_candidate = linear.value
            if self.closure_kind == "neumann":
                solution_candidate = self.operators.gauge_project(solution_candidate)
            solve_success = linear.successful
            transform = None
            route = "iterative"
        if self.closure_kind == "neumann":
            increment_candidate = solution_candidate
            pressure_candidate = self.operators.gauge_project(
                incoming_pressure + increment_candidate
            )
        else:
            pressure_candidate = solution_candidate
            increment_candidate = pressure_candidate - incoming_pressure
        correction_gradient = self.boundaries.pressure_gradient(
            solution_candidate,
            stage,
            homogeneous=self.closure_kind == "neumann",
        )
        corrected_candidate = tuple(
            component - coefficient * gradient
            for component, coefficient, gradient in zip(
                values,
                face_inverse,
                correction_gradient,
                strict=True,
            )
        )
        action = _WeightedMACPressureAction(self.operators, self.boundaries, face_inverse)
        residual = action(solution_candidate) - rhs
        volumes = self.operators.discretization.cell_volumes.astype(dtype)
        residual_norm = jnp.sqrt(jnp.sum(volumes * residual**2))
        rhs_norm = jnp.sqrt(jnp.sum(volumes * rhs**2))
        gauge_defect = (
            jnp.abs(jnp.sum(volumes * pressure_candidate))
            if self.closure_kind == "neumann"
            else jnp.asarray(0.0, dtype=dtype)
        )
        divergence_candidate = self.operators.divergence(corrected_candidate)
        divergence_norm = jnp.sqrt(jnp.sum(volumes * divergence_candidate**2))
        integrated_mass_flux = jnp.sum(volumes * divergence_candidate)
        mass_defect = jnp.abs(integrated_mass_flux)
        finite = (
            stage.finite
            & jnp.all(jnp.isfinite(solution_candidate))
            & jnp.all(jnp.isfinite(divergence_candidate))
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(gauge_defect)
            & jnp.isfinite(mass_defect)
        )
        converged = (
            solve_success
            & stage.successful
            & finite
            & (residual_norm <= self.tolerance * jnp.maximum(rhs_norm, 1.0))
            & (divergence_norm <= self.tolerance * jnp.maximum(rhs_norm, 1.0))
            & (gauge_defect <= self.tolerance)
        )
        corrected = tuple(
            jnp.where(converged, candidate, original)
            for candidate, original in zip(corrected_candidate, values, strict=True)
        )
        pressure_value = jnp.where(converged, pressure_candidate, incoming_pressure)
        increment = jnp.where(
            converged, increment_candidate, jnp.zeros_like(increment_candidate)
        )
        divergence_after = self.operators.divergence(corrected)
        reported_mass_flux = jnp.sum(volumes * divergence_after)
        closure = MACPressureClosureReport(
            kind=self.closure_kind,
            gauge=self.gauge_kind,
            compatibility=self.compatibility_kind,
            integrated_mass_flux=reported_mass_flux,
            mass_defect=jnp.abs(reported_mass_flux),
            gauge_defect=gauge_defect,
            finite=finite,
            successful=converged,
            closure_id=self.closure_id,
        )
        return MACPressureProjectionResult(
            velocity=corrected,
            pressure=pressure_value,
            pressure_increment=increment,
            divergence_before=divergence_before,
            divergence_after=divergence_after,
            pressure_residual=residual,
            compatible_rhs=rhs,
            face_inverse_momentum=face_inverse,
            gauge_defect=gauge_defect,
            closure=closure,
            solve_method=route,
            linear=linear,
            transform=transform,
            finite=finite,
            converged=converged,
            projection_id=self.plan_id,
        )

    def project_rate(
        self,
        rate: FaceVelocity,
        /,
        *,
        boundary_stage: MACBoundaryStageData | None = None,
    ) -> MACRateProjectionResult:
        """Project one face-velocity rate without changing essential normal faces."""
        projected = self.project(rate, 1.0, boundary_stage=boundary_stage)
        return MACRateProjectionResult(
            rate=projected.velocity,
            pressure=projected.pressure_increment,
            divergence_before=projected.divergence_before,
            divergence_after=projected.divergence_after,
            pressure_residual=projected.pressure_residual,
            compatible_rhs=projected.compatible_rhs,
            face_inverse_density=projected.face_inverse_momentum,
            gauge_defect=projected.gauge_defect,
            closure=projected.closure,
            solve_method=projected.solve_method,
            linear=projected.linear,
            transform=projected.transform,
            finite=projected.finite,
            converged=projected.converged,
            projection_id=self.plan_id,
        )


__all__ = [
    "MACPressureClosureReport",
    "MACPressureCompatibilityKind",
    "MACPressureGaugeKind",
    "MACRateProjectionResult",
    "MACPressureProjectionPlan",
    "MACPressureProjectionResult",
    "MACPressureSolveMethod",
]
