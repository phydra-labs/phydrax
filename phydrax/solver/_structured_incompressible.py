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


class _WeightedMACPressureAction(StrictModule, NonTrainableState):
    operators: PreparedMACOperators
    face_inverse_momentum: FaceVelocity

    def __init__(
        self,
        operators: PreparedMACOperators,
        face_inverse_momentum: FaceVelocity,
        /,
    ):
        self.operators = operators
        self.face_inverse_momentum = operators.validate_velocity(face_inverse_momentum)

    def __call__(self, pressure: Array, /) -> Array:
        return self.operators.positive_gauged_weighted_laplacian(
            pressure, self.face_inverse_momentum
        )


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
    solve_method: str = eqx.field(static=True)
    linear: LinearSolveResult | None
    transform: TransformDiagonalSolveResult | None
    converged: Array


class MACPressureProjectionPlan(StrictModule, NonTrainableState):
    """Prepared compatible MAC projection with transform or linalg pressure solve."""

    operators: PreparedMACOperators
    density: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    solve_method: MACPressureSolveMethod = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    linear_problem: LinearSystem
    prepared_linear: PreparedLinearSolve
    transform_plan: FDLaplacianSolvePlan | None
    operator_id: str = eqx.field(static=True)
    pressure_problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        /,
        *,
        density: float = 1.0,
        tolerance: float = 1e-9,
        maximum_iterations: int = 500,
        solve_method: MACPressureSolveMethod = "auto",
        linear_policy: LinearSolvePolicy | None = None,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
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
        unit_face = tuple(
            jnp.ones(layout.shape, dtype=operators.pressure_space.dtype)
            for layout in operators.discretization.face_layouts
        )
        operator_id = canonical_fingerprint(
            {
                "kind": "mac-gauged-pressure-operator-v1",
                "operators": operators.prepared_id,
            }
        )
        pressure_operator = FunctionLinearOperator(
            _WeightedMACPressureAction(operators, unit_face),
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
            {
                "kind": "mac-pressure-system-v1",
                "operator": operator_id,
            }
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
        transform_plan = self._prepare_transform(operators, tolerance_)
        if solve_method == "transform" and transform_plan is None:
            raise ValueError(
                "Transform MAC projection requires an exact uniform tensor operator."
            )
        identifier = canonical_fingerprint(
            {
                "kind": "mac-pressure-projection-plan-v2",
                "operators": operators.prepared_id,
                "density": density_,
                "tolerance": tolerance_,
                "solve_method": solve_method,
                "linear_plan": prepared_linear.plan.plan_id,
                "transform_plan": (
                    None if transform_plan is None else transform_plan.plan_id
                ),
            }
        )
        self.operators = operators
        self.density = density_
        self.tolerance = tolerance_
        self.solve_method = solve_method
        self.linear_policy = policy
        self.linear_problem = problem
        self.prepared_linear = prepared_linear
        self.transform_plan = transform_plan
        self.operator_id = operator_id
        self.pressure_problem_id = pressure_problem_id
        self.plan_id = identifier

    @staticmethod
    def _prepare_transform(
        operators: PreparedMACOperators,
        tolerance: float,
        /,
    ) -> FDLaplacianSolvePlan | None:
        if not operators.report.transform_eligible:
            return None
        boundaries = {
            name: (
                ("periodic", "periodic")
                if axis.periodic
                else ("neumann", "neumann")
            )
            for name, axis in zip(
                operators.discretization.grid.axis_names,
                operators.discretization.grid.structured_axes,
                strict=True,
            )
        }
        diagonalization = diagonalize_fd_laplacian(
            operators.discretization.grid, boundaries
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

    def project(
        self,
        velocity: FaceVelocity,
        step_size: ArrayLike,
        /,
        *,
        pressure: ArrayLike | None = None,
        inverse_momentum_diagonal: ArrayLike | None = None,
    ) -> MACPressureProjectionResult:
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
            else self.operators.gauge_project(pressure)
        )
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
        rhs = -self.operators.compatibility_project(divergence_before)
        constant_inverse = inverse_momentum_diagonal is None
        use_transform = (
            self.solve_method != "iterative"
            and constant_inverse
            and self.transform_plan is not None
        )
        if self.solve_method == "transform" and not use_transform:
            raise ValueError(
                "Transform projection does not support variable inverse momentum."
            )
        if use_transform:
            alpha = step / self.density
            transform = self.transform_plan.solve(rhs / alpha)
            increment_candidate = self.operators.gauge_project(transform.value)
            solve_success = transform.converged
            linear = None
            route = "transform"
        else:
            pressure_operator = FunctionLinearOperator(
                _WeightedMACPressureAction(self.operators, face_inverse),
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
            problem = LinearSystem(
                pressure_operator, problem_id=self.pressure_problem_id
            )
            prepared = refresh(self.prepared_linear, problem)
            linear = solve(prepared, rhs, initial_guess=incoming_pressure)
            increment_candidate = self.operators.gauge_project(linear.value)
            solve_success = linear.successful
            transform = None
            route = "iterative"
        corrected_candidate = tuple(
            component - coefficient * gradient
            for component, coefficient, gradient in zip(
                values,
                face_inverse,
                self.operators.gradient(increment_candidate),
                strict=True,
            )
        )
        pressure_candidate = self.operators.gauge_project(
            incoming_pressure + increment_candidate
        )
        residual = (
            self.operators.positive_gauged_weighted_laplacian(
                increment_candidate, face_inverse
            )
            - rhs
        )
        volumes = self.operators.discretization.cell_volumes.astype(dtype)
        residual_norm = jnp.sqrt(jnp.sum(volumes * residual**2))
        rhs_norm = jnp.sqrt(jnp.sum(volumes * rhs**2))
        gauge_defect = jnp.abs(jnp.sum(volumes * pressure_candidate))
        divergence_candidate = self.operators.divergence(corrected_candidate)
        divergence_norm = jnp.sqrt(jnp.sum(volumes * divergence_candidate**2))
        converged = (
            solve_success
            & jnp.all(jnp.isfinite(increment_candidate))
            & jnp.all(jnp.isfinite(divergence_candidate))
            & (residual_norm <= self.tolerance * jnp.maximum(rhs_norm, 1.0))
            & (divergence_norm <= self.tolerance * jnp.maximum(rhs_norm, 1.0))
            & (gauge_defect <= self.tolerance)
        )
        corrected = tuple(
            jnp.where(converged, candidate, original)
            for candidate, original in zip(corrected_candidate, values, strict=True)
        )
        pressure_value = jnp.where(converged, pressure_candidate, incoming_pressure)
        increment = jnp.where(converged, increment_candidate, jnp.zeros_like(increment_candidate))
        divergence_after = self.operators.divergence(corrected)
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
            solve_method=route,
            linear=linear,
            transform=transform,
            converged=converged,
        )


__all__ = [
    "MACPressureProjectionPlan",
    "MACPressureProjectionResult",
    "MACPressureSolveMethod",
]
