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
from ..discretization.finite_volume._incompressible import (
    FaceVelocity,
    PreparedMACOperators,
)
from ..linalg import (
    DiagonalPreconditioner,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    PCG,
    PreconditioningPolicy,
    prepare,
    PreparedLinearSolve,
    refresh,
    solve,
    TolerancePolicy,
)


def _maximum_abs(values: tuple[Array, ...], dtype: jnp.dtype, /) -> Array:
    if not values:
        return jnp.asarray(0.0, dtype=dtype)
    return jnp.max(jnp.stack(tuple(jnp.max(jnp.abs(value)) for value in values)))


class _VariableCoefficientMACPressureAction(StrictModule, NonTrainableState):
    operators: PreparedMACOperators
    face_coefficient: FaceVelocity

    def __init__(
        self,
        operators: PreparedMACOperators,
        face_coefficient: FaceVelocity,
        /,
    ):
        self.operators = operators
        self.face_coefficient = operators.validate_velocity(face_coefficient)

    def __call__(self, pressure: Array, /) -> Array:
        return self.operators.positive_gauged_weighted_laplacian(
            pressure, self.face_coefficient
        )


class MACVariableDensityProjectionResult(StrictModule):
    """Fail-closed pressure impulse applied to staggered face momentum."""

    momentum: FaceVelocity
    velocity: FaceVelocity
    pressure: Array
    pressure_increment: Array
    pressure_impulse: FaceVelocity
    divergence_before: Array
    divergence_after: Array
    pressure_residual: Array
    compatible_rhs: Array
    face_inverse_density: FaceVelocity
    gauge_defect: Array
    coefficient_contrast: Array
    preparation_id: str = eqx.field(static=True)
    residual_norm: Array
    divergence_norm: Array
    momentum_impulse_residual: Array
    velocity_identity_residual: Array
    minimum_face_density: Array
    positive: Array
    finite: Array
    linear: LinearSolveResult
    converged: Array
    successful: Array
    solve_method: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)


class MACVariableDensityRateProjectionResult(StrictModule):
    """Projected velocity rate and its pressure force per unit face volume."""

    velocity_rate: FaceVelocity
    momentum_pressure_rate: FaceVelocity
    pressure: Array
    divergence_before: Array
    divergence_after: Array
    pressure_residual: Array
    compatible_rhs: Array
    face_inverse_density: FaceVelocity
    gauge_defect: Array
    coefficient_contrast: Array
    preparation_id: str = eqx.field(static=True)
    residual_norm: Array
    divergence_norm: Array
    positive: Array
    finite: Array
    linear: LinearSolveResult
    converged: Array
    successful: Array
    solve_method: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)


class MACVariableDensityProjectionPlan(StrictModule, NonTrainableState):
    """Prepared iterative variable-coefficient MAC pressure projection."""

    operators: PreparedMACOperators
    tolerance: float = eqx.field(static=True)
    solve_method: str = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    linear_problem: LinearSystem
    prepared_linear: PreparedLinearSolve
    operator_id: str = eqx.field(static=True)
    pressure_problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        /,
        *,
        tolerance: float = 1e-9,
        maximum_iterations: int = 500,
        solve_method: str = "auto",
        linear_policy: LinearSolvePolicy | None = None,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        tolerance_ = float(tolerance)
        iterations = int(maximum_iterations)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0 or iterations <= 0:
            raise ValueError("Projection tolerance and maximum_iterations are invalid.")
        if solve_method not in ("auto", "iterative", "direct"):
            raise ValueError("solve_method must be 'auto', 'iterative', or 'direct'.")
        if solve_method == "direct":
            raise ValueError(
                "Variable-density direct pressure solve is unsupported here; no "
                "iterative fallback was taken."
            )
        unit_face = tuple(
            jnp.ones(layout.shape, dtype=operators.pressure_space.dtype)
            for layout in operators.discretization.face_layouts
        )
        operator_id = canonical_fingerprint(
            {
                "kind": "mac-variable-coefficient-gauged-pressure-operator",
                "operators": operators.prepared_id,
            }
        )
        pressure_operator = FunctionLinearOperator(
            _VariableCoefficientMACPressureAction(operators, unit_face),
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
        problem_id = canonical_fingerprint(
            {
                "kind": "mac-variable-density-pressure-system",
                "operator": operator_id,
            }
        )
        problem = LinearSystem(pressure_operator, problem_id=problem_id)
        policy = (
            LinearSolvePolicy(
                PCG(),
                tolerance=TolerancePolicy(
                    relative=tolerance_,
                    absolute=tolerance_,
                    max_steps=iterations,
                ),
                preconditioning=PreconditioningPolicy(
                    DiagonalPreconditioner(
                        jnp.ones(
                            (operators.pressure_space.size,),
                            dtype=operators.pressure_space.dtype,
                        ),
                        space=operators.pressure_space,
                        positive_definite=True,
                        preconditioner_id=canonical_fingerprint(
                            {
                                "kind": "mac-variable-density-constant-preconditioner",
                                "operators": operators.prepared_id,
                            }
                        ),
                    ),
                    side="left",
                    refresh="frozen",
                ),
            )
            if linear_policy is None
            else linear_policy
        )
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be LinearSolvePolicy or None.")
        prepared = prepare(problem, policy)
        identifier = canonical_fingerprint(
            {
                "kind": "mac-variable-density-projection-plan",
                "operators": operators.prepared_id,
                "tolerance": tolerance_,
                "linear_plan": prepared.plan.plan_id,
                "route": "pcg",
            }
        )
        self.operators = operators
        self.tolerance = tolerance_
        self.solve_method = "pcg"
        self.linear_policy = policy
        self.linear_problem = problem
        self.prepared_linear = prepared
        self.operator_id = operator_id
        self.pressure_problem_id = problem_id
        self.plan_id = identifier

    def validate_face_inverse_density(
        self, face_inverse_density: FaceVelocity, /
    ) -> FaceVelocity:
        values = self.operators.validate_velocity(face_inverse_density)
        valid = jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(value) & (value > 0.0)) for value in values)
            )
        )
        return tuple(
            eqx.error_if(
                value,
                ~valid,
                "Projection face inverse density must be positive and finite.",
            )
            for value in values
        )

    def project(
        self,
        momentum: FaceVelocity,
        face_inverse_density: FaceVelocity,
        step_size: ArrayLike,
        /,
        *,
        pressure: ArrayLike | None = None,
    ) -> MACVariableDensityProjectionResult:
        values = self.operators.validate_velocity(momentum)
        inverse = self.validate_face_inverse_density(face_inverse_density)
        dtype = self.operators.pressure_space.dtype
        finite_momentum = jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in values))
        )
        values = tuple(
            eqx.error_if(
                value,
                ~finite_momentum,
                "Projected MAC face momentum must be finite.",
            )
            for value in values
        )
        step = jnp.asarray(step_size, dtype=dtype).reshape(())
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Variable-density projection step_size must be positive and finite.",
        )
        incoming_pressure = (
            jnp.zeros(self.operators.discretization.cell_shape, dtype=dtype)
            if pressure is None
            else self.operators.gauge_project(pressure)
        )
        velocity_before = tuple(
            coefficient * component
            for coefficient, component in zip(inverse, values, strict=True)
        )
        divergence_before = self.operators.divergence(velocity_before)
        rhs = -self.operators.compatibility_project(divergence_before)
        coefficient = tuple(step * value for value in inverse)
        pressure_operator = FunctionLinearOperator(
            _VariableCoefficientMACPressureAction(self.operators, coefficient),
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
        increment_candidate = self.operators.gauge_project(linear.value)
        gradient = self.operators.gradient(increment_candidate)
        impulse_candidate = tuple(-step * value for value in gradient)
        momentum_candidate = tuple(
            component + impulse
            for component, impulse in zip(values, impulse_candidate, strict=True)
        )
        velocity_candidate = tuple(
            inverse_value * component
            for inverse_value, component in zip(inverse, momentum_candidate, strict=True)
        )
        pressure_candidate = self.operators.gauge_project(
            incoming_pressure + increment_candidate
        )
        residual = (
            self.operators.positive_gauged_weighted_laplacian(
                increment_candidate, coefficient
            )
            - rhs
        )
        volumes = self.operators.discretization.cell_volumes.astype(dtype)
        residual_norm = jnp.sqrt(jnp.sum(volumes * residual**2))
        rhs_norm = jnp.sqrt(jnp.sum(volumes * rhs**2))
        divergence_candidate = self.operators.divergence(velocity_candidate)
        divergence_norm = jnp.sqrt(jnp.sum(volumes * divergence_candidate**2))
        gauge_defect = jnp.abs(jnp.sum(volumes * pressure_candidate))
        coefficient_minimum = jnp.min(
            jnp.stack(tuple(jnp.min(value) for value in inverse))
        )
        coefficient_maximum = jnp.max(
            jnp.stack(tuple(jnp.max(value) for value in inverse))
        )
        coefficient_contrast = coefficient_maximum / coefficient_minimum
        impulse_residual = _maximum_abs(
            tuple(
                candidate - original - impulse
                for candidate, original, impulse in zip(
                    momentum_candidate, values, impulse_candidate, strict=True
                )
            ),
            dtype,
        )
        velocity_residual = _maximum_abs(
            tuple(
                velocity_value - inverse_value * momentum_value
                for velocity_value, inverse_value, momentum_value in zip(
                    velocity_candidate, inverse, momentum_candidate, strict=True
                )
            ),
            dtype,
        )
        positive = jnp.all(jnp.stack(tuple(jnp.all(value > 0.0) for value in inverse)))
        finite = (
            jnp.all(jnp.isfinite(increment_candidate))
            & jnp.all(jnp.isfinite(divergence_candidate))
            & jnp.all(
                jnp.stack(
                    tuple(jnp.all(jnp.isfinite(value)) for value in momentum_candidate)
                )
            )
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(gauge_defect)
        )
        scale = jnp.maximum(rhs_norm, 1.0)
        tolerance = self.tolerance * scale
        converged = (
            linear.successful
            & positive
            & finite
            & (residual_norm <= tolerance)
            & (divergence_norm <= tolerance)
            & (gauge_defect <= self.tolerance)
            & (impulse_residual <= tolerance)
            & (velocity_residual <= tolerance)
        )
        momentum_value = tuple(
            jnp.where(converged, candidate, original)
            for candidate, original in zip(momentum_candidate, values, strict=True)
        )
        velocity_value = tuple(
            inverse_value * component
            for inverse_value, component in zip(inverse, momentum_value, strict=True)
        )
        pressure_value = jnp.where(converged, pressure_candidate, incoming_pressure)
        increment = jnp.where(
            converged, increment_candidate, jnp.zeros_like(increment_candidate)
        )
        impulse = tuple(
            jnp.where(converged, candidate, jnp.zeros_like(candidate))
            for candidate in impulse_candidate
        )
        divergence_after = self.operators.divergence(velocity_value)
        return MACVariableDensityProjectionResult(
            momentum=momentum_value,
            velocity=velocity_value,
            pressure=pressure_value,
            pressure_increment=increment,
            pressure_impulse=impulse,
            divergence_before=divergence_before,
            divergence_after=divergence_after,
            pressure_residual=residual,
            compatible_rhs=rhs,
            face_inverse_density=inverse,
            gauge_defect=gauge_defect,
            coefficient_contrast=coefficient_contrast,
            preparation_id=self.plan_id,
            residual_norm=residual_norm,
            divergence_norm=divergence_norm,
            momentum_impulse_residual=impulse_residual,
            velocity_identity_residual=velocity_residual,
            minimum_face_density=1.0
            / jnp.max(jnp.stack(tuple(jnp.max(value) for value in inverse))),
            positive=positive,
            finite=finite,
            linear=linear,
            converged=converged,
            successful=converged,
            solve_method=self.solve_method,
            projection_id=self.plan_id,
        )

    def project_velocity_rate(
        self,
        velocity_rate: FaceVelocity,
        face_inverse_density: FaceVelocity,
        /,
        *,
        pressure: ArrayLike | None = None,
    ) -> MACVariableDensityRateProjectionResult:
        inverse = self.validate_face_inverse_density(face_inverse_density)
        rate = self.operators.validate_velocity(velocity_rate)
        equivalent_momentum_rate = tuple(
            value / coefficient for value, coefficient in zip(rate, inverse, strict=True)
        )
        projected = self.project(
            equivalent_momentum_rate,
            inverse,
            1.0,
            pressure=pressure,
        )
        return MACVariableDensityRateProjectionResult(
            velocity_rate=projected.velocity,
            momentum_pressure_rate=projected.pressure_impulse,
            pressure=projected.pressure_increment,
            divergence_before=projected.divergence_before,
            divergence_after=projected.divergence_after,
            pressure_residual=projected.pressure_residual,
            compatible_rhs=projected.compatible_rhs,
            face_inverse_density=projected.face_inverse_density,
            gauge_defect=projected.gauge_defect,
            coefficient_contrast=projected.coefficient_contrast,
            preparation_id=projected.preparation_id,
            residual_norm=projected.residual_norm,
            divergence_norm=projected.divergence_norm,
            positive=projected.positive,
            finite=projected.finite,
            linear=projected.linear,
            converged=projected.converged,
            successful=projected.successful,
            solve_method=self.solve_method,
            projection_id=self.plan_id,
        )


__all__ = [
    "MACVariableDensityProjectionPlan",
    "MACVariableDensityProjectionResult",
    "MACVariableDensityRateProjectionResult",
]
