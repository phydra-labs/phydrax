#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume import PreparedUnstructuredCollocatedOperators
from ..linalg import (
    ArraySpace,
    DiagonalPairing,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    prepare,
    PreparedLinearSolve,
    refresh,
    solve,
    TolerancePolicy,
)


MomentumPredictor = Callable[[Array, Array, Any], Array]


class _WeightedGaugedPressureAction(StrictModule, NonTrainableState):
    operators: PreparedUnstructuredCollocatedOperators
    face_inverse_momentum: Array

    def __init__(
        self,
        operators: PreparedUnstructuredCollocatedOperators,
        face_inverse_momentum: ArrayLike,
        /,
    ):
        self.operators = operators
        self.face_inverse_momentum = operators.validate_face_scalar(
            face_inverse_momentum, "Face inverse momentum"
        )

    def __call__(self, pressure: Array, /) -> Array:
        return self.operators.positive_gauged_weighted_laplacian(
            pressure, self.face_inverse_momentum
        )


class UnstructuredPressureProjectionResult(StrictModule):
    velocity: Array
    face_normal_velocity: Array
    pressure: Array
    pressure_increment: Array
    divergence_before: Array
    divergence_after: Array
    pressure_residual: Array
    compatible_rhs: Array
    rhie_chow_correction: Array
    face_inverse_momentum: Array
    gauge_defect: Array
    linear: LinearSolveResult
    converged: Array


class UnstructuredPressureProjectionPlan(StrictModule, NonTrainableState):
    """Gauged matrix-free pressure projection for collocated unstructured FV."""

    operators: PreparedUnstructuredCollocatedOperators
    density: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    pressure_space: ArraySpace
    linear_policy: LinearSolvePolicy
    linear_problem: LinearSystem
    prepared_linear: PreparedLinearSolve
    operator_id: str = eqx.field(static=True)
    pressure_problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedUnstructuredCollocatedOperators,
        /,
        *,
        density: float = 1.0,
        tolerance: float = 1e-9,
        maximum_iterations: int = 200,
        dtype: Any | None = None,
        linear_policy: LinearSolvePolicy | None = None,
    ):
        if not isinstance(operators, PreparedUnstructuredCollocatedOperators):
            raise TypeError("operators must be PreparedUnstructuredCollocatedOperators.")
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
        selected_dtype = (
            operators.discretization.cell_volumes.dtype if dtype is None else dtype
        )
        volumes = jnp.asarray(operators.discretization.cell_volumes, dtype=selected_dtype)
        space = ArraySpace(
            (operators.discretization.cell_count,),
            dtype=volumes.dtype,
            pairing=DiagonalPairing(volumes),
        )
        operator_id = canonical_fingerprint(
            {
                "kind": "unstructured-gauged-pressure-operator",
                "operators": operators.prepared_id,
                "dtype": jnp.dtype(volumes.dtype).name,
            }
        )
        base_action = _WeightedGaugedPressureAction(
            operators,
            jnp.ones((operators.discretization.face_measures.size,), dtype=volumes.dtype),
        )
        operator = FunctionLinearOperator(
            base_action,
            source=space,
            target=space,
            operator_id=operator_id,
        )
        pressure_problem_id = canonical_fingerprint(
            {
                "kind": "unstructured-pressure-system",
                "operator": operator_id,
            }
        )
        problem = LinearSystem(operator, problem_id=pressure_problem_id)
        policy = (
            LinearSolvePolicy(
                GMRES(restart=min(32, operators.discretization.cell_count)),
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
            raise TypeError("linear_policy must be LinearSolvePolicy.")
        self.operators = operators
        self.density = density_
        self.tolerance = tolerance_
        self.pressure_space = space
        self.operator_id = operator_id
        self.pressure_problem_id = pressure_problem_id
        self.linear_policy = policy
        self.linear_problem = problem
        self.prepared_linear = prepare(problem, policy)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-pressure-projection-plan",
                "operators": operators.prepared_id,
                "density": density_,
                "tolerance": tolerance_,
                "linear_plan": self.prepared_linear.plan.plan_id,
            }
        )

    def project(
        self,
        velocity: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        pressure: ArrayLike | None = None,
        inverse_momentum_diagonal: ArrayLike | None = None,
        face_normal_velocity: ArrayLike | None = None,
        boundary_normal_velocity: ArrayLike | None = None,
    ) -> UnstructuredPressureProjectionResult:
        value = self.operators.validate_cell_velocity(velocity)
        dtype = self.pressure_space.dtype
        value = value.astype(dtype)
        dt = jnp.asarray(step_size, dtype=dtype).reshape(())
        dt = eqx.error_if(
            dt,
            ~jnp.isfinite(dt) | (dt <= 0.0),
            "Pressure projection step_size must be positive and finite.",
        )
        pressure_ = (
            jnp.zeros((self.operators.discretization.cell_count,), dtype=dtype)
            if pressure is None
            else self.operators.validate_cell_scalar(pressure, "Pressure").astype(dtype)
        )
        inverse = (
            jnp.full_like(pressure_, dt / self.density)
            if inverse_momentum_diagonal is None
            else self.operators.validate_cell_scalar(
                inverse_momentum_diagonal, "Inverse momentum diagonal"
            ).astype(dtype)
        )
        face_inverse = self.operators.interpolate_inverse_momentum(inverse)
        pressure_operator = FunctionLinearOperator(
            _WeightedGaugedPressureAction(self.operators, face_inverse),
            source=self.pressure_space,
            target=self.pressure_space,
            operator_id=self.operator_id,
        )
        pressure_problem = LinearSystem(
            pressure_operator, problem_id=self.pressure_problem_id
        )
        prepared_linear = refresh(self.prepared_linear, pressure_problem)
        arithmetic_face = self.operators.interpolate_normal_velocity(value)
        predicted_face = (
            self.operators.rhie_chow_face_velocity(value, pressure_, inverse)
            if face_normal_velocity is None
            else self.operators.validate_face_scalar(
                face_normal_velocity, "Face-normal velocity"
            ).astype(dtype)
        )
        if boundary_normal_velocity is not None:
            boundary_value = self.operators.validate_face_scalar(
                boundary_normal_velocity, "Boundary-normal velocity"
            ).astype(dtype)
            predicted_face = jnp.where(
                self.operators.interior_faces, predicted_face, boundary_value
            )
        divergence_before = self.operators.divergence(predicted_face)
        volumes = self.operators.discretization.cell_volumes.astype(dtype)
        mean_divergence = jnp.sum(volumes * divergence_before) / jnp.sum(volumes)
        compatible_divergence = divergence_before - mean_divergence
        rhs = -compatible_divergence
        linear = solve(prepared_linear, rhs)
        increment = self.operators.gauge_project(linear.value)
        corrected_face_candidate = (
            predicted_face - face_inverse * self.operators.face_normal_gradient(increment)
        )
        corrected_velocity_candidate = value - inverse[
            :, None
        ] * self.operators.cell_gradient(increment)
        pressure_candidate = self.operators.gauge_project(pressure_ + increment)
        corrected_face = jnp.where(
            linear.successful, corrected_face_candidate, predicted_face
        )
        corrected_velocity = jnp.where(
            linear.successful, corrected_velocity_candidate, value
        )
        pressure_value = jnp.where(linear.successful, pressure_candidate, pressure_)
        divergence_after = self.operators.divergence(corrected_face)
        residual = (
            self.operators.positive_gauged_weighted_laplacian(increment, face_inverse)
            - rhs
        )
        residual_norm = jnp.sqrt(jnp.sum(volumes * residual**2))
        rhs_norm = jnp.sqrt(jnp.sum(volumes * rhs**2))
        converged = linear.successful & (
            residual_norm <= self.tolerance * jnp.maximum(rhs_norm, 1.0)
        )
        return UnstructuredPressureProjectionResult(
            velocity=corrected_velocity,
            face_normal_velocity=corrected_face,
            pressure=pressure_value,
            pressure_increment=increment,
            divergence_before=divergence_before,
            divergence_after=divergence_after,
            pressure_residual=residual,
            compatible_rhs=rhs,
            face_inverse_momentum=face_inverse,
            rhie_chow_correction=predicted_face - arithmetic_face,
            gauge_defect=jnp.sum(volumes * pressure_value) / jnp.sum(volumes),
            linear=linear,
            converged=converged,
        )


class UnstructuredPressureCorrectionResult(StrictModule):
    velocity: Array
    face_normal_velocity: Array
    pressure: Array
    divergence_history: Array
    linear_status_history: Array
    converged: Array


class UnstructuredPressureCorrectionPlan(StrictModule, NonTrainableState):
    """Fixed-count collocated predictor/projection correction loop."""

    projection: UnstructuredPressureProjectionPlan
    correctors: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        projection: UnstructuredPressureProjectionPlan,
        correctors: int = 2,
        /,
    ):
        if not isinstance(projection, UnstructuredPressureProjectionPlan):
            raise TypeError("projection must be UnstructuredPressureProjectionPlan.")
        correctors_ = int(correctors)
        if correctors_ <= 0:
            raise ValueError("correctors must be positive.")
        self.projection = projection
        self.correctors = correctors_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-pressure-correction-plan",
                "projection": projection.plan_id,
                "correctors": correctors_,
            }
        )

    def advance(
        self,
        time: ArrayLike,
        velocity: ArrayLike,
        pressure: ArrayLike,
        step_size: ArrayLike,
        predictor: MomentumPredictor,
        args: Any = None,
        /,
        *,
        inverse_momentum_diagonal: ArrayLike | None = None,
        boundary_normal_velocity: ArrayLike | None = None,
    ) -> UnstructuredPressureCorrectionResult:
        if not callable(predictor):
            raise TypeError("predictor must be callable.")
        dtype = self.projection.pressure_space.dtype
        current_velocity = self.projection.operators.validate_cell_velocity(
            velocity
        ).astype(dtype)
        current_pressure = self.projection.operators.validate_cell_scalar(
            pressure, "Pressure"
        ).astype(dtype)
        time_ = jnp.asarray(time, dtype=dtype).reshape(())
        predicted = predictor(time_, current_velocity, args)
        predicted = self.projection.operators.validate_cell_velocity(predicted).astype(
            dtype
        )
        history = jnp.zeros((self.correctors,), dtype=dtype)
        status_history = jnp.full((self.correctors,), -1, dtype=jnp.int32)
        initial_face = self.projection.operators.interpolate_normal_velocity(
            predicted
        ).astype(dtype)

        def body(index, carry):
            velocity_, face_, pressure_, history_, statuses_, converged_ = carry
            result = self.projection.project(
                velocity_,
                step_size,
                pressure=pressure_,
                inverse_momentum_diagonal=inverse_momentum_diagonal,
                boundary_normal_velocity=boundary_normal_velocity,
            )
            volumes = self.projection.operators.discretization.cell_volumes.astype(
                result.divergence_after.dtype
            )
            norm = jnp.sqrt(jnp.sum(volumes * result.divergence_after**2))
            return (
                result.velocity,
                result.face_normal_velocity,
                result.pressure,
                history_.at[index].set(norm),
                statuses_.at[index].set(result.linear.status),
                converged_ & result.converged,
            )

        corrected, face, pressure_, history, statuses, converged = jax.lax.fori_loop(
            0,
            self.correctors,
            body,
            (
                predicted,
                initial_face,
                current_pressure,
                history,
                status_history,
                jnp.asarray(True),
            ),
        )
        return UnstructuredPressureCorrectionResult(
            velocity=corrected,
            face_normal_velocity=face,
            pressure=pressure_,
            divergence_history=history,
            linear_status_history=statuses,
            converged=converged,
        )


__all__ = [
    "UnstructuredPressureCorrectionPlan",
    "UnstructuredPressureCorrectionResult",
    "UnstructuredPressureProjectionPlan",
    "UnstructuredPressureProjectionResult",
]
