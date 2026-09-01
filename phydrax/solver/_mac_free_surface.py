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
from ..discretization.finite_volume import (
    FaceVelocity,
    MACBoundaryPlan,
    MACBoundaryStageData,
    PreparedMACBoundaryPlan,
    PreparedMACOperators,
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
)


class _MaskedMACPressureAction(StrictModule, NonTrainableState):
    operators: PreparedMACOperators
    boundaries: PreparedMACBoundaryPlan
    face_inverse_momentum: FaceVelocity
    liquid_mask: Array

    def __call__(self, pressure: Array, /) -> Array:
        value = self.operators.validate_pressure(pressure)
        liquid = self.liquid_mask
        all_liquid = jnp.all(liquid)
        volumes = self.operators.discretization.cell_volumes.astype(value.dtype)
        mean = jnp.sum(volumes * value) / jnp.sum(volumes)
        restricted = jnp.where(
            all_liquid,
            value - mean,
            jnp.where(liquid, value, 0.0),
        )
        gradient = self.boundaries.pressure_gradient(restricted, None, homogeneous=True)
        weighted = tuple(
            coefficient * derivative
            for coefficient, derivative in zip(
                self.face_inverse_momentum, gradient, strict=True
            )
        )
        core = -self.operators.divergence(weighted)
        return jnp.where(
            all_liquid,
            core + mean,
            jnp.where(liquid, core, value),
        )


class MACFreeSurfaceProjectionResult(StrictModule):
    velocity: FaceVelocity
    pressure: Array
    pressure_increment: Array
    liquid_mask: Array
    divergence_before: Array
    divergence_after: Array
    pressure_residual: Array
    active_divergence_norm: Array
    residual_norm: Array
    air_pressure_defect: Array
    energy_before: Array
    energy_after: Array
    energy_increase: Array
    liquid_count: Array
    air_count: Array
    linear: LinearSolveResult | None
    finite: Array
    converged: Array
    successful: Array
    route: Array
    projection_id: str = eqx.field(static=True)


class MACFreeSurfaceProjectionPlan(StrictModule, NonTrainableState):
    """Compatible MAC projection with runtime liquid and atmospheric air cells."""
    operators: PreparedMACOperators
    boundaries: PreparedMACBoundaryPlan
    density: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    prepared_linear: PreparedLinearSolve
    operator_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        /,
        *,
        boundaries: PreparedMACBoundaryPlan | MACBoundaryPlan | None = None,
        density: float = 1.0,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 500,
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
            raise TypeError("boundaries must be prepared MAC boundaries, a plan, or None.")
        if boundaries_.operators.prepared_id != operators.prepared_id:
            raise ValueError("Free-surface projection boundaries use another MAC grid.")
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
        shape = operators.discretization.cell_shape
        representative_mask = jnp.ones(shape, dtype=bool).at[(0,) * len(shape)].set(False)
        unit_face = tuple(
            jnp.ones(layout.shape, dtype=operators.pressure_space.dtype)
            for layout in operators.discretization.face_layouts
        )
        operator_id = canonical_fingerprint(
            {
                "kind": "mac-free-surface-pressure-operator",
                "operators": operators.prepared_id,
                "boundaries": boundaries_.prepared_id,
            }
        )
        action = _MaskedMACPressureAction(
            operators, boundaries_, unit_face, representative_mask
        )
        operator = FunctionLinearOperator(
            action,
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
            {"kind": "mac-free-surface-pressure-system", "operator": operator_id}
        )
        problem = LinearSystem(operator, problem_id=problem_id)
        policy = (
            LinearSolvePolicy(
                ConjugateGradient(),
                tolerance=TolerancePolicy(
                    relative=0.1 * tolerance_,
                    absolute=0.1 * tolerance_,
                    max_steps=iterations,
                ),
            )
            if linear_policy is None
            else linear_policy
        )
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be LinearSolvePolicy or None.")
        self.operators = operators
        self.boundaries = boundaries_
        self.density = density_
        self.tolerance = tolerance_
        self.linear_policy = policy
        self.prepared_linear = prepare(problem, policy)
        self.operator_id = operator_id
        self.problem_id = problem_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-free-surface-projection",
                "operators": operators.prepared_id,
                "boundaries": boundaries_.prepared_id,
                "density": density_,
                "linear": self.prepared_linear.plan.plan_id,
            }
        )

    def project(
        self,
        velocity: FaceVelocity,
        liquid_mask: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        pressure: ArrayLike | None = None,
        boundary_stage: MACBoundaryStageData | None = None,
    ) -> MACFreeSurfaceProjectionResult:
        values = self.operators.validate_velocity(velocity)
        liquid = jnp.asarray(liquid_mask, dtype=bool)
        if liquid.shape != self.operators.discretization.cell_shape:
            raise ValueError("liquid_mask must match the MAC cell shape.")
        dtype = self.operators.pressure_space.dtype
        dt = jnp.asarray(step_size, dtype=dtype).reshape(())
        stage = (
            self.boundaries.evaluate(jnp.asarray(0.0, dtype=dtype), None)
            if boundary_stage is None
            else self.boundaries.validate_stage(boundary_stage)
        )
        incoming = (
            jnp.zeros(self.operators.discretization.cell_shape, dtype=dtype)
            if pressure is None
            else self.operators.validate_pressure(pressure)
        )
        liquid_count = jnp.sum(liquid, dtype=jnp.int32)
        air_count = liquid.size - liquid_count
        all_liquid = jnp.all(liquid)
        any_liquid = jnp.any(liquid)


        def atmospheric(_):
            coefficient_cell = jnp.full(
                self.operators.discretization.cell_shape,
                dt / self.density,
                dtype=dtype,
            )
            face_inverse = self.operators.interpolate_inverse_momentum(coefficient_cell)
            divergence_before = self.operators.divergence(values)
            boundary_gradient = self.boundaries.pressure_gradient(
                jnp.zeros_like(divergence_before), stage, homogeneous=False
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
            rhs = jnp.where(
                all_liquid,
                self.operators.compatibility_project(raw_rhs),
                jnp.where(liquid, raw_rhs, 0.0),
            )
            incoming_ = jnp.where(
                all_liquid,
                self.operators.gauge_project(incoming),
                jnp.where(liquid, incoming, 0.0),
            )
            action = _MaskedMACPressureAction(
                self.operators, self.boundaries, face_inverse, liquid
            )
            operator = FunctionLinearOperator(
                action,
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
            prepared = refresh(
                self.prepared_linear,
                LinearSystem(operator, problem_id=self.problem_id),
            )
            linear = solve(prepared, rhs, initial_guess=incoming_)
            pressure_candidate = jnp.where(
                all_liquid,
                self.operators.gauge_project(linear.value),
                jnp.where(liquid, linear.value, 0.0),
            )
            gradient = self.boundaries.pressure_gradient(
                pressure_candidate, stage, homogeneous=False
            )
            corrected_candidate = tuple(
                component - coefficient * derivative
                for component, coefficient, derivative in zip(
                    values, face_inverse, gradient, strict=True
                )
            )
            corrected_candidate = self.boundaries.enforce(corrected_candidate, stage)
            residual = action(pressure_candidate) - rhs
            divergence_candidate = self.operators.divergence(corrected_candidate)
            volumes = self.operators.discretization.cell_volumes.astype(dtype)
            residual_norm = jnp.sqrt(jnp.sum(volumes * residual**2))
            divergence_norm = jnp.sqrt(
                jnp.sum(volumes * jnp.where(liquid, divergence_candidate, 0.0) ** 2)
            )
            rhs_norm = jnp.sqrt(jnp.sum(volumes * rhs**2))
            air_defect = jnp.max(
                jnp.where(~liquid, jnp.abs(pressure_candidate), 0.0), initial=0.0
            )
            finite = (
                stage.finite
                & jnp.all(jnp.isfinite(pressure_candidate))
                & jnp.all(
                    jnp.stack(
                        tuple(jnp.all(jnp.isfinite(value)) for value in corrected_candidate)
                    )
                )
                & jnp.isfinite(residual_norm)
                & jnp.isfinite(divergence_norm)
            )
            converged = (
                any_liquid
                & stage.successful
                & finite
                & (
                    residual_norm
                    <= 10.0 * self.tolerance * jnp.maximum(rhs_norm, 1.0)
                )
                & (
                    divergence_norm
                    <= 10.0 * self.tolerance * jnp.maximum(rhs_norm, 1.0)
                )
                & (air_defect <= self.tolerance)
            )
            corrected = tuple(
                jnp.where(converged, candidate, original)
                for candidate, original in zip(corrected_candidate, values, strict=True)
            )
            pressure_value = jnp.where(converged, pressure_candidate, incoming_)
            divergence_after = self.operators.divergence(corrected)
            energy_before = 0.5 * jnp.real(self.operators.velocity_space.inner(values, values))
            energy_after = 0.5 * jnp.real(
                self.operators.velocity_space.inner(corrected, corrected)
            )
            return MACFreeSurfaceProjectionResult(
                corrected,
                pressure_value,
                pressure_value - incoming_,
                liquid,
                divergence_before,
                divergence_after,
                residual,
                divergence_norm,
                residual_norm,
                air_defect,
                energy_before,
                energy_after,
                jnp.maximum(energy_after - energy_before, 0.0),
                liquid_count,
                air_count,
                linear,
                finite,
                converged,
                converged,
                jnp.where(all_liquid, 0, 1).astype(jnp.int32),
                self.plan_id,
            )

        return atmospheric(None)


__all__ = ["MACFreeSurfaceProjectionPlan", "MACFreeSurfaceProjectionResult"]
