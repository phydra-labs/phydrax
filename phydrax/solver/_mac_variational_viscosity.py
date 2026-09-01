#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume import (
    FaceVelocity,
    MACFreeSurfaceViscousMeasures,
    PreparedMACOperators,
)
from ..linalg import (
    ConjugateGradient,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    prepare,
    solve,
    TolerancePolicy,
)


def _face_to_cell(value, axis, cell_shape):
    if value.shape[axis] == cell_shape[axis]:
        return 0.5 * (value + jnp.roll(value, 1, axis=axis))
    lower = jnp.take(value, jnp.arange(cell_shape[axis]), axis=axis)
    upper = jnp.take(value, jnp.arange(1, cell_shape[axis] + 1), axis=axis)
    return 0.5 * (lower + upper)


class _VariationalViscousAction(StrictModule):
    operators: PreparedMACOperators
    measures: MACFreeSurfaceViscousMeasures
    step_size: Array

    def _dissipation(self, velocity: FaceVelocity, /) -> Array:
        values = self.operators.validate_velocity(velocity)
        cell_shape = self.operators.discretization.cell_shape
        cell_velocity = tuple(
            _face_to_cell(value, axis, cell_shape) for axis, value in enumerate(values)
        )
        axes = self.operators.discretization.grid.structured_axes
        gradients = []
        for component in range(len(values)):
            component_gradients = []
            for axis, grid_axis in enumerate(axes):
                spacing = jnp.mean(grid_axis.interval_widths)
                derivative = (
                    jnp.roll(cell_velocity[component], -1, axis=axis)
                    - jnp.roll(cell_velocity[component], 1, axis=axis)
                ) / (2.0 * spacing)
                component_gradients.append(derivative)
            gradients.append(tuple(component_gradients))
        strain_squared = jnp.zeros(cell_shape, dtype=values[0].dtype)
        for i in range(len(values)):
            for j in range(len(values)):
                strain = 0.5 * (gradients[i][j] + gradients[j][i])
                strain_squared = strain_squared + strain * strain
        volumes = self.operators.discretization.cell_volumes.astype(values[0].dtype)
        return jnp.sum(
            2.0
            * self.measures.cell_viscosity
            * self.measures.cell_fraction
            * strain_squared
            * volumes
        )

    def __call__(self, velocity: FaceVelocity, /) -> FaceVelocity:
        values = self.operators.validate_velocity(velocity)
        viscous = jax.grad(lambda selected: 0.5 * self._dissipation(selected))(values)
        return tuple(
            mass * value + self.step_size * force
            for mass, value, force in zip(
                self.measures.face_mass, values, viscous, strict=True
            )
        )


class MACVariationalViscosityResult(StrictModule):
    velocity: FaceVelocity
    dissipation: Array
    energy_before: Array
    energy_after: Array
    energy_increase: Array
    residual_norm: Array
    finite: Array
    converged: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class MACVariationalViscosityPlan(StrictModule, NonTrainableState):
    operators: PreparedMACOperators
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        /,
        *,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 500,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        self.operators = operators
        self.tolerance = float(tolerance)
        self.maximum_iterations = int(maximum_iterations)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-variational-viscosity",
                "operators": operators.prepared_id,
                "tolerance": float(tolerance),
                "maximum_iterations": int(maximum_iterations),
            }
        )

    def solve(
        self,
        velocity: FaceVelocity,
        measures: MACFreeSurfaceViscousMeasures,
        step_size: ArrayLike,
        /,
        *,
        wall_rhs: FaceVelocity | None = None,
    ) -> MACVariationalViscosityResult:
        values = self.operators.validate_velocity(velocity)
        dt = jnp.asarray(step_size, dtype=values[0].dtype).reshape(())
        action = _VariationalViscousAction(self.operators, measures, dt)
        operator = FunctionLinearOperator(
            action,
            source=self.operators.velocity_space,
            target=self.operators.velocity_space,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
            operator_id=canonical_fingerprint(
                {
                    "kind": "mac-variational-viscosity-step",
                    "plan": self.plan_id,
                    "measure": measures.measure_id,
                }
            ),
        )
        rhs = tuple(
            mass * value for mass, value in zip(measures.face_mass, values, strict=True)
        )
        if wall_rhs is not None:
            wall = self.operators.validate_velocity(wall_rhs)
            rhs = tuple(left + right for left, right in zip(rhs, wall, strict=True))
        policy = LinearSolvePolicy(
            ConjugateGradient(),
            tolerance=TolerancePolicy(
                relative=self.tolerance,
                absolute=self.tolerance,
                max_steps=self.maximum_iterations,
            ),
        )
        linear = solve(prepare(LinearSystem(operator), policy), rhs, initial_guess=values)
        candidate = linear.value
        residual = action(candidate)
        residual = tuple(left - right for left, right in zip(residual, rhs, strict=True))
        residual_norm = jnp.sqrt(sum(jnp.sum(value**2) for value in residual))
        dissipation = action._dissipation(candidate)
        energy_before = 0.5 * sum(
            jnp.sum(mass * value**2)
            for mass, value in zip(measures.face_mass, values, strict=True)
        )
        energy_after = 0.5 * sum(
            jnp.sum(mass * value**2)
            for mass, value in zip(measures.face_mass, candidate, strict=True)
        )
        increase = jnp.maximum(energy_after - energy_before, 0.0)
        finite = (
            measures.finite
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in candidate))
            )
            & jnp.isfinite(residual_norm + dissipation + increase)
        )
        converged = linear.successful & (
            residual_norm
            <= self.tolerance
            * jnp.maximum(1.0, jnp.sqrt(sum(jnp.sum(value**2) for value in rhs)))
        )
        successful = (
            measures.successful
            & finite
            & converged
            & (increase <= 100.0 * self.tolerance)
        )
        accepted = tuple(
            jnp.where(successful, proposed, old)
            for proposed, old in zip(candidate, values, strict=True)
        )
        return MACVariationalViscosityResult(
            accepted,
            dissipation,
            energy_before,
            energy_after,
            increase,
            residual_norm,
            finite,
            converged,
            successful,
            self.plan_id,
        )


__all__ = ["MACVariationalViscosityPlan", "MACVariationalViscosityResult"]
