#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._system import PreparedAtomisticSystem


class DistanceConstraintPlan(StrictModule, NonTrainableState):
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_iterations: int = 32,
        tolerance: float = 1.0e-10,
    ):
        iterations = int(maximum_iterations)
        threshold = float(tolerance)
        if iterations <= 0 or not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("Constraint iterations and tolerance must be positive.")
        self.maximum_iterations = iterations
        self.tolerance = threshold
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distance-constraint-plan",
                "maximum_iterations": iterations,
                "tolerance": threshold,
            }
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedDistanceConstraints":
        return PreparedDistanceConstraints(self, system)


class ConstraintProjection(StrictModule):
    positions: Array
    momenta: Array
    multipliers: Array
    position_residual: Array
    velocity_residual: Array
    iterations: Array
    successful: Array


class PreparedDistanceConstraints(StrictModule, NonTrainableState):
    plan: DistanceConstraintPlan
    system: PreparedAtomisticSystem
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: DistanceConstraintPlan, system: PreparedAtomisticSystem, /):
        if not isinstance(plan, DistanceConstraintPlan):
            raise TypeError("plan must be DistanceConstraintPlan.")
        if not isinstance(system, PreparedAtomisticSystem):
            raise TypeError("system must be PreparedAtomisticSystem.")
        self.plan = plan
        self.system = system
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-distance-constraints",
                "plan": plan.plan_id,
                "system": system.prepared_id,
                "topology": system.topology.topology_id,
            }
        )

    def project_positions(
        self,
        previous_positions: ArrayLike,
        proposed_positions: ArrayLike,
        momenta: ArrayLike,
        /,
    ) -> ConstraintProjection:
        previous = jnp.asarray(previous_positions)
        proposed = jnp.asarray(proposed_positions, dtype=previous.dtype)
        momentum = jnp.asarray(momenta, dtype=previous.dtype)
        indices = self.system.topology.constraint_indices
        targets = self.system.topology.constraint_distances.astype(previous.dtype)
        count = int(indices.shape[0])
        if count == 0:
            zero = jnp.zeros((), dtype=previous.dtype)
            return ConstraintProjection(
                proposed,
                momentum,
                jnp.zeros((0,), dtype=previous.dtype),
                zero,
                zero,
                jnp.zeros((), dtype=jnp.int32),
                jnp.asarray(True),
            )
        inverse_mass = self.system.inverse_masses.astype(previous.dtype)
        left = indices[:, 0]
        right = indices[:, 1]

        def iteration(_, carry):
            position, multipliers = carry
            displacement = position[left] - position[right]
            squared = jnp.sum(displacement * displacement, axis=-1)
            residual = squared - targets * targets
            denominator = 2.0 * (inverse_mass[left] + inverse_mass[right]) * squared
            valid = denominator > 0.0
            increment = jnp.where(valid, -residual / denominator, 0.0)
            correction = increment[:, None] * displacement
            delta = jnp.zeros_like(position)
            delta = delta.at[left].add(inverse_mass[left, None] * correction)
            delta = delta.at[right].add(-inverse_mass[right, None] * correction)
            position = position + delta
            return position, multipliers + increment

        positions, multipliers = jax.lax.fori_loop(
            0,
            self.plan.maximum_iterations,
            iteration,
            (proposed, jnp.zeros((count,), dtype=previous.dtype)),
        )
        displacement = positions[left] - positions[right]
        distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
        position_residual = jnp.max(jnp.abs(distance - targets))
        projected_momenta, velocity_residual = self.project_momenta(positions, momentum)
        successful = (
            jnp.all(jnp.isfinite(positions))
            & jnp.all(jnp.isfinite(projected_momenta))
            & (position_residual <= self.plan.tolerance)
            & (velocity_residual <= self.plan.tolerance)
        )
        return ConstraintProjection(
            positions,
            projected_momenta,
            multipliers,
            position_residual,
            velocity_residual,
            jnp.asarray(self.plan.maximum_iterations, dtype=jnp.int32),
            successful,
        )

    def project_momenta(
        self, positions: ArrayLike, momenta: ArrayLike, /
    ) -> tuple[Array, Array]:
        position = jnp.asarray(positions)
        momentum = jnp.asarray(momenta, dtype=position.dtype)
        indices = self.system.topology.constraint_indices
        count = int(indices.shape[0])
        if count == 0:
            return momentum, jnp.zeros((), dtype=position.dtype)
        inverse_mass = self.system.inverse_masses.astype(position.dtype)
        left = indices[:, 0]
        right = indices[:, 1]

        def iteration(_, value):
            displacement = position[left] - position[right]
            relative_velocity = (
                value[left] * inverse_mass[left, None]
                - value[right] * inverse_mass[right, None]
            )
            squared = jnp.sum(displacement * displacement, axis=-1)
            denominator = (inverse_mass[left] + inverse_mass[right]) * squared
            increment = jnp.where(
                denominator > 0.0,
                -jnp.sum(displacement * relative_velocity, axis=-1) / denominator,
                0.0,
            )
            correction = increment[:, None] * displacement
            delta = jnp.zeros_like(value)
            delta = delta.at[left].add(correction)
            delta = delta.at[right].add(-correction)
            return value + delta

        projected = jax.lax.fori_loop(
            0, self.plan.maximum_iterations, iteration, momentum
        )
        displacement = position[left] - position[right]
        relative_velocity = (
            projected[left] * inverse_mass[left, None]
            - projected[right] * inverse_mass[right, None]
        )
        residual = jnp.max(jnp.abs(jnp.sum(displacement * relative_velocity, axis=-1)))
        return projected, residual


__all__ = [
    "ConstraintProjection",
    "DistanceConstraintPlan",
    "PreparedDistanceConstraints",
]
