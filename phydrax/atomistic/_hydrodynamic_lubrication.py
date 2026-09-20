#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)
from ._hydrodynamic_mobility import (
    AbstractPreparedHydrodynamicMobility,
    materialize_mobility,
)


class HardSphereLubricationPlan(StrictModule):
    hydrodynamic_radius: float = eqx.field(static=True)
    dynamic_viscosity: float = eqx.field(static=True)
    activation_gap: float = eqx.field(static=True)
    minimum_gap: float = eqx.field(static=True)
    maximum_dofs: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        hydrodynamic_radius: float,
        dynamic_viscosity: float,
        activation_gap: float,
        minimum_gap: float,
        /,
        *,
        maximum_dofs: int,
    ):
        values = (
            hydrodynamic_radius,
            dynamic_viscosity,
            activation_gap,
            minimum_gap,
        )
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0 for value in values
        ):
            raise ValueError("Lubrication material and gap scales must be positive.")
        if float(minimum_gap) >= float(activation_gap):
            raise ValueError("minimum_gap must be smaller than activation_gap.")
        if int(maximum_dofs) <= 0:
            raise ValueError("maximum_dofs must be positive.")
        self.hydrodynamic_radius = float(hydrodynamic_radius)
        self.dynamic_viscosity = float(dynamic_viscosity)
        self.activation_gap = float(activation_gap)
        self.minimum_gap = float(minimum_gap)
        self.maximum_dofs = int(maximum_dofs)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hard-sphere-normal-lubrication",
                "radius": self.hydrodynamic_radius,
                "viscosity": self.dynamic_viscosity,
                "activation_gap": self.activation_gap,
                "minimum_gap": self.minimum_gap,
                "maximum_dofs": self.maximum_dofs,
            }
        )


class HardSphereLubricationResult(StrictModule):
    uncorrected_velocity: Array
    corrected_velocity: Array
    resistance_matrix: Array
    correction_system: Array
    pair_count: Array
    minimum_surface_gap: Array
    solve_residual: Array
    gap_admissible: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    mobility_id: str = eqx.field(static=True)


def _resistance_matrix(
    plan: HardSphereLubricationPlan, positions: Array
) -> tuple[Array, Array, Array, Array]:
    count = positions.shape[0]
    displacement = positions[:, None, :] - positions[None, :, :]
    squared = jnp.sum(displacement * displacement, axis=-1)
    identity_pair = jnp.eye(count, dtype=jnp.bool_)
    distance = jnp.sqrt(jnp.where(identity_pair, 1.0, squared))
    direction = jnp.where(
        identity_pair[..., None], 0.0, displacement / distance[..., None]
    )
    gap = distance - 2.0 * plan.hydrodynamic_radius
    active_pair = (~identity_pair) & (gap < plan.activation_gap) & (gap > 0.0)
    safe_gap = jnp.maximum(gap, plan.minimum_gap)
    coefficient = (
        3.0 * jnp.pi * plan.dynamic_viscosity * plan.hydrodynamic_radius**2 / 2.0
    ) * (1.0 / safe_gap - 1.0 / plan.activation_gap)
    coefficient = jnp.where(active_pair, jnp.maximum(coefficient, 0.0), 0.0)
    outer = direction[..., :, None] * direction[..., None, :]
    pair_blocks = coefficient[..., None, None] * outer
    blocks = -pair_blocks
    diagonal = jnp.sum(pair_blocks, axis=1)
    indices = jnp.arange(count)
    blocks = blocks.at[indices, indices].set(diagonal)
    matrix = jnp.transpose(blocks, (0, 2, 1, 3)).reshape((3 * count, 3 * count))
    upper = jnp.triu(jnp.ones((count, count), dtype=jnp.bool_), 1)
    pair_count = jnp.sum((active_pair & upper).astype(jnp.int32))
    minimum = jnp.min(jnp.where(upper, gap, jnp.inf))
    admissible = jnp.all(jnp.where(upper, gap >= plan.minimum_gap, True))
    return matrix, pair_count, minimum, admissible


def hard_sphere_lubrication_correction(
    plan: HardSphereLubricationPlan,
    mobility: AbstractPreparedHydrodynamicMobility,
    positions: ArrayLike,
    forces: ArrayLike,
    /,
) -> HardSphereLubricationResult:
    if not isinstance(plan, HardSphereLubricationPlan):
        raise TypeError("plan must be HardSphereLubricationPlan.")
    if not isinstance(mobility, AbstractPreparedHydrodynamicMobility):
        raise TypeError("mobility must be an AbstractPreparedHydrodynamicMobility.")
    position = mobility.coordinate_space.validate(jnp.asarray(positions))
    force = mobility.coordinate_space.validate(jnp.asarray(forces))
    if mobility.coordinate_space.size > plan.maximum_dofs:
        raise ValueError("Lubrication correction exceeds maximum_dofs.")
    far = materialize_mobility(mobility, position, maximum_dofs=plan.maximum_dofs)
    resistance, pair_count, minimum_gap, gap_admissible = _resistance_matrix(
        plan, position
    )
    flat_force = mobility.coordinate_space.flatten(force)
    uncorrected_flat = contract("ij,j->i", far, flat_force)
    system = jnp.eye(far.shape[0], dtype=far.dtype) + contract(
        "ij,jk->ik", far, resistance
    )
    solved = solve(
        LinearSystem(DenseLinearOperator(system)),
        uncorrected_flat,
        policy=LinearSolvePolicy(DenseLU()),
    )
    residual = contract("ij,j->i", system, solved.value) - uncorrected_flat
    residual_norm = jnp.sqrt(jnp.sum(residual * residual))
    finite = (
        jnp.all(jnp.isfinite(resistance))
        & jnp.all(jnp.isfinite(solved.value))
        & jnp.isfinite(residual_norm)
    )
    successful = finite & gap_admissible & (solved.status == 0)
    return HardSphereLubricationResult(
        mobility.coordinate_space.unflatten(uncorrected_flat),
        mobility.coordinate_space.unflatten(solved.value),
        resistance,
        system,
        pair_count,
        minimum_gap,
        residual_norm,
        gap_admissible,
        finite,
        successful,
        plan.plan_id,
        mobility.prepared_id,
    )


__all__ = [
    "HardSphereLubricationPlan",
    "HardSphereLubricationResult",
    "hard_sphere_lubrication_correction",
]
