#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

import phydrax.ein as ein

from ..._strict import StrictModule


class MPMNormalizedGrid(StrictModule):
    mass: Array
    momentum: Array
    velocity: Array
    active: Array
    mass_tolerance: Array


class MPMGridAdvance(StrictModule):
    acceleration: Array
    velocity: Array
    maximum_acceleration: Array


def normalize_grid_momentum(
    mass: Array,
    momentum: Array,
    /,
    *,
    mass_tolerance_factor: float,
) -> MPMNormalizedGrid:
    maximum_mass = jnp.max(mass, initial=0.0)
    tolerance = (
        mass_tolerance_factor * jnp.finfo(mass.dtype).eps * jnp.maximum(maximum_mass, 1.0)
    )
    active = mass > tolerance
    denominator = jnp.where(active, mass, 1.0)
    velocity = jnp.where(active[..., None], momentum / denominator[..., None], 0.0)
    return MPMNormalizedGrid(mass, momentum, velocity, active, tolerance)


def advance_grid_velocity(
    grid: MPMNormalizedGrid,
    internal_force: Array,
    external_force: Array,
    step_size: Array,
    /,
) -> MPMGridAdvance:
    denominator = jnp.where(grid.active, grid.mass, 1.0)
    acceleration = jnp.where(
        grid.active[..., None],
        (internal_force + external_force) / denominator[..., None],
        0.0,
    )
    velocity = grid.velocity + step_size * acceleration
    maximum_acceleration = jnp.max(
        jnp.where(
            grid.active,
            jnp.sqrt(jnp.sum(acceleration * acceleration, axis=-1)),
            0.0,
        ),
        initial=0.0,
    )
    return MPMGridAdvance(acceleration, velocity, maximum_acceleration)


def update_deformation(
    deformation_gradient: Array,
    velocity_gradient: Array,
    step_size: Array,
    /,
) -> Array:
    dimension = int(deformation_gradient.shape[-1])
    identity = jnp.broadcast_to(
        jnp.eye(dimension, dtype=deformation_gradient.dtype),
        deformation_gradient.shape,
    )
    return ein.contract(
        "pij,pjk->pik",
        identity + step_size * velocity_gradient,
        deformation_gradient,
    )


__all__ = [
    "MPMGridAdvance",
    "MPMNormalizedGrid",
    "advance_grid_velocity",
    "normalize_grid_momentum",
    "update_deformation",
]
