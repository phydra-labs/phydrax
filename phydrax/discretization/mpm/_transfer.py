#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from ..._interpolation import gather_patches
from ..._numerics._compensated import compensated_sum
from ..._strict import StrictModule
from ...discretization.splatting import ParticleGridSplatState
from ...linalg import SmallLinearSolvePlan, solve_small_linear


class APICGatherResult(StrictModule):
    velocity: Array
    velocity_gradient: Array
    affine_moment: Array
    particle_moment: Array
    affine_velocity: Array
    condition_estimate: Array
    successful: Array


def _cross(left: Array, right: Array, dimension: int, /) -> Array:
    if dimension == 1:
        return jnp.zeros(left.shape[:-1], dtype=left.dtype)
    if dimension == 2:
        return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]
    return jnp.cross(left, right)


def apic_particle_kinetic_energy(
    mass: Array,
    velocity: Array,
    affine_velocity: Array,
    particle_moment: Array,
    active: Array,
    /,
) -> Array:
    translational = jnp.sum(velocity * velocity, axis=-1)
    affine = oe.contract(
        "pij,pjk,pik->p",
        affine_velocity,
        particle_moment,
        affine_velocity,
    )
    terms = 0.5 * mass * (translational + affine)
    return compensated_sum(jnp.where(active, terms, 0.0))


def apic_particle_angular_momentum(
    position: Array,
    velocity: Array,
    affine_velocity: Array,
    mass: Array,
    state: ParticleGridSplatState,
    active: Array,
    /,
) -> Array:
    dimension = int(position.shape[-1])
    orbital = _cross(position, mass[:, None] * velocity, dimension)
    affine_route_velocity = oe.contract(
        "pij,prj->pri", affine_velocity, state.route_offsets
    )
    affine_route_momentum = (
        mass[:, None, None] * state.stencil.weights[..., None] * affine_route_velocity
    )
    affine = compensated_sum(
        jnp.where(
            state.stencil.valid.reshape(
                state.stencil.valid.shape + (1,) * (dimension == 3)
            ),
            _cross(state.route_offsets, affine_route_momentum, dimension),
            0.0,
        ),
        axis=(0, 1),
    )
    return (
        compensated_sum(
            jnp.where(active[..., None] if dimension == 3 else active, orbital, 0.0),
            axis=0,
        )
        + affine
    )


def grid_angular_momentum(
    coordinates: Array,
    momentum: Array,
    active: Array,
    /,
) -> Array:
    dimension = int(coordinates.shape[-1])
    values = _cross(coordinates, momentum, dimension)
    mask = active[..., None] if dimension == 3 else active
    return compensated_sum(jnp.where(mask, values, 0.0), axis=0)


def build_apic_route_payload(
    state: ParticleGridSplatState,
    mass: Array,
    velocity: Array,
    affine_velocity: Array,
    reference_volume: Array,
    first_piola: Array,
    deformation_gradient: Array,
    external_acceleration: Array,
    active: Array,
    /,
) -> Array:
    affine_route_velocity = oe.contract(
        "pij,prj->pri", affine_velocity, state.route_offsets
    )
    weights = state.stencil.weights
    route_momentum = (
        weights[..., None]
        * mass[:, None, None]
        * (velocity[:, None, :] + affine_route_velocity)
    )
    kirchhoff = oe.contract("pij,pkj->pik", first_piola, deformation_gradient)
    internal_force = -reference_volume[:, None, None] * oe.contract(
        "pij,prj->pri", kirchhoff, state.weight_gradients
    )
    external_force = (
        weights[..., None] * mass[:, None, None] * external_acceleration[:, None, :]
    )
    mask = active[:, None, None] & state.stencil.valid[..., None]
    return jnp.where(
        mask,
        jnp.concatenate((route_momentum, internal_force, external_force), axis=-1),
        0.0,
    )


def gather_apic(
    state: ParticleGridSplatState,
    grid_velocity: Array,
    active: Array,
    maximum_condition: float,
    /,
) -> APICGatherResult:
    dimension = int(grid_velocity.shape[-1])
    gathered, route_valid = gather_patches(grid_velocity, state.stencil)
    mask = route_valid[..., None]
    route_velocity = jnp.where(mask, gathered, 0.0)
    weights = jnp.where(route_valid, state.stencil.weights, 0.0)
    velocity = oe.contract("pr,pri->pi", weights, route_velocity)
    velocity_gradient = oe.contract(
        "pri,prj->pij", route_velocity, state.weight_gradients
    )
    affine_moment = oe.contract(
        "pr,pri,prj->pij", weights, route_velocity, state.route_offsets
    )
    particle_moment = state.second_moments
    identity = jnp.broadcast_to(
        jnp.eye(dimension, dtype=particle_moment.dtype), particle_moment.shape
    )
    safe_moment = jnp.where(active[:, None, None], particle_moment, identity)
    solve = solve_small_linear(
        SmallLinearSolvePlan(dimension),
        safe_moment,
        affine_moment.swapaxes(-1, -2),
    )
    affine_velocity = solve.value.swapaxes(-1, -2)
    successful = jnp.all(
        (~active)
        | (
            solve.successful
            & jnp.isfinite(solve.condition_estimate)
            & (solve.condition_estimate <= maximum_condition)
        )
    )
    velocity = jnp.where(active[:, None], velocity, 0.0)
    velocity_gradient = jnp.where(active[:, None, None], velocity_gradient, 0.0)
    affine_velocity = jnp.where(active[:, None, None], affine_velocity, 0.0)
    return APICGatherResult(
        velocity,
        velocity_gradient,
        affine_moment,
        particle_moment,
        affine_velocity,
        solve.condition_estimate,
        successful,
    )


__all__ = [
    "APICGatherResult",
    "apic_particle_angular_momentum",
    "apic_particle_kinetic_energy",
    "build_apic_route_payload",
    "gather_apic",
    "grid_angular_momentum",
]
