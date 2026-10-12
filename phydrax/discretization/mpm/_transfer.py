#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

import phydrax.ein as ein

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
    *,
    first_moment: Array | None = None,
) -> Array:
    translational = jnp.sum(velocity * velocity, axis=-1)
    affine = ein.contract(
        "pij,pjk,pik->p",
        affine_velocity,
        particle_moment,
        affine_velocity,
    )
    if first_moment is not None:
        affine = affine + 2.0 * ein.contract(
            "pi,pij,pj->p", velocity, affine_velocity, first_moment
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
    dimension = position.shape[-1]
    orbital = _cross(position, mass[:, None] * velocity, dimension)
    affine_route_velocity = ein.contract(
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
    dimension = coordinates.shape[-1]
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
    *,
    physical_deformation_gradient: Array | None = None,
) -> Array:
    affine_route_velocity = ein.contract(
        "pij,prj->pri", affine_velocity, state.route_offsets
    )
    weights = state.stencil.weights
    route_momentum = (
        weights[..., None]
        * mass[:, None, None]
        * (velocity[:, None, :] + affine_route_velocity)
    )
    kirchhoff = ein.contract("pij,pkj->pik", first_piola, deformation_gradient)
    if physical_deformation_gradient is not None:
        dimension = deformation_gradient.shape[-1]
        identity = jnp.broadcast_to(
            jnp.eye(dimension, dtype=deformation_gradient.dtype),
            deformation_gradient.shape,
        )
        plan = SmallLinearSolvePlan(dimension)
        assumed_j = solve_small_linear(plan, deformation_gradient, identity).determinant
        physical_j = solve_small_linear(
            plan, physical_deformation_gradient, identity
        ).determinant
        reference_volume = reference_volume * physical_j / assumed_j
    spatial_dimension = state.weight_gradients.shape[-1]
    internal_force = -reference_volume[:, None, None] * ein.contract(
        "pij,prj->pri",
        kirchhoff[:, :spatial_dimension, :spatial_dimension],
        state.weight_gradients,
    )
    if state.hoop_gradients is not None:
        # Current-ring virtual work V*sigma:L equals V0*tau:L.
        hoop = reference_volume[:, None] * kirchhoff[:, 2, 2, None] * state.hoop_gradients
        internal_force = internal_force.at[..., 0].add(-hoop)
    if state.cell_average_gradients is not None:
        mean_stress = jnp.trace(kirchhoff, axis1=-2, axis2=-1) / kirchhoff.shape[-1]
        internal_force = internal_force - (
            reference_volume[:, None, None]
            * mean_stress[:, None, None]
            * (state.cell_average_gradients - state.weight_gradients)
        )
        if state.hoop_gradients is not None:
            if state.cell_average_hoop_gradients is None:
                raise RuntimeError(
                    "Axisymmetric B-bar requires the cell-average hoop operator."
                )
            internal_force = internal_force.at[..., 0].add(
                -reference_volume[:, None]
                * mean_stress[:, None]
                * (state.cell_average_hoop_gradients - state.hoop_gradients)
            )
    external_force = (
        weights[..., None] * mass[:, None, None] * external_acceleration[:, None, :]
    )
    if state.radial_parity is not None:
        route_momentum = route_momentum.at[..., 0].multiply(state.radial_parity)
        internal_force = internal_force.at[..., 0].multiply(state.radial_parity)
        external_force = external_force.at[..., 0].multiply(state.radial_parity)
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
    dimension = grid_velocity.shape[-1]
    gathered, route_valid = gather_patches(grid_velocity, state.stencil)
    mask = route_valid[..., None]
    route_velocity = jnp.where(mask, gathered, 0.0)
    weights = jnp.where(route_valid, state.stencil.weights, 0.0)
    if state.radial_parity is not None:
        route_velocity = route_velocity.at[..., 0].multiply(state.radial_parity)
    velocity = ein.contract("pr,pri->pi", weights, route_velocity)
    velocity_gradient = ein.contract(
        "pri,prj->pij", route_velocity, state.weight_gradients
    )
    if state.hoop_gradients is None and state.cell_average_gradients is not None:
        projected_divergence = ein.contract(
            "pri,pri->p", route_velocity, state.cell_average_gradients
        )
        correction = (
            projected_divergence - jnp.trace(velocity_gradient, axis1=-2, axis2=-1)
        ) / dimension
        velocity_gradient = velocity_gradient + (
            correction[:, None, None]
            * jnp.eye(dimension, dtype=velocity_gradient.dtype)[None, :, :]
        )
    if state.hoop_gradients is not None:
        hoop_rate = ein.contract("pr,pr->p", route_velocity[..., 0], state.hoop_gradients)
        constitutive_gradient = jnp.zeros(
            (velocity_gradient.shape[0], 3, 3), dtype=velocity_gradient.dtype
        )
        velocity_gradient = (
            constitutive_gradient.at[:, :2, :2]
            .set(velocity_gradient)
            .at[:, 2, 2]
            .set(hoop_rate)
        )
        if state.cell_average_gradients is not None:
            if state.cell_average_hoop_gradients is None:
                raise RuntimeError(
                    "Axisymmetric B-bar requires the cell-average hoop operator."
                )
            projected_divergence = ein.contract(
                "pri,pri->p", route_velocity, state.cell_average_gradients
            ) + ein.contract(
                "pr,pr->p", route_velocity[..., 0], state.cell_average_hoop_gradients
            )
            correction = (
                projected_divergence - jnp.trace(velocity_gradient, axis1=-2, axis2=-1)
            ) / 3.0
            velocity_gradient = velocity_gradient + (
                correction[:, None, None]
                * jnp.eye(3, dtype=velocity_gradient.dtype)[None, :, :]
            )
    affine_moment = ein.contract(
        "pr,pri,prj->pij", weights, route_velocity, state.route_offsets
    )
    particle_moment = state.second_moments
    first_moment = state.first_moments
    if state.radial_parity is not None:
        particle_moment = particle_moment - (
            first_moment[:, :, None] * first_moment[:, None, :]
        )
        affine_moment = affine_moment - velocity[:, :, None] * first_moment[:, None, :]
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
    if state.radial_parity is not None:
        velocity = velocity - ein.contract("pij,pj->pi", affine_velocity, first_moment)
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
        state.second_moments,
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
