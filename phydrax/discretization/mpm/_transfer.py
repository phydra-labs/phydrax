#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from ..._interpolation import gather_patches
from ..._numerics._compensated import compensated_sum
from ..._strict import StrictModule
from ...discretization.splatting import ParticleGridSplatState
from ...linalg import (
    determinant_small_linear,
    SmallLinearSolvePlan,
    solve_small_linear,
)
from ._moments_native import streaming_moments


class APICGatherResult(StrictModule):
    velocity: Array
    velocity_gradient: Array
    affine_moment: Array
    particle_moment: Array
    affine_velocity: Array
    condition_estimate: Array
    successful: Array


# APIC local algebra is written as native elementwise products plus one route
# reduction per moment. Static physical axes (at most three components) are
# unrolled in index order rather than lowered to batched ``dot_general``. The
# Cartesian gather moments are one streaming primitive whose JVP, batching and
# dead-code elimination use its JAX reference; other profiles stay inline.


def _bounded_sum(terms: Array, /) -> Array:
    """Sum the static trailing physical axis in index order."""
    total = terms[..., 0]
    for index in range(1, terms.shape[-1]):
        total = total + terms[..., index]
    return total


def _bounded_dot(left: Array, right: Array, /) -> Array:
    """Contract one static trailing physical axis of broadcast operands in order."""
    width = left.shape[-1]
    if right.shape[-1] != width:
        raise ValueError(
            f"Contracted APIC axes differ: {left.shape[-1]} and {right.shape[-1]}."
        )
    total = left[..., 0] * right[..., 0]
    for index in range(1, width):
        total = total + left[..., index] * right[..., index]
    return total


def _bounded_trace(matrix: Array, /) -> Array:
    total = matrix[..., 0, 0]
    for index in range(1, matrix.shape[-1]):
        total = total + matrix[..., index, index]
    return total


def _route_sum(terms: Array, /) -> Array:
    """Reduce the static route axis of ``(particle, route, ...)`` terms."""
    return jnp.sum(terms, axis=1)


def _radial_mask(rank: int, width: int, /) -> Array:
    """Select component zero with an explicit-rank (rank-promotion safe) mask."""
    return (jnp.arange(width) == 0).reshape((1,) * (rank - 1) + (width,))


def _radial_scale(parity: Array, width: int, dtype: jnp.dtype, /) -> Array:
    """Scale only the radial component; other components multiply by exact one."""
    return jnp.where(
        _radial_mask(parity.ndim + 1, width),
        parity[..., None].astype(dtype),
        jnp.ones((), dtype=dtype),
    )


def _radial_component(values: Array, width: int, /) -> Array:
    """Place route scalars on the radial component with exact zeros elsewhere."""
    return jnp.where(
        _radial_mask(values.ndim + 1, width),
        values[..., None],
        jnp.zeros((), dtype=values.dtype),
    )


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
    translational = _bounded_dot(velocity, velocity)
    # sum_ijk C_ij D_jk C_ik, contracted as (C D)_ik C_ik over k then i.
    affine_moment = _bounded_dot(
        affine_velocity[:, :, None, :], particle_moment.swapaxes(-1, -2)[:, None, :, :]
    )
    affine = _bounded_sum(_bounded_dot(affine_moment, affine_velocity))
    if first_moment is not None:
        affine = affine + 2.0 * _bounded_dot(
            velocity, _bounded_dot(affine_velocity, first_moment[:, None, :])
        )
    terms = 0.5 * mass * (translational + affine)
    return compensated_sum(jnp.where(active, terms, 0.0))


def _affine_route_velocity(affine_velocity: Array, route_offsets: Array, /) -> Array:
    """Evaluate ``C_p (x_i - x_p)`` for every route as ``(particle, route, i)``."""
    return _bounded_dot(affine_velocity[:, None, :, :], route_offsets[:, :, None, :])


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
    affine_route_velocity = _affine_route_velocity(affine_velocity, state.route_offsets)
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
    affine_route_velocity = _affine_route_velocity(affine_velocity, state.route_offsets)
    weights = state.stencil.weights
    route_momentum = (
        weights[..., None]
        * mass[:, None, None]
        * (velocity[:, None, :] + affine_route_velocity)
    )
    # Kirchhoff stress tau_ik = sum_j P_ij F_kj.
    kirchhoff = _bounded_dot(
        first_piola[:, :, None, :], deformation_gradient[:, None, :, :]
    )
    if physical_deformation_gradient is not None:
        plan = SmallLinearSolvePlan(deformation_gradient.shape[-1])
        assumed_j = determinant_small_linear(plan, deformation_gradient)
        physical_j = determinant_small_linear(plan, physical_deformation_gradient)
        reference_volume = reference_volume * physical_j / assumed_j
    spatial_dimension = state.weight_gradients.shape[-1]
    internal_force = -reference_volume[:, None, None] * _bounded_dot(
        kirchhoff[:, None, :spatial_dimension, :spatial_dimension],
        state.weight_gradients[:, :, None, :],
    )
    if state.hoop_gradients is not None:
        # Current-ring virtual work V*sigma:L equals V0*tau:L.
        hoop = reference_volume[:, None] * kirchhoff[:, 2, 2, None] * state.hoop_gradients
        internal_force = internal_force - _radial_component(hoop, spatial_dimension)
    if state.cell_average_gradients is not None:
        mean_stress = _bounded_trace(kirchhoff) / kirchhoff.shape[-1]
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
            internal_force = internal_force - _radial_component(
                reference_volume[:, None]
                * mean_stress[:, None]
                * (state.cell_average_hoop_gradients - state.hoop_gradients),
                spatial_dimension,
            )
    external_force = (
        weights[..., None] * mass[:, None, None] * external_acceleration[:, None, :]
    )
    if state.radial_parity is not None:
        route_momentum = route_momentum * _radial_scale(
            state.radial_parity, route_momentum.shape[-1], route_momentum.dtype
        )
        internal_force = internal_force * _radial_scale(
            state.radial_parity, internal_force.shape[-1], internal_force.dtype
        )
        external_force = external_force * _radial_scale(
            state.radial_parity, external_force.shape[-1], external_force.dtype
        )
    mask = active[:, None, None] & state.stencil.valid[..., None]
    return jnp.where(
        mask,
        jnp.concatenate((route_momentum, internal_force, external_force), axis=-1),
        0.0,
    )


def _streaming_profile(state: ParticleGridSplatState, grid_velocity: Array, /) -> bool:
    """Cartesian non-B-bar single-case routes with one shared floating dtype."""
    stencil = state.stencil
    dtype = grid_velocity.dtype
    return (
        state.radial_parity is None
        and state.hoop_gradients is None
        and state.cell_average_gradients is None
        and state.cell_average_hoop_gradients is None
        and stencil.case_shape == ()
        and grid_velocity.ndim == 2
        and grid_velocity.shape[0] == stencil.source_size
        and jnp.issubdtype(dtype, jnp.inexact)
        and state.weight_gradients.ndim == 3
        and state.route_offsets.ndim == 3
        and all(
            value.dtype == dtype
            for value in (stencil.weights, state.weight_gradients, state.route_offsets)
        )
    )


def _generic_moments(
    state: ParticleGridSplatState, grid_velocity: Array, /
) -> tuple[Array, Array, Array]:
    """Route moments for every profile, including axisymmetric and B-bar."""
    dimension = grid_velocity.shape[-1]
    # Invalid routes are already exact zeros in the gathered payload.
    route_velocity, route_valid = gather_patches(grid_velocity, state.stencil)
    weights = jnp.where(route_valid, state.stencil.weights, 0.0)
    if state.radial_parity is not None:
        route_velocity = route_velocity * _radial_scale(
            state.radial_parity, dimension, route_velocity.dtype
        )
    # One shared weighted route load feeds both the velocity and affine moment.
    weighted_velocity = weights[..., None] * route_velocity
    velocity = _route_sum(weighted_velocity)
    velocity_gradient = _route_sum(
        route_velocity[..., :, None] * state.weight_gradients[:, :, None, :]
    )
    if state.hoop_gradients is None and state.cell_average_gradients is not None:
        projected_divergence = _route_sum(
            _bounded_dot(route_velocity, state.cell_average_gradients)
        )
        correction = (
            projected_divergence - _bounded_trace(velocity_gradient)
        ) / dimension
        velocity_gradient = velocity_gradient + (
            correction[:, None, None]
            * jnp.eye(dimension, dtype=velocity_gradient.dtype)[None, :, :]
        )
    if state.hoop_gradients is not None:
        hoop_rate = _route_sum(route_velocity[..., 0] * state.hoop_gradients)
        velocity_gradient = jnp.where(
            (jnp.arange(3)[None, :, None] == 2) & (jnp.arange(3)[None, None, :] == 2),
            hoop_rate[:, None, None].astype(velocity_gradient.dtype),
            jnp.pad(velocity_gradient, ((0, 0), (0, 1), (0, 1))),
        )
        if state.cell_average_gradients is not None:
            if state.cell_average_hoop_gradients is None:
                raise RuntimeError(
                    "Axisymmetric B-bar requires the cell-average hoop operator."
                )
            projected_divergence = _route_sum(
                _bounded_dot(route_velocity, state.cell_average_gradients)
            ) + _route_sum(route_velocity[..., 0] * state.cell_average_hoop_gradients)
            correction = (projected_divergence - _bounded_trace(velocity_gradient)) / 3.0
            velocity_gradient = velocity_gradient + (
                correction[:, None, None]
                * jnp.eye(3, dtype=velocity_gradient.dtype)[None, :, :]
            )
    affine_moment = _route_sum(
        weighted_velocity[..., :, None] * state.route_offsets[:, :, None, :]
    )
    return velocity, velocity_gradient, affine_moment


def gather_apic(
    state: ParticleGridSplatState,
    grid_velocity: Array,
    active: Array,
    maximum_condition: float,
    /,
) -> APICGatherResult:
    dimension = grid_velocity.shape[-1]
    if _streaming_profile(state, grid_velocity):
        # One primitive owns the Cartesian moments: native streaming on
        # admitted CUDA targets, the identical JAX reference elsewhere.
        velocity, velocity_gradient, affine_moment = streaming_moments(
            state.stencil.indices,
            state.stencil.valid,
            state.stencil.weights,
            state.weight_gradients,
            state.route_offsets,
            grid_velocity,
        )
    else:
        velocity, velocity_gradient, affine_moment = _generic_moments(
            state, grid_velocity
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
        velocity = velocity - _bounded_dot(affine_velocity, first_moment[:, None, :])
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
