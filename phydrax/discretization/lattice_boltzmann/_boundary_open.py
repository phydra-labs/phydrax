#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._strict import StrictModule
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ._link_topology import (
    CompiledLatticeBoltzmannLinkTopology,
    LatticeBoltzmannLinkOwner,
)


OpenNormal: TypeAlias = tuple[int, int]
VelocityOpenNormal: TypeAlias = tuple[int, int, Literal["any", "inlet", "outlet"]]


class LatticeBoltzmannBoundaryState(StrictModule):
    """Rollback-safe history carried by stateful boundary owners."""

    convective_history: Array
    convective_initialized: Array

    def __init__(
        self,
        convective_history: ArrayLike,
        convective_initialized: ArrayLike,
        /,
    ):
        history = jnp.asarray(convective_history)
        initialized = jnp.asarray(convective_initialized, dtype=bool)
        if history.shape != initialized.shape:
            raise ValueError("Convective history and initialization masks must match.")
        self.convective_history = history
        self.convective_initialized = initialized


def _weighted_projection(
    current: Array,
    base: Array,
    unknown: Array,
    matrix: Array,
    right_hand_side: Array,
    weights: Array,
    active: Array,
    /,
) -> Array:
    dtype = current.dtype
    weighted_matrix = (
        unknown[..., None] * weights.reshape((1,) * (unknown.ndim - 1) + (-1, 1))
    ) * matrix
    gram = ein.contract("...qi,...qj->...ij", weighted_matrix, matrix)
    base_moment = ein.contract("...qi,...q->...i", matrix, jnp.where(unknown, base, 0.0))
    residual = right_hand_side - base_moment
    dimension = matrix.shape[-1]
    identity = jnp.eye(dimension, dtype=dtype)
    safe_gram = jnp.where(active[..., None, None], gram, identity)
    solve = solve_small_linear(
        SmallLinearSolvePlan(
            dimension,
            singular_tolerance=1.0e-12,
            maximum_condition=1.0e12,
        ),
        safe_gram,
        jnp.where(active[..., None], residual, 0.0),
    )
    multiplier = eqx.error_if(
        solve.value,
        jnp.any(active & ~solve.successful),
        "Open-boundary moment constraints are singular for the compiled support.",
    )
    correction = ein.contract("...qi,...i->...q", weighted_matrix, multiplier)
    return jnp.where(unknown, base + correction, current)


def _velocity_reconstruction(
    current: Array,
    post_collision: Array,
    topology: CompiledLatticeBoltzmannLinkTopology,
    velocities: Array,
    opposite: Array,
    weights: Array,
    targets: Array,
    half_force_density: Array,
    normals: Sequence[VelocityOpenNormal],
    /,
) -> Array:
    candidate = current
    base = jnp.take(post_collision, opposite, axis=-1)
    for parameter_index, (axis, sign, flow_direction) in enumerate(normals):
        unknown = (topology.owner == int(LatticeBoltzmannLinkOwner.VELOCITY)) & (
            topology.parameter_index == parameter_index
        )
        active = jnp.any(unknown, axis=-1)
        target = targets[parameter_index]
        speed_squared = jnp.sum(target * target)
        target = eqx.error_if(
            target,
            ~jnp.all(jnp.isfinite(target)) | (speed_squared >= 1.0),
            "Velocity boundary targets must be finite and sub-lattice-speed.",
        )
        normal_velocity = sign * target[axis]
        if flow_direction == "inlet":
            target = eqx.error_if(
                target,
                normal_velocity >= 0.0,
                "Velocity inlet target points out of the domain.",
            )
        elif flow_direction == "outlet":
            target = eqx.error_if(
                target,
                normal_velocity <= 0.0,
                "Velocity outlet backflow is not admissible.",
            )
        target_field = jnp.broadcast_to(target, (*active.shape, target.shape[0]))
        known = ~unknown
        known_mass = jnp.sum(jnp.where(known, candidate, 0.0), axis=-1)
        known_momentum = ein.contract(
            "...q,qd->...d", jnp.where(known, candidate, 0.0), velocities
        )
        matrix = (
            velocities.reshape((1,) * active.ndim + velocities.shape)
            - target_field[..., None, :]
        )
        right_hand_side = (
            target_field * known_mass[..., None] - half_force_density - known_momentum
        )
        candidate = _weighted_projection(
            candidate,
            base,
            unknown,
            matrix,
            right_hand_side,
            weights,
            active,
        )
        density = jnp.sum(candidate, axis=-1)
        momentum = ein.contract("...q,qd->...d", candidate, velocities)
        reconstructed_velocity = (momentum + half_force_density) / density[..., None]
        invalid = active & (
            ~jnp.isfinite(density)
            | (density <= 0.0)
            | jnp.any(~jnp.isfinite(reconstructed_velocity), axis=-1)
            | (jnp.max(jnp.abs(reconstructed_velocity - target_field), axis=-1) > 5.0e-6)
        )
        candidate = eqx.error_if(
            candidate,
            jnp.any(invalid),
            "Velocity boundary reconstruction did not satisfy its constrained moments.",
        )
    return candidate


def _pressure_reconstruction(
    current: Array,
    post_collision: Array,
    topology: CompiledLatticeBoltzmannLinkTopology,
    velocities: Array,
    opposite: Array,
    weights: Array,
    densities: Array,
    tangential_velocities: Array,
    half_force_density: Array,
    normals: Sequence[OpenNormal],
    /,
) -> Array:
    candidate = current
    base = jnp.take(post_collision, opposite, axis=-1)
    dimension = velocities.shape[1]
    for parameter_index, (axis, sign) in enumerate(normals):
        unknown = (topology.owner == int(LatticeBoltzmannLinkOwner.PRESSURE)) & (
            topology.parameter_index == parameter_index
        )
        active = jnp.any(unknown, axis=-1)
        density_target = densities[parameter_index]
        tangential_target = tangential_velocities[parameter_index]
        density_target = eqx.error_if(
            density_target,
            ~jnp.isfinite(density_target) | (density_target <= 0.0),
            "Pressure boundary density targets must be finite and positive.",
        )
        tangential_target = eqx.error_if(
            tangential_target,
            ~jnp.all(jnp.isfinite(tangential_target))
            | (tangential_target[axis] != 0.0)
            | (jnp.sum(tangential_target * tangential_target) >= 1.0),
            "Pressure tangential targets must be finite, tangential, and sub-lattice-speed.",
        )
        known = ~unknown
        known_mass = jnp.sum(jnp.where(known, candidate, 0.0), axis=-1)
        known_momentum = ein.contract(
            "...q,qd->...d", jnp.where(known, candidate, 0.0), velocities
        )
        tangent_axes = tuple(index for index in range(dimension) if index != axis)
        columns = [jnp.ones((velocities.shape[0],), dtype=candidate.dtype)]
        columns.extend(velocities[:, tangent] for tangent in tangent_axes)
        local_matrix = jnp.stack(tuple(columns), axis=-1)
        matrix = jnp.broadcast_to(local_matrix, (*active.shape, *local_matrix.shape))
        right_columns = [jnp.broadcast_to(density_target, active.shape) - known_mass]
        right_columns.extend(
            density_target * tangential_target[tangent]
            - half_force_density[..., tangent]
            - known_momentum[..., tangent]
            for tangent in tangent_axes
        )
        right_hand_side = jnp.stack(tuple(right_columns), axis=-1)
        candidate = _weighted_projection(
            candidate,
            base,
            unknown,
            matrix,
            right_hand_side,
            weights,
            active,
        )
        density = jnp.sum(candidate, axis=-1)
        momentum = ein.contract("...q,qd->...d", candidate, velocities)
        physical_momentum = momentum + half_force_density
        outward_velocity = sign * physical_momentum[..., axis] / density
        tangent_error = jnp.zeros(active.shape, dtype=candidate.dtype)
        for tangent in tangent_axes:
            tangent_error = jnp.maximum(
                tangent_error,
                jnp.abs(
                    physical_momentum[..., tangent] / density - tangential_target[tangent]
                ),
            )
        invalid = active & (
            ~jnp.isfinite(density)
            | (jnp.abs(density - density_target) > 5.0e-6)
            | (tangent_error > 5.0e-6)
            | ~jnp.isfinite(outward_velocity)
            | (outward_velocity <= 0.0)
        )
        candidate = eqx.error_if(
            candidate,
            jnp.any(invalid),
            "Pressure outlet reconstruction failed its moments or detected backflow.",
        )
    return candidate


def _convective_reconstruction(
    current: Array,
    post_collision: Array,
    topology: CompiledLatticeBoltzmannLinkTopology,
    state: LatticeBoltzmannBoundaryState,
    velocities: Sequence[tuple[int, ...]],
    speeds: Array,
    normals: Sequence[OpenNormal],
    /,
) -> tuple[Array, LatticeBoltzmannBoundaryState]:
    axes = tuple(range(len(velocities[0])))
    interior = jnp.stack(
        tuple(
            jnp.roll(
                post_collision[..., direction],
                shift=tuple(-v for v in velocity),
                axis=axes,
            )
            for direction, velocity in enumerate(velocities)
        ),
        axis=-1,
    )
    candidate = current
    history = state.convective_history
    initialized = state.convective_initialized
    for parameter_index, _ in enumerate(normals):
        unknown = (topology.owner == int(LatticeBoltzmannLinkOwner.CONVECTIVE)) & (
            topology.parameter_index == parameter_index
        )
        speed = speeds[parameter_index]
        speed = eqx.error_if(
            speed,
            ~jnp.isfinite(speed) | (speed <= 0.0) | (speed > 1.0),
            "Convective outlet speed must lie in (0, 1].",
        )
        previous = jnp.where(initialized, history, interior)
        update = (previous + speed * interior) / (1.0 + speed)
        update = eqx.error_if(
            update,
            jnp.any(unknown & ~jnp.isfinite(update)),
            "Convective outlet produced a non-finite population.",
        )
        candidate = jnp.where(unknown, update, candidate)
        history = jnp.where(unknown, update, history)
        initialized = initialized | unknown
    return candidate, LatticeBoltzmannBoundaryState(history, initialized)


def apply_open_boundaries(
    current: Array,
    post_collision: Array,
    topology: CompiledLatticeBoltzmannLinkTopology,
    state: LatticeBoltzmannBoundaryState,
    velocities: Array,
    velocity_tuples: Sequence[tuple[int, ...]],
    opposite: Array,
    weights: Array,
    velocity_targets: Array,
    pressure_densities: Array,
    pressure_tangential_velocities: Array,
    convective_speeds: Array,
    half_force_density: Array,
    velocity_normals: Sequence[VelocityOpenNormal],
    pressure_normals: Sequence[OpenNormal],
    convective_normals: Sequence[OpenNormal],
    /,
) -> tuple[Array, LatticeBoltzmannBoundaryState]:
    """Apply all open owners without writing stream or wall populations."""

    candidate = _velocity_reconstruction(
        current,
        post_collision,
        topology,
        velocities,
        opposite,
        weights,
        velocity_targets,
        half_force_density,
        velocity_normals,
    )
    candidate = _pressure_reconstruction(
        candidate,
        post_collision,
        topology,
        velocities,
        opposite,
        weights,
        pressure_densities,
        pressure_tangential_velocities,
        half_force_density,
        pressure_normals,
    )
    return _convective_reconstruction(
        candidate,
        post_collision,
        topology,
        state,
        velocity_tuples,
        convective_speeds,
        convective_normals,
    )


__all__ = [
    "LatticeBoltzmannBoundaryState",
    "OpenNormal",
    "VelocityOpenNormal",
    "apply_open_boundaries",
]
