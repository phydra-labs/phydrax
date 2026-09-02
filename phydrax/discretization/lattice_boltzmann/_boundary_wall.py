#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

import phydrax.ein as ein

from ..._strict import StrictModule
from ._link_topology import (
    CompiledLatticeBoltzmannLinkTopology,
    LatticeBoltzmannLinkOwner,
)


class LatticeBoltzmannWallLedger(StrictModule):
    """Conservative per-body momentum, load, torque, and work accounting."""

    fluid_impulse: Array
    body_impulse: Array
    force: Array
    angular_impulse: Array
    torque: Array
    work: Array
    fluid_work: Array

    def __init__(
        self,
        fluid_impulse: Array,
        body_impulse: Array,
        force: Array,
        angular_impulse: Array,
        torque: Array,
        work: Array,
        fluid_work: Array,
        /,
    ):
        self.fluid_impulse = fluid_impulse
        self.body_impulse = body_impulse
        self.force = force
        self.angular_impulse = angular_impulse
        self.torque = torque
        self.work = work
        self.fluid_work = fluid_work


def _wall_velocity(
    link_points: Array,
    body_index: Array,
    body_centers: Array,
    body_linear_velocities: Array,
    body_angular_velocities: Array,
    /,
) -> Array:
    dimension = link_points.shape[-1]
    velocity = jnp.zeros_like(link_points)
    for body in range(body_centers.shape[0]):
        selected = body_index == body
        radius = link_points - body_centers[body]
        if dimension == 2:
            omega = body_angular_velocities[body, 0]
            rotational = jnp.stack(
                (-omega * radius[..., 1], omega * radius[..., 0]), axis=-1
            )
        else:
            angular = body_angular_velocities[body]
            rotational = jnp.stack(
                (
                    angular[1] * radius[..., 2] - angular[2] * radius[..., 1],
                    angular[2] * radius[..., 0] - angular[0] * radius[..., 2],
                    angular[0] * radius[..., 1] - angular[1] * radius[..., 0],
                ),
                axis=-1,
            )
        rigid = body_linear_velocities[body] + rotational
        velocity = jnp.where(selected[..., None], rigid, velocity)
    return velocity


def _ledger(
    wall_mask: Array,
    body_index: Array,
    link_points: Array,
    wall_velocity: Array,
    incoming: Array,
    outgoing: Array,
    velocities: Array,
    body_centers: Array,
    time_step: Array,
    /,
) -> LatticeBoltzmannWallLedger:
    dimension = velocities.shape[1]
    body_count = body_centers.shape[0]
    angular_dimension = 1 if dimension == 2 else 3
    if body_count == 0:
        empty_vector = jnp.zeros((0, dimension), dtype=incoming.dtype)
        empty_angular = jnp.zeros((0, angular_dimension), dtype=incoming.dtype)
        empty_scalar = jnp.zeros((0,), dtype=incoming.dtype)
        return LatticeBoltzmannWallLedger(
            empty_vector,
            empty_vector,
            empty_vector,
            empty_angular,
            empty_angular,
            empty_scalar,
            empty_scalar,
        )
    fluid_link_impulse = (incoming + outgoing)[..., None] * velocities.reshape(
        (1,) * (incoming.ndim - 1) + velocities.shape
    )
    fluid_link_impulse = jnp.where(wall_mask[..., None], fluid_link_impulse, 0.0)
    one_hot = body_index[..., None] == jnp.arange(body_count)
    fluid_impulse = ein.contract(
        "...qb,...qd->bd", one_hot.astype(incoming.dtype), fluid_link_impulse
    )
    body_impulse = -fluid_impulse
    body_link_impulse = -fluid_link_impulse
    radius = link_points - body_centers[body_index.clip(0)]
    radius = jnp.where(wall_mask[..., None], radius, 0.0)
    if dimension == 2:
        angular_link = (
            radius[..., 0] * body_link_impulse[..., 1]
            - radius[..., 1] * body_link_impulse[..., 0]
        )[..., None]
    else:
        angular_link = jnp.stack(
            (
                radius[..., 1] * body_link_impulse[..., 2]
                - radius[..., 2] * body_link_impulse[..., 1],
                radius[..., 2] * body_link_impulse[..., 0]
                - radius[..., 0] * body_link_impulse[..., 2],
                radius[..., 0] * body_link_impulse[..., 1]
                - radius[..., 1] * body_link_impulse[..., 0],
            ),
            axis=-1,
        )
    angular_impulse = ein.contract(
        "...qb,...qa->ba", one_hot.astype(incoming.dtype), angular_link
    )
    link_work = ein.contract("...qd,...qd->...q", body_link_impulse, wall_velocity)
    work = ein.contract("...qb,...q->b", one_hot.astype(incoming.dtype), link_work)
    return LatticeBoltzmannWallLedger(
        fluid_impulse,
        body_impulse,
        body_impulse / time_step,
        angular_impulse,
        angular_impulse / time_step,
        work,
        -work,
    )


def apply_wall_boundaries(
    current: Array,
    post_collision: Array,
    density: Array,
    topology: CompiledLatticeBoltzmannLinkTopology,
    velocity_tuples: Sequence[tuple[int, ...]],
    velocities: Array,
    opposite: Array,
    weights: Array,
    sound_speed_squared: Array,
    coordinates: Array,
    cell_size: Array,
    body_centers: Array,
    body_linear_velocities: Array,
    body_angular_velocities: Array,
    time_step: Array,
    /,
) -> tuple[Array, LatticeBoltzmannWallLedger]:
    """Apply halfway and both Bouzidi branches and accumulate conservative loads."""

    halfway = topology.owner == int(LatticeBoltzmannLinkOwner.HALFWAY)
    bouzidi = topology.owner == int(LatticeBoltzmannLinkOwner.BOUZIDI)
    wall = halfway | bouzidi
    time_step = eqx.error_if(
        time_step,
        ~jnp.isfinite(time_step) | (time_step <= 0.0),
        "Boundary time_step must be finite and positive.",
    )
    fraction = jnp.where(wall, topology.link_fraction, 0.5).astype(current.dtype)
    lattice_velocities = velocities.astype(current.dtype)
    velocity_field = lattice_velocities.reshape(
        (1,) * (current.ndim - 1) + lattice_velocities.shape
    )
    link_points = (
        coordinates[..., None, :]
        - fraction[..., None] * cell_size.astype(current.dtype) * velocity_field
    )
    wall_velocity = _wall_velocity(
        link_points,
        topology.body_index,
        body_centers,
        body_linear_velocities,
        body_angular_velocities,
    )
    wall_velocity = eqx.error_if(
        wall_velocity,
        jnp.any(wall[..., None] & ~jnp.isfinite(wall_velocity)),
        "Wall kinematics must be finite on every owned link.",
    )
    projection = ein.contract("...qd,...qd->...q", velocity_field, wall_velocity)
    correction = (
        2.0
        * weights.astype(current.dtype).reshape(
            (1,) * (current.ndim - 1) + (weights.shape[0],)
        )
        * density[..., None]
        * projection
        / sound_speed_squared.astype(current.dtype)
    )
    outgoing = jnp.take(post_collision, opposite, axis=-1)
    halfway_value = outgoing + correction

    axes = tuple(range(len(velocity_tuples[0])))
    interior_outgoing = jnp.stack(
        tuple(
            jnp.roll(
                outgoing[..., direction],
                shift=tuple(-component for component in velocity),
                axis=axes,
            )
            for direction, velocity in enumerate(velocity_tuples)
        ),
        axis=-1,
    )
    near_value = (
        2.0 * fraction * outgoing
        + (1.0 - 2.0 * fraction) * interior_outgoing
        + correction
    )
    far_value = (
        outgoing / (2.0 * fraction)
        + (2.0 * fraction - 1.0) * post_collision / (2.0 * fraction)
        + correction / (2.0 * fraction)
    )
    bouzidi_value = jnp.where(fraction < 0.5, near_value, far_value)
    incoming = jnp.where(halfway, halfway_value, bouzidi_value)
    incoming = eqx.error_if(
        incoming,
        jnp.any(wall & ~jnp.isfinite(incoming)),
        "Wall reconstruction produced a non-finite population.",
    )
    candidate = jnp.where(wall, incoming, current)
    ledger = _ledger(
        wall,
        topology.body_index,
        link_points,
        wall_velocity,
        incoming,
        outgoing,
        lattice_velocities,
        body_centers,
        time_step,
    )
    return candidate, ledger


__all__ = ["LatticeBoltzmannWallLedger", "apply_wall_boundaries"]
