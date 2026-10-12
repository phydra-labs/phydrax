#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._dem import DEMRuntimeState, DEMStepEvaluation, PreparedCompliantDEMDynamics
from ._verlet import PreparedVerletParticleNeighborhood


class DEMBatchExecutionMode(StrEnum):
    REFERENCE_VMAP = "reference_vmap"
    UNIFORM_REBUILD = "uniform_rebuild"
    ALWAYS_BUILD = "always_build"


class DEMBatchExecutionPlan(StrictModule, NonTrainableState):
    mode: DEMBatchExecutionMode = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, mode: DEMBatchExecutionMode = DEMBatchExecutionMode.REFERENCE_VMAP, /
    ) -> None:
        if not isinstance(mode, DEMBatchExecutionMode):
            raise TypeError("mode must be a DEMBatchExecutionMode.")
        self.mode = mode
        self.plan_id = canonical_fingerprint(
            {"kind": "dem-batch-execution-plan", "mode": mode.value}
        )


def stack_dem_states(states: Sequence[DEMRuntimeState], /) -> DEMRuntimeState:
    values = tuple(states)
    if not values or any(not isinstance(value, DEMRuntimeState) for value in values):
        raise TypeError("states must contain DEMRuntimeState values.")
    structure = jax.tree.structure(values[0])
    if any(jax.tree.structure(value) != structure for value in values[1:]):
        raise ValueError("DEM batch states must have identical PyTree structures.")
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *values)


def initialize_dem_batch(
    dynamics: PreparedCompliantDEMDynamics,
    time: Array,
    positions: Array,
    velocities: Array,
    angular_velocities: Array | None = None,
    /,
    *,
    args: Any = None,
    orientations: Array | None = None,
) -> DEMRuntimeState:
    if not isinstance(dynamics, PreparedCompliantDEMDynamics):
        raise TypeError("dynamics must be PreparedCompliantDEMDynamics.")
    if positions.ndim != 3 or velocities.shape != positions.shape:
        raise ValueError("Batched positions and velocities must have shape (batch,N,D).")
    if angular_velocities is None:
        angular_velocities = jnp.zeros(
            (
                positions.shape[0],
                dynamics.bodies.capacity,
                dynamics.bodies.angular_dimension,
            ),
            dtype=positions.dtype,
        )
    if orientations is not None:
        expected = (
            positions.shape[0],
            dynamics.bodies.capacity,
            dynamics.bodies.orientation_dimension,
        )
        if orientations.shape != expected:
            raise ValueError(
                "Batched orientations must match prepared body orientation shape."
            )
        return jax.vmap(
            lambda position, velocity, angular, orientation: dynamics.initialize_state(
                time, position, velocity, angular, orientation=orientation, args=args
            )
        )(positions, velocities, angular_velocities, orientations)
    if dynamics.shapes.shape_kind != "sphere":
        raise ValueError("Nonspherical DEM batches require explicit orientations.")
    return jax.vmap(
        lambda position, velocity, angular: dynamics.initialize_state(
            time, position, velocity, angular, args=args
        )
    )(positions, velocities, angular_velocities)


def _force_uniform_rebuild(
    dynamics: PreparedCompliantDEMDynamics,
    states: DEMRuntimeState,
    step_size: Array,
    /,
) -> DEMRuntimeState:
    cache = states.neighborhood_cache
    if cache is None:
        return states
    half_velocity = states.kinematics.velocity + 0.5 * step_size * (
        states.body_properties.inverse_masses[:, :, None] * states.loads.total.force
    )
    predicted = step_size * jnp.sqrt(jnp.sum(half_velocity**2, axis=-1))
    if dynamics.shapes.shape_kind != "sphere":
        angular_speed = jnp.sqrt(jnp.sum(states.kinematics.angular_velocity**2, axis=-1))
        predicted = (
            predicted
            + step_size
            * angular_speed
            * dynamics.shapes.bounding_radii[None, :]
            * states.body_properties.shape_scale
        )
    needs_rebuild = jnp.any(jnp.max(predicted, axis=-1) > cache.certificate_margin)
    successful = jnp.where(needs_rebuild, False, cache.successful)
    return eqx.tree_at(
        lambda value: value.neighborhood_cache.successful,
        states,
        successful,
    )


def batch_step_detailed(
    dynamics: PreparedCompliantDEMDynamics,
    states: DEMRuntimeState,
    step_index: Array,
    time: Array,
    step_size: Array,
    plan: DEMBatchExecutionPlan,
    /,
    *,
    args: Any = None,
) -> DEMStepEvaluation:
    if not isinstance(dynamics, PreparedCompliantDEMDynamics):
        raise TypeError("dynamics must be PreparedCompliantDEMDynamics.")
    if not isinstance(plan, DEMBatchExecutionPlan):
        raise TypeError("plan must be a DEMBatchExecutionPlan.")
    prepared_states = states
    cache = states.neighborhood_cache
    if isinstance(dynamics.neighborhood, PreparedVerletParticleNeighborhood):
        if plan.mode is DEMBatchExecutionMode.ALWAYS_BUILD and cache is not None:
            prepared_states = eqx.tree_at(
                lambda value: value.neighborhood_cache.successful,
                states,
                jnp.zeros_like(cache.successful),
            )
        elif plan.mode is DEMBatchExecutionMode.UNIFORM_REBUILD:
            prepared_states = _force_uniform_rebuild(dynamics, states, step_size)
    return jax.vmap(
        lambda state: dynamics.step_detailed(step_index, time, state, step_size, args)
    )(prepared_states)


__all__ = [
    "DEMBatchExecutionMode",
    "DEMBatchExecutionPlan",
    "batch_step_detailed",
    "initialize_dem_batch",
    "stack_dem_states",
]
