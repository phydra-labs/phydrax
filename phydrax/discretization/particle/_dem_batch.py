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
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._dem import DEMRuntimeState, DEMStepEvaluation, PreparedSoftSphereDEMDynamics
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
    ):
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
    dynamics: PreparedSoftSphereDEMDynamics,
    time: Array,
    positions: Array,
    velocities: Array,
    angular_velocities: Array | None = None,
    /,
    *,
    args: Any = None,
) -> DEMRuntimeState:
    if not isinstance(dynamics, PreparedSoftSphereDEMDynamics):
        raise TypeError("dynamics must be PreparedSoftSphereDEMDynamics.")
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
    return jax.vmap(
        lambda position, velocity, angular: dynamics.initialize_state(
            time, position, velocity, angular, args=args
        )
    )(positions, velocities, angular_velocities)


def _force_uniform_rebuild(
    dynamics: PreparedSoftSphereDEMDynamics,
    states: DEMRuntimeState,
    step_size: Array,
    /,
) -> DEMRuntimeState:
    cache = states.neighborhood_cache
    if cache is None:
        return states
    half_velocity = states.kinematics.velocity + 0.5 * step_size * (
        dynamics.bodies.inverse_masses[None, :, None] * states.loads.total.force
    )
    predicted = step_size * jnp.sqrt(jnp.sum(half_velocity**2, axis=-1))
    needs_rebuild = jnp.any(jnp.max(predicted, axis=-1) > cache.certificate_margin)
    successful = jnp.where(needs_rebuild, False, cache.successful)
    return eqx.tree_at(
        lambda value: value.neighborhood_cache.successful,
        states,
        successful,
    )


def batch_step_detailed(
    dynamics: PreparedSoftSphereDEMDynamics,
    states: DEMRuntimeState,
    step_index: Array,
    time: Array,
    step_size: Array,
    plan: DEMBatchExecutionPlan,
    /,
    *,
    args: Any = None,
) -> DEMStepEvaluation:
    if not isinstance(dynamics, PreparedSoftSphereDEMDynamics):
        raise TypeError("dynamics must be PreparedSoftSphereDEMDynamics.")
    if not isinstance(plan, DEMBatchExecutionPlan):
        raise TypeError("plan must be a DEMBatchExecutionPlan.")
    prepared_states = states
    if isinstance(dynamics.neighborhood, PreparedVerletParticleNeighborhood):
        if plan.mode is DEMBatchExecutionMode.ALWAYS_BUILD:
            prepared_states = eqx.tree_at(
                lambda value: value.neighborhood_cache.successful,
                states,
                jnp.zeros_like(states.neighborhood_cache.successful),
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
