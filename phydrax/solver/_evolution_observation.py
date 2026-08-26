#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from operator import index
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..dynamics import AbstractEvolution


OBSERVATION_NONFINITE = -1


class BoundedEvolutionObservationPlan(StrictModule, NonTrainableState):
    """Fixed-capacity sampling schedule for one pure evolution observable."""

    observable: Callable[[Array, Array, Any], ArrayLike]
    observable_shape: tuple[int, ...] = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    sample_stride: int = eqx.field(static=True)
    include_initial: bool = eqx.field(static=True)
    observer_id: str = eqx.field(static=True)

    def __init__(
        self,
        observable: Callable[[Array, Array, Any], ArrayLike],
        observable_shape: Sequence[int],
        capacity: int,
        /,
        *,
        sample_stride: int = 1,
        include_initial: bool = True,
        observer_id: str | None = None,
    ):
        if not callable(observable):
            raise TypeError("observable must be callable.")
        if any(isinstance(size, bool) for size in observable_shape):
            raise TypeError("observable_shape dimensions must be integers.")
        shape = tuple(index(size) for size in observable_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("observable_shape dimensions must be positive.")
        if isinstance(capacity, bool) or isinstance(sample_stride, bool):
            raise TypeError("capacity and sample_stride must be integers.")
        capacity_ = index(capacity)
        stride = index(sample_stride)
        if capacity_ < 1 or stride < 1:
            raise ValueError("capacity and sample_stride must be positive.")
        if not isinstance(include_initial, (bool, np.bool_)):
            raise TypeError("include_initial must be Boolean.")
        if observer_id is None:
            raise ValueError(
                "observer_id is required because observable callables have no stable identity."
            )
        identifier = str(observer_id)
        if not identifier:
            raise ValueError("observer_id must be non-empty.")
        self.observable = observable
        self.observable_shape = shape
        self.capacity = capacity_
        self.sample_stride = stride
        self.include_initial = bool(include_initial)
        self.observer_id = identifier


class BoundedEvolutionObservation(StrictModule):
    coordinates: Array
    values: Array
    valid: Array
    final_state: Array
    final_valid: Array
    count: Array
    overflow: Array
    final_status: Array
    evolution_id: str = eqx.field(static=True)
    observer_id: str = eqx.field(static=True)

    @property
    def sample_mask(self) -> Array:
        return jnp.arange(self.coordinates.size) < self.count


def observe_evolution_bounded(
    evolution: AbstractEvolution,
    initial_state: ArrayLike,
    coordinates: ArrayLike,
    plan: BoundedEvolutionObservationPlan,
    /,
    *,
    args: Any = None,
) -> BoundedEvolutionObservation:
    """Advance one path while retaining only a fixed-capacity observable buffer."""
    if not isinstance(evolution, AbstractEvolution):
        raise TypeError("evolution must be an AbstractEvolution.")
    if not isinstance(plan, BoundedEvolutionObservationPlan):
        raise TypeError("plan must be BoundedEvolutionObservationPlan.")
    raw_grid = jnp.asarray(coordinates)
    if jnp.iscomplexobj(raw_grid):
        raise TypeError("coordinates must be real.")
    grid = raw_grid.astype(jnp.result_type(raw_grid.dtype, jnp.float32))
    grid_host = np.asarray(grid)
    if (
        grid.ndim != 1
        or grid.size < 2
        or np.any(~np.isfinite(grid_host))
        or np.any(np.diff(grid_host) <= 0.0)
    ):
        raise ValueError("coordinates must be finite, increasing, and rank one.")
    initial = jnp.asarray(initial_state)
    if initial.shape != evolution.state_layout.shape:
        raise ValueError(f"initial_state must have shape {evolution.state_layout.shape}.")
    initial_value = jnp.asarray(plan.observable(grid[0], initial, args))
    if initial_value.shape != plan.observable_shape:
        raise ValueError(
            f"Observable must return shape {plan.observable_shape}; "
            f"got {initial_value.shape}."
        )
    coordinate_buffer = jnp.full((plan.capacity,), jnp.inf, dtype=grid.dtype)
    value_buffer = jnp.zeros(
        (plan.capacity,) + plan.observable_shape, dtype=initial_value.dtype
    )
    valid_buffer = jnp.zeros((plan.capacity,), dtype=bool)
    initial_finite = jnp.all(jnp.isfinite(initial)) & jnp.all(jnp.isfinite(initial_value))
    initial_count = jnp.asarray(int(plan.include_initial), dtype=jnp.int32)
    initial_status = jnp.where(
        initial_finite,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(OBSERVATION_NONFINITE, dtype=jnp.int32),
    )
    if plan.include_initial:
        coordinate_buffer = coordinate_buffer.at[0].set(grid[0])
        value_buffer = value_buffer.at[0].set(initial_value)
        valid_buffer = valid_buffer.at[0].set(initial_finite)

    def advance(carry, data):
        (
            state,
            cumulative_valid,
            saved_coordinates,
            saved_values,
            saved_valid,
            count,
            overflow,
            previous_status,
        ) = carry
        index, source, target = data
        step = evolution.advance(state, source, target, args)
        value = jnp.asarray(plan.observable(target, step.final_state, args))
        finite_value = jnp.all(jnp.isfinite(value))
        step_valid = cumulative_valid & step.valid & finite_value
        next_status = jnp.where(
            cumulative_valid,
            jnp.where(finite_value, step.status, OBSERVATION_NONFINITE),
            previous_status,
        )
        requested = ((index + 1) % plan.sample_stride) == 0
        available = count < plan.capacity
        write = requested & available
        safe_index = jnp.minimum(count, plan.capacity - 1)
        saved_coordinates = jax.lax.cond(
            write,
            lambda buffer: buffer.at[safe_index].set(target),
            lambda buffer: buffer,
            saved_coordinates,
        )
        saved_values = jax.lax.cond(
            write,
            lambda buffer: buffer.at[safe_index].set(value),
            lambda buffer: buffer,
            saved_values,
        )
        saved_valid = jax.lax.cond(
            write,
            lambda buffer: buffer.at[safe_index].set(step_valid),
            lambda buffer: buffer,
            saved_valid,
        )
        return (
            step.final_state,
            step_valid,
            saved_coordinates,
            saved_values,
            saved_valid,
            count + write.astype(jnp.int32),
            overflow | (requested & ~available),
            next_status,
        ), None

    initial_carry = (
        initial,
        initial_finite,
        coordinate_buffer,
        value_buffer,
        valid_buffer,
        initial_count,
        jnp.asarray(False),
        initial_status,
    )
    indices = jnp.arange(grid.size - 1, dtype=jnp.int32)
    final, _ = jax.lax.scan(
        advance,
        initial_carry,
        (indices, grid[:-1], grid[1:]),
    )
    return BoundedEvolutionObservation(
        coordinates=final[2],
        values=final[3],
        valid=final[4],
        final_state=final[0],
        final_valid=final[1],
        count=final[5],
        overflow=final[6],
        final_status=final[7],
        evolution_id=evolution.evolution_id,
        observer_id=plan.observer_id,
    )


__all__ = [
    "BoundedEvolutionObservation",
    "BoundedEvolutionObservationPlan",
    "observe_evolution_bounded",
    "OBSERVATION_NONFINITE",
]
