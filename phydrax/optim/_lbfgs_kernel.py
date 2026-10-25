#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Limited-memory BFGS secant history and two-loop recursion.

One fixed-capacity ring of secant pairs per optimizer group. The projected
line-search method and the trust-scaled force-driven method both own their
globalization; this module owns only the curvature-pair admission, history
recurrence and the inverse-Hessian action.
"""

from __future__ import annotations

from typing import final

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike
from jaxtyping import PyTree

from .._strict import StrictModule
from ..typing import Dim, Float, Int32
from ._group_layout import OptimizerGroupDim, OptimizerGroupLayout


class LBFGSHistoryDim(Dim, minimum=1):
    """Capacity of the stored secant-pair ring."""


@final
class LBFGSHistory(StrictModule):
    """Per-group ring of admitted secant pairs ``s = x_k - x_{k-1}``, ``y = g_k - g_{k-1}``.

    ``steps`` and ``gradient_changes`` leaves carry a leading history axis.
    ``next_slot[g]`` is the slot the next admitted pair of group ``g`` occupies and
    ``count[g]`` the number of valid pairs, newest at ``next_slot - 1``. Slots of a
    group beyond its count are never read.
    """

    __strict_contract__ = True

    steps: PyTree[Array]
    gradient_changes: PyTree[Array]
    curvatures: Float[OptimizerGroupDim, LBFGSHistoryDim]
    change_squared_norms: Float[OptimizerGroupDim, LBFGSHistoryDim]
    next_slot: Int32[OptimizerGroupDim]
    count: Int32[OptimizerGroupDim]

    @property
    def capacity(self) -> int:
        return self.curvatures.shape[1]


def initial_lbfgs_history(
    parameters: PyTree[Array],
    history_size: int,
    layout: OptimizerGroupLayout,
    dtype: DTypeLike,
    /,
) -> LBFGSHistory:
    """Empty history with ``history_size`` slots per group."""

    if isinstance(history_size, bool) or not isinstance(history_size, int):
        raise TypeError("history_size must be an int.")
    if history_size < 1:
        raise ValueError("history_size must be positive.")
    buffers = jax.tree.map(
        lambda value: jnp.zeros((history_size, *jnp.shape(value)), dtype=value.dtype),
        parameters,
    )
    zero_pairs = jnp.zeros((layout.group_count, history_size), dtype=dtype)
    zero_counts = jnp.zeros((layout.group_count,), dtype=jnp.int32)
    return LBFGSHistory(
        steps=buffers,
        gradient_changes=buffers,
        curvatures=zero_pairs,
        change_squared_norms=zero_pairs,
        next_slot=zero_counts,
        count=zero_counts,
    )


def lbfgs_pair_curvature(
    step: PyTree[Array],
    change: PyTree[Array],
    layout: OptimizerGroupLayout,
    /,
) -> tuple[Array, Array, Array]:
    """Group ``s·y``, ``s·s`` and ``y·y`` of one candidate secant pair."""

    return (
        layout.inner(step, change),
        layout.squared_norm(step),
        layout.squared_norm(change),
    )


def admit_lbfgs_pair(
    curvature: Array,
    step_squared_norm: Array,
    change_squared_norm: Array,
    tolerance: float,
    /,
) -> Array:
    """Cosine floor ``s·y > tolerance |s| |y|`` with finite inner products."""

    threshold = tolerance * jnp.sqrt(step_squared_norm) * jnp.sqrt(change_squared_norm)
    return jnp.isfinite(curvature) & jnp.isfinite(threshold) & (curvature > threshold)


def reset_lbfgs_history(history: LBFGSHistory, reset: Array, /) -> LBFGSHistory:
    """Forget every pair of the selected groups."""

    reset_ = jnp.asarray(reset, dtype=jnp.bool_)
    zero = jnp.zeros_like(history.count)
    return LBFGSHistory(
        steps=history.steps,
        gradient_changes=history.gradient_changes,
        curvatures=history.curvatures,
        change_squared_norms=history.change_squared_norms,
        next_slot=jnp.where(reset_, zero, history.next_slot),
        count=jnp.where(reset_, zero, history.count),
    )


def _write_slot(
    buffers: PyTree[Array],
    values: PyTree[Array],
    slot: Array,
    admitted: Array,
    layout: OptimizerGroupLayout,
    /,
) -> PyTree[Array]:
    if not layout.grouped:
        return jax.tree.map(
            lambda buffer, value: buffer.at[slot[0]].set(
                jnp.where(admitted[0], value, buffer[slot[0]])
            ),
            buffers,
            values,
        )
    slots = layout.expand(slot, values)
    writes = layout.expand(admitted, values)

    def write(buffer: Array, value: Array, site_slot: Array, write_: Array) -> Array:
        positions = jnp.arange(buffer.shape[0], dtype=jnp.int32).reshape(
            (buffer.shape[0],) + (1,) * value.ndim
        )
        selected = (positions == site_slot[None]) & write_[None]
        return jnp.where(selected, value[None], buffer)

    return jax.tree.map(write, buffers, values, slots, writes)


def append_lbfgs_pair(
    history: LBFGSHistory,
    step: PyTree[Array],
    change: PyTree[Array],
    curvature: Array,
    change_squared_norm: Array,
    admitted: Array,
    layout: OptimizerGroupLayout,
    /,
) -> LBFGSHistory:
    """Store one admitted pair per selected group, replacing its oldest when full."""

    admitted_ = jnp.asarray(admitted, dtype=jnp.bool_)
    capacity = history.capacity
    slot = history.next_slot
    selected = admitted_[:, None] & (
        jnp.arange(capacity, dtype=jnp.int32)[None, :] == slot[:, None]
    )
    return LBFGSHistory(
        steps=_write_slot(history.steps, step, slot, admitted_, layout),
        gradient_changes=_write_slot(
            history.gradient_changes, change, slot, admitted_, layout
        ),
        curvatures=jnp.where(
            selected,
            curvature.astype(history.curvatures.dtype)[:, None],
            history.curvatures,
        ),
        change_squared_norms=jnp.where(
            selected,
            change_squared_norm.astype(history.change_squared_norms.dtype)[:, None],
            history.change_squared_norms,
        ),
        next_slot=jnp.where(admitted_, (slot + 1) % capacity, slot),
        count=jnp.where(
            admitted_, jnp.minimum(history.count + 1, capacity), history.count
        ),
    )


def _read_slot(
    buffers: PyTree[Array],
    slot: Array,
    layout: OptimizerGroupLayout,
    like: PyTree[Array],
    /,
) -> PyTree[Array]:
    if not layout.grouped:
        return jax.tree.map(
            lambda buffer: jax.lax.dynamic_index_in_dim(buffer, slot[0], keepdims=False),
            buffers,
        )
    slots = layout.expand(slot, like)
    return jax.tree.map(
        lambda buffer, site_slot: jnp.take_along_axis(buffer, site_slot[None], axis=0)[0],
        buffers,
        slots,
    )


def lbfgs_direction(
    gradient: PyTree[Array],
    history: LBFGSHistory,
    layout: OptimizerGroupLayout,
    /,
) -> PyTree[Array]:
    """Return ``-H g`` from the two-loop recursion over each group's history.

    The initial inverse Hessian is ``(s·y / y·y) I`` of the newest pair, and the
    identity for a group without pairs. Inactive layout sites are returned zero.
    """

    capacity = history.capacity
    count = history.count
    dtype = history.curvatures.dtype
    group_rows = jnp.arange(layout.group_count)

    def pair_slot(age: Array) -> Array:
        return (history.next_slot - 1 - age) % capacity

    def inverse_curvature(age: Array, active: Array) -> Array:
        curvature = history.curvatures[group_rows, pair_slot(age)]
        return jnp.where(active, 1.0 / jnp.where(active, curvature, 1.0), 0.0)

    def newest_first(
        index: Array, carry: tuple[PyTree[Array], Array]
    ) -> tuple[PyTree[Array], Array]:
        q, alphas = carry
        age = jnp.asarray(index, dtype=jnp.int32)
        active = age < count
        slot = pair_slot(age)
        step = _read_slot(history.steps, slot, layout, gradient)
        change = _read_slot(history.gradient_changes, slot, layout, gradient)
        alpha = inverse_curvature(age, active) * layout.inner(step, q)
        q = jax.tree.map(jnp.subtract, q, layout.scale(alpha, change))
        return q, alphas.at[age].set(alpha)

    q, alphas = jax.lax.fori_loop(
        0,
        capacity,
        newest_first,
        (gradient, jnp.zeros((capacity, layout.group_count), dtype=dtype)),
    )
    newest = pair_slot(jnp.zeros((), dtype=jnp.int32))
    gamma = history.curvatures[group_rows, newest] / jnp.maximum(
        history.change_squared_norms[group_rows, newest],
        1e-30,
    )
    gamma = jnp.where(count > 0, gamma, 1.0)
    result = layout.scale(gamma, q)

    def oldest_first(position: Array, result: PyTree[Array]) -> PyTree[Array]:
        age = jnp.asarray(capacity - 1 - position, dtype=jnp.int32)
        active = age < count
        slot = pair_slot(age)
        step = _read_slot(history.steps, slot, layout, gradient)
        change = _read_slot(history.gradient_changes, slot, layout, gradient)
        beta = inverse_curvature(age, active) * layout.inner(change, result)
        coefficient = jnp.where(active, alphas[age] - beta, 0.0)
        return jax.tree.map(jnp.add, result, layout.scale(coefficient, step))

    result = jax.lax.fori_loop(0, capacity, oldest_first, result)
    return jax.tree.map(jnp.negative, result)


__all__ = [
    "LBFGSHistory",
    "LBFGSHistoryDim",
    "admit_lbfgs_pair",
    "append_lbfgs_pair",
    "initial_lbfgs_history",
    "lbfgs_direction",
    "lbfgs_pair_curvature",
    "reset_lbfgs_history",
]
