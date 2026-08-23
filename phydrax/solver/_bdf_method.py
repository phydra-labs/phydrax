#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._temporal_method import TemporalMethodCapabilities


_MAXIMUM_BDF_ORDER = 5
_HISTORY_CAPACITY = _MAXIMUM_BDF_ORDER


class BDFMethod(StrictModule, NonTrainableState):
    """Variable-step backward differentiation formula up to order five."""

    capabilities: TemporalMethodCapabilities
    maximum_order: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(self, maximum_order: int = 2, /):
        order = int(maximum_order)
        if order < 1 or order > _MAXIMUM_BDF_ORDER:
            raise ValueError("BDFMethod maximum_order must lie in [1, 5].")
        identifier = f"temporal:bdf:max-order-{order}"
        self.maximum_order = order
        self.method_id = identifier
        self.capabilities = TemporalMethodCapabilities(
            equation_forms=("implicit-residual", "split-residual"),
            method_class="bdf",
            order=order,
            embedded_order=order - 1 if order > 1 else None,
            adaptive=True,
            history_depth=order + 1,
            a_stable=order <= 2,
            l_stable=order == 1,
            stiffly_accurate=True,
            verified=True,
            method_id=identifier,
        )


def _derivative_coefficients(nodes: Array, /) -> Array:
    count = int(nodes.shape[0])
    scale = nodes[0] - nodes[1]
    offsets = (nodes - nodes[0]) / scale
    powers = jnp.arange(count, dtype=offsets.dtype)[:, None]
    vandermonde = offsets[None, :] ** powers
    right_hand_side = jnp.zeros((count,), dtype=offsets.dtype).at[1].set(1.0)
    return jnp.linalg.solve(vandermonde, right_hand_side) / scale


def bdf_coefficients(
    history_times: Array,
    target_time: Array,
    order: Array,
    /,
) -> Array:
    """Return six padded derivative coefficients for one variable-step BDF stage."""
    times = jnp.asarray(history_times)
    target = jnp.asarray(target_time)
    if times.shape != (_HISTORY_CAPACITY,):
        raise ValueError(f"history_times must have shape {(_HISTORY_CAPACITY,)}.")

    def branch(count: int):
        def evaluate(_):
            nodes = jnp.concatenate((target[None], times[:count]))
            values = _derivative_coefficients(nodes)
            return jnp.pad(values, (0, _MAXIMUM_BDF_ORDER + 1 - values.size))

        return evaluate

    branches = tuple(branch(count) for count in range(1, _MAXIMUM_BDF_ORDER + 1))
    selected = jnp.clip(jnp.asarray(order, dtype=jnp.int32), 1, _MAXIMUM_BDF_ORDER)
    return lax.switch(selected - 1, branches, operand=None)


def bdf_shift_offset(
    state_history: Array,
    history_times: Array,
    target_time: Array,
    order: Array,
    /,
) -> tuple[Array, Array]:
    """Return ``state_rate = shift * state + offset`` for one BDF stage."""
    if state_history.shape[0] != _HISTORY_CAPACITY:
        raise ValueError("state_history must have five leading history slots.")
    coefficients = bdf_coefficients(history_times, target_time, order)
    offset = jnp.tensordot(coefficients[1:], state_history, axes=1)
    return coefficients[0], offset


def bdf_rate(
    state: Array,
    state_history: Array,
    history_times: Array,
    target_time: Array,
    order: Array,
    /,
) -> Array:
    shift, offset = bdf_shift_offset(state_history, history_times, target_time, order)
    return shift * state + offset


def _extrapolate(nodes: Array, values: Array, target: Array, /) -> Array:
    count = int(nodes.shape[0])
    weights = []
    for index in range(count):
        numerator = jnp.asarray(1.0, dtype=nodes.dtype)
        denominator = jnp.asarray(1.0, dtype=nodes.dtype)
        for other in range(count):
            if other != index:
                numerator = numerator * (target - nodes[other])
                denominator = denominator * (nodes[index] - nodes[other])
        weights.append(numerator / denominator)
    return jnp.tensordot(jnp.stack(weights), values, axes=1)


def bdf_predict(
    state_history: Array,
    rate_history: Array,
    history_times: Array,
    target_time: Array,
    order: Array,
    history_depth: Array | None = None,
) -> Array:
    """Polynomial BDF predictor with a first-order derivative startup."""
    if (
        state_history.shape[0] != _HISTORY_CAPACITY
        or rate_history.shape != state_history.shape
        or history_times.shape != (_HISTORY_CAPACITY,)
    ):
        raise ValueError("BDF predictor histories do not align.")

    def first(_):
        return state_history[0] + (target_time - history_times[0]) * rate_history[0]

    def branch(count: int):
        def evaluate(_):
            return _extrapolate(history_times[:count], state_history[:count], target_time)

        return evaluate

    available = (
        jnp.asarray(_HISTORY_CAPACITY, dtype=jnp.int32)
        if history_depth is None
        else jnp.asarray(history_depth, dtype=jnp.int32)
    )
    interpolation_count = jnp.clip(
        jnp.minimum(order.astype(jnp.int32) + 1, available),
        2,
        _HISTORY_CAPACITY,
    )
    branches = (branch(2), branch(3), branch(4), branch(5))
    return lax.cond(
        order == 1,
        first,
        lambda _: lax.switch(interpolation_count - 2, branches, operand=None),
        operand=None,
    )


__all__ = [
    "BDFMethod",
    "bdf_coefficients",
    "bdf_predict",
    "bdf_rate",
    "bdf_shift_offset",
]
