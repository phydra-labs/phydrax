#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState

from ._axis import (
    _normalize_axis,
    _positive_int,
    _promote_signal_and_taps,
    _replace_axis_size,
    _valid_prefix,
)
from ._convolution import convolve


class FIRFilterState(StrictModule):
    """Immutable causal FIR history and absolute sample cursor."""

    history: Array
    sample_count: Array
    plan_id: str = eqx.field(static=True)


class FIRFilterResult(StrictModule):
    """Fixed-capacity FIR output with prefix validity metadata."""

    values: Array
    active: Array
    sample_offset: Array


class FIRFilterPlan(StrictModule, NonTrainableState):
    """Static topology for one shared-tap causal FIR filter."""

    tap_count: int = eqx.field(static=True)
    axis: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, tap_count: int, /, *, axis: int = -1):
        self.tap_count = _positive_int(tap_count, "tap_count")
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("axis must be an integer.")
        self.axis = axis
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fir_filter",
                "tap_count": self.tap_count,
                "axis": self.axis,
            }
        )

    def initial_state(
        self,
        input_shape: Sequence[int],
        /,
        *,
        dtype: jnp.dtype,
    ) -> FIRFilterState:
        """Return zero history for arrays with ``input_shape``."""
        shape = tuple(int(size) for size in input_shape)
        resolved_axis = _normalize_axis(self.axis, len(shape))
        resolved_dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(resolved_dtype, jnp.inexact):
            raise TypeError("FIR state dtype must be inexact.")
        history_shape = _replace_axis_size(
            shape,
            resolved_axis,
            self.tap_count - 1,
        )
        return FIRFilterState(
            history=jnp.zeros(history_shape, dtype=resolved_dtype),
            sample_count=jnp.asarray(0, dtype=jnp.int64),
            plan_id=self.plan_id,
        )

    def step(
        self,
        state: FIRFilterState,
        chunk: ArrayLike,
        taps: ArrayLike,
        /,
        *,
        valid_length: ArrayLike | None = None,
    ) -> tuple[FIRFilterState, FIRFilterResult]:
        """Filter one fixed-capacity chunk without consuming its inactive suffix."""
        if state.plan_id != self.plan_id:
            raise ValueError("FIR state was created by a different filter plan.")
        values, coefficients = _promote_signal_and_taps(chunk, taps)
        if coefficients.shape != (self.tap_count,):
            raise ValueError(
                f"taps must have shape {(self.tap_count,)}; got {coefficients.shape}."
            )
        resolved_axis = _normalize_axis(self.axis, values.ndim)
        sample_count = int(values.shape[resolved_axis])
        if sample_count <= 0:
            raise ValueError("FIR chunks must have positive sample capacity.")
        expected_history_shape = _replace_axis_size(
            values.shape,
            resolved_axis,
            self.tap_count - 1,
        )
        if state.history.shape != expected_history_shape:
            raise ValueError(
                "FIR state history shape does not match the chunk streams: "
                f"expected {expected_history_shape}, got {state.history.shape}."
            )
        if state.history.dtype != values.dtype:
            raise TypeError(
                "FIR state dtype must equal the promoted signal/tap dtype; "
                f"got {state.history.dtype} and {values.dtype}."
            )
        valid, active = _valid_prefix(sample_count, valid_length)
        canonical = jnp.moveaxis(values, resolved_axis, -1)
        history = jnp.moveaxis(state.history, resolved_axis, -1)
        active_values = jnp.where(active, canonical, jnp.zeros_like(canonical))
        extended = jnp.concatenate((history, active_values), axis=-1)
        full = convolve(extended, coefficients, axis=-1, mode="full", method="direct")
        filtered = full[..., self.tap_count - 1 : self.tap_count - 1 + sample_count]
        filtered = jnp.where(active, filtered, jnp.zeros_like(filtered))

        history_length = self.tap_count - 1
        if history_length == 0:
            next_history = history
        else:
            history_indices = valid + jnp.arange(history_length, dtype=valid.dtype)
            next_history = jnp.take(extended, history_indices, axis=-1)
        next_state = FIRFilterState(
            history=jnp.moveaxis(next_history, -1, resolved_axis),
            sample_count=state.sample_count + valid,
            plan_id=self.plan_id,
        )
        result = FIRFilterResult(
            values=jnp.moveaxis(filtered, -1, resolved_axis),
            active=active,
            sample_offset=state.sample_count,
        )
        return next_state, result

    def flush(
        self,
        state: FIRFilterState,
        taps: ArrayLike,
        /,
    ) -> tuple[FIRFilterState, FIRFilterResult]:
        """Emit the finite FIR tail and return a reset state."""
        if state.plan_id != self.plan_id:
            raise ValueError("FIR state was created by a different filter plan.")
        resolved_axis = _normalize_axis(self.axis, state.history.ndim)
        tail_length = self.tap_count - 1
        tail_shape = _replace_axis_size(state.history.shape, resolved_axis, tail_length)
        if tail_length == 0:
            reset = FIRFilterState(
                history=jnp.zeros_like(state.history),
                sample_count=jnp.asarray(0, dtype=state.sample_count.dtype),
                plan_id=self.plan_id,
            )
            return reset, FIRFilterResult(
                values=jnp.zeros(tail_shape, dtype=state.history.dtype),
                active=jnp.zeros((0,), dtype=bool),
                sample_offset=state.sample_count,
            )
        zeros = jnp.zeros(tail_shape, dtype=state.history.dtype)
        _, result = self.step(state, zeros, taps)
        reset = FIRFilterState(
            history=jnp.zeros_like(state.history),
            sample_count=jnp.asarray(0, dtype=state.sample_count.dtype),
            plan_id=self.plan_id,
        )
        return reset, result


def fir_filter(
    values: ArrayLike,
    taps: ArrayLike,
    /,
    *,
    axis: int = -1,
) -> Array:
    """Apply one shared-tap zero-state causal FIR filter."""
    array, coefficients = _promote_signal_and_taps(values, taps)
    plan = FIRFilterPlan(int(coefficients.shape[0]), axis=axis)
    state = plan.initial_state(array.shape, dtype=array.dtype)
    _, result = plan.step(state, array, coefficients)
    return result.values


__all__ = [
    "FIRFilterPlan",
    "FIRFilterResult",
    "FIRFilterState",
    "fir_filter",
]
