#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import gcd

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
from ._windows import kaiser_window


def _raw_output_length(
    sample_count: int,
    tap_count: int,
    up: int,
    down: int,
    /,
) -> int:
    return ((sample_count - 1) * up + tap_count - 1) // down + 1


def _phase_taps(taps: Array, up: int, /) -> Array:
    phase_length = (int(taps.shape[0]) + up - 1) // up
    padded_length = phase_length * up
    padded = jnp.pad(taps, (0, padded_length - int(taps.shape[0])))
    return padded.reshape((phase_length, up)).T


def _evaluate_polyphase_outputs(
    values: Array,
    taps: Array,
    *,
    up: int,
    down: int,
    output_indices: Array,
    input_origin: Array,
    valid_input_end: Array,
) -> Array:
    """Evaluate global downsampled convolution indices from a local input window."""
    phases = _phase_taps(taps, up)
    phase_length = int(phases.shape[1])
    high_rate_indices = output_indices * down
    phase_indices = jnp.mod(high_rate_indices, up)
    base_inputs = high_rate_indices // up
    global_inputs = (
        base_inputs[:, None]
        - jnp.arange(
            phase_length,
            dtype=base_inputs.dtype,
        )[None, :]
    )
    local_inputs = global_inputs - input_origin
    local_count = int(values.shape[-1])
    clipped = jnp.clip(local_inputs, 0, local_count - 1)
    gathered = jnp.take(values, clipped, axis=-1)
    valid = (
        (global_inputs >= 0)
        & (global_inputs < valid_input_end)
        & (local_inputs >= 0)
        & (local_inputs < local_count)
    )
    coefficients = phases[phase_indices]
    return jnp.sum(
        jnp.where(valid, gathered, jnp.zeros_like(gathered)) * coefficients,
        axis=-1,
    )


def upfirdn(
    values: ArrayLike,
    taps: ArrayLike,
    /,
    *,
    up: int = 1,
    down: int = 1,
    axis: int = -1,
) -> Array:
    """Upsample, FIR-filter, and downsample without materializing inserted zeros.

    ``up`` and ``down`` are not reduced and the supplied taps are not scaled.
    Output phase is zero and finite records use zero extension.
    """
    up_factor = _positive_int(up, "up")
    down_factor = _positive_int(down, "down")
    array, coefficients = _promote_signal_and_taps(values, taps)
    resolved_axis = _normalize_axis(axis, array.ndim)
    canonical = jnp.moveaxis(array, resolved_axis, -1)
    sample_count = int(canonical.shape[-1])
    if sample_count <= 0:
        raise ValueError("The signal axis must contain at least one sample.")
    output_length = _raw_output_length(
        sample_count,
        int(coefficients.shape[0]),
        up_factor,
        down_factor,
    )
    output_indices = jnp.arange(output_length, dtype=jnp.int64)
    output = _evaluate_polyphase_outputs(
        canonical,
        coefficients,
        up=up_factor,
        down=down_factor,
        output_indices=output_indices,
        input_origin=jnp.asarray(0, dtype=output_indices.dtype),
        valid_input_end=jnp.asarray(sample_count, dtype=output_indices.dtype),
    )
    return jnp.moveaxis(output, -1, resolved_axis)


def kaiser_sinc_resampling_filter(
    up: int,
    down: int,
    /,
    *,
    half_width: int = 10,
    beta: ArrayLike = 5.0,
    dtype: jnp.dtype | None = None,
) -> Array:
    """Construct an odd unit-DC Kaiser-sinc prototype for a rational ratio."""
    up_factor = _positive_int(up, "up")
    down_factor = _positive_int(down, "down")
    width = _positive_int(half_width, "half_width")
    common = gcd(up_factor, down_factor)
    up_factor //= common
    down_factor //= common
    maximum_rate = max(up_factor, down_factor)
    half_length = width * maximum_rate
    resolved_dtype = jnp.asarray(0.0).dtype if dtype is None else jnp.dtype(dtype)
    if not jnp.issubdtype(resolved_dtype, jnp.floating):
        raise TypeError("Resampling prototypes require a real floating dtype.")
    positions = jnp.arange(
        -half_length,
        half_length + 1,
        dtype=resolved_dtype,
    )
    cutoff = jnp.asarray(1.0 / maximum_rate, dtype=resolved_dtype)
    window = kaiser_window(
        2 * half_length + 1,
        beta,
        periodic=False,
        dtype=resolved_dtype,
    )
    prototype = cutoff * jnp.sinc(cutoff * positions) * window
    return prototype / jnp.sum(prototype)


def resample_poly(
    values: ArrayLike,
    up: int,
    down: int,
    /,
    *,
    taps: ArrayLike | None = None,
    axis: int = -1,
) -> Array:
    """Resample a finite record with centered zero-extended polyphase filtering."""
    up_factor = _positive_int(up, "up")
    down_factor = _positive_int(down, "down")
    common = gcd(up_factor, down_factor)
    up_factor //= common
    down_factor //= common
    array = jnp.asarray(values)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    resolved_axis = _normalize_axis(axis, array.ndim)
    sample_count = int(array.shape[resolved_axis])
    if sample_count <= 0:
        raise ValueError("The signal axis must contain at least one sample.")
    if up_factor == down_factor == 1:
        return jnp.array(array, copy=True)

    if taps is None:
        prototype = kaiser_sinc_resampling_filter(
            up_factor,
            down_factor,
            dtype=array.real.dtype,
        )
    else:
        _, prototype = _promote_signal_and_taps(array, taps)
    if int(prototype.shape[0]) % 2 == 0:
        raise ValueError("Centered rational resampling requires an odd tap count.")
    dtype = jnp.result_type(array.dtype, prototype.dtype)
    array = array.astype(dtype)
    prototype = prototype.astype(dtype)

    output_length = (sample_count * up_factor + down_factor - 1) // down_factor
    half_length = (int(prototype.shape[0]) - 1) // 2
    pre_padding = down_factor - half_length % down_factor
    pre_remove = (half_length + pre_padding) // down_factor
    post_padding = 0
    while (
        _raw_output_length(
            sample_count,
            int(prototype.shape[0]) + pre_padding + post_padding,
            up_factor,
            down_factor,
        )
        < output_length + pre_remove
    ):
        post_padding += 1
    scaled = prototype * jnp.asarray(up_factor, dtype=prototype.dtype)
    padded = jnp.pad(scaled, (pre_padding, post_padding))
    filtered = upfirdn(
        array,
        padded,
        up=up_factor,
        down=down_factor,
        axis=resolved_axis,
    )
    slices = [slice(None)] * filtered.ndim
    slices[resolved_axis] = slice(pre_remove, pre_remove + output_length)
    return filtered[tuple(slices)]


class RationalResamplingState(StrictModule):
    """Immutable polyphase history and absolute stream cursors."""

    history: Array
    input_count: Array
    output_count: Array
    plan_id: str = eqx.field(static=True)


class RationalResamplingResult(StrictModule):
    """Fixed-capacity causal resampling output."""

    values: Array
    active: Array
    sample_offset: Array


class RationalResamplingPlan(StrictModule, NonTrainableState):
    """Static topology for causal fixed-capacity rational resampling."""

    up: int = eqx.field(static=True)
    down: int = eqx.field(static=True)
    tap_count: int = eqx.field(static=True)
    chunk_length: int = eqx.field(static=True)
    axis: int = eqx.field(static=True)
    history_length: int = eqx.field(static=True)
    output_capacity: int = eqx.field(static=True)
    tail_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        up: int,
        down: int,
        tap_count: int,
        chunk_length: int,
        /,
        *,
        axis: int = -1,
    ):
        up_factor = _positive_int(up, "up")
        down_factor = _positive_int(down, "down")
        common = gcd(up_factor, down_factor)
        self.up = up_factor // common
        self.down = down_factor // common
        self.tap_count = _positive_int(tap_count, "tap_count")
        self.chunk_length = _positive_int(chunk_length, "chunk_length")
        if self.chunk_length % self.down != 0:
            raise ValueError("chunk_length must be divisible by the reduced down factor.")
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("axis must be an integer.")
        self.axis = axis
        self.history_length = (self.tap_count + self.up - 1) // self.up - 1
        self.output_capacity = self.chunk_length * self.up // self.down
        self.tail_capacity = (self.tap_count - 1 + self.down - 1) // self.down + 1
        self.plan_id = canonical_fingerprint(
            {
                "kind": "rational_resampling",
                "up": self.up,
                "down": self.down,
                "tap_count": self.tap_count,
                "chunk_length": self.chunk_length,
                "axis": self.axis,
            }
        )

    def initial_state(
        self,
        input_shape: Sequence[int],
        /,
        *,
        dtype: jnp.dtype,
    ) -> RationalResamplingState:
        """Return zero history for fixed-capacity chunks with ``input_shape``."""
        shape = tuple(int(size) for size in input_shape)
        resolved_axis = _normalize_axis(self.axis, len(shape))
        if shape[resolved_axis] != self.chunk_length:
            raise ValueError(
                f"The input sample axis must have length {self.chunk_length}; "
                f"got {shape[resolved_axis]}."
            )
        resolved_dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(resolved_dtype, jnp.inexact):
            raise TypeError("Resampling state dtype must be inexact.")
        history_shape = _replace_axis_size(
            shape,
            resolved_axis,
            self.history_length,
        )
        return RationalResamplingState(
            history=jnp.zeros(history_shape, dtype=resolved_dtype),
            input_count=jnp.asarray(0, dtype=jnp.int64),
            output_count=jnp.asarray(0, dtype=jnp.int64),
            plan_id=self.plan_id,
        )

    def step(
        self,
        state: RationalResamplingState,
        chunk: ArrayLike,
        taps: ArrayLike,
        /,
        *,
        valid_length: ArrayLike | None = None,
    ) -> tuple[RationalResamplingState, RationalResamplingResult]:
        """Causally resample one fixed-capacity input chunk."""
        if state.plan_id != self.plan_id:
            raise ValueError("Resampling state was created by a different plan.")
        values, prototype = _promote_signal_and_taps(chunk, taps)
        if prototype.shape != (self.tap_count,):
            raise ValueError(
                f"taps must have shape {(self.tap_count,)}; got {prototype.shape}."
            )
        resolved_axis = _normalize_axis(self.axis, values.ndim)
        if values.shape[resolved_axis] != self.chunk_length:
            raise ValueError(
                f"The input sample axis must have length {self.chunk_length}; "
                f"got {values.shape[resolved_axis]}."
            )
        expected_history_shape = _replace_axis_size(
            values.shape,
            resolved_axis,
            self.history_length,
        )
        if state.history.shape != expected_history_shape:
            raise ValueError(
                "Resampling state history shape does not match the chunk streams: "
                f"expected {expected_history_shape}, got {state.history.shape}."
            )
        if state.history.dtype != values.dtype:
            raise TypeError(
                "Resampling state dtype must equal the promoted signal/tap dtype; "
                f"got {state.history.dtype} and {values.dtype}."
            )
        valid, input_active = _valid_prefix(self.chunk_length, valid_length)
        canonical = jnp.moveaxis(values, resolved_axis, -1)
        history = jnp.moveaxis(state.history, resolved_axis, -1)
        active_values = jnp.where(
            input_active,
            canonical,
            jnp.zeros_like(canonical),
        )
        extended = jnp.concatenate((history, active_values), axis=-1)
        next_input_count = state.input_count + valid
        next_output_count = (next_input_count * self.up + self.down - 1) // self.down
        output_indices = state.output_count + jnp.arange(
            self.output_capacity,
            dtype=state.output_count.dtype,
        )
        output_active = output_indices < next_output_count
        scaled_taps = prototype * jnp.asarray(self.up, dtype=prototype.dtype)
        evaluated = _evaluate_polyphase_outputs(
            extended,
            scaled_taps,
            up=self.up,
            down=self.down,
            output_indices=output_indices,
            input_origin=state.input_count - self.history_length,
            valid_input_end=next_input_count,
        )
        evaluated = jnp.where(
            output_active,
            evaluated,
            jnp.zeros_like(evaluated),
        )
        if self.history_length == 0:
            next_history = history
        else:
            history_indices = valid + jnp.arange(
                self.history_length,
                dtype=valid.dtype,
            )
            next_history = jnp.take(extended, history_indices, axis=-1)
        next_state = RationalResamplingState(
            history=jnp.moveaxis(next_history, -1, resolved_axis),
            input_count=next_input_count,
            output_count=next_output_count,
            plan_id=self.plan_id,
        )
        result = RationalResamplingResult(
            values=jnp.moveaxis(evaluated, -1, resolved_axis),
            active=output_active,
            sample_offset=state.output_count,
        )
        return next_state, result

    def flush(
        self,
        state: RationalResamplingState,
        taps: ArrayLike,
        /,
    ) -> tuple[RationalResamplingState, RationalResamplingResult]:
        """Emit the causal polyphase tail and return a reset state."""
        if state.plan_id != self.plan_id:
            raise ValueError("Resampling state was created by a different plan.")
        prototype = jnp.asarray(taps)
        if not jnp.issubdtype(prototype.dtype, jnp.inexact):
            prototype = prototype.astype(float)
        if prototype.shape != (self.tap_count,):
            raise ValueError(
                f"taps must have shape {(self.tap_count,)}; got {prototype.shape}."
            )
        dtype = jnp.result_type(state.history.dtype, prototype.dtype)
        if state.history.dtype != dtype:
            raise TypeError(
                "Resampling state dtype must equal the promoted state/tap dtype; "
                f"got {state.history.dtype} and {dtype}."
            )
        prototype = prototype.astype(dtype)
        resolved_axis = _normalize_axis(self.axis, state.history.ndim)
        canonical_history = jnp.moveaxis(state.history, resolved_axis, -1)
        output_indices = state.output_count + jnp.arange(
            self.tail_capacity,
            dtype=state.output_count.dtype,
        )
        final_output_count = jnp.where(
            state.input_count > 0,
            ((state.input_count - 1) * self.up + self.tap_count - 1) // self.down + 1,
            0,
        )
        active = output_indices < final_output_count
        if self.history_length == 0:
            evaluated = jnp.zeros(
                (*canonical_history.shape[:-1], self.tail_capacity),
                dtype=dtype,
            )
        else:
            scaled_taps = prototype * jnp.asarray(self.up, dtype=prototype.dtype)
            evaluated = _evaluate_polyphase_outputs(
                canonical_history,
                scaled_taps,
                up=self.up,
                down=self.down,
                output_indices=output_indices,
                input_origin=state.input_count - self.history_length,
                valid_input_end=state.input_count,
            )
        evaluated = jnp.where(active, evaluated, jnp.zeros_like(evaluated))
        reset = RationalResamplingState(
            history=jnp.zeros_like(state.history),
            input_count=jnp.asarray(0, dtype=state.input_count.dtype),
            output_count=jnp.asarray(0, dtype=state.output_count.dtype),
            plan_id=self.plan_id,
        )
        result = RationalResamplingResult(
            values=jnp.moveaxis(evaluated, -1, resolved_axis),
            active=active,
            sample_offset=state.output_count,
        )
        return reset, result


__all__ = [
    "RationalResamplingPlan",
    "RationalResamplingResult",
    "RationalResamplingState",
    "kaiser_sinc_resampling_filter",
    "resample_poly",
    "upfirdn",
]
