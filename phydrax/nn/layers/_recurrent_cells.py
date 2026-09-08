#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from functools import partial
from math import isfinite, sqrt
from typing import Any, Literal

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

import phydrax.ein as ein

from ..._doc import DOC_KEY0
from .._keys import EvalKey, split_eval_key
from ._recurrent import (
    _recurrent_output_from_state,
    AbstractRecurrentCell,
    AbstractRecurrentOutputCell,
    AbstractTimeAwareRecurrentCell,
)


RNNActivation = Literal["tanh", "relu"]


def _validate_widths(input_size: int, hidden_size: int, /) -> tuple[int, int]:
    input_width = int(input_size)
    hidden_width = int(hidden_size)
    if input_width <= 0 or hidden_width <= 0:
        raise ValueError("input_size and hidden_size must be positive.")
    return input_width, hidden_width


def _validate_real_dtype(dtype: Any, /) -> jnp.dtype:
    resolved = jnp.dtype(dtype)
    if not jnp.issubdtype(resolved, jnp.floating):
        raise TypeError("Recurrent cells require a real floating dtype.")
    return resolved


def _validate_step_shapes(
    inputs: Array,
    state: Array,
    /,
    *,
    input_size: int,
    hidden_size: int,
) -> tuple[Array, Array]:
    values = jnp.asarray(inputs)
    hidden = jnp.asarray(state)
    if values.ndim < 1 or int(values.shape[-1]) != int(input_size):
        raise ValueError(f"inputs must end with width {input_size}; got {values.shape}.")
    if hidden.ndim < 1 or int(hidden.shape[-1]) != int(hidden_size):
        raise ValueError(f"state must end with width {hidden_size}; got {hidden.shape}.")
    if values.shape[:-1] != hidden.shape[:-1]:
        raise ValueError("Recurrent inputs and states must share their case shape.")
    return values, hidden


class RNNCell(AbstractRecurrentCell):
    """Elman recurrent cell with explicit packed-sequence semantics."""

    weight_ih: Array
    weight_hh: Array
    bias: Array | None
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    activation: RNNActivation = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        /,
        *,
        activation: RNNActivation = "tanh",
        use_bias: bool = True,
        dtype: Any = jnp.float32,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.input_size, self.hidden_size = _validate_widths(input_size, hidden_size)
        if activation not in ("tanh", "relu"):
            raise ValueError("activation must be 'tanh' or 'relu'.")
        self.activation = activation
        resolved_dtype = _validate_real_dtype(dtype)
        input_key, hidden_key, bias_key = jr.split(key, 3)
        limit = 1.0 / sqrt(float(self.hidden_size))
        self.weight_ih = jr.uniform(
            input_key,
            (self.hidden_size, self.input_size),
            minval=-limit,
            maxval=limit,
            dtype=resolved_dtype,
        )
        self.weight_hh = jr.uniform(
            hidden_key,
            (self.hidden_size, self.hidden_size),
            minval=-limit,
            maxval=limit,
            dtype=resolved_dtype,
        )
        self.bias = (
            jr.uniform(
                bias_key,
                (self.hidden_size,),
                minval=-limit,
                maxval=limit,
                dtype=resolved_dtype,
            )
            if use_bias
            else None
        )

    def initial_state(self, case_shape: tuple[int, ...], /, *, dtype: Any) -> Array:
        return jnp.zeros(
            tuple(case_shape) + (self.hidden_size,),
            dtype=jnp.result_type(dtype, self.weight_ih.dtype),
        )

    def step(
        self,
        state: Array,
        inputs: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> tuple[Array, Array]:
        del key
        values, hidden = _validate_step_shapes(
            inputs,
            state,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
        )
        preactivation = ein.contract("oi,...i->...o", self.weight_ih, values)
        preactivation = preactivation + ein.contract(
            "oi,...i->...o", self.weight_hh, hidden
        )
        if self.bias is not None:
            preactivation = preactivation + self.bias
        next_hidden = (
            jnp.tanh(preactivation)
            if self.activation == "tanh"
            else jnn.relu(preactivation)
        )
        return next_hidden, next_hidden

    def input_width(self) -> int:
        return self.input_size

    def output_width(self) -> int:
        return self.hidden_size


class GRUCell(AbstractRecurrentCell):
    """Vectorized adapter for Equinox's gated recurrent unit equations."""

    cell: eqx.nn.GRUCell
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        /,
        *,
        use_bias: bool = True,
        dtype: Any = jnp.float32,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.input_size, self.hidden_size = _validate_widths(input_size, hidden_size)
        resolved_dtype = _validate_real_dtype(dtype)
        self.cell = eqx.nn.GRUCell(
            self.input_size,
            self.hidden_size,
            use_bias=bool(use_bias),
            dtype=resolved_dtype,
            key=key,
        )

    def initial_state(self, case_shape: tuple[int, ...], /, *, dtype: Any) -> Array:
        return jnp.zeros(
            tuple(case_shape) + (self.hidden_size,),
            dtype=jnp.result_type(dtype, self.cell.weight_ih.dtype),
        )

    def step(
        self,
        state: Array,
        inputs: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> tuple[Array, Array]:
        del key
        values, hidden = _validate_step_shapes(
            inputs,
            state,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
        )
        bias = 0.0 if self.cell.bias is None else self.cell.bias
        bias_n = 0.0 if self.cell.bias_n is None else self.cell.bias_n
        input_gates = jnp.split(
            ein.contract("oi,...i->...o", self.cell.weight_ih, values) + bias,
            3,
            axis=-1,
        )
        hidden_gates = jnp.split(
            ein.contract("oi,...i->...o", self.cell.weight_hh, hidden),
            3,
            axis=-1,
        )
        reset = jnn.sigmoid(input_gates[0] + hidden_gates[0])
        update = jnn.sigmoid(input_gates[1] + hidden_gates[1])
        candidate = jnp.tanh(input_gates[2] + reset * (hidden_gates[2] + bias_n))
        next_hidden = candidate + update * (hidden - candidate)
        return next_hidden, next_hidden

    def input_width(self) -> int:
        return self.input_size

    def output_width(self) -> int:
        return self.hidden_size


class LSTMCell(AbstractRecurrentOutputCell):
    """Vectorized adapter for Equinox's long short-term memory equations."""

    cell: eqx.nn.LSTMCell
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        /,
        *,
        use_bias: bool = True,
        dtype: Any = jnp.float32,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.input_size, self.hidden_size = _validate_widths(input_size, hidden_size)
        resolved_dtype = _validate_real_dtype(dtype)
        self.cell = eqx.nn.LSTMCell(
            self.input_size,
            self.hidden_size,
            use_bias=bool(use_bias),
            dtype=resolved_dtype,
            key=key,
        )

    def initial_state(
        self,
        case_shape: tuple[int, ...],
        /,
        *,
        dtype: Any,
    ) -> tuple[Array, Array]:
        shape = tuple(case_shape) + (self.hidden_size,)
        resolved_dtype = jnp.result_type(dtype, self.cell.weight_ih.dtype)
        zeros = jnp.zeros(shape, dtype=resolved_dtype)
        return zeros, zeros

    def step(
        self,
        state: tuple[Array, Array],
        inputs: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> tuple[tuple[Array, Array], Array]:
        del key
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError("LSTM state must be a (hidden, cell) tuple.")
        values, hidden = _validate_step_shapes(
            inputs,
            state[0],
            input_size=self.input_size,
            hidden_size=self.hidden_size,
        )
        memory = jnp.asarray(state[1])
        if memory.shape != hidden.shape:
            raise ValueError("LSTM hidden and cell-memory states must have equal shapes.")
        gates = ein.contract("oi,...i->...o", self.cell.weight_ih, values)
        gates = gates + ein.contract("oi,...i->...o", self.cell.weight_hh, hidden)
        if self.cell.bias is not None:
            gates = gates + self.cell.bias
        input_gate, forget_gate, candidate, output_gate = jnp.split(gates, 4, axis=-1)
        input_gate = jnn.sigmoid(input_gate)
        forget_gate = jnn.sigmoid(forget_gate)
        candidate = jnp.tanh(candidate)
        output_gate = jnn.sigmoid(output_gate)
        next_memory = forget_gate * memory + input_gate * candidate
        next_hidden = output_gate * jnp.tanh(next_memory)
        return (next_hidden, next_memory), next_hidden

    def output_from_state(self, state: tuple[Array, Array], /) -> Array:
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError("LSTM state must be a (hidden, cell) tuple.")
        return state[0]

    def input_width(self) -> int:
        return self.input_size

    def output_width(self) -> int:
        return self.hidden_size


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _artificial_spike(margin: Array, family: str, /) -> Array:
    del family
    return (margin >= 0).astype(margin.dtype)


@_artificial_spike.defjvp
def _artificial_spike_jvp(family, primals, tangents):
    (margin,) = primals
    (tangent,) = tangents
    slope = (
        0.5 / jnp.square(1.0 + jnp.abs(margin))
        if family == "fast_sigmoid"
        else jnp.maximum(1.0 - jnp.abs(margin), 0.0)
    )
    return _artificial_spike(margin, family), slope * tangent


class ArtificialLIFCell(AbstractTimeAwareRecurrentCell, AbstractRecurrentOutputCell):
    """Clocked artificial leaky integrate-and-fire recurrent cell.

    State is ``(membrane, previous_spike)``; output is a real array of binary
    spikes. The affine drive ``W_ih x + W_hh previous_spike + bias`` is held over
    the arriving interval, in normalized membrane units above ``resting``.
    Leakage is integrated exactly, followed by at most one threshold/reset.
    This is not a continuous-time event-localizing biophysical simulator.

    Without batch times, every step advances ``dt_ms``. With batch times
    (in milliseconds), the native runner supplies elapsed durations: segment
    starts and repeated-time nodes have zero duration, leave both state leaves
    unchanged, and emit zero. The input at the arriving node drives its interval.
    Streaming requires both ``final_state`` and ``final_context``.

    The hard threshold uses a custom-JVP surrogate, not the derivative of the
    physical spike event. For margin ``m`` and width ``w``, ``fast_sigmoid`` has
    slope ``1 / (2*w*(1 + abs(m/w))**2)``; ``triangular`` has slope
    ``max(1 - abs(m/w), 0) / w``. Both preserve exact binary forward spikes.
    ``detach_reset`` stops the spike tangent only in the reset equation, not in
    the observable spikes or their next-step recurrent connection.

    Only ``weight_ih``, ``weight_hh`` and optional ``bias`` are trainable array
    leaves. Time constants, voltage levels, reset and surrogate policies are
    fixed scalar configuration. Use the existing recurrent model and optimizer.
    """

    weight_ih: Array
    weight_hh: Array
    bias: Array | None
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    time_constant_ms: float = eqx.field(static=True)
    dt_ms: float = eqx.field(static=True)
    threshold: float = eqx.field(static=True)
    reset: float = eqx.field(static=True)
    resting: float = eqx.field(static=True)
    reset_mode: Literal["subtract", "hard"] = eqx.field(static=True)
    surrogate: Literal["fast_sigmoid", "triangular"] = eqx.field(static=True)
    surrogate_width: float = eqx.field(static=True)
    detach_reset: bool = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        /,
        *,
        time_constant_ms: float = 20.0,
        dt_ms: float = 1.0,
        threshold: float = 1.0,
        reset: float = 0.0,
        resting: float = 0.0,
        reset_mode: Literal["subtract", "hard"] = "subtract",
        surrogate: Literal["fast_sigmoid", "triangular"] = "fast_sigmoid",
        surrogate_width: float = 1.0,
        detach_reset: bool = False,
        use_bias: bool = True,
        dtype: Any = jnp.float32,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.input_size, self.hidden_size = _validate_widths(input_size, hidden_size)
        constants = tuple(
            float(value)
            for value in (
                time_constant_ms,
                dt_ms,
                threshold,
                reset,
                resting,
                surrogate_width,
            )
        )
        if any(not isfinite(value) for value in constants):
            raise ValueError("Artificial LIF configuration must be finite.")
        tau, dt, threshold_, reset_, resting_, width = constants
        if tau <= 0.0 or dt <= 0.0 or width <= 0.0:
            raise ValueError(
                "time_constant_ms, dt_ms and surrogate_width must be positive."
            )
        if threshold_ <= max(reset_, resting_):
            raise ValueError("threshold must exceed reset and resting.")
        if reset_mode not in ("subtract", "hard"):
            raise ValueError("reset_mode must be 'subtract' or 'hard'.")
        if surrogate not in ("fast_sigmoid", "triangular"):
            raise ValueError("surrogate must be 'fast_sigmoid' or 'triangular'.")
        self.time_constant_ms, self.dt_ms = tau, dt
        self.threshold, self.reset, self.resting = threshold_, reset_, resting_
        self.reset_mode, self.surrogate = reset_mode, surrogate
        self.surrogate_width = width
        self.detach_reset = bool(detach_reset)
        resolved_dtype = _validate_real_dtype(dtype)
        input_key, hidden_key = jr.split(key)
        input_limit = 1.0 / sqrt(float(self.input_size))
        hidden_limit = 1.0 / sqrt(float(self.hidden_size))
        self.weight_ih = jr.uniform(
            input_key,
            (self.hidden_size, self.input_size),
            minval=-input_limit,
            maxval=input_limit,
            dtype=resolved_dtype,
        )
        self.weight_hh = jr.uniform(
            hidden_key,
            (self.hidden_size, self.hidden_size),
            minval=-hidden_limit,
            maxval=hidden_limit,
            dtype=resolved_dtype,
        )
        self.bias = (
            jnp.zeros((self.hidden_size,), dtype=resolved_dtype) if use_bias else None
        )

    def initial_state(
        self, case_shape: tuple[int, ...], /, *, dtype: Any
    ) -> tuple[Array, Array]:
        shape = tuple(case_shape) + (self.hidden_size,)
        resolved_dtype = jnp.result_type(dtype, self.weight_ih.dtype)
        return (
            jnp.full(shape, self.resting, dtype=resolved_dtype),
            jnp.zeros(shape, dtype=resolved_dtype),
        )

    def step(
        self,
        state: tuple[Array, Array],
        inputs: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> tuple[tuple[Array, Array], Array]:
        return self.step_with_context(
            state,
            inputs,
            time=jnp.asarray(0.0),
            interval=jnp.asarray(self.dt_ms),
            key=key,
        )

    def step_with_context(
        self,
        state: tuple[Array, Array],
        inputs: Array,
        /,
        *,
        time: Array,
        interval: Array,
        key: EvalKey = None,
    ) -> tuple[tuple[Array, Array], Array]:
        del time, key
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError("Artificial LIF state must be (membrane, previous_spike).")
        values, membrane = _validate_step_shapes(
            inputs,
            state[0],
            input_size=self.input_size,
            hidden_size=self.hidden_size,
        )
        previous_spike = jnp.asarray(state[1])
        if previous_spike.shape != membrane.shape:
            raise ValueError(
                "Artificial LIF membrane and spike states must have equal shapes."
            )
        elapsed = jnp.broadcast_to(
            jnp.asarray(interval, dtype=membrane.dtype), membrane.shape[:-1]
        )
        elapsed = eqx.error_if(
            elapsed,
            jnp.any(~jnp.isfinite(elapsed) | (elapsed < 0)),
            "Artificial LIF intervals must be finite and non-negative.",
        )[..., None]
        drive = ein.contract("oi,...i->...o", self.weight_ih, values)
        drive = drive + ein.contract("oi,...i->...o", self.weight_hh, previous_spike)
        if self.bias is not None:
            drive = drive + self.bias
        fraction = -jnp.expm1(-elapsed / self.time_constant_ms)
        charged = membrane + fraction * (self.resting - membrane + drive)
        spike = _artificial_spike(
            (charged - self.threshold) / self.surrogate_width, self.surrogate
        )
        reset_spike = jax.lax.stop_gradient(spike) if self.detach_reset else spike
        reset_membrane = (
            charged + reset_spike * (self.reset - charged)
            if self.reset_mode == "hard"
            else charged - reset_spike * (self.threshold - self.reset)
        )
        advancing = elapsed > 0
        next_state = (
            jnp.where(advancing, reset_membrane, membrane),
            jnp.where(advancing, spike, previous_spike),
        )
        return next_state, jnp.where(advancing, spike, jnp.zeros_like(spike))

    def output_from_state(self, state: tuple[Array, Array], /) -> Array:
        return state[1]

    def input_width(self) -> int:
        return self.input_size

    def output_width(self) -> int:
        return self.hidden_size


def _recurrent_cell_input_width(cell: AbstractRecurrentCell, /) -> int | None:
    if isinstance(cell, (RNNCell, GRUCell, LSTMCell, ArtificialLIFCell)):
        return cell.input_size
    if isinstance(cell, StackedRecurrentCell):
        return _recurrent_cell_input_width(cell.cells[0])
    return None


def _recurrent_cell_output_width(cell: AbstractRecurrentCell, /) -> int | None:
    if isinstance(cell, (RNNCell, GRUCell, LSTMCell, ArtificialLIFCell)):
        return cell.hidden_size
    if isinstance(cell, StackedRecurrentCell):
        return _recurrent_cell_output_width(cell.cells[-1])
    return None


class StackedRecurrentCell(AbstractTimeAwareRecurrentCell, AbstractRecurrentOutputCell):
    """Compose recurrent cells depth-wise within every sequence step."""

    cells: tuple[AbstractRecurrentCell, ...]

    def __init__(
        self, cells: tuple[AbstractRecurrentCell, ...] | list[AbstractRecurrentCell]
    ):
        resolved = tuple(cells)
        if not resolved or any(
            not isinstance(cell, AbstractRecurrentCell) for cell in resolved
        ):
            raise TypeError("cells must be a non-empty sequence of recurrent cells.")
        for earlier, later in zip(resolved[:-1], resolved[1:], strict=True):
            out_width = _recurrent_cell_output_width(earlier)
            in_width = _recurrent_cell_input_width(later)
            if out_width is not None and in_width is not None and out_width != in_width:
                raise ValueError(
                    "Adjacent recurrent cells have incompatible output/input widths: "
                    f"{out_width} and {in_width}."
                )
        self.cells = resolved

    def initial_state(
        self, case_shape: tuple[int, ...], /, *, dtype: Any
    ) -> tuple[Any, ...]:
        return tuple(cell.initial_state(case_shape, dtype=dtype) for cell in self.cells)

    def step(
        self,
        state: tuple[Any, ...],
        inputs: Any,
        /,
        *,
        key: EvalKey = None,
    ) -> tuple[tuple[Any, ...], Any]:
        if not isinstance(state, tuple) or len(state) != len(self.cells):
            raise TypeError("Stacked recurrent state must align with cells.")
        keys = split_eval_key(key, len(self.cells))
        value = inputs
        next_states = []
        for cell, cell_state, cell_key in zip(self.cells, state, keys, strict=True):
            next_state, value = cell.step(cell_state, value, key=cell_key)
            next_states.append(next_state)
        return tuple(next_states), value

    def step_with_context(
        self,
        state: tuple[Any, ...],
        inputs: Any,
        /,
        *,
        time: Array,
        interval: Array,
        key: EvalKey = None,
    ) -> tuple[tuple[Any, ...], Any]:
        if not isinstance(state, tuple) or len(state) != len(self.cells):
            raise TypeError("Stacked recurrent state must align with cells.")
        keys = split_eval_key(key, len(self.cells))
        value = inputs
        next_states = []
        for cell, cell_state, cell_key in zip(self.cells, state, keys, strict=True):
            if isinstance(cell, AbstractTimeAwareRecurrentCell):
                next_state, value = cell.step_with_context(
                    cell_state, value, time=time, interval=interval, key=cell_key
                )
            else:
                next_state, value = cell.step(cell_state, value, key=cell_key)
            next_states.append(next_state)
        return tuple(next_states), value

    def output_from_state(self, state: tuple[Any, ...], /) -> Any:
        if not isinstance(state, tuple) or len(state) != len(self.cells):
            raise TypeError("Stacked recurrent state must align with cells.")
        return _recurrent_output_from_state(self.cells[-1], state[-1])

    def input_width(self) -> int | None:
        return _recurrent_cell_input_width(self)

    def output_width(self) -> int | None:
        return _recurrent_cell_output_width(self)


__all__ = [
    "ArtificialLIFCell",
    "GRUCell",
    "LSTMCell",
    "RNNActivation",
    "RNNCell",
    "StackedRecurrentCell",
]
