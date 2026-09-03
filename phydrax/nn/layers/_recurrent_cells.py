#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import sqrt
from typing import Any, Literal

import equinox as eqx
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


def _recurrent_cell_input_width(cell: AbstractRecurrentCell, /) -> int | None:
    if isinstance(cell, (RNNCell, GRUCell, LSTMCell)):
        return cell.input_size
    if isinstance(cell, StackedRecurrentCell):
        return _recurrent_cell_input_width(cell.cells[0])
    return None


def _recurrent_cell_output_width(cell: AbstractRecurrentCell, /) -> int | None:
    if isinstance(cell, (RNNCell, GRUCell, LSTMCell)):
        return cell.hidden_size
    if isinstance(cell, StackedRecurrentCell):
        return _recurrent_cell_output_width(cell.cells[-1])
    return None


class StackedRecurrentCell(AbstractRecurrentOutputCell):
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

    def output_from_state(self, state: tuple[Any, ...], /) -> Any:
        if not isinstance(state, tuple) or len(state) != len(self.cells):
            raise TypeError("Stacked recurrent state must align with cells.")
        return _recurrent_output_from_state(self.cells[-1], state[-1])

    def input_width(self) -> int | None:
        return _recurrent_cell_input_width(self)

    def output_width(self) -> int | None:
        return _recurrent_cell_output_width(self)


__all__ = [
    "GRUCell",
    "LSTMCell",
    "RNNActivation",
    "RNNCell",
    "StackedRecurrentCell",
]
