#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from .._keys import EvalKey
from ._recurrent import (
    AffineRecurrence,
    RecurrentBatch,
    RecurrentResult,
    run_affine_recurrence,
)


LinearRecurrenceExecution = Literal["serial", "associative"]


def _last_valid_array(values: Array, valid: Array, /) -> Array:
    sequence_axis = valid.ndim - 1
    positions = jnp.broadcast_to(jnp.arange(valid.shape[-1]), valid.shape)
    last = jnp.max(jnp.where(valid, positions, -jnp.ones_like(positions)), axis=-1)
    safe_last = jnp.maximum(last, 0)
    index = safe_last.reshape(safe_last.shape + (1,) * (values.ndim - safe_last.ndim))
    selected = jnp.take_along_axis(values, index, axis=sequence_axis)
    selected = jnp.squeeze(selected, axis=sequence_axis)
    has_value = (last >= 0).reshape(last.shape + (1,) * (selected.ndim - last.ndim))
    return jnp.where(has_value, selected, jnp.zeros_like(selected))


class LinearRecurrentUnit(eqx.Module):
    """Stable complex-diagonal linear recurrence with real input/output maps."""

    raw_radius: Array
    phase: Array
    input_matrix_real: Array
    input_matrix_imag: Array
    output_matrix_real: Array
    output_matrix_imag: Array
    skip_matrix: Array
    input_size: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    min_radius: float = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        state_size: int,
        /,
        *,
        output_size: int | None = None,
        min_radius: float = 0.0,
        max_initial_radius: float = 0.99,
        dtype: Any = jnp.float32,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.input_size = int(input_size)
        self.state_size = int(state_size)
        self.output_size = self.input_size if output_size is None else int(output_size)
        if self.input_size <= 0 or self.state_size <= 0 or self.output_size <= 0:
            raise ValueError("input_size, state_size, and output_size must be positive.")
        self.min_radius = float(min_radius)
        maximum = float(max_initial_radius)
        if (
            not math.isfinite(self.min_radius)
            or not math.isfinite(maximum)
            or not 0.0 <= self.min_radius < maximum < 1.0
        ):
            raise ValueError(
                "Radii must satisfy 0 <= min_radius < max_initial_radius < 1."
            )
        resolved_dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(resolved_dtype, jnp.floating):
            raise TypeError("dtype must be a real floating dtype.")
        radius_key, phase_key, input_key, output_key = jr.split(key, 4)
        initial_radius = jr.uniform(
            radius_key,
            (self.state_size,),
            minval=self.min_radius + 0.25 * (maximum - self.min_radius),
            maxval=maximum,
            dtype=resolved_dtype,
        )
        normalized = (initial_radius - self.min_radius) / (1.0 - self.min_radius)
        self.raw_radius = jnp.log(normalized) - jnp.log1p(-normalized)
        self.phase = jr.uniform(
            phase_key,
            (self.state_size,),
            minval=-jnp.pi,
            maxval=jnp.pi,
            dtype=resolved_dtype,
        )
        input_scale = 1.0 / jnp.sqrt(float(self.input_size))
        input_parts = input_scale * jr.normal(
            input_key,
            (2, self.state_size, self.input_size),
            dtype=resolved_dtype,
        )
        self.input_matrix_real = input_parts[0]
        self.input_matrix_imag = input_parts[1]
        output_scale = 1.0 / jnp.sqrt(float(2 * self.state_size))
        output_parts = output_scale * jr.normal(
            output_key,
            (2, self.output_size, self.state_size),
            dtype=resolved_dtype,
        )
        self.output_matrix_real = output_parts[0]
        self.output_matrix_imag = output_parts[1]
        self.skip_matrix = jnp.zeros(
            (self.output_size, self.input_size), dtype=resolved_dtype
        )

    def eigenvalues(self, /) -> Array:
        radius = self.min_radius + (1.0 - self.min_radius) * jax.nn.sigmoid(
            self.raw_radius
        )
        return radius * jnp.exp(1j * self.phase)

    def initial_state(self, case_shape: tuple[int, ...], /, *, dtype: Any) -> Array:
        complex_dtype = jnp.result_type(dtype, self.raw_radius.dtype, jnp.complex64)
        return jnp.zeros(tuple(case_shape) + (self.state_size,), dtype=complex_dtype)

    def evaluate_with_state(
        self,
        batch: RecurrentBatch,
        /,
        *,
        initial_state: Array | None = None,
        execution: LinearRecurrenceExecution = "associative",
        key: EvalKey = None,
    ) -> RecurrentResult:
        del key
        if not isinstance(batch, RecurrentBatch):
            raise TypeError("batch must be a RecurrentBatch.")
        values = jnp.asarray(batch.inputs)
        if values.ndim < 1 or int(values.shape[-1]) != self.input_size:
            raise ValueError(f"Recurrent inputs must end in width {self.input_size}.")
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("LinearRecurrentUnit inputs must be real-valued.")
        compute_dtype = jnp.result_type(values.dtype, self.raw_radius.dtype)
        values = values.astype(compute_dtype)
        safe_values = jnp.where(batch.valid[..., None], values, jnp.zeros_like(values))
        complex_dtype = jnp.result_type(compute_dtype, jnp.complex64)
        input_matrix = self.input_matrix_real.astype(
            complex_dtype
        ) + 1j * self.input_matrix_imag.astype(complex_dtype)
        additions = oe.contract("...ti,mi->...tm", safe_values, input_matrix)
        transition = jnp.broadcast_to(
            self.eigenvalues().astype(complex_dtype), additions.shape
        )
        state0 = (
            self.initial_state(batch.case_shape, dtype=compute_dtype)
            if initial_state is None
            else jnp.asarray(initial_state, dtype=complex_dtype)
        )
        if state0.shape != batch.case_shape + (self.state_size,):
            raise ValueError("initial_state has an incompatible case or state shape.")
        recurrence = AffineRecurrence(jnp.zeros((self.state_size,), dtype=complex_dtype))
        recurrence_result = run_affine_recurrence(
            recurrence,
            RecurrentBatch(
                (transition, additions),
                batch.valid,
                reset=batch.reset,
                time=batch.time,
                time_direction=batch.time_direction,
            ),
            initial_state=state0,
            execution=execution,
        )
        output_matrix = self.output_matrix_real.astype(
            complex_dtype
        ) + 1j * self.output_matrix_imag.astype(complex_dtype)
        dynamic = 2.0 * jnp.real(
            oe.contract("om,...tm->...to", output_matrix, recurrence_result.states)
        )
        skip = oe.contract("oi,...ti->...to", self.skip_matrix, safe_values)
        outputs = jnp.where(
            batch.valid[..., None],
            dynamic.astype(skip.dtype) + skip,
            jnp.zeros_like(skip),
        )
        return RecurrentResult(
            states=recurrence_result.states,
            outputs=outputs,
            final_state=recurrence_result.final_state,
            final_output=_last_valid_array(outputs, batch.valid),
        )


__all__ = ["LinearRecurrenceExecution", "LinearRecurrentUnit"]
