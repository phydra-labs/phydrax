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
from ..._strict import StrictModule
from .._keys import EvalKey
from ._linear_recurrent_unit import _last_valid_array
from ._physical_sequence import normalize_physical_schedule
from ._recurrent import (
    AffineRecurrence,
    RecurrentBatch,
    RecurrentResult,
    run_affine_recurrence,
)


SelectiveSequenceExecution = Literal["serial", "associative"]


class CausalConvolutionResult(StrictModule):
    outputs: Array
    final_state: Array


class SelectiveStateSpaceState(StrictModule):
    convolution: Array
    recurrent: Array
    last_time: Array
    has_time: Array


class ResetAwareCausalConv1D(StrictModule):
    """Depthwise causal convolution whose history clears at packed resets."""

    weight: Array
    bias: Array | None
    channels: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        kernel_size: int = 4,
        /,
        *,
        use_bias: bool = True,
        dtype: Any = jnp.float32,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        if self.channels <= 0 or self.kernel_size <= 0:
            raise ValueError("channels and kernel_size must be positive.")
        resolved_dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(resolved_dtype, jnp.floating):
            raise TypeError("dtype must be a real floating dtype.")
        scale = 1.0 / math.sqrt(float(self.kernel_size))
        self.weight = scale * jr.normal(
            key,
            (self.kernel_size, self.channels),
            dtype=resolved_dtype,
        )
        self.bias = (
            jnp.zeros((self.channels,), dtype=resolved_dtype) if use_bias else None
        )

    def initial_state(self, case_shape: tuple[int, ...], /, *, dtype: Any) -> Array:
        return jnp.zeros(
            tuple(case_shape) + (self.kernel_size, self.channels),
            dtype=jnp.result_type(dtype, self.weight.dtype),
        )

    def evaluate_with_state(
        self,
        batch: RecurrentBatch,
        /,
        *,
        initial_state: Array | None = None,
    ) -> CausalConvolutionResult:
        if not isinstance(batch, RecurrentBatch):
            raise TypeError("batch must be a RecurrentBatch.")
        values = jnp.asarray(batch.inputs)
        if values.ndim < 1 or int(values.shape[-1]) != self.channels:
            raise ValueError(
                f"Causal convolution inputs must end in width {self.channels}."
            )
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("Causal convolution inputs must be real-valued.")
        compute_dtype = jnp.result_type(values.dtype, self.weight.dtype)
        values = values.astype(compute_dtype)
        history0 = (
            self.initial_state(batch.case_shape, dtype=compute_dtype)
            if initial_state is None
            else jnp.asarray(initial_state, dtype=compute_dtype)
        )
        expected = batch.case_shape + (self.kernel_size, self.channels)
        if history0.shape != expected:
            raise ValueError(f"initial_state must have shape {expected}.")
        sequence_axis = len(batch.case_shape)
        scan_values = jnp.moveaxis(values, sequence_axis, 0)
        scan_valid = jnp.moveaxis(batch.valid, -1, 0)
        scan_reset = jnp.moveaxis(batch.reset, -1, 0)
        weight = self.weight.astype(compute_dtype)
        bias = None if self.bias is None else self.bias.astype(compute_dtype)

        def step(history: Array, step_inputs: tuple[Array, Array, Array]):
            inputs, valid, reset = step_inputs
            clear = (valid & reset).reshape(valid.shape + (1, 1))
            restarted = jnp.where(clear, jnp.zeros_like(history), history)
            safe_inputs = jnp.where(valid[..., None], inputs, jnp.zeros_like(inputs))
            proposal = jnp.concatenate(
                (safe_inputs[..., None, :], restarted[..., :-1, :]),
                axis=-2,
            )
            select = valid.reshape(valid.shape + (1, 1))
            next_history = jnp.where(select, proposal, history)
            output = jnp.sum(next_history * weight, axis=-2)
            if bias is not None:
                output = output + bias
            output = jnp.where(valid[..., None], output, jnp.zeros_like(output))
            return next_history, output

        final_state, scan_outputs = jax.lax.scan(
            step,
            history0,
            (scan_values, scan_valid, scan_reset),
        )
        outputs = jnp.moveaxis(scan_outputs, 0, sequence_axis)
        return CausalConvolutionResult(outputs=outputs, final_state=final_state)

    def __call__(self, batch: RecurrentBatch, /) -> Array:
        return self.evaluate_with_state(batch).outputs


class SelectiveStateSpaceBlock(StrictModule):
    """Reset-aware selective diagonal state-space block with causal local mixing."""

    input_projection: Array
    input_bias: Array
    delta_weight: Array
    delta_bias: Array
    raw_decay: Array
    input_state_weight: Array
    output_state_weight: Array
    output_projection: Array
    output_bias: Array
    convolution: ResetAwareCausalConv1D
    input_size: int = eqx.field(static=True)
    inner_size: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    min_decay: float = eqx.field(static=True)
    min_step_scale: float = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        state_size: int,
        /,
        *,
        inner_size: int | None = None,
        convolution_size: int = 4,
        min_decay: float = 1e-4,
        min_step_scale: float = 1e-4,
        dtype: Any = jnp.float32,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.input_size = int(input_size)
        self.state_size = int(state_size)
        self.inner_size = 2 * self.input_size if inner_size is None else int(inner_size)
        if self.input_size <= 0 or self.state_size <= 0 or self.inner_size <= 0:
            raise ValueError("input_size, inner_size, and state_size must be positive.")
        self.min_decay = float(min_decay)
        self.min_step_scale = float(min_step_scale)
        if not math.isfinite(self.min_decay) or self.min_decay <= 0.0:
            raise ValueError("min_decay must be positive and finite.")
        if not math.isfinite(self.min_step_scale) or self.min_step_scale <= 0.0:
            raise ValueError("min_step_scale must be positive and finite.")
        resolved_dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(resolved_dtype, jnp.floating):
            raise TypeError("dtype must be a real floating dtype.")
        keys = jr.split(key, 8)
        input_scale = 1.0 / math.sqrt(float(self.input_size))
        self.input_projection = input_scale * jr.normal(
            keys[0],
            (2 * self.inner_size, self.input_size),
            dtype=resolved_dtype,
        )
        self.input_bias = jnp.zeros((2 * self.inner_size,), dtype=resolved_dtype)
        inner_scale = 1.0 / math.sqrt(float(self.inner_size))
        self.delta_weight = 0.05 * jr.normal(
            keys[1],
            (self.inner_size, self.inner_size),
            dtype=resolved_dtype,
        )
        delta_target = 1.0 - self.min_step_scale
        if delta_target <= 0.0:
            raise ValueError("min_step_scale must be less than one.")
        self.delta_bias = jnp.full(
            (self.inner_size,),
            math.log(math.expm1(delta_target)),
            dtype=resolved_dtype,
        )
        self.raw_decay = jr.normal(
            keys[2],
            (self.inner_size, self.state_size),
            dtype=resolved_dtype,
        )
        self.input_state_weight = inner_scale * jr.normal(
            keys[3],
            (self.state_size, self.inner_size),
            dtype=resolved_dtype,
        )
        self.output_state_weight = inner_scale * jr.normal(
            keys[4],
            (self.state_size, self.inner_size),
            dtype=resolved_dtype,
        )
        self.output_projection = inner_scale * jr.normal(
            keys[5],
            (self.input_size, self.inner_size),
            dtype=resolved_dtype,
        )
        self.output_bias = jnp.zeros((self.input_size,), dtype=resolved_dtype)
        self.convolution = ResetAwareCausalConv1D(
            self.inner_size,
            convolution_size,
            dtype=resolved_dtype,
            key=keys[6],
        )

    def initial_state(
        self,
        case_shape: tuple[int, ...],
        /,
        *,
        dtype: Any,
    ) -> SelectiveStateSpaceState:
        resolved_dtype = jnp.result_type(dtype, self.input_projection.dtype)
        return SelectiveStateSpaceState(
            convolution=self.convolution.initial_state(case_shape, dtype=resolved_dtype),
            recurrent=jnp.zeros(
                tuple(case_shape) + (self.inner_size, self.state_size),
                dtype=resolved_dtype,
            ),
            last_time=jnp.zeros(tuple(case_shape), dtype=resolved_dtype),
            has_time=jnp.zeros(tuple(case_shape), dtype=bool),
        )

    def evaluate_with_state(
        self,
        batch: RecurrentBatch,
        /,
        *,
        initial_state: SelectiveStateSpaceState | None = None,
        execution: SelectiveSequenceExecution = "associative",
        key: EvalKey = None,
    ) -> RecurrentResult:
        del key
        if not isinstance(batch, RecurrentBatch):
            raise TypeError("batch must be a RecurrentBatch.")
        values = jnp.asarray(batch.inputs)
        if values.ndim < 1 or int(values.shape[-1]) != self.input_size:
            raise ValueError(
                f"Selective block inputs must end in width {self.input_size}."
            )
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("Selective block inputs must be real-valued.")
        compute_dtype = jnp.result_type(values.dtype, self.input_projection.dtype)
        values = values.astype(compute_dtype)
        safe_values = jnp.where(batch.valid[..., None], values, jnp.zeros_like(values))
        state0 = (
            self.initial_state(batch.case_shape, dtype=compute_dtype)
            if initial_state is None
            else initial_state
        )
        if not isinstance(state0, SelectiveStateSpaceState):
            raise TypeError("initial_state must be a SelectiveStateSpaceState.")
        if (
            jnp.asarray(state0.last_time).shape != batch.case_shape
            or jnp.asarray(state0.has_time).shape != batch.case_shape
        ):
            raise ValueError(
                "Initial selective time state has incompatible case dimensions."
            )
        projected = oe.contract(
            "oi,...ti->...to",
            self.input_projection.astype(compute_dtype),
            safe_values,
        ) + self.input_bias.astype(compute_dtype)
        content, gate = jnp.split(projected, 2, axis=-1)
        convolution_result = self.convolution.evaluate_with_state(
            RecurrentBatch(
                content,
                batch.valid,
                reset=batch.reset,
                time=batch.time,
                time_direction=batch.time_direction,
            ),
            initial_state=state0.convolution,
        )
        content = jax.nn.silu(convolution_result.outputs)
        gate = jax.nn.silu(gate)

        if batch.time is None:
            physical_step = batch.valid.astype(compute_dtype)
            final_time = jnp.asarray(state0.last_time, dtype=compute_dtype)
            final_has_time = jnp.asarray(state0.has_time, dtype=bool)
        else:
            direction_sign = jnp.asarray(
                1.0 if batch.time_direction == "forward" else -1.0,
                dtype=compute_dtype,
            )
            directed_times, _, _, continuation = normalize_physical_schedule(
                direction_sign * batch.time,
                case_shape=batch.case_shape,
                sequence_length=batch.sequence_length,
                mask=batch.valid,
                reset=batch.reset,
                dtype=compute_dtype,
                require_prefix=False,
            )
            times = direction_sign * directed_times
            previous_time = jnp.asarray(state0.last_time, dtype=compute_dtype)
            previous_has_time = jnp.asarray(state0.has_time, dtype=bool)
            if (
                previous_time.shape != batch.case_shape
                or previous_has_time.shape != batch.case_shape
            ):
                raise ValueError(
                    "Initial selective time state has incompatible case dimensions."
                )
            first_continuation = (
                batch.valid[..., 0] & ~batch.reset[..., 0] & previous_has_time
            )
            directed_first_step = direction_sign * (times[..., 0] - previous_time)
            times = eqx.error_if(
                times,
                jnp.any(first_continuation & (directed_first_step < 0)),
                "Continuation times must follow the declared physical-time "
                "direction across sequence chunks.",
            )
            first_step = jnp.where(
                first_continuation,
                directed_first_step,
                jnp.zeros_like(previous_time),
            )
            intervals = direction_sign * (times[..., 1:] - times[..., :-1])
            physical_step = jnp.concatenate(
                (
                    first_step[..., None],
                    jnp.where(continuation, intervals, jnp.zeros_like(intervals)),
                ),
                axis=-1,
            )
            scan_times = jnp.moveaxis(times, -1, 0)
            scan_valid = jnp.moveaxis(batch.valid, -1, 0)

            def update_time(current: Array, item: tuple[Array, Array]) -> Array:
                time, valid = item
                return jnp.where(valid, time, current)

            final_time, _ = jax.lax.scan(
                lambda current, item: (update_time(current, item), None),
                previous_time,
                (scan_times, scan_valid),
            )
            final_has_time = previous_has_time | jnp.any(batch.valid, axis=-1)
        step_scale = (
            jax.nn.softplus(
                oe.contract(
                    "oi,...ti->...to",
                    self.delta_weight.astype(compute_dtype),
                    content,
                )
                + self.delta_bias.astype(compute_dtype)
            )
            + self.min_step_scale
        )
        effective_step = physical_step[..., None] * step_scale
        decay = jax.nn.softplus(self.raw_decay.astype(compute_dtype)) + self.min_decay
        exponent = effective_step[..., None] * decay
        transition = jnp.exp(-exponent)
        coefficient = -jnp.expm1(-exponent) / decay
        input_state = oe.contract(
            "mi,...ti->...tm",
            self.input_state_weight.astype(compute_dtype),
            content,
        )
        drive = content[..., :, None] * input_state[..., None, :]
        addition = coefficient * drive
        transition = jnp.where(
            batch.valid[..., None, None],
            transition,
            jnp.ones_like(transition),
        )
        addition = jnp.where(
            batch.valid[..., None, None],
            addition,
            jnp.zeros_like(addition),
        )
        expected_recurrent = batch.case_shape + (self.inner_size, self.state_size)
        recurrent0 = jnp.asarray(state0.recurrent, dtype=compute_dtype)
        if recurrent0.shape != expected_recurrent:
            raise ValueError(
                f"initial recurrent state must have shape {expected_recurrent}."
            )
        recurrence = AffineRecurrence(
            jnp.zeros((self.inner_size, self.state_size), dtype=compute_dtype)
        )
        recurrence_result = run_affine_recurrence(
            recurrence,
            RecurrentBatch(
                (transition, addition),
                batch.valid,
                reset=batch.reset,
                time=batch.time,
                time_direction=batch.time_direction,
            ),
            initial_state=recurrent0,
            execution=execution,
        )
        output_state = oe.contract(
            "mi,...ti->...tm",
            self.output_state_weight.astype(compute_dtype),
            content,
        )
        latent = jnp.sum(
            recurrence_result.states * output_state[..., None, :],
            axis=-1,
        )
        latent = latent * gate
        branch = oe.contract(
            "oi,...ti->...to",
            self.output_projection.astype(compute_dtype),
            latent,
        ) + self.output_bias.astype(compute_dtype)
        outputs = jnp.where(
            batch.valid[..., None],
            safe_values + branch,
            jnp.zeros_like(safe_values),
        )
        final_state = SelectiveStateSpaceState(
            convolution=convolution_result.final_state,
            recurrent=recurrence_result.final_state,
            last_time=final_time,
            has_time=final_has_time,
        )
        return RecurrentResult(
            states=recurrence_result.states,
            outputs=outputs,
            final_state=final_state,
            final_output=_last_valid_array(outputs, batch.valid),
        )

    def __call__(
        self,
        batch: RecurrentBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        return self.evaluate_with_state(batch, key=key).outputs


__all__ = [
    "CausalConvolutionResult",
    "ResetAwareCausalConv1D",
    "SelectiveSequenceExecution",
    "SelectiveStateSpaceBlock",
    "SelectiveStateSpaceState",
]
