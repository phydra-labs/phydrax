# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

import math
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax._strict import StrictModule
from phydrax.nn._keys import EvalKey
from phydrax.nn._utils import _get_size
from phydrax.nn.layers import AffineRecurrence, RecurrentBatch, run_affine_recurrence
from phydrax.nn.layers._physical_sequence import (
    normalize_physical_schedule,
    stable_exponential_phi,
)
from phydrax.nn.operator.data import FunctionSamples, OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel


SelectiveInputIntegration = Literal["zoh", "linear"]
SelectiveExecution = Literal["recurrent", "associative"]


class SelectiveStateSpaceDiagnostics(StrictModule):
    """Observed physical/effective step ranges and extrapolation evidence."""

    minimum_physical_step: Array
    maximum_physical_step: Array
    minimum_effective_step: Array
    maximum_effective_step: Array
    extrapolated_fraction: Array
    interval_count: Array
    segment_count: Array


def _contract_configuration(
    model: SelectiveStateSpaceMixer,
) -> tuple[tuple[str, object], ...]:
    return (
        ("state_size", model.state_size),
        ("input_integration", model.input_integration),
        ("execution", model.execution),
        ("training_delta_range", model.training_delta_range),
        ("source_key", model.source_key),
        ("method_id", model.method_id),
    )


class SelectiveStateSpaceMixer(AbstractOperatorModel):
    r"""Input-selective exact diagonal state-space mixer on physical schedules.

    Every continuation interval uses an input-dependent positive time scale,
    input gate, and readout gate while remaining affine in the latent state.
    Consequently serial recurrence and associative affine composition are
    mathematically identical. Physical intervals use exact zero-order-hold or
    linearly interpolated integration. Valid reset nodes restart from the
    canonical zero state; ``initial_state`` carries streaming state into a chunk.
    Invalid padded nodes preserve state and emit zero.
    """

    operator_architecture = "SelectiveStateSpaceMixer"
    _operator_contract_configuration = staticmethod(_contract_configuration)

    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    state_size: int
    input_integration: SelectiveInputIntegration
    execution: SelectiveExecution
    time_axis: str
    source_key: str | None
    min_decay: float
    min_step_scale: float
    training_delta_range: tuple[float, float] | None
    discretization: Literal["exact"]
    approximation: Literal["input-selective-diagonal"]

    raw_decay: Array
    delta_weight: Array
    delta_bias: Array
    input_matrix: Array
    input_gate_weight: Array
    input_gate_bias: Array
    output_matrix: Array
    output_gate_weight: Array
    output_gate_bias: Array
    skip_matrix: Array

    def __init__(
        self,
        *,
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] | None = None,
        state_size: int = 32,
        input_integration: SelectiveInputIntegration = "zoh",
        execution: SelectiveExecution = "recurrent",
        time_axis: str = "time",
        source_key: str | None = None,
        initial_decay: float = 0.1,
        min_decay: float = 1e-4,
        min_step_scale: float = 1e-4,
        training_delta_range: tuple[float, float] | None = None,
        dtype: Any = jnp.float32,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_size = in_channels
        self.out_size = in_channels if out_channels is None else out_channels
        self.state_size = int(state_size)
        self.input_integration = input_integration
        self.execution = execution
        self.time_axis = str(time_axis)
        self.source_key = source_key
        self.discretization = "exact"
        self.approximation = "input-selective-diagonal"
        if self.state_size <= 0:
            raise ValueError("state_size must be positive.")
        if input_integration not in ("zoh", "linear"):
            raise ValueError("input_integration must be 'zoh' or 'linear'.")
        if execution not in ("recurrent", "associative"):
            raise ValueError("execution must be 'recurrent' or 'associative'.")
        if not self.time_axis:
            raise ValueError("time_axis must be non-empty.")
        in_count = _get_size(self.in_size)
        out_count = _get_size(self.out_size)
        if in_count <= 0 or out_count <= 0:
            raise ValueError("Input and output channel counts must be positive.")
        real_dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(real_dtype, jnp.floating):
            raise TypeError("dtype must be a real floating dtype.")
        initial_decay_value = float(initial_decay)
        self.min_decay = float(min_decay)
        self.min_step_scale = float(min_step_scale)
        if (
            not math.isfinite(initial_decay_value)
            or not math.isfinite(self.min_decay)
            or initial_decay_value <= self.min_decay
            or self.min_decay <= 0.0
        ):
            raise ValueError("initial_decay must exceed positive finite min_decay.")
        if not math.isfinite(self.min_step_scale) or not 0.0 < self.min_step_scale < 1.0:
            raise ValueError("min_step_scale must lie strictly between zero and one.")
        if training_delta_range is None:
            self.training_delta_range = None
        else:
            lower, upper = (float(value) for value in training_delta_range)
            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower < 0.0
                or upper <= lower
            ):
                raise ValueError(
                    "training_delta_range must be finite non-negative increasing bounds."
                )
            self.training_delta_range = (lower, upper)

        keys = jr.split(key, 6)
        unconstrained_decay = initial_decay_value - self.min_decay
        raw_decay_location = unconstrained_decay + jnp.log(
            -jnp.expm1(-unconstrained_decay)
        )
        self.raw_decay = raw_decay_location + 0.05 * jr.normal(
            keys[0], (self.state_size,), dtype=real_dtype
        )
        delta_target = 1.0 - self.min_step_scale
        self.delta_bias = jnp.full(
            (self.state_size,), math.log(math.expm1(delta_target)), dtype=real_dtype
        )
        self.delta_weight = 0.05 * jr.normal(
            keys[1], (self.state_size, in_count), dtype=real_dtype
        )
        self.input_matrix = jr.normal(
            keys[2], (self.state_size, in_count), dtype=real_dtype
        ) / jnp.sqrt(float(in_count))
        self.input_gate_weight = jr.normal(
            keys[3], (self.state_size, in_count), dtype=real_dtype
        ) / jnp.sqrt(float(in_count))
        self.input_gate_bias = jnp.zeros((self.state_size,), dtype=real_dtype)
        self.output_matrix = jr.normal(
            keys[4], (out_count, self.state_size), dtype=real_dtype
        ) / jnp.sqrt(float(self.state_size))
        self.output_gate_weight = jr.normal(
            keys[5], (self.state_size, in_count), dtype=real_dtype
        ) / jnp.sqrt(float(in_count))
        self.output_gate_bias = jnp.zeros((self.state_size,), dtype=real_dtype)
        self.skip_matrix = jnp.zeros((out_count, in_count), dtype=real_dtype)

    @property
    def method_id(self) -> str:
        return (
            f"selective-state-space-mixer/exact/{self.input_integration}/{self.execution}"
        )

    def decay_rates(self, /) -> Array:
        """Return strictly positive continuous-time latent decay rates."""
        return jax.nn.softplus(self.raw_decay) + self.min_decay

    @staticmethod
    def _phi_functions(z: Array, /) -> tuple[Array, Array]:
        return stable_exponential_phi(z)

    def _prepare_sequence(
        self,
        inputs: Array,
        times: Array,
        mask: Array | None,
        reset: Array | None,
        initial_state: Array | None,
        /,
    ) -> tuple[Array, Array, Array, Array, Array]:
        values = jnp.asarray(inputs)
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("SelectiveStateSpaceMixer inputs must be real-valued.")
        time_values = jnp.asarray(times)
        if jnp.issubdtype(time_values.dtype, jnp.complexfloating):
            raise TypeError("SelectiveStateSpaceMixer times must be real-valued.")
        if time_values.ndim < 1 or int(time_values.shape[-1]) <= 0:
            raise ValueError("times must contain a non-empty trailing sequence axis.")
        length = int(time_values.shape[-1])
        in_count = _get_size(self.in_size)
        if self.in_size == "scalar":
            if values.ndim < 1 or int(values.shape[-1]) != length:
                raise ValueError("Scalar inputs must have shape case_shape + (length,).")
            values = values[..., None]
        elif (
            values.ndim < 2
            or int(values.shape[-2]) != length
            or int(values.shape[-1]) != in_count
        ):
            raise ValueError("inputs must have shape case_shape + (length, in_channels).")
        case_shape = tuple(int(size) for size in values.shape[:-2])
        compute_dtype = jnp.result_type(
            values.dtype, time_values.dtype, self.raw_decay.dtype
        )
        if not jnp.issubdtype(compute_dtype, jnp.floating):
            compute_dtype = self.raw_decay.dtype
        values = values.astype(compute_dtype)
        time_values, valid, resets, _ = normalize_physical_schedule(
            time_values,
            case_shape=case_shape,
            sequence_length=length,
            mask=mask,
            reset=reset,
            dtype=compute_dtype,
            require_prefix=False,
        )
        values = jnp.where(valid[..., None], values, jnp.zeros_like(values))
        if initial_state is None:
            state = jnp.zeros(case_shape + (self.state_size,), dtype=compute_dtype)
        else:
            state = jnp.asarray(initial_state, dtype=compute_dtype)
            if state.shape == (self.state_size,):
                state = jnp.broadcast_to(state, case_shape + (self.state_size,))
            elif state.shape != case_shape + (self.state_size,):
                raise ValueError(
                    "initial_state must be shared or have shape case_shape + (state_size,)."
                )
        return values, time_values, valid, resets, state

    def _selective_scale(self, context: Array, /) -> Array:
        return (
            jax.nn.softplus(
                jnp.einsum("...ti,mi->...tm", context, self.delta_weight)
                + self.delta_bias
            )
            + self.min_step_scale
        )

    def _drive(self, values: Array, context: Array, /) -> Array:
        projection = jnp.einsum("...ti,mi->...tm", values, self.input_matrix)
        gate = jax.nn.sigmoid(
            jnp.einsum("...ti,mi->...tm", context, self.input_gate_weight)
            + self.input_gate_bias
        )
        return gate * projection

    def _affine_steps(
        self,
        values: Array,
        times: Array,
        valid: Array,
        reset: Array,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        delta = times[..., 1:] - times[..., :-1]
        continuation = valid[..., :-1] & valid[..., 1:] & ~reset[..., 1:]
        safe_delta = jnp.where(continuation, delta, 0.0)
        context = (
            values[..., :-1, :]
            if self.input_integration == "zoh"
            else 0.5 * (values[..., :-1, :] + values[..., 1:, :])
        )
        scale = self._selective_scale(context)
        effective_step = safe_delta[..., None] * scale
        z = -self.decay_rates() * effective_step
        transition = jnp.exp(z)
        phi_one, phi_two = self._phi_functions(z)
        left_drive = self._drive(values[..., :-1, :], context)
        if self.input_integration == "zoh":
            injection = effective_step * phi_one * left_drive
        else:
            right_drive = self._drive(values[..., 1:, :], context)
            right = effective_step * phi_two
            left = effective_step * (phi_one - phi_two)
            injection = left * left_drive + right * right_drive
        transition = jnp.where(
            continuation[..., None], transition, jnp.ones_like(transition)
        )
        injection = jnp.where(
            continuation[..., None], injection, jnp.zeros_like(injection)
        )
        return transition, injection, effective_step, continuation

    def _state_trajectory(
        self,
        values: Array,
        times: Array,
        valid: Array,
        reset: Array,
        initial_state: Array,
        /,
        *,
        execution: Literal["serial", "associative"],
    ) -> tuple[Array, Array, Array]:
        first_reset = valid[..., 0] & reset[..., 0]
        state_at_first = jnp.where(
            first_reset[..., None],
            jnp.zeros_like(initial_state),
            initial_state,
        )
        if int(values.shape[-2]) == 1:
            empty = jnp.zeros(
                values.shape[:-2] + (0, self.state_size), dtype=values.dtype
            )
            continuation = jnp.zeros(values.shape[:-2] + (0,), dtype=bool)
            return state_at_first[..., None, :], empty, continuation
        transition, injection, effective_step, continuation = self._affine_steps(
            values, times, valid, reset
        )
        recurrence = AffineRecurrence(
            jnp.zeros((self.state_size,), dtype=initial_state.dtype)
        )
        step_valid = valid[..., 1:]
        result = run_affine_recurrence(
            recurrence,
            RecurrentBatch((transition, injection), step_valid, reset=reset[..., 1:]),
            initial_state=state_at_first,
            execution=execution,
        )
        states = jnp.concatenate((state_at_first[..., None, :], result.states), axis=-2)
        return states, effective_step, continuation

    def _readout(self, states: Array, values: Array, valid: Array, /) -> Array:
        gate = jax.nn.sigmoid(
            jnp.einsum("...ti,mi->...tm", values, self.output_gate_weight)
            + self.output_gate_bias
        )
        dynamic = jnp.einsum("om,...tm->...to", self.output_matrix, gate * states)
        skip = jnp.einsum("oi,...ti->...to", self.skip_matrix, values)
        output = jnp.where(valid[..., None], dynamic + skip, jnp.zeros_like(dynamic))
        return output[..., 0] if self.out_size == "scalar" else output

    def _diagnostics(
        self,
        times: Array,
        valid: Array,
        reset: Array,
        effective_step: Array,
        continuation: Array,
        /,
    ) -> SelectiveStateSpaceDiagnostics:
        delta = times[..., 1:] - times[..., :-1]
        count = jnp.sum(continuation)
        physical_min = jnp.min(
            jnp.where(continuation, delta, jnp.asarray(jnp.inf, dtype=delta.dtype)),
            initial=jnp.asarray(jnp.inf, dtype=delta.dtype),
        )
        physical_max = jnp.max(
            jnp.where(continuation, delta, jnp.asarray(-jnp.inf, dtype=delta.dtype)),
            initial=jnp.asarray(-jnp.inf, dtype=delta.dtype),
        )
        effective_mask = continuation[..., None]
        effective_min = jnp.min(
            jnp.where(
                effective_mask,
                effective_step,
                jnp.asarray(jnp.inf, dtype=effective_step.dtype),
            ),
            initial=jnp.asarray(jnp.inf, dtype=effective_step.dtype),
        )
        effective_max = jnp.max(
            jnp.where(
                effective_mask,
                effective_step,
                jnp.asarray(-jnp.inf, dtype=effective_step.dtype),
            ),
            initial=jnp.asarray(-jnp.inf, dtype=effective_step.dtype),
        )
        physical_min = jnp.where(count > 0, physical_min, 0.0)
        physical_max = jnp.where(count > 0, physical_max, 0.0)
        effective_min = jnp.where(count > 0, effective_min, 0.0)
        effective_max = jnp.where(count > 0, effective_max, 0.0)
        if self.training_delta_range is None:
            extrapolated_fraction = jnp.asarray(0.0, dtype=delta.dtype)
        else:
            lower, upper = self.training_delta_range
            outside = continuation & ((delta < lower) | (delta > upper))
            extrapolated_fraction = jnp.where(
                count > 0,
                jnp.sum(outside) / count,
                jnp.asarray(0.0, dtype=delta.dtype),
            )
        segment_count = jnp.sum(valid[..., :1]) + jnp.sum(valid[..., 1:] & reset[..., 1:])
        return SelectiveStateSpaceDiagnostics(
            minimum_physical_step=physical_min,
            maximum_physical_step=physical_max,
            minimum_effective_step=effective_min,
            maximum_effective_step=effective_max,
            extrapolated_fraction=extrapolated_fraction,
            interval_count=count,
            segment_count=segment_count,
        )

    def evaluate_with_diagnostics(
        self,
        inputs: Array,
        times: Array,
        /,
        *,
        mask: Array | None = None,
        reset: Array | None = None,
        initial_state: Array | None = None,
        execution: SelectiveExecution | None = None,
    ) -> tuple[Array, SelectiveStateSpaceDiagnostics]:
        """Evaluate one packed schedule and return explicit extrapolation evidence."""
        values, schedule, valid, resets, state0 = self._prepare_sequence(
            inputs, times, mask, reset, initial_state
        )
        selected = self.execution if execution is None else execution
        if selected not in ("recurrent", "associative"):
            raise ValueError("execution must be 'recurrent' or 'associative'.")
        states, effective_step, continuation = self._state_trajectory(
            values,
            schedule,
            valid,
            resets,
            state0,
            execution="serial" if selected == "recurrent" else "associative",
        )
        return self._readout(states, values, valid), self._diagnostics(
            schedule, valid, resets, effective_step, continuation
        )

    def recurrent(
        self,
        inputs: Array,
        times: Array,
        /,
        *,
        mask: Array | None = None,
        reset: Array | None = None,
        initial_state: Array | None = None,
    ) -> Array:
        return self.evaluate_with_diagnostics(
            inputs,
            times,
            mask=mask,
            reset=reset,
            initial_state=initial_state,
            execution="recurrent",
        )[0]

    def associative(
        self,
        inputs: Array,
        times: Array,
        /,
        *,
        mask: Array | None = None,
        reset: Array | None = None,
        initial_state: Array | None = None,
    ) -> Array:
        return self.evaluate_with_diagnostics(
            inputs,
            times,
            mask=mask,
            reset=reset,
            initial_state=initial_state,
            execution="associative",
        )[0]

    def _source(self, batch: OperatorBatch, /) -> FunctionSamples:
        if self.source_key is not None:
            return batch.input(self.source_key)
        if len(batch.inputs) != 1:
            raise ValueError("source_key is required for multiple OperatorBatch inputs.")
        return next(iter(batch.inputs.values()))

    def _schedule(
        self,
        samples: FunctionSamples,
        case_shape: tuple[int, ...],
        /,
        *,
        role: str,
    ) -> tuple[Array, Array]:
        if len(samples.sample_shape) != 1:
            raise ValueError(f"{role} must have exactly one temporal sample axis.")
        if samples.axes and (
            len(samples.axes) != 1 or samples.axes[0].name != self.time_axis
        ):
            raise ValueError(f"{role} tensor grid requires one {self.time_axis!r} axis.")
        coordinates = samples.coordinates_array(case_shape=case_shape)
        if int(coordinates.shape[-1]) != 1:
            raise ValueError(f"{role} point coordinates must contain time only.")
        return coordinates[..., 0], samples.mask_array(case_shape=case_shape)

    def _operator_batch_data(
        self, batch: OperatorBatch, /
    ) -> tuple[Array, Array, Array, Array]:
        source = self._source(batch)
        query = batch.require_single_query()
        if source.values is None:
            raise ValueError("SelectiveStateSpaceMixer source values cannot be None.")
        if source.sample_shape != query.sample_shape:
            raise ValueError("Source and query schedules must be coincident.")
        source_times, source_mask = self._schedule(
            source, batch.case_shape, role="Source"
        )
        query_times, query_mask = self._schedule(query, batch.case_shape, role="Query")
        source_times = eqx.error_if(
            source_times,
            jnp.any(query_mask & ~source_mask),
            "Every valid query node requires a valid coincident source node.",
        )
        source_times = eqx.error_if(
            source_times,
            jnp.any(query_mask & (source_times != query_times)),
            "Source and query physical time nodes must coincide.",
        )
        return source.values, source_times, source_mask, query_mask

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        del key
        values, times, source_mask, query_mask = self._operator_batch_data(batch)
        output = self.evaluate_with_diagnostics(
            values, times, mask=source_mask, execution=self.execution
        )[0]
        return jnp.where(
            query_mask if self.out_size == "scalar" else query_mask[..., None],
            output,
            jnp.zeros_like(output),
        )

    def __call__(
        self,
        x: Array | tuple[Array, ...] | OperatorBatch,
        times: Array | None = None,
        /,
        *,
        mask: Array | None = None,
        reset: Array | None = None,
        initial_state: Array | None = None,
        execution: SelectiveExecution | None = None,
        key: EvalKey = None,
    ) -> Array:
        del key
        if isinstance(x, OperatorBatch):
            if any(value is not None for value in (times, mask, reset, initial_state)):
                raise ValueError(
                    "OperatorBatch supplies schedules; sequence overrides are invalid."
                )
            if execution is None or execution == self.execution:
                return self.__call_operator_batch__(x)
            values, schedule, source_mask, query_mask = self._operator_batch_data(x)
            output = self.evaluate_with_diagnostics(
                values, schedule, mask=source_mask, execution=execution
            )[0]
            return jnp.where(
                query_mask if self.out_size == "scalar" else query_mask[..., None],
                output,
                jnp.zeros_like(output),
            )

        resolved_mask = mask
        resolved_reset = reset
        if isinstance(x, tuple):
            if times is not None:
                raise ValueError("times must not be supplied twice.")
            if len(x) == 2:
                values, schedule = x
            elif len(x) == 3:
                if mask is not None:
                    raise ValueError("mask must not be supplied twice.")
                values, schedule, resolved_mask = x
            elif len(x) == 4:
                if mask is not None or reset is not None:
                    raise ValueError("mask and reset must not be supplied twice.")
                values, schedule, resolved_mask, resolved_reset = x
            else:
                raise ValueError(
                    "Sequence tuples must be (inputs, times[, mask[, reset]])."
                )
        else:
            if times is None:
                raise ValueError("Sequence evaluation requires physical times.")
            values = x
            schedule = times
        return self.evaluate_with_diagnostics(
            values,
            schedule,
            mask=resolved_mask,
            reset=resolved_reset,
            initial_state=initial_state,
            execution=execution,
        )[0]


__all__ = ["SelectiveStateSpaceDiagnostics", "SelectiveStateSpaceMixer"]
