#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
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

from phydrax._doc import DOC_KEY0
from phydrax.nn._keys import EvalKey
from phydrax.nn._utils import _get_size
from phydrax.nn.layers import (
    AffineRecurrence,
    RecurrentBatch,
    run_affine_recurrence,
)
from phydrax.nn.layers._physical_sequence import (
    normalize_physical_schedule,
    stable_exponential_phi,
)
from phydrax.nn.operator.data import FunctionSamples, OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel


InputIntegration = Literal["zoh", "linear"]
MixerExecution = Literal["recurrent", "associative"]


def _diagonal_state_space_contract_configuration(
    model: DiagonalStateSpaceMixer,
) -> tuple[tuple[str, object], ...]:
    return (
        ("state_size", model.state_size),
        ("input_integration", model.input_integration),
        ("execution", model.execution),
        ("discretization", model.discretization),
        ("approximation", model.approximation),
        ("source_key", model.source_key),
        ("method_id", model.method_id),
    )


class DiagonalStateSpaceMixer(AbstractOperatorModel):
    r"""Exact continuous-time diagonal state-space sequence mixer.

    ``state_size`` counts stored complex modes. Each stored mode represents a
    conjugate pair, so the corresponding real state-space realization has order
    ``2 * state_size``. The continuous poles are

    $$
    \lambda_m=-\left(\operatorname{softplus}(r_m)+\epsilon\right)+i\omega_m,
    $$

    which keeps every pole strictly in the open left half-plane without
    clipping or post-update repair. Input and output matrices use the same
    implicit conjugate pairing, making every returned sequence real-valued.

    Inputs are samples at physical time nodes. The state at the first node is
    the supplied initial state (zero by default), and each subsequent state is
    obtained by exact variable-step integration. ``"zoh"`` holds the input at
    the left endpoint; ``"linear"`` integrates the linear interpolant between
    both endpoints. Boolean masks describe valid prefixes of padded, ragged
    schedules. Invalid suffix entries neither advance the state nor contribute
    output.

    The default recurrent implementation is linear in sequence length. The
    associative implementation composes the identical affine transitions with
    ``jax.lax.associative_scan``. ``direct_convolution`` is an intentionally
    dense reference implementation and rejects sequences longer than
    ``max_direct_length``.
    """

    operator_architecture = "DiagonalStateSpaceMixer"
    _operator_contract_configuration = staticmethod(
        _diagonal_state_space_contract_configuration
    )

    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    state_size: int
    input_integration: InputIntegration
    execution: MixerExecution
    time_axis: str
    source_key: str | None
    min_decay: float
    max_direct_length: int
    discretization: Literal["exact"]
    approximation: Literal["none"]

    raw_decay: Array
    frequencies: Array
    input_matrix_real: Array
    input_matrix_imag: Array
    output_matrix_real: Array
    output_matrix_imag: Array
    skip_matrix: Array

    def __init__(
        self,
        *,
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] | None = None,
        state_size: int = 32,
        input_integration: InputIntegration = "zoh",
        execution: MixerExecution = "recurrent",
        time_axis: str = "time",
        source_key: str | None = None,
        initial_decay: float = 0.1,
        min_decay: float = 1e-4,
        frequency_scale: float = 1.0,
        max_direct_length: int = 2048,
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
        self.max_direct_length = int(max_direct_length)
        self.discretization = "exact"
        self.approximation = "none"

        real_dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(real_dtype, jnp.floating):
            raise TypeError("dtype must be a real floating dtype.")
        min_decay_value = float(jnp.asarray(min_decay, dtype=real_dtype))
        initial_decay_value = float(jnp.asarray(initial_decay, dtype=real_dtype))
        frequency_scale_value = float(jnp.asarray(frequency_scale, dtype=real_dtype))
        if not all(
            math.isfinite(value)
            for value in (
                min_decay_value,
                initial_decay_value,
                frequency_scale_value,
            )
        ):
            raise ValueError(
                "initial_decay, min_decay, and frequency_scale must be finite."
            )
        self.min_decay = min_decay_value
        if self.min_decay <= 0.0 or initial_decay_value <= self.min_decay:
            raise ValueError("initial_decay must be greater than positive min_decay.")
        if frequency_scale_value < 0.0:
            raise ValueError("frequency_scale must be non-negative.")
        in_count = _get_size(self.in_size)
        out_count = _get_size(self.out_size)
        if in_count <= 0 or out_count <= 0:
            raise ValueError("Input and output channel counts must be positive.")
        if self.state_size <= 0:
            raise ValueError("state_size must be positive.")
        if input_integration not in ("zoh", "linear"):
            raise ValueError("input_integration must be 'zoh' or 'linear'.")
        if execution not in ("recurrent", "associative"):
            raise ValueError("execution must be 'recurrent' or 'associative'.")
        if not self.time_axis:
            raise ValueError("time_axis must be non-empty.")
        if self.max_direct_length <= 0:
            raise ValueError("max_direct_length must be positive.")

        decay_key, frequency_key, input_key, output_key = jr.split(key, 4)
        unconstrained_decay = jnp.asarray(
            initial_decay_value - self.min_decay, dtype=real_dtype
        )
        raw_decay_location = unconstrained_decay + jnp.log(
            -jnp.expm1(-unconstrained_decay)
        )
        decay_perturbation = 0.05 * jr.normal(
            decay_key, (self.state_size,), dtype=real_dtype
        )
        self.raw_decay = raw_decay_location + decay_perturbation
        self.frequencies = frequency_scale_value * jr.uniform(
            frequency_key,
            (self.state_size,),
            minval=-jnp.pi,
            maxval=jnp.pi,
            dtype=real_dtype,
        )

        input_scale = 1.0 / jnp.sqrt(jnp.asarray(in_count, dtype=real_dtype))
        input_parts = input_scale * jr.normal(
            input_key, (2, self.state_size, in_count), dtype=real_dtype
        )
        self.input_matrix_real = input_parts[0]
        self.input_matrix_imag = input_parts[1]

        output_scale = 1.0 / jnp.sqrt(jnp.asarray(2 * self.state_size, dtype=real_dtype))
        output_parts = output_scale * jr.normal(
            output_key, (2, out_count, self.state_size), dtype=real_dtype
        )
        self.output_matrix_real = output_parts[0]
        self.output_matrix_imag = output_parts[1]
        self.skip_matrix = jnp.zeros((out_count, in_count), dtype=real_dtype)

    @property
    def method_id(self) -> str:
        """Stable identifier for the configured exact execution policy."""
        return (
            f"diagonal-state-space-mixer/exact/{self.input_integration}/{self.execution}"
        )

    def decay_rates(self, /) -> Array:
        """Return the strictly positive decay rate of each stored mode."""
        return jax.nn.softplus(self.raw_decay) + self.min_decay

    def continuous_poles(self, /) -> Array:
        """Return all poles of the equivalent real conjugate-pair realization."""
        half = self._half_poles()
        return jnp.concatenate((half, jnp.conj(half)), axis=0)

    def _half_poles(self, /) -> Array:
        return -self.decay_rates() + 1j * self.frequencies

    def _input_matrix(self, dtype: jnp.dtype, /) -> Array:
        return self.input_matrix_real.astype(dtype) + 1j * self.input_matrix_imag.astype(
            dtype
        )

    def _output_matrix(self, dtype: jnp.dtype, /) -> Array:
        return self.output_matrix_real.astype(
            dtype
        ) + 1j * self.output_matrix_imag.astype(dtype)

    @staticmethod
    def _phi_functions(z: Array, /) -> tuple[Array, Array]:
        return stable_exponential_phi(z)

    def discretize(self, delta_time: Array | float, /) -> tuple[Array, Array, Array]:
        """Return exact diagonal transition and left/right input coefficients."""
        delta = jnp.asarray(delta_time)
        if jnp.issubdtype(delta.dtype, jnp.complexfloating):
            raise TypeError("delta_time must be real-valued.")
        compute_dtype = jnp.result_type(delta.dtype, self.raw_decay.dtype)
        delta = delta.astype(compute_dtype)
        delta = eqx.error_if(
            delta, jnp.any(~jnp.isfinite(delta)), "delta_time must be finite."
        )
        delta = eqx.error_if(
            delta, jnp.any(delta < 0.0), "delta_time must be non-negative."
        )
        complex_dtype = jnp.result_type(compute_dtype, jnp.complex64)
        poles = self._half_poles().astype(complex_dtype)
        z = delta[..., None].astype(complex_dtype) * poles
        transition = jnp.exp(z)
        phi_one, phi_two = self._phi_functions(z)
        if self.input_integration == "zoh":
            left = delta[..., None] * phi_one
            right = jnp.zeros_like(left)
        else:
            right = delta[..., None] * phi_two
            left = delta[..., None] * (phi_one - phi_two)
        return transition, left, right

    def _prepare_sequence(
        self,
        inputs: Array,
        times: Array,
        mask: Array | None,
        initial_state: Array | None,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        values = jnp.asarray(inputs)
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("DiagonalStateSpaceMixer inputs must be real-valued.")
        time_values = jnp.asarray(times)
        if jnp.issubdtype(time_values.dtype, jnp.complexfloating):
            raise TypeError("DiagonalStateSpaceMixer times must be real-valued.")
        if time_values.ndim < 1 or int(time_values.shape[-1]) <= 0:
            raise ValueError("times must contain a non-empty trailing sequence axis.")
        sequence_length = int(time_values.shape[-1])
        in_count = _get_size(self.in_size)

        if self.in_size == "scalar":
            if values.ndim < 1 or int(values.shape[-1]) != sequence_length:
                raise ValueError(
                    "Scalar inputs must have shape case_shape + (sequence_length,)."
                )
            values = values[..., None]
        elif (
            values.ndim < 2
            or int(values.shape[-2]) != sequence_length
            or int(values.shape[-1]) != in_count
        ):
            raise ValueError(
                "inputs must have shape case_shape + (sequence_length, in_channels)."
            )
        case_shape = tuple(int(size) for size in values.shape[:-2])
        compute_dtype = jnp.result_type(
            values.dtype, time_values.dtype, self.raw_decay.dtype
        )
        if not jnp.issubdtype(compute_dtype, jnp.floating):
            compute_dtype = self.raw_decay.dtype
        values = values.astype(compute_dtype)
        time_values, valid, _, _ = normalize_physical_schedule(
            time_values,
            case_shape=case_shape,
            sequence_length=sequence_length,
            mask=mask,
            reset=None,
            dtype=compute_dtype,
            require_prefix=True,
        )
        values = jnp.where(valid[..., None], values, jnp.zeros_like(values))

        complex_dtype = jnp.result_type(compute_dtype, jnp.complex64)
        if initial_state is None:
            state = jnp.zeros(case_shape + (self.state_size,), dtype=complex_dtype)
        else:
            state = jnp.asarray(initial_state, dtype=complex_dtype)
            if state.shape == (self.state_size,):
                state = jnp.broadcast_to(state, case_shape + (self.state_size,))
            elif tuple(int(size) for size in state.shape) != case_shape + (
                self.state_size,
            ):
                raise ValueError(
                    "initial_state must be shared or have shape case_shape + (state_size,)."
                )
        return values, time_values, valid, state

    def _affine_steps(
        self, values: Array, times: Array, valid: Array, /
    ) -> tuple[Array, Array]:
        delta = times[..., 1:] - times[..., :-1]
        active = valid[..., :-1] & valid[..., 1:]
        transition, left, right = self.discretize(jnp.where(active, delta, 0.0))
        complex_dtype = transition.dtype
        input_matrix = self._input_matrix(complex_dtype)
        left_projection = oe.contract(
            "...ti,mi->...tm", values[..., :-1, :], input_matrix
        )
        injection = left * left_projection
        if self.input_integration == "linear":
            right_projection = oe.contract(
                "...ti,mi->...tm", values[..., 1:, :], input_matrix
            )
            injection = injection + right * right_projection
        transition = jnp.where(active[..., None], transition, jnp.ones_like(transition))
        injection = jnp.where(active[..., None], injection, jnp.zeros_like(injection))
        return transition, injection

    def _state_trajectory(
        self,
        values: Array,
        times: Array,
        valid: Array,
        initial_state: Array,
        /,
        *,
        execution: Literal["serial", "associative"],
    ) -> Array:
        if int(values.shape[-2]) == 1:
            return initial_state[..., None, :]
        transition, injection = self._affine_steps(values, times, valid)
        active = valid[..., :-1] & valid[..., 1:]
        recurrence = AffineRecurrence(
            jnp.zeros((self.state_size,), dtype=initial_state.dtype)
        )
        result = run_affine_recurrence(
            recurrence,
            RecurrentBatch((transition, injection), active),
            initial_state=initial_state,
            execution=execution,
        )
        return jnp.concatenate((initial_state[..., None, :], result.states), axis=-2)

    def _readout(self, states: Array, values: Array, valid: Array, /) -> Array:
        output_matrix = self._output_matrix(states.dtype)
        dynamic = 2.0 * jnp.real(oe.contract("om,...tm->...to", output_matrix, states))
        skip = oe.contract(
            "oi,...ti->...to", self.skip_matrix.astype(values.dtype), values
        )
        output = dynamic.astype(skip.dtype) + skip
        output = jnp.where(valid[..., None], output, jnp.zeros_like(output))
        if self.out_size == "scalar":
            return output[..., 0]
        return output

    def recurrent(
        self,
        inputs: Array,
        times: Array,
        /,
        *,
        mask: Array | None = None,
        initial_state: Array | None = None,
    ) -> Array:
        """Mix a sequence through the shared serial affine recurrence."""
        values, time_values, valid, state0 = self._prepare_sequence(
            inputs, times, mask, initial_state
        )
        states = self._state_trajectory(
            values, time_values, valid, state0, execution="serial"
        )
        return self._readout(states, values, valid)

    def associative(
        self,
        inputs: Array,
        times: Array,
        /,
        *,
        mask: Array | None = None,
        initial_state: Array | None = None,
    ) -> Array:
        """Mix a sequence through shared associative affine composition."""
        values, time_values, valid, state0 = self._prepare_sequence(
            inputs, times, mask, initial_state
        )
        states = self._state_trajectory(
            values, time_values, valid, state0, execution="associative"
        )
        return self._readout(states, values, valid)

    def direct_convolution(
        self,
        inputs: Array,
        times: Array,
        /,
        *,
        mask: Array | None = None,
        initial_state: Array | None = None,
    ) -> Array:
        """Mix with the dense analytic convolution reference implementation."""
        values, time_values, valid, state0 = self._prepare_sequence(
            inputs, times, mask, initial_state
        )
        sequence_length = int(values.shape[-2])
        if sequence_length > self.max_direct_length:
            raise ValueError(
                "direct_convolution sequence length exceeds max_direct_length; "
                "use recurrent or associative execution."
            )
        _, injection = self._affine_steps(values, time_values, valid)
        complex_dtype = injection.dtype
        poles = self._half_poles().astype(complex_dtype)
        interval_end = time_values[..., 1:]
        lag = time_values[..., :, None] - interval_end[..., None, :]
        target_index = jnp.arange(sequence_length)[:, None]
        interval_index = jnp.arange(max(sequence_length - 1, 0))[None, :]
        causal = target_index > interval_index
        active = (
            causal & valid[..., :, None] & (valid[..., None, :-1] & valid[..., None, 1:])
        )
        safe_lag = jnp.where(active, lag, 0.0)
        propagation = jnp.exp(safe_lag[..., None].astype(complex_dtype) * poles)
        propagated = jnp.where(
            active[..., None],
            propagation * injection[..., None, :, :],
            jnp.zeros_like(propagation),
        )
        states = jnp.sum(propagated, axis=-2)
        initial_lag = jnp.where(valid, time_values - time_values[..., :1], 0.0)
        initial_propagation = jnp.exp(
            initial_lag[..., None].astype(complex_dtype) * poles
        )
        states = states + initial_propagation * state0[..., None, :]
        return self._readout(states, values, valid)

    def _source(self, batch: OperatorBatch, /) -> FunctionSamples:
        if self.source_key is not None:
            return batch.input(self.source_key)
        if len(batch.inputs) != 1:
            raise ValueError(
                "source_key is required when OperatorBatch has multiple inputs."
            )
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
            raise ValueError("DiagonalStateSpaceMixer source values cannot be None.")
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
        if not isinstance(batch, OperatorBatch):
            raise TypeError("DiagonalStateSpaceMixer requires an OperatorBatch.")
        values, times, source_mask, query_mask = self._operator_batch_data(batch)
        output = self._execute(
            values,
            times,
            mask=source_mask,
            initial_state=None,
            execution=self.execution,
        )
        if self.out_size == "scalar":
            return jnp.where(query_mask, output, jnp.zeros_like(output))
        return jnp.where(query_mask[..., None], output, jnp.zeros_like(output))

    def _execute(
        self,
        inputs: Array,
        times: Array,
        /,
        *,
        mask: Array | None,
        initial_state: Array | None,
        execution: MixerExecution,
    ) -> Array:
        if execution == "recurrent":
            return self.recurrent(inputs, times, mask=mask, initial_state=initial_state)
        if execution == "associative":
            return self.associative(inputs, times, mask=mask, initial_state=initial_state)
        raise ValueError("execution must be 'recurrent' or 'associative'.")

    def __call__(
        self,
        x: Array | tuple[Array, Array] | tuple[Array, Array, Array] | OperatorBatch,
        times: Array | None = None,
        /,
        *,
        mask: Array | None = None,
        initial_state: Array | None = None,
        execution: MixerExecution | None = None,
        key: EvalKey = None,
    ) -> Array:
        del key
        if isinstance(x, OperatorBatch):
            if times is not None or mask is not None or initial_state is not None:
                raise ValueError(
                    "OperatorBatch supplies times and masks; sequence overrides are invalid."
                )
            if execution is None or execution == self.execution:
                return self.__call_operator_batch__(x)
            values, schedule, source_mask, query_mask = self._operator_batch_data(x)
            output = self._execute(
                values,
                schedule,
                mask=source_mask,
                initial_state=None,
                execution=execution,
            )
            if self.out_size == "scalar":
                return jnp.where(query_mask, output, jnp.zeros_like(output))
            return jnp.where(query_mask[..., None], output, jnp.zeros_like(output))

        resolved_mask = mask
        if isinstance(x, tuple):
            if times is not None:
                raise ValueError("times must not be supplied twice.")
            if len(x) == 2:
                values, schedule = x
            elif len(x) == 3:
                if mask is not None:
                    raise ValueError("mask must not be supplied twice.")
                values, schedule, resolved_mask = x
            else:
                raise ValueError(
                    "Sequence tuples must be (inputs, times) or (inputs, times, mask)."
                )
        else:
            if times is None:
                raise ValueError("Sequence evaluation requires physical times.")
            values = x
            schedule = times
        selected = self.execution if execution is None else execution
        return self._execute(
            values,
            schedule,
            mask=resolved_mask,
            initial_state=initial_state,
            execution=selected,
        )


__all__ = ["DiagonalStateSpaceMixer"]
