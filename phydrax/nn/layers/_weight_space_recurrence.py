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
from ._recurrent import (
    AffineRecurrence,
    RecurrentBatch,
    RecurrentResult,
    run_affine_recurrence,
)


WeightSpaceInputMode = Literal["value", "difference"]
WeightSpaceExecution = Literal["serial", "associative"]


class WeightSpaceState(StrictModule):
    """Selected parameter vector and previous observation for streaming continuation."""

    parameters: Array
    previous_input: Array


class WeightSpaceRecurrence(StrictModule):
    """Stable diagonal recurrence in a selected root-model parameter subspace."""

    raw_retention: Array
    raw_phase: Array | None
    input_weight: Array
    input_size: int = eqx.field(static=True)
    parameter_size: int = eqx.field(static=True)
    maximum_retention: float = eqx.field(static=True)
    input_mode: WeightSpaceInputMode = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        parameter_size: int,
        /,
        *,
        input_mode: WeightSpaceInputMode = "difference",
        maximum_retention: float = 0.999,
        input_scale: float = 1e-2,
        dtype: Any = jnp.float32,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.input_size = int(input_size)
        self.parameter_size = int(parameter_size)
        self.maximum_retention = float(maximum_retention)
        self.input_mode = input_mode
        if self.input_size <= 0 or self.parameter_size <= 0:
            raise ValueError("input_size and parameter_size must be positive.")
        if input_mode not in ("value", "difference"):
            raise ValueError("input_mode must be 'value' or 'difference'.")
        if (
            not math.isfinite(self.maximum_retention)
            or not 0.0 < self.maximum_retention < 1.0
        ):
            raise ValueError("maximum_retention must lie strictly between zero and one.")
        scale = float(input_scale)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("input_scale must be finite and non-negative.")
        resolved_dtype = jnp.dtype(dtype)
        if not (
            jnp.issubdtype(resolved_dtype, jnp.floating)
            or jnp.issubdtype(resolved_dtype, jnp.complexfloating)
        ):
            raise TypeError("dtype must be a homogeneous real or complex dtype.")
        retention_key, phase_key, input_key = jr.split(key, 3)
        real_dtype = jnp.empty((), dtype=resolved_dtype).real.dtype
        initial_retention = jr.uniform(
            retention_key,
            (self.parameter_size,),
            minval=0.5,
            maxval=self.maximum_retention,
            dtype=real_dtype,
        )
        normalized = initial_retention / self.maximum_retention
        self.raw_retention = jnp.log(normalized) - jnp.log1p(-normalized)
        if jnp.issubdtype(resolved_dtype, jnp.complexfloating):
            self.raw_phase = jr.uniform(
                phase_key,
                (self.parameter_size,),
                minval=-jnp.pi,
                maxval=jnp.pi,
                dtype=real_dtype,
            )
            normal = jr.normal(
                input_key,
                (2, self.parameter_size, self.input_size),
                dtype=real_dtype,
            )
            weight = (normal[0] + 1j * normal[1]) / jnp.sqrt(2.0)
        else:
            self.raw_phase = None
            weight = jr.normal(
                input_key,
                (self.parameter_size, self.input_size),
                dtype=real_dtype,
            )
        self.input_weight = (
            scale * weight.astype(resolved_dtype) / jnp.sqrt(float(self.input_size))
        )

    def retention(self, /) -> Array:
        magnitude = self.maximum_retention * jax.nn.sigmoid(self.raw_retention)
        return (
            magnitude
            if self.raw_phase is None
            else magnitude * jnp.exp(1j * self.raw_phase)
        )

    def initial_state(
        self,
        center: Array,
        case_shape: tuple[int, ...],
        /,
    ) -> WeightSpaceState:
        parameter_center = jnp.asarray(center)
        if parameter_center.shape != (self.parameter_size,):
            raise ValueError(f"center must have shape ({self.parameter_size},).")
        return WeightSpaceState(
            parameters=jnp.broadcast_to(
                parameter_center,
                tuple(case_shape) + (self.parameter_size,),
            ),
            previous_input=jnp.zeros(
                tuple(case_shape) + (self.input_size,),
                dtype=parameter_center.dtype,
            ),
        )

    def _drives(
        self,
        batch: RecurrentBatch,
        previous_input: Array,
        /,
    ) -> tuple[Array, Array]:
        values = jnp.asarray(batch.inputs)
        sequence_axis = len(batch.case_shape)
        scan_values = jnp.moveaxis(values, sequence_axis, 0)
        scan_valid = jnp.moveaxis(batch.valid, -1, 0)
        scan_reset = jnp.moveaxis(batch.reset, -1, 0)

        def step(previous: Array, step_inputs: tuple[Array, Array, Array]):
            inputs, valid, reset = step_inputs
            reference = jnp.where(
                (valid & reset)[..., None],
                jnp.zeros_like(previous),
                previous,
            )
            drive = inputs if self.input_mode == "value" else inputs - reference
            drive = jnp.where(valid[..., None], drive, jnp.zeros_like(drive))
            next_previous = jnp.where(valid[..., None], inputs, previous)
            return next_previous, drive

        final_previous, scan_drives = jax.lax.scan(
            step,
            previous_input,
            (scan_values, scan_valid, scan_reset),
        )
        return jnp.moveaxis(scan_drives, 0, sequence_axis), final_previous

    def evaluate_with_state(
        self,
        batch: RecurrentBatch,
        center: Array,
        /,
        *,
        initial_state: WeightSpaceState | None = None,
        execution: WeightSpaceExecution = "associative",
        key: EvalKey = None,
    ) -> RecurrentResult:
        del key
        if not isinstance(batch, RecurrentBatch):
            raise TypeError("batch must be a RecurrentBatch.")
        values = jnp.asarray(batch.inputs)
        if values.ndim < 1 or int(values.shape[-1]) != self.input_size:
            raise ValueError(f"Weight-space inputs must end in width {self.input_size}.")
        parameter_center = jnp.asarray(center)
        values_complex = jnp.issubdtype(values.dtype, jnp.complexfloating)
        center_complex = jnp.issubdtype(
            parameter_center.dtype,
            jnp.complexfloating,
        )
        weight_complex = jnp.issubdtype(
            self.input_weight.dtype,
            jnp.complexfloating,
        )
        if not (values_complex == center_complex == weight_complex):
            raise TypeError(
                "Weight-space recurrence requires homogeneous real or complex inputs, "
                "selected parameters, and recurrence weights."
            )
        compute_dtype = jnp.result_type(
            values.dtype,
            parameter_center.dtype,
            self.input_weight.dtype,
        )
        parameter_center = parameter_center.astype(compute_dtype)
        state0 = (
            self.initial_state(parameter_center, batch.case_shape)
            if initial_state is None
            else initial_state
        )
        if not isinstance(state0, WeightSpaceState):
            raise TypeError("initial_state must be a WeightSpaceState.")
        expected_parameters = batch.case_shape + (self.parameter_size,)
        expected_input = batch.case_shape + (self.input_size,)
        initial_parameters = jnp.asarray(state0.parameters, dtype=compute_dtype)
        previous_input = jnp.asarray(state0.previous_input, dtype=compute_dtype)
        if (
            initial_parameters.shape != expected_parameters
            or previous_input.shape != expected_input
        ):
            raise ValueError(
                "Weight-space initial state has incompatible case dimensions."
            )
        safe_values = jnp.where(batch.valid[..., None], values, jnp.zeros_like(values))
        drives, final_previous = self._drives(
            RecurrentBatch(
                safe_values,
                batch.valid,
                reset=batch.reset,
                time=batch.time,
                time_direction=batch.time_direction,
            ),
            previous_input,
        )
        additions = oe.contract(
            "pi,...ti->...tp",
            self.input_weight.astype(compute_dtype),
            drives,
        )
        transitions = jnp.broadcast_to(
            self.retention().astype(compute_dtype),
            additions.shape,
        )
        recurrence = AffineRecurrence(
            jnp.zeros((self.parameter_size,), dtype=compute_dtype)
        )
        initial_deviation = initial_parameters - parameter_center
        recurrence_result = run_affine_recurrence(
            recurrence,
            RecurrentBatch(
                (transitions, additions),
                batch.valid,
                reset=batch.reset,
                time=batch.time,
                time_direction=batch.time_direction,
            ),
            initial_state=initial_deviation,
            execution=execution,
        )
        parameter_states = recurrence_result.states + parameter_center
        final_parameters = recurrence_result.final_state + parameter_center
        final_state = WeightSpaceState(
            parameters=final_parameters,
            previous_input=final_previous,
        )
        return RecurrentResult(
            states=parameter_states,
            outputs=parameter_states,
            final_state=final_state,
            final_output=final_parameters,
        )


__all__ = [
    "WeightSpaceExecution",
    "WeightSpaceInputMode",
    "WeightSpaceRecurrence",
    "WeightSpaceState",
]
