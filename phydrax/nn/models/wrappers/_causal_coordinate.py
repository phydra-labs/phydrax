# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ..._keys import EvalKey
from ...layers import (
    AbstractRecurrentCell,
    RecurrentBatch,
    RecurrentResult,
    RecurrentTimeContext,
    run_recurrent,
)


class CausalCoordinatePlan(StrictModule):
    coordinates: Array
    order: Array
    inverse_order: Array
    reset: Array
    duplicate_rule: Literal["error", "zero_step"] = eqx.field(static=True)
    sequence_length: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        /,
        *,
        reset: ArrayLike | None = None,
        duplicate_rule: Literal["error", "zero_step"] = "error",
    ):
        values = np.asarray(coordinates, dtype=float)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("Causal coordinates must be one finite nonempty schedule.")
        if duplicate_rule not in ("error", "zero_step"):
            raise ValueError("Unknown duplicate coordinate rule.")
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        if duplicate_rule == "error" and np.any(np.diff(sorted_values) <= 0.0):
            raise ValueError("Causal coordinate schedule must be strictly ordered.")
        if duplicate_rule == "zero_step" and np.any(np.diff(sorted_values) < 0.0):
            raise ValueError("Causal coordinate schedule ordering failed.")
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        reset_values = (
            np.zeros(order.size, dtype=bool)
            if reset is None
            else np.asarray(reset, dtype=bool)
        )
        if reset_values.shape != values.shape:
            raise ValueError("Causal reset mask must match coordinate schedule.")
        self.coordinates = jnp.asarray(sorted_values)
        self.order = jnp.asarray(order, dtype=jnp.int32)
        self.inverse_order = jnp.asarray(inverse, dtype=jnp.int32)
        self.reset = jnp.asarray(reset_values[order], dtype=bool)
        self.duplicate_rule = duplicate_rule
        self.sequence_length = int(order.size)
        reset_identity = tuple(bool(value) for value in reset_values[order])
        self.plan_id = (
            f"causal-coordinate:{tuple(sorted_values)}:{reset_identity}:{duplicate_rule}"
        )


class CausalCoordinateResult(StrictModule):
    coordinates: Array
    outputs: Any
    ordered_outputs: Any
    final_carry: Any
    final_context: RecurrentTimeContext | None
    recurrent: RecurrentResult
    plan_id: str = eqx.field(static=True)


class CausalCoordinateNetwork(StrictModule):
    cell: AbstractRecurrentCell

    def __init__(self, cell: AbstractRecurrentCell, /):
        if not isinstance(cell, AbstractRecurrentCell):
            raise TypeError("cell must be AbstractRecurrentCell.")
        self.cell = cell

    def __call__(
        self,
        plan: CausalCoordinatePlan,
        inputs: Any,
        /,
        *,
        valid: ArrayLike | None = None,
        initial_carry: Any | None = None,
        initial_context: RecurrentTimeContext | None = None,
        key: EvalKey = None,
    ) -> CausalCoordinateResult:
        if not isinstance(plan, CausalCoordinatePlan):
            raise TypeError("plan must be CausalCoordinatePlan.")
        ordered_inputs = jnp.take(jnp.asarray(inputs), plan.order, axis=-2)
        case_shape = ordered_inputs.shape[:-2]
        validity = (
            jnp.ones(case_shape + (plan.sequence_length,), dtype=bool)
            if valid is None
            else jnp.take(jnp.asarray(valid, dtype=bool), plan.order, axis=-1)
        )
        reset = jnp.broadcast_to(plan.reset, validity.shape)
        coordinates = jnp.broadcast_to(plan.coordinates, validity.shape)
        batch = RecurrentBatch(ordered_inputs, validity, reset=reset, time=coordinates)
        result = run_recurrent(
            self.cell,
            batch,
            initial_state=initial_carry,
            initial_context=initial_context,
            key=key,
        )
        outputs = jnp.take(result.outputs, plan.inverse_order, axis=len(case_shape))
        return CausalCoordinateResult(
            jnp.take(plan.coordinates, plan.inverse_order),
            outputs,
            result.outputs,
            result.final_state,
            result.final_context,
            result,
            plan.plan_id,
        )


__all__ = ["CausalCoordinateNetwork", "CausalCoordinatePlan", "CausalCoordinateResult"]
