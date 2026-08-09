#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
from jaxtyping import Array

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from .._keys import EvalKey
from ..layers import RecurrentBatch, RecurrentResult
from ..layers._linear_recurrent_unit import (
    LinearRecurrenceExecution,
    LinearRecurrentUnit,
)


LinearRecurrentReturnMode = Literal["sequence", "final"]


class LinearRecurrentModel(StrictModule):
    """Sequence-to-sequence or sequence-to-final model backed by an LRU."""

    unit: LinearRecurrentUnit
    execution: LinearRecurrenceExecution = eqx.field(static=True)
    return_mode: LinearRecurrentReturnMode = eqx.field(static=True)

    def __init__(
        self,
        unit: LinearRecurrentUnit,
        /,
        *,
        execution: LinearRecurrenceExecution = "associative",
        return_mode: LinearRecurrentReturnMode = "sequence",
    ):
        if not isinstance(unit, LinearRecurrentUnit):
            raise TypeError("unit must be a LinearRecurrentUnit.")
        if execution not in ("serial", "associative"):
            raise ValueError("execution must be 'serial' or 'associative'.")
        if return_mode not in ("sequence", "final"):
            raise ValueError("return_mode must be 'sequence' or 'final'.")
        self.unit = unit
        self.execution = execution
        self.return_mode = return_mode

    def evaluate_with_state(
        self,
        batch: RecurrentBatch,
        /,
        *,
        initial_state: Array | None = None,
        key: EvalKey = DOC_KEY0,
    ) -> RecurrentResult:
        return self.unit.evaluate_with_state(
            batch,
            initial_state=initial_state,
            execution=self.execution,
            key=key,
        )

    def __call__(
        self,
        batch: RecurrentBatch,
        /,
        *,
        initial_state: Array | None = None,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        result = self.evaluate_with_state(batch, initial_state=initial_state, key=key)
        return result.outputs if self.return_mode == "sequence" else result.final_output


__all__ = ["LinearRecurrentModel", "LinearRecurrentReturnMode"]
