#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from .._keys import EvalKey, split_eval_key
from ..layers import RecurrentBatch, RecurrentResult
from ..layers._linear_recurrent_unit import _last_valid_array
from ..layers._selective_sequence import (
    SelectiveSequenceExecution,
    SelectiveStateSpaceBlock,
    SelectiveStateSpaceState,
)


SelectiveReturnMode = Literal["sequence", "final"]


class SelectiveSequenceModel(StrictModule):
    """Stacked selective state-space sequence model with streaming state."""

    blocks: tuple[SelectiveStateSpaceBlock, ...]
    execution: SelectiveSequenceExecution = eqx.field(static=True)
    return_mode: SelectiveReturnMode = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        state_size: int,
        /,
        *,
        inner_size: int | None = None,
        depth: int = 1,
        convolution_size: int = 4,
        execution: SelectiveSequenceExecution = "associative",
        return_mode: SelectiveReturnMode = "sequence",
        dtype: Any = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        resolved_depth = int(depth)
        if resolved_depth <= 0:
            raise ValueError("depth must be positive.")
        if execution not in ("serial", "associative"):
            raise ValueError("execution must be 'serial' or 'associative'.")
        if return_mode not in ("sequence", "final"):
            raise ValueError("return_mode must be 'sequence' or 'final'.")
        keys = jr.split(key, resolved_depth)
        block_kwargs = {
            "inner_size": inner_size,
            "convolution_size": convolution_size,
        }
        if dtype is not None:
            block_kwargs["dtype"] = dtype
        self.blocks = tuple(
            SelectiveStateSpaceBlock(
                input_size,
                state_size,
                **block_kwargs,
                key=block_key,
            )
            for block_key in keys
        )
        self.execution = execution
        self.return_mode = return_mode

    def evaluate_with_state(
        self,
        batch: RecurrentBatch,
        /,
        *,
        initial_state: tuple[SelectiveStateSpaceState, ...] | None = None,
        key: EvalKey = DOC_KEY0,
    ) -> RecurrentResult:
        if not isinstance(batch, RecurrentBatch):
            raise TypeError("batch must be a RecurrentBatch.")
        states0 = (None,) * len(self.blocks) if initial_state is None else initial_state
        if not isinstance(states0, tuple) or len(states0) != len(self.blocks):
            raise TypeError("initial_state must align with the selective model blocks.")
        keys = split_eval_key(key, len(self.blocks))
        values = batch.inputs
        trajectories = []
        final_states = []
        for block, state0, block_key in zip(
            self.blocks,
            states0,
            keys,
            strict=True,
        ):
            result = block.evaluate_with_state(
                RecurrentBatch(
                    values,
                    batch.valid,
                    reset=batch.reset,
                    time=batch.time,
                    time_direction=batch.time_direction,
                ),
                initial_state=state0,
                execution=self.execution,
                key=block_key,
            )
            values = result.outputs
            trajectories.append(result.states)
            final_states.append(result.final_state)
        return RecurrentResult(
            states=tuple(trajectories),
            outputs=values,
            final_state=tuple(final_states),
            final_output=_last_valid_array(values, batch.valid),
        )

    def __call__(
        self,
        batch: RecurrentBatch,
        /,
        *,
        initial_state: tuple[SelectiveStateSpaceState, ...] | None = None,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        result = self.evaluate_with_state(batch, initial_state=initial_state, key=key)
        return result.outputs if self.return_mode == "sequence" else result.final_output


__all__ = ["SelectiveReturnMode", "SelectiveSequenceModel"]
