#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._stage_transition import (
    FiniteVolumeStageEpochTransition,
)


class UnstructuredSSPRK3EpochStageResult(StrictModule):
    stage_content: Array
    accepted_ledger_content: Array
    accepted: Array


StageEpochExecutor = Callable[
    [int, Any, Array, Array, Array, Any], UnstructuredSSPRK3EpochStageResult
]


class UnstructuredSSPRK3EpochResult(StrictModule):
    content: Array
    accepted_ledger_content: Array
    accepted: Array
    failed_stage: Array
    final_dynamics: Any
    runtime_id: str = eqx.field(static=True)


class PreparedUnstructuredSSPRK3Runtime(StrictModule, NonTrainableState):
    """Host-segmented three-stage runtime with two prepared epoch slots."""

    initial_dynamics: Any
    stage_executor: StageEpochExecutor
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_dynamics: Any,
        stage_executor: StageEpochExecutor,
        /,
        *,
        dynamics_id: str,
        executor_id: str,
    ):
        if not callable(stage_executor):
            raise TypeError("stage_executor must be callable.")
        dynamics = str(dynamics_id)
        executor = str(executor_id)
        if not dynamics or not executor:
            raise ValueError("Runtime dynamics/executor identities must be non-empty.")
        self.initial_dynamics = initial_dynamics
        self.stage_executor = stage_executor
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "prepared-unstructured-ssprk3-epoch-runtime",
                "dynamics": dynamics,
                "executor": executor,
                "transition_slots": 2,
            }
        )

    def step(
        self,
        step_start_content: ArrayLike,
        step_size: ArrayLike,
        args: Any = None,
        /,
        *,
        stage_transitions: tuple[
            FiniteVolumeStageEpochTransition | None,
            FiniteVolumeStageEpochTransition | None,
        ] = (None, None),
    ) -> UnstructuredSSPRK3EpochResult:
        if len(stage_transitions) != 2:
            raise ValueError("SSPRK3 requires exactly two internal transition slots.")
        content0 = jnp.asarray(step_start_content)
        dt = jnp.asarray(step_size, dtype=content0.dtype)
        if dt.shape != () or not bool(jnp.isfinite(dt) & (dt > 0)):
            raise ValueError("step_size must be a positive finite scalar.")
        ledger = jnp.zeros_like(content0)
        dynamics = self.initial_dynamics
        current = content0
        active_start = content0
        for stage in range(1, 4):
            stage_result = self.stage_executor(
                stage, dynamics, active_start, current, dt, args
            )
            if not isinstance(stage_result, UnstructuredSSPRK3EpochStageResult):
                raise TypeError("stage_executor returned the wrong stage result.")
            if (
                stage_result.stage_content.shape != current.shape
                or stage_result.accepted_ledger_content.shape != current.shape
            ):
                raise ValueError("Stage executor changed fixed epoch capacity.")
            if not bool(stage_result.accepted):
                return UnstructuredSSPRK3EpochResult(
                    content0,
                    jnp.zeros_like(content0),
                    jnp.asarray(False),
                    jnp.asarray(stage, dtype=jnp.int32),
                    self.initial_dynamics,
                    self.runtime_id,
                )
            current = stage_result.stage_content
            ledger = stage_result.accepted_ledger_content
            if stage < 3:
                transition = stage_transitions[stage - 1]
                if transition is not None:
                    transfer = transition.transfer(active_start, current, ledger)
                    if not bool(transfer.accepted):
                        return UnstructuredSSPRK3EpochResult(
                            content0,
                            jnp.zeros_like(content0),
                            jnp.asarray(False),
                            jnp.asarray(stage, dtype=jnp.int32),
                            self.initial_dynamics,
                            self.runtime_id,
                        )
                    active_start = transfer.step_start_content
                    current = transfer.stage_content
                    ledger = transfer.accepted_ledger_content
                    dynamics = transition.successor_dynamics
        return UnstructuredSSPRK3EpochResult(
            current,
            ledger,
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
            dynamics,
            self.runtime_id,
        )


__all__ = [
    "PreparedUnstructuredSSPRK3Runtime",
    "StageEpochExecutor",
    "UnstructuredSSPRK3EpochResult",
    "UnstructuredSSPRK3EpochStageResult",
]
