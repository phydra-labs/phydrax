"""Accepted rod trajectory and the evidence of every attempted step."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...solver import ProductionRunResult
from ._acceptance import FuelStepEvidence
from ._state import FuelRodState


class FuelRodResult(StrictModule):
    """Accepted states in time order, with every attempt including refused ones."""

    states: tuple[FuelRodState, ...]
    attempts: tuple[FuelStepEvidence, ...]
    runtime_result: ProductionRunResult
    attempt_ids: tuple[str, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)

    def __init__(
        self,
        states: tuple[FuelRodState, ...],
        attempts: tuple[FuelStepEvidence, ...],
        runtime_result: ProductionRunResult,
        prepared_id: str,
        /,
        *,
        bundle_id: str,
        history_id: str,
    ) -> None:
        """Retain one run of ``prepared_id`` with model bundle ``bundle_id``.

        Every attempt must carry the run's prepared, bundle, and history IDs.
        """
        if not states or any(state.prepared_id != prepared_id for state in states):
            raise ValueError("Result states must share one prepared physical identity.")
        if any(
            evidence.prepared_id != prepared_id
            or evidence.bundle_id != bundle_id
            or evidence.history_id != history_id
            for evidence in attempts
        ):
            raise ValueError(
                "Every attempt must carry the result's prepared, bundle, and history IDs."
            )
        times = tuple(float(np.asarray(state.time)) for state in states)
        if any(
            later <= earlier for earlier, later in zip(times[:-1], times[1:], strict=True)
        ):
            raise ValueError(
                "Retained physical states must have strictly increasing times."
            )
        identifiers = tuple(
            canonical_fingerprint(
                {
                    "kind": "fuel-step-attempt",
                    "run": runtime_result.run_id,
                    "prepared": prepared_id,
                    "bundle": evidence.bundle_id,
                    "history": evidence.history_id,
                    "step": np.asarray(evidence.step_index).item(),
                    "time": np.asarray(evidence.time).item(),
                    "dt": np.asarray(evidence.time_step).item(),
                }
            )
            for evidence in attempts
        )
        self.states = states
        self.attempts = attempts
        self.runtime_result = runtime_result
        self.attempt_ids = identifiers
        self.prepared_id = prepared_id
        self.run_id = runtime_result.run_id

    @property
    def times(self) -> Array:
        return jnp.stack(tuple(state.time for state in self.states))

    @property
    def successful(self) -> Array:
        return self.runtime_result.successful

    @property
    def final_state(self) -> FuelRodState:
        return self.states[-1]
