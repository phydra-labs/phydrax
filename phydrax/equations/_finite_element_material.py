#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class ConstitutiveResponse(StrictModule):
    """One pure local constitutive response and candidate internal state."""

    response: Array
    trial_state: Array
    diagnostics: Mapping[str, Array]

    def __init__(
        self,
        response: ArrayLike,
        trial_state: ArrayLike,
        /,
        *,
        diagnostics: Mapping[str, ArrayLike] | None = None,
    ):
        response_ = jnp.asarray(response)
        trial = jnp.asarray(trial_state)
        if not jnp.issubdtype(response_.dtype, jnp.inexact):
            response_ = response_.astype(float)
        if not jnp.issubdtype(trial.dtype, jnp.inexact):
            trial = trial.astype(float)
        self.response = response_
        self.trial_state = trial
        self.diagnostics = (
            {}
            if diagnostics is None
            else {str(name): jnp.asarray(value) for name, value in diagnostics.items()}
        )


class ConstitutiveModel(StrictModule, NonTrainableState):
    """Pure quadrature-local material update with explicit state shape."""

    evaluator: Callable
    state_shape: tuple[int, ...] = eqx.field(static=True)
    response_shape: tuple[int, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluator: Callable,
        /,
        *,
        state_shape: tuple[int, ...],
        response_shape: tuple[int, ...],
        model_id: str,
    ):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        state = tuple(int(size) for size in state_shape)
        response = tuple(int(size) for size in response_shape)
        if any(size <= 0 for size in state + response):
            raise ValueError("Constitutive state/response dimensions must be positive.")
        identifier = str(model_id)
        if not identifier:
            raise ValueError("model_id must be non-empty.")
        self.evaluator = evaluator
        self.state_shape = state
        self.response_shape = response
        self.model_id = identifier

    def evaluate(
        self,
        kinematics: ArrayLike,
        committed_state: ArrayLike,
        parameters: object,
        time: ArrayLike,
        dt: ArrayLike,
        /,
    ) -> ConstitutiveResponse:
        committed = jnp.asarray(committed_state)
        if committed.shape[-len(self.state_shape) :] != self.state_shape:
            raise ValueError("Committed constitutive state shape is invalid.")
        response = self.evaluator(
            jnp.asarray(kinematics),
            committed,
            parameters,
            jnp.asarray(time),
            jnp.asarray(dt),
        )
        if not isinstance(response, ConstitutiveResponse):
            raise TypeError("Constitutive evaluator must return ConstitutiveResponse.")
        if response.response.shape[-len(self.response_shape) :] != self.response_shape:
            raise ValueError("Constitutive response shape is invalid.")
        if response.trial_state.shape != committed.shape:
            raise ValueError("Trial constitutive state must preserve committed shape.")
        return response


class FiniteElementMaterialState(StrictModule, NonTrainableState):
    """Committed and candidate quadrature state for one material region."""

    committed: Array
    trial: Array
    material_id: str = eqx.field(static=True)
    state_version: int = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        material_id: str,
        committed: ArrayLike,
        /,
        *,
        trial: ArrayLike | None = None,
        state_version: int = 0,
    ):
        identifier = str(material_id)
        committed_ = jnp.asarray(committed)
        trial_ = committed_ if trial is None else jnp.asarray(trial)
        version = int(state_version)
        if not identifier or committed_.shape != trial_.shape or version < 0:
            raise ValueError("Material state identity, shape, or version is invalid.")
        self.committed = committed_
        self.trial = trial_
        self.material_id = identifier
        self.state_version = version
        self.state_id = canonical_fingerprint(
            {
                "kind": "finite-element-material-state",
                "material_id": identifier,
                "shape": list(committed_.shape),
                "state_version": version,
            }
        )

    def with_trial(self, trial: ArrayLike, /) -> FiniteElementMaterialState:
        return FiniteElementMaterialState(
            self.material_id,
            self.committed,
            trial=trial,
            state_version=self.state_version,
        )

    def commit(self, /) -> FiniteElementMaterialState:
        return FiniteElementMaterialState(
            self.material_id,
            self.trial,
            state_version=self.state_version + 1,
        )

    def rollback(self, /) -> FiniteElementMaterialState:
        return FiniteElementMaterialState(
            self.material_id,
            self.committed,
            state_version=self.state_version,
        )


class FiniteElementMaterialTransaction(StrictModule, NonTrainableState):
    """Atomic committed/trial state transaction across material regions."""

    states: tuple[FiniteElementMaterialState, ...]
    transaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        states: tuple[FiniteElementMaterialState, ...],
        /,
    ):
        states_ = tuple(states)
        if not states_ or not all(
            isinstance(state, FiniteElementMaterialState) for state in states_
        ):
            raise TypeError(
                "states must contain one or more FiniteElementMaterialState values."
            )
        identifiers = tuple(state.material_id for state in states_)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Material transaction IDs must be unique.")
        self.states = states_
        self.transaction_id = canonical_fingerprint(
            {
                "kind": "finite-element-material-transaction",
                "states": [state.state_id for state in states_],
            }
        )

    def with_trials(
        self,
        trials: Mapping[str, ArrayLike],
        /,
    ) -> FiniteElementMaterialTransaction:
        unknown = set(trials) - {state.material_id for state in self.states}
        if unknown:
            raise ValueError(f"Unknown material trial IDs {sorted(unknown)!r}.")
        return FiniteElementMaterialTransaction(
            tuple(
                state.with_trial(trials.get(state.material_id, state.trial))
                for state in self.states
            )
        )

    def commit(self, /) -> FiniteElementMaterialTransaction:
        return FiniteElementMaterialTransaction(
            tuple(state.commit() for state in self.states)
        )

    def rollback(self, /) -> FiniteElementMaterialTransaction:
        return FiniteElementMaterialTransaction(
            tuple(state.rollback() for state in self.states)
        )


__all__ = [
    "ConstitutiveModel",
    "ConstitutiveResponse",
    "FiniteElementMaterialState",
    "FiniteElementMaterialTransaction",
]
