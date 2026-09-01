#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..diagnostics import Diagnostic


class MaterialSiteId(StrictModule, NonTrainableState):
    """Provider-neutral stable identity for one constitutive integration site."""

    key: str = eqx.field(static=True)
    site_id: str = eqx.field(static=True)

    def __init__(self, key: str, /):
        key_ = str(key).strip()
        if not key_:
            raise ValueError("Material site key must be non-empty.")
        self.key = key_
        self.site_id = canonical_fingerprint({"kind": "material-site", "key": key_})


class ConstitutiveResponse(StrictModule):
    """One pure local constitutive response and candidate internal state."""

    response: Array
    trial_state: Array
    consistent_tangent: Array | None
    energy: Array
    dissipation: Array
    valid: Array
    diagnostic: Diagnostic
    diagnostics: Mapping[str, Array]

    def __init__(
        self,
        response: ArrayLike,
        trial_state: ArrayLike,
        /,
        *,
        consistent_tangent: ArrayLike | None = None,
        energy: ArrayLike = 0.0,
        dissipation: ArrayLike = 0.0,
        valid: ArrayLike = True,
        diagnostic: Diagnostic | None = None,
        diagnostics: Mapping[str, ArrayLike] | None = None,
    ):
        response_ = _inexact_array(response)
        trial = _inexact_array(trial_state)
        tangent = (
            None if consistent_tangent is None else _inexact_array(consistent_tangent)
        )
        energy_ = _inexact_array(energy)
        dissipation_ = _inexact_array(dissipation)
        valid_ = jnp.asarray(valid, dtype=bool)
        diagnostic_ = (
            Diagnostic(
                "material.constitutive-response",
                "info",
                "material-integration",
                "Constitutive response evaluated.",
            )
            if diagnostic is None
            else diagnostic
        )
        if not isinstance(diagnostic_, Diagnostic):
            raise TypeError("diagnostic must be a Diagnostic or None.")
        self.response = response_
        self.trial_state = trial
        self.consistent_tangent = tangent
        self.energy = energy_
        self.dissipation = dissipation_
        self.valid = valid_
        self.diagnostic = diagnostic_
        self.diagnostics = (
            {}
            if diagnostics is None
            else {str(name): jnp.asarray(value) for name, value in diagnostics.items()}
        )


class ConstitutiveModel(StrictModule, NonTrainableState):
    """Pure integration-site material update with explicit state/response shapes."""

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
        identifier = str(model_id).strip()
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
        kinematics_ = _inexact_array(kinematics)
        committed = _inexact_array(committed_state)
        if committed.shape[-len(self.state_shape) :] != self.state_shape:
            raise ValueError("Committed constitutive state shape is invalid.")
        response = self.evaluator(
            kinematics_,
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

        state_batch = committed.shape[: -len(self.state_shape)]
        response_batch = response.response.shape[: -len(self.response_shape)]
        if response_batch != state_batch:
            raise ValueError("Constitutive response and state batch shapes must agree.")
        if kinematics_.shape[: len(state_batch)] != state_batch:
            raise ValueError("Constitutive kinematics and state batch shapes must agree.")
        local_kinematics_shape = kinematics_.shape[len(state_batch) :]
        local_tangent_shape = state_batch + self.response_shape + local_kinematics_shape
        full_tangent_shape = response.response.shape + kinematics_.shape
        tangent = response.consistent_tangent
        if tangent is None:
            tangent = jax.jacfwd(
                lambda value: (
                    self.evaluator(
                        value,
                        committed,
                        parameters,
                        jnp.asarray(time),
                        jnp.asarray(dt),
                    ).response
                )
            )(kinematics_)
        if tangent.shape not in (local_tangent_shape, full_tangent_shape):
            raise ValueError("Consistent constitutive tangent shape is invalid.")

        energy = _broadcast_site_scalar(response.energy, state_batch, "energy")
        dissipation = _broadcast_site_scalar(
            response.dissipation, state_batch, "dissipation"
        )
        valid = _broadcast_site_scalar(response.valid, state_batch, "valid").astype(bool)
        finite = (
            jnp.all(jnp.isfinite(response.response))
            & jnp.all(jnp.isfinite(response.trial_state))
            & jnp.all(jnp.isfinite(tangent))
            & jnp.all(jnp.isfinite(energy))
            & jnp.all(jnp.isfinite(dissipation))
        )
        tolerance = 64.0 * jnp.finfo(dissipation.dtype).eps
        valid = valid & finite & (dissipation >= -tolerance)
        return ConstitutiveResponse(
            response.response,
            response.trial_state,
            consistent_tangent=tangent,
            energy=energy,
            dissipation=jnp.maximum(dissipation, 0.0),
            valid=valid,
            diagnostic=response.diagnostic,
            diagnostics=response.diagnostics,
        )


class MaterialState(StrictModule, NonTrainableState):
    """Committed and trial state for one provider-neutral material site."""

    committed: Array
    trial: Array
    site_id: MaterialSiteId
    model_id: str = eqx.field(static=True)
    state_version: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_id: MaterialSiteId,
        model_id: str,
        committed: ArrayLike,
        /,
        *,
        trial: ArrayLike | None = None,
        state_version: int = 0,
    ):
        if not isinstance(site_id, MaterialSiteId):
            raise TypeError("site_id must be a MaterialSiteId.")
        model = str(model_id).strip()
        committed_ = _inexact_array(committed)
        trial_ = committed_ if trial is None else _inexact_array(trial)
        version = int(state_version)
        if not model or committed_.shape != trial_.shape or version < 0:
            raise ValueError("Material state identity, shape, or version is invalid.")
        if committed_.dtype != trial_.dtype:
            raise TypeError("Committed and trial material state dtypes must agree.")
        self.committed = committed_
        self.trial = trial_
        self.site_id = site_id
        self.model_id = model
        self.state_version = version
        self.layout_id = canonical_fingerprint(
            {
                "kind": "material-state-layout",
                "site_id": site_id.site_id,
                "model_id": model,
                "shape": list(committed_.shape),
                "dtype": str(committed_.dtype),
            }
        )
        self.state_id = canonical_fingerprint(
            {
                "kind": "material-state-revision",
                "layout_id": self.layout_id,
                "state_version": version,
            }
        )

    def with_trial(self, trial: ArrayLike, /) -> MaterialState:
        return MaterialState(
            self.site_id,
            self.model_id,
            self.committed,
            trial=trial,
            state_version=self.state_version,
        )

    def commit(self, /) -> MaterialState:
        return MaterialState(
            self.site_id,
            self.model_id,
            self.trial,
            state_version=self.state_version + 1,
        )

    def rollback(self, /) -> MaterialState:
        return MaterialState(
            self.site_id,
            self.model_id,
            self.committed,
            state_version=self.state_version,
        )


class MaterialTransaction(StrictModule, NonTrainableState):
    """Atomic committed/trial state across a deterministic heterogeneous site table."""

    states: tuple[MaterialState, ...]
    layout_id: str = eqx.field(static=True)
    transaction_id: str = eqx.field(static=True)

    def __init__(self, states: Sequence[MaterialState], /):
        states_ = tuple(states)
        if not states_ or not all(isinstance(state, MaterialState) for state in states_):
            raise TypeError("states must contain one or more MaterialState values.")
        ordered = tuple(sorted(states_, key=lambda state: state.site_id.key))
        identifiers = tuple(state.site_id.key for state in ordered)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Material transaction site IDs must be unique.")
        self.states = ordered
        self.layout_id = canonical_fingerprint(
            {
                "kind": "material-transaction-layout",
                "states": [state.layout_id for state in ordered],
            }
        )
        self.transaction_id = canonical_fingerprint(
            {
                "kind": "material-transaction-revision",
                "layout_id": self.layout_id,
                "states": [state.state_id for state in ordered],
            }
        )

    def state(self, site_id: MaterialSiteId | str, /) -> MaterialState:
        key = _site_key(site_id)
        for state in self.states:
            if state.site_id.key == key:
                return state
        raise KeyError(f"Unknown material site {key!r}.")

    def with_trials(
        self,
        trials: Mapping[MaterialSiteId | str, ArrayLike],
        /,
    ) -> MaterialTransaction:
        normalized = {_site_key(site_id): value for site_id, value in trials.items()}
        known = {state.site_id.key for state in self.states}
        unknown = set(normalized) - known
        if unknown:
            raise ValueError(f"Unknown material trial site IDs {sorted(unknown)!r}.")
        return MaterialTransaction(
            tuple(
                state.with_trial(normalized[state.site_id.key])
                if state.site_id.key in normalized
                else state
                for state in self.states
            )
        )

    def commit(self, /) -> MaterialTransaction:
        return MaterialTransaction(tuple(state.commit() for state in self.states))

    def rollback(self, /) -> MaterialTransaction:
        return MaterialTransaction(tuple(state.rollback() for state in self.states))

    def checkpoint_payload(
        self, /, *, plan_id: str | None = None
    ) -> MaterialCheckpointPayload:
        return MaterialCheckpointPayload(self, plan_id=plan_id)


class MaterialCheckpointPayload(StrictModule, NonTrainableState):
    """Content-bound committed material payload for portable checkpoints."""

    state: MaterialTransaction
    plan_id: str | None = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)

    def __init__(
        self,
        state: MaterialTransaction,
        /,
        *,
        plan_id: str | None = None,
    ):
        if not isinstance(state, MaterialTransaction):
            raise TypeError("state must be a MaterialTransaction.")
        plan = None if plan_id is None else str(plan_id).strip()
        if plan_id is not None and not plan:
            raise ValueError("plan_id must be non-empty or None.")
        accepted = state.rollback()
        self.state = accepted
        self.plan_id = plan
        self.layout_id = accepted.layout_id
        self.payload_id = _material_payload_id(accepted, plan)

    def restore(self, /) -> MaterialTransaction:
        if _material_payload_id(self.state, self.plan_id) != self.payload_id:
            raise ValueError("Material checkpoint payload content identity mismatch.")
        return self.state.rollback()


class MaterialIntegrationPlan(StrictModule, NonTrainableState):
    """Deterministic provider-neutral dispatch table for heterogeneous materials."""

    site_ids: tuple[MaterialSiteId, ...]
    models: tuple[ConstitutiveModel, ...]
    layout_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sites: Sequence[tuple[MaterialSiteId, ConstitutiveModel]],
        /,
        *,
        plan_id: str | None = None,
    ):
        entries = tuple(sites)
        if not entries:
            raise ValueError("Material integration plan requires one or more sites.")
        if not all(
            isinstance(site, MaterialSiteId) and isinstance(model, ConstitutiveModel)
            for site, model in entries
        ):
            raise TypeError(
                "Material sites must pair MaterialSiteId and ConstitutiveModel values."
            )
        ordered = tuple(sorted(entries, key=lambda entry: entry[0].key))
        site_ids = tuple(site for site, _ in ordered)
        if len({site.key for site in site_ids}) != len(site_ids):
            raise ValueError("Material integration site IDs must be unique.")
        models = tuple(model for _, model in ordered)
        layout = canonical_fingerprint(
            {
                "kind": "material-integration-layout",
                "sites": [
                    {
                        "site_id": site.site_id,
                        "model_id": model.model_id,
                        "state_shape": list(model.state_shape),
                        "response_shape": list(model.response_shape),
                    }
                    for site, model in ordered
                ],
            }
        )
        identifier = layout if plan_id is None else str(plan_id).strip()
        if not identifier:
            raise ValueError("plan_id must be non-empty or None.")
        self.site_ids = site_ids
        self.models = models
        self.layout_id = layout
        self.plan_id = identifier

    def initialize(
        self,
        initial_values: Mapping[MaterialSiteId | str, ArrayLike]
        | Sequence[ArrayLike]
        | None = None,
        /,
        *,
        dtype: Any = float,
    ) -> MaterialTransaction:
        if initial_values is None:
            values = tuple(
                jnp.zeros(model.state_shape, dtype=dtype) for model in self.models
            )
        elif isinstance(initial_values, Mapping):
            normalized = {
                _site_key(site_id): value for site_id, value in initial_values.items()
            }
            expected = {site.key for site in self.site_ids}
            if set(normalized) != expected:
                raise ValueError(
                    "Initial material state mapping must cover every integration site."
                )
            values = tuple(normalized[site.key] for site in self.site_ids)
        else:
            values = tuple(initial_values)
            if len(values) != len(self.site_ids):
                raise ValueError(
                    "Initial material state sequence must match the integration sites."
                )
        states = tuple(
            MaterialState(site, model.model_id, value)
            for site, model, value in zip(self.site_ids, self.models, values, strict=True)
        )
        transaction = MaterialTransaction(states)
        self.validate(transaction)
        return transaction

    def validate(self, state: MaterialTransaction, /) -> None:
        if not isinstance(state, MaterialTransaction):
            raise TypeError("state must be a MaterialTransaction.")
        if len(state.states) != len(self.site_ids):
            raise ValueError("Material state does not match integration site count.")
        for material_state, site, model in zip(
            state.states, self.site_ids, self.models, strict=True
        ):
            if (
                material_state.site_id.site_id != site.site_id
                or material_state.model_id != model.model_id
                or material_state.committed.shape != model.state_shape
                or material_state.trial.shape != model.state_shape
            ):
                raise ValueError("Material state does not match integration plan layout.")

    def evaluate(
        self,
        site_id: MaterialSiteId | str,
        kinematics: ArrayLike,
        state: MaterialTransaction,
        parameters: object,
        time: ArrayLike,
        dt: ArrayLike,
        /,
    ) -> ConstitutiveResponse:
        self.validate(state)
        index = self._site_index(site_id)
        return self.models[index].evaluate(
            kinematics,
            state.states[index].committed,
            parameters,
            time,
            dt,
        )

    def evaluate_all(
        self,
        kinematics: Sequence[ArrayLike],
        state: MaterialTransaction,
        parameters: Sequence[object],
        time: ArrayLike,
        dt: ArrayLike,
        /,
    ) -> tuple[ConstitutiveResponse, ...]:
        self.validate(state)
        kinematics_ = tuple(kinematics)
        parameters_ = tuple(parameters)
        if len(kinematics_) != len(self.models) or len(parameters_) != len(self.models):
            raise ValueError(
                "Material kinematics and parameters must match integration sites."
            )
        return tuple(
            model.evaluate(
                local_kinematics,
                material_state.committed,
                local_parameters,
                time,
                dt,
            )
            for model, local_kinematics, material_state, local_parameters in zip(
                self.models,
                kinematics_,
                state.states,
                parameters_,
                strict=True,
            )
        )

    def with_responses(
        self,
        state: MaterialTransaction,
        responses: Mapping[MaterialSiteId | str, ConstitutiveResponse]
        | Sequence[ConstitutiveResponse],
        /,
    ) -> MaterialTransaction:
        self.validate(state)
        if isinstance(responses, Mapping):
            normalized = {
                _site_key(site_id): response for site_id, response in responses.items()
            }
            if set(normalized) != {site.key for site in self.site_ids}:
                raise ValueError("Material responses must cover every integration site.")
            responses_ = tuple(normalized[site.key] for site in self.site_ids)
        else:
            responses_ = tuple(responses)
            if len(responses_) != len(self.site_ids):
                raise ValueError("Material responses must match integration sites.")
        if not all(isinstance(response, ConstitutiveResponse) for response in responses_):
            raise TypeError("responses must contain ConstitutiveResponse values.")
        trials: dict[str, Array] = {}
        for site, model, response in zip(
            self.site_ids, self.models, responses_, strict=True
        ):
            if response.trial_state.shape != model.state_shape:
                raise ValueError("Material response trial state shape is invalid.")
            trials[site.key] = response.trial_state
        return state.with_trials(trials)

    def checkpoint_payload(
        self, state: MaterialTransaction, /
    ) -> MaterialCheckpointPayload:
        self.validate(state)
        return state.checkpoint_payload(plan_id=self.plan_id)

    def restore_payload(
        self, payload: MaterialCheckpointPayload, /
    ) -> MaterialTransaction:
        if not isinstance(payload, MaterialCheckpointPayload):
            raise TypeError("payload must be a MaterialCheckpointPayload.")
        if payload.plan_id != self.plan_id:
            raise ValueError("Material checkpoint payload is bound to another plan.")
        state = payload.restore()
        self.validate(state)
        return state

    def _site_index(self, site_id: MaterialSiteId | str, /) -> int:
        key = _site_key(site_id)
        for index, site in enumerate(self.site_ids):
            if site.key == key:
                return index
        raise KeyError(f"Unknown material site {key!r}.")


def _inexact_array(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _broadcast_site_scalar(
    value: Array, batch_shape: tuple[int, ...], name: str, /
) -> Array:
    if value.shape == ():
        return jnp.broadcast_to(value, batch_shape)
    if value.shape != batch_shape:
        raise ValueError(f"Constitutive {name} must be scalar per integration site.")
    return value


def _site_key(site_id: MaterialSiteId | str, /) -> str:
    if isinstance(site_id, MaterialSiteId):
        return site_id.key
    key = str(site_id).strip()
    if not key:
        raise ValueError("Material site key must be non-empty.")
    return key


def _material_payload_id(state: MaterialTransaction, plan_id: str | None, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "material-checkpoint-payload",
            "plan_id": plan_id,
            "layout_id": state.layout_id,
            "states": [
                {
                    "site_id": material_state.site_id.site_id,
                    "model_id": material_state.model_id,
                    "state_version": material_state.state_version,
                }
                for material_state in state.states
            ],
            "committed": array_tree_fingerprint(
                tuple(
                    np.asarray(material_state.committed)
                    for material_state in state.states
                )
            ),
        }
    )


__all__ = [
    "ConstitutiveModel",
    "ConstitutiveResponse",
    "MaterialCheckpointPayload",
    "MaterialIntegrationPlan",
    "MaterialSiteId",
    "MaterialState",
    "MaterialTransaction",
]
