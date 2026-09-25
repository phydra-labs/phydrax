#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import Any, ClassVar, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._admissibility import (
    AdmissibilityHeader,
    AdmissibilityReason,
    guard_derivative_validity,
    reason_bits_where,
)
from .._differentiation import (
    admit_regularity,
    ComponentAuthority,
    DerivativeRoute,
    DerivativeSurface,
    DifferentiationRequest,
    RegularityPolicy,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._model import (
    AbstractArrayModel,
    AbstractComponentSlot,
    admit_randomness,
    bind_component,
    ComponentBinding,
    ComponentContract,
    FrozenRealization,
    ModelPorts,
    PortMapping,
    ValuePort,
)
from .._model._ports import require_mapped_order
from .._strict import StrictModule
from .._trainable import fixed_field, NonTrainableState
from .._validation import canonical_identifier
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
    """One pure local constitutive response and candidate internal state.

    `header` is the admissibility evidence of a learned law, one entry per
    site; it is `None` for laws whose site validity is the `valid` flag alone.
    """

    response: Array
    trial_state: Array
    consistent_tangent: Array | None
    energy: Array
    dissipation: Array
    valid: Array
    header: AdmissibilityHeader | None
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
        header: AdmissibilityHeader | None = None,
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
        valid_ = jnp.asarray(valid, dtype=jnp.bool_)
        if header is not None and not isinstance(header, AdmissibilityHeader):
            raise TypeError("header must be an AdmissibilityHeader or None.")
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
        self.header = header
        self.diagnostic = diagnostic_
        self.diagnostics = (
            {}
            if diagnostics is None
            else {str(name): jnp.asarray(value) for name, value in diagnostics.items()}
        )


class AbstractConstitutiveModel(AbstractComponentSlot):
    """Integration-site constitutive law slot with `MODEL` authority.

    An implementation declares `state_shape`, `response_shape`, and `model_id`
    and supplies the raw local law `local_response`. `evaluate` owns the
    response contract for every implementation: kinematics, committed state,
    and response share one site batch shape and the trial state keeps the
    committed shape; when the law supplies no consistent tangent it is the
    forward-mode derivative of `local_response` in the kinematics; a tangent is
    site-local (`batch + response_shape + local kinematics shape`) or full
    (`response.shape + kinematics.shape`). A site is valid only where the law
    reports it valid, its response, trial state, energy, dissipation, and
    tangent are finite (a full tangent couples every site), and its dissipation
    is nonnegative. A state axis of extent zero declares a law without internal
    variables.
    """

    component_authority: ClassVar[ComponentAuthority] = ComponentAuthority.MODEL
    slot_semantic_id: ClassVar[str] = "phydrax.equations.constitutive-model"

    state_shape: eqx.AbstractVar[tuple[int, ...]]
    response_shape: eqx.AbstractVar[tuple[int, ...]]
    model_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def local_response(
        self,
        kinematics: Array,
        committed_state: Array,
        parameters: object,
        time: Array,
        dt: Array,
        /,
    ) -> ConstitutiveResponse:
        """Evaluate the raw local law; `evaluate` validates the result."""
        raise NotImplementedError

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
        time_ = jnp.asarray(time)
        dt_ = jnp.asarray(dt)
        if _trailing(committed.shape, self.state_shape) != self.state_shape:
            raise ValueError("Committed constitutive state shape is invalid.")
        response = self.local_response(kinematics_, committed, parameters, time_, dt_)
        if not isinstance(response, ConstitutiveResponse):
            raise TypeError("Constitutive law must return a ConstitutiveResponse.")
        if _trailing(response.response.shape, self.response_shape) != (
            self.response_shape
        ):
            raise ValueError("Constitutive response shape is invalid.")
        if response.trial_state.shape != committed.shape:
            raise ValueError("Trial constitutive state must preserve committed shape.")

        state_batch = committed.shape[: committed.ndim - len(self.state_shape)]
        response_batch = response.response.shape[
            : response.response.ndim - len(self.response_shape)
        ]
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
                    self.local_response(value, committed, parameters, time_, dt_).response
                )
            )(kinematics_)
        if tangent.shape not in (local_tangent_shape, full_tangent_shape):
            raise ValueError("Consistent constitutive tangent shape is invalid.")

        batch_ndim = len(state_batch)
        energy = _broadcast_site_scalar(response.energy, state_batch, "energy")
        dissipation = _broadcast_site_scalar(
            response.dissipation, state_batch, "dissipation"
        )
        valid = _broadcast_site_scalar(response.valid, state_batch, "valid").astype(
            "bool"
        )
        finite = (
            _site_finite(response.response, batch_ndim)
            & _site_finite(response.trial_state, batch_ndim)
            & _site_finite(
                tangent, batch_ndim if tangent.shape == local_tangent_shape else 0
            )
            & jnp.isfinite(energy)
            & jnp.isfinite(dissipation)
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
            header=response.header,
            diagnostic=response.diagnostic,
            diagnostics=response.diagnostics,
        )


@final
class ConstitutiveModel(AbstractConstitutiveModel, NonTrainableState):
    """Fixed analytic integration-site law given by one pure `evaluator`.

    `evaluator(kinematics, committed_state, parameters, time, dt)` returns a
    `ConstitutiveResponse`. Material coefficients arrive through `parameters`;
    the law holds no numeric state of its own and is FIXED wherever it is held.
    """

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
        state = tuple(state_shape)
        response = tuple(response_shape)
        if any(size <= 0 for size in state + response):
            raise ValueError("Constitutive state/response dimensions must be positive.")
        identifier = str(model_id).strip()
        if not identifier:
            raise ValueError("model_id must be non-empty.")
        self.evaluator = evaluator
        self.state_shape = state
        self.response_shape = response
        self.model_id = identifier

    def local_response(
        self,
        kinematics: Array,
        committed_state: Array,
        parameters: object,
        time: Array,
        dt: Array,
        /,
    ) -> ConstitutiveResponse:
        return self.evaluator(kinematics, committed_state, parameters, time, dt)


# The learned-law admission request: first derivatives in the model's value
# arguments and parameters, under the MODEL authority the constitutive slots confer.
_LEARNED_LAW_REQUEST = DifferentiationRequest(
    (DerivativeSurface.INPUT, DerivativeSurface.MODEL_PARAMETER),
    authority=ComponentAuthority.MODEL,
)
_LEARNED_VALUE_REQUEST = DifferentiationRequest(
    (DerivativeSurface.INPUT,), authority=ComponentAuthority.MODEL
)


def _bind_learned_law(
    model: AbstractArrayModel,
    slot: type[AbstractComponentSlot],
    owner_ports: ModelPorts | None,
    port_mapping: PortMapping | None,
    site: str,
    /,
) -> tuple[ComponentBinding, ComponentContract]:
    """Bind a learned model to a `MODEL` constitutive slot for implicit mechanics.

    The bound model must admit first `INPUT` and `MODEL_PARAMETER` derivatives
    under `MODEL` authority, declare classical `C^1` value regularity (or a
    branch-margin certificate) as implicit derivatives require, have admitted
    deterministic randomness, and declare a `ComponentPrecisionContract`.
    Returns the binding and its bound contract; raises `ValueError` naming
    `site` otherwise.
    """
    if not isinstance(model, AbstractArrayModel):
        raise TypeError(f"{site} model must be an AbstractArrayModel.")
    binding = bind_component(
        model, slot, port_mapping=port_mapping, owner_ports=owner_ports
    )
    contract = binding.contract(request=_LEARNED_LAW_REQUEST)
    model_contract = contract.model_contract
    implicit = admit_regularity(
        model_contract.regularity,
        _LEARNED_VALUE_REQUEST,
        route=DerivativeRoute.IMPLICIT,
        policy=RegularityPolicy(),
    )
    reasons = sorted({*contract.derivative_admission.reasons, *implicit.reasons})
    if reasons:
        raise ValueError(
            f"{site} needs a learned law with first input and parameter derivatives "
            "and classical C1 value regularity near the root (or a branch-margin "
            f"certificate) for implicit mechanics; the model is rejected: "
            f"{', '.join(reasons)}."
        )
    admitted, reason = admit_randomness(
        model_contract.randomness,
        implicit=True,
        authoritative=True,
        realization_bound=isinstance(model, FrozenRealization),
        inference_state_bound=False,
    )
    if not admitted:
        raise ValueError(f"{site} needs a deterministic learned law: {reason}.")
    if model_contract.precision is None:
        raise ValueError(
            f"{site} needs the learned model to declare a ComponentPrecisionContract; "
            "its residual error floor cannot be derived otherwise."
        )
    return binding, contract


def _model_size(value: Any, name: str, /) -> int:
    if value == "scalar":
        return 1
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Learned constitutive models need an int or 'scalar' {name}.")
    return value


def _support_box(
    lower: ArrayLike, upper: ArrayLike, size: int, /
) -> tuple[np.ndarray, np.ndarray]:
    lower_ = np.asarray(lower, dtype=np.float64)
    upper_ = np.asarray(upper, dtype=np.float64)
    if (
        lower_.shape != (size,)
        or upper_.shape != (size,)
        or not np.all(np.isfinite(lower_))
        or not np.all(np.isfinite(upper_))
        or np.any(upper_ <= lower_)
    ):
        raise ValueError(
            f"The learned-law support needs finite lower < upper bounds for each of "
            f"its {size} input components."
        )
    return lower_, upper_


@final
class LearnedConstitutiveModel(AbstractConstitutiveModel):
    """Integration-site law evaluated by a learned model child.

    The model is bound to the constitutive slot (`binding`, `MODEL` authority)
    and stays a dynamic child with its own array roles: a `ParameterOwner` model
    trains through every owner holding this law, a `FrozenModel` stays fixed.
    The owner ports are `kinematics_port` and, for a law with internal
    variables, `state_port` as inputs, and `response_port` and `state_port`
    (the trial state) as outputs; each port declares its component dimensions.
    Per site the model maps the flattened input components, in port order, to
    the flattened output components; a model declaring ports binds them with an
    explicit `port_mapping` in that order. Without `state_port` the law has no
    internal variables: `state_shape` is `(0,)` and the committed state passes
    through. `parameters`, `time`, and `dt` are not model inputs.

    `lower` and `upper` bound the model's input components (the training
    support). Every site's `header` carries the support margin (the smallest
    distance to a bound relative to the bound width) with `OUTSIDE_SUPPORT` and
    `NONFINITE` reasons; an ineligible site is invalid and keeps its primal
    response and trial state while their derivatives are NaN. The consistent
    tangent is the exact forward-mode derivative of the learned response in the
    site kinematics, so it is NaN at ineligible sites. The law declares no free
    energy and no dissipation.

    Construction admits the model for implicit mechanics: first input and
    parameter derivatives under `MODEL` authority, classical `C^1` value
    regularity, deterministic randomness, and a declared precision contract.
    """

    binding: ComponentBinding
    lower: Array = fixed_field()
    upper: Array = fixed_field()
    kinematics_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    response_shape: tuple[int, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: AbstractArrayModel,
        kinematics_port: ValuePort,
        response_port: ValuePort,
        /,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        model_id: str,
        state_port: ValuePort | None = None,
        port_mapping: PortMapping | None = None,
    ):
        site = "LearnedConstitutiveModel"
        inputs = (
            (kinematics_port,) if state_port is None else (kinematics_port, state_port)
        )
        outputs = (response_port,) if state_port is None else (response_port, state_port)
        if any(not isinstance(port, ValuePort) for port in (*inputs, response_port)):
            raise TypeError(f"{site} ports must be ValuePort values.")
        undeclared = sorted(
            {port.semantic_id for port in (*inputs, *outputs) if port.dimensions is None}
        )
        if undeclared:
            raise ValueError(
                f"{site} ports must declare component dimensions (units); "
                f"undeclared: {undeclared}."
            )
        identifier = canonical_identifier(model_id, "model_id")
        owner_ports = ModelPorts(inputs=inputs, outputs=outputs)
        input_size = sum(prod(port.event_shape) for port in inputs)
        output_size = sum(prod(port.event_shape) for port in outputs)
        if not isinstance(model, AbstractArrayModel):
            raise TypeError(f"{site} model must be an AbstractArrayModel.")
        if (
            _model_size(model.in_size, "in_size") != input_size
            or _model_size(model.out_size, "out_size") != output_size
        ):
            raise ValueError(
                f"{site} model must map the {input_size} input components to the "
                f"{output_size} output components of its ports."
            )
        lower_, upper_ = _support_box(lower, upper, input_size)
        binding, contract = _bind_learned_law(
            model, AbstractConstitutiveModel, owner_ports, port_mapping, site
        )
        if contract.port_binding is not None:
            for direction, ports in (("input", inputs), ("output", outputs)):
                require_mapped_order(
                    contract.port_binding,
                    direction,
                    tuple(port.port_id for port in ports),
                    site=site,
                )
        self.binding = binding
        self.lower = jnp.asarray(lower_)
        self.upper = jnp.asarray(upper_)
        self.kinematics_shape = kinematics_port.event_shape
        self.state_shape = (0,) if state_port is None else state_port.event_shape
        self.response_shape = response_port.event_shape
        self.model_id = canonical_fingerprint(
            {
                "kind": "learned-constitutive-model",
                "declared_id": identifier,
                "slot": AbstractConstitutiveModel.slot_semantic_id,
                "ports": owner_ports.ports_id,
            }
        )
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "learned-constitutive-admissibility",
                "model_id": self.model_id,
                "lower": array_tree_fingerprint(lower_),
                "upper": array_tree_fingerprint(upper_),
            }
        )

    def local_response(
        self,
        kinematics: Array,
        committed_state: Array,
        parameters: object,
        time: Array,
        dt: Array,
        /,
    ) -> ConstitutiveResponse:
        del parameters, time, dt
        batch = committed_state.shape[: committed_state.ndim - len(self.state_shape)]
        if kinematics.shape != batch + self.kinematics_shape:
            raise ValueError(
                "Learned constitutive kinematics must be the site batch followed by "
                "the kinematics port shape."
            )
        sites = prod(batch)
        flat_kinematics = kinematics.reshape((sites, prod(self.kinematics_shape)))
        flat_state = committed_state.reshape((sites, prod(self.state_shape)))
        tangent, (outputs, margin, reasons) = jax.vmap(self._site)(
            flat_kinematics, flat_state
        )
        header = AdmissibilityHeader(
            margin.reshape(batch),
            reasons.reshape(batch),
            self.model_id,
            self.evidence_id,
        )
        eligible = (reasons == 0)[:, None]
        response_size = prod(self.response_shape)
        trial = outputs[:, response_size:] if self.state_shape != (0,) else flat_state
        # The guard wraps the finished site values: a guard inside the site
        # Jacobian would only guard the inner tangent, never the returned primal.
        response, trial, tangent = guard_derivative_validity(
            (
                outputs[:, :response_size],
                trial,
                jnp.where(eligible, tangent.reshape((sites, -1)), jnp.nan),
            ),
            eligible,
            dependencies=(kinematics, committed_state),
        )
        return ConstitutiveResponse(
            response.reshape(batch + self.response_shape),
            trial.reshape(committed_state.shape),
            consistent_tangent=tangent.reshape(
                batch + self.response_shape + self.kinematics_shape
            ),
            valid=header.eligible,
            header=header,
            diagnostics={"support_margin": header.margin},
        )

    def _site(
        self, kinematics: Array, state: Array, /
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        """Response tangent, flat outputs, support margin, and reasons of one site."""
        model = self.binding.model
        response_size = prod(self.response_shape)
        stateful = self.state_shape != (0,)

        def response_of(value: Array):
            features = jnp.concatenate((value, state)) if stateful else value
            raw = model(features[0] if model.in_size == "scalar" else features)
            outputs = jnp.reshape(jnp.asarray(raw), (-1,))
            scale = self.upper - self.lower
            margin = jnp.min(
                jnp.minimum(features - self.lower, self.upper - features) / scale
            )
            reasons = reason_bits_where(
                margin >= 0.0, AdmissibilityReason.OUTSIDE_SUPPORT
            ) | reason_bits_where(
                jnp.all(jnp.isfinite(outputs)) & jnp.isfinite(margin),
                AdmissibilityReason.NONFINITE,
            )
            return outputs[:response_size], (outputs, margin, reasons)

        return jax.jacfwd(response_of, has_aux=True)(kinematics)


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


class MaterialIntegrationPlan(StrictModule):
    """Deterministic provider-neutral dispatch table for heterogeneous materials.

    Each site holds one `AbstractConstitutiveModel`. The plan is neutral: every
    law keeps its own array roles, so a learned law trains through the plan and
    a fixed analytic law stays FIXED.
    """

    site_ids: tuple[MaterialSiteId, ...]
    models: tuple[AbstractConstitutiveModel, ...]
    layout_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sites: Sequence[tuple[MaterialSiteId, AbstractConstitutiveModel]],
        /,
        *,
        plan_id: str | None = None,
    ):
        entries = tuple(sites)
        if not entries:
            raise ValueError("Material integration plan requires one or more sites.")
        if not all(
            isinstance(site, MaterialSiteId)
            and isinstance(model, AbstractConstitutiveModel)
            for site, model in entries
        ):
            raise TypeError(
                "Material sites must pair MaterialSiteId and AbstractConstitutiveModel "
                "values."
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
        dtype: Any = jnp.float64,
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
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype("float64")


def _trailing(shape: tuple[int, ...], trailing: tuple[int, ...], /) -> tuple[int, ...]:
    """The last `len(trailing)` axes of `shape` (all of `shape` when it is shorter)."""
    return shape[max(len(shape) - len(trailing), 0) :]


def _site_finite(value: Array, batch_ndim: int, /) -> Array:
    """Per-site finiteness over every axis after the leading `batch_ndim` axes."""
    return jnp.all(jnp.isfinite(value), axis=tuple(range(batch_ndim, value.ndim)))


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
    "AbstractConstitutiveModel",
    "ConstitutiveModel",
    "ConstitutiveResponse",
    "LearnedConstitutiveModel",
    "MaterialCheckpointPayload",
    "MaterialIntegrationPlan",
    "MaterialSiteId",
    "MaterialState",
    "MaterialTransaction",
]
