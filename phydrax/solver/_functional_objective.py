#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .._frozendict import frozendict
from .._strict import StrictModule
from .._term import AbstractSamplingTerm, AbstractScalarTerm, evaluate
from ..enforcement import EnforcementProgram
from ..integration import AdaptiveIntegration
from ..integration._adaptive_signed import AdaptiveSignedEstimator
from ..integration._execution import resolve_integration
from ..operators.differential._runtime import derivative_runtime_context
from ..sampling.collocation import ControlledCollocationPolicy
from ..terms._integral_functional import IntegralFunctional
from ..terms._integrated import prepare_term_realization
from ..terms._moment import MomentPenalty
from ..terms._residual import ResidualPenalty
from ._model_losses import function_model_loss_values


@runtime_checkable
class _SupportsDataMetrics(Protocol):
    def data_metrics(
        self,
        functions: Any,
        /,
        *,
        key: Any,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


_ObjectiveTermMode = Literal["plain", "sampled", "adaptive_population"]


def _terms_tuple(
    value: AbstractScalarTerm | Sequence[AbstractScalarTerm],
    /,
    *,
    name: str,
) -> tuple[AbstractScalarTerm, ...]:
    terms = (value,) if isinstance(value, AbstractScalarTerm) else tuple(value)
    invalid = tuple(term for term in terms if not isinstance(term, AbstractScalarTerm))
    if invalid:
        raise TypeError(
            f"All {name} must be scalar terms; got "
            f"{tuple(type(term).__name__ for term in invalid)!r}."
        )
    return terms


def _adaptive_policy(term: AbstractScalarTerm, /):
    if not isinstance(term, (ResidualPenalty, IntegralFunctional)) or not isinstance(
        term.source, AdaptiveIntegration
    ):
        raise TypeError(
            "Adaptive objective populations require ResidualPenalty or "
            "IntegralFunctional with AdaptiveIntegration."
        )
    if isinstance(term, IntegralFunctional) and not isinstance(
        term.source.policy, AdaptiveSignedEstimator
    ):
        raise TypeError("Adaptive integral terms require AdaptiveSignedEstimator.")
    return term.source.policy


def _term_mode(term: AbstractScalarTerm, /) -> _ObjectiveTermMode:
    if isinstance(term, (ResidualPenalty, IntegralFunctional)) and isinstance(
        term.source, AdaptiveIntegration
    ):
        return "adaptive_population"
    if isinstance(term, AbstractSamplingTerm):
        return "sampled"
    return "plain"


class _ObjectiveTerm(StrictModule):
    """One scalar objective term and its solver-owned persistent state."""

    term: AbstractScalarTerm
    population: Any | None
    index: int = eqx.field(static=True)
    label: str = eqx.field(static=True)
    mode: _ObjectiveTermMode = eqx.field(static=True)

    def __init__(
        self,
        term: AbstractScalarTerm,
        population: Any | None,
        /,
        *,
        index: int,
    ):
        mode = _term_mode(term)
        if (mode == "adaptive_population") != (population is not None):
            raise ValueError(
                "Adaptive objective terms must own one population and "
                "non-adaptive terms must not own one."
            )
        self.term = term
        self.population = population
        self.index = int(index)
        self.label = term.label or type(term).__name__
        self.mode = mode

    def with_population(self, population: Any, /) -> "_ObjectiveTerm":
        if self.mode != "adaptive_population":
            raise ValueError("Only adaptive objective terms own populations.")
        return eqx.tree_at(lambda slot: slot.population, self, population)


_PreparedPayloadKind = Literal["none", "batch", "realization"]


class _TermSelection(StrictModule):
    """Ordered active-term indices and their unbiased objective scale."""

    scale: Any
    indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, indices: Sequence[int], scale: Any = 1.0, /):
        self.indices = tuple(int(index) for index in indices)
        self.scale = jnp.asarray(scale, dtype=float).reshape(())


class _PreparedTerm(StrictModule):
    """One scalar term with its same-update evaluation payload and key."""

    term: AbstractScalarTerm
    payload: Any
    key: Any
    kwargs: frozendict[str, Any]
    index: int = eqx.field(static=True)
    payload_kind: _PreparedPayloadKind = eqx.field(static=True)

    def __init__(
        self,
        slot: _ObjectiveTerm,
        /,
        *,
        payload: Any,
        key: Any,
        payload_kind: _PreparedPayloadKind,
        evaluation_kwargs: Mapping[str, Any],
    ):
        if payload_kind == "none":
            if payload is not None:
                raise ValueError("Prepared term payload kind does not match its payload.")
        elif payload is None:
            raise ValueError("Prepared term payload kind requires a payload.")
        kwargs = dict(evaluation_kwargs)
        if payload_kind == "batch":
            kwargs["batch"] = payload
        elif payload_kind == "realization":
            kwargs["realization"] = prepare_term_realization(payload)
        self.term = slot.term
        self.payload = payload
        self.key = key
        self.index = slot.index
        self.payload_kind = payload_kind
        self.kwargs = frozendict(kwargs)


class _PreparedObjective(StrictModule):
    """One immutable same-update objective shared by all candidate evaluations."""

    terms: tuple[_PreparedTerm, ...]
    selection: _TermSelection
    model_loss_key: Any
    iteration: Any
    enforcement: EnforcementProgram | None

    def __init__(
        self,
        terms: Sequence[_PreparedTerm],
        selection: _TermSelection,
        model_loss_key: Any,
        iteration: Any,
        enforcement: EnforcementProgram | None,
        /,
    ):
        self.terms = tuple(terms)
        self.selection = selection
        self.model_loss_key = model_loss_key
        self.iteration = iteration
        self.enforcement = enforcement


class _ObjectiveValues(StrictModule):
    """Canonical scalar objective and its ordered component values."""

    total: Any
    term_values: Any
    model_loss_values: Any

    def __init__(
        self,
        total: Any,
        term_values: Any,
        model_loss_values: Any,
        /,
    ):
        self.total = total
        self.term_values = term_values
        self.model_loss_values = model_loss_values

    @property
    def flat_values(self) -> Any:
        return jnp.concatenate((self.term_values, self.model_loss_values), axis=0)


def _prepare_slots(
    slots: Sequence[_ObjectiveTerm],
    /,
    *,
    selection: _TermSelection,
    evaluation_key: Any,
    sampling_key: Any,
    iteration: Any,
    enforcement: EnforcementProgram | None,
    evaluation_kwargs: Mapping[str, Any] | None = None,
) -> _PreparedObjective:
    selected = tuple(slots[index] for index in selection.indices)
    evaluation_keys = jr.split(evaluation_key, len(selected))
    sampling_keys = jr.split(sampling_key, len(selected))
    prepared: list[_PreparedTerm] = []
    for slot, term_key, sample_key in zip(
        selected,
        evaluation_keys,
        sampling_keys,
        strict=True,
    ):
        if slot.mode == "adaptive_population":
            policy = _adaptive_policy(slot.term)
            if isinstance(slot.term, IntegralFunctional):
                payload = policy.loss_realization(slot.population)
            else:
                batch, local_weight = policy.loss_batch_and_weight(slot.population)
                payload = slot.term._adaptive_realization(
                    batch,
                    local_weight,
                    key=term_key,
                )
            payload_kind: _PreparedPayloadKind = "realization"
        elif isinstance(slot.term, AbstractSamplingTerm) and (
            evaluation_kwargs is None or "batch" not in evaluation_kwargs
        ):
            payload = slot.term.sample(key=sample_key)
            payload_kind = "none" if payload is None else "batch"
        elif isinstance(slot.term, (ResidualPenalty, MomentPenalty)) and (
            evaluation_kwargs is None or "realization" not in evaluation_kwargs
        ):
            payload = resolve_integration(slot.term.source, key=term_key)
            payload_kind = "realization"
        else:
            payload = None
            payload_kind = "none"
        prepared.append(
            _PreparedTerm(
                slot,
                payload=payload,
                key=term_key,
                payload_kind=payload_kind,
                evaluation_kwargs=(
                    {} if evaluation_kwargs is None else evaluation_kwargs
                ),
            )
        )
    return _PreparedObjective(
        tuple(prepared),
        selection,
        jr.fold_in(evaluation_key, len(selected)),
        iteration,
        enforcement,
    )


def evaluate_prepared_objective(
    prepared: _PreparedObjective,
    functions: Any,
    /,
    *,
    include_model_losses: bool = True,
) -> _ObjectiveValues:
    """Evaluate one prepared objective without rematerializing stochastic payloads."""
    enforced = (
        functions
        if prepared.enforcement is None
        else prepared.enforcement.apply(functions)
    )
    term_values: list[Any] = []
    total = jnp.asarray(0.0, dtype=float)
    scale = jnp.asarray(prepared.selection.scale, dtype=float).reshape(())
    with derivative_runtime_context():
        for prepared_term in prepared.terms:
            value = evaluate(
                prepared_term.term,
                enforced,
                key=prepared_term.key,
                step=prepared.iteration,
                **prepared_term.kwargs,
            ).value
            value = scale * jnp.asarray(value, dtype=float).reshape(())
            term_values.append(value)
            total = total + value
        model_loss_values: list[Any] = []
        if include_model_losses:
            for value in function_model_loss_values(
                functions,
                key=prepared.model_loss_key,
                iter_=prepared.iteration,
            ):
                value = jnp.asarray(value, dtype=float).reshape(())
                model_loss_values.append(value)
                total = total + value
    terms_array = (
        jnp.stack(term_values, axis=0) if term_values else jnp.zeros((0,), dtype=float)
    )
    model_array = (
        jnp.stack(model_loss_values, axis=0)
        if model_loss_values
        else jnp.zeros((0,), dtype=float)
    )
    return _ObjectiveValues(total, terms_array, model_array)


def evaluate_prepared_scalar_remainder(
    prepared: _PreparedObjective,
    functions: Any,
    /,
) -> Any:
    """Evaluate non-residual terms and model losses on one frozen realization."""

    enforced = (
        functions
        if prepared.enforcement is None
        else prepared.enforcement.apply(functions)
    )
    total = jnp.asarray(0.0, dtype=float)
    scale = jnp.asarray(prepared.selection.scale, dtype=float).reshape(())
    with derivative_runtime_context():
        for prepared_term in prepared.terms:
            if isinstance(prepared_term.term, ResidualPenalty):
                continue
            value = evaluate(
                prepared_term.term,
                enforced,
                key=prepared_term.key,
                step=prepared.iteration,
                **prepared_term.kwargs,
            ).value
            total = total + scale * jnp.asarray(value, dtype=float).reshape(())
        for value in function_model_loss_values(
            functions,
            key=prepared.model_loss_key,
            iter_=prepared.iteration,
        ):
            total = total + jnp.asarray(value, dtype=float).reshape(())
    return total


def prepared_data_metrics(
    prepared: _PreparedObjective,
    functions: Any,
    /,
) -> tuple[dict[str, Any], ...]:
    """Evaluate diagnostics from exactly the payloads used by the objective."""
    enforced = (
        functions
        if prepared.enforcement is None
        else prepared.enforcement.apply(functions)
    )
    metrics = []
    with derivative_runtime_context():
        for prepared_term in prepared.terms:
            if not isinstance(prepared_term.term, _SupportsDataMetrics):
                metrics.append({})
                continue
            metrics.append(
                prepared_term.term.data_metrics(
                    enforced,
                    key=prepared_term.key,
                    iter_=prepared.iteration,
                    **prepared_term.kwargs,
                )
            )
    return tuple(metrics)


class _FunctionalObjective(StrictModule):
    """Ordered scalar objective plus all solver-owned per-term runtime state."""

    training: tuple[_ObjectiveTerm, ...]
    evaluation: tuple[_ObjectiveTerm, ...]
    enforcement: EnforcementProgram | None

    def __init__(
        self,
        *,
        terms: AbstractScalarTerm | Sequence[AbstractScalarTerm],
        evaluation_terms: AbstractScalarTerm | Sequence[AbstractScalarTerm] = (),
        enforcement: EnforcementProgram | None = None,
        collocation_key: Any,
    ):
        training_terms = _terms_tuple(terms, name="terms")
        diagnostic_terms = _terms_tuple(evaluation_terms, name="evaluation_terms")
        if any(_term_mode(term) == "adaptive_population" for term in diagnostic_terms):
            raise ValueError(
                "AdaptiveIntegration sources are only supported in training terms, "
                "which own solver-managed collocation populations."
            )
        if enforcement is not None and not isinstance(enforcement, EnforcementProgram):
            raise TypeError("enforcement must be an EnforcementProgram or None.")

        keys = jr.split(collocation_key, len(training_terms))
        self.training = tuple(
            _ObjectiveTerm(
                term,
                (
                    _adaptive_policy(term).initialize(term, key=term_key)
                    if _term_mode(term) == "adaptive_population"
                    else None
                ),
                index=index,
            )
            for index, (term, term_key) in enumerate(
                zip(training_terms, keys, strict=True)
            )
        )
        self.evaluation = tuple(
            _ObjectiveTerm(term, None, index=index)
            for index, term in enumerate(diagnostic_terms)
        )
        self.enforcement = enforcement

    @property
    def terms(self) -> tuple[AbstractScalarTerm, ...]:
        return tuple(slot.term for slot in self.training)

    @property
    def evaluation_terms(self) -> tuple[AbstractScalarTerm, ...]:
        return tuple(slot.term for slot in self.evaluation)

    @property
    def populations(self) -> tuple[Any | None, ...]:
        return tuple(slot.population for slot in self.training)

    def prepare_training(
        self,
        indices: Sequence[int],
        /,
        *,
        scale: Any,
        evaluation_key: Any,
        sampling_key: Any,
        iteration: Any,
        evaluation_kwargs: Mapping[str, Any] | None = None,
    ) -> _PreparedObjective:
        selection = _TermSelection(indices, scale)
        return _prepare_slots(
            self.training,
            selection=selection,
            evaluation_key=evaluation_key,
            sampling_key=sampling_key,
            iteration=iteration,
            enforcement=self.enforcement,
            evaluation_kwargs=evaluation_kwargs,
        )

    def prepare_evaluation(
        self,
        /,
        *,
        key: Any,
        iteration: Any,
        evaluation_kwargs: Mapping[str, Any] | None = None,
    ) -> _PreparedObjective:
        selection = _TermSelection(range(len(self.evaluation)), 1.0)
        return _prepare_slots(
            self.evaluation,
            selection=selection,
            evaluation_key=key,
            sampling_key=jr.fold_in(key, 1),
            iteration=iteration,
            enforcement=self.enforcement,
            evaluation_kwargs=evaluation_kwargs,
        )

    def with_populations(
        self,
        populations: Sequence[Any | None],
        /,
    ) -> "_FunctionalObjective":
        populations = tuple(populations)
        if len(populations) != len(self.training):
            raise ValueError(
                "Objective population count must match the number of training terms."
            )
        updated = []
        for slot, population in zip(self.training, populations, strict=True):
            if slot.mode == "adaptive_population":
                if population is None:
                    raise ValueError("Adaptive objective populations cannot be None.")
                slot = slot.with_population(population)
            elif population is not None:
                raise ValueError("Non-adaptive objective terms cannot own populations.")
            updated.append(slot)
        return eqx.tree_at(lambda objective: objective.training, self, tuple(updated))

    def append_training_terms(
        self,
        terms: AbstractScalarTerm | Sequence[AbstractScalarTerm],
        /,
        *,
        key: Any,
    ) -> "_FunctionalObjective":
        appended_terms = _terms_tuple(terms, name="terms")
        keys = jr.split(key, len(appended_terms))
        start = len(self.training)
        appended = tuple(
            _ObjectiveTerm(
                term,
                (
                    _adaptive_policy(term).initialize(term, key=term_key)
                    if _term_mode(term) == "adaptive_population"
                    else None
                ),
                index=start + offset,
            )
            for offset, (term, term_key) in enumerate(
                zip(appended_terms, keys, strict=True)
            )
        )
        return eqx.tree_at(
            lambda objective: objective.training,
            self,
            self.training + appended,
        )

    def retain_training_prefix(self, count: int, /) -> "_FunctionalObjective":
        count = int(count)
        if count < 0 or count > len(self.training):
            raise ValueError("Training objective prefix is out of range.")
        return eqx.tree_at(
            lambda objective: objective.training,
            self,
            self.training[:count],
        )

    def refresh(
        self,
        functions: Any,
        /,
        *,
        key: Any,
        iter_: Any,
    ) -> "_FunctionalObjective":
        enforced = (
            functions if self.enforcement is None else self.enforcement.apply(functions)
        )
        keys = jr.split(key, len(self.training))
        updated = []
        for slot, term_key in zip(self.training, keys, strict=True):
            if slot.mode != "adaptive_population":
                updated.append(slot)
                continue
            policy = _adaptive_policy(slot.term)
            population = slot.population
            if bool(policy.should_refresh(population, iter_)):
                population = policy.refresh(
                    slot.term,
                    enforced,
                    population,
                    key=term_key,
                    iter_=iter_,
                )
                slot = slot.with_population(population)
            updated.append(slot)
        return eqx.tree_at(lambda objective: objective.training, self, tuple(updated))

    def settle(
        self,
        functions: Any,
        /,
        *,
        key: Any,
        iter_: Any,
    ) -> "_FunctionalObjective":
        enforced = (
            functions if self.enforcement is None else self.enforcement.apply(functions)
        )
        keys = jr.split(key, len(self.training))
        updated = []
        for slot, term_key in zip(self.training, keys, strict=True):
            if slot.mode != "adaptive_population":
                updated.append(slot)
                continue
            policy = _adaptive_policy(slot.term)
            if isinstance(policy, ControlledCollocationPolicy):
                population = policy.settle(
                    slot.term,
                    enforced,
                    slot.population,
                    key=term_key,
                    iter_=iter_,
                )
                slot = slot.with_population(population)
            updated.append(slot)
        return eqx.tree_at(lambda objective: objective.training, self, tuple(updated))

    def record_training_evaluations(
        self,
        /,
        *,
        multiplier: int = 1,
        term_indices: tuple[int, ...] | None = None,
    ) -> "_FunctionalObjective":
        selected = (
            None
            if term_indices is None
            else frozenset(int(index) for index in term_indices)
        )
        updated = []
        for slot in self.training:
            if slot.mode != "adaptive_population" or (
                selected is not None and slot.index not in selected
            ):
                updated.append(slot)
                continue
            policy = _adaptive_policy(slot.term)
            if isinstance(policy, (ControlledCollocationPolicy, AdaptiveSignedEstimator)):
                population = policy.record_training_evaluation(
                    slot.population,
                    multiplier=multiplier,
                )
                slot = slot.with_population(population)
            updated.append(slot)
        return eqx.tree_at(lambda objective: objective.training, self, tuple(updated))

    def collocation_data_metrics(self) -> tuple[dict[str, jax.Array], ...]:
        metrics: list[dict[str, jax.Array]] = []
        for slot in self.training:
            if slot.mode != "adaptive_population":
                metrics.append({})
                continue
            metrics.append(_adaptive_policy(slot.term).data_metrics(slot.population))
        return tuple(metrics)


__all__ = [
    "_FunctionalObjective",
    "_ObjectiveTerm",
    "_ObjectiveValues",
    "_PreparedObjective",
    "_PreparedTerm",
    "_TermSelection",
    "_adaptive_policy",
    "evaluate_prepared_scalar_remainder",
    "evaluate_prepared_objective",
    "_SupportsDataMetrics",
    "prepared_data_metrics",
]
