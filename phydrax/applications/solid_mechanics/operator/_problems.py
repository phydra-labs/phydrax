#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Key

from ...._fingerprint import canonical_fingerprint
from ...._frozendict import frozendict
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ...._training_objective import _ObjectiveContribution
from ....nn.operator.data import (
    OperatorBatch,
    OperatorFieldBatch,
    OperatorPrediction,
    OperatorTargetBatch,
)
from ....nn.operator.training._losses import (
    AbstractOperatorLossTerm,
    OperatorLossContext,
)
from ....nn.operator.training._risk import (
    MechanicsCaseReduction,
    MechanicsCaseReductionResult,
)
from ._cases import (
    MechanicsCaseBuilder,
    MechanicsOperatorCase,
    OperatorTrialFieldAdapter,
)


MechanicsOperatorFormulation = Literal["conservative", "residual", "mixed"]
MechanicsCaseFunctionalKind = Literal["energy", "residual", "mixed_block", "gauge"]


class MechanicsCaseFunctional(StrictModule, NonTrainableState):
    """One fully reduced, explicitly measured scalar for one physical case.

    The evaluator receives adapted fields, the unbatched physical prediction and
    batch, and the immutable mechanics case. It must complete its own spatial
    reduction and return exactly one real scalar. The named query branch is not
    averaged here: it binds the physical quadrature and measure used by the
    evaluator, preventing parameter weights from being confused with spatial
    weights.
    """

    evaluator: Callable = eqx.field(static=True)
    name: str = eqx.field(static=True)
    kind: MechanicsCaseFunctionalKind = eqx.field(static=True)
    query_name: str = eqx.field(static=True)
    expected_measure_id: str | None = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    functional_id: str = eqx.field(static=True)
    validity: Callable | None = eqx.field(static=True)
    functional_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        evaluator: Callable,
        /,
        *,
        kind: MechanicsCaseFunctionalKind,
        query_name: str,
        functional_id: str,
        expected_measure_id: str | None = None,
        scale: float = 1.0,
        validity: Callable | None = None,
    ):
        resolved_name = str(name)
        resolved_query = str(query_name)
        identifier = str(functional_id)
        if not resolved_name or not resolved_query or not identifier:
            raise ValueError(
                "Mechanics functional names, query names, and IDs must be non-empty."
            )
        if kind not in ("energy", "residual", "mixed_block", "gauge"):
            raise ValueError("Unknown mechanics case functional kind.")
        if not callable(evaluator):
            raise TypeError("Mechanics case functional evaluators must be callable.")
        if validity is not None and not callable(validity):
            raise TypeError("Mechanics case validity checks must be callable or None.")
        expected = None if expected_measure_id is None else str(expected_measure_id)
        if expected == "":
            raise ValueError("Expected mechanics measure IDs must be non-empty.")
        resolved_scale = float(scale)
        if not math.isfinite(resolved_scale) or resolved_scale == 0.0:
            raise ValueError("Mechanics functional scales must be finite and nonzero.")
        fingerprint = canonical_fingerprint(
            {
                "kind": "mechanics-case-functional",
                "name": resolved_name,
                "functional_kind": kind,
                "query": resolved_query,
                "expected_measure": expected,
                "scale": resolved_scale,
                "functional_id": identifier,
                "has_validity": validity is not None,
            }
        )
        self.evaluator = evaluator
        self.name = resolved_name
        self.kind = kind
        self.query_name = resolved_query
        self.expected_measure_id = expected
        self.scale = resolved_scale
        self.functional_id = identifier
        self.validity = validity
        self.functional_fingerprint = fingerprint

    def evaluate(
        self,
        fields: Mapping[str, Any],
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        case: MechanicsOperatorCase,
        /,
    ) -> Array:
        query = batch.query(self.query_name)
        if not query.has_physical_quadrature or query.measure_id is None:
            raise ValueError(
                f"Mechanics functional {self.name!r} requires explicit physical "
                f"quadrature on query {self.query_name!r}."
            )
        if (
            self.expected_measure_id is not None
            and query.measure_id != self.expected_measure_id
        ):
            raise ValueError(
                f"Mechanics functional {self.name!r} received measure "
                f"{query.measure_id!r}; expected {self.expected_measure_id!r}."
            )
        physical_weights = query.weights()
        invalid_measure = (
            jnp.any(~jnp.isfinite(physical_weights))
            | jnp.any(physical_weights < 0.0)
            | (jnp.sum(physical_weights) <= 0.0)
        )
        value = jnp.asarray(self.evaluator(fields, prediction, batch, case))
        if value.shape != ():
            raise ValueError(
                f"Mechanics functional {self.name!r} must return one scalar per case."
            )
        if jnp.iscomplexobj(value):
            raise TypeError("Mechanics case functionals must return real scalars.")
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(float)
        valid = jnp.asarray(True)
        if self.validity is not None:
            valid = jnp.asarray(
                self.validity(fields, prediction, batch, case),
                dtype=bool,
            )
            if valid.shape != ():
                raise ValueError(
                    f"Mechanics validity for {self.name!r} must return one Boolean."
                )
        value = eqx.error_if(
            value,
            invalid_measure,
            f"Mechanics functional {self.name!r} has an invalid physical measure.",
        )
        value = eqx.error_if(
            value,
            ~jnp.isfinite(value),
            f"Mechanics functional {self.name!r} returned a nonfinite case scalar.",
        )
        value = eqx.error_if(
            value,
            ~valid,
            "Invalid physical cases cannot be dropped or renormalized.",
        )
        return jnp.asarray(self.scale, dtype=value.dtype) * value


class MechanicsPerCaseResult(StrictModule):
    """Named complete physical-case scalars before outer parameter risk."""

    values: Array
    term_values: frozendict[str, Array]
    valid: Array
    case_ids: tuple[str, ...] = eqx.field(static=True)
    stratum_ids: tuple[str, ...] = eqx.field(static=True)
    measure_ids: frozendict[str, tuple[str, ...]] = eqx.field(static=True)
    formulation: MechanicsOperatorFormulation = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    problem_fingerprint: str = eqx.field(static=True)


class MechanicsOperatorLossResult(StrictModule):
    """A mechanics loss with its unreduced cases and explicit outer-risk evidence."""

    value: Array
    cases: MechanicsPerCaseResult
    risk: MechanicsCaseReductionResult
    loss_fingerprint: str = eqx.field(static=True)


@dataclass(frozen=True)
class _MechanicsOperatorProblem:
    case_builder: MechanicsCaseBuilder
    adapters: tuple[OperatorTrialFieldAdapter, ...]
    terms: tuple[MechanicsCaseFunctional, ...]
    formulation: MechanicsOperatorFormulation
    problem_id: str
    cases: tuple[MechanicsOperatorCase, ...]
    physical_batch: OperatorBatch
    evaluation_batches: tuple[OperatorBatch, ...]
    problem_fingerprint: str

    @classmethod
    def create(
        cls,
        case_builder: MechanicsCaseBuilder,
        adapters: Sequence[OperatorTrialFieldAdapter],
        terms: Sequence[MechanicsCaseFunctional],
        /,
        *,
        formulation: MechanicsOperatorFormulation,
        problem_id: str,
        allowed_kinds: tuple[MechanicsCaseFunctionalKind, ...],
    ) -> "_MechanicsOperatorProblem":
        if not isinstance(case_builder, MechanicsCaseBuilder):
            raise TypeError("case_builder must be a MechanicsCaseBuilder.")
        resolved_adapters = tuple(adapters)
        if not resolved_adapters or any(
            not isinstance(adapter, OperatorTrialFieldAdapter)
            for adapter in resolved_adapters
        ):
            raise TypeError(
                "Mechanics operator problems require OperatorTrialFieldAdapter values."
            )
        names = tuple(
            name for adapter in resolved_adapters for name in adapter.field_names
        )
        if len(set(names)) != len(names):
            raise ValueError(
                "Mechanics trial field adapters must own disjoint field names."
            )
        resolved_terms = tuple(terms)
        if not resolved_terms or any(
            not isinstance(term, MechanicsCaseFunctional) for term in resolved_terms
        ):
            raise TypeError(
                "Mechanics operator problems require MechanicsCaseFunctional values."
            )
        term_names = tuple(term.name for term in resolved_terms)
        if len(set(term_names)) != len(term_names):
            raise ValueError(
                "Mechanics functional names must be unique within a problem."
            )
        if any(term.kind not in allowed_kinds for term in resolved_terms):
            raise ValueError(
                f"{formulation} mechanics problem received an incompatible functional kind."
            )
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("Mechanics operator problem IDs must be non-empty.")
        cases = case_builder.build_all()
        physical_batch = case_builder.stacked_batch(cases)
        evaluation_batches = tuple(
            physical_batch.take(index, axis="parameter") for index in range(len(cases))
        )
        fingerprint = canonical_fingerprint(
            {
                "kind": "mechanics-operator-problem",
                "formulation": formulation,
                "problem_id": identifier,
                "case_builder": case_builder.builder_id,
                "adapters": [
                    adapter.adapter_fingerprint for adapter in resolved_adapters
                ],
                "terms": [term.functional_fingerprint for term in resolved_terms],
                "cases": [case.case_fingerprint for case in cases],
            }
        )
        return cls(
            case_builder=case_builder,
            adapters=resolved_adapters,
            terms=resolved_terms,
            formulation=formulation,
            problem_id=identifier,
            cases=cases,
            physical_batch=physical_batch,
            evaluation_batches=evaluation_batches,
            problem_fingerprint=fingerprint,
        )

    @property
    def metadata(self) -> frozendict[str, str]:
        values = dict(self.case_builder.metadata)
        values["mechanics_operator_problem_fingerprint"] = self.problem_fingerprint
        values["mechanics_trial_adapter_fingerprint"] = canonical_fingerprint(
            [adapter.adapter_fingerprint for adapter in self.adapters]
        )
        values["mechanics_case_design_fingerprint"] = canonical_fingerprint(
            [case.case_fingerprint for case in self.cases]
        )
        values["mechanics_geometry_design_fingerprint"] = canonical_fingerprint(
            [case.geometry.geometry_fingerprint for case in self.cases]
        )
        return frozendict(values)

    @property
    def parameter_weights(self) -> Array:
        return jnp.asarray(tuple(case.parameter_weight for case in self.cases))

    def batch(self, /) -> OperatorBatch:
        return self.physical_batch

    def evaluate_cases(
        self,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        /,
    ) -> MechanicsPerCaseResult:
        _validate_parameter_batch(
            prediction,
            batch,
            self.physical_batch,
            len(self.cases),
        )
        term_values: dict[str, list[Array]] = {term.name: [] for term in self.terms}
        measure_ids: dict[str, list[str]] = {term.name: [] for term in self.terms}
        for index, (case, case_batch) in enumerate(
            zip(self.cases, self.evaluation_batches, strict=True)
        ):
            case_prediction = _take_prediction(prediction, case_batch, index)
            adapted: dict[str, Any] = {}
            for adapter in self.adapters:
                selected = _select_prediction_fields(
                    case_prediction,
                    adapter.field_names,
                )
                adapted.update(
                    adapter(
                        selected,
                        case.realization,
                        geometry=case.geometry,
                    )
                )
            for term in self.terms:
                value = term.evaluate(adapted, case_prediction, case_batch, case)
                term_values[term.name].append(value)
                measure_id = case_batch.query(term.query_name).measure_id
                if measure_id is None:
                    raise RuntimeError(
                        "Validated mechanics functional lost its physical measure ID."
                    )
                measure_ids[term.name].append(measure_id)
        stacked = frozendict(
            {name: jnp.stack(values) for name, values in term_values.items()}
        )
        totals = jnp.sum(jnp.stack(tuple(stacked.values()), axis=0), axis=0)
        valid = jnp.ones(totals.shape, dtype=bool)
        return MechanicsPerCaseResult(
            values=totals,
            term_values=stacked,
            valid=valid,
            case_ids=tuple(case.realization.case_id for case in self.cases),
            stratum_ids=tuple(case.realization.stratum_id for case in self.cases),
            measure_ids=frozendict(
                {name: tuple(values) for name, values in measure_ids.items()}
            ),
            formulation=self.formulation,
            problem_id=self.problem_id,
            problem_fingerprint=self.problem_fingerprint,
        )


class _MechanicsProblemView:
    _problem: _MechanicsOperatorProblem

    @property
    def case_builder(self) -> MechanicsCaseBuilder:
        return self._problem.case_builder

    @property
    def adapters(self) -> tuple[OperatorTrialFieldAdapter, ...]:
        return self._problem.adapters

    @property
    def terms(self) -> tuple[MechanicsCaseFunctional, ...]:
        return self._problem.terms

    @property
    def formulation(self) -> MechanicsOperatorFormulation:
        return self._problem.formulation

    @property
    def problem_id(self) -> str:
        return self._problem.problem_id

    @property
    def cases(self) -> tuple[MechanicsOperatorCase, ...]:
        return self._problem.cases

    @property
    def problem_fingerprint(self) -> str:
        return self._problem.problem_fingerprint

    @property
    def metadata(self) -> frozendict[str, str]:
        return self._problem.metadata

    @property
    def parameter_weights(self) -> Array:
        return self._problem.parameter_weights

    def batch(self, /) -> OperatorBatch:
        return self._problem.batch()

    def evaluate_cases(
        self,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        /,
    ) -> MechanicsPerCaseResult:
        return self._problem.evaluate_cases(prediction, batch)


@dataclass(frozen=True)
class ConservativeMechanicsOperatorProblem(_MechanicsProblemView):
    """Verified scalar-potential mechanics over complete physical cases."""

    _problem: _MechanicsOperatorProblem

    def __init__(
        self,
        case_builder: MechanicsCaseBuilder,
        trial_fields: OperatorTrialFieldAdapter,
        energy_terms: Sequence[MechanicsCaseFunctional],
        /,
        *,
        problem_id: str,
    ):
        problem = _MechanicsOperatorProblem.create(
            case_builder,
            (trial_fields,),
            energy_terms,
            formulation="conservative",
            problem_id=problem_id,
            allowed_kinds=("energy",),
        )
        object.__setattr__(self, "_problem", problem)


@dataclass(frozen=True)
class MechanicsResidualOperatorProblem(_MechanicsProblemView):
    """Non-minimization mechanics residuals reduced on explicit physical measures."""

    _problem: _MechanicsOperatorProblem

    def __init__(
        self,
        case_builder: MechanicsCaseBuilder,
        trial_fields: OperatorTrialFieldAdapter,
        residual_terms: Sequence[MechanicsCaseFunctional],
        /,
        *,
        problem_id: str,
    ):
        problem = _MechanicsOperatorProblem.create(
            case_builder,
            (trial_fields,),
            residual_terms,
            formulation="residual",
            problem_id=problem_id,
            allowed_kinds=("residual",),
        )
        object.__setattr__(self, "_problem", problem)


@dataclass(frozen=True)
class MixedMechanicsOperatorProblem(_MechanicsProblemView):
    """Primal/dual block residual system; never represented as a fake potential."""

    _problem: _MechanicsOperatorProblem

    def __init__(
        self,
        case_builder: MechanicsCaseBuilder,
        primal_fields: OperatorTrialFieldAdapter,
        dual_fields: OperatorTrialFieldAdapter,
        residual_blocks: Sequence[MechanicsCaseFunctional],
        /,
        *,
        problem_id: str,
        gauge_blocks: Sequence[MechanicsCaseFunctional] = (),
    ):
        problem = _MechanicsOperatorProblem.create(
            case_builder,
            (primal_fields, dual_fields),
            tuple(residual_blocks) + tuple(gauge_blocks),
            formulation="mixed",
            problem_id=problem_id,
            allowed_kinds=("mixed_block", "gauge"),
        )
        if not any(term.kind == "mixed_block" for term in problem.terms):
            raise ValueError(
                "Mixed mechanics requires at least one physical residual block."
            )
        object.__setattr__(self, "_problem", problem)


MechanicsOperatorProblem = (
    ConservativeMechanicsOperatorProblem
    | MechanicsResidualOperatorProblem
    | MixedMechanicsOperatorProblem
)


def _unwrap_problem(problem: MechanicsOperatorProblem, /) -> _MechanicsOperatorProblem:
    if not isinstance(
        problem,
        (
            ConservativeMechanicsOperatorProblem,
            MechanicsResidualOperatorProblem,
            MixedMechanicsOperatorProblem,
        ),
    ):
        raise TypeError("Unknown mechanics operator problem type.")
    return problem._problem


@dataclass(frozen=True)
class _MechanicsLossBase(AbstractOperatorLossTerm):
    problem: MechanicsOperatorProblem
    parameter_reduction: MechanicsCaseReduction
    name: str
    weight: float
    expected_formulation: MechanicsOperatorFormulation

    def __post_init__(self):
        resolved = _unwrap_problem(self.problem)
        if resolved.formulation != self.expected_formulation:
            raise TypeError(
                f"Mechanics loss requires a {self.expected_formulation} problem."
            )
        if not isinstance(self.parameter_reduction, MechanicsCaseReduction):
            raise TypeError("parameter_reduction must be a MechanicsCaseReduction.")
        if (
            self.parameter_reduction.reduction_id
            != resolved.case_builder.reduction.reduction_id
        ):
            raise ValueError(
                "Mechanics loss reduction must match the case builder metadata binding."
            )
        if not self.name:
            raise ValueError("Mechanics operator loss names must be non-empty.")
        if not math.isfinite(float(self.weight)):
            raise ValueError("Mechanics operator loss weights must be finite.")

    def evaluate(
        self,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        /,
    ) -> MechanicsOperatorLossResult:
        problem = _unwrap_problem(self.problem)
        cases = problem.evaluate_cases(prediction, batch)
        risk = self.parameter_reduction.evaluate(
            cases.values,
            probability_weights=problem.parameter_weights,
            valid=cases.valid,
        )
        value = jnp.asarray(self.weight, dtype=risk.value.dtype) * risk.value
        return MechanicsOperatorLossResult(
            value=value,
            cases=cases,
            risk=risk,
            loss_fingerprint=self.fingerprint,
        )

    def __call__(
        self,
        model: Any,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        targets: OperatorTargetBatch,
        /,
        *,
        key: Key[Array, ""],
        step: Array,
        training: bool,
        context: OperatorLossContext,
    ) -> Array:
        del model, prediction, batch, targets, key, step, training
        physical_prediction, physical_batch, _ = context.view("physical")
        return self.evaluate(physical_prediction, physical_batch).value

    def contribution(
        self,
        model: Any,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        targets: OperatorTargetBatch,
        /,
        *,
        key: Key[Array, ""],
        step: Array,
        training: bool,
        context: OperatorLossContext,
    ) -> _ObjectiveContribution:
        value = self(
            model,
            prediction,
            batch,
            targets,
            key=key,
            step=step,
            training=training,
            context=context,
        )
        return _ObjectiveContribution(value, jnp.asarray(1.0, dtype=value.dtype))

    @property
    def fingerprint(self) -> str:
        problem = _unwrap_problem(self.problem)
        return canonical_fingerprint(
            {
                "kind": "mechanics-operator-loss",
                "name": self.name,
                "weight": float(self.weight),
                "formulation": self.expected_formulation,
                "problem": problem.problem_fingerprint,
                "parameter_reduction": self.parameter_reduction.reduction_id,
                "space": "physical",
            }
        )


@dataclass(frozen=True, init=False)
class ExpectedMechanicsEnergyLoss(_MechanicsLossBase):
    """Outer parameter risk of complete signed per-case physical energies."""

    def __init__(
        self,
        problem: ConservativeMechanicsOperatorProblem,
        parameter_reduction: MechanicsCaseReduction,
        /,
        *,
        name: str = "expected_mechanics_energy",
        weight: float = 1.0,
    ):
        _initialize_loss(
            self,
            problem,
            parameter_reduction,
            name=name,
            weight=weight,
            formulation="conservative",
        )


@dataclass(frozen=True, init=False)
class MechanicsResidualLoss(_MechanicsLossBase):
    """Outer parameter risk of complete physical residual-case scalars."""

    def __init__(
        self,
        problem: MechanicsResidualOperatorProblem,
        parameter_reduction: MechanicsCaseReduction,
        /,
        *,
        name: str = "mechanics_residual",
        weight: float = 1.0,
    ):
        _initialize_loss(
            self,
            problem,
            parameter_reduction,
            name=name,
            weight=weight,
            formulation="residual",
        )


@dataclass(frozen=True, init=False)
class MixedMechanicsLoss(_MechanicsLossBase):
    """Outer parameter risk of named mixed/KKT residual blocks and gauges."""

    def __init__(
        self,
        problem: MixedMechanicsOperatorProblem,
        parameter_reduction: MechanicsCaseReduction,
        /,
        *,
        name: str = "mixed_mechanics",
        weight: float = 1.0,
    ):
        _initialize_loss(
            self,
            problem,
            parameter_reduction,
            name=name,
            weight=weight,
            formulation="mixed",
        )


def _initialize_loss(
    loss: _MechanicsLossBase,
    problem: MechanicsOperatorProblem,
    reduction: MechanicsCaseReduction,
    /,
    *,
    name: str,
    weight: float,
    formulation: MechanicsOperatorFormulation,
) -> None:
    object.__setattr__(loss, "problem", problem)
    object.__setattr__(loss, "parameter_reduction", reduction)
    object.__setattr__(loss, "name", str(name))
    object.__setattr__(loss, "weight", float(weight))
    object.__setattr__(loss, "expected_formulation", formulation)
    loss.__post_init__()


def mechanics_operator_metadata(
    problem: MechanicsOperatorProblem,
    /,
) -> frozendict[str, str]:
    """Return canonical mechanics bindings for existing free metadata maps."""
    return _unwrap_problem(problem).metadata


def _validate_parameter_batch(
    prediction: OperatorPrediction,
    batch: OperatorBatch,
    expected_batch: OperatorBatch,
    case_count: int,
    /,
) -> None:
    if not isinstance(prediction, OperatorPrediction):
        raise TypeError("prediction must be an OperatorPrediction.")
    if not isinstance(batch, OperatorBatch):
        raise TypeError("batch must be an OperatorBatch.")
    expected_axes = ("parameter",)
    expected_shape = (case_count,)
    if batch.case_axes != expected_axes or batch.case_shape != expected_shape:
        raise ValueError(
            "Mechanics losses require the complete declared support on one "
            "'parameter' case axis."
        )
    if prediction.case_axes != expected_axes or prediction.case_shape != expected_shape:
        raise ValueError(
            "Mechanics predictions must retain the complete 'parameter' case axis."
        )
    if set(prediction.queries) != set(batch.queries):
        raise ValueError("Mechanics prediction and batch query branches must match.")
    if set(batch.inputs) != set(expected_batch.inputs):
        raise ValueError(
            "Mechanics batch input branches differ from the declared design."
        )
    if set(batch.queries) != set(expected_batch.queries):
        raise ValueError(
            "Mechanics batch query branches differ from the declared design."
        )
    for name, expected in expected_batch.queries.items():
        actual = batch.query(name)
        if (
            actual.support_id != expected.support_id
            or actual.measure_id != expected.measure_id
        ):
            raise ValueError(
                f"Mechanics query {name!r} changed physical support or measure."
            )


def _take_prediction(
    prediction: OperatorPrediction,
    batch: OperatorBatch,
    index: int,
    /,
) -> OperatorPrediction:
    fields = {
        name: OperatorFieldBatch(
            jnp.take(field.values, index, axis=0),
            query_name=field.query_name,
            spec=field.spec,
        )
        for name, field in prediction.fields.items()
    }
    return OperatorPrediction(fields, batch.queries)


def _select_prediction_fields(
    prediction: OperatorPrediction,
    names: Sequence[str],
    /,
) -> OperatorPrediction:
    selected = tuple(str(name) for name in names)
    fields = {name: prediction.field(name) for name in selected}
    query_names = tuple(dict.fromkeys(field.query_name for field in fields.values()))
    queries = {name: prediction.query_geometry(name) for name in query_names}
    return OperatorPrediction(fields, queries)


__all__ = [
    "ConservativeMechanicsOperatorProblem",
    "ExpectedMechanicsEnergyLoss",
    "MechanicsCaseFunctional",
    "MechanicsCaseFunctionalKind",
    "MechanicsOperatorFormulation",
    "MechanicsOperatorLossResult",
    "MechanicsPerCaseResult",
    "MechanicsResidualLoss",
    "MechanicsResidualOperatorProblem",
    "MixedMechanicsLoss",
    "MixedMechanicsOperatorProblem",
    "mechanics_operator_metadata",
]
