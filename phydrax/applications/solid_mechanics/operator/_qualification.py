#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ...._doc import DOC_KEY0
from ...._fingerprint import canonical_fingerprint
from ...._frozendict import frozendict
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....nn._keys import EvalKey
from ....nn.operator.data import (
    OperatorBatch,
    OperatorFieldBatch,
    OperatorPrediction,
)
from ....nn.operator.training._risk import (
    MechanicsCaseReduction,
    MechanicsCaseReductionResult,
)
from ....nn.operator.training._trained_operator import TrainedOperator
from ._cases import MechanicsCaseBuilder, MechanicsOperatorCase
from ._parameters import (
    MechanicsParameterDistribution,
    MechanicsParameterRealization,
    MechanicsParameterSpec,
)


MechanicsSupportStatus = Literal["supported", "out_of_support", "invalid_case"]


@dataclass(frozen=True)
class MechanicsSupportEvidence:
    """Explicit parameter/geometry support decision for one inference case."""

    status: MechanicsSupportStatus
    reason: str
    case_id: str
    parameter_spec_fingerprint: str
    realization_fingerprint: str
    geometry_fingerprint: str

    def __post_init__(self):
        if self.status not in ("supported", "out_of_support", "invalid_case"):
            raise ValueError("Unknown mechanics support status.")
        if not self.reason or not self.case_id or not self.parameter_spec_fingerprint:
            raise ValueError("Mechanics support evidence identities must be non-empty.")

    @property
    def supported(self) -> bool:
        return self.status == "supported"


class MechanicsQualificationMetric(StrictModule, NonTrainableState):
    """One held-out physical-unit metric reduced independently per case."""

    evaluator: Callable = eqx.field(static=True)
    name: str = eqx.field(static=True)
    query_name: str = eqx.field(static=True)
    unit: str = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)
    expected_measure_id: str | None = eqx.field(static=True)
    metric_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        evaluator: Callable,
        /,
        *,
        query_name: str,
        unit: str,
        metric_id: str,
        expected_measure_id: str | None = None,
    ):
        resolved_name = str(name)
        resolved_query = str(query_name)
        resolved_unit = str(unit)
        identifier = str(metric_id)
        if not resolved_name or not resolved_query or not resolved_unit or not identifier:
            raise ValueError(
                "Qualification metric names, query names, units, and IDs must be non-empty."
            )
        if not callable(evaluator):
            raise TypeError("Qualification metric evaluators must be callable.")
        expected = None if expected_measure_id is None else str(expected_measure_id)
        if expected == "":
            raise ValueError("Expected qualification measure IDs must be non-empty.")
        fingerprint = canonical_fingerprint(
            {
                "kind": "mechanics-qualification-metric",
                "name": resolved_name,
                "query": resolved_query,
                "unit": resolved_unit,
                "metric_id": identifier,
                "expected_measure": expected,
            }
        )
        self.evaluator = evaluator
        self.name = resolved_name
        self.query_name = resolved_query
        self.unit = resolved_unit
        self.metric_id = identifier
        self.expected_measure_id = expected
        self.metric_fingerprint = fingerprint

    def evaluate(
        self,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        case: MechanicsOperatorCase,
        /,
    ) -> Array:
        query = batch.query(self.query_name)
        if not query.has_physical_quadrature or query.measure_id is None:
            raise ValueError(
                f"Qualification metric {self.name!r} requires explicit physical "
                f"quadrature on query {self.query_name!r}."
            )
        if (
            self.expected_measure_id is not None
            and query.measure_id != self.expected_measure_id
        ):
            raise ValueError(
                f"Qualification metric {self.name!r} received measure "
                f"{query.measure_id!r}; expected {self.expected_measure_id!r}."
            )
        physical_weights = query.weights()
        invalid_measure = (
            jnp.any(~jnp.isfinite(physical_weights))
            | jnp.any(physical_weights < 0.0)
            | (jnp.sum(physical_weights) <= 0.0)
        )
        value = jnp.asarray(self.evaluator(prediction, batch, case))
        if value.shape != ():
            raise ValueError(
                f"Qualification metric {self.name!r} must return one scalar per case."
            )
        if jnp.iscomplexobj(value):
            raise TypeError("Qualification metrics must be real.")
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(float)
        value = eqx.error_if(
            value,
            invalid_measure,
            f"Qualification metric {self.name!r} has an invalid physical measure.",
        )
        return eqx.error_if(
            value,
            ~jnp.isfinite(value),
            f"Qualification metric {self.name!r} returned a nonfinite value.",
        )


class MechanicsOperatorQualification(StrictModule, NonTrainableState):
    """Frozen held-out parameter design and artifact compatibility contract."""

    training_distribution: MechanicsParameterDistribution
    held_out_case_builder: MechanicsCaseBuilder
    metrics: tuple[MechanicsQualificationMetric, ...]
    parameter_reduction: MechanicsCaseReduction
    support_spec: MechanicsParameterSpec
    required_metadata: frozendict[str, str] = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)
    qualification_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        training_distribution: MechanicsParameterDistribution,
        held_out_case_builder: MechanicsCaseBuilder,
        metrics: Sequence[MechanicsQualificationMetric],
        parameter_reduction: MechanicsCaseReduction,
        /,
        *,
        required_metadata: Mapping[str, str],
        qualification_id: str,
        support_spec: MechanicsParameterSpec | None = None,
    ):
        if not isinstance(training_distribution, MechanicsParameterDistribution):
            raise TypeError(
                "training_distribution must be a MechanicsParameterDistribution."
            )
        if not isinstance(held_out_case_builder, MechanicsCaseBuilder):
            raise TypeError("held_out_case_builder must be a MechanicsCaseBuilder.")
        if not isinstance(parameter_reduction, MechanicsCaseReduction):
            raise TypeError("parameter_reduction must be a MechanicsCaseReduction.")
        resolved_metrics = tuple(metrics)
        if not resolved_metrics or any(
            not isinstance(metric, MechanicsQualificationMetric)
            for metric in resolved_metrics
        ):
            raise TypeError(
                "Mechanics qualification requires MechanicsQualificationMetric values."
            )
        names = tuple(metric.name for metric in resolved_metrics)
        if len(set(names)) != len(names):
            raise ValueError("Qualification metric names must be unique.")
        held_out = held_out_case_builder.distribution
        training_distribution.assert_disjoint(held_out, by="realization")
        resolved_support = (
            training_distribution.spec if support_spec is None else support_spec
        )
        if not isinstance(resolved_support, MechanicsParameterSpec):
            raise TypeError("support_spec must be a MechanicsParameterSpec.")
        if resolved_support.spec_fingerprint != held_out.spec.spec_fingerprint:
            raise ValueError(
                "Held-out and support parameter specifications must be identical."
            )
        if (
            parameter_reduction.reduction_id
            != held_out_case_builder.reduction.reduction_id
        ):
            raise ValueError(
                "Qualification reduction must match held-out case provenance metadata."
            )
        bindings = frozendict(
            {str(name): str(value) for name, value in required_metadata.items()}
        )
        if not bindings or any(not name or not value for name, value in bindings.items()):
            raise ValueError(
                "Qualification requires non-empty mechanics artifact metadata bindings."
            )
        identifier = str(qualification_id)
        if not identifier:
            raise ValueError("Mechanics qualification IDs must be non-empty.")
        fingerprint = canonical_fingerprint(
            {
                "kind": "mechanics-operator-qualification",
                "qualification_id": identifier,
                "training_distribution": (training_distribution.distribution_fingerprint),
                "held_out_distribution": held_out.distribution_fingerprint,
                "held_out_case_builder": held_out_case_builder.builder_id,
                "support_spec": resolved_support.spec_fingerprint,
                "metrics": [metric.metric_fingerprint for metric in resolved_metrics],
                "parameter_reduction": parameter_reduction.reduction_id,
                "required_metadata": dict(bindings),
            }
        )
        self.training_distribution = training_distribution
        self.held_out_case_builder = held_out_case_builder
        self.metrics = resolved_metrics
        self.parameter_reduction = parameter_reduction
        self.support_spec = resolved_support
        self.required_metadata = bindings
        self.qualification_id = identifier
        self.qualification_fingerprint = fingerprint


class MechanicsOperatorEvidence(StrictModule):
    """Held-out physical evidence without a schema-version surrogate."""

    metric_values: frozendict[str, Array]
    metric_risks: frozendict[str, MechanicsCaseReductionResult]
    observed_worst_case: frozendict[str, Array]
    support: tuple[MechanicsSupportEvidence, ...] = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    stratum_ids: tuple[str, ...] = eqx.field(static=True)
    measure_ids: frozendict[str, tuple[str, ...]] = eqx.field(static=True)
    metric_units: frozendict[str, str] = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)
    operator_contract_fingerprint: str = eqx.field(static=True)
    held_out_distribution_fingerprint: str = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)
    qualification_fingerprint: str = eqx.field(static=True)
    evidence_fingerprint: str = eqx.field(static=True)


class MechanicsOperatorInferenceResult(StrictModule):
    """Frozen one-pass inference or an explicit unsupported/OOD refusal."""

    prediction: OperatorPrediction | None
    case: MechanicsOperatorCase
    support: MechanicsSupportEvidence = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)
    inference_fingerprint: str = eqx.field(static=True)
    inference_kind: Literal["amortized"] = eqx.field(static=True)


def assess_mechanics_support(
    support_spec: MechanicsParameterSpec,
    realization: MechanicsParameterRealization,
    /,
    *,
    geometry_fingerprint: str = "",
    validity: Callable[[MechanicsParameterRealization], bool] | None = None,
) -> MechanicsSupportEvidence:
    """Classify support without evaluating an operator outside its declared law."""
    if not isinstance(support_spec, MechanicsParameterSpec):
        raise TypeError("support_spec must be a MechanicsParameterSpec.")
    if not isinstance(realization, MechanicsParameterRealization):
        raise TypeError("realization must be a MechanicsParameterRealization.")
    geometry = str(geometry_fingerprint)
    if not support_spec.contains(realization.values):
        return MechanicsSupportEvidence(
            status="out_of_support",
            reason="parameter values lie outside the declared operator support",
            case_id=realization.case_id,
            parameter_spec_fingerprint=support_spec.spec_fingerprint,
            realization_fingerprint=realization.realization_fingerprint,
            geometry_fingerprint=geometry,
        )
    if validity is not None:
        if not callable(validity):
            raise TypeError("support validity must be callable or None.")
        valid = bool(validity(realization))
        if not valid:
            return MechanicsSupportEvidence(
                status="invalid_case",
                reason="parameter values are supported but the physical case is invalid",
                case_id=realization.case_id,
                parameter_spec_fingerprint=support_spec.spec_fingerprint,
                realization_fingerprint=realization.realization_fingerprint,
                geometry_fingerprint=geometry,
            )
    return MechanicsSupportEvidence(
        status="supported",
        reason="parameter values and physical case satisfy declared support",
        case_id=realization.case_id,
        parameter_spec_fingerprint=support_spec.spec_fingerprint,
        realization_fingerprint=realization.realization_fingerprint,
        geometry_fingerprint=geometry,
    )


def infer_mechanics_operator(
    trained_operator: TrainedOperator,
    case: MechanicsOperatorCase,
    support_spec: MechanicsParameterSpec,
    /,
    *,
    required_metadata: Mapping[str, str],
    key: EvalKey = DOC_KEY0,
) -> MechanicsOperatorInferenceResult:
    """Run one frozen direct evaluation; no prior state or optimizer is accepted."""
    if not isinstance(trained_operator, TrainedOperator):
        raise TypeError("trained_operator must be a TrainedOperator.")
    if not isinstance(case, MechanicsOperatorCase):
        raise TypeError("case must be a MechanicsOperatorCase.")
    _require_metadata(trained_operator, required_metadata)
    support = assess_mechanics_support(
        support_spec,
        case.realization,
        geometry_fingerprint=case.geometry.geometry_fingerprint,
    )
    prediction = (
        None if not support.supported else trained_operator.predict(case.batch, key=key)
    )
    fingerprint = canonical_fingerprint(
        {
            "kind": "amortized-mechanics-operator-inference",
            "artifact": trained_operator.artifact_id,
            "operator_contract": trained_operator.contract_fingerprint,
            "case": case.case_fingerprint,
            "support": support.status,
        }
    )
    return MechanicsOperatorInferenceResult(
        prediction=prediction,
        case=case,
        support=support,
        artifact_id=trained_operator.artifact_id,
        inference_fingerprint=fingerprint,
        inference_kind="amortized",
    )


def qualify_mechanics_operator(
    trained_operator: TrainedOperator,
    qualification: MechanicsOperatorQualification,
    /,
    *,
    key: EvalKey = DOC_KEY0,
) -> MechanicsOperatorEvidence:
    """Evaluate a frozen operator on a disjoint, complete held-out design."""
    if not isinstance(trained_operator, TrainedOperator):
        raise TypeError("trained_operator must be a TrainedOperator.")
    if not isinstance(qualification, MechanicsOperatorQualification):
        raise TypeError("qualification must be a MechanicsOperatorQualification.")
    _require_metadata(trained_operator, qualification.required_metadata)
    cases = qualification.held_out_case_builder.build_all()
    support = tuple(
        assess_mechanics_support(
            qualification.support_spec,
            case.realization,
            geometry_fingerprint=case.geometry.geometry_fingerprint,
        )
        for case in cases
    )
    if any(not item.supported for item in support):
        raise ValueError(
            "Held-out qualification contains unsupported or invalid physical cases."
        )
    batch = qualification.held_out_case_builder.stacked_batch(cases)
    prediction = trained_operator.predict(batch, key=key)
    expected_shape = (len(cases),)
    if prediction.case_axes != ("parameter",) or prediction.case_shape != expected_shape:
        raise ValueError(
            "Held-out prediction must preserve the complete 'parameter' case axis."
        )
    metric_values: dict[str, Array] = {}
    measure_ids: dict[str, tuple[str, ...]] = {}
    metric_risks: dict[str, MechanicsCaseReductionResult] = {}
    observed: dict[str, Array] = {}
    parameter_weights = jnp.asarray(tuple(case.parameter_weight for case in cases))
    for metric in qualification.metrics:
        per_case: list[Array] = []
        metric_measures: list[str] = []
        for index, case in enumerate(cases):
            case_batch = batch.take(index, axis="parameter")
            case_prediction = _take_prediction(prediction, case_batch, index)
            per_case.append(metric.evaluate(case_prediction, case_batch, case))
            measure_id = case_batch.query(metric.query_name).measure_id
            if measure_id is None:
                raise RuntimeError(
                    "Validated qualification metric lost its physical measure ID."
                )
            metric_measures.append(measure_id)
        values = jnp.stack(per_case)
        metric_values[metric.name] = values
        measure_ids[metric.name] = tuple(metric_measures)
        metric_risks[metric.name] = qualification.parameter_reduction.evaluate(
            values,
            probability_weights=parameter_weights,
            valid=jnp.ones(values.shape, dtype=bool),
        )
        observed[metric.name] = jnp.max(values)
    evidence_fingerprint = canonical_fingerprint(
        {
            "kind": "mechanics-operator-evidence",
            "qualification": qualification.qualification_fingerprint,
            "artifact": trained_operator.artifact_id,
            "operator_contract": trained_operator.contract_fingerprint,
            "cases": [case.case_fingerprint for case in cases],
            "metrics": [metric.metric_fingerprint for metric in qualification.metrics],
        }
    )
    return MechanicsOperatorEvidence(
        metric_values=frozendict(metric_values),
        metric_risks=frozendict(metric_risks),
        observed_worst_case=frozendict(observed),
        support=support,
        case_ids=tuple(case.realization.case_id for case in cases),
        stratum_ids=tuple(case.realization.stratum_id for case in cases),
        measure_ids=frozendict(measure_ids),
        metric_units=frozendict(
            {metric.name: metric.unit for metric in qualification.metrics}
        ),
        artifact_id=trained_operator.artifact_id,
        operator_contract_fingerprint=trained_operator.contract_fingerprint,
        held_out_distribution_fingerprint=(
            qualification.held_out_case_builder.distribution.distribution_fingerprint
        ),
        qualification_id=qualification.qualification_id,
        qualification_fingerprint=qualification.qualification_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
    )


def _require_metadata(
    trained_operator: TrainedOperator,
    required: Mapping[str, str],
    /,
) -> None:
    bindings = {str(name): str(value) for name, value in required.items()}
    if not bindings or any(not name or not value for name, value in bindings.items()):
        raise ValueError("Mechanics inference requires non-empty metadata bindings.")
    missing = set(bindings) - set(trained_operator.provenance)
    if missing:
        raise ValueError(
            f"Trained operator is missing mechanics metadata bindings {sorted(missing)}."
        )
    mismatched = tuple(
        name
        for name, expected in bindings.items()
        if trained_operator.provenance[name] != expected
    )
    if mismatched:
        raise ValueError(
            f"Trained operator mechanics metadata mismatch for {sorted(mismatched)}."
        )


def _take_prediction(
    prediction: OperatorPrediction,
    batch: OperatorBatch,
    index: int,
    /,
) -> OperatorPrediction:
    return OperatorPrediction(
        {
            name: OperatorFieldBatch(
                jnp.take(field.values, index, axis=0),
                query_name=field.query_name,
                spec=field.spec,
            )
            for name, field in prediction.fields.items()
        },
        batch.queries,
    )


__all__ = [
    "MechanicsOperatorEvidence",
    "MechanicsOperatorInferenceResult",
    "MechanicsOperatorQualification",
    "MechanicsQualificationMetric",
    "MechanicsSupportEvidence",
    "MechanicsSupportStatus",
    "assess_mechanics_support",
    "infer_mechanics_operator",
    "qualify_mechanics_operator",
]
