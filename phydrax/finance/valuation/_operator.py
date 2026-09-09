#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite, prod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...nn.operator.capabilities import (
    OperatorCompatibilityReport,
    OperatorProblemSpec,
    OperatorTrainingEvidence,
)
from ...nn.operator.data import OperatorBatch, OperatorPrediction
from ...nn.operator.protocols import OperatorModel
from ..core import FinanceEvidenceBinding, PricingLaw


class OperatorValuationApplicability(StrictModule):
    """Immutable law, domain, output and resource envelope for one operator."""

    pricing_law: PricingLaw
    problem_spec: OperatorProblemSpec
    architecture: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    output_field: str = eqx.field(static=True)
    training_independence_id: str = eqx.field(static=True)
    validation_tolerance: float = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
    maximum_output_values: int = eqx.field(static=True)

    def __init__(
        self,
        pricing_law: PricingLaw,
        problem_spec: OperatorProblemSpec,
        /,
        *,
        architecture: str,
        model_id: str,
        contract_id: str,
        domain_id: str,
        support_id: str,
        output_field: str,
        training_independence_id: str,
        validation_tolerance: float,
        constraint_tolerance: float,
        maximum_output_values: int,
    ):
        if not isinstance(pricing_law, PricingLaw):
            raise TypeError("pricing_law must be a PricingLaw.")
        if not isinstance(problem_spec, OperatorProblemSpec):
            raise TypeError("problem_spec must be an OperatorProblemSpec.")
        identifiers = tuple(
            str(value)
            for value in (
                architecture,
                model_id,
                contract_id,
                domain_id,
                support_id,
                output_field,
                training_independence_id,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError("Operator applicability identities must be nonempty.")
        validation = float(validation_tolerance)
        constraint = float(constraint_tolerance)
        output_budget = int(maximum_output_values)
        if (
            not isfinite(validation)
            or validation < 0.0
            or not isfinite(constraint)
            or constraint < 0.0
            or output_budget < 1
        ):
            raise ValueError("Operator tolerances/resource budget are invalid.")
        self.pricing_law = pricing_law
        self.problem_spec = problem_spec
        (
            self.architecture,
            self.model_id,
            self.contract_id,
            self.domain_id,
            self.support_id,
            self.output_field,
            self.training_independence_id,
        ) = identifiers
        self.validation_tolerance = validation
        self.constraint_tolerance = constraint
        self.maximum_output_values = output_budget


class OperatorDomainEvidence(StrictModule):
    """Typed financial-domain and support audit, independent of array shape."""

    maximum_support_violation: Array
    domain_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    factor_layout_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    valid: Array

    def __init__(
        self,
        maximum_support_violation: ArrayLike,
        /,
        *,
        domain_id: str,
        support_id: str,
        factor_layout_id: str,
        evidence_id: str,
        tolerance: float,
    ):
        violation = jnp.asarray(maximum_support_violation, dtype=float)
        identifiers = tuple(
            str(value) for value in (domain_id, support_id, factor_layout_id, evidence_id)
        )
        tolerance_ = float(tolerance)
        if violation.shape != () or any(not value for value in identifiers):
            raise ValueError("Operator domain evidence is malformed.")
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("Domain tolerance must be finite and nonnegative.")
        self.maximum_support_violation = violation
        self.domain_id, self.support_id, self.factor_layout_id, self.evidence_id = (
            identifiers
        )
        self.tolerance = tolerance_
        self.valid = jnp.isfinite(violation) & (violation <= tolerance_)


class OperatorCausalityEvidence(StrictModule):
    """Typed future-dependency audit against the pricing-law filtration."""

    maximum_future_dependency: Array
    filtration_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    checked_rollout_steps: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    valid: Array

    def __init__(
        self,
        maximum_future_dependency: ArrayLike,
        /,
        *,
        filtration_id: str,
        evidence_id: str,
        checked_rollout_steps: int,
        tolerance: float,
    ):
        dependency = jnp.asarray(maximum_future_dependency, dtype=float)
        filtration = str(filtration_id)
        identifier = str(evidence_id)
        steps = int(checked_rollout_steps)
        tolerance_ = float(tolerance)
        if dependency.shape != () or not filtration or not identifier or steps < 1:
            raise ValueError("Operator causality evidence is malformed.")
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("Causality tolerance must be finite and nonnegative.")
        self.maximum_future_dependency = dependency
        self.filtration_id = filtration
        self.evidence_id = identifier
        self.checked_rollout_steps = steps
        self.tolerance = tolerance_
        self.valid = jnp.isfinite(dependency) & (dependency <= tolerance_)


class OperatorIndependentValidation(StrictModule):
    """Held-out residual and baseline evidence distinct from training evidence."""

    root_mean_square_error: Array
    maximum_absolute_error: Array
    relative_error: Array
    constraint_residual: Array
    baseline_error: Array
    active_value_count: Array
    validation_id: str = eqx.field(static=True)
    validation_independence_id: str = eqx.field(static=True)
    training_independence_id: str = eqx.field(static=True)
    baseline_id: str = eqx.field(static=True)
    independent: bool = eqx.field(static=True)
    finite: Array
    valid: Array


class OperatorResourceEvidence(StrictModule):
    """Bounded output materialization evidence for one evaluation."""

    output_value_count: int = eqx.field(static=True)
    maximum_output_values: int = eqx.field(static=True)
    within_budget: bool = eqx.field(static=True)


class OperatorValuationEvidence(StrictModule):
    compatibility: OperatorCompatibilityReport = eqx.field(static=True)
    validation: OperatorIndependentValidation
    domain: OperatorDomainEvidence
    causality: OperatorCausalityEvidence
    resources: OperatorResourceEvidence
    evidence_binding: FinanceEvidenceBinding
    law_compatible: bool = eqx.field(static=True)
    complete_evidence_binding: bool = eqx.field(static=True)


class OperatorValuationCandidate(StrictModule):
    """Candidate-only neural-operator prediction with fail-closed evidence."""

    applicability: OperatorValuationApplicability
    prediction: OperatorPrediction
    evidence: OperatorValuationEvidence
    accepted: Array
    candidate_only: bool = eqx.field(static=True, default=True)


def operator_independent_validation(
    applicability: OperatorValuationApplicability,
    prediction: OperatorPrediction,
    reference_values: ArrayLike,
    batch: OperatorBatch,
    /,
    *,
    validation_id: str,
    validation_independence_id: str,
    constraint_residual: ArrayLike,
    baseline_error: ArrayLike,
    baseline_id: str,
) -> OperatorIndependentValidation:
    """Compare a prediction with a separately identified validation target."""
    if not isinstance(applicability, OperatorValuationApplicability):
        raise TypeError("applicability must be OperatorValuationApplicability.")
    if not isinstance(prediction, OperatorPrediction) or not isinstance(
        batch, OperatorBatch
    ):
        raise TypeError("prediction and batch must use canonical operator contracts.")
    identifiers = tuple(
        str(value) for value in (validation_id, validation_independence_id, baseline_id)
    )
    if any(not value for value in identifiers):
        raise ValueError("Operator validation identities must be nonempty.")
    field = prediction.field(applicability.output_field)
    reference = jnp.asarray(reference_values)
    if reference.shape != field.values.shape:
        raise ValueError("Independent target shape must equal the selected output shape.")
    constraint = jnp.asarray(constraint_residual, dtype=float)
    baseline = jnp.asarray(baseline_error, dtype=float)
    if constraint.shape != () or baseline.shape != ():
        raise ValueError("Constraint and baseline errors must be scalars.")
    query = batch.query(field.query_name)
    mask = query.mask_array(case_shape=batch.case_shape)
    if field.spec.channel_shape:
        mask = mask.reshape(mask.shape + (1,) * len(field.spec.channel_shape))
        mask = jnp.broadcast_to(mask, field.values.shape)
    residual = jnp.where(mask, field.values - reference, 0.0)
    count = jnp.sum(mask)
    squared = jnp.sum(jnp.abs(residual) ** 2) / jnp.maximum(count, 1)
    rmse = jnp.sqrt(squared)
    maximum = jnp.max(jnp.where(mask, jnp.abs(residual), 0.0))
    scale = jnp.sqrt(
        jnp.sum(jnp.where(mask, jnp.abs(reference) ** 2, 0.0)) / jnp.maximum(count, 1)
    )
    relative = rmse / jnp.where(scale > 0.0, scale, 1.0)
    independent = identifiers[1] != applicability.training_independence_id
    finite = (
        (count > 0)
        & jnp.isfinite(rmse)
        & jnp.isfinite(maximum)
        & jnp.isfinite(relative)
        & jnp.isfinite(constraint)
        & jnp.isfinite(baseline)
        & (baseline >= 0.0)
    )
    valid = (
        independent
        & finite
        & (relative <= applicability.validation_tolerance)
        & (relative <= baseline)
        & (constraint <= applicability.constraint_tolerance)
    )
    return OperatorIndependentValidation(
        rmse,
        maximum,
        relative,
        constraint,
        baseline,
        count,
        identifiers[0],
        identifiers[1],
        applicability.training_independence_id,
        identifiers[2],
        independent,
        finite,
        valid,
    )


def _complete_binding(binding: FinanceEvidenceBinding, /) -> bool:
    return bool(
        binding.data_evidence_ids
        and binding.model_evidence_ids
        and binding.numerical_evidence_ids
        and binding.use_evidence_ids
    )


def evaluate_operator_valuation_candidate(
    applicability: OperatorValuationApplicability,
    model: OperatorModel,
    batch: OperatorBatch,
    pricing_law: PricingLaw,
    training_evidence: OperatorTrainingEvidence,
    domain: OperatorDomainEvidence,
    causality: OperatorCausalityEvidence,
    evidence_binding: FinanceEvidenceBinding,
    reference_values: ArrayLike,
    /,
    *,
    validation_id: str,
    validation_independence_id: str,
    constraint_residual: ArrayLike,
    baseline_error: ArrayLike,
    baseline_id: str,
    key: Any = None,
) -> OperatorValuationCandidate:
    """Evaluate and gate one operator without bypassing native runtime contracts."""
    if not isinstance(applicability, OperatorValuationApplicability):
        raise TypeError("applicability must be OperatorValuationApplicability.")
    if not isinstance(model, OperatorModel):
        raise TypeError("model must implement the OperatorModel contract.")
    if not isinstance(batch, OperatorBatch):
        raise TypeError("batch must be an OperatorBatch.")
    if not isinstance(pricing_law, PricingLaw):
        raise TypeError("pricing_law must be a PricingLaw.")
    if not isinstance(training_evidence, OperatorTrainingEvidence):
        raise TypeError("training_evidence must be OperatorTrainingEvidence.")
    if not isinstance(domain, OperatorDomainEvidence):
        raise TypeError("domain must be OperatorDomainEvidence.")
    if not isinstance(causality, OperatorCausalityEvidence):
        raise TypeError("causality must be OperatorCausalityEvidence.")
    if not isinstance(evidence_binding, FinanceEvidenceBinding):
        raise TypeError("evidence_binding must be a FinanceEvidenceBinding.")
    contract = model.operator_contract
    compatibility = contract.validate(
        batch,
        problem=applicability.problem_spec,
        training_evidence=training_evidence,
    )
    compatibility.require_runtime()
    prediction = model.evaluate(batch, key=key)
    validation = operator_independent_validation(
        applicability,
        prediction,
        reference_values,
        batch,
        validation_id=validation_id,
        validation_independence_id=validation_independence_id,
        constraint_residual=constraint_residual,
        baseline_error=baseline_error,
        baseline_id=baseline_id,
    )
    field = prediction.field(applicability.output_field)
    output_count = prod(field.values.shape)
    resources = OperatorResourceEvidence(
        output_count,
        applicability.maximum_output_values,
        output_count <= applicability.maximum_output_values,
    )
    expected = applicability.pricing_law
    law_compatible = (
        pricing_law.law_id == expected.law_id
        and pricing_law.measure_id == expected.measure_id
        and pricing_law.numeraire_id == expected.numeraire_id
        and pricing_law.collateral_convention_id == expected.collateral_convention_id
        and pricing_law.factor_layout_id == expected.factor_layout_id
        and pricing_law.filtration_id == expected.filtration_id
    )
    typed_domain = (
        contract.architecture == applicability.architecture
        and domain.domain_id == applicability.domain_id
        and domain.support_id == applicability.support_id
        and domain.factor_layout_id == pricing_law.factor_layout_id
        and causality.filtration_id == pricing_law.filtration_id
        and causality.checked_rollout_steps == applicability.problem_spec.rollout_steps
    )
    complete = _complete_binding(evidence_binding)
    accepted = (
        compatibility.accepted
        & law_compatible
        & typed_domain
        & domain.valid
        & causality.valid
        & validation.valid
        & resources.within_budget
        & complete
    )
    evidence = OperatorValuationEvidence(
        compatibility,
        validation,
        domain,
        causality,
        resources,
        evidence_binding,
        law_compatible,
        complete,
    )
    return OperatorValuationCandidate(
        applicability,
        prediction,
        evidence,
        accepted,
        True,
    )


__all__ = [
    "OperatorCausalityEvidence",
    "OperatorDomainEvidence",
    "OperatorIndependentValidation",
    "OperatorResourceEvidence",
    "OperatorValuationApplicability",
    "OperatorValuationCandidate",
    "OperatorValuationEvidence",
    "evaluate_operator_valuation_candidate",
    "operator_independent_validation",
]
