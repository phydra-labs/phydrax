#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...solver._deep_bsde import DeepBSDEResult
from ..core import FinanceEvidenceBinding, PricingLaw


class DeepBSDEApplicability(StrictModule):
    """Immutable financial and resource envelope for one Deep-BSDE candidate."""

    pricing_law: PricingLaw
    contract_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    training_path_id: str = eqx.field(static=True)
    training_independence_id: str = eqx.field(static=True)
    max_validation_paths: int = eqx.field(static=True)
    max_time_steps: int = eqx.field(static=True)
    terminal_rmse_tolerance: float = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
    minimum_valid_fraction: float = eqx.field(static=True)

    def __init__(
        self,
        pricing_law: PricingLaw,
        /,
        *,
        contract_id: str,
        problem_id: str,
        process_id: str,
        support_id: str,
        training_path_id: str,
        training_independence_id: str,
        max_validation_paths: int,
        max_time_steps: int,
        terminal_rmse_tolerance: float,
        constraint_tolerance: float,
        minimum_valid_fraction: float = 1.0,
    ):
        if not isinstance(pricing_law, PricingLaw):
            raise TypeError("pricing_law must be a PricingLaw, not a P/stress law.")
        identifiers = tuple(
            str(value)
            for value in (
                contract_id,
                problem_id,
                process_id,
                support_id,
                training_path_id,
                training_independence_id,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError("Deep-BSDE applicability identities must be nonempty.")
        path_budget = int(max_validation_paths)
        step_budget = int(max_time_steps)
        terminal_tolerance = float(terminal_rmse_tolerance)
        constraint_tolerance_ = float(constraint_tolerance)
        valid_fraction = float(minimum_valid_fraction)
        if path_budget < 1 or step_budget < 1:
            raise ValueError("Deep-BSDE validation resource budgets must be positive.")
        if (
            not isfinite(terminal_tolerance)
            or terminal_tolerance < 0.0
            or not isfinite(constraint_tolerance_)
            or constraint_tolerance_ < 0.0
            or not isfinite(valid_fraction)
            or not 0.0 < valid_fraction <= 1.0
        ):
            raise ValueError("Deep-BSDE tolerances/valid fraction are invalid.")
        self.pricing_law = pricing_law
        (
            self.contract_id,
            self.problem_id,
            self.process_id,
            self.support_id,
            self.training_path_id,
            self.training_independence_id,
        ) = identifiers
        self.max_validation_paths = path_budget
        self.max_time_steps = step_budget
        self.terminal_rmse_tolerance = terminal_tolerance
        self.constraint_tolerance = constraint_tolerance_
        self.minimum_valid_fraction = valid_fraction


class DeepBSDECausalityEvidence(StrictModule):
    """Typed predictability audit against one declared pricing filtration."""

    maximum_future_dependency: Array
    checked_time_nodes: int = eqx.field(static=True)
    filtration_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    valid: Array

    def __init__(
        self,
        maximum_future_dependency: ArrayLike,
        /,
        *,
        checked_time_nodes: int,
        filtration_id: str,
        evidence_id: str,
        tolerance: float,
    ):
        dependency = jnp.asarray(maximum_future_dependency, dtype=float)
        nodes = int(checked_time_nodes)
        tolerance_ = float(tolerance)
        filtration = str(filtration_id)
        identifier = str(evidence_id)
        if dependency.shape != () or nodes < 1:
            raise ValueError(
                "Causality evidence requires a scalar defect and time nodes."
            )
        if not filtration or not identifier:
            raise ValueError("Causality filtration/evidence IDs must be nonempty.")
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("Causality tolerance must be finite and nonnegative.")
        self.maximum_future_dependency = dependency
        self.checked_time_nodes = nodes
        self.filtration_id = filtration
        self.evidence_id = identifier
        self.tolerance = tolerance_
        self.valid = jnp.isfinite(dependency) & (dependency <= tolerance_)


class DeepBSDESupportEvidence(StrictModule):
    """Typed state-domain/support audit for held-out paths."""

    maximum_support_violation: Array
    active_path_count: Array
    support_id: str = eqx.field(static=True)
    factor_layout_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    valid: Array

    def __init__(
        self,
        maximum_support_violation: ArrayLike,
        active_path_count: ArrayLike,
        /,
        *,
        support_id: str,
        factor_layout_id: str,
        evidence_id: str,
        tolerance: float,
    ):
        violation = jnp.asarray(maximum_support_violation, dtype=float)
        count = jnp.asarray(active_path_count, dtype=jnp.int32)
        tolerance_ = float(tolerance)
        identifiers = tuple(
            str(value) for value in (support_id, factor_layout_id, evidence_id)
        )
        if violation.shape != () or count.shape != ():
            raise ValueError("Support violation and active path count must be scalars.")
        if any(not value for value in identifiers):
            raise ValueError("Support evidence identities must be nonempty.")
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("Support tolerance must be finite and nonnegative.")
        self.maximum_support_violation = violation
        self.active_path_count = count
        self.support_id, self.factor_layout_id, self.evidence_id = identifiers
        self.tolerance = tolerance_
        self.valid = jnp.isfinite(violation) & (violation <= tolerance_) & (count > 0)


class DeepBSDEIndependentValidation(StrictModule):
    """Held-out residual, baseline, and resource evidence kept separate from fit."""

    terminal_rmse: Array
    terminal_bias: Array
    control_rms: Array
    valid_fraction: Array
    constraint_residual: Array
    baseline_error: Array
    validation_path_count: int = eqx.field(static=True)
    validation_time_steps: int = eqx.field(static=True)
    validation_path_id: str = eqx.field(static=True)
    validation_independence_id: str = eqx.field(static=True)
    training_path_id: str = eqx.field(static=True)
    training_independence_id: str = eqx.field(static=True)
    baseline_id: str = eqx.field(static=True)
    independent: bool = eqx.field(static=True)
    finite: Array
    valid: Array


class DeepBSDECandidateEvidence(StrictModule):
    """Separated applicability, validation, causality, support and resource evidence."""

    independent_validation: DeepBSDEIndependentValidation
    causality: DeepBSDECausalityEvidence
    support: DeepBSDESupportEvidence
    evidence_binding: FinanceEvidenceBinding
    problem_compatible: bool = eqx.field(static=True)
    law_compatible: bool = eqx.field(static=True)
    resource_compatible: bool = eqx.field(static=True)
    complete_evidence_binding: bool = eqx.field(static=True)


class DeepBSDEValuationCandidate(StrictModule):
    """Candidate wrapper over an existing DeepBSDEResult; never a pricing claim."""

    applicability: DeepBSDEApplicability
    result: DeepBSDEResult
    evidence: DeepBSDECandidateEvidence
    accepted: Array
    candidate_only: bool = eqx.field(static=True, default=True)


def deep_bsde_independent_validation(
    applicability: DeepBSDEApplicability,
    result: DeepBSDEResult,
    /,
    *,
    validation_independence_id: str,
    constraint_residual: ArrayLike,
    baseline_error: ArrayLike,
    baseline_id: str,
) -> DeepBSDEIndependentValidation:
    """Build validation evidence from the solver's retained held-out rollout."""
    if not isinstance(applicability, DeepBSDEApplicability):
        raise TypeError("applicability must be a DeepBSDEApplicability.")
    if not isinstance(result, DeepBSDEResult):
        raise TypeError("result must be a DeepBSDEResult.")
    validation_independence = str(validation_independence_id)
    baseline_identifier = str(baseline_id)
    if not validation_independence or not baseline_identifier:
        raise ValueError("Validation independence and baseline IDs must be nonempty.")
    constraint = jnp.asarray(constraint_residual, dtype=float)
    baseline = jnp.asarray(baseline_error, dtype=float)
    if constraint.shape != () or baseline.shape != ():
        raise ValueError("Constraint residual and baseline error must be scalars.")
    paths = result.rollout.paths
    diagnostics = result.diagnostics
    independent = (
        paths.path_id != applicability.training_path_id
        and validation_independence != applicability.training_independence_id
    )
    finite = (
        diagnostics.finite
        & jnp.isfinite(constraint)
        & jnp.isfinite(baseline)
        & (baseline >= 0.0)
    )
    within_resources = (
        paths.num_paths <= applicability.max_validation_paths
        and paths.num_steps <= applicability.max_time_steps
    )
    valid = (
        independent
        & within_resources
        & finite
        & (diagnostics.terminal_rmse <= applicability.terminal_rmse_tolerance)
        & (diagnostics.terminal_rmse <= baseline)
        & (diagnostics.valid_fraction >= applicability.minimum_valid_fraction)
        & (constraint <= applicability.constraint_tolerance)
    )
    return DeepBSDEIndependentValidation(
        diagnostics.terminal_rmse,
        diagnostics.terminal_bias,
        diagnostics.control_rms,
        diagnostics.valid_fraction,
        constraint,
        baseline,
        paths.num_paths,
        paths.num_steps,
        paths.path_id,
        validation_independence,
        applicability.training_path_id,
        applicability.training_independence_id,
        baseline_identifier,
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


def assess_deep_bsde_candidate(
    applicability: DeepBSDEApplicability,
    result: DeepBSDEResult,
    pricing_law: PricingLaw,
    validation: DeepBSDEIndependentValidation,
    causality: DeepBSDECausalityEvidence,
    support: DeepBSDESupportEvidence,
    evidence_binding: FinanceEvidenceBinding,
    /,
) -> DeepBSDEValuationCandidate:
    """Gate a Deep-BSDE result by typed law, domain, causality and held-out evidence."""
    if not isinstance(applicability, DeepBSDEApplicability):
        raise TypeError("applicability must be a DeepBSDEApplicability.")
    if not isinstance(result, DeepBSDEResult):
        raise TypeError("result must be a DeepBSDEResult.")
    if not isinstance(pricing_law, PricingLaw):
        raise TypeError("pricing_law must be a PricingLaw.")
    if not isinstance(validation, DeepBSDEIndependentValidation):
        raise TypeError("validation must be DeepBSDEIndependentValidation evidence.")
    if not isinstance(causality, DeepBSDECausalityEvidence):
        raise TypeError("causality must be DeepBSDECausalityEvidence.")
    if not isinstance(support, DeepBSDESupportEvidence):
        raise TypeError("support must be DeepBSDESupportEvidence.")
    if not isinstance(evidence_binding, FinanceEvidenceBinding):
        raise TypeError("evidence_binding must be a FinanceEvidenceBinding.")
    expected = applicability.pricing_law
    law_compatible = (
        pricing_law.law_id == expected.law_id
        and pricing_law.measure_id == expected.measure_id
        and pricing_law.numeraire_id == expected.numeraire_id
        and pricing_law.collateral_convention_id == expected.collateral_convention_id
        and pricing_law.factor_layout_id == expected.factor_layout_id
        and pricing_law.filtration_id == expected.filtration_id
    )
    problem_compatible = (
        result.problem_id == applicability.problem_id
        and result.process_id == applicability.process_id
        and validation.validation_path_id == result.rollout.paths.path_id
        and validation.training_path_id == applicability.training_path_id
        and validation.training_independence_id == applicability.training_independence_id
    )
    resource_compatible = (
        validation.validation_path_count <= applicability.max_validation_paths
        and validation.validation_time_steps <= applicability.max_time_steps
    )
    typed_domain = (
        causality.filtration_id == pricing_law.filtration_id
        and causality.checked_time_nodes == result.rollout.paths.num_steps
        and support.support_id == applicability.support_id
        and support.factor_layout_id == pricing_law.factor_layout_id
    )
    complete = _complete_binding(evidence_binding)
    accepted = (
        law_compatible
        & problem_compatible
        & resource_compatible
        & typed_domain
        & validation.valid
        & causality.valid
        & support.valid
        & complete
    )
    evidence = DeepBSDECandidateEvidence(
        validation,
        causality,
        support,
        evidence_binding,
        law_compatible=law_compatible,
        problem_compatible=problem_compatible,
        resource_compatible=resource_compatible,
        complete_evidence_binding=complete,
    )
    return DeepBSDEValuationCandidate(
        applicability,
        result,
        evidence,
        accepted,
        True,
    )


__all__ = [
    "DeepBSDEApplicability",
    "DeepBSDECandidateEvidence",
    "DeepBSDECausalityEvidence",
    "DeepBSDEIndependentValidation",
    "DeepBSDESupportEvidence",
    "DeepBSDEValuationCandidate",
    "assess_deep_bsde_candidate",
    "deep_bsde_independent_validation",
]
