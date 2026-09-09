#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...stochastic._state_space import StateSpaceProblem
from ...uq._kalman import KalmanFilterResult, KalmanSmootherResult, rts_smoother
from ...uq._state_space_inference import (
    exact_state_space_log_likelihood,
    ExactStateSpaceLikelihood,
    finite_state_backward_smoother,
    finite_state_expected_transition_counts,
    finite_state_viterbi,
    FiniteStateFilterResult,
    FiniteStateSmootherResult,
    FiniteStateTransitionCountResult,
    FiniteStateViterbiResult,
)
from ..core import FinanceEvidenceBinding, PhysicalLaw


class FinancialStateSpaceDefinition(StrictModule):
    """Declared exact linear-Gaussian finance state-space inference route."""

    model_id: str = eqx.field(static=True)
    temporal_method: Literal["sequential", "parallel", "auto"] = eqx.field(static=True)
    covariance_regularization: float = eqx.field(static=True)
    smooth: bool = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        model_id: str,
        /,
        *,
        temporal_method: Literal["sequential", "parallel", "auto"] = "auto",
        covariance_regularization: float = 0.0,
        smooth: bool = True,
    ):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be nonempty.")
        if temporal_method not in ("sequential", "parallel", "auto"):
            raise ValueError("temporal_method must be sequential, parallel, or auto.")
        regularization = float(covariance_regularization)
        if not jnp.isfinite(regularization) or regularization < 0.0:
            raise ValueError("covariance_regularization must be finite and nonnegative.")
        self.model_id = model_id.strip()
        self.temporal_method = temporal_method
        self.covariance_regularization = regularization
        self.smooth = bool(smooth)
        self.definition_id = canonical_fingerprint(
            {
                "kind": "financial-state-space-definition",
                "model_id": self.model_id,
                "temporal_method": temporal_method,
                "covariance_regularization": regularization,
                "smooth": bool(smooth),
            }
        )


class RegimeDefinition(StrictModule):
    """Declared exact finite-state filtering, smoothing, and Viterbi route."""

    model_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(self, model_id: str, /):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be nonempty.")
        self.model_id = model_id.strip()
        self.definition_id = canonical_fingerprint(
            {"kind": "financial-regime-definition", "model_id": self.model_id}
        )


class FinancialStateSpacePlan(StrictModule):
    problem_id: str = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class FinancialStateSpaceFit(StrictModule):
    likelihood: ExactStateSpaceLikelihood
    filter_result: KalmanFilterResult
    smoother_result: KalmanSmootherResult | None
    log_likelihood: Any
    valid: Any
    status: Any
    evidence: FinanceEvidenceBinding = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)


class FinancialStateSpaceResult(StrictModule):
    filtered_state_mean: Any
    filtered_state_covariance: Any
    smoothed_state_mean: Any
    smoothed_state_covariance: Any
    fit: FinancialStateSpaceFit


class RegimeFit(StrictModule):
    likelihood: ExactStateSpaceLikelihood
    filter_result: FiniteStateFilterResult
    smoother_result: FiniteStateSmootherResult
    viterbi_result: FiniteStateViterbiResult
    transition_counts: FiniteStateTransitionCountResult
    evidence: FinanceEvidenceBinding = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)


class RegimeResult(StrictModule):
    filtered_probabilities: Any
    smoothed_probabilities: Any
    most_likely_states: Any
    log_likelihood: Any
    valid: Any
    fit: RegimeFit


def _evidence(data_id: str, definition_id: str, route: str) -> FinanceEvidenceBinding:
    return FinanceEvidenceBinding(
        (canonical_fingerprint({"kind": "state-space-data-evidence", "data": data_id}),),
        (
            canonical_fingerprint(
                {"kind": "state-space-model-evidence", "model": definition_id}
            ),
        ),
        (
            canonical_fingerprint(
                {"kind": "state-space-numerical-evidence", "route": route}
            ),
        ),
        (
            canonical_fingerprint(
                {"kind": "state-space-use-evidence", "trade_emission": False}
            ),
        ),
    )


def prepare_financial_state_space(
    problem: StateSpaceProblem,
    data_id: str,
    definition: FinancialStateSpaceDefinition,
    law: PhysicalLaw,
    /,
) -> FinancialStateSpacePlan:
    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    if not isinstance(definition, FinancialStateSpaceDefinition):
        raise TypeError("definition must be a FinancialStateSpaceDefinition.")
    if not isinstance(law, PhysicalLaw):
        raise TypeError("financial state estimation requires a PhysicalLaw.")
    if problem.model.model_id != definition.model_id:
        raise ValueError("problem model does not match the state-space definition.")
    plan_id = canonical_fingerprint(
        {
            "kind": "financial-state-space-plan",
            "problem": problem.problem_id,
            "data": str(data_id),
            "definition": definition.definition_id,
            "law": law.law_id,
        }
    )
    return FinancialStateSpacePlan(
        problem_id=problem.problem_id,
        data_id=str(data_id),
        definition_id=definition.definition_id,
        law_id=law.law_id,
        plan_id=plan_id,
    )


def fit_financial_state_space(
    problem: StateSpaceProblem,
    data_id: str,
    definition: FinancialStateSpaceDefinition,
    law: PhysicalLaw,
    /,
) -> FinancialStateSpaceFit:
    """Invoke native exact Kalman inference without reproducing its recursion."""

    plan = prepare_financial_state_space(problem, data_id, definition, law)
    likelihood = exact_state_space_log_likelihood(
        problem,
        method="kalman",
        covariance_regularization=definition.covariance_regularization,
        temporal_method=definition.temporal_method,
    )
    backend = likelihood.backend
    if not isinstance(backend, KalmanFilterResult):
        raise TypeError("the declared Kalman route did not return a Kalman result.")
    smoother = rts_smoother(backend) if definition.smooth else None
    return FinancialStateSpaceFit(
        likelihood=likelihood,
        filter_result=backend,
        smoother_result=smoother,
        log_likelihood=likelihood.total_log_likelihood,
        valid=likelihood.valid,
        status=likelihood.status,
        evidence=_evidence(str(data_id), definition.definition_id, "exact-kalman"),
        plan_id=plan.plan_id,
        law_id=law.law_id,
    )


def summarize_financial_state_space(
    fit: FinancialStateSpaceFit,
    /,
) -> FinancialStateSpaceResult:
    if not isinstance(fit, FinancialStateSpaceFit):
        raise TypeError("fit must be a FinancialStateSpaceFit.")
    smoother = fit.smoother_result
    return FinancialStateSpaceResult(
        filtered_state_mean=fit.filter_result.filtered_means,
        filtered_state_covariance=fit.filter_result.filtered_covariances,
        smoothed_state_mean=(
            fit.filter_result.filtered_means if smoother is None else smoother.means
        ),
        smoothed_state_covariance=(
            fit.filter_result.filtered_covariances
            if smoother is None
            else smoother.covariances
        ),
        fit=fit,
    )


def fit_regime_model(
    problem: StateSpaceProblem,
    data_id: str,
    definition: RegimeDefinition,
    law: PhysicalLaw,
    /,
) -> RegimeFit:
    """Invoke native exact finite-state inference and retain all path evidence."""

    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    if not isinstance(definition, RegimeDefinition):
        raise TypeError("definition must be a RegimeDefinition.")
    if not isinstance(law, PhysicalLaw):
        raise TypeError("regime inference requires a PhysicalLaw.")
    if problem.model.model_id != definition.model_id:
        raise ValueError("problem model does not match the regime definition.")
    likelihood = exact_state_space_log_likelihood(problem, method="finite-state")
    backend = likelihood.backend
    if not isinstance(backend, FiniteStateFilterResult):
        raise TypeError(
            "the declared finite-state route returned an incompatible backend."
        )
    smoother = finite_state_backward_smoother(backend)
    viterbi = finite_state_viterbi(backend)
    counts = finite_state_expected_transition_counts(smoother)
    return RegimeFit(
        likelihood=likelihood,
        filter_result=backend,
        smoother_result=smoother,
        viterbi_result=viterbi,
        transition_counts=counts,
        evidence=_evidence(str(data_id), definition.definition_id, "exact-finite-state"),
        data_id=str(data_id),
        definition_id=definition.definition_id,
        law_id=law.law_id,
    )


def summarize_regime_model(fit: RegimeFit, /) -> RegimeResult:
    if not isinstance(fit, RegimeFit):
        raise TypeError("fit must be a RegimeFit.")
    return RegimeResult(
        filtered_probabilities=fit.filter_result.filtered_probabilities,
        smoothed_probabilities=fit.smoother_result.smoothed_probabilities,
        most_likely_states=fit.viterbi_result.state_indices,
        log_likelihood=fit.likelihood.total_log_likelihood,
        valid=fit.likelihood.valid,
        fit=fit,
    )


__all__ = [
    "FinancialStateSpaceDefinition",
    "FinancialStateSpaceFit",
    "FinancialStateSpacePlan",
    "FinancialStateSpaceResult",
    "RegimeDefinition",
    "RegimeFit",
    "RegimeResult",
    "fit_financial_state_space",
    "fit_regime_model",
    "prepare_financial_state_space",
    "summarize_financial_state_space",
    "summarize_regime_model",
]
