#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Longitudinal, targeted, instrumental, survival, and transport estimands."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint


def _probabilities(values: ArrayLike, name: str, /, *, minimum: float) -> Array:
    result = jnp.asarray(values)
    host = np.asarray(result)
    if (
        not np.all(np.isfinite(host))
        or np.any(host < minimum)
        or np.any(host > 1.0 - minimum)
    ):
        raise ValueError(f"{name} violates the declared positivity margin.")
    return result


@dataclass(frozen=True, slots=True)
class MarginalStructuralModelResult:
    coefficients: Array
    stabilized_weights: Array
    residuals: Array
    effective_sample_size: Array
    successful: Array
    result_id: str


def fit_marginal_structural_model(
    design: ArrayLike,
    outcome: ArrayLike,
    treatments: ArrayLike,
    numerator_propensity: ArrayLike,
    denominator_propensity: ArrayLike,
    /,
    *,
    uncensored: ArrayLike | None = None,
    minimum_probability: float = 1.0e-4,
) -> MarginalStructuralModelResult:
    """Fit a linear MSM with exact stabilized treatment/censoring weights."""

    matrix = jnp.asarray(design)
    response = jnp.asarray(outcome)
    treatment = jnp.asarray(treatments, dtype=jnp.bool_)
    if matrix.ndim != 2 or response.shape != (matrix.shape[0],):
        raise ValueError("MSM design and outcome shapes do not align.")
    if treatment.ndim != 2 or treatment.shape[0] != matrix.shape[0]:
        raise ValueError("MSM treatments must have shape (case, time).")
    numerator = _probabilities(
        numerator_propensity, "numerator propensity", minimum=minimum_probability
    )
    denominator = _probabilities(
        denominator_propensity, "denominator propensity", minimum=minimum_probability
    )
    if numerator.shape != treatment.shape or denominator.shape != treatment.shape:
        raise ValueError("MSM propensity arrays must match treatments.")
    selected_numerator = jnp.where(treatment, numerator, 1.0 - numerator)
    selected_denominator = jnp.where(treatment, denominator, 1.0 - denominator)
    log_weights = jnp.sum(
        jnp.log(selected_numerator) - jnp.log(selected_denominator), axis=1
    )
    if uncensored is not None:
        censoring = jnp.asarray(uncensored, dtype=jnp.bool_)
        if censoring.shape != treatment.shape:
            raise ValueError("uncensored must match treatments.")
        log_weights = jnp.where(jnp.all(censoring, axis=1), log_weights, -jnp.inf)
    weights = jnp.exp(log_weights - jnp.max(log_weights))
    normal = matrix.T @ (weights[:, None] * matrix)
    right = matrix.T @ (weights * response)
    coefficients = jnp.linalg.solve(normal, right)
    residuals = response - matrix @ coefficients
    sum_weights = jnp.sum(weights)
    ess = sum_weights**2 / jnp.sum(weights**2)
    successful = (
        jnp.all(jnp.isfinite(coefficients)) & jnp.isfinite(ess) & (ess > matrix.shape[1])
    )
    payload = {
        "kind": "marginal-structural-model",
        "cases": matrix.shape[0],
        "parameters": matrix.shape[1],
        "time_points": treatment.shape[1],
        "minimum_probability": float(minimum_probability),
    }
    return MarginalStructuralModelResult(
        coefficients,
        weights,
        residuals,
        ess,
        successful,
        canonical_fingerprint(payload),
    )


@dataclass(frozen=True, slots=True)
class TMLEResult:
    treatment_effect: Array
    updated_treated: Array
    updated_control: Array
    fluctuation: Array
    influence_curve: Array
    standard_error: Array
    successful: Array


def tmle_ate(
    outcome: ArrayLike,
    treatment: ArrayLike,
    initial_treated: ArrayLike,
    initial_control: ArrayLike,
    propensity: ArrayLike,
    /,
    *,
    minimum_probability: float = 1.0e-4,
) -> TMLEResult:
    """One-step continuous-outcome TMLE for an average treatment effect."""

    outcome_ = jnp.asarray(outcome)
    treatment_ = jnp.asarray(treatment, dtype=jnp.bool_)
    treated = jnp.asarray(initial_treated)
    control = jnp.asarray(initial_control)
    probability = _probabilities(
        propensity, "treatment propensity", minimum=minimum_probability
    )
    if not (
        outcome_.shape
        == treatment_.shape
        == treated.shape
        == control.shape
        == probability.shape
    ):
        raise ValueError("TMLE inputs must be aligned case vectors.")
    observed = jnp.where(treatment_, treated, control)
    clever = jnp.where(treatment_, 1.0 / probability, -1.0 / (1.0 - probability))
    denominator = jnp.sum(clever * clever)
    fluctuation = jnp.sum(clever * (outcome_ - observed)) / denominator
    updated_treated = treated + fluctuation / probability
    updated_control = control - fluctuation / (1.0 - probability)
    effect = jnp.mean(updated_treated - updated_control)
    updated_observed = jnp.where(treatment_, updated_treated, updated_control)
    influence = (
        clever * (outcome_ - updated_observed)
        + updated_treated
        - updated_control
        - effect
    )
    standard_error = jnp.std(influence, ddof=1) / jnp.sqrt(outcome_.size)
    successful = jnp.all(jnp.isfinite(influence)) & jnp.isfinite(standard_error)
    return TMLEResult(
        effect,
        updated_treated,
        updated_control,
        fluctuation,
        influence,
        standard_error,
        successful,
    )


@dataclass(frozen=True, slots=True)
class InstrumentalVariableResult:
    late: Array
    first_stage: Array
    reduced_form: Array
    successful: Array


def binary_instrument_late(
    outcome: ArrayLike,
    treatment: ArrayLike,
    instrument: ArrayLike,
    /,
    *,
    minimum_first_stage: float = 1.0e-6,
) -> InstrumentalVariableResult:
    """Wald LATE for one declared binary instrument under caller-owned assumptions."""

    outcome_ = jnp.asarray(outcome)
    treatment_ = jnp.asarray(treatment)
    instrument_ = jnp.asarray(instrument, dtype=jnp.bool_)
    if outcome_.shape != treatment_.shape or outcome_.shape != instrument_.shape:
        raise ValueError("IV inputs must be aligned vectors.")
    treated_outcome = jnp.mean(outcome_[instrument_])
    control_outcome = jnp.mean(outcome_[~instrument_])
    treated_exposure = jnp.mean(treatment_[instrument_])
    control_exposure = jnp.mean(treatment_[~instrument_])
    reduced = treated_outcome - control_outcome
    first = treated_exposure - control_exposure
    late = reduced / first
    successful = jnp.isfinite(late) & (jnp.abs(first) >= minimum_first_stage)
    return InstrumentalVariableResult(late, first, reduced, successful)


@dataclass(frozen=True, slots=True)
class CompetingRiskResult:
    times: Array
    survival: Array
    cumulative_incidence: Array


def aalen_johansen(
    event_times: ArrayLike,
    event_types: ArrayLike,
    /,
    *,
    cause_count: int,
) -> CompetingRiskResult:
    """Aalen–Johansen survival and cause-specific cumulative incidence."""

    times = np.asarray(event_times, dtype=np.float64)
    types = np.asarray(event_types, dtype=np.int64)
    if times.ndim != 1 or types.shape != times.shape or not np.all(np.isfinite(times)):
        raise ValueError("Competing-risk times/types must be aligned finite vectors.")
    causes = int(cause_count)
    if causes <= 0 or np.any(types < 0) or np.any(types > causes):
        raise ValueError("Event types must use zero for censoring and declared causes.")
    unique = np.unique(times[types > 0])
    survival = 1.0
    incidence = np.zeros((causes,), dtype=np.float64)
    survivals = []
    incidences = []
    for time in unique:
        at_risk = np.count_nonzero(times >= time)
        events = np.asarray(
            [
                np.count_nonzero((times == time) & (types == cause))
                for cause in range(1, causes + 1)
            ]
        )
        incidence += survival * events / at_risk
        survival *= 1.0 - np.sum(events) / at_risk
        survivals.append(survival)
        incidences.append(incidence.copy())
    return CompetingRiskResult(
        jnp.asarray(unique),
        jnp.asarray(survivals),
        jnp.asarray(incidences),
    )


def transportability_weights(
    target_probability: ArrayLike,
    source_probability: ArrayLike,
    /,
    *,
    minimum_probability: float = 1.0e-4,
) -> Array:
    """Return target/source odds weights under an explicit sampling model."""

    target = _probabilities(
        target_probability, "target probability", minimum=minimum_probability
    )
    source = _probabilities(
        source_probability, "source probability", minimum=minimum_probability
    )
    if target.shape != source.shape:
        raise ValueError("Transport probabilities must align.")
    return target * (1.0 - source) / (source * (1.0 - target))


def e_value(risk_ratio: ArrayLike, /) -> Array:
    """Minimum confounder association strength for a harmful risk-ratio estimate."""

    ratio = jnp.asarray(risk_ratio)
    harmful = jnp.where(ratio >= 1.0, ratio, 1.0 / ratio)
    return harmful + jnp.sqrt(harmful * (harmful - 1.0))


def dynamic_regime_value(
    outcome: ArrayLike,
    observed_treatment: ArrayLike,
    recommended_treatment: ArrayLike,
    propensity: ArrayLike,
    /,
    *,
    minimum_probability: float = 1.0e-4,
) -> Array:
    """IPW value of a deterministic longitudinal treatment regime."""

    outcome_ = jnp.asarray(outcome)
    observed = jnp.asarray(observed_treatment, dtype=jnp.bool_)
    recommended = jnp.asarray(recommended_treatment, dtype=jnp.bool_)
    probability = _probabilities(
        propensity, "regime propensity", minimum=minimum_probability
    )
    if observed.shape != recommended.shape or observed.shape != probability.shape:
        raise ValueError("Regime treatment and propensity arrays must align.")
    if outcome_.shape != (observed.shape[0],):
        raise ValueError("Regime outcomes must have one value per case.")
    selected = jnp.where(observed, probability, 1.0 - probability)
    adherent = jnp.all(observed == recommended, axis=1)
    weights = jnp.where(adherent, 1.0 / jnp.prod(selected, axis=1), 0.0)
    return jnp.sum(weights * outcome_) / jnp.sum(weights)


__all__ = [
    "CompetingRiskResult",
    "InstrumentalVariableResult",
    "MarginalStructuralModelResult",
    "TMLEResult",
    "aalen_johansen",
    "binary_instrument_late",
    "dynamic_regime_value",
    "e_value",
    "fit_marginal_structural_model",
    "tmle_ate",
    "transportability_weights",
]
