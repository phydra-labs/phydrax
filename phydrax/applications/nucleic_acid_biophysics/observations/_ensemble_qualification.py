# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Preparation-grouped qualification for the conditional ensemble ladder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....qualification import (
    CampaignRole,
    QualificationEvidence,
    ScientificCampaign,
    ScientificCase,
    ScientificClaimProfile,
)
from ._conditional_mapping import ConditionalMappingFit, ConditionalMutationLaw
from ._ensemble_inference import (
    EnsembleDiagnostics,
    EnsembleSupportComparison,
    FiniteEnsembleFit,
    FiniteStructuralEnsembleModel,
)
from ._mutation_profiles import MutationProfileBatch


ModelLadderLevel = Literal[
    "binary-accessibility", "context", "hierarchical", "finite-mixture"
]
PredictiveStageId = Literal["predictive-calibration", "locked-prediction"]
_SCORE_UNIT_ID = "natural-log-unit-per-observed-event"
_SCORE_AGGREGATION = "independent_unit_macro"
_SCORE_ROLE_BY_STAGE = {
    "predictive-calibration": "interval_calibration",
    "locked-prediction": "locked_evaluation",
}
_SCORE_METRIC_BY_STAGE = {
    "predictive-calibration": "conditional-ensemble-calibration-log-score",
    "locked-prediction": "conditional-ensemble-locked-log-score",
}
_MIXTURE_ADVANTAGE_METRIC_ID = "conditional-ensemble-locked-mixture-log-score-advantage"
_ENSEMBLE_REQUIRED_STAGES = frozenset(
    (
        "source-admission",
        "measurement-calibration",
        "parameter-identifiability",
        "predictive-calibration",
        "locked-prediction",
    )
)


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


class EnsemblePredictiveScoreCriterion(StrictModule, NonTrainableState):
    """Predeclared threshold for one exact grouped predictive-score stage."""

    minimum_independent_unit_macro: float = eqx.field(static=True)
    stage_id: PredictiveStageId = eqx.field(static=True)
    score_role: str = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)
    aggregation: str = eqx.field(static=True)
    criterion_id: str = eqx.field(static=True)

    def __init__(
        self,
        stage_id: PredictiveStageId,
        minimum_independent_unit_macro: float,
        /,
    ):
        if stage_id not in _SCORE_ROLE_BY_STAGE:
            raise ValueError(
                "Predictive score stage must be predictive-calibration or locked-prediction."
            )
        if isinstance(minimum_independent_unit_macro, bool) or not isinstance(
            minimum_independent_unit_macro, Real
        ):
            raise TypeError("Predictive score threshold must be a finite real number.")
        minimum = float(minimum_independent_unit_macro)
        if not np.isfinite(minimum):
            raise ValueError("Predictive score threshold must be finite.")
        self.minimum_independent_unit_macro = minimum
        self.stage_id = stage_id
        self.score_role = _SCORE_ROLE_BY_STAGE[stage_id]
        self.metric_id = _SCORE_METRIC_BY_STAGE[stage_id]
        self.unit_id = _SCORE_UNIT_ID
        self.aggregation = _SCORE_AGGREGATION
        self.criterion_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "conditional-ensemble-predictive-score-criterion",
            "stage_id": self.stage_id,
            "score_role": self.score_role,
            "metric_id": self.metric_id,
            "minimum_independent_unit_macro": self.minimum_independent_unit_macro,
            "unit_id": self.unit_id,
            "aggregation": self.aggregation,
        }

    def to_record(self) -> dict[str, object]:
        """Return the exact threshold and its content address."""
        return {**self._content_record(), "criterion_id": self.criterion_id}

    @classmethod
    def from_record(
        cls, record: Mapping[str, object], /
    ) -> EnsemblePredictiveScoreCriterion:
        """Reconstruct and content-verify a predictive-score threshold."""
        if not isinstance(record, Mapping):
            raise TypeError("Predictive-score criterion record must be a mapping.")
        value = cls(
            str(record["stage_id"]),
            record["minimum_independent_unit_macro"],
        )
        expected = value._content_record()
        if any(record.get(name) != item for name, item in expected.items()):
            raise ValueError("Serialized predictive-score criterion is inconsistent.")
        recorded_id = record.get("criterion_id")
        if recorded_id is not None and str(recorded_id) != value.criterion_id:
            raise ValueError(
                "Serialized predictive-score criterion has an invalid content address."
            )
        return value


class EnsembleMixtureAdvantageCriterion(StrictModule, NonTrainableState):
    """Positive locked-unit advantage required for a finite-mixture selection."""

    minimum_independent_unit_macro: float = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)
    aggregation: str = eqx.field(static=True)
    criterion_id: str = eqx.field(static=True)

    def __init__(self, minimum_independent_unit_macro: float, /):
        if isinstance(minimum_independent_unit_macro, bool) or not isinstance(
            minimum_independent_unit_macro, Real
        ):
            raise TypeError("Mixture-advantage threshold must be a finite real number.")
        minimum = float(minimum_independent_unit_macro)
        if not np.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("Mixture-advantage threshold must be finite and positive.")
        self.minimum_independent_unit_macro = minimum
        self.metric_id = _MIXTURE_ADVANTAGE_METRIC_ID
        self.unit_id = _SCORE_UNIT_ID
        self.aggregation = _SCORE_AGGREGATION
        self.criterion_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "conditional-ensemble-mixture-advantage-criterion",
            "metric_id": self.metric_id,
            "minimum_independent_unit_macro": self.minimum_independent_unit_macro,
            "unit_id": self.unit_id,
            "aggregation": self.aggregation,
        }

    def to_record(self) -> dict[str, object]:
        """Return the exact locked-mixture threshold and its content address."""
        return {**self._content_record(), "criterion_id": self.criterion_id}

    @classmethod
    def from_record(
        cls, record: Mapping[str, object], /
    ) -> EnsembleMixtureAdvantageCriterion:
        """Reconstruct and content-verify a mixture-advantage threshold."""
        if not isinstance(record, Mapping):
            raise TypeError("Mixture-advantage criterion record must be a mapping.")
        value = cls(record["minimum_independent_unit_macro"])
        expected = value._content_record()
        if any(record.get(name) != item for name, item in expected.items()):
            raise ValueError("Serialized mixture-advantage criterion is inconsistent.")
        recorded_id = record.get("criterion_id")
        if recorded_id is not None and str(recorded_id) != value.criterion_id:
            raise ValueError(
                "Serialized mixture-advantage criterion has an invalid content address."
            )
        return value


def _fit_identity(model, fit, campaign: ScientificCampaign, /):
    if (
        isinstance(model, ConditionalMutationLaw)
        and isinstance(fit, ConditionalMappingFit)
    ) or (
        isinstance(model, FiniteStructuralEnsembleModel)
        and isinstance(fit, FiniteEnsembleFit)
    ):
        pass
    else:
        raise TypeError(
            "Posterior uncertainty requires a matching frozen ensemble model and fit."
        )
    if fit.model_id != model.model_id:
        raise ValueError("Frozen fit belongs to a different model.")
    calibration = next(role for role in campaign.roles if role.name == "calibration")
    calibration_mask = model.batch.profile_mask_for_cases(calibration.case_ids)
    if set(model.batch.case_ids) != set(campaign.case_ids):
        raise ValueError("Frozen fit batch does not match the scientific campaign.")
    if not np.array_equal(np.asarray(fit.fit_profile_mask), np.asarray(calibration_mask)):
        raise ValueError("Frozen fit must use exactly the campaign calibration cases.")
    if tuple(fit.source_ids) != tuple(model.batch.source_ids):
        raise ValueError("Frozen fit source identity does not match its model batch.")
    expected_problem = model.posterior_problem(fit_profile_mask=calibration_mask)
    if array_tree_fingerprint(
        fit.optimization.problem.initial_position
    ) != array_tree_fingerprint(expected_problem.initial_position):
        raise ValueError("Frozen fit posterior problem does not match the model.")
    problem_id = canonical_fingerprint(
        {
            "kind": "conditional-ensemble-calibration-posterior-problem",
            "campaign": campaign.campaign_id,
            "model": model.model_id,
            "calibration_cases": calibration.case_ids,
            "initial_position": array_tree_fingerprint(expected_problem.initial_position),
        }
    )
    flat, unravel = ravel_pytree(fit.optimization.parameters)
    fit_id = canonical_fingerprint(
        {
            "kind": "conditional-ensemble-frozen-fit",
            "campaign": campaign.campaign_id,
            "model": model.model_id,
            "calibration_cases": calibration.case_ids,
            "fit_profile_mask": array_tree_fingerprint(fit.fit_profile_mask),
            "posterior_problem": problem_id,
            "parameters": array_tree_fingerprint(fit.optimization.parameters),
            "execution_converged": bool(fit.optimization.converged),
        }
    )
    return calibration.case_ids, flat, unravel, fit_id


class EnsemblePosteriorUncertainty(StrictModule, NonTrainableState):
    """Deterministic Laplace draws from one converged frozen calibration fit."""

    parameter_draws: Array
    raw_parameter_draws: Array
    parameter_covariance: Array
    model_id: str = eqx.field(static=True)
    fit_id: str = eqx.field(static=True)
    calibration_case_ids: tuple[str, ...] = eqx.field(static=True)
    parameter_count: int = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    uncertainty_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: ConditionalMutationLaw | FiniteStructuralEnsembleModel,
        fit: ConditionalMappingFit | FiniteEnsembleFit,
        campaign: ScientificCampaign,
        /,
    ):
        if not isinstance(campaign, ScientificCampaign):
            raise TypeError("campaign must be a ScientificCampaign.")
        cases, fitted, _, fit_id = _fit_identity(model, fit, campaign)
        if not bool(fit.optimization.converged):
            raise ValueError(
                "Posterior uncertainty requires a converged calibration fit."
            )
        raw_position, unravel_raw = ravel_pytree(fit.optimization.position)
        hessian = np.asarray(
            jax.hessian(
                lambda value: fit.optimization.problem.negative_log_density(
                    unravel_raw(value)
                )
            )(raw_position),
            float,
        )
        hessian = 0.5 * (hessian + hessian.T)
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        if (
            np.any(~np.isfinite(eigenvalues))
            or float(np.min(eigenvalues)) <= 1e-10 * scale
        ):
            raise ValueError(
                "Calibration posterior Hessian is not positive definite for Laplace UQ."
            )
        covariance = (eigenvectors / eigenvalues[None, :]) @ eigenvectors.T
        root = (eigenvectors / np.sqrt(eigenvalues)[None, :]) @ eigenvectors.T
        raw = np.asarray(raw_position)
        dimension = raw.size
        offsets = np.sqrt(float(dimension)) * root.T
        raw_draws = np.concatenate((raw[None] + offsets, raw[None] - offsets), axis=0)
        physical_draws = []
        for draw in raw_draws:
            physical = fit.optimization.problem.parameter_space.constrain(
                unravel_raw(jnp.asarray(draw))
            )
            flattened, _ = ravel_pytree(physical)
            physical_draws.append(np.asarray(flattened))
        draws = np.stack(physical_draws)
        if draws.shape[1] != fitted.size or np.any(~np.isfinite(draws)):
            raise ValueError("Laplace parameter draws are outside model support.")
        self.parameter_draws = jnp.asarray(draws)
        self.raw_parameter_draws = jnp.asarray(raw_draws)
        self.parameter_covariance = jnp.asarray(covariance)
        self.model_id = model.model_id
        self.fit_id = fit_id
        self.calibration_case_ids = cases
        self.parameter_count = int(fitted.size)
        self.approximation_id = "laplace-spherical-radial-equal-weight-v1"
        self.uncertainty_id = canonical_fingerprint(
            {
                "kind": "conditional-ensemble-laplace-uncertainty",
                "campaign": campaign.campaign_id,
                "model": model.model_id,
                "fit": fit_id,
                "calibration_cases": cases,
                "approximation": self.approximation_id,
                "raw_parameter_draws": array_tree_fingerprint(raw_draws),
                "hessian": array_tree_fingerprint(hessian),
                "parameter_covariance": array_tree_fingerprint(covariance),
                "parameter_draws": array_tree_fingerprint(draws),
            }
        )


def prepare_conditional_ensemble_campaign(
    batch: MutationProfileBatch,
    roles: Sequence[CampaignRole],
    /,
    *,
    preprocessing_source_ids: Sequence[str] = (),
    criteria_ids: Sequence[str] = (),
) -> ScientificCampaign:
    """Bind admitted cases to caller-declared roles without row-level splitting."""
    if not isinstance(batch, MutationProfileBatch):
        raise TypeError("batch must be a MutationProfileBatch.")
    cases = tuple(
        ScientificCase(
            case.case_id,
            case.independent_unit_id,
            case.construct_id,
            case.condition_id,
            case.preparation_id,
            case.batch_id,
            case.source_manifest_ids,
            case.parent_case_ids,
        )
        for case in batch.cases
    )
    return ScientificCampaign(
        cases,
        roles,
        preprocessing_source_ids=preprocessing_source_ids,
        criteria_ids=criteria_ids,
    )


class GroupedPredictiveScore(StrictModule, NonTrainableState):
    """Content-addressed model score grouped over one exact campaign role."""

    per_case: Array
    per_independent_unit: Array
    pooled_per_observed_event: Array
    independent_unit_macro: Array
    observed_event_count: Array
    valid: Array
    campaign_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    role: str = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    independent_unit_ids: tuple[str, ...] = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)
    aggregation: str = eqx.field(static=True)
    fit_id: str | None = eqx.field(static=True)
    calibration_case_ids: tuple[str, ...] = eqx.field(static=True)
    epistemic_uncertainty_id: str | None = eqx.field(static=True)
    score_id: str = eqx.field(static=True)


def _group_resolved_profile_log_scores(
    batch: MutationProfileBatch,
    campaign: ScientificCampaign,
    role: str,
    per_profile_log_score: ArrayLike,
    /,
    *,
    model_id: str,
    posterior_uncertainty: EnsemblePosteriorUncertainty | None = None,
) -> GroupedPredictiveScore:
    """Aggregate point or posterior-integrated read scores over independent units."""
    if not isinstance(batch, MutationProfileBatch) or not isinstance(
        campaign, ScientificCampaign
    ):
        raise TypeError("Grouped scores require a mutation batch and ScientificCampaign.")
    if set(batch.case_ids) != set(campaign.case_ids):
        raise ValueError("Campaign cases must exactly cover the admitted mutation batch.")
    model = _identifier(model_id, "model_id")
    role_record = next((value for value in campaign.roles if value.name == role), None)
    if role_record is None:
        raise ValueError("Unknown scientific campaign role.")
    role_cases = role_record.case_ids
    if not role_cases:
        raise ValueError("Cannot score an empty scientific campaign role.")
    raw_scores = jnp.asarray(per_profile_log_score)
    if posterior_uncertainty is None:
        uncertainty = None
        fit_id = None
        calibration_case_ids = ()
        if raw_scores.shape != (batch.profile_count,):
            raise ValueError(
                "Point per_profile_log_score must contain one value per profile."
            )
        scores = raw_scores
    else:
        if not isinstance(posterior_uncertainty, EnsemblePosteriorUncertainty):
            raise TypeError(
                "posterior_uncertainty must be an EnsemblePosteriorUncertainty."
            )
        calibration = next(
            value for value in campaign.roles if value.name == "calibration"
        )
        if (
            posterior_uncertainty.model_id != model
            or posterior_uncertainty.calibration_case_ids != calibration.case_ids
        ):
            raise ValueError(
                "Posterior uncertainty must bind this model and exact calibration role."
            )
        expected = (
            posterior_uncertainty.parameter_draws.shape[0],
            batch.profile_count,
        )
        if raw_scores.shape != expected:
            raise ValueError(
                "Posterior predictive log scores require one profile score per "
                "parameter draw."
            )
        maximum = jnp.max(raw_scores, axis=0)
        fit_id = posterior_uncertainty.fit_id
        calibration_case_ids = posterior_uncertainty.calibration_case_ids
        scores = maximum + jnp.log(
            jnp.mean(jnp.exp(raw_scores - maximum[None, :]), axis=0)
        )
        uncertainty = posterior_uncertainty.uncertainty_id
    campaign_cases = {case.case_id: case for case in campaign.cases}
    event_count = jnp.sum(
        (batch.observed_mask & batch.analysis_mask[:, None]).astype(jnp.int32), axis=-1
    )
    per_case_values, per_case_events = [], []
    for case_id in role_cases:
        rows = batch.case_index == batch.case_ids.index(case_id)
        count = jnp.sum(jnp.where(rows, event_count, 0))
        total = jnp.sum(jnp.where(rows, scores, 0.0))
        per_case_values.append(jnp.where(count > 0, total / count, jnp.nan))
        per_case_events.append(count)
    case_scores = jnp.stack(tuple(per_case_values))
    case_events = jnp.stack(tuple(per_case_events))
    unit_ids = tuple(
        dict.fromkeys(
            campaign_cases[case_id].independent_unit_id for case_id in role_cases
        )
    )
    unit_scores = []
    for unit_id in unit_ids:
        selected = jnp.asarray(
            tuple(
                campaign_cases[case_id].independent_unit_id == unit_id
                for case_id in role_cases
            )
        )
        total_events = jnp.sum(jnp.where(selected, case_events, 0))
        total_score = jnp.sum(jnp.where(selected, case_scores * case_events, 0.0))
        unit_scores.append(
            jnp.where(total_events > 0, total_score / total_events, jnp.nan)
        )
    grouped = jnp.stack(tuple(unit_scores))
    total_events = jnp.sum(case_events)
    pooled = jnp.where(
        total_events > 0,
        jnp.sum(jnp.where(jnp.isfinite(case_scores), case_scores * case_events, 0.0))
        / total_events,
        jnp.nan,
    )
    macro = jnp.where(jnp.all(jnp.isfinite(grouped)), jnp.mean(grouped), jnp.nan)
    valid = jnp.isfinite(pooled) & jnp.isfinite(macro)
    score_id = canonical_fingerprint(
        {
            "kind": "grouped-mutation-predictive-score",
            "batch": batch.batch_fingerprint,
            "campaign": campaign.campaign_id,
            "model": model,
            "role": role,
            "cases": role_cases,
            "independent_units": unit_ids,
            "unit_id": _SCORE_UNIT_ID,
            "aggregation": _SCORE_AGGREGATION,
            "fit": fit_id,
            "calibration_cases": calibration_case_ids,
            "within_unit_aggregation": "per-observed-event",
            "epistemic_uncertainty": uncertainty,
            "values": array_tree_fingerprint(
                (case_scores, case_events, grouped, pooled, macro)
            ),
        }
    )
    return GroupedPredictiveScore(
        case_scores,
        grouped,
        pooled,
        macro,
        total_events,
        valid,
        campaign.campaign_id,
        model,
        role,
        role_cases,
        unit_ids,
        _SCORE_UNIT_ID,
        _SCORE_AGGREGATION,
        fit_id,
        calibration_case_ids,
        uncertainty,
        score_id,
    )


def group_profile_log_scores(
    batch: MutationProfileBatch,
    campaign: ScientificCampaign,
    role: str,
    per_profile_log_score: ArrayLike,
    /,
    *,
    model_id: str,
) -> GroupedPredictiveScore:
    """Group point scores, which remain ineligible for predictive-stage passage."""
    return _group_resolved_profile_log_scores(
        batch,
        campaign,
        role,
        per_profile_log_score,
        model_id=model_id,
    )


def group_posterior_predictive_log_scores(
    batch: MutationProfileBatch,
    campaign: ScientificCampaign,
    role: str,
    model: ConditionalMutationLaw | FiniteStructuralEnsembleModel,
    posterior_uncertainty: EnsemblePosteriorUncertainty,
    /,
) -> GroupedPredictiveScore:
    """Evaluate every frozen parameter draw and group its log predictive density."""
    if role not in ("interval_calibration", "locked_evaluation"):
        raise ValueError(
            "Posterior predictive scores require interval_calibration or "
            "locked_evaluation role."
        )
    if not isinstance(
        model, (ConditionalMutationLaw, FiniteStructuralEnsembleModel)
    ) or not isinstance(posterior_uncertainty, EnsemblePosteriorUncertainty):
        raise TypeError(
            "Posterior predictive grouping requires a frozen ensemble model and "
            "EnsemblePosteriorUncertainty."
        )
    if (
        model.batch.batch_fingerprint != batch.batch_fingerprint
        or posterior_uncertainty.model_id != model.model_id
    ):
        raise ValueError(
            "Posterior predictive model, uncertainty, and batch identities differ."
        )
    calibration = next(value for value in campaign.roles if value.name == "calibration")
    if posterior_uncertainty.calibration_case_ids != calibration.case_ids:
        raise ValueError(
            "Posterior uncertainty does not bind the exact campaign calibration cases."
        )
    if isinstance(model, ConditionalMutationLaw):
        if posterior_uncertainty.parameter_count != len(model.parameter_names):
            raise ValueError(
                "Posterior draw width does not match the conditional mapping model."
            )
        parameters = tuple(posterior_uncertainty.parameter_draws)
    else:
        calibration_mask = batch.profile_mask_for_cases(calibration.case_ids)
        problem = model.posterior_problem(fit_profile_mask=calibration_mask)
        template = problem.parameter_space.constrain(problem.initial_position)
        flat, unravel = ravel_pytree(template)
        if posterior_uncertainty.parameter_count != flat.size:
            raise ValueError(
                "Posterior draw width does not match the finite ensemble model."
            )
        parameters = tuple(
            unravel(draw) for draw in posterior_uncertainty.parameter_draws
        )
    draw_scores = jnp.stack(
        tuple(model.per_profile_log_likelihood(value) for value in parameters)
    )
    return _group_resolved_profile_log_scores(
        batch,
        campaign,
        role,
        draw_scores,
        model_id=model.model_id,
        posterior_uncertainty=posterior_uncertainty,
    )


class ModelLadderEvaluation(StrictModule, NonTrainableState):
    model_selection: GroupedPredictiveScore
    predictive_calibration: GroupedPredictiveScore | None
    locked_evaluation: GroupedPredictiveScore
    posterior_uncertainty: EnsemblePosteriorUncertainty | None
    execution_valid: Array
    model_id: str = eqx.field(static=True)
    level: ModelLadderLevel = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)
    derivation_source_case_ids: tuple[str, ...] = eqx.field(static=True)
    derivation_parent_case_ids: tuple[str, ...] = eqx.field(static=True)
    support_id: str | None = eqx.field(static=True)
    hypothesis_id: str | None = eqx.field(static=True)
    derivation_id: str = eqx.field(static=True)
    epistemic_uncertainty_id: str | None = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)

    def __init__(
        self,
        model_id: str,
        level: ModelLadderLevel,
        model_selection: GroupedPredictiveScore,
        locked_evaluation: GroupedPredictiveScore,
        *,
        predictive_calibration: GroupedPredictiveScore | None = None,
        derivation_source_case_ids: tuple[str, ...],
        derivation_parent_case_ids: tuple[str, ...] = (),
        support_id: str | None = None,
        hypothesis_id: str | None = None,
        derivation_id: str | None = None,
        posterior_uncertainty: EnsemblePosteriorUncertainty | None = None,
        execution_valid: bool | ArrayLike,
    ):
        model = _identifier(model_id, "model_id")
        if level not in (
            "binary-accessibility",
            "context",
            "hierarchical",
            "finite-mixture",
        ):
            raise ValueError("Unknown model-ladder level.")
        scores = tuple(
            score
            for score in (
                model_selection,
                predictive_calibration,
                locked_evaluation,
            )
            if score is not None
        )
        if any(not isinstance(score, GroupedPredictiveScore) for score in scores):
            raise TypeError("Model-ladder scores must be GroupedPredictiveScore values.")
        if (
            model_selection.role != "model_selection"
            or locked_evaluation.role != "locked_evaluation"
            or (
                predictive_calibration is not None
                and predictive_calibration.role != "interval_calibration"
            )
        ):
            raise ValueError(
                "Ladder scores must come from model_selection, interval_calibration, "
                "and locked_evaluation roles."
            )
        if len({score.campaign_id for score in scores}) != 1:
            raise ValueError("Ladder scores must belong to one campaign.")
        if any(score.model_id != model for score in scores):
            raise ValueError("Every ladder score must be bound to the evaluated model.")
        sources = tuple(derivation_source_case_ids)
        parents = tuple(derivation_parent_case_ids)
        if (
            not sources
            or len(set(sources)) != len(sources)
            or len(set(parents)) != len(parents)
            or any(
                not isinstance(value, str) or not value or value != value.strip()
                for value in (*sources, *parents)
            )
        ):
            raise ValueError(
                "Model evaluation requires canonical derivation source and parent cases."
            )
        support = None if support_id is None else _identifier(support_id, "support_id")
        hypothesis = (
            None if hypothesis_id is None else _identifier(hypothesis_id, "hypothesis_id")
        )
        if level == "finite-mixture" and (support is None or hypothesis is None):
            raise ValueError(
                "Finite-mixture evaluation requires exact support and hypothesis IDs."
            )
        if level != "finite-mixture" and (support is not None or hypothesis is not None):
            raise ValueError(
                "Only finite-mixture evaluation accepts structural support identity."
            )
        valid = jnp.asarray(execution_valid)
        if valid.shape != () or valid.dtype != bool:
            raise TypeError(
                "execution_valid must be one boolean model-execution outcome."
            )
        self.model_id, self.level = model, level
        self.campaign_id = model_selection.campaign_id
        self.model_selection = model_selection
        self.predictive_calibration = predictive_calibration
        self.locked_evaluation = locked_evaluation
        self.execution_valid = valid
        self.derivation_source_case_ids = sources
        self.derivation_parent_case_ids = parents
        self.support_id = support
        self.hypothesis_id = hypothesis
        self.derivation_id = (
            canonical_fingerprint(
                {
                    "kind": "conditional-ensemble-model-derivation",
                    "model": model,
                    "support": support,
                    "hypothesis": hypothesis,
                    "source_cases": sources,
                    "parent_cases": parents,
                }
            )
            if derivation_id is None
            else _identifier(derivation_id, "derivation_id")
        )
        if posterior_uncertainty is not None and not isinstance(
            posterior_uncertainty, EnsemblePosteriorUncertainty
        ):
            raise TypeError(
                "posterior_uncertainty must be an EnsemblePosteriorUncertainty."
            )
        if posterior_uncertainty is not None and posterior_uncertainty.model_id != model:
            raise ValueError("Posterior uncertainty belongs to a different model.")
        self.posterior_uncertainty = posterior_uncertainty
        self.epistemic_uncertainty_id = (
            None
            if posterior_uncertainty is None
            else posterior_uncertainty.uncertainty_id
        )
        if any(
            score.epistemic_uncertainty_id != self.epistemic_uncertainty_id
            for score in (predictive_calibration, locked_evaluation)
            if score is not None
        ):
            raise ValueError(
                "Predictive scores must integrate the evaluated posterior uncertainty."
            )
        if posterior_uncertainty is not None and any(
            score.fit_id != posterior_uncertainty.fit_id
            or score.calibration_case_ids != posterior_uncertainty.calibration_case_ids
            for score in (predictive_calibration, locked_evaluation)
            if score is not None
        ):
            raise ValueError(
                "Predictive scores must bind the posterior fit and calibration cases."
            )
        self.evaluation_id = canonical_fingerprint(
            {
                "kind": "conditional-ensemble-model-ladder-evaluation",
                "campaign": self.campaign_id,
                "model": model,
                "level": level,
                "model_selection_score": model_selection.score_id,
                "predictive_calibration_score": (
                    None
                    if predictive_calibration is None
                    else predictive_calibration.score_id
                ),
                "locked_evaluation_score": locked_evaluation.score_id,
                "execution_valid": bool(valid),
                "derivation": self.derivation_id,
                "epistemic_uncertainty": self.epistemic_uncertainty_id,
            }
        )


class EnsembleWorkflowAssessment(StrictModule, NonTrainableState):
    model_selection_scores: Array
    predictive_calibration_scores: Array
    locked_scores: Array
    selection_candidates: Array
    selected_index: Array
    model_selection_unique: Array
    locked_diagnostics_valid: Array
    mixture_advantage: Array
    execution_valid: Array
    unique_state_interpretation_supported: Array
    campaign_id: str = eqx.field(static=True)
    workflow_id: str = eqx.field(static=True)
    model_ids: tuple[str, ...] = eqx.field(static=True)
    levels: tuple[ModelLadderLevel, ...] = eqx.field(static=True)
    support_ids: tuple[str | None, ...] = eqx.field(static=True)
    hypothesis_ids: tuple[str | None, ...] = eqx.field(static=True)
    evaluation_ids: tuple[str, ...] = eqx.field(static=True)
    model_selection_score_ids: tuple[str, ...] = eqx.field(static=True)
    predictive_calibration_score_ids: tuple[str | None, ...] = eqx.field(static=True)
    locked_diagnostics_id: str = eqx.field(static=True)
    locked_score_ids: tuple[str, ...] = eqx.field(static=True)
    epistemic_uncertainty_ids: tuple[str | None, ...] = eqx.field(static=True)
    diagnostics_id: str = eqx.field(static=True)
    assessment_id: str = eqx.field(static=True)

    @property
    def selected_model_id(self) -> str | None:
        if not bool(self.model_selection_unique):
            return None
        return self.model_ids[int(self.selected_index)]

    @property
    def selected_predictive_calibration_score_id(self) -> str | None:
        if not bool(self.model_selection_unique):
            return None
        return self.predictive_calibration_score_ids[int(self.selected_index)]

    @property
    def selected_locked_score_id(self) -> str | None:
        if not bool(self.model_selection_unique):
            return None
        return self.locked_score_ids[int(self.selected_index)]

    @property
    def selected_epistemic_uncertainty_id(self) -> str | None:
        if not bool(self.model_selection_unique):
            return None
        return self.epistemic_uncertainty_ids[int(self.selected_index)]


class ConditionalEnsembleQualificationWorkflow(StrictModule, NonTrainableState):
    """Frozen model selection plus locked evaluation and nonidentifiability routing."""

    batch: MutationProfileBatch
    campaign: ScientificCampaign = eqx.field(static=True)
    model_selection_tolerance: float = eqx.field(static=True)
    workflow_id: str = eqx.field(static=True)

    def __init__(
        self,
        batch: MutationProfileBatch,
        campaign: ScientificCampaign,
        /,
        *,
        model_selection_tolerance: float,
    ):
        if not isinstance(batch, MutationProfileBatch) or not isinstance(
            campaign, ScientificCampaign
        ):
            raise TypeError("Workflow requires a mutation batch and ScientificCampaign.")
        if set(batch.case_ids) != set(campaign.case_ids):
            raise ValueError(
                "Campaign cases must exactly cover the admitted mutation batch."
            )
        tolerance = float(model_selection_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Model-selection tolerance must be finite and nonnegative.")
        self.batch, self.campaign = batch, campaign
        self.model_selection_tolerance = tolerance
        self.workflow_id = canonical_fingerprint(
            {
                "kind": "conditional-ensemble-qualification-workflow",
                "batch": batch.batch_fingerprint,
                "campaign": campaign.campaign_id,
                "model_selection_tolerance": tolerance,
                "split_unit": "preparation-and-construct",
            }
        )

    def profile_mask(self, role: str, /) -> Array:
        """Return every admitted read row in one frozen campaign role."""
        record = next(
            (value for value in self.campaign.roles if value.name == role), None
        )
        if record is None or not record.case_ids:
            raise ValueError("The requested campaign role is absent or empty.")
        return self.batch.profile_mask_for_cases(record.case_ids)

    def assess(
        self,
        evaluations: Sequence[ModelLadderEvaluation],
        /,
        *,
        support_comparison: EnsembleSupportComparison,
        ensemble_diagnostics: EnsembleDiagnostics,
        locked_ensemble_diagnostics: EnsembleDiagnostics,
    ) -> EnsembleWorkflowAssessment:
        values = tuple(evaluations)
        if any(not isinstance(value, ModelLadderEvaluation) for value in values):
            raise TypeError("evaluations must contain ModelLadderEvaluation values.")
        if len(values) != 4 or tuple(value.level for value in values) != (
            "binary-accessibility",
            "context",
            "hierarchical",
            "finite-mixture",
        ):
            raise ValueError(
                "Qualification requires the complete ordered four-model ladder."
            )
        if len({value.model_id for value in values}) != len(values):
            raise ValueError("Model-ladder IDs must be unique.")
        if any(value.campaign_id != self.campaign.campaign_id for value in values):
            raise ValueError("Every model score must belong to this campaign.")
        if (
            not isinstance(support_comparison, EnsembleSupportComparison)
            or not isinstance(ensemble_diagnostics, EnsembleDiagnostics)
            or not isinstance(locked_ensemble_diagnostics, EnsembleDiagnostics)
        ):
            raise TypeError(
                "Support comparison plus calibration and locked finite-ensemble "
                "diagnostics are required."
            )
        finite_evaluation = values[3]
        if (
            ensemble_diagnostics.model_id != finite_evaluation.model_id
            or locked_ensemble_diagnostics.model_id != finite_evaluation.model_id
        ):
            raise ValueError(
                "Calibration and locked diagnostics must belong to the evaluated "
                "finite-mixture model."
            )
        if not np.array_equal(
            np.asarray(ensemble_diagnostics.profile_mask),
            np.asarray(self.profile_mask("calibration")),
        ):
            raise ValueError("Fit diagnostics must use exactly the calibration role.")
        if not np.array_equal(
            np.asarray(locked_ensemble_diagnostics.profile_mask),
            np.asarray(self.profile_mask("locked_evaluation")),
        ):
            raise ValueError(
                "Locked diagnostics must use exactly the locked-evaluation role."
            )

        permitted_derivations = set(self.campaign.preprocessing_source_ids)
        cases_by_id = {case.case_id: case for case in self.campaign.cases}
        for value in values:
            sources = set(value.derivation_source_case_ids)
            parents = set(value.derivation_parent_case_ids)
            if not sources.issubset(permitted_derivations):
                raise ValueError(
                    "Model derivation sources must be frozen calibration/model-selection "
                    "preprocessing cases."
                )
            expected_parents = {
                parent
                for source in sources
                for parent in cases_by_id[source].parent_case_ids
            }
            if parents != expected_parents:
                raise ValueError(
                    "Model derivation parent-case lineage does not match the campaign."
                )
        calibration_cases = next(
            role.case_ids for role in self.campaign.roles if role.name == "calibration"
        )
        if any(
            value.posterior_uncertainty is not None
            and value.posterior_uncertainty.calibration_case_ids != calibration_cases
            for value in values
        ):
            raise ValueError(
                "Posterior uncertainty calibration cases do not match the campaign."
            )

        if support_comparison.campaign_id != self.campaign.campaign_id:
            raise ValueError("Support comparison belongs to a different campaign.")
        matching_supports = tuple(
            index
            for index, support_id in enumerate(support_comparison.support_ids)
            if (
                support_id == finite_evaluation.support_id
                and support_comparison.model_ids[index] == finite_evaluation.model_id
                and support_comparison.hypothesis_ids[index]
                == finite_evaluation.hypothesis_id
                and support_comparison.derivation_ids[index]
                == finite_evaluation.derivation_id
                and support_comparison.score_ids[index]
                == finite_evaluation.model_selection.score_id
            )
        )
        if len(matching_supports) != 1:
            raise ValueError(
                "Support comparison does not bind the finite evaluation's exact "
                "support, model, hypothesis, derivation, and score."
            )
        if (
            bool(support_comparison.unique_best)
            and int(support_comparison.best_support_index) != matching_supports[0]
        ):
            raise ValueError(
                "Winning support comparison entry does not equal the finite evaluation."
            )

        selection = jnp.stack(
            tuple(value.model_selection.independent_unit_macro for value in values)
        )
        calibration = jnp.stack(
            tuple(
                (
                    jnp.asarray(jnp.nan, dtype=selection.dtype)
                    if value.predictive_calibration is None
                    else value.predictive_calibration.independent_unit_macro
                )
                for value in values
            )
        )
        locked = jnp.stack(
            tuple(value.locked_evaluation.independent_unit_macro for value in values)
        )
        valid_selection = jnp.isfinite(selection)
        safe_selection = jnp.where(valid_selection, selection, -jnp.inf)
        best = jnp.argmax(safe_selection)
        any_valid = jnp.any(valid_selection)
        candidates = (
            valid_selection
            & any_valid
            & (safe_selection[best] - safe_selection <= self.model_selection_tolerance)
        )
        ladder_unique = any_valid & (jnp.sum(candidates.astype(jnp.int32)) == 1)
        selection_unique = ladder_unique & ((best != 3) | support_comparison.unique_best)
        finite_locked = locked[3]
        non_mixture_best = jnp.max(
            jnp.where(jnp.isfinite(locked[:3]), locked[:3], -jnp.inf)
        )
        advantage = jnp.where(
            jnp.isfinite(finite_locked) & jnp.isfinite(non_mixture_best),
            finite_locked - non_mixture_best,
            jnp.nan,
        )
        locked_diagnostics_valid = (
            locked_ensemble_diagnostics.numerical_valid
            & locked_ensemble_diagnostics.support_complete
            & locked_ensemble_diagnostics.residual_assumption_supported
        )
        execution_valid = (
            jnp.all(jnp.stack(tuple(value.execution_valid for value in values)))
            & jnp.all(jnp.stack(tuple(value.model_selection.valid for value in values)))
            & jnp.all(jnp.stack(tuple(value.locked_evaluation.valid for value in values)))
            & ensemble_diagnostics.numerical_valid
            & locked_ensemble_diagnostics.numerical_valid
            & jnp.any(support_comparison.valid_support)
        )
        unique_state = (
            execution_valid
            & locked_diagnostics_valid
            & selection_unique
            & (best == 3)
            & ensemble_diagnostics.unique_state_interpretation_supported
        )
        assessment_id = canonical_fingerprint(
            {
                "kind": "conditional-ensemble-workflow-assessment",
                "workflow": self.workflow_id,
                "evaluations": [value.evaluation_id for value in values],
                "support_comparison": support_comparison.comparison_id,
                "ensemble_diagnostics": ensemble_diagnostics.diagnostics_id,
                "locked_ensemble_diagnostics": (
                    locked_ensemble_diagnostics.diagnostics_id
                ),
            }
        )
        return EnsembleWorkflowAssessment(
            selection,
            calibration,
            locked,
            candidates,
            jnp.where(selection_unique, best, -1).astype(jnp.int32),
            selection_unique,
            locked_diagnostics_valid,
            advantage,
            execution_valid,
            unique_state,
            self.campaign.campaign_id,
            self.workflow_id,
            tuple(value.model_id for value in values),
            tuple(value.level for value in values),
            tuple(value.support_id for value in values),
            tuple(value.hypothesis_id for value in values),
            tuple(value.evaluation_id for value in values),
            tuple(value.model_selection.score_id for value in values),
            tuple(
                (
                    None
                    if value.predictive_calibration is None
                    else value.predictive_calibration.score_id
                )
                for value in values
            ),
            locked_ensemble_diagnostics.diagnostics_id,
            tuple(value.locked_evaluation.score_id for value in values),
            tuple(value.epistemic_uncertainty_id for value in values),
            ensemble_diagnostics.diagnostics_id,
            assessment_id,
        )

    def stage_evidence(
        self,
        stage_id: str,
        assessment: EnsembleWorkflowAssessment,
        /,
        *,
        predictive_criterion: EnsemblePredictiveScoreCriterion | None = None,
        mixture_advantage_criterion: EnsembleMixtureAdvantageCriterion | None = None,
        source_evidence: QualificationEvidence | None = None,
        build_id: str,
        environment_id: str,
        backend: str,
        topology: str,
        precision: str,
        reduction: str,
        replay_id: str,
        reviewer_id: str,
        issued_at: int,
        expires_at: int,
    ) -> QualificationEvidence:
        """Derive stage outcomes from the frozen assessment and exact score gate."""
        if not isinstance(assessment, EnsembleWorkflowAssessment):
            raise TypeError("assessment must be an EnsembleWorkflowAssessment.")
        if (
            assessment.campaign_id != self.campaign.campaign_id
            or assessment.workflow_id != self.workflow_id
        ):
            raise ValueError("Assessment identity does not belong to this workflow.")
        if predictive_criterion is not None and not isinstance(
            predictive_criterion, EnsemblePredictiveScoreCriterion
        ):
            raise TypeError(
                "predictive_criterion must be an EnsemblePredictiveScoreCriterion."
            )
        if mixture_advantage_criterion is not None and not isinstance(
            mixture_advantage_criterion, EnsembleMixtureAdvantageCriterion
        ):
            raise TypeError(
                "mixture_advantage_criterion must be an "
                "EnsembleMixtureAdvantageCriterion."
            )
        if mixture_advantage_criterion is not None and stage_id != "locked-prediction":
            raise ValueError(
                "Mixture-advantage criteria apply only to locked prediction."
            )
        frozen_criteria = set(self.campaign.criteria_ids)
        if (
            predictive_criterion is not None
            and predictive_criterion.criterion_id not in frozen_criteria
        ):
            raise ValueError("Predictive score criterion was not frozen in the campaign.")
        if (
            mixture_advantage_criterion is not None
            and mixture_advantage_criterion.criterion_id not in frozen_criteria
        ):
            raise ValueError(
                "Mixture-advantage criterion was not frozen in the campaign."
            )
        if source_evidence is not None and not isinstance(
            source_evidence, QualificationEvidence
        ):
            raise TypeError("source_evidence must be a QualificationEvidence record.")
        if source_evidence is not None and stage_id != "source-admission":
            raise ValueError("Reviewed source evidence applies only to source admission.")
        subjects = [
            self.campaign.campaign_id,
            self.workflow_id,
            assessment.assessment_id,
        ]
        criteria_ids = [stage_id]
        selected_model_id = assessment.selected_model_id
        if stage_id == "source-admission":
            if predictive_criterion is not None:
                raise ValueError(
                    "Source admission does not accept a predictive criterion."
                )
            synthetic_sources = tuple(
                source.manifest_id
                for source in self.batch.sources
                if "synthetic" in source.license_id.casefold()
            )
            if source_evidence is None or synthetic_sources:
                outcome = "inconclusive"
                reason = (
                    "source admission requires separately reviewed experimental "
                    "provenance and rights"
                    if not synthetic_sources
                    else "synthetic source manifests cannot support scientific admission"
                )
            else:
                required_criteria = {
                    "source-admission",
                    "experimental-source-provenance",
                    "source-rights-review",
                }
                required_subjects = {
                    self.campaign.campaign_id,
                    self.batch.batch_fingerprint,
                    *self.batch.source_ids,
                }
                if (
                    source_evidence.evidence_kind != "scientific"
                    or not required_criteria.issubset(source_evidence.criteria_ids)
                    or not required_subjects.issubset(source_evidence.subject_ids)
                    or set(source_evidence.raw_artifact_ids) != set(self.batch.source_ids)
                ):
                    raise ValueError(
                        "Reviewed source evidence has mismatched experimental "
                        "provenance, rights, campaign, batch, or artifact identity."
                    )
                return source_evidence
        elif stage_id == "numerical-validity":
            if predictive_criterion is not None:
                raise ValueError(
                    "Numerical validity does not accept a predictive criterion."
                )
            outcome = "passed" if bool(assessment.execution_valid) else "failed"
            reason = (
                "finite complete ladder execution"
                if outcome == "passed"
                else "one or more ladder executions are invalid"
            )
        elif stage_id == "parameter-identifiability":
            if predictive_criterion is not None:
                raise ValueError(
                    "Parameter identifiability does not accept a predictive criterion."
                )
            outcome = (
                "passed"
                if bool(assessment.unique_state_interpretation_supported)
                else "inconclusive"
            )
            reason = (
                "unique supported finite-state interpretation"
                if outcome == "passed"
                else "equivalent, missing, or locally nonidentifiable state support remains"
            )
            if selected_model_id is not None:
                subjects.append(selected_model_id)
        elif stage_id in _SCORE_ROLE_BY_STAGE:
            if (
                predictive_criterion is not None
                and predictive_criterion.stage_id != stage_id
            ):
                raise ValueError(
                    "Predictive criterion stage does not match requested stage evidence."
                )
            if predictive_criterion is not None:
                subjects.append(predictive_criterion.criterion_id)
                criteria_ids.append(predictive_criterion.metric_id)
            if mixture_advantage_criterion is not None:
                subjects.append(mixture_advantage_criterion.criterion_id)
                criteria_ids.append(mixture_advantage_criterion.metric_id)
            if selected_model_id is not None:
                subjects.append(selected_model_id)
            uncertainty_id = assessment.selected_epistemic_uncertainty_id
            if uncertainty_id is not None:
                subjects.append(uncertainty_id)
            if not bool(assessment.execution_valid):
                outcome, reason = (
                    "failed",
                    "predictive score is unusable because ladder execution is invalid",
                )
            elif selected_model_id is None:
                outcome, reason = (
                    "inconclusive",
                    "no model and structural support were uniquely frozen",
                )
            elif uncertainty_id is None:
                outcome, reason = (
                    "inconclusive",
                    "selected model has no posterior parameter uncertainty record",
                )
            elif stage_id == "locked-prediction" and not bool(
                assessment.locked_diagnostics_valid
            ):
                outcome, reason = (
                    "failed",
                    "locked preparations violate finite-mixture support or "
                    "residual-assumption diagnostics",
                )
            elif predictive_criterion is None:
                outcome, reason = (
                    "inconclusive",
                    "no content-addressed predictive-score threshold was supplied",
                )
            else:
                selected_index = int(assessment.selected_index)
                if stage_id == "predictive-calibration":
                    score_id = assessment.selected_predictive_calibration_score_id
                    score = assessment.predictive_calibration_scores[selected_index]
                else:
                    score_id = assessment.selected_locked_score_id
                    score = assessment.locked_scores[selected_index]
                if score_id is None or not bool(jnp.isfinite(score)):
                    outcome, reason = (
                        "inconclusive",
                        f"selected model has no valid {predictive_criterion.score_role} score",
                    )
                else:
                    subjects.append(score_id)
                    score_value = float(score)
                    score_reason = (
                        f"{predictive_criterion.metric_id}={score_value!r};"
                        f"minimum={predictive_criterion.minimum_independent_unit_macro!r};"
                        f"score_id={score_id}"
                    )
                    if score_value < predictive_criterion.minimum_independent_unit_macro:
                        outcome = "failed"
                        reason = score_reason
                    elif (
                        stage_id == "locked-prediction"
                        and assessment.levels[selected_index] == "finite-mixture"
                    ):
                        if mixture_advantage_criterion is None:
                            outcome, reason = (
                                "inconclusive",
                                "selected finite mixture has no content-addressed "
                                "locked mixture-advantage threshold",
                            )
                        elif not bool(jnp.isfinite(assessment.mixture_advantage)):
                            outcome, reason = (
                                "inconclusive",
                                "selected finite mixture has no finite locked "
                                "independent-unit advantage",
                            )
                        else:
                            advantage = float(assessment.mixture_advantage)
                            outcome = (
                                "passed"
                                if advantage
                                >= mixture_advantage_criterion.minimum_independent_unit_macro
                                else "failed"
                            )
                            reason = (
                                f"{score_reason};"
                                f"{mixture_advantage_criterion.metric_id}={advantage!r};"
                                "minimum="
                                f"{mixture_advantage_criterion.minimum_independent_unit_macro!r}"
                            )
                    else:
                        outcome, reason = "passed", score_reason
        else:
            raise ValueError(
                "This workflow only emits source, numerical, identifiability, and predictive stage evidence."
            )
        return QualificationEvidence(
            "scientific",
            outcome,
            tuple(dict.fromkeys(subjects)),
            build_id=build_id,
            environment_id=environment_id,
            backend=backend,
            topology=topology,
            precision=precision,
            reduction=reduction,
            replay_id=replay_id,
            criteria_ids=tuple(criteria_ids),
            raw_artifact_ids=self.batch.source_ids,
            reviewer_id=reviewer_id,
            issued_at=issued_at,
            expires_at=expires_at,
            reason=reason,
            requalification_triggers=(
                "model-parameters",
                "observation-law",
                "preprocessing",
                "source-artifact",
                "operating-domain",
                "predictive-score-threshold",
                "mixture-advantage-threshold",
            ),
            campaign_start_record_ids=(),
            campaign_observation_record_ids=(),
        )

    def evaluate_claim(
        self,
        profile: ScientificClaimProfile,
        metric_values: Mapping[str, float],
        stage_evidence: Sequence[QualificationEvidence],
        /,
        *,
        assessment: EnsembleWorkflowAssessment,
        metric_units: Mapping[str, str],
        metric_aggregations: Mapping[str, str],
        build_id: str,
        environment_id: str,
        backend: str,
        topology: str,
        precision: str,
        reduction: str,
        replay_id: str,
        reviewer_id: str,
        issued_at: int,
        expires_at: int,
    ) -> QualificationEvidence:
        if not isinstance(assessment, EnsembleWorkflowAssessment):
            raise TypeError("assessment must be an EnsembleWorkflowAssessment.")
        if (
            assessment.campaign_id != self.campaign.campaign_id
            or assessment.workflow_id != self.workflow_id
        ):
            raise ValueError("Assessment identity does not belong to this workflow.")
        if (
            not isinstance(profile, ScientificClaimProfile)
            or profile.campaign_id != self.campaign.campaign_id
        ):
            raise ValueError(
                "Scientific claim profile must be bound to this exact campaign."
            )
        if profile.frozen_criteria_ids != self.campaign.criteria_ids:
            raise ValueError(
                "Scientific claim frozen criteria must exactly match the campaign."
            )
        if set(profile.required_stage_ids) != _ENSEMBLE_REQUIRED_STAGES:
            raise ValueError(
                "Scientific claim must require the exact conditional-ensemble "
                "qualification stage set."
            )
        if len(self.batch.construct_ids) != 1 or len(self.batch.protocol_ids) != 1:
            raise ValueError(
                "Conditional-ensemble claims require one exact construct and protocol."
            )
        selected_index = (
            -1 if assessment.selected_model_id is None else int(assessment.selected_index)
        )
        selected_hypothesis = (
            None if selected_index < 0 else assessment.hypothesis_ids[selected_index]
        )
        expected_support = {
            "construct": self.batch.construct_ids[0],
            "generalization": "held-out-condition-and-perturbation",
            "observable": "mapped-mutation-profile",
            "protocol": self.batch.protocol_ids[0],
            "state-support": selected_hypothesis or "no-finite-state-support",
        }
        if (
            profile.capability_name != "rna-conditional-ensemble-inference"
            or profile.observable_ids != ("mapped-mutation-profile",)
            or profile.condition_domain_ids != tuple(sorted(self.batch.condition_ids))
            or dict(profile.support.attributes) != expected_support
        ):
            raise ValueError(
                "Scientific claim capability, observable, conditions, construct, "
                "protocol, or state support does not exactly match this workflow."
            )
        if not isinstance(stage_evidence, Sequence) or isinstance(stage_evidence, str):
            raise TypeError("stage_evidence must be a sequence of QualificationEvidence.")
        if any(not isinstance(item, QualificationEvidence) for item in stage_evidence):
            raise TypeError("stage_evidence must contain QualificationEvidence values.")
        selected_model_id = assessment.selected_model_id
        model_stages = {
            "parameter-identifiability",
            "predictive-calibration",
            "locked-prediction",
        }
        common_subjects = {
            self.campaign.campaign_id,
            self.workflow_id,
            assessment.assessment_id,
        }
        for evidence in stage_evidence:
            owned_stages = set(evidence.criteria_ids) & {
                "source-admission",
                "numerical-validity",
                *model_stages,
            }
            if evidence.outcome != "passed" or not owned_stages:
                continue
            required_subjects = (
                {
                    self.campaign.campaign_id,
                    self.batch.batch_fingerprint,
                    *self.batch.source_ids,
                }
                if "source-admission" in owned_stages
                else set(common_subjects)
            )
            if owned_stages & model_stages:
                if selected_model_id is None:
                    raise ValueError(
                        "Passed model-specific evidence requires a uniquely frozen model."
                    )
                required_subjects.add(selected_model_id)
                uncertainty_id = assessment.selected_epistemic_uncertainty_id
                if uncertainty_id is None:
                    raise ValueError(
                        "Passed model-specific evidence requires parameter uncertainty."
                    )
                required_subjects.add(uncertainty_id)
            if "predictive-calibration" in owned_stages:
                score_id = assessment.selected_predictive_calibration_score_id
                if score_id is None:
                    raise ValueError(
                        "Passed predictive calibration requires a selected model score."
                    )
                required_subjects.add(score_id)
            if "locked-prediction" in owned_stages:
                score_id = assessment.selected_locked_score_id
                if score_id is None:
                    raise ValueError(
                        "Passed locked prediction requires a selected model score."
                    )
                required_subjects.add(score_id)
                if (
                    assessment.levels[int(assessment.selected_index)] == "finite-mixture"
                    and _MIXTURE_ADVANTAGE_METRIC_ID not in evidence.criteria_ids
                ):
                    raise ValueError(
                        "Passed finite-mixture locked evidence requires its frozen "
                        "mixture-advantage criterion."
                    )
            if not required_subjects.issubset(evidence.subject_ids):
                raise ValueError(
                    "Passed workflow stage evidence has mismatched campaign, model, "
                    "assessment, or score identity."
                )
        return profile.evaluate(
            metric_values,
            stage_evidence,
            metric_units=metric_units,
            metric_aggregations=metric_aggregations,
            build_id=build_id,
            environment_id=environment_id,
            backend=backend,
            topology=topology,
            precision=precision,
            reduction=reduction,
            replay_id=replay_id,
            raw_artifact_ids=self.batch.source_ids,
            reviewer_id=reviewer_id,
            issued_at=issued_at,
            expires_at=expires_at,
        )


__all__ = [
    "ConditionalEnsembleQualificationWorkflow",
    "EnsembleMixtureAdvantageCriterion",
    "EnsemblePosteriorUncertainty",
    "EnsemblePredictiveScoreCriterion",
    "EnsembleWorkflowAssessment",
    "GroupedPredictiveScore",
    "ModelLadderEvaluation",
    "ModelLadderLevel",
    "group_profile_log_scores",
    "group_posterior_predictive_log_scores",
    "prepare_conditional_ensemble_campaign",
]
