#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import enum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from .._fingerprint import canonical_fingerprint
from .._identity import strict_module_payload
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..ml import AbstractRecipe, FeatureSchema, MLBatch, TargetSchema
from ..ml.model_selection import AbstractSplitPlan
from ._core import CausalProblem, TargetPopulationKind
from ._identify import AdjustmentExpression, IdentificationCertificate


class NuisanceStatus(enum.IntEnum):
    SUCCESS = 0
    CERTIFICATE_MISMATCH = 1
    UNSUPPORTED_FEATURE = 2
    INCOMPLETE_FOLDS = 3
    OUTCOME_FIT_FAILED = 4
    PROPENSITY_FIT_FAILED = 5
    INVALID_PREDICTION = 6


class OverlapStatus(enum.IntEnum):
    SUCCESS = 0
    PROPENSITY_OUT_OF_RANGE = 1
    MINIMUM_PROPENSITY = 2
    MAXIMUM_WEIGHT = 3
    MINIMUM_EFFECTIVE_SAMPLE = 4
    EMPTY_TARGET = 5


class EstimationStatus(enum.IntEnum):
    SUCCESS = 0
    NUISANCE_FAILED = 1
    OVERLAP_FAILED = 2
    UNSUPPORTED_CERTIFICATE = 3
    NONFINITE = 4
    INSUFFICIENT_CLUSTERS = 5


class EstimatorKind(enum.StrEnum):
    G_COMPUTATION = "g_computation"
    IPW = "ipw"
    AIPW = "aipw"


class IPWNormalization(enum.StrEnum):
    HORVITZ_THOMPSON = "horvitz_thompson"
    HAJEK = "hajek"


class DiagnosticOutcome(enum.StrEnum):
    REJECTED = "rejected"
    NOT_REJECTED = "not_rejected"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"
    NOT_APPLICABLE = "not_applicable"
    NOT_RUN = "not_run"


class NuisancePlan(StrictModule):
    outcome_recipe: AbstractRecipe
    propensity_recipe: AbstractRecipe | None
    split_plan: AbstractSplitPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        outcome_recipe: AbstractRecipe,
        split_plan: AbstractSplitPlan,
        propensity_recipe: AbstractRecipe | None = None,
    ) -> None:
        if not isinstance(outcome_recipe, AbstractRecipe):
            raise TypeError("outcome_recipe must be an AbstractRecipe.")
        if propensity_recipe is not None and not isinstance(
            propensity_recipe, AbstractRecipe
        ):
            raise TypeError("propensity_recipe must be an AbstractRecipe when supplied.")
        if not isinstance(split_plan, AbstractSplitPlan):
            raise TypeError("split_plan must be an AbstractSplitPlan.")
        payload = {
            "outcome": strict_module_payload(outcome_recipe),
            "propensity": (
                None
                if propensity_recipe is None
                else strict_module_payload(propensity_recipe)
            ),
            "split": strict_module_payload(split_plan),
        }
        object.__setattr__(self, "outcome_recipe", outcome_recipe)
        object.__setattr__(self, "propensity_recipe", propensity_recipe)
        object.__setattr__(self, "split_plan", split_plan)
        object.__setattr__(self, "plan_id", canonical_fingerprint(payload))


class CrossFittedNuisanceResult(StrictModule, NonTrainableState):
    status: Array
    outcome_active: Array
    outcome_reference: Array
    propensity_active: Array
    propensity_reference: Array
    fold_assignment: Array
    certificate_id: str = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    fit_methods: tuple[str, ...] = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(NuisanceStatus.SUCCESS)


class OverlapPolicy(StrictModule, NonTrainableState):
    minimum_propensity: float = eqx.field(static=True)
    maximum_inverse_weight: float = eqx.field(static=True)
    minimum_effective_sample_size: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_propensity: float = 0.01,
        maximum_inverse_weight: float = 100.0,
        minimum_effective_sample_size: float = 10.0,
    ) -> None:
        minimum = float(minimum_propensity)
        maximum = float(maximum_inverse_weight)
        effective = float(minimum_effective_sample_size)
        if not 0.0 < minimum < 1.0:
            raise ValueError("minimum_propensity must lie in (0, 1).")
        if maximum <= 1.0 or effective <= 0.0:
            raise ValueError(
                "Overlap weight and effective-sample limits must be positive."
            )
        object.__setattr__(self, "minimum_propensity", minimum)
        object.__setattr__(self, "maximum_inverse_weight", maximum)
        object.__setattr__(self, "minimum_effective_sample_size", effective)
        object.__setattr__(
            self,
            "policy_id",
            canonical_fingerprint(
                {
                    "minimum_propensity": minimum,
                    "maximum_inverse_weight": maximum,
                    "minimum_effective_sample_size": effective,
                }
            ),
        )


class OverlapResult(StrictModule, NonTrainableState):
    status: Array
    minimum_active_propensity: Array
    minimum_reference_propensity: Array
    maximum_inverse_weight: Array
    active_effective_sample_size: Array
    reference_effective_sample_size: Array
    policy_id: str = eqx.field(static=True)
    nuisance_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(OverlapStatus.SUCCESS)


class CausalEstimate(StrictModule, NonTrainableState):
    status: Array
    active_mean: Array
    reference_mean: Array
    effect: Array
    standard_error: Array
    interval_lower: Array
    interval_upper: Array
    influence: Array
    estimator_kind: EstimatorKind = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    nuisance_id: str = eqx.field(static=True)
    overlap_id: str = eqx.field(static=True)
    uncertainty_basis: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(EstimationStatus.SUCCESS)


class SensitivityResult(StrictModule, NonTrainableState):
    bias_bound: Array
    lower: Array
    upper: Array
    estimate_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class CausalDiagnostic(StrictModule, NonTrainableState):
    outcome: DiagnosticOutcome = eqx.field(static=True)
    criterion: str = eqx.field(static=True)
    subject_id: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    diagnostic_id: str = eqx.field(static=True)


def fit_cross_fitted_nuisance(
    problem: CausalProblem,
    certificate: IdentificationCertificate,
    plan: NuisancePlan,
    *,
    key: PRNGKeyArray,
) -> CrossFittedNuisanceResult:
    """Fit outcome and propensity nuisances out of fold."""
    expression = _verify_adjustment_certificate(problem, certificate)
    features, feature_names = _adjustment_features(problem, expression)
    treatment = jnp.asarray(problem.dataset.value(expression.treatment), dtype=jnp.int32)
    outcome = jnp.asarray(problem.dataset.value(expression.outcome))
    sample_weight = problem.dataset.sampling_weight * problem.dataset.measure_weight
    feature_schema = FeatureSchema(feature_names)
    split_batch = MLBatch(
        features,
        outcome,
        sample_mask=problem.dataset.sample_mask,
        sample_weight=sample_weight,
        groups=problem.dataset.cluster_id,
        feature_schema=feature_schema,
    )
    split_result = plan.split_plan.split(split_batch, key=key)
    n_samples = problem.dataset.n_samples
    outcome_active = jnp.full(
        (n_samples,), jnp.nan, dtype=jnp.result_type(outcome, float)
    )
    outcome_reference = jnp.full_like(outcome_active, jnp.nan)
    propensity_active = jnp.full_like(outcome_active, jnp.nan)
    propensity_reference = jnp.full_like(outcome_active, jnp.nan)
    fold_assignment = jnp.full((n_samples,), -1, dtype=jnp.int32)
    methods: list[str] = []
    active_value = int(problem.query.contrast.active.value)
    reference_value = int(problem.query.contrast.reference.value)
    treatment_variable = problem.dataset.schema.variable(expression.treatment)
    classes = int(treatment_variable.cardinality or 0)
    if classes < 2:
        return _nuisance_failure(
            problem, certificate, plan, NuisanceStatus.UNSUPPORTED_FEATURE
        )
    for fold in split_result.folds:
        train = fold.train_indices
        validation = fold.validation_indices
        train_active = problem.dataset.sample_mask[train]
        active_train = train[train_active & (treatment[train] == active_value)]
        reference_train = train[train_active & (treatment[train] == reference_value)]
        if int(active_train.size) == 0 or int(reference_train.size) == 0:
            return _nuisance_failure(
                problem,
                certificate,
                plan,
                NuisanceStatus.OUTCOME_FIT_FAILED,
            )
        active_fit = plan.outcome_recipe.fit_batch(
            MLBatch(
                features[active_train],
                outcome[active_train],
                sample_weight=sample_weight[active_train],
                feature_schema=feature_schema,
            ),
            key=jax.random.fold_in(key, fold.fold_id * 3),
        )
        reference_fit = plan.outcome_recipe.fit_batch(
            MLBatch(
                features[reference_train],
                outcome[reference_train],
                sample_weight=sample_weight[reference_train],
                feature_schema=feature_schema,
            ),
            key=jax.random.fold_in(key, fold.fold_id * 3 + 1),
        )
        if not bool(np.asarray(active_fit.valid)) or not bool(
            np.asarray(reference_fit.valid)
        ):
            return _nuisance_failure(
                problem,
                certificate,
                plan,
                NuisanceStatus.OUTCOME_FIT_FAILED,
            )
        active_prediction = jax.vmap(active_fit.model)(features[validation])
        reference_prediction = jax.vmap(reference_fit.model)(features[validation])
        if problem.design.known_assignment_probability is None:
            if plan.propensity_recipe is None:
                return _nuisance_failure(
                    problem,
                    certificate,
                    plan,
                    NuisanceStatus.PROPENSITY_FIT_FAILED,
                )
            target_schema = TargetSchema(
                "binary" if classes == 2 else "multiclass",
                class_labels=tuple(range(classes)),
            )
            propensity_fit = plan.propensity_recipe.fit_batch(
                MLBatch(
                    features[train],
                    treatment[train],
                    sample_weight=sample_weight[train],
                    sample_mask=problem.dataset.sample_mask[train],
                    feature_schema=feature_schema,
                    target_schema=target_schema,
                ),
                key=jax.random.fold_in(key, fold.fold_id * 3 + 2),
            )
            if not bool(np.asarray(propensity_fit.valid)):
                return _nuisance_failure(
                    problem,
                    certificate,
                    plan,
                    NuisanceStatus.PROPENSITY_FIT_FAILED,
                )
            raw_probability = jax.vmap(propensity_fit.model)(features[validation])
            if classes == 2:
                probabilities = jnp.stack(
                    (1.0 - raw_probability, raw_probability), axis=-1
                )
            else:
                probabilities = raw_probability
            methods.append(propensity_fit.method)
        else:
            known = problem.design.known_assignment_probability
            if known.shape[0] != n_samples:
                raise ValueError(
                    "Known assignment probabilities must align with the dataset."
                )
            if known.ndim == 1:
                if classes != 2:
                    raise ValueError("One-dimensional known propensity is binary only.")
                positive = known[validation]
                probabilities = jnp.stack((1.0 - positive, positive), axis=-1)
            else:
                if known.shape[1] != classes:
                    raise ValueError(
                        "Known assignment probability class axis is invalid."
                    )
                probabilities = known[validation]
            methods.append("known-assignment-probability")
        if probabilities.shape != (int(validation.size), classes):
            return _nuisance_failure(
                problem,
                certificate,
                plan,
                NuisanceStatus.INVALID_PREDICTION,
            )
        outcome_active = outcome_active.at[validation].set(
            active_prediction.reshape((-1,))
        )
        outcome_reference = outcome_reference.at[validation].set(
            reference_prediction.reshape((-1,))
        )
        propensity_active = propensity_active.at[validation].set(
            probabilities[:, active_value]
        )
        propensity_reference = propensity_reference.at[validation].set(
            probabilities[:, reference_value]
        )
        fold_assignment = fold_assignment.at[validation].set(fold.fold_id)
        methods.extend((active_fit.method, reference_fit.method))
    complete = np.asarray(fold_assignment) >= 0
    finite = np.isfinite(
        np.asarray(
            outcome_active + outcome_reference + propensity_active + propensity_reference
        )
    )
    probabilities_valid = (
        (np.asarray(propensity_active) > 0)
        & (np.asarray(propensity_active) <= 1)
        & (np.asarray(propensity_reference) > 0)
        & (np.asarray(propensity_reference) <= 1)
    )
    active = np.asarray(problem.dataset.sample_mask)
    if not np.all(complete[active]):
        return _nuisance_failure(
            problem,
            certificate,
            plan,
            NuisanceStatus.INCOMPLETE_FOLDS,
        )
    if not np.all((finite & probabilities_valid)[active]):
        return _nuisance_failure(
            problem,
            certificate,
            plan,
            NuisanceStatus.INVALID_PREDICTION,
        )
    result_id = canonical_fingerprint(
        {
            "certificate_id": certificate.certificate_id,
            "data_id": problem.dataset.data_id,
            "plan_id": plan.plan_id,
            "outcome_active": outcome_active,
            "outcome_reference": outcome_reference,
            "propensity_active": propensity_active,
            "propensity_reference": propensity_reference,
            "fold_assignment": fold_assignment,
            "fit_methods": tuple(methods),
        }
    )
    return CrossFittedNuisanceResult(
        status=jnp.asarray(int(NuisanceStatus.SUCCESS), dtype=jnp.int32),
        outcome_active=outcome_active,
        outcome_reference=outcome_reference,
        propensity_active=propensity_active,
        propensity_reference=propensity_reference,
        fold_assignment=fold_assignment,
        certificate_id=certificate.certificate_id,
        data_id=problem.dataset.data_id,
        plan_id=plan.plan_id,
        fit_methods=tuple(methods),
        result_id=result_id,
    )


def evaluate_overlap(
    problem: CausalProblem,
    nuisance: CrossFittedNuisanceResult,
    policy: OverlapPolicy,
) -> OverlapResult:
    if nuisance.data_id != problem.dataset.data_id:
        raise ValueError("Nuisance result is stale for this dataset.")
    target = _target_mask(problem)
    treatment = jnp.asarray(problem.dataset.value(problem.design.exposure_variable))
    active_value = problem.query.contrast.active.value
    reference_value = problem.query.contrast.reference.value
    base_weight = problem.dataset.sampling_weight * problem.dataset.measure_weight
    active_weight = jnp.where(
        target & (treatment == active_value),
        base_weight / nuisance.propensity_active,
        0.0,
    )
    reference_weight = jnp.where(
        target & (treatment == reference_value),
        base_weight / nuisance.propensity_reference,
        0.0,
    )
    minimum_active = jnp.min(jnp.where(target, nuisance.propensity_active, jnp.inf))
    minimum_reference = jnp.min(jnp.where(target, nuisance.propensity_reference, jnp.inf))
    maximum_inverse = jnp.max(
        jnp.where(
            target,
            jnp.maximum(
                1.0 / nuisance.propensity_active, 1.0 / nuisance.propensity_reference
            ),
            0.0,
        )
    )
    active_ess = _effective_sample_size(active_weight)
    reference_ess = _effective_sample_size(reference_weight)
    target_count = int(np.asarray(jnp.sum(target)))
    probabilities_valid = bool(
        np.asarray(
            jnp.all(
                jnp.where(
                    target,
                    (nuisance.propensity_active > 0)
                    & (nuisance.propensity_active <= 1)
                    & (nuisance.propensity_reference > 0)
                    & (nuisance.propensity_reference <= 1),
                    True,
                )
            )
        )
    )
    if target_count == 0:
        status_value = OverlapStatus.EMPTY_TARGET
    elif not probabilities_valid:
        status_value = OverlapStatus.PROPENSITY_OUT_OF_RANGE
    elif (
        float(minimum_active) < policy.minimum_propensity
        or float(minimum_reference) < policy.minimum_propensity
    ):
        status_value = OverlapStatus.MINIMUM_PROPENSITY
    elif float(maximum_inverse) > policy.maximum_inverse_weight:
        status_value = OverlapStatus.MAXIMUM_WEIGHT
    elif (
        float(active_ess) < policy.minimum_effective_sample_size
        or float(reference_ess) < policy.minimum_effective_sample_size
    ):
        status_value = OverlapStatus.MINIMUM_EFFECTIVE_SAMPLE
    else:
        status_value = OverlapStatus.SUCCESS
    status = jnp.asarray(int(status_value), dtype=jnp.int32)
    result_id = canonical_fingerprint(
        {
            "status": int(np.asarray(status)),
            "minimum_active": minimum_active,
            "minimum_reference": minimum_reference,
            "maximum_inverse": maximum_inverse,
            "active_ess": active_ess,
            "reference_ess": reference_ess,
            "policy_id": policy.policy_id,
            "nuisance_id": nuisance.result_id,
        }
    )
    return OverlapResult(
        status=status,
        minimum_active_propensity=minimum_active,
        minimum_reference_propensity=minimum_reference,
        maximum_inverse_weight=maximum_inverse,
        active_effective_sample_size=active_ess,
        reference_effective_sample_size=reference_ess,
        policy_id=policy.policy_id,
        nuisance_id=nuisance.result_id,
        result_id=result_id,
    )


def estimate_g_computation(
    problem: CausalProblem,
    certificate: IdentificationCertificate,
    nuisance: CrossFittedNuisanceResult,
    overlap: OverlapResult,
    *,
    confidence_level: float = 0.95,
) -> CausalEstimate:
    return _estimate(
        EstimatorKind.G_COMPUTATION,
        problem,
        certificate,
        nuisance,
        overlap,
        normalization=IPWNormalization.HORVITZ_THOMPSON,
        confidence_level=confidence_level,
    )


def estimate_ipw(
    problem: CausalProblem,
    certificate: IdentificationCertificate,
    nuisance: CrossFittedNuisanceResult,
    overlap: OverlapResult,
    *,
    normalization: IPWNormalization | str = IPWNormalization.HAJEK,
    confidence_level: float = 0.95,
) -> CausalEstimate:
    return _estimate(
        EstimatorKind.IPW,
        problem,
        certificate,
        nuisance,
        overlap,
        normalization=IPWNormalization(normalization),
        confidence_level=confidence_level,
    )


def estimate_aipw(
    problem: CausalProblem,
    certificate: IdentificationCertificate,
    nuisance: CrossFittedNuisanceResult,
    overlap: OverlapResult,
    *,
    confidence_level: float = 0.95,
) -> CausalEstimate:
    return _estimate(
        EstimatorKind.AIPW,
        problem,
        certificate,
        nuisance,
        overlap,
        normalization=IPWNormalization.HORVITZ_THOMPSON,
        confidence_level=confidence_level,
    )


def bounded_bias_sensitivity(
    estimate: CausalEstimate,
    *,
    absolute_bias_bound: float,
) -> SensitivityResult:
    bound = float(absolute_bias_bound)
    if not np.isfinite(bound) or bound < 0:
        raise ValueError("absolute_bias_bound must be finite and non-negative.")
    lower = estimate.effect - bound
    upper = estimate.effect + bound
    return SensitivityResult(
        bias_bound=jnp.asarray(bound),
        lower=lower,
        upper=upper,
        estimate_id=estimate.result_id,
        result_id=canonical_fingerprint(
            {
                "estimate_id": estimate.result_id,
                "absolute_bias_bound": bound,
            }
        ),
    )


def overlap_diagnostic(overlap: OverlapResult) -> CausalDiagnostic:
    successful = bool(np.asarray(overlap.successful))
    outcome = DiagnosticOutcome.NOT_REJECTED if successful else DiagnosticOutcome.REJECTED
    reason = (
        "The declared finite-sample overlap thresholds were not rejected."
        if successful
        else "At least one declared finite-sample overlap threshold failed."
    )
    return CausalDiagnostic(
        outcome=outcome,
        criterion="finite_sample_overlap",
        subject_id=overlap.result_id,
        reason=reason,
        diagnostic_id=canonical_fingerprint(
            {
                "outcome": outcome.value,
                "criterion": "finite_sample_overlap",
                "subject_id": overlap.result_id,
                "reason": reason,
            }
        ),
    )


def _estimate(
    kind: EstimatorKind,
    problem: CausalProblem,
    certificate: IdentificationCertificate,
    nuisance: CrossFittedNuisanceResult,
    overlap: OverlapResult,
    *,
    normalization: IPWNormalization,
    confidence_level: float,
) -> CausalEstimate:
    _verify_adjustment_certificate(problem, certificate)
    if (
        nuisance.certificate_id != certificate.certificate_id
        or nuisance.data_id != problem.dataset.data_id
    ):
        raise ValueError("Nuisance result is stale for this estimate.")
    if overlap.nuisance_id != nuisance.result_id:
        raise ValueError("Overlap result is stale for this nuisance result.")
    if not bool(np.asarray(nuisance.successful)):
        return _estimate_failure(
            kind,
            problem,
            certificate,
            nuisance,
            overlap,
            EstimationStatus.NUISANCE_FAILED,
        )
    if not bool(np.asarray(overlap.successful)):
        return _estimate_failure(
            kind, problem, certificate, nuisance, overlap, EstimationStatus.OVERLAP_FAILED
        )
    confidence = float(confidence_level)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie in (0, 1).")
    target = _target_mask(problem)
    treatment = jnp.asarray(problem.dataset.value(problem.design.exposure_variable))
    outcome = jnp.asarray(problem.dataset.value(problem.query.outcome_variable))
    active_value = problem.query.contrast.active.value
    reference_value = problem.query.contrast.reference.value
    active_indicator = (treatment == active_value).astype(outcome.dtype)
    reference_indicator = (treatment == reference_value).astype(outcome.dtype)
    raw_weight = jnp.where(
        target,
        problem.dataset.sampling_weight * problem.dataset.measure_weight,
        0.0,
    )
    target_weight = raw_weight / jnp.sum(raw_weight)
    uncertainty_available = True
    if kind is EstimatorKind.G_COMPUTATION:
        active_score = nuisance.outcome_active
        reference_score = nuisance.outcome_reference
        uncertainty_basis = "unavailable-without-refit-bootstrap"
        uncertainty_available = False
    elif kind is EstimatorKind.IPW:
        active_term = active_indicator * outcome / nuisance.propensity_active
        reference_term = reference_indicator * outcome / nuisance.propensity_reference
        if normalization is IPWNormalization.HAJEK:
            active_denominator = jnp.sum(
                target_weight * active_indicator / nuisance.propensity_active
            )
            reference_denominator = jnp.sum(
                target_weight * reference_indicator / nuisance.propensity_reference
            )
            active_score = active_term / active_denominator
            reference_score = reference_term / reference_denominator
        else:
            active_score = active_term
            reference_score = reference_term
        uncertainty_basis = f"ipw-{normalization.value}-empirical-influence"
    else:
        active_score = nuisance.outcome_active + (
            active_indicator
            * (outcome - nuisance.outcome_active)
            / nuisance.propensity_active
        )
        reference_score = nuisance.outcome_reference + (
            reference_indicator
            * (outcome - nuisance.outcome_reference)
            / nuisance.propensity_reference
        )
        uncertainty_basis = "cross-fitted-aipw-influence"
    active_mean = jnp.sum(target_weight * active_score)
    reference_mean = jnp.sum(target_weight * reference_score)
    effect = active_mean - reference_mean
    score = active_score - reference_score
    influence = jnp.where(target, score - effect, 0.0)
    if uncertainty_available:
        standard_error = _cluster_standard_error(
            influence,
            target_weight,
            problem.dataset.cluster_id,
            target,
        )
        quantile = jax.scipy.special.ndtri(0.5 + confidence / 2.0)
        lower = effect - quantile * standard_error
        upper = effect + quantile * standard_error
        finite = jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (active_mean, reference_mean, effect, standard_error, lower, upper)
                )
            )
        )
    else:
        standard_error = jnp.asarray(jnp.nan, dtype=effect.dtype)
        lower = jnp.asarray(jnp.nan, dtype=effect.dtype)
        upper = jnp.asarray(jnp.nan, dtype=effect.dtype)
        finite = jnp.all(jnp.isfinite(jnp.stack((active_mean, reference_mean, effect))))
    cluster_count = np.unique(
        np.asarray(problem.dataset.cluster_id)[np.asarray(target)]
    ).size
    status = (
        EstimationStatus.INSUFFICIENT_CLUSTERS
        if cluster_count < 2
        else EstimationStatus.SUCCESS
    )
    if not bool(np.asarray(finite)):
        status = EstimationStatus.NONFINITE
    result_id = canonical_fingerprint(
        {
            "status": int(status),
            "kind": kind.value,
            "active_mean": active_mean,
            "reference_mean": reference_mean,
            "effect": effect,
            "standard_error": standard_error,
            "interval_lower": lower,
            "interval_upper": upper,
            "certificate_id": certificate.certificate_id,
            "data_id": problem.dataset.data_id,
            "nuisance_id": nuisance.result_id,
            "overlap_id": overlap.result_id,
            "uncertainty_basis": uncertainty_basis,
        }
    )
    return CausalEstimate(
        status=jnp.asarray(int(status), dtype=jnp.int32),
        active_mean=active_mean,
        reference_mean=reference_mean,
        effect=effect,
        standard_error=standard_error,
        interval_lower=lower,
        interval_upper=upper,
        influence=influence,
        estimator_kind=kind,
        certificate_id=certificate.certificate_id,
        data_id=problem.dataset.data_id,
        nuisance_id=nuisance.result_id,
        overlap_id=overlap.result_id,
        uncertainty_basis=uncertainty_basis,
        result_id=result_id,
    )


def _adjustment_features(
    problem: CausalProblem,
    expression: AdjustmentExpression,
) -> tuple[Array, tuple[str, ...]]:
    if not expression.adjustment:
        return jnp.ones((problem.dataset.n_samples, 1)), ("intercept_only",)
    arrays: list[Array] = []
    for name in expression.adjustment:
        variable = problem.dataset.schema.variable(name)
        if variable.event_shape:
            raise ValueError(
                "Observed-data adjustment currently requires scalar covariates."
            )
        observed = np.asarray(problem.dataset.observed_mask(name))
        active = np.asarray(problem.dataset.sample_mask)
        if not bool(np.all(observed[active])):
            raise ValueError("Adjustment covariates must be observed on active samples.")
        arrays.append(jnp.asarray(problem.dataset.value(name)))
    return jnp.stack(arrays, axis=-1), expression.adjustment


def _target_mask(problem: CausalProblem) -> Array:
    target = problem.dataset.sample_mask
    if problem.query.population.kind is TargetPopulationKind.SUBGROUP:
        eligibility = problem.query.population.eligibility
        if eligibility is None or eligibility.shape != (problem.dataset.n_samples,):
            raise ValueError("Subgroup eligibility must align with the dataset.")
        target = target & eligibility
    return target


def _verify_adjustment_certificate(
    problem: CausalProblem,
    certificate: IdentificationCertificate,
) -> AdjustmentExpression:
    expected = (
        problem.dataset.schema.schema_id,
        problem.design.design_id,
        problem.available_law.law_id,
        problem.query.query_id,
        problem.query.population.population_id,
        problem.design.assumptions.ledger_id,
    )
    actual = (
        certificate.schema_id,
        certificate.design_id,
        certificate.law_id,
        certificate.query_id,
        certificate.population_id,
        certificate.assumptions_id,
    )
    if actual != expected:
        raise ValueError("Identification certificate is stale for this causal problem.")
    if certificate.evaluator_capability != "adjustment_mean" or not isinstance(
        certificate.expression,
        AdjustmentExpression,
    ):
        raise ValueError("This estimator requires an adjustment-mean certificate.")
    return certificate.expression


def _effective_sample_size(weights: Array) -> Array:
    total = jnp.sum(weights)
    square = jnp.sum(jnp.square(weights))
    return jnp.where(square > 0, jnp.square(total) / square, 0.0)


def _cluster_standard_error(
    influence: Array,
    normalized_weight: Array,
    cluster_id: Array,
    target: Array,
) -> Array:
    active_clusters, inverse = np.unique(
        np.asarray(cluster_id)[np.asarray(target)],
        return_inverse=True,
    )
    count = int(active_clusters.size)
    if count < 2:
        return jnp.asarray(jnp.nan)
    contribution = normalized_weight[target] * influence[target]
    totals = (
        jnp.zeros((count,), dtype=contribution.dtype)
        .at[jnp.asarray(inverse)]
        .add(contribution)
    )
    return jnp.sqrt((count / (count - 1.0)) * jnp.sum(jnp.square(totals)))


def _nuisance_failure(
    problem: CausalProblem,
    certificate: IdentificationCertificate,
    plan: NuisancePlan,
    status: NuisanceStatus,
) -> CrossFittedNuisanceResult:
    n = problem.dataset.n_samples
    nan = jnp.full((n,), jnp.nan)
    folds = jnp.full((n,), -1, dtype=jnp.int32)
    result_id = canonical_fingerprint(
        {
            "status": int(status),
            "certificate_id": certificate.certificate_id,
            "data_id": problem.dataset.data_id,
            "plan_id": plan.plan_id,
        }
    )
    return CrossFittedNuisanceResult(
        status=jnp.asarray(int(status), dtype=jnp.int32),
        outcome_active=nan,
        outcome_reference=nan,
        propensity_active=nan,
        propensity_reference=nan,
        fold_assignment=folds,
        certificate_id=certificate.certificate_id,
        data_id=problem.dataset.data_id,
        plan_id=plan.plan_id,
        fit_methods=(),
        result_id=result_id,
    )


def _estimate_failure(
    kind: EstimatorKind,
    problem: CausalProblem,
    certificate: IdentificationCertificate,
    nuisance: CrossFittedNuisanceResult,
    overlap: OverlapResult,
    status: EstimationStatus,
) -> CausalEstimate:
    nan = jnp.asarray(jnp.nan)
    status_array = jnp.asarray(int(status), dtype=jnp.int32)
    influence = jnp.full((problem.dataset.n_samples,), jnp.nan)
    uncertainty_basis = "unavailable"
    result_id = canonical_fingerprint(
        {
            "status": int(status),
            "kind": kind.value,
            "active_mean": nan,
            "reference_mean": nan,
            "effect": nan,
            "standard_error": nan,
            "interval_lower": nan,
            "interval_upper": nan,
            "certificate_id": certificate.certificate_id,
            "data_id": problem.dataset.data_id,
            "nuisance_id": nuisance.result_id,
            "overlap_id": overlap.result_id,
            "uncertainty_basis": uncertainty_basis,
        }
    )
    return CausalEstimate(
        status=status_array,
        active_mean=nan,
        reference_mean=nan,
        effect=nan,
        standard_error=nan,
        interval_lower=nan,
        interval_upper=nan,
        influence=influence,
        estimator_kind=kind,
        certificate_id=certificate.certificate_id,
        data_id=problem.dataset.data_id,
        nuisance_id=nuisance.result_id,
        overlap_id=overlap.result_id,
        uncertainty_basis=uncertainty_basis,
        result_id=result_id,
    )


__all__ = [
    "CausalDiagnostic",
    "CausalEstimate",
    "CrossFittedNuisanceResult",
    "DiagnosticOutcome",
    "EstimationStatus",
    "EstimatorKind",
    "IPWNormalization",
    "NuisancePlan",
    "NuisanceStatus",
    "OverlapPolicy",
    "OverlapResult",
    "OverlapStatus",
    "SensitivityResult",
    "bounded_bias_sensitivity",
    "estimate_aipw",
    "estimate_g_computation",
    "estimate_ipw",
    "evaluate_overlap",
    "fit_cross_fitted_nuisance",
    "overlap_diagnostic",
]
