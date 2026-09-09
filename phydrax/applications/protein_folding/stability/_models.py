#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Leakage-clean regularized protein-stability predictor ladder."""

from __future__ import annotations

import abc
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._numerics import solve_weighted_least_squares
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....qualification import ScientificCampaign
from ....uq import DenseCovariance, fit_laplace, ParameterSpace, PosteriorProblem
from ....uq._linearized import propagate_linearized
from ..interchange._megascale import (
    parse_mutation_code,
    ProteinStabilityCohort,
    ProteinStabilityMeasurement,
)
from ._features import (
    DoubleMutationFeatures,
    fit_protein_feature_transform,
    ProteinFeatureTransform,
    ProteinMutationFeatures,
)


PredictionValidity = Literal["valid", "abstained"]
UncertaintyKind = Literal["aleatoric", "epistemic"]


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _positive(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _role_case_ids(campaign: ScientificCampaign, role_name: str, /) -> tuple[str, ...]:
    if not isinstance(campaign, ScientificCampaign):
        raise TypeError("campaign must be a ScientificCampaign.")
    for role in campaign.roles:
        if role.name == role_name:
            return role.case_ids
    return ()


class UncertaintyComponent(StrictModule, NonTrainableState):
    """One labelled predictive variance component; no anonymous variance sums."""

    variance: Array
    label: str = eqx.field(static=True)
    kind: UncertaintyKind = eqx.field(static=True)
    conditionally_independent: bool = eqx.field(static=True)

    def __init__(
        self,
        variance: ArrayLike,
        /,
        *,
        label: str,
        kind: UncertaintyKind,
        conditionally_independent: bool,
    ):
        values = np.asarray(variance, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Uncertainty variances must be finite non-negative vectors.")
        if kind not in ("aleatoric", "epistemic"):
            raise ValueError("Uncertainty kind must be aleatoric or epistemic.")
        if not isinstance(conditionally_independent, bool):
            raise TypeError("conditionally_independent must be boolean.")
        self.variance = jnp.asarray(values)
        self.label = _identifier(label, "uncertainty label")
        self.kind = kind
        self.conditionally_independent = conditionally_independent


class StabilityPrediction(StrictModule, NonTrainableState):
    """Case-leading predictions with explicit validity and uncertainty provenance."""

    mean: Array
    valid: Array
    case_ids: tuple[str, ...] = eqx.field(static=True)
    abstention_reasons: tuple[str | None, ...] = eqx.field(static=True)
    uncertainty_components: tuple[UncertaintyComponent, ...]
    unquantified_uncertainty: tuple[str, ...] = eqx.field(static=True)
    observable: str = eqx.field(static=True)
    sign_convention: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        mean: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        case_ids: Sequence[str],
        abstention_reasons: Sequence[str | None],
        uncertainty_components: Sequence[UncertaintyComponent],
        unquantified_uncertainty: Sequence[str],
        observable: str,
        sign_convention: str,
        model_id: str,
    ):
        means = np.asarray(mean, dtype=float)
        validity = np.asarray(valid, dtype=bool)
        ids = tuple(_identifier(value, "prediction case ID") for value in case_ids)
        reasons = tuple(abstention_reasons)
        if means.shape != (len(ids),) or validity.shape != means.shape:
            raise ValueError("Predictions require one leading scalar per case.")
        if len(reasons) != len(ids):
            raise ValueError("Abstention reasons must align with prediction cases.")
        if len(set(ids)) != len(ids):
            raise ValueError("Prediction case IDs must be unique.")
        if any(validity[index] != (reasons[index] is None) for index in range(len(ids))):
            raise ValueError(
                "Valid predictions must have no reason; abstentions must have one."
            )
        if any(
            not math.isfinite(means[index])
            for index in range(len(ids))
            if validity[index]
        ):
            raise ValueError("Valid predictive means must be finite.")
        components = tuple(uncertainty_components)
        if any(
            not isinstance(value, UncertaintyComponent)
            or value.variance.shape != means.shape
            for value in components
        ):
            raise ValueError("Uncertainty components must align with prediction cases.")
        unquantified = tuple(
            _identifier(value, "unquantified uncertainty label")
            for value in unquantified_uncertainty
        )
        if len(set(unquantified)) != len(unquantified):
            raise ValueError("Unquantified uncertainty labels must be unique.")
        self.mean = jnp.asarray(means)
        self.valid = jnp.asarray(validity)
        self.case_ids = ids
        self.abstention_reasons = reasons
        self.uncertainty_components = components
        self.unquantified_uncertainty = tuple(sorted(unquantified))
        self.observable = _identifier(observable, "prediction observable")
        self.sign_convention = _identifier(sign_convention, "prediction sign convention")
        self.model_id = _identifier(model_id, "prediction model ID")

    @property
    def aleatoric_variance(self) -> Array | None:
        values = tuple(
            component.variance
            for component in self.uncertainty_components
            if component.kind == "aleatoric"
        )
        return None if not values else sum(values[1:], values[0])

    @property
    def parameter_variance(self) -> Array | None:
        values = tuple(
            component.variance
            for component in self.uncertainty_components
            if component.kind == "epistemic"
        )
        return None if not values else sum(values[1:], values[0])

    @property
    def total_variance(self) -> Array | None:
        if self.unquantified_uncertainty or any(
            not component.conditionally_independent
            for component in self.uncertainty_components
        ):
            return None
        if not self.uncertainty_components:
            return None
        values = tuple(component.variance for component in self.uncertainty_components)
        return sum(values[1:], values[0])


class AbstractProteinStabilityPredictor(StrictModule):
    """Scientific predictor interface; implementations retain immutable fit lineage."""

    model_id: str
    training_case_ids: tuple[str, ...]
    training_family_ids: tuple[str, ...]
    campaign_id: str
    cohort_id: str
    source_id: str
    transform_id: str
    training_source_manifest_ids: tuple[str, ...]

    @abc.abstractmethod
    def predict(
        self, features: Sequence[ProteinMutationFeatures], /
    ) -> StabilityPrediction:
        raise NotImplementedError

    @abc.abstractmethod
    def predict_additive(
        self,
        first: Sequence[ProteinMutationFeatures],
        second: Sequence[ProteinMutationFeatures],
        /,
    ) -> StabilityPrediction:
        raise NotImplementedError


ProteinStabilityPredictor = AbstractProteinStabilityPredictor


@dataclass(frozen=True, slots=True)
class ProteinStabilityModelFit:
    """Native fit result bound to one exact prepared protein-stability problem."""

    predictor: AbstractProteinStabilityPredictor | RegularizedPairInteractionModel | None
    successful: bool
    reasons: tuple[str, ...]
    calibration_case_ids: tuple[str, ...]
    numerical_status: int
    campaign_id: str
    cohort_id: str | None
    source_id: str | None
    transform_id: str
    training_source_manifest_ids: tuple[str, ...]
    fit_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.successful, bool):
            raise TypeError("successful must be boolean.")
        if self.successful != (self.predictor is not None) or (
            self.successful and self.reasons
        ):
            raise ValueError(
                "Successful fits require one predictor and no reasons; failed fits "
                "require no predictor."
            )
        for value, name in (
            (self.campaign_id, "campaign_id"),
            (self.transform_id, "transform_id"),
            (self.fit_id, "fit_id"),
        ):
            _identifier(value, name)
        for value, name in (
            (self.cohort_id, "cohort_id"),
            (self.source_id, "source_id"),
        ):
            if value is not None:
                _identifier(value, name)
        case_ids = tuple(
            sorted(
                _identifier(value, "calibration case ID")
                for value in self.calibration_case_ids
            )
        )
        reasons = tuple(
            sorted(_identifier(value, "fit reason") for value in self.reasons)
        )
        source_ids = tuple(
            sorted(
                _identifier(value, "training source manifest ID")
                for value in self.training_source_manifest_ids
            )
        )
        if (
            len(set(case_ids)) != len(case_ids)
            or len(set(reasons)) != len(reasons)
            or len(set(source_ids)) != len(source_ids)
        ):
            raise ValueError("Fit case, reason, and source identities must be unique.")
        object.__setattr__(self, "calibration_case_ids", case_ids)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "training_source_manifest_ids", source_ids)
        expected = _protein_stability_fit_id(
            self.predictor,
            successful=self.successful,
            reasons=reasons,
            calibration_case_ids=case_ids,
            numerical_status=self.numerical_status,
            campaign_id=self.campaign_id,
            cohort_id=self.cohort_id,
            source_id=self.source_id,
            transform_id=self.transform_id,
            training_source_manifest_ids=source_ids,
        )
        if self.fit_id != expected:
            raise ValueError("fit_id is not the content address of this fit result.")

    def validate(
        self,
        cohort: ProteinStabilityCohort,
        features: Sequence[ProteinMutationFeatures],
        /,
    ) -> None:
        """Require exact cohort, transform, source, rights, and split lineage."""
        _validate_single_fit_lineage(self, cohort, tuple(features))


class GlobalSubstitutionBaseline(AbstractProteinStabilityPredictor):
    """Regularized calibration-only residue identity/delta baseline."""

    coefficients: Array
    parameter_covariance: Array
    residual_variance: Array
    transform: ProteinFeatureTransform
    feature_indices: tuple[int, ...] = eqx.field(static=True)
    training_case_ids: tuple[str, ...] = eqx.field(static=True)
    training_family_ids: tuple[str, ...] = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)
    cohort_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)
    training_source_manifest_ids: tuple[str, ...] = eqx.field(static=True)
    feature_definition_id: str = eqx.field(static=True)
    observable: str = eqx.field(static=True)
    assay_channel: str = eqx.field(static=True)
    condition_id: str = eqx.field(static=True)
    sign_convention: str = eqx.field(static=True)
    ridge: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        coefficients: ArrayLike,
        parameter_covariance: ArrayLike,
        residual_variance: ArrayLike,
        transform: ProteinFeatureTransform,
        /,
        *,
        feature_indices: Sequence[int],
        training_case_ids: Sequence[str],
        training_family_ids: Sequence[str],
        campaign_id: str,
        cohort_id: str,
        source_id: str,
        training_source_manifest_ids: Sequence[str],
        observable: str,
        sign_convention: str,
        assay_channel: str,
        condition_id: str,
        ridge: float,
    ):
        coefficient_array = jnp.asarray(coefficients, dtype=float)
        covariance = jnp.asarray(parameter_covariance, dtype=float)
        residual = jnp.asarray(residual_variance, dtype=float).reshape(())
        indices = tuple(int(value) for value in feature_indices)
        if (
            coefficient_array.ndim != 1
            or covariance.shape != (coefficient_array.size, coefficient_array.size)
            or residual < 0.0
            or not bool(jnp.all(jnp.isfinite(coefficient_array)))
            or not bool(jnp.all(jnp.isfinite(covariance)))
            or not bool(jnp.isfinite(residual))
        ):
            raise ValueError(
                "Baseline parameters/covariance/residual variance are invalid."
            )
        if coefficient_array.size != len(indices) + 1:
            raise ValueError(
                "Baseline coefficient layout must be intercept plus selected features."
            )
        self.coefficients = coefficient_array
        self.parameter_covariance = covariance
        self.residual_variance = residual
        self.transform = transform
        self.feature_indices = indices
        training_ids = tuple(
            sorted(_identifier(value, "training case ID") for value in training_case_ids)
        )
        source_ids = tuple(
            sorted(
                _identifier(value, "training source manifest ID")
                for value in training_source_manifest_ids
            )
        )
        if len(set(training_ids)) != len(training_ids) or len(set(source_ids)) != len(
            source_ids
        ):
            raise ValueError("Training case and source identities must be unique.")
        self.training_case_ids = training_ids
        self.training_family_ids = tuple(
            sorted(
                {
                    _identifier(value, "training family ID")
                    for value in training_family_ids
                }
            )
        )
        self.campaign_id = _identifier(campaign_id, "campaign_id")
        self.cohort_id = _identifier(cohort_id, "cohort_id")
        self.source_id = _identifier(source_id, "source_id")
        self.transform_id = transform.transform_id
        self.training_source_manifest_ids = source_ids
        self.feature_definition_id = transform.feature_definition_id
        self.observable = _identifier(observable, "observable")
        self.assay_channel = _identifier(assay_channel, "assay channel")
        self.condition_id = _identifier(condition_id, "condition ID")
        self.sign_convention = _identifier(sign_convention, "sign convention")
        self.ridge = _positive(ridge, "ridge")
        self.model_id = canonical_fingerprint(
            {
                "kind": "global-substitution-baseline",
                "coefficients": [
                    float(value).hex() for value in np.asarray(coefficient_array)
                ],
                "parameter_covariance": [
                    [float(value).hex() for value in row]
                    for row in np.asarray(covariance)
                ],
                "residual_variance": float(residual).hex(),
                "transform_id": transform.transform_id,
                "feature_indices": list(indices),
                "training_case_ids": list(self.training_case_ids),
                "training_family_ids": list(self.training_family_ids),
                "campaign_id": self.campaign_id,
                "cohort_id": self.cohort_id,
                "source_id": self.source_id,
                "training_source_manifest_ids": list(self.training_source_manifest_ids),
                "observable": self.observable,
                "assay_channel": self.assay_channel,
                "condition_id": self.condition_id,
                "sign_convention": self.sign_convention,
                "ridge": self.ridge.hex(),
            }
        )

    def _design(
        self, features: Sequence[ProteinMutationFeatures], /
    ) -> tuple[Array, Array, tuple[str | None, ...]]:
        rows, valid, reasons = _transformed_rows(
            features,
            self.transform,
            assay_channel=self.assay_channel,
            condition_id=self.condition_id,
            observable=self.observable,
            sign_convention=self.sign_convention,
        )
        selected = rows[:, self.feature_indices]
        return (
            jnp.concatenate((jnp.ones((selected.shape[0], 1)), selected), axis=1),
            valid,
            reasons,
        )

    def predict(
        self, features: Sequence[ProteinMutationFeatures], /
    ) -> StabilityPrediction:
        values = tuple(features)
        design, valid, reasons = self._design(values)
        return _linear_prediction(
            design,
            valid,
            reasons,
            case_ids=tuple(value.measurement_id for value in values),
            coefficients=self.coefficients,
            covariance=self.parameter_covariance,
            residual_variance=self.residual_variance,
            model_id=self.model_id,
            observable=self.observable,
            sign_convention=self.sign_convention,
        )

    def predict_additive(
        self,
        first: Sequence[ProteinMutationFeatures],
        second: Sequence[ProteinMutationFeatures],
        /,
    ) -> StabilityPrediction:
        first_values, second_values = tuple(first), tuple(second)
        if len(first_values) != len(second_values):
            raise ValueError("Additive component batches must have equal leading size.")
        first_design, first_valid, first_reasons = self._design(first_values)
        second_design, second_valid, second_reasons = self._design(second_values)
        design = first_design + second_design
        valid = first_valid & second_valid
        reasons = tuple(
            None if bool(valid[index]) else first_reasons[index] or second_reasons[index]
            for index in range(len(first_values))
        )
        return _linear_prediction(
            design,
            valid,
            reasons,
            case_ids=tuple(
                f"additive:{first_value.measurement_id}:{second_value.measurement_id}"
                for first_value, second_value in zip(
                    first_values, second_values, strict=True
                )
            ),
            coefficients=self.coefficients,
            covariance=self.parameter_covariance,
            residual_variance=None,
            model_id=self.model_id,
            observable=self.observable,
            sign_convention=self.sign_convention,
            unquantified_uncertainty=("shared-wt-observation-covariance",),
        )


class RegularizedEnvironmentModel(AbstractProteinStabilityPredictor):
    """Hierarchical ridge model with calibration-family effects integrated at transfer."""

    coefficients: Array
    parameter_covariance: Array
    residual_variance: Array
    unseen_family_variance: Array
    transform: ProteinFeatureTransform
    training_case_ids: tuple[str, ...] = eqx.field(static=True)
    training_family_ids: tuple[str, ...] = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)
    cohort_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)
    training_source_manifest_ids: tuple[str, ...] = eqx.field(static=True)
    observable: str = eqx.field(static=True)
    assay_channel: str = eqx.field(static=True)
    condition_id: str = eqx.field(static=True)
    sign_convention: str = eqx.field(static=True)
    family_effect_scale: float = eqx.field(static=True)
    ridge: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        coefficients: ArrayLike,
        parameter_covariance: ArrayLike,
        residual_variance: ArrayLike,
        unseen_family_variance: ArrayLike,
        transform: ProteinFeatureTransform,
        /,
        *,
        training_case_ids: Sequence[str],
        training_family_ids: Sequence[str],
        campaign_id: str,
        cohort_id: str,
        source_id: str,
        training_source_manifest_ids: Sequence[str],
        observable: str,
        assay_channel: str,
        condition_id: str,
        sign_convention: str,
        family_effect_scale: float,
        ridge: float,
    ):
        coefficients_ = jnp.asarray(coefficients, dtype=float)
        covariance = jnp.asarray(parameter_covariance, dtype=float)
        residual = jnp.asarray(residual_variance, dtype=float).reshape(())
        unseen = jnp.asarray(unseen_family_variance, dtype=float).reshape(())
        families = tuple(
            sorted(
                _identifier(value, "training family ID") for value in training_family_ids
            )
        )
        expected = 1 + transform.mean.size + len(families)
        if (
            coefficients_.shape != (expected,)
            or covariance.shape != (expected, expected)
            or not bool(jnp.all(jnp.isfinite(coefficients_)))
            or not bool(jnp.all(jnp.isfinite(covariance)))
            or not bool(jnp.isfinite(residual) & (residual >= 0.0))
            or not bool(jnp.isfinite(unseen) & (unseen >= 0.0))
        ):
            raise ValueError("Environment-model parameters or uncertainty are invalid.")
        self.coefficients = coefficients_
        self.parameter_covariance = covariance
        self.residual_variance = residual
        self.unseen_family_variance = unseen
        self.transform = transform
        training_ids = tuple(
            sorted(_identifier(value, "training case ID") for value in training_case_ids)
        )
        source_ids = tuple(
            sorted(
                _identifier(value, "training source manifest ID")
                for value in training_source_manifest_ids
            )
        )
        if len(set(training_ids)) != len(training_ids) or len(set(source_ids)) != len(
            source_ids
        ):
            raise ValueError("Training case and source identities must be unique.")
        self.training_case_ids = training_ids
        self.training_family_ids = families
        self.campaign_id = _identifier(campaign_id, "campaign_id")
        self.cohort_id = _identifier(cohort_id, "cohort_id")
        self.source_id = _identifier(source_id, "source_id")
        self.transform_id = transform.transform_id
        self.training_source_manifest_ids = source_ids
        self.observable = _identifier(observable, "observable")
        self.assay_channel = _identifier(assay_channel, "assay channel")
        self.condition_id = _identifier(condition_id, "condition ID")
        self.sign_convention = _identifier(sign_convention, "sign convention")
        self.family_effect_scale = _positive(family_effect_scale, "family_effect_scale")
        self.ridge = _positive(ridge, "ridge")
        self.model_id = canonical_fingerprint(
            {
                "kind": "regularized-environment-model",
                "coefficients": [
                    float(value).hex() for value in np.asarray(coefficients_)
                ],
                "parameter_covariance": [
                    [float(value).hex() for value in row]
                    for row in np.asarray(covariance)
                ],
                "residual_variance": float(residual).hex(),
                "unseen_family_variance": float(unseen).hex(),
                "transform_id": transform.transform_id,
                "training_case_ids": list(self.training_case_ids),
                "training_family_ids": list(families),
                "campaign_id": self.campaign_id,
                "cohort_id": self.cohort_id,
                "source_id": self.source_id,
                "training_source_manifest_ids": list(self.training_source_manifest_ids),
                "observable": self.observable,
                "assay_channel": self.assay_channel,
                "condition_id": self.condition_id,
                "sign_convention": self.sign_convention,
                "family_effect_scale": self.family_effect_scale.hex(),
                "ridge": self.ridge.hex(),
            }
        )

    def _design(
        self, features: Sequence[ProteinMutationFeatures], /
    ) -> tuple[Array, Array, tuple[str | None, ...], Array]:
        rows, valid, reasons = _transformed_rows(
            features,
            self.transform,
            assay_channel=self.assay_channel,
            condition_id=self.condition_id,
            observable=self.observable,
            sign_convention=self.sign_convention,
        )
        family_rows = jnp.asarray(
            [
                [
                    self.family_effect_scale * float(item.domain_family_id == family)
                    for family in self.training_family_ids
                ]
                for item in features
            ],
            dtype=rows.dtype,
        )
        design = jnp.concatenate(
            (jnp.ones((rows.shape[0], 1)), rows, family_rows), axis=1
        )
        unseen = jnp.asarray(
            [item.domain_family_id not in self.training_family_ids for item in features]
        )
        return design, valid, reasons, unseen

    def predict(
        self, features: Sequence[ProteinMutationFeatures], /
    ) -> StabilityPrediction:
        values = tuple(features)
        design, valid, reasons, unseen = self._design(values)
        prediction = _linear_prediction(
            design,
            valid,
            reasons,
            case_ids=tuple(value.measurement_id for value in values),
            coefficients=self.coefficients,
            covariance=self.parameter_covariance,
            residual_variance=self.residual_variance,
            model_id=self.model_id,
            observable=self.observable,
            sign_convention=self.sign_convention,
        )
        transfer_component = UncertaintyComponent(
            jnp.where(unseen, self.unseen_family_variance, 0.0),
            label="integrated-unseen-family-effect",
            kind="epistemic",
            conditionally_independent=True,
        )
        return StabilityPrediction(
            prediction.mean,
            prediction.valid,
            case_ids=prediction.case_ids,
            abstention_reasons=prediction.abstention_reasons,
            uncertainty_components=(
                *prediction.uncertainty_components,
                transfer_component,
            ),
            unquantified_uncertainty=prediction.unquantified_uncertainty,
            observable=prediction.observable,
            sign_convention=prediction.sign_convention,
            model_id=prediction.model_id,
        )

    def predict_additive(
        self,
        first: Sequence[ProteinMutationFeatures],
        second: Sequence[ProteinMutationFeatures],
        /,
    ) -> StabilityPrediction:
        first_values, second_values = tuple(first), tuple(second)
        if len(first_values) != len(second_values):
            raise ValueError("Additive component batches must have equal leading size.")
        first_design, first_valid, first_reasons, first_unseen = self._design(
            first_values
        )
        second_design, second_valid, second_reasons, second_unseen = self._design(
            second_values
        )
        design = first_design + second_design
        valid = first_valid & second_valid
        reasons = tuple(
            None if bool(valid[index]) else first_reasons[index] or second_reasons[index]
            for index in range(len(first_values))
        )
        prediction = _linear_prediction(
            design,
            valid,
            reasons,
            case_ids=tuple(
                f"additive:{first_value.measurement_id}:{second_value.measurement_id}"
                for first_value, second_value in zip(
                    first_values, second_values, strict=True
                )
            ),
            coefficients=self.coefficients,
            covariance=self.parameter_covariance,
            residual_variance=None,
            model_id=self.model_id,
            observable=self.observable,
            sign_convention=self.sign_convention,
            unquantified_uncertainty=("shared-wt-observation-covariance",),
        )
        transfer = UncertaintyComponent(
            jnp.where(
                first_unseen | second_unseen,
                4.0 * self.unseen_family_variance,
                0.0,
            ),
            label="integrated-unseen-family-effect-additive",
            kind="epistemic",
            conditionally_independent=True,
        )
        return StabilityPrediction(
            prediction.mean,
            prediction.valid,
            case_ids=prediction.case_ids,
            abstention_reasons=prediction.abstention_reasons,
            uncertainty_components=(*prediction.uncertainty_components, transfer),
            unquantified_uncertainty=prediction.unquantified_uncertainty,
            observable=prediction.observable,
            sign_convention=prediction.sign_convention,
            model_id=prediction.model_id,
        )


@dataclass(frozen=True, slots=True)
class DoubleMutantCase:
    """One double measurement with its same-condition observed single components."""

    double_measurement: ProteinStabilityMeasurement
    first_single: ProteinStabilityMeasurement
    second_single: ProteinStabilityMeasurement
    pair_features: DoubleMutationFeatures

    def __post_init__(self) -> None:
        double = self.double_measurement
        singles = (self.first_single, self.second_single)
        if double.mutation_order != 2 or any(
            item.mutation_order != 1 for item in singles
        ):
            raise ValueError("Double-mutant cases require one double and two singles.")
        if any(
            item.domain_id != double.domain_id
            or item.domain_family_id != double.domain_family_id
            or item.background_id != double.background_id
            or item.sequence != double.sequence
            or item.shared_wt_id != double.shared_wt_id
            or item.condition_id != double.condition_id
            or item.assay_channel != double.assay_channel
            or item.observable != double.observable
            or item.sign_convention != double.sign_convention
            for item in singles
        ):
            raise ValueError(
                "Double and component singles must share the exact background, "
                "shared WT, condition, and observation law."
            )
        if double.observable != "delta_delta_g":
            raise ValueError(
                "The additive double-mutant challenge requires delta_delta_g; "
                "absolute delta_g needs an explicit WT offset."
            )
        if any(item.censoring != "none" for item in (double, *singles)):
            raise ValueError(
                "The Gaussian double-mutant challenge does not accept censored values."
            )
        double_tokens = {
            f"{wild}{position}{mutant}"
            for wild, position, mutant in parse_mutation_code(double.mutation_code)
        }
        single_tokens = {item.mutation_code for item in singles}
        if double_tokens != single_tokens:
            raise ValueError(
                "Component single mutations do not compose the double mutation."
            )
        if self.pair_features.case_id != double.measurement_id:
            raise ValueError("Pair features must identify the double measurement case.")
        if self.pair_features.pair_unit_id != double.pair_unit_id:
            raise ValueError(
                "Pair features must use the exact double-mutant residue-pair unit."
            )

    @property
    def observed_coupling_kcal_per_mol(self) -> float:
        return self.double_measurement.value_kcal_per_mol - (
            self.first_single.value_kcal_per_mol + self.second_single.value_kcal_per_mol
        )


class RegularizedPairInteractionModel(StrictModule):
    """Regularized symmetric pair correction trained on calibration-domain pairs only."""

    coefficients: Array
    parameter_covariance: Array
    residual_variance: Array
    mean: Array
    scale: Array
    feature_names: tuple[str, ...] = eqx.field(static=True)
    feature_definition_id: str = eqx.field(static=True)
    training_case_ids: tuple[str, ...] = eqx.field(static=True)
    training_pair_unit_ids: tuple[str, ...] = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)
    training_source_manifest_ids: tuple[str, ...] = eqx.field(static=True)
    single_model_id: str = eqx.field(static=True)
    ridge: float = eqx.field(static=True)
    assay_channel: str = eqx.field(static=True)
    condition_id: str = eqx.field(static=True)
    sign_convention: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        coefficients: ArrayLike,
        parameter_covariance: ArrayLike,
        residual_variance: ArrayLike,
        mean: ArrayLike,
        scale: ArrayLike,
        /,
        *,
        feature_names: Sequence[str],
        feature_definition_id: str,
        training_case_ids: Sequence[str],
        training_pair_unit_ids: Sequence[str],
        campaign_id: str,
        training_source_manifest_ids: Sequence[str],
        single_model_id: str,
        ridge: float,
        assay_channel: str,
        condition_id: str,
        sign_convention: str,
    ):
        coefficients_ = jnp.asarray(coefficients, dtype=float)
        covariance = jnp.asarray(parameter_covariance, dtype=float)
        residual = jnp.asarray(residual_variance, dtype=float).reshape(())
        mean_ = jnp.asarray(mean, dtype=float)
        scale_ = jnp.asarray(scale, dtype=float)
        names = tuple(feature_names)
        if (
            mean_.shape != (len(names),)
            or scale_.shape != mean_.shape
            or coefficients_.shape != (len(names) + 1,)
            or covariance.shape != (coefficients_.size, coefficients_.size)
            or not bool(jnp.all(jnp.isfinite(coefficients_)))
            or not bool(jnp.all(jnp.isfinite(covariance)))
            or not bool(jnp.all(jnp.isfinite(mean_)))
            or not bool(jnp.all(jnp.isfinite(scale_) & (scale_ > 0.0)))
            or not bool(jnp.isfinite(residual) & (residual >= 0.0))
        ):
            raise ValueError("Pair model parameters and feature transform are invalid.")
        self.coefficients = coefficients_
        self.parameter_covariance = covariance
        self.residual_variance = residual
        self.mean = mean_
        self.scale = scale_
        self.feature_names = names
        self.feature_definition_id = _identifier(
            feature_definition_id, "pair feature definition ID"
        )
        training_ids = tuple(
            sorted(_identifier(value, "training case ID") for value in training_case_ids)
        )
        pair_ids = tuple(
            sorted(
                _identifier(value, "training pair unit ID")
                for value in training_pair_unit_ids
            )
        )
        source_ids = tuple(
            sorted(
                _identifier(value, "training source manifest ID")
                for value in training_source_manifest_ids
            )
        )
        if any(
            len(set(values)) != len(values)
            for values in (training_ids, pair_ids, source_ids)
        ):
            raise ValueError("Pair training case, unit, and source IDs must be unique.")
        self.training_case_ids = training_ids
        self.training_pair_unit_ids = pair_ids
        self.campaign_id = _identifier(campaign_id, "campaign_id")
        self.training_source_manifest_ids = source_ids
        self.transform_id = canonical_fingerprint(
            {
                "kind": "protein-pair-feature-transform",
                "campaign_id": self.campaign_id,
                "feature_definition_id": self.feature_definition_id,
                "training_case_ids": list(training_ids),
                "mean": [float(value).hex() for value in np.asarray(mean_)],
                "scale": [float(value).hex() for value in np.asarray(scale_)],
            }
        )
        self.single_model_id = _identifier(single_model_id, "single_model_id")
        self.ridge = _positive(ridge, "ridge")
        self.assay_channel = _identifier(assay_channel, "assay channel")
        self.condition_id = _identifier(condition_id, "condition ID")
        self.sign_convention = _identifier(sign_convention, "sign convention")
        self.model_id = canonical_fingerprint(
            {
                "kind": "regularized-pair-interaction-model",
                "coefficients": [
                    float(value).hex() for value in np.asarray(coefficients_)
                ],
                "parameter_covariance": [
                    [float(value).hex() for value in row]
                    for row in np.asarray(covariance)
                ],
                "residual_variance": float(residual).hex(),
                "mean": [float(value).hex() for value in np.asarray(mean_)],
                "scale": [float(value).hex() for value in np.asarray(scale_)],
                "feature_definition_id": self.feature_definition_id,
                "training_case_ids": list(self.training_case_ids),
                "training_pair_unit_ids": list(self.training_pair_unit_ids),
                "campaign_id": self.campaign_id,
                "transform_id": self.transform_id,
                "training_source_manifest_ids": list(self.training_source_manifest_ids),
                "single_model_id": self.single_model_id,
                "ridge": self.ridge.hex(),
                "assay_channel": self.assay_channel,
                "condition_id": self.condition_id,
                "sign_convention": self.sign_convention,
            }
        )

    def predict(
        self, features: Sequence[DoubleMutationFeatures], /
    ) -> StabilityPrediction:
        values = tuple(features)
        if not values:
            raise ValueError("Pair prediction requires at least one case.")
        valid = np.asarray(
            [
                item.feature_names == self.feature_names
                and item.feature_definition_id == self.feature_definition_id
                for item in values
            ]
        )
        rows = jnp.stack(
            tuple(
                item.values if valid[index] else jnp.zeros_like(self.mean)
                for index, item in enumerate(values)
            )
        )
        design = jnp.concatenate(
            (jnp.ones((len(values), 1)), (rows - self.mean) / self.scale), axis=1
        )
        reasons = tuple(
            None if value else "pair-feature-definition-mismatch" for value in valid
        )
        return _linear_prediction(
            design,
            valid,
            reasons,
            case_ids=tuple(item.case_id for item in values),
            coefficients=self.coefficients,
            covariance=self.parameter_covariance,
            residual_variance=self.residual_variance,
            model_id=self.model_id,
            observable="thermodynamic_coupling",
            sign_convention=self.sign_convention,
        )


def _protein_stability_fit_id(
    predictor: AbstractProteinStabilityPredictor | RegularizedPairInteractionModel | None,
    /,
    *,
    successful: bool,
    reasons: Sequence[str],
    calibration_case_ids: Sequence[str],
    numerical_status: int,
    campaign_id: str,
    cohort_id: str | None,
    source_id: str | None,
    transform_id: str,
    training_source_manifest_ids: Sequence[str],
) -> str:
    return canonical_fingerprint(
        {
            "kind": "protein-stability-model-fit",
            "successful": successful,
            "model_id": None if predictor is None else predictor.model_id,
            "reasons": list(reasons),
            "calibration_case_ids": list(calibration_case_ids),
            "numerical_status": int(numerical_status),
            "campaign_id": campaign_id,
            "cohort_id": cohort_id,
            "source_id": source_id,
            "transform_id": transform_id,
            "training_source_manifest_ids": list(training_source_manifest_ids),
        }
    )


def _feature_matches_measurement(
    feature: ProteinMutationFeatures,
    measurement: ProteinStabilityMeasurement,
    /,
) -> bool:
    return (
        feature.measurement_id == measurement.measurement_id
        and feature.source_measurement_record_id == measurement.record_id
        and feature.domain_id == measurement.domain_id
        and feature.domain_family_id == measurement.domain_family_id
        and feature.background_id == measurement.background_id
        and feature.wt_sequence == measurement.sequence
        and feature.mutation_code == measurement.mutation_code
        and feature.assay_channel == measurement.assay_channel
        and feature.condition_id == measurement.condition_id
        and feature.observable == measurement.observable
        and feature.sign_convention == measurement.sign_convention
        and feature.shared_wt_id == measurement.shared_wt_id
    )


def _training_source_manifest_ids(
    features: Sequence[ProteinMutationFeatures],
    measurements: Sequence[ProteinStabilityMeasurement],
    /,
) -> tuple[str, ...]:
    manifests = {}
    for feature, measurement in zip(features, measurements, strict=True):
        if not _feature_matches_measurement(feature, measurement):
            raise ValueError(
                "Feature ABI does not identify its exact admitted measurement."
            )
        if measurement.source_manifest is None:
            raise ValueError("Training measurement lacks its admitted source manifest.")
        manifests[measurement.source_manifest.manifest_id] = measurement.source_manifest
        if measurement.standard_error_kcal_per_mol is not None:
            if measurement.uncertainty_source_manifest is None:
                raise ValueError(
                    "Training uncertainty lacks its admitted source manifest."
                )
            manifests[measurement.uncertainty_source_manifest.manifest_id] = (
                measurement.uncertainty_source_manifest
            )
        for manifest in feature.source_manifests:
            manifests[manifest.manifest_id] = manifest
    for manifest in manifests.values():
        manifest.require_rights(training_use=True)
    return tuple(sorted(manifests))


def _validate_single_fit_lineage(
    model_fit: ProteinStabilityModelFit,
    cohort: ProteinStabilityCohort,
    features: Sequence[ProteinMutationFeatures],
    /,
) -> None:
    if not isinstance(cohort, ProteinStabilityCohort):
        raise TypeError("cohort must be a ProteinStabilityCohort.")
    values = tuple(features)
    if not values or any(
        not isinstance(item, ProteinMutationFeatures) for item in values
    ):
        raise TypeError("features must contain ProteinMutationFeatures.")
    feature_by_id = {item.measurement_id: item for item in values}
    if len(feature_by_id) != len(values):
        raise ValueError("Feature measurement IDs must be unique.")
    measurement_by_id = {
        measurement.measurement_id: measurement for measurement in cohort.measurements
    }
    transform = fit_protein_feature_transform(values, cohort)
    fit_ids = transform.fit_case_ids
    selected_features = tuple(feature_by_id[case_id] for case_id in fit_ids)
    selected_measurements = tuple(measurement_by_id[case_id] for case_id in fit_ids)
    source_ids = _training_source_manifest_ids(selected_features, selected_measurements)
    expected_families = tuple(
        sorted({measurement.domain_family_id for measurement in selected_measurements})
    )
    locked_families = {
        measurement_by_id[case_id].domain_family_id
        for case_id in _role_case_ids(cohort.campaign, "locked_evaluation")
        if measurement_by_id[case_id].mutation_order == 1
    }
    if (
        model_fit.campaign_id != cohort.campaign.campaign_id
        or model_fit.cohort_id != cohort.cohort_id
        or model_fit.source_id != cohort.source_id
        or model_fit.transform_id != transform.transform_id
        or model_fit.calibration_case_ids != fit_ids
        or model_fit.training_source_manifest_ids != source_ids
    ):
        raise ValueError(
            "Protein stability fit must bind the exact cohort, campaign calibration "
            "cases, sources, and feature transform."
        )
    if model_fit.predictor is not None:
        predictor = model_fit.predictor
        if not isinstance(predictor, AbstractProteinStabilityPredictor):
            raise TypeError("Single-mutant fits require a protein stability predictor.")
        if (
            predictor.campaign_id != cohort.campaign.campaign_id
            or predictor.cohort_id != cohort.cohort_id
            or predictor.source_id != cohort.source_id
            or predictor.transform_id != transform.transform_id
            or predictor.transform.transform_id != transform.transform_id
            or predictor.training_case_ids != fit_ids
            or predictor.training_family_ids != expected_families
            or predictor.training_source_manifest_ids != source_ids
        ):
            raise ValueError(
                "Protein stability predictor identity does not match its exact fit "
                "campaign, cohort, source, transform, and training rights."
            )
        if locked_families.intersection(predictor.training_family_ids):
            raise ValueError(
                "Protein stability training families cannot enter locked evaluation."
            )


def _candidate_hyperparameters(
    model_fit: ProteinStabilityModelFit,
    /,
) -> tuple[tuple[str, float], ...]:
    predictor = model_fit.predictor
    if isinstance(predictor, GlobalSubstitutionBaseline):
        return (("ridge", predictor.ridge),)
    if isinstance(predictor, RegularizedEnvironmentModel):
        return (
            ("family_effect_scale", predictor.family_effect_scale),
            ("ridge", predictor.ridge),
        )
    raise TypeError(
        "Protein stability selection candidates must be successful baseline or "
        "regularized-environment fits."
    )


@dataclass(frozen=True, slots=True, init=False)
class ProteinStabilityModelSelectionRecord:
    """Selection scores computed only on frozen model-selection families."""

    candidate_fits: tuple[ProteinStabilityModelFit, ...]
    candidate_fit_ids: tuple[str, ...]
    candidate_hyperparameters: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    candidate_scores: tuple[tuple[str, float], ...]
    model_selection_case_ids: tuple[str, ...]
    model_selection_measurement_record_ids: tuple[str, ...]
    model_selection_feature_ids: tuple[str, ...]
    selection_source_manifest_ids: tuple[str, ...]
    score_name: str
    campaign_id: str
    cohort_id: str
    source_id: str
    transform_id: str
    chosen_fit: ProteinStabilityModelFit
    chosen_fit_id: str
    chosen_baseline_fit: ProteinStabilityModelFit
    chosen_baseline_fit_id: str
    selection_id: str

    def __init__(
        self,
        cohort: ProteinStabilityCohort,
        features: Sequence[ProteinMutationFeatures],
        candidate_fits: Sequence[ProteinStabilityModelFit],
        /,
        *,
        candidate_hyperparameters: Mapping[str, Mapping[str, float]],
        score_name: str = "family-macro-mae-kcal-per-mol",
    ):
        if not isinstance(cohort, ProteinStabilityCohort):
            raise TypeError("cohort must be a ProteinStabilityCohort.")
        values = tuple(features)
        candidates = tuple(candidate_fits)
        if not values or any(
            not isinstance(item, ProteinMutationFeatures) for item in values
        ):
            raise TypeError("features must contain ProteinMutationFeatures.")
        if not candidates or any(
            not isinstance(item, ProteinStabilityModelFit) for item in candidates
        ):
            raise TypeError(
                "candidate_fits must contain ProteinStabilityModelFit values."
            )
        if any(not item.successful for item in candidates):
            raise ValueError("Model-selection candidates must be successful native fits.")
        ordered = tuple(sorted(candidates, key=lambda item: item.fit_id))
        fit_ids = tuple(item.fit_id for item in ordered)
        if len(set(fit_ids)) != len(fit_ids):
            raise ValueError("Model-selection candidate fits must be unique.")
        for model_fit in ordered:
            _validate_single_fit_lineage(model_fit, cohort, values)
        if not isinstance(candidate_hyperparameters, Mapping) or set(
            candidate_hyperparameters
        ) != set(fit_ids):
            raise ValueError(
                "Candidate hyperparameters must cover the prespecified fits exactly."
            )
        normalized_hyperparameters = []
        for model_fit in ordered:
            provided = candidate_hyperparameters[model_fit.fit_id]
            if not isinstance(provided, Mapping):
                raise TypeError("Each candidate hyperparameter record must be a mapping.")
            normalized = tuple(
                sorted(
                    (
                        _identifier(str(name), "hyperparameter name"),
                        float(value),
                    )
                    for name, value in provided.items()
                )
            )
            if any(
                isinstance(value, bool) or not math.isfinite(value)
                for value in provided.values()
            ):
                raise ValueError("Candidate hyperparameters must be finite real values.")
            expected = _candidate_hyperparameters(model_fit)
            if normalized != expected:
                raise ValueError(
                    "Candidate hyperparameters must exactly match the fitted model."
                )
            normalized_hyperparameters.append((model_fit.fit_id, normalized))
        if score_name != "family-macro-mae-kcal-per-mol":
            raise ValueError(
                "Protein stability selection requires the prespecified family-macro "
                "MAE score."
            )
        feature_by_id = {item.measurement_id: item for item in values}
        if len(feature_by_id) != len(values):
            raise ValueError("Feature measurement IDs must be unique.")
        measurement_by_id = {
            measurement.measurement_id: measurement for measurement in cohort.measurements
        }
        selection_ids = _role_case_ids(cohort.campaign, "model_selection")
        if not selection_ids:
            raise ValueError("Protein stability model selection requires frozen cases.")
        if any(
            case_id not in feature_by_id
            or measurement_by_id[case_id].mutation_order != 1
            or measurement_by_id[case_id].censoring != "none"
            for case_id in selection_ids
        ):
            raise ValueError(
                "Every model-selection case requires one exact uncensored "
                "single-mutant feature."
            )
        selection_features = tuple(feature_by_id[case_id] for case_id in selection_ids)
        selection_measurements = tuple(
            measurement_by_id[case_id] for case_id in selection_ids
        )
        source_ids = _training_source_manifest_ids(
            selection_features, selection_measurements
        )
        scores = []
        for model_fit in ordered:
            prediction = model_fit.predictor.predict(selection_features)
            validity = np.asarray(prediction.valid)
            means = np.asarray(prediction.mean)
            if (
                prediction.case_ids != selection_ids
                or not np.all(validity)
                or not np.all(np.isfinite(means))
            ):
                raise ValueError(
                    "Every prespecified candidate must predict every model-selection "
                    "case without abstention."
                )
            group_errors: dict[str, list[float]] = {}
            for index, measurement in enumerate(selection_measurements):
                group_errors.setdefault(measurement.independent_group_id, []).append(
                    abs(float(means[index]) - measurement.value_kcal_per_mol)
                )
            score = float(np.mean([np.mean(errors) for errors in group_errors.values()]))
            scores.append((model_fit.fit_id, score))
        candidate_scores = tuple(scores)
        score_by_fit_id = dict(candidate_scores)
        baseline_fit_ids = tuple(
            item.fit_id
            for item in ordered
            if isinstance(item.predictor, GlobalSubstitutionBaseline)
        )
        model_fit_ids = tuple(
            item.fit_id
            for item in ordered
            if isinstance(item.predictor, RegularizedEnvironmentModel)
        )
        if not baseline_fit_ids or not model_fit_ids:
            raise ValueError(
                "Model selection requires prespecified baseline and environment "
                "candidate ladders."
            )
        chosen_baseline_fit_id = min(
            baseline_fit_ids,
            key=lambda fit_id: (score_by_fit_id[fit_id], fit_id),
        )
        chosen_fit_id = min(
            model_fit_ids,
            key=lambda fit_id: (score_by_fit_id[fit_id], fit_id),
        )
        fit_by_id = {item.fit_id: item for item in ordered}
        chosen_baseline_fit = fit_by_id[chosen_baseline_fit_id]
        chosen_fit = fit_by_id[chosen_fit_id]
        transform_id = ordered[0].transform_id
        measurement_record_ids = tuple(
            measurement.record_id for measurement in selection_measurements
        )
        feature_ids = tuple(feature.feature_id for feature in selection_features)
        selection_id = canonical_fingerprint(
            {
                "kind": "protein-stability-model-selection",
                "campaign_id": cohort.campaign.campaign_id,
                "cohort_id": cohort.cohort_id,
                "source_id": cohort.source_id,
                "transform_id": transform_id,
                "candidate_fit_ids": list(fit_ids),
                "candidate_hyperparameters": [
                    (
                        fit_id,
                        [(name, value.hex()) for name, value in hyperparameters],
                    )
                    for fit_id, hyperparameters in normalized_hyperparameters
                ],
                "score_name": score_name,
                "model_selection_case_ids": list(selection_ids),
                "model_selection_measurement_record_ids": list(measurement_record_ids),
                "model_selection_feature_ids": list(feature_ids),
                "selection_source_manifest_ids": list(source_ids),
                "candidate_scores": [
                    (fit_id, score.hex()) for fit_id, score in candidate_scores
                ],
                "chosen_fit_id": chosen_fit_id,
                "chosen_baseline_fit_id": chosen_baseline_fit_id,
            }
        )
        for name, value in (
            ("candidate_fits", ordered),
            ("candidate_fit_ids", fit_ids),
            ("candidate_hyperparameters", tuple(normalized_hyperparameters)),
            ("candidate_scores", candidate_scores),
            ("model_selection_case_ids", selection_ids),
            ("model_selection_measurement_record_ids", measurement_record_ids),
            ("model_selection_feature_ids", feature_ids),
            ("selection_source_manifest_ids", source_ids),
            ("score_name", score_name),
            ("campaign_id", cohort.campaign.campaign_id),
            ("cohort_id", cohort.cohort_id),
            ("source_id", cohort.source_id),
            ("transform_id", transform_id),
            ("chosen_fit", chosen_fit),
            ("chosen_fit_id", chosen_fit_id),
            ("chosen_baseline_fit", chosen_baseline_fit),
            ("chosen_baseline_fit_id", chosen_baseline_fit_id),
            ("selection_id", selection_id),
        ):
            object.__setattr__(self, name, value)

    def validate(
        self,
        cohort: ProteinStabilityCohort,
        features: Sequence[ProteinMutationFeatures],
        /,
    ) -> None:
        """Recompute exact held-out scores and chosen fit from prepared inputs."""
        hyperparameters = {
            fit_id: dict(values) for fit_id, values in self.candidate_hyperparameters
        }
        expected = ProteinStabilityModelSelectionRecord(
            cohort,
            tuple(features),
            self.candidate_fits,
            candidate_hyperparameters=hyperparameters,
            score_name=self.score_name,
        )
        if (
            self.selection_id != expected.selection_id
            or self.candidate_fit_ids != expected.candidate_fit_ids
            or self.candidate_scores != expected.candidate_scores
            or self.chosen_fit_id != expected.chosen_fit_id
            or self.chosen_fit.fit_id != expected.chosen_fit.fit_id
            or self.chosen_baseline_fit_id != expected.chosen_baseline_fit_id
            or self.chosen_baseline_fit.fit_id != expected.chosen_baseline_fit.fit_id
        ):
            raise ValueError(
                "Model-selection record does not match exact frozen selection inputs."
            )


def _transformed_rows(
    features: Sequence[ProteinMutationFeatures],
    transform: ProteinFeatureTransform,
    /,
    *,
    assay_channel: str,
    condition_id: str,
    observable: str,
    sign_convention: str,
) -> tuple[Array, Array, tuple[str | None, ...]]:
    values = tuple(features)
    if not values:
        raise ValueError("Prediction requires at least one case.")
    rows: list[Array] = []
    valid: list[bool] = []
    reasons: list[str | None] = []
    for item in values:
        if not isinstance(item, ProteinMutationFeatures):
            raise TypeError("Prediction features must be ProteinMutationFeatures.")
        reason = None
        if (
            item.feature_names != transform.feature_names
            or item.feature_definition_id != transform.feature_definition_id
        ):
            reason = "feature-definition-mismatch"
        elif item.assay_channel != assay_channel:
            reason = "assay-channel-domain-mismatch"
        elif item.condition_id != condition_id:
            reason = "assay-condition-domain-mismatch"
        elif item.observable != observable:
            reason = "observable-domain-mismatch"
        elif item.sign_convention != sign_convention:
            reason = "sign-convention-domain-mismatch"
        compatible = reason is None
        valid.append(compatible)
        reasons.append(reason)
        rows.append(
            transform.transform(item) if compatible else jnp.zeros_like(transform.mean)
        )
    return jnp.stack(tuple(rows)), jnp.asarray(valid), tuple(reasons)


def _linear_prediction(
    design: Array,
    valid: Array,
    reasons: Sequence[str | None],
    /,
    *,
    case_ids: Sequence[str],
    coefficients: Array,
    covariance: Array,
    residual_variance: Array | None,
    model_id: str,
    observable: str,
    sign_convention: str,
    unquantified_uncertainty: Sequence[str] = (),
) -> StabilityPrediction:
    propagation = propagate_linearized(
        lambda values: design @ values,
        coefficients,
        DenseCovariance(covariance),
        source="epistemic",
    )
    parameter_variance = propagation.exact_variance()
    components: list[UncertaintyComponent] = [
        UncertaintyComponent(
            parameter_variance,
            label="regularized-linear-laplace-parameter",
            kind="epistemic",
            conditionally_independent=True,
        )
    ]
    if residual_variance is not None:
        components.append(
            UncertaintyComponent(
                jnp.broadcast_to(residual_variance, design.shape[:1]),
                label="calibration-residual-discrepancy",
                kind="aleatoric",
                conditionally_independent=True,
            )
        )
    return StabilityPrediction(
        propagation.mean,
        valid,
        case_ids=case_ids,
        abstention_reasons=reasons,
        uncertainty_components=components,
        unquantified_uncertainty=unquantified_uncertainty,
        observable=observable,
        sign_convention=sign_convention,
        model_id=model_id,
    )


def _measurement_weights(
    measurements: Sequence[ProteinStabilityMeasurement], /
) -> tuple[Array | None, tuple[str, ...]]:
    if any(
        item.standard_error_kcal_per_mol is None
        or item.uncertainty_source_manifest_id is None
        for item in measurements
    ):
        return None, ("calibration-measurement-uncertainty-unquantified",)
    shared_wt_blocks: dict[tuple[str, str], int] = {}
    for item in measurements:
        block = (item.shared_wt_id, item.assay_channel)
        shared_wt_blocks[block] = shared_wt_blocks.get(block, 0) + 1
    if any(count > 1 for count in shared_wt_blocks.values()):
        return None, ("shared-wt-block-covariance-unquantified",)
    return (
        jnp.asarray(
            [1.0 / float(item.standard_error_kcal_per_mol) ** 2 for item in measurements]
        ),
        (),
    )


def _fit_linear(
    design: Array,
    target: Array,
    weights: Array,
    /,
    *,
    ridge: float,
) -> tuple[Array | None, Array | None, Array | None, int, tuple[str, ...]]:
    ridge_ = _positive(ridge, "ridge")
    if (
        design.ndim != 2
        or target.shape != design.shape[:1]
        or weights.shape != target.shape
    ):
        raise ValueError("Linear fit design, targets, and weights must align.")
    if design.shape[0] < 2:
        return None, None, None, -1, ("fewer-than-two-calibration-cases",)
    if design.shape[1] > 256:
        return None, None, None, -1, ("dense-posterior-dimension-exceeds-256",)
    result = solve_weighted_least_squares(
        design,
        target,
        weights=weights,
        ridge=ridge_,
        min_samples=2,
        max_features=design.shape[1],
    )
    status = int(result.status)
    if not bool(result.valid):
        return None, None, None, status, ("native-regularized-linear-fit-invalid",)
    coefficients = jnp.asarray(result.coefficients).reshape((design.shape[1],))
    normalized_weights = weights / jnp.sum(weights)
    space = ParameterSpace(
        coefficients,
        log_prior=lambda value: -0.5 * ridge_ * jnp.sum(value**2),
    )
    problem = PosteriorProblem(
        space,
        lambda value: -0.5 * jnp.sum(normalized_weights * (design @ value - target) ** 2),
    )
    posterior = fit_laplace(
        problem,
        coefficients,
        stationarity_tolerance=None,
        max_dimension=256,
    )
    residual = target - design @ coefficients
    residual_variance = jnp.sum(normalized_weights * residual**2)
    return coefficients, posterior.covariance, residual_variance, status, ()


def _aligned_calibration(
    features: Sequence[ProteinMutationFeatures],
    cohort: ProteinStabilityCohort,
    transform: ProteinFeatureTransform,
    /,
) -> tuple[
    tuple[ProteinMutationFeatures, ...],
    tuple[ProteinStabilityMeasurement, ...],
    tuple[str, ...],
]:
    if not isinstance(cohort, ProteinStabilityCohort):
        raise TypeError("cohort must be a ProteinStabilityCohort.")
    feature_by_id = {item.measurement_id: item for item in features}
    measurement_by_id = {item.measurement_id: item for item in cohort.measurements}
    if len(feature_by_id) != len(features):
        raise ValueError("Feature measurement IDs must be unique.")
    expected_transform = fit_protein_feature_transform(features, cohort)
    if transform.transform_id != expected_transform.transform_id:
        raise ValueError(
            "Feature transform must be generated from the exact campaign "
            "calibration features."
        )
    selected_ids = expected_transform.fit_case_ids
    calibration_ids = frozenset(_role_case_ids(cohort.campaign, "calibration"))
    if not set(selected_ids).issubset(calibration_ids):
        raise ValueError("Feature transform fit cases must belong to calibration.")
    if any(
        case_id not in feature_by_id or case_id not in measurement_by_id
        for case_id in selected_ids
    ):
        raise ValueError(
            "Every transformed calibration case needs one measurement and feature."
        )
    calibration_features = tuple(feature_by_id[case_id] for case_id in selected_ids)
    calibration_measurements = tuple(
        measurement_by_id[case_id] for case_id in selected_ids
    )
    source_ids = _training_source_manifest_ids(
        calibration_features, calibration_measurements
    )
    return calibration_features, calibration_measurements, source_ids


def fit_global_substitution_baseline(
    features: Sequence[ProteinMutationFeatures],
    cohort: ProteinStabilityCohort,
    transform: ProteinFeatureTransform,
    /,
    *,
    ridge: float,
) -> ProteinStabilityModelFit:
    """Fit the declared identity/delta baseline on calibration cases only."""
    selected_features, selected_measurements, source_ids = _aligned_calibration(
        tuple(features), cohort, transform
    )
    metadata = {
        "campaign_id": cohort.campaign.campaign_id,
        "cohort_id": cohort.cohort_id,
        "source_id": cohort.source_id,
        "transform_id": transform.transform_id,
        "training_source_manifest_ids": source_ids,
    }
    if any(item.mutation_order != 1 for item in selected_measurements):
        reasons = ("single-mutant-calibration-required",)
        return _failed_fit(None, reasons, selected_measurements, -1, **metadata)
    if any(item.censoring != "none" for item in selected_measurements):
        reasons = ("gaussian-baseline-does-not-support-censored-calibration",)
        return _failed_fit(None, reasons, selected_measurements, -1, **metadata)
    observables = {item.observable for item in selected_measurements}
    signs = {item.sign_convention for item in selected_measurements}
    channels = {item.assay_channel for item in selected_measurements}
    conditions = {item.condition_id for item in selected_measurements}
    if any(len(values) != 1 for values in (observables, signs, channels, conditions)):
        reasons = ("mixed-condition-channel-observable-or-sign-calibration",)
        return _failed_fit(None, reasons, selected_measurements, -1, **metadata)
    indices = tuple(
        index
        for index, name in enumerate(transform.feature_names)
        if name.startswith("wt:")
        or name.startswith("mutant:")
        or name in ("delta-volume", "delta-charge", "delta-hydropathy")
    )
    rows = jnp.stack(tuple(transform.transform(item) for item in selected_features))
    design = jnp.concatenate((jnp.ones((len(rows), 1)), rows[:, indices]), axis=1)
    target = jnp.asarray([item.value_kcal_per_mol for item in selected_measurements])
    weights, uncertainty_reasons = _measurement_weights(selected_measurements)
    if weights is None:
        return _failed_fit(
            None,
            uncertainty_reasons,
            selected_measurements,
            -1,
            **metadata,
        )
    result = _fit_linear(design, target, weights, ridge=ridge)
    coefficients, covariance, residual, status, reasons = result
    if reasons:
        return _failed_fit(None, reasons, selected_measurements, status, **metadata)
    predictor = GlobalSubstitutionBaseline(
        coefficients,
        covariance,
        residual,
        transform,
        feature_indices=indices,
        training_case_ids=tuple(item.measurement_id for item in selected_measurements),
        training_family_ids=tuple(
            item.domain_family_id for item in selected_measurements
        ),
        campaign_id=cohort.campaign.campaign_id,
        cohort_id=cohort.cohort_id,
        source_id=cohort.source_id,
        training_source_manifest_ids=source_ids,
        observable=next(iter(observables)),
        sign_convention=next(iter(signs)),
        assay_channel=next(iter(channels)),
        condition_id=next(iter(conditions)),
        ridge=ridge,
    )
    return _successful_fit(predictor, selected_measurements, status, **metadata)


def fit_regularized_environment_model(
    features: Sequence[ProteinMutationFeatures],
    cohort: ProteinStabilityCohort,
    transform: ProteinFeatureTransform,
    /,
    *,
    ridge: float,
    family_effect_scale: float,
) -> ProteinStabilityModelFit:
    """Fit global environment terms and shrunk calibration-family intercepts."""
    selected_features, selected_measurements, source_ids = _aligned_calibration(
        tuple(features), cohort, transform
    )
    metadata = {
        "campaign_id": cohort.campaign.campaign_id,
        "cohort_id": cohort.cohort_id,
        "source_id": cohort.source_id,
        "transform_id": transform.transform_id,
        "training_source_manifest_ids": source_ids,
    }
    if any(item.mutation_order != 1 for item in selected_measurements):
        return _failed_fit(
            None,
            ("single-mutant-calibration-required",),
            selected_measurements,
            -1,
            **metadata,
        )
    if any(item.censoring != "none" for item in selected_measurements):
        return _failed_fit(
            None,
            ("gaussian-environment-model-does-not-support-censored-calibration",),
            selected_measurements,
            -1,
            **metadata,
        )
    observables = {item.observable for item in selected_measurements}
    signs = {item.sign_convention for item in selected_measurements}
    channels = {item.assay_channel for item in selected_measurements}
    conditions = {item.condition_id for item in selected_measurements}
    if any(len(values) != 1 for values in (observables, signs, channels, conditions)):
        return _failed_fit(
            None,
            ("mixed-condition-channel-observable-or-sign-calibration",),
            selected_measurements,
            -1,
            **metadata,
        )
    family_scale = _positive(family_effect_scale, "family_effect_scale")
    families = tuple(sorted({item.domain_family_id for item in selected_features}))
    rows = jnp.stack(tuple(transform.transform(item) for item in selected_features))
    family_rows = jnp.asarray(
        [
            [family_scale * float(item.domain_family_id == family) for family in families]
            for item in selected_features
        ]
    )
    design = jnp.concatenate((jnp.ones((len(rows), 1)), rows, family_rows), axis=1)
    target = jnp.asarray([item.value_kcal_per_mol for item in selected_measurements])
    weights, uncertainty_reasons = _measurement_weights(selected_measurements)
    if weights is None:
        return _failed_fit(
            None,
            uncertainty_reasons,
            selected_measurements,
            -1,
            **metadata,
        )
    coefficients, covariance, residual, status, reasons = _fit_linear(
        design,
        target,
        weights,
        ridge=ridge,
    )
    if reasons:
        return _failed_fit(None, reasons, selected_measurements, status, **metadata)
    family_start = 1 + rows.shape[1]
    family_effects = coefficients[family_start:] * family_scale
    unseen_family_variance = (
        jnp.var(family_effects, ddof=1) if len(families) > 1 else jnp.asarray(0.0)
    )
    predictor = RegularizedEnvironmentModel(
        coefficients,
        covariance,
        residual,
        unseen_family_variance,
        transform,
        training_case_ids=tuple(item.measurement_id for item in selected_measurements),
        training_family_ids=families,
        campaign_id=cohort.campaign.campaign_id,
        cohort_id=cohort.cohort_id,
        source_id=cohort.source_id,
        training_source_manifest_ids=source_ids,
        observable=next(iter(observables)),
        sign_convention=next(iter(signs)),
        assay_channel=next(iter(channels)),
        condition_id=next(iter(conditions)),
        family_effect_scale=family_scale,
        ridge=ridge,
    )
    return _successful_fit(predictor, selected_measurements, status, **metadata)


def fit_regularized_pair_interaction_model(
    cases: Sequence[DoubleMutantCase],
    campaign: ScientificCampaign,
    /,
    *,
    single_model_id: str,
    ridge: float,
) -> ProteinStabilityModelFit:
    """Fit observed coupling on calibration pairs only after a single model is frozen."""
    values = tuple(cases)
    if not values or any(not isinstance(item, DoubleMutantCase) for item in values):
        raise TypeError("cases must contain DoubleMutantCase values.")
    if not isinstance(campaign, ScientificCampaign):
        raise TypeError("campaign must be a ScientificCampaign.")
    by_id = {item.double_measurement.measurement_id: item for item in values}
    if len(by_id) != len(values):
        raise ValueError("Double-mutant measurement IDs must be unique.")
    if set(by_id) != set(campaign.case_ids):
        raise ValueError("Double-mutant fit cases must equal the campaign cases.")
    role_by_case = {
        case_id: role.name for role in campaign.roles for case_id in role.case_ids
    }
    case_by_id = {case.case_id: case for case in campaign.cases}
    pair_roles: dict[str, set[str]] = {}
    for item in values:
        case_id = item.double_measurement.measurement_id
        expected_sources = tuple(
            sorted(
                {
                    artifact_id
                    for measurement in (
                        item.double_measurement,
                        item.first_single,
                        item.second_single,
                    )
                    for artifact_id in (
                        measurement.source_manifest_id,
                        measurement.uncertainty_source_manifest_id,
                    )
                    if artifact_id is not None
                }
            )
        )
        case = case_by_id[case_id]
        if (
            case.independent_unit_id != item.pair_features.pair_unit_id
            or case.condition_id != item.double_measurement.condition_id
            or case.source_manifest_ids != expected_sources
        ):
            raise ValueError(
                "Double-mutant campaign cases must retain exact pair, condition, "
                "and source lineage."
            )
        pair_roles.setdefault(item.pair_features.pair_unit_id, set()).add(
            role_by_case[case_id]
        )
    if any(len(roles) != 1 for roles in pair_roles.values()):
        raise ValueError(
            "All substitutions at a residue pair must share one campaign role."
        )
    calibration_ids = _role_case_ids(campaign, "calibration")
    selected = tuple(by_id[case_id] for case_id in calibration_ids)
    if not selected:
        context_id = canonical_fingerprint(
            {
                "kind": "protein-pair-fit-context",
                "campaign_id": campaign.campaign_id,
                "calibration_case_ids": [],
            }
        )
        return _failed_fit(
            None,
            ("no-calibration-double-mutant-pairs",),
            (),
            -1,
            campaign_id=campaign.campaign_id,
            cohort_id=None,
            source_id=None,
            transform_id=context_id,
            training_source_manifest_ids=(),
        )
    manifests = {}
    for item in selected:
        for measurement in (
            item.double_measurement,
            item.first_single,
            item.second_single,
        ):
            if measurement.source_manifest is None:
                raise ValueError(
                    "Pair calibration measurement lacks its admitted source manifest."
                )
            manifests[measurement.source_manifest.manifest_id] = (
                measurement.source_manifest
            )
            if measurement.standard_error_kcal_per_mol is not None:
                if measurement.uncertainty_source_manifest is None:
                    raise ValueError(
                        "Pair calibration uncertainty lacks its admitted source manifest."
                    )
                manifests[measurement.uncertainty_source_manifest.manifest_id] = (
                    measurement.uncertainty_source_manifest
                )
        for manifest in item.pair_features.source_manifests:
            manifests[manifest.manifest_id] = manifest
    for manifest in manifests.values():
        manifest.require_rights(training_use=True)
    source_ids = tuple(sorted(manifests))
    names = selected[0].pair_features.feature_names
    definition = selected[0].pair_features.feature_definition_id
    context_id = canonical_fingerprint(
        {
            "kind": "protein-pair-fit-context",
            "campaign_id": campaign.campaign_id,
            "calibration_case_ids": list(calibration_ids),
            "pair_feature_ids": [
                item.pair_features.feature_definition_id for item in selected
            ],
        }
    )
    metadata = {
        "campaign_id": campaign.campaign_id,
        "cohort_id": None,
        "source_id": None,
        "transform_id": context_id,
        "training_source_manifest_ids": source_ids,
    }
    if any(
        item.pair_features.feature_names != names
        or item.pair_features.feature_definition_id != definition
        for item in selected
    ):
        return _failed_fit(
            None,
            ("mixed-pair-feature-definition",),
            tuple(item.double_measurement for item in selected),
            -1,
            **metadata,
        )
    signs = {item.double_measurement.sign_convention for item in selected}
    channels = {item.double_measurement.assay_channel for item in selected}
    conditions = {item.double_measurement.condition_id for item in selected}
    if any(len(values_) != 1 for values_ in (signs, channels, conditions)):
        return _failed_fit(
            None,
            ("mixed-condition-channel-or-sign-pair-calibration",),
            tuple(item.double_measurement for item in selected),
            -1,
            **metadata,
        )
    matrix = np.stack([np.asarray(item.pair_features.values) for item in selected])
    mean = np.mean(matrix, axis=0)
    raw_scale = np.std(matrix, axis=0, ddof=0)
    scale = np.where(raw_scale == 0.0, 1.0, raw_scale)
    normalized = jnp.asarray((matrix - mean) / scale)
    design = jnp.concatenate((jnp.ones((len(selected), 1)), normalized), axis=1)
    target = jnp.asarray([item.observed_coupling_kcal_per_mol for item in selected])
    measurements = tuple(item.double_measurement for item in selected)
    weights, uncertainty_reasons = _measurement_weights(measurements)
    if weights is None:
        return _failed_fit(None, uncertainty_reasons, measurements, -1, **metadata)
    coefficients, covariance, residual, status, reasons = _fit_linear(
        design,
        target,
        weights,
        ridge=ridge,
    )
    if reasons:
        return _failed_fit(None, reasons, measurements, status, **metadata)
    predictor = RegularizedPairInteractionModel(
        coefficients,
        covariance,
        residual,
        mean,
        scale,
        feature_names=names,
        feature_definition_id=definition,
        training_case_ids=calibration_ids,
        training_pair_unit_ids=tuple(
            item.pair_features.pair_unit_id for item in selected
        ),
        campaign_id=campaign.campaign_id,
        training_source_manifest_ids=source_ids,
        single_model_id=single_model_id,
        ridge=ridge,
        assay_channel=next(iter(channels)),
        condition_id=next(iter(conditions)),
        sign_convention=next(iter(signs)),
    )
    metadata["transform_id"] = predictor.transform_id
    return _successful_fit(predictor, measurements, status, **metadata)


def _failed_fit(
    predictor,
    reasons: Sequence[str],
    measurements: Sequence[ProteinStabilityMeasurement],
    numerical_status: int,
    /,
    *,
    campaign_id: str,
    cohort_id: str | None,
    source_id: str | None,
    transform_id: str,
    training_source_manifest_ids: Sequence[str],
) -> ProteinStabilityModelFit:
    ids = tuple(sorted(item.measurement_id for item in measurements))
    reasons_ = tuple(sorted(reasons))
    fit_id = _protein_stability_fit_id(
        predictor,
        successful=False,
        reasons=reasons_,
        calibration_case_ids=ids,
        numerical_status=numerical_status,
        campaign_id=campaign_id,
        cohort_id=cohort_id,
        source_id=source_id,
        transform_id=transform_id,
        training_source_manifest_ids=training_source_manifest_ids,
    )
    return ProteinStabilityModelFit(
        predictor,
        False,
        reasons_,
        ids,
        numerical_status,
        campaign_id,
        cohort_id,
        source_id,
        transform_id,
        tuple(training_source_manifest_ids),
        fit_id,
    )


def _successful_fit(
    predictor: AbstractProteinStabilityPredictor | RegularizedPairInteractionModel,
    measurements: Sequence[ProteinStabilityMeasurement],
    numerical_status: int,
    /,
    *,
    campaign_id: str,
    cohort_id: str | None,
    source_id: str | None,
    transform_id: str,
    training_source_manifest_ids: Sequence[str],
) -> ProteinStabilityModelFit:
    ids = tuple(sorted(item.measurement_id for item in measurements))
    fit_id = _protein_stability_fit_id(
        predictor,
        successful=True,
        reasons=(),
        calibration_case_ids=ids,
        numerical_status=numerical_status,
        campaign_id=campaign_id,
        cohort_id=cohort_id,
        source_id=source_id,
        transform_id=transform_id,
        training_source_manifest_ids=training_source_manifest_ids,
    )
    return ProteinStabilityModelFit(
        predictor,
        True,
        (),
        ids,
        numerical_status,
        campaign_id,
        cohort_id,
        source_id,
        transform_id,
        tuple(training_source_manifest_ids),
        fit_id,
    )


__all__ = [
    "AbstractProteinStabilityPredictor",
    "DoubleMutantCase",
    "GlobalSubstitutionBaseline",
    "PredictionValidity",
    "ProteinStabilityModelFit",
    "ProteinStabilityModelSelectionRecord",
    "ProteinStabilityPredictor",
    "RegularizedEnvironmentModel",
    "RegularizedPairInteractionModel",
    "StabilityPrediction",
    "UncertaintyComponent",
    "fit_global_substitution_baseline",
    "fit_regularized_environment_model",
    "fit_regularized_pair_interaction_model",
]
