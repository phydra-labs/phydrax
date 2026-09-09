#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Calibrated raw plasmid-gel observation law for initial topology forms."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.qualification import ReferenceArtifactManifest

from ._interactions import _text
from .interchange._history_profile import TimedRadiationHistoryProfile


PLASMID_FORMS = ("supercoiled", "open-circle", "linear")


@dataclass(frozen=True, slots=True, init=False)
class PlasmidGelAssay:
    """Band intensity conditional on topology fractions and measured lane gain.

    The response matrix maps true plasmid forms (columns) to reported bands (rows).
    It may encode calibrated staining and band cross-talk. It is not inferred from
    lesion-site union probabilities and does not turn a site expectation into DSBs.
    """

    response_matrix: Array
    background: Array
    calibration_covariance: Array | None
    calibration: ReferenceArtifactManifest
    assay_id: str

    def __init__(
        self,
        response_matrix: ArrayLike,
        background: ArrayLike,
        calibration: ReferenceArtifactManifest,
        /,
        *,
        calibration_covariance: ArrayLike | None = None,
    ):
        response = np.asarray(response_matrix, dtype=float)
        offset = np.asarray(background, dtype=float)
        if (
            response.shape != (3, 3)
            or not np.all(np.isfinite(response))
            or np.any(response < 0.0)
            or np.any(np.sum(response, axis=0) <= 0.0)
        ):
            raise ValueError(
                "Gel response must be a finite nonnegative 3x3 matrix with visible forms."
            )
        if (
            offset.shape != (3,)
            or not np.all(np.isfinite(offset))
            or np.any(offset < 0.0)
        ):
            raise ValueError("Gel background must contain three nonnegative intensities.")
        if not isinstance(calibration, ReferenceArtifactManifest):
            raise TypeError("Gel calibration must be a ReferenceArtifactManifest.")
        calibration.require_rights()
        calibration.require_uncertainty()
        covariance = (
            None
            if calibration_covariance is None
            else np.asarray(calibration_covariance, dtype=float)
        )
        if covariance is not None and (
            covariance.shape != (12, 12)
            or not np.all(np.isfinite(covariance))
            or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12)
            or np.min(np.linalg.eigvalsh(covariance)) < -1e-12
        ):
            raise ValueError(
                "Gel calibration covariance must be a finite positive-semidefinite "
                "12x12 matrix over response and background parameters."
            )
        object.__setattr__(self, "response_matrix", jnp.asarray(response))
        object.__setattr__(self, "background", jnp.asarray(offset))
        object.__setattr__(
            self,
            "calibration_covariance",
            None if covariance is None else jnp.asarray(covariance),
        )
        object.__setattr__(self, "calibration", calibration)
        object.__setattr__(
            self,
            "assay_id",
            canonical_fingerprint(
                {
                    "kind": "plasmid-gel-assay",
                    "forms": PLASMID_FORMS,
                    "response": response.tolist(),
                    "background": offset.tolist(),
                    "calibration": calibration.manifest_id,
                    "calibration_covariance": (
                        None if covariance is None else array_tree_fingerprint(covariance)
                    ),
                }
            ),
        )

    def expected_intensity(
        self, form_fractions: ArrayLike, lane_gain: ArrayLike, /
    ) -> Array:
        fractions = jnp.asarray(form_fractions, dtype=self.response_matrix.dtype)
        gain = jnp.asarray(lane_gain, dtype=self.response_matrix.dtype)
        fraction_values = np.asarray(fractions)
        gain_values = np.asarray(gain)
        if (
            fractions.shape[-1:] != (3,)
            or not np.all(np.isfinite(fraction_values))
            or np.any(fraction_values < 0.0)
            or not np.allclose(
                np.sum(fraction_values, axis=-1), 1.0, rtol=0.0, atol=1e-10
            )
        ):
            raise ValueError(
                "Form fractions must be finite supercoiled/open-circle/linear probability rows."
            )
        if (
            gain.shape != fractions.shape[:-1]
            or not np.all(np.isfinite(gain_values))
            or np.any(gain_values <= 0.0)
        ):
            raise ValueError("Lane gain must contain one finite positive value per row.")
        return self.background + gain[..., None] * (fractions @ self.response_matrix.T)

    def calibration_intensity_covariance(
        self,
        form_fractions: ArrayLike,
        lane_gain: ArrayLike,
        lane_gain_standard_errors: ArrayLike,
        /,
    ) -> Array:
        """Propagate response/background and lane-gain calibration uncertainty."""

        if self.calibration_covariance is None:
            raise ValueError("Gel response/background calibration covariance is unknown.")
        fractions = jnp.asarray(form_fractions, dtype=self.response_matrix.dtype)
        gain = jnp.asarray(lane_gain, dtype=self.response_matrix.dtype)
        gain_errors = jnp.asarray(
            lane_gain_standard_errors, dtype=self.response_matrix.dtype
        )
        self.expected_intensity(fractions, gain)
        gain_error_values = np.asarray(gain_errors)
        if (
            gain_errors.shape != gain.shape
            or not np.all(np.isfinite(gain_error_values))
            or np.any(gain_error_values < 0.0)
        ):
            raise ValueError(
                "Lane-gain uncertainty must be finite, nonnegative, and aligned."
            )
        parameters = jnp.concatenate((self.response_matrix.reshape(-1), self.background))

        def calibrated_mean(parameter_vector):
            response = parameter_vector[:9].reshape((3, 3))
            background = parameter_vector[9:]
            return background + gain[..., None] * (fractions @ response.T)

        sensitivity = jax.jacfwd(calibrated_mean)(parameters)
        response_covariance = jnp.einsum(
            "...oi,ij,...pj->...op",
            sensitivity,
            self.calibration_covariance,
            sensitivity,
        )
        lane_sensitivity = fractions @ self.response_matrix.T
        lane_covariance = (
            gain_errors[..., None, None] ** 2
            * lane_sensitivity[..., :, None]
            * lane_sensitivity[..., None, :]
        )
        return response_covariance + lane_covariance


@dataclass(frozen=True, slots=True, init=False)
class PlasmidGelObservations:
    """Raw band intensities grouped by irradiation day and physical tuple."""

    observation_ids: tuple[str, ...]
    preparation_ids: tuple[str, ...]
    irradiation_day_ids: tuple[str, ...]
    physical_tuple_ids: tuple[str, ...]
    lane_gain: Array
    lane_gain_standard_errors: Array | None
    intensities: Array
    standard_errors: Array
    observation_covariance: Array
    source: ReferenceArtifactManifest
    observation_id: str

    def __init__(
        self,
        observation_ids: tuple[str, ...],
        irradiation_day_ids: tuple[str, ...],
        physical_tuple_ids: tuple[str, ...],
        /,
        *,
        preparation_ids: tuple[str, ...],
        lane_gain: ArrayLike,
        intensities: ArrayLike,
        standard_errors: ArrayLike,
        source: ReferenceArtifactManifest,
        lane_gain_standard_errors: ArrayLike | None = None,
        observation_covariance: ArrayLike | None = None,
    ):
        identifiers = tuple(observation_ids)
        preparations = tuple(preparation_ids)
        days = tuple(irradiation_day_ids)
        tuples = tuple(physical_tuple_ids)
        n = len(identifiers)
        if (
            not n
            or len(set(identifiers)) != n
            or len(preparations) != n
            or len(days) != n
            or len(tuples) != n
        ):
            raise ValueError(
                "Unique observation IDs and aligned preparation/day/physical tuples "
                "are required."
            )
        for value in (*identifiers, *preparations, *days, *tuples):
            _text(value, "plasmid-gel identity")
        gain = np.asarray(lane_gain, dtype=float)
        gain_errors = (
            None
            if lane_gain_standard_errors is None
            else np.asarray(lane_gain_standard_errors, dtype=float)
        )
        values = np.asarray(intensities, dtype=float)
        errors = np.asarray(standard_errors, dtype=float)
        if gain.shape != (n,) or not np.all(np.isfinite(gain)) or np.any(gain <= 0.0):
            raise ValueError("Each gel lane requires a finite positive calibrated gain.")
        if gain_errors is not None and (
            gain_errors.shape != gain.shape
            or not np.all(np.isfinite(gain_errors))
            or np.any(gain_errors <= 0.0)
        ):
            raise ValueError(
                "Known lane-gain standard errors must be finite, positive, and aligned."
            )
        if (
            values.shape != (n, 3)
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise ValueError(
                "Raw gel intensities must be finite nonnegative (lane, 3) values."
            )
        if (
            errors.shape != values.shape
            or not np.all(np.isfinite(errors))
            or np.any(errors <= 0.0)
        ):
            raise ValueError(
                "Every gel-band intensity requires positive measured uncertainty."
            )
        error_covariance = (
            np.stack(tuple(np.diag(row**2) for row in errors))
            if observation_covariance is None
            else np.asarray(observation_covariance, dtype=float)
        )
        if (
            error_covariance.shape != (n, 3, 3)
            or not np.all(np.isfinite(error_covariance))
            or not np.allclose(
                error_covariance,
                np.swapaxes(error_covariance, -1, -2),
                rtol=1e-12,
                atol=1e-12,
            )
            or np.any(np.linalg.eigvalsh(error_covariance) < -1e-12)
            or not np.allclose(
                np.diagonal(error_covariance, axis1=-2, axis2=-1),
                errors**2,
                rtol=1e-10,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "Gel observation covariance must be finite positive-semidefinite "
                "with diagonal matching the reported band standard errors."
            )
        if not isinstance(source, ReferenceArtifactManifest):
            raise TypeError("Gel observations require a ReferenceArtifactManifest.")
        source.require_rights()
        source.require_uncertainty()
        object.__setattr__(self, "observation_ids", identifiers)
        object.__setattr__(self, "preparation_ids", preparations)
        object.__setattr__(self, "irradiation_day_ids", days)
        object.__setattr__(self, "physical_tuple_ids", tuples)
        object.__setattr__(self, "lane_gain", jnp.asarray(gain))
        object.__setattr__(
            self,
            "lane_gain_standard_errors",
            None if gain_errors is None else jnp.asarray(gain_errors),
        )
        object.__setattr__(self, "intensities", jnp.asarray(values))
        object.__setattr__(self, "standard_errors", jnp.asarray(errors))
        object.__setattr__(self, "observation_covariance", jnp.asarray(error_covariance))
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "observation_id",
            canonical_fingerprint(
                {
                    "kind": "plasmid-gel-observations",
                    "observations": identifiers,
                    "preparations": preparations,
                    "days": days,
                    "physical_tuples": tuples,
                    "lane_gain": gain.tolist(),
                    "lane_gain_standard_errors": (
                        None
                        if gain_errors is None
                        else array_tree_fingerprint(gain_errors)
                    ),
                    "intensities": array_tree_fingerprint(values),
                    "standard_errors": array_tree_fingerprint(errors),
                    "observation_covariance": array_tree_fingerprint(error_covariance),
                    "source": source.manifest_id,
                    "forms": PLASMID_FORMS,
                }
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class PlasmidFormPrediction:
    """Frozen topology fractions with exact model, fit, campaign, and data lineage."""

    campaign_id: str
    model_id: str
    fit_id: str
    prediction_source_artifact_id: str
    history_profile_id: str
    physical_tuple_ids: tuple[str, ...]
    fit_physical_tuple_ids: tuple[str, ...]
    fit_independent_unit_ids: tuple[str, ...]
    fit_preparation_ids: tuple[str, ...]
    form_fractions: Array
    form_fraction_covariance: Array | None
    prediction_id: str

    def __init__(
        self,
        history_profile: TimedRadiationHistoryProfile,
        physical_tuple_ids: tuple[str, ...],
        form_fractions: ArrayLike,
        /,
        *,
        campaign_id: str,
        model_id: str,
        fit_id: str,
        prediction_source_artifact_id: str,
        fit_physical_tuple_ids: tuple[str, ...],
        fit_independent_unit_ids: tuple[str, ...],
        fit_preparation_ids: tuple[str, ...],
        form_fraction_covariance: ArrayLike | None = None,
    ):
        if not isinstance(history_profile, TimedRadiationHistoryProfile):
            raise TypeError("history_profile must be TimedRadiationHistoryProfile.")
        for value, name in (
            (campaign_id, "prediction campaign ID"),
            (model_id, "prediction model ID"),
            (fit_id, "prediction fit ID"),
            (prediction_source_artifact_id, "prediction source artifact ID"),
        ):
            _text(value, name)
        campaign = campaign_id
        model = model_id
        fit = fit_id
        source = prediction_source_artifact_id
        tuples = tuple(physical_tuple_ids)
        fit_tuples = tuple(fit_physical_tuple_ids)
        fit_units = tuple(fit_independent_unit_ids)
        fit_preparations = tuple(fit_preparation_ids)
        if (
            not tuples
            or not fit_tuples
            or len(set(fit_tuples)) != len(fit_tuples)
            or not fit_units
            or len(set(fit_units)) != len(fit_units)
            or not fit_preparations
            or len(set(fit_preparations)) != len(fit_preparations)
        ):
            raise ValueError(
                "Predictions require nonempty tuple rows and unique fit independent "
                "units, history tuples, and preparations."
            )
        for value in (*tuples, *fit_tuples, *fit_units, *fit_preparations):
            _text(value, "plasmid form prediction identity")
        profile_tuples = set(history_profile.physical_tuple_ids)
        if not set((*tuples, *fit_tuples)) <= profile_tuples:
            raise ValueError(
                "Prediction and fit physical tuples must belong to the history profile."
            )
        fractions = np.asarray(form_fractions, dtype=float)
        if (
            fractions.shape != (len(tuples), 3)
            or not np.all(np.isfinite(fractions))
            or np.any(fractions < 0.0)
            or not np.allclose(np.sum(fractions, axis=-1), 1.0, rtol=0.0, atol=1e-10)
        ):
            raise ValueError(
                "Predicted topology fractions must be finite probability rows."
            )
        object.__setattr__(self, "campaign_id", campaign)
        fraction_covariance = (
            None
            if form_fraction_covariance is None
            else np.asarray(form_fraction_covariance, dtype=float)
        )
        if fraction_covariance is not None and (
            fraction_covariance.shape != (len(tuples), 3, 3)
            or not np.all(np.isfinite(fraction_covariance))
            or not np.allclose(
                fraction_covariance,
                np.swapaxes(fraction_covariance, -1, -2),
                rtol=1e-12,
                atol=1e-12,
            )
            or np.any(np.linalg.eigvalsh(fraction_covariance) < -1e-12)
            or not np.allclose(
                fraction_covariance @ np.ones(3),
                0.0,
                rtol=1e-10,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "Form-fraction covariance must be finite positive-semidefinite, "
                "aligned by tuple, and preserve unit-sum topology fractions."
            )
        object.__setattr__(self, "model_id", model)
        object.__setattr__(self, "fit_id", fit)
        object.__setattr__(self, "prediction_source_artifact_id", source)
        object.__setattr__(self, "history_profile_id", history_profile.profile_id)
        object.__setattr__(self, "physical_tuple_ids", tuples)
        object.__setattr__(self, "fit_physical_tuple_ids", tuple(sorted(fit_tuples)))
        object.__setattr__(self, "fit_independent_unit_ids", tuple(sorted(fit_units)))
        object.__setattr__(self, "fit_preparation_ids", tuple(sorted(fit_preparations)))
        object.__setattr__(self, "form_fractions", jnp.asarray(fractions))
        object.__setattr__(
            self,
            "form_fraction_covariance",
            None if fraction_covariance is None else jnp.asarray(fraction_covariance),
        )
        object.__setattr__(
            self,
            "prediction_id",
            canonical_fingerprint(
                {
                    "kind": "plasmid-form-prediction",
                    "campaign": campaign,
                    "model": model,
                    "fit": fit,
                    "source_artifact": source,
                    "history_profile": history_profile.profile_id,
                    "physical_tuples": tuples,
                    "fit_physical_tuples": tuple(sorted(fit_tuples)),
                    "fit_independent_units": tuple(sorted(fit_units)),
                    "fit_preparations": tuple(sorted(fit_preparations)),
                    "form_fractions": array_tree_fingerprint(fractions),
                    "form_fraction_covariance": (
                        None
                        if fraction_covariance is None
                        else array_tree_fingerprint(fraction_covariance)
                    ),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class PlasmidGelEvaluation:
    assay_id: str
    observation_id: str
    campaign_id: str
    model_id: str
    fit_id: str
    prediction_id: str
    prediction_source_artifact_id: str
    history_profile_id: str
    physical_tuple_ids: tuple[str, ...]
    predicted_intensities: Array
    combined_standard_errors: Array
    standardized_residuals: Array
    day_macro_standardized_rms: float
    uncertainty_limitations: tuple[str, ...]
    finite: bool
    evaluation_id: str


def evaluate_plasmid_gel(
    assay: PlasmidGelAssay,
    observations: PlasmidGelObservations,
    prediction: PlasmidFormPrediction,
    /,
) -> PlasmidGelEvaluation:
    """Compare profile-bound form fractions with raw held-out gel bands."""

    if not isinstance(assay, PlasmidGelAssay) or not isinstance(
        observations, PlasmidGelObservations
    ):
        raise TypeError("Gel evaluation requires a prepared assay and observations.")
    if not isinstance(prediction, PlasmidFormPrediction):
        raise TypeError("prediction must be a profile-bound PlasmidFormPrediction.")
    if prediction.physical_tuple_ids != observations.physical_tuple_ids:
        raise ValueError(
            "Predicted physical tuples must align exactly with gel observations."
        )
    if set(prediction.fit_physical_tuple_ids) & set(observations.physical_tuple_ids):
        raise ValueError("Locked gel physical tuples overlap prediction fit tuples.")
    if set(prediction.fit_preparation_ids) & set(observations.preparation_ids):
        raise ValueError("Locked gel preparations overlap prediction fit preparations.")
    if set(prediction.fit_independent_unit_ids) & set(observations.irradiation_day_ids):
        raise ValueError(
            "Locked gel independent units overlap prediction fit independent units."
        )
    fractions = np.asarray(prediction.form_fractions)
    predicted = assay.expected_intensity(fractions, observations.lane_gain)
    limitations = []
    calibration_covariance = np.zeros((*fractions.shape[:-1], 3, 3), dtype=float)
    lane_gain_errors = observations.lane_gain_standard_errors
    if assay.calibration_covariance is None:
        limitations.append("gel-response-background-calibration-covariance")
    else:
        gain_errors = (
            np.zeros(np.asarray(observations.lane_gain).shape, dtype=float)
            if lane_gain_errors is None
            else np.asarray(lane_gain_errors)
        )
        calibration_covariance += np.asarray(
            assay.calibration_intensity_covariance(
                fractions,
                observations.lane_gain,
                gain_errors,
            )
        )
    if lane_gain_errors is None:
        limitations.append("gel-lane-gain-uncertainty")
    elif assay.calibration_covariance is None:
        lane_sensitivity = fractions @ np.asarray(assay.response_matrix).T
        calibration_covariance += (
            np.asarray(lane_gain_errors)[..., None, None] ** 2
            * lane_sensitivity[..., :, None]
            * lane_sensitivity[..., None, :]
        )
    if prediction.form_fraction_covariance is None:
        limitations.append("radiation-form-prediction-covariance")
    else:
        fraction_response = (
            np.asarray(observations.lane_gain)[..., None, None]
            * np.asarray(assay.response_matrix)[None, ...]
        )
        calibration_covariance += np.einsum(
            "...oi,...ij,...pj->...op",
            fraction_response,
            np.asarray(prediction.form_fraction_covariance),
            fraction_response,
        )
    total_covariance = (
        np.asarray(observations.observation_covariance) + calibration_covariance
    )
    combined_standard_errors = np.sqrt(
        np.maximum(
            np.diagonal(total_covariance, axis1=-2, axis2=-1),
            0.0,
        )
    )
    raw_residuals = np.asarray(predicted) - np.asarray(observations.intensities)
    residuals = np.zeros_like(raw_residuals, dtype=float)
    covariance_valid = True
    for lane in range(raw_residuals.shape[0]):
        covariance = total_covariance[lane]
        if not np.all(np.isfinite(covariance)) or not np.allclose(
            covariance, covariance.T, rtol=1e-10, atol=1e-12
        ):
            covariance_valid = False
            limitations.append(f"invalid-gel-observation-covariance:lane-{lane}")
            continue
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        tolerance = 1e-12 * scale
        if float(eigenvalues[0]) < -tolerance:
            covariance_valid = False
            limitations.append(f"invalid-gel-observation-covariance:lane-{lane}")
            continue
        if float(eigenvalues[0]) <= tolerance:
            covariance_valid = False
            limitations.append(f"singular-gel-observation-covariance:lane-{lane}")
            continue
        residuals[lane] = (eigenvectors.T @ raw_residuals[lane]) / np.sqrt(eigenvalues)
    finite = bool(
        covariance_valid
        and np.all(np.isfinite(np.asarray(predicted)))
        and np.all(np.isfinite(combined_standard_errors))
        and np.all(combined_standard_errors > 0.0)
        and np.all(np.isfinite(residuals))
    )
    per_day = []
    for day in dict.fromkeys(observations.irradiation_day_ids):
        selected = np.asarray(observations.irradiation_day_ids) == day
        per_day.append(float(np.sqrt(np.mean(np.square(residuals[selected])))))
    macro = float(np.mean(per_day)) if finite else float("inf")
    uncertainty_limitations = tuple(limitations)
    evaluation_id = canonical_fingerprint(
        {
            "kind": "plasmid-gel-evaluation",
            "assay": assay.assay_id,
            "observations": observations.observation_id,
            "prediction": prediction.prediction_id,
            "campaign": prediction.campaign_id,
            "model": prediction.model_id,
            "fit": prediction.fit_id,
            "prediction_source_artifact": prediction.prediction_source_artifact_id,
            "history_profile": prediction.history_profile_id,
            "physical_tuples": prediction.physical_tuple_ids,
            "fractions": array_tree_fingerprint(fractions),
            "combined_standard_errors": array_tree_fingerprint(combined_standard_errors),
            "uncertainty_limitations": uncertainty_limitations,
            "day_macro_standardized_rms": float(macro).hex(),
            "finite": finite,
        }
    )
    return PlasmidGelEvaluation(
        assay.assay_id,
        observations.observation_id,
        prediction.campaign_id,
        prediction.model_id,
        prediction.fit_id,
        prediction.prediction_id,
        prediction.prediction_source_artifact_id,
        prediction.history_profile_id,
        prediction.physical_tuple_ids,
        predicted,
        jnp.asarray(combined_standard_errors),
        jnp.asarray(residuals),
        macro,
        uncertainty_limitations,
        finite,
        evaluation_id,
    )


__all__ = [
    "PlasmidFormPrediction",
    "PLASMID_FORMS",
    "PlasmidGelAssay",
    "PlasmidGelEvaluation",
    "PlasmidGelObservations",
    "evaluate_plasmid_gel",
]
