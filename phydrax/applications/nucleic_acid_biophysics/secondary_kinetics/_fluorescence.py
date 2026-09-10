#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Raw-fluorescence observation and strand-displacement forward models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

import phydrax.linalg as la
from phydrax import ein

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....qualification import ReferenceArtifactManifest, ScientificCampaign
from ....uq import (
    AbstractPosteriorTerm,
    find_map,
    fit_laplace,
    LaplaceResult,
    MAPResult,
    MCMCResult,
    ParameterSpace,
    PosteriorProblem,
)
from ..interchange._strand_displacement import FluorescenceTimeTrace
from ._compile import CompiledSecondaryTarget, PreparedSecondaryKinetics
from ._state import SecondaryStructureState


_PARAMETER_NAME = "rate_constant_per_molar_second"
_AVOGADRO_CONSTANT_PER_MOL = 6.02214076e23
_RIGHTS_KEYS = frozenset(("commercial_use", "redistribution", "training_use", "export"))


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _identifiers(
    values: Sequence[str], name: str, /, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of identifiers.")
    result = tuple(_identifier(value, name) for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique identifiers.")
    return tuple(sorted(result))


def _ordered_identifiers(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of identifiers.")
    result = tuple(_identifier(value, name) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be non-empty and unique.")
    return result


def _positive(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _role_cases(campaign: ScientificCampaign, names: Sequence[str], /) -> frozenset[str]:
    selected = frozenset(names)
    return frozenset(
        case_id
        for role in campaign.roles
        if role.name in selected
        for case_id in role.case_ids
    )


def _validate_campaign_role_traces(
    traces: Sequence[FluorescenceTimeTrace],
    campaign: ScientificCampaign,
    role: str,
    /,
) -> tuple[FluorescenceTimeTrace, ...]:
    if not isinstance(campaign, ScientificCampaign):
        raise TypeError("campaign must be a ScientificCampaign.")
    values = tuple(traces)
    if not values or any(
        not isinstance(trace, FluorescenceTimeTrace) for trace in values
    ):
        raise ValueError(f"{role} requires admitted raw fluorescence traces.")
    expected_ids = _role_cases(campaign, (role,))
    actual_ids = {trace.case_id for trace in values}
    if len(actual_ids) != len(values) or actual_ids != expected_ids:
        raise ValueError(
            f"{role} traces must exactly equal the campaign's frozen {role} role."
        )
    cases = {case.case_id: case for case in campaign.cases}
    for trace in values:
        case = cases[trace.case_id]
        if (
            trace.sequence_family_id != case.independent_unit_id
            or canonical_fingerprint(trace.construct_ids) != case.construct_id
            or trace.condition_id != case.condition_id
            or trace.identity.preparation_id != case.preparation_id
            or trace.identity.plate_id != case.batch_id
            or trace.source_manifest_ids != case.source_manifest_ids
        ):
            raise ValueError(
                f"Trace {trace.case_id!r} does not match its frozen campaign case."
            )
    return tuple(sorted(values, key=lambda trace: trace.case_id))


def _admitted_source_manifests(
    source_manifests: Sequence[ReferenceArtifactManifest],
    requested_use: Mapping[str, bool],
    expected_ids: Sequence[str],
    owner: str,
    /,
) -> tuple[
    tuple[ReferenceArtifactManifest, ...],
    tuple[tuple[str, bool], ...],
]:
    values = tuple(source_manifests)
    if not values or any(
        not isinstance(value, ReferenceArtifactManifest) for value in values
    ):
        raise TypeError(f"{owner} sources must be admitted manifests.")
    if (
        not isinstance(requested_use, Mapping)
        or set(requested_use) != _RIGHTS_KEYS
        or any(type(value) is not bool for value in requested_use.values())
    ):
        raise ValueError(f"{owner} must declare all four requested-use rights.")
    by_id = {manifest.manifest_id: manifest for manifest in values}
    if len(by_id) != len(values) or set(by_id) != set(expected_ids):
        raise ValueError(f"{owner} manifests must exactly match frozen source IDs.")
    for manifest in values:
        manifest.require_rights(**dict(requested_use))
    ordered = tuple(by_id[manifest_id] for manifest_id in sorted(by_id))
    return ordered, tuple(sorted(requested_use.items()))


class ReporterCalibration(StrictModule, NonTrainableState):
    """Reporter gain/background/delay fitted only to calibration-role cases.

    Covariance parameter order is background, gain, then the optional single
    reporter response time constant.  ``None`` preserves unknown calibration
    uncertainty; it is never interpreted as zero.
    """

    gain: Array
    background: Array
    delay_parameters: Array
    covariance: Array | None
    reporter_id: str = eqx.field(static=True)
    calibration_case_ids: tuple[str, ...] = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)
    source_manifest_ids: tuple[str, ...] = eqx.field(static=True)
    source_manifests: tuple[ReferenceArtifactManifest, ...] = eqx.field(static=True)
    requested_use: tuple[tuple[str, bool], ...] = eqx.field(static=True)
    calibration_id: str = eqx.field(static=True)

    def __init__(
        self,
        reporter_id: str,
        gain: ArrayLike,
        background: ArrayLike,
        delay_parameters: ArrayLike,
        covariance: ArrayLike | None,
        calibration_case_ids: Sequence[str],
        campaign: ScientificCampaign,
        /,
        *,
        source_manifests: Sequence[ReferenceArtifactManifest],
        requested_use: Mapping[str, bool],
    ):
        if not isinstance(campaign, ScientificCampaign):
            raise TypeError("campaign must be a ScientificCampaign.")
        gain_ = jnp.asarray(gain, dtype=float)
        background_ = jnp.asarray(background, dtype=float)
        delay = jnp.asarray(delay_parameters, dtype=float)
        if gain_.shape != () or background_.shape != ():
            raise ValueError("Reporter v1 requires scalar gain and background.")
        if not bool(jnp.isfinite(gain_)) or float(gain_) <= 0.0:
            raise ValueError("Reporter gain must be finite and positive.")
        if not bool(jnp.isfinite(background_)):
            raise ValueError("Reporter background must be finite.")
        if delay.ndim != 1 or delay.size not in (0, 1):
            raise ValueError(
                "Reporter delay parameters must be empty or one time constant."
            )
        if bool(jnp.any(~jnp.isfinite(delay) | (delay <= 0.0))):
            raise ValueError("Reporter delay time constant must be finite and positive.")
        cases = _identifiers(calibration_case_ids, "calibration_case_ids")
        allowed = _role_cases(campaign, ("calibration",))
        if not set(cases) <= allowed:
            raise ValueError("Reporter calibration may use calibration-role cases only.")
        manifests_ = tuple(source_manifests)
        if not manifests_ or any(
            not isinstance(value, ReferenceArtifactManifest) for value in manifests_
        ):
            raise TypeError("Reporter calibration sources must be admitted manifests.")
        if (
            not isinstance(requested_use, Mapping)
            or set(requested_use) != _RIGHTS_KEYS
            or any(type(value) is not bool for value in requested_use.values())
        ):
            raise ValueError(
                "Declare all four reporter-calibration requested-use rights explicitly."
            )
        for manifest in manifests_:
            manifest.require_rights(**dict(requested_use))
        manifest_by_id = {manifest.manifest_id: manifest for manifest in manifests_}
        expected_source_ids = {
            source_id
            for case in campaign.cases
            if case.case_id in cases
            for source_id in case.source_manifest_ids
        }
        if set(manifest_by_id) != expected_source_ids:
            raise ValueError(
                "Reporter calibration manifests must exactly match calibration cases."
            )
        manifests = tuple(sorted(manifest_by_id))
        covariance_ = None if covariance is None else jnp.asarray(covariance, dtype=float)
        parameter_count = 2 + int(delay.size)
        if covariance_ is not None:
            if covariance_.shape != (parameter_count, parameter_count):
                raise ValueError(
                    "Reporter covariance shape must match background/gain/delay parameters."
                )
            host_covariance = np.asarray(covariance_)
            if (
                np.any(~np.isfinite(host_covariance))
                or not np.allclose(
                    host_covariance, host_covariance.T, rtol=1e-12, atol=1e-12
                )
                or np.min(np.linalg.eigvalsh(host_covariance)) < -1e-12
            ):
                raise ValueError(
                    "Reporter covariance must be finite, symmetric and positive semidefinite."
                )
        self.gain = gain_
        self.background = background_
        self.delay_parameters = delay
        self.covariance = covariance_
        self.reporter_id = _identifier(reporter_id, "reporter_id")
        self.calibration_case_ids = cases
        self.campaign_id = campaign.campaign_id
        self.source_manifests = tuple(
            manifest_by_id[manifest_id] for manifest_id in manifests
        )
        self.source_manifest_ids = manifests
        self.requested_use = tuple(sorted(requested_use.items()))
        self.calibration_id = canonical_fingerprint(
            {
                "kind": "reporter-calibration",
                "reporter": self.reporter_id,
                "campaign": self.campaign_id,
                "cases": cases,
                "sources": manifests,
                "requested_use": dict(self.requested_use),
                "parameters": array_tree_fingerprint(
                    (gain_, background_, delay, covariance_)
                )["sha256"],
                "covariance_parameter_order": ["background", "gain", "delay_seconds"],
            }
        )


class ReporterObservationModel(StrictModule, NonTrainableState):
    """A calibrated fluorescence law with independently estimated residual noise."""

    calibration: ReporterCalibration
    noise_standard_deviation_intensity: Array
    noise_basis_case_ids: tuple[str, ...] = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)
    intensity_unit_id: str = eqx.field(static=True)
    observation_model_id: str = eqx.field(static=True)
    uncertainty_limitations: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        calibration: ReporterCalibration,
        noise_standard_deviation_intensity: float,
        noise_basis_case_ids: Sequence[str],
        campaign: ScientificCampaign,
        /,
        *,
        intensity_unit_id: str = "instrument-fluorescence-unit",
    ):
        if not isinstance(calibration, ReporterCalibration):
            raise TypeError("calibration must be a ReporterCalibration.")
        if not isinstance(campaign, ScientificCampaign):
            raise TypeError("campaign must be a ScientificCampaign.")
        if calibration.campaign_id != campaign.campaign_id:
            raise ValueError(
                "Reporter calibration and observation model must share one campaign."
            )
        sigma = _positive(
            noise_standard_deviation_intensity,
            "noise_standard_deviation_intensity",
        )
        cases = _identifiers(noise_basis_case_ids, "noise_basis_case_ids")
        allowed = _role_cases(campaign, ("calibration", "interval_calibration"))
        if not set(cases) <= allowed:
            raise ValueError(
                "Observation noise may use calibration or interval-calibration cases only."
            )
        self.calibration = calibration
        self.noise_standard_deviation_intensity = jnp.asarray(sigma)
        self.campaign_id = campaign.campaign_id
        self.noise_basis_case_ids = cases
        self.intensity_unit_id = _identifier(intensity_unit_id, "intensity_unit_id")
        self.uncertainty_limitations = (
            ("reporter-calibration-parameter-uncertainty-unquantified",)
            if calibration.covariance is None
            else ()
        )
        self.observation_model_id = canonical_fingerprint(
            {
                "kind": "strand-displacement-reporter-observation-model",
                "calibration": calibration.calibration_id,
                "noise_sigma": sigma,
                "noise_basis_cases": cases,
                "campaign": self.campaign_id,
                "intensity_unit": self.intensity_unit_id,
            }
        )

    @property
    def includes_calibration_uncertainty(self) -> bool:
        return self.calibration.covariance is not None

    def _reported_product(self, time_seconds: Array, product_molar: Array) -> Array:
        if self.calibration.delay_parameters.size == 0:
            return product_molar
        tau = self.calibration.delay_parameters[0]
        first_time = jnp.maximum(time_seconds[0], 0.0)
        first = product_molar[0] * (1.0 - jnp.exp(-first_time / tau))

        def advance(previous, values):
            product, delta_time = values
            reported = product + (previous - product) * jnp.exp(-delta_time / tau)
            return reported, reported

        _, remainder = jax.lax.scan(
            advance,
            first,
            (product_molar[1:], jnp.diff(time_seconds)),
        )
        return jnp.concatenate((first[None], remainder))

    def mean_intensity(
        self, time_seconds: ArrayLike, product_molar: ArrayLike, /
    ) -> Array:
        time = jnp.asarray(time_seconds, dtype=float)
        product = jnp.asarray(product_molar, dtype=float)
        if time.ndim != 1 or product.shape != time.shape:
            raise ValueError(
                "Reporter input requires aligned one-dimensional time/product arrays."
            )
        return (
            self.calibration.background
            + self.calibration.gain * self._reported_product(time, product)
        )

    def prediction(
        self,
        trace: FluorescenceTimeTrace,
        product_molar: ArrayLike,
        /,
        *,
        forward_model_id: str,
        model_intensity_sensitivity: ArrayLike | None = None,
        model_parameter_covariance: ArrayLike | None = None,
        model_intensity_draws: ArrayLike | None = None,
        model_uncertainty_basis_id: str | None = None,
        model_uncertainty_limitation: str | None = None,
    ) -> FluorescencePrediction:
        if trace.reporter_id != self.calibration.reporter_id:
            raise ValueError("Trace reporter identity is outside this calibration.")
        if trace.intensity_unit_id != self.intensity_unit_id:
            raise ValueError("Trace intensity unit is outside this observation model.")
        product = jnp.asarray(product_molar, dtype=float)
        if product.shape != trace.time_seconds.shape:
            raise ValueError(
                "Product prediction must align exactly to the raw trace clock."
            )
        mean = self.mean_intensity(trace.time_seconds, product)
        residual_variance = jnp.full_like(
            mean, self.noise_standard_deviation_intensity**2
        )
        sensitivity_parts: list[Array] = []
        covariance_parts: list[Array] = []
        includes_model_uncertainty = False
        if model_intensity_draws is not None:
            if (
                model_intensity_sensitivity is not None
                or model_parameter_covariance is not None
            ):
                raise ValueError(
                    "Use posterior model draws or a model covariance, never both."
                )
            draws = jnp.asarray(model_intensity_draws, dtype=float)
            if draws.ndim != 2 or draws.shape[0] < 2 or draws.shape[1:] != mean.shape:
                raise ValueError(
                    "Model intensity draws require at least two aligned posterior draws."
                )
            if bool(jnp.any(~jnp.isfinite(draws))):
                raise ValueError("Model intensity draws must be finite.")
            mean = jnp.mean(draws, axis=0)
            sensitivity_parts.append(
                (draws - mean[None, :]).T / math.sqrt(draws.shape[0] - 1)
            )
            covariance_parts.append(jnp.eye(draws.shape[0], dtype=draws.dtype))
            includes_model_uncertainty = True
        elif (
            model_intensity_sensitivity is not None
            or model_parameter_covariance is not None
        ):
            if model_intensity_sensitivity is None or model_parameter_covariance is None:
                raise ValueError(
                    "Model covariance propagation requires covariance and sensitivity."
                )
            model_sensitivity = jnp.asarray(model_intensity_sensitivity, dtype=float)
            model_covariance = jnp.asarray(model_parameter_covariance, dtype=float)
            if (
                model_sensitivity.ndim != 2
                or model_sensitivity.shape[0] != mean.size
                or model_covariance.shape
                != (model_sensitivity.shape[1], model_sensitivity.shape[1])
                or bool(
                    jnp.any(
                        ~jnp.isfinite(model_sensitivity) | ~jnp.isfinite(model_covariance)
                    )
                )
            ):
                raise ValueError(
                    "Model covariance and full fluorescence sensitivity must align."
                )
            sensitivity_parts.append(model_sensitivity)
            covariance_parts.append(model_covariance)
            includes_model_uncertainty = True

        reporter_covariance = self.calibration.covariance
        if reporter_covariance is not None:
            delay_count = int(self.calibration.delay_parameters.size)
            parameters = jnp.concatenate(
                (
                    self.calibration.background.reshape((1,)),
                    self.calibration.gain.reshape((1,)),
                    self.calibration.delay_parameters,
                )
            )

            def response(parameter_vector):
                background, gain = parameter_vector[0], parameter_vector[1]
                if delay_count == 0:
                    reported = product
                else:
                    tau = parameter_vector[2]
                    first_time = jnp.maximum(trace.time_seconds[0], 0.0)
                    first = product[0] * (1.0 - jnp.exp(-first_time / tau))

                    def advance(previous, values):
                        current, delta_time = values
                        result = current + (previous - current) * jnp.exp(
                            -delta_time / tau
                        )
                        return result, result

                    _, remainder = jax.lax.scan(
                        advance,
                        first,
                        (product[1:], jnp.diff(trace.time_seconds)),
                    )
                    reported = jnp.concatenate((first[None], remainder))
                return background + gain * reported

            sensitivity_parts.append(jax.jacfwd(response)(parameters))
            covariance_parts.append(reporter_covariance)

        sensitivity = None
        covariance = None
        epistemic_variance = jnp.zeros_like(mean)
        if sensitivity_parts:
            sensitivity = jnp.concatenate(tuple(sensitivity_parts), axis=1)
            dimensions = tuple(int(value.shape[0]) for value in covariance_parts)
            covariance = jnp.zeros(
                (sum(dimensions), sum(dimensions)), dtype=sensitivity.dtype
            )
            offset = 0
            for value, dimension in zip(covariance_parts, dimensions, strict=True):
                covariance = covariance.at[
                    offset : offset + dimension, offset : offset + dimension
                ].set(value)
                offset += dimension
            epistemic_variance = ein.contract(
                "ti,ij,tj->t", sensitivity, covariance, sensitivity
            )
        standard_deviation = jnp.sqrt(
            jnp.maximum(residual_variance + epistemic_variance, 0.0)
        )
        limitations = tuple(
            (
                *self.uncertainty_limitations,
                *(
                    (model_uncertainty_limitation,)
                    if model_uncertainty_limitation
                    else ()
                ),
            )
        )
        basis_id = (
            None
            if model_uncertainty_basis_id is None
            else _identifier(model_uncertainty_basis_id, "model_uncertainty_basis_id")
        )
        return FluorescencePrediction(
            case_id=trace.case_id,
            trace_id=trace.trace_id,
            forward_model_id=_identifier(forward_model_id, "forward_model_id"),
            time_seconds=trace.time_seconds,
            mean_intensity=mean,
            standard_deviation_intensity=standard_deviation,
            residual_standard_deviation_intensity=self.noise_standard_deviation_intensity,
            epistemic_sensitivity=sensitivity,
            epistemic_parameter_covariance=covariance,
            lower_95_intensity=mean - 1.959963984540054 * standard_deviation,
            upper_95_intensity=mean + 1.959963984540054 * standard_deviation,
            intensity_unit_id=self.intensity_unit_id,
            time_unit_id="second",
            concentration_unit_id="molar",
            includes_model_uncertainty=includes_model_uncertainty,
            includes_reporter_uncertainty=self.includes_calibration_uncertainty,
            uncertainty_limitations=limitations,
            uncertainty_basis_id=canonical_fingerprint(
                {
                    "observation_model": self.observation_model_id,
                    "noise_basis_cases": self.noise_basis_case_ids,
                    "model_uncertainty_basis": basis_id,
                    "model_uncertainty_propagation": (
                        "posterior-draws"
                        if model_intensity_draws is not None
                        else "linearized-covariance"
                        if model_parameter_covariance is not None
                        else "unavailable"
                    ),
                    "reporter_covariance": reporter_covariance is not None,
                    "uncertainty_limitations": limitations,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class FluorescencePrediction:
    """Raw time-trace prediction with model, reporter, and residual uncertainty."""

    case_id: str
    trace_id: str
    forward_model_id: str
    time_seconds: Array
    mean_intensity: Array
    standard_deviation_intensity: Array
    residual_standard_deviation_intensity: Array
    epistemic_sensitivity: Array | None
    epistemic_parameter_covariance: Array | None
    lower_95_intensity: Array
    upper_95_intensity: Array
    intensity_unit_id: str
    time_unit_id: str
    concentration_unit_id: str
    includes_model_uncertainty: bool
    includes_reporter_uncertainty: bool
    uncertainty_limitations: tuple[str, ...]
    uncertainty_basis_id: str


def _bimolecular_product_concentration(
    first: Array, second: Array, rate: Array, time: Array, /
) -> Array:
    """Evaluate irreversible A+B product without an exponentially growing branch."""
    limiting = jnp.minimum(first, second)
    excess = jnp.maximum(first, second)
    difference = excess - limiting
    scale = jnp.maximum(excess, jnp.finfo(time.dtype).tiny)
    equal = difference <= 32.0 * jnp.finfo(time.dtype).eps * scale
    scaled_time = limiting * rate * time
    equal_product = limiting * scaled_time / (1.0 + scaled_time)
    one_minus_decay = -jnp.expm1(-difference * rate * time)
    unequal_denominator = difference + limiting * one_minus_decay
    unequal_product = (
        limiting
        * excess
        * one_minus_decay
        / jnp.where(equal, jnp.ones_like(unequal_denominator), unequal_denominator)
    )
    return jnp.where(equal, equal_product, unequal_product)


class EffectiveDisplacementRateModel(StrictModule, NonTrainableState):
    """Effective mass action frozen from one content-addressed posterior fit."""

    fit: StrandDisplacementModelFit
    rate_constant_per_molar_second: Array
    reactant_construct_ids: tuple[str, str] = eqx.field(static=True)
    condition_id: str = eqx.field(static=True)
    chemistry_direction: str = eqx.field(static=True)
    temperature_kelvin: float = eqx.field(static=True)
    fit_case_ids: tuple[str, ...] = eqx.field(static=True)
    parameter_plan_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(self, fit: StrandDisplacementModelFit, /):
        if not isinstance(fit, StrandDisplacementModelFit):
            raise TypeError(
                "Effective displacement models require a StrandDisplacementModelFit."
            )
        prepared = fit.selected_prepared
        if not isinstance(prepared, PreparedEffectiveDisplacementInference):
            raise ValueError("Fit did not select an effective mass-action model.")
        if prepared.parameter_plan.parameter_names != (_PARAMETER_NAME,):
            raise ValueError("Effective fit has incompatible parameter semantics.")
        rate = _positive(float(fit.parameter_values[0]), _PARAMETER_NAME)
        self.fit = fit
        self.rate_constant_per_molar_second = jnp.asarray(rate)
        self.reactant_construct_ids = prepared.reactant_construct_ids
        self.condition_id = prepared.parameter_plan.condition_domain_id
        self.chemistry_direction = prepared.parameter_plan.chemistry
        self.temperature_kelvin = prepared.parameter_plan.temperature_kelvin
        self.fit_case_ids = fit.fit_case_ids
        self.parameter_plan_id = prepared.parameter_plan.parameter_plan_id
        self.model_id = canonical_fingerprint(
            {
                "kind": "fitted-effective-strand-displacement-rate-model",
                "fit": fit.fit_id,
                "rate_per_molar_second": rate,
                "reactants": self.reactant_construct_ids,
                "condition": self.condition_id,
                "chemistry_direction": self.chemistry_direction,
                "temperature_kelvin": self.temperature_kelvin,
                "parameter_plan": self.parameter_plan_id,
            }
        )

    @property
    def uncertainty_limitations(self) -> tuple[str, ...]:
        return (
            ()
            if self.fit.has_epistemic_uncertainty
            else ("effective-rate-epistemic-uncertainty-unquantified",)
        )

    def support_reasons(self, trace: FluorescenceTimeTrace, /) -> tuple[str, ...]:
        reasons: list[str] = []
        if trace.condition_id != self.condition_id:
            reasons.append("condition-outside-effective-model-support")
        if trace.temperature_kelvin != self.temperature_kelvin:
            reasons.append("temperature-outside-effective-model-support")
        if trace.chemistry_direction != self.chemistry_direction:
            reasons.append("chemistry-direction-outside-effective-model-support")
        if not set(self.reactant_construct_ids) <= set(trace.construct_ids):
            reasons.append("reactant-constructs-outside-effective-model-support")
        return tuple(reasons)

    def parameterized_product_concentration(
        self, trace: FluorescenceTimeTrace, parameters: ArrayLike, /
    ) -> Array:
        reasons = self.support_reasons(trace)
        if reasons:
            raise ValueError(";".join(reasons))
        values = jnp.asarray(parameters, dtype=float)
        if values.shape != (1,):
            raise ValueError("Effective fitted parameters must have shape (1,).")
        rate = values[0]
        concentration_by_construct = dict(
            zip(
                trace.construct_ids,
                trace.initial_concentrations_molar,
                strict=True,
            )
        )
        first = concentration_by_construct[self.reactant_construct_ids[0]]
        second = concentration_by_construct[self.reactant_construct_ids[1]]
        time = jnp.maximum(trace.time_seconds, 0.0)
        product = _bimolecular_product_concentration(first, second, rate, time)
        return jnp.where(trace.time_seconds <= 0.0, 0.0, product)

    def product_concentration(self, trace: FluorescenceTimeTrace, /) -> Array:
        return self.parameterized_product_concentration(trace, self.fit.parameter_values)


class MechanisticDisplacementRateModel(StrictModule, NonTrainableState):
    """Exhaustive CTMC frozen from one content-addressed posterior rate-scale fit."""

    fit: StrandDisplacementModelFit
    prepared: PreparedSecondaryKinetics
    base_generator: Array
    product_mask: Array
    rate_scale: Array
    initial_state_index: int = eqx.field(static=True)
    state_count: int = eqx.field(static=True)
    product_state_ids: tuple[str, ...] = eqx.field(static=True)
    reactant_construct_ids: tuple[str, str] = eqx.field(static=True)
    chemistry_direction: str = eqx.field(static=True)
    supported_initial_concentrations_molar: tuple[float, float] = eqx.field(static=True)
    condition_id: str = eqx.field(static=True)
    temperature_kelvin: float = eqx.field(static=True)
    fit_case_ids: tuple[str, ...] = eqx.field(static=True)
    parameter_plan_id: str = eqx.field(static=True)
    state_capacity: int = eqx.field(static=True)
    channel_capacity: int = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(self, fit: StrandDisplacementModelFit, /):
        if not isinstance(fit, StrandDisplacementModelFit):
            raise TypeError(
                "Mechanistic displacement models require a StrandDisplacementModelFit."
            )
        template = fit.selected_prepared
        if not isinstance(template, PreparedMechanisticDisplacementInference):
            raise ValueError("Fit did not select an exhaustive mechanistic model.")
        if template.parameter_plan.parameter_names != ("rate_scale",):
            raise ValueError("Mechanistic fit has incompatible parameter semantics.")
        scale = _positive(float(fit.parameter_values[0]), "rate_scale")
        self.fit = fit
        self.prepared = template.prepared
        self.base_generator = template.base_generator
        self.product_mask = template.product_mask
        self.rate_scale = jnp.asarray(scale)
        self.initial_state_index = template.initial_state_index
        self.state_count = template.state_count
        self.product_state_ids = template.product_state_ids
        self.reactant_construct_ids = template.reactant_construct_ids
        self.chemistry_direction = template.chemistry_direction
        self.supported_initial_concentrations_molar = (
            template.supported_initial_concentrations_molar
        )
        self.condition_id = template.condition_id
        self.temperature_kelvin = template.prepared.model.temperature_kelvin
        self.fit_case_ids = fit.fit_case_ids
        self.parameter_plan_id = template.parameter_plan.parameter_plan_id
        self.state_capacity = template.state_capacity
        self.channel_capacity = template.channel_capacity
        self.model_id = canonical_fingerprint(
            {
                "kind": "fitted-exact-secondary-strand-displacement-model",
                "fit": fit.fit_id,
                "process": self.prepared.process.process_id,
                "rate_scale": scale,
                "initial_state_index": self.initial_state_index,
                "product_states": self.product_state_ids,
                "state_capacity": self.state_capacity,
                "channel_capacity": self.channel_capacity,
            }
        )

    @property
    def uncertainty_limitations(self) -> tuple[str, ...]:
        return (
            ()
            if self.fit.has_epistemic_uncertainty
            else ("mechanistic-rate-epistemic-uncertainty-unquantified",)
        )

    def support_reasons(self, trace: FluorescenceTimeTrace, /) -> tuple[str, ...]:
        reasons: list[str] = []
        if trace.condition_id != self.condition_id:
            reasons.append("condition-outside-mechanistic-model-support")
        if trace.temperature_kelvin != self.temperature_kelvin:
            reasons.append("temperature-outside-mechanistic-model-support")
        if trace.chemistry_direction != self.chemistry_direction:
            reasons.append("chemistry-direction-outside-mechanistic-model-support")
        concentration_by_construct = dict(
            zip(
                trace.construct_ids,
                np.asarray(trace.initial_concentrations_molar),
                strict=True,
            )
        )
        if not set(self.reactant_construct_ids) <= set(concentration_by_construct):
            reasons.append("reactant-constructs-outside-mechanistic-model-support")
        elif (
            tuple(
                float(concentration_by_construct[construct])
                for construct in self.reactant_construct_ids
            )
            != self.supported_initial_concentrations_molar
        ):
            reasons.append("concentration-outside-mechanistic-model-support")
        if len(self.prepared.states) > self.state_capacity:
            reasons.append("enumerated-state-capacity-exceeded")
        if self.prepared.process.num_channels > self.channel_capacity:
            reasons.append("enumerated-channel-capacity-exceeded")
        if self.prepared.association.mode != "fixed_volume":
            reasons.append("mechanistic-concentration-scale-undefined")
        else:
            prepared_molar = 1.0 / (
                1000.0 * _AVOGADRO_CONSTANT_PER_MOL * self.prepared.association.volume_m3
            )
            if any(
                not math.isclose(value, prepared_molar, rel_tol=1e-12, abs_tol=0.0)
                for value in self.supported_initial_concentrations_molar
            ):
                reasons.append("concentration-outside-prepared-ctmc-volume")
        return tuple(reasons)

    def parameterized_product_concentration(
        self, trace: FluorescenceTimeTrace, parameters: ArrayLike, /
    ) -> Array:
        reasons = self.support_reasons(trace)
        if reasons:
            raise ValueError(";".join(reasons))
        values = jnp.asarray(parameters, dtype=float)
        if values.shape != (1,):
            raise ValueError("Mechanistic fitted parameters must have shape (1,).")
        generator = self.base_generator * values[0]
        initial = jax.nn.one_hot(self.initial_state_index, self.state_count)
        nonnegative_time = jnp.maximum(trace.time_seconds, 0.0)

        def occupancy(time):
            action = la.matrix_exponential_action(generator.T, initial, time)
            probabilities = eqx.error_if(
                jnp.asarray(action.value),
                ~action.converged,
                "Secondary-kinetics occupancy propagation did not converge.",
            )
            return probabilities @ self.product_mask.astype(probabilities.dtype)

        occupancy_ = jax.vmap(occupancy)(nonnegative_time)
        limiting = min(self.supported_initial_concentrations_molar)
        product = limiting * occupancy_
        return jnp.where(trace.time_seconds <= 0.0, 0.0, product)

    def product_concentration(self, trace: FluorescenceTimeTrace, /) -> Array:
        return self.parameterized_product_concentration(trace, self.fit.parameter_values)


class StrandDisplacementForwardModel(Protocol):
    fit: StrandDisplacementModelFit
    model_id: str
    fit_case_ids: tuple[str, ...]
    uncertainty_limitations: tuple[str, ...]

    def support_reasons(self, trace: FluorescenceTimeTrace, /) -> tuple[str, ...]: ...

    def parameterized_product_concentration(
        self, trace: FluorescenceTimeTrace, parameters: ArrayLike, /
    ) -> Array: ...

    def product_concentration(self, trace: FluorescenceTimeTrace, /) -> Array: ...


@dataclass(frozen=True, slots=True)
class SecondaryKineticParameterPlan:
    """Source/campaign identity for fitted kinetic parameters, never a rate-law mutation."""

    parameter_names: tuple[str, ...]
    chemistry: str
    temperature_kelvin: float
    condition_domain_id: str
    source_manifest_ids: tuple[str, ...]
    source_manifests: tuple[ReferenceArtifactManifest, ...]
    requested_use: tuple[tuple[str, bool], ...]
    prior_ids: tuple[str, ...]
    parameter_plan_id: str

    def __init__(
        self,
        parameter_names: Sequence[str],
        chemistry: str,
        temperature_kelvin: float,
        condition_domain_id: str,
        source_manifests: Sequence[ReferenceArtifactManifest],
        prior_ids: Sequence[str],
        *,
        requested_use: Mapping[str, bool],
    ):
        names = _ordered_identifiers(parameter_names, "parameter_names")
        priors = _ordered_identifiers(prior_ids, "prior_ids")
        if len(names) != len(priors):
            raise ValueError("Every kinetic parameter requires one declared prior ID.")
        temperature = _positive(temperature_kelvin, "temperature_kelvin")
        manifest_values = tuple(source_manifests)
        if not manifest_values or any(
            not isinstance(value, ReferenceArtifactManifest) for value in manifest_values
        ):
            raise TypeError("Kinetic parameter sources must be admitted manifests.")
        if (
            not isinstance(requested_use, Mapping)
            or set(requested_use) != _RIGHTS_KEYS
            or any(type(value) is not bool for value in requested_use.values())
        ):
            raise ValueError(
                "Declare all four kinetic-parameter requested-use rights explicitly."
            )
        for manifest in manifest_values:
            manifest.require_rights(**dict(requested_use))
        manifest_by_id = {manifest.manifest_id: manifest for manifest in manifest_values}
        if len(manifest_by_id) != len(manifest_values):
            raise ValueError("Kinetic parameter manifests must be unique.")
        manifests = tuple(sorted(manifest_by_id))
        chemistry_ = _identifier(chemistry, "chemistry")
        condition = _identifier(condition_domain_id, "condition_domain_id")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "chemistry", chemistry_)
        object.__setattr__(self, "temperature_kelvin", temperature)
        object.__setattr__(self, "condition_domain_id", condition)
        object.__setattr__(self, "source_manifest_ids", manifests)
        object.__setattr__(
            self,
            "source_manifests",
            tuple(manifest_by_id[manifest_id] for manifest_id in manifests),
        )
        object.__setattr__(self, "requested_use", tuple(sorted(requested_use.items())))
        object.__setattr__(self, "prior_ids", priors)
        object.__setattr__(
            self,
            "parameter_plan_id",
            canonical_fingerprint(
                {
                    "kind": "secondary-kinetic-parameter-plan",
                    "parameters": names,
                    "chemistry": chemistry_,
                    "temperature_kelvin": temperature,
                    "condition": condition,
                    "sources": manifests,
                    "requested_use": dict(self.requested_use),
                    "priors": priors,
                }
            ),
        )

    def values(self, parameters: Mapping[str, ArrayLike], /) -> tuple[Array, ...]:
        if not isinstance(parameters, Mapping) or set(parameters) != set(
            self.parameter_names
        ):
            raise ValueError(
                "Kinetic parameter mapping must exactly match the parameter plan."
            )
        result = tuple(
            jnp.asarray(parameters[name], dtype=float) for name in self.parameter_names
        )
        if any(value.shape != () for value in result):
            raise ValueError(
                "Kinetic parameter plan v1 requires scalar parameter values."
            )
        return result


@dataclass(frozen=True, slots=True)
class StrandDisplacementPrediction:
    """Independent leading-axis raw-trace predictions without timepoint pseudo-cases."""

    predictions: tuple[FluorescencePrediction, ...]
    case_ids: tuple[str, ...]
    model_id: str


def trace_log_probability(
    trace: FluorescenceTimeTrace,
    prediction: FluorescencePrediction,
    /,
) -> Array:
    """Return one Gaussian/censored log probability for one independent trace."""

    if trace.case_id != prediction.case_id or trace.trace_id != prediction.trace_id:
        raise ValueError("Prediction and observation identities must match exactly.")
    mean = prediction.mean_intensity
    sigma = prediction.standard_deviation_intensity
    if mean.shape != trace.intensity.shape or sigma.shape != trace.intensity.shape:
        raise ValueError("Prediction arrays must align to the observed trace.")
    sigma = eqx.error_if(
        sigma,
        jnp.any(~jnp.isfinite(mean) | ~jnp.isfinite(sigma) | (sigma <= 0.0)),
        "Trace likelihood requires finite means and positive uncertainty.",
    )
    saturated = trace.saturation_mask
    any_saturated = trace.has_saturated_observations
    if any_saturated and trace.saturation_threshold_intensity is None:
        raise ValueError(
            "Censored saturated observations require an instrument threshold."
        )
    covariance = prediction.epistemic_parameter_covariance
    sensitivity = prediction.epistemic_sensitivity
    if covariance is not None:
        if sensitivity is None:
            raise ValueError(
                "Epistemic covariance requires aligned response sensitivities."
            )
        if any_saturated:
            raise ValueError(
                "Correlated epistemic uncertainty with censored samples is unsupported."
            )
        residual_sigma = prediction.residual_standard_deviation_intensity
        residual_sigma = eqx.error_if(
            residual_sigma,
            (~jnp.isfinite(residual_sigma)) | (residual_sigma <= 0.0),
            "Correlated trace likelihood requires positive residual uncertainty.",
        )
        eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
        factor = eigenvectors * jnp.sqrt(jnp.maximum(eigenvalues, 0.0))[None, :]
        low_rank = sensitivity @ factor
        variance = residual_sigma**2
        correction = (
            jnp.eye(low_rank.shape[1], dtype=low_rank.dtype)
            + low_rank.T @ low_rank / variance
        )
        factorization = la.factorize(
            correction,
            la.FactorizationPolicy("cholesky"),
            properties=la.OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
        )
        centered = trace.intensity - mean
        projected = low_rank.T @ centered
        solve_result = factorization.solve(projected)
        solved = eqx.error_if(
            solve_result.value,
            ~solve_result.successful,
            "Correlated trace covariance solve failed.",
        )
        correction_log_determinant = factorization.log_abs_determinant()
        quadratic = centered @ centered / variance - projected @ solved / variance**2
        log_determinant = centered.size * jnp.log(variance) + correction_log_determinant
        return (
            -0.5 * (centered.size * math.log(2.0 * math.pi) + log_determinant + quadratic)
        ).reshape(())
    safe_observation = jnp.where(saturated, mean, trace.intensity)
    z = (safe_observation - mean) / sigma
    ordinary = -0.5 * z**2 - jnp.log(sigma) - 0.5 * math.log(2.0 * math.pi)
    threshold = (
        0.0
        if trace.saturation_threshold_intensity is None
        else trace.saturation_threshold_intensity
    )
    censored = jsp.special.log_ndtr((mean - threshold) / sigma)
    return jnp.sum(jnp.where(saturated, censored, ordinary)).reshape(())


class PreparedEffectiveDisplacementInference(StrictModule, NonTrainableState):
    """Calibration-role effective-rate posterior over exact admitted well traces."""

    traces: tuple[FluorescenceTimeTrace, ...]
    observation_model: ReporterObservationModel
    parameter_plan: SecondaryKineticParameterPlan
    parameter_space: ParameterSpace
    reactant_construct_ids: tuple[str, str] = eqx.field(static=True)
    trace_source_manifests: tuple[ReferenceArtifactManifest, ...] = eqx.field(static=True)
    trace_requested_use: tuple[tuple[str, bool], ...] = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)
    fit_case_ids: tuple[str, ...] = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)

    def __init__(
        self,
        traces: Sequence[FluorescenceTimeTrace],
        observation_model: ReporterObservationModel,
        parameter_plan: SecondaryKineticParameterPlan,
        parameter_space: ParameterSpace,
        reactant_construct_ids: Sequence[str],
        campaign: ScientificCampaign,
        /,
        *,
        trace_source_manifests: Sequence[ReferenceArtifactManifest],
        trace_requested_use: Mapping[str, bool],
    ):
        traces_ = _validate_campaign_role_traces(traces, campaign, "calibration")
        if observation_model.campaign_id != campaign.campaign_id:
            raise ValueError(
                "Effective inference and reporter model must share the exact campaign."
            )
        if parameter_plan.parameter_names != (_PARAMETER_NAME,):
            raise ValueError(
                f"Effective inference parameter plan must contain only {_PARAMETER_NAME!r}."
            )
        if not isinstance(parameter_space, ParameterSpace):
            raise TypeError("parameter_space must be a ParameterSpace.")
        reactants = _identifiers(tuple(reactant_construct_ids), "reactant_construct_ids")
        if len(reactants) != 2:
            raise ValueError("Effective inference requires exactly two reactants.")
        if any(
            trace.condition_id != parameter_plan.condition_domain_id
            or trace.temperature_kelvin != parameter_plan.temperature_kelvin
            or trace.chemistry_direction != parameter_plan.chemistry
            or not set(reactants) <= set(trace.construct_ids)
            for trace in traces_
        ):
            raise ValueError(
                "Inference traces must lie exactly inside the parameter plan support."
            )
        expected_sources = tuple(
            sorted({source for trace in traces_ for source in trace.source_manifest_ids})
        )
        trace_manifests, trace_use = _admitted_source_manifests(
            trace_source_manifests,
            trace_requested_use,
            expected_sources,
            "Raw fluorescence trace",
        )
        self.traces = traces_
        self.observation_model = observation_model
        self.parameter_plan = parameter_plan
        self.parameter_space = parameter_space
        self.reactant_construct_ids = (reactants[0], reactants[1])
        self.trace_source_manifests = trace_manifests
        self.trace_requested_use = trace_use
        self.campaign_id = campaign.campaign_id
        self.fit_case_ids = tuple(trace.case_id for trace in traces_)
        self.preparation_id = canonical_fingerprint(
            {
                "kind": "prepared-effective-strand-displacement-inference",
                "campaign": campaign.campaign_id,
                "preprocessing_sources": campaign.preprocessing_source_ids,
                "traces": [trace.trace_id for trace in traces_],
                "observation": observation_model.observation_model_id,
                "parameters": parameter_plan.parameter_plan_id,
                "trace_sources": [manifest.manifest_id for manifest in trace_manifests],
                "trace_requested_use": dict(trace_use),
                "reactants": reactants,
            }
        )

    def _product_concentration(
        self,
        trace: FluorescenceTimeTrace,
        rate_constant_per_molar_second: Array,
        /,
    ) -> Array:
        rate = eqx.error_if(
            rate_constant_per_molar_second,
            (~jnp.isfinite(rate_constant_per_molar_second))
            | (rate_constant_per_molar_second <= 0.0),
            "Effective displacement rate must be finite and positive.",
        )
        concentration_by_construct = dict(
            zip(
                trace.construct_ids,
                trace.initial_concentrations_molar,
                strict=True,
            )
        )
        first = concentration_by_construct[self.reactant_construct_ids[0]]
        second = concentration_by_construct[self.reactant_construct_ids[1]]
        time = jnp.maximum(trace.time_seconds, 0.0)
        product = _bimolecular_product_concentration(first, second, rate, time)
        return jnp.where(trace.time_seconds <= 0.0, 0.0, product)

    def predict_cases(
        self,
        position: Mapping[str, ArrayLike],
        cases: Sequence[FluorescenceTimeTrace] | None = None,
        /,
    ) -> StrandDisplacementPrediction:
        (rate,) = self.parameter_plan.values(position)
        selected = self.traces if cases is None else tuple(cases)
        forward_model_id = canonical_fingerprint(
            {
                "preparation": self.preparation_id,
                "parameter_semantics": "physical-posterior-position",
            }
        )
        predictions = tuple(
            self.observation_model.prediction(
                trace,
                self._product_concentration(trace, rate),
                forward_model_id=forward_model_id,
            )
            for trace in selected
        )
        return StrandDisplacementPrediction(
            predictions,
            tuple(trace.case_id for trace in selected),
            forward_model_id,
        )

    def per_case_log_prob(self, position: Mapping[str, ArrayLike], /) -> Array:
        prediction = self.predict_cases(position)
        return jnp.stack(
            tuple(
                trace_log_probability(trace, predicted)
                for trace, predicted in zip(
                    self.traces, prediction.predictions, strict=True
                )
            )
        )

    def sample_observations(
        self,
        key: Array,
        position: Mapping[str, ArrayLike],
        cases: Sequence[FluorescenceTimeTrace] | None = None,
        /,
    ) -> tuple[Array, ...]:
        prediction = self.predict_cases(position, cases)
        keys = jr.split(key, len(prediction.predictions))

        def sample(item: FluorescencePrediction, sample_key: Array, /) -> Array:
            residual_key, epistemic_key = jr.split(sample_key)
            result = (
                item.mean_intensity
                + item.residual_standard_deviation_intensity
                * jr.normal(residual_key, item.mean_intensity.shape)
            )
            if item.epistemic_parameter_covariance is not None:
                epistemic_draw = jr.multivariate_normal(
                    epistemic_key,
                    jnp.zeros(item.epistemic_parameter_covariance.shape[0]),
                    item.epistemic_parameter_covariance,
                )
                result = result + item.epistemic_sensitivity @ epistemic_draw
            return result

        return tuple(
            sample(item, sample_key)
            for sample_key, item in zip(keys, prediction.predictions, strict=True)
        )

    def posterior_problem(self) -> PosteriorProblem:
        term = EffectiveFluorescencePosteriorTerm(self)
        return PosteriorProblem.from_terms(
            self.parameter_space,
            (term,),
            predict=self.predict_cases,
            sample_observation=self.sample_observations,
        )


class EffectiveFluorescencePosteriorTerm(AbstractPosteriorTerm):
    """Trace-level likelihood term: the leading axis is wells, never timepoints."""

    prepared: PreparedEffectiveDisplacementInference

    def __init__(self, prepared: PreparedEffectiveDisplacementInference):
        if not isinstance(prepared, PreparedEffectiveDisplacementInference):
            raise TypeError("prepared must be PreparedEffectiveDisplacementInference.")
        self.prepared = prepared
        self.label = "strand-displacement-effective-raw-fluorescence"

    def per_case_log_prob(self, parameters: PyTree, /) -> Array:
        return self.prepared.per_case_log_prob(parameters)


class PreparedMechanisticDisplacementInference(StrictModule, NonTrainableState):
    """Calibration-role posterior for one global exhaustive-CTMC rate scale."""

    traces: tuple[FluorescenceTimeTrace, ...]
    observation_model: ReporterObservationModel
    parameter_plan: SecondaryKineticParameterPlan
    parameter_space: ParameterSpace
    prepared: PreparedSecondaryKinetics
    product_mask: Array
    base_generator: Array
    trace_source_manifests: tuple[ReferenceArtifactManifest, ...] = eqx.field(static=True)
    trace_requested_use: tuple[tuple[str, bool], ...] = eqx.field(static=True)
    initial_state_index: int = eqx.field(static=True)
    state_count: int = eqx.field(static=True)
    product_state_ids: tuple[str, ...] = eqx.field(static=True)
    reactant_construct_ids: tuple[str, str] = eqx.field(static=True)
    supported_initial_concentrations_molar: tuple[float, float] = eqx.field(static=True)
    chemistry_direction: str = eqx.field(static=True)
    condition_id: str = eqx.field(static=True)
    state_capacity: int = eqx.field(static=True)
    channel_capacity: int = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)
    fit_case_ids: tuple[str, ...] = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)

    def __init__(
        self,
        traces: Sequence[FluorescenceTimeTrace],
        observation_model: ReporterObservationModel,
        parameter_plan: SecondaryKineticParameterPlan,
        parameter_space: ParameterSpace,
        prepared: PreparedSecondaryKinetics,
        initial_state: SecondaryStructureState,
        product_target: CompiledSecondaryTarget,
        product_state_ids: Sequence[str],
        reactant_construct_ids: Sequence[str],
        supported_initial_concentrations_molar: Sequence[float],
        chemistry_direction: str,
        condition_id: str,
        campaign: ScientificCampaign,
        /,
        *,
        trace_source_manifests: Sequence[ReferenceArtifactManifest],
        trace_requested_use: Mapping[str, bool],
        state_capacity: int,
        channel_capacity: int,
    ):
        traces_ = _validate_campaign_role_traces(traces, campaign, "calibration")
        if observation_model.campaign_id != campaign.campaign_id:
            raise ValueError(
                "Mechanistic inference and reporter model must share the exact campaign."
            )
        if parameter_plan.parameter_names != ("rate_scale",):
            raise ValueError(
                "Mechanistic inference parameter plan must contain only 'rate_scale'."
            )
        if not isinstance(parameter_space, ParameterSpace):
            raise TypeError("parameter_space must be a ParameterSpace.")
        if not isinstance(prepared, PreparedSecondaryKinetics):
            raise TypeError("prepared must be PreparedSecondaryKinetics.")
        if not isinstance(product_target, CompiledSecondaryTarget):
            raise TypeError("product_target must be a CompiledSecondaryTarget.")
        if product_target.process_id != prepared.process.process_id:
            raise ValueError(
                "Product target belongs to a different prepared kinetics process."
            )
        if type(state_capacity) is not int or type(channel_capacity) is not int:
            raise TypeError("Mechanistic support capacities must be integers.")
        if state_capacity <= 0 or channel_capacity <= 0:
            raise ValueError("Mechanistic capacities must be positive.")
        initial_index = int(prepared.encode(initial_state)[0])
        mask = jnp.asarray(product_target.mask, dtype=bool)
        if (
            mask.shape != (len(prepared.states),)
            or not bool(jnp.any(mask))
            or bool(jnp.all(mask))
        ):
            raise ValueError(
                "Product macrostate must select a nonempty proper subset of support."
            )
        if bool(mask[initial_index]):
            raise ValueError(
                "Initial state must not already belong to the product macrostate."
            )
        declared = _identifiers(product_state_ids, "product_state_ids")
        actual = tuple(
            sorted(
                state.fingerprint()
                for state, selected in zip(prepared.states, np.asarray(mask), strict=True)
                if selected
            )
        )
        if declared != actual:
            raise ValueError(
                "Declared product state IDs must equal the exact target macrostate support."
            )
        reactants = _ordered_identifiers(
            tuple(reactant_construct_ids), "reactant_construct_ids"
        )
        concentrations = tuple(
            float(value) for value in supported_initial_concentrations_molar
        )
        if (
            len(reactants) != 2
            or len(concentrations) != 2
            or not set(reactants) <= set(prepared.construct.strand_ids)
            or any(not math.isfinite(value) or value < 0.0 for value in concentrations)
        ):
            raise ValueError(
                "Mechanistic inference requires two prepared reactants and concentrations."
            )
        chemistry = _identifier(chemistry_direction, "chemistry_direction")
        condition = _identifier(condition_id, "condition_id")
        if (
            parameter_plan.chemistry != chemistry
            or parameter_plan.condition_domain_id != condition
            or parameter_plan.temperature_kelvin != prepared.model.temperature_kelvin
            or any(
                trace.condition_id != condition
                or trace.temperature_kelvin != prepared.model.temperature_kelvin
                or trace.chemistry_direction != chemistry
                or tuple(
                    float(
                        dict(
                            zip(
                                trace.construct_ids,
                                np.asarray(trace.initial_concentrations_molar),
                                strict=True,
                            )
                        )[reactant]
                    )
                    for reactant in reactants
                )
                != concentrations
                for trace in traces_
            )
        ):
            raise ValueError(
                "Mechanistic inference traces must lie exactly inside prepared support."
            )
        expected_sources = tuple(
            sorted({source for trace in traces_ for source in trace.source_manifest_ids})
        )
        trace_manifests, trace_use = _admitted_source_manifests(
            trace_source_manifests,
            trace_requested_use,
            expected_sources,
            "Raw fluorescence trace",
        )
        self.traces = traces_
        self.observation_model = observation_model
        self.parameter_plan = parameter_plan
        self.parameter_space = parameter_space
        self.prepared = prepared
        self.base_generator = prepared.generator().matrix
        self.product_mask = mask
        self.trace_source_manifests = trace_manifests
        self.trace_requested_use = trace_use
        self.initial_state_index = initial_index
        self.state_count = len(prepared.states)
        self.product_state_ids = declared
        self.reactant_construct_ids = (reactants[0], reactants[1])
        self.supported_initial_concentrations_molar = (
            concentrations[0],
            concentrations[1],
        )
        self.chemistry_direction = chemistry
        self.condition_id = condition
        self.state_capacity = state_capacity
        self.channel_capacity = channel_capacity
        self.campaign_id = campaign.campaign_id
        self.fit_case_ids = tuple(trace.case_id for trace in traces_)
        self.preparation_id = canonical_fingerprint(
            {
                "kind": "prepared-mechanistic-strand-displacement-inference",
                "campaign": campaign.campaign_id,
                "preprocessing_sources": campaign.preprocessing_source_ids,
                "traces": [trace.trace_id for trace in traces_],
                "observation": observation_model.observation_model_id,
                "parameter_plan": parameter_plan.parameter_plan_id,
                "trace_sources": [manifest.manifest_id for manifest in trace_manifests],
                "trace_requested_use": dict(trace_use),
                "process": prepared.process.process_id,
                "initial_state": initial_state.fingerprint(),
                "product_target": product_target.target_id,
                "product_states": declared,
                "reactants": reactants,
                "concentrations": concentrations,
                "chemistry": chemistry,
                "condition": condition,
                "state_capacity": state_capacity,
                "channel_capacity": channel_capacity,
            }
        )

    def _product_concentration(
        self, trace: FluorescenceTimeTrace, rate_scale: Array, /
    ) -> Array:
        scale = eqx.error_if(
            rate_scale,
            (~jnp.isfinite(rate_scale)) | (rate_scale <= 0.0),
            "Mechanistic rate scale must be finite and positive.",
        )
        generator = self.base_generator * scale
        initial = jax.nn.one_hot(self.initial_state_index, self.state_count)
        time = jnp.maximum(trace.time_seconds, 0.0)

        def occupancy(value):
            action = la.matrix_exponential_action(generator.T, initial, value)
            probabilities = eqx.error_if(
                jnp.asarray(action.value),
                ~action.converged,
                "Secondary-kinetics occupancy propagation did not converge.",
            )
            return probabilities @ self.product_mask.astype(probabilities.dtype)

        product = min(self.supported_initial_concentrations_molar) * jax.vmap(occupancy)(
            time
        )
        return jnp.where(trace.time_seconds <= 0.0, 0.0, product)

    def predict_cases(
        self,
        position: Mapping[str, ArrayLike],
        cases: Sequence[FluorescenceTimeTrace] | None = None,
        /,
    ) -> StrandDisplacementPrediction:
        (rate_scale,) = self.parameter_plan.values(position)
        selected = self.traces if cases is None else tuple(cases)
        forward_model_id = canonical_fingerprint(
            {
                "preparation": self.preparation_id,
                "parameter_semantics": "physical-posterior-position",
            }
        )
        predictions = tuple(
            self.observation_model.prediction(
                trace,
                self._product_concentration(trace, rate_scale),
                forward_model_id=forward_model_id,
            )
            for trace in selected
        )
        return StrandDisplacementPrediction(
            predictions,
            tuple(trace.case_id for trace in selected),
            forward_model_id,
        )

    def per_case_log_prob(self, position: Mapping[str, ArrayLike], /) -> Array:
        prediction = self.predict_cases(position)
        return jnp.stack(
            tuple(
                trace_log_probability(trace, predicted)
                for trace, predicted in zip(
                    self.traces, prediction.predictions, strict=True
                )
            )
        )

    def posterior_problem(self) -> PosteriorProblem:
        return PosteriorProblem.from_terms(
            self.parameter_space,
            (MechanisticFluorescencePosteriorTerm(self),),
            predict=self.predict_cases,
        )


class MechanisticFluorescencePosteriorTerm(AbstractPosteriorTerm):
    """Independent raw-trace likelihood for one exhaustive CTMC rate scale."""

    prepared: PreparedMechanisticDisplacementInference

    def __init__(self, prepared: PreparedMechanisticDisplacementInference):
        if not isinstance(prepared, PreparedMechanisticDisplacementInference):
            raise TypeError("prepared must be PreparedMechanisticDisplacementInference.")
        self.prepared = prepared
        self.label = "strand-displacement-mechanistic-raw-fluorescence"

    def per_case_log_prob(self, parameters: PyTree, /) -> Array:
        return self.prepared.per_case_log_prob(parameters)


PreparedStrandDisplacementInference = (
    PreparedEffectiveDisplacementInference | PreparedMechanisticDisplacementInference
)
StrandPosteriorResult = MAPResult | LaplaceResult | MCMCResult


def _posterior_summary(
    prepared: PreparedStrandDisplacementInference,
    result: StrandPosteriorResult,
    /,
) -> tuple[Array, Array | None, Array | None, str]:
    if not isinstance(result, (MAPResult, LaplaceResult, MCMCResult)):
        raise TypeError("Candidate result must be a native MAP, Laplace, or MCMC result.")
    problem = result.problem
    likelihood = problem.log_likelihood_fn
    terms = getattr(likelihood, "terms", ())
    if (
        problem.parameter_space is not prepared.parameter_space
        or len(terms) != 1
        or getattr(terms[0], "prepared", None) is not prepared
    ):
        raise ValueError(
            "Posterior result was not produced by the exact prepared strand problem."
        )
    names = prepared.parameter_plan.parameter_names
    if len(names) != 1:
        raise ValueError("Strand-displacement fit v1 requires exactly one parameter.")
    covariance = None
    draws = None
    if isinstance(result, MAPResult):
        if not result.converged:
            raise ValueError("A non-converged MAP result cannot freeze a strand model.")
        physical = result.parameters
        method = "map"
    elif isinstance(result, LaplaceResult):
        physical = result.map_parameters
        covariance = jnp.asarray(result.physical_covariance(), dtype=float)
        method = "laplace"
    else:
        if not isinstance(result.samples, Mapping) or set(result.samples) != set(names):
            raise ValueError("MCMC samples do not match the kinetic parameter plan.")
        sample = jnp.asarray(result.samples[names[0]], dtype=float)
        if sample.ndim < 1:
            raise ValueError("MCMC result must retain a posterior draw axis.")
        draws = sample.reshape((-1, 1))
        if draws.shape[0] < 2:
            raise ValueError("MCMC uncertainty requires at least two posterior draws.")
        physical = {names[0]: jnp.mean(draws[:, 0])}
        covariance = jnp.cov(draws, rowvar=False).reshape((1, 1))
        method = "mcmc"
    if not isinstance(physical, Mapping) or set(physical) != set(names):
        raise ValueError("Posterior parameters do not match the kinetic parameter plan.")
    parameters = jnp.stack(
        tuple(jnp.asarray(physical[name], dtype=float).reshape(()) for name in names)
    )
    if (
        bool(jnp.any(~jnp.isfinite(parameters) | (parameters <= 0.0)))
        or (
            covariance is not None
            and (
                covariance.shape != (1, 1)
                or bool(jnp.any(~jnp.isfinite(covariance)))
                or float(covariance[0, 0]) < 0.0
            )
        )
        or (draws is not None and bool(jnp.any(~jnp.isfinite(draws) | (draws <= 0.0))))
    ):
        raise ValueError(
            "Posterior strand parameters and uncertainty must be finite and physical."
        )
    if not bool(jnp.isfinite(problem.log_likelihood(physical))):
        raise ValueError("Selected posterior parameters have non-finite fit likelihood.")
    return parameters, covariance, draws, method


def _candidate_hyperparameters(
    prepared: PreparedStrandDisplacementInference, /
) -> tuple[tuple[str, str], ...]:
    if isinstance(prepared, PreparedEffectiveDisplacementInference):
        values = {
            "forward_model_kind": "effective-mass-action",
            "parameter_plan_id": prepared.parameter_plan.parameter_plan_id,
            "reactant_order_id": canonical_fingerprint(prepared.reactant_construct_ids),
        }
    else:
        values = {
            "association_id": prepared.prepared.association.fingerprint(),
            "channel_capacity": str(prepared.channel_capacity),
            "energy_model_id": prepared.prepared.model.model_id,
            "forward_model_kind": "exhaustive-secondary-kinetics",
            "parameter_plan_id": prepared.parameter_plan.parameter_plan_id,
            "rate_law_id": prepared.prepared.rate_law.fingerprint(),
            "state_capacity": str(prepared.state_capacity),
            "target_support_id": canonical_fingerprint(prepared.product_state_ids),
        }
    return tuple(sorted(values.items()))


def _training_dependency_manifests(
    prepared: PreparedStrandDisplacementInference, /
) -> tuple[ReferenceArtifactManifest, ...]:
    dependencies = (
        (
            "raw fluorescence trace",
            prepared.trace_source_manifests,
            prepared.trace_requested_use,
        ),
        (
            "reporter calibration",
            prepared.observation_model.calibration.source_manifests,
            prepared.observation_model.calibration.requested_use,
        ),
        (
            "kinetic parameter",
            prepared.parameter_plan.source_manifests,
            prepared.parameter_plan.requested_use,
        ),
    )
    if isinstance(prepared, PreparedMechanisticDisplacementInference):
        dependencies = (
            *dependencies,
            (
                "secondary energy",
                (prepared.prepared.model.manifest,),
                prepared.prepared.model.requested_use,
            ),
        )
    by_id: dict[str, ReferenceArtifactManifest] = {}
    for owner, manifests, requested_use in dependencies:
        if not dict(requested_use).get("training_use", False):
            raise ValueError(f"{owner.title()} admission did not authorize training use.")
        for manifest in manifests:
            manifest.require_rights(training_use=True)
            by_id[manifest.manifest_id] = manifest
    return tuple(by_id[manifest_id] for manifest_id in sorted(by_id))


class StrandDisplacementModelFit(StrictModule, NonTrainableState):
    """Selected, content-addressed native posterior fit for one frozen campaign."""

    selected_prepared: PreparedStrandDisplacementInference
    parameter_values: Array
    parameter_covariance: Array | None
    parameter_draws: Array | None
    campaign_id: str = eqx.field(static=True)
    fit_case_ids: tuple[str, ...] = eqx.field(static=True)
    model_selection_case_ids: tuple[str, ...] = eqx.field(static=True)
    source_trace_ids: tuple[str, ...] = eqx.field(static=True)
    source_manifest_ids: tuple[str, ...] = eqx.field(static=True)
    source_manifests: tuple[ReferenceArtifactManifest, ...] = eqx.field(static=True)
    model_selection_requested_use: tuple[tuple[str, bool], ...] = eqx.field(static=True)
    preprocessing_source_ids: tuple[str, ...] = eqx.field(static=True)
    parameter_names: tuple[str, ...] = eqx.field(static=True)
    selected_hyperparameters: tuple[tuple[str, str], ...] = eqx.field(static=True)
    candidate_score_ids: tuple[str, ...] = eqx.field(static=True)
    candidate_scores: tuple[float, ...] = eqx.field(static=True)
    selected_candidate_index: int = eqx.field(static=True)
    posterior_method: str = eqx.field(static=True)
    fit_id: str = eqx.field(static=True)

    def __init__(
        self,
        candidates: Sequence[
            tuple[PreparedStrandDisplacementInference, StrandPosteriorResult]
        ],
        model_selection_traces: Sequence[FluorescenceTimeTrace],
        campaign: ScientificCampaign,
        /,
        *,
        model_selection_source_manifests: Sequence[ReferenceArtifactManifest],
        model_selection_requested_use: Mapping[str, bool],
    ):
        values = tuple(candidates)
        if not values:
            raise ValueError("Model selection requires at least one posterior candidate.")
        if any(not isinstance(value, tuple) or len(value) != 2 for value in values):
            raise TypeError(
                "Each model-selection candidate must pair preparation and native result."
            )
        selection = _validate_campaign_role_traces(
            model_selection_traces, campaign, "model_selection"
        )
        selection_source_ids = tuple(
            sorted(
                {source for trace in selection for source in trace.source_manifest_ids}
            )
        )
        selection_manifests, selection_use = _admitted_source_manifests(
            model_selection_source_manifests,
            model_selection_requested_use,
            selection_source_ids,
            "Model-selection fluorescence trace",
        )
        if not dict(selection_use)["training_use"]:
            raise ValueError(
                "Model-selection trace admission did not authorize training use."
            )
        for manifest in selection_manifests:
            manifest.require_rights(training_use=True)
        family = type(values[0][0])
        if family not in (
            PreparedEffectiveDisplacementInference,
            PreparedMechanisticDisplacementInference,
        ) or any(
            type(prepared) is not family
            or prepared.campaign_id != campaign.campaign_id
            or prepared.fit_case_ids
            != tuple(sorted(_role_cases(campaign, ("calibration",))))
            for prepared, _ in values
        ):
            raise ValueError(
                "Candidate preparations must share one model family and exact campaign."
            )
        candidate_dependency_manifests = tuple(
            _training_dependency_manifests(prepared) for prepared, _ in values
        )
        summaries = tuple(
            _posterior_summary(prepared, result) for prepared, result in values
        )
        configurations = tuple(
            _candidate_hyperparameters(prepared) for prepared, _ in values
        )
        if len(set(configurations)) != len(configurations):
            raise ValueError("Model-selection candidate hyperparameters must be unique.")
        candidate_ids = tuple(
            canonical_fingerprint(
                {
                    "kind": "strand-displacement-posterior-candidate",
                    "preparation": prepared.preparation_id,
                    "hyperparameters": dict(configuration),
                    "parameters": array_tree_fingerprint((parameters, covariance, draws))[
                        "sha256"
                    ],
                    "posterior_method": method,
                }
            )
            for (prepared, _), configuration, (
                parameters,
                covariance,
                draws,
                method,
            ) in zip(values, configurations, summaries, strict=True)
        )
        scores: list[float] = []
        score_ids: list[str] = []
        observation_count = sum(int(trace.time_seconds.size) for trace in selection)
        for (prepared, _), candidate_id, summary in zip(
            values, candidate_ids, summaries, strict=True
        ):
            parameters = summary[0]
            named = {
                name: parameters[index]
                for index, name in enumerate(prepared.parameter_plan.parameter_names)
            }
            prediction = prepared.predict_cases(named, selection)
            score = float(
                sum(
                    float(trace_log_probability(trace, predicted))
                    for trace, predicted in zip(
                        selection, prediction.predictions, strict=True
                    )
                )
                / observation_count
            )
            if not math.isfinite(score):
                raise ValueError("Model-selection score must be finite.")
            scores.append(score)
            score_ids.append(
                canonical_fingerprint(
                    {
                        "kind": "strand-displacement-model-selection-score",
                        "campaign": campaign.campaign_id,
                        "candidate": candidate_id,
                        "cases": [trace.trace_id for trace in selection],
                        "mean_log_score_per_observation": score,
                    }
                )
            )
        selected_index = min(
            range(len(values)),
            key=lambda index: (-scores[index], candidate_ids[index]),
        )
        selected_prepared = values[selected_index][0]
        parameters, covariance, draws, method = summaries[selected_index]
        fit_traces = selected_prepared.traces
        source_trace_ids = tuple(
            sorted(trace.trace_id for trace in (*fit_traces, *selection))
        )
        source_manifest_by_id = {
            manifest.manifest_id: manifest
            for manifests in (*candidate_dependency_manifests, selection_manifests)
            for manifest in manifests
        }
        source_manifest_ids = tuple(sorted(source_manifest_by_id))
        source_manifests = tuple(
            source_manifest_by_id[manifest_id] for manifest_id in source_manifest_ids
        )
        self.selected_prepared = selected_prepared
        self.parameter_values = parameters
        self.parameter_covariance = covariance
        self.parameter_draws = draws
        self.campaign_id = campaign.campaign_id
        self.fit_case_ids = selected_prepared.fit_case_ids
        self.model_selection_case_ids = tuple(trace.case_id for trace in selection)
        self.source_trace_ids = source_trace_ids
        self.source_manifest_ids = source_manifest_ids
        self.source_manifests = source_manifests
        self.model_selection_requested_use = selection_use
        self.preprocessing_source_ids = campaign.preprocessing_source_ids
        self.parameter_names = selected_prepared.parameter_plan.parameter_names
        self.selected_hyperparameters = configurations[selected_index]
        self.candidate_score_ids = tuple(score_ids)
        self.candidate_scores = tuple(scores)
        self.selected_candidate_index = selected_index
        self.posterior_method = method
        self.fit_id = canonical_fingerprint(
            {
                "kind": "selected-strand-displacement-model-fit",
                "campaign": campaign.campaign_id,
                "fit_cases": self.fit_case_ids,
                "model_selection_cases": self.model_selection_case_ids,
                "source_traces": source_trace_ids,
                "source_manifests": source_manifest_ids,
                "model_selection_requested_use": dict(selection_use),
                "preprocessing_sources": campaign.preprocessing_source_ids,
                "parameters": array_tree_fingerprint((parameters, covariance, draws))[
                    "sha256"
                ],
                "parameter_names": self.parameter_names,
                "selected_hyperparameters": dict(self.selected_hyperparameters),
                "candidate_scores": tuple(zip(score_ids, scores, strict=True)),
                "selected_candidate": candidate_ids[selected_index],
                "posterior_method": method,
            }
        )

    @property
    def has_epistemic_uncertainty(self) -> bool:
        return self.parameter_covariance is not None or self.parameter_draws is not None


def fit_strand_displacement_model(
    candidates: Sequence[PreparedStrandDisplacementInference],
    model_selection_traces: Sequence[FluorescenceTimeTrace],
    campaign: ScientificCampaign,
    /,
    *,
    model_selection_source_manifests: Sequence[ReferenceArtifactManifest],
    model_selection_requested_use: Mapping[str, bool],
    max_steps: int = 500,
    gradient_tolerance: float = 1e-6,
    laplace_damping: float = 0.0,
) -> StrandDisplacementModelFit:
    """Run native MAP/Laplace fitting for candidates and freeze the best score."""
    prepared_candidates = tuple(candidates)
    selection = _validate_campaign_role_traces(
        model_selection_traces, campaign, "model_selection"
    )
    selection_source_ids = tuple(
        sorted({source for trace in selection for source in trace.source_manifest_ids})
    )
    selection_manifests, selection_use = _admitted_source_manifests(
        model_selection_source_manifests,
        model_selection_requested_use,
        selection_source_ids,
        "Model-selection fluorescence trace",
    )
    if not dict(selection_use)["training_use"]:
        raise ValueError(
            "Model-selection trace admission did not authorize training use."
        )
    for manifest in selection_manifests:
        manifest.require_rights(training_use=True)
    for prepared in prepared_candidates:
        if isinstance(
            prepared,
            (
                PreparedEffectiveDisplacementInference,
                PreparedMechanisticDisplacementInference,
            ),
        ):
            _training_dependency_manifests(prepared)

    results: list[tuple[PreparedStrandDisplacementInference, StrandPosteriorResult]] = []
    for prepared in prepared_candidates:
        if not isinstance(
            prepared,
            (
                PreparedEffectiveDisplacementInference,
                PreparedMechanisticDisplacementInference,
            ),
        ):
            raise TypeError(
                "candidates must contain prepared strand-displacement inferences."
            )
        problem = prepared.posterior_problem()
        mode = find_map(
            problem,
            max_steps=max_steps,
            gradient_tolerance=gradient_tolerance,
        )
        posterior = fit_laplace(
            problem,
            mode.position,
            damping=laplace_damping,
        )
        results.append((prepared, posterior))
    return StrandDisplacementModelFit(
        results,
        model_selection_traces,
        campaign,
        model_selection_source_manifests=model_selection_source_manifests,
        model_selection_requested_use=model_selection_requested_use,
    )


def predict_locked_fluorescence(
    model: StrandDisplacementForwardModel,
    observation_model: ReporterObservationModel,
    traces: Sequence[FluorescenceTimeTrace],
    /,
) -> StrandDisplacementPrediction:
    """Propagate the frozen model posterior and reporter law over locked clocks."""

    values = tuple(traces)
    if not values:
        raise ValueError("Locked prediction requires at least one trace.")
    if observation_model.campaign_id != model.fit.campaign_id:
        raise ValueError(
            "Locked forward fit and reporter model must share the exact campaign."
        )
    reasons = tuple(reason for trace in values for reason in model.support_reasons(trace))
    if reasons:
        raise ValueError(
            "Locked prediction refused unsupported cases: "
            + ",".join(sorted(set(reasons)))
        )

    def prediction(trace: FluorescenceTimeTrace) -> FluorescencePrediction:
        product = model.product_concentration(trace)
        covariance = model.fit.parameter_covariance
        posterior_draws = model.fit.parameter_draws
        sensitivity = None
        intensity_draws = None
        if posterior_draws is not None:
            intensity_draws = jax.vmap(
                lambda parameters: observation_model.mean_intensity(
                    trace.time_seconds,
                    model.parameterized_product_concentration(trace, parameters),
                )
            )(posterior_draws)
            covariance = None
        elif covariance is not None:
            sensitivity = jax.jacfwd(
                lambda parameters: observation_model.mean_intensity(
                    trace.time_seconds,
                    model.parameterized_product_concentration(trace, parameters),
                )
            )(model.fit.parameter_values)
        limitation = (
            None
            if model.fit.has_epistemic_uncertainty
            else model.uncertainty_limitations[0]
        )
        return observation_model.prediction(
            trace,
            product,
            forward_model_id=model.model_id,
            model_intensity_sensitivity=sensitivity,
            model_parameter_covariance=covariance,
            model_intensity_draws=intensity_draws,
            model_uncertainty_basis_id=model.fit.fit_id,
            model_uncertainty_limitation=limitation,
        )

    predictions = tuple(prediction(trace) for trace in values)
    return StrandDisplacementPrediction(
        predictions,
        tuple(trace.case_id for trace in values),
        model.model_id,
    )


__all__ = [
    "EffectiveDisplacementRateModel",
    "EffectiveFluorescencePosteriorTerm",
    "fit_strand_displacement_model",
    "FluorescencePrediction",
    "MechanisticDisplacementRateModel",
    "MechanisticFluorescencePosteriorTerm",
    "PreparedEffectiveDisplacementInference",
    "PreparedMechanisticDisplacementInference",
    "ReporterCalibration",
    "ReporterObservationModel",
    "SecondaryKineticParameterPlan",
    "StrandDisplacementModelFit",
    "StrandDisplacementPrediction",
    "predict_locked_fluorescence",
    "trace_log_probability",
]
