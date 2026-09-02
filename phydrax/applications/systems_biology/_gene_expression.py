#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Telegraph gene expression, count measurement, fitting, and identifiability."""

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._network import (
    CompartmentSpec,
    MassActionPropensity,
    PreparedStoichiometricNetwork,
    PromoterTransitionPropensity,
    SpeciesSpec,
    StoichiometricNetworkPlan,
    StoichiometricProcessSpec,
    StoichiometricRuntime,
)


class CountLikelihoodStatus(IntEnum):
    """Fail-closed count measurement status."""

    SUCCESS = 0
    INVALID_OBSERVATION = 1
    CAPACITY_EXCEEDED = 2
    NUMERICAL_FAILURE = 3


_COUNT_SUCCESS = CountLikelihoodStatus.SUCCESS
_COUNT_INVALID_OBSERVATION = CountLikelihoodStatus.INVALID_OBSERVATION
_COUNT_CAPACITY_EXCEEDED = CountLikelihoodStatus.CAPACITY_EXCEEDED
_COUNT_NUMERICAL_FAILURE = CountLikelihoodStatus.NUMERICAL_FAILURE


def _positive_rate(value: ArrayLike, owner: str, /) -> Array:
    raw = jnp.asarray(value)
    if raw.dtype == jnp.bool_:
        raise TypeError(f"{owner} must be numeric, not boolean.")
    result = raw.astype(float)
    if result.shape != ():
        raise ValueError(f"{owner} must be scalar.")
    host = float(result)
    if not isfinite(host) or host <= 0.0:
        raise ValueError(f"{owner} must be finite and positive.")
    return result


class TelegraphGeneExpressionPlan(StrictModule, NonTrainableState):
    """Two-state promoter with transcription, splicing, and mature-RNA decay."""

    activation_rate: Array
    deactivation_rate: Array
    transcription_rate: Array
    splicing_rate: Array
    degradation_rate: Array
    name: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        activation_rate: ArrayLike,
        deactivation_rate: ArrayLike,
        transcription_rate: ArrayLike,
        splicing_rate: ArrayLike,
        degradation_rate: ArrayLike,
        /,
        *,
        name: str = "telegraph-gene-expression",
    ):
        if not isinstance(name, str) or not name or name.strip() != name:
            raise ValueError("Telegraph model name must be a non-empty, trimmed string.")
        rates = (
            _positive_rate(activation_rate, "Promoter activation rate"),
            _positive_rate(deactivation_rate, "Promoter deactivation rate"),
            _positive_rate(transcription_rate, "Transcription rate"),
            _positive_rate(splicing_rate, "Splicing rate"),
            _positive_rate(degradation_rate, "Degradation rate"),
        )
        self.activation_rate = rates[0]
        self.deactivation_rate = rates[1]
        self.transcription_rate = rates[2]
        self.splicing_rate = rates[3]
        self.degradation_rate = rates[4]
        self.name = name
        self.plan_id = canonical_fingerprint(
            {
                "kind": "telegraph-gene-expression-plan",
                "name": name,
                "rates": [float(value) for value in rates],
            }
        )

    def prepare(self) -> PreparedTelegraphGeneExpression:
        return PreparedTelegraphGeneExpression(self)


class TelegraphStationaryMoments(StrictModule):
    """Exact stationary first and second moments for promoter and RNA counts."""

    promoter_mean: Array
    nascent_mean: Array
    mature_mean: Array
    promoter_variance: Array
    nascent_variance: Array
    mature_variance: Array
    promoter_nascent_covariance: Array
    promoter_mature_covariance: Array
    nascent_mature_covariance: Array
    finite: Array
    valid: Array
    model_id: str = eqx.field(static=True)

    @property
    def fitting_vector(self) -> Array:
        return jnp.stack(
            (
                self.nascent_mean,
                self.mature_mean,
                self.nascent_variance,
                self.mature_variance,
                self.nascent_mature_covariance,
            )
        )


class TelegraphFitTarget(StrictModule, NonTrainableState):
    """Five observable stationary moments with positive uncertainty scales."""

    moments: Array
    standard_errors: Array
    target_id: str = eqx.field(static=True)

    def __init__(self, moments: ArrayLike, standard_errors: ArrayLike, /):
        values = jnp.asarray(moments, dtype=float)
        errors = jnp.asarray(standard_errors, dtype=values.dtype)
        if values.shape != (5,) or errors.shape != (5,):
            raise ValueError(
                "Fit target moments and standard_errors must have shape (5,)."
            )
        values_host = np.asarray(values)
        errors_host = np.asarray(errors)
        if np.any(~np.isfinite(values_host)) or np.any(~np.isfinite(errors_host)):
            raise ValueError("Fit target values must be finite.")
        if np.any(errors_host <= 0.0):
            raise ValueError("Fit target standard_errors must be positive.")
        self.moments = values
        self.standard_errors = errors
        self.target_id = canonical_fingerprint(
            {
                "kind": "telegraph-fit-target",
                "moments": array_tree_fingerprint(values_host),
                "standard_errors": array_tree_fingerprint(errors_host),
            }
        )


class TelegraphFitEvaluation(StrictModule):
    """Differentiable moment-fitting objective with fail-closed parameter evidence."""

    objective: Array
    rates: Array
    residuals: Array
    finite: Array
    valid: Array
    model_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)


class IdentifiabilityEvidence(StrictModule, NonTrainableState):
    """Host rank evidence for the local log-rate-to-moment sensitivity map."""

    sensitivity: Array
    singular_values: Array
    rank: int = eqx.field(static=True)
    parameter_count: int = eqx.field(static=True)
    condition_number: Array
    locally_identifiable: Array
    model_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class CountMeasurementPlan(StrictModule, NonTrainableState):
    """Binomial capture followed by independent Poisson false-positive counts."""

    capture_probability: Array
    background_rate: Array
    observation_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        capture_probability: ArrayLike,
        background_rate: ArrayLike = 0.0,
        /,
        *,
        observation_capacity: int,
    ):
        capture_raw = jnp.asarray(capture_probability)
        background_raw = jnp.asarray(background_rate)
        if capture_raw.dtype == jnp.bool_ or background_raw.dtype == jnp.bool_:
            raise TypeError(
                "Capture probability and background rate must not be boolean."
            )
        capture = capture_raw.astype(float)
        background = background_raw.astype(capture.dtype)
        if capture.shape != () or background.shape != ():
            raise ValueError("Capture probability and background rate must be scalar.")
        capture_host = float(capture)
        background_host = float(background)
        if isinstance(observation_capacity, bool) or not isinstance(
            observation_capacity, (int, np.integer)
        ):
            raise ValueError("observation_capacity must be an integer.")
        capacity = int(observation_capacity)
        if not isfinite(capture_host) or not 0.0 <= capture_host <= 1.0:
            raise ValueError("capture_probability must be finite and in [0, 1].")
        if not isfinite(background_host) or background_host < 0.0:
            raise ValueError("background_rate must be finite and nonnegative.")
        if capacity < 0:
            raise ValueError("observation_capacity must be nonnegative.")
        self.capture_probability = capture
        self.background_rate = background
        self.observation_capacity = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "count-measurement-plan",
                "capture": capture_host,
                "background": background_host,
                "capacity": capacity,
            }
        )

    def prepare(self) -> PreparedCountMeasurement:
        return PreparedCountMeasurement(self)


class CountLikelihoodEvaluation(StrictModule):
    """Exact finite-sum count log likelihood and explicit capacity status."""

    log_likelihood: Array
    valid: Array
    status: Array
    measurement_id: str = eqx.field(static=True)


class PreparedCountMeasurement(StrictModule, NonTrainableState):
    """Fixed-capacity exact convolution of capture and background count laws."""

    plan: CountMeasurementPlan
    capture_indices: Array
    measurement_id: str = eqx.field(static=True)

    def __init__(self, plan: CountMeasurementPlan, /):
        if not isinstance(plan, CountMeasurementPlan):
            raise TypeError("plan must be CountMeasurementPlan.")
        self.plan = plan
        self.capture_indices = jnp.arange(plan.observation_capacity + 1, dtype=float)
        self.measurement_id = canonical_fingerprint(
            {"kind": "prepared-count-measurement", "plan": plan.plan_id}
        )

    def log_likelihood(
        self, observed: ArrayLike, latent_count: ArrayLike, /
    ) -> CountLikelihoodEvaluation:
        observed_value, latent_value = jnp.broadcast_arrays(
            jnp.asarray(observed, dtype=float), jnp.asarray(latent_count, dtype=float)
        )
        valid_observation = (
            jnp.isfinite(observed_value)
            & jnp.isfinite(latent_value)
            & (observed_value >= 0.0)
            & (latent_value >= 0.0)
            & (observed_value == jnp.floor(observed_value))
            & (latent_value == jnp.floor(latent_value))
        )
        capacity_valid = observed_value <= self.plan.observation_capacity
        captured = self.capture_indices.reshape(
            (1,) * observed_value.ndim + (self.capture_indices.shape[0],)
        )
        observed_axis = observed_value[..., None]
        latent_axis = latent_value[..., None]
        background = observed_axis - captured
        capture_valid = (
            (captured <= observed_axis) & (captured <= latent_axis) & (background >= 0.0)
        )
        positive_indices = self.capture_indices[1:]
        numerators = latent_value[..., None] - positive_indices + 1.0
        safe_numerators = jnp.where(numerators > 0.0, numerators, 1.0)
        increments = jnp.log(safe_numerators) - jnp.log(positive_indices)
        log_combination = jnp.concatenate(
            (
                jnp.zeros(observed_value.shape + (1,), dtype=observed_value.dtype),
                jnp.cumsum(increments, axis=-1),
            ),
            axis=-1,
        )
        capture_log_probability = (
            log_combination
            + jsp.special.xlogy(captured, self.plan.capture_probability)
            + jsp.special.xlog1py(latent_axis - captured, -self.plan.capture_probability)
        )
        background_log_probability = (
            jsp.special.xlogy(background, self.plan.background_rate)
            - self.plan.background_rate
            - jsp.special.gammaln(background + 1.0)
        )
        terms = jnp.where(
            capture_valid,
            capture_log_probability + background_log_probability,
            -jnp.inf,
        )
        value = jsp.special.logsumexp(terms, axis=-1)
        computed_valid = ~jnp.isnan(value) & (value != jnp.inf)
        valid = valid_observation & capacity_valid & computed_valid
        status = jnp.where(
            ~valid_observation,
            _COUNT_INVALID_OBSERVATION,
            jnp.where(
                ~capacity_valid,
                _COUNT_CAPACITY_EXCEEDED,
                jnp.where(
                    ~computed_valid,
                    _COUNT_NUMERICAL_FAILURE,
                    _COUNT_SUCCESS,
                ),
            ),
        )
        return CountLikelihoodEvaluation(
            jnp.where(valid, value, jnp.nan),
            valid,
            jnp.asarray(status, dtype=jnp.int32),
            self.measurement_id,
        )

    def observed_moments(
        self, latent_mean: ArrayLike, latent_variance: ArrayLike, /
    ) -> tuple[Array, Array]:
        dtype = self.plan.capture_probability.dtype
        mean = jnp.asarray(latent_mean, dtype=dtype)
        variance = jnp.asarray(latent_variance, dtype=dtype)
        capture = self.plan.capture_probability
        background = self.plan.background_rate
        observed_mean = capture * mean + background
        observed_variance = (
            capture * capture * variance + capture * (1.0 - capture) * mean + background
        )
        return observed_mean, observed_variance


class PreparedTelegraphGeneExpression(StrictModule, NonTrainableState):
    """Prepared fixed-topology telegraph network and analytic inference surface."""

    plan: TelegraphGeneExpressionPlan
    network: PreparedStoichiometricNetwork
    rates: Array
    model_id: str = eqx.field(static=True)
    exact_path_differentiable: bool = eqx.field(static=True)
    analytic_moments_differentiable: bool = eqx.field(static=True)

    def __init__(self, plan: TelegraphGeneExpressionPlan, /):
        if not isinstance(plan, TelegraphGeneExpressionPlan):
            raise TypeError("plan must be TelegraphGeneExpressionPlan.")
        compartment = CompartmentSpec("nucleus-cytosol", 1.0, unit="cell")
        species = (
            SpeciesSpec("promoter_off", compartment.name),
            SpeciesSpec("promoter_on", compartment.name),
            SpeciesSpec("nascent", compartment.name),
            SpeciesSpec("mature", compartment.name),
        )
        processes = (
            StoichiometricProcessSpec(
                "promoter_activation",
                {"promoter_off": -1, "promoter_on": 1},
                PromoterTransitionPropensity(plan.activation_rate, "promoter_off"),
            ),
            StoichiometricProcessSpec(
                "promoter_deactivation",
                {"promoter_off": 1, "promoter_on": -1},
                PromoterTransitionPropensity(plan.deactivation_rate, "promoter_on"),
            ),
            StoichiometricProcessSpec(
                "transcription",
                {"nascent": 1},
                MassActionPropensity(plan.transcription_rate, {"promoter_on": 1}),
            ),
            StoichiometricProcessSpec(
                "splicing",
                {"nascent": -1, "mature": 1},
                MassActionPropensity(plan.splicing_rate, {"nascent": 1}),
            ),
            StoichiometricProcessSpec(
                "mature_degradation",
                {"mature": -1},
                MassActionPropensity(plan.degradation_rate, {"mature": 1}),
            ),
        )
        network = StoichiometricNetworkPlan(
            plan.name,
            (compartment,),
            species,
            processes,
            stoichiometry_capacity=2,
        ).prepare()
        rates = jnp.stack(
            (
                plan.activation_rate,
                plan.deactivation_rate,
                plan.transcription_rate,
                plan.splicing_rate,
                plan.degradation_rate,
            )
        )
        self.plan = plan
        self.network = network
        self.rates = rates
        self.model_id = canonical_fingerprint(
            {
                "kind": "prepared-telegraph-gene-expression",
                "plan": plan.plan_id,
                "network": network.network_id,
            }
        )
        self.exact_path_differentiable = False
        self.analytic_moments_differentiable = True

    def initial_state(
        self, /, *, promoter_on: bool = False, nascent: int = 0, mature: int = 0
    ) -> Array:
        if not isinstance(promoter_on, bool):
            raise TypeError("promoter_on must be bool.")
        if (
            isinstance(nascent, bool)
            or isinstance(mature, bool)
            or not isinstance(nascent, (int, np.integer))
            or not isinstance(mature, (int, np.integer))
            or nascent < 0
            or mature < 0
        ):
            raise ValueError("Nascent and mature counts must be nonnegative integers.")
        return self.network.initial_state(
            [
                0.0 if promoter_on else 1.0,
                1.0 if promoter_on else 0.0,
                nascent,
                mature,
            ]
        )

    def runtime(self, rates: ArrayLike | None = None, /) -> StoichiometricRuntime:
        rate_values = (
            self.rates if rates is None else jnp.asarray(rates, dtype=self.rates.dtype)
        )
        if rate_values.shape != (5,):
            raise ValueError("Telegraph rates must have shape (5,).")
        parameters = self.network.propensity_parameters.at[:, 0].set(rate_values)
        return StoichiometricRuntime(parameters)

    def stationary_moments(
        self, rates: ArrayLike | None = None, /
    ) -> TelegraphStationaryMoments:
        values = (
            self.rates if rates is None else jnp.asarray(rates, dtype=self.rates.dtype)
        )
        if values.shape != (5,):
            raise ValueError("Telegraph rates must have shape (5,).")
        moments = _stationary_moment_values(values)
        finite = jnp.all(jnp.isfinite(moments))
        valid = finite & jnp.all(values > 0.0)
        return TelegraphStationaryMoments(
            *tuple(moments),
            finite,
            valid,
            self.model_id,
        )

    def fitting_objective(
        self, log_rates: ArrayLike, target: TelegraphFitTarget, /
    ) -> Array:
        return self.fit_evaluation(log_rates, target).objective

    def fit_evaluation(
        self, log_rates: ArrayLike, target: TelegraphFitTarget, /
    ) -> TelegraphFitEvaluation:
        if not isinstance(target, TelegraphFitTarget):
            raise TypeError("target must be TelegraphFitTarget.")
        log_values = jnp.asarray(log_rates, dtype=self.rates.dtype)
        if log_values.shape != (5,):
            raise ValueError("log_rates must have shape (5,).")
        rates = jnp.exp(log_values)
        vector = _stationary_fitting_vector(rates)
        residuals = (vector - target.moments) / target.standard_errors
        objective = 0.5 * jnp.sum(residuals * residuals)
        finite = jnp.isfinite(objective) & jnp.all(jnp.isfinite(residuals))
        valid = finite & jnp.all(rates > 0.0)
        return TelegraphFitEvaluation(
            jnp.where(valid, objective, jnp.inf),
            rates,
            residuals,
            finite,
            valid,
            self.model_id,
            target.target_id,
        )

    def identifiability_evidence(
        self, log_rates: ArrayLike | None = None, /
    ) -> IdentifiabilityEvidence:
        values = (
            jnp.log(self.rates)
            if log_rates is None
            else jnp.asarray(log_rates, dtype=self.rates.dtype)
        )
        if values.shape != (5,):
            raise ValueError("log_rates must have shape (5,).")
        host_values = np.asarray(values)
        host_rates = np.asarray(jnp.exp(values))
        if (
            np.any(~np.isfinite(host_values))
            or np.any(~np.isfinite(host_rates))
            or np.any(host_rates <= 0.0)
        ):
            raise ValueError("log_rates must map to finite, positive model rates.")
        sensitivity = jax.jacrev(lambda logs: _stationary_fitting_vector(jnp.exp(logs)))(
            values
        )
        device_sensitivity = np.asarray(sensitivity)
        host = device_sensitivity.astype(float)
        singular_values = np.linalg.svd(host, compute_uv=False)
        threshold = (
            10.0
            * max(host.shape)
            * np.finfo(device_sensitivity.dtype).eps
            * max(float(np.max(singular_values, initial=0.0)), 1.0)
        )
        rank = int(np.sum(singular_values > threshold))
        condition = (
            np.inf
            if singular_values[-1] <= threshold
            else float(singular_values[0] / singular_values[-1])
        )
        evidence_id = canonical_fingerprint(
            {
                "kind": "telegraph-identifiability",
                "model": self.model_id,
                "log_rates": array_tree_fingerprint(np.asarray(values)),
                "sensitivity": array_tree_fingerprint(host),
            }
        )
        return IdentifiabilityEvidence(
            sensitivity,
            jnp.asarray(singular_values),
            rank,
            5,
            jnp.asarray(condition),
            jnp.asarray(rank == 5),
            self.model_id,
            evidence_id,
        )

    def evidence_fields(self) -> dict[str, object]:
        """Return exact host fields accepted by biological evidence bindings."""
        fields = self.network.evidence_fields()
        fields.update(
            {
                "telegraph.plan_id": self.plan.plan_id,
                "telegraph.prepared_id": self.model_id,
                "telegraph.name": self.plan.name,
                "telegraph.activation_rate": float(self.plan.activation_rate),
                "telegraph.deactivation_rate": float(self.plan.deactivation_rate),
                "telegraph.transcription_rate": float(self.plan.transcription_rate),
                "telegraph.splicing_rate": float(self.plan.splicing_rate),
                "telegraph.degradation_rate": float(self.plan.degradation_rate),
            }
        )
        return fields

    def evidence_units(self) -> dict[str, str]:
        """Return exact units aligned with every biological evidence field."""
        units = self.network.evidence_units()
        units.update(
            {
                "telegraph.plan_id": "identity",
                "telegraph.prepared_id": "identity",
                "telegraph.name": "label",
                "telegraph.activation_rate": "s^-1",
                "telegraph.deactivation_rate": "s^-1",
                "telegraph.transcription_rate": "s^-1",
                "telegraph.splicing_rate": "s^-1",
                "telegraph.degradation_rate": "s^-1",
            }
        )
        return units


def _stationary_moment_values(rates: Array, /) -> Array:
    activation, deactivation, transcription, splicing, degradation = rates
    switching = activation + deactivation
    promoter_mean = activation / switching
    promoter_variance = promoter_mean * (1.0 - promoter_mean)
    nascent_mean = transcription * promoter_mean / splicing
    mature_mean = transcription * promoter_mean / degradation
    promoter_nascent = transcription * promoter_variance / (switching + splicing)
    promoter_mature = splicing * promoter_nascent / (switching + degradation)
    nascent_variance = nascent_mean + transcription * promoter_nascent / splicing
    nascent_mature = (
        transcription * promoter_mature
        + splicing * nascent_variance
        - transcription * promoter_mean
    ) / (splicing + degradation)
    mature_variance = mature_mean + splicing * nascent_mature / degradation
    return jnp.stack(
        (
            promoter_mean,
            nascent_mean,
            mature_mean,
            promoter_variance,
            nascent_variance,
            mature_variance,
            promoter_nascent,
            promoter_mature,
            nascent_mature,
        )
    )


def _stationary_fitting_vector(rates: Array, /) -> Array:
    moments = _stationary_moment_values(rates)
    return moments[jnp.asarray([1, 2, 4, 5, 8], dtype=jnp.int32)]


__all__ = [
    "CountLikelihoodEvaluation",
    "CountLikelihoodStatus",
    "CountMeasurementPlan",
    "IdentifiabilityEvidence",
    "PreparedCountMeasurement",
    "PreparedTelegraphGeneExpression",
    "TelegraphFitEvaluation",
    "TelegraphFitTarget",
    "TelegraphGeneExpressionPlan",
    "TelegraphStationaryMoments",
]
