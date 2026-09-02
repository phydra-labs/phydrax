#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Rates, committors, and correlated uncertainty for path ensembles."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Key
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ReactiveFluxEstimate(StrictModule, NonTrainableState):
    flux: Array
    crossing_count: Array
    observation_time: Array
    valid: Array


def estimate_reactive_flux(
    positive_crossings: ArrayLike,
    observation_time: ArrayLike,
    /,
) -> ReactiveFluxEstimate:
    """Estimate positive reactive flux from fixed observation exposure."""

    raw_crossings = jnp.asarray(positive_crossings)
    crossing_host = np.asarray(raw_crossings)
    if (
        raw_crossings.size == 0
        or np.iscomplexobj(crossing_host)
        or (
            crossing_host.dtype != np.dtype(bool)
            and (
                not np.all(np.isfinite(crossing_host))
                or not np.all((crossing_host == 0) | (crossing_host == 1))
            )
        )
    ):
        raise ValueError(
            "Reactive crossings must be non-empty Boolean or finite binary indicators."
        )
    crossings = raw_crossings.astype(bool).reshape((-1,))
    raw_exposure = jnp.asarray(observation_time)
    if raw_exposure.shape != () or jnp.iscomplexobj(raw_exposure):
        raise ValueError("Reactive flux requires one real scalar exposure.")
    exposure = raw_exposure.astype(float)
    valid = jnp.isfinite(exposure) & (exposure > 0.0)
    count = jnp.sum(crossings, dtype=jnp.int32)
    flux = jnp.where(valid, count.astype(exposure.dtype) / exposure, jnp.nan)
    return ReactiveFluxEstimate(flux, count, exposure, valid)


class TISRateFactorization(StrictModule, NonTrainableState):
    """Flux times conditional interface-crossing probabilities."""

    rate: Array
    log_rate: Array
    reactive_flux: Array
    crossing_probabilities: Array
    log_factors: Array
    valid: Array
    factorization_id: str = eqx.field(static=True)


def factorize_tis_rate(
    reactive_flux: ReactiveFluxEstimate | ArrayLike,
    crossing_probabilities: ArrayLike,
    /,
    *,
    factorization_id: str | None = None,
) -> TISRateFactorization:
    """Form k_AB = flux_A,0 times the ordered TIS crossing factors."""

    flux = jnp.asarray(
        reactive_flux.flux
        if isinstance(reactive_flux, ReactiveFluxEstimate)
        else reactive_flux,
        dtype=float,
    )
    probabilities = jnp.asarray(crossing_probabilities, dtype=flux.dtype)
    if probabilities.ndim != 1 or probabilities.size == 0 or flux.shape != ():
        raise ValueError("TIS rate factors require scalar flux and a non-empty vector.")
    valid = (
        jnp.isfinite(flux)
        & (flux >= 0.0)
        & jnp.all(jnp.isfinite(probabilities))
        & jnp.all((probabilities >= 0.0) & (probabilities <= 1.0))
    )
    log_flux = jnp.where(flux > 0.0, jnp.log(flux), -jnp.inf)
    log_probabilities = jnp.where(probabilities > 0.0, jnp.log(probabilities), -jnp.inf)
    log_factors = jnp.concatenate((log_flux.reshape((1,)), log_probabilities))
    log_rate = jnp.sum(log_factors)
    rate = jnp.where(valid, flux * jnp.prod(probabilities), jnp.nan)
    identity = factorization_id or canonical_fingerprint(
        {"kind": "tis-rate-factorization-v1", "factor_count": int(probabilities.size)}
    )
    if not isinstance(identity, str) or not identity:
        raise ValueError("factorization_id must be non-empty.")
    return TISRateFactorization(
        rate, log_rate, flux, probabilities, log_factors, valid, identity
    )


class CommittorFitPlan(StrictModule, NonTrainableState):
    """Fixed-work weighted logistic committor fit."""

    feature_count: int = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    l2_regularization: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        feature_count: int,
        /,
        *,
        maximum_iterations: int = 1000,
        learning_rate: float = 0.05,
        l2_regularization: float = 0.0,
        tolerance: float = 1.0e-7,
        plan_id: str | None = None,
    ):
        features, iterations = int(feature_count), int(maximum_iterations)
        rate, regularization, tolerance_ = (
            float(learning_rate),
            float(l2_regularization),
            float(tolerance),
        )
        if features <= 0 or iterations <= 0:
            raise ValueError("feature_count and maximum_iterations must be positive.")
        if (
            not all(isfinite(value) for value in (rate, regularization, tolerance_))
            or rate <= 0.0
            or regularization < 0.0
            or tolerance_ <= 0.0
        ):
            raise ValueError("Committor fit controls are invalid.")
        identity = plan_id or canonical_fingerprint(
            {
                "kind": "committor-logistic-fit-v1",
                "feature_count": features,
                "maximum_iterations": iterations,
                "learning_rate": rate.hex(),
                "l2_regularization": regularization.hex(),
                "tolerance": tolerance_.hex(),
            }
        )
        if not isinstance(identity, str) or not identity:
            raise ValueError("plan_id must be non-empty.")
        self.feature_count = features
        self.maximum_iterations = iterations
        self.learning_rate = rate
        self.l2_regularization = regularization
        self.tolerance = tolerance_
        self.plan_id = identity


class CommittorFitResult(StrictModule):
    coefficients: Array
    loss: Array
    gradient_norm: Array
    iterations: Array
    converged: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


def fit_committor(
    plan: CommittorFitPlan,
    features: ArrayLike,
    outcomes: ArrayLike,
    /,
    *,
    weights: ArrayLike | None = None,
) -> CommittorFitResult:
    """Fit q(x)=sigmoid(beta_0 + beta^T phi(x)) by fixed gradient descent."""

    if not isinstance(plan, CommittorFitPlan):
        raise TypeError("plan must be CommittorFitPlan.")
    design = jnp.asarray(features, dtype=float)
    labels = jnp.asarray(outcomes, dtype=design.dtype)
    if (
        design.ndim != 2
        or design.shape[1] != plan.feature_count
        or labels.shape != design.shape[:1]
    ):
        raise ValueError("Committor features and outcomes have incompatible shapes.")
    sample_weights = (
        jnp.ones(labels.shape, dtype=design.dtype)
        if weights is None
        else jnp.asarray(weights, dtype=design.dtype)
    )
    if sample_weights.shape != labels.shape:
        raise ValueError("Committor weights must align with outcomes.")
    host_labels, host_weights = np.asarray(labels), np.asarray(sample_weights)
    if (
        design.shape[0] == 0
        or not np.all(np.isfinite(np.asarray(design)))
        or not np.all(np.isfinite(host_labels))
        or not np.all((host_labels >= 0.0) & (host_labels <= 1.0))
        or not np.all(np.isfinite(host_weights) & (host_weights >= 0.0))
        or float(np.sum(host_weights)) <= 0.0
    ):
        raise ValueError("Committor training data must be finite with non-negative mass.")
    augmented = jnp.concatenate(
        (jnp.ones((design.shape[0], 1), dtype=design.dtype), design), axis=1
    )
    normalizer = jnp.sum(sample_weights)

    def objective(coefficients):
        logits = contract("np,p->n", augmented, coefficients)
        likelihood = (
            jnp.maximum(logits, 0.0)
            - logits * labels
            + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        )
        penalty = 0.5 * plan.l2_regularization * jnp.sum(coefficients[1:] ** 2)
        return jnp.sum(sample_weights * likelihood) / normalizer + penalty

    value_and_gradient = jax.value_and_grad(objective)

    def body(_, carry):
        coefficients, converged, iterations = carry
        loss, gradient = value_and_gradient(coefficients)
        norm = jnp.sqrt(jnp.sum(gradient**2))
        current_converged = norm <= plan.tolerance
        active = ~converged & ~current_converged
        updated = coefficients - plan.learning_rate * gradient
        return (
            jnp.where(active, updated, coefficients),
            converged | current_converged,
            iterations + active.astype(jnp.int32),
        )

    coefficients, _, iterations = jax.lax.fori_loop(
        0,
        plan.maximum_iterations,
        body,
        (
            jnp.zeros((plan.feature_count + 1,), dtype=design.dtype),
            jnp.asarray(False),
            jnp.asarray(0, jnp.int32),
        ),
    )
    loss, gradient = value_and_gradient(coefficients)
    gradient_norm = jnp.sqrt(jnp.sum(gradient**2))
    valid = (
        jnp.all(jnp.isfinite(coefficients))
        & jnp.isfinite(loss)
        & jnp.isfinite(gradient_norm)
    )
    converged = valid & (gradient_norm <= plan.tolerance)
    return CommittorFitResult(
        coefficients,
        loss,
        gradient_norm,
        iterations,
        converged,
        valid,
        plan.plan_id,
    )


def predict_committor(result: CommittorFitResult, features: ArrayLike, /) -> Array:
    if not isinstance(result, CommittorFitResult):
        raise TypeError("result must be CommittorFitResult.")
    values = jnp.asarray(features)
    if values.shape[-1:] != (result.coefficients.shape[0] - 1,):
        raise ValueError("Committor feature count changed.")
    return jax.nn.sigmoid(
        result.coefficients[0] + contract("...p,p->...", values, result.coefficients[1:])
    )


class CorrelatedUncertainty(StrictModule, NonTrainableState):
    mean: Array
    standard_error: Array
    effective_sample_size: Array
    integrated_autocorrelation_time: Array
    lower: Array
    upper: Array
    valid: Array
    method: str = eqx.field(static=True)


def integrated_autocorrelation_time(values: ArrayLike, /, *, maximum_lag: int) -> Array:
    """Estimate integrated autocorrelation time with a positive-sequence window."""

    samples = jnp.asarray(values, dtype=float).reshape((-1,))
    lag_count = int(maximum_lag)
    if samples.size < 2 or lag_count <= 0 or lag_count >= samples.size:
        raise ValueError("maximum_lag must lie in [1, sample_count).")
    if not bool(jnp.all(jnp.isfinite(samples))):
        raise ValueError("Integrated autocorrelation time requires finite samples.")
    centered = samples - jnp.mean(samples)
    variance = jnp.mean(centered**2)
    indices = jnp.arange(samples.size, dtype=jnp.int32)

    def correlation(lag):
        paired = indices < samples.size - lag
        product = centered * centered[jnp.clip(indices + lag, 0, samples.size - 1)]
        return jnp.sum(jnp.where(paired, product, 0.0)) / jnp.maximum(
            jnp.sum(paired) * variance, jnp.finfo(samples.dtype).tiny
        )

    correlations = jax.vmap(correlation)(jnp.arange(1, lag_count + 1, dtype=jnp.int32))
    positive_prefix = jnp.cumprod((correlations > 0.0).astype(jnp.int32)).astype(bool)
    return jnp.maximum(
        1.0 + 2.0 * jnp.sum(jnp.where(positive_prefix, correlations, 0.0)), 1.0
    )


def block_mean_uncertainty(
    values: ArrayLike, /, *, block_size: int
) -> CorrelatedUncertainty:
    """Standard error from non-overlapping block means."""

    samples = jnp.asarray(values, dtype=float).reshape((-1,))
    size = int(block_size)
    block_count = samples.size // size if size > 0 else 0
    if size <= 0 or block_count < 2 or not bool(jnp.all(jnp.isfinite(samples))):
        raise ValueError(
            "Block uncertainty requires finite samples and at least two blocks."
        )
    retained = samples[: block_count * size].reshape((block_count, size))
    block_means = jnp.mean(retained, axis=1)
    mean = jnp.mean(samples)
    standard_error = jnp.std(block_means, ddof=1) / jnp.sqrt(block_count)
    naive_variance = jnp.var(samples, ddof=1)
    tau = jnp.maximum(
        standard_error**2
        * samples.size
        / jnp.maximum(naive_variance, jnp.finfo(samples.dtype).tiny),
        1.0,
    )
    effective = samples.size / tau
    return CorrelatedUncertainty(
        mean,
        standard_error,
        effective,
        tau,
        mean - 1.959963984540054 * standard_error,
        mean + 1.959963984540054 * standard_error,
        jnp.isfinite(standard_error),
        "block-means",
    )


def autocorrelation_uncertainty(
    values: ArrayLike,
    /,
    *,
    maximum_lag: int,
) -> CorrelatedUncertainty:
    samples = jnp.asarray(values, dtype=float).reshape((-1,))
    if not bool(jnp.all(jnp.isfinite(samples))):
        raise ValueError("Autocorrelation uncertainty requires finite samples.")
    tau = integrated_autocorrelation_time(samples, maximum_lag=maximum_lag)
    effective = samples.size / tau
    mean = jnp.mean(samples)
    standard_error = jnp.sqrt(jnp.var(samples, ddof=1) / effective)
    return CorrelatedUncertainty(
        mean,
        standard_error,
        effective,
        tau,
        mean - 1.959963984540054 * standard_error,
        mean + 1.959963984540054 * standard_error,
        jnp.isfinite(standard_error),
        "autocorrelation",
    )


def moving_block_bootstrap_uncertainty(
    key: Key[Array, ""],
    values: ArrayLike,
    /,
    *,
    block_length: int,
    resamples: int,
    confidence: float = 0.95,
) -> CorrelatedUncertainty:
    """Circular moving-block bootstrap with fixed resample and block capacities."""

    samples = jnp.asarray(values, dtype=float).reshape((-1,))
    length, count = int(block_length), int(resamples)
    confidence_ = float(confidence)
    if (
        samples.size < 2
        or length <= 0
        or length > samples.size
        or count <= 1
        or not isfinite(confidence_)
        or confidence_ <= 0.0
        or confidence_ >= 1.0
        or not bool(jnp.all(jnp.isfinite(samples)))
    ):
        raise ValueError("Moving-block bootstrap controls or samples are invalid.")
    blocks = (samples.size + length - 1) // length
    starts = jax.random.randint(key, (count, blocks), 0, samples.size, dtype=jnp.int32)
    offsets = jnp.arange(length, dtype=jnp.int32)
    indices = (starts[..., None] + offsets) % samples.size
    bootstrap = samples[indices].reshape((count, blocks * length))[:, : samples.size]
    means = jnp.mean(bootstrap, axis=1)
    alpha = 0.5 * (1.0 - confidence_)
    lower, upper = jnp.quantile(means, jnp.asarray([alpha, 1.0 - alpha]))
    mean = jnp.mean(samples)
    standard_error = jnp.std(means, ddof=1)
    naive_variance = jnp.var(samples, ddof=1)
    tau = jnp.maximum(
        standard_error**2
        * samples.size
        / jnp.maximum(naive_variance, jnp.finfo(samples.dtype).tiny),
        1.0,
    )
    return CorrelatedUncertainty(
        mean,
        standard_error,
        samples.size / tau,
        tau,
        lower,
        upper,
        jnp.isfinite(standard_error) & jnp.isfinite(lower) & jnp.isfinite(upper),
        "moving-block-bootstrap",
    )


__all__ = [
    "autocorrelation_uncertainty",
    "block_mean_uncertainty",
    "CommittorFitPlan",
    "CommittorFitResult",
    "CorrelatedUncertainty",
    "estimate_reactive_flux",
    "factorize_tis_rate",
    "fit_committor",
    "integrated_autocorrelation_time",
    "moving_block_bootstrap_uncertainty",
    "predict_committor",
    "ReactiveFluxEstimate",
    "TISRateFactorization",
]
