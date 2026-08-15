#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import opt_einsum as oe
from jaxtyping import Array

from ..._model import AbstractArrayModel
from ..._strict import StrictModule
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitResult,
    GradientContract,
    ML_INSUFFICIENT_DATA,
    ML_NONCONVERGED,
    ML_NONFINITE,
    ML_SUCCESS,
)


CovarianceType: TypeAlias = Literal["full", "tied", "diagonal", "spherical"]
MixtureInitialization: TypeAlias = Literal["random", "first"]
EmptyComponentPolicy: TypeAlias = Literal["retain", "reseed", "error"]


def _real_dtype(dtype: jnp.dtype) -> jnp.dtype:
    return jnp.empty((), dtype=dtype).real.dtype


def _validated_scalar(value: Any, name: str, /, *, allow_zero: bool) -> Array:
    scalar = jnp.asarray(value)
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    relation = "nonnegative" if allow_zero else "positive"
    invalid = ~jnp.isfinite(scalar) | (scalar < 0.0 if allow_zero else scalar <= 0.0)
    return eqx.error_if(scalar, invalid, f"{name} must be finite and {relation}.")


def _nonnegative_scalar(value: Any, name: str, /) -> Array:
    return _validated_scalar(value, name, allow_zero=True)


def _positive_scalar(value: Any, name: str, /) -> Array:
    return _validated_scalar(value, name, allow_zero=False)


def _active_data(batch: MLBatch, policy: WeightPolicy) -> tuple[Array, Array, Array]:
    x = batch.dense_features()
    raw = batch.effective_weight(policy)
    complete = jnp.all(batch.feature_mask, axis=-1)
    finite_x = jnp.all(jnp.isfinite(x), axis=-1)
    weights_ok = jnp.isfinite(raw) & (raw >= 0.0)
    active = complete & finite_x & weights_ok
    w = jnp.where(active, raw, 0.0).astype(_real_dtype(x.dtype))
    x = jnp.where(active[..., None], x, 0)
    invalid = jnp.any(batch.sample_mask & (~weights_ok | (complete & ~finite_x)), axis=-1)
    return x, w, invalid


def _regularized_geometry(
    covariance: Array, regularization: Array
) -> tuple[Array, Array, Array, Array]:
    p = covariance.shape[-1]
    real_dtype = _real_dtype(covariance.dtype)
    covariance = (covariance + jnp.conj(jnp.swapaxes(covariance, -1, -2))) * 0.5
    trace_scale = jnp.maximum(
        jnp.real(jnp.trace(covariance, axis1=-2, axis2=-1)) / p, 1.0
    )
    floor = jnp.maximum(
        regularization * trace_scale, jnp.finfo(real_dtype).eps * trace_scale
    )
    values, vectors = jnp.linalg.eigh(covariance)
    singular = jnp.any(values <= floor[..., None], axis=-1)
    values = jnp.maximum(values, floor[..., None])
    covariance = oe.contract(
        "...ik,...k,...jk->...ij", vectors, values, jnp.conj(vectors)
    )
    precision = oe.contract(
        "...ik,...k,...jk->...ij", vectors, 1.0 / values, jnp.conj(vectors)
    )
    log_det = jnp.sum(jnp.log(values), axis=-1)
    return covariance, precision, log_det, singular


def _component_log_prob(
    x: Array, means: Array, precision: Array, log_det: Array
) -> Array:
    difference = x[..., :, None, :] - means[..., None, :, :]
    quadratic = jnp.real(
        oe.contract(
            "...nki,...kij,...nkj->...nk", jnp.conj(difference), precision, difference
        )
    )
    if jnp.issubdtype(x.dtype, jnp.complexfloating):
        return -(quadratic + (x.shape[-1] * jnp.log(jnp.pi) + log_det)[..., None, :])
    return -0.5 * (
        quadratic + (x.shape[-1] * jnp.log(2.0 * jnp.pi) + log_det)[..., None, :]
    )


def _responsibilities(
    x: Array, mixing: Array, means: Array, precision: Array, log_det: Array
) -> tuple[Array, Array]:
    component = _component_log_prob(x, means, precision, log_det)
    logits = (
        component
        + jnp.log(jnp.maximum(mixing, jnp.finfo(mixing.dtype).tiny))[..., None, :]
    )
    log_normalizer = jsp.special.logsumexp(logits, axis=-1)
    responsibilities = jnp.exp(logits - log_normalizer[..., None])
    responsibilities = jnp.where(jnp.isfinite(responsibilities), responsibilities, 0.0)
    return responsibilities, log_normalizer


def _covariance_structure(
    covariance: Array, mass: Array, covariance_type: CovarianceType
) -> Array:
    p = covariance.shape[-1]
    eye = jnp.eye(p, dtype=covariance.dtype)
    if covariance_type == "full":
        return covariance
    if covariance_type == "diagonal":
        diagonal = jnp.real(jnp.diagonal(covariance, axis1=-2, axis2=-1))
        return diagonal[..., :, :, None] * eye
    if covariance_type == "spherical":
        variance = jnp.real(jnp.trace(covariance, axis1=-2, axis2=-1)) / p
        return variance[..., :, None, None] * eye
    total = jnp.maximum(jnp.sum(mass, axis=-1), jnp.finfo(mass.dtype).tiny)
    tied = oe.contract("...k,...kij->...ij", mass, covariance) / total[..., None, None]
    return jnp.broadcast_to(tied[..., None, :, :], covariance.shape)


def _initial_means(
    x: Array,
    w: Array,
    component_count: int,
    initialization: MixtureInitialization,
    key: Any,
) -> Array:
    case_shape = x.shape[:-2]
    n, p = x.shape[-2:]
    case_count = 1
    for size in case_shape:
        case_count *= size
    flat_x = x.reshape((case_count, n, p))
    flat_w = w.reshape((case_count, n))
    if initialization == "random":
        if key is None:
            raise ValueError(
                "random mixture initialization requires an explicit JAX key."
            )
        keys = jax.random.split(key, case_count)

        def choose(values, weights, case_key):
            logits = jnp.where(weights > 0.0, jnp.log(weights), -jnp.inf)
            indices = jax.random.categorical(case_key, logits, shape=(component_count,))
            return values[indices]

        means = jax.vmap(choose)(flat_x, flat_w, keys)
    else:
        indices = jnp.floor(jnp.arange(component_count) * n / component_count).astype(
            jnp.int32
        )
        means = flat_x[:, indices, :]
    return means.reshape(case_shape + (component_count, p))


def _fit_gaussian_mixture(
    batch: MLBatch,
    *,
    component_count: int,
    covariance_type: CovarianceType,
    max_iterations: int,
    tolerance: Array,
    regularization: Array,
    initialization: MixtureInitialization,
    empty_policy: EmptyComponentPolicy,
    weight_policy: WeightPolicy,
    key: Any,
    concentration: Array | None = None,
    mean_precision: Array | float = 0.0,
    degrees_of_freedom: Array | float = 0.0,
) -> tuple[Array, ...]:
    if component_count > batch.sample_count:
        raise ValueError("component_count cannot exceed fixed sample capacity.")
    x, w, invalid = _active_data(batch, weight_policy)
    p = batch.feature_count
    mass_total = jnp.sum(w, axis=-1)
    mean_global = (
        oe.contract("...n,...nf->...f", w, x)
        / jnp.maximum(mass_total, jnp.finfo(w.dtype).tiny)[..., None]
    )
    centered = jnp.where(w[..., None] > 0.0, x - mean_global[..., None, :], 0)
    covariance_global = oe.contract(
        "...ni,...n,...nj->...ij", jnp.conj(centered), w, centered
    )
    covariance_global = (
        covariance_global
        / jnp.maximum(mass_total, jnp.finfo(w.dtype).tiny)[..., None, None]
    )
    covariance_global = jnp.broadcast_to(
        covariance_global[..., None, :, :], batch.case_shape + (component_count, p, p)
    )
    covariance, precision, log_det, _ = _regularized_geometry(
        covariance_global, regularization
    )
    means = _initial_means(x, w, component_count, initialization, key)
    mixing = jnp.full(
        batch.case_shape + (component_count,), 1.0 / component_count, dtype=w.dtype
    )
    initial_objective = jnp.full(batch.case_shape, jnp.inf, dtype=w.dtype)
    empty_seen = jnp.zeros(batch.case_shape, dtype=bool)
    singular_seen = jnp.zeros(batch.case_shape, dtype=bool)
    concentration_ = 0.0 if concentration is None else concentration

    def em_step(_, state):
        (
            mixing,
            means,
            covariance,
            precision,
            log_det,
            previous_objective,
            empty_seen,
            singular_seen,
        ) = state
        responsibility, log_probability = _responsibilities(
            x, mixing, means, precision, log_det
        )
        weighted = w[..., :, None] * responsibility
        component_mass = jnp.sum(weighted, axis=-2)
        empty = component_mass <= jnp.finfo(w.dtype).eps * jnp.maximum(
            mass_total[..., None], 1.0
        )
        safe_mass = jnp.maximum(component_mass, jnp.finfo(w.dtype).tiny)
        empirical_means = (
            oe.contract("...nk,...nf->...kf", weighted, x) / safe_mass[..., :, None]
        )
        if concentration is not None:
            posterior_mass = component_mass + mean_precision
            next_means = (
                component_mass[..., :, None] * empirical_means
                + mean_precision * mean_global[..., None, :]
            ) / jnp.maximum(posterior_mass, jnp.finfo(w.dtype).tiny)[..., :, None]
        else:
            next_means = empirical_means
        difference = x[..., :, None, :] - next_means[..., None, :, :]
        next_covariance = oe.contract(
            "...nki,...nk,...nkj->...kij", jnp.conj(difference), weighted, difference
        )
        if concentration is not None:
            prior_difference = next_means - mean_global[..., None, :]
            prior_scatter = oe.contract(
                "...ki,...kj->...kij", jnp.conj(prior_difference), prior_difference
            )
            prior_scale = covariance_global + mean_precision * prior_scatter
            denominator = safe_mass + degrees_of_freedom + p + 1.0
            next_covariance = (next_covariance + prior_scale) / denominator[
                ..., :, None, None
            ]
            next_mixing = (component_mass + concentration_) / jnp.maximum(
                mass_total + component_count * concentration_, jnp.finfo(w.dtype).tiny
            )[..., None]
        else:
            next_covariance = next_covariance / safe_mass[..., :, None, None]
            next_mixing = (
                component_mass
                / jnp.maximum(mass_total, jnp.finfo(w.dtype).tiny)[..., None]
            )

        if empty_policy == "reseed":
            current_difference = x[..., :, None, :] - means[..., None, :, :]
            distance = jnp.real(
                jnp.sum(jnp.conj(current_difference) * current_difference, axis=-1)
            )
            farthest_score = jnp.min(distance, axis=-1)
            farthest_score = jnp.where(w > 0.0, farthest_score, -jnp.inf)
            candidates = jnp.argsort(-farthest_score, axis=-1, stable=True)[
                ..., :component_count
            ]
            candidate_means = jnp.take_along_axis(x, candidates[..., :, None], axis=-2)
            next_means = jnp.where(empty[..., :, None], candidate_means, next_means)
            next_covariance = jnp.where(
                empty[..., :, None, None], covariance_global, next_covariance
            )
            reset_mass = jnp.maximum(mass_total / component_count, jnp.finfo(w.dtype).eps)
            next_mixing = jnp.where(empty, reset_mass[..., None], next_mixing)
            next_mixing = next_mixing / jnp.sum(next_mixing, axis=-1, keepdims=True)
        else:
            next_means = jnp.where(empty[..., :, None], means, next_means)
            next_covariance = jnp.where(
                empty[..., :, None, None], covariance, next_covariance
            )
            next_mixing = jnp.where(empty, mixing, next_mixing)
            next_mixing = next_mixing / jnp.sum(next_mixing, axis=-1, keepdims=True)
        next_covariance = _covariance_structure(
            next_covariance, safe_mass, covariance_type
        )

        next_covariance, next_precision, next_log_det, singular = _regularized_geometry(
            next_covariance, regularization
        )
        objective = -jnp.sum(w * log_probability, axis=-1) / jnp.maximum(
            mass_total, jnp.finfo(w.dtype).tiny
        )
        return (
            next_mixing,
            next_means,
            next_covariance,
            next_precision,
            next_log_det,
            objective,
            empty_seen | jnp.any(empty, axis=-1),
            singular_seen | jnp.any(singular, axis=-1),
        )

    state = (
        mixing,
        means,
        covariance,
        precision,
        log_det,
        initial_objective,
        empty_seen,
        singular_seen,
    )
    (
        mixing,
        means,
        covariance,
        precision,
        log_det,
        objective,
        empty_seen,
        singular_seen,
    ) = jax.lax.fori_loop(0, max_iterations, em_step, state)
    responsibilities, log_probability = _responsibilities(
        x, mixing, means, precision, log_det
    )
    final_objective = -jnp.sum(w * log_probability, axis=-1) / jnp.maximum(
        mass_total, jnp.finfo(w.dtype).tiny
    )
    objective_delta = jnp.abs(final_objective - objective)
    converged = objective_delta <= tolerance * jnp.maximum(1.0, jnp.abs(objective))
    component_mass = jnp.sum(w[..., :, None] * responsibilities, axis=-2)
    valid_data = (mass_total > 0.0) & (jnp.sum(w > 0.0, axis=-1) >= component_count)
    finite = jnp.isfinite(final_objective) & jnp.all(jnp.isfinite(means), axis=(-2, -1))
    empty_error = empty_seen & (empty_policy == "error")
    valid = valid_data & finite & ~invalid & ~empty_error & converged
    status = jnp.where(
        invalid | ~finite,
        ML_NONFINITE,
        jnp.where(
            ~valid_data | empty_error,
            ML_INSUFFICIENT_DATA,
            jnp.where(~converged, ML_NONCONVERGED, ML_SUCCESS),
        ),
    )
    squared_mass = jnp.sum(w * w, axis=-1)
    effective_samples = jnp.where(
        squared_mass > 0.0, mass_total * mass_total / squared_mass, 0.0
    )
    return (
        mixing,
        means,
        covariance,
        precision,
        log_det,
        responsibilities,
        final_objective,
        objective_delta,
        component_mass,
        empty_seen,
        singular_seen,
        converged,
        valid,
        status,
        effective_samples,
    )


class MixtureDiagnostics(StrictModule):
    valid: Array
    status: Array
    negative_log_likelihood: Array
    objective_delta: Array
    iterations: Array
    effective_samples: Array
    component_mass: Array
    empty_components_seen: Array
    singular_components_seen: Array
    converged: Array
    method: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        valid: Array,
        status: Array,
        negative_log_likelihood: Array,
        objective_delta: Array,
        iterations: int,
        effective_samples: Array,
        component_mass: Array,
        empty_components_seen: Array,
        singular_components_seen: Array,
        converged: Array,
        method: str,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.negative_log_likelihood = jnp.asarray(negative_log_likelihood)
        self.objective_delta = jnp.asarray(objective_delta)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.effective_samples = jnp.asarray(effective_samples)
        self.component_mass = jnp.asarray(component_mass)
        self.empty_components_seen = jnp.asarray(empty_components_seen, dtype=bool)
        self.singular_components_seen = jnp.asarray(singular_components_seen, dtype=bool)
        self.converged = jnp.asarray(converged, dtype=bool)
        self.method = str(method)


def _input_sample_ndim(values: Array, case_shape: tuple[int, ...], in_size: int) -> int:
    minimum_rank = len(case_shape) + 1
    if values.ndim < minimum_rank:
        raise ValueError("case axes and the final feature axis must be distinct.")
    if values.shape[: len(case_shape)] != case_shape or values.shape[-1] != in_size:
        raise ValueError("input must have shape case + sample_shape + (feature,).")
    return values.ndim - minimum_rank


class GaussianMixtureModel(AbstractArrayModel):
    mixing_weights: Array
    means: Array
    covariance: Array
    precision: Array
    log_determinant: Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    covariance_type: CovarianceType = eqx.field(static=True)

    def __init__(
        self,
        mixing_weights: Array,
        means: Array,
        covariance: Array,
        precision: Array,
        log_determinant: Array,
        /,
        *,
        covariance_type: CovarianceType,
    ):
        self.mixing_weights = jnp.asarray(mixing_weights)
        self.means = jnp.asarray(means)
        self.covariance = jnp.asarray(covariance)
        self.precision = jnp.asarray(precision)
        self.log_determinant = jnp.asarray(log_determinant)
        self.in_size = self.means.shape[-1]
        self.out_size = self.means.shape[-2]
        self.case_shape = self.means.shape[:-2]
        self.covariance_type = covariance_type

    def component_log_prob(self, x: Any, /) -> Array:
        values = jnp.asarray(x)
        sample_ndim = _input_sample_ndim(values, self.case_shape, self.in_size)
        means = self.means.reshape(
            self.case_shape + (1,) * sample_ndim + self.means.shape[-2:]
        )
        precision = self.precision.reshape(
            self.case_shape + (1,) * sample_ndim + self.precision.shape[-3:]
        )
        log_det = self.log_determinant.reshape(
            self.case_shape + (1,) * sample_ndim + self.log_determinant.shape[-1:]
        )
        difference = values[..., None, :] - means
        quadratic = jnp.real(
            oe.contract(
                "...ki,...kij,...kj->...k", jnp.conj(difference), precision, difference
            )
        )
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            return -(quadratic + self.in_size * jnp.log(jnp.pi) + log_det)
        return -0.5 * (quadratic + self.in_size * jnp.log(2.0 * jnp.pi) + log_det)

    def log_prob(self, x: Any, /) -> Array:
        component = self.component_log_prob(x)
        sample_ndim = component.ndim - len(self.case_shape) - 1
        mixing = self.mixing_weights.reshape(
            self.case_shape + (1,) * sample_ndim + (self.out_size,)
        )
        return jsp.special.logsumexp(
            component + jnp.log(jnp.maximum(mixing, jnp.finfo(mixing.dtype).tiny)),
            axis=-1,
        )

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        component = self.component_log_prob(x)
        sample_ndim = component.ndim - len(self.case_shape) - 1
        mixing = self.mixing_weights.reshape(
            self.case_shape + (1,) * sample_ndim + (self.out_size,)
        )
        logits = component + jnp.log(jnp.maximum(mixing, jnp.finfo(mixing.dtype).tiny))
        return jax.nn.softmax(logits, axis=-1)

    def predict(self, x: Any, /) -> Array:
        return jax.lax.stop_gradient(jnp.argmax(self(x), axis=-1).astype(jnp.int32))


class BayesianGaussianMixtureModel(AbstractArrayModel):
    mixing_weights: Array
    means: Array
    covariance: Array
    precision: Array
    log_determinant: Array
    concentration: Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    covariance_type: CovarianceType = eqx.field(static=True)

    def __init__(
        self,
        mixing_weights: Array,
        means: Array,
        covariance: Array,
        precision: Array,
        log_determinant: Array,
        concentration: Array,
        /,
        *,
        covariance_type: CovarianceType,
    ):
        self.mixing_weights = jnp.asarray(mixing_weights)
        self.means = jnp.asarray(means)
        self.covariance = jnp.asarray(covariance)
        self.precision = jnp.asarray(precision)
        self.log_determinant = jnp.asarray(log_determinant)
        self.concentration = jnp.asarray(concentration)
        self.in_size = self.means.shape[-1]
        self.out_size = self.means.shape[-2]
        self.case_shape = self.means.shape[:-2]
        self.covariance_type = covariance_type

    def component_log_prob(self, x: Any, /) -> Array:
        values = jnp.asarray(x)
        sample_ndim = _input_sample_ndim(values, self.case_shape, self.in_size)
        means = self.means.reshape(
            self.case_shape + (1,) * sample_ndim + self.means.shape[-2:]
        )
        precision = self.precision.reshape(
            self.case_shape + (1,) * sample_ndim + self.precision.shape[-3:]
        )
        log_det = self.log_determinant.reshape(
            self.case_shape + (1,) * sample_ndim + self.log_determinant.shape[-1:]
        )
        difference = values[..., None, :] - means
        quadratic = jnp.real(
            oe.contract(
                "...ki,...kij,...kj->...k", jnp.conj(difference), precision, difference
            )
        )
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            return -(quadratic + self.in_size * jnp.log(jnp.pi) + log_det)
        return -0.5 * (quadratic + self.in_size * jnp.log(2.0 * jnp.pi) + log_det)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        component = self.component_log_prob(x)
        sample_ndim = component.ndim - len(self.case_shape) - 1
        mixing = self.mixing_weights.reshape(
            self.case_shape + (1,) * sample_ndim + (self.out_size,)
        )
        return jax.nn.softmax(
            component + jnp.log(jnp.maximum(mixing, jnp.finfo(mixing.dtype).tiny)),
            axis=-1,
        )

    def log_prob(self, x: Any, /) -> Array:
        component = self.component_log_prob(x)
        sample_ndim = component.ndim - len(self.case_shape) - 1
        mixing = self.mixing_weights.reshape(
            self.case_shape + (1,) * sample_ndim + (self.out_size,)
        )
        return jsp.special.logsumexp(
            component + jnp.log(jnp.maximum(mixing, jnp.finfo(mixing.dtype).tiny)),
            axis=-1,
        )

    def predict(self, x: Any, /) -> Array:
        return jax.lax.stop_gradient(jnp.argmax(self(x), axis=-1).astype(jnp.int32))


class GaussianMixture(AbstractRecipe):
    component_count: int = eqx.field(static=True)
    covariance_type: CovarianceType = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    tolerance: Array
    regularization: Array
    initialization: MixtureInitialization = eqx.field(static=True)
    empty_policy: EmptyComponentPolicy = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        component_count: int,
        /,
        *,
        covariance_type: CovarianceType = "full",
        max_iterations: int = 64,
        tolerance: float = 1e-4,
        regularization: float = 1e-6,
        initialization: MixtureInitialization = "random",
        empty_policy: EmptyComponentPolicy = "reseed",
        weight_policy: WeightPolicy = "statistical",
    ):
        if component_count <= 0 or max_iterations <= 0:
            raise ValueError("component_count and max_iterations must be positive.")
        if (
            covariance_type not in ("full", "tied", "diagonal", "spherical")
            or initialization not in ("random", "first")
            or empty_policy not in ("retain", "reseed", "error")
        ):
            raise ValueError("unsupported Gaussian mixture policy.")
        self.component_count = int(component_count)
        self.covariance_type = covariance_type
        self.max_iterations = int(max_iterations)
        self.tolerance = _nonnegative_scalar(tolerance, "tolerance")
        self.regularization = _positive_scalar(regularization, "regularization")
        self.initialization = initialization
        self.empty_policy = empty_policy
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        values = _fit_gaussian_mixture(
            batch,
            component_count=self.component_count,
            covariance_type=self.covariance_type,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            regularization=self.regularization,
            initialization=self.initialization,
            empty_policy=self.empty_policy,
            weight_policy=self.weight_policy,
            key=key,
        )
        (
            mixing,
            means,
            covariance,
            precision,
            log_det,
            _,
            objective,
            delta,
            component_mass,
            empty,
            singular,
            converged,
            valid,
            status,
            mass,
        ) = values
        model = GaussianMixtureModel(
            mixing,
            means,
            covariance,
            precision,
            log_det,
            covariance_type=self.covariance_type,
        )
        diagnostics = MixtureDiagnostics(
            valid=valid,
            status=status,
            negative_log_likelihood=objective,
            objective_delta=delta,
            iterations=self.max_iterations,
            effective_samples=mass,
            component_mass=component_mass,
            empty_components_seen=empty,
            singular_components_seen=singular,
            converged=converged,
            method="gaussian-mixture",
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="conditional",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="unrolled",
            nondifferentiable_outputs=("predict",),
            conditions=(
                "fixed initialization indices",
                "fixed active mask",
                "nondegenerate covariance spectrum",
            ),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="gaussian-mixture",
            gradient_contract=contract,
        )


class BayesianGaussianMixture(AbstractRecipe):
    component_count: int = eqx.field(static=True)
    covariance_type: CovarianceType = eqx.field(static=True)
    concentration: Array
    mean_precision: Array
    degrees_of_freedom: Array
    degrees_of_freedom_provided: bool = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    tolerance: Array
    regularization: Array
    initialization: MixtureInitialization = eqx.field(static=True)
    empty_policy: EmptyComponentPolicy = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        component_count: int,
        /,
        *,
        covariance_type: CovarianceType = "full",
        concentration: float = 1.0,
        mean_precision: float = 1.0,
        degrees_of_freedom: float | None = None,
        max_iterations: int = 64,
        tolerance: float = 1e-4,
        regularization: float = 1e-6,
        initialization: MixtureInitialization = "random",
        empty_policy: EmptyComponentPolicy = "retain",
        weight_policy: WeightPolicy = "statistical",
    ):
        if component_count <= 0 or max_iterations <= 0:
            raise ValueError("component_count and max_iterations must be positive.")
        if (
            covariance_type not in ("full", "tied", "diagonal", "spherical")
            or initialization not in ("random", "first")
            or empty_policy not in ("retain", "reseed", "error")
        ):
            raise ValueError("unsupported Bayesian Gaussian mixture policy.")
        self.component_count = int(component_count)
        self.covariance_type = covariance_type
        self.concentration = _positive_scalar(concentration, "concentration")
        self.mean_precision = _positive_scalar(mean_precision, "mean_precision")
        self.degrees_of_freedom_provided = degrees_of_freedom is not None
        self.degrees_of_freedom = (
            _positive_scalar(degrees_of_freedom, "degrees_of_freedom")
            if degrees_of_freedom is not None
            else jnp.asarray(0.0)
        )
        self.max_iterations = int(max_iterations)
        self.tolerance = _nonnegative_scalar(tolerance, "tolerance")
        self.regularization = _positive_scalar(regularization, "regularization")
        self.initialization = initialization
        self.empty_policy = empty_policy
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if self.degrees_of_freedom_provided:
            dof = eqx.error_if(
                self.degrees_of_freedom,
                self.degrees_of_freedom <= batch.feature_count - 1,
                "degrees_of_freedom must exceed feature_count - 1.",
            )
        else:
            dof = jnp.asarray(batch.feature_count + 2, dtype=self.concentration.dtype)
        values = _fit_gaussian_mixture(
            batch,
            component_count=self.component_count,
            covariance_type=self.covariance_type,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            regularization=self.regularization,
            initialization=self.initialization,
            empty_policy=self.empty_policy,
            weight_policy=self.weight_policy,
            key=key,
            concentration=self.concentration,
            mean_precision=self.mean_precision,
            degrees_of_freedom=dof,
        )
        (
            mixing,
            means,
            covariance,
            precision,
            log_det,
            _,
            objective,
            delta,
            component_mass,
            empty,
            singular,
            converged,
            valid,
            status,
            mass,
        ) = values
        posterior_concentration = component_mass + self.concentration
        model = BayesianGaussianMixtureModel(
            mixing,
            means,
            covariance,
            precision,
            log_det,
            posterior_concentration,
            covariance_type=self.covariance_type,
        )
        diagnostics = MixtureDiagnostics(
            valid=valid,
            status=status,
            negative_log_likelihood=objective,
            objective_delta=delta,
            iterations=self.max_iterations,
            effective_samples=mass,
            component_mass=component_mass,
            empty_components_seen=empty,
            singular_components_seen=singular,
            converged=converged,
            method="bayesian-gaussian-mixture",
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="conditional",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="unrolled",
            nondifferentiable_outputs=("predict",),
            conditions=(
                "fixed initialization indices",
                "fixed active mask",
                "positive variational prior",
            ),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="bayesian-gaussian-mixture",
            gradient_contract=contract,
        )


__all__ = [
    "BayesianGaussianMixture",
    "BayesianGaussianMixtureModel",
    "CovarianceType",
    "EmptyComponentPolicy",
    "GaussianMixture",
    "GaussianMixtureModel",
    "MixtureDiagnostics",
    "MixtureInitialization",
]
