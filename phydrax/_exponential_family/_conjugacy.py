#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._categorical import CategoricalFamily
from ._contracts import ExponentialFamilyLaw, NaturalCoordinates
from ._dirichlet import DirichletFamily
from ._gamma import GammaFamily


SampleAxes: TypeAlias = int | tuple[int, ...] | None


def _normalize_sample_axes(ndim: int, sample_axes: SampleAxes, /) -> tuple[int, ...]:
    if sample_axes is None:
        return tuple(range(ndim))
    raw_axes = (sample_axes,) if isinstance(sample_axes, int) else tuple(sample_axes)
    axes = tuple(axis + ndim if axis < 0 else axis for axis in raw_axes)
    if any(axis < 0 or axis >= ndim for axis in axes):
        raise ValueError(f"sample_axes {raw_axes} are invalid for rank {ndim}.")
    if len(set(axes)) != len(axes):
        raise ValueError("sample_axes must not contain duplicates.")
    return tuple(sorted(axes))


class GammaPoissonStatistics(StrictModule):
    """Mergeable sufficient statistics for conditionally Poisson observations."""

    total_count: Array
    total_exposure: Array
    log_base_measure: Array
    num_observations: Array
    valid: Array

    def __init__(
        self,
        total_count: ArrayLike,
        total_exposure: ArrayLike,
        log_base_measure: ArrayLike,
        num_observations: ArrayLike,
        valid: ArrayLike,
    ):
        arrays = tuple(
            jnp.asarray(value)
            for value in (
                total_count,
                total_exposure,
                log_base_measure,
                num_observations,
            )
        )
        if any(jnp.issubdtype(value.dtype, jnp.complexfloating) for value in arrays):
            raise TypeError("Gamma-Poisson statistics must be real-valued.")
        count, exposure, base_measure, observations = jnp.broadcast_arrays(*arrays)
        declared_valid = jnp.broadcast_to(jnp.asarray(valid, dtype=bool), count.shape)
        canonical_valid = (
            declared_valid
            & jnp.isfinite(count)
            & (count >= 0.0)
            & (count == jnp.floor(count))
            & jnp.isfinite(exposure)
            & (exposure >= 0.0)
            & jnp.isfinite(base_measure)
            & jnp.isfinite(observations)
            & (observations >= 0.0)
            & (observations == jnp.floor(observations))
            & jnp.where(
                observations > 0.0,
                exposure > 0.0,
                (count == 0.0) & (exposure == 0.0) & (base_measure == 0.0),
            )
        )
        self.total_count = count
        self.total_exposure = exposure
        self.log_base_measure = base_measure
        self.num_observations = observations
        self.valid = canonical_valid

    def merge(self, other: "GammaPoissonStatistics", /) -> "GammaPoissonStatistics":
        if not isinstance(other, GammaPoissonStatistics):
            raise TypeError("other must be GammaPoissonStatistics.")
        return GammaPoissonStatistics(
            self.total_count + other.total_count,
            self.total_exposure + other.total_exposure,
            self.log_base_measure + other.log_base_measure,
            self.num_observations + other.num_observations,
            self.valid & other.valid,
        )


class GammaPoissonUpdate(StrictModule):
    """Audited Gamma posterior, evidence, and predictive Poisson law."""

    family: GammaFamily
    prior_natural: NaturalCoordinates
    posterior_natural: NaturalCoordinates
    statistics: GammaPoissonStatistics
    log_evidence: Array
    valid: Array

    def __init__(
        self,
        family: GammaFamily,
        prior_natural: NaturalCoordinates,
        posterior_natural: NaturalCoordinates,
        statistics: GammaPoissonStatistics,
        log_evidence: Array,
        valid: Array,
    ):
        self.family = family
        self.prior_natural = prior_natural
        self.posterior_natural = posterior_natural
        self.statistics = statistics
        self.log_evidence = log_evidence
        self.valid = valid

    @property
    def posterior_law(self) -> ExponentialFamilyLaw:
        return self.family.law(self.posterior_natural)

    @property
    def posterior_shape_rate(self) -> tuple[Array, Array]:
        return self.family.shape_rate_from_natural(self.posterior_natural)

    def predictive_log_prob(
        self, count: ArrayLike, /, *, exposure: ArrayLike = 1.0
    ) -> Array:
        shape, rate = self.posterior_shape_rate
        count_array, exposure_array, shape, rate = jnp.broadcast_arrays(
            jnp.asarray(count), jnp.asarray(exposure), shape, rate
        )
        dtype = jnp.result_type(count_array, exposure_array, shape, rate, 0.0)
        count_array = count_array.astype(dtype)
        exposure_array = exposure_array.astype(dtype)
        observation_valid = (
            jnp.isfinite(count_array)
            & (count_array >= 0.0)
            & (count_array == jnp.floor(count_array))
            & jnp.isfinite(exposure_array)
            & (exposure_array > 0.0)
        )
        safe_count = jnp.where(observation_valid, count_array, 0.0)
        safe_exposure = jnp.where(observation_valid, exposure_array, 1.0)
        value = (
            jsp.special.gammaln(shape + safe_count)
            - jsp.special.gammaln(shape)
            - jsp.special.gammaln(safe_count + 1.0)
            + shape * (jnp.log(rate) - jnp.log(rate + safe_exposure))
            + safe_count * (jnp.log(safe_exposure) - jnp.log(rate + safe_exposure))
        )
        return jnp.where(
            self.valid,
            jnp.where(observation_valid, value, -jnp.inf),
            jnp.nan,
        )

    def sample_predictive(
        self,
        key,
        sample_shape: tuple[int, ...] = (),
        *,
        exposure: ArrayLike = 1.0,
    ) -> Array:
        shape, rate = self.posterior_shape_rate
        exposure_array, shape, rate = jnp.broadcast_arrays(
            jnp.asarray(exposure), shape, rate
        )
        dtype = jnp.result_type(exposure_array, shape, rate, 0.0)
        exposure_array = exposure_array.astype(dtype)
        checked_exposure = eqx.error_if(
            exposure_array,
            jnp.any(~jnp.isfinite(exposure_array) | (exposure_array <= 0.0)),
            "Predictive exposure must be finite and strictly positive.",
        )
        checked_shape = eqx.error_if(
            shape,
            jnp.any(~self.valid),
            "Cannot sample an invalid Gamma-Poisson update.",
        )
        rate_key, count_key = jr.split(key)
        latent_rate = (
            jr.gamma(
                rate_key,
                checked_shape,
                shape=tuple(sample_shape) + checked_shape.shape,
                dtype=dtype,
            )
            / rate
        )
        return jr.poisson(count_key, checked_exposure * latent_rate)


class GammaPoissonConjugacy(StrictModule):
    """Explicit Gamma-prior/Poisson-likelihood conjugate pair."""

    family: GammaFamily
    prior_natural: NaturalCoordinates

    def __init__(
        self,
        shape: ArrayLike,
        rate: ArrayLike,
        *,
        family: GammaFamily | None = None,
    ):
        selected_family = GammaFamily() if family is None else family
        if not isinstance(selected_family, GammaFamily):
            raise TypeError("family must be a GammaFamily.")
        prior_natural = selected_family.natural_from_shape_rate(shape, rate)
        if not bool(jnp.all(selected_family.natural_domain(prior_natural).valid)):
            raise ValueError("Gamma prior shape and rate must be finite and positive.")
        self.family = selected_family
        self.prior_natural = prior_natural

    @property
    def prior_law(self) -> ExponentialFamilyLaw:
        return self.family.law(self.prior_natural)

    def summarize(
        self,
        counts: ArrayLike,
        /,
        *,
        exposure: ArrayLike = 1.0,
        sample_axes: SampleAxes = None,
    ) -> GammaPoissonStatistics:
        count_array, exposure_array = jnp.broadcast_arrays(
            jnp.asarray(counts), jnp.asarray(exposure)
        )
        dtype = jnp.result_type(count_array, exposure_array, 0.0)
        count_array = count_array.astype(dtype)
        exposure_array = exposure_array.astype(dtype)
        observation_valid = (
            jnp.isfinite(count_array)
            & (count_array >= 0.0)
            & (count_array == jnp.floor(count_array))
            & jnp.isfinite(exposure_array)
            & (exposure_array > 0.0)
        )
        safe_count = jnp.where(observation_valid, count_array, 0.0)
        safe_exposure = jnp.where(observation_valid, exposure_array, 1.0)
        axes = _normalize_sample_axes(count_array.ndim, sample_axes)
        result_shape = tuple(
            size for axis, size in enumerate(count_array.shape) if axis not in axes
        )
        count = jnp.sum(safe_count, axis=axes)
        total_exposure = jnp.sum(safe_exposure, axis=axes)
        base_measure = jnp.sum(
            jnp.where(safe_count == 0.0, 0.0, safe_count * jnp.log(safe_exposure))
            - jsp.special.gammaln(safe_count + 1.0),
            axis=axes,
        )
        valid = jnp.all(observation_valid, axis=axes)
        observation_count = jnp.full(
            result_shape,
            prod(count_array.shape[axis] for axis in axes),
            dtype=jnp.int32,
        )
        return GammaPoissonStatistics(
            count, total_exposure, base_measure, observation_count, valid
        )

    def update(
        self,
        counts: ArrayLike,
        /,
        *,
        exposure: ArrayLike = 1.0,
        sample_axes: SampleAxes = None,
    ) -> GammaPoissonUpdate:
        return self.update_statistics(
            self.summarize(counts, exposure=exposure, sample_axes=sample_axes)
        )

    def update_statistics(
        self, statistics: GammaPoissonStatistics, /
    ) -> GammaPoissonUpdate:
        if not isinstance(statistics, GammaPoissonStatistics):
            raise TypeError("statistics must be GammaPoissonStatistics.")
        statistic_arrays = (
            statistics.total_count,
            statistics.total_exposure,
            statistics.log_base_measure,
            statistics.num_observations,
        )
        if any(
            jnp.issubdtype(jnp.asarray(value).dtype, jnp.complexfloating)
            for value in statistic_arrays
        ):
            raise TypeError("Gamma-Poisson statistics must be real-valued.")
        prior_shape, prior_rate = self.family.shape_rate_from_natural(self.prior_natural)
        (
            prior_shape,
            prior_rate,
            total_count,
            total_exposure,
            log_base_measure,
            num_observations,
        ) = jnp.broadcast_arrays(
            prior_shape,
            prior_rate,
            statistics.total_count,
            statistics.total_exposure,
            statistics.log_base_measure,
            statistics.num_observations,
        )
        declared_valid = jnp.broadcast_to(
            jnp.asarray(statistics.valid, dtype=bool), prior_shape.shape
        )
        statistics_valid = (
            declared_valid
            & jnp.isfinite(total_count)
            & (total_count >= 0.0)
            & (total_count == jnp.floor(total_count))
            & jnp.isfinite(total_exposure)
            & (total_exposure >= 0.0)
            & jnp.isfinite(log_base_measure)
            & jnp.isfinite(num_observations)
            & (num_observations >= 0.0)
            & (num_observations == jnp.floor(num_observations))
            & jnp.where(
                num_observations > 0.0,
                total_exposure > 0.0,
                (total_count == 0.0)
                & (total_exposure == 0.0)
                & (log_base_measure == 0.0),
            )
        )
        posterior_shape = prior_shape + total_count
        posterior_rate = prior_rate + total_exposure
        posterior_natural = self.family.natural_from_shape_rate(
            posterior_shape, posterior_rate
        )
        posterior_natural = self.family.natural(
            jnp.where(statistics_valid[..., None], posterior_natural.values, jnp.nan)
        )
        log_evidence = (
            jsp.special.gammaln(posterior_shape)
            - jsp.special.gammaln(prior_shape)
            + prior_shape * jnp.log(prior_rate)
            - posterior_shape * jnp.log(posterior_rate)
            + log_base_measure
        )
        return GammaPoissonUpdate(
            self.family,
            self.prior_natural,
            posterior_natural,
            statistics,
            jnp.where(statistics_valid, log_evidence, jnp.nan),
            statistics_valid,
        )


class DirichletCategoricalStatistics(StrictModule):
    """Mergeable category counts for an ordered categorical sample."""

    category_counts: Array
    num_observations: Array
    valid: Array

    def __init__(
        self,
        category_counts: ArrayLike,
        num_observations: ArrayLike,
        valid: ArrayLike,
    ):
        counts = jnp.asarray(category_counts)
        observations = jnp.asarray(num_observations)
        if counts.ndim == 0 or int(counts.shape[-1]) < 2:
            raise ValueError(
                "Dirichlet-categorical counts require at least two categories."
            )
        if jnp.issubdtype(counts.dtype, jnp.complexfloating) or jnp.issubdtype(
            observations.dtype, jnp.complexfloating
        ):
            raise TypeError("Dirichlet-categorical statistics must be real-valued.")
        result_shape = jnp.broadcast_shapes(counts.shape[:-1], observations.shape)
        counts = jnp.broadcast_to(counts, result_shape + (counts.shape[-1],))
        observations = jnp.broadcast_to(observations, result_shape)
        declared_valid = jnp.broadcast_to(jnp.asarray(valid, dtype=bool), result_shape)
        canonical_valid = (
            declared_valid
            & jnp.all(jnp.isfinite(counts), axis=-1)
            & jnp.all(counts >= 0.0, axis=-1)
            & jnp.all(counts == jnp.floor(counts), axis=-1)
            & jnp.isfinite(observations)
            & (observations >= 0.0)
            & (observations == jnp.floor(observations))
            & (jnp.sum(counts, axis=-1) == observations)
        )
        self.category_counts = counts
        self.num_observations = observations
        self.valid = canonical_valid

    def merge(
        self, other: "DirichletCategoricalStatistics", /
    ) -> "DirichletCategoricalStatistics":
        if not isinstance(other, DirichletCategoricalStatistics):
            raise TypeError("other must be DirichletCategoricalStatistics.")
        if self.category_counts.shape[-1] != other.category_counts.shape[-1]:
            raise ValueError("Categorical statistics have different category counts.")
        return DirichletCategoricalStatistics(
            self.category_counts + other.category_counts,
            self.num_observations + other.num_observations,
            self.valid & other.valid,
        )


class DirichletCategoricalUpdate(StrictModule):
    """Audited Dirichlet posterior, evidence, and categorical predictive law."""

    dirichlet_family: DirichletFamily
    categorical_family: CategoricalFamily
    prior_natural: NaturalCoordinates
    posterior_natural: NaturalCoordinates
    statistics: DirichletCategoricalStatistics
    log_evidence: Array
    valid: Array

    def __init__(
        self,
        dirichlet_family: DirichletFamily,
        categorical_family: CategoricalFamily,
        prior_natural: NaturalCoordinates,
        posterior_natural: NaturalCoordinates,
        statistics: DirichletCategoricalStatistics,
        log_evidence: Array,
        valid: Array,
    ):
        self.dirichlet_family = dirichlet_family
        self.categorical_family = categorical_family
        self.prior_natural = prior_natural
        self.posterior_natural = posterior_natural
        self.statistics = statistics
        self.log_evidence = log_evidence
        self.valid = valid

    @property
    def posterior_law(self) -> ExponentialFamilyLaw:
        return self.dirichlet_family.law(self.posterior_natural)

    @property
    def posterior_concentration(self) -> Array:
        return self.dirichlet_family.concentration_from_natural(self.posterior_natural)

    @property
    def predictive_probabilities(self) -> Array:
        concentration = self.posterior_concentration
        probabilities = concentration / jnp.sum(concentration, axis=-1, keepdims=True)
        return jnp.where(self.valid[..., None], probabilities, jnp.nan)

    @property
    def predictive_law(self) -> ExponentialFamilyLaw:
        natural = self.categorical_family.natural_from_logits(
            jnp.log(self.posterior_concentration)
        )
        return self.categorical_family.law(natural)

    def predictive_log_prob(self, label: ArrayLike, /) -> Array:
        probabilities = self.predictive_probabilities
        raw_label = jnp.asarray(label)
        label_array, batch_value = jnp.broadcast_arrays(raw_label, probabilities[..., 0])
        label_valid = (
            jnp.isfinite(label_array)
            & (label_array >= 0)
            & (label_array < self.dirichlet_family.num_categories)
            & (label_array == jnp.floor(label_array))
        )
        safe_label = jnp.where(label_valid, label_array, 0).astype(jnp.int32)
        broadcast_probabilities = jnp.broadcast_to(
            probabilities, batch_value.shape + (self.dirichlet_family.num_categories,)
        )
        value = jnp.log(
            jnp.take_along_axis(broadcast_probabilities, safe_label[..., None], axis=-1)[
                ..., 0
            ]
        )
        return jnp.where(
            self.valid,
            jnp.where(label_valid, value, -jnp.inf),
            jnp.nan,
        )

    def sample_predictive(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        return self.predictive_law.sample(key, tuple(sample_shape))


class DirichletCategoricalConjugacy(StrictModule):
    """Explicit Dirichlet-prior/categorical-likelihood conjugate pair."""

    dirichlet_family: DirichletFamily
    categorical_family: CategoricalFamily
    prior_natural: NaturalCoordinates

    def __init__(
        self,
        concentration: ArrayLike,
        *,
        family: DirichletFamily | None = None,
    ):
        concentration_array = jnp.asarray(concentration)
        if concentration_array.ndim == 0:
            raise ValueError("Dirichlet concentration must have a category axis.")
        selected_family = (
            DirichletFamily(int(concentration_array.shape[-1]))
            if family is None
            else family
        )
        if not isinstance(selected_family, DirichletFamily):
            raise TypeError("family must be a DirichletFamily.")
        prior_natural = selected_family.natural_from_concentration(concentration_array)
        if not bool(jnp.all(selected_family.natural_domain(prior_natural).valid)):
            raise ValueError("Dirichlet concentrations must be finite and positive.")
        self.dirichlet_family = selected_family
        self.categorical_family = CategoricalFamily(selected_family.num_categories)
        self.prior_natural = prior_natural

    @property
    def prior_law(self) -> ExponentialFamilyLaw:
        return self.dirichlet_family.law(self.prior_natural)

    def summarize(
        self,
        labels: ArrayLike,
        /,
        *,
        sample_axes: SampleAxes = None,
    ) -> DirichletCategoricalStatistics:
        raw_labels = jnp.asarray(labels)
        dtype = jnp.result_type(raw_labels, 0.0)
        label_values = raw_labels.astype(dtype)
        valid_labels = (
            jnp.isfinite(label_values)
            & (label_values >= 0.0)
            & (label_values < self.dirichlet_family.num_categories)
            & (label_values == jnp.floor(label_values))
        )
        safe_labels = jnp.where(valid_labels, label_values, 0.0).astype(jnp.int32)
        encoded = jax.nn.one_hot(
            safe_labels,
            self.dirichlet_family.num_categories,
            dtype=dtype,
        )
        axes = _normalize_sample_axes(label_values.ndim, sample_axes)
        result_shape = tuple(
            size for axis, size in enumerate(label_values.shape) if axis not in axes
        )
        counts = jnp.sum(encoded, axis=axes)
        valid = jnp.all(valid_labels, axis=axes)
        observation_count = jnp.full(
            result_shape,
            prod(label_values.shape[axis] for axis in axes),
            dtype=jnp.int32,
        )
        return DirichletCategoricalStatistics(counts, observation_count, valid)

    def update(
        self,
        labels: ArrayLike,
        /,
        *,
        sample_axes: SampleAxes = None,
    ) -> DirichletCategoricalUpdate:
        return self.update_statistics(self.summarize(labels, sample_axes=sample_axes))

    def update_statistics(
        self, statistics: DirichletCategoricalStatistics, /
    ) -> DirichletCategoricalUpdate:
        if not isinstance(statistics, DirichletCategoricalStatistics):
            raise TypeError("statistics must be DirichletCategoricalStatistics.")
        if (
            statistics.category_counts.ndim == 0
            or statistics.category_counts.shape[-1]
            != self.dirichlet_family.num_categories
        ):
            raise ValueError("Categorical statistics have an incompatible category axis.")
        if jnp.issubdtype(
            statistics.category_counts.dtype, jnp.complexfloating
        ) or jnp.issubdtype(
            jnp.asarray(statistics.num_observations).dtype, jnp.complexfloating
        ):
            raise TypeError("Dirichlet-categorical statistics must be real-valued.")
        prior_concentration = self.dirichlet_family.concentration_from_natural(
            self.prior_natural
        )
        prior_concentration, counts = jnp.broadcast_arrays(
            prior_concentration, statistics.category_counts
        )
        result_shape = prior_concentration.shape[:-1]
        num_observations = jnp.broadcast_to(
            jnp.asarray(statistics.num_observations), result_shape
        )
        declared_valid = jnp.broadcast_to(
            jnp.asarray(statistics.valid, dtype=bool), result_shape
        )
        valid = (
            declared_valid
            & jnp.all(jnp.isfinite(counts), axis=-1)
            & jnp.all(counts >= 0.0, axis=-1)
            & jnp.all(counts == jnp.floor(counts), axis=-1)
            & jnp.isfinite(num_observations)
            & (num_observations >= 0.0)
            & (num_observations == jnp.floor(num_observations))
            & (jnp.sum(counts, axis=-1) == num_observations)
        )
        posterior_concentration = prior_concentration + counts
        posterior_natural = self.dirichlet_family.natural_from_concentration(
            posterior_concentration
        )
        posterior_natural = self.dirichlet_family.natural(
            jnp.where(valid[..., None], posterior_natural.values, jnp.nan)
        )
        log_beta_prior = jnp.sum(
            jsp.special.gammaln(prior_concentration), axis=-1
        ) - jsp.special.gammaln(jnp.sum(prior_concentration, axis=-1))
        log_beta_posterior = jnp.sum(
            jsp.special.gammaln(posterior_concentration), axis=-1
        ) - jsp.special.gammaln(jnp.sum(posterior_concentration, axis=-1))
        return DirichletCategoricalUpdate(
            self.dirichlet_family,
            self.categorical_family,
            self.prior_natural,
            posterior_natural,
            statistics,
            jnp.where(valid, log_beta_posterior - log_beta_prior, jnp.nan),
            valid,
        )


__all__ = [
    "DirichletCategoricalConjugacy",
    "DirichletCategoricalStatistics",
    "DirichletCategoricalUpdate",
    "GammaPoissonConjugacy",
    "GammaPoissonStatistics",
    "GammaPoissonUpdate",
]
