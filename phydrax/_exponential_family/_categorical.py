#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike

from ._contracts import (
    _AbstractAnalyticExponentialFamily,
    _mean_domain_result,
    _natural_domain_result,
    ExponentialFamilyDomainResult,
    ExponentialFamilySignature,
    NaturalCoordinates,
    StatisticBatch,
)


class CategoricalFamily(_AbstractAnalyticExponentialFamily):
    """Categorical laws in last-category-reference log-odds coordinates."""

    num_categories: int = eqx.field(static=True)
    _signature: ExponentialFamilySignature = eqx.field(static=True)

    def __init__(self, num_categories: int):
        categories = int(num_categories)
        if categories < 2:
            raise ValueError("num_categories must be at least two.")
        self.num_categories = categories
        self._signature = ExponentialFamilySignature(
            "categorical",
            categories - 1,
            (),
            "counting",
            f"integer-categories-{categories}",
            f"last-category-reference-log-odds-{categories}",
        )

    @property
    def signature(self) -> ExponentialFamilySignature:
        return self._signature

    def natural_from_logits(self, logits: ArrayLike, /) -> NaturalCoordinates:
        """Convert conventional full logits to the identified natural chart."""
        values = jnp.asarray(logits)
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("Categorical logits must be real-valued.")
        if values.ndim == 0 or int(values.shape[-1]) != self.num_categories:
            raise ValueError(
                "Categorical full logits must end in num_categories="
                f"{self.num_categories}; got {values.shape}."
            )
        values = values.astype(jnp.result_type(values, 0.0))
        return self.natural(values[..., :-1] - values[..., -1, None])

    def log_prob_from_logits(
        self,
        logits: ArrayLike,
        target: ArrayLike,
        /,
    ) -> Array:
        """Return hard-label log probabilities without materializing one-hot targets."""
        values = jnp.asarray(logits)
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("Categorical logits must be real-valued.")
        if values.ndim == 0 or int(values.shape[-1]) != self.num_categories:
            raise ValueError(
                "Categorical full logits must end in num_categories="
                f"{self.num_categories}; got {values.shape}."
            )
        values = values.astype(jnp.result_type(values, 0.0))

        raw_target = jnp.asarray(target)
        if jnp.issubdtype(raw_target.dtype, jnp.complexfloating):
            raise TypeError("Categorical observations must be real-valued labels.")
        if raw_target.shape != values.shape[:-1]:
            raise ValueError(
                "Categorical targets must match the logits batch shape; "
                f"got logits={values.shape} and target={raw_target.shape}."
            )
        observation = raw_target.astype(jnp.result_type(raw_target, 0.0))
        target_valid = (
            jnp.isfinite(observation)
            & (observation >= 0.0)
            & (observation < self.num_categories)
            & (observation == jnp.floor(observation))
        )
        logits_valid = jnp.all(jnp.isfinite(values), axis=-1)
        valid = logits_valid & target_valid
        safe_target = jnp.where(target_valid, observation, 0.0).astype(jnp.int32)
        safe_logits = jnp.where(logits_valid[..., None], values, 0.0)
        selected = jnp.take_along_axis(safe_logits, safe_target[..., None], axis=-1)[
            ..., 0
        ]
        result = selected - jax.nn.logsumexp(safe_logits, axis=-1)
        return jnp.where(valid, result, -jnp.inf)

    def probabilities_from_natural(self, natural: NaturalCoordinates, /) -> Array:
        """Return conventional full category probabilities."""
        domain = self.natural_domain(natural)
        logits = jnp.concatenate(
            (natural.values, jnp.zeros_like(natural.values[..., :1])), axis=-1
        )
        probabilities = jax.nn.softmax(logits, axis=-1)
        return jnp.where(domain.valid[..., None], probabilities, jnp.nan)

    def _natural_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        shape = values.shape[:-1]
        return _natural_domain_result(
            self.signature,
            values,
            interior=jnp.ones(shape, dtype=bool),
            boundary=jnp.zeros(shape, dtype=bool),
        )

    def _mean_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        reference = 1.0 - jnp.sum(values, axis=-1)
        nonnegative = jnp.all(values >= 0.0, axis=-1) & (reference >= 0.0)
        positive = jnp.all(values > 0.0, axis=-1) & (reference > 0.0)
        return _mean_domain_result(
            self.signature,
            values,
            interior=positive,
            boundary=nonnegative & ~positive,
        )

    def _sufficient_statistics(self, value: ArrayLike, /) -> StatisticBatch:
        raw = jnp.asarray(value)
        if jnp.issubdtype(raw.dtype, jnp.complexfloating):
            raise TypeError("Categorical observations must be real-valued labels.")
        observation = raw.astype(jnp.result_type(raw, 0.0))
        valid = (
            jnp.isfinite(observation)
            & (observation >= 0.0)
            & (observation < self.num_categories)
            & (observation == jnp.floor(observation))
        )
        safe = jnp.where(valid, observation, 0.0).astype(jnp.int32)
        one_hot = jax.nn.one_hot(safe, self.num_categories, dtype=observation.dtype)
        return StatisticBatch(one_hot[..., :-1], valid, self.signature)

    def _log_base_density(self, value: ArrayLike, /) -> Array:
        return jnp.zeros_like(jnp.asarray(value, dtype=float))

    def _log_normalizer(self, natural_values: Array, /) -> Array:
        logits = jnp.concatenate(
            (natural_values, jnp.zeros_like(natural_values[..., :1])), axis=-1
        )
        return jax.nn.logsumexp(logits, axis=-1)

    def _mean_values(self, natural_values: Array, /) -> Array:
        logits = jnp.concatenate(
            (natural_values, jnp.zeros_like(natural_values[..., :1])), axis=-1
        )
        return jax.nn.softmax(logits, axis=-1)[..., :-1]

    def _natural_from_mean_values(self, mean_values: Array, /) -> Array:
        reference = 1.0 - jnp.sum(mean_values, axis=-1, keepdims=True)
        safe_values = jnp.where(mean_values > 0.0, mean_values, 1.0)
        safe_reference = jnp.where(reference > 0.0, reference, 1.0)
        return jnp.log(safe_values) - jnp.log(safe_reference)

    def _sample(
        self,
        key,
        natural_values: Array,
        sample_shape: tuple[int, ...],
        /,
    ) -> Array:
        logits = jnp.concatenate(
            (natural_values, jnp.zeros_like(natural_values[..., :1])), axis=-1
        )
        return jr.categorical(
            key,
            logits,
            axis=-1,
            shape=sample_shape + natural_values.shape[:-1],
        )


__all__ = ["CategoricalFamily"]
