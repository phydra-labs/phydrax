#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Literal

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from .._probability import AbstractProbabilityLaw


class AbstractDistribution(AbstractProbabilityLaw):
    """Minimal scalar distribution protocol for native uncertain inputs."""

    @property
    def event_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def density_measure_kind(self) -> Literal["lebesgue"]:
        return "lebesgue"

    @abstractmethod
    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        raise NotImplementedError

    @abstractmethod
    def icdf(self, value: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, value: ArrayLike, /) -> Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def mean(self) -> Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def variance(self) -> Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def support(self) -> tuple[Array, Array] | None:
        raise NotImplementedError

    @abstractmethod
    def contains(self, value: ArrayLike, /) -> Array:
        raise NotImplementedError

    def equivalent(self, other: object, /) -> bool:
        return type(self) is type(other) and bool(
            jnp.all(jnp.asarray(jax_tree_equal(self, other), dtype=bool))
        )


class Uniform(AbstractDistribution):
    low: Array
    high: Array

    def __init__(self, low: ArrayLike, high: ArrayLike):
        low_array = jnp.asarray(low, dtype=float).reshape(())
        high_array = jnp.asarray(high, dtype=float).reshape(())
        if not bool(jnp.isfinite(low_array)) or not bool(jnp.isfinite(high_array)):
            raise ValueError("Uniform bounds must be finite.")
        if not bool(low_array < high_array):
            raise ValueError("Uniform low must be less than high.")
        self.low = low_array
        self.high = high_array

    @property
    def density_measure_kind(self) -> Literal["lebesgue"]:
        return "lebesgue"

    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        return jr.uniform(
            key,
            shape=tuple(sample_shape),
            minval=self.low,
            maxval=self.high,
            dtype=self.low.dtype,
        )

    def icdf(self, value: ArrayLike, /) -> Array:
        value_array = jnp.asarray(value, dtype=float)
        return self.low + value_array * (self.high - self.low)

    def log_prob(self, value: ArrayLike, /) -> Array:
        value_array = jnp.asarray(value, dtype=float)
        density = -jnp.log(self.high - self.low)
        return jnp.where(self.contains(value_array), density, -jnp.inf)

    @property
    def mean(self) -> Array:
        return (self.low + self.high) / 2.0

    @property
    def variance(self) -> Array:
        return (self.high - self.low) ** 2 / 12.0

    @property
    def support(self) -> tuple[Array, Array]:
        return self.low, self.high

    def contains(self, value: ArrayLike, /) -> Array:
        value_array = jnp.asarray(value)
        return (value_array >= self.low) & (value_array <= self.high)

    @property
    def reference_measure(self) -> Literal["uniform"]:
        return "uniform"

    def to_reference(self, value: ArrayLike, /) -> Array:
        value_array = jnp.asarray(value, dtype=float)
        return 2.0 * (value_array - self.low) / (self.high - self.low) - 1.0

    def from_reference(self, value: ArrayLike, /) -> Array:
        reference = jnp.asarray(value, dtype=float)
        return self.low + 0.5 * (reference + 1.0) * (self.high - self.low)


class Normal(AbstractDistribution):
    location: Array
    scale: Array

    def __init__(self, location: ArrayLike, scale: ArrayLike):
        location_array = jnp.asarray(location, dtype=float).reshape(())
        scale_array = jnp.asarray(scale, dtype=float).reshape(())
        if not bool(jnp.isfinite(location_array)):
            raise ValueError("Normal location must be finite.")
        if not bool(jnp.isfinite(scale_array)) or not bool(scale_array > 0.0):
            raise ValueError("Normal scale must be finite and positive.")
        self.location = location_array
        self.scale = scale_array

    @property
    def density_measure_kind(self) -> Literal["lebesgue"]:
        return "lebesgue"

    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        return self.location + self.scale * jr.normal(
            key, shape=tuple(sample_shape), dtype=self.location.dtype
        )

    def icdf(self, value: ArrayLike, /) -> Array:
        probability = _open_unit_interval(value)
        return self.location + self.scale * jnp.sqrt(2.0) * jsp.special.erfinv(
            2.0 * probability - 1.0
        )

    def log_prob(self, value: ArrayLike, /) -> Array:
        standardized = (jnp.asarray(value, dtype=float) - self.location) / self.scale
        return -0.5 * standardized**2 - jnp.log(self.scale) - 0.5 * jnp.log(2.0 * jnp.pi)

    @property
    def mean(self) -> Array:
        return self.location

    @property
    def variance(self) -> Array:
        return self.scale**2

    @property
    def support(self) -> None:
        return None

    def contains(self, value: ArrayLike, /) -> Array:
        return jnp.isfinite(jnp.asarray(value))

    @property
    def reference_measure(self) -> Literal["standard-normal"]:
        return "standard-normal"

    def to_reference(self, value: ArrayLike, /) -> Array:
        return (jnp.asarray(value, dtype=float) - self.location) / self.scale

    def from_reference(self, value: ArrayLike, /) -> Array:
        return self.location + self.scale * jnp.asarray(value, dtype=float)


class LogNormal(AbstractDistribution):
    location: Array
    scale: Array

    def __init__(self, location: ArrayLike, scale: ArrayLike):
        location_array = jnp.asarray(location, dtype=float).reshape(())
        scale_array = jnp.asarray(scale, dtype=float).reshape(())
        if not bool(jnp.isfinite(location_array)):
            raise ValueError("LogNormal location must be finite.")
        if not bool(jnp.isfinite(scale_array)) or not bool(scale_array > 0.0):
            raise ValueError("LogNormal scale must be finite and positive.")
        self.location = location_array
        self.scale = scale_array

    @property
    def density_measure_kind(self) -> Literal["lebesgue"]:
        return "lebesgue"

    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        normal = self.location + self.scale * jr.normal(
            key, shape=tuple(sample_shape), dtype=self.location.dtype
        )
        return jnp.exp(normal)

    def icdf(self, value: ArrayLike, /) -> Array:
        probability = _open_unit_interval(value)
        normal = self.location + self.scale * jnp.sqrt(2.0) * jsp.special.erfinv(
            2.0 * probability - 1.0
        )
        return jnp.exp(normal)

    def log_prob(self, value: ArrayLike, /) -> Array:
        value_array = jnp.asarray(value, dtype=float)
        positive = value_array > 0.0
        log_value = jnp.log(jnp.where(positive, value_array, 1.0))
        standardized = (log_value - self.location) / self.scale
        density = (
            -0.5 * standardized**2
            - jnp.log(self.scale)
            - log_value
            - 0.5 * jnp.log(2.0 * jnp.pi)
        )
        return jnp.where(positive, density, -jnp.inf)

    @property
    def mean(self) -> Array:
        return jnp.exp(self.location + 0.5 * self.scale**2)

    @property
    def variance(self) -> Array:
        variance = (jnp.exp(self.scale**2) - 1.0) * jnp.exp(
            2.0 * self.location + self.scale**2
        )
        return variance

    @property
    def support(self) -> None:
        return None

    def contains(self, value: ArrayLike, /) -> Array:
        value_array = jnp.asarray(value)
        return jnp.isfinite(value_array) & (value_array > 0.0)

    @property
    def reference_measure(self) -> Literal["standard-normal"]:
        return "standard-normal"

    def to_reference(self, value: ArrayLike, /) -> Array:
        value_array = jnp.asarray(value, dtype=float)
        return (jnp.log(value_array) - self.location) / self.scale

    def from_reference(self, value: ArrayLike, /) -> Array:
        reference = jnp.asarray(value, dtype=float)
        return jnp.exp(self.location + self.scale * reference)


class EmpiricalDistribution(AbstractDistribution):
    values: Array
    probabilities: Array

    def __init__(
        self,
        values: ArrayLike,
        probabilities: ArrayLike | None = None,
    ):
        values_array = jnp.asarray(values, dtype=float)
        if values_array.ndim != 1 or int(values_array.shape[0]) <= 0:
            raise ValueError("Empirical values must be a non-empty 1D array.")
        if bool(jnp.any(~jnp.isfinite(values_array))):
            raise ValueError("Empirical values must be finite.")
        if probabilities is None:
            probability_array = jnp.full(
                values_array.shape, 1.0 / float(values_array.shape[0]), dtype=float
            )
        else:
            probability_array = jnp.asarray(probabilities, dtype=float)
            if probability_array.shape != values_array.shape:
                raise ValueError("Empirical probabilities must match values shape.")
            if bool(jnp.any(~jnp.isfinite(probability_array))) or bool(
                jnp.any(probability_array < 0.0)
            ):
                raise ValueError(
                    "Empirical probabilities must be finite and non-negative."
                )
            total = jnp.sum(probability_array)
            if not bool(total > 0.0):
                raise ValueError("Empirical probabilities must have positive total mass.")
            probability_array = probability_array / total
        order = jnp.argsort(values_array)
        self.values = values_array[order]
        self.probabilities = probability_array[order]

    @property
    def density_measure_kind(self) -> Literal["counting"]:
        return "counting"

    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        indices = jr.choice(
            key,
            int(self.values.shape[0]),
            shape=tuple(sample_shape),
            p=self.probabilities,
        )
        return self.values[indices]

    def icdf(self, value: ArrayLike, /) -> Array:
        probability = jnp.clip(jnp.asarray(value, dtype=float), 0.0, 1.0)
        cumulative = jnp.cumsum(self.probabilities)
        indices = jnp.searchsorted(cumulative, probability, side="left")
        return self.values[jnp.minimum(indices, int(self.values.shape[0]) - 1)]

    def log_prob(self, value: ArrayLike, /) -> Array:
        value_array = jnp.asarray(value, dtype=float)
        matches = value_array[..., None] == self.values
        mass = jnp.sum(jnp.where(matches, self.probabilities, 0.0), axis=-1)
        return jnp.where(mass > 0.0, jnp.log(mass), -jnp.inf)

    @property
    def mean(self) -> Array:
        return jnp.sum(self.probabilities * self.values)

    @property
    def variance(self) -> Array:
        return jnp.sum(self.probabilities * (self.values - self.mean) ** 2)

    @property
    def support(self) -> tuple[Array, Array]:
        return self.values[0], self.values[-1]

    def contains(self, value: ArrayLike, /) -> Array:
        return jnp.any(jnp.asarray(value)[..., None] == self.values, axis=-1)


def _open_unit_interval(value: ArrayLike) -> Array:
    probability = jnp.asarray(value, dtype=float)
    epsilon = jnp.finfo(probability.dtype).eps
    return jnp.clip(probability, epsilon, 1.0 - epsilon)


def jax_tree_equal(left: Any, right: Any) -> Array:
    """Array-safe equality for small immutable distribution PyTrees."""
    if type(left) is not type(right):
        return jnp.asarray(False)
    left_leaves = __import__("jax").tree_util.tree_leaves(left)
    right_leaves = __import__("jax").tree_util.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        return jnp.asarray(False)
    equal = jnp.asarray(True)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        equal = equal & jnp.array_equal(jnp.asarray(left_leaf), jnp.asarray(right_leaf))
    return equal


__all__ = [
    "AbstractDistribution",
    "EmpiricalDistribution",
    "LogNormal",
    "Normal",
    "Uniform",
]
