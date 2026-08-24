#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from ._classification import (
    categorical_probabilities_from_logits,
    independent_bernoulli_log_prob_from_logits,
    independent_bernoulli_probabilities_from_logits,
    ordinal_class_probabilities_from_location,
    ordinal_log_prob_from_location,
)
from ._exponential_family import AbstractExponentialFamily, CategoricalFamily
from ._strict import StrictModule


class AbstractLikelihood(StrictModule):
    """Elementwise observation likelihood protocol."""

    @abstractmethod
    def log_prob(
        self, location: ArrayLike, target: ArrayLike, /, **parameters: Any
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def sample(self, key, location: ArrayLike, /, **parameters: Any) -> Array:
        raise NotImplementedError

    @abstractmethod
    def align_observations(
        self, location: ArrayLike, target: ArrayLike, /
    ) -> tuple[Array, Array]:
        """Align model outputs and observations under this likelihood's event contract."""
        location_array = jnp.asarray(location)
        target_array = jnp.asarray(target)
        if location_array.shape == target_array.shape:
            return location_array, target_array
        if (
            location_array.ndim == 2
            and target_array.ndim == 1
            and int(location_array.shape[1]) == 1
        ):
            location_array = location_array[:, 0]
        elif (
            target_array.ndim == 2
            and location_array.ndim == 1
            and int(target_array.shape[1]) == 1
        ):
            target_array = target_array[:, 0]
        if location_array.shape != target_array.shape:
            raise ValueError(
                "Likelihood prediction and target shapes are incompatible: "
                f"prediction={location_array.shape}, target={target_array.shape}."
            )
        return location_array, target_array


class _AbstractElementwiseLikelihood(AbstractLikelihood):
    def align_observations(
        self, location: ArrayLike, target: ArrayLike, /
    ) -> tuple[Array, Array]:
        return super().align_observations(location, target)


class ScalarNaturalExponentialFamilyLikelihood(_AbstractElementwiseLikelihood):
    """Elementwise likelihood whose model output is one scalar natural parameter."""

    family: AbstractExponentialFamily

    def __init__(self, family: AbstractExponentialFamily):
        if not isinstance(family, AbstractExponentialFamily):
            raise TypeError("family must implement AbstractExponentialFamily.")
        signature = family.signature
        if signature.dimension != 1 or signature.event_shape:
            raise ValueError(
                "ScalarNaturalExponentialFamilyLikelihood requires a scalar-event "
                "family with one natural coordinate."
            )
        self.family = family

    def _natural(self, location: ArrayLike, /):
        values = jnp.asarray(location)
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("Natural-parameter predictions must be real-valued.")
        values = values.astype(jnp.result_type(values, 0.0))
        return self.family.natural(values[..., None])

    def log_prob(
        self, location: ArrayLike, target: ArrayLike, /, **parameters: Any
    ) -> Array:
        if parameters:
            raise TypeError(
                "ScalarNaturalExponentialFamilyLikelihood received unknown parameters "
                f"{tuple(parameters)!r}."
            )
        return self.family.log_prob(self._natural(location), target)

    def sample(self, key, location: ArrayLike, /, **parameters: Any) -> Array:
        if parameters:
            raise TypeError(
                "ScalarNaturalExponentialFamilyLikelihood received unknown parameters "
                f"{tuple(parameters)!r}."
            )
        return self.family.sample(key, self._natural(location))


class CategoricalExponentialFamilyLikelihood(AbstractLikelihood):
    """Categorical likelihood with an explicit model-output coordinate convention."""

    family: CategoricalFamily
    prediction_coordinates: Literal["natural", "full_logits"] = eqx.field(static=True)

    def __init__(
        self,
        family: CategoricalFamily,
        *,
        prediction_coordinates: Literal["natural", "full_logits"],
    ):
        if not isinstance(family, CategoricalFamily):
            raise TypeError("family must be a CategoricalFamily.")
        if prediction_coordinates not in ("natural", "full_logits"):
            raise ValueError("prediction_coordinates must be 'natural' or 'full_logits'.")
        self.family = family
        self.prediction_coordinates = prediction_coordinates

    @property
    def prediction_dimension(self) -> int:
        return (
            self.family.signature.dimension
            if self.prediction_coordinates == "natural"
            else self.family.num_categories
        )

    def align_observations(
        self, location: ArrayLike, target: ArrayLike, /
    ) -> tuple[Array, Array]:
        location_array = jnp.asarray(location)
        target_array = jnp.asarray(target)
        if location_array.ndim == 0 or int(location_array.shape[-1]) != (
            self.prediction_dimension
        ):
            raise ValueError(
                "Categorical predictions must end in coordinate dimension "
                f"{self.prediction_dimension}; got {location_array.shape}."
            )
        target_shape = location_array.shape[:-1]
        if target_array.shape == target_shape + (1,):
            target_array = target_array[..., 0]
        if target_array.shape != target_shape:
            raise ValueError(
                "Categorical prediction and target shapes are incompatible: "
                f"prediction={location_array.shape}, target={target_array.shape}."
            )
        return location_array, target_array

    def _full_logits(self, location: ArrayLike, /) -> Array:
        values = jnp.asarray(location)
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("Categorical predictions must be real-valued.")
        if values.ndim == 0 or int(values.shape[-1]) != self.prediction_dimension:
            raise ValueError(
                "Categorical predictions must end in coordinate dimension "
                f"{self.prediction_dimension}; got {values.shape}."
            )
        values = values.astype(jnp.result_type(values, 0.0))
        if self.prediction_coordinates == "natural":
            return jnp.concatenate(
                (values, jnp.zeros_like(values[..., :1])),
                axis=-1,
            )
        return values

    def _natural(self, location: ArrayLike, /):
        return self.family.natural_from_logits(self._full_logits(location))

    def class_probabilities(self, location: ArrayLike, /) -> Array:
        """Return conventional probabilities on the complete category simplex."""
        return categorical_probabilities_from_logits(
            self._full_logits(location),
            class_count=self.family.num_categories,
        )

    def log_prob(
        self, location: ArrayLike, target: ArrayLike, /, **parameters: Any
    ) -> Array:
        if parameters:
            raise TypeError(
                "CategoricalExponentialFamilyLikelihood received unknown parameters "
                f"{tuple(parameters)!r}."
            )
        aligned_location, aligned_target = self.align_observations(location, target)
        return self.family.log_prob_from_logits(
            self._full_logits(aligned_location),
            aligned_target,
        )

    def sample(self, key, location: ArrayLike, /, **parameters: Any) -> Array:
        if parameters:
            raise TypeError(
                "CategoricalExponentialFamilyLikelihood received unknown parameters "
                f"{tuple(parameters)!r}."
            )
        return self.family.sample(key, self._natural(location))


class IndependentBernoulliLikelihood(AbstractLikelihood):
    """Vector-event likelihood for conditionally independent binary labels."""

    label_count: int = eqx.field(static=True)

    def __init__(self, label_count: int):
        count = int(label_count)
        if count <= 0:
            raise ValueError("label_count must be positive.")
        self.label_count = count

    def align_observations(
        self, location: ArrayLike, target: ArrayLike, /
    ) -> tuple[Array, Array]:
        location_array = jnp.asarray(location)
        target_array = jnp.asarray(target)
        if (
            location_array.ndim == 0
            or int(location_array.shape[-1]) != self.label_count
            or target_array.shape != location_array.shape
        ):
            raise ValueError(
                "Independent Bernoulli prediction and target shapes must match and "
                f"end in label_count={self.label_count}; got "
                f"prediction={location_array.shape}, target={target_array.shape}."
            )
        return location_array, target_array

    def positive_probabilities(self, location: ArrayLike, /) -> Array:
        values = jnp.asarray(location)
        if values.ndim == 0 or int(values.shape[-1]) != self.label_count:
            raise ValueError(
                f"Multilabel logits must end in label_count={self.label_count}."
            )
        return independent_bernoulli_probabilities_from_logits(values)

    def log_prob(
        self, location: ArrayLike, target: ArrayLike, /, **parameters: Any
    ) -> Array:
        target_mask = parameters.pop("target_mask", None)
        if parameters:
            raise TypeError(
                "IndependentBernoulliLikelihood received unknown parameters "
                f"{tuple(parameters)!r}."
            )
        location_array, target_array = self.align_observations(location, target)
        return independent_bernoulli_log_prob_from_logits(
            location_array,
            target_array,
            target_mask=target_mask,
        )

    def sample(self, key, location: ArrayLike, /, **parameters: Any) -> Array:
        if parameters:
            raise TypeError(
                "IndependentBernoulliLikelihood received unknown parameters "
                f"{tuple(parameters)!r}."
            )
        return jr.bernoulli(key, self.positive_probabilities(location)).astype(jnp.int32)


class OrdinalCumulativeLinkLikelihood(AbstractLikelihood):
    """Ordered-logistic likelihood with fixed strictly increasing thresholds."""

    thresholds: Array

    def __init__(self, thresholds: ArrayLike):
        values = jnp.asarray(thresholds, dtype=float)
        if values.ndim != 1 or int(values.shape[0]) < 2:
            raise ValueError(
                "Ordinal thresholds must contain at least two ordered values."
            )
        if bool(jnp.any(~jnp.isfinite(values))) or bool(
            jnp.any(values[1:] <= values[:-1])
        ):
            raise ValueError("Ordinal thresholds must be finite and strictly increasing.")
        self.thresholds = values

    @property
    def class_count(self) -> int:
        return int(self.thresholds.shape[0]) + 1

    def align_observations(
        self, location: ArrayLike, target: ArrayLike, /
    ) -> tuple[Array, Array]:
        location_array = jnp.asarray(location)
        target_array = jnp.asarray(target)
        if location_array.ndim >= 1 and int(location_array.shape[-1]) == 1:
            location_array = location_array[..., 0]
        if target_array.shape == location_array.shape + (1,):
            target_array = target_array[..., 0]
        if target_array.shape != location_array.shape:
            raise ValueError(
                "Ordinal prediction and target shapes are incompatible: "
                f"prediction={location_array.shape}, target={target_array.shape}."
            )
        return location_array, target_array

    def class_probabilities(self, location: ArrayLike, /) -> Array:
        return ordinal_class_probabilities_from_location(location, self.thresholds)

    def cumulative_probabilities(self, location: ArrayLike, /) -> Array:
        values = jnp.asarray(location, dtype=float)
        return jax.nn.sigmoid(self.thresholds - values[..., None])

    def exceedance_probabilities(self, location: ArrayLike, /) -> Array:
        values = jnp.asarray(location, dtype=float)
        return jax.nn.sigmoid(values[..., None] - self.thresholds)

    def log_prob(
        self, location: ArrayLike, target: ArrayLike, /, **parameters: Any
    ) -> Array:
        if parameters:
            raise TypeError(
                "OrdinalCumulativeLinkLikelihood received unknown parameters "
                f"{tuple(parameters)!r}."
            )
        location_array, target_array = self.align_observations(location, target)
        return ordinal_log_prob_from_location(
            location_array,
            target_array,
            self.thresholds,
        )

    def sample(self, key, location: ArrayLike, /, **parameters: Any) -> Array:
        if parameters:
            raise TypeError(
                "OrdinalCumulativeLinkLikelihood received unknown parameters "
                f"{tuple(parameters)!r}."
            )
        probabilities = self.class_probabilities(location)
        return jr.categorical(key, jnp.log(probabilities), axis=-1).astype(jnp.int32)

    def mode_category(self, location: ArrayLike, /) -> Array:
        return jnp.argmax(self.class_probabilities(location), axis=-1).astype(jnp.int32)

    def median_category(self, location: ArrayLike, /) -> Array:
        cumulative = self.cumulative_probabilities(location)
        return jnp.sum(cumulative < 0.5, axis=-1).astype(jnp.int32)


class GaussianLikelihood(_AbstractElementwiseLikelihood):
    """Gaussian observation likelihood with a fixed positive scale."""

    scale: Array

    def __init__(self, scale: ArrayLike):
        scale_array = jnp.asarray(scale, dtype=float)
        if bool(jnp.any(~jnp.isfinite(scale_array))) or bool(jnp.any(scale_array <= 0.0)):
            raise ValueError("Gaussian scale must be finite and strictly positive.")
        self.scale = scale_array

    def log_prob(
        self, location: ArrayLike, target: ArrayLike, /, **parameters: Any
    ) -> Array:
        if parameters:
            raise TypeError(
                f"GaussianLikelihood received unknown parameters {tuple(parameters)!r}."
            )
        location_array = jnp.asarray(location, dtype=float)
        target_array = jnp.asarray(target, dtype=float)
        standardized = (target_array - location_array) / self.scale
        return -0.5 * standardized**2 - jnp.log(self.scale) - 0.5 * jnp.log(2.0 * jnp.pi)

    def sample(self, key, location: ArrayLike, /, **parameters: Any) -> Array:
        if parameters:
            raise TypeError(
                f"GaussianLikelihood received unknown parameters {tuple(parameters)!r}."
            )
        location_array = jnp.asarray(location, dtype=float)
        shape = jnp.broadcast_shapes(location_array.shape, self.scale.shape)
        noise = jr.normal(key, shape=shape, dtype=location_array.dtype)
        return location_array + self.scale * noise


class GaussianLocationScaleLikelihood(_AbstractElementwiseLikelihood):
    """Heteroscedastic Gaussian with a softplus-transformed raw scale."""

    min_scale: float

    def __init__(self, *, min_scale: float = 1e-6):
        minimum = float(min_scale)
        if not jnp.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("min_scale must be finite and strictly positive.")
        self.min_scale = minimum

    def scale_from_raw(self, raw_scale: ArrayLike, /) -> Array:
        return jax_softplus(jnp.asarray(raw_scale, dtype=float)) + self.min_scale

    def log_prob(
        self,
        location: ArrayLike,
        target: ArrayLike,
        /,
        *,
        raw_scale: ArrayLike | None = None,
        **parameters: Any,
    ) -> Array:
        if parameters:
            raise TypeError(
                "GaussianLocationScaleLikelihood received unknown parameters "
                f"{tuple(parameters)!r}."
            )
        if raw_scale is None:
            raise ValueError("raw_scale is required.")
        scale = self.scale_from_raw(raw_scale)
        location_array = jnp.asarray(location, dtype=float)
        target_array = jnp.asarray(target, dtype=float)
        standardized = (target_array - location_array) / scale
        return -0.5 * standardized**2 - jnp.log(scale) - 0.5 * jnp.log(2.0 * jnp.pi)

    def sample(
        self,
        key,
        location: ArrayLike,
        /,
        *,
        raw_scale: ArrayLike | None = None,
        **parameters: Any,
    ) -> Array:
        if parameters:
            raise TypeError(
                "GaussianLocationScaleLikelihood received unknown parameters "
                f"{tuple(parameters)!r}."
            )
        if raw_scale is None:
            raise ValueError("raw_scale is required.")
        location_array = jnp.asarray(location, dtype=float)
        scale = self.scale_from_raw(raw_scale)
        shape = jnp.broadcast_shapes(location_array.shape, scale.shape)
        return location_array + scale * jr.normal(
            key, shape=shape, dtype=location_array.dtype
        )


class StudentTLikelihood(_AbstractElementwiseLikelihood):
    """Student-t observation likelihood with fixed degrees of freedom and scale."""

    df: Array
    scale: Array

    def __init__(self, df: ArrayLike, scale: ArrayLike):
        df_array = jnp.asarray(df, dtype=float)
        scale_array = jnp.asarray(scale, dtype=float)
        if bool(jnp.any(~jnp.isfinite(df_array))) or bool(jnp.any(df_array <= 0.0)):
            raise ValueError("Student-t degrees of freedom must be finite and positive.")
        if bool(jnp.any(~jnp.isfinite(scale_array))) or bool(jnp.any(scale_array <= 0.0)):
            raise ValueError("Student-t scale must be finite and strictly positive.")
        self.df = df_array
        self.scale = scale_array

    def log_prob(
        self, location: ArrayLike, target: ArrayLike, /, **parameters: Any
    ) -> Array:
        if parameters:
            raise TypeError(
                f"StudentTLikelihood received unknown parameters {tuple(parameters)!r}."
            )
        location_array = jnp.asarray(location, dtype=float)
        target_array = jnp.asarray(target, dtype=float)
        standardized = (target_array - location_array) / self.scale
        normalizer = (
            jsp.special.gammaln((self.df + 1.0) / 2.0)
            - jsp.special.gammaln(self.df / 2.0)
            - 0.5 * jnp.log(self.df * jnp.pi)
            - jnp.log(self.scale)
        )
        return normalizer - 0.5 * (self.df + 1.0) * jnp.log1p(standardized**2 / self.df)

    def sample(self, key, location: ArrayLike, /, **parameters: Any) -> Array:
        if parameters:
            raise TypeError(
                f"StudentTLikelihood received unknown parameters {tuple(parameters)!r}."
            )
        location_array = jnp.asarray(location, dtype=float)
        shape = jnp.broadcast_shapes(
            location_array.shape, self.scale.shape, self.df.shape
        )
        return location_array + self.scale * jr.t(
            key, self.df, shape=shape, dtype=location_array.dtype
        )


def jax_softplus(value: Array, /) -> Array:
    """Stable softplus kept local to avoid a public activation dependency."""
    return jnp.logaddexp(value, jnp.zeros((), dtype=value.dtype))


__all__ = [
    "AbstractLikelihood",
    "CategoricalExponentialFamilyLikelihood",
    "ScalarNaturalExponentialFamilyLikelihood",
    "IndependentBernoulliLikelihood",
    "OrdinalCumulativeLinkLikelihood",
    "GaussianLikelihood",
    "GaussianLocationScaleLikelihood",
    "StudentTLikelihood",
]
