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
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ._classification import (
    categorical_probabilities_from_logits,
    independent_bernoulli_log_prob_from_logits,
    independent_bernoulli_probabilities_from_logits,
    ordinal_class_probabilities_from_cumulative_logits,
    ordinal_log_prob_from_cumulative_logits,
    soft_ordinal_cross_entropy_from_cumulative_logits,
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
    """Ordered-logistic likelihood with fixed or model-produced cutpoints."""

    thresholds: Array | None
    prediction_mode: Literal["location", "cumulative_logits"] = eqx.field(static=True)
    _class_count: int = eqx.field(static=True)

    def __init__(
        self,
        thresholds: ArrayLike | None = None,
        *,
        class_count: int | None = None,
        prediction_mode: Literal["location", "cumulative_logits"] = "location",
    ):
        if prediction_mode not in ("location", "cumulative_logits"):
            raise ValueError("prediction_mode must be 'location' or 'cumulative_logits'.")
        if prediction_mode == "location":
            if thresholds is None:
                raise ValueError("Location-mode ordinal likelihood requires thresholds.")
            values = jnp.asarray(thresholds, dtype=float)
            if values.ndim != 1 or int(values.shape[0]) < 2:
                raise ValueError(
                    "Ordinal thresholds must contain at least two ordered values."
                )
            if bool(jnp.any(~jnp.isfinite(values))) or bool(
                jnp.any(values[1:] <= values[:-1])
            ):
                raise ValueError(
                    "Ordinal thresholds must be finite and strictly increasing."
                )
            resolved_count = int(values.shape[0]) + 1
            if class_count is not None and int(class_count) != resolved_count:
                raise ValueError("class_count does not match the fixed thresholds.")
            self.thresholds = values
        else:
            if thresholds is not None:
                raise ValueError(
                    "Cumulative-logit mode obtains cutpoints from the model output."
                )
            if class_count is None or int(class_count) < 3:
                raise ValueError(
                    "Cumulative-logit mode requires class_count of at least three."
                )
            resolved_count = int(class_count)
            self.thresholds = None
        self.prediction_mode = prediction_mode
        self._class_count = resolved_count

    @property
    def class_count(self) -> int:
        return self._class_count

    def _cumulative_logits(self, prediction: ArrayLike, /) -> Array:
        values = jnp.asarray(prediction, dtype=float)
        if self.prediction_mode == "location":
            if values.ndim >= 1 and int(values.shape[-1]) == 1:
                values = values[..., 0]
            assert self.thresholds is not None
            return self.thresholds - values[..., None]
        if values.ndim < 1 or int(values.shape[-1]) != self.class_count - 1:
            raise ValueError(
                "Learned ordinal predictions must end in class_count - 1 "
                "cumulative logits."
            )
        return values

    def align_observations(
        self, location: ArrayLike, target: ArrayLike, /
    ) -> tuple[Array, Array]:
        prediction = jnp.asarray(location)
        target_array = jnp.asarray(target)
        prefix = (
            prediction.shape
            if self.prediction_mode == "location"
            else prediction.shape[:-1]
        )
        if self.prediction_mode == "location" and prediction.ndim >= 1:
            if int(prediction.shape[-1]) == 1:
                prediction = prediction[..., 0]
                prefix = prediction.shape
        if target_array.shape == prefix + (1,):
            target_array = target_array[..., 0]
        if target_array.shape not in (prefix, prefix + (self.class_count,)):
            raise ValueError(
                "Ordinal prediction and target shapes are incompatible: "
                f"prediction={prediction.shape}, target={target_array.shape}."
            )
        return prediction, target_array

    def class_probabilities(self, location: ArrayLike, /) -> Array:
        return ordinal_class_probabilities_from_cumulative_logits(
            self._cumulative_logits(location)
        )

    def cumulative_probabilities(self, location: ArrayLike, /) -> Array:
        return jax.nn.sigmoid(self._cumulative_logits(location))

    def exceedance_probabilities(self, location: ArrayLike, /) -> Array:
        return jax.nn.sigmoid(-self._cumulative_logits(location))

    def log_prob(
        self, location: ArrayLike, target: ArrayLike, /, **parameters: Any
    ) -> Array:
        if parameters:
            raise TypeError(
                "OrdinalCumulativeLinkLikelihood received unknown parameters "
                f"{tuple(parameters)!r}."
            )
        prediction, target_array = self.align_observations(location, target)
        cumulative_logits = self._cumulative_logits(prediction)
        if target_array.shape == cumulative_logits.shape[:-1]:
            return ordinal_log_prob_from_cumulative_logits(
                cumulative_logits,
                target_array,
            )
        return -soft_ordinal_cross_entropy_from_cumulative_logits(
            cumulative_logits,
            target_array,
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
        location_array, target_array = _real_location_target(location, target)
        standardized = (target_array - location_array) / self.scale
        return -0.5 * standardized**2 - jnp.log(self.scale) - 0.5 * jnp.log(2.0 * jnp.pi)

    def sample(self, key, location: ArrayLike, /, **parameters: Any) -> Array:
        if parameters:
            raise TypeError(
                f"GaussianLikelihood received unknown parameters {tuple(parameters)!r}."
            )
        location_array = _real_location(location)
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
        location_array, target_array = _real_location_target(location, target)
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
        location_array = _real_location(location)
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
        location_array, target_array = _real_location_target(location, target)
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
        location_array = _real_location(location)
        shape = jnp.broadcast_shapes(
            location_array.shape, self.scale.shape, self.df.shape
        )
        return location_array + self.scale * jr.t(
            key, self.df, shape=shape, dtype=location_array.dtype
        )


class CircularComplexGaussianLikelihood(AbstractLikelihood):
    """Elementwise proper circular complex Gaussian likelihood.

    ``scale`` is the complex standard deviation: each real coordinate has
    variance ``scale**2 / 2``.  The density is normalized against product
    Lebesgue measure over real and imaginary coordinates.
    """

    scale: Array

    def __init__(self, scale: ArrayLike):
        scale_array = jnp.asarray(scale)
        if not jnp.issubdtype(scale_array.dtype, jnp.floating):
            raise TypeError("Circular complex Gaussian scale must be real floating.")
        if bool(jnp.any(~jnp.isfinite(scale_array))) or bool(jnp.any(scale_array <= 0.0)):
            raise ValueError(
                "Circular complex Gaussian scale must be finite and strictly positive."
            )
        self.scale = scale_array

    def align_observations(
        self, location: ArrayLike, target: ArrayLike, /
    ) -> tuple[Array, Array]:
        location_array, target_array = super().align_observations(location, target)
        if not (
            jnp.issubdtype(location_array.dtype, jnp.complexfloating)
            or jnp.issubdtype(target_array.dtype, jnp.complexfloating)
        ):
            raise TypeError(
                "CircularComplexGaussianLikelihood requires complex observations."
            )
        dtype = jnp.result_type(location_array.dtype, target_array.dtype)
        return location_array.astype(dtype), target_array.astype(dtype)

    def log_prob(
        self, location: ArrayLike, target: ArrayLike, /, **parameters: Any
    ) -> Array:
        if parameters:
            raise TypeError(
                "CircularComplexGaussianLikelihood received unknown parameters "
                f"{tuple(parameters)!r}."
            )
        location_array, target_array = self.align_observations(location, target)
        squared_residual = jnp.real(
            (target_array - location_array) * jnp.conj(target_array - location_array)
        )
        variance = self.scale**2
        return -squared_residual / variance - jnp.log(jnp.pi * variance)

    def sample(self, key, location: ArrayLike, /, **parameters: Any) -> Array:
        if parameters:
            raise TypeError(
                "CircularComplexGaussianLikelihood received unknown parameters "
                f"{tuple(parameters)!r}."
            )
        location_array = jnp.asarray(location)
        if not jnp.issubdtype(location_array.dtype, jnp.complexfloating):
            raise TypeError(
                "CircularComplexGaussianLikelihood requires a complex location."
            )
        shape = jnp.broadcast_shapes(location_array.shape, self.scale.shape)
        real_dtype = jnp.real(location_array).dtype
        real_key, imaginary_key = jr.split(key)
        noise = (
            jr.normal(real_key, shape=shape, dtype=real_dtype)
            + 1j * jr.normal(imaginary_key, shape=shape, dtype=real_dtype)
        ) / jnp.sqrt(jnp.asarray(2.0, dtype=real_dtype))
        return location_array + self.scale * noise


class ComplexGaussianLikelihood(AbstractLikelihood):
    """Dense proper or improper complex Gaussian over one trailing event.

    Construction prepares the equivalent real covariance for
    ``[Re(event), Im(event)]``.  No repair is implicit: callers must explicitly
    declare any diagonal ``regularization`` and all invalid factors fail before
    transformed evaluation.
    """

    real_covariance: Array
    real_precision: Array
    real_factor: Array
    log_normalizer: Array
    regularization: Array
    covariance: Array
    pseudo_covariance: Array
    event_size: int = eqx.field(static=True)
    proper: bool = eqx.field(static=True)

    def __init__(
        self,
        covariance: ArrayLike,
        pseudo_covariance: ArrayLike | None = None,
        /,
        *,
        regularization: ArrayLike = 0.0,
        hermitian_tolerance: float = 0.0,
        symmetry_tolerance: float = 0.0,
    ):
        covariance_array = jnp.asarray(covariance)
        if (
            covariance_array.ndim != 2
            or covariance_array.shape[0] == 0
            or covariance_array.shape[0] != covariance_array.shape[1]
        ):
            raise ValueError("covariance must be a nonempty square matrix.")
        if not jnp.issubdtype(covariance_array.dtype, jnp.complexfloating):
            raise TypeError("Complex covariance must have complex floating dtype.")
        event_size = int(covariance_array.shape[0])
        pseudo_array = (
            jnp.zeros_like(covariance_array)
            if pseudo_covariance is None
            else jnp.asarray(pseudo_covariance, dtype=covariance_array.dtype)
        )
        if pseudo_array.shape != covariance_array.shape:
            raise ValueError("pseudo_covariance must have the covariance shape.")
        if bool(jnp.any(~jnp.isfinite(covariance_array))) or bool(
            jnp.any(~jnp.isfinite(pseudo_array))
        ):
            raise ValueError("Complex covariance inputs must be finite.")
        hermitian_defect = jnp.max(
            jnp.abs(covariance_array - jnp.conj(covariance_array.T))
        )
        if bool(hermitian_defect > float(hermitian_tolerance)):
            raise ValueError(
                "covariance must be Hermitian within the declared tolerance."
            )
        symmetry_defect = jnp.max(jnp.abs(pseudo_array - pseudo_array.T))
        if bool(symmetry_defect > float(symmetry_tolerance)):
            raise ValueError(
                "pseudo_covariance must be symmetric within the declared tolerance."
            )
        real_dtype = jnp.real(covariance_array).dtype
        regularization_array = jnp.asarray(regularization, dtype=real_dtype)
        if (
            regularization_array.ndim != 0
            or bool(~jnp.isfinite(regularization_array))
            or bool(regularization_array < 0.0)
        ):
            raise ValueError("regularization must be a finite nonnegative real scalar.")
        covariance_plus = covariance_array + pseudo_array
        covariance_minus = covariance_array - pseudo_array
        real_block = 0.5 * jnp.block(
            [
                [jnp.real(covariance_plus), -jnp.imag(covariance_minus)],
                [jnp.imag(covariance_plus), jnp.real(covariance_minus)],
            ]
        )
        real_block = real_block + regularization_array * jnp.eye(
            2 * event_size, dtype=real_dtype
        )
        eigenvalues, eigenvectors = jnp.linalg.eigh(real_block, symmetrize_input=False)
        if bool(jnp.any(~jnp.isfinite(eigenvalues))) or bool(jnp.any(eigenvalues <= 0.0)):
            raise ValueError(
                "The declared complex covariance does not define a nonsingular density."
            )
        inverse_values = 1.0 / eigenvalues
        root_values = jnp.sqrt(eigenvalues)
        precision = oe.contract(
            "ik,k,jk->ij", eigenvectors, inverse_values, jnp.conj(eigenvectors)
        )
        factor = eigenvectors * root_values[None, :]
        log_determinant = jnp.sum(jnp.log(eigenvalues))
        log_normalizer = -0.5 * (
            log_determinant
            + jnp.asarray(2 * event_size, dtype=real_dtype)
            * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=real_dtype))
        )
        self.real_covariance = real_block
        self.real_precision = jnp.real(precision)
        self.real_factor = jnp.real(factor)
        self.log_normalizer = jnp.real(log_normalizer)
        self.regularization = regularization_array
        self.covariance = covariance_array
        self.pseudo_covariance = pseudo_array
        self.event_size = event_size
        self.proper = bool(jnp.all(pseudo_array == 0.0))

    @classmethod
    def from_covariances(
        cls,
        covariance: ArrayLike,
        pseudo_covariance: ArrayLike | None = None,
        /,
        **kwargs: Any,
    ) -> ComplexGaussianLikelihood:
        """Prepare a normalized dense complex Gaussian likelihood."""
        return cls(covariance, pseudo_covariance, **kwargs)

    def align_observations(
        self, location: ArrayLike, target: ArrayLike, /
    ) -> tuple[Array, Array]:
        location_array, target_array = super().align_observations(location, target)
        if location_array.shape[-1:] != (self.event_size,):
            raise ValueError(
                "Complex Gaussian observations must have the prepared trailing event."
            )
        if not (
            jnp.issubdtype(location_array.dtype, jnp.complexfloating)
            or jnp.issubdtype(target_array.dtype, jnp.complexfloating)
        ):
            raise TypeError("ComplexGaussianLikelihood requires complex observations.")
        dtype = jnp.result_type(location_array.dtype, target_array.dtype)
        return location_array.astype(dtype), target_array.astype(dtype)

    def log_prob(
        self, location: ArrayLike, target: ArrayLike, /, **parameters: Any
    ) -> Array:
        if parameters:
            raise TypeError(
                "ComplexGaussianLikelihood received unknown parameters "
                f"{tuple(parameters)!r}."
            )
        location_array, target_array = self.align_observations(location, target)
        residual = target_array - location_array
        coordinates = jnp.concatenate((jnp.real(residual), jnp.imag(residual)), axis=-1)
        quadratic = oe.contract(
            "...i,ij,...j->...", coordinates, self.real_precision, coordinates
        )
        return jnp.real(self.log_normalizer - 0.5 * quadratic)

    def sample(self, key, location: ArrayLike, /, **parameters: Any) -> Array:
        if parameters:
            raise TypeError(
                "ComplexGaussianLikelihood received unknown parameters "
                f"{tuple(parameters)!r}."
            )
        location_array = jnp.asarray(location)
        if location_array.shape[-1:] != (self.event_size,):
            raise ValueError(
                "Complex Gaussian location must have the prepared trailing event."
            )
        if not jnp.issubdtype(location_array.dtype, jnp.complexfloating):
            raise TypeError("ComplexGaussianLikelihood requires a complex location.")
        real_dtype = jnp.real(location_array).dtype
        noise_shape = (*location_array.shape[:-1], 2 * self.event_size)
        standard = jr.normal(key, shape=noise_shape, dtype=real_dtype)
        coordinates = oe.contract("ij,...j->...i", self.real_factor, standard)
        noise = (
            coordinates[..., : self.event_size] + 1j * coordinates[..., self.event_size :]
        )
        return location_array + noise


def _real_location(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError("Real observation likelihoods do not accept complex values.")
    return array.astype(jnp.result_type(array.dtype, float))


def _real_location_target(
    location: ArrayLike, target: ArrayLike, /
) -> tuple[Array, Array]:
    return _real_location(location), _real_location(target)


def jax_softplus(value: Array, /) -> Array:
    """Stable softplus kept local to avoid a public activation dependency."""
    return jnp.logaddexp(value, jnp.zeros((), dtype=value.dtype))


__all__ = [
    "AbstractLikelihood",
    "CategoricalExponentialFamilyLikelihood",
    "CircularComplexGaussianLikelihood",
    "ComplexGaussianLikelihood",
    "ScalarNaturalExponentialFamilyLikelihood",
    "IndependentBernoulliLikelihood",
    "OrdinalCumulativeLinkLikelihood",
    "GaussianLikelihood",
    "GaussianLocationScaleLikelihood",
    "StudentTLikelihood",
]
