#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact declared kernel-mean capabilities for Bayesian quadrature."""

from __future__ import annotations

from abc import abstractmethod
from numbers import Integral
from typing import Any

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from phydrax.domain import Interval1d, PointBatch
from phydrax.kernels import (
    AbstractFiniteFeatureKernel,
    AbstractPositiveDefiniteKernel,
    AmplitudeKernel,
    kernel_features,
    Matern32Kernel,
    Matern52Kernel,
    ScaleKernel,
    SquaredExponentialKernel,
)

from .._measure_weights import normalized_weights
from .._strict import StrictModule
from ._measure_transform import FiniteMeasureRealization, lower_finite_measure


class AbstractKernelMean(StrictModule):
    """Bound target/kernel embedding m(x) and double mean k_μμ."""

    kernel: AbstractPositiveDefiniteKernel
    target_mass: Array
    target_id: str = eqx.field(static=True)
    normalized: bool = eqx.field(static=True)
    exactness: str = eqx.field(static=True)
    hypotheses: str = eqx.field(static=True)

    @abstractmethod
    def mean(self, points: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def double_mean(self, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return self.kernel.matrix(left, right)

    @property
    def input_ndim(self) -> int:
        return self.kernel.input_ndim

    def __call__(self, points: ArrayLike, /) -> Array:
        return self.mean(points)


class IntervalKernelMean(AbstractKernelMean):
    """Exact interval mean for SE, Matérn-3/2, or Matérn-5/2 covariance."""

    lower: Array
    upper: Array
    amplitude: Array
    family: str = eqx.field(static=True)

    def __init__(
        self,
        interval: Interval1d,
        kernel: AbstractPositiveDefiniteKernel,
        /,
        *,
        normalized: bool = False,
        target_id: str = "interval",
    ):
        if not isinstance(interval, Interval1d):
            raise TypeError("interval must be an Interval1d.")
        base, amplitude = _single_scale(kernel)
        if not isinstance(
            base, (SquaredExponentialKernel, Matern32Kernel, Matern52Kernel)
        ):
            raise TypeError(
                "IntervalKernelMean supports squared-exponential and half-integer "
                "Matérn kernels with at most one scale/amplitude wrapper."
            )
        if base.length_scale.ndim != 0:
            raise ValueError("Interval kernel length_scale must be scalar.")
        lower = jnp.asarray(interval.start, dtype=base.length_scale.dtype).reshape(())
        upper = jnp.asarray(interval.end, dtype=base.length_scale.dtype).reshape(())
        length = upper - lower
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("target_id must be nonempty.")
        self.kernel = kernel
        self.lower = lower
        self.upper = upper
        self.amplitude = amplitude
        self.target_mass = jnp.asarray(1.0, dtype=length.dtype) if normalized else length
        self.target_id = target_id
        self.normalized = bool(normalized)
        self.family = type(base).__name__
        self.exactness = "analytic"
        self.hypotheses = "finite nondegenerate interval and supported stationary kernel"

    def mean(self, points: ArrayLike, /) -> Array:
        values = _scalar_points(points)
        inside = (values >= self.lower) & (values <= self.upper)
        values = eqx.error_if(
            values,
            jnp.any(~inside),
            "Interval kernel-mean points must lie inside the bound interval.",
        )
        left = values - self.lower
        right = self.upper - values
        mass = self.amplitude * (self._primitive(left) + self._primitive(right))
        length = self.upper - self.lower
        return mass / length if self.normalized else mass

    def double_mean(self, /) -> Array:
        length = self.upper - self.lower
        first = self._primitive(length)
        weighted = self._weighted_primitive(length)
        value = 2.0 * self.amplitude * (length * first - weighted)
        return value / (length * length) if self.normalized else value

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return super().matrix(left, right)

    def _primitive(self, distance: Array, /) -> Array:
        base, _ = _single_scale(self.kernel)
        length_scale = jnp.asarray(base.length_scale, dtype=distance.dtype)
        if isinstance(base, SquaredExponentialKernel):
            return (
                length_scale
                * jnp.sqrt(jnp.pi / 2.0)
                * jsp.special.erf(distance / (jnp.sqrt(2.0) * length_scale))
            )
        rate, coefficients = _matern_polynomial(base, dtype=distance.dtype)
        return _exponential_polynomial_integral(distance, rate, coefficients, shift=0)

    def _weighted_primitive(self, distance: Array, /) -> Array:
        base, _ = _single_scale(self.kernel)
        length_scale = jnp.asarray(base.length_scale, dtype=distance.dtype)
        if isinstance(base, SquaredExponentialKernel):
            return (
                length_scale
                * length_scale
                * (1.0 - jnp.exp(-0.5 * (distance / length_scale) ** 2))
            )
        rate, coefficients = _matern_polynomial(base, dtype=distance.dtype)
        return _exponential_polynomial_integral(distance, rate, coefficients, shift=1)


class FiniteMeasureKernelMean(AbstractKernelMean):
    """Exact kernel embedding of one explicit positive finite measure."""

    support: Array
    physical_weights: Array
    block_size: int = eqx.field(static=True)

    def __init__(
        self,
        realization: Any,
        kernel: AbstractPositiveDefiniteKernel,
        /,
        *,
        block_size: int = 256,
        target_id: str | None = None,
    ):
        if not isinstance(kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("kernel must be a positive-definite kernel.")
        if not isinstance(block_size, Integral) or isinstance(block_size, bool):
            raise TypeError("block_size must be an integer.")
        size = int(block_size)
        if size <= 0:
            raise ValueError("block_size must be positive.")
        measure = lower_finite_measure(realization)
        support = _kernel_support(measure, input_ndim=kernel.input_ndim)
        probabilities, _, valid, _ = normalized_weights(
            measure.count,
            log_weights=measure.log_weights,
            mask=measure.mask,
        )
        probabilities = eqx.error_if(
            probabilities,
            ~valid,
            "Finite kernel means require a positive finite source mass.",
        )
        physical_weights = probabilities * measure.physical_mass
        identifier = (
            f"finite-measure:{measure.source_provenance}"
            if target_id is None
            else str(target_id)
        )
        if not identifier:
            raise ValueError("target_id must be nonempty.")
        self.kernel = kernel
        self.support = support
        self.physical_weights = physical_weights
        self.target_mass = measure.physical_mass
        self.target_id = identifier
        self.normalized = measure.normalized
        self.block_size = size
        self.exactness = "explicit-finite-measure"
        self.hypotheses = "positive one-axis finite realization"

    def mean(self, points: ArrayLike, /) -> Array:
        values = jnp.asarray(points)
        matrix = self.kernel.matrix(values, self.support)
        return oe.contract("ij,j->i", matrix, self.physical_weights)

    def double_mean(self, /) -> Array:
        gram = self.kernel.matrix(self.support, self.support)
        return oe.contract("i,ij,j->", self.physical_weights, gram, self.physical_weights)

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return super().matrix(left, right)


class FiniteFeatureKernelMean(AbstractKernelMean):
    """Exact declared moment for a finite real feature kernel."""

    feature_moment: Array

    def __init__(
        self,
        kernel: AbstractFiniteFeatureKernel,
        feature_moment: ArrayLike,
        /,
        *,
        target_mass: ArrayLike = 1.0,
        target_id: str,
        normalized: bool = True,
    ):
        if not isinstance(kernel, AbstractFiniteFeatureKernel):
            raise TypeError("kernel must be an AbstractFiniteFeatureKernel.")
        moment = jnp.asarray(feature_moment, dtype=kernel.feature_factor.dtype)
        if moment.shape != (kernel.feature_rank,):
            raise ValueError("feature_moment must align with the whitened feature rank.")
        moment = eqx.error_if(
            moment,
            jnp.any(~jnp.isfinite(moment)),
            "feature_moment must be finite.",
        )
        mass = jnp.asarray(target_mass, dtype=moment.dtype)
        if mass.ndim != 0:
            raise ValueError("target_mass must be scalar.")
        mass = eqx.error_if(
            mass,
            ~jnp.isfinite(mass) | (mass <= 0.0),
            "target_mass must be finite and positive.",
        )
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("target_id must be nonempty.")
        self.kernel = kernel
        self.feature_moment = moment
        self.target_mass = mass
        self.target_id = target_id
        self.normalized = bool(normalized)
        self.exactness = "declared-finite-feature-moment"
        self.hypotheses = "caller certifies the supplied feature moment"

    def mean(self, points: ArrayLike, /) -> Array:
        return kernel_features(self.kernel, points) @ self.feature_moment

    def double_mean(self, /) -> Array:
        return oe.contract("i,i->", self.feature_moment, self.feature_moment)

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return super().matrix(left, right)


def _single_scale(
    kernel: AbstractPositiveDefiniteKernel,
    /,
) -> tuple[AbstractPositiveDefiniteKernel, Array]:
    if isinstance(kernel, ScaleKernel):
        if isinstance(kernel.kernel, (ScaleKernel, AmplitudeKernel)):
            raise TypeError("Only one interval kernel scale wrapper is supported.")
        return kernel.kernel, kernel.scale
    if isinstance(kernel, AmplitudeKernel):
        if isinstance(kernel.kernel, (ScaleKernel, AmplitudeKernel)):
            raise TypeError("Only one interval kernel scale wrapper is supported.")
        return kernel.kernel, kernel.variance_scale
    return kernel, jnp.asarray(1.0, dtype=float)


def _matern_polynomial(
    kernel: Matern32Kernel | Matern52Kernel,
    /,
    *,
    dtype: jnp.dtype,
) -> tuple[Array, tuple[Array, ...]]:
    length_scale = jnp.asarray(kernel.length_scale, dtype=dtype)
    if isinstance(kernel, Matern32Kernel):
        rate = jnp.sqrt(jnp.asarray(3.0, dtype=dtype)) / length_scale
        return rate, (jnp.asarray(1.0, dtype=dtype), rate)
    rate = jnp.sqrt(jnp.asarray(5.0, dtype=dtype)) / length_scale
    return rate, (
        jnp.asarray(1.0, dtype=dtype),
        rate,
        rate * rate / 3.0,
    )


def _exponential_polynomial_integral(
    distance: Array,
    rate: Array,
    coefficients: tuple[Array, ...],
    /,
    *,
    shift: int,
) -> Array:
    result = jnp.zeros_like(distance)
    for order, coefficient in enumerate(coefficients):
        power = order + shift
        gamma = jnp.asarray(float(_factorial(power)), dtype=distance.dtype)
        result = result + coefficient * gamma * jsp.special.gammainc(
            power + 1, rate * distance
        ) / rate ** (power + 1)
    return result


def _factorial(value: int, /) -> int:
    result = 1
    for factor in range(2, value + 1):
        result *= factor
    return result


def _scalar_points(value: ArrayLike, /) -> Array:
    points = jnp.asarray(value)
    if points.ndim == 2 and points.shape[1] == 1:
        points = points[:, 0]
    if points.ndim != 1:
        raise ValueError("Interval kernel-mean points must be scalar.")
    return points


def _kernel_support(
    measure: FiniteMeasureRealization,
    /,
    *,
    input_ndim: int,
) -> Array:
    samples = measure.samples
    axis = measure.axis
    if isinstance(samples, cx.Field):
        position = samples.dims.index(axis) if isinstance(axis, str) else axis
        support = jnp.moveaxis(samples.data, position, 0)
    elif isinstance(samples, (jax.Array, jax.core.Tracer)):
        position = axis if isinstance(axis, int) else 0
        support = jnp.moveaxis(samples, position, 0)
    elif isinstance(samples, PointBatch):
        fields = tuple(samples.points.values())
        if len(fields) != 1:
            raise TypeError(
                "Structured finite kernel means require one explicit kernel-input field."
            )
        field = fields[0]
        position = field.dims.index(axis) if isinstance(axis, str) else axis
        support = jnp.moveaxis(field.data, position, 0)
    else:
        raise TypeError(
            "Finite kernel means require array, Field, or one-field PointBatch samples."
        )
    if support.ndim != input_ndim + 1 or support.shape[0] != measure.count:
        raise ValueError("Finite-measure support does not match the kernel input rank.")
    return support


__all__ = [
    "AbstractKernelMean",
    "FiniteFeatureKernelMean",
    "FiniteMeasureKernelMean",
    "IntervalKernelMean",
]
