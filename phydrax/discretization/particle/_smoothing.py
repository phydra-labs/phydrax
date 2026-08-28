#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class AbstractSPHSmoothingKernel(StrictModule, NonTrainableState):
    """Normalized compact radial kernel with q = distance / smoothing_length."""

    dimension: int = eqx.field(static=True)
    support_factor: float = eqx.field(static=True)
    normalization: float = eqx.field(static=True)
    regularity: str = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def _profile(self, q: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def _profile_derivative(self, q: Array, /) -> Array:
        raise NotImplementedError

    def _arguments(
        self, distance: ArrayLike, smoothing_length: ArrayLike, /
    ) -> tuple[Array, Array, Array]:
        distance_ = jnp.asarray(distance)
        if not jnp.issubdtype(distance_.dtype, jnp.inexact):
            distance_ = distance_.astype(jnp.float32)
        smoothing_ = jnp.asarray(smoothing_length, dtype=distance_.dtype)
        distance_ = eqx.error_if(
            distance_,
            jnp.any(~jnp.isfinite(distance_) | (distance_ < 0.0)),
            "SPH kernel distances must be finite and non-negative.",
        )
        smoothing_ = eqx.error_if(
            smoothing_,
            jnp.any(~jnp.isfinite(smoothing_) | (smoothing_ <= 0.0)),
            "SPH smoothing lengths must be finite and positive.",
        )
        return distance_, smoothing_, distance_ / smoothing_

    def value(self, distance: ArrayLike, smoothing_length: ArrayLike, /) -> Array:
        distance_, smoothing_, q = self._arguments(distance, smoothing_length)
        del distance_
        scale = (
            jnp.asarray(self.normalization, dtype=q.dtype) / smoothing_**self.dimension
        )
        return scale * self._profile(q)

    def radial_derivative(
        self, distance: ArrayLike, smoothing_length: ArrayLike, /
    ) -> Array:
        distance_, smoothing_, q = self._arguments(distance, smoothing_length)
        del distance_
        scale = jnp.asarray(self.normalization, dtype=q.dtype) / smoothing_ ** (
            self.dimension + 1
        )
        return scale * self._profile_derivative(q)

    def gradient(
        self,
        displacement: ArrayLike,
        distance: ArrayLike,
        smoothing_length: ArrayLike,
        /,
    ) -> Array:
        displacement_ = jnp.asarray(displacement)
        distance_ = jnp.asarray(distance, dtype=displacement_.dtype)
        if displacement_.ndim < 1 or displacement_.shape[-1] != self.dimension:
            raise ValueError("SPH displacement must end in the kernel dimension.")
        if distance_.shape != displacement_.shape[:-1]:
            raise ValueError("SPH distances must match displacement leading axes.")
        derivative = self.radial_derivative(distance_, smoothing_length)
        positive = distance_ > 0.0
        safe_distance = jnp.where(positive, distance_, 1.0)
        direction = jnp.where(
            positive[..., None],
            displacement_ / safe_distance[..., None],
            0.0,
        )
        return derivative[..., None] * direction

    def smoothing_length_derivative(
        self, distance: ArrayLike, smoothing_length: ArrayLike, /
    ) -> Array:
        distance_, smoothing_, q = self._arguments(distance, smoothing_length)
        del distance_
        profile = self._profile(q)
        derivative = self._profile_derivative(q)
        scale = jnp.asarray(self.normalization, dtype=q.dtype) / smoothing_ ** (
            self.dimension + 1
        )
        return scale * (-self.dimension * profile - q * derivative)

    def support_radius(self, smoothing_length: ArrayLike, /) -> Array:
        smoothing_ = jnp.asarray(smoothing_length)
        smoothing_ = eqx.error_if(
            smoothing_,
            jnp.any(~jnp.isfinite(smoothing_) | (smoothing_ <= 0.0)),
            "SPH smoothing lengths must be finite and positive.",
        )
        return self.support_factor * smoothing_


class WendlandC2SPHKernel(AbstractSPHSmoothingKernel):
    """Wendland C2 kernel with compact support radius 2h."""

    def __init__(self, dimension: int, /):
        dimension_ = int(dimension)
        normalizations = {
            1: 3.0 / 4.0,
            2: 7.0 / (4.0 * np.pi),
            3: 21.0 / (16.0 * np.pi),
        }
        if dimension_ not in normalizations:
            raise ValueError("WendlandC2SPHKernel supports dimensions 1, 2, and 3.")
        normalization = normalizations[dimension_]
        self.dimension = dimension_
        self.support_factor = 2.0
        self.normalization = normalization
        self.regularity = "C2"
        self.kernel_id = canonical_fingerprint(
            {
                "kind": "wendland-c2-sph-kernel",
                "dimension": dimension_,
                "support_factor": 2.0,
                "normalization": normalization,
            }
        )

    def _profile(self, q: Array, /) -> Array:
        inside = (q >= 0.0) & (q < self.support_factor)
        factor = jnp.maximum(1.0 - 0.5 * q, 0.0)
        return jnp.where(inside, factor**4 * (2.0 * q + 1.0), 0.0)

    def _profile_derivative(self, q: Array, /) -> Array:
        inside = (q >= 0.0) & (q < self.support_factor)
        factor = jnp.maximum(1.0 - 0.5 * q, 0.0)
        return jnp.where(inside, -5.0 * q * factor**3, 0.0)


class CubicSplineSPHKernel(AbstractSPHSmoothingKernel):
    """Cubic B-spline SPH kernel with compact support radius 2h."""

    def __init__(self, dimension: int, /):
        dimension_ = int(dimension)
        normalizations = {
            1: 2.0 / 3.0,
            2: 10.0 / (7.0 * np.pi),
            3: 1.0 / np.pi,
        }
        if dimension_ not in normalizations:
            raise ValueError("CubicSplineSPHKernel supports dimensions 1, 2, and 3.")
        normalization = normalizations[dimension_]
        self.dimension = dimension_
        self.support_factor = 2.0
        self.normalization = normalization
        self.regularity = "C2-piecewise"
        self.kernel_id = canonical_fingerprint(
            {
                "kind": "cubic-spline-sph-kernel",
                "dimension": dimension_,
                "support_factor": 2.0,
                "normalization": normalization,
            }
        )

    def _profile(self, q: Array, /) -> Array:
        lower = 1.0 - 1.5 * q**2 + 0.75 * q**3
        upper = 0.25 * (2.0 - q) ** 3
        return jnp.where(
            (q >= 0.0) & (q < 1.0),
            lower,
            jnp.where((q >= 1.0) & (q < 2.0), upper, 0.0),
        )

    def _profile_derivative(self, q: Array, /) -> Array:
        lower = -3.0 * q + 2.25 * q**2
        upper = -0.75 * (2.0 - q) ** 2
        return jnp.where(
            (q >= 0.0) & (q < 1.0),
            lower,
            jnp.where((q >= 1.0) & (q < 2.0), upper, 0.0),
        )


__all__ = [
    "AbstractSPHSmoothingKernel",
    "CubicSplineSPHKernel",
    "WendlandC2SPHKernel",
]
