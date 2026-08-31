#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._multiindex import total_degree_multiindices


class ScaledMonomialBasis(StrictModule, NonTrainableState):
    """Complete total-degree monomials under cellwise affine scaling."""

    exponents: Array
    dimension: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        degree: int,
        /,
        *,
        maximum_features: int = 4096,
    ):
        indices = total_degree_multiindices(dimension, degree)
        if len(indices) > int(maximum_features):
            raise ValueError("Scaled monomial feature budget exceeded.")
        exponents = np.asarray(indices, dtype=np.int32)
        self.exponents = jnp.asarray(exponents)
        self.dimension = int(dimension)
        self.degree = int(degree)
        self.feature_count = len(indices)
        self.basis_id = canonical_fingerprint(
            {
                "kind": "scaled-total-degree-monomials",
                "dimension": self.dimension,
                "degree": self.degree,
                "exponents": array_tree_fingerprint(exponents),
            }
        )

    def _normalized(
        self,
        points: ArrayLike,
        centers: ArrayLike,
        scales: ArrayLike,
        /,
    ) -> Array:
        points_ = jnp.asarray(points)
        centers_ = jnp.asarray(centers)
        scales_ = jnp.asarray(scales)
        if points_.shape[-1] != self.dimension or centers_.shape[-1] != self.dimension:
            raise ValueError("Monomial points and centers have incompatible dimension.")
        if points_.ndim == centers_.ndim + 1:
            centers_ = centers_[..., None, :]
            scales_ = scales_[..., None]
        safe = jnp.maximum(jnp.abs(scales_), jnp.finfo(points_.dtype).tiny)
        return (points_ - centers_) / safe[..., None]

    def evaluate(
        self,
        points: ArrayLike,
        centers: ArrayLike,
        scales: ArrayLike,
        /,
    ) -> Array:
        normalized = self._normalized(points, centers, scales)
        return jnp.prod(
            normalized[..., None, :] ** self.exponents.astype(normalized.dtype),
            axis=-1,
        )

    def gradient(
        self,
        points: ArrayLike,
        centers: ArrayLike,
        scales: ArrayLike,
        /,
    ) -> Array:
        normalized = self._normalized(points, centers, scales)
        scales_ = jnp.asarray(scales)
        if normalized.ndim == scales_.ndim + 2:
            scales_ = scales_[..., None]
        safe = jnp.maximum(jnp.abs(scales_), jnp.finfo(normalized.dtype).tiny)
        components = []
        exponents = self.exponents
        for axis in range(self.dimension):
            coefficient = exponents[:, axis].astype(normalized.dtype)
            reduced = exponents.at[:, axis].set(jnp.maximum(exponents[:, axis] - 1, 0))
            values = jnp.prod(
                normalized[..., None, :] ** reduced.astype(normalized.dtype), axis=-1
            )
            components.append(values * coefficient / safe[..., None])
        return jnp.stack(tuple(components), axis=-1)

    def laplacian(
        self,
        points: ArrayLike,
        centers: ArrayLike,
        scales: ArrayLike,
        /,
    ) -> Array:
        normalized = self._normalized(points, centers, scales)
        scales_ = jnp.asarray(scales)
        if normalized.ndim == scales_.ndim + 2:
            scales_ = scales_[..., None]
        safe_squared = jnp.maximum(
            scales_ * scales_, jnp.finfo(normalized.dtype).tiny
        )
        result = jnp.zeros(normalized.shape[:-1] + (self.feature_count,), dtype=normalized.dtype)
        exponents = self.exponents
        for axis in range(self.dimension):
            exponent = exponents[:, axis]
            coefficient = (exponent * jnp.maximum(exponent - 1, 0)).astype(
                normalized.dtype
            )
            reduced = exponents.at[:, axis].set(jnp.maximum(exponent - 2, 0))
            values = jnp.prod(
                normalized[..., None, :] ** reduced.astype(normalized.dtype), axis=-1
            )
            result = result + values * coefficient / safe_squared[..., None]
        return result

    @property
    def storage_bytes(self) -> int:
        return self.feature_count * self.dimension * math.ceil(32 / 8)


__all__ = ["ScaledMonomialBasis"]
