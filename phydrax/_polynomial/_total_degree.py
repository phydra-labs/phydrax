#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
import math
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


_DEFAULT_MAXIMUM_FEATURES = 4096
_DEFAULT_MAXIMUM_BYTES = 64 * 1024**2


class TotalDegreePolynomialFeatures(StrictModule, NonTrainableState):
    """Prepared nonconstant total-degree monomials in standardized coordinates."""

    exponents: Array
    dimension: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    storage_bytes: int = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        degree: int,
        /,
        *,
        maximum_features: int = _DEFAULT_MAXIMUM_FEATURES,
        maximum_feature_bytes: int = _DEFAULT_MAXIMUM_BYTES,
    ):
        dimension_ = _positive_integer(dimension, "dimension")
        degree_ = _nonnegative_integer(degree, "degree")
        maximum = _positive_integer(maximum_features, "maximum_features")
        maximum_bytes = _positive_integer(maximum_feature_bytes, "maximum_feature_bytes")
        feature_count = math.comb(dimension_ + degree_, degree_) - 1
        if feature_count > maximum:
            raise ValueError(
                f"Total-degree basis requires {feature_count} features, "
                f"exceeding maximum_features={maximum}."
            )
        exponent_rows: list[np.ndarray] = []
        for total_degree in range(1, degree_ + 1):
            for coordinates in itertools.combinations_with_replacement(
                range(dimension_), total_degree
            ):
                exponent = np.zeros((dimension_,), dtype=np.int32)
                for coordinate in coordinates:
                    exponent[coordinate] += 1
                exponent_rows.append(exponent)
        exponents = np.asarray(exponent_rows, dtype=np.int32).reshape(
            (feature_count, dimension_)
        )
        storage_bytes = int(exponents.nbytes)
        if storage_bytes > maximum_bytes:
            raise ValueError("Total-degree exponent data exceeds maximum_feature_bytes.")
        self.exponents = jnp.asarray(exponents)
        self.dimension = dimension_
        self.degree = degree_
        self.feature_count = feature_count
        self.storage_bytes = storage_bytes
        self.feature_id = canonical_fingerprint(
            {
                "kind": "total-degree-polynomial-features-v1",
                "dimension": dimension_,
                "degree": degree_,
                "feature_count": feature_count,
                "storage_bytes": storage_bytes,
                "exponents": array_tree_fingerprint(exponents),
            }
        )

    def evaluate(
        self,
        points: ArrayLike,
        weights: ArrayLike,
        /,
    ) -> tuple[Array, Array, Array]:
        """Evaluate the polynomial span after weighted affine standardization."""
        values = jnp.asarray(points, dtype=float)
        normalized_weights = jnp.asarray(weights, dtype=float).reshape((-1,))
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"points must have shape (num_points, {self.dimension}).")
        if normalized_weights.shape != values.shape[:1]:
            raise ValueError("weights must contain one value per point.")
        center = normalized_weights @ values
        centered = values - center
        variance = normalized_weights @ (centered * centered)
        floor = jnp.asarray(jnp.finfo(values.dtype).eps * 64.0, dtype=values.dtype)
        scale = jnp.sqrt(jnp.maximum(variance, 0.0))
        safe_scale = jnp.where(scale > floor, scale, 1.0)
        standardized = centered / safe_scale
        if self.feature_count == 0:
            features = jnp.zeros((values.shape[0], 0), dtype=values.dtype)
        else:
            features = jnp.prod(
                standardized[:, None, :] ** self.exponents[None, :, :],
                axis=-1,
            )
        return features, center, safe_scale


def _positive_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


__all__ = ["TotalDegreePolynomialFeatures"]
