#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
import math
from numbers import Integral
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._orthogonal import standard_vandermonde


PolynomialChaosMeasure: TypeAlias = Literal["uniform", "standard-normal"]
_DEFAULT_MAXIMUM_FEATURES = 4096
_DEFAULT_MAXIMUM_BYTES = 64 * 1024**2


class PolynomialMultiIndexSet(StrictModule, NonTrainableState):
    """Deterministic graded total-degree multiindices including the constant mode."""

    indices: Array
    dimension: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    storage_bytes: int = eqx.field(static=True)
    content_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        degree: int,
        /,
        *,
        maximum_features: int = _DEFAULT_MAXIMUM_FEATURES,
        maximum_storage_bytes: int = _DEFAULT_MAXIMUM_BYTES,
    ):
        dimension_ = _positive_integer(dimension, "dimension")
        degree_ = _nonnegative_integer(degree, "degree")
        maximum = _positive_integer(maximum_features, "maximum_features")
        maximum_bytes = _positive_integer(
            maximum_storage_bytes, "maximum_storage_bytes"
        )
        feature_count = math.comb(dimension_ + degree_, degree_)
        if feature_count > maximum:
            raise ValueError(
                f"Total-degree polynomial chaos requires {feature_count} features, "
                f"exceeding maximum_features={maximum}."
            )
        storage_bytes = (
            feature_count * dimension_ * np.dtype(np.int32).itemsize
        )
        if storage_bytes > maximum_bytes:
            raise ValueError(
                "Total-degree multiindex data exceeds maximum_storage_bytes."
            )

        rows: list[np.ndarray] = [np.zeros((dimension_,), dtype=np.int32)]
        for total_degree in range(1, degree_ + 1):
            for coordinates in itertools.combinations_with_replacement(
                range(dimension_), total_degree
            ):
                index = np.zeros((dimension_,), dtype=np.int32)
                for coordinate in coordinates:
                    index[coordinate] += 1
                rows.append(index)
        indices = np.asarray(rows, dtype=np.int32).reshape(
            (feature_count, dimension_)
        )
        if int(indices.nbytes) != storage_bytes:
            raise RuntimeError(
                "Polynomial multiindex storage accounting is inconsistent."
            )

        self.indices = jnp.asarray(indices)
        self.dimension = dimension_
        self.degree = degree_
        self.feature_count = feature_count
        self.storage_bytes = storage_bytes
        self.content_id = canonical_fingerprint(
            {
                "kind": "polynomial-chaos-total-degree-multiindices-v1",
                "dimension": dimension_,
                "degree": degree_,
                "feature_count": feature_count,
                "storage_bytes": storage_bytes,
                "indices": array_tree_fingerprint(indices),
            }
        )


def normalized_vandermonde(
    measure: PolynomialChaosMeasure,
    points: ArrayLike,
    degree: int,
    /,
) -> Array:
    """Evaluate a one-dimensional orthonormal polynomial family."""
    values = jnp.asarray(points)
    degree_ = _nonnegative_integer(degree, "degree")
    if measure == "uniform":
        vandermonde = standard_vandermonde("legendre", values.reshape((-1,)), degree_)
        normalization = jnp.sqrt(
            2 * jnp.arange(degree_ + 1, dtype=values.dtype) + 1
        )
    elif measure == "standard-normal":
        vandermonde = standard_vandermonde("hermite_e", values.reshape((-1,)), degree_)
        normalization = jnp.sqrt(
            jnp.asarray(
                [math.factorial(index) for index in range(degree_ + 1)],
                dtype=values.dtype,
            )
        )
        normalization = 1.0 / normalization
    else:
        raise ValueError(f"Unsupported polynomial-chaos measure {measure!r}.")
    return (vandermonde * normalization).reshape(values.shape + (degree_ + 1,))


def evaluate_tensor_basis(
    reference_points: ArrayLike,
    measures: tuple[PolynomialChaosMeasure, ...],
    multiindices: PolynomialMultiIndexSet,
    /,
) -> Array:
    """Evaluate tensor modes in multiindex order using named factor semantics."""
    points = jnp.asarray(reference_points)
    if points.ndim < 1 or points.shape[-1] != multiindices.dimension:
        raise ValueError(
            "reference_points must end with the multiindex-set dimension."
        )
    if len(measures) != multiindices.dimension:
        raise ValueError("One polynomial measure is required per reference coordinate.")

    result = jnp.ones(
        points.shape[:-1] + (multiindices.feature_count,), dtype=points.dtype
    )
    for coordinate, measure in enumerate(measures):
        univariate = normalized_vandermonde(
            measure,
            points[..., coordinate],
            multiindices.degree,
        )
        selected = jnp.take(
            univariate,
            multiindices.indices[:, coordinate],
            axis=-1,
        )
        result = oe.contract("...k,...k->...k", result, selected)
    return result


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


__all__ = [
    "evaluate_tensor_basis",
    "normalized_vandermonde",
    "PolynomialChaosMeasure",
    "PolynomialMultiIndexSet",
]
