#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..sparse import RowRelation


class SparseFeatures(StrictModule):
    """Fixed-width sparse rows over a declared feature axis."""

    values: Array
    columns: RowRelation
    case_shape: tuple[int, ...] = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        column_indices: ArrayLike,
        /,
        *,
        feature_count: int,
        valid: ArrayLike | None = None,
        case_shape: tuple[int, ...] = (),
    ):
        values_ = jnp.asarray(values)
        columns_ = jnp.asarray(column_indices)
        cases = tuple(int(size) for size in case_shape)
        if values_.shape != columns_.shape:
            raise ValueError("Sparse feature values and column indices must match.")
        if values_.ndim != len(cases) + 2:
            raise ValueError(
                "Sparse features must have shape case_shape + (sample, row_width)."
            )
        if tuple(int(size) for size in values_.shape[: len(cases)]) != cases:
            raise ValueError("Sparse feature values do not begin with case_shape.")
        if int(values_.shape[-2]) <= 0 or int(values_.shape[-1]) <= 0:
            raise ValueError(
                "Sparse features require positive sample and row capacities."
            )
        feature_count_ = int(feature_count)
        if feature_count_ <= 0:
            raise ValueError("feature_count must be positive.")
        relation = RowRelation(
            columns_,
            source_size=feature_count_,
            valid=valid,
            case_shape=cases,
        )
        self.values = values_
        self.columns = relation
        self.case_shape = cases
        self.sample_count = int(values_.shape[-2])
        self.feature_count = feature_count_

    @property
    def shape(self) -> tuple[int, ...]:
        return self.case_shape + (self.sample_count, self.feature_count)

    @property
    def row_width(self) -> int:
        return int(self.values.shape[-1])

    def to_dense(self, /) -> Array:
        """Materialize dense features explicitly."""
        cases = self.columns.num_cases
        values = self.values.reshape((cases, self.sample_count, self.row_width))
        indices = self.columns.source_indices.reshape(values.shape)
        valid = self.columns.valid.reshape(values.shape)

        def materialize(case_values, case_indices, case_valid):
            rows = jnp.broadcast_to(
                jnp.arange(self.sample_count, dtype=jnp.int32)[:, None],
                case_indices.shape,
            )
            out = jnp.zeros(
                (self.sample_count, self.feature_count), dtype=case_values.dtype
            )
            return out.at[rows, case_indices].add(jnp.where(case_valid, case_values, 0))

        dense = jax.vmap(materialize)(values, indices, valid)
        return dense.reshape(self.shape)

    def right_matmul(self, matrix: ArrayLike, /) -> Array:
        """Apply the sparse row matrix to a dense feature-leading matrix."""
        matrix_ = jnp.asarray(matrix)
        if matrix_.ndim < 1 or int(matrix_.shape[0]) != self.feature_count:
            raise ValueError(
                f"matrix must begin with feature dimension {self.feature_count}."
            )
        gathered = matrix_[self.columns.source_indices]
        valid = self.columns.valid.reshape(
            self.columns.valid.shape + (1,) * (matrix_.ndim - 1)
        )
        weights = self.values.reshape(self.values.shape + (1,) * (matrix_.ndim - 1))
        return jnp.sum(jnp.where(valid, weights * gathered, 0), axis=-matrix_.ndim)

    def take_rows(self, indices: ArrayLike, /) -> "SparseFeatures":
        selected = jnp.asarray(indices, dtype=jnp.int32)
        if selected.ndim != 1:
            raise ValueError("Sparse row indices must be one-dimensional.")
        axis = len(self.case_shape)
        return SparseFeatures(
            jnp.take(self.values, selected, axis=axis),
            jnp.take(self.columns.source_indices, selected, axis=axis),
            feature_count=self.feature_count,
            valid=jnp.take(self.columns.valid, selected, axis=axis),
            case_shape=self.case_shape,
        )


FeatureArray = Array | SparseFeatures


def dense_features(features: FeatureArray, /) -> Array:
    """Return dense features, materializing sparse input only when explicitly called."""
    return features.to_dense() if isinstance(features, SparseFeatures) else features


__all__ = ["FeatureArray", "SparseFeatures", "dense_features"]
