#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ._cones import AbstractConvexCone


@jax.custom_jvp
def _positive_semidefinite_part(matrix: Array, /) -> Array:
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    positive = jnp.maximum(eigenvalues, 0.0)
    projected = (eigenvectors * positive[..., None, :]) @ jnp.swapaxes(
        eigenvectors, -1, -2
    )
    return 0.5 * (projected + jnp.swapaxes(projected, -1, -2))


@_positive_semidefinite_part.defjvp
def _positive_semidefinite_part_jvp(primals, tangents):
    (matrix,) = primals
    (tangent,) = tangents
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    positive = jnp.maximum(eigenvalues, 0.0)
    projected = (eigenvectors * positive[..., None, :]) @ jnp.swapaxes(
        eigenvectors, -1, -2
    )
    projected = 0.5 * (projected + jnp.swapaxes(projected, -1, -2))
    left = eigenvalues[..., :, None]
    right = eigenvalues[..., None, :]
    difference = left - right
    scale = jnp.maximum(jnp.maximum(jnp.abs(left), jnp.abs(right)), 1.0)
    close = jnp.abs(difference) <= jnp.sqrt(jnp.finfo(matrix.dtype).eps) * scale
    same_positive = (left > 0.0) & (right > 0.0)
    same_negative = (left < 0.0) & (right < 0.0)
    same_zero = (left == 0.0) & (right == 0.0)
    use_limit = close & (same_positive | same_negative | same_zero)
    safe_difference = jnp.where(use_limit, 1.0, difference)
    divided = (jnp.maximum(left, 0.0) - jnp.maximum(right, 0.0)) / safe_difference
    selected_derivative = jnp.where(
        same_positive,
        1.0,
        jnp.where(same_negative, 0.0, 0.5),
    )
    loewner = jnp.where(use_limit, selected_derivative, divided)
    transformed = jnp.swapaxes(eigenvectors, -1, -2) @ tangent @ eigenvectors
    derivative = (
        eigenvectors @ (loewner * transformed) @ jnp.swapaxes(eigenvectors, -1, -2)
    )
    derivative = 0.5 * (derivative + jnp.swapaxes(derivative, -1, -2))
    return projected, derivative


class PositiveSemidefiniteCone(AbstractConvexCone):
    """Real PSD cone in Clarabel's scaled upper-column triangular coordinates."""

    matrix_size: int = eqx.field(static=True)
    _row_indices: tuple[int, ...] = eqx.field(static=True)
    _column_indices: tuple[int, ...] = eqx.field(static=True)
    _scales: tuple[float, ...] = eqx.field(static=True)

    def __init__(self, matrix_size: int, /):
        size = int(matrix_size)
        if size < 1:
            raise ValueError("PositiveSemidefiniteCone matrix_size must be positive.")
        rows: list[int] = []
        columns: list[int] = []
        for column in range(size):
            for row in range(column + 1):
                rows.append(row)
                columns.append(column)
        row_indices = tuple(rows)
        column_indices = tuple(columns)
        scales = tuple(
            1.0 if row == column else 2.0**0.5
            for row, column in zip(row_indices, column_indices, strict=True)
        )
        self.matrix_size = size
        self.dimension = size * (size + 1) // 2
        self._row_indices = row_indices
        self._column_indices = column_indices
        self._scales = scales
        self.cone_id = canonical_fingerprint(
            {
                "kind": "positive-semidefinite-cone",
                "matrix_size": size,
                "packing": "scaled-upper-column-triangle",
            }
        )

    def _pack_unchecked(self, matrix: Array, /) -> Array:
        rows = jnp.asarray(self._row_indices, dtype=jnp.int32)
        columns = jnp.asarray(self._column_indices, dtype=jnp.int32)
        scales = jnp.asarray(self._scales, dtype=matrix.dtype)
        return matrix[..., rows, columns] * scales

    def pack(self, matrix: ArrayLike, /) -> Array:
        """Pack one real symmetric matrix with the Frobenius-isometric svec map."""

        value = jnp.asarray(matrix)
        expected = (self.matrix_size, self.matrix_size)
        if value.ndim < 2 or value.shape[-2:] != expected:
            raise ValueError(
                f"PSD matrix must end in shape {expected}; got {value.shape}."
            )
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError("PSD matrices must be real floating-point arrays.")
        scale = jnp.maximum(jnp.max(jnp.abs(value), axis=(-2, -1)), 1.0)
        residual = jnp.max(jnp.abs(value - jnp.swapaxes(value, -1, -2)), axis=(-2, -1))
        tolerance = 64.0 * float(self.matrix_size) * jnp.finfo(value.dtype).eps * scale
        guarded = eqx.error_if(
            value,
            jnp.any(residual > tolerance),
            "PSD pack requires a symmetric matrix.",
        )
        return self._pack_unchecked(guarded)

    def unpack(self, value: Any, /) -> Array:
        """Unpack scaled upper-column coordinates into a real symmetric matrix."""

        array = self._validate(value)
        rows = jnp.asarray(self._row_indices, dtype=jnp.int32)
        columns = jnp.asarray(self._column_indices, dtype=jnp.int32)
        scales = jnp.asarray(self._scales, dtype=array.dtype)
        upper = jnp.zeros(
            array.shape[:-1] + (self.matrix_size, self.matrix_size),
            dtype=array.dtype,
        )
        upper = upper.at[..., rows, columns].set(array / scales)
        diagonal = jnp.diagonal(upper, axis1=-2, axis2=-1)
        return (
            upper
            + jnp.swapaxes(upper, -1, -2)
            - jnp.eye(self.matrix_size, dtype=array.dtype) * diagonal[..., None, :]
        )

    def project(self, value: Any, /) -> Array:
        array = self._validate(value)
        return self._pack_unchecked(_positive_semidefinite_part(self.unpack(array)))

    def project_dual(self, value: Any, /) -> Array:
        return self.project(value)

    def interior_margin(self, value: Any, /) -> Array:
        array = self._validate(value)
        return jnp.min(jnp.linalg.eigvalsh(self.unpack(array)), axis=-1)

    def dual_projection_smoothness_margin(self, value: Any, /) -> Array:
        array = self._validate(value)
        eigenvalues = jnp.linalg.eigvalsh(self.unpack(array))
        return jnp.min(jnp.abs(eigenvalues), axis=-1)


__all__ = ["PositiveSemidefiniteCone"]
