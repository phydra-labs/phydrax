#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from operator import index
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ...linalg import HermitianSpectrum
from ._cones import AbstractConvexCone


@jax.custom_jvp
def _positive_semidefinite_part(matrix: Array, /) -> Array:
    working_dtype = jnp.result_type(matrix.dtype, jnp.float32)
    working = matrix.astype(working_dtype)
    symmetric = 0.5 * working + 0.5 * jnp.swapaxes(working, -1, -2)
    spectrum = HermitianSpectrum(
        symmetric,
        tolerance=64.0 * jnp.finfo(working_dtype).eps,
    )
    eigenvalues = spectrum.eigenvalues
    eigenvectors = spectrum.eigenvectors
    positive = jnp.maximum(eigenvalues, 0.0)
    projected = (eigenvectors * positive[..., None, :]) @ jnp.swapaxes(
        eigenvectors, -1, -2
    )
    projected = 0.5 * projected + 0.5 * jnp.swapaxes(projected, -1, -2)
    return projected.astype(matrix.dtype)


@_positive_semidefinite_part.defjvp
def _positive_semidefinite_part_jvp(primals, tangents):
    (matrix,) = primals
    (tangent,) = tangents
    working_dtype = jnp.result_type(matrix.dtype, jnp.float32)
    matrix = matrix.astype(working_dtype)
    tangent = tangent.astype(working_dtype)
    matrix = 0.5 * matrix + 0.5 * jnp.swapaxes(matrix, -1, -2)
    tangent = 0.5 * tangent + 0.5 * jnp.swapaxes(tangent, -1, -2)
    spectrum = HermitianSpectrum(
        matrix,
        tolerance=64.0 * jnp.finfo(working_dtype).eps,
    )
    eigenvalues = spectrum.eigenvalues
    eigenvectors = spectrum.eigenvectors
    positive = jnp.maximum(eigenvalues, 0.0)
    projected = (eigenvectors * positive[..., None, :]) @ jnp.swapaxes(
        eigenvectors, -1, -2
    )
    projected = 0.5 * projected + 0.5 * jnp.swapaxes(projected, -1, -2)
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
    derivative = 0.5 * derivative + 0.5 * jnp.swapaxes(derivative, -1, -2)
    return projected.astype(primals[0].dtype), derivative.astype(tangents[0].dtype)


class PositiveSemidefiniteCone(AbstractConvexCone):
    """Real PSD cone in Clarabel's scaled upper-column triangular coordinates."""

    matrix_size: int = eqx.field(static=True)
    _row_indices: tuple[int, ...] = eqx.field(static=True)
    _column_indices: tuple[int, ...] = eqx.field(static=True)
    _scales: tuple[float, ...] = eqx.field(static=True)

    def __init__(self, matrix_size: int, /):
        if isinstance(matrix_size, bool):
            raise TypeError("PositiveSemidefiniteCone matrix_size must be an integer.")
        size = index(matrix_size)
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
        symmetric = 0.5 * matrix + 0.5 * jnp.swapaxes(matrix, -1, -2)
        return symmetric[..., rows, columns] * scales

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
        finite = jnp.all(jnp.isfinite(value), axis=(-2, -1))
        scale = jnp.maximum(jnp.max(jnp.abs(value), axis=(-2, -1)), 1.0)
        residual = jnp.max(
            jnp.abs(value - jnp.swapaxes(value, -1, -2)),
            axis=(-2, -1),
        )
        tolerance = 64.0 * float(self.matrix_size) * jnp.finfo(value.dtype).eps * scale
        guarded = eqx.error_if(
            value,
            jnp.any(~finite | (residual > tolerance)),
            "PSD pack requires a finite symmetric matrix.",
        )
        return self._pack_unchecked(guarded)

    def unpack(self, value: Any, /) -> Array:
        """Unpack scaled upper-column coordinates into a real symmetric matrix."""

        array = self._validate(value)
        rows = jnp.asarray(self._row_indices, dtype=jnp.int32)
        columns = jnp.asarray(self._column_indices, dtype=jnp.int32)
        scales = jnp.asarray(self._scales, dtype=array.dtype)
        matrix = jnp.zeros(
            array.shape[:-1] + (self.matrix_size, self.matrix_size),
            dtype=array.dtype,
        )
        entries = array / scales
        matrix = matrix.at[..., rows, columns].set(entries)
        matrix = matrix.at[..., columns, rows].set(entries)
        return matrix

    def project(self, value: Any, /) -> Array:
        array = self._validate(value)
        array = eqx.error_if(
            array,
            jnp.any(~jnp.isfinite(array)),
            "PSD projection requires finite coordinates.",
        )
        return self._pack_unchecked(_positive_semidefinite_part(self.unpack(array)))

    def project_dual(self, value: Any, /) -> Array:
        return self.project(value)

    def interior_margin(self, value: Any, /) -> Array:
        array = self._validate(value)
        working = self.unpack(array).astype(jnp.result_type(array.dtype, jnp.float32))
        spectrum = HermitianSpectrum(
            working,
            tolerance=64.0 * jnp.finfo(working.dtype).eps,
        )
        return spectrum.minimum_eigenvalue.astype(array.dtype)

    def dual_projection_smoothness_margin(self, value: Any, /) -> Array:
        array = self._validate(value)
        working = self.unpack(array).astype(jnp.result_type(array.dtype, jnp.float32))
        spectrum = HermitianSpectrum(
            working,
            tolerance=64.0 * jnp.finfo(working.dtype).eps,
        )
        return jnp.min(jnp.abs(spectrum.eigenvalues), axis=-1).astype(array.dtype)


__all__ = ["PositiveSemidefiniteCone"]
