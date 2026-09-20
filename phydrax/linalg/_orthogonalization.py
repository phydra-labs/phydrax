#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


class OrthonormalFrameResult(StrictModule):
    """Thin tangent frame, normal space, and numerical-rank evidence."""

    tangents: Array
    normal_basis: Array
    singular_values: Array
    rank: Array
    condition_estimate: Array
    regularity_margin: Array
    finite: Array
    successful: Array


def orthonormal_frame(matrix: ArrayLike, /) -> OrthonormalFrameResult:
    """Orthonormalize full-column-rank frames and retain failure evidence."""
    value = jnp.asarray(matrix)
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        value = value.astype(jnp.float64)
    if value.ndim < 2:
        raise ValueError("matrix must have trailing row and column axes.")
    ambient_dimension, intrinsic_dimension = value.shape[-2:]
    if not 0 < intrinsic_dimension < ambient_dimension:
        raise ValueError(
            "orthonormal_frame requires 0 < columns < rows on the trailing axes."
        )

    complete, triangular = jnp.linalg.qr(value, mode="complete")
    diagonal = jnp.diagonal(triangular[..., :intrinsic_dimension, :], axis1=-2, axis2=-1)
    diagonal_sign = jnp.where(diagonal < 0.0, -1.0, 1.0)
    tangents = complete[..., :, :intrinsic_dimension] * diagonal_sign[..., None, :]
    normal_basis = complete[..., :, intrinsic_dimension:]

    frame = jnp.concatenate((tangents, normal_basis), axis=-1)
    orientation = jnp.linalg.det(frame)
    first_normal_sign = jnp.where(orientation < 0.0, -1.0, 1.0)
    normal_basis = normal_basis.at[..., :, 0].multiply(first_normal_sign[..., None])

    singular_values = jnp.linalg.svd(value, full_matrices=False, compute_uv=False)
    largest = singular_values[..., 0]
    smallest = singular_values[..., -1]
    tolerance = (
        jnp.asarray(max(ambient_dimension, intrinsic_dimension), dtype=largest.dtype)
        * jnp.finfo(jnp.real(value).dtype).eps
        * jnp.maximum(largest, 1.0)
    )
    rank = jnp.sum(singular_values > tolerance[..., None], axis=-1, dtype=jnp.int32)
    finite = (
        jnp.all(jnp.isfinite(value), axis=(-2, -1))
        & jnp.all(jnp.isfinite(tangents), axis=(-2, -1))
        & jnp.all(jnp.isfinite(normal_basis), axis=(-2, -1))
        & jnp.all(jnp.isfinite(singular_values), axis=-1)
    )
    successful = finite & (rank == intrinsic_dimension)
    condition = largest / jnp.maximum(smallest, jnp.finfo(largest.dtype).tiny)
    return OrthonormalFrameResult(
        tangents=tangents,
        normal_basis=normal_basis,
        singular_values=singular_values,
        rank=rank,
        condition_estimate=condition,
        regularity_margin=smallest - tolerance,
        finite=finite,
        successful=successful,
    )


__all__ = ["OrthonormalFrameResult", "orthonormal_frame"]
