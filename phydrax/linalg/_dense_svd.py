# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def economy_svd(
    matrix: Array,
    /,
    *,
    algorithm: jax.lax.linalg.SvdAlgorithm | None = None,
    hermitian: bool = False,
) -> tuple[Array, Array, Array]:
    """Economy factors without a long-dimension square CUDA scratch factor.

    At a 2:1 aspect ratio, reduced QR pays for itself by replacing the SVD's
    rectangular input with a min-dimension square. Near-square inputs retain
    the direct driver. Only the last two axes are matrix axes; conjugate
    transposes preserve the same contract for complex and batched inputs.
    """
    rows, columns = matrix.shape[-2:]
    rank = min(rows, columns)
    reduce = rank > 0 and max(rows, columns) >= 2 * rank and not hermitian
    if reduce:
        tall = matrix if rows >= columns else matrix.conj().swapaxes(-1, -2)
        basis, triangular = jnp.linalg.qr(tall, mode="reduced")
        value = triangular if rows >= columns else triangular.conj().swapaxes(-1, -2)
    else:
        value = matrix
    if hermitian:
        left, singular_values, right_adjoint = jnp.linalg.svd(
            value, full_matrices=False, hermitian=True
        )
    else:
        left, singular_values, right_adjoint = jax.lax.linalg.svd(
            value, full_matrices=False, algorithm=algorithm
        )
    if reduce:
        if rows >= columns:
            left = jnp.matmul(basis, left, precision=jax.lax.Precision.HIGHEST)
        else:
            right_adjoint = jnp.matmul(
                right_adjoint,
                basis.conj().swapaxes(-1, -2),
                precision=jax.lax.Precision.HIGHEST,
            )
    return left, singular_values, right_adjoint


def economy_svd_workspace_bytes(rows: int, columns: int, itemsize: int, /) -> int:
    """Conservative QR, square-driver, lifting, and alignment scratch bound.

    Reduction keeps every square scratch array at min(rows, columns); below
    the 2:1 crossover even a provider's full long-side factor is < 4 rank².
    Factors returned to the caller are charged separately as retained storage.
    """
    rank = min(rows, columns)
    return (4 * rows * columns + 8 * rank * rank + 4 * (rows + columns)) * itemsize + 4096
