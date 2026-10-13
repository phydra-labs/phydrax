"""Exact local-QR least squares for diagonal sample and adjacent coupling blocks."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from ._banded_spectrum import (
    banded_singular_value_brackets,
    banded_singular_value_certificate,
)
from ._results import LinearSolveStatus
from ._svd_contracts import SVDRankEvidence


class BlockBandedLeastSquaresResult(StrictModule):
    value: Array
    minimum_singular_value: Array
    maximum_singular_value: Array
    condition_number: Array
    rank_evidence: SVDRankEvidence
    residual_norm: Array
    finite: Array
    successful: Array
    status: Array


def _qr_chain(
    samples: Array,
    targets: Array,
    lower: Array,
    upper: Array,
    coupling_target: Array,
    regularization: float,
) -> tuple[Array, Array, Array]:
    width = samples.shape[-1]
    identity = jnp.eye(width, dtype=samples.dtype)
    zero = jnp.zeros((width, width), dtype=samples.dtype)
    lower = jnp.concatenate((lower, zero[None]), axis=0)
    upper = jnp.concatenate((upper, zero[None]), axis=0)
    coupling_target = jnp.concatenate(
        (coupling_target, jnp.zeros((1, width), dtype=samples.dtype)), axis=0
    )

    def eliminate(
        carry: tuple[Array, Array], row: tuple[Array, Array, Array, Array, Array]
    ) -> tuple[tuple[Array, Array], tuple[Array, Array, Array]]:
        previous, previous_target = carry
        sample, target, left, right, coupling = row
        current = jnp.concatenate(
            (
                previous,
                sample,
                jnp.sqrt(jnp.asarray(regularization, dtype=samples.dtype)) * identity,
                left,
            ),
            axis=0,
        )
        following = jnp.concatenate(
            (jnp.zeros((2 * width + sample.shape[0], width), dtype=samples.dtype), right),
            axis=0,
        )
        q, r = jnp.linalg.qr(
            jnp.concatenate((current, following), axis=1), mode="reduced"
        )
        transformed = q.T.conj() @ jnp.concatenate(
            (previous_target, target, jnp.zeros((width,), dtype=samples.dtype), coupling)
        )
        return (r[width:, width:], transformed[width:]), (
            r[:width, :width],
            r[:width, width:],
            transformed[:width],
        )

    _, (diagonal, off_diagonal, rhs) = jax.lax.scan(
        eliminate,
        (zero, jnp.zeros((width,), dtype=samples.dtype)),
        (samples, targets, lower, upper, coupling_target),
    )
    return diagonal, off_diagonal, rhs


def _back_substitute(diagonal: Array, off_diagonal: Array, rhs: Array) -> Array:
    width = diagonal.shape[-1]

    def substitute(
        following: Array, row: tuple[Array, Array, Array]
    ) -> tuple[Array, Array]:
        diagonal_row, upper_row, rhs_row = row
        value = jsp_linalg.solve_triangular(
            diagonal_row, rhs_row - upper_row @ following, lower=False
        )
        return value, value

    _, values = jax.lax.scan(
        substitute,
        jnp.zeros((width,), dtype=diagonal.dtype),
        (diagonal[::-1], off_diagonal[::-1], rhs[::-1]),
    )
    return values[::-1]


def _spectrum(
    diagonal: Array, upper: Array, rows: int
) -> tuple[Array, Array, Array, SVDRankEvidence]:
    count, width, _ = diagonal.shape
    bandwidth = 2 * width - 1
    row = jnp.arange(width)[:, None]
    column = jnp.arange(width)[None, :]
    base = jnp.arange(count)[:, None, None] * width
    bands = jnp.zeros((2 * width, count * width), dtype=diagonal.dtype)
    bands = bands.at[
        jnp.broadcast_to(jnp.maximum(bandwidth + row - column, 0), diagonal.shape),
        jnp.broadcast_to(base + column, diagonal.shape),
    ].add(jnp.where(row <= column, diagonal, 0.0))
    bands = bands.at[
        jnp.broadcast_to(bandwidth - width + row - column, upper[:-1].shape),
        jnp.broadcast_to(base[1:] + column, upper[:-1].shape),
    ].add(upper[:-1])
    brackets = banded_singular_value_brackets(
        bands,
        rows=count * width,
        lower_bandwidth=0,
        upper_bandwidth=bandwidth,
        relative_tolerance=1e-10,
    )
    minimum = (brackets.minimum_lower + brackets.minimum_upper) * 0.5
    maximum = (brackets.maximum_lower + brackets.maximum_upper) * 0.5
    cutoff = (
        jnp.finfo(bands.dtype).eps
        * max(rows, count * width)
        * jnp.maximum(brackets.maximum_upper, 1.0)
    )
    # A certified lower enclosure already proves the same strict rank threshold.
    # Only unresolved brackets need another interval Gram/Cholesky certificate.
    certified = jax.lax.cond(
        brackets.minimum_lower > cutoff,
        lambda: jnp.asarray(True),
        lambda: (
            banded_singular_value_certificate(
                bands,
                rows=count * width,
                lower_bandwidth=0,
                upper_bandwidth=bandwidth,
                threshold=cutoff,
            ).certified
        ),
    )
    finite = jnp.all(jnp.isfinite(bands)) & jnp.isfinite(minimum) & jnp.isfinite(maximum)
    rank = SVDRankEvidence(
        jnp.where(certified, count * width, 0),
        jnp.asarray(count * width, dtype=jnp.int32),
        cutoff,
        cutoff,
        finite,
        True,
        certified,
        "exact-spectrum",
        jnp.zeros((), dtype=bands.dtype),
    )
    return minimum, maximum, brackets.condition_upper_bound, rank


def solve_block_banded_least_squares(
    samples: ArrayLike,
    targets: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    coupling_targets: ArrayLike,
    /,
    *,
    regularization: float = 0.0,
) -> BlockBandedLeastSquaresResult:
    """Minimize local samples plus adjacent coupling residuals and λ‖x‖².

    No global design, normal-equation solve, or dense spectral object is formed.
    Numerical rank deficiency refuses explicitly; use ``dense-svd`` for truncated
    minimum-norm semantics. Diagnostic spectrum always excludes regularization.
    """
    a = jnp.asarray(samples)
    b, left, right, coupling = (
        jnp.asarray(value, dtype=a.dtype)
        for value in (targets, lower, upper, coupling_targets)
    )
    if a.ndim != 3 or not jnp.issubdtype(a.dtype, jnp.floating):
        raise ValueError("samples must be a real floating (blocks, rows, columns) array.")
    count, rows, width = a.shape
    if (
        count < 1
        or width < 1
        or rows < width
        or b.shape != (count, rows)
        or left.shape != (count - 1, width, width)
        or right.shape != left.shape
        or coupling.shape != (count - 1, width)
    ):
        raise ValueError("Block-banded least-squares shapes are incompatible.")
    if not isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and nonnegative.")
    diagonal, off_diagonal, rhs = _qr_chain(a, b, left, right, coupling, regularization)
    value = _back_substitute(diagonal, off_diagonal, rhs)
    if regularization > 0.0:
        diagonal, off_diagonal, _ = _qr_chain(a, b, left, right, coupling, 0.0)
    minimum, maximum, condition, rank = _spectrum(
        diagonal, off_diagonal, count * rows + (count - 1) * width
    )
    full_rank = rank.lower_bound == count * width
    value = eqx.error_if(
        value,
        rank.available & ~full_rank,
        "Block-banded least squares cannot certify numerical full rank; select explicit dense-svd for the minimum-norm solve.",
    )
    residual = jax.vmap(lambda block, x: block @ x)(a, value) - b
    continuity = (
        jax.vmap(lambda l, r, x, y: l @ x + r @ y)(left, right, value[:-1], value[1:])
        - coupling
    )
    residual_norm = jnp.sqrt(jnp.sum(residual**2) + jnp.sum(continuity**2))
    finite = jnp.all(jnp.isfinite(value)) & jnp.isfinite(residual_norm) & rank.available
    successful = finite & full_rank
    status = jnp.where(
        successful,
        int(LinearSolveStatus.SUCCESS),
        int(LinearSolveStatus.NONFINITE_OUTPUT),
    )
    return BlockBandedLeastSquaresResult(
        value,
        minimum,
        maximum,
        condition,
        rank,
        residual_norm,
        finite,
        successful,
        status,
    )


__all__ = ["BlockBandedLeastSquaresResult", "solve_block_banded_least_squares"]
