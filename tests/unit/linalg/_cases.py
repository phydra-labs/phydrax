from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

import phydrax.linalg as la


@dataclass(frozen=True, slots=True)
class OperatorCase:
    case_id: str
    build: Callable[[], la.AbstractLinearOperator]
    matrix: jax.Array
    primal: jax.Array
    target: jax.Array
    supports_diagonal: bool


def operator_cases() -> tuple[OperatorCase, ...]:
    space = la.ArraySpace((3,), dtype=jnp.float64)
    dense = jnp.asarray(
        [[2.0, 1.0, 0.0], [3.0, 4.0, 5.0], [0.0, 6.0, 7.0]],
        dtype=jnp.float64,
    )
    diagonal = jnp.diag(jnp.asarray([2.0, 4.0, 7.0], dtype=jnp.float64))
    permutation = jnp.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=jnp.float64,
    )
    triangular = jnp.asarray(
        [[2.0, 0.0, 0.0], [3.0, 4.0, 0.0], [1.0, 6.0, 7.0]],
        dtype=jnp.float64,
    )
    tridiagonal = jnp.asarray(
        [[2.0, 1.0, 0.0], [3.0, 4.0, 5.0], [0.0, 6.0, 7.0]],
        dtype=jnp.float64,
    )
    low_rank_left = jnp.asarray(
        [[1.0, 2.0], [0.0, 1.0], [2.0, -1.0]],
        dtype=jnp.float64,
    )
    low_rank_right = jnp.asarray(
        [[2.0, 0.0], [1.0, 3.0], [-1.0, 2.0]],
        dtype=jnp.float64,
    )
    symmetric_weights = jnp.asarray([2.0, -0.5], dtype=jnp.float64)
    rank_one_left = jnp.asarray([[1.0], [2.0], [-1.0]], dtype=jnp.float64)
    rank_one_right = jnp.asarray([[3.0], [-1.0], [2.0]], dtype=jnp.float64)
    primal = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)
    target = jnp.asarray([-0.25, 1.5, 2.0], dtype=jnp.float64)

    return (
        OperatorCase(
            "dense",
            lambda: la.DenseLinearOperator(dense, source=space, target=space),
            dense,
            primal,
            target,
            True,
        ),
        OperatorCase(
            "diagonal",
            lambda: la.DiagonalLinearOperator(jnp.diag(diagonal), space=space),
            diagonal,
            primal,
            target,
            True,
        ),
        OperatorCase(
            "identity",
            lambda: la.IdentityLinearOperator(space),
            jnp.eye(3, dtype=jnp.float64),
            primal,
            target,
            True,
        ),
        OperatorCase(
            "permutation",
            lambda: la.PermutationLinearOperator(
                jnp.asarray([0, 2, 1], dtype=jnp.int32),
                space=space,
            ),
            permutation,
            primal,
            target,
            True,
        ),
        OperatorCase(
            "triangular",
            lambda: la.TriangularLinearOperator(triangular, lower=True, space=space),
            triangular,
            primal,
            target,
            True,
        ),
        OperatorCase(
            "tridiagonal",
            lambda: la.TridiagonalLinearOperator(
                jnp.asarray([3.0, 6.0]),
                jnp.asarray([2.0, 4.0, 7.0]),
                jnp.asarray([1.0, 5.0]),
                space=space,
            ),
            tridiagonal,
            primal,
            target,
            True,
        ),
        OperatorCase(
            "banded",
            lambda: la.BandedLinearOperator(
                jnp.asarray([[0.0, 1.0, 5.0], [2.0, 4.0, 7.0], [3.0, 6.0, 0.0]]),
                lower_bandwidth=1,
                upper_bandwidth=1,
                space=space,
            ),
            tridiagonal,
            primal,
            target,
            True,
        ),
        OperatorCase(
            "low-rank",
            lambda: la.LowRankLinearOperator(
                low_rank_left,
                low_rank_right,
                source=space,
                target=space,
            ),
            low_rank_left @ low_rank_right.T,
            primal,
            target,
            True,
        ),
        OperatorCase(
            "symmetric-low-rank",
            lambda: la.SymmetricLowRankLinearOperator(
                low_rank_left,
                weights=symmetric_weights,
                space=space,
            ),
            (low_rank_left * symmetric_weights[None, :]) @ low_rank_left.T,
            primal,
            target,
            True,
        ),
        OperatorCase(
            "diagonal-plus-low-rank",
            lambda: la.DiagonalPlusLowRankLinearOperator(
                jnp.diag(diagonal),
                rank_one_left,
                rank_one_right,
                space=space,
            ),
            diagonal + rank_one_left @ rank_one_right.T,
            primal,
            target,
            True,
        ),
    )
