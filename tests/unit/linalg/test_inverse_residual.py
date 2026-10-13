from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.linalg import (
    DenseLinearOperator,
    inverse_residual_certificate,
    LinearSolveStatus,
)


def _inverse_action(inverse: Array) -> Callable[[Array], tuple[Array, Array]]:
    def apply(columns: Array) -> tuple[Array, Array]:
        return inverse @ columns, jnp.ones((columns.shape[1],), dtype=jnp.bool_)

    return apply


@pytest.mark.parametrize(
    "block_size", [1, 2, 4], ids=["scalar", "partial-final", "one-block"]
)
def test_inverse_residual_matches_independent_dense_reference(block_size: int) -> None:
    matrix = np.asarray(
        [[4.0, -1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 1.0, 2.0]], dtype=np.float64
    )
    inverse = 0.99 * np.linalg.inv(matrix)
    certificate = inverse_residual_certificate(
        DenseLinearOperator(jnp.asarray(matrix)),
        _inverse_action(jnp.asarray(inverse)),
        block_size=block_size,
    )
    error = np.linalg.norm(np.eye(3) - matrix @ inverse, ord=np.inf)
    condition = (
        3
        * np.linalg.norm(matrix, ord=np.inf)
        * np.linalg.norm(inverse, ord=np.inf)
        / (1.0 - error)
    )
    assert bool(certificate.valid)
    assert int(certificate.status) == int(LinearSolveStatus.SUCCESS)
    assert int(certificate.rank) == 3
    np.testing.assert_allclose(certificate.residual_upper_bound, error, atol=1e-13)
    np.testing.assert_allclose(certificate.condition_upper_bound, condition, rtol=1e-12)
    assert certificate.condition_upper_bound >= np.linalg.cond(matrix)
    assert certificate.evidence == "verified"


@pytest.mark.parametrize("diagonal", [0.0, 1e-14], ids=["singular", "ill-conditioned"])
def test_inverse_certificate_refuses_unresolved_directions(diagonal: float) -> None:
    matrix = np.diag(np.asarray([1.0, diagonal], dtype=np.float64))
    inverse = np.diag(
        np.asarray([1.0, 0.0 if diagonal == 0.0 else 1.0 / diagonal], dtype=np.float64)
    )
    result = inverse_residual_certificate(
        DenseLinearOperator(jnp.asarray(matrix)),
        _inverse_action(jnp.asarray(inverse)),
        block_size=1,
    )
    assert not bool(result.valid)
    assert int(result.status) != int(LinearSolveStatus.SUCCESS)
    assert int(result.rank) == -1
    assert bool(jnp.isinf(result.condition_upper_bound))


def test_inverse_certificate_detects_dropped_column_block() -> None:
    matrix = jnp.diag(jnp.asarray([2.0, 3.0, 4.0], dtype=jnp.float64))
    inverse = jnp.diag(jnp.asarray([0.5, 1.0 / 3.0, 0.25], dtype=jnp.float64))

    def dropped(columns: Array) -> tuple[Array, Array]:
        result = inverse @ columns
        return result.at[2].set(0.0), jnp.asarray(True)

    result = inverse_residual_certificate(
        DenseLinearOperator(matrix), dropped, block_size=2
    )
    assert not bool(result.valid)
    assert result.residual_upper_bound >= 1.0


def test_inverse_certificate_preserves_action_failure_evidence() -> None:
    def failed(columns: Array) -> tuple[Array, Array]:
        return columns, jnp.asarray(False)

    result = inverse_residual_certificate(
        DenseLinearOperator(jnp.eye(3, dtype=jnp.float64)), failed
    )
    assert not bool(result.valid)
    assert int(result.status) == int(LinearSolveStatus.RESIDUAL_TOO_LARGE)


def test_inverse_certificate_compiles_without_dense_output() -> None:
    operator = DenseLinearOperator(
        jnp.diag(jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64))
    )
    certificate = jax.jit(
        lambda values: inverse_residual_certificate(
            operator, _inverse_action(values), block_size=2
        )
    )(jnp.diag(jnp.asarray([1.0, 0.5, 1.0 / 3.0], dtype=jnp.float64)))
    assert bool(certificate.valid)
    assert all(leaf.ndim == 0 for leaf in jax.tree.leaves(certificate))
