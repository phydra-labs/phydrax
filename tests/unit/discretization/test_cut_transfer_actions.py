import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from phydrax.discretization.amr._cut_transfer_actions import (
    CommutingAction,
    ComponentMinimumNorm,
    SparseAction,
)


@pytest.mark.parametrize("tolerance", [1.0e-9, 0.1])
def test_component_minimum_norm_matches_global_pinv(tolerance: float) -> None:
    matrix = sp.block_diag(
        (
            np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64),
            np.diag(np.array([0.2, 0.01], dtype=np.float64)),
        ),
        format="csr",
    )
    matrix = sp.csr_matrix(matrix)
    solve = ComponentMinimumNorm(matrix, tolerance, 1024 * 1024)
    dense = matrix.toarray()
    reference = np.linalg.pinv(dense, rcond=tolerance)
    action = np.asarray(jax.vmap(solve)(jnp.eye(4, dtype=jnp.float64)).T)
    np.testing.assert_allclose(action, reference, atol=1.0e-12)
    assert solve.rank == np.count_nonzero(
        np.linalg.svd(dense, compute_uv=False)
        > tolerance * np.linalg.svd(dense, compute_uv=False)[0]
    )
    np.testing.assert_allclose(
        np.eye(4) - action @ dense, np.eye(4) - reference @ dense, atol=1.0e-12
    )
    rhs = jnp.arange(4, dtype=jnp.float64)
    dual = jnp.arange(4, dtype=jnp.float64) + 0.5
    np.testing.assert_allclose(
        jnp.vdot(solve(rhs), dual),
        jnp.vdot(rhs, jax.linear_transpose(solve, rhs)(dual)[0]),
        atol=1.0e-12,
    )


def test_component_workspace_refuses_before_factorization() -> None:
    with pytest.raises(ValueError, match="component.*budget"):
        ComponentMinimumNorm(sp.csr_matrix(np.ones((4, 4), dtype=np.float64)), 1.0e-9, 16)


def test_commuting_action_preserves_geometric_nullspace() -> None:
    differential = sp.csr_matrix(np.array([[1.0, -1.0]], dtype=np.float64))
    geometric = sp.eye(2, format="csr", dtype=np.float64)
    upper = SparseAction(sp.csr_matrix(np.array([[2.0]], dtype=np.float64)))
    action = CommutingAction(
        upper,
        SparseAction(differential),
        SparseAction(differential),
        SparseAction(geometric),
        ComponentMinimumNorm(differential, 1.0e-9, 4096),
    )
    dense = differential.toarray()
    inverse = np.linalg.pinv(dense, rcond=1.0e-9)
    reference = inverse @ (2.0 * dense) + (np.eye(2) - inverse @ dense)
    value = jnp.array([3.0, 1.0], dtype=jnp.float64)
    np.testing.assert_allclose(action(value), reference @ np.asarray(value), atol=1.0e-12)
    np.testing.assert_allclose(
        action(jnp.ones(2, dtype=jnp.float64)), np.ones(2), atol=1.0e-12
    )
    assert not np.allclose(inverse @ (2.0 * dense) @ np.asarray(value), action(value))


def test_global_rank_cutoff_near_threshold() -> None:
    diagonal = np.array([100.0, 1.0 + 1.0e-8, 1.0 - 1.0e-8, 0.0], dtype=np.float64)
    matrix = sp.diags(diagonal, format="csr")
    solve = ComponentMinimumNorm(matrix, 0.01, 4096)
    assert solve.rank == 2
    np.testing.assert_allclose(
        solve(jnp.ones(4, dtype=jnp.float64)),
        np.linalg.pinv(matrix.toarray(), rcond=0.01) @ np.ones(4, dtype=np.float64),
        atol=1.0e-12,
    )
