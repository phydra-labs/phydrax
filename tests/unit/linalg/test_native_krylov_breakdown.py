# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native Krylov near-invariant subspaces and scale invariance.

Oracles are host NumPy dense solves of the same explicit matrices.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.linalg as la


_SIZE = 24


def _shifted_laplacian() -> np.ndarray:
    # rho I - c nu L for a 1-D Dirichlet Laplacian: symmetric positive definite
    # with the known sine eigenvectors.
    laplacian = (
        np.diag(np.full(_SIZE, -2.0))
        + np.diag(np.ones(_SIZE - 1), 1)
        + np.diag(np.ones(_SIZE - 1), -1)
    )
    return np.eye(_SIZE) - 0.05 * laplacian


def _operator(matrix: np.ndarray) -> la.DenseLinearOperator:
    return la.DenseLinearOperator(
        jnp.asarray(matrix),
        properties=la.OperatorProperties(
            self_adjoint=True, evidence={"self_adjoint": "asserted"}
        ),
    )


_METHODS = {
    "gmres": la.GMRES(),
    "fgmres": la.FGMRES(),
    "minres": la.MINRES(),
}


@pytest.mark.parametrize("method", tuple(_METHODS), ids=tuple(_METHODS))
def test_near_eigenvector_rhs_restarts_instead_of_breaking_down(method: str) -> None:
    # After one Arnoldi step the subspace is invariant to ~1e-10 relative, far
    # below sqrt(eps), while the least-squares residual is still far above the
    # requested 1e-11: the cycle must restart from the true residual.
    matrix = _shifted_laplacian()
    index = np.arange(1, _SIZE + 1)
    eigenvector = np.sin(np.pi * index / (_SIZE + 1))
    eigenvector /= np.linalg.norm(eigenvector)
    rhs = eigenvector + 1e-10 * np.random.default_rng(5).standard_normal(_SIZE)
    result = la.solve(
        la.LinearSystem(_operator(matrix)),
        jnp.asarray(rhs),
        policy=la.LinearSolvePolicy(
            _METHODS[method],
            tolerance=la.TolerancePolicy(relative=1e-11, absolute=0.0, max_steps=200),
        ),
    )
    assert int(result.status) == int(la.LinearSolveStatus.SUCCESS)
    expected = np.linalg.solve(matrix, rhs)
    assert np.linalg.norm(
        rhs - matrix @ np.asarray(result.value)
    ) <= 1e-11 * np.linalg.norm(rhs)
    np.testing.assert_allclose(np.asarray(result.value), expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("method", tuple(_METHODS), ids=tuple(_METHODS))
def test_small_norm_operator_is_not_a_breakdown(method: str) -> None:
    # Scaling A and b by 1e-18 leaves the solution unchanged; no absolute floor
    # may turn the scaled Krylov basis into a breakdown.
    scale = 1e-18
    matrix = scale * _shifted_laplacian()
    unscaled_rhs = np.random.default_rng(9).standard_normal(_SIZE)
    result = la.solve(
        la.LinearSystem(_operator(matrix)),
        jnp.asarray(scale * unscaled_rhs),
        policy=la.LinearSolvePolicy(
            _METHODS[method],
            tolerance=la.TolerancePolicy(relative=1e-10, absolute=0.0, max_steps=200),
        ),
    )
    assert int(result.status) == int(la.LinearSolveStatus.SUCCESS)
    np.testing.assert_allclose(
        np.asarray(result.value),
        np.linalg.solve(_shifted_laplacian(), unscaled_rhs),
        rtol=1e-8,
    )


def test_exact_invariant_subspace_is_a_lucky_breakdown() -> None:
    # An exact eigenvector rhs gives an invariant one-dimensional Krylov space
    # whose least-squares solution is exact: success, not breakdown.
    matrix = np.diag(np.linspace(1.0, 3.0, _SIZE))
    rhs = np.zeros(_SIZE)
    rhs[4] = 2.0
    result = la.solve(
        la.LinearSystem(la.DenseLinearOperator(jnp.asarray(matrix))),
        jnp.asarray(rhs),
        policy=la.LinearSolvePolicy(
            la.GMRES(),
            tolerance=la.TolerancePolicy(relative=1e-12, absolute=0.0, max_steps=50),
        ),
    )
    assert int(result.status) == int(la.LinearSolveStatus.SUCCESS)
    assert int(result.diagnostics.iterations) == 1
    np.testing.assert_allclose(np.asarray(result.value), rhs / np.diag(matrix))
