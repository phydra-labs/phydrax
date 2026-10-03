# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Least-squares LSMR stops on the same stationarity criterion its status uses.

Oracles are host NumPy least-squares solves of the same explicit matrices.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.linalg as la


def _nearly_orthogonal_problem() -> tuple[np.ndarray, np.ndarray]:
    # Tall full-column-rank design with condition 1e2 and a right-hand side whose
    # range component is 1e-2 of its norm: ||A^T b|| << ||A|| ||b||, as in a
    # weak pressure projection of a nearly solenoidal field. A relative 1e-10 of
    # ||A^T b|| is then far below 1e-10 ||A|| ||r|| yet far above roundoff.
    rng = np.random.default_rng(17)
    left = np.linalg.qr(rng.standard_normal((200, 40)))[0]
    right = np.linalg.qr(rng.standard_normal((40, 40)))[0]
    matrix = left @ np.diag(np.logspace(0.0, -2.0, 40)) @ right.T
    orthogonal = rng.standard_normal(200)
    orthogonal -= left @ (left.T @ orthogonal)
    rhs = 1e-2 * (matrix @ rng.standard_normal(40)) + orthogonal / np.linalg.norm(
        orthogonal
    )
    return matrix, rhs


@pytest.mark.parametrize(
    "method", (la.LSMR(), la.GeneralizedLSMR()), ids=("lsmr", "glsmr")
)
def test_relative_normal_tolerance_is_met_when_rhs_is_nearly_orthogonal(
    method: la.AbstractLinearMethod,
) -> None:
    matrix, rhs = _nearly_orthogonal_problem()
    result = la.solve(
        la.LeastSquaresProblem(la.DenseLinearOperator(jnp.asarray(matrix))),
        jnp.asarray(rhs),
        policy=la.LinearSolvePolicy(
            method,
            tolerance=la.TolerancePolicy(relative=1e-10, absolute=0.0, max_steps=2000),
        ),
    )
    value = np.asarray(result.value)
    assert int(result.status) == int(la.LinearSolveStatus.SUCCESS)
    assert np.linalg.norm(matrix.T @ (rhs - matrix @ value)) <= 1e-10 * np.linalg.norm(
        matrix.T @ rhs
    )
    np.testing.assert_allclose(
        value, np.linalg.lstsq(matrix, rhs, rcond=None)[0], rtol=1e-6, atol=1e-12
    )


def test_unattainable_relative_tolerance_is_accepted_at_the_roundoff_floor() -> None:
    # 1e-15 of ||A^T b|| is below what any floating-point evaluation of A^T r can
    # resolve here; the stationary point at its roundoff floor is the solution,
    # not an exhausted iteration.
    matrix, rhs = _nearly_orthogonal_problem()
    result = la.solve(
        la.LeastSquaresProblem(la.DenseLinearOperator(jnp.asarray(matrix))),
        jnp.asarray(rhs),
        policy=la.LinearSolvePolicy(
            la.LSMR(),
            tolerance=la.TolerancePolicy(relative=1e-15, absolute=0.0, max_steps=2000),
        ),
    )
    value = np.asarray(result.value)
    residual = rhs - matrix @ value
    # The floor uses LSMR's running Frobenius estimate of ||A||, which may
    # exceed the true ||A||_F; allow that estimate a factor of four.
    floor = (
        4.0
        * np.sqrt(200.0)
        * np.finfo(np.float64).eps
        * np.linalg.norm(matrix)
        * np.linalg.norm(residual)
    )
    assert int(result.status) == int(la.LinearSolveStatus.SUCCESS)
    assert np.linalg.norm(matrix.T @ residual) <= 4.0 * floor
    np.testing.assert_allclose(
        value, np.linalg.lstsq(matrix, rhs, rcond=None)[0], rtol=1e-6, atol=1e-12
    )
