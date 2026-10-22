# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native Krylov near-invariant subspaces and scale invariance.

Oracles are host NumPy dense solves of the same explicit matrices.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.linalg as la
from phydrax.linalg._runtime import _run_callable_gmres
from phydrax.linalg.backends._native_krylov import _fgmres_raw, KrylovLoopDriver
from phydrax.linalg.krylov._results import KrylovBreakdownStatus


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


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("complex_action", [False, True])
def test_callable_gmres_eigendirection_and_zero_columns(
    compiled: bool, complex_action: bool
) -> None:
    matrix = 4.0 * np.eye(16) + np.full((16, 16), 0.05)
    if complex_action:
        matrix = matrix * (1.0 + 0.5j)
    # Include primal, transpose, and Hermitian-adjoint actions explicitly; a
    # complex coordinate transpose is not the Hermitian adjoint.
    for operator in (matrix, matrix.T, matrix.conj().T):
        array = jnp.asarray(operator)
        right = jnp.asarray(
            np.stack((np.ones(16), np.zeros(16), np.arange(16) + 1), axis=1),
            dtype=array.dtype,
        )

        def solve_column(rhs: jax.Array) -> tuple[jax.Array, ...]:
            result = _run_callable_gmres(
                lambda vector: array @ vector,
                rhs,
                max_steps=16,
                restart=16,
                stagnation_iterations=17,
                relative=1e-12,
                absolute=0.0,
            )
            return (
                result.value,
                result.residual_norm,
                result.iterations,
                result.breakdown,
                result.valid,
            )

        solve_columns = jax.vmap(solve_column, in_axes=1, out_axes=0)
        if compiled:
            solve_columns = jax.jit(solve_columns)
        values, residuals, iterations, breakdowns, valid = solve_columns(right)
        expected = np.linalg.solve(operator, np.asarray(right))
        assert np.all(np.asarray(valid))
        assert int(iterations[0]) == 1
        assert int(iterations[1]) == 0
        assert int(breakdowns[0]) == int(KrylovBreakdownStatus.HAPPY)
        np.testing.assert_allclose(values.T, expected, rtol=1e-11, atol=1e-13)
        true_residuals = np.linalg.norm(np.asarray(right) - operator @ values.T, axis=0)
        assert np.all(true_residuals <= 1e-12 * np.linalg.norm(right, axis=0))
        np.testing.assert_allclose(residuals, true_residuals, atol=1e-13)


@pytest.mark.parametrize("driver", ["early-exit", "fixed-trip"])
def test_gmres_rank_loss_keeps_finite_partial_update_and_failure_evidence(
    driver: KrylovLoopDriver,
) -> None:
    # A nonzero RHS in the nullspace has a zero Hessenberg column. Its trailing
    # reduced RHS is also zero, which is not a convergence certificate.
    rhs = jnp.asarray([0.0, 1.0])

    def run(right: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        value, auxiliary, _ = _fgmres_raw(
            lambda vector: jnp.asarray([vector[0], 0.0]),
            right,
            jnp.zeros_like(right),
            jnp.vdot,
            lambda vector, _: vector,
            4,
            2,
            5,
            jnp.asarray(1e-12),
            jnp.asarray(0.0),
            driver=driver,
            identity_preconditioner=True,
        )
        return value, auxiliary

    value, auxiliary = jax.jit(run)(rhs)
    np.testing.assert_array_equal(value, jnp.zeros_like(rhs))
    assert int(auxiliary[0]) == 1
    assert float(auxiliary[1]) == 1.0
    assert int(auxiliary[4]) == int(KrylovBreakdownStatus.NEAR_BREAKDOWN)


def test_callable_gmres_iteration_exhaustion_is_not_happy_breakdown() -> None:
    matrix = jnp.diag(jnp.asarray([1.0, 2.0, 4.0]))
    rhs = jnp.ones(3, dtype=jnp.float64)
    result = _run_callable_gmres(
        lambda vector: matrix @ vector,
        rhs,
        max_steps=1,
        restart=1,
        stagnation_iterations=2,
        relative=1e-12,
        absolute=0.0,
    )
    assert np.all(np.isfinite(result.value))
    assert not bool(result.valid)
    assert int(result.iterations) == 1
    assert int(result.breakdown) == int(KrylovBreakdownStatus.NONE)
    assert float(result.residual_norm) > 1e-12 * np.linalg.norm(rhs)


def test_fixed_trip_happy_breakdown_has_finite_jvp_and_vjp() -> None:
    rhs = jnp.ones(4, dtype=jnp.float64)

    def solution(scale: jax.Array) -> jax.Array:
        value, _, _ = _fgmres_raw(
            lambda vector: scale * vector,
            rhs,
            jnp.zeros_like(rhs),
            jnp.vdot,
            lambda vector, _: vector,
            4,
            4,
            5,
            jnp.asarray(1e-12),
            jnp.asarray(0.0),
            driver="fixed-trip",
            identity_preconditioner=True,
        )
        return value

    selected = jax.jit(solution)
    scale = jnp.asarray(4.0)
    value, tangent = jax.jvp(selected, (scale,), (jnp.asarray(1.0),))
    np.testing.assert_allclose(value, np.full(4, 0.25), rtol=1e-12)
    np.testing.assert_allclose(tangent, np.full(4, -0.0625), rtol=1e-12)
    _, pullback = jax.vjp(selected, scale)
    for cotangent, expected in ((rhs, -0.25), (jnp.zeros_like(rhs), 0.0)):
        (gradient,) = pullback(cotangent)
        np.testing.assert_allclose(gradient, expected, rtol=1e-12, atol=0.0)
