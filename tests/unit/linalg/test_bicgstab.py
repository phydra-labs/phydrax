# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""BiCGStab convergence and genuine scalar-breakdown contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.linalg as la


def test_bicgstab_exact_half_step_convergence_skips_zero_omega_quotient() -> None:
    matrix = jnp.diag(jnp.asarray([4.0, 3.0], dtype=jnp.float64))
    rhs = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    result = la.solve(
        la.LinearSystem(la.DenseLinearOperator(matrix)),
        rhs,
        policy=la.LinearSolvePolicy(
            la.BiCGStab(),
            tolerance=la.TolerancePolicy(relative=1e-12, max_steps=10),
            preconditioning=la.PreconditioningPolicy(
                la.DiagonalPreconditioner(jnp.diag(matrix))
            ),
        ),
    )
    assert int(result.status) == int(la.LinearSolveStatus.SUCCESS)
    assert int(result.diagnostics.iterations) == 1
    assert int(result.diagnostics.matvec_count) == 4
    assert result.diagnostics.residual_norm == 0.0
    np.testing.assert_allclose(np.asarray(result.value), np.asarray([0.25, 2.0 / 3.0]))


def test_bicgstab_zero_rhs_needs_no_iteration() -> None:
    result = la.solve(
        la.LinearSystem(la.DenseLinearOperator(jnp.eye(2, dtype=jnp.float64))),
        jnp.zeros((2,), dtype=jnp.float64),
        policy=la.LinearSolvePolicy(la.BiCGStab()),
    )
    assert int(result.status) == int(la.LinearSolveStatus.SUCCESS)
    assert int(result.diagnostics.iterations) == 0
    assert int(result.diagnostics.matvec_count) == 2
    np.testing.assert_array_equal(np.asarray(result.value), np.zeros(2))


@pytest.mark.parametrize(
    "matrix,rhs,expected",
    [
        ([[0.0, 1.0], [-1.0, 0.0]], [1.0, 0.0], [0.0, 0.0]),
        ([[1.0, 1.0], [0.0, 0.0]], [1.0, 1.0], [1.0, 1.0]),
    ],
    ids=["alpha-denominator", "omega-denominator"],
)
def test_bicgstab_genuine_breakdown_keeps_a_finite_valid_iterate(
    matrix: list[list[float]], rhs: list[float], expected: list[float]
) -> None:
    result = la.solve(
        la.LinearSystem(la.DenseLinearOperator(jnp.asarray(matrix, dtype=jnp.float64))),
        jnp.asarray(rhs, dtype=jnp.float64),
        policy=la.LinearSolvePolicy(
            la.BiCGStab(), tolerance=la.TolerancePolicy(relative=1e-12, max_steps=10)
        ),
    )
    assert int(result.status) == int(la.LinearSolveStatus.BREAKDOWN)
    assert not bool(result.successful)
    assert bool(result.diagnostics.finite)
    assert result.diagnostics.residual_norm > 1e-12
    assert int(result.diagnostics.iterations) == 1
    np.testing.assert_allclose(np.asarray(result.value), expected)


def test_bicgstab_maximum_steps_does_not_report_convergence() -> None:
    result = la.solve(
        la.LinearSystem(
            la.DenseLinearOperator(
                jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
            )
        ),
        jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        policy=la.LinearSolvePolicy(
            la.BiCGStab(), tolerance=la.TolerancePolicy(relative=1e-12, max_steps=1)
        ),
    )
    assert int(result.status) == int(la.LinearSolveStatus.MAXIMUM_STEPS_REACHED)
    assert int(result.diagnostics.iterations) == 1
    assert bool(result.diagnostics.finite)
    assert not bool(result.successful)


def test_bicgstab_compiled_multiple_rhs_preserve_mathematical_derivatives() -> None:
    matrix = np.asarray([[2.0 + 1.0j, 1.0 - 2.0j], [-0.5 + 0.2j, 3.0 - 1.0j]])
    rhs = jnp.asarray([[1.0 + 2.0j, 0.0], [-0.5 + 1.0j, 0.0]], dtype=jnp.complex128)
    tangent = jnp.asarray([[0.2 - 0.1j, 1.0], [0.3 + 0.4j, -1.0j]], dtype=jnp.complex128)
    prepared = la.prepare(
        la.LinearSystem(la.DenseLinearOperator(jnp.asarray(matrix))),
        la.LinearSolvePolicy(
            la.BiCGStab(), tolerance=la.TolerancePolicy(relative=1e-12, max_steps=10)
        ),
    )
    result = jax.jit(lambda target: la.solve(prepared, target))(rhs)
    assert bool(jnp.all(result.successful))
    assert jnp.all(result.status == int(la.LinearSolveStatus.SUCCESS))
    np.testing.assert_allclose(
        np.asarray(result.value), np.linalg.solve(matrix, rhs), rtol=1e-12
    )
    _, derivative = jax.jvp(
        lambda target: la.solve(prepared, target).value, (rhs,), (tangent,)
    )
    np.testing.assert_allclose(
        np.asarray(derivative), np.linalg.solve(matrix, tangent), rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("max_steps,actions", [(None, 62), (4, 14)])
def test_bicgstab_plan_bounds_count_true_residual_confirmations(
    max_steps: int | None, actions: int
) -> None:
    problem = la.LinearSystem(la.DenseLinearOperator(jnp.eye(2, dtype=jnp.float64)))
    plan = la.plan(
        problem,
        la.LinearSolvePolicy(
            la.BiCGStab(), tolerance=la.TolerancePolicy(max_steps=max_steps)
        ),
    )
    assert plan.backend == "native-krylov"
    estimate = plan.candidates[-1]
    assert estimate.forward_actions_per_rhs == actions
    assert estimate.adjoint_actions_per_rhs == 0
