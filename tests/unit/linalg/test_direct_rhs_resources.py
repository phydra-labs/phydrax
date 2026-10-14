"""Charge derivative-only Krylov storage when differentiation executes."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.linalg as la


@pytest.mark.parametrize("planned", [False, True])
def test_direct_primal_defers_derivative_basis_budget(planned: bool) -> None:
    with jax.enable_x64():
        matrix = jnp.asarray([[2.0, 0.1], [0.1, 3.0]], dtype=jnp.float64)
        problem = la.LinearSystem(la.DenseLinearOperator(matrix))
        # One derivative RHS fits; two do not. A direct primal needs no basis.
        policy = la.LinearSolvePolicy(
            la.DenseLU(), resources=la.SolveResourcePolicy(krylov_basis_bytes=128)
        )
        selected = (
            la.plan(problem, policy, rhs_layout=la.RHSLayout((2,))) if planned else policy
        )
        rhs = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64)

        def solve(value: jax.Array) -> jax.Array:
            return la.solve(
                problem, value, policy=selected, rhs_layout=la.RHSLayout((2,))
            ).value

        np.testing.assert_allclose(
            solve(rhs), np.linalg.solve(np.asarray(matrix), np.asarray(rhs)), atol=2e-12
        )
        for compiled in (False, True):
            action = jax.jit(solve) if compiled else solve
            np.testing.assert_allclose(action(rhs), solve(rhs), atol=2e-12)
            with pytest.raises(
                (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
                match="Implicit derivative requires",
            ):
                jax.block_until_ready(jax.jvp(action, (rhs,), (jnp.ones_like(rhs),))[1])
            _, pullback = jax.vjp(action, rhs)
            with pytest.raises(
                (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
                match="Implicit derivative requires",
            ):
                jax.block_until_ready(pullback(jnp.ones_like(rhs)))


def test_direct_primal_still_enforces_actual_rhs_workspace() -> None:
    with jax.enable_x64():
        problem = la.LinearSystem(la.DenseLinearOperator(jnp.eye(2, dtype=jnp.float64)))
        policy = la.LinearSolvePolicy(
            la.DenseLU(), resources=la.SolveResourcePolicy(workspace_bytes=64)
        )
        with pytest.raises(ValueError, match="workspace"):
            la.solve(
                problem,
                jnp.ones((2, 3), dtype=jnp.float64),
                policy=policy,
                rhs_layout=la.RHSLayout((3,)),
            )


def test_direct_plan_retains_single_derivative_rhs_reservation() -> None:
    with jax.enable_x64():
        problem = la.LinearSystem(la.DenseLinearOperator(jnp.eye(2, dtype=jnp.float64)))
        policy = la.LinearSolvePolicy(
            la.DenseLU(), resources=la.SolveResourcePolicy(krylov_basis_bytes=127)
        )
        with pytest.raises(ValueError, match="Krylov basis"):
            la.plan(problem, policy, rhs_layout=la.RHSLayout((2,)))
