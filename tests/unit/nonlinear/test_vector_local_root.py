"""Vector local roots expose transposable IFT actions and actual Newton counts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.nonlinear import VectorLocalRootPlan


pytestmark = pytest.mark.strict_jax


def test_vector_root_implicit_forward_reverse_matches_analytic_derivative() -> None:
    plan = VectorLocalRootPlan(2, tolerance=1e-12, plan_id="vector-implicit-transpose")
    known_root = np.asarray((0.7, 1.1), dtype=np.float64)
    target = jnp.asarray(known_root + known_root**3 + 0.1 * known_root[::-1])

    def solve(parameters: Array) -> Array:
        def residual(root: Array) -> Array:
            return root + root**3 + 0.1 * root[::-1] - parameters

        return plan.solve(residual, jnp.full((2,), 0.5, dtype=parameters.dtype))

    expected = np.linalg.solve(
        np.diag(1.0 + 3.0 * known_root**2) + np.asarray(((0.0, 0.1), (0.1, 0.0))),
        np.eye(2, dtype=np.float64),
    )
    np.testing.assert_allclose(jax.jit(solve)(target), known_root, rtol=1e-12, atol=1e-12)
    forward = jax.jit(jax.jacfwd(solve))(target)
    reverse = jax.jit(jax.jacrev(solve))(target)
    np.testing.assert_allclose(forward, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(reverse, expected, rtol=1e-11, atol=1e-12)
    gradient = jax.jit(jax.grad(lambda p: jnp.sum(solve(p))))(target)
    np.testing.assert_allclose(gradient, np.sum(expected, axis=0), rtol=1e-11, atol=1e-12)
    batch = jnp.stack((target, target))
    batched = jax.jit(jax.vmap(jax.jacrev(solve)))(batch)
    np.testing.assert_allclose(
        batched, np.broadcast_to(expected, (2, 2, 2)), rtol=1e-11, atol=1e-12
    )


@pytest.mark.parametrize(
    "already_converged", (False, True), ids=("one-step", "zero-steps")
)
def test_vector_root_diagnostics_count_actual_updates(already_converged: bool) -> None:
    plan = VectorLocalRootPlan(
        2, maximum_steps=12, tolerance=1e-12, plan_id="vector-iterations"
    )
    target = jnp.asarray((1.0, 2.0), dtype=jnp.float64)
    initial = target if already_converged else jnp.zeros_like(target)
    root, diagnostics = plan.solve_with_diagnostics(lambda value: value - target, initial)
    np.testing.assert_allclose(root, target, atol=1e-12)
    assert bool(diagnostics.converged)
    assert int(diagnostics.iterations) == (0 if already_converged else 1)


def test_vector_root_failure_diagnostics_retain_initial_state() -> None:
    plan = VectorLocalRootPlan(2, maximum_steps=3, plan_id="vector-singular")
    initial = jnp.asarray((0.2, 0.3), dtype=jnp.float64)
    root, diagnostics = plan.solve_with_diagnostics(
        lambda value: jnp.ones_like(value), initial
    )
    np.testing.assert_array_equal(root, initial)
    assert not bool(diagnostics.converged)
    assert int(diagnostics.iterations) == 0
