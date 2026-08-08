#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qpax

import phydrax as phx


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_qpax_implicit_matches_phydrax_kkt_contract(dtype):
    problem = phx.optim.QuadraticProgram(
        jnp.array([[4.0, 1.0], [1.0, 2.0]], dtype=dtype),
        jnp.array([-1.0, -1.0], dtype=dtype),
        equality_matrix=jnp.array([[1.0, 1.0]], dtype=dtype),
        equality_rhs=jnp.array([1.0], dtype=dtype),
        inequality_matrix=jnp.array([[-1.0, 0.0], [0.0, -1.0]], dtype=dtype),
        inequality_rhs=jnp.zeros(2, dtype=dtype),
    )
    tolerance = 2e-5 if dtype == jnp.float32 else 1e-7
    expected = phx.optim.solve_quadratic_program(
        problem,
        method="dense-primal-dual",
        tolerance=tolerance,
    )
    actual = phx.optim.solve_quadratic_program(
        problem,
        method="qpax-implicit",
        tolerance=tolerance,
    )
    np.testing.assert_allclose(
        actual.primal,
        expected.primal,
        atol=5e-4 if dtype == jnp.float32 else 2e-6,
        rtol=5e-4 if dtype == jnp.float32 else 2e-6,
    )
    assert actual.status == phx.optim.QP_SUCCESS
    assert actual.valid
    assert actual.backend_converged
    assert actual.backend == "qpax-0.1.4"
    assert actual.method == "qpax-implicit"
    assert actual.kkt_residual_norm <= tolerance


def test_qpax_implicit_public_primal_api_is_differentiable_and_batched():
    quadratic = jnp.broadcast_to(jnp.eye(2), (2, 2, 2))
    inequality_matrix = jnp.broadcast_to(-jnp.eye(2), (2, 2, 2))
    inequality_rhs = jnp.zeros((2, 2))

    def objective(linear):
        problem = phx.optim.QuadraticProgram(
            quadratic,
            linear,
            inequality_matrix=inequality_matrix,
            inequality_rhs=inequality_rhs,
        )
        primal = phx.optim.solve_quadratic_program_primal(
            problem,
            method="qpax-implicit",
            tolerance=1e-6,
        )
        return jnp.sum(primal)

    linear = jnp.array([[-2.0, -3.0], [-1.0, -4.0]])
    primal_gradient = jax.grad(objective)(linear)
    np.testing.assert_allclose(primal_gradient, -jnp.ones_like(linear), atol=2e-3)


@pytest.mark.parametrize(
    ("solver_name", "qpax_entry_point"),
    [
        ("solve_quadratic_program", "solve_qp"),
        ("solve_quadratic_program_primal", "solve_qp_primal"),
    ],
)
def test_qpax_implicit_rejects_nondefault_step_fraction_before_backend_call(
    monkeypatch, solver_name, qpax_entry_point
):
    called = False

    def unexpected_call(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("QPax must not be invoked for an unsupported option.")

    monkeypatch.setattr(qpax, qpax_entry_point, unexpected_call)
    problem = phx.optim.QuadraticProgram(jnp.eye(1), jnp.array([-1.0]))

    with pytest.raises(
        ValueError,
        match=(
            "method='qpax-implicit' does not support configurable step_fraction; "
            "QPax 0.1.4 fixes its fraction-to-boundary multiplier at 0.99"
        ),
    ):
        getattr(phx.optim, solver_name)(
            problem,
            method="qpax-implicit",
            step_fraction=0.9,
        )

    assert not called


def test_qpax_explicit_differentiation_is_rejected_without_importing_private_api():
    problem = phx.optim.QuadraticProgram(jnp.eye(1), jnp.array([-1.0]))
    with pytest.raises(ValueError, match="explicit differentiation"):
        phx.optim.solve_quadratic_program_primal(
            problem,
            method="qpax-explicit",
        )
