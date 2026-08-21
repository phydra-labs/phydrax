#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _termination(*, maximum_steps=40):
    return phx.optim.OptimizationTermination(
        absolute_optimality=1e-10,
        relative_optimality=0.0,
        maximum_steps=maximum_steps,
    )


def test_composite_problem_preserves_signed_scalar_semantics():
    problem = phx.optim.CompositeLeastSquaresProblem(
        lambda parameters, _: jnp.array([parameters[0] - 1.0]),
        lambda parameters, _: -0.25 * parameters[0] ** 2,
    )
    result = phx.optim.composite_least_squares(
        problem,
        jnp.array([0.0]),
        termination=_termination(),
    )

    np.testing.assert_allclose(result.parameters, jnp.array([2.0]), atol=1e-7)
    np.testing.assert_allclose(result.residual_objective, 0.5, atol=1e-7)
    np.testing.assert_allclose(result.scalar_objective, -1.0, atol=1e-7)
    np.testing.assert_allclose(result.objective, -0.5, atol=1e-7)
    assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)


def test_generalized_gauss_newton_matches_linear_quadratic_solution():
    design = jnp.array([[1.0, 2.0], [2.0, -1.0], [1.0, 1.0]])
    target = jnp.array([1.0, -2.0, 0.5])
    regularizer = jnp.array([[2.0, 0.25], [0.25, 1.0]])
    linear = jnp.array([0.5, -1.0])
    problem = phx.optim.CompositeLeastSquaresProblem(
        lambda parameters, _: design @ parameters - target,
        lambda parameters, _: (
            0.5 * parameters @ regularizer @ parameters - linear @ parameters
        ),
    )
    expected = jnp.linalg.solve(
        design.T @ design + regularizer,
        design.T @ target + linear,
    )

    result = phx.optim.composite_least_squares(
        problem,
        jnp.zeros(2),
        termination=_termination(),
    )

    np.testing.assert_allclose(result.parameters, expected, atol=2e-7)
    assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)
    assert int(result.diagnostics.hvp_evaluations) > 0
    assert int(result.diagnostics.jvp_evaluations) > 0
    assert int(result.diagnostics.vjp_evaluations) > 0


def test_composite_optimizer_retries_indefinite_model_then_falls_back():
    problem = phx.optim.CompositeLeastSquaresProblem(
        lambda parameters, _: jnp.array([parameters[0] - 1.0]),
        lambda parameters, _: -2.0 * parameters[0] ** 2 + 0.25 * parameters[0] ** 4,
    )
    method = phx.optim.GeneralizedGaussNewton(
        initial_damping=1e-8,
        damping_increase=10.0,
        maximum_trials=3,
    )
    initial = jnp.array([0.1])
    initial_objective = problem.objective(initial)
    result = phx.optim.composite_least_squares(
        problem,
        initial,
        method=method,
        termination=_termination(maximum_steps=60),
    )

    assert result.objective < initial_objective
    assert int(result.diagnostics.direction_fallbacks) > 0
    assert jnp.all(jnp.isfinite(result.parameters))


def test_composite_optimizer_compiles_with_dynamic_arguments():
    problem = phx.optim.CompositeLeastSquaresProblem(
        lambda parameters, target: parameters - target,
        lambda parameters, _: 0.5 * jnp.sum(parameters**2),
    )
    solve = eqx.filter_jit(
        lambda target: phx.optim.composite_least_squares(
            problem,
            jnp.zeros(2),
            args=target,
            termination=_termination(),
        )
    )

    result = solve(jnp.array([2.0, -4.0]))
    np.testing.assert_allclose(result.parameters, jnp.array([1.0, -2.0]), atol=1e-7)
    assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)


def test_composite_problem_rejects_non_scalar_auxiliary_objective():
    problem = phx.optim.CompositeLeastSquaresProblem(
        lambda parameters, _: parameters,
        lambda parameters, _: parameters,
    )

    try:
        problem.scalar_value(jnp.ones(2))
    except TypeError as error:
        assert "one real scalar" in str(error)
    else:
        raise AssertionError("A vector-valued scalar objective must be rejected.")
