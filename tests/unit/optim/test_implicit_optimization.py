#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest

import phydrax as phx


def _termination(*, maximum_steps=60):
    return phx.optim.OptimizationTermination(
        absolute_optimality=1e-10,
        relative_optimality=0.0,
        maximum_steps=maximum_steps,
    )


def test_implicit_minimize_quadratic_jacobian_matches_inverse_hessian():
    hessian = jnp.array([[4.0, 1.0], [1.0, 3.0]])

    def solution(linear):
        return phx.optim.implicit_minimize(
            lambda parameters, rhs: (
                0.5 * parameters @ hessian @ parameters - rhs @ parameters
            ),
            jnp.zeros(2),
            args=linear,
            termination=_termination(),
        )

    linear = jnp.array([1.0, -2.0])
    expected = jnp.linalg.solve(hessian, linear)
    np.testing.assert_allclose(solution(linear), expected, atol=1e-8)
    np.testing.assert_allclose(
        jax.jacrev(solution)(linear),
        jnp.linalg.inv(hessian),
        atol=1e-7,
    )


def test_implicit_minimize_nonlinear_solution_uses_stationarity_derivative():
    def solution(parameter):
        return phx.optim.implicit_minimize(
            lambda state, value: 0.5 * (state[0] ** 2 - value) ** 2,
            jnp.array([1.5]),
            args=parameter,
            termination=_termination(),
        )[0]

    parameter = jnp.array(2.25)
    expected = jnp.sqrt(parameter)
    np.testing.assert_allclose(solution(parameter), expected, atol=1e-8)
    np.testing.assert_allclose(
        jax.grad(solution)(parameter),
        0.5 / expected,
        atol=2e-7,
    )


def test_implicit_least_squares_differentiates_nonzero_residual_solution():
    def solution(parameter):
        return phx.optim.implicit_least_squares(
            lambda state, value: jnp.array([state[0] - value, 2.0 * state[0] + 1.0]),
            jnp.array([0.0]),
            args=parameter,
            termination=_termination(),
        )[0]

    parameter = jnp.array(3.0)
    expected = (parameter - 2.0) / 5.0
    np.testing.assert_allclose(solution(parameter), expected, atol=1e-8)
    np.testing.assert_allclose(jax.grad(solution)(parameter), 0.2, atol=1e-7)


def test_implicit_least_squares_exact_fit_has_exact_sensitivity():
    design = jnp.array([[1.0, 2.0], [2.0, -1.0], [1.0, 1.0]])

    def solution(target):
        return phx.optim.implicit_least_squares(
            lambda parameters, data: design @ parameters - data,
            jnp.zeros(2),
            args=target,
            termination=_termination(),
        )

    parameters = jnp.array([1.5, -0.25])
    target = design @ parameters
    expected_jacobian = jnp.linalg.solve(design.T @ design, design.T)
    np.testing.assert_allclose(solution(target), parameters, atol=1e-8)
    np.testing.assert_allclose(
        jax.jacrev(solution)(target),
        expected_jacobian,
        atol=2e-7,
    )


def test_implicit_minimize_supports_nested_parameters_and_dynamic_args():
    initial = {"position": jnp.array([0.0, 0.0]), "scale": (jnp.array(0.0),)}

    def objective(parameters, target):
        return (
            jnp.sum((parameters["position"] - target["position"]) ** 2)
            + (parameters["scale"][0] - target["scale"]) ** 2
        )

    def summed_solution(target):
        solution = phx.optim.implicit_minimize(
            objective,
            initial,
            args=target,
            termination=_termination(),
        )
        return jnp.sum(solution["position"]) + solution["scale"][0]

    target = {"position": jnp.array([1.0, -2.0]), "scale": jnp.array(3.0)}
    tangent = {"position": jnp.array([0.5, 0.25]), "scale": jnp.array(-1.0)}
    value, derivative = jax.jvp(summed_solution, (target,), (tangent,))

    np.testing.assert_allclose(value, 2.0, atol=1e-8)
    np.testing.assert_allclose(derivative, -0.25, atol=1e-8)


def test_implicit_apis_compose_with_jit_jvp_and_vmap():
    def solution(parameter):
        return phx.optim.implicit_minimize(
            lambda state, target: jnp.sum((state - target) ** 2),
            jnp.array([0.0]),
            args=parameter,
            termination=_termination(),
        )[0]

    compiled = eqx.filter_jit(solution)
    primal, tangent = jax.jvp(compiled, (jnp.array(2.0),), (jnp.array(1.0),))
    batched = eqx.filter_jit(jax.vmap(solution))(jnp.array([1.0, 2.0, 3.0]))

    np.testing.assert_allclose(primal, 2.0, atol=1e-8)
    np.testing.assert_allclose(tangent, 1.0, atol=1e-8)
    np.testing.assert_allclose(batched, jnp.array([1.0, 2.0, 3.0]), atol=1e-8)


def test_initial_guess_has_zero_implicit_sensitivity_on_fixed_branch():
    def solution(initial):
        return phx.optim.implicit_minimize(
            lambda state, target: jnp.sum((state - target) ** 2),
            initial,
            args=jnp.array([2.0]),
            termination=_termination(),
        )[0]

    np.testing.assert_allclose(
        jax.grad(solution)(jnp.array([-3.0])),
        jnp.array([0.0]),
        atol=1e-12,
    )


def test_implicit_derivative_matches_centered_finite_difference():
    def solution(parameter):
        return phx.optim.implicit_minimize(
            lambda state, value: 0.5 * (state[0] ** 2 - value) ** 2,
            jnp.array([1.2]),
            args=parameter,
            termination=_termination(),
        )[0]

    parameter = jnp.array(1.7)
    step = 1e-4
    finite_difference = (solution(parameter + step) - solution(parameter - step)) / (
        2.0 * step
    )
    np.testing.assert_allclose(
        jax.grad(solution)(parameter),
        finite_difference,
        rtol=2e-5,
        atol=2e-6,
    )


def test_native_implicit_derivative_agrees_with_optimistix():
    def objective(state, target):
        return jnp.sum((state - target) ** 2)

    def native(target):
        return phx.optim.implicit_minimize(
            objective,
            jnp.array([0.0]),
            args=target,
            termination=_termination(),
        )[0]

    def upstream(target):
        return optx.minimise(
            objective,
            optx.BFGS(rtol=1e-10, atol=1e-10),
            jnp.array([0.0]),
            target,
            {},
            max_steps=60,
            adjoint=optx.ImplicitAdjoint(),
            throw=True,
        ).value[0]

    target = jnp.array(2.0)
    np.testing.assert_allclose(jax.grad(native)(target), jax.grad(upstream)(target))


def test_derivative_is_stable_when_iteration_budget_changes():
    def solution(parameter, maximum_steps):
        return phx.optim.implicit_minimize(
            lambda state, value: 0.5 * (state[0] ** 2 - value) ** 2,
            jnp.array([1.4]),
            args=parameter,
            termination=_termination(maximum_steps=maximum_steps),
        )[0]

    short = jax.grad(lambda value: solution(value, 20))(jnp.array(2.0))
    long = jax.grad(lambda value: solution(value, 80))(jnp.array(2.0))
    np.testing.assert_allclose(short, long, atol=1e-8)


def test_implicit_minimize_rejects_unsuccessful_primal_solve():
    solve = eqx.filter_jit(
        lambda target: phx.optim.implicit_minimize(
            lambda state, value: jnp.sum((state - value) ** 4),
            jnp.array([0.0]),
            args=target,
            termination=_termination(maximum_steps=1),
        )
    )

    with pytest.raises(Exception, match="successful regular stationary point"):
        solve(jnp.array([3.0]))


def test_implicit_minimize_rejects_singular_hessian():
    solve = eqx.filter_jit(
        lambda parameter: phx.optim.implicit_minimize(
            lambda state, value: state[0] ** 4 + value * state[0],
            jnp.array([0.0]),
            args=parameter,
            termination=_termination(),
        )
    )

    with pytest.raises(Exception, match="singular or did not converge"):
        solve(jnp.array(0.0))


def test_implicit_least_squares_rejects_rank_deficient_stationarity():
    solve = eqx.filter_jit(
        lambda target: phx.optim.implicit_least_squares(
            lambda state, value: jnp.array([state[0] + state[1] - value]),
            jnp.array([0.0, 0.0]),
            args=target,
            termination=_termination(),
        )
    )

    with pytest.raises(Exception, match="singular or did not converge"):
        solve(jnp.array(1.0))


def test_implicit_minimize_rejects_nonfinite_solution_data():
    solve = eqx.filter_jit(
        lambda target: phx.optim.implicit_minimize(
            lambda state, value: jnp.sum((state - value) ** 2),
            jnp.array([0.0]),
            args=target,
            termination=_termination(),
        )
    )

    with pytest.raises(Exception, match="successful regular stationary point"):
        solve(jnp.array([jnp.nan]))
