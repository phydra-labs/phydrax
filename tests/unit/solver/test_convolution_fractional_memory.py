#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pytest

import phydrax as phx


def test_translation_invariant_convolution_backend_matches_causal_integral():
    times = jnp.linspace(0.0, 1.0, 101)
    problem = phx.solver.ConvolutionVolterraProblem(
        lambda time, state, args: jnp.ones_like(state),
        jnp.asarray([1.0]),
        t0=0.0,
        t1=1.0,
        kernel=lambda lag, args: jnp.exp(-lag),
    )
    solution = phx.solver.solve_convolution_volterra(problem, times=times)

    assert jnp.isclose(solution.states[-1, 0], 2.0 - jnp.exp(-1.0), atol=3.3e-3)
    assert solution.solver_id == "solver:volterra:causal-convolution-euler:v1"
    assert solution.metadata["kernel_structure"] == "translation-invariant"
    assert solution.metadata["convolution_backend"] == "direct-causal"


def test_convolution_backend_is_jittable_and_differentiable():
    times = jnp.linspace(0.0, 1.0, 21)

    def terminal(rate):
        problem = phx.solver.ConvolutionVolterraProblem(
            lambda time, state, args: args * jnp.ones_like(state),
            jnp.asarray([1.0]),
            t0=0.0,
            t1=1.0,
            kernel=lambda lag, args: jnp.exp(-lag),
            args=rate,
        )
        return phx.solver.solve_convolution_volterra(
            problem, times=times
        ).states[-1, 0]

    value, gradient = jax.jit(jax.value_and_grad(terminal))(jnp.asarray(2.0))
    expected_gradient = jnp.sum(
        jnp.diff(times) * jnp.exp(-(1.0 - times[:-1]))
    )

    assert jnp.isclose(value, 1.0 + 2.0 * expected_gradient)
    assert jnp.isclose(gradient, expected_gradient)


def test_caputo_product_integration_is_exact_for_constant_forcing_nonuniform_grid():
    times = jnp.asarray([0.0, 0.03, 0.11, 0.4, 1.0])
    order = 0.6
    rate = 2.0
    problem = phx.solver.CaputoFractionalProblem(
        lambda time, state, args: args * jnp.ones_like(state),
        jnp.asarray([1.0]),
        order,
        t0=0.0,
        t1=1.0,
        args=rate,
    )
    solution = phx.solver.solve_caputo_fractional(problem, times=times)
    expected = 1.0 + rate * times**order / jsp.special.gamma(order + 1.0)

    assert jnp.allclose(solution.states[:, 0], expected, rtol=0.0, atol=2e-15)
    assert solution.solver_id == "solver:fractional:caputo-product-integration:v1"
    assert solution.stats["num_memory_cells"] == 10
    assert solution.metadata["grid"] == "nonuniform-supported"


def test_caputo_orders_above_one_include_initial_derivative():
    times = jnp.asarray([0.0, 0.2, 0.7, 1.0])
    order = 1.4
    problem = phx.solver.CaputoFractionalProblem(
        lambda time, state, args: 0.5 * jnp.ones_like(state),
        jnp.asarray([2.0]),
        order,
        t0=0.0,
        t1=1.0,
        initial_derivative=jnp.asarray([-0.3]),
    )
    solution = phx.solver.solve_caputo_fractional(problem, times=times)
    expected = (
        2.0
        - 0.3 * times
        + 0.5 * times**order / jsp.special.gamma(order + 1.0)
    )

    assert jnp.allclose(solution.states[:, 0], expected, rtol=0.0, atol=2e-15)


def test_caputo_backend_is_jittable_and_differentiable():
    times = jnp.asarray([0.0, 0.03, 0.11, 0.4, 1.0])
    order = 0.6

    def terminal(rate):
        problem = phx.solver.CaputoFractionalProblem(
            lambda time, state, args: args * jnp.ones_like(state),
            jnp.asarray([1.0]),
            order,
            t0=0.0,
            t1=1.0,
            args=rate,
        )
        return phx.solver.solve_caputo_fractional(
            problem, times=times
        ).states[-1, 0]

    value, gradient = jax.jit(jax.value_and_grad(terminal))(jnp.asarray(2.0))
    expected_gradient = 1.0 / jsp.special.gamma(order + 1.0)

    assert jnp.isclose(value, 1.0 + 2.0 * expected_gradient)
    assert jnp.isclose(gradient, expected_gradient)


@pytest.mark.parametrize("order", [0.0, -0.2, 2.1, jnp.inf])
def test_caputo_problem_rejects_invalid_orders(order):
    with pytest.raises(ValueError, match="order"):
        phx.solver.CaputoFractionalProblem(
            lambda time, state, args: jnp.ones_like(state),
            jnp.asarray([1.0]),
            order,
            t0=0.0,
            t1=1.0,
        )


def test_caputo_problem_requires_derivative_only_above_order_one():
    with pytest.raises(ValueError, match="require initial_derivative"):
        phx.solver.CaputoFractionalProblem(
            lambda time, state, args: jnp.ones_like(state),
            jnp.asarray([1.0]),
            1.2,
            t0=0.0,
            t1=1.0,
        )
    with pytest.raises(ValueError, match="only valid"):
        phx.solver.CaputoFractionalProblem(
            lambda time, state, args: jnp.ones_like(state),
            jnp.asarray([1.0]),
            0.8,
            t0=0.0,
            t1=1.0,
            initial_derivative=jnp.asarray([0.0]),
        )
