import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


class _AffineKernel(eqx.Module):
    scale: jax.Array

    def __call__(self, time, lag, state, args):
        del time, state, args
        return self.scale * (1.0 + lag)


def _polynomial_moments(lower, upper):
    return tuple(
        (upper ** (degree + 1) - lower ** (degree + 1)) / (degree + 1)
        for degree in range(4)
    )


def _polynomial_exact(time, *, lower=0.55, upper=0.95):
    moment0, moment1, moment2, moment3 = _polynomial_moments(lower, upper)
    quadratic = moment0 + 2.0 * moment1
    linear = 2.0 * moment0 + 2.0 * moment1 - 4.0 * moment2
    constant = 3.0 * moment0 + 4.0 * moment1 - 3.0 * moment2 + 2.0 * moment3
    return 3.0 + quadratic * time**3 / 3.0 + linear * time**2 / 2.0 + constant * time


def _polynomial_problem(order):
    term = phx.solver.DistributedDelay(
        "polynomial",
        lambda time, lag, state, args: 1.0 + 2.0 * lag,
        (0.55, 0.95),
        quadrature=phx.integration.GaussLegendreRule(order),
    )

    def history(time, args):
        del args
        return jnp.asarray([time**2 + 2.0 * time + 3.0])

    def drift(time, state, memory, args):
        del time, state, args
        return memory["polynomial"]

    return phx.solver.DelayDifferentialProblem(
        drift,
        history,
        (term,),
        t0=0.0,
        t1=0.4,
    )


def test_gauss_legendre_distributed_delay_is_polynomially_exact():
    times = jnp.linspace(0.0, 0.4, 9)
    solution = phx.solver.solve_diffrax_delay(
        _polynomial_problem(2),
        save_times=times,
        rtol=1e-10,
        atol=1e-12,
        max_steps=512,
    )

    assert bool(solution.successful)
    assert solution.states.shape == (9, 1)
    assert jnp.allclose(
        solution.states[:, 0],
        _polynomial_exact(times),
        rtol=2e-9,
        atol=2e-10,
    )


def test_distributed_delay_recovers_exponential_matrix_trajectory():
    rate = 0.35
    lower, upper = 0.2, 0.45
    base = jnp.asarray([[1.0, -0.25], [0.5, 1.75]])
    kernel_coefficients = jnp.asarray([[1.0, 0.75], [1.5, 2.0]])

    def exponential_integral(bound):
        return -jnp.exp(-rate * bound) * ((1.0 + bound) / rate + 1.0 / rate**2)

    scalar_factor = exponential_integral(upper) - exponential_integral(lower)
    factor = kernel_coefficients * scalar_factor

    def history(time, args):
        del args
        return jnp.exp(rate * time) * base

    def kernel(time, lag, state, args):
        del time, state, args
        return kernel_coefficients * (1.0 + lag)

    term = phx.solver.DistributedDelay(
        "matrix_average",
        kernel,
        (lower, upper),
        quadrature=phx.integration.GaussLegendreRule(12),
    )
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: rate * memory[0] / factor,
        history,
        (term,),
        t0=0.0,
        t1=0.9,
    )
    times = jnp.linspace(0.0, 0.9, 10)
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=times,
        rtol=2e-8,
        atol=2e-10,
        max_steps=2048,
    )
    expected = jnp.exp(rate * times)[:, None, None] * base

    assert solution.states.shape == (10, 2, 2)
    assert jnp.allclose(solution.states, expected, rtol=2e-6, atol=2e-8)


def test_mixed_point_and_distributed_terms_preserve_names_and_provenance():
    rate = 0.3
    point_lag = 0.31
    lower, upper = 0.22, 0.42
    base = jnp.asarray([1.0, -0.4])
    distributed_factor = (jnp.exp(-rate * lower) - jnp.exp(-rate * upper)) / rate
    point = phx.solver.ConstantDelay("point", point_lag)
    spread = phx.solver.DistributedDelay(
        "spread",
        lambda time, lag, state, args: jnp.asarray(1.0),
        (lower, upper),
        quadrature=phx.integration.GaussLegendreRule(8),
    )

    def history(time, args):
        del args
        return jnp.exp(rate * time) * base

    def drift(time, state, memory, args):
        del time, state, args
        point_part = 0.4 * rate * jnp.exp(rate * point_lag) * memory["point"]
        spread_part = 0.6 * rate * memory[1] / distributed_factor
        return point_part + spread_part

    problem = phx.solver.DelayDifferentialProblem(
        drift,
        history,
        (point, spread),
        t0=0.0,
        t1=0.8,
    )
    times = jnp.linspace(0.0, 0.8, 9)
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=times,
        rtol=2e-8,
        atol=2e-10,
        max_steps=2048,
    )
    expected = jnp.exp(rate * times)[:, None] * base
    provenance = solution.metadata["distributed_delay_quadrature"]

    assert problem.delay_names == ("point", "spread")
    assert jnp.allclose(solution.states, expected, rtol=2e-6, atol=2e-8)
    assert len(provenance) == 1
    assert provenance[0]["name"] == "spread"
    assert provenance[0]["family"] == "GaussLegendreRule"
    assert provenance[0]["order"] == 8
    assert provenance[0]["node_count"] == 8
    effective_lower, effective_upper = provenance[0]["effective_lag_range"]
    assert jnp.allclose(effective_lower, spread.minimum_delay)
    assert jnp.allclose(effective_upper, spread.maximum_delay)


def test_existing_fixed_interval_rule_materializes_without_parallel_rule_path():
    lower, upper = 0.25, 0.5
    term = phx.solver.DistributedDelay(
        "spread",
        lambda time, lag, state, args: 1.0 / (upper - lower),
        (lower, upper),
        quadrature=phx.integration.ClenshawCurtisRule(level=3),
    )
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: memory["spread"] - jnp.ones_like(state),
        lambda time, args: jnp.ones((1,)),
        (term,),
        t0=0.0,
        t1=0.35,
    )
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.0, 0.35]),
        max_steps=256,
    )
    provenance = solution.metadata["distributed_delay_quadrature"][0]

    assert term.node_count == 9
    assert jnp.all(jnp.isfinite(term.nodes))
    assert jnp.all(jnp.isfinite(term.weights))
    assert jnp.allclose(solution.states, 1.0)
    assert provenance["family"] == "ClenshawCurtisRule"
    assert provenance["order"] == 9
    assert provenance["node_count"] == 9
    assert jnp.allclose(
        jnp.stack(provenance["effective_lag_range"]),
        jnp.asarray([lower, upper]),
    )


def test_temporal_convergence_is_independent_of_quadrature_refinement():
    terminal = jnp.asarray([0.4])
    step_sizes = (0.1, 0.05, 0.025)

    def errors(order):
        values = []
        for step in step_sizes:
            solution = phx.solver.solve_diffrax_delay(
                _polynomial_problem(order),
                save_times=terminal,
                solver=dfx.Euler(),
                stepsize_controller=dfx.ConstantStepSize(),
                dt0=step,
                max_steps=128,
            )
            values.append(jnp.abs(solution.states[0, 0] - _polynomial_exact(0.4)))
        return jnp.stack(values)

    order_two = errors(2)
    order_six = errors(6)

    assert jnp.all(order_two[1:] < order_two[:-1])
    assert jnp.all(order_two[:-1] / order_two[1:] > 1.8)
    assert jnp.allclose(order_two, order_six, rtol=2e-11, atol=2e-13)


def test_kernel_and_interval_endpoints_support_jit_vmap_and_grad():
    final_time = 0.1

    def terminal(parameters):
        scale, lower, upper = parameters
        term = phx.solver.DistributedDelay(
            "trainable",
            _AffineKernel(scale),
            (lower, upper),
            quadrature=phx.integration.GaussLegendreRule(2),
        )
        problem = phx.solver.DelayDifferentialProblem(
            lambda time, state, memory, args: memory["trainable"],
            lambda time, args: jnp.ones((1,)),
            (term,),
            t0=0.0,
            t1=final_time,
        )
        return phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([final_time]),
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            dt0=0.025,
            max_steps=32,
        ).states[0, 0]

    parameters = jnp.asarray([1.3, 0.4, 0.8])
    scale, lower, upper = parameters
    integral = (upper - lower) + 0.5 * (upper**2 - lower**2)
    expected = 1.0 + final_time * scale * integral
    expected_gradient = final_time * jnp.asarray(
        [integral, -scale * (1.0 + lower), scale * (1.0 + upper)]
    )

    assert jnp.allclose(jax.jit(terminal)(parameters), expected, atol=2e-10)
    assert jnp.allclose(jax.grad(terminal)(parameters), expected_gradient, atol=2e-9)

    batched = jnp.stack((parameters, jnp.asarray([0.7, 0.35, 0.75])))
    batched_expected = 1.0 + final_time * batched[:, 0] * (
        (batched[:, 2] - batched[:, 1]) + 0.5 * (batched[:, 2] ** 2 - batched[:, 1] ** 2)
    )
    assert jnp.allclose(jax.vmap(terminal)(batched), batched_expected, atol=2e-9)


def test_distributed_delay_validates_rule_kernel_shape_and_lag_bounds():
    with pytest.raises(TypeError, match="kernel must be callable"):
        phx.solver.DistributedDelay("bad", 1.0, (0.2, 0.4))
    with pytest.raises(TypeError, match="reducer must be callable"):
        phx.solver.DistributedDelay(
            "bad",
            lambda time, lag, state, args: 1.0,
            (0.2, 0.4),
            reducer=1.0,
        )
    with pytest.raises(TypeError, match="Unsupported interval rule"):
        phx.solver.DistributedDelay(
            "bad",
            lambda time, lag, state, args: 1.0,
            (0.2, 0.4),
            quadrature=phx.integration.AdaptiveQuadraturePlan(),
        )
    with pytest.raises(ValueError, match="bounds must be scalar"):
        phx.solver.DistributedDelay(
            "bad",
            lambda time, lag, state, args: 1.0,
            (jnp.asarray([0.2]), 0.4),
        )
    with pytest.raises(Exception, match="finite and exceed"):
        phx.solver.DistributedDelay(
            "bad",
            lambda time, lag, state, args: 1.0,
            (0.4, 0.4),
        )
    with pytest.raises(Exception, match="finite and nonnegative"):
        phx.solver.DistributedDelay(
            "bad",
            lambda time, lag, state, args: 1.0,
            (jnp.nan, 0.4),
        )
    open_rule = phx.solver.DistributedDelay(
        "open_rule",
        lambda time, lag, state, args: 1.0,
        (0.0, 0.4),
        quadrature=phx.integration.GaussLegendreRule(2),
    )
    assert open_rule.minimum_delay > 0.0
    with pytest.raises(Exception, match="positive lags"):
        phx.solver.DistributedDelay(
            "bad",
            lambda time, lag, state, args: 1.0,
            (0.0, 0.4),
            quadrature=phx.integration.ClenshawCurtisRule(level=2),
        )

    malformed = phx.solver.DistributedDelay(
        "malformed",
        lambda time, lag, state, args: jnp.ones((1,)),
        (0.2, 0.4),
    )
    with pytest.raises(ValueError, match="exact state shape"):
        phx.solver.DelayDifferentialProblem(
            lambda time, state, memory, args: state,
            lambda time, args: jnp.ones((2,)),
            (malformed,),
            t0=0.0,
            t1=0.1,
        )


def test_non_euclidean_distributed_delay_requires_valid_reducer():
    geometry = phx.metrix.SpecialOrthogonalStateGeometry(2)
    history = lambda time, args: jnp.eye(2)
    drift = lambda time, state, memory, args: jnp.zeros_like(state)
    without_reducer = phx.solver.DistributedDelay(
        "rotation_average",
        lambda time, lag, state, args: 1.0,
        (0.2, 0.4),
    )

    with pytest.raises(ValueError, match="require an explicit reducer"):
        phx.solver.DelayDifferentialProblem(
            drift,
            history,
            (without_reducer,),
            t0=0.0,
            t1=0.1,
            state_geometry=geometry,
        )

    valid = phx.solver.DistributedDelay(
        "rotation_average",
        lambda time, lag, state, args: 1.0,
        (0.2, 0.4),
        reducer=lambda time, state, lags, weights, kernels, values, args: state,
    )
    problem = phx.solver.DelayDifferentialProblem(
        drift,
        history,
        (valid,),
        t0=0.0,
        t1=0.1,
        state_geometry=geometry,
    )
    assert problem.state_geometry_id == geometry.geometry_id
    invalid_history = lambda time, args: jnp.where(
        time < 0.0,
        jnp.zeros((2, 2)),
        jnp.eye(2),
    )
    with pytest.raises(Exception, match="initial history outside state_geometry"):
        phx.solver.DelayDifferentialProblem(
            drift,
            invalid_history,
            (valid,),
            t0=0.0,
            t1=0.1,
            state_geometry=geometry,
        )

    invalid = phx.solver.DistributedDelay(
        "rotation_average",
        lambda time, lag, state, args: 1.0,
        (0.2, 0.4),
        reducer=lambda time, state, lags, weights, kernels, values, args: jnp.zeros_like(
            state
        ),
    )
    with pytest.raises(Exception, match="outside state_geometry"):
        phx.solver.DelayDifferentialProblem(
            drift,
            history,
            (invalid,),
            t0=0.0,
            t1=0.1,
            state_geometry=geometry,
        )
