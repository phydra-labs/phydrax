import jax.numpy as jnp
import pytest

import phydrax as phx


def _brownian_tree_paths(*, num_steps=4, repeats=16):
    num_branches = 2**num_steps
    indices = jnp.repeat(jnp.arange(num_branches, dtype=jnp.int32), repeats)
    bits = (indices[:, None] >> jnp.arange(num_steps, dtype=jnp.int32)) & 1
    dt = 1.0 / num_steps
    increments = (2.0 * bits - 1.0) * jnp.sqrt(dt)
    states = jnp.concatenate(
        (jnp.zeros((indices.shape[0], 1)), jnp.cumsum(increments, axis=1)),
        axis=1,
    )
    return phx.stochastic.BSDEPathBatch(
        jnp.linspace(0.0, 1.0, num_steps + 1),
        states[..., None],
        increments[..., None],
        sample_shape=(indices.shape[0],),
        state_shape=(1,),
        noise_shape=(1,),
        path_id="balanced-brownian-tree",
        process_id="wiener",
    )


def _problem(paths, *, generator, terminal):
    return phx.stochastic.BSDEProblem(
        lambda key: paths,
        lambda time, state, args: jnp.zeros((1,)),
        lambda time, state, args: jnp.ones((1, 1)),
        generator,
        terminal,
        state_shape=(1,),
        noise_shape=(1,),
        output_shape=(1,),
        problem_id="regression-bsde",
        process_id="wiener",
    )


def test_explicit_least_squares_recovers_linear_martingale_and_control():
    paths = _brownian_tree_paths()
    problem = _problem(
        paths,
        generator=lambda time, state, value, control, args: jnp.zeros((1,)),
        terminal=lambda state, args: state,
    )
    basis = phx.solver.PolynomialBSDERegressionBasis((1,), 1)

    result = phx.solver.solve_bsde_least_squares(
        problem,
        basis,
        paths=paths,
        ridge=1e-10,
    )

    assert result.values.shape == (256, 5, 1)
    assert result.controls.shape == (256, 4, 1, 1)
    assert jnp.allclose(result.values, paths.states, atol=2e-8)
    assert jnp.allclose(result.controls, 1.0, atol=2e-8)
    assert jnp.allclose(result.local_residuals, 0.0, atol=3e-8)
    assert jnp.all(result.valid_steps)
    assert result.successful
    assert jnp.allclose(
        phx.solver.predict_bsde_least_squares_value(
            result, 2, jnp.asarray([[-0.5], [0.5]])
        ),
        jnp.asarray([[-0.5], [0.5]]),
        atol=2e-8,
    )
    assert jnp.allclose(
        phx.solver.predict_bsde_least_squares_control(
            result, 2, jnp.asarray([[-0.5], [0.5]])
        ),
        1.0,
        atol=2e-8,
    )
    diagnostics = phx.solver.least_squares_bsde_diagnostics(result)
    assert diagnostics.passed
    assert diagnostics.max_value_normal_equation_error < 1e-8
    assert diagnostics.max_control_normal_equation_error < 1e-8


def test_implicit_least_squares_solves_current_value_backward_euler_equation():
    num_paths = 64
    num_steps = 4
    times = jnp.linspace(0.0, 1.0, num_steps + 1)
    paths = phx.stochastic.BSDEPathBatch(
        times,
        jnp.zeros((num_paths, num_steps + 1, 1)),
        jnp.zeros((num_paths, num_steps, 1)),
        sample_shape=(num_paths,),
        state_shape=(1,),
        noise_shape=(1,),
        path_id="deterministic-paths",
        process_id="wiener",
    )
    rate = 0.2
    problem = _problem(
        paths,
        generator=lambda time, state, value, control, args: rate * value,
        terminal=lambda state, args: jnp.ones((1,)),
    )
    basis = phx.solver.PolynomialBSDERegressionBasis((1,), 0)

    implicit = phx.solver.solve_bsde_least_squares(
        problem,
        basis,
        paths=paths,
        scheme="implicit",
        ridge=0.0,
        max_picard_steps=32,
        picard_tolerance=1e-11,
    )
    explicit = phx.solver.solve_bsde_least_squares(
        problem,
        basis,
        paths=paths,
        scheme="explicit",
        ridge=0.0,
    )

    implicit_factor = 1.0 / (1.0 - rate / num_steps)
    explicit_factor = 1.0 + rate / num_steps
    implicit_expected = jnp.asarray(
        [implicit_factor ** (num_steps - step) for step in range(num_steps + 1)]
    )
    explicit_expected = jnp.asarray(
        [explicit_factor ** (num_steps - step) for step in range(num_steps + 1)]
    )
    assert jnp.allclose(implicit.values[0, :, 0], implicit_expected, atol=2e-11)
    assert jnp.allclose(explicit.values[0, :, 0], explicit_expected, atol=1e-12)
    assert jnp.allclose(implicit.local_residuals, 0.0, atol=2e-11)
    assert jnp.all(implicit.picard_converged)
    assert jnp.all(implicit.picard_iterations > 1)
    assert phx.solver.least_squares_bsde_diagnostics(implicit).passed


def test_least_squares_reports_insufficient_conditional_sample_budget():
    paths = _brownian_tree_paths(num_steps=1, repeats=1)
    problem = _problem(
        paths,
        generator=lambda time, state, value, control, args: jnp.zeros((1,)),
        terminal=lambda state, args: state,
    )
    basis = phx.solver.PolynomialBSDERegressionBasis((1,), 2)

    result = phx.solver.solve_bsde_least_squares(problem, basis, paths=paths)

    assert not result.successful
    assert not jnp.all(result.valid_steps)
    with pytest.raises(RuntimeError, match="failed validation"):
        phx.solver.solve_bsde_least_squares(
            problem,
            basis,
            paths=paths,
            raise_on_failure=True,
        )
