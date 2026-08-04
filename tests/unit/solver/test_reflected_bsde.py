import jax.numpy as jnp
import pytest

import phydrax as phx


def _tree_paths(*, num_steps=3, repeats=32):
    branches = 2**num_steps
    indices = jnp.repeat(jnp.arange(branches, dtype=jnp.int32), repeats)
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
        path_id="reflected-tree",
        process_id="wiener",
    )


def _deterministic_paths(num_paths=64):
    return phx.stochastic.BSDEPathBatch(
        jnp.asarray([0.0, 0.5, 1.0]),
        jnp.zeros((num_paths, 3, 1)),
        jnp.zeros((num_paths, 2, 1)),
        sample_shape=(num_paths,),
        state_shape=(1,),
        noise_shape=(1,),
        path_id="reflected-deterministic",
        process_id="wiener",
    )


def test_reflected_solver_uses_nonanticipative_path_features_and_lower_obstacle():
    repeats = 32
    paths = _tree_paths(repeats=repeats)

    def path_features(times, history, args):
        del times, args
        return jnp.asarray([history[-1, 0], jnp.max(history[:, 0])])

    def lower_obstacle(time, times, history, args):
        del times, args
        return jnp.asarray([(1.0 - time) * jnp.max(history[:, 0])])

    problem = phx.stochastic.ReflectedPathDependentBSDEProblem(
        lambda key: paths,
        path_features,
        lambda time, times, history, value, control, args: jnp.zeros((1,)),
        lambda times, history, args: jnp.zeros((1,)),
        state_shape=(1,),
        noise_shape=(1,),
        regression_state_shape=(2,),
        output_shape=(1,),
        problem_id="lookback-obstacle",
        process_id="wiener",
        lower_obstacle=lower_obstacle,
    )
    basis = phx.solver.PolynomialBSDERegressionBasis((2,), 1)

    result = phx.solver.solve_reflected_path_dependent_bsde(
        problem,
        basis,
        paths=paths,
        ridge=1e-10,
    )

    positive_then_negative = repeats
    negative_then_positive = 2 * repeats
    assert jnp.allclose(
        paths.states[positive_then_negative, 2],
        paths.states[negative_then_positive, 2],
    )
    assert result.values[positive_then_negative, 2, 0] > 0.0
    assert jnp.allclose(result.values[negative_then_positive, 2, 0], 0.0)
    assert jnp.all(result.values >= result.lower_obstacles)
    assert jnp.any(result.lower_reflection_increments > 0.0)
    histories = jnp.stack(
        (
            paths.states[positive_then_negative, :3],
            paths.states[negative_then_positive, :3],
        )
    )
    predicted = phx.solver.predict_reflected_path_dependent_value(result, 2, histories)
    assert jnp.allclose(
        predicted,
        result.values[[positive_then_negative, negative_then_positive], 2],
        atol=1e-9,
    )
    diagnostics = phx.solver.reflected_path_dependent_bsde_diagnostics(result)
    assert diagnostics.passed
    assert diagnostics.lower_constraint_violation == 0.0
    assert diagnostics.lower_complementarity_error == 0.0


def test_doubly_reflected_implicit_scheme_projects_inside_picard_iteration():
    paths = _deterministic_paths()
    problem = phx.stochastic.ReflectedPathDependentBSDEProblem(
        lambda key: paths,
        lambda times, history, args: jnp.zeros((1,)),
        lambda time, times, history, value, control, args: 1.0 + 0.2 * value,
        lambda times, history, args: jnp.zeros((1,)),
        state_shape=(1,),
        noise_shape=(1,),
        regression_state_shape=(1,),
        output_shape=(1,),
        problem_id="double-reflection",
        process_id="wiener",
        lower_obstacle=lambda time, times, history, args: jnp.asarray([-0.1]),
        upper_obstacle=lambda time, times, history, args: jnp.asarray([0.3]),
    )
    basis = phx.solver.PolynomialBSDERegressionBasis((1,), 0)

    result = phx.solver.solve_reflected_path_dependent_bsde(
        problem,
        basis,
        paths=paths,
        scheme="implicit",
        ridge=0.0,
        picard_tolerance=1e-12,
    )

    assert jnp.allclose(result.values[0, :, 0], jnp.asarray([0.3, 0.3, 0.0]))
    assert jnp.allclose(result.generator_values, 1.06)
    assert jnp.allclose(
        result.upper_reflection_increments[0, :, 0],
        jnp.asarray([0.53, 0.23]),
        atol=1e-12,
    )
    assert jnp.allclose(result.lower_reflection_increments, 0.0)
    assert jnp.allclose(result.local_residuals, 0.0, atol=1e-12)
    assert jnp.all(result.picard_converged)
    assert phx.solver.reflected_path_dependent_bsde_diagnostics(result).passed


def test_reflected_solver_never_projects_incompatible_terminal_data():
    paths = _deterministic_paths()
    problem = phx.stochastic.ReflectedPathDependentBSDEProblem(
        lambda key: paths,
        lambda times, history, args: jnp.zeros((1,)),
        lambda time, times, history, value, control, args: jnp.zeros((1,)),
        lambda times, history, args: jnp.zeros((1,)),
        state_shape=(1,),
        noise_shape=(1,),
        regression_state_shape=(1,),
        output_shape=(1,),
        problem_id="incompatible-terminal",
        process_id="wiener",
        lower_obstacle=lambda time, times, history, args: jnp.asarray([0.5]),
    )
    basis = phx.solver.PolynomialBSDERegressionBasis((1,), 0)

    result = phx.solver.solve_reflected_path_dependent_bsde(problem, basis, paths=paths)

    assert jnp.all(result.values[..., -1, :] == 0.0)
    assert not jnp.any(result.terminal_compatible)
    assert not result.successful
    with pytest.raises(RuntimeError, match="failed validation"):
        phx.solver.solve_reflected_path_dependent_bsde(
            problem,
            basis,
            paths=paths,
            raise_on_failure=True,
        )
