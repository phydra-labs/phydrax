from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _brownian_paths(*, num_paths=128, num_steps=8):
    times = jnp.linspace(0.0, 1.0, num_steps + 1)
    increments = (
        jr.normal(jr.key(30), (num_paths, num_steps, 1))
        * jnp.sqrt(jnp.diff(times))[None, :, None]
    )
    states = jnp.concatenate(
        (jnp.zeros((num_paths, 1, 1)), jnp.cumsum(increments, axis=1)), axis=1
    )
    return phx.stochastic.BSDEPathBatch(
        times,
        states,
        increments,
        sample_shape=(num_paths,),
        state_shape=(1,),
        noise_shape=(1,),
        path_id="brownian-paths",
        process_id="brownian",
    )


def _problem(paths):
    return phx.stochastic.BSDEProblem(
        lambda key: paths,
        lambda time, state, args: jnp.zeros_like(state),
        lambda time, state, args: jnp.ones((1, 1)),
        lambda time, state, value, control, args: jnp.zeros_like(value),
        lambda state, args: jnp.asarray([state[0]]),
        state_shape=(1,),
        noise_shape=(1,),
        output_shape=(1,),
        problem_id="linear-bsde",
        process_id="brownian",
    )


def test_bsde_path_contract_rejects_misaligned_increments():
    paths = _brownian_paths(num_paths=4, num_steps=3)
    assert paths.num_steps == 3
    assert jnp.all(paths.successful)

    with pytest.raises(ValueError, match="align"):
        phx.stochastic.BSDEPathBatch(
            paths.times,
            paths.states,
            paths.wiener_increments[:, :-1],
            sample_shape=(4,),
            state_shape=(1,),
            noise_shape=(1,),
            path_id="bad",
            process_id="brownian",
        )


def test_exact_linear_bsde_has_zero_local_global_and_terminal_residuals():
    paths = _brownian_paths()
    problem = _problem(paths)
    value = lambda time, state: jnp.asarray([state[0]])
    control = lambda time, state: jnp.ones((1, 1))
    evaluation = phx.stochastic.evaluate_bsde(
        problem,
        paths,
        value,
        control_predictor=control,
        control_mode="explicit",
    )

    assert jnp.allclose(evaluation.terminal_residual, 0.0)
    assert jnp.allclose(evaluation.local_residuals, 0.0)
    assert jnp.allclose(evaluation.global_residual, 0.0)
    for mode in ("terminal", "local", "global", "joint"):
        assert jnp.allclose(
            phx.stochastic.bsde_objective_loss(evaluation, mode=mode), 0.0
        )
    assert phx.stochastic.bsde_diagnostics(evaluation).passed


def test_autodiff_control_matches_explicit_control_and_heat_pde_residual():
    paths = _brownian_paths()
    problem = _problem(paths)
    linear_value = lambda time, state: jnp.asarray([state[0]])
    explicit = phx.stochastic.evaluate_bsde(
        problem,
        paths,
        linear_value,
        control_predictor=lambda time, state: jnp.ones((1, 1)),
    )
    automatic = phx.stochastic.evaluate_bsde(
        problem,
        paths,
        linear_value,
        control_mode="autodiff",
    )

    assert jnp.allclose(automatic.controls, explicit.controls)
    heat_problem = phx.stochastic.BSDEProblem(
        lambda key: paths,
        problem.drift,
        problem.diffusion,
        problem.generator,
        lambda state, args: jnp.asarray([state[0] ** 2]),
        state_shape=(1,),
        noise_shape=(1,),
        output_shape=(1,),
        problem_id="heat-bsde",
        process_id="brownian",
    )
    heat_value = lambda time, state: jnp.asarray([state[0] ** 2 + 1.0 - time])

    assert jnp.allclose(
        phx.stochastic.autodiff_bsde_control(
            heat_value, 0.2, jnp.asarray([0.3]), heat_problem
        ),
        jnp.asarray([[0.6]]),
    )
    assert jnp.allclose(
        phx.stochastic.semilinear_pde_residual(
            heat_problem, heat_value, 0.2, jnp.asarray([0.3])
        ),
        0.0,
    )


def test_bsde_objective_integrates_domain_functions_and_fixed_paths():
    paths = _brownian_paths(num_paths=32)
    problem = _problem(paths)
    domain = phx.domain.Interval1d(-5.0, 5.0) @ phx.domain.TimeInterval(0.0, 1.0)
    value = domain.Function("t", "x")(lambda time, state: jnp.asarray([state[0]]))
    objective = phx.terms.BSDETerm(
        problem,
        value_name="value",
        control_mode="autodiff",
        mode="joint",
        sampling_mode="fixed",
        fixed_paths=paths,
    )

    assert objective.sample() is paths
    assert jnp.allclose(objective.loss({"value": value}), 0.0)


def test_bsde_quadrature_modes_are_explicit_and_finite():
    paths = _brownian_paths(num_paths=32)
    problem = _problem(paths)
    value = lambda time, state: jnp.asarray([state[0] + time])
    control = lambda time, state: jnp.ones((1, 1))

    for quadrature in ("left", "trapezoid"):
        evaluation = phx.stochastic.evaluate_bsde(
            problem,
            paths,
            value,
            control_predictor=control,
            quadrature=quadrature,
        )
        assert jnp.all(jnp.isfinite(evaluation.local_residuals))
        assert phx.stochastic.bsde_objective_loss(evaluation, mode="joint") > 0.0

    invalid_quadrature: Any = "midpoint"
    with pytest.raises(ValueError, match="left.*trapezoid"):
        phx.stochastic.evaluate_bsde(
            problem,
            paths,
            value,
            control_predictor=control,
            quadrature=invalid_quadrature,
        )


def test_differential_solution_conversion_collapses_shared_batched_time_grid():
    times = jnp.linspace(0.0, 1.0, 4)
    realization = phx.stochastic.WienerRealization(
        jr.key(31),
        (1,),
        support=(0.0, 1.0),
        sample_shape=(3,),
        tolerance=1e-3,
        noise_id="conversion-wiener",
    )
    increments = realization.increments(times[:-1], times[1:])
    states = jnp.concatenate(
        (jnp.zeros((3, 1, 1)), jnp.cumsum(increments, axis=1)),
        axis=1,
    )
    solution = SimpleNamespace(
        times=jnp.broadcast_to(times, (3, 4)),
        states=states,
        valid=jnp.ones((3, 4), dtype=bool),
        sample_shape=(3,),
        realization=realization,
        solver_name="test-solver",
        interpretation="ito",
    )

    paths = phx.stochastic.bsde_paths_from_differential_solution(
        solution,
        path_id="converted",
        process_id="conversion-wiener",
    )

    assert jnp.array_equal(paths.times, times)
    assert jnp.array_equal(paths.states, states)
    assert jnp.array_equal(paths.wiener_increments, increments)
