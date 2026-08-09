#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from tests._control_systems import (
    make_differential_control_dynamics,
    make_discrete_control_dynamics,
)


def _quadratic_problem(*, initial_state=None, num_steps=2):
    times = jnp.linspace(0.0, 1.0, num_steps + 1)
    time_grid = phx.dynamics.TimeGrid(times, time_id=f"search-time-{num_steps}")
    dynamics = make_discrete_control_dynamics(
        lambda time, state, control, args: state + control,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="search-integrator",
    )
    state = jnp.asarray([0.0]) if initial_state is None else initial_state
    problem = phx.control.ControlProblem(
        dynamics,
        time_grid,
        state,
        running_cost=lambda time, state, control, args: 0.05 * jnp.sum(control * control),
        terminal_cost=lambda time, state, args: (state[0] - 1.0) ** 2,
        problem_id="quadratic-control-search",
    )
    parameterization = phx.control.PiecewiseConstantControlParameterization(
        time_grid,
        (1,),
        parameterization_id="search-piecewise",
    )
    return problem, parameterization


def _search(population_size=8, max_generations=2):
    return phx.optim.DifferentialEvolutionSearch(
        population_size,
        max_generations,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
        design=phx.sampling.SobolDesign(scrambled=True),
    )


def test_search_is_reproducible_bounded_and_preserves_provenance():
    problem, parameterization = _quadratic_problem()
    shape = parameterization.parameter_shape
    lower = -jnp.ones(shape)
    upper = jnp.ones(shape)
    kwargs = {
        "coefficient_bounds": (lower, upper),
        "initial_coefficients": lower,
    }
    search = _search()

    result = phx.control.search_control(
        problem,
        parameterization,
        search,
        key=jr.key(40),
        **kwargs,
    )
    replay = phx.control.search_control(
        problem,
        parameterization,
        search,
        key=jr.key(40),
        **kwargs,
    )
    different = phx.control.search_control(
        problem,
        parameterization,
        search,
        key=jr.key(41),
        **kwargs,
    )

    np.testing.assert_array_equal(result.coefficients, replay.coefficients)
    np.testing.assert_array_equal(
        result.population_coefficients,
        replay.population_coefficients,
    )
    np.testing.assert_array_equal(
        result.population_objectives,
        replay.population_objectives,
    )
    np.testing.assert_array_equal(
        result.best_objective_history,
        replay.best_objective_history,
    )
    assert not np.array_equal(
        np.asarray(result.population_coefficients),
        np.asarray(different.population_coefficients),
    )
    assert result.population_coefficients.shape == (8,) + shape
    assert result.lower_bounds.shape == shape
    assert result.upper_bounds.shape == shape
    assert np.all(np.asarray(result.population_coefficients) >= np.asarray(lower))
    assert np.all(np.asarray(result.population_coefficients) <= np.asarray(upper))
    assert result.objective_evaluations == 8 * (result.generations + 1)
    assert result.valid_evaluations + result.invalid_candidates == (
        result.objective_evaluations
    )
    assert result.problem_id == problem.problem_id
    assert result.control_id == parameterization.parameterization_id
    assert result.parameterization_id == parameterization.parameterization_id
    assert result.approximation_id == parameterization.approximation_id
    assert result.time_id == problem.time_grid.time_id
    assert result.method_id == "bounded-differential-evolution-control-search"
    np.testing.assert_array_equal(jr.key_data(result.key), jr.key_data(jr.key(40)))
    assert np.all(np.diff(np.asarray(result.best_objective_history)) <= 0.0)


def test_coefficient_layout_and_bounds_are_strict_and_never_repaired():
    problem, parameterization = _quadratic_problem()
    search = _search(population_size=4, max_generations=0)
    shape = parameterization.parameter_shape

    with pytest.raises(ValueError, match="coefficient layout"):
        phx.control.search_control(
            problem,
            parameterization,
            search,
            key=jr.key(0),
            coefficient_bounds=(-jnp.ones((2,)), jnp.ones(shape)),
        )
    with pytest.raises(ValueError, match="smaller"):
        phx.control.search_control(
            problem,
            parameterization,
            search,
            key=jr.key(0),
            coefficient_bounds=(jnp.ones(shape), -jnp.ones(shape)),
        )
    with pytest.raises(ValueError, match="outside"):
        phx.control.search_control(
            problem,
            parameterization,
            search,
            key=jr.key(0),
            coefficient_bounds=(-jnp.ones(shape), jnp.ones(shape)),
            initial_coefficients=2.0 * jnp.ones(shape),
        )

    midpoint = phx.control.search_control(
        problem,
        parameterization,
        search,
        key=jr.key(0),
        coefficient_bounds=(-2.0 * jnp.ones(shape), jnp.ones(shape)),
    )
    np.testing.assert_array_equal(
        midpoint.population_coefficients[0],
        -0.5 * jnp.ones(shape),
    )


def test_invalid_candidates_are_counted_without_hiding_valid_rollouts():
    times = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 1.0]),
        time_id="invalid-search-time",
    )
    dynamics = make_discrete_control_dynamics(
        lambda time, state, control, args: state + control,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="invalid-search-integrator",
    )
    parameterization = phx.control.PiecewiseConstantControlParameterization(
        times,
        (1,),
        parameterization_id="invalid-search-piecewise",
    )
    shape = parameterization.parameter_shape
    bounds = (-jnp.ones(shape), jnp.ones(shape))
    partly_valid = phx.control.ControlProblem(
        dynamics,
        times,
        jnp.asarray([0.0]),
        running_cost=lambda time, state, control, args: jnp.where(
            control[0] <= 0.0,
            control[0] ** 2,
            jnp.nan,
        ),
        problem_id="partly-valid-control-search",
    )

    result = phx.control.search_control(
        partly_valid,
        parameterization,
        _search(population_size=8, max_generations=0),
        key=jr.key(42),
        coefficient_bounds=bounds,
        initial_coefficients=-0.5 * jnp.ones(shape),
    )
    assert 0 < result.invalid_candidates < result.objective_evaluations
    assert result.valid_evaluations == (
        result.objective_evaluations - result.invalid_candidates
    )
    assert jnp.isfinite(result.objective)
    assert result.termination_reason != "no_finite_candidates"

    invalid = phx.control.ControlProblem(
        dynamics,
        times,
        jnp.asarray([0.0]),
        running_cost=lambda time, state, control, args: jnp.asarray(jnp.nan),
        problem_id="invalid-control-search",
    )
    invalid_result = phx.control.search_control(
        invalid,
        parameterization,
        _search(population_size=4, max_generations=2),
        key=jr.key(43),
        coefficient_bounds=bounds,
    )
    assert invalid_result.invalid_candidates == 4
    assert invalid_result.valid_evaluations == 0
    assert invalid_result.termination_reason == "no_finite_candidates"
    assert jnp.isnan(invalid_result.objective)
    assert jnp.all(jnp.isinf(invalid_result.population_objectives))


def test_population_evaluation_preserves_case_and_coefficient_axes():
    initial_states = jnp.asarray([[0.0], [0.25]])
    problem, parameterization = _quadratic_problem(initial_state=initial_states)
    shape = problem.case_shape + parameterization.parameter_shape
    result = phx.control.search_control(
        problem,
        parameterization,
        _search(),
        key=jr.key(44),
        coefficient_bounds=(-jnp.ones(shape), jnp.ones(shape)),
    )

    assert result.population_coefficients.shape == (8,) + shape
    assert result.coefficients.shape == shape
    assert result.case_shape == problem.case_shape
    assert result.parameter_shape == parameterization.parameter_shape
    assert result.coefficient_shape == shape
    assert result.trajectory.states.shape == (2, 3, 1)
    assert result.trajectory.controls.shape == (2, 2, 1)
    np.testing.assert_allclose(
        result.objective,
        jnp.sum(result.evaluation.sampled_loss.total),
    )


def test_bspline_search_improves_objective_and_emits_a_local_control_seed():
    problem, _ = _quadratic_problem(num_steps=4)
    grid = phx.nn.models.BSplineGrid(
        jnp.asarray([0.0, 0.0, 0.5, 1.0, 1.0]),
        1,
    )
    parameterization = phx.control.BSplineControlParameterization(
        grid,
        (1,),
        parameterization_id="search-bspline",
    )
    shape = parameterization.parameter_shape
    lower = -jnp.ones(shape)
    upper = jnp.ones(shape)
    initial = lower
    initial_result = problem.evaluate(parameterization, initial)

    result = phx.control.search_control(
        problem,
        parameterization,
        _search(max_generations=3),
        key=jr.key(45),
        coefficient_bounds=(lower, upper),
        initial_coefficients=initial,
    )

    assert result.coefficients.shape == shape
    assert result.population_coefficients.shape == (8,) + shape
    assert result.objective < jnp.sum(initial_result.sampled_loss.total)
    assert result.trajectory.controls.shape == (
        problem.time_grid.num_steps,
        *problem.control_shape,
    )
    np.testing.assert_allclose(
        result.controls,
        parameterization.sample(
            result.coefficients,
            problem.time_grid.times[:-1],
        ),
    )

    local_parameterization = phx.control.PiecewiseConstantControlParameterization(
        problem.time_grid,
        problem.control_shape,
        parameterization_id="global-to-local-seed",
    )
    seeded = problem.evaluate(local_parameterization, result.trajectory.controls)
    np.testing.assert_allclose(
        seeded.trajectory.controls,
        result.trajectory.controls,
    )


def test_differential_search_rollouts_respect_piecewise_constant_jumps():
    grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 1.0, 2.5]), time_id="search-control-jump"
    )
    dynamics = make_differential_control_dynamics(
        lambda time, state, control, args: control,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="search-control-jump-dynamics",
    )
    problem = phx.control.ControlProblem(
        dynamics,
        grid,
        jnp.asarray([0.0]),
        terminal_cost=lambda time, state, args: (state[0] - 1.5) ** 2,
        problem_id="search-control-jump-problem",
    )
    parameterization = phx.control.PiecewiseConstantControlParameterization(
        grid,
        (1,),
        parameterization_id="search-control-jump-parameterization",
    )
    lower = jnp.asarray([[-0.01], [0.99]])
    upper = jnp.asarray([[0.01], [1.01]])

    result = phx.control.search_control(
        problem,
        parameterization,
        _search(population_size=8, max_generations=1),
        key=jr.key(20260807),
        coefficient_bounds=(lower, upper),
        initial_coefficients=jnp.asarray([[0.0], [1.0]]),
        solver=dfx.Euler(),
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=2.0,
    )

    expected_final = (
        result.coefficients[0, 0] * grid.durations[0]
        + result.coefficients[1, 0] * grid.durations[1]
    )
    assert result.successful
    np.testing.assert_allclose(
        result.trajectory.final_state,
        jnp.asarray([expected_final]),
        atol=1.0e-7,
    )
