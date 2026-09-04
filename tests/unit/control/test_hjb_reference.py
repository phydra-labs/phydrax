#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.control.stochastic import (
    BoundedUniformGrid1D,
    DiscreteHJBProblem,
    DiscreteHJBStatus,
    refine_discrete_hjb_reference,
    solve_discrete_hjb_reference,
)
from phydrax.dynamics import TimeGrid


def _zero_drift(time, state, action, args):
    del time, state, action, args
    return 0.0


def _zero_diffusion(time, state, action, args):
    del time, state, action, args
    return 0.0


def _linear_cost_problem(*, cost, problem_id):
    grid = BoundedUniformGrid1D(-1.0, 1.0, 9)
    time_grid = TimeGrid(jnp.linspace(0.0, 0.2, 5), time_id=problem_id)
    terminal = np.asarray(grid.points)
    boundary = np.column_stack(
        (
            -np.ones(time_grid.num_times) + cost * (0.2 - np.asarray(time_grid.times)),
            np.ones(time_grid.num_times) + cost * (0.2 - np.asarray(time_grid.times)),
        )
    )
    return DiscreteHJBProblem(
        grid,
        time_grid,
        jnp.asarray([0.0]),
        terminal,
        boundary,
        _zero_drift,
        _zero_diffusion,
        lambda time, state, action, args: cost,
        problem_id=problem_id,
    )


def test_deterministic_reduction_matches_running_cost_and_earns_discrete_label():
    problem = _linear_cost_problem(cost=2.0, problem_id="deterministic-hjb")
    result = solve_discrete_hjb_reference(problem)
    times = np.asarray(problem.time_grid.times)
    expected = (
        np.asarray(problem.spatial_grid.points)[None, :]
        + 2.0 * (times[-1] - times)[:, None]
    )

    np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-12)
    assert bool(result.successful)
    assert int(result.status) == int(DiscreteHJBStatus.SUCCESS_DISCRETE_REFERENCE)
    assert result.status_label == "SUCCESS_DISCRETE_REFERENCE"
    assert result.evidence.scope == "declared-bounded-grid-discrete-residuals-only"
    assert result.evidence.maximum_operator_residual < 1e-12
    assert result.evidence.maximum_boundary_residual == 0.0
    assert result.evidence.maximum_terminal_residual == 0.0
    assert bool(result.evidence.refinement_passed)


def test_positive_diffusion_reduction_matches_quadratic_heat_solution():
    sigma = 0.2
    terminal_time = 0.1
    grid = BoundedUniformGrid1D(-1.0, 1.0, 21)
    time_grid = TimeGrid(jnp.linspace(0.0, terminal_time, 41), time_id="diffusive-hjb")
    points = np.asarray(grid.points)
    times = np.asarray(time_grid.times)
    terminal = points * points
    boundary_trace = 1.0 + sigma * sigma * (terminal_time - times)
    boundary = np.column_stack((boundary_trace, boundary_trace))
    problem = DiscreteHJBProblem(
        grid,
        time_grid,
        jnp.asarray([0.0]),
        terminal,
        boundary,
        _zero_drift,
        lambda time, state, action, args: sigma,
        lambda time, state, action, args: 0.0,
        problem_id="diffusive-hjb",
    )

    result = solve_discrete_hjb_reference(problem)
    expected = points[None, :] ** 2 + sigma * sigma * (terminal_time - times)[:, None]

    np.testing.assert_allclose(result.values, expected, rtol=2e-6, atol=2e-6)
    assert result.evidence.maximum_courant_number < 1.0
    assert result.evidence.minimum_monotonicity_margin > 0.0
    assert bool(result.successful)


def test_upwind_drift_uses_forward_difference_for_positive_generator_drift():
    grid = BoundedUniformGrid1D(-1.0, 1.0, 5)
    time_grid = TimeGrid(jnp.asarray([0.0, 0.1]), time_id="upwind-positive")
    terminal = np.asarray(grid.points) ** 2
    boundary = np.asarray([[1.0, 1.0], [1.0, 1.0]])
    problem = DiscreteHJBProblem(
        grid,
        time_grid,
        jnp.asarray([0.0]),
        terminal,
        boundary,
        lambda time, state, action, args: 1.0,
        _zero_diffusion,
        lambda time, state, action, args: 0.0,
        problem_id="upwind-positive",
    )

    result = solve_discrete_hjb_reference(
        problem,
        refinement_absolute_tolerance=10.0,
        refinement_relative_tolerance=0.0,
    )
    forward = (terminal[2:] - terminal[1:-1]) / grid.spacing
    expected = terminal[1:-1] + 0.1 * forward
    np.testing.assert_allclose(result.values[0, 1:-1], expected, rtol=1e-12, atol=1e-12)


def test_upwind_drift_uses_backward_difference_for_negative_generator_drift():
    grid = BoundedUniformGrid1D(-1.0, 1.0, 5)
    time_grid = TimeGrid(jnp.asarray([0.0, 0.1]), time_id="upwind-negative")
    terminal = np.asarray(grid.points) ** 2
    boundary = np.asarray([[1.0, 1.0], [1.0, 1.0]])
    problem = DiscreteHJBProblem(
        grid,
        time_grid,
        jnp.asarray([0.0]),
        terminal,
        boundary,
        lambda time, state, action, args: -1.0,
        _zero_diffusion,
        lambda time, state, action, args: 0.0,
        problem_id="upwind-negative",
    )

    result = solve_discrete_hjb_reference(
        problem,
        refinement_absolute_tolerance=10.0,
        refinement_relative_tolerance=0.0,
    )
    backward = (terminal[1:-1] - terminal[:-2]) / grid.spacing
    expected = terminal[1:-1] - 0.1 * backward
    np.testing.assert_allclose(result.values[0, 1:-1], expected, rtol=1e-12, atol=1e-12)


def test_terminal_boundary_data_and_action_selector_are_explicit():
    grid = BoundedUniformGrid1D(0.0, 1.0, 5)
    time_grid = TimeGrid(jnp.asarray([0.0, 0.1, 0.2]), time_id="action-hjb")
    terminal = np.asarray(grid.points)
    boundary = np.column_stack(
        (np.zeros(time_grid.num_times), np.ones(time_grid.num_times))
    )
    actions = jnp.asarray([-1.0, 0.5, 2.0])
    problem = DiscreteHJBProblem(
        grid,
        time_grid,
        actions,
        terminal,
        boundary,
        _zero_drift,
        _zero_diffusion,
        lambda time, state, action, args: (action - 0.5) ** 2,
        problem_id="action-hjb",
    )

    result = solve_discrete_hjb_reference(problem)

    np.testing.assert_array_equal(result.values[-1], terminal)
    np.testing.assert_array_equal(result.values[:, 0], boundary[:, 0])
    np.testing.assert_array_equal(result.values[:, -1], boundary[:, 1])
    np.testing.assert_array_equal(result.action_selectors, 1)
    np.testing.assert_allclose(result.selected_actions, 0.5)
    assert result.evidence.maximum_action_minimum_residual == 0.0

    incompatible = boundary.copy()
    incompatible[-1, 0] = 0.25
    with pytest.raises(ValueError, match="incompatible"):
        DiscreteHJBProblem(
            grid,
            time_grid,
            actions,
            terminal,
            incompatible,
            _zero_drift,
            _zero_diffusion,
            lambda time, state, action, args: 0.0,
            problem_id="incompatible-hjb",
        )


def test_unsupported_callback_shape_and_nonmonotone_step_fail_before_integration():
    grid = BoundedUniformGrid1D(-1.0, 1.0, 5)
    time_grid = TimeGrid(jnp.asarray([0.0, 0.1]), time_id="invalid-hjb")
    terminal = np.zeros(grid.num_points)
    boundary = np.zeros((time_grid.num_times, 2))
    vector_problem = DiscreteHJBProblem(
        grid,
        time_grid,
        jnp.asarray([0.0]),
        terminal,
        boundary,
        lambda time, state, action, args: jnp.asarray([0.0]),
        _zero_diffusion,
        lambda time, state, action, args: 0.0,
        problem_id="vector-callback-hjb",
    )
    with pytest.raises(ValueError, match="return a scalar"):
        solve_discrete_hjb_reference(vector_problem)

    nonmonotone_problem = DiscreteHJBProblem(
        grid,
        time_grid,
        jnp.asarray([0.0]),
        terminal,
        boundary,
        lambda time, state, action, args: 10.0,
        _zero_diffusion,
        lambda time, state, action, args: 0.0,
        problem_id="nonmonotone-hjb",
    )
    with pytest.raises(ValueError, match="monotone"):
        solve_discrete_hjb_reference(nonmonotone_problem)


def test_refinement_result_exposes_gate_failure_without_broad_claim_language():
    grid = BoundedUniformGrid1D(-1.0, 1.0, 7)
    time_grid = TimeGrid(jnp.linspace(0.0, 0.1, 5), time_id="refinement-hjb")
    terminal = np.asarray(grid.points) ** 4
    boundary = np.column_stack(
        (np.ones(time_grid.num_times), np.ones(time_grid.num_times))
    )
    problem = DiscreteHJBProblem(
        grid,
        time_grid,
        jnp.asarray([0.0]),
        terminal,
        boundary,
        lambda time, state, action, args: 0.2,
        _zero_diffusion,
        lambda time, state, action, args: 0.0,
        problem_id="refinement-hjb",
    )

    refinement = refine_discrete_hjb_reference(
        problem,
        refinement_absolute_tolerance=0.0,
        refinement_relative_tolerance=0.0,
    )

    assert refinement.common_grid_difference.shape == refinement.result.values.shape
    assert refinement.refined_values.shape == (
        4 * (time_grid.num_times - 1) + 1,
        2 * (grid.num_points - 1) + 1,
    )
    assert not bool(refinement.passed)
    assert refinement.status_label == "REFINEMENT_GATE_FAILED"
    assert int(refinement.status) == int(DiscreteHJBStatus.REFINEMENT_GATE_FAILED)
    assert "GLOBAL" not in refinement.status_label
    assert "VISCOSITY" not in refinement.status_label
