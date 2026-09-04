#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.control.games import (
    DiscreteHJBIStatus,
    DiscreteZeroSumHJBIProblem,
    scalar_lq_hjbi_solution,
    ScalarLQHJBIStatus,
    solve_discrete_hjbi_reference,
)
from phydrax.control.stochastic import BoundedUniformGrid1D
from phydrax.dynamics import TimeGrid


def _zero_game_coefficient(time, state, minimizer, maximizer, args):
    del time, state, minimizer, maximizer, args
    return 0.0


def _matrix_payoff(time, state, minimizer, maximizer, args):
    del time, state, args
    return 2.0 * (1.0 - minimizer) * maximizer + minimizer * (1.0 - maximizer)


def _static_game(*, lower_order, upper_order, payoff, problem_id):
    grid = BoundedUniformGrid1D(-1.0, 1.0, 5)
    time_grid = TimeGrid(jnp.asarray([0.0, 0.1, 0.2]), time_id=problem_id)
    return DiscreteZeroSumHJBIProblem(
        grid,
        time_grid,
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([0.0, 1.0]),
        jnp.zeros(grid.num_points),
        jnp.zeros((time_grid.num_times, 2)),
        _zero_game_coefficient,
        _zero_game_coefficient,
        payoff,
        lower_order=lower_order,
        upper_order=upper_order,
        problem_id=problem_id,
    )


def test_declared_minmax_orders_and_all_four_action_selectors_are_independent():
    canonical = solve_discrete_hjbi_reference(
        _static_game(
            lower_order="max_min",
            upper_order="min_max",
            payoff=_matrix_payoff,
            problem_id="canonical-orders",
        )
    )
    mutated = solve_discrete_hjbi_reference(
        _static_game(
            lower_order="min_max",
            upper_order="max_min",
            payoff=_matrix_payoff,
            problem_id="mutated-orders",
        )
    )

    np.testing.assert_allclose(canonical.lower_values[0, 1:-1], 0.0)
    np.testing.assert_allclose(canonical.upper_values[0, 1:-1], 0.2)
    np.testing.assert_array_equal(canonical.lower_minimizer_selectors, 0)
    np.testing.assert_array_equal(canonical.lower_maximizer_selectors, 0)
    np.testing.assert_array_equal(canonical.upper_minimizer_selectors, 1)
    np.testing.assert_array_equal(canonical.upper_maximizer_selectors, 0)
    np.testing.assert_allclose(mutated.lower_values, canonical.upper_values)
    np.testing.assert_allclose(mutated.upper_values, canonical.lower_values)
    np.testing.assert_array_equal(
        mutated.lower_minimizer_selectors, canonical.upper_minimizer_selectors
    )
    np.testing.assert_array_equal(
        mutated.upper_maximizer_selectors, canonical.lower_maximizer_selectors
    )
    assert canonical.lower_order == "max_min"
    assert canonical.upper_order == "min_max"
    assert mutated.lower_order == "min_max"
    assert mutated.upper_order == "max_min"
    assert not bool(mutated.evidence.canonical_saddle_action_orders)
    assert not bool(mutated.saddle)
    assert mutated.status_label == "NONCANONICAL_SADDLE_ACTION_ORDERS"


@pytest.mark.parametrize("identical_order", ("max_min", "min_max"))
def test_identical_action_orders_cannot_turn_a_non_isaacs_game_into_a_saddle(
    identical_order,
):
    result = solve_discrete_hjbi_reference(
        _static_game(
            lower_order=identical_order,
            upper_order=identical_order,
            payoff=_matrix_payoff,
            problem_id=f"identical-{identical_order}-orders",
        ),
        refinement_absolute_tolerance=0.0,
        refinement_relative_tolerance=0.0,
        isaacs_absolute_tolerance=0.0,
        isaacs_relative_tolerance=0.0,
    )

    np.testing.assert_allclose(result.lower_values, result.upper_values)
    assert result.evidence.maximum_isaacs_gap == 0.0
    assert bool(result.evidence.refinement_passed)
    assert bool(result.evidence.isaacs_gap_passed)
    assert not bool(result.evidence.canonical_saddle_action_orders)
    assert not bool(result.saddle)
    assert int(result.status) == int(DiscreteHJBIStatus.NONCANONICAL_SADDLE_ACTION_ORDERS)
    assert result.status_label == "NONCANONICAL_SADDLE_ACTION_ORDERS"


def test_non_isaacs_game_reports_gap_and_never_receives_saddle_label():
    result = solve_discrete_hjbi_reference(
        _static_game(
            lower_order="max_min",
            upper_order="min_max",
            payoff=_matrix_payoff,
            problem_id="non-isaacs",
        ),
        isaacs_absolute_tolerance=0.0,
        isaacs_relative_tolerance=0.0,
    )

    assert result.evidence.maximum_isaacs_gap == pytest.approx(0.2)
    assert not bool(result.evidence.isaacs_gap_passed)
    assert not bool(result.saddle)
    assert int(result.status) == int(DiscreteHJBIStatus.ISAACS_GAP_EXCEEDS_TOLERANCE)
    assert result.status_label == "ISAACS_GAP_EXCEEDS_TOLERANCE"
    assert "SADDLE" not in result.status_label


def test_discrete_saddle_label_requires_operator_boundary_refinement_and_gap_gates():
    problem = _static_game(
        lower_order="max_min",
        upper_order="min_max",
        payoff=lambda time, state, minimizer, maximizer, args: (
            minimizer**2 - maximizer**2
        ),
        problem_id="saddle-game",
    )
    result = solve_discrete_hjbi_reference(
        problem,
        refinement_absolute_tolerance=0.0,
        refinement_relative_tolerance=0.0,
        isaacs_absolute_tolerance=0.0,
        isaacs_relative_tolerance=0.0,
    )

    np.testing.assert_allclose(result.lower_values, result.upper_values)
    np.testing.assert_array_equal(result.lower_minimizer_selectors, 0)
    np.testing.assert_array_equal(result.lower_maximizer_selectors, 0)
    assert bool(result.evidence.finite)
    assert bool(result.evidence.boundary_passed)
    assert bool(result.evidence.terminal_passed)
    assert bool(result.evidence.operator_passed)
    assert bool(result.evidence.action_orders_passed)
    assert bool(result.evidence.canonical_saddle_action_orders)
    assert bool(result.evidence.refinement_passed)
    assert bool(result.evidence.isaacs_gap_passed)
    assert bool(result.saddle)
    assert result.status_label == "SUCCESS_DISCRETE_SADDLE_REFERENCE"
    assert int(result.status) == int(DiscreteHJBIStatus.SUCCESS_DISCRETE_SADDLE_REFERENCE)
    assert result.evidence.scope == "declared-bounded-grid-discrete-residuals-only"


def test_single_action_positive_diffusion_reduces_to_scalar_heat_solution():
    sigma = 0.2
    terminal_time = 0.1
    grid = BoundedUniformGrid1D(-1.0, 1.0, 21)
    time_grid = TimeGrid(jnp.linspace(0.0, terminal_time, 41), time_id="diffusive-hjbi")
    points = np.asarray(grid.points)
    times = np.asarray(time_grid.times)
    boundary_trace = 1.0 + sigma * sigma * (terminal_time - times)
    problem = DiscreteZeroSumHJBIProblem(
        grid,
        time_grid,
        jnp.asarray([0.0]),
        jnp.asarray([0.0]),
        points * points,
        np.column_stack((boundary_trace, boundary_trace)),
        _zero_game_coefficient,
        lambda time, state, minimizer, maximizer, args: sigma,
        _zero_game_coefficient,
        lower_order="max_min",
        upper_order="min_max",
        problem_id="diffusive-hjbi",
    )

    result = solve_discrete_hjbi_reference(problem)
    expected = points[None, :] ** 2 + sigma * sigma * (terminal_time - times)[:, None]

    np.testing.assert_allclose(result.lower_values, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(result.upper_values, expected, rtol=2e-6, atol=2e-6)
    assert bool(result.saddle)


def test_refinement_failure_prevents_saddle_even_when_isaacs_gap_is_zero():
    grid = BoundedUniformGrid1D(-1.0, 1.0, 7)
    time_grid = TimeGrid(jnp.linspace(0.0, 0.1, 5), time_id="hjbi-refinement")
    terminal = np.asarray(grid.points) ** 4
    boundary = np.column_stack(
        (np.ones(time_grid.num_times), np.ones(time_grid.num_times))
    )
    problem = DiscreteZeroSumHJBIProblem(
        grid,
        time_grid,
        jnp.asarray([0.0]),
        jnp.asarray([0.0]),
        terminal,
        boundary,
        lambda time, state, minimizer, maximizer, args: 0.2,
        _zero_game_coefficient,
        _zero_game_coefficient,
        lower_order="max_min",
        upper_order="min_max",
        problem_id="hjbi-refinement",
    )

    result = solve_discrete_hjbi_reference(
        problem,
        refinement_absolute_tolerance=0.0,
        refinement_relative_tolerance=0.0,
    )

    assert bool(result.evidence.isaacs_gap_passed)
    assert not bool(result.evidence.refinement_passed)
    assert not bool(result.saddle)
    assert result.status_label == "REFINEMENT_GATE_FAILED"
    assert int(result.status) == int(DiscreteHJBIStatus.REFINEMENT_GATE_FAILED)


def test_invalid_action_order_is_rejected_before_coefficient_callbacks_execute():
    calls = []

    def observed_coefficient(time, state, minimizer, maximizer, args):
        calls.append((time, state, minimizer, maximizer, args))
        return 0.0

    grid = BoundedUniformGrid1D(-1.0, 1.0, 5)
    time_grid = TimeGrid(jnp.asarray([0.0, 0.1]), time_id="invalid-hjbi-order")
    with pytest.raises(ValueError, match="max_min.*min_max"):
        DiscreteZeroSumHJBIProblem(
            grid,
            time_grid,
            jnp.asarray([0.0]),
            jnp.asarray([0.0]),
            jnp.zeros(grid.num_points),
            jnp.zeros((time_grid.num_times, 2)),
            observed_coefficient,
            observed_coefficient,
            observed_coefficient,
            lower_order="simultaneous",
            upper_order="min_max",
            problem_id="invalid-hjbi-order",
        )
    assert calls == []


def test_corrected_scalar_lq_hjbi_formula_accepts_well_posed_gamma_cases():
    time_grid = TimeGrid(jnp.asarray([0.0, 0.1, 0.2]), time_id="scalar-lq-hjbi")

    for gamma in (2.0, 1.0, 0.8):
        solution = scalar_lq_hjbi_solution(
            time_grid,
            terminal_weight=1.0,
            gamma=gamma,
        )
        times = np.asarray(time_grid.times)
        denominator = 1.0 + (1.0 - gamma**-2) * (times[-1] - times)
        coefficient = 1.0 / denominator
        np.testing.assert_allclose(solution.denominator, denominator, rtol=1e-12)
        np.testing.assert_allclose(solution.value_coefficient, coefficient, rtol=1e-12)
        np.testing.assert_allclose(solution.minimizer_feedback_gain, -coefficient)
        np.testing.assert_allclose(
            solution.maximizer_feedback_gain, coefficient / gamma**2
        )
        assert bool(solution.well_posed)
        assert int(solution.status) == int(
            ScalarLQHJBIStatus.SUCCESS_ANALYTIC_SCALAR_LQ_HJBI
        )
    np.testing.assert_array_equal(
        scalar_lq_hjbi_solution(time_grid, terminal_weight=1.0, gamma=1.0).denominator,
        1.0,
    )


def test_scalar_lq_hjbi_rejects_nonpositive_gamma_and_singular_riccati_horizon():
    short_grid = TimeGrid(jnp.asarray([0.0, 0.2]), time_id="gamma-gate")
    with pytest.raises(ValueError, match="strictly positive"):
        scalar_lq_hjbi_solution(short_grid, terminal_weight=1.0, gamma=0.0)

    singular_grid = TimeGrid(jnp.asarray([0.0, 1.0]), time_id="singular-riccati")
    with pytest.raises(ValueError, match="denominator.*strictly positive"):
        scalar_lq_hjbi_solution(
            singular_grid,
            terminal_weight=1.0,
            gamma=1.0 / np.sqrt(2.0),
        )

    well_posed_subunit = scalar_lq_hjbi_solution(
        short_grid,
        terminal_weight=1.0,
        gamma=0.8,
    )
    assert np.all(np.asarray(well_posed_subunit.denominator) > 0.0)
