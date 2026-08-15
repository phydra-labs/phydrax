#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from tests.unit.control.test_global_control_search import _quadratic_problem


def _catalog(rows):
    return phx.optim.FiniteProductSpace(
        phx.optim.FiniteAxis(jnp.asarray(rows, dtype=float))
    )


def test_control_candidate_search_finds_exact_catalog_minimum_and_reconstructs():
    problem, parameterization = _quadratic_problem()
    candidates = _catalog(
        [
            [[-1.0], [-1.0]],
            [[0.5], [0.5]],
            [[1.0], [0.0]],
        ]
    )

    result = phx.control.search_control_candidates(
        problem,
        parameterization,
        candidates,
        search=phx.optim.FiniteExhaustiveSearch(2),
    )

    assert isinstance(result, phx.control.ControlCandidateSearchResult)
    assert result.valid
    assert bool(result.successful)
    assert result.termination_reason == "finite_minimum"
    np.testing.assert_array_equal(result.coefficients, jnp.asarray([[0.5], [0.5]]))
    assert result.objective == pytest.approx(0.0125)
    assert result.flat_index == 1
    assert result.product_index == (1,)
    assert result.axis_paths == ("<root>",)
    assert result.product_shape == (3,)
    assert result.candidate_count == 3
    assert result.objective_evaluations == 3
    assert result.valid_evaluations == 3
    assert result.invalid_candidates == 0
    assert result.winner_evaluations == 1
    assert result.total_control_evaluations == 4
    assert result.effective_batch_size == 2
    assert result.problem_id == problem.problem_id
    assert result.dynamics_id == problem.dynamics.dynamics_id
    assert result.time_id == problem.time_grid.time_id
    assert result.control_id == parameterization.parameterization_id
    assert result.parameterization_id == parameterization.parameterization_id
    assert result.approximation_id == parameterization.approximation_id
    assert result.method_id == "finite-exhaustive-control-candidate-search-v1"
    assert result.candidate_signature == candidates.signature()
    assert result.control_shape == (1,)
    assert result.case_shape == ()
    assert result.parameter_shape == (2, 1)
    assert result.coefficient_shape == (2, 1)
    assert result.trajectory is result.evaluation.trajectory
    np.testing.assert_array_equal(result.controls, result.evaluation.trajectory.controls)


def test_control_candidate_search_counts_infeasible_candidates_as_invalid():
    base, parameterization = _quadratic_problem()
    problem = phx.control.ControlProblem(
        base.dynamics,
        base.time_grid,
        base.initial_state,
        running_cost=base.running_cost,
        terminal_cost=base.terminal_cost,
        path_constraints=(lambda time, state, control, args: control[0] - 0.6,),
        problem_id="constrained-candidate-search",
    )
    candidates = _catalog(
        [
            [[1.0], [1.0]],
            [[0.5], [0.5]],
            [[0.8], [0.0]],
        ]
    )

    result = phx.control.search_control_candidates(problem, parameterization, candidates)

    assert result.valid
    assert result.flat_index == 1
    assert result.invalid_candidates == 2
    assert result.valid_evaluations == 1
    assert result.objective_evaluations == 3
    assert result.winner_evaluations == 1


def test_control_candidate_search_has_explicit_all_invalid_result():
    base, parameterization = _quadratic_problem()
    problem = phx.control.ControlProblem(
        base.dynamics,
        base.time_grid,
        base.initial_state,
        running_cost=base.running_cost,
        terminal_cost=base.terminal_cost,
        path_constraints=(lambda time, state, control, args: jnp.asarray(1.0),),
        problem_id="invalid-candidate-search",
    )
    candidates = _catalog(
        [
            [[0.0], [0.0]],
            [[0.5], [0.5]],
        ]
    )

    result = phx.control.search_control_candidates(
        problem,
        parameterization,
        candidates,
        search=phx.optim.FiniteExhaustiveSearch(8),
    )

    assert not result.valid
    assert not bool(result.successful)
    assert result.termination_reason == "no_finite_candidates"
    assert result.coefficients is None
    assert result.evaluation is None
    assert jnp.isnan(result.objective)
    assert result.flat_index == -1
    assert result.product_index == (-1,)
    assert result.objective_evaluations == 2
    assert result.valid_evaluations == 0
    assert result.invalid_candidates == 2
    assert result.winner_evaluations == 0
    assert result.total_control_evaluations == 2
    assert result.control_id is None
    with pytest.raises(RuntimeError, match="no valid trajectory"):
        _ = result.trajectory
    with pytest.raises(RuntimeError, match="no valid trajectory"):
        _ = result.controls


def test_control_candidate_search_preserves_case_and_coefficient_axes():
    initial_state = jnp.asarray([[0.0], [0.5]])
    problem, parameterization = _quadratic_problem(initial_state=initial_state)
    candidates = _catalog(
        [
            [[[0.0], [0.0]], [[0.0], [0.0]]],
            [[[0.5], [0.5]], [[0.25], [0.25]]],
        ]
    )

    result = phx.control.search_control_candidates(problem, parameterization, candidates)

    assert result.case_shape == (2,)
    assert result.parameter_shape == (2, 1)
    assert result.coefficient_shape == (2, 2, 1)
    assert result.coefficients is not None
    assert result.coefficients.shape == (2, 2, 1)
    assert result.trajectory.states.shape == (2, 3, 1)
    assert result.trajectory.controls.shape == (2, 2, 1)


def test_control_candidate_search_has_stable_first_index_ties():
    problem, parameterization = _quadratic_problem()
    repeated = jnp.asarray([[0.5], [0.5]])
    candidates = _catalog([repeated, repeated, [[1.0], [0.0]]])

    result = phx.control.search_control_candidates(
        problem,
        parameterization,
        candidates,
        search=phx.optim.FiniteExhaustiveSearch(2),
    )

    assert result.flat_index == 0


def test_control_candidate_space_rejects_ambiguous_or_invalid_coefficients():
    problem, parameterization = _quadratic_problem()

    wrong_shape = _catalog([[[0.0]], [[1.0]]])
    with pytest.raises(ValueError, match="coefficient shape"):
        phx.control.search_control_candidates(problem, parameterization, wrong_shape)

    factored = phx.optim.FiniteProductSpace(
        (
            phx.optim.FiniteAxis(jnp.asarray([0.0, 1.0])),
            phx.optim.FiniteAxis(jnp.asarray([0.0, 1.0])),
        )
    )
    with pytest.raises(ValueError, match="one coefficient array"):
        phx.control.search_control_candidates(problem, parameterization, factored)

    integer = phx.optim.FiniteProductSpace(
        phx.optim.FiniteAxis(jnp.zeros((2, 2, 1), dtype=jnp.int32))
    )
    with pytest.raises(TypeError, match="floating-point"):
        phx.control.search_control_candidates(problem, parameterization, integer)

    nonfinite = _catalog([[[0.0], [jnp.nan]]])
    with pytest.raises(ValueError, match="finite"):
        phx.control.search_control_candidates(problem, parameterization, nonfinite)

    incompatible_parameterization = phx.control.PiecewiseConstantControlParameterization(
        problem.time_grid,
        (2,),
        parameterization_id="wrong-control-shape",
    )
    incompatible_candidates = _catalog([jnp.zeros((2, 2))])
    with pytest.raises(ValueError, match="control_shape"):
        phx.control.search_control_candidates(
            problem,
            incompatible_parameterization,
            incompatible_candidates,
        )
