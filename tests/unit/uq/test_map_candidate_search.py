#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _quadratic_problem(target=jnp.asarray([1.2, 2.2])):
    space = phx.uq.ParameterSpace(
        jnp.zeros((2,)),
        log_prior=lambda _: jnp.zeros(()),
    )
    return phx.uq.PosteriorProblem(
        space,
        lambda value: -jnp.sum((value - target) ** 2),
    )


def test_map_candidate_search_finds_exact_correlated_catalog_minimum():
    problem = _quadratic_problem()
    candidates = phx.optim.FiniteProductSpace(
        phx.optim.FiniteAxis(
            jnp.asarray(
                [
                    [-1.0, -1.0],
                    [1.0, 2.0],
                    [1.0, 3.0],
                    [3.0, 4.0],
                ]
            )
        )
    )

    result = phx.uq.search_map_candidates(
        problem,
        candidates,
        search=phx.optim.FiniteExhaustiveSearch(3),
    )

    assert isinstance(result, phx.uq.MAPCandidateSearchResult)
    assert result.valid
    assert result.termination_reason == "finite_minimum"
    np.testing.assert_array_equal(result.position, jnp.asarray([1.0, 2.0]))
    np.testing.assert_array_equal(result.parameters, result.position)
    assert result.objective == pytest.approx(0.08)
    assert result.log_density == pytest.approx(-0.08)
    assert result.flat_index == 1
    assert result.product_index == (1,)
    assert result.axis_paths == ("<root>",)
    assert result.product_shape == (4,)
    assert result.candidate_count == 4
    assert result.objective_evaluations == 4
    assert result.valid_evaluations == 4
    assert result.invalid_evaluations == 0
    assert result.effective_batch_size == 3
    assert result.method_id == "finite-exhaustive-map-candidate-search-v1"
    assert result.candidate_signature == candidates.signature()

    replay = phx.uq.search_map_candidates(problem, candidates)
    assert replay.flat_index == result.flat_index
    assert replay.candidate_signature == result.candidate_signature


def test_map_candidate_space_can_factor_independent_structured_coordinates():
    position = {"offset": jnp.asarray(0.0), "slope": jnp.zeros((2,))}
    space = phx.uq.ParameterSpace(position, log_prior=lambda _: jnp.zeros(()))
    problem = phx.uq.PosteriorProblem(
        space,
        lambda value: (
            -(
                (value["offset"] - 1.0) ** 2
                + jnp.sum((value["slope"] - jnp.asarray([2.0, 3.0])) ** 2)
            )
        ),
    )
    candidates = phx.optim.FiniteProductSpace(
        {
            "offset": phx.optim.FiniteAxis(jnp.asarray([-1.0, 1.0])),
            "slope": phx.optim.FiniteAxis(
                jnp.asarray([[0.0, 0.0], [2.0, 3.0], [4.0, 5.0]])
            ),
        }
    )

    result = phx.uq.search_map_candidates(
        problem,
        candidates,
        search=phx.optim.FiniteExhaustiveSearch(4),
    )

    assert result.product_shape == (2, 3)
    assert result.axis_paths == ("['offset']", "['slope']")
    assert result.flat_index == 4
    assert result.product_index == (1, 1)
    assert result.objective == pytest.approx(0.0)
    assert result.position is not None
    np.testing.assert_array_equal(result.position["offset"], 1.0)
    np.testing.assert_array_equal(result.position["slope"], jnp.asarray([2.0, 3.0]))


def test_map_candidate_search_reports_partial_and_complete_invalidity():
    space = phx.uq.ParameterSpace(
        jnp.asarray(0.0),
        log_prior=lambda _: jnp.zeros(()),
    )
    partial_problem = phx.uq.PosteriorProblem(
        space,
        lambda value: jnp.where(value < 0.0, jnp.nan, -((value - 1.0) ** 2)),
    )
    candidates = phx.optim.FiniteProductSpace(
        phx.optim.FiniteAxis(jnp.asarray([-1.0, 0.0, 1.0]))
    )

    partial = phx.uq.search_map_candidates(partial_problem, candidates)
    assert partial.valid
    assert partial.flat_index == 2
    assert partial.invalid_evaluations == 1
    assert partial.valid_evaluations == 2

    invalid_problem = phx.uq.PosteriorProblem(
        space,
        lambda _: jnp.asarray(jnp.nan),
    )
    invalid = phx.uq.search_map_candidates(
        invalid_problem,
        candidates,
        search=phx.optim.FiniteExhaustiveSearch(2),
    )
    assert not invalid.valid
    assert invalid.termination_reason == "no_finite_candidates"
    assert invalid.position is None
    assert invalid.parameters is None
    assert jnp.isnan(invalid.objective)
    assert jnp.isnan(invalid.log_density)
    assert invalid.flat_index == -1
    assert invalid.product_index == (-1,)
    assert invalid.objective_evaluations == 3
    assert invalid.valid_evaluations == 0
    assert invalid.invalid_evaluations == 3


def test_map_candidate_search_rejects_incompatible_candidate_points():
    problem = _quadratic_problem()

    wrong_structure = phx.optim.FiniteProductSpace(
        {"value": phx.optim.FiniteAxis(jnp.asarray([[1.0, 2.0]]))}
    )
    with pytest.raises(ValueError, match="PyTree structure"):
        phx.uq.search_map_candidates(problem, wrong_structure)

    wrong_shape = phx.optim.FiniteProductSpace(
        phx.optim.FiniteAxis(jnp.asarray([[1.0, 2.0, 3.0]]))
    )
    with pytest.raises(ValueError, match="must have shape"):
        phx.uq.search_map_candidates(problem, wrong_shape)

    wrong_dtype = phx.optim.FiniteProductSpace(
        phx.optim.FiniteAxis(jnp.asarray([[1, 2]], dtype=jnp.int32))
    )
    with pytest.raises(TypeError, match="dtype"):
        phx.uq.search_map_candidates(problem, wrong_dtype)

    nonfinite = phx.optim.FiniteProductSpace(
        phx.optim.FiniteAxis(jnp.asarray([[1.0, jnp.nan]]))
    )
    with pytest.raises(ValueError, match="finite"):
        phx.uq.search_map_candidates(problem, nonfinite)


def test_finite_screen_can_seed_local_map_without_claiming_continuous_optimality():
    problem = _quadratic_problem()
    candidates = phx.optim.FiniteProductSpace(
        phx.optim.FiniteAxis(jnp.asarray([[-2.0, -2.0], [1.0, 2.0], [3.0, 3.0]]))
    )

    screened = phx.uq.search_map_candidates(problem, candidates)
    assert screened.position is not None
    refined = phx.uq.find_map(problem, screened.position)

    assert refined.objective < screened.objective
    np.testing.assert_allclose(refined.position, jnp.asarray([1.2, 2.2]), atol=1e-7)
