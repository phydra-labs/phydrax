#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


def _multimodal_problem(initial=-1.5):
    space = phx.uq.ParameterSpace(
        jnp.asarray(initial),
        log_prior=lambda value: jnp.asarray(0.0),
    )

    def log_likelihood(value):
        local_well = (value + 1.5) ** 2 + 0.2
        global_well = (value - 1.0) ** 2
        return -jnp.minimum(local_well, global_well)

    return phx.uq.PosteriorProblem(space, log_likelihood)


def test_global_search_finds_better_mode_and_composes_with_local_map():
    problem = _multimodal_problem()
    search = phx.optim.DifferentialEvolutionSearch(
        32,
        20,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
        design=phx.sampling.SobolDesign(scrambled=True),
    )

    global_mode = phx.uq.search_map(
        problem,
        search,
        key=jr.key(20),
        position_bounds=(jnp.asarray(-3.0), jnp.asarray(3.0)),
    )
    local_mode = phx.uq.find_map(
        problem,
        global_mode.position,
        gradient_tolerance=1e-8,
    )

    assert isinstance(global_mode, phx.uq.MAPSearchResult)
    assert global_mode.position > 0.5
    assert global_mode.objective < 5e-3
    assert local_mode.converged
    assert local_mode.position == pytest.approx(1.0, abs=1e-7)
    assert local_mode.objective <= global_mode.objective + 1e-10
    assert global_mode.objective_evaluations == 32 * (global_mode.generations + 1)
    assert global_mode.best_objective_history.shape == (global_mode.generations + 1,)
    assert np.all(np.diff(np.asarray(global_mode.best_objective_history)) <= 0.0)


def test_search_preserves_nested_positions_bijectors_and_population_axes():
    initial = {
        "positive": jnp.asarray(0.0, dtype=jnp.float32),
        "bounded": jnp.asarray(0.0),
        "vector": jnp.asarray([0.0, 0.0]),
    }
    space = phx.uq.ParameterSpace(
        initial,
        log_prior=lambda value: jnp.asarray(0.0),
        bijectors={
            "positive": phx.uq.ExpBijector(),
            "bounded": phx.uq.SigmoidIntervalBijector(-2.0, 3.0),
            "vector": phx.uq.IdentityBijector(),
        },
    )
    problem = phx.uq.PosteriorProblem(
        space,
        lambda value: (
            -(
                (value["positive"] - 1.5) ** 2
                + (value["bounded"] - 0.25) ** 2
                + jnp.sum((value["vector"] - jnp.asarray([0.2, -0.4])) ** 2)
            )
        ),
    )
    lower = {"positive": -2.0, "bounded": -3.0, "vector": -1.0}
    upper = {"positive": 2.0, "bounded": 3.0, "vector": 1.0}
    search = phx.optim.DifferentialEvolutionSearch(8, 2)

    result = phx.uq.search_map(
        problem,
        search,
        key=jr.key(21),
        position_bounds=(lower, upper),
    )

    assert result.position["positive"].dtype == initial["positive"].dtype
    assert result.position["vector"].shape == (2,)
    assert result.population_positions["positive"].shape == (8,)
    assert result.population_positions["bounded"].shape == (8,)
    assert result.population_positions["vector"].shape == (8, 2)
    assert result.lower_bounds["vector"].shape == (2,)
    assert result.upper_bounds["vector"].shape == (2,)
    assert result.parameters["positive"] > 0.0
    assert -2.0 < result.parameters["bounded"] < 3.0
    assert result.objective == pytest.approx(
        problem.negative_log_density(result.position),
    )
    assert result.log_density == pytest.approx(-result.objective)
    assert result.population_objectives.shape == (8,)


def test_position_bounds_and_initial_position_are_strictly_validated():
    problem = _multimodal_problem(initial=0.0)
    search = phx.optim.DifferentialEvolutionSearch(4, 0)

    with pytest.raises(TypeError, match="tuple"):
        phx.uq.search_map(
            problem,
            search,
            key=jr.key(0),
            position_bounds=cast(tuple[object, object], [-1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="PyTree structure"):
        phx.uq.search_map(
            problem,
            search,
            key=jr.key(0),
            position_bounds=({"x": -1.0}, {"x": 1.0}),
        )
    with pytest.raises(ValueError, match="finite"):
        phx.uq.search_map(
            problem,
            search,
            key=jr.key(0),
            position_bounds=(jnp.asarray(jnp.nan), jnp.asarray(1.0)),
        )
    with pytest.raises(ValueError, match="smaller"):
        phx.uq.search_map(
            problem,
            search,
            key=jr.key(0),
            position_bounds=(jnp.asarray(1.0), jnp.asarray(-1.0)),
        )
    with pytest.raises(ValueError, match="outside"):
        phx.uq.search_map(
            problem,
            search,
            key=jr.key(0),
            position_bounds=(jnp.asarray(-1.0), jnp.asarray(1.0)),
            initial_position=jnp.asarray(2.0),
        )

    vector_problem = phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(
            jnp.zeros((2,)),
            log_prior=lambda value: jnp.asarray(0.0),
        ),
        lambda value: -jnp.sum(value * value),
    )
    with pytest.raises(ValueError, match="shape"):
        phx.uq.search_map(
            vector_problem,
            search,
            key=jr.key(0),
            position_bounds=(jnp.zeros((3,)), jnp.ones((2,))),
        )


def test_invalid_posterior_evaluations_are_counted_without_rejecting_search():
    space = phx.uq.ParameterSpace(
        jnp.asarray(0.0),
        log_prior=lambda value: jnp.asarray(0.0),
    )
    partly_valid = phx.uq.PosteriorProblem(
        space,
        lambda value: jnp.where(
            jnp.abs(value) < 0.1,
            jnp.nan,
            -((value - 0.8) ** 2),
        ),
    )
    search = phx.optim.DifferentialEvolutionSearch(8, 0)
    result = phx.uq.search_map(
        partly_valid,
        search,
        key=jr.key(22),
        position_bounds=(jnp.asarray(-2.0), jnp.asarray(2.0)),
    )

    assert result.invalid_evaluations >= 1
    assert result.termination_reason != "no_finite_candidates"
    assert jnp.isfinite(result.objective)

    invalid = phx.uq.PosteriorProblem(
        space,
        lambda value: jnp.asarray(jnp.nan),
    )
    invalid_result = phx.uq.search_map(
        invalid,
        search,
        key=jr.key(22),
        position_bounds=(jnp.asarray(-2.0), jnp.asarray(2.0)),
    )
    assert not invalid_result.population_converged
    assert invalid_result.termination_reason == "no_finite_candidates"
    assert invalid_result.invalid_evaluations == 8
    assert jnp.isnan(invalid_result.objective)
    assert jnp.all(jnp.isinf(invalid_result.population_objectives))


def test_search_replays_from_the_same_root_key():
    problem = _multimodal_problem(initial=0.0)
    search = phx.optim.DifferentialEvolutionSearch(
        8,
        2,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
        design=phx.sampling.SobolDesign(scrambled=True),
    )
    kwargs = {
        "position_bounds": (jnp.asarray(-3.0), jnp.asarray(3.0)),
    }

    result = phx.uq.search_map(problem, search, key=jr.key(23), **kwargs)
    replay = phx.uq.search_map(problem, search, key=jr.key(23), **kwargs)
    different = phx.uq.search_map(problem, search, key=jr.key(24), **kwargs)

    np.testing.assert_array_equal(result.position, replay.position)
    np.testing.assert_array_equal(
        result.population_positions,
        replay.population_positions,
    )
    np.testing.assert_array_equal(
        result.population_objectives,
        replay.population_objectives,
    )
    assert not np.array_equal(
        np.asarray(result.population_positions),
        np.asarray(different.population_positions),
    )
