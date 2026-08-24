#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax.uq._gp_backend import (
    exact_gp_conditioner_from_covariances,
    exact_gp_predict_diagonal_from_covariances,
)
from phydrax.uq._map_gp_search import _expected_improvement, _posterior


def _search(max_evaluations=12, *, noise_scale=0.0):
    return phx.uq.GaussianProcessMAPSearch(
        max_evaluations,
        surrogate=phx.uq.GaussianProcessLikelihoodState(
            kernel=phx.kernels.Matern52Kernel(length_scale=0.25),
            noise_scale=noise_scale,
            jitter=1e-8,
        ),
        initial_evaluations=4,
        candidate_count=64,
        design=phx.sampling.SobolDesign(scrambled=True),
    )


def _problem(initial=-1.5):
    space = phx.uq.ParameterSpace(
        jnp.asarray(initial),
        log_prior=lambda value: jnp.asarray(0.0),
    )

    def log_likelihood(value):
        local_well = (value + 1.5) ** 2 + 0.2
        global_well = (value - 1.0) ** 2
        return -jnp.minimum(local_well, global_well)

    return phx.uq.PosteriorProblem(space, log_likelihood)


def test_configuration_rejects_invalid_budget_surrogate_and_noise_shape():
    surrogate = phx.uq.GaussianProcessLikelihoodState(noise_scale=0.0)
    with pytest.raises(ValueError, match="max_evaluations"):
        phx.uq.GaussianProcessMAPSearch(1, surrogate=surrogate)
    with pytest.raises(ValueError, match="initial_evaluations"):
        phx.uq.GaussianProcessMAPSearch(
            4,
            surrogate=surrogate,
            initial_evaluations=1,
        )
    with pytest.raises(ValueError, match="candidate_count"):
        phx.uq.GaussianProcessMAPSearch(
            8,
            surrogate=surrogate,
            candidate_count=1,
        )
    with pytest.raises(ValueError, match="scalar"):
        phx.uq.GaussianProcessMAPSearch(
            8,
            surrogate=phx.uq.GaussianProcessLikelihoodState(
                noise_scale=jnp.asarray([0.1, 0.2])
            ),
        )


def test_raw_noise_standardization_is_affine_equivariant():
    points = jnp.asarray([[0.0], [0.4], [1.0]])
    queries = jnp.asarray([[0.2], [0.8]])
    values = jnp.asarray([1.0, -0.2, 0.7])
    base = phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.Matern52Kernel(length_scale=0.3),
        noise_scale=0.2,
        jitter=1e-8,
    )
    scaled = phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.Matern52Kernel(length_scale=0.3),
        noise_scale=2.0,
        jitter=1e-8,
    )

    mean, standard_deviation, usable = _posterior(points, values, queries, base)
    scaled_mean, scaled_standard_deviation, scaled_usable = _posterior(
        points,
        10.0 * values + 5.0,
        queries,
        scaled,
    )

    assert usable
    assert scaled_usable
    np.testing.assert_allclose(scaled_mean, 10.0 * mean + 5.0, rtol=1e-5)
    np.testing.assert_allclose(
        scaled_standard_deviation,
        10.0 * standard_deviation,
        rtol=1e-5,
    )


def test_diagonal_prediction_matches_dense_conditioner():
    cholesky = jnp.linalg.cholesky(jnp.asarray([[1.2, 0.2], [0.2, 1.1]]))
    cross = jnp.asarray([[0.5, 0.1], [0.2, 0.4], [0.1, 0.3]])
    query_covariance = jnp.asarray([[1.0, 0.2, 0.1], [0.2, 1.1, 0.3], [0.1, 0.3, 0.9]])
    residual = jnp.asarray([0.4, -0.2])
    projection, _covariance, dense_variance = exact_gp_conditioner_from_covariances(
        cholesky,
        cross,
        query_covariance,
    )
    mean, variance = exact_gp_predict_diagonal_from_covariances(
        cholesky,
        cross,
        jnp.diag(query_covariance),
        residual,
    )

    np.testing.assert_allclose(mean, projection @ residual)
    np.testing.assert_allclose(variance, dense_variance)


def test_expected_improvement_handles_zero_variance_analytically():
    utility = _expected_improvement(
        jnp.asarray([0.5, 1.0, 1.5]),
        jnp.zeros((3,)),
        jnp.asarray(1.0),
        0.0,
    )
    np.testing.assert_array_equal(utility, jnp.asarray([0.5, 0.0, 0.0]))


def test_search_replays_and_retains_complete_evidence():
    problem = _problem()
    search = _search()
    kwargs = {
        "position_bounds": (jnp.asarray(-3.0), jnp.asarray(3.0)),
    }

    result = phx.uq.search_map(problem, search, key=jr.key(31), **kwargs)
    replay = phx.uq.search_map(problem, search, key=jr.key(31), **kwargs)
    different = phx.uq.search_map(problem, search, key=jr.key(32), **kwargs)

    assert isinstance(result, phx.uq.GaussianProcessMAPSearchResult)
    assert result.valid
    assert result.objective_evaluations == search.max_evaluations
    assert result.raw_objectives.shape == (search.max_evaluations,)
    assert result.evaluated_positions.shape == (search.max_evaluations,)
    assert result.proposal_kinds.shape == (search.max_evaluations,)
    assert result.best_objective_history.shape == (search.max_evaluations,)
    assert np.all(np.diff(np.asarray(result.best_objective_history)) <= 0.0)
    np.testing.assert_array_equal(result.evaluated_positions, replay.evaluated_positions)
    assert not np.array_equal(result.evaluated_positions, different.evaluated_positions)


def test_invalid_observations_are_retained_without_inventing_a_mode():
    space = phx.uq.ParameterSpace(
        jnp.asarray(0.0),
        log_prior=lambda value: jnp.asarray(0.0),
    )
    partly_valid = phx.uq.PosteriorProblem(
        space,
        lambda value: jnp.where(jnp.abs(value) < 0.1, jnp.nan, -((value - 0.8) ** 2)),
    )
    valid = phx.uq.search_map(
        partly_valid,
        _search(8),
        key=jr.key(33),
        position_bounds=(jnp.asarray(-2.0), jnp.asarray(2.0)),
    )
    assert valid.valid
    assert valid.invalid_evaluations >= 1
    assert jnp.isfinite(valid.objective)

    invalid = phx.uq.PosteriorProblem(
        space,
        lambda value: jnp.asarray(jnp.nan),
    )
    result = phx.uq.search_map(
        invalid,
        _search(8),
        key=jr.key(34),
        position_bounds=(jnp.asarray(-2.0), jnp.asarray(2.0)),
    )
    assert not result.valid
    assert result.termination_reason == "no_finite_candidates"
    assert result.invalid_evaluations == 8
    assert jnp.isnan(result.objective)
    assert not jnp.any(result.valid_evaluations)


def test_result_archive_preserves_search_units_and_evidence(tmp_path):
    result = phx.uq.search_map(
        _problem(),
        _search(8, noise_scale=0.1),
        key=jr.key(35),
        position_bounds=(jnp.asarray(-3.0), jnp.asarray(3.0)),
    )
    destination = tmp_path / "gp-map.phxuq"
    phx.uq.export_result(result, destination)
    archive = phx.uq.read_result_archive(destination)

    assert archive.kind == "gaussian_process_map_search"
    assert archive.metadata["search"]["noise_scale_units"] == "raw_negative_log_density"
    assert archive.metadata["search"]["jitter_units"] == "standardized_covariance"
    assert "raw_objectives" in archive.fields
    assert "evaluated_positions" in archive.trees
