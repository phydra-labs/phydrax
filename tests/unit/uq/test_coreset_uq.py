#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _gaussian_mcmc_result():
    chains = 2
    draws = 48
    parameter_space = phx.uq.ParameterSpace(
        {"location": jnp.asarray([0.0])},
        log_prior=lambda _parameters: 0.0,
    )
    problem = phx.uq.PosteriorProblem(
        parameter_space,
        lambda parameters: -0.5 * jnp.sum(parameters["location"] ** 2),
        predict=lambda parameters, query: cx.Field(
            parameters["location"][0] + query.data,
            dims=query.dims,
        ),
    )
    unconstrained = {
        "location": jr.normal(jr.key(10), (chains, draws, 1)),
    }
    log_density = -0.5 * jnp.squeeze(unconstrained["location"] ** 2, axis=-1)
    diagnostics = phx.uq.MCMCDiagnostics(
        rhat={"location": jnp.ones((1,))},
        bulk_ess={"location": jnp.full((1,), 70.0)},
        tail_ess={"location": jnp.full((1,), 60.0)},
        acceptance_rate=jnp.full((chains,), 0.9),
        divergent=jnp.zeros((chains,), dtype=jnp.int32),
    )
    result = phx.uq.MCMCResult(
        problem=problem,
        samples=unconstrained,
        unconstrained_samples=unconstrained,
        log_density=log_density,
        acceptance_rate=jnp.full((chains, draws), 0.9),
        divergent=jnp.zeros((chains, draws), dtype=bool),
        energy=jnp.zeros((chains, draws)),
        num_integration_steps=jnp.ones((chains, draws), dtype=jnp.int32),
        num_trajectory_expansions=jnp.zeros((chains, draws), dtype=jnp.int32),
        final_states=(),
        warmup=(),
        diagnostics=diagnostics,
        root_key=jr.key(0),
        chain_keys=jr.split(jr.key(1), chains),
        algorithm="nuts",
        duration_seconds=1.0,
        max_num_doublings=10,
        chain_method="vectorized",
        adaptation_duration_seconds=0.4,
        sampling_duration_seconds=0.6,
    )
    return result


def test_pivoted_cholesky_selection_builds_a_sparse_gp_factor():
    coordinates = jnp.linspace(-1.0, 1.0, 64)
    points = jnp.stack((coordinates, coordinates**2), axis=1)
    selection = phx.uq.select_inducing_points(points, 12, key=jr.key(3))
    factor = phx.uq.SparseGaussianProcessFactor(
        points,
        selection.points,
        amplitude=0.2,
        length_scale=0.4,
        noise_scale=0.01,
    )

    assert selection.points.shape == (12, 2)
    assert jnp.unique(selection.indices).shape == selection.indices.shape
    assert selection.diagnostics.residual_trace < selection.diagnostics.initial_trace
    assert jnp.isfinite(factor.log_probability(jnp.sin(points[:, 0])))


def test_inducing_selection_reports_numerical_rank_exhaustion():
    with pytest.raises(ValueError, match="kernel rank"):
        phx.uq.select_inducing_points(jnp.zeros((8, 2)), 2, key=jr.key(4))


def test_stein_thinning_preserves_chains_source_indices_and_diagnostics():
    result = _gaussian_mcmc_result()
    method = phx.uq.SteinThinning(10)

    coreset = phx.uq.thin_posterior(result, method, key=jr.key(5))
    repeated = phx.uq.thin_posterior(result, method, key=jr.key(5))

    assert result.samples["location"].shape == (2, 48, 1)
    assert coreset.samples["location"].shape == (2, 10, 1)
    assert coreset.draw_indices.shape == (2, 10)
    assert jnp.array_equal(coreset.draw_indices, repeated.draw_indices)
    assert jnp.all(jax_unique_counts(coreset.draw_indices) == 10)
    assert jnp.array_equal(
        coreset.samples["location"],
        jnp.take_along_axis(
            result.samples["location"],
            coreset.draw_indices[..., None],
            axis=1,
        ),
    )
    assert coreset.source_diagnostics is result.diagnostics
    assert coreset.source_algorithm == result.algorithm
    assert coreset.source_num_draws == 48
    assert jnp.all(jnp.isfinite(coreset.kernel_stein_discrepancy))
    prediction = coreset.predict(
        cx.Field(jnp.linspace(0.0, 1.0, 3), dims=("x",)),
        batch_size=4,
    )
    assert prediction.samples.dims == ("__phydra_uq_chain", "__phydra_uq_draw", "x")
    assert prediction.samples.shape == (2, 10, 3)


def test_stein_thinning_rejects_more_points_than_each_chain_contains():
    result = _gaussian_mcmc_result()

    with pytest.raises(ValueError, match="more posterior draws"):
        phx.uq.thin_posterior(result, phx.uq.SteinThinning(49))


def jax_unique_counts(indices):
    return jnp.asarray([jnp.unique(row).size for row in indices])
