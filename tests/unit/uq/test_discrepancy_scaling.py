#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_multi_output_gp_uses_declared_cross_output_covariance():
    observation_x = jnp.linspace(0.0, 1.0, 18)
    base = jnp.stack([observation_x, 2.0 * observation_x], axis=1)
    latent = 0.15 * jnp.sin(2.0 * jnp.pi * observation_x)
    discrepancy = jnp.stack([latent, -0.6 * latent], axis=1)
    output_covariance = jnp.array([[1.0, -0.75], [-0.75, 1.0]])
    model = phx.uq.MultiOutputGaussianProcessDiscrepancy(
        observation_x,
        base + discrepancy,
        output_covariance=output_covariance,
        output_names=("velocity", "pressure"),
        kernel="exp_squared",
    )
    query = jnp.linspace(0.0, 1.0, 31)
    query_base = jnp.stack([query, 2.0 * query], axis=1)
    conditioned = model.condition(
        base,
        query,
        amplitude=0.15,
        length_scale=0.25,
        noise_scale=jnp.array([0.01, 0.015]),
        point_dim="x",
        output_dim="field",
    )
    prediction = conditioned.predictive_field(
        query_base,
        jr.key(2),
        num_samples=20,
        observation_variance=jnp.array([0.01**2, 0.015**2]),
    )

    expected = jnp.stack(
        [
            0.15 * jnp.sin(2.0 * jnp.pi * query),
            -0.09 * jnp.sin(2.0 * jnp.pi * query),
        ],
        axis=1,
    )
    posterior_cross_correlation = conditioned.covariance[0, 1] / jnp.sqrt(
        conditioned.covariance[0, 0] * conditioned.covariance[1, 1]
    )
    assert model.num_outputs == 2
    assert conditioned.output_names == ("velocity", "pressure")
    assert conditioned.mean.shape == (31, 2)
    assert conditioned.sample(jr.key(1), num_samples=7).shape == (7, 31, 2)
    assert prediction.samples.dims == (
        "__phydra_uq_discrepancy",
        "x",
        "field",
    )
    assert jnp.sqrt(jnp.mean((conditioned.mean - expected) ** 2)) < 0.01
    assert posterior_cross_correlation < -0.05

    with pytest.raises(ValueError, match="positive definite"):
        phx.uq.MultiOutputGaussianProcessDiscrepancy(
            observation_x,
            base,
            output_covariance=jnp.array([[1.0, 2.0], [2.0, 1.0]]),
        )


def test_sparse_fitc_matches_exact_gp_without_quadratic_training_storage():
    observation_x = jnp.linspace(0.0, 1.0, 120)
    physical_mean = 1.2 * observation_x
    discrepancy = 0.2 * jnp.sin(2.0 * jnp.pi * observation_x)
    observations = physical_mean + discrepancy
    exact = phx.uq.ExactGaussianProcessDiscrepancy(
        observation_x,
        observations,
        kernel="exp_squared",
    )
    sparse = phx.uq.SparseGaussianProcessDiscrepancy.from_evenly_spaced_subset(
        observation_x,
        observations,
        num_inducing=24,
        kernel="exp_squared",
    )
    query = jnp.linspace(0.0, 1.0, 75)
    settings = dict(amplitude=0.25, length_scale=0.2, noise_scale=0.02)
    exact_condition = exact.condition(physical_mean, query, **settings)
    sparse_condition = sparse.condition(physical_mean, query, **settings)
    exact_log_probability = exact.log_marginal_likelihood(physical_mean, **settings)
    sparse_log_probability = sparse.log_marginal_likelihood(physical_mean, **settings)
    gradient = jax.grad(
        lambda length_scale: sparse.log_marginal_likelihood(
            physical_mean,
            amplitude=settings["amplitude"],
            length_scale=length_scale,
            noise_scale=settings["noise_scale"],
        )
    )(settings["length_scale"])

    assert sparse.num_inducing == 24
    assert sparse.factor_storage_elements < observation_x.size**2 / 2
    assert jnp.isfinite(gradient)
    assert jnp.abs(sparse_log_probability - exact_log_probability) < 0.2
    assert jnp.sqrt(jnp.mean((sparse_condition.mean - exact_condition.mean) ** 2)) < 2e-3
    assert (
        jnp.sqrt(jnp.mean((sparse_condition.variance - exact_condition.variance) ** 2))
        < 2e-3
    )


def test_fixed_gp_factors_reuse_likelihood_gradients_and_conditioning_geometry():
    coordinate = jnp.linspace(0.0, 1.0, 32)
    observation_points = jnp.stack([coordinate, coordinate**2], axis=1)
    physical = lambda parameter: parameter * coordinate
    observations = physical(0.8) + 0.2 * jnp.sin(2.0 * jnp.pi * coordinate)
    query_coordinate = jnp.linspace(0.0, 1.0, 21)
    query_points = jnp.stack([query_coordinate, query_coordinate**2], axis=1)
    settings = {
        "amplitude": 0.25,
        "length_scale": jnp.array([0.2, 0.35]),
        "noise_scale": 0.015,
    }
    exact = phx.uq.ExactGaussianProcessDiscrepancy(
        observation_points,
        observations,
        kernel="matern52",
    )
    sparse = phx.uq.SparseGaussianProcessDiscrepancy.from_evenly_spaced_subset(
        observation_points,
        observations,
        num_inducing=12,
        kernel="matern52",
    )
    exact_factor = exact.factor(**settings)
    sparse_factor = sparse.factor(**settings)
    exact_conditioner = exact_factor.conditioner(query_points, output_dim="query")
    sparse_conditioner = sparse_factor.conditioner(query_points, output_dim="query")

    def dynamic_exact(parameter):
        return exact.log_marginal_likelihood(physical(parameter), **settings)

    def factored_exact(parameter):
        return exact_factor.log_probability(exact.residual(physical(parameter)))

    def dynamic_sparse(parameter):
        return sparse.log_marginal_likelihood(physical(parameter), **settings)

    def factored_sparse(parameter):
        return sparse_factor.log_probability(sparse.residual(physical(parameter)))

    def dynamic_exact_length_scale(length_scale):
        return exact.log_marginal_likelihood(
            physical(parameter),
            amplitude=settings["amplitude"],
            length_scale=length_scale,
            noise_scale=settings["noise_scale"],
        )

    def dynamic_sparse_length_scale(length_scale):
        return sparse.log_marginal_likelihood(
            physical(parameter),
            amplitude=settings["amplitude"],
            length_scale=length_scale,
            noise_scale=settings["noise_scale"],
        )

    parameter = jnp.asarray(0.75)
    exact_dynamic_condition = exact.condition(
        physical(parameter),
        query_points,
        output_dim="query",
        **settings,
    )
    sparse_dynamic_condition = sparse.condition(
        physical(parameter),
        query_points,
        output_dim="query",
        **settings,
    )
    exact_factored_condition = exact_conditioner.condition(
        exact.residual(physical(parameter))
    )
    sparse_factored_condition = sparse_conditioner.condition(
        sparse.residual(physical(parameter))
    )

    assert exact_factor.factor_storage_elements == observation_points.shape[0] ** 2
    assert sparse_factor.factor_storage_elements == sparse.factor_storage_elements
    assert jnp.allclose(dynamic_exact(parameter), factored_exact(parameter))
    assert jnp.allclose(dynamic_sparse(parameter), factored_sparse(parameter))
    assert jnp.allclose(
        jax.grad(dynamic_exact)(parameter), jax.grad(factored_exact)(parameter)
    )
    assert jnp.allclose(
        jax.grad(dynamic_sparse)(parameter),
        jax.grad(factored_sparse)(parameter),
    )
    assert jnp.all(
        jnp.isfinite(jax.grad(dynamic_exact_length_scale)(settings["length_scale"]))
    )
    assert jnp.all(
        jnp.isfinite(jax.grad(dynamic_sparse_length_scale)(settings["length_scale"]))
    )
    assert jnp.allclose(
        exact_dynamic_condition.mean,
        exact_factored_condition.mean,
    )
    assert jnp.allclose(
        exact_dynamic_condition.covariance,
        exact_factored_condition.covariance,
    )
    assert jnp.allclose(
        sparse_dynamic_condition.mean,
        sparse_factored_condition.mean,
    )
    assert jnp.allclose(
        sparse_dynamic_condition.covariance,
        sparse_factored_condition.covariance,
    )
    assert exact_factored_condition.output_dims == ("query",)
    assert sparse_factored_condition.output_dims == ("query",)


def test_repeated_identifiability_report_gates_bias_scores_coverage_and_confounding():
    baseline = jnp.array([1.48, 1.52, 1.47, 1.51, 1.50, 1.49])
    fixed = jnp.array([1.19, 1.23, 1.18, 1.22, 1.21, 1.20])
    joint = jnp.array([1.18, 1.24, 1.19, 1.21, 1.22, 1.20])
    common = dict(
        true_parameters=jnp.asarray(1.2),
        baseline_parameter_estimates=baseline,
        fixed_gp_parameter_estimates=fixed,
        joint_gp_parameter_estimates=joint,
        baseline_nll=jnp.full(6, 1.4),
        fixed_gp_nll=jnp.full(6, 0.7),
        baseline_crps=jnp.full(6, 0.3),
        fixed_gp_crps=jnp.full(6, 0.1),
        fixed_gp_coverage=jnp.full(6, 0.9),
    )
    report = phx.uq.discrepancy_identifiability_report(
        **common,
        joint_parameter_gp_correlations=jnp.full((6, 1, 3), 0.4),
    )
    confounded = phx.uq.discrepancy_identifiability_report(
        **common,
        joint_parameter_gp_correlations=jnp.full((6, 1, 3), 0.99),
    )

    assert report.passed
    assert report.num_repeats == 6
    assert report.nll_improvement == pytest.approx(0.7)
    assert report.crps_improvement == pytest.approx(0.2)
    assert not confounded.passed
    assert "parameter/GP correlation" in confounded.failures[-1]
    with pytest.raises(RuntimeError, match="identifiability gates failed"):
        confounded.raise_on_failure()
