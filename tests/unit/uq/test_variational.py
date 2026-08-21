#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _gaussian_problem():
    prior = phx.uq.Normal(0.0, 1.0)
    likelihood = phx.uq.Normal(1.5, 0.5)
    return phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(jnp.asarray(0.0), priors=prior),
        lambda value: likelihood.log_prob(value),
    )


def test_mean_field_samples_and_normalized_log_density_are_consistent():
    family = phx.uq.MeanFieldGaussianFamily.from_position(
        {"a": jnp.asarray([0.2, -0.1]), "b": jnp.asarray(0.3)},
        initial_scale=0.4,
    )
    samples, sampled_log_prob = family.sample_and_log_prob(
        jax.random.key(1),
        sample_shape=(16,),
    )

    assert samples["a"].shape == (16, 2)
    assert samples["b"].shape == (16,)
    assert sampled_log_prob.shape == (16,)
    assert jnp.array_equal(sampled_log_prob, family.log_prob(samples))
    assert all(
        bool(jnp.all(leaf))
        for leaf in jax.tree.leaves(jax.tree.map(jnp.isfinite, family.scale))
    )


def test_mean_field_vi_recovers_analytic_gaussian_posterior():
    problem = _gaussian_problem()
    result = phx.uq.fit_variational(
        problem,
        key=jax.random.key(2),
        config=phx.uq.VariationalConfig(
            num_steps=300,
            samples_per_step=32,
            learning_rate=0.03,
            record_every=20,
        ),
        num_samples=1000,
    )
    expected_mean = 1.2
    expected_scale = jnp.sqrt(0.2)

    assert abs(float(result.family.location) - expected_mean) < 0.08
    assert abs(float(result.family.scale) - float(expected_scale)) < 0.08
    assert abs(float(jnp.mean(result.samples)) - expected_mean) < 0.1
    assert abs(float(jnp.std(result.samples)) - float(expected_scale)) < 0.1
    assert jnp.all(result.diagnostics.finite)
    assert result.num_draws == 1000
    assert result.approximation_id == "reverse-kl/mean-field-gaussian"


def test_variational_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    problem = _gaussian_problem()
    root_key = jax.random.key(3)
    common = dict(
        samples_per_step=16,
        learning_rate=0.02,
        record_every=5,
    )
    checkpoint = tmp_path / "variational.npz"
    phx.uq.fit_variational(
        problem,
        key=root_key,
        config=phx.uq.VariationalConfig(num_steps=20, **common),
        num_samples=32,
        checkpoint_path=checkpoint,
        checkpoint_every=5,
        checkpoint_id="gaussian-vi",
    )
    resumed = phx.uq.fit_variational(
        problem,
        key=root_key,
        config=phx.uq.VariationalConfig(num_steps=40, **common),
        num_samples=32,
        resume_from=checkpoint,
        checkpoint_every=5,
        checkpoint_id="gaussian-vi",
    )
    uninterrupted = phx.uq.fit_variational(
        problem,
        key=root_key,
        config=phx.uq.VariationalConfig(num_steps=40, **common),
        num_samples=32,
    )

    assert jax.tree.all(
        jax.tree.map(
            jnp.array_equal,
            resumed.unconstrained_samples,
            uninterrupted.unconstrained_samples,
        )
    )
    assert jax.tree.all(
        jax.tree.map(jnp.array_equal, resumed.family, uninterrupted.family)
    )
    assert jnp.array_equal(resumed.diagnostics.elbo, uninterrupted.diagnostics.elbo)


def test_variational_family_preserves_constrained_parameter_coordinates():
    likelihood = phx.uq.Normal(2.0, 0.3)
    space = phx.uq.ParameterSpace(
        jnp.asarray(0.0),
        priors=phx.uq.LogNormal(0.0, 0.5),
        bijectors=phx.uq.ExpBijector(),
    )
    problem = phx.uq.PosteriorProblem(
        space,
        lambda value: likelihood.log_prob(value),
    )
    result = phx.uq.fit_variational(
        problem,
        key=jax.random.key(4),
        config=phx.uq.VariationalConfig(
            num_steps=150,
            samples_per_step=16,
            learning_rate=0.02,
            record_every=10,
        ),
        num_samples=128,
    )

    assert jnp.all(result.samples > 0.0)
    assert jnp.all(jnp.isfinite(result.log_target))
    assert jnp.all(jnp.isfinite(result.log_variational))
