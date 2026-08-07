#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_posterior_diagnostics_cover_compilation_vectorization_and_capabilities():
    problem = phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(
            {
                "positive": jnp.asarray(0.2),
                "bounded": jnp.asarray(-0.4),
            },
            priors={
                "positive": phx.uq.LogNormal(0.0, 1.0),
                "bounded": phx.uq.Uniform(-2.0, 3.0),
            },
            bijectors={
                "positive": phx.uq.ExpBijector(),
                "bounded": phx.uq.SigmoidIntervalBijector(-2.0, 3.0),
            },
        ),
        lambda values: -0.5 * ((values["positive"] - 1.0) ** 2 + values["bounded"] ** 2),
        predict=lambda values, query: values["positive"] * query,
        observation_variance=lambda values, query: jnp.ones_like(query),
        sample_observation=lambda key, values, query: values["positive"] * query,
        gauss_newton_residual=lambda values: jnp.asarray(
            [values["positive"] - 1.0, values["bounded"]]
        ),
    )

    diagnostics = phx.uq.diagnose_posterior(
        problem,
        key=jr.key(1),
        num_prior_samples=16,
    )

    assert diagnostics.passed
    assert diagnostics.max_roundtrip_error < 1e-6
    assert diagnostics.prior_sample_finite_fraction == 1.0
    assert diagnostics.capabilities.factorized_prior
    assert diagnostics.capabilities.prediction
    assert diagnostics.capabilities.observation_variance
    assert diagnostics.capabilities.observation_sampling
    assert diagnostics.capabilities.gauss_newton_residual
    assert diagnostics.capabilities.automatic_flow_nuts_initialization
    assert diagnostics.as_dict()["failures"] == ()


def test_posterior_diagnostics_report_nonfinite_density_and_gradient_locations():
    problem = phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(
            {"x": jnp.asarray(-1.0)},
            priors={"x": phx.uq.Normal(0.0, 1.0)},
        ),
        lambda values: jnp.sqrt(values["x"]),
    )

    diagnostics = phx.uq.diagnose_posterior(problem)

    assert not diagnostics.passed
    assert "initial_log_density_nonfinite" in diagnostics.failures
    assert "initial_gradient_nonfinite" in diagnostics.failures
    assert diagnostics.nonfinite_gradient_locations == ("['x']",)


def test_custom_prior_capabilities_do_not_claim_automatic_initialization():
    problem = phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(
            jnp.asarray(0.0),
            log_prior=lambda value: -0.5 * value**2,
        ),
        lambda value: -0.5 * value**2,
    )

    diagnostics = phx.uq.diagnose_posterior(problem, key=jr.key(2))

    assert diagnostics.passed
    assert not diagnostics.capabilities.factorized_prior
    assert not diagnostics.capabilities.automatic_prior_sampling
    assert not diagnostics.capabilities.automatic_flow_nuts_initialization
    assert diagnostics.prior_sample_finite_fraction is None
