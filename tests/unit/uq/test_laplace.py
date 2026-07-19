#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _gaussian_problem():
    likelihood_precision = jnp.asarray(
        [
            [5.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 4.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 3.5, 0.4, 0.0, 0.0],
            [0.0, 0.0, 0.4, 3.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.3, 2.5, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.2, 2.0],
        ]
    )
    center = jnp.zeros((6,))
    space = phx.uq.ParameterSpace(center, priors=phx.uq.Normal(0.0, 1.0))
    problem = phx.uq.PosteriorProblem(
        space,
        lambda value: -0.5 * value @ likelihood_precision @ value,
        predict=lambda value, design: cx.Field(design @ value, dims=("x",)),
    )
    return problem, likelihood_precision


def test_dense_laplace_recovers_correlated_gaussian_and_predicts_named_fields():
    problem, likelihood_precision = _gaussian_problem()
    expected_covariance = jnp.linalg.inv(likelihood_precision + jnp.eye(6))

    result = phx.uq.fit_laplace(problem, jnp.zeros(6))
    samples = result.sample(jr.key(0), num_samples=20_000)
    design = jnp.stack([jnp.ones(6), jnp.arange(6.0)], axis=0)
    prediction = result.predict(
        jr.key(1),
        design,
        num_samples=19,
        batch_size=4,
    )

    assert isinstance(result, phx.uq.LaplaceResult)
    assert result.backend == "dense"
    assert jnp.allclose(result.covariance, expected_covariance, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(
        jnp.cov(samples, rowvar=False),
        expected_covariance,
        rtol=0.08,
        atol=6e-3,
    )
    assert prediction.samples.dims == ("__phydra_uq_draw", "x")
    assert prediction.samples.shape == (19, 2)
    assert jnp.all(prediction.valid.data)


def test_structured_laplax_curvatures_match_their_declared_approximations():
    problem, likelihood_precision = _gaussian_problem()
    probe = jnp.arange(1.0, 7.0)
    exact_covariance = jnp.linalg.inv(likelihood_precision + jnp.eye(6))
    diagonal_covariance = 1.0 / (jnp.diag(likelihood_precision) + 1.0)

    full = phx.uq.fit_laplace(
        problem,
        jnp.zeros(6),
        curvature="full",
        prior_precision=1.0,
    )
    diagonal = phx.uq.fit_laplace(
        problem,
        jnp.zeros(6),
        curvature="diagonal",
        prior_precision=1.0,
    )
    lanczos = phx.uq.fit_laplace(
        problem,
        jnp.zeros(6),
        curvature="lanczos",
        prior_precision=1.0,
        rank=2,
        key=jr.key(2),
    )
    lobpcg = phx.uq.fit_laplace(
        problem,
        jnp.zeros(6),
        curvature="lobpcg",
        prior_precision=1.0,
        rank=1,
        key=jr.key(3),
    )

    assert isinstance(full, phx.uq.StructuredLaplaceResult)
    assert jnp.allclose(full.covariance_vector_product(probe), exact_covariance @ probe)
    assert jnp.allclose(
        diagonal.covariance_vector_product(probe), diagonal_covariance * probe
    )
    for result, expected_rank in ((lanczos, 2), (lobpcg, 1)):
        draw = result.sample(jr.key(4), num_samples=8)
        assert result.rank == expected_rank
        assert draw.shape == (8, 6)
        assert jnp.all(jnp.isfinite(draw))
        assert jnp.all(jnp.isfinite(result.covariance_vector_product(probe)))


def test_laplace_rejects_nonstationary_centers_and_implicit_regularization():
    problem, _ = _gaussian_problem()

    with pytest.raises(phx.uq.LaplaceCurvatureError, match="not stationary"):
        phx.uq.fit_laplace(problem, jnp.ones(6))
    whitened = phx.uq.fit_laplace(problem, curvature="diagonal")
    assert whitened.whitening is not None
    assert float(whitened.prior_precision) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="not dense damping"):
        phx.uq.fit_laplace(
            problem,
            curvature="diagonal",
            prior_precision=1.0,
            damping=1e-3,
        )

    transformed_space = phx.uq.ParameterSpace(
        jnp.zeros(2),
        priors=phx.uq.LogNormal(0.0, 1.0),
        bijectors=phx.uq.ExpBijector(),
    )
    transformed_problem = phx.uq.PosteriorProblem(
        transformed_space,
        lambda value: -0.5 * jnp.sum(value**2),
    )
    with pytest.raises(ValueError, match="identity parameter bijectors"):
        phx.uq.fit_laplace(
            transformed_problem,
            curvature="diagonal",
            prior_precision=1.0,
            stationarity_tolerance=None,
        )
