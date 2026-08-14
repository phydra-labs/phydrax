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


def test_dense_laplace_linearized_prediction_matches_covariance_and_draws():
    problem, likelihood_precision = _gaussian_problem()
    result = phx.uq.fit_laplace(problem, jnp.zeros(6))
    design = jnp.asarray(
        [[1.0, -0.5, 0.0, 0.25, 0.0, 0.1], [0.0, 0.2, 1.0, 0.0, -0.4, 0.3]]
    )
    expected = design @ result.covariance @ design.T

    linearized = result.linearized_predict(design)
    sampled = result.predict(
        jr.key(17),
        design,
        num_samples=40_000,
        batch_size=2_003,
    )

    assert linearized.mean.dims == ("x",)
    assert jnp.allclose(linearized.materialize_covariance().matrix, expected)
    assert jnp.allclose(linearized.exact_variance().data, jnp.diag(expected))
    assert jnp.allclose(
        jnp.cov(sampled.samples.data, rowvar=False),
        expected,
        rtol=0.03,
        atol=3e-3,
    )


def test_dense_laplace_transports_covariance_through_parameter_bijectors():
    center = jnp.log(jnp.asarray(2.0))
    precision = 999.0
    space = phx.uq.ParameterSpace(
        center,
        priors=phx.uq.LogNormal(center, 1.0),
        bijectors=phx.uq.ExpBijector(),
    )
    problem = phx.uq.PosteriorProblem(
        space,
        lambda physical: -0.5 * precision * (jnp.log(physical) - center) ** 2,
        predict=lambda physical: cx.Field(
            jnp.atleast_1d(physical**2),
            dims=("x",),
        ),
    )
    result = phx.uq.fit_laplace(problem, center)
    prediction = result.linearized_predict()

    assert jnp.allclose(result.covariance, 1.0 / (precision + 1.0))
    assert jnp.allclose(result.physical_covariance(), 4.0 / (precision + 1.0))
    assert jnp.allclose(prediction.mean.data, jnp.asarray([4.0]))
    assert jnp.allclose(prediction.exact_variance().data, 64.0 / (precision + 1.0))


def test_structured_laplace_linearized_prediction_stays_matrix_free():
    problem, likelihood_precision = _gaussian_problem()
    result = phx.uq.fit_laplace(
        problem,
        jnp.zeros(6),
        curvature="full",
        prior_precision=1.0,
    )
    design = jnp.asarray(
        [[1.0, 0.0, -0.3, 0.0, 0.2, 0.0], [0.0, 0.5, 0.0, 1.0, 0.0, -0.1]]
    )
    expected_parameter_covariance = jnp.linalg.inv(likelihood_precision + jnp.eye(6))
    expected = design @ expected_parameter_covariance @ design.T
    linearized = result.linearized_predict(design)

    assert linearized.input_covariance_representation == "operator"
    assert jnp.allclose(linearized.materialize_covariance().matrix, expected)
    assert jnp.allclose(
        result.physical_covariance_vector_product(jnp.arange(1.0, 7.0)),
        expected_parameter_covariance @ jnp.arange(1.0, 7.0),
    )
    with pytest.raises(ValueError, match="estimate_variance"):
        linearized.exact_variance()
    estimate = linearized.estimate_variance(
        jr.key(18),
        num_probes=8_192,
        batch_size=511,
    )
    assert jnp.allclose(estimate.variance.data, jnp.diag(expected), atol=0.02)
