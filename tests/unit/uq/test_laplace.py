#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
import phydrax.axes as cx
from phydrax.uq._structured_laplace import StructuredCurvature


def _gaussian_problem() -> Any:
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
        predict=lambda value, design: cx.AxisArray(design @ value, dims=("x",)),
    )
    return problem, likelihood_precision


def test_laplace_scenario_1() -> None:
    problem, likelihood_precision = _gaussian_problem()
    expected_covariance = jnp.linalg.inv(likelihood_precision + jnp.eye(6))

    result = phx.uq.fit_laplace(problem, jnp.zeros(6))
    assert isinstance(result, phx.uq.LaplaceResult)
    samples = result.sample(jr.key(0), num_samples=20_000)
    design = jnp.stack([jnp.ones(6), jnp.arange(6.0)], axis=0)
    prediction = result.predict(
        jr.key(1),
        design,
        num_samples=19,
        batch_size=4,
    )
    assert isinstance(prediction, phx.uq.PredictiveField)
    assert prediction.valid is not None

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
    assert jnp.all(jnp.asarray(prediction.valid.data))
    problem, likelihood_precision = _gaussian_problem()
    result = phx.uq.fit_laplace(problem, jnp.zeros(6))
    assert isinstance(result, phx.uq.LaplaceResult)
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
    assert isinstance(sampled, phx.uq.PredictiveField)

    assert linearized.mean.dims == ("x",)
    assert jnp.allclose(linearized.materialize_covariance().matrix, expected)
    assert jnp.allclose(jnp.asarray(linearized.exact_variance().data), jnp.diag(expected))
    assert jnp.allclose(
        jnp.cov(jnp.asarray(sampled.samples.data), rowvar=False),
        expected,
        rtol=0.03,
        atol=3e-3,
    )
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
        predict=lambda physical: cx.AxisArray(
            jnp.atleast_1d(physical**2),
            dims=("x",),
        ),
    )
    result = phx.uq.fit_laplace(problem, center)
    assert isinstance(result, phx.uq.LaplaceResult)
    prediction = result.linearized_predict()

    assert jnp.allclose(result.covariance, 1.0 / (precision + 1.0))
    assert jnp.allclose(result.physical_covariance(), 4.0 / (precision + 1.0))
    assert jnp.allclose(jnp.asarray(prediction.mean.data), jnp.asarray([4.0]))
    assert jnp.allclose(
        jnp.asarray(prediction.exact_variance().data), 64.0 / (precision + 1.0)
    )
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
    assert isinstance(diagonal, phx.uq.StructuredLaplaceResult)
    assert isinstance(lanczos, phx.uq.StructuredLaplaceResult)
    assert isinstance(lobpcg, phx.uq.StructuredLaplaceResult)

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
    problem, likelihood_precision = _gaussian_problem()
    result = phx.uq.fit_laplace(
        problem,
        jnp.zeros(6),
        curvature="full",
        prior_precision=1.0,
    )
    assert isinstance(result, phx.uq.StructuredLaplaceResult)
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
    assert jnp.allclose(
        jnp.asarray(estimate.variance.data), jnp.diag(expected), atol=0.02
    )
    problem, _ = _gaussian_problem()

    with pytest.raises(phx.uq.LaplaceCurvatureError, match="not stationary"):
        phx.uq.fit_laplace(problem, jnp.ones(6))
    for tolerance in (jnp.nan, jnp.inf):
        with pytest.raises(ValueError, match="finite"):
            phx.uq.fit_laplace(
                problem,
                jnp.zeros(6),
                stationarity_tolerance=tolerance,
            )
    whitened = phx.uq.fit_laplace(problem, curvature="diagonal")
    assert isinstance(whitened, phx.uq.StructuredLaplaceResult)
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


@pytest.mark.parametrize("curvature", ["full", "diagonal", "lanczos", "lobpcg"])
@pytest.mark.parametrize("whitened", [False, True])
def test_structured_actions_dense_reference(
    curvature: StructuredCurvature, whitened: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import equinox as eqx
    import numpy as np

    from phydrax.linalg import eigen as eigen_api

    original = eigen_api.eigensolve

    def complex_phase_spectrum(*args: Any, **kwargs: Any) -> Any:
        spectrum = original(*args, **kwargs)
        return eqx.tree_at(
            lambda value: value.eigenvectors,
            spectrum,
            spectrum.eigenvectors * jnp.asarray(1j, dtype=jnp.complex128),
        )

    monkeypatch.setattr(eigen_api, "eigensolve", complex_phase_spectrum)

    scales = np.asarray([0.7, 1.1, 1.4, 0.9, 1.8, 1.2], dtype=np.float64)
    hessian = np.diag(np.asarray([7, 5, 3, 2, 1, 0.5], dtype=np.float64))
    hessian[0, 1] = hessian[1, 0] = 0.4
    space = phx.uq.ParameterSpace(
        jnp.zeros(6, dtype=jnp.float64),
        priors=phx.uq.Normal(0.0, jnp.asarray(scales)),
    )
    problem = phx.uq.PosteriorProblem(
        space, lambda value: -0.5 * value @ jnp.asarray(hessian) @ value
    )
    result = phx.uq.fit_laplace(
        problem,
        curvature=curvature,
        prior_precision=None if whitened else 1.3,
        rank=2,
        key=jr.key(42),
        tolerance=1e-5,
    )
    assert isinstance(result, phx.uq.StructuredLaplaceResult)
    transform = np.diag(scales) if whitened else np.eye(6)
    working_hessian = transform @ hessian @ transform
    if curvature == "diagonal":
        approximation = np.diag(np.diag(working_hessian))
    elif curvature == "full":
        approximation = working_hessian
    else:
        values, vectors = np.linalg.eigh(working_hessian)
        selected = np.argsort(np.abs(values))[-2:]
        approximation = (vectors[:, selected] * values[selected]) @ vectors[:, selected].T
    precision = approximation + (1.0 if whitened else 1.3) * np.eye(6)
    values, vectors = np.linalg.eigh(precision)
    covariance = transform @ np.linalg.solve(precision, transform.T)
    scale = transform @ (vectors * values**-0.5) @ vectors.T
    probe = np.asarray([1, -2, 3, -4, 5, -6], dtype=np.float64)
    np.testing.assert_allclose(
        result.covariance_vector_product(jnp.asarray(probe)),
        covariance @ probe,
        rtol=2e-4,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        result.scale_mv(jnp.asarray(probe)),
        scale @ probe,
        rtol=2e-4,
        atol=2e-5,
    )
    key = jr.key(9)
    noise = jr.normal(jr.split(key, 1)[0], (5, 6), dtype=jnp.float64)
    np.testing.assert_allclose(
        result.sample_unconstrained(key, num_samples=5),
        np.asarray(noise) @ scale.T,
        rtol=2e-4,
        atol=2e-5,
    )
    expected_bytes = sum(
        leaf.nbytes
        for leaf in jax.tree.leaves((result.curvature_estimate, result.posterior_state))
    )
    assert result.approximate_memory_bytes == expected_bytes


@pytest.mark.parametrize("curvature", ["full", "diagonal", "lanczos", "lobpcg"])
def test_structured_indefinite_precision_refuses(curvature: StructuredCurvature) -> None:
    space = phx.uq.ParameterSpace(
        jnp.zeros(6, dtype=jnp.float64), priors=phx.uq.Normal(0.0, 1.0)
    )
    weights = jnp.asarray([-10.0, -9.0, 2.0, 1.0, 0.5, 0.1])
    problem = phx.uq.PosteriorProblem(
        space, lambda value: -0.5 * jnp.sum(weights * value**2)
    )
    with pytest.raises(Exception, match="not positive definite"):
        phx.uq.fit_laplace(
            problem,
            curvature=curvature,
            prior_precision=1.0,
            rank=2,
            key=jr.key(42),
        )
