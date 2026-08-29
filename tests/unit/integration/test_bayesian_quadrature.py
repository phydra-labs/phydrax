import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _problem(
    *,
    dtype=jnp.float64,
    count=12,
    length_scale=0.8,
    kernel_scale=None,
    observation_noise=0.0,
    solve_regularization=1.0e-10,
    target_id="normal-expectation",
):
    location = jnp.asarray(0.3, dtype=dtype)
    scale = jnp.asarray(1.1, dtype=dtype)
    probability = phx.domain.ProbabilityDomain(
        phx.uq.Normal(location, scale), label="z"
    )
    target = phx.integration.expectation(probability, target_id=target_id)
    kernel = phx.kernels.SquaredExponentialKernel(
        length_scale=jnp.asarray(length_scale, dtype=dtype)
    )
    if kernel_scale is not None:
        kernel = phx.kernels.ScaleKernel(
            kernel, jnp.asarray(kernel_scale, dtype=dtype)
        )
    kernel_mean = phx.integration.GaussianKernelMean(target, kernel)
    plan = phx.integration.BayesianQuadraturePlan(
        kernel_mean,
        phx.domain.PointSampling(count, design="hammersley"),
        observation_noise=jnp.asarray(observation_noise, dtype=dtype),
        solve_regularization=jnp.asarray(solve_regularization, dtype=dtype),
    )
    return probability, target, kernel_mean, plan




def test_gaussian_kernel_mean_matches_analytic_scalar_formulas():
    _, _, kernel_mean, _ = _problem(length_scale=0.7, solve_regularization=0.0)
    points = jnp.asarray([-1.0, 0.25, 1.5], dtype=jnp.float64)
    variance = jnp.asarray(1.1**2, dtype=points.dtype)
    denominator = 0.7**2 + variance
    expected = 0.7 / jnp.sqrt(denominator) * jnp.exp(
        -0.5 * (points - 0.3) ** 2 / denominator
    )
    expected_double = 0.7 / jnp.sqrt(0.7**2 + 2.0 * variance)

    assert jnp.allclose(kernel_mean.mean(points), expected, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(kernel_mean(points), expected, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(
        kernel_mean.double_mean(), expected_double, rtol=1e-12, atol=1e-12
    )


def test_scaled_kernel_mean_scales_single_and_double_integrals():
    _, _, unscaled, _ = _problem(kernel_scale=None)
    _, _, scaled, _ = _problem(kernel_scale=3.25)
    points = jnp.asarray([-0.4, 0.7])

    assert jnp.allclose(scaled.mean(points), 3.25 * unscaled.mean(points))
    assert jnp.allclose(scaled.double_mean(), 3.25 * unscaled.double_mean())


def test_fixed_bq_matches_independent_dense_oracle_and_retains_solve_evidence():
    probability, target, kernel_mean, plan = _problem(
        count=9, observation_noise=2.0e-4, solve_regularization=3.0e-6
    )
    realization = phx.integration.materialize(target, plan)
    function = probability.Function("z")(lambda z: jnp.exp(0.2 * z) + z**2)
    estimate = phx.integration.reduce(function, realization)

    points = np.asarray(realization.batch.points.points["z"].data)
    values = np.exp(0.2 * points) + points**2
    matrix = np.asarray(kernel_mean.kernel.matrix(points, points))
    matrix = matrix + (2.0e-4 + 3.0e-6) * np.eye(points.size)
    mean = np.asarray(kernel_mean.mean(points))
    weights = np.linalg.solve(matrix, mean)
    expected_value = weights @ values
    expected_variance = np.asarray(kernel_mean.double_mean()) - mean @ weights

    assert estimate.successful
    assert np.allclose(np.asarray(estimate.value.data), expected_value)
    assert np.allclose(np.asarray(estimate.error_estimate), np.sqrt(expected_variance))
    assert estimate.error_kind == "bayesian-posterior-standard-deviation"
    assert estimate.diagnostics.solve.successful
    assert estimate.diagnostics.observation_noise == pytest.approx(2.0e-4)
    assert estimate.diagnostics.solve_regularization == pytest.approx(3.0e-6)


def test_constants_kernel_sections_arrays_fields_and_pytrees_use_same_weights():
    probability, target, kernel_mean, plan = _problem(count=8)
    realization = phx.integration.materialize(target, plan)
    points = realization.batch.points.points["z"]
    weights = realization.batch.weights
    section_point = jnp.asarray(0.2)

    constant = phx.integration.reduce(jnp.asarray(2.0), realization)
    section = phx.integration.reduce(
        probability.Function("z")(
            lambda z: kernel_mean.kernel.pairwise(z, section_point)
        ),
        realization,
    )
    vector = phx.integration.reduce(
        probability.Function("z")(lambda z: jnp.stack((z, z**2), axis=-1)),
        realization,
    )
    field = phx.integration.reduce(
        cx.Field(points.data**3, dims=points.dims), realization
    )
    tree = phx.integration.reduce(
        {
            "linear": probability.Function("z")(lambda z: z),
            "quadratic": probability.Function("z")(lambda z: z**2),
        },
        realization,
    )

    assert jnp.allclose(constant.value.data, 2.0 * jnp.sum(weights))
    assert jnp.allclose(section.value.data, kernel_mean.mean(section_point[None])[0])
    assert vector.value.shape == (2,)
    assert vector.value.dims == (None,)
    assert jnp.allclose(field.value.data, jnp.sum(weights * points.data**3))
    assert set(tree.value) == {"linear", "quadratic"}
    assert tree.error_kind == "bayesian-posterior-standard-deviation"


def test_materialized_design_replays_deterministically():
    _, target, _, plan = _problem(count=16)
    first = phx.integration.materialize(target, plan)
    second = phx.integration.materialize(target, plan)

    assert jnp.array_equal(
        first.batch.points.points["z"].data, second.batch.points.points["z"].data
    )
    assert jnp.array_equal(first.batch.weights, second.batch.weights)
    assert first.batch.points.provenance == second.batch.points.provenance


def test_bq_reduce_is_jittable_and_supports_jvp_and_vjp():
    probability, target, _, plan = _problem(count=10)
    realization = phx.integration.materialize(target, plan)

    def objective(coefficient):
        function = probability.Function("z")(lambda z: coefficient * z**2 + z)
        return phx.integration.reduce(function, realization).value.data

    compiled = jax.jit(objective)
    primal, tangent = jax.jvp(compiled, (jnp.asarray(2.0),), (jnp.asarray(1.0),))
    value, pullback = jax.vjp(compiled, jnp.asarray(2.0))
    cotangent = pullback(jnp.asarray(1.0))[0]
    expected_tangent = phx.integration.reduce(
        probability.Function("z")(lambda z: z**2), realization
    ).value.data

    assert jnp.isfinite(primal)
    assert jnp.allclose(value, primal)
    assert jnp.allclose(tangent, expected_tangent)
    assert jnp.allclose(cotangent, expected_tangent)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_bq_preserves_integration_precision_stages(dtype):
    probability, target, _, plan = _problem(dtype=dtype, count=7)
    precision = phx.integration.IntegrationPrecisionPolicy(
        evaluation_dtype=dtype,
        accumulation_dtype=dtype,
        decision_dtype=dtype,
        output_dtype=dtype,
    )
    estimate = phx.integration.integrate(
        probability.Function("z")(lambda z: z**2),
        target,
        plan,
        precision=precision,
    )

    assert estimate.value.data.dtype == dtype
    assert estimate.error_estimate.dtype == dtype
    assert estimate.diagnostics.posterior_variance.dtype == dtype
    assert (
        estimate.diagnostics.solve.provenance.effective_precision.operator_dtype
        == np.dtype(dtype).name
    )


def test_zero_noise_is_supported_for_a_nonsingular_design():
    probability, target, _, plan = _problem(
        count=6, observation_noise=0.0, solve_regularization=0.0
    )
    estimate = phx.integration.integrate(
        probability.Function("z")(lambda z: z), target, plan
    )

    assert estimate.successful
    assert estimate.diagnostics.observation_noise == 0.0
    assert estimate.diagnostics.solve_regularization == 0.0


def test_singular_design_fails_closed_with_child_solve_status():
    probability = phx.domain.ProbabilityDomain(
        phx.uq.Normal(jnp.float32(0.0), jnp.float32(1.0e-30)), label="z"
    )
    target = phx.integration.expectation(probability, target_id="singular")
    mean = phx.integration.GaussianKernelMean(
        target, phx.kernels.SquaredExponentialKernel(length_scale=jnp.float32(1.0))
    )
    plan = phx.integration.BayesianQuadraturePlan(
        mean,
        phx.domain.PointSampling(4, design="hammersley"),
        observation_noise=0.0,
        solve_regularization=0.0,
    )
    estimate = phx.integration.integrate(
        probability.Function("z")(lambda z: z), target, plan
    )

    assert estimate.status == int(phx.integration.IntegrationStatus.LINEAR_SOLVE_FAILED)
    assert not estimate.diagnostics.solve.successful
    assert jnp.isnan(estimate.value.data)
    assert jnp.isnan(estimate.error_estimate)


def test_target_identity_mismatch_is_rejected_before_integrand_evaluation():
    _, _, kernel_mean, _ = _problem(target_id="first")
    other_probability = phx.domain.ProbabilityDomain(
        phx.uq.Normal(0.3, 1.1), label="z"
    )
    other = phx.integration.expectation(other_probability, target_id="second")
    mismatched = phx.integration.BayesianQuadraturePlan(
        kernel_mean, phx.domain.PointSampling(4, design="hammersley")
    )

    with pytest.raises(ValueError, match="target identity"):
        phx.integration.materialize(other, mismatched)


def test_unsupported_target_kernel_and_dimension_fail_closed():
    uniform = phx.domain.ProbabilityDomain(phx.uq.Uniform(-1.0, 1.0), label="u")
    uniform_target = phx.integration.expectation(uniform)
    squared_exponential = phx.kernels.SquaredExponentialKernel()
    with pytest.raises(TypeError, match="Normal"):
        phx.integration.GaussianKernelMean(uniform_target, squared_exponential)

    _, target, _, _ = _problem()
    with pytest.raises(TypeError, match="SquaredExponentialKernel"):
        phx.integration.GaussianKernelMean(
            target, phx.kernels.Matern32Kernel(length_scale=1.0)
        )
    with pytest.raises(TypeError, match="wrapped once"):
        phx.integration.GaussianKernelMean(
            target,
            phx.kernels.ScaleKernel(
                phx.kernels.ScaleKernel(squared_exponential, 2.0), 3.0
            ),
        )
    with pytest.raises(ValueError, match="dimension does not match"):
        phx.integration.GaussianKernelMean(
            target,
            phx.kernels.SquaredExponentialKernel(
                length_scale=jnp.asarray([1.0, 2.0])
            ),
        )
    kernel_mean = phx.integration.GaussianKernelMean(target, squared_exponential)
    with pytest.raises(ValueError, match="expected dimension 1"):
        kernel_mean.mean(jnp.ones((3, 2)))


def test_nonfinite_integrand_and_invalid_posterior_variance_fail_closed():
    probability, target, _, plan = _problem(count=6)
    realization = phx.integration.materialize(target, plan)
    nonfinite = phx.integration.reduce(
        probability.Function("z")(lambda z: jnp.where(z > 0.0, jnp.nan, z)),
        realization,
    )
    invalid_realization = eqx.tree_at(
        lambda value: value.batch.posterior_variance,
        realization,
        jnp.asarray(-1.0, dtype=realization.batch.posterior_variance.dtype),
    )
    invalid = phx.integration.reduce(
        probability.Function("z")(lambda z: z**2), invalid_realization
    )

    assert nonfinite.status == int(phx.integration.IntegrationStatus.NONFINITE_INTEGRAND)
    assert jnp.isnan(nonfinite.value.data)
    assert invalid.status == int(
        phx.integration.IntegrationStatus.INVALID_POSTERIOR_VARIANCE
    )
    assert jnp.isnan(invalid.value.data)
    assert jnp.isnan(invalid.error_estimate)


def test_point_and_linear_resource_guards_fire_before_execution():
    _, target, kernel_mean, _ = _problem(count=4)
    with pytest.raises(ValueError, match="no kernel matrix was allocated"):
        phx.integration.BayesianQuadraturePlan(
            kernel_mean,
            phx.domain.PointSampling(5, design="hammersley"),
            max_points=4,
        )

    constrained = phx.integration.BayesianQuadraturePlan(
        kernel_mean,
        phx.domain.PointSampling(4, design="hammersley"),
        solve_policy=phx.linalg.LinearSolvePolicy(
            phx.linalg.DenseLU(),
            failure=phx.linalg.FailurePolicy("status"),
            resources=phx.linalg.SolveResourcePolicy(
                factorization_bytes=1,
                workspace_bytes=1,
            ),
        ),
    )
    with pytest.raises(ValueError, match="budget"):
        phx.integration.materialize(target, constrained)


def test_randomized_point_design_requires_key_and_replays_with_same_key():
    import jax.random as jr

    _, target, kernel_mean, _ = _problem(count=5)
    plan = phx.integration.BayesianQuadraturePlan(
        kernel_mean, phx.domain.PointSampling(5, design="latin_hypercube")
    )
    with pytest.raises(ValueError, match="requires key"):
        phx.integration.materialize(target, plan)

    first = phx.integration.materialize(target, plan, key=jr.key(7))
    second = phx.integration.materialize(target, plan, key=jr.key(7))
    assert jnp.array_equal(
        first.batch.points.points["z"].data, second.batch.points.points["z"].data
    )
