import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import pytest

import phydrax as phx
from phydrax.domain import BatchEvaluator


class _EndpointSensitiveNormal(phx.uq.AbstractDistribution):
    def sample(self, key, sample_shape=()):
        return jr.normal(key, tuple(sample_shape))

    def icdf(self, value):
        return jsp.special.ndtri(jnp.asarray(value))

    def log_prob(self, value):
        values = jnp.asarray(value)
        return -0.5 * values**2 - 0.5 * jnp.log(2.0 * jnp.pi)

    @property
    def mean(self):
        return jnp.asarray(0.0)

    @property
    def variance(self):
        return jnp.asarray(1.0)

    @property
    def support(self):
        return None

    def contains(self, value):
        return jnp.isfinite(jnp.asarray(value))


class _KeyConsumingBatchIntegrand(BatchEvaluator):
    def __call_batch__(self, batch, /, *, key, **kwargs):
        del kwargs
        reference = batch["z"]
        value = jr.uniform(key)
        return cx.Field(
            jnp.broadcast_to(value, reference.data.shape), dims=reference.dims
        )


class _AlternatingBatchIntegrand(BatchEvaluator):
    def __call_batch__(self, batch, /, *, key, **kwargs):
        del key, kwargs
        reference = batch["x"]
        index = jnp.arange(reference.data.shape[0])
        values = jnp.where(index % 2 == 0, 1.0, -1.0)
        return cx.Field(values, dims=reference.dims)


def _uniform_problem():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    target = phx.integration.over(domain.component())
    return domain, target


def test_iid_monte_carlo_reports_sampling_standard_error():
    domain, target = _uniform_problem()
    function = domain.Function("x")(lambda x: x**2)

    estimate = phx.integration.integrate(
        function,
        target,
        phx.integration.MonteCarloPlan(4096),
        key=jr.key(1),
    )

    assert jnp.allclose(estimate.value.data, 1.0 / 3.0, atol=2e-2)
    assert estimate.error_kind == "iid-standard-error"
    assert estimate.error_estimate > 0.0
    assert estimate.diagnostics.num_independent_replicates == 1


def test_antithetic_error_uses_independent_pairs():
    domain, target = _uniform_problem()
    function = domain.Function("x")(lambda x: x)
    plan = phx.integration.MonteCarloPlan(512, design=phx.integration.AntitheticDesign())

    estimate = phx.integration.integrate(function, target, plan, key=jr.key(2))

    assert jnp.allclose(estimate.value.data, 0.5, atol=1e-14)
    assert estimate.error_kind == "antithetic-pair-standard-error"
    assert estimate.diagnostics.num_pairs == 256
    assert estimate.diagnostics.pair_covariance < 0.0


def test_latin_hypercube_does_not_claim_iid_uncertainty():
    domain, target = _uniform_problem()
    function = domain.Function("x")(lambda x: x**2)
    plan = phx.integration.MonteCarloPlan(
        256, design=phx.integration.LatinHypercubeDesign()
    )

    estimate = phx.integration.integrate(function, target, plan, key=jr.key(3))

    assert jnp.allclose(estimate.value.data, 1.0 / 3.0, atol=1e-3)
    assert estimate.error_estimate is None
    assert estimate.error_kind is None
    assert estimate.diagnostics.standard_error is None


def test_qmc_uncertainty_requires_independent_randomized_replicates():
    domain, target = _uniform_problem()
    function = domain.Function("x")(lambda x: x**2)
    deterministic = phx.integration.integrate(
        function,
        target,
        phx.integration.QuasiMonteCarloPlan(256, scrambled=False, num_replicates=1),
    )
    randomized = phx.integration.integrate(
        function,
        target,
        phx.integration.QuasiMonteCarloPlan(256, num_replicates=4),
        key=jr.key(4),
    )

    assert deterministic.error_estimate is None
    assert deterministic.error_kind is None
    assert deterministic.diagnostics.standard_error is None
    assert jnp.allclose(randomized.value.data, 1.0 / 3.0, atol=2e-4)
    assert randomized.error_kind == "randomized-qmc-replicate-error"
    assert randomized.error_estimate >= 0.0
    assert randomized.diagnostics.replicate_estimates.shape == (4,)


def test_importance_sampling_reports_raw_weight_diagnostics():
    probability = phx.domain.ProbabilityDomain(phx.uq.Normal(0.0, 1.0), label="z")
    function = probability.Function("z")(lambda z: z**2)
    plan = phx.integration.ImportanceSamplingPlan(8192, phx.uq.Normal(1.0, 2.0))

    estimate = phx.integration.integrate(
        function,
        phx.integration.expectation(probability),
        plan,
        key=jr.key(5),
    )

    assert jnp.allclose(estimate.value.data, 1.0, atol=5e-2)
    assert estimate.error_kind == "weighted-iid-standard-error"
    assert estimate.error_estimate > 0.0
    assert estimate.diagnostics.weights.weight_ess > 0.0
    assert jnp.allclose(estimate.diagnostics.normalizer_estimate, 1.0, atol=5e-2)


def test_importance_sampling_reports_proposal_support_failure():
    probability = phx.domain.ProbabilityDomain(phx.uq.Normal(0.0, 1.0), label="z")
    plan = phx.integration.ImportanceSamplingPlan(256, phx.uq.Uniform(-1.0, 1.0))

    estimate = phx.integration.integrate(
        1.0,
        phx.integration.expectation(probability),
        plan,
        key=jr.key(8),
    )

    assert estimate.status == int(
        phx.integration.IntegrationStatus.PROPOSAL_SUPPORT_FAILURE
    )
    assert not estimate.successful


def test_external_weighted_samples_do_not_invent_independence():
    samples = jnp.asarray([1.0, 2.0, 3.0])
    log_weights = jnp.log(jnp.asarray([1.0, 2.0, 1.0]))
    target = phx.integration.weighted(samples, log_weights)

    estimate = phx.integration.integrate(lambda values: values, target)

    assert jnp.allclose(estimate.value.data, 2.0, atol=1e-12)
    assert estimate.error_estimate is None
    assert estimate.error_kind is None
    assert jnp.allclose(estimate.diagnostics.weights.weight_ess, 8.0 / 3.0)


def test_control_variate_coefficients_fit_on_disjoint_iid_pilot():
    domain, target = _uniform_problem()
    function = domain.Function("x")(lambda x: 3.0 * x + 2.0)
    control = domain.Function("x")(lambda x: x)
    estimator = phx.integration.ControlVariateEstimator(
        (control,), (0.5,), pilot_samples=64
    )
    plan = phx.integration.MonteCarloPlan(1024, control_variate=estimator)

    estimate = phx.integration.integrate(function, target, plan, key=jr.key(7))

    assert jnp.allclose(estimate.value.data, 3.5, atol=1e-10)
    assert estimate.error_estimate < 1e-10
    assert estimate.num_evaluations == 960


def test_explicit_stratification_preserves_physical_measure():
    square = phx.domain.GeometryDomain(phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile())
    vertices = jnp.asarray(
        [
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0]],
            [[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        ]
    )
    partition = phx.geometry.GeometryMeasurePartition(
        vertices, jnp.asarray([2.0, 2.0]), kind="triangle"
    )
    plan = phx.integration.StratifiedMonteCarloPlan(
        64, phx.integration.StratifiedDesign(partition)
    )

    estimate = phx.integration.integrate(
        1.0,
        phx.integration.over(square.component()),
        plan,
        key=jr.key(6),
    )

    assert jnp.allclose(estimate.value.data, 4.0, atol=1e-12)
    assert estimate.error_kind == "stratified-standard-error"
    assert jnp.all(estimate.diagnostics.samples_per_stratum > 0)


def test_direct_monte_carlo_excludes_fixed_component_labels():
    space = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    time = phx.domain.TimeInterval(0.0, 2.0)
    domain = phx.domain.ProductDomain(space, time)
    component = domain.component({"t": phx.domain.FixedStart()})

    estimate = phx.integration.integrate(
        1.0,
        phx.integration.over(component),
        phx.integration.MonteCarloPlan(32),
        key=jr.key(9),
    )

    assert estimate.successful
    assert jnp.allclose(estimate.value.data, 1.0, atol=1e-12)


def test_antithetic_zero_density_reports_invalid_normalization_mass():
    domain, base = _uniform_problem()
    target = phx.integration.normalized_density(
        base,
        domain.Function("x")(lambda x: jnp.full_like(x, -jnp.inf)),
    )
    plan = phx.integration.MonteCarloPlan(
        32,
        design=phx.integration.AntitheticDesign(),
    )

    estimate = phx.integration.integrate(1.0, target, plan, key=jr.key(10))

    assert estimate.status == int(
        phx.integration.IntegrationStatus.INVALID_NORMALIZATION_MASS
    )
    assert not estimate.successful


def test_stratified_zero_density_reports_invalid_normalization_mass():
    square = phx.domain.GeometryDomain(phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile())
    vertices = jnp.asarray(
        [
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0]],
            [[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        ]
    )
    partition = phx.geometry.GeometryMeasurePartition(
        vertices,
        jnp.asarray([2.0, 2.0]),
        kind="triangle",
    )
    target = phx.integration.normalized_density(
        phx.integration.over(square.component()),
        square.Function("x")(lambda x: jnp.full(x.shape[:-1], -jnp.inf)),
    )
    plan = phx.integration.StratifiedMonteCarloPlan(
        32,
        phx.integration.StratifiedDesign(partition),
    )

    estimate = phx.integration.integrate(1.0, target, plan, key=jr.key(11))

    assert estimate.status == int(
        phx.integration.IntegrationStatus.INVALID_NORMALIZATION_MASS
    )
    assert not estimate.successful


def test_weighted_samples_reject_degenerate_weights_and_nonfinite_values():
    invalid_weights = phx.integration.weighted(
        jnp.arange(4.0),
        jnp.full((4,), -jnp.inf),
        independent=True,
    )
    invalid_values = phx.integration.weighted(
        jnp.arange(4.0),
        jnp.zeros((4,)),
        independent=True,
    )

    weight_estimate = phx.integration.integrate(lambda values: values, invalid_weights)
    value_estimate = phx.integration.integrate(
        lambda values: jnp.full_like(values, jnp.nan),
        invalid_values,
    )

    assert weight_estimate.status == int(
        phx.integration.IntegrationStatus.INVALID_WEIGHTS
    )
    assert value_estimate.status == int(
        phx.integration.IntegrationStatus.NONFINITE_INTEGRAND
    )
    assert not weight_estimate.successful
    assert not value_estimate.successful


def test_antithetic_errors_require_independent_replicate_pairs():
    domain, target = _uniform_problem()
    function = domain.Function("x")(lambda x: x**2)
    one_pair = phx.integration.integrate(
        function,
        target,
        phx.integration.MonteCarloPlan(
            2,
            design=phx.integration.AntitheticDesign(),
        ),
        key=jr.key(15),
    )
    latin_pairs = phx.integration.integrate(
        function,
        target,
        phx.integration.MonteCarloPlan(
            16,
            design=phx.integration.AntitheticDesign(
                phx.integration.LatinHypercubeDesign()
            ),
        ),
        key=jr.key(16),
    )

    for estimate in (one_pair, latin_pairs):
        assert estimate.successful
        assert estimate.error_estimate is None
        assert estimate.error_kind is None
        assert estimate.diagnostics.standard_error is None


def test_antithetic_sampling_rejects_boundary_component_selectors():
    domain, _ = _uniform_problem()
    target = phx.integration.over(domain.component({"x": phx.domain.Boundary()}))
    plan = phx.integration.MonteCarloPlan(
        16,
        design=phx.integration.AntitheticDesign(),
    )

    with pytest.raises(TypeError, match=r"Interior\(\)"):
        phx.integration.materialize(target, plan, key=jr.key(17))


def test_randomized_probability_sampling_and_evaluation_use_independent_keys():
    distribution = _EndpointSensitiveNormal()
    probability = phx.domain.ProbabilityDomain(distribution, label="z")
    target = phx.integration.expectation(probability)
    plan = phx.integration.MonteCarloPlan(16)
    caller_key = jr.key(18)
    sampling_key, evaluation_key = jr.split(caller_key)

    realization = phx.integration.materialize(target, plan, key=caller_key)
    expected_samples = distribution.sample(sampling_key, sample_shape=(16,))
    function = probability.Function("z")(_KeyConsumingBatchIntegrand())
    estimate = phx.integration.reduce(function, realization)

    assert jnp.array_equal(realization.batch.points["z"].data, expected_samples)
    assert jnp.array_equal(
        jr.key_data(realization.key),
        jr.key_data(evaluation_key),
    )
    assert jnp.allclose(estimate.value.data, jr.uniform(evaluation_key), atol=1e-14)


def test_antithetic_sampling_rejects_partial_coupled_target_axes():
    x = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    y = phx.domain.ScalarInterval(0.0, 1.0, label="y")
    domain = phx.domain.ProductDomain(x, y)
    target = phx.integration.over(domain.component(), axes="x")
    plan = phx.integration.MonteCarloPlan(
        16,
        design=phx.integration.AntitheticDesign(),
    )

    with pytest.raises(ValueError, match="coupled block"):
        phx.integration.materialize(target, plan, key=jr.key(19))


def test_raw_weighted_scalar_integrand_broadcasts_over_samples():
    target = phx.integration.weighted(
        jnp.arange(5.0),
        jnp.log(jnp.asarray([1.0, 2.0, 3.0, 2.0, 1.0])),
    )

    estimate = phx.integration.integrate(7.0, target)

    assert estimate.successful
    assert jnp.allclose(estimate.value.data, 7.0, atol=1e-14)
    assert estimate.num_evaluations == 5


def test_dependent_weighted_diagnostics_hide_both_standard_errors():
    target = phx.integration.weighted(
        jnp.asarray([1.0, 2.0, 4.0]),
        jnp.log(jnp.asarray([1.0, 2.0, 1.0])),
        independent=False,
    )

    estimate = phx.integration.integrate(lambda values: values, target)

    assert estimate.successful
    assert estimate.diagnostics.standard_error is None
    assert estimate.diagnostics.normalizer_standard_error is None


@pytest.mark.parametrize("sequence", ("sobol", "halton"))
def test_deterministic_qmc_uses_open_probability_quantiles(sequence):
    probability = phx.domain.ProbabilityDomain(
        _EndpointSensitiveNormal(),
        label="z",
    )
    target = phx.integration.expectation(probability)
    plan = phx.integration.QuasiMonteCarloPlan(
        16,
        sequence=sequence,
        scrambled=False,
        num_replicates=1,
    )

    realization = phx.integration.materialize(target, plan)
    estimate = phx.integration.reduce(
        probability.Function("z")(lambda z: z**2),
        realization,
    )

    assert realization.key is None
    assert jnp.all(jnp.isfinite(realization.batch.points["z"].data))
    assert estimate.successful
    assert jnp.all(jnp.isfinite(estimate.value.data))


def test_identical_unbounded_importance_proposal_passes_support_probes():
    distribution = _EndpointSensitiveNormal()
    probability = phx.domain.ProbabilityDomain(distribution, label="z")
    target = phx.integration.expectation(probability)
    plan = phx.integration.ImportanceSamplingPlan(128, distribution)

    estimate = phx.integration.integrate(1.0, target, plan, key=jr.key(20))

    assert estimate.successful
    assert estimate.status == int(phx.integration.IntegrationStatus.CONVERGED)


def test_importance_sampling_requires_an_explicit_random_key():
    probability = phx.domain.ProbabilityDomain(phx.uq.Normal(0.0, 1.0), label="z")
    target = phx.integration.expectation(probability)
    plan = phx.integration.ImportanceSamplingPlan(16, phx.uq.Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="requires key"):
        phx.integration.materialize(target, plan)


def test_large_measure_monte_carlo_rejects_overflowed_standard_error():
    upper = float(jnp.finfo(float).max / 4.0)
    domain = phx.domain.ScalarInterval(0.0, upper, label="x")
    target = phx.integration.over(domain.component())
    function = domain.Function("x")(_AlternatingBatchIntegrand())

    estimate = phx.integration.integrate(
        function,
        target,
        phx.integration.MonteCarloPlan(16),
        key=jr.key(21),
    )

    assert jnp.all(jnp.isfinite(estimate.value.data))
    assert jnp.all(jnp.isinf(estimate.diagnostics.standard_error))
    assert estimate.status == int(phx.integration.IntegrationStatus.NONFINITE_INTEGRAND)
    assert not estimate.successful


def test_raw_weighted_reduction_rejects_log_weight_overflow():
    target = phx.integration.weighted(
        jnp.ones((4,)),
        jnp.full((4,), 1000.0),
        normalized=False,
        independent=True,
    )

    estimate = phx.integration.integrate(lambda values: values, target)

    assert jnp.isinf(estimate.value.data)
    assert jnp.isinf(estimate.diagnostics.normalizer_estimate)
    assert estimate.status == int(phx.integration.IntegrationStatus.INVALID_WEIGHTS)
    assert not estimate.successful
