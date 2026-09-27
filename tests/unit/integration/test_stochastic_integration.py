from typing import Any

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import pytest

import phydrax as phx
import phydrax.axes as cx
from phydrax.domain import BatchEvaluator


class _EndpointSensitiveNormal(phx.uq.AbstractDistribution):
    def sample(self, key: Any, sample_shape: Any = ()) -> Any:
        return jr.normal(key, tuple(sample_shape))

    def icdf(self, value: Any) -> Any:
        return jsp.special.ndtri(jnp.asarray(value))

    def log_prob(self, value: Any) -> Any:
        values = jnp.asarray(value)
        return -0.5 * values**2 - 0.5 * jnp.log(2.0 * jnp.pi)

    @property
    def mean(self) -> Any:
        return jnp.asarray(0.0)

    @property
    def variance(self) -> Any:
        return jnp.asarray(1.0)

    @property
    def support(self) -> Any:
        return None

    def contains(self, value: Any) -> Any:
        return jnp.isfinite(jnp.asarray(value))


class _KeyConsumingBatchIntegrand(BatchEvaluator):
    def __call_batch__(
        self, batch: Any, /, *, key: Any = jr.key(0), **kwargs: Any
    ) -> Any:
        del kwargs
        reference = batch["z"]
        value = jr.uniform(key)
        return cx.AxisArray(
            jnp.broadcast_to(value, reference.data.shape), dims=reference.dims
        )


class _AlternatingBatchIntegrand(BatchEvaluator):
    def __call_batch__(
        self, batch: Any, /, *, key: Any = jr.key(0), **kwargs: Any
    ) -> Any:
        del key, kwargs
        reference = batch["x"]
        index = jnp.arange(reference.data.shape[0])
        values = jnp.where(index % 2 == 0, 1.0, -1.0)
        return cx.AxisArray(values, dims=reference.dims)


def _uniform_problem() -> Any:
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    target = phx.integration.over(domain.component())
    return domain, target


def test_stochastic_integration_scenario_1() -> None:
    domain, target = _uniform_problem()
    function = domain.Function("x")(lambda x: x**2)

    estimate = phx.integration.integrate(
        function,
        target,
        phx.integration.MonteCarloPlan(4096),
        key=jr.key(1),
    )

    assert jnp.allclose(jnp.asarray(estimate.value.data), 1.0 / 3.0, atol=2e-2)
    assert estimate.error_kind == "iid-standard-error"
    # ty: ignore[unsupported-operator]
    assert estimate.error_estimate > 0.0
    assert estimate.diagnostics.num_independent_replicates == 1
    domain, target = _uniform_problem()
    function = domain.Function("x")(lambda x: x)
    plan = phx.integration.MonteCarloPlan(512, design=phx.integration.AntitheticDesign())

    estimate = phx.integration.integrate(function, target, plan, key=jr.key(2))

    assert jnp.allclose(jnp.asarray(estimate.value.data), 0.5, atol=1e-14)
    assert estimate.error_kind == "antithetic-pair-standard-error"
    assert estimate.diagnostics.num_pairs == 256
    assert estimate.diagnostics.pair_covariance < 0.0
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
    domain, _ = _uniform_problem()
    target = phx.integration.over(domain.component({"x": phx.domain.Boundary()}))
    plan = phx.integration.MonteCarloPlan(
        16,
        design=phx.integration.AntitheticDesign(),
    )

    with pytest.raises(TypeError, match=r"Interior\(\)"):
        phx.integration.materialize(target, plan, key=jr.key(17))
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
    domain, target = _uniform_problem()
    function = domain.Function("x")(lambda x: x**2)
    plan = phx.integration.MonteCarloPlan(
        256, design=phx.integration.LatinHypercubeDesign()
    )

    estimate = phx.integration.integrate(function, target, plan, key=jr.key(3))

    assert jnp.allclose(jnp.asarray(estimate.value.data), 1.0 / 3.0, atol=1e-3)
    assert estimate.error_estimate is None
    assert estimate.error_kind is None
    assert estimate.diagnostics.standard_error is None


def test_stochastic_integration_scenario_2() -> None:
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
    assert jnp.allclose(jnp.asarray(randomized.value.data), 1.0 / 3.0, atol=2e-4)
    assert randomized.error_kind == "randomized-qmc-replicate-error"
    # ty: ignore[unsupported-operator]
    assert randomized.error_estimate >= 0.0
    assert randomized.diagnostics.replicate_estimates.shape == (4,)
    probability = phx.domain.ProbabilityDomain(phx.uq.Normal(0.0, 1.0), label="z")
    function = probability.Function("z")(lambda z: z**2)
    plan = phx.integration.ImportanceSamplingPlan(8192, phx.uq.Normal(1.0, 2.0))

    estimate = phx.integration.integrate(
        function,
        phx.integration.expectation(probability),
        plan,
        key=jr.key(5),
    )

    assert jnp.allclose(jnp.asarray(estimate.value.data), 1.0, atol=5e-2)
    assert estimate.error_kind == "weighted-iid-standard-error"
    # ty: ignore[unsupported-operator]
    assert estimate.error_estimate > 0.0
    assert estimate.diagnostics.weights.weight_ess > 0.0
    assert jnp.allclose(estimate.diagnostics.normalizer_estimate, 1.0, atol=5e-2)
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
    probability = phx.domain.ProbabilityDomain(phx.uq.Normal(0.0, 1.0), label="z")
    target = phx.integration.expectation(probability)
    plan = phx.integration.ImportanceSamplingPlan(16, phx.uq.Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="requires key"):
        phx.integration.materialize(target, plan)
    samples = jnp.asarray([1.0, 2.0, 3.0])
    log_weights = jnp.log(jnp.asarray([1.0, 2.0, 1.0]))
    target = phx.integration.weighted(samples, log_weights)

    estimate = phx.integration.integrate(lambda values: values, target)

    assert jnp.allclose(jnp.asarray(estimate.value.data), 2.0, atol=1e-12)
    assert estimate.error_estimate is None
    assert estimate.error_kind is None
    assert jnp.allclose(estimate.diagnostics.weights.weight_ess, 8.0 / 3.0)


def test_stochastic_integration_scenario_3() -> None:
    domain, target = _uniform_problem()
    function = domain.Function("x")(lambda x: 3.0 * x + 2.0)
    control = domain.Function("x")(lambda x: x)
    estimator = phx.integration.ControlVariateEstimator(
        (control,), (0.5,), pilot_samples=64
    )
    plan = phx.integration.MonteCarloPlan(1024, control_variate=estimator)

    estimate = phx.integration.integrate(function, target, plan, key=jr.key(7))

    assert jnp.allclose(jnp.asarray(estimate.value.data), 3.5, atol=1e-10)
    # ty: ignore[unsupported-operator]
    assert estimate.error_estimate < 1e-10
    assert estimate.num_evaluations == 960
    square = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
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

    assert jnp.allclose(jnp.asarray(estimate.value.data), 4.0, atol=1e-12)
    assert estimate.error_kind == "stratified-standard-error"
    assert jnp.all(estimate.diagnostics.samples_per_stratum > 0)
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
    assert jnp.allclose(jnp.asarray(estimate.value.data), 1.0, atol=1e-12)
    restricted = phx.sampling.RandomizedQMCDesign(
        sequence="sobol",
        allow_arbitrary_count=False,
    )
    with pytest.raises(ValueError, match="power of two"):
        phx.sampling.materialize_design(
            restricted,
            count=6,
            dimension=2,
            key=jr.key(42),
        )

    arbitrary = phx.sampling.RandomizedQMCDesign(
        sequence="sobol",
        allow_arbitrary_count=True,
    )
    points = phx.sampling.materialize_design(
        arbitrary,
        count=6,
        dimension=2,
        key=jr.key(42),
    )
    assert points.shape == (6, 2)


def test_stochastic_integration_scenario_4() -> None:
    square = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
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

    assert jnp.array_equal(
        jnp.asarray(realization.batch.points["z"].data), expected_samples
    )
    assert jnp.array_equal(
        jr.key_data(realization.key),
        jr.key_data(evaluation_key),
    )
    assert jnp.allclose(
        jnp.asarray(estimate.value.data), jr.uniform(evaluation_key), atol=1e-14
    )


def test_stochastic_integration_scenario_5() -> None:
    target = phx.integration.weighted(
        jnp.arange(5.0),
        jnp.log(jnp.asarray([1.0, 2.0, 3.0, 2.0, 1.0])),
    )

    estimate = phx.integration.integrate(7.0, target)

    assert estimate.successful
    assert jnp.allclose(jnp.asarray(estimate.value.data), 7.0, atol=1e-14)
    assert estimate.num_evaluations == 5
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
    target = phx.integration.weighted(
        jnp.asarray([1.0, 2.0, 4.0]),
        jnp.log(jnp.asarray([1.0, 2.0, 1.0])),
        independent=False,
    )

    estimate = phx.integration.integrate(lambda values: values, target)

    assert estimate.successful
    assert estimate.diagnostics.standard_error is None
    assert estimate.diagnostics.normalizer_standard_error is None
    for sequence in ("sobol", "halton"):
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
        assert jnp.all(jnp.isfinite(jnp.asarray(realization.batch.points["z"].data)))
        assert estimate.successful
        assert jnp.all(jnp.isfinite(jnp.asarray(estimate.value.data)))


def test_stochastic_integration_scenario_6() -> None:
    distribution = _EndpointSensitiveNormal()
    probability = phx.domain.ProbabilityDomain(distribution, label="z")
    target = phx.integration.expectation(probability)
    plan = phx.integration.ImportanceSamplingPlan(128, distribution)

    estimate = phx.integration.integrate(1.0, target, plan, key=jr.key(20))

    assert estimate.successful
    assert estimate.status == int(phx.integration.IntegrationStatus.CONVERGED)
    upper = float(jnp.finfo(jnp.float64).max / 4.0)
    domain = phx.domain.ScalarInterval(0.0, upper, label="x")
    target = phx.integration.over(domain.component())
    function = domain.Function("x")(_AlternatingBatchIntegrand())

    estimate = phx.integration.integrate(
        function,
        target,
        phx.integration.MonteCarloPlan(16),
        key=jr.key(21),
    )

    assert jnp.all(jnp.isfinite(jnp.asarray(estimate.value.data)))
    assert jnp.all(jnp.isinf(estimate.diagnostics.standard_error))
    assert estimate.status == int(phx.integration.IntegrationStatus.NONFINITE_INTEGRAND)
    assert not estimate.successful
