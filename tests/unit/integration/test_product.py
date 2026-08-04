import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _product_problem():
    space = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    time = phx.domain.ScalarInterval(0.0, 2.0, label="t")
    domain = phx.domain.ProductDomain(space, time)
    function = domain.Function("x", "t")(lambda x, t: x**2 + t)
    return domain, function


def test_deterministic_product_plan_composes_axis_rules():
    domain, function = _product_problem()
    plan = phx.integration.ProductIntegrationPlan(
        {
            "x": phx.integration.FixedQuadraturePlan(
                phx.integration.GaussLegendreRule(6)
            ),
            "t": phx.integration.FixedQuadraturePlan(
                phx.integration.ClenshawCurtisRule(4)
            ),
        }
    )

    estimate = phx.integration.integrate(
        function, phx.integration.over(domain.component()), plan
    )

    assert jnp.allclose(estimate.value.data, 8.0 / 3.0, atol=1e-12)
    assert estimate.error_estimate is None
    assert estimate.provenance.method == "product"


def test_sparse_grid_axis_group_composes_with_fixed_factor():
    x = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    y = phx.domain.ScalarInterval(0.0, 1.0, label="y")
    t = phx.domain.ScalarInterval(0.0, 2.0, label="t")
    domain = phx.domain.ProductDomain(x, y, t)
    function = domain.Function("x", "y", "t")(lambda x, y, t: x**2 + y**2 + t)
    plan = phx.integration.ProductIntegrationPlan(
        {
            ("x", "y"): phx.integration.SparseGridPlan(2, 4),
            "t": phx.integration.FixedQuadraturePlan(
                phx.integration.GaussLegendreRule(5)
            ),
        }
    )

    estimate = phx.integration.integrate(
        function, phx.integration.over(domain.component()), plan
    )

    assert jnp.allclose(estimate.value.data, 10.0 / 3.0, atol=1e-11)


def test_mixed_fixed_and_iid_plan_reports_only_stochastic_axis_error():
    domain, function = _product_problem()
    plan = phx.integration.ProductIntegrationPlan(
        {
            "x": phx.integration.FixedQuadraturePlan(
                phx.integration.GaussLegendreRule(8)
            ),
            "t": phx.integration.MonteCarloPlan(4096),
        }
    )

    estimate = phx.integration.integrate(
        function,
        phx.integration.over(domain.component()),
        plan,
        key=jr.key(1),
    )

    assert jnp.allclose(estimate.value.data, 8.0 / 3.0, atol=5e-2)
    assert estimate.error_kind == "iid-standard-error"
    assert estimate.error_estimate > 0.0


def test_mixed_qmc_needs_replicates_for_uncertainty():
    domain, function = _product_problem()
    fixed = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(8))
    deterministic_plan = phx.integration.ProductIntegrationPlan(
        {
            "x": fixed,
            "t": phx.integration.QuasiMonteCarloPlan(
                256, scrambled=False, num_replicates=1
            ),
        }
    )
    randomized_plan = phx.integration.ProductIntegrationPlan(
        {
            "x": fixed,
            "t": phx.integration.QuasiMonteCarloPlan(256, num_replicates=4),
        }
    )

    deterministic = phx.integration.integrate(
        function, phx.integration.over(domain.component()), deterministic_plan
    )
    randomized = phx.integration.integrate(
        function,
        phx.integration.over(domain.component()),
        randomized_plan,
        key=jr.key(2),
    )

    assert deterministic.error_estimate is None
    assert deterministic.error_kind is None
    assert jnp.allclose(randomized.value.data, 8.0 / 3.0, atol=2e-3)
    assert randomized.error_kind == "randomized-qmc-replicate-error"


def test_grouped_qmc_uses_one_joint_reference_design():
    x = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    y = phx.domain.ScalarInterval(0.0, 1.0, label="y")
    domain = phx.domain.ProductDomain(x, y)
    function = domain.Function("x", "y")(lambda x, y: (x - y) ** 2)
    plan = phx.integration.ProductIntegrationPlan(
        {
            ("x", "y"): phx.integration.QuasiMonteCarloPlan(
                8,
                scrambled=False,
                num_replicates=1,
            )
        }
    )

    realization = phx.integration.materialize(
        phx.integration.over(domain.component()),
        plan,
    )
    estimate = phx.integration.reduce(function, realization)

    x_points = realization.batch.batches[0].points["x"].data
    y_points = realization.batch.batches[0].points["y"].data
    assert not jnp.array_equal(x_points, y_points)
    assert estimate.value.data == pytest.approx(0.125)


def test_product_plan_requires_exact_nonfixed_label_coverage():
    domain, function = _product_problem()
    plan = phx.integration.ProductIntegrationPlan(
        {"x": phx.integration.FixedQuadraturePlan()}
    )

    with pytest.raises(ValueError, match="cover every interior label"):
        phx.integration.integrate(
            function, phx.integration.over(domain.component()), plan
        )


def test_product_density_normalization_uses_full_product_measure():
    domain, _ = _product_problem()
    function = domain.Function("x", "t")(lambda x, t: x + t)
    log_density = domain.Function("x", "t")(lambda x, t: jnp.log1p(x) + jnp.log1p(t))
    target = phx.integration.normalized_density(
        phx.integration.over(domain.component()), log_density
    )
    plan = phx.integration.ProductIntegrationPlan(
        {
            "x": phx.integration.FixedQuadraturePlan(
                phx.integration.GaussLegendreRule(8)
            ),
            "t": phx.integration.FixedQuadraturePlan(
                phx.integration.GaussLegendreRule(8)
            ),
        }
    )

    estimate = phx.integration.integrate(function, target, plan)

    assert jnp.allclose(estimate.value.data, 31.0 / 18.0, atol=1e-12)


def test_product_plan_preserves_unintegrated_target_axes():
    domain, function = _product_problem()
    plan = phx.integration.ProductIntegrationPlan(
        {
            "x": phx.integration.FixedQuadraturePlan(
                phx.integration.GaussLegendreRule(6)
            ),
            "t": phx.integration.FixedQuadraturePlan(
                phx.integration.GaussLegendreRule(5)
            ),
        }
    )
    target = phx.integration.over(domain.component(), axes="x")
    realization = phx.integration.materialize(target, plan)

    estimate = phx.integration.reduce(function, realization)
    time_points = realization.batch.batches[0].points.points["t"]

    assert estimate.value.dims == time_points.dims
    assert jnp.allclose(estimate.value.data, 1.0 / 3.0 + time_points.data, atol=1e-12)


def test_product_plan_integrates_multiple_complete_axis_blocks():
    x = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    y = phx.domain.ScalarInterval(0.0, 1.0, label="y")
    t = phx.domain.ScalarInterval(0.0, 1.0, label="t")
    domain = phx.domain.ProductDomain(x, y, t)
    function = domain.Function("x", "y", "t")(lambda x, y, t: x + y + t)
    plan = phx.integration.ProductIntegrationPlan(
        {
            label: phx.integration.FixedQuadraturePlan(
                phx.integration.GaussLegendreRule(3)
            )
            for label in ("x", "y", "t")
        }
    )
    target = phx.integration.over(domain.component(), axes=("x", "y"))
    realization = phx.integration.materialize(target, plan)

    estimate = phx.integration.reduce(function, realization)
    time_points = realization.batch.batches[0].points.points["t"]

    assert estimate.value.dims == time_points.dims
    assert jnp.allclose(estimate.value.data, 1.0 + time_points.data, atol=1e-12)


def test_product_plan_rejects_unsupported_control_variates():
    domain, function = _product_problem()
    control = domain.Function("t")(lambda t: t)
    estimator = phx.integration.ControlVariateEstimator(
        (control,),
        (1.0,),
        coefficients=jnp.asarray([1.0]),
    )
    plan = phx.integration.ProductIntegrationPlan(
        {
            "x": phx.integration.FixedQuadraturePlan(),
            "t": phx.integration.MonteCarloPlan(
                32,
                control_variate=estimator,
            ),
        }
    )

    with pytest.raises(ValueError, match="does not support control variates"):
        phx.integration.integrate(
            function,
            phx.integration.over(domain.component()),
            plan,
            key=jr.key(12),
        )


def test_randomized_product_materialization_is_jittable():
    domain, _ = _product_problem()
    target = phx.integration.over(domain.component())
    plan = phx.integration.ProductIntegrationPlan(
        {
            "x": phx.integration.FixedQuadraturePlan(
                phx.integration.GaussLegendreRule(4)
            ),
            "t": phx.integration.MonteCarloPlan(
                16,
                design=phx.integration.LatinHypercubeDesign(),
            ),
        }
    )

    @jax.jit
    def materialized_time_points(key):
        realization = phx.integration.materialize(target, plan, key=key)
        return realization.batch.batches[0].points.points["t"].data

    points = materialized_time_points(jr.key(18))

    assert points.shape == (16,)
    assert jnp.all(jnp.isfinite(points))


def test_mixed_product_density_preserves_a_normalized_component_base():
    space = phx.domain.ScalarInterval(0.0, 2.0, label="x")
    time = phx.domain.ScalarInterval(0.0, 3.0, label="t")
    domain = phx.domain.ProductDomain(space, time)
    target = phx.integration.density(
        phx.integration.mean_over(domain.component()),
        domain.Function()(lambda: 0.0),
    )
    plan = phx.integration.ProductIntegrationPlan(
        {
            "x": phx.integration.FixedQuadraturePlan(
                phx.integration.GaussLegendreRule(4)
            ),
            "t": phx.integration.MonteCarloPlan(32),
        }
    )

    estimate = phx.integration.integrate(1.0, target, plan, key=jr.key(19))

    assert estimate.successful
    assert jnp.allclose(estimate.value.data, 1.0, atol=1e-12)


def test_open_probability_product_materialization_is_jittable():
    probability = phx.domain.ProbabilityDomain(
        phx.uq.Normal(0.0, 1.0),
        label="z",
    )
    target = phx.integration.over(probability.component())
    plan = phx.integration.ProductIntegrationPlan(
        {"z": phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(8))}
    )

    @jax.jit
    def materialized_probability_points():
        realization = phx.integration.materialize(target, plan)
        return realization.batch.batches[0].points.points["z"].data

    points = materialized_probability_points()

    assert points.shape == (8,)
    assert jnp.all(jnp.isfinite(points))


@pytest.mark.parametrize(
    "factor_plan",
    (
        phx.integration.FixedQuadraturePlan(phx.integration.ClenshawCurtisRule(3)),
        phx.integration.SparseGridPlan(1, 2),
    ),
)
def test_endpoint_inclusive_probability_product_requires_bounded_support(
    factor_plan,
):
    probability = phx.domain.ProbabilityDomain(
        phx.uq.Normal(0.0, 1.0),
        label="z",
    )
    target = phx.integration.over(probability.component())
    plan = phx.integration.ProductIntegrationPlan({"z": factor_plan})

    with pytest.raises(ValueError, match="bounded probability support"):
        phx.integration.materialize(target, plan)
