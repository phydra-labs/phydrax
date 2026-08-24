import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


@pytest.mark.parametrize(
    "rule",
    [
        phx.integration.GaussKronrodRule(21),
        phx.integration.ClenshawCurtisRule(4),
        phx.integration.TanhSinhRule(3),
    ],
)
def test_adaptive_rules_share_one_plan_and_estimate_contract(rule):
    domain = phx.domain.ScalarInterval(-1.0, 2.0, label="x")
    function = domain.Function("x")(lambda x: x**4 - 2.0 * x + 1.0)
    plan = phx.integration.AdaptiveQuadraturePlan(
        rule,
        absolute_tolerance=1e-9,
        relative_tolerance=1e-9,
        max_intervals=32,
        collect_partition=True,
    )

    estimate = phx.integration.integrate(
        function, phx.integration.over(domain.component()), plan
    )

    assert estimate.successful
    assert jnp.allclose(jnp.asarray(estimate.value.data), 6.6, atol=1e-8)
    assert estimate.error_kind == "embedded-rule"
    assert estimate.error_estimate is not None
    assert estimate.diagnostics.partition is not None


def test_adaptive_callable_interval_reuses_plan_and_diagnostics():
    plan = phx.integration.AdaptiveQuadraturePlan(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        max_intervals=16,
        collect_partition=True,
    )
    estimate = phx.integration.adaptive_interval_callable(
        lambda points: points**4 - 2.0 * points + 1.0,
        jnp.asarray((-1.0, 2.0)),
        plan,
    )

    assert estimate.successful
    assert jnp.allclose(estimate.value, 6.6, atol=1e-10)
    assert estimate.error_kind == "embedded-rule"
    assert estimate.provenance.target == "callable"
    assert estimate.diagnostics.partition is not None


def test_adaptive_breakpoints_cover_each_initial_subinterval():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    function = domain.Function("x")(lambda x: jnp.abs(x - 0.3))
    plan = phx.integration.AdaptiveQuadraturePlan(
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
        breakpoints=(0.3,),
        max_intervals=8,
        collect_partition=True,
    )

    estimate = phx.integration.integrate(
        function, phx.integration.over(domain.component()), plan
    )

    assert estimate.successful
    assert jnp.allclose(jnp.asarray(estimate.value.data), 0.29, atol=1e-11)
    assert estimate.diagnostics.partition.count >= 2


def test_adaptive_resource_failure_is_status_when_throw_is_false():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    function = domain.Function("x")(lambda x: jnp.sin(100.0 * x))
    plan = phx.integration.AdaptiveQuadraturePlan(
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
        max_intervals=1,
        throw=False,
    )

    estimate = phx.integration.integrate(
        function, phx.integration.over(domain.component()), plan
    )

    assert estimate.status == int(
        phx.integration.IntegrationStatus.MAXIMUM_INTERVALS_REACHED
    )
    assert not estimate.successful


def test_adaptive_nonfinite_integrand_has_distinct_status():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    function = domain.Function("x")(lambda x: jnp.where(x > 0.5, jnp.nan, x))
    plan = phx.integration.AdaptiveQuadraturePlan(max_intervals=4, throw=False)

    estimate = phx.integration.integrate(
        function, phx.integration.over(domain.component()), plan
    )

    assert estimate.status == int(phx.integration.IntegrationStatus.NONFINITE_INTEGRAND)


def test_adaptive_execution_is_jittable_and_differentiable():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    target = phx.integration.over(domain.component())
    plan = phx.integration.AdaptiveQuadraturePlan(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        max_intervals=16,
    )

    def objective(coefficient):
        function = domain.Function("x")(lambda x: coefficient * x**2)
        return phx.integration.integrate(function, target, plan).value.data

    value = jax.jit(objective)(3.0)
    tangent = jax.jvp(objective, (3.0,), (1.0,))[1]
    gradient = jax.grad(objective)(3.0)

    assert jnp.allclose(value, 1.0, atol=1e-12)
    assert jnp.allclose(tangent, 1.0 / 3.0, atol=1e-12)
    assert jnp.allclose(gradient, 1.0 / 3.0, atol=1e-12)


def test_normalized_adaptive_density_uses_density_in_both_ratio_terms():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    function = domain.Function("x")(lambda x: x)
    log_density = domain.Function("x")(lambda x: jnp.log1p(x))
    target = phx.integration.normalized_density(
        phx.integration.over(domain.component()), log_density
    )

    estimate = phx.integration.integrate(
        function,
        target,
        phx.integration.AdaptiveQuadraturePlan(
            absolute_tolerance=1e-11,
            relative_tolerance=1e-11,
        ),
    )

    assert estimate.successful
    assert jnp.allclose(jnp.asarray(estimate.value.data), 5.0 / 9.0, atol=1e-10)
    assert estimate.error_kind == "ratio-embedded-rule"


def test_adaptive_initial_rule_never_exceeds_evaluation_budget():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    function = domain.Function("x")(lambda x: jnp.ones_like(x))
    plan = phx.integration.AdaptiveQuadraturePlan(
        max_evaluations=1,
        throw=False,
    )

    estimate = phx.integration.integrate(
        function,
        phx.integration.over(domain.component()),
        plan,
    )

    assert estimate.num_evaluations == 1
    assert estimate.status == int(
        phx.integration.IntegrationStatus.MAXIMUM_EVALUATIONS_REACHED
    )
    assert not estimate.successful


def test_adaptive_ratio_rechecks_the_propagated_error_tolerance():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    function = domain.Function("x")(lambda x: jnp.exp(3.0 * x))
    log_density = domain.Function("x")(lambda x: -5.0 + 2.0 * x)
    target = phx.integration.normalized_density(
        phx.integration.over(domain.component()),
        log_density,
    )
    plan = phx.integration.AdaptiveQuadraturePlan(
        phx.integration.ClenshawCurtisRule(1),
        absolute_tolerance=0.5,
        relative_tolerance=0.0,
        max_intervals=1,
        throw=False,
    )

    estimate = phx.integration.integrate(function, target, plan)

    assert estimate.error_estimate > plan.absolute_tolerance
    assert estimate.status == int(phx.integration.IntegrationStatus.REFINEMENT_STAGNATION)
    assert not estimate.successful


def test_adaptive_component_sum_applies_aggregate_throw_policy():
    x = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    t = phx.domain.ScalarInterval(0.0, 1.0, label="t")
    domain = phx.domain.ProductDomain(x, t)
    union = phx.domain.ComponentSum(
        (
            domain.component({"t": phx.domain.FixedStart()}),
            domain.component({"t": phx.domain.FixedEnd()}),
        )
    )
    function = domain.Function("x", "t")(lambda x, t: (1.0 - 2.0 * t) * x**2)
    target = phx.integration.over(union)
    plan = phx.integration.AdaptiveQuadraturePlan(
        phx.integration.ClenshawCurtisRule(1),
        absolute_tolerance=0.0,
        relative_tolerance=0.51,
        max_intervals=1,
        throw=False,
    )

    estimate = phx.integration.integrate(function, target, plan)

    assert estimate.error_estimate > plan.relative_tolerance * jnp.abs(
        estimate.value.data
    )
    assert estimate.status == int(phx.integration.IntegrationStatus.REFINEMENT_STAGNATION)
    assert not estimate.successful

    throwing_plan = phx.integration.AdaptiveQuadraturePlan(
        phx.integration.ClenshawCurtisRule(1),
        absolute_tolerance=0.0,
        relative_tolerance=0.51,
        max_intervals=1,
        throw=True,
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="component-sum"):
        throwing_estimate = phx.integration.integrate(function, target, throwing_plan)
        jax.block_until_ready(throwing_estimate.value.data)
