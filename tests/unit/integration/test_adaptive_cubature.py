import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _cube_moment(exponents):
    result = 1.0
    for exponent in exponents:
        if exponent % 2:
            return 0.0
        result *= 2.0 / (exponent + 1)
    return result


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("degree", [7, 9, 11, 13])
def test_genz_malik_rule_satisfies_declared_total_degree(dimension, degree):
    rule = phx.integration.GenzMalikRule(dimension, degree)
    points = np.asarray(rule.prepared.points)
    weights = np.asarray(rule.prepared.weights)
    embedded = np.asarray(rule.prepared.embedded_weights)

    for exponents in itertools.product(range(degree + 1), repeat=dimension):
        total_degree = sum(exponents)
        if total_degree > degree:
            continue
        values = np.prod(points ** np.asarray(exponents), axis=1)
        assert np.isclose(weights @ values, _cube_moment(exponents), atol=2e-10)
        if total_degree <= degree - 2:
            assert np.isclose(embedded @ values, _cube_moment(exponents), atol=2e-10)

    assert np.all(np.abs(points) < 1.0)
    assert np.allclose(np.sum(rule.prepared.split_weights, axis=1), 0.0)
    assert rule.rule_id == phx.integration.GenzMalikRule(dimension, degree).rule_id


def test_adaptive_rule_resource_guards_precede_expansion():
    with pytest.raises(ValueError, match="145 points"):
        phx.integration.GenzMalikRule(4, 9, maximum_points=144)
    with pytest.raises(ValueError, match="maximum_rule_bytes"):
        phx.integration.GenzMalikRule(3, 9, maximum_rule_bytes=1)
    with pytest.raises(ValueError, match="225 points"):
        phx.integration.TensorProductCubatureRule(
            phx.integration.GaussKronrodRule(15),
            dimension=2,
            maximum_points=224,
        )


def test_tensor_product_rule_shares_embedded_grid():
    rule = phx.integration.TensorProductCubatureRule(
        phx.integration.GaussKronrodRule(15), dimension=2
    )
    data = rule.prepared
    assert rule.num_points == 225
    assert np.isclose(np.sum(data.weights), 4.0)
    assert np.isclose(np.sum(data.embedded_weights), 4.0)
    assert np.allclose(np.sum(data.split_weights, axis=1), 0.0)
    assert rule.negative_weight_mass == 0.0


def test_adaptive_cubature_reports_rule_evidence_on_exact_polynomial():
    rule = phx.integration.GenzMalikRule(4, 9)
    plan = phx.integration.AdaptiveCubaturePlan(
        rule,
        absolute_tolerance=1e-8,
        max_cells=16,
        collect_partition=True,
        throw=False,
    )
    estimate = phx.integration.adaptive_cubature_callable(
        lambda x: (x[:, 0] * x[:, 1] + x[:, 2] ** 2)[:, None],
        plan,
        precision=phx.integration.IntegrationPrecisionPolicy(),
    )

    assert estimate.successful
    assert jnp.allclose(estimate.value, jnp.asarray([16.0 / 3.0]), atol=1e-12)
    assert int(estimate.num_evaluations) == rule.num_points
    assert estimate.error_kind == "embedded-cubature-indicator"
    assert estimate.provenance.realization == rule.rule_id
    assert estimate.diagnostics.rule_id == rule.rule_id
    assert estimate.diagnostics.partition.split_indicators.shape == (16, 4)


def test_breakpoints_and_batch_cap_are_observable_contracts():
    def integrand(points):
        if points.shape[0] > 7:
            raise ValueError("batch cap exceeded")
        return (jnp.abs(points[:, 0]) + points[:, 1] ** 2)[:, None]

    rule = phx.integration.GenzMalikRule(2, 9)
    plan = phx.integration.AdaptiveCubaturePlan(
        rule,
        breakpoints=((0.0,), ()),
        max_batch_points=7,
        absolute_tolerance=1e-10,
        max_cells=8,
        collect_partition=True,
        throw=False,
    )
    estimate = phx.integration.adaptive_cubature_callable(
        integrand,
        plan,
        precision=phx.integration.IntegrationPrecisionPolicy(),
    )

    assert estimate.successful
    assert jnp.allclose(estimate.value, jnp.asarray([10.0 / 3.0]), atol=1e-11)
    assert int(estimate.num_evaluations) == 2 * rule.num_points
    assert int(estimate.diagnostics.partition.count) == 2


def test_split_indicator_refines_the_difficult_axis():
    plan = phx.integration.AdaptiveCubaturePlan(
        phx.integration.GenzMalikRule(2, 9),
        absolute_tolerance=1e-14,
        relative_tolerance=0.0,
        max_cells=12,
        collect_partition=True,
        throw=False,
    )
    estimate = phx.integration.adaptive_cubature_callable(
        lambda x: jnp.exp(-200.0 * (x[:, 0] - 0.1) ** 2),
        plan,
        precision=phx.integration.IntegrationPrecisionPolicy(),
    )
    partition = estimate.diagnostics.partition
    widths = partition.upper_bounds - partition.lower_bounds
    active_widths = widths[partition.active]

    assert jnp.min(active_widths[:, 0]) < jnp.min(active_widths[:, 1])


def test_initial_evaluation_budget_is_never_overshot():
    rule = phx.integration.GenzMalikRule(2, 9)
    plan = phx.integration.AdaptiveCubaturePlan(
        rule,
        max_evaluations=rule.num_points - 1,
        collect_partition=True,
        throw=False,
    )
    estimate = phx.integration.adaptive_cubature_callable(
        lambda x: jnp.sum(x**2, axis=-1),
        plan,
        precision=phx.integration.IntegrationPrecisionPolicy(),
    )

    assert estimate.status == int(
        phx.integration.IntegrationStatus.MAXIMUM_EVALUATIONS_REACHED
    )
    assert int(estimate.num_evaluations) == 0
    assert int(estimate.diagnostics.partition.count) == 0


def test_nonfinite_integrand_fails_closed():
    plan = phx.integration.AdaptiveCubaturePlan(
        phx.integration.GenzMalikRule(2, 9), throw=False
    )
    estimate = phx.integration.adaptive_cubature_callable(
        lambda x: jnp.where(x[:, 0] > 0.0, jnp.nan, 1.0),
        plan,
        precision=phx.integration.IntegrationPrecisionPolicy(),
    )

    assert estimate.status == int(phx.integration.IntegrationStatus.NONFINITE_INTEGRAND)


def test_hyperrectangle_and_scalar_factors_share_one_cubature_layout():
    space = phx.domain.HyperRectangle([0.0, 0.0], [1.0, 2.0], label="x")
    time = phx.domain.ScalarInterval(0.0, 1.0, label="t")
    domain = phx.domain.ProductDomain(space, time)
    function = domain.Function("x", "t")(lambda x, t: jnp.sum(x**2, axis=-1) + t)
    plan = phx.integration.AdaptiveCubaturePlan(
        phx.integration.GenzMalikRule(3, 9),
        absolute_tolerance=1e-10,
        max_cells=8,
        throw=False,
    )

    estimate = phx.integration.integrate(
        function, phx.integration.over(domain.component()), plan
    )

    assert estimate.successful
    assert jnp.allclose(estimate.value.data, 13.0 / 3.0, atol=1e-11)


def test_physical_breakpoint_is_mapped_to_reference_cube():
    domain = phx.domain.ScalarInterval(0.0, 2.0, label="x")
    function = domain.Function("x")(lambda x: jnp.abs(x - 1.0))
    rule = phx.integration.GenzMalikRule(1, 9)
    plan = phx.integration.AdaptiveCubaturePlan(
        rule,
        breakpoints=((1.0,),),
        absolute_tolerance=1e-10,
        max_cells=4,
        throw=False,
    )

    estimate = phx.integration.integrate(
        function, phx.integration.over(domain.component()), plan
    )

    assert estimate.successful
    assert jnp.allclose(estimate.value.data, 1.0, atol=1e-12)
    assert int(estimate.num_evaluations) == 2 * rule.num_points


def test_adaptive_cubature_supports_jit_vmap_and_parameter_gradients():
    plan = phx.integration.AdaptiveCubaturePlan(
        phx.integration.GenzMalikRule(2, 9),
        absolute_tolerance=1e-10,
        max_cells=8,
        throw=False,
    )
    precision = phx.integration.IntegrationPrecisionPolicy()

    def integrate_parameter(parameter):
        return phx.integration.adaptive_cubature_callable(
            lambda x: parameter * (x[:, 0] ** 2 + x[:, 1] ** 2),
            plan,
            precision=precision,
        ).value

    compiled = jax.jit(integrate_parameter)(jnp.asarray(2.0))
    mapped = jax.vmap(integrate_parameter)(jnp.asarray([1.0, 2.0]))
    forward = jax.jacfwd(integrate_parameter)(jnp.asarray(2.0))
    reverse = jax.jacrev(integrate_parameter)(jnp.asarray(2.0))

    assert jnp.allclose(compiled, 16.0 / 3.0, atol=1e-11)
    assert jnp.allclose(mapped, jnp.asarray([8.0 / 3.0, 16.0 / 3.0]), atol=1e-11)
    assert jnp.allclose(forward, 8.0 / 3.0, atol=1e-11)
    assert jnp.allclose(reverse, 8.0 / 3.0, atol=1e-11)


def test_cusp_regression_cannot_report_false_convergence():
    coefficients = jnp.asarray(
        [0.2660088941584163, 0.4430043456880922, 0.4918098187571035]
    )
    centers = jnp.asarray([0.25354825920421914, 0.7997477160722586, 0.19929440330793777])
    axes = tuple(
        phx.domain.ScalarInterval(0.02, 0.97, label=label) for label in ("x", "y", "z")
    )
    domain = phx.domain.ProductDomain(*axes)
    function = domain.Function("x", "y", "z")(
        lambda x, y, z: jnp.exp(
            -jnp.sum(
                coefficients * jnp.abs(jnp.stack((x, y, z), axis=-1) - centers), axis=-1
            )
        )
    )
    plan = phx.integration.AdaptiveCubaturePlan(
        phx.integration.GenzMalikRule(3, 9),
        absolute_tolerance=1e-8,
        relative_tolerance=1e-8,
        max_cells=256,
        throw=False,
    )
    estimate = phx.integration.integrate(
        function, phx.integration.over(domain.component()), plan
    )
    exact = 0.5882689256199701
    tolerance = 1e-8 + 1e-8 * abs(exact)

    assert (not bool(estimate.successful)) or abs(
        float(estimate.value.data) - exact
    ) <= tolerance


def test_normalized_density_retains_positive_mass_contract():
    x = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    y = phx.domain.ScalarInterval(0.0, 1.0, label="y")
    domain = phx.domain.ProductDomain(x, y)
    function = domain.Function("x", "y")(lambda x, y: x + y)
    log_density = domain.Function("x", "y")(lambda x, y: 0.0 * (x + y))
    target = phx.integration.normalized_density(
        phx.integration.over(domain.component()), log_density
    )
    plan = phx.integration.AdaptiveCubaturePlan(
        phx.integration.GenzMalikRule(2, 9),
        absolute_tolerance=1e-10,
        max_cells=8,
        throw=False,
    )

    estimate = phx.integration.integrate(function, target, plan)

    assert estimate.successful
    assert jnp.allclose(estimate.value.data, 1.0, atol=1e-11)
    assert estimate.error_kind == "ratio-embedded-cubature-indicator"
