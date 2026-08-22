import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _problem():
    vertices = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    faces = jnp.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))
    domain = phx.domain.GeometryDomain(phx.geometry.MeshRegion(vertices, faces).compile())
    boundary = domain.component({"x": phx.domain.Boundary()})
    return domain, boundary


def test_adaptive_triangle_polynomial_converges_on_initial_partition():
    domain, boundary = _problem()
    function = domain.Function("x")(lambda x: jnp.sum(x * x))
    plan = phx.integration.AdaptiveTrianglePlan(
        max_cells=16,
        collect_partition=True,
    )

    estimate = phx.integration.integrate(
        function,
        phx.integration.over(boundary),
        plan,
    )

    assert estimate.successful
    assert estimate.error_kind == "paired-reference-rule"
    assert estimate.error_estimate < 2e-12
    assert estimate.diagnostics.partition.count == 4
    assert estimate.num_evaluations == 4 * (
        plan.low_rule.num_points + plan.high_rule.num_points
    )


def test_adaptive_triangle_reports_cell_exhaustion_without_throwing():
    domain, boundary = _problem()
    function = domain.Function("x")(lambda x: jnp.exp(8.0 * x[0]))
    plan = phx.integration.AdaptiveTrianglePlan(
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
        max_cells=4,
        throw=False,
        collect_partition=True,
    )

    estimate = phx.integration.integrate(
        function,
        phx.integration.over(boundary),
        plan,
    )
    assert estimate.status == int(phx.integration.IntegrationStatus.MAXIMUM_CELLS_REACHED)
    assert not estimate.successful
    assert estimate.diagnostics.partition.count == 4


def test_adaptive_triangle_initial_budget_is_never_exceeded():
    domain, boundary = _problem()
    plan = phx.integration.AdaptiveTrianglePlan(max_evaluations=1, throw=False)
    estimate = phx.integration.integrate(
        domain.Function("x")(lambda x: x[0]),
        phx.integration.over(boundary),
        plan,
    )

    assert estimate.status == int(
        phx.integration.IntegrationStatus.MAXIMUM_EVALUATIONS_REACHED
    )
    assert estimate.num_evaluations == 1


def test_adaptive_triangle_nonfinite_integrand_has_distinct_status():
    domain, boundary = _problem()
    plan = phx.integration.AdaptiveTrianglePlan(max_cells=4, throw=False)
    estimate = phx.integration.integrate(
        domain.Function("x")(lambda x: jnp.where(x[0] > 0.2, jnp.nan, x[0])),
        phx.integration.over(boundary),
        plan,
    )
    assert estimate.status == int(phx.integration.IntegrationStatus.NONFINITE_INTEGRAND)


def test_adaptive_triangle_normalized_target_uses_paired_mass_integral():
    domain, boundary = _problem()
    plan = phx.integration.AdaptiveTrianglePlan(max_cells=16)
    estimate = phx.integration.integrate(
        domain.Function("x")(lambda x: jnp.ones_like(x[0]) * 3.0),
        phx.integration.mean_over(boundary),
        plan,
    )
    assert estimate.successful
    assert estimate.value.data == pytest.approx(3.0, rel=2e-13)
    assert estimate.error_kind == "ratio-paired-reference-rule"


def test_adaptive_triangle_is_jittable_and_differentiable():
    domain, boundary = _problem()
    target = phx.integration.over(boundary)
    plan = phx.integration.AdaptiveTrianglePlan(max_cells=7)

    def objective(scale):
        function = domain.Function("x")(lambda x: scale * jnp.sum(x * x))
        return phx.integration.integrate(function, target, plan).value.data

    value = jax.jit(objective)(2.0)
    derivative = jax.grad(objective)(2.0)
    assert value == pytest.approx(2.0 * 0.9330127018922193, rel=2e-13)
    assert derivative == pytest.approx(0.9330127018922193, rel=2e-13)
