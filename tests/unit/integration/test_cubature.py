import itertools
import math

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_fixed_gauss_hermite_uses_matched_standard_normal_measure():
    normal = phx.domain.ProbabilityDomain(phx.uq.Normal(0.0, 1.0), label="z")
    function = normal.Function("z")(lambda z: z**20)
    plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussHermiteRule(11))

    estimate = phx.integration.integrate(
        function,
        phx.integration.expectation(normal),
        plan,
    )

    assert estimate.successful
    assert estimate.num_evaluations == 11
    assert estimate.error_estimate is None
    assert estimate.value.data == pytest.approx(
        float(math.prod(range(1, 20, 2))), rel=2e-13, abs=2e-13
    )


def test_gauss_hermite_requires_a_standard_normal_reference_transform():
    uniform = phx.domain.ProbabilityDomain(phx.uq.Uniform(0.0, 1.0), label="z")
    with pytest.raises(ValueError, match="standard-normal reference"):
        phx.integration.materialize(
            phx.integration.expectation(uniform),
            phx.integration.FixedQuadraturePlan(phx.integration.GaussHermiteRule(5)),
        )


def test_mapped_xiao_gimbutas_rules_preserve_mass_and_node_counts():
    triangle = phx.integration.CubatureRule("triangle", 10)
    triangle_target = phx.integration.mapped(
        triangle,
        lambda reference: reference,
        lambda reference: jnp.ones((reference.shape[0],)),
    )
    triangle_estimate = phx.integration.integrate(
        lambda point: point[:, 0] + point[:, 1],
        triangle_target,
        phx.integration.CellQuadraturePlan(triangle),
    )

    tetrahedron = phx.integration.CubatureRule("tetrahedron", 10)
    tetrahedron_target = phx.integration.mapped(
        tetrahedron,
        lambda reference: reference,
        lambda reference: jnp.ones((reference.shape[0],)),
    )
    tetrahedron_estimate = phx.integration.integrate(
        lambda point: jnp.sum(point, axis=-1),
        tetrahedron_target,
        phx.integration.CellQuadraturePlan(tetrahedron),
    )

    assert triangle.family == tetrahedron.family == "xiao-gimbutas"
    assert triangle.num_points == 25
    assert tetrahedron.num_points == 74
    assert triangle_estimate.value.data == pytest.approx(1.0 / 3.0, abs=2e-13)
    assert tetrahedron_estimate.value.data == pytest.approx(1.0 / 8.0, abs=2e-13)
    assert triangle_estimate.provenance.realization == triangle.rule_id
    assert triangle_estimate.error_estimate is None


def test_symmetric_triangle_cubature_is_vertex_permutation_invariant():
    vertices = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.2, 0.9)))
    rule = phx.integration.CubatureRule("triangle", 10)
    estimates = []
    for permutation in itertools.permutations(range(3)):
        triangle = vertices[jnp.asarray(permutation)]
        origin = triangle[0]
        matrix = jnp.stack((triangle[1] - origin, triangle[2] - origin), axis=1)
        target = phx.integration.mapped(
            rule,
            lambda reference, origin=origin, matrix=matrix: origin + reference @ matrix.T,
            lambda reference, matrix=matrix: jnp.full(
                (reference.shape[0],), jnp.linalg.det(matrix)
            ),
        )
        estimate = phx.integration.integrate(
            lambda point: jnp.exp(7.0 * point[:, 0] + 3.0 * point[:, 1]),
            target,
            phx.integration.CellQuadraturePlan(rule),
        )
        estimates.append(estimate.value.data)

    assert jnp.max(jnp.asarray(estimates)) - jnp.min(jnp.asarray(estimates)) < 2e-12


def test_cubature_mapping_is_jittable_and_differentiable():
    rule = phx.integration.CubatureRule("disk", 6)

    def objective(radius):
        target = phx.integration.mapped(
            rule,
            lambda reference: radius * reference,
            lambda reference: jnp.full((reference.shape[0],), radius**2),
        )
        return phx.integration.integrate(
            1.0,
            target,
            phx.integration.CellQuadraturePlan(rule),
        ).value.data

    value = jax.jit(objective)(2.0)
    derivative = jax.grad(objective)(2.0)
    assert value == pytest.approx(4.0 * math.pi, rel=2e-13)
    assert derivative == pytest.approx(4.0 * math.pi, rel=2e-13)


def test_simplex_fallback_is_explicit_and_positive():
    fallback = phx.integration.CubatureRule("triangle", 31)
    assert fallback.family == "duffy"
    assert fallback.exact_degree >= 31
    assert jnp.all(fallback.prepared.weights > 0.0)

    with pytest.raises(ValueError, match="maximum certified degree is 30"):
        phx.integration.CubatureRule("triangle", 31, allow_duffy_fallback=False)


def test_mixed_hermite_and_interval_product_preserves_measure():
    normal = phx.domain.ProbabilityDomain(phx.uq.Normal(0.0, 1.0), label="z")
    interval = phx.domain.ScalarInterval(0.0, 2.0, label="t")
    domain = phx.domain.ProductDomain(normal, interval)
    function = domain.Function("z", "t")(lambda z, t: z**2 + t)
    plan = phx.integration.ProductIntegrationPlan(
        {
            "z": phx.integration.FixedQuadraturePlan(phx.integration.GaussHermiteRule(3)),
            "t": phx.integration.FixedQuadraturePlan(
                phx.integration.GaussLegendreRule(3)
            ),
        }
    )

    estimate = phx.integration.integrate(
        function,
        phx.integration.over(domain.component()),
        plan,
    )
    assert estimate.value.data == pytest.approx(4.0, abs=2e-13)
