import math

import jax.numpy as jnp
import pytest

import phydrax as phx


def _integrate_geometry(source, reference, degree, *, boundary=False, integrand=1.0):
    domain = phx.domain.GeometryDomain(source.compile())
    component = domain.component({"x": phx.domain.Boundary()} if boundary else None)
    rule = phx.integration.CubatureRule(reference, degree)
    estimate = phx.integration.integrate(
        integrand if not callable(integrand) else domain.Function("x")(integrand),
        phx.integration.over(component),
        phx.integration.FixedQuadraturePlan(rule),
    )
    return domain, estimate


def test_native_disk_and_circle_rules_preserve_radial_moments():
    source = phx.geometry.Circle((0.0, 0.0), 2.0)
    _, area = _integrate_geometry(source, "disk", 6)
    _, radial_second = _integrate_geometry(
        source,
        "disk",
        6,
        integrand=lambda x: x[0] ** 2 + x[1] ** 2,
    )
    _, perimeter = _integrate_geometry(source, "circle", 5, boundary=True)

    assert area.value.data == pytest.approx(4.0 * math.pi, rel=2e-13)
    assert radial_second.value.data == pytest.approx(8.0 * math.pi, rel=2e-13)
    assert perimeter.value.data == pytest.approx(4.0 * math.pi, rel=2e-13)
    assert area.num_evaluations < 100


def test_native_sphere_rule_is_rotationally_balanced():
    source = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0)
    domain = phx.domain.GeometryDomain(source.compile())
    boundary = domain.component({"x": phx.domain.Boundary()})
    rule = phx.integration.CubatureRule("sphere", 3)
    realization = phx.integration.materialize(
        phx.integration.over(boundary),
        phx.integration.FixedQuadraturePlan(rule),
    )
    moments = tuple(
        phx.integration.reduce(
            domain.Function("x")(lambda x, axis=axis: x[axis] ** 2),
            realization,
        ).value.data
        for axis in range(3)
    )

    assert rule.num_points == 6
    assert jnp.asarray(moments) == pytest.approx(
        jnp.full((3,), 4.0 * math.pi / 3.0), rel=2e-13, abs=2e-13
    )


def test_native_ball_rule_preserves_volume_and_second_moment():
    source = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0)
    _, volume = _integrate_geometry(source, "ball", 4)
    _, x_second = _integrate_geometry(
        source,
        "ball",
        4,
        integrand=lambda x: x[0] ** 2,
    )

    assert volume.value.data == pytest.approx(4.0 * math.pi / 3.0, rel=2e-13)
    assert x_second.value.data == pytest.approx(4.0 * math.pi / 15.0, rel=2e-13)


def test_translation_and_uniform_scaling_preserve_native_cubature():
    source = phx.geometry.Circle((0.0, 0.0), 1.0).scaled(2.0).translated((3.0, -4.0))
    _, area = _integrate_geometry(source, "disk", 4)
    assert area.value.data == pytest.approx(4.0 * math.pi, rel=2e-13)

    nonuniform = phx.domain.GeometryDomain(
        phx.geometry.Circle((0.0, 0.0), 1.0).scaled((2.0, 1.0)).compile()
    )
    with pytest.raises(NotImplementedError, match="cubature"):
        phx.integration.materialize(
            phx.integration.over(nonuniform.component()),
            phx.integration.FixedQuadraturePlan(phx.integration.CubatureRule("disk", 4)),
        )


def _tetrahedron_region():
    vertices = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    faces = jnp.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))
    return phx.domain.GeometryDomain(phx.geometry.MeshRegion(vertices, faces).compile())


def test_mesh_boundary_cubature_matches_exact_face_measure_and_selection():
    region = _tetrahedron_region()
    boundary = region.component({"x": phx.domain.Boundary()})
    rule = phx.integration.CubatureRule("triangle", 5)
    estimate = phx.integration.integrate(
        1.0,
        phx.integration.over(boundary),
        phx.integration.FixedQuadraturePlan(rule),
    )
    selected = region.component({"x": phx.domain.Boundary(entity_ids=(0,))})
    selected_estimate = phx.integration.integrate(
        1.0,
        phx.integration.over(selected),
        phx.integration.FixedQuadraturePlan(rule),
    )

    assert estimate.value.data == pytest.approx(float(region.boundary_measure), rel=2e-13)
    assert selected_estimate.value.data == pytest.approx(0.5, rel=2e-13)
    assert estimate.num_evaluations == 4 * rule.num_points


def test_native_geometry_cubature_composes_in_product_plans():
    space = phx.domain.GeometryDomain(
        phx.geometry.Circle((0.0, 0.0), 1.0).compile(), label="x"
    )
    time = phx.domain.ScalarInterval(0.0, 2.0, label="t")
    domain = phx.domain.ProductDomain(space, time)
    plan = phx.integration.ProductIntegrationPlan(
        {
            "x": phx.integration.FixedQuadraturePlan(
                phx.integration.CubatureRule("disk", 4)
            ),
            "t": phx.integration.FixedQuadraturePlan(
                phx.integration.GaussLegendreRule(3)
            ),
        }
    )
    estimate = phx.integration.integrate(
        1.0,
        phx.integration.over(domain.component()),
        plan,
    )
    assert estimate.value.data == pytest.approx(2.0 * math.pi, rel=2e-13)
