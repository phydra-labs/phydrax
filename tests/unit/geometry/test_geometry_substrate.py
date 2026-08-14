#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_design_state_updates_preserve_schema_and_tree_structure():
    source = phx.geometry.Sphere(
        center=(0.0, 0.0, 0.0),
        radius=1.0,
        feature_id="body",
    )
    compiled = source.compile()
    radius_id = phx.geometry.ParameterId("body", "radius")
    radius_index = compiled.schema.index(radius_id)

    updated = compiled.with_parameters({radius_id: 2.0})

    assert updated.schema is compiled.schema
    assert jax.tree_util.tree_structure(updated.state) == jax.tree_util.tree_structure(
        compiled.state
    )
    assert compiled.boundary_field(jnp.array([1.5, 0.0, 0.0])) == pytest.approx(0.5)
    assert updated.boundary_field(jnp.array([1.5, 0.0, 0.0])) == pytest.approx(-0.5)

    def volume(radius):
        state = compiled.state.replace_at(radius_index, radius)
        return compiled.kernel.measure(state)

    assert jax.grad(volume)(jnp.asarray(1.0)) == pytest.approx(4.0 * jnp.pi)
    evaluate = jax.jit(lambda point: updated.boundary_field(point))
    assert evaluate(jnp.array([2.0, 0.0, 0.0])) == pytest.approx(0.0)


def test_field_certificates_propagate_through_translation_and_sharp_union():
    left = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        1.0,
        feature_id="left",
    )
    translated = left.translated((2.0, 0.0, 0.0)).compile()
    union = (
        left
        | phx.geometry.Sphere(
            (2.0, 0.0, 0.0),
            1.0,
            feature_id="right",
        )
    ).compile()

    assert translated.field_certificate.is_signed_distance
    assert translated.field_certificate.provenance[-1] == "rigid_translation"
    assert (
        union.field_certificate.distance_semantics
        is phx.geometry.DistanceSemantics.LEVEL_SET
    )
    assert (
        union.field_certificate.sign_reliability is phx.geometry.SignReliability.RELIABLE
    )
    assert union.contains(jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])).all()
    with pytest.raises(NotImplementedError, match="signed_distance"):
        union.signed_distance(jnp.array([0.0, 0.0, 0.0]))


def test_analytic_domains_have_exact_measures_queries_and_fixed_shape_samples():
    circle = phx.domain.GeometryDomain(phx.geometry.Circle((1.0, -1.0), 2.0).compile())
    sphere = phx.domain.GeometryDomain(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 2.0).compile()
    )
    box = phx.domain.GeometryDomain(
        phx.geometry.Box((0.0, 0.0, 0.0), (2.0, 3.0, 4.0)).compile()
    )

    assert circle.area == pytest.approx(4.0 * jnp.pi)
    assert circle.boundary_measure == pytest.approx(4.0 * jnp.pi)
    assert sphere.volume == pytest.approx((32.0 / 3.0) * jnp.pi)
    assert sphere.boundary_measure == pytest.approx(16.0 * jnp.pi)
    assert box.volume == pytest.approx(24.0)
    assert box.boundary_measure == pytest.approx(52.0)

    key = jax.random.key(4)
    assert circle.sample_interior(1, key=key).shape == (1, 2)
    assert sphere.sample_boundary(1, key=key).shape == (1, 3)
    box_boundary = box.sample_boundary(128, key=key)
    assert box_boundary.shape == (128, 3)
    assert jnp.all(box._on_boundary(box_boundary))


def test_bounded_rejection_reports_failure_without_hanging_or_underfilling_silently():
    plan = phx.geometry.RejectionSamplingPlan(
        proposals_per_round=4,
        maximum_rounds=3,
    )

    def run(key):
        return phx.geometry.bounded_rejection_sample(
            lambda _key, count: jnp.zeros((count, 2)),
            lambda points: jnp.zeros((points.shape[0],), dtype=bool),
            num_points=5,
            point_dimension=2,
            key=key,
            plan=plan,
        )

    result = jax.jit(run)(jax.random.key(0))

    assert result.points.shape == (5, 2)
    assert not result.valid.any()
    assert not result.report.complete
    assert result.report.accepted == 0
    assert result.report.proposed == plan.maximum_proposals
    assert result.report.rounds == plan.maximum_rounds


def test_triangle_mesh_query_index_returns_closest_features_and_surface_atlas():
    mesh = phx.geometry.TriangleMesh(
        vertices=jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        faces=jnp.array([[0, 1, 2]]),
        source_id="plate",
    )
    index = mesh.query_index()
    query = jax.jit(lambda points: index.query(points))(
        jnp.array([[0.25, 0.25, 1.0], [2.0, 0.0, 0.0]])
    )

    assert mesh.measure == pytest.approx(0.5)
    assert jnp.allclose(
        query.closest_point,
        jnp.array([[0.25, 0.25, 0.0], [1.0, 0.0, 0.0]]),
    )
    assert jnp.allclose(query.distance, jnp.array([1.0, 1.0]))
    assert jnp.array_equal(query.face_index, jnp.array([0, 0]))

    reference = jnp.array([[[0.5, 0.5]]])
    physical = mesh.boundary_atlas.map(jnp.array([[0]]), reference)
    jacobian = mesh.boundary_atlas.jacobian(jnp.array([[0]]), reference)
    assert jnp.allclose(physical, jnp.array([[[0.5, 0.25, 0.0]]]))
    assert jnp.allclose(jacobian, jnp.array([[0.5]]))


def test_boundary_atlas_integration_and_hard_constraint_work_end_to_end():
    source = phx.geometry.Circle(
        (0.0, 0.0),
        1.5,
        feature_id="integration-circle",
    ).translated((2.0, -1.0))
    geometry = phx.domain.GeometryDomain(source.compile())
    boundary = geometry.component({"x": phx.domain.Boundary()})
    structure = phx.domain.SampleLayout((("x",),))

    plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(5))
    integral = phx.integration.integrate(
        1.0,
        phx.integration.over(boundary),
        plan,
    )
    assert integral.value.data == pytest.approx(geometry.boundary_measure)

    @geometry.Function("x")
    def raw_field(x):
        return jnp.sum(x * x)

    enforced = phx.enforcement.enforce_dirichlet(raw_field, boundary, target=3.0)
    points = boundary.sample(
        phx.domain.PointSampling(32, layout=structure, design="uniform"),
        key=jax.random.key(7),
    )
    values = enforced(points).data
    assert jnp.allclose(values, 3.0, atol=1e-10)
