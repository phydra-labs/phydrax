#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax._mass import EstimatedMass, ExactMass


class _HyperplaneMap(phx.geometry.BoundaryMap):
    @property
    def num_charts(self):
        return 1

    @property
    def reference_dimension(self):
        return 3

    @property
    def ambient_dimension(self):
        return 4

    def map(self, chart_indices, reference, /):
        del chart_indices
        return jnp.concatenate(
            (reference, jnp.zeros((*reference.shape[:-1], 1), dtype=reference.dtype)),
            axis=-1,
        )

    def jacobian(self, chart_indices, reference, /):
        del reference
        return jnp.ones(jnp.asarray(chart_indices).shape, dtype=jnp.float64)


def test_ball_and_orthotope_are_dimension_generic_with_exact_mass():
    ball = phx.geometry.Ball(jnp.zeros((4,)), 2.0, feature_id="ball4").compile()
    box = phx.geometry.Orthotope(
        jnp.zeros((5,)), jnp.asarray((1.0, 2.0, 3.0, 4.0, 5.0)), feature_id="box5"
    ).compile()

    assert jnp.allclose(ball.measure, 8.0 * jnp.pi**2)
    assert jnp.allclose(ball.boundary_measure, 16.0 * jnp.pi**2)
    assert isinstance(ball.interior_mass, ExactMass)
    assert jnp.allclose(box.measure, 120.0)
    assert jnp.allclose(
        box.boundary_measure,
        2.0 * 120.0 * jnp.sum(1.0 / jnp.asarray((1, 2, 3, 4, 5))),
    )
    samples = box.sample_boundary(128, key=jr.key(3)).points
    half_size = 0.5 * jnp.asarray((1.0, 2.0, 3.0, 4.0, 5.0))
    assert jnp.all(jnp.isclose(jnp.max(jnp.abs(samples / half_size), axis=-1), 1.0))


def test_axis_aligned_ellipsoid_has_truthful_nd_capabilities():
    geometry = phx.geometry.AxisAlignedEllipsoid(
        jnp.zeros((4,)),
        jnp.asarray((1.0, 2.0, 3.0, 4.0)),
    ).compile()

    assert jnp.allclose(geometry.measure, 12.0 * jnp.pi**2)
    assert geometry.has_capability(phx.geometry.GeometryCapability.INTERIOR_MEASURE)
    assert not geometry.has_capability(phx.geometry.GeometryCapability.BOUNDARY_MEASURE)
    samples = geometry.sample_interior(64, key=jr.key(4)).points
    assert jnp.all(
        jnp.sum((samples / jnp.asarray((1.0, 2.0, 3.0, 4.0))) ** 2, axis=-1) <= 1.0
    )


def test_estimated_boundary_mass_and_scalar_point_contract_are_explicit():
    ellipse = phx.geometry.Ellipse((0.0, 0.0), (2.0, 1.0)).compile()
    domain = phx.domain.GeometryDomain(ellipse)

    assert isinstance(ellipse.boundary_mass, EstimatedMass)
    assert ellipse.boundary_mass.evaluations == 72
    with pytest.raises(ValueError, match="Scalar geometry points"):
        domain.adf(jnp.asarray(0.5))


def test_extrusion_and_embedded_simplex_support_arbitrary_ambient_dimension():
    extrusion = phx.geometry.Extrusion(
        phx.geometry.Ball(jnp.zeros((4,)), 1.0),
        2.0,
    ).compile()
    triangle = phx.geometry.simplicial.AffineSimplexMap(
        jnp.asarray(
            (
                (0.0, 0.0, 0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0, 0.0),
            )
        )
    )

    assert extrusion.ambient_dimension == 5
    assert jnp.allclose(extrusion.measure, jnp.pi**2)
    assert triangle.ambient_dimension == 5
    assert jnp.allclose(triangle.evidence.measure, 0.5)
    assert jnp.allclose(triangle.evidence.jacobian_measure, 1.0)


def test_codimension_one_atlas_frame_and_partition_work_in_four_dimensions():
    atlas = phx.geometry.BoundaryAtlas(
        _HyperplaneMap(),
        source_entity_ids=jnp.asarray((0,), dtype=jnp.int32),
        source_id="hyperplane4",
    )
    reference = jnp.asarray(((0.2, 0.3, 0.4),))
    frame = atlas.frame(jnp.asarray((0,), dtype=jnp.int32), reference)
    partition = phx.geometry.BoundaryAtlasPartition(
        atlas,
        quadrature_order=3,
        maximum_quadrature_points=27,
    )

    assert bool(frame.regular[0])
    assert frame.tangents.shape == (1, 3, 4)
    assert jnp.allclose(frame.normal, jnp.asarray(((0.0, 0.0, 0.0, 1.0),)))
    assert jnp.allclose(partition.total_measure, 1.0)
    with pytest.raises(ValueError, match="maximum_quadrature_points"):
        phx.geometry.BoundaryAtlasPartition(
            atlas,
            quadrature_order=4,
            maximum_quadrature_points=63,
        )


def test_planar_wall_and_implicit_curve_cover_higher_and_lower_dimensions():
    wall = phx.geometry.PlanarWallFramePlan(
        jnp.zeros((4,)),
        jnp.asarray((1.0, 0.0, 0.0, 0.0)),
        jnp.asarray(
            (
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        ),
        gap=1.0,
        cross_section_area=1.0,
        length_unit_id="m",
        lower_wall_id="lower",
        upper_wall_id="upper",
    )
    coordinates = wall.coordinates(jnp.asarray(((0.5, 1.0, 2.0, 3.0),)))
    assert coordinates.tangential_coordinates.shape == (1, 3)

    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(19) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((-1.2, -1.2), (1.2, 1.2))))
    geometry = phx.geometry.Circle(
        (0.0, 0.0),
        0.73,
        feature_id="curve-source",
    ).compile()
    plan = phx.geometry.discover_implicit_curve(
        geometry,
        grid,
        source_id="circle-curve",
    )
    realization = plan.realize(geometry.state)
    curve = realization.to_segment_mesh()
    radius_index = geometry.schema.index(
        phx.geometry.ParameterId("curve-source", "radius")
    )
    derivative = jax.grad(
        lambda radius: jnp.sum(
            plan.realize(
                geometry.state.replace_at(radius_index, radius)
            ).proposed_vertices
        )
    )(jnp.asarray(0.73))

    assert bool(realization.accepted)
    assert curve.vertices.shape[1] == 2
    assert curve.topology.edges.shape[1] == 2
    assert jnp.isfinite(derivative)
