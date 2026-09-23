#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_parameter_bounds_and_superquadric_conditions_are_executable():
    sphere = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        1.0,
        feature_id="sphere",
    ).compile()
    radius = sphere.schema.index(phx.geometry.ParameterId("sphere", "radius"))
    invalid_sphere = sphere.state.replace_at(radius, jnp.asarray(-1.0))

    assert bool(sphere.validity().accepted)
    assert not bool(sphere.validity(invalid_sphere).accepted)
    assert int(sphere.validity(invalid_sphere).disposition) == int(
        phx.geometry.GeometryValidityDisposition.INVALID
    )

    superquadric = (
        phx.geometry.Superquadric(
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 3.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
            first_blockiness=2.5,
            second_blockiness=3.0,
            feature_id="sq",
        )
        .translated((1.0, 0.0, 0.0))
        .compile()
    )
    orientation = superquadric.schema.index(phx.geometry.ParameterId("sq", "orientation"))
    invalid_orientation = superquadric.state.replace_at(
        orientation,
        jnp.zeros((4,)),
    )

    assert bool(superquadric.validity().accepted)
    assert not bool(superquadric.validity(invalid_orientation).accepted)


def test_geometry_validity_is_jittable_and_keeps_fixed_evidence_shape():
    compiled = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        1.0,
        feature_id="body",
    ).compile()
    radius = compiled.schema.index(phx.geometry.ParameterId("body", "radius"))

    def evaluate(value):
        state = compiled.state.replace_at(radius, value)
        evidence = compiled.validity(state)
        return evidence.accepted, evidence.margins

    accepted, margins = jax.jit(evaluate)(jnp.asarray(2.0))
    rejected, rejected_margins = jax.jit(evaluate)(jnp.asarray(-0.5))

    assert bool(accepted)
    assert not bool(rejected)
    assert margins.shape == rejected_margins.shape
    assert jnp.all(jnp.isfinite(margins))


def test_live_topology_classifications_reject_invalid_design_states():
    polygon = phx.geometry.Polygon(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        feature_id="polygon",
    ).compile()
    polygon_index = polygon.schema.index(phx.geometry.ParameterId("polygon", "vertices"))
    invalid_vertices = (
        jnp.asarray(((0.0, 0.0), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        jnp.asarray(((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0))),
        polygon.state.values[polygon_index][::-1],
    )
    assert all(
        not bool(
            polygon.validity(polygon.state.replace_at(polygon_index, vertices)).accepted
        )
        for vertices in invalid_vertices
    )

    full_sources = (
        phx.geometry.Cylinder(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            0.5,
            feature_id="cylinder",
        ),
        phx.geometry.Cone(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            0.5,
            feature_id="cone",
        ),
        phx.geometry.Torus(
            (0.0, 0.0, 0.0),
            0.5,
            1.5,
            feature_id="torus",
        ),
    )
    for source in full_sources:
        geometry = source.compile()
        angle_index = next(
            index
            for index, spec in enumerate(geometry.schema.specs)
            if spec.parameter_id.name == "angle"
        )
        sector_state = geometry.state.replace_at(angle_index, jnp.asarray(jnp.pi))
        assert bool(geometry.validity().accepted)
        assert not bool(geometry.validity(sector_state).accepted)

    uniform_scale = (
        phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0, feature_id="scaled-sphere")
        .scaled((1.0, 1.0, 1.0))
        .compile()
    )
    scale_index = next(
        index
        for index, spec in enumerate(uniform_scale.schema.specs)
        if spec.parameter_id.name == "scale"
    )
    anisotropic_state = uniform_scale.state.replace_at(
        scale_index, jnp.asarray((2.0, 1.0, 1.0))
    )
    assert not bool(uniform_scale.validity(anisotropic_state).accepted)

    with pytest.raises(ValueError, match="top_extent"):
        phx.geometry.Wedge((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.0)
    wedge = phx.geometry.Wedge(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        0.5,
        feature_id="wedge",
    ).compile()
    top_index = wedge.schema.index(phx.geometry.ParameterId("wedge", "top_extent"))
    assert not bool(
        wedge.validity(wedge.state.replace_at(top_index, jnp.asarray(0.0))).accepted
    )


def test_geometry_wrappers_implement_every_advertised_capability():
    sphere = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0, feature_id="capability-sphere")
    translated_sphere = sphere.translated((1.0, 0.0, 0.0)).compile()
    curvature = translated_sphere.contact_curvature(jnp.asarray(((2.0, 0.0, 0.0),)))
    assert bool(curvature.valid[0])
    scaled_curvature = (
        sphere.scaled(2.0).compile().contact_curvature(jnp.asarray(((2.0, 0.0, 0.0),)))
    )
    assert bool(scaled_curvature.valid[0])

    superquadric = phx.geometry.Superquadric(
        (0.0, 0.0, 0.0),
        (1.0, 2.0, 3.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
        feature_id="support-superquadric",
    )
    direction = jnp.asarray(((1.0, 0.5, -0.25),))
    base_support = superquadric.compile().support_map(direction)
    translated = superquadric.translated((2.0, -1.0, 0.5)).compile()
    rigid = superquadric.transformed(
        phx.geometry.RigidFrame(jnp.eye(3), jnp.asarray((2.0, -1.0, 0.5)))
    ).compile()
    scaled = superquadric.scaled((2.0, 1.5, 0.5)).compile()
    offset = jnp.asarray((2.0, -1.0, 0.5))
    assert jnp.allclose(translated.support_map(direction), base_support + offset)
    assert jnp.allclose(rigid.support_map(direction), base_support + offset)
    assert jnp.all(jnp.isfinite(scaled.support_map(direction)))

    report = phx.geometry.ReconstructionReport(
        "synthetic",
        "identity",
        "digest",
        4,
        4,
        4,
        4,
        1,
        True,
        True,
        (0.0, 0.0, 0.0),
        (),
    )
    reconstructed = phx.geometry.ReconstructedGeometrySource(sphere, report).compile()
    closest = reconstructed.closest_point(jnp.asarray(((2.0, 0.0, 0.0),)))
    reconstructed_curvature = reconstructed.contact_curvature(
        jnp.asarray(((1.0, 0.0, 0.0),))
    )
    assert bool(closest.unique[0])
    assert bool(reconstructed_curvature.valid[0])
    assert reconstructed.cubature_atlas("boundary").num_charts > 0
