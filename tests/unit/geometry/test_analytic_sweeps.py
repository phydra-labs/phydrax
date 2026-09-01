#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_circle_extrusion_is_an_exact_centered_cylinder():
    compiled = (
        phx.geometry.Circle(
            (0.0, 0.0),
            2.0,
            feature_id="profile",
        )
        .extruded(6.0, feature_id="sweep")
        .compile()
    )
    points = jnp.asarray(
        [
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 4.0],
        ]
    )

    assert compiled.field_certificate.is_signed_distance
    assert compiled.has_capability(phx.geometry.GeometryCapability.MEASURE)
    assert jnp.allclose(
        compiled.signed_distance(points),
        jnp.asarray([0.0, 0.0, -2.0, jnp.sqrt(2.0)]),
        atol=1.0e-6,
    )
    assert compiled.measure == pytest.approx(24.0 * jnp.pi)
    assert compiled.boundary_measure == pytest.approx(32.0 * jnp.pi)
    assert bool(compiled.validity().accepted)


def test_rectangle_extrusion_matches_box_distance_and_has_finite_gradients():
    compiled = (
        phx.geometry.Rectangle(
            center=(0.0, 0.0),
            size=(2.0, 4.0),
            feature_id="profile",
        )
        .extruded(6.0, feature_id="sweep")
        .compile()
    )
    height_index = compiled.schema.index(phx.geometry.ParameterId("sweep", "height"))
    points = jnp.asarray([[1.5, 0.0, 0.0], [0.0, 0.0, 3.5]])

    assert jnp.allclose(compiled.signed_distance(points), jnp.asarray([0.5, 0.5]))

    def volume(height):
        state = compiled.state.replace_at(height_index, height)
        return compiled.kernel.measure(state)

    assert jax.grad(volume)(jnp.asarray(6.0)) == pytest.approx(8.0)
    normals = eqx.filter_jit(compiled.boundary_normal)(
        jnp.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    )
    assert jnp.all(jnp.isfinite(normals))
    assert jnp.allclose(jnp.linalg.norm(normals, axis=-1), 1.0)


def test_offset_circle_revolution_is_an_exact_torus_field():
    compiled = (
        phx.geometry.Circle(
            center=(2.0, 0.0),
            radius=0.5,
            feature_id="profile",
        )
        .revolved(feature_id="revolution")
        .compile()
    )
    points = jnp.asarray(
        [
            [2.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.5],
        ]
    )

    assert compiled.field_certificate.is_signed_distance
    assert bool(compiled.validity().accepted)
    assert jnp.allclose(
        compiled.signed_distance(points),
        jnp.asarray([0.0, -0.5, 1.5, 0.0]),
        atol=1.0e-6,
    )
    assert jnp.all(jnp.isfinite(eqx.filter_jit(compiled.boundary_normal)(points)))
    with pytest.raises(NotImplementedError, match="does not provide measure"):
        _ = compiled.measure


def test_revolution_rejects_a_profile_crossing_the_axis():
    compiled = (
        phx.geometry.Circle(
            center=(0.25, 0.0),
            radius=0.5,
            feature_id="profile",
        )
        .revolved()
        .compile()
    )

    evidence = compiled.validity()

    assert not bool(evidence.accepted)
    assert "minimum_profile_radius" in evidence.margin_names
