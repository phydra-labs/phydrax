#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry.analytic import Tube
from phydrax.geometry.brep._intersection_curve import CurveRange
from phydrax.geometry.brep._patches import CircleCurve, EllipseCurve, LineCurve
from phydrax.units import METER


# Empirical `test_tube_closed_forms_enclosures_and_gradients`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_TUBE_SIGNED_DISTANCE_RTOL = 1e-14
# Empirical `test_tube_closed_forms_enclosures_and_gradients`: matched max |delta|=1.97699e-11; joint allowance fraction=0.00872065, margin=99.1279%.
_TUBE_GRADIENT_FINITE_DIFFERENCE_RTOL = 1e-8
# Empirical `test_tube_closed_forms_enclosures_and_gradients`: matched max |delta|=1.97699e-11; joint allowance fraction=0.00872065, margin=99.1279%.
_TUBE_GRADIENT_FINITE_DIFFERENCE_ATOL = 1e-10


def test_analytic_sweeps_scenario_1() -> None:
    compiled = (
        phx.geometry.Circle(
            (0.0, 0.0),
            2.0,
            feature_id="profile",
        )
        .extruded(6.0, feature_id="sweep")
        .compile(coordinate_unit=METER)
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
    assert compiled.has_capability(phx.geometry.GeometryCapability.INTERIOR_MEASURE)
    assert compiled.has_capability(phx.geometry.GeometryCapability.BOUNDARY_MEASURE)
    assert jnp.allclose(
        compiled.signed_distance(points),
        jnp.asarray([0.0, 0.0, -2.0, jnp.sqrt(2.0)]),
        atol=1.0e-6,
    )
    assert compiled.measure == pytest.approx(24.0 * jnp.pi)
    assert compiled.boundary_measure == pytest.approx(32.0 * jnp.pi)
    assert bool(compiled.validity().accepted)
    compiled = (
        phx.geometry.Circle(
            center=(2.0, 0.0),
            radius=0.5,
            feature_id="profile",
        )
        .revolved(feature_id="revolution")
        .compile(coordinate_unit=METER)
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
    with pytest.raises(NotImplementedError, match="does not provide interior_measure"):
        _ = compiled.measure
    compiled = (
        phx.geometry.Circle(
            center=(0.25, 0.0),
            radius=0.5,
            feature_id="profile",
        )
        .revolved()
        .compile(coordinate_unit=METER)
    )

    evidence = compiled.validity()

    assert not bool(evidence.accepted)
    assert "minimum_profile_radius" in evidence.margin_names


def test_rectangle_extrusion_matches_box_distance_and_has_finite_gradients() -> None:
    compiled = (
        phx.geometry.Rectangle(
            center=(0.0, 0.0),
            size=(2.0, 4.0),
            feature_id="profile",
        )
        .extruded(6.0, feature_id="sweep")
        .compile(coordinate_unit=METER)
    )
    height_index = compiled.schema.index(phx.geometry.ParameterId("sweep", "height"))
    points = jnp.asarray([[1.5, 0.0, 0.0], [0.0, 0.0, 3.5]])

    assert jnp.allclose(compiled.signed_distance(points), jnp.asarray([0.5, 0.5]))

    def volume(height: Any) -> Any:
        state = compiled.state.replace_at(height_index, height)
        return compiled.kernel.measure(state)

    assert jax.grad(volume)(jnp.asarray(6.0)) == pytest.approx(8.0)
    normals = eqx.filter_jit(compiled.boundary_normal)(
        jnp.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    )
    assert jnp.all(jnp.isfinite(normals))
    assert jnp.allclose(jnp.linalg.norm(normals, axis=-1), 1.0)


def test_tube_closed_forms_enclosures_and_gradients() -> None:
    paths = (
        CurveRange(LineCurve((0, 0, 0), (0, 0, 1)), -2.0, 2.0),
        CircleCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 2.0),
    )
    points = jnp.asarray(((3.0, 0.0, 0.0), (1.0, 0.0, 3.0)), dtype=jnp.float64)
    oracles = (np.asarray((3.0, np.sqrt(2.0))), np.asarray((1.0, np.sqrt(10.0))))
    for path, oracle in zip(paths, oracles, strict=True):
        source = Tube(path, 0.25)
        compiled = source.compile(coordinate_unit=METER)
        np.testing.assert_allclose(
            compiled.signed_distance(points),
            oracle - 0.25,
            rtol=_TUBE_SIGNED_DISTANCE_RTOL,
        )
        enclosure = source.distance_bounds(points)
        assert np.all(enclosure[:, 0] <= oracle)
        assert np.all(oracle <= enclosure[:, 1])

        def field(point: jax.Array) -> jax.Array:
            return compiled.signed_distance(point[None, :])[0]

        point = jnp.asarray((3.0, 0.5, 1.0), dtype=jnp.float64)
        gradient = jax.grad(field)(point)
        step = 1e-5
        differences = jnp.stack(
            tuple(
                (field(point + step * axis) - field(point - step * axis)) / (2 * step)
                for axis in jnp.eye(3, dtype=jnp.float64)
            )
        )
        np.testing.assert_allclose(
            gradient,
            differences,
            rtol=_TUBE_GRADIENT_FINITE_DIFFERENCE_RTOL,
            atol=_TUBE_GRADIENT_FINITE_DIFFERENCE_ATOL,
        )


def test_tube_reach_boundary_and_unsupported_paths() -> None:
    circle = CircleCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 2.0)
    assert Tube(circle, np.nextafter(2.0, 0.0)).reach == 2.0
    for radius in (2.0, np.nextafter(2.0, np.inf), 0.0, np.inf):
        with pytest.raises(ValueError, match="strictly below"):
            Tube(circle, radius)
    ellipse = EllipseCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 2.0, 1.0)
    with pytest.raises(ValueError, match="unsupported"):
        Tube(CurveRange(ellipse), 0.25)
