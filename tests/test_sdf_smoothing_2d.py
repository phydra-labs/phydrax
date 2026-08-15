#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax
import jax.numpy as jnp
import meshio
import numpy as np
import pytest

import phydrax as phx
from phydrax.domain import Boundary
from phydrax.enforcement import enforce_dirichlet


def test_geometry2d_sdf_signs_and_boundary():
    # Simple in-memory mesh: unit square split into two triangles
    pts = np.array(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    m = meshio.Mesh(points=pts, cells=[("triangle", faces)])

    geom = phx.domain.GeometryDomain(
        phx.geometry.planar_region_from_source(m, recenter=False).compile()
    )

    inside = jnp.array([[0.0, 0.0]], dtype=float)
    outside = jnp.array([[2.0, 0.0]], dtype=float)
    boundary = jnp.array([[0.5, 0.0]], dtype=float)

    sdf = jax.vmap(geom.adf)
    sd_inside = sdf(inside)
    sd_outside = sdf(outside)
    sd_boundary = sdf(boundary)

    assert sd_inside.shape == (1,)
    assert sd_outside.shape == (1,)
    assert sd_boundary.shape == (1,)

    assert float(sd_inside[0]) < 0.0
    assert float(sd_outside[0]) > 0.0
    assert np.isclose(float(sd_boundary[0]), 0.0, atol=1e-2)


def test_dense_curved_mesh_adf_has_only_the_geometric_zero_set():
    geometry = phx.domain.GeometryDomain(
        phx.geometry.Ellipse((0.0, 0.0), (1.25, 0.7)).compile()
    )
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 96, endpoint=False)
    normalized_radii = jnp.asarray([0.75, 1.1, 1.3, 1.5])
    points = jnp.stack(
        [
            1.25 * normalized_radii[:, None] * jnp.cos(theta),
            0.7 * normalized_radii[:, None] * jnp.sin(theta),
        ],
        axis=-1,
    ).reshape((-1, 2))

    expected_inside = jnp.repeat(normalized_radii < 1.0, theta.size)
    predicate_inside = geometry.geometry.contains(points)
    factor_inside = geometry.adf(points) < 0.0

    assert jnp.array_equal(predicate_inside, expected_inside)
    assert jnp.array_equal(factor_inside, expected_inside)


def _scaled_square(scale: float) -> phx.domain.GeometryDomain:
    half = 0.5 * scale
    points = np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    return phx.domain.GeometryDomain(
        phx.geometry.planar_region_from_source(
            meshio.Mesh(points=points, cells=[("triangle", faces)]),
            recenter=False,
        ).compile()
    )


def _derivative(function, order: int):
    derivative = function
    for _ in range(order):
        derivative = jax.grad(derivative)
    return derivative


def test_boundary_factor_is_scale_covariant_with_exact_boundary_collar():
    scales = (1e-7, 1.0)
    geometries = tuple(_scaled_square(scale) for scale in scales)
    normalized_points = jnp.array(
        [[0.5, 0.0], [0.49, 0.1], [0.4, -0.15], [0.0, 0.0], [0.75, 0.1]]
    )
    normalized_values = []
    normalized_ansatz_values = []

    for scale, geometry in zip(scales, geometries, strict=True):
        point = jnp.array([0.5 * scale, 0.0])
        normal = jnp.array([1.0, 0.0])
        assert abs(float(geometry.adf(point))) <= 1e-12 * scale
        assert jnp.allclose(
            jax.grad(geometry.adf)(point),
            normal,
            atol=1e-12,
            rtol=0.0,
        )
        ansatz_factor = geometry.boundary_ansatz_factor
        assert abs(float(ansatz_factor(point))) <= 1e-12 * scale
        assert jnp.allclose(
            jax.grad(ansatz_factor)(point),
            normal,
            atol=1e-10,
            rtol=0.0,
        )
        normalized_ansatz_values.append(ansatz_factor(scale * normalized_points) / scale)

        offsets = jnp.array([-0.02, -0.01, -0.002, 0.002, 0.01, 0.02]) * scale
        profile = geometry.adf(point + offsets[:, None] * normal)
        assert jnp.allclose(profile, offsets, atol=1e-12 * scale, rtol=1e-12)
        normalized_values.append(geometry.adf(scale * normalized_points) / scale)

    assert jnp.allclose(
        normalized_values[0],
        normalized_values[1],
        atol=1e-10,
        rtol=1e-10,
    )
    assert jnp.allclose(
        normalized_ansatz_values[0],
        normalized_ansatz_values[1],
        atol=1e-10,
        rtol=1e-10,
    )


def test_enforcement_gate_is_dimensionless_scale_invariant_and_broad():
    scales = (1e-7, 1.0)
    normalized_points = jnp.array([[0.5, 0.0], [0.4, 0.0], [0.25, 0.0], [0.0, 0.0]])
    normalized_values = []
    normalized_boundary_gradients = []

    for scale in scales:
        geometry = _scaled_square(scale)
        gate = geometry.make_enforcement_gate()
        normalized_values.append(gate(scale * normalized_points))
        normalized_boundary_gradients.append(
            scale * jax.grad(gate)(jnp.array([0.5 * scale, 0.0]))
        )

    assert jnp.allclose(
        normalized_values[0],
        normalized_values[1],
        atol=1e-10,
        rtol=1e-10,
    )
    assert jnp.allclose(
        normalized_boundary_gradients[0],
        normalized_boundary_gradients[1],
        atol=1e-10,
        rtol=1e-10,
    )
    values = normalized_values[0]
    assert jnp.allclose(values[0], 0.0, atol=1e-10)
    assert 0.2 < float(values[1]) < 0.4
    assert float(values[2]) > 0.5
    assert float(values[3]) > 0.9
    assert jnp.all(jnp.diff(values) > 0.0)
    assert jnp.linalg.norm(normalized_boundary_gradients[0]) < 5.0


def test_compact_enforcement_gate_saturation_controls_transition_extent():
    geometry = _scaled_square(1.0)
    point = jnp.array([0.4, 0.0])

    broad = geometry.make_enforcement_gate(method="compact", saturation_fraction=0.5)
    compact = geometry.make_enforcement_gate(method="compact", saturation_fraction=0.2)

    assert float(broad(point)) < 0.4
    assert float(compact(point)) > 0.6


def test_enforcement_gate_rejects_unknown_method():
    geometry = _scaled_square(1.0)
    invalid_method: Any = "unknown"

    with pytest.raises(ValueError, match="method must be"):
        geometry.make_enforcement_gate(method=invalid_method)


def test_boundary_field_has_finite_flat_high_order_face_jets():
    geometry = _scaled_square(1.0)
    boundary_point = jnp.array([0.5, 0.0])
    normal = jnp.array([1.0, 0.0])

    def boundary_profile(offset):
        return geometry.adf(boundary_point + offset * normal)

    for order in (2, 3, 4):
        value = _derivative(boundary_profile, order)(jnp.asarray(0.0))
        assert jnp.isfinite(value)
        assert jnp.allclose(value, 0.0, atol=1e-10)


def test_cad_hard_ansatz_uses_certified_boundary_field():
    geometry = _scaled_square(1.0)
    assert geometry.geometry.field_certificate.is_signed_distance
    assert geometry.geometry.contains(jnp.array([0.0, 0.0]))

    component = geometry.component({"x": Boundary()})

    @geometry.Function("x")
    def field(x):
        del x
        return jnp.asarray(1.0)

    enforced = enforce_dirichlet(field, component, var="x", target=0.0)
    boundary_value = enforced.func(jnp.array([0.5, 0.0]))
    assert jnp.allclose(boundary_value, 0.0, atol=1e-12)
