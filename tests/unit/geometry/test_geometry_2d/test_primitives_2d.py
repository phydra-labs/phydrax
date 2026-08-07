#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

import phydrax as phx


def test_circle():
    radius = 1.0
    circle = phx.domain.GeometryDomain(
        phx.geometry.Circle(center=(0.0, 0.0), radius=radius).compile()
    )
    expected_area = np.pi * radius**2
    computed_area = float(circle.area)
    assert np.isclose(computed_area, expected_area, rtol=0.05)


def test_ellipse():
    x_radius = 2.0
    y_radius = 1.0
    ellipse = phx.domain.GeometryDomain(
        phx.geometry.Ellipse((0.0, 0.0), (x_radius, y_radius)).compile()
    )
    expected_area = np.pi * x_radius * y_radius
    computed_area = float(ellipse.area)
    assert np.isclose(computed_area, expected_area, rtol=0.05)


def test_rectangle():
    width = 3.0
    height = 2.0
    rectangle = phx.domain.GeometryDomain(
        phx.geometry.Rectangle((0.0, 0.0), (width, height)).compile()
    )
    expected_area = width * height
    computed_area = float(rectangle.area)
    assert np.isclose(computed_area, expected_area, rtol=0.05)


def test_square():
    side = 2.0
    square = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=side).compile()
    )
    expected_area = side**2
    computed_area = float(square.area)
    assert np.isclose(computed_area, expected_area, rtol=0.05)


def test_polygon():
    vertices = [(0.0, 0.0), (2.0, 0.0), (1.0, 1.0)]
    polygon = phx.domain.GeometryDomain(phx.geometry.Polygon(vertices=vertices).compile())
    expected_area = 1.0  # Triangle area: 0.5 * base * height = 0.5 * 2 * 1
    computed_area = float(polygon.area)
    assert np.isclose(computed_area, expected_area, rtol=0.05)


def test_triangle():
    vertices = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    triangle = phx.domain.GeometryDomain(
        phx.geometry.Triangle(vertices=vertices).compile()
    )
    expected_area = 0.5  # Right triangle area: 0.5 * base * height = 0.5 * 1 * 1
    computed_area = float(triangle.area)
    assert np.isclose(computed_area, expected_area, rtol=0.05)


def test_polygon_non_unique_vertices():
    vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    with pytest.raises(ValueError, match="Non-unique vertices"):
        phx.domain.GeometryDomain(phx.geometry.Polygon(vertices=vertices).compile())


def test_polygon_self_intersection():
    vertices = [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]
    with pytest.raises(ValueError, match="Self-intersection"):
        phx.domain.GeometryDomain(phx.geometry.Polygon(vertices=vertices).compile())


def test_triangle_invalid_vertices():
    vertices = [(0.0, 0.0), (1.0, 0.0)]  # Only 2 vertices
    with pytest.raises(ValueError, match="Triangle must have exactly 3 vertices"):
        phx.domain.GeometryDomain(phx.geometry.Triangle(vertices=vertices).compile())


def test_polygon_preserves_absolute_coordinates():
    # A non-centered polygon; ensure analytic bounds match input bounds.
    vertices = [
        (1.0, 0.0),
        (1.0, 1.0),
        (0.8, 1.0),
        (0.8, 0.2),
        (0.6, 2.0),
        (0.6, 1.0),
        (0.4, 1.0),
        (0.4, 0.2),
        (0.2, 0.0),
    ]
    poly = phx.domain.GeometryDomain(phx.geometry.Polygon(vertices=vertices).compile())
    import numpy as np

    bounds = np.asarray(poly.bounds)
    in_min = np.min(np.asarray(vertices), axis=0)
    in_max = np.max(np.asarray(vertices), axis=0)
    out_min, out_max = bounds
    assert np.allclose(out_min, in_min, atol=1e-6)
    assert np.allclose(out_max, in_max, atol=1e-6)


def test_boundary_normals_circle():
    # Circle centered at origin with radius 1
    c = phx.domain.GeometryDomain(
        phx.geometry.Circle(center=(0.0, 0.0), radius=1.0).compile()
    )
    # Points on boundary along axes
    from jax import numpy as jnp

    pts = jnp.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=float,
    )
    normals = c._boundary_normals(pts)
    expected = jnp.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )
    import numpy as np

    assert np.allclose(normals, expected, atol=1e-3)
