#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

import phydrax as phx


def test_primitives_2d_scenario_1() -> None:
    radius = 1.0
    x_radius, y_radius = 2.0, 1.0
    width, height = 3.0, 2.0
    side = 2.0
    cases = (
        (
            "circle",
            phx.geometry.Circle(center=(0.0, 0.0), radius=radius),
            np.pi * radius**2,
        ),
        (
            "ellipse",
            phx.geometry.Ellipse((0.0, 0.0), (x_radius, y_radius)),
            np.pi * x_radius * y_radius,
        ),
        (
            "rectangle",
            phx.geometry.Rectangle((0.0, 0.0), (width, height)),
            width * height,
        ),
        (
            "square",
            phx.geometry.Square(center=(0.0, 0.0), side=side),
            side**2,
        ),
        (
            "polygon",
            phx.geometry.Polygon(vertices=((0.0, 0.0), (2.0, 0.0), (1.0, 1.0))),
            1.0,
        ),
        (
            "triangle",
            phx.geometry.Triangle(vertices=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            0.5,
        ),
    )
    for case_id, primitive, expected_area in cases:
        domain = phx.domain.GeometryDomain(primitive.compile())
        assert np.isclose(float(domain.area), expected_area, rtol=0.05), case_id
    with pytest.raises(ValueError, match="Non-unique vertices"):
        phx.domain.GeometryDomain(
            phx.geometry.Polygon(
                vertices=((0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 1.0))
            ).compile()
        )
    with pytest.raises(ValueError, match="Self-intersection"):
        phx.domain.GeometryDomain(
            phx.geometry.Polygon(
                vertices=((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0))
            ).compile()
        )
    with pytest.raises(ValueError, match="Triangle must have exactly 3 vertices"):
        phx.domain.GeometryDomain(
            phx.geometry.Triangle(vertices=((0.0, 0.0), (1.0, 0.0))).compile()
        )
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

    bounds = np.asarray(poly.bounds)
    in_min = np.min(np.asarray(vertices), axis=0)
    in_max = np.max(np.asarray(vertices), axis=0)
    out_min, out_max = bounds
    assert np.allclose(out_min, in_min, atol=1e-6)
    assert np.allclose(out_max, in_max, atol=1e-6)
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
        dtype="float64",
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

    assert np.allclose(normals, expected, atol=1e-3)
