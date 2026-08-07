#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np

import phydrax as phx


def test_sphere():
    radius = 1.0
    sphere = phx.domain.GeometryDomain(
        phx.geometry.Sphere(center=(0.0, 0.0, 0.0), radius=radius).compile()
    )
    expected_volume = (4 / 3) * np.pi * radius**3
    computed_volume = float(sphere.volume)
    assert np.isclose(computed_volume, expected_volume, rtol=0.05)


def test_ellipsoid():
    radii = (1.0, 2.0, 3.0)
    ellipsoid = phx.domain.GeometryDomain(
        phx.geometry.Ellipsoid(center=(0.0, 0.0, 0.0), radii=radii).compile()
    )
    expected_volume = (4 / 3) * np.pi * np.prod(radii)
    computed_volume = float(ellipsoid.volume)
    assert np.isclose(computed_volume, expected_volume, rtol=0.05)


def test_cuboid():
    dimensions = (1.0, 2.0, 3.0)
    cuboid = phx.domain.GeometryDomain(
        phx.geometry.Box((0.0, 0.0, 0.0), dimensions).compile()
    )
    expected_volume = np.prod(dimensions)
    computed_volume = float(cuboid.volume)
    assert np.isclose(computed_volume, expected_volume, rtol=0.05)


def test_cube():
    side = 2.0
    cube = phx.domain.GeometryDomain(
        phx.geometry.Cube(center=(0.0, 0.0, 0.0), side=side).compile()
    )
    expected_volume = side**3
    computed_volume = float(cube.volume)
    assert np.isclose(computed_volume, expected_volume, rtol=0.05)


def test_cylinder():
    radius = 1.0
    height = 2.0
    cylinder = phx.domain.GeometryDomain(
        phx.geometry.Cylinder((0.0, 0.0, 0.0), (0.0, 0.0, height), radius).compile()
    )
    expected_volume = np.pi * radius**2 * height
    computed_volume = float(cylinder.volume)
    assert np.isclose(computed_volume, expected_volume, rtol=0.05)


def test_cone():
    radius0 = 1.0
    height = 3.0
    cone = phx.domain.GeometryDomain(
        phx.geometry.Cone(
            base_center=(0.0, 0.0, 0.0),
            axis=(0.0, 0.0, height),
            radius0=radius0,
        ).compile()
    )
    expected_volume = (1 / 3) * np.pi * radius0**2 * height
    computed_volume = float(cone.volume)
    assert np.isclose(computed_volume, expected_volume, rtol=0.05)


def test_truncated_cone():
    radius0 = 2.0
    radius1 = 1.0
    height = 3.0
    cone = phx.domain.GeometryDomain(
        phx.geometry.Cone(
            base_center=(0.0, 0.0, 0.0),
            axis=(0.0, 0.0, height),
            radius0=radius0,
            radius1=radius1,
        ).compile()
    )
    expected_volume = (
        (1 / 3) * np.pi * height * (radius0**2 + radius0 * radius1 + radius1**2)
    )
    computed_volume = float(cone.volume)
    assert np.isclose(computed_volume, expected_volume, rtol=0.05)


def test_torus():
    inner_radius = 1.0
    outer_radius = 2.0
    torus = phx.domain.GeometryDomain(
        phx.geometry.Torus(
            center=(0.0, 0.0, 0.0),
            inner_radius=inner_radius,
            outer_radius=outer_radius,
        ).compile()
    )

    # Calculate the major and minor radii correctly
    major_radius = (
        inner_radius + outer_radius
    ) / 2  # Distance from center to center of tube
    minor_radius = (outer_radius - inner_radius) / 2  # Radius of the tube

    # Calculate expected volume using the correct formula
    expected_volume = 2 * np.pi**2 * major_radius * minor_radius**2

    # Get computed volume
    computed_volume = float(torus.volume)

    # Use a slightly larger tolerance due to mesh approximation
    assert np.isclose(computed_volume, expected_volume, rtol=0.1)


def test_wedge():
    x0 = (0.0, 0.0, 0.0)
    extends = (2.0, 2.0, 2.0)
    top_extent = 1.0
    wedge = phx.domain.GeometryDomain(
        phx.geometry.Wedge(x0, extends, top_extent).compile()
    )
    expected_volume = 0.5 * extends[1] * extends[2] * (extends[0] + top_extent)
    computed_volume = float(wedge.volume)
    assert np.isclose(computed_volume, expected_volume, rtol=0.05)


def test_boundary_normals_sphere():
    s = phx.domain.GeometryDomain(
        phx.geometry.Sphere(center=(0.0, 0.0, 0.0), radius=1.0).compile()
    )
    from jax import numpy as jnp

    pts = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    normals = s._boundary_normals(pts)
    expected = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    import numpy as np

    assert np.allclose(normals, expected, atol=1e-3)
