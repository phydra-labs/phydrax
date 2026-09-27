#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_spatial_primitives_preserve_analytic_volume() -> None:
    sphere_radius = 1.0
    ellipsoid_radii = (1.0, 2.0, 3.0)
    box_dimensions = (1.0, 2.0, 3.0)
    cube_side = 2.0
    cylinder_radius, cylinder_height = 1.0, 2.0
    cone_radius, cone_height = 1.0, 3.0
    frustum_radius0, frustum_radius1, frustum_height = 2.0, 1.0, 3.0
    inner_radius, outer_radius = 1.0, 2.0
    major_radius = (inner_radius + outer_radius) / 2
    minor_radius = (outer_radius - inner_radius) / 2
    wedge_extents = (2.0, 2.0, 2.0)
    wedge_top = 1.0

    cases = (
        (
            "sphere",
            phx.domain.GeometryDomain(
                phx.geometry.Sphere(
                    center=(0.0, 0.0, 0.0),
                    radius=sphere_radius,
                ).compile()
            ),
            (4 / 3) * np.pi * sphere_radius**3,
            0.05,
        ),
        (
            "ellipsoid",
            phx.domain.GeometryDomain(
                phx.geometry.Ellipsoid(
                    center=(0.0, 0.0, 0.0),
                    radii=ellipsoid_radii,
                ).compile()
            ),
            (4 / 3) * np.pi * np.prod(ellipsoid_radii),
            0.05,
        ),
        (
            "box",
            phx.domain.GeometryDomain(
                phx.geometry.Box((0.0, 0.0, 0.0), box_dimensions).compile()
            ),
            np.prod(box_dimensions),
            0.05,
        ),
        (
            "cube",
            phx.domain.GeometryDomain(
                phx.geometry.Cube(
                    center=(0.0, 0.0, 0.0),
                    side=cube_side,
                ).compile()
            ),
            cube_side**3,
            0.05,
        ),
        (
            "cylinder",
            phx.domain.GeometryDomain(
                phx.geometry.Cylinder(
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, cylinder_height),
                    cylinder_radius,
                ).compile()
            ),
            np.pi * cylinder_radius**2 * cylinder_height,
            0.05,
        ),
        (
            "cone",
            phx.domain.GeometryDomain(
                phx.geometry.Cone(
                    base_center=(0.0, 0.0, 0.0),
                    axis=(0.0, 0.0, cone_height),
                    radius0=cone_radius,
                ).compile()
            ),
            (1 / 3) * np.pi * cone_radius**2 * cone_height,
            0.05,
        ),
        (
            "frustum",
            phx.domain.GeometryDomain(
                phx.geometry.Cone(
                    base_center=(0.0, 0.0, 0.0),
                    axis=(0.0, 0.0, frustum_height),
                    radius0=frustum_radius0,
                    radius1=frustum_radius1,
                ).compile()
            ),
            (1 / 3)
            * np.pi
            * frustum_height
            * (
                frustum_radius0**2
                + frustum_radius0 * frustum_radius1
                + frustum_radius1**2
            ),
            0.05,
        ),
        (
            "torus",
            phx.domain.GeometryDomain(
                phx.geometry.Torus(
                    center=(0.0, 0.0, 0.0),
                    inner_radius=inner_radius,
                    outer_radius=outer_radius,
                ).compile()
            ),
            2 * np.pi**2 * major_radius * minor_radius**2,
            0.1,
        ),
        (
            "wedge",
            phx.domain.GeometryDomain(
                phx.geometry.Wedge(
                    (0.0, 0.0, 0.0),
                    wedge_extents,
                    wedge_top,
                ).compile()
            ),
            0.5 * wedge_extents[1] * wedge_extents[2] * (wedge_extents[0] + wedge_top),
            0.05,
        ),
    )
    for case_id, domain, expected_volume, rtol in cases:
        assert np.isclose(float(domain.volume), expected_volume, rtol=rtol), case_id


def test_boundary_normals_sphere() -> None:
    s = phx.domain.GeometryDomain(
        phx.geometry.Sphere(center=(0.0, 0.0, 0.0), radius=1.0).compile()
    )

    pts = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype="float64",
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

    assert np.allclose(normals, expected, atol=1e-3)
