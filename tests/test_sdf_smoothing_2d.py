#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import meshio
import numpy as np

from phydrax.domain.geometry2d._from_cad import Geometry2DFromCAD


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

    geom = Geometry2DFromCAD(m, recenter=False)

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
