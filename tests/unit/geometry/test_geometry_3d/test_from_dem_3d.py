#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import numpy as np

from phydrax.domain.geometry3d import Geometry3DFromDEM


def test_geometry3d_from_dem_basic():
    # DEM grid: z = sin(x) * cos(y)
    ny, nx = 30, 36
    x = np.linspace(-2.0, 2.0, nx)
    y = np.linspace(-1.5, 1.5, ny)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)

    geom = Geometry3DFromDEM(Z, x=x, y=y, extrude_depth=0.25, alpha=None)

    # Mesh exists and has triangles
    assert geom.mesh is not None
    assert geom.mesh_vertices.shape[1] == 3
    assert geom.mesh_faces.shape[1] == 3

    # Bounds are finite and consistent
    bounds = np.asarray(geom.bounds, dtype=float)
    assert bounds.shape == (2, 3)
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
    assert xmin < xmax and ymin < ymax and zmin < zmax

    # Area and volume are finite
    assert float(geom.surface_area_value) > 0.0
    assert float(geom.volume) > 0.0

    # Sampling interior returns points with negative SDF
    interior = geom.sample_interior(32)
    assert interior.shape == (32, 3)
    sd = jax.vmap(geom.adf_orig)(interior)
    assert np.all(sd <= 1e-8)

    # Boundary sampling returns points with ~0 SDF
    boundary = geom.sample_boundary(24)
    assert boundary.shape == (24, 3)
    sd_b = jax.vmap(geom.adf_orig)(boundary)
    assert np.allclose(sd_b, 0.0, atol=1e-6)
