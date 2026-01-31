#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import numpy as np

from phydrax.domain.geometry3d import Geometry3DFromLidarScene


def _fibonacci_sphere(n: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = np.pi * (1 + np.sqrt(5.0)) * k
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.c_[x, y, z]


def test_geometry3d_from_lidar_scene_basic():
    # Two clusters; crop ROI to keep the first
    obj1 = _fibonacci_sphere(600)
    obj2 = _fibonacci_sphere(400) * 0.5 + np.array([2.0, 0.0, 0.0])
    pts = np.vstack([obj1, obj2])

    geom = Geometry3DFromLidarScene(
        pts,
        roi=(-1.2, 1.2, -1.2, 1.2, -1.2, 1.2),
        voxel_size=0.05,
        close_depth=0.2,
    )

    # Mesh exists and has triangles
    assert geom.mesh is not None
    assert geom.mesh_vertices.shape[1] == 3
    assert geom.mesh_faces.shape[1] == 3

    # Bounds are finite
    bounds = np.asarray(geom.bounds, dtype=float)
    assert bounds.shape == (2, 3)
    assert float(bounds[0, 0]) < float(bounds[1, 0])

    # Area and volume are finite
    assert float(geom.surface_area_value) > 0.0
    assert float(geom.volume) > 0.0

    # Interior sampling returns negative SDF
    interior = geom.sample_interior(24)
    assert interior.shape == (24, 3)
    sd = jax.vmap(geom.adf_orig)(interior)
    assert np.all(sd <= 1e-8)

    # Boundary sampling returns ~0 SDF
    boundary = geom.sample_boundary(12)
    assert boundary.shape == (12, 3)
    sd_b = jax.vmap(geom.adf_orig)(boundary)
    assert np.allclose(sd_b, 0.0, atol=1e-6)
