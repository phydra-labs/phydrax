#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def test_planar_crack_surface_exposes_oriented_front_and_exact_discrete_ledgers():
    geometry = phx.applications.fracture.CrackSurfaceGeometry3D(
        jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        ),
        jnp.asarray(((0, 1, 2), (0, 2, 3))),
    )
    quadrature = phx.applications.fracture.SharpCrackQuadrature3D(geometry)
    mesh = geometry.as_triangle_mesh()

    assert jnp.allclose(geometry.surface_area, 1.0)
    assert jnp.allclose(geometry.front_length, 4.0)
    assert jnp.allclose(quadrature.represented_surface_area, 1.0)
    assert jnp.allclose(quadrature.represented_front_length, 4.0)
    assert jnp.allclose(jnp.linalg.norm(geometry.front_tangents, axis=-1), 1.0)
    assert mesh.vertices.shape[1] == 3


def test_crack_surface_rejects_same_direction_shared_edge():
    vertices = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        )
    )
    with pytest.raises(ValueError, match="shared edges oppositely"):
        phx.applications.fracture.CrackSurfaceGeometry3D(
            vertices,
            jnp.asarray(((0, 1, 2), (0, 1, 3))),
        )
