#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

from phydrax.geometry.simplicial import AffineSimplexMap


def test_affine_tetrahedron_maps_values_gradients_and_orientation():
    vertices = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 3.0, 0.0),
            (0.0, 0.0, 4.0),
        )
    )
    simplex = AffineSimplexMap(vertices)
    point = jnp.asarray((0.5, 0.75, 1.0))
    barycentric = jax.jit(lambda value: simplex.barycentric(value))(point)
    nodal = vertices @ jnp.asarray((1.0, 2.0, 3.0)) + 5.0

    assert bool(simplex.evidence.successful)
    assert jnp.allclose(barycentric, jnp.full((4,), 0.25))
    assert bool(simplex.contains(point))
    assert jnp.allclose(simplex.reference_to_physical(barycentric), point)
    assert jnp.allclose(simplex.physical_gradient(nodal), jnp.asarray((1.0, 2.0, 3.0)))
    assert jnp.allclose(simplex.evidence.orientation_determinant, 24.0)
    assert jnp.allclose(simplex.evidence.measure, 4.0)


def test_embedded_triangle_and_degenerate_simplex_report_geometry_evidence():
    triangle = AffineSimplexMap(
        jnp.asarray(
            (
                (0.0, 0.0, 1.0),
                (2.0, 0.0, 1.0),
                (0.0, 3.0, 1.0),
            )
        )
    )
    barycentric = triangle.barycentric(jnp.asarray((0.5, 0.75, 1.0)))
    degenerate = AffineSimplexMap(jnp.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))))

    assert bool(triangle.evidence.successful)
    assert jnp.isnan(triangle.evidence.orientation_determinant)
    assert jnp.allclose(triangle.evidence.measure, 3.0)
    assert jnp.allclose(barycentric, jnp.asarray((0.5, 0.25, 0.25)))
    assert bool(triangle.contains(jnp.asarray((0.5, 0.75, 1.0))))
    assert not bool(degenerate.evidence.successful)
