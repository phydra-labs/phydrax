#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
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


def test_indexed_affine_simplex_operations_select_without_cross_product():
    vertices = jnp.asarray(
        (
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            ((2.0, 0.0), (3.0, 0.0), (2.0, 1.0)),
        )
    )
    simplices = AffineSimplexMap(vertices)
    points = jnp.asarray(((0.25, 0.25), (2.5, 0.25)))
    indices = jnp.asarray((0, 1), dtype=jnp.int32)
    barycentric = eqx.filter_jit(simplices.barycentric_at)(points, indices)
    nodal = jnp.asarray(((0.0, 1.0, 2.0), (4.0, 5.0, 6.0)))

    assert barycentric.shape == (2, 3)
    assert jnp.allclose(barycentric, jnp.asarray(((0.5, 0.25, 0.25), (0.25, 0.5, 0.25))))
    assert jnp.all(simplices.contains_at(points, indices))
    assert jnp.allclose(
        simplices.physical_gradient_at(nodal, indices),
        jnp.asarray(((1.0, 2.0), (1.0, 2.0))),
    )
