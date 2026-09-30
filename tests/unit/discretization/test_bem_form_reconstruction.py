from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.bem import (
    OrientedTriangleSurfaceComplex3D,
    prepare_boundary_form_query,
    prepare_buffa_christiansen_dual_3d,
    RWGSurfaceCurrentSpace3D,
)
from phydrax.discretization.bem._rwg import rwg_gram_entries


_VERTICES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)


def _rwg() -> RWGSurfaceCurrentSpace3D:
    return RWGSurfaceCurrentSpace3D(OrientedTriangleSurfaceComplex3D(_VERTICES, _FACES))


def _analytic_values(
    space: RWGSurfaceCurrentSpace3D, points: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    surface = space.surface
    edges = np.asarray(surface.face_edges)
    scale = (
        np.asarray(surface.face_edge_signs)
        * np.asarray(surface.edge_lengths)[edges]
        / (2.0 * np.asarray(surface.face_areas)[:, None])
    )
    opposite = np.asarray(surface.vertices)[np.asarray(surface.opposite_vertices)]
    basis = scale[:, :, None] * (points[:, None] - opposite)
    return np.sum(basis * coefficients[edges, None], axis=1)


def test_rwg_embedded_flux_gram_matches_closed_form_to_roundoff() -> None:
    rwg = _rwg()
    surface = rwg.surface
    vertices = np.asarray(surface.vertices)
    edges = np.asarray(surface.face_edges)
    opposite = vertices[np.asarray(surface.opposite_vertices)]
    offsets = vertices[np.asarray(surface.triangles)][:, None] - opposite[:, :, None]
    sums = np.sum(offsets, axis=2)
    moments = sums @ np.swapaxes(sums, -1, -2) + offsets.reshape(
        (-1, 3, 9)
    ) @ np.swapaxes(offsets.reshape((-1, 3, 9)), -1, -2)
    areas = np.asarray(surface.face_areas)
    scale = (
        np.asarray(surface.face_edge_signs)
        * np.asarray(surface.edge_lengths)[edges]
        / (2.0 * areas[:, None])
    )
    expected = (
        areas[:, None, None] / 12.0 * moments * scale[:, :, None] * scale[:, None, :]
    )
    _, _, actual = rwg_gram_entries(surface)
    np.testing.assert_allclose(
        np.asarray(actual).reshape(expected.shape), expected, rtol=1.0e-14, atol=1.0e-14
    )
    assert rwg.form_type.dimension == 2
    assert rwg.form_type.ambient_dimension == 3
    assert rwg.form_type.degree == 1
    assert rwg.form_type.twist == "twisted"
    assert rwg.trace_capability().form_type == rwg.form_type


def test_rwg_gram_geometry_gradient_uses_full_solution_map() -> None:
    rwg = _rwg()
    surface = rwg.surface
    rhs = jnp.linspace(-0.5, 1.0, rwg.size).astype(jnp.complex128)

    def energy(scale: jax.Array) -> jax.Array:
        moved = eqx.tree_at(
            lambda item: (
                item.vertices,
                item.edge_lengths,
                item.face_areas,
                item.face_centroids,
            ),
            surface,
            (
                surface.vertices * scale,
                surface.edge_lengths * scale,
                surface.face_areas * scale**2,
                surface.face_centroids * scale,
            ),
        )
        space = eqx.tree_at(lambda item: item.surface, rwg, moved)
        capability = space.trace_capability(numeric_revision="uniform-scaling")
        potential = capability.gram_space.inverse_riesz(rhs)
        return jnp.real(jnp.vdot(rhs, potential))

    baseline = energy(jnp.asarray(1.0, dtype=jnp.float64))
    derivative = jax.grad(energy)(jnp.asarray(1.0, dtype=jnp.float64))
    np.testing.assert_allclose(derivative, -2.0 * baseline, rtol=1.0e-10, atol=1.0e-11)


def test_prepared_rwg_surface_query_reuses_exact_trace_and_transpose() -> None:
    rwg = _rwg()
    points = np.asarray(rwg.surface.face_centroids)
    query = prepare_boundary_form_query(rwg, points)
    coefficients = np.array([0.2, -0.4, 0.7, 1.2, -0.8, 0.5], dtype=np.complex128) * (
        1.0 + 0.3j
    )
    expected = _analytic_values(rwg, points, coefficients)
    np.testing.assert_allclose(query.apply(coefficients), expected, atol=1.0e-13)
    changed = coefficients * 1.7 - 0.2j
    np.testing.assert_allclose(
        eqx.filter_jit(query.apply)(changed),
        _analytic_values(rwg, points, changed),
        atol=1.0e-13,
    )
    covector = jnp.asarray(np.arange(12).reshape((4, 3)) / 7.0 + 0.1j)
    primal = jnp.sum(query.apply(coefficients) * covector)
    dual = jnp.sum(jnp.asarray(coefficients) * query.transpose(covector))
    np.testing.assert_allclose(primal, dual, atol=1.0e-13)
    assert query.value_port.form == rwg.value_spec


def test_surface_directional_derivative_is_tangential_affine_derivative() -> None:
    rwg = _rwg()
    query = prepare_boundary_form_query(
        rwg, rwg.surface.face_centroids, derivative=(1, 0, 0)
    )
    coefficients = np.linspace(-1.0, 2.0, rwg.size).astype(np.complex128)
    edges = np.asarray(rwg.surface.face_edges)
    scale = (
        np.asarray(rwg.surface.face_edge_signs)
        * np.asarray(rwg.surface.edge_lengths)[edges]
        / (2.0 * np.asarray(rwg.surface.face_areas)[:, None])
    )
    amplitude = np.sum(scale * coefficients[edges], axis=1)
    normals = np.asarray(rwg.surface.face_normals)
    direction = np.array([1.0, 0.0, 0.0])[None] - normals * normals[:, :1]
    np.testing.assert_allclose(
        query.apply(coefficients), amplitude[:, None] * direction, atol=1.0e-13
    )


def test_surface_query_refuses_volume_points_and_requires_edge_side() -> None:
    rwg = _rwg()
    with pytest.raises(ValueError, match="invalid"):
        prepare_boundary_form_query(rwg, np.array([[0.1, 0.1, 0.1]]))
    midpoint = np.mean(_VERTICES[[0, 1]], axis=0)[None]
    with pytest.raises(ValueError, match="invalid"):
        prepare_boundary_form_query(rwg, midpoint)
    owner = prepare_boundary_form_query(
        rwg, midpoint, side="owner", cell_ids=np.array([0])
    )
    coefficients = np.ones((rwg.size,), dtype=np.complex128)
    expected = _analytic_values(rwg, np.broadcast_to(midpoint, (4, 3)), coefficients)[0]
    np.testing.assert_allclose(owner.apply(coefficients)[0], expected, atol=1.0e-13)
    with pytest.raises(ValueError, match="contain"):
        prepare_boundary_form_query(rwg, midpoint, side="owner", cell_ids=np.array([2]))


def test_bc_boundary_query_evaluates_genuine_refined_surface_field() -> None:
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    faces = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=np.int32,
    )
    dual = prepare_buffa_christiansen_dual_3d(
        RWGSurfaceCurrentSpace3D(OrientedTriangleSurfaceComplex3D(vertices, faces))
    )
    coefficients = np.linspace(-0.7, 1.4, dual.size).astype(np.complex128)
    points = np.asarray(dual.barycentric_surface.face_centroids)
    query = prepare_boundary_form_query(dual, points)
    refined = np.asarray(dual.barycentric_transform) @ coefficients
    expected = _analytic_values(dual.barycentric_rwg, points, refined)
    np.testing.assert_allclose(query.apply(coefficients), expected, atol=1.0e-13)
    covector = jnp.asarray(
        np.cos(np.arange(points.size)).reshape(points.shape), dtype=jnp.complex128
    )
    np.testing.assert_allclose(
        jnp.sum(query.apply(coefficients) * covector),
        jnp.sum(jnp.asarray(coefficients) * query.transpose(covector)),
        atol=1.0e-12,
    )
    form = query.value_port.form
    if form is None:
        raise AssertionError("Boundary query must declare its surface form.")
    assert dual.form_type == form.form_type
