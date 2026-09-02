import numpy as np
import pytest

from phydrax.discretization.bem import (
    OrientedTriangleSurfaceComplex3D,
    prepare_buffa_christiansen_dual_3d,
    RWGSurfaceCurrentSpace3D,
)


_TETRAHEDRON = (
    np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    ),
    np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))),
)

_ANISOTROPIC_OCTAHEDRON = (
    np.asarray(
        (
            (1.8, -0.2, 0.3),
            (-1.6, -0.2, 0.3),
            (0.1, 0.6, 0.3),
            (0.1, -1.0, 0.3),
            (0.1, -0.2, 1.5),
            (0.1, -0.2, -0.9),
        )
    ),
    np.asarray(
        (
            (0, 2, 4),
            (2, 1, 4),
            (1, 3, 4),
            (3, 0, 4),
            (2, 0, 5),
            (1, 2, 5),
            (3, 1, 5),
            (0, 3, 5),
        )
    ),
)


def _bc_values_at_refined_centroids(dual):
    refined = dual.barycentric_surface
    local_basis = np.asarray(dual.barycentric_rwg.centroid_basis)
    local_transform = np.asarray(dual.barycentric_transform)[
        np.asarray(refined.face_edges)
    ]
    return np.einsum("flv,fle->fev", local_basis, local_transform)


def _oriented_refined_edge_flux_dofs(dual):
    refined = dual.barycentric_surface
    points = np.asarray(refined.vertices)
    triangles = np.asarray(refined.triangles)
    face_edges = np.asarray(refined.face_edges)
    face_signs = np.asarray(refined.face_edge_signs)
    opposite = np.asarray(refined.opposite_vertices)
    areas = np.asarray(refined.face_areas)
    normals = np.asarray(refined.face_normals)
    lengths = np.asarray(refined.edge_lengths)
    transform = np.asarray(dual.barycentric_transform)
    flux_dofs = np.empty((refined.edge_count, dual.size))

    for refined_edge_id in range(refined.edge_count):
        face_id, local_id = np.argwhere(face_edges == refined_edge_id)[0]
        start = points[triangles[face_id, local_id]]
        stop = points[triangles[face_id, (local_id + 1) % 3]]
        midpoint = 0.5 * (start + stop)
        tangent = (stop - start) / lengths[refined_edge_id]
        outward_conormal = np.cross(tangent, normals[face_id])
        local_scale = (
            face_signs[face_id] * lengths[face_edges[face_id]] / (2.0 * areas[face_id])
        )
        local_basis = local_scale[:, None] * (
            midpoint[None, :] - points[opposite[face_id]]
        )
        bc_values = np.einsum("lv,le->ev", local_basis, transform[face_edges[face_id]])
        flux_dofs[refined_edge_id] = (
            face_signs[face_id, local_id]
            * lengths[refined_edge_id]
            * (bc_values @ outward_conormal)
        )
    return flux_dofs


@pytest.mark.parametrize("vertices, faces", (_TETRAHEDRON, _ANISOTROPIC_OCTAHEDRON))
def test_bc_basis_on_barycentric_refinement_has_stable_rwg_duality(vertices, faces):
    primal = RWGSurfaceCurrentSpace3D(OrientedTriangleSurfaceComplex3D(vertices, faces))
    dual = prepare_buffa_christiansen_dual_3d(primal)

    refined = dual.barycentric_surface
    assert refined.vertex_count == (
        primal.surface.vertex_count
        + primal.surface.edge_count
        + primal.surface.face_count
    )
    assert refined.face_count == 6 * primal.surface.face_count
    assert dual.barycentric_rwg.surface is refined
    assert dual.barycentric_transform.shape == (refined.edge_count, primal.size)
    assert refined.edge_count > primal.size
    assert np.linalg.matrix_rank(np.asarray(dual.barycentric_transform)) == primal.size

    cross_mass = np.asarray(dual.cross_mass.matrix)
    singular_values = np.linalg.svd(cross_mass, compute_uv=False)
    assert singular_values[-1] > 0.0
    assert np.linalg.matrix_rank(cross_mass) == primal.size
    assert np.linalg.cond(cross_mass) < 5.0
    assert dual.evidence.cross_mass_condition_number == pytest.approx(
        np.linalg.cond(cross_mass), rel=2e-6
    )
    assert dual.evidence.minimum_cross_mass_singular_value == pytest.approx(
        singular_values[-1], rel=2e-6
    )
    assert not np.allclose(cross_mass, np.eye(primal.size), atol=1e-12)

    coefficients = np.arange(1, primal.size + 1) * (1.0 + 0.25j)
    np.testing.assert_allclose(
        dual.barycentric_rwg_coefficients(coefficients),
        np.asarray(dual.barycentric_transform) @ coefficients,
    )


def test_bc_dual_edge_flux_dofs_follow_coarse_edge_orientation():
    primal = RWGSurfaceCurrentSpace3D(
        OrientedTriangleSurfaceComplex3D(*_ANISOTROPIC_OCTAHEDRON)
    )
    dual = prepare_buffa_christiansen_dual_3d(primal)
    surface = primal.surface
    refined_edge_ids = {
        tuple(int(value) for value in edge): edge_id
        for edge_id, edge in enumerate(np.asarray(dual.barycentric_surface.edge_vertices))
    }
    flux_dofs = _oriented_refined_edge_flux_dofs(dual)
    vertex_count = surface.vertex_count
    edge_count = surface.edge_count
    face_edges = np.asarray(surface.face_edges)
    face_signs = np.asarray(surface.face_edge_signs)

    for coarse_edge_id in range(edge_count):
        midpoint = vertex_count + coarse_edge_id
        for face_id, local_id in np.argwhere(face_edges == coarse_edge_id):
            centroid = vertex_count + edge_count + int(face_id)
            refined_edge_id = refined_edge_ids[
                (min(midpoint, centroid), max(midpoint, centroid))
            ]
            assert flux_dofs[refined_edge_id, coarse_edge_id] == pytest.approx(
                0.5 * face_signs[face_id, local_id]
            )

    nonzero = np.abs(flux_dofs[:, 0][np.abs(flux_dofs[:, 0]) > 1e-10])
    np.testing.assert_allclose(
        np.sort(nonzero),
        np.sort(
            np.asarray(
                (
                    0.5,
                    0.5,
                    3 / 8,
                    3 / 8,
                    3 / 8,
                    3 / 8,
                    1 / 4,
                    1 / 4,
                    1 / 4,
                    1 / 4,
                    1 / 8,
                    1 / 8,
                    1 / 8,
                    1 / 8,
                )
            )
        ),
        atol=1e-7,
    )
    assert dual.evidence.minimum_orientation_alignment > 0.0
    assert dual.evidence.barycentric_area_defect < 1e-12


def test_bc_basis_obeys_surface_piola_map_under_anisotropic_scaling():
    vertices, faces = _ANISOTROPIC_OCTAHEDRON
    affine = np.asarray(((3.2, 0.4, 0.0), (0.0, 0.45, 0.2), (0.1, 0.0, 1.7)))
    translation = np.asarray((0.3, -1.1, 0.7))
    reference = prepare_buffa_christiansen_dual_3d(
        RWGSurfaceCurrentSpace3D(OrientedTriangleSurfaceComplex3D(vertices, faces))
    )
    mapped = prepare_buffa_christiansen_dual_3d(
        RWGSurfaceCurrentSpace3D(
            OrientedTriangleSurfaceComplex3D(vertices @ affine.T + translation, faces)
        )
    )

    reference_refined = reference.barycentric_surface
    mapped_refined = mapped.barycentric_surface
    edge_scale = np.asarray(mapped_refined.edge_lengths) / np.asarray(
        reference_refined.edge_lengths
    )
    assert np.ptp(edge_scale) > 1.0
    surface_jacobian = np.asarray(mapped_refined.face_areas) / np.asarray(
        reference_refined.face_areas
    )
    expected = (
        np.einsum("ij,fej->fei", affine, _bc_values_at_refined_centroids(reference))
        / surface_jacobian[:, None, None]
    )
    np.testing.assert_allclose(
        _bc_values_at_refined_centroids(mapped), expected, rtol=2e-5, atol=2e-6
    )


def test_bc_divergence_commutes_to_dual_vertex_dofs_on_anisotropic_mesh():
    primal = RWGSurfaceCurrentSpace3D(
        OrientedTriangleSurfaceComplex3D(*_ANISOTROPIC_OCTAHEDRON)
    )
    dual = prepare_buffa_christiansen_dual_3d(primal)
    refined = dual.barycentric_surface
    integrated_divergence = np.asarray(refined.face_areas)[:, None] * (
        np.asarray(dual.barycentric_rwg.divergence_matrix)
        @ np.asarray(dual.barycentric_transform)
    )

    coarse_edges = np.asarray(primal.surface.edge_vertices)
    valence = np.bincount(coarse_edges.reshape(-1), minlength=primal.surface.vertex_count)
    dual_vertex_dofs = np.zeros((refined.face_count, primal.surface.vertex_count))
    for face_id, triangle in enumerate(np.asarray(refined.triangles)):
        pole = triangle[triangle < primal.surface.vertex_count]
        assert pole.shape == (1,)
        dual_vertex_dofs[face_id, pole[0]] = 1.0 / (2.0 * valence[pole[0]])

    expected = (
        dual_vertex_dofs[:, coarse_edges[:, 0]] - dual_vertex_dofs[:, coarse_edges[:, 1]]
    )
    np.testing.assert_allclose(integrated_divergence, expected, atol=1e-7)


def test_bc_preparation_fails_closed_outside_condition_envelope():
    primal = RWGSurfaceCurrentSpace3D(OrientedTriangleSurfaceComplex3D(*_TETRAHEDRON))
    with pytest.raises(ValueError, match="finite and exceed one"):
        prepare_buffa_christiansen_dual_3d(primal, maximum_condition_number=np.inf)
    with pytest.raises(ValueError, match="condition envelope"):
        prepare_buffa_christiansen_dual_3d(primal, maximum_condition_number=1.1)
