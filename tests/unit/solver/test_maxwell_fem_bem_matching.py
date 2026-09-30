from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

from phydrax.discretization import CellMesh
from phydrax.discretization._cell_complex import TetrahedralConnectivity
from phydrax.discretization.bem import (
    OrientedTriangleSurfaceComplex3D,
    prepare_buffa_christiansen_dual_3d,
    RWGSurfaceCurrentSpace3D,
)
from phydrax.discretization.bem._bc_dual import BuffaChristiansenDualSpace3D
from phydrax.discretization.fem._de_rham import FiniteElementDeRhamComplex
from phydrax.discretization.fem._interface_mortar3d import (
    prepare_maxwell_mortar_interface_trace_3d,
)
from phydrax.linalg import (
    mass_form,
    ScaledLinearOperator,
    stiffness_form,
    SumLinearOperator,
)
from phydrax.operators.integral.layer_potential._maxwell3d import MaxwellEFIEPolicy3D
from phydrax.operators.integral.layer_potential._maxwell_bc3d import (
    prepare_maxwell_bc_efie_3d,
)
from phydrax.solver._fem_bem_vector import (
    prepare_matching_maxwell_fem_bem_3d,
    PreparedMatchingMaxwellFEMBEM3D,
)
from phydrax.solver._nonmatching_fem_bem3d import prepare_maxwell_fem_bem_3d


_POINTS = np.asarray(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
_FACES = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)


def _boundary_policy() -> MaxwellEFIEPolicy3D:
    return MaxwellEFIEPolicy3D(
        regular_order=3,
        singular_order=3,
        near_order=3,
        absolute_tolerance=1.0,
        relative_tolerance=1.0,
        max_kh=2.0,
        max_condition_number=1e15,
    )


@pytest.fixture(scope="module")
def matching() -> Iterator[PreparedMatchingMaxwellFEMBEM3D]:
    with jax.enable_x64(True):
        complex = FiniteElementDeRhamComplex(
            CellMesh.from_tetrahedra(_POINTS, np.asarray([[0, 1, 2, 3]], dtype=np.int32)),
            family="trimmed",
            order=1,
            coefficient_dtype=jnp.complex128,
        )
        hilbert = complex.hilbert_complex()
        interior = SumLinearOperator(
            stiffness_form(hilbert, 1),
            ScaledLinearOperator(
                mass_form(hilbert, 1), jnp.asarray(-(0.8**2) + 0.1j, dtype=jnp.complex128)
            ),
        )
        yield prepare_matching_maxwell_fem_bem_3d(
            complex,
            interior,
            wavenumber=0.8,
            boundary_policy=_boundary_policy(),
            residual_tolerance=1e-9,
        )


def _coarse_tet_bc_cross_mass(
    dual: BuffaChristiansenDualSpace3D,
    edges: npt.NDArray[np.int32],
) -> npt.NDArray[np.float64]:
    """Independent degree-two quadrature of BC dot Whitney electric trace.

    The tetrahedron is the coordinate simplex, so barycentric gradients are
    known analytically; this oracle never calls the package FE reconstruction
    or matching trace and works on a genuinely different surface triangulation.
    """
    refined = dual.barycentric_surface
    points = np.asarray(refined.vertices)
    faces = np.asarray(refined.triangles)
    opposite = points[np.asarray(refined.opposite_vertices)]
    face_edges = np.asarray(refined.face_edges)
    scales = (
        np.asarray(refined.face_edge_signs)
        * np.asarray(refined.edge_lengths)[face_edges]
        / (2.0 * np.asarray(refined.face_areas)[:, None])
    )
    transform = np.asarray(dual.barycentric_transform)
    gradients = np.asarray(
        [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    quadrature = np.asarray(
        [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ],
        dtype=np.float64,
    )
    mass = np.zeros((dual.size, edges.shape[0]), dtype=np.float64)
    for face, triangle in enumerate(faces):
        for position in quadrature @ points[triangle]:
            barycentric = np.concatenate(([1.0 - position.sum()], position))
            electric = (
                barycentric[edges[:, 0], None] * gradients[edges[:, 1]]
                - barycentric[edges[:, 1], None] * gradients[edges[:, 0]]
            )
            local_rwg = scales[face, :, None] * (position - opposite[face])
            bc_values = transform[face_edges[face]].T @ local_rwg
            mass += np.asarray(refined.face_areas)[face] / 3.0 * (bc_values @ electric.T)
    return mass


def _tet_edges(complex: FiniteElementDeRhamComplex) -> npt.NDArray[np.int32]:
    connectivity = complex.mesh.connectivity
    if not isinstance(connectivity, TetrahedralConnectivity):
        raise TypeError("The Maxwell contract fixture requires tetrahedral connectivity.")
    return np.asarray(connectivity.edges, dtype=np.int32)


def _analytic_maxwell_matrix(edges: npt.NDArray[np.int32]) -> npt.NDArray[np.complex128]:
    """Exact coordinate-simplex Whitney curl-curl minus lossy mass form."""
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Whitney edge endpoints must have shape (edge_count, 2).")
    gradients = np.asarray(
        [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    moments = (np.ones((4, 4), dtype=np.float64) + np.eye(4, dtype=np.float64)) / 120.0
    mass = np.empty((edges.shape[0], edges.shape[0]), dtype=np.float64)
    for row in range(edges.shape[0]):
        a, b = edges[row, 0], edges[row, 1]
        for column in range(edges.shape[0]):
            c, d = edges[column, 0], edges[column, 1]
            mass[row, column] = (
                moments[a, c] * (gradients[b] @ gradients[d])
                - moments[a, d] * (gradients[b] @ gradients[c])
                - moments[b, c] * (gradients[a] @ gradients[d])
                + moments[b, d] * (gradients[a] @ gradients[c])
            )
    curls = 2.0 * np.cross(gradients[edges[:, 0]], gradients[edges[:, 1]])
    return np.asarray(
        (curls @ curls.T) / 6.0 + (-(0.8**2) + 0.1j) * mass, dtype=np.complex128
    )


def test_matching_trace_is_outward_rotated_whitney_and_dual(
    matching: PreparedMatchingMaxwellFEMBEM3D,
) -> None:
    with jax.enable_x64(True):
        constant = np.asarray([0.3, -0.4, 0.7])
        edges = _tet_edges(matching.volume_complex)
        circulation = (constant @ (_POINTS[edges[:, 1]] - _POINTS[edges[:, 0]]).T).astype(
            np.complex128
        )
        trace = matching.rwg_trace.mv(jnp.asarray(circulation))
        rwg = matching.boundary.current_space.primal
        surface = rwg.surface
        local = np.asarray(trace)[np.asarray(surface.face_edges)]
        reconstructed = np.sum(local[:, :, None] * np.asarray(rwg.centroid_basis), axis=1)
        np.testing.assert_allclose(
            reconstructed,
            np.cross(np.asarray(surface.face_normals), constant),
            atol=1e-12,
        )
        cross = _coarse_tet_bc_cross_mass(matching.boundary.current_space, edges)
        np.testing.assert_allclose(
            matching.dual_trace.mv(jnp.asarray(circulation)),
            cross @ circulation,
            atol=1e-12,
        )


def test_matching_actual_boundary_solve_recovers_prescribed_state(
    matching: PreparedMatchingMaxwellFEMBEM3D,
) -> None:
    with jax.enable_x64(True):
        n = matching.volume_complex.cell_counts[1]
        q = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.complex128)
        u = np.arange(1, n + 1, dtype=np.float64).astype(np.complex128) * (0.1 + 0.02j)
        edges = _tet_edges(matching.volume_complex)
        cross = _coarse_tet_bc_cross_mass(matching.boundary.current_space, edges)
        magnetic_trace = np.linalg.solve(
            np.asarray(matching.boundary.gram_operator.matrix),
            np.asarray(matching.boundary.magnetic_operator.matrix),
        )
        upper = 1j * 0.8 * cross.T @ magnetic_trace
        f = _analytic_maxwell_matrix(edges) @ u + upper @ q
        g = cross @ u - np.asarray(matching.boundary.operator.matrix) @ q
        result = matching.solve(f, g)
        assert bool(result.successful)
        assert result.relative_block_residual < 1e-9
        np.testing.assert_allclose(result.interior, u, atol=1e-8)
        np.testing.assert_allclose(result.boundary, q, atol=1e-8)
        assert not result.continuum_certified
        assert result.evidence_ids == matching.coupled.evidence_ids
        mortar = prepare_maxwell_mortar_interface_trace_3d(
            cross,
            upper.conj().T,
            volume_complex=matching.volume_complex,
            boundary_space=matching.boundary.current_space.vector_space,
            coverage_fraction=1.0,
            orientation_margin=1.0,
            geometric_residual=0.0,
            commuting_defect=0.0,
        )
        nonmatching = prepare_maxwell_fem_bem_3d(
            matching.coupled.interior_operator,
            matching.boundary,
            mortar,
            residual_tolerance=1e-9,
        ).solve(f, g)
        assert bool(nonmatching.successful)
        np.testing.assert_allclose(nonmatching.interior, result.interior, atol=1e-8)
        np.testing.assert_allclose(nonmatching.boundary, result.boundary, atol=1e-8)


def test_nonmatching_solved_surface_and_conormal_fault_adequacy(
    matching: PreparedMatchingMaxwellFEMBEM3D,
) -> None:
    with jax.enable_x64(True):
        # Subdivide one face only: this boundary is not the volume facet mesh.
        points = np.concatenate((_POINTS, _POINTS[_FACES[0]].mean(axis=0)[None]))
        faces = np.concatenate(
            (np.asarray([[0, 2, 4], [2, 1, 4], [1, 0, 4]], dtype=np.int32), _FACES[1:])
        )
        primal = RWGSurfaceCurrentSpace3D(OrientedTriangleSurfaceComplex3D(points, faces))
        dual = prepare_buffa_christiansen_dual_3d(primal)
        boundary = prepare_maxwell_bc_efie_3d(dual, 0.8, policy=_boundary_policy())
        edges = _tet_edges(matching.volume_complex)
        trace = _coarse_tet_bc_cross_mass(dual, edges)
        magnetic_trace = np.linalg.solve(
            np.asarray(boundary.gram_operator.matrix),
            np.asarray(boundary.magnetic_operator.matrix),
        )
        conormal = (
            (1j * 0.8 * trace.T @ magnetic_trace).conj().T
            * np.linspace(0.8, 1.2, dual.size)[:, None]
            * (1.0 + 0.15j)
        )
        mortar = prepare_maxwell_mortar_interface_trace_3d(
            trace,
            conormal,
            volume_complex=matching.volume_complex,
            boundary_space=boundary.current_space.vector_space,
            coverage_fraction=1.0,
            orientation_margin=1.0,
            geometric_residual=0.0,
            commuting_defect=0.0,
        )
        interior = matching.coupled.interior_operator
        coupled = prepare_maxwell_fem_bem_3d(
            interior, boundary, mortar, residual_tolerance=1e-9
        )
        u = np.linspace(0.1, 0.6, edges.shape[0]).astype(np.complex128)
        q = (np.linspace(-0.4, 0.3, dual.size) + 0.1j).astype(np.complex128)
        f = _analytic_maxwell_matrix(edges) @ u + conormal.conj().T @ q
        g = trace @ u - np.asarray(boundary.operator.matrix) @ q
        result = coupled.solve(f, g)
        assert bool(result.successful)
        np.testing.assert_allclose(result.interior, u, atol=1e-8)
        np.testing.assert_allclose(result.boundary, q, atol=1e-8)
        np.testing.assert_allclose(result.interior_residual, 0.0, atol=1e-8)
        np.testing.assert_allclose(result.interface_residual, 0.0, atol=1e-8)
        # A plausible broken implementation replacing N* with T* cannot pass.
        wrong = np.block(
            [
                [_analytic_maxwell_matrix(edges), trace.T],
                [trace, -np.asarray(boundary.operator.matrix)],
            ]
        )
        wrong_state = np.linalg.solve(wrong, np.concatenate((f, g)))
        assert np.linalg.norm(wrong_state - np.concatenate((u, q))) > 1e-3


def test_matching_refuses_low_frequency_unsupported_boundary(
    matching: PreparedMatchingMaxwellFEMBEM3D,
) -> None:
    with jax.enable_x64(True), pytest.raises(ValueError, match="Low-frequency"):
        prepare_matching_maxwell_fem_bem_3d(
            matching.volume_complex,
            matching.coupled.interior_operator,
            wavenumber=1e-5,
            boundary_policy=_boundary_policy(),
        )
