import jax
import numpy as np
import numpy.typing as npt
import pytest

import phydrax as phx
from phydrax.discretization.fem._form_elements import form_element, FormElementFamily
from phydrax.discretization.fem._generic import (
    FiniteElementDiscretization,
    FiniteElementDofMap,
)
from phydrax.discretization.fem._reference import FiniteElementSpec


_VERTICES = np.asarray(
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
)
_FACES = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))


def _physical_values(
    element: FiniteElementSpec,
    cell_points: npt.NDArray[np.float64],
    physical_points: npt.NDArray[np.float64],
    transform: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    jacobian = (cell_points[1:] - cell_points[0]).T
    reference_points = np.linalg.solve(jacobian, (physical_points - cell_points[0]).T).T
    reference_values, _ = element.tabulate(reference_points)
    physical = np.einsum("ab,qkb->qka", jacobian, np.asarray(reference_values))
    physical /= np.linalg.det(jacobian)
    return np.einsum("qia,ij->qja", physical, transform)


def test_tetrahedral_trimmed_one_has_unit_canonical_face_integrals() -> None:
    element = form_element("tetrahedron", 2, 1, twist="untwisted", proxy="flux")
    faces = element.entity_vertices[2]
    centers = np.asarray([np.mean(_VERTICES[list(face)], axis=0) for face in faces])
    values, _ = element.tabulate(centers)
    rows = []
    for face_index, face in enumerate(faces):
        corners = _VERTICES[np.asarray(face, dtype=np.int32)]
        area_normal = 0.5 * np.cross(corners[1] - corners[0], corners[2] - corners[0])
        rows.append(np.asarray(values)[face_index] @ area_normal)
    expected = np.zeros((4, element.local_dof_count), dtype=np.float64)
    for face_index, face_dofs in enumerate(element.entity_dofs[2]):
        expected[face_index, face_dofs[0]] = 1.0
    np.testing.assert_allclose(np.asarray(rows), expected, atol=1e-12)


def test_hdiv_stokes_prepares_bdm_dg_pair_with_explicit_gauge() -> None:
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        _VERTICES, np.asarray(((0, 1, 2, 3),))
    )
    prepared = phx.equations.fem.HDivStokesPlan(
        mesh, phx.discretization.PressureGaugePolicy("mean-zero")
    ).prepare()
    assert bool(prepared.evidence.successful)
    tangential = np.asarray(prepared.tangential_operator.as_dense())
    assert tangential.shape == (30, 30)
    assert np.linalg.norm(tangential) > 0.0
    velocity, pressure_state = prepared.state_space.zeros()
    assert velocity.shape == (30,)
    assert pressure_state.shape == (4,)
    divergence_coupling = jax.jacfwd(
        lambda coefficients: prepared.problem.residual((coefficients, pressure_state))[1]
    )(velocity)
    assert np.linalg.matrix_rank(np.asarray(divergence_coupling)) == 4
    np.testing.assert_allclose(tangential, tangential.T, atol=1e-12)
    pressure = prepared.gauge_pressure(np.asarray((1.0, 2.0, 4.0, 8.0)))
    np.testing.assert_allclose(np.mean(pressure), 0.0, atol=1e-12)


def test_hdiv_nitsche_couples_shared_tetrahedral_face_symmetrically() -> None:
    coordinates = np.concatenate((_VERTICES, np.asarray(((0.0, 0.0, -1.0),))))
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        coordinates, np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)))
    )
    prepared = phx.equations.fem.HDivStokesPlan(
        mesh, phx.discretization.PressureGaugePolicy("mean-zero")
    ).prepare()
    matrix = np.asarray(prepared.tangential_operator.as_dense())
    assert matrix.shape == (54, 54)
    np.testing.assert_allclose(matrix, matrix.T, atol=2e-12)
    face_count = mesh.entity_set(2).count
    assert prepared.tangential_operator.coefficients.size <= face_count * 60**2
    state = prepared.problem.state_space.zeros()
    residual = prepared.residual(state)
    np.testing.assert_allclose(residual[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(residual[1], 0.0, atol=1e-12)


def _face_pair(
    left_face: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Reflect the opposite vertex and misalign both local face numberings."""
    face = _FACES[left_face]
    corners = _VERTICES[np.asarray(face, dtype=np.int32)]
    normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    normal /= np.linalg.norm(normal)
    opposite = next(vertex for vertex in range(4) if vertex not in face)
    distance = np.dot(_VERTICES[opposite] - corners[0], normal)
    reflected = _VERTICES[opposite] - 2.0 * distance * normal
    coordinates = np.concatenate((_VERTICES, reflected[None, :]))
    right_face = _FACES[(left_face + 1) % 4]
    right = np.full((4,), 4, dtype=np.int32)
    right[np.asarray(right_face, dtype=np.int32)] = np.asarray(face[::-1], dtype=np.int32)
    jacobian = (coordinates[right[1:]] - coordinates[right[0]]).T
    if np.linalg.det(jacobian) < 0.0:
        first, second = right_face[:2]
        right[first], right[second] = right[second], right[first]
    return coordinates, np.stack((np.arange(4, dtype=np.int32), right))


def _assert_nonzero_shared_normal_continuity(
    element: FiniteElementSpec,
    dof_map: FiniteElementDofMap,
    coordinates: npt.NDArray[np.float64],
    cells: npt.NDArray[np.int32],
    left_face: int,
) -> None:
    face_points = coordinates[np.asarray(_FACES[left_face], dtype=np.int32)]
    barycentric = np.asarray(
        ((0.2, 0.3, 0.5), (0.6, 0.1, 0.3), (0.17, 0.61, 0.22)),
        dtype=np.float64,
    )
    points = barycentric @ face_points
    normal = np.cross(face_points[1] - face_points[0], face_points[2] - face_points[0])
    normal /= np.linalg.norm(normal)
    routes = np.asarray(dof_map.cell_dofs[0], dtype=np.int32)
    shared = np.intersect1d(routes[0], routes[1])
    assert shared.size == len(element.entity_dofs[2][left_face])
    bases = tuple(
        _physical_values(
            element,
            coordinates[cells[cell]],
            points,
            np.asarray(dof_map.cell_transforms[0][cell], dtype=np.float64),
        )
        for cell in range(2)
    )
    for global_dof in shared:
        traces = tuple(
            np.sum(bases[cell][:, routes[cell] == global_dof, :], axis=1) @ normal
            for cell in range(2)
        )
        # Continuity of a zero trace is vacuous: each shared functional must
        # produce a genuinely supported normal field on the claimed face.
        assert np.linalg.norm(traces[0]) > 1e-8
        assert np.linalg.norm(traces[1]) > 1e-8
        np.testing.assert_allclose(traces[0], traces[1], atol=1e-12)


@pytest.mark.parametrize("left_face", range(4))
@pytest.mark.parametrize("family,order", (("trimmed", 1), ("full", 1), ("full", 2)))
def test_hdiv_all_faces_have_nonzero_continuous_normal_trace(
    left_face: int, family: FormElementFamily, order: int
) -> None:
    coordinates, cells = _face_pair(left_face)
    mesh = phx.discretization.CellMesh.from_tetrahedra(coordinates, cells)
    element = form_element(
        "tetrahedron", 2, order, family=family, twist="untwisted", proxy="flux"
    )
    dof_map = FiniteElementDofMap(mesh, (element,))
    _assert_nonzero_shared_normal_continuity(
        element, dof_map, coordinates, cells, left_face
    )


@pytest.mark.parametrize("left_face", range(4))
def test_hdiv_stokes_all_faces_have_nonzero_continuous_normal_trace(
    left_face: int,
) -> None:
    coordinates, cells = _face_pair(left_face)
    mesh = phx.discretization.CellMesh.from_tetrahedra(coordinates, cells)
    prepared = phx.equations.fem.HDivStokesPlan(
        mesh, phx.discretization.PressureGaugePolicy("mean-zero")
    ).prepare()
    discretization = prepared.problem.discretization
    if not isinstance(discretization, FiniteElementDiscretization):
        raise AssertionError("HDivStokes must prepare its concrete FE discretization.")
    _assert_nonzero_shared_normal_continuity(
        discretization.elements[0][0],
        discretization.dof_maps[0],
        coordinates,
        cells,
        left_face,
    )


@pytest.mark.parametrize("face_id", (41, 43, 47, 53))
def test_hdiv_normal_flow_constraint_and_resistance_use_global_face_identity(
    face_id: int,
) -> None:
    mesh = phx.discretization.CellMesh(
        _VERTICES,
        (
            phx.discretization.CellBlock(
                "tetrahedra", "tetrahedron", np.asarray(((0, 1, 2, 3),))
            ),
        ),
        entity_global_ids={2: np.asarray((41, 43, 47, 53))},
    )
    boundary = phx.equations.fem.HDivNormalBoundaryCondition(
        np.asarray((face_id,)),
        resistance=2.5,
        prescribed_flux=3.0,
    )
    prepared = phx.equations.fem.HDivStokesPlan(
        mesh,
        phx.discretization.PressureGaugePolicy("mean-zero"),
        normal_boundaries=(boundary,),
    ).prepare()

    _, pressure, multiplier = prepared.state_space.zeros()
    discretization = prepared.problem.discretization
    if not isinstance(discretization, FiniteElementDiscretization):
        raise AssertionError("HDivStokes must prepare its concrete FE discretization.")
    element = discretization.elements[0][0]
    dof_map = discretization.dof_maps[0]
    face_index = int(
        np.flatnonzero(np.asarray(mesh.entity_set(2).entity_ids) == face_id)[0]
    )
    connectivity = mesh.connectivity
    if not isinstance(connectivity, phx.discretization.TetrahedralConnectivity):
        raise AssertionError("Normal flow requires tetrahedral face support.")
    corners = _VERTICES[np.asarray(connectivity.faces)[face_index]]
    area_normal = 0.5 * np.cross(corners[1] - corners[0], corners[2] - corners[0])
    if np.dot(area_normal, np.mean(corners, axis=0) - np.mean(_VERTICES, axis=0)) < 0:
        area_normal = -area_normal
    # Degree-two triangle quadrature and the physical reference basis give
    # an independent outward flux row on the selected global face support.
    points = (
        np.asarray(((2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)))
        @ corners
    )
    basis = _physical_values(
        element, _VERTICES, points, np.asarray(dof_map.cell_transforms[0][0])
    )
    local_flux = np.mean(basis, axis=0) @ area_normal
    flux = np.zeros(dof_map.global_dof_count)
    flux[np.asarray(dof_map.cell_dofs[0][0])] = local_flux
    assert np.linalg.norm(flux) > 1e-8
    velocity = 3.0 * flux / np.dot(flux, flux)
    np.testing.assert_allclose(np.dot(flux, velocity), 3.0, atol=1e-12)
    np.testing.assert_array_equal(prepared.normal_flux_face_ids, np.asarray((face_id,)))
    np.testing.assert_allclose(prepared.normal_flux(velocity), np.asarray((3.0,)))
    flux_roundoff = 50.0 * np.finfo(velocity.dtype).eps * 3.0
    np.testing.assert_allclose(
        prepared.normal_flux_residual(velocity), 0.0, atol=flux_roundoff
    )
    resistance_action = prepared.normal_resistance_operator.mv(velocity)
    np.testing.assert_allclose(
        np.vdot(np.asarray(velocity), np.asarray(resistance_action)),
        2.5 * 3.0**2,
        atol=1e-12,
    )
    constrained = prepared.residual((velocity, pressure, multiplier))
    np.testing.assert_allclose(constrained[2], 0.0, atol=1e-12)


def test_hdiv_normal_flow_rejects_an_interior_face() -> None:
    coordinates = np.concatenate((_VERTICES, np.asarray(((0.0, 0.0, -1.0),))))
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        coordinates, np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)))
    )
    interior = int(
        np.asarray(mesh.entity_set(2).entity_ids)[
            # ty: ignore[unresolved-attribute]
            np.flatnonzero(~np.asarray(mesh.connectivity.boundary_faces))[0]
        ]
    )
    boundary = phx.equations.fem.HDivNormalBoundaryCondition(
        np.asarray((interior,)),
        prescribed_flux=0.0,
    )
    with pytest.raises(ValueError, match="not exterior"):
        phx.equations.fem.HDivStokesPlan(
            mesh,
            phx.discretization.PressureGaugePolicy("mean-zero"),
            normal_boundaries=(boundary,),
        ).prepare()
