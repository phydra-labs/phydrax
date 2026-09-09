import jax
import numpy as np
import pytest

import phydrax as phx


_VERTICES = np.asarray(
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
)
_FACES = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))
_NORMALS = np.asarray(
    ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (1.0, 1.0, 1.0), (-1.0, 0.0, 0.0))
)
_NORMALS /= np.linalg.norm(_NORMALS, axis=1)[:, None]
_AREAS = np.asarray((0.5, 0.5, np.sqrt(3.0) / 2.0, 0.5))


def _unit_gauss(order):
    points, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (points + 1.0), 0.5 * weights


def _face_quadrature(face, order=5):
    nodes, weights = _unit_gauss(order)
    vertices = _VERTICES[list(face)]
    surface_jacobian = np.linalg.norm(
        np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    )
    points = []
    physical_weights = []
    barycentric = []
    for first, first_weight in zip(nodes, weights, strict=True):
        for second, second_weight in zip(nodes, weights, strict=True):
            coordinates = np.asarray(
                (
                    1.0 - first - (1.0 - first) * second,
                    first,
                    (1.0 - first) * second,
                )
            )
            points.append(coordinates @ vertices)
            barycentric.append(coordinates)
            physical_weights.append(
                first_weight * second_weight * (1.0 - first) * surface_jacobian
            )
    return np.asarray(points), np.asarray(physical_weights), np.asarray(barycentric)


def _tetrahedron_quadrature(order=6):
    nodes, weights = _unit_gauss(order)
    points = []
    physical_weights = []
    for first, first_weight in zip(nodes, weights, strict=True):
        for second, second_weight in zip(nodes, weights, strict=True):
            for third, third_weight in zip(nodes, weights, strict=True):
                points.append(
                    (
                        first,
                        (1.0 - first) * second,
                        (1.0 - first) * (1.0 - second) * third,
                    )
                )
                physical_weights.append(
                    first_weight
                    * second_weight
                    * third_weight
                    * (1.0 - first) ** 2
                    * (1.0 - second)
                )
    return np.asarray(points), np.asarray(physical_weights)


def _physical_values(element, cell_points, physical_points, orientation):
    jacobian = (cell_points[1:] - cell_points[0]).T
    reference_points = (physical_points - cell_points[0]) @ np.linalg.inv(jacobian).T
    reference_values, _ = element.tabulate(reference_points)
    return (
        np.einsum("ab,qkb->qka", jacobian, np.asarray(reference_values))
        / np.linalg.det(jacobian)
        * orientation[None, :, None]
    )


def test_tetrahedral_rt0_has_unit_oriented_face_fluxes():
    element = phx.discretization.tetrahedral_rt_element()
    centers = np.asarray([np.mean(_VERTICES[list(face)], axis=0) for face in _FACES])
    values, gradients = element.tabulate(centers)
    flux = np.asarray(
        [
            [
                _AREAS[face] * np.dot(values[face, basis], _NORMALS[face])
                for basis in range(4)
            ]
            for face in range(4)
        ]
    )
    np.testing.assert_allclose(flux, np.eye(4), atol=1e-12)
    divergence = np.trace(np.asarray(gradients), axis1=-2, axis2=-1)
    np.testing.assert_allclose(divergence, 6.0)


def test_tetrahedral_bdm1_is_dual_to_linear_face_flux_moments():
    element = phx.discretization.tetrahedral_bdm_element()
    rows = []
    for face_index, face in enumerate(_FACES):
        face_vertices = _VERTICES[list(face)]
        barycentric = np.asarray(
            ((2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3))
        )
        points = barycentric @ face_vertices
        values, _ = element.tabulate(points)
        normal_flux = np.asarray(values) @ _NORMALS[face_index]
        for moment in range(3):
            rows.append(
                _AREAS[face_index]
                / 3.0
                * np.sum(barycentric[:, moment, None] * normal_flux, axis=0)
            )
    np.testing.assert_allclose(np.asarray(rows), np.eye(12), atol=2e-12)


def test_tetrahedral_bdm2_is_dual_to_face_and_interior_moments():
    element = phx.discretization.tetrahedral_bdm_element(2)
    rows = []
    for face_index, face in enumerate(_FACES):
        points, weights, barycentric = _face_quadrature(face)
        values, _ = element.tabulate(points)
        normal_flux = np.asarray(values) @ _NORMALS[face_index]
        first, second, third = barycentric.T
        moments = np.column_stack(
            (
                first * first,
                second * second,
                third * third,
                2.0 * first * second,
                2.0 * second * third,
                2.0 * third * first,
            )
        )
        for moment in range(6):
            rows.append(
                np.sum(
                    weights[:, None] * moments[:, moment, None] * normal_flux,
                    axis=0,
                )
            )

    points, weights = _tetrahedron_quadrature()
    values, _ = element.tabulate(points)
    x, y, z = points.T
    zero = np.zeros_like(x)
    interior_tests = (
        np.column_stack((np.ones_like(x), zero, zero)),
        np.column_stack((zero, np.ones_like(x), zero)),
        np.column_stack((zero, zero, np.ones_like(x))),
        np.column_stack((zero, -z, y)),
        np.column_stack((z, zero, -x)),
        np.column_stack((-y, x, zero)),
    )
    for test in interior_tests:
        rows.append(
            np.sum(weights[:, None] * np.sum(values * test[:, None, :], axis=2), axis=0)
        )
    np.testing.assert_allclose(np.asarray(rows), np.eye(30), atol=2e-11)


def test_hdiv_stokes_prepares_bdm_dg_pair_with_explicit_gauge():
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


def test_hdiv_nitsche_couples_shared_tetrahedral_face_symmetrically():
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


def test_bdm2_normal_trace_is_continuous_across_a_shared_face():
    coordinates = np.concatenate((_VERTICES, np.asarray(((0.0, 0.0, -1.0),))))
    cells = np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)))
    mesh = phx.discretization.CellMesh.from_tetrahedra(coordinates, cells)
    prepared = phx.equations.fem.HDivStokesPlan(
        mesh, phx.discretization.PressureGaugePolicy("mean-zero")
    ).prepare()
    discretization = prepared.problem.discretization
    dof_map = discretization.dof_maps[0]
    element = discretization.elements[0][0]
    shared_face = int(np.flatnonzero(~np.asarray(mesh.connectivity.boundary_faces))[0])
    face_vertices = np.asarray(mesh.connectivity.faces)[shared_face]
    face_points = coordinates[face_vertices]
    barycentric = np.asarray(((0.2, 0.3, 0.5), (0.6, 0.1, 0.3)))
    physical_points = barycentric @ face_points
    center = np.mean(face_points, axis=0)
    base_normal = np.cross(
        face_points[1] - face_points[0], face_points[2] - face_points[0]
    )
    base_normal /= np.linalg.norm(base_normal)

    physical_bases = []
    outward_normals = []
    for cell in range(2):
        cell_points = coordinates[cells[cell]]
        physical_bases.append(
            _physical_values(
                element,
                cell_points,
                physical_points,
                np.asarray(dof_map.orientations[0][cell]),
            )
        )
        outward = base_normal.copy()
        if np.dot(outward, np.mean(cell_points, axis=0) - center) > 0.0:
            outward = -outward
        outward_normals.append(outward)

    for moment in range(6):
        velocity = np.zeros((dof_map.global_dof_count,))
        velocity[shared_face * 6 + moment] = 1.0
        traces = []
        for cell in range(2):
            local = velocity[np.asarray(dof_map.cell_dofs[0][cell])]
            field = np.einsum("k,qka->qa", local, physical_bases[cell])
            traces.append(field @ outward_normals[cell])
        np.testing.assert_allclose(traces[0] + traces[1], 0.0, atol=2e-11)


def test_hdiv_normal_flow_constraint_and_resistance_use_global_face_identity():
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
        np.asarray((47,)),
        resistance=2.5,
        prescribed_flux=3.0,
    )
    prepared = phx.equations.fem.HDivStokesPlan(
        mesh,
        phx.discretization.PressureGaugePolicy("mean-zero"),
        normal_boundaries=(boundary,),
    ).prepare()

    _, pressure, multiplier = prepared.state_space.zeros()
    velocity = prepared.normal_flux_operator.adjoint_mv(np.asarray((0.5,)))
    np.testing.assert_array_equal(prepared.normal_flux_face_ids, np.asarray((47,)))
    np.testing.assert_allclose(prepared.normal_flux(velocity), np.asarray((3.0,)))
    np.testing.assert_allclose(prepared.normal_flux_residual(velocity), 0.0)
    resistance_action = prepared.normal_resistance_operator.mv(velocity)
    np.testing.assert_allclose(
        np.vdot(np.asarray(velocity), np.asarray(resistance_action)),
        2.5 * 3.0**2,
        atol=1e-12,
    )
    constrained = prepared.residual((velocity, pressure, multiplier))
    np.testing.assert_allclose(constrained[2], 0.0, atol=1e-12)


def test_hdiv_normal_flow_rejects_an_interior_face():
    coordinates = np.concatenate((_VERTICES, np.asarray(((0.0, 0.0, -1.0),))))
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        coordinates, np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)))
    )
    interior = int(
        np.asarray(mesh.entity_set(2).entity_ids)[
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
