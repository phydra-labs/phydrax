#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax


jax.config.update("jax_enable_x64", True)

import numpy as np

import phydrax as phx
from phydrax import ein
from phydrax.discretization._cell_geometry import coordinate_lagrange_element
from phydrax.discretization.fem import (
    form_element,
    prepare_nested_field_transfer,
)
from phydrax.discretization.fem._generic import _evaluate_paired_field_basis


D = phx.discretization

_VERTICES = np.concatenate((np.zeros((1, 3)), np.eye(3)))
# Reference tetrahedron edge order (local vertex pairs, oriented a -> b).
_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(4)
_GAUSS_NODES = 0.5 * (_GAUSS_NODES + 1.0)
_GAUSS_WEIGHTS = 0.5 * _GAUSS_WEIGHTS


def _kuhn_cube(count: int) -> Any:
    """Unit cube split into ``6 count^3`` positively oriented Kuhn tetrahedra."""

    axis = np.linspace(0.0, 1.0, count + 1)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        (-1, 3)
    )
    cells = []
    for corner in np.ndindex(count, count, count):
        for order in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
            vertex = np.asarray(corner)
            path = [vertex.copy()]
            for axis_index in order:
                vertex[axis_index] += 1
                path.append(vertex.copy())
            cells.append([np.ravel_multi_index(tuple(v), (count + 1,) * 3) for v in path])
    tetrahedra = np.asarray(cells, dtype=np.int32)
    corners = points[tetrahedra]
    volume = np.linalg.det(corners[:, 1:] - corners[:, :1])
    tetrahedra[volume < 0.0] = tetrahedra[volume < 0.0][:, (0, 2, 1, 3)]
    return D.CellMesh.from_tetrahedra(points, tetrahedra)


def _contract(subscripts: str, *operands: Any) -> np.ndarray:
    return np.asarray(ein.contract(subscripts, *operands))


def _exact(points: np.ndarray) -> np.ndarray:
    # curl curl E = pi^2 E for this field.
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return np.stack((np.sin(np.pi * y), np.sin(np.pi * z), np.sin(np.pi * x)), axis=-1)


def _exact_curl(points: np.ndarray) -> np.ndarray:
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return -np.pi * np.stack(
        (np.cos(np.pi * z), np.cos(np.pi * x), np.cos(np.pi * y)), axis=-1
    )


def _edge_circulations(mesh: Any) -> np.ndarray:
    """Gauss line integrals of the exact field along lower-to-higher edges."""

    coordinates = np.asarray(mesh.coordinates)
    edges = np.asarray(mesh.connectivity.edges)
    start, stop = coordinates[edges[:, 0]], coordinates[edges[:, 1]]
    points = start[:, None] + _GAUSS_NODES[None, :, None] * (stop - start)[:, None]
    tangential = np.sum(_exact(points) * (stop - start)[:, None], axis=-1)
    return tangential @ _GAUSS_WEIGHTS


def _second_order_moments(mesh: Any, field: Any) -> np.ndarray:
    """Independent order-two trimmed exterior moments of ``field``.

    Each sorted edge uses endpoint barycentrics ``t`` then ``1 - t``.
    On each sorted face the wedge-paired constant test one-forms give
    ``-int u . (x_g2 - x_g0)`` then ``int u . (x_g1 - x_g0)``.
    """

    coordinates = np.asarray(mesh.coordinates)
    edges = coordinates[np.asarray(mesh.connectivity.edges)]
    tangent = edges[:, 1] - edges[:, 0]
    points = edges[:, :1] + _GAUSS_NODES[None, :, None] * tangent[:, None]
    tangential = np.sum(field(points) * tangent[:, None], axis=-1)
    edge_moments = np.stack(
        (
            tangential @ (_GAUSS_WEIGHTS * _GAUSS_NODES),
            tangential @ (_GAUSS_WEIGHTS * (1.0 - _GAUSS_NODES)),
        ),
        axis=1,
    )
    faces = coordinates[np.asarray(mesh.connectivity.faces)]
    first, second = np.meshgrid(_GAUSS_NODES, _GAUSS_NODES, indexing="ij")
    s, t = first.reshape(-1), ((1.0 - first) * second).reshape(-1)
    weights = (_GAUSS_WEIGHTS[:, None] * _GAUSS_WEIGHTS[None] * (1.0 - first)).reshape(-1)
    frame = faces[:, 1:] - faces[:, :1]
    face_points = (
        faces[:, None, 0]
        + s[None, :, None] * frame[:, None, 0]
        + t[None, :, None] * frame[:, None, 1]
    )
    values = field(face_points)
    tangential_face = _contract("q,fqd,fkd->fk", weights, values, frame)
    face_moments = np.stack((-tangential_face[:, 1], tangential_face[:, 0]), axis=1)
    return np.concatenate((edge_moments.reshape(-1), face_moments.reshape(-1)))


def _curl(gradients: np.ndarray) -> np.ndarray:
    # gradients[..., m, j] = d(phi_m)/d(x_j).
    return np.stack(
        (
            gradients[..., 2, 1] - gradients[..., 1, 2],
            gradients[..., 0, 2] - gradients[..., 2, 0],
            gradients[..., 1, 0] - gradients[..., 0, 1],
        ),
        axis=-1,
    )


def _oriented_basis(discretization: Any) -> tuple[Any, ...]:
    rule = phx.integration.ReferenceTetrahedronRule(
        phx.integration.GaussLegendreRule(4)
    ).materialize()
    points, weights = rule.points, rule.weights
    geometry = discretization.evaluate_block_geometry(
        "E", 0, discretization.default_runtime.coordinates, points, weights
    )
    dof_map = discretization.dof_maps[0]
    transforms = np.asarray(dof_map.cell_transforms[0])
    return (
        np.asarray(dof_map.cell_dofs[0]),
        _contract("cqav,cai->cqiv", np.asarray(geometry.basis_values), transforms),
        _contract(
            "cqav,cai->cqiv", _curl(np.asarray(geometry.physical_gradients)), transforms
        ),
        np.asarray(geometry.physical_weights),
        np.asarray(geometry.physical_points),
    )


def _solve_maxwell(count: int, degree: int = 0) -> tuple[float, float]:
    """Solve curl curl E + E = f with exact tangential traces; L2 and curl errors."""

    mesh = _kuhn_cube(count)
    discretization = D.FiniteElementPlan(
        mesh,
        D.FiniteElementFieldSpec(
            "E", form_element("tetrahedron", 1, degree + 1, proxy="circulation")
        ),
    ).prepare()
    routes, basis, curls, weights, points = _oriented_basis(discretization)
    size = discretization.dof_maps[0].global_dof_count
    local = _contract("cq,cqiv,cqjv->cij", weights, curls, curls) + _contract(
        "cq,cqiv,cqjv->cij", weights, basis, basis
    )
    load = _contract("cq,cqiv,cqv->ci", weights, basis, (np.pi**2 + 1.0) * _exact(points))
    matrix = np.zeros((size, size))
    np.add.at(matrix, (routes[:, :, None], routes[:, None, :]), local)
    rhs = np.zeros((size,))
    np.add.at(rhs, routes, load)
    boundary = np.asarray(discretization.dof_maps[0].boundary_dof_mask)
    traces = (
        _edge_circulations(mesh) if degree == 0 else _second_order_moments(mesh, _exact)
    )
    solution = np.where(boundary, traces, 0.0)
    free = ~boundary
    solution[free] = np.linalg.solve(
        matrix[np.ix_(free, free)], (rhs - matrix @ solution)[free]
    )
    field = _contract("cqiv,ci->cqv", basis, solution[routes])
    curl = _contract("cqiv,ci->cqv", curls, solution[routes])
    return (
        float(np.sqrt(np.sum(weights * np.sum((field - _exact(points)) ** 2, -1)))),
        float(np.sqrt(np.sum(weights * np.sum((curl - _exact_curl(points)) ** 2, -1)))),
    )


def test_tetrahedral_nedelec_element_has_unit_oriented_edge_circulations() -> None:
    element = form_element("tetrahedron", 1, 1, proxy="circulation")
    circulations = np.empty((6, 6))
    for edge, (start, stop) in enumerate(_EDGES):
        tangent = _VERTICES[stop] - _VERTICES[start]
        points = _VERTICES[start] + _GAUSS_NODES[:, None] * tangent
        values, _ = element.tabulate(points)
        circulations[edge] = _GAUSS_WEIGHTS @ (np.asarray(values) @ tangent)

    np.testing.assert_allclose(circulations, np.eye(6), atol=1e-14)


def test_generic_tetrahedral_hcurl_matches_the_edge_operator_space() -> None:
    mesh = _kuhn_cube(2)
    discretization = D.FiniteElementPlan(
        mesh,
        D.FiniteElementFieldSpec(
            "E", form_element("tetrahedron", 1, 1, proxy="circulation")
        ),
    ).prepare()
    routes, basis, curls, weights, _ = _oriented_basis(discretization)
    space = D.FiniteElementDeRhamComplex(mesh, family="trimmed", order=1)
    values = np.random.default_rng(3).normal(size=space.cell_counts[1])
    local = _contract("cq,cqiv,cqjv->cij", weights, curls, curls) + 2.0 * _contract(
        "cq,cqiv,cqjv->cij", weights, basis, basis
    )
    action = np.zeros_like(values)
    np.add.at(action, routes, _contract("cij,cj->ci", local, values[routes]))

    np.testing.assert_allclose(
        action,
        np.asarray(
            2.0 * space.hodge_star(1, values)
            + space.hilbert_complex()
            .differential(1)
            .transpose_mv(space.hodge_star(2, space.exterior_derivative(1, values)))
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_tetrahedral_maxwell_manufactured_solution_converges() -> None:
    coarse = _solve_maxwell(2)
    fine = _solve_maxwell(4)

    # Lowest-order Nedelec: first-order convergence of the field and its curl
    # (observed errors 0.51/1.35 at h = 1/2 and 0.27/0.69 at h = 1/4; a sign or
    # orientation fault in the edge layout breaks tangential continuity and
    # stalls the rate).
    assert fine[0] < 0.3 and fine[1] < 0.75
    assert coarse[0] / fine[0] > 1.8
    assert coarse[1] / fine[1] > 1.8


def test_second_order_tetrahedral_nedelec_is_dual_to_its_moments() -> None:
    element = form_element("tetrahedron", 1, 2, proxy="circulation")
    nodes = _GAUSS_NODES
    first, second = np.meshgrid(nodes, nodes, indexing="ij")
    s, t = first.reshape(-1), ((1.0 - first) * second).reshape(-1)
    area_weights = (
        _GAUSS_WEIGHTS[:, None] * _GAUSS_WEIGHTS[None] * (1.0 - first)
    ).reshape(-1)
    rows = []
    for start, stop in _EDGES:
        tangent = _VERTICES[stop] - _VERTICES[start]
        values, _ = element.tabulate(_VERTICES[start] + nodes[:, None] * tangent)
        tangential = np.asarray(values) @ tangent
        rows.extend(
            (
                (_GAUSS_WEIGHTS * nodes) @ tangential,
                (_GAUSS_WEIGHTS * (1.0 - nodes)) @ tangential,
            )
        )
    for a, b, c in _FACES:
        points = _VERTICES[a] + s[:, None] * (_VERTICES[b] - _VERTICES[a])
        points = points + t[:, None] * (_VERTICES[c] - _VERTICES[a])
        values = np.asarray(element.tabulate(points)[0])
        rows.append(-area_weights @ (values @ (_VERTICES[c] - _VERTICES[a])))
        rows.append(area_weights @ (values @ (_VERTICES[b] - _VERTICES[a])))

    # The wedge-paired edge and face functionals are independently integrated.
    np.testing.assert_allclose(np.stack(rows), np.eye(20), atol=1e-13)


def test_second_order_nedelec_interpolates_its_space_with_continuous_traces() -> None:
    # a + B x + x × (A x) spans the second-order first-kind space; its
    # canonical moments on a Kuhn cube (every edge and face orientation occurs)
    # must reproduce it on every cell through the cell transformations.
    constant = np.asarray((0.3, -1.0, 0.7))
    linear = np.asarray(((0.5, -1.2, 0.1), (0.4, 0.9, -0.3), (-0.6, 0.2, 1.1)))
    quadratic = np.asarray(((1.0, 0.3, -0.5), (-0.2, 0.8, 0.6), (0.7, -0.4, 0.2)))

    def field(points: np.ndarray) -> np.ndarray:
        return constant + points @ linear.T + np.cross(points, points @ quadratic.T)

    mesh = _kuhn_cube(2)
    discretization = D.FiniteElementPlan(
        mesh,
        D.FiniteElementFieldSpec(
            "E", form_element("tetrahedron", 1, 2, proxy="circulation")
        ),
    ).prepare()
    routes, basis, _, _, points = _oriented_basis(discretization)
    interpolated = _contract(
        "cqiv,ci->cqv", basis, _second_order_moments(mesh, field)[routes]
    )

    np.testing.assert_allclose(interpolated, field(points), atol=1e-12)


def test_second_order_tetrahedral_maxwell_converges_quadratically() -> None:
    coarse = _solve_maxwell(2, 1)
    fine = _solve_maxwell(4, 1)

    # Second-order Nedelec: O(h^2) field and curl errors, so the energy error
    # drops by ~4 per halving (a transformation fault breaks tangential
    # continuity across faces and stalls the rate).
    coarse_energy, fine_energy = np.hypot(*coarse), np.hypot(*fine)
    assert coarse_energy / fine_energy > 3.5
    assert coarse[1] / fine[1] > 3.5


def test_second_order_nested_transfer_preserves_independent_entity_moments() -> None:
    source_mesh, target_mesh = _kuhn_cube(1), _kuhn_cube(2)
    element = form_element("tetrahedron", 1, 2, proxy="circulation")
    source, target = (
        D.FiniteElementPlan(mesh, D.FiniteElementFieldSpec("E", element)).prepare()
        for mesh in (source_mesh, target_mesh)
    )
    source_corners = np.asarray(source_mesh.coordinates)[
        np.asarray(source_mesh.blocks[0].vertices)
    ]
    target_corners = np.asarray(target_mesh.coordinates)[
        np.asarray(target_mesh.blocks[0].vertices)
    ]
    frames = np.swapaxes(source_corners[:, 1:] - source_corners[:, :1], -1, -2)
    centers = target_corners.mean(axis=1)
    reference = np.linalg.solve(
        frames[:, None], (centers[None] - source_corners[:, None, 0])[..., None]
    )[..., 0]
    barycentric = np.concatenate(
        (1.0 - reference.sum(axis=-1, keepdims=True), reference), -1
    )
    parents = np.argmax(np.all(barycentric >= -1e-13, axis=-1), axis=0)
    transfer = prepare_nested_field_transfer(source, target, parents, field_name="E")

    def field(points: np.ndarray) -> np.ndarray:
        linear = np.asarray(((0.5, -1.2, 0.1), (0.4, 0.9, -0.3), (-0.6, 0.2, 1.1)))
        quadratic = np.asarray(((1.0, 0.3, -0.5), (-0.2, 0.8, 0.6), (0.7, -0.4, 0.2)))
        return (
            np.asarray((0.3, -1.0, 0.7))
            + points @ linear.T
            + np.cross(points, points @ quadratic.T)
        )

    transferred = transfer.transfer.apply(_second_order_moments(source_mesh, field))
    assert transfer.evidence.passed
    np.testing.assert_allclose(
        transferred,
        _second_order_moments(target_mesh, field),
        atol=2e-12,
    )


def test_paired_curved_nedelec_derivative_includes_piola_factor_derivatives() -> None:
    # This positive vertex permutation also requires nonmonomial face changes.
    cells = np.asarray(((1, 0, 3, 2),), dtype=np.int32)
    mesh = D.CellMesh.from_tetrahedra(_VERTICES, cells)
    origin = _VERTICES[cells[0, 0]]
    frame = (_VERTICES[cells[0, 1:]] - origin).T

    def physical(reference: np.ndarray) -> np.ndarray:
        warped = reference.copy()
        warped[:, 2] += 0.07 * reference[:, 0] * (1.0 - reference[:, 0])
        return origin + warped @ frame.T

    def pullback(points: np.ndarray) -> np.ndarray:
        reference = np.linalg.solve(frame, (points - origin).T).T
        reference[:, 2] -= 0.07 * reference[:, 0] * (1.0 - reference[:, 0])
        return reference

    coordinate = coordinate_lagrange_element("tetrahedron", 2)
    controls = physical(np.asarray(coordinate.reference_nodes))
    discretization = D.FiniteElementPlan(
        mesh,
        D.FiniteElementFieldSpec(
            "E", form_element("tetrahedron", 1, 2, proxy="circulation")
        ),
        coordinate_spec=D.CellGeometrySpec(
            {"tetrahedra": coordinate},
            {"tetrahedra": np.arange(controls.shape[0], dtype=np.int32)[None]},
            controls,
        ),
    ).prepare()
    reference = np.asarray(((0.13, 0.17, 0.19), (0.20, 0.15, 0.14), (0.22, 0.13, 0.08)))
    selected = np.zeros(reference.shape[0], dtype=np.int32)
    dof_map = discretization.dof_maps[0]
    moments = np.random.default_rng(52).normal(size=dof_map.global_dof_count)
    local = moments[np.asarray(dof_map.cell_dofs[0])[0]]
    coordinates = discretization.default_runtime.coordinates
    physical_points = physical(reference)
    step = 1e-6
    for axis in range(3):
        shift = np.eye(3)[axis] * step
        plus, plus_valid = _evaluate_paired_field_basis(
            discretization,
            "E",
            0,
            selected,
            pullback(physical_points + shift),
            coordinates,
        )
        minus, minus_valid = _evaluate_paired_field_basis(
            discretization,
            "E",
            0,
            selected,
            pullback(physical_points - shift),
            coordinates,
        )
        derivative, valid = _evaluate_paired_field_basis(
            discretization, "E", 0, selected, reference, coordinates, derivative_axis=axis
        )
        actual = _contract("piv,i->pv", derivative, local)
        expected = _contract("piv,i->pv", (plus - minus) / (2.0 * step), local)
        assert np.all(valid) and np.all(plus_valid) and np.all(minus_valid)
        # A frozen-J Piola gradient passes affine cases but fails this actual
        # curved physical-coordinate derivative.
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-8)
