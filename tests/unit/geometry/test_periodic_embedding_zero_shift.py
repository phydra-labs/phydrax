from __future__ import annotations

from itertools import permutations, product

import numpy as np
import pytest

from phydrax.discretization._cell_complex import PolyhedralConnectivity
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import certify_cell_geometry_validity
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_topology import PeriodicMeshTopology
from phydrax.geometry._global_embedding import certify_global_embedding
from phydrax.meshing._audit import audit_cell_mesh


def _periodic_tetrahedral_cube(
    *, interior_tetrahedron: bool = False
) -> tuple[CellMesh, CellGeometrySpec]:
    points = tuple(product((0, 1, 2), repeat=3))
    coordinates = np.asarray(tuple(tuple(value * 0.5 for value in point) for point in points), dtype=np.float64)
    vertex_index = {point: index for index, point in enumerate(points)}
    cells: list[tuple[int, int, int, int]] = []
    for cube in product((0, 1), repeat=3):
        for permutation in permutations(range(3)):
            path = [cube]
            first = list(cube)
            first[permutation[0]] += 1
            path.append(tuple(first))
            second = first.copy()
            second[permutation[1]] += 1
            path.append(tuple(second))
            path.append(tuple(value + 1 for value in cube))
            inversions = sum(
                permutation[left] > permutation[right]
                for left in range(3)
                for right in range(left + 1, 3)
            )
            row: tuple[int, int, int, int] = (
                vertex_index[path[0]],
                vertex_index[path[1]],
                vertex_index[path[2]],
                vertex_index[path[3]],
            )
            if inversions % 2:
                row = (row[0], row[2], row[1], row[3])
            cells.append(row)
    representative_index = {
        point: index for index, point in enumerate(points) if all(value < 2 for value in point)
    }
    representatives = np.asarray(
        [representative_index[tuple(value % 2 for value in point)] for point in points],
        dtype=np.int32,
    )
    shifts = np.asarray([tuple(value // 2 for value in point) for point in points], dtype=np.int32)
    if interior_tetrahedron:
        coordinates = np.concatenate(
            (
                coordinates,
                np.asarray(
                    (
                        (0.08, 0.08, 0.08),
                        (0.18, 0.08, 0.08),
                        (0.08, 0.18, 0.08),
                        (0.08, 0.08, 0.18),
                    ),
                    dtype=np.float64,
                ),
            )
        )
        cells.append((27, 28, 29, 30))
        representatives = np.concatenate((representatives, np.arange(27, 31, dtype=np.int32)))
        shifts = np.concatenate((shifts, np.zeros((4, 3), dtype=np.int32)))
    block = CellBlock("tetrahedra", "tetrahedron", np.asarray(cells, dtype=np.int32))
    lifted = CellMesh(coordinates, (block,))
    topology = PeriodicMeshTopology(
        lifted,
        PeriodicCell(np.eye(3, dtype=np.float64)),
        representatives,
        shifts,
    )
    mesh = CellMesh(coordinates, (block,), periodic_topology=topology)
    return mesh, CellGeometrySpec.affine(mesh)


def test_closed_periodic_tetrahedral_cube_is_embedded() -> None:
    mesh, geometry = _periodic_tetrahedral_cube()
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert certificate.periodic is mesh.periodic_topology


def test_zero_shift_finds_interior_tetrahedron_overlapping_periodic_cube_cover() -> None:
    mesh, geometry = _periodic_tetrahedral_cube(interior_tetrahedron=True)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "violated"
    assert any(finding.check == "periodic_cell_contact" for finding in certificate.findings)


def test_periodic_polyhedral_quotient_rejects_an_interior_overlap() -> None:
    cube_faces = (
        np.asarray((0, 3, 2, 1), dtype=np.int32),
        np.asarray((4, 5, 6, 7), dtype=np.int32),
        np.asarray((0, 1, 5, 4), dtype=np.int32),
        np.asarray((1, 2, 6, 5), dtype=np.int32),
        np.asarray((2, 3, 7, 6), dtype=np.int32),
        np.asarray((3, 0, 4, 7), dtype=np.int32),
    )
    cube_vertices = (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    )
    grid_indices = tuple(product((0, 1, 2), repeat=3))
    grid_vertex_rows = {
        point: row for row, point in enumerate(grid_indices)
    }
    grid_coordinates = np.asarray(
        [tuple(value / 2.0 for value in point) for point in grid_indices],
        dtype=np.float64,
    )
    component_offset = np.asarray((0.25, 0.25, 0.25), dtype=np.float64)
    coordinates = np.concatenate(
        (grid_coordinates, grid_coordinates + component_offset)
    )
    polyhedral_cells = []
    for component in range(2):
        component_base = component * len(grid_indices)
        for cube in product((0, 1), repeat=3):
            corners = tuple(
                component_base
                + grid_vertex_rows[
                    tuple(cube[axis] + local[axis] for axis in range(3))
                ]
                for local in cube_vertices
            )
            polyhedral_cells.append(
                tuple(
                    np.asarray(
                        [corners[int(index)] for index in face], dtype=np.int32
                    )
                    for face in cube_faces
                )
            )
    cell_ids = np.arange(
        100, 100 + len(polyhedral_cells), dtype=np.int64
    )
    vertex_ids = np.arange(coordinates.shape[0], dtype=np.int64)
    lifted = CellMesh.from_polyhedra(
        coordinates,
        tuple(polyhedral_cells),
        vertex_global_ids=vertex_ids,
        cell_global_ids=cell_ids,
    )
    representatives = np.arange(coordinates.shape[0], dtype=np.int32)
    shifts = np.zeros((coordinates.shape[0], 3), dtype=np.int32)
    for component in range(2):
        component_base = component * len(grid_indices)
        for row, point in enumerate(grid_indices):
            target_row = component_base + row
            representatives[target_row] = component_base + grid_vertex_rows[
                tuple(value % 2 for value in point)
            ]
            shifts[target_row] = tuple(value // 2 for value in point)
    periodic = PeriodicMeshTopology(
        lifted,
        PeriodicCell(np.eye(3, dtype=np.float64)),
        representatives,
        shifts,
    )
    connectivity = lifted.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Periodic overlap fixture requires polyhedral connectivity.")
    mesh = CellMesh(
        coordinates,
        lifted.blocks,
        vertex_global_ids=lifted.vertex_global_ids,
        polyhedral_connectivity=connectivity,
        periodic_topology=periodic,
    )
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = certify_global_embedding(mesh, geometry, validity)
    audit = audit_cell_mesh(
        mesh,
        geometry,
        validity=validity,
        global_embedding=embedding,
    )

    assert validity.all_certified
    assert embedding.status == "violated"
    assert any(
        finding.check == "periodic_cell_contact" for finding in embedding.findings
    )
    assert "self_intersection" in audit.issues


def test_periodic_audit_rejects_mismatched_embedding_proof_binding() -> None:
    mesh, geometry = _periodic_tetrahedral_cube()
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    other_mesh, other_geometry = _periodic_tetrahedral_cube(
        interior_tetrahedron=True
    )
    other_validity = certify_cell_geometry_validity(other_geometry, mesh=other_mesh)
    unrelated_embedding = certify_global_embedding(
        other_mesh, other_geometry, other_validity
    )

    with pytest.raises(ValueError, match="MeshCertificateBinding"):
        audit_cell_mesh(
            mesh,
            geometry,
            validity=validity,
            global_embedding=unrelated_embedding,
        )


def test_periodic_audit_rejects_mismatched_periodic_topology_proof_binding() -> None:
    mesh, geometry = _periodic_tetrahedral_cube()
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = certify_global_embedding(mesh, geometry, validity)
    if mesh.periodic_topology is None:
        raise TypeError("Periodic proof fixture lost its quotient topology.")
    lifted = CellMesh(
        mesh.coordinates,
        mesh.blocks,
        vertex_global_ids=mesh.vertex_global_ids,
    )
    alternate_topology = PeriodicMeshTopology(
        lifted,
        mesh.periodic_topology.cell,
        np.arange(mesh.coordinates.shape[0], dtype=np.int32),
        np.zeros((mesh.coordinates.shape[0], 3), dtype=np.int32),
    )
    alternate_mesh = CellMesh(
        mesh.coordinates,
        mesh.blocks,
        vertex_global_ids=mesh.vertex_global_ids,
        periodic_topology=alternate_topology,
    )
    alternate_geometry = CellGeometrySpec.affine(alternate_mesh)
    alternate_validity = certify_cell_geometry_validity(
        alternate_geometry, mesh=alternate_mesh
    )

    with pytest.raises(ValueError, match="MeshCertificateBinding"):
        audit_cell_mesh(
            alternate_mesh,
            alternate_geometry,
            validity=alternate_validity,
            global_embedding=embedding,
        )
