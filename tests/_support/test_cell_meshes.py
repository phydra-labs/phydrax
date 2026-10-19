"""Consumer invariants of shared cell mesh builders."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax import SpatialCoordinateContract
from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry import coordinate_lagrange_element
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.meshing import (
    BisectionHierarchy,
    certify_cell_mesh,
    MarkedMeshAdaptation,
    MeshAdaptationRoute,
    MeshAdaptationStatus,
)
from tests._support.cell_meshes import (
    bisect,
    kuhn_cube,
    red_refinement,
    reference_cell,
    storage,
    tetrahedral_bisection,
    unit_square,
    unit_tetrahedron,
)


# Host determinant reductions below have at most 48 terms; 128 eps includes
# the 3x3 determinant operations and the final sum for this bounded fixture.
_MEASURE_ROUNDOFF = 128 * np.finfo(np.float64).eps


def test_unit_tetrahedron_retains_authored_vertices_and_detects_orientation_fault() -> (
    None
):
    mesh = unit_tetrahedron()
    points = np.asarray(mesh.coordinates)
    cells = np.asarray(mesh.blocks[0].vertices)
    np.testing.assert_array_equal(points, ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)))
    np.testing.assert_array_equal(cells, ((0, 1, 2, 3),))
    corners = points[cells[0]]
    assert np.linalg.det(corners[1:] - corners[0]) == 1.0
    corrupted = cells[0, (0, 2, 1, 3)]
    altered = points[corrupted]
    assert np.linalg.det(altered[1:] - altered[0]) == -1.0
    with pytest.raises(AssertionError):
        assert np.linalg.det(altered[1:] - altered[0]) == 1.0


def test_unit_square_retains_oriented_partition_boundary_and_explicit_ids() -> None:
    vertex_ids = 100 + 3 * np.arange(25, dtype=np.int64)
    cell_ids = 1000 + 5 * np.arange(32, dtype=np.int64)
    mesh = unit_square(4, 0.3, vertex_global_ids=vertex_ids, cell_global_ids=cell_ids)
    points, cells = np.asarray(mesh.coordinates), np.asarray(mesh.blocks[0].vertices)
    assert points.shape == (25, 2) and cells.shape == (32, 3)
    np.testing.assert_array_equal(mesh.vertex_global_ids, vertex_ids)
    np.testing.assert_array_equal(mesh.blocks[0].global_ids, cell_ids)
    corners = points[cells]
    measures = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 2
    assert np.all(measures > 0)
    assert abs(measures.sum() - 1.0) <= _MEASURE_ROUNDOFF
    axis = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    original = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape((-1, 2))
    boundary = np.any((original == 0) | (original == 1), axis=1)
    np.testing.assert_array_equal(points[boundary], original[boundary])
    np.testing.assert_array_equal(points[original[:, 0] == 0.5, 0], 0.5)
    assert np.any(points[~boundary] != original[~boundary])


def test_kuhn_cube_retains_oriented_partition_boundary_and_explicit_ids() -> None:
    vertex_ids = 200 + 3 * np.arange(27, dtype=np.int64)
    cell_ids = 2000 + 5 * np.arange(48, dtype=np.int64)
    mesh = kuhn_cube(2, 0.3, vertex_global_ids=vertex_ids, cell_global_ids=cell_ids)
    points, cells = np.asarray(mesh.coordinates), np.asarray(mesh.blocks[0].vertices)
    assert points.shape == (27, 3) and cells.shape == (48, 4)
    np.testing.assert_array_equal(mesh.vertex_global_ids, vertex_ids)
    np.testing.assert_array_equal(mesh.blocks[0].global_ids, cell_ids)
    corners = points[cells]
    measures = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6
    assert np.all(measures > 0)
    assert abs(measures.sum() - 1.0) <= _MEASURE_ROUNDOFF
    original = np.asarray(tuple(np.ndindex(3, 3, 3)), dtype=np.float64) / 2
    boundary = np.any((original == 0) | (original == 1), axis=1)
    np.testing.assert_array_equal(points[boundary], original[boundary])
    np.testing.assert_array_equal(points[original[:, 0] == 0.5, 0], 0.5)
    assert np.any(points[~boundary] != original[~boundary])


@pytest.mark.parametrize("kind", ("triangle", "tetrahedron", "hexahedron"))
def test_reference_cell_retains_canonical_vertex_order(kind: str) -> None:
    mesh = reference_cell(kind)
    topology = reference_cell_topology(kind)
    np.testing.assert_array_equal(mesh.coordinates, topology.vertices)
    np.testing.assert_array_equal(
        mesh.blocks[0].vertices, np.arange(len(topology.vertices))[None]
    )
    assert mesh.blocks[0].cell_kind == kind and mesh.blocks[0].cell_count == 1


def test_bisect_uses_stable_marks_and_retains_coarsening_forest() -> None:
    mesh = unit_square(cell_global_ids=np.asarray((10, 30), dtype=np.int64))
    source = certify_cell_mesh(mesh, SpatialCoordinateContract.si())
    refined = bisect(source, refine=(10,))
    assert refined.route is MeshAdaptationRoute.NATIVE_BISECTION
    assert refined.status is MeshAdaptationStatus.COMPLETE
    assert isinstance(refined.request, MarkedMeshAdaptation)
    np.testing.assert_array_equal(refined.request.refine_cell_ids, (10,))
    assert sum(block.cell_count for block in refined.target.mesh.blocks) > 2
    assert isinstance(refined.hierarchy, BisectionHierarchy)
    # Coordinate restrictions may partition descendants into several blocks;
    # restoring the conforming forest needs every refined sibling.
    children = np.setdiff1d(
        np.concatenate(
            [np.asarray(block.global_ids) for block in refined.target.mesh.blocks]
        ),
        np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks]),
    )
    restored = bisect(refined.target, coarsen=children, hierarchy=refined.hierarchy)
    assert restored.status is MeshAdaptationStatus.COMPLETE
    assert isinstance(restored.request, MarkedMeshAdaptation)
    np.testing.assert_array_equal(restored.request.coarsen_cell_ids, children)
    assert restored.target.mesh.topology_id == mesh.topology_id


def test_storage_restores_coordinate_basis_coefficients_and_stable_routes() -> None:
    mesh = reference_cell("triangle")
    element = coordinate_lagrange_element("triangle", 2)
    coefficients = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    coefficients[:, 1] += 0.1 * coefficients[:, 0] * coefficients[:, 1]
    geometry = CellGeometrySpec(
        {"cells": element},
        {"cells": np.arange(len(coefficients), dtype=np.int64)[None]},
        coefficients,
    )
    bank = storage(mesh, geometry)
    restored = bank.restore_geometry()
    assert cell_geometry_id(restored) == cell_geometry_id(geometry)
    assert cell_geometry_id(restored) != cell_geometry_id(CellGeometrySpec.affine(mesh))
    np.testing.assert_array_equal(restored.coordinates, geometry.coordinates)
    np.testing.assert_array_equal(restored.geometry_dofs[0], geometry.geometry_dofs[0])
    np.testing.assert_array_equal(
        bank.coordinate_global_ids, np.arange(len(coefficients)) + 100
    )
    np.testing.assert_array_equal(
        bank.coordinate_owner, np.zeros(len(coefficients), dtype=np.int32)
    )
    assert bank.logical_topology_id == mesh.topology_id
    assert bank.logical_coordinate_geometry_id == cell_geometry_id(geometry)
    assert bank.partition_index == 0 and bank.partition_count == 2


def _assert_authored_partition(
    source: CellMesh, target: CellMesh, parents: NDArray[np.int64]
) -> None:
    """Independent barycentric containment and parent measure conservation."""
    dimension = source.coordinates.shape[1]
    source_corners = np.asarray(source.coordinates)[np.asarray(source.blocks[0].vertices)]
    target_corners = np.asarray(target.coordinates)[np.asarray(target.blocks[0].vertices)]
    source_frames = source_corners[:, 1:] - source_corners[:, :1]
    target_frames = target_corners[:, 1:] - target_corners[:, :1]
    source_measure = np.linalg.det(source_frames)
    target_measure = np.linalg.det(target_frames)
    assert np.all(source_measure > 0) and np.all(target_measure > 0)
    assert parents.shape == (len(target_corners),)
    for parent in range(len(source_corners)):
        children = target_corners[parents == parent]
        coordinates = np.linalg.solve(
            source_frames[parent].T,
            (children - source_corners[parent, 0]).reshape((-1, dimension)).T,
        ).T
        assert np.all(coordinates >= -_MEASURE_ROUNDOFF)
        assert np.all(coordinates.sum(axis=1) <= 1 + _MEASURE_ROUNDOFF)
        assert (
            abs(target_measure[parents == parent].sum() - source_measure[parent])
            <= _MEASURE_ROUNDOFF
        )


@pytest.mark.parametrize(
    "builder",
    (red_refinement, tetrahedral_bisection),
    ids=("triangle-red", "tetrahedron-edge"),
)
def test_authored_refinement_retains_parent_partition(
    builder: Callable[[], tuple[CellMesh, CellMesh, NDArray[np.int64]]],
) -> None:
    source, target, parents = builder()
    _assert_authored_partition(source, target, parents)
    np.testing.assert_array_equal(
        parents, np.repeat(np.arange(2, dtype=np.int64), target.blocks[0].cell_count // 2)
    )
    assert source.blocks[0].cell_count == 2
    assert target.blocks[0].cell_count == (8 if source.coordinates.shape[1] == 2 else 4)


@pytest.mark.parametrize(
    "builder",
    (red_refinement, tetrahedral_bisection),
    ids=("triangle-red", "tetrahedron-edge"),
)
def test_authored_refinement_oracle_detects_corrupted_parent(
    builder: Callable[[], tuple[CellMesh, CellMesh, NDArray[np.int64]]],
) -> None:
    source, target, parents = builder()
    corrupted = parents.copy()
    corrupted[0] = 1
    with pytest.raises(AssertionError):
        _assert_authored_partition(source, target, corrupted)
