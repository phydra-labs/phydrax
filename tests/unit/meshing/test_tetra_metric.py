from __future__ import annotations

from fractions import Fraction
from itertools import combinations, permutations

import numpy as np
import pytest

import phydrax as phx
from phydrax._geometry_predicates import PredicateMode
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._tetra_metric import (
    execute_tetra_metric_adaptation,
    TetraMetricOutcome,
    TetraMetricStatus,
)
from phydrax.meshing._topology_edit import entity_keys, TopologyEditBlock


def _source(
    shape: int = 3,
    /,
    *,
    nonuniform: bool = False,
    close_edge: bool = False,
    vertex_id_stride: int = 1,
) -> CellMeshingResult:
    vertex_rows = np.arange(shape**3, dtype=np.int64).reshape((shape,) * 3)
    axis = (
        np.asarray((0.0, 0.25, 0.5, 1.0), dtype=np.float64)
        if nonuniform
        else np.arange(shape, dtype=np.float64) / float(shape - 1)
    )
    coordinates = np.stack(
        np.meshgrid(*(axis for _ in range(3)), indexing="ij"), axis=-1
    ).reshape((-1, 3))
    if close_edge:
        if shape != 4:
            raise ValueError("The close-edge fixture requires a four-point grid.")
        coordinates[int(vertex_rows[2, 1, 1]), 0] = 0.40625
    tetrahedra = []
    for cube in np.ndindex((shape - 1,) * 3):
        for order in permutations(range(3)):
            point = np.asarray(cube, dtype=np.int64)
            path = [point.copy()]
            for axis_index in order:
                point = point.copy()
                point[axis_index] += 1
                path.append(point)
            tetrahedron = np.asarray(
                [vertex_rows[tuple(row)] for row in path], dtype=np.int32
            )
            if (
                np.linalg.det(coordinates[tetrahedron[1:]] - coordinates[tetrahedron[0]])
                < 0.0
            ):
                tetrahedron[[1, 2]] = tetrahedron[[2, 1]]
            tetrahedra.append(tetrahedron)
    block = CellBlock(
        "tetrahedra",
        "tetrahedron",
        np.asarray(tetrahedra, dtype=np.int32),
        global_ids=np.arange(3000, 3000 + len(tetrahedra), dtype=np.int64),
    )
    mesh = CellMesh(
        coordinates,
        (block,),
        vertex_global_ids=100 * (vertex_id_stride - 1)
        + vertex_id_stride * np.arange(shape**3, dtype=np.int64),
    )
    return certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


def _adapt(
    source: CellMeshingResult,
    tensor: np.ndarray,
    /,
    *,
    protected_edges: np.ndarray | None = None,
    topology_operations: bool = True,
    relocation: bool = True,
) -> TetraMetricOutcome:
    mesh = source.mesh
    edge_count = mesh.entity_set(1).entity_ids.size
    return execute_tetra_metric_adaptation(
        source,
        np.broadcast_to(tensor, (mesh.vertex_global_ids.size, 3, 3)).copy(),
        cell_classes=np.zeros(mesh.entity_set(3).entity_ids.size, dtype=np.int64),
        facet_classes=np.zeros(mesh.entity_set(2).entity_ids.size, dtype=np.int64),
        edge_classes=np.zeros(edge_count, dtype=np.int64),
        protected_edges=(
            np.zeros(edge_count, dtype=np.bool_)
            if protected_edges is None
            else protected_edges
        ),
        fixed_vertices=np.zeros(mesh.vertex_global_ids.size, dtype=np.bool_),
        predicate_mode=PredicateMode.EXACT,
        topology_operations=topology_operations,
        relocation=relocation,
    )


def _target_edges(outcome: TetraMetricOutcome, /) -> set[tuple[int, int]]:
    vertex_ids = np.asarray(outcome.edit.vertex_global_ids, dtype=np.int64)
    edges: set[tuple[int, int]] = set()
    for block in outcome.edit.blocks:
        assert isinstance(block, TopologyEditBlock)
        for cell in np.asarray(block.cells, dtype=np.int64):
            cell_ids = tuple(int(vertex_ids[int(row)]) for row in cell)
            for first, second in combinations(cell_ids, 2):
                edges.add((first, second) if first < second else (second, first))
    return edges


def test_nonperiodic_refinement_reaches_the_unit_metric() -> None:
    outcome = _adapt(_source(), np.eye(3) * 4.0, relocation=False)

    assert outcome.evidence.status is TetraMetricStatus.COMPLETE
    assert outcome.evidence.splits > 0
    assert outcome.evidence.out_of_range_edges == 0
    assert outcome.evidence.minimum_metric_quality > 0.0
    assert outcome.evidence.maximum_region_volume_error <= 1.0e-12


def test_relocation_uses_live_native_vertices_and_exact_source_stencils() -> None:
    source = _source(4, nonuniform=True)
    outcome = _adapt(
        source,
        np.eye(3) * 4.0,
        topology_operations=False,
        relocation=True,
    )

    assert outcome.evidence.status is TetraMetricStatus.STALLED
    assert outcome.evidence.relocations > 0
    assert outcome.evidence.out_of_range_edges > 0
    assert outcome.evidence.minimum_metric_quality > 0.0
    assert outcome.evidence.maximum_region_volume_error <= 1.0e-12
    source_points = {
        int(identifier): np.asarray(source.mesh.coordinates[row])
        for row, identifier in enumerate(source.mesh.vertex_global_ids.tolist())
    }
    assert any(
        not np.array_equal(source_points[int(identifier)], point)
        for identifier, point in zip(
            outcome.edit.vertex_global_ids,
            outcome.edit.coordinates,
            strict=True,
        )
    )

    target_ids = np.asarray(outcome.edit.vertex_global_ids, dtype=np.int64)
    stencil_sources = np.asarray(outcome.edit.stencil_sources, dtype=np.int64)
    stencil_weights = np.asarray(outcome.edit.stencil_weights, dtype=np.float64)
    stencil_valid = np.asarray(outcome.edit.stencil_valid, dtype=np.bool_)
    source_rows = {
        int(identifier): row
        for row, identifier in enumerate(source.mesh.vertex_global_ids.tolist())
    }
    relocated = 0
    for row, target_id in enumerate(target_ids.tolist()):
        support = np.flatnonzero(stencil_valid[row])
        if not support.size:
            continue
        exact_weights = tuple(
            Fraction.from_float(float(stencil_weights[row, column]))
            for column in support.tolist()
        )
        assert sum(exact_weights, Fraction(0)) == 1
        if target_id not in source_rows or np.array_equal(
            source_points[target_id], outcome.edit.coordinates[row]
        ):
            continue
        relocated += 1
        source_support = stencil_sources[row, support]
        assert not (
            source_support.size == 1
            and int(source_support[0]) == target_id
            and exact_weights[0] == 1
        )
        exact_point = np.asarray(
            [
                float(
                    sum(
                        (
                            weight
                            * Fraction.from_float(
                                float(
                                    source.mesh.coordinates[
                                        source_rows[int(source_id)], axis
                                    ]
                                )
                            )
                            for source_id, weight in zip(
                                source_support.tolist(), exact_weights, strict=True
                            )
                        ),
                        Fraction(0),
                    )
                )
                for axis in range(3)
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(exact_point, outcome.edit.coordinates[row])
    assert relocated > 0


def test_relocation_is_invariant_under_sparse_vertex_ids() -> None:
    # Relocation reads source tetrahedra by snapshot row: global vertex IDs that
    # are not row positions must not change any relocated coordinate.
    contiguous = _adapt(
        _source(4, nonuniform=True),
        np.eye(3) * 4.0,
        topology_operations=False,
        relocation=True,
    )
    sparse = _adapt(
        _source(4, nonuniform=True, vertex_id_stride=7),
        np.eye(3) * 4.0,
        topology_operations=False,
        relocation=True,
    )

    assert sparse.evidence.status is contiguous.evidence.status
    assert sparse.evidence.relocations == contiguous.evidence.relocations > 0
    np.testing.assert_array_equal(
        (np.asarray(sparse.edit.vertex_global_ids) - 600) // 7,
        contiguous.edit.vertex_global_ids,
    )
    np.testing.assert_array_equal(sparse.edit.coordinates, contiguous.edit.coordinates)


def test_coarsening_updates_live_topology_and_conserves_region_volume() -> None:
    outcome = _adapt(_source(4, close_edge=True), np.eye(3) * 5.76, relocation=False)

    assert outcome.evidence.status is TetraMetricStatus.STALLED
    assert outcome.evidence.collapses > 0
    assert outcome.evidence.out_of_range_edges > 0
    assert outcome.evidence.minimum_metric_quality > 0.0
    assert outcome.evidence.maximum_region_volume_error <= 1.0e-12


def test_protected_boundary_edge_and_invalid_metric_are_refused() -> None:
    source = _source()
    mesh = source.mesh
    source_edges = np.asarray(entity_keys(mesh, 1), dtype=np.int64)
    protected = np.zeros((source_edges.shape[0],), dtype=np.bool_)
    protected_row = next(
        row for row, edge in enumerate(source_edges.tolist()) if tuple(edge) == (0, 1)
    )
    protected[protected_row] = True
    outcome = _adapt(
        source,
        np.eye(3) * 4.0,
        protected_edges=protected,
        relocation=False,
    )

    assert outcome.evidence.status is TetraMetricStatus.COMPLETE
    assert (0, 1) in _target_edges(outcome)
    target_ids = np.asarray(outcome.edit.vertex_global_ids, dtype=np.int64)
    target_rows = {
        int(identifier): row for row, identifier in enumerate(target_ids.tolist())
    }
    for identifier in (0, 1):
        np.testing.assert_array_equal(
            outcome.edit.coordinates[target_rows[identifier]],
            mesh.coordinates[identifier],
        )

    invalid = np.broadcast_to(np.eye(3), (mesh.vertex_global_ids.size, 3, 3)).copy()
    invalid[0, 0, 1] = 1.0
    with pytest.raises(ValueError, match="symmetric positive-definite"):
        _adapt(source, invalid)
