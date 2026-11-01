from __future__ import annotations

from fractions import Fraction
from itertools import combinations, permutations

import numpy as np
import pytest

import phydrax as phx
from phydrax._geometry_predicates import PredicateMode
from phydrax._meshcore import TetMeshEdit, TetMeshEditKind, TetMeshEditStatus
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._tetra_metric import (
    _facet_group_map,
    _NativeState,
    _prepare_native_state,
    _prepare_native_state,
    _split_phase,
    _unit_metric_status,
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


def test_graded_split_points_are_exact_dyadic_edge_points() -> None:
    # A graded size places metric midpoints at full-precision parameters; such a
    # split point leaves a source facet diagonal by rounding and its nested
    # ancestry outgrows binary64. Short dyadic parameters keep every split exact.
    source = _source()
    mesh = source.mesh
    sizes = 0.12 + 0.3 * np.asarray(mesh.coordinates[:, 0], dtype=np.float64)
    metric = np.eye(3)[None] / (sizes * sizes)[:, None, None]
    state = _prepare_native_state(
        source,
        metric,
        cell_classes=np.zeros(mesh.entity_set(3).entity_ids.size, dtype=np.int64),
        facet_classes=np.zeros(mesh.entity_set(2).entity_ids.size, dtype=np.int64),
        edge_classes=np.zeros(mesh.entity_set(1).entity_ids.size, dtype=np.int64),
        protected_edges=np.zeros(mesh.entity_set(1).entity_ids.size, dtype=np.bool_),
        fixed_vertices=np.zeros(mesh.vertex_global_ids.size, dtype=np.bool_),
    )
    counters = dict.fromkeys(
        (
            "splits",
            "rejected_topology",
            "rejected_protected",
            "rejected_uncertain",
            "rejected_invalid",
        ),
        0,
    )

    applied = _split_phase(state, counters)

    assert applied == counters["splits"] > 6
    assert counters["rejected_uncertain"] == counters["rejected_invalid"] == 0
    _, outside, _, longest, _ = _unit_metric_status(state)
    assert outside > 0 and longest <= np.sqrt(2.0)
    source_rows = {
        int(identifier): row for row, identifier in enumerate(mesh.vertex_global_ids)
    }
    snapshot = state.snapshot
    for identifier, point in zip(
        np.asarray(snapshot.vertex_ids).tolist(), np.asarray(snapshot.points), strict=True
    ):
        stencil = state.vertex_stencils[int(identifier)]
        exact = [
            sum(
                (
                    weight * Fraction(float(mesh.coordinates[source_rows[key], axis]))
                    for key, weight in stencil.items()
                ),
                Fraction(0),
            )
            for axis in range(3)
        ]
        assert [Fraction(float(value)) for value in point] == exact
        assert all(
            Fraction(float(weight)) == weight and weight.denominator.bit_count() == 1
            for weight in stencil.values()
        )


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

    assert outcome.evidence.status is TetraMetricStatus.COMPLETE
    assert outcome.evidence.collapses > 0
    assert outcome.evidence.out_of_range_edges == 0
    assert outcome.evidence.minimum_metric_quality > 0.0
    assert outcome.evidence.maximum_region_volume_error <= 1.0e-12
    # COMPLETE is the measured state of the published target: every target edge
    # has unit-metric length 2.4 |e| inside [1/sqrt(2), sqrt(2)].
    target_ids = np.asarray(outcome.edit.vertex_global_ids, dtype=np.int64)
    rows = {int(identifier): row for row, identifier in enumerate(target_ids.tolist())}
    coordinates = np.asarray(outcome.edit.coordinates, dtype=np.float64)
    lengths = np.asarray(
        [
            2.4 * np.linalg.norm(coordinates[rows[first]] - coordinates[rows[second]])
            for first, second in sorted(_target_edges(outcome))
        ]
    )
    assert lengths.size and np.all(
        (lengths >= 1.0 / np.sqrt(2.0)) & (lengths <= np.sqrt(2.0))
    )


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


def _mesh_result(
    coordinates: np.ndarray, tetrahedra: list[list[int]], /
) -> CellMeshingResult:
    cells = np.asarray(tetrahedra, dtype=np.int32)
    for cell in cells:
        if np.linalg.det(coordinates[cell[1:]] - coordinates[cell[0]]) < 0.0:
            cell[[1, 2]] = cell[[2, 1]]
    block = CellBlock(
        "tetrahedra",
        "tetrahedron",
        cells,
        global_ids=np.arange(3000, 3000 + cells.shape[0], dtype=np.int64),
    )
    mesh = CellMesh(
        coordinates,
        (block,),
        vertex_global_ids=np.arange(coordinates.shape[0], dtype=np.int64),
    )
    return certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())

def _state(source: CellMeshingResult, metric: np.ndarray, /) -> _NativeState:
    mesh = source.mesh
    edge_count = mesh.entity_set(1).entity_ids.size
    return _prepare_native_state(
        source,
        np.broadcast_to(metric, (mesh.vertex_global_ids.size, 3, 3)).copy(),
        cell_classes=np.zeros(mesh.entity_set(3).entity_ids.size, dtype=np.int64),
        facet_classes=np.zeros(mesh.entity_set(2).entity_ids.size, dtype=np.int64),
        edge_classes=np.zeros(edge_count, dtype=np.int64),
        protected_edges=np.zeros(edge_count, dtype=np.bool_),
        fixed_vertices=np.zeros(mesh.vertex_global_ids.size, dtype=np.bool_),
    )

@pytest.mark.parametrize("layout", ("C", "F", "strided"))
@pytest.mark.parametrize("explicit_removed", (False, True))
def test_multi_pair_collapse_edit_commits_through_the_native_wrapper(
    layout: str, explicit_removed: bool
) -> None:
    # Axis vertices A p q s t B of a double cone over a hexagon: one COLLAPSE
    # edit removes p into q and t into s. The wrapper hands C the pair columns;
    # a strided survivor column was read as (q, t), naming a removed vertex.
    angles = np.arange(6) * (np.pi / 3.0)
    ring = np.stack((np.zeros(6), 0.9 * np.cos(angles), 0.9 * np.sin(angles)), axis=-1)
    axis = np.asarray((-1.6, -0.7, -0.3, 0.3, 0.7, 1.6))
    coordinates = np.concatenate(
        (np.stack((axis, np.zeros(6), np.zeros(6)), axis=-1), ring)
    )
    cells = [
        [first, first + 1, 6 + index, 6 + (index + 1) % 6]
        for index in range(6)
        for first in range(5)
    ]
    state = _state(_mesh_result(coordinates, cells), np.eye(3))
    snapshot = state.snapshot
    vertex_ids = np.asarray(snapshot.vertex_ids, dtype=np.int64)
    cell_vertices = vertex_ids[np.asarray(snapshot.tetrahedra, dtype=np.int64)]
    cell_ids = np.asarray(snapshot.cell_ids, dtype=np.int64)
    groups = _facet_group_map(state, snapshot)
    pairs = ((1, 2), (4, 3))
    survivor = dict(pairs)
    removed_cells: list[int] = []
    parents: list[int] = []
    replacements: list[list[int]] = []
    for identifier, vertices in zip(
        cell_ids.tolist(), cell_vertices.tolist(), strict=True
    ):
        removed = [vertex for vertex in vertices if vertex in survivor]
        if not removed:
            continue
        removed_cells.append(identifier)
        if survivor[removed[0]] in vertices:
            continue
        parents.append(identifier)
        replacements.append([survivor.get(vertex, vertex) for vertex in vertices])
    facet_sources = np.asarray(
        [
            [
                groups.get(tuple(sorted(cell[corner] for corner in face)), -1)
                for face in ((1, 3, 2), (0, 2, 3), (0, 3, 1), (0, 1, 2))
            ]
            for cell in replacements
        ],
        dtype=np.int32,
    )
    count = len(replacements)
    pair_bank = np.asarray(pairs, dtype=np.int32, order="F" if layout == "F" else "C")
    if layout == "strided":
        padded = np.zeros((4, 4), dtype=np.int32)
        padded[::2, ::2] = pair_bank
        pair_bank = padded[::2, ::2]
    edit = TetMeshEdit(
        kind=TetMeshEditKind.COLLAPSE,
        expected_generation=int(snapshot.generation),
        source_id=state.source_identity[0],
        source_revision=state.source_identity[1],
        source_map_id=state.source_identity[2],
        frame_id=state.source_identity[3],
        collapse_pairs=pair_bank,
        removed_vertex_ids=np.asarray((1, 4), dtype=np.uint64) if explicit_removed else None,
        removed_cell_ids=np.asarray(sorted(removed_cells), dtype=np.uint64),
        replacement_cell_ancestry_offsets=np.arange(count + 1, dtype=np.int64),
        replacement_cell_ancestor_ids=np.asarray(parents, dtype=np.uint64),
        replacement_cell_ids=np.arange(
            state.next_native_cell, state.next_native_cell + count, dtype=np.uint64
        ),
        replacement_cell_vertices=np.asarray(replacements, dtype=np.uint64),
        replacement_cell_facet_sources=facet_sources,
        replacement_regions=np.zeros((count,), dtype=np.int32),
    )

    assert state.native.apply_edit(edit) is TetMeshEditStatus.APPLIED
    live = set(np.asarray(state.native.arrays().vertex_ids).tolist())
    assert {1, 4}.isdisjoint(live) and {0, 2, 3, 5} <= live
