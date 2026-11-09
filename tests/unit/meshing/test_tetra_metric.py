from __future__ import annotations

from fractions import Fraction
from itertools import combinations, permutations
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax._geometry_predicates import PredicateMode
from phydrax._meshcore import TetMeshEdit, TetMeshEditKind, TetMeshEditStatus
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._lineage import EntityLineageKind
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._tetra_metric import (
    _apply_collapse_batch,
    _apply_relocation_batch,
    _best_ring_triangulation,
    _boundary_records,
    _carrier_simplices,
    _collapse_metric_candidates,
    _collapse_phase,
    _entity_frame,
    _entity_weights,
    _facet_group_map,
    _locate_source_weights,
    _metric_enclosing_center,
    _metric_logarithms,
    _NativeState,
    _prepare_native_state,
    _quantized_relocation,
    _relocation_goals,
    _relocation_proposals,
    _replacement_candidates,
    _source_metric_from_stencil,
    _SourceSupport,
    _split_phase,
    _stencil_image,
    _topology,
    _unit_metric_status,
    execute_tetra_metric_adaptation,
    TetraMetricOutcome,
    TetraMetricStatus,
    TetraMetricUnmetKind,
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
        source.mesh,
        source.geometry,
        source.coordinate_contract,
        np.broadcast_to(metric, (mesh.vertex_global_ids.size, 3, 3)).copy(),
        cell_classes=np.zeros(mesh.entity_set(3).entity_ids.size, dtype=np.int64),
        facet_classes=np.zeros(mesh.entity_set(2).entity_ids.size, dtype=np.int64),
        edge_classes=np.zeros(edge_count, dtype=np.int64),
        protected_edges=np.zeros(edge_count, dtype=np.bool_),
        fixed_vertices=np.zeros(mesh.vertex_global_ids.size, dtype=np.bool_),
    )


def _counters() -> dict[str, int]:
    return dict.fromkeys(
        (
            "splits",
            "collapses",
            "relocations",
            "rejected_topology",
            "rejected_protected",
            "rejected_uncertain",
            "rejected_invalid",
        ),
        0,
    )


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


def test_tetra_metric_seeds_authoritative_exact_source_vertices() -> None:
    from phydrax.discretization._exact_plc_geometry import ExactPlcCellGeometrySource

    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    mesh = CellMesh(
        points, (CellBlock("tet", "tetrahedron", np.asarray([[0, 1, 2, 3]])),)
    )
    exact = tuple(tuple(Fraction(float(x)) for x in row) for row in points)
    exact = (
        exact[0],
        (Fraction(1) + Fraction(1, 2**54), Fraction(0), Fraction(0)),
        *exact[2:],
    )
    bank = ExactPlcCellGeometrySource(
        "authored-source",
        "revision",
        np.asarray(mesh.vertex_global_ids),
        exact,
        vertex_source_entity_ids=(None,) * 4,
        vertex_occurrence_paths=((),) * 4,
    )
    affine = phx.discretization.CellGeometrySpec.affine(mesh)
    geometry = phx.discretization.CellGeometrySpec(
        dict(zip(affine.block_names, affine.elements, strict=True)),
        dict(zip(affine.block_names, affine.geometry_dofs, strict=True)),
        mesh.coordinates,
        source_map=bank,
        source_mesh=mesh,
    )
    state = _prepare_native_state(
        mesh,
        geometry,
        phx.SpatialCoordinateContract.si(),
        np.broadcast_to(np.eye(3), (4, 3, 3)).copy(),
        cell_classes=np.zeros(mesh.entity_set(3).entity_ids.size, dtype=np.int64),
        facet_classes=np.zeros(mesh.entity_set(2).entity_ids.size, dtype=np.int64),
        edge_classes=np.zeros(mesh.entity_set(1).entity_ids.size, dtype=np.int64),
        protected_edges=np.zeros(mesh.entity_set(1).entity_ids.size, dtype=np.bool_),
        fixed_vertices=np.zeros(4, dtype=np.bool_),
    )
    assert state.source_exact_points == exact
    assert state.source_exact_points[1][0] != Fraction(float(state.source_points[1, 0]))
    located = _locate_source_weights(exact[1], state.source_records)
    assert located is not None
    assert located[2] == (Fraction(0), Fraction(1), Fraction(0), Fraction(0))


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
        source.mesh,
        source.geometry,
        source.coordinate_contract,
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

    # Disjoint collapses apply as one native edit whatever the order of their
    # removed vertices (pairs are aligned with the sorted removed IDs).
    groups = _collapse_metric_candidates(state, {})
    assert len(groups) >= 2
    batch = sorted(
        (proposal for group in groups[:2] for proposal in group),
        key=lambda proposal: -int(proposal["removed"]),
    )
    assert _apply_collapse_batch(state, batch) is TetMeshEditStatus.APPLIED
    live = set(np.asarray(state.snapshot.vertex_ids).tolist())
    assert all(int(proposal["removed"]) not in live for proposal in batch)


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


def _implicit_plane_pyramid(height: float, tolerance: float) -> _NativeState:
    from phydrax._meshcore import IntervalBoxProgram, TetMeshImplicitSource
    from phydrax.discretization._exact_plc_geometry import ExactPlcCellGeometrySource

    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.25, 0.25, height),
            (0.25, 0.25, 0.0),
        ),
        dtype=np.float64,
    )
    cells = [[0, 1, 4, 3], [1, 2, 4, 3], [2, 0, 4, 3]]
    source = _mesh_result(points, cells)
    exact = tuple(tuple(Fraction(float(value)) for value in point) for point in points)
    bank = ExactPlcCellGeometrySource(
        "plane",
        "revision",
        np.asarray(source.mesh.vertex_global_ids),
        exact,
        vertex_source_entity_ids=(None,) * len(points),
        vertex_occurrence_paths=((),) * len(points),
    )
    affine = phx.discretization.CellGeometrySpec.affine(source.mesh)
    geometry = phx.discretization.CellGeometrySpec(
        dict(zip(affine.block_names, affine.elements, strict=True)),
        dict(zip(affine.block_names, affine.geometry_dofs, strict=True)),
        source.mesh.coordinates,
        source_map=bank,
        source_mesh=source.mesh,
    )
    source = certify_cell_mesh(source.mesh, source.coordinate_contract, geometry=geometry)
    state = _state(source, np.eye(3, dtype=np.float64))
    records = _boundary_records(state)
    groups = {records.facets[key][1] for key in records.vertex_facets[4]}
    assert len(groups) == 1
    group = next(iter(groups))
    value = IntervalBoxProgram(
        np.asarray([[2, 4, 1, 2, -1, 3, -1, 0]], dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64),
        np.zeros(5, dtype=np.float64),
        np.zeros(5, dtype=np.float64),
        np.asarray([4], dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )
    gradient = IntervalBoxProgram(
        np.empty((0, 8), dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64),
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        np.asarray([3, 4, 5], dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )
    state.native.bind_implicit_source(
        TetMeshImplicitSource(
            value,
            gradient,
            np.asarray([group], dtype=np.int32),
            tolerance,
            0,
            "plane",
            "revision",
            bank.source_map_id,
            source.coordinate_contract.reference_frame,
        )
    )
    state.snapshot = state.native.arrays()
    state.implicit_groups = frozenset((group,))
    return state


def test_implicit_boundary_survivor_proposals_keep_exact_source_stencils() -> None:
    from phydrax.meshing._tetra_metric import _implicit_only_vertex, _survivor_moves

    state = _implicit_plane_pyramid(1.0, 0.01)
    points = state.source_points
    group = next(iter(state.implicit_groups))
    try:
        records = _boundary_records(state)
        assert _implicit_only_vertex(state, records, 4)
        topology = _topology(state.snapshot)
        local = points[[4, 0, 1, 2, 3]]
        moves = list(
            _survivor_moves(
                state, records, topology, 4, local, np.broadcast_to(np.eye(3), (5, 3, 3))
            )
        )
        assert moves
        for move in moves:
            exact = _stencil_image(state, move["stencil"])
            np.testing.assert_array_equal(
                move["point"], [float(value) for value in exact]
            )
        state.native.protect_facet_group_rows(np.asarray([group], dtype=np.int32))
        state.snapshot = state.native.arrays()
        assert not _implicit_only_vertex(state, _boundary_records(state), 4)
    finally:
        state.native.close()


def test_implicit_relocation_continues_after_native_distance_refusal() -> None:
    from phydrax.meshing._tetra_metric import _relocation_phase

    # The tall apex pulls the base vertex out of its accepted zero-set tube.
    # Smaller existing goals improve the metric while staying inside the tube.
    tolerance = 0.01
    state = _implicit_plane_pyramid(2.0, tolerance)
    try:
        counters = _counters()
        _relocation_phase(state, counters, topology_operations=False)
        vertex_ids = np.asarray(state.snapshot.vertex_ids)
        moved = np.asarray(state.snapshot.points)[np.flatnonzero(vertex_ids == 4)[0]]
        assert counters["rejected_uncertain"] > 0
        refusals = state.native.implicit_refusals
        assert refusals is not None
        assert refusals[1] > 0
        assert 0.0 < moved[2] <= tolerance
        assert np.linalg.norm(moved - state.source_points[3]) < np.linalg.norm(
            state.source_points[4] - state.source_points[3]
        )
        np.testing.assert_array_equal(
            moved,
            [float(value) for value in _stencil_image(state, state.vertex_stencils[4])],
        )
    finally:
        state.native.close()


def _relocation_cache_witness_state() -> _NativeState:
    import phydrax.meshing._tetra_metric as metric
    from phydrax._meshcore import IntervalBoxProgram
    from phydrax.discretization._exact_plc_geometry import ExactPlcCellGeometrySource

    # Numeric initial carrier, metric and sphere interval programs from the
    # native sphere diagnostic (radius .371, fidelity .005); no checkpoint or
    # accepted edit is embedded. Reconstruct a new exact test-source identity.
    with np.load(Path(__file__).with_name("relocation_cache_sphere.npz")) as fixture:
        points = fixture["points"].copy()
        tetrahedra = fixture["tetrahedra"].tolist()
        tensors = fixture["metric"].copy()
        programs = [
            IntervalBoxProgram(
                *(
                    fixture[f"{name}_{field}"].copy()
                    for field in IntervalBoxProgram._fields
                )
            )
            for name in ("value", "gradient")
        ]
    source = _mesh_result(points, tetrahedra)
    bank = ExactPlcCellGeometrySource(
        "relocation-cache-witness",
        "1",
        np.asarray(source.mesh.vertex_global_ids),
        tuple(tuple(Fraction(float(value)) for value in point) for point in points),
        vertex_source_entity_ids=(None,) * len(points),
        vertex_occurrence_paths=((),) * len(points),
    )
    affine = phx.discretization.CellGeometrySpec.affine(source.mesh)
    geometry = phx.discretization.CellGeometrySpec(
        dict(zip(affine.block_names, affine.elements, strict=True)),
        dict(zip(affine.block_names, affine.geometry_dofs, strict=True)),
        source.mesh.coordinates,
        source_map=bank,
        source_mesh=source.mesh,
    )
    source = certify_cell_mesh(source.mesh, source.coordinate_contract, geometry=geometry)
    mesh = source.mesh
    implicit = metric._ImplicitMetricSource(
        programs[0],
        programs[1],
        0.005,
        bank.source_id,
        bank.source_revision,
        bank.source_map_id,
        source.coordinate_contract.reference_frame,
    )
    return _prepare_native_state(
        mesh,
        geometry,
        source.coordinate_contract,
        tensors,
        cell_classes=np.zeros(mesh.entity_set(3).entity_ids.size, dtype=np.int64),
        facet_classes=np.zeros(mesh.entity_set(2).entity_ids.size, dtype=np.int64),
        edge_classes=np.zeros(mesh.entity_set(1).entity_ids.size, dtype=np.int64),
        protected_edges=np.zeros(mesh.entity_set(1).entity_ids.size, dtype=np.bool_),
        fixed_vertices=np.zeros(points.shape[0], dtype=np.bool_),
        implicit_source=implicit,
    )


def test_native_implicit_tangent_relocation_reduces_band() -> None:
    import phydrax.meshing._tetra_metric as metric
    from phydrax._meshcore import PlcBoundaryPolicy, TetMesh3D, TetMeshSourceComplex

    # Terminal sphere carrier from the native diagnostic, with its original
    # numeric source hull, metric and interval programs from the adjacent
    # fixture. This is a fresh test source and native owner, not a checkpoint.
    # Only vertex 39 is movable; the unrelated terminal vertex lineages are
    # not reconstructed or used by this private relocation search.
    state = _relocation_cache_witness_state()
    initial = state.snapshot
    with np.load(Path(__file__).with_name("relocation_tangent_sphere.npz")) as data:
        points = data["points"].copy()
        cells = data["tetrahedra"].copy()
        faces = data["faces"].copy()
        vertex_ids = data["vertex_ids"].copy()
        cell_ids = data["cell_ids"].copy()
    count = len(vertex_ids)
    source = TetMeshSourceComplex(
        initial.source_points,
        initial.source_faces,
        initial.source_face_group_rows,
        initial.source_face_tolerances,
        initial.source_segments,
        initial.source_segment_group_rows,
        initial.source_segment_tolerances,
        np.zeros(count, dtype=np.int8),
        np.full(count, -1, dtype=np.int32),
        np.zeros((count, 2), dtype=np.float64),
        initial.source_face_ids,
        initial.source_segment_ids,
        initial.source_face_planes,
        initial.source_face_plane_tolerances,
    )
    implicit = initial.implicit_source
    assert implicit is not None
    try:
        native = TetMesh3D(
            points,
            cells,
            np.zeros(len(cells), dtype=np.int32),
            faces,
            np.full(len(faces), initial.source_face_ids[0], dtype=np.int64),
            np.empty((0, 2), dtype=np.int32),
            np.empty(0, dtype=np.int64),
            boundary_policy=PlcBoundaryPolicy.CONFORMING,
            protection_radii=np.zeros(count, dtype=np.float64),
            protected_vertices=(vertex_ids != 39).astype(np.int8),
            vertex_ids=vertex_ids,
            cell_ids=cell_ids,
            source=source,
            implicit_source=implicit.source,
            source_id=state.source_identity[0],
            source_revision=state.source_identity[1],
            source_map_id=state.source_identity[2],
            frame_id=state.source_identity[3],
        )
        state.native.close()
        state.native = native
        state.snapshot = native.arrays()
        state.tables = None
        state.fixed_vertices = set(map(int, vertex_ids)) - {39}
        state.vertex_target_ids = {int(vertex): int(vertex) for vertex in vertex_ids}
        state.cell_blocks = {int(cell): 0 for cell in cell_ids}
        state.metric_by_vertex = {
            int(vertex): state.source_metric[0].copy() for vertex in vertex_ids
        }
        state.vertex_stencils = {
            39: {22: Fraction(141, 1024), 23: Fraction(139, 1024), 39: Fraction(93, 128)}
        }
        topology = metric._state_topology(state)
        row = topology.vertex_row[39]
        before_point = np.asarray(state.snapshot.points)[row].copy()
        np.testing.assert_array_equal(
            before_point,
            [float(value) for value in _stencil_image(state, state.vertex_stencils[39])],
        )
        _, _, lengths = metric._metric_edge_table(state, topology)
        incident = [39 in edge for edge in topology.edge_cells]
        before_band = metric._band_violations(lengths[incident])
        cache: dict[int, metric._RelocationSearch | None] = {}
        while True:
            groups = metric._relocation_proposals(
                state, cache, deep=True, active_only=True, band_only=True
            )
            assert groups, "No native-admissible band-reducing relocation"
            proposal = groups[0][0]
            stencil = proposal["stencil"]
            assert all(weight > 0 for weight in stencil.values())
            assert sum(stencil.values()) == 1
            np.testing.assert_array_equal(
                proposal["point"],
                [float(value) for value in _stencil_image(state, stencil)],
            )
            status = native.apply_edit(metric._relocation_edit(state, [proposal]))
            if status is TetMeshEditStatus.APPLIED:
                break
            metric._raise_fatal_edit_status(status, "relocation")
            search = cache[39]
            assert search is not None
            search.proposal = None
        state.snapshot = native.arrays()
        state.tables = None
        topology = metric._state_topology(state)
        after_point = np.asarray(state.snapshot.points)[topology.vertex_row[39]]
        assert not np.array_equal(after_point, before_point)
        _, _, lengths = metric._metric_edge_table(state, topology)
        incident = [39 in edge for edge in topology.edge_cells]
        assert metric._band_violations(lengths[incident]) < before_band
    finally:
        state.native.close()


def test_native_refusal_search_survives_collapse_unblock_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dataclasses
    import hashlib
    import pickle

    import phydrax.meshing._tetra_metric as metric
    from phydrax._meshcore import TetMesh3D

    state = _relocation_cache_witness_state()
    native_apply = TetMesh3D.apply_edit
    relocation_apply = metric._apply_relocation_batch
    proposals = metric._relocation_proposals
    unblocks_collapse = metric._relocation_unblocks_collapse
    stage: tuple[bool, ...] = ()
    # A topology edit may legitimately invalidate every cache, even on a
    # geometrically unchanged star; compare only within one topology epoch.
    seen: dict[tuple[Any, ...], tuple[int, int, tuple[int, ...], bytes]] = {}
    moved = 0
    unblocks = 0
    witness: tuple[tuple[int, ...], bytes] | None = None

    class PrefixComplete(Exception):
        pass

    def star_points(vertices: tuple[int, ...]) -> bytes:
        rows = {int(vertex): row for row, vertex in enumerate(state.snapshot.vertex_ids)}
        return np.asarray(state.snapshot.points)[[rows[v] for v in vertices]].tobytes()

    def observe_native(owner: TetMesh3D, edit: TetMeshEdit) -> TetMeshEditStatus:
        nonlocal witness
        result = native_apply(owner, edit)
        if (
            result is TetMeshEditStatus.APPLIED
            and edit.kind is not TetMeshEditKind.RELOCATE
        ):
            seen.clear()
            witness = None
        return result

    def observe_proposals(
        owner: _NativeState,
        cache: dict[int, metric._RelocationSearch | None],
        /,
        *,
        deep: bool = False,
        refine: bool = False,
        active_only: bool = False,
        band_only: bool = False,
        prepared: metric._RelocationPreparation | None = None,
    ) -> list[tuple[dict[str, Any], ...]]:
        nonlocal stage
        stage = (deep, refine, active_only, band_only)
        return proposals(
            owner,
            cache,
            deep=deep,
            refine=refine,
            active_only=active_only,
            band_only=band_only,
            prepared=prepared,
        )

    def observe_relocation(
        owner: _NativeState, batch: list[dict[str, Any]]
    ) -> TetMeshEditStatus:
        nonlocal moved
        key = None
        vertices: tuple[int, ...] = ()
        coordinates = b""
        if len(batch) == 1:
            proposal = batch[0]
            vertices = tuple(sorted(map(int, proposal["star_vertices"])))
            coordinates = star_points(vertices)
            edit = dataclasses.replace(
                metric._relocation_edit(owner, batch), expected_generation=0
            )
            digest = hashlib.sha256(pickle.dumps(edit, protocol=5)).digest()
            key = (
                stage,
                digest,
                vertices,
                coordinates,
                tuple(sorted(proposal["star_cell_ids"])),
            )
        before = owner.native.implicit_refusals
        result = relocation_apply(owner, batch)
        after = owner.native.implicit_refusals
        assert before is not None and after is not None
        if result is TetMeshEditStatus.APPLIED:
            moved += len(batch)
        elif key is not None and after[1] > before[1]:
            # Only singleton native refusals count; partition retries of a
            # larger failed batch are different native requests.
            previous = seen.get(key)
            assert previous is None or previous[0] == unblocks, (
                "An unchanged star restarted its refused exact relocation after a yield"
            )
            seen.setdefault(key, (unblocks, moved, vertices, coordinates))
            if witness is not None and len(seen) >= 25:
                raise PrefixComplete
        return result

    def observe_unblock(
        owner: _NativeState,
        moved_proposals: list[dict[str, Any]],
        cache: dict[tuple[int, int], tuple[dict[str, Any] | None, frozenset[int]]],
        prepared: metric._RelocationPreparation,
        /,
    ) -> bool:
        nonlocal unblocks, witness
        result = unblocks_collapse(owner, moved_proposals, cache, prepared)
        if result:
            unblocks += 1
            for _, earlier_moves, vertices, coordinates in seen.values():
                if earlier_moves < moved and star_points(vertices) == coordinates:
                    witness = (vertices, coordinates)
                    break
        return result

    monkeypatch.setattr(TetMesh3D, "apply_edit", observe_native)
    monkeypatch.setattr(metric, "_relocation_proposals", observe_proposals)
    monkeypatch.setattr(metric, "_apply_relocation_batch", observe_relocation)
    monkeypatch.setattr(metric, "_relocation_unblocks_collapse", observe_unblock)
    try:
        # Stop at a finite observed prefix, not a claim of mesh convergence.
        with pytest.raises(PrefixComplete):
            metric._execute_native_state(
                state, PredicateMode.EXACT, topology_operations=True, relocation=True
            )
        assert witness is not None
        assert unblocks > 0 and moved > 0
        refusals = state.native.implicit_refusals
        assert refusals is not None and refusals[1] > 0
        native_snapshot = state.native.arrays()
        np.testing.assert_array_equal(native_snapshot.points, state.snapshot.points)
        np.testing.assert_array_equal(
            native_snapshot.tetrahedra, state.snapshot.tetrahedra
        )
        assert metric._unit_metric_status(state)[4] > 0.0
    finally:
        state.native.close()


@pytest.mark.parametrize("restored", (False, True))
def test_implicit_collapse_admits_survivor_off_original_facet(
    tmp_path: Path, restored: bool
) -> None:
    from phydrax._meshcore import IntervalBoxProgram, TetMeshImplicitSource
    from phydrax.discretization._exact_plc_geometry import ExactPlcCellGeometrySource
    from phydrax.meshing._tetra_metric import (
        _current_edge_controls,
        _directed_collapse,
        _source_permits_collapse,
        _state_topology,
    )

    grid = _source(4)
    points = np.asarray(grid.mesh.coordinates).copy()
    points[points[:, 2] == 1.0 / 3.0, 2] = 0.005
    if restored:
        points += np.asarray((1.0, -2.0, 0.0))
    source = _mesh_result(points, np.asarray(grid.mesh.blocks[0].vertices).tolist())
    bank = ExactPlcCellGeometrySource(
        "plane",
        "revision",
        np.asarray(source.mesh.vertex_global_ids),
        tuple(tuple(Fraction(float(value)) for value in point) for point in points),
        vertex_source_entity_ids=(None,) * len(points),
        vertex_occurrence_paths=((),) * len(points),
    )
    affine = phx.discretization.CellGeometrySpec.affine(source.mesh)
    geometry = phx.discretization.CellGeometrySpec(
        dict(zip(affine.block_names, affine.elements, strict=True)),
        dict(zip(affine.block_names, affine.geometry_dofs, strict=True)),
        source.mesh.coordinates,
        source_map=bank,
        source_mesh=source.mesh,
    )
    source = certify_cell_mesh(source.mesh, source.coordinate_contract, geometry=geometry)
    if restored:
        from phydrax.meshing._model_codec import (
            read_model_artifact,
            restore_model_archive_value,
            write_model_artifact,
        )

        path = write_model_artifact(tmp_path / "source.phx", source, limits=None)
        source = read_model_artifact(
            path,
            expected_type=CellMeshingResult,
            restore=restore_model_archive_value,
            limits=None,
        )
    state = _state(source, np.eye(3))
    removed, kept = 20, 24
    records = _boundary_records(state)
    groups = {records.facets[key][1] for key in records.vertex_facets[removed]}
    assert len(groups) == 1
    group = next(iter(groups))
    value = IntervalBoxProgram(
        np.asarray([[2, 4, 1, 2, -1, 3, -1, 0]], dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0),
        np.zeros(5),
        np.zeros(5),
        np.asarray([4]),
        np.empty(0, dtype=np.int64),
    )
    gradient = IntervalBoxProgram(
        np.empty((0, 8), dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0),
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        np.asarray([3, 4, 5]),
        np.empty(0, dtype=np.int64),
    )
    try:
        state.native.bind_implicit_source(
            TetMeshImplicitSource(
                value,
                gradient,
                np.asarray([group], dtype=np.int32),
                0.01,
                0,
                "plane",
                "revision",
                bank.source_map_id,
                source.coordinate_contract.reference_frame,
            )
        )
        state.snapshot = state.native.arrays()
        state.implicit_groups = frozenset((group,))
        topology = _state_topology(state)
        cell_ids = np.asarray(state.snapshot.cell_ids, dtype=np.int64)
        stencil = {kept: Fraction(1, 2), kept + 1: Fraction(1, 2)}
        move = {
            "vertex": kept,
            "point": np.asarray(
                [float(value) for value in _stencil_image(state, stencil)]
            ),
            "metric": np.eye(3),
            "stencil": stencil,
            "star_cell_ids": tuple(
                int(cell_ids[row]) for row in topology.vertex_cells[kept]
            ),
        }
        assert _apply_relocation_batch(state, [move]) is TetMeshEditStatus.APPLIED
        records = _boundary_records(state)
        topology = _state_topology(state)
        assert not _source_permits_collapse(state, topology, records, removed, 21)
        assert not set(stencil) <= state.source_face_vertices[group]
        state.implicit_groups = frozenset()
        assert not _source_permits_collapse(state, topology, records, removed, kept)
        state.implicit_groups = frozenset((group,))
        assert _source_permits_collapse(state, topology, records, removed, kept)
        _, protected = _current_edge_controls(state, topology)
        proposal = _directed_collapse(state, topology, records, protected, removed, kept)
        assert proposal is not None
        assert _apply_collapse_batch(state, [proposal]) is TetMeshEditStatus.APPLIED
    finally:
        state.native.close()


def test_collapse_moves_its_survivor_to_the_collapsed_edge_midpoint() -> None:
    # Interior vertices r, k on the axis of a double cone over a hexagon: the
    # short edge rk (0.6) collapses in place only by creating a long spoke
    # (1.6 to the far apex), which the objective refuses in both directions.
    # Collapsing to the midpoint (the origin) leaves every spoke in the band.
    angles = np.arange(6) * (np.pi / 3.0)
    ring = np.stack((np.zeros(6), 0.9 * np.cos(angles), 0.9 * np.sin(angles)), axis=-1)
    coordinates = np.concatenate(
        (
            np.asarray(
                ((-1.3, 0.0, 0.0), (-0.3, 0.0, 0.0), (0.3, 0.0, 0.0), (1.3, 0.0, 0.0))
            ),
            ring,
        )
    )
    cells = [
        [first, second, 4 + index, 4 + (index + 1) % 6]
        for index in range(6)
        for first, second in ((0, 1), (1, 2), (2, 3))
    ]
    state = _state(_mesh_result(coordinates, cells), np.eye(3))
    counters = _counters()

    assert _collapse_phase(state, counters) == counters["collapses"] == 1
    assert counters["rejected_invalid"] == counters["rejected_topology"] == 0
    snapshot = state.snapshot
    survivors = {1, 2} & set(np.asarray(snapshot.vertex_ids).tolist())
    assert len(survivors) == 1
    survivor = survivors.pop()
    row = np.asarray(snapshot.vertex_ids).tolist().index(survivor)
    np.testing.assert_array_equal(np.asarray(snapshot.points)[row], np.zeros(3))
    assert state.vertex_stencils[survivor] == {1: Fraction(1, 2), 2: Fraction(1, 2)}
    removed = 3 - survivor
    assert (removed, survivor, EntityLineageKind.COLLAPSED_INTO) in state.history_vertex
    assert (survivor, survivor, EntityLineageKind.RELOCATED) in state.history_vertex
    assert not any(
        kind is EntityLineageKind.UNKNOWN for _, _, kind in state.history_vertex
    )
    assert len(snapshot.tetrahedra) == 12
    converged, outside, shortest, longest, quality = _unit_metric_status(state)
    # Only the hull spokes apex-ring (1.58) stay long; the merged star is in band.
    assert shortest >= 1.0 / np.sqrt(2.0) and quality > 0.0
    assert not converged and outside == 12 and longest > np.sqrt(2.0)
    stencil = {1: Fraction(1, 4), 2: Fraction(3, 4)}
    point = np.asarray(
        [
            float(
                sum(
                    (
                        weight * Fraction(float(coordinates[vertex, axis]))
                        for vertex, weight in stencil.items()
                    ),
                    Fraction(0),
                )
            )
            for axis in range(3)
        ]
    )
    proposal = {
        "vertex": survivor,
        "point": point,
        "metric": np.eye(3),
        "stencil": stencil,
        "star_cell_ids": tuple(int(value) for value in state.snapshot.cell_ids),
    }
    assert _apply_relocation_batch(state, [proposal]) is TetMeshEditStatus.APPLIED
    assert (removed, survivor, EntityLineageKind.COLLAPSED_INTO) in state.history_vertex
    assert (survivor, survivor, EntityLineageKind.RELOCATED) in state.history_vertex
    row = np.asarray(state.snapshot.vertex_ids).tolist().index(survivor)
    np.testing.assert_array_equal(np.asarray(state.snapshot.points)[row], point)
    assert state.vertex_stencils[survivor] == stencil
    state.fixed_vertices.add(survivor)
    cache = {
        int(identifier): None
        for identifier in np.asarray(state.snapshot.vertex_ids)
        if int(identifier) != survivor
    }
    assert not _relocation_proposals(state, cache)


def test_relocation_backtracks_to_binary64_resolution() -> None:
    point = np.ones(3)
    goals = list(_relocation_goals(point, np.asarray(((1.0, 0.0, 0.0),))))
    assert len(goals) > 6
    np.testing.assert_array_equal(goals[12], point + np.asarray((2.0**-12, 0.0, 0.0)))
    assert not np.array_equal(goals[-1], point)
    np.testing.assert_array_equal(point + (goals[-1] - point) / 2.0, point)


def test_relocation_can_refine_coarse_source_carriers() -> None:
    from phydrax.meshing._tetra_metric import _quantized_relocation_candidates

    state = _state(_source(), np.eye(3))
    current = np.asarray((0.05625916, 0.68115234, 0.2355957))
    goal = current + np.asarray((5.03073591e-05, 0.00612579696, 0.00986193289))
    coarse = _quantized_relocation(state, None, goal, current)
    fine = next(
        _quantized_relocation_candidates(state, None, goal, current, fine_first=True)
    )
    assert coarse is not None
    assert np.linalg.norm(fine[1] - goal) < np.linalg.norm(coarse[1] - goal)
    assert fine[0] != coarse[0]


def test_deep_boundary_relocation_keeps_affine_projection() -> None:
    state = _state(_source(4, nonuniform=True), np.eye(3) * 4.0)
    proposals = _relocation_proposals(
        state, {}, deep=True, active_only=True, band_only=True
    )
    assert proposals


def test_active_relocation_prepares_only_violating_vertex_stars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.meshing._tetra_metric as metric

    state = _state(_source(4, nonuniform=True), np.eye(3) * 4.0)
    prepared = metric._prepare_relocation_geometry(state)
    active = {
        vertex
        for edge, length in zip(
            prepared.topology.edge_cells, prepared.lengths, strict=True
        )
        if not prepared.protected[edge]
        and (length < metric._LOWER or length > metric._UPPER)
        for vertex in edge
    }
    cache: dict[int, Any] = {}

    def unexpected_orbits(_state: Any) -> Any:
        raise AssertionError("Nonperiodic proposals need no global orbit enumeration.")

    monkeypatch.setattr(metric, "_periodic_vertex_orbit_groups", unexpected_orbits)
    proposals = metric._relocation_proposals(
        state, cache, active_only=True, band_only=True, prepared=prepared
    )
    assert proposals and cache.keys() <= active


def test_relocation_preparation_reuses_only_identical_incidence() -> None:
    import phydrax.meshing._tetra_metric as metric

    state = _state(_source(), np.eye(3) * 4.0)
    original = state.snapshot
    prepared = metric._prepare_relocation_geometry(state)
    state.snapshot = original._replace(points=original.points.copy())
    unchanged = metric._prepare_relocation_geometry(state, previous=(original, prepared))
    assert unchanged.topology is prepared.topology
    assert unchanged.classes is prepared.classes
    assert unchanged.protected is prepared.protected
    assert unchanged.records is prepared.records
    np.testing.assert_array_equal(unchanged.lengths, prepared.lengths)

    vertex = int(original.vertex_ids[0])
    points = original.points.copy()
    points[0] += (0.01, 0.02, 0.03)
    state.snapshot = original._replace(points=points)
    old_metric = state.metric_by_vertex[vertex]
    state.metric_by_vertex[vertex] = old_metric * 1.2
    updated = metric._prepare_relocation_geometry(
        state, previous=(original, prepared), moved_vertices=frozenset((vertex,))
    )
    expected_points, expected_metrics, expected_lengths = metric._metric_edge_table(
        state, metric._topology(state.snapshot)
    )
    np.testing.assert_array_equal(updated.points, expected_points)
    np.testing.assert_array_equal(updated.metrics, expected_metrics)
    np.testing.assert_array_equal(updated.lengths, expected_lengths)
    state.metric_by_vertex[vertex] = old_metric

    state.snapshot = original._replace(
        tetrahedra=original.tetrahedra[::-1].copy(),
        cell_ids=original.cell_ids[::-1].copy(),
        tetrahedron_regions=original.tetrahedron_regions[::-1].copy(),
    )
    changed = metric._prepare_relocation_geometry(state, previous=(original, prepared))
    assert changed.topology is not prepared.topology
    np.testing.assert_array_equal(
        changed.topology.cell_vertex_ids,
        metric._topology(state.snapshot).cell_vertex_ids,
    )


def test_relocation_reuses_identical_projected_goals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.meshing._tetra_metric as metric

    source = _source(4, nonuniform=True)
    coordinates = np.asarray(source.mesh.coordinates)
    tensors = (4.0 + coordinates[:, 0])[:, None, None] * np.eye(3)
    state = _state(source, tensors)
    original_quantize = metric._quantized_relocation
    original_candidates = metric._quantized_relocation_candidates
    original_goals = metric._relocation_goals
    calls = 0

    def quantize(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original_quantize(*args, **kwargs)

    def candidates(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original_candidates(*args, **kwargs)

    monkeypatch.setattr(metric, "_quantized_relocation", quantize)
    monkeypatch.setattr(metric, "_quantized_relocation_candidates", candidates)
    baseline = metric._relocation_proposals(
        state, {}, deep=True, refine=True, active_only=True, band_only=True
    )
    baseline_calls = calls
    assert baseline and baseline_calls

    def duplicated(*args: Any, **kwargs: Any) -> Any:
        for goal in original_goals(*args, **kwargs):
            yield goal
            yield goal.copy()

    monkeypatch.setattr(metric, "_relocation_goals", duplicated)
    calls = 0
    repeated = metric._relocation_proposals(
        state, {}, deep=True, refine=True, active_only=True, band_only=True
    )
    assert calls == baseline_calls
    assert len(repeated) == len(baseline)
    for expected, actual in zip(baseline, repeated, strict=True):
        assert len(expected) == len(actual)
        for first, second in zip(expected, actual, strict=True):
            assert first["vertex"] == second["vertex"]
            assert first["stencil"] == second["stencil"]
            np.testing.assert_array_equal(first["point"], second["point"])
            np.testing.assert_array_equal(first["metric"], second["metric"])


def test_relocation_reuses_rejected_exact_stencils(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.meshing._tetra_metric as metric

    source = _source(4, nonuniform=True)
    coordinates = np.asarray(source.mesh.coordinates)
    tensors = (4.0 + coordinates[:, 0])[:, None, None] * np.eye(3)
    state = _state(source, tensors)
    original_quantize = metric._quantized_relocation
    original_interpolate = metric._source_metric_from_stencil
    first: dict[bytes, Any] = {}
    current = b""
    quantizations = 0
    interpolations: dict[bytes, int] = {}

    def quantize(
        native: Any, simplices: Any, goal: np.ndarray, point: np.ndarray, **kwargs: Any
    ) -> Any:
        nonlocal current, quantizations
        current = point.tobytes()
        quantizations += 1
        if current not in first:
            result = original_quantize(native, simplices, goal, point, **kwargs)
            if result is None:
                return None
            first[current] = result
        return first[current]

    def interpolate(*args: Any, **kwargs: Any) -> Any:
        interpolations[current] = interpolations.get(current, 0) + 1
        return original_interpolate(*args, **kwargs)

    monkeypatch.setattr(metric, "_quantized_relocation", quantize)
    monkeypatch.setattr(metric, "_source_metric_from_stencil", interpolate)
    monkeypatch.setattr(metric, "_band_violations", lambda _lengths: ((), ()))
    proposals = metric._relocation_proposals(
        state, {}, deep=False, active_only=False, band_only=True
    )
    assert not proposals
    assert interpolations and quantizations > len(interpolations)
    assert all(count == 1 for count in interpolations.values())


def test_relocation_moves_boundary_vertices_within_their_facet_and_edge() -> None:
    source = _source(4, nonuniform=True)
    outcome = _adapt(source, np.eye(3) * 4.0, topology_operations=False, relocation=True)

    # Quantized relocation targets are representable source points: the native
    # mesh refuses none of them.
    assert outcome.evidence.relocations > 0
    assert outcome.evidence.rejected_invalid == 0
    stencil_weights = np.asarray(outcome.edit.stencil_weights, dtype=np.float64)
    stencil_valid = np.asarray(outcome.edit.stencil_valid, dtype=np.bool_)
    for weight in stencil_weights[stencil_valid].tolist():
        assert Fraction(weight).denominator.bit_count() == 1
    source_points = {
        int(identifier): np.asarray(source.mesh.coordinates[row])
        for row, identifier in enumerate(source.mesh.vertex_global_ids.tolist())
    }
    moved_in_facet = moved_on_edge = 0
    for identifier, point in zip(
        np.asarray(outcome.edit.vertex_global_ids).tolist(),
        np.asarray(outcome.edit.coordinates),
        strict=True,
    ):
        original = source_points[int(identifier)]
        on_boundary = (original == 0.0) | (original == 1.0)
        # Boundary coordinates are kept exactly: a vertex stays on its facet
        # plane(s); a corner does not move.
        np.testing.assert_array_equal(point[on_boundary], original[on_boundary])
        moved = not np.array_equal(point, original)
        boundary_count = int(np.count_nonzero(on_boundary))
        moved_in_facet += int(moved and boundary_count == 1)
        moved_on_edge += int(moved and boundary_count == 2)
        assert not (moved and boundary_count == 3)
    assert moved_in_facet > 0 and moved_on_edge > 0


def test_declared_plane_admits_a_rounded_relocation_on_a_slanted_facet() -> None:
    # The slanted facet x + y + z = a of a corner tetrahedron, split at V into
    # three coplanar triangles. With a = 1 + 2**-50 a dyadic point of the facet
    # needs more than 53 bits: its carrier rounds off the exact plane, and only
    # the declared plane (rounding-box admission) lets V move within the facet.
    a = 1.0 + 2.0**-50
    coordinates = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (a, 0.0, 0.0),
            (0.0, a, 0.0),
            (0.0, 0.0, a),
            (a / 4.0, a / 4.0, a / 2.0),
        )
    )
    state = _state(
        _mesh_result(coordinates, [[0, 1, 2, 4], [0, 2, 3, 4], [0, 3, 1, 4]]),
        np.eye(3),
    )
    records = _boundary_records(state)
    simplices = _carrier_simplices(state, records, 4)
    assert simplices is not None and len(simplices) == 3
    current = coordinates[4]
    centroid = np.full(3, a / 3.0)
    goal = current + 0.5 * (centroid - current)
    quantized = _quantized_relocation(state, simplices, goal, current)
    assert quantized is not None
    stencil, carrier = quantized
    frames: dict[tuple[int, ...], _SourceSupport] = {}
    images: dict[Any, Any] = {}
    for factor in (0.5, 0.25, 0.125):
        trial = current + factor * (centroid - current)
        uncached = _quantized_relocation(state, simplices, trial, current)
        cached = _quantized_relocation(
            state, simplices, trial, current, frames=frames, images=images
        )
        assert uncached is not None and cached is not None
        assert cached[0] == uncached[0]
        np.testing.assert_array_equal(cached[1], uncached[1])
        repeated = _quantized_relocation(
            state, simplices, trial, current, frames=frames, images=images
        )
        assert repeated is not None and repeated[0] == cached[0]
        np.testing.assert_array_equal(repeated[1], cached[1])
    assert frames
    assert images
    assert np.linalg.norm(carrier - goal) < np.linalg.norm(current - goal)
    exact = [
        sum(
            (
                weight * Fraction(float(coordinates[vertex, axis]))
                for vertex, weight in stencil.items()
            ),
            Fraction(0),
        )
        for axis in range(3)
    ]
    assert sum(exact, Fraction(0)) == Fraction(a)
    assert sum((Fraction(float(value)) for value in carrier), Fraction(0)) != Fraction(a)
    proposal = {
        "vertex": 4,
        "point": carrier,
        "metric": np.eye(3),
        "stencil": stencil,
        "star_cell_ids": tuple(int(value) for value in state.snapshot.cell_ids),
    }
    status = _apply_relocation_batch(state, [proposal])
    assert status is TetMeshEditStatus.APPLIED
    row = np.asarray(state.snapshot.vertex_ids).tolist().index(4)
    np.testing.assert_array_equal(np.asarray(state.snapshot.points)[row], carrier)
    # A carrier outside the plane's rounding box is refused exactly.
    off_plane = dict(proposal, point=carrier + 2.0**-30)
    assert (
        _apply_relocation_batch(state, [off_plane]) is TetMeshEditStatus.INVALID_GEOMETRY
    )


def test_stalled_adaptation_reports_typed_unmet_edges() -> None:
    # Coarsening to h = 2 collapses everything collapsible; the cube edges
    # between pinned source corners stay short and are reported as
    # SOURCE_BOUNDARY witnesses naming their corner vertices.
    source = _source()
    outcome = _adapt(source, np.eye(3) * 0.25)
    evidence = outcome.evidence

    assert evidence.status is TetraMetricStatus.STALLED
    assert evidence.out_of_range_edges == len(evidence.unmet_edges) > 0
    kinds = [edge.kind for edge in evidence.unmet_edges]
    assert evidence.source_boundary_edges == kinds.count(
        TetraMetricUnmetKind.SOURCE_BOUNDARY
    )
    assert evidence.unimproved_edges == kinds.count(TetraMetricUnmetKind.UNIMPROVED)
    assert evidence.unrepresentable_edges == kinds.count(
        TetraMetricUnmetKind.UNREPRESENTABLE
    )
    coordinates = np.asarray(source.mesh.coordinates, dtype=np.float64)
    corners = {
        int(identifier)
        for identifier, point in zip(
            source.mesh.vertex_global_ids.tolist(), coordinates, strict=True
        )
        if np.all((point == 0.0) | (point == 1.0))
    }
    boundary = [
        edge
        for edge in evidence.unmet_edges
        if edge.kind is TetraMetricUnmetKind.SOURCE_BOUNDARY
    ]
    assert boundary
    for edge in boundary:
        assert set(edge.vertex_ids) <= corners
        assert edge.metric_length < 1.0 / np.sqrt(2.0)
    target_edges = _target_edges(outcome)
    assert all(edge.vertex_ids in target_edges for edge in evidence.unmet_edges)


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
        removed_vertex_ids=np.asarray((1, 4), dtype=np.uint64)
        if explicit_removed
        else None,
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


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_collapse_enclosing_center_handles_degenerate_metric_supports(
    dimension: int,
) -> None:
    metric = np.diag((4.0, 1.0, 0.25))
    axes = np.diag((0.5, 1.0, 2.0))[:dimension]
    offset = np.asarray((0.25, -0.5, 0.75))
    # More points than Python's recursion depth: support size, not a work
    # allowance or the call stack, bounds the mathematical recursion.
    interior = (
        np.linspace(-0.5, 0.5, 1025)[:, None]
        * np.sum(axes, axis=0)[None, :]
        / np.sqrt(dimension)
    )
    points = np.concatenate((axes, -axes, interior)) + offset
    for cloud in (points, points[::-1]):
        np.testing.assert_allclose(
            _metric_enclosing_center(cloud, metric), offset, rtol=0.0, atol=1e-14
        )


def test_exact_entity_frames_project_rational_edge_and_triangle() -> None:
    edge = tuple(tuple(Fraction(v) for v in row) for row in ((1, 2, 3), (3, 3, 5)))
    assert _entity_weights(
        _entity_frame(edge), (Fraction(2), Fraction(2), Fraction(4))
    ) == (Fraction(5, 9), Fraction(4, 9))
    triangle = tuple(
        tuple(Fraction(v) for v in row) for row in ((0, 0, 0), (2, 1, 0), (1, 0, 1))
    )
    u, v = Fraction(1, 3), Fraction(1, 4)
    point_values = tuple(
        u * triangle[1][k] + v * triangle[2][k] + Fraction((-1, 2, 1)[k], 7)
        for k in range(3)
    )
    point = (point_values[0], point_values[1], point_values[2])
    assert _entity_weights(_entity_frame(triangle), point) == (1 - u - v, u, v)
    assert _entity_weights(_entity_frame((edge[0], edge[0])), point) is None


def test_source_stencil_dyadic_image_matches_fraction_oracle() -> None:
    state = _state(_source(), np.eye(3))
    coordinates = np.asarray(state.source_mesh.coordinates)
    stencils = (
        {0: Fraction(1, 8), 1: Fraction(3, 8), 4: Fraction(1, 2)},
        {0: Fraction(1, 3), 1: Fraction(2, 3)},
        {},
    )
    for stencil in stencils:
        expected = tuple(
            sum(
                (
                    weight
                    * Fraction(float(coordinates[state.source_vertex_rows[vertex], axis]))
                    for vertex, weight in stencil.items()
                ),
                Fraction(0),
            )
            for axis in range(3)
        )
        assert _stencil_image(state, stencil) == expected


@pytest.mark.parametrize("skewed", (False, True))
def test_source_locator_exact_boxes_match_exhaustive_barycentric_search(
    skewed: bool,
) -> None:
    source = (
        _mesh_result(
            np.asarray(
                ((0.0, 0.0, 0.0), (0.8, 0.1, 0.0), (0.2, 0.9, 0.1), (0.1, 0.2, 0.9))
            ),
            [[0, 1, 2, 3]],
        )
        if skewed
        else _source()
    )
    state = _state(source, np.eye(3))
    for x in (Fraction(-1, 10), Fraction(0), Fraction(1, 3), Fraction(1, 2), Fraction(1)):
        for y in (Fraction(0), Fraction(1, 2), Fraction(1)):
            for z in (Fraction(0), Fraction(1, 2), Fraction(1)):
                point = (x, y, z)
                expected = None
                for (
                    cell_id,
                    vertices,
                    inverse,
                    origin,
                    _,
                    _,
                    _,
                    _,
                ) in state.source_records:
                    delta = tuple(point[k] - origin[k] for k in range(3))
                    tail = tuple(
                        sum(
                            (inverse[row][axis] * delta[axis] for axis in range(3)),
                            Fraction(0),
                        )
                        for row in range(3)
                    )
                    weights = (Fraction(1) - sum(tail, Fraction(0)), *tail)
                    if all(weight >= 0 for weight in weights):
                        expected = (cell_id, vertices, weights)
                        break
                assert _locate_source_weights(point, state.source_records) == expected


def test_source_metric_log_reuse_matches_uncached_interpolation() -> None:
    tensors = np.asarray(
        (
            ((3.0, 0.5, 0.1), (0.5, 2.0, 0.2), (0.1, 0.2, 1.0)),
            ((2.0, -0.2, 0.3), (-0.2, 4.0, 0.1), (0.3, 0.1, 1.5)),
            ((5.0, 0.1, 0.0), (0.1, 1.0, -0.2), (0.0, -0.2, 3.0)),
        )
    )
    rows = {0: 0, 1: 1, 2: 2}
    logs = _metric_logarithms(tensors)
    for stencil in (
        {0: Fraction(1, 2), 1: Fraction(1, 2)},
        {0: Fraction(1, 4), 1: Fraction(1, 4), 2: Fraction(1, 2)},
        {2: Fraction(1)},
    ):
        direct = _source_metric_from_stencil(tensors, rows, stencil)
        reused = _source_metric_from_stencil(
            tensors, rows, stencil, source_metric_log=logs
        )
        np.testing.assert_array_equal(reused, direct)


def test_ring_negative_cache_rechecks_changed_metric() -> None:
    state = _state(_source(), np.eye(3) * 9.0)
    cache: dict[tuple[int, int], tuple[Any, ...]] = {}

    def candidates(
        cache_arg: dict[tuple[int, int], tuple[Any, ...]] | None,
    ) -> list[
        tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...], float]
    ]:
        groups = _replacement_candidates(
            state,
            TetMeshEditKind.RECONNECT,
            set(),
            PredicateMode.EXACT,
            negative_rings=cache_arg,
        )
        return [
            (
                tuple(row["candidate_key"]),
                tuple(row["removed_cell_ids"]),
                tuple(map(tuple, row["replacement_cell_vertices"].tolist())),
                row["gain"],
            )
            for group in groups
            for row in group
        ]

    assert candidates(cache) == candidates(None)
    assert cache
    assert candidates(cache) == candidates(None)
    vertex = next(iter(state.metric_by_vertex))
    state.metric_by_vertex[vertex] = state.metric_by_vertex[vertex] * 1.01
    assert candidates(cache) == candidates(None)


def test_ring_reconnection_prefers_unit_band_over_higher_cell_quality() -> None:
    coordinates = np.asarray(
        (
            (-0.010318703849283394, -0.3251258635134156, -0.3),
            (-0.010318703849283394, -0.3251258635134156, 0.3),
            (0.5059103067167272, 0.09339615338399498, 0.0),
            (0.11208761898062111, 0.8085287611804144, 0.0),
            (-1.0785973893300873, -0.03984950581288704, 0.0),
            (0.13430042260838848, -0.5753954871741337, 0.0),
        )
    )
    ring = (2, 3, 4, 5)
    state = _state(
        _mesh_result(
            coordinates,
            [[0, 1, ring[index], ring[(index + 1) % 4]] for index in range(4)],
        ),
        np.eye(3),
    )
    result = _best_ring_triangulation(
        coordinates,
        np.broadcast_to(np.eye(3), (6, 3, 3)),
        _topology(state.snapshot),
        (0, 1),
        ring,
        PredicateMode.EXACT,
    )
    assert result is not None
    edges = {tuple(sorted(edge)) for cell in result[0] for edge in combinations(cell, 2)}
    assert (3, 5) in edges and (2, 4) not in edges
    assert np.linalg.norm(coordinates[3] - coordinates[5]) < np.sqrt(2.0)
    assert np.linalg.norm(coordinates[2] - coordinates[4]) > np.sqrt(2.0)
    # The alternative is geometrically valid and has higher minimum quality,
    # but its long diagonal defeats the requested metric band.
    other = coordinates[
        np.asarray(((0, 2, 3, 4), (1, 2, 4, 3), (0, 2, 4, 5), (1, 2, 5, 4)))
    ]
    volumes = np.abs(np.linalg.det(other[:, 1:] - other[:, :1])) / 6.0
    denominator = sum(
        np.sum((other[:, first] - other[:, second]) ** 2, axis=1)
        for first, second in combinations(range(4), 2)
    )
    assert result[1] < np.min(12.0 * (3.0 * volumes) ** (2.0 / 3.0) / denominator)
