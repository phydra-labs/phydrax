from __future__ import annotations

from fractions import Fraction
from itertools import combinations, permutations
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax._geometry_predicates import PredicateMode
from phydrax._meshcore import TetMeshEdit, TetMeshEditKind, TetMeshEditStatus
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._tetra_metric import (
    _apply_collapse_batch,
    _apply_relocation_batch,
    _best_ring_triangulation,
    _boundary_records,
    _carrier_simplices,
    _collapse_metric_candidates,
    _collapse_phase,
    _EntityFrame,
    _entity_frame,
    _entity_weights,
    _facet_group_map,
    _metric_enclosing_center,
    _locate_source_weights,
    _metric_logarithms,
    _NativeState,
    _prepare_native_state,
    _quantized_relocation,
    _relocation_goals,
    _relocation_proposals,
    _source_metric_from_stencil,
    _replacement_candidates,
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


def test_relocation_stage_cache_preserves_only_unchanged_stars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import phydrax.meshing._tetra_metric as metric

    state = SimpleNamespace(
        target_mesh=SimpleNamespace(periodic_topology=None),
        vertex_target_ids={1: 1},
    )
    proposal = {
        "vertex": 1,
        "star_vertices": frozenset((1, 2)),
        "star_cell_ids": (10,),
        "cavity": frozenset((10,)),
        "badness": 2.0,
        "gain": 1.0,
    }
    deep_calls = 0

    def candidates(
        _state: Any, cache: dict[int, Any], /, *, deep: bool,
        active_only: bool, band_only: bool,
    ) -> list[tuple[dict[str, Any], ...]]:
        nonlocal deep_calls
        if not deep:
            return []
        deep_calls += 1
        if deep_calls == 1:
            cache.update({1: proposal, 99: None})
            return [(proposal,)]
        assert cache == {99: None}
        return []

    monkeypatch.setattr(metric, "_relocation_proposals", candidates)
    monkeypatch.setattr(metric, "_disjoint_candidates", lambda rows: [rows[0][1]])
    monkeypatch.setattr(
        metric, "_apply_relocation_partition",
        lambda _state, selected, _counters, _cache: list(selected[0]),
    )
    assert metric._relocation_phase(state, {}, topology_operations=False) == 1
    assert deep_calls == 2


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
        source,
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
    assert len(snapshot.tetrahedra) == 12
    converged, outside, shortest, longest, quality = _unit_metric_status(state)
    # Only the hull spokes apex-ring (1.58) stay long; the merged star is in band.
    assert shortest >= 1.0 / np.sqrt(2.0) and quality > 0.0
    assert not converged and outside == 12 and longest > np.sqrt(2.0)


def test_relocation_backtracks_to_binary64_resolution() -> None:
    point = np.ones(3)
    goals = list(_relocation_goals(point, np.asarray(((1.0, 0.0, 0.0),))))
    assert len(goals) > 6
    np.testing.assert_array_equal(goals[12], point + np.asarray((2.0**-12, 0.0, 0.0)))
    assert not np.array_equal(goals[-1], point)
    np.testing.assert_array_equal(point + (goals[-1] - point) / 2.0, point)


def test_deep_boundary_relocation_keeps_affine_projection() -> None:
    state = _state(_source(4, nonuniform=True), np.eye(3) * 4.0)
    proposals = _relocation_proposals(
        state, {}, deep=True, active_only=True, band_only=True
    )
    assert proposals


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
    frames: dict[tuple[int, ...], _EntityFrame] = {}
    for factor in (0.5, 0.25, 0.125):
        trial = current + factor * (centroid - current)
        uncached = _quantized_relocation(state, simplices, trial, current)
        cached = _quantized_relocation(state, simplices, trial, current, frames=frames)
        assert uncached is not None and cached is not None
        assert cached[0] == uncached[0]
        np.testing.assert_array_equal(cached[1], uncached[1])
    assert frames
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
    triangle = tuple(tuple(Fraction(v) for v in row) for row in ((0, 0, 0), (2, 1, 0), (1, 0, 1)))
    u, v = Fraction(1, 3), Fraction(1, 4)
    point = tuple(u * triangle[1][k] + v * triangle[2][k] + Fraction((-1, 2, 1)[k], 7) for k in range(3))
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
                (weight * Fraction(float(coordinates[state.source_vertex_rows[vertex], axis]))
                 for vertex, weight in stencil.items()),
                Fraction(0),
            )
            for axis in range(3)
        )
        assert _stencil_image(state, stencil) == expected


def test_source_locator_exact_boxes_match_exhaustive_barycentric_search() -> None:
    state = _state(_source(), np.eye(3))
    for x in (Fraction(-1, 10), Fraction(0), Fraction(1, 3), Fraction(1, 2), Fraction(1)):
        for y in (Fraction(0), Fraction(1, 2), Fraction(1)):
            for z in (Fraction(0), Fraction(1, 2), Fraction(1)):
                point = (x, y, z)
                expected = None
                for cell_id, vertices, inverse, origin, _, _ in state.source_records:
                    delta = tuple(point[k] - origin[k] for k in range(3))
                    tail = tuple(
                        sum((inverse[row][axis] * delta[axis] for axis in range(3)), Fraction(0))
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

    def candidates(cache_arg: dict[tuple[int, int], tuple[Any, ...]] | None):
        groups = _replacement_candidates(
            state, TetMeshEditKind.RECONNECT, set(), PredicateMode.EXACT,
            negative_rings=cache_arg,
        )
        return [
            (
                tuple(row["candidate_key"]),
                tuple(row["removed_cell_ids"]),
                tuple(map(tuple, row["replacement_cell_vertices"].tolist())),
                row["gain"],
            )
            for group in groups for row in group
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
