from __future__ import annotations

from dataclasses import fields, replace
from itertools import permutations
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax._geometry_predicates import orient3d, PredicateMode
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.meshing import (
    _tetra_metric as tetra_metric,
    execute_mesh_adaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    MeshAdaptationStatus,
    MeshingEntityKind,
    MeshingScope,
    MeshMetricField,
    MetricMeshAdaptation,
    prepare_mesh_adaptation,
)
from phydrax.meshing._metric import metric_edge_lengths
from phydrax.meshing._topology_edit import key_rows


_LOWER = 1.0 / np.sqrt(2.0)
_UPPER = np.sqrt(2.0)


def _source(
    *, curved_boundary: bool = False, exhausted_dimension: int | None = None
) -> Any:
    shape = 3
    vertex_rows = np.arange(shape**3, dtype=np.int64).reshape((shape,) * 3)
    axis = np.linspace(0.0, 1.0, shape, dtype=np.float64)
    coordinates = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        (-1, 3)
    )
    if curved_boundary:
        x, y, z = coordinates.T
        coordinates[:, 0] += 0.075 * np.sin(np.pi * y) * np.sin(np.pi * z)
        coordinates[:, 1] += 0.055 * np.sin(np.pi * x) * np.sin(np.pi * z)
        coordinates[:, 2] += 0.035 * np.sin(np.pi * x) * np.sin(np.pi * y)
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
            corners = coordinates[tetrahedron]
            if np.linalg.det(corners[1:] - corners[:1]) < 0.0:
                tetrahedron[[1, 2]] = tetrahedron[[2, 1]]
            tetrahedra.append(tetrahedron)
    cell_ids = np.arange(3000, 3000 + len(tetrahedra), dtype=np.int64)
    vertex_ids = np.arange(shape**3, dtype=np.int64)
    if exhausted_dimension == 0:
        vertex_ids[-1] = np.iinfo(np.int64).max
    if exhausted_dimension == 3:
        cell_ids[-1] = np.iinfo(np.int64).max
    block = CellBlock(
        "tetrahedra",
        "tetrahedron",
        np.asarray(tetrahedra, dtype=np.int32),
        global_ids=cell_ids,
    )
    mesh = CellMesh(
        coordinates,
        (block,),
        vertex_global_ids=vertex_ids,
    )
    return phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


def _request(source: Any, *, values: np.ndarray | None = None) -> MetricMeshAdaptation:
    mesh = source.mesh
    vertices = mesh.entity_set(0)
    scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        0,
        vertices.entity_set_id,
        np.asarray(vertices.entity_ids, dtype=np.int64),
    )
    if values is None:
        tensor = np.diag((4.0, 3.8, 3.6))
        values = np.broadcast_to(tensor, (vertices.entity_ids.size, 3, 3)).copy()
    return MetricMeshAdaptation(
        MeshMetricField(
            scope,
            values,
            minimum_size=1.0e-3,
            maximum_size=10.0,
            maximum_anisotropy=10.0,
        )
    )


def _run(source: Any, request: MetricMeshAdaptation, route: Any) -> Any:
    policy = MeshAdaptationPolicy(route, relocation=False)
    prepared = prepare_mesh_adaptation(source, request, policy=policy)
    return execute_mesh_adaptation(prepared)


def _edge_lengths(result: Any) -> np.ndarray:
    metric = result.metric
    if metric is None:
        raise AssertionError("A metric route must return a target-bound metric.")
    mesh = result.target.mesh
    values = np.asarray(metric.values, dtype=np.float64)
    metric_rows = key_rows(
        np.asarray(metric.scope.entity_ids, dtype=np.int64)[:, None],
        np.asarray(mesh.vertex_global_ids, dtype=np.int64)[:, None],
    )
    return np.asarray(
        metric_edge_lengths(
            values[metric_rows],
            np.asarray(mesh.coordinates, dtype=np.float64),
            np.asarray(mesh.connectivity.edges, dtype=np.int32),
        ),
        dtype=np.float64,
    )


def _assert_valid_metric_mesh(result: Any) -> None:
    result.target.audit.require_passed()
    lengths = _edge_lengths(result)
    assert np.all(lengths >= _LOWER - 1.0e-12)
    assert np.all(lengths <= _UPPER + 1.0e-12)
    points = np.asarray(result.target.mesh.coordinates, dtype=np.float64)
    cells = np.concatenate(
        [
            np.asarray(block.vertices, dtype=np.int64)
            for block in result.target.mesh.blocks
        ],
        axis=0,
    )
    corners = points[cells]
    orientation = orient3d(
        corners[:, 0],
        corners[:, 1],
        corners[:, 2],
        corners[:, 3],
        mode=PredicateMode.EXACT,
    )
    assert np.all(np.asarray(orientation.certain, dtype=np.bool_))
    assert np.all(np.asarray(orientation.signs, dtype=np.int8) > 0)


def test_native_metric_route_refines_a_kuhn_cube_with_exact_lineage() -> None:
    from fractions import Fraction

    from phydrax.meshing._tetra_metric import TetraMetricEvidence, TetraMetricStatus

    source = _source()
    result = _run(source, _request(source), MeshAdaptationRoute.NATIVE_METRIC_3D)

    assert result.status in (MeshAdaptationStatus.COMPLETE, MeshAdaptationStatus.STALLED)
    evidence = result.evidence
    assert isinstance(evidence, TetraMetricEvidence)
    assert evidence.status is TetraMetricStatus.COMPLETE
    assert evidence.splits > 0
    assert evidence.out_of_range_edges == 0
    _assert_valid_metric_mesh(result)
    assert result.lineage is not None and result.transition is not None

    stencil = result.stencil
    assert stencil is not None
    target_ids = np.asarray(stencil.target_global_ids, dtype=np.int64)
    source_ids = set(np.asarray(source.mesh.vertex_global_ids).tolist())
    new_vertices = 0
    for row, weights in enumerate(np.asarray(stencil.weights, dtype=np.float64)):
        valid = np.asarray(stencil.valid, dtype=np.bool_)[row]
        exact = [Fraction(float(value)) for value in weights[valid]]
        assert sum(exact, Fraction(0)) == 1
        assert all(value.denominator.bit_count() == 1 for value in exact)
        new_vertices += int(target_ids[row]) not in source_ids
    assert new_vertices > 0


def test_only_committed_retriangulations_change_local_evidence_identity() -> None:
    from phydrax.meshing._tetra_metric import TetraMetricEvidence, TetraMetricStatus

    def evidence(retriangulations: int) -> TetraMetricEvidence:
        return TetraMetricEvidence(
            status=TetraMetricStatus.COMPLETE,
            passes=1,
            splits=1,
            collapses=0,
            flips23=0,
            flips32=0,
            ring_reconnections=0,
            relocations=0,
            rejected_topology=0,
            rejected_protected=0,
            rejected_uncertain=0,
            rejected_invalid=0,
            retriangulations=retriangulations,
            measured_edges=9,
            out_of_range_edges=0,
            minimum_metric_length=0.75,
            maximum_metric_length=1.25,
            minimum_metric_quality=0.625,
            maximum_region_volume_error=0.0,
            unmet_edges=(),
        )

    local = evidence(0)
    assert local.retriangulations == 0
    # Published local-only representation at 7a40f9ca, with identical measurements.
    assert (
        local.evidence_id
        == "755ba0c1006da8970354f5283155bd653e59cef6e94a3ee1af669ea9227921ad"
    )
    identifiers = {local.evidence_id, evidence(1).evidence_id, evidence(2).evidence_id}
    assert len(identifiers) == 3


def _graded_box_values(source: Any, *, scalar: bool) -> np.ndarray:
    points = np.asarray(source.mesh.coordinates)
    values = np.zeros((points.shape[0], 3, 3), dtype=np.float64)
    if scalar:
        rate = 4.0 + points[:, 0] + 0.5 * points[:, 2]
        axes = np.arange(3, dtype=np.int64)
        values[:, axes, axes] = (rate * rate)[:, None]
    else:
        values[:, 0, 0] = values[:, 1, 1] = 6.76
        values[:, 2, 2] = (3.2 + points[:, 2]) ** 2
    return values


def _assert_same_evidence(
    actual: tetra_metric.TetraMetricEvidence,
    expected: tetra_metric.TetraMetricEvidence,
) -> None:
    for field in fields(tetra_metric.TetraMetricEvidence):
        assert getattr(actual, field.name) == getattr(expected, field.name), field.name


@pytest.mark.parametrize("scalar", [False, True], ids=["axis-graded", "scalar-graded"])
def test_graded_box_transfers_across_retired_source_cells(
    scalar: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source()
    measure = tetra_metric._adapted_evidence
    reference_ids: list[str] = []

    def compare(
        state: tetra_metric._NativeState,
        adaptation: tetra_metric.TetMeshMetricAdaptation,
        snapshot: tetra_metric.TetMeshArrays,
        lower: float,
    ) -> tetra_metric.TetraMetricEvidence:
        assert adaptation.closed_measurements is not None
        measured = measure(state, adaptation, snapshot, lower)
        reference = measure(
            state, replace(adaptation, closed_measurements=None), snapshot, lower
        )
        _assert_same_evidence(measured, reference)
        reference_ids.append(reference.evidence_id)
        return measured

    monkeypatch.setattr(tetra_metric, "_adapted_evidence", compare)
    values = _graded_box_values(source, scalar=scalar)
    policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_METRIC_3D, relocation=True)
    result = execute_mesh_adaptation(
        prepare_mesh_adaptation(source, _request(source, values=values), policy=policy)
    )
    evidence = result.evidence
    assert result.status is MeshAdaptationStatus.COMPLETE
    assert isinstance(evidence, tetra_metric.TetraMetricEvidence)
    assert evidence.status is tetra_metric.TetraMetricStatus.COMPLETE
    assert evidence.evidence_id == reference_ids[-1]
    assert evidence.retriangulations == 1
    assert evidence.out_of_range_edges == 0
    if scalar:
        corners = np.asarray(result.target.mesh.coordinates)[
            np.asarray(result.target.mesh.blocks[0].vertices)
        ]
        determinant = np.linalg.det(corners[:, 1:] - corners[:, :1])
        squared_edges = 4.0 * np.sum(
            (corners - corners.mean(axis=1, keepdims=True)) ** 2, axis=(1, 2)
        )
        assert np.all(12.0 * np.cbrt((0.5 * determinant) ** 2) / squared_edges >= 0.60)
    _assert_valid_metric_mesh(result)
    stencil = result.stencil
    assert stencil is not None
    transferred = stencil.apply(source.mesh.vertex_global_ids, source.mesh.coordinates)
    target_rows = key_rows(
        np.asarray(result.target.mesh.vertex_global_ids)[:, None],
        np.asarray(stencil.target_global_ids)[:, None],
    )
    assert np.all(target_rows >= 0)
    np.testing.assert_allclose(
        transferred,
        np.asarray(result.target.mesh.coordinates)[target_rows],
        rtol=0.0,
        atol=4 * np.finfo(np.float64).eps,
    )


def test_closed_measurements_reject_contradictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    measure = tetra_metric._adapted_evidence

    def check(
        state: tetra_metric._NativeState,
        adaptation: tetra_metric.TetMeshMetricAdaptation,
        snapshot: tetra_metric.TetMeshArrays,
        lower: float,
    ) -> tetra_metric.TetraMetricEvidence:
        closed = adaptation.closed_measurements
        assert closed is not None
        for contradictory in (
            replace(closed, edge_count=0),
            replace(closed, minimum_length=float(np.nextafter(lower, 0.0))),
            replace(closed, maximum_length=float("nan")),
            replace(closed, total_region_volume=float("inf")),
            replace(closed, region=closed.region + 1),
        ):
            with pytest.raises(ValueError, match="closed metric measurements"):
                measure(
                    state,
                    replace(adaptation, closed_measurements=contradictory),
                    snapshot,
                    lower,
                )
        with pytest.raises(ValueError, match="measurements must be finite"):
            measure(
                state,
                replace(
                    adaptation,
                    closed_measurements=replace(
                        closed, minimum_metric_quality=float("nan")
                    ),
                ),
                snapshot,
                lower,
            )
        with pytest.raises(ValueError, match="closed metric measurements"):
            measure(state, replace(adaptation, status=2), snapshot, lower)
        with pytest.raises(ValueError, match="closed metric measurements"):
            measure(
                state,
                adaptation,
                snapshot._replace(tetrahedra=snapshot.tetrahedra[:0]),
                lower,
            )
        invalid = replace(state, source_region_volumes={closed.region: float("nan")})
        stopped = measure(invalid, adaptation, snapshot, lower)
        reference = measure(
            invalid, replace(adaptation, closed_measurements=None), snapshot, lower
        )
        assert stopped.status is tetra_metric.TetraMetricStatus.NUMERICAL_STOP
        _assert_same_evidence(stopped, reference)
        return measure(state, adaptation, snapshot, lower)

    monkeypatch.setattr(tetra_metric, "_adapted_evidence", check)
    request = _request(source, values=_graded_box_values(source, scalar=False))
    policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_METRIC_3D, relocation=True)
    result = execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=policy)
    )
    assert result.status is MeshAdaptationStatus.COMPLETE


def test_protected_edge_remeasurement_keeps_numerical_stop_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    mesh = source.mesh
    edges = mesh.entity_set(1)
    protected = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        1,
        edges.entity_set_id,
        np.asarray(edges.entity_ids[:1], dtype=np.int64),
    )
    measure = tetra_metric._adapted_evidence
    baselines: list[tetra_metric.TetraMetricEvidence] = []

    def check(
        state: tetra_metric._NativeState,
        adaptation: tetra_metric.TetMeshMetricAdaptation,
        snapshot: tetra_metric.TetMeshArrays,
        lower: float,
    ) -> tetra_metric.TetraMetricEvidence:
        assert state.protected_edge_ids
        assert adaptation.closed_measurements is None
        baselines.append(measure(state, adaptation, snapshot, lower))
        # Model a nonfinite source measure after the exact geometry is committed.
        invalid = replace(
            state,
            source_region_volumes={
                region: float("nan") for region in state.source_region_volumes
            },
        )
        return measure(invalid, adaptation, snapshot, lower)

    monkeypatch.setattr(tetra_metric, "_adapted_evidence", check)
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_METRIC_3D,
        protected_scopes=(protected,),
        relocation=False,
    )
    result = execute_mesh_adaptation(
        prepare_mesh_adaptation(source, _request(source), policy=policy)
    )
    evidence = result.evidence
    assert result.status is MeshAdaptationStatus.NUMERICAL_STOP
    assert isinstance(evidence, tetra_metric.TetraMetricEvidence)
    assert evidence.status is tetra_metric.TetraMetricStatus.NUMERICAL_STOP
    assert evidence.maximum_region_volume_error == np.finfo(np.float64).max
    assert 0 < evidence.measured_edges < result.target.mesh.entity_set(1).count
    baseline = baselines[-1]
    assert baseline.status is not tetra_metric.TetraMetricStatus.NUMERICAL_STOP
    assert evidence.evidence_id != baseline.evidence_id
    for field in fields(tetra_metric.TetraMetricEvidence):
        if field.name not in ("evidence_id", "status", "maximum_region_volume_error"):
            assert getattr(evidence, field.name) == getattr(baseline, field.name), (
                field.name
            )


@pytest.mark.parametrize("dimension", [0, 3], ids=["vertex", "cell"])
def test_native_metric_route_refuses_signed_id_exhaustion(dimension: int) -> None:
    source = _source(exhausted_dimension=dimension)
    with pytest.raises(OverflowError, match="stable int64 identifier range"):
        _run(source, _request(source), MeshAdaptationRoute.NATIVE_METRIC_3D)
