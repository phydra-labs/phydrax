from __future__ import annotations

from itertools import permutations
from typing import Any

import numpy as np
import pytest

import phydrax as phx
import phydrax.meshing._device_tetra_metric as _device_tetra_metric
from phydrax._geometry_predicates import orient3d, PredicateMode
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.meshing import (
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
from phydrax.meshing._tetra_metric import TetraMetricStatus
from phydrax.meshing._topology_edit import key_rows


_LOWER = 1.0 / np.sqrt(2.0)
_UPPER = np.sqrt(2.0)


def _source(*, curved_boundary: bool = False) -> Any:
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
    block = CellBlock(
        "tetrahedra",
        "tetrahedron",
        np.asarray(tetrahedra, dtype=np.int32),
        global_ids=np.arange(3000, 3000 + len(tetrahedra), dtype=np.int64),
    )
    mesh = CellMesh(
        coordinates,
        (block,),
        vertex_global_ids=np.arange(shape**3, dtype=np.int64),
    )
    return phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


def _request(source: Any) -> MetricMeshAdaptation:
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


def test_device_tetra_metric_agrees_with_native_on_cube_and_curved_boundary() -> None:
    for curved_boundary in (False, True):
        source = _source(curved_boundary=curved_boundary)
        request = _request(source)
        native = _run(source, request, MeshAdaptationRoute.NATIVE_METRIC_3D)
        device = _run(source, request, MeshAdaptationRoute.DEVICE_METRIC_3D)

        assert native.status is device.status
        assert device.status is MeshAdaptationStatus.COMPLETE
        assert device.evidence is not None
        assert device.evidence.status is TetraMetricStatus.COMPLETE
        assert device.evidence.metric_evidence is not None
        assert device.evidence.metric_evidence.out_of_range_edges == 0
        assert device.evidence.host_barriers > 0
        assert device.evidence.committed
        _assert_valid_metric_mesh(native)
        _assert_valid_metric_mesh(device)
        np.testing.assert_array_equal(
            device.target.mesh.vertex_global_ids, native.target.mesh.vertex_global_ids
        )
        np.testing.assert_array_equal(
            device.target.mesh.blocks[0].global_ids,
            native.target.mesh.blocks[0].global_ids,
        )
        np.testing.assert_array_equal(
            device.target.mesh.blocks[0].vertices, native.target.mesh.blocks[0].vertices
        )
        np.testing.assert_array_equal(
            device.target.mesh.coordinates, native.target.mesh.coordinates
        )

        selected = device.evidence.candidate_edge_keys
        assert len(selected) == device.evidence.independent_split_candidates
        assert selected
        assert all(first < second for first, second in selected)
        row_by_id = {
            int(identifier): row
            for row, identifier in enumerate(source.mesh.vertex_global_ids.tolist())
        }
        source_edges = np.asarray(
            [[row_by_id[first], row_by_id[second]] for first, second in selected],
            dtype=np.int32,
        )
        metric_rows = np.asarray(request.metric.values, dtype=np.float64)
        selected_lengths = np.asarray(
            metric_edge_lengths(
                metric_rows,
                np.asarray(source.mesh.coordinates, dtype=np.float64),
                source_edges,
            ),
            dtype=np.float64,
        )
        priority = [
            (-float(length), key[0], key[1])
            for key, length in zip(selected, selected_lengths, strict=True)
        ]
        assert priority == sorted(priority)


def test_device_tetra_metric_rolls_back_an_invalid_exact_candidate_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    request = _request(source)

    def reject_batch(
        points: np.ndarray,
        metric: np.ndarray,
        cells: np.ndarray,
        edge_rows: np.ndarray,
        cavities: tuple[tuple[int, ...], ...],
        selected: tuple[int, ...],
        /,
    ) -> tuple[bool, int]:
        return False, 2 * len(selected)

    monkeypatch.setattr(_device_tetra_metric, "_resolve_exact_split_batch", reject_batch)
    result = _run(source, request, MeshAdaptationRoute.DEVICE_METRIC_3D)

    assert result.status is MeshAdaptationStatus.NUMERICAL_STOP
    assert result.target is source
    assert result.transition is None
    assert result.lineage is None
    assert result.evidence.status is TetraMetricStatus.NUMERICAL_STOP
    assert result.evidence.metric_evidence is None
    assert result.evidence.exact_predicates_resolved > 0
    assert not result.evidence.committed
    assert result.target.mesh.mesh_id == source.mesh.mesh_id
