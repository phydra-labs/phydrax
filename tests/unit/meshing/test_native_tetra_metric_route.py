from __future__ import annotations

from itertools import permutations
from typing import Any

import numpy as np
import pytest

import phydrax as phx
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


@pytest.mark.parametrize("dimension", [0, 3], ids=["vertex", "cell"])
def test_native_metric_route_refuses_signed_id_exhaustion(dimension: int) -> None:
    source = _source(exhausted_dimension=dimension)
    with pytest.raises(OverflowError, match="stable int64 identifier range"):
        _run(source, _request(source), MeshAdaptationRoute.NATIVE_METRIC_3D)
