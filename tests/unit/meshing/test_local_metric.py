#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from fractions import Fraction
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax._geometry_predicates import PredicateMode
from phydrax.meshing import (
    _local_metric as local_metric,
    execute_mesh_adaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    MeshAdaptationStatus,
    MeshingEntityKind,
    MeshingScope,
    MeshMetricField,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
    MetricMeshAdaptation,
    prepare_mesh_adaptation,
    RelocationMeshAdaptation,
)


_LOWER = 1.0 / np.sqrt(2.0)
_UPPER = np.sqrt(2.0)


def test_host_interpolation_preserves_identical_tensor_exactly() -> None:
    tensor = np.array([[2.3, 0.4], [0.4, 1.7]], dtype=np.float64)
    values = np.broadcast_to(tensor, (3, 3, 2, 2)).copy()
    weights = np.array([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0], [0.1, 0.7, 0.2]])
    np.testing.assert_array_equal(
        local_metric._interpolate_metrics(values, weights),
        np.broadcast_to(tensor, (3, 2, 2)),
    )


def _grid(count: Any, /, *, perturbation: Any = 0.0) -> Any:
    x, y = np.meshgrid(np.linspace(0.0, 1.0, count + 1), np.linspace(0.0, 1.0, count + 1))
    points = np.stack((x.ravel(), y.ravel()), axis=1)
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    offsets = np.where(interior[:, None], perturbation * np.asarray((1.0, -0.6)), 0.0)
    shift = np.sin(7.0 * points[:, :1] + 3.0 * points[:, 1:])
    points = points + offsets * shift / count
    index = np.arange((count + 1) ** 2).reshape((count + 1, count + 1))
    a = index[:-1, :-1].ravel()
    b = index[:-1, 1:].ravel()
    c = index[1:, 1:].ravel()
    d = index[1:, :-1].ravel()
    triangles = np.concatenate((np.stack((a, b, c), 1), np.stack((a, c, d), 1)))
    return points, triangles


def _scope(mesh: Any, dimension: Any, entity_ids: Any, /) -> Any:
    entities = mesh.entity_set(dimension)
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        entities.entity_set_id,
        np.asarray(entity_ids, dtype=np.int64),
    )


def _source(count: Any = 4, /, *, perturbation: Any = 0.0, organized: Any = False) -> Any:
    points, triangles = _grid(count, perturbation=perturbation)
    mesh = phx.discretization.CellMesh.from_triangles(points, triangles)
    mesh = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si()).mesh
    if not organized:
        return phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    coordinates = np.asarray(mesh.coordinates)
    cells = np.asarray(mesh.entity_set(2).entity_ids)
    centroids = np.concatenate(
        [coordinates[np.asarray(block.vertices)].mean(axis=1) for block in mesh.blocks]
    )
    # ty: ignore[unresolved-attribute]
    edges = np.asarray(mesh.connectivity.edges)
    bottom = np.all(coordinates[edges][:, :, 1] == 0.0, axis=1)
    zones = (
        MeshZone(
            "left", MeshZoneRole.REGION, _scope(mesh, 2, cells[centroids[:, 0] < 0.5])
        ),
        MeshZone(
            "right", MeshZoneRole.REGION, _scope(mesh, 2, cells[centroids[:, 0] > 0.5])
        ),
    )
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids)
    patches = (MeshPatch("bottom", _scope(mesh, 1, edge_ids[bottom])),)
    return phx.meshing.certify_cell_mesh(
        mesh, phx.SpatialCoordinateContract.si(), patches=patches, zones=zones
    )


def _metric(source: Any, tensor: Any, /) -> Any:
    mesh = source.mesh
    vertices = mesh.entity_set(0)
    scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        0,
        vertices.entity_set_id,
        vertices.entity_ids,
    )
    values = np.broadcast_to(tensor, (vertices.entity_ids.shape[0], 2, 2))
    return MeshMetricField(scope, values, minimum_size=1.0e-3, maximum_size=10.0)


def _adapt(source: Any, request: Any, /, **options: Any) -> Any:
    policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_METRIC_2D, **options)
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=policy)
    )


def _signed_areas(mesh: Any, /) -> Any:
    points = np.asarray(mesh.coordinates)
    areas = []
    for block in mesh.blocks:
        a, b, c = np.moveaxis(points[np.asarray(block.vertices)], 1, 0)
        areas.append(
            0.5
            * (
                (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
            )
        )
    return np.concatenate(areas)


def _edge_metric_lengths(mesh: Any, tensor: Any, /) -> Any:
    points = np.asarray(mesh.coordinates)
    edges = np.asarray(mesh.connectivity.edges)
    delta = points[edges[:, 1]] - points[edges[:, 0]]
    return np.sqrt(np.sum((delta @ tensor) * delta, axis=1))


def test_local_metric_scenario_1() -> None:
    for tensor in [np.eye(2) / 0.2**2, np.diag((1.0 / 0.5**2, 1.0 / 0.1**2))]:
        source = _source()
        result = _adapt(source, MetricMeshAdaptation(_metric(source, tensor)))

        assert result.status is MeshAdaptationStatus.COMPLETE
        assert result.evidence.converged
        assert result.evidence.unit_fraction == 1.0
        lengths = _edge_metric_lengths(result.target.mesh, tensor)
        assert np.all((lengths >= _LOWER - 1.0e-12) & (lengths <= _UPPER + 1.0e-12))
        areas = _signed_areas(result.target.mesh)
        assert np.all(areas > 0.0)
        assert np.isclose(np.sum(areas), 1.0, rtol=0.0, atol=1.0e-14)
    source = _source(8)
    tensor = np.eye(2) / 0.45**2
    result = _adapt(source, MetricMeshAdaptation(_metric(source, tensor)))

    assert result.status is MeshAdaptationStatus.COMPLETE
    assert result.evidence.collapses > 0
    assert result.target.mesh.coordinates.shape[0] < source.mesh.coordinates.shape[0]
    assert np.all(_signed_areas(result.target.mesh) > 0.0)
    source = _source(organized=True)
    tensor = np.eye(2) / 0.12**2
    result = _adapt(source, MetricMeshAdaptation(_metric(source, tensor)))
    target = result.target
    mesh = target.mesh

    assert result.status is MeshAdaptationStatus.COMPLETE
    areas = _signed_areas(mesh)
    assert np.all(areas > 0.0)
    cell_ids = np.asarray(mesh.entity_set(2).entity_ids)
    zone_areas = {
        zone.name: np.sum(areas[np.isin(cell_ids, np.asarray(zone.scope.entity_ids))])
        for zone in target.zones
    }
    assert zone_areas == pytest.approx({"left": 0.5, "right": 0.5}, abs=1.0e-14)
    points = np.asarray(mesh.coordinates)
    centroids = np.concatenate(
        [points[np.asarray(block.vertices)].mean(axis=1) for block in mesh.blocks]
    )
    (left_zone,) = (zone for zone in target.zones if zone.name == "left")
    left = np.isin(cell_ids, np.asarray(left_zone.scope.entity_ids))
    assert np.all(centroids[left, 0] < 0.5) and np.all(centroids[~left, 0] > 0.5)
    edges = np.asarray(mesh.connectivity.edges)
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids)
    (patch,) = target.patches
    patch_edges = edges[np.isin(edge_ids, np.asarray(patch.scope.entity_ids))]
    assert np.all(points[patch_edges][:, :, 1] == 0.0)
    patch_length = np.sum(np.abs(np.diff(points[patch_edges][:, :, 0], axis=1)))
    assert patch_length == pytest.approx(1.0, abs=1.0e-14)
    on_interface = np.all(points[edges][:, :, 0] == 0.5, axis=1)
    interface_length = np.sum(
        np.abs(np.diff(points[edges[on_interface]][:, :, 1], axis=1))
    )
    assert interface_length == pytest.approx(1.0, abs=1.0e-14)
    source = _source(perturbation=0.3)
    metric = _metric(source, np.diag((1.0 / 0.3**2, 1.0 / 0.12**2)))

    first = _adapt(source, MetricMeshAdaptation(metric))
    second = _adapt(source, MetricMeshAdaptation(metric))

    assert first.result_id == second.result_id
    assert first.target.mesh.mesh_id == second.target.mesh.mesh_id
    source = _source(6, perturbation=0.35)
    result = _adapt(
        source, RelocationMeshAdaptation(_metric(source, np.eye(2) / 0.17**2))
    )

    assert result.evidence.relocations > 0
    assert (
        result.evidence.splits == result.evidence.collapses == result.evidence.flips == 0
    )
    assert result.target.mesh.topology_id == source.mesh.topology_id
    assert not np.array_equal(
        np.asarray(result.target.mesh.coordinates), np.asarray(source.mesh.coordinates)
    )
    assert np.all(_signed_areas(result.target.mesh) > 0.0)


def test_local_metric_preserves_rotated_spd_tensor() -> None:
    angle = np.deg2rad(31.0)
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    tensor = rotation @ np.diag((1.0 / 0.45**2, 1.0 / 0.22**2)) @ rotation.T
    source = _source(perturbation=0.2)
    result = _adapt(source, MetricMeshAdaptation(_metric(source, tensor)))

    assert result.status is MeshAdaptationStatus.COMPLETE
    assert result.evidence.converged
    lengths = _edge_metric_lengths(result.target.mesh, tensor)
    assert np.all((lengths >= _LOWER - 1.0e-12) & (lengths <= _UPPER + 1.0e-12))
    assert np.all(_signed_areas(result.target.mesh) > 0.0)
    assert np.sum(_signed_areas(result.target.mesh)) == pytest.approx(1.0, abs=1.0e-14)


def test_metric_adaptation_transfers_linear_fields_exactly() -> None:
    source = _source(perturbation=0.3)
    result = _adapt(
        source,
        MetricMeshAdaptation(_metric(source, np.diag((1.0 / 0.3**2, 1.0 / 0.1**2)))),
    )

    def field(points: Any) -> Any:
        return 2.0 + 3.0 * points[:, 0] - 1.5 * points[:, 1]

    source_mesh = source.mesh
    target_mesh = result.target.mesh
    values = result.stencil.apply(
        np.asarray(source_mesh.vertex_global_ids),
        field(np.asarray(source_mesh.coordinates)),
    )
    order = np.argsort(np.asarray(target_mesh.vertex_global_ids))
    rows = order[
        np.searchsorted(
            np.asarray(target_mesh.vertex_global_ids)[order],
            np.asarray(result.stencil.target_global_ids),
        )
    ]
    expected = field(np.asarray(target_mesh.coordinates)[rows])
    np.testing.assert_allclose(np.asarray(values), expected, rtol=0.0, atol=1.0e-13)


def test_local_metric_disjoint_selection_exhausts_finite_candidates() -> None:
    from phydrax.meshing._local_metric import _disjoint_selection

    count = 130
    operations = np.repeat(np.arange(count, dtype=np.int64), 2)
    items = np.column_stack(
        (np.arange(count, dtype=np.int64), np.arange(1, count + 1, dtype=np.int64))
    ).reshape((-1,))
    selected = _disjoint_selection(
        np.arange(count, dtype=np.int64),
        operations,
        items,
        count + 1,
    )

    np.testing.assert_array_equal(np.flatnonzero(selected), np.arange(0, count, 2))


def test_split_preserves_an_unrepresentable_exact_boundary_point() -> None:
    # This annulus boundary segment has no binary64 interior point on its
    # exact line. Its authored midpoint remains a rational source point, not
    # the slightly outside point represented by its rounded carrier.
    points = np.asarray(
        (
            (0.48170439590427927, 0.6296209328044939),
            (0.7071067811865476, 0.7071067811865475),
            (0.38268343236508984, 0.9238795325112867),
        )
    )
    mesh = phx.discretization.CellMesh.from_triangles(
        points, np.asarray(((0, 1, 2),), dtype=np.int32)
    )
    metric = np.broadcast_to(np.eye(2) / 0.2**2, (3, 2, 2))
    state, source = local_metric._initial_state(
        mesh,
        metric,
        np.zeros(1, dtype=np.int64),
        np.zeros(3, dtype=np.int64),
        np.zeros(3, dtype=np.bool_),
        np.zeros(3, dtype=np.bool_),
        PredicateMode.EXACT,
    )
    topology = local_metric._topology(state, PredicateMode.EXACT)
    changed, _, count = local_metric._split_round(
        state, topology, local_metric._Counts(), PredicateMode.EXACT, None, source
    )
    assert count == 1
    exact_midpoint = tuple(
        (Fraction(float(a)) + Fraction(float(b))) / 2
        for a, b in zip(points[1], points[2], strict=True)
    )
    assert changed.source_stencil_exact[3] == ((1, Fraction(1, 2)), (2, Fraction(1, 2)))
    np.testing.assert_array_equal(
        changed.points[3], np.asarray(exact_midpoint, dtype=np.float64)
    )
    assert tuple(Fraction(float(v)) for v in changed.points[3]) != exact_midpoint
    published = local_metric._stencil(changed, source, np.arange(4), PredicateMode.EXACT)
    assert sum(published[3][3], Fraction(0)) == 1

    # A subsequent epoch must retain the authored exact point, not promote
    # the rounded coordinate to a new exact source vertex.
    next_mesh = phx.discretization.CellMesh.from_triangles(changed.points, changed.cells)
    exact_bank = (*source.exact_points, exact_midpoint)
    next_state, next_source = local_metric._initial_state(
        next_mesh,
        np.broadcast_to(np.eye(2), (4, 2, 2)),
        np.zeros(2, dtype=np.int64),
        np.zeros(next_mesh.entity_set(1).count, dtype=np.int64),
        np.zeros(next_mesh.entity_set(1).count, dtype=np.bool_),
        np.zeros(4, dtype=np.bool_),
        PredicateMode.EXACT,
        exact_source_points=exact_bank,
    )
    quarter = tuple(
        (a + b) / 2 for a, b in zip(exact_bank[1], exact_midpoint, strict=True)
    )
    located = local_metric._located_stencil_bank(
        next_source,
        np.asarray((quarter,), dtype=np.float64),
        PredicateMode.EXACT,
        exact_points=(quarter,),
    )
    assert located[4] == (((1, Fraction(1, 2)), (3, Fraction(1, 2))),)
    local_metric._stencil(next_state, next_source, np.arange(4), PredicateMode.EXACT)


def test_local_metric_rejects_an_exact_source_bank_with_another_carrier() -> None:
    source = _source()
    mesh = source.mesh
    bank = tuple(tuple(Fraction(float(v)) for v in row) for row in mesh.coordinates)
    changed = ((bank[0][0] + 1, bank[0][1]), *bank[1:])
    with pytest.raises(ValueError, match="canonically round"):
        local_metric.execute_local_metric_adaptation(
            mesh,
            np.broadcast_to(np.eye(2), (len(bank), 2, 2)),
            cell_classes=np.zeros(mesh.entity_set(2).count, dtype=np.int64),
            edge_classes=np.zeros(mesh.entity_set(1).count, dtype=np.int64),
            protected_edges=np.zeros(mesh.entity_set(1).count, dtype=np.bool_),
            fixed_vertices=np.zeros(len(bank), dtype=np.bool_),
            predicate_mode=PredicateMode.EXACT,
            topology_operations=False,
            relocation=False,
            exact_source_points=changed,
        )
