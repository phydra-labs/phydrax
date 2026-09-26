#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing import (
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


def _grid(count, /, *, perturbation=0.0):
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


def _scope(mesh, dimension, entity_ids, /):
    entities = mesh.entity_set(dimension)
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        entities.entity_set_id,
        np.asarray(entity_ids, dtype=np.int64),
    )


def _source(count=4, /, *, perturbation=0.0, organized=False):
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


def _metric(source, tensor, /):
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


def _adapt(source, request, /, **options):
    policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_METRIC_2D, **options)
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=policy)
    )


def _signed_areas(mesh, /):
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


def _edge_metric_lengths(mesh, tensor, /):
    points = np.asarray(mesh.coordinates)
    edges = np.asarray(mesh.connectivity.edges)
    delta = points[edges[:, 1]] - points[edges[:, 0]]
    return np.sqrt(np.sum((delta @ tensor) * delta, axis=1))


@pytest.mark.parametrize(
    "tensor",
    [np.eye(2) / 0.2**2, np.diag((1.0 / 0.5**2, 1.0 / 0.1**2))],
    ids=["isotropic", "anisotropic"],
)
def test_metric_adaptation_reaches_the_unit_mesh(tensor):
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


def test_metric_adaptation_coarsens_a_fine_mesh():
    source = _source(8)
    tensor = np.eye(2) / 0.45**2
    result = _adapt(source, MetricMeshAdaptation(_metric(source, tensor)))

    assert result.status is MeshAdaptationStatus.COMPLETE
    assert result.evidence.collapses > 0
    assert result.target.mesh.coordinates.shape[0] < source.mesh.coordinates.shape[0]
    assert np.all(_signed_areas(result.target.mesh) > 0.0)


def test_metric_adaptation_preserves_zones_patches_and_interfaces():
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


def test_metric_adaptation_is_deterministic():
    source = _source(perturbation=0.3)
    metric = _metric(source, np.diag((1.0 / 0.3**2, 1.0 / 0.12**2)))

    first = _adapt(source, MetricMeshAdaptation(metric))
    second = _adapt(source, MetricMeshAdaptation(metric))

    assert first.result_id == second.result_id
    assert first.target.mesh.mesh_id == second.target.mesh.mesh_id


def test_metric_adaptation_transfers_linear_fields_exactly():
    source = _source(perturbation=0.3)
    result = _adapt(
        source,
        MetricMeshAdaptation(_metric(source, np.diag((1.0 / 0.3**2, 1.0 / 0.1**2)))),
    )

    def field(points):
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


def test_relocation_only_adaptation_keeps_topology():
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
