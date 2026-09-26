#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import AdaptiveSimplexPolicy, AdaptiveSimplexStatus
from phydrax.meshing import (
    adapt_device_metric,
    commit_device_metric_adaptation,
    execute_mesh_adaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    MeshAdaptationStatus,
    MeshingEntityKind,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingScope,
    MeshMetricField,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
    MetricMeshAdaptation,
    prepare_device_metric_adaptation,
    prepare_mesh_adaptation,
    RelocationMeshAdaptation,
)


_LOWER = 1.0 / np.sqrt(2.0)
_UPPER = np.sqrt(2.0)
# One capacity bucket for every test: one compiled executable per worker.
_DEVICE = AdaptiveSimplexPolicy(vertex_capacity=256, cell_capacity=512)


def _grid(count, /, *, perturbation=0.0, shear=0.0):
    x, y = np.meshgrid(np.linspace(0.0, 1.0, count + 1), np.linspace(0.0, 1.0, count + 1))
    points = np.stack((x.ravel(), y.ravel()), axis=1)
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    offsets = np.where(interior[:, None], perturbation * np.asarray((1.0, -0.6)), 0.0)
    shift = np.sin(7.0 * points[:, :1] + 3.0 * points[:, 1:])
    points = points + offsets * shift / count
    points[:, 0] += shear * points[:, 1]
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


def _source(count=4, /, *, perturbation=0.0, shear=0.0, organized=False):
    points, triangles = _grid(count, perturbation=perturbation, shear=shear)
    mesh = phx.discretization.CellMesh.from_triangles(points, triangles)
    contract = phx.SpatialCoordinateContract.si()
    mesh = phx.meshing.certify_cell_mesh(mesh, contract).mesh
    if not organized:
        return phx.meshing.certify_cell_mesh(mesh, contract)
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
    return phx.meshing.certify_cell_mesh(mesh, contract, patches=patches, zones=zones)


def _metric(source, tensor, /):
    mesh = source.mesh
    vertices = mesh.entity_set(0)
    scope = _scope(mesh, 0, vertices.entity_ids)
    values = np.broadcast_to(tensor, (vertices.entity_ids.shape[0], 2, 2))
    return MeshMetricField(scope, values, minimum_size=1.0e-3, maximum_size=10.0)


def _policy(device=_DEVICE, /, **options):
    return MeshAdaptationPolicy(
        MeshAdaptationRoute.DEVICE_METRIC_2D, device_policy=device, **options
    )


def _adapt(source, request, /, **options):
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=_policy(**options))
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
    ("tensor", "perturbation"),
    [(np.eye(2) / 0.2**2, 0.0), (np.diag((1.0 / 0.5**2, 1.0 / 0.1**2)), 0.3)],
    ids=["isotropic", "anisotropic"],
)
def test_device_metric_adaptation_reaches_a_certified_unit_mesh(tensor, perturbation):
    source = _source(perturbation=perturbation)
    result = _adapt(source, MetricMeshAdaptation(_metric(source, tensor)))

    assert result.route is MeshAdaptationRoute.DEVICE_METRIC_2D
    assert result.status is MeshAdaptationStatus.COMPLETE
    assert result.evidence.converged and result.evidence.unit_fraction == 1.0
    assert result.evidence.splits > 0
    result.target.audit.require_passed()
    lengths = _edge_metric_lengths(result.target.mesh, tensor)
    assert np.all((lengths >= _LOWER - 1.0e-12) & (lengths <= _UPPER + 1.0e-12))
    areas = _signed_areas(result.target.mesh)
    assert np.all(areas > 0.0)
    assert np.isclose(np.sum(areas), 1.0, rtol=0.0, atol=1.0e-14)

    def field(points):
        return 2.0 + 3.0 * points[:, 0] - 1.5 * points[:, 1]

    target = result.target.mesh
    values = result.stencil.apply(
        np.asarray(source.mesh.vertex_global_ids),
        field(np.asarray(source.mesh.coordinates)),
    )
    order = np.argsort(np.asarray(target.vertex_global_ids))
    rows = order[
        np.searchsorted(
            np.asarray(target.vertex_global_ids)[order],
            np.asarray(result.stencil.target_global_ids),
        )
    ]
    expected = field(np.asarray(target.coordinates)[rows])
    np.testing.assert_allclose(np.asarray(values), expected, rtol=0.0, atol=1.0e-13)


def test_device_metric_adaptation_preserves_zones_and_patches():
    source = _source(organized=True)
    result = _adapt(source, MetricMeshAdaptation(_metric(source, np.eye(2) / 0.12**2)))
    mesh = result.target.mesh

    assert result.status is MeshAdaptationStatus.COMPLETE
    areas = _signed_areas(mesh)
    cell_ids = np.asarray(mesh.entity_set(2).entity_ids)
    zone_areas = {
        zone.name: np.sum(areas[np.isin(cell_ids, np.asarray(zone.scope.entity_ids))])
        for zone in result.target.zones
    }
    assert zone_areas == pytest.approx({"left": 0.5, "right": 0.5}, abs=1.0e-14)
    points = np.asarray(mesh.coordinates)
    edges = np.asarray(mesh.connectivity.edges)
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids)
    (patch,) = result.target.patches
    patch_edges = edges[np.isin(edge_ids, np.asarray(patch.scope.entity_ids))]
    assert np.all(points[patch_edges][:, :, 1] == 0.0)
    patch_length = np.sum(np.abs(np.diff(points[patch_edges][:, :, 0], axis=1)))
    assert patch_length == pytest.approx(1.0, abs=1.0e-14)


def test_device_relocation_moves_vertices_without_topology_change():
    source = _source(6, perturbation=0.35)
    result = _adapt(
        source, RelocationMeshAdaptation(_metric(source, np.eye(2) / 0.17**2))
    )
    evidence = result.evidence

    assert evidence.relocations > 0
    assert evidence.splits == evidence.collapses == evidence.flips == 0
    assert result.target.mesh.topology_id == source.mesh.topology_id
    assert not np.array_equal(
        np.asarray(result.target.mesh.coordinates), np.asarray(source.mesh.coordinates)
    )
    assert np.all(_signed_areas(result.target.mesh) > 0.0)


def test_uncertain_device_predicates_escalate_without_applying_the_operation():
    # The slanted sides x = y / 2 and x = 1 + y / 2 are exactly straight, but
    # FILTERED_DEVICE certifies zero orientations only structurally: collapsing
    # or sliding a vertex of a slanted side needs host resolution.
    source = _source(8, shear=0.5)
    request = MetricMeshAdaptation(_metric(source, np.eye(2) / 0.3**2))
    prepared = prepare_device_metric_adaptation(source, request, policy=_policy())
    update = adapt_device_metric(prepared.layout, prepared.state)
    report = jax.device_get(update.report)

    assert AdaptiveSimplexStatus(int(report.status)) & (
        AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION
    )
    assert not report.failed
    assert report.rejected_uncertain > 0
    result = commit_device_metric_adaptation(prepared, update.state)
    assert result.evidence.status & AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION
    assert result.evidence.collapses > 0
    points = np.asarray(source.mesh.coordinates)
    identifiers = np.asarray(source.mesh.vertex_global_ids)
    slanted = np.isclose(points[:, 0], 0.5 * points[:, 1]) | np.isclose(
        points[:, 0], 1.0 + 0.5 * points[:, 1]
    )
    target = result.target.mesh
    target_ids = np.asarray(target.vertex_global_ids)
    order = np.argsort(target_ids)
    rows = order[np.searchsorted(target_ids[order], identifiers[slanted])]
    np.testing.assert_array_equal(target_ids[rows], identifiers[slanted])
    np.testing.assert_array_equal(np.asarray(target.coordinates)[rows], points[slanted])
    areas = _signed_areas(target)
    assert np.all(areas > 0.0)
    assert np.isclose(np.sum(areas), 1.0, rtol=0.0, atol=1.0e-14)


def test_device_capacity_overflow_refuses_the_call_and_keeps_the_state():
    source = _source()
    request = MetricMeshAdaptation(_metric(source, np.eye(2) / 0.2**2))
    tiny = AdaptiveSimplexPolicy(vertex_capacity=25, cell_capacity=32)
    prepared = prepare_device_metric_adaptation(source, request, policy=_policy(tiny))
    update = adapt_device_metric(prepared.layout, prepared.state)
    report = jax.device_get(update.report)

    assert AdaptiveSimplexStatus(int(report.status)) & (
        AdaptiveSimplexStatus.CAPACITY_EXCEEDED
    )
    assert report.failed
    assert report.splits == report.collapses == report.flips == report.relocations == 0
    for before, after in zip(
        jax.tree_util.tree_leaves(prepared.state),
        jax.tree_util.tree_leaves(update.state),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
    with pytest.raises(MeshingFailure) as failure:
        execute_mesh_adaptation(
            prepare_mesh_adaptation(source, request, policy=_policy(tiny))
        )
    assert failure.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED


def test_device_metric_adaptation_is_deterministic():
    source = _source(perturbation=0.3)
    request = MetricMeshAdaptation(
        _metric(source, np.diag((1.0 / 0.3**2, 1.0 / 0.12**2)))
    )

    first = _adapt(source, request)
    second = _adapt(source, request)
    prepared = prepare_device_metric_adaptation(source, request, policy=_policy())
    update = adapt_device_metric(prepared.layout, prepared.state)
    epoch = commit_device_metric_adaptation(prepared, update.state)

    assert first.result_id == second.result_id == epoch.result_id
    assert first.target.mesh.mesh_id == second.target.mesh.mesh_id
