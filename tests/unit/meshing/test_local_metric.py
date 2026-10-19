#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import numpy as np
import pytest

from phydrax.discretization import CellMesh, PolygonalConnectivity
from phydrax.meshing import (
    CellMeshingResult,
    execute_mesh_adaptation,
    LocalMetricEvidence,
    MeshAdaptationPolicy,
    MeshAdaptationResult,
    MeshAdaptationRoute,
    MeshAdaptationStatus,
    MetricMeshAdaptation,
    prepare_mesh_adaptation,
    RelocationMeshAdaptation,
)
from tests._support.metric_meshes import constant_mesh_metric, planar_metric_source


_LOWER = 1.0 / np.sqrt(2.0)
_UPPER = np.sqrt(2.0)

# Empirical source: planar_metric_source refinement fixture's metric edge-length range.
REFINED_METRIC_EDGE_LENGTH_ATOL = 1e-12
# Empirical `test_metric_refinement_reaches_unit_range`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
REFINED_TRIANGLE_AREA_SUM_ATOL = 1e-14


def _adapt(
    source: CellMeshingResult, request: MetricMeshAdaptation | RelocationMeshAdaptation, /
) -> MeshAdaptationResult:
    policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_METRIC_2D)
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=policy)
    )


def _signed_areas(mesh: CellMesh, /) -> np.ndarray:
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


def _edge_metric_lengths(mesh: CellMesh, tensor: np.ndarray, /) -> np.ndarray:
    points = np.asarray(mesh.coordinates)
    assert isinstance(mesh.connectivity, PolygonalConnectivity)
    edges = np.asarray(mesh.connectivity.edges)
    delta = points[edges[:, 1]] - points[edges[:, 0]]
    return np.sqrt(np.sum((delta @ tensor) * delta, axis=1))


@pytest.mark.parametrize(
    "tensor",
    (np.eye(2) / 0.2**2, np.diag((1.0 / 0.5**2, 1.0 / 0.1**2))),
    ids=("isotropic", "anisotropic"),
)
def test_metric_refinement_reaches_unit_range(tensor: np.ndarray) -> None:
    source = planar_metric_source()
    result = _adapt(source, MetricMeshAdaptation(constant_mesh_metric(source, tensor)))
    assert isinstance(result.evidence, LocalMetricEvidence)
    assert result.status is MeshAdaptationStatus.COMPLETE
    assert result.evidence.converged
    assert result.evidence.unit_fraction == 1.0
    lengths = _edge_metric_lengths(result.target.mesh, tensor)
    assert np.all(
        (lengths >= _LOWER - REFINED_METRIC_EDGE_LENGTH_ATOL)
        & (lengths <= _UPPER + REFINED_METRIC_EDGE_LENGTH_ATOL)
    )
    areas = _signed_areas(result.target.mesh)
    assert np.all(areas > 0.0)
    assert np.isclose(np.sum(areas), 1.0, rtol=0.0, atol=REFINED_TRIANGLE_AREA_SUM_ATOL)


def test_metric_coarsening_keeps_positive_cells() -> None:
    source = planar_metric_source(8)
    tensor = np.eye(2) / 0.45**2
    result = _adapt(source, MetricMeshAdaptation(constant_mesh_metric(source, tensor)))

    assert result.status is MeshAdaptationStatus.COMPLETE
    assert isinstance(result.evidence, LocalMetricEvidence)
    assert result.evidence.collapses > 0
    assert result.target.mesh.coordinates.shape[0] < source.mesh.coordinates.shape[0]
    assert np.all(_signed_areas(result.target.mesh) > 0.0)


def test_metric_adaptation_preserves_region_and_curve_classification() -> None:
    source = planar_metric_source(organized=True)
    tensor = np.eye(2) / 0.12**2
    result = _adapt(source, MetricMeshAdaptation(constant_mesh_metric(source, tensor)))
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
    # The baseline absolute conservation allowance is 46 binary64 eps.
    assert zone_areas == pytest.approx({"left": 0.5, "right": 0.5}, abs=1.0e-14)
    points = np.asarray(mesh.coordinates)
    centroids = np.concatenate(
        [points[np.asarray(block.vertices)].mean(axis=1) for block in mesh.blocks]
    )
    (left_zone,) = (zone for zone in target.zones if zone.name == "left")
    left = np.isin(cell_ids, np.asarray(left_zone.scope.entity_ids))
    assert np.all(centroids[left, 0] < 0.5) and np.all(centroids[~left, 0] > 0.5)
    assert isinstance(mesh.connectivity, PolygonalConnectivity)
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


def test_metric_adaptation_is_deterministic() -> None:
    source = planar_metric_source(perturbation=0.3)
    metric = constant_mesh_metric(source, np.diag((1.0 / 0.3**2, 1.0 / 0.12**2)))

    first = _adapt(source, MetricMeshAdaptation(metric))
    second = _adapt(source, MetricMeshAdaptation(metric))

    assert first.result_id == second.result_id
    assert first.target.mesh.mesh_id == second.target.mesh.mesh_id


def test_relocation_changes_coordinates_without_topology() -> None:
    source = planar_metric_source(6, perturbation=0.35)
    result = _adapt(
        source,
        RelocationMeshAdaptation(constant_mesh_metric(source, np.eye(2) / 0.17**2)),
    )

    assert isinstance(result.evidence, LocalMetricEvidence)
    assert result.evidence.relocations > 0
    assert (
        result.evidence.splits == result.evidence.collapses == result.evidence.flips == 0
    )
    assert result.target.mesh.topology_id == source.mesh.topology_id
    assert not np.array_equal(
        np.asarray(result.target.mesh.coordinates), np.asarray(source.mesh.coordinates)
    )
    assert np.all(_signed_areas(result.target.mesh) > 0.0)


def test_metric_adaptation_transfers_linear_fields_exactly() -> None:
    source = planar_metric_source(perturbation=0.3)
    result = _adapt(
        source,
        MetricMeshAdaptation(
            constant_mesh_metric(source, np.diag((1.0 / 0.3**2, 1.0 / 0.1**2)))
        ),
    )

    def field(points: np.ndarray) -> np.ndarray:
        return 2.0 + 3.0 * points[:, 0] - 1.5 * points[:, 1]

    source_mesh = source.mesh
    target_mesh = result.target.mesh
    assert result.stencil is not None
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
    # Baseline P1 affine interpolation allowance: 451 binary64 eps.
    np.testing.assert_allclose(np.asarray(values), expected, rtol=0.0, atol=1.0e-13)
