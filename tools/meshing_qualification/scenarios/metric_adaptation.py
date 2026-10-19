# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._records import _quality_summary, _result_identity
from tools.meshing_qualification._runtime import (
    _contract,
    _measured,
    _runtime_identity,
    qualification_failure,
)
from tools.meshing_qualification.scenarios.providers import _mesh_scope


class _MetricUnitMeasurements(TypedDict):
    edges: int
    minimum_length: float
    mean_length: float
    maximum_length: float
    unit_fraction: float
    minimum_metric_quality: float
    mean_metric_quality: float


class _AdaptationTransferEvidence(TypedDict):
    transfer_id: str
    preserves_constants: bool
    preserves_linear: bool
    conservative: bool
    positivity_preserving: bool
    constant_error: float
    linear_error: float
    area_residual: float
    linear_integral_residual: float


class _MetricAdaptationExecution(TypedDict):
    source: phx.meshing.CellMeshingResult
    metric: phx.meshing.MeshMetricField
    native: phx.meshing.MeshAdaptationResult
    device: phx.meshing.MeshAdaptationResult
    relocation: phx.meshing.MeshAdaptationResult
    prepared: phx.meshing.PreparedDeviceMetricAdaptation
    report: phx.meshing.DeviceMetricReport
    resources: dict[str, dict[str, float]]


class _MetricAdaptationMeasurements(TypedDict):
    native_evidence: phx.meshing.LocalMetricEvidence
    device_evidence: phx.meshing.DeviceMetricEvidence
    relocation_evidence: phx.meshing.LocalMetricEvidence
    adapted: dict[str, phx.meshing.MeshAdaptationResult]
    route_evidence: dict[
        str, phx.meshing.LocalMetricEvidence | phx.meshing.DeviceMetricEvidence
    ]
    target_metrics: dict[str, phx.meshing.MeshMetricField]
    before: _MetricUnitMeasurements
    after: dict[str, _MetricUnitMeasurements]
    relocated: _MetricUnitMeasurements
    transfer: dict[str, _AdaptationTransferEvidence]
    relocation_transfer: _AdaptationTransferEvidence


def _metric_lattice(
    count: int, lower: float, upper: float, /
) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and counterclockwise triangles of a ``count x count`` square lattice."""
    axis = np.linspace(lower, upper, count + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    index = np.arange((count + 1) ** 2, dtype=np.int32).reshape((count + 1, count + 1))
    first, second = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
    third, fourth = index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    triangles = np.concatenate(
        (
            np.stack((first, second, third), axis=1),
            np.stack((first, third, fourth), axis=1),
        )
    )
    return points, triangles


def _layer_metric_tensors(points: np.ndarray, /) -> np.ndarray:
    """Diagonal internal-layer metric: size 0.04 across x = 1/2 growing to 0.3.

    Every other ambient axis (along the layer, and the normal of an embedded
    planar face) requests size 0.25.
    """
    sizes = np.full(points.shape, 0.25, dtype=np.float64)
    sizes[:, 0] = 0.04 + 0.52 * np.abs(points[:, 0] - 0.5)
    return np.eye(points.shape[1], dtype=np.float64) * sizes[:, None, :] ** -2


def _vertex_scope_rows(mesh: phx.discretization.CellMesh, /) -> np.ndarray:
    """Vertex rows in the sorted global-ID row order of a vertex scope."""
    return np.argsort(np.asarray(mesh.vertex_global_ids), kind="stable")


def _layer_metric(mesh: phx.discretization.CellMesh, /) -> phx.meshing.MeshMetricField:

    rows = _vertex_scope_rows(mesh)
    return phx.meshing.MeshMetricField(
        _mesh_scope(mesh, 0, np.asarray(mesh.vertex_global_ids, dtype=np.int64)),
        _layer_metric_tensors(np.asarray(mesh.coordinates, dtype=np.float64)[rows]),
        minimum_size=0.04,
        maximum_size=0.31,
        maximum_anisotropy=6.5,
    )


def _triangle_areas(points: np.ndarray, cells: np.ndarray, /) -> np.ndarray:
    spatial = np.pad(points, ((0, 0), (0, 3 - points.shape[1])))
    first = spatial[cells[:, 1]] - spatial[cells[:, 0]]
    second = spatial[cells[:, 2]] - spatial[cells[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(first, second), axis=1)


def _mesh_triangles(mesh: phx.discretization.CellMesh, /) -> np.ndarray:
    return np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])


def _vertex_dual_areas(mesh: phx.discretization.CellMesh, /) -> np.ndarray:
    """Return barycentric dual areas for P1 integration over the vertex rows."""
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = _mesh_triangles(mesh)
    areas = np.zeros((points.shape[0],), dtype=np.float64)
    shares = np.repeat(_triangle_areas(points, cells)[:, None] / 3.0, 3, axis=1)
    np.add.at(areas, cells, shares)
    return areas


def _metric_unit_measurements(
    mesh: phx.discretization.CellMesh, metric: phx.meshing.MeshMetricField, /
) -> _MetricUnitMeasurements:
    """Riemannian edge lengths and metric mean-ratio quality of a planar mesh."""
    rows = _vertex_scope_rows(mesh)
    inverse = np.empty_like(rows)
    inverse[rows] = np.arange(rows.shape[0])
    points = np.asarray(mesh.coordinates, dtype=np.float64)[rows]
    values = np.asarray(metric.values, dtype=np.float64)
    connectivity = mesh.connectivity
    if not isinstance(connectivity, phx.discretization.PolygonalConnectivity):
        raise TypeError(
            "metric_adaptation.connectivity must be PolygonalConnectivity; "
            f"got {type(connectivity).__name__}"
        )
    lengths = np.asarray(
        phx.meshing.metric_edge_lengths(
            values, points, inverse[np.asarray(connectivity.edges)]
        )
    )
    cells = inverse[_mesh_triangles(mesh)]
    cell_metric = np.asarray(
        phx.meshing.interpolate_mesh_metric(
            values[cells], np.full(cells.shape, 1.0 / 3.0, dtype=np.float64)
        )
    )
    corners = points[cells]
    sides = corners[:, (1, 2, 0)] - corners
    stretched = (cell_metric[:, None] @ sides[..., None])[..., 0]
    squared = np.sum(sides * stretched, axis=(1, 2))
    determinant = cell_metric[:, 0, 0] * cell_metric[:, 1, 1] - cell_metric[:, 0, 1] ** 2
    areas = _triangle_areas(points, cells)
    quality = 4.0 * np.sqrt(3.0) * areas * np.sqrt(determinant) / squared
    unit = (lengths >= 1.0 / np.sqrt(2.0)) & (lengths <= np.sqrt(2.0))
    return {
        "edges": lengths.shape[0],
        "minimum_length": float(np.min(lengths)),
        "mean_length": float(np.mean(lengths)),
        "maximum_length": float(np.max(lengths)),
        "unit_fraction": float(np.mean(unit)),
        "minimum_metric_quality": float(np.min(quality)),
        "mean_metric_quality": float(np.mean(quality)),
    }


def _linear_probe(points: np.ndarray, /) -> np.ndarray:
    return 2.0 + 3.0 * points[:, 0] - 1.5 * points[:, 1]


def _adaptation_transfer_evidence(
    adaptation: phx.meshing.MeshAdaptationResult, /
) -> _AdaptationTransferEvidence:
    """Measured constant/linear reproduction and P1 integrals of the P1 transfer."""

    transfer = adaptation.transfer
    if transfer is None:
        raise qualification_failure(
            "_adaptation_transfer_evidence",
            (f"The {adaptation.route.value} adaptation has no transfer."),
            {"transfer": transfer},
        )
    source = adaptation.source.mesh
    target = adaptation.target.mesh
    source_points = np.asarray(source.coordinates, dtype=np.float64)
    target_points = np.asarray(target.coordinates, dtype=np.float64)
    source_values = _linear_probe(source_points)
    transferred = np.asarray(transfer.apply(jnp.asarray(source_values)))
    constant = np.asarray(
        transfer.apply(jnp.ones((source_points.shape[0],), dtype=jnp.float64))
    )
    source_area = _vertex_dual_areas(source)
    target_area = _vertex_dual_areas(target)
    return {
        "transfer_id": transfer.transfer_id,
        "preserves_constants": transfer.preserves_constants,
        "preserves_linear": transfer.preserves_linear,
        "conservative": transfer.conservative,
        "positivity_preserving": transfer.positivity_preserving,
        "constant_error": float(np.max(np.abs(constant - 1.0))),
        "linear_error": float(np.max(np.abs(transferred - _linear_probe(target_points)))),
        "area_residual": abs(float(np.sum(target_area) - np.sum(source_area))),
        "linear_integral_residual": abs(
            float(np.dot(target_area, transferred) - np.dot(source_area, source_values))
        ),
    }


def _metric_operation_evidence(
    evidence: phx.meshing.LocalMetricEvidence | phx.meshing.DeviceMetricEvidence, /
) -> dict[str, object]:
    """Fields shared by `LocalMetricEvidence` and `DeviceMetricEvidence`."""
    return {
        "evidence_id": evidence.evidence_id,
        "passes": evidence.passes,
        "splits": evidence.splits,
        "collapses": evidence.collapses,
        "flips": evidence.flips,
        "relocations": evidence.relocations,
        "rejected_uncertain": evidence.rejected_uncertain,
        "rejected_invalid": evidence.rejected_invalid,
        "converged": evidence.converged,
        "stalled": evidence.stalled,
        "measured_edges": evidence.measured_edges,
        "out_of_range_edges": evidence.out_of_range_edges,
        "minimum_metric_length": evidence.minimum_metric_length,
        "maximum_metric_length": evidence.maximum_metric_length,
        "unit_fraction": evidence.unit_fraction,
        "minimum_metric_quality": evidence.minimum_metric_quality,
    }


def _execute_metric_adaptation() -> _MetricAdaptationExecution:
    """Prepare the source and execute native, device, and relocation routes."""

    points, triangles = _metric_lattice(6, 0.0, 1.0)
    source, certify_record = _measured(
        lambda: phx.meshing.certify_cell_mesh(
            phx.discretization.CellMesh.from_triangles(points, triangles), _contract()
        )
    )

    metric = _layer_metric(source.mesh)
    request = phx.meshing.MetricMeshAdaptation(metric)
    native_policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_METRIC_2D
    )
    native, native_record = _measured(
        lambda: phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(source, request, policy=native_policy)
        )
    )
    device_policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.DEVICE_METRIC_2D,
        device_policy=phx.discretization.AdaptiveSimplexPolicy(
            vertex_capacity=256, cell_capacity=512
        ),
    )
    prepared, prepare_record = _measured(
        lambda: phx.meshing.prepare_device_metric_adaptation(
            source,
            request,
            policy=device_policy,
        )
    )
    update, adapt_record = _measured(
        lambda: jax.block_until_ready(
            phx.meshing.adapt_device_metric(prepared.layout, prepared.state)
        )
    )

    report = jax.device_get(update.report)
    device, commit_record = _measured(
        lambda: phx.meshing.commit_device_metric_adaptation(prepared, update.state)
    )
    relocation, relocation_record = _measured(
        lambda: phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(
                source,
                phx.meshing.RelocationMeshAdaptation(metric),
                policy=native_policy,
            )
        )
    )
    return {
        "source": source,
        "metric": metric,
        "native": native,
        "device": device,
        "relocation": relocation,
        "prepared": prepared,
        "report": report,
        "resources": {
            "certify": certify_record,
            "native_adaptation": native_record,
            "device_prepare": prepare_record,
            "device_adapt_cold": adapt_record,
            "device_commit": commit_record,
            "relocation": relocation_record,
        },
    }


def _measure_metric_adaptation(
    execution: _MetricAdaptationExecution, /
) -> _MetricAdaptationMeasurements:
    """Require route evidence and measure bound metrics and P1 transfers."""

    source, metric = execution["source"], execution["metric"]
    native, device = execution["native"], execution["device"]
    relocation = execution["relocation"]
    native_evidence, device_evidence = native.evidence, device.evidence
    relocation_evidence = relocation.evidence
    if not isinstance(native_evidence, phx.meshing.LocalMetricEvidence):
        raise qualification_failure(
            "qualify_metric_adaptation",
            ("Native metric qualification requires local metric operation evidence."),
            {"native_evidence": native_evidence},
        )
    if not isinstance(device_evidence, phx.meshing.DeviceMetricEvidence):
        raise qualification_failure(
            "qualify_metric_adaptation",
            ("Device metric qualification requires device metric operation evidence."),
            {"device_evidence": device_evidence},
        )
    if not isinstance(relocation_evidence, phx.meshing.LocalMetricEvidence):
        raise qualification_failure(
            "qualify_metric_adaptation",
            ("Native relocation qualification requires local metric operation evidence."),
            {"relocation_evidence": relocation_evidence},
        )
    route_evidence: dict[
        str, phx.meshing.LocalMetricEvidence | phx.meshing.DeviceMetricEvidence
    ] = {"native": native_evidence, "device": device_evidence}
    adapted = {"native": native, "device": device}

    if any(result.metric is None for result in (*adapted.values(), relocation)):
        raise qualification_failure(
            "qualify_metric_adaptation",
            ("A metric adaptation did not bind its metric to the target."),
            {"adapted": adapted, "relocation": relocation},
        )

    before = _metric_unit_measurements(source.mesh, metric)
    after: dict[str, _MetricUnitMeasurements] = {}
    target_metrics: dict[str, phx.meshing.MeshMetricField] = {}
    for name, result in adapted.items():
        target_metric = result.metric
        if target_metric is None:
            raise qualification_failure(
                "qualify_metric_adaptation",
                ("A metric adaptation did not bind its metric to the target."),
                {"target_metric": target_metric},
            )
        target_metrics[name] = target_metric
        after[name] = _metric_unit_measurements(result.target.mesh, target_metric)
    relocation_metric = relocation.metric
    if relocation_metric is None:
        raise qualification_failure(
            "qualify_metric_adaptation",
            ("A metric adaptation did not bind its metric to the target."),
            {"relocation_metric": relocation_metric},
        )
    target_metrics["relocation"] = relocation_metric
    relocated = _metric_unit_measurements(relocation.target.mesh, relocation_metric)
    transfer = {
        name: _adaptation_transfer_evidence(result) for name, result in adapted.items()
    }
    relocation_transfer = _adaptation_transfer_evidence(relocation)
    return {
        "native_evidence": native_evidence,
        "device_evidence": device_evidence,
        "relocation_evidence": relocation_evidence,
        "adapted": adapted,
        "route_evidence": route_evidence,
        "target_metrics": target_metrics,
        "before": before,
        "after": after,
        "relocated": relocated,
        "transfer": transfer,
        "relocation_transfer": relocation_transfer,
    }


def _accept_metric_adaptation(
    execution: _MetricAdaptationExecution,
    measurements: _MetricAdaptationMeasurements,
    /,
) -> dict[str, bool]:
    """Apply the audit, unit-metric, transfer, and area acceptance gates."""

    source = execution["source"]
    native, device = execution["native"], execution["device"]
    relocation, report = execution["relocation"], execution["report"]
    relocation_evidence = measurements["relocation_evidence"]
    adapted, route_evidence = measurements["adapted"], measurements["route_evidence"]
    before, after = measurements["before"], measurements["after"]
    transfer = measurements["transfer"]
    checks = {
        "native target audited": native.target.audit.passed,
        "device target audited": device.target.audit.passed,
        "relocation target audited": relocation.target.audit.passed,
        "device passes did not fail": not bool(report.failed),
        "relocation kept the topology": (
            relocation.target.mesh.topology_id == source.mesh.topology_id
            and relocation_evidence.relocations > 0
            and relocation_evidence.splits == relocation_evidence.collapses == 0
        ),
    }
    for name, result in adapted.items():
        checks[f"{name} moved toward the unit mesh"] = (
            after[name]["unit_fraction"] > before["unit_fraction"]
            and route_evidence[name].unit_fraction >= 0.9
        )
        checks[f"{name} transfer reproduces linear fields"] = (
            transfer[name]["preserves_linear"]
            and transfer[name]["linear_error"] <= 1.0e-12
            and transfer[name]["constant_error"] <= 1.0e-12
        )
        checks[f"{name} target covers the source area"] = (
            transfer[name]["area_residual"] <= 1.0e-12
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise qualification_failure(
            "qualify_metric_adaptation",
            (f"Metric adaptation qualification failed: {failed}."),
            {"failed": failed},
        )
    return checks


def _metric_adaptation_record(
    execution: _MetricAdaptationExecution,
    measurements: _MetricAdaptationMeasurements,
    checks: dict[str, bool],
    /,
) -> dict[str, object]:
    """Construct the qualification JSON record from accepted measurements."""

    source, metric = execution["source"], execution["metric"]
    native, device = execution["native"], execution["device"]
    relocation = execution["relocation"]
    prepared, report = execution["prepared"], execution["report"]
    native_evidence = measurements["native_evidence"]
    device_evidence = measurements["device_evidence"]
    relocation_evidence = measurements["relocation_evidence"]
    adapted, target_metrics = measurements["adapted"], measurements["target_metrics"]
    before, after = measurements["before"], measurements["after"]
    relocated = measurements["relocated"]
    transfer, relocation_transfer = (
        measurements["transfer"],
        measurements["relocation_transfer"],
    )
    return {
        "status": "passed",
        "checks": sorted(checks),
        "source": {**_result_identity(source), "metric_id": metric.metric_id},
        "target": {
            name: {
                **_result_identity(result.target),
                "metric_id": target_metrics[name].metric_id,
            }
            for name, result in (*adapted.items(), ("relocation", relocation))
        },
        "runtime": {
            **_runtime_identity(),
            "native_predicate_mode": native_evidence.predicate_mode.value,
            "device_layout_id": prepared.layout.signature_id,
        },
        "quality": {
            "before": {**_quality_summary(source.quality), "metric": before},
            "after": {
                name: {**_quality_summary(result.target.quality), "metric": after[name]}
                for name, result in adapted.items()
            },
            "relocation": {
                **_quality_summary(relocation.target.quality),
                "metric": relocated,
            },
        },
        "transfer": {**transfer, "relocation": relocation_transfer},
        "conservation": {
            name: {
                "area_residual": record["area_residual"],
                "linear_integral_residual": record["linear_integral_residual"],
            }
            for name, record in (*transfer.items(), ("relocation", relocation_transfer))
        },
        "routes": {
            "native": {
                "adaptation_status": native.status.value,
                "evidence": _metric_operation_evidence(native_evidence),
            },
            "device": {
                "adaptation_status": device.status.value,
                "evidence": {
                    **_metric_operation_evidence(device_evidence),
                    "rejected_cavity": device_evidence.rejected_cavity,
                    "status_flags": int(device_evidence.status),
                    "status_names": device_evidence.status.name,
                },
                "report": {
                    "status_flags": int(report.status),
                    "passes": int(report.passes),
                    "unit_fraction": float(report.unit_fraction),
                },
            },
            "relocation": {
                "adaptation_status": relocation.status.value,
                "evidence": _metric_operation_evidence(relocation_evidence),
            },
        },
        "resources": {
            **execution["resources"],
            "device_capacity": {
                "vertices": prepared.layout.vertex_capacity,
                "cells": prepared.layout.cell_capacity,
            },
            "device_state_bytes": sum(
                leaf.nbytes for leaf in jax.tree.leaves(prepared.state)
            ),
        },
    }


def qualify_metric_adaptation() -> dict[str, object]:
    execution = _execute_metric_adaptation()
    measurements = _measure_metric_adaptation(execution)
    checks = _accept_metric_adaptation(execution, measurements)
    return _metric_adaptation_record(execution, measurements, checks)


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_metric_adaptation()
