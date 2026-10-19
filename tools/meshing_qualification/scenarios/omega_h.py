# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import shutil
from typing import final, TypedDict

import numpy as np

import phydrax as phx
from phydrax import model_array_bytes
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._records import _quality_summary, _result_identity
from tools.meshing_qualification._runtime import (
    _contract,
    _measured,
    _missing_dependency,
    _runtime_identity,
    qualification_failure,
    StageLedger,
)
from tools.meshing_qualification.scenarios.distribution import (
    _distribution_cells,
    _distribution_lattice,
    _distribution_rows,
)
from tools.meshing_qualification.scenarios.providers import _mesh_scope


_OMEGA_H_WORKER_HINT = (
    "Build native/providers/omega_h against an Omega_h CMake package (MPI-enabled "
    "for omega-h-distributed; docs/guides_meshing.md), then pass --worker, set "
    "PHYDRAX_OMEGA_H_WORKER, or put phydrax-omega-h-worker on PATH."
)


def _omega_h_worker(worker: str | None, /) -> str | None:
    """Resolve the Omega_h worker exactly as ``OmegaHProvider`` does."""
    configured = (
        phx.config.snapshot().workers.get("omega_h") if worker is None else worker
    )
    return shutil.which("phydrax-omega-h-worker" if configured is None else configured)


@final
class _OmegaHRequest(phx.StrictModule):
    source: phx.meshing.CellMeshingResult
    metric: phx.meshing.MeshMetricField
    fields: tuple[phx.meshing.OmegaHField, ...]
    cell_ids: np.ndarray
    areas: np.ndarray
    density: np.ndarray
    gradient: np.ndarray
    certify_stage: dict[str, float]


@final
class _OmegaHExecution(phx.StrictModule):
    result: phx.meshing.OmegaHAdaptationResult
    adapt_stage: dict[str, float]


class _OmegaHPartitionRecord(TypedDict):
    rank: int
    partition_id: str
    owned_cells: int
    ghost_cells: int
    owned_vertices: int
    ghost_vertices: int


@final
class _OmegaHAcceptance(phx.StrictModule):
    target: phx.meshing.CellMeshingResult
    linear_error: float
    mass_before: float
    mass_after: float
    mass_residual: float
    reported_before: float
    reported_after: float
    left_area: float
    partitions: list[_OmegaHPartitionRecord]
    checks: dict[str, bool | np.bool_]


def _prepare_omega_h_request(ledger: StageLedger) -> _OmegaHRequest:
    """Certify the source and prepare metric, transfer fields, and independent baselines."""
    cells_per_axis = 4
    mesh = _distribution_lattice(cells_per_axis)
    cell_ids, areas, centroids = _distribution_cells(mesh)
    zones = tuple(
        phx.meshing.MeshZone(
            name, phx.meshing.MeshZoneRole.REGION, _mesh_scope(mesh, 2, cell_ids[mask])
        )
        for name, mask in (
            ("left", centroids[:, 0] < 0.5),
            ("right", centroids[:, 0] > 0.5),
        )
    )
    source, certify_stage = ledger.run(
        "certify_cell_mesh",
        lambda: _measured(
            lambda: phx.meshing.certify_cell_mesh(mesh, _contract(), zones=zones)
        ),
    )
    vertex_order = np.argsort(np.asarray(mesh.vertex_global_ids))
    vertex_ids = np.asarray(mesh.vertex_global_ids)[vertex_order]
    points = np.asarray(mesh.coordinates)[vertex_order]
    matrix = np.eye(2) * (2.0 * cells_per_axis) ** 2
    matrix[0, 1] = matrix[1, 0] = 0.15 * matrix[0, 0]
    metric = phx.meshing.MeshMetricField(
        _mesh_scope(mesh, 0, vertex_ids),
        np.broadcast_to(matrix, (points.shape[0], 2, 2)),
        minimum_size=1.0 / (3.0 * cells_per_axis),
        maximum_size=1.0 / cells_per_axis,
        maximum_anisotropy=2.0,
    )
    gradient = np.asarray((1.0, 2.0))
    cell_order = np.argsort(cell_ids)
    density = 1.0 + np.where(centroids[:, 0] < 0.5, 0.0, 4.0) + centroids[:, 1] ** 2
    fields = (
        phx.meshing.OmegaHField(
            "potential",
            phx.meshing.OmegaHFieldTransfer.LINEAR,
            _mesh_scope(mesh, 0, vertex_ids),
            points @ gradient,
        ),
        phx.meshing.OmegaHField(
            "density",
            phx.meshing.OmegaHFieldTransfer.CONSERVE,
            _mesh_scope(mesh, 2, cell_ids[cell_order]),
            density[cell_order],
        ),
    )
    return _OmegaHRequest(
        source=source,
        metric=metric,
        fields=fields,
        cell_ids=cell_ids,
        areas=areas,
        density=density,
        gradient=gradient,
        certify_stage=certify_stage,
    )


def _execute_omega_h_request(
    executable: str, ranks: int, request: _OmegaHRequest, ledger: StageLedger
) -> _OmegaHExecution:
    """Measure gathered adaptation within the provider session's lifetime."""
    with phx.meshing.OmegaHProvider(executable) as provider:
        result, adapt_stage = ledger.run(
            "execute",
            lambda: _measured(
                lambda: provider.execute(
                    request.source,
                    request.metric,
                    fields=request.fields,
                    ranks=ranks,
                    gather=True,
                )
            ),
        )
    return _OmegaHExecution(result=result, adapt_stage=adapt_stage)


def _accept_omega_h_result(
    request: _OmegaHRequest, execution: _OmegaHExecution, ranks: int
) -> _OmegaHAcceptance:
    """Accept transfer, classification, metric, and rank evidence against the source."""
    result = execution.result
    cell_ids, areas, density, gradient = (
        request.cell_ids,
        request.areas,
        request.density,
        request.gradient,
    )
    target = result.target
    if target is None:
        raise qualification_failure(
            "_qualify_omega_h_ranks",
            ("Gathered Omega_h qualification did not publish its target mesh."),
            {"target": target},
        )
    target_ids, target_areas, _ = _distribution_cells(target.mesh)

    potential, transferred = result.fields
    target_points = np.asarray(target.mesh.coordinates)[
        np.argsort(np.asarray(target.mesh.vertex_global_ids))
    ]
    linear_error = float(
        np.max(np.abs(np.asarray(potential.values) - target_points @ gradient))
    )
    rows = _distribution_rows(np.asarray(transferred.scope.entity_ids), target_ids)
    mass_before = float(np.sum(density * areas))
    mass_after = float(np.sum(np.asarray(transferred.values)[rows] * target_areas))
    mass_residual = abs(mass_after - mass_before) / abs(mass_before)

    conserve = result.evidence.fields[1]
    reported_before = float(np.sum(np.asarray(conserve.integral_before)))
    reported_after = float(np.sum(np.asarray(conserve.integral_after)))
    left = next(zone for zone in target.zones if zone.name == "left")
    left_area = float(
        np.sum(target_areas[np.isin(target_ids, np.asarray(left.scope.entity_ids))])
    )
    partitions: list[_OmegaHPartitionRecord] = [
        {
            "rank": partition.rank,
            "partition_id": partition.partition_id,
            "owned_cells": int(np.count_nonzero(np.asarray(partition.cell_owned))),
            "ghost_cells": int(np.count_nonzero(np.asarray(partition.cell_ghosts))),
            "owned_vertices": int(np.count_nonzero(np.asarray(partition.vertex_owned))),
            "ghost_vertices": int(np.count_nonzero(np.asarray(partition.vertex_ghosts))),
        }
        for partition in result.partitions
    ]
    owned = [
        int(np.count_nonzero(np.asarray(partition.cell_owned)))
        for partition in result.partitions
    ]
    target_metric = result.metric
    if target_metric is None:
        raise qualification_failure(
            "_qualify_omega_h_ranks",
            ("Omega_h metric adaptation omitted its gathered target metric."),
            {"target_metric": target_metric},
        )
    checks: dict[str, bool | np.bool_] = {
        "audit": target.audit.passed,
        "positive_cells": bool(np.all(target_areas > 0.0)),
        "domain_area": bool(np.isclose(np.sum(target_areas), 1.0, rtol=0.0, atol=1e-12)),
        "refined": target_ids.size > cell_ids.size,
        "lineage_unknown": result.lineage_status == "unknown",
        "zone_area": bool(np.isclose(left_area, 0.5, rtol=0.0, atol=1e-12)),
        "linear_reproduction": linear_error <= 1e-12,
        "conservation": mass_residual <= 1e-12,
        "reported_conservation": np.isclose(
            reported_after, reported_before, rtol=1e-12, atol=0.0
        ),
        "ownership": sum(owned) == target_ids.size and min(owned) > 0,
        "ghosts": ranks == 1
        or all(
            np.count_nonzero(np.asarray(partition.cell_ghosts)) > 0
            for partition in result.partitions
        ),
        "ranks": result.evidence.ranks == ranks and len(partitions) == ranks,
        "distribution": (result.distribution is None) == (ranks == 1),
        "metric": bool(np.all(np.linalg.eigvalsh(np.asarray(target_metric.values)) > 0)),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise qualification_failure(
            "_qualify_omega_h_ranks",
            (f"Omega_h qualification failed checks: {failed}."),
            {"failed": failed},
        )
    return _OmegaHAcceptance(
        target=target,
        linear_error=linear_error,
        mass_before=mass_before,
        mass_after=mass_after,
        mass_residual=mass_residual,
        reported_before=reported_before,
        reported_after=reported_after,
        left_area=left_area,
        partitions=partitions,
        checks=checks,
    )


def _omega_h_result_record(
    executable: str,
    request: _OmegaHRequest,
    execution: _OmegaHExecution,
    acceptance: _OmegaHAcceptance,
    measurements: dict[str, object],
) -> dict[str, object]:
    """Publish accepted evidence in the qualification record's canonical ordering."""
    source, result, target = request.source, execution.result, acceptance.target
    record: dict[str, object] = {
        "status": "passed",
        "source": _result_identity(source),
        "target": {
            **_result_identity(target),
            "omega_h_result_id": result.result_id,
            "lineage_status": result.lineage_status,
        },
        "runtime": {
            **_runtime_identity(),
            "provider_version": result.provider.version,
            "actual_version": result.runtime.actual_version,
            "worker_executable": executable,
            "worker_identity": result.evidence.identity_id,
            "worker_session": result.evidence.session_id,
            "ranks": result.evidence.ranks,
        },
        "quality": {
            "before": _quality_summary(source.quality),
            "after": _quality_summary(target.quality),
            "omega_h_metric_quality": [
                result.evidence.minimum_quality,
                result.evidence.maximum_quality,
            ],
            "omega_h_metric_length": [
                result.evidence.minimum_length,
                result.evidence.maximum_length,
            ],
            "iterations": result.evidence.iterations,
        },
        "transfer": {
            "methods": {value.name: value.method for value in result.evidence.fields},
            "linear_reproduction_error": acceptance.linear_error,
        },
        "conservation": {
            "integral_before": acceptance.mass_before,
            "integral_after": acceptance.mass_after,
            "relative_residual": acceptance.mass_residual,
            "omega_h_integral_before": acceptance.reported_before,
            "omega_h_integral_after": acceptance.reported_after,
        },
        "classification": {
            "left_zone_area": acceptance.left_area,
            "zones": len(target.zones),
        },
        "partitions": acceptance.partitions,
        "checks_passed": sorted(acceptance.checks),
        "resources": {
            "certification": request.certify_stage,
            "adaptation": execution.adapt_stage,
            "worker_peak_rss_bytes": result.evidence.peak_rss_bytes,
            "target_retained_bytes": model_array_bytes(target),
        },
    }
    record["stage_measurements"] = measurements["stage_measurements"]

    if result.distribution is not None:
        distribution = result.distribution.evidence
        record["distribution"] = {
            "distribution_id": result.distribution.distribution_id,
            "provenance": distribution.provenance,
            "part_weights": np.asarray(distribution.part_weights).tolist(),
            "imbalance": float(distribution.imbalance),
            "edge_cut": int(distribution.edge_cut),
            "halo_replicas": int(distribution.halo_replicas),
        }
    return record


def _qualify_omega_h_ranks(worker: str | None, ranks: int, /) -> dict[str, object]:
    """Metric adaptation with LINEAR and CONSERVE fields on ``ranks`` MPI ranks."""
    measurements: dict[str, object] = {}
    ledger = StageLedger(measurements)

    executable = _omega_h_worker(worker)
    if executable is None:
        return _missing_dependency("phydrax-omega-h-worker", _OMEGA_H_WORKER_HINT)
    if ranks > 1 and shutil.which("mpiexec") is None:
        return _missing_dependency(
            "mpiexec",
            "Distributed Omega_h adaptation launches its worker under the MPI "
            "launcher of the MPI implementation Omega_h was built with.",
        )
    request = _prepare_omega_h_request(ledger)
    execution = _execute_omega_h_request(executable, ranks, request, ledger)
    acceptance = _accept_omega_h_result(request, execution, ranks)
    return _omega_h_result_record(
        executable, request, execution, acceptance, measurements
    )


def qualify_omega_h(worker: str | None) -> dict[str, object]:
    return _qualify_omega_h_ranks(worker, 1)


def qualify_omega_h_distributed(worker: str | None) -> dict[str, object]:
    return _qualify_omega_h_ranks(worker, 2)


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    match args.scenario:
        case "omega-h":
            return qualify_omega_h(args.worker)
        case "omega-h-distributed":
            return qualify_omega_h_distributed(args.worker)
        case _:
            raise ValueError(
                f"qualification.scenario must match this module; got {args.scenario!r}"
            )
