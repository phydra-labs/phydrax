# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import Any

import jax
import numpy as np

import phydrax as phx
from phydrax import model_array_bytes
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._records import _quality_summary, _result_identity
from tools.meshing_qualification._runtime import (
    _contract,
    _measured,
    _runtime_identity,
    qualification_failure,
)
from tools.meshing_qualification.scenarios.bisection import (
    _bisection_adapt,
    _bisection_cell_ids,
    _bisection_corner_cells,
    _bisection_evidence,
    _bisection_lineage,
    _bisection_meshes,
    _bisection_transfer,
)
from tools.meshing_qualification.scenarios.providers import _mesh_scope


def _bisection_target_bytes(
    result: phx.meshing.MeshAdaptationResult, /
) -> dict[str, object]:
    """Committed arrays and identities compared byte for byte across routes."""

    mesh = result.target.mesh
    block = mesh.blocks[0]
    lineage, stencil, transfer = result.lineage, result.stencil, result.transfer
    hierarchy, evidence = result.hierarchy, result.evidence
    if (
        lineage is None
        or stencil is None
        or transfer is None
        or not isinstance(hierarchy, phx.meshing.BisectionHierarchy)
        or not isinstance(evidence, phx.meshing.BisectionEvidence)
    ):
        raise qualification_failure(
            "_bisection_target_bytes",
            (
                "Bisection parity requires committed lineage for every entity, hierarchy, and transfer evidence."
            ),
            {
                "evidence": evidence,
                "hierarchy": hierarchy,
                "lineage": lineage,
                "stencil": stencil,
                "transfer": transfer,
            },
        )
    return {
        "status": result.status.value,
        "topology_id": mesh.topology_id,
        "coordinates": np.asarray(mesh.coordinates).tobytes(),
        "vertex_global_ids": np.asarray(mesh.vertex_global_ids).tobytes(),
        "cells": np.asarray(block.vertices).tobytes(),
        "cell_global_ids": np.asarray(block.global_ids).tobytes(),
        "lineage_id": lineage.lineage_id,
        "hierarchy_id": hierarchy.hierarchy_id,
        "stencil_id": stencil.stencil_id,
        "transfer_id": transfer.transfer_id,
        "rejected_coarsening_ids": np.asarray(evidence.rejected_coarsening_ids).tobytes(),
    }


def _bisection_same_target(host: Any, device: Any, /) -> list[str]:

    first, second = _bisection_target_bytes(host), _bisection_target_bytes(device)
    differences = [key for key in first if first[key] != second[key]]
    if differences:
        raise qualification_failure(
            "_bisection_same_target",
            (f"DEVICE_BISECTION differs from NATIVE_BISECTION: {differences}"),
            {"differences": differences},
        )
    return list(first)


def _bisection_simplex_report(report: Any, /) -> dict[str, object]:
    status = phx.discretization.AdaptiveSimplexStatus(int(report.status))
    return {
        "status": int(status),
        "flags": [
            flag.name
            for flag in phx.discretization.AdaptiveSimplexStatus
            if flag & status
        ],
        "failed": bool(report.failed),
        "requested": int(report.requested),
        "accepted": int(report.accepted),
        "rejected": int(np.count_nonzero(np.asarray(report.rejected))),
        "operations": int(report.operations),
        "iterations": int(report.iterations),
        "vertices": int(report.vertices),
        "uncertain_cells": int(report.uncertain_cells),
        "invalid_cells": int(report.invalid_cells),
        "minimum_quality": float(report.minimum_quality),
    }


def _bisection_device_parity(
    mesh: phx.discretization.CellMesh,
    capacity: tuple[int, int],
    /,
) -> dict[str, object]:
    """Host and device routes for the same marks, plus one explicit device epoch."""

    host_route = phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
    device_route = phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION
    device_policy = phx.discretization.AdaptiveSimplexPolicy(
        vertex_capacity=capacity[0], cell_capacity=capacity[1]
    )
    source = phx.meshing.certify_cell_mesh(mesh, _contract())
    marks = np.sort(_bisection_cell_ids(mesh))[::3]
    host, host_first = _bisection_adapt(
        host_route, source, marks, np.empty((0,), dtype=np.int64), None
    )
    device, device_first = _bisection_adapt(
        device_route,
        source,
        marks,
        np.empty((0,), dtype=np.int64),
        None,
        device_policy=device_policy,
    )
    compared = _bisection_same_target(host, device)
    refine = _bisection_corner_cells(host.target.mesh)
    coarsen = np.setdiff1d(np.sort(_bisection_cell_ids(host.target.mesh))[-24:], refine)
    host_second, host_stages = _bisection_adapt(
        host_route, host.target, refine, coarsen, host.hierarchy
    )
    device_second, device_stages = _bisection_adapt(
        device_route,
        device.target,
        refine,
        coarsen,
        device.hierarchy,
        device_policy=device_policy,
    )
    _bisection_same_target(host_second, device_second)
    policy = phx.meshing.MeshAdaptationPolicy(device_route, device_policy=device_policy)
    hierarchy = device.hierarchy
    if not isinstance(hierarchy, phx.meshing.BisectionHierarchy):
        raise qualification_failure(
            "_bisection_device_parity",
            ("Device bisection did not retain its hierarchy."),
            {"hierarchy": hierarchy},
        )
    epoch, preparation = _measured(
        lambda: phx.meshing.prepare_adaptive_simplex(
            device.target,
            policy=policy,
            hierarchy=hierarchy,
        )
    )

    layout = epoch.layout
    refined, refinement = _measured(
        lambda: jax.block_until_ready(
            phx.discretization.refine_adaptive_simplex(
                layout,
                epoch.state,
                epoch.cell_marks(refine),
            )
        )
    )
    coarsened, coarsening = _measured(
        lambda: jax.block_until_ready(
            phx.discretization.coarsen_adaptive_simplex(
                layout,
                refined.state,
                epoch.cell_marks(coarsen),
            )
        )
    )
    committed, commit = _measured(
        lambda: phx.meshing.commit_adaptive_simplex(epoch, coarsened.state)
    )
    _bisection_same_target(host_second, committed)
    reports = {
        "refine": _bisection_simplex_report(refined.report),
        "coarsen": _bisection_simplex_report(coarsened.report),
    }
    if any(report["failed"] for report in reports.values()):
        raise qualification_failure(
            "_bisection_device_parity",
            ("The explicit device epoch refused an operation."),
            {"reports": reports},
        )
    return {
        "source": _result_identity(source),
        "target": _result_identity(device_second.target),
        "statuses": {
            "first": [host.status.value, device.status.value],
            "second": [host_second.status.value, device_second.status.value],
            "explicit_epoch": committed.status.value,
        },
        "byte_identical_fields": compared,
        "quality": {
            "before": _quality_summary(source.quality),
            "after": _quality_summary(device_second.target.quality),
        },
        "transfer": {
            "first": _bisection_transfer(device),
            "second": _bisection_transfer(device_second),
        },
        "lineage": _bisection_lineage(device_second),
        "evidence": {
            "host": _bisection_evidence(host_second.evidence),
            "device": _bisection_evidence(device_second.evidence),
        },
        "device_reports": reports,
        "layout": {
            "signature_id": layout.signature_id,
            "dimension": layout.dimension,
            "vertex_capacity": layout.vertex_capacity,
            "cell_capacity": layout.cell_capacity,
            "protected_edge_capacity": layout.protected_edge_capacity,
            "maximum_closure_iterations": layout.maximum_closure_iterations,
            "maximum_coarsening_passes": layout.maximum_coarsening_passes,
        },
        "resources": {
            "host_first": host_first,
            "device_first": device_first,
            "host_second": host_stages,
            "device_second": device_stages,
            "epoch_prepare": preparation,
            "epoch_refine": refinement,
            "epoch_coarsen": coarsening,
            "epoch_commit": commit,
            "epoch_state_bytes": model_array_bytes(coarsened.state),
            "transfer_retained_bytes": model_array_bytes(device_second.transfer),
        },
    }


def _bisection_metric_measurements(
    evidence: phx.meshing.LocalMetricEvidence | phx.meshing.DeviceMetricEvidence, /
) -> dict[str, object]:
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


def _bisection_metric_comparison(
    mesh: phx.discretization.CellMesh, /
) -> dict[str, object]:
    """Host and device planar metric adaptation toward one isotropic size 0.2."""

    source = phx.meshing.certify_cell_mesh(mesh, _contract())
    vertices = np.asarray(source.mesh.entity_set(0).entity_ids)
    metric = phx.meshing.MeshMetricField(
        _mesh_scope(source.mesh, 0, vertices),
        np.broadcast_to(np.eye(2) / 0.2**2, (vertices.shape[0], 2, 2)),
        minimum_size=1.0e-3,
        maximum_size=10.0,
    )
    request = phx.meshing.MetricMeshAdaptation(metric)
    native_policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_METRIC_2D
    )
    native, native_seconds = _measured(
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
    prepared, preparation = _measured(
        lambda: phx.meshing.prepare_device_metric_adaptation(
            source, request, policy=device_policy
        )
    )
    update, adaptation = _measured(
        lambda: jax.block_until_ready(
            phx.meshing.adapt_device_metric(prepared.layout, prepared.state)
        )
    )

    report = jax.device_get(update.report)
    device, commit = _measured(
        lambda: phx.meshing.commit_device_metric_adaptation(prepared, update.state)
    )
    native_evidence, device_evidence = native.evidence, device.evidence
    if not isinstance(native_evidence, phx.meshing.LocalMetricEvidence):
        raise qualification_failure(
            "_bisection_metric_comparison",
            ("Native metric comparison requires local metric operation evidence."),
            {"native_evidence": native_evidence},
        )
    if not isinstance(device_evidence, phx.meshing.DeviceMetricEvidence):
        raise qualification_failure(
            "_bisection_metric_comparison",
            ("Device metric comparison requires device metric operation evidence."),
            {"device_evidence": device_evidence},
        )
    complete = phx.meshing.MeshAdaptationStatus.COMPLETE
    if (
        bool(report.failed)
        or device.status is not complete
        or not device_evidence.converged
        or not native.target.audit.passed
        or not device.target.audit.passed
    ):
        raise qualification_failure(
            "_bisection_metric_comparison",
            ("DEVICE_METRIC_2D did not reach a certified unit mesh."),
            {
                "complete": complete,
                "device": device,
                "device_evidence": device_evidence,
                "native": native,
                "report": report,
            },
        )
    status = phx.discretization.AdaptiveSimplexStatus(int(report.status))
    return {
        "source": _result_identity(source),
        "target": {
            "native": _result_identity(native.target),
            "device": _result_identity(device.target),
        },
        "statuses": {"native": native.status.value, "device": device.status.value},
        "quality": {
            "before": _quality_summary(source.quality),
            "after": {
                "native": _quality_summary(native.target.quality),
                "device": _quality_summary(device.target.quality),
            },
        },
        "measurements": {
            "native": _bisection_metric_measurements(native_evidence),
            "device": _bisection_metric_measurements(device_evidence),
        },
        "device_report": {
            "status": int(status),
            "flags": [
                flag.name
                for flag in phx.discretization.AdaptiveSimplexStatus
                if flag & status
            ],
            "passes": int(report.passes),
            "splits": int(report.splits),
            "collapses": int(report.collapses),
            "flips": int(report.flips),
            "relocations": int(report.relocations),
            "rejected_cavity": int(report.rejected_cavity),
            "unit_fraction": float(report.unit_fraction),
            "minimum_metric_quality": float(report.minimum_metric_quality),
        },
        "layout": {
            "signature_id": prepared.layout.signature_id,
            "vertex_capacity": prepared.layout.vertex_capacity,
            "cell_capacity": prepared.layout.cell_capacity,
            "record_capacity": prepared.layout.record_capacity,
            "ball_width": prepared.layout.ball_width,
        },
        "transfer": {
            "native": _bisection_transfer(native),
            "device": _bisection_transfer(device),
        },
        "resources": {
            "native_adaptation": native_seconds,
            "device_prepare": preparation,
            "device_adapt": adaptation,
            "device_commit": commit,
            "device_state_bytes": model_array_bytes(update.state),
        },
    }


def qualify_device_adaptation() -> dict[str, object]:
    """DEVICE_BISECTION commits NATIVE_BISECTION byte for byte; metric routes agree."""

    meshes = _bisection_meshes()
    bisection = {
        "triangles": _bisection_device_parity(meshes["triangles"], (256, 512)),
        "tetrahedra": _bisection_device_parity(meshes["tetrahedra"], (256, 1024)),
    }
    metric = _bisection_metric_comparison(meshes["triangles"])
    return {
        "status": "passed",
        "source": {
            **{name: record["source"] for name, record in bisection.items()},
            "metric": metric["source"],
        },
        "target": {
            **{name: record["target"] for name, record in bisection.items()},
            "metric": metric["target"],
        },
        "quality": {
            **{name: record["quality"] for name, record in bisection.items()},
            "metric": metric["quality"],
        },
        "transfer": {
            **{name: record["transfer"] for name, record in bisection.items()},
            "metric": metric["transfer"],
        },
        "resources": {
            **{name: record["resources"] for name, record in bisection.items()},
            "metric": metric["resources"],
        },
        "bisection": bisection,
        "metric": metric,
        "runtime": _runtime_identity(),
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_device_adaptation()
