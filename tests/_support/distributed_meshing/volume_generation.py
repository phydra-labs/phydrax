# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""PLC coverage admission observations in an isolated process."""

from typing import Any

import numpy as np

import phydrax as phx
from phydrax._meshcore import NativeExecutionScope
from phydrax.geometry._embedding_state import coordinate_certificate_request
from phydrax.meshing._volume_generation import NativeVolumeSchedule, prepare_plc_source
from tests._support.meshing_fixtures import volume_specification


M = phx.meshing
_SI = phx.SpatialCoordinateContract.si()


# Nondyadic corners: every facet has its own rational plane, and the red-refined
# affine carrier retains the PLC source.
_NONDYADIC_TETRAHEDRON = np.asarray(
    ((0.1, 0.0, 0.0), (1.0, 0.3, 0.0), (0.0, 0.7, 0.1), (0.2, 0.1, 0.9)),
    dtype=np.float64,
)


def _mixed_refinement_of_nondyadic_plc(
    source_limits: M.MeshingLimits,
    *,
    adaptation_limits: M.MeshingLimits | None = None,
) -> tuple[M.CellMeshingResult, M.MeshAdaptationResult]:
    complex_ = M.PiecewiseLinearComplex(
        _NONDYADIC_TETRAHEDRON,
        tuple(
            np.asarray(row, dtype=np.int64)
            for row in ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
        ),
        np.arange(4, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (4, 1)),
        ("solid",),
        boundary="conforming",
    )
    source = M.NativePlcSource(
        complex_, "nondyadic-coverage-source", "nondyadic-revision"
    )
    result = (
        M.NativeMeshingProvider(
            M.NativeMeshingOptions(
                "plc_tetrahedral",
                volume_schedule=NativeVolumeSchedule(
                    refinement_rounds=1,
                    improvement_passes=1,
                    minimum_dihedral_degrees=0.0,
                ),
            )
        )
        .plan(
            source,
            volume_specification(source, size=4.0, limits=source_limits),
            coordinate_contract=_SI,
        )
        .execute()
    )
    adaptation_limits = (
        M.MeshingLimits(maximum_vertices=100, maximum_cells=1000)
        if adaptation_limits is None
        else adaptation_limits
    )
    with NativeExecutionScope(adaptation_limits.native_limits()) as scope:
        transfer = prepare_plc_source(
            complex_,
            source.source_id,
            source.source_revision,
            _SI,
            limits=adaptation_limits,
            scope=scope,
        ).association_transfer
    adaptation = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            result,
            M.MarkedMeshAdaptation(np.asarray(result.mesh.blocks[0].global_ids)),
            policy=M.MeshAdaptationPolicy(
                M.MeshAdaptationRoute.NATIVE_MIXED,
                limits=adaptation_limits,
                association_transfer=transfer,
                audit_policy=M.CellMeshAuditPolicy(
                    require_complete_association=True,
                    watertight_boundary=M.CellMeshAuditDisposition.REJECT,
                ),
            ),
        )
    )
    return result, adaptation


def coverage_threshold(shortfall: str) -> dict[str, Any]:
    generous = M.MeshingLimits(maximum_vertices=100, maximum_cells=1000)
    _, adaptation = _mixed_refinement_of_nondyadic_plc(generous)
    target = adaptation.target
    certificate = target.certification
    if certificate is None or certificate.embedding is None:
        raise AssertionError(
            "coverage_threshold.certification must retain an embedding certificate; got missing certification or embedding."
        )
    request = certificate.request
    if request is None or request.domain is None or request.cell_regions is None:
        raise AssertionError(
            "coverage_threshold.request must retain declared coverage inputs; got missing domain or cell regions."
        )
    # A phase requirement excludes coordinate-scope preparation. Measure the entire
    # public invocation independently instead of using it as an adaptation limit.
    with coordinate_certificate_request(request.limits) as budget:
        baseline = phx.geometry.certify_domain_coverage(
            target.mesh,
            target.geometry,
            request.domain,
            request.cell_regions,
            embedding=certificate.embedding,
            limits=request.limits,
            budget=budget,
        )
        required = budget.required_work_units
    if baseline.status != "certified":
        raise AssertionError(
            f"coverage_threshold.baseline.status must be certified; got {baseline.status!r}."
        )
    limits = phx.geometry.MeshCertificateLimits(
        maximum_work_units=required - int(shortfall),
        maximum_scratch_bytes=request.limits.maximum_scratch_bytes,
    )
    with coordinate_certificate_request(request.limits) as budget:
        report = phx.geometry.certify_domain_coverage(
            target.mesh,
            target.geometry,
            request.domain,
            request.cell_regions,
            embedding=certificate.embedding,
            limits=limits,
            budget=budget,
        )
        charged = budget.work_units
    if report.status != "certified":
        finding = next(
            item
            for item in report.findings
            if item.check == "mapped_coverage_resource_budget"
        )
        requested = dict(finding.requested)
        return {
            "status": "RESOURCE",
            "resource": finding.resource,
            "limit": requested["limit"],
            "requested": requested["requested"],
        }
    if (
        report.source_expression_required_work_units
        != baseline.source_expression_required_work_units
        or report.source_expression_work_units != baseline.source_expression_work_units
    ):
        raise AssertionError(
            "coverage_threshold.phase must retain its measured work counters; got changed phase requirements or charges."
        )
    return {
        "status": "ACCEPT",
        "required_work": required,
        "charged_work": charged,
        "phase_required_work": report.source_expression_required_work_units,
        "phase_charged_work": report.source_expression_work_units,
    }
