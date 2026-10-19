# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import hashlib
import resource
from dataclasses import asdict
from time import perf_counter
from typing import Any

import numpy as np

import phydrax as phx
from phydrax import array_tree_fingerprint, canonical_fingerprint
from phydrax.geometry.reconstruction import ReconstructionFailure
from phydrax.geometry.simplicial import TriangleBVH, TriangleMesh
from phydrax.meshing import (
    declared_plc_domain,
    NativeMeshingPhaseMeasurement,
    NativeVolumeSchedule,
)
from tools.meshing_qualification._controls import (
    capacity_meshing_limits,
    QualificationControls,
)
from tools.meshing_qualification._records import (
    _phase_recordings,
    _result_identity,
    record_failure,
)
from tools.meshing_qualification._runtime import (
    _measured,
    _runtime_identity,
    Deadline,
    peak_resident_bytes,
    qualification_failure,
    StageLedger,
)
from tools.meshing_qualification.scenarios.envelope import _envelope_qualification_epochs
from tools.meshing_qualification.scenarios.triangulation import _simplex_measures


def _reconstruction_qualification_source(
    resolution: int,
    capacity: int,
) -> tuple[
    phx.geometry.ReconstructedGeometrySource, phx.meshing.NativePlcSource, np.ndarray
]:
    """Reconstruct a noisy, nonspherical cloud with native numerical owners."""

    count = 8 * resolution * resolution
    index = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / count
    angle = np.pi * (3.0 - np.sqrt(5.0)) * index
    radial = np.sqrt(1.0 - z * z)
    direction = np.stack((radial * np.cos(angle), radial * np.sin(angle), z), axis=1)
    radius = 0.2 * (1.0 + 0.04 * np.sin(3.0 * angle) * radial * radial)
    noise = np.random.default_rng(1709).normal(0.0, 0.0008, size=direction.shape)
    points = (
        direction * radius[:, None]
        + noise
        + np.asarray((0.07, -0.03, 0.02), dtype=np.float64)
    )
    reconstructed = phx.geometry.reconstruct_surface_region(
        points,
        recenter=False,
        sample_spacing=0.44 / resolution,
        maximum_grid_nodes=5 * capacity,
        discretization="octree",
        robustness=phx.geometry.ReconstructionRobustness(
            incomplete_sampling="refuse",
            coverage_distance=0.44 / resolution,
            thin_feature_distance=0.22 / resolution,
        ),
        feature_id="qualification-native-reconstructed-samples",
    )
    child = reconstructed.source
    if not isinstance(child, phx.geometry.MeshRegion):
        raise TypeError(
            "qualification.reconstruction.source must be a MeshRegion; "
            f"got {type(child).__name__}"
        )
    vertices = np.asarray(child.vertices, dtype=np.float64)
    faces = np.asarray(child.faces, dtype=np.int64)
    revision = canonical_fingerprint(
        {
            "kind": "native-reconstructed-plc-source",
            "report": asdict(reconstructed.report),
            "vertices": array_tree_fingerprint(vertices),
            "triangles": array_tree_fingerprint(faces),
        }
    )
    complex_ = phx.meshing.PiecewiseLinearComplex(
        vertices,
        tuple(faces),
        np.arange(faces.shape[0], dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (faces.shape[0], 1)),
        ("reconstructed-solid",),
        boundary="conforming",
    )
    return (
        reconstructed,
        phx.meshing.NativePlcSource(complex_, child.feature_id, revision),
        points,
    )


def _reconstruction_qualification_request(
    source: phx.meshing.NativePlcSource,
    capacity: int,
    timeout: float,
) -> tuple[phx.meshing.VolumeMeshingSpec, phx.meshing.NativeMeshingOptions]:
    """Freeze the envelope profile on the extracted PLC."""

    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "reconstructed-facets",
        np.arange(source.complex.facet_count, dtype=np.int64),
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 2.0, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        limits=capacity_meshing_limits(
            capacity, timeout, cells=5, work_units=1000, cavity_cells=None
        ),
    )
    return specification, phx.meshing.NativeMeshingOptions(
        "plc_tetrahedral",
        volume_schedule=NativeVolumeSchedule(
            refinement_rounds=1, improvement_passes=1, minimum_dihedral_degrees=0.0
        ),
    )


def _reconstruction_qualification_acceptance(
    reconstructed: phx.geometry.ReconstructedGeometrySource,
    source: phx.meshing.NativePlcSource,
    points: np.ndarray,
    result: phx.meshing.CellMeshingResult,
) -> dict[str, object]:
    """Separate sampled reconstruction evidence from extracted-domain fidelity."""

    report = reconstructed.report
    if (
        report.input_digest
        != hashlib.sha256(
            np.ascontiguousarray(points, dtype=np.float64).tobytes()
        ).hexdigest()
    ):
        raise qualification_failure(
            "_reconstruction_qualification_acceptance",
            ("Reconstruction report binds a different sample cloud."),
            {"points": points, "report": report},
        )
    if (
        not report.watertight
        or not report.winding_consistent
        or report.poisson_evidence is None
        or not report.poisson_evidence.solver_converged
        or report.component_evidence is None
        or report.component_evidence.unresolved_components
        or report.coverage_evidence is None
        or report.coverage_evidence.unsupported_triangles
        or report.deviation_evidence is None
        or report.input_points != points.shape[0]
    ):
        raise qualification_failure(
            "_reconstruction_qualification_acceptance",
            (
                "Reconstructed source lacks accepted native topology, solve, or sampling evidence."
            ),
            {"points": points, "report": report},
        )
    certificate = result.certification
    if (
        certificate is None
        or not certificate.passed
        or not result.audit.passed
        or not result.compliance.passed
    ):
        raise qualification_failure(
            "_reconstruction_qualification_acceptance",
            ("Reconstructed volume requires strict native acceptance."),
            {"certificate": certificate, "result": result},
        )
    domain = declared_plc_domain(source.complex, source.source_id)
    coverage, embedding, fidelity = (
        certificate.coverage,
        certificate.embedding,
        certificate.fidelity,
    )
    if (
        coverage is None
        or embedding is None
        or coverage.status != "certified"
        or embedding.status != "certified"
        or certificate.request.domain is None
        or certificate.request.domain.domain_id != domain.domain_id
        or coverage.covered_source_facet_count != coverage.source_facet_count
        or coverage.region_ids != ("reconstructed-solid",)
        or {zone.name for zone in result.zones} != {"reconstructed-solid"}
    ):
        raise qualification_failure(
            "_reconstruction_qualification_acceptance",
            ("Native fill lost reconstructed source-domain coverage or region identity."),
            {
                "certificate": certificate,
                "coverage": coverage,
                "domain": domain,
                "embedding": embedding,
                "result": result,
            },
        )
    faces = np.asarray(source.complex.polygon_vertices, dtype=np.int64).reshape((-1, 3))
    skin = source.complex.vertices[faces]
    surface_volume = float(np.sum(skin[:, 0] * np.cross(skin[:, 1], skin[:, 2])) / 6.0)
    mesh_points = np.asarray(result.mesh.coordinates, dtype=np.float64)
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in result.mesh.blocks]
    )
    volumes = _simplex_measures(mesh_points, cells)
    volume = float(np.sum(volumes))
    if (
        np.any(volumes <= 0.0)
        or surface_volume <= 0.0
        or abs(volume - surface_volume) > 1.0e-12
    ):
        raise qualification_failure(
            "_reconstruction_qualification_acceptance",
            ("Native reconstructed fill fails independent signed-volume closure."),
            {"surface_volume": surface_volume, "volume": volume, "volumes": volumes},
        )
    sample_distance = np.asarray(
        TriangleBVH(TriangleMesh(source.complex.vertices, faces)).query(points).distance,
        dtype=np.float64,
    )
    vertex_distance = np.min(
        np.linalg.norm(source.complex.vertices[:, None, :] - points[None, :, :], axis=-1),
        axis=1,
    )
    deviation = report.deviation_evidence
    if (
        abs(float(np.max(sample_distance)) - deviation.sample_to_surface_max) > 1.0e-12
        or abs(float(np.max(vertex_distance)) - deviation.surface_to_sample_max) > 1.0e-12
    ):
        raise qualification_failure(
            "_reconstruction_qualification_acceptance",
            (
                "Reconstruction report disagrees with independent sample-cloud distance queries."
            ),
            {
                "deviation": deviation,
                "sample_distance": sample_distance,
                "vertex_distance": vertex_distance,
            },
        )
    return {
        "passed": True,
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "input_digest": report.input_digest,
        "reconstruction_report": asdict(report),
        "source_domain_id": domain.domain_id,
        "embedding_status": embedding.status,
        "coverage_status": coverage.status,
        "fidelity_status": "exact-domain-coverage",
        "coverage_certificate_id": coverage.certificate_id,
        "additional_fidelity_certificate_id": None
        if fidelity is None
        else fidelity.certificate_id,
        "source_boundary_volume": surface_volume,
        "mesh_volume": volume,
        "volume_defect": volume - surface_volume,
        "minimum_cell_volume": float(np.min(volumes)),
        "sample_to_surface_max": float(np.max(sample_distance)),
        "surface_vertex_to_sample_max": float(np.max(vertex_distance)),
        "reconstruction_fidelity_semantics": "sampled-two-sided-deviation-not-continuous-Hausdorff",
        "volume_fidelity_semantics": "exact-represented-extracted-PLC-domain",
    }


def _reconstruction_qualification_lifecycle(
    args: QualificationControls,
) -> dict[str, object]:
    """Run sampled reconstruction, native volume meshing, and transferred solver lifecycle."""

    if (
        args.resolution < 4
        or args.capacity < 1
        or args.repeats < 1
        or args.adaptation_rounds < 1
        or not np.isfinite(args.timeout)
        or args.timeout <= 0.0
        or not np.isfinite(args.target_error)
        or args.target_error <= 0.0
    ):
        raise ValueError(
            "qualification.reconstruction.controls must have resolution >= 4, capacity >= 1, "
            "repeats >= 1, adaptation_rounds >= 1, and finite positive timeout and target_error; "
            f"got resolution={args.resolution!r}, capacity={args.capacity!r}, "
            f"repeats={args.repeats!r}, adaptation_rounds={args.adaptation_rounds!r}, "
            f"timeout={args.timeout!r}, target_error={args.target_error!r}"
        )
    started = perf_counter()
    deadline = Deadline(started + args.timeout)
    phases: list[NativeMeshingPhaseMeasurement] = []
    record: dict[str, Any] = {
        "scenario": "native-reconstruction-lifecycle",
        "engine": "native",
        "status": "failed",
        "runtime": _runtime_identity(),
        "expected_admission": "positive",
        "targets": {"l2_error": args.target_error, "wall_seconds": args.timeout},
        "control_capacities": {
            "resolution": args.resolution,
            "entity_capacity": args.capacity,
            "adaptation_rounds": args.adaptation_rounds,
            "warm_repeats": args.repeats,
        },
        "epochs": [],
        "stages_seconds": {},
        "time_to_target_error_seconds": None,
        "mandatory_workflow_completion": "incomplete",
        "active_phase": "native_reconstruction",
        "nonclaims": [
            "Sampled reconstruction bounds are not continuous Hausdorff certificates.",
            "No analytic surface replaces the actual reconstructed triangles.",
            "This routine case does not close underspecified reconstruction or the mandatory corpus.",
        ],
    }
    _ledger = StageLedger(record, deadline=deadline)
    stages = record["stages_seconds"]
    try:
        fixture, measured = _ledger.run(
            record.get("active_phase", "_reconstruction_qualification_source"),
            lambda: _measured(
                lambda: _reconstruction_qualification_source(
                    args.resolution, args.capacity
                )
            ),
        )
        reconstructed, source, points = fixture
        stages[record["active_phase"]] = measured["wall_seconds"]
        record["reconstruction"] = asdict(reconstructed.report)
        record["source_binding_id"] = source.binding_id

        specification, options = _ledger.run(
            "reconstructed_request_preparation",
            lambda: _reconstruction_qualification_request(
                source, args.capacity, args.timeout
            ),
        )
        record["request_id"] = specification.specification_id
        record["native_options_id"] = options.options_id

        plan, measured = _ledger.run(
            "geometry_constraint_preparation",
            lambda: _measured(
                lambda: phx.meshing.NativeMeshingProvider(options).plan(
                    source,
                    specification,
                    coordinate_contract=phx.SpatialCoordinateContract.si(),
                    record_phase=phases.append,
                )
            ),
        )
        stages[record["active_phase"]] = measured["wall_seconds"]

        result, measured = _ledger.run(
            "generation_publication_certification",
            lambda: _measured(lambda: plan.execute(record_phase=phases.append)),
        )
        stages[record["active_phase"]] = measured["wall_seconds"]
        record["generation"] = _result_identity(result)

        def accept_result(candidate: phx.meshing.CellMeshingResult) -> dict[str, object]:
            return _reconstruction_qualification_acceptance(
                reconstructed, source, points, candidate
            )

        record["independent_acceptance"] = _ledger.run(
            "independent_acceptance", lambda: accept_result(result)
        )
        _envelope_qualification_epochs(
            source,
            result,
            specification,
            args,
            record,
            started,
            deadline=deadline,
            accept_result=accept_result,
        )
        final_error = record["epochs"][-1]["l2_error"]
        record.update(
            {
                "status": "passed" if final_error <= args.target_error else "failed",
                "physical_error": final_error,
                "target_error_reached": final_error <= args.target_error,
                "implemented_admitted_workflow": "native-reconstruction-volume-CompositionRebind-scalar-solve",
            }
        )
    except ReconstructionFailure as failure:
        record_failure(record, failure, record["active_phase"])
        record.update({"reconstruction": asdict(failure.report)})
    except phx.meshing.MeshingFailure as failure:
        record_failure(record, failure, record["active_phase"])
    except (RuntimeError, ValueError, TypeError, TimeoutError) as failure:
        record_failure(record, failure, record["active_phase"])
    finally:
        record["native_phases"] = _phase_recordings(phases)
        record["end_to_end_seconds"] = perf_counter() - started
        if record["end_to_end_seconds"] > args.timeout:
            record["status"] = "timeout"
        record["peak_process_resident_bytes"] = peak_resident_bytes(resource.RUSAGE_SELF)
        record["process_peak_scope"] = "process-lifetime"
        record.pop("active_phase")
    return record


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return _reconstruction_qualification_lifecycle(args)
