# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import json
import os
import resource
import sys
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, assert_never, final

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax import (
    array_tree_fingerprint,
    ArrayArchiveLimits,
    canonical_fingerprint,
    model_array_bytes,
)
from phydrax.lifecycle import read_meshing_source_closure, write_meshing_source_closure
from phydrax.meshing import NativeMeshingPhaseMeasurement, PlcAssociationTransfer
from phydrax.meshing.qualification import (
    declared_geometry_edge_lengths,
    execute_native_consumer,
    prepare_plc_associations,
    qualify_simplex_resources,
    transition_envelope_state,
)
from tools._meshing_cases import (
    _box_union_payload,
    analytic_region_indices,
    corpus_identity,
    frozen_request_record,
    meshing_case,
    MeshingCase,
    prepare_case,
    source_payload,
)
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._descent import certify_descent
from tools.meshing_qualification._physics import (
    _compiled_solver_stages,
    _physical_error,
    _prepare_corpus_solver,
    CorpusCompilerEvidence,
)
from tools.meshing_qualification._records import (
    _phase_recordings,
    _quality_summary,
    _result_identity,
    record_failure,
)
from tools.meshing_qualification._runtime import (
    _measured,
    _runtime_identity,
    ColdWorkerTimeout,
    Deadline,
    peak_resident_bytes,
    qualification_failure,
    QualificationFailure,
    run_cold_worker,
    StageLedger,
)
from tools.meshing_qualification.scenarios.bisection import (
    _bisection_p1_integral,
    _bisection_transfer,
)
from tools.meshing_qualification.scenarios.triangulation import _simplex_measures


def _corpus_domain(case: MeshingCase) -> Any:
    """Independently describe the analytic domain, not the generated boundary."""
    payload = source_payload(case)
    if case.name in ("plc-nonconvex", "plc-thin-channel"):
        # Independent cuboid decomposition, not a fan of the concave source cap.

        payload = _box_union_payload(case)
    if case.dimension == 2:
        facets = [
            (loop[i], loop[(i + 1) % len(loop)])
            for loop in payload["loops"]
            for i in range(len(loop))
        ]
        pairs = [(0, -1)] * len(facets)
    else:
        facets = []
        pairs = []
        for polygon, facet in zip(payload["polygons"], payload["facets"], strict=True):
            for i in range(1, len(polygon) - 1):
                facets.append((polygon[0], polygon[i], polygon[i + 1]))
                # PLC incidence is (positive, negative); the coverage owner uses
                # normal direction from its first region into its second.
                pairs.append(payload["facet_regions"][facet][::-1])
    return phx.geometry.PiecewiseLinearDomain(
        np.asarray(payload["vertices"], dtype=np.float64),
        np.asarray(facets, dtype=np.int64),
        np.asarray(pairs, dtype=np.int64),
        tuple(name for name, _ in case.expected_regions),
        source_id=str(corpus_identity(case)["source_revision"]),
    )


def _embedded_feature_coverage(
    case: MeshingCase,
    mesh: phx.discretization.CellMesh,
) -> dict[str, object]:
    payload = source_payload(case)
    if case.dimension == 2:
        features = np.asarray(payload["embedded"], dtype=np.float64)
        segments = (
            features[None, :, :]
            if features.size
            else np.empty((0, 2, 2), dtype=np.float64)
        )
    else:
        segments = np.asarray(payload["vertices"], dtype=np.float64)[payload["segments"]]
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    connectivity = mesh.connectivity
    if not isinstance(
        connectivity,
        (
            phx.discretization.PolygonalConnectivity,
            phx.discretization.TetrahedralConnectivity,
        ),
    ):
        raise TypeError(
            f"corpus.connectivity must be polygonal or tetrahedral; got {type(connectivity).__name__}"
        )
    edges = points[np.asarray(connectivity.edges)]
    failures = []
    for index, segment in enumerate(segments):
        direction = segment[1] - segment[0]
        length_squared = np.sum(direction * direction)
        parameters = ((edges - segment[0]) @ direction) / length_squared
        projected = segment[0] + parameters[..., None] * direction
        on_source = np.all(np.linalg.norm(edges - projected, axis=-1) <= 1.0e-12, axis=1)
        contained = np.all(
            (parameters >= -1.0e-12) & (parameters <= 1.0 + 1.0e-12), axis=1
        )
        intervals = np.sort(parameters[on_source & contained], axis=1)
        intervals = intervals[np.argsort(intervals[:, 0], kind="stable")]
        endpoint = 0.0
        for lower, upper in intervals:
            if abs(lower - endpoint) > 1.0e-12 or upper <= lower:
                failures.append(index)
                break
            endpoint = float(upper)
        if abs(endpoint - 1.0) > 1.0e-12:
            failures.append(index)
    return {
        "passed": not failures,
        "source_segments": segments.shape[0],
        "uncovered_or_multiply_covered_segments": sorted(set(failures)),
    }


def _material_semantics(
    case: MeshingCase,
    result: phx.meshing.CellMeshingResult,
    regions: np.ndarray,
) -> dict[str, object]:
    """Recount material assignments and the polygon-derived interface scope."""
    mesh = result.mesh
    zones = {
        zone.name: zone
        for zone in result.zones
        if zone.role is phx.meshing.MeshZoneRole.REGION
    }
    expected = {name for name, _ in case.expected_regions}
    passed = set(zones) == expected
    identifiers = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    for index, (name, _) in enumerate(case.expected_regions):
        if name not in zones:
            passed = False
            continue
        zone = zones[name]
        passed = passed and (
            zone.material_id == f"{name}-material"
            and zone.region_role is phx.meshing.RegionRole.SOLID
            and np.array_equal(
                np.sort(zone.scope.entity_ids), np.sort(identifiers[regions == index])
            )
        )
    connectivity = mesh.connectivity
    if not isinstance(connectivity, phx.discretization.TetrahedralConnectivity):
        raise TypeError(
            f"corpus.connectivity must be TetrahedralConnectivity; got {type(connectivity).__name__}"
        )
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    corners = points[np.asarray(connectivity.faces)]
    payload = source_payload(case)
    source_vertices = np.asarray(payload["vertices"], dtype=np.float64)
    patches = {patch.name: patch for patch in result.patches}
    face_ids = np.asarray(mesh.entity_set(2).entity_ids)
    patch_passed = True
    area = 0.0
    for facet, pair in enumerate(payload["facet_regions"]):
        if min(pair) < 0:
            continue
        selected_polygons = [
            polygon
            for polygon, identity in zip(
                payload["polygons"], payload["facets"], strict=True
            )
            if identity == facet
        ]
        selected = np.zeros(corners.shape[0], dtype=np.bool_)
        expected_area = 0.0
        for polygon in selected_polygons:
            # A polygon tuple selects vertex rows, not coordinate axes.
            reference = source_vertices[np.asarray(polygon, dtype=np.int64)]
            normal = np.cross(reference[1] - reference[0], reference[2] - reference[0])
            norm = np.linalg.norm(normal)
            normal /= norm
            coplanar = np.all(
                np.abs((corners - reference[0]) @ normal) <= 1.0e-12, axis=1
            )
            bounded = np.all(
                (corners >= reference.min(axis=0) - 1.0e-12)
                & (corners <= reference.max(axis=0) + 1.0e-12),
                axis=(1, 2),
            )
            selected |= coplanar & bounded
            expected_area += sum(
                0.5
                * np.linalg.norm(
                    np.cross(reference[i] - reference[0], reference[i + 1] - reference[0])
                )
                for i in range(1, len(reference) - 1)
            )
        measured_area = float(
            0.5
            * np.linalg.norm(
                np.cross(
                    corners[selected, 1] - corners[selected, 0],
                    corners[selected, 2] - corners[selected, 0],
                ),
                axis=1,
            ).sum()
        )
        area += measured_area
        name = (
            "left-right-interface"
            if case.name == "plc-material-cavity"
            else f"interface-{facet}"
        )
        patch = patches.get(name)
        adjacent = {
            zones[case.expected_regions[index][0]].zone_id
            for index in pair
            if case.expected_regions[index][0] in zones
        }
        patch_passed = patch_passed and (
            patch is not None
            and np.array_equal(
                np.sort(patch.scope.entity_ids), np.sort(face_ids[selected])
            )
            and set(patch.adjacent_zone_ids) == adjacent
            and abs(measured_area - expected_area) <= 1.0e-11
        )
    return {
        "passed": bool(passed and patch_passed),
        "interface_area": float(area),
        "interface_scope_passed": bool(patch_passed),
        "missing_regions": sorted(expected - set(zones)),
        "unexpected_regions": sorted(set(zones) - expected),
        "materials": {name: zone.material_id for name, zone in sorted(zones.items())},
    }


def _corpus_acceptance(
    case: MeshingCase,
    result: phx.meshing.CellMeshingResult,
    specification: Any,
    *,
    epoch_zero: phx.discretization.CellMesh | None = None,
    hierarchy: phx.meshing.BisectionHierarchy | None = None,
) -> dict[str, object]:
    """Check frozen source controls, using certified descent for adapted epochs."""

    mesh = result.mesh
    simplices = qualify_simplex_resources(result, specification.limits)
    points, cells = simplices.points, simplices.cells
    corners = points[cells]
    descent = None
    if epoch_zero is not None:
        if hierarchy is None:
            raise qualification_failure(
                "native_lifecycle.descent",
                "native_lifecycle.descent.hierarchy must declare refinement labels; got None",
                {},
            )
        descent = certify_descent(epoch_zero, mesh, hierarchy)
    if case.dimension == 2:
        measures = _simplex_measures(points, cells)
        regions = np.zeros(cells.shape[0], dtype=np.int64)
    else:
        measures = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
        regions = analytic_region_indices(case, corners.mean(axis=1))
    embedding = phx.geometry.certify_global_embedding(
        mesh, result.geometry, result.audit.validity
    )
    coverage = phx.geometry.certify_domain_coverage(
        mesh, result.geometry, _corpus_domain(case), regions, embedding=embedding
    )
    features = _embedded_feature_coverage(case, mesh)
    measured = {
        name: float(np.sum(measures[regions == index]))
        for index, (name, _) in enumerate(case.expected_regions)
    }
    residual = max(
        abs(measured[name] - expected) for name, expected in case.expected_regions
    )
    semantics = (
        _material_semantics(case, result, regions)
        if case.dimension == 3
        else {"passed": True, "status": "not-requested"}
    )
    connectivity = mesh.connectivity
    if not isinstance(
        connectivity,
        (
            phx.discretization.PolygonalConnectivity,
            phx.discretization.TetrahedralConnectivity,
        ),
    ):
        raise TypeError(
            f"corpus.connectivity must be polygonal or tetrahedral; got {type(connectivity).__name__}"
        )
    lengths = declared_geometry_edge_lengths(
        mesh,
        result.geometry,
        np.asarray(connectivity.edges, dtype=np.int64),
        failure_factory=qualification_failure,
    )
    control = specification.size_controls[0]
    tolerance = specification.size_compliance.relative_tolerance
    p50 = float(np.quantile(lengths, 0.5))
    max_edge = float(np.max(lengths))
    size_passed = ("hard-size" not in case.controls and descent is None) or (
        (
            descent is not None
            or abs(p50 - control.target_size) <= tolerance * control.target_size
        )
        and max_edge <= control.maximum_size * (1.0 + tolerance)
    )
    quality_passed = (
        (
            descent is not None
            or result.quality.minimum_angle >= specification.quality_target.minimum_angle
        )
        if case.dimension == 2
        else "minimum-quality" not in case.controls
        or result.quality.minimum_mean_ratio >= 0.05
    )
    fixed_passed = True
    if "fixed-boundary" in case.controls:
        faces = np.concatenate(
            [
                cells[:, list(face)]
                for face in ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
            ]
        )
        keys, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
        published = {tuple(sorted(map(tuple, points[row]))) for row in keys[counts == 1]}
        payload = source_payload(case)
        source_points = np.asarray(payload["vertices"], dtype=np.float64)
        declared = {
            tuple(sorted(map(tuple, source_points[list(face)])))
            for face in payload["polygons"]
        }
        fixed_passed = published == declared
    minimum_dihedral = None
    if "dihedral-25" in case.controls:
        normals = []
        for face in ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)):
            triangle = corners[:, face, :]
            normal = np.cross(
                triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
            )
            normals.append(normal / np.linalg.norm(normal, axis=1)[:, None])
        angles = [
            np.arccos(np.clip(-np.sum(normals[i] * normals[j], axis=1), -1.0, 1.0))
            for i in range(4)
            for j in range(i + 1, 4)
        ]
        minimum_dihedral = float(np.min(angles))
        quality_passed = quality_passed and (
            descent is not None or minimum_dihedral >= np.deg2rad(25.0)
        )
    required_limits = {
        "vertices",
        "edges",
        "faces",
        "cells",
        "connectivity_entries",
        "data_bytes",
        "scratch_bytes",
        "work_units",
        "wall_seconds",
        "cavity_cells",
        "geometry_queries",
    }
    resource_passed = case.dimension != 3 or (
        not result.runtime.unenforced_limits
        and required_limits <= set(result.runtime.enforced_limits)
    )
    passed = bool(
        result.audit.passed
        and embedding.status == "certified"
        and coverage.status == "certified"
        and np.all(measures > 0.0)
        and residual <= 1.0e-11
        and size_passed
        and quality_passed
        and fixed_passed
        and features["passed"]
        and semantics["passed"]
        and resource_passed
    )
    return {
        "passed": passed,
        "descent": descent,
        "embedding_certificate": embedding.certificate_id,
        "coverage_certificate": coverage.certificate_id,
        "embedding_status": embedding.status,
        "coverage_status": coverage.status,
        "region_measures": measured,
        "maximum_measure_residual": residual,
        "minimum_signed_cell_measure": float(np.min(measures)),
        "size_passed": size_passed,
        "quality_passed": bool(quality_passed),
        "fixed_boundary_passed": fixed_passed,
        "independent_minimum_dihedral": minimum_dihedral,
        "median_edge": p50,
        "maximum_edge": max_edge,
        "embedded_features": features,
        "material_semantics": semantics,
        "resource_guarantees": {
            "passed": resource_passed,
            "enforced_limits": list(result.runtime.enforced_limits),
            "unenforced_limits": list(result.runtime.unenforced_limits),
            "missing_limits": sorted(
                required_limits - set(result.runtime.enforced_limits)
            )
            if case.dimension == 3
            else [],
            "scope": "Declared owning execution guarantees; no missing guarantee inferred from success.",
        },
        "quality": _quality_summary(result.quality),
    }


def _triangle_comparison(
    case: MeshingCase,
    specification: Any,
) -> phx.meshing.CellMeshingResult:
    """Explicit optional Triangle algorithm; never called by a native route."""
    # Triangle is an optional comparison provider: import it at its only use.
    import triangle

    payload = source_payload(case)
    points = np.asarray(payload["vertices"], dtype=np.float64)
    segments = [
        (loop[i], loop[(i + 1) % len(loop)])
        for loop in payload["loops"]
        for i in range(len(loop))
    ]
    embedded = np.asarray(payload["embedded"], dtype=np.float64)
    if embedded.size:
        segments.append((points.shape[0], points.shape[0] + 1))
        points = np.concatenate((points, embedded))
    source = {"vertices": points, "segments": np.asarray(segments, dtype=np.int32)}
    if len(payload["loops"]) > 1:
        source["holes"] = np.asarray(((1.0, 1.0),), dtype=np.float64)
    size = specification.size_controls[0].target_size
    angle = np.degrees(specification.quality_target.minimum_angle)
    output = triangle.triangulate(source, f"pq{angle}a{0.25 * size * size}")
    return phx.meshing.certify_cell_mesh(
        phx.discretization.CellMesh.from_triangles(
            output["vertices"], output["triangles"]
        ),
        phx.SpatialCoordinateContract.si(),
    )


def _corpus_native_consumer[T](
    result: phx.meshing.CellMeshingResult,
    specification: Any,
    call: Callable[[], T],
    *,
    native: bool,
    target: Callable[[T], phx.meshing.CellMeshingResult] | None = None,
) -> tuple[phx.meshing.CellMeshingResult, T]:
    """Continue the latest ended root without resetting its limits."""
    if not native:
        value = call()
        return result if target is None else target(value), value
    consumed = execute_native_consumer(
        result,
        specification.limits,
        call,
        target=target,
        failure_factory=qualification_failure,
    )
    return consumed.result, consumed.value


def _corpus_cold_worker(
    path: str,
    content_id: str,
    expected: tuple[str, ...],
    repeats: int,
    target_error: float,
) -> dict[str, object]:
    """Restore every accepted field epoch and continue its PDE."""

    restored = read_meshing_source_closure(
        Path(path), expected_content_id=content_id, limits=ArrayArchiveLimits()
    )
    if not isinstance(restored, tuple) or len(restored) != 7:
        raise qualification_failure(
            "_corpus_cold_worker",
            "corpus.cold_packet must retain seven ordered source, control and history roles",
            {"packet_type": type(restored).__name__},
        )
    source, result, specification, options, name, history, engine = restored
    if (
        not isinstance(result, phx.meshing.CellMeshingResult)
        or not isinstance(name, str)
        or not isinstance(history, tuple)
        or not history
        or (
            source.binding_id,
            result.result_id,
            specification.specification_id,
            options.options_id,
            canonical_fingerprint(
                array_tree_fingerprint((source, specification, options, name, engine))
            ),
            canonical_fingerprint(array_tree_fingerprint(history)),
        )
        != expected
    ):
        raise qualification_failure(
            "_corpus_cold_worker",
            ("Cold corpus closure changed source, controls or accepted field histories."),
            {
                "array_tree_fingerprint": array_tree_fingerprint,
                "expected": expected,
                "history": history,
                "name": name,
                "options": options,
                "restored": restored,
                "result": result,
                "source": source,
                "specification": specification,
            },
        )
    case = meshing_case(name)
    bound, cold = _corpus_native_consumer(
        result,
        specification,
        lambda: _corpus_cold_fields(
            case, result, specification, history, repeats, target_error
        ),
        native=engine == "native",
    )
    return {**cold, "continued_result": _result_identity(bound)}


def _corpus_cold_fields(
    case: MeshingCase,
    result: phx.meshing.CellMeshingResult,
    specification: Any,
    history: tuple[Any, ...],
    repeats: int,
    target_error: float,
) -> dict[str, object]:

    name = case.name

    accepted = []
    epoch_zero = history[0][0].mesh
    for publication, fields, original_error, hierarchy in history:
        acceptance = _corpus_acceptance(
            case,
            publication,
            specification,
            epoch_zero=epoch_zero if hierarchy is not None else None,
            hierarchy=hierarchy,
        )
        if not acceptance["passed"]:
            raise qualification_failure(
                "_corpus_cold_fields",
                ("Cold accepted corpus epoch failed independent source acceptance."),
                {"acceptance": acceptance},
            )
        if not isinstance(fields, tuple) or len(fields) != 3:
            raise qualification_failure(
                "_corpus_cold_fields",
                ("Cold corpus epoch lost a declared solved, density or history field."),
                {"fields": fields},
            )
        space, _, _ = _prepare_corpus_solver(
            publication, material_x=name == "plc-material-cavity"
        )
        error = _physical_error(
            publication.mesh, fields[0], material_x=name == "plc-material-cavity"
        )
        if error != original_error:
            raise qualification_failure(
                "_corpus_cold_fields",
                ("Cold accepted field changed its initial independent physical error."),
                {"error": error, "original_error": original_error},
            )
        density_error = float(jnp.max(jnp.abs(fields[1] - 2.0)))
        history_error = float(
            jnp.max(jnp.abs(fields[2] - (space.dof_maps[0].dof_coordinates[:, 0] + 0.5)))
        )
        if density_error > 1.0e-12 or history_error > 1.0e-12:
            raise qualification_failure(
                "_corpus_cold_fields",
                (
                    "Cold corpus density/history failed independent authored field oracles."
                ),
                {"density_error": density_error, "history_error": history_error},
            )
        accepted.append(
            {
                "result": _result_identity(publication),
                "acceptance": acceptance,
                "physical_error": error,
                "density_max_defect": density_error,
                "history_max_defect": history_error,
            }
        )
    if history[-1][0].result_id != result.result_id:
        raise qualification_failure(
            "_corpus_cold_fields",
            ("Cold final corpus fields belong to a different accepted publication."),
            {"history": history, "result": result},
        )
    _, problem, free = _prepare_corpus_solver(
        result, material_x=name == "plc-material-cavity"
    )
    values, compiler, timings = _compiled_solver_stages(
        problem, jnp.asarray(history[-1][1][0])[free], repeats
    )
    error = _physical_error(result.mesh, values, material_x=name == "plc-material-cavity")
    if not np.isfinite(error) or error > target_error:
        raise qualification_failure(
            "_corpus_cold_fields",
            ("Cold continued corpus PDE missed the final physical error target."),
            {"error": error, "target_error": target_error},
        )
    return {
        "passed": True,
        "process_id": os.getpid(),
        "runtime": _runtime_identity(),
        "accepted_epochs": accepted,
        "field_names": ["u", "density", "history"],
        "accepted_fields_bitwise_preserved": True,
        "prepared_fe_caches_serialized": False,
        "continued_physical_error": error,
        "compiler": compiler,
        "solver_seconds": timings,
        "seed_origin": "actual-restored-final-accepted-u",
    }


def _corpus_cold_reanalysis(
    source: Any,
    result: phx.meshing.CellMeshingResult,
    specification: Any,
    options: Any,
    case: MeshingCase,
    history: tuple[Any, ...],
    repeats: int,
    target_error: float,
    deadline: Deadline,
    record: dict[str, Any],
) -> None:

    _ledger = StageLedger(record, deadline=deadline)

    limits = ArrayArchiveLimits()
    records = (
        source,
        result,
        specification,
        options,
        case.name,
        history,
        record["engine"],
    )
    with TemporaryDirectory(prefix="phydrax-native-corpus-source-") as directory:
        receipt, measured = _ledger.run(
            record.get("active_phase", "write_meshing_source_closure"),
            lambda: _measured(
                lambda: write_meshing_source_closure(
                    Path(directory) / "corpus.checkpoint.zip", records, limits=limits
                )
            ),
        )
        record["stages_seconds"]["allfield_history_checkpoint_publication"] = measured[
            "wall_seconds"
        ]
        expected = (
            source.binding_id,
            result.result_id,
            specification.specification_id,
            options.options_id,
            canonical_fingerprint(
                array_tree_fingerprint(
                    (source, specification, options, case.name, record["engine"])
                )
            ),
            canonical_fingerprint(array_tree_fingerprint(history)),
        )
        remaining = deadline.remaining()
        # The scenario module owns its worker command.
        cold_started = perf_counter()
        try:
            cold = run_cold_worker(
                "scenarios.native_lifecycle",
                (
                    str(receipt.path),
                    receipt.content_id,
                    json.dumps(expected),
                    str(repeats),
                    str(target_error),
                ),
                deadline=deadline,
                ledger=_ledger,
            )
            measured = {"wall_seconds": perf_counter() - cold_started}
        except ColdWorkerTimeout as failure:
            raise TimeoutError(
                f"corpus.cold_restart must finish within {remaining!r} seconds; got {type(failure).__name__}"
            ) from failure
        record["stages_seconds"]["fresh_process_allfield_history_continued_PDE"] = (
            measured["wall_seconds"]
        )
        # Worker return-code evidence is retained by StageLedger.
        # Parsed and process-validated by run_cold_worker.
        if (
            not isinstance(cold, dict)
            or cold.get("passed") is not True
            or cold.get("process_id") == os.getpid()
        ):
            raise qualification_failure(
                "_corpus_cold_reanalysis",
                ("Corpus cold evidence did not originate in an independent process."),
                {"cold": cold},
            )
        record["checkpoint"] = {
            **cold,
            "content_id": receipt.content_id,
            "container_bytes": receipt.path.stat().st_size,
            "archive_limits": {
                "members": limits.max_members,
                "nesting": limits.max_manifest_nesting,
                "rank": limits.max_array_rank,
            },
            "source_authority_stripped": False,
        }


type _CorpusSource = phx.meshing.NativePlanarSource | phx.meshing.NativePlcSource
type _CorpusRequest = phx.meshing.SurfaceMeshingSpec | phx.meshing.VolumeMeshingSpec


def _corpus_bisection_compatibility(
    source: _CorpusSource,
) -> phx.meshing.BisectionCompatibility:
    """Select label construction for the declared source family."""
    # Triangle closure repairs neighboring labels locally; recovered PLC
    # tetrahedra need uniform initialization of matching bisection labels.
    match source:
        case phx.meshing.NativePlanarSource():
            return phx.meshing.BisectionCompatibility.CONFORMING_CLOSURE
        case phx.meshing.NativePlcSource():
            return phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT
        case invalid:
            assert_never(invalid)


@final
class _NativeLifecycleContext(phx.StrictModule):
    case: MeshingCase
    engine: str
    repeats: int
    target_error: float
    adaptation_rounds: int
    started: float
    deadline: Deadline
    record: dict[str, Any]
    stages: dict[str, float]
    native_phases: list[NativeMeshingPhaseMeasurement]
    ledger: StageLedger
    set_phase: Callable[[str], None]


def _publish_native_lifecycle(
    context: _NativeLifecycleContext,
    source: _CorpusSource,
    specification: _CorpusRequest,
    options: phx.meshing.NativeMeshingOptions,
) -> phx.meshing.CellMeshingResult | None:
    case = context.case
    engine = context.engine
    record = context.record
    stages = context.stages
    native_phases = context.native_phases
    _ledger = context.ledger
    phase = "geometry_constraint_preparation"
    context.set_phase(phase)
    if engine == "native":
        provider = phx.meshing.NativeMeshingProvider(options)
        plan, measured = _ledger.run(
            phase,
            lambda: _measured(
                lambda: provider.plan(
                    source,
                    specification,
                    coordinate_contract=phx.SpatialCoordinateContract.si(),
                    record_phase=native_phases.append,
                )
            ),
        )
        stages[phase] = measured["wall_seconds"]
        record["prepared_id"] = plan.plan_id
        phase = "generation_publication_certification"
        context.set_phase(phase)
        result, measured = _ledger.run(
            phase,
            lambda: _measured(lambda: plan.execute(record_phase=native_phases.append)),
        )
    elif engine == "triangle":
        record["phase_availability"] = {
            "geometry_constraint_preparation": "included-in-comparison-generation-no-reusable-plan"
        }
        phase = "comparison_generation_publication"
        context.set_phase(phase)
        result, measured = _ledger.run(
            phase,
            lambda: _measured(lambda: _triangle_comparison(case, specification)),
        )
    else:
        raise ValueError(f"corpus.engine must be native or triangle; got {engine!r}")
    stages[phase] = measured["wall_seconds"]
    if case.expected_admission == "refusal":
        record_failure(
            record,
            RuntimeError(
                f"corpus.admission must refuse the declared refusal case; got result_id={result.result_id!r}"
            ),
            phase,
        )
        return None
    phase = "independent_acceptance"
    context.set_phase(phase)
    accepted, measured = _ledger.run(
        phase,
        lambda: _measured(
            lambda: _corpus_native_consumer(
                result,
                specification,
                lambda: _corpus_acceptance(case, result, specification),
                native=engine == "native",
            )
        ),
    )
    result, acceptance = accepted
    stages[phase] = measured["wall_seconds"]
    record["independent_acceptance"] = acceptance
    record["generation"] = _result_identity(result)
    if not acceptance["passed"]:
        record_failure(
            record,
            RuntimeError(
                f"corpus.acceptance must pass independent source/control checks; got {acceptance!r}"
            ),
            phase,
        )
        return None
    record["native_certification_id"] = (
        None if result.certification is None else result.certification.report_id
    )
    return result


@final
class _NativeLifecycleAssociation(phx.StrictModule):
    result: phx.meshing.CellMeshingResult
    transfer: PlcAssociationTransfer | None
    prepared: phx.meshing.qualification.PreparedPlcAssociations | None


def _prepare_native_lifecycle_association(
    context: _NativeLifecycleContext,
    source: _CorpusSource,
    specification: _CorpusRequest,
    result: phx.meshing.CellMeshingResult,
) -> _NativeLifecycleAssociation:
    engine = context.engine
    record = context.record
    stages = context.stages
    native_phases = context.native_phases
    _ledger = context.ledger
    association_transfer = None
    prepared_associations = None
    if isinstance(source, phx.meshing.NativePlcSource):
        phase = "authoritative_source_association_preparation"
        context.set_phase(phase)

        def inspect_source() -> phx.meshing.qualification.PreparedPlcAssociations:
            return prepare_plc_associations(
                source, result, specification.limits, record_phase=native_phases.append
            )

        inspected, measured = _ledger.run(
            phase,
            lambda: _measured(
                lambda: _corpus_native_consumer(
                    result, specification, inspect_source, native=engine == "native"
                )
            ),
        )
        result, bound_source = inspected
        prepared_associations = bound_source
        stages[phase] = measured["wall_seconds"]
        association_transfer = bound_source.association_transfer
        record["source_association_prepared_id"] = bound_source.prepared_id
        record["source_association_transfer_id"] = association_transfer.transfer_id
        record["source_association_counters"] = dict(bound_source.preparation_counters)
        record["source_association_memory_counters"] = (
            None
            if bound_source.memory_counters is None
            else dict(bound_source.memory_counters.items())
        )
    elif isinstance(source, phx.meshing.NativePlanarSource):
        if result.certification is None:
            raise ValueError(
                "native_lifecycle.certification must retain the planar source "
                "domain; got None"
            )
        source_vertices = np.asarray(source.region.vertices, dtype=np.float64)
        source_edges = np.asarray(source.region.edges, dtype=np.int64)
        if source.embedded is not None:
            source_edges = np.concatenate(
                (
                    source_edges,
                    np.asarray(source.embedded.edges, dtype=np.int64)
                    + source_vertices.shape[0],
                )
            )
            source_vertices = np.concatenate(
                (
                    source_vertices,
                    np.asarray(source.embedded.vertices, dtype=np.float64),
                )
            )
        domain = result.certification.request.domain
        if not isinstance(domain, phx.geometry.PiecewiseLinearDomain):
            raise TypeError(
                "native_lifecycle.domain must be PiecewiseLinearDomain; "
                f"got {type(domain).__name__}"
            )
        association_transfer = PlcAssociationTransfer(
            domain,
            result.coordinate_contract,
            source.source_revision,
            source_vertices=source_vertices,
            edge_vertices=source_edges,
            triangle_vertices=np.empty((0, 3), dtype=np.int64),
            triangle_facets=np.empty((0,), dtype=np.int64),
            facet_regions=domain.facet_regions,
            maximum_support_queries=specification.limits.maximum_geometry_queries,
        )
        association_transfer.source_associations(result)
        record["source_association_transfer_id"] = association_transfer.transfer_id
    return _NativeLifecycleAssociation(
        result=result, transfer=association_transfer, prepared=prepared_associations
    )


def _solve_native_lifecycle_epoch(
    context: _NativeLifecycleContext,
    result: phx.meshing.CellMeshingResult,
    previous_values: jax.Array | None,
    epoch: int,
) -> tuple[
    phx.discretization.FiniteElementDiscretization,
    phx.equations.CompiledFiniteElementProblem,
    jax.Array,
    CorpusCompilerEvidence,
    float,
]:
    case = context.case
    repeats = context.repeats
    stages = context.stages
    _ledger = context.ledger
    phase = f"solver_prepare_epoch_{epoch}"
    context.set_phase(phase)
    solver, measured = _ledger.run(
        phase,
        lambda: _measured(
            lambda: _prepare_corpus_solver(
                result, material_x=case.name == "plc-material-cavity"
            )
        ),
    )
    space, problem, free_dofs = solver
    stages[phase] = measured["wall_seconds"]
    phase = f"solver_execution_epoch_{epoch}"
    context.set_phase(phase)
    seed = (
        jnp.zeros((free_dofs.size,), dtype=result.mesh.coordinates.dtype)
        if previous_values is None
        else jnp.asarray(previous_values)[free_dofs]
    )
    values, compiler, timings = _compiled_solver_stages(problem, seed, repeats)
    stages.update({f"{key}_epoch_{epoch}": value for key, value in timings.items()})
    error = _physical_error(
        result.mesh, values, material_x=case.name == "plc-material-cavity"
    )
    return space, problem, values, compiler, error


def _adapt_native_lifecycle_fields(
    context: _NativeLifecycleContext,
    source: _CorpusSource,
    specification: _CorpusRequest,
    result: phx.meshing.CellMeshingResult,
    association_transfer: PlcAssociationTransfer | None,
    problem: phx.equations.CompiledFiniteElementProblem,
    values: jax.Array,
    previous_auxiliary: tuple[jax.Array, jax.Array],
    epoch: int,
) -> tuple[
    phx.meshing.MeshAdaptationResult,
    jax.Array,
    tuple[jax.Array, jax.Array],
    dict[str, object],
]:
    stages = context.stages
    _ledger = context.ledger
    phase = f"adaptation_prepare_epoch_{epoch}"
    context.set_phase(phase)
    identifiers = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in result.mesh.blocks]
    )
    request = phx.meshing.MarkedMeshAdaptation(
        identifiers, np.empty((0,), dtype=np.int64)
    )
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
        compatibility=_corpus_bisection_compatibility(source),
        limits=specification.limits,
        association_transfer=association_transfer,
        audit_policy=phx.meshing.CellMeshAuditPolicy(
            require_complete_association=True,
            watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
        ),
    )
    prepared, measured = _ledger.run(
        phase,
        lambda: _measured(
            lambda: phx.meshing.prepare_mesh_adaptation(result, request, policy=policy)
        ),
    )
    stages[phase] = measured["wall_seconds"]
    phase = f"adaptation_execute_epoch_{epoch}"
    context.set_phase(phase)
    candidate, measured = _ledger.run(
        phase,
        lambda: _measured(lambda: phx.meshing.execute_mesh_adaptation(prepared)),
    )
    stages[phase] = measured["wall_seconds"]
    if candidate.transfer is None:
        raise qualification_failure(
            "adapt_fields",
            ("Native lifecycle adaptation lacks its required FE state transfer."),
            {"candidate": candidate},
        )
    phase = f"field_transfer_epoch_{epoch}"
    context.set_phase(phase)
    transitioned, measured = _ledger.run(
        phase,
        lambda: _measured(
            lambda: transition_envelope_state(
                result,
                candidate,
                problem,
                values,
                epoch,
                previous_auxiliary,
                failure_factory=qualification_failure,
                physical_error=_physical_error,
            )
        ),
    )
    stages[phase] = measured["wall_seconds"]
    return candidate, *transitioned


def _accept_native_lifecycle_transfer(
    context: _NativeLifecycleContext,
    specification: _CorpusRequest,
    bound_target: phx.meshing.CellMeshingResult,
    adaptation: phx.meshing.MeshAdaptationResult,
    values: jax.Array,
    previous_values: jax.Array,
    state_transfer: dict[str, object],
    epoch: int,
    epoch_zero: phx.discretization.CellMesh,
) -> phx.meshing.CellMeshingResult:
    case = context.case
    engine = context.engine
    record = context.record
    stages = context.stages
    _ledger = context.ledger
    record["epochs"][-1]["transfer"] = _bisection_transfer(adaptation)
    record["epochs"][-1]["solver_state_transfer"] = state_transfer
    before = _bisection_p1_integral(adaptation.source.mesh, values)
    after = _bisection_p1_integral(adaptation.target.mesh, previous_values)
    record["epochs"][-1]["state_inventory"] = {
        "source": before,
        "target": after,
        "defect": after - before,
    }
    if abs(after - before) > 1.0e-10 * max(1.0, abs(before)):
        raise qualification_failure(
            "qualify_native_lifecycle",
            ("Transferred solver-state inventory was not conserved."),
            {"after": after, "before": before},
        )
    if abs(record["epochs"][-1]["transfer"]["p1_integral_residual"]) > 1.0e-10:
        raise qualification_failure(
            "qualify_native_lifecycle",
            ("Transferred P1 inventory was not conserved."),
            {"record": record},
        )
    result = bound_target
    phase = f"independent_acceptance_epoch_{epoch + 1}"
    context.set_phase(phase)
    try:
        accepted, measured = _ledger.run(
            phase,
            lambda: _measured(
                lambda: _corpus_native_consumer(
                    result,
                    specification,
                    lambda: _corpus_acceptance(
                        case,
                        result,
                        specification,
                        epoch_zero=epoch_zero,
                        hierarchy=_lifecycle_descent_hierarchy(adaptation),
                    ),
                    native=engine == "native",
                )
            ),
        )
    except QualificationFailure as failure:
        record["epochs"][-1]["target_acceptance"] = {
            "passed": False,
            "descent": {"status": "refused", "failure": str(failure)},
        }
        raise
    result, acceptance = accepted
    stages[phase] = measured["wall_seconds"]
    record["epochs"][-1]["target_acceptance"] = acceptance
    if not acceptance["passed"]:
        raise qualification_failure(
            "qualify_native_lifecycle",
            ("Adapted target violated the frozen source/control contract."),
            {"acceptance": acceptance},
        )
    return result


def _lifecycle_descent_hierarchy(
    adaptation: phx.meshing.MeshAdaptationResult,
) -> phx.meshing.BisectionHierarchy:
    match adaptation.hierarchy:
        case phx.meshing.BisectionHierarchy() as hierarchy:
            return hierarchy
        case _:
            raise qualification_failure(
                "native_lifecycle.descent",
                "native_lifecycle.descent.hierarchy must declare tagged-simplex labels; "
                f"got {type(adaptation.hierarchy).__name__}",
                {},
            )


@final
class _NativeLifecycleEpochs(phx.StrictModule):
    result: phx.meshing.CellMeshingResult
    history: tuple[
        tuple[
            phx.meshing.CellMeshingResult,
            tuple[jax.Array, ...],
            float,
            phx.meshing.BisectionHierarchy | None,
        ],
        ...,
    ]
    final_error: float


def _execute_native_lifecycle_epochs(
    context: _NativeLifecycleContext,
    source: _CorpusSource,
    specification: _CorpusRequest,
    options: phx.meshing.NativeMeshingOptions,
    result: phx.meshing.CellMeshingResult,
    association_transfer: PlcAssociationTransfer | None,
) -> _NativeLifecycleEpochs:
    case = context.case
    engine = context.engine
    target_error = context.target_error
    adaptation_rounds = context.adaptation_rounds
    started = context.started
    record = context.record
    previous_values = None
    previous_auxiliary = None
    accepted_history = []
    epoch_zero = result.mesh
    accepted_hierarchy = None
    for epoch in range(adaptation_rounds + 1):
        result, solved = _corpus_native_consumer(
            result,
            specification,
            lambda: _solve_native_lifecycle_epoch(
                context, result, previous_values, epoch
            ),
            native=engine == "native",
        )
        space, problem, values, compiler, error = solved
        if previous_auxiliary is None:
            previous_auxiliary = (
                jnp.full(values.shape, 2.0, dtype=values.dtype),
                space.dof_maps[0].dof_coordinates[:, 0] + 0.5,
            )
        accepted_history.append(
            (result, (values, *previous_auxiliary), error, accepted_hierarchy)
        )
        if error <= target_error and record["time_to_target_error_seconds"] is None:
            record["time_to_target_error_seconds"] = perf_counter() - started
        record["epochs"].append(
            {
                "epoch": epoch,
                "result": _result_identity(result),
                "l2_error": error,
                "dofs": space.dof_maps[0].global_dof_count,
                "compiler": compiler,
                "retained_bytes": model_array_bytes(
                    (
                        source,
                        specification,
                        options,
                        tuple(accepted_history),
                        space,
                        problem,
                    )
                ),
                "transferred_state_l2_error": (
                    None
                    if previous_values is None
                    else _physical_error(
                        result.mesh,
                        previous_values,
                        material_x=case.name == "plc-material-cavity",
                    )
                ),
            }
        )
        if epoch == adaptation_rounds:
            break
        auxiliary = previous_auxiliary
        bound_target, transitioned = _corpus_native_consumer(
            result,
            specification,
            lambda: _adapt_native_lifecycle_fields(
                context,
                source,
                specification,
                result,
                association_transfer,
                problem,
                values,
                auxiliary,
                epoch,
            ),
            native=engine == "native",
            target=lambda value: value[0].target,
        )
        adaptation, previous_values, previous_auxiliary, state_transfer = transitioned
        result = _accept_native_lifecycle_transfer(
            context,
            specification,
            bound_target,
            adaptation,
            values,
            previous_values,
            state_transfer,
            epoch,
            epoch_zero,
        )
        accepted_hierarchy = _lifecycle_descent_hierarchy(adaptation)
    return _NativeLifecycleEpochs(
        result=result, history=tuple(accepted_history), final_error=error
    )


def _complete_native_lifecycle(
    context: _NativeLifecycleContext,
    source: _CorpusSource,
    specification: _CorpusRequest,
    options: phx.meshing.NativeMeshingOptions,
    epochs: _NativeLifecycleEpochs,
    timeout: float,
) -> None:
    case = context.case
    repeats = context.repeats
    target_error = context.target_error
    started = context.started
    record = context.record
    result, accepted_history = epochs.result, epochs.history
    phase = "allfield_history_source_checkpoint_and_cold_continued_PDE"
    context.set_phase(phase)
    _corpus_cold_reanalysis(
        source,
        result,
        specification,
        options,
        case,
        tuple(accepted_history),
        repeats,
        target_error,
        context.deadline,
        record,
    )
    elapsed = perf_counter() - started
    final_error = epochs.final_error
    record["physical_target_first_reached_seconds"] = record[
        "time_to_target_error_seconds"
    ]
    if final_error <= target_error:
        record["time_to_target_error_seconds"] = elapsed
    record.update(
        {
            "status": "passed"
            if final_error <= target_error and elapsed <= timeout
            else "failed",
            "target_error_reached": final_error <= target_error,
            "physical_error": final_error,
            "end_to_end_seconds": elapsed,
            "peak_process_resident_bytes": peak_resident_bytes(resource.RUSAGE_SELF),
            "independent_scientific_certification": "passed",
            "mandatory_workflow_completion": "incomplete",
            "case_contract_completion": "complete"
            if final_error <= target_error and elapsed <= timeout
            else "incomplete",
            "nonclaims": [
                "This routine corpus case is not the entire W15 mandatory workflow.",
                "No release or performance leadership is inferred from this run.",
            ],
        }
    )
    if elapsed > timeout:
        record["status"] = "timeout"


def _record_native_lifecycle_refusal(
    context: _NativeLifecycleContext,
    failure: phx.meshing.MeshingFailure,
    phase: str,
    source: _CorpusSource | None,
    source_fingerprint: dict[str, Any] | None,
) -> None:
    case = context.case
    started = context.started
    record = context.record
    expected = {
        "planar-capacity-refusal": phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED,
        "planar-stale-revision": phx.meshing.MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
        "plc-incompatible-constraints": phx.meshing.MeshingFailureCategory.INVALID_SOURCE,
        "plc-stale-revision": phx.meshing.MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
    }.get(case.name)
    if case.expected_admission == "refusal" and case.workflow == "resource-boundary":
        expected = phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    source_unchanged = (
        source is not None
        and source_fingerprint is not None
        and array_tree_fingerprint(source) == source_fingerprint
    )
    record_failure(record, failure, phase)
    record.update(
        {
            "status": "expected-refusal"
            if failure.category is expected and source_unchanged
            else "timeout"
            if failure.category is phx.meshing.MeshingFailureCategory.TIMED_OUT
            else "failed",
            "source_bits_preserved": source_unchanged,
            "mandatory_workflow_completion": "incomplete",
            "end_to_end_seconds": perf_counter() - started,
        }
    )


def _finish_native_lifecycle_measurements(context: _NativeLifecycleContext) -> None:
    started = context.started
    record = context.record
    native_phases = context.native_phases
    record["native_phases"] = _phase_recordings(native_phases)
    record["end_to_end_seconds"] = perf_counter() - started
    record["peak_process_resident_bytes"] = peak_resident_bytes(resource.RUSAGE_SELF)
    record["process_peak_scope"] = "process-lifetime"
    devices = []
    for device in jax.local_devices():
        statistics = device.memory_stats()
        devices.append(
            {
                "device_id": device.id,
                "platform": device.platform,
                "device_kind": device.device_kind,
                "peak_bytes": None
                if statistics is None
                else statistics.get("peak_bytes_in_use"),
                "memory_statistics": statistics,
                "status": "unavailable" if statistics is None else "backend-reported",
            }
        )
    record["device_memory"] = devices


def qualify_native_lifecycle(
    case_name: str,
    *,
    resolution: int = 4,
    capacity: int = 20000,
    timeout: float = 120.0,
    repeats: int = 3,
    target_error: float = 0.05,
    adaptation_rounds: int = 1,
    engine: str = "native",
) -> dict[str, object]:
    """Native geometry → certification → transfer → solve with fixed targets."""
    case = meshing_case(case_name)
    started = perf_counter()
    deadline = Deadline(started + timeout)
    record: dict[str, Any] = {
        "scenario": "native-lifecycle",
        "engine": engine,
        "case": corpus_identity(case),
        "runtime": _runtime_identity(),
        "status": "failed",
        "expected_admission": case.expected_admission,
        "control_capacities": {
            "resolution": resolution,
            "entity_capacity": capacity,
            "adaptation_rounds": adaptation_rounds,
            "solver_warm_repeats": repeats,
        },
        "targets": {"l2_error": target_error, "wall_seconds": timeout},
        "adaptation_engine": "native-bisection",
        "time_to_target_error_seconds": None,
        "stages_seconds": {},
        "epochs": [],
        "release_evidence": "unassessed",
        "leadership_evidence": "unassessed",
    }
    _ledger = StageLedger(record, deadline=deadline)
    if (
        repeats < 1
        or adaptation_rounds < 0
        or not np.isfinite(target_error)
        or target_error <= 0.0
    ):
        raise ValueError(
            f"corpus.controls must have positive repeats and finite positive target error with nonnegative rounds; got repeats={repeats!r}, target_error={target_error!r}, rounds={adaptation_rounds!r}"
        )
    if engine == "triangle" and find_spec("triangle") is None:
        return {**record, "status": "missing-dependency", "dependency": "triangle"}
    if engine == "triangle" and (
        case.dimension != 2 or case.expected_admission != "positive"
    ):
        return {
            **record,
            "status": "ineligible-comparison",
            "detail": "Triangle cannot match this source/admission contract.",
        }
    stages = record["stages_seconds"]
    native_phases: list[NativeMeshingPhaseMeasurement] = []
    phase = "source_preparation"
    source = None
    source_fingerprint = None

    def set_phase(name: str) -> None:
        nonlocal phase
        phase = name

    context = _NativeLifecycleContext(
        case=case,
        engine=engine,
        repeats=repeats,
        target_error=target_error,
        adaptation_rounds=adaptation_rounds,
        started=started,
        deadline=deadline,
        record=record,
        stages=stages,
        native_phases=native_phases,
        ledger=_ledger,
        set_phase=set_phase,
    )
    try:
        prepared_case, measured = _ledger.run(
            phase,
            lambda: _measured(lambda: prepare_case(case, resolution, capacity, timeout)),
        )
        source, specification, options = prepared_case

        source_fingerprint = array_tree_fingerprint(source)
        stages[phase] = measured["wall_seconds"]
        record["request_id"] = specification.specification_id
        record["source_binding_id"] = source.binding_id
        record["limits_id"] = specification.limits.limits_id
        compatibility = _corpus_bisection_compatibility(source)
        record["bisection_compatibility"] = compatibility.value
        frozen_controls = {
            key: value
            for key, value in frozen_request_record(
                source, specification, options
            ).items()
            if key != "frozen_request_id"
        }
        frozen_controls["bisection_compatibility"] = compatibility.value
        record["frozen_request"] = {
            **frozen_controls,
            "frozen_request_id": canonical_fingerprint(frozen_controls),
        }
        result = _publish_native_lifecycle(context, source, specification, options)
        if result is None:
            return record
        association = _prepare_native_lifecycle_association(
            context, source, specification, result
        )
        epochs = _execute_native_lifecycle_epochs(
            context,
            source,
            specification,
            options,
            association.result,
            association.transfer,
        )
        _complete_native_lifecycle(
            context, source, specification, options, epochs, timeout
        )
    except phx.meshing.MeshingFailure as failure:
        _record_native_lifecycle_refusal(
            context, failure, phase, source, source_fingerprint
        )
    except TimeoutError as failure:
        record_failure(record, failure, phase)
        record.update(
            {
                "status": "timeout",
                "mandatory_workflow_completion": "incomplete",
                "end_to_end_seconds": perf_counter() - started,
            }
        )
    except (RuntimeError, ValueError, TypeError) as failure:
        # This is an evidence boundary: failures are preserved, never replaced
        # by another algorithm or silently removed from campaign statistics.
        record_failure(record, failure, phase)
        record.update({"end_to_end_seconds": perf_counter() - started})
    finally:
        _finish_native_lifecycle_measurements(context)
    return record


def worker(argv: tuple[str, ...]) -> dict[str, Any]:
    """Decode process arguments and run the scenario-owned cold consumer."""
    return _corpus_cold_worker(
        argv[0], argv[1], tuple(json.loads(argv[2])), int(argv[3]), float(argv[4])
    )


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "worker":
        raise ValueError(
            "qualification.subcommand must equal worker; got " + repr(sys.argv[1:])
        )
    print(json.dumps(worker(tuple(sys.argv[2:]))))


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    match args.scenario:
        case "native-lifecycle":
            return qualify_native_lifecycle(
                args.corpus_case,
                resolution=args.resolution,
                capacity=args.capacity,
                timeout=args.timeout,
                repeats=args.repeats,
                target_error=args.target_error,
                adaptation_rounds=args.adaptation_rounds,
            )
        case "comparison-triangle-lifecycle":
            return qualify_native_lifecycle(
                args.corpus_case,
                resolution=args.resolution,
                capacity=args.capacity,
                timeout=args.timeout,
                repeats=args.repeats,
                target_error=args.target_error,
                adaptation_rounds=args.adaptation_rounds,
                engine="triangle",
            )
        case _:
            raise ValueError(
                f"qualification.scenario must match this module; got {args.scenario!r}"
            )
