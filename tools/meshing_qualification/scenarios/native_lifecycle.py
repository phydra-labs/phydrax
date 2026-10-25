# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from contextlib import nullcontext
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
from phydrax._meshcore import current_native_execution_scope
from phydrax.discretization._coordinate_operands import CoordinateOperandCollector
from phydrax.lifecycle import (
    MeshingAcceptedEpoch,
    read_meshing_source_closure,
    read_meshing_source_execution_controls,
    validate_meshing_accepted_epochs,
    write_meshing_source_closure,
)
from phydrax.lifecycle._meshing_field_records import (
    rebind_meshing_adaptation_execution_endpoints,
)
from phydrax.meshing import NativeMeshingPhaseMeasurement, PlcAssociationTransfer
from phydrax.meshing.qualification import (
    declared_geometry_edge_lengths,
    execute_native_consumer,
    execute_native_request,
    prepare_envelope_accepted_epoch,
    prepare_plc_associations,
    qualify_simplex_resources,
    transition_envelope_state,
)
from tools._meshing_cases import (
    _box_union_payload,
    analytic_region_indices,
    corpus_identity,
    corpus_solution_field_degree,
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
    _physical_inventory,
    _prepare_corpus_solver,
    CorpusCompilerEvidence,
    CorpusPreconditioning,
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
    ColdControlPacket,
    ColdWorkerReport,
    ColdWorkerTimeout,
    Deadline,
    peak_resident_bytes,
    PEAK_RESIDENT_SCOPE,
    prepare_cold_worker,
    qualification_failure,
    QualificationFailure,
    StageLedger,
)
from tools.meshing_qualification.scenarios.bisection import _bisection_transfer
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


def _corpus_simplex_rows(mesh: phx.discretization.CellMesh) -> tuple[np.ndarray, ...]:
    """Ascending vertex rows of every topology degree of an all-simplex mesh.

    Rows are read from the mesh's canonical cell-complex incidences in entity
    order, whatever packing its connectivity uses (an adapted target of several
    tetrahedron blocks carries packed polyhedral incidence). Every degree-``d``
    entity must have ``d + 1`` facets whose vertex rows cover ``d + 1``
    distinct vertices exactly ``d`` times each, every block must declare the
    simplex kind, and the top rows must be the blocks' sorted cells in order;
    any other polygon or polyhedron is refused rather than relabeled.
    """
    kind = {2: "triangle", 3: "tetrahedron"}.get(mesh.topological_dimension)
    kinds = sorted({block.cell_kind for block in mesh.blocks})
    if kind is None or kinds != [kind]:
        raise TypeError(
            f"corpus.mesh.blocks must all declare one simplex kind; got {kinds} in dimension {mesh.topological_dimension}"
        )
    topology = mesh.topology
    rows = [np.arange(topology.entity_sets[0].count, dtype=np.int64)[:, None]]
    for degree, incidence in enumerate(topology.incidences, start=1):
        count = topology.entity_sets[degree].count
        valid = np.asarray(incidence.relation.valid, dtype=np.bool_)
        sources = np.asarray(incidence.relation.source_indices, dtype=np.int64)[valid]
        targets = np.asarray(incidence.relation.target_indices, dtype=np.int64)[valid]
        if np.any(np.bincount(targets, minlength=count) != degree + 1):
            raise TypeError(
                f"corpus.topology must give every degree-{degree} entity {degree + 1} facets; got a non-simplicial entity"
            )
        facets = sources[np.argsort(targets, kind="stable")].reshape((count, degree + 1))
        vertices = np.sort(rows[-1][facets].reshape((count, -1)), axis=1).reshape(
            (count, degree + 1, degree)
        )
        if np.any(vertices != vertices[:, :, :1]) or np.any(
            vertices[:, 1:, 0] <= vertices[:, :-1, 0]
        ):
            raise TypeError(
                f"corpus.topology must span {degree + 1} distinct vertices per degree-{degree} simplex; got another cell"
            )
        rows.append(vertices[:, :, 0])
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    if len(rows) != mesh.topological_dimension + 1 or not np.array_equal(
        rows[-1], np.sort(cells, axis=1)
    ):
        raise TypeError(
            "corpus.topology cells must be the declared block simplices in block order; got different cells"
        )
    return tuple(rows)


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
    edges = points[_corpus_simplex_rows(mesh)[1]]
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
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    corners = points[_corpus_simplex_rows(mesh)[2]]
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
    budget: phx.discretization.CoordinateEnclosureBudget | None = None,
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
        mesh, result.geometry, result.audit.validity, budget=budget
    )
    coverage = phx.geometry.certify_domain_coverage(
        mesh, result.geometry, _corpus_domain(case), regions, embedding=embedding,
        budget=budget,
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
    lengths = declared_geometry_edge_lengths(
        mesh,
        result.geometry,
        _corpus_simplex_rows(mesh)[1],
        failure_factory=qualification_failure,
        budget=budget,
    )
    control = specification.size_controls[0]
    tolerance = specification.size_compliance.relative_tolerance
    p50 = float(np.quantile(lengths, 0.5))
    max_edge = float(np.max(lengths))
    target_size_satisfied = abs(p50 - control.target_size) <= tolerance * control.target_size
    maximum_size_satisfied = max_edge <= control.maximum_size * (1.0 + tolerance)
    size_passed = "hard-size" not in case.controls or (
        (descent is not None or target_size_satisfied) and maximum_size_satisfied
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
        "size_control_strength": control.strength.value,
        "target_size_satisfied": bool(target_size_satisfied),
        "maximum_size_satisfied": bool(maximum_size_satisfied),
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
    call: Callable[[phx.discretization.CoordinateEnclosureBudget | None], T],
    *,
    native: bool,
    target: Callable[[T], phx.meshing.CellMeshingResult] | None = None,
    coordinate_operands: CoordinateOperandCollector | None = None,
) -> tuple[phx.meshing.CellMeshingResult, T]:
    """Continue the request that produced ``result`` under its remaining allowance."""
    if not native:
        value = call(None)
        return result if target is None else target(value), value
    consumed = execute_native_consumer(
        result,
        specification.limits,
        call,
        target=target,
        failure_factory=qualification_failure,
        observe_managed_admission=True,
        prepared_operands=None
        if coordinate_operands is None
        else coordinate_operands.for_result(result.mesh, result.geometry),
    )
    return consumed.result, consumed.value


def _corpus_native_request[T](
    source: phx.meshing.CellMeshingResult,
    specification: Any,
    call: Callable[[phx.discretization.CoordinateEnclosureBudget | None], T],
    *,
    native: bool,
    target: Callable[[T], phx.meshing.CellMeshingResult],
    deadline: Deadline | None = None,
    coordinate_operands: CoordinateOperandCollector | None = None,
) -> tuple[phx.meshing.CellMeshingResult, T]:
    """Run one new complete adaptation request under its original limits.

    The source's completed request is history only; the workflow deadline
    intersects this request's own wall budget and is never reset.
    """
    if not native:
        value = call(None)
        return target(value), value
    try:
        requested = execute_native_request(
            source,
            specification.limits,
            call,
            target=target,
            failure_factory=qualification_failure,
            observe_managed_admission=True,
            workflow_remaining_seconds=None if deadline is None else deadline.remaining(),
        )
        if coordinate_operands is not None and coordinate_operands.pending:
            receipt = requested.result.execution_evidence
            if receipt is None:
                raise RuntimeError("Native request must bind its actual operand earning phase.")
            coordinate_operands.complete(receipt)
    except BaseException:
        if coordinate_operands is not None:
            coordinate_operands.abort()
        raise
    return requested.result, requested.value


def _corpus_cold_worker(
    path: Path,
    content_id: str,
    expected: tuple[str, ...],
    repeats: int,
    target_error: float,
    name: str,
    engine: str,
    expected_errors: tuple[float, ...],
    *,
    budget: phx.discretization.CoordinateEnclosureBudget | None,
) -> tuple[phx.meshing.CellMeshingResult, ColdWorkerReport]:
    """Restore all accepted epochs and continue their PDE in the owning request."""

    started = perf_counter()
    restored = read_meshing_source_closure(
        path, expected_content_id=content_id, limits=ArrayArchiveLimits(), budget=budget,
    )
    if not isinstance(restored, Mapping) or not {
        "generation_source", "generation_specification", "generation_options",
        "accepted_target", "accepted_data",
    } <= restored.keys():
        raise qualification_failure(
            "_corpus_cold_worker",
            "corpus.cold_packet must retain the canonical accepted-epoch source envelope",
            {"packet_type": type(restored).__name__},
        )
    source = restored["generation_source"]
    result = restored["accepted_target"]
    specification = restored["generation_specification"]
    options = restored["generation_options"]
    history = validate_meshing_accepted_epochs(restored)
    if (
        not isinstance(result, phx.meshing.CellMeshingResult)
        or not isinstance(name, str)
        or not isinstance(history, tuple)
        or not history
        or (engine == "native") != (budget is not None)
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
    read_seconds = perf_counter() - started
    case = meshing_case(name)
    cold = _corpus_cold_fields(
        case, result, specification, history, repeats, target_error,
        expected_errors=expected_errors, budget=budget,
    )
    return result, ColdWorkerReport(record=cold, read_seconds=read_seconds)


def _corpus_cold_fields(
    case: MeshingCase,
    result: phx.meshing.CellMeshingResult,
    specification: Any,
    history: tuple[MeshingAcceptedEpoch, ...],
    repeats: int,
    target_error: float,
    *,
    expected_errors: tuple[float, ...],
    budget: phx.discretization.CoordinateEnclosureBudget | None = None,
) -> dict[str, object]:

    name = case.name

    accepted = []
    epoch_zero = history[0].carrier.mesh
    for epoch, original_error in zip(history, expected_errors, strict=True):
        publication = epoch.carrier
        fields = tuple(epoch.fields[name] for name in ("u", "density", "history"))
        if epoch.transition is not None and not isinstance(
            epoch.transition, phx.meshing.MeshAdaptationResult
        ):
            raise qualification_failure(
                "_corpus_cold_fields",
                "Cold corpus history must retain its actual mesh adaptation transaction.",
                {"transition": epoch.transition},
            )
        hierarchy = (
            None if epoch.transition is None
            else _lifecycle_descent_hierarchy(epoch.transition)
        )
        acceptance = _corpus_acceptance(
            case,
            publication,
            specification,
            epoch_zero=epoch_zero if hierarchy is not None else None,
            hierarchy=hierarchy,
            budget=budget,
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
            publication,
            material_x=name == "plc-material-cavity",
            solution_degree=corpus_solution_field_degree(case),
            budget=budget,
        )
        error = _physical_error(
            publication.mesh,
            fields[0],
            material_x=name == "plc-material-cavity",
            space=space,
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
    if history[-1].carrier.result_id != result.result_id:
        raise qualification_failure(
            "_corpus_cold_fields",
            ("Cold final corpus fields belong to a different accepted publication."),
            {"history": history, "result": result},
        )
    space, problem, free = _prepare_corpus_solver(
        result,
        material_x=name == "plc-material-cavity",
        solution_degree=corpus_solution_field_degree(case),
        budget=budget,
    )
    values, compiler, timings = _compiled_solver_stages(
        problem, jnp.asarray(history[-1].fields["u"])[free], repeats
    )
    error = _physical_error(
        result.mesh, values, material_x=name == "plc-material-cavity", space=space
    )
    if not np.isfinite(error) or error > target_error:
        raise qualification_failure(
            "_corpus_cold_fields",
            ("Cold continued corpus PDE missed the final physical error target."),
            {"error": error, "target_error": target_error},
        )
    return {
        "passed": True,
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
    history: tuple[MeshingAcceptedEpoch, ...],
    repeats: int,
    target_error: float,
    deadline: Deadline,
    record: dict[str, Any],
) -> None:

    _ledger = StageLedger(
        record,
        deadline=deadline,
        cold_phase="fresh_process_allfield_history_continued_PDE",
    )

    limits = ArrayArchiveLimits()
    if result.certification is None:
        raise qualification_failure(
            "_corpus_cold_reanalysis",
            "Canonical accepted history requires its actual target certification inputs.",
            {"result": result},
        )
    records = {
        "certification_inputs": result.certification.request,
        "report": result.certification,
        "associations": result.associations,
        "generation_source": source,
        "generation_specification": specification,
        "generation_options": options,
        "accepted_target": result,
        "accepted_data": {"accepted_epochs": history},
    }
    with (
        TemporaryDirectory(prefix="phydrax-native-corpus-source-") as directory,
        prepare_cold_worker(
            "tools.meshing_qualification.scenarios.native_lifecycle",
            deadline=deadline,
            ledger=_ledger,
        ) as cold_worker,
    ):
        receipt, measured = _ledger.run(
            "allfield_history_checkpoint_publication",
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
        cold_started = perf_counter()
        try:
            cold = cold_worker.execute(
                receipt.path,
                receipt.content_id,
                {
                    "expected": list(expected),
                    "repeats": repeats,
                    "target_error": target_error,
                    "engine": record["engine"],
                    "case_name": case.name,
                    "epoch_errors": [epoch["l2_error"] for epoch in record["epochs"]],
                },
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
        # Parsed and process-bound by ColdWorker.execute.
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


def _corpus_generation_options(
    source: _CorpusSource,
    options: phx.meshing.NativeMeshingOptions,
    *,
    native: bool,
    adaptation_rounds: int,
) -> phx.meshing.NativeMeshingOptions:
    """Select explicit source preparation before freezing generation controls."""
    if (
        not native
        or adaptation_rounds == 0
        or not isinstance(source, phx.meshing.NativePlcSource)
        or source.complex.boundary != "fixed"
    ):
        return options
    schedule = options.volume_schedule
    if schedule is None:
        raise ValueError("Fixed-boundary native adaptation requires a volume schedule.")
    if schedule.prepare_fixed_boundary_adaptation:
        return options
    prepared_schedule = phx.meshing.NativeVolumeSchedule(
        radius_edge_bound=schedule.radius_edge_bound,
        minimum_dihedral_degrees=schedule.minimum_dihedral_degrees,
        refinement_rounds=schedule.refinement_rounds,
        improvement_passes=schedule.improvement_passes,
        exudation_passes=schedule.exudation_passes,
        exudation_weight_fraction=schedule.exudation_weight_fraction,
        metric_optimization_passes=schedule.metric_optimization_passes,
        prepare_fixed_boundary_adaptation=True,
    )
    return phx.meshing.NativeMeshingOptions(
        options.route,
        implicit_policy=options.implicit_policy,
        curve_schedule=options.curve_schedule,
        surface_schedule=options.surface_schedule,
        volume_schedule=prepared_schedule,
        polyhedral_schedule=options.polyhedral_schedule,
        structured_schedule=options.structured_schedule,
        hex_core_fraction=options.hex_core_fraction,
        grid_schedule=options.grid_schedule,
    )


def _corpus_bisection_compatibility(
    source: _CorpusSource,
) -> phx.meshing.BisectionCompatibility:
    """Select the declared source family's conforming subdivision law."""
    # Tetrahedral whole-edge stars need no matching-label initialization and
    # retain the exact source domain without a barycentric compatibility prefix.
    match source:
        case phx.meshing.NativePlanarSource():
            return phx.meshing.BisectionCompatibility.CONFORMING_CLOSURE
        case phx.meshing.NativePlcSource():
            return phx.meshing.BisectionCompatibility.WHOLE_EDGE_STAR
        case invalid:
            assert_never(invalid)


def _corpus_refinement_plan(
    result: phx.meshing.CellMeshingResult,
    target_error: float,
    maximum_cells: int,
    /,
    *,
    material_x: bool,
    current_error: float | None = None,
) -> tuple[int, int | None, dict[str, object]]:
    """Provision whole-star generations from a quadratic interpolation proxy.

    The proxy is the closed-form P1 interpolation L2 integral, not an FE error
    estimator or a convergence theorem. A regular-tetrahedron reference gives
    a desired cell count; binary generation counts provision that count only.
    Actual closure capacities and the final physical solve remain authoritative.
    """
    mesh = result.mesh
    if mesh.topological_dimension != 3:
        return 1, None, {"rule": "one-triangle-generation", "maximum_generations": 1}
    if current_error is not None and current_error <= target_error:
        return 1, None, {
            "rule": "physical-target-met-mandatory-whole-star",
            "current_physical_error": current_error,
            "target_error": target_error,
            "maximum_generations": 1,
            "convergence_claim": False,
            "acceptance_authority": "actual-final-physical-solve",
        }
    count = sum(block.cell_count for block in mesh.blocks)
    scope = current_native_execution_scope()
    work, scratch = 128 * count, 512 * count
    with (
        scope.batch(work, bytes_upper=scratch)
        if scope is not None else nullcontext()
    ) as ticket:
        if ticket is not None:
            ticket.consume(work, managed_bytes=scratch)
        points = np.asarray(mesh.coordinates)
        cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
        corners = points[cells]
        volumes = _simplex_measures(points, cells)
        pairs = np.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
        differences = corners[:, pairs[:, 0]] - corners[:, pairs[:, 1]]
        coefficient = np.ones(count)
        if material_x:
            coefficient = np.where(np.mean(corners[:, :, 0], axis=1) > 1.0, 0.5, 1.0)
            squared = differences[:, :, 0] ** 2 * coefficient[:, None]
        else:
            squared = np.sum(differences * differences, axis=2)
        incident = np.zeros((count, 4))
        for edge, (first, second) in enumerate(pairs):
            incident[:, first] += squared[:, edge]
            incident[:, second] += squared[:, edge]
        # E[(sum_e |e|² lambda_i lambda_j)²], Dirichlet(1,1,1,1).
        integral = volumes * (
            np.sum(squared, axis=1) ** 2 + np.sum(squared * squared, axis=1)
            + np.sum(incident * incident, axis=1)
        ) / 840.0
        proxy = float(np.sqrt(np.sum(integral)))
        volume = float(np.sum(volumes))
        regular_constant = np.sqrt(13.0 / 140.0) * (6.0 * np.sqrt(2.0)) ** (2.0 / 3.0)
        if material_x:
            regular_constant *= float(np.sum(volumes * coefficient) / volume) / 3.0
        desired = max(
            count,
            int(np.ceil((regular_constant * volume ** (7.0 / 6.0) / target_error) ** 1.5)),
        )
        desired_depth = max(1, int(np.floor(np.log2(desired / count))))
        capacity_depth = max(1, int(np.floor(np.log2(maximum_cells / count))))
        generations = 1 if proxy <= target_error else min(desired_depth, capacity_depth)
    return generations, desired, {
        "rule": "quadratic-interpolation-regular-reference-provisioning",
        "interpolation_l2_proxy": proxy,
        "target_error": target_error,
        "source_cells": count,
        "domain_measure": volume,
        "desired_cells": desired,
        "maximum_generations": generations,
        "binary_minimum_if_all_generations_execute": count * 2**generations,
        "convergence_claim": False,
        "acceptance_authority": "actual-final-physical-solve",
    }


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
    coordinate_operands: CoordinateOperandCollector | None


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
            lambda: _measured(
                lambda: plan.execute(
                    record_phase=native_phases.append,
                    coordinate_operands=context.coordinate_operands,
                )
            ),
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
                lambda budget: _corpus_acceptance(
                    case, result, specification, budget=budget
                ),
                native=engine == "native",
                coordinate_operands=context.coordinate_operands,
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

        def inspect_source(
            _budget: phx.discretization.CoordinateEnclosureBudget | None,
        ) -> phx.meshing.qualification.PreparedPlcAssociations:
            return prepare_plc_associations(
                source, result, specification.limits, record_phase=native_phases.append,
                prepared_operands=None
                if context.coordinate_operands is None
                else context.coordinate_operands.for_result(result.mesh, result.geometry),
                budget=_budget,
            )

        inspected, measured = _ledger.run(
            phase,
            lambda: _measured(
                lambda: _corpus_native_consumer(
                    result, specification, inspect_source, native=engine == "native",
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
    previous_auxiliary: tuple[jax.Array, jax.Array] | None,
    previous_carrier: phx.meshing.CellMeshingResult | None,
    transition: phx.meshing.MeshAdaptationResult | None,
    epoch: int,
    *,
    budget: phx.discretization.CoordinateEnclosureBudget | None = None,
) -> tuple[
    phx.discretization.FiniteElementDiscretization,
    phx.equations.CompiledFiniteElementProblem,
    jax.Array,
    CorpusCompilerEvidence,
    float,
    MeshingAcceptedEpoch,
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
                result,
                material_x=case.name == "plc-material-cavity",
                solution_degree=corpus_solution_field_degree(case),
                budget=budget,
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
    # Whole-domain diffusion has an admitted exact diagonal. The region-restricted
    # material form retains its authored policy rather than substituting a diagonal.
    values, compiler, timings = _compiled_solver_stages(
        problem,
        seed,
        repeats,
        preconditioning=(
            CorpusPreconditioning.NONE
            if case.name == "plc-material-cavity"
            else CorpusPreconditioning.EXACT_JACOBI
        ),
    )
    stages.update({f"{key}_epoch_{epoch}": value for key, value in timings.items()})
    error = _physical_error(
        result.mesh, values, material_x=case.name == "plc-material-cavity", space=space
    )
    if previous_auxiliary is None:
        previous_auxiliary = (
            jnp.full((result.mesh.coordinates.shape[0],), 2.0, dtype=values.dtype),
            space.dof_maps[0].dof_coordinates[:, 0] + 0.5,
        )
    if transition is not None:
        if previous_carrier is None:
            raise qualification_failure(
                "_solve_native_lifecycle_epoch",
                "An adapted accepted epoch must retain its actual preceding carrier.",
                {"transition": transition},
            )
        transition = rebind_meshing_adaptation_execution_endpoints(
            transition, previous_carrier, result,
        )
    accepted = prepare_envelope_accepted_epoch(
        result, space, values, previous_auxiliary, epoch,
        part_name=case.name, transition=transition, budget=budget,
    )
    return space, problem, values, compiler, error, accepted


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
    *,
    budget: phx.discretization.CoordinateEnclosureBudget | None = None,
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
    generations, target_cells, refinement_plan = _corpus_refinement_plan(
        result, context.target_error, specification.limits.maximum_cells,
        material_x=context.case.name == "plc-material-cavity",
        current_error=context.record["epochs"][-1]["l2_error"],
    )
    request = phx.meshing.MarkedMeshAdaptation(
        identifiers, np.empty((0,), dtype=np.int64),
        refinement_generations=generations,
        refinement_target_cells=target_cells,
    )
    context.record.setdefault("refinement_plans", []).append(
        {"epoch": epoch, "request_id": request.request_id, **refinement_plan}
    )
    protected_scopes: tuple[phx.meshing.MeshingScope, ...] = ()
    if isinstance(source, phx.meshing.NativePlcSource) and source.complex.boundary == "fixed":
        facets = result.mesh.entity_set(result.mesh.topological_dimension - 1)
        if budget is not None:
            budget.reserve(facets.count, 32 * facets.count)
        boundary = np.asarray(facets.subset("boundary").mask, dtype=np.bool_)
        fixed_boundary = phx.meshing.MeshingScope(
            result.mesh.mesh_id, result.mesh.numeric_version,
            phx.meshing.MeshingEntityKind.MESH, result.mesh.topological_dimension - 1,
            facets.entity_set_id, np.asarray(facets.entity_ids)[boundary],
        )
        protected_scopes = (
            fixed_boundary,
            *(patch.scope for patch in result.patches
              if patch.scope.entity_dimension == result.mesh.topological_dimension - 1),
        )
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
        compatibility=_corpus_bisection_compatibility(source),
        protected_scopes=protected_scopes,
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
            lambda: phx.meshing.prepare_mesh_adaptation(
                result, request, policy=policy, observe_managed_admission=True
            )
        ),
    )
    stages[phase] = measured["wall_seconds"]
    phase = f"adaptation_execute_epoch_{epoch}"
    context.set_phase(phase)
    candidate, measured = _ledger.run(
        phase,
        lambda: _measured(
            lambda: phx.meshing.execute_mesh_adaptation(
                prepared,
                coordinate_operands=context.coordinate_operands,
                observe_managed_admission=True,
            )
        ),
    )
    stages[phase] = measured["wall_seconds"]
    if budget is not None:
        identity_count = identifiers.size + sum(
            block.cell_count for block in candidate.target.mesh.blocks
        )
        budget.reserve(identity_count, 32 * identity_count)
    target_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in candidate.target.mesh.blocks]
    )
    retained_marks = np.intersect1d(identifiers, target_ids)
    if candidate.status is not phx.meshing.MeshAdaptationStatus.COMPLETE or retained_marks.size:
        raise qualification_failure(
            "adapt_fields",
            "Native lifecycle adaptation must genuinely refine every originally marked cell.",
            {"status": candidate.status.value, "retained_marked_cells": retained_marks.tolist()},
        )
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
                budget=budget,
                solution_inventory=_physical_inventory,
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
    inventory = state_transfer["solution_inventory"]
    if not isinstance(inventory, dict):
        raise TypeError("corpus.state_transfer must retain actual FE solution inventories")
    before, after = inventory["source"], inventory["target"]
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
                    lambda budget: _corpus_acceptance(
                        case,
                        result,
                        specification,
                        epoch_zero=epoch_zero,
                        hierarchy=_lifecycle_descent_hierarchy(adaptation),
                        budget=budget,
                    ),
                    native=engine == "native",
                    coordinate_operands=context.coordinate_operands,
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
    history: tuple[MeshingAcceptedEpoch, ...]
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
    accepted_history: list[MeshingAcceptedEpoch] = []
    epoch_zero = result.mesh
    accepted_transition = None
    for epoch in range(adaptation_rounds + 1):
        result, solved = _corpus_native_consumer(
            result,
            specification,
            lambda budget: _solve_native_lifecycle_epoch(
                context, result, previous_values, previous_auxiliary,
                accepted_history[-1].carrier if accepted_history else None,
                accepted_transition, epoch, budget=budget,
            ),
            native=engine == "native",
            coordinate_operands=context.coordinate_operands,
        )
        space, problem, values, compiler, error, prepared_epoch = solved
        transition = prepared_epoch.transition
        if transition is not None:
            if not isinstance(transition, phx.meshing.MeshAdaptationResult):
                raise qualification_failure(
                    "_execute_native_lifecycle_epochs",
                    "The corpus history must retain its actual mesh adaptation transaction.",
                    {"transition": transition},
                )
            transition = rebind_meshing_adaptation_execution_endpoints(
                transition, accepted_history[-1].carrier, result,
            )
        accepted = MeshingAcceptedEpoch(
            result, prepared_epoch.declarations, prepared_epoch.fields,
            transition=transition,
        )
        accepted_history.append(accepted)
        previous_auxiliary = (
            jnp.asarray(accepted.fields["density"]),
            jnp.asarray(accepted.fields["history"]),
        )
        if error <= target_error and record["time_to_target_error_seconds"] is None:
            record["time_to_target_error_seconds"] = perf_counter() - started
        record["epochs"].append(
            {
                "epoch": epoch,
                "result": _result_identity(result),
                "l2_error": error,
                "dofs": space.dof_maps[0].global_dof_count,
                "solution_field_degree": corpus_solution_field_degree(case),
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
                        space=space,
                    )
                ),
            }
        )
        if epoch == adaptation_rounds:
            break
        auxiliary = previous_auxiliary
        bound_target, transitioned = _corpus_native_request(
            result,
            specification,
            lambda budget: _adapt_native_lifecycle_fields(
                context,
                source,
                specification,
                result,
                association_transfer,
                problem,
                values,
                auxiliary,
                epoch,
                budget=budget,
            ),
            native=engine == "native",
            target=lambda value: value[0].target,
            deadline=context.deadline,
            coordinate_operands=context.coordinate_operands,
        )
        adaptation, previous_values, previous_auxiliary, state_transfer = transitioned
        result = _accept_native_lifecycle_transfer(
            context,
            specification,
            bound_target,
            adaptation,
            state_transfer,
            epoch,
            epoch_zero,
        )
        accepted_transition = adaptation
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
    lifecycle_complete = (
        final_error <= target_error
        and elapsed <= timeout
        and context.adaptation_rounds > 0
        and len(accepted_history) == context.adaptation_rounds + 1
    )
    # time_to_target_error_seconds keeps its first-reach value; end_to_end_seconds
    # separately records the whole lifecycle including cold reanalysis.
    record.update(
        {
            "status": "passed"
            if final_error <= target_error and elapsed <= timeout
            else "failed",
            "target_error_reached": final_error <= target_error,
            "physical_error": final_error,
            "end_to_end_seconds": elapsed,
            "peak_process_resident_bytes": peak_resident_bytes(),
            "independent_scientific_certification": "passed",
            "mandatory_workflow_completion": "complete"
            if lifecycle_complete
            else "incomplete",
            "case_contract_completion": "complete"
            if final_error <= target_error and elapsed <= timeout
            else "incomplete",
            "nonclaims": [
                "This certifies one lifecycle; the corpus also requires its other routine, larger, and combined profiles.",
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
    if failure.category is expected and source_unchanged:
        record["status"] = "expected-refusal"
    record.update(
        {
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
    record["peak_process_resident_bytes"] = peak_resident_bytes()
    record["process_peak_scope"] = PEAK_RESIDENT_SCOPE
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
    maximum_work_units: int | None = None,
    maximum_geometry_queries: int | None = None,
    maximum_scratch_bytes: int | None = None,
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
        coordinate_operands=CoordinateOperandCollector() if engine == "native" else None,
    )
    try:
        prepared_case, measured = _ledger.run(
            phase,
            lambda: _measured(
                lambda: prepare_case(
                    case, resolution, capacity, timeout,
                    maximum_work_units=maximum_work_units,
                    maximum_geometry_queries=maximum_geometry_queries,
                    maximum_scratch_bytes=maximum_scratch_bytes,
                )
            ),
        )
        source, specification, options = prepared_case
        options = _corpus_generation_options(
            source, options, native=engine == "native", adaptation_rounds=adaptation_rounds
        )

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


def cold_worker(packet: ColdControlPacket) -> ColdWorkerReport:
    """Run the scenario-owned cold consumer on its one control packet."""
    controls = packet.controls
    arguments = (
        packet.archive_path,
        packet.content_id,
        tuple(controls["expected"]),
        int(controls["repeats"]),
        float(controls["target_error"]),
        str(controls["case_name"]),
        str(controls["engine"]),
        tuple(float(value) for value in controls["epoch_errors"]),
    )
    if controls["engine"] != "native":
        restored, report = _corpus_cold_worker(*arguments, budget=None)
        return ColdWorkerReport(
            record={**report.record, "continued_result": _result_identity(restored)},
            read_seconds=report.read_seconds,
        )
    controls_started = perf_counter()
    raw = read_meshing_source_execution_controls(
        packet.archive_path, expected_content_id=packet.content_id,
        limits=ArrayArchiveLimits(),
    )
    controls_seconds = perf_counter() - controls_started
    request = execute_native_request(
        raw.execution_evidence,
        raw.limits,
        lambda budget: _corpus_cold_worker(*arguments, budget=budget),
        target=lambda value: value[0],
        failure_factory=qualification_failure,
        observe_managed_admission=True,
        workflow_remaining_seconds=packet.deadline.remaining(),
    )
    report = request.value[1]
    return ColdWorkerReport(
        record={**report.record, "continued_result": _result_identity(request.result)},
        read_seconds=controls_seconds + report.read_seconds,
    )


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    match args.scenario:
        case "native-lifecycle":
            return qualify_native_lifecycle(
                args.corpus_case,
                resolution=args.resolution,
                capacity=args.capacity,
                timeout=args.timeout,
                maximum_work_units=args.maximum_work_units,
                maximum_geometry_queries=args.maximum_geometry_queries,
                maximum_scratch_bytes=args.maximum_scratch_bytes,
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
                maximum_work_units=args.maximum_work_units,
                maximum_geometry_queries=args.maximum_geometry_queries,
                maximum_scratch_bytes=args.maximum_scratch_bytes,
                repeats=args.repeats,
                target_error=args.target_error,
                adaptation_rounds=args.adaptation_rounds,
                engine="triangle",
            )
        case _:
            raise ValueError(
                f"qualification.scenario must match this module; got {args.scenario!r}"
            )
