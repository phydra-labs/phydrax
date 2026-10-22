# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

import phydrax as phx
from phydrax import model_array_bytes
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._fixtures import _kuhn_tetrahedra, _planar_triangles
from tools.meshing_qualification._records import _quality_summary, _result_identity
from tools.meshing_qualification._runtime import (
    _contract,
    _measured,
    _runtime_identity,
    qualification_failure,
    StageMeasurement,
)
from tools.meshing_qualification.scenarios.triangulation import _simplex_measures


class _BisectionTransferRecord(TypedDict):
    transfer_id: str
    preserves_constants: bool
    preserves_linear: bool
    conservative: bool
    positivity_preserving: bool
    linear_reproduction_error: float
    constant_reproduction_error: float
    p1_integral_residual: float
    retained_bytes: int


class _BisectionRoundTripRecord(TypedDict):
    source: dict[str, object]
    refined: dict[str, object]
    target: dict[str, object]
    statuses: list[str]
    restores_source_topology: bool
    quality: dict[str, dict[str, object]]
    transfer: dict[str, _BisectionTransferRecord]
    lineage: dict[str, dict[str, object]]
    evidence: dict[str, dict[str, object]]
    hierarchy_id: str
    shape_classes: dict[str, object]
    resources: dict[str, int | StageMeasurement | dict[str, StageMeasurement]]


def _bisection_meshes() -> dict[str, phx.discretization.CellMesh]:
    """A 4 x 4 right-triangle lattice of the unit square and 2^3 Kuhn cubes."""

    axis = np.linspace(0.0, 1.0, 5)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    spatial, tetrahedra = _kuhn_tetrahedra(2)
    return {
        "triangles": phx.discretization.CellMesh.from_triangles(
            points, _planar_triangles(4)
        ),
        "tetrahedra": phx.discretization.CellMesh.from_tetrahedra(spatial, tetrahedra),
    }


def _bisection_cell_ids(mesh: phx.discretization.CellMesh, /) -> np.ndarray:
    return np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)


def _bisection_corner_cells(mesh: phx.discretization.CellMesh, /) -> np.ndarray:
    """Cells touching the vertex nearest the origin (repeated local refinement)."""
    corner = np.argmin(np.linalg.norm(np.asarray(mesh.coordinates), axis=1))
    touching = np.any(np.asarray(mesh.blocks[0].vertices) == corner, axis=1)
    return np.sort(_bisection_cell_ids(mesh)[touching])


def _bisection_adapt(
    route: phx.meshing.MeshAdaptationRoute,
    source: phx.meshing.CellMeshingResult,
    refine: np.ndarray,
    coarsen: np.ndarray,
    hierarchy: Any,
    /,
    *,
    device_policy: phx.discretization.AdaptiveSimplexPolicy | None = None,
) -> tuple[phx.meshing.MeshAdaptationResult, dict[str, StageMeasurement]]:

    request = phx.meshing.MarkedMeshAdaptation(
        np.asarray(refine, dtype=np.int64),
        np.asarray(coarsen, dtype=np.int64),
        hierarchy=hierarchy,
    )
    policy = phx.meshing.MeshAdaptationPolicy(route, device_policy=device_policy)
    prepared, preparation = _measured(
        lambda: phx.meshing.prepare_mesh_adaptation(source, request, policy=policy)
    )
    result, execution = _measured(lambda: phx.meshing.execute_mesh_adaptation(prepared))

    if not result.target.audit.passed:
        raise qualification_failure(
            "_bisection_adapt",
            (f"{route.name} produced a target that failed its audit."),
            {"result": result},
        )
    return result, {"prepare": preparation, "execute": execution}


def _bisection_lineage(result: phx.meshing.MeshAdaptationResult, /) -> dict[str, object]:
    """Every entity is an identity, a lineage relation, or created/deleted."""

    source, target = result.source.mesh, result.target.mesh
    lineage = result.lineage
    if lineage is None:
        raise qualification_failure(
            "_bisection_lineage",
            ("Bisection qualification requires lineage for every entity."),
            {"lineage": lineage},
        )
    accounting = lineage.validate_complete(source, target)
    return {
        "lineage_id": accounting.lineage_id,
        "complete": True,
        **{
            f"dimension_{item.dimension}": item.to_record()
            for item in accounting.dimensions
        },
    }


def _bisection_evidence(evidence: Any, /) -> dict[str, object]:
    return {
        "evidence_id": evidence.evidence_id,
        "requested_refinements": evidence.requested_refinements,
        "accepted_refinements": evidence.accepted_refinements,
        "rejected_refinements": evidence.rejected_refinement_ids.shape[0],
        "admissibility_tests": evidence.admissibility_tests,
        "bisections": evidence.bisections,
        "closure_iterations": evidence.closure_iterations,
        "created_vertices": evidence.created_vertices,
        "maximum_generation": evidence.maximum_generation,
        "initially_compatible": evidence.initially_compatible,
        "incompatible_facets": evidence.incompatible_facets,
        "uniform_refinement_applied": evidence.uniform_refinement_applied,
        "requested_coarsenings": evidence.requested_coarsenings,
        "coarsened_vertices": evidence.coarsened_vertices,
        "coarsening_passes": evidence.coarsening_passes,
        "restored_cells": evidence.restored_cells,
        "rejected_coarsenings": evidence.rejected_coarsening_ids.shape[0],
    }


def _bisection_p1_integral(mesh: phx.discretization.CellMesh, values: Any, /) -> float:
    """Integrate the P1 interpolant of vertex values in closed form."""

    match mesh.topological_dimension:
        case 2:
            kind = "triangle"
        case 3:
            kind = "tetrahedron"
        case _:
            raise ValueError(
                f"_bisection_p1_integral.topological_dimension must be two or three; got {mesh.topological_dimension!r}."
            )
    if any(block.cell_kind != kind for block in mesh.blocks):
        raise ValueError(
            f"_bisection_p1_integral.blocks must have the same simplex cell kind; got {tuple(block.cell_kind for block in mesh.blocks)!r}."
        )
    cells = (
        np.asarray(mesh.blocks[0].vertices)
        if len(mesh.blocks) == 1
        else np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    )
    measures = np.abs(_simplex_measures(np.asarray(mesh.coordinates), cells))
    return float(np.sum(measures * np.mean(np.asarray(values)[cells], axis=1)))


def _bisection_transfer(
    result: phx.meshing.MeshAdaptationResult, /
) -> _BisectionTransferRecord:
    """Claims, P1 linear reproduction, and P1 integral change of one transfer."""

    transfer = result.transfer
    if transfer is None:
        raise qualification_failure(
            "_bisection_transfer",
            ("Bisection qualification requires a finite-element topology transfer."),
            {"transfer": transfer},
        )
    source = np.asarray(result.source.mesh.coordinates)
    target = np.asarray(result.target.mesh.coordinates)
    slope = np.linspace(0.5, 1.5, source.shape[1])

    linear = np.asarray(transfer.apply(source @ slope + 0.25))
    reproduction = float(np.max(np.abs(linear - (target @ slope + 0.25))))

    constant = np.asarray(transfer.apply(np.ones((source.shape[0],))))
    smooth = np.sin(3.0 * source[:, 0]) * np.cos(2.0 * source[:, -1])

    moved = np.asarray(transfer.apply(smooth))
    residual = _bisection_p1_integral(result.target.mesh, moved) - _bisection_p1_integral(
        result.source.mesh, smooth
    )
    if (
        not transfer.preserves_constants
        or not transfer.preserves_linear
        or reproduction > 1.0e-12
        or float(np.max(np.abs(constant - 1.0))) > 1.0e-14
    ):
        raise qualification_failure(
            "_bisection_transfer",
            ("The bisection transfer does not reproduce P1 fields."),
            {"constant": constant, "reproduction": reproduction, "transfer": transfer},
        )
    return {
        "transfer_id": transfer.transfer_id,
        "preserves_constants": transfer.preserves_constants,
        "preserves_linear": transfer.preserves_linear,
        "conservative": transfer.conservative,
        "positivity_preserving": transfer.positivity_preserving,
        "linear_reproduction_error": reproduction,
        "constant_reproduction_error": float(np.max(np.abs(constant - 1.0))),
        "p1_integral_residual": residual,
        "retained_bytes": model_array_bytes(transfer),
    }


def _bisection_shape_classes(
    source: phx.meshing.CellMeshingResult, rounds: int, cycles: int, /
) -> dict[str, object]:
    """Uniform class bound, then repeated corner refinement against it.

    Newest-vertex (Maubach) bisection generates finitely many similarity
    classes; the first uniform generations visit them all, so locally refined
    meshes never fall below the uniform bound.
    """

    route = phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
    angle = source.quality.minimum_angle
    mean_ratio = source.quality.minimum_mean_ratio
    current, hierarchy = source, None
    for _ in range(rounds):
        cells = np.sort(_bisection_cell_ids(current.mesh))
        result, _ = _bisection_adapt(
            route, current, cells, np.empty((0,), dtype=np.int64), hierarchy
        )
        current, hierarchy = result.target, result.hierarchy
        angle = min(angle, current.quality.minimum_angle)
        mean_ratio = min(mean_ratio, current.quality.minimum_mean_ratio)
    current, hierarchy = source, None
    angles, ratios = [], []
    for _ in range(cycles):
        marks = _bisection_corner_cells(current.mesh)
        result, _ = _bisection_adapt(
            route, current, marks, np.empty((0,), dtype=np.int64), hierarchy
        )
        current, hierarchy = result.target, result.hierarchy
        angles.append(current.quality.minimum_angle)
        ratios.append(current.quality.minimum_mean_ratio)
    bounded = min(angles) >= angle * (1.0 - 1.0e-9) and min(ratios) >= 0.999 * mean_ratio

    evidence = result.evidence
    if not isinstance(evidence, phx.meshing.BisectionEvidence):
        raise qualification_failure(
            "_bisection_shape_classes",
            ("Bisection shape-class qualification lacks its native evidence."),
            {"evidence": evidence},
        )
    if not bounded or evidence.maximum_generation < cycles:
        raise qualification_failure(
            "_bisection_shape_classes",
            ("Repeated bisection left the uniform shape-class bound."),
            {"bounded": bounded, "cycles": cycles, "evidence": evidence},
        )
    return {
        "uniform_rounds": rounds,
        "uniform_minimum_angle_bound": angle,
        "uniform_minimum_mean_ratio_bound": mean_ratio,
        "corner_cycles": cycles,
        "corner_minimum_angles": angles,
        "corner_minimum_mean_ratios": ratios,
        "minimum_angle_ratio_to_source": min(angles) / source.quality.minimum_angle,
        "minimum_mean_ratio_ratio_to_source": min(ratios)
        / source.quality.minimum_mean_ratio,
        "maximum_generation": evidence.maximum_generation,
        "final_cells": current.mesh.blocks[0].cell_count,
        "bounded": bounded,
    }


def _bisection_round_trip(
    mesh: phx.discretization.CellMesh, rounds: int, cycles: int, /
) -> _BisectionRoundTripRecord:

    route = phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
    source, certification = _measured(
        lambda: phx.meshing.certify_cell_mesh(mesh, _contract())
    )
    marks = np.sort(_bisection_cell_ids(mesh))[::3]
    first, first_stages = _bisection_adapt(
        route, source, marks, np.empty((0,), dtype=np.int64), None
    )
    corner = _bisection_corner_cells(first.target.mesh)
    second, second_stages = _bisection_adapt(
        route,
        first.target,
        corner,
        np.empty((0,), dtype=np.int64),
        first.hierarchy,
    )
    everything = np.sort(_bisection_cell_ids(second.target.mesh))
    restored, restore_stages = _bisection_adapt(
        route,
        second.target,
        np.empty((0,), dtype=np.int64),
        everything,
        second.hierarchy,
    )
    back = restored.target.mesh
    identical = (
        back.topology_id == mesh.topology_id
        and np.array_equal(np.asarray(back.coordinates), np.asarray(mesh.coordinates))
        and np.array_equal(
            np.asarray(back.vertex_global_ids), np.asarray(mesh.vertex_global_ids)
        )
        and np.array_equal(_bisection_cell_ids(back), _bisection_cell_ids(mesh))
    )
    statuses = (first.status, second.status, restored.status)
    complete = phx.meshing.MeshAdaptationStatus.COMPLETE
    if not identical or first.status is not complete or second.status is not complete:
        raise qualification_failure(
            "_bisection_round_trip",
            (
                "Bisection round trip must retain topology, coordinates and entity IDs with accepted refinement statuses."
            ),
            {
                "complete": complete,
                "first": first,
                "identical": identical,
                "second": second,
            },
        )
    refinement = _bisection_transfer(first)
    if not refinement["conservative"] or abs(refinement["p1_integral_residual"]) > 1e-13:
        raise qualification_failure(
            "_bisection_round_trip",
            ("Refinement transfer does not conserve P1 integrals."),
            {"refinement": refinement},
        )
    hierarchy = second.hierarchy
    if not isinstance(hierarchy, phx.meshing.BisectionHierarchy):
        raise qualification_failure(
            "_bisection_round_trip",
            ("Bisection round trip did not retain its hierarchy."),
            {"hierarchy": hierarchy},
        )
    return {
        "source": _result_identity(source),
        "refined": _result_identity(second.target),
        "target": _result_identity(restored.target),
        "statuses": [status.value for status in statuses],
        "restores_source_topology": identical,
        "quality": {
            "before": _quality_summary(source.quality),
            "refined": _quality_summary(second.target.quality),
            "after": _quality_summary(restored.target.quality),
        },
        "transfer": {
            "refinement": refinement,
            "local_refinement": _bisection_transfer(second),
            "coarsening": _bisection_transfer(restored),
        },
        "lineage": {
            "refinement": _bisection_lineage(first),
            "local_refinement": _bisection_lineage(second),
            "coarsening": _bisection_lineage(restored),
        },
        "evidence": {
            "refinement": _bisection_evidence(first.evidence),
            "local_refinement": _bisection_evidence(second.evidence),
            "coarsening": _bisection_evidence(restored.evidence),
        },
        "hierarchy_id": hierarchy.hierarchy_id,
        "shape_classes": _bisection_shape_classes(source, rounds, cycles),
        "resources": {
            "certify": certification,
            "refinement": first_stages,
            "local_refinement": second_stages,
            "coarsening": restore_stages,
            "hierarchy_retained_bytes": model_array_bytes(second.hierarchy),
            "target_mesh_retained_bytes": model_array_bytes(second.target.mesh),
        },
    }


def qualify_bisection() -> dict[str, object]:
    """NATIVE_BISECTION refine/coarsen round trips of triangles and tetrahedra."""

    meshes = _bisection_meshes()
    dimensions = {
        "triangles": _bisection_round_trip(meshes["triangles"], 4, 12),
        "tetrahedra": _bisection_round_trip(meshes["tetrahedra"], 3, 6),
    }
    return {
        "status": "passed",
        "route": phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION.value,
        "source": {name: record["source"] for name, record in dimensions.items()},
        "target": {name: record["target"] for name, record in dimensions.items()},
        "quality": {
            name: {
                "before": record["quality"]["before"],
                "after": record["quality"]["after"],
            }
            for name, record in dimensions.items()
        },
        "transfer": {name: record["transfer"] for name, record in dimensions.items()},
        "conservation": {
            name: {
                stage: values["p1_integral_residual"]
                for stage, values in record["transfer"].items()
            }
            for name, record in dimensions.items()
        },
        "resources": {name: record["resources"] for name, record in dimensions.items()},
        "dimensions": dimensions,
        "runtime": _runtime_identity(),
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_bisection()
