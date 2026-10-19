# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import resource
from dataclasses import (
    asdict,
)
from pathlib import (
    Path,
)
from tempfile import (
    TemporaryDirectory,
)
from time import (
    perf_counter,
)
from typing import (
    Any,
    final,
    TypedDict,
)

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax import (
    ArrayArchiveLimits,
    canonical_fingerprint,
    model_array_bytes,
)
from phydrax.discretization import (
    PolygonalConnectivity,
)
from phydrax.geometry import (
    CommonRefinementEvidence,
)
from phydrax.lifecycle import (
    prepare_meshing_field_owner,
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing import (
    NativeMeshingPhaseMeasurement,
    OversetConnectivityError,
    OversetPartSpec,
    OversetPolicy,
    OversetVertexStatus,
    prepare_overset_connectivity,
    prepare_overset_conservative_state_transport,
    prepare_overset_field_transfer,
    prepare_overset_motion_rebind,
)
from phydrax.meshing.qualification import (
    deform_planar_interior,
    field_declarations,
    field_entries,
    field_plans,
    poisson_solve_policy,
    prepare_poisson_update,
    rigid_planar_successor,
)
from tools.meshing_qualification._controls import (
    _overset_qualification_requirements,
    capacity_meshing_limits,
    QualificationControls,
)
from tools.meshing_qualification._physics import (
    _compiled_solver_stages,
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
    Deadline,
    peak_resident_bytes,
    qualification_failure,
    StageLedger,
)
from tools.meshing_qualification.scenarios.metric_adaptation import (
    _mesh_triangles,
)
from tools.meshing_qualification.scenarios.triangulation import (
    _simplex_measures,
)


def _overset_qualification_source(
    name: str,
    resolution: int,
    capacity: int,
    timeout: float,
) -> tuple[Any, Any, Any, dict[str, object]]:
    """Author both provider inputs with protected source wall/exterior obligations."""
    if name not in ("background", "moving"):
        raise ValueError(f"overset.name must be background or moving; got {name!r}")
    half = 3.0 if name == "background" else 2.25
    vertices = np.asarray(
        ((-half, -half), (half, -half), (half, half), (-half, half)), dtype=np.float64
    )
    loops = ((0, 1, 2, 3),)
    if name == "moving":
        vertices = np.concatenate(
            (
                vertices,
                np.asarray(
                    ((-0.25, -0.25), (-0.25, 0.25), (0.25, 0.25), (0.25, -0.25)),
                    dtype=np.float64,
                ),
            )
        )
        loops = (*loops, (4, 5, 6, 7))
    region = phx.geometry.PlanarMeshRegion(vertices, loops, feature_id=name)
    revision = canonical_fingerprint(
        {
            "kind": "overset-authored-planar-source",
            "name": name,
            "vertices": vertices.tolist(),
            "loops": loops,
        }
    )
    source = phx.meshing.NativePlanarSource(region, revision)
    scope = phx.meshing.MeshingScope(
        name,
        revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "overset-fluid-region",
        np.asarray((0,), dtype=np.int64),
    )
    exterior = phx.meshing.MeshingScope(
        name,
        revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        1,
        "overset-exterior",
        np.arange(4, dtype=np.int64),
    )
    protected = [
        phx.meshing.ProtectedFeature(
            exterior, phx.meshing.FeatureKind.CURVE, maximum_deviation=1.0e-10, hard=True
        )
    ]
    scopes: dict[str, object] = {"fluid": scope.scope_id, "boundary": exterior.scope_id}
    if name == "moving":
        wall = phx.meshing.MeshingScope(
            name,
            revision,
            phx.meshing.MeshingEntityKind.GEOMETRY,
            1,
            "overset-cavity-wall",
            np.arange(4, 8, dtype=np.int64),
        )
        protected.append(
            phx.meshing.ProtectedFeature(
                wall, phx.meshing.FeatureKind.CURVE, maximum_deviation=1.0e-10, hard=True
            )
        )
        scopes["cavity_wall"] = wall.scope_id
    limits = capacity_meshing_limits(
        capacity,
        timeout,
        edges=8,
        faces=8,
        cells=8,
        connectivity_entries=64,
        data_bytes=1024,
        work_units=32,
        cavity_cells=1,
        geometry_queries=64,
        scratch_bytes=4096,
    )
    size = 0.8 / resolution
    request = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 2, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        ),
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                size,
                maximum_size=1.5 * size,
                strength=phx.meshing.SizeControlStrength.HARD,
            ),
        ),
        size_compliance=phx.meshing.SizeCompliancePolicy(
            relative_tolerance=0.5, target_statistics=("p50",)
        ),
        protected_features=tuple(protected),
        quality_target=phx.meshing.MeshQualityTarget(
            minimum_angle=np.radians(25.0), hard=True
        ),
        limits=limits,
    )
    return (
        source,
        request,
        phx.meshing.NativeMeshingOptions("planar_constrained_delaunay"),
        scopes,
    )


def _overset_qualification_part(
    name: str,
    args: QualificationControls,
    record: dict[str, Any],
    phases: list[NativeMeshingPhaseMeasurement],
    deadline: Deadline,
    *,
    resolution: int | None = None,
) -> tuple[phx.meshing.MeshPart, phx.meshing.NativeMeshingPlan]:
    """Execute a real source/provider plan and independently recount its domain."""

    _ledger = StageLedger(record, deadline=deadline)

    chosen = args.resolution if resolution is None else resolution
    key = name if resolution is None else f"{name}-conservative-target"

    prepared, timing = _ledger.run(
        f"{key}:source_preparation",
        lambda: _measured(
            lambda: _overset_qualification_source(
                name, chosen, args.capacity, args.timeout
            )
        ),
    )
    source, request, options, scopes = prepared
    record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
    evidence: dict[str, Any] = {
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "source_binding_id": source.binding_id,
        "source_scopes": scopes,
        "request_id": request.specification_id,
        "limits_id": request.limits.limits_id,
        "quality_target_id": request.quality_target.target_id,
        "protected_feature_ids": [
            feature.feature_id for feature in request.protected_features
        ],
        "source_vertices": np.asarray(source.region.vertices).tolist(),
    }
    record["parts"][key] = evidence

    provider = _ledger.run(
        f"{key}:geometry_constraint_preparation",
        lambda: phx.meshing.NativeMeshingProvider(options),
    )
    plan, timing = _ledger.run(
        record.get("active_phase", "plan"),
        lambda: _measured(
            lambda: provider.plan(
                source,
                request,
                coordinate_contract=phx.SpatialCoordinateContract.si(),
                record_phase=phases.append,
            )
        ),
    )
    record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
    evidence["prepared_id"] = plan.plan_id

    result, timing = _ledger.run(
        f"{key}:generation_publication_certification",
        lambda: _measured(lambda: plan.execute(record_phase=phases.append)),
    )
    record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
    evidence["generation"] = _result_identity(result)
    if (
        result.certification is None
        or not result.audit.passed
        or not result.compliance.passed
    ):
        raise qualification_failure(
            "_overset_qualification_part",
            ("Overset provider carrier lacks passing source/control certification."),
            {"result": result},
        )
    cells = _mesh_triangles(result.mesh)
    points = np.asarray(result.mesh.coordinates)
    areas = _simplex_measures(points, cells)
    expected = 36.0 if name == "background" else 20.0
    connectivity = result.mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError(
            f"overset.connectivity must be PolygonalConnectivity; got {type(connectivity).__name__}"
        )
    boundary = np.asarray(connectivity.boundary_edges)
    edges = np.asarray(connectivity.edges)[boundary]
    corners = points[edges]
    inner = np.all(np.abs(corners) <= 0.25 + 1.0e-10, axis=(1, 2))
    lengths = np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1)
    measured_area = float(np.sum(areas))
    outer_length = float(np.sum(lengths[~inner]))
    wall_length = float(np.sum(lengths[inner]))
    if np.any(areas <= 0.0) or abs(measured_area - expected) > 1.0e-9:
        raise qualification_failure(
            "_overset_qualification_part",
            (
                "Overset source area or triangle orientation disagrees with the analytic fixture."
            ),
            {"areas": areas, "expected": expected, "measured_area": measured_area},
        )
    if abs(outer_length - (24.0 if name == "background" else 18.0)) > 1.0e-9:
        raise qualification_failure(
            "_overset_qualification_part",
            ("Overset source exterior is not covered exactly once."),
            {"name": name, "outer_length": outer_length},
        )
    if name == "moving" and (
        abs(wall_length - 2.0) > 1.0e-9
        or np.any(np.all(np.abs(points[cells].mean(axis=1)) < 0.25, axis=1))
    ):
        raise qualification_failure(
            "_overset_qualification_part",
            ("Overset protected cavity wall or excluded cavity was lost."),
            {
                "cells": cells,
                "name": name,
                "points": points,
                "wall_length": wall_length,
            },
        )
    evidence["independent_acceptance"] = {
        "expected_area": expected,
        "measured_area": measured_area,
        "outer_boundary_length": outer_length,
        "cavity_wall_length": wall_length,
        "quality": _quality_summary(
            phx.meshing.summarize_cell_quality(
                phx.meshing.evaluate_cell_quality(result.mesh)
            )
        ),
        "certification_id": result.certification.report_id,
        "requested": dict(result.compliance.requested),
        "achieved": dict(result.compliance.achieved),
    }
    record["attempts"].append(
        {
            "phase": record["active_phase"],
            "status": "passed",
            "source_binding_id": source.binding_id,
            "result_id": result.result_id,
        }
    )
    return phx.meshing.MeshPart(name, result), plan


def _overset_qualification_connectivity(
    background: Any,
    moving: Any,
    args: QualificationControls,
) -> Any:
    """Bind cavity and artificial-boundary scopes to native entity IDs."""

    mesh = moving.carrier.mesh
    boundary = np.asarray(mesh.connectivity.boundary_edges)
    corners = np.asarray(mesh.coordinates)[np.asarray(mesh.connectivity.edges)]
    wall = boundary & np.all(np.abs(corners) <= 0.25 + 1.0e-10, axis=(1, 2))
    identifiers = np.asarray(mesh.entity_set(1).entity_ids)
    registration = prepare_overset_connectivity(
        phx.meshing.MeshAssembly((background, moving)),
        (
            OversetPartSpec(background.name),
            OversetPartSpec(
                moving.name,
                wall=moving.scope(1, identifiers[wall]),
                boundary=moving.scope(1, identifiers[boundary & ~wall]),
            ),
        ),
        policy=OversetPolicy(
            fringe_layers=1,
            maximum_wall_candidate_pairs=args.overset_wall_pairs,
            maximum_donor_candidate_pairs=args.overset_donor_pairs,
        ),
    )
    return registration


def _overset_qualification_registration_evidence(connectivity: Any) -> dict[str, object]:
    """Expose blanking/search statuses, including incomplete registrations."""

    return {
        "connectivity_id": connectivity.connectivity_id,
        "epoch": connectivity.epoch,
        "predecessor_id": connectivity.predecessor_id,
        "policy_id": connectivity.policy.policy_id,
        "complete": connectivity.complete,
        "orphan_count": connectivity.orphan_count,
        "protected_wall_conflict_count": connectivity.conflict_count,
        "ambiguous_count": connectivity.ambiguous_count,
        "wall_candidate_pairs": connectivity.wall_candidate_pairs,
        "parts": [
            {
                "part": item.part_name,
                "blanking_id": item.blanking_id,
                "hole_vertex_ids": item.vertex_ids_with(
                    OversetVertexStatus.HOLE
                ).tolist(),
                "vertex_status": np.asarray(item.vertex_status).tolist(),
                "cell_status": np.asarray(item.cell_status).tolist(),
            }
            for item in connectivity.blanking
        ],
        "receptors": [
            {
                "part": item.part_name,
                "evidence_id": item.evidence_id,
                "orphan_ids": item.orphan_ids.tolist(),
                "status": np.asarray(item.status).tolist(),
                "receptor_ids": np.asarray(item.receptor_ids).tolist(),
                "donor_parts": np.asarray(item.donor_parts).tolist(),
                "donor_cells": np.asarray(item.donor_cells).tolist(),
                "candidate_parts": np.asarray(item.candidate_parts).tolist(),
                "residuals": [
                    float(value) if np.isfinite(value) else None
                    for value in np.asarray(item.residuals)
                ],
                "donor_measures": [
                    float(value) if np.isfinite(value) else None
                    for value in np.asarray(item.donor_measures)
                ],
            }
            for item in connectivity.receptors
        ],
        "donor_packet_ids": [packet.packet_id for packet in connectivity.packets],
    }


class _OversetSolveTolerance(TypedDict):
    relative: float
    absolute: float
    maximum_steps: int | None


class _OversetSolveReport(TypedDict):
    """Serialized residual evidence from the scenario's carried-state solve."""

    passed: bool
    compiler: CorpusCompilerEvidence
    solve_tolerance: _OversetSolveTolerance
    absolute_residual: float | None
    relative_residual: float | None
    carried_boundary_maximum_error: float | None
    physical_error: float | None
    error_quantity: str
    analytic_solution_l2_error: None
    retained_solver_bytes: int


def _overset_qualification_solve(
    field: phx.discretization.FiniteElementDiscretization,
    carried: jax.Array,
    repeats: int,
) -> tuple[jax.Array, _OversetSolveReport, dict[str, float]]:
    """Compile the example's carried-state Poisson update, then check it."""

    prepared, preparation = _measured(
        lambda: (
            prepare_poisson_update(field, carried),
            phx.discretization.dirichlet_constraint(field, "u"),
            poisson_solve_policy(field),
        )
    )
    problem, constraint, policy = prepared
    values, compiler, stages = _compiled_solver_stages(
        problem, carried[constraint.free_dofs], repeats, policy=policy
    )
    stages["solver_preparation"] = preparation["wall_seconds"]
    system, rhs = problem.linear_system()
    if not isinstance(rhs, jax.Array):
        raise TypeError(f"overset.rhs must be a JAX array; got {type(rhs).__name__}")
    defect = system.operator.mv(values[constraint.free_dofs]) - rhs
    absolute = float(jnp.max(jnp.abs(defect)))
    relative = float(
        jnp.linalg.norm(defect)
        / jnp.maximum(jnp.linalg.norm(rhs), jnp.finfo(rhs.dtype).eps)
    )
    boundary = float(
        jnp.max(
            jnp.abs(
                values[constraint.constrained_dofs] - carried[constraint.constrained_dofs]
            )
        )
    )
    successful = compiler["solver"]["successful"]
    passed = (
        successful
        and bool(np.all(np.isfinite((absolute, relative, boundary))))
        and absolute <= 1.0e-9
        and boundary <= 1.0e-12
    )
    return (
        values,
        {
            "passed": passed,
            "compiler": compiler,
            "solve_tolerance": {
                "relative": policy.tolerance.relative,
                "absolute": policy.tolerance.absolute,
                "maximum_steps": policy.tolerance.max_steps,
            },
            "absolute_residual": absolute if np.isfinite(absolute) else None,
            "relative_residual": relative if np.isfinite(relative) else None,
            "carried_boundary_maximum_error": boundary if np.isfinite(boundary) else None,
            "physical_error": max(relative, boundary) if passed else None,
            "error_quantity": "relative-algebraic-residual-and-carried-Dirichlet-error",
            "analytic_solution_l2_error": None,
            "retained_solver_bytes": model_array_bytes(problem),
        },
        stages,
    )


def _overset_qualification_error(report: _OversetSolveReport) -> float:
    """Narrow the successful, independently measured scalar error for comparisons."""

    error = report["physical_error"]
    if error is None or not np.isfinite(error):
        raise qualification_failure(
            "_overset_qualification_error",
            ("Accepted overset physics requires a finite scalar error."),
            {"error": error},
        )
    return error


@final
class _OversetConservativeMotion(phx.StrictModule):
    """Published interior motion and the ordered source/target composition entries."""

    source_carrier: phx.meshing.CellMeshingResult
    target_carrier: phx.meshing.CellMeshingResult
    source_certification_id: str
    target_certification_id: str
    previous: phx.meshing.OversetConnectivity
    candidate: phx.meshing.OversetConnectivity
    source_entry: phx.lifecycle.CompositionEntry
    target_entry: phx.lifecycle.CompositionEntry
    source_cells: np.ndarray
    source_points: np.ndarray
    coordinates: np.ndarray
    row: int
    centers: np.ndarray


@final
class _OversetConservativeTransport(phx.StrictModule):
    """Accepted all-role transport with its rollback and overlap evidence."""

    density: jax.Array
    states: tuple[phx.lifecycle.CompositionEntry, ...]
    composition: phx.lifecycle.Composition
    policy: phx.geometry.CommonRefinementPolicy
    prepared: phx.discretization.PreparedUnstructuredConservativeRemap
    evidence: CommonRefinementEvidence
    remap: phx.discretization.UnstructuredConservativeRemapPlan
    transports: tuple[phx.lifecycle.CompositionTransport, ...]
    refused: phx.lifecycle.CompositionRebindReceipt
    receipt: phx.lifecycle.CompositionRebindReceipt


def _overset_publish_conservative_motion(
    connectivity: phx.meshing.OversetConnectivity,
    plan: phx.meshing.NativeMeshingPlan,
    record: dict[str, Any],
    phases: list[NativeMeshingPhaseMeasurement],
    deadline: Deadline,
) -> _OversetConservativeMotion:
    """Move one interior vertex and publish its source-certified native successor."""

    _ledger = StageLedger(record, deadline=deadline)
    background = connectivity.assembly.part("background")
    source_carrier = background.carrier
    if not isinstance(source_carrier, phx.meshing.CellMeshingResult):
        raise TypeError(
            f"overset.source_carrier must be CellMeshingResult; got {type(source_carrier).__name__}"
        )
    source_certification = source_carrier.certification
    if source_certification is None:
        raise qualification_failure(
            "_overset_qualification_conservative",
            ("Moving overset inventory requires source certification."),
            {"source_certification": source_certification},
        )
    source_mesh = source_carrier.mesh
    incidence = source_mesh.connectivity

    if not isinstance(incidence, PolygonalConnectivity):
        raise TypeError(
            f"overset.incidence must be PolygonalConnectivity; got {type(incidence).__name__}"
        )
    coordinates = np.asarray(source_mesh.coordinates).copy()
    boundary = np.unique(
        np.asarray(incidence.edges)[np.asarray(incidence.boundary_edges)]
    )
    interior = np.setdiff1d(np.arange(len(coordinates)), boundary)
    if not interior.size:
        raise qualification_failure(
            "_overset_qualification_conservative",
            ("The authored native mesh has no movable interior vertex."),
            {"interior": interior},
        )
    row = int(interior[len(interior) // 2])
    neighbors = np.asarray(incidence.edges)[
        np.any(np.asarray(incidence.edges) == row, axis=1)
    ]
    local_size = float(
        np.min(
            np.linalg.norm(
                coordinates[neighbors[neighbors != row]] - coordinates[row], axis=1
            )
        )
    )
    coordinates[row] += local_size * np.asarray([0.005, -0.003])

    target, timing = _ledger.run(
        "separate_conservative_native_motion_publication",
        lambda: _measured(
            lambda: deform_planar_interior(
                background,
                coordinates,
                motion_id="qualification-conservative-interior-motion",
                plan=plan,
                record_phase=phases.append,
            )
        ),
    )
    record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
    target_carrier = target.carrier
    if not isinstance(target_carrier, phx.meshing.CellMeshingResult):
        raise TypeError(
            f"overset.target_carrier must be CellMeshingResult; got {type(target_carrier).__name__}"
        )
    target_certification = target_carrier.certification
    if target_certification is None:
        raise qualification_failure(
            "_overset_qualification_conservative",
            ("Moving overset inventory requires successor source certification."),
            {"target_certification": target_certification},
        )
    previous = connectivity
    candidate = previous.reregister({background.name: target})
    source_entry = previous.composition_entries()[0]
    target_entry = candidate.composition_entries()[0]
    source_cells = _mesh_triangles(source_mesh)
    source_points = np.asarray(source_mesh.coordinates)
    centers = source_points[source_cells].mean(axis=1)
    return _OversetConservativeMotion(
        source_carrier=source_carrier,
        target_carrier=target_carrier,
        source_certification_id=source_certification.report_id,
        target_certification_id=target_certification.report_id,
        previous=previous,
        candidate=candidate,
        source_entry=source_entry,
        target_entry=target_entry,
        source_cells=source_cells,
        source_points=source_points,
        coordinates=coordinates,
        row=row,
        centers=centers,
    )


def _overset_commit_conservative_transport(
    motion_state: _OversetConservativeMotion,
    args: QualificationControls,
    record: dict[str, Any],
    deadline: Deadline,
) -> _OversetConservativeTransport:
    """Transport physical, material, and history inventories through one accepted rebind."""

    _ledger = StageLedger(record, deadline=deadline)
    centers = motion_state.centers
    previous, candidate = motion_state.previous, motion_state.candidate
    source_entry, target_entry = motion_state.source_entry, motion_state.target_entry
    density = jnp.asarray(2.0 + centers[:, 0] / 8.0 - centers[:, 1] / 10.0)
    fraction = jnp.asarray(0.5 + centers[:, 0] / 20.0)
    values = (
        density,
        jnp.stack((fraction, 1.0 - fraction), axis=1),
        jnp.stack((density, density**2), axis=1),
    )
    states = tuple(
        phx.lifecycle.CompositionEntry(
            value,
            entry_id=f"conservative:{name}",
            role=role,
            owner_id="qualification-native-moving-inventory",
            structure_id=source_entry.structure_id,
            revision_id=f"accepted:{name}",
            semantics_id=meaning,
            dependencies=(source_entry.binding("revision"),),
        )
        for name, role, meaning, value in zip(
            ("density", "materials", "history"),
            ("physical-state", "model-state", "history"),
            (
                "normalized-cell-density",
                "two-material-volume-fractions",
                "spatial-density-history",
            ),
            values,
            strict=True,
        )
    )
    composition = phx.lifecycle.Composition(
        (*previous.composition_entries(), *states),
        boundary_id="accepted-conservative-state",
    )
    policy = phx.geometry.CommonRefinementPolicy(
        maximum_candidate_pairs=args.overset_overlap_pairs,
        maximum_accepted_pairs=args.overset_overlap_pairs,
        maximum_memory_bytes=4096 * args.capacity,
        coverage_tolerance=1.0e-10,
    )

    payload, timing = _ledger.run(
        "separate_moving_conservative_state_transport",
        lambda: _measured(
            lambda: prepare_overset_conservative_state_transport(
                source_entry,
                target_entry,
                states,
                content_tolerance=1.0e-9,
                policy=policy,
            )
        ),
    )
    prepared, transports = payload
    record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
    evidence, remap = prepared.evidence, prepared.plan
    if not isinstance(evidence, CommonRefinementEvidence) or remap is None:
        raise TypeError(
            f"overset.transport must include CommonRefinementEvidence and a remap route; got evidence={type(evidence).__name__}, remap={remap!r}"
        )
    motion = prepare_overset_motion_rebind(
        composition, previous, candidate, transports=transports
    )
    refused = motion.commit(accepted_boundary=False)
    receipt = motion.commit(accepted_boundary=True)
    if (
        refused.published
        or refused.composition is not composition
        or not receipt.published
    ):
        raise qualification_failure(
            "_overset_qualification_conservative",
            ("Moving conservative state violated rollback or accepted publication."),
            {"composition": composition, "receipt": receipt, "refused": refused},
        )
    return _OversetConservativeTransport(
        density=density,
        states=states,
        composition=composition,
        policy=policy,
        prepared=prepared,
        evidence=evidence,
        remap=remap,
        transports=transports,
        refused=refused,
        receipt=receipt,
    )


def _overset_check_conservative_inventories(
    motion_state: _OversetConservativeMotion,
    transport: _OversetConservativeTransport,
    plan: phx.meshing.NativeMeshingPlan,
    record: dict[str, Any],
) -> None:
    """Check transported inventories, material closure, analytic content, and constants."""

    target_carrier = motion_state.target_carrier
    source_cells, source_points = motion_state.source_cells, motion_state.source_points
    density, states = (transport.density, transport.states)
    remap = transport.remap
    transports, receipt = (transport.transports, transport.receipt)
    target_mesh = target_carrier.mesh
    source_measures = _simplex_measures(source_points, source_cells)
    target_measures = _simplex_measures(
        np.asarray(target_mesh.coordinates), _mesh_triangles(target_mesh)
    )
    inventories = []
    targets = tuple(transport.targets[0] for transport in transports)
    for old, new in zip(states, targets, strict=True):
        trailing = (1,) * (old.value.ndim - 1)
        old_content = np.sum(
            source_measures.reshape((-1,) + trailing) * np.asarray(old.value), axis=0
        )
        new_content = np.sum(
            target_measures.reshape((-1,) + trailing) * np.asarray(new.value), axis=0
        )
        defect = float(np.max(np.abs(old_content - new_content)))
        if defect > 1.0e-9 or new.semantics_id != old.semantics_id:
            raise qualification_failure(
                "_overset_qualification_conservative",
                (
                    "Moving conservative state lost physical/material/history content or identity."
                ),
                {"defect": defect, "new": new, "old": old},
            )
        inventories.append(
            {
                "entry_id": old.entry_id,
                "role": old.role,
                "semantics_id": old.semantics_id,
                "source": np.asarray(old_content).tolist(),
                "target": np.asarray(new_content).tolist(),
                "independent_inventory_absolute_error": defect,
            }
        )
    material = receipt.composition.entry("conservative:materials").value
    if (
        float(jnp.max(jnp.abs(jnp.sum(material, axis=1) - 1.0))) > 1.0e-10
        or float(jnp.min(material)) < 0.0
    ):
        raise qualification_failure(
            "_overset_qualification_conservative",
            ("Moving conservative material fractions lost closure or positivity."),
            {"material": material},
        )
    authority = plan.source
    if not isinstance(authority, phx.meshing.NativePlanarSource):
        raise TypeError(
            f"overset.plan.source must be NativePlanarSource; got {type(authority).__name__}"
        )
    edge_points = np.asarray(authority.region.vertices)[
        np.asarray(authority.region.edges)
    ]
    first, second = edge_points[:, 0], edge_points[:, 1]
    cross = first[:, 0] * second[:, 1] - second[:, 0] * first[:, 1]
    analytic_inventory = float(
        np.sum(cross)
        + np.sum((first[:, 0] + second[:, 0]) * cross) / 48.0
        - np.sum((first[:, 1] + second[:, 1]) * cross) / 60.0
    )
    source_inventory = float(np.sum(source_measures * np.asarray(density)))
    constant_error = float(jnp.max(jnp.abs(remap.apply(jnp.ones_like(density)) - 1.0)))
    minimum_density = float(
        jnp.min(receipt.composition.entry("conservative:density").value)
    )
    if (
        abs(source_inventory - analytic_inventory) > 1.0e-9
        or constant_error > 1.0e-10
        or minimum_density <= 0.0
    ):
        raise qualification_failure(
            "_overset_qualification_conservative",
            (
                "Moving conservative inventory lost its analytic source integral, constants, or positivity."
            ),
            {
                "analytic_inventory": analytic_inventory,
                "constant_error": constant_error,
                "minimum_density": minimum_density,
                "source_inventory": source_inventory,
            },
        )
    _overset_record_conservative_profile(
        motion_state,
        transport,
        record,
        inventories,
        analytic_inventory,
        source_inventory,
        minimum_density,
        constant_error,
    )


def _overset_record_conservative_profile(
    motion_state: _OversetConservativeMotion,
    transport: _OversetConservativeTransport,
    record: dict[str, Any],
    inventories: list[dict[str, Any]],
    analytic_inventory: float,
    source_inventory: float,
    minimum_density: float,
    constant_error: float,
) -> None:
    """Publish the independent conservation measurements and retained overlap evidence."""
    source_mesh, target_mesh = (
        motion_state.source_carrier.mesh,
        motion_state.target_carrier.mesh,
    )
    coordinates, row, source_points = (
        motion_state.coordinates,
        motion_state.row,
        motion_state.source_points,
    )
    policy, prepared, evidence = transport.policy, transport.prepared, transport.evidence
    transports, refused, receipt, composition = (
        transport.transports,
        transport.refused,
        transport.receipt,
        transport.composition,
    )
    record["conservative_profile"] = {
        "profile": "moving-native-interior-mesh-cell-average-complete-overlap",
        "moving_fe_transfer_certified": False,
        "moving_cell_average_transfer_certified": True,
        "physical_boundary_moved": False,
        "interior_vertex_global_id": int(source_mesh.vertex_global_ids[row]),
        "maximum_coordinate_displacement": float(
            np.max(np.abs(coordinates - source_points))
        ),
        "source_geometry_id": source_mesh.geometry_id,
        "target_geometry_id": target_mesh.geometry_id,
        "policy_id": policy.policy_id,
        "remap_id": prepared.remap_id,
        "status": prepared.status.name,
        "evidence_id": evidence.evidence_id,
        "candidate_pair_count": evidence.candidate_pair_count,
        "accepted_pair_count": evidence.accepted_pair_count,
        "piece_pair_count": evidence.piece_pair_count,
        "source_gap_count": evidence.source_gap_count,
        "target_gap_count": evidence.target_gap_count,
        "source_double_count": evidence.source_double_count,
        "target_double_count": evidence.target_double_count,
        "retained_bytes": evidence.retained_bytes,
        "working_bytes": evidence.working_bytes,
        "inventories": inventories,
        "transport_ids": [transport.transport_id for transport in transports],
        "analytic_source_inventory": analytic_inventory,
        "source_inventory": source_inventory,
        "minimum_target_average": minimum_density,
        "rollback_preserved": refused.composition is composition,
        "accepted_composition_id": receipt.composition.composition_id,
        "source_certification_id": motion_state.source_certification_id,
        "target_certification_id": motion_state.target_certification_id,
        "constant_reproduction_error": constant_error,
    }


def _overset_qualification_conservative(
    connectivity: phx.meshing.OversetConnectivity,
    plan: phx.meshing.NativeMeshingPlan,
    args: QualificationControls,
    record: dict[str, Any],
    phases: list[NativeMeshingPhaseMeasurement],
    deadline: Deadline,
) -> None:
    """Move the native mesh and commit fields/material/history inventories."""
    motion = _overset_publish_conservative_motion(
        connectivity, plan, record, phases, deadline
    )
    transport = _overset_commit_conservative_transport(motion, args, record, deadline)
    _overset_check_conservative_inventories(motion, transport, plan, record)


def _overset_qualification_fields(
    connectivity: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retain the declared physical plans without inferring them from coefficients."""

    plans = field_plans(connectivity)
    return plans, {name: plan.prepare() for name, plan in plans.items()}


def _overset_qualification_motion(
    connectivity: Any,
    composition: Any,
    coefficients: dict[str, Any],
    successor: Any,
    step: int,
    args: QualificationControls,
    record: dict[str, Any],
    deadline: Deadline,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Join source-aware motion to FE remap, rollback, commit, and physics."""

    _ledger = StageLedger(record, deadline=deadline)

    offset = 0.55 if step == 0 else 0.05

    candidate, timing = _ledger.run(
        f"motion-{step}:wall_cutting_donor_refresh",
        lambda: _measured(lambda: connectivity.reregister({"moving": successor})),
    )
    record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
    epoch: dict[str, Any] = {
        "step": step,
        "offset": offset,
        "source_successor_result_id": successor.carrier.result_id,
        "source_successor_certification_id": successor.carrier.certification.report_id,
        "registration": _overset_qualification_registration_evidence(candidate),
    }
    record["epochs"].append(epoch)
    candidate.require_complete()

    field_payload, timing = _ledger.run(
        f"motion-{step}:field_and_transfer_preparation",
        lambda: _measured(lambda: _overset_qualification_fields(candidate)),
    )
    next_plans, next_fields = field_payload
    record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
    route, timing = _ledger.run(
        record.get("active_phase", "prepare_overset_field_transfer"),
        lambda: _measured(
            lambda: prepare_overset_field_transfer(
                candidate, next_fields, "u", previous=connectivity
            )
        ),
    )
    record["stages_seconds"][f"motion-{step}:prepared_field_query"] = timing[
        "wall_seconds"
    ]

    receptors, timing = _ledger.run(
        f"motion-{step}:state_transfer",
        lambda: _measured(lambda: jax.block_until_ready(route.apply(coefficients))),
    )
    record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
    constant = route.apply(
        {name: jnp.ones_like(value) for name, value in coefficients.items()}
    )
    constant_error = max(
        (
            float(jnp.max(jnp.abs(value - 1.0)))
            for value in constant.values()
            if value.size
        ),
        default=0.0,
    )
    if route.conservative or constant_error > 1.0e-10:
        raise qualification_failure(
            "_overset_qualification_motion",
            ("FE interpolation lost constants or mislabeled its conservation contract."),
            {"constant_error": constant_error, "route": route},
        )
    next_values = dict(coefficients)
    for receptor in route.receptors:
        next_values[receptor.part_name] = (
            next_values[receptor.part_name]
            .at[receptor.receptor_rows]
            .set(receptors[receptor.part_name])
        )
    prepared, states = field_entries(
        candidate, next_fields, next_values, f"motion-{step}"
    )
    old_states = tuple(
        composition.entry(f"temperature:{name}") for name in sorted(coefficients)
    )
    transport = phx.lifecycle.CompositionTransport(
        "physical-remap",
        tuple(entry.entry_id for entry in old_states),
        states,
        source_structure_ids=tuple(entry.structure_id for entry in old_states),
        route_id=route.transfer_id,
        successful=True,
    )

    motion, timing = _ledger.run(
        f"motion-{step}:composition_rebind",
        lambda: _measured(
            lambda: prepare_overset_motion_rebind(
                composition,
                connectivity,
                candidate,
                transports=(transport,),
                reprepare=prepared,
            )
        ),
    )
    record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
    refused, timing = _ledger.run(
        record.get("active_phase", "commit"),
        lambda: _measured(lambda: motion.commit(accepted_boundary=False)),
    )
    record["stages_seconds"][f"motion-{step}:refused_boundary"] = timing["wall_seconds"]
    if refused.published or refused.composition is not composition:
        raise qualification_failure(
            "_overset_qualification_motion",
            ("Refused overset motion changed the accepted Composition."),
            {"composition": composition, "refused": refused},
        )
    receipt, timing = _ledger.run(
        record.get("active_phase", "commit"),
        lambda: _measured(lambda: motion.commit(accepted_boundary=True)),
    )
    record["stages_seconds"][f"motion-{step}:accepted_boundary"] = timing["wall_seconds"]
    if not receipt.published:
        raise qualification_failure(
            "_overset_qualification_motion",
            ("Source-aware motion failed its accepted Composition boundary."),
            {"receipt": receipt},
        )
    epoch["transfer"] = {
        "transfer_id": route.transfer_id,
        "prepared_queries": [
            {
                "query_id": query.query_id,
                "complete": query.complete,
                "coverage": query.coverage,
                "status": np.asarray(query.evidence.status).tolist(),
                "support_count": np.asarray(query.evidence.support_count).tolist(),
                "conditioning": [
                    float(value) if np.isfinite(value) else None
                    for value in np.asarray(query.evidence.conditioning)
                ],
                "unreported_conditioning_is_unavailable": True,
            }
            for query in route.queries
        ],
        "previous_connectivity_id": route.previous_id,
        "conservative": route.conservative,
        "constant_reproduction_maximum_error": constant_error,
        "retained_transfer_bytes": model_array_bytes(route),
        "rollback_preserved": refused.composition is composition,
        "accepted_boundary_published": receipt.published,
        "target_composition_id": receipt.composition.composition_id,
    }

    next_values["moving"], solver, stages = _ledger.run(
        f"motion-{step}:carried_state_poisson",
        lambda: _overset_qualification_solve(
            next_fields["moving"], next_values["moving"], args.repeats
        ),
    )
    record["stages_seconds"].update(
        {f"motion-{step}:{name}": value for name, value in stages.items()}
    )
    epoch["poisson"] = solver
    if not solver["passed"]:
        raise qualification_failure(
            "_overset_qualification_motion",
            ("Carried-state Poisson update failed independent residual/boundary checks."),
            {"solver": solver},
        )
    prepared, states = field_entries(
        candidate, next_fields, next_values, f"poisson-{step}"
    )
    accepted = phx.lifecycle.Composition(
        (*candidate.composition_entries(), *prepared, *states),
        boundary_id="accepted-poisson-step",
    )
    record["attempts"].append(
        {
            "phase": record["active_phase"],
            "status": "passed",
            "connectivity_id": candidate.connectivity_id,
        }
    )
    return candidate, accepted, next_fields, next_values, next_plans


def _overset_qualification_checkpoint(
    connectivity: Any,
    composition: Any,
    plans: dict[str, Any],
    args: QualificationControls,
    record: dict[str, Any],
    physical_plans: dict[str, Any],
    discretizations: dict[str, Any],
    deadline: Deadline,
) -> tuple[
    phx.meshing.OversetConnectivity,
    dict[str, jax.Array],
    dict[str, phx.discretization.FiniteElementDiscretization],
    dict[str, Any],
]:
    """Persist scientific carriers and accepted data through the native codec."""

    _ledger = StageLedger(record, deadline=deadline)

    byte_budget = 4096 * args.capacity
    limits = ArrayArchiveLimits(
        max_container_bytes=byte_budget,
        max_aggregate_bytes=byte_budget,
        max_member_bytes=byte_budget,
        max_manifest_bytes=1024 * args.capacity,
        max_members=64 * args.capacity,
        max_central_directory_bytes=1024 * args.capacity,
        max_axis_length=64 * args.capacity,
        max_array_elements=64 * args.capacity,
        max_total_array_elements=512 * args.capacity,
        max_manifest_nesting=128,
    )
    policy = asdict(limits)
    policy["allowed_dtype_kinds"] = sorted(limits.allowed_dtype_kinds)
    names = tuple(sorted(plans))
    states = {name: composition.entry(f"temperature:{name}") for name in names}
    background = connectivity.assembly.part("background").carrier
    declarations = field_declarations(
        connectivity,
        physical_plans,
        discretizations,
        {name: states[name].value for name in names},
    )
    records = {
        "certification_inputs": background.certification.request,
        "report": background.certification,
        "associations": background.associations,
        "registration": connectivity.registration(),
        "primary_generation_part": "background",
        "generation_sources": {name: plans[name].source for name in names},
        "generation_specifications": {name: plans[name].specification for name in names},
        "generation_options": {name: plans[name].options for name in names},
        "field_declarations": declarations,
        "accepted_data": {
            "fields": {states[name].entry_id: states[name].value for name in names}
        },
    }
    evidence: dict[str, Any] = {
        "archive_limits": policy,
        "source_authority_stripped": False,
        "prepared_fe_or_overset_caches_serialized": False,
        "physical_state_bindings": [
            {
                "part_name": name,
                "entry_id": states[name].entry_id,
                "structure_id": states[name].structure_id,
                "revision_id": states[name].revision_id,
                "semantics_id": states[name].semantics_id,
                "owner_id": states[name].owner_id,
            }
            for name in names
        ],
    }
    record["checkpoint"] = evidence
    with TemporaryDirectory(prefix="phydrax-native-overset-source-") as directory:
        receipt, timing = _ledger.run(
            "native_checkpoint_publication",
            lambda: _measured(
                lambda: write_meshing_source_closure(
                    Path(directory) / "overset.checkpoint.zip", records, limits=limits
                )
            ),
        )
        record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
        evidence["content_id"] = receipt.content_id
        evidence["container_bytes"] = receipt.path.stat().st_size

        restored, timing = _ledger.run(
            "native_checkpoint_verified_restore",
            lambda: _measured(
                lambda: read_meshing_source_closure(
                    receipt.path, expected_content_id=receipt.content_id, limits=limits
                )
            ),
        )
        record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
        for name in names:
            if (
                restored["generation_sources"][name].binding_id
                != plans[name].source.binding_id
            ):
                raise qualification_failure(
                    "_overset_qualification_checkpoint",
                    ("Checkpoint changed the authored source recipe."),
                    {"name": name, "plans": plans, "restored": restored},
                )
            if (
                restored["generation_specifications"][name].specification_id
                != plans[name].specification.specification_id
            ):
                raise qualification_failure(
                    "_overset_qualification_checkpoint",
                    ("Checkpoint changed the hard native request."),
                    {"name": name, "plans": plans, "restored": restored},
                )
        if set(restored["accepted_data"]["fields"]) != {
            states[name].entry_id for name in names
        }:
            raise qualification_failure(
                "_overset_qualification_checkpoint",
                ("Checkpoint lost or relabeled accepted physical-state identities."),
                {"names": names, "restored": restored, "states": states},
            )

        rebuilt, timing = _ledger.run(
            "native_restart_wall_donor_repreparation",
            lambda: _measured(restored["registration"].prepare),
        )
        record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
        if not isinstance(rebuilt, phx.meshing.OversetConnectivity):
            raise TypeError(
                f"overset.rebuilt must be OversetConnectivity; got {type(rebuilt).__name__}"
            )
        rebuilt.require_complete()
        values = {
            name: jnp.asarray(restored["accepted_data"]["fields"][states[name].entry_id])
            for name in names
        }

        def restore_fields() -> dict[str, phx.discretization.FiniteElementDiscretization]:
            fields: dict[str, phx.discretization.FiniteElementDiscretization] = {}
            for name in names:
                prepared, _ = _ledger.run(
                    "native_restart_physical_field_repreparation",
                    lambda: prepare_meshing_field_owner(
                        restored["field_declarations"][name],
                        rebuilt.assembly.part(name).carrier,
                    ),
                )
                if not isinstance(
                    prepared, phx.discretization.FiniteElementDiscretization
                ):
                    raise TypeError(
                        f"overset.prepared must be FiniteElementDiscretization; got {type(prepared).__name__}"
                    )
                fields[name] = prepared
            return fields

        restored_fields, timing = _ledger.run(
            record.get("active_phase", "measured"), lambda: _measured(restore_fields)
        )
        record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
    evidence["current_source_and_hard_request_identities_reproduced"] = True
    evidence["original_physical_declarations_and_field_space_ids_reproduced"] = True
    record["attempts"].append(
        {
            "phase": record["active_phase"],
            "status": "passed",
            "content_id": receipt.content_id,
        }
    )
    return rebuilt, values, restored_fields, evidence


def _overset_qualification_restart(
    connectivity: Any,
    discretizations: dict[str, Any],
    coefficients: dict[str, Any],
    args: QualificationControls,
    record: dict[str, Any],
    restored_payload: tuple[
        phx.meshing.OversetConnectivity,
        dict[str, jax.Array],
        dict[str, phx.discretization.FiniteElementDiscretization],
        dict[str, Any],
    ],
    deadline: Deadline,
) -> None:
    """Restore the native recipe and continue the carried-state solve."""

    _ledger = StageLedger(record, deadline=deadline)
    restored, restored_values, restored_fields, evidence = restored_payload
    record["restart"] = {"checkpoint_content_id": evidence["content_id"]}
    record["restart"]["restored_registration"] = (
        _overset_qualification_registration_evidence(restored)
    )
    if restored.connectivity_id != connectivity.connectivity_id:
        raise qualification_failure(
            "_overset_qualification_restart",
            ("Native overset restart changed accepted wall/donor registration."),
            {"connectivity": connectivity, "restored": restored},
        )
    for name, value in coefficients.items():
        if not np.array_equal(np.asarray(restored_values[name]), np.asarray(value)):
            raise qualification_failure(
                "_overset_qualification_restart",
                ("Native overset restart changed accepted physical coefficients."),
                {"name": name, "restored_values": restored_values, "value": value},
            )
        before, after = (
            connectivity.assembly.part(name).carrier,
            restored.assembly.part(name).carrier,
        )
        if not isinstance(before, phx.meshing.CellMeshingResult) or not isinstance(
            after, phx.meshing.CellMeshingResult
        ):
            raise TypeError(
                f"overset.restart.carriers must be CellMeshingResult; got before={type(before).__name__}, after={type(after).__name__}"
            )
        before_certification, after_certification = (
            before.certification,
            after.certification,
        )
        if before_certification is None or after_certification is None:
            raise qualification_failure(
                "_overset_qualification_restart",
                ("Native overset restart omitted carrier certification."),
                {
                    "after_certification": after_certification,
                    "before_certification": before_certification,
                },
            )
        if (
            before.result_id != after.result_id
            or before_certification.report_id != after_certification.report_id
        ):
            raise qualification_failure(
                "_overset_qualification_restart",
                ("Native overset restart changed source-certified carrier identity."),
                {
                    "after": after,
                    "after_certification": after_certification,
                    "before": before,
                    "before_certification": before_certification,
                },
            )
        before_spaces = {
            space.name: space.field_space_id
            for space in discretizations[name].field_spaces
        }
        after_spaces = {
            space.name: space.field_space_id
            for space in restored_fields[name].field_spaces
        }
        if before_spaces != after_spaces:
            raise qualification_failure(
                "_overset_qualification_restart",
                ("Native overset restart changed the declared physical field owner."),
                {"after_spaces": after_spaces, "before_spaces": before_spaces},
            )

    replay, replay_solver, stages = _ledger.run(
        "restart_continued_poisson",
        lambda: _overset_qualification_solve(
            restored_fields["moving"], restored_values["moving"], args.repeats
        ),
    )
    record["stages_seconds"].update(
        {f"restart:{name}": value for name, value in stages.items()}
    )
    uninterrupted, reference_solver, stages = _overset_qualification_solve(
        discretizations["moving"], coefficients["moving"], args.repeats
    )
    record["stages_seconds"].update(
        {f"uninterrupted:{name}": value for name, value in stages.items()}
    )
    error = float(jnp.max(jnp.abs(replay - uninterrupted)))
    record["restart"].update(
        {
            "continued_poisson": replay_solver,
            "uninterrupted_poisson": reference_solver,
            "continuation_maximum_error": error if np.isfinite(error) else None,
            "physical_coefficients_reproduced": True,
            "source_certified_carriers_reproduced": True,
        }
    )
    if (
        not replay_solver["passed"]
        or not reference_solver["passed"]
        or not np.isfinite(error)
        or error > 1.0e-12
    ):
        raise qualification_failure(
            "_overset_qualification_restart",
            (
                "Native restart continued solve disagreed with uninterrupted physical update."
            ),
            {
                "error": error,
                "reference_solver": reference_solver,
                "replay_solver": replay_solver,
            },
        )
    record["physical_error"] = max(
        error,
        _overset_qualification_error(replay_solver),
        _overset_qualification_error(reference_solver),
    )


def _overset_qualification_execute(
    args: QualificationControls,
    record: dict[str, Any],
    phases: list[NativeMeshingPhaseMeasurement],
    started: float,
    deadline: Deadline,
) -> None:
    """Drive every native caller handoff without substituting certified raw grids."""

    _ledger = StageLedger(record, deadline=deadline)

    background, background_plan = _overset_qualification_part(
        "background", args, record, phases, deadline
    )
    moving, moving_plan = _overset_qualification_part(
        "moving", args, record, phases, deadline
    )

    connectivity, timing = _ledger.run(
        "native_wall_cutting_donor_search",
        lambda: _measured(
            lambda: _overset_qualification_connectivity(background, moving, args)
        ),
    )
    record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
    record["connectivity"] = _overset_qualification_registration_evidence(connectivity)
    connectivity.require_complete()
    _overset_qualification_conservative(
        connectivity, background_plan, args, record, phases, deadline
    )

    field_payload, timing = _ledger.run(
        "initial_actual_fe_preparation",
        lambda: _measured(lambda: _overset_qualification_fields(connectivity)),
    )
    physical_plans, discretizations = field_payload
    record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
    coefficients = {
        name: 5.0 + jnp.sum(field.dof_maps[0].dof_coordinates, axis=1)
        for name, field in discretizations.items()
    }
    recipe_bindings: dict[str, dict[str, str]] = {}
    for name, plan in (("background", background_plan), ("moving", moving_plan)):
        source = plan.source
        if not isinstance(source, phx.meshing.NativePlanarSource):
            raise TypeError(
                f"overset.plan.source must be NativePlanarSource; got {type(source).__name__}"
            )
        recipe_bindings[name] = {
            "source_binding_id": source.binding_id,
            "request_id": plan.specification.specification_id,
            "plan_id": plan.plan_id,
        }
    record["generation_recipe_bindings"] = recipe_bindings

    coefficients["moving"], solver, stages = _ledger.run(
        "initial_carried_state_poisson",
        lambda: _overset_qualification_solve(
            discretizations["moving"], coefficients["moving"], args.repeats
        ),
    )
    record["stages_seconds"].update(
        {f"initial:{name}": value for name, value in stages.items()}
    )
    record["initial_poisson"] = solver
    if not solver["passed"]:
        raise qualification_failure(
            "_overset_qualification_execute",
            (
                "Initial carried-state Poisson update failed independent residual/boundary checks."
            ),
            {"solver": solver},
        )
    record["physical_error"] = _overset_qualification_error(solver)
    prepared, states = field_entries(
        connectivity, discretizations, coefficients, "initial-poisson"
    )
    composition = phx.lifecycle.Composition(
        (*connectivity.composition_entries(), *prepared, *states),
        boundary_id="accepted-poisson-step",
    )
    record["initial_composition_id"] = composition.composition_id
    record["motion_sources"] = []
    for step in range(args.overset_motion_steps):
        if perf_counter() - started > args.timeout:
            raise TimeoutError(
                f"overset.elapsed_seconds must not exceed {args.timeout!r} before motion; got {perf_counter() - started!r}"
            )
        offset = np.asarray((0.55 if step == 0 else 0.05, 0.0), dtype=np.float64)

        successor_payload, timing = _ledger.run(
            f"motion-{step}:source_successor",
            lambda: _measured(
                lambda: rigid_planar_successor(
                    connectivity.assembly.part("moving"),
                    offset,
                    motion_id=f"qualification-accepted-motion-{step}",
                    plan=moving_plan,
                    record_phase=phases.append,
                )
            ),
        )
        successor, moving_plan = successor_payload
        moving_source, successor_carrier = moving_plan.source, successor.carrier
        if not isinstance(moving_source, phx.meshing.NativePlanarSource):
            raise TypeError(
                f"overset.moving_source must be NativePlanarSource; got {type(moving_source).__name__}"
            )
        if not isinstance(successor_carrier, phx.meshing.CellMeshingResult):
            raise TypeError(
                f"overset.successor_carrier must be CellMeshingResult; got {type(successor_carrier).__name__}"
            )
        record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
        record["motion_sources"].append(
            {
                "source_binding_id": moving_source.binding_id,
                "source_revision": moving_source.source_revision,
                "request_id": moving_plan.specification.specification_id,
                "limits_id": moving_plan.specification.limits.limits_id,
                "prepared_id": moving_plan.plan_id,
                "result_id": successor_carrier.result_id,
            }
        )
        connectivity, composition, discretizations, coefficients, physical_plans = (
            _overset_qualification_motion(
                connectivity,
                composition,
                coefficients,
                successor,
                step,
                args,
                record,
                deadline,
            )
        )
    record["physical_error"] = max(
        record["physical_error"],
        *(epoch["poisson"]["physical_error"] for epoch in record["epochs"]),
    )
    record["final_connectivity_id"] = connectivity.connectivity_id
    record["final_composition_id"] = composition.composition_id
    record["retained_registration_bytes"] = model_array_bytes(connectivity)
    restored_payload = _overset_qualification_checkpoint(
        connectivity,
        composition,
        {"background": background_plan, "moving": moving_plan},
        args,
        record,
        physical_plans,
        discretizations,
        deadline,
    )
    _overset_qualification_restart(
        connectivity,
        discretizations,
        coefficients,
        args,
        record,
        restored_payload,
        deadline,
    )
    record["physical_error"] = max(
        record["physical_error"],
        _overset_qualification_error(solver),
        *(epoch["poisson"]["physical_error"] for epoch in record["epochs"]),
    )
    record["target_error_reached"] = record["physical_error"] <= args.target_error
    if record["target_error_reached"] and perf_counter() - started <= args.timeout:
        record["time_to_target_error_seconds"] = perf_counter() - started
        record["status"] = "passed"
    else:
        record["failure"] = (
            "The frozen residual/boundary target was not reached within the wall budget."
        )


def _overset_qualification_lifecycle(args: QualificationControls) -> dict[str, object]:
    """Native sources → wall/donor refresh → remap/rebind → Poisson → native restart."""

    if (
        any(
            value < 1
            for value in (
                args.resolution,
                args.capacity,
                args.repeats,
                args.overset_wall_pairs,
                args.overset_donor_pairs,
                args.overset_overlap_pairs,
            )
        )
        or args.overset_motion_steps < 2
    ):
        raise ValueError(
            f"overset.controls must have positive capacities, repeats, and resolution with at least two motion steps; got resolution={args.resolution!r}, capacity={args.capacity!r}, repeats={args.repeats!r}, wall_pairs={args.overset_wall_pairs!r}, donor_pairs={args.overset_donor_pairs!r}, overlap_pairs={args.overset_overlap_pairs!r}, motion_steps={args.overset_motion_steps!r}"
        )
    if (
        not np.isfinite(args.timeout)
        or args.timeout <= 0.0
        or not np.isfinite(args.target_error)
        or args.target_error <= 0.0
    ):
        raise ValueError(
            f"overset.controls must have finite positive timeout and residual target; got timeout={args.timeout!r}, target_error={args.target_error!r}"
        )
    started = perf_counter()
    deadline = Deadline(started + args.timeout)
    record: dict[str, Any] = {
        "scenario": "native-overset-lifecycle",
        "status": "failed",
        "runtime": _runtime_identity(),
        "engine": "native",
        "expected_admission": "positive",
        "physics_preparation_owner": (
            f"{prepare_poisson_update.__module__}.{prepare_poisson_update.__qualname__}"
        ),
        "requirements": _overset_qualification_requirements(),
        "control_capacities": {
            "resolution": args.resolution,
            "entity_capacity": args.capacity,
            "motion_steps": args.overset_motion_steps,
            "fringe_layers": 1,
            "maximum_wall_candidate_pairs": args.overset_wall_pairs,
            "maximum_donor_candidate_pairs": args.overset_donor_pairs,
            "maximum_overlap_candidate_pairs": args.overset_overlap_pairs,
            "maximum_overlap_accepted_pairs": args.overset_overlap_pairs,
            "maximum_overlap_memory_bytes": 4096 * args.capacity,
        },
        "targets": {
            "independent_relative_residual_and_boundary_error": args.target_error,
            "wall_seconds": args.timeout,
        },
        "parts": {},
        "stages_seconds": {},
        "epochs": [],
        "attempts": [],
        "physical_error": None,
        "time_to_target_error_seconds": None,
        "positive_prerequisite": None,
        "mandatory_workflow_completion": "unassessed",
        "full_corpus_completion": "unassessed",
        "release_evidence": "unassessed",
        "leadership_evidence": "unassessed",
        "optional_external_baseline": "not-requested",
        "active_phase": "source_preparation",
    }
    phases: list[NativeMeshingPhaseMeasurement] = []
    try:
        _overset_qualification_execute(args, record, phases, started, deadline)
    except phx.meshing.MeshingFailure as failure:
        record_failure(record, failure, record["active_phase"])
    except OversetConnectivityError as failure:
        record_failure(record, failure, record["active_phase"])
        record["failed_registration"] = _overset_qualification_registration_evidence(
            failure.connectivity
        )
    except (RuntimeError, ValueError, TypeError, OSError) as failure:
        record_failure(record, failure, record["active_phase"])
        if isinstance(failure, TimeoutError):
            record["status"] = "timeout"
    finally:
        elapsed = perf_counter() - started
        record["end_to_end_seconds"] = elapsed
        record["native_phases"] = _phase_recordings(phases)
        record["peak_process_resident_bytes"] = peak_resident_bytes(resource.RUSAGE_SELF)
        record["process_peak_scope"] = "process-lifetime"
        record["device_memory"] = [
            {
                "device_id": device.id,
                "platform": device.platform,
                "status": "unavailable" if statistics is None else "backend-reported",
                "peak_bytes": None
                if statistics is None
                else statistics.get("peak_bytes_in_use"),
                "memory_statistics": statistics,
            }
            for device in jax.local_devices()
            for statistics in (device.memory_stats(),)
        ]
        if elapsed > args.timeout:
            record["status"] = "timeout"
            record["time_to_target_error_seconds"] = None
            record["wall_budget_evidence"] = {
                "requested_wall_seconds": args.timeout,
                "achieved_wall_seconds": elapsed,
                "phase": record["active_phase"],
                "status": "timeout",
            }
        if record["status"] != "passed":
            record["failure_phase"] = record["active_phase"]
            record["attempts"].append(
                {
                    "phase": record["active_phase"],
                    "status": record["status"],
                    "failure": record.get("failure"),
                }
            )
            record["positive_prerequisite"] = {
                "phase": record["active_phase"],
                "detail": record.get("failure"),
                "source_authority_stripped": False,
                "safe_refusal_is_positive": False,
                "owner": (
                    "DistributedNative.NativeSourceArchiveRegistry"
                    if any(
                        stage in record["active_phase"]
                        for stage in ("checkpoint", "restart")
                    )
                    else "OversetResume"
                    if "motion" in record["active_phase"]
                    else "native-source-owner"
                ),
            }
        del record["active_phase"]
    return record


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return _overset_qualification_lifecycle(args)
