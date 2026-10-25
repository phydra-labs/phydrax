# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, final

import equinox as eqx
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
from phydrax._frozendict import frozendict
from phydrax._meshcore import (
    NativeExecutionScope,
    NativeHostStorageWorkspace,
    NativeParallelExecution,
)
from phydrax.discretization._cell_geometry import coordinate_lagrange_element
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_operands import CoordinateOperandCollector
from phydrax.discretization.fem._reference_types import FiniteElementSpec
from phydrax.geometry.brep._query_projection import NativeBRepProjection
from phydrax.geometry.brep._request_preparations import TrimRequestPreparations
from phydrax.lifecycle import (
    MeshingAcceptedEpoch,
    MeshingFieldStateRole,
    MeshingFiniteElementFieldDeclaration,
    prepare_meshing_field_index_binding,
    read_meshing_source_closure,
    read_meshing_source_execution_controls,
    recertify_restored_meshing_source,
    validate_meshing_accepted_epochs,
    write_meshing_source_closure,
)
from phydrax.lifecycle._meshing_field_records import (
    rebind_meshing_accepted_epoch_execution,
)
from phydrax.lifecycle._meshing_sources import (
    meshing_source_owner_reconstruction_scope,
    MeshingSourceClosureReceipt,
    native_family_audit_policy,
    require_same_source_evidence,
)
from phydrax.meshing import _curving as curving_owner
from phydrax.meshing._association import RESOLVED_STATUS
from phydrax.meshing._association_state import GeometryAssociation
from phydrax.meshing._audit import required_audit_policy
from phydrax.meshing._certification import MeshCertificationPreparedEvidence
from phydrax.meshing._measurements import NativeExecutionRecord
from phydrax.meshing.providers._native_publication import (
    bind_native_execution_result,
    NativePublicationExecution,
)
from phydrax.meshing.qualification import qualify_hybrid_curved_publication
from phydrax.meshing.qualification._native_lifecycle import (
    execute_native_consumer,
    execute_native_request,
)
from phydrax.typing import parse
from tools._meshing_cases import _strict_json_object, load_cad_fixture
from tools.meshing_qualification._controls import (
    _cad_qualification_requirements,
    archive_limit_values,
    capacity_meshing_limits,
    qualification_archive_limits,
    QualificationControls,
)
from tools.meshing_qualification._physics import (
    _compiled_solver_stages,
    _manufactured_value,
    _physical_error,
)
from tools.meshing_qualification._records import (
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
    StageLedger,
)
from tools.meshing_qualification.scenarios.bisection import (
    _bisection_p1_integral,
    _bisection_transfer,
)
from tools.meshing_qualification.scenarios.metric_adaptation import (
    _mesh_triangles,
    _triangle_areas,
)


def _cad_interchange_policies(
    capacity: int, contract: phx.SpatialCoordinateContract
) -> tuple[phx.interchange.CadImportPolicy, phx.interchange.CadExportPolicy]:
    """Bounded import/export resources scaled by the declared entity capacity."""
    byte_capacity = 4096 * capacity
    import_policy = phx.interchange.CadImportPolicy(
        contract,
        phx.interchange.ResourceLimits(
            max_bytes=byte_capacity,
            max_depth=64,
            max_nodes=64 * capacity,
            max_attributes=512 * capacity,
            max_losses=0,
        ),
        # Import consumes exact carriers and metadata, not a display mesh.
        # The resolution stage separately constructs/certifies the required
        # whole native surface and volume under the physical request.
        tessellation=phx.geometry.brep.BRepTessellationPolicy(realize=False),
        maximum_occurrences=capacity,
        pcurve_fit=None,
    )
    export_policy = phx.interchange.CadExportPolicy(
        maximum_bytes=byte_capacity, maximum_entities=64 * capacity
    )
    return import_policy, export_policy


def _cad_format_writer(format_name: str) -> tuple[Callable[..., Any], str]:
    match format_name:
        case "step":
            return phx.interchange.write_step, ".step"
        case "iges":
            return phx.interchange.write_iges, ".iges"
        case "brep":
            return phx.interchange.write_brep_text, ".brep"
        case _:
            raise ValueError(
                f"_cad_qualification_import.format_name must be 'step', 'iges' or 'brep'; got {format_name!r}."
            )


def _cad_qualification_import(
    directory: Path,
    format_name: str,
    capacity: int,
    stages: dict[str, float],
) -> tuple[phx.geometry.brep.BRepModel, phx.geometry.brep.BRepModel, dict[str, object]]:
    """Construct licensed analytic operands, write bytes, then bounded decode."""

    fixture = load_cad_fixture()
    raw_operands = fixture["operands"]
    if not isinstance(raw_operands, list):
        raise TypeError(
            f"_cad_qualification_import.operands must be a list; got {type(raw_operands).__name__!r}."
        )
    source_operands: list[tuple[str, tuple[float, ...], tuple[float, ...]]] = []
    for operand in raw_operands:
        if not isinstance(operand, dict):
            raise TypeError(
                f"_cad_qualification_import.operand must be a record; got {type(operand).__name__!r}."
            )
        name, lower, upper = (
            operand.get("name"),
            operand.get("lower"),
            operand.get("upper"),
        )
        if (
            not isinstance(name, str)
            or not isinstance(lower, list)
            or not isinstance(upper, list)
            or len(lower) != 3
            or len(upper) != 3
            or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in (*lower, *upper)
            )
        ):
            raise ValueError(
                f"_cad_qualification_import.operand must have a name and three-dimensional numeric bounds; got {operand!r}."
            )
        source_operands.append(
            (
                name,
                tuple(float(value) for value in lower),
                tuple(float(value) for value in upper),
            )
        )
    contract = phx.SpatialCoordinateContract.si()
    import_policy, export_policy = _cad_interchange_policies(capacity, contract)
    writer, suffix = _cad_format_writer(format_name)
    operands = []
    imports = []
    for name, lower, upper in source_operands:
        model, timing = _measured(
            lambda: phx.geometry.brep.brep_box(lower, upper, coordinate_contract=contract)
        )
        stages[f"source_construction_{name}"] = timing["wall_seconds"]
        path = directory / f"{name}{suffix}"
        exported, timing = _measured(lambda: writer(model, path, policy=export_policy))
        stages[f"format_encode_{name}"] = timing["wall_seconds"]
        if exported.approximations:
            raise qualification_failure(
                "_cad_qualification_import",
                ("The routine planar CAD fixture requires format export."),
                {"exported": exported},
            )
        imported, timing = _measured(
            lambda: phx.interchange.read_cad(
                path,
                import_policy,
                trusted_root=directory,
                source_length_unit=contract.length_unit
                if format_name == "brep"
                else None,
            )
        )
        stages[f"format_decode_{name}"] = timing["wall_seconds"]
        if imported.coverage.pcurves_fitted or imported.model.geometry is None:
            raise qualification_failure(
                "_cad_qualification_import",
                ("The native CAD consumer requires carriers and p-curves."),
                {"imported": imported},
            )
        operands.append(imported.model)
        imports.append(
            {
                "operand": name,
                "format": imported.format,
                "schema": imported.schema,
                "source_digest": imported.source_digest,
                "import_id": imported.result_id,
                "model_id": imported.model.model_id,
                "source_id": imported.model.source_id,
                "source_revision": imported.model.source_revision,
                "import_policy_id": import_policy.policy_id,
                "export_policy_id": export_policy.policy_id,
                "report_id": imported.report.report_id,
                "entity_counts": dict(imported.coverage.entity_counts),
                "pcurves_honored": imported.coverage.pcurves_honored,
                "pcurves_exact": imported.coverage.pcurves_exact,
                "pcurves_fitted": imported.coverage.pcurves_fitted,
                "source_length_unit_meters": imported.source_length_unit_meters,
                "provenance": dict(imported.provenance),
            }
        )
    reference = fixture["independent_reference"]
    if not isinstance(reference, dict):
        raise TypeError(
            f"_cad_qualification_import.reference must be a record; got {type(reference).__name__!r}."
        )
    return (
        operands[0],
        operands[1],
        {
            "operands": imports,
            "source_record": fixture,
            "source_record_id": fixture["record_id"],
            "provenance": fixture["provenance"],
            "license": fixture["license"],
            "reference_bounds": [[0.25, 0.0, 1.0], [0.75, 1.0, 1.0]],
            "reference_area": reference["selected_top_area"],
            "reference_solid_volume": reference["intersection_volume"],
            "independent_reference": reference,
        },
    )


def _cad_qualification_source(
    model: phx.geometry.brep.BRepModel,
    resolution: int,
    capacity: int,
    timeout: float,
    *,
    maximum_work_units: int | None = None,
    maximum_geometry_queries: int | None = None,
) -> tuple[
    phx.meshing.NativeSurfaceSource,
    phx.meshing.SurfaceMeshingSpec,
    phx.meshing.NativeMeshingOptions,
    tuple[int, ...],
]:
    """Select a planar face by topology; retain the entire CAD source model."""

    geometry = model.geometry
    if geometry is None:
        raise ValueError(
            f"_cad_qualification_source.geometry must contain B-Rep geometry; got {geometry!r}."
        )
    vertices = np.asarray(geometry.vertex_points, dtype=np.float64)
    selected = []
    for face, loops in enumerate(geometry.face_loops):
        members = np.unique(
            [
                vertex
                for loop in loops
                for coedge in loop
                for vertex in geometry.edge_vertices[geometry.coedge_edges[coedge]]
            ]
        )
        if np.all(np.abs(vertices[members, 2] - 1.0) <= 1.0e-12):
            selected.append(face)
    if not selected:
        raise qualification_failure(
            "_cad_qualification_source",
            ("The Boolean rectangle has no top planar face."),
            {"selected": selected},
        )
    domain = phx.geometry.MeshingDomain.from_brep(model)
    patches = tuple(
        local
        for local, source_index in enumerate(domain.source_indices[2])
        if source_index in selected
    )
    if not patches or domain.authority_id != model.model_id:
        raise qualification_failure(
            "_cad_qualification_source",
            ("The surface bridge did not retain the Boolean CAD source model."),
            {"domain": domain, "model": model, "patches": patches},
        )
    source = phx.meshing.NativeSurfaceSource(domain)
    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray(domain.scope_indices(2), dtype=np.int64)[list(patches)],
    )
    size = 0.5 / resolution
    limits = capacity_meshing_limits(
        capacity,
        timeout,
        maximum_work_units=maximum_work_units,
        maximum_geometry_queries=maximum_geometry_queries,
        cells=8,
        edges=8,
        faces=8,
        connectivity_entries=64,
        data_bytes=1024,
        work_units=32,
        cavity_cells=1,
        geometry_queries=64,
        scratch_bytes=4096,
    )
    request = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
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
        protected_features=(
            phx.meshing.ProtectedFeature(
                scope, phx.meshing.FeatureKind.SURFACE, maximum_deviation=1.0e-8
            ),
        ),
        quality_target=phx.meshing.MeshQualityTarget(
            minimum_angle=math.radians(10.0), hard=True
        ),
        limits=limits,
    )
    return (
        source,
        request,
        phx.meshing.NativeMeshingOptions("parametric_surface"),
        patches,
    )


def _cad_qualification_acceptance(
    result: phx.meshing.CellMeshingResult,
    source: phx.meshing.NativeSurfaceSource,
    patches: tuple[int, ...],
    request: phx.meshing.SurfaceMeshingSpec,
) -> dict[str, object]:
    """Analytic rectangle coverage plus strict source model/fidelity/control evidence."""

    certificate = result.certification
    if (
        certificate is None
        or not certificate.passed
        or not result.audit.passed
        or not result.compliance.passed
    ):
        raise qualification_failure(
            "_cad_qualification_acceptance",
            (
                "The CAD surface requires strict audit, source/global and control certification."
            ),
            {"certificate": certificate, "result": result},
        )
    quality_target = request.quality_target
    if quality_target is None:
        raise ValueError(
            f"_cad_qualification_acceptance.quality_target must be a frozen hard quality target; got {quality_target!r}."
        )
    mesh = result.mesh
    if mesh.topological_dimension != 2 or mesh.ambient_dimension != 3:
        raise qualification_failure(
            "_cad_qualification_acceptance",
            ("The CAD face consumer requires embedded triangular cells."),
            {"mesh": mesh},
        )
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    triangles = _mesh_triangles(mesh)
    area = float(np.sum(_triangle_areas(points, triangles)))
    plane_error = float(np.max(np.abs(points[:, 2] - 1.0)))
    outside = bool(
        np.any(points[:, 0] < 0.25 - 1.0e-12)
        or np.any(points[:, 0] > 0.75 + 1.0e-12)
        or np.any(points[:, 1] < -1.0e-12)
        or np.any(points[:, 1] > 1.0 + 1.0e-12)
    )
    expected_faces = {source.domain.entity_id(2, patch) for patch in patches}
    face_associations = [
        association
        for association in result.associations
        if association.target_entity_set_id == mesh.entity_set(2).entity_set_id
    ]
    if not result.associations or any(
        association.association_kind is not phx.meshing.GeometryAssociationKind.BREP
        or association.source_id != source.source_id
        or association.source_revision != source.source_revision
        for association in result.associations
    ):
        raise qualification_failure(
            "_cad_qualification_acceptance",
            ("The surface result lost its CAD source/revision association."),
            {"result": result, "source": source},
        )
    if (
        len(face_associations) != 1
        or set(face_associations[0].source_entities.entity_ids()) != expected_faces
    ):
        raise qualification_failure(
            "_cad_qualification_acceptance",
            ("The published face association lost the selected CAD entity scope."),
            {"expected_faces": expected_faces, "face_associations": face_associations},
        )
    quality = phx.meshing.summarize_cell_quality(phx.meshing.evaluate_cell_quality(mesh))
    passed = (
        abs(area - 0.5) <= 1.0e-10
        and plane_error <= 1.0e-10
        and not outside
        and quality.minimum_angle >= quality_target.minimum_angle - 1.0e-10
    )
    if not passed:
        raise qualification_failure(
            "_cad_qualification_acceptance",
            ("Independent planar CAD coverage, bounds or hard quality failed."),
            {"passed": passed},
        )
    return {
        "passed": passed,
        "analytic_area": 0.5,
        "mesh_area": area,
        "area_defect": area - 0.5,
        "plane_deviation": plane_error,
        "outside_analytic_bounds": outside,
        "selected_source_entities": sorted(expected_faces),
        "certification_id": certificate.report_id,
        "outcomes": {outcome.check: outcome.status for outcome in certificate.outcomes},
        "requested": dict(certificate.requested),
        "achieved": dict(certificate.achieved),
        "quality": _quality_summary(quality),
        "organization": {
            "patches": [patch.name for patch in result.patches],
            "labels": [label.name for label in result.labels],
        },
    }


def _cad_qualification_forcing(points: Any, context: Any) -> Any:
    del context
    # The planar surface has topological dimension two, not ambient dimension three.
    return jnp.full(points.shape[:-1], -4.0, dtype=points.dtype)


def _cad_qualification_solver(
    result: phx.meshing.CellMeshingResult,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[Any, Any, Any]:

    space = phx.discretization.FiniteElementPlan(
        result.mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
        coordinate_spec=result.geometry,
        budget=budget,
    ).prepare(budget=budget)
    constraint = phx.discretization.dirichlet_constraint(space, "u")
    if constraint.free_dofs.size == 0:
        raise qualification_failure(
            "_cad_qualification_solver",
            ("The CAD PDE fixture needs interior scalar H1 degrees of freedom."),
            {"constraint": constraint},
        )
    form = phx.equations.FiniteElementForm(
        "native-cad-planar-laplace-beltrami",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    _cad_qualification_forcing,
                    coefficient_id="planar-surface-quadratic-minus-four",
                ),
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form, space, constraint=constraint, dirichlet_values=_manufactured_value,
        budget=budget,
    )
    return space, problem, constraint.free_dofs


def _cad_qualification_epochs(
    result: phx.meshing.CellMeshingResult,
    model: phx.geometry.brep.BRepModel,
    source: phx.meshing.NativeSurfaceSource,
    patches: tuple[int, ...],
    request: phx.meshing.SurfaceMeshingSpec,
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
    deadline: Deadline,
) -> None:
    """Continue the PDE from its accepted scalar state through native lineage."""

    _ledger = StageLedger(record, deadline=deadline)
    stages = record["stages_seconds"]
    previous_values = None
    hierarchy = None
    projection, timing = _ledger.run(
        record.get("active_phase", "prepare_brep_projection"),
        lambda: _measured(lambda: phx.geometry.brep.prepare_brep_projection(model)),
    )
    stages["cad_projection_preparation"] = timing["wall_seconds"]
    association_transfer = phx.meshing.BRepAssociationTransfer(projection)
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
        limits=request.limits,
        compatibility=phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
        association_transfer=association_transfer,
    )
    for epoch in range(args.adaptation_rounds + 1):
        consumed, timing = _ledger.run(
            f"solver_prepare_epoch_{epoch}",
            lambda: _measured(
                lambda: execute_native_consumer(
                    result, request.limits,
                    lambda budget: _cad_qualification_solver(result, budget=budget),
                    failure_factory=qualification_failure,
                    observe_managed_admission=True,
                )
            ),
        )
        result, solver = consumed.result, consumed.value
        space, problem, free_dofs = solver
        stages[record["active_phase"]] = timing["wall_seconds"]
        seed = (
            jnp.zeros((free_dofs.size,), dtype=result.mesh.coordinates.dtype)
            if previous_values is None
            else jnp.asarray(previous_values)[free_dofs]
        )

        values, compiler, timings = _ledger.run(
            f"solver_execution_epoch_{epoch}",
            lambda: _compiled_solver_stages(problem, seed, args.repeats),
        )
        stages.update(
            {f"{name}_epoch_{epoch}": elapsed for name, elapsed in timings.items()}
        )
        error = _physical_error(result.mesh, values)
        if error <= args.target_error and record["time_to_target_error_seconds"] is None:
            record["time_to_target_error_seconds"] = perf_counter() - started
        row = {
            "epoch": epoch,
            "result": _result_identity(result),
            "l2_error": error,
            "dofs": space.dof_maps[0].global_dof_count,
            "compiler": compiler,
            "retained_bytes": model_array_bytes((result, space, problem, hierarchy)),
            "transferred_state_consumed_as_seed": previous_values is not None,
            "transferred_state_l2_error": (
                None
                if previous_values is None
                else _physical_error(result.mesh, previous_values)
            ),
        }
        record["epochs"].append(row)
        if epoch == args.adaptation_rounds:
            break

        marked = _ledger.run(
            f"adaptation_prepare_epoch_{epoch}",
            lambda: phx.meshing.MarkedMeshAdaptation(
                np.asarray(result.mesh.blocks[0].global_ids, dtype=np.int64),
                np.empty((0,), dtype=np.int64),
                hierarchy=hierarchy,
            ),
        )
        prepared, timing = _ledger.run(
            record.get("active_phase", "prepare_mesh_adaptation"),
            lambda: _measured(
                lambda: phx.meshing.prepare_mesh_adaptation(
                    result, marked, policy=policy, observe_managed_admission=True
                )
            ),
        )
        stages[record["active_phase"]] = timing["wall_seconds"]

        adaptation, timing = _ledger.run(
            f"adaptation_execute_epoch_{epoch}",
            lambda: _measured(
                lambda: phx.meshing.execute_mesh_adaptation(
                    prepared, observe_managed_admission=True
                )
            ),
        )
        stages[record["active_phase"]] = timing["wall_seconds"]
        state_transfer = adaptation.transfer
        if (
            adaptation.status is not phx.meshing.MeshAdaptationStatus.COMPLETE
            or state_transfer is None
        ):
            raise qualification_failure(
                "_cad_qualification_epochs",
                ("CAD-preserving bisection omitted scalar state transfer roles."),
                {"adaptation": adaptation, "state_transfer": state_transfer},
            )

        previous_values, timing = _ledger.run(
            f"field_transfer_epoch_{epoch}",
            lambda: _measured(
                lambda: jax.block_until_ready(state_transfer.apply(values))
            ),
        )
        stages[record["active_phase"]] = timing["wall_seconds"]
        row["transfer"] = _bisection_transfer(adaptation)
        before = _bisection_p1_integral(result.mesh, values)
        after = _bisection_p1_integral(adaptation.target.mesh, previous_values)
        row["state_inventory"] = {
            "source": before,
            "target": after,
            "defect": after - before,
        }
        if abs(after - before) > 1.0e-10 * max(1.0, abs(before)):
            raise qualification_failure(
                "_cad_qualification_epochs",
                ("The CAD scalar solver-state transfer lost P1 inventory."),
                {"after": after, "before": before},
            )

        target_acceptance = _ledger.run(
            f"independent_acceptance_epoch_{epoch + 1}",
            lambda: _cad_qualification_acceptance(
                adaptation.target, source, patches, request
            ),
        )
        if (
            target_acceptance["organization"]
            != record["independent_acceptance"]["organization"]
        ):
            raise qualification_failure(
                "_cad_qualification_epochs",
                ("CAD adaptation lost patch/curve organization."),
                {"record": record, "target_acceptance": target_acceptance},
            )
        row["target_acceptance"] = target_acceptance
        result, hierarchy = adaptation.target, adaptation.hierarchy
    record["physical_error"] = record["epochs"][-1]["l2_error"]
    record["target_error_reached"] = record["physical_error"] <= args.target_error


@final
class _CadPublication(phx.StrictModule):
    source: phx.meshing.NativeSurfaceSource
    request: phx.meshing.SurfaceMeshingSpec
    patches: tuple[int, ...]
    result: phx.meshing.CellMeshingResult


def _cad_qualification_boolean_query(
    first: phx.geometry.brep.BRepModel,
    second: phx.geometry.brep.BRepModel,
    args: QualificationControls,
    record: dict[str, Any],
    deadline: Deadline,
) -> phx.geometry.brep.BRepModel:
    """Intersect decoded CAD operands and audit the native query against analytic volume."""

    _ledger = StageLedger(record, deadline=deadline)
    stages = record["stages_seconds"]
    boolean, timing = _ledger.run(
        "native_boolean",
        lambda: _measured(
            lambda: phx.geometry.brep.boolean_brep(
                first,
                second,
                "intersection",
                policy=phx.geometry.brep.BRepBooleanPolicy(
                    maximum_cells=args.capacity, maximum_faces=args.capacity
                ),
            )
        ),
    )
    stages[record["active_phase"]] = timing["wall_seconds"]
    model = boolean.model
    record["boolean"] = {
        "operation": boolean.operation,
        "certificate_id": boolean.certificate_id,
        "association_graph_id": boolean.association_graph.graph_id,
        "deleted_source_ids": list(boolean.deleted_source_ids),
        "model_id": model.model_id,
    }

    query, timing = _ledger.run(
        "native_query_preparation",
        lambda: _measured(lambda: phx.geometry.brep.prepare_brep_query(model)),
    )
    stages[record["active_phase"]] = timing["wall_seconds"]
    volume = float(np.sum(np.asarray(query.measures.solid_volumes)))
    if abs(volume - 0.5) > 1.0e-10:
        raise qualification_failure(
            "_cad_qualification_lifecycle",
            ("The native Boolean query disagrees with analytic intersection volume."),
            {"volume": volume},
        )
    record["query"] = {
        "query_id": query.query_id,
        "source_revision": query.source_revision,
        "analytic_solid_volume": 0.5,
        "solid_volume": volume,
        "face_areas": np.asarray(query.measures.face_areas).tolist(),
        "face_area_error_estimates": np.asarray(
            query.measures.face_area_error_estimates
        ).tolist(),
        "solid_volume_error_estimates": np.asarray(
            query.measures.solid_volume_error_estimates
        ).tolist(),
    }
    return model


def _cad_qualification_publish_source(
    model: phx.geometry.brep.BRepModel,
    args: QualificationControls,
    record: dict[str, Any],
    record_phase: Callable[[phx.meshing.NativeMeshingPhaseMeasurement], None],
    deadline: Deadline,
) -> _CadPublication:
    """Bind the full CAD source model, publish its selected face, and audit source acceptance."""

    _ledger = StageLedger(record, deadline=deadline)
    stages = record["stages_seconds"]
    prepared, timing = _ledger.run(
        "authoritative_source_request_preparation",
        lambda: _measured(
            lambda: _cad_qualification_source(
                model, args.resolution, args.capacity, args.timeout,
                maximum_work_units=args.maximum_work_units,
                maximum_geometry_queries=args.maximum_geometry_queries,
            )
        ),
    )
    source, request, options, patches = prepared
    stages[record["active_phase"]] = timing["wall_seconds"]
    record["source"] = {
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "binding_id": source.binding_id,
        "authority_id": source.domain.authority_id,
        "domain_id": source.domain.domain_id,
        "selected_entities": [source.domain.entity_id(2, patch) for patch in patches],
        "accuracy": source.domain.accuracy,
        "whole_domain_retained": True,
    }
    quality_target = request.quality_target
    if quality_target is None:
        raise ValueError(
            f"_cad_qualification_publish_source.quality_target must be a frozen hard quality target; got {quality_target!r}."
        )
    record["request"] = {
        "specification_id": request.specification_id,
        "limits_id": request.limits.limits_id,
        "scope_id": request.scope.scope_id,
        "quality_target_id": quality_target.target_id,
    }

    provider = _ledger.run(
        "geometry_constraint_preparation",
        lambda: phx.meshing.NativeMeshingProvider(options),
    )
    plan, timing = _ledger.run(
        record.get("active_phase", "plan"),
        lambda: _measured(
            lambda: provider.plan(
                source,
                request,
                coordinate_contract=model.coordinate_contract,
                record_phase=record_phase,
            )
        ),
    )
    stages[record["active_phase"]] = timing["wall_seconds"]
    record["prepared_id"] = plan.plan_id

    result, timing = _ledger.run(
        "generation_publication_certification",
        lambda: _measured(lambda: plan.execute(record_phase=record_phase)),
    )
    stages[record["active_phase"]] = timing["wall_seconds"]
    record["generation"] = _result_identity(result)

    acceptance, timing = _ledger.run(
        "independent_source_acceptance",
        lambda: _measured(
            lambda: _cad_qualification_acceptance(result, source, patches, request)
        ),
    )
    stages[record["active_phase"]] = timing["wall_seconds"]
    record["independent_acceptance"] = acceptance
    return _CadPublication(source=source, request=request, patches=patches, result=result)


def _cad_qualification_completion(
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
) -> None:
    """Finalize process resource evidence and the CAD wall-budget disposition."""

    record["end_to_end_seconds"] = perf_counter() - started
    record["peak_process_resident_bytes"] = peak_resident_bytes()
    record["process_peak_scope"] = PEAK_RESIDENT_SCOPE
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
    if record["status"] != "passed":
        record["failure_phase"] = record["active_phase"]
    if record["end_to_end_seconds"] > args.timeout:
        record["status"] = "timeout"
    if record["status"] != "passed":
        record["case_contract_completion"] = "incomplete"
        record["independent_scientific_certification"] = "unassessed"
    del record["active_phase"]


def _cad_qualification_lifecycle(args: QualificationControls) -> dict[str, object]:
    """Bounded native format → Boolean/query → strict CAD face → H1 state loop."""

    requirements = _cad_qualification_requirements()
    if (
        args.resolution < 1
        or args.capacity < 1
        or args.repeats < 1
        or args.adaptation_rounds < 1
    ):
        raise ValueError(
            f"_cad_qualification_lifecycle.bounds must have positive resolution, capacity, repeats and adaptation rounds; got {(args.resolution, args.capacity, args.repeats, args.adaptation_rounds)!r}."
        )
    if (
        not np.isfinite(args.timeout)
        or args.timeout <= 0.0
        or not np.isfinite(args.target_error)
        or args.target_error <= 0.0
    ):
        raise ValueError(
            f"_cad_qualification_lifecycle.bounds must have finite positive timeout and target error; got {(args.timeout, args.target_error)!r}."
        )
    started = perf_counter()
    deadline = Deadline(started + args.timeout)
    record: dict[str, Any] = {
        "scenario": "native-cad-lifecycle",
        "status": "failed",
        "runtime": _runtime_identity(),
        "requirements": requirements,
        "format": args.cad_format,
        "stages_seconds": {},
        "native_phases": [],
        "epochs": [],
        "targets": {"l2_error": args.target_error, "wall_seconds": args.timeout},
        "control_capacities": {
            "resolution": args.resolution,
            "entity_capacity": args.capacity,
            "adaptation_rounds": args.adaptation_rounds,
        },
        "time_to_target_error_seconds": None,
        "positive_prerequisite": None,
        "mandatory_workflow_completion": "incomplete",
        "case_contract_completion": "incomplete",
        "independent_scientific_certification": "unassessed",
        "full_curved_multiformat_gate": "incomplete",
        "release_evidence": "unassessed",
        "leadership_evidence": "unassessed",
        "nonclaims": requirements["nonclaims"],
        "optional_external_baseline": "not-requested",
        "pde": "-Delta_S |x|^2 = -4 on an exact planar CAD rectangle",
        "active_phase": "source_format_preparation",
    }
    _ledger = StageLedger(record, deadline=deadline)

    def record_phase(measurement: phx.meshing.NativeMeshingPhaseMeasurement) -> None:
        record["native_phases"].append(
            {
                "phase": measurement.phase,
                "elapsed_seconds": measurement.elapsed_seconds,
                "work_units": measurement.work_units,
                "invocations": measurement.invocations,
            }
        )

    stages = record["stages_seconds"]
    try:
        with TemporaryDirectory(prefix="phydrax-native-cad-") as temporary:
            first, second, imports = _cad_qualification_import(
                Path(temporary), args.cad_format, args.capacity, stages
            )
            record["imports"] = imports

            model = _cad_qualification_boolean_query(
                first, second, args, record, deadline
            )
            publication = _cad_qualification_publish_source(
                model, args, record, record_phase, deadline
            )
            source, request, patches, result = (
                publication.source,
                publication.request,
                publication.patches,
                publication.result,
            )
            _cad_qualification_epochs(
                result, model, source, patches, request, args, record, started, deadline
            )
            record["status"] = "passed" if record["target_error_reached"] else "failed"
            if record["target_error_reached"]:
                # Source and every adapted epoch passed independent acceptance,
                # inventory and organization checks before reaching here.
                record["case_contract_completion"] = "complete"
                record["independent_scientific_certification"] = "passed"
    except phx.meshing.MeshingFailure as failure:
        record_failure(record, failure, record["active_phase"])
        record.update(
            {
                "positive_prerequisite": {
                    "phase": record["active_phase"],
                    "detail": str(failure),
                    "category": failure.category.value,
                    "safe_refusal_is_positive": False,
                }
            }
        )
    except (RuntimeError, ValueError, TypeError, TimeoutError) as failure:
        record_failure(record, failure, record["active_phase"])
        record.update(
            {
                "positive_prerequisite": {
                    "phase": record["active_phase"],
                    "detail": str(failure),
                    "source_organization_stripped": False,
                    "safe_refusal_is_positive": False,
                }
            }
        )
    finally:
        _cad_qualification_completion(args, record, started)
    return record


# -- Curved, external and combined native CAD cases --------------------------------

CAD_CASES = (
    "planar-box",
    "quadric-routine",
    "quadric-larger",
    "external-routine",
    "external-combined",
)
_CAD_DATA = Path(__file__).resolve().parents[3] / "tests/data/cad"
CAD_QUADRIC_FIXTURE_PATH = _CAD_DATA / "native_authored_quadrics.json"
CAD_EXTERNAL_FIXTURE_PATH = _CAD_DATA / "nist_ftc_11_external.json"
_CAD_LOOP_FIELDS = {"axis_radius", "transverse_radius", "transverse_center_z", "side"}


def _cad_fixture_body(path: Path, kind: str, fields: set[str]) -> dict[str, Any]:
    raw = path.read_bytes()
    value = _strict_json_object(raw, f"CAD fixture {path.name}")
    body = {name: field for name, field in value.items() if name != "record_id"}
    if (
        set(body) != fields
        or body.get("kind") != kind
        or value.get("record_id") != canonical_fingerprint(body)
    ):
        raise ValueError(
            f"_cad_fixture_body.value must match the {kind!r} fields and record fingerprint; got {sorted(value)!r}."
        )
    return {**value, "file_sha256": hashlib.sha256(raw).hexdigest()}


def _cad_loops(value: Any, name: str) -> list[dict[str, int | float]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name}.loops must be a nonempty list; got {value!r}.")
    for loop in value:
        if (
            not isinstance(loop, dict)
            or set(loop) != _CAD_LOOP_FIELDS
            or loop["side"] not in (-1, 1)
            or any(
                isinstance(loop[key], bool)
                or not isinstance(loop[key], (int, float))
                or not math.isfinite(loop[key])
                for key in _CAD_LOOP_FIELDS
            )
            or not 0.0 < loop["transverse_radius"] < loop["axis_radius"]
        ):
            raise ValueError(
                f"{name}.loop must declare unequal perpendicular cylinder radii and a side; got {loop!r}."
            )
    return value


def load_cad_quadric_fixture(path: Path = CAD_QUADRIC_FIXTURE_PATH) -> dict[str, Any]:
    """Licensed authored quadric operands and their independent analytic oracle."""
    value = _cad_fixture_body(
        path,
        "native-authored-cad-quadrics",
        {
            "kind",
            "title",
            "author",
            "license",
            "provenance",
            "coordinate_unit",
            "operands",
            "operation",
            "independent_reference",
        },
    )
    operands = value["operands"]
    reference = value["independent_reference"]
    if (
        value["license"] != "LicenseRef-PHYDRA-Proprietary"
        or value["coordinate_unit"] != "meter"
        or value["operation"] != "intersection"
        or not isinstance(operands, list)
        or len(operands) != 2
        or any(
            not isinstance(operand, dict)
            or set(operand) != {"name", "radius", "height", "base_center", "axis"}
            or not isinstance(operand["name"], str)
            or not 0.0 < float(operand["radius"]) < float(operand["height"])
            or len(operand["base_center"]) != 3
            or len(operand["axis"]) != 3
            for operand in operands
        )
        or not isinstance(reference, dict)
        or set(reference)
        != {"method", "volume", "loops", "solids", "manufactured_length"}
    ):
        raise ValueError(
            f"load_cad_quadric_fixture.body must declare two authored cylinders, meter units and an analytic reference; got {value!r}."
        )
    reference["loops"] = _cad_loops(reference["loops"], "load_cad_quadric_fixture")
    return value


def load_cad_external_fixture(path: Path = CAD_EXTERNAL_FIXTURE_PATH) -> dict[str, Any]:
    """External NIST STEP provenance, unmodified bytes and its independent oracle."""
    value = _cad_fixture_body(
        path,
        "external-cad-step-source",
        {
            "kind",
            "title",
            "author",
            "license",
            "provenance",
            "source_file",
            "source_sha256",
            "source_length_unit_meters",
            "independent_reference",
            "authored_hole",
            "manufactured_length",
        },
    )
    source = path.parent / str(value["source_file"])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    provenance = value["provenance"]
    hole = value["authored_hole"]
    if (
        digest != value["source_sha256"]
        or source.parent != path.parent
        or not isinstance(provenance, dict)
        or not {"archive_url", "archive_sha256", "archive_member"} <= set(provenance)
        or value["source_length_unit_meters"] != 0.001
        or not isinstance(hole, dict)
        or hole.get("operation") != "difference"
    ):
        raise ValueError(
            f"load_cad_external_fixture.source must be the unmodified declared external file with provenance; got digest {digest!r}."
        )
    hole["loops"] = _cad_loops(hole["loops"], "load_cad_external_fixture")
    return {**value, "source_path": str(source)}


def _gauss_interval(count: int, lower: float, upper: float) -> tuple[Any, Any]:
    nodes, weights = np.polynomial.legendre.leggauss(count)
    half = 0.5 * (upper - lower)
    return lower + half * (nodes + 1.0), half * weights


def _cross_cylinder_volume(axis_radius: float, transverse_radius: float) -> float:
    """∫ 4√(R²−y²)√(r²−y²) dy with y = r sin θ, smooth for every r < R."""
    theta, weights = _gauss_interval(256, -0.5 * math.pi, 0.5 * math.pi)
    r, big = transverse_radius, axis_radius
    integrand = 4.0 * r * r * np.cos(theta) ** 2 * np.sqrt(big**2 - (r * np.sin(theta)) ** 2)
    return float(np.sum(weights * integrand))


def _cross_loop_length(axis_radius: float, transverse_radius: float) -> float:
    """Length of one branch (±√(R²−r²sin²θ), r sin θ, r cos θ)."""
    theta, weights = _gauss_interval(512, 0.0, 2.0 * math.pi)
    r, big = transverse_radius, axis_radius
    sine, cosine = np.sin(theta), np.cos(theta)
    speed = np.sqrt(r * r + r**4 * sine**2 * cosine**2 / (big**2 - r * r * sine**2))
    return float(np.sum(weights * speed))


def _washer_volume(reference: dict[str, Any]) -> float:
    """Pappus volume of the manually read FTC-11 revolution profile (mm³)."""
    bottom = float(reference["bottom_z"])
    inner, outer = float(reference["inner_radius"]), float(reference["outer_radius"])
    flat_inner = float(reference["flat_inner_radius"])
    flat_outer = float(reference["flat_outer_radius"])
    flat = float(reference["flat_z"])
    # Base slab below z = 0 and the flat top between the beads.
    volume = math.pi * (outer**2 - inner**2) * (-bottom)
    volume += math.pi * (flat_outer**2 - flat_inner**2) * flat
    for bead, start, end in (
        (reference["inner_bead"], inner, flat_inner),
        (reference["outer_bead"], flat_outer, outer),
    ):
        center, minor = float(bead["center_radius"]), float(bead["minor_radius"])
        first = math.acos(max(-1.0, min(1.0, (start - center) / minor)))
        last = math.acos(max(-1.0, min(1.0, (end - center) / minor)))
        angle, weights = _gauss_interval(128, min(first, last), max(first, last))
        rho = center + minor * np.cos(angle)
        height = minor * np.sin(angle)
        # dρ = −a sin φ dφ; the arc lies above z = 0 on both beads.
        volume += 2.0 * math.pi * float(
            np.sum(weights * rho * height * minor * np.sin(angle))
        )
    return volume


def _radial_hole_volume(
    reference: dict[str, Any], hole: dict[str, Any]
) -> float:
    """Material removed by the authored radial cylinder through both walls (mm³)."""
    inner, outer = float(reference["inner_radius"]), float(reference["outer_radius"])
    radius = float(hole["radius"])
    rho, rho_weights = _gauss_interval(64, 0.0, radius)
    phi, phi_weights = _gauss_interval(128, 0.0, 2.0 * math.pi)
    y = rho[:, None] * np.cos(phi)[None, :]
    chord = 2.0 * (np.sqrt(outer**2 - y**2) - np.sqrt(inner**2 - y**2))
    return float(np.sum(rho_weights[:, None] * phi_weights[None, :] * chord * rho[:, None]))


@final
class _CadCase(phx.StrictModule):
    """One curved/external CAD case after bounded import and native Boolean."""

    model: phx.geometry.brep.BRepModel
    loops: tuple[tuple[float, float, float, float], ...]
    reference_volume: float = eqx.field(static=True)
    manufactured_length: float = eqx.field(static=True)
    size: float = eqx.field(static=True)
    deviation: float = eqx.field(static=True)
    minimum_radius: float = eqx.field(static=True)


def _cad_case_import(
    path: Path,
    format_name: str,
    import_policy: phx.interchange.CadImportPolicy,
    directory: Path,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
    source_length_unit: Any = None,
) -> tuple[phx.geometry.brep.BRepModel, dict[str, object]]:
    imported = phx.interchange.read_cad(
        path,
        import_policy,
        trusted_root=directory,
        source_length_unit=source_length_unit,
        coordinate_budget=coordinate_budget,
    )
    if imported.coverage.pcurves_fitted or imported.model.geometry is None:
        raise qualification_failure(
            "_cad_case_import",
            "The native CAD consumer requires exact carriers and unfitted p-curves.",
            {"imported": imported},
        )
    return imported.model, {
        "format": imported.format,
        "schema": imported.schema,
        "source_digest": imported.source_digest,
        "import_id": imported.result_id,
        "model_id": imported.model.model_id,
        "source_id": imported.model.source_id,
        "source_revision": imported.model.source_revision,
        "import_policy_id": import_policy.policy_id,
        "import_mode": "exact-model-only"
        if import_policy.tessellation is not None and not import_policy.tessellation.realize
        else "exact-model-with-derived-tessellation",
        "report_id": imported.report.report_id,
        "report_status": imported.report.status.name,
        "source_profile_qualifiers": (
            {} if imported.report.source_profile is None
            else dict(imported.report.source_profile.qualifiers)
        ),
        "entity_counts": dict(imported.coverage.entity_counts),
        "pcurves_honored": imported.coverage.pcurves_honored,
        "pcurves_exact": imported.coverage.pcurves_exact,
        "pcurves_fitted": imported.coverage.pcurves_fitted,
        "source_length_unit_meters": imported.source_length_unit_meters,
        "provenance": dict(imported.provenance),
    }


def _cad_case_roundtrip(
    model: phx.geometry.brep.BRepModel,
    name: str,
    format_name: str,
    capacity: int,
    directory: Path,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
) -> tuple[phx.geometry.brep.BRepModel, dict[str, object]]:
    """Exact export in the requested format, then bounded native decode."""
    import_policy, export_policy = _cad_interchange_policies(
        capacity, model.coordinate_contract
    )
    writer, suffix = _cad_format_writer(format_name)
    path = directory / f"{name}{suffix}"
    exported = writer(
        model, path, policy=export_policy, coordinate_budget=coordinate_budget
    )
    if exported.approximations:
        raise qualification_failure(
            "_cad_case_roundtrip",
            "An analytic operand must export exactly without approximation.",
            {"exported": exported},
        )
    imported, record = _cad_case_import(
        path,
        format_name,
        import_policy,
        directory,
        coordinate_budget=coordinate_budget,
        source_length_unit=model.coordinate_contract.length_unit
        if format_name == "brep"
        else None,
    )
    return imported, {
        **record,
        "operand": name,
        "export_policy_id": export_policy.policy_id,
        "exported_entity_counts": dict(exported.entity_counts),
    }


def _cad_native_limits(args: QualificationControls) -> phx.meshing.MeshingLimits:
    """The existing CAD request allowance, shared by host and native owners."""
    return capacity_meshing_limits(
        args.capacity,
        args.timeout,
        maximum_work_units=args.maximum_work_units,
        maximum_geometry_queries=args.maximum_geometry_queries,
        vertices=64,
        cells=256,
        edges=512,
        faces=512,
        connectivity_entries=4096,
        data_bytes=65536,
        work_units=65536,
        cavity_cells=64,
        geometry_queries=65536,
        scratch_bytes=262144,
    )


def _cad_boolean(
    first: phx.geometry.brep.BRepModel,
    second: phx.geometry.brep.BRepModel,
    operation: phx.geometry.brep.BRepBooleanOperation,
    args: QualificationControls,
    *,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> tuple[phx.geometry.brep.BRepModel, dict[str, object]]:
    def execute(budget: CoordinateEnclosureBudget) -> phx.geometry.brep.BRepBooleanResult:
        budget.ticket.checkpoint()
        result = phx.geometry.brep.boolean_brep(
            first,
            second,
            operation,
            policy=phx.geometry.brep.BRepBooleanPolicy(
                maximum_cells=args.capacity,
                maximum_faces=args.capacity,
                tessellation=phx.geometry.brep.BRepTessellationPolicy(realize=False),
            ),
            budget=budget,
        )
        budget.ticket.checkpoint()
        return result

    execution: NativeExecutionRecord | None = None
    if coordinate_budget is not None:
        boolean = execute(coordinate_budget)
    else:
        limits = _cad_native_limits(args)
        # A standalone invocation borrows the active stage deadline. A source
        # stage with an existing ticket takes the branch above without a reset.
        with NativeExecutionScope(limits.native_limits()) as scope:
            with scope.batch(
                limits.maximum_work_units,
                queries_upper=limits.maximum_geometry_queries,
                bytes_upper=limits.maximum_scratch_bytes,
                cavity_upper=limits.maximum_cavity_cells,
            ) as ticket:
                budget = CoordinateEnclosureBudget(
                    ticket,
                    maximum_work_units=limits.maximum_work_units,
                    maximum_memory_bytes=limits.maximum_scratch_bytes,
                )
                with ticket.native_target(), budget.activate():
                    boolean = execute(budget)
        execution = NativeExecutionRecord(
            scope.completed_evidence(),
            owner_id=canonical_fingerprint(
                {
                    "kind": "native-cad-boolean-request",
                    "sources": (first.model_id, second.model_id),
                    "operation": operation,
                    "capacity": args.capacity,
                    "timeout": args.timeout,
                }
            ),
        )
    coverage = boolean.association_graph.transaction.coverage
    if boolean.empty or not (coverage.source_exhaustive and coverage.target_exhaustive):
        raise qualification_failure(
            "_cad_boolean",
            "The native Boolean must be nonempty with exhaustive lineage.",
            {"boolean": boolean},
        )
    record: dict[str, object] = {
        "operation": boolean.operation,
        "certificate_id": boolean.certificate_id,
        "association_graph_id": boolean.association_graph.graph_id,
        "deleted_source_ids": list(boolean.deleted_source_ids),
        "model_id": boolean.model.model_id,
        "solids": boolean.model.topology.num_solids,
        "execution_scope": (
            "shared-source-stage" if coordinate_budget is not None else "boolean-request"
        ),
    }
    if execution is not None:
        record["execution"] = execution.to_record()
    return boolean.model, record


def _cad_authored_case(
    args: QualificationControls, directory: Path, record: dict[str, Any], scale: int,
    *, coordinate_budget: CoordinateEnclosureBudget,
) -> _CadCase:
    fixture = load_cad_quadric_fixture()
    contract = phx.SpatialCoordinateContract.si()
    exact = phx.geometry.brep.BRepTessellationPolicy(realize=False)
    operands, imports = [], []
    for operand in fixture["operands"]:
        model = phx.geometry.brep.brep_cylinder(
            float(operand["radius"]),
            float(operand["height"]),
            base_center=tuple(float(value) for value in operand["base_center"]),
            axis=tuple(float(value) for value in operand["axis"]),
            coordinate_contract=contract,
            tessellation=exact,
            source_id=f"authored-{operand['name']}",
            budget=coordinate_budget,
        )
        imported, entry = _cad_case_roundtrip(
            model, str(operand["name"]), args.cad_format, args.capacity, directory,
            coordinate_budget=coordinate_budget,
        )
        operands.append(imported)
        imports.append(entry)
    model, boolean = _cad_boolean(
        operands[0],
        operands[1],
        parse(
            str(fixture["operation"]),
            phx.geometry.brep.BRepBooleanOperation,
            "cad_fixture.operation",
        ),
        args,
        coordinate_budget=coordinate_budget,
    )
    reference = fixture["independent_reference"]
    loops = tuple(
        (
            loop["axis_radius"],
            loop["transverse_radius"],
            loop["transverse_center_z"],
            loop["side"],
        )
        for loop in reference["loops"]
    )
    radii = [float(operand["radius"]) for operand in fixture["operands"]]
    size = 0.5 * max(radii) / (args.resolution * scale)
    record["source_fixture"] = {
        "kind": fixture["kind"],
        "record_id": fixture["record_id"],
        "file_sha256": fixture["file_sha256"],
        "license": fixture["license"],
        "provenance": fixture["provenance"],
        "independent_reference_method": reference["method"],
        "external_corpus": False,
    }
    record["imports"] = imports
    record["boolean"] = boolean
    return _CadCase(
        model=model,
        loops=loops,
        reference_volume=_cross_cylinder_volume(radii[0], radii[1]),
        manufactured_length=float(reference["manufactured_length"]),
        size=size,
        deviation=size * size / (4.0 * min(radii)),
        minimum_radius=min(radii),
    )


def _cad_external_case(
    args: QualificationControls,
    directory: Path,
    record: dict[str, Any],
    scale: int,
    *,
    combined: bool,
    coordinate_budget: CoordinateEnclosureBudget,
) -> _CadCase:
    fixture = load_cad_external_fixture()
    contract = phx.SpatialCoordinateContract.si()
    import_policy, _ = _cad_interchange_policies(args.capacity, contract)
    source = Path(fixture["source_path"])
    external, primary = _cad_case_import(
        source, "step", import_policy, source.parent,
        coordinate_budget=coordinate_budget,
    )
    unit = float(fixture["source_length_unit_meters"])
    if primary["source_length_unit_meters"] != unit:
        raise qualification_failure(
            "_cad_external_case",
            "The external STEP length unit differs from its declared provenance.",
            {"primary": primary},
        )
    # The external part is also written and decoded in the requested format.
    model, reencoded = _cad_case_roundtrip(
        external, "external", args.cad_format, args.capacity, directory,
        coordinate_budget=coordinate_budget,
    )
    reference = fixture["independent_reference"]
    volume = _washer_volume(reference) * unit**3
    loops: tuple[tuple[float, float, float, float], ...] = ()
    minimum_radius = unit * float(reference["inner_bead"]["minor_radius"])
    boolean = None
    if combined:
        hole = fixture["authored_hole"]
        half = float(hole["half_length"]) * unit
        cutter = phx.geometry.brep.brep_cylinder(
            float(hole["radius"]) * unit,
            2.0 * half,
            base_center=(-half, 0.0, float(hole["center_z"]) * unit),
            axis=tuple(float(value) for value in hole["axis"]),
            coordinate_contract=contract,
            tessellation=phx.geometry.brep.BRepTessellationPolicy(realize=False),
            source_id="authored-radial-hole",
            budget=coordinate_budget,
        )
        model, boolean = _cad_boolean(
            model, cutter, "difference", args, coordinate_budget=coordinate_budget
        )
        volume -= _radial_hole_volume(reference, hole) * unit**3
        loops = tuple(
            (
                loop["axis_radius"] * unit,
                loop["transverse_radius"] * unit,
                loop["transverse_center_z"] * unit,
                loop["side"],
            )
            for loop in hole["loops"]
        )
        minimum_radius = float(hole["radius"]) * unit
    size = 6.0e-3 / (args.resolution * scale)
    record["source_fixture"] = {
        "kind": fixture["kind"],
        "record_id": fixture["record_id"],
        "file_sha256": fixture["file_sha256"],
        "source_sha256": fixture["source_sha256"],
        "license": fixture["license"],
        "provenance": fixture["provenance"],
        "independent_reference_method": reference["method"],
        "external_corpus": True,
        "authored_operand": combined,
    }
    record["imports"] = [{**primary, "operand": "nist-ftc-11"}, reencoded]
    record["boolean"] = boolean
    return _CadCase(
        model=model,
        loops=loops,
        reference_volume=volume,
        manufactured_length=float(fixture["manufactured_length"]) * unit,
        size=size,
        deviation=4.0e-5 / scale**2,
        minimum_radius=minimum_radius,
    )


def _cad_intersection_evidence(
    model: phx.geometry.brep.BRepModel,
    loops: tuple[tuple[float, float, float, float], ...],
    *,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> dict[str, object]:
    """Independent branch/loop, 3D and both p-curve checks against the quadrics."""
    geometry = model.geometry
    if geometry is None:
        raise ValueError("_cad_intersection_evidence.model must retain carriers.")
    if coordinate_budget is not None:
        coordinate_budget.ticket.checkpoint()
    edges: dict[int, phx.geometry.brep.IntersectionCurve] = {}
    for edge, curve_index in enumerate(geometry.edge_curves):
        curve = geometry.curves[curve_index]
        if isinstance(curve, phx.geometry.brep.IntersectionCurve):
            edges[edge] = curve
    if not loops:
        if edges:
            raise qualification_failure(
                "_cad_intersection_evidence",
                "An unexpected intersection branch appeared in a planar/analytic case.",
                {"edges": list(edges)},
            )
        return {"branches": 0, "loops": []}
    scale = max(loop[0] for loop in loops)
    tolerance = 1.0e-9 * scale
    ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    lengths: dict[int, float] = {index: 0.0 for index in range(len(loops))}
    endpoints: dict[int, list[int]] = {index: [] for index in range(len(loops))}
    worst = {"three_dimensional": 0.0, "first_pcurve": 0.0, "second_pcurve": 0.0}
    branch_ids = set()
    preparations = TrimRequestPreparations()
    for edge, curve in edges.items():
        if not curve.fully_certified:
            raise qualification_failure(
                "_cad_intersection_evidence",
                "A Boolean intersection branch has uncertified continuation charts.",
                {"edge": edge},
            )
        branch_ids.add(curve.branch_id)
        first, last = ranges[edge]
        parameters = np.linspace(first, last, 2049)
        if coordinate_budget is not None:
            coordinate_budget.ticket.checkpoint()
            coordinate_budget.reserve_geometry_queries(parameters.size)
        evaluated = curve.evaluate(
            jnp.asarray(parameters),
            budget=coordinate_budget,
            preparations=preparations,
        )
        points = np.asarray(evaluated.point, dtype=np.float64)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        side = int(np.sign(np.median(x)))
        axis = np.hypot(x, y)
        candidates = [
            index
            for index, loop in enumerate(loops)
            if loop[3] == side and np.max(np.abs(axis - loop[0])) <= 1.0e-6 * scale
        ]
        if len(candidates) != 1 or np.any(np.sign(x) != side):
            raise qualification_failure(
                "_cad_intersection_evidence",
                "A branch does not lie on exactly one declared analytic loop.",
                {"edge": edge, "candidates": candidates},
            )
        index = candidates[0]
        big, small, center, _ = loops[index]
        residual = np.maximum(
            np.abs(axis - big), np.abs(np.hypot(y, z - center) - small)
        )
        worst["three_dimensional"] = max(worst["three_dimensional"], float(np.max(residual)))
        for name, patch_parameters, patch in (
            ("first_pcurve", evaluated.first_parameters, curve.first.patch),
            ("second_pcurve", evaluated.second_parameters, curve.second.patch),
        ):
            if coordinate_budget is not None:
                coordinate_budget.ticket.checkpoint()
                coordinate_budget.reserve_geometry_queries(parameters.size)
            lifted = np.asarray(patch.evaluate(patch_parameters), dtype=np.float64)
            worst[name] = max(worst[name], float(np.max(np.linalg.norm(lifted - points, axis=1))))
        lengths[index] += float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        endpoints[index].extend(geometry.edge_vertices[edge])
        if coordinate_budget is not None:
            coordinate_budget.ticket.checkpoint()
    loop_records = []
    for index, (big, small, center, side) in enumerate(loops):
        expected = _cross_loop_length(big, small)
        counts = np.unique(endpoints[index], return_counts=True)[1]
        closed = bool(endpoints[index]) and bool(np.all(counts % 2 == 0))
        relative = abs(lengths[index] - expected) / expected
        loop_records.append(
            {
                "axis_radius": big,
                "transverse_radius": small,
                "transverse_center_z": center,
                "side": side,
                "analytic_length": expected,
                "branch_length": lengths[index],
                "relative_length_defect": relative,
                "closed": closed,
            }
        )
        if not closed or relative > 1.0e-5:
            raise qualification_failure(
                "_cad_intersection_evidence",
                "A declared non-rational intersection loop is incomplete or not closed.",
                {"loop": loop_records[-1]},
            )
    if max(worst.values()) > tolerance:
        raise qualification_failure(
            "_cad_intersection_evidence",
            "An intersection branch or p-curve leaves its analytic quadrics.",
            {"worst": worst, "tolerance": tolerance},
        )
    return {
        "branches": len(branch_ids),
        "edges": len(edges),
        "branch_ids": sorted(branch_ids),
        "maximum_residuals": worst,
        "residual_tolerance": tolerance,
        "loops": loop_records,
        "nonrational_reason": "unequal-radius perpendicular cylinders: smooth genus-one quartic",
    }


def _cad_query(
    model: phx.geometry.brep.BRepModel, reference_volume: float,
    *, coordinate_budget: CoordinateEnclosureBudget,
) -> dict[str, object]:
    query = phx.geometry.brep.prepare_brep_query(model, budget=coordinate_budget)
    volume = float(np.sum(np.asarray(query.measures.solid_volumes)))
    relative = abs(volume - reference_volume) / reference_volume
    if relative > 1.0e-6:
        raise qualification_failure(
            "_cad_query",
            "The native B-Rep query disagrees with the independent volume.",
            {"volume": volume, "reference": reference_volume},
        )
    return {
        "query_id": query.query_id,
        "source_revision": query.source_revision,
        "independent_volume": reference_volume,
        "solid_volume": volume,
        "relative_volume_defect": relative,
        "solid_volume_error_estimates": np.asarray(
            query.measures.solid_volume_error_estimates
        ).tolist(),
    }


def _cad_branch_payloads(
    model: phx.geometry.brep.BRepModel,
) -> dict[str, Any]:
    geometry = model.geometry
    if geometry is None:
        raise ValueError("_cad_branch_payloads.model must retain carriers.")
    return {
        curve.branch_id: canonical_fingerprint(curve.payload())
        for curve in geometry.curves
        if isinstance(curve, phx.geometry.brep.IntersectionCurve)
    }


def _cad_persistence(
    model: phx.geometry.brep.BRepModel, directory: Path,
    *, coordinate_budget: CoordinateEnclosureBudget,
) -> tuple[phx.geometry.brep.BRepModel, dict[str, object]]:
    """Lossless native archive round trip; the restored model drives meshing."""
    coordinate_budget.ticket.checkpoint()
    path = directory / "native-cad-model.phx"
    receipt = phx.interchange.save_brep_archive(
        model, path, coordinate_budget=coordinate_budget
    )
    restored = phx.interchange.load_brep_archive(
        path, coordinate_budget=coordinate_budget
    )
    before, after = _cad_branch_payloads(model), _cad_branch_payloads(restored)
    if restored.model_id != model.model_id or before != after:
        raise qualification_failure(
            "_cad_persistence",
            "The native archive changed exact model or branch identity.",
            {"model": model.model_id, "restored": restored.model_id},
        )
    return restored, {
        "receipt_model_id": receipt.model_id,
        "receipt_geometry_id": receipt.geometry_id,
        "model_id": restored.model_id,
        "branch_payloads": after,
        "archive_bytes": path.stat().st_size,
    }


def _cad_interchange_boundaries(
    model: phx.geometry.brep.BRepModel,
    case: _CadCase,
    capacity: int,
    directory: Path,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
) -> dict[str, object]:
    """Exact export refusal and explicitly bounded STEP/IGES approximation."""
    _, export_policy = _cad_interchange_policies(capacity, model.coordinate_contract)
    record: dict[str, object] = {}
    if not case.loops:
        for format_name in ("step", "iges", "brep"):
            reread, entry = _cad_case_roundtrip(
                model, f"exact-{format_name}", format_name, capacity, directory,
                coordinate_budget=coordinate_budget,
            )
            record[f"exact_{format_name}"] = {**entry, "reread_model_id": reread.model_id}
        return record
    for format_name in ("step", "iges", "brep"):
        writer, suffix = _cad_format_writer(format_name)
        path = directory / f"refused{suffix}"
        try:
            writer(
                model, path, policy=export_policy, coordinate_budget=coordinate_budget
            )
        except phx.interchange.CadInterchangeError as error:
            if path.exists() or error.refusal.reason != "inexact-export":
                raise qualification_failure(
                    "_cad_interchange_boundaries",
                    "Exact export refusal must be inexact-export without publication.",
                    {"error": str(error), "exists": path.exists()},
                ) from error
            record[f"exact_{format_name}_refusal"] = {
                "reason": error.refusal.reason,
                "entity": error.refusal.entity,
                "chain": list(error.refusal.chain),
            }
        else:
            raise qualification_failure(
                "_cad_interchange_boundaries",
                "A non-rational branch was exported as an exact external entity.",
                {"format": format_name},
            )
    tolerance = 1.0e-6 * case.minimum_radius
    fit = phx.interchange.CadCurveFitPolicy(tolerance=tolerance)
    approximate = phx.interchange.CadExportPolicy(
        intersection_approximation=fit,
        maximum_bytes=export_policy.maximum_bytes,
        maximum_entities=export_policy.maximum_entities,
    )
    import_policy, _ = _cad_interchange_policies(capacity, model.coordinate_contract)
    source_branches = set(_cad_branch_payloads(model))
    for format_name in ("step", "iges"):
        writer, suffix = _cad_format_writer(format_name)
        path = directory / f"approximate{suffix}"
        exported = writer(
            model, path, policy=approximate, coordinate_budget=coordinate_budget
        )
        reread, entry = _cad_case_import(
            path, format_name, import_policy, directory,
            coordinate_budget=coordinate_budget,
        )
        export_profile = exported.report.target_profile
        import_qualifiers = entry["source_profile_qualifiers"]
        export_qualifiers = {} if export_profile is None else dict(export_profile.qualifiers)
        descriptor = export_qualifiers.get("bounded-approximation-descriptor")
        if (
            not isinstance(descriptor, str)
            or not isinstance(import_qualifiers, dict)
            or import_qualifiers.get("bounded-approximation-descriptor") != descriptor
            or entry["source_digest"] != exported.receipt.content_sha256
            or exported.report.status != phx.interchange.AdapterStatus.DECLARED_LOSS
            or entry["report_status"] != "DECLARED_LOSS"
        ):
            raise qualification_failure(
                "_cad_interchange_boundaries",
                "A bounded artifact must bind its SHA, complete declared gap law and explicit import/export interpretation loss.",
                {"export_qualifiers": export_qualifiers, "import": entry},
            )
        bounds = [
            approximation.continuous_evidence.distance_bound
            for approximation in exported.approximations
        ]
        geometry = reread.geometry
        actual_gaps = np.asarray(reread.coedge_deviation_bounds)
        if (
            {approximation.branch_id for approximation in exported.approximations}
            != source_branches
            or not bounds
            or max(bounds) > tolerance
            or reread.model_id == model.model_id
            or geometry is None
            or geometry.edge_face_gap_bounds is None
            or actual_gaps.shape != (len(geometry.coedge_edges),)
            or not np.all(np.isfinite(actual_gaps))
            or np.any(actual_gaps > np.asarray(geometry.edge_face_gap_bounds))
            or any(
                isinstance(curve, phx.geometry.brep.IntersectionCurve)
                for curve in geometry.curves
            )
        ):
            raise qualification_failure(
                "_cad_interchange_boundaries",
                "Bounded approximation must cover every branch within tolerance under a new identity.",
                {"bounds": bounds, "tolerance": tolerance},
            )
        sampled = _cad_approximate_edge_distance(reread, case.loops)
        if sampled > tolerance * (1.0 + 1.0e-6) + 1.0e-12 * case.minimum_radius:
            raise qualification_failure(
                "_cad_interchange_boundaries",
                "An approximated edge leaves the analytic quadrics beyond its bound.",
                {"sampled": sampled, "tolerance": tolerance},
            )
        record[f"approximate_{format_name}"] = {
            **entry,
            "export_policy_id": approximate.policy_id,
            "fit_policy_id": fit.policy_id,
            "tolerance": tolerance,
            "approximation_ids": [
                approximation.approximation_id for approximation in exported.approximations
            ],
            "continuous_distance_bounds": bounds,
            "independent_sampled_distance": sampled,
            "exact_model_id": model.model_id,
            "approximate_model_id": reread.model_id,
            "declared_representation": "bounded-coupled-b-spline-approximation",
            "artifact_sha256": exported.receipt.content_sha256,
            "bounded_approximation_descriptor": descriptor,
            "declared_stored_gap_bounds": list(geometry.edge_face_gap_bounds),
            "actual_carrier_gap_bounds": actual_gaps.tolist(),
            "stored_gap_proof": "independent continuous emitted carrier-to-surface validation",
        }
    return record


def _cad_approximate_edge_distance(
    model: phx.geometry.brep.BRepModel,
    loops: tuple[tuple[float, float, float, float], ...],
) -> float:
    """Largest sampled distance of approximated branch edges to both quadrics."""
    geometry = model.geometry
    if geometry is None:
        raise ValueError("_cad_approximate_edge_distance.model must retain carriers.")
    ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    scale = max(loop[0] for loop in loops)
    worst, matched = 0.0, 0
    for edge, index in enumerate(geometry.edge_curves):
        first, last = ranges[edge]
        points = np.asarray(
            geometry.curves[index].evaluate(jnp.linspace(first, last, 1025)),
            dtype=np.float64,
        )
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        axis = np.hypot(x, y)
        for big, small, center, side in loops:
            transverse = np.hypot(y, z - center)
            if (
                np.all(np.sign(x) == side)
                and np.max(np.abs(axis - big)) <= 1.0e-4 * scale
                and np.max(np.abs(transverse - small)) <= 1.0e-4 * scale
            ):
                matched += 1
                worst = max(
                    worst,
                    float(np.max(np.abs(axis - big))),
                    float(np.max(np.abs(transverse - small))),
                )
    if matched == 0:
        raise qualification_failure(
            "_cad_approximate_edge_distance",
            "No approximated branch edge was found on the declared quadrics.",
            {"edges": len(geometry.edge_curves)},
        )
    return worst


def _cad_whole_scope(
    domain: phx.geometry.MeshingDomain, dimension: int
) -> phx.meshing.MeshingScope:
    return phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        dimension,
        domain.entity_set_id(dimension),
        np.asarray(domain.scope_indices(dimension), dtype=np.int64),
    )


@final
class _CadVolume(phx.StrictModule):
    domain: phx.geometry.MeshingDomain
    surface: phx.meshing.CellMeshingResult
    volume: phx.meshing.CellMeshingResult
    generation_source: phx.meshing.NativeCadVolumeSource
    generation_specification: phx.meshing.VolumeMeshingSpec
    generation_options: phx.meshing.NativeMeshingOptions
    generation_plan: phx.meshing.NativeMeshingPlan


def _cad_volume_mesh(
    model: phx.geometry.brep.BRepModel,
    case: _CadCase,
    args: QualificationControls,
    record_phase: Callable[[phx.meshing.NativeMeshingPhaseMeasurement], None],
) -> _CadVolume:
    """Own one original-limit request before preparing either native route."""
    limits = _cad_native_limits(args)
    with NativeExecutionScope(limits.native_limits()) as scope:
        published = _cad_volume_mesh_in_execution(model, case, args, record_phase)
    if scope.evidence is None:
        raise RuntimeError("CAD surface-volume request must retain its ended evidence.")
    volume = bind_native_execution_result(
        published.volume, scope.evidence, limits, scope.operation_started
    )
    return _CadVolume(
        domain=published.domain,
        surface=published.surface,
        volume=volume,
        generation_source=published.generation_source,
        generation_specification=published.generation_specification,
        generation_options=published.generation_options,
        generation_plan=published.generation_plan,
    )


def _cad_volume_mesh_in_execution(
    model: phx.geometry.brep.BRepModel,
    case: _CadCase,
    args: QualificationControls,
    record_phase: Callable[[phx.meshing.NativeMeshingPhaseMeasurement], None],
) -> _CadVolume:
    """Prepare the surface, construct its CAD source and fill under the live root."""
    domain = phx.geometry.MeshingDomain.from_brep(model)
    if domain.authority_id != model.model_id:
        raise qualification_failure(
            "_cad_volume_mesh",
            "The meshing domain lost its B-Rep authority.",
            {"domain": domain.authority_id, "model": model.model_id},
        )
    faces = _cad_whole_scope(domain, 2)
    limits = _cad_native_limits(args)
    surface_source = phx.meshing.NativeSurfaceSource(domain)
    deviation = phx.meshing.ProtectedFeature(
        faces, phx.meshing.FeatureKind.SURFACE, maximum_deviation=case.deviation
    )
    surface_request = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        faces,
        size_controls=(
            phx.meshing.UniformSizeControl(
                faces, case.size, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        protected_features=(deviation,),
        limits=limits,
    )
    surface_operands = CoordinateOperandCollector(
        capture_affine_areas=True,
    )
    surface = (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("parametric_surface")
        )
        .plan(
            surface_source,
            surface_request,
            coordinate_contract=model.coordinate_contract,
            record_phase=record_phase,
        )
        .execute(record_phase=record_phase, coordinate_operands=surface_operands)
    )
    source = phx.meshing.NativeCadVolumeSource(surface_source, surface)
    volume_request = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        faces,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                faces, case.size, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        protected_features=(deviation,),
        limits=limits,
    )
    volume_options = phx.meshing.NativeMeshingOptions("cad_tetrahedral")
    volume_plan = phx.meshing.NativeMeshingProvider(volume_options).plan(
        source,
        volume_request,
        coordinate_contract=model.coordinate_contract,
        record_phase=record_phase,
    )
    volume = volume_plan.execute(
        record_phase=record_phase,
        coordinate_operands=CoordinateOperandCollector(
            source_operands=surface_operands.prepared
        ),
    )
    return _CadVolume(
        domain=domain,
        surface=surface,
        volume=volume,
        generation_source=source,
        generation_specification=volume_request,
        generation_options=volume_options,
        generation_plan=volume_plan,
    )


def _cad_volume_acceptance(
    published: _CadVolume, model: phx.geometry.brep.BRepModel, case: _CadCase
) -> dict[str, object]:
    """Root-CAD publication evidence plus independent region-measure checks."""
    result = published.volume
    certificate = result.certification
    outcomes = (
        {}
        if certificate is None
        else {outcome.check: outcome.status for outcome in certificate.outcomes}
    )
    if (
        certificate is None
        or not certificate.passed
        or not result.audit.passed
        or {"source_fidelity", "domain_coverage", "global_embedding"} - set(outcomes)
        or result.trace.binding is None
        or (result.trace.binding.source_id, result.trace.binding.source_revision)
        != (model.source_id, model.source_revision)
    ):
        raise qualification_failure(
            "_cad_volume_acceptance",
            "The tetrahedra need root B-Rep fidelity, coverage and embedding certification.",
            {"outcomes": outcomes},
        )
    if not result.associations or any(
        association.association_kind is not phx.meshing.GeometryAssociationKind.BREP
        or association.source_id != model.source_id
        or association.source_revision != model.source_revision
        for association in result.associations
    ):
        raise qualification_failure(
            "_cad_volume_acceptance",
            "The tetrahedra lost exact B-Rep associations.",
            {"associations": len(result.associations)},
        )
    mesh = result.mesh
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    corners = points[cells]
    signed = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
    affine_volume = float(np.sum(signed))
    surface_points = np.asarray(published.surface.mesh.coordinates, dtype=np.float64)
    if np.any(signed <= 0.0):
        raise qualification_failure(
            "_cad_volume_acceptance",
            "An affine tetrahedron is not positively oriented.",
            {"minimum": float(np.min(signed))},
        )
    return {
        "surface": _result_identity(published.surface),
        "volume": _result_identity(result),
        "domain_id": published.domain.domain_id,
        "certification_id": certificate.report_id,
        "boundary_fidelity": None
        if certificate.fidelity is None
        else {
            "certificate_id": certificate.fidelity.certificate_id,
            "tolerance": certificate.fidelity.tolerance,
            "mesh_to_source_upper": certificate.fidelity.mesh_to_source_upper,
            "source_to_mesh_upper": certificate.fidelity.source_to_mesh_upper,
        },
        "outcomes": outcomes,
        "requested": dict(certificate.requested),
        "achieved": dict(certificate.achieved),
        "surface_vertices": int(surface_points.shape[0]),
        "vertices": int(points.shape[0]),
        "tetrahedra": int(cells.shape[0]),
        "affine_volume": affine_volume,
        "affine_relative_volume_defect": abs(affine_volume - case.reference_volume)
        / case.reference_volume,
        "zones": [zone.name for zone in result.zones],
        "quality": _quality_summary(
            phx.meshing.summarize_cell_quality(phx.meshing.evaluate_cell_quality(mesh))
        ),
    }


def _cad_curving(
    published: _CadVolume, model: phx.geometry.brep.BRepModel, case: _CadCase
) -> tuple[phx.meshing.CellMeshingResult, dict[str, object]]:
    from phydrax.meshing._native_preparation import stage_native_completed_source_issue

    limits = published.generation_specification.limits
    with (
        NativeExecutionScope(limits.native_limits()) as scope,
        NativeHostStorageWorkspace(scope) as workspace,
    ):
        with scope.batch(0) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
        with (
            budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)),
            meshing_source_owner_reconstruction_scope(budget=budget),
        ):
            carrier, evidence = _cad_curving_in_execution(
                published, model, case,
                NativePublicationExecution(scope, workspace, None, budget),
            )
            with budget.bind_ticket(scope.ticket_view()):
                issue = stage_native_completed_source_issue(
                    scope, carrier, owner_id=published.generation_plan.plan_id, budget=budget
                )
    if scope.evidence is None:
        raise RuntimeError("CAD curving must retain its complete request evidence")
    bound = bind_native_execution_result(
        carrier, scope.evidence, limits, scope.operation_started,
        owner_id=published.generation_plan.plan_id,
        history_evidence=()
        if published.volume.execution_evidence is None
        else (published.volume.execution_evidence,),
    )
    issue.complete(bound)
    return bound, evidence


def _remember_cad_publication_registration(
    plan: phx.meshing.NativeMeshingPlan,
    carrier: phx.meshing.CellMeshingResult,
    execution: NativePublicationExecution,
) -> None:
    """Seal the actual P2 report only after full real inspection and publication."""
    from phydrax.lifecycle._meshing_registration import (
        _remember_issued_native_generation_registration,
    )
    from phydrax.meshing.providers._native import _require_issued_native_plan

    budget = execution.budget
    if budget is None:
        raise ValueError("CAD registration must use its actual publication budget")
    if (
        not isinstance(plan.source, phx.meshing.NativeCadVolumeSource)
        or not isinstance(plan.specification, phx.meshing.VolumeMeshingSpec)
    ):
        raise TypeError("CAD registration must retain the actual CAD volume generation plan")
    with budget.bind_ticket(execution.scope.ticket_view()):
        _require_issued_native_plan(plan, budget)
        _remember_issued_native_generation_registration(
            plan.source, plan.specification, plan.options, carrier, budget=budget
        )

def _cad_straight_rejection(
    mesh: phx.discretization.CellMesh,
    association: GeometryAssociation,
    projection: NativeBRepProjection,
    straight: phx.discretization.CellGeometrySpec,
    curved: phx.discretization.CellGeometrySpec,
    policy: phx.meshing.HighOrderCurvingPolicy,
    budget: CoordinateEnclosureBudget,
) -> dict[str, object]:
    """Fresh lower-distance evidence for one failed straight-node obligation."""
    query = projection.query
    if not isinstance(query, phx.geometry.brep.PreparedBRepQuery):
        raise TypeError("CAD rejection witness requires its prepared B-Rep query owner.")
    if (
        association.source_id != projection.source_id
        or association.source_revision != projection.source_revision
        or query.model.source_id != projection.source_id
        or query.model.model_id != projection.model_id
        or query.source_revision != projection.source_revision
        or straight.geometry_layout_id != curved.geometry_layout_id
        or straight.coordinates.shape != curved.coordinates.shape
    ):
        raise ValueError(
            "CAD rejection witness must retain exact association, query, and "
            "geometry-layout identity."
        )
    count, dimension = straight.coordinates.shape
    with budget.temporary_scope():
        # The class owner bounds its own temporary classification work. This
        # reservation covers its returned node metadata and our host coordinate,
        # candidate, displacement, and ordering arrays before any are materialized.
        budget.reserve(
            count * (4 * dimension + max(1, count.bit_length())),
            4096 + count * (40 * dimension + 96),
        )
        classes = curving_owner._node_classes(mesh, straight, association, projection)
        if (
            classes.dimensions.shape != (count,)
            or classes.indices.shape != (count,)
            or classes.resolved.shape != (count,)
            or classes.constrained.shape != (count,)
            or len(classes.occurrence_paths) != count
        ):
            raise ValueError(
                "CAD rejection node classes must align with the straight geometry layout."
            )
        straight_points = np.asarray(straight.coordinates, dtype=np.float64)
        curved_points = np.asarray(curved.coordinates, dtype=np.float64)
        candidate_mask = classes.constrained & classes.resolved
        candidate_mask &= classes.dimensions == 2
        candidates = np.flatnonzero(candidate_mask)
        if not candidates.size:
            raise qualification_failure(
                "_cad_curving",
                "Straight P2 rejection has no resolved lower-distance witness.",
                {"candidates": 0, "statuses": {}},
            )
        displacement = curved_points - straight_points
        squared = np.sum(displacement * displacement, axis=1)
        del displacement
        order = np.lexsort((candidates, -squared[candidates]))
        preparations = TrimRequestPreparations()
        statuses: dict[int, int] = {}
        for rank, position in enumerate(order):
            node = int(candidates[position])
            face = int(classes.indices[node])
            path = classes.occurrence_paths[node]
            if face < 0 or face >= len(query.faces):
                raise ValueError("CAD rejection witness must retain its assigned source face.")
            expected_occurrence = -1
            if path:
                budget.reserve(len(query.occurrences))
                expected_occurrence = next(
                    (
                        index
                        for index, occurrence in enumerate(query.occurrences)
                        if occurrence.path == path
                    ),
                    -1,
                )
                if expected_occurrence < 0:
                    raise ValueError(
                        "CAD rejection witness must retain its assigned face occurrence."
                    )
            point = straight_points[node : node + 1]
            if path:
                result = query.closest_point(
                    point,
                    faces=(face,),
                    path=path,
                    coordinate_budget=budget,
                    preparations=preparations,
                )
            elif not query.occurrences:
                result = query.closest_point(
                    point,
                    faces=(face,),
                    coordinate_budget=budget,
                    preparations=preparations,
                )
            else:
                result = query.closest_definition(
                    jnp.asarray(point, dtype=jnp.float64),
                    (face,),
                    coordinate_budget=budget,
                    preparations=preparations,
                )
            if result.source_revision != projection.source_revision:
                raise ValueError(
                    "CAD rejection query must preserve the bound source revision."
                )
            result_face = int(result.faces[0])
            occurrence = int(result.occurrences[0])
            if result_face != face:
                raise ValueError(
                    "CAD rejection query must preserve the assigned source face."
                )
            if occurrence != expected_occurrence:
                raise ValueError(
                    "CAD rejection query must preserve the assigned face occurrence."
                )
            if path and query.occurrences[occurrence].path != path:
                raise ValueError(
                    "CAD rejection query must preserve the assigned face occurrence."
                )
            status = int(result.status[0])
            statuses[status] = statuses.get(status, 0) + 1
            lower = float(result.distance_lower_bounds[0])
            upper = float(result.distances[0])
            if (
                status not in RESOLVED_STATUS
                or bool(result.unresolved[0])
                or not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower > upper
                or lower <= policy.residual_tolerance
            ):
                continue
            budget.retain_operand_metadata(
                4096 + sum(64 + 4 * len(part) for part in path)
            )
            # The fresh lower bound proves one failed nodal obligation, not a
            # maximum-residual or global-fidelity certificate.
            return {
                "criterion": "constrained-node-distance",
                "bound_scope": "single-node-lower-bound-not-global-maximum",
                "node": node,
                "entity_dimension": int(classes.dimensions[node]),
                "entity_index": face,
                "face": face,
                "occurrence_path": list(path),
                "query_point": straight_points[node].tolist(),
                "query_occurrence": occurrence,
                "query_status": status,
                "distance_lower_bound": lower,
                "distance_upper_bound": upper,
                "residual_tolerance": policy.residual_tolerance,
                "candidates_tried": rank + 1,
                "source_id": projection.source_id,
                "source_revision": result.source_revision,
                "projection_id": projection.projection_id,
                "query_id": query.query_id,
                "geometry_layout_id": straight.geometry_layout_id,
            }
    raise qualification_failure(
        "_cad_curving",
        "Straight P2 rejection has no resolved lower-distance witness.",
        {"candidates": candidates.size, "statuses": statuses},
    )



def _cad_curving_in_execution(
    published: _CadVolume,
    model: phx.geometry.brep.BRepModel,
    case: _CadCase,
    execution: NativePublicationExecution,
) -> tuple[phx.meshing.CellMeshingResult, dict[str, object]]:
    """Certified P2 geometry on the root B-Rep with continuous fidelity/coverage."""
    budget = execution.budget
    if budget is None:
        raise ValueError("CAD curving requires its actual publication coordinate budget")
    result = published.volume
    certification = result.certification
    if certification is None or certification.request.source is None:
        raise qualification_failure(
            "_cad_curving",
            "Curving requires the validated root-CAD source retained by publication.",
            {"certification": None if certification is None else certification.report_id},
        )
    source = certification.request.source
    mesh = result.mesh
    vertex_set = mesh.entity_set(0).entity_set_id
    association = next(
        value for value in result.associations if value.target_entity_set_id == vertex_set
    )
    projection = phx.geometry.brep.prepare_brep_projection(model)
    tolerance = 0.25 * case.deviation
    if len(published.domain.regions) != 1:
        raise qualification_failure(
            "_cad_curving",
            "Certified CAD coverage of curved geometry is declared for one solid.",
            {"regions": len(published.domain.regions)},
        )
    source_domain = phx.geometry.CadSourceDomain(
        model, region_ids=(published.domain.regions[0].name,)
    )
    regions = np.zeros(mesh.blocks[0].cell_count, dtype=np.int64)
    publication_audit = required_audit_policy(
        native_family_audit_policy(
            published.generation_source, published.generation_specification, mesh
        ),
        certification.schedule.required_audit_checks,
    )
    curving_policy = phx.meshing.HighOrderCurvingPolicy(
        degree=2, fidelity_tolerance=tolerance,
        validity=publication_audit.validity_policy,
    )
    with budget.bind_ticket(execution.scope.ticket_view()):
        curved = phx.meshing.curve_cell_mesh(
            mesh,
            association,
            projection,
            policy=curving_policy,
            source=source,
            domain=source_domain,
            cell_regions=regions,
            certificate_limits=certification.request.limits,
            budget=budget,
        )
    evidence = curved.evidence
    fidelity, coverage = evidence.fidelity, evidence.coverage
    if (
        curved.status is not phx.meshing.HighOrderCurvingStatus.CURVED
        or not evidence.accepted
        or not evidence.certificate.all_certified
        or fidelity is None
        or fidelity.status != "certified"
        or coverage is None
        or coverage.status != "certified"
        or not isinstance(coverage, phx.geometry.ApproximateSourceCoverageCertificate)
        or coverage.distance_upper > fidelity.tolerance
    ):
        raise qualification_failure(
            "_cad_curving",
            "P2 CAD geometry lacks continuous fidelity, coverage or validity.",
            {"status": curved.status, "failures": evidence.certification_failures},
        )
    recertified = phx.discretization.certify_cell_geometry_validity(
        curved.geometry, mesh=mesh, policy=curving_policy.validity
    )
    if recertified.certificate_id != evidence.certificate.certificate_id:
        raise qualification_failure(
            "_cad_curving",
            "Independent validity recertification differs from curving evidence.",
            {"recertified": recertified.certificate_id},
        )
    with budget.bind_ticket(execution.scope.ticket_view()):
        straight_rejection = _cad_straight_rejection(
            mesh, association, projection, curved.straight, curved.geometry, curving_policy, budget
        )
    inputs = phx.meshing.MeshCertificationInputs(
        mesh,
        curved.geometry,
        certification.schedule,
        domain=source_domain,
        cell_regions=regions,
        source=source,
        fidelity_tolerance=tolerance,
        fidelity_sample_order=certification.request.fidelity_sample_order,
        limits=certification.request.limits,
        budget=budget,
    )
    carrier = qualify_hybrid_curved_publication(
        result,
        curved.geometry,
        association,
        inputs,
        limits=published.generation_specification.limits,
        audit_policy=publication_audit,
        curving=evidence,
        execution=execution,
    )
    if (
        published.generation_plan.source is not published.generation_source
        or published.generation_plan.specification is not published.generation_specification
        or published.generation_plan.options is not published.generation_options
    ):
        raise ValueError("CAD publication must retain its exact inspected generation owners")
    _remember_cad_publication_registration(published.generation_plan, carrier, execution)
    return carrier, {
        "published_result": _result_identity(carrier),
        "published_certification_id": None
        if carrier.certification is None else carrier.certification.report_id,
        "result_id": curved.result_id,
        "evidence_id": evidence.evidence_id,
        "accepted_round": curved.accepted_round,
        "relaxation": [
            {
                "method": minimization.provenance.method,
                "status": np.asarray(minimization.status).tolist(),
                "iterations": np.asarray(minimization.diagnostics.iterations).tolist(),
                "final_optimality_norm": np.asarray(
                    minimization.diagnostics.final_optimality_norm
                ).tolist(),
                "method_evidence_fingerprint": array_tree_fingerprint(
                    minimization.method_evidence
                ),
            }
            for minimization in curved.minimizations
        ],
        "curving_status": curved.status.value,
        "curving_accepted": evidence.accepted,
        "geometry_layout_id": curved.geometry.geometry_layout_id,
        "fidelity_tolerance": tolerance,
        "fidelity_certificate": fidelity.certificate_id,
        "fidelity_status": fidelity.status,
        "mesh_to_source_upper": fidelity.mesh_to_source_upper,
        "source_to_mesh_upper": fidelity.source_to_mesh_upper,
        "coverage_certificate": coverage.certificate_id,
        "coverage_status": coverage.status,
        "coverage_distance_upper": coverage.distance_upper,
        "cad_closure": None
        if fidelity.cad_closure is None
        else fidelity.cad_closure.to_record(),
        "maximum_node_residual": evidence.maximum_residual,
        "residual_tolerance": evidence.residual_tolerance,
        "minimum_scaled_jacobian": evidence.minimum_scaled_jacobian,
        "maximum_distortion": evidence.maximum_distortion,
        "global_embedding_certificate": evidence.embedding.certificate_id,
        "global_embedding_status": evidence.embedding.status,
        "straight_node_rejection": straight_rejection,
        "validity_certificate": recertified.certificate_id,
        "certified_valid_cells": recertified.certified_valid_count,
    }


_P2_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _p2_reference_rule(order: int) -> tuple[Any, Any, Any, Any]:
    """Collapsed Gauss rule on the unit tetrahedron and canonical P2 tabulation."""
    nodes, weights = _gauss_interval(order, 0.0, 1.0)
    a, b, c = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    wa, wb, wc = np.meshgrid(weights, weights, weights, indexing="ij")
    x = a.ravel()
    y = (1.0 - a.ravel()) * b.ravel()
    z = (1.0 - a.ravel()) * (1.0 - b.ravel()) * c.ravel()
    w = (wa * wb * wc).ravel() * (1.0 - a.ravel()) ** 2 * (1.0 - b.ravel())
    lam = np.stack((1.0 - x - y - z, x, y, z), axis=1)
    grad_lam = np.asarray(
        ((-1.0, -1.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    values = [lam[:, i] * (2.0 * lam[:, i] - 1.0) for i in range(4)]
    gradients = [
        (4.0 * lam[:, i] - 1.0)[:, None] * grad_lam[i][None, :] for i in range(4)
    ]
    for i, j in _P2_EDGES:
        values.append(4.0 * lam[:, i] * lam[:, j])
        gradients.append(
            4.0 * (lam[:, j, None] * grad_lam[i] + lam[:, i, None] * grad_lam[j])
        )
    return np.stack(values, axis=1), np.stack(gradients, axis=1), w, lam


def _p2_canonical_order(mesh: Any, element: FiniteElementSpec) -> Any:
    """Canonical roles from exact entity functionals, not positional searches."""
    coordinate = coordinate_lagrange_element("tetrahedron", 2)
    field = _cad_p2_field().elements[0]
    if element.element_id not in (coordinate.element_id, field.element_id):
        raise qualification_failure(
            "_p2_canonical_order", "The basis must be a canonical P2 scalar basis.", {}
        )
    vertices, edges = element.entity_dofs[:2]
    if len(vertices) != 4 or len(edges) != 6 or any(
        len(dofs) != 1 for dofs in (*vertices, *edges)
    ):
        raise qualification_failure(
            "_p2_canonical_order", "P2 needs one functional per vertex and edge.", {}
        )
    topology = phx.discretization.reference_cell_topology("tetrahedron")
    edge_rows = {
        tuple(sorted(edge)): row for row, edge in enumerate(topology.entities[1])
    }
    order = np.asarray(
        [dofs[0] for dofs in vertices]
        + [edges[edge_rows[edge]][0] for edge in _P2_EDGES],
        dtype=np.int64,
    )
    if not np.array_equal(np.sort(order), np.arange(10)):
        raise qualification_failure(
            "_p2_canonical_order", "P2 functionals lack unique vertex/edge roles.", {}
        )
    return np.broadcast_to(order, (mesh.blocks[0].cell_count, 10))


def _cad_manufactured(points: Any, length: float) -> Any:
    scaled = points / length
    return np.exp(scaled[..., 0]) * np.sin(scaled[..., 1]) * np.cos(scaled[..., 2])


def _cad_p2_error(
    mesh: Any,
    geometry: Any,
    space: Any,
    values: Any,
    constrained: Any,
    length: float,
) -> dict[str, float]:
    """Independent NumPy isoparametric L2 error, volume and Jacobian sign."""
    order = _p2_canonical_order(mesh, geometry.elements[0])
    field_order = _p2_canonical_order(mesh, _cad_p2_field().elements[0])
    rows = np.arange(order.shape[0])[:, None]
    geometry = np.asarray(geometry.coordinates, dtype=np.float64)[
        np.asarray(geometry.geometry_dofs[0], dtype=np.int64)
    ][rows, order]
    dofs = np.asarray(space.dof_maps[0].cell_dofs[0], dtype=np.int64)
    if dofs.shape != order.shape:
        raise qualification_failure(
            "_cad_p2_error", "The P2 field is not one dof per geometry node.", {}
        )
    dofs = dofs[rows, field_order]
    nodal = np.asarray(values, dtype=np.float64)[dofs]
    fixed = np.isin(dofs, np.asarray(constrained, dtype=np.int64))
    exact_nodes = _cad_manufactured(geometry, length)
    amplitude = max(1.0, float(np.max(np.abs(exact_nodes))))
    if not np.any(fixed) or float(
        np.max(np.abs(nodal[fixed] - exact_nodes[fixed]))
    ) > 1.0e-10 * amplitude:
        raise qualification_failure(
            "_cad_p2_error",
            "Dirichlet field nodes do not coincide with curved geometry nodes.",
            {},
        )
    basis, gradients, weights, _ = _p2_reference_rule(6)
    squared = norm = volume = 0.0
    minimum_determinant = math.inf
    for start in range(0, geometry.shape[0], 2048):
        x = geometry[start : start + 2048]
        u = nodal[start : start + 2048]
        jacobian = np.einsum("ckd,qke->cqde", x, gradients)
        determinant = np.linalg.det(jacobian)
        minimum_determinant = min(minimum_determinant, float(np.min(determinant)))
        physical = np.einsum("qk,ckd->cqd", basis, x)
        exact = _cad_manufactured(physical, length)
        discrete = np.einsum("qk,ck->cq", basis, u)
        measure = determinant * weights[None, :]
        squared += float(np.sum(measure * (discrete - exact) ** 2))
        norm += float(np.sum(measure * exact**2))
        volume += float(np.sum(measure))
    if not minimum_determinant > 0.0:
        raise qualification_failure(
            "_cad_p2_error",
            "A curved P2 tetrahedron has a nonpositive sampled Jacobian.",
            {"minimum_determinant": minimum_determinant},
        )
    return {
        "l2_error": math.sqrt(squared),
        "relative_l2_error": math.sqrt(squared / norm),
        "curved_volume": volume,
        "minimum_sampled_jacobian": minimum_determinant,
    }


def _cad_p2_field() -> phx.discretization.FiniteElementFieldSpec:
    return phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("tetrahedron", 2)
    )


def _cad_prepare_p2_solver(
    carrier: phx.meshing.CellMeshingResult,
    length: float,
    declaration: MeshingFiniteElementFieldDeclaration | None = None,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[Any, Any, Any]:
    """Prepare the original diffusion equation, including its curved boundary lift."""
    if (
        not np.isfinite(length)
        or length <= 0.0
        or len(carrier.mesh.blocks) != 1
        or carrier.mesh.blocks[0].cell_kind != "tetrahedron"
        or any(
            not isinstance(element, FiniteElementSpec) or element.degree != 2
            for element in carrier.geometry.elements
        )
    ):
        raise qualification_failure(
            "_cad_prepare_p2_solver",
            "CAD diffusion requires an actual P2 tetrahedral carrier and its original length.",
            {"length": length, "carrier": carrier.result_id},
        )
    field = _cad_p2_field()
    if declaration is None:
        space = phx.discretization.FiniteElementPlan(
            carrier.mesh, field, coordinate_spec=carrier.geometry, budget=budget
        ).prepare(budget=budget)
    else:
        if (
            len(declaration.finite_element_fields) != 1
            or declaration.finite_element_fields[0].field_spec_id != field.field_spec_id
        ):
            raise ValueError("cad.cold.declaration must retain the original scalar P2 field")
        space = phx.discretization.FiniteElementPlan(
            carrier.mesh,
            declaration.finite_element_fields,
            coordinate_spec=carrier.geometry,
            precision_policy=declaration.precision_policy,
            budget=budget,
        ).prepare(numeric_version=declaration.numeric_version, budget=budget)
        if dict(declaration.field_space_ids) != {
            "u": space.field_spaces[0].field_space_id
        }:
            raise ValueError("cad.cold.declaration must bind the restored P2 coefficient owner")
    constraint = phx.discretization.dirichlet_constraint(space, "u")
    if not constraint.free_dofs.size:
        raise qualification_failure(
            "_cad_prepare_p2_solver",
            "The CAD P2 diffusion solve must have interior degrees of freedom.",
            {"carrier": carrier.result_id},
        )

    def exact(points: Any) -> Any:
        scaled = points / length
        return jnp.exp(scaled[..., 0]) * jnp.sin(scaled[..., 1]) * jnp.cos(scaled[..., 2])

    def forcing(points: Any, context: Any) -> Any:
        del context
        return exact(points) / (length * length)

    form = phx.equations.FiniteElementForm(
        "native-cad-curved-p2-diffusion",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    forcing,
                    coefficient_id=canonical_fingerprint(
                        {"kind": "exp-sin-cos-negative-laplacian", "length": length}
                    ),
                ),
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form, space, constraint=constraint, dirichlet_values=exact, budget=budget
    )
    return space, problem, constraint.free_dofs


def _cad_p2_accepted_epoch(
    carrier: phx.meshing.CellMeshingResult,
    space: Any,
    problem: Any,
    values: Any,
    free: Any,
) -> MeshingAcceptedEpoch:
    """Retain solved coefficients and the exact essential conditions, not a reseed."""
    count = space.dof_maps[0].global_dof_count
    mask = np.ones(count, dtype=np.bool_)
    mask[np.asarray(free, dtype=np.int64)] = False
    fields = {
        "u": values,
        "boundary-mask": mask,
        "boundary-values": jnp.asarray(problem.lift),
        "accepted-epoch": np.asarray(0, dtype=np.int64),
    }
    binding = prepare_meshing_field_index_binding(
        carrier, "field/global_coefficient_ids", field_space=space.field_spaces[0]
    )
    declaration = MeshingFiniteElementFieldDeclaration(
        part_name="cad",
        topology_id=carrier.mesh.topology_id,
        geometry_layout_id=carrier.geometry.geometry_layout_id,
        field_space_ids=frozendict({"u": space.field_spaces[0].field_space_id}),
        value_units=frozendict({"u": (phx.units.ONE,)}),
        maximum_derivative_orders=frozendict({"u": 1}),
        branch_policy=phx.BranchDifferentiationPolicy.FROZEN_DECISION,
        trace_policy=phx.discretization.FieldTracePolicy("cell-sided"),
        state_roles=tuple(
            MeshingFieldStateRole(
                name,
                "u",
                role,
                tuple(fields[name].shape),
                np.dtype(fields[name].dtype),
                0,
                index_binding=None if role == "state-epoch" else binding,
            )
            for name, role in (
                ("u", "coefficients"),
                ("boundary-mask", "boundary-mask"),
                ("boundary-values", "boundary-values"),
                ("accepted-epoch", "state-epoch"),
            )
        ),
        history_policy="accepted-state",
        numeric_version=space.numeric_version,
        finite_element_fields=(_cad_p2_field(),),
        precision_policy=space.precision_policy,
    )
    return MeshingAcceptedEpoch(carrier, (declaration,), fields)


def _cad_p2_solve(
    carrier: phx.meshing.CellMeshingResult,
    case: _CadCase,
    args: QualificationControls,
    *,
    deadline: Deadline,
    budget: CoordinateEnclosureBudget,
) -> tuple[MeshingAcceptedEpoch, Any, dict[str, Any]]:
    """Solve, accept the actual P2 state, then continue from that accepted state."""
    deadline.remaining()
    space, problem, free = _cad_prepare_p2_solver(
        carrier, case.manufactured_length, budget=budget
    )
    seed = jnp.zeros((free.size,), dtype=carrier.mesh.coordinates.dtype)
    values, compiler, timings = _compiled_solver_stages(problem, seed, args.repeats)
    accepted = _cad_p2_accepted_epoch(carrier, space, problem, values, free)
    constrained = np.flatnonzero(np.asarray(accepted.fields["boundary-mask"]))
    error = _cad_p2_error(
        carrier.mesh, carrier.geometry, space, values, constrained, case.manufactured_length
    )
    deadline.remaining()
    continued, continued_compiler, continued_timings = _compiled_solver_stages(
        problem, jnp.asarray(accepted.fields["u"])[free], args.repeats
    )
    continued_error = _cad_p2_error(
        carrier.mesh, carrier.geometry, space, continued, constrained, case.manufactured_length
    )
    return accepted, continued, {
        **error,
        "dofs": space.dof_maps[0].global_dof_count,
        "free_dofs": int(free.size),
        "compiler": compiler,
        "stages_seconds": timings,
        "compilation_id": problem.compilation_id,
        "accepted_epoch_id": accepted.epoch_id,
        "accepted_field_fingerprint": array_tree_fingerprint(accepted.fields),
        "continuation": {
            **continued_error,
            "compiler": continued_compiler,
            "stages_seconds": continued_timings,
            "seed_fingerprint": array_tree_fingerprint(jnp.asarray(accepted.fields["u"])[free]),
            "solution_fingerprint": array_tree_fingerprint(continued),
            "started_from": "accepted-P2-coefficients",
        },
        "curved_relative_volume_defect": abs(error["curved_volume"] - case.reference_volume)
        / case.reference_volume,
    }


def _cad_checkpoint_identity(records: Any) -> dict[str, Any]:
    """Bind the source request and every canonical accepted numerical bank."""
    history = validate_meshing_accepted_epochs(records)
    source, request, options = (
        records["generation_source"],
        records["generation_specification"],
        records["generation_options"],
    )
    if (
        not isinstance(source, phx.meshing.NativeCadVolumeSource)
        or not isinstance(request, phx.meshing.VolumeMeshingSpec)
        or not isinstance(options, phx.meshing.NativeMeshingOptions)
        or len(history) != 1
        or len(history[0].declarations) != 1
        or type(history[0].declarations[0]) is not MeshingFiniteElementFieldDeclaration
        or set(history[0].fields)
        != {"u", "boundary-mask", "boundary-values", "accepted-epoch"}
    ):
        raise ValueError("cad.checkpoint must retain the complete original CAD/P2 physical epoch")
    carrier = history[-1].carrier
    certificate = carrier.certification
    if (
        certificate is None
        or not certificate.passed
        or certificate.fidelity is None
        or not isinstance(certificate.coverage, phx.geometry.ApproximateSourceCoverageCertificate)
        or not isinstance(certificate.request.domain, phx.geometry.CadSourceDomain)
        or certificate.embedding is None
        or any(
            not isinstance(element, FiniteElementSpec) or element.degree != 2
            for element in carrier.geometry.elements
        )
    ):
        raise ValueError("cad.checkpoint must retain certified P2 source fidelity, coverage and embedding")
    return {
        "source_binding_id": source.binding_id,
        "specification_id": request.specification_id,
        "options_id": options.options_id,
        "source_request_fingerprint": array_tree_fingerprint((source, request, options)),
        "result": _result_identity(carrier),
        "certification_id": certificate.report_id,
        "accepted_epoch_ids": [epoch.epoch_id for epoch in history],
        "accepted_carrier_fingerprint": array_tree_fingerprint((carrier.mesh, carrier.geometry)),
        "accepted_field_fingerprint": array_tree_fingerprint(history[-1].fields),
    }


def cold_worker(packet: ColdControlPacket) -> ColdWorkerReport:
    """Earn restoration and physical continuation under one fresh cold request."""
    packet.deadline.remaining()
    started = perf_counter()
    args = QualificationControls.from_record(packet.controls["controls"])
    archive_limits = qualification_archive_limits(args)
    if packet.controls["archive_limits"] != archive_limit_values(archive_limits):
        raise ValueError("cad.cold archive limits must equal the producer's declared policy")
    controls = read_meshing_source_execution_controls(
        packet.archive_path,
        expected_content_id=packet.content_id,
        limits=archive_limits,
    )
    control_read_seconds = perf_counter() - started
    with meshing_source_owner_reconstruction_scope(archive_limits=archive_limits):
        continued = execute_native_request(
            controls.execution_evidence,
            controls.limits,
            lambda budget: _cad_continue_restored(packet, budget, archive_limits=archive_limits),
            target=lambda value: value[0],
            failure_factory=qualification_failure,
            workflow_remaining_seconds=packet.deadline.remaining(),
            observe_managed_admission=True,
        )
    _, report = continued.value
    receipt = continued.result.execution_evidence
    if receipt is None:
        raise RuntimeError("CAD cold continuation must retain its complete request receipt")
    return ColdWorkerReport(
        record={
            **report.record,
            "cold_request_result": _result_identity(continued.result),
            "cold_request": {
                "history_owner_id": controls.execution_evidence.owner_id,
                "owner_id": receipt.owner_id,
                "work_units": int(np.asarray(receipt.total_work_units)),
                "geometry_queries": int(np.asarray(receipt.total_geometry_queries)),
                "receipt_fingerprint": array_tree_fingerprint(receipt),
                "limits_id": controls.limits.limits_id,
            },
        },
        read_seconds=control_read_seconds + report.read_seconds,
    )


def _cad_continue_restored(
    packet: ColdControlPacket, budget: CoordinateEnclosureBudget,
    *, archive_limits: ArrayArchiveLimits,
) -> tuple[phx.meshing.CellMeshingResult, ColdWorkerReport]:
    """Restore the authenticated physical epoch and solve from its accepted arrays."""
    packet.deadline.remaining()
    started = perf_counter()
    records = read_meshing_source_closure(
        packet.archive_path,
        expected_content_id=packet.content_id,
        limits=archive_limits,
        budget=budget,
    )
    identity = _cad_checkpoint_identity(records)
    controls = packet.controls
    if canonical_fingerprint(identity) != controls["expected_identity_id"]:
        raise ValueError("cad.cold archive must match the producer's accepted source and field identities")
    epoch = validate_meshing_accepted_epochs(records)[-1]
    carrier = epoch.carrier
    fields = epoch.fields
    read_seconds = perf_counter() - started
    packet.deadline.remaining()
    certificate = carrier.certification
    if certificate is None:
        raise ValueError("cad.cold must retain the accepted P2 certification report")
    renewed = recertify_restored_meshing_source(
        certificate.request,
        carrier.mesh,
        carrier.geometry,
        carrier.audit,
        archive_limits=archive_limits,
        budget=budget,
    )
    renewed.require_passed()
    original, renewed = require_same_source_evidence(
        certificate,
        renewed,
        carrier.mesh,
        carrier.geometry,
        carrier.audit,
        limits=archive_limits,
        budget=budget,
    )
    declaration = epoch.declarations[0]
    if not isinstance(declaration, MeshingFiniteElementFieldDeclaration):
        raise TypeError("cad.cold must retain the scalar finite-element declaration")
    length = controls["manufactured_length"]
    space, problem, free = _cad_prepare_p2_solver(carrier, length, declaration, budget=budget)
    mask = np.ones(space.dof_maps[0].global_dof_count, dtype=np.bool_)
    mask[np.asarray(free, dtype=np.int64)] = False
    if (
        problem.compilation_id != controls["compilation_id"]
        or not np.array_equal(mask, np.asarray(fields["boundary-mask"]))
        or array_tree_fingerprint(problem.lift)
        != array_tree_fingerprint(fields["boundary-values"])
    ):
        raise ValueError("cad.cold must reproduce the exact archived PDE owner, boundary mask and lift")
    seed = jnp.asarray(fields["u"])[free]
    seed_fingerprint = array_tree_fingerprint(seed)
    if seed_fingerprint != controls["seed_fingerprint"]:
        raise ValueError("cad.cold continuation must start from restored accepted coefficients")
    packet.deadline.remaining()
    values, compiler, timings = _compiled_solver_stages(problem, seed, controls["repeats"])
    error = _cad_p2_error(
        carrier.mesh, carrier.geometry, space, values, np.flatnonzero(mask), length
    )
    if (
        error["relative_l2_error"] != controls["warm_relative_l2_error"]
        or array_tree_fingerprint(values) != controls["warm_solution_fingerprint"]
    ):
        raise qualification_failure(
            "cad.cold",
            "Independent accepted-state continuation must reproduce the warm coefficients and physical error.",
            {"error": error, "warm_relative_l2_error": controls["warm_relative_l2_error"]},
        )
    packet.deadline.remaining()
    return carrier, ColdWorkerReport(
        record={
            "passed": True,
            **identity,
            "accepted_fields_bitwise_preserved": True,
            "prepared_fe_caches_serialized": False,
            "prepared_field_id": space.prepared_id,
            "compilation_id": problem.compilation_id,
            "original_certification_id": original.report_id,
            "renewed_certification_id": renewed.report_id,
            "source_certification_scientifically_preserved": True,
            "continuation_seed_fingerprint": seed_fingerprint,
            "continued_solution_fingerprint": array_tree_fingerprint(values),
            "warm_cold_continuation_bitwise_equal": True,
            "continued_from_epoch_id": epoch.epoch_id,
            "physical_error": error,
            "target_error_reached": error["relative_l2_error"] <= controls["target_error"],
            "compiler": compiler,
            "stages_seconds": timings,
            "peak_process_resident_bytes": peak_resident_bytes(),
        },
        read_seconds=read_seconds,
    )


def _cad_p2_checkpoint(
    published: _CadVolume,
    accepted: MeshingAcceptedEpoch,
    continued: Any,
    solve: dict[str, Any],
    case: _CadCase,
    args: QualificationControls,
    directory: Path,
    ledger: StageLedger,
) -> dict[str, Any]:
    """Publish one canonical epoch and independently continue within the same expiry."""
    deadline = ledger.deadline
    if deadline is None:
        raise ValueError("cad.checkpoint must use the original whole-workflow deadline")
    carrier = accepted.carrier
    certificate = carrier.certification
    if certificate is None:
        raise ValueError("cad.checkpoint must retain the accepted P2 report")
    records = {
        "certification_inputs": certificate.request,
        "report": certificate,
        "associations": carrier.associations,
        "generation_source": published.generation_source,
        "generation_specification": published.generation_specification,
        "generation_options": published.generation_options,
        "accepted_target": carrier,
        "accepted_data": {"accepted_epochs": (accepted,)},
    }
    expected = _cad_checkpoint_identity(records)
    archive_limits = qualification_archive_limits(args)

    def prepare_and_write(
        budget: CoordinateEnclosureBudget,
    ) -> MeshingSourceClosureReceipt:
        deadline.remaining()
        request = certificate.request
        coverage = certificate.coverage
        if (
            certificate.embedding is None
            or certificate.fidelity is None
            or not isinstance(
                coverage, phx.geometry.ApproximateSourceCoverageCertificate
            )
            or coverage.fidelity.cad_closure is None
        ):
            raise ValueError(
                "cad.checkpoint must retain current validity, embedding, approximate coverage and issued CAD closure owners"
            )
        if (
            not isinstance(request.domain, phx.geometry.CadSourceDomain)
            or request.cell_regions is None
            or not isinstance(request.source, phx.geometry.MeshingDomainBoundarySource)
            or request.fidelity_tolerance is None
            or request.junction_vertices is not None
        ):
            raise ValueError(
                "cad.checkpoint prepared evidence must retain the complete original CAD certification request"
            )
        prepared = MeshCertificationPreparedEvidence(
            carrier.mesh,
            carrier.geometry,
            carrier.audit.validity,
            schedule=request.schedule,
            domain=request.domain,
            cell_regions=request.cell_regions,
            embedding=certificate.embedding,
            coverage=coverage,
            limits=request.limits,
            source=request.source,
            fidelity_tolerance=request.fidelity_tolerance,
            fidelity_sample_order=request.fidelity_sample_order,
            scoped_fidelity=request.scoped_fidelity,
            budget=budget,
        )
        renewed = recertify_restored_meshing_source(
            request,
            carrier.mesh,
            carrier.geometry,
            carrier.audit,
            archive_limits=archive_limits,
            prepared=prepared,
            budget=budget,
        )
        require_same_source_evidence(
            certificate,
            renewed,
            carrier.mesh,
            carrier.geometry,
            carrier.audit,
            limits=archive_limits,
            budget=budget,
        )
        deadline.remaining()
        receipt = write_meshing_source_closure(
            directory / "cad.physical.checkpoint.zip",
            records,
            limits=archive_limits,
        )
        deadline.remaining()
        return receipt

    def retain_carrier(
        _: MeshingSourceClosureReceipt,
    ) -> phx.meshing.CellMeshingResult:
        return carrier

    with (
        meshing_source_owner_reconstruction_scope(archive_limits=archive_limits),
        prepare_cold_worker(
            "tools.meshing_qualification.scenarios.cad", deadline=deadline, ledger=ledger
        ) as worker,
    ):
        consumed = ledger.run(
            "physical_accepted_epoch_checkpoint",
            lambda: execute_native_consumer(
                carrier,
                published.generation_specification.limits,
                prepare_and_write,
                failure_factory=qualification_failure,
                target=retain_carrier,
                observe_managed_admission=True,
            ),
        )
        receipt = consumed.value
        deadline.remaining()
        warm_execution = consumed.result.execution_evidence
        if warm_execution is None:
            raise RuntimeError("CAD warm archive request must retain its native receipt")
        warm_execution.require_valid()
        warm_request = {
            "owner_id": warm_execution.owner_id,
            "work_units": int(np.asarray(warm_execution.total_work_units)),
            "geometry_queries": int(np.asarray(warm_execution.total_geometry_queries)),
            "receipt_fingerprint": array_tree_fingerprint(warm_execution),
            "limits_id": published.generation_specification.limits.limits_id,
            "execution_evidence": warm_execution.to_record(),
        }
        try:
            restored = ledger.run(
                "fresh_process_accepted_P2_continuation",
                lambda: worker.execute(
                    receipt.path,
                    receipt.content_id,
                    {
                        # Bind the complete static/numerical identity without
                        # copying a potentially multi-megabyte tree into IPC.
                        "expected_identity_id": canonical_fingerprint(expected),
                        "controls": args.to_record(),
                        "archive_limits": archive_limit_values(archive_limits),
                        "manufactured_length": case.manufactured_length,
                        "compilation_id": solve["compilation_id"],
                        "seed_fingerprint": solve["continuation"]["seed_fingerprint"],
                        "warm_solution_fingerprint": array_tree_fingerprint(continued),
                        "warm_relative_l2_error": solve["continuation"]["relative_l2_error"],
                        "target_error": args.target_error,
                        "repeats": args.repeats,
                    },
                ),
            )
        except ColdWorkerTimeout as error:
            raise phx.meshing.MeshingFailure(
                phx.meshing.MeshingFailureCategory.TIMED_OUT,
                "CAD accepted-state cold continuation exhausted the original wall allowance.",
                stage="fresh_process_accepted_P2_continuation",
            ) from error
    if (
        restored.get("passed") is not True
        or restored.get("process_id") == os.getpid()
        or restored.get("accepted_field_fingerprint") != expected["accepted_field_fingerprint"]
        or restored.get("continued_solution_fingerprint") != array_tree_fingerprint(continued)
    ):
        raise ValueError("cad.checkpoint must retain independent accepted-state continuation evidence")
    if restored["peak_process_resident_bytes"] > args.maximum_memory_bytes:
        raise phx.meshing.MeshingFailure(
            phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "CAD cold continuation exceeded the original process-memory allowance.",
            stage="fresh_process_accepted_P2_continuation",
        )
    with receipt.path.open("rb") as stream:
        container_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "status": "passed",
        "content_id": receipt.content_id,
        "container_sha256": container_sha256,
        "container_bytes": receipt.path.stat().st_size,
        "source_authority_stripped": False,
        "prepared_fe_caches_serialized": False,
        "accepted_fields_bitwise_preserved": True,
        "field_names": sorted(accepted.fields),
        "archive_limits": archive_limit_values(archive_limits),
        "requested_maximum_recipe_bytes": args.maximum_recipe_bytes,
        "accepted_epoch_ids": expected["accepted_epoch_ids"],
        "warm_archive_request": warm_request,
        "fresh_interpreter": restored,
        "cold_process": ledger.record["cold_process"],
    }


_CAD_CASE_PLANS: dict[str, tuple[str, tuple[int, ...]]] = {
    "quadric-routine": ("authored", (1,)),
    "quadric-larger": ("authored", (1, 2)),
    "external-routine": ("external", (1,)),
    "external-combined": ("external-combined", (1,)),
}


def _cad_case_resolution(
    args: QualificationControls,
    kind: str,
    scale: int,
    directory: Path,
    record: dict[str, Any],
    ledger: StageLedger,
    record_phase: Callable[[phx.meshing.NativeMeshingPhaseMeasurement], None],
) -> dict[str, object]:
    """One complete source → Boolean → archive → interchange → mesh → P2 → solve pass."""
    stages = record["stages_seconds"]
    section: dict[str, Any] = {"scale": scale}

    def stage(name: str, function: Callable[[], Any]) -> Any:
        value, timing = ledger.run(f"{name}_x{scale}", lambda: _measured(function))
        stages[f"{name}_x{scale}"] = timing["wall_seconds"]
        return value

    # One request ledger owns every source codec, archive and exact proof.
    # Changing format does not reset the allowance or its earned proof bank.
    limits = _cad_native_limits(args)
    remaining = args.timeout if ledger.deadline is None else ledger.deadline.remaining()
    with (
        NativeParallelExecution(None, remaining),
        NativeExecutionScope(limits.native_limits()) as source_scope,
    ):
        with source_scope.batch(
            limits.maximum_work_units,
            queries_upper=limits.maximum_geometry_queries,
            bytes_upper=limits.maximum_scratch_bytes,
            cavity_upper=limits.maximum_cavity_cells,
        ) as source_ticket:
            source_budget = CoordinateEnclosureBudget(
                source_ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            with source_ticket.native_target(), source_budget.activate():
                source_ticket.checkpoint()
                case = stage(
                    "source_import_boolean",
                    lambda budget=source_budget: _cad_authored_case(
                        args, directory, section, scale, coordinate_budget=budget
                    )
                    if kind == "authored"
                    else _cad_external_case(
                        args, directory, section, scale,
                        combined=kind == "external-combined",
                        coordinate_budget=budget,
                    ),
                )
                section["size"], section["deviation"] = case.size, case.deviation
                section["intersection"] = stage(
                    "intersection_evidence",
                    lambda budget=source_budget: _cad_intersection_evidence(
                        case.model, case.loops, coordinate_budget=budget
                    ),
                )
                section["query"] = stage(
                    "native_query",
                    lambda budget=source_budget: _cad_query(
                        case.model, case.reference_volume, coordinate_budget=budget
                    ),
                )
                restored, section["persistence"] = stage(
                    "native_persistence",
                    lambda budget=source_budget: _cad_persistence(
                        case.model, directory, coordinate_budget=budget
                    ),
                )
                section["interchange_export"] = stage(
                    "interchange_boundaries",
                    lambda budget=source_budget: _cad_interchange_boundaries(
                        restored, case, args.capacity, directory,
                        coordinate_budget=budget,
                    ),
                )
                source_ticket.checkpoint()
    # Request-owned proof banks do not outlive their admitted source ticket.
    del source_budget
    section["source_execution"] = NativeExecutionRecord(
        source_scope.completed_evidence(),
        owner_id=canonical_fingerprint(
            {
                "kind": "native-cad-source-stage",
                "cad_case": args.cad_case,
                "format": args.cad_format,
                "scale": scale,
                "limits": limits.limits_id,
                "model": restored.model_id,
                "source_digests": [entry["source_digest"] for entry in section["imports"]],
            }
        ),
    ).to_record()
    published = stage(
        "native_surface_volume",
        lambda: _cad_volume_mesh(restored, case, args, record_phase),
    )
    section["volume_mesh"] = stage(
        "volume_acceptance", lambda: _cad_volume_acceptance(published, restored, case)
    )
    carrier, section["curving"] = stage(
        "p2_curving", lambda: _cad_curving(published, restored, case)
    )
    deadline = ledger.deadline
    if deadline is None:
        raise ValueError("CAD physical solve must retain the original workflow deadline")
    consumed = stage(
        "p2_solve",
        lambda: execute_native_consumer(
            carrier, published.generation_specification.limits,
            lambda budget: _cad_p2_solve(
                carrier, case, args, deadline=deadline, budget=budget
            ),
            target=lambda value: value[0].carrier,
            failure_factory=qualification_failure,
            observe_managed_admission=True,
        ),
    )
    accepted, continued, section["solve"] = consumed.value
    accepted = rebind_meshing_accepted_epoch_execution(
        accepted, consumed.result, transition=None
    )
    section["physical_checkpoint"] = stage(
        "accepted_P2_archive_and_cold_continuation",
        lambda: _cad_p2_checkpoint(
            published, accepted, continued, section["solve"], case, args, directory, ledger
        ),
    )
    straight_defect = section["volume_mesh"]["affine_relative_volume_defect"]
    curved_defect = section["solve"]["curved_relative_volume_defect"]
    section["geometry_volume_comparison"] = {
        "straight_relative_defect": straight_defect,
        "curved_relative_defect": curved_defect,
        "role": "supplementary; continuous fidelity is the curving certificate",
    }
    return section


def _cad_curved_lifecycle(args: QualificationControls) -> dict[str, object]:
    """Curved/external native CAD workflow with independent analytic oracles."""
    requirements = _cad_qualification_requirements()
    requirement_nonclaims = requirements["nonclaims"]
    if not isinstance(requirement_nonclaims, list):
        raise TypeError("CAD requirements.nonclaims must contain a list of strings.")
    nonclaims: list[str] = []
    for value in requirement_nonclaims:
        if not isinstance(value, str):
            raise TypeError("CAD requirements.nonclaims must contain strings.")
        nonclaims.append(value)
    if args.cad_case not in _CAD_CASE_PLANS:
        raise ValueError(
            f"_cad_curved_lifecycle.cad_case must be one of {tuple(_CAD_CASE_PLANS)!r}; got {args.cad_case!r}."
        )
    if args.resolution < 1 or args.capacity < 1 or args.repeats < 1:
        raise ValueError(
            f"_cad_curved_lifecycle.bounds must have positive resolution, capacity and repeats; got {(args.resolution, args.capacity, args.repeats)!r}."
        )
    if (
        not np.isfinite(args.timeout)
        or args.timeout <= 0.0
        or not np.isfinite(args.target_error)
        or args.target_error <= 0.0
    ):
        raise ValueError(
            f"_cad_curved_lifecycle.bounds must have finite positive timeout and target error; got {(args.timeout, args.target_error)!r}."
        )
    kind, scales = _CAD_CASE_PLANS[args.cad_case]
    started = perf_counter()
    deadline = Deadline(started + args.timeout)
    record: dict[str, Any] = {
        "scenario": "native-cad-lifecycle",
        "cad_case": args.cad_case,
        "status": "failed",
        "runtime": _runtime_identity(),
        "requirements": requirements,
        "format": args.cad_format,
        "stages_seconds": {},
        "native_phases": [],
        "resolutions": [],
        "targets": {
            "relative_l2_error": args.target_error,
            "wall_seconds": args.timeout,
            "minimum_observed_order": 2.5 if len(scales) > 1 else None,
        },
        "control_capacities": {
            "resolution": args.resolution,
            "entity_capacity": args.capacity,
            "scales": list(scales),
        },
        "time_to_target_error_seconds": None,
        "positive_prerequisite": None,
        "mandatory_workflow_completion": "incomplete",
        "case_contract_completion": "incomplete",
        "independent_scientific_certification": "unassessed",
        "full_curved_multiformat_gate": "incomplete",
        "release_evidence": "unassessed",
        "leadership_evidence": "unassessed",
        "nonclaims": [
            *nonclaims,
            "Volume-error comparison is supplementary; continuous fidelity is the curving certificate.",
            "Multi-solid curved coverage is not certified by this case.",
            "One record covers one interchange format; the multiformat gate aggregates three records.",
        ],
        "optional_external_baseline": "not-requested",
        "pde": "-Delta u = u / L^2, u = exp(x/L) sin(y/L) cos(z/L), P2 on certified P2 CAD geometry",
        "active_phase": "source_format_preparation",
    }
    ledger = StageLedger(record, deadline=deadline)

    def record_phase(measurement: phx.meshing.NativeMeshingPhaseMeasurement) -> None:
        record["native_phases"].append(
            {
                "phase": measurement.phase,
                "elapsed_seconds": measurement.elapsed_seconds,
                "work_units": measurement.work_units,
                "invocations": measurement.invocations,
            }
        )

    errors: list[float] = []
    try:
        with (
            NativeParallelExecution(None, deadline.remaining()),
            TemporaryDirectory(prefix="phydrax-native-cad-") as temporary,
        ):
            for scale in scales:
                directory = Path(temporary) / f"x{scale}"
                directory.mkdir()
                section = _cad_case_resolution(
                    args, kind, scale, directory, record, ledger, record_phase
                )
                record["resolutions"].append(section)
                solve = section["solve"]
                if not isinstance(solve, dict):
                    raise TypeError("CAD resolution.solve must contain a result mapping.")
                checkpoint = section["physical_checkpoint"]
                if not isinstance(checkpoint, dict):
                    raise TypeError("CAD resolution.physical_checkpoint must contain its cold evidence mapping.")
                relative_error = max(
                    solve["relative_l2_error"],
                    solve["continuation"]["relative_l2_error"],
                    checkpoint["fresh_interpreter"]["physical_error"]["relative_l2_error"],
                )
                if not isinstance(relative_error, float):
                    raise TypeError("CAD solve.relative_l2_error must be a float.")
                errors.append(relative_error)
                if (
                    relative_error <= args.target_error
                    and record["time_to_target_error_seconds"] is None
                ):
                    record["time_to_target_error_seconds"] = perf_counter() - started
        record["physical_error"] = errors[-1]
        record["target_error_reached"] = errors[-1] <= args.target_error
        order = None
        if len(errors) > 1:
            order = math.log(errors[0] / errors[1]) / math.log(scales[1] / scales[0])
        record["observed_order"] = order
        converged = order is None or order >= 2.5
        record["status"] = (
            "passed" if record["target_error_reached"] and converged else "failed"
        )
        if record["status"] == "passed":
            # Every section raised before here unless its independent check passed.
            record["case_contract_completion"] = "complete"
            record["independent_scientific_certification"] = "passed"
            record["combined_case_contract"] = {
                "case": args.cad_case,
                "format": args.cad_format,
                "external_corpus": record["resolutions"][-1]["source_fixture"][
                    "external_corpus"
                ],
                "nonrational_boolean": bool(
                    record["resolutions"][-1]["intersection"]["branches"]
                ),
                "passed": True,
            }
    except phx.meshing.MeshingFailure as failure:
        record_failure(record, failure, record["active_phase"])
        record["positive_prerequisite"] = {
            "phase": record["active_phase"],
            "detail": str(failure),
            "category": failure.category.value,
            "safe_refusal_is_positive": False,
        }
    except (RuntimeError, ValueError, TypeError, TimeoutError) as failure:
        record_failure(record, failure, record["active_phase"])
        record["positive_prerequisite"] = {
            "phase": record["active_phase"],
            "detail": str(failure),
            "safe_refusal_is_positive": False,
        }
    finally:
        _cad_qualification_completion(args, record, started)
    return record


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    if args.cad_case == "planar-box":
        return _cad_qualification_lifecycle(args)
    return _cad_curved_lifecycle(args)
