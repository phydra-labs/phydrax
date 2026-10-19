# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import math
import resource
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, final

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax import model_array_bytes
from tools._meshing_cases import load_cad_fixture
from tools.meshing_qualification._controls import (
    _cad_qualification_requirements,
    capacity_meshing_limits,
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
    Deadline,
    peak_resident_bytes,
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
        maximum_occurrences=capacity,
        pcurve_fit=None,
    )
    export_policy = phx.interchange.CadExportPolicy(
        maximum_bytes=byte_capacity, maximum_entities=64 * capacity
    )
    match format_name:
        case "step":
            writer, suffix = phx.interchange.write_step, ".step"
        case "iges":
            writer, suffix = phx.interchange.write_iges, ".iges"
        case "brep":
            writer, suffix = phx.interchange.write_brep_text, ".brep"
        case _:
            raise ValueError(
                f"_cad_qualification_import.format_name must be 'step', 'iges' or 'brep'; got {format_name!r}."
            )
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
) -> tuple[Any, Any, Any]:

    space = phx.discretization.FiniteElementPlan(
        result.mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
        coordinate_spec=result.geometry,
    ).prepare()
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
        form, space, constraint=constraint, dirichlet_values=_manufactured_value
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
        solver, timing = _ledger.run(
            f"solver_prepare_epoch_{epoch}",
            lambda: _measured(lambda: _cad_qualification_solver(result)),
        )
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
                lambda: phx.meshing.prepare_mesh_adaptation(result, marked, policy=policy)
            ),
        )
        stages[record["active_phase"]] = timing["wall_seconds"]

        adaptation, timing = _ledger.run(
            f"adaptation_execute_epoch_{epoch}",
            lambda: _measured(lambda: phx.meshing.execute_mesh_adaptation(prepared)),
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
                model, args.resolution, args.capacity, args.timeout
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
    if record["status"] != "passed":
        record["failure_phase"] = record["active_phase"]
    if record["end_to_end_seconds"] > args.timeout:
        record["status"] = "timeout"
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


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return _cad_qualification_lifecycle(args)
