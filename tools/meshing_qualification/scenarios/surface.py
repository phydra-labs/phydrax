# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import json
import os
import resource
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, final, TypedDict

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from examples.native_surface_meshing import sphere_error, zero_mean_constraint
from phydrax import array_tree_fingerprint, ArrayArchiveLimits
from phydrax.geometry import PreparedSurfaceSourceSupport
from phydrax.lifecycle import (
    CompositionRebind,
    prepare_meshing_field_owner,
    read_meshing_source_closure,
    recertify_restored_meshing_source,
    write_meshing_source_closure,
)
from phydrax.meshing import (
    NativeMeshingPhaseMeasurement,
    qualification as originals,
    SurfaceAssociationTransfer,
)
from tools._meshing_cases import prepare_sphere_case
from tools.meshing_qualification._controls import (
    capacity_meshing_limits,
    QualificationControls,
)
from tools.meshing_qualification._physics import (
    _compiled_solver_stages,
    CorpusCompilerEvidence,
)
from tools.meshing_qualification._records import (
    _phase_recordings,
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
    run_cold_worker,
    StageLedger,
)
from tools.meshing_qualification.scenarios.native_lifecycle import _corpus_native_consumer


def _surface_forcing(points: Any, context: Any) -> Any:
    del context
    return 2.0 * points[..., 0] / jnp.linalg.norm(points, axis=-1)


def _prepare_surface_solver(
    result: phx.meshing.CellMeshingResult,
) -> tuple[Any, Any, Any]:

    space = phx.discretization.FiniteElementPlan(
        result.mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
        coordinate_spec=result.geometry,
    ).prepare()
    constraint, weights = zero_mean_constraint(space)
    form = phx.equations.FiniteElementForm(
        "sphere-laplace-beltrami",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    _surface_forcing, coefficient_id="unit-sphere-degree-one-harmonic"
                ),
            ),
        ),
        # Unit diffusion is SPD on the connected mesh's positive-mass mean-zero subspace.
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form, space, constraint=constraint
    )
    return space, problem, weights


class _SurfaceLaw(TypedDict):
    name: str
    area: float
    curvature: float
    closed: bool
    euler: int


def _surface_qualification_prepare(
    args: QualificationControls,
) -> tuple[Any, Any, Any, _SurfaceLaw]:

    name = args.surface_case
    if not isinstance(name, str):
        raise TypeError(
            f"qualification.surface_case must be a registered profile name string; got {type(name).__name__}"
        )
    if name == "sphere":
        source, request, options = prepare_sphere_case(
            args.resolution, args.capacity, args.timeout
        )
        law = _SurfaceLaw(
            name=name, area=float(4.0 * np.pi), curvature=1.0, closed=True, euler=2
        )
        return source, request, options, law
    factories = {
        "torus": (originals.torus_surface, 0.45, 1.0 / 0.6),
        "capped_cylinder": (originals.capped_cylinder_surface, 0.3, 1.0 / 0.8),
        "annulus": (originals.annulus_surface, 0.2, 1.0 / 0.4),
        "cylinder_sheet": (originals.cylinder_sheet_surface, 0.25, 1.0),
        "folded_plates": (originals.folded_plates_surface, 0.3, 0.0),
    }
    factory, size, curvature = factories[name]
    original = factory()
    domain = original.domain
    source = phx.meshing.NativeSurfaceSource(domain)
    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray(domain.scope_indices(2), dtype=np.int64),
    )
    limits = capacity_meshing_limits(
        args.capacity,
        args.timeout,
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
    options = phx.meshing.NativeMeshingOptions("parametric_surface")
    request = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, size, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        limits=limits,
    )
    euler = original.euler_characteristic
    if euler is None:
        raise ValueError(
            f"surface.euler_characteristic must provide an Euler reference; got {euler!r}"
        )
    law = _SurfaceLaw(
        name=name,
        area=original.area,
        curvature=curvature,
        closed=original.closed,
        euler=euler,
    )
    return source, request, options, law


def _surface_qualification_value(points: Any, name: str) -> Any:
    if name == "sphere":
        return points[..., 0] / jnp.linalg.norm(points, axis=-1)
    if name in ("torus", "cylinder_sheet"):
        return points[..., 2]
    if name == "capped_cylinder":
        z = points[..., 2]
        return z * z * (1.5 - z) ** 2
    if name == "annulus":
        return points[..., 0]
    if name == "folded_plates":
        return points[..., 1]
    raise ValueError(
        f"surface.profile must be sphere, torus, cylinder_sheet, capped_cylinder, annulus, or folded_plates; got {name!r}"
    )


def _surface_qualification_solver(
    result: Any, law: _SurfaceLaw
) -> tuple[Any, Any, np.ndarray, Any]:
    name = str(law["name"])
    if name == "sphere":
        space, problem, weights = _prepare_surface_solver(result)
        free = np.flatnonzero(np.arange(weights.size) != int(np.argmax(weights)))
        return space, problem, free, weights
    space = phx.discretization.FiniteElementPlan(
        result.mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
        coordinate_spec=result.geometry,
    ).prepare()

    def exact(points: Any) -> Any:
        return _surface_qualification_value(points, name)

    def forcing(points: Any, context: Any) -> Any:
        del context
        if name == "torus":
            rho = jnp.linalg.norm(points[..., :2], axis=-1)
            z = points[..., 2]
            return z * (1.0 + (2.0 * rho - 2.0) / (0.6**2 * rho))
        if name == "capped_cylinder":
            z = points[..., 2]
            second = 2.0 * 1.5**2 - 12.0 * 1.5 * z + 12.0 * z * z
            # Caps carry u=0. The lateral quartic has zero conormal derivative
            # at each authored rim, so no unrepresented interface load is used.
            return jnp.where((z == 0.0) | (z == 1.5), 0.0, exact(points) - second)
        return jnp.zeros(points.shape[:-1], dtype=points.dtype)

    closed = bool(law["closed"])
    actions = [phx.equations.DiffusionAction("u", 1.0)]
    if closed:
        actions.append(phx.equations.MassAction("u", 1.0))
    actions.append(
        phx.equations.SourceAction(
            "u",
            phx.equations.coefficient(
                forcing, coefficient_id=f"original-{name}-manufactured-source"
            ),
        )
    )
    form = phx.equations.FiniteElementForm(
        f"original-{name}-physical-surface", "u", tuple(actions)
    )
    if closed:
        problem = phx.equations.compile_finite_element_problem(form, space)
        free = np.arange(space.field_spaces[0].layout.size)
    else:
        constraint = phx.discretization.dirichlet_constraint(space, "u")
        problem = phx.equations.compile_finite_element_problem(
            form, space, constraint=constraint, dirichlet_values=exact
        )
        free = np.asarray(constraint.free_dofs)
    return space, problem, free, None


def _surface_qualification_error(space: Any, values: Any, law: _SurfaceLaw) -> float:
    if law["name"] == "sphere":
        return sphere_error(space, values)
    rule = phx.integration.ReferenceTriangleRule(phx.integration.GaussLegendreRule(5))
    data = phx.integration.reference_rule_data(rule)
    total = 0.0
    for index, _ in enumerate(space.mesh.blocks):
        geometry = space.evaluate_block_geometry(
            "u", index, space.default_runtime.coordinates, data.points, data.weights
        )
        routes = np.asarray(space.dof_maps[0].cell_dofs[index])
        approximate = np.einsum(
            "ql,cl->cq", np.asarray(geometry.basis_values), np.asarray(values)[routes]
        )
        exact = np.asarray(
            _surface_qualification_value(geometry.physical_points, str(law["name"]))
        )
        total += float(
            np.sum(np.asarray(geometry.physical_weights) * (approximate - exact) ** 2)
        )
    return float(np.sqrt(total))


def _surface_qualification_content(space: Any, name: str, values: Any) -> float:
    index = next(
        i for i, field in enumerate(space.construction_plan.fields) if field.name == name
    )
    total = 0.0
    for block, geometry in enumerate(
        space.evaluate_geometry(name, space.default_runtime.coordinates)
    ):
        routes = np.asarray(space.dof_maps[index].cell_dofs[block])
        sampled = np.einsum(
            "ql,cl->cq", np.asarray(geometry.basis_values), np.asarray(values)[routes]
        )
        total += float(np.sum(sampled * np.asarray(geometry.physical_weights)))
    return total


def _surface_qualification_vector_norm(space: Any, name: str, values: Any) -> float:
    data = phx.integration.reference_rule_data(
        phx.integration.ReferenceTriangleRule(phx.integration.GaussLegendreRule(5))
    )
    total = 0.0
    for index, block in enumerate(space.mesh.blocks):
        geometry = space.evaluate_block_geometry(
            name, index, space.default_runtime.coordinates, data.points, data.weights
        )
        physical = np.asarray(space.reconstruct(name, values, block.name, data.points))
        total += float(
            np.sum(
                np.asarray(geometry.physical_weights)
                * np.sum(physical * physical, axis=-1)
            )
        )
    return float(np.sqrt(total))


def _surface_qualification_curl_seed(points: Any, context: Any) -> Any:
    del context
    return jnp.stack(
        (
            1.0 + 0.2 * points[..., 0],
            -0.4 + 0.1 * points[..., 1],
            0.3 + 0.15 * points[..., 2],
        ),
        axis=-1,
    )


def _surface_qualification_flux_seed(points: Any, context: Any) -> Any:
    del context
    return jnp.stack(
        (
            0.7 + 0.1 * points[..., 1],
            0.8 + 0.2 * points[..., 2],
            -0.2 + 0.05 * points[..., 0],
        ),
        axis=-1,
    )


def _surface_qualification_initial_fields(result: Any, scalar: Any) -> tuple[Any, ...]:
    space = phx.meshing.qualification.prepare_surface_fields(result)
    return (
        scalar,
        jnp.full((space.field_spaces[1].layout.size,), 2.0, dtype=scalar.dtype),
        space.dof_maps[2].dof_coordinates[:, 0] + 0.5,
        space.project("curl", _surface_qualification_curl_seed),
        space.project("flux", _surface_qualification_flux_seed),
    )


def _surface_qualification_material_initial(
    result: Any,
) -> phx.equations.MaterialTransaction:
    """Initialize nonzero cell history on the accepted physical FV owner."""
    owner = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
        result.mesh,
        field_name="surface-history",
        component_names=("surface-history",),
    ).prepare(cell_geometry=result.geometry)
    history = (1.0 + owner.cell_centers[:, 0] ** 2)[:, None]
    return phx.equations.MaterialTransaction(
        (
            phx.equations.MaterialState(
                phx.equations.MaterialSiteId("surface-history"),
                "surface-history",
                history,
            ),
        )
    )


def _surface_qualification_material_transport(
    sourceResult: Any,
    adaptation: Any,
    materials: phx.equations.MaterialTransaction,
    limits: Any,
    *,
    source_workspace: Any,
) -> tuple[phx.solver.MaterialTopologyTransferResult, dict[str, object]]:
    """Carry committed Kelvin history through the native chart remap."""

    source = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
        sourceResult.mesh,
        field_name="surface-history",
        component_names=("surface-history",),
    ).prepare(cell_geometry=sourceResult.geometry)
    target = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
        adaptation.target.mesh,
        field_name="surface-history",
        component_names=("surface-history",),
    ).prepare(cell_geometry=adaptation.target.geometry)
    source_workspace.retain_owner(
        (source, target, materials, materials.state("surface-history").committed)
    )
    remap = phx.discretization.prepare_unstructured_conservative_remap(
        source,
        target,
        provenance="surface-history",
        policy=phx.geometry.CommonRefinementPolicy(
            maximum_exact_work=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        ),
        source_geometry=sourceResult.geometry,
        target_geometry=adaptation.target.geometry,
        surface_chart_deformation=adaptation.geometry_transition.chart_deformation,
        adaptation=adaptation,
        source_workspace=source_workspace,
    )
    evidence = remap.evidence
    if (
        not remap.succeeded
        or remap.plan is None
        or not isinstance(
            evidence, phx.discretization.finite_volume.MappedSurfaceChartRemapEvidence
        )
        or not evidence.passed
    ):
        raise qualification_failure(
            "_surface_qualification_material_transport",
            (f"Surface material remap refused publication: {remap.reason}"),
            {"evidence": evidence, "remap": remap},
        )
    state = materials.state("surface-history")
    source_history = state.committed
    if source_history.shape != (source.cell_count, 1):
        raise ValueError(
            f"surface.material_history must have cell-by-one Kelvin shape {(source.cell_count, 1)}; got {source_history.shape}"
        )
    target_history = remap.plan.apply(source_history)
    defect = np.asarray(remap.plan.conservation_defect(source_history, target_history))
    source_terms = np.asarray(remap.plan.source_volumes)[:, None] * np.asarray(
        source_history
    )
    target_terms = np.asarray(remap.plan.target_volumes)[:, None] * np.asarray(
        target_history
    )
    source_inventory = np.sum(source_terms, axis=0)
    target_inventory = np.sum(target_terms, axis=0)
    coverage_error = float(
        np.asarray(remap.plan.report.total_source_coverage_error_bound)
    ) * np.max(np.abs(np.asarray(source_history)), axis=0)
    publication_roundoff = (
        64.0
        * np.finfo(np.asarray(target_history).dtype).eps
        * (np.sum(np.abs(source_terms), axis=0) + np.sum(np.abs(target_terms), axis=0))
    )
    allowance = coverage_error + publication_roundoff
    if not np.all(np.isfinite(defect)) or np.any(np.abs(defect) > allowance):
        raise qualification_failure(
            "_surface_qualification_material_transport",
            (
                "Surface material remap exceeds declared physical coverage and publication roundoff."
            ),
            {"allowance": allowance, "defect": defect},
        )
    carried = phx.equations.MaterialTransaction(
        (
            phx.equations.MaterialState(
                state.site_id,
                state.model_id,
                target_history,
                state_version=state.state_version + 1,
            ),
        )
    )
    return phx.solver.MaterialTopologyTransferResult(carried, (remap,)), {
        "site": "surface-history",
        "value_units": ["Kelvin"],
        "source_result_id": sourceResult.result_id,
        "source_owner_id": source.prepared_id,
        "target_owner_id": target.prepared_id,
        "source_material_state_id": state.state_id,
        "target_material_state_id": carried.state("surface-history").state_id,
        "remap_id": remap.remap_id,
        "plan_id": remap.plan.plan_id,
        "evidence_id": array_tree_fingerprint(evidence),
        "chart_deformation_id": array_tree_fingerprint(
            adaptation.geometry_transition.chart_deformation
        ),
        "source_physical_inventory": source_inventory.tolist(),
        "target_physical_inventory": target_inventory.tolist(),
        "conservation_defect": defect.tolist(),
        "source_coverage_error_bound": coverage_error.tolist(),
        "publication_roundoff_bound": publication_roundoff.tolist(),
        "passed": bool(
            remap.succeeded and evidence.passed and np.all(np.abs(defect) <= allowance)
        ),
    }


def _surface_qualification_acceptance(
    source: Any, result: Any, request: Any, law: _SurfaceLaw
) -> dict[str, object]:

    certification = result.certification
    if certification is None or certification.fidelity is None:
        raise qualification_failure(
            "_surface_qualification_acceptance",
            ("Surface publication lost its source fidelity graph."),
            {"certification": certification},
        )
    fidelity = certification.fidelity
    fidelity.binding.require(result.mesh, result.geometry)
    charts = result.surface_source
    if charts is None:
        raise qualification_failure(
            "_surface_qualification_acceptance",
            ("Surface publication lost its SCI-ordered native source chart binding."),
            {"charts": charts},
        )
    charts.require_bound(result.mesh, result.geometry, certification.request.source)
    if charts.domain.content_domain_id() != source.domain.content_domain_id():
        raise qualification_failure(
            "_surface_qualification_acceptance",
            ("Surface publication changed source coefficient/stratum/control banks."),
            {"charts": charts, "source": source},
        )
    if (fidelity.binding.source_id, fidelity.binding.source_revision) != (
        source.source_id,
        source.source_revision,
    ):
        raise qualification_failure(
            "_surface_qualification_acceptance",
            ("Surface publication source certification changed source identity."),
            {"fidelity": fidelity, "source": source},
        )
    coordinates = np.asarray(result.mesh.coordinates)
    cells = np.concatenate([np.asarray(block.vertices) for block in result.mesh.blocks])
    edges, uses = np.unique(
        np.sort(
            np.concatenate((cells[:, (0, 1)], cells[:, (1, 2)], cells[:, (2, 0)])), axis=1
        ),
        axis=0,
        return_counts=True,
    )
    euler = np.unique(cells).size - edges.shape[0] + cells.shape[0]
    space = phx.meshing.qualification.prepare_surface_fields(result)
    one = jnp.ones((space.field_spaces[0].layout.size,), dtype=coordinates.dtype)
    area = _surface_qualification_content(space, "u", one)
    size = request.size_controls[0].target_size
    bound = float(law["area"]) * max(
        0.25 * (float(law["curvature"]) * size) ** 2, 1.0e-12
    )
    passed = bool(
        result.audit.passed
        and result.compliance.passed
        and certification.passed
        and fidelity.status == "certified"
        and fidelity.chart_coverage is not None
        and fidelity.chart_coverage.complete
        and all(item.complete for item in result.associations)
        and np.all((uses == 1) | (uses == 2))
        and (not law["closed"] or np.all(uses == 2))
        and euler == law["euler"]
        and abs(area - float(law["area"])) <= bound
    )
    return {
        "passed": passed,
        "source_binding_id": source.binding_id,
        "original_source_chart_id": charts.source_chart_id,
        "original_source_mesh_topology_id": charts.mesh.topology_id,
        "original_source_boundary_scope_id": charts.boundary_source.source_scope_id,
        "fidelity_certificate_id": fidelity.certificate_id,
        "chart_coverage_id": None
        if fidelity.chart_coverage is None
        else fidelity.chart_coverage.cover_id,
        "physical_area": area,
        "independent_source_area": law["area"],
        "area_error_bound": bound,
        "euler_characteristic": euler,
        "complete_source_associations": all(
            item.complete for item in result.associations
        ),
        "result": _result_identity(result),
    }


def _surface_qualification_transition(
    source: Any,
    adaptation: Any,
    problem: Any,
    values: Any,
    auxiliary: tuple[Any, ...],
    epoch: int,
    law: _SurfaceLaw,
    materials: Any,
    limits: Any,
) -> tuple[Any, tuple[Any, ...], Any, dict[str, object]]:

    full = phx.meshing.qualification.prepare_surface_fields(source)
    fields = full.construction_plan.fields
    accepted = phx.solver.FiniteElementAcceptedState(
        (values, *auxiliary),
        0.0,
        epoch,
        source.mesh.topology_id,
        full.prepared_id,
        problem.compilation_id,
        state_version=epoch,
        materials=materials,
    )
    source_density = _surface_qualification_content(full, "density", auxiliary[0])
    target_space = phx.meshing.qualification.prepare_surface_fields(adaptation.target)
    rebinds: list[CompositionRebind] = []
    material_evidence: list[dict[str, object]] = []

    def transfer_material(
        original: Any, lineage: object, context: object, *, source_workspace: Any
    ) -> Any:
        del lineage, context
        if original.transaction_id != materials.transaction_id:
            raise ValueError(
                f"surface.material.transaction_id must equal accepted source transaction {materials.transaction_id!r}; got {original.transaction_id!r}"
            )
        transfer, evidence = _surface_qualification_material_transport(
            source, adaptation, original, limits, source_workspace=source_workspace
        )
        material_evidence.append(evidence)
        return transfer

    def capture(rebind: CompositionRebind) -> CompositionRebind:
        rebinds.append(rebind)
        return rebind

    def certify(
        mesh: Any,
        moved: tuple[Any, ...],
        candidate_materials: object,
        lineage: object,
        context: object,
    ) -> bool:
        del lineage, context
        if mesh.topology_id != adaptation.target.mesh.topology_id:
            return False
        if (
            not isinstance(candidate_materials, phx.equations.MaterialTransaction)
            or candidate_materials.state("surface-history").state_version
            != materials.state("surface-history").state_version + 1
        ):
            return False
        content = _surface_qualification_content(target_space, "density", moved[1])
        return (
            all(bool(jnp.all(jnp.isfinite(item))) for item in moved)
            and abs(content - source_density) <= 1.0e-10 * max(1.0, abs(source_density))
            and _surface_qualification_vector_norm(target_space, "curl", moved[3]) > 0.0
            and _surface_qualification_vector_norm(target_space, "flux", moved[4]) > 0.0
        )

    def impossible_density(
        mesh: Any,
        moved: tuple[Any, ...],
        materials: object,
        lineage: object,
        context: object,
    ) -> bool:
        del mesh, materials, lineage, context
        return _surface_qualification_content(target_space, "density", moved[1]) <= 0.0

    semantics = {
        field.name: role
        for role, field in phx.meshing.qualification.surface_field_templates()
    }
    rejected = phx.solver.FiniteElementTopologyTransaction(
        impossible_density,
        fields=fields,
        surface_chart_semantics=semantics,
        material_transfer=transfer_material,
        callback_owners=(full, target_space, materials, accepted.fields),
    ).execute(accepted, source.mesh, adaptation)
    if (
        bool(rejected.committed)
        or rejected.receipt is None
        or rejected.receipt.published
        or rejected.state is not accepted
        or rejected.mesh is not source.mesh
        or rejected.state.materials is not accepted.materials
        or any(
            not np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(rejected.state.fields, accepted.fields, strict=True)
        )
    ):
        raise qualification_failure(
            "_surface_qualification_transition",
            ("Rejected physical surface transaction changed its accepted source epoch."),
            {"accepted": accepted, "rejected": rejected, "source": source},
        )
    transaction = phx.solver.FiniteElementTopologyTransaction(
        certify,
        fields=fields,
        surface_chart_semantics=semantics,
        composition_rebind=capture,
        material_transfer=transfer_material,
        callback_owners=(full, target_space, materials, accepted.fields),
    ).execute(accepted, source.mesh, adaptation)
    if (
        not bool(transaction.committed)
        or transaction.receipt is None
        or not transaction.receipt.published
        or not rebinds
        or not all(item.evidence.passed for item in transaction.transfers)
    ):
        raise qualification_failure(
            "_surface_qualification_transition",
            (f"Physical surface CompositionRebind refused: {transaction.diagnostics}"),
            {"rebinds": rebinds, "transaction": transaction},
        )
    target_density = _surface_qualification_content(
        target_space, "density", transaction.state.fields[1]
    )
    scalar, _, _, _ = _surface_qualification_solver(adaptation.target, law)
    return (
        transaction.state.fields[0],
        transaction.state.fields[1:],
        transaction.state.materials,
        {
            "receipt_id": transaction.receipt.receipt_id,
            "source_composition_id": rebinds[-1].source.composition_id,
            "target_composition_id": transaction.receipt.composition.composition_id,
            "field_names": [field.name for field in fields],
            "surface_chart_semantics": semantics,
            "source_density_content": source_density,
            "target_density_content": target_density,
            "density_content_defect": target_density - source_density,
            "material_transport": material_evidence[-1],
            "remapped_physical_error": _surface_qualification_error(
                scalar, transaction.state.fields[0], law
            ),
            "field_transfers": [
                {
                    "transfer_id": item.transfer_id,
                    "passed": item.evidence.passed,
                    "defects": dict(item.evidence.defects),
                    "bounds": dict(item.evidence.bounds),
                }
                for item in transaction.transfers
            ],
            "rollback": {
                "passed": True,
                "requested_density_content_upper": 0.0,
                "receipt_id": rejected.receipt.receipt_id,
                "diagnostics": rejected.diagnostics,
            },
        },
    )


@final
class _SurfaceSolvedEpoch(phx.StrictModule):
    problem: phx.equations.CompiledFiniteElementProblem
    fields: tuple[Array, ...]
    compiler: CorpusCompilerEvidence
    timings: dict[str, float]
    error: float
    mean: float | None


def _surface_qualification_restart(
    payload: Path,
    args: QualificationControls,
    record: dict[str, Any],
    evidence: dict[str, Any],
    started: float,
    deadline: Deadline,
) -> None:
    """Restart each accepted archive in a fresh process within the campaign deadline."""

    remaining = deadline.expires_at - perf_counter()
    if remaining <= 0.0:
        raise TimeoutError(
            f"surface.remaining_wall_seconds must be positive for publication; got {remaining!r}"
        )
    # The scenario module owns its worker command.

    try:
        cold = run_cold_worker(
            "scenarios.surface",
            (str(payload),),
            deadline=deadline,
            ledger=StageLedger(
                record,
                deadline=deadline,
                cold_phase="surface_checkpoint_fresh_process_restart",
            ),
        )
    except ColdWorkerTimeout as failure:
        evidence.update(
            {
                "timeout_seconds": remaining,
                "stdout": failure.stdout,
                "stderr": failure.stderr,
            }
        )
        raise TimeoutError(
            f"surface.cold_restart must finish within {remaining!r} remaining wall seconds; got {type(failure).__name__}"
        ) from failure
    evidence.update(
        {
            "returncode": 0,
            "stdout": json.dumps(cold),
            "stderr": "",
        }
    )
    # Worker return-code evidence is retained by StageLedger.
    # Parsed and process-validated by run_cold_worker.
    if cold["process_id"] == os.getpid():
        raise qualification_failure(
            "_surface_qualification_checkpoint",
            ("Surface restart did not use an independent process."),
            {"cold": cold},
        )
    evidence["cold_restart"] = cold
    continuation = cold["continued_solver_epoch"]
    continuation["epoch"] = len(record["epochs"])
    record["epochs"].append(continuation)
    if (
        continuation["l2_error"] <= args.target_error
        and record["time_to_target_error_seconds"] is None
    ):
        record["time_to_target_error_seconds"] = perf_counter() - started


def _surface_qualification_checkpoint(
    source: Any,
    result: Any,
    request: Any,
    options: Any,
    law: Any,
    history: tuple[phx.meshing.qualification.SurfaceAcceptedEpoch, ...],
    adaptations: tuple[phx.meshing.MeshAdaptationResult, ...],
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
    deadline: Deadline,
) -> None:
    """Publish each atomic accepted epoch with its native source closure."""

    _ledger = StageLedger(record, deadline=deadline)

    if not history or history[-1].result.result_id != result.result_id:
        raise qualification_failure(
            "_surface_qualification_checkpoint",
            (
                "Surface checkpoint requires all accepted history through the current result."
            ),
            {"history": history, "result": result},
        )
    limits = ArrayArchiveLimits()
    policy = asdict(limits)
    policy["allowed_dtype_kinds"] = sorted(limits.allowed_dtype_kinds)
    evidence = {
        "archive_limits": policy,
        "prepared_fe_caches_serialized": False,
        "archives": [],
    }
    record["checkpoint"] = evidence
    with TemporaryDirectory(prefix="phydrax-native-surface-epochs-") as directory:
        publications = []
        for epoch, accepted in enumerate(history):
            state = phx.meshing.qualification.prepare_surface_checkpoint(
                accepted, epoch, failure_factory=qualification_failure
            )
            carrier = accepted.result
            certification = carrier.certification
            if certification is None:
                raise qualification_failure(
                    "_surface_qualification_checkpoint",
                    "Surface checkpoint carrier.certification must retain source request evidence; got None",
                    {"carrier": carrier},
                )
            records = {
                "certification_inputs": certification.request,
                "report": certification,
                "associations": carrier.associations,
                "generation_part": phx.meshing.MeshPart(
                    "generation", carrier, coordinate_contract=carrier.coordinate_contract
                ),
                "generation_source": source,
                "generation_specification": request,
                "generation_options": options,
                "field_declarations": {"generation": state.declaration},
                "accepted_data": {
                    "fields": state.fields,
                    "adaptation_results": tuple(adaptations[:epoch]),
                },
            }
            owner, values, material_identity = (
                state.owner,
                state.values,
                state.material_identity,
            )

            receipt = _ledger.run(
                f"surface_checkpoint_epoch_{epoch}",
                lambda: write_meshing_source_closure(
                    Path(directory) / f"epoch-{epoch}.zip", records, limits=limits
                ),
            )
            physical = {
                "error": accepted.error,
                "density_content": accepted.density_content,
                "history_content": _surface_qualification_content(
                    owner, "history", values["history"]
                ),
                "curl_norm": _surface_qualification_vector_norm(
                    owner, "curl", values["curl"]
                ),
                "flux_norm": _surface_qualification_vector_norm(
                    owner, "flux", values["flux"]
                ),
            }
            publication = {
                "path": str(receipt.path),
                "content_id": receipt.content_id,
                "fingerprint": array_tree_fingerprint(records),
                "physical": physical,
                "material": material_identity,
            }
            publications.append(publication)
            evidence["archives"].append(
                {
                    "epoch": epoch,
                    "content_id": receipt.content_id,
                    "container_bytes": receipt.path.stat().st_size,
                    "fingerprint": publication["fingerprint"],
                }
            )
        payload = Path(directory) / "cold-input.json"
        payload.write_text(
            json.dumps(
                {
                    "publications": publications,
                    "law": law,
                    "repeats": args.repeats,
                    "target_error": args.target_error,
                }
            )
        )
        _surface_qualification_restart(payload, args, record, evidence, started, deadline)


def _surface_qualification_cold_worker(
    publications: list[dict[str, Any]],
    law: _SurfaceLaw,
    repeats: int,
    target_error: float,
) -> dict[str, Any]:
    """Renew every retained native map and continue the final accepted PDE state."""

    renewed = []
    names = ("u", "density", "history", "curl", "flux")
    for epoch, publication in enumerate(publications):
        restored = read_meshing_source_closure(
            Path(publication["path"]),
            expected_content_id=publication["content_id"],
            limits=ArrayArchiveLimits(),
        )
        if array_tree_fingerprint(restored) != publication["fingerprint"]:
            raise qualification_failure(
                "_surface_qualification_cold_worker",
                ("Surface cold closure changed source, controls, result or field bits."),
                {
                    "array_tree_fingerprint": array_tree_fingerprint,
                    "publication": publication,
                    "restored": restored,
                },
            )
        carrier = restored["generation_part"].carrier
        report = recertify_restored_meshing_source(
            restored["certification_inputs"],
            carrier.mesh,
            carrier.geometry,
            carrier.audit,
            archive_limits=ArrayArchiveLimits(),
        )
        report.require_passed()
        if report.report_id != restored["report"].report_id:
            raise qualification_failure(
                "_surface_qualification_cold_worker",
                ("Surface source renewal changed the accepted certification theorem."),
                {"report": report, "restored": restored},
            )
        owner, _ = prepare_meshing_field_owner(
            restored["field_declarations"]["generation"], carrier
        )
        arrays = restored["accepted_data"]["fields"]
        if int(np.asarray(arrays["surface:accepted-epoch"])) != epoch:
            raise qualification_failure(
                "_surface_qualification_cold_worker",
                ("Surface restored accepted epoch changed."),
                {"arrays": arrays, "epoch": epoch},
            )
        values = tuple(jnp.asarray(arrays[f"surface:{name}"]) for name in names)
        scalar, problem, free, weights = _surface_qualification_solver(carrier, law)
        physical = {
            "error": _surface_qualification_error(scalar, values[0], law),
            "density_content": _surface_qualification_content(
                owner, "density", values[1]
            ),
            "history_content": _surface_qualification_content(
                owner, "history", values[2]
            ),
            "curl_norm": _surface_qualification_vector_norm(owner, "curl", values[3]),
            "flux_norm": _surface_qualification_vector_norm(owner, "flux", values[4]),
        }
        if physical != publication["physical"]:
            raise qualification_failure(
                "_surface_qualification_cold_worker",
                (
                    "Surface restored physical error, inventory or mapped vector norm changed."
                ),
                {"physical": physical, "publication": publication},
            )
        material = publication["material"]
        if material is not None:
            transaction = phx.equations.MaterialTransaction(
                (
                    phx.equations.MaterialState(
                        phx.equations.MaterialSiteId("surface-history"),
                        "surface-history",
                        arrays["surface:material"],
                        trial=arrays["surface:material-trial"],
                        state_version=material["state_version"],
                    ),
                )
            )
            if transaction.transaction_id != material["transaction_id"]:
                raise qualification_failure(
                    "_surface_qualification_cold_worker",
                    ("Surface cold material primitive state changed."),
                    {"material": material, "transaction": transaction},
                )
        renewed.append(
            {
                "epoch": epoch,
                "result": _result_identity(carrier),
                "report_id": report.report_id,
                "physical": physical,
            }
        )
    if not renewed:
        raise qualification_failure(
            "_surface_qualification_cold_worker",
            ("Surface cold continuation has no accepted epoch."),
            {"renewed": renewed},
        )
    solved, compiler, stages = _compiled_solver_stages(problem, values[0][free], repeats)
    if weights is not None:
        solved = solved - jnp.sum(weights * solved) / jnp.sum(weights)
    error = _surface_qualification_error(scalar, solved, law)
    return {
        "process_id": os.getpid(),
        "runtime": _runtime_identity(),
        "accepted_fields_bitwise_preserved": True,
        "prepared_fe_caches_serialized": False,
        "renewed_epochs": renewed,
        "continued_solver_epoch": {
            "result": _result_identity(carrier),
            "l2_error": error,
            "target_error": target_error,
            "target_error_met": error <= target_error,
            "seed_origin": "actual-cold-restored-accepted-u-free",
            "compiler": compiler,
            "stages_seconds": stages,
        },
    }


def _surface_qualification_adapt(
    result: Any, initial: Any, request: Any, record: dict[str, Any]
) -> Any:

    charts = initial.surface_source
    if (
        charts is None
        or initial.certification is None
        or initial.certification.fidelity is None
    ):
        raise qualification_failure(
            "_surface_qualification_adapt",
            ("Surface adaptation requires retained native charts and source fidelity."),
            {"charts": charts, "initial": initial},
        )
    charts.require_root(initial.mesh, initial.geometry)
    support = PreparedSurfaceSourceSupport(
        charts.domain,
        initial,
        charts.root_parameters,
        charts.boundary_source,
        maximum_support_queries=charts.maximum_support_queries,
    )
    vertices = result.mesh.entity_set(0)
    scope = phx.meshing.MeshingScope(
        result.mesh.mesh_id,
        result.mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        0,
        vertices.entity_set_id,
        vertices.entity_ids,
    )
    size = request.size_controls[0].target_size / 2.0
    metric = phx.meshing.MeshMetricField(
        scope,
        np.broadcast_to(np.eye(3) / size**2, (result.mesh.coordinates.shape[0], 3, 3)),
        minimum_size=size,
        maximum_size=size,
    )
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_SURFACE_METRIC,
        limits=request.limits,
        association_transfer=SurfaceAssociationTransfer(support),
        geometry_transition=phx.discretization.CellGeometryTransitionPolicy(
            reconstruction="bounded_chart_deformation",
            reconstruction_tolerance=initial.certification.fidelity.tolerance,
            maximum_evaluations=request.limits.maximum_work_units,
        ),
        audit_policy=phx.meshing.CellMeshAuditPolicy(require_complete_association=True),
    )
    adaptation = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            result,
            phx.meshing.MetricMeshAdaptation(
                metric, coordinate_contract=result.coordinate_contract
            ),
            policy=policy,
        )
    )
    if (
        adaptation.status is not phx.meshing.MeshAdaptationStatus.COMPLETE
        or adaptation.target.mesh.topology_id == result.mesh.topology_id
    ):
        evidence: Any = adaptation.evidence
        record["adaptation_refusal"] = {
            "status": adaptation.status.value,
            "source_result_id": adaptation.source.result_id,
            "target_result_id": adaptation.target.result_id,
            "resource_message": evidence.resource_message,
            "resource_requested": dict(evidence.resource_requested),
            "resource_achieved": dict(evidence.resource_achieved),
            "work_units": evidence.work_units,
            "native_work_units": evidence.native_work_units,
            "passes": evidence.passes,
            "splits": evidence.splits,
            "collapses": evidence.collapses,
            "flips": evidence.flips,
            "relocations": evidence.relocations,
            "execution": (
                None
                if evidence.execution_evidence is None
                else evidence.execution_evidence.to_record()
            ),
        }
        raise qualification_failure(
            "_surface_qualification_adapt",
            (
                f"Source surface adaptation did not finish a topology transition: {adaptation.status.value}."
            ),
            {"adaptation": adaptation, "result": result},
        )
    return adaptation


def _surface_qualification_state_inventory(
    result: Any, fields: tuple[Any, ...], material: Any
) -> tuple[Any, float, dict[str, float]]:
    actual_material = (
        _surface_qualification_material_initial(result) if material is None else material
    )
    owner = phx.meshing.qualification.prepare_surface_fields(result)
    density = _surface_qualification_content(owner, "density", fields[1])
    norms = {
        name: _surface_qualification_vector_norm(owner, name, fields[index])
        for name, index in (("curl", 3), ("flux", 4))
    }
    return actual_material, density, norms


def _surface_qualification_solve_epoch(
    result: phx.meshing.CellMeshingResult,
    law: _SurfaceLaw,
    previous: tuple[Array, ...] | None,
    repeats: int,
) -> _SurfaceSolvedEpoch:
    """Solve from authored or carried coefficients and evaluate the physical oracle."""

    space, problem, free, weights = _surface_qualification_solver(result, law)
    seed = (
        problem.state_space.zeros()
        if previous is None
        else jnp.asarray(previous[0])[free]
    )
    values, compiler, timings = _compiled_solver_stages(problem, seed, repeats)
    fields = (
        _surface_qualification_initial_fields(result, values)
        if previous is None
        else (values, *previous[1:])
    )
    error = _surface_qualification_error(space, values, law)
    mean = (
        None if weights is None else float(weights @ np.asarray(values) / np.sum(weights))
    )
    return _SurfaceSolvedEpoch(
        problem=problem,
        fields=fields,
        compiler=compiler,
        timings=timings,
        error=error,
        mean=mean,
    )


def _surface_qualification_accept_epoch(
    result: phx.meshing.CellMeshingResult,
    request: phx.meshing.SurfaceMeshingSpec,
    solved: _SurfaceSolvedEpoch,
    previous_material: phx.equations.MaterialTransaction | None,
    acceptance: dict[str, object],
    epoch: int,
    authored_initial: bool,
    record: dict[str, Any],
) -> phx.meshing.qualification.SurfaceAcceptedEpoch:
    """Require nonzero physical inventories and retain one accepted epoch."""

    fields, error, mean = solved.fields, solved.error, solved.mean
    compiler, timings = solved.compiler, solved.timings
    result, inventory = _corpus_native_consumer(
        result,
        request,
        lambda: _surface_qualification_state_inventory(result, fields, previous_material),
        native=True,
    )
    material, density, vector_norms = inventory
    if density <= 0.0 or any(
        value <= 0.0 or not np.isfinite(value) for value in vector_norms.values()
    ):
        raise qualification_failure(
            "_surface_qualification_lifecycle",
            (
                "Physical surface fields lost their authored nonzero density or compatible moments."
            ),
            {"density": density, "vector_norms": vector_norms},
        )
    record["epochs"].append(
        {
            "epoch": epoch,
            "result": _result_identity(result),
            "source_acceptance": acceptance,
            "physical_error": error,
            "surface_mean": mean,
            "density_content": density,
            "compatible_physical_norms": vector_norms,
            "compiler": compiler,
            "seed_origin": "authored-initial-state"
            if authored_initial
            else "actual-carried-u",
        }
    )
    record["stages_seconds"].update(
        {f"{name}_epoch_{epoch}": value for name, value in timings.items()}
    )
    return phx.meshing.qualification.SurfaceAcceptedEpoch(
        result=result,
        fields=fields,
        material=material,
        error=error,
        density_content=density,
    )


def _surface_qualification_record(args: QualificationControls) -> dict[str, Any]:
    """Initialize the surface qualification's external evidence record."""

    record: dict[str, Any] = {
        "scenario": "native-surface-pde",
        "surface_case": args.surface_case,
        "status": "failed",
        "runtime": _runtime_identity(),
        "targets": {"physical_error": args.target_error, "wall_seconds": args.timeout},
        "control_capacities": {
            "resolution": args.resolution,
            "entity_capacity": args.capacity,
        },
        "stages_seconds": {},
        "epochs": [],
        "mandatory_workflow_completion": "incomplete",
        "time_to_target_error_seconds": None,
        "release_evidence": "unassessed",
        "leadership_evidence": "unassessed",
    }
    return record


def _surface_qualification_completion(
    accepted: phx.meshing.qualification.SurfaceAcceptedEpoch,
    acceptance: dict[str, object],
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
) -> None:
    """Publish the final physical target verdict and bounded completion evidence."""
    elapsed = perf_counter() - started
    final_error = accepted.error
    completed = final_error <= args.target_error and elapsed <= args.timeout
    record.update(
        {
            "status": "passed"
            if completed
            else "timeout"
            if elapsed > args.timeout
            else "failed",
            "physical_error": final_error,
            "target_error_reached": final_error <= args.target_error,
            "independent_scientific_certification": (
                "passed" if final_error <= args.target_error else "failed"
            ),
            "mandatory_workflow_completion": "complete" if completed else "incomplete",
            "source_acceptance": acceptance,
            "time_to_target_error_seconds": elapsed if completed else None,
            "field_names": ["u", "density", "history", "curl", "flux"],
        }
    )


def _surface_qualification_lifecycle(args: QualificationControls) -> dict[str, object]:

    started = perf_counter()
    deadline = Deadline(started + args.timeout)
    record = _surface_qualification_record(args)
    _ledger = StageLedger(record, deadline=deadline)
    phases: list[NativeMeshingPhaseMeasurement] = []
    phase = "original_source_request_preparation"
    try:
        prepared, timing = _ledger.run(
            phase, lambda: _measured(lambda: _surface_qualification_prepare(args))
        )
        source, request, options, law = prepared
        record["stages_seconds"][phase] = timing["wall_seconds"]
        record.update(
            {
                "source_id": source.source_id,
                "source_revision": source.source_revision,
                "source_binding_id": source.binding_id,
                "request_id": request.specification_id,
                "limits_id": request.limits.limits_id,
                "profile_law": law,
            }
        )
        phase = "original_native_source_preparation"
        provider = phx.meshing.NativeMeshingProvider(options)
        plan, timing = _ledger.run(
            phase,
            lambda: _measured(
                lambda: provider.plan(
                    source,
                    request,
                    coordinate_contract=phx.SpatialCoordinateContract.si(),
                    record_phase=phases.append,
                )
            ),
        )
        record["stages_seconds"][phase] = timing["wall_seconds"]
        phase = "original_public_surface_publication"
        result, timing = _ledger.run(
            phase, lambda: _measured(lambda: plan.execute(record_phase=phases.append))
        )
        record["stages_seconds"][phase] = timing["wall_seconds"]
        initial = result
        record["generation"] = _result_identity(initial)
        accepted_history: list[phx.meshing.qualification.SurfaceAcceptedEpoch] = []
        adaptations = []
        previous = None
        for epoch in range(args.adaptation_rounds + 1):
            phase = f"source_acceptance_epoch_{epoch}"
            result, acceptance = _corpus_native_consumer(
                result,
                request,
                lambda: _surface_qualification_acceptance(source, result, request, law),
                native=True,
            )
            if not acceptance["passed"]:
                raise qualification_failure(
                    "_surface_qualification_lifecycle",
                    (
                        "Surface publication failed the source, physical measure or topology oracle."
                    ),
                    {"acceptance": acceptance},
                )
            phase = f"physical_solver_epoch_{epoch}"

            result, solved = _corpus_native_consumer(
                result,
                request,
                lambda: _surface_qualification_solve_epoch(
                    result, law, previous, args.repeats
                ),
                native=True,
            )
            problem, fields, error, mean = (
                solved.problem,
                solved.fields,
                solved.error,
                solved.mean,
            )
            if not np.isfinite(error) or (
                law["name"] == "sphere" and (mean is None or abs(mean) > 1.0e-10)
            ):
                raise qualification_failure(
                    "_surface_qualification_lifecycle",
                    (
                        "Physical surface PDE failed its independent error/zero-mean oracle."
                    ),
                    {"error": error, "law": law, "mean": mean},
                )
            accepted = _surface_qualification_accept_epoch(
                result,
                request,
                solved,
                None if previous is None else accepted_history[-1].material,
                acceptance,
                epoch,
                previous is None,
                record,
            )
            result, material = accepted.result, accepted.material
            accepted_history.append(accepted)
            if epoch == args.adaptation_rounds:
                break
            phase = f"original_source_metric_adaptation_epoch_{epoch}"
            result, adaptation = _corpus_native_consumer(
                result,
                request,
                lambda: _surface_qualification_adapt(result, initial, request, record),
                native=True,
            )
            adaptations.append(adaptation)
            phase = f"allphysical_history_composition_epoch_{epoch}"
            moved, auxiliary, previous_material, transition = (
                _surface_qualification_transition(
                    adaptation.source,
                    adaptation,
                    problem,
                    fields[0],
                    fields[1:],
                    epoch,
                    law,
                    material,
                    request.limits,
                )
            )
            previous = (moved, *auxiliary)
            record["epochs"][-1].update(
                {
                    "state_transition": transition,
                    "adaptation": {
                        "status": adaptation.status.value,
                        "source_result_id": adaptation.source.result_id,
                        "target": _result_identity(adaptation.target),
                    },
                }
            )
            result = adaptation.target
        phase = "default_allfield_accepted_history_checkpoint_cold_continuation"
        _surface_qualification_checkpoint(
            source,
            result,
            request,
            options,
            law,
            tuple(accepted_history),
            tuple(adaptations),
            args,
            record,
            started,
            deadline,
        )
        _surface_qualification_completion(
            accepted_history[-1], acceptance, args, record, started
        )
    except phx.meshing.MeshingFailure as failure:
        record_failure(record, failure, phase)
    except (RuntimeError, ValueError, TypeError, TimeoutError) as failure:
        record_failure(record, failure, phase)
    finally:
        record["native_phases"] = _phase_recordings(phases)
        record["end_to_end_seconds"] = perf_counter() - started
        record["peak_process_resident_bytes"] = peak_resident_bytes(resource.RUSAGE_SELF)
    return record


def qualify_native_surface(args: QualificationControls) -> dict[str, object]:
    """Exercise the source, all physical states and cold continued PDE."""
    return _surface_qualification_lifecycle(args)


def worker(argv: tuple[str, ...]) -> dict[str, Any]:
    """Decode process arguments and run the scenario-owned cold consumer."""
    p = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    return _surface_qualification_cold_worker(**p)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "worker":
        raise ValueError(
            "qualification.subcommand must equal worker; got " + repr(sys.argv[1:])
        )
    print(json.dumps(worker(tuple(sys.argv[2:]))))


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host qualification controls."""
    return qualify_native_surface(args)
