# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, perf_counter
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
import phydrax.meshing.qualification._polyhedral as polyhedral_consumer_owner
from examples.native_polyhedral_meshing import (
    prepare_fv_system,
    prepare_vem_system,
)
from phydrax import array_tree_fingerprint, ArrayArchiveLimits, model_array_bytes
from phydrax.discretization import (
    FieldEpochTransition,
    TopologyEpoch,
    TopologyEpochTransition,
    TransferGeometryBinding,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_operands import (
    CoordinateOperandCollector,
    PreparedCoordinateOperands,
)
from phydrax.discretization.finite_volume import (
    HybridDiffusionBoundary,
    HybridMimeticDiffusion,
    UnstructuredConservativeRemapPlan,
    UnstructuredFiniteVolumePlan,
)
from phydrax.discretization.vem import (
    prepare_polyhedral_h1_virtual_element_3d,
    PreparedPolyhedralH1VirtualElement3D,
)
from phydrax.discretization.vem._polyhedral import ZeroMeanLoadCompatibilityBound
from phydrax.geometry._domain_coverage import DomainCoverageCertificate
from phydrax.lifecycle import (
    _meshing_field_records as polyhedral_field_owner,
    commit_composition_rebind,
    Composition,
    CompositionEntry,
    CompositionRebind,
    CompositionRole,
    MeshingAcceptedEpoch,
    prepare_meshing_field_owner,
    read_meshing_source_closure,
    read_meshing_source_execution_controls,
    write_meshing_source_closure,
)
from phydrax.meshing import (
    meshcore_available,
    MeshcoreStatus,
    NativeExecutionRecord,
    NativeMeshingPhaseMeasurement,
)
from phydrax.meshing.qualification import (
    execute_native_capacity_trial,
    execute_native_consumer,
    execute_native_request,
    prepare_periodic_source_quadrature,
)
from phydrax.units import derived_unit, METER, ONE, UnitDefinition
from tools._meshing_cases import (
    frozen_request_record,
    MeshingCase,
    polyhedral_case,
    polyhedral_identity,
    prepare_polyhedral_case,
)
from tools.meshing_qualification._controls import (
    meshing_limit_values,
    MeshingLimitControls,
    QualificationControls,
    with_meshing_limits,
)
from tools.meshing_qualification._records import (
    _phase_recordings,
    _require_source_prefix_trial,
    _result_identity,
    _source_prefix_boundary_limits,
    FailureRecord,
    record_failure,
)
from tools.meshing_qualification._runtime import (
    _measured,
    _missing_dependency,
    _PolyhedralJitEntry,
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


def _polyhedral_memory_measurements() -> dict[str, Any]:
    """Report actual backend/process observations outside native admission."""
    return {
        "peak_process_resident_bytes": peak_resident_bytes(),
        "process_peak_scope": PEAK_RESIDENT_SCOPE,
        "device_memory": [
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
        ],
    }


def _polyhedral_acceptance(
    result: phx.meshing.CellMeshingResult,
    case: MeshingCase,
) -> tuple[np.ndarray, dict[str, object]]:
    """Read independently certified domain coverage and cell labels."""

    certification = result.certification
    if (
        certification is None
        or not certification.passed
        or not result.audit.passed
        or not result.compliance.passed
    ):
        raise qualification_failure(
            "_polyhedral_acceptance",
            ("Polyhedral publication lacks independent scientific certification."),
            {"certification": certification, "result": result},
        )
    if any(block.cell_kind != "polyhedron" for block in result.mesh.blocks):
        raise qualification_failure(
            "_polyhedral_acceptance",
            ("Polyhedral request published a different cell family."),
            {"result": result},
        )
    coverage = certification.coverage
    if not isinstance(coverage, DomainCoverageCertificate) or coverage.status != "certified":
        raise qualification_failure(
            "_polyhedral_acceptance",
            ("Polyhedral publication lacks certified source/material coverage."),
            {"coverage": coverage},
        )
    expected = dict(case.expected_regions)
    measured = dict(
        zip(coverage.region_ids, coverage.achieved_region_measures, strict=True)
    )
    if set(measured) != set(expected):
        raise qualification_failure(
            "_polyhedral_acceptance",
            ("Independent polyhedral material measures disagree with analytic voxels."),
            {"expected": expected, "measured": measured},
        )
    for name, wanted in expected.items():
        achieved = measured[name]
        if achieved is None or abs(achieved - wanted) > 1e-10 * max(1.0, wanted):
            raise qualification_failure(
                "_polyhedral_acceptance",
                (
                    "Independent polyhedral material measures disagree with analytic voxels."
                ),
                {"achieved": achieved, "wanted": wanted},
            )
    connectivity = result.mesh.connectivity
    if not isinstance(connectivity, phx.discretization.PolyhedralConnectivity):
        raise TypeError(
            f"result.mesh.connectivity must be PolyhedralConnectivity; got {type(connectivity).__name__}"
        )
    identifiers = np.asarray(connectivity.cell_global_ids)
    regions = np.full(identifiers.size, -1, dtype=np.int64)
    zones = {
        zone.name: zone
        for zone in result.zones
        if zone.role is phx.meshing.MeshZoneRole.REGION
    }
    if set(zones) != set(expected):
        raise qualification_failure(
            "_polyhedral_acceptance",
            ("Polyhedral result lost its requested material organization."),
            {"expected": expected, "zones": zones},
        )
    for index, (name, _) in enumerate(case.expected_regions):
        zone = zones[name]
        selected = np.isin(identifiers, np.asarray(zone.scope.entity_ids))
        periodic = "periodic-quotient" in case.controls
        material_id = None if periodic else f"{name}-material"
        role = None if periodic else phx.meshing.RegionRole.SOLID
        if (
            zone.material_id != material_id
            or zone.region_role is not role
            or not np.any(selected)
            or np.any(regions[selected] >= 0)
            or np.count_nonzero(selected) != len(zone.scope.entity_ids)
        ):
            raise qualification_failure(
                "_polyhedral_acceptance",
                (f"Polyhedral region {name} has invalid material or cell ownership."),
                {
                    "material_id": material_id,
                    "regions": regions,
                    "role": role,
                    "selected": selected,
                    "zone": zone,
                },
            )
        regions[selected] = index
    if np.any(regions < 0):
        raise qualification_failure(
            "_polyhedral_acceptance",
            ("Polyhedral material zones do not assign every mesh cell."),
            {"regions": regions},
        )
    certified_regions = certification.request.cell_regions
    if certified_regions is None or not np.array_equal(
        regions, np.asarray(certified_regions)
    ):
        raise qualification_failure(
            "_polyhedral_acceptance",
            ("Polyhedral material organization disagrees with certified cell regions."),
            {"certified_regions": certified_regions, "regions": regions},
        )
    return regions, {
        "passed": True,
        "certification_id": certification.report_id,
        "coverage_id": coverage.certificate_id,
        "coverage_status": coverage.status,
        "expected_region_measures": expected,
        "achieved_region_measures": measured,
        "source_facets": coverage.source_facet_count,
        "covered_source_facets": coverage.covered_source_facet_count,
        "material_organization": "exclusive canonical cell IDs with requested material IDs",
    }


@eqx.filter_jit
def _solve_polyhedral_system(system: Any, rhs: Any, seed: Any) -> tuple[Any, Any, Any]:
    solved = phx.linalg.solve(
        system,
        rhs,
        policy=phx.linalg.LinearSolvePolicy(phx.linalg.ConjugateGradient()),
        initial_guess=seed,
    )
    return solved.value, solved.successful, solved.diagnostics


def _compiled_polyhedral_stages(
    system: Any,
    rhs: Any,
    seed: Any,
    repeats: int,
) -> tuple[Any, dict[str, object], dict[str, float]]:
    """Measure compilation and execution of the prepared linear system."""

    entry = _solve_polyhedral_system
    if not isinstance(entry, _PolyhedralJitEntry):
        raise TypeError(
            f"polyhedral.solver must expose its compiled lowering boundary; got {type(entry).__name__}"
        )
    lowered, lowering = _measured(lambda: entry.lower(system, rhs, seed))
    compiled, compilation = _measured(lowered.compile)
    memory = compiled.compiled.memory_analysis()
    first, execution = _measured(
        lambda: jax.block_until_ready(compiled(system, rhs, seed))
    )
    warm = []
    for _ in range(repeats):
        _, stage = _measured(lambda: jax.block_until_ready(compiled(system, rhs, seed)))
        warm.append(stage["wall_seconds"])
    values, successful, diagnostics = first
    if not bool(np.all(np.asarray(successful))):
        raise qualification_failure(
            "_compiled_polyhedral_stages",
            ("Polyhedral prepared VEM/FV solver did not converge."),
            {"successful": successful},
        )
    return (
        values,
        {
            "successful": True,
            "kernel": "_solve_polyhedral_system",
            "temporary_bytes": None if memory is None else memory.temp_size_in_bytes,
            "iterations": np.asarray(diagnostics.iterations).tolist(),
            "residual_norm": np.asarray(diagnostics.residual_norm).tolist(),
            "relative_residual": np.asarray(diagnostics.relative_residual).tolist(),
            "finite": bool(np.all(np.asarray(diagnostics.finite))),
            "output_bytes": None if memory is None else memory.output_size_in_bytes,
            "argument_bytes": None if memory is None else memory.argument_size_in_bytes,
            "generated_code_bytes": None
            if memory is None
            else memory.generated_code_size_in_bytes,
            "memory_status": "unavailable" if memory is None else "measured",
            "memory_scope": "compiled-executable-static-memory-requirements; outside native admission; not compiler-process peak",
            "lowering_seconds": lowering["wall_seconds"],
            "compile_seconds": compilation["wall_seconds"],
            "first_execution_seconds": execution["wall_seconds"],
            "compilation_count": 1,
            "compilation_count_semantics": "explicit compile calls; not backend cache misses",
            "warm_samples_seconds": warm,
            "logical_retained_bytes": model_array_bytes((system, rhs, seed)),
        },
        {
            "lowering": lowering["wall_seconds"],
            "compile": compilation["wall_seconds"],
            "first_execution": execution["wall_seconds"],
            "warm_execution": float(np.median(warm)),
        },
    )


@final
class _PolyhedralMaterialRemap(phx.StrictModule):
    common: phx.geometry.PreparedCommonRefinement
    remapped: np.ndarray
    evidence: dict[str, Any]
    inventory_after: float


@final
class _PolyhedralTransferredPhysics(phx.StrictModule):
    record: dict[str, Any]
    error: float
    retained: tuple[phx.StrictModule | jax.Array, ...]
    transferred_source_integral: float
    domain_measure: float
    fields: dict[str, jax.Array]
    owners: polyhedral_consumer_owner.PolyhedralArchiveOwners


def _polyhedral_remap(
    source: Any,
    target: Any,
    source_regions: np.ndarray,
    target_regions: np.ndarray,
    policy: Any,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PolyhedralMaterialRemap:

    common = phx.geometry.prepare_common_refinement(
        source.mesh,
        target.mesh,
        policy=policy,
        source_geometry=source.geometry,
        target_geometry=target.geometry,
        budget=budget,
    )
    if common.status is not phx.geometry.CommonRefinementStatus.SUCCESS:
        raise qualification_failure(
            "_polyhedral_remap",
            (f"Polyhedral common refinement failed: {common.evidence.reason}"),
            {"common": common},
        )
    values = np.where(source_regions == 0, 2.0, 5.0)
    source_rows, target_rows = (
        np.asarray(common.source_cells),
        np.asarray(common.target_cells),
    )
    volumes = np.asarray(common.volumes)
    if np.any(source_regions[source_rows] != target_regions[target_rows]):
        raise qualification_failure(
            "_polyhedral_remap",
            ("Polyhedral overlap transfers inventory across material interfaces."),
            {
                "source_regions": source_regions,
                "source_rows": source_rows,
                "target_regions": target_regions,
                "target_rows": target_rows,
            },
        )
    masses = np.bincount(
        target_rows,
        weights=volumes * values[source_rows],
        minlength=target.mesh.connectivity.cell_count,
    )
    transferred = masses / np.asarray(common.target_measures)
    inventories = {}
    for region, name in enumerate(("left", "right")):
        before = float(
            np.dot(
                values[source_regions == region],
                np.asarray(common.source_measures)[source_regions == region],
            )
        )
        after = float(
            np.dot(
                transferred[target_regions == region],
                np.asarray(common.target_measures)[target_regions == region],
            )
        )
        if abs(before - after) > 1e-10 * max(1.0, abs(before)):
            raise qualification_failure(
                "_polyhedral_remap",
                (f"Polyhedral conservative remap loses {name} inventory."),
                {"after": after, "before": before},
            )
        inventories[name] = {
            "before": before,
            "after": after,
            "absolute_defect": abs(before - after),
        }
    if not np.all(np.isfinite(transferred)) or np.any(transferred <= 0.0):
        raise qualification_failure(
            "_polyhedral_remap",
            ("Polyhedral remap did not preserve positive finite material state."),
            {"transferred": transferred},
        )
    return _PolyhedralMaterialRemap(
        common=common,
        remapped=transferred,
        inventory_after=sum(item["after"] for item in inventories.values()),
        evidence={
            "status": "success",
            "semantics": "common-refinement overlap-volume cell averages",
            "policy_id": policy.policy_id,
            "maximum_candidate_pairs": policy.maximum_candidate_pairs,
            "maximum_accepted_pairs": policy.maximum_accepted_pairs,
            "maximum_memory_bytes": policy.maximum_memory_bytes,
            "maximum_exact_work": policy.maximum_exact_work,
            "overlap_pieces": volumes.size,
            "material_inventories": inventories,
            "maximum_relative_source_defect": common.evidence.maximum_relative_source_defect,
            "maximum_relative_target_defect": common.evidence.maximum_relative_target_defect,
            "consumed_by": "target mimetic diffusion integrated cell source",
        },
    )


def _polyhedral_affine_fem_solution(points: jax.Array) -> jax.Array:
    """A harmonic affine field tangent to the authored x-normal material interface."""
    return 1.0 + points[:, 2]


def _reanalyse_periodic_fem_history(
    plan: polyhedral_consumer_owner.PolyhedralSourceStarFEMPlan,
    updated: dict[str, jax.Array],
    conserved_history: jax.Array,
) -> None:
    values_by_name: dict[str, list[jax.Array]] = {}
    for coefficients, conserved in zip(updated["fem_solution_history"], conserved_history, strict=True):
        density = conserved[:, 0]
        mean = jnp.sum(density * plan.cell_volumes) / jnp.sum(plan.cell_volumes)
        bound = polyhedral_consumer_owner._source_star_load_bound(plan, density, True)
        physical = polyhedral_consumer_owner.evaluate_polyhedral_source_star_fields(
            plan, coefficients, density - mean, jnp.ones_like(density),
            compatibility_bound=bound,
        )
        for name, value in (
            ("fem_cell_solution", physical.cell_values),
            ("fem_gradient", physical.cell_gradients),
            ("fem_vector_flux", physical.cell_vector_flux),
            ("fem_physical_residual", physical.physical_residual),
            ("fem_physical_load", physical.nodal_load),
        ):
            values_by_name.setdefault(name, []).append(value)
    for name, values in values_by_name.items():
        updated[name + "_history"] = jnp.stack(values)


def _polyhedral_fem_physics(
    result: phx.meshing.CellMeshingResult,
    fields: dict[str, jax.Array],
    material_or_density: jax.Array,
    stages: dict[str, float],
    *,
    repeats: int,
    amplitude: float | None = None,
    budget: CoordinateEnclosureBudget | None = None,
    prepared_plan: polyhedral_consumer_owner.PolyhedralSourceStarFEMPlan | None = None,
) -> tuple[dict[str, Any], float, dict[str, jax.Array]]:
    plan, timing = _measured(
        lambda: polyhedral_consumer_owner.prepare_polyhedral_source_star_fem(
            result, budget=budget
        ) if prepared_plan is None else prepared_plan
    )
    stages["fem_source_star_preparation"] = timing["wall_seconds"]
    if result.geometry is None or plan.source_mesh_id != result.mesh.mesh_id or plan.source_geometry_id != cell_geometry_id(result.geometry):
        raise ValueError("FE physics must retain its actual canonical source-star support.")
    exact: Callable[[jax.Array], jax.Array]
    if amplitude is None:
        exact = _polyhedral_affine_fem_solution
        forcing = jnp.zeros_like(material_or_density)
        material = material_or_density
    else:
        def periodic_exact(points: jax.Array) -> jax.Array:
            x = jnp.mod(points[:, 0], 1.0)
            right = x - 0.5
            return amplitude * jnp.where(
                x <= 0.5, 0.5 * x * x - 0.25 * x,
                -0.5 * right * right + 0.25 * right,
            )

        exact = periodic_exact
        forcing = material_or_density
        material = jnp.ones_like(material_or_density)
    state, timing = _measured(
        lambda: polyhedral_consumer_owner.solve_polyhedral_source_star_diffusion(
            plan, forcing, cell_material=material, exact_solution=exact,
            seed=fields.get("fem_solution"),
            history=tuple(fields.get("fem_solution_history", ())),
            periodic_density=amplitude is not None,
            budget=budget,
            repeats=repeats,
        )
    )
    stages["fem_physical_solve"] = timing["wall_seconds"]
    numerical = {
        "fem_solution": state.coefficients,
        "fem_cell_solution": state.cell_values,
        "fem_gradient": state.cell_gradients,
        "fem_vector_flux": state.cell_vector_flux,
        "fem_physical_residual": state.physical_residual,
        "fem_physical_load": state.nodal_load,
    }
    updated = dict(numerical)
    for name, value in numerical.items():
        if amplitude is None or name == "fem_solution":
            updated[name + "_history"] = _polyhedral_qualification_field_history(fields, name, value)
    if amplitude is not None:
        _reanalyse_periodic_fem_history(plan, updated, fields["conserved_history"])
    error = max(
        float(state.l2_error), float(state.maximum_free_residual),
        float(state.conservation_defect),
    )
    return {
        "authority": "canonical exact-source conforming P1 star tetrahedra; owning finite-element equations and linear solver",
        "plan_id": plan.plan_id,
        "source_mesh_id": plan.source_mesh_id,
        "source_geometry_id": state.source_geometry_id,
        "prepared_id": state.prepared_id,
        "runtime_id": state.runtime_id,
        "compiler": dict(state.compiler),
        "tetrahedron_parent_fingerprint": array_tree_fingerprint(plan.tetrahedron_parents),
        "source_preparation_work_units": plan.source_preparation_work_units,
        "physical_l2_error": float(state.l2_error),
        "maximum_free_physical_residual": float(state.maximum_free_residual),
        "global_weak_conservation_defect": float(state.conservation_defect),
        "load_compatibility_residual": float(state.load_compatibility_residual),
        "load_compatibility_bound": float(state.load_compatibility_bound),
        "vector_flux_finite": bool(jnp.all(jnp.isfinite(state.vector_flux))),
        "restored_or_transferred_seed_consumed": "fem_solution" in fields,
        "seed_fingerprint": array_tree_fingerprint(state.seed),
        "solution_fingerprint": array_tree_fingerprint(state.coefficients),
        "initial_residual_norm": float(jnp.linalg.norm(state.initial_residual)),
        "history_reanalysis": "native source-star evaluation of actual retained primal/coefficient history; no historic equilibrium claim",
        "solver_status": np.asarray(state.solver_status).tolist(),
    }, error, updated


@eqx.filter_jit
def _material_fv_physical_fields(
    diffusion: HybridMimeticDiffusion,
    tensor: jax.Array,
    boundary: HybridDiffusionBoundary,
    inventory: jax.Array,
    values: jax.Array,
) -> dict[str, jax.Array]:
    cells, traces = values[:diffusion.cell_count], values[diffusion.cell_count:]
    gradient = diffusion.gradient(cells, traces)
    local = diffusion.local_fluxes(cells, traces, tensor)
    owner = diffusion.owner_rates(local)
    source = inventory * diffusion.discretization.cell_volumes
    return {
        "fv_source": source,
        "fv_gradient": gradient,
        "fv_vector_flux": -jnp.einsum("ij,nj->ni", tensor, gradient),
        "fv_local_flux": local,
        "fv_owner_flux": owner,
        "fv_physical_residual": diffusion.residual(
            cells, traces, tensor, boundary, source=source
        ),
        "fv_flux_balance": diffusion.cell_divergence(owner) - source,
    }


@eqx.filter_jit
def _material_vem_residual_history(
    operator: phx.linalg.AbstractLinearOperator,
    values: jax.Array,
    load: jax.Array,
) -> jax.Array:
    return jax.vmap(operator.mv)(values) - load


def _material_seed_bank(
    owners: polyhedral_consumer_owner.PolyhedralArchiveOwners,
    accepted: dict[str, jax.Array],
    *,
    initial_state: bool,
) -> dict[str, jax.Array]:
    seeds = dict(accepted)
    if not initial_state:
        required = (
            "inventory", "previous_inventory", "inventory_history",
            "vem_solution", "vem_solution_history",
            "fv_affine_solution", "fv_affine_solution_history",
            "fv_solution", "fv_solution_history", "fem_solution", "fem_solution_history",
        )
        if any(name not in seeds for name in required):
            raise ValueError("Accepted material continuation must retain every actual primitive and historical seed; got a missing bank.")
        return seeds
    if "inventory_history" not in seeds:
        seeds["inventory_history"] = jnp.broadcast_to(seeds["inventory"], (3,) + seeds["inventory"].shape)
    for name, count in (
        ("vem_solution", owners.vem.dof_count),
        ("fv_affine_solution", owners.hybrid.cell_count + owners.hybrid.face_count),
        ("fem_solution", owners.source_star.space.dof_maps[0].global_dof_count),
    ):
        if name not in seeds:
            seeds[name] = jnp.zeros(count, dtype=jnp.float64)
        if name + "_history" not in seeds:
            seeds[name + "_history"] = jnp.broadcast_to(seeds[name], (3,) + seeds[name].shape)
    return seeds


def _material_derived_seed_history(
    seeds: dict[str, jax.Array],
    operator: phx.linalg.AbstractLinearOperator,
    load: jax.Array,
    affine_solution: jax.Array,
    diffusion: HybridMimeticDiffusion,
    tensor: jax.Array,
    boundary: HybridDiffusionBoundary,
) -> None:
    if "fv_solution" not in seeds:
        seeds["fv_solution"] = affine_solution
    if "fv_solution_history" not in seeds:
        seeds["fv_solution_history"] = jnp.broadcast_to(seeds["fv_solution"], (3,) + seeds["fv_solution"].shape)
    if "vem_physical_load_history" not in seeds:
        seeds["vem_physical_load_history"] = jnp.broadcast_to(load, (3,) + load.shape)
    if "vem_physical_residual_history" not in seeds:
        seeds["vem_physical_residual_history"] = _material_vem_residual_history(
            operator, seeds["vem_solution_history"], load
        )
    names = (
        "fv_source", "fv_gradient", "fv_vector_flux", "fv_local_flux",
        "fv_owner_flux", "fv_physical_residual", "fv_flux_balance",
    )
    missing = tuple(name for name in names if name + "_history" not in seeds)
    if missing:
        derived = jax.vmap(
            _material_fv_physical_fields, in_axes=(None, None, None, 0, 0)
        )(
            diffusion, tensor, boundary, seeds["inventory_history"],
            seeds["fv_solution_history"],
        )
        for name in missing:
            seeds[name + "_history"] = derived[name]


def _polyhedral_physics(
    result: phx.meshing.CellMeshingResult,
    remapped: np.ndarray,
    repeats: int,
    stages: dict[str, float],
    accepted_fields: dict[str, jax.Array],
    *,
    transfer_evidence: dict[str, Any] | None = None,
    budget: CoordinateEnclosureBudget | None = None,
    owners: polyhedral_consumer_owner.PolyhedralArchiveOwners | None = None,
    initial_state: bool = False,
) -> _PolyhedralTransferredPhysics:

    geometry = result.geometry
    mesh = result.mesh
    connectivity = mesh.connectivity
    if not isinstance(connectivity, phx.discretization.PolyhedralConnectivity):
        raise TypeError("Material physical consumers require their actual polyhedral topology.")
    if owners is None:
        _, owners = polyhedral_field_owner.prepare_meshing_polyhedral_owner_binding(
            result, budget=budget
        )
    if geometry is None or owners.source_star.source_mesh_id != result.mesh.mesh_id or owners.source_star.source_geometry_id != cell_geometry_id(geometry):
        raise ValueError("Material physics must consume its actual authenticated polyhedral owners.")
    seeds = _material_seed_bank(owners, accepted_fields, initial_state=initial_state)

    prepared, measured = _measured(lambda: prepare_vem_system(owners.vem))
    vem, operator, mass = prepared
    stages["vem_preparation"] = measured["wall_seconds"]
    values, vem_compiler, timings = _compiled_polyhedral_stages(
        phx.linalg.LinearSystem(operator),
        mass,
        seeds["vem_solution"],
        repeats,
    )
    stages.update({f"vem_{name}": value for name, value in timings.items()})
    vem_error = float(jnp.max(jnp.abs(values - 1.0)))
    prepared, measured = _measured(lambda: prepare_fv_system(owners.hybrid))
    fv, diffusion, tensor, boundary = prepared
    stages["fv_preparation"] = measured["wall_seconds"]
    if not np.array_equal(np.asarray(seeds["inventory"]), remapped):
        raise ValueError(
            "polyhedral.accepted_inventory must be the archived remapped material state; got changed values"
        )
    system_rhs, measured = _measured(lambda: diffusion.linear_system(tensor, boundary))
    system, rhs = system_rhs
    stages["fv_system_preparation"] = measured["wall_seconds"]
    initial, fv_compiler, timings = _compiled_polyhedral_stages(
        system, rhs, seeds["fv_affine_solution"], repeats
    )
    stages.update({f"fv_{name}": value for name, value in timings.items()})
    exact = 1.0 + fv.cell_centers @ jnp.asarray((1.0, 2.0, 3.0), dtype=jnp.float64)
    fv_error = float(jnp.max(jnp.abs(initial[: diffusion.cell_count] - exact)))
    _material_derived_seed_history(
        seeds, operator, mass, initial, diffusion, tensor, boundary
    )
    integrated_source = jnp.asarray(remapped) * fv.cell_volumes
    system_rhs, measured = _measured(
        lambda: diffusion.linear_system(tensor, boundary, source=integrated_source)
    )
    update_system, update_rhs = system_rhs
    stages["fv_transferred_system_preparation"] = measured["wall_seconds"]
    updated, update_compiler, timings = _compiled_polyhedral_stages(
        update_system, update_rhs, seeds["fv_solution"], repeats
    )
    stages.update({f"fv_transferred_{name}": value for name, value in timings.items()})
    residual = diffusion.residual(
        updated[: diffusion.cell_count],
        updated[diffusion.cell_count :],
        tensor,
        boundary,
        source=integrated_source,
    )
    residual_error = float(jnp.max(jnp.abs(residual)))
    response = float(
        jnp.max(
            jnp.abs(updated[: diffusion.cell_count] - initial[: diffusion.cell_count])
        )
    )
    normals = np.asarray(fv.area_vectors)
    offsets, faces = (
        np.asarray(connectivity.cell_face_offsets),
        np.asarray(connectivity.cell_face_values),
    )
    rank_margin = min(
        float(
            np.linalg.svd(
                normals[faces[a:b]] / float(fv.cell_volumes[cell]), compute_uv=False
            )[-1]
        )
        for cell, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True))
    )
    cells, traces = updated[: diffusion.cell_count], updated[diffusion.cell_count :]
    gradient = diffusion.gradient(cells, traces)
    vector_flux = -jnp.einsum("ij,nj->ni", tensor, gradient)
    local_flux = diffusion.local_fluxes(cells, traces, tensor)
    owner_flux = diffusion.owner_rates(local_flux)
    continuity = diffusion.continuity_residual(local_flux)
    interior = np.asarray(fv.neighbor_cells) >= 0
    continuity_error = float(jnp.max(jnp.abs(continuity[jnp.asarray(interior)]))) if np.any(interior) else 0.0
    balance = diffusion.cell_divergence(owner_flux) - integrated_source
    balance_error = float(jnp.max(jnp.abs(balance)))
    field_values = {
        "inventory": jnp.asarray(remapped),
        "vem_solution": values,
        "vem_physical_load": mass,
        "vem_physical_residual": operator.mv(values) - mass,
        "fv_affine_solution": initial,
        "fv_solution": updated,
        "fv_source": integrated_source,
        "fv_gradient": gradient,
        "fv_vector_flux": vector_flux,
        "fv_local_flux": local_flux,
        "fv_owner_flux": owner_flux,
        "fv_physical_residual": residual,
        "fv_flux_balance": balance,
    }
    vem_residual_error = float(jnp.max(jnp.abs(field_values["vem_physical_residual"])))
    fields = dict(field_values)
    for name, value in field_values.items():
        fields[name + "_history"] = _polyhedral_qualification_field_history(seeds, name, value)
    fields["previous_inventory"] = jnp.asarray(seeds["previous_inventory"])
    if (
        vem.evidence.minimum_rank_margin <= 0.0
        or rank_margin <= 0.0
        or vem.evidence.maximum_reproduction_defect > 1e-7
        or response <= 1e-10
        or not np.isfinite(response)
    ):
        raise qualification_failure(
            "_polyhedral_physics",
            ("Polyhedral consumer rank or transferred-source physics response failed."),
            {"rank_margin": rank_margin, "response": response, "vem": vem},
        )
    retained = (
        vem,
        operator,
        mass,
        values,
        fv,
        diffusion,
        tensor,
        boundary,
        system,
        rhs,
        update_system,
        update_rhs,
        initial,
        updated,
        integrated_source,
    )
    physics_record: dict[str, Any] = {
        "canonical_state_rebind": {
            "published": None if transfer_evidence is None else transfer_evidence["published"],
            "receipt_id": None if transfer_evidence is None else transfer_evidence["receipt_id"],
            "roles": ("physical-state", "history"),
            "target_result_id": result.result_id,
            "accepted_state_consumed": True,
            "initial_state_admitted": initial_state,
            "accepted_previous_inventory_fingerprint": array_tree_fingerprint(fields["previous_inventory"]),
        },
        "vem": {
            "compiler": vem_compiler,
            "constant_solution_error": vem_error,
            "minimum_rank_margin": vem.evidence.minimum_rank_margin,
            "maximum_reproduction_defect": vem.evidence.maximum_reproduction_defect,
            "maximum_original_physical_residual": vem_residual_error,
        },
        "fv": {
            "compiler": fv_compiler,
            "linear_solution_error": fv_error,
            "domain_measure": float(jnp.sum(fv.cell_volumes)),
            "minimum_normal_rank_margin": rank_margin,
            "maximum_nonorthogonality_degrees": float(
                fv.quality.maximum_nonorthogonality_degrees
            ),
            "maximum_closure_residual": float(fv.quality.maximum_closure_residual),
            "transferred_update_compiler": update_compiler,
            "transferred_source_residual": residual_error,
            "transferred_source_integral": float(jnp.sum(integrated_source)),
            "transferred_state_physical_response": response,
            "maximum_interior_flux_continuity_defect": continuity_error,
            "maximum_cell_flux_balance_defect": balance_error,
            "vector_flux_finite": bool(jnp.all(jnp.isfinite(vector_flux))),
            "net_outward_flux": float(jnp.sum(diffusion.cell_divergence(owner_flux))),
        },
    }
    fem_record, fem_error, fem_fields = _polyhedral_fem_physics(
        result, seeds, jnp.asarray(remapped), stages, budget=budget,
        prepared_plan=owners.source_star,
        repeats=repeats,
    )
    fem_record["restored_or_transferred_seed_consumed"] = not initial_state
    fields.update(fem_fields)
    physics_record["fe"] = fem_record
    return _PolyhedralTransferredPhysics(
        record=physics_record,
        error=max(
            vem_error, fv_error, residual_error, continuity_error, balance_error, fem_error,
            vem_residual_error,
        ),
        retained=(*retained, *fields.values()),
        domain_measure=physics_record["fv"]["domain_measure"],
        transferred_source_integral=physics_record["fv"]["transferred_source_integral"],
        fields=fields,
        owners=owners,
    )


def _polyhedral_qualification_halfcell_solution(x: Any, amplitude: float) -> np.ndarray:
    """Independent periodic solution of -u''=(-a,+a), before gauge shift."""
    x = np.mod(np.asarray(x, dtype=np.float64), 1.0)
    right = x - 0.5
    return np.where(
        x <= 0.5,
        amplitude * (0.5 * x * x - 0.25 * x),
        amplitude * (-0.5 * right * right + 0.25 * right),
    )


def _polyhedral_qualification_field_history(
    fields: dict[str, Any], name: str, value: Any
) -> jax.Array:
    """Retain three successive field states."""
    value = jnp.asarray(value)
    old = fields.get(name + "_history")
    if old is None:
        # Initial history is the admitted starting seed, not the newly solved state.
        seed = jnp.asarray(fields.get(name, jnp.zeros_like(value)), dtype=value.dtype)
        old = jnp.broadcast_to(seed, (3,) + value.shape)
    old = jnp.asarray(old)
    if old.shape != (3,) + value.shape or old.dtype != value.dtype:
        raise ValueError(
            f"polyhedral.field_history must have shape {(3,) + value.shape} and dtype {value.dtype}; got field {name + '_history'!r}, shape {old.shape}, dtype {old.dtype}"
        )
    return jnp.concatenate((old[1:], value[None]), axis=0)


@eqx.filter_jit
def _polyhedral_qualification_operator_defects(
    operator: phx.linalg.AbstractLinearOperator,
    rhs: jax.Array,
    solution: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    residual = operator.mv(solution) - rhs
    size = solution.size
    left = jnp.sin(jnp.arange(size, dtype=solution.dtype) + 0.3)
    right = jnp.cos(jnp.arange(size, dtype=solution.dtype) + 0.7)
    transpose_defect = jnp.abs(
        jnp.vdot(left, operator.mv(right))
        - jnp.vdot(operator.transpose_mv(left), right)
    )
    return jnp.max(jnp.abs(residual)), transpose_defect


def _polyhedral_qualification_operator_evidence(
    operator: phx.linalg.AbstractLinearOperator,
    rhs: jax.Array,
    solution: jax.Array,
) -> dict[str, float]:
    residual, transpose = _polyhedral_qualification_operator_defects(
        operator, rhs, solution
    )
    return {
        "maximum_algebraic_residual": float(residual),
        "raw_transpose_pairing_defect": float(transpose),
    }


@eqx.filter_jit
def _polyhedral_reconstruction_defects(
    operator: phx.linalg.AbstractLinearOperator,
    source_points: jax.Array,
    target_points: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    source_basis = jnp.concatenate(
        (jnp.ones((source_points.shape[0], 1), dtype=source_points.dtype), source_points), axis=1
    )
    target_basis = jnp.concatenate(
        (jnp.ones((target_points.shape[0], 1), dtype=target_points.dtype), target_points), axis=1
    )
    image = jax.vmap(operator.mv, in_axes=1, out_axes=1)(source_basis)
    difference = jnp.abs(image - target_basis)
    return jnp.max(difference[:, 0]), jnp.max(difference[:, 1:])


def _polyhedral_qualification_reprepare_fv_seed(
    diffusion: Any,
    tensor: Any,
    boundary: Any,
    fields: dict[str, Any],
    repeats: int,
    stages: dict[str, float],
) -> tuple[jax.Array, jax.Array, dict[str, Any]]:
    """Reprepare current and three historical traces from transported cell states.

    Full restored seeds are handled by the caller without reconstruction.
    On a topology change, each fixed-cell trace RHS comes from the
    mimetic stationarity equations rather than nodal/face interpolation.
    """

    cell_seed = jnp.asarray(fields["fv_cell_solution"])
    cell_history = jnp.asarray(fields["fv_cell_solution_history"])
    if (
        cell_seed.shape != (diffusion.cell_count,)
        or cell_history.shape != (3, diffusion.cell_count)
        or cell_history.dtype != cell_seed.dtype
    ):
        raise ValueError(
            f"fields.fv_cell_solution_history must match cell seed shape {(diffusion.cell_count,)} and dtype {cell_seed.dtype} across three states; got seed shape {cell_seed.shape}, history shape {cell_history.shape}, dtype {cell_history.dtype}"
        )
    all_cells = (cell_seed, *tuple(cell_history[index] for index in range(3)))
    full_values, evidence = [], []
    prepared, measured = _measured(
        lambda: diffusion.trace_reconstruction_operators(tensor, boundary)
    )
    stages["periodic_fv_trace_reprepare_preparation"] = measured["wall_seconds"]
    system, cell_rhs_action, rhs_offset = prepared
    for index, cells in enumerate(all_cells):
        rhs, measured = _measured(
            lambda cells=cells: jax.block_until_ready(
                cell_rhs_action.mv(cells) + rhs_offset
            )
        )
        stages[f"periodic_fv_trace_reprepare_{index}_rhs"] = measured["wall_seconds"]
        trace, compiler, timings = _compiled_polyhedral_stages(
            system, rhs, jnp.zeros_like(rhs), repeats
        )
        stages.update(
            {
                f"periodic_fv_trace_reprepare_{index}_{name}": value
                for name, value in timings.items()
            }
        )
        full_values.append(jnp.concatenate((cells.astype(trace.dtype), trace)))
        local = diffusion.local_fluxes(cells, trace, tensor)
        evidence.append(
            {
                "state": "current" if index == 0 else f"history-{index - 1}",
                "compiler": compiler,
                **_polyhedral_qualification_operator_evidence(
                    system.operator, rhs, trace
                ),
                "maximum_face_stationarity_residual": float(
                    jnp.max(
                        jnp.abs(
                            boundary.face_residual(
                                trace, diffusion.continuity_residual(local)
                            )
                        )
                    )
                ),
            }
        )
    return (
        full_values[0],
        jnp.stack(full_values[1:]),
        {
            "authority": "HybridMimeticDiffusion.trace_reconstruction_operators fixed-cell physical stationarity",
            "coefficient_preparation_count": 1,
            "current_and_history": evidence,
            "facet_interpolation_used": False,
        },
    )


@final
class _PeriodicPhysicsContext(phx.StrictModule):
    result: phx.meshing.CellMeshingResult
    fields: dict[str, jax.Array]
    seeds: dict[str, jax.Array]
    repeats: int
    stages: dict[str, float]
    measurements: dict[str, Any]
    ledger: StageLedger
    budget: CoordinateEnclosureBudget | None
    owners: polyhedral_consumer_owner.PolyhedralArchiveOwners


@final
class _PeriodicPhysicalSource(phx.StrictModule):
    fv: phx.discretization.UnstructuredFiniteVolumeDiscretization
    diffusion: HybridMimeticDiffusion
    boundary: HybridDiffusionBoundary
    tensor: jax.Array
    volumes: np.ndarray
    density: np.ndarray
    mean: float
    force: jax.Array
    integrated_source: jax.Array
    source_vertices: np.ndarray
    cell_vertices: tuple[np.ndarray, ...]
    quadrature_points: tuple[np.ndarray, ...]
    quadrature_weights: tuple[np.ndarray, ...]
    source_volumes: np.ndarray
    volume_defect: float
    amplitude: float
    force_defect: float
    analytic_averages: np.ndarray
    fv_gauge: float


def _prepare_periodic_physical_source(
    context: _PeriodicPhysicsContext, conserved: jax.Array
) -> _PeriodicPhysicalSource:
    result = context.result
    stages = context.stages
    ledger = context.ledger
    certification = result.certification
    if certification is None:
        raise ValueError("Periodic physical preparation requires accepted source certification.")
    fv, measured = ledger.run(
        "prepare",
        lambda: _measured(lambda: context.owners.finite_volume),
    )
    stages["periodic_fv_preparation"] = measured["wall_seconds"]
    diffusion, boundary = context.owners.hybrid, HybridDiffusionBoundary(fv)
    tensor = jnp.eye(3, dtype=fv.cell_volumes.dtype)
    volumes = np.asarray(fv.cell_volumes)
    density = np.asarray(conserved[:, 0])
    mean = float(np.dot(density, volumes) / np.sum(volumes))
    force = jnp.asarray(density - mean, dtype=fv.cell_volumes.dtype)
    integrated_source = force * fv.cell_volumes
    source_data, measured = ledger.run(
        "_polyhedral_qualification_source_quadrature",
        lambda: _measured(
            lambda: prepare_periodic_source_quadrature(
                result, failure_factory=qualification_failure, budget=context.budget
            )
        ),
    )
    stages["periodic_independent_source_quadrature"] = measured["wall_seconds"]
    source_vertices, cell_vertices, quadrature_points, quadrature_weights = (
        source_data.points,
        source_data.cell_vertices,
        source_data.cell_points,
        source_data.cell_weights,
    )
    source_volumes = np.asarray([np.sum(w) for w in quadrature_weights])
    volume_defect = float(np.max(np.abs(source_volumes - volumes)))
    if volume_defect > 1e-10:
        raise qualification_failure(
            "_polyhedral_qualification_periodic_physics",
            ("Prepared FV volume disagrees with independent source-volume quadrature."),
            {"volume_defect": volume_defect},
        )
    sides = []
    for ids in cell_vertices:
        xs = source_vertices[ids, 0]
        if np.max(xs) <= 0.5:
            sides.append(False)
        elif np.min(xs) >= 0.5:
            sides.append(True)
        else:
            raise qualification_failure(
                "_polyhedral_qualification_periodic_physics",
                ("Independent halfcell force requires source cells split at x=0.5."),
                {},
            )
    sides = np.asarray(sides, dtype=np.bool_)
    if not np.any(sides) or np.all(sides):
        raise qualification_failure(
            "_polyhedral_qualification_periodic_physics",
            ("Source must retain both physical density halfcells."),
            {"sides": sides},
        )
    amplitude = float(
        (
            np.average(density[sides], weights=volumes[sides])
            - np.average(density[~sides], weights=volumes[~sides])
        )
        / 2
    )
    force_defect = float(
        np.max(np.abs(np.asarray(force) - np.where(sides, amplitude, -amplitude)))
    )
    if amplitude == 0 or not np.isfinite(amplitude) or force_defect > 1e-10:
        raise qualification_failure(
            "_polyhedral_qualification_periodic_physics",
            (
                "Transported density is not the compatible nonconstant physical halfcell state."
            ),
            {"amplitude": amplitude, "force_defect": force_defect},
        )
    analytic_averages = np.asarray(
        [
            np.dot(w, _polyhedral_qualification_halfcell_solution(p[:, 0], amplitude))
            / np.sum(w)
            for p, w in zip(quadrature_points, quadrature_weights, strict=True)
        ]
    )
    fv_gauge = float(analytic_averages[0])
    return _PeriodicPhysicalSource(
        fv=fv,
        diffusion=diffusion,
        boundary=boundary,
        tensor=tensor,
        volumes=volumes,
        density=density,
        mean=mean,
        force=force,
        integrated_source=integrated_source,
        source_vertices=source_vertices,
        cell_vertices=cell_vertices,
        quadrature_points=quadrature_points,
        quadrature_weights=quadrature_weights,
        source_volumes=source_volumes,
        volume_defect=volume_defect,
        amplitude=amplitude,
        force_defect=force_defect,
        analytic_averages=analytic_averages,
        fv_gauge=fv_gauge,
    )


@final
class _PeriodicFiniteVolumeSolve(phx.StrictModule):
    system: phx.linalg.LinearSystem
    rhs: jax.Array
    seed_fields: dict[str, jax.Array]
    trace_repreparation: dict[str, Any] | None
    fv_solution: jax.Array
    fv_compiler: dict[str, object]
    cells: jax.Array
    traces: jax.Array
    physical_residual: jax.Array
    local_fluxes: jax.Array
    owner_rates: jax.Array
    continuity: jax.Array
    balance: jax.Array
    fv_error: float
    fv_average_error: float


def _solve_periodic_finite_volume(
    context: _PeriodicPhysicsContext, source: _PeriodicPhysicalSource
) -> _PeriodicFiniteVolumeSolve:
    result = context.result
    fields = context.fields
    seeds = context.seeds
    repeats = context.repeats
    stages = context.stages
    ledger = context.ledger
    diffusion = source.diffusion
    boundary = source.boundary
    tensor = source.tensor
    integrated_source = source.integrated_source
    quadrature_points = source.quadrature_points
    quadrature_weights = source.quadrature_weights
    source_volumes = source.source_volumes
    amplitude = source.amplitude
    analytic_averages = source.analytic_averages
    fv_gauge = source.fv_gauge
    count = result.mesh.connectivity.cell_count
    system_rhs, measured = ledger.run(
        "pinned_linear_system",
        lambda: _measured(
            lambda: diffusion.pinned_linear_system(
                tensor, boundary, source=integrated_source, pin_cell=0, pin_value=0.0
            )
        ),
    )
    stages["periodic_fv_system_preparation"] = measured["wall_seconds"]
    system, rhs = system_rhs
    seed_fields = {**fields, **seeds}
    trace_repreparation = None
    if "fv_solution" not in seed_fields and "fv_cell_solution" in seed_fields:
        full, histories, trace_repreparation = (
            _polyhedral_qualification_reprepare_fv_seed(
                diffusion, tensor, boundary, seed_fields, repeats, stages
            )
        )
        seed_fields["fv_solution"], seed_fields["fv_solution_history"] = full, histories
        seed_fields["fv_trace_solution_history"] = histories[:, count:]
    fv_seed = jnp.asarray(
        seed_fields.get("fv_solution", jnp.zeros_like(rhs)), dtype=rhs.dtype
    )
    if fv_seed.shape != rhs.shape:
        raise ValueError(
            f"fields.fv_solution must have quotient cell/trace shape {rhs.shape}; got {fv_seed.shape}"
        )
    fv_solution, fv_compiler, timings = _compiled_polyhedral_stages(
        system, rhs, fv_seed, repeats
    )
    stages.update({"periodic_fv_" + key: value for key, value in timings.items()})
    cells, traces = fv_solution[:count], fv_solution[count:]
    physical_residual = diffusion.residual(
        cells, traces, tensor, boundary, source=integrated_source
    )
    local_fluxes = diffusion.local_fluxes(cells, traces, tensor)
    owner_rates = diffusion.owner_rates(local_fluxes)
    continuity = diffusion.continuity_residual(local_fluxes)
    balance = diffusion.cell_divergence(owner_rates) - integrated_source
    fv_squared_error = sum(
        float(
            np.dot(
                w,
                (
                    float(cells[cell])
                    - (
                        _polyhedral_qualification_halfcell_solution(p[:, 0], amplitude)
                        - fv_gauge
                    )
                )
                ** 2,
            )
        )
        for cell, (p, w) in enumerate(
            zip(quadrature_points, quadrature_weights, strict=True)
        )
    )
    fv_error = float(np.sqrt(fv_squared_error / np.sum(source_volumes)))
    fv_average_error = float(
        np.max(np.abs(np.asarray(cells) - (analytic_averages - fv_gauge)))
    )
    return _PeriodicFiniteVolumeSolve(
        system=system,
        rhs=rhs,
        seed_fields=seed_fields,
        trace_repreparation=trace_repreparation,
        fv_solution=fv_solution,
        fv_compiler=fv_compiler,
        cells=cells,
        traces=traces,
        physical_residual=physical_residual,
        local_fluxes=local_fluxes,
        owner_rates=owner_rates,
        continuity=continuity,
        balance=balance,
        fv_error=fv_error,
        fv_average_error=fv_average_error,
    )


@final
class _PeriodicVirtualElementSolve(phx.StrictModule):
    vem: PreparedPolyhedralH1VirtualElement3D
    load: jax.Array
    compatibility_tolerance: float
    compatibility_bound: ZeroMeanLoadCompatibilityBound
    pinned_load: jax.Array
    operator: phx.linalg.AbstractLinearOperator
    solved_free: jax.Array
    vem_compiler: dict[str, object]
    vem_solution: jax.Array
    vem_gauge: float
    exact_nodal: np.ndarray
    vem_error: float
    vem_physical_residual: jax.Array


def _solve_periodic_virtual_element(
    context: _PeriodicPhysicsContext, source: _PeriodicPhysicalSource
) -> _PeriodicVirtualElementSolve:
    result = context.result
    fields = context.fields
    seeds = context.seeds
    repeats = context.repeats
    stages = context.stages
    ledger = context.ledger
    force = source.force
    source_vertices = source.source_vertices
    cell_vertices = source.cell_vertices
    source_volumes = source.source_volumes
    amplitude = source.amplitude
    vem, measured = ledger.run(
        "prepare_polyhedral_h1_virtual_element_3d",
        lambda: _measured(lambda: context.owners.vem),
    )
    stages["periodic_vem_preparation"] = measured["wall_seconds"]
    load = vem.assemble_cellwise_constant_load(force)
    # Approved contract: only the a-priori binary64 rounding bound of the mean
    # subtraction, load assembly and fsum reduction is admitted; the mean and
    # force provenance are checked, so physical incompatibility still refuses.
    compatibility_bound = vem.zero_mean_load_compatibility_bound(
        source.density, source.volumes, source.mean, np.asarray(force)
    )
    compatibility_tolerance = compatibility_bound.tolerance
    pinned_load = vem.compatible_pinned_load(
        load, 0, compatibility_tolerance=compatibility_tolerance
    )
    operator = vem.pinned_diffusion_operator(0)
    vem_seed = jnp.asarray(
        seeds.get(
            "vem_solution",
            fields.get(
                "vem_solution", jnp.zeros((vem.dof_count,), dtype=pinned_load.dtype)
            ),
        ),
        dtype=pinned_load.dtype,
    )
    if vem_seed.shape != (vem.dof_count,):
        raise ValueError(
            f"fields.vem_solution must have quotient nodal shape {(vem.dof_count,)}; got {vem_seed.shape}"
        )
    free = jnp.arange(1, vem.dof_count, dtype=jnp.int32)
    solved_free, vem_compiler, timings = _compiled_polyhedral_stages(
        phx.linalg.LinearSystem(operator), pinned_load, vem_seed[free], repeats
    )
    stages.update({"periodic_vem_" + key: value for key, value in timings.items()})
    vem_solution = jnp.zeros_like(vem_seed).at[free].set(solved_free)
    periodic = result.mesh.periodic_topology
    if periodic is None:
        raise qualification_failure(
            "_solve_periodic_virtual_element",
            "Periodic VEM mesh.periodic_topology must retain quotient orbits; got None",
            {"result": result},
        )
    representatives = np.asarray(periodic.orbit_representatives(0), dtype=np.int64)
    exact_nodal = _polyhedral_qualification_halfcell_solution(
        source_vertices[representatives, 0], amplitude
    )
    vem_gauge = float(exact_nodal[0])
    exact_nodal -= vem_gauge
    nodal_weights = np.zeros((vem.dof_count,), dtype=np.float64)
    route = np.asarray(periodic.orbits(0)[0], dtype=np.int64)
    for ids, volume in zip(cell_vertices, source_volumes, strict=True):
        np.add.at(nodal_weights, route[ids], volume / len(ids))
    vem_error = float(
        np.sqrt(
            np.dot(nodal_weights, (np.asarray(vem_solution) - exact_nodal) ** 2)
            / np.sum(source_volumes)
        )
    )
    vem_physical_residual = vem.mv(vem_solution) - load
    return _PeriodicVirtualElementSolve(
        vem=vem,
        load=load,
        compatibility_tolerance=compatibility_tolerance,
        compatibility_bound=compatibility_bound,
        pinned_load=pinned_load,
        operator=operator,
        solved_free=solved_free,
        vem_compiler=vem_compiler,
        vem_solution=vem_solution,
        vem_gauge=vem_gauge,
        exact_nodal=exact_nodal,
        vem_error=vem_error,
        vem_physical_residual=vem_physical_residual,
    )


@final
class _PeriodicPhysicalResponse(phx.StrictModule):
    updated: dict[str, jax.Array]
    fv_response: float
    vem_response: float
    vem_load_nonzero: bool


def _reanalyse_periodic_history(
    context: _PeriodicPhysicsContext,
    source: _PeriodicPhysicalSource,
    updated: dict[str, jax.Array],
) -> None:
    """Reevaluate dual and flux histories, never interpolate them as primals."""
    density_history = np.asarray(updated["conserved_history"])[:, :, 0]
    volumes = np.asarray(source.fv.cell_volumes)
    means = np.asarray([
        np.dot(density, volumes) / np.sum(volumes) for density in density_history
    ], dtype=np.float64)
    forces = jnp.asarray(density_history - means[:, None], dtype=jnp.float64)
    fv_history = jax.vmap(
        _material_fv_physical_fields, in_axes=(None, None, None, 0, 0)
    )(
        source.diffusion, source.tensor, source.boundary,
        forces, updated["fv_solution_history"],
    )
    updated["physical_force_history"] = forces
    updated["physical_source_load_history"] = fv_history["fv_source"]
    count = source.diffusion.cell_count
    updated["fv_cell_solution_history"] = updated["fv_solution_history"][:, :count]
    updated["fv_trace_solution_history"] = updated["fv_solution_history"][:, count:]
    for name in ("fv_local_flux", "fv_owner_flux", "fv_physical_residual", "fv_flux_balance"):
        updated[name + "_history"] = fv_history[name]
    vem = context.owners.vem
    loads = []
    bounds = []
    for density, mean, force in zip(density_history, means, forces, strict=True):
        load = vem.assemble_cellwise_constant_load(force)
        bound = vem.zero_mean_load_compatibility_bound(
            density, volumes, float(mean), force
        )
        measured = math.fsum(map(float, np.asarray(load)))
        if abs(measured) > bound.tolerance:
            raise ValueError("Historical VEM load must satisfy its original physical compatibility admission.")
        loads.append(load)
        bounds.append({
            "bound_id": bound.bound_id, "inputs_id": bound.inputs_id,
            "observed_total": measured, "a_priori_bound": bound.tolerance,
        })
    updated["vem_physical_load_history"] = jnp.stack(loads)
    updated["vem_physical_residual_history"] = _material_vem_residual_history(
        vem.as_linear_operator(), updated["vem_solution_history"],
        updated["vem_physical_load_history"],
    )
    context.measurements["history_reanalysis"] = {
        "semantics": "actual retained primitive and Euler coefficient histories evaluated by their native current-carrier owners",
        "historic_equilibrium_claimed": False,
        "vem_compatibility": bounds,
    }


def _retain_periodic_physical_fields(
    context: _PeriodicPhysicsContext,
    source: _PeriodicPhysicalSource,
    finite_volume: _PeriodicFiniteVolumeSolve,
    virtual_element: _PeriodicVirtualElementSolve,
) -> _PeriodicPhysicalResponse:
    fields = context.fields
    force = source.force
    integrated_source = source.integrated_source
    analytic_averages = source.analytic_averages
    fv_gauge = source.fv_gauge
    seed_fields = finite_volume.seed_fields
    fv_solution = finite_volume.fv_solution
    cells = finite_volume.cells
    traces = finite_volume.traces
    physical_residual = finite_volume.physical_residual
    local_fluxes = finite_volume.local_fluxes
    owner_rates = finite_volume.owner_rates
    balance = finite_volume.balance
    load = virtual_element.load
    pinned_load = virtual_element.pinned_load
    vem_solution = virtual_element.vem_solution
    exact_nodal = virtual_element.exact_nodal
    vem_physical_residual = virtual_element.vem_physical_residual
    updated = dict(fields)
    fv_response = float(np.ptp(np.asarray(cells)))
    vem_response = float(np.ptp(np.asarray(vem_solution)))
    vem_load_nonzero = bool(np.any(np.asarray(pinned_load) != 0))
    if (
        fv_response <= 1e-10
        or (vem_load_nonzero and vem_response <= 1e-10)
        or not np.all(np.isfinite(np.asarray(fv_solution)))
        or not np.all(np.isfinite(np.asarray(vem_solution)))
    ):
        raise qualification_failure(
            "_polyhedral_qualification_periodic_physics",
            (
                "Quotient physical solve did not retain its required finite physical response."
            ),
            {
                "fv_response": fv_response,
                "fv_solution": fv_solution,
                "vem_load_nonzero": vem_load_nonzero,
                "vem_response": vem_response,
                "vem_solution": vem_solution,
            },
        )
    for name, value in (("fv_solution", fv_solution), ("vem_solution", vem_solution)):
        updated[name + "_history"] = _polyhedral_qualification_field_history(
            seed_fields, name, value
        )
        updated[name] = value
    updated.update(
        physical_force=force,
        physical_source_load=integrated_source,
        fv_cell_solution=cells,
        fv_trace_solution=traces,
        fv_owner_flux=owner_rates,
        fv_local_flux=local_fluxes,
        fv_physical_residual=physical_residual,
        fv_flux_balance=balance,
        vem_physical_load=load,
        vem_physical_residual=vem_physical_residual,
        fv_analytic_cell_average=jnp.asarray(analytic_averages - fv_gauge),
        vem_analytic_nodal=jnp.asarray(exact_nodal),
    )
    _reanalyse_periodic_history(context, source, updated)
    return _PeriodicPhysicalResponse(
        updated=updated,
        fv_response=fv_response,
        vem_response=vem_response,
        vem_load_nonzero=vem_load_nonzero,
    )


def _record_periodic_physical_evidence(
    context: _PeriodicPhysicsContext,
    source: _PeriodicPhysicalSource,
    finite_volume: _PeriodicFiniteVolumeSolve,
    virtual_element: _PeriodicVirtualElementSolve,
    response: _PeriodicPhysicalResponse,
    conserved: jax.Array,
    history: jax.Array,
) -> tuple[dict[str, Any], float, dict[str, jax.Array]]:
    result = context.result
    fields = context.fields
    seeds = context.seeds
    measurements = context.measurements
    diffusion = source.diffusion
    mean = source.mean
    force = source.force
    integrated_source = source.integrated_source
    source_volumes = source.source_volumes
    volume_defect = source.volume_defect
    amplitude = source.amplitude
    force_defect = source.force_defect
    fv_gauge = source.fv_gauge
    system = finite_volume.system
    rhs = finite_volume.rhs
    trace_repreparation = finite_volume.trace_repreparation
    fv_solution = finite_volume.fv_solution
    fv_compiler = finite_volume.fv_compiler
    physical_residual = finite_volume.physical_residual
    owner_rates = finite_volume.owner_rates
    continuity = finite_volume.continuity
    balance = finite_volume.balance
    fv_error = finite_volume.fv_error
    fv_average_error = finite_volume.fv_average_error
    vem = virtual_element.vem
    load = virtual_element.load
    compatibility_tolerance = virtual_element.compatibility_tolerance
    pinned_load = virtual_element.pinned_load
    operator = virtual_element.operator
    solved_free = virtual_element.solved_free
    vem_compiler = virtual_element.vem_compiler
    vem_gauge = virtual_element.vem_gauge
    vem_error = virtual_element.vem_error
    vem_physical_residual = virtual_element.vem_physical_residual
    updated = response.updated
    fv_response = response.fv_response
    vem_response = response.vem_response
    vem_load_nonzero = response.vem_load_nonzero
    fv_evidence = _polyhedral_qualification_operator_evidence(
        system.operator, rhs, fv_solution
    )
    vem_evidence = _polyhedral_qualification_operator_evidence(
        operator, pinned_load, solved_free
    )
    physical_defects = [
        float(np.max(np.abs(np.asarray(value))))
        for value in (physical_residual, continuity, balance, vem_physical_residual)
    ]
    physical_error = max(
        fv_error,
        vem_error,
        *physical_defects,
        fv_evidence["raw_transpose_pairing_defect"],
        vem_evidence["raw_transpose_pairing_defect"],
    )
    inventories = np.asarray(conserved).T @ source_volumes
    history_inventories = np.einsum("tnk,n->tk", np.asarray(history), source_volumes)
    record = {
        "source_result_id": result.result_id,
        "source_authority": "actual CellGeometrySpec.source_coordinates and canonical source-face triangulation",
        "numerical_authority": "prepared quotient HybridMimeticDiffusion / polyhedral H1 VEM native CG",
        "physical_force": {
            "law": "transported density minus volume-weighted mean",
            "density_mean": mean,
            "halfcell_amplitude": amplitude,
            "maximum_halfcell_force_defect": force_defect,
            "integrated_source": float(jnp.sum(integrated_source)),
            "independent_source_integral": float(
                np.dot(np.asarray(force), source_volumes)
            ),
        },
        "independent_analytic_solution": {
            "law": "periodic halfcell piecewise quadratic -u''=(-a,+a)",
            "fv_gauge": fv_gauge,
            "vem_gauge": vem_gauge,
            "quadrature": "source-star tetrahedra, tensor Gauss4 Duffy; degree-four exact",
            "domain_measure": float(np.sum(source_volumes)),
            "maximum_volume_defect": volume_defect,
        },
        "inventories": {
            "conserved_columns": [
                "density",
                "momentum_x",
                "momentum_y",
                "momentum_z",
                "total_energy",
                "species_0",
                "species_1",
            ],
            "conserved": inventories.tolist(),
            "conserved_history": history_inventories.tolist(),
            "pressure_volume_integral": float(
                np.dot(np.asarray(fields["pressure"]), source_volumes)
            ),
            "temperature_volume_integral": float(
                np.dot(np.asarray(fields["temperature"]), source_volumes)
            ),
        },
        "fv": {
            "compiler": fv_compiler,
            **fv_evidence,
            "physical_response": fv_response,
            "source_volume_l2_error": fv_error,
            "maximum_cell_average_error": fv_average_error,
            "maximum_original_physical_residual": physical_defects[0],
            "maximum_reciprocal_flux_continuity_defect": physical_defects[1],
            "maximum_cell_flux_balance_defect": physical_defects[2],
            "total_outward_integrated_flux": float(
                jnp.sum(diffusion.cell_divergence(owner_rates))
            ),
            "restored_seed_consumed": "fv_solution" in seeds or "fv_solution" in fields,
            "transferred_cell_seed_consumed": trace_repreparation is not None,
            "trace_repreparation": trace_repreparation,
            "pin_cell": 0,
            "pin_value": 0.0,
        },
        "vem": {
            "compiler": vem_compiler,
            **vem_evidence,
            "physical_response": vem_response,
            "projected_physical_load_nonzero": vem_load_nonzero,
            "response_requirement": "nonzero for nonzero actual weak load; coarse zero projection is retained honestly",
            "source_volume_vertex_lumped_l2_error": vem_error,
            "error_semantics": "actual quotient nodal values, source-volume vertex lumping; not reconstructed interior VEM error",
            "maximum_original_physical_residual": physical_defects[3],
            "compatibility_tolerance": compatibility_tolerance,
            "compatibility_bound": {
                "bound_id": virtual_element.compatibility_bound.bound_id,
                "inputs_id": virtual_element.compatibility_bound.inputs_id,
                "mean": virtual_element.compatibility_bound.mean,
                "mean_reduction_bound": virtual_element.compatibility_bound.mean_reduction_bound,
                "mean_subtraction_bound": virtual_element.compatibility_bound.mean_subtraction_bound,
                "assembly_bound": virtual_element.compatibility_bound.assembly_bound,
                "accumulation_bound": virtual_element.compatibility_bound.accumulation_bound,
                "maximum_incidence": virtual_element.compatibility_bound.maximum_incidence,
                "cell_count": virtual_element.compatibility_bound.cell_count,
                "measured_fsum_total": math.fsum(map(float, np.asarray(load))),
            },
            "total_physical_load": float(jnp.sum(load)),
            "minimum_rank_margin": vem.evidence.minimum_rank_margin,
            "maximum_reproduction_defect": vem.evidence.maximum_reproduction_defect,
            "restored_seed_consumed": "vem_solution" in seeds or "vem_solution" in fields,
            "pin_dof": 0,
            "pin_value": 0.0,
        },
        "retained_field_names": sorted(updated),
        "prepared_solver_archived": False,
        "history_reanalysis": measurements["history_reanalysis"],
    }
    record["stage_measurements"] = measurements["stage_measurements"]
    return record, physical_error, updated


def _polyhedral_qualification_periodic_physics(
    result: phx.meshing.CellMeshingResult,
    fields: dict[str, jax.Array],
    repeats: int,
    stages: dict[str, float],
    *,
    seeds: dict[str, jax.Array] | None = None,
    deadline: Deadline | None = None,
    budget: CoordinateEnclosureBudget | None = None,
    owners: polyhedral_consumer_owner.PolyhedralArchiveOwners | None = None,
) -> tuple[dict[str, Any], float, dict[str, jax.Array]]:
    """Solve quotient FV/VEM systems with native CG and retain all field values.

    fields: conserved[N,7] (rho,mom3,total-energy,species2),
    conserved_history[3,N,7], pressure[N], temperature[N], plus optional
    fv_solution[N+F], vem_solution[Q] and their three-state histories.
    seeds optionally overrides fv_solution/vem_solution; it never alters force.
    Returns (JSON-safe physics_record, physical_error, updated_fields).
    Geometry/SCI restoration and ALLfield transfer remain the caller's owners.
    """
    measurements: dict[str, object] = {}
    _ledger = StageLedger(measurements, deadline=deadline)

    if repeats < 1:
        raise ValueError(f"polyhedral.repeats must be positive; got {repeats!r}")
    seeds = {} if seeds is None else seeds
    if owners is None:
        owners, measured = _ledger.run(
            "canonical_physical_owner_preparation",
            lambda: _measured(
                lambda: polyhedral_field_owner.prepare_meshing_polyhedral_owner_binding(
                    result, budget=budget
                )[1]
            ),
        )
        stages["periodic_physical_owner_preparation"] = measured["wall_seconds"]
    if result.geometry is None or owners.source_star.source_mesh_id != result.mesh.mesh_id or owners.source_star.source_geometry_id != cell_geometry_id(result.geometry):
        raise ValueError("Periodic physics must consume its actual authenticated polyhedral owners.")
    context = _PeriodicPhysicsContext(
        result=result,
        fields=fields,
        seeds=seeds,
        repeats=repeats,
        stages=stages,
        measurements=measurements,
        ledger=_ledger,
        budget=budget,
        owners=owners,
    )
    conserved = jnp.asarray(fields["conserved"])
    history = jnp.asarray(fields["conserved_history"])
    count = result.mesh.connectivity.cell_count
    if (
        conserved.shape != (count, 7)
        or history.shape != (3, count, 7)
        or history.dtype != conserved.dtype
    ):
        raise ValueError(
            f"fields.conserved_history must match conserved shape {(count, 7)} and dtype {conserved.dtype} across three states; got conserved shape {conserved.shape}, history shape {history.shape}, dtype {history.dtype}"
        )
    if not np.all(np.isfinite(np.asarray(conserved))) or np.any(
        np.asarray(conserved[:, 0]) <= 0
    ):
        raise ValueError(
            f"fields.conserved must be finite with positive density; got {np.asarray(conserved)!r}"
        )
    for name in ("pressure", "temperature"):
        value = np.asarray(fields[name])
        if (
            value.shape != (count,)
            or not np.all(np.isfinite(value))
            or np.any(value <= 0)
        ):
            raise ValueError(
                f"fields.{name} must have shape {(count,)} and finite positive values; got shape {value.shape}, values {value!r}"
            )
    source = _prepare_periodic_physical_source(context, conserved)
    finite_volume = _solve_periodic_finite_volume(context, source)
    virtual_element = _solve_periodic_virtual_element(context, source)
    response = _retain_periodic_physical_fields(
        context, source, finite_volume, virtual_element
    )
    record, error, updated = _record_periodic_physical_evidence(
        context, source, finite_volume, virtual_element, response, conserved, history
    )
    fem_record, fem_error, fem_fields = _polyhedral_fem_physics(
        result, {**fields, **seeds}, conserved[:, 0], stages,
        amplitude=source.amplitude, budget=budget, prepared_plan=owners.source_star,
        repeats=repeats,
    )
    record["fe"] = fem_record
    updated.update(fem_fields)
    return record, max(error, fem_error), updated


_POLYHEDRAL_CONSERVED_COMPONENTS = (
    "density", "momentum_x", "momentum_y", "momentum_z",
    "total_energy", "species_0", "species_1",
)


def _polyhedral_archive_role(
    name: str,
    value: jax.Array,
    epoch: int,
    binding: polyhedral_field_owner.MeshingPolyhedralOwnerBinding,
) -> polyhedral_field_owner.MeshingPolyhedralFieldStateRole:
    history = name.endswith("_history")
    base = name.removesuffix("_history")
    inverse_length = derived_unit("1/m", ((METER, -1),))
    inverse_area = derived_unit("1/m^2", ((METER, -2),))
    owner: polyhedral_field_owner.MeshingPolyhedralOwner
    bank: polyhedral_field_owner.MeshingPolyhedralIndexBank
    algebraic: polyhedral_field_owner.MeshingPolyhedralAlgebraicRole
    kind: polyhedral_field_owner.MeshingFieldStateKind
    unit: UnitDefinition
    components: tuple[str, ...] = ()
    match base:
        case "vem_solution":
            owner, bank, algebraic, kind, unit = "vem", "coefficients", "primal", "coefficients", ONE
        case "vem_analytic_nodal":
            owner, bank, algebraic, kind, unit = "vem", "coefficients", "primal", "physical-values", ONE
        case "vem_physical_load" | "vem_physical_residual":
            owner, bank, algebraic, kind, unit = "vem", "coefficients", "dual", "dual-values", METER
        case "fem_solution":
            owner, bank, algebraic, kind, unit = "source_star", "coefficients", "primal", "coefficients", ONE
        case "fem_physical_load" | "fem_physical_residual":
            owner, bank, algebraic, kind, unit = "source_star", "coefficients", "dual", "dual-values", METER
        case "fem_cell_solution":
            owner, bank, algebraic, kind, unit = "source_star", "cell", "primal", "physical-values", ONE
        case "fem_gradient" | "fem_vector_flux":
            owner, bank, algebraic, kind, unit = "source_star", "cell", "physical", "physical-values", inverse_length
            components = ("x", "y", "z")
        case "inventory" | "previous_inventory" | "conserved" | "pressure" | "temperature":
            owner, bank, algebraic, kind, unit = "hybrid", "cell", "physical", "physical-values", ONE
            if base == "conserved":
                components = _POLYHEDRAL_CONSERVED_COMPONENTS
            elif base == "previous_inventory":
                kind = "material-history"
        case "physical_force":
            owner, bank, algebraic, kind, unit = "hybrid", "cell", "physical", "physical-values", inverse_area
        case "physical_source_load" | "fv_source" | "fv_flux_balance":
            owner, bank, algebraic, kind, unit = "hybrid", "cell", "dual", "dual-values", METER
        case "fv_cell_solution" | "fv_analytic_cell_average":
            owner, bank, algebraic, kind, unit = "hybrid", "cell", "primal", "physical-values", ONE
        case "fv_trace_solution":
            owner, bank, algebraic, kind, unit = "hybrid", "active-trace", "primal", "trace-values", ONE
        case "fv_solution" | "fv_affine_solution":
            owner, bank, algebraic, kind, unit = "hybrid", "cell-trace", "primal", "physical-values", ONE
        case "fv_physical_residual":
            owner, bank, algebraic, kind, unit = "hybrid", "cell-trace", "dual", "dual-values", METER
        case "fv_owner_flux":
            owner, bank, algebraic, kind, unit = "hybrid", "active-trace", "physical", "physical-values", METER
        case "fv_local_flux":
            owner, bank, algebraic, kind, unit = "hybrid", "local-facet", "physical", "physical-values", METER
        case "fv_gradient" | "fv_vector_flux":
            owner, bank, algebraic, kind, unit = "hybrid", "cell", "physical", "physical-values", inverse_length
            components = ("x", "y", "z")
        case _:
            raise ValueError(f"Polyhedral accepted state must declare every scientific field explicitly; got {name!r}.")
    index_axes = (0, 1) if bank == "local-facet" else (0,)
    if history:
        index_axes = tuple(axis + 1 for axis in index_axes)
        kind = "stage-history"
    return polyhedral_field_owner.MeshingPolyhedralFieldStateRole(
        entry_name=name, field_name=owner, role=kind,
        shape=tuple(value.shape), dtype=np.dtype(value.dtype), state_epoch=epoch,
        value_units=(unit,) * max(1, len(components)),
        index_binding=polyhedral_field_owner.MeshingPolyhedralFieldIndexBinding(
            owner_binding=binding, owner=owner, bank_name=bank, index_axes=index_axes,
        ),
        algebraic_role=algebraic, component_names=components,
    )


def _prepare_polyhedral_accepted_epoch(
    carrier: phx.meshing.CellMeshingResult,
    fields: dict[str, jax.Array],
    epoch: int,
    *,
    transition: phx.meshing.MeshAdaptationResult | None = None,
    owners: polyhedral_consumer_owner.PolyhedralArchiveOwners | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> MeshingAcceptedEpoch:
    if carrier.coordinate_contract.length_unit.unit_id != METER.unit_id:
        raise ValueError("This frozen polyhedral benchmark requires its declared physical meter coordinates and one-meter reference length.")
    binding, _ = polyhedral_field_owner.prepare_meshing_polyhedral_owner_binding(
        carrier, owners=owners, budget=budget,
    )
    declaration = polyhedral_field_owner.prepare_meshing_polyhedral_field_declaration(
        carrier, binding,
        tuple(_polyhedral_archive_role(name, value, epoch, binding) for name, value in sorted(fields.items())),
        part_name="polyhedral",
        normalization_reference_length=METER,
    )
    return MeshingAcceptedEpoch(carrier, (declaration,), fields, transition=transition)


def _polyhedral_source_records(
    source: phx.meshing.NativePlcSource | phx.meshing.NativePolyhedralSource,
    request: phx.meshing.VolumeMeshingSpec,
    options: phx.meshing.NativeMeshingOptions,
    epochs: tuple[MeshingAcceptedEpoch, ...],
) -> dict[str, Any]:
    target = epochs[-1].carrier
    certification = target.certification
    if certification is None:
        raise ValueError("Polyhedral accepted epochs require their actual source certification.")
    return {
        "certification_inputs": certification.request,
        "report": certification,
        "associations": target.associations,
        "generation_source": source,
        "generation_specification": request,
        "generation_options": options,
        "accepted_target": target,
        "accepted_data": {"accepted_epochs": epochs},
    }


def _polyhedral_epoch_owners(
    epoch: MeshingAcceptedEpoch,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> polyhedral_consumer_owner.PolyhedralArchiveOwners:
    if len(epoch.declarations) != 1 or type(epoch.declarations[0]) is not polyhedral_field_owner.MeshingPolyhedralFieldDeclaration:
        raise TypeError("A polyhedral accepted epoch must retain its actual single physical owner declaration.")
    actual, _ = prepare_meshing_field_owner(
        epoch.declarations[0], epoch.carrier, budget=budget
    )
    if not isinstance(actual, polyhedral_consumer_owner.PolyhedralArchiveOwners):
        raise TypeError("A polyhedral accepted epoch must reconstruct its actual VEM, hybrid and source-star owners.")
    return actual


def _polyhedral_field_hashes(
    fields: Mapping[str, jax.Array | np.ndarray],
) -> dict[str, dict[str, Any]]:
    result = {}
    for name, value in fields.items():
        array = np.asarray(value)
        result[name] = {
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "bytes": array.nbytes,
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    return result


def _polyhedral_archive_evidence(
    root: Mapping[str, Any],
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> dict[str, Any]:
    """Bind canonical accepted epochs, true physical owners and every field byte."""
    epochs = polyhedral_field_owner.validate_meshing_accepted_epochs(root)
    records = []
    for epoch in epochs:
        result = epoch.carrier
        certification = result.certification
        if certification is None or not certification.passed or not result.audit.passed or not result.compliance.passed:
            raise ValueError("A polyhedral archive must retain accepted certified physical epochs.")
        owners = _polyhedral_epoch_owners(epoch, budget=budget)
        volumes = np.asarray(owners.finite_volume.cell_volumes)
        fields = epoch.fields
        inventories: dict[str, Any] = {}
        if "conserved" in fields:
            inventories = {
                "conserved": np.einsum("nk,n->k", np.asarray(fields["conserved"]), volumes).tolist(),
                "conserved_history": np.einsum("tnk,n->tk", np.asarray(fields["conserved_history"]), volumes).tolist(),
                "pressure": float(np.dot(np.asarray(fields["pressure"]), volumes)),
                "temperature": float(np.dot(np.asarray(fields["temperature"]), volumes)),
            }
        elif "inventory" in fields:
            inventories = {
                "inventory": float(np.dot(np.asarray(fields["inventory"]), volumes)),
                "inventory_history": np.asarray(fields["inventory_history"]) @ volumes,
                "previous_inventory": float(np.dot(np.asarray(fields["previous_inventory"]), volumes)),
            }
            inventories["inventory_history"] = inventories["inventory_history"].tolist()
        else:
            raise ValueError("A polyhedral epoch must retain its actual declared conserved or material inventory.")
        records.append({
            "epoch_id": epoch.epoch_id,
            "result": _result_identity(result),
            "result_fingerprint": array_tree_fingerprint(result),
            "geometry_fingerprint": array_tree_fingerprint(result.geometry),
            "domain_fingerprint": array_tree_fingerprint(certification.request.domain),
            "certification_id": certification.report_id,
            "certification_fingerprint": array_tree_fingerprint(certification),
            "declarations_fingerprint": array_tree_fingerprint(epoch.declarations),
            "transition_id": None if epoch.transition is None else epoch.transition.result_id,
            "inventories": inventories,
            "fields": _polyhedral_field_hashes(fields),
        })
    return {
        "source_fingerprint": array_tree_fingerprint(root["generation_source"]),
        "request_fingerprint": array_tree_fingerprint(root["generation_specification"]),
        "options_fingerprint": array_tree_fingerprint(root["generation_options"]),
        "request_id": root["generation_specification"].specification_id,
        "epochs": records,
    }



def _polyhedral_material_cold_process(
    path: Path,
    content_id: str,
    expected: dict[str, Any],
    repeats: int,
    target_error: float,
    *,
    budget: CoordinateEnclosureBudget,
    deadline: Deadline,
) -> tuple[phx.meshing.CellMeshingResult, ColdWorkerReport]:
    started = perf_counter()
    root = read_meshing_source_closure(
        path, expected_content_id=content_id, limits=ArrayArchiveLimits(), budget=budget
    )
    evidence = _polyhedral_archive_evidence(root, budget=budget)
    if evidence != expected:
        raise ValueError("Material cold restore must preserve the complete authored source, accepted epochs and field bytes.")
    epochs = polyhedral_field_owner.validate_meshing_accepted_epochs(root)
    read_seconds = perf_counter() - started
    continued: list[dict[str, Any]] = []
    threshold = min(target_error, 1e-7)
    for epoch in epochs:
        deadline.remaining()
        fields = {name: jnp.asarray(value) for name, value in epoch.fields.items()}
        stages: dict[str, float] = {}
        physical = _polyhedral_physics(
            epoch.carrier, np.asarray(fields["inventory"]), repeats, stages, fields,
            budget=budget, owners=_polyhedral_epoch_owners(epoch, budget=budget),
        )
        differences = {
            name: float(jnp.max(jnp.abs(physical.fields[name] - fields[name])))
            for name in ("vem_solution", "fv_affine_solution", "fv_solution", "fem_solution")
        }
        histories = all(
            np.array_equal(np.asarray(physical.fields[name + "_history"][-2]), np.asarray(fields[name]))
            for name in ("inventory", "vem_solution", "fv_affine_solution", "fv_solution", "fem_solution")
        )
        continued.append({
            "epoch_id": epoch.epoch_id,
            "result_id": epoch.carrier.result_id,
            "status": "passed" if np.isfinite(physical.error) and physical.error <= threshold and max(differences.values()) <= threshold and histories else "failed",
            "physics": physical.record,
            "physical_error": physical.error,
            "continued_solution_defects": differences,
            "accepted_primitive_history_predecessors_preserved": histories,
            "continued_fields": _polyhedral_field_hashes(physical.fields),
            "stages_seconds": stages,
        })
    if _polyhedral_archive_evidence(root, budget=budget) != evidence:
        raise ValueError("Material cold continuation must not mutate any archived accepted owner or field.")
    deadline.remaining()
    return epochs[-1].carrier, ColdWorkerReport(
        record={
            "status": "passed" if all(item["status"] == "passed" for item in continued) else "failed",
            "restored": evidence,
            "accepted_fields_bitwise_preserved": True,
            "continued_from_result_id": epochs[-1].carrier.result_id,
            "continued_epochs": continued,
            "physical_error": max(item["physical_error"] for item in continued),
            "continued_solution_defects": continued[-1]["continued_solution_defects"],
            "accepted_history_predecessors_preserved": all(item["accepted_primitive_history_predecessors_preserved"] for item in continued),
            "interpreter": sys.executable,
            "runtime": _runtime_identity(),
            **_polyhedral_memory_measurements(),
        },
        read_seconds=read_seconds,
    )




def _polyhedral_qualification_periodic_cold_process(
    path: Path,
    content_id: str,
    expected: dict[str, Any],
    repeats: int,
    target_error: float,
    *,
    budget: CoordinateEnclosureBudget,
    deadline: Deadline,
) -> tuple[phx.meshing.CellMeshingResult, ColdWorkerReport]:
    started = perf_counter()
    root = read_meshing_source_closure(
        path, expected_content_id=content_id, limits=ArrayArchiveLimits(), budget=budget
    )
    evidence = _polyhedral_archive_evidence(root, budget=budget)
    if evidence != expected:
        raise ValueError("Periodic cold restore must preserve source controls, certified epochs and every accepted field.")
    epochs = polyhedral_field_owner.validate_meshing_accepted_epochs(root)
    read_seconds = perf_counter() - started
    continued: list[dict[str, Any]] = []
    for epoch in epochs:
        deadline.remaining()
        fields = {name: jnp.asarray(value) for name, value in epoch.fields.items()}
        timings: dict[str, float] = {}
        physics, error, updated = _polyhedral_qualification_periodic_physics(
            epoch.carrier, fields, repeats, timings, seeds=fields,
            deadline=deadline, budget=budget,
            owners=_polyhedral_epoch_owners(epoch, budget=budget),
        )
        continued.append({
            "epoch_id": epoch.epoch_id,
            "result_id": epoch.carrier.result_id,
            "physics": physics,
            "target_error_reached": bool(np.isfinite(error) and error <= target_error),
            "physical_error": error,
            "stages_seconds": timings,
            "continued_fields": _polyhedral_field_hashes(updated),
        })
    if _polyhedral_archive_evidence(root, budget=budget) != evidence:
        raise ValueError("Periodic cold continuation must not mutate any archived accepted owner or field.")
    deadline.remaining()
    return epochs[-1].carrier, ColdWorkerReport(
        record={
            "status": "passed" if all(item["target_error_reached"] for item in continued) else "failed",
            "restored": evidence,
            "continued_epochs": continued,
            "accepted_fields_bitwise_preserved": True,
            "interpreter": sys.executable,
            "runtime": _runtime_identity(),
            **_polyhedral_memory_measurements(),
        },
        read_seconds=read_seconds,
    )


def _polyhedral_qualification_periodic_cold(
    source: phx.meshing.NativePolyhedralSource,
    request: phx.meshing.VolumeMeshingSpec,
    options: phx.meshing.NativeMeshingOptions,
    epochs: tuple[MeshingAcceptedEpoch, ...],
    repeats: int,
    target_error: float,
    stages: dict[str, Any],
    deadline: Deadline,
    *,
    expected_build_id: str,
) -> dict[str, Any]:
    root = _polyhedral_source_records(source, request, options, epochs)
    expected = _polyhedral_archive_evidence(root)
    return _polyhedral_qualification_cold(
        root, expected, "periodic", len(epochs), repeats, target_error, stages, deadline,
        expected_build_id=expected_build_id,
    )


def _polyhedral_qualification_cold(
    root: Any,
    expected: dict[str, Any],
    kind: str,
    accepted_epochs: int,
    repeats: int,
    target_error: float,
    stages: dict[str, Any],
    deadline: Deadline,
    *,
    expected_build_id: str,
) -> dict[str, Any]:
    if "_campaign_started" not in stages:
        raise ValueError(
            "stages._campaign_started must contain the perf_counter campaign start; got missing key"
        )
    limits = ArrayArchiveLimits()
    record = {
        "status": "failed",
        "archive_limits": {
            "members": limits.max_members,
            "nesting": limits.max_manifest_nesting,
            "rank": limits.max_array_rank,
        },
        "accepted_epochs": accepted_epochs,
        "source_authority_stripped": False,
    }
    _ledger = StageLedger(
        record, deadline=deadline, cold_phase="fresh_process_restore_continued_pde"
    )
    with (
        TemporaryDirectory(prefix=f"phydrax-poly-{kind}-cold-") as directory,
        prepare_cold_worker(
            "tools.meshing_qualification.scenarios.polyhedral",
            deadline=deadline,
            ledger=_ledger,
        ) as cold_worker,
    ):
        receipt, timing = _ledger.run(
            "write_meshing_source_closure",
            lambda: _measured(
                lambda: write_meshing_source_closure(
                    Path(directory) / "accepted.zip", root, limits=limits
                )
            ),
        )
        stages["periodic_checkpoint_publication"] = timing["wall_seconds"]
        remaining = deadline.expires_at - perf_counter()
        record.update(
            {
                "content_id": receipt.content_id,
                "container_bytes": receipt.path.stat().st_size,
                "remaining_original_seconds": remaining,
            }
        )
        if remaining <= 0:
            return {
                **record,
                "status": "timeout",
                "failure": "Original campaign deadline expired before cold restore.",
            }
        # The scenario module owns its cold consumer.
        cold_started = perf_counter()
        try:
            restored = cold_worker.execute(
                receipt.path,
                receipt.content_id,
                {
                    "kind": kind,
                    "expected": expected,
                    "repeats": repeats,
                    "target_error": target_error,
                },
            )
            timing = {"wall_seconds": perf_counter() - cold_started}
        except ColdWorkerTimeout as failure:
            return {
                **record,
                "status": "timeout",
                "failure_type": type(failure).__name__,
                "stdout": failure.stdout.decode(errors="replace")
                if isinstance(failure.stdout, bytes)
                else failure.stdout,
                "stderr": failure.stderr.decode(errors="replace")
                if isinstance(failure.stderr, bytes)
                else failure.stderr,
            }
        except QualificationFailure as failure:
            record_failure(record, failure, "fresh_process_restore_continued_pde")
            return record
        stages["periodic_cold_restore_continued_pde"] = timing["wall_seconds"]
        record.update(
            {
                "returncode": 0,
                "stdout": json.dumps(restored),
                "stderr": "",
            }
        )
        # Parsed and process-bound by ColdWorker.execute.
        if restored["restored"] != expected or restored["process_id"] == os.getpid():
            raise qualification_failure(
                "_polyhedral_qualification_periodic_cold",
                (
                    "Periodic cold evidence does not bind the archived source and a fresh process."
                ),
                {"expected": expected, "restored": restored},
            )
        same_build = restored["runtime"]["build_id"] == expected_build_id
        record.update(
            {
                "status": restored["status"] if same_build else "failed",
                "warm_build_id": expected_build_id,
                "cold_build_matches_warm": same_build,
                "fresh_interpreter": restored,
                "accepted_fields_bitwise_preserved": True,
            }
        )
        if not same_build:
            record["failure"] = "Fresh polyhedral continuation must use the warm source build; got source/build drift."
        return record


@final
class _NativeResourceObservation(phx.StrictModule):
    receipt: NativeExecutionRecord
    phases: tuple[NativeExecutionRecord, ...]
    observed: dict[str, int | float]
    declared: dict[str, int | float]
    enforced: set[str]
    bounds: dict[str, dict[str, Any]]
    managed_admission_boundary: int | None
    scratch_observation: dict[str, int | None]


def _observe_native_resources(
    request: phx.meshing.VolumeMeshingSpec, result: phx.meshing.CellMeshingResult
) -> _NativeResourceObservation:
    receipt = result.execution_evidence
    if receipt is None:
        raise qualification_failure(
            "_observe_native_resources",
            ("Publication has no successful native execution record."),
            {"receipt": receipt},
        )
    receipt.require_valid()
    pending, seen, phases = [receipt], set(), []
    while pending:
        phase = pending.pop()
        if id(phase) in seen:
            continue
        seen.add(id(phase))
        if int(np.asarray(phase.status)) != int(MeshcoreStatus.OK):
            raise qualification_failure(
                "_observe_native_resources",
                ("Positive admission contains a non-successful native phase record."),
                {"MeshcoreStatus": MeshcoreStatus, "phase": phase},
            )
        phases.append(phase)
        pending.extend(
            item
            for item in (phase.preparation_evidence, phase.consumer_evidence)
            if item is not None
        )
    mesh = result.mesh
    native_peak = max(
        int(np.asarray(phase.memory_counters.peak_bytes)) for phase in phases
    )
    host_upper = int(np.asarray(receipt.total_host_storage_peak_bytes_upper))
    admission = receipt.total_managed_admission_peak_bytes_upper
    admission_boundary = None if admission is None else int(np.asarray(admission))
    conservative_combined_upper = native_peak + host_upper
    observed = {
        "maximum_work_units": int(np.asarray(receipt.total_work_units)),
        "maximum_geometry_queries": int(np.asarray(receipt.total_geometry_queries)),
        "maximum_cavity_cells": max(
            int(np.asarray(phase.work_counters.peak_cavity_cells)) for phase in phases
        ),
        "maximum_scratch_bytes": conservative_combined_upper if admission_boundary is None else admission_boundary,
        "maximum_cells": sum(block.cell_count for block in result.mesh.blocks),
        "maximum_vertices": int(result.mesh.coordinates.shape[0]),
        "maximum_edges": int(mesh.entity_set(1).entity_ids.size),
        "maximum_faces": int(mesh.entity_set(2).entity_ids.size),
        "maximum_connectivity_entries": int(
            sum(block.vertices.size for block in mesh.blocks)
            + sum(
                incidence.relation.source_indices.size
                + incidence.relation.target_indices.size
                for incidence in mesh.topology.incidences
            )
        ),
        "maximum_data_bytes": model_array_bytes(result),
        "maximum_wall_seconds": float(np.asarray(receipt.total_elapsed_seconds)),
    }
    declared = meshing_limit_values(request.limits)
    enforced = {"maximum_" + name for name in result.runtime.enforced_limits}
    bounds = {}
    for name, limit in declared.items():
        value = observed.get(name)
        bounds[name] = {
            "original_limit": limit,
            "observed": value,
            "runtime_enforced": name in enforced,
            "within_original_bound": None if value is None else value <= limit,
            "observation": "unobserved"
            if value is None
            else "actual native receipt or accepted carrier",
        }
        if value is not None and value > limit:
            raise qualification_failure(
                "_observe_native_resources",
                (f"Native execution record exceeds {name}."),
                {"limit": limit, "value": value},
            )
    return _NativeResourceObservation(
        receipt=receipt,
        phases=tuple(phases),
        observed=observed,
        declared=declared,
        enforced=enforced,
        bounds=bounds,
        managed_admission_boundary=admission_boundary,
        scratch_observation={
            "native_allocator_peak_bytes": native_peak,
            "bounded_host_peak_bytes_upper": host_upper,
            "conservative_native_plus_host_peak_bytes_upper": conservative_combined_upper,
            "managed_admission_peak_bytes_upper": admission_boundary,
        },
    )


def _complete_native_request_resources(
    request: phx.meshing.VolumeMeshingSpec,
    result: phx.meshing.CellMeshingResult,
) -> dict[str, Any]:
    observation = _observe_native_resources(request, result)
    return {
        "within_original_bounds": all(
            item["within_original_bound"] is True and item["runtime_enforced"] is True
            for item in observation.bounds.values()
        ),
        "bounds": observation.bounds,
        "scratch_observation": observation.scratch_observation,
        "execution_owner_id": observation.receipt.owner_id,
        "receipt_id": observation.receipt.receipt_hex(),
        "scope": "complete native generation/adaptation request including paid scientific preparation and accepted-epoch consumers",
        "complete_request_R_claimed": False,
    }


def _exercise_polyhedral_source_prefix_boundary(
    source: phx.meshing.NativePlcSource | phx.meshing.NativePolyhedralSource,
    request: phx.meshing.VolumeMeshingSpec,
    options: phx.meshing.NativeMeshingOptions,
    result: phx.meshing.CellMeshingResult,
    name: str,
    value: int | float,
    declared: dict[str, int | float],
    operation_started: float,
    original: dict[str, Any],
    prepared_plan: phx.meshing.NativeMeshingPlan,
) -> dict[str, Any]:
    is_clock = name == "maximum_wall_seconds"
    sharp_positive = None
    is_work = name == "maximum_work_units"
    if name == "maximum_scratch_bytes" or is_work:
        sharp_limits = replace(
            MeshingLimitControls.from_limits(request.limits), **{name: value}
        ).to_limits()
        sharp_measurements = []
        sharp_request = request if is_work else with_meshing_limits(request, sharp_limits)
        admitted = execute_native_capacity_trial(
            source, sharp_request, options, result,
            operation_started=operation_started,
            record_phase=sharp_measurements.append,
            coordinate_operands=CoordinateOperandCollector(),
            prepared_plan=prepared_plan if is_work else None,
            root_maximum_work_units=int(value) if is_work else None,
        )
        admitted_receipt = admitted.execution_evidence
        if admitted_receipt is None:
            raise ValueError("Measured R admission must retain its execution receipt.")
        if not is_work and admitted_receipt.total_managed_admission_peak_bytes_upper is None:
            raise ValueError("Measured managed R admission must retain complete request-window observation; a different resource ancestor is not a matched boundary proof.")
        if is_work and (
            int(np.asarray(admitted_receipt.total_work_units)) != value
            or admitted.result_id != result.result_id
        ):
            raise ValueError("Mechanical work R must reproduce the original work and scientific result identity.")
        mesh_fingerprint = array_tree_fingerprint(admitted.mesh)
        geometry_fingerprint = array_tree_fingerprint(admitted.geometry)
        if mesh_fingerprint != array_tree_fingerprint(result.mesh) or geometry_fingerprint != array_tree_fingerprint(result.geometry):
            raise ValueError("The measured managed admission boundary must reproduce the original scientific mesh and source geometry bytes.")
        sharp_positive = {
            "limit": value,
            "limits_id": sharp_limits.limits_id,
            "result": _result_identity(admitted),
            "frozen_request": frozen_request_record(source, sharp_request, options),
            "mesh_fingerprint": mesh_fingerprint,
            "geometry_fingerprint": geometry_fingerprint,
            "native_phases": _phase_recordings(sharp_measurements),
            "mesh_and_geometry_bitwise_preserved": True,
            "observed_managed_admission_envelope": (
                None if admitted_receipt.total_managed_admission_peak_bytes_upper is None
                else int(np.asarray(admitted_receipt.total_managed_admission_peak_bytes_upper))
            ),
            "observed_work_units": int(np.asarray(admitted_receipt.total_work_units)),
            "mechanical_admission": meshing_limit_values(sharp_limits) if is_work else None,
            "plan_id": prepared_plan.plan_id if is_work else None,
            "preparation_receipt_id": (
                prepared_plan.preparation_evidence.receipt_hex()
                if is_work and prepared_plan.preparation_evidence is not None else None
            ),
        }
    reduced_limits = _source_prefix_boundary_limits(request.limits, name, value)
    lowered = meshing_limit_values(reduced_limits)[name]
    reduced_request = request if is_work else with_meshing_limits(request, reduced_limits)
    measurements = []
    try:
        reduced_admission = execute_native_capacity_trial(
            source,
            reduced_request,
            options,
            result,
            operation_started=operation_started,
            record_phase=measurements.append,
            coordinate_operands=CoordinateOperandCollector(),
            prepared_plan=prepared_plan if is_work else None,
            root_maximum_work_units=int(lowered) if is_work else None,
        )
    except phx.meshing.MeshingFailure as failure:
        expected_category = (
            phx.meshing.MeshingFailureCategory.TIMED_OUT
            if is_clock
            else phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
        )
        if failure.category is not expected_category:
            raise qualification_failure(
                "_polyhedral_source_prefix_resource_refusals",
                (f"Reduced {name} did not yield a resource refusal: {failure}"),
                {"expected_category": expected_category, "failure": failure},
            ) from failure
        refusal = {
            "control": name,
            "original_limit": declared[name],
            "reduced_limit": lowered,
            "limits_id": reduced_limits.limits_id,
            "frozen_request": frozen_request_record(source, reduced_request, options),
            **FailureRecord.from_exception(failure, failure.evidence.stage).to_record(),
            "logical_findings_count": len(failure.evidence.logical_findings),
            "native_phases": _phase_recordings(measurements),
            "measured_R_positive": sharp_positive,
            "mechanical_admission": meshing_limit_values(reduced_limits) if is_work else None,
            "plan_id": prepared_plan.plan_id if is_work else None,
            "preparation_receipt_id": (
                prepared_plan.preparation_evidence.receipt_hex()
                if is_work and prepared_plan.preparation_evidence is not None else None
            ),
        }
    except Exception as failure:
        raise qualification_failure(
            "_polyhedral_source_prefix_resource_refusals",
            f"Reduced {name} did not return its required typed refusal: {failure}",
            {
                "control": name, "original_limit": declared[name],
                "observed_R": value, "reduced_limit": lowered,
                "failure_type": type(failure).__name__,
            },
        ) from failure
    else:
        raise qualification_failure(
            "_polyhedral_source_prefix_resource_refusals",
            (f"Reduced observed {name} unexpectedly admitted a publication."),
            {
                "control": name,
                "R_candidate": value,
                "reduced_limit": lowered,
                "admitted_managed_envelope": (
                    None if reduced_admission.execution_evidence is None
                    or reduced_admission.execution_evidence.total_managed_admission_peak_bytes_upper is None
                    else int(np.asarray(reduced_admission.execution_evidence.total_managed_admission_peak_bytes_upper))
                ),
                "admitted_work": (
                    None if reduced_admission.execution_evidence is None
                    else int(np.asarray(reduced_admission.execution_evidence.total_work_units))
                ),
            },
        )
    if array_tree_fingerprint((source, request, options, result)) != original:
        raise qualification_failure(
            "_polyhedral_source_prefix_resource_refusals",
            ("Resource campaign mutated source bytes, SCI or controls."),
            {
                "array_tree_fingerprint": array_tree_fingerprint,
                "options": options,
                "original": original,
                "request": request,
                "result": result,
                "source": source,
            },
        )
    if is_work:
        _require_source_prefix_trial(
            refusal, value, source, request, options,
            {
                "result": _result_identity(result),
                "mesh_fingerprint": array_tree_fingerprint(result.mesh),
                "geometry_fingerprint": array_tree_fingerprint(result.geometry),
            },
            None,
        )
    return refusal


def _polyhedral_source_prefix_resource_refusals(
    source: phx.meshing.NativePlcSource | phx.meshing.NativePolyhedralSource,
    request: phx.meshing.VolumeMeshingSpec,
    options: phx.meshing.NativeMeshingOptions,
    result: phx.meshing.CellMeshingResult,
    *,
    prepared_plan: phx.meshing.NativeMeshingPlan,
    campaign_started: float | None = None,
) -> dict[str, Any]:
    """Prove only the original generation prefix's admitted resource boundaries."""
    observation = _observe_native_resources(request, result)
    receipt = observation.receipt
    phases = observation.phases
    observed = observation.observed
    declared = observation.declared
    enforced = observation.enforced
    bounds = observation.bounds
    original = array_tree_fingerprint((source, request, options, result))

    # A supplied start retains the campaign clock rather than granting a new allowance.
    debit = (
        float(np.asarray(receipt.total_elapsed_seconds))
        if campaign_started is None
        else perf_counter() - campaign_started
    )
    operation_started = monotonic() - debit
    refusals, not_exercised = [], []
    for name, value in observed.items():
        if name == "maximum_scratch_bytes" and observation.managed_admission_boundary is None:
            not_exercised.append({
                "control": name,
                "reason": "A complete managed admission envelope was not observed; separate native/host peak uppers do not prove a sharp R.",
            })
            continue
        is_clock = name == "maximum_wall_seconds"
        if name not in enforced or value <= (0 if is_clock else 1):
            not_exercised.append(
                {
                    "control": name,
                    "reason": "No positive reducible runtime-enforced observation.",
                }
            )
            continue
        refusals.append(
            _exercise_polyhedral_source_prefix_boundary(
                source,
                request,
                options,
                result,
                name,
                value,
                declared,
                operation_started,
                original,
                prepared_plan,
            )
        )
    if not refusals:
        raise qualification_failure(
            "_polyhedral_source_prefix_resource_refusals",
            ("No observed resource boundary was exercised."),
            {"refusals": refusals},
        )
    missing_gate = [
        name
        for name, item in bounds.items()
        if item["observed"] is None or not item["runtime_enforced"]
    ]
    if observation.managed_admission_boundary is None:
        missing_gate.append("maximum_scratch_bytes:managed_admission_boundary_unobserved")
    return {
        "status": "incomplete" if missing_gate else "passed",
        "mandatory_capacity_gate": "incomplete" if missing_gate else "complete",
        "unobserved_or_unenforced_original_bounds": missing_gate,
        "frozen_request": frozen_request_record(source, request, options),
        "positive": {
            "result": _result_identity(result),
            "mesh_fingerprint": array_tree_fingerprint(result.mesh),
            "geometry_fingerprint": array_tree_fingerprint(result.geometry),
            "receipt_fingerprint": array_tree_fingerprint(receipt),
            "owner_id": receipt.owner_id,
            "bounds": bounds,
            "scratch_observation": observation.scratch_observation,
            "phase_receipt_count": len(phases),
            "source_preparation_queries": int(
                np.asarray(receipt.source_preparation_geometry_queries)
            ),
            "cumulative_queries": int(np.asarray(receipt.total_geometry_queries)),
            "native_primitive_queries_by_phase": [
                int(np.asarray(phase.native_primitive_queries)) for phase in phases
            ],
            "externally_charged_queries_by_phase": [
                int(np.asarray(phase.externally_charged_geometry_queries))
                for phase in phases
            ],
            "retained_live_upper_by_phase": [
                int(np.asarray(phase.host_storage_live_bytes_upper)) for phase in phases
            ],
            "retained_peak_upper_by_phase": [
                int(np.asarray(phase.host_storage_peak_bytes_upper)) for phase in phases
            ],
            "managed_memory_counters_by_phase": [
                {
                    name: int(np.asarray(value))
                    for name, value in phase.memory_counters.items()
                }
                for phase in phases
            ],
        },
        "refusals": refusals,
        "not_exercised": not_exercised,
        "original_source_and_controls_preserved": True,
        "unobserved_enforcement_claimed": False,
        "admitted_scope": "SOURCEPREFIX",
        "scope": "initial native generation prefix only; not the later complete scientific consumer request",
        "complete_request_R_claimed": False,
        "original_elapsed_debit_seconds": debit,
    }


def _polyhedral_qualification_periodic_initial_fields(
    owners: polyhedral_consumer_owner.PolyhedralArchiveOwners,
) -> dict[str, jax.Array]:

    fv = owners.finite_volume
    density = 2.0 + fv.cell_centers[:, 0]
    velocity = jnp.asarray((0.1, 0.2, -0.1), dtype=density.dtype)
    pressure = jnp.ones_like(density)
    momentum = density[:, None] * velocity
    energy = pressure / 0.4 + 0.5 * density * jnp.vdot(velocity, velocity)
    species = density[:, None] * jnp.asarray((0.2, 0.3), dtype=density.dtype)
    conserved = jnp.concatenate(
        (density[:, None], momentum, energy[:, None], species), axis=1
    )
    history = jnp.stack((0.8 * conserved, 0.9 * conserved, conserved))
    return {
        "conserved": conserved,
        "conserved_history": history,
        "pressure": pressure,
        "temperature": pressure / density,
        "pressure_history": jnp.stack((0.8 * pressure, 0.9 * pressure, pressure)),
        "temperature_history": jnp.stack((pressure / density,) * 3),
    }


@final
class _PeriodicTransportPreparation(phx.StrictModule):
    source: phx.meshing.CellMeshingResult
    target: phx.meshing.CellMeshingResult
    common: phx.geometry.PreparedCommonRefinement
    old_fv: phx.discretization.UnstructuredFiniteVolumeDiscretization
    new_fv: phx.discretization.UnstructuredFiniteVolumeDiscretization
    remap: UnstructuredConservativeRemapPlan
    old_epoch: TopologyEpoch
    new_epoch: TopologyEpoch
    transfer: TopologyEpochTransition
    old_topology: CompositionEntry
    new_topology: CompositionEntry
    entries: list[CompositionEntry]
    transports: list[phx.lifecycle.CompositionTransport]
    output: dict[str, jax.Array]
    inventory_defects: dict[str, float]
    cell_fields: tuple[str, ...]
    cell_roles: dict[str, CompositionRole]


type _PolyhedralOwnerPair = tuple[
    polyhedral_consumer_owner.PolyhedralArchiveOwners,
    polyhedral_consumer_owner.PolyhedralArchiveOwners,
]


def _prepare_periodic_transport(
    adaptation: phx.meshing.MeshAdaptationResult,
    epoch: int,
    cell_roles: dict[str, CompositionRole],
    owners: _PolyhedralOwnerPair | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> _PeriodicTransportPreparation:
    source, target, common = (
        adaptation.source,
        adaptation.target,
        adaptation.common_refinement,
    )
    if (
        common is None
        or not common.succeeded
        or adaptation.status is not phx.meshing.MeshAdaptationStatus.COMPLETE
    ):
        raise qualification_failure(
            "_polyhedral_qualification_periodic_transport",
            (
                "Periodic accepted transition requires certified overlap and successful adaptation."
            ),
            {"adaptation": adaptation, "common": common},
        )
    if owners is not None:
        for carrier, owned in zip((source, target), owners, strict=True):
            if carrier.geometry is None or owned.source_star.source_mesh_id != carrier.mesh.mesh_id or owned.source_star.source_geometry_id != cell_geometry_id(carrier.geometry):
                raise ValueError("Polyhedral transport must bind its actual source/target prepared owners.")
    old_fv = (
        UnstructuredFiniteVolumePlan.from_cell_mesh(source.mesh, budget=budget).prepare(cell_geometry=source.geometry, budget=budget)
        if owners is None else owners[0].finite_volume
    )
    new_fv = (
        UnstructuredFiniteVolumePlan.from_cell_mesh(target.mesh, budget=budget).prepare(cell_geometry=target.geometry, budget=budget)
        if owners is None else owners[1].finite_volume
    )
    remap = UnstructuredConservativeRemapPlan(
        old_fv,
        new_fv,
        common.target_offsets,
        common.source_cells,
        common.volumes,
        method="common-refinement",
        provenance=common.refinement_id,
        intersection_error_bounds=common.volume_error_bounds,
    )
    old_epoch = TopologyEpoch(epoch, old_fv.geometry_id, old_fv.topology_id, "serial")
    new_epoch = TopologyEpoch(epoch + 1, new_fv.geometry_id, new_fv.topology_id, "serial")
    transfer = remap.epoch_transition(
        old_fv.cell_space, new_fv.cell_space, old_epoch, new_epoch
    )
    old_topology = CompositionEntry(
        source,
        entry_id="mesh",
        role="topology",
        owner_id="meshing",
        structure_id=old_epoch.epoch_id,
        revision_id=source.result_id,
        semantics_id="accepted-native-polyhedral-source",
    )
    new_topology = CompositionEntry(
        target,
        entry_id="mesh",
        role="topology",
        owner_id="meshing",
        structure_id=new_epoch.epoch_id,
        revision_id=target.result_id,
        semantics_id="accepted-native-polyhedral-source",
    )
    entries, transports, output = [old_topology], [], {}
    inventory_defects = {}
    return _PeriodicTransportPreparation(
        source=source,
        target=target,
        common=common,
        old_fv=old_fv,
        new_fv=new_fv,
        remap=remap,
        old_epoch=old_epoch,
        new_epoch=new_epoch,
        transfer=transfer,
        old_topology=old_topology,
        new_topology=new_topology,
        entries=entries,
        transports=transports,
        output=output,
        inventory_defects=inventory_defects,
        cell_fields=tuple(cell_roles),
        cell_roles=cell_roles,
    )


def _stage_periodic_cell_transport(
    prepared: _PeriodicTransportPreparation, fields: dict[str, jax.Array]
) -> None:
    source = prepared.source
    target = prepared.target
    old_fv = prepared.old_fv
    new_fv = prepared.new_fv
    remap = prepared.remap
    old_epoch = prepared.old_epoch
    new_epoch = prepared.new_epoch
    transfer = prepared.transfer
    old_topology = prepared.old_topology
    new_topology = prepared.new_topology
    entries = prepared.entries
    transports = prepared.transports
    output = prepared.output
    inventory_defects = prepared.inventory_defects
    cell_fields = prepared.cell_fields
    for name in cell_fields:
        if name not in fields:
            continue
        field_entries: list[tuple[str, Any, CompositionRole]] = [
            (name, fields[name], prepared.cell_roles[name]),
        ]
        for index, item in enumerate(fields.get(name + "_history", ())):
            field_entries.append((f"{name}-history:{index}", item, "history"))
        history_values = []
        for role_name, value, role in field_entries:
            value = jnp.asarray(value)
            target_value = remap.apply(value)
            components = (
                ((role_name, value, target_value),)
                if value.ndim == 1
                else tuple(
                    (
                        f"{role_name}:component{column}",
                        value[:, column],
                        target_value[:, column],
                    )
                    for column in range(value.shape[1])
                )
            )
            for component_name, before_value, after_value in components:
                old_entry = CompositionEntry(
                    before_value,
                    entry_id=component_name,
                    role=role,
                    owner_id="finite-volume",
                    structure_id=old_epoch.epoch_id,
                    revision_id=source.result_id,
                    semantics_id="cell-average",
                    dependencies=(old_topology.binding("structure"),),
                )
                new_entry = CompositionEntry(
                    after_value,
                    entry_id=component_name,
                    role=role,
                    owner_id="finite-volume",
                    structure_id=new_epoch.epoch_id,
                    revision_id=target.result_id,
                    semantics_id="cell-average",
                    dependencies=(new_topology.binding("structure"),),
                )
                entries.append(old_entry)
                transports.append(transfer.composition_transport(old_entry, new_entry))
            before = np.tensordot(
                np.asarray(old_fv.cell_volumes), np.asarray(value), axes=(0, 0)
            )
            after = np.tensordot(
                np.asarray(new_fv.cell_volumes), np.asarray(target_value), axes=(0, 0)
            )
            defect = float(np.max(np.abs(after - before)))
            if defect > 1e-10 * max(1.0, float(np.max(np.abs(before)))):
                raise qualification_failure(
                    "_polyhedral_qualification_periodic_transport",
                    (f"Conservative periodic transfer loses {role_name} inventory."),
                    {"before": before, "defect": defect},
                )
            inventory_defects[role_name] = defect
            if role_name == name:
                output[name] = target_value
            else:
                history_values.append(target_value)
        if name + "_history" in fields:
            output[name + "_history"] = jnp.stack(history_values)


@final
class _PeriodicNodalTransport(phx.StrictModule):
    nodal: phx.discretization.FieldTransfer
    nodal_epoch: FieldEpochTransition
    old_nodal_epoch: TopologyEpoch
    new_nodal_epoch: TopologyEpoch
    nodal_evidence: dict[str, Any]


def _prepare_periodic_nodal_transport(
    prepared: _PeriodicTransportPreparation,
    adaptation: phx.meshing.MeshAdaptationResult,
    fields: dict[str, jax.Array],
    epoch: int,
    *,
    owners: _PolyhedralOwnerPair | None = None,
    prepared_transfer: phx.discretization.FieldTransfer | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> _PeriodicNodalTransport:
    source = prepared.source
    target = prepared.target
    old_vem = (
        prepare_polyhedral_h1_virtual_element_3d(source.mesh, cell_geometry=source.geometry, budget=budget)
        if owners is None else owners[0].vem
    )
    new_vem = (
        prepare_polyhedral_h1_virtual_element_3d(target.mesh, cell_geometry=target.geometry, budget=budget)
        if owners is None else owners[1].vem
    )
    binding = TransferGeometryBinding(
        source.audit.geometry_id,
        target.audit.geometry_id,
        "topology-correspondence",
        source_topology_id=source.mesh.topology_id,
        target_topology_id=target.mesh.topology_id,
        coverage_defect=None,
    )
    if prepared_transfer is None:
        if adaptation.transfer is None:
            raise qualification_failure(
                "_polyhedral_qualification_periodic_transport",
                "Scalar VEM edge-trace transport requires its actual source-backed vertex stencil.",
                {"adaptation": adaptation},
            )
        nodal = old_vem.vertex_field_transfer(
            new_vem, adaptation.transfer, binding, field_name="vem"
        )
    else:
        nodal = prepared_transfer
        if nodal.source.field_space_id != old_vem.vertex_field_space("vem").field_space_id or nodal.target.field_space_id != new_vem.vertex_field_space("vem").field_space_id or nodal.geometry is None or nodal.geometry.source_geometry_id != binding.source_geometry_id or nodal.geometry.target_geometry_id != binding.target_geometry_id:
            raise ValueError("Prepared VEM reconstruction must bind the actual source and target field/geometry owners.")
    source_probe = jnp.sin(
        jnp.arange(old_vem.dof_count, dtype=fields["vem_solution"].dtype) + 0.3
    )
    target_probe = jnp.cos(jnp.arange(new_vem.dof_count, dtype=source_probe.dtype) + 0.7)
    dual_pullback = nodal.dual_pullback_operator
    if dual_pullback is None:
        raise qualification_failure(
            "_polyhedral_qualification_periodic_transport",
            ("Quotient VEM transfer omitted its adjoint-paired operator."),
            {"dual_pullback": dual_pullback},
        )
    pairing_left = jnp.vdot(target_probe, nodal.primal_operator.mv(source_probe))
    pairing_right = jnp.vdot(dual_pullback.mv(target_probe), source_probe)
    defect = float(jnp.abs(pairing_left - pairing_right))
    bound = float(
        100
        * jnp.finfo(source_probe.dtype).eps
        * jnp.maximum(
            jnp.maximum(jnp.abs(pairing_left), jnp.abs(pairing_right)),
            jnp.finfo(source_probe.dtype).tiny,
        )
    )
    if defect > bound:
        raise qualification_failure(
            "_polyhedral_qualification_periodic_transport",
            ("Nontrivial quotient VEM edge map has a mismatched raw transpose."),
            {"bound": bound, "defect": defect},
        )
    nodal_evidence = {
        "transfer_id": nodal.transfer_id,
        "source_dofs": old_vem.dof_count,
        "target_dofs": new_vem.dof_count,
        "raw_transpose_defect": defect,
        "storage_roundoff_bound": bound,
        "physical_stencil_present": adaptation.stencil is not None,
        "semantics": nodal.properties.semantics,
        "positivity_preserving": nodal.properties.positivity_preserving,
        "conservative": nodal.properties.conservative,
        "interpretation": "explicit continuous source-star H1 reconstruction, not unknown virtual-interior evaluation" if nodal.properties.semantics == "reconstruction" else "source-backed vertex/edge functionals",
    }
    if nodal.properties.semantics == "reconstruction":
        if owners is None:
            raise ValueError("VEM reconstruction oracles require the actual source-star physical owners.")
        constant_defect, affine_defect = _polyhedral_reconstruction_defects(
            nodal.primal_operator,
            owners[0].source_star.space.mesh.coordinates[:old_vem.dof_count],
            owners[1].source_star.space.mesh.coordinates[:new_vem.dof_count],
        )
        if max(float(constant_defect), float(affine_defect)) > 1e-7:
            raise ValueError("VEM reconstruction must satisfy the original affine reproduction accuracy.")
        nodal_evidence.update(
            constant_reproduction_defect=float(constant_defect),
            affine_reproduction_defect=float(affine_defect),
            source_star_plan_ids=(owners[0].source_star.plan_id, owners[1].source_star.plan_id),
        )
    old_nodal_epoch = TopologyEpoch(
        epoch, binding.source_geometry_id, source.mesh.topology_id, "serial"
    )
    new_nodal_epoch = TopologyEpoch(
        epoch + 1, binding.target_geometry_id, target.mesh.topology_id, "serial"
    )
    nodal_epoch = FieldEpochTransition(
        old_nodal_epoch,
        new_nodal_epoch,
        nodal,
        evidence_passed=bool(
            nodal.properties.constant_preserving
            and (nodal.properties.positivity_preserving or nodal.properties.semantics == "reconstruction")
            and nodal.properties.adjoint_paired
        ),
        evidence_id=nodal.transfer_id,
    )
    return _PeriodicNodalTransport(
        nodal=nodal,
        nodal_epoch=nodal_epoch,
        nodal_evidence=nodal_evidence,
        old_nodal_epoch=old_nodal_epoch,
        new_nodal_epoch=new_nodal_epoch,
    )


def _stage_periodic_nodal_transport(
    prepared: _PeriodicTransportPreparation,
    nodal_preparation: _PeriodicNodalTransport,
    fields: dict[str, jax.Array],
) -> None:
    source = prepared.source
    target = prepared.target
    old_topology = prepared.old_topology
    new_topology = prepared.new_topology
    entries = prepared.entries
    transports = prepared.transports
    output = prepared.output
    nodal = nodal_preparation.nodal
    nodal_epoch = nodal_preparation.nodal_epoch
    old_nodal_epoch = nodal_preparation.old_nodal_epoch
    new_nodal_epoch = nodal_preparation.new_nodal_epoch
    nodal_entries: list[tuple[str, jax.Array, CompositionRole]] = [
        ("vem_solution", fields["vem_solution"], "physical-state"),
    ]
    for index, item in enumerate(fields.get("vem_solution_history", ())):
        nodal_entries.append((f"vem_solution-history:{index}", item, "history"))
    for role_name, value, role in nodal_entries:
        target_value = nodal.primal_operator.mv(value)
        old_entry = CompositionEntry(
            value,
            entry_id=role_name,
            role=role,
            owner_id="virtual-element",
            structure_id=old_nodal_epoch.epoch_id,
            revision_id=source.result_id,
            semantics_id="scalar-quotient-vertex-trace",
            dependencies=(old_topology.binding("structure"),),
        )
        new_entry = CompositionEntry(
            target_value,
            entry_id=role_name,
            role=role,
            owner_id="virtual-element",
            structure_id=new_nodal_epoch.epoch_id,
            revision_id=target.result_id,
            semantics_id="scalar-quotient-vertex-trace",
            dependencies=(new_topology.binding("structure"),),
        )
        entries.append(old_entry)
        transports.append(nodal_epoch.composition_transport(old_entry, new_entry))
        if role == "physical-state":
            output["vem_solution"] = target_value
    output["vem_solution_history"] = jnp.stack(
        tuple(nodal.primal_operator.mv(item) for item in fields["vem_solution_history"])
    )


def _stage_periodic_fem_transport(
    prepared: _PeriodicTransportPreparation,
    fields: dict[str, jax.Array],
    epoch: int,
    *,
    owners: _PolyhedralOwnerPair | None = None,
    prepared_transfer: phx.discretization.FieldTransfer | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> dict[str, Any]:
    old = (
        polyhedral_consumer_owner.prepare_polyhedral_source_star_fem(prepared.source, budget=budget)
        if owners is None else owners[0].source_star
    )
    new = (
        polyhedral_consumer_owner.prepare_polyhedral_source_star_fem(prepared.target, budget=budget)
        if owners is None else owners[1].source_star
    )
    transfer = (
        polyhedral_consumer_owner.prepare_polyhedral_source_star_field_transfer(old, new, budget=budget)
        if prepared_transfer is None else prepared_transfer
    )
    if transfer.source.field_space_id != old.space.field_spaces[0].field_space_id or transfer.target.field_space_id != new.space.field_spaces[0].field_space_id:
        raise ValueError("Prepared FE transport must retain its actual source-star spaces.")
    operator = transfer.primal_operator
    source_probe = jnp.sin(jnp.arange(operator.source.size, dtype=jnp.float64) + 0.3)
    target_probe = jnp.cos(jnp.arange(operator.target.size, dtype=jnp.float64) + 0.7)
    left = jnp.vdot(target_probe, operator.mv(source_probe))
    right = jnp.vdot(operator.transpose_mv(target_probe), source_probe)
    transpose_defect = float(jnp.abs(left - right))
    transpose_bound = float(100 * jnp.finfo(jnp.float64).eps * jnp.maximum(
        jnp.maximum(jnp.abs(left), jnp.abs(right)), jnp.finfo(jnp.float64).tiny
    ))
    if transpose_defect > transpose_bound:
        raise ValueError("polyhedral.fem.transfer must cover the whole target with an adjoint-paired source reconstruction")
    old_epoch = TopologyEpoch(epoch, old.source_geometry_id, old.space.mesh.topology_id, "serial")
    new_epoch = TopologyEpoch(epoch + 1, new.source_geometry_id, new.space.mesh.topology_id, "serial")
    transition = FieldEpochTransition(
        old_epoch, new_epoch, transfer,
        evidence_passed=transpose_defect <= transpose_bound,
        evidence_id=transfer.transfer_id,
    )
    rows = (fields["fem_solution"], *tuple(fields["fem_solution_history"]))
    results = []
    for index, values in enumerate(rows):
        name = "fem_solution" if index == 0 else f"fem_solution-history:{index - 1}"
        role: CompositionRole = "physical-state" if index == 0 else "history"
        target_values = operator.mv(values)
        previous = CompositionEntry(
            values, entry_id=name, role=role, owner_id="finite-element",
            structure_id=old_epoch.epoch_id, revision_id=prepared.source.result_id,
            semantics_id="continuous-p1-source-star-interpolation",
            dependencies=(prepared.old_topology.binding("structure"),),
        )
        successor = CompositionEntry(
            target_values, entry_id=name, role=role, owner_id="finite-element",
            structure_id=new_epoch.epoch_id, revision_id=prepared.target.result_id,
            semantics_id="continuous-p1-source-star-interpolation",
            dependencies=(prepared.new_topology.binding("structure"),),
        )
        prepared.entries.append(previous)
        prepared.transports.append(transition.composition_transport(previous, successor))
        results.append(target_values)
    prepared.output["fem_solution"] = results[0]
    prepared.output["fem_solution_history"] = jnp.stack(results[1:])
    return {
        "transfer_id": transfer.transfer_id,
        "source_plan_id": old.plan_id,
        "target_plan_id": new.plan_id,
        "point_functional_id": operator.operator_id,
        "raw_transpose_defect": transpose_defect,
        "storage_roundoff_bound": transpose_bound,
        "history_states_transferred": len(rows) - 1,
        "semantics": "owned conforming P1 source-star reconstruction; not nested-star equality",
    }


def _commit_periodic_transport(
    prepared: _PeriodicTransportPreparation, nodal_evidence: dict[str, Any] | None
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    source = prepared.source
    target = prepared.target
    common = prepared.common
    new_topology = prepared.new_topology
    entries = prepared.entries
    transports = prepared.transports
    output = prepared.output
    inventory_defects = prepared.inventory_defects
    cell_fields = prepared.cell_fields
    receipt = commit_composition_rebind(
        CompositionRebind(
            Composition(tuple(entries), boundary_id="accepted-native-polyhedral-step"),
            reprepare=(new_topology,),
            transports=tuple(transports),
        ),
        accepted_boundary=True,
    )
    if not receipt.published:
        raise qualification_failure(
            "_polyhedral_qualification_periodic_transport",
            ("ALL physical/history polyhedral composition did not publish atomically."),
            {"receipt": receipt},
        )
    for name in cell_fields:
        if name in output:
            if output[name].ndim == 1:
                output[name] = jnp.asarray(receipt.composition.value(name))
            else:
                output[name] = jnp.stack(
                    tuple(
                        jnp.asarray(
                            receipt.composition.value(f"{name}:component{column}")
                        )
                        for column in range(output[name].shape[1])
                    ),
                    axis=1,
                )
    for name in ("vem_solution", "fem_solution"):
        if name in output:
            output[name] = jnp.asarray(receipt.composition.value(name))
    return output, {
        "receipt_id": receipt.receipt_id,
        "published": receipt.published,
        "source_result_id": source.result_id,
        "target_result_id": target.result_id,
        "physical_roles": tuple(
            entry.entry_id for entry in entries if entry.role == "physical-state"
        ),
        "history_roles": tuple(
            entry.entry_id for entry in entries if entry.role == "history"
        ),
        "inventory_defects": inventory_defects,
        "common_refinement_id": common.refinement_id,
        "nodal_semantics": None if nodal_evidence is None else nodal_evidence["interpretation"],
        "nodal_transfer": nodal_evidence,
        "derived_geometry_flux_and_residuals": "old accepted epoch retained; target reprepared from transported physical state",
    }


def _polyhedral_qualification_periodic_transport(
    adaptation: phx.meshing.MeshAdaptationResult,
    fields: dict[str, jax.Array],
    epoch: int,
    *,
    cell_roles: dict[str, CompositionRole] | None = None,
    owners: _PolyhedralOwnerPair | None = None,
    vem_transfer: phx.discretization.FieldTransfer | None = None,
    fem_transfer: phx.discretization.FieldTransfer | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    if cell_roles is None:
        cell_roles = {
            "conserved": "physical-state",
            "pressure": "physical-state",
            "temperature": "physical-state",
            "fv_cell_solution": "physical-state",
        }
    prepared = _prepare_periodic_transport(adaptation, epoch, cell_roles, owners, budget)
    _stage_periodic_cell_transport(prepared, fields)
    nodal_evidence = None
    if "vem_solution" in fields:
        nodal_preparation = _prepare_periodic_nodal_transport(
            prepared, adaptation, fields, epoch,
            owners=owners, prepared_transfer=vem_transfer, budget=budget,
        )
        _stage_periodic_nodal_transport(prepared, nodal_preparation, fields)
        nodal_evidence = nodal_preparation.nodal_evidence
    fem_evidence = None
    if "fem_solution" in fields:
        fem_evidence = _stage_periodic_fem_transport(
            prepared, fields, epoch, owners=owners, prepared_transfer=fem_transfer, budget=budget
        )
    fields, proof = _commit_periodic_transport(prepared, nodal_evidence)
    return fields, {
        **proof, "adaptation_status": adaptation.status.value,
        "fem_transfer": fem_evidence,
    }


def _polyhedral_qualification_periodic_adapt(
    result: phx.meshing.CellMeshingResult,
    request: phx.meshing.PolyhedralMeshAdaptation,
    limits: phx.meshing.MeshingLimits,
    *,
    coordinate_operands: CoordinateOperandCollector | None = None,
) -> phx.meshing.MeshAdaptationResult:
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_POLYHEDRAL,
        limits=limits,
        audit_policy=phx.meshing.CellMeshAuditPolicy(
            require_complete_association=True,
            watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
        ),
    )
    return phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            result, request, policy=policy, observe_managed_admission=True
        ),
        coordinate_operands=coordinate_operands,
        observe_managed_admission=True,
    )


def _polyhedral_qualification_periodic_inverse_groups(
    adaptation: Any,
) -> tuple[tuple[int, ...], ...]:

    common = adaptation.common_refinement
    if common is None or not common.succeeded:
        raise qualification_failure(
            "_polyhedral_qualification_periodic_inverse_groups",
            ("Periodic inverse requires certified source/target overlap."),
            {"common": common},
        )
    sources, targets = np.asarray(common.source_cells), np.asarray(common.target_cells)
    ids = np.asarray(adaptation.target.mesh.connectivity.cell_global_ids)
    owner = {}
    groups = {}
    for source, target in zip(sources, targets, strict=True):
        if int(target) in owner and owner[int(target)] != int(source):
            raise qualification_failure(
                "_polyhedral_qualification_periodic_inverse_groups",
                (
                    "A split child overlaps multiple source parents; it has no simple inverse group."
                ),
                {"owner": owner, "source": source, "target": target},
            )
        owner[int(target)] = int(source)
        groups.setdefault(int(source), set()).add(int(ids[target]))
    if set(owner) != set(range(len(ids))):
        raise qualification_failure(
            "_polyhedral_qualification_periodic_inverse_groups",
            ("Certified periodic inverse groups omit target cells."),
            {"ids": ids, "owner": owner},
        )
    return tuple(
        tuple(sorted(values)) for _, values in sorted(groups.items()) if len(values) > 1
    )


@final
class _PolyhedralLifecycleContext(phx.StrictModule):
    args: QualificationControls
    case: MeshingCase
    started: float
    deadline: Deadline
    stages: dict[str, float]
    phases: list[NativeMeshingPhaseMeasurement]
    record: dict[str, Any]
    ledger: StageLedger
    set_phase: Callable[[str], None]
    coordinate_operands: dict[str, PreparedCoordinateOperands] = field(default_factory=dict)


def _record_polyhedral_warm_publication(
    context: _PolyhedralLifecycleContext, epoch: MeshingAcceptedEpoch, index: int,
) -> None:
    """Publish the actual sealed warm carrier, never a resource-trial surrogate."""
    if epoch.carrier.execution_evidence is None:
        raise ValueError("Warm scientific publication must retain its actual ended native consumer receipt.")
    publications = context.record.setdefault("warm_publications", [])
    if len(publications) != index:
        raise ValueError("Warm scientific publications must retain every actual epoch in accepted order.")
    publications.append({
        "epoch": index, "epoch_id": epoch.epoch_id,
        "result": _result_identity(epoch.carrier),
    })


def _remember_polyhedral_operands(
    context: _PolyhedralLifecycleContext,
    carrier: phx.meshing.CellMeshingResult,
    collector: CoordinateOperandCollector,
) -> None:
    if collector.prepared is None:
        raise qualification_failure(
            "_remember_polyhedral_operands",
            "Actual native publication did not issue its numeric coordinate bank.",
            {"result_id": carrier.result_id},
        )
    context.coordinate_operands[carrier.result_id] = collector.prepared


@final
class _PeriodicAcceptedEpochs(phx.StrictModule):
    current: phx.meshing.CellMeshingResult
    fields: dict[str, jax.Array]
    epochs: list[MeshingAcceptedEpoch]
    physics_records: list[dict[str, Any]]
    transfers: list[dict[str, Any]]
    error: float


@final
class _PeriodicLifecyclePublication(phx.StrictModule):
    source: phx.meshing.NativePolyhedralSource
    request: phx.meshing.VolumeMeshingSpec
    options: phx.meshing.NativeMeshingOptions
    original: phx.meshing.CellMeshingResult
    plan: phx.meshing.NativeMeshingPlan
    state: _PeriodicAcceptedEpochs


def _record_first_periodic_target(
    context: _PolyhedralLifecycleContext, error: float
) -> None:
    if (
        context.record["time_to_target_error_seconds"] is None
        and 0.0 <= error <= context.args.target_error
    ):
        context.record["time_to_target_error_seconds"] = perf_counter() - context.started


@final
class _PeriodicAcceptedScientificState(phx.StrictModule):
    epoch: MeshingAcceptedEpoch
    fields: dict[str, jax.Array]
    physics: dict[str, Any]
    error: float
    transfer: dict[str, Any] | None


def _accept_periodic_native_epoch(
    context: _PolyhedralLifecycleContext,
    carrier: phx.meshing.CellMeshingResult,
    limits: phx.meshing.MeshingLimits,
    epoch_index: int,
    *,
    previous: MeshingAcceptedEpoch | None = None,
    transition: phx.meshing.MeshAdaptationResult | None = None,
) -> _PeriodicAcceptedScientificState:
    def accept(budget: CoordinateEnclosureBudget) -> _PeriodicAcceptedScientificState:
        _, owners = polyhedral_field_owner.prepare_meshing_polyhedral_owner_binding(
            carrier, budget=budget
        )
        transfer = None
        if previous is None:
            if epoch_index != 0 or transition is not None:
                raise ValueError("Periodic generation acceptance requires the actual first carrier without a transition.")
            fields = _polyhedral_qualification_periodic_initial_fields(owners)
        else:
            if transition is None:
                raise ValueError("Periodic successor acceptance requires its genuine adaptation transition.")
            old = _polyhedral_epoch_owners(previous, budget=budget)
            fem_transfer = polyhedral_consumer_owner.prepare_polyhedral_source_star_field_transfer(
                old.source_star, owners.source_star, budget=budget
            )
            fields, transfer = _polyhedral_qualification_periodic_transport(
                transition,
                {name: jnp.asarray(value) for name, value in previous.fields.items()},
                epoch_index - 1,
                owners=(old, owners), fem_transfer=fem_transfer, budget=budget,
            )
        _polyhedral_acceptance(carrier, context.case)
        physics, error, fields = _polyhedral_qualification_periodic_physics(
            carrier, fields, context.args.repeats, context.stages,
            seeds=None if previous is None else fields,
            deadline=context.deadline, budget=budget, owners=owners,
        )
        epoch = _prepare_polyhedral_accepted_epoch(
            carrier, fields, epoch_index,
            transition=transition, owners=owners, budget=budget,
        )
        return _PeriodicAcceptedScientificState(
            epoch=epoch, fields=fields, physics=physics, error=error, transfer=transfer
        )

    stage = f"epoch{epoch_index}_complete_native_scientific_consumer"
    context.set_phase(stage)
    consumed, measured = context.ledger.run(
        stage,
        lambda: _measured(
            lambda: execute_native_consumer(
                carrier, limits, accept, failure_factory=qualification_failure,
                prepared_operands=context.coordinate_operands.get(carrier.result_id),
                observe_managed_admission=True,
            )
        ),
    )
    context.stages[stage] = measured["wall_seconds"]
    value = consumed.value
    sealed_transition = None
    if transition is not None:
        if previous is None:
            raise ValueError("Periodic adaptation acceptance must retain its actual preceding epoch.")
        sealed_transition = polyhedral_field_owner.rebind_meshing_adaptation_execution_endpoints(
            transition, previous.carrier, consumed.result
        )
    sealed_epoch = polyhedral_field_owner.rebind_meshing_accepted_epoch_execution(
        value.epoch, consumed.result, transition=sealed_transition
    )
    _record_polyhedral_warm_publication(context, sealed_epoch, epoch_index)
    return _PeriodicAcceptedScientificState(
        epoch=sealed_epoch, fields=value.fields, physics=value.physics,
        error=value.error, transfer=value.transfer,
    )


def _publish_periodic_lifecycle(
    context: _PolyhedralLifecycleContext,
) -> _PeriodicLifecyclePublication:
    args = context.args
    case = context.case
    stages = context.stages
    phases = context.phases
    record = context.record
    ledger = context.ledger
    source, request, options = prepare_polyhedral_case(
        case.name, args.resolution, args.capacity, args.timeout, source=True,
        maximum_work_units=args.maximum_work_units,
        maximum_geometry_queries=args.maximum_geometry_queries,
        maximum_scratch_bytes=args.maximum_scratch_bytes,
    )
    if (
        not isinstance(source, phx.meshing.NativePolyhedralSource)
        or len(source.sites) != 2
        or len(request.periodic_constraints) != 3
    ):
        raise qualification_failure(
            "_polyhedral_qualification_periodic_lifecycle",
            ("Periodic lifecycle source or controls were replaced."),
            {"request": request, "source": source},
        )
    record["source_id"], record["source_revision"] = (
        source.source_id,
        source.source_revision,
    )
    record["request_id"] = request.specification_id
    phase = "source_generation_publication_certification"
    context.set_phase(phase)
    provider = phx.meshing.NativeMeshingProvider(options)
    plan, measured = ledger.run(
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
    stages["source_planning"] = measured["wall_seconds"]
    coordinate_operands = CoordinateOperandCollector()
    current, measured = ledger.run(
        phase, lambda: _measured(
            lambda: plan.execute(
                record_phase=phases.append, coordinate_operands=coordinate_operands,
            )
        )
    )
    stages[phase] = measured["wall_seconds"]
    _remember_polyhedral_operands(context, current, coordinate_operands)
    _, record["source_acceptance"] = _polyhedral_acceptance(current, case)
    original = current
    accepted = _accept_periodic_native_epoch(context, current, request.limits, 0)
    current = accepted.epoch.carrier
    fields, physics, error = accepted.fields, accepted.physics, accepted.error
    epochs, physics_records, transfers = [accepted.epoch], [physics], []
    _record_first_periodic_target(context, error)
    state = _PeriodicAcceptedEpochs(
        current=current,
        fields=fields,
        epochs=epochs,
        physics_records=physics_records,
        transfers=transfers,
        error=error,
    )
    return _PeriodicLifecyclePublication(
        source=source, request=request, options=options, original=original, state=state,
        plan=plan,
    )


@final
class _PeriodicRefinedEpoch(phx.StrictModule):
    state: _PeriodicAcceptedEpochs
    inverse_groups: tuple[tuple[int, ...], ...]


def _refine_periodic_lifecycle_epoch(
    context: _PolyhedralLifecycleContext,
    publication: _PeriodicLifecyclePublication,
    state: _PeriodicAcceptedEpochs,
    cycle: int,
) -> _PeriodicRefinedEpoch:
    stages = context.stages
    ledger = context.ledger
    request = publication.request
    current = state.current
    epochs = state.epochs
    physics_records = state.physics_records
    transfers = state.transfers
    error = state.error
    phase = f"round{cycle}_actual_plane_refinement"
    context.set_phase(phase)
    split_request = phx.meshing.PolyhedralMeshAdaptation(
        phx.meshing.PolyhedralAdaptationOperation.PLANE_SPLIT,
        planes=((1.0, 0.0, 0.0, 0.25), (1.0, 0.0, 0.0, 0.75)),
    )
    coordinate_operands = CoordinateOperandCollector()
    split, measured = ledger.run(
        phase,
        lambda: _measured(
            lambda: _polyhedral_qualification_periodic_adapt(
                current, split_request, request.limits,
                coordinate_operands=coordinate_operands,
            )
        ),
    )
    stages[phase] = measured["wall_seconds"]
    _remember_polyhedral_operands(context, split.target, coordinate_operands)
    if split.target.mesh.connectivity.cell_count <= current.mesh.connectivity.cell_count:
        raise qualification_failure(
            "_polyhedral_qualification_periodic_lifecycle",
            ("Requested periodic refinement did not subdivide cells."),
            {"current": current, "split": split},
        )
    inverse_groups = _polyhedral_qualification_periodic_inverse_groups(split)
    accepted = _accept_periodic_native_epoch(
        context, split.target, request.limits, len(epochs),
        previous=epochs[-1], transition=split,
    )
    if accepted.transfer is None:
        raise ValueError("Periodic refinement must retain its real all-field transition proof.")
    transfers.append(accepted.transfer)
    current = accepted.epoch.carrier
    fields, physics, target_error = accepted.fields, accepted.physics, accepted.error
    if physics["vem"]["physical_response"] <= 1e-10:
        raise qualification_failure(
            "_polyhedral_qualification_periodic_lifecycle",
            ("Refined source did not produce a nonconstant quotient VEM response."),
            {"physics": physics},
        )
    _record_first_periodic_target(context, target_error)
    error = max(error, target_error)
    epochs.append(accepted.epoch)
    physics_records.append(physics)
    state = _PeriodicAcceptedEpochs(
        current=current,
        fields=fields,
        epochs=epochs,
        physics_records=physics_records,
        transfers=transfers,
        error=error,
    )
    return _PeriodicRefinedEpoch(state=state, inverse_groups=inverse_groups)


def _coarsen_periodic_lifecycle_epoch(
    context: _PolyhedralLifecycleContext,
    publication: _PeriodicLifecyclePublication,
    refined: _PeriodicRefinedEpoch,
    cycle: int,
) -> _PeriodicAcceptedEpochs:
    stages = context.stages
    ledger = context.ledger
    request = publication.request
    original = publication.original
    state, inverse_groups = refined.state, refined.inverse_groups
    current = state.current
    epochs = state.epochs
    physics_records = state.physics_records
    transfers = state.transfers
    error = state.error
    phase = f"round{cycle}_actual_inverse_agglomeration"
    context.set_phase(phase)
    coarse_request = phx.meshing.PolyhedralMeshAdaptation(
        phx.meshing.PolyhedralAdaptationOperation.AGGLOMERATE,
        agglomerations=inverse_groups,
    )
    coordinate_operands = CoordinateOperandCollector()
    coarse, measured = ledger.run(
        phase,
        lambda: _measured(
            lambda: _polyhedral_qualification_periodic_adapt(
                current, coarse_request, request.limits,
                coordinate_operands=coordinate_operands,
            )
        ),
    )
    stages[phase] = measured["wall_seconds"]
    _remember_polyhedral_operands(context, coarse.target, coordinate_operands)
    accepted = _accept_periodic_native_epoch(
        context, coarse.target, request.limits, len(epochs),
        previous=epochs[-1], transition=coarse,
    )
    if accepted.transfer is None:
        raise ValueError("Periodic inverse must retain its real all-field transition proof.")
    transfers.append(accepted.transfer)
    current = accepted.epoch.carrier
    if current.mesh.connectivity.cell_count != original.mesh.connectivity.cell_count:
        raise qualification_failure(
            "_polyhedral_qualification_periodic_lifecycle",
            ("Periodic inverse failed to restore initial cell coverage."),
            {"current": current, "original": original},
        )
    fields, physics, target_error = accepted.fields, accepted.physics, accepted.error
    _record_first_periodic_target(context, target_error)
    error = max(error, target_error)
    epochs.append(accepted.epoch)
    physics_records.append(physics)
    return _PeriodicAcceptedEpochs(
        current=current,
        fields=fields,
        epochs=epochs,
        physics_records=physics_records,
        transfers=transfers,
        error=error,
    )


def _join_polyhedral_cold_and_source_prefix(
    context: _PolyhedralLifecycleContext,
    source: phx.meshing.NativePlcSource | phx.meshing.NativePolyhedralSource,
    request: phx.meshing.VolumeMeshingSpec,
    options: phx.meshing.NativeMeshingOptions,
    publication: phx.meshing.CellMeshingResult,
    cold: Callable[[], dict[str, Any]],
    cold_phase: str,
    prepared_plan: phx.meshing.NativeMeshingPlan,
) -> None:
    """Join independent actual requests without masking either refusal."""
    cold_failure: Exception | None = None
    resource_failure: Exception | None = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        resources = executor.submit(
            _polyhedral_source_prefix_resource_refusals,
            source, request, options, publication, campaign_started=context.started,
            prepared_plan=prepared_plan,
        )
        context.set_phase(cold_phase)
        try:
            context.record["cold"] = cold()
        except Exception as failure:
            cold_failure = failure
            refusal: dict[str, Any] = {"status": "failed"}
            record_failure(refusal, failure, cold_phase)
            context.record["cold"] = refusal
        try:
            context.record["source_prefix_resources"] = resources.result()
        except Exception as failure:
            resource_failure = failure
            refusal = {"status": "failed", "admitted_scope": "SOURCEPREFIX"}
            record_failure(refusal, failure, "source_prefix_resource_boundary_proof")
            context.record["source_prefix_resources"] = refusal
    if cold_failure is not None:
        raise cold_failure from resource_failure
    if resource_failure is not None:
        raise resource_failure


def _complete_periodic_lifecycle(
    context: _PolyhedralLifecycleContext,
    publication: _PeriodicLifecyclePublication,
    state: _PeriodicAcceptedEpochs,
) -> None:
    args = context.args
    started = context.started
    stages = context.stages
    record = context.record
    source = publication.source
    request = publication.request
    options = publication.options
    original = publication.original
    epochs = state.epochs
    physics_records = state.physics_records
    transfers = state.transfers
    error = state.error
    # Retain every completed scientific epoch even when a later resource or
    # independent-process continuation refuses.
    record.update(physics=physics_records, physical_error=error, transfers=transfers)
    complete_resources = [
        _complete_native_request_resources(request, epoch.carrier) for epoch in epochs
    ]
    missing = [] if all(item["within_original_bounds"] for item in complete_resources) else [
        "complete_native_request_scientific_consumers"
    ]
    record["resources"] = {
        "scope": "complete warm native generation/adaptation scientific consumer requests",
        "mandatory_capacity_gate": "incomplete" if missing else "complete",
        "unobserved_or_unenforced_original_bounds": missing,
        "complete_native_requests": complete_resources,
        "complete_request_R_claimed": False,
    }
    # Reduced-limit requests consume only the immutable original publication;
    # their independent scopes can run while a fresh process restores the
    # accepted epochs. Both branches retain the same launch-origin expiry.
    _join_polyhedral_cold_and_source_prefix(
        context, source, request, options, original,
        lambda: _polyhedral_qualification_periodic_cold(
            source, request, options, tuple(epochs),
            args.repeats, args.target_error, stages, context.deadline,
            expected_build_id=record["runtime"]["build_id"],
        ),
        "default_source_allfield_history_fresh_cold_continued_pde",
        publication.plan,
    )
    if record["cold"]["status"] != "passed":
        raise qualification_failure(
            "_polyhedral_qualification_periodic_lifecycle",
            ("Default cold source continuation did not pass."),
            {"record": record},
        )
    elapsed = perf_counter() - started
    obligations = _periodic_completion_obligations(record, args, state, elapsed)
    complete = all(obligations.values())
    scientific = all(
        passed
        for name, passed in obligations.items()
        if name not in ("original_resource_capacity_gate", "source_prefix_resource_boundary", "wall_time")
    )
    record.update(
        status="timeout"
        if elapsed > args.timeout
        else "passed"
        if complete
        else "failed",
        physical_error=error,
        target_error_reached=error <= args.target_error,
        completion_obligations=obligations,
        case_contract_completion="complete" if complete else "incomplete",
        mandatory_workflow_completion="complete" if complete else "incomplete",
        physics=physics_records,
        canonical_adaptation={
            "status": "complete"
            if obligations["refine_coarsen_orbit_lifecycle"]
            else "incomplete",
            "cycles": args.adaptation_rounds,
            "transfers": transfers,
        },
        qualification_scale={
            "site_count": len(source.sites),
            "epoch_cell_counts": [
                epoch.carrier.mesh.connectivity.cell_count for epoch in epochs
            ],
            "epoch_face_counts": [
                epoch.carrier.mesh.entity_set(2).entity_ids.size for epoch in epochs
            ],
        },
        demanding_controls=[
            "three-axis-zero-tolerance-periodic-quotient",
            "plane-split-refinement",
            "inverse-agglomeration-coarsening",
            "allfield-history-conservative-transfer",
            "fresh-process-cold-continuation",
        ],
        accepted_epochs=[epoch.epoch_id for epoch in epochs],
        logical_retained_bytes=model_array_bytes((source, request, options, epochs)),
        independent_scientific_certification="passed" if scientific else "failed",
    )


def _periodic_completion_obligations(
    record: dict[str, Any],
    args: QualificationControls,
    state: _PeriodicAcceptedEpochs,
    elapsed: float,
) -> dict[str, bool]:
    """Derive every periodic lifecycle obligation from emitted evidence alone."""
    rounds = args.adaptation_rounds
    transfers = state.transfers
    acceptance = record.get("source_acceptance", {})
    cold = record.get("cold", {})
    prefix = record.get("source_prefix_resources", {})
    return {
        "certified_periodic_power_generation": acceptance.get("passed") is True
        and acceptance.get("coverage_status") == "certified",
        "quotient_vem_fv_physics_every_epoch": len(state.physics_records)
        == 1 + 2 * rounds
        and bool(np.isfinite(state.error))
        and 0.0 <= state.error <= args.target_error,
        "conforming_fe_physics_every_epoch": len(state.physics_records) == 1 + 2 * rounds
        and all(
            item.get("fe", {}).get("physical_l2_error", np.inf) <= args.target_error
            and item.get("fe", {}).get("maximum_free_physical_residual", np.inf) <= args.target_error
            and item.get("fe", {}).get("vector_flux_finite") is True
            for item in state.physics_records
        ),
        "accepted_fe_history_transferred": len(transfers) == 2 * rounds
        and all(
            proof.get("fem_transfer", {}).get("history_states_transferred") == 3
            for proof in transfers
        ),
        "refine_coarsen_orbit_lifecycle": rounds >= 1
        and len(transfers) == 2 * rounds
        and all(
            proof["published"] is True
            and proof["adaptation_status"]
            == phx.meshing.MeshAdaptationStatus.COMPLETE.value
            for proof in transfers
        ),
        "conservative_allfield_history_transfer": len(transfers) == 2 * rounds
        and all(
            {"conserved", "conserved-history:0", "pressure", "temperature"}
            <= set(proof["inventory_defects"])
            for proof in transfers
        ),
        "fresh_process_cold_continuation": cold.get("status") == "passed"
        and cold.get("accepted_fields_bitwise_preserved") is True,
        "original_resource_capacity_gate": record["resources"][
            "mandatory_capacity_gate"
        ]
        == "complete",
        "source_prefix_resource_boundary": prefix.get("status") == "passed"
        and prefix.get("mandatory_capacity_gate") == "complete"
        and prefix.get("admitted_scope") == "SOURCEPREFIX"
        and prefix.get("complete_request_R_claimed") is False
        and prefix.get("original_source_and_controls_preserved") is True,
        "wall_time": elapsed <= args.timeout,
    }


def _polyhedral_qualification_periodic_lifecycle(
    args: QualificationControls,
) -> dict[str, Any]:
    started = perf_counter()
    deadline = Deadline(started + args.timeout)
    stages = {"_campaign_started": started}
    phases = []
    case = polyhedral_case(args.polyhedral_source)
    record = {
        "scenario": "native-polyhedral-lifecycle",
        "case": polyhedral_identity(case.name),
        "status": "failed",
        "engine": "native",
        "runtime": _runtime_identity(),
        "control_capacities": {
            "site_count": 2,
            "requested_resolution": args.resolution,
            "entity_capacity": args.capacity,
        },
        "targets": {
            "maximum_physics_error": args.target_error,
            "wall_seconds": args.timeout,
        },
        "stages_seconds": stages,
        "time_to_target_error_seconds": None,
        "mandatory_workflow_completion": "incomplete",
        "release": "unassessed",
        "leadership": "unassessed",
    }
    _ledger = StageLedger(record, deadline=deadline)
    phase = "original_source_preparation"

    def set_phase(name: str) -> None:
        nonlocal phase
        phase = name

    context = _PolyhedralLifecycleContext(
        args=args,
        case=case,
        started=started,
        deadline=deadline,
        stages=stages,
        phases=phases,
        record=record,
        ledger=_ledger,
        set_phase=set_phase,
    )
    try:
        publication = _publish_periodic_lifecycle(context)
        state = publication.state
        for cycle in range(args.adaptation_rounds):
            refined = _refine_periodic_lifecycle_epoch(context, publication, state, cycle)
            state = _coarsen_periodic_lifecycle_epoch(
                context, publication, refined, cycle
            )
        _complete_periodic_lifecycle(context, publication, state)
    except Exception as failure:
        record_failure(record, failure, phase)
    finally:
        record["end_to_end_seconds"] = perf_counter() - started
        record["native_phases"] = _phase_recordings(phases)
        record.update(_polyhedral_memory_measurements())
    return record


@final
class _PolyhedralGeneration(phx.StrictModule):
    results: dict[str, phx.meshing.CellMeshingResult]
    labels: dict[str, np.ndarray]
    acceptance: dict[str, dict[str, object]]
    plans: list[phx.meshing.NativeMeshingPlan]
    geometry: phx.meshing.NativePolyhedralSource
    request: phx.meshing.VolumeMeshingSpec
    options: phx.meshing.NativeMeshingOptions
    source_publication: phx.meshing.CellMeshingResult


def _publish_polyhedral_generation(
    context: _PolyhedralLifecycleContext,
) -> _PolyhedralGeneration:
    args = context.args
    case = context.case
    stages = context.stages
    phases = context.phases
    record = context.record
    ledger = context.ledger
    results, labels, acceptance, plans = {}, {}, {}, []
    for epoch in ("source", "target"):
        phase = f"{epoch}_preparation"
        context.set_phase(phase)
        prepared, measured = ledger.run(
            phase,
            lambda: _measured(
                lambda epoch=epoch: prepare_polyhedral_case(
                    case.name,
                    args.resolution,
                    args.capacity,
                    args.timeout,
                    source=epoch == "source",
                    maximum_work_units=args.maximum_work_units,
                    maximum_geometry_queries=args.maximum_geometry_queries,
                    maximum_scratch_bytes=args.maximum_scratch_bytes,
                )
            ),
        )
        geometry, request, options = prepared
        stages[phase] = measured["wall_seconds"]
        record[f"{epoch}_request_id"] = request.specification_id
        if epoch == "target":
            record["request_id"] = request.specification_id
        record[f"{epoch}_source_revision"] = geometry.source_revision
        phase = f"{epoch}_planning"
        context.set_phase(phase)
        provider = phx.meshing.NativeMeshingProvider(options)
        plan, measured = ledger.run(
            phase,
            lambda: _measured(
                lambda: provider.plan(
                    geometry,
                    request,
                    coordinate_contract=phx.SpatialCoordinateContract.si(),
                    record_phase=phases.append,
                )
            ),
        )
        stages[phase] = measured["wall_seconds"]
        plans.append(plan)
        phase = f"{epoch}_generation_publication_certification"
        context.set_phase(phase)
        coordinate_operands = CoordinateOperandCollector() if epoch == "source" else None
        result, measured = ledger.run(
            phase,
            lambda: _measured(
                lambda: plan.execute(
                    record_phase=phases.append, coordinate_operands=coordinate_operands,
                )
            ),
        )
        stages[phase] = measured["wall_seconds"]
        if coordinate_operands is not None:
            _remember_polyhedral_operands(context, result, coordinate_operands)
        phase = f"{epoch}_independent_acceptance"
        context.set_phase(phase)
        checked, measured = ledger.run(
            phase, lambda: _measured(lambda: _polyhedral_acceptance(result, case))
        )
        labels[epoch], acceptance[epoch] = checked
        stages[phase] = measured["wall_seconds"]
        bounds = _observe_native_resources(request, result).bounds
        record.setdefault("resources", {})[epoch] = {
            "within_original_bounds": all(
                item["within_original_bound"] is not False for item in bounds.values()
            ),
            "bounds": bounds,
        }
        results[epoch] = result
    return _PolyhedralGeneration(
        results=results,
        labels=labels,
        acceptance=acceptance,
        plans=plans,
        geometry=geometry,
        request=request,
        options=options,
        source_publication=results["source"],
    )


def _accept_material_source_physics(
    context: _PolyhedralLifecycleContext, generation: _PolyhedralGeneration
) -> tuple[_PolyhedralTransferredPhysics, MeshingAcceptedEpoch]:
    source = generation.results["source"]
    inventory = np.where(generation.labels["source"] == 0, 2.0, 5.0)
    admitted = {
        "inventory": jnp.asarray(inventory),
        "previous_inventory": jnp.asarray(0.9 * inventory),
    }

    def accept(
        budget: CoordinateEnclosureBudget,
    ) -> tuple[_PolyhedralTransferredPhysics, MeshingAcceptedEpoch]:
        physical = _polyhedral_physics(
            source, inventory, context.args.repeats, context.stages, admitted,
            initial_state=True, budget=budget,
        )
        if not np.isfinite(physical.error) or physical.error > min(context.args.target_error, 1e-7):
            raise qualification_failure(
                "polyhedral.material.source",
                "The original material generation must accept its actual physical state.",
                {"physical_error": physical.error},
            )
        epoch = _prepare_polyhedral_accepted_epoch(
            source, physical.fields, 0, owners=physical.owners, budget=budget
        )
        return physical, epoch

    context.set_phase("accepted_material_source_physics")
    consumed, measured = context.ledger.run(
        "accepted_material_source_physics",
        lambda: _measured(
            lambda: execute_native_consumer(
                source, generation.plans[0].specification.limits, accept,
                failure_factory=qualification_failure,
                prepared_operands=context.coordinate_operands.get(source.result_id),
                observe_managed_admission=True,
            )
        ),
    )
    context.stages["accepted_material_source_physics"] = measured["wall_seconds"]
    physical, admitted_epoch = consumed.value
    epoch = polyhedral_field_owner.rebind_meshing_accepted_epoch_execution(
        admitted_epoch, consumed.result, transition=None
    )
    generation.results["source"] = consumed.result
    context.record["source_physics"] = physical.record
    context.record["source_physical_error"] = physical.error
    _record_polyhedral_warm_publication(context, epoch, 0)
    return physical, epoch


def _regenerate_polyhedral_target(
    context: _PolyhedralLifecycleContext, generation: _PolyhedralGeneration
) -> phx.meshing.MeshAdaptationResult:
    case = context.case
    stages = context.stages
    record = context.record
    ledger = context.ledger
    results = generation.results
    labels = generation.labels
    acceptance = generation.acceptance
    geometry = generation.geometry
    request = generation.request
    options = generation.options
    phase = "canonical_regeneration"
    context.set_phase(phase)
    if not isinstance(geometry, phx.meshing.NativePolyhedralSource):
        raise TypeError(
            f"polyhedral.geometry must be NativePolyhedralSource with authored sites/weights; got {type(geometry).__name__}"
        )
    coordinate_operands = CoordinateOperandCollector()
    canonical, measured = ledger.run(
        phase,
        lambda: _measured(
            lambda: phx.meshing.execute_mesh_adaptation(
                phx.meshing.prepare_mesh_adaptation(
                    results["source"],
                    phx.meshing.PolyhedralMeshAdaptation(
                        phx.meshing.PolyhedralAdaptationOperation.REGENERATE,
                        complex_=geometry.complex,
                        sites=geometry.sites,
                        weights=geometry.weights,
                        schedule=options.polyhedral_schedule,
                    ),
                    policy=phx.meshing.MeshAdaptationPolicy(
                        phx.meshing.MeshAdaptationRoute.NATIVE_POLYHEDRAL,
                        limits=request.limits,
                        audit_policy=phx.meshing.CellMeshAuditPolicy(
                            require_complete_association=True,
                            watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
                        ),
                    ),
                    observe_managed_admission=True,
                ),
                coordinate_operands=coordinate_operands,
                observe_managed_admission=True,
            )
        ),
    )
    stages[phase] = measured["wall_seconds"]
    _remember_polyhedral_operands(context, canonical.target, coordinate_operands)
    original_organization = results["source"]
    if (
        {patch.name for patch in canonical.target.patches}
        != {patch.name for patch in original_organization.patches}
        or {label.name for label in canonical.target.labels}
        != {label.name for label in original_organization.labels}
        or {
            (zone.name, zone.material_id, zone.region_role)
            for zone in canonical.target.zones
        }
        != {
            (zone.name, zone.material_id, zone.region_role)
            for zone in original_organization.zones
        }
    ):
        raise qualification_failure(
            "qualify_native_polyhedral",
            ("Polyhedral regeneration lost authored source organization."),
            {
                "canonical": canonical,
                "original_organization": original_organization,
            },
        )
    record["independent_target_generation"] = _result_identity(results["target"])
    results["target"] = canonical.target
    labels["target"], acceptance["target"] = _polyhedral_acceptance(
        canonical.target, case
    )
    if canonical.common_refinement is None or not canonical.common_refinement.succeeded:
        raise qualification_failure(
            "qualify_native_polyhedral",
            ("Polyhedral regeneration lacks certified overlap."),
            {"canonical": canonical},
        )
    record["canonical_adaptation"] = {
        "claimed": True,
        "status": canonical.status.value,
        "target_result_id": canonical.target.result_id,
        "transition_id": None
        if canonical.transition is None
        else canonical.transition.transition_id,
        "source_organization_preserved": True,
    }
    return canonical


def _record_polyhedral_generation(
    context: _PolyhedralLifecycleContext, generation: _PolyhedralGeneration
) -> None:
    case = context.case
    record = context.record
    results = generation.results
    acceptance = generation.acceptance
    record["generation"] = {
        name: {
            **_result_identity(result),
            "family_counts": {
                block.cell_kind: block.cell_count for block in result.mesh.blocks
            },
        }
        for name, result in results.items()
    }
    original_request = generation.plans[0].specification
    if not isinstance(original_request, phx.meshing.VolumeMeshingSpec):
        raise TypeError("Polyhedral volume resource admission must retain its actual original volume specification.")
    complete_resources = {
        name: _complete_native_request_resources(
            original_request if name == "source" else generation.request, result
        )
        for name, result in results.items()
    }
    record["resources"] = complete_resources
    record["independent_acceptance"] = acceptance
    record["controls"] = {
        "family": "required polyhedron",
        "size_strength": "soft",
        "hard_size_claim": False,
        "materials": dict(case.expected_regions),
        "requested_resource_limits": "frozen; only runtime-listed bounds are enforced",
        "generation_resource_guarantees": {
            name: {
                "enforced": list(result.runtime.enforced_limits),
                "unenforced": list(result.runtime.unenforced_limits),
            }
            for name, result in results.items()
        },
        "target_request_id": record["request_id"],
    }


def _polyhedral_common_refinement_policy(
    capacity: int, limits: phx.meshing.MeshingLimits,
) -> phx.geometry.CommonRefinementPolicy:
    """Keep remap work/storage on the frozen native request's exact allowances."""
    return phx.geometry.CommonRefinementPolicy(
        maximum_candidate_pairs=32 * capacity,
        maximum_accepted_pairs=32 * capacity,
        maximum_memory_bytes=limits.maximum_scratch_bytes,
        maximum_exact_work=limits.maximum_work_units,
    )


def _remap_polyhedral_materials(
    context: _PolyhedralLifecycleContext,
    generation: _PolyhedralGeneration,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PolyhedralMaterialRemap:
    args = context.args
    stages = context.stages
    record = context.record
    ledger = context.ledger
    results = generation.results
    labels = generation.labels
    phase = "common_refinement_and_conservative_remap"
    context.set_phase(phase)
    remap_policy = _polyhedral_common_refinement_policy(
        args.capacity, generation.request.limits,
    )
    transferred, measured = ledger.run(
        phase,
        lambda: _measured(
            lambda: _polyhedral_remap(
                results["source"],
                results["target"],
                labels["source"],
                labels["target"],
                remap_policy,
                budget=budget,
            )
        ),
    )
    remap_evidence = transferred.evidence
    stages[phase] = measured["wall_seconds"]
    record["remap"] = remap_evidence
    return transferred


def _material_fv_transfer_seeds(
    owners: polyhedral_consumer_owner.PolyhedralArchiveOwners,
    fields: dict[str, jax.Array],
    repeats: int,
    stages: dict[str, float],
) -> dict[str, Any]:
    _, diffusion, tensor, boundary = prepare_fv_system(owners.hybrid)
    trace_evidence = {}
    for name, cells in (
        ("fv_solution", "fv_cell_solution"),
        ("fv_affine_solution", "fv_affine_cell_solution"),
    ):
        full, histories, evidence = _polyhedral_qualification_reprepare_fv_seed(
            diffusion, tensor, boundary,
            {
                "fv_cell_solution": fields[cells],
                "fv_cell_solution_history": fields[cells + "_history"],
            },
            repeats, stages,
        )
        fields[name], fields[name + "_history"] = full, histories
        trace_evidence[name] = evidence
    states = jnp.concatenate((fields["fv_solution"][None, :], fields["fv_solution_history"]), axis=0)
    inventories = jnp.concatenate((fields["inventory"][None, :], fields["inventory_history"]), axis=0)
    derived = jax.vmap(
        _material_fv_physical_fields, in_axes=(None, None, None, 0, 0)
    )(diffusion, tensor, boundary, inventories, states)
    for name, values in derived.items():
        fields[name], fields[name + "_history"] = values[0], values[1:]
    return trace_evidence


def _material_fem_transfer_seeds(
    plan: polyhedral_consumer_owner.PolyhedralSourceStarFEMPlan,
    fields: dict[str, jax.Array],
) -> None:
    derived: dict[str, list[jax.Array]] = {}
    states = (fields["fem_solution"], *tuple(fields["fem_solution_history"]))
    materials = (fields["inventory"], *tuple(fields["inventory_history"]))
    for coefficients, material in zip(states, materials, strict=True):
        physical = polyhedral_consumer_owner.evaluate_polyhedral_source_star_fields(
            plan, coefficients, jnp.zeros_like(material), material,
            compatibility_bound=0.0,
        )
        for name, value in (
            ("fem_cell_solution", physical.cell_values),
            ("fem_gradient", physical.cell_gradients),
            ("fem_vector_flux", physical.cell_vector_flux),
            ("fem_physical_residual", physical.physical_residual),
            ("fem_physical_load", physical.nodal_load),
        ):
            derived.setdefault(name, []).append(value)
    for name, values in derived.items():
        fields[name], fields[name + "_history"] = values[0], jnp.stack(values[1:])


def _transport_material_accepted_state(
    adaptation: phx.meshing.MeshAdaptationResult,
    source: _PolyhedralTransferredPhysics,
    repeats: int,
    stages: dict[str, float],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[dict[str, jax.Array], dict[str, Any], polyhedral_consumer_owner.PolyhedralArchiveOwners]:
    fields = dict(source.fields)
    count = adaptation.source.mesh.connectivity.cell_count
    for name, cells in (
        ("fv_solution", "fv_cell_solution"),
        ("fv_affine_solution", "fv_affine_cell_solution"),
    ):
        fields[cells] = fields[name][:count]
        fields[cells + "_history"] = fields[name + "_history"][:, :count]
    _, target_owners = polyhedral_field_owner.prepare_meshing_polyhedral_owner_binding(
        adaptation.target, budget=budget
    )
    fem_transfer = polyhedral_consumer_owner.prepare_polyhedral_source_star_field_transfer(
        source.owners.source_star, target_owners.source_star, budget=budget
    )
    vem_transfer = polyhedral_consumer_owner.prepare_polyhedral_vem_reconstruction_transfer(
        source.owners.vem, target_owners.vem,
        source.owners.source_star, target_owners.source_star, fem_transfer,
    )
    transferred, proof = _polyhedral_qualification_periodic_transport(
        adaptation, fields, 0,
        cell_roles={
            "inventory": "physical-state",
            "previous_inventory": "history",
            "fv_cell_solution": "physical-state",
            "fv_affine_cell_solution": "physical-state",
        },
        owners=(source.owners, target_owners),
        vem_transfer=vem_transfer,
        fem_transfer=fem_transfer,
        budget=budget,
    )
    proof["trace_repreparation"] = _material_fv_transfer_seeds(
        target_owners, transferred, repeats, stages
    )
    _material_fem_transfer_seeds(target_owners.source_star, transferred)
    proof["derived_history"] = "actual target-owner evaluation of transported current and three historical primitive states"
    proof["independent_primal_reconstruction_errors"] = {
        "vem_constant": float(jnp.max(jnp.abs(transferred["vem_solution"] - 1.0))),
        "fem_affine": float(jnp.max(jnp.abs(
            transferred["fem_solution"] - _polyhedral_affine_fem_solution(
                target_owners.source_star.space.dof_maps[0].dof_coordinates
            )
        ))),
    }
    return transferred, proof, target_owners


def _accept_material_target_physics(
    context: _PolyhedralLifecycleContext,
    generation: _PolyhedralGeneration,
    source_physical: _PolyhedralTransferredPhysics,
    adaptation: phx.meshing.MeshAdaptationResult,
) -> tuple[_PolyhedralMaterialRemap, _PolyhedralTransferredPhysics, MeshingAcceptedEpoch, phx.meshing.MeshAdaptationResult]:
    target = generation.results["target"]

    def accept(
        budget: CoordinateEnclosureBudget,
    ) -> tuple[_PolyhedralMaterialRemap, _PolyhedralTransferredPhysics, MeshingAcceptedEpoch]:
        remap = _remap_polyhedral_materials(context, generation, budget=budget)
        fields, proof, owners = _transport_material_accepted_state(
            adaptation, source_physical, context.args.repeats, context.stages,
            budget=budget,
        )
        if not np.array_equal(np.asarray(fields["inventory"]), remap.remapped):
            raise ValueError("Material all-field transport must match the independent conservative inventory remap.")
        context.record["material_allfield_transfer"] = proof
        if max(proof["independent_primal_reconstruction_errors"].values()) > min(context.args.target_error, 1e-7):
            raise ValueError("Material H1 reconstructed accepted state must satisfy its unchanged physical accuracy request.")
        physical = _polyhedral_physics(
            target, remap.remapped, context.args.repeats, context.stages, fields,
            transfer_evidence=proof, owners=owners, budget=budget,
        )
        epoch = _prepare_polyhedral_accepted_epoch(
            target, physical.fields, 1, transition=adaptation,
            owners=owners, budget=budget,
        )
        return remap, physical, epoch

    context.set_phase("accepted_material_target_physics")
    consumed, measured = context.ledger.run(
        "accepted_material_target_physics",
        lambda: _measured(
            lambda: execute_native_consumer(
                target, generation.request.limits, accept,
                failure_factory=qualification_failure,
                prepared_operands=context.coordinate_operands.get(target.result_id),
                observe_managed_admission=True,
            )
        ),
    )
    context.stages["accepted_material_target_physics"] = measured["wall_seconds"]
    remap, physical, admitted_epoch = consumed.value
    sealed_transition = polyhedral_field_owner.rebind_meshing_adaptation_execution_endpoints(
        adaptation, generation.results["source"], consumed.result
    )
    epoch = polyhedral_field_owner.rebind_meshing_accepted_epoch_execution(
        admitted_epoch, consumed.result, transition=sealed_transition
    )
    generation.results["target"] = consumed.result
    _record_polyhedral_warm_publication(context, epoch, 1)
    return remap, physical, epoch, sealed_transition


def _complete_polyhedral_physics(
    context: _PolyhedralLifecycleContext,
    generation: _PolyhedralGeneration,
    source_epoch: MeshingAcceptedEpoch,
    source_physical: _PolyhedralTransferredPhysics,
    adaptation: phx.meshing.MeshAdaptationResult,
) -> None:
    args = context.args
    case = context.case
    started = context.started
    stages = context.stages
    record = context.record
    results = generation.results
    labels = generation.labels
    plans = generation.plans
    remap, physical, target_epoch, adaptation = _accept_material_target_physics(
        context, generation, source_physical, adaptation
    )
    common = remap.common
    remapped = remap.remapped
    _record_polyhedral_generation(context, generation)
    physics = physical.record
    error = max(source_physical.error, physical.error)
    record.update(physics=physics, physical_error=error)
    retained = (*source_physical.retained, *physical.retained)
    remap_evidence = remap.evidence
    inventory_after = remap.inventory_after
    consumed_inventory = physical.transferred_source_integral
    inventory_defect = abs(consumed_inventory - inventory_after)
    domain_measure = physical.domain_measure
    domain_defect = abs(domain_measure - sum(dict(case.expected_regions).values()))
    if inventory_defect > 1e-10 * max(1.0, abs(inventory_after)) or domain_defect > 1e-10:
        raise qualification_failure(
            "qualify_native_polyhedral",
            ("FV physics did not consume the certified remapped material inventory."),
            {
                "domain_defect": domain_defect,
                "inventory_after": inventory_after,
                "inventory_defect": inventory_defect,
            },
        )
    remap_evidence["consumed_inventory_defect"] = inventory_defect
    context.set_phase("accepted_material_allfield_history_fresh_cold_continued_pde")
    original = generation.plans[0]
    if not isinstance(original.source, (phx.meshing.NativePlcSource, phx.meshing.NativePolyhedralSource)) or not isinstance(original.specification, phx.meshing.VolumeMeshingSpec):
        raise TypeError("Material accepted history must retain its actual original polyhedral generation controls.")
    epochs = (source_epoch, target_epoch)
    root = _polyhedral_source_records(
        original.source, original.specification, original.options, epochs
    )
    record["accepted_epochs"] = [epoch.epoch_id for epoch in epochs]
    _join_polyhedral_cold_and_source_prefix(
        context, original.source, original.specification, original.options,
        generation.source_publication,
        lambda: _polyhedral_qualification_cold(
            root, _polyhedral_archive_evidence(root), "material", len(epochs),
            args.repeats, args.target_error, stages, context.deadline,
            expected_build_id=record["runtime"]["build_id"],
        ),
        "accepted_material_allfield_history_fresh_cold_continued_pde",
        original,
    )
    elapsed = perf_counter() - started
    record.update(
        {
            "target_error_reached": error <= args.target_error,
            "time_to_target_error_seconds": elapsed
            if error <= args.target_error
            else None,
            "logical_retained_bytes": model_array_bytes(
                (retained, results, plans, common, remapped, labels, epochs)
            ),
            "qualification_scale": {
                "site_count": args.resolution,
                "cell_counts": {
                    name: result.mesh.connectivity.cell_count
                    for name, result in results.items()
                },
                "face_counts": {
                    name: result.mesh.entity_set(2).entity_ids.size
                    for name, result in results.items()
                },
                "overlap_pieces": remap_evidence["overlap_pieces"],
            },
            "demanding_controls": list(case.controls),
        }
    )
    obligations = _polyhedral_completion_obligations(record, args, elapsed)
    complete = all(obligations.values())
    scientific = all(
        passed
        for name, passed in obligations.items()
        if name not in ("original_resource_bounds", "source_prefix_resource_boundary", "wall_time")
    )
    record.update(
        {
            "status": "timeout"
            if elapsed > args.timeout
            else "passed"
            if complete
            else "failed",
            "completion_obligations": obligations,
            "case_contract_completion": "complete" if complete else "incomplete",
            "mandatory_workflow_completion": "complete" if complete else "incomplete",
            "independent_scientific_certification": "passed" if scientific else "failed",
        }
    )


def _polyhedral_completion_obligations(
    record: dict[str, Any], args: QualificationControls, elapsed: float
) -> dict[str, bool]:
    """Derive every material polyhedral obligation from emitted evidence alone."""
    acceptance = record.get("independent_acceptance", {})
    generation = record.get("generation", {})
    canonical = record.get("canonical_adaptation", {})
    remap = record.get("remap", {})
    physics = record.get("physics", {})
    vem, fv, fe = physics.get("vem", {}), physics.get("fv", {}), physics.get("fe", {})
    resources = record.get("resources", {})
    error = record.get("physical_error")
    inventories = remap.get("material_inventories", {})
    consumed = remap.get("consumed_inventory_defect")
    source_error = record.get("source_physical_error")
    source_physics = record.get("source_physics", {})
    transport = record.get("material_allfield_transfer", {})
    prefix = record.get("source_prefix_resources", {})
    return {
        "certified_restricted_power_generation": set(acceptance) == {"source", "target"}
        and all(
            item.get("passed") is True and item.get("coverage_status") == "certified"
            for item in acceptance.values()
        )
        and all(
            set(item.get("family_counts", {})) == {"polyhedron"}
            for item in generation.values()
        ),
        "canonical_site_regeneration": canonical.get("claimed") is True
        and canonical.get("status") == phx.meshing.MeshAdaptationStatus.COMPLETE.value
        and canonical.get("source_organization_preserved") is True,
        "accepted_source_fe_vem_fv_physics": source_error is not None
        and bool(np.isfinite(source_error))
        and 0.0 <= source_error <= min(args.target_error, 1e-7)
        and source_physics.get("fv", {}).get("vector_flux_finite") is True
        and source_physics.get("fe", {}).get("vector_flux_finite") is True,
        "genuine_allfield_regeneration_history": len(record.get("accepted_epochs", ())) == 2
        and transport.get("published") is True
        and transport.get("adaptation_status") == phx.meshing.MeshAdaptationStatus.COMPLETE.value
        and transport.get("fem_transfer", {}).get("history_states_transferred") == 3
        and transport.get("nodal_transfer", {}).get("semantics") == "reconstruction"
        and {"inventory", "previous_inventory", "fv_cell_solution", "fv_affine_cell_solution"}
        <= set(transport.get("inventory_defects", {})),
        "conservative_material_remap_consumed": bool(inventories)
        and all(
            item["absolute_defect"] <= 1e-10 * max(1.0, abs(item["before"]))
            for item in inventories.values()
        )
        and consumed is not None
        and consumed
        <= 1e-10 * max(1.0, sum(abs(item["after"]) for item in inventories.values())),
        "consumer_conditioning": vem.get("minimum_rank_margin", 0.0) > 0.0
        and vem.get("maximum_reproduction_defect", np.inf) <= 1e-7
        and fv.get("minimum_normal_rank_margin", 0.0) > 0.0
        and bool(np.isfinite(fv.get("maximum_nonorthogonality_degrees", np.inf))),
        "accepted_vem_fv_physics": error is not None
        and bool(np.isfinite(error))
        and 0.0 <= error <= min(args.target_error, 1e-7),
        "conforming_material_fe_physics": fe.get("physical_l2_error", np.inf) <= min(args.target_error, 1e-7)
        and fe.get("maximum_free_physical_residual", np.inf) <= 1e-7
        and fe.get("global_weak_conservation_defect", np.inf) <= 1e-7
        and fe.get("vector_flux_finite") is True,
        "vector_flux_and_conservation": fv.get("vector_flux_finite") is True
        and fv.get("maximum_interior_flux_continuity_defect", np.inf) <= 1e-7
        and fv.get("maximum_cell_flux_balance_defect", np.inf) <= 1e-7,
        "fresh_process_accepted_state_cold_continuation": record.get("cold", {}).get("status") == "passed"
        and record.get("cold", {}).get("accepted_fields_bitwise_preserved") is True,
        "original_resource_bounds": set(resources) == {"source", "target"}
        and all(item["within_original_bounds"] is True for item in resources.values()),
        "source_prefix_resource_boundary": prefix.get("status") == "passed"
        and prefix.get("mandatory_capacity_gate") == "complete"
        and prefix.get("admitted_scope") == "SOURCEPREFIX"
        and prefix.get("complete_request_R_claimed") is False
        and prefix.get("original_source_and_controls_preserved") is True,
        "wall_time": elapsed <= args.timeout,
    }


def qualify_native_polyhedral(args: QualificationControls) -> dict[str, object]:
    """Nonconvex material PLC → power cells → conservative state → VEM/FV."""
    if "periodic-quotient" in polyhedral_case(args.polyhedral_source).controls:
        return _polyhedral_qualification_periodic_lifecycle(args)
    started = perf_counter()
    deadline = Deadline(started + args.timeout)
    case = polyhedral_case(args.polyhedral_source)
    stages: dict[str, float] = {"_campaign_started": started}
    phases: list[NativeMeshingPhaseMeasurement] = []
    record: dict[str, Any] = {
        "scenario": "native-polyhedral-lifecycle",
        "status": "failed",
        "engine": "native",
        "case": polyhedral_identity(case.name),
        "runtime": _runtime_identity(),
        "control_capacities": {
            "site_count": args.resolution,
            "entity_capacity": args.capacity,
        },
        "targets": {
            "maximum_physics_error": args.target_error,
            "wall_seconds": args.timeout,
        },
        "stages_seconds": stages,
        "time_to_target_error_seconds": None,
        "mandatory_workflow_completion": "incomplete",
        "release": "unassessed",
        "leadership": "unassessed",
        "canonical_adaptation": {"claimed": False, "status": "unassessed"},
        "nonclaims": [
            "Soft size is not a hard size guarantee.",
            "Fixed-topology site derivatives are not claimed.",
            "Routine source/site sweeps do not complete larger demanding-control corpus.",
        ],
    }
    _ledger = StageLedger(record, deadline=deadline)
    phase = "source_preparation"

    def set_phase(name: str) -> None:
        nonlocal phase
        phase = name

    context = _PolyhedralLifecycleContext(
        args=args,
        case=case,
        started=started,
        deadline=deadline,
        stages=stages,
        phases=phases,
        record=record,
        ledger=_ledger,
        set_phase=set_phase,
    )
    try:
        if not meshcore_available():
            record.update(
                _missing_dependency(
                    "phydrax-meshcore",
                    "The required native construction library is unavailable.",
                )
            )
            return record
        if (
            args.repeats < 1
            or not np.isfinite(args.target_error)
            or args.target_error <= 0.0
        ):
            raise ValueError(
                f"qualification.controls must have positive repeats and a finite positive physical error target; got repeats {args.repeats!r}, target_error {args.target_error!r}"
            )
        generation = _publish_polyhedral_generation(context)
        source_physical, source_epoch = _accept_material_source_physics(context, generation)
        adaptation = _regenerate_polyhedral_target(context, generation)
        _complete_polyhedral_physics(
            context, generation, source_epoch, source_physical, adaptation
        )
    except phx.meshing.MeshingFailure as failure:
        record_failure(record, failure, phase)
        record.update(
            {
                "failure_message": failure.evidence.message,
            }
        )
    except (RuntimeError, ValueError, TypeError, TimeoutError) as failure:
        record_failure(record, failure, phase)
    finally:
        record["native_phases"] = _phase_recordings(phases)
        record["end_to_end_seconds"] = perf_counter() - started
        record.update(_polyhedral_memory_measurements())
    return record


def cold_worker(packet: ColdControlPacket) -> ColdWorkerReport:
    """Run the scenario-owned cold consumer on its one control packet."""
    controls = packet.controls
    match controls["kind"]:
        case "periodic":
            program = _polyhedral_qualification_periodic_cold_process
        case "material":
            program = _polyhedral_material_cold_process
        case invalid:
            raise ValueError(f"polyhedral.cold.kind must identify periodic or material; got {invalid!r}")
    began = perf_counter()
    execution = read_meshing_source_execution_controls(
        packet.archive_path,
        expected_content_id=packet.content_id,
        limits=ArrayArchiveLimits(),
    )
    control_read_seconds = perf_counter() - began

    def continue_accepted(
        budget: CoordinateEnclosureBudget,
    ) -> tuple[phx.meshing.CellMeshingResult, ColdWorkerReport]:
        return program(
            packet.archive_path,
            packet.content_id,
            controls["expected"],
            int(controls["repeats"]),
            float(controls["target_error"]),
            budget=budget,
            deadline=packet.deadline,
        )

    continued = execute_native_request(
        execution.execution_evidence,
        execution.limits,
        continue_accepted,
        target=lambda value: value[0],
        failure_factory=qualification_failure,
        workflow_remaining_seconds=packet.deadline.remaining(),
        observe_managed_admission=True,
    )
    _, report = continued.value
    receipt = continued.result.execution_evidence
    if receipt is None:
        raise qualification_failure(
            "polyhedral.cold.execution",
            "A complete cold request must retain its actual ended execution record.",
            {"result": continued.result},
        )
    return ColdWorkerReport(
        record={
            **report.record,
            "cold_request_result": _result_identity(continued.result),
            "cold_request": {
                "history_owner_id": execution.execution_evidence.owner_id,
                "owner_id": receipt.owner_id,
                "work_units": int(np.asarray(receipt.total_work_units)),
                "geometry_queries": int(np.asarray(receipt.total_geometry_queries)),
                "receipt_fingerprint": array_tree_fingerprint(receipt),
                "original_limits": meshing_limit_values(execution.limits),
            },
        },
        read_seconds=control_read_seconds + report.read_seconds,
    )


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host qualification controls."""
    return qualify_native_polyhedral(args)
