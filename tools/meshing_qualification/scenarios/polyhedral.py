# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, perf_counter
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from examples.native_polyhedral_meshing import (
    prepare_fv,
    prepare_vem,
    rebind_inventory_history,
)
from phydrax import array_tree_fingerprint, ArrayArchiveLimits, model_array_bytes
from phydrax.discretization import (
    FieldEpochTransition,
    TopologyEpoch,
    TopologyEpochTransition,
    TransferGeometryBinding,
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
from phydrax.lifecycle import (
    commit_composition_rebind,
    Composition,
    CompositionEntry,
    CompositionRebind,
    CompositionRole,
    read_meshing_source_closure,
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
    prepare_periodic_source_quadrature,
)
from tools._meshing_cases import (
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
    _result_identity,
    FailureRecord,
    record_failure,
)
from tools.meshing_qualification._runtime import (
    _measured,
    _missing_dependency,
    _PolyhedralJitEntry,
    _runtime_identity,
    ColdWorkerTimeout,
    Deadline,
    peak_resident_bytes,
    qualification_failure,
    run_cold_worker,
    StageLedger,
)


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
    if coverage is None or coverage.status != "certified":
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
        rhs - system.operator.mv(seed),
        policy=phx.linalg.LinearSolvePolicy(phx.linalg.ConjugateGradient()),
    )
    return seed + solved.value, solved.successful, solved.diagnostics


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
            "compilation_count": 1,
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


def _polyhedral_remap(
    source: Any,
    target: Any,
    source_regions: np.ndarray,
    target_regions: np.ndarray,
    policy: Any,
) -> _PolyhedralMaterialRemap:

    common = phx.geometry.prepare_common_refinement(
        source.mesh,
        target.mesh,
        policy=policy,
        source_geometry=source.geometry,
        target_geometry=target.geometry,
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
            "overlap_pieces": volumes.size,
            "material_inventories": inventories,
            "maximum_relative_source_defect": common.evidence.maximum_relative_source_defect,
            "maximum_relative_target_defect": common.evidence.maximum_relative_target_defect,
            "consumed_by": "target mimetic diffusion integrated cell source",
        },
    )


def _polyhedral_physics(
    result: Any,
    remapped: np.ndarray,
    repeats: int,
    stages: dict[str, float],
    source: Any,
    common: Any,
    source_regions: np.ndarray,
) -> _PolyhedralTransferredPhysics:

    mesh, geometry = result.mesh, result.geometry

    prepared, measured = _measured(lambda: prepare_vem(mesh, geometry))
    vem, operator, mass = prepared
    stages["vem_preparation"] = measured["wall_seconds"]
    values, vem_compiler, timings = _compiled_polyhedral_stages(
        phx.linalg.LinearSystem(operator), mass, jnp.zeros_like(mass), repeats
    )
    stages.update({f"vem_{name}": value for name, value in timings.items()})
    vem_error = float(jnp.max(jnp.abs(values - 1.0)))
    prepared, measured = _measured(lambda: prepare_fv(mesh, geometry))
    fv, diffusion, tensor, boundary = prepared
    stages["fv_preparation"] = measured["wall_seconds"]
    rebound, measured = _measured(
        lambda: rebind_inventory_history(
            source, result, common, fv, np.where(source_regions == 0, 2.0, 5.0)
        )
    )
    receipt, remapped = rebound
    stages["canonical_physical_history_rebind"] = measured["wall_seconds"]
    system_rhs, measured = _measured(lambda: diffusion.linear_system(tensor, boundary))
    system, rhs = system_rhs
    stages["fv_system_preparation"] = measured["wall_seconds"]
    initial, fv_compiler, timings = _compiled_polyhedral_stages(
        system, rhs, jnp.zeros_like(rhs), repeats
    )
    stages.update({f"fv_{name}": value for name, value in timings.items()})
    exact = 1.0 + fv.cell_centers @ jnp.asarray((1.0, 2.0, 3.0), dtype=jnp.float64)
    fv_error = float(jnp.max(jnp.abs(initial[: diffusion.cell_count] - exact)))
    integrated_source = jnp.asarray(remapped) * fv.cell_volumes
    system_rhs, measured = _measured(
        lambda: diffusion.linear_system(tensor, boundary, source=integrated_source)
    )
    update_system, update_rhs = system_rhs
    stages["fv_transferred_system_preparation"] = measured["wall_seconds"]
    updated, update_compiler, timings = _compiled_polyhedral_stages(
        update_system, update_rhs, initial, repeats
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
        np.asarray(mesh.connectivity.cell_face_offsets),
        np.asarray(mesh.connectivity.cell_face_values),
    )
    rank_margin = min(
        float(
            np.linalg.svd(
                normals[faces[a:b]] / float(fv.cell_volumes[cell]), compute_uv=False
            )[-1]
        )
        for cell, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True))
    )
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
        receipt,
    )
    physics_record: dict[str, Any] = {
        "canonical_state_rebind": {
            "published": receipt.published,
            "receipt_id": receipt.receipt_id,
            "roles": ("physical-state", "history"),
            "target_result_id": result.result_id,
        },
        "vem": {
            "compiler": vem_compiler,
            "constant_solution_error": vem_error,
            "minimum_rank_margin": vem.evidence.minimum_rank_margin,
            "maximum_reproduction_defect": vem.evidence.maximum_reproduction_defect,
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
        },
    }
    return _PolyhedralTransferredPhysics(
        record=physics_record,
        error=max(vem_error, fv_error, residual_error),
        retained=retained,
        domain_measure=physics_record["fv"]["domain_measure"],
        transferred_source_integral=physics_record["fv"]["transferred_source_integral"],
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


def _polyhedral_qualification_operator_evidence(
    operator: Any, rhs: Any, solution: Any
) -> dict[str, float]:
    residual = operator.mv(solution) - rhs
    size = solution.size
    left = jnp.sin(jnp.arange(size, dtype=solution.dtype) + 0.3)
    right = jnp.cos(jnp.arange(size, dtype=solution.dtype) + 0.7)
    transpose_defect = abs(
        float(
            jnp.vdot(left, operator.mv(right))
            - jnp.vdot(operator.transpose_mv(left), right)
        )
    )
    return {
        "maximum_algebraic_residual": float(jnp.max(jnp.abs(residual))),
        "raw_transpose_pairing_defect": transpose_defect,
    }


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
    fv, measured = ledger.run(
        "prepare",
        lambda: _measured(
            lambda: UnstructuredFiniteVolumePlan.from_cell_mesh(
                result.mesh, field_name="state", component_names=("value",)
            ).prepare(cell_geometry=result.geometry)
        ),
    )
    stages["periodic_fv_preparation"] = measured["wall_seconds"]
    diffusion, boundary = HybridMimeticDiffusion(fv), HybridDiffusionBoundary(fv)
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
                result, failure_factory=qualification_failure
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
        lambda: _measured(
            lambda: prepare_polyhedral_h1_virtual_element_3d(
                result.mesh, cell_geometry=result.geometry
            )
        ),
    )
    stages["periodic_vem_preparation"] = measured["wall_seconds"]
    load = vem.assemble_cellwise_constant_load(force)
    # Zero tolerance preserves strict load compatibility without force correction.
    compatibility_tolerance = 0.0
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
    for name in (
        "physical_force",
        "physical_source_load",
        "fv_cell_solution",
        "fv_trace_solution",
        "fv_owner_flux",
        "fv_local_flux",
        "fv_physical_residual",
        "fv_flux_balance",
        "vem_physical_load",
        "vem_physical_residual",
    ):
        updated[name + "_history"] = _polyhedral_qualification_field_history(
            seed_fields, name, updated[name]
        )
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
            "total_physical_load": float(jnp.sum(load)),
            "minimum_rank_margin": vem.evidence.minimum_rank_margin,
            "maximum_reproduction_defect": vem.evidence.maximum_reproduction_defect,
            "restored_seed_consumed": "vem_solution" in seeds or "vem_solution" in fields,
            "pin_dof": 0,
            "pin_value": 0.0,
        },
        "retained_field_names": sorted(updated),
        "prepared_solver_archived": False,
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
    context = _PeriodicPhysicsContext(
        result=result,
        fields=fields,
        seeds=seeds,
        repeats=repeats,
        stages=stages,
        measurements=measurements,
        ledger=_ledger,
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
    return _record_periodic_physical_evidence(
        context, source, finite_volume, virtual_element, response, conserved, history
    )


def _polyhedral_qualification_periodic_archive_evidence(
    root: Any,
) -> dict[str, Any]:
    """Return byte-level owner and allfield evidence independent of archive IDs."""

    source, request, options, epochs = root
    records = []
    for result, pairs in epochs:
        certification = result.certification
        if (
            certification is None
            or not certification.passed
            or not result.audit.passed
            or not result.compliance.passed
        ):
            raise qualification_failure(
                "_polyhedral_qualification_periodic_archive_evidence",
                ("Cold periodic epoch lacks accepted certification."),
                {"certification": certification, "result": result},
            )
        fields = {}

        prepared = UnstructuredFiniteVolumePlan.from_cell_mesh(
            result.mesh, field_name="state", component_names=("value",)
        ).prepare(cell_geometry=result.geometry)
        volumes = np.asarray(prepared.cell_volumes)
        accepted_fields = dict(pairs)
        inventories = {
            "conserved": np.einsum(
                "nk,n->k", np.asarray(accepted_fields["conserved"]), volumes
            ).tolist(),
            "conserved_history": np.einsum(
                "tnk,n->tk", np.asarray(accepted_fields["conserved_history"]), volumes
            ).tolist(),
            "pressure": float(np.dot(np.asarray(accepted_fields["pressure"]), volumes)),
            "temperature": float(
                np.dot(np.asarray(accepted_fields["temperature"]), volumes)
            ),
        }
        for name, value in pairs:
            array = np.asarray(value)
            fields[name] = {
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "bytes": array.nbytes,
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        records.append(
            {
                "result": _result_identity(result),
                "result_fingerprint": array_tree_fingerprint(result),
                "geometry_fingerprint": array_tree_fingerprint(result.geometry),
                "domain_fingerprint": array_tree_fingerprint(
                    certification.request.domain
                ),
                "certification_id": certification.report_id,
                "certification_fingerprint": array_tree_fingerprint(certification),
                "inventories": inventories,
                "fields": fields,
            }
        )
    return {
        "source_fingerprint": array_tree_fingerprint(source),
        "request_fingerprint": array_tree_fingerprint(request),
        "options_fingerprint": array_tree_fingerprint(options),
        "request_id": request.specification_id,
        "epochs": records,
    }


def _polyhedral_qualification_periodic_cold_process(
    path: str,
    content_id: str,
    expected_path: str,
    repeats: int,
    target_error: float,
) -> dict[str, Any]:

    root = read_meshing_source_closure(
        path, expected_content_id=content_id, limits=ArrayArchiveLimits()
    )
    with open(expected_path, encoding="utf-8") as stream:
        expected = json.load(stream)
    evidence = _polyhedral_qualification_periodic_archive_evidence(root)
    if evidence != expected:
        raise qualification_failure(
            "_polyhedral_qualification_periodic_cold_process",
            (
                "Fresh periodic restore changed source bytes, SCI, controls, certification or allfields."
            ),
            {"evidence": evidence, "expected": expected},
        )
    continued = []
    for result, pairs in root[3]:
        fields = dict(pairs)
        timings = {}
        physics, error, updated = _polyhedral_qualification_periodic_physics(
            result, fields, repeats, timings, seeds=fields
        )
        if not np.isfinite(error) or error > target_error:
            raise qualification_failure(
                "_polyhedral_qualification_periodic_cold_process",
                (
                    "Restored periodic PDE continuation missed the requested physical target."
                ),
                {"error": error, "target_error": target_error},
            )
        continued.append(
            {
                "result_id": result.result_id,
                "physics": physics,
                "physical_error": error,
                "stages_seconds": timings,
                "continued_fields": _polyhedral_qualification_periodic_archive_evidence(
                    (
                        root[0],
                        root[1],
                        root[2],
                        ((result, tuple(sorted(updated.items()))),),
                    )
                )["epochs"][0]["fields"],
            }
        )
    peak = peak_resident_bytes(resource.RUSAGE_SELF)
    return {
        "status": "passed",
        "restored": evidence,
        "continued_epochs": continued,
        "process_id": os.getpid(),
        "interpreter": sys.executable,
        "runtime": _runtime_identity(),
        "peak_process_resident_bytes": peak,
    }


def _polyhedral_qualification_periodic_cold(
    source: Any,
    request: Any,
    options: Any,
    epochs: Any,
    repeats: int,
    target_error: float,
    stages: dict[str, Any],
    deadline: Deadline,
) -> dict[str, Any]:

    if "_campaign_started" not in stages:
        raise ValueError(
            "stages._campaign_started must contain the perf_counter campaign start; got missing key"
        )
    root = (
        source,
        request,
        options,
        tuple((result, tuple(sorted(fields.items()))) for result, fields in epochs),
    )
    expected = _polyhedral_qualification_periodic_archive_evidence(root)
    limits = ArrayArchiveLimits()
    record = {
        "status": "failed",
        "archive_limits": {
            "members": limits.max_members,
            "nesting": limits.max_manifest_nesting,
            "rank": limits.max_array_rank,
        },
        "accepted_epochs": len(epochs),
        "source_authority_stripped": False,
    }
    _ledger = StageLedger(record, deadline=deadline)
    with TemporaryDirectory(prefix="phydrax-poly-periodic-cold-") as directory:
        receipt, timing = _ledger.run(
            "write_meshing_source_closure",
            lambda: _measured(
                lambda: write_meshing_source_closure(
                    Path(directory) / "periodic.zip", root, limits=limits
                )
            ),
        )
        stages["periodic_checkpoint_publication"] = timing["wall_seconds"]
        expected_path = Path(directory) / "expected.json"
        expected_path.write_text(json.dumps(expected), encoding="utf-8")
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
        # The scenario module owns its worker command.
        cold_started = perf_counter()
        try:
            restored = run_cold_worker(
                "scenarios.polyhedral",
                (
                    str(receipt.path),
                    receipt.content_id,
                    str(expected_path),
                    str(repeats),
                    str(target_error),
                ),
                deadline=deadline,
                ledger=_ledger,
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
        stages["periodic_cold_restore_continued_pde"] = timing["wall_seconds"]
        record.update(
            {
                "returncode": 0,
                "stdout": json.dumps(restored),
                "stderr": "",
            }
        )
        # Parsed and process-validated by run_cold_worker.
        if restored["restored"] != expected or restored["process_id"] == os.getpid():
            raise qualification_failure(
                "_polyhedral_qualification_periodic_cold",
                (
                    "Periodic cold evidence does not bind the archived source and a fresh process."
                ),
                {"expected": expected, "restored": restored},
            )
        record.update(
            {
                "status": restored["status"],
                "fresh_interpreter": restored,
                "accepted_fields_bitwise_preserved": True,
            }
        )
        return record


@final
class _PeriodicResourceObservation(phx.StrictModule):
    receipt: NativeExecutionRecord
    phases: tuple[NativeExecutionRecord, ...]
    observed: dict[str, int | float]
    declared: dict[str, int | float]
    enforced: set[str]
    bounds: dict[str, dict[str, Any]]


def _observe_periodic_resources(
    request: phx.meshing.VolumeMeshingSpec, result: phx.meshing.CellMeshingResult
) -> _PeriodicResourceObservation:
    receipt = result.execution_evidence
    if receipt is None:
        raise qualification_failure(
            "_polyhedral_qualification_periodic_resource_refusals",
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
                "_polyhedral_qualification_periodic_resource_refusals",
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
    observed = {
        "maximum_work_units": int(np.asarray(receipt.total_work_units)),
        "maximum_geometry_queries": int(np.asarray(receipt.total_geometry_queries)),
        "maximum_cavity_cells": max(
            int(np.asarray(phase.work_counters.peak_cavity_cells)) for phase in phases
        ),
        "maximum_scratch_bytes": max(
            int(np.asarray(phase.memory_counters.peak_bytes)) for phase in phases
        ),
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
                "_polyhedral_qualification_periodic_resource_refusals",
                (f"Periodic execution record exceeds {name}."),
                {"limit": limit, "value": value},
            )
    return _PeriodicResourceObservation(
        receipt=receipt,
        phases=tuple(phases),
        observed=observed,
        declared=declared,
        enforced=enforced,
        bounds=bounds,
    )


def _exercise_periodic_resource_boundary(
    source: phx.meshing.NativePolyhedralSource,
    request: phx.meshing.VolumeMeshingSpec,
    options: phx.meshing.NativeMeshingOptions,
    result: phx.meshing.CellMeshingResult,
    name: str,
    value: int | float,
    declared: dict[str, int | float],
    operation_started: float,
    original: dict[str, Any],
) -> dict[str, Any]:
    is_clock = name == "maximum_wall_seconds"
    lowered = value * 0.5 if is_clock else value - 1
    reduced_limits = replace(
        MeshingLimitControls.from_limits(request.limits), **{name: lowered}
    ).to_limits()
    reduced_request = with_meshing_limits(request, reduced_limits)
    measurements = []
    try:
        execute_native_capacity_trial(
            source,
            reduced_request,
            options,
            result,
            operation_started=operation_started,
            record_phase=measurements.append,
        )
    except phx.meshing.MeshingFailure as failure:
        expected_category = (
            phx.meshing.MeshingFailureCategory.TIMED_OUT
            if is_clock
            else phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
        )
        if failure.category is not expected_category:
            raise qualification_failure(
                "_polyhedral_qualification_periodic_resource_refusals",
                (f"Reduced {name} did not yield a resource refusal: {failure}"),
                {"expected_category": expected_category, "failure": failure},
            ) from failure
        refusal = {
            "control": name,
            "original_limit": declared[name],
            "reduced_limit": lowered,
            "limits_id": reduced_limits.limits_id,
            **FailureRecord.from_exception(failure, failure.evidence.stage).to_record(),
            "native_phases": _phase_recordings(measurements),
        }
    else:
        raise qualification_failure(
            "_polyhedral_qualification_periodic_resource_refusals",
            (f"Reduced observed {name} unexpectedly admitted a publication."),
            {},
        )
    if array_tree_fingerprint((source, request, options, result)) != original:
        raise qualification_failure(
            "_polyhedral_qualification_periodic_resource_refusals",
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
    return refusal


def _polyhedral_qualification_periodic_resource_refusals(
    source: phx.meshing.NativePolyhedralSource,
    request: phx.meshing.VolumeMeshingSpec,
    options: phx.meshing.NativeMeshingOptions,
    result: phx.meshing.CellMeshingResult,
    *,
    campaign_started: float | None = None,
) -> dict[str, Any]:
    """Check execution-record admission and one-control observed-boundary refusals."""
    observation = _observe_periodic_resources(request, result)
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
            _exercise_periodic_resource_boundary(
                source,
                request,
                options,
                result,
                name,
                value,
                declared,
                operation_started,
                original,
            )
        )
    if not refusals:
        raise qualification_failure(
            "_polyhedral_qualification_periodic_resource_refusals",
            ("No observed resource boundary was exercised."),
            {"refusals": refusals},
        )
    missing_gate = [
        name
        for name, item in bounds.items()
        if item["observed"] is None or not item["runtime_enforced"]
    ]
    return {
        "status": "incomplete" if missing_gate else "passed",
        "mandatory_capacity_gate": "incomplete" if missing_gate else "complete",
        "unobserved_or_unenforced_original_bounds": missing_gate,
        "positive": {
            "receipt_fingerprint": array_tree_fingerprint(receipt),
            "owner_id": receipt.owner_id,
            "bounds": bounds,
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
        "original_elapsed_debit_seconds": debit,
    }


def _polyhedral_qualification_periodic_initial_fields(
    result: Any,
) -> dict[str, jax.Array]:

    fv = UnstructuredFiniteVolumePlan.from_cell_mesh(result.mesh).prepare(
        cell_geometry=result.geometry
    )
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


def _prepare_periodic_transport(
    adaptation: phx.meshing.MeshAdaptationResult, epoch: int
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
    old_fv = UnstructuredFiniteVolumePlan.from_cell_mesh(source.mesh).prepare(
        cell_geometry=source.geometry
    )
    new_fv = UnstructuredFiniteVolumePlan.from_cell_mesh(target.mesh).prepare(
        cell_geometry=target.geometry
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
        semantics_id="original-periodic-power-source",
    )
    new_topology = CompositionEntry(
        target,
        entry_id="mesh",
        role="topology",
        owner_id="meshing",
        structure_id=new_epoch.epoch_id,
        revision_id=target.result_id,
        semantics_id="original-periodic-power-source",
    )
    entries, transports, output = [old_topology], [], {}
    inventory_defects = {}
    cell_fields = ("conserved", "pressure", "temperature", "fv_cell_solution")
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
        cell_fields=cell_fields,
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
            (name, fields[name], "physical-state"),
        ]
        for index, item in enumerate(fields.get(name + "_history", ())):
            field_entries.append((f"{name}-history:{index}", item, "history"))
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
            if role == "physical-state":
                output[name] = target_value
        if name + "_history" in fields:
            output[name + "_history"] = jnp.stack(
                tuple(remap.apply(item) for item in fields[name + "_history"])
            )


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
) -> _PeriodicNodalTransport:
    source = prepared.source
    target = prepared.target
    if adaptation.transfer is None:
        raise qualification_failure(
            "_polyhedral_qualification_periodic_transport",
            ("Scalar VEM state requires a source-backed vertex stencil."),
            {"adaptation": adaptation},
        )
    old_vem = prepare_polyhedral_h1_virtual_element_3d(
        source.mesh, cell_geometry=source.geometry
    )
    new_vem = prepare_polyhedral_h1_virtual_element_3d(
        target.mesh, cell_geometry=target.geometry
    )
    binding = TransferGeometryBinding(
        source.audit.geometry_id,
        target.audit.geometry_id,
        "topology-correspondence",
        source_topology_id=source.mesh.topology_id,
        target_topology_id=target.mesh.topology_id,
        coverage_defect=None,
    )
    nodal = old_vem.vertex_field_transfer(
        new_vem, adaptation.transfer, binding, field_name="temperature-vem"
    )
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
    }
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
            and nodal.properties.positivity_preserving
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
            Composition(tuple(entries), boundary_id="accepted-periodic-step"),
            reprepare=(new_topology,),
            transports=tuple(transports),
        ),
        accepted_boundary=True,
    )
    if not receipt.published:
        raise qualification_failure(
            "_polyhedral_qualification_periodic_transport",
            ("ALL physical/history periodic composition did not publish atomically."),
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
    if "vem_solution" in output:
        output["vem_solution"] = jnp.asarray(receipt.composition.value("vem_solution"))
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
        "nodal_semantics": "source-backed scalar edge trace interpolation, not interior nested VEM equality",
        "nodal_transfer": nodal_evidence,
        "derived_geometry_flux_and_residuals": "old accepted epoch retained; target reprepared from transported physical state",
    }


def _polyhedral_qualification_periodic_transport(
    adaptation: phx.meshing.MeshAdaptationResult, fields: dict[str, jax.Array], epoch: int
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    prepared = _prepare_periodic_transport(adaptation, epoch)
    _stage_periodic_cell_transport(prepared, fields)
    nodal_evidence = None
    if "vem_solution" in fields:
        nodal_preparation = _prepare_periodic_nodal_transport(
            prepared, adaptation, fields, epoch
        )
        _stage_periodic_nodal_transport(prepared, nodal_preparation, fields)
        nodal_evidence = nodal_preparation.nodal_evidence
    return _commit_periodic_transport(prepared, nodal_evidence)


def _polyhedral_qualification_periodic_adapt(
    result: Any, request: Any, limits: Any
) -> Any:
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_POLYHEDRAL,
        limits=limits,
        audit_policy=phx.meshing.CellMeshAuditPolicy(
            require_complete_association=True,
            watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
        ),
    )
    return phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(result, request, policy=policy)
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


@final
class _PeriodicAcceptedEpochs(phx.StrictModule):
    current: phx.meshing.CellMeshingResult
    fields: dict[str, jax.Array]
    epochs: list[tuple[phx.meshing.CellMeshingResult, dict[str, jax.Array]]]
    physics_records: list[dict[str, Any]]
    transfers: list[dict[str, Any]]
    error: float


@final
class _PeriodicLifecyclePublication(phx.StrictModule):
    source: phx.meshing.NativePolyhedralSource
    request: phx.meshing.VolumeMeshingSpec
    options: phx.meshing.NativeMeshingOptions
    original: phx.meshing.CellMeshingResult
    state: _PeriodicAcceptedEpochs


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
        case.name, args.resolution, args.capacity, args.timeout, source=True
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
    current, measured = ledger.run(
        phase, lambda: _measured(lambda: plan.execute(record_phase=phases.append))
    )
    stages[phase] = measured["wall_seconds"]
    _, record["source_acceptance"] = _polyhedral_acceptance(current, case)
    original = current
    fields = _polyhedral_qualification_periodic_initial_fields(current)
    epochs, physics_records, transfers = [], [], []
    phase = "original_source_quotient_physics"
    context.set_phase(phase)
    physics, error, fields = _polyhedral_qualification_periodic_physics(
        current, fields, args.repeats, stages, deadline=context.deadline
    )
    epochs.append((current, fields))
    physics_records.append(physics)
    state = _PeriodicAcceptedEpochs(
        current=current,
        fields=fields,
        epochs=epochs,
        physics_records=physics_records,
        transfers=transfers,
        error=error,
    )
    return _PeriodicLifecyclePublication(
        source=source, request=request, options=options, original=original, state=state
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
    args = context.args
    case = context.case
    stages = context.stages
    ledger = context.ledger
    request = publication.request
    current = state.current
    fields = state.fields
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
    split, measured = ledger.run(
        phase,
        lambda: _measured(
            lambda: _polyhedral_qualification_periodic_adapt(
                current, split_request, request.limits
            )
        ),
    )
    stages[phase] = measured["wall_seconds"]
    if split.target.mesh.connectivity.cell_count <= current.mesh.connectivity.cell_count:
        raise qualification_failure(
            "_polyhedral_qualification_periodic_lifecycle",
            ("Requested periodic refinement did not subdivide cells."),
            {"current": current, "split": split},
        )
    inverse_groups = _polyhedral_qualification_periodic_inverse_groups(split)
    phase = f"round{cycle}_allfield_history_refine_transfer"
    context.set_phase(phase)
    fields, proof = _polyhedral_qualification_periodic_transport(
        split, fields, len(epochs) - 1
    )
    transfers.append(proof)
    current = split.target
    _, proof = _polyhedral_acceptance(current, case)
    phase = f"round{cycle}_refined_quotient_continued_pde"
    context.set_phase(phase)
    physics, target_error, fields = _polyhedral_qualification_periodic_physics(
        current, fields, args.repeats, stages, seeds=fields, deadline=context.deadline
    )
    if physics["vem"]["physical_response"] <= 1e-10:
        raise qualification_failure(
            "_polyhedral_qualification_periodic_lifecycle",
            ("Refined source did not produce a nonconstant quotient VEM response."),
            {"physics": physics},
        )
    error = max(error, target_error)
    epochs.append((current, fields))
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
    args = context.args
    case = context.case
    stages = context.stages
    ledger = context.ledger
    request = publication.request
    original = publication.original
    state, inverse_groups = refined.state, refined.inverse_groups
    current = state.current
    fields = state.fields
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
    coarse, measured = ledger.run(
        phase,
        lambda: _measured(
            lambda: _polyhedral_qualification_periodic_adapt(
                current, coarse_request, request.limits
            )
        ),
    )
    stages[phase] = measured["wall_seconds"]
    phase = f"round{cycle}_allfield_history_inverse_transfer"
    context.set_phase(phase)
    fields, proof = _polyhedral_qualification_periodic_transport(
        coarse, fields, len(epochs) - 1
    )
    transfers.append(proof)
    current = coarse.target
    if current.mesh.connectivity.cell_count != original.mesh.connectivity.cell_count:
        raise qualification_failure(
            "_polyhedral_qualification_periodic_lifecycle",
            ("Periodic inverse failed to restore initial cell coverage."),
            {"current": current, "original": original},
        )
    _polyhedral_acceptance(current, case)
    phase = f"round{cycle}_coarse_quotient_continued_pde"
    context.set_phase(phase)
    physics, target_error, fields = _polyhedral_qualification_periodic_physics(
        current, fields, args.repeats, stages, seeds=fields, deadline=context.deadline
    )
    error = max(error, target_error)
    epochs.append((current, fields))
    physics_records.append(physics)
    return _PeriodicAcceptedEpochs(
        current=current,
        fields=fields,
        epochs=epochs,
        physics_records=physics_records,
        transfers=transfers,
        error=error,
    )


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
    phase = "original_cumulative_resource_capacity_proof"
    context.set_phase(phase)
    record["resources"] = _polyhedral_qualification_periodic_resource_refusals(
        source, request, options, original, campaign_started=started
    )
    phase = "default_source_allfield_history_fresh_cold_continued_pde"
    context.set_phase(phase)
    record["cold"] = _polyhedral_qualification_periodic_cold(
        source,
        request,
        options,
        tuple(epochs),
        args.repeats,
        args.target_error,
        stages,
        context.deadline,
    )
    if record["cold"]["status"] != "passed":
        raise qualification_failure(
            "_polyhedral_qualification_periodic_lifecycle",
            ("Default cold source continuation did not pass."),
            {"record": record},
        )
    elapsed = perf_counter() - started
    capacity_complete = record["resources"]["mandatory_capacity_gate"] == "complete"
    record.update(
        status="passed"
        if error <= args.target_error and elapsed <= args.timeout and capacity_complete
        else "failed",
        physical_error=error,
        target_error_reached=error <= args.target_error,
        time_to_target_error_seconds=elapsed if error <= args.target_error else None,
        mandatory_workflow_completion="complete" if capacity_complete else "incomplete",
        physics=physics_records,
        canonical_adaptation={
            "status": "complete",
            "cycles": args.adaptation_rounds,
            "transfers": transfers,
        },
        accepted_epochs=[_result_identity(result) for result, _ in epochs],
        logical_retained_bytes=model_array_bytes((source, request, options, epochs)),
        independent_scientific_certification="passed",
    )
    if elapsed > args.timeout:
        record["status"] = "timeout"


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
        record["peak_process_resident_bytes"] = peak_resident_bytes(resource.RUSAGE_SELF)
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
        result, measured = ledger.run(
            phase,
            lambda: _measured(lambda: plan.execute(record_phase=phases.append)),
        )
        stages[phase] = measured["wall_seconds"]
        phase = f"{epoch}_independent_acceptance"
        context.set_phase(phase)
        checked, measured = ledger.run(
            phase, lambda: _measured(lambda: _polyhedral_acceptance(result, case))
        )
        labels[epoch], acceptance[epoch] = checked
        stages[phase] = measured["wall_seconds"]
        results[epoch] = result
    return _PolyhedralGeneration(
        results=results,
        labels=labels,
        acceptance=acceptance,
        plans=plans,
        geometry=geometry,
        request=request,
        options=options,
    )


def _regenerate_polyhedral_target(
    context: _PolyhedralLifecycleContext, generation: _PolyhedralGeneration
) -> None:
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
                )
            )
        ),
    )
    stages[phase] = measured["wall_seconds"]
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


def _remap_polyhedral_materials(
    context: _PolyhedralLifecycleContext, generation: _PolyhedralGeneration
) -> _PolyhedralMaterialRemap:
    args = context.args
    stages = context.stages
    record = context.record
    ledger = context.ledger
    results = generation.results
    labels = generation.labels
    phase = "common_refinement_and_conservative_remap"
    context.set_phase(phase)
    remap_policy = phx.geometry.CommonRefinementPolicy(
        maximum_candidate_pairs=32 * args.capacity,
        maximum_accepted_pairs=32 * args.capacity,
        maximum_memory_bytes=4096 * args.capacity,
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
            )
        ),
    )
    remap_evidence = transferred.evidence
    stages[phase] = measured["wall_seconds"]
    record["remap"] = remap_evidence
    return transferred


def _complete_polyhedral_physics(
    context: _PolyhedralLifecycleContext,
    generation: _PolyhedralGeneration,
    remap: _PolyhedralMaterialRemap,
) -> None:
    args = context.args
    case = context.case
    started = context.started
    stages = context.stages
    record = context.record
    results = generation.results
    labels = generation.labels
    plans = generation.plans
    common = remap.common
    remapped = remap.remapped
    inventory_after = remap.inventory_after
    phase = "prepared_consumers_and_transferred_physics"
    context.set_phase(phase)
    physical = _polyhedral_physics(
        results["target"],
        remapped,
        args.repeats,
        stages,
        results["source"],
        common,
        labels["source"],
    )
    physics, error, retained = physical.record, physical.error, physical.retained
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
    elapsed = perf_counter() - started
    record.update(
        {
            "physics": physics,
            "physical_error": error,
            "mandatory_workflow_completion": "complete",
            "target_error_reached": error <= args.target_error,
            "status": "passed" if error <= min(args.target_error, 1e-7) else "failed",
            "time_to_target_error_seconds": elapsed
            if error <= args.target_error
            else None,
            "logical_retained_bytes": model_array_bytes(
                (retained, results, plans, common, remapped, labels)
            ),
            "independent_scientific_certification": "passed",
        }
    )
    if elapsed > args.timeout:
        record["status"] = "timeout"


def qualify_native_polyhedral(args: QualificationControls) -> dict[str, object]:
    """Nonconvex material PLC → power cells → conservative state → VEM/FV."""
    if "periodic-quotient" in polyhedral_case(args.polyhedral_source).controls:
        return _polyhedral_qualification_periodic_lifecycle(args)
    started = perf_counter()
    deadline = Deadline(started + args.timeout)
    case = polyhedral_case(args.polyhedral_source)
    stages: dict[str, float] = {}
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
        _regenerate_polyhedral_target(context, generation)
        _record_polyhedral_generation(context, generation)
        remap = _remap_polyhedral_materials(context, generation)
        _complete_polyhedral_physics(context, generation, remap)
    except phx.meshing.MeshingFailure as failure:
        record_failure(record, failure, phase)
        record.update(
            {
                "status": "timeout"
                if failure.category is phx.meshing.MeshingFailureCategory.TIMED_OUT
                else "failed",
                "failure_message": failure.evidence.message,
            }
        )
    except (RuntimeError, ValueError, TypeError, TimeoutError) as failure:
        record_failure(record, failure, phase)
    finally:
        record["native_phases"] = _phase_recordings(phases)
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
    return record


def worker(argv: tuple[str, ...]) -> dict[str, Any]:
    """Decode process arguments and run the scenario-owned cold consumer."""
    return _polyhedral_qualification_periodic_cold_process(
        argv[0], argv[1], argv[2], int(argv[3]), float(argv[4])
    )


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "worker":
        raise ValueError(
            "qualification.subcommand must equal worker; got " + repr(sys.argv[1:])
        )
    print(json.dumps(worker(tuple(sys.argv[2:]))))


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host qualification controls."""
    return qualify_native_polyhedral(args)
