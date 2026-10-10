#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native nonconvex/material polyhedra -> canonical adaptation -> VEM/FV.

Run from the repository with the native meshcore library selected. No external
mesher is invoked. Both carriers publish through independent volume coverage;
solver status, projector rank, FV geometry defects and conservative physical/
history CompositionRebind evidence are printed. The source is an L-shaped
three-voxel domain with two materials; regeneration adds a y=1 power bisector.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx
from phydrax.discretization import (
    CellGeometrySpec,
    CellMesh,
    PolyhedralConnectivity,
    prepare_polyhedral_h1_virtual_element_3d,
    PreparedPolyhedralH1VirtualElement3D,
    TopologyEpoch,
    UnstructuredConservativeRemapPlan,
)
from phydrax.discretization.finite_volume import (
    HybridDiffusionBoundary,
    HybridMimeticDiffusion,
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
)
from phydrax.geometry import CommonRefinementStatus, PreparedCommonRefinement
from phydrax.lifecycle import (
    commit_composition_rebind,
    Composition,
    CompositionEntry,
    CompositionRebind,
    CompositionRebindReceipt,
)
from phydrax.linalg import FunctionLinearOperator, LinearSolveResult
from phydrax.meshing import CellMeshingResult, PiecewiseLinearComplex


M = phx.meshing


def material_domain() -> PiecewiseLinearComplex:
    loops = (
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    )
    vertices: list[tuple[int, int, int]] = []
    ids: dict[tuple[int, int, int], int] = {}
    faces: dict[tuple[int, ...], list[tuple[tuple[int, ...], int]]] = {}
    for cube, region in (((0, 0, 0), 0), ((1, 0, 0), 1), ((0, 1, 0), 0)):
        corners = []
        for i in range(8):
            point = (
                cube[0] + (i & 1),
                cube[1] + ((i >> 1) & 1),
                cube[2] + ((i >> 2) & 1),
            )
            if point not in ids:
                ids[point] = len(vertices)
                vertices.append(point)
            corners.append(ids[point])
        for local in loops:
            face = tuple(corners[i] for i in local)
            faces.setdefault(tuple(sorted(face)), []).append((face, region))
    polygons, incidence = [], []
    for entries in faces.values():
        loop, region = entries[0]
        other = entries[1][1] if len(entries) == 2 else -1
        if other != region:
            polygons.append(loop)
            incidence.append((other, region))
    pairs, facets = np.unique(incidence, axis=0, return_inverse=True)
    return M.PiecewiseLinearComplex(vertices, polygons, facets, pairs, ("left", "right"))


def generate(
    complex_: PiecewiseLinearComplex, *, sites: ArrayLike | None = None
) -> CellMeshingResult:
    source = (
        M.NativePlcSource(complex_, "polyhedral-l-domain", "r1")
        if sites is None
        else M.NativePolyhedralSource(complex_, "polyhedral-l-domain", "r1", sites=sites)
    )
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "polyhedral-domain-facets",
        np.arange(complex_.facet_count, dtype=np.int64),
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(3, 3, M.CellFamilyPolicy(required=("polyhedron",))),
        scope,
        M.VolumeFillStrategy.POLYHEDRAL,
        size_controls=(
            M.UniformSizeControl(scope, 10.0, strength=M.SizeControlStrength.SOFT),
        ),
    )
    provider = M.NativeMeshingProvider(M.NativeMeshingOptions("plc_restricted_power"))
    return provider.plan(
        source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
    ).execute()


def prepare_vem(
    mesh: CellMesh, geometry: CellGeometrySpec
) -> tuple[PreparedPolyhedralH1VirtualElement3D, FunctionLinearOperator, Array]:
    """Prepare -Laplace(u)+u=1 with homogeneous natural flux, constant u=1.

    Reaction and forcing use the same positive vertex-lumped volume rule,
    whose integral of a constant is the exact reported cell measure. All VEM
    vertices remain unknown; this is not a boundary-only prescribed solution.
    """
    vem = prepare_polyhedral_h1_virtual_element_3d(mesh, cell_geometry=geometry)
    c = mesh.connectivity
    if not isinstance(c, PolyhedralConnectivity):
        raise TypeError("Native polyhedral VEM requires root PolyhedralConnectivity.")
    offsets, vertices = (
        np.asarray(c.cell_vertex_offsets),
        np.asarray(c.cell_vertex_values),
    )
    lumped = np.zeros(c.vertex_count)
    for cell, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        np.add.at(lumped, vertices[a:b], float(vem.evidence.cell_volumes[cell]) / (b - a))
    if np.any(~np.isfinite(lumped)) or np.any(lumped <= 0):
        raise ValueError(
            "VEM reaction solve requires positive mass on every unknown vertex."
        )
    mass = jnp.asarray(lumped)
    base = vem.as_linear_operator()
    operator = phx.linalg.FunctionLinearOperator(
        lambda x: base.mv(x) + mass * x,
        source=base.source,
        target=base.target,
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
    )
    return vem, operator, mass


def solve_vem(
    mesh: CellMesh, geometry: CellGeometrySpec
) -> tuple[PreparedPolyhedralH1VirtualElement3D, LinearSolveResult, float]:
    vem, operator, mass = prepare_vem(mesh, geometry)
    solved = phx.linalg.solve(
        phx.linalg.LinearSystem(operator),
        mass,
        policy=phx.linalg.LinearSolvePolicy(phx.linalg.ConjugateGradient()),
    )
    if not bool(jnp.all(solved.successful)):
        raise RuntimeError("Native polyhedral VEM reaction-diffusion solve failed.")
    return vem, solved, float(jnp.max(jnp.abs(solved.value - 1)))


def prepare_fv(
    mesh: CellMesh, geometry: CellGeometrySpec
) -> tuple[
    UnstructuredFiniteVolumeDiscretization,
    HybridMimeticDiffusion,
    Array,
    HybridDiffusionBoundary,
]:
    """Prepare a linear harmonic field with a rotated SPD diffusion tensor.

    The global shared-face mimetic formulation is used, not an orthogonal TPFA
    assumption silently applied to a nonorthogonal power cell.
    """
    fv = UnstructuredFiniteVolumePlan.from_cell_mesh(mesh).prepare(cell_geometry=geometry)
    if not isinstance(fv, UnstructuredFiniteVolumeDiscretization):
        raise TypeError(
            "Native polyhedral FV requires canonical prepared unstructured geometry."
        )
    diffusion = HybridMimeticDiffusion(fv)
    coefficients = jnp.asarray([1.0, 2.0, 3.0])
    exact_faces = jnp.asarray(
        1 + fv.face_centers @ coefficients, dtype=fv.face_centers.dtype
    )
    boundary = HybridDiffusionBoundary(
        fv,
        dirichlet={
            int(face): float(exact_faces[face])
            for face in np.flatnonzero(np.asarray(fv.neighbor_cells) < 0)
        },
    )
    tensor = jnp.asarray([[2.0, 0.25, 0.0], [0.25, 3.0, 0.1], [0.0, 0.1, 1.0]])
    return fv, diffusion, tensor, boundary


def _solve_prepared_fv(
    fv: UnstructuredFiniteVolumeDiscretization,
    diffusion: HybridMimeticDiffusion,
    tensor: ArrayLike,
    boundary: HybridDiffusionBoundary,
) -> tuple[LinearSolveResult, float, float]:
    solved = diffusion.solve(tensor, boundary)
    if not bool(jnp.all(solved.successful)):
        raise RuntimeError("Native polyhedral mimetic FV diffusion solve failed.")
    exact_cells = 1 + fv.cell_centers @ jnp.asarray([1.0, 2.0, 3.0])
    error = float(jnp.max(jnp.abs(solved.value[: diffusion.cell_count] - exact_cells)))
    residual = diffusion.residual(
        solved.value[: diffusion.cell_count],
        solved.value[diffusion.cell_count :],
        tensor,
        boundary,
    )
    return solved, error, float(jnp.max(jnp.abs(residual)))


def solve_fv(
    mesh: CellMesh, geometry: CellGeometrySpec
) -> tuple[UnstructuredFiniteVolumeDiscretization, LinearSolveResult, float, float]:
    prepared = prepare_fv(mesh, geometry)
    return prepared[0], *_solve_prepared_fv(*prepared)


def rebind_inventory_history(
    source: CellMeshingResult,
    target: CellMeshingResult,
    common: PreparedCommonRefinement,
    fv: UnstructuredFiniteVolumeDiscretization,
    inventory: np.ndarray,
    /,
) -> tuple[CompositionRebindReceipt, np.ndarray]:
    """Transfer actual physical/history cell-average state at an accepted boundary."""
    old_fv = UnstructuredFiniteVolumePlan.from_cell_mesh(source.mesh).prepare(
        cell_geometry=source.geometry
    )
    remap = UnstructuredConservativeRemapPlan(
        old_fv,
        fv,
        common.target_offsets,
        common.source_cells,
        common.volumes,
        method="common-refinement",
        provenance=common.refinement_id,
        intersection_error_bounds=common.volume_error_bounds,
    )
    remapped = np.asarray(remap.apply(inventory))
    previous = 0.9 * inventory
    source_epoch = TopologyEpoch(0, old_fv.geometry_id, old_fv.topology_id, "serial")
    target_epoch = TopologyEpoch(1, fv.geometry_id, fv.topology_id, "serial")
    transfer = remap.epoch_transition(
        old_fv.cell_space, fv.cell_space, source_epoch, target_epoch
    )
    old_topology = CompositionEntry(
        source,
        entry_id="mesh",
        role="topology",
        owner_id="meshing",
        structure_id=source_epoch.epoch_id,
        revision_id=source.result_id,
        semantics_id="represented-polyhedral-domain",
    )
    new_topology = CompositionEntry(
        target,
        entry_id="mesh",
        role="topology",
        owner_id="meshing",
        structure_id=target_epoch.epoch_id,
        revision_id=target.result_id,
        semantics_id="represented-polyhedral-domain",
    )
    old_state = CompositionEntry(
        inventory,
        entry_id="inventory",
        role="physical-state",
        owner_id="finite-volume",
        structure_id=source_epoch.epoch_id,
        revision_id="accepted",
        semantics_id="cell-average",
        dependencies=(old_topology.binding("structure"),),
    )
    new_state = CompositionEntry(
        remapped,
        entry_id="inventory",
        role="physical-state",
        owner_id="finite-volume",
        structure_id=target_epoch.epoch_id,
        revision_id="regenerated",
        semantics_id="cell-average",
        dependencies=(new_topology.binding("structure"),),
    )
    old_history = CompositionEntry(
        previous,
        entry_id="previous-inventory",
        role="history",
        owner_id="finite-volume",
        structure_id=source_epoch.epoch_id,
        revision_id="accepted-history",
        semantics_id="cell-average",
        dependencies=(old_topology.binding("structure"),),
    )
    new_history = CompositionEntry(
        remap.apply(previous),
        entry_id="previous-inventory",
        role="history",
        owner_id="finite-volume",
        structure_id=target_epoch.epoch_id,
        revision_id="regenerated-history",
        semantics_id="cell-average",
        dependencies=(new_topology.binding("structure"),),
    )
    composition = Composition(
        (old_topology, old_state, old_history), boundary_id="accepted-step"
    )
    receipt = commit_composition_rebind(
        CompositionRebind(
            composition,
            reprepare=(new_topology,),
            transports=(
                transfer.composition_transport(old_state, new_state),
                transfer.composition_transport(old_history, new_history),
            ),
        ),
        accepted_boundary=True,
    )
    if not receipt.published:
        raise RuntimeError("Conservative physical/history polyhedral rebind failed.")
    remapped = np.asarray(receipt.composition.value("inventory"))
    return receipt, remapped


def main() -> None:
    complex_ = material_domain()
    source = generate(complex_)
    adaptation = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            source,
            M.PolyhedralMeshAdaptation(
                M.PolyhedralAdaptationOperation.REGENERATE,
                complex_=complex_,
                sites=np.asarray([[0.5, 0.25, 0.5], [0.5, 1.75, 0.5]], dtype=np.float64),
            ),
            policy=M.MeshAdaptationPolicy(
                M.MeshAdaptationRoute.NATIVE_POLYHEDRAL,
                audit_policy=M.CellMeshAuditPolicy(
                    require_complete_association=True,
                    watertight_boundary=M.CellMeshAuditDisposition.REJECT,
                ),
            ),
        )
    )
    target = adaptation.target
    vem, vem_solve, vem_error = solve_vem(target.mesh, target.geometry)
    if source.certification is None or target.certification is None:
        raise RuntimeError(
            "Native polyhedral publications require certification evidence."
        )
    fv, diffusion, tensor, boundary = prepare_fv(target.mesh, target.geometry)
    fv_solve, fv_error, fv_residual = _solve_prepared_fv(fv, diffusion, tensor, boundary)
    connectivity = target.mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Native polyhedral target requires root PolyhedralConnectivity.")
    source_connectivity = source.mesh.connectivity
    if not isinstance(source_connectivity, PolyhedralConnectivity):
        raise TypeError("Native polyhedral source requires root PolyhedralConnectivity.")
    offsets, faces = (
        np.asarray(connectivity.cell_face_offsets),
        np.asarray(connectivity.cell_face_values),
    )
    normals = np.asarray(fv.area_vectors)
    fv_rank_margin = min(
        float(
            np.linalg.svd(
                normals[faces[a:b]] / float(fv.cell_volumes[cell]), compute_uv=False
            )[-1]
        )
        for cell, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True))
    )
    common = adaptation.common_refinement
    if common is None or common.status is not CommonRefinementStatus.SUCCESS:
        raise RuntimeError(
            "Native polyhedral adaptation lacks complete common refinement."
        )
    # The field has a material jump. Transfer each material's inventory through
    # actual overlap volumes rather than a generator-site interpolation stencil.
    # Zones bind canonical cell IDs, so the scientific values need no private
    # construction object or guessed row order.
    source_ids = np.asarray(source_connectivity.cell_global_ids)
    inventory = np.zeros(source_ids.size)
    for zone in source.zones:
        inventory[np.isin(source_ids, np.asarray(zone.scope.entity_ids))] = (
            2.0 if zone.name == "left" else 5.0
        )
    receipt, remapped = rebind_inventory_history(source, target, common, fv, inventory)
    before = float(np.dot(inventory, np.asarray(common.source_measures)))
    after = float(np.dot(remapped, np.asarray(common.target_measures)))
    # The transferred inventory enters a real subsequent PDE update.
    integrated_source = jnp.asarray(remapped) * fv.cell_volumes
    continued = diffusion.solve(tensor, boundary, source=integrated_source)
    if not bool(jnp.all(continued.successful)):
        raise RuntimeError(
            "Native polyhedral FV continuation failed after conservative remap."
        )
    continued_residual = diffusion.residual(
        continued.value[: diffusion.cell_count],
        continued.value[diffusion.cell_count :],
        tensor,
        boundary,
        source=integrated_source,
    )
    continuation_residual = float(jnp.max(jnp.abs(continued_residual)))
    continued_change = float(
        jnp.max(
            jnp.abs(
                continued.value[: diffusion.cell_count]
                - fv_solve.value[: diffusion.cell_count]
            )
        )
    )
    if (
        fv_rank_margin <= 0
        or abs(before - after) > 1e-10
        or max(vem_error, fv_error, fv_residual, continuation_residual) > 1e-7
    ):
        raise RuntimeError(
            "Native polyhedral consumer evidence exceeds the requested tolerance."
        )
    print(
        json.dumps(
            {
                "source_cells": source_connectivity.cell_count,
                "target_cells": connectivity.cell_count,
                "source_certified": source.certification.passed,
                "target_certified": target.certification.passed,
                "domain_measure": float(jnp.sum(fv.cell_volumes)),
                "vem_successful": bool(jnp.all(vem_solve.successful)),
                "vem_minimum_rank_margin": vem.evidence.minimum_rank_margin,
                "vem_reproduction_defect": vem.evidence.maximum_reproduction_defect,
                "vem_constant_solution_error": vem_error,
                "fv_successful": bool(jnp.all(fv_solve.successful)),
                "fv_minimum_normal_rank_margin": fv_rank_margin,
                "fv_linear_solution_error": fv_error,
                "fv_residual": fv_residual,
                "continuation_successful": bool(jnp.all(continued.successful)),
                "continuation_residual": continuation_residual,
                "continued_field_change": continued_change,
                "remap_status": common.status.name.lower(),
                "inventory_before": before,
                "inventory_after": after,
                "canonical_adaptation_status": adaptation.status.value,
                "composition_rebind_published": receipt.published,
                "composition_rebind_receipt_id": receipt.receipt_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
