# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Restricted-power polyhedra in a two-material L, consumed by mimetic FV.

Set PHYDRAX_MESHCORE_LIBRARY to the intended library and JAX_ENABLE_X64=true.
"""

import json
import os

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization import PolyhedralConnectivity
from phydrax.discretization.finite_volume import (
    HybridDiffusionBoundary,
    HybridMimeticDiffusion,
    UnstructuredFiniteVolumePlan,
)
from phydrax.discretization.vem import prepare_polyhedral_h1_virtual_element_3d
from phydrax.linalg import (
    ConjugateGradient,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    TolerancePolicy,
)
from phydrax.meshing import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingEntityKind,
    MeshingScope,
    NativePolyhedralSchedule,
    PiecewiseLinearComplex,
    PolyhedralAdaptationOperation,
    PolyhedralAdaptationRequest,
    RegionControl,
    RegionRole,
    RegionSeed,
    SizeControlStrength,
    UniformSizeControl,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing.providers import (
    NativeMeshingOptions,
    NativeMeshingProvider,
    NativePlcSource,
    NativePolyhedralSource,
)


def material_domain(name: str) -> PiecewiseLinearComplex:
    """Three unit voxels with the internal two-material interface retained."""
    face_loops = (
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    )
    vertices, positions, faces = [], {}, {}
    for cube, region in (((0, 0, 0), 0), ((1, 0, 0), 1), ((0, 1, 0), 0)):
        corners = []
        for corner in range(8):
            p = (
                cube[0] + (corner & 1),
                cube[1] + ((corner >> 1) & 1),
                cube[2] + ((corner >> 2) & 1),
            )
            if p not in positions:
                positions[p] = len(vertices)
                vertices.append(tuple(float(x) for x in p))
            corners.append(positions[p])
        for loop in face_loops:
            face = tuple(corners[i] for i in loop)
            faces.setdefault(tuple(sorted(face)), []).append((face, region))
    polygons, regions = [], []
    for entries in faces.values():
        face, region = entries[0]
        other = entries[1][1] if len(entries) == 2 else -1
        if other != region:
            polygons.append(np.asarray(face, dtype=np.int64))
            regions.append((other, region))
    pairs, facets = np.unique(
        np.asarray(regions, dtype=np.int64), axis=0, return_inverse=True
    )
    vertex_ids = tuple(f"{name}:vertex:{i}:{p}" for i, p in enumerate(vertices))
    facet_ids = tuple(f"{name}:facet:region-pair:{int(a)}:{int(b)}" for a, b in pairs)
    return PiecewiseLinearComplex(
        np.asarray(vertices, dtype=np.float64),
        polygons,
        facets,
        pairs,
        ("left", "right"),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=vertex_ids,
        vertex_occurrence_paths=((),) * len(vertex_ids),
        facet_ids=facet_ids,
        facet_occurrence_paths=((),) * len(facet_ids),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((), ()),
    )


def main() -> None:
    if not os.environ.get("PHYDRAX_MESHCORE_LIBRARY"):
        raise RuntimeError(
            "Set PHYDRAX_MESHCORE_LIBRARY to the native library before running."
        )
    source_id = "polyhedral-l-domain"
    complex_ = material_domain(source_id)
    revision = "r1"
    source = NativePlcSource(complex_, source_id, revision)
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    controls = tuple(
        RegionControl(
            MeshingScope(
                source.source_id,
                source.source_revision,
                MeshingEntityKind.GEOMETRY,
                3,
                source.scope_entity_set_id(3),
                np.asarray((complex_.region_entity_ids[i],), dtype=np.int64),
            ),
            name,
            f"{name}-material",
            RegionRole.SOLID,
        )
        for i, name in enumerate(complex_.region_ids)
    )
    seeds = (
        RegionSeed(
            np.asarray((0.5, 0.5, 0.5), dtype=np.float64),
            "left",
            "left-material",
            RegionRole.SOLID,
        ),
        RegionSeed(
            np.asarray((1.5, 0.5, 0.5), dtype=np.float64),
            "right",
            "right-material",
            RegionRole.SOLID,
        ),
    )
    spec = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("polyhedron",))),
        scope,
        VolumeFillStrategy.POLYHEDRAL,
        size_controls=(
            UniformSizeControl(scope, 10.0, strength=SizeControlStrength.SOFT),
        ),
        region_controls=controls,
        region_seeds=seeds,
    )
    options = NativeMeshingOptions("plc_restricted_power")
    plan = NativeMeshingProvider(options).plan(
        source, spec, coordinate_contract=phx.SpatialCoordinateContract.si()
    )
    if plan.route != "plc_restricted_power":
        raise RuntimeError(f"Unexpected route {plan.route}")
    result = plan.execute()
    if (
        not result.audit.passed
        or result.certification is None
        or not result.certification.passed
    ):
        raise RuntimeError("Native source certification failed")
    connectivity = result.mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity) or {
        b.cell_kind for b in result.mesh.blocks if b.cell_count
    } != {"polyhedron"}:
        raise RuntimeError("Native result is not polyhedral")
    coverage = result.certification.coverage
    if coverage is None:
        raise RuntimeError("Missing material coverage certificate")
    measures = dict(
        zip(coverage.region_ids, coverage.achieved_region_measures, strict=True)
    )
    if set(measures) != {"left", "right"}:
        raise RuntimeError(f"Material coverage mismatch: {measures}")
    measured_volumes: dict[str, float] = {}
    for region, expected in (("left", 2.0), ("right", 1.0)):
        measure = measures[region]
        if measure is None or abs(float(measure) - expected) > 1e-10:
            raise RuntimeError(f"Material coverage mismatch: {measures}")
        measured_volumes[region] = float(measure)
    attribute = next((a for a in result.attributes if a.name == "power_site_index"), None)
    cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    if attribute is None or not np.array_equal(
        np.asarray(attribute.scope.entity_ids, dtype=np.int64), np.sort(cell_ids)
    ):
        raise RuntimeError("Missing cell-aligned power-site ownership")
    indices = np.asarray(attribute.values, dtype=np.int64)
    site_ids = tuple(
        label.name.removeprefix("power-site:")
        for label in result.labels
        if label.name.startswith("power-site:")
    )
    if indices.shape != cell_ids.shape or len(set(indices.tolist())) != len(site_ids):
        raise RuntimeError("Generated power-site ownership is incomplete")
    fv = UnstructuredFiniteVolumePlan.from_cell_mesh(result.mesh).prepare()
    centers = np.asarray(fv.face_centers, dtype=np.float64)
    faces = np.flatnonzero(np.asarray(fv.neighbor_cells) < 0)
    boundary = HybridDiffusionBoundary(
        fv,
        dirichlet={
            int(i): float(1.0 + centers[i, 0] + 2.0 * centers[i, 1] + 3.0 * centers[i, 2])
            for i in faces
        },
    )
    system = HybridMimeticDiffusion(fv)
    policy = LinearSolvePolicy(
        GMRES(restart=40, stagnation_iterations=40),
        tolerance=TolerancePolicy(relative=1e-12, absolute=1e-14),
    )
    tensor = jnp.asarray(((2.0, 0.25, 0.0), (0.25, 3.0, 0.1), (0.0, 0.1, 1.0)))
    solved = system.solve(tensor, boundary, policy=policy)
    count = int(fv.cell_volumes.shape[0])
    cc = np.asarray(fv.cell_centers, dtype=np.float64)
    exact = 1.0 + cc[:, 0] + 2.0 * cc[:, 1] + 3.0 * cc[:, 2]
    error = float(
        np.max(
            np.abs(np.asarray(solved.value[:count], dtype=np.float64) - exact),
            initial=0.0,
        )
    )
    operator, rhs = system.linear_system(tensor, boundary)
    residual = float(jnp.linalg.norm(operator.operator.mv(solved.value) - rhs)) / max(
        1.0, float(jnp.linalg.norm(rhs))
    )
    if not bool(jnp.all(solved.successful)) or error > 2e-8 or residual > 2e-8:
        raise RuntimeError(f"Mimetic FV affine solve failed: {error=} {residual=}")
    target_sites = np.asarray(((0.5, 0.25, 0.5), (0.5, 1.75, 0.5)), dtype=np.float64)
    target_ids = tuple(f"{source_id}:regenerated-site:{i}" for i in range(2))
    target_revision = canonical_fingerprint(
        {
            "complex": complex_.complex_id,
            "sites": target_sites.tolist(),
            "coordinate_unit": "meter",
            "parent_source": revision,
            "operation": "regenerate-authored-sites",
        }
    )
    successor = NativePolyhedralSource(
        complex_,
        source_id,
        target_revision,
        sites=target_sites,
        weights=np.zeros(len(target_sites), dtype=np.float64),
        site_ids=target_ids,
    )
    successor_scope = MeshingScope(
        successor.source_id,
        successor.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        successor.scope_entity_set_id(2),
        successor.scope_entity_ids(2),
    )
    successor_controls = tuple(
        RegionControl(
            MeshingScope(
                successor.source_id,
                successor.source_revision,
                MeshingEntityKind.GEOMETRY,
                3,
                successor.scope_entity_set_id(3),
                np.asarray((complex_.region_entity_ids[i],), dtype=np.int64),
            ),
            name,
            f"{name}-material",
            RegionRole.SOLID,
        )
        for i, name in enumerate(complex_.region_ids)
    )
    successor_spec = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("polyhedron",))),
        successor_scope,
        VolumeFillStrategy.POLYHEDRAL,
        size_controls=(
            UniformSizeControl(successor_scope, 10.0, strength=SizeControlStrength.SOFT),
        ),
        region_controls=successor_controls,
        region_seeds=seeds,
    )
    request = PolyhedralAdaptationRequest(
        PolyhedralAdaptationOperation.REGENERATE,
        regeneration_source=successor,
        volume_specification=successor_spec,
        schedule=NativePolyhedralSchedule(),
        frame_id=result.coordinate_contract.spatial_id,
    )
    adaptation = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            result,
            request,
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_POLYHEDRAL
            ),
        )
    )
    target = adaptation.target
    if (
        target.mesh.topology_id == result.mesh.topology_id
        or target.certification is None
        or not target.certification.passed
    ):
        raise RuntimeError("Regeneration did not publish a certified successor")
    target_connectivity = target.mesh.connectivity
    if not isinstance(target_connectivity, PolyhedralConnectivity):
        raise TypeError("Regenerated VEM requires polyhedral connectivity.")
    vem = prepare_polyhedral_h1_virtual_element_3d(target.mesh)
    offsets = np.asarray(target_connectivity.cell_vertex_offsets, dtype=np.int64)
    vertex_rows = np.asarray(target_connectivity.cell_vertex_values, dtype=np.int64)
    lumped = np.zeros(target_connectivity.vertex_count, dtype=np.float64)
    for cell, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        np.add.at(
            lumped,
            vertex_rows[start:stop],
            float(vem.evidence.cell_volumes[cell]) / (stop - start),
        )
    if np.any(~np.isfinite(lumped)) or np.any(lumped <= 0.0):
        raise RuntimeError("VEM reaction mass must be positive")
    mass = jnp.asarray(lumped)
    base = vem.as_linear_operator()
    reaction = FunctionLinearOperator(
        lambda values: base.mv(values) + mass * values,
        source=base.source,
        target=base.target,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
    )
    vem_solved = phx.linalg.solve(
        LinearSystem(reaction), mass, policy=LinearSolvePolicy(ConjugateGradient())
    )
    vem_error = float(jnp.max(jnp.abs(vem_solved.value - 1.0)))
    if (
        not bool(jnp.all(vem_solved.successful))
        or vem_error > 2e-8
        or vem.evidence.minimum_rank_margin <= 0
    ):
        raise RuntimeError(f"Polyhedral VEM reaction solve failed: {vem_error=}")
    common = phx.geometry.prepare_common_refinement(result.mesh, target.mesh)
    if common.status is not phx.geometry.CommonRefinementStatus.SUCCESS:
        raise RuntimeError(f"Common refinement failed: {common.status}")
    target_fv = UnstructuredFiniteVolumePlan.from_cell_mesh(target.mesh).prepare()
    face_offsets = np.asarray(target_connectivity.cell_face_offsets, dtype=np.int64)
    face_rows = np.asarray(target_connectivity.cell_face_values, dtype=np.int64)
    normals = np.asarray(target_fv.area_vectors, dtype=np.float64)
    rank_margin = min(
        float(
            np.linalg.svd(
                normals[face_rows[start:stop]] / float(target_fv.cell_volumes[cell]),
                compute_uv=False,
            )[-1]
        )
        for cell, (start, stop) in enumerate(
            zip(face_offsets[:-1], face_offsets[1:], strict=True)
        )
    )
    if rank_margin <= 0.0:
        raise RuntimeError("Regenerated FV face-normal rank is deficient.")
    remap = phx.discretization.UnstructuredConservativeRemapPlan(
        fv,
        target_fv,
        common.target_offsets,
        common.source_cells,
        common.volumes,
        method="common-refinement",
        provenance=common.refinement_id,
    )
    inventory = np.zeros(len(cell_ids), dtype=np.float64)
    for zone in result.zones:
        if zone.name in ("left", "right"):
            inventory[
                np.isin(cell_ids, np.asarray(zone.scope.entity_ids, dtype=np.int64))
            ] = 2.0 if zone.name == "left" else 5.0
    if np.any(inventory == 0.0):
        raise RuntimeError("Source material zones do not cover the FV inventory")
    transferred = np.asarray(remap.apply(inventory), dtype=np.float64)
    before = float(np.dot(inventory, np.asarray(common.source_measures)))
    after = float(np.dot(transferred, np.asarray(common.target_measures)))
    if abs(before - after) > 1e-10:
        raise RuntimeError("Regeneration changed material inventory")
    source_epoch = phx.discretization.TopologyEpoch(
        0, fv.geometry_id, fv.topology_id, "serial"
    )
    target_epoch = phx.discretization.TopologyEpoch(
        1, target_fv.geometry_id, target_fv.topology_id, "serial"
    )
    transition = remap.epoch_transition(
        fv.cell_space, target_fv.cell_space, source_epoch, target_epoch
    )
    old_mesh = phx.lifecycle.CompositionEntry(
        result,
        entry_id="mesh",
        role="topology",
        owner_id="meshing",
        structure_id=source_epoch.epoch_id,
        revision_id=result.result_id,
        semantics_id="represented-polyhedral-domain",
    )
    new_mesh = phx.lifecycle.CompositionEntry(
        target,
        entry_id="mesh",
        role="topology",
        owner_id="meshing",
        structure_id=target_epoch.epoch_id,
        revision_id=target.result_id,
        semantics_id="represented-polyhedral-domain",
    )
    old_state = phx.lifecycle.CompositionEntry(
        inventory,
        entry_id="inventory",
        role="physical-state",
        owner_id="finite-volume",
        structure_id=source_epoch.epoch_id,
        revision_id="accepted",
        semantics_id="cell-average",
        dependencies=(old_mesh.binding("structure"),),
    )
    new_state = phx.lifecycle.CompositionEntry(
        transferred,
        entry_id="inventory",
        role="physical-state",
        owner_id="finite-volume",
        structure_id=target_epoch.epoch_id,
        revision_id="regenerated",
        semantics_id="cell-average",
        dependencies=(new_mesh.binding("structure"),),
    )
    old_history = phx.lifecycle.CompositionEntry(
        0.9 * inventory,
        entry_id="previous-inventory",
        role="history",
        owner_id="finite-volume",
        structure_id=source_epoch.epoch_id,
        revision_id="accepted-history",
        semantics_id="cell-average",
        dependencies=(old_mesh.binding("structure"),),
    )
    new_history = phx.lifecycle.CompositionEntry(
        remap.apply(0.9 * inventory),
        entry_id="previous-inventory",
        role="history",
        owner_id="finite-volume",
        structure_id=target_epoch.epoch_id,
        revision_id="regenerated-history",
        semantics_id="cell-average",
        dependencies=(new_mesh.binding("structure"),),
    )
    composition = phx.lifecycle.Composition(
        (old_mesh, old_state, old_history), boundary_id="accepted-step"
    )
    receipt = phx.lifecycle.commit_composition_rebind(
        phx.lifecycle.CompositionRebind(
            composition,
            reprepare=(new_mesh,),
            transports=(
                transition.composition_transport(old_state, new_state),
                transition.composition_transport(old_history, new_history),
            ),
        ),
        accepted_boundary=True,
    )
    if not receipt.published:
        raise RuntimeError("Conservative physical/history rebind failed")
    target_centers = np.asarray(target_fv.face_centers, dtype=np.float64)
    target_faces = np.flatnonzero(np.asarray(target_fv.neighbor_cells) < 0)
    target_boundary = HybridDiffusionBoundary(
        target_fv,
        dirichlet={
            int(i): float(
                1.0
                + target_centers[i, 0]
                + 2.0 * target_centers[i, 1]
                + 3.0 * target_centers[i, 2]
            )
            for i in target_faces
        },
    )
    target_system = HybridMimeticDiffusion(target_fv)
    target_solved = target_system.solve(tensor, target_boundary, policy=policy)
    target_cell_centers = np.asarray(target_fv.cell_centers, dtype=np.float64)
    target_exact = 1.0 + target_cell_centers @ np.asarray((1.0, 2.0, 3.0))
    target_error = float(
        np.max(
            np.abs(
                np.asarray(target_solved.value[: target_system.cell_count]) - target_exact
            )
        )
    )
    target_residual = float(
        jnp.max(
            jnp.abs(
                target_system.residual(
                    target_solved.value[: target_system.cell_count],
                    target_solved.value[target_system.cell_count :],
                    tensor,
                    target_boundary,
                )
            )
        )
    )
    if (
        not bool(jnp.all(target_solved.successful))
        or target_error > 1e-7
        or target_residual > 1e-7
    ):
        raise RuntimeError(
            f"Regenerated mimetic FV affine solve failed: {target_error=} {target_residual=}"
        )
    load = jnp.asarray(receipt.composition.value("inventory")) * target_fv.cell_volumes
    continued = target_system.solve(tensor, target_boundary, source=load, policy=policy)
    continued_residual = float(
        jnp.max(
            jnp.abs(
                target_system.residual(
                    continued.value[: target_system.cell_count],
                    continued.value[target_system.cell_count :],
                    tensor,
                    target_boundary,
                    source=load,
                )
            )
        )
    )
    if not bool(jnp.all(continued.successful)) or continued_residual > 1e-7:
        raise RuntimeError(f"FV continuation failed: {continued_residual=}")
    continued_change = float(
        jnp.max(
            jnp.abs(
                continued.value[: target_system.cell_count]
                - target_solved.value[: target_system.cell_count]
            )
        )
    )
    print(
        json.dumps(
            {
                "route": plan.route,
                "source_id": source.source_id,
                "source_revision": source.source_revision,
                "complex_id": complex_.complex_id,
                "topology_id": result.mesh.topology_id,
                "site_ids": site_ids,
                "represented_site_indices": sorted(set(indices.tolist())),
                "region_measures": measured_volumes,
                "family": "polyhedron",
                "cells": count,
                "affine_error": error,
                "normalized_residual": residual,
                "vem_constant_error": vem_error,
                "target_affine_error": target_error,
                "target_fv_rank_margin": rank_margin,
                "target_fv_residual": target_residual,
                "continued_field_change": continued_change,
                "vem_minimum_rank_margin": vem.evidence.minimum_rank_margin,
                "adaptation_status": adaptation.status.value,
                "target_topology_id": target.mesh.topology_id,
                "target_source_revision": target_revision,
                "common_refinement_id": common.refinement_id,
                "inventory_before": before,
                "inventory_after": after,
                "rebind_id": receipt.receipt_id,
                "continuation_residual": continued_residual,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
