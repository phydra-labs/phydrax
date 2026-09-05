#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .._physical import SpatialCoordinateContract
from ..discretization import CellMesh
from ..discretization._reference_cell import reference_cell_topology
from ..discretization.fem import (
    FiniteElementHPLineage,
    FiniteElementHPTopology,
    FiniteElementTransferBundle,
    refine_triangles_local,
)
from ._canonical import certify_cell_mesh
from ._lineage import (
    CellMeshTransition,
    EntityLineage,
    EntityLineageKind,
    MeshLineage,
    MeshTransitionKind,
    VertexInterpolationStencil,
)


def _preserved_intermediate_entities(
    source: CellMesh,
    target: CellMesh,
    dimension: int,
    /,
) -> EntityLineage:
    source_entities = source.entity_set(dimension)
    target_entities = target.entity_set(dimension)
    source_ids = np.asarray(source_entities.entity_ids, dtype=np.int64)
    target_ids = np.asarray(target_entities.entity_ids, dtype=np.int64)
    preserved = np.intersect1d(source_ids, target_ids)
    created = np.setdiff1d(target_ids, preserved)
    deleted = np.setdiff1d(source_ids, preserved)
    return EntityLineage(
        dimension,
        source_entities.entity_set_id,
        target_entities.entity_set_id,
        preserved,
        preserved,
        np.full(preserved.shape, int(EntityLineageKind.PRESERVED), dtype=np.int32),
        created_target_ids=created,
        deleted_source_ids=deleted,
    )


def refine_triangle_mesh(
    mesh: CellMesh,
    marked_cell_ids: ArrayLike,
    coordinate_contract: SpatialCoordinateContract,
    /,
    *,
    numeric_version: str = "locally-refined",
) -> tuple[CellMeshTransition, FiniteElementTransferBundle]:
    """Prepare, certify, and describe one conforming local triangle refinement."""

    refined, adaptation, transfer = refine_triangles_local(
        mesh,
        np.asarray(marked_cell_ids),
        numeric_version=numeric_version,
    )
    target = certify_cell_mesh(refined, coordinate_contract)
    refined = target.mesh
    source_vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    target_vertices = np.asarray(refined.vertex_global_ids, dtype=np.int64)
    source_vertex_set = mesh.entity_set(0)
    target_vertex_set = refined.entity_set(0)
    source_rows = np.zeros((target_vertices.size, 2), dtype=np.int64)
    weights = np.zeros((target_vertices.size, 2), dtype=float)
    valid = np.zeros((target_vertices.size, 2), dtype=bool)
    source_set = set(int(value) for value in source_vertices)
    midpoint_parent_by_id = {
        int(midpoint): tuple(int(value) for value in parents)
        for midpoint, parents in zip(
            np.asarray(adaptation.midpoint_vertex_ids),
            np.asarray(adaptation.midpoint_parent_vertex_ids),
            strict=True,
        )
    }
    vertex_sources = []
    vertex_targets = []
    vertex_kinds = []
    for row, target_id in enumerate(target_vertices):
        identifier = int(target_id)
        if identifier in source_set:
            source_rows[row, 0] = identifier
            weights[row, 0] = 1.0
            valid[row, 0] = True
            vertex_sources.append(identifier)
            vertex_targets.append(identifier)
            vertex_kinds.append(int(EntityLineageKind.PRESERVED))
        else:
            parents = midpoint_parent_by_id[identifier]
            source_rows[row] = parents
            weights[row] = 0.5
            valid[row] = True
            vertex_sources.extend(parents)
            vertex_targets.extend((identifier, identifier))
            vertex_kinds.extend(
                (int(EntityLineageKind.SPLIT_FROM), int(EntityLineageKind.SPLIT_FROM))
            )
    vertex_lineage = EntityLineage(
        0,
        source_vertex_set.entity_set_id,
        target_vertex_set.entity_set_id,
        np.asarray(vertex_sources, dtype=np.int64),
        np.asarray(vertex_targets, dtype=np.int64),
        np.asarray(vertex_kinds, dtype=np.int32),
    )
    stencil = VertexInterpolationStencil(
        source_vertex_set.entity_set_id,
        target_vertex_set.entity_set_id,
        target_vertices,
        source_rows,
        weights,
        valid,
        preserves_constants=True,
    )

    source_cells = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    target_cells = np.asarray(refined.blocks[0].global_ids, dtype=np.int64)
    refined_parents = {
        int(parent): tuple(int(value) for value in children[valid_row])
        for parent, children, valid_row in zip(
            np.asarray(adaptation.parent_cell_ids),
            np.asarray(adaptation.child_cell_ids),
            np.asarray(adaptation.child_valid),
            strict=True,
        )
    }
    cell_sources = []
    cell_targets = []
    cell_kinds = []
    for source_id in source_cells:
        identifier = int(source_id)
        if identifier in refined_parents:
            children = refined_parents[identifier]
            cell_sources.extend((identifier,) * len(children))
            cell_targets.extend(children)
            cell_kinds.extend((int(EntityLineageKind.REFINED_FROM),) * len(children))
        elif identifier in set(int(value) for value in target_cells):
            cell_sources.append(identifier)
            cell_targets.append(identifier)
            cell_kinds.append(int(EntityLineageKind.PRESERVED))
    cell_lineage = EntityLineage(
        2,
        mesh.entity_set(2).entity_set_id,
        refined.entity_set(2).entity_set_id,
        np.asarray(cell_sources, dtype=np.int64),
        np.asarray(cell_targets, dtype=np.int64),
        np.asarray(cell_kinds, dtype=np.int32),
    )
    edge_lineage = _preserved_intermediate_entities(mesh, refined, 1)
    lineage = MeshLineage(
        mesh.topology_id,
        refined.topology_id,
        (vertex_lineage, edge_lineage, cell_lineage),
    )
    transition = CellMeshTransition(
        mesh.mesh_id,
        mesh.topology_id,
        target,
        lineage,
        MeshTransitionKind.REFINE,
        vertex_stencil=stencil,
    )
    return transition, transfer


def project_hp_lineage(
    source: FiniteElementHPTopology,
    target: FiniteElementHPTopology,
    lineage: FiniteElementHPLineage,
    /,
) -> MeshLineage:
    """Project fixed-capacity hp slot lineage into stable global-ID evidence."""

    if not isinstance(source, FiniteElementHPTopology) or not isinstance(
        target, FiniteElementHPTopology
    ):
        raise TypeError("source and target must be FiniteElementHPTopology.")
    if not isinstance(lineage, FiniteElementHPLineage):
        raise TypeError("lineage must be FiniteElementHPLineage.")
    if (
        lineage.source_topology_id != source.topology_id
        or lineage.target_topology_id != target.topology_id
    ):
        raise ValueError("hp lineage endpoints do not match supplied topologies.")
    valid = np.asarray(lineage.valid, dtype=bool)
    source_slots = np.asarray(lineage.source_slots, dtype=np.int32)[valid]
    target_slots = np.asarray(lineage.target_slots, dtype=np.int32)[valid]
    source_ids = np.asarray(source.cell_global_ids, dtype=np.int64)[source_slots]
    target_ids = np.asarray(target.cell_global_ids, dtype=np.int64)[target_slots]
    kinds = np.full(source_ids.shape, int(EntityLineageKind.PRESERVED), dtype=np.int32)
    refinement = np.asarray(lineage.relation_mask("refinement"), dtype=bool)[valid]
    coarsening = np.asarray(lineage.relation_mask("coarsening"), dtype=bool)[valid]
    kinds[refinement] = int(EntityLineageKind.REFINED_FROM)
    kinds[coarsening] = int(EntityLineageKind.COARSENED_INTO)
    dimension = reference_cell_topology(source.cell_kind).dimension
    entities = EntityLineage(
        dimension,
        f"{source.topology_id}:cells",
        f"{target.topology_id}:cells",
        source_ids,
        target_ids,
        kinds,
    )
    return MeshLineage(source.topology_id, target.topology_id, (entities,))


__all__ = ["project_hp_lineage", "refine_triangle_mesh"]
