#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared topology, ancestry, and failure operations for polyhedral edits."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

import numpy as np

from ..discretization._cell_complex import PolyhedralConnectivity, polyhedral_connectivity
from ..discretization._cell_mesh import CellMesh
from ._contracts import MeshingFailure, MeshingFailureCategory
from ._lineage import EntityLineageKind
from ._topology_edit import (
    CellTopologyEdit,
    EntityRelations,
    PolyhedralTopologyEditBlock,
    SharedFaceWitnesses,
    entity_keys,
)


class PolyhedralAdaptationError(MeshingFailure):
    """A polyhedral edit could not satisfy its geometric or topology contract."""

    def __init__(
        self,
        message: str,
        /,
        *,
        category: MeshingFailureCategory = MeshingFailureCategory.COMPLIANCE_FAILED,
        entity_ids: tuple[int, ...] = (),
    ) -> None:
        super().__init__(
            category,
            message,
            stage="polyhedral_adaptation",
            entity_ids=entity_ids,
        )




def polyhedral_face_loops(mesh: CellMesh, /) -> tuple[tuple[np.ndarray, ...], ...]:
    """Return outward-oriented cell loops in canonical connectivity order."""
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("mesh must have packed polyhedral connectivity.")
    cell_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.int8)
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_vertices = np.asarray(connectivity.face_vertex_values, dtype=np.int32)
    return tuple(
        tuple(
            face_vertices[face_offsets[face] : face_offsets[face + 1]][
                :: int(signs[position])
            ]
            for position in range(cell_offsets[cell], cell_offsets[cell + 1])
            for face in (int(cell_faces[position]),)
        )
        for cell in range(cell_offsets.size - 1)
    )


def polyhedral_signed_volume(
    coordinates: np.ndarray,
    faces: tuple[np.ndarray, ...],
    /,
) -> Fraction:
    """Compute the exact signed volume of an oriented piecewise-planar shell."""
    points = tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in np.asarray(coordinates, dtype=np.float64)
    )
    six_volume = Fraction(0)
    for face in faces:
        loop = tuple(int(value) for value in np.asarray(face, dtype=np.int64))
        anchor = points[loop[0]]
        for index in range(1, len(loop) - 1):
            first = points[loop[index]]
            second = points[loop[index + 1]]
            six_volume += (
                anchor[0] * (first[1] * second[2] - first[2] * second[1])
                + anchor[1] * (first[2] * second[0] - first[0] * second[2])
                + anchor[2] * (first[0] * second[1] - first[1] * second[0])
            )
    return six_volume / 6


def _shared_face_witnesses(target: CellMesh, /) -> SharedFaceWitnesses:
    connectivity = target.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("target must have packed polyhedral connectivity.")
    owner = np.asarray(connectivity.face_owner, dtype=np.int64)
    neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int64)
    shared = np.flatnonzero(neighbor >= 0)
    loops = polyhedral_face_loops(target)
    owner_local = np.asarray(connectivity.face_owner_local, dtype=np.int64)
    neighbor_local = np.asarray(connectivity.face_neighbor_local, dtype=np.int64)
    cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    first_cells = cell_ids[owner[shared]]
    second_cells = cell_ids[neighbor[shared]]
    first_facets = owner_local[shared]
    second_facets = neighbor_local[shared]
    width = max(
        (
            len(loops[int(owner[row])][int(owner_local[row])])
            for row in shared
        ),
        default=3,
    )
    permutations = np.full((shared.size, width), -1, dtype=np.int64)
    for output_row, face in enumerate(shared.tolist()):
        first = np.asarray(loops[int(owner[face])][int(owner_local[face])])
        second = np.asarray(loops[int(neighbor[face])][int(neighbor_local[face])])
        second_positions = {int(vertex): index for index, vertex in enumerate(second)}
        if set(int(value) for value in first) != set(second_positions):
            raise ValueError("Canonical shared-face incidence lost its vertex signature.")
        permutations[output_row, : first.size] = np.asarray(
            [second_positions[int(vertex)] for vertex in first], dtype=np.int64
        )
    return SharedFaceWitnesses(
        first_cells,
        first_facets,
        second_cells,
        second_facets,
        permutations,
    )


def _vertex_stencil(
    source: CellMesh,
    target: CellMesh,
    supports: Mapping[int, Mapping[int, Fraction]] | None,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[Fraction, ...], ...]]:
    source_ids = set(int(value) for value in np.asarray(source.vertex_global_ids))
    target_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    rows: list[tuple[tuple[int, Fraction], ...]] = []
    for identifier in target_ids.tolist():
        if supports is None:
            rows.append(())
            continue
        row = supports.get(int(identifier))
        if row is None and int(identifier) in source_ids:
            row = {int(identifier): Fraction(1)}
        active = () if row is None else tuple(
            sorted(
                (int(source_id), coefficient)
                for source_id, coefficient in row.items()
                if coefficient != 0
            )
        )
        if active and sum((value for _, value in active), Fraction(0)) != 1:
            raise ValueError("Polyhedral vertex ancestry must preserve constants exactly.")
        rows.append(active)
    width = max(1, max((len(row) for row in rows), default=0))
    source_rows = np.full((len(rows), width), -1, dtype=np.int64)
    weights = np.zeros((len(rows), width), dtype=np.float64)
    valid = np.zeros((len(rows), width), dtype=np.bool_)
    exact = [[Fraction(0) for _ in range(width)] for _ in rows]
    for row_index, row in enumerate(rows):
        for column, (identifier, coefficient) in enumerate(row):
            source_rows[row_index, column] = identifier
            weights[row_index, column] = float(coefficient)
            valid[row_index, column] = True
            exact[row_index][column] = coefficient
    return source_rows, weights, valid, tuple(tuple(row) for row in exact)


def _relation_records(
    source: CellMesh,
    target: CellMesh,
    supports: Mapping[int, Mapping[int, Fraction]] | None,
    cell_relations: tuple[tuple[int, int, EntityLineageKind], ...],
    /,
) -> tuple[EntityRelations, ...]:
    source_cells = entity_keys(source, source.topological_dimension)[:, 0]
    target_cells = entity_keys(target, target.topological_dimension)[:, 0]
    source_cell_rows = {int(identifier): row for row, identifier in enumerate(source_cells)}
    target_cell_rows = {int(identifier): row for row, identifier in enumerate(target_cells)}
    top_sources: list[np.ndarray] = []
    top_targets: list[np.ndarray] = []
    top_kinds: list[int] = []
    for source_id, target_id, kind in cell_relations:
        if source_id not in source_cell_rows or target_id not in target_cell_rows:
            raise ValueError("Cell ancestry must name present source and target cells.")
        top_sources.append(np.asarray((source_id,), dtype=np.int64))
        top_targets.append(np.asarray((target_id,), dtype=np.int64))
        top_kinds.append(int(kind))
    top_order = sorted(
        range(len(top_kinds)),
        key=lambda index: (
            int(top_sources[index][0]),
            int(top_targets[index][0]),
            top_kinds[index],
        ),
    )
    top = EntityRelations(
        source.topological_dimension,
        np.asarray([top_sources[index] for index in top_order], dtype=np.int64).reshape(
            (-1, 1)
        ),
        np.asarray([top_targets[index] for index in top_order], dtype=np.int64).reshape(
            (-1, 1)
        ),
        np.asarray([top_kinds[index] for index in top_order], dtype=np.int32),
    )
    records: list[EntityRelations] = []
    for dimension in range(source.topological_dimension + 1):
        if dimension == source.topological_dimension:
            records.append(top)
            continue
        source_keys = entity_keys(source, dimension)
        target_keys = entity_keys(target, dimension)
        parent_rows: list[np.ndarray] = []
        child_rows: list[np.ndarray] = []
        for target_key in target_keys:
            child_vertices = tuple(int(value) for value in target_key if value >= 0)
            if not child_vertices:
                continue
            if supports is None:
                continue
            ancestry = set().union(
                *(
                    set(supports.get(vertex, {vertex}))
                    for vertex in child_vertices
                )
            )
            if not ancestry:
                continue
            child_identity = set(child_vertices)
            for source_key in source_keys:
                parent_vertices = tuple(int(value) for value in source_key if value >= 0)
                parent_identity = set(parent_vertices)
                if ancestry.issubset(parent_identity) and child_identity != parent_identity:
                    parent_rows.append(source_key)
                    child_rows.append(target_key)
        ordering = sorted(
            range(len(parent_rows)),
            key=lambda index: (
                tuple(int(value) for value in parent_rows[index]),
                tuple(int(value) for value in child_rows[index]),
            ),
        )
        records.append(
            EntityRelations(
                dimension,
                np.asarray([parent_rows[index] for index in ordering], dtype=np.int64).reshape(
                    (-1, source_keys.shape[1])
                ),
                np.asarray([child_rows[index] for index in ordering], dtype=np.int64).reshape(
                    (-1, target_keys.shape[1])
                ),
                np.full(len(ordering), int(EntityLineageKind.REFINED_FROM), dtype=np.int32),
            )
        )
    return tuple(records)


def build_polyhedral_topology_edit(
    source: CellMesh,
    target: CellMesh,
    supports: Mapping[int, Mapping[int, Fraction]] | None,
    cell_relations: tuple[tuple[int, int, EntityLineageKind], ...],
    /,
) -> CellTopologyEdit:
    """Build one complete edit record from validated source and target shells."""
    source_connectivity = source.connectivity
    target_connectivity = target.connectivity
    if not isinstance(source_connectivity, PolyhedralConnectivity) or not isinstance(
        target_connectivity, PolyhedralConnectivity
    ):
        raise TypeError("Polyhedral topology edits require polyhedral source and target meshes.")
    target_loops = polyhedral_face_loops(target)
    target_cell_ids = np.asarray(target_connectivity.cell_global_ids, dtype=np.int64)
    rows_by_id = {
        int(identifier): row for row, identifier in enumerate(target_cell_ids.tolist())
    }
    blocks: list[PolyhedralTopologyEditBlock] = []
    for block in target.blocks:
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        loops = tuple(target_loops[rows_by_id[int(identifier)]] for identifier in identifiers)
        connectivity = polyhedral_connectivity(
            loops,
            target.coordinates.shape[0],
            vertex_global_ids=np.asarray(target.vertex_global_ids, dtype=np.int64),
            cell_global_ids=identifiers,
        )
        blocks.append(
            PolyhedralTopologyEditBlock(block.name, None, connectivity, identifiers)
        )
    sources, weights, valid, exact = _vertex_stencil(source, target, supports)
    relations = _relation_records(source, target, supports, cell_relations)
    shared = _shared_face_witnesses(target)
    edit = CellTopologyEdit(
        np.asarray(target.coordinates, dtype=np.float64),
        np.asarray(target.vertex_global_ids, dtype=np.int64),
        tuple(blocks),
        sources,
        weights,
        valid,
        relations,
        shared_faces=shared,
        stencil_exact_weights=exact,
    )
    return edit


__all__ = [
    "PolyhedralAdaptationError",
    "build_polyhedral_topology_edit",
    "polyhedral_face_loops",
    "polyhedral_signed_volume",
]
