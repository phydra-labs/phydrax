#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical host topology-edit record shared by the native adaptation routes.

A route describes its target mesh relative to the source in global identities only:
vertices and cells by global ID, intermediate entities (edges, faces) by their sorted
vertex-global-ID keys. `CellTopologyEdit` carries typed fixed-family or packed
polyhedral target blocks, the operation class, lineage relations, any complete vertex stencil, oriented shared-face
closure witnesses of template edits, and the reference-coordinate geometry witnesses
of nested operations. `assemble_topology_edit` owns the one conversion into a
canonical `CellMesh` with deterministic intermediate entity IDs, a complete
`MeshLineage`, and the sparse vertex interpolation stencil.
"""

from __future__ import annotations

from typing import assert_never, Literal, NamedTuple, TYPE_CHECKING, TypeAlias

import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..discretization import CellBlock, CellMesh, PolyhedralConnectivity
from ..discretization._cell_complex import (
    PolygonalConnectivity,
    polyhedral_connectivity,
    TetrahedralConnectivity,
)
from ..discretization._cell_geometry_transfer import NestedReferenceWitnesses
from ..discretization._cell_mesh import PolyhedralBlock
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._reference_cell import (
    facet_orientation_between,
    reference_cell_topology,
)
from ..typing import parse
from ._lineage import (
    EntityLineage,
    EntityLineageKind,
    MeshLineage,
    VertexInterpolationStencil,
)


if TYPE_CHECKING:
    from ..discretization._cell_geometry import CellGeometrySpec
    from ..discretization._exact_power_geometry import ExactPowerCellGeometrySource
    from ..discretization._periodic_topology import PeriodicMeshTopology
    from ..geometry._supermesh import PreparedCommonRefinement
    from ._periodic import PeriodicRefinement


TopologyEditOperation: TypeAlias = Literal[
    "nested_refinement",
    "nested_coarsening",
    "nested_adaptation",
    "local_reconnection",
    "relocation",
]
"""Operation class of one edit; it selects the geometry-transition obligation.

``nested_*`` edits keep every target cell inside one source cell or assemble it
from complete source cells and carry reference witnesses; ``local_reconnection``
creates cells spanning several source cells; ``relocation`` keeps the topology
and moves vertices.
"""


class EntityRelations(NamedTuple):
    """Explicit lineage relations of one entity dimension.

    Vertex and cell keys are ``(r, 1)`` global IDs; edge and face keys are
    ascending vertex global IDs (padded with ``-1`` to the widest face of a mixed
    mesh). A key present in both meshes also receives the identity PRESERVED
    relation unless an explicit relation already maps that source key onto the
    same target key.
    """

    dimension: int
    source_keys: np.ndarray
    target_keys: np.ndarray
    kinds: np.ndarray


class PrescribedEntityIds(NamedTuple):
    """Global IDs a route restores for new target edges or faces.

    Coarsening restores retired entities under their recorded IDs; ``keys`` are
    ascending vertex-global-ID rows that must not name source entities.
    """

    dimension: int
    keys: np.ndarray
    ids: np.ndarray


class TopologyEditBlock(NamedTuple):
    """Target cells of one typed fixed-family block.

    ``cells`` are target vertex rows in the ``cell_kind`` reference order and
    ``cell_ids`` their global IDs. ``source_kind`` is the family of the source block
    of the same name, or ``None`` for a block the edit creates.
    """

    name: str
    cell_kind: str
    source_kind: str | None
    cells: np.ndarray
    cell_ids: np.ndarray


class PolyhedralTopologyEditBlock(NamedTuple):
    """Packed outward-oriented face connectivity of a target polyhedral block.

    The connectivity's cell order matches ``cell_ids``. Vertex rows reference the
    edit coordinate array; face identities are reconstructed once by assembly.
    """

    name: str
    source_kind: str | None
    connectivity: PolyhedralConnectivity
    cell_ids: np.ndarray

    @property
    def cell_kind(self) -> str:
        return "polyhedron"


class SharedFaceWitnesses(NamedTuple):
    """Oriented closure of faces a template edit shares between target cells.

    Face ``f`` is local facet ``first_facets[f]`` of target cell ``first_cells[f]``
    and local facet ``second_facets[f]`` of ``second_cells[f]``; ``permutations[f]``
    (padded with ``-1``) is the orientation action taking the first cell's facet
    vertex order to the second's. Assembly requires every witnessed face to be
    shared by exactly these two cells with exactly this orientation.
    """

    first_cells: np.ndarray
    first_facets: np.ndarray
    second_cells: np.ndarray
    second_facets: np.ndarray
    permutations: np.ndarray


class PeriodicEntityIdentityBank(NamedTuple):
    """Canonical relative-shift entity keys and their scientific allocation history."""

    degree: int
    entity_keys: tuple[tuple[int, ...], ...]
    entity_global_ids: tuple[int, ...]
    allocator_next_id: int


class PeriodicNonnestedGeometryAuthority(NamedTuple):
    """One real complete overlap of unchanged authored periodic material domain."""

    source: CellMesh
    target: CellMesh
    source_geometry: CellGeometrySpec
    target_geometry: CellGeometrySpec
    common_refinement: PreparedCommonRefinement
    binding_id: str


class PeriodicVertexOrbitWitness(NamedTuple):
    """Explicit complete target vertex orbits bound to one scientific source epoch.

    Representatives are GLOBAL target vertex IDs, never local source/target rows.
    Quotient entity banks include live/restored/retired keyed IDs and the owning
    allocator high-water; ``history`` binds retained periodic adaptation epochs.
    ``allocation_prior`` retains an actual bound private target and its canonical
    witness when multiple staged edits allocate against one immutable source.
    """

    source_topology_id: str
    identification_id: str
    vertex_representative_ids: np.ndarray
    vertex_shifts: np.ndarray
    quotient_entities: tuple[PeriodicEntityIdentityBank, ...]
    history: PeriodicRefinement | None = None
    retained_quotient_entities: tuple[PeriodicEntityIdentityBank, ...] = ()
    allocation_prior: tuple[CellMesh, PeriodicVertexOrbitWitness] | None = None
    nonnested_geometry: PeriodicNonnestedGeometryAuthority | None = None


class CellTopologyEdit(NamedTuple):
    """Target mesh of one native topology edit, relative to its source.

    The stencil rows interpolate every target vertex from source vertex global
    IDs. ``relations`` holds one record per dimension ``0..D`` in ascending order.
    ``prescribed_entity_ids`` fixes IDs of restored intermediate entities; other
    new intermediate entities receive fresh IDs above every source and prescribed
    ID, in sorted key order. ``refinement``/``coarsening`` are the geometry
    witnesses of nested operations (required by, and only by, ``nested_*``):
    refinement witnesses have target fine cells in source coarse cells (preserved
    cells map onto themselves with the identity witness); coarsening witnesses
    have source fine cells in target coarse cells.
    """

    operation: TopologyEditOperation
    coordinates: np.ndarray
    vertex_global_ids: np.ndarray
    blocks: tuple[TopologyEditBlock | PolyhedralTopologyEditBlock, ...]
    stencil_sources: np.ndarray
    stencil_weights: np.ndarray
    stencil_valid: np.ndarray
    relations: tuple[EntityRelations, ...]
    prescribed_entity_ids: tuple[PrescribedEntityIds, ...] = ()
    shared_faces: SharedFaceWitnesses | None = None
    refinement: NestedReferenceWitnesses | None = None
    coarsening: NestedReferenceWitnesses | None = None
    periodic_orbits: PeriodicVertexOrbitWitness | None = None
    target_periodic_topology: PeriodicMeshTopology | None = None


def source_family_blocks(
    families: tuple[tuple[str, str], ...],
    block_cells: tuple[np.ndarray, ...],
    block_cell_ids: tuple[np.ndarray, ...],
    /,
) -> tuple[TopologyEditBlock, ...]:
    """Keep source families with surviving target cells; omit retired empty groups."""

    if len(block_cells) != len(families) or len(block_cell_ids) != len(families):
        raise ValueError("A family-preserving edit needs one target block per source.")
    return tuple(
        TopologyEditBlock(name, kind, kind, cells, ids)
        for (name, kind), cells, ids in zip(
            families, block_cells, block_cell_ids, strict=True
        )
        if ids.size
    )


def nested_reference_vertices(
    fine_sources: np.ndarray,
    fine_weights: np.ndarray,
    coarse_rows: np.ndarray,
    /,
) -> np.ndarray:
    """Coarse-simplex reference coordinates of fine-simplex vertices.

    ``fine_sources[c, i]``/``fine_weights[c, i]`` are the barycentric stencil of
    fine vertex ``i`` over coarse vertex identifiers (``-1`` padded) and
    ``coarse_rows[c]`` the coarse cell's vertex identifiers in local order. The
    reference simplex has vertex ``0`` at the origin and vertex ``j`` at the unit
    vector ``e_j``, so the reference coordinates are the barycentric weights of
    vertices ``1..d``. A stencil source outside its coarse cell refuses the
    witness.
    """

    sources = np.asarray(fine_sources, dtype=np.int64)
    weights = np.asarray(fine_weights, dtype=np.float64)
    rows = np.asarray(coarse_rows, dtype=np.int64)
    if sources.ndim != 3 or weights.shape != sources.shape or rows.ndim != 2:
        raise ValueError("Nested witnesses need (C, d+1, w) stencils and (C, d+1) rows.")
    if sources.shape[:2] != rows.shape:
        raise ValueError("Fine and coarse simplices must have equal vertex counts.")
    matches = (sources[..., :, None] == rows[:, None, None, :]) & (
        sources[..., :, None] >= 0
    )
    if np.any(np.any(matches, axis=-1) != (sources >= 0)):
        raise ValueError("A nested witness references a vertex outside its coarse cell.")
    barycentric = np.sum(np.where(matches, weights[..., None], 0.0), axis=2)
    return barycentric[..., 1:]


def entity_keys(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    """Keys aligned with ``mesh.entity_set(dimension)`` rows.

    Vertices and cells give ``(n, 1)`` global IDs; edges and faces give ascending
    vertex-global-ID rows, padded with ``-1`` after the IDs when faces of a mixed
    mesh have different arities.
    """

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    target = int(dimension)
    if target == 0:
        return np.asarray(mesh.vertex_global_ids, dtype=np.int64)[:, None]
    if target == mesh.topological_dimension:
        return np.asarray(mesh.entity_set(target).entity_ids, dtype=np.int64)[:, None]
    connectivity = mesh.connectivity
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if target == 1 and isinstance(
        connectivity,
        (
            PolygonalConnectivity,
            TetrahedralConnectivity,
            HexahedralConnectivity,
            PolyhedralConnectivity,
        ),
    ):
        return np.sort(vertex_ids[np.asarray(connectivity.edges, dtype=np.int64)], axis=1)
    if target == 2 and isinstance(
        connectivity, (TetrahedralConnectivity, HexahedralConnectivity)
    ):
        return np.sort(vertex_ids[np.asarray(connectivity.faces, dtype=np.int64)], axis=1)
    if target == 2 and isinstance(connectivity, PolyhedralConnectivity):
        offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        values = vertex_ids[np.asarray(connectivity.face_vertex_values, dtype=np.int64)]
        arity = np.diff(offsets)
        width = int(np.max(arity, initial=0))
        column = np.arange(values.size, dtype=np.int64) - np.repeat(offsets[:-1], arity)
        padded = np.full((arity.size, width), np.iinfo(np.int64).max, dtype=np.int64)
        padded[np.repeat(np.arange(arity.size), arity), column] = values
        ordered = np.sort(padded, axis=1)
        return np.where(ordered == np.iinfo(np.int64).max, -1, ordered)
    raise ValueError(f"Entity keys of dimension {target} are unavailable.")


def key_rows(table: np.ndarray, queries: np.ndarray, /) -> np.ndarray:
    """Row of each query key in ``table`` (unique rows), or -1 when absent."""

    table_ = np.asarray(table, dtype=np.int64)
    queries_ = np.asarray(queries, dtype=np.int64)
    if queries_.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if table_.shape[0] == 0:
        return np.full((queries_.shape[0],), -1, dtype=np.int64)
    width = table_.shape[1]
    if queries_.ndim != 2 or queries_.shape[1] != width:
        raise ValueError("Key queries must match the table key width.")
    combined = np.concatenate((table_, queries_), axis=0)
    _, inverse = np.unique(combined, axis=0, return_inverse=True)
    inverse = inverse.reshape((-1,))
    owner = np.full((int(np.max(inverse)) + 1,), -1, dtype=np.int64)
    owner[inverse[: table_.shape[0]]] = np.arange(table_.shape[0], dtype=np.int64)
    return owner[inverse[table_.shape[0] :]]


def _target_entity_ids(
    source: CellMesh,
    probe: CellMesh,
    dimension: int,
    prescribed: tuple[PrescribedEntityIds, ...],
    /,
) -> np.ndarray:
    """Preserved keys keep source IDs, restored keys their prescribed IDs, and the
    remaining new keys fresh IDs in sorted key order."""

    source_keys = entity_keys(source, dimension)
    source_ids = np.asarray(source.entity_set(dimension).entity_ids, dtype=np.int64)
    target_keys = entity_keys(probe, dimension)
    width = max(source_keys.shape[1], target_keys.shape[1])
    source_keys = np.pad(
        source_keys, ((0, 0), (0, width - source_keys.shape[1])), constant_values=-1
    )
    target_keys = np.pad(
        target_keys, ((0, 0), (0, width - target_keys.shape[1])), constant_values=-1
    )
    rows = key_rows(source_keys, target_keys)
    identifiers = np.where(rows >= 0, source_ids[np.maximum(rows, 0)], -1)
    records = tuple(value for value in prescribed if value.dimension == dimension)
    width = target_keys.shape[1]
    keys = np.concatenate(
        [np.asarray(value.keys, dtype=np.int64) for value in records]
        or [np.zeros((0, width), dtype=np.int64)]
    )
    restored = np.concatenate(
        [np.asarray(value.ids, dtype=np.int64) for value in records]
        or [np.zeros((0,), dtype=np.int64)]
    )
    if (
        keys.shape != (restored.size, width)
        or np.unique(restored).size != restored.size
        or np.any(key_rows(source_keys, keys) >= 0)
        or np.intersect1d(restored, source_ids).size
    ):
        raise ValueError("Prescribed entity IDs must name new, unique entities.")
    matched = key_rows(keys, target_keys)
    restoring = np.flatnonzero(matched >= 0)
    identifiers[restoring] = restored[matched[restoring]]
    new = np.flatnonzero(identifiers < 0)
    order = np.lexsort(target_keys[new].T[::-1])
    start = int(max(np.max(source_ids, initial=-1), np.max(restored, initial=-1))) + 1
    identifiers[new[order]] = np.arange(start, start + new.size, dtype=np.int64)
    return identifiers


def _build_mesh(
    edit: CellTopologyEdit,
    numeric_version: str,
    entity_ids: dict[int, np.ndarray] | None,
    /,
) -> CellMesh:
    blocks: list[CellBlock | PolyhedralBlock] = []
    entries: list[tuple[str, np.ndarray] | tuple[np.ndarray, ...]] = []
    for block in edit.blocks:
        order = np.argsort(block.cell_ids, kind="stable")
        ids = np.asarray(block.cell_ids, dtype=np.int64)[order]
        if isinstance(block, PolyhedralTopologyEditBlock):
            connectivity = block.connectivity
            co = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
            cf = np.asarray(connectivity.cell_face_values, dtype=np.int64)
            signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.int32)
            fo = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
            fv = np.asarray(connectivity.face_vertex_values, dtype=np.int32)
            if co.size != ids.size + 1:
                raise ValueError("Packed edit connectivity must match its cell IDs.")
            loops = []
            rows = []
            for row in order:
                faces = tuple(
                    fv[fo[cf[position]] : fo[cf[position] + 1]][:: int(signs[position])]
                    for position in range(co[row], co[row + 1])
                )
                loops.append(faces)
                rows.append(np.unique(np.concatenate(faces)))
            if len({row.size for row in rows}) != 1:
                raise ValueError("A packed edit block needs one exact vertex arity.")
            blocks.append(PolyhedralBlock(block.name, np.stack(rows), global_ids=ids))
            entries.extend(loops)
        else:
            cells = np.asarray(block.cells, dtype=np.int32)[order]
            blocks.append(CellBlock(block.name, block.cell_kind, cells, global_ids=ids))
            entries.append((block.cell_kind, cells))
    packed = (
        polyhedral_connectivity(
            tuple(entries),
            edit.coordinates.shape[0],
            vertex_global_ids=edit.vertex_global_ids,
            cell_global_ids=np.concatenate(
                [np.asarray(block.global_ids) for block in blocks]
            ),
            edge_global_ids=None if entity_ids is None else entity_ids.get(1),
            face_global_ids=None if entity_ids is None else entity_ids.get(2),
        )
        if any(isinstance(block, PolyhedralTopologyEditBlock) for block in edit.blocks)
        else None
    )
    target = CellMesh(
        edit.coordinates,
        tuple(blocks),
        vertex_global_ids=edit.vertex_global_ids,
        entity_global_ids=entity_ids,
        numeric_version=numeric_version,
        polyhedral_connectivity=packed,
    )
    if edit.target_periodic_topology is not None:
        descriptor = edit.target_periodic_topology.rebuilt(target)
        target = CellMesh(
            target.coordinates,
            target.blocks,
            vertex_global_ids=target.vertex_global_ids,
            entity_global_ids=entity_ids,
            numeric_version=numeric_version,
            polyhedral_connectivity=packed,
            periodic_topology=descriptor,
        )
    return target


def _entity_lineage(
    source: CellMesh,
    target: CellMesh,
    relations: EntityRelations,
    /,
) -> EntityLineage:
    dimension = relations.dimension
    source_keys = entity_keys(source, dimension)
    target_keys = entity_keys(target, dimension)
    source_ids = np.asarray(source.entity_set(dimension).entity_ids, dtype=np.int64)
    target_ids = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
    explicit_sources = key_rows(source_keys, relations.source_keys)
    explicit_targets = key_rows(target_keys, relations.target_keys)
    if np.any(explicit_sources < 0) or np.any(explicit_targets < 0):
        raise ValueError(
            f"Dimension-{dimension} relations reference entities outside the meshes."
        )
    related_targets = np.zeros((target_ids.size,), dtype=np.bool_)
    related_targets[explicit_targets] = True
    related_sources = np.zeros((source_ids.size,), dtype=np.bool_)
    related_sources[explicit_sources] = True
    # A key present in both meshes names the same entity; it keeps an identity
    # relation unless the route already states that identity explicitly.
    width = max(source_keys.shape[1], target_keys.shape[1])
    implicit = key_rows(
        np.pad(
            source_keys, ((0, 0), (0, width - source_keys.shape[1])), constant_values=-1
        ),
        np.pad(
            target_keys, ((0, 0), (0, width - target_keys.shape[1])), constant_values=-1
        ),
    )
    explicit_identity = np.zeros((target_ids.size,), dtype=np.bool_)
    explicit_identity[
        explicit_targets[implicit[explicit_targets] == explicit_sources]
    ] = True
    preserved = np.flatnonzero((implicit >= 0) & ~explicit_identity)
    related_sources[implicit[preserved]] = True
    related_targets[preserved] = True
    sources = np.concatenate(
        (source_ids[explicit_sources], source_ids[implicit[preserved]])
    )
    targets = np.concatenate((target_ids[explicit_targets], target_ids[preserved]))
    kinds = np.concatenate(
        (
            np.asarray(relations.kinds, dtype=np.int32),
            np.full(preserved.shape, int(EntityLineageKind.PRESERVED), dtype=np.int32),
        )
    )
    order = np.lexsort((kinds, targets, sources))
    return EntityLineage(
        dimension,
        source.entity_set(dimension).entity_set_id,
        target.entity_set(dimension).entity_set_id,
        sources[order],
        targets[order],
        kinds[order],
        created_target_ids=np.sort(target_ids[~related_targets]),
        deleted_source_ids=np.sort(source_ids[~related_sources]),
    )


def _cell_rows(mesh: CellMesh, identifiers: np.ndarray, /) -> tuple[np.ndarray, ...]:
    """Block index and block row of each cell global ID."""

    ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    blocks = np.concatenate(
        [np.full((block.cell_count,), index) for index, block in enumerate(mesh.blocks)]
    )
    rows = np.concatenate([np.arange(block.cell_count) for block in mesh.blocks])
    order = np.argsort(ids, kind="stable")
    position = np.searchsorted(ids[order], identifiers)
    position = np.minimum(position, ids.size - 1)
    if np.any(ids[order][position] != identifiers):
        raise ValueError("A witness references a cell absent from its mesh.")
    return blocks[order][position], rows[order][position]


def _facet_vertices(
    mesh: CellMesh, cell_ids: np.ndarray, facets: np.ndarray, /
) -> list[tuple[int, ...]]:
    blocks, rows = _cell_rows(mesh, cell_ids)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    dimension = mesh.topological_dimension
    if isinstance(mesh.connectivity, PolyhedralConnectivity) and any(
        mesh.blocks[int(index)].cell_kind == "polyhedron" for index in blocks
    ):
        connectivity = mesh.connectivity
        ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
        indices = {int(identifier): index for index, identifier in enumerate(ids)}
        co = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
        cf = np.asarray(connectivity.cell_face_values, dtype=np.int64)
        signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.int32)
        fo = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        fv = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
        loops = []
        for identifier, facet, block_index, block_row in zip(
            cell_ids, facets, blocks, rows, strict=True
        ):
            block = mesh.blocks[int(block_index)]
            if block.cell_kind != "polyhedron":
                local = reference_cell_topology(block.cell_kind).entities[dimension - 1]
                if not 0 <= facet < len(local):
                    raise ValueError(
                        "A shared-face witness names an undeclared local facet."
                    )
                cell = np.asarray(block.vertices[int(block_row)], dtype=np.int64)
                loops.append(
                    tuple(int(value) for value in vertex_ids[cell[list(local[facet])]])
                )
                continue
            row = indices[int(identifier)]
            if not 0 <= facet < co[row + 1] - co[row]:
                raise ValueError("A packed face witness names an undeclared facet.")
            position = co[row] + facet
            face = cf[position]
            loop = fv[fo[face] : fo[face + 1]][:: int(signs[position])]
            loops.append(tuple(int(value) for value in vertex_ids[loop]))
        return loops
    result = []
    for block_index, row, facet in zip(blocks, rows, facets, strict=True):
        block = mesh.blocks[int(block_index)]
        topology = reference_cell_topology(block.cell_kind)
        local = topology.entities[dimension - 1]
        if not 0 <= int(facet) < len(local):
            raise ValueError("A shared-face witness names an undeclared local facet.")
        cell = np.asarray(block.vertices[int(row)], dtype=np.int64)
        result.append(tuple(int(value) for value in vertex_ids[cell[list(local[facet])]]))
    return result


def _facet_counts(mesh: CellMesh, keys: np.ndarray, /) -> np.ndarray:
    """Number of cells having each ``-1``-padded sorted facet key as a facet."""

    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    dimension = mesh.topological_dimension
    width = keys.shape[1]
    if isinstance(mesh.connectivity, PolyhedralConnectivity):
        face_keys = entity_keys(mesh, 2)
        # Width may exceed that of a particular witness batch.
        query = np.full((keys.shape[0], face_keys.shape[1]), -1, dtype=np.int64)
        query[:, : keys.shape[1]] = keys
        found = key_rows(face_keys, query)
        counts = np.asarray(mesh.connectivity.face_cell_counts, dtype=np.int64)
        return np.where(found >= 0, counts[np.maximum(found, 0)], 0)
    occurrences = []
    for block in mesh.blocks:
        cells = np.asarray(block.vertices, dtype=np.int64)
        for facet in reference_cell_topology(block.cell_kind).entities[dimension - 1]:
            padded = np.full((cells.shape[0], width), -1, dtype=np.int64)
            padded[:, : len(facet)] = np.sort(vertex_ids[cells[:, list(facet)]], axis=1)
            occurrences.append(padded)
    table, counts = np.unique(np.concatenate(occurrences), axis=0, return_counts=True)
    found = key_rows(table, keys)
    return np.where(found >= 0, counts[np.maximum(found, 0)], 0)


def _require_shared_faces(target: CellMesh, witnesses: SharedFaceWitnesses, /) -> None:
    """Every witnessed face joins exactly its two cells under its orientation."""

    count = witnesses.first_cells.size
    if (
        any(
            value.shape != (count,)
            for value in (
                witnesses.first_cells,
                witnesses.first_facets,
                witnesses.second_cells,
                witnesses.second_facets,
            )
        )
        or witnesses.permutations.ndim != 2
        or witnesses.permutations.shape[0] != count
    ):
        raise ValueError("Shared-face witness arrays must have aligned face rows.")
    if count == 0:
        return
    first = _facet_vertices(target, witnesses.first_cells, witnesses.first_facets)
    second = _facet_vertices(target, witnesses.second_cells, witnesses.second_facets)
    width = max((len(value) for value in first), default=0)
    keys = np.full((len(first), width), -1, dtype=np.int64)
    for index, value in enumerate(first):
        keys[index, : len(value)] = np.sort(value)
    counts = _facet_counts(target, keys)
    for index, (left, right) in enumerate(zip(first, second, strict=True)):
        if set(left) != set(right):
            raise ValueError("A shared-face witness joins faces of different vertices.")
        action = facet_orientation_between(left, right)
        declared = tuple(
            int(value) for value in witnesses.permutations[index] if value >= 0
        )
        if action.permutation != declared:
            raise ValueError("A shared-face witness declares the wrong orientation.")
        if counts[index] != 2:
            raise ValueError("A witnessed face is not shared by exactly two cells.")


def _require_nested(
    fine: CellMesh,
    coarse: CellMesh,
    witnesses: NestedReferenceWitnesses,
    name: str,
    /,
) -> None:
    fine_ids = np.asarray(witnesses.fine_cell_ids, dtype=np.int64)
    coarse_ids = np.asarray(witnesses.coarse_cell_ids, dtype=np.int64)
    vertices = np.asarray(witnesses.fine_reference_vertices, dtype=np.float64)
    dimension = fine.topological_dimension
    if (
        fine_ids.ndim != 1
        or coarse_ids.shape != fine_ids.shape
        or vertices.ndim != 3
        or vertices.shape[0] != fine_ids.size
        or vertices.shape[2] != dimension
        or not np.all(np.isfinite(vertices))
        or np.unique(fine_ids).size != fine_ids.size
    ):
        raise ValueError(
            f"{name} witnesses must be unique finite padded reference records."
        )
    blocks, _ = _cell_rows(fine, fine_ids)
    if any(
        vertices.shape[1]
        < len(reference_cell_topology(fine.blocks[int(block)].cell_kind).vertices)
        for block in blocks
    ):
        raise ValueError("A nested witness omits fine-cell reference corners.")
    _cell_rows(coarse, coarse_ids)


def _power_domain_root(geometry: CellGeometrySpec) -> ExactPowerCellGeometrySource:
    from ..discretization._exact_power_geometry import (
        ExactPowerCellGeometryLinearActionSource,
        ExactPowerCellGeometryRestrictionSource,
        ExactPowerCellGeometrySource,
    )

    current = geometry.exact_source
    visited = set()
    while isinstance(
        current,
        (
            ExactPowerCellGeometryRestrictionSource,
            ExactPowerCellGeometryLinearActionSource,
        ),
    ):
        if id(current) in visited:
            raise ValueError("Periodic domain source ancestry contains a cycle.")
        visited.add(id(current))
        current = current.parent
    if not isinstance(current, ExactPowerCellGeometrySource):
        raise TypeError(
            "Nonnested periodic edits require actual exact power domain roots."
        )
    if current.domain_source is None or not current.periodic_constraints:
        raise ValueError(
            "A nonnested periodic source omits its original domain/control authority."
        )
    from ..discretization._coordinate_enclosure import prepared_coordinate_source_bank

    prepared_coordinate_source_bank(geometry)
    return current


def _nonnested_binding(authority: PeriodicNonnestedGeometryAuthority) -> str:
    common = authority.common_refinement
    return canonical_fingerprint(
        {
            "kind": "periodic-nonnested-geometry-authority",
            "source_mesh": authority.source.mesh_id,
            "source_numeric_version": authority.source.numeric_version,
            "source_arrays": array_tree_fingerprint(authority.source),
            "target_mesh": authority.target.mesh_id,
            "target_numeric_version": authority.target.numeric_version,
            "target_arrays": array_tree_fingerprint(authority.target),
            "source_geometry": array_tree_fingerprint(authority.source_geometry),
            "target_geometry": array_tree_fingerprint(authority.target_geometry),
            "common_refinement": common.refinement_id,
            "common_arrays": array_tree_fingerprint(common),
            "policy": common.policy.policy_id,
            "evidence": common.evidence.evidence_id,
            "status": int(common.status),
        }
    )


def require_periodic_nonnested_source(
    source: CellMesh, authority: PeriodicNonnestedGeometryAuthority, /
) -> None:
    """Bind the actual caller's source epoch, not only a reused quotient descriptor."""
    if not isinstance(authority, PeriodicNonnestedGeometryAuthority):
        raise TypeError("Nonnested source admission requires its canonical authority.")
    if (
        source.mesh_id != authority.source.mesh_id
        or source.numeric_version != authority.source.numeric_version
        or canonical_fingerprint(array_tree_fingerprint(source))
        != canonical_fingerprint(array_tree_fingerprint(authority.source))
    ):
        raise ValueError(
            "Nonnested authority belongs to a different numeric/scientific source epoch."
        )


def require_periodic_nonnested_geometry(
    source: PeriodicMeshTopology,
    target: CellMesh,
    authority: PeriodicNonnestedGeometryAuthority,
    /,
) -> None:
    from ..discretization._cell_geometry_validity import cell_geometry_id
    from ..discretization._periodic_topology import _identification_id
    from ..geometry._supermesh import CommonRefinementCoverage, CommonRefinementStatus

    if not isinstance(authority, PeriodicNonnestedGeometryAuthority):
        raise TypeError(
            "Nonnested periodic geometry requires its canonical overlap authority."
        )
    original, successor = authority.source.periodic_topology, target.periodic_topology
    if original is None or successor is None:
        raise ValueError(
            "Nonnested periodic overlap omits actual source/target descriptors."
        )
    if original.periodic_topology_id != source.periodic_topology_id:
        raise ValueError(
            "Nonnested authority belongs to a different source periodic epoch."
        )
    if _identification_id(source.cell) != _identification_id(successor.cell):
        raise ValueError("Nonnested periodic regeneration changes the authored group.")
    before, after = (
        _power_domain_root(authority.source_geometry),
        _power_domain_root(authority.target_geometry),
    )
    if before.authority_binding != after.authority_binding:
        raise ValueError(
            "Nonnested regeneration changes original domain/facets/materials/periodic controls."
        )
    original_geometry, successor_geometry = (
        original.actual_geometry,
        successor.actual_geometry,
    )
    if original_geometry is None or successor_geometry is None:
        raise ValueError(
            "Nonnested overlap requires real descriptor geometry authorities."
        )
    declared_source, declared_target = (
        authority.source_geometry.exact_source,
        authority.target_geometry.exact_source,
    )
    if (
        declared_source is None
        or declared_target is None
        or original_geometry.exact_source is None
        or successor_geometry.exact_source is None
        or original_geometry.exact_source.source_id != declared_source.source_id
        or successor_geometry.exact_source.source_id != declared_target.source_id
    ):
        raise ValueError("Nonnested overlap substitutes its descriptor source geometry.")
    common, evidence = authority.common_refinement, authority.common_refinement.evidence
    if (
        common.status is not CommonRefinementStatus.SUCCESS
        or evidence.status is not CommonRefinementStatus.SUCCESS
        or common.policy.coverage is not CommonRefinementCoverage.COMPLETE
    ):
        raise ValueError(
            "Nonnested periodic overlap is not a complete successful certificate."
        )
    if (
        authority.target.mesh_id != target.mesh_id
        or authority.target.numeric_version != target.numeric_version
        or common.source_mesh_id != authority.source.mesh_id
        or common.target_mesh_id != target.mesh_id
        or common.source_topology_id != authority.source.topology_id
        or common.target_topology_id != target.topology_id
        or common.source_geometry_id != cell_geometry_id(authority.source_geometry)
        or common.target_geometry_id != cell_geometry_id(authority.target_geometry)
    ):
        raise ValueError(
            "Nonnested overlap has stale mesh/geometry/scientific identity bindings."
        )
    if any(
        (
            evidence.source_gap_count,
            evidence.target_gap_count,
            evidence.source_double_count,
            evidence.target_double_count,
            evidence.invalid_cell_count,
            evidence.intersection_failure_count,
            evidence.uncertain_predicate_count,
        )
    ):
        raise ValueError(
            "Nonnested overlap retains unresolved coverage or predicate evidence."
        )
    if (
        evidence.candidate_pair_count > common.policy.maximum_candidate_pairs
        or evidence.accepted_pair_count > common.policy.maximum_accepted_pairs
        or evidence.retained_bytes + evidence.working_bytes
        > common.policy.maximum_memory_bytes
    ):
        raise ValueError(
            "Nonnested overlap exceeds its original admitted resource policy."
        )
    volumes = np.asarray(common.volumes)
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
        raise ValueError("Nonnested overlap contains invalid positive-measure entries.")
    for indices, measures, tolerances in (
        (
            common.source_cells,
            common.source_measures,
            evidence.source_coverage_tolerances,
        ),
        (
            common.target_cells,
            common.target_measures,
            evidence.target_coverage_tolerances,
        ),
    ):
        measured = np.asarray(measures)
        covered = np.bincount(
            np.asarray(indices), weights=volumes, minlength=measured.size
        )
        if covered.shape != measured.shape or np.any(
            np.abs(covered - measured) > np.asarray(tolerances)
        ):
            raise ValueError(
                "Nonnested overlap entries do not certify every original source/target cell."
            )
    if authority.binding_id != _nonnested_binding(authority):
        raise ValueError(
            "Nonnested authority numerical/source payload changed after preparation."
        )


def prepare_periodic_nonnested_geometry(
    source: CellMesh,
    target: CellMesh,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    common_refinement: PreparedCommonRefinement,
    /,
) -> PeriodicNonnestedGeometryAuthority:
    authority = PeriodicNonnestedGeometryAuthority(
        source, target, source_geometry, target_geometry, common_refinement, ""
    )
    authority = authority._replace(binding_id=_nonnested_binding(authority))
    if source.periodic_topology is None:
        raise ValueError("Nonnested periodic authority requires a periodic predecessor.")
    require_periodic_nonnested_geometry(source.periodic_topology, target, authority)
    return authority


def prepare_topology_edit_target(
    source: CellMesh,
    edit: CellTopologyEdit,
    /,
    *,
    numeric_version: str,
) -> CellMesh:
    """Stage canonical lifted IDs before certifying a nonnested periodic overlap."""
    probe = _build_mesh(edit, numeric_version, None)
    entity_ids = {
        degree: _target_entity_ids(source, probe, degree, edit.prescribed_entity_ids)
        for degree in range(1, source.topological_dimension)
    }
    return _build_mesh(edit, numeric_version, entity_ids or None)


def assemble_topology_edit(
    source: CellMesh,
    edit: CellTopologyEdit,
    /,
    *,
    numeric_version: str,
) -> tuple[CellMesh, MeshLineage, VertexInterpolationStencil | None]:
    """Build target and lineage; return no interpolation when target rows are unknown."""

    if not isinstance(source, CellMesh):
        raise TypeError("source must be CellMesh.")
    if not isinstance(edit, CellTopologyEdit):
        raise TypeError("edit must be CellTopologyEdit.")
    if source.periodic_topology is not None:
        if edit.periodic_orbits is None:
            raise ValueError(
                "A periodic topology edit requires a complete orbit/identity witness."
            )
    elif edit.periodic_orbits is not None:
        raise ValueError("A periodic orbit witness cannot relabel a nonperiodic source.")
    match parse(edit.operation, TopologyEditOperation, "operation"):
        case "nested_refinement" | "nested_coarsening" | "nested_adaptation":
            nested = True
        case "local_reconnection" | "relocation":
            nested = False
        case operation:
            assert_never(operation)
    if nested != (edit.refinement is not None or edit.coarsening is not None):
        raise ValueError("Nested operations, and only they, carry geometry witnesses.")
    if (
        edit.operation == "nested_refinement"
        and edit.coarsening is not None
        and np.asarray(edit.coarsening.fine_cell_ids).size
    ):
        raise ValueError("A refinement operation cannot carry coarsening witnesses.")
    if edit.operation == "nested_coarsening" and edit.coarsening is None:
        raise ValueError("A coarsening operation requires coarsening witnesses.")
    dimension = source.topological_dimension
    if tuple(value.dimension for value in edit.relations) != tuple(range(dimension + 1)):
        raise ValueError("A topology edit needs one relation record per dimension.")
    kinds = {block.name: block.cell_kind for block in source.blocks}
    for block in edit.blocks:
        if block.source_kind is not None and kinds.get(block.name) != block.source_kind:
            raise ValueError(
                f"Edit block {block.name!r} misstates its source cell family."
            )
    target = prepare_topology_edit_target(source, edit, numeric_version=numeric_version)
    if edit.periodic_orbits is not None:
        from ._periodic import bind_periodic_topology_edit

        target = bind_periodic_topology_edit(source, target, edit.periodic_orbits)
    if edit.shared_faces is not None:
        _require_shared_faces(target, edit.shared_faces)
    if edit.refinement is not None:
        _require_nested(target, source, edit.refinement, "Refinement")
    if edit.coarsening is not None:
        _require_nested(source, target, edit.coarsening, "Coarsening")
    if nested:
        refined_ids = (
            np.asarray(edit.refinement.fine_cell_ids, dtype=np.int64)
            if edit.refinement is not None
            else np.zeros(0, dtype=np.int64)
        )
        coarse_ids = (
            np.unique(np.asarray(edit.coarsening.coarse_cell_ids, dtype=np.int64))
            if edit.coarsening is not None
            else np.zeros(0, dtype=np.int64)
        )
        target_ids = np.concatenate(
            [np.asarray(block.global_ids) for block in target.blocks]
        )
        covered = np.concatenate((refined_ids, coarse_ids))
        if not np.array_equal(np.sort(covered), np.sort(target_ids)):
            raise ValueError(
                "Every target cell needs exactly one nested geometry witness."
            )
    lineage = MeshLineage(
        source.topology_id,
        target.topology_id,
        tuple(_entity_lineage(source, target, value) for value in edit.relations),
    )
    stencil = (
        VertexInterpolationStencil(
            source.entity_set(0).entity_set_id,
            target.entity_set(0).entity_set_id,
            edit.vertex_global_ids,
            edit.stencil_sources,
            edit.stencil_weights,
            edit.stencil_valid,
            preserves_constants=True,
        )
        if np.all(np.any(edit.stencil_valid, axis=1))
        else None
    )
    return target, lineage, stencil


__all__ = [
    "CellTopologyEdit",
    "EntityRelations",
    "PeriodicEntityIdentityBank",
    "PeriodicVertexOrbitWitness",
    "PrescribedEntityIds",
    "PolyhedralTopologyEditBlock",
    "SharedFaceWitnesses",
    "TopologyEditBlock",
    "TopologyEditOperation",
    "assemble_topology_edit",
    "entity_keys",
    "key_rows",
    "nested_reference_vertices",
    "source_family_blocks",
]
