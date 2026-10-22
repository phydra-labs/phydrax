#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Assemble host topology-edit records for native adaptation routes.

A route describes its target mesh relative to the source in global identities:
vertices and cells by global ID, intermediate entities by sorted
vertex-global-ID keys. ``CellTopologyEdit`` carries fixed-family or packed
polyhedral blocks, lineage relations, vertex stencils (optionally with their rational
homogeneous bank), oriented shared-face closure, and reference-coordinate geometry
witnesses of nested operations. ``assemble_topology_edit`` builds a ``CellMesh``
with deterministic intermediate entity IDs, ``MeshLineage``, and a sparse vertex
interpolation stencil.
"""

from __future__ import annotations

from contextlib import nullcontext
from fractions import Fraction
from typing import assert_never, Literal, NamedTuple, TYPE_CHECKING

import numpy as np

import phydrax.discretization._cell_complex as cell_complex_owner
import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.discretization._reference_cell_types as reference_cell_types_owner
import phydrax.meshing._lineage_records as lineage_owner
from phydrax.discretization._periodic_exact import identification_id
from phydrax.discretization._reference_cell_types import PolyhedralBlock

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..discretization._cell_complex import (
    polyhedral_connectivity,
    PolyhedralConnectivity,
)
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_transition_types import (
    NestedReferenceWitnesses,
    reference_homogeneous_preparation_bounds,
    require_reference_homogeneous,
)
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._coordinate_enclosure import prepared_coordinate_source_bank
from ..discretization._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from ..discretization._periodic_topology import PeriodicMeshTopology
from ..discretization._reference_cell import (
    facet_orientation_between,
    reference_cell_topology,
)
from ..geometry._supermesh import (
    CommonRefinementCoverage,
    CommonRefinementStatus,
    PreparedCommonRefinement,
)
from ..typing import checked, parse
from ._entity_keys import entity_keys, key_rows


if TYPE_CHECKING:
    from ._periodic import PeriodicRefinement


type TopologyEditOperation = Literal[
    "nested_refinement",
    "nested_coarsening",
    "nested_adaptation",
    "local_reconnection",
    "relocation",
]
"""Operation class of one edit; it selects the geometry-transition obligation.

``nested_*`` edits keep every target cell inside one source cell or assemble it
from whole source cells and carry reference witnesses; ``local_reconnection``
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
    connectivity: cell_complex_owner.PolyhedralConnectivity
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
    """Relative-shift entity keys and their scientific allocation history."""

    degree: int
    entity_keys: tuple[tuple[int, ...], ...]
    entity_global_ids: tuple[int, ...]
    allocator_next_id: int


class PeriodicNonnestedGeometryAuthority(NamedTuple):
    """Overlap coverage of an unchanged authored periodic material domain."""

    source: cell_mesh_owner.CellMesh
    target: cell_mesh_owner.CellMesh
    source_geometry: CellGeometrySpec
    target_geometry: CellGeometrySpec
    common_refinement: PreparedCommonRefinement
    binding_id: str


class PeriodicVertexOrbitWitness(NamedTuple):
    """Target vertex orbits bound to one scientific source epoch.

    Representatives are global target vertex IDs, never local source/target rows.
    Quotient entity banks include live/restored/retired keyed IDs and allocator
    high-water; ``history`` binds retained periodic adaptation epochs.
    ``allocation_prior`` retains a bound staged target and its orbit correspondence
    when several edits allocate against one immutable source.
    """

    source_topology_id: str
    identification_id: str
    vertex_representative_ids: np.ndarray
    vertex_shifts: np.ndarray
    quotient_entities: tuple[PeriodicEntityIdentityBank, ...]
    history: PeriodicRefinement | None = None
    retained_quotient_entities: tuple[PeriodicEntityIdentityBank, ...] = ()
    allocation_prior: (
        tuple[cell_mesh_owner.CellMesh, PeriodicVertexOrbitWitness] | None
    ) = None
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
    cells map onto themselves with the identity correspondence); coarsening witnesses
    have source fine cells in target coarse cells.

    ``stencil_homogeneous`` optionally retains exact source-owned stencil rows on
    the ``stencil_valid`` support: finite sign-bit-free binary64 homogeneous
    weights (``+0.0`` elsewhere) with a positive exact sum in every row, whose
    rational barycentrics ``h_j / sum(h)`` the ``stencil_weights`` round to nearest
    bitwise. Refinement witnesses of such an edit retain the exact reference bank
    ``exact_nested_reference_vertices`` derives from these rows.
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
    stencil_homogeneous: np.ndarray | None = None


def source_family_blocks(
    families: tuple[tuple[str, str], ...],
    block_cells: tuple[np.ndarray, ...],
    block_cell_ids: tuple[np.ndarray, ...],
    /,
) -> tuple[TopologyEditBlock, ...]:
    """Keep source families with surviving target cells; omit retired empty groups."""

    if len(block_cells) != len(families) or len(block_cell_ids) != len(families):
        raise ValueError(
            "CellTopologyEdit.blocks must supply one target block per source family; got misaligned family arrays."
        )
    return tuple(
        TopologyEditBlock(name, kind, kind, cells, ids)
        for (name, kind), cells, ids in zip(
            families, block_cells, block_cell_ids, strict=True
        )
        if ids.size
    )


def _coarse_corner_matches(
    fine_sources: np.ndarray, coarse_rows: np.ndarray, /
) -> np.ndarray:
    """Match ``(C, d+1, w)`` fine stencil supports to ``(C, d+1)`` coarse corners."""

    if fine_sources.ndim != 3 or coarse_rows.ndim != 2:
        raise ValueError(
            "CellTopologyEdit.stencils must have (C, d+1, w) stencils and (C, d+1) rows; got incompatible shapes."
        )
    if fine_sources.shape[:2] != coarse_rows.shape:
        raise ValueError(
            "CellTopologyEdit.stencils must give fine and coarse simplices equal vertex counts; got incompatible counts."
        )
    matches = (fine_sources[..., :, None] == coarse_rows[:, None, None, :]) & (
        fine_sources[..., :, None] >= 0
    )
    if np.any(np.any(matches, axis=-1) != (fine_sources >= 0)):
        raise ValueError(
            "CellTopologyEdit.stencils must reference vertices inside their coarse cell; got an outside vertex."
        )
    return matches


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
    reference-coordinate correspondence.
    """

    sources = np.asarray(fine_sources, dtype=np.int64)
    weights = np.asarray(fine_weights, dtype=np.float64)
    if weights.shape != sources.shape:
        raise ValueError(
            "CellTopologyEdit.stencils must have (C, d+1, w) stencils and (C, d+1) rows; got incompatible shapes."
        )
    matches = _coarse_corner_matches(sources, np.asarray(coarse_rows, dtype=np.int64))
    barycentric = np.sum(np.where(matches, weights[..., None], 0.0), axis=2)
    return barycentric[..., 1:]


_BINARY64_MAX = Fraction(float(np.finfo(np.float64).max))


def _nested_array_bytes(cells: int, corners: int, width: int, /) -> int:
    # Corner matches with their comparison temporaries, contributions, positive
    # masks and rounded contributions, plus per-corner counts and banks.
    return 24 * cells * corners * width * corners + 32 * cells * corners * corners


def _corner_lookup_bytes(cells: int, corners: int, width: int, /) -> int:
    # Fine and coarse corner IDs with sorted lookup positions, plus the gathered
    # stencil sources, masks and homogeneous rows of every fine corner.
    return cells * corners * (48 + 32 * width)


def _aggregation_bounds(pairs: int, corners: int, width: int, /) -> tuple[int, int]:
    # Per aggregated fine corner and coarse corner: ``width`` conversions and
    # sums, four binary64-exactness operations (bound, round, convert, compare),
    # the total, one quotient and two roundings; two fine corners' Fractions are
    # live at once.
    return pairs * corners * (2 * width + 8), lineage_owner.exact_fraction_bytes(
        2 * (2 * corners + width + 1), max(width, corners)
    )


def _exact_nested_reference_bounds(
    cells: int, corners: int, width: int, /
) -> tuple[int, int]:
    """Upper bound ``exact_nested_reference_vertices`` when every corner aggregates."""
    rounding_work, rounding_bytes = lineage_owner.homogeneous_rounding_bounds(
        cells * corners, width
    )
    aggregation_work, aggregation_bytes = _aggregation_bounds(
        cells * corners, corners, width
    )
    return (
        rounding_work + aggregation_work,
        rounding_bytes + aggregation_bytes + _nested_array_bytes(cells, corners, width),
    )


def exact_nested_reference_vertices(
    fine_sources: np.ndarray,
    fine_homogeneous: np.ndarray,
    coarse_rows: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact coarse-simplex reference banks of fine-simplex vertices.

    ``fine_homogeneous[c, i]`` is the exact homogeneous stencil (as in
    ``CellTopologyEdit.stencil_homogeneous``) of fine vertex ``i`` over
    ``fine_sources[c, i]``, on the support ``fine_sources >= 0``. The bank
    ``H[c, i, j]`` sums the weights naming coarse corner ``j`` and must be exactly representable in
    binary64; the exact reference coordinate ``j`` is ``H[c, i, j] / sum(H[c, i])``
    in rational arithmetic. Returns the correctly rounded reference vertices
    (coordinates ``1..d``, as ``nested_reference_vertices``) and ``H``.

    A ``budget`` admits the vectorized banks and the stencil-rounding bound before
    validation and charges the rational work of every aggregated fine corner
    before aggregating it; ``None`` declares unbounded host validation.
    """

    sources = np.asarray(fine_sources, dtype=np.int64)
    weights = np.asarray(fine_homogeneous, dtype=np.float64)
    if weights.shape != sources.shape or sources.ndim != 3:
        raise ValueError(
            "CellTopologyEdit.stencil_homogeneous must align (C, d+1, w) weights with their stencil sources; got incompatible shapes."
        )
    cells, corners, width = sources.shape
    with nullcontext() if budget is None else budget.temporary_scope():
        if budget is not None:
            rounding_work, rounding_bytes = lineage_owner.homogeneous_rounding_bounds(
                cells * corners, width
            )
            array_bytes = _nested_array_bytes(cells, corners, width)
            budget.admit_work_bound(
                rounding_work, storage_upper=rounding_bytes + array_bytes
            )
            budget.reserve(0, array_bytes)
        rounded = lineage_owner.rounded_homogeneous_barycentrics(
            weights, sources >= 0, budget=budget
        )
        matches = _coarse_corner_matches(sources, np.asarray(coarse_rows, dtype=np.int64))
        contributions = np.where(matches, weights[..., None], 0.0)
        homogeneous = np.sum(contributions, axis=2)
        # With at most one positive weight per corner, every corner sum is exactly representable and
        # its rounded coordinate is the rounded stencil weight naming that corner.
        reference = np.sum(np.where(matches, rounded[..., None], 0.0), axis=2)
        aggregated = np.argwhere(
            np.any(np.count_nonzero(contributions > 0.0, axis=2) > 1, axis=2)
        )
        if budget is not None and aggregated.shape[0]:
            budget.reserve(*_aggregation_bounds(aggregated.shape[0], corners, width))
        for cell, vertex in aggregated.tolist():
            exact = [
                sum((Fraction(float(value)) for value in column), Fraction(0))
                for column in np.asarray(contributions[cell, vertex]).T.tolist()
            ]
            if any(value > _BINARY64_MAX for value in exact) or any(
                Fraction(float(value)) != value for value in exact
            ):
                raise ValueError(
                    "CellTopologyEdit.stencil_homogeneous must aggregate each coarse corner exactly in binary64; got an inexact corner sum."
                )
            total = sum(exact, Fraction(0))
            homogeneous[cell, vertex] = [float(value) for value in exact]
            reference[cell, vertex] = [float(value / total) for value in exact]
    return reference[..., 1:], homogeneous


def topology_edit_exact_preparation_bounds(edit: CellTopologyEdit, /) -> tuple[int, int]:
    """Upper bound the exact-bank work and bytes ``assemble_topology_edit`` charges.

    Zero without exact banks. Otherwise the shape-only bound covers the stencil
    rounding, the refinement conversion of every witnessed fine corner with its
    corner lookups, and coarsening reference-bank validation.
    """
    work = storage = 0
    exact = edit.stencil_homogeneous
    refinement = edit.refinement
    coarsening = edit.coarsening
    coarse_bank = None if coarsening is None else coarsening.fine_reference_homogeneous
    if (
        (exact is not None and np.ndim(exact) != 2)
        or (
            exact is not None
            and refinement is not None
            and np.ndim(refinement.fine_reference_vertices) != 3
        )
        or (coarse_bank is not None and np.ndim(coarse_bank) != 3)
    ):
        raise ValueError(
            "CellTopologyEdit.exact_banks must have (V, w) stencils and (C, corners, d) reference rows; got incompatible ranks."
        )
    if exact is not None:
        rows, width = np.shape(exact)
        stencil_work, stencil_bytes = lineage_owner.homogeneous_rounding_bounds(
            rows, width
        )
        work += stencil_work
        storage += stencil_bytes
        if refinement is not None:
            cells = np.size(refinement.fine_cell_ids)
            corners = np.shape(refinement.fine_reference_vertices)[2] + 1
            nested_work, nested_bytes = _exact_nested_reference_bounds(
                cells, corners, width
            )
            work += nested_work
            storage += nested_bytes + _corner_lookup_bytes(cells, corners, width)
    if coarse_bank is not None:
        cells, padded, homogeneous_width = np.shape(coarse_bank)
        coarse_work, coarse_bytes = reference_homogeneous_preparation_bounds(
            cells * padded, homogeneous_width
        )
        work += coarse_work
        storage += coarse_bytes
    return work, storage


def _target_entity_ids(
    source: cell_mesh_owner.CellMesh,
    probe: cell_mesh_owner.CellMesh,
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
        raise ValueError(
            "CellTopologyEdit.prescribed_entity_ids must name new, unique entities; got reused IDs or incompatible keys."
        )
    matched = key_rows(keys, target_keys)
    restoring = np.flatnonzero(matched >= 0)
    identifiers[restoring] = restored[matched[restoring]]
    new = np.flatnonzero(identifiers < 0)
    order = np.lexsort(target_keys[new].T[::-1])
    start = int(max(np.max(source_ids, initial=-1), np.max(restored, initial=-1))) + 1
    identifiers[new[order]] = np.arange(start, start + new.size, dtype=np.int64)
    return identifiers


def _lifted_mesh(
    edit: CellTopologyEdit,
    numeric_version: str,
    entity_ids: dict[int, np.ndarray] | None,
    /,
) -> tuple[cell_mesh_owner.CellMesh, PolyhedralConnectivity | None]:
    """Build the edit's nonperiodic lifted carrier and its packed connectivity."""
    blocks: list[reference_cell_types_owner.CellBlock | PolyhedralBlock] = []
    entries: list[tuple[str, np.ndarray] | tuple[np.ndarray, ...]] = []
    for block in edit.blocks:
        order = np.argsort(block.cell_ids, kind="stable")
        ids = np.asarray(block.cell_ids, dtype=np.int64)[order]
        match block:
            case PolyhedralTopologyEditBlock():
                connectivity = block.connectivity
                co = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
                cf = np.asarray(connectivity.cell_face_values, dtype=np.int64)
                signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.int32)
                fo = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
                fv = np.asarray(connectivity.face_vertex_values, dtype=np.int32)
                if co.size != ids.size + 1:
                    raise ValueError(
                        "CellTopologyEdit.connectivity must match its cell IDs; got incompatible cell offsets."
                    )
                loops = []
                rows = []
                for row in order:
                    faces = tuple(
                        fv[fo[cf[position]] : fo[cf[position] + 1]][
                            :: int(signs[position])
                        ]
                        for position in range(co[row], co[row + 1])
                    )
                    loops.append(faces)
                    rows.append(np.unique(np.concatenate(faces)))
                if len({row.size for row in rows}) != 1:
                    raise ValueError(
                        "CellTopologyEdit.blocks must have one vertex arity per packed block; got differing vertex counts."
                    )
                blocks.append(PolyhedralBlock(block.name, np.stack(rows), global_ids=ids))
                entries.extend(loops)
            case TopologyEditBlock():
                cells = np.asarray(block.cells, dtype=np.int32)[order]
                blocks.append(
                    reference_cell_types_owner.CellBlock(
                        block.name, block.cell_kind, cells, global_ids=ids
                    )
                )
                entries.append((block.cell_kind, cells))
            case _:
                raise TypeError(
                    f"CellTopologyEdit.blocks must contain typed topology edit blocks; got {type(block).__name__}."
                )
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
    target = cell_mesh_owner.CellMesh(
        edit.coordinates,
        tuple(blocks),
        vertex_global_ids=edit.vertex_global_ids,
        entity_global_ids=entity_ids,
        numeric_version=numeric_version,
        polyhedral_connectivity=packed,
    )
    return target, packed


@checked
def build_mesh(
    edit: CellTopologyEdit,
    numeric_version: str,
    entity_ids: dict[int, np.ndarray] | None,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> cell_mesh_owner.CellMesh:
    """Build the edit target, rebuilding and proving its periodic quotient once.

    A periodic target is a deterministic function of its lifted carrier, numeric
    version and authored descriptor; one earned in this request is reused.
    """
    target, packed = _lifted_mesh(edit, numeric_version, entity_ids)
    periodic = edit.target_periodic_topology
    if periodic is None:
        return target
    key = (
        target.mesh_id, str(numeric_version),
        canonical_fingerprint(array_tree_fingerprint(periodic)),
    )
    earned = budget.periodic_lifted_mesh_cache.get(key)
    if earned is not None:
        return earned
    descriptor = periodic.rebuilt(target, budget=budget)
    target = cell_mesh_owner.CellMesh(
        target.coordinates,
        target.blocks,
        vertex_global_ids=target.vertex_global_ids,
        entity_global_ids=entity_ids,
        numeric_version=numeric_version,
        polyhedral_connectivity=packed,
        periodic_topology=descriptor,
        budget=budget,
    )
    budget.retain_owner(target)
    budget.periodic_lifted_mesh_cache[key] = target
    return target


@checked
def entity_lineage(
    source: cell_mesh_owner.CellMesh,
    target: cell_mesh_owner.CellMesh,
    relations: EntityRelations,
    /,
) -> lineage_owner.EntityLineage:
    dimension = relations.dimension
    source_keys = entity_keys(source, dimension)
    target_keys = entity_keys(target, dimension)
    source_ids = np.asarray(source.entity_set(dimension).entity_ids, dtype=np.int64)
    target_ids = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
    explicit_sources = key_rows(source_keys, relations.source_keys)
    explicit_targets = key_rows(target_keys, relations.target_keys)
    if np.any(explicit_sources < 0) or np.any(explicit_targets < 0):
        raise ValueError(
            f"CellTopologyEdit.relations must reference entities in the meshes; got outside entities in dimension {dimension}."
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
            np.full(
                preserved.shape,
                int(lineage_owner.EntityLineageKind.PRESERVED),
                dtype=np.int32,
            ),
        )
    )
    order = np.lexsort((kinds, targets, sources))
    return lineage_owner.EntityLineage(
        dimension,
        source.entity_set(dimension).entity_set_id,
        target.entity_set(dimension).entity_set_id,
        sources[order],
        targets[order],
        kinds[order],
        created_target_ids=np.sort(target_ids[~related_targets]),
        deleted_source_ids=np.sort(source_ids[~related_sources]),
    )


def _cell_rows(
    mesh: cell_mesh_owner.CellMesh, identifiers: np.ndarray, /
) -> tuple[np.ndarray, ...]:
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
        raise ValueError(
            "CellTopologyEdit.witnesses must reference cells in their mesh; got an absent cell ID."
        )
    return blocks[order][position], rows[order][position]


def _host_block_vertices(
    mesh: cell_mesh_owner.CellMesh,
    block_index: int,
    host_vertices: dict[int, np.ndarray],
    /,
) -> np.ndarray:
    """Transfer each used vertex bank once during a shared-face validation scope."""
    vertices = host_vertices.get(block_index)
    if vertices is None:
        vertices = np.asarray(mesh.blocks[block_index].vertices, dtype=np.int64)
        host_vertices[block_index] = vertices
    return vertices


def _facet_vertices(
    mesh: cell_mesh_owner.CellMesh,
    cell_ids: np.ndarray,
    facets: np.ndarray,
    host_vertices: dict[int, np.ndarray],
    /,
) -> list[tuple[int, ...]]:
    blocks, rows = _cell_rows(mesh, cell_ids)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    dimension = mesh.topological_dimension
    if isinstance(mesh.connectivity, cell_complex_owner.PolyhedralConnectivity) and any(
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
                        "CellTopologyEdit.shared_faces must name declared local facets; got an out-of-range local facet."
                    )
                cell = _host_block_vertices(mesh, int(block_index), host_vertices)[
                    int(block_row)
                ]
                loops.append(
                    tuple(int(value) for value in vertex_ids[cell[list(local[facet])]])
                )
                continue
            row = indices[int(identifier)]
            if not 0 <= facet < co[row + 1] - co[row]:
                raise ValueError(
                    "CellTopologyEdit.shared_faces must name declared packed facets; got an out-of-range facet."
                )
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
            raise ValueError(
                "CellTopologyEdit.shared_faces must name declared local facets; got an out-of-range local facet."
            )
        cell = _host_block_vertices(mesh, int(block_index), host_vertices)[int(row)]
        result.append(tuple(int(value) for value in vertex_ids[cell[list(local[facet])]]))
    return result


def _facet_counts(mesh: cell_mesh_owner.CellMesh, keys: np.ndarray, /) -> np.ndarray:
    """Number of cells having each ``-1``-padded sorted facet key as a facet."""

    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    dimension = mesh.topological_dimension
    width = keys.shape[1]
    if isinstance(mesh.connectivity, cell_complex_owner.PolyhedralConnectivity):
        face_keys = entity_keys(mesh, 2)
        # Width may exceed that of a particular reference-coordinate batch.
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


def _require_shared_faces(
    target: cell_mesh_owner.CellMesh, witnesses: SharedFaceWitnesses, /
) -> None:
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
        raise ValueError(
            "CellTopologyEdit.shared_faces must have aligned face rows; got incompatible correspondence arrays."
        )
    if count == 0:
        return
    host_vertices: dict[int, np.ndarray] = {}
    first = _facet_vertices(
        target, witnesses.first_cells, witnesses.first_facets, host_vertices
    )
    second = _facet_vertices(
        target, witnesses.second_cells, witnesses.second_facets, host_vertices
    )
    width = max((len(value) for value in first), default=0)
    keys = np.full((len(first), width), -1, dtype=np.int64)
    for index, value in enumerate(first):
        keys[index, : len(value)] = np.sort(value)
    counts = _facet_counts(target, keys)
    for index, (left, right) in enumerate(zip(first, second, strict=True)):
        if set(left) != set(right):
            raise ValueError(
                "CellTopologyEdit.shared_faces must join faces of the same vertices; got different vertex sets."
            )
        action = facet_orientation_between(left, right)
        declared = tuple(
            int(value) for value in witnesses.permutations[index] if value >= 0
        )
        if action.permutation != declared:
            raise ValueError(
                "CellTopologyEdit.shared_faces must declare the facet orientation action; got the wrong orientation."
            )
        if counts[index] != 2:
            raise ValueError(
                "CellTopologyEdit.shared_faces must join exactly two cells; got a differing face incidence count."
            )


def _require_nested(
    fine: cell_mesh_owner.CellMesh,
    coarse: cell_mesh_owner.CellMesh,
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
            f"CellTopologyEdit.{name} must have unique finite padded reference records; got invalid correspondence arrays."
        )
    blocks, _ = _cell_rows(fine, fine_ids)
    if any(
        vertices.shape[1]
        < len(reference_cell_topology(fine.blocks[int(block)].cell_kind).vertices)
        for block in blocks
    ):
        raise ValueError(
            "CellTopologyEdit.witnesses must include fine-cell reference corners; got omitted corners."
        )
    _cell_rows(coarse, coarse_ids)


def _simplex_vertex_ids(
    mesh: cell_mesh_owner.CellMesh, blocks: np.ndarray, rows: np.ndarray, /
) -> np.ndarray:
    """Vertex global IDs, in local order, of simplex cells named by block and row."""

    dimension = mesh.topological_dimension
    identifiers = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    result = np.empty((rows.size, dimension + 1), dtype=np.int64)
    for index in np.unique(blocks).tolist():
        block = mesh.blocks[index]
        cells = np.asarray(block.vertices, dtype=np.int64)
        if block.cell_kind == "polyhedron" or cells.shape[1] != dimension + 1:
            raise ValueError(
                f"CellTopologyEdit.stencil_homogeneous must nest simplex cells; got {block.cell_kind!r} cells."
            )
        selected = blocks == index
        result[selected] = identifiers[cells[rows[selected]]]
    return result


def _bitwise_equal(stored: object, expected: np.ndarray, /) -> bool:
    value = np.ascontiguousarray(stored, dtype=np.float64)
    return value.shape == expected.shape and np.array_equal(
        value.view(np.uint64), np.ascontiguousarray(expected).view(np.uint64)
    )


def _require_exact_refinement(
    source: cell_mesh_owner.CellMesh,
    target: cell_mesh_owner.CellMesh,
    edit: CellTopologyEdit,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Bind refinement reference banks to the rational stencils of the fine corners."""

    witnesses = edit.refinement
    bank = None if witnesses is None else witnesses.fine_reference_homogeneous
    if edit.stencil_homogeneous is None:
        if bank is not None:
            raise ValueError(
                "CellTopologyEdit.refinement must derive exact reference banks from stencil_homogeneous; got a rational bank without rational stencils."
            )
        return
    exact = np.asarray(edit.stencil_homogeneous, dtype=np.float64)
    if exact.ndim != 2 or exact.shape != np.shape(edit.stencil_sources):
        raise ValueError(
            f"CellTopologyEdit.stencil_homogeneous must align with stencil_sources; got shapes {exact.shape}/{np.shape(edit.stencil_sources)}."
        )
    if witnesses is None:
        return
    if bank is None:
        raise ValueError(
            "CellTopologyEdit.refinement must retain the exact reference bank of exact stencils; got rounded witnesses only."
        )
    fine_ids = np.asarray(witnesses.fine_cell_ids, dtype=np.int64)
    with budget.temporary_scope():
        budget.reserve(
            0,
            _corner_lookup_bytes(
                fine_ids.size, target.topological_dimension + 1, exact.shape[-1]
            ),
        )
        fine = _simplex_vertex_ids(target, *_cell_rows(target, fine_ids))
        coarse = _simplex_vertex_ids(
            source,
            *_cell_rows(source, np.asarray(witnesses.coarse_cell_ids, dtype=np.int64)),
        )
        identifiers = np.asarray(edit.vertex_global_ids, dtype=np.int64)
        order = np.argsort(identifiers, kind="stable")
        position = np.minimum(np.searchsorted(identifiers[order], fine), order.size - 1)
        if np.any(identifiers[order][position] != fine):
            raise ValueError(
                "CellTopologyEdit.refinement must reference stenciled target vertices; got an unstenciled corner."
            )
        rows = order[position]
        valid = np.asarray(edit.stencil_valid, dtype=np.bool_)[rows]
        vertices, homogeneous = exact_nested_reference_vertices(
            np.where(valid, np.asarray(edit.stencil_sources, dtype=np.int64)[rows], -1),
            exact[rows],
            coarse,
            budget=budget,
        )
        if not (
            _bitwise_equal(witnesses.fine_reference_vertices, vertices)
            and _bitwise_equal(bank, homogeneous)
        ):
            raise ValueError(
                "CellTopologyEdit.refinement must be the rational reference conversion of the fine-corner stencils; got different reference banks."
            )


def power_domain_source(geometry: CellGeometrySpec, /) -> ExactPowerCellGeometrySource:
    """Return the authored exact-power root retaining the PLC and periodic controls."""
    current = geometry.exact_source
    visited = set()
    while True:
        match current:
            case (
                ExactPowerCellGeometryRestrictionSource()
                | ExactPowerCellGeometryLinearActionSource()
            ):
                if id(current) in visited:
                    raise ValueError(
                        "CellTopologyEdit.geometry_source must have acyclic ancestry; "
                        "got a cycle."
                    )
                visited.add(id(current))
                current = current.parent
            case ExactPowerCellGeometrySource():
                break
            case _:
                raise TypeError(
                    "CellTopologyEdit.geometry_source must have "
                    "ExactPowerCellGeometrySource domain roots; got another source kind."
                )
    if current.domain_source is None or not current.periodic_constraints:
        raise ValueError(
            "CellTopologyEdit.geometry_source must retain domain source and periodic constraints; got missing domain/control data."
        )
    return current


@checked
def power_domain_root(
    geometry: CellGeometrySpec, *, budget: CoordinateEnclosureBudget
) -> ExactPowerCellGeometrySource:
    """Return the authored root after preparing ``geometry``'s exact coordinates."""
    root = power_domain_source(geometry)
    prepared_coordinate_source_bank(geometry, budget=budget)
    return root


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


@checked
def require_periodic_nonnested_source(
    source: cell_mesh_owner.CellMesh, authority: PeriodicNonnestedGeometryAuthority, /
) -> None:
    """Bind a source mesh's numeric and scientific epoch to the overlap."""
    if (
        source.mesh_id != authority.source.mesh_id
        or source.numeric_version != authority.source.numeric_version
        or canonical_fingerprint(array_tree_fingerprint(source))
        != canonical_fingerprint(array_tree_fingerprint(authority.source))
    ):
        raise ValueError(
            "CellTopologyEdit.nonnested_geometry must bind the source numeric/scientific epoch; got a different epoch."
        )


@checked
def require_periodic_nonnested_geometry(
    source: PeriodicMeshTopology,
    target: cell_mesh_owner.CellMesh,
    authority: PeriodicNonnestedGeometryAuthority,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:

    original, successor = authority.source.periodic_topology, target.periodic_topology
    if original is None or successor is None:
        raise ValueError(
            "CellTopologyEdit.nonnested_geometry must have source/target periodic descriptors; got missing descriptors."
        )
    if original.periodic_topology_id != source.periodic_topology_id:
        raise ValueError(
            "CellTopologyEdit.nonnested_geometry must bind the source periodic epoch; got a different epoch."
        )
    if identification_id(source.cell) != identification_id(successor.cell):
        raise ValueError(
            "CellTopologyEdit.nonnested_geometry must preserve the authored group; got a changed identification."
        )
    before, after = (
        power_domain_root(authority.source_geometry, budget=budget),
        power_domain_root(authority.target_geometry, budget=budget),
    )
    if before.authority_binding != after.authority_binding:
        raise ValueError(
            "CellTopologyEdit.nonnested_geometry must preserve domain/facets/materials/periodic controls; got changed source bindings."
        )
    original_geometry, successor_geometry = (
        original.actual_geometry,
        successor.actual_geometry,
    )
    if original_geometry is None or successor_geometry is None:
        raise ValueError(
            "CellTopologyEdit.nonnested_geometry must include descriptor geometry; got missing source or target geometry."
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
        raise ValueError(
            "CellTopologyEdit.nonnested_geometry must bind the descriptor source geometry; got substituted geometry."
        )
    common, evidence = authority.common_refinement, authority.common_refinement.evidence
    if (
        common.status is not CommonRefinementStatus.SUCCESS
        or evidence.status is not CommonRefinementStatus.SUCCESS
        or common.policy.coverage is not CommonRefinementCoverage.COMPLETE
    ):
        raise ValueError(
            "CellTopologyEdit.nonnested_geometry must certify successful COMPLETE overlap coverage; got failed or incomplete coverage."
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
            "CellTopologyEdit.nonnested_geometry must bind current mesh/geometry/scientific identities; got stale bindings."
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
            "CellTopologyEdit.nonnested_geometry must resolve coverage and predicates; got unresolved evidence."
        )
    if (
        evidence.candidate_pair_count > common.policy.maximum_candidate_pairs
        or evidence.accepted_pair_count > common.policy.maximum_accepted_pairs
        or evidence.retained_bytes + evidence.working_bytes
        > common.policy.maximum_memory_bytes
    ):
        raise ValueError(
            "CellTopologyEdit.nonnested_geometry must obey its admitted resource policy; got exceeded limits."
        )
    volumes = np.asarray(common.volumes)
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
        raise ValueError(
            "CellTopologyEdit.nonnested_geometry must contain finite positive-measure entries; got invalid overlap volumes."
        )
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
                "CellTopologyEdit.nonnested_geometry must certify every source/target cell; got coverage outside its declared tolerances."
            )
    if authority.binding_id != _nonnested_binding(authority):
        raise ValueError(
            "CellTopologyEdit.nonnested_geometry must retain its prepared numerical/source payload; got a changed binding ID."
        )


@checked
def prepare_periodic_nonnested_geometry(
    source: cell_mesh_owner.CellMesh,
    target: cell_mesh_owner.CellMesh,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    common_refinement: PreparedCommonRefinement,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> PeriodicNonnestedGeometryAuthority:
    authority = PeriodicNonnestedGeometryAuthority(
        source, target, source_geometry, target_geometry, common_refinement, ""
    )
    authority = authority._replace(binding_id=_nonnested_binding(authority))
    if source.periodic_topology is None:
        raise ValueError(
            "CellTopologyEdit.nonnested_geometry must have a periodic predecessor; got a nonperiodic source."
        )
    require_periodic_nonnested_geometry(
        source.periodic_topology, target, authority, budget=budget
    )
    return authority


@checked
def prepare_topology_edit_target(
    source: cell_mesh_owner.CellMesh,
    edit: CellTopologyEdit,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    numeric_version: str,
) -> cell_mesh_owner.CellMesh:
    """Stage lifted IDs before certifying a nonnested periodic overlap.

    Entity keys depend only on the lifted carrier, so the ID probe skips the
    periodic quotient; the published target rebuilds and proves it once.
    """
    probe, _ = _lifted_mesh(edit, numeric_version, None)
    entity_ids = {
        degree: _target_entity_ids(source, probe, degree, edit.prescribed_entity_ids)
        for degree in range(1, source.topological_dimension)
    }
    return build_mesh(edit, numeric_version, entity_ids or None, budget=budget)


@checked
def assemble_topology_edit(
    source: cell_mesh_owner.CellMesh,
    edit: CellTopologyEdit,
    /,
    *,
    numeric_version: str,
    budget: CoordinateEnclosureBudget,
) -> tuple[
    cell_mesh_owner.CellMesh,
    lineage_owner.MeshLineage,
    lineage_owner.VertexInterpolationStencil | None,
]:
    """Build target and lineage; return no interpolation when target rows are unknown.

    Exact-bank validation charges ``budget`` before each rational phase, within
    ``topology_edit_exact_preparation_bounds``.
    """

    if source.periodic_topology is not None:
        if edit.periodic_orbits is None:
            raise ValueError(
                "CellTopologyEdit.periodic_orbits must supply orbit and identity correspondences for a periodic source; got missing orbits."
            )
    elif edit.periodic_orbits is not None:
        raise ValueError(
            "CellTopologyEdit.periodic_orbits must be absent for a nonperiodic source; got an orbit correspondence."
        )
    match parse(edit.operation, TopologyEditOperation, "operation"):
        case "nested_refinement" | "nested_coarsening" | "nested_adaptation":
            nested = True
        case "local_reconnection" | "relocation":
            nested = False
        case operation:
            assert_never(operation)
    if nested != (edit.refinement is not None or edit.coarsening is not None):
        raise ValueError(
            "CellTopologyEdit.operation must carry geometry correspondences only for nested operations; got a correspondence/operation mismatch."
        )
    if (
        edit.operation == "nested_refinement"
        and edit.coarsening is not None
        and np.asarray(edit.coarsening.fine_cell_ids).size
    ):
        raise ValueError(
            "CellTopologyEdit.coarsening must be empty for a refinement operation; got coarsening witnesses."
        )
    if edit.operation == "nested_coarsening" and edit.coarsening is None:
        raise ValueError(
            "CellTopologyEdit.coarsening must be supplied for a coarsening operation; got missing witnesses."
        )
    dimension = source.topological_dimension
    if tuple(value.dimension for value in edit.relations) != tuple(range(dimension + 1)):
        raise ValueError(
            "CellTopologyEdit.relations must supply one record per dimension in ascending order; got incompatible dimensions."
        )
    kinds = {block.name: block.cell_kind for block in source.blocks}
    for block in edit.blocks:
        if block.source_kind is not None and kinds.get(block.name) != block.source_kind:
            raise ValueError(
                f"CellTopologyEdit.blocks must state their source cell family; got a mismatched family in block {block.name!r}."
            )
    target = prepare_topology_edit_target(
        source, edit, numeric_version=numeric_version, budget=budget
    )
    if edit.periodic_orbits is not None:
        # _topology_edit ↔ _periodic: bind after the edit and orbit owners initialize.
        from ._periodic import bind_periodic_topology_edit

        target = bind_periodic_topology_edit(
            source, target, edit.periodic_orbits, budget=budget
        )
    if edit.shared_faces is not None:
        _require_shared_faces(target, edit.shared_faces)
    if edit.refinement is not None:
        _require_nested(target, source, edit.refinement, "Refinement")
    _require_exact_refinement(source, target, edit, budget=budget)
    if edit.coarsening is not None:
        _require_nested(source, target, edit.coarsening, "Coarsening")
        require_reference_homogeneous(
            np.asarray(edit.coarsening.fine_reference_vertices, dtype=np.float64),
            edit.coarsening.fine_reference_homogeneous,
            "CellTopologyEdit.coarsening",
            budget=budget,
        )
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
                "CellTopologyEdit.witnesses must supply one nested geometry correspondence per target cell; got differing cell coverage."
            )
    lineage = lineage_owner.MeshLineage(
        source.topology_id,
        target.topology_id,
        tuple(entity_lineage(source, target, value) for value in edit.relations),
    )
    stencil = (
        lineage_owner.VertexInterpolationStencil(
            source.entity_set(0).entity_set_id,
            target.entity_set(0).entity_set_id,
            edit.vertex_global_ids,
            edit.stencil_sources,
            edit.stencil_weights,
            edit.stencil_valid,
            preserves_constants=True,
            homogeneous_weights=edit.stencil_homogeneous,
            budget=budget,
        )
        # A rational bank admits only full rows; the stencil refuses others.
        if edit.stencil_homogeneous is not None
        or np.all(np.any(edit.stencil_valid, axis=1))
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
    "exact_nested_reference_vertices",
    "nested_reference_vertices",
    "source_family_blocks",
    "topology_edit_exact_preparation_bounds",
]
