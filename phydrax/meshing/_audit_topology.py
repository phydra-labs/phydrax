#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host topology audit of the geometrically welded cell complex.

CellMesh construction already rejects combinatorial duplicates, facets with
more than two cells, and inconsistent facet orientation. Those invariants are
re-examined here after welding vertices that coincide within tolerance, which
exposes cracks, stacked cells, pinched vertices, and folded facets hidden behind
duplicated vertex rows. Vertex and edge manifoldness (star connectivity) are not
construction invariants and are always evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from typing import TYPE_CHECKING

import numpy as np
from jax.typing import ArrayLike
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import phydrax.discretization._cell_complex as cell_complex_owner
import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._cell_mesh as cell_mesh_owner

from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import (
    orient2d,
    orient3d_host,
    PredicateMode,
    PredicateResult,
    resolve_host_predicate_mode,
    segment_intersections_2d,
    SegmentIntersectionStatus,
)
from ..discretization._cell_geometry_validity import polyhedral_star_tables
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._periodic_topology import lifted_loops, PeriodicMeshTopology
from ..discretization._reference_cell import reference_cell_topology
from ..discretization.spatial._morton import morton_encode_integer_host
from ..geometry._polygon_triangulation import polygon_loop_triangles
from ..typing import checked
from ._entity_keys import entity_keys, key_rows


if TYPE_CHECKING:
    from ..discretization._cell_geometry import CellGeometrySpec
    from ..discretization._cell_geometry_validity import CellValidityCertificate
    from ..geometry._embedding_state import MeshCertificateLimits
    from ..geometry._global_embedding import GlobalEmbeddingCertificate
    from ._audit import CellMeshAuditPolicy


_MORTON_BITS = {1: 62, 2: 31, 3: 21}


@dataclass(frozen=True)
class WeldedTopologyEvidence:
    """Counts of the welded-topology checks plus unresolved dispositions."""

    coincident_vertices: int
    collapsed_cells: int
    duplicate_cells: int
    nonmanifold_facets: int
    nonmanifold_edges: int
    nonmanifold_vertices: int
    inconsistent_facets: int
    open_facets: int
    self_intersections: int
    unresolved: tuple[str, ...]


# Coincident vertices ------------------------------------------------------------


def _coincident_representatives(
    points: np.ndarray, tolerance: float, capacity: int, /
) -> tuple[np.ndarray, int, bool]:
    """Weld vertices closer than ``tolerance`` through a Morton-sorted grid.

    Grid cells are at least ``tolerance`` wide, so every close pair lies in the
    same or an adjacent cell; candidates come from binary searches of the
    neighbor codes in the sorted code array. Returns the smallest vertex index of
    every welded class, the welded-vertex count, and whether the candidate
    capacity was exceeded (then no welding is applied).
    """

    count, dimension = points.shape
    identity = np.arange(count)
    if count == 0:
        return identity, 0, False
    lower = np.min(points, axis=0)
    extent = float(np.max(np.max(points, axis=0) - lower))
    bits = _MORTON_BITS[dimension]
    width = max(tolerance, extent / float(1 << bits), np.finfo(np.float64).tiny)
    cells = np.minimum(
        np.floor((points - lower) / width).astype(np.int64), (1 << bits) - 2
    )
    codes = morton_encode_integer_host(cells, bits)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    firsts = []
    seconds = []
    total = 0
    for offset in np.ndindex(*(3,) * dimension):
        neighbor = cells + np.asarray(offset) - 1
        inside = np.all(neighbor >= 0, axis=1)
        neighbor_codes = morton_encode_integer_host(np.maximum(neighbor, 0), bits)
        start = np.searchsorted(sorted_codes, neighbor_codes, side="left")
        stop = np.searchsorted(sorted_codes, neighbor_codes, side="right")
        spans = np.where(inside, stop - start, 0)
        total += int(np.sum(spans))
        if total > capacity:
            return identity, 0, True
        source = np.repeat(identity, spans)
        rank = np.arange(source.size) - np.repeat(np.cumsum(spans) - spans, spans)
        target = order[np.repeat(start, spans) + rank]
        keep = source < target
        firsts.append(source[keep])
        seconds.append(target[keep])
    first = np.concatenate(firsts)
    second = np.concatenate(seconds)
    close = np.linalg.norm(points[first] - points[second], axis=1) <= tolerance
    first = first[close]
    second = second[close]
    if first.size == 0:
        return identity, 0, False
    graph = coo_matrix(
        (np.ones(first.size, dtype=np.int8), (first, second)), shape=(count, count)
    )
    _, labels = connected_components(graph, directed=False)
    representative = np.full(labels.max() + 1, count, dtype=np.int64)
    np.minimum.at(representative, labels, identity)
    welded = representative[labels]
    return welded, int(np.count_nonzero(np.bincount(labels)[labels] > 1)), False


# Welded cells and facets -----------------------------------------------------------


def _padded(rows: list[np.ndarray], width: int, /) -> np.ndarray:
    result = np.full((len(rows), width), -1, dtype=np.int64)
    for index, row in enumerate(rows):
        result[index, : row.size] = row
    return result


def _csr_rows(offsets: np.ndarray, values: np.ndarray, /) -> np.ndarray:
    sizes = np.diff(offsets)
    width = int(np.max(sizes))
    rows = np.repeat(np.arange(sizes.size), sizes)
    rank = np.arange(values.size) - offsets[rows]
    result = np.full((sizes.size, width), -1, dtype=np.int64)
    result[rows, rank] = values
    return result


@dataclass(frozen=True)
class _Complex:
    cells: np.ndarray
    facets: np.ndarray
    facet_cells: np.ndarray
    facet_signs: np.ndarray
    edges: np.ndarray
    edge_cells: np.ndarray


def _welded_complex(mesh: cell_mesh_owner.CellMesh, welded: np.ndarray, /) -> _Complex:
    """Cells, oriented facet occurrences, and (3-D) cell edges in welded ids."""

    connectivity = mesh.connectivity
    dimension = mesh.topological_dimension
    if not mesh.blocks:
        if mesh.storage is None or (
            mesh.storage.global_entity_counts[dimension] <= 0
            or welded.shape != (0,)
            or any(entities.count != 0 for entities in mesh.topology.entity_sets)
        ):
            raise ValueError(
                "mesh.blocks must accompany zero resident topology and distributed storage when empty; "
                f"got resident counts={tuple(entities.count for entities in mesh.topology.entity_sets)!r}, "
                f"storage={type(mesh.storage).__name__}, welded.shape={welded.shape!r}"
            )
        return _Complex(
            np.empty((0, 0), dtype=np.int64),
            np.empty((0, dimension), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int8),
            np.empty((0, 2), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    cell_rows = []
    facet_rows = []
    facet_cells = []
    facet_signs = []
    edge_rows = []
    edge_cells = []
    cursor = 0
    tables = (
        polyhedral_star_tables(connectivity)
        if isinstance(connectivity, cell_complex_owner.PolyhedralConnectivity)
        else None
    )
    for block in mesh.blocks:
        count = block.cell_count
        owners = np.arange(cursor, cursor + count)
        if block.cell_kind == "polyhedron":
            # ty: ignore[unresolved-attribute]
            selected = np.isin(tables.star_cell, owners)
            vertices = _csr_rows(
                # ty: ignore[unresolved-attribute]
                np.concatenate(((0,), np.cumsum(tables.cell_vertex_counts))),
                # ty: ignore[unresolved-attribute]
                tables.cell_vertex_values,
            )[owners]
            cell_rows.append(np.where(vertices >= 0, welded[np.maximum(vertices, 0)], -1))
            # ty: ignore[unresolved-attribute]
            faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
            # ty: ignore[unresolved-attribute]
            face_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
            # ty: ignore[unresolved-attribute]
            signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.float64)
            incidence_cell = np.repeat(
                np.arange(face_offsets.size - 1), np.diff(face_offsets)
            )
            chosen = np.isin(incidence_cell, owners)
            loops = _csr_rows(
                # ty: ignore[unresolved-attribute]
                np.asarray(connectivity.face_vertex_offsets, dtype=np.int64),
                # ty: ignore[unresolved-attribute]
                np.asarray(connectivity.face_vertex_values, dtype=np.int64),
            )[faces[chosen]]
            lengths = np.sum(loops >= 0, axis=1)
            reverse = signs[chosen] < 0.0
            reversed_loops = loops.copy()
            for length in np.unique(lengths[reverse]):
                rows = reverse & (lengths == length)
                reversed_loops[rows, :length] = loops[rows, :length][:, ::-1]
            facet_rows.append(
                np.where(reversed_loops >= 0, welded[np.maximum(reversed_loops, 0)], -1)
            )
            facet_cells.append(incidence_cell[chosen])
            facet_signs.append(np.zeros(np.count_nonzero(chosen), dtype=np.int8))
            edge_rows.append(
                welded[
                    np.stack(
                        # ty: ignore[unresolved-attribute]
                        (tables.star_first[selected], tables.star_second[selected]),
                        1,
                    )
                ]
            )
            # ty: ignore[unresolved-attribute]
            edge_cells.append(tables.star_cell[selected])
        else:
            rows = welded[np.asarray(block.vertices, dtype=np.int64)]
            cell_rows.append(rows)
            if dimension == 1:
                local_facets = ((0,), (1,))
                local_signs = (-1, 1)
            elif dimension == 2:
                arity = rows.shape[1]
                local_facets = tuple(
                    (index, (index + 1) % arity) for index in range(arity)
                )
                local_signs = (0,) * arity
            else:
                topology = reference_cell_topology(block.cell_kind)
                local_facets = tuple(topology.entities[2])
                local_signs = (0,) * len(local_facets)
                local_edges = np.asarray(topology.entities[1])
                edge_rows.append(rows[:, local_edges].reshape(-1, 2))
                edge_cells.append(np.repeat(owners, local_edges.shape[0]))
            if dimension == 2 and block.cell_kind == "triangle":
                # Match mesh_facets' cell-major polygon_edges occurrence order.
                facet_rows.append(rows[:, np.asarray(local_facets)].reshape(-1, 2))
                facet_cells.append(np.repeat(owners, 3))
                facet_signs.append(np.zeros(3 * count, dtype=np.int8))
            else:
                width = max(len(face) for face in local_facets)
                for face, sign in zip(local_facets, local_signs, strict=True):
                    values = np.full((count, width), -1, dtype=np.int64)
                    values[:, : len(face)] = rows[:, np.asarray(face)]
                    facet_rows.append(values)
                    facet_cells.append(owners)
                    facet_signs.append(np.full(count, sign, dtype=np.int8))
        cursor += count
    width = max(value.shape[1] for value in facet_rows)
    facets = np.concatenate(
        [
            np.pad(value, ((0, 0), (0, width - value.shape[1])), constant_values=-1)
            for value in facet_rows
        ]
    )
    cell_width = max(value.shape[1] for value in cell_rows)
    cells = np.concatenate(
        [
            np.pad(value, ((0, 0), (0, cell_width - value.shape[1])), constant_values=-1)
            for value in cell_rows
        ]
    )
    return _Complex(
        cells,
        facets,
        np.concatenate(facet_cells),
        np.concatenate(facet_signs),
        np.concatenate(edge_rows) if edge_rows else np.empty((0, 2), dtype=np.int64),
        np.concatenate(edge_cells) if edge_cells else np.empty((0,), dtype=np.int64),
    )


def _sorted_keys(rows: np.ndarray, /) -> np.ndarray:
    return np.sort(np.where(rows < 0, np.iinfo(np.int64).max, rows), axis=1)


def _collapsed(rows: np.ndarray, /) -> np.ndarray:
    keys = _sorted_keys(rows)
    active = keys[:, 1:] != np.iinfo(np.int64).max
    return np.any(active & (keys[:, 1:] == keys[:, :-1]), axis=1)


def _duplicate_count(rows: np.ndarray, /) -> int:
    _, counts = np.unique(_sorted_keys(rows), axis=0, return_counts=True)
    return int(np.sum(counts[counts > 1] - 1))


def _loop_neighbors(rows: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Successor and predecessor of the smallest vertex of every padded loop."""

    lengths = np.sum(rows >= 0, axis=1)
    position = np.argmin(np.where(rows >= 0, rows, np.iinfo(np.int64).max), axis=1)
    index = np.arange(rows.shape[0])
    successor = rows[index, (position + 1) % lengths]
    predecessor = rows[index, (position - 1) % lengths]
    return successor, predecessor


def _star_components(
    entity_rows: np.ndarray,
    entity_cells: np.ndarray,
    links: tuple[np.ndarray, np.ndarray, np.ndarray],
    complete: np.ndarray | None = None,
    /,
) -> int:
    """Count entities whose incident cells split into several link components.

    ``links`` holds ``(entity keys, first cell, second cell)``: the two cells
    share a facet containing that entity. The neighborhood mask marks owner-local
    cells whose every facet is matched or physical boundary. An entity is judged
    only when all its resident incident cells satisfy that condition: then walking across
    facets through it cannot leave the resident set, so its resident star is
    its whole star. A star truncated at an artificial halo boundary carries no
    manifold evidence either way.
    """

    if entity_rows.shape[0] == 0:
        return 0
    nodes = np.unique(
        np.concatenate((entity_rows, entity_cells[:, None]), axis=1), axis=0
    )
    keys, first_cells, second_cells = links
    first = _lookup(nodes, np.concatenate((keys, first_cells[:, None]), axis=1))
    second = _lookup(nodes, np.concatenate((keys, second_cells[:, None]), axis=1))
    graph = coo_matrix(
        (np.ones(first.size, dtype=np.int8), (first, second)),
        shape=(nodes.shape[0], nodes.shape[0]),
    )
    _, labels = connected_components(graph, directed=False)
    entities, entity_index = np.unique(nodes[:, :-1], axis=0, return_inverse=True)
    pairs = np.unique(np.stack((entity_index.reshape(-1), labels), axis=1), axis=0)
    components = np.bincount(pairs[:, 0], minlength=entities.shape[0])
    split = components > 1
    if complete is not None:
        truncated = np.zeros(entities.shape[0], dtype=np.bool_)
        np.logical_or.at(truncated, entity_index.reshape(-1), ~complete[nodes[:, -1]])
        split &= ~truncated
    return int(np.count_nonzero(split))


def _lookup(table: np.ndarray, rows: np.ndarray, /) -> np.ndarray:
    """Row positions of ``rows`` inside the lexicographically sorted ``table``."""

    view = (
        np.ascontiguousarray(table)
        .view(np.dtype((np.void, table.dtype.itemsize * table.shape[1])))
        .reshape(-1)
    )
    query = np.ascontiguousarray(rows).view(view.dtype).reshape(-1)
    order = np.argsort(view, kind="stable")
    position = order[np.searchsorted(view[order], query)]
    if not np.array_equal(table[position], rows):
        raise ValueError(
            "welded_star.links must reference declared incidences; got undeclared incidence rows"
        )
    return position


def _facet_links(inverse: np.ndarray, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Consecutive occurrence pairs and their already-computed stable group order."""

    order = np.argsort(inverse, kind="stable")
    same = inverse[order[1:]] == inverse[order[:-1]]
    return order[:-1][same], order[1:][same], order


def _cache_audit_facets(
    mesh: cell_mesh_owner.CellMesh,
    points: np.ndarray,
    welded: np.ndarray,
    complex_: _Complex,
    inverse: np.ndarray,
    counts: np.ndarray,
    order: np.ndarray,
    /,
    *,
    audit_policy_id: str | None,
    audit_geometry: cell_geometry_owner.CellGeometrySpec | None,
    budget: CoordinateEnclosureBudget | None,
) -> None:
    """Retain current, unwelded triangle incidence data, never an audit verdict."""
    from ..discretization._coordinate_identity import coordinate_scope_key
    from ..geometry._embedding_state import Facets

    if (
        budget is None
        or audit_geometry is None
        or audit_policy_id is None
        or mesh.topological_dimension != 2
        or mesh.periodic_topology is not None
        or mesh.storage is not None
        or audit_geometry.storage_id is not None
        or not mesh.blocks
        or any(block.cell_kind != "triangle" for block in mesh.blocks)
        or points.shape != mesh.coordinates.shape
    ):
        return
    with budget.temporary_scope():
        budget.reserve(0, 4096)
        source_key = coordinate_scope_key(mesh, audit_geometry)
        if source_key in budget.facet_cache:
            return
        # Admit identity comparison and packed-offset storage before allocation.
        # All occurrence arrays and the stable group order were earned by the
        # audit; packing them creates no additional semantic work channel.
        budget.reserve(
            0, 16 * welded.size + 24 * points.size + 8 * (counts.size + 1)
        )
        if not np.array_equal(welded, np.arange(points.shape[0])) or not np.array_equal(
            np.asarray(points, dtype=np.float64).view(np.uint64),
            np.asarray(mesh.coordinates, dtype=np.float64).view(np.uint64),
        ):
            return
        group_offsets = np.empty(counts.size + 1, dtype=np.int64)
        group_offsets[0] = 0
        np.cumsum(counts, out=group_offsets[1:])
        prepared = Facets(
            complex_.facets,
            complex_.facet_signs,
            complex_.facet_cells,
            inverse,
            counts,
            order,
            group_offsets,
        )
        arrays = (
            prepared.rows,
            prepared.signs,
            prepared.cells,
            prepared.group,
            prepared.counts,
            prepared.order,
            prepared.group_offsets,
        )
        for array in arrays:
            array.flags.writeable = False
        # The existing retained_storage owner carries these arrays into later
        # phases, whose pairing and coverage checks still consume them normally.
        budget.retain_basis((audit_policy_id, source_key, prepared, arrays))
        budget.facet_cache[source_key] = prepared


def _vertex_links(
    complex_: _Complex, first: np.ndarray, second: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    facets = complex_.facets[first]
    slots = facets >= 0
    return (
        facets[slots][:, None],
        np.broadcast_to(complex_.facet_cells[first][:, None], facets.shape)[slots],
        np.broadcast_to(complex_.facet_cells[second][:, None], facets.shape)[slots],
    )


def _periodic_facet_vertex_positions(
    mesh: cell_mesh_owner.CellMesh, topology: PeriodicMeshTopology, /
) -> dict[tuple[int, ...], tuple[int, dict[int, int]]]:
    """Index facet corners by quotient orbit and relative-image correspondence."""
    degree = mesh.topological_dimension - 1
    orbits = np.asarray(topology.orbits(degree)[0], dtype=np.int64)
    permutations = topology.entity_vertex_permutations(mesh, degree)
    result = {}
    for rows, corners in lifted_loops(mesh, degree):
        for row, loop in zip(rows, corners, strict=True):
            permutation = permutations[row]
            result[tuple(sorted(int(vertex) for vertex in loop))] = (
                int(orbits[row]),
                {
                    int(vertex): int(position)
                    for vertex, position in zip(loop, permutation, strict=True)
                },
            )
    return result


def _periodic_vertex_star_components(
    mesh: cell_mesh_owner.CellMesh,
    topology: PeriodicMeshTopology,
    complex_: _Complex,
    welded: np.ndarray,
    physical_links: tuple[np.ndarray, np.ndarray, np.ndarray],
    complete: np.ndarray | None,
    /,
) -> int:
    """Count disconnected quotient vertex stars without collapsing winding corners."""
    original = (
        complex_
        if np.array_equal(welded, np.arange(welded.size, dtype=np.int64))
        else _welded_complex(mesh, np.arange(welded.size, dtype=np.int64))
    )
    slots = complex_.cells >= 0
    owners = np.broadcast_to(
        np.arange(complex_.cells.shape[0], dtype=np.int64)[:, None], complex_.cells.shape
    )[slots]
    nodes = np.unique(np.stack((owners, complex_.cells[slots]), axis=1), axis=0)
    indices = {
        (int(owner), int(vertex)): index for index, (owner, vertex) in enumerate(nodes)
    }
    representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
    roots = representatives[nodes[:, 1]]
    facets = _periodic_facet_vertex_positions(mesh, topology)
    occurrences: dict[tuple[int, int, int], list[int]] = {}
    for facet, owner in zip(original.facets, original.facet_cells, strict=True):
        vertices = tuple(int(vertex) for vertex in facet if vertex >= 0)
        orbit, positions = facets[tuple(sorted(vertices))]
        for vertex in vertices:
            node = indices[(int(owner), int(welded[vertex]))]
            occurrences.setdefault(
                (int(roots[node]), orbit, positions[vertex]), []
            ).append(node)
    keys: list[tuple[int]] = []
    first_nodes: list[int] = []
    second_nodes: list[int] = []
    for (root, _, _), group in sorted(occurrences.items()):
        for first, second in zip(group[:-1], group[1:], strict=True):
            keys.append((root,))
            first_nodes.append(first)
            second_nodes.append(second)
    # Explicit geometric welding can add facet adjacency beyond the authored
    # periodic descriptor. Keep that adjacency alongside quotient gluing.
    vertices, first_cells, second_cells = physical_links
    for (vertex,), first, second in zip(vertices, first_cells, second_cells, strict=True):
        keys.append((int(representatives[vertex]),))
        first_nodes.append(indices[(int(first), int(vertex))])
        second_nodes.append(indices[(int(second), int(vertex))])
    return _star_components(
        roots[:, None],
        np.arange(nodes.shape[0], dtype=np.int64),
        (
            np.asarray(keys, dtype=np.int64).reshape((-1, 1)),
            np.asarray(first_nodes, dtype=np.int64),
            np.asarray(second_nodes, dtype=np.int64),
        ),
        None if complete is None else complete[nodes[:, 0]],
    )


def _edge_links(
    complex_: _Complex, first: np.ndarray, second: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    facets = complex_.facets[first]
    lengths = np.sum(facets >= 0, axis=1)
    width = facets.shape[1]
    column = np.arange(width)
    following = facets[
        np.arange(facets.shape[0])[:, None], (column + 1) % lengths[:, None]
    ]
    slots = column[None, :] < lengths[:, None]
    edges = np.sort(np.stack((facets[slots], following[slots]), axis=1), axis=1)
    return (
        edges,
        np.broadcast_to(complex_.facet_cells[first][:, None], facets.shape)[slots],
        np.broadcast_to(complex_.facet_cells[second][:, None], facets.shape)[slots],
    )


def _facet_ridges(facets: np.ndarray, /) -> np.ndarray:
    """Sorted ridge rows of facet occurrences (none for point facets)."""
    if facets.shape[1] == 1:
        return np.zeros((0, 1), dtype=np.int64)
    if facets.shape[1] == 2:
        return np.concatenate((facets[:, :1], facets[:, 1:2]))
    lengths = np.sum(facets >= 0, axis=1)
    column = np.arange(facets.shape[1])
    following = facets[
        np.arange(facets.shape[0])[:, None], (column + 1) % lengths[:, None]
    ]
    slots = column[None, :] < lengths[:, None]
    return np.sort(np.stack((facets[slots], following[slots]), axis=1), axis=1)


def _truncated_facet_slots(
    mesh: cell_mesh_owner.CellMesh,
    complex_: _Complex,
    boundary: np.ndarray,
    /,
) -> np.ndarray | None:
    """Unpaired facet occurrences that cut an owner-local closure, not the domain.

    A dense mesh has none. Owner-local storage declares which resident facets
    lie on the physical boundary; every other locally unpaired facet truncates
    a resident star, and the collective subdivision and reciprocal-facet
    checks own its global closure.
    """
    storage = mesh.storage
    if (
        storage is None
        or storage.local_physical_boundary_facets is None
        or mesh.periodic_topology is not None
    ):
        return None
    physical = np.asarray(storage.local_physical_boundary_facets, dtype=np.bool_)
    keys = entity_keys(mesh, mesh.topological_dimension - 1)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    facets = complex_.facets[boundary]
    valid = facets >= 0
    sentinel = np.iinfo(np.int64).max
    slot_keys = np.sort(
        np.where(valid, vertex_ids[np.maximum(facets, 0)], sentinel), axis=1
    )
    slot_keys = np.where(slot_keys == sentinel, -1, slot_keys)
    rows = key_rows(keys, slot_keys)
    if physical.shape != (keys.shape[0],) or np.any(rows < 0):
        raise ValueError(
            "mesh.storage.local_physical_boundary_facets must label every resident facet; "
            f"got mask shape={physical.shape!r} for {keys.shape[0]} facets"
        )
    truncated = np.zeros(boundary.shape, dtype=np.bool_)
    truncated[np.flatnonzero(boundary)] = ~physical[rows]
    return truncated


def _open_boundary_count(
    complex_: _Complex,
    counts: np.ndarray,
    inverse: np.ndarray,
    embedded: bool,
    truncated: np.ndarray | None,
    /,
) -> int:
    unpaired = counts[inverse] == 1
    cut = np.zeros(unpaired.shape, dtype=np.bool_) if truncated is None else truncated
    boundary = unpaired & ~cut
    if embedded:
        return int(np.count_nonzero(boundary))
    ridges = _facet_ridges(complex_.facets[boundary])
    if ridges.shape[0] == 0:
        return 0
    unique, ridge_counts = np.unique(ridges, axis=0, return_counts=True)
    open_ = ridge_counts != 2
    if np.any(cut):
        # A ridge on a closure cut has its other physical facets beyond the
        # resident star; only the collective closure checks can pair them.
        open_ &= key_rows(
            np.unique(_facet_ridges(complex_.facets[cut]), axis=0), unique
        ) < 0
    return int(np.count_nonzero(open_))


# Self-intersection ---------------------------------------------------------------


def _signs(result: PredicateResult) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(result.signs, dtype=np.int8), np.asarray(
        result.certain, dtype=np.bool_
    )


# Meshcore adaptive predicates are used when available; dyadic host arithmetic
# resolves filtered-uncertain signs otherwise.
@checked
def orientation3d_signs(
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> tuple[np.ndarray, np.ndarray]:
    mode = resolve_host_predicate_mode(PredicateMode.EXACT)
    result = orient3d_host(
        a, b, c, d, mode=mode, budget=budget, memoize_native=False
    )
    return result.signs, result.certain


def _orient2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> tuple[np.ndarray, np.ndarray]:
    mode = resolve_host_predicate_mode(PredicateMode.EXACT)
    return _signs(orient2d(a, b, c, mode=mode, budget=budget))


def _segment_triangle_decision(
    side_a: tuple[np.ndarray, np.ndarray],
    side_b: tuple[np.ndarray, np.ndarray],
    edges: tuple[tuple[np.ndarray, np.ndarray], ...],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed segment/triangle contact for segments not lying in the plane.

    ``side_a``/``side_b`` are the exact ``(sign, certain)`` orientations of the
    segment endpoints against the triangle plane ``orient(p, q, r, end)``;
    ``edges`` are ``orient(a, b, p, q)``, ``orient(a, b, q, r)`` and
    ``orient(a, b, r, p)`` for segment ``ab`` and triangle ``pqr``.
    """

    (sign_a, certain_a), (sign_b, certain_b) = side_a, side_b
    edge_signs = np.stack([sign for sign, _ in edges], axis=1).astype(np.int16)
    straddles = (sign_a.astype(np.int16) * sign_b <= 0) & ~((sign_a == 0) & (sign_b == 0))
    inside = np.all(edge_signs >= 0, axis=1) | np.all(edge_signs <= 0, axis=1)
    known = np.stack([certain for _, certain in edges], axis=1)
    plane_known = certain_a & certain_b
    # A certified plane miss, or two certified opposite edge signs, decides the
    # contact whatever the remaining signs are.
    missed = plane_known & (sign_a.astype(np.int16) * sign_b > 0)
    separated = np.any(known & (edge_signs > 0), axis=1) & np.any(
        known & (edge_signs < 0), axis=1
    )
    certain = plane_known & (missed | separated | np.all(known, axis=1))
    return straddles & inside, certain


def _projection_axes(triangles: np.ndarray, /) -> np.ndarray:
    normal = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    dropped = np.argmax(np.abs(normal), axis=1)
    axes = np.asarray(((1, 2), (0, 2), (0, 1)))
    return axes[dropped]


def _project(points: np.ndarray, axes: np.ndarray, /) -> np.ndarray:
    return np.take_along_axis(points, axes, axis=1)


def _inside_triangle_2d(
    point: np.ndarray,
    triangle: tuple[np.ndarray, np.ndarray, np.ndarray],
    /,
    *,
    strict: bool,
    budget: CoordinateEnclosureBudget | None,
) -> tuple[np.ndarray, np.ndarray]:
    p, q, r = triangle
    signs = []
    certainties = []
    for start, stop in ((p, q), (q, r), (r, p)):
        sign, known = _orient2d(start, stop, point, budget=budget)
        signs.append(sign.astype(np.int16))
        certainties.append(known)
    stacked = np.stack(signs, axis=1)
    known = np.stack(certainties, axis=1)
    # Two certified opposite signs (or, for strict containment, one certified
    # zero) place the point outside whatever the unresolved signs are.
    outside = np.any(known & (stacked > 0), axis=1) & np.any(
        known & (stacked < 0), axis=1
    )
    if strict:
        inside = np.all(stacked > 0, axis=1) | np.all(stacked < 0, axis=1)
        outside |= np.any(known & (stacked == 0), axis=1)
    else:
        inside = np.all(stacked >= 0, axis=1) | np.all(stacked <= 0, axis=1)
    certain = np.all(known, axis=1) | outside
    return inside & ~outside, certain


def _filtered_support_side(
    origin: np.ndarray,
    edge: np.ndarray,
    point: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Certified line sides, including algebraic axis-aligned zero signs."""
    horizontal = origin[:, 1] == edge[:, 1]
    vertical = origin[:, 0] == edge[:, 0]
    certain = horizontal | vertical
    signs = np.zeros(origin.shape[0], dtype=np.int16)
    for selected, axis, orientation in ((horizontal, 0, 1), (~horizontal & vertical, 1, -1)):
        rows = np.flatnonzero(selected)
        direction = (edge[rows, axis] > origin[rows, axis]).astype(np.int16) - (
            edge[rows, axis] < origin[rows, axis]
        )
        side = (point[rows, 1 - axis] > origin[rows, 1 - axis]).astype(np.int16) - (
            point[rows, 1 - axis] < origin[rows, 1 - axis]
        )
        signs[rows] = orientation * direction * side
    remaining = np.flatnonzero(~certain)
    if remaining.size:
        result = orient2d(
            origin[remaining], edge[remaining], point[remaining],
            mode=PredicateMode.FILTERED, budget=budget,
        )
        signs[remaining] = np.asarray(result.signs)
        certain[remaining] = np.asarray(result.certain)
    return signs, certain


def _coplanar_filtered_separation(
    one: np.ndarray,
    two: np.ndarray,
    shared: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> np.ndarray:
    """Prove separation by a triangle edge without resolving uncertain signs.

    A shared vertex is local zero in both rows. Other vertices must lie outside
    its supporting edge, or on the opposite ray of that edge; convexity then
    confines the intersection to the permitted shared point. With no shared
    vertex all target vertices must be strictly outside. Unknown filter signs
    go to the complete predicate, never to a disjoint verdict.
    """
    separated = np.zeros(one.shape[0], dtype=np.bool_)
    if one.dtype == object:
        return separated
    for source, target in ((one, two), (two, one)):
        for edge, opposite in ((1, 2), (2, 1)):
            rows = np.flatnonzero(~separated)
            if not rows.size:
                return separated
            signs, known = _filtered_support_side(
                source[rows, 0],
                source[rows, edge],
                source[rows, opposite],
                budget=budget,
            )
            proved = known & (signs != 0)
            for vertex in range(3):
                candidates = np.flatnonzero(
                    proved & ((shared[rows] == 0) if vertex == 0 else True)
                )
                if not candidates.size:
                    continue
                selected = rows[candidates]
                origin = source[selected, 0]
                endpoint = source[selected, edge]
                point = target[selected, vertex]
                outside, known = _filtered_support_side(
                    origin, endpoint, point, budget=budget
                )
                opposite_ray = np.any(
                    ((point < origin) & (endpoint > origin))
                    | ((point > origin) & (endpoint < origin)),
                    axis=1,
                )
                proved[candidates] &= known & (
                    (signs[candidates] * outside < 0)
                    | ((shared[selected] == 1) & (outside == 0) & opposite_ray)
                )
            separated[rows] |= proved
    return separated


def _shared_vertex_cone_contact(
    one: np.ndarray,
    two: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decide contact of two nondegenerate triangles at their common vertex.

    Each triangle occupies a convex angular cone of opening less than pi.
    Their intersection contains a nonzero ray exactly when a boundary ray of
    either cone belongs to the other closed cone. Any such ray has a positive
    initial segment inside both finite triangles. Six orientation signs thus
    decide contact beyond the shared point, without testing segments whose
    common endpoint is already known. Degenerate cones retain the full route.
    """
    origin = one[:, 0]

    def side(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        signs, known = _filtered_support_side(origin, first, second, budget=budget)
        pending = np.flatnonzero(~known)
        if pending.size:
            signs[pending], known[pending] = _orient2d(
                origin[pending], first[pending], second[pending], budget=budget
            )
        return signs, known

    a, b = one[:, 1], one[:, 2]
    c, d = two[:, 1], two[:, 2]
    ab, known_ab = side(a, b)
    cd, known_cd = side(c, d)
    ac, known_ac = side(a, c)
    ad, known_ad = side(a, d)
    bc, known_bc = side(b, c)
    bd, known_bd = side(b, d)
    certain = (
        known_ab & known_cd & known_ac & known_ad & known_bc & known_bd
        & (ab != 0) & (cd != 0)
    )
    hit = (
        ((ac * ab >= 0) & (bc * ab <= 0))
        | ((ad * ab >= 0) & (bd * ab <= 0))
        | ((ac * cd <= 0) & (ad * cd >= 0))
        | ((bc * cd <= 0) & (bd * cd >= 0))
    )
    return hit, certain


def _coplanar_overlap(
    first: np.ndarray,
    second: np.ndarray,
    shared: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed coplanar triangle overlap away from shared vertices.

    Pairs sharing a vertex are reordered so that it sits at local index zero of
    both triangles; contact at that vertex is then not an intersection.
    """

    axes = _projection_axes(first)
    one = np.stack([_project(first[:, index], axes) for index in range(3)], axis=1)
    two = np.stack([_project(second[:, index], axes) for index in range(3)], axis=1)
    resolved = _coplanar_filtered_separation(one, two, shared, budget=budget)
    hit = np.zeros(first.shape[0], dtype=np.bool_)
    certain = np.ones(first.shape[0], dtype=np.bool_)
    vertex_rows = np.flatnonzero(~resolved & (shared == 1))
    if vertex_rows.size:
        contact, decided = _shared_vertex_cone_contact(
            one[vertex_rows], two[vertex_rows], budget=budget
        )
        hit[vertex_rows[decided]] = contact[decided]
        resolved[vertex_rows[decided]] = True
    pending = np.flatnonzero(~resolved)
    if pending.size:
        hit[pending], certain[pending] = _coplanar_projected_overlap(
            one[pending], two[pending], shared[pending], budget=budget
        )
    return hit, certain


def _coplanar_projected_overlap(
    one: np.ndarray,
    two: np.ndarray,
    shared: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decide the remaining contacts after certified supporting-edge separation."""
    hit = np.zeros(one.shape[0], dtype=np.bool_)
    certain = np.ones(one.shape[0], dtype=np.bool_)
    touching = shared == 0
    mode = resolve_host_predicate_mode(PredicateMode.EXACT)
    for index in range(3):
        for other in range(3):
            adjacent = (index == 0 or (index + 1) % 3 == 0) and (
                other == 0 or (other + 1) % 3 == 0
            )
            status = np.asarray(
                segment_intersections_2d(
                    one[:, index],
                    one[:, (index + 1) % 3],
                    two[:, other],
                    two[:, (other + 1) % 3],
                    mode=mode,
                    budget=budget,
                ).status
            )
            # Edges through the shared vertex meet there by construction; only
            # contact beyond it is an intersection.
            beyond_shared = (status == SegmentIntersectionStatus.PROPER_CROSSING) | (
                status == SegmentIntersectionStatus.COLLINEAR_OVERLAP
            )
            contact = status != SegmentIntersectionStatus.DISJOINT
            hit |= np.where(touching | (not adjacent), contact, beyond_shared)
            certain &= status != SegmentIntersectionStatus.UNCERTAIN
    for index in range(3):
        inside_two, known_two = _inside_triangle_2d(
            one[:, index],
            (two[:, 0], two[:, 1], two[:, 2]),
            strict=True,
            budget=budget,
        )
        inside_one, known_one = _inside_triangle_2d(
            two[:, index],
            (one[:, 0], one[:, 1], one[:, 2]),
            strict=True,
            budget=budget,
        )
        hit |= inside_two | inside_one
        certain &= known_two & known_one
    return hit, certain


@checked
def triangle_pairs_intersect(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reuse value-bound contact batches without omitting any requested pair.

    The source digest binds every binary coordinate, while canonical vertex rows
    preserve welded identities. Broad-phase enumeration and its capacity checks
    remain with each caller; only already decided narrow-phase predicates reuse
    evidence. Exact object-coordinate banks keep their own arithmetic route.
    """

    if budget is None or points.dtype != np.float64 or not points.flags.c_contiguous:
        return _triangle_pairs_intersect(points, first, second, budget=budget)
    count = first.shape[0]
    with budget.temporary_scope():
        # Canonical rows, sorting/gather scratch, lookup masks and verdicts.
        # This is byte admission, not a new CPU-pass work-unit definition.
        budget.reserve(0, 4096 + count * 512)
        source = (
            points.shape,
            points.dtype.str,
            sha256(memoryview(points)).digest(),
            resolve_host_predicate_mode(PredicateMode.EXACT).value,
            budget.maximum_work_units,
            budget.maximum_memory_bytes,
        )
        left = np.sort(first, axis=1)
        right = np.sort(second, axis=1)
        row_type = np.dtype((np.void, 3 * np.dtype(np.int64).itemsize))
        left_keys = np.ascontiguousarray(left, dtype=np.int64).view(row_type).ravel()
        right_keys = np.ascontiguousarray(right, dtype=np.int64).view(row_type).ravel()
        swap = np.argsort(
            np.stack((left_keys, right_keys), axis=1), axis=1, kind="stable"
        )[:, 0] == 1
        rows = np.concatenate(
            (np.where(swap[:, None], right, left), np.where(swap[:, None], left, right)),
            axis=1,
        ).astype(np.int64, copy=False)
        keys = np.ascontiguousarray(rows).view(
            np.dtype((np.void, 6 * np.dtype(np.int64).itemsize))
        ).ravel()
        hit = np.zeros(count, dtype=np.bool_)
        certain = np.ones(count, dtype=np.bool_)
        pending = np.ones(count, dtype=np.bool_)
        for prior_keys, prior_hit, prior_certain in budget.triangle_contact_batches.get(
            source, ()
        ):
            budget.reserve(0)
            positions = np.flatnonzero(pending)
            if not positions.size:
                break
            indices = np.searchsorted(prior_keys, keys[positions])
            inside = indices < prior_keys.size
            selected = positions[inside]
            indices = indices[inside]
            matches = prior_keys[indices] == keys[selected]
            selected = selected[matches]
            indices = indices[matches]
            hit[selected] = prior_hit[indices]
            certain[selected] = prior_certain[indices]
            pending[selected] = False
        missing = np.flatnonzero(pending)
        if missing.size:
            hit[missing], certain[missing] = _triangle_pairs_intersect(
                points, first[missing], second[missing], budget=budget
            )
            order = missing[np.argsort(keys[missing], kind="stable")]
            record = (keys[order], hit[order], certain[order])
            for value in record:
                value.flags.writeable = False
            budget.retain_basis((source, record))
            budget.triangle_contact_batches.setdefault(source, []).append(record)
        budget.reserve(0)
        return hit, certain


def _triangle_pairs_intersect(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersection of triangle pairs beyond their shared (welded) vertices."""

    shared_mask = first[:, :, None] == second[:, None, :]
    shared = np.sum(shared_mask, axis=(1, 2))
    hit = np.zeros(first.shape[0], dtype=np.bool_)
    certain = np.ones(first.shape[0], dtype=np.bool_)
    # Rotate both triangles so a shared vertex (if any) is local vertex zero.
    first_rotation = np.argmax(np.any(shared_mask, axis=2), axis=1)
    second_rotation = np.argmax(np.any(shared_mask, axis=1), axis=1)
    rotate = np.stack([np.arange(3) + offset for offset in range(3)]) % 3
    one = np.take_along_axis(first, rotate[first_rotation], axis=1)
    two = np.take_along_axis(second, rotate[second_rotation], axis=1)
    one = np.where((shared == 0)[:, None], first, one)
    two = np.where((shared == 0)[:, None], second, two)
    tri_one = points[one]
    tri_two = points[two]
    # An axis-aligned plane containing all six source coordinates proves the
    # three plane signs are exactly zero, without three orientation queries.
    # Equality compares source values, not a tolerance or rounded normal.
    coordinate_plane = np.any(
        np.all(tri_one == tri_one[:, :1], axis=1)
        & np.all(tri_two == tri_one[:, :1], axis=1),
        axis=1,
    )
    signs = np.zeros((first.shape[0], 3), dtype=np.int8)
    plane_known = np.ones(first.shape[0], dtype=np.bool_)
    plane_certain = np.ones((first.shape[0], 3), dtype=np.bool_)
    shared_second = np.any(one[:, :, None] == two[:, None, :], axis=1)
    # A shared index repeats one plane operand, so its determinant is exactly
    # zero. Avoid sending that known identity through the exact predicate.
    # Unknown object atoms and nonfinite operands keep their admission paths.
    shared_zero = np.zeros_like(shared_second)
    if np.issubdtype(points.dtype, np.floating) and points.dtype.itemsize <= 8:
        shared_zero = shared_second & np.all(
            np.isfinite(tri_one), axis=(1, 2)
        )[:, None]
    elif points.dtype == object and np.any(shared_second):
        if budget is not None:
            budget.reserve(points.size)
        if all(type(value) is int for value in points.flat):
            shared_zero = shared_second
    for index in range(3):
        plane_rows = np.flatnonzero(~coordinate_plane & ~shared_zero[:, index])
        if not plane_rows.size:
            continue
        sign, known = orientation3d_signs(
            tri_one[plane_rows, 0],
            tri_one[plane_rows, 1],
            tri_one[plane_rows, 2],
            tri_two[plane_rows, index],
            budget=budget,
        )
        signs[plane_rows, index] = sign
        plane_certain[plane_rows, index] = known
        plane_known[plane_rows] &= known
    coplanar = np.all(signs == 0, axis=1)
    # Strict same-side nonshared vertices can meet the first plane only at
    # declared shared vertices. Convexity certifies disjointness beyond those
    # permitted features without segment or projected-overlap predicates.
    separated = (
        plane_known
        & np.any(~shared_second, axis=1)
        & (
            np.all((signs > 0) | shared_second, axis=1)
            | np.all((signs < 0) | shared_second, axis=1)
        )
    )
    certain &= plane_known

    edge_share = (shared == 2) & ~separated
    if np.any(edge_share):
        rows = np.flatnonzero(edge_share)
        # Two shared vertices: after rotation vertex 0 is shared; the other
        # shared vertex is found in each triangle and the free vertices compared.
        tri = points[first[rows]]
        other = points[second[rows]]
        free_first = ~np.any(shared_mask[rows], axis=2)
        free_second = ~np.any(shared_mask[rows], axis=1)
        common = first[rows][np.any(shared_mask[rows], axis=2)].reshape(-1, 2)
        a = points[common[:, 0]]
        b = points[common[:, 1]]
        c = tri[free_first]
        d = other[free_second]
        axes = _projection_axes(tri)
        side_c, known_c = _orient2d(
            _project(a, axes),
            _project(b, axes),
            _project(c, axes),
            budget=budget,
        )
        side_d, known_d = _orient2d(
            _project(a, axes),
            _project(b, axes),
            _project(d, axes),
            budget=budget,
        )
        folded = coplanar[rows] & (side_c == side_d) & (side_c != 0)
        hit[rows] = folded
        certain[rows] &= known_c & known_d
    general = (shared <= 1) & ~separated
    if np.any(general & coplanar):
        rows = np.flatnonzero(general & coplanar)
        overlap, known = _coplanar_overlap(
            tri_one[rows], tri_two[rows], shared[rows], budget=budget
        )
        hit[rows] = overlap
        certain[rows] &= known
    crossing_rows = np.flatnonzero(general & ~coplanar)
    if crossing_rows.size:
        a = tri_one[crossing_rows]
        b = tri_two[crossing_rows]
        count = crossing_rows.size
        # A shared vertex (local zero) touches the other triangle by
        # construction; only the opposite edge (index one) carries new contact.
        free = shared[crossing_rows] == 0
        relevant = (free, np.ones((count,), dtype=np.bool_), free)
        # Each exact orientation is evaluated once. The sides of b's vertices
        # against a's plane are the plane signs above. Orientation alternates in
        # its four points and (a, b, c, d) -> (c, d, a, b) is even, so the
        # edge sign orient(b_j, b_j+1, a_i, a_i+1) equals orient(a_i, a_i+1,
        # b_j, b_j+1) for both segment/triangle directions.
        b_sides = (signs[crossing_rows], plane_certain[crossing_rows])
        a_sides = (
            np.zeros((count, 3), dtype=np.int8),
            np.ones((count, 3), dtype=np.bool_),
        )
        for vertex in range(3):
            rows = np.flatnonzero(relevant[vertex] | relevant[(vertex - 1) % 3])
            if rows.size:
                a_sides[0][rows, vertex], a_sides[1][rows, vertex] = (
                    orientation3d_signs(
                        b[rows, 0], b[rows, 1], b[rows, 2], a[rows, vertex], budget=budget
                    )
                )
        edge_signs = np.zeros((count, 3, 3), dtype=np.int8)
        edge_known = np.ones((count, 3, 3), dtype=np.bool_)
        for i in range(3):
            for j in range(3):
                rows = np.flatnonzero(relevant[i] | relevant[j])
                if rows.size:
                    edge_signs[rows, i, j], edge_known[rows, i, j] = orientation3d_signs(
                        a[rows, i],
                        a[rows, (i + 1) % 3],
                        b[rows, j],
                        b[rows, (j + 1) % 3],
                        budget=budget,
                    )
        contact = np.zeros((count,), dtype=np.bool_)
        known = np.ones((count,), dtype=np.bool_)
        for sides, transposed in ((a_sides, False), (b_sides, True)):
            for index in range(3):
                rows = np.flatnonzero(relevant[index])
                if not rows.size:
                    continue
                result, certain_ = _segment_triangle_decision(
                    (sides[0][rows, index], sides[1][rows, index]),
                    (sides[0][rows, (index + 1) % 3], sides[1][rows, (index + 1) % 3]),
                    tuple(
                        (
                            edge_signs[rows, edge, index],
                            edge_known[rows, edge, index],
                        )
                        if transposed
                        else (
                            edge_signs[rows, index, edge],
                            edge_known[rows, index, edge],
                        )
                        for edge in range(3)
                    ),
                )
                contact[rows] |= result
                known[rows] &= certain_
        hit[crossing_rows] = contact
        certain[crossing_rows] &= known
    unresolved = np.flatnonzero(~certain)
    if unresolved.size:
        disjoint = _projected_disjoint(
            points,
            first[unresolved],
            second[unresolved],
            tri_one[unresolved],
            tri_two[unresolved],
            shared[unresolved],
            budget=budget,
        )
        hit[unresolved[disjoint]] = False
        certain[unresolved[disjoint]] = True
    return hit, certain


def _projected_disjoint(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    tri_one: np.ndarray,
    tri_two: np.ndarray,
    shared: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> np.ndarray:
    """Certified disjointness beyond shared features from one coordinate projection.

    On a coordinate plane where both triangles project nondegenerately each
    triangle is a graph, so projections meeting only at shared features imply
    3D intersection only there, whether the triangles are coplanar or not. This decides
    near-coplanar pairs (flat sheets) left unresolved by filtered 3D signs.
    """
    axes = _projection_axes(tri_one)
    projected_one = np.stack([_project(tri_one[:, index], axes) for index in range(3)], 1)
    projected_two = np.stack([_project(tri_two[:, index], axes) for index in range(3)], 1)
    sign_one, known_one = _orient2d(
        projected_one[:, 0],
        projected_one[:, 1],
        projected_one[:, 2],
        budget=budget,
    )
    sign_two, known_two = _orient2d(
        projected_two[:, 0],
        projected_two[:, 1],
        projected_two[:, 2],
        budget=budget,
    )
    graphs = known_one & known_two & (sign_one != 0) & (sign_two != 0)
    disjoint = np.zeros(first.shape[0], dtype=np.bool_)
    general = np.flatnonzero(graphs & (shared <= 1))
    if general.size:
        overlap, known = _coplanar_overlap(
            tri_one[general], tri_two[general], shared[general], budget=budget
        )
        disjoint[general] = known & ~overlap
    edge_rows = np.flatnonzero(graphs & (shared == 2))
    if edge_rows.size:
        shared_mask = first[edge_rows][:, :, None] == second[edge_rows][:, None, :]
        common = first[edge_rows][np.any(shared_mask, axis=2)].reshape(-1, 2)
        free_one = points[first[edge_rows][~np.any(shared_mask, axis=2)]]
        free_two = points[second[edge_rows][~np.any(shared_mask, axis=1)]]
        local_axes = axes[edge_rows]
        start = _project(points[common[:, 0]], local_axes)
        stop = _project(points[common[:, 1]], local_axes)
        side_one, known_side_one = _orient2d(
            start, stop, _project(free_one, local_axes), budget=budget
        )
        side_two, known_side_two = _orient2d(
            start, stop, _project(free_two, local_axes), budget=budget
        )
        opposite = side_one.astype(np.int16) * side_two < 0
        disjoint[edge_rows] = known_side_one & known_side_two & opposite
    return disjoint


def _surface_triangles(
    points: np.ndarray,
    complex_: _Complex,
    boundary: np.ndarray | None,
    candidate_capacity: int,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> tuple[np.ndarray, int, int, int, bool]:
    """Triangles of the tested surface under one polygon-candidate budget."""

    loops = complex_.facets[boundary] if boundary is not None else complex_.cells
    lengths = np.sum(loops >= 0, axis=1)
    triangles = []
    intersecting = 0
    undecided = 0
    candidate_count = 0
    exceeded = False
    for length in np.unique(lengths):
        rows = loops[lengths == length][:, :length]
        if length >= 5:
            collapsed = _collapsed(rows)
            polygon_rows = rows[~collapsed]
            remaining = candidate_capacity - candidate_count
            if polygon_rows.size and remaining > 0:
                clipped, _, crossing, unknown, used, exhausted = polygon_loop_triangles(
                    points, polygon_rows, remaining, budget=budget
                )
                triangles.append(clipped)
                intersecting += int(np.count_nonzero(crossing))
                undecided += int(np.count_nonzero(unknown))
                candidate_count += used
                exceeded |= exhausted
            elif polygon_rows.size:
                undecided += polygon_rows.shape[0]
                exceeded = True
            rows = rows[collapsed]
        for index in range(1, length - 1):
            triangles.append(rows[:, (0, index, index + 1)])
    return (
        np.concatenate(triangles) if triangles else np.empty((0, 3), dtype=np.int64),
        intersecting,
        undecided,
        candidate_count,
        exceeded,
    )


def _self_intersections(
    points: np.ndarray,
    complex_: _Complex,
    boundary: np.ndarray,
    dimension: int,
    candidate_capacity: int,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> tuple[int, int, bool]:
    """Return bounded intersection count, unresolved count, and capacity status."""

    if points.shape[1] == 2:
        points = np.pad(points, ((0, 0), (0, 1)))
    match (points.shape[1], dimension):
        case (3, 2):
            triangles, loops, undecided, used, exceeded = _surface_triangles(
                points, complex_, None, candidate_capacity, budget=budget
            )
        case (3, 3):
            triangles, loops, undecided, used, exceeded = _surface_triangles(
                points, complex_, boundary, candidate_capacity, budget=budget
            )
        case _:
            return 0, 0, False
    triangles = triangles[~_collapsed(triangles)]
    if triangles.shape[0] < 2:
        return loops, undecided, exceeded
    corners = points[triangles]
    bvh = prepare_bvh(np.min(corners, axis=1), np.max(corners, axis=1), dtype=np.float64)
    intersections = loops
    remaining = candidate_capacity - used
    for first, second in bvh_overlap_pair_blocks(bvh, bvh, include_touching=True):
        keep = first < second
        first = first[keep]
        second = second[keep]
        if first.size > remaining:
            first = first[:remaining]
            second = second[:remaining]
            exceeded = True
        remaining -= first.size
        if first.size:
            hit, certain = triangle_pairs_intersect(
                points, triangles[first], triangles[second], budget=budget
            )
            intersections += int(np.count_nonzero(hit & certain))
            undecided += int(np.count_nonzero(~certain))
        if exceeded:
            undecided += 1
            break
    return intersections, undecided, exceeded


@dataclass(frozen=True)
class _PlanarAuditEmbeddingProof:
    """Request-owned completed theorem, never an externally supplied exemption."""

    key: tuple[object, ...]
    geometry: CellGeometrySpec
    policy: CellMeshAuditPolicy
    limits: MeshCertificateLimits
    junction_vertices: tuple[int, ...]
    validity: CellValidityCertificate
    embedding: GlobalEmbeddingCertificate


def _planar_audit_embedding_key(
    mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec,
    policy: CellMeshAuditPolicy,
    limits: MeshCertificateLimits,
    junction_vertices: tuple[int, ...],
    /,
) -> tuple[object, ...]:
    from ..discretization._coordinate_identity import coordinate_scope_key

    # The live tree includes triangle order/global IDs, all topology incidence
    # arrays, source controls and routes. Do not trust a stored mesh/source ID
    # alone, nor a storage owner's shortcut for a mutable numerical carrier.
    return (
        coordinate_scope_key(mesh, geometry),
        canonical_fingerprint(array_tree_fingerprint((mesh, geometry))),
        policy.policy_id,
        limits.limits_id,
        junction_vertices,
        resolve_host_predicate_mode(PredicateMode.EXACT),
    )


def prepare_planar_audit_embedding(
    mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec,
    *,
    policy: CellMeshAuditPolicy,
    limits: MeshCertificateLimits,
    junction_vertices: ArrayLike | None,
    budget: CoordinateEnclosureBudget,
) -> tuple[CellValidityCertificate | None, GlobalEmbeddingCertificate | None]:
    """Earn the planar interior nonintersection theorem before the audit.

    Degree-one triangle sources, including proven affine chart chains, are
    eligible. EVERY routed corner's exact source image must equal the binary64
    carrier as a rational number.
    Rounded equality is insufficient: an embedded ideal source can round into
    an intersecting carrier. Positive cell validity plus the existing global
    theorem's facet pairing, closed contact-free boundary and degree-one
    conclusion then proves that interiors are disjoint and all contacts shared.
    No positivity, contact obligation, or certificate limit is synthesized here.
    """
    from ..discretization import _coordinate_enclosure as enclosure_owner
    from ..discretization._cell_geometry_validity import certify_cell_geometry_validity
    from ..geometry._global_embedding import (
        _junction_vertices,
        certify_global_embedding,
        coordinate_scope,
    )

    if (
        mesh.topological_dimension != 2
        or mesh.ambient_dimension != 2
        or mesh.periodic_topology is not None
        or mesh.storage is not None
        or not mesh.blocks
        or any(block.cell_kind != "triangle" for block in mesh.blocks)
        or policy.self_intersection == "skip"
    ):
        budget.planar_audit_embedding_proof = None
        return None, None
    junctions = tuple(_junction_vertices(mesh, junction_vertices).tolist())
    budget.reserve(0)
    key = _planar_audit_embedding_key(mesh, geometry, policy, limits, junctions)
    earned = budget.planar_audit_embedding_proof
    if earned is not None and earned.key == key:
        return earned.validity, earned.embedding
    # Clear obsolete authority before attempting a replacement that may fail.
    budget.planar_audit_embedding_proof = None
    with budget.activate(), budget.temporary_scope():
        elements, _, _ = geometry._resolve(
            mesh, exact_source_prepared=True, budget=budget
        )
        if any(
            not enclosure_owner.affine_simplex_source(element, budget=budget)
            or element.cell_kind != "triangle"
            or element.degree != 1
            for element in elements
        ):
            return None, None
        images = enclosure_owner.prepared_coordinate_vertex_images(
            mesh,
            geometry,
            mismatch="planar_audit_embedding.corner_images must agree at every shared vertex; got unequal exact images",
            budget=budget,
        )
        if images is None:
            return None, None
        points = np.asarray(mesh.coordinates, dtype=np.float64)
        # Source images are already earned. Admit the actual routed-vertex,
        # unique-row and binary64 comparison workspaces before packing them.
        budget.reserve(0, 1024 + 512 * sum(block.cell_count for block in mesh.blocks))
        used = np.unique(
            np.concatenate(
                [
                    np.asarray(block.vertices, dtype=np.int64).reshape(-1)
                    for block in mesh.blocks
                ]
            )
        )
        rounded = np.empty((used.size, 2), dtype=np.float64)
        for position, vertex in enumerate(used.tolist()):
            image = images[vertex]
            if image is None or len(image) != 2:
                return None, None
            rounded[position] = tuple(float(value) for value in image)
            if not np.all(np.isfinite(rounded[position])) or any(
                Fraction(float(value)) != exact
                for value, exact in zip(rounded[position], image, strict=True)
            ):
                return None, None
        if not np.array_equal(
            rounded.view(np.uint64), points[used].view(np.uint64)
        ):
            return None, None
        scope = coordinate_scope(mesh, geometry, budget=budget)
        if scope.coordinate_scope != "affine":
            return None, None
        validity = certify_cell_geometry_validity(
            geometry, mesh=mesh, policy=policy.validity_policy, budget=budget
        )
        if not validity.all_certified:
            return None, None
        embedding = certify_global_embedding(
            mesh,
            geometry,
            validity,
            limits=limits,
            junction_vertices=junction_vertices,
            budget=budget,
        )
        if embedding.status != "certified":
            return None, None
        proof = _PlanarAuditEmbeddingProof(
            key, geometry, policy, limits, junctions, validity, embedding
        )
        budget.retain_basis((key, proof, validity, embedding))
        budget.planar_audit_embedding_proof = proof
        return validity, embedding


def _completed_planar_audit_embedding(
    mesh: cell_mesh_owner.CellMesh,
    points: np.ndarray,
    welded: np.ndarray,
    /,
    *,
    audit_policy_id: str | None,
    audit_geometry: cell_geometry_owner.CellGeometrySpec | None,
    coincident_tolerance: float | None,
    candidate_capacity: int,
    intersection_candidate_capacity: int,
    check_manifold: bool,
    check_watertight: bool,
    budget: CoordinateEnclosureBudget | None,
) -> bool:
    """Import only the actual completed theorem for this unchanged carrier."""
    proof = None if budget is None else budget.planar_audit_embedding_proof
    if proof is None or audit_policy_id != proof.policy.policy_id:
        return False
    policy = proof.policy
    if (
        coincident_tolerance
        != (None if policy.coincident_vertices == "skip" else policy.coincident_vertex_tolerance)
        or candidate_capacity != policy.maximum_coincidence_candidates
        or intersection_candidate_capacity != policy.maximum_intersection_candidates
        or check_manifold != (policy.nonmanifold != "skip")
        or check_watertight != (policy.watertight_boundary != "skip")
        or proof.embedding.status != "certified"
        or not proof.validity.all_certified
        or proof.embedding.candidate_pair_count > intersection_candidate_capacity
        or not np.array_equal(welded, np.arange(points.shape[0]))
        or not np.array_equal(
            points.view(np.uint64),
            np.asarray(mesh.coordinates, dtype=np.float64).view(np.uint64),
        )
    ):
        return False
    budget.reserve(0)
    return proof.key == _planar_audit_embedding_key(
        mesh,
        proof.geometry if audit_geometry is None else audit_geometry,
        policy,
        proof.limits,
        proof.junction_vertices,
    )


# Entry point ---------------------------------------------------------------------


@checked
def audit_welded_topology(
    mesh: cell_mesh_owner.CellMesh,
    points: np.ndarray,
    /,
    *,
    coincident_tolerance: float | None,
    candidate_capacity: int,
    intersection_candidate_capacity: int,
    check_manifold: bool,
    check_watertight: bool,
    check_self_intersection: bool,
    budget: CoordinateEnclosureBudget | None = None,
    audit_policy_id: str | None = None,
    audit_geometry: cell_geometry_owner.CellGeometrySpec | None = None,
) -> WeldedTopologyEvidence:
    """Evaluate the welded-topology checks of one mesh with host coordinates.

    ``coincident_tolerance`` is relative to the bounding-box diagonal; ``None``
    disables welding.
    """
    if candidate_capacity <= 0 or intersection_candidate_capacity <= 0:
        raise ValueError(
            "audit_welded_topology.candidate_capacity must be positive along with intersection_candidate_capacity; "
            f"got coincidence={candidate_capacity}, intersection={intersection_candidate_capacity}"
        )
    if points.shape[0] == 0 and (
        mesh.storage is None
        or mesh.storage.global_entity_counts[mesh.topological_dimension] <= 0
        or points.shape != mesh.coordinates.shape
        or any(entities.count != 0 for entities in mesh.topology.entity_sets)
    ):
        raise ValueError(
            "mesh.blocks must accompany zero resident topology and distributed storage when empty; "
            f"got resident counts={tuple(entities.count for entities in mesh.topology.entity_sets)!r}, "
            f"storage={type(mesh.storage).__name__}, points.shape={points.shape!r}"
        )

    if coincident_tolerance is None:
        welded, coincident, exceeded = np.arange(points.shape[0]), 0, False
    else:
        diagonal = (
            0.0
            if points.shape[0] == 0
            else float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))
        )
        welded, coincident, exceeded = _coincident_representatives(
            points, coincident_tolerance * diagonal, candidate_capacity
        )
    unresolved = ["coincident_vertex_capacity"] if exceeded else []
    complex_ = _welded_complex(mesh, welded)
    keys = _sorted_keys(complex_.facets)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    first, second, order = _facet_links(inverse)
    twice = counts[inverse[first]] == 2
    pair_first = first[twice]
    pair_second = second[twice]
    if mesh.topological_dimension == 1:
        inconsistent = (
            complex_.facet_signs[pair_first] == complex_.facet_signs[pair_second]
        )
    elif mesh.topological_dimension == 2:
        direction = complex_.facets[:, 0] < complex_.facets[:, 1]
        inconsistent = direction[pair_first] == direction[pair_second]
    else:
        successor, predecessor = _loop_neighbors(complex_.facets)
        inconsistent = successor[pair_first] != predecessor[pair_second]
    nonmanifold_vertices = 0
    nonmanifold_edges = 0
    if check_manifold and mesh.topological_dimension >= 2:
        cell_vertices = complex_.cells
        complete = (
            None
            if mesh.storage is None or mesh.storage.local_neighborhood_complete is None
            else np.asarray(mesh.storage.local_neighborhood_complete, dtype=np.bool_)
        )
        if complete is not None and complete.shape != (cell_vertices.shape[0],):
            raise ValueError(
                "mesh.storage.local_neighborhood_complete must label every resident cell; "
                f"got shape={complete.shape!r}, resident_cells={cell_vertices.shape[0]}"
            )
        slots = cell_vertices >= 0
        owners = np.broadcast_to(
            np.arange(cell_vertices.shape[0])[:, None], cell_vertices.shape
        )[slots]
        vertex_links = _vertex_links(complex_, first, second)
        if mesh.periodic_topology is None:
            nonmanifold_vertices = _star_components(
                cell_vertices[slots][:, None], owners, vertex_links, complete
            )
        else:
            nonmanifold_vertices = _periodic_vertex_star_components(
                mesh, mesh.periodic_topology, complex_, welded, vertex_links, complete
            )
        if mesh.topological_dimension == 3:
            nonmanifold_edges = _star_components(
                np.sort(complex_.edges, axis=1),
                complex_.edge_cells,
                _edge_links(complex_, first, second),
                complete,
            )
    embedded = mesh.ambient_dimension > mesh.topological_dimension
    open_facets = (
        _open_boundary_count(
            complex_,
            counts,
            inverse,
            embedded,
            _truncated_facet_slots(mesh, complex_, counts[inverse] == 1),
        )
        if check_watertight
        else 0
    )
    intersections = 0
    if check_self_intersection and not _completed_planar_audit_embedding(
        mesh,
        points,
        welded,
        audit_policy_id=audit_policy_id,
        audit_geometry=audit_geometry,
        coincident_tolerance=coincident_tolerance,
        candidate_capacity=candidate_capacity,
        intersection_candidate_capacity=intersection_candidate_capacity,
        check_manifold=check_manifold,
        check_watertight=check_watertight,
        budget=budget,
    ):
        boundary = counts[inverse] == 1
        intersections, uncertain, intersection_exceeded = _self_intersections(
            points[welded],
            complex_,
            boundary,
            mesh.topological_dimension,
            intersection_candidate_capacity,
            budget=budget,
        )
        if intersection_exceeded:
            unresolved.append("self_intersection_capacity")
        if uncertain:
            unresolved.append("self_intersection_predicates")
    _cache_audit_facets(
        mesh,
        points,
        welded,
        complex_,
        inverse,
        counts,
        order,
        audit_policy_id=audit_policy_id,
        audit_geometry=audit_geometry,
        budget=budget,
    )
    return WeldedTopologyEvidence(
        coincident_vertices=coincident,
        collapsed_cells=int(np.count_nonzero(_collapsed(complex_.cells))),
        duplicate_cells=_duplicate_count(complex_.cells),
        nonmanifold_facets=int(np.count_nonzero(counts > 2)),
        nonmanifold_edges=nonmanifold_edges,
        nonmanifold_vertices=nonmanifold_vertices,
        inconsistent_facets=int(np.count_nonzero(inconsistent)),
        open_facets=open_facets,
        self_intersections=intersections,
        unresolved=tuple(unresolved),
    )


__all__ = [
    "WeldedTopologyEvidence",
    "audit_welded_topology",
    "prepare_planar_audit_embedding",
]
