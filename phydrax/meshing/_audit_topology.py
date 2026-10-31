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
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._geometry_predicates import (
    orient2d,
    orient3d,
    polygon_simplicity_2d,
    PolygonSimplicityStatus,
    PredicateMode,
    PredicateSign,
    resolve_host_predicate_mode,
    segment_intersections_2d,
    SegmentIntersectionStatus,
)
from ..discretization import CellMesh, PolyhedralConnectivity
from ..discretization._cell_geometry_validity import polyhedral_star_tables
from ..discretization._reference_cell import reference_cell_topology


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
    points: np.ndarray, tolerance: float, /
) -> tuple[np.ndarray, int]:
    """Stream geometric coincidences into a diagnostic, deterministic partition.

    The union-find stores only vertex connectivity, not the potentially
    quadratic candidate graph. It does not change the mesh or its entity IDs.
    """
    count = points.shape[0]
    parents = np.arange(count, dtype=np.int64)
    if count < 2:
        return parents, 0

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    bvh = prepare_bvh(points, points, dtype=np.float64)
    for first, second in bvh_overlap_pair_blocks(
        bvh, bvh, include_touching=True, absolute_tolerance=tolerance
    ):
        unique = first < second
        first, second = first[unique], second[unique]
        close = np.linalg.norm(points[first] - points[second], axis=1) <= tolerance
        for first_id, second_id in zip(first[close], second[close], strict=True):
            first_root, second_root = root(int(first_id)), root(int(second_id))
            if first_root != second_root:
                parents[max(first_root, second_root)] = min(first_root, second_root)
    for index in range(count):
        parents[index] = root(index)
    sizes = np.bincount(parents, minlength=count)
    return parents, int(np.count_nonzero(sizes[parents] > 1))


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


def _dense_block_facets(
    rows: np.ndarray, cell_kind: str, dimension: int, owners: np.ndarray, /
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray | None,
    np.ndarray | None,
]:
    """Oriented facet occurrences (and 3-D cell edges) of one dense cell block."""

    count = rows.shape[0]
    edges = None
    edge_cells = None
    if dimension == 1:
        local_facets = ((0,), (1,))
        local_signs = (-1, 1)
    elif dimension == 2:
        arity = rows.shape[1]
        local_facets = tuple((index, (index + 1) % arity) for index in range(arity))
        local_signs = (0,) * arity
    else:
        topology = reference_cell_topology(cell_kind)
        local_facets = tuple(topology.entities[2])
        local_signs = (0,) * len(local_facets)
        local_edges = np.asarray(topology.entities[1])
        edges = rows[:, local_edges].reshape(-1, 2)
        edge_cells = np.repeat(owners, local_edges.shape[0])
    width = max(len(face) for face in local_facets)
    facet_rows = []
    facet_cells = []
    facet_signs = []
    for face, sign in zip(local_facets, local_signs, strict=True):
        values = np.full((count, width), -1, dtype=np.int64)
        values[:, : len(face)] = rows[:, np.asarray(face)]
        facet_rows.append(values)
        facet_cells.append(owners)
        facet_signs.append(np.full(count, sign, dtype=np.int8))
    return facet_rows, facet_cells, facet_signs, edges, edge_cells


def _welded_complex(mesh: CellMesh, welded: np.ndarray, /) -> _Complex:
    """Cells, oriented facet occurrences, and (3-D) cell edges in welded ids."""

    connectivity = mesh.connectivity
    dimension = mesh.topological_dimension
    cell_rows = []
    facet_rows = []
    facet_cells = []
    facet_signs = []
    edge_rows = []
    edge_cells = []
    cursor = 0
    tables = (
        polyhedral_star_tables(connectivity)
        if isinstance(connectivity, PolyhedralConnectivity)
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
            values, cells, signs, edges, incident = _dense_block_facets(
                rows, block.cell_kind, dimension, owners
            )
            facet_rows.extend(values)
            facet_cells.extend(cells)
            facet_signs.extend(signs)
            if edges is not None and incident is not None:
                edge_rows.append(edges)
                edge_cells.append(incident)
        cursor += count
    return _assemble_complex(
        cell_rows, facet_rows, facet_cells, facet_signs, edge_rows, edge_cells
    )


def _assemble_complex(
    cell_rows: list[np.ndarray],
    facet_rows: list[np.ndarray],
    facet_cells: list[np.ndarray],
    facet_signs: list[np.ndarray],
    edge_rows: list[np.ndarray],
    edge_cells: list[np.ndarray],
    /,
) -> _Complex:
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


def _supplemented_star_complex(
    complex_: _Complex,
    supplement: tuple[tuple[str, np.ndarray], ...],
    dimension: int,
    /,
) -> _Complex:
    """Append non-resident dense star cells, keyed like the welded complex."""

    cell_rows = [complex_.cells]
    facet_rows = [complex_.facets]
    facet_cells = [complex_.facet_cells]
    facet_signs = [complex_.facet_signs]
    edge_rows = [complex_.edges]
    edge_cells = [complex_.edge_cells]
    cursor = complex_.cells.shape[0]
    for cell_kind, rows in supplement:
        owners = np.arange(cursor, cursor + rows.shape[0])
        cell_rows.append(rows)
        values, cells, signs, edges, incident = _dense_block_facets(
            rows, cell_kind, dimension, owners
        )
        facet_rows.extend(values)
        facet_cells.extend(cells)
        facet_signs.extend(signs)
        if edges is not None and incident is not None:
            edge_rows.append(edges)
            edge_cells.append(incident)
        cursor += rows.shape[0]
    return _assemble_complex(
        cell_rows, facet_rows, facet_cells, facet_signs, edge_rows, edge_cells
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
    scope: np.ndarray | None = None,
    /,
) -> int:
    """Count entities whose incident cells split into several link components.

    ``links`` holds ``(entity keys, first cell, second cell)``: the two cells
    share a facet containing that entity. ``scope`` restricts the count to
    entities with at least one vertex key in it.
    """

    keys, first_cells, second_cells = links
    if scope is not None:
        kept = np.any(np.isin(entity_rows, scope), axis=1)
        entity_rows = entity_rows[kept]
        entity_cells = entity_cells[kept]
        linked = np.any(np.isin(keys, scope), axis=1)
        keys = keys[linked]
        first_cells = first_cells[linked]
        second_cells = second_cells[linked]
    if entity_rows.shape[0] == 0:
        return 0
    nodes = np.unique(
        np.concatenate((entity_rows, entity_cells[:, None]), axis=1), axis=0
    )
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
    return int(np.count_nonzero(components > 1))


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
        raise ValueError("Welded star links reference unknown incidences.")
    return position


def _facet_links(inverse: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Consecutive occurrence pairs of every shared facet group."""

    order = np.argsort(inverse, kind="stable")
    same = inverse[order[1:]] == inverse[order[:-1]]
    return order[:-1][same], order[1:][same]


def _vertex_links(complex_: _Complex, first: np.ndarray, second: np.ndarray, /) -> Any:
    facets = complex_.facets[first]
    slots = facets >= 0
    return (
        facets[slots][:, None],
        np.broadcast_to(complex_.facet_cells[first][:, None], facets.shape)[slots],
        np.broadcast_to(complex_.facet_cells[second][:, None], facets.shape)[slots],
    )


def _edge_links(complex_: _Complex, first: np.ndarray, second: np.ndarray, /) -> Any:
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


def _open_boundary_count(
    complex_: _Complex, counts: np.ndarray, inverse: np.ndarray, embedded: bool, /
) -> int:
    boundary = counts[inverse] == 1
    if embedded:
        return int(np.count_nonzero(counts == 1))
    facets = complex_.facets[boundary]
    if facets.shape[1] == 1:
        return 0
    if facets.shape[1] == 2:
        ridges = np.concatenate((facets[:, :1], facets[:, 1:2]))
    else:
        lengths = np.sum(facets >= 0, axis=1)
        column = np.arange(facets.shape[1])
        following = facets[
            np.arange(facets.shape[0])[:, None], (column + 1) % lengths[:, None]
        ]
        slots = column[None, :] < lengths[:, None]
        ridges = np.sort(np.stack((facets[slots], following[slots]), axis=1), axis=1)
    if ridges.shape[0] == 0:
        return 0
    _, ridge_counts = np.unique(ridges, axis=0, return_counts=True)
    return int(np.count_nonzero(ridge_counts != 2))


# Self-intersection ---------------------------------------------------------------


def _signs(result: Any) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(result.signs, dtype=np.int8), np.asarray(
        result.certain, dtype=np.bool_
    )


# Meshcore adaptive predicates are used when available; the exact dyadic host
# route resolves filtered-uncertain signs otherwise.
def _orient3d(a: Any, b: Any, c: Any, d: Any, /) -> tuple[np.ndarray, np.ndarray]:
    mode = resolve_host_predicate_mode(PredicateMode.EXACT)
    return _signs(orient3d(a, b, c, d, mode=mode))


def _orient2d(a: Any, b: Any, c: Any, /) -> tuple[np.ndarray, np.ndarray]:
    mode = resolve_host_predicate_mode(PredicateMode.EXACT)
    return _signs(orient2d(a, b, c, mode=mode))


def _segment_crosses_triangle(
    segment: Any, triangle: Any, /
) -> tuple[np.ndarray, np.ndarray]:
    """Closed segment/triangle contact for segments not lying in the plane."""

    a, b = segment
    p, q, r = triangle
    side_a, certain_a = _orient3d(p, q, r, a)
    side_b, certain_b = _orient3d(p, q, r, b)
    first, certain_1 = _orient3d(a, b, p, q)
    second, certain_2 = _orient3d(a, b, q, r)
    third, certain_3 = _orient3d(a, b, r, p)
    edges = np.stack((first, second, third), axis=1).astype(np.int16)
    straddles = (side_a.astype(np.int16) * side_b <= 0) & ~((side_a == 0) & (side_b == 0))
    inside = np.all(edges >= 0, axis=1) | np.all(edges <= 0, axis=1)
    known = np.stack((certain_1, certain_2, certain_3), axis=1)
    plane_known = certain_a & certain_b
    # A certified plane miss, or two certified opposite edge signs, decides the
    # contact whatever the remaining signs are.
    missed = plane_known & (side_a.astype(np.int16) * side_b > 0)
    separated = np.any(known & (edges > 0), axis=1) & np.any(known & (edges < 0), axis=1)
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


def _inside_triangle_2d(point: Any, triangle: Any, /, *, strict: bool) -> Any:
    p, q, r = triangle
    signs = []
    certainties = []
    for start, stop in ((p, q), (q, r), (r, p)):
        sign, known = _orient2d(start, stop, point)
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


def _coplanar_overlap(
    first: np.ndarray, second: np.ndarray, shared: np.ndarray, /
) -> Any:
    """Closed coplanar triangle overlap away from shared vertices.

    Pairs sharing a vertex are reordered so that it sits at local index zero of
    both triangles; contact at that vertex is then not an intersection.
    """

    axes = _projection_axes(first)
    one = np.stack([_project(first[:, index], axes) for index in range(3)], axis=1)
    two = np.stack([_project(second[:, index], axes) for index in range(3)], axis=1)
    hit = np.zeros(first.shape[0], dtype=np.bool_)
    certain = np.ones(first.shape[0], dtype=np.bool_)
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
            one[:, index], (two[:, 0], two[:, 1], two[:, 2]), strict=True
        )
        inside_one, known_one = _inside_triangle_2d(
            two[:, index], (one[:, 0], one[:, 1], one[:, 2]), strict=True
        )
        hit |= inside_two | inside_one
        certain &= known_two & known_one
    return hit, certain


def _triangle_pairs_intersect(
    points: np.ndarray, first: np.ndarray, second: np.ndarray, /
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
    plane_signs = []
    plane_known = np.ones(first.shape[0], dtype=np.bool_)
    for index in range(3):
        sign, known = _orient3d(
            tri_one[:, 0], tri_one[:, 1], tri_one[:, 2], tri_two[:, index]
        )
        plane_signs.append(sign)
        plane_known &= known
    coplanar = np.all(np.stack(plane_signs, axis=1) == 0, axis=1)
    certain &= plane_known

    edge_share = shared == 2
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
            _project(a, axes), _project(b, axes), _project(c, axes)
        )
        side_d, known_d = _orient2d(
            _project(a, axes), _project(b, axes), _project(d, axes)
        )
        folded = coplanar[rows] & (side_c == side_d) & (side_c != 0)
        hit[rows] = folded
        certain[rows] &= known_c & known_d
    general = shared <= 1
    if np.any(general & coplanar):
        rows = np.flatnonzero(general & coplanar)
        overlap, known = _coplanar_overlap(tri_one[rows], tri_two[rows], shared[rows])
        hit[rows] = overlap
        certain[rows] &= known
    crossing_rows = np.flatnonzero(general & ~coplanar)
    if crossing_rows.size:
        a = tri_one[crossing_rows]
        b = tri_two[crossing_rows]
        start = np.where(shared[crossing_rows] == 1, 1, 0)
        contact = np.zeros(crossing_rows.size, dtype=np.bool_)
        known = np.ones(crossing_rows.size, dtype=np.bool_)
        for source, target in ((a, b), (b, a)):
            for index in range(3):
                begin = source[:, index]
                end = source[:, (index + 1) % 3]
                # A shared vertex (local zero) touches the other triangle by
                # construction; only the opposite edge carries new contact.
                relevant = (start == 0) | (index == 1)
                result, certain_ = _segment_crosses_triangle(
                    (begin, end), (target[:, 0], target[:, 1], target[:, 2])
                )
                contact |= relevant & result
                known &= ~relevant | certain_
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
) -> np.ndarray:
    """Certified disjointness beyond shared features from one coordinate projection.

    On a coordinate plane where both triangles project nondegenerately each
    triangle is a graph, so projections meeting only at shared features imply
    3D intersection only there, whatever the exact coplanarity. This decides
    near-coplanar pairs (flat sheets) left unresolved by filtered 3D signs.
    """
    axes = _projection_axes(tri_one)
    projected_one = np.stack([_project(tri_one[:, index], axes) for index in range(3)], 1)
    projected_two = np.stack([_project(tri_two[:, index], axes) for index in range(3)], 1)
    sign_one, known_one = _orient2d(
        projected_one[:, 0], projected_one[:, 1], projected_one[:, 2]
    )
    sign_two, known_two = _orient2d(
        projected_two[:, 0], projected_two[:, 1], projected_two[:, 2]
    )
    graphs = known_one & known_two & (sign_one != 0) & (sign_two != 0)
    disjoint = np.zeros(first.shape[0], dtype=np.bool_)
    general = np.flatnonzero(graphs & (shared <= 1))
    if general.size:
        overlap, known = _coplanar_overlap(
            tri_one[general], tri_two[general], shared[general]
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
        side_one, known_side_one = _orient2d(start, stop, _project(free_one, local_axes))
        side_two, known_side_two = _orient2d(start, stop, _project(free_two, local_axes))
        opposite = side_one.astype(np.int16) * side_two < 0
        disjoint[edge_rows] = known_side_one & known_side_two & opposite
    return disjoint


_EAR_WORKING_ENTRIES = 1 << 20


def _ear_blocked(
    loops: Any,
    before: Any,
    after: Any,
    alive: Any,
    predecessor: Any,
    successor: Any,
    sign: Any,
    mode: Any,
    /,
) -> Any:
    """Candidate ears containing another live vertex, or not decided exactly."""

    count, corner_count, _ = loops.shape
    apex = loops[:, :, None]
    left = before[:, :, None]
    right = after[:, :, None]
    other = loops[:, None, :]
    inside = np.ones((count, corner_count, corner_count), dtype=np.bool_)
    known = np.ones((count, corner_count, corner_count), dtype=np.bool_)
    for start, stop in ((left, apex), (apex, right), (right, left)):
        side = orient2d(start, stop, other, mode=mode)
        inside &= np.asarray(side.signs, dtype=np.int16) * sign[:, :, None] >= 0
        known &= np.asarray(side.certain)
    local = np.arange(corner_count)
    candidate = (
        alive[:, None, :]
        & (local[None, None, :] != local[None, :, None])
        & (local[None, None, :] != predecessor[:, :, None])
        & (local[None, None, :] != successor[:, :, None])
    )
    return np.any(candidate & (inside | ~known), axis=2)


def _ear_clip(
    loops: np.ndarray, orientation: np.ndarray, mode: PredicateMode, /
) -> tuple[np.ndarray, np.ndarray]:
    """Exact ear clipping of simple planar loops ``(count, n, 2)``.

    An ear is a strictly convex live vertex whose closed triangle holds no other
    live vertex; simple loops always have one (Meisters' two-ears theorem).
    Returns local triangles ``(count, n - 2, 3)`` and whether every ear decision
    of a loop was certified; uncertified loops must be discarded.
    """

    count, corner_count, _ = loops.shape
    rows = np.arange(count)
    local = np.arange(corner_count)
    successor = np.tile((local + 1) % corner_count, (count, 1))
    predecessor = np.tile((local - 1) % corner_count, (count, 1))
    alive = np.ones((count, corner_count), dtype=np.bool_)
    certified = np.ones((count,), dtype=np.bool_)
    triangles = np.empty((count, corner_count - 2, 3), dtype=np.int64)
    sign = orientation.astype(np.int16)[:, None]
    for step in range(corner_count - 3):
        before = loops[rows[:, None], predecessor]
        after = loops[rows[:, None], successor]
        turn = orient2d(before, loops, after, mode=mode)
        convex = (
            alive
            & np.asarray(turn.certain)
            & (np.asarray(turn.signs, dtype=np.int16) * sign > 0)
        )
        ear = convex & ~_ear_blocked(
            loops, before, after, alive, predecessor, successor, sign, mode
        )
        found = np.any(ear, axis=1)
        certified &= found
        choice = np.argmax(np.where(found[:, None], ear, alive), axis=1)
        left = predecessor[rows, choice]
        right = successor[rows, choice]
        triangles[:, step] = np.stack((left, choice, right), axis=1)
        alive[rows, choice] = False
        successor[rows, left] = right
        predecessor[rows, right] = left
    last = np.argmax(alive, axis=1)
    middle = successor[rows, last]
    triangles[:, -1] = np.stack((last, middle, successor[rows, middle]), axis=1)
    return triangles, certified


def _polygon_loop_triangles(
    points: np.ndarray, loops: np.ndarray, /
) -> tuple[np.ndarray, int, int]:
    """Exact triangles and complete simplicity findings for polygon loops."""

    corner_count = loops.shape[1]
    mode = resolve_host_predicate_mode(PredicateMode.EXACT)
    corners = points[loops]
    centered = corners - np.mean(corners, axis=1, keepdims=True)
    normal = np.sum(np.cross(centered, np.roll(centered, -1, axis=1)), axis=1)
    axes = np.asarray(((1, 2), (0, 2), (0, 1)))[np.argmax(np.abs(normal), axis=1)]
    planar = np.take_along_axis(corners, axes[:, None, :], axis=2)
    simplicity = polygon_simplicity_2d(planar, mode=mode)
    status = np.asarray(simplicity.status)
    orientation = np.asarray(simplicity.orientation)
    ready = orientation != PredicateSign.UNCERTAIN
    triangles = []
    undecided = int(np.count_nonzero(status == PolygonSimplicityStatus.UNCERTAIN))
    undecided += int(
        np.count_nonzero((status == PolygonSimplicityStatus.SIMPLE) & ~ready)
    )
    selected = np.flatnonzero(ready)
    chunk = max(1, _EAR_WORKING_ENTRIES // (corner_count * corner_count))
    for start in range(0, selected.size, chunk):
        rows = selected[start : start + chunk]
        local, certified = _ear_clip(planar[rows], orientation[rows], mode)
        undecided += int(np.count_nonzero(~certified))
        global_rows = loops[rows[certified]]
        cells = np.arange(global_rows.shape[0])[:, None, None]
        triangles.append(global_rows[cells, local[certified]].reshape((-1, 3)))
    intersecting = int(
        np.count_nonzero(status == PolygonSimplicityStatus.SELF_INTERSECTING)
    )
    return (
        np.concatenate(triangles) if triangles else np.empty((0, 3), dtype=np.int64),
        intersecting,
        undecided,
    )


def _surface_triangles(
    points: np.ndarray,
    complex_: _Complex,
    boundary: np.ndarray | None,
    /,
) -> tuple[np.ndarray, int, int]:
    """Triangulate every tested polygon without truncating candidate work."""

    loops = complex_.facets[boundary] if boundary is not None else complex_.cells
    lengths = np.sum(loops >= 0, axis=1)
    triangles = []
    intersecting = 0
    undecided = 0
    for length in np.unique(lengths):
        rows = loops[lengths == length][:, :length]
        if length >= 5:
            collapsed = _collapsed(rows)
            polygon_rows = rows[~collapsed]
            if polygon_rows.size:
                clipped, crossing, unknown = _polygon_loop_triangles(points, polygon_rows)
                triangles.append(clipped)
                intersecting += crossing
                undecided += unknown
            rows = rows[collapsed]
        for index in range(1, length - 1):
            triangles.append(rows[:, (0, index, index + 1)])
    return (
        np.concatenate(triangles) if triangles else np.empty((0, 3), dtype=np.int64),
        intersecting,
        undecided,
    )


def _self_intersections(
    points: np.ndarray,
    complex_: _Complex,
    boundary: np.ndarray,
    dimension: int,
    /,
) -> tuple[int, int]:
    """Return complete intersection and unresolved-predicate counts."""

    if points.shape[1] == 2:
        points = np.pad(points, ((0, 0), (0, 1)))
    match (points.shape[1], dimension):
        case (3, 2):
            triangles, loops, undecided = _surface_triangles(points, complex_, None)
        case (3, 3):
            triangles, loops, undecided = _surface_triangles(points, complex_, boundary)
        case _:
            return 0, 0
    triangles = triangles[~_collapsed(triangles)]
    if triangles.shape[0] < 2:
        return loops, undecided
    corners = points[triangles]
    bvh = prepare_bvh(np.min(corners, axis=1), np.max(corners, axis=1), dtype=np.float64)
    intersections = loops
    for first, second in bvh_overlap_pair_blocks(bvh, bvh, include_touching=True):
        keep = first < second
        first = first[keep]
        second = second[keep]
        if first.size:
            hit, certain = _triangle_pairs_intersect(
                points, triangles[first], triangles[second]
            )
            intersections += int(np.count_nonzero(hit & certain))
            undecided += int(np.count_nonzero(~certain))
    return intersections, undecided


# Entry point ---------------------------------------------------------------------


def audit_welded_topology(
    mesh: CellMesh,
    points: np.ndarray,
    /,
    *,
    coincident_tolerance: float | None,
    check_manifold: bool,
    check_watertight: bool,
    check_self_intersection: bool,
    manifold_vertices: np.ndarray | None = None,
    star_supplement: tuple[tuple[str, np.ndarray], ...] = (),
) -> WeldedTopologyEvidence:
    """Evaluate the welded-topology checks of one mesh with host coordinates.

    ``coincident_tolerance`` is relative to the bounding-box diagonal; ``None``
    disables welding. ``manifold_vertices`` (mesh vertex rows) restricts the
    vertex and edge star checks to the stars of those vertices;
    ``star_supplement`` adds dense ``(cell kind, vertex keys)`` blocks of star
    cells absent from ``mesh``, where keys below the vertex count are mesh rows
    and larger keys name non-resident vertices. Supplementary cells only enter
    the star checks.
    """

    if coincident_tolerance is None:
        welded, coincident = np.arange(points.shape[0]), 0
    else:
        diagonal = float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))
        welded, coincident = _coincident_representatives(
            points, coincident_tolerance * diagonal
        )
    unresolved: list[str] = []
    complex_ = _welded_complex(mesh, welded)
    keys = _sorted_keys(complex_.facets)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    first, second = _facet_links(inverse)
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
        star = complex_
        star_first, star_second = first, second
        if star_supplement:
            vertex_count = points.shape[0]
            star = _supplemented_star_complex(
                complex_,
                tuple(
                    (
                        cell_kind,
                        np.where(
                            rows < vertex_count,
                            welded[np.minimum(rows, vertex_count - 1)],
                            rows,
                        ),
                    )
                    for cell_kind, rows in star_supplement
                ),
                mesh.topological_dimension,
            )
            _, star_inverse = np.unique(
                _sorted_keys(star.facets), axis=0, return_inverse=True
            )
            star_first, star_second = _facet_links(star_inverse.reshape(-1))
        scope = (
            None
            if manifold_vertices is None
            else np.unique(welded[np.asarray(manifold_vertices, dtype=np.int64)])
        )
        cell_vertices = star.cells
        slots = cell_vertices >= 0
        owners = np.broadcast_to(
            np.arange(cell_vertices.shape[0])[:, None], cell_vertices.shape
        )[slots]
        nonmanifold_vertices = _star_components(
            cell_vertices[slots][:, None],
            owners,
            _vertex_links(star, star_first, star_second),
            scope,
        )
        if mesh.topological_dimension == 3:
            nonmanifold_edges = _star_components(
                np.sort(star.edges, axis=1),
                star.edge_cells,
                _edge_links(star, star_first, star_second),
                scope,
            )
    embedded = mesh.ambient_dimension > mesh.topological_dimension
    open_facets = (
        _open_boundary_count(complex_, counts, inverse, embedded)
        if check_watertight
        else 0
    )
    intersections = 0
    if check_self_intersection:
        boundary = counts[inverse] == 1
        intersections, uncertain = _self_intersections(
            points[welded],
            complex_,
            boundary,
            mesh.topological_dimension,
        )
        if uncertain:
            unresolved.append("self_intersection_predicates")
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


__all__ = ["WeldedTopologyEvidence", "audit_welded_topology"]
