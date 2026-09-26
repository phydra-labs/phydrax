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

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .._bvh import bvh_overlap_pairs_host, prepare_bvh
from ..discretization import CellMesh, PolyhedralConnectivity
from ..discretization._cell_geometry_validity import polyhedral_star_tables
from ..discretization._reference_cell import reference_cell_topology
from ..discretization.spatial._morton import morton_encode_integer
from ..geometry._predicates import (
    orient2d,
    orient3d,
    PredicateMode,
    resolve_host_predicate_mode,
)


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
    lower = np.min(points, axis=0)
    extent = float(np.max(np.max(points, axis=0) - lower))
    bits = _MORTON_BITS[dimension]
    width = max(tolerance, extent / float(1 << bits), np.finfo(np.float64).tiny)
    cells = np.minimum(
        np.floor((points - lower) / width).astype(np.int64), (1 << bits) - 2
    )
    codes = np.asarray(morton_encode_integer(cells, bits)).astype(np.uint64)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    firsts = []
    seconds = []
    total = 0
    for offset in np.ndindex(*(3,) * dimension):
        neighbor = cells + np.asarray(offset) - 1
        inside = np.all(neighbor >= 0, axis=1)
        neighbor_codes = np.asarray(
            morton_encode_integer(np.maximum(neighbor, 0), bits)
        ).astype(np.uint64)
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
            selected = np.isin(tables.star_cell, owners)
            vertices = _csr_rows(
                np.concatenate(((0,), np.cumsum(tables.cell_vertex_counts))),
                tables.cell_vertex_values,
            )[owners]
            cell_rows.append(np.where(vertices >= 0, welded[np.maximum(vertices, 0)], -1))
            faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
            face_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
            signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.float64)
            incidence_cell = np.repeat(
                np.arange(face_offsets.size - 1), np.diff(face_offsets)
            )
            chosen = np.isin(incidence_cell, owners)
            loops = _csr_rows(
                np.asarray(connectivity.face_vertex_offsets, dtype=np.int64),
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
                        (tables.star_first[selected], tables.star_second[selected]), 1
                    )
                ]
            )
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
    /,
) -> int:
    """Count entities whose incident cells split into several link components.

    ``links`` holds ``(entity keys, first cell, second cell)``: the two cells
    share a facet containing that entity.
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


def _vertex_links(complex_: _Complex, first: np.ndarray, second: np.ndarray, /):
    facets = complex_.facets[first]
    slots = facets >= 0
    return (
        facets[slots][:, None],
        np.broadcast_to(complex_.facet_cells[first][:, None], facets.shape)[slots],
        np.broadcast_to(complex_.facet_cells[second][:, None], facets.shape)[slots],
    )


def _edge_links(complex_: _Complex, first: np.ndarray, second: np.ndarray, /):
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


def _signs(result) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(result.signs, dtype=np.int8), np.asarray(
        result.certain, dtype=np.bool_
    )


# Exact predicates when meshcore is available, otherwise the float64 filter
# whose unresolved signs surface as an explicit unresolved disposition.
def _orient3d(a, b, c, d, /) -> tuple[np.ndarray, np.ndarray]:
    mode = resolve_host_predicate_mode(PredicateMode.EXACT)
    return _signs(orient3d(a, b, c, d, mode=mode))


def _orient2d(a, b, c, /) -> tuple[np.ndarray, np.ndarray]:
    mode = resolve_host_predicate_mode(PredicateMode.EXACT)
    return _signs(orient2d(a, b, c, mode=mode))


def _segment_crosses_triangle(segment, triangle, /) -> tuple[np.ndarray, np.ndarray]:
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
    certain = certain_a & certain_b & certain_1 & certain_2 & certain_3
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


def _segments_intersect_2d(
    a, b, c, d, /, *, proper: bool
) -> tuple[np.ndarray, np.ndarray]:
    first, certain_1 = _orient2d(a, b, c)
    second, certain_2 = _orient2d(a, b, d)
    third, certain_3 = _orient2d(c, d, a)
    fourth, certain_4 = _orient2d(c, d, b)
    certain = certain_1 & certain_2 & certain_3 & certain_4
    if proper:
        crosses = (first.astype(np.int16) * second < 0) & (
            third.astype(np.int16) * fourth < 0
        )
        return crosses, certain
    straddle = (first.astype(np.int16) * second <= 0) & (
        third.astype(np.int16) * fourth <= 0
    )
    collinear = (first == 0) & (second == 0)
    overlap = np.all(
        (
            np.maximum(np.minimum(a, b), np.minimum(c, d))
            <= np.minimum(np.maximum(a, b), np.maximum(c, d))
        ),
        axis=1,
    )
    return np.where(collinear, overlap, straddle), certain


def _inside_triangle_2d(point, triangle, /, *, strict: bool):
    p, q, r = triangle
    signs = []
    certain = np.ones(point.shape[0], dtype=np.bool_)
    for start, stop in ((p, q), (q, r), (r, p)):
        sign, known = _orient2d(start, stop, point)
        signs.append(sign.astype(np.int16))
        certain &= known
    stacked = np.stack(signs, axis=1)
    if strict:
        inside = np.all(stacked > 0, axis=1) | np.all(stacked < 0, axis=1)
    else:
        inside = np.all(stacked >= 0, axis=1) | np.all(stacked <= 0, axis=1)
    return inside, certain


def _coplanar_overlap(first: np.ndarray, second: np.ndarray, shared: np.ndarray, /):
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
    for index in range(3):
        for other in range(3):
            adjacent = (index == 0 or (index + 1) % 3 == 0) and (
                other == 0 or (other + 1) % 3 == 0
            )
            result, known = _segments_intersect_2d(
                one[:, index],
                one[:, (index + 1) % 3],
                two[:, other],
                two[:, (other + 1) % 3],
                proper=False,
            )
            proper, proper_known = _segments_intersect_2d(
                one[:, index],
                one[:, (index + 1) % 3],
                two[:, other],
                two[:, (other + 1) % 3],
                proper=True,
            )
            contact = np.where(touching | (not adjacent), result, proper)
            hit |= contact
            certain &= known & proper_known
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
    return hit, certain


def _surface_triangles(complex_: _Complex, boundary: np.ndarray | None, /) -> np.ndarray:
    loops = complex_.facets[boundary] if boundary is not None else complex_.cells
    lengths = np.sum(loops >= 0, axis=1)
    triangles = []
    for length in np.unique(lengths):
        rows = loops[lengths == length]
        for index in range(1, length - 1):
            triangles.append(rows[:, (0, index, index + 1)])
    return np.concatenate(triangles) if triangles else np.empty((0, 3), dtype=np.int64)


def _self_intersections(
    points: np.ndarray, complex_: _Complex, boundary: np.ndarray, dimension: int, /
) -> tuple[int, int]:
    """Return (intersecting triangle pair count, unresolved pair count).

    Surfaces embedded in 3-D test their own triangles, volume meshes their
    welded boundary, and planar meshes their cells (lifted to z = 0, where every
    pair takes the coplanar overlap route).
    """

    match (points.shape[1], dimension):
        case (3, 2) | (2, 2):
            triangles = _surface_triangles(complex_, None)
        case (3, 3):
            triangles = _surface_triangles(complex_, boundary)
        case _:
            return 0, 0
    if points.shape[1] == 2:
        points = np.pad(points, ((0, 0), (0, 1)))
    triangles = triangles[~_collapsed(triangles)]
    if triangles.shape[0] < 2:
        return 0, 0
    corners = points[triangles]
    bvh = prepare_bvh(np.min(corners, axis=1), np.max(corners, axis=1), dtype=np.float64)
    first, second = bvh_overlap_pairs_host(bvh, bvh, include_touching=True)
    keep = first < second
    first = triangles[first[keep]]
    second = triangles[second[keep]]
    if first.shape[0] == 0:
        return 0, 0
    hit, certain = _triangle_pairs_intersect(points, first, second)
    return int(np.count_nonzero(hit & certain)), int(np.count_nonzero(~certain))


# Entry point ---------------------------------------------------------------------


def audit_welded_topology(
    mesh: CellMesh,
    points: np.ndarray,
    /,
    *,
    coincident_tolerance: float | None,
    candidate_capacity: int,
    check_manifold: bool,
    check_watertight: bool,
    check_self_intersection: bool,
) -> WeldedTopologyEvidence:
    """Evaluate the welded-topology checks of one mesh with host coordinates.

    ``coincident_tolerance`` is relative to the bounding-box diagonal; ``None``
    disables welding.
    """

    if coincident_tolerance is None:
        welded, coincident, exceeded = np.arange(points.shape[0]), 0, False
    else:
        diagonal = float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))
        welded, coincident, exceeded = _coincident_representatives(
            points, coincident_tolerance * diagonal, candidate_capacity
        )
    unresolved = ["coincident_vertex_capacity"] if exceeded else []
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
        cell_vertices = complex_.cells
        slots = cell_vertices >= 0
        owners = np.broadcast_to(
            np.arange(cell_vertices.shape[0])[:, None], cell_vertices.shape
        )[slots]
        nonmanifold_vertices = _star_components(
            cell_vertices[slots][:, None], owners, _vertex_links(complex_, first, second)
        )
        if mesh.topological_dimension == 3:
            nonmanifold_edges = _star_components(
                np.sort(complex_.edges, axis=1),
                complex_.edge_cells,
                _edge_links(complex_, first, second),
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
            points[welded], complex_, boundary, mesh.topological_dimension
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
