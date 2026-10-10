#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Screened Poisson reconstruction on an adaptive octree.

The cube around the samples is refined to the finest width ``h`` only in the
cells that contain a sample (Kazhdan and Hoppe 2013) and completed to a
2:1-balanced linear octree by `refined_octree_leaves`; every sample lies in a
finest leaf and the cells coarsen geometrically away from the samples, so the
unknowns scale with the sampled surface area rather than the bounding volume.

The indicator is the continuous piecewise-trilinear finite-element function on
the leaves. A leaf corner lying inside an edge or face of a coarser leaf is a
hanging node whose value is the trilinear interpolant of that coarser leaf, so
the unknowns are the remaining free nodes and every nodal value is ``P z`` for
the sparse prolongation ``P`` (constraint chains are resolved to free masters).
The reduced operator ``P^T (K + alpha S) P`` and right-hand side ``P^T D`` are
assembled on the host from the leaf element tensors and solved by the shared
native PCG.

Extraction splits every leaf whose corners change sign into tetrahedra coned
from the leaf center over a triangulation of its faces. A face whose center is
a node (finer neighbor) is split into its four quarter faces, each fanned from
its center; any other face is fanned from its center through its corners and
present edge midpoints. Both sides of a shared face therefore produce the same
triangles, and every face or body center takes the average of the nodal values
spanning it in a canonical order, which equals the conforming indicator there.
The piecewise-linear field is thus globally conforming across levels and its
level set is crack free.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ...discretization.spatial._level_octree import refined_octree_leaves
from ...discretization.spatial._morton import (
    morton_encode_integer_host,
    MortonAddressPlan,
)
from ...linalg import linear_status_message
from ._poisson import (
    _CORNERS,
    _shape_values,
    _SOLVER_STEPS_PER_GRID_EXTENT,
    _STIFFNESS,
    cell_divergence,
    iso_statistics,
    PoissonIndicator,
    PoissonSolveEvidence,
    solve_indicator,
    splat_field,
    TetrahedralField,
)


# Quadrupled lattice keys of depth-16 trees stay below 2**63.
_MAXIMUM_DEPTH = 16
# Half-lattice boundary points of a cell that are not corners: 12 edge midpoints
# and 6 face centers, in doubled cell units.
_HALF_POINTS = np.asarray(
    [
        offset
        for offset in itertools.product((0, 1, 2), repeat=3)
        if 1 in offset and any(value != 1 for value in offset)
    ],
    dtype=np.int64,
)
_HALF_WEIGHTS = _shape_values(0.5 * _HALF_POINTS.astype(np.float64))


def _half_slot(offset: list[int], /) -> int:
    return int(np.flatnonzero(np.all(_HALF_POINTS == np.asarray(offset), axis=1))[0])


def _face_candidates() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Static candidate face triangles of one leaf in quarter-cell coordinates.

    Returns vertex coordinates ``(c, 3, 3)`` in ``{0, ..., 4}``, the face axis
    ``(c,)``, the condition kind ``(c,)`` (0: unsplit face without the edge
    midpoint, 1: unsplit face with it, 2: split face), the half-point slot
    ``(c,)`` of the tested edge midpoint (``-1`` when none), and the half-point
    slot ``(c,)`` of the face center deciding whether the face is split.
    """

    vertices: list[list[list[int]]] = []
    axes: list[int] = []
    kinds: list[int] = []
    midpoints: list[int] = []
    centers: list[int] = []
    for axis, side in itertools.product(range(3), (0, 1)):
        first, second = (other for other in range(3) if other != axis)

        def point(u: int, v: int, /) -> list[int]:
            result = [0, 0, 0]
            result[axis], result[first], result[second] = 4 * side, u, v
            return result

        def half_slot(u: int, v: int, /) -> int:
            return _half_slot([value // 2 for value in point(u, v)])

        center = half_slot(2, 2)
        cycle = ((0, 0), (4, 0), (4, 4), (0, 4))
        for start, stop in zip(cycle, cycle[1:] + cycle[:1], strict=True):
            middle = ((start[0] + stop[0]) // 2, (start[1] + stop[1]) // 2)
            vertices.append([point(2, 2), point(*start), point(*stop)])
            vertices.append([point(2, 2), point(*start), point(*middle)])
            vertices.append([point(2, 2), point(*middle), point(*stop)])
            kinds += [0, 1, 1]
            midpoints += [half_slot(*middle)] * 3
        for low_u, low_v in itertools.product((0, 2), repeat=2):
            quarter = [
                (low_u + du, low_v + dv) for du, dv in ((0, 0), (2, 0), (2, 2), (0, 2))
            ]
            for start, stop in zip(quarter, quarter[1:] + quarter[:1], strict=True):
                vertices.append(
                    [point(low_u + 1, low_v + 1), point(*start), point(*stop)]
                )
                kinds.append(2)
                midpoints.append(-1)
        count = len(vertices) - len(axes)
        axes += [axis] * count
        centers += [center] * count
    return (
        np.asarray(vertices, dtype=np.int64),
        np.asarray(axes, dtype=np.int64),
        np.asarray(kinds, dtype=np.int64),
        np.asarray(midpoints, dtype=np.int64),
        np.asarray(centers, dtype=np.int64),
    )


(
    _FACE_VERTICES,
    _FACE_AXES,
    _FACE_KINDS,
    _FACE_MIDPOINTS,
    _FACE_CENTERS,
) = _face_candidates()


@dataclass(frozen=True, slots=True)
class _Octree:
    """Leaves, nodes, and hanging-node prolongation of a balanced octree."""

    origin: np.ndarray
    spacing: float
    depth: int
    corners: np.ndarray
    sizes: np.ndarray
    code_starts: np.ndarray
    node_keys: np.ndarray
    leaf_nodes: np.ndarray
    half_nodes: np.ndarray
    prolongation_nodes: np.ndarray
    prolongation_free: np.ndarray
    prolongation_weights: np.ndarray
    free_count: int
    hanging_count: int

    @property
    def node_count(self) -> int:
        return self.node_keys.shape[0]

    def node_lookup(self, lattice: np.ndarray, /) -> np.ndarray:
        """Node index of finest-lattice points, ``-1`` where no node exists."""

        extent = (1 << self.depth) + 1
        keys = (lattice[..., 0] * extent + lattice[..., 1]) * extent + lattice[..., 2]
        position = np.minimum(
            np.searchsorted(self.node_keys, keys), self.node_keys.shape[0] - 1
        )
        return np.where(self.node_keys[position] == keys, position, -1)

    def prolong(self, reduced: np.ndarray, /) -> np.ndarray:
        """Nodal values ``P z`` of reduced free-node values ``z``."""

        return np.bincount(
            self.prolongation_nodes,
            weights=self.prolongation_weights * reduced[self.prolongation_free],
            minlength=self.node_count,
        )

    def restrict(self, nodal: np.ndarray, /) -> np.ndarray:
        """Reduced dual values ``P^T b`` of nodal dual values ``b``."""

        return np.bincount(
            self.prolongation_free,
            weights=self.prolongation_weights * nodal[self.prolongation_nodes],
            minlength=self.free_count,
        )

    def locate(self, points: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
        """Containing leaves ``(n,)`` and their trilinear corner weights ``(n, 8)``."""

        scaled = (points - self.origin) / self.spacing
        cells = np.clip(np.floor(scaled).astype(np.int64), 0, (1 << self.depth) - 1)
        codes = morton_encode_integer_host(cells, self.depth)
        leaves = np.searchsorted(self.code_starts, codes, side="right") - 1
        local = (scaled - self.corners[leaves]) / self.sizes[leaves, None]
        return leaves, _shape_values(np.clip(local, 0.0, 1.0))


def _refinement(cells: np.ndarray, depth: int, /) -> list[np.ndarray]:
    """Level prefixes of the ancestors of every sample cell, root first."""

    refined = [
        np.unique(morton_encode_integer_host(np.unique(cells >> 1, axis=0), depth - 1))
    ]
    for _ in range(depth - 2):
        refined.insert(0, np.unique(refined[0] >> np.uint64(3)))
    return [np.zeros((1,), dtype=np.uint64), *refined]


def _resolved_constraints(
    hanging: np.ndarray,
    masters: np.ndarray,
    weights: np.ndarray,
    is_hanging: np.ndarray,
    depth: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Substitute hanging masters until every constraint uses free masters.

    A chain descends at least one level per substitution, so ``depth`` rounds
    suffice; duplicate (hanging, master) pairs are coalesced.
    """

    for _ in range(depth + 1):
        chained = is_hanging[masters]
        if not np.any(chained):
            node_count = is_hanging.shape[0]
            keys, inverse = np.unique(hanging * node_count + masters, return_inverse=True)
            return (
                keys // node_count,
                keys % node_count,
                np.bincount(inverse, weights=weights),
            )
        order = np.argsort(hanging, kind="stable")
        sorted_hanging = hanging[order]
        start = np.searchsorted(sorted_hanging, masters[chained], side="left")
        counts = np.searchsorted(sorted_hanging, masters[chained], side="right") - start
        owner = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
        position = order[
            np.repeat(start, counts)
            + np.arange(owner.size, dtype=np.int64)
            - np.repeat(np.cumsum(counts) - counts, counts)
        ]
        hanging = np.concatenate((hanging[~chained], hanging[chained][owner]))
        weights = np.concatenate(
            (weights[~chained], weights[chained][owner] * weights[position])
        )
        masters = np.concatenate((masters[~chained], masters[position]))
    raise RuntimeError(
        "Hanging-node constraints did not resolve within the octree depth."
    )


def build_octree(
    points: np.ndarray,
    spacing: float,
    padding_cells: int,
    maximum_unknowns: int,
    /,
) -> _Octree:
    """Refine, balance, number, and constrain the adaptive octree of the samples."""

    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    extent = float(np.max(upper - lower)) + 2.0 * padding_cells * spacing
    depth = max(2, int(np.ceil(np.log2(extent / spacing))))
    if depth > _MAXIMUM_DEPTH:
        raise ValueError(
            f"Poisson octree needs depth {depth}, above {_MAXIMUM_DEPTH}; "
            "increase sample_spacing."
        )
    side = spacing * float(1 << depth)
    origin = 0.5 * (lower + upper) - 0.5 * side
    address = MortonAddressPlan(tuple(origin), tuple(origin + side), depth)
    cells = np.clip(
        np.floor((points - origin) / spacing).astype(np.int64), 0, (1 << depth) - 1
    )
    _, levels, corners = refined_octree_leaves(
        address, _refinement(cells, depth), balanced=True
    )
    corners = corners.astype(np.int64)
    sizes = np.left_shift(np.int64(1), depth - levels.astype(np.int64))
    extent_nodes = (1 << depth) + 1
    lattice = corners[:, None, :] + _CORNERS[None, :, :] * sizes[:, None, None]
    keys = (lattice[..., 0] * extent_nodes + lattice[..., 1]) * extent_nodes + lattice[
        ..., 2
    ]
    node_keys, leaf_nodes = np.unique(keys.reshape((-1,)), return_inverse=True)
    leaf_nodes = leaf_nodes.reshape((-1, 8))
    half_lattice = (
        corners[:, None, :] + _HALF_POINTS[None, :, :] * (sizes // 2)[:, None, None]
    )
    half_keys = (
        half_lattice[..., 0] * extent_nodes + half_lattice[..., 1]
    ) * extent_nodes + half_lattice[..., 2]
    position = np.minimum(np.searchsorted(node_keys, half_keys), node_keys.size - 1)
    half_nodes = np.where(
        (node_keys[position] == half_keys) & (sizes[:, None] > 1), position, -1
    )
    leaf, slot = np.nonzero(half_nodes >= 0)
    hanging_nodes, first = np.unique(half_nodes[leaf, slot], return_index=True)
    leaf, slot = leaf[first], slot[first]
    is_hanging = np.zeros((node_keys.size,), dtype=np.bool_)
    is_hanging[hanging_nodes] = True
    free_count = node_keys.size - hanging_nodes.size
    if free_count > maximum_unknowns:
        raise ValueError(
            f"Poisson octree needs {free_count} unknowns, above "
            f"maximum_grid_nodes={maximum_unknowns}; increase sample_spacing or the "
            "node budget."
        )
    constraint_weights = _HALF_WEIGHTS[slot]
    has_weight = constraint_weights > 0.0
    hanging, masters, weights = _resolved_constraints(
        np.broadcast_to(hanging_nodes[:, None], (hanging_nodes.size, 8))[has_weight],
        leaf_nodes[leaf][has_weight],
        constraint_weights[has_weight],
        is_hanging,
        depth,
    )
    free_index = np.cumsum(~is_hanging) - 1
    free_nodes = np.flatnonzero(~is_hanging)
    nodes = np.concatenate((free_nodes, hanging))
    order = np.argsort(nodes, kind="stable")
    return _Octree(
        origin=origin,
        spacing=spacing,
        depth=depth,
        corners=corners,
        sizes=sizes,
        code_starts=morton_encode_integer_host(corners, depth),
        node_keys=node_keys,
        leaf_nodes=leaf_nodes,
        half_nodes=half_nodes,
        prolongation_nodes=nodes[order],
        prolongation_free=np.concatenate((free_index[free_nodes], free_index[masters]))[
            order
        ],
        prolongation_weights=np.concatenate(
            (np.ones((free_nodes.size,), dtype=np.float64), weights)
        )[order],
        free_count=free_count,
        hanging_count=hanging_nodes.size,
    )


def _prolonged_entries(
    tree: _Octree, nodes: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Entry owner, free index, and weight of ``P`` rows of every listed node."""

    start = np.searchsorted(tree.prolongation_nodes, nodes, side="left")
    counts = np.searchsorted(tree.prolongation_nodes, nodes, side="right") - start
    owner = np.repeat(np.arange(nodes.size, dtype=np.int64), counts)
    position = (
        np.repeat(start, counts)
        + np.arange(owner.size, dtype=np.int64)
        - np.repeat(np.cumsum(counts) - counts, counts)
    )
    return owner, tree.prolongation_free[position], tree.prolongation_weights[position]


def _coalesced(
    rows: np.ndarray, columns: np.ndarray, values: np.ndarray, size: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys, inverse = np.unique(rows * size + columns, return_inverse=True)
    return keys // size, keys % size, np.bincount(inverse, weights=values)


def _reduced_operator(
    tree: _Octree,
    leaves: np.ndarray,
    weights: np.ndarray,
    screening_weights: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coalesced ``P^T (K + alpha S) P`` entries on the free nodes."""

    widths = tree.spacing * tree.sizes.astype(np.float64)
    element_rows = np.broadcast_to(tree.leaf_nodes[:, :, None], (widths.size, 8, 8))
    element_columns = np.broadcast_to(tree.leaf_nodes[:, None, :], (widths.size, 8, 8))
    element_values = widths[:, None, None] * _STIFFNESS[None, :, :]
    sample_nodes = tree.leaf_nodes[leaves]
    sample_values = (
        screening_weights[:, None, None] * weights[:, :, None] * weights[:, None, :]
    )
    rows, columns, values = _coalesced(
        np.concatenate(
            (
                element_rows.reshape((-1,)),
                np.broadcast_to(sample_nodes[:, :, None], sample_values.shape).reshape(
                    (-1,)
                ),
            )
        ),
        np.concatenate(
            (
                element_columns.reshape((-1,)),
                np.broadcast_to(sample_nodes[:, None, :], sample_values.shape).reshape(
                    (-1,)
                ),
            )
        ),
        np.concatenate((element_values.reshape((-1,)), sample_values.reshape((-1,)))),
        tree.node_count,
    )
    owner, free_columns, column_weights = _prolonged_entries(tree, columns)
    rows, values = rows[owner], values[owner] * column_weights
    owner, free_rows, row_weights = _prolonged_entries(tree, rows)
    return _coalesced(
        free_rows,
        free_columns[owner],
        values[owner] * row_weights,
        tree.free_count,
    )


def _node_averages(
    tree: _Octree,
    nodal: np.ndarray,
    leaves: np.ndarray,
    quarter: np.ndarray,
    axes: np.ndarray,
    /,
) -> np.ndarray:
    """Values at leaf points in quarter-cell coordinates ``{0, ..., 4}``.

    Nodes take their nodal value. A face or quarter-face center averages the
    four nodes at ``+-r`` along its in-face axes (``r = 2`` for even, ``r = 1``
    for odd coordinates) in global axis order, and the leaf center averages the
    eight leaf corners, so shared points evaluate identically from every leaf.
    """

    lattice4 = 4 * tree.corners[leaves] + quarter * tree.sizes[leaves][:, None]
    node = np.where(
        np.all(lattice4 % 4 == 0, axis=1), tree.node_lookup(lattice4 // 4), -1
    )
    values = np.where(node >= 0, nodal[np.maximum(node, 0)], 0.0)
    center = np.all(quarter == 2, axis=1)
    values = np.where(center, np.mean(nodal[tree.leaf_nodes[leaves]], axis=1), values)
    index = np.flatnonzero((node < 0) & ~center)
    radius = np.where(np.all(quarter[index] % 2 == 0, axis=1), 2, 1)
    in_face = np.sort(
        np.stack([(axes[index] + 1) % 3, (axes[index] + 2) % 3], axis=1), axis=1
    )
    rows = np.arange(index.size, dtype=np.int64)
    total = np.zeros((index.size,), dtype=np.float64)
    for sign_u, sign_v in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        offset = np.zeros((index.size, 3), dtype=np.int64)
        offset[rows, in_face[:, 0]] = sign_u * radius
        offset[rows, in_face[:, 1]] = sign_v * radius
        spanning = tree.node_lookup(
            (lattice4[index] + offset * tree.sizes[leaves[index]][:, None]) // 4
        )
        if np.any(spanning < 0):
            raise RuntimeError("A face center is not spanned by octree nodes.")
        total = total + nodal[spanning]
    values[index] = 0.25 * total
    return values


def _octree_field(tree: _Octree, nodal: np.ndarray, /) -> TetrahedralField:
    """Conforming tetrahedra of the leaves whose corner values change sign."""

    corner_inside = nodal[tree.leaf_nodes] < 0.0
    mixed = np.flatnonzero(np.any(corner_inside, axis=1) & ~np.all(corner_inside, axis=1))
    split = tree.half_nodes[mixed][:, _FACE_CENTERS] >= 0
    midpoint = np.where(
        _FACE_MIDPOINTS[None, :] >= 0,
        tree.half_nodes[mixed][:, np.maximum(_FACE_MIDPOINTS, 0)] >= 0,
        False,
    )
    valid = np.where(
        _FACE_KINDS[None, :] == 2,
        split,
        ~split & (midpoint == (_FACE_KINDS[None, :] == 1)),
    )
    owner, candidate = np.nonzero(valid)
    leaves = mixed[owner]
    quarter = np.concatenate(
        (
            np.full((leaves.size, 1, 3), 2, dtype=np.int64),
            _FACE_VERTICES[candidate],
        ),
        axis=1,
    ).reshape((-1, 3))
    vertex_leaves = np.repeat(leaves, 4)
    axes = np.repeat(_FACE_AXES[candidate], 4)
    values = _node_averages(tree, nodal, vertex_leaves, quarter, axes)
    lattice4 = (
        4 * tree.corners[vertex_leaves] + quarter * tree.sizes[vertex_leaves][:, None]
    )
    extent = 4 * (1 << tree.depth) + 1
    keys = (lattice4[:, 0] * extent + lattice4[:, 1]) * extent + lattice4[:, 2]
    unique_keys, first, tetrahedra = np.unique(
        keys, return_index=True, return_inverse=True
    )
    del unique_keys
    points = tree.origin + 0.25 * tree.spacing * lattice4[first].astype(np.float64)
    return TetrahedralField(points, values[first], tetrahedra.reshape((-1, 4)))


def solve_octree_poisson(
    points: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    /,
    *,
    spacing: float,
    screening: float,
    padding_cells: int,
    maximum_grid_nodes: int,
) -> PoissonIndicator:
    """Assemble and solve the screened Poisson indicator on an adaptive octree."""

    tree = build_octree(points, spacing, padding_cells, maximum_grid_nodes)
    leaves, weights = tree.locate(points)
    rows, columns, values = _reduced_operator(
        tree, leaves, weights, screening / spacing * areas
    )
    widths = tree.spacing * tree.sizes.astype(np.float64)
    field = splat_field(
        tree.leaf_nodes[leaves],
        weights,
        areas[:, None] * normals / widths[leaves, None] ** 3,
        tree.node_count,
    )
    rhs = tree.restrict(cell_divergence(tree.leaf_nodes, widths, field, tree.node_count))
    record = solve_indicator(
        tree.free_count,
        rows,
        columns,
        values,
        rhs,
        _SOLVER_STEPS_PER_GRID_EXTENT * (1 << tree.depth),
    )
    nodal = tree.prolong(record.values)
    sampled = np.sum(nodal[tree.leaf_nodes[leaves]] * weights, axis=1)
    iso, spread = iso_statistics(sampled, areas)
    lattice_nodes = (1 << tree.depth) + 1
    evidence = PoissonSolveEvidence(
        discretization="octree",
        grid_shape=(lattice_nodes, lattice_nodes, lattice_nodes),
        grid_origin=(
            float(tree.origin[0]),
            float(tree.origin[1]),
            float(tree.origin[2]),
        ),
        grid_spacing=spacing,
        octree_depth=tree.depth,
        unknowns=tree.free_count,
        leaf_cells=tree.sizes.shape[0],
        hanging_nodes=tree.hanging_count,
        screening=screening,
        represented_area=float(np.sum(areas)),
        solver_status=record.status,
        solver_message=linear_status_message(record.status),
        solver_converged=record.converged,
        solver_iterations=record.iterations,
        solver_relative_residual=record.relative_residual,
        iso_value=iso,
        iso_value_spread=spread,
    )
    return PoissonIndicator(sampled - iso, _octree_field(tree, nodal - iso), evidence)
