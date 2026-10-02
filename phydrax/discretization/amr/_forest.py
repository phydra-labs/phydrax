#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Forest-of-trees (quadtree/octree) AMR topology over a uniform root grid.

Every root block is one cell of an interval-primary uniform tensor grid, so the
roots form a Cartesian brick with optional periodic axes and aligned faces.  A leaf
is identified by its level and its global cell coordinate on that level's lattice
(``root_shape * 2**level``).  Leaves are stored as the canonical sorted Morton prefix
of fixed-capacity worksets: the order key is the row-major root index followed by
the Morton interleave of the leaf anchor at the finest admitted level, so every tree
node owns one contiguous key interval and point location is one binary search.

Host preparation (adaptation, balance closure, neighbor routes) is NumPy; the
execution view is the bucket-stable :class:`ForestLeafWorkset`, whose static
structure depends only on capacity buckets so compiled consumers are reused across
adaptation cycles that stay inside one bucket.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum, StrEnum
from math import prod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import checked
from .._tensor_support import PreparedTensorGrid
from .._topology_epoch import TopologyEpoch
from ..spatial import morton_encode_integer
from ._canonical import BlockAMRResourcePlan
from ._core import (
    BlockHierarchyPlan,
    BlockHierarchyTopology,
    BlockLevelPlan,
    canonical_block_metadata,
)
from ._cut_complex import EmbeddedLevelSetBodySet, MultivaluedCutCellPlan
from ._cut_complex_2d import MultivaluedCutCell2DPlan
from ._mapped_geometry import CanonicalMappedGeometryPlan, PatchCoordinateMapSet


class AMRBalanceStencil(StrEnum):
    """Neighborhood over which adjacent leaves may differ by at most one level."""

    FACE = "face"
    EDGE = "edge"
    CORNER = "corner"


class ForestFaceKind(IntEnum):
    """Per-leaf face relation to the adjacent region."""

    BOUNDARY = 0
    CONFORMING = 1
    COARSER = 2
    FINER = 3


def balance_directions(dimension: int, stencil: AMRBalanceStencil, /) -> np.ndarray:
    """Nonzero offsets in ``{-1, 0, 1}**d`` whose support fits the balance stencil.

    FACE admits one nonzero axis, EDGE at most two, CORNER all ``d``.  Offsets are
    returned as int64 rows in canonical row-major order of ``{-1, 0, 1}**d``.
    """
    dimension_ = int(dimension)
    if dimension_ <= 0:
        raise ValueError("Balance directions require a positive dimension.")
    match AMRBalanceStencil(stencil):
        case AMRBalanceStencil.FACE:
            codimension = 1
        case AMRBalanceStencil.EDGE:
            codimension = min(2, dimension_)
        case AMRBalanceStencil.CORNER:
            codimension = dimension_
        case _:
            raise ValueError("Unknown AMR balance stencil.")
    offsets = np.indices((3,) * dimension_).reshape(dimension_, -1).T - 1
    support = np.count_nonzero(offsets, axis=1)
    return offsets[(support >= 1) & (support <= codimension)].astype(np.int64)


def forest_capacity_bucket(count: int, minimum: int, /) -> int:
    """Power-of-two capacity bucket admitting ``count`` entries, floored at ``minimum``."""
    count_ = int(count)
    floor = int(minimum)
    if count_ < 0 or floor <= 0:
        raise ValueError("Forest capacity buckets require nonnegative counts.")
    return max(floor, 1 << max(count_ - 1, 0).bit_length())


class ForestPlan(StrictModule, NonTrainableState):
    """Static forest plan: root grid, depth limit, balance, and capacity policy.

    ``maps`` optionally maps global reference coordinates of the root grid to
    physical space; it must be a single default chart because every descendant of a
    root shares that root's reference coordinates.
    """

    grid: PreparedTensorGrid
    maps: PatchCoordinateMapSet | None
    maximum_level: int = eqx.field(static=True)
    balance: AMRBalanceStencil = eqx.field(static=True)
    minimum_leaf_capacity: int = eqx.field(static=True)
    maximum_leaf_capacity: int = eqx.field(static=True)
    root_shape: tuple[int, ...] = eqx.field(static=True)
    periodic_axes: tuple[bool, ...] = eqx.field(static=True)
    lower_bounds: tuple[float, ...] = eqx.field(static=True)
    root_spacing: tuple[float, ...] = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        grid: PreparedTensorGrid,
        /,
        *,
        maximum_level: int,
        balance: AMRBalanceStencil = AMRBalanceStencil.FACE,
        minimum_leaf_capacity: int = 8,
        maximum_leaf_capacity: int,
        maps: PatchCoordinateMapSet | None = None,
    ) -> None:
        if maps is not None and not isinstance(maps, PatchCoordinateMapSet):
            raise TypeError("Forest maps must be a PatchCoordinateMapSet or None.")
        if maps is not None and maps.patch_ids:
            raise ValueError(
                "Forest maps must use one default chart over global reference coordinates."
            )
        dimension = len(grid.shape)
        if dimension not in (1, 2, 3):
            raise ValueError(
                "Forest AMR supports one-, two-, and three-dimensional roots."
            )
        if any(axis.primary_entity != "interval" for axis in grid.axes):
            raise ValueError("Forest AMR requires an interval-primary tensor grid.")
        widths = tuple(
            np.asarray(axis.interval_widths, dtype=np.float64)
            for axis in grid.structured_axes
        )
        if any(
            axis.basis != "uniform" or width.size == 0 or not np.all(width == width[0])
            for axis, width in zip(grid.axes, widths, strict=True)
        ):
            raise ValueError("Forest AMR requires uniform tensor-grid axes.")
        level = int(maximum_level)
        minimum = int(minimum_leaf_capacity)
        maximum = int(maximum_leaf_capacity)
        root_shape = tuple(grid.shape)
        root_count = prod(root_shape)
        if level < 0:
            raise ValueError("Forest maximum level must be nonnegative.")
        if (root_count - 1).bit_length() + dimension * level + 1 > 62:
            raise ValueError("Forest tree-path identifiers exceed the int64 budget.")
        if minimum <= 0 or maximum < max(minimum, root_count):
            raise ValueError(
                "Forest leaf capacities must be positive and admit every root."
            )
        balance_ = AMRBalanceStencil(balance)
        lower = tuple(
            float(np.asarray(axis.bounds[0], dtype=np.float64))
            for axis in grid.structured_axes
        )
        spacing = tuple(float(width[0]) for width in widths)
        geometry_id = (
            grid.support.embedding_id
            if maps is None
            else canonical_fingerprint(
                {
                    "kind": "forest-mapped-geometry",
                    "embedding": grid.support.embedding_id,
                    "maps": maps.map_set_id,
                }
            )
        )
        self.grid = grid
        self.maps = maps
        self.maximum_level = level
        self.balance = balance_
        self.minimum_leaf_capacity = minimum
        self.maximum_leaf_capacity = maximum
        self.root_shape = root_shape
        self.periodic_axes = tuple(bool(axis.periodic) for axis in grid.axes)
        self.lower_bounds = lower
        self.root_spacing = spacing
        self.geometry_id = geometry_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "forest-plan",
                "grid": grid.prepared_id,
                "geometry": geometry_id,
                "maximum_level": level,
                "balance": balance_.value,
                "minimum_leaf_capacity": minimum,
                "maximum_leaf_capacity": maximum,
            }
        )

    @property
    def dimension(self) -> int:
        return len(self.root_shape)

    @property
    def root_count(self) -> int:
        return prod(self.root_shape)

    @property
    def children_per_node(self) -> int:
        return 1 << self.dimension

    def level_shape(self, level: int, /) -> tuple[int, ...]:
        """Global cell lattice shape of one refinement level."""
        level_ = int(level)
        if level_ < 0 or level_ > self.maximum_level:
            raise ValueError("Forest level is out of range.")
        return tuple(extent << level_ for extent in self.root_shape)


def _point_keys(plan: ForestPlan, points: np.ndarray, /) -> np.ndarray:
    """Morton order keys of in-domain points on the finest lattice."""
    rows = np.asarray(points, dtype=np.int64).reshape(-1, plan.dimension)
    if rows.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    depth = plan.maximum_level
    roots = np.ravel_multi_index(tuple((rows >> depth).T), plan.root_shape).astype(
        np.int64
    )
    if depth == 0:
        return roots
    local = rows & ((1 << depth) - 1)
    morton = np.asarray(
        morton_encode_integer(jnp.asarray(local, dtype=jnp.uint64), depth)
    ).astype(np.int64)
    return (roots << (plan.dimension * depth)) | morton


def _node_keys(plan: ForestPlan, levels: np.ndarray, coordinates: np.ndarray, /) -> Any:
    """Morton order keys of node anchors on the finest lattice."""
    shift = (plan.maximum_level - np.asarray(levels, dtype=np.int64))[:, None]
    return _point_keys(plan, np.asarray(coordinates, dtype=np.int64) << shift)


def _path_ids(plan: ForestPlan, levels: np.ndarray, coordinates: np.ndarray, /) -> Any:
    """Stable tree-path identifiers ``root << (d*L + 1) | 1 << (d*l) | morton``.

    The leading bit marks the depth, so every node at every admitted level has one
    identifier that is independent of the forest it currently belongs to.
    """
    levels_ = np.asarray(levels, dtype=np.int64)
    rows = np.asarray(coordinates, dtype=np.int64)
    dimension = plan.dimension
    depth = plan.maximum_level
    roots = np.ravel_multi_index(
        tuple((rows >> levels_[:, None]).T), plan.root_shape
    ).astype(np.int64)
    local = rows & ((np.int64(1) << levels_[:, None]) - 1)
    morton = (
        np.zeros(levels_.shape, dtype=np.int64)
        if depth == 0
        else np.asarray(
            morton_encode_integer(jnp.asarray(local, dtype=jnp.uint64), depth)
        ).astype(np.int64)
    )
    return (
        (roots << (dimension * depth + 1))
        | (np.int64(1) << (dimension * levels_))
        | morton
    )


def _locate(keys: np.ndarray, query: np.ndarray, /) -> np.ndarray:
    """Slots of the leaves containing each query key (keys partition the domain)."""
    return np.searchsorted(keys, query, side="right") - 1


def _shifted_cells(
    plan: ForestPlan,
    levels: np.ndarray,
    coordinates: np.ndarray,
    offsets: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Offset same-level cells with periodic wrap; returns ``(cells, inside)``."""
    extent = np.asarray(plan.root_shape, dtype=np.int64)[None, :] << levels[:, None]
    cells = coordinates + offsets
    periodic = np.asarray(plan.periodic_axes, dtype=np.bool_)
    cells = np.where(periodic, cells % extent, cells)
    inside = np.all((cells >= 0) & (cells < extent), axis=-1)
    return cells, inside


def _locate_cells(
    plan: ForestPlan,
    keys: np.ndarray,
    levels: np.ndarray,
    cells: np.ndarray,
    /,
) -> np.ndarray:
    """Slots of the leaves containing the anchors of in-domain level cells."""
    return _locate(keys, _node_keys(plan, levels, cells))


def _balance_violations(
    plan: ForestPlan,
    levels: np.ndarray,
    coordinates: np.ndarray,
    keys: np.ndarray,
    /,
) -> np.ndarray:
    """Sorted unique slots of leaves coarser than a stencil neighbor by two levels."""
    directions = balance_directions(plan.dimension, plan.balance)
    candidates = np.flatnonzero(levels >= 2)
    if candidates.size == 0:
        return np.zeros((0,), dtype=np.int64)
    source_levels = np.repeat(levels[candidates], directions.shape[0])
    cells, inside = _shifted_cells(
        plan,
        source_levels,
        np.repeat(coordinates[candidates], directions.shape[0], axis=0),
        np.tile(directions, (candidates.size, 1)),
    )
    slots = _locate_cells(plan, keys, source_levels[inside], cells[inside])
    return np.unique(slots[levels[slots] < source_levels[inside] - 1])


def _refine(
    levels: np.ndarray,
    coordinates: np.ndarray,
    refined: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = coordinates.shape[1]
    children = np.indices((2,) * dimension).reshape(dimension, -1).T.astype(np.int64)
    parents = coordinates[refined]
    child_coordinates = (2 * parents[:, None, :] + children[None, :, :]).reshape(
        -1, dimension
    )
    child_levels = np.repeat(levels[refined] + 1, children.shape[0])
    keep = ~refined
    return (
        np.concatenate((levels[keep], child_levels)),
        np.concatenate((coordinates[keep], child_coordinates)),
    )


def _canonical_order(
    plan: ForestPlan, levels: np.ndarray, coordinates: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = _node_keys(plan, levels, coordinates)
    order = np.argsort(keys, kind="stable")
    return levels[order], coordinates[order], keys[order]


def _close_balance(
    plan: ForestPlan,
    levels: np.ndarray,
    coordinates: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Ripple-refine coarse leaves until the forest is 2:1 balanced."""
    levels_, coordinates_, keys = _canonical_order(plan, levels, coordinates)
    refinements = 0
    while True:
        violating = _balance_violations(plan, levels_, coordinates_, keys)
        if violating.size == 0:
            return levels_, coordinates_, keys, refinements
        mask = np.zeros(levels_.shape, dtype=np.bool_)
        mask[violating] = True
        refinements += violating.size
        levels_, coordinates_ = _refine(levels_, coordinates_, mask)
        levels_, coordinates_, keys = _canonical_order(plan, levels_, coordinates_)


def _coarsen(
    plan: ForestPlan,
    levels: np.ndarray,
    coordinates: np.ndarray,
    keys: np.ndarray,
    marked: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Coarsen complete marked sibling families whose parents keep 2:1 balance.

    A family is admissible when every sibling is a marked leaf and no stencil
    neighbor of any sibling is finer than the siblings.  The veto is evaluated
    against the pre-coarsening forest, which is conservative for simultaneous
    coarsening because coarsening only lowers neighbor levels.
    """
    candidates = np.flatnonzero(marked & (levels >= 1))
    if candidates.size == 0:
        return levels, coordinates, 0, 0
    parent_levels = levels[candidates] - 1
    parent_coordinates = coordinates[candidates] >> 1
    family_keys = _path_ids(plan, parent_levels, parent_coordinates)
    unique, inverse, counts = np.unique(
        family_keys, return_inverse=True, return_counts=True
    )
    complete = counts[inverse] == plan.children_per_node
    directions = balance_directions(plan.dimension, plan.balance)
    source_levels = np.repeat(levels[candidates], directions.shape[0])
    cells, inside = _shifted_cells(
        plan,
        source_levels,
        np.repeat(coordinates[candidates], directions.shape[0], axis=0),
        np.tile(directions, (candidates.size, 1)),
    )
    finer = np.zeros(source_levels.shape, dtype=np.bool_)
    slots = _locate_cells(plan, keys, source_levels[inside], cells[inside])
    finer[inside] = levels[slots] > source_levels[inside]
    vetoed_member = np.any(finer.reshape(candidates.size, -1), axis=1)
    family_vetoed = np.zeros(unique.shape, dtype=np.bool_)
    np.logical_or.at(family_vetoed, inverse, vetoed_member)
    family_complete = np.zeros(unique.shape, dtype=np.bool_)
    family_complete[inverse[complete]] = True
    accepted_family = family_complete & ~family_vetoed
    accepted = accepted_family[inverse]
    removed = np.zeros(levels.shape, dtype=np.bool_)
    removed[candidates[accepted]] = True
    first = np.unique(inverse[accepted], return_index=True)[1]
    new_levels = parent_levels[accepted][first]
    new_coordinates = parent_coordinates[accepted][first]
    return (
        np.concatenate((levels[~removed], new_levels)),
        np.concatenate((coordinates[~removed], new_coordinates)),
        int(np.count_nonzero(accepted_family)),
        int(np.count_nonzero(family_complete & family_vetoed)),
    )


def _face_routes(
    plan: ForestPlan,
    levels: np.ndarray,
    coordinates: np.ndarray,
    keys: np.ndarray,
    /,
) -> tuple[np.ndarray, ...]:
    """Per-leaf face kinds/neighbors plus canonical interior and boundary faces.

    Returns ``kinds (n, d, 2)``, ``neighbors (n, d, 2, 2**(d-1))``, the interior
    face columns ``minus, plus, axis, level, coarse_fine`` (each face once, at the
    finer side's granularity, oriented along ``+axis``) and the boundary columns
    ``slot, axis, side``.
    """
    count = levels.shape[0]
    dimension = plan.dimension
    subfaces = 1 << (dimension - 1)
    kinds = np.full((count, dimension, 2), ForestFaceKind.BOUNDARY, dtype=np.int8)
    neighbors = np.full((count, dimension, 2, subfaces), -1, dtype=np.int32)
    slots = np.arange(count, dtype=np.int64)
    interior: list[tuple[np.ndarray, ...]] = []
    boundary: list[tuple[np.ndarray, ...]] = []
    for axis in range(dimension):
        tangential = [value for value in range(dimension) if value != axis]
        sub_offsets = np.zeros((subfaces, dimension), dtype=np.int64)
        if tangential:
            sub_offsets[:, tangential] = (
                np.indices((2,) * (dimension - 1)).reshape(dimension - 1, -1).T
            )
        for side, delta in ((0, -1), (1, 1)):
            offset = np.zeros((1, dimension), dtype=np.int64)
            offset[0, axis] = delta
            cells, inside = _shifted_cells(plan, levels, coordinates, offset)
            located = np.full((count,), -1, dtype=np.int64)
            located[inside] = _locate_cells(plan, keys, levels[inside], cells[inside])
            located_levels = np.where(inside, levels[located.clip(min=0)], -1)
            conforming = inside & (located_levels == levels)
            coarser = inside & (located_levels < levels)
            finer = inside & (located_levels > levels)
            if np.any(coarser & (located_levels != levels - 1)) or np.any(
                finer & (levels >= plan.maximum_level)
            ):
                raise ValueError("Forest face neighbors violate 2:1 face balance.")
            kinds[conforming, axis, side] = ForestFaceKind.CONFORMING
            kinds[coarser, axis, side] = ForestFaceKind.COARSER
            kinds[finer, axis, side] = ForestFaceKind.FINER
            neighbors[conforming | coarser, axis, side, 0] = located[conforming | coarser]
            if np.any(finer):
                fine_slots = np.flatnonzero(finer)
                sub_offsets_side = sub_offsets.copy()
                sub_offsets_side[:, axis] = 1 if side == 0 else 0
                sub_cells = (
                    2 * cells[fine_slots][:, None, :] + sub_offsets_side[None, :, :]
                ).reshape(-1, dimension)
                sub_levels = np.repeat(levels[fine_slots] + 1, subfaces)
                sub_located = _locate_cells(plan, keys, sub_levels, sub_cells)
                if np.any(levels[sub_located] != sub_levels):
                    raise ValueError("Forest face neighbors violate 2:1 face balance.")
                neighbors[fine_slots, axis, side, :] = sub_located.reshape(-1, subfaces)
            boundary.append(
                (
                    slots[~inside],
                    np.full((np.count_nonzero(~inside),), axis, dtype=np.int64),
                    np.full((np.count_nonzero(~inside),), side, dtype=np.int64),
                )
            )
            if side == 1:
                owned = conforming | coarser
                interior.append(
                    (
                        slots[owned],
                        located[owned],
                        np.full((np.count_nonzero(owned),), axis, dtype=np.int64),
                        levels[owned],
                        coarser[owned],
                    )
                )
            else:
                interior.append(
                    (
                        located[coarser],
                        slots[coarser],
                        np.full((np.count_nonzero(coarser),), axis, dtype=np.int64),
                        levels[coarser],
                        np.ones((np.count_nonzero(coarser),), dtype=np.bool_),
                    )
                )
    minus, plus, axes, face_levels, coarse_fine = (
        np.concatenate(column) for column in zip(*interior, strict=True)
    )
    order = np.lexsort((plus, minus, axes))
    boundary_slots, boundary_axes, boundary_sides = (
        np.concatenate(column) for column in zip(*boundary, strict=True)
    )
    boundary_order = np.lexsort((boundary_slots, boundary_sides, boundary_axes))
    return (
        kinds,
        neighbors,
        minus[order],
        plus[order],
        axes[order],
        face_levels[order],
        coarse_fine[order],
        boundary_slots[boundary_order],
        boundary_axes[boundary_order],
        boundary_sides[boundary_order],
    )


def _padded(values: np.ndarray, capacity: int, fill: Any, dtype: Any, /) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    result = np.full((capacity,) + array.shape[1:], fill, dtype=dtype)
    result[: array.shape[0]] = array
    return result


class ForestWorksetSignature(StrictModule, NonTrainableState):
    """Bucket-level static structure of one forest execution workset.

    Two worksets with equal signatures have identical array shapes and static
    metadata, so compiled consumers are reused across adaptation cycles.
    """

    dimension: int = eqx.field(static=True)
    maximum_level: int = eqx.field(static=True)
    leaf_capacity: int = eqx.field(static=True)
    face_capacity: int = eqx.field(static=True)
    boundary_capacity: int = eqx.field(static=True)
    level_capacities: tuple[int, ...] = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        maximum_level: int,
        leaf_capacity: int,
        face_capacity: int,
        boundary_capacity: int,
        level_capacities: Sequence[int],
        /,
    ) -> None:
        values = (
            int(dimension),
            int(maximum_level),
            int(leaf_capacity),
            int(face_capacity),
            int(boundary_capacity),
        )
        levels = tuple(int(value) for value in level_capacities)
        if (
            values[0] not in (1, 2, 3)
            or values[1] < 0
            or min(values[2:]) <= 0
            or len(levels) != values[1] + 1
            or any(value <= 0 for value in levels)
        ):
            raise ValueError("Forest workset signature capacities are invalid.")
        self.dimension = values[0]
        self.maximum_level = values[1]
        self.leaf_capacity = values[2]
        self.face_capacity = values[3]
        self.boundary_capacity = values[4]
        self.level_capacities = levels
        self.signature_id = canonical_fingerprint(
            {
                "kind": "forest-workset-signature",
                "values": values,
                "level_capacities": levels,
            }
        )


class ForestLeafWorkset(StrictModule, NonTrainableState):
    """Fixed-capacity leaf, face, boundary, and per-level execution arrays.

    Leaves occupy the canonical Morton prefix ``[0, leaf_count)``; padding uses
    ``-1`` indices and ``False`` validity.  Interior faces are oriented from
    ``face_minus`` to ``face_plus`` along ``+face_axes`` and live at the finer
    adjacent level (``face_levels``); ``face_coarse_fine`` marks hanging faces.
    Boundary sides use ``0`` for the lower and ``1`` for the upper physical side.
    """

    signature: ForestWorksetSignature
    leaf_valid: Array
    leaf_levels: Array
    leaf_coordinates: Array
    leaf_trees: Array
    leaf_path_ids: Array
    face_kinds: Array
    face_neighbors: Array
    face_valid: Array
    face_minus: Array
    face_plus: Array
    face_axes: Array
    face_levels: Array
    face_coarse_fine: Array
    boundary_valid: Array
    boundary_slots: Array
    boundary_axes: Array
    boundary_sides: Array
    level_slots: tuple[Array, ...]
    level_valid: tuple[Array, ...]


def _canonical_leaves(
    plan: ForestPlan,
    levels: ArrayLike,
    coordinates: ArrayLike,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate leaves as an exact 2:1-balanced partition; return Morton order."""
    levels_ = np.asarray(levels)
    coordinates_ = np.asarray(coordinates)
    if not np.issubdtype(levels_.dtype, np.integer) or not np.issubdtype(
        coordinates_.dtype, np.integer
    ):
        raise TypeError("Forest leaf levels and coordinates must be integers.")
    dimension = plan.dimension
    if (
        levels_.ndim != 1
        or coordinates_.shape != (levels_.shape[0], dimension)
        or levels_.shape[0] == 0
    ):
        raise ValueError("Forest leaves require (n,) levels and (n, d) coordinates.")
    levels_ = levels_.astype(np.int64)
    coordinates_ = coordinates_.astype(np.int64)
    if np.any(levels_ < 0) or np.any(levels_ > plan.maximum_level):
        raise ValueError("Forest leaf levels exceed the admitted depth.")
    extent = np.asarray(plan.root_shape, dtype=np.int64)[None, :] << levels_[:, None]
    if np.any(coordinates_ < 0) or np.any(coordinates_ >= extent):
        raise ValueError("Forest leaf coordinates lie outside their level lattice.")
    if levels_.shape[0] > plan.maximum_leaf_capacity:
        raise ValueError("Forest leaf count exceeds the maximum leaf capacity.")
    levels_, coordinates_, keys = _canonical_order(plan, levels_, coordinates_)
    # Leaves partition the domain exactly iff consecutive Morton intervals tile
    # the full key range without gaps or overlaps.
    spans = np.int64(1) << (dimension * (plan.maximum_level - levels_))
    total = np.int64(plan.root_count) << (dimension * plan.maximum_level)
    if (
        keys[0] != 0
        or np.any(keys[1:] != keys[:-1] + spans[:-1])
        or keys[-1] + spans[-1] != total
    ):
        raise ValueError("Forest leaves must partition the root grid exactly.")
    if _balance_violations(plan, levels_, coordinates_, keys).size:
        raise ValueError("Forest leaves violate the plan's 2:1 balance stencil.")
    return levels_, coordinates_, keys


def _leaf_workset(
    plan: ForestPlan,
    levels: np.ndarray,
    coordinates: np.ndarray,
    keys: np.ndarray,
    /,
) -> tuple[ForestLeafWorkset, np.ndarray, int, int]:
    """Bucketed execution workset, stable path IDs, and face/boundary counts."""
    count = levels.shape[0]
    dimension = plan.dimension
    (
        kinds,
        neighbors,
        minus,
        plus,
        axes,
        face_levels,
        coarse_fine,
        boundary_slots,
        boundary_axes,
        boundary_sides,
    ) = _face_routes(plan, levels, coordinates, keys)
    path_ids = _path_ids(plan, levels, coordinates)
    trees = np.ravel_multi_index(
        tuple((coordinates >> levels[:, None]).T), plan.root_shape
    )
    leaf_capacity = min(
        forest_capacity_bucket(count, plan.minimum_leaf_capacity),
        plan.maximum_leaf_capacity,
    )
    face_capacity = forest_capacity_bucket(minus.shape[0], 1)
    boundary_capacity = forest_capacity_bucket(boundary_slots.shape[0], 1)
    level_counts = np.bincount(levels, minlength=plan.maximum_level + 1)
    level_capacities = tuple(
        forest_capacity_bucket(value, 1) for value in level_counts.tolist()
    )
    signature = ForestWorksetSignature(
        dimension,
        plan.maximum_level,
        leaf_capacity,
        face_capacity,
        boundary_capacity,
        level_capacities,
    )
    face_neighbors = np.full(
        (leaf_capacity, dimension, 2, 1 << (dimension - 1)), -1, dtype=np.int32
    )
    face_neighbors[:count] = neighbors
    face_kinds = np.full((leaf_capacity, dimension, 2), -1, dtype=np.int8)
    face_kinds[:count] = kinds
    workset = ForestLeafWorkset(
        signature=signature,
        leaf_valid=jnp.asarray(np.arange(leaf_capacity) < count),
        leaf_levels=jnp.asarray(_padded(levels, leaf_capacity, -1, np.int32)),
        leaf_coordinates=jnp.asarray(_padded(coordinates, leaf_capacity, -1, np.int32)),
        leaf_trees=jnp.asarray(_padded(trees, leaf_capacity, -1, np.int32)),
        leaf_path_ids=jnp.asarray(_padded(path_ids, leaf_capacity, -1, np.int64)),
        face_kinds=jnp.asarray(face_kinds),
        face_neighbors=jnp.asarray(face_neighbors),
        face_valid=jnp.asarray(np.arange(face_capacity) < minus.shape[0]),
        face_minus=jnp.asarray(_padded(minus, face_capacity, -1, np.int32)),
        face_plus=jnp.asarray(_padded(plus, face_capacity, -1, np.int32)),
        face_axes=jnp.asarray(_padded(axes, face_capacity, -1, np.int32)),
        face_levels=jnp.asarray(_padded(face_levels, face_capacity, -1, np.int32)),
        face_coarse_fine=jnp.asarray(
            _padded(coarse_fine, face_capacity, False, np.bool_)
        ),
        boundary_valid=jnp.asarray(
            np.arange(boundary_capacity) < boundary_slots.shape[0]
        ),
        boundary_slots=jnp.asarray(
            _padded(boundary_slots, boundary_capacity, -1, np.int32)
        ),
        boundary_axes=jnp.asarray(
            _padded(boundary_axes, boundary_capacity, -1, np.int32)
        ),
        boundary_sides=jnp.asarray(
            _padded(boundary_sides, boundary_capacity, -1, np.int32)
        ),
        level_slots=tuple(
            jnp.asarray(_padded(np.flatnonzero(levels == level), capacity, -1, np.int32))
            for level, capacity in enumerate(level_capacities)
        ),
        level_valid=tuple(
            jnp.asarray(np.arange(capacity) < level_counts[level])
            for level, capacity in enumerate(level_capacities)
        ),
    )
    return workset, path_ids, minus.shape[0], boundary_slots.shape[0]


class ForestHierarchyTopology(StrictModule, NonTrainableState):
    """Immutable canonical 2:1-balanced forest leaf set and topology epoch."""

    plan: ForestPlan
    workset: ForestLeafWorkset
    leaf_count: int = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    boundary_count: int = eqx.field(static=True)
    epoch: TopologyEpoch
    topology_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: ForestPlan,
        levels: ArrayLike,
        coordinates: ArrayLike,
        /,
        *,
        epoch: TopologyEpoch | None = None,
    ) -> None:
        levels_, coordinates_, keys = _canonical_leaves(plan, levels, coordinates)
        workset, path_ids, face_count, boundary_count = _leaf_workset(
            plan, levels_, coordinates_, keys
        )
        topology_id = canonical_fingerprint(
            {
                "kind": "forest-hierarchy-topology",
                "plan": plan.plan_id,
                "path_ids": array_tree_fingerprint(path_ids),
            }
        )
        partition_id = canonical_fingerprint(
            {
                "kind": "forest-slot-layout",
                "topology": topology_id,
                "signature": workset.signature.signature_id,
            }
        )
        epoch_ = (
            TopologyEpoch(0, plan.geometry_id, topology_id, partition_id)
            if epoch is None
            else epoch
        )
        if not isinstance(epoch_, TopologyEpoch) or (
            epoch_.geometry_id != plan.geometry_id
            or epoch_.topology_id != topology_id
            or epoch_.partition_id != partition_id
        ):
            raise ValueError("Topology epoch identities do not match the forest leaves.")
        self.plan = plan
        self.workset = workset
        self.leaf_count = levels_.shape[0]
        self.face_count = face_count
        self.boundary_count = boundary_count
        self.epoch = epoch_
        self.topology_id = topology_id
        self.partition_id = partition_id

    @property
    def signature(self) -> ForestWorksetSignature:
        return self.workset.signature

    def leaf_levels(self, /) -> np.ndarray:
        """Host int64 levels of the active canonical leaves."""
        return np.asarray(self.workset.leaf_levels, dtype=np.int64)[: self.leaf_count]

    def leaf_coordinates(self, /) -> np.ndarray:
        """Host int64 level-lattice coordinates of the active canonical leaves."""
        return np.asarray(self.workset.leaf_coordinates, dtype=np.int64)[
            : self.leaf_count
        ]

    def leaf_keys(self, /) -> np.ndarray:
        """Host Morton order keys of the active canonical leaves (increasing)."""
        return _node_keys(self.plan, self.leaf_levels(), self.leaf_coordinates())

    def locate_cells(self, levels: ArrayLike, cells: ArrayLike, /) -> np.ndarray:
        """Host slots of the leaves containing the anchors of level-lattice cells."""
        levels_ = np.asarray(levels, dtype=np.int64).reshape(-1)
        cells_ = np.asarray(cells, dtype=np.int64).reshape(-1, self.plan.dimension)
        if cells_.shape[0] != levels_.shape[0]:
            raise ValueError("Forest cell queries require one level per cell.")
        if np.any(levels_ < 0) or np.any(levels_ > self.plan.maximum_level):
            raise ValueError("Forest cell query levels exceed the admitted depth.")
        extent = (
            np.asarray(self.plan.root_shape, dtype=np.int64)[None, :] << levels_[:, None]
        )
        if np.any(cells_ < 0) or np.any(cells_ >= extent):
            raise ValueError("Forest cell queries lie outside their level lattice.")
        return _locate_cells(self.plan, self.leaf_keys(), levels_, cells_)

    def reference_bounds(self, /) -> tuple[np.ndarray, np.ndarray]:
        """Host padded ``(C, d)`` reference lower/upper corners of every leaf slot."""
        levels = np.asarray(self.workset.leaf_levels, dtype=np.int64)
        coordinates = np.asarray(self.workset.leaf_coordinates, dtype=np.float64)
        spacing = (
            np.asarray(self.plan.root_spacing, dtype=np.float64)[None, :]
            / (2.0 ** levels.clip(min=0))[:, None]
        )
        lower = np.asarray(self.plan.lower_bounds, dtype=np.float64)[None, :] + (
            spacing * coordinates
        )
        valid = np.asarray(self.workset.leaf_valid)[:, None]
        return np.where(valid, lower, 0.0), np.where(valid, lower + spacing, 0.0)


def forest_common_refinement(
    first: ForestHierarchyTopology,
    second: ForestHierarchyTopology,
    /,
) -> ForestHierarchyTopology:
    """Coarsest forest refining both inputs: the pointwise finer leaf everywhere.

    The pointwise maximum of two 2:1-balanced level fields is 2:1 balanced under
    the same stencil, so the result is a valid topology of the shared plan.
    """
    if not isinstance(first, ForestHierarchyTopology) or not isinstance(
        second, ForestHierarchyTopology
    ):
        raise TypeError("Common refinement requires two forest topologies.")
    if first.plan.plan_id != second.plan.plan_id:
        raise ValueError("Common refinement requires one shared forest plan.")
    first_levels = first.leaf_levels()
    second_levels = second.leaf_levels()
    first_keys = first.leaf_keys()
    second_keys = second.leaf_keys()
    keep_first = second_levels[_locate(second_keys, first_keys)] <= first_levels
    keep_second = first_levels[_locate(first_keys, second_keys)] < second_levels
    return ForestHierarchyTopology(
        first.plan,
        np.concatenate((first_levels[keep_first], second_levels[keep_second])),
        np.concatenate(
            (
                first.leaf_coordinates()[keep_first],
                second.leaf_coordinates()[keep_second],
            )
        ),
    )


class ForestAdaptStatus(StrictModule, NonTrainableState):
    """Atomic host adaptation outcome."""

    code: str = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    changed: bool = eqx.field(static=True)
    message: str = eqx.field(static=True)
    status_id: str = eqx.field(static=True)

    def __init__(
        self, code: str, successful: bool, changed: bool, message: str, /
    ) -> None:
        code_ = str(code)
        message_ = str(message)
        if code_ not in ("initialized", "success", "unchanged", "capacity_exceeded"):
            raise ValueError("Unknown forest adaptation status.")
        if not message_:
            raise ValueError("Forest adaptation status message must be non-empty.")
        self.code = code_
        self.successful = bool(successful)
        self.changed = bool(changed)
        self.message = message_
        self.status_id = canonical_fingerprint(
            {
                "kind": "forest-adapt-status",
                "code": code_,
                "successful": bool(successful),
                "changed": bool(changed),
                "message": message_,
            }
        )


class ForestAdaptEvidence(StrictModule, NonTrainableState):
    """Auditable refinement, closure, coarsening, and capacity counts."""

    requested_refinements: int = eqx.field(static=True)
    depth_limited_refinements: int = eqx.field(static=True)
    balance_refinements: int = eqx.field(static=True)
    requested_coarsenings: int = eqx.field(static=True)
    coarsened_families: int = eqx.field(static=True)
    balance_vetoed_families: int = eqx.field(static=True)
    source_leaf_count: int = eqx.field(static=True)
    target_leaf_count: int = eqx.field(static=True)
    source_signature_id: str = eqx.field(static=True)
    target_signature_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        requested_refinements: int,
        depth_limited_refinements: int,
        balance_refinements: int,
        requested_coarsenings: int,
        coarsened_families: int,
        balance_vetoed_families: int,
        source_leaf_count: int,
        target_leaf_count: int,
        source_signature_id: str,
        target_signature_id: str,
    ) -> None:
        counts = (
            int(requested_refinements),
            int(depth_limited_refinements),
            int(balance_refinements),
            int(requested_coarsenings),
            int(coarsened_families),
            int(balance_vetoed_families),
            int(source_leaf_count),
            int(target_leaf_count),
        )
        if any(value < 0 for value in counts) or min(counts[6:]) <= 0:
            raise ValueError("Forest adaptation evidence counts are invalid.")
        if not source_signature_id or not target_signature_id:
            raise ValueError("Forest adaptation evidence requires signature IDs.")
        (
            self.requested_refinements,
            self.depth_limited_refinements,
            self.balance_refinements,
            self.requested_coarsenings,
            self.coarsened_families,
            self.balance_vetoed_families,
            self.source_leaf_count,
            self.target_leaf_count,
        ) = counts
        self.source_signature_id = str(source_signature_id)
        self.target_signature_id = str(target_signature_id)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "forest-adapt-evidence",
                "counts": counts,
                "source_signature": self.source_signature_id,
                "target_signature": self.target_signature_id,
            }
        )

    @property
    def signature_changed(self) -> bool:
        return self.source_signature_id != self.target_signature_id


class ForestAdaptResult(StrictModule, NonTrainableState):
    """Atomic successor topology, status, and evidence of one adaptation."""

    topology: ForestHierarchyTopology
    status: ForestAdaptStatus
    evidence: ForestAdaptEvidence
    result_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        topology: ForestHierarchyTopology,
        status: ForestAdaptStatus,
        evidence: ForestAdaptEvidence,
        /,
    ) -> None:
        if not isinstance(status, ForestAdaptStatus) or not isinstance(
            evidence, ForestAdaptEvidence
        ):
            raise TypeError("Forest adaptation results require status and evidence.")
        self.topology = topology
        self.status = status
        self.evidence = evidence
        self.result_id = canonical_fingerprint(
            {
                "kind": "forest-adapt-result",
                "epoch": topology.epoch.epoch_id,
                "status": status.status_id,
                "evidence": evidence.evidence_id,
            }
        )


class ForestTopologyCompiler(StrictModule, NonTrainableState):
    """Host local refine/coarsen with 2:1 closure over one forest plan."""

    plan: ForestPlan
    compiler_id: str = eqx.field(static=True)

    @checked
    def __init__(self, plan: ForestPlan, /) -> None:
        self.plan = plan
        self.compiler_id = canonical_fingerprint(
            {"kind": "forest-topology-compiler", "plan": plan.plan_id}
        )

    def initialize(self, level: int = 0, /) -> ForestAdaptResult:
        """Uniform forest refining every root to ``level``."""
        level_ = int(level)
        if level_ < 0 or level_ > self.plan.maximum_level:
            raise ValueError("Initial forest level exceeds the admitted depth.")
        shape = self.plan.level_shape(level_)
        coordinates = np.indices(shape).reshape(len(shape), -1).T
        levels = np.full((coordinates.shape[0],), level_, dtype=np.int64)
        topology = ForestHierarchyTopology(self.plan, levels, coordinates)
        signature = topology.signature.signature_id
        evidence = ForestAdaptEvidence(
            requested_refinements=0,
            depth_limited_refinements=0,
            balance_refinements=0,
            requested_coarsenings=0,
            coarsened_families=0,
            balance_vetoed_families=0,
            source_leaf_count=topology.leaf_count,
            target_leaf_count=topology.leaf_count,
            source_signature_id=signature,
            target_signature_id=signature,
        )
        return ForestAdaptResult(
            topology,
            ForestAdaptStatus("initialized", True, True, "Uniform forest initialized."),
            evidence,
        )

    def adapt(
        self,
        source: ForestHierarchyTopology,
        marks: ArrayLike,
        /,
    ) -> ForestAdaptResult:
        """Refine (+1), keep (0), or coarsen (-1) leaves with 2:1 closure.

        Refinement requests beyond the maximum level are dropped and counted.
        Balance closure then ripple-refines coarse neighbors; finally complete
        sibling families whose members are all marked for coarsening, and none of
        which was refined, merge unless that would break 2:1 balance.  Exceeding
        the maximum leaf capacity returns the unchanged source with a failed status.
        """
        if (
            not isinstance(source, ForestHierarchyTopology)
            or source.plan.plan_id != self.plan.plan_id
        ):
            raise ValueError("Forest adaptation source does not match its plan.")
        values = np.asarray(marks)
        capacity = source.signature.leaf_capacity
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("Forest adaptation marks must be integers.")
        if values.shape != (capacity,):
            raise ValueError("Forest adaptation marks must match the leaf capacity.")
        if np.any((values < -1) | (values > 1)):
            raise ValueError("Forest adaptation marks must lie in {-1, 0, 1}.")
        count = source.leaf_count
        if np.any(values[count:] != 0):
            raise ValueError("Inactive forest slots cannot carry adaptation marks.")
        active = values[:count].astype(np.int64)
        levels = source.leaf_levels()
        coordinates = source.leaf_coordinates()
        requested = active > 0
        admissible = requested & (levels < self.plan.maximum_level)
        refined_levels, refined_coordinates = _refine(levels, coordinates, admissible)
        balanced_levels, balanced_coordinates, keys, closure = _close_balance(
            self.plan, refined_levels, refined_coordinates
        )
        # Coarsening marks follow surviving source leaves by stable path identity.
        source_ids = _path_ids(self.plan, levels, coordinates)
        balanced_ids = _path_ids(self.plan, balanced_levels, balanced_coordinates)
        coarsen_ids = np.sort(source_ids[active < 0])
        position = np.searchsorted(coarsen_ids, balanced_ids).clip(
            max=max(coarsen_ids.size - 1, 0)
        )
        marked = (
            coarsen_ids[position] == balanced_ids
            if coarsen_ids.size
            else np.zeros(balanced_ids.shape, dtype=np.bool_)
        )
        final_levels, final_coordinates, coarsened, vetoed = _coarsen(
            self.plan, balanced_levels, balanced_coordinates, keys, marked
        )
        evidence_counts = {
            "requested_refinements": int(np.count_nonzero(requested)),
            "depth_limited_refinements": int(np.count_nonzero(requested & ~admissible)),
            "balance_refinements": closure,
            "requested_coarsenings": int(np.count_nonzero(active < 0)),
            "coarsened_families": coarsened,
            "balance_vetoed_families": vetoed,
            "source_leaf_count": count,
        }
        if final_levels.shape[0] > self.plan.maximum_leaf_capacity:
            evidence = ForestAdaptEvidence(
                **evidence_counts,
                target_leaf_count=final_levels.shape[0],
                source_signature_id=source.signature.signature_id,
                target_signature_id=source.signature.signature_id,
            )
            return ForestAdaptResult(
                source,
                ForestAdaptStatus(
                    "capacity_exceeded",
                    False,
                    False,
                    "Adapted forest exceeds the maximum leaf capacity.",
                ),
                evidence,
            )
        candidate = ForestHierarchyTopology(
            self.plan, final_levels, final_coordinates, epoch=None
        )
        evidence = ForestAdaptEvidence(
            **evidence_counts,
            target_leaf_count=candidate.leaf_count,
            source_signature_id=source.signature.signature_id,
            target_signature_id=candidate.signature.signature_id,
        )
        if (
            candidate.topology_id == source.topology_id
            and candidate.partition_id == source.partition_id
        ):
            return ForestAdaptResult(
                source,
                ForestAdaptStatus(
                    "unchanged", True, False, "Adaptation marks preserve the epoch."
                ),
                evidence,
            )
        target = ForestHierarchyTopology(
            self.plan,
            final_levels,
            final_coordinates,
            epoch=TopologyEpoch(
                source.epoch.index + 1,
                self.plan.geometry_id,
                candidate.topology_id,
                candidate.partition_id,
            ),
        )
        return ForestAdaptResult(
            target,
            ForestAdaptStatus(
                "success", True, True, "Adaptation marks compiled into a successor epoch."
            ),
            evidence,
        )


def _reference_identity(points: Any, time: Any, args: Any) -> Any:
    """Identity chart from global reference coordinates to physical space."""
    del time, args
    return points


class ForestBlockLowering(StrictModule, NonTrainableState):
    """Exact block-hierarchy image of one forest topology.

    Level zero holds every root as a unit block and level ``l > 0`` holds one
    ``2**d`` block per refined level-``l - 1`` node, so the canonical leaf cells of
    the image are exactly the forest leaves.  The existing block-AMR geometry and
    cut-complex owners consume the image; ``leaf_block_slots`` and
    ``leaf_local_cells`` route every forest leaf slot to its canonical block cell.
    """

    topology: BlockHierarchyTopology
    leaf_block_slots: Array
    leaf_local_cells: Array
    forest_topology_id: str = eqx.field(static=True)
    lowering_id: str = eqx.field(static=True)

    @checked
    def __init__(self, forest: ForestHierarchyTopology, /) -> None:
        plan = forest.plan
        dimension = plan.dimension
        levels = forest.leaf_levels()
        coordinates = forest.leaf_coordinates()
        refined: list[np.ndarray] = []
        for level in range(plan.maximum_level):
            deeper = levels > level
            nodes = coordinates[deeper] >> (levels[deeper] - level)[:, None]
            refined.append(
                np.unique(nodes, axis=0)
                if nodes.shape[0]
                else np.zeros((0, dimension), dtype=np.int64)
            )
        level_plans = [
            BlockLevelPlan(0, (1,) * dimension, plan.root_count, refinement_ratio=2)
        ]
        level_plans.extend(
            BlockLevelPlan(
                level + 1,
                (2,) * dimension,
                forest_capacity_bucket(nodes.shape[0], 1),
                refinement_ratio=2,
            )
            for level, nodes in enumerate(refined)
        )
        hierarchy = BlockHierarchyPlan(plan.grid, level_plans)
        rows = [np.indices(plan.root_shape).reshape(dimension, -1).T, *refined]
        block_topology = BlockHierarchyTopology(
            hierarchy,
            tuple(
                # ty: ignore[invalid-argument-type]
                canonical_block_metadata(hierarchy, level, level_rows)
                for level, level_rows in enumerate(rows)
            ),
        )
        # Blocks at level ``l > 0`` are keyed by their parent node, so a leaf's
        # block is ``coordinate >> 1`` and its local cell is ``coordinate & 1``.
        block_rows = np.where(levels[:, None] > 0, coordinates >> 1, coordinates)
        local = np.where(levels[:, None] > 0, coordinates & 1, 0)
        slots = np.empty(levels.shape, dtype=np.int64)
        for level, level_rows in enumerate(rows):
            members = levels == level
            if not np.any(members):
                continue
            lattice = hierarchy.block_lattice_shapes[level]
            sorted_linear = np.ravel_multi_index(tuple(level_rows.T), lattice)
            order = np.argsort(sorted_linear)
            query = np.ravel_multi_index(tuple(block_rows[members].T), lattice)
            slots[members] = order[np.searchsorted(sorted_linear[order], query)]
        local_flat = np.where(
            levels > 0,
            np.ravel_multi_index(tuple(local.T), (2,) * dimension),
            0,
        )
        capacity = forest.signature.leaf_capacity
        self.topology = block_topology
        self.leaf_block_slots = jnp.asarray(_padded(slots, capacity, -1, np.int32))
        self.leaf_local_cells = jnp.asarray(_padded(local_flat, capacity, -1, np.int32))
        self.forest_topology_id = forest.topology_id
        self.lowering_id = canonical_fingerprint(
            {
                "kind": "forest-block-lowering",
                "forest": forest.topology_id,
                "blocks": block_topology.topology_id,
            }
        )

    def leaf_values(
        self,
        level_values: Sequence[ArrayLike],
        leaf_levels: ArrayLike,
        /,
    ) -> Array:
        """Gather per-level canonical block-cell arrays to padded forest leaf slots.

        ``level_values[l]`` has shape ``(maximum_blocks_l, *block_shape_l, *T)``;
        the result has shape ``(leaf_capacity, *T)`` with zero padding.
        """
        plan = self.topology.plan
        values = tuple(jnp.asarray(value) for value in level_values)
        if len(values) != len(plan.levels):
            raise ValueError("Leaf gathering requires one array per block level.")
        flats = []
        offsets = [0]
        trailing = None
        for level_plan, value in zip(plan.levels, values, strict=True):
            prefix = (level_plan.maximum_blocks,) + level_plan.block_shape
            if value.shape[: len(prefix)] != prefix:
                raise ValueError("Block-level values do not match the lowered layout.")
            tail = value.shape[len(prefix) :]
            if trailing is not None and tail != trailing:
                raise ValueError("Block-level values must share trailing shapes.")
            trailing = tail
            cells = level_plan.maximum_blocks * prod(level_plan.block_shape)
            flats.append(value.reshape((cells,) + tail))
            offsets.append(offsets[-1] + cells)
        stacked = jnp.concatenate(flats, axis=0)
        levels = jnp.asarray(leaf_levels, dtype=jnp.int32)
        valid = levels >= 0
        cells_per_block = jnp.asarray(
            [prod(level_plan.block_shape) for level_plan in plan.levels],
            dtype=jnp.int32,
        )
        safe_level = jnp.where(valid, levels, 0)
        index = (
            jnp.asarray(offsets[:-1], dtype=jnp.int32)[safe_level]
            + jnp.where(valid, self.leaf_block_slots, 0) * cells_per_block[safe_level]
            + jnp.where(valid, self.leaf_local_cells, 0)
        )
        gathered = stacked[index]
        # ty: ignore[invalid-argument-type]
        mask = valid.reshape(valid.shape + (1,) * len(trailing))
        return jnp.where(mask, gathered, jnp.zeros((), dtype=gathered.dtype))


class ForestLeafGeometry(StrictModule, NonTrainableState):
    """Physical leaf volumes/centers and oriented interior/boundary face areas.

    ``face_area_vectors`` point from ``face_minus`` to ``face_plus``;
    ``boundary_area_vectors`` are outward.  Padding rows are zero and padded
    volumes are one so volume-normalized consumers remain finite.  Static structure
    depends only on the workset capacity bucket, so compiled consumers are reused.
    """

    volumes: Array
    centers: Array
    face_area_vectors: Array
    face_measures: Array
    boundary_area_vectors: Array
    boundary_measures: Array
    valid: Array


def _affine_leaf_geometry(topology: ForestHierarchyTopology, /) -> ForestLeafGeometry:
    plan = topology.plan
    workset = topology.workset
    dimension = plan.dimension
    spacing = jnp.asarray(plan.root_spacing, dtype=jnp.float64)
    lower_bounds = jnp.asarray(plan.lower_bounds, dtype=jnp.float64)
    leaf_valid = workset.leaf_valid
    scale = jnp.exp2(-jnp.where(leaf_valid, workset.leaf_levels, 0).astype(jnp.float64))
    widths = spacing[None, :] * scale[:, None]
    centers = lower_bounds[None, :] + widths * (
        workset.leaf_coordinates.astype(jnp.float64) + 0.5
    )
    volumes = jnp.where(leaf_valid, jnp.prod(widths, axis=1), 1.0)

    def oriented_areas(axes: Any, levels: Any, valid: Any) -> Any:
        safe_axes = jnp.where(valid, axes, 0)
        face_scale = jnp.exp2(-jnp.where(valid, levels, 0).astype(jnp.float64))
        full = jnp.prod(spacing) / spacing[safe_axes]
        measure = jnp.where(valid, full * face_scale ** (dimension - 1), 0.0)
        normal = jnp.eye(dimension, dtype=jnp.float64)[safe_axes]
        return measure[:, None] * normal, measure

    face_vectors, face_measures = oriented_areas(
        workset.face_axes, workset.face_levels, workset.face_valid
    )
    boundary_levels = workset.leaf_levels[
        jnp.where(workset.boundary_valid, workset.boundary_slots, 0)
    ]
    boundary_vectors, boundary_measures = oriented_areas(
        workset.boundary_axes, boundary_levels, workset.boundary_valid
    )
    side_sign = jnp.where(workset.boundary_sides == 1, 1.0, -1.0)
    return ForestLeafGeometry(
        volumes=volumes,
        centers=jnp.where(leaf_valid[:, None], centers, 0.0),
        face_area_vectors=face_vectors,
        face_measures=face_measures,
        boundary_area_vectors=boundary_vectors * side_sign[:, None],
        boundary_measures=boundary_measures,
        valid=jnp.asarray(True),
    )


def _mapped_leaf_geometry(
    topology: ForestHierarchyTopology,
    time: ArrayLike,
    args: Any,
    quadrature_order: int,
    /,
) -> ForestLeafGeometry:
    plan = topology.plan
    workset = topology.workset
    lowering = ForestBlockLowering(topology)
    state = CanonicalMappedGeometryPlan(
        lowering.topology,
        # ty: ignore[invalid-argument-type]
        plan.maps,
        quadrature_order=quadrature_order,
    ).evaluate(time, args)
    # Every lowered level has exactly one canonical bucket.
    volumes = lowering.leaf_values(
        tuple(level[0] for level in state.cell_volumes), workset.leaf_levels
    )
    centers = lowering.leaf_values(
        tuple(level[0] for level in state.cell_centers), workset.leaf_levels
    )
    face_areas = lowering.leaf_values(
        tuple(jnp.sum(level[0], axis=-2) for level in state.face_weighted_area_vectors),
        workset.leaf_levels,
    )
    face_valid = workset.face_valid
    minus = jnp.where(face_valid, workset.face_minus, 0)
    plus = jnp.where(face_valid, workset.face_plus, 0)
    axes = jnp.where(face_valid, workset.face_axes, 0)
    minus_finer = workset.leaf_levels[minus] == workset.face_levels
    face_vectors = jnp.where(
        minus_finer[:, None],
        face_areas[minus, 2 * axes + 1],
        -face_areas[plus, 2 * axes],
    )
    face_vectors = jnp.where(face_valid[:, None], face_vectors, 0.0)
    boundary_valid = workset.boundary_valid
    boundary_vectors = face_areas[
        jnp.where(boundary_valid, workset.boundary_slots, 0),
        2 * jnp.where(boundary_valid, workset.boundary_axes, 0)
        + jnp.where(boundary_valid, workset.boundary_sides, 0),
    ]
    boundary_vectors = jnp.where(boundary_valid[:, None], boundary_vectors, 0.0)
    return ForestLeafGeometry(
        volumes=jnp.where(workset.leaf_valid, volumes, 1.0),
        centers=centers,
        face_area_vectors=face_vectors,
        face_measures=jnp.linalg.norm(face_vectors, axis=-1),
        boundary_area_vectors=boundary_vectors,
        boundary_measures=jnp.linalg.norm(boundary_vectors, axis=-1),
        valid=state.evidence.valid,
    )


def forest_leaf_geometry(
    topology: ForestHierarchyTopology,
    /,
    *,
    time: ArrayLike = 0.0,
    args: Any = None,
    quadrature_order: int = 3,
) -> ForestLeafGeometry:
    """Evaluate physical leaf and face geometry of one forest topology.

    Affine roots use exact closed forms; mapped roots evaluate the plan's chart
    through the canonical mapped block geometry of the forest's block image, so
    metric closure and GCL evidence reach ``valid``.
    """
    if not isinstance(topology, ForestHierarchyTopology):
        raise TypeError("Forest geometry requires a ForestHierarchyTopology.")
    if topology.plan.maps is None:
        return _affine_leaf_geometry(topology)
    return _mapped_leaf_geometry(topology, time, args, quadrature_order)


class ForestCutComplex(StrictModule, NonTrainableState):
    """Cut-cell complex of a forest built by the block-AMR cut-complex owner.

    ``component_leaf_slots`` maps every padded cut component to the forest leaf
    slot containing it (``-1`` for inactive components).
    """

    complex: Any
    component_leaf_slots: Array
    forest_topology_id: str = eqx.field(static=True)
    embedding_id: str = eqx.field(static=True)


def prepare_forest_cut_complex(
    topology: ForestHierarchyTopology,
    bodies: EmbeddedLevelSetBodySet,
    resources: BlockAMRResourcePlan,
    /,
    *,
    subdivision: int = 1,
    predicate_tolerance: float = 1.0e-12,
    time: ArrayLike = 0.0,
    args: Any = None,
) -> ForestCutComplex:
    """Embed level-set bodies in forest leaves through the multivalued cut owner."""
    if not isinstance(topology, ForestHierarchyTopology):
        raise TypeError("Forest cut complexes require a ForestHierarchyTopology.")
    plan = topology.plan
    lowering = ForestBlockLowering(topology)
    chart = _reference_identity if plan.maps is None else plan.maps
    chart_id = "reference-identity" if plan.maps is None else plan.maps.map_set_id
    match plan.dimension:
        case 2:
            cut_plan = MultivaluedCutCell2DPlan(
                lowering.topology,
                chart,
                chart_id,
                bodies,
                resources,
                subdivision=subdivision,
                predicate_tolerance=predicate_tolerance,
            )
        case 3:
            cut_plan = MultivaluedCutCellPlan(
                lowering.topology,
                chart,
                chart_id,
                bodies,
                resources,
                subdivision=subdivision,
                predicate_tolerance=predicate_tolerance,
            )
        case _:
            raise ValueError("Forest cut complexes require two or three dimensions.")
    complex_ = cut_plan.prepare(time, args)
    active = np.asarray(complex_.component_active, dtype=np.bool_)
    slots = np.full(active.shape, -1, dtype=np.int32)
    if np.any(active):
        slots[active] = topology.locate_cells(
            np.asarray(complex_.component_levels)[active],
            np.asarray(complex_.component_cell_coordinates)[active],
        )
    return ForestCutComplex(
        complex=complex_,
        component_leaf_slots=jnp.asarray(slots),
        forest_topology_id=topology.topology_id,
        embedding_id=canonical_fingerprint(
            {
                "kind": "forest-cut-complex",
                "forest": topology.topology_id,
                "lowering": lowering.lowering_id,
                "cut_plan": cut_plan.plan_id,
            }
        ),
    )


__all__ = [
    "AMRBalanceStencil",
    "ForestAdaptEvidence",
    "ForestAdaptResult",
    "ForestAdaptStatus",
    "ForestBlockLowering",
    "ForestCutComplex",
    "ForestFaceKind",
    "ForestHierarchyTopology",
    "ForestLeafGeometry",
    "ForestLeafWorkset",
    "ForestPlan",
    "ForestTopologyCompiler",
    "ForestWorksetSignature",
    "forest_common_refinement",
    "forest_leaf_geometry",
    "prepare_forest_cut_complex",
]
