#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Adaptive linear octrees with bounded adaptive-FMM interaction lists."""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import product
from operator import index
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.sparse import EdgeRelation, RelationExecutionPlan, RelationExecutionState

from ._morton import morton_encode_integer, MortonAddressPlan


class AdaptiveOctreeInteractionList(NonTrainableState, StrictModule):
    """One adaptive-FMM interaction list stored target-major with bounded capacity.

    ``routes`` holds ``(source node, target node)`` pairs sorted by target node and
    then source node; ``row_offsets[n]:row_offsets[n + 1]`` addresses the stored
    routes of target node ``n``. ``required_routes`` counts the complete list and
    ``overflow`` reports that routes beyond ``routes.capacity`` were not stored.
    ``row_width`` covers every stored row and is at least one.
    """

    routes: EdgeRelation
    execution: RelationExecutionState
    row_offsets: Array
    required_routes: Array
    overflow: Array
    row_width: int = eqx.field(static=True)

    def rows(self, target_nodes: ArrayLike, /) -> tuple[Array, Array]:
        """Gather the fixed-width source-node rows of ``target_nodes``."""
        nodes = jnp.asarray(target_nodes, dtype=jnp.int32)
        starts = self.row_offsets[nodes]
        ends = self.row_offsets[nodes + 1]
        slots = starts[..., None] + jnp.arange(self.row_width, dtype=jnp.int32)
        valid = slots < ends[..., None]
        safe = jnp.minimum(slots, self.routes.capacity - 1)
        return jnp.where(valid, self.routes.source_indices[safe], 0), valid


class AdaptiveOctreeEvidence(NonTrainableState, StrictModule):
    """Refinement and interaction-capacity evidence of one adaptive octree."""

    node_count: Array
    leaf_count: Array
    maximum_level: Array
    maximum_leaf_occupancy: Array
    leaf_capacity_satisfied: Array
    successful: Array


class AdaptiveOctree(NonTrainableState, StrictModule):
    """Complete adaptive linear octree over one Morton-sorted point set.

    Nodes are stored level by level and in Morton order within each level. Every
    refined node owns all ``2**dimension`` children in one contiguous block, so the
    leaves tile the address box and every domain point has exactly one containing
    leaf; unoccupied children are empty leaves. Node point spans index
    ``point_order``, the canonical ``(Morton code, logical index)`` point order.

    Two nodes are separated when, along some axis, their gap is at least the width
    of the smaller node plus ``separation_padding``; with zero padding this is the
    classical non-adjacency test, and a positive padding keeps that clearance for
    sources displaced by at most the padding. For target node ``B``:

    - ``u_list``: source leaves not separated from leaf ``B`` (direct P2P);
    - ``v_list``: separated same-level children of the colleagues of the parent of
      ``B`` (M2L);
    - ``w_list``: separated finer descendants of the colleagues of leaf ``B`` whose
      parents are not separated from ``B`` (M2P at the points of ``B``);
    - ``x_list``: separated coarser leaves ``A`` with ``B`` in ``w_list(A)`` (P2L).

    Lists contain occupied source nodes only. For every leaf ``B``, the ``u_list``
    and ``w_list`` of ``B`` together with the ``v_list`` and ``x_list`` of ``B`` and
    all of its ancestors cover every point exactly once.
    """

    address_plan: MortonAddressPlan
    node_prefixes: Array
    node_levels: Array
    node_parents: Array
    node_first_children: Array
    node_child_counts: Array
    node_point_starts: Array
    node_point_ends: Array
    node_centers: Array
    node_half_widths: Array
    leaf_nodes: Array
    leaf_code_starts: Array
    point_order: Array
    point_leaves: Array
    u_list: AdaptiveOctreeInteractionList
    v_list: AdaptiveOctreeInteractionList
    w_list: AdaptiveOctreeInteractionList
    x_list: AdaptiveOctreeInteractionList
    evidence: AdaptiveOctreeEvidence
    level_offsets: tuple[int, ...] = eqx.field(static=True)
    maximum_leaf_occupancy: int = eqx.field(static=True)
    separation_padding: float = eqx.field(static=True)
    tree_id: str = eqx.field(static=True)

    @property
    def node_count(self) -> int:
        return self.node_levels.shape[0]

    @property
    def point_count(self) -> int:
        return self.point_order.shape[0]

    @property
    def maximum_level(self) -> int:
        return len(self.level_offsets) - 2

    @property
    def node_is_leaf(self) -> Array:
        return self.node_child_counts == 0

    def locate(self, points: ArrayLike, /) -> Array:
        """Return the leaf containing each point, or ``-1`` outside the address box."""
        encoding = self.address_plan.encode(jnp.asarray(points))
        slots = jnp.searchsorted(self.leaf_code_starts, encoding.codes, side="right")
        safe = jnp.clip(slots - 1, 0, self.leaf_nodes.shape[0] - 1)
        return jnp.where(encoding.in_domain, self.leaf_nodes[safe], -1).astype(jnp.int32)

    def far_route_geometry(self) -> tuple[Array, Array, Array, Array]:
        """Source nodes, radii, distances, and validity of the V, W, and X routes.

        Routes are concatenated in V, W, X order and radii are cell half-diagonals.
        V pairs the padded source ball and the target ball with their center
        distance, W the padded source ball with its distance to the target leaf
        box, and X the target ball with its distance to the padded source leaf box.
        ``radii / distances < 1`` certifies convergence of the route expansion for
        sources displaced by at most ``separation_padding``.
        """
        centers = self.node_centers
        half_widths = self.node_half_widths
        radius = jnp.linalg.norm(half_widths, axis=-1)
        padding = self.separation_padding

        def box_distance(points, boxes):
            excess = jnp.abs(points - centers[boxes]) - half_widths[boxes]
            return jnp.linalg.norm(jnp.maximum(excess, 0.0), axis=-1)

        far = self.v_list.routes
        point = self.w_list.routes
        leaf = self.x_list.routes
        radii = jnp.concatenate(
            (
                radius[far.source_indices] + padding + radius[far.target_indices],
                radius[point.source_indices] + padding,
                radius[leaf.target_indices],
            )
        )
        distances = jnp.concatenate(
            (
                jnp.linalg.norm(
                    centers[far.target_indices] - centers[far.source_indices], axis=-1
                ),
                box_distance(centers[point.source_indices], point.target_indices),
                box_distance(centers[leaf.target_indices], leaf.source_indices) - padding,
            )
        )
        sources = jnp.concatenate(
            (far.source_indices, point.source_indices, leaf.source_indices)
        )
        valid = jnp.concatenate((far.valid, point.valid, leaf.valid))
        return sources, radii, distances, valid

    def leaf_points(self, leaf_nodes: ArrayLike, /) -> tuple[Array, Array]:
        """Gather logical point indices of leaves in ``maximum_leaf_occupancy`` slots."""
        nodes = jnp.asarray(leaf_nodes, dtype=jnp.int32)
        starts = self.node_point_starts[nodes]
        ends = self.node_point_ends[nodes]
        slots = starts[..., None] + jnp.arange(
            self.maximum_leaf_occupancy, dtype=jnp.int32
        )
        valid = slots < ends[..., None]
        safe = jnp.minimum(slots, self.point_count - 1)
        return jnp.where(valid, self.point_order[safe], 0), valid

    def upward_pass(
        self,
        values: PyTree[Array],
        translate: Callable[[PyTree[Array], Array, Array], PyTree[Array]],
        /,
    ) -> PyTree[Array]:
        """Add translated child payloads into their parents, deepest level first.

        Every ``values`` leaf begins with the node axis. ``translate(child_values,
        child_centers, parent_centers)`` maps one level of child payloads to the
        expansion centers of their parents; each parent sums its children in child
        order.
        """
        branching = 1 << self.address_plan.dimension
        for level in range(self.maximum_level, 0, -1):
            start = self.level_offsets[level]
            stop = self.level_offsets[level + 1]
            parents = self.node_parents[start:stop]
            translated = translate(
                jax.tree.map(lambda value: value[start:stop], values),
                self.node_centers[start:stop],
                self.node_centers[parents],
            )
            block_parents = parents[::branching]
            values = jax.tree.map(
                lambda value, moved: value.at[block_parents].add(
                    moved.reshape((-1, branching) + moved.shape[1:]).sum(axis=1)
                ),
                values,
                translated,
            )
        return values

    def downward_pass(
        self,
        values: PyTree[Array],
        translate: Callable[[PyTree[Array], Array, Array], PyTree[Array]],
        /,
    ) -> PyTree[Array]:
        """Add translated parent payloads into their children, coarsest level first.

        ``translate(parent_values, parent_centers, child_centers)`` maps one level of
        parent payloads to the expansion centers of their children.
        """
        for level in range(1, self.maximum_level + 1):
            start = self.level_offsets[level]
            stop = self.level_offsets[level + 1]
            parents = self.node_parents[start:stop]
            inherited = translate(
                jax.tree.map(lambda value: value[parents], values),
                self.node_centers[parents],
                self.node_centers[start:stop],
            )
            values = jax.tree.map(
                lambda value, added: value.at[start:stop].add(added),
                values,
                inherited,
            )
        return values


class _NodeTable(NamedTuple):
    """Host node arrays of one complete adaptive tree."""

    prefixes: np.ndarray
    levels: np.ndarray
    parents: np.ndarray
    first_children: np.ndarray
    child_counts: np.ndarray
    point_starts: np.ndarray
    point_ends: np.ndarray
    corners: np.ndarray
    sizes: np.ndarray
    code_starts: np.ndarray
    level_offsets: tuple[int, ...]


def _prefix_corners(
    prefixes: np.ndarray, level: int, address: MortonAddressPlan
) -> np.ndarray:
    """Finest-cell integer lower corners of level-``level`` Morton prefixes."""
    shift = np.uint64(address.dimension * (address.maximum_depth - level))
    return np.asarray(address.decode(jnp.asarray(prefixes << shift)), dtype=np.int64)


def _corner_prefixes(
    corners: np.ndarray, level: int, address: MortonAddressPlan
) -> np.ndarray:
    """Level-``level`` Morton prefixes of cells with finest-cell lower corners."""
    codes = np.asarray(
        morton_encode_integer(jnp.asarray(corners), address.maximum_depth),
        dtype=np.uint64,
    )
    return codes >> np.uint64(address.dimension * (address.maximum_depth - level))


def _subdivided_prefixes(
    codes: np.ndarray, address: MortonAddressPlan, leaf_capacity: int
) -> list[np.ndarray]:
    """Sorted prefixes of the cells refined at each level ``0 <= level < depth``."""
    refined = []
    for level in range(address.maximum_depth):
        if level > 0 and refined[-1].size == 0:
            refined.append(np.zeros((0,), dtype=np.uint64))
            continue
        shift = np.uint64(address.dimension * (address.maximum_depth - level))
        prefixes = codes >> shift
        run_starts = np.flatnonzero(
            np.concatenate(([True], prefixes[1:] != prefixes[:-1]))
        )
        counts = np.diff(np.append(run_starts, prefixes.size))
        refined.append(prefixes[run_starts][counts > leaf_capacity])
    return refined


def _balanced_prefixes(
    refined: list[np.ndarray], address: MortonAddressPlan
) -> list[np.ndarray]:
    """Refine until the colleagues of every refined node exist (2:1 leaf balance).

    Sweeping from the finest refined level to the root is sufficient: each sweep
    step only adds refinements at coarser levels, which later steps process.
    """
    dimension = address.dimension
    depth = address.maximum_depth
    offsets = np.asarray(
        [offset for offset in product((-1, 0, 1), repeat=dimension) if any(offset)],
        dtype=np.int64,
    )
    balanced = list(refined)
    for level in range(depth - 1, 0, -1):
        if balanced[level].size == 0:
            continue
        size = np.int64(1) << np.int64(depth - level)
        corners = _prefix_corners(balanced[level], level, address)
        neighbors = (corners[:, None, :] + offsets[None, :, :] * size).reshape(
            (-1, dimension)
        )
        inside = np.all((neighbors >= 0) & (neighbors < (1 << depth)), axis=1)
        required = np.unique(_corner_prefixes(neighbors[inside], level - 1, address))
        for ancestor in range(level - 1, -1, -1):
            balanced[ancestor] = np.union1d(balanced[ancestor], required)
            required = np.unique(required >> np.uint64(dimension))
    return balanced


def _node_table(
    refined: list[np.ndarray], codes: np.ndarray, address: MortonAddressPlan
) -> _NodeTable:
    """Build level-ordered nodes, child blocks, and point spans of the tree."""
    dimension = address.dimension
    depth = address.maximum_depth
    branching = 1 << dimension
    digits = np.arange(branching, dtype=np.uint64)
    level_prefixes = [np.zeros((1,), dtype=np.uint64)]
    for level in range(1, depth + 1):
        parents = refined[level - 1]
        if parents.size == 0:
            break
        level_prefixes.append(
            ((parents[:, None] << np.uint64(dimension)) | digits[None, :]).reshape(-1)
        )
    sizes_per_level = [prefixes.size for prefixes in level_prefixes]
    offsets = np.concatenate(([0], np.cumsum(sizes_per_level))).astype(np.int64)
    node_count = offsets[-1]
    prefixes = np.concatenate(level_prefixes)
    levels = np.repeat(np.arange(len(level_prefixes), dtype=np.int32), sizes_per_level)
    parents = np.full((node_count,), -1, dtype=np.int32)
    first_children = np.full((node_count,), -1, dtype=np.int32)
    child_counts = np.zeros((node_count,), dtype=np.int32)
    for level in range(1, len(level_prefixes)):
        parent_nodes = offsets[level - 1] + np.searchsorted(
            level_prefixes[level - 1], refined[level - 1]
        )
        parents[offsets[level] : offsets[level + 1]] = np.repeat(parent_nodes, branching)
        first_children[parent_nodes] = offsets[level] + branching * np.arange(
            parent_nodes.size
        )
        child_counts[parent_nodes] = branching
    shifts = (dimension * (depth - levels)).astype(np.uint64)
    code_starts = prefixes << shifts
    code_ends = (prefixes + np.uint64(1)) << shifts
    corners = np.asarray(address.decode(jnp.asarray(code_starts)), dtype=np.int64)
    return _NodeTable(
        prefixes=prefixes,
        levels=levels,
        parents=parents,
        first_children=first_children,
        child_counts=child_counts,
        point_starts=np.searchsorted(codes, code_starts, side="left").astype(np.int32),
        point_ends=np.searchsorted(codes, code_ends, side="left").astype(np.int32),
        corners=corners,
        sizes=np.left_shift(np.int64(1), (depth - levels).astype(np.int64)),
        code_starts=code_starts,
        level_offsets=tuple(offsets.tolist()),
    )


def _separated(
    targets: np.ndarray,
    sources: np.ndarray,
    table: _NodeTable,
    finest_width: np.ndarray,
    padding: float,
) -> np.ndarray:
    """Whether some axis gap reaches the smaller node width plus ``padding``."""
    target_lower = table.corners[targets]
    source_lower = table.corners[sources]
    target_upper = target_lower + table.sizes[targets, None]
    source_upper = source_lower + table.sizes[sources, None]
    gap = np.maximum(
        np.maximum(source_lower - target_upper, target_lower - source_upper), 0
    )
    width = np.minimum(table.sizes[targets], table.sizes[sources])[:, None]
    return np.any((gap - width) * finest_width[None, :] >= padding, axis=1)


def _child_pairs(
    targets: np.ndarray, sources: np.ndarray, table: _NodeTable, branching: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand unseparated pairs: refine internal sides, keep leaf sides fixed.

    Returns child targets, child sources, and the expansion kind: ``0`` when both
    sides were refined, ``1`` when only the source side was, ``2`` when only the
    target side was.
    """
    digits = np.arange(branching, dtype=np.int64)
    target_leaf = table.child_counts[targets] == 0
    source_leaf = table.child_counts[sources] == 0
    both = ~target_leaf & ~source_leaf
    target_first = table.first_children[targets].astype(np.int64)
    source_first = table.first_children[sources].astype(np.int64)
    both_targets = np.broadcast_to(
        target_first[both, None, None] + digits[None, :, None],
        (np.count_nonzero(both), branching, branching),
    ).reshape(-1)
    both_sources = np.broadcast_to(
        source_first[both, None, None] + digits[None, None, :],
        (np.count_nonzero(both), branching, branching),
    ).reshape(-1)
    source_targets = np.repeat(targets[target_leaf], branching)
    source_sources = (source_first[target_leaf, None] + digits[None, :]).reshape(-1)
    target_targets = (target_first[source_leaf, None] + digits[None, :]).reshape(-1)
    target_sources = np.repeat(sources[source_leaf], branching)
    kinds = np.concatenate(
        (
            np.zeros(both_targets.size, dtype=np.int8),
            np.ones(source_targets.size, dtype=np.int8),
            np.full(target_targets.size, 2, dtype=np.int8),
        )
    )
    return (
        np.concatenate((both_targets, source_targets, target_targets)),
        np.concatenate((both_sources, source_sources, target_sources)),
        kinds,
    )


def _interaction_pairs(
    table: _NodeTable, address: MortonAddressPlan, padding: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Classify every target/source node pair by one level-synchronous dual descent.

    The descent starts at ``(root, root)`` and refines each unseparated pair on its
    internal sides, so the admitted pairs partition all point pairs.
    """
    branching = 1 << address.dimension
    extent = np.asarray(address.upper, dtype=np.float64) - np.asarray(
        address.lower, dtype=np.float64
    )
    finest_width = extent / np.float64(address.resolution)
    leaf = table.child_counts == 0
    occupied = table.point_ends > table.point_starts
    found: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "u": [],
        "v": [],
        "w": [],
        "x": [],
    }
    targets = np.zeros((1,), dtype=np.int64)
    sources = np.zeros((1,), dtype=np.int64)
    if leaf[0]:
        found["u"].append((targets, sources))
        targets = targets[:0]
        sources = sources[:0]
    while targets.size:
        targets, sources, kinds = _child_pairs(targets, sources, table, branching)
        keep = occupied[sources]
        targets, sources, kinds = targets[keep], sources[keep], kinds[keep]
        separated = _separated(targets, sources, table, finest_width, padding)
        for name, kind in (("v", 0), ("w", 1), ("x", 2)):
            selected = separated & (kinds == kind)
            found[name].append((targets[selected], sources[selected]))
        both_leaves = leaf[targets] & leaf[sources]
        direct = ~separated & both_leaves
        found["u"].append((targets[direct], sources[direct]))
        descend = ~separated & ~both_leaves
        targets, sources = targets[descend], sources[descend]
    return {
        name: (
            np.concatenate([pair[0] for pair in pairs] + [np.zeros((0,), np.int64)]),
            np.concatenate([pair[1] for pair in pairs] + [np.zeros((0,), np.int64)]),
        )
        for name, pairs in found.items()
    }


def _interaction_list(
    targets: np.ndarray,
    sources: np.ndarray,
    node_count: int,
    capacity: int | None,
) -> AdaptiveOctreeInteractionList:
    """Store one list target-major, truncating canonically beyond ``capacity``."""
    order = np.lexsort((sources, targets))
    targets = targets[order]
    sources = sources[order]
    required = targets.size
    route_capacity = max(required, 1) if capacity is None else capacity
    stored = min(required, route_capacity)
    counts = np.bincount(targets[:stored], minlength=node_count)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    source_indices = np.zeros((route_capacity,), dtype=np.int32)
    target_indices = np.zeros((route_capacity,), dtype=np.int32)
    source_indices[:stored] = sources[:stored]
    target_indices[:stored] = targets[:stored]
    routes = EdgeRelation(
        jnp.asarray(source_indices),
        jnp.asarray(target_indices),
        source_size=node_count,
        target_size=node_count,
        valid=jnp.asarray(np.arange(route_capacity) < stored),
    )
    return AdaptiveOctreeInteractionList(
        routes=routes,
        execution=RelationExecutionPlan(maximum_active_targets=node_count).prepare(
            routes
        ),
        row_offsets=jnp.asarray(offsets),
        required_routes=jnp.asarray(required, dtype=jnp.int32),
        overflow=jnp.asarray(required > route_capacity),
        row_width=max(np.max(counts, initial=0).item(), 1),
    )


class AdaptiveOctreePlan(StrictModule):
    """Host preparation of adaptive linear octrees and their FMM interaction lists.

    A cell is subdivided while it holds more than ``leaf_capacity`` points and lies
    above ``address_plan.maximum_depth``. ``balanced=True`` additionally refines
    leaves until leaves that touch differ by at most one level. ``u_capacity``,
    ``v_capacity``, ``w_capacity``, and ``x_capacity`` bound the stored routes of
    each list; ``None`` stores the complete list.
    """

    address_plan: MortonAddressPlan
    leaf_capacity: int = eqx.field(static=True)
    balanced: bool = eqx.field(static=True)
    separation_padding: float = eqx.field(static=True)
    u_capacity: int | None = eqx.field(static=True)
    v_capacity: int | None = eqx.field(static=True)
    w_capacity: int | None = eqx.field(static=True)
    x_capacity: int | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        address_plan: MortonAddressPlan,
        /,
        *,
        leaf_capacity: int,
        balanced: bool = False,
        separation_padding: float = 0.0,
        u_capacity: int | None = None,
        v_capacity: int | None = None,
        w_capacity: int | None = None,
        x_capacity: int | None = None,
    ) -> None:
        if not isinstance(address_plan, MortonAddressPlan):
            raise TypeError("address_plan must be a MortonAddressPlan.")
        if not isinstance(balanced, bool):
            raise TypeError("balanced must be a bool.")
        if any(address_plan.periodic_axes):
            raise ValueError(
                "Adaptive octree interaction lists are free-space; periodic axes are unsupported."
            )
        capacity = index(leaf_capacity)
        padding = float(separation_padding)
        list_capacities = tuple(
            None if value is None else index(value)
            for value in (u_capacity, v_capacity, w_capacity, x_capacity)
        )
        if capacity < 1:
            raise ValueError("leaf_capacity must be positive.")
        if not math.isfinite(padding) or padding < 0.0:
            raise ValueError("separation_padding must be finite and nonnegative.")
        if any(value is not None and value < 1 for value in list_capacities):
            raise ValueError("Interaction list capacities must be positive.")
        self.address_plan = address_plan
        self.leaf_capacity = capacity
        self.balanced = balanced
        self.separation_padding = padding
        self.u_capacity, self.v_capacity, self.w_capacity, self.x_capacity = (
            list_capacities
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "adaptive-octree-plan",
                "address_plan_id": address_plan.plan_id,
                "leaf_capacity": capacity,
                "balanced": balanced,
                "separation_padding": padding,
                "list_capacities": list(list_capacities),
            }
        )

    def prepare(self, points: ArrayLike, /) -> AdaptiveOctree:
        """Build the tree and its interaction lists from host point coordinates."""
        address = self.address_plan
        positions = np.asarray(points, dtype=np.float64)
        if (
            positions.ndim != 2
            or positions.shape[0] == 0
            or positions.shape[1] != address.dimension
        ):
            raise ValueError(
                f"points must have shape (count, {address.dimension}) with count >= 1."
            )
        encoding = address.encode(jnp.asarray(positions))
        if not bool(jnp.all(encoding.in_domain)):
            raise ValueError("Octree points must be finite and lie in [lower, upper).")
        raw_codes = np.asarray(encoding.codes, dtype=np.uint64)
        order = np.lexsort((np.arange(positions.shape[0]), raw_codes))
        codes = raw_codes[order]
        refined = _subdivided_prefixes(codes, address, self.leaf_capacity)
        if self.balanced:
            refined = _balanced_prefixes(refined, address)
        table = _node_table(refined, codes, address)
        node_count = table.levels.shape[0]
        leaf_mask = table.child_counts == 0
        leaf_order = np.argsort(table.code_starts[leaf_mask], kind="stable")
        leaf_nodes = np.flatnonzero(leaf_mask)[leaf_order].astype(np.int32)
        leaf_code_starts = table.code_starts[leaf_nodes]
        point_leaves = np.empty((positions.shape[0],), dtype=np.int32)
        point_leaves[order] = leaf_nodes[
            np.searchsorted(leaf_code_starts, codes, side="right") - 1
        ]
        occupancy = (table.point_ends - table.point_starts)[leaf_nodes]
        maximum_occupancy = np.max(occupancy).item()
        pairs = _interaction_pairs(table, address, self.separation_padding)
        lists = {
            name: _interaction_list(*pairs[name], node_count, capacity)
            for name, capacity in (
                ("u", self.u_capacity),
                ("v", self.v_capacity),
                ("w", self.w_capacity),
                ("x", self.x_capacity),
            )
        }
        overflow = any(bool(entry.overflow) for entry in lists.values())
        geometry = address.cell_geometry(
            jnp.asarray(table.prefixes), jnp.asarray(table.levels)
        )
        evidence = AdaptiveOctreeEvidence(
            node_count=jnp.asarray(node_count, dtype=jnp.int32),
            leaf_count=jnp.asarray(leaf_nodes.size, dtype=jnp.int32),
            maximum_level=jnp.asarray(len(table.level_offsets) - 2, dtype=jnp.int32),
            maximum_leaf_occupancy=jnp.asarray(maximum_occupancy, dtype=jnp.int32),
            leaf_capacity_satisfied=jnp.asarray(maximum_occupancy <= self.leaf_capacity),
            successful=jnp.asarray(not overflow),
        )
        return AdaptiveOctree(
            address_plan=address,
            node_prefixes=jnp.asarray(table.prefixes),
            node_levels=jnp.asarray(table.levels),
            node_parents=jnp.asarray(table.parents),
            node_first_children=jnp.asarray(table.first_children),
            node_child_counts=jnp.asarray(table.child_counts),
            node_point_starts=jnp.asarray(table.point_starts),
            node_point_ends=jnp.asarray(table.point_ends),
            node_centers=geometry.center,
            node_half_widths=geometry.half_width,
            leaf_nodes=jnp.asarray(leaf_nodes),
            leaf_code_starts=jnp.asarray(leaf_code_starts),
            point_order=jnp.asarray(order.astype(np.int32)),
            point_leaves=jnp.asarray(point_leaves),
            u_list=lists["u"],
            v_list=lists["v"],
            w_list=lists["w"],
            x_list=lists["x"],
            evidence=evidence,
            level_offsets=table.level_offsets,
            maximum_leaf_occupancy=maximum_occupancy,
            separation_padding=self.separation_padding,
            tree_id=canonical_fingerprint(
                {
                    "kind": "adaptive-octree",
                    "plan": self.plan_id,
                    "points": array_tree_fingerprint(positions),
                }
            ),
        )


__all__ = [
    "AdaptiveOctree",
    "AdaptiveOctreeEvidence",
    "AdaptiveOctreeInteractionList",
    "AdaptiveOctreePlan",
]
