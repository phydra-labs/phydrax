#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Device connected components over fixed-capacity relations and their transitions.

Labeling follows FastSV (Zhang, Azad & Hu, "FastSV: A distributed-memory connected
component algorithm with fast convergence", SIAM Conference on Parallel Processing
for Scientific Computing, 2020). With the parent vector ``f`` (initially
``f[u] = u``) and the grandparent vector ``gf = f[f]``, every round computes the
minimum neighbour grandparent ``mngf[u] = min{gf[v] : (u, v) participates}`` and
applies, all from the round-start values,

- stochastic hooking ``f[f[u]] <- min(f[f[u]], mngf[u])``,
- aggressive hooking ``f[u] <- min(f[u], mngf[u])``,
- shortcutting ``f[u] <- min(f[u], gf[u])``.

Parents only decrease and always name an entity of the same component, so
``f[u] <= u`` and every tree root is the minimum entity index of its tree. The
iteration stops when ``gf`` is unchanged by a round, at which point ``f`` is a star
forest whose trees are the components, or when the declared round bound is reached
(reported as not converged).

Roots are compacted by an exclusive prefix sum of the root flags, numbering
components in increasing root index without sorting. More components than the
declared capacity is a refusal: slots at or above the capacity are never issued.

The transition between two labelings groups the observed ``(old, new)`` label
pairs of entities labeled in both through a fixed-capacity key grouping; no
``old_capacity x new_capacity`` table is formed.
"""

from __future__ import annotations

from enum import IntEnum
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import nonnegative_integer
from ..sparse import EdgeRelation, KeyGroupPlan
from ..typing import as_array, Bool, Dim, Float, Int32, parse, Scalar, Scope, Size


_INT32_MAX = int(np.iinfo(np.int32).max)
_INT64_MAX = int(np.iinfo(np.int64).max)


class _EntityDim(Dim, minimum=1):
    """Number of labeled entities (relation vertices)."""


class _EdgeDim(Dim):
    """Number of relation routes."""


class _ComponentDim(Dim, minimum=1):
    """Compact component slot capacity of one labeling."""


class _OldComponentDim(Dim, minimum=1):
    """Compact component slot capacity of the earlier labeling."""


class _NewComponentDim(Dim, minimum=1):
    """Compact component slot capacity of the later labeling."""


class _PairDim(Dim, minimum=1):
    """Capacity of observed ``(old, new)`` component pairs."""


class ComponentLabelingStatus(IntEnum):
    """Outcome of one labeling; ``NOT_CONVERGED`` outranks ``CAPACITY_EXCEEDED``."""

    CONVERGED = 0
    NOT_CONVERGED = 1
    CAPACITY_EXCEEDED = 2


class ComponentEventKind(IntEnum):
    """Topological role of one component slot across a labeling transition.

    New slots take ``NONE``, ``CONTINUE``, ``MERGE``, ``SPLIT``, ``CREATE`` or
    ``RECONNECT``; old slots take ``NONE``, ``CONTINUE``, ``MERGE``, ``SPLIT``,
    ``VANISH`` or ``RECONNECT``. ``RECONNECT`` marks a slot whose pair graph
    neighbourhood both merges and splits.
    """

    NONE = 0
    CONTINUE = 1
    MERGE = 2
    SPLIT = 3
    CREATE = 4
    RECONNECT = 5
    VANISH = 6


def _positive_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def grid_adjacency_relation(
    cell_shape: tuple[int, ...], periodic: tuple[bool, ...], /
) -> EdgeRelation:
    """Return the undirected face-neighbour edges of a structured cell grid.

    Cells are numbered in row-major (C) order. Every undirected edge is listed once
    as ``(lower index, upper index)`` in ascending lexicographic order. Periodic
    axes wrap; an axis of length 1 contributes no edge and a periodic axis of
    length 2 contributes its single neighbour pair once.
    """
    if not isinstance(cell_shape, tuple) or not isinstance(periodic, tuple):
        raise TypeError("cell_shape and periodic must be tuples.")
    if not cell_shape:
        raise ValueError("cell_shape must have at least one axis.")
    if len(periodic) != len(cell_shape):
        raise ValueError("periodic must have one entry per cell_shape axis.")
    if any(not isinstance(flag, bool) for flag in periodic):
        raise TypeError("periodic entries must be bool.")
    shape = tuple(_positive_integer(size, "cell_shape entries") for size in cell_shape)
    cell_count = int(np.prod(shape, dtype=np.int64))
    if cell_count > _INT32_MAX:
        raise ValueError("cell_shape exceeds the int32 entity index range.")
    index = np.arange(cell_count, dtype=np.int64).reshape(shape)
    lower_parts: list[np.ndarray] = []
    upper_parts: list[np.ndarray] = []
    for axis, (size, wraps) in enumerate(zip(shape, periodic)):
        lower_parts.append(np.take(index, np.arange(size - 1), axis=axis).ravel())
        upper_parts.append(np.take(index, np.arange(1, size), axis=axis).ravel())
        if wraps and size > 2:
            lower_parts.append(np.take(index, 0, axis=axis).ravel())
            upper_parts.append(np.take(index, size - 1, axis=axis).ravel())
    lower = np.concatenate(lower_parts)
    upper = np.concatenate(upper_parts)
    order = np.lexsort((upper, lower))
    return EdgeRelation(
        lower[order].astype(np.int32),
        upper[order].astype(np.int32),
        source_size=cell_count,
        target_size=cell_count,
    )


class ComponentLabelingResult(StrictModule, NonTrainableState):
    """Roots, compact labels, component slots, convergence and capacity evidence.

    ``root`` is the minimum entity index of each active entity's component and
    ``label`` its compact slot, both ``-1`` for inactive entities. Slots number
    components in increasing root index. ``count`` is the true component count
    even when it exceeds the capacity; overflowing components then carry label
    ``-1`` and the result is unsuccessful.
    """

    __strict_contract__ = True

    root: Int32[_EntityDim]
    label: Int32[_EntityDim]
    count: Int32[Scalar]
    component_root: Int32[_ComponentDim]
    component_size: Int32[_ComponentDim]
    rounds: Int32[Scalar]
    converged: Bool[Scalar]
    overflow: Bool[Scalar]
    successful: Bool[Scalar]
    status: Int32[Scalar]
    plan_id: str = eqx.field(static=True)

    @property
    def component_mask(self) -> Array:
        """Return which compact component slots hold a component."""
        return self.component_root >= 0


class ConnectedComponentPlan(StrictModule):
    """Fixed-capacity FastSV connected-component labeling over a square relation.

    Each relation route is an undirected edge; it participates when it is valid,
    enabled by the call's ``edge_valid`` and both endpoints are active.
    """

    __strict_contract__ = True

    relation: EdgeRelation
    entity_count: Size[_EntityDim] = eqx.field(static=True)
    component_capacity: Size[_ComponentDim] = eqx.field(static=True)
    maximum_rounds: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        relation: EdgeRelation,
        /,
        *,
        component_capacity: int,
        maximum_rounds: int,
    ) -> None:
        if not isinstance(relation, EdgeRelation):
            raise TypeError("relation must be a phydrax.sparse.EdgeRelation.")
        if relation.source_size != relation.target_size:
            raise ValueError("relation must map one entity space onto itself.")
        entities = _positive_integer(relation.source_size, "relation entity count")
        if entities > _INT32_MAX:
            raise ValueError("relation entity count exceeds the int32 index range.")
        capacity = _positive_integer(component_capacity, "component_capacity")
        rounds = _positive_integer(maximum_rounds, "maximum_rounds")
        if rounds > _INT32_MAX:
            raise ValueError("maximum_rounds exceeds the int32 round-counter range.")
        self.relation = relation
        self.entity_count = entities
        self.component_capacity = capacity
        self.maximum_rounds = rounds
        self.plan_id = canonical_fingerprint(
            {
                "kind": "connected-component-plan",
                "entity_count": entities,
                "source_indices": relation.source_indices,
                "target_indices": relation.target_indices,
                "valid": relation.valid,
                "component_capacity": capacity,
                "maximum_rounds": rounds,
            }
        )

    def label(
        self, active: ArrayLike, /, *, edge_valid: ArrayLike | None = None
    ) -> ComponentLabelingResult:
        """Label the components of the active entities over participating edges."""
        scope = Scope()
        parse(self.entity_count, Size[_EntityDim], "entity_count", scope=scope)
        parse(self.relation.capacity, Size[_EdgeDim], "edge_capacity", scope=scope)
        active_mask = as_array(active, Bool[_EntityDim], "active", scope=scope)
        enabled = self.relation.valid
        if edge_valid is not None:
            enabled = enabled & as_array(
                edge_valid, Bool[_EdgeDim], "edge_valid", scope=scope
            )
        parents, rounds, converged = _fastsv_parents(
            self.relation, enabled, active_mask, self.maximum_rounds
        )
        return _compact_components(
            parents, active_mask, rounds, converged, self.component_capacity, self.plan_id
        )


def _fastsv_parents(
    relation: EdgeRelation, enabled: Array, active: Array, maximum_rounds: int, /
) -> tuple[Array, Array, Array]:
    """Run bounded FastSV rounds; return parents, rounds and the convergence flag."""
    entity_count = relation.source_size
    source = jnp.where(relation.valid, relation.source_indices, 0)
    target = jnp.where(relation.valid, relation.target_indices, 0)
    participating = enabled & active[source] & active[target]
    # Both edge directions as one route list. `sparse.route_reduce(..., "min")`
    # zero-fills targets without routes, which would hook isolated entities onto
    # entity 0, so the neighbour minimum uses an explicit int32-max identity.
    owners = jnp.concatenate((source, target))
    neighbours = jnp.concatenate((target, source))
    routes_on = jnp.concatenate((participating, participating))
    empty = jnp.asarray(_INT32_MAX, dtype=jnp.int32)

    def neighbour_minimum(grandparents: Array, /) -> Array:
        values = jnp.where(routes_on, grandparents[neighbours], empty)
        return jax.ops.segment_min(values, owners, num_segments=entity_count)

    def continues(carry: tuple[Array, Array, Array, Array], /) -> Array:
        _, _, rounds, changed = carry
        return changed & (rounds < maximum_rounds)

    def fastsv_round(
        carry: tuple[Array, Array, Array, Array],
        /,
    ) -> tuple[Array, Array, Array, Array]:
        parents, grandparents, rounds, _ = carry
        minimum = neighbour_minimum(grandparents)
        hooked = parents.at[parents].min(minimum)
        updated = jnp.minimum(jnp.minimum(hooked, minimum), grandparents)
        updated_grandparents = updated[updated]
        changed = jnp.any(updated_grandparents != grandparents)
        return updated, updated_grandparents, rounds + 1, changed

    initial = jnp.arange(entity_count, dtype=jnp.int32)
    parents, _, rounds, changed = jax.lax.while_loop(
        continues,
        fastsv_round,
        (
            initial,
            initial,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(True),
        ),
    )
    return parents, rounds, ~changed


def _compact_components(
    parents: Array,
    active: Array,
    rounds: Array,
    converged: Array,
    capacity: int,
    plan_id: str,
    /,
) -> ComponentLabelingResult:
    """Number roots by exclusive prefix sum and refuse slots beyond the capacity."""
    entity_count = parents.shape[0]
    entities = jnp.arange(entity_count, dtype=jnp.int32)
    root_flag = active & (parents == entities)
    flags = root_flag.astype(jnp.int32)
    slot_of_root = jnp.cumsum(flags, dtype=jnp.int32) - flags
    count = jnp.sum(flags, dtype=jnp.int32)
    # An unconverged forest may leave entities below a non-root parent; they keep
    # label -1 rather than borrowing another component's slot.
    slot = slot_of_root[parents]
    labeled = active & root_flag[parents] & (slot < capacity)
    label = jnp.where(labeled, slot, -1)
    component_root = (
        jnp.full((capacity,), -1, dtype=jnp.int32)
        .at[jnp.where(root_flag, slot_of_root, capacity)]
        .set(entities, mode="drop")
    )
    component_size = jax.ops.segment_sum(
        labeled.astype(jnp.int32), jnp.where(labeled, label, 0), num_segments=capacity
    )
    overflow = count > capacity
    status = jnp.where(
        ~converged,
        jnp.int32(ComponentLabelingStatus.NOT_CONVERGED),
        jnp.where(
            overflow,
            jnp.int32(ComponentLabelingStatus.CAPACITY_EXCEEDED),
            jnp.int32(ComponentLabelingStatus.CONVERGED),
        ),
    )
    return ComponentLabelingResult(
        root=jnp.where(active, parents, -1),
        label=label,
        count=count,
        component_root=component_root,
        component_size=component_size,
        rounds=rounds,
        converged=converged,
        overflow=overflow,
        successful=converged & ~overflow,
        status=status,
        plan_id=plan_id,
    )


class ComponentTransitionResult(StrictModule, NonTrainableState):
    """Observed component pairs, overlaps, parent/child counts and event kinds.

    Pairs are listed in ascending ``old * new_capacity + new`` order with ``-1``
    padding. ``merge_count`` and ``create_count`` count new slots with those
    events, ``split_count`` and ``vanish_count`` old slots, and
    ``reconnect_count`` new slots marked ``RECONNECT``.
    """

    __strict_contract__ = True

    pair_old: Int32[_PairDim]
    pair_new: Int32[_PairDim]
    pair_overlap: Float[_PairDim]
    pair_active: Bool[_PairDim]
    old_child_count: Int32[_OldComponentDim]
    new_parent_count: Int32[_NewComponentDim]
    old_event: Int32[_OldComponentDim]
    new_event: Int32[_NewComponentDim]
    merge_count: Int32[Scalar]
    split_count: Int32[Scalar]
    create_count: Int32[Scalar]
    vanish_count: Int32[Scalar]
    reconnect_count: Int32[Scalar]
    pair_overflow: Bool[Scalar]
    successful: Bool[Scalar]
    plan_id: str = eqx.field(static=True)


class ComponentTransitionPlan(StrictModule):
    """Fixed-capacity correspondence between two component labelings.

    ``reciprocal_dominance`` removes an overlap pair only when both of its
    endpoints have a strictly larger overlap partner. This preserves every
    branch of a genuine merge or split while discarding minor cross-overlaps
    caused by a moving discrete partition.
    """

    __strict_contract__ = True

    grouping: KeyGroupPlan
    entity_count: Size[_EntityDim] = eqx.field(static=True)
    old_capacity: Size[_OldComponentDim] = eqx.field(static=True)
    new_capacity: Size[_NewComponentDim] = eqx.field(static=True)
    pair_capacity: Size[_PairDim] = eqx.field(static=True)
    reciprocal_dominance: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        entity_count: int,
        /,
        *,
        old_capacity: int,
        new_capacity: int,
        pair_capacity: int,
        reciprocal_dominance: bool = False,
    ) -> None:
        entities = _positive_integer(entity_count, "entity_count")
        old = _positive_integer(old_capacity, "old_capacity")
        new = _positive_integer(new_capacity, "new_capacity")
        pairs = _positive_integer(pair_capacity, "pair_capacity")
        if not isinstance(reciprocal_dominance, bool):
            raise TypeError("reciprocal_dominance must be a bool.")
        if entities > _INT32_MAX:
            raise ValueError("entity_count exceeds the int32 index range.")
        if old * new - 1 > _INT64_MAX:
            raise ValueError("old_capacity * new_capacity exceeds the int64 key range.")
        self.grouping = KeyGroupPlan(entities, pairs, old * new - 1)
        self.entity_count = entities
        self.old_capacity = old
        self.new_capacity = new
        self.pair_capacity = pairs
        self.reciprocal_dominance = reciprocal_dominance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "component-transition-plan",
                "entity_count": entities,
                "old_capacity": old,
                "new_capacity": new,
                "pair_capacity": pairs,
                "reciprocal_dominance": reciprocal_dominance,
            }
        )

    @property
    def key_dtype(self) -> DTypeLike:
        """Return the narrowest pair-key dtype holding every ``(old, new)`` key."""
        bound = nonnegative_integer(
            self.grouping.key_upper_bound, "component pair key bound"
        )
        if bound < 0 or bound > _INT64_MAX:
            raise ValueError(
                "component pair key bound must fit the nonnegative int64 range."
            )
        if bound <= _INT32_MAX:
            return jnp.int32
        return jnp.int64

    def transition(
        self,
        old: ComponentLabelingResult,
        new: ComponentLabelingResult,
        weight: ArrayLike,
        /,
    ) -> ComponentTransitionResult:
        """Pair the components of two labelings by the entities labeled in both.

        ``weight`` is the per-entity overlap measure (for example gas volume); each
        pair's overlap is the sum over its shared entities.
        """
        if not isinstance(old, ComponentLabelingResult) or not isinstance(
            new, ComponentLabelingResult
        ):
            raise TypeError("old and new must be ComponentLabelingResult values.")
        scope = Scope()
        parse(self.entity_count, Size[_EntityDim], "entity_count", scope=scope)
        parse(self.old_capacity, Size[_OldComponentDim], "old_capacity", scope=scope)
        parse(self.new_capacity, Size[_NewComponentDim], "new_capacity", scope=scope)
        parse(old.label, Int32[_EntityDim], "old.label", scope=scope)
        parse(new.label, Int32[_EntityDim], "new.label", scope=scope)
        parse(
            old.component_root, Int32[_OldComponentDim], "old.component_root", scope=scope
        )
        parse(
            new.component_root, Int32[_NewComponentDim], "new.component_root", scope=scope
        )
        weights = as_array(weight, Float[_EntityDim], "weight", scope=scope)
        return self._pair_components(old, new, weights)

    def _pair_components(
        self,
        old: ComponentLabelingResult,
        new: ComponentLabelingResult,
        weights: Array,
        /,
    ) -> ComponentTransitionResult:
        shared = (old.label >= 0) & (new.label >= 0)
        keys = old.label.astype(self.key_dtype) * self.new_capacity + new.label.astype(
            self.key_dtype
        )
        groups = self.grouping.build(keys, shared)
        slots = groups.item_group_slots
        grouped = shared & (slots >= 0) & (slots < self.pair_capacity)
        pair_overlap = jax.ops.segment_sum(
            jnp.where(grouped, weights, jnp.zeros((), dtype=weights.dtype)),
            jnp.where(grouped, slots, 0),
            num_segments=self.pair_capacity,
        )
        pair_active = groups.group_active
        pair_old = jnp.where(
            pair_active, groups.group_keys // self.new_capacity, -1
        ).astype(jnp.int32)
        pair_new = jnp.where(
            pair_active, groups.group_keys % self.new_capacity, -1
        ).astype(jnp.int32)
        if self.reciprocal_dominance:
            pair_old, pair_new, pair_overlap, pair_active = _reciprocal_dominant_pairs(
                pair_old,
                pair_new,
                pair_overlap,
                pair_active,
                self.old_capacity,
                self.new_capacity,
            )
        old_event, new_event, old_children, new_parents = _classify_events(
            pair_old,
            pair_new,
            pair_active,
            old.component_mask,
            new.component_mask,
        )
        return ComponentTransitionResult(
            pair_old=pair_old,
            pair_new=pair_new,
            pair_overlap=pair_overlap,
            pair_active=pair_active,
            old_child_count=old_children,
            new_parent_count=new_parents,
            old_event=old_event,
            new_event=new_event,
            merge_count=_event_count(new_event, ComponentEventKind.MERGE),
            split_count=_event_count(old_event, ComponentEventKind.SPLIT),
            create_count=_event_count(new_event, ComponentEventKind.CREATE),
            vanish_count=_event_count(old_event, ComponentEventKind.VANISH),
            reconnect_count=_event_count(new_event, ComponentEventKind.RECONNECT),
            pair_overflow=groups.evidence.group_overflow,
            successful=old.successful & new.successful & groups.evidence.successful,
            plan_id=self.plan_id,
        )


def _event_count(events: Array, kind: ComponentEventKind, /) -> Array:
    return jnp.sum(events == jnp.int32(kind), dtype=jnp.int32)


def _reciprocal_dominant_pairs(
    pair_old: Array,
    pair_new: Array,
    pair_overlap: Array,
    pair_active: Array,
    old_capacity: int,
    new_capacity: int,
    /,
) -> tuple[Array, Array, Array, Array]:
    """Compact pairs that are maximal for at least one endpoint."""

    safe_old = jnp.where(pair_active, pair_old, 0)
    safe_new = jnp.where(pair_active, pair_new, 0)
    negative = jnp.asarray(-jnp.inf, dtype=pair_overlap.dtype)
    old_maximum = jax.ops.segment_max(
        jnp.where(pair_active, pair_overlap, negative),
        safe_old,
        num_segments=old_capacity,
    )
    new_maximum = jax.ops.segment_max(
        jnp.where(pair_active, pair_overlap, negative),
        safe_new,
        num_segments=new_capacity,
    )
    retained = pair_active & (
        (pair_overlap >= old_maximum[safe_old]) | (pair_overlap >= new_maximum[safe_new])
    )
    capacity = pair_active.size
    rank = jnp.cumsum(retained.astype(jnp.int32)) - 1
    target = jnp.where(retained, rank, capacity)
    compact_old = (
        jnp.full((capacity + 1,), -1, dtype=jnp.int32).at[target].set(pair_old)[:capacity]
    )
    compact_new = (
        jnp.full((capacity + 1,), -1, dtype=jnp.int32).at[target].set(pair_new)[:capacity]
    )
    compact_overlap = (
        jnp.zeros((capacity + 1,), dtype=pair_overlap.dtype)
        .at[target]
        .set(pair_overlap)[:capacity]
    )
    compact_active = jnp.arange(capacity) < jnp.sum(retained, dtype=jnp.int32)
    return compact_old, compact_new, compact_overlap, compact_active


def _classify_events(
    pair_old: Array,
    pair_new: Array,
    pair_active: Array,
    old_active: Array,
    new_active: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    """Classify slots from the bipartite pair graph's degrees and neighbour degrees."""
    old_capacity = old_active.shape[0]
    new_capacity = new_active.shape[0]
    safe_old = jnp.where(pair_active, pair_old, 0)
    safe_new = jnp.where(pair_active, pair_new, 0)
    ones = pair_active.astype(jnp.int32)
    old_children = jax.ops.segment_sum(ones, safe_old, num_segments=old_capacity)
    new_parents = jax.ops.segment_sum(ones, safe_new, num_segments=new_capacity)
    zero = jnp.zeros((), dtype=jnp.int32)
    # Largest child count among a new slot's parents, and vice versa.
    parent_fanout = jax.ops.segment_max(
        jnp.where(pair_active, old_children[safe_old], zero),
        safe_new,
        num_segments=new_capacity,
    )
    child_fanin = jax.ops.segment_max(
        jnp.where(pair_active, new_parents[safe_new], zero),
        safe_old,
        num_segments=old_capacity,
    )
    new_event = _slot_events(
        new_active,
        new_parents,
        parent_fanout,
        isolated=ComponentEventKind.CREATE,
        gathered=ComponentEventKind.MERGE,
        spread=ComponentEventKind.SPLIT,
    )
    old_event = _slot_events(
        old_active,
        old_children,
        child_fanin,
        isolated=ComponentEventKind.VANISH,
        gathered=ComponentEventKind.SPLIT,
        spread=ComponentEventKind.MERGE,
    )
    return old_event, new_event, old_children, new_parents


def _slot_events(
    active: Array,
    degree: Array,
    neighbour_degree: Array,
    /,
    *,
    isolated: ComponentEventKind,
    gathered: ComponentEventKind,
    spread: ComponentEventKind,
) -> Array:
    """Map a slot's pair degree and its neighbours' largest degree to an event.

    ``gathered`` is the event of a slot with several neighbours that each have
    only this slot, ``spread`` that of a slot with one neighbour that has several;
    several neighbours with at least one of degree two or more reconnect.
    """
    many_here = degree >= 2
    many_there = neighbour_degree >= 2
    event = jnp.where(
        many_here,
        jnp.where(many_there, ComponentEventKind.RECONNECT, gathered),
        jnp.where(many_there, spread, ComponentEventKind.CONTINUE),
    )
    event = jnp.where(degree == 0, isolated, event)
    return jnp.where(active, event, ComponentEventKind.NONE).astype(jnp.int32)


__all__ = [
    "ComponentEventKind",
    "ComponentLabelingResult",
    "ComponentLabelingStatus",
    "ConnectedComponentPlan",
    "ComponentTransitionPlan",
    "ComponentTransitionResult",
    "grid_adjacency_relation",
]
