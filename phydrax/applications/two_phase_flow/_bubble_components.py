#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bubble identity from device connected components of the VOF gas phase.

Every accepted step labels the gas support with
`phydrax.topology.ConnectedComponentPlan`. A cell belongs to the gas support
when its gas fraction ``1 - alpha`` is at least the declared
``gas_threshold``, which is the mixed-cell policy. With multi-marker VOF, gas
cells connect only to face neighbors of the same dominant marker color.
Components connected to a declared vent side form the atmosphere, and they
share the reserved stable identity `ATMOSPHERE_ID`.

Stable identities follow a deterministic rule on the observed overlap pairs of
`phydrax.topology.ComponentTransitionPlan`:

- a component that continues a single non-atmosphere parent keeps its
  parent's identity;
- every other non-atmosphere component (merge, split, reconnect, creation,
  entrainment) receives a fresh identity. Fresh identities are allocated in
  increasing root order from ``next_id``.

A proposal whose set of non-atmosphere identities changed is a topology event.
The device only detects it. The host-side compartment transaction (see
`_bubble_thermodynamics`) commits it and journals it, and no derivative is
claimed across it.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_integer
from ...topology import (
    ComponentEventKind,
    ComponentLabelingResult,
    ComponentTransitionPlan,
    ComponentTransitionResult,
    ConnectedComponentPlan,
    grid_adjacency_relation,
)
from ...typing import parse
from ._vof import PreparedIncompressibleTwoPhaseVOF


ATMOSPHERE_ID = 0
"""Stable identity shared by every gas component connected to a vent side."""

type BoundarySide = Literal["lower", "upper"]


class BubbleIdentityStatus(IntEnum):
    """Outcome of one bubble-identity proposal."""

    CONTINUED = 0
    TOPOLOGY_EVENT = 1
    LABELING_FAILED = 2
    TRANSITION_OVERFLOW = 3


class BubbleComponentLabels(StrictModule):
    """Compact gas components of one volume-fraction field.

    Slot arrays have the component capacity. ``volume`` is the gas volume of
    each component, ``centroid`` its gas-volume centroid (circular mean along
    periodic axes) and ``atmosphere`` marks components that touch a vent side.
    """

    labeling: ComponentLabelingResult
    gas_volume: Array
    color: Array
    volume: Array
    centroid: Array
    atmosphere: Array


class BubbleComponentState(StrictModule):
    """Committed bubble identities of the current accepted state.

    ``slot_ids[k]`` is the stable identity of compact component ``k``
    (`ATMOSPHERE_ID` for atmosphere slots, ``-1`` for empty slots).
    ``next_id`` is the next fresh identity and ``epoch`` counts committed
    topology events.
    """

    labels: BubbleComponentLabels
    slot_ids: Array
    next_id: Array
    epoch: Array


class BubbleTopologyEvidence(StrictModule):
    """Counts and status of one bubble-identity proposal."""

    component_count: Array
    bubble_count: Array
    atmosphere_count: Array
    merge_count: Array
    split_count: Array
    create_count: Array
    vanish_count: Array
    reconnect_count: Array
    fresh_count: Array
    consumed_count: Array
    converged: Array
    overflow: Array
    pair_overflow: Array
    topology_changed: Array
    derivative_available: Array
    status: Array


class BubbleIdentityEvent(StrictModule):
    """Slot-sized record of one proposal, enough to journal its events.

    It omits every cell-sized array, so a continuation state can carry it.
    """

    transition: ComponentTransitionResult
    slot_ids: Array
    fresh: Array
    consumed: Array
    volume: Array
    atmosphere: Array
    topology_changed: Array


class BubbleIdentityProposal(StrictModule):
    """Device proposal of the next identity state and its transition."""

    labels: BubbleComponentLabels
    transition: ComponentTransitionResult
    slot_ids: Array
    fresh: Array
    parent_slot: Array
    consumed: Array
    previous_ids: Array
    next_id: Array
    evidence: BubbleTopologyEvidence

    @property
    def event(self) -> BubbleIdentityEvent:
        return BubbleIdentityEvent(
            transition=self.transition,
            slot_ids=self.slot_ids,
            fresh=self.fresh,
            consumed=self.consumed,
            volume=self.labels.volume,
            atmosphere=self.labels.atmosphere,
            topology_changed=self.evidence.topology_changed,
        )

    def commit(self, previous: BubbleComponentState, /) -> BubbleComponentState:
        """Identity state after this proposal (the epoch advances on events)."""

        changed = self.evidence.topology_changed
        return BubbleComponentState(
            labels=self.labels,
            slot_ids=self.slot_ids,
            next_id=self.next_id,
            epoch=previous.epoch + changed.astype(jnp.int32),
        )


def _vent_mask(
    cell_shape: tuple[int, ...], sides: tuple[tuple[int, BoundarySide], ...], /
) -> np.ndarray:
    mask = np.zeros(cell_shape, dtype=np.bool_)
    for axis, side in sides:
        index: list[slice | int] = [slice(None)] * len(cell_shape)
        index[axis] = 0 if side == "lower" else cell_shape[axis] - 1
        mask[tuple(index)] = True
    return mask


class BubbleComponentPlan(StrictModule, NonTrainableState):
    """Device bubble identity on the cells of one structured two-phase grid."""

    labeling: ConnectedComponentPlan
    transition: ComponentTransitionPlan
    marker_transition: ComponentTransitionPlan
    cell_measure: Array
    cell_centers: Array
    vent_mask: Array
    axis_lower: Array
    axis_period: Array
    gas_threshold: float = eqx.field(static=True)
    vent_sides: tuple[tuple[int, BoundarySide], ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    cell_shape: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        two_phase: PreparedIncompressibleTwoPhaseVOF,
        /,
        *,
        component_capacity: int,
        maximum_rounds: int,
        pair_capacity: int,
        gas_threshold: float = 1.0e-6,
        vent_sides: tuple[tuple[int, BoundarySide], ...] = (),
    ) -> None:
        if not isinstance(two_phase, PreparedIncompressibleTwoPhaseVOF):
            raise TypeError("two_phase must be PreparedIncompressibleTwoPhaseVOF.")
        capacity = positive_integer(component_capacity, "component_capacity")
        rounds = positive_integer(maximum_rounds, "maximum_rounds")
        pairs = positive_integer(pair_capacity, "pair_capacity")
        threshold = float(gas_threshold)
        if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("gas_threshold must lie in (0, 1].")
        discretization = two_phase.plan.discretization
        cell_shape = tuple(discretization.cell_shape)
        axes = discretization.grid.structured_axes
        periodic = tuple(axis.periodic for axis in axes)
        sides = []
        for entry in vent_sides:
            axis, side = entry
            if not 0 <= axis < len(cell_shape):
                raise ValueError("vent side axis lies outside the grid.")
            if periodic[axis]:
                raise ValueError("vent sides must lie on nonperiodic axes.")
            sides.append((int(axis), parse(side, BoundarySide, "vent side")))
        canonical_sides = tuple(sorted(set(sides)))
        entity_count = int(np.prod(cell_shape))
        relation = grid_adjacency_relation(cell_shape, periodic)
        labeling = ConnectedComponentPlan(
            relation, component_capacity=capacity, maximum_rounds=rounds
        )
        transition = ComponentTransitionPlan(
            entity_count,
            old_capacity=capacity,
            new_capacity=capacity,
            pair_capacity=pairs,
        )
        marker_transition = ComponentTransitionPlan(
            entity_count,
            old_capacity=capacity,
            new_capacity=capacity,
            pair_capacity=pairs,
            reciprocal_dominance=True,
        )
        bounds = np.asarray(
            [np.asarray(axis.bounds, dtype=np.float64) for axis in axes],
            dtype=np.float64,
        )
        dtype = discretization.cell_volumes.dtype
        self.labeling = labeling
        self.transition = transition
        self.marker_transition = marker_transition
        self.cell_measure = jnp.asarray(two_phase.cell_fluid_measure, dtype=dtype)
        self.cell_centers = jnp.asarray(discretization.cell_centers, dtype=dtype)
        self.vent_mask = jnp.asarray(_vent_mask(cell_shape, canonical_sides))
        self.axis_lower = jnp.asarray(bounds[:, 0], dtype=dtype)
        self.axis_period = jnp.asarray(bounds[:, 1] - bounds[:, 0], dtype=dtype)
        self.gas_threshold = threshold
        self.vent_sides = canonical_sides
        self.periodic = periodic
        self.cell_shape = cell_shape
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bubble-component-plan",
                "two_phase": two_phase.prepared_id,
                "labeling": labeling.plan_id,
                "transition": transition.plan_id,
                "marker_transition": marker_transition.plan_id,
                "gas_threshold": threshold,
                "vent_sides": [list(side) for side in canonical_sides],
            }
        )

    @property
    def component_capacity(self) -> int:
        return self.labeling.component_capacity

    def gas_content(self, alpha: ArrayLike, /) -> Array:
        """Gas volume content ``V (1 - alpha)`` of every cell."""

        alpha_ = jnp.asarray(alpha, dtype=self.cell_measure.dtype)
        if alpha_.shape != self.cell_shape:
            raise ValueError("alpha must match the two-phase cell shape.")
        return self.cell_measure * (1.0 - alpha_)

    def _centroids(self, label: Array, gas: Array, volume: Array, /) -> Array:
        capacity = self.component_capacity
        safe = jnp.where(label >= 0, label, capacity)
        weights = jnp.where(label >= 0, gas, 0.0).reshape(-1)
        segments = safe.reshape(-1)
        denominator = jnp.where(volume > 0.0, volume, 1.0)
        columns = []
        for axis, periodic in enumerate(self.periodic):
            coordinate = self.cell_centers[..., axis].reshape(-1)
            if periodic:
                angle = (
                    2.0
                    * jnp.pi
                    * (coordinate - self.axis_lower[axis])
                    / self.axis_period[axis]
                )
                sine = jax.ops.segment_sum(
                    weights * jnp.sin(angle), segments, capacity + 1
                )[:capacity]
                cosine = jax.ops.segment_sum(
                    weights * jnp.cos(angle), segments, capacity + 1
                )[:capacity]
                mean_angle = jnp.mod(jnp.arctan2(sine, cosine), 2.0 * jnp.pi)
                columns.append(
                    self.axis_lower[axis]
                    + self.axis_period[axis] * mean_angle / (2.0 * jnp.pi)
                )
            else:
                moment = jax.ops.segment_sum(
                    weights * coordinate, segments, capacity + 1
                )[:capacity]
                columns.append(moment / denominator)
        return jnp.where((volume > 0.0)[:, None], jnp.stack(columns, axis=-1), 0.0)

    def label(
        self, alpha: ArrayLike, /, *, color: ArrayLike | None = None
    ) -> BubbleComponentLabels:
        """Label the gas support of ``alpha`` (optionally split by color)."""

        gas = self.gas_content(alpha)
        active = gas >= self.gas_threshold * self.cell_measure
        cell_color = (
            jnp.zeros(self.cell_shape, dtype=jnp.int32)
            if color is None
            else jnp.asarray(color, dtype=jnp.int32)
        )
        if cell_color.shape != self.cell_shape:
            raise ValueError("color must match the two-phase cell shape.")
        cell_color = jnp.where(active, cell_color, -1)
        flat_color = cell_color.reshape(-1)
        relation = self.labeling.relation
        edge_valid = (
            flat_color[relation.source_indices] == flat_color[relation.target_indices]
        )
        labeling = self.labeling.label(active.reshape(-1), edge_valid=edge_valid)
        label = labeling.label.reshape(self.cell_shape)
        capacity = self.component_capacity
        safe = jnp.where(label >= 0, label, capacity).reshape(-1)
        volume = jax.ops.segment_sum(
            jnp.where(label >= 0, gas, 0.0).reshape(-1), safe, capacity + 1
        )[:capacity]
        atmosphere = (
            jax.ops.segment_max(
                (self.vent_mask & (label >= 0)).reshape(-1).astype(jnp.int32),
                safe,
                capacity + 1,
            )[:capacity]
            > 0
        ) & labeling.component_mask
        return BubbleComponentLabels(
            labeling=labeling,
            gas_volume=gas,
            color=cell_color,
            volume=volume,
            centroid=self._centroids(label, gas, volume),
            atmosphere=atmosphere,
        )

    def initial_state(
        self, alpha: ArrayLike, /, *, color: ArrayLike | None = None
    ) -> BubbleComponentState:
        """Identities of the initial field: atmosphere 0, bubbles 1, 2, ..."""

        labels = self.label(alpha, color=color)
        bubble = labels.labeling.component_mask & ~labels.atmosphere
        rank = jnp.cumsum(bubble.astype(jnp.int32))
        slot_ids = jnp.where(
            bubble,
            ATMOSPHERE_ID + rank,
            jnp.where(labels.atmosphere, ATMOSPHERE_ID, -1),
        ).astype(jnp.int32)
        return BubbleComponentState(
            labels=labels,
            slot_ids=slot_ids,
            next_id=(ATMOSPHERE_ID + 1 + jnp.sum(bubble, dtype=jnp.int32)).astype(
                jnp.int32
            ),
            epoch=jnp.asarray(0, dtype=jnp.int32),
        )

    def propose(
        self,
        state: BubbleComponentState,
        alpha: ArrayLike,
        /,
        *,
        color: ArrayLike | None = None,
    ) -> BubbleIdentityProposal:
        """Label ``alpha`` and assign stable identities against ``state``."""

        labels = self.label(alpha, color=color)
        previous = state.labels
        transition_plan = self.transition if color is None else self.marker_transition
        transition = transition_plan.transition(
            previous.labeling, labels.labeling, labels.gas_volume.reshape(-1)
        )
        capacity = self.component_capacity
        pair_new = jnp.where(transition.pair_active, transition.pair_new, capacity)
        parent_slot = jax.ops.segment_max(
            jnp.where(transition.pair_active, transition.pair_old, -1),
            pair_new,
            capacity + 1,
        )[:capacity]
        new_active = labels.labeling.component_mask
        previous_atmosphere = previous.atmosphere
        safe_parent = jnp.clip(parent_slot, 0, capacity - 1)
        continued = (
            new_active
            & ~labels.atmosphere
            & (transition.new_event == ComponentEventKind.CONTINUE)
            & (parent_slot >= 0)
            & ~previous_atmosphere[safe_parent]
        )
        fresh = new_active & ~labels.atmosphere & ~continued
        offset = jnp.cumsum(fresh.astype(jnp.int32)) - fresh.astype(jnp.int32)
        slot_ids = jnp.where(
            continued,
            state.slot_ids[safe_parent],
            jnp.where(
                fresh,
                state.next_id + offset,
                jnp.where(new_active & labels.atmosphere, ATMOSPHERE_ID, -1),
            ),
        ).astype(jnp.int32)
        inherited = (
            jnp.zeros((capacity + 1,), dtype=jnp.int32)
            .at[jnp.where(continued, parent_slot, capacity)]
            .add(1)[:capacity]
        )
        previous_bubble = previous.labeling.component_mask & ~previous_atmosphere
        consumed = previous_bubble & (inherited == 0)
        fresh_count = jnp.sum(fresh, dtype=jnp.int32)
        consumed_count = jnp.sum(consumed, dtype=jnp.int32)
        changed = (fresh_count > 0) | (consumed_count > 0)
        labeling_ok = labels.labeling.successful
        status = jnp.where(
            ~labeling_ok,
            BubbleIdentityStatus.LABELING_FAILED,
            jnp.where(
                ~transition.successful,
                BubbleIdentityStatus.TRANSITION_OVERFLOW,
                jnp.where(
                    changed,
                    BubbleIdentityStatus.TOPOLOGY_EVENT,
                    BubbleIdentityStatus.CONTINUED,
                ),
            ),
        ).astype(jnp.int32)
        evidence = BubbleTopologyEvidence(
            component_count=labels.labeling.count,
            bubble_count=jnp.sum(new_active & ~labels.atmosphere, dtype=jnp.int32),
            atmosphere_count=jnp.sum(labels.atmosphere, dtype=jnp.int32),
            merge_count=transition.merge_count,
            split_count=transition.split_count,
            create_count=transition.create_count,
            vanish_count=transition.vanish_count,
            reconnect_count=transition.reconnect_count,
            fresh_count=fresh_count,
            consumed_count=consumed_count,
            converged=labels.labeling.converged,
            overflow=labels.labeling.overflow,
            pair_overflow=transition.pair_overflow,
            topology_changed=changed,
            derivative_available=~changed,
            status=status,
        )
        return BubbleIdentityProposal(
            labels=labels,
            transition=transition,
            slot_ids=slot_ids,
            fresh=fresh,
            parent_slot=parent_slot,
            consumed=consumed,
            previous_ids=state.slot_ids,
            next_id=(state.next_id + fresh_count).astype(jnp.int32),
            evidence=evidence,
        )


type BubbleTransitionKind = Literal[
    "create", "entrain", "merge", "reconnect", "split", "vanish", "vent"
]


class BubbleTransitionRecord(StrictModule):
    """One journaled bubble topology event (host record, canonical order).

    Parent and child identities are sorted; `ATMOSPHERE_ID` stands for the
    atmosphere. ``overlaps`` lists ``(parent_id, child_id, gas_volume)`` for the
    observed overlap pairs of the event.
    """

    epoch: int = eqx.field(static=True)
    kind: BubbleTransitionKind = eqx.field(static=True)
    parent_ids: tuple[int, ...] = eqx.field(static=True)
    child_ids: tuple[int, ...] = eqx.field(static=True)
    parent_volumes: tuple[float, ...] = eqx.field(static=True)
    child_volumes: tuple[float, ...] = eqx.field(static=True)
    child_slots: tuple[int, ...] = eqx.field(static=True)
    overlaps: tuple[tuple[int, int, float], ...] = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        epoch: int,
        kind: BubbleTransitionKind,
        /,
        *,
        parent_ids: tuple[int, ...],
        child_ids: tuple[int, ...],
        parent_volumes: tuple[float, ...],
        child_volumes: tuple[float, ...],
        child_slots: tuple[int, ...],
        overlaps: tuple[tuple[int, int, float], ...],
    ) -> None:
        if len(parent_ids) != len(parent_volumes):
            raise ValueError("parent_ids and parent_volumes must align.")
        if len(child_ids) != len(child_volumes) or len(child_ids) != len(child_slots):
            raise ValueError("child_ids, child_volumes and child_slots must align.")
        self.epoch = int(epoch)
        self.kind = parse(kind, BubbleTransitionKind, "kind")
        self.parent_ids = tuple(int(value) for value in parent_ids)
        self.child_ids = tuple(int(value) for value in child_ids)
        self.parent_volumes = tuple(float(value) for value in parent_volumes)
        self.child_volumes = tuple(float(value) for value in child_volumes)
        self.child_slots = tuple(int(value) for value in child_slots)
        self.overlaps = tuple(
            (int(parent), int(child), float(volume)) for parent, child, volume in overlaps
        )
        self.record_id = canonical_fingerprint(
            {
                "kind": "bubble-transition-record",
                "epoch": self.epoch,
                "event": self.kind,
                "parents": list(self.parent_ids),
                "children": list(self.child_ids),
                "parent_volumes": list(self.parent_volumes),
                "child_volumes": list(self.child_volumes),
                "overlaps": [list(entry) for entry in self.overlaps],
            }
        )


def _event_kind(parents: list[int], children: list[int], /) -> BubbleTransitionKind:
    if ATMOSPHERE_ID in children:
        return "vent"
    if ATMOSPHERE_ID in parents:
        return "entrain"
    if not parents:
        return "create"
    if not children:
        return "vanish"
    if len(parents) >= 2 and len(children) == 1:
        return "merge"
    if len(parents) == 1 and len(children) >= 2:
        return "split"
    return "reconnect"


def _find(parent: list[int], node: int, /) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def transition_records(
    event: BubbleIdentityEvent,
    previous: BubbleComponentState,
    /,
) -> tuple[BubbleTransitionRecord, ...]:
    """Host journal records of one proposal, in canonical order.

    This is the host epoch boundary: it synchronizes the fixed-capacity device
    proposal once. Overlap pairs that involve a consumed parent, a fresh child
    or an atmosphere change are grouped by bipartite connectivity, and each
    group becomes one record. Atmosphere-to-atmosphere continuation is not an
    event.
    """

    capacity = event.slot_ids.shape[0]
    epoch = int(previous.epoch) + 1
    old_ids = np.asarray(previous.slot_ids)
    old_volume = np.asarray(previous.labels.volume)
    old_atmosphere = np.asarray(previous.labels.atmosphere)
    new_ids = np.asarray(event.slot_ids)
    new_volume = np.asarray(event.volume)
    new_atmosphere = np.asarray(event.atmosphere)
    fresh = np.asarray(event.fresh)
    consumed = np.asarray(event.consumed)
    transition = event.transition
    pair_active = np.asarray(transition.pair_active)
    pair_old = np.asarray(transition.pair_old)[pair_active]
    pair_new = np.asarray(transition.pair_new)[pair_active]
    pair_overlap = np.asarray(transition.pair_overlap)[pair_active]
    # Nodes 0..K-1 are previous slots, K..2K-1 are proposed slots.
    parent = list(range(2 * capacity))
    involved = np.zeros(2 * capacity, dtype=np.bool_)
    involved[:capacity] = consumed
    involved[capacity:] = fresh
    event_pairs = []
    for old, new, overlap in zip(pair_old, pair_new, pair_overlap, strict=True):
        relevant = (
            consumed[old] or fresh[new] or (old_atmosphere[old] != new_atmosphere[new])
        )
        if not relevant:
            continue
        event_pairs.append((int(old), int(new), float(overlap)))
        involved[old] = True
        involved[capacity + new] = True
        root_old = _find(parent, int(old))
        root_new = _find(parent, capacity + int(new))
        if root_old != root_new:
            parent[max(root_old, root_new)] = min(root_old, root_new)
    groups: dict[int, list[int]] = {}
    for node in np.flatnonzero(involved):
        groups.setdefault(_find(parent, int(node)), []).append(int(node))
    records = []
    for members in groups.values():
        old_slots = sorted(node for node in members if node < capacity)
        new_slots = sorted(node - capacity for node in members if node >= capacity)
        parents = [int(old_ids[slot]) for slot in old_slots]
        children = [int(new_ids[slot]) for slot in new_slots]
        overlaps = tuple(
            sorted(
                (int(old_ids[old]), int(new_ids[new]), overlap)
                for old, new, overlap in event_pairs
                if old in old_slots and new in new_slots
            )
        )
        parent_order = np.argsort(np.asarray(parents, dtype=np.int64), kind="stable")
        child_order = np.argsort(np.asarray(children, dtype=np.int64), kind="stable")
        records.append(
            BubbleTransitionRecord(
                epoch,
                _event_kind(parents, children),
                parent_ids=tuple(parents[index] for index in parent_order),
                child_ids=tuple(children[index] for index in child_order),
                parent_volumes=tuple(
                    float(old_volume[old_slots[index]]) for index in parent_order
                ),
                child_volumes=tuple(
                    float(new_volume[new_slots[index]]) for index in child_order
                ),
                child_slots=tuple(new_slots[index] for index in child_order),
                overlaps=overlaps,
            )
        )
    return tuple(
        sorted(records, key=lambda record: (record.child_ids, record.parent_ids))
    )


class BubbleTransitionJournal(StrictModule):
    """Deterministic append-only lineage of committed bubble topology events."""

    records: tuple[BubbleTransitionRecord, ...] = eqx.field(static=True)
    journal_id: str = eqx.field(static=True)

    def __init__(self, records: tuple[BubbleTransitionRecord, ...] = (), /) -> None:
        if not all(isinstance(record, BubbleTransitionRecord) for record in records):
            raise TypeError("records must be BubbleTransitionRecord values.")
        self.records = tuple(records)
        self.journal_id = canonical_fingerprint(
            {
                "kind": "bubble-transition-journal",
                "records": [record.record_id for record in self.records],
            }
        )

    def extend(
        self, records: tuple[BubbleTransitionRecord, ...], /
    ) -> BubbleTransitionJournal:
        return BubbleTransitionJournal(self.records + tuple(records))

    def lineage(self, bubble_id: int, /) -> tuple[int, ...]:
        """Every non-atmosphere ancestor identity of ``bubble_id``, sorted."""

        ancestors: set[int] = set()
        frontier = {int(bubble_id)}
        for record in reversed(self.records):
            if frontier.intersection(record.child_ids):
                parents = set(record.parent_ids) - {ATMOSPHERE_ID}
                ancestors |= parents
                frontier |= parents
        return tuple(sorted(ancestors))


__all__ = [
    "ATMOSPHERE_ID",
    "BubbleComponentLabels",
    "BubbleComponentPlan",
    "BubbleComponentState",
    "BubbleIdentityEvent",
    "BubbleIdentityProposal",
    "BubbleIdentityStatus",
    "BubbleTopologyEvidence",
    "BubbleTransitionJournal",
    "BubbleTransitionKind",
    "BubbleTransitionRecord",
    "transition_records",
]
