#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Multi-marker VOF: bubble identity without numerical coalescence.

The authoritative gas content ``V (1 - alpha)`` of every cell is partitioned
among ``marker_capacity`` colors. Markers are transported by replaying the
geometric face-flux bundle of the VOF step (`TwoPhaseFluxBundle`):

- the gas face flux ``total - liquid`` of every sweep is split among the
  colors by the donor cell's marker shares at the start of that sweep;
- the non-flux gas source of the sweep (the Weymouth–Yue dilation, together
  with any declared compartment dilatation) is split by the cell's own
  shares.

Every color therefore uses the same flux bundle, and the color contents sum
to the authoritative gas content up to rounding, which is reported.

Gas cells connect only to face neighbors of the same dominant color
(`BubbleComponentPlan`). Two nearby bubbles with different colors therefore
keep separate identities even when their mixed cells touch.

`MultiMarkerPlan.proximity` builds a bounded close-interface graph. For every
gas cell it finds the nearest foreign component of the same color and the
nearest one of a different color inside a static stencil radius, and it
groups the observed component pairs with `phydrax.sparse.KeyGroupPlan`.

Recoloring is a host transaction. A same-color conflict is resolved
deterministically: the larger identity takes the smallest color unused by
its close neighbors. The color content moves conservatively, cell by cell.
The transaction refuses rather than aliasing colors when none is free, and
no derivative is claimed across it.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from enum import IntEnum

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
from ...sparse import KeyGroupPlan
from ...typing import checked
from ._bubble_components import ATMOSPHERE_ID, BubbleComponentLabels
from ._flux_bundle import cell_net_flux, face_donor, TwoPhaseFluxBundle
from ._vof import PreparedIncompressibleTwoPhaseVOF


class MarkerRecolorStatus(IntEnum):
    """Outcome of one host recoloring transaction."""

    COMMITTED = 0
    NO_CONFLICT = 1
    MARKER_CAPACITY_EXCEEDED = 2
    PROXIMITY_OVERFLOW = 3


class MultiMarkerState(StrictModule):
    """Gas volume content of every color, shape ``(markers,) + cell_shape``."""

    content: Array


class MarkerTransportResult(StrictModule):
    """Markers after one replayed transport step and their evidence.

    ``sum_residual`` is the maximum over cells of ``|sum_m content_m - gas|``
    divided by the cell measure. ``minimum_content`` is the most negative
    color content divided by its cell measure.
    """

    state: MultiMarkerState
    sum_residual: Array
    minimum_content: Array
    finite: Array
    successful: Array


class MarkerProximity(StrictModule):
    """Bounded close-interface graph of one component labeling.

    Pair arrays (``pair_capacity``) list the observed component-slot pairs
    ``first < second`` in canonical key order. ``distance`` is the smallest
    center-to-center distance between their gas cells (inside the stencil).
    ``same_color`` marks a pair of equal color, which is a recoloring
    conflict. The per-cell fields hold the nearest foreign component of a
    different color and its distance. They drive the near-contact potential.
    """

    pair_first: Array
    pair_second: Array
    pair_active: Array
    distance: Array
    same_color: Array
    cell_foreign_slot: Array
    cell_foreign_distance: Array
    cell_pair_slot: Array
    pair_overflow: Array


class MarkerRecolorResult(StrictModule):
    """Result of one recoloring transaction (host epoch transition)."""

    state: MultiMarkerState
    moved_content: Array
    status: MarkerRecolorStatus = eqx.field(static=True)
    recolored: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    transaction_id: str = eqx.field(static=True)

    @property
    def committed(self) -> bool:
        return self.status is MarkerRecolorStatus.COMMITTED


def _stencil(
    dimension: int, radius: int, spacing: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    offsets = [
        offset
        for offset in itertools.product(range(-radius, radius + 1), repeat=dimension)
        if any(offset) and sum(value * value for value in offset) <= radius * radius
    ]
    array = np.asarray(offsets, dtype=np.int64)
    lengths = np.sqrt(np.sum((array * spacing[None, :]) ** 2, axis=1))
    order = np.lexsort(tuple(array.T[::-1]) + (lengths,))
    return array[order], lengths[order]


def _resolve_colors(
    ids: list[int],
    colors: list[int],
    pairs: list[tuple[int, int]],
    marker_capacity: int,
    /,
    *,
    merge_pairs: tuple[tuple[int, int], ...],
    exempt_pairs: tuple[tuple[int, int], ...],
) -> list[int] | None:
    """Deterministic greedy coloring of the close-interface graph (host).

    Merge pairs pin both slots and give the absorbed slot the color of the
    kept one. Conflicts are visited in order of the larger identity. The
    unpinned, non-atmosphere slot with the larger identity takes the smallest
    color that none of its close neighbors uses. ``None`` means the color
    capacity is exhausted.
    """

    slot_of = {bubble: slot for slot, bubble in enumerate(ids) if bubble >= 0}
    neighbours: dict[int, set[int]] = {slot: set() for slot in range(len(ids))}
    for first, second in pairs:
        neighbours[first].add(second)
        neighbours[second].add(first)
    exempt = {tuple(sorted(pair)) for pair in exempt_pairs + merge_pairs}
    target = list(colors)
    pinned: set[int] = set()
    for keep, absorb in sorted(merge_pairs):
        if keep in slot_of and absorb in slot_of and absorb != ATMOSPHERE_ID:
            target[slot_of[absorb]] = target[slot_of[keep]]
            pinned |= {slot_of[keep], slot_of[absorb]}
    ordered = sorted(pairs, key=lambda pair: (max(ids[pair[0]], ids[pair[1]]), pair))
    for first, second in ordered:
        if target[first] != target[second]:
            continue
        if tuple(sorted((ids[first], ids[second]))) in exempt:
            continue
        candidates = [
            slot
            for slot in (first, second)
            if slot not in pinned and ids[slot] != ATMOSPHERE_ID
        ]
        if not candidates:
            return None
        slot = max(candidates, key=lambda value: ids[value])
        used = {target[other] for other in neighbours[slot]} | {target[slot]}
        free = [color for color in range(marker_capacity) if color not in used]
        if not free:
            return None
        target[slot] = free[0]
    return target


class MultiMarkerPlan(StrictModule, NonTrainableState):
    """Static color capacity, flux replay and close-interface graph."""

    cell_measure: Array
    offsets: Array
    offset_lengths: Array
    cell_index: Array
    pair_groups: KeyGroupPlan
    marker_capacity: int = eqx.field(static=True)
    component_capacity: int = eqx.field(static=True)
    proximity_radius: int = eqx.field(static=True)
    contact_distance: float = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    cell_shape: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        two_phase: PreparedIncompressibleTwoPhaseVOF,
        /,
        *,
        marker_capacity: int,
        component_capacity: int,
        pair_capacity: int,
        proximity_radius: int = 3,
    ) -> None:
        markers = positive_integer(marker_capacity, "marker_capacity")
        if markers < 2:
            raise ValueError("marker_capacity must be at least 2.")
        components = positive_integer(component_capacity, "component_capacity")
        pairs = positive_integer(pair_capacity, "pair_capacity")
        radius = positive_integer(proximity_radius, "proximity_radius")
        discretization = two_phase.plan.discretization
        cell_shape = tuple(discretization.cell_shape)
        axes = discretization.grid.structured_axes
        spacing = []
        for axis in axes:
            widths = np.asarray(axis.interval_widths, dtype=np.float64)
            if not np.allclose(widths, widths[0], rtol=1.0e-10, atol=0.0):
                raise ValueError("Multi-marker proximity requires uniform spacing.")
            spacing.append(float(widths[0]))
        spacing_ = np.asarray(spacing, dtype=np.float64)
        offsets, lengths = _stencil(len(cell_shape), radius, spacing_)
        entity_count = int(np.prod(cell_shape))
        dtype = discretization.cell_volumes.dtype
        self.cell_measure = jnp.asarray(two_phase.cell_fluid_measure, dtype=dtype)
        self.offsets = jnp.asarray(offsets, dtype=jnp.int32)
        self.offset_lengths = jnp.asarray(lengths, dtype=dtype)
        self.cell_index = jnp.stack(
            jnp.meshgrid(
                *(jnp.arange(count, dtype=jnp.int32) for count in cell_shape),
                indexing="ij",
            ),
            axis=-1,
        )
        self.pair_groups = KeyGroupPlan(
            2 * entity_count, pairs, components * components - 1
        )
        self.marker_capacity = markers
        self.component_capacity = components
        self.proximity_radius = radius
        self.contact_distance = float(np.max(spacing_))
        self.periodic = tuple(axis.periodic for axis in axes)
        self.cell_shape = cell_shape
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multi-marker-plan",
                "two_phase": two_phase.prepared_id,
                "marker_capacity": markers,
                "component_capacity": components,
                "pair_capacity": pairs,
                "proximity_radius": radius,
            }
        )

    def initial_state(self, alpha: ArrayLike, color: ArrayLike, /) -> MultiMarkerState:
        """Assign every cell's gas content to its declared color."""

        alpha_ = jnp.asarray(alpha, dtype=self.cell_measure.dtype)
        color_ = jnp.asarray(color, dtype=jnp.int32)
        if alpha_.shape != self.cell_shape or color_.shape != self.cell_shape:
            raise ValueError("alpha and color must match the cell shape.")
        if bool(jnp.any((color_ < 0) | (color_ >= self.marker_capacity))):
            raise ValueError("Initial colors must lie in [0, marker_capacity).")
        gas = self.cell_measure * (1.0 - alpha_)
        onehot = jax.nn.one_hot(color_, self.marker_capacity, dtype=gas.dtype)
        return MultiMarkerState(jnp.moveaxis(onehot, -1, 0) * gas[None])

    def gas_content(self, state: MultiMarkerState, /) -> Array:
        return jnp.sum(state.content, axis=0)

    def shares(self, state: MultiMarkerState, /) -> Array:
        """Color fractions of every cell's gas content (zero without gas)."""

        total = self.gas_content(state)
        positive = total > 0.0
        return jnp.where(
            positive[None],
            jnp.maximum(state.content, 0.0) / jnp.where(positive, total, 1.0)[None],
            0.0,
        )

    def color(self, state: MultiMarkerState, /) -> Array:
        """Dominant color of every cell (lowest color on ties)."""

        return jnp.argmax(state.content, axis=0).astype(jnp.int32)

    def transport(
        self, state: MultiMarkerState, bundle: TwoPhaseFluxBundle, /
    ) -> MarkerTransportResult:
        """Replay the step's sweeps on every color (see module docstring)."""

        periodic = self.periodic
        dimension = len(self.cell_shape)
        dt = bundle.step_size
        contents = bundle.sweep_contents(periodic)

        def sweep(markers: Array, axis: int, liquid_before: Array) -> Array:
            gas_before = self.cell_measure - liquid_before
            positive = gas_before > 0.0
            shares = jnp.where(
                positive[None],
                markers / jnp.where(positive, gas_before, 1.0)[None],
                0.0,
            )
            gas_flux = bundle.total_rates[axis] - bundle.liquid_rates[axis]
            net_total = cell_net_flux(bundle.total_rates[axis], axis, periodic[axis])
            source = net_total - bundle.dilation_rates[axis]

            def one(color_shares: Array) -> Array:
                donor = face_donor(color_shares, gas_flux, axis, periodic[axis])
                return cell_net_flux(gas_flux * donor, axis, periodic[axis])

            net = jax.vmap(one)(shares)
            return markers - dt * net + dt * source[None] * shares

        def ordered(offset: int, /) -> Callable[[Array], Array]:
            def run(markers: Array) -> Array:
                for position in range(dimension):
                    axis = (offset + position) % dimension
                    markers = sweep(markers, axis, contents[position])
                return markers

            return run

        content = jax.lax.switch(
            jnp.asarray(bundle.sweep_offset, dtype=jnp.int32) % dimension,
            [ordered(offset) for offset in range(dimension)],
            state.content,
        )
        gas_after = self.cell_measure - contents[-1]
        measure = jnp.where(self.cell_measure > 0.0, self.cell_measure, 1.0)
        residual = jnp.max(jnp.abs(jnp.sum(content, axis=0) - gas_after) / measure)
        minimum = jnp.min(content / measure[None])
        finite = jnp.all(jnp.isfinite(content))
        tolerance = 4096.0 * jnp.finfo(content.dtype).eps
        return MarkerTransportResult(
            state=MultiMarkerState(content),
            sum_residual=residual,
            minimum_content=minimum,
            finite=finite,
            successful=finite & (residual <= tolerance) & (minimum >= -tolerance),
        )

    def _shift(self, values: Array, offset: Array, fill: Array, /) -> Array:
        shifted = values
        valid = jnp.ones(self.cell_shape, dtype=jnp.bool_)
        for axis, periodic in enumerate(self.periodic):
            shifted = jnp.roll(shifted, -offset[axis], axis=axis)
            if not periodic:
                target = self.cell_index[..., axis] + offset[axis]
                valid = valid & (target >= 0) & (target < self.cell_shape[axis])
        return jnp.where(valid, shifted, fill)

    def proximity(self, labels: BubbleComponentLabels, /) -> MarkerProximity:
        """Close-interface graph of one per-color component labeling."""

        label = labels.labeling.label.reshape(self.cell_shape)
        color = labels.color
        component_color = self._component_color(labels)
        capacity = self.component_capacity
        infinite = jnp.asarray(jnp.inf, dtype=self.offset_lengths.dtype)
        none = jnp.asarray(-1, dtype=jnp.int32)

        def visit(
            carry: tuple[Array, Array, Array, Array], item: tuple[Array, Array]
        ) -> tuple[tuple[Array, Array, Array, Array], None]:
            same_distance, same_slot, other_distance, other_slot = carry
            offset, length = item
            neighbour = self._shift(label, offset, none)
            foreign = (label >= 0) & (neighbour >= 0) & (neighbour != label)
            safe = jnp.clip(neighbour, 0, capacity - 1)
            equal = foreign & (component_color[safe] == color)
            different = foreign & ~equal
            closer_same = equal & (length < same_distance)
            closer_other = different & (length < other_distance)
            return (
                jnp.where(closer_same, length, same_distance),
                jnp.where(closer_same, neighbour, same_slot),
                jnp.where(closer_other, length, other_distance),
                jnp.where(closer_other, neighbour, other_slot),
            ), None

        start = (
            jnp.full(self.cell_shape, infinite),
            jnp.full(self.cell_shape, -1, dtype=jnp.int32),
            jnp.full(self.cell_shape, infinite),
            jnp.full(self.cell_shape, -1, dtype=jnp.int32),
        )
        (same_distance, same_slot, other_distance, other_slot), _ = jax.lax.scan(
            visit, start, (self.offsets, self.offset_lengths)
        )
        own = label.reshape(-1)
        partners = jnp.concatenate((same_slot.reshape(-1), other_slot.reshape(-1)))
        owners = jnp.concatenate((own, own))
        distances = jnp.concatenate(
            (same_distance.reshape(-1), other_distance.reshape(-1))
        )
        valid = (owners >= 0) & (partners >= 0)
        first = jnp.minimum(owners, partners)
        second = jnp.maximum(owners, partners)
        keys = jnp.where(valid, first * capacity + second, 0).astype(jnp.int32)
        grouped = self.pair_groups.build(keys, valid)
        slots = jnp.where(
            valid, grouped.item_group_slots, self.pair_groups.group_capacity
        )
        pair_distance = jax.ops.segment_min(
            jnp.where(valid, distances, infinite),
            slots,
            self.pair_groups.group_capacity + 1,
        )[: self.pair_groups.group_capacity]
        group_keys = jnp.where(grouped.group_active, grouped.group_keys, 0)
        pair_first = jnp.where(grouped.group_active, group_keys // capacity, -1)
        pair_second = jnp.where(grouped.group_active, group_keys % capacity, -1)
        safe_first = jnp.clip(pair_first, 0, capacity - 1)
        safe_second = jnp.clip(pair_second, 0, capacity - 1)
        return MarkerProximity(
            pair_first=pair_first.astype(jnp.int32),
            pair_second=pair_second.astype(jnp.int32),
            pair_active=grouped.group_active,
            distance=jnp.where(grouped.group_active, pair_distance, infinite),
            same_color=grouped.group_active
            & (component_color[safe_first] == component_color[safe_second]),
            cell_foreign_slot=other_slot,
            cell_foreign_distance=other_distance,
            cell_pair_slot=grouped.item_group_slots[own.shape[0] :].reshape(
                self.cell_shape
            ),
            pair_overflow=grouped.evidence.group_overflow,
        )

    def _component_color(self, labels: BubbleComponentLabels, /) -> Array:
        capacity = self.component_capacity
        label = labels.labeling.label
        safe = jnp.where(label >= 0, label, capacity)
        return jax.ops.segment_max(
            jnp.where(label >= 0, labels.color.reshape(-1), -1),
            safe,
            capacity + 1,
        )[:capacity]

    def recolor(
        self,
        state: MultiMarkerState,
        labels: BubbleComponentLabels,
        slot_ids: ArrayLike,
        proximity: MarkerProximity,
        /,
        *,
        merge_pairs: tuple[tuple[int, int], ...] = (),
        exempt_pairs: tuple[tuple[int, int], ...] = (),
    ) -> MarkerRecolorResult:
        """Resolve same-color conflicts and coalescence merges (host).

        ``merge_pairs`` lists identity pairs ``(keep, absorb)`` whose film
        ruptured. ``absorb`` takes the color of ``keep`` and the pair becomes
        exempt from conflict resolution. ``exempt_pairs`` are identity pairs
        that were already merged physically and are still separate
        components. The atmosphere is never recolored.
        """

        ids = [int(value) for value in np.asarray(slot_ids)]
        colors = [int(value) for value in np.asarray(self._component_color(labels))]
        active = np.asarray(proximity.pair_active)
        if bool(np.asarray(proximity.pair_overflow)):
            return self._refuse(state, MarkerRecolorStatus.PROXIMITY_OVERFLOW)
        pairs = list(
            zip(
                np.asarray(proximity.pair_first)[active].tolist(),
                np.asarray(proximity.pair_second)[active].tolist(),
                strict=True,
            )
        )
        target = _resolve_colors(
            ids,
            colors,
            pairs,
            self.marker_capacity,
            merge_pairs=merge_pairs,
            exempt_pairs=exempt_pairs,
        )
        if target is None:
            return self._refuse(state, MarkerRecolorStatus.MARKER_CAPACITY_EXCEEDED)
        moves = [
            (ids[slot], before, after)
            for slot, (before, after) in enumerate(zip(colors, target, strict=True))
            if before != after and ids[slot] >= 0
        ]
        if not moves:
            return self._refuse(state, MarkerRecolorStatus.NO_CONFLICT)
        content, moved = self._move(state, labels, colors, target)
        recolored = tuple(sorted(moves))
        return MarkerRecolorResult(
            state=MultiMarkerState(content),
            moved_content=moved,
            status=MarkerRecolorStatus.COMMITTED,
            recolored=recolored,
            transaction_id=canonical_fingerprint(
                {
                    "kind": "multi-marker-recolor",
                    "plan": self.plan_id,
                    "moves": [list(move) for move in recolored],
                }
            ),
        )

    def _move(
        self,
        state: MultiMarkerState,
        labels: BubbleComponentLabels,
        colors: list[int],
        target: list[int],
        /,
    ) -> tuple[Array, Array]:
        label = labels.labeling.label.reshape(self.cell_shape)
        capacity = self.component_capacity
        source = jnp.asarray(colors, dtype=jnp.int32)
        destination = jnp.asarray(target, dtype=jnp.int32)
        safe = jnp.clip(label, 0, capacity - 1)
        moving = (label >= 0) & (source[safe] != destination[safe])
        from_color = jnp.where(moving, source[safe], 0)
        to_color = jnp.where(moving, destination[safe], 0)
        amount = jnp.where(
            moving,
            jnp.take_along_axis(state.content, from_color[None], axis=0)[0],
            0.0,
        )
        markers = jnp.arange(self.marker_capacity, dtype=jnp.int32).reshape(
            (-1,) + (1,) * len(self.cell_shape)
        )
        content = (
            state.content
            - jnp.where(markers == from_color[None], amount[None], 0.0)
            + jnp.where(markers == to_color[None], amount[None], 0.0)
        )
        return content, jnp.sum(amount)

    def _refuse(
        self, state: MultiMarkerState, status: MarkerRecolorStatus, /
    ) -> MarkerRecolorResult:
        return MarkerRecolorResult(
            state=state,
            moved_content=jnp.zeros((), dtype=state.content.dtype),
            status=status,
            recolored=(),
            transaction_id=canonical_fingerprint(
                {
                    "kind": "multi-marker-recolor",
                    "plan": self.plan_id,
                    "outcome": status.name.lower(),
                }
            ),
        )


__all__ = [
    "MarkerProximity",
    "MarkerRecolorResult",
    "MarkerRecolorStatus",
    "MarkerTransportResult",
    "MultiMarkerPlan",
    "MultiMarkerState",
]
