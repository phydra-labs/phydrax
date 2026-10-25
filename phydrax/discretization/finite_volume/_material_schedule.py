#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded chronological slot schedule of a volume material chain.

Each live cell is a birth-occurrence token: the source mesh's cells, or the
cells an edit creates. A token is written once and read at most once, so the
only dependency between edits is read-after-write on tokens. An edit's level
is one more than the highest level that created a token it reads; edits on one
level read and write disjoint tokens, so executing level by level reproduces
the chronological operator exactly. Levels are packed greedily into chunks of
fixed read, write and weight-lane capacity; a chunk gathers every slot it
reads before it overwrites any slot, so freed slots are reused within the
chunk, then from a deterministic free list, then fresh.
"""

from __future__ import annotations

import equinox as eqx
import jax
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry._volume_material_map import (
    volume_material_edit_weights,
    volume_material_target_measures,
    VolumeMaterialChain,
)
from ...sparse import InjectiveRouteWindows
from .._coordinate_budget import CoordinateEnclosureBudget


class MaterialChainSchedule(StrictModule, NonTrainableState):
    """Token-slot chunks of one chain between two FV endpoints.

    Chunk ``c`` gathers the slots ``read_windows[c]``, maps their densities
    through its weight lanes (``pair_sources``/``pair_targets`` local read and
    write lanes, global weight rows ``weight_windows[c]`` and their exact
    exponents ``pair_exponents``) and overwrites ``write_windows[c]``. Source
    row ``k`` starts in slot ``k``; target row ``t`` ends in
    ``final_windows[0, t]``. Invalid lanes hold each window's target size.
    `prepare_material_chain_schedule` guarantees what the action relies on:
    each live token is read by at most one chunk and never written again, and
    each weight row occupies exactly one lane of one chunk.
    """

    read_windows: InjectiveRouteWindows
    write_windows: InjectiveRouteWindows
    final_windows: InjectiveRouteWindows
    weight_windows: InjectiveRouteWindows
    pair_sources: Array
    pair_targets: Array
    pair_exponents: Array
    source_count: int = eqx.field(static=True)
    target_count: int = eqx.field(static=True)

    def __init__(
        self,
        read_windows: InjectiveRouteWindows,
        write_windows: InjectiveRouteWindows,
        final_windows: InjectiveRouteWindows,
        weight_windows: InjectiveRouteWindows,
        pair_sources: Array,
        pair_targets: Array,
        pair_exponents: Array,
        /,
        *,
        source_count: int,
        target_count: int,
    ) -> None:
        self.read_windows = read_windows
        self.write_windows = write_windows
        self.final_windows = final_windows
        self.weight_windows = weight_windows
        self.pair_sources = pair_sources
        self.pair_targets = pair_targets
        self.pair_exponents = pair_exponents
        self.source_count = int(source_count)
        self.target_count = int(target_count)

    @staticmethod
    def _padded(
        schedule: MaterialChainSchedule,
        /,
        *,
        read: Array,
        write: Array,
        weights: Array,
        sources: Array,
        targets: Array,
        exponents: Array,
    ) -> MaterialChainSchedule:
        """The same schedule with inert trailing chunks (every lane invalid)."""
        return eqx.tree_at(
            lambda value: (
                value.read_windows.target_indices,
                value.write_windows.target_indices,
                value.weight_windows.target_indices,
                value.pair_sources,
                value.pair_targets,
                value.pair_exponents,
            ),
            schedule,
            (read, write, weights, sources, targets, exponents),
        )

    @property
    def chunk_count(self) -> int:
        return self.pair_sources.shape[0]

    @property
    def slot_count(self) -> int:
        return self.read_windows.target_size

    @property
    def read_capacity(self) -> int:
        return self.read_windows.capacity

    @property
    def write_capacity(self) -> int:
        return self.write_windows.capacity

    @property
    def pair_capacity(self) -> int:
        return self.pair_sources.shape[1]


class MaterialChainBounds(StrictModule):
    """Certified relative error bounds of the two nonnegative report actions.

    ``content`` bounds ``|computed - exact| / computed`` of the forward action
    on unit densities times the bound target measures (each target's carried
    content); ``coverage`` bounds the transpose action on those target measures
    (each source cell's covered volume). ``absolute`` is a representable
    allowance added once for a subnormal result.
    """

    content: np.ndarray
    coverage: np.ndarray
    absolute: float = eqx.field(static=True)


def _next_power(value: int, /) -> int:
    return 1 << max(int(value) - 1, 0).bit_length()


def _one_plus(value: np.ndarray | float, /) -> np.ndarray:
    """Upward ``1 + value`` for ``value >= 0``."""
    return np.nextafter(1.0 + np.asarray(value, dtype=np.float64), np.inf)


def _times(left: np.ndarray, right: np.ndarray | float, /) -> np.ndarray:
    """Upward product of nonnegative factors."""
    return np.nextafter(left * right, np.inf)


def _relative(numerator: np.ndarray, denominator: np.ndarray, /) -> np.ndarray:
    """Upward ``numerator / (denominator - numerator)`` for ``0 <= numerator < denominator``."""
    gap = np.nextafter(denominator - numerator, -np.inf)
    return np.nextafter(numerator / gap, np.inf)


def _gamma(counts: np.ndarray, unit: float, /) -> np.ndarray:
    """Upward Higham ``gamma_n = n u / (1 - n u)``."""
    scaled = counts.astype(np.float64) * unit
    return np.nextafter(scaled / np.nextafter(1.0 - scaled, -np.inf), np.inf)


def _step_bounds(
    inherited: np.ndarray,
    weights: np.ndarray,
    counts: np.ndarray,
    unit: float,
    tiny: float,
    /,
) -> np.ndarray:
    """One fused step: product rounding, weight and input bounds, sum, dropped lanes.

    ``1 + delta = (1 + u)(1 + gamma_{n-1})(1 + q)(1 + R)(1 + 2 n tiny)``; the
    relative bound on the computed value is ``delta / (1 - delta)``.
    """
    factor = _times(_one_plus(unit), _one_plus(_gamma(np.maximum(counts - 1, 0), unit)))
    factor = _times(factor, _one_plus(weights))
    factor = _times(factor, _one_plus(inherited))
    factor = _times(factor, _one_plus(_times(2.0 * counts.astype(np.float64), tiny)))
    delta = np.nextafter(factor - 1.0, np.inf)
    if np.any(~np.isfinite(delta)) or np.any(delta >= 1.0):
        raise ValueError(
            "remap.material_map content error bounds must meet the coverage tolerance; "
            "got insufficient certified accuracy"
        )
    return _relative(delta, np.ones_like(delta))


def prepare_material_chain_schedule(
    chain: VolumeMaterialChain,
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    budget: CoordinateEnclosureBudget,
    report: np.dtype,
    /,
) -> tuple[MaterialChainSchedule, MaterialChainBounds]:
    """Tokens, levels, chunks, slots and the certified report bounds of ``chain``.

    ``report`` is the floating dtype the report actions actually execute in;
    the bounds are derived for it. Every pass is charged to ``budget`` before
    its storage is allocated.
    """
    edits = chain.edits
    source_count, target_count = int(source_ids.size), int(target_ids.size)
    weight_count = int(chain.weight_mantissas.shape[0])
    reads_total = sum(int(edit.source_cell_ids.shape[0]) for edit in edits)
    writes_total = sum(int(edit.target_cell_ids.shape[0]) for edit in edits)
    # Host staging, admitted from shapes before any bank is converted: the
    # weight banks (int64 offsets, two int32 lanes, two float64 and one
    # int32), target measures and bounds, the edits' int64 IDs and the int64
    # pair counts, with one header for each of those nine banks.
    budget.reserve(
        6 * weight_count + 2 * target_count + 2 * len(edits) + reads_total + writes_total,
        9 * 256
        + 16 * (len(edits) + 1)
        + 28 * weight_count
        + 16 * target_count
        + 8 * (reads_total + writes_total)
        + 256 * 2 * len(edits),
    )
    offsets, local_targets, local_sources, mantissas, exponents, errors = (
        volume_material_edit_weights(chain)
    )
    measures, measure_errors = volume_material_target_measures(chain)
    removed = [np.asarray(edit.source_cell_ids, dtype=np.int64) for edit in edits]
    created = [np.asarray(edit.target_cell_ids, dtype=np.int64) for edit in edits]
    pairs = np.diff(offsets)
    # Endpoint binding, tokens and levels: one dict slot per live token.
    budget.reserve(
        source_count + target_count + 3 * (reads_total + writes_total),
        256 + 256 * (source_count + writes_total) + 32 * (len(edits) + reads_total + writes_total),
    )
    live = {int(identity): row for row, identity in enumerate(source_ids.tolist())}
    if len(live) != source_count:
        raise ValueError("remap.material_map source cell IDs must be unique")
    token_level = [0] * source_count
    edit_reads: list[list[int]] = []
    edit_writes: list[list[int]] = []
    levels = []
    for removed_ids, created_ids in zip(removed, created, strict=True):
        tokens = []
        for identity in removed_ids.tolist():
            token = live.pop(identity, None)
            if token is None:
                raise ValueError(
                    "remap.material_map edits must remove live cells of the source endpoint"
                )
            tokens.append(token)
        level = 1 + max((token_level[token] for token in tokens), default=0)
        made = []
        for identity in created_ids.tolist():
            if identity in live:
                raise ValueError("remap.material_map edits must create absent cell IDs")
            live[identity] = len(token_level)
            made.append(len(token_level))
            token_level.append(level)
        edit_reads.append(tokens)
        edit_writes.append(made)
        levels.append(level)
    final_tokens = []
    for identity in target_ids.tolist():
        token = live.get(int(identity))
        if token is None:
            raise ValueError("remap.material_map target cells must descend from the source")
        final_tokens.append(token)
    if len(live) != target_count:
        raise ValueError("remap.material_map chain must end in exactly the target cells")
    order = sorted(range(len(edits)), key=lambda index: (levels[index], index))
    reads = np.asarray([value.size for value in removed], dtype=np.int64)
    writes = np.asarray([value.size for value in created], dtype=np.int64)
    # Capacity classes: each packs every edit once.
    classes = []
    smallest = _next_power(max(int(pairs.max(initial=1)), 1))
    largest = _next_power(max(int(pairs.sum()), 1))
    width = smallest
    while width <= largest:
        classes.append(width)
        width *= 2
    budget.reserve(len(classes) * len(edits) + len(edits) * max(len(edits), 1).bit_length())

    def pack(capacity: int) -> list[list[int]]:
        chunks: list[list[int]] = []
        current: list[int] = []
        used = 0
        level = None
        for index in order:
            if current and (levels[index] != level or used + pairs[index] > capacity):
                chunks.append(current)
                current, used = [], 0
            current.append(index)
            used += int(pairs[index])
            level = levels[index]
        if current:
            chunks.append(current)
        return chunks

    best = None
    for capacity in classes:
        chunks = pack(capacity)
        read_capacity = max(int(reads[chunk].sum()) for chunk in chunks)
        write_capacity = max(int(writes[chunk].sum()) for chunk in chunks)
        cost = len(chunks) * (2 * read_capacity + 3 * capacity + 2 * write_capacity)
        if best is None or cost < best[0]:
            best = (cost, capacity, chunks, max(read_capacity, 1), max(write_capacity, 1))
    assert best is not None
    _, pair_capacity, chunks, read_capacity, write_capacity = best
    count = len(chunks)
    # Window and lane banks, their host build and the one-time injectivity
    # validation sorts.
    budget.reserve(
        count * (2 * read_capacity + 2 * write_capacity + 5 * pair_capacity)
        + count
        * (
            read_capacity * read_capacity.bit_length()
            + write_capacity * write_capacity.bit_length()
            + pair_capacity * pair_capacity.bit_length()
        )
        + target_count * max(target_count, 1).bit_length(),
        4 * 256
        + count * (16 * read_capacity + 16 * write_capacity + 33 * pair_capacity)
        + 16 * target_count
        + 64 * (reads_total + writes_total),
    )
    slot_of = np.empty((len(token_level),), dtype=np.int64)
    slot_of[:source_count] = np.arange(source_count)
    free: list[int] = []
    slots = source_count
    read_rows = np.empty((count, read_capacity), dtype=np.int64)
    write_rows = np.empty((count, write_capacity), dtype=np.int64)
    weight_rows = np.full((count, pair_capacity), weight_count, dtype=np.int64)
    source_lanes = np.zeros((count, pair_capacity), dtype=np.int32)
    target_lanes = np.zeros((count, pair_capacity), dtype=np.int32)
    exponent_lanes = np.zeros((count, pair_capacity), dtype=np.int32)
    read_lists, write_lists = [], []
    for position, chunk in enumerate(chunks):
        gathered = [slot_of[token] for index in chunk for token in edit_reads[index]]
        freed = sorted(int(slot) for slot in gathered)
        written = []
        reuse = 0
        for index in chunk:
            for token in edit_writes[index]:
                if reuse < len(freed):
                    slot = freed[reuse]
                    reuse += 1
                elif free:
                    slot = free.pop()
                else:
                    slot = slots
                    slots += 1
                slot_of[token] = slot
                written.append(slot)
        free.extend(freed[reuse:])
        read_lists.append(gathered)
        write_lists.append(written)
        lane = 0
        read_offset = 0
        write_offset = 0
        for index in chunk:
            start, stop = int(offsets[index]), int(offsets[index + 1])
            span = stop - start
            weight_rows[position, lane : lane + span] = np.arange(start, stop)
            source_lanes[position, lane : lane + span] = read_offset + local_sources[start:stop]
            target_lanes[position, lane : lane + span] = write_offset + local_targets[start:stop]
            exponent_lanes[position, lane : lane + span] = exponents[start:stop]
            lane += span
            read_offset += reads[index]
            write_offset += writes[index]
    for position in range(count):
        read_rows[position] = slots
        write_rows[position] = slots
        read_rows[position, : len(read_lists[position])] = read_lists[position]
        write_rows[position, : len(write_lists[position])] = write_lists[position]
    final_slots = slot_of[np.asarray(final_tokens, dtype=np.int64)].reshape((1, -1))
    bounds = _report_bounds(
        read_rows,
        write_rows,
        final_slots[0],
        weight_rows,
        source_lanes,
        target_lanes,
        mantissas,
        errors,
        measures,
        measure_errors,
        slots,
        source_count,
        report,
        budget,
    )
    read_windows = InjectiveRouteWindows(
        read_rows, target_size=slots, valid=read_rows != slots
    )
    write_windows = InjectiveRouteWindows(
        write_rows, target_size=slots, valid=write_rows != slots
    )
    final_windows = InjectiveRouteWindows(final_slots, target_size=slots)
    weight_windows = InjectiveRouteWindows(
        weight_rows, target_size=weight_count, valid=weight_rows != weight_count
    )
    schedule = MaterialChainSchedule(
        read_windows,
        write_windows,
        final_windows,
        weight_windows,
        jax.device_put(source_lanes),
        jax.device_put(target_lanes),
        jax.device_put(exponent_lanes),
        source_count=source_count,
        target_count=target_count,
    )
    return schedule, bounds


def _stored_bounds(
    values: np.ndarray, errors: np.ndarray, report: np.dtype, /
) -> np.ndarray:
    """Relative bounds of binary64 banks once cast to the report dtype.

    The cast's error is measured exactly on the host (both values are binary64
    numbers; their difference is enclosed upward), so an overflowing or
    subnormal cast yields an infinite or correspondingly large bound rather
    than an assumed half-unit rounding.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        cast = values.astype(report).astype(np.float64)
        drift = np.where(
            np.isfinite(cast), np.nextafter(np.abs(cast - values), np.inf), np.inf
        )
    numerator = np.nextafter(errors + drift, np.inf)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        relative = _relative(numerator, values)
    # A cast that loses the whole value certifies nothing.
    return np.where(numerator < values, relative, np.inf)


def _report_bounds(
    read_rows: np.ndarray,
    write_rows: np.ndarray,
    final_slots: np.ndarray,
    weight_rows: np.ndarray,
    source_lanes: np.ndarray,
    target_lanes: np.ndarray,
    mantissas: np.ndarray,
    errors: np.ndarray,
    measures: np.ndarray,
    measure_errors: np.ndarray,
    slots: int,
    source_count: int,
    report: np.dtype,
    budget: CoordinateEnclosureBudget,
    /,
) -> MaterialChainBounds:
    """Static certified relative bounds of both nonnegative report actions.

    They depend only on structure (lane counts, weight and measure rounding)
    and the actual report dtype, never on values: unit roundoff, least normal
    and least subnormal come from ``report``, and the weight and measure banks
    carry their exact casts to it. Every bound operation rounds upward.
    """
    count, pair_capacity = weight_rows.shape
    write_capacity = write_rows.shape[1]
    read_capacity = read_rows.shape[1]
    weight_count = mantissas.size
    # Two passes over every lane and local cell, eight bound operations each,
    # and the two casts of the weight and measure banks.
    budget.reserve(
        2 * count * (8 * pair_capacity + 16 * (read_capacity + write_capacity))
        + 6 * (weight_count + measures.size),
        4 * 256
        + 16 * slots
        + 40 * (weight_count + measures.size)
        + 64 * (pair_capacity + read_capacity + write_capacity),
    )
    info = np.finfo(report)
    unit = float(info.eps) / 2
    tiny = float(info.tiny)
    weight_bounds = _stored_bounds(mantissas, errors, report)
    measure_bounds = _stored_bounds(measures, measure_errors, report)
    forward = np.zeros((slots,), dtype=np.float64)
    for position in range(count):
        valid = weight_rows[position] < weight_count
        lanes_q = weight_bounds[weight_rows[position, valid]]
        reads = read_rows[position][source_lanes[position, valid]]
        inherited = forward[reads] / np.nextafter(1.0 - forward[reads], -np.inf)
        destinations = target_lanes[position, valid]
        counts = np.bincount(destinations, minlength=write_capacity)
        q = np.zeros((write_capacity,), dtype=np.float64)
        r = np.zeros((write_capacity,), dtype=np.float64)
        np.maximum.at(q, destinations, lanes_q)
        np.maximum.at(r, destinations, np.nextafter(inherited, np.inf))
        step = _step_bounds(r, q, counts, unit, tiny)
        written = write_rows[position] < slots
        forward[write_rows[position, written]] = step[written]
    content = forward[final_slots]
    # Output: the measure's rounding, one product and the final composition.
    rounding = _one_plus(unit)
    content = np.nextafter(
        _times(_times(_times(_one_plus(content), _one_plus(measure_bounds)), rounding), rounding)
        - 1.0,
        np.inf,
    )
    reverse = np.zeros((slots,), dtype=np.float64)
    reverse[final_slots] = measure_bounds
    for position in reversed(range(count)):
        valid = weight_rows[position] < weight_count
        lanes_q = weight_bounds[weight_rows[position, valid]]
        writes = write_rows[position][target_lanes[position, valid]]
        inherited = reverse[writes] / np.nextafter(1.0 - reverse[writes], -np.inf)
        destinations = source_lanes[position, valid]
        counts = np.bincount(destinations, minlength=read_capacity)
        q = np.zeros((read_capacity,), dtype=np.float64)
        r = np.zeros((read_capacity,), dtype=np.float64)
        np.maximum.at(q, destinations, lanes_q)
        np.maximum.at(r, destinations, np.nextafter(inherited, np.inf))
        step = _step_bounds(r, q, counts, unit, tiny)
        written = write_rows[position] < slots
        reverse[write_rows[position, written]] = 0.0
        read = read_rows[position] < slots
        reverse[read_rows[position, read]] = step[read]
    coverage = np.nextafter(_times(_one_plus(reverse[:source_count]), rounding) - 1.0, np.inf)
    return MaterialChainBounds(content, coverage, float(info.smallest_subnormal))


__all__ = [
    "MaterialChainBounds",
    "MaterialChainSchedule",
    "prepare_material_chain_schedule",
]
