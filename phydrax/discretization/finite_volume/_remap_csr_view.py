"""Material-route composition of explicit final CSR rows from a chain schedule."""

from __future__ import annotations

import jax
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ...linalg import LinearCapabilityError, SparseAssemblyPolicy
from .._coordinate_budget import CoordinateEnclosureBudget, CoordinateLiveStorage
from ._material_schedule import MaterialChainSchedule


__all__ = ["material_csr_rows"]

_ENTRY_BYTES = 24
_ARRAY_HEADER_BYTES = 112
# Three numpy array headers and the tuple header per live slot row.
_ROW_OVERHEAD_BYTES = 3 * _ARRAY_HEADER_BYTES + 64
# Slot row list container and its header.
_ROW_LIST_BYTES_PER_SLOT = 8
_ROW_LIST_HEADER_BYTES = 56
# At the final grouping step, contribution arrays retain 92 bytes per input
# entry and group arrays retain at most 69 more; one filtered array copy can
# coexist (8 bytes). Round that 169-byte peak to a full 16-byte word boundary.
_TEMPORARY_BYTES_PER_CONTRIBUTION = 176
# Lane arrays, row references, and one transient concatenate-input list.
_TEMPORARY_BYTES_PER_LANE = 64
_CHUNK_ARRAY_HEADERS = 40
_CHUNK_LIST_HEADERS = 3
# Final conversion arrays can coexist with their source rows and outputs.
_FINAL_TEMPORARY_BYTES_PER_ENTRY = 84
_FINAL_TEMPORARY_BYTES_PER_TARGET = 56
_FINAL_ARRAY_HEADERS = 32
# Gather, multiply, renormalize, align, sum and scatter per contribution.
_WORK_PER_CONTRIBUTION = 8

_Row = tuple[np.ndarray, np.ndarray, np.ndarray]


def _host(value: ArrayLike, dtype: type, /) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=dtype)


def _size(value: ArrayLike, /) -> int:
    return int(np.prod(np.shape(value), dtype=np.int64))


def _staging_bytes(value: Array | np.ndarray, dtype: type, /) -> int:
    """Host transfer/cast peak, including each newly created array header."""
    size = _size(value)
    output = np.dtype(dtype)
    if isinstance(value, np.ndarray):
        if value.dtype == output:
            return 0
        return size * output.itemsize + _ARRAY_HEADER_BYTES
    input_bytes = size * np.dtype(value.dtype).itemsize + _ARRAY_HEADER_BYTES
    if np.dtype(value.dtype) == output:
        return input_bytes
    return input_bytes + size * output.itemsize + _ARRAY_HEADER_BYTES


def _row_bytes(entries: int, rows: int, slots: int, /) -> int:
    return (
        entries * _ENTRY_BYTES
        + rows * _ROW_OVERHEAD_BYTES
        + slots * _ROW_LIST_BYTES_PER_SLOT
        + _ROW_LIST_HEADER_BYTES
    )

def _sort_work(count: int, /) -> int:
    return count * max(1, int(count).bit_length())


def _chunk_temporary_bytes(
    lanes: int, lane_capacity: int, contributions: int, written: int, /
) -> int:
    return (
        contributions * _TEMPORARY_BYTES_PER_CONTRIBUTION
        + lanes * _TEMPORARY_BYTES_PER_LANE
        + lane_capacity
        + 16 * (written + 1)
        + _CHUNK_ARRAY_HEADERS * _ARRAY_HEADER_BYTES
        + _CHUNK_LIST_HEADERS * 56
    )


def _empty_row() -> _Row:
    return (
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.float64),
        np.zeros(0, dtype=np.int64),
    )


def _refuse(message: str, /) -> LinearCapabilityError:
    return LinearCapabilityError(f"Material remap CSR view refused: {message}")


def material_csr_rows(
    schedule: MaterialChainSchedule,
    mantissas: Array | np.ndarray,
    measures: Array | np.ndarray,
    policy: SparseAssemblyPolicy,
    budget: CoordinateEnclosureBudget,
    output: CoordinateLiveStorage,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the CSR rows ``(offsets, sources, measures)`` of ``measures_t * A_to``.

    Every entry is carried as an exact-range pair ``(mantissa, exponent)`` so
    chains whose weights leave the float64 range compose without overflow or
    premature underflow; only the final measure conversion rounds into float64.
    The policy admits staging, contributions, live workspace, nonzeros and
    output bytes before the corresponding arrays exist. The budget is billed
    before every allocation; superseded rows release their live charge. The
    returned rows are admitted, at their pre-underflow capacity, by raising
    the caller's ``output`` lease (of ``budget``) before they are allocated;
    the caller owns that credit afterwards.

    The numerical passes intentionally share one lexical scope: the workspace
    bound follows simultaneous array lifetimes, including superseded rows and
    final outputs. Keeping allocations beside their lease transitions makes
    that peak auditable without hiding live arrays behind helper returns.
    """
    banks = (
        (schedule.read_windows.target_indices, np.int64),
        (schedule.write_windows.target_indices, np.int64),
        (schedule.final_windows.target_indices, np.int64),
        (schedule.weight_windows.target_indices, np.int64),
        (schedule.pair_sources, np.int64),
        (schedule.pair_targets, np.int64),
        (schedule.pair_exponents, np.int64),
        (mantissas, np.float64),
        (measures, np.float64),
    )
    slot_count = int(schedule.read_windows.target_size)
    weight_count = int(schedule.weight_windows.target_size)
    source_count = int(schedule.source_count)
    target_count = int(schedule.target_count)
    if np.shape(measures) != (target_count,):
        raise ValueError(
            "material_csr_rows.measures must hold one measure per target cell; "
            f"got shape {np.shape(measures)}."
        )
    if np.shape(mantissas) != (weight_count,):
        raise ValueError(
            "material_csr_rows.mantissas must hold one mantissa per weight row; "
            f"got shape {np.shape(mantissas)}."
        )
    staging_work = sum(_size(bank) for bank, _ in banks)
    staging_bytes = sum(_staging_bytes(bank, dtype) for bank, dtype in banks)
    initial_live = _row_bytes(source_count, source_count, slot_count)
    if staging_bytes + initial_live > policy.max_workspace_bytes:
        raise _refuse(
            f"staging {staging_bytes} bytes plus initial rows {initial_live} bytes "
            f"exceed max_workspace_bytes={policy.max_workspace_bytes}."
        )

    with budget.temporary_scope():
        budget.reserve(staging_work, staging_bytes)
        read = _host(schedule.read_windows.target_indices, np.int64)
        write = _host(schedule.write_windows.target_indices, np.int64)
        final = _host(schedule.final_windows.target_indices, np.int64)
        weights = _host(schedule.weight_windows.target_indices, np.int64)
        pair_sources = _host(schedule.pair_sources, np.int64)
        pair_targets = _host(schedule.pair_targets, np.int64)
        pair_exponents = _host(schedule.pair_exponents, np.int64)
        weight_mantissas = _host(mantissas, np.float64)
        target_measures = _host(measures, np.float64)

        with budget.live_storage() as live:
            live.set_bound(initial_live, work=source_count)
            rows: list[_Row | None] = [None] * slot_count
            for k in range(source_count):
                rows[k] = (
                    np.array((k,), dtype=np.int64),
                    np.array((0.5,), dtype=np.float64),
                    np.array((1,), dtype=np.int64),
                )
            live_entries = source_count
            live_rows = source_count
            contributions = 0

            for chunk in range(read.shape[0]):
                budget.reserve(weights.shape[1] + write.shape[1])
                lane_count = 0
                chunk_contributions = 0
                for lane in range(weights.shape[1]):
                    weight_row = int(weights[chunk, lane])
                    if weight_row >= weight_count:
                        continue
                    lane_count += 1
                    slot = int(
                        read[chunk, int(pair_sources[chunk, lane])]
                    )
                    row = rows[slot] if slot < slot_count else None
                    if row is not None:
                        chunk_contributions += row[0].size
                if lane_count == 0:
                    continue
                contributions += chunk_contributions
                if contributions > policy.max_contributions:
                    raise _refuse(
                        f"{contributions} contributions exceed "
                        f"max_contributions={policy.max_contributions}."
                    )
                written_size = write.shape[1]
                written_count = sum(
                    int(write[chunk, lane] < slot_count)
                    for lane in range(written_size)
                )
                temporary_bytes = _chunk_temporary_bytes(
                    lane_count, weights.shape[1], chunk_contributions, written_size
                )
                # Old and new rows coexist until the chunk's writes complete.
                live_upper = _row_bytes(
                    live_entries + chunk_contributions,
                    live_rows + written_count,
                    slot_count,
                )
                workspace = staging_bytes + live_upper + temporary_bytes
                if workspace > policy.max_workspace_bytes:
                    raise _refuse(
                        f"live row workspace {workspace} bytes exceeds "
                        f"max_workspace_bytes={policy.max_workspace_bytes}."
                    )
                live.set_bound(live_upper)
                budget.reserve(
                    chunk_contributions * _WORK_PER_CONTRIBUTION
                    + _sort_work(chunk_contributions)
                    + written_size
                    + lane_count,
                    temporary_bytes,
                )
                lanes = np.flatnonzero(weights[chunk] < weight_count)
                written = write[chunk]
                read_slots = read[chunk, pair_sources[chunk, lanes]]
                write_lanes = pair_targets[chunk, lanes]
                lane_rows = [
                    rows[slot] if slot < slot_count and rows[slot] is not None else None
                    for slot in read_slots
                ]
                lengths = np.fromiter(
                    (0 if row is None else row[0].size for row in lane_rows),
                    dtype=np.int64,
                    count=lane_count,
                )

                if chunk_contributions:
                    present = [row for row in lane_rows if row is not None]
                    sources = np.concatenate([row[0] for row in present])
                    entry_mantissas = np.concatenate([row[1] for row in present])
                    entry_exponents = np.concatenate([row[2] for row in present])
                    weight_rows = weights[chunk, lanes]
                    lane_mantissas = np.repeat(weight_mantissas[weight_rows], lengths)
                    lane_exponents = np.repeat(pair_exponents[chunk, lanes], lengths)
                    keys = np.repeat(write_lanes, lengths)
                    product, shift = np.frexp(entry_mantissas * lane_mantissas)
                    exponents = (
                        entry_exponents + lane_exponents + shift.astype(np.int64)
                    )
                    order = np.lexsort((sources, keys))
                    keys = keys[order]
                    sources = sources[order]
                    product = product[order]
                    exponents = exponents[order]
                    starts = np.flatnonzero(
                        np.concatenate(
                            (
                                np.ones(1, dtype=np.bool_),
                                (keys[1:] != keys[:-1])
                                | (sources[1:] != sources[:-1]),
                            )
                        )
                    )
                    group_max = np.maximum.reduceat(exponents, starts)
                    group_sizes = np.diff(np.append(starts, keys.size))
                    offset = exponents - np.repeat(group_max, group_sizes)
                    aligned = np.ldexp(
                        product, np.maximum(offset, -2100).astype(np.int32)
                    )
                    sums = np.add.reduceat(aligned, starts)
                    sum_mantissas, sum_shift = np.frexp(sums)
                    sum_exponents = group_max + sum_shift.astype(np.int64)
                    group_keys = keys[starts]
                    group_sources = sources[starts]
                    keep = sum_mantissas != 0.0
                    group_keys = group_keys[keep]
                    group_sources = group_sources[keep]
                    sum_mantissas = sum_mantissas[keep]
                    sum_exponents = sum_exponents[keep]
                    bounds = np.searchsorted(
                        group_keys, np.arange(written.size + 1, dtype=np.int64)
                    )
                else:
                    bounds = np.zeros(written.size + 1, dtype=np.int64)
                for lane in range(written.size):
                    slot = int(written[lane])
                    if slot >= slot_count:
                        continue
                    prior = rows[slot]
                    if prior is not None:
                        live_entries -= prior[0].size
                        live_rows -= 1
                    begin, end = int(bounds[lane]), int(bounds[lane + 1])
                    if begin == end:
                        rows[slot] = _empty_row()
                    else:
                        rows[slot] = (
                            group_sources[begin:end].copy(),
                            sum_mantissas[begin:end].copy(),
                            sum_exponents[begin:end].copy(),
                        )
                    live_entries += end - begin
                    live_rows += 1
                if chunk_contributions:
                    del (
                        aligned,
                        bounds,
                        entry_exponents,
                        entry_mantissas,
                        sources,
                        exponents,
                        group_keys,
                        group_max,
                        group_sizes,
                        group_sources,
                        keep,
                        keys,
                        offset,
                        order,
                        present,
                        product,
                        shift,
                        starts,
                        sum_exponents,
                        sum_mantissas,
                        sum_shift,
                        sums,
                        lane_exponents,
                        lane_mantissas,
                        weight_rows,
                    )
                else:
                    del bounds
                del lanes, lane_rows, lengths, read_slots, row, write_lanes, written
                if written_count:
                    del prior
                budget.release_temporary_storage(temporary_bytes)
                # Superseded rows are gone; the charge follows the live rows.
                live.set_bound(_row_bytes(live_entries, live_rows, slot_count))

            # Count rows by scalar traversal: no final slot list, row list or
            # length bank exists before the final pass is admitted.
            nnz_upper = 0
            empty_rows = 0
            for target in range(target_count):
                slot = int(final[0, target])
                row = rows[slot] if slot < slot_count else None
                if row is None:
                    empty_rows += 1
                else:
                    nnz_upper += row[0].size
            if nnz_upper > policy.max_nnz:
                raise _refuse(
                    f"{nnz_upper} nonzeros exceed max_nnz={policy.max_nnz}."
                )
            # max_bytes caps the complete view payload, including its target
            # routes which are created by the caller after these rows return.
            output_upper = 4 * (target_count + 1) + 16 * nnz_upper
            if output_upper > policy.max_bytes:
                raise _refuse(
                    f"output {output_upper} bytes exceeds max_bytes={policy.max_bytes}."
                )
            final_temporary = (
                nnz_upper * _FINAL_TEMPORARY_BYTES_PER_ENTRY
                + target_count * _FINAL_TEMPORARY_BYTES_PER_TARGET
                + target_count * 8
                + 56
                + empty_rows * _ROW_OVERHEAD_BYTES
                + _FINAL_ARRAY_HEADERS * _ARRAY_HEADER_BYTES
            )
            host_output = (
                4 * (target_count + 1)
                + 12 * nnz_upper
                + 3 * _ARRAY_HEADER_BYTES
            )
            peak = (
                staging_bytes
                + _row_bytes(live_entries, live_rows, slot_count)
                + final_temporary
                + host_output
            )
            if peak > policy.max_workspace_bytes:
                raise _refuse(
                    f"final workspace {peak} bytes exceeds "
                    f"max_workspace_bytes={policy.max_workspace_bytes}."
                )
            budget.reserve(
                nnz_upper * 4 + 3 * target_count,
                final_temporary,
            )
            # The returned host rows outlive this scratch scope; admit their
            # exact payload and array headers on the caller's lease first.
            output.set_bound(
                output.bound + host_output,
                work=target_count + 1 + 2 * nnz_upper,
            )
            final_rows = []
            for target in range(target_count):
                slot = int(final[0, target])
                row = rows[slot] if slot < slot_count else None
                final_rows.append(_empty_row() if row is None else row)
            lengths = np.fromiter(
                (row[0].size for row in final_rows),
                dtype=np.int64,
                count=target_count,
            )
            measure_mantissas, measure_exponents = np.frexp(target_measures)
            if nnz_upper:
                sources = np.concatenate([row[0] for row in final_rows])
                entry_mantissas = np.concatenate([row[1] for row in final_rows])
                entry_exponents = np.concatenate([row[2] for row in final_rows])
                values = np.ldexp(
                    np.repeat(measure_mantissas, lengths) * entry_mantissas,
                    np.clip(
                        entry_exponents
                        + np.repeat(measure_exponents.astype(np.int64), lengths),
                        -2200,
                        2200,
                    ).astype(np.int32),
                )
                del entry_mantissas, entry_exponents
                rows_of = np.repeat(np.arange(target_count, dtype=np.int64), lengths)
                del final_rows, lengths, measure_exponents, measure_mantissas
                keep = values != 0.0
                sources = sources[keep]
                values = values[keep]
                rows_of = rows_of[keep]
                del keep
            else:
                sources = np.zeros(0, dtype=np.int64)
                values = np.zeros(0, dtype=np.float64)
                rows_of = np.zeros(0, dtype=np.int64)
                del final_rows, lengths, measure_exponents, measure_mantissas
            offsets = np.zeros(target_count + 1, dtype=np.int32)
            counts = np.bincount(rows_of, minlength=target_count)
            np.cumsum(counts, out=offsets[1:], dtype=np.int32)
            del counts, rows_of
            source_indices = sources.astype(np.int32)
    return offsets, source_indices, values
