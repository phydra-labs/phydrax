from __future__ import annotations

from fractions import Fraction

import jax
import numpy as np

from ...linalg import LinearCapabilityError, SparseAssemblyPolicy
from ._material_schedule import MaterialChainSchedule


__all__ = ["material_csr_rows"]


def _fraction(mantissa: float, exponent: int, /) -> Fraction:
    value = Fraction.from_float(mantissa)
    if exponent >= 0:
        return value * (Fraction(2) ** exponent)
    return value / (Fraction(2) ** -exponent)


def _fraction_bits(value: Fraction, /) -> int:
    return max(abs(value.numerator).bit_length(), value.denominator.bit_length())


def _weight_bits(mantissa: float, exponent: int, /) -> int:
    numerator, denominator = Fraction.from_float(mantissa).as_integer_ratio()
    return max(
        abs(numerator).bit_length() + max(exponent, 0),
        denominator.bit_length() + max(-exponent, 0),
    )


def material_csr_rows(
    schedule: MaterialChainSchedule,
    policy: SparseAssemblyPolicy,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose the scheduled sparse action into sorted target-row CSR on request."""
    if not isinstance(schedule, MaterialChainSchedule):
        raise TypeError("schedule must be a MaterialChainSchedule.")
    if not isinstance(policy, SparseAssemblyPolicy):
        raise TypeError("policy must be a SparseAssemblyPolicy.")
    staging_bytes = 256
    for bank in (
        schedule.source_row_slots,
        schedule.target_row_slots,
        schedule.source_row_valid,
        schedule.target_row_valid,
        schedule.edge_source_slots,
        schedule.edge_target_slots,
        schedule.edge_weight_mantissas,
        schedule.edge_weight_exponents,
        schedule.edge_valid,
        schedule.initial_slots,
        schedule.final_slots,
    ):
        staging_bytes += bank.size * np.dtype(bank.dtype).itemsize + 112
    if staging_bytes > policy.max_workspace_bytes:
        raise LinearCapabilityError(
            "Material remap CSR view exceeds SparseAssemblyPolicy.max_workspace_bytes."
        )
    source_rows = np.asarray(jax.device_get(schedule.source_row_slots), dtype=np.int64)
    target_rows = np.asarray(jax.device_get(schedule.target_row_slots), dtype=np.int64)
    source_valid = np.asarray(jax.device_get(schedule.source_row_valid), dtype=np.bool_)
    target_valid = np.asarray(jax.device_get(schedule.target_row_valid), dtype=np.bool_)
    edge_sources = np.asarray(jax.device_get(schedule.edge_source_slots), dtype=np.int64)
    edge_targets = np.asarray(jax.device_get(schedule.edge_target_slots), dtype=np.int64)
    edge_mantissas = np.asarray(
        jax.device_get(schedule.edge_weight_mantissas), dtype=np.float64
    )
    edge_exponents = np.asarray(
        jax.device_get(schedule.edge_weight_exponents), dtype=np.int64
    )
    edge_valid = np.asarray(jax.device_get(schedule.edge_valid), dtype=np.bool_)
    rows: list[dict[int, Fraction]] = [
        {source: Fraction(1)} for source in range(schedule.source_count)
    ]
    current_slots = np.asarray(jax.device_get(schedule.initial_slots), dtype=np.int64)
    contributions = schedule.source_count
    live_entries = schedule.source_count
    for map_index in range(schedule.map_count):
        sources = source_rows[map_index, source_valid[map_index]]
        targets = target_rows[map_index, target_valid[map_index]]
        source_row_by_slot = {int(slot): row for row, slot in enumerate(sources)}
        target_row_by_slot = {int(slot): row for row, slot in enumerate(targets)}
        source_slot_to_rows = np.asarray(
            [
                source_row_by_slot[int(slot)]
                for slot in edge_sources[map_index, edge_valid[map_index]]
            ],
            dtype=np.int64,
        )
        target_slot_to_rows = np.asarray(
            [
                target_row_by_slot[int(slot)]
                for slot in edge_targets[map_index, edge_valid[map_index]]
            ],
            dtype=np.int64,
        )
        lanes = np.flatnonzero(edge_valid[map_index])
        expanded = sum(len(rows[int(row)]) for row in source_slot_to_rows)
        if contributions + expanded > policy.max_contributions:
            raise LinearCapabilityError(
                "Material remap CSR view exceeds SparseAssemblyPolicy.max_contributions."
            )
        input_bits = max(
            (_fraction_bits(value) for row in rows for value in row.values()),
            default=1,
        )
        weight_bits = max(
            (
                _weight_bits(
                    float(edge_mantissas[map_index, lane]),
                    int(edge_exponents[map_index, lane]),
                )
                for lane in lanes
            ),
            default=1,
        )
        result_bits = input_bits + weight_bits + expanded.bit_length() + 1
        fraction_bytes = 64 + 2 * ((result_bits + 7) // 8)
        workspace = (
            staging_bytes
            + 256 * (len(rows) + expanded + targets.size)
            + fraction_bytes * (live_entries + 2 * expanded)
        )
        if workspace > policy.max_workspace_bytes:
            raise LinearCapabilityError(
                "Material remap CSR view exceeds SparseAssemblyPolicy.max_workspace_bytes."
            )
        output_rows: list[dict[int, Fraction]] = [dict() for _ in range(targets.size)]
        for lane, source_row, target_row in zip(
            lanes.tolist(),
            source_slot_to_rows.tolist(),
            target_slot_to_rows.tolist(),
            strict=True,
        ):
            weight = _fraction(
                float(edge_mantissas[map_index, lane]),
                int(edge_exponents[map_index, lane]),
            )
            destination = output_rows[target_row]
            for column, value in rows[source_row].items():
                destination[column] = (
                    destination.get(column, Fraction(0)) + weight * value
                )
        live_entries = sum(len(row) for row in output_rows)
        if live_entries > policy.max_nnz:
            raise LinearCapabilityError(
                "Material remap CSR view exceeds SparseAssemblyPolicy.max_nnz."
            )
        contributions += expanded
        rows = output_rows
        current_slots = targets
    if not np.array_equal(
        current_slots,
        np.asarray(jax.device_get(schedule.final_slots), dtype=np.int64),
    ):
        raise ValueError("Material remap CSR view lost the scheduled target endpoint.")
    offsets = np.zeros((schedule.target_count + 1,), dtype=np.int64)
    for row, values in enumerate(rows):
        offsets[row + 1] = offsets[row] + len(values)
    nnz = int(offsets[-1])
    output_bytes = offsets.nbytes + nnz * (
        np.dtype(np.int64).itemsize + np.dtype(np.float64).itemsize
    )
    if nnz > policy.max_nnz or output_bytes > policy.max_bytes:
        raise LinearCapabilityError(
            "Material remap CSR view exceeds sparse output policy."
        )
    sources = np.empty((nnz,), dtype=np.int64)
    values = np.empty((nnz,), dtype=np.float64)
    position = 0
    for row in rows:
        for column, exact in sorted(row.items()):
            rounded = float(exact)
            if not np.isfinite(rounded) or rounded == 0.0:
                raise LinearCapabilityError(
                    "Material remap CSR coefficients are not representable in binary64."
                )
            sources[position] = column
            values[position] = rounded
            position += 1
    return offsets, sources, values
