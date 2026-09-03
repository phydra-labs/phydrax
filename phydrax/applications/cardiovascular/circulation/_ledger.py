#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ._periodic import pressure_volume_work
from ._valves import ValveEventCandidate


def _time_series(
    time: ArrayLike,
    value: ArrayLike,
    owner: str,
    /,
) -> tuple[Array, Array]:
    time_ = jnp.asarray(time)
    value_ = jnp.asarray(value)
    if time_.ndim != 1 or time_.shape[0] < 2:
        raise ValueError(
            "time must be a one-dimensional series with at least two samples."
        )
    if value_.shape[0] != time_.shape[0]:
        raise ValueError(f"{owner} leading axis must match time.")
    host_time = np.asarray(time_)
    if (
        not np.all(np.isfinite(host_time))
        or np.any(np.diff(host_time) <= 0.0)
        or not bool(jnp.all(jnp.isfinite(value_)))
    ):
        raise ValueError(f"time and {owner} must be finite with increasing time.")
    return time_, value_


def _cumulative_trapezoid(time: Array, value: Array, /) -> Array:
    increments = 0.5 * (value[1:] + value[:-1]) * (time[1:] - time[:-1])
    return jnp.concatenate(
        (jnp.zeros((1,), dtype=increments.dtype), jnp.cumsum(increments))
    )


class TotalVolumeLedger(StrictModule):
    """Evidence for closed or boundary-driven blood-volume conservation."""

    time: Array
    total_volume: Array
    expected_volume: Array
    balance_residual: Array
    maximum_absolute_residual: Array
    relative_residual: Array
    finite: Array
    conserved: Array
    tolerance: Array
    ledger_id: str = eqx.field(static=True)


def audit_total_volume(
    time: ArrayLike,
    total_volume: ArrayLike,
    /,
    *,
    net_boundary_flow: ArrayLike | None = None,
    tolerance: ArrayLike = 1.0e-8,
) -> TotalVolumeLedger:
    """Audit V(t)-V(0)=∫(Qin-Qout)dt in kernel volume units."""

    time_, volume = _time_series(time, total_volume, "total_volume")
    if volume.ndim != 1:
        raise ValueError("total_volume must be one-dimensional.")
    tolerance_ = jnp.asarray(tolerance)
    if tolerance_.shape != () or not bool(jnp.isfinite(tolerance_) & (tolerance_ >= 0.0)):
        raise ValueError("tolerance must be finite and nonnegative.")
    boundary = (
        jnp.zeros_like(volume)
        if net_boundary_flow is None
        else jnp.asarray(net_boundary_flow)
    )
    if boundary.shape != volume.shape or not bool(jnp.all(jnp.isfinite(boundary))):
        raise ValueError("net_boundary_flow must be a finite time-aligned series.")
    transfer = _cumulative_trapezoid(time_, boundary)
    expected = volume[0] + transfer
    residual = volume - expected
    maximum = jnp.max(jnp.abs(residual))
    scale = jnp.maximum(jnp.max(jnp.abs(expected)), 1.0)
    relative = maximum / scale
    finite = (
        jnp.all(jnp.isfinite(expected))
        & jnp.all(jnp.isfinite(residual))
        & jnp.isfinite(relative)
    )
    conserved = finite & (relative <= tolerance_)
    return TotalVolumeLedger(
        time_,
        volume,
        expected,
        residual,
        maximum,
        relative,
        finite,
        conserved,
        tolerance_,
        canonical_fingerprint(
            {
                "kind": "circulation-total-volume-ledger",
                "samples": int(time_.shape[0]),
                "tolerance": float(tolerance_).hex(),
            }
        ),
    )


class PassivityLedger(StrictModule):
    """Integrated power balance ΔE + D - Win ≤ tolerance."""

    time: Array
    input_power: Array
    stored_energy: Array
    dissipated_power: Array
    input_work: Array
    dissipated_energy: Array
    stored_energy_change: Array
    balance_residual: Array
    passivity_violation: Array
    finite: Array
    passive: Array
    tolerance: Array
    ledger_id: str = eqx.field(static=True)


def audit_passivity(
    time: ArrayLike,
    input_power: ArrayLike,
    stored_energy: ArrayLike,
    /,
    *,
    dissipated_power: ArrayLike | None = None,
    tolerance: ArrayLike = 1.0e-8,
) -> PassivityLedger:
    """Audit supplied work against stored-energy growth and dissipation."""

    time_, power = _time_series(time, input_power, "input_power")
    energy = jnp.asarray(stored_energy)
    if power.ndim != 1 or energy.shape != power.shape:
        raise ValueError("Power and stored energy must be time-aligned vectors.")
    if not bool(jnp.all(jnp.isfinite(energy))):
        raise ValueError("stored_energy must be finite.")
    dissipation = (
        jnp.zeros_like(power)
        if dissipated_power is None
        else jnp.asarray(dissipated_power)
    )
    if dissipation.shape != power.shape or not bool(
        jnp.all(jnp.isfinite(dissipation)) & jnp.all(dissipation >= 0.0)
    ):
        raise ValueError("dissipated_power must be finite, nonnegative, and aligned.")
    tolerance_ = jnp.asarray(tolerance)
    if tolerance_.shape != () or not bool(jnp.isfinite(tolerance_) & (tolerance_ >= 0.0)):
        raise ValueError("tolerance must be finite and nonnegative.")
    input_work = _cumulative_trapezoid(time_, power)
    dissipated_energy = _cumulative_trapezoid(time_, dissipation)
    energy_change = energy - energy[0]
    residual = energy_change + dissipated_energy - input_work
    scale = jnp.maximum(
        jnp.maximum(jnp.max(jnp.abs(input_work)), jnp.max(jnp.abs(energy))),
        1.0,
    )
    violation = jnp.maximum(jnp.max(residual) / scale, 0.0)
    finite = (
        jnp.all(jnp.isfinite(input_work))
        & jnp.all(jnp.isfinite(dissipated_energy))
        & jnp.all(jnp.isfinite(residual))
    )
    passive = finite & (violation <= tolerance_)
    return PassivityLedger(
        time_,
        power,
        energy,
        dissipation,
        input_work,
        dissipated_energy,
        energy_change,
        residual,
        violation,
        finite,
        passive,
        tolerance_,
        canonical_fingerprint(
            {
                "kind": "circulation-passivity-ledger",
                "samples": int(time_.shape[0]),
                "tolerance": float(tolerance_).hex(),
            }
        ),
    )


class ValveEventRecord(StrictModule):
    valve_id: str = eqx.field(static=True)
    event_id: str = eqx.field(static=True)
    event_index: int = eqx.field(static=True)
    time: Array
    pressure_drop: Array
    direction: Array
    is_open: Array


class ValveEventLedger(StrictModule):
    """Immutable ordered record of committed deterministic valve transitions."""

    records: tuple[ValveEventRecord, ...]
    ledger_id: str = eqx.field(static=True)

    def __init__(self, records: Sequence[ValveEventRecord] = (), /) -> None:
        values = tuple(records)
        if any(not isinstance(value, ValveEventRecord) for value in values):
            raise TypeError("records must contain ValveEventRecord values.")
        for index, record in enumerate(values):
            if record.event_index != index:
                raise ValueError("Valve event indices must be contiguous and ordered.")
            if index > 0 and float(record.time) < float(values[index - 1].time):
                raise ValueError("Valve event times must be nondecreasing.")
        self.records = values
        self.ledger_id = canonical_fingerprint(
            {
                "kind": "circulation-valve-event-ledger",
                "events": [value.event_id for value in values],
            }
        )


class ValveLedgerEvidence(StrictModule):
    finite: Array
    chronological: Array
    alternating: Array
    dwell_satisfied: Array
    deterministic: Array
    event_count: Array
    evidence_id: str = eqx.field(static=True)


def record_valve_event(
    ledger: ValveEventLedger,
    candidate: ValveEventCandidate,
    /,
) -> ValveEventLedger:
    """Commit one required event to a new immutable valve ledger."""

    if not isinstance(ledger, ValveEventLedger):
        raise TypeError("ledger must be a ValveEventLedger.")
    if not isinstance(candidate, ValveEventCandidate):
        raise TypeError("candidate must be a ValveEventCandidate.")
    if not bool(candidate.event_required):
        raise ValueError("A non-event candidate cannot be recorded.")
    index = len(ledger.records)
    direction = int(candidate.direction)
    if direction not in (-1, 1):
        raise ValueError(
            "Committed valve events require an opening or closing direction."
        )
    event_id = canonical_fingerprint(
        {
            "kind": "circulation-valve-event",
            "valve": candidate.valve_id,
            "index": index,
            "time": float(candidate.event_time).hex(),
            "direction": direction,
        }
    )
    record = ValveEventRecord(
        candidate.valve_id,
        event_id,
        index,
        candidate.event_time,
        candidate.pressure_drop,
        candidate.direction,
        candidate.candidate_state.is_open,
    )
    return ValveEventLedger(ledger.records + (record,))


def audit_valve_events(
    ledger: ValveEventLedger,
    /,
    *,
    minimum_dwell_time: ArrayLike = 0.0,
) -> ValveLedgerEvidence:
    """Check event order, per-valve alternation, dwell, and finite evidence."""

    if not isinstance(ledger, ValveEventLedger):
        raise TypeError("ledger must be a ValveEventLedger.")
    dwell = jnp.asarray(minimum_dwell_time)
    if dwell.shape != () or not bool(jnp.isfinite(dwell) & (dwell >= 0.0)):
        raise ValueError("minimum_dwell_time must be finite and nonnegative.")
    if not ledger.records:
        true = jnp.asarray(True)
        return ValveLedgerEvidence(
            true,
            true,
            true,
            true,
            true,
            jnp.asarray(0, dtype=jnp.int32),
            canonical_fingerprint(
                {
                    "kind": "circulation-valve-ledger-evidence",
                    "ledger": ledger.ledger_id,
                    "minimum_dwell_time": float(dwell).hex(),
                }
            ),
        )
    times = jnp.stack(tuple(value.time for value in ledger.records))
    pressures = jnp.stack(tuple(value.pressure_drop for value in ledger.records))
    finite = jnp.all(jnp.isfinite(times)) & jnp.all(jnp.isfinite(pressures))
    chronological = jnp.all(times[1:] >= times[:-1])
    alternating = jnp.asarray(True)
    dwell_satisfied = jnp.asarray(True)
    by_valve: dict[str, list[ValveEventRecord]] = {}
    for record in ledger.records:
        if record.valve_id not in by_valve:
            by_valve[record.valve_id] = []
        by_valve[record.valve_id].append(record)
    for records in by_valve.values():
        for previous, current in zip(records[:-1], records[1:], strict=True):
            alternating = alternating & (current.direction == -previous.direction)
            dwell_satisfied = dwell_satisfied & (current.time - previous.time >= dwell)
    deterministic = finite & chronological & alternating & dwell_satisfied
    return ValveLedgerEvidence(
        finite,
        chronological,
        alternating,
        dwell_satisfied,
        deterministic,
        jnp.asarray(len(ledger.records), dtype=jnp.int32),
        canonical_fingerprint(
            {
                "kind": "circulation-valve-ledger-evidence",
                "ledger": ledger.ledger_id,
                "minimum_dwell_time": float(dwell).hex(),
            }
        ),
    )


class PressureVolumeWorkLedger(StrictModule):
    pressure: Array
    volume: Array
    chamber_work: Array
    pressure_closure: Array
    volume_closure: Array
    stroke_volume: Array
    finite: Array
    closed: Array
    ledger_id: str = eqx.field(static=True)


def audit_pressure_volume_cycle(
    pressure: ArrayLike,
    volume: ArrayLike,
    /,
    *,
    closure_tolerance: ArrayLike = 1.0e-8,
) -> PressureVolumeWorkLedger:
    """Audit cycle closure and compute signed chamber pressure-volume work."""

    pressure_ = jnp.asarray(pressure)
    volume_ = jnp.asarray(volume)
    if pressure_.ndim != 1 or volume_.shape != pressure_.shape:
        raise ValueError("pressure and volume must be aligned one-dimensional cycles.")
    tolerance = jnp.asarray(closure_tolerance)
    if tolerance.shape != () or not bool(jnp.isfinite(tolerance) & (tolerance >= 0.0)):
        raise ValueError("closure_tolerance must be finite and nonnegative.")
    work = pressure_volume_work(pressure_, volume_)
    pressure_closure = pressure_[-1] - pressure_[0]
    volume_closure = volume_[-1] - volume_[0]
    pressure_scale = jnp.maximum(jnp.max(jnp.abs(pressure_)), 1.0)
    volume_scale = jnp.maximum(jnp.max(jnp.abs(volume_)), 1.0)
    finite = (
        jnp.all(jnp.isfinite(pressure_))
        & jnp.all(jnp.isfinite(volume_))
        & jnp.isfinite(work)
    )
    closed = (
        finite
        & (jnp.abs(pressure_closure) <= tolerance * pressure_scale)
        & (jnp.abs(volume_closure) <= tolerance * volume_scale)
    )
    return PressureVolumeWorkLedger(
        pressure_,
        volume_,
        work,
        pressure_closure,
        volume_closure,
        jnp.max(volume_) - jnp.min(volume_),
        finite,
        closed,
        canonical_fingerprint(
            {
                "kind": "circulation-pressure-volume-work-ledger",
                "samples": int(pressure_.shape[0]),
                "closure_tolerance": float(tolerance).hex(),
            }
        ),
    )


__all__ = [
    "PassivityLedger",
    "PressureVolumeWorkLedger",
    "TotalVolumeLedger",
    "ValveEventLedger",
    "ValveEventRecord",
    "ValveLedgerEvidence",
    "audit_passivity",
    "audit_pressure_volume_cycle",
    "audit_total_volume",
    "audit_valve_events",
    "record_valve_event",
]
