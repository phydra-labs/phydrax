#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact deterministic and SSA circulating-blood dose scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ....measurement import (
    QuantitySpec,
    RadiationQuantityKind,
    resolve_radiation_quantity,
)
from ....solver import JumpSolution, solve_direct_ssa
from ....stochastic import JUMP_MAX_EVENTS, PoissonClockRealization
from ....units import GRAY_PER_SECOND
from ._model import _identifier, PreparedCirculatingBloodModel


ABSORBED_DOSE_RATE_REFERENCE = "circulating-blood-absorbed-dose-per-second"
DOSE_TO_WATER_RATE_REFERENCE = "circulating-blood-dose-to-water-per-second"
DOSE_TO_MEDIUM_RATE_REFERENCE = "circulating-blood-dose-to-medium-per-second"
CIRCULATING_BLOOD_DOSE_RATE_REFERENCES = (
    ABSORBED_DOSE_RATE_REFERENCE,
    DOSE_TO_WATER_RATE_REFERENCE,
    DOSE_TO_MEDIUM_RATE_REFERENCE,
)
CIRCULATING_BLOOD_DOSE_RATE_SUPPORT = (
    "circulating-blood-compartment-piecewise-constant-time-interval"
)


def circulating_blood_dose_rate_quantity(
    name: str,
    reference_configuration: str,
    /,
) -> QuantitySpec:
    """Resolve one admitted Gy/s score without collapsing its material basis."""

    if reference_configuration not in CIRCULATING_BLOOD_DOSE_RATE_REFERENCES:
        raise ValueError(
            "Circulating-blood dose rate must be absorbed dose, dose to water, "
            "or dose to medium on the declared per-second profile."
        )
    return resolve_radiation_quantity(
        name,
        RadiationQuantityKind.DOSE_RATE,
        GRAY_PER_SECOND,
        support_association=CIRCULATING_BLOOD_DOSE_RATE_SUPPORT,
        reference_configuration=reference_configuration,
    )


def _require_dose_rate_quantity(quantity: QuantitySpec, /) -> None:
    if not isinstance(quantity, QuantitySpec):
        raise TypeError("quantity must be a QuantitySpec.")
    if quantity.reference_configuration not in CIRCULATING_BLOOD_DOSE_RATE_REFERENCES:
        raise ValueError(
            "Dose-rate reference_configuration is outside the circulating-blood "
            "absorbed/water/medium allow-list."
        )
    expected = circulating_blood_dose_rate_quantity(
        "circulating_blood_dose_rate",
        quantity.reference_configuration,
    )
    if not quantity.compatible_with(expected):
        raise ValueError(
            "Dose-rate quantity must be a nonnegative Gy/s circulating-blood score "
            "on the exact compartment/time support."
        )
    if quantity.unit.unit_id != GRAY_PER_SECOND.unit_id:
        raise ValueError("Circulating-blood dose-rate values must use Gy/s explicitly.")


def _readonly_vector(value: ArrayLike, name: str, /) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 1 or array.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty vector.")
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class DoseRateInterval:
    """One half-open interval carrying a compartment-aligned constant dose rate.

    Supplied standard uncertainties are independent across compartment entries and
    across schedule intervals. Omit them when that independence is not justified.
    """

    start_s: float
    end_s: float
    quantity: QuantitySpec
    dose_rates_gy_per_s: np.ndarray
    standard_uncertainties_gy_per_s: np.ndarray | None = None
    interval_id: str = field(init=False)

    def __post_init__(self) -> None:
        start = float(self.start_s)
        end = float(self.end_s)
        if not math.isfinite(start) or not math.isfinite(end) or not end > start:
            raise ValueError(
                "Dose-rate intervals require finite bounds with end > start."
            )
        _require_dose_rate_quantity(self.quantity)
        rates = _readonly_vector(self.dose_rates_gy_per_s, "dose_rates_gy_per_s")
        uncertainty = (
            None
            if self.standard_uncertainties_gy_per_s is None
            else _readonly_vector(
                self.standard_uncertainties_gy_per_s,
                "standard_uncertainties_gy_per_s",
            )
        )
        if uncertainty is not None and uncertainty.shape != rates.shape:
            raise ValueError(
                "Dose-rate values and uncertainties must have matching shape."
            )
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "end_s", end)
        object.__setattr__(self, "dose_rates_gy_per_s", rates)
        object.__setattr__(self, "standard_uncertainties_gy_per_s", uncertainty)
        object.__setattr__(
            self,
            "interval_id",
            canonical_fingerprint(
                {
                    "kind": "circulating-blood-dose-rate-interval",
                    "start_s": start,
                    "end_s": end,
                    "quantity": self.quantity.quantity_id,
                    "rates": array_tree_fingerprint(rates),
                    "uncertainty": None
                    if uncertainty is None
                    else array_tree_fingerprint(uncertainty),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class PiecewiseConstantDoseRateSchedule:
    """Ordered, non-overlapping dose-rate intervals; uncovered gaps have zero reward."""

    intervals: tuple[DoseRateInterval, ...]
    schedule_id: str = field(init=False)

    def __post_init__(self) -> None:
        intervals = tuple(self.intervals)
        if not intervals or any(
            not isinstance(value, DoseRateInterval) for value in intervals
        ):
            raise ValueError("intervals must contain DoseRateInterval values.")
        for previous, current in zip(intervals, intervals[1:], strict=False):
            if current.start_s < previous.end_s:
                raise ValueError(
                    "Dose-rate intervals must be ordered and cannot overlap."
                )
        first = intervals[0]
        if any(
            value.dose_rates_gy_per_s.shape != first.dose_rates_gy_per_s.shape
            for value in intervals
        ):
            raise ValueError("Every interval must use the same compartment axis.")
        if any(not value.quantity.compatible_with(first.quantity) for value in intervals):
            raise ValueError(
                "Every interval must use one exact absorbed/water/medium dose-rate meaning."
            )
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(
            self,
            "schedule_id",
            canonical_fingerprint(
                {
                    "kind": "circulating-blood-dose-rate-schedule",
                    "intervals": [value.interval_id for value in intervals],
                }
            ),
        )

    @property
    def compartment_count(self) -> int:
        return int(self.intervals[0].dose_rates_gy_per_s.shape[0])

    @property
    def quantity(self) -> QuantitySpec:
        return self.intervals[0].quantity


@dataclass(frozen=True, slots=True)
class DeterministicBloodDoseResult:
    """Exact expected score conditional on the supplied generator and schedule.

    ``standard_uncertainty_gy`` propagates only declared independent dose-rate
    uncertainty. It remains ``None`` when any active interval lacks that evidence.
    """

    initial_probabilities: Array
    final_probabilities: Array
    occupation_seconds: Array
    compartment_dose_gy: Array
    total_dose_gy: Array
    standard_uncertainty_gy: Array | None
    start_s: float
    end_s: float
    model_id: str
    schedule_id: str


@dataclass(frozen=True, slots=True)
class HistoryCapacityEvidence:
    event_capacity: int
    events_used: Array
    capacity_exceeded: Array
    solver_successful: Array

    @property
    def successful(self) -> Array:
        return self.solver_successful & ~self.capacity_exceeded


@dataclass(frozen=True, slots=True)
class StochasticBloodDoseResult:
    """Pathwise dwell score with explicit solver/capacity evidence.

    Dose-rate standard uncertainty is conditional on each realized dwell history;
    stochastic path dispersion is intentionally kept separate.
    """

    solution: JumpSolution
    occupation_seconds: Array
    compartment_dose_gy: Array
    total_dose_gy: Array
    standard_uncertainty_gy: Array | None
    capacity: HistoryCapacityEvidence
    model_id: str
    schedule_id: str

    @property
    def successful(self) -> Array:
        return self.capacity.successful


def _time_bounds(t0_s: float, t1_s: float, /) -> tuple[float, float]:
    start = float(t0_s)
    end = float(t1_s)
    if not math.isfinite(start) or not math.isfinite(end) or not end > start:
        raise ValueError("Dose scoring requires finite bounds with t1_s > t0_s.")
    return start, end


def _validate_schedule_model(
    prepared: PreparedCirculatingBloodModel,
    schedule: PiecewiseConstantDoseRateSchedule,
    /,
) -> int:
    if not isinstance(prepared, PreparedCirculatingBloodModel):
        raise TypeError("prepared must be a PreparedCirculatingBloodModel.")
    if not isinstance(schedule, PiecewiseConstantDoseRateSchedule):
        raise TypeError("schedule must be a PiecewiseConstantDoseRateSchedule.")
    count = len(prepared.compartment_ids)
    if schedule.compartment_count != count:
        raise ValueError(
            "Dose-rate compartment axis must match the prepared circulation model."
        )
    return count


def _advance_probability(
    matrix: Array,
    probabilities: Array,
    duration_s: float,
    /,
) -> tuple[Array, Array]:
    """Advance a row law and integrate its exact occupation by one block exponential."""

    count = int(matrix.shape[0])
    block = jnp.zeros((2 * count, 2 * count), dtype=matrix.dtype)
    block = block.at[:count, :count].set(matrix)
    block = block.at[:count, count:].set(jnp.eye(count, dtype=matrix.dtype))
    exponential = jsp.linalg.expm(jnp.asarray(duration_s, dtype=matrix.dtype) * block)
    transition = exponential[:count, :count]
    integrated_transition = exponential[:count, count:]
    return probabilities @ transition, probabilities @ integrated_transition


def integrate_circulating_blood_dose(
    prepared: PreparedCirculatingBloodModel,
    schedule: PiecewiseConstantDoseRateSchedule,
    initial_probabilities: ArrayLike,
    /,
    *,
    t0_s: float,
    t1_s: float,
) -> DeterministicBloodDoseResult:
    """Exactly integrate expected occupations and rewards over a fixed generator."""

    count = _validate_schedule_model(prepared, schedule)
    start, end = _time_bounds(t0_s, t1_s)
    initial_host = np.asarray(initial_probabilities, dtype=float)
    if initial_host.shape != (count,):
        raise ValueError(f"initial_probabilities must have shape {(count,)}.")
    tolerance = 128.0 * np.finfo(initial_host.dtype).eps
    if (
        np.any(~np.isfinite(initial_host))
        or np.any(initial_host < -tolerance)
        or not np.isclose(np.sum(initial_host), 1.0, atol=tolerance, rtol=0.0)
    ):
        raise ValueError(
            "initial_probabilities must be finite, nonnegative, and sum to one."
        )
    matrix = prepared.generator.matrix
    probabilities = jnp.asarray(initial_host, dtype=matrix.dtype)
    occupation = jnp.zeros((count,), dtype=matrix.dtype)
    dose = jnp.zeros((count,), dtype=matrix.dtype)
    variance = jnp.zeros((count,), dtype=matrix.dtype)
    uncertainty_known = True
    boundaries = {start, end}
    for interval in schedule.intervals:
        if start < interval.start_s < end:
            boundaries.add(interval.start_s)
        if start < interval.end_s < end:
            boundaries.add(interval.end_s)
    ordered = sorted(boundaries)
    for left, right in zip(ordered, ordered[1:], strict=False):
        probabilities, local_occupation = _advance_probability(
            matrix, probabilities, right - left
        )
        occupation = occupation + local_occupation
        active = next(
            (
                interval
                for interval in schedule.intervals
                if interval.start_s <= left and right <= interval.end_s
            ),
            None,
        )
        if active is None:
            continue
        rates = jnp.asarray(active.dose_rates_gy_per_s, dtype=matrix.dtype)
        contribution = local_occupation * rates
        dose = dose + contribution
        if active.standard_uncertainties_gy_per_s is None:
            uncertainty_known = False
        else:
            uncertainty = jnp.asarray(
                active.standard_uncertainties_gy_per_s,
                dtype=matrix.dtype,
            )
            variance = variance + jnp.square(local_occupation * uncertainty)
    standard_uncertainty = jnp.sqrt(jnp.sum(variance)) if uncertainty_known else None
    return DeterministicBloodDoseResult(
        jnp.asarray(initial_host, dtype=matrix.dtype),
        probabilities,
        occupation,
        dose,
        jnp.sum(dose),
        standard_uncertainty,
        start,
        end,
        prepared.model.model_id,
        schedule.schedule_id,
    )


def score_circulating_blood_histories(
    prepared: PreparedCirculatingBloodModel,
    schedule: PiecewiseConstantDoseRateSchedule,
    solution: JumpSolution,
    /,
    *,
    t0_s: float,
    t1_s: float,
) -> StochasticBloodDoseResult:
    """Score exact event dwell times from an existing pure-jump solution."""

    count = _validate_schedule_model(prepared, schedule)
    start, end = _time_bounds(t0_s, t1_s)
    if not isinstance(solution, JumpSolution):
        raise TypeError("solution must be a JumpSolution.")
    if solution.realization.process_id != prepared.process.process_id:
        raise ValueError("Jump solution belongs to a different circulation process.")
    if solution.state_shape != (1,):
        raise ValueError("Circulating-blood jump states must be scalar indices.")
    saved_times = np.asarray(jax.device_get(solution.times))
    time_tolerance = (
        16.0 * np.finfo(saved_times.dtype).eps * max(1.0, abs(start), abs(end))
    )
    if not np.allclose(saved_times[[0, -1]], (start, end), rtol=0.0, atol=time_tolerance):
        raise ValueError("Jump solution must save both requested scoring boundaries.")
    events = solution.events
    if events.pre_states is None or events.post_states is None:
        raise ValueError("Exact dwell scoring requires event pre/post state evidence.")
    safe_event_times = jnp.where(events.valid, events.times, end)
    starts = jnp.concatenate(
        (
            jnp.full(safe_event_times.shape[:-1] + (1,), start),
            safe_event_times,
        ),
        axis=-1,
    )
    ends = jnp.concatenate(
        (
            safe_event_times,
            jnp.full(safe_event_times.shape[:-1] + (1,), end),
        ),
        axis=-1,
    )
    terminal_state = solution.states[..., -1, 0].astype(jnp.int32)
    event_states = jnp.where(
        events.valid,
        events.pre_states[..., 0].astype(jnp.int32),
        terminal_state[..., None],
    )
    segment_states = jnp.concatenate((event_states, terminal_state[..., None]), axis=-1)
    one_hot = jax.nn.one_hot(segment_states, count, dtype=solution.states.dtype)
    duration = jnp.maximum(ends - starts, 0.0)
    occupation = jnp.sum(duration[..., None] * one_hot, axis=-2)
    dose = jnp.zeros_like(occupation)
    variance = jnp.zeros_like(occupation)
    uncertainty_known = True
    for interval in schedule.intervals:
        overlap = jnp.maximum(
            jnp.minimum(ends, interval.end_s) - jnp.maximum(starts, interval.start_s),
            0.0,
        )
        dwell = jnp.sum(overlap[..., None] * one_hot, axis=-2)
        rates = jnp.asarray(interval.dose_rates_gy_per_s, dtype=dose.dtype)
        dose = dose + dwell * rates
        if max(start, interval.start_s) < min(end, interval.end_s):
            if interval.standard_uncertainties_gy_per_s is None:
                uncertainty_known = False
            else:
                uncertainty = jnp.asarray(
                    interval.standard_uncertainties_gy_per_s,
                    dtype=dose.dtype,
                )
                variance = variance + jnp.square(dwell * uncertainty)
    successful = solution.successful
    occupation = jnp.where(successful[..., None], occupation, jnp.nan)
    dose = jnp.where(successful[..., None], dose, jnp.nan)
    total = jnp.sum(dose, axis=-1)
    standard_uncertainty = (
        jnp.where(successful, jnp.sqrt(jnp.sum(variance, axis=-1)), jnp.nan)
        if uncertainty_known
        else None
    )
    capacity = HistoryCapacityEvidence(
        events.max_events,
        events.counts,
        events.status == JUMP_MAX_EVENTS,
        successful,
    )
    return StochasticBloodDoseResult(
        solution,
        occupation,
        dose,
        total,
        standard_uncertainty,
        capacity,
        prepared.model.model_id,
        schedule.schedule_id,
    )


def simulate_circulating_blood_dose(
    prepared: PreparedCirculatingBloodModel,
    schedule: PiecewiseConstantDoseRateSchedule,
    realization: PoissonClockRealization,
    initial_compartment_id: str,
    /,
    *,
    t0_s: float,
    t1_s: float,
    max_events: int | None = None,
    lane_count: int | None = None,
) -> StochasticBloodDoseResult:
    """Run direct SSA and exactly score its realized compartment dwell history."""

    _validate_schedule_model(prepared, schedule)
    if not isinstance(realization, PoissonClockRealization):
        raise TypeError("realization must be a PoissonClockRealization.")
    initial = prepared.encode(
        _identifier(initial_compartment_id, "initial_compartment_id")
    )
    start, end = _time_bounds(t0_s, t1_s)
    solution = solve_direct_ssa(
        prepared.process,
        realization,
        initial,
        t0=start,
        t1=end,
        save_times=jnp.asarray((start, end)),
        max_events=max_events,
        lane_count=lane_count,
    )
    return score_circulating_blood_histories(
        prepared,
        schedule,
        solution,
        t0_s=start,
        t1_s=end,
    )


__all__ = [
    "ABSORBED_DOSE_RATE_REFERENCE",
    "CIRCULATING_BLOOD_DOSE_RATE_REFERENCES",
    "CIRCULATING_BLOOD_DOSE_RATE_SUPPORT",
    "DOSE_TO_MEDIUM_RATE_REFERENCE",
    "DOSE_TO_WATER_RATE_REFERENCE",
    "DeterministicBloodDoseResult",
    "DoseRateInterval",
    "HistoryCapacityEvidence",
    "PiecewiseConstantDoseRateSchedule",
    "StochasticBloodDoseResult",
    "circulating_blood_dose_rate_quantity",
    "integrate_circulating_blood_dose",
    "score_circulating_blood_histories",
    "simulate_circulating_blood_dose",
]
