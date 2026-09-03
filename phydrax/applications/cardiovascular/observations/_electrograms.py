#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Prepared cardiovascular electrical observation operators.

Transmembrane-voltage timing, extracellular electrograms, torso potentials, and
ECG leads are deliberately separate contracts.  In particular,
:class:`ElectrogramPlan` accepts only an explicitly labelled extracellular source
density; sampled transmembrane voltage is never reinterpreted as an electrogram.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....observation import CoordinateLayout, LinearObservationPlan
from ._metadata import ObservationRecord, TimeBase


if TYPE_CHECKING:
    from ..electrophysiology._activation import ActivationObservationResult


def _labels(values: tuple[str, ...], name: str, /) -> tuple[str, ...]:
    labels = tuple(str(value).strip() for value in values)
    if (
        not labels
        or any(not value for value in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError(f"{name} must contain unique non-empty labels.")
    return labels


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def _finite_scalar(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _floating_array(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return jnp.asarray(array, dtype=jnp.result_type(array.dtype, jnp.float32))


def _require_temporal_capacity(timebase: TimeBase, /) -> None:
    if timebase.sample_count < 2:
        raise ValueError(
            "Electrical observation time bases require at least two samples."
        )


def _same_timebase(left: TimeBase, right: TimeBase, /) -> bool:
    return left.timebase_id == right.timebase_id and np.array_equal(
        left.sample_times_ms, right.sample_times_ms
    )


def _timebase_evidence(timebase: TimeBase, /) -> TimeBaseEvidence:
    times = jnp.asarray(timebase.sample_times_ms)
    finite = jnp.all(jnp.isfinite(times))
    if timebase.sample_count == 1:
        zero = jnp.asarray(0.0, dtype=times.dtype)
        truth = jnp.asarray(True)
        return TimeBaseEvidence(
            zero,
            zero,
            jnp.asarray(False),
            finite,
            truth,
            truth,
            finite,
            timebase.timebase_id,
        )
    intervals = jnp.diff(times)
    mean_interval = jnp.mean(intervals)
    maximum_deviation = jnp.max(jnp.abs(intervals - mean_interval))
    increasing = jnp.all(intervals > 0.0)
    tolerance = (
        64.0
        * jnp.finfo(times.dtype).eps
        * jnp.maximum(jnp.abs(mean_interval), jnp.asarray(1.0, dtype=times.dtype))
    )
    uniform = increasing & (maximum_deviation <= tolerance)
    return TimeBaseEvidence(
        mean_interval,
        maximum_deviation,
        jnp.asarray(True),
        finite,
        increasing,
        uniform,
        finite & increasing,
        timebase.timebase_id,
    )


def _trace_matrix(
    values: ArrayLike, timebase: TimeBase, width: int, name: str, /
) -> Array:
    trace = _floating_array(values)
    if trace.shape != (timebase.sample_count, width):
        raise ValueError(f"{name} must have shape ({timebase.sample_count}, {width}).")
    return trace


class TimeBaseEvidence(StrictModule):
    """Numerical evidence for a centralized cardiovascular time base."""

    interval_ms: Array
    maximum_interval_deviation_ms: Array
    has_interval: Array
    finite: Array
    strictly_increasing: Array
    uniform: Array
    successful: Array
    timebase_id: str = eqx.field(static=True)


class ActivationTimingEvidence(StrictModule):
    """Per-site occurrence, censoring, and crossing-identifiability evidence."""

    occurred: Array
    censored: Array
    ambiguous: Array
    finite: Array
    bracket_width_ms: Array
    successful: Array


class ActivationTimeResult(StrictModule):
    """Fixed-shape local activation-time observation in milliseconds."""

    activation_time_ms: Array
    evidence: ActivationTimingEvidence
    timebase: TimeBase
    plan_id: str = eqx.field(static=True)


class ActivationTimePlan(StrictModule, NonTrainableState):
    """First rising threshold crossing with explicit right-censoring.

    The selected crossing bracket is a fixed-topology, piecewise-smooth map.  A
    second rising crossing is retained as an ambiguous observation rather than
    silently choosing a beat.
    """

    timebase: TimeBase
    threshold_mv: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, timebase: TimeBase, /, *, threshold_mv: float):
        _require_temporal_capacity(timebase)
        threshold = _finite_scalar(threshold_mv, "threshold_mv")
        self.timebase = timebase
        self.threshold_mv = threshold
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-activation-time-plan",
                "timebase": timebase.timebase_id,
                "sample_times_ms": array_tree_fingerprint(timebase.sample_times_ms),
                "threshold_mv": threshold,
            }
        )

    def evaluate(self, transmembrane_voltage_mv: ArrayLike, /) -> ActivationTimeResult:
        voltage = _floating_array(transmembrane_voltage_mv)
        if voltage.ndim != 2 or voltage.shape[0] != self.timebase.sample_count:
            raise ValueError(
                "Transmembrane voltage must have shape (timebase.sample_count, sites)."
            )
        times = jnp.asarray(self.timebase.sample_times_ms, dtype=voltage.dtype)
        crossings = (voltage[:-1] < self.threshold_mv) & (
            voltage[1:] >= self.threshold_mv
        )
        crossing_count = jnp.sum(crossings, axis=0)
        occurred = crossing_count > 0
        ambiguous = crossing_count > 1
        index = jnp.argmax(crossings, axis=0)
        sites = jnp.arange(voltage.shape[1])
        before = voltage[index, sites]
        after = voltage[index + 1, sites]
        denominator = jnp.where(after != before, after - before, 1.0)
        fraction = (self.threshold_mv - before) / denominator
        selected = times[index] + fraction * (times[index + 1] - times[index])
        finite = jnp.all(jnp.isfinite(voltage), axis=0)
        censored = finite & ~occurred
        successful = finite & occurred & ~ambiguous
        activation_time = jnp.where(occurred & finite, selected, jnp.nan)
        bracket_width = jnp.where(
            occurred & finite, times[index + 1] - times[index], jnp.nan
        )
        evidence = ActivationTimingEvidence(
            occurred,
            censored,
            ambiguous,
            finite,
            bracket_width,
            successful,
        )
        return ActivationTimeResult(
            activation_time, evidence, self.timebase, self.plan_id
        )

    def from_record(self, record: ObservationRecord, /) -> ActivationTimeResult:
        """Evaluate a normalized transmembrane-voltage observation record."""

        if (
            record.modality != "transmembrane-voltage"
            or record.quantity != "transmembrane_potential"
            or record.unit != "mV"
        ):
            raise ValueError(
                "Activation records must be transmembrane-voltage/"
                "transmembrane_potential/mV."
            )
        if record.timebase_id != self.timebase.timebase_id:
            raise ValueError("Activation record and plan time bases differ.")
        values = _floating_array(record.values)
        valid = jnp.asarray(record.valid_mask, dtype=bool)
        if valid.shape != values.shape:
            raise ValueError("Activation record validity mask must match its values.")
        return self.evaluate(jnp.where(valid, values, jnp.nan))

    def consume_activation_observation(
        self, result: ActivationObservationResult, /
    ) -> ActivationTimeResult:
        """Normalize a foundation activation observation without aliasing its type."""

        activation_time = jnp.asarray(result.activation_times_ms)
        occurred = jnp.asarray(result.activated, dtype=bool)
        node_ids = jnp.asarray(result.node_ids)
        status = jnp.asarray(result.status)
        observation_successful = jnp.asarray(result.successful, dtype=bool)
        if (
            activation_time.shape != occurred.shape
            or node_ids.shape != occurred.shape
            or status.shape != ()
            or observation_successful.shape != ()
        ):
            raise ValueError(
                "Activation observation node arrays must match and status must be scalar."
            )
        lifecycle_ok = observation_successful & (status == 0)
        finite = lifecycle_ok & jnp.where(occurred, jnp.isfinite(activation_time), True)
        censored = finite & ~occurred
        successful = finite & occurred
        bracket_width = jnp.full_like(activation_time, jnp.nan)
        evidence = ActivationTimingEvidence(
            occurred,
            censored,
            jnp.zeros_like(occurred),
            finite,
            bracket_width,
            successful,
        )
        return ActivationTimeResult(
            jnp.where(occurred & finite, activation_time, jnp.nan),
            evidence,
            self.timebase,
            self.plan_id,
        )


class ActionPotentialDurationEvidence(StrictModule):
    """Activation/repolarization occurrence and censoring evidence."""

    activation_occurred: Array
    repolarization_occurred: Array
    activation_censored: Array
    repolarization_censored: Array
    ambiguous_activation: Array
    positive_amplitude: Array
    finite: Array
    successful: Array


class ActionPotentialDurationResult(StrictModule):
    """Activation time, repolarization time, and APD in milliseconds."""

    activation_time_ms: Array
    repolarization_time_ms: Array
    duration_ms: Array
    repolarization_level_mv: Array
    evidence: ActionPotentialDurationEvidence
    timebase: TimeBase
    plan_id: str = eqx.field(static=True)


class ActionPotentialDurationPlan(StrictModule, NonTrainableState):
    """APD from a rising activation crossing and post-peak repolarization crossing."""

    timebase: TimeBase
    activation_threshold_mv: float = eqx.field(static=True)
    resting_potential_mv: float = eqx.field(static=True)
    repolarization_fraction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        timebase: TimeBase,
        /,
        *,
        activation_threshold_mv: float,
        resting_potential_mv: float,
        repolarization_fraction: float = 0.9,
    ):
        _require_temporal_capacity(timebase)
        threshold = _finite_scalar(activation_threshold_mv, "activation_threshold_mv")
        resting = _finite_scalar(resting_potential_mv, "resting_potential_mv")
        fraction = _finite_scalar(repolarization_fraction, "repolarization_fraction")
        if not 0.0 < fraction < 1.0:
            raise ValueError(
                "repolarization_fraction must lie strictly between zero and one."
            )
        self.timebase = timebase
        self.activation_threshold_mv = threshold
        self.resting_potential_mv = resting
        self.repolarization_fraction = fraction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-apd-plan",
                "timebase": timebase.timebase_id,
                "sample_times_ms": array_tree_fingerprint(timebase.sample_times_ms),
                "activation_threshold_mv": threshold,
                "resting_potential_mv": resting,
                "repolarization_fraction": fraction,
            }
        )

    def evaluate(
        self, transmembrane_voltage_mv: ArrayLike, /
    ) -> ActionPotentialDurationResult:
        voltage = _floating_array(transmembrane_voltage_mv)
        if voltage.ndim != 2 or voltage.shape[0] != self.timebase.sample_count:
            raise ValueError(
                "Transmembrane voltage must have shape (timebase.sample_count, sites)."
            )
        times = jnp.asarray(self.timebase.sample_times_ms, dtype=voltage.dtype)
        rising = (voltage[:-1] < self.activation_threshold_mv) & (
            voltage[1:] >= self.activation_threshold_mv
        )
        activation_count = jnp.sum(rising, axis=0)
        activation_occurred = activation_count > 0
        activation_index = jnp.argmax(rising, axis=0)
        sites = jnp.arange(voltage.shape[1])
        activation_before = voltage[activation_index, sites]
        activation_after = voltage[activation_index + 1, sites]
        activation_denominator = jnp.where(
            activation_after != activation_before,
            activation_after - activation_before,
            1.0,
        )
        activation_fraction = (
            self.activation_threshold_mv - activation_before
        ) / activation_denominator
        activation_time = times[activation_index] + activation_fraction * (
            times[activation_index + 1] - times[activation_index]
        )

        time_indices = jnp.arange(voltage.shape[0])[:, None]
        after_activation = time_indices >= (activation_index + 1)[None, :]
        peak_search = jnp.where(after_activation, voltage, -jnp.inf)
        peak_index = jnp.argmax(peak_search, axis=0)
        peak_voltage = voltage[peak_index, sites]
        positive_amplitude = peak_voltage > self.resting_potential_mv
        repolarization_level = peak_voltage - self.repolarization_fraction * (
            peak_voltage - self.resting_potential_mv
        )
        after_peak = jnp.arange(voltage.shape[0] - 1)[:, None] >= peak_index[None, :]
        falling = (
            after_peak
            & (voltage[:-1] > repolarization_level[None, :])
            & (voltage[1:] <= repolarization_level[None, :])
        )
        repolarization_count = jnp.sum(falling, axis=0)
        repolarization_occurred = repolarization_count > 0
        repolarization_index = jnp.argmax(falling, axis=0)
        repolarization_before = voltage[repolarization_index, sites]
        repolarization_after = voltage[repolarization_index + 1, sites]
        repolarization_denominator = jnp.where(
            repolarization_after != repolarization_before,
            repolarization_after - repolarization_before,
            1.0,
        )
        repolarization_fraction = (
            repolarization_level - repolarization_before
        ) / repolarization_denominator
        repolarization_time = times[repolarization_index] + repolarization_fraction * (
            times[repolarization_index + 1] - times[repolarization_index]
        )

        finite = jnp.all(jnp.isfinite(voltage), axis=0)
        ambiguous_activation = activation_count > 1
        activation_censored = finite & ~activation_occurred
        repolarization_censored = finite & activation_occurred & ~repolarization_occurred
        successful = (
            finite
            & activation_occurred
            & repolarization_occurred
            & ~ambiguous_activation
            & positive_amplitude
            & (repolarization_time >= activation_time)
        )
        activation_time = jnp.where(
            finite & activation_occurred, activation_time, jnp.nan
        )
        repolarization_time = jnp.where(
            finite & repolarization_occurred, repolarization_time, jnp.nan
        )
        duration = jnp.where(successful, repolarization_time - activation_time, jnp.nan)
        evidence = ActionPotentialDurationEvidence(
            activation_occurred,
            repolarization_occurred,
            activation_censored,
            repolarization_censored,
            ambiguous_activation,
            positive_amplitude,
            finite,
            successful,
        )
        return ActionPotentialDurationResult(
            activation_time,
            repolarization_time,
            duration,
            repolarization_level,
            evidence,
            self.timebase,
            self.plan_id,
        )

    def from_record(self, record: ObservationRecord, /) -> ActionPotentialDurationResult:
        """Evaluate a normalized transmembrane-voltage observation record."""

        if (
            record.modality != "transmembrane-voltage"
            or record.quantity != "transmembrane_potential"
            or record.unit != "mV"
        ):
            raise ValueError(
                "APD records must be transmembrane-voltage/transmembrane_potential/mV."
            )
        if record.timebase_id != self.timebase.timebase_id:
            raise ValueError("APD record and plan time bases differ.")
        values = _floating_array(record.values)
        valid = jnp.asarray(record.valid_mask, dtype=bool)
        if valid.shape != values.shape:
            raise ValueError("APD record validity mask must match its values.")
        return self.evaluate(jnp.where(valid, values, jnp.nan))


class ExtracellularSourceDensity(StrictModule):
    """Explicit extracellular source-density history consumed by EGM/ECG plans."""

    values: Array
    timebase: TimeBase
    source_labels: tuple[str, ...] = eqx.field(static=True)
    unit: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        timebase: TimeBase,
        source_labels: tuple[str, ...],
        /,
        *,
        unit: str,
        source_id: str,
    ):
        labels = _labels(source_labels, "source_labels")
        trace = _trace_matrix(
            values, timebase, len(labels), "Extracellular source density"
        )
        unit_ = _identifier(unit, "unit")
        if unit_ != "uA/mm2":
            raise ValueError(
                "Extracellular source density must use the kernel unit uA/mm2."
            )
        identifier = _identifier(source_id, "source_id")
        self.values = trace
        self.timebase = timebase
        self.source_labels = labels
        self.unit = unit_
        self.source_id = identifier

    @classmethod
    def from_record(
        cls,
        record: ObservationRecord,
        timebase: TimeBase,
        source_labels: tuple[str, ...],
        /,
    ) -> ExtracellularSourceDensity:
        """Admit a normalized membrane-current-density source record."""

        if (
            record.modality != "extracellular-source-density"
            or record.quantity != "membrane_current_density"
            or record.unit != "uA/mm2"
        ):
            raise ValueError(
                "EGM source records must be extracellular-source-density/"
                "membrane_current_density/uA/mm2."
            )
        if record.timebase_id != timebase.timebase_id:
            raise ValueError("EGM source record and time base differ.")
        if not np.all(record.valid_mask):
            raise ValueError("EGM source records must have complete valid support.")
        return cls(
            record.values,
            timebase,
            source_labels,
            unit=record.unit,
            source_id=record.record_id,
        )


class ElectricalGaugeEvidence(StrictModule):
    """Reference-gauge annihilation and finite-output evidence."""

    reference_weight_error: Array
    common_mode_residual: Array
    finite: Array
    successful: Array


class ElectricalGaugePlan(StrictModule, NonTrainableState):
    """Fixed linear reference gauge ``I - 1 wᵀ`` over named electrodes."""

    response: LinearObservationPlan
    reference_weights: Array
    electrode_labels: tuple[str, ...] = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)
    gauge_id: str = eqx.field(static=True)

    def __init__(
        self,
        electrode_labels: tuple[str, ...],
        reference_weights: ArrayLike,
        /,
        *,
        reference_id: str,
    ):
        labels = _labels(electrode_labels, "electrode_labels")
        weights_host = np.asarray(reference_weights, dtype=float)
        if weights_host.shape != (len(labels),) or np.any(~np.isfinite(weights_host)):
            raise ValueError("Reference weights must be one finite value per electrode.")
        if abs(float(np.sum(weights_host)) - 1.0) > 1.0e-10:
            raise ValueError("Reference weights must sum to one.")
        reference = _identifier(reference_id, "reference_id")
        matrix = np.eye(len(labels)) - np.ones((len(labels), 1)) * weights_host[None, :]
        layout = CoordinateLayout(labels)
        self.response = LinearObservationPlan(matrix, layout, layout)
        self.reference_weights = jax.lax.stop_gradient(jnp.asarray(weights_host))
        self.electrode_labels = labels
        self.reference_id = reference
        self.gauge_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-electrical-gauge",
                "reference_id": reference,
                "response": self.response.plan_id,
            }
        )

    def apply(
        self, electrode_potential_mv: ArrayLike, /
    ) -> tuple[Array, ElectricalGaugeEvidence]:
        values = _floating_array(electrode_potential_mv)
        if values.shape[-1:] != (len(self.electrode_labels),):
            raise ValueError(
                "Electrode-potential trailing axis does not match the gauge."
            )
        referenced = contract("oi,...i->...o", self.response.matrix, values)
        ones = jnp.ones((len(self.electrode_labels),), dtype=values.dtype)
        common_mode_residual = jnp.max(jnp.abs(self.response.matrix @ ones))
        reference_weight_error = jnp.abs(jnp.sum(self.reference_weights) - 1.0)
        finite = jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(referenced))
        tolerance = 64.0 * jnp.finfo(values.dtype).eps
        successful = (
            finite
            & (common_mode_residual <= tolerance)
            & (reference_weight_error <= tolerance)
        )
        return referenced, ElectricalGaugeEvidence(
            reference_weight_error, common_mode_residual, finite, successful
        )


class FilterEvidence(StrictModule):
    """Fixed FIR and timebase evidence."""

    dc_gain: Array
    finite: Array
    uniform_timebase: Array
    successful: Array


class FIRFilterPlan(StrictModule, NonTrainableState):
    """Fixed causal FIR filter with edge extension and an explicit DC gain."""

    coefficients: Array
    timebase: TimeBase
    filter_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coefficients: ArrayLike,
        timebase: TimeBase,
        /,
        *,
        filter_id: str,
    ):
        _require_temporal_capacity(timebase)
        host = np.asarray(coefficients, dtype=float)
        if host.ndim != 1 or host.size < 1 or np.any(~np.isfinite(host)):
            raise ValueError("FIR coefficients must be a non-empty finite vector.")
        identifier = _identifier(filter_id, "filter_id")
        self.coefficients = jax.lax.stop_gradient(jnp.asarray(host))
        self.timebase = timebase
        self.filter_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-fir-filter",
                "filter_id": identifier,
                "timebase": timebase.timebase_id,
                "sample_times_ms": array_tree_fingerprint(timebase.sample_times_ms),
                "coefficients": array_tree_fingerprint(self.coefficients),
            }
        )

    def apply(self, traces: ArrayLike, /) -> tuple[Array, FilterEvidence]:
        values = _floating_array(traces)
        if values.ndim != 2 or values.shape[0] != self.timebase.sample_count:
            raise ValueError(
                "Filtered traces must have shape (timebase.sample_count, channels)."
            )
        count = self.coefficients.size

        def filter_channel(channel: Array) -> Array:
            padded = jnp.pad(channel, (count - 1, 0), mode="edge")
            return jnp.convolve(padded, self.coefficients, mode="valid")

        filtered = jax.vmap(filter_channel, in_axes=1, out_axes=1)(values)
        time_evidence = _timebase_evidence(self.timebase)
        finite = jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(filtered))
        successful = finite & time_evidence.uniform
        return filtered, FilterEvidence(
            jnp.sum(self.coefficients), finite, time_evidence.uniform, successful
        )


class ElectrodeTransferEvidence(StrictModule):
    """Finite, responsive fixed-electrode transfer evidence."""

    source_count: Array
    electrode_count: Array
    finite: Array
    every_electrode_responsive: Array
    successful: Array


def _electrode_transfer_evidence(
    response: LinearObservationPlan, /
) -> ElectrodeTransferEvidence:
    matrix = response.matrix
    finite = jnp.all(jnp.isfinite(matrix))
    responsive = jnp.all(jnp.sum(jnp.abs(matrix), axis=1) > 0.0)
    return ElectrodeTransferEvidence(
        jnp.asarray(matrix.shape[1], dtype=jnp.int32),
        jnp.asarray(matrix.shape[0], dtype=jnp.int32),
        finite,
        responsive,
        finite & responsive,
    )


class ElectricalTraceEvidence(StrictModule):
    """Gauge, filter, timebase, and runtime evidence for an EGM trace."""

    timebase: TimeBaseEvidence
    electrode: ElectrodeTransferEvidence
    gauge: ElectricalGaugeEvidence
    filter: FilterEvidence
    finite_source: Array
    successful: Array


class ElectricalTraceResult(StrictModule):
    """Referenced and filtered electrogram values in millivolts."""

    values_mv: Array
    timebase: TimeBase
    channel_labels: tuple[str, ...] = eqx.field(static=True)
    evidence: ElectricalTraceEvidence
    plan_id: str = eqx.field(static=True)


class ElectrogramPlan(StrictModule, NonTrainableState):
    """Extracellular source-density to referenced electrode electrograms.

    ``observe`` intentionally rejects bare arrays.  Its input type makes the
    extracellular source semantics explicit and prevents sampled ``Vm`` from
    being relabelled as an EGM.
    """

    transfer: LinearObservationPlan
    gauge: ElectricalGaugePlan
    filter: FIRFilterPlan
    timebase: TimeBase
    source_labels: tuple[str, ...] = eqx.field(static=True)
    electrode_labels: tuple[str, ...] = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer_matrix: ArrayLike,
        source_labels: tuple[str, ...],
        gauge: ElectricalGaugePlan,
        filter_plan: FIRFilterPlan,
        timebase: TimeBase,
        /,
        *,
        transfer_id: str,
    ):
        sources = _labels(source_labels, "source_labels")
        if not _same_timebase(timebase, filter_plan.timebase):
            raise ValueError("Electrogram and filter time bases must match.")
        matrix = np.asarray(transfer_matrix, dtype=float)
        if matrix.shape != (len(gauge.electrode_labels), len(sources)):
            raise ValueError(
                "Electrogram transfer matrix shape does not match its layouts."
            )
        identifier = _identifier(transfer_id, "transfer_id")
        self.transfer = LinearObservationPlan(
            matrix, CoordinateLayout(sources), CoordinateLayout(gauge.electrode_labels)
        )
        self.gauge = gauge
        self.filter = filter_plan
        self.timebase = timebase
        self.source_labels = sources
        self.electrode_labels = gauge.electrode_labels
        self.transfer_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-electrogram-plan",
                "transfer_id": identifier,
                "transfer": self.transfer.plan_id,
                "gauge": gauge.gauge_id,
                "filter": filter_plan.plan_id,
                "timebase": timebase.timebase_id,
                "sample_times_ms": array_tree_fingerprint(timebase.sample_times_ms),
            }
        )

    def observe(self, source: ExtracellularSourceDensity, /) -> ElectricalTraceResult:
        if not isinstance(source, ExtracellularSourceDensity):
            raise TypeError(
                "ElectrogramPlan requires ExtracellularSourceDensity, not sampled Vm."
            )
        if not _same_timebase(source.timebase, self.timebase):
            raise ValueError("Extracellular source and electrogram time bases differ.")
        if source.source_labels != self.source_labels:
            raise ValueError(
                "Extracellular source labels do not match the transfer plan."
            )
        raw = contract("oi,ti->to", self.transfer.matrix, source.values)
        referenced, gauge_evidence = self.gauge.apply(raw)
        filtered, filter_evidence = self.filter.apply(referenced)
        time_evidence = _timebase_evidence(self.timebase)
        finite_source = jnp.all(jnp.isfinite(source.values))
        electrode_evidence = _electrode_transfer_evidence(self.transfer)
        successful = (
            finite_source
            & electrode_evidence.successful
            & gauge_evidence.successful
            & filter_evidence.successful
            & time_evidence.successful
        )
        output = jnp.where(successful, filtered, jnp.zeros_like(filtered))
        evidence = ElectricalTraceEvidence(
            time_evidence,
            electrode_evidence,
            gauge_evidence,
            filter_evidence,
            finite_source,
            successful,
        )
        return ElectricalTraceResult(
            output,
            self.timebase,
            self.electrode_labels,
            evidence,
            self.plan_id,
        )


class TorsoPotentialEvidence(StrictModule):
    """Torso transfer, reference gauge, and timebase evidence."""

    timebase: TimeBaseEvidence
    gauge: ElectricalGaugeEvidence
    electrode: ElectrodeTransferEvidence
    finite_source: Array
    successful: Array


class TorsoPotentialResult(StrictModule):
    """Referenced torso-electrode potentials in millivolts."""

    values_mv: Array
    timebase: TimeBase
    electrode_labels: tuple[str, ...] = eqx.field(static=True)
    evidence: TorsoPotentialEvidence
    plan_id: str = eqx.field(static=True)


class TorsoObservationPlan(StrictModule, NonTrainableState):
    """Fixed torso transfer followed by an explicit electrode-reference gauge."""

    transfer: LinearObservationPlan
    gauge: ElectricalGaugePlan
    timebase: TimeBase
    source_labels: tuple[str, ...] = eqx.field(static=True)
    electrode_labels: tuple[str, ...] = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer_matrix: ArrayLike,
        source_labels: tuple[str, ...],
        gauge: ElectricalGaugePlan,
        timebase: TimeBase,
        /,
        *,
        transfer_id: str,
    ):
        sources = _labels(source_labels, "source_labels")
        matrix = np.asarray(transfer_matrix, dtype=float)
        if matrix.shape != (len(gauge.electrode_labels), len(sources)):
            raise ValueError("Torso transfer matrix shape does not match its layouts.")
        identifier = _identifier(transfer_id, "transfer_id")
        self.transfer = LinearObservationPlan(
            matrix, CoordinateLayout(sources), CoordinateLayout(gauge.electrode_labels)
        )
        self.gauge = gauge
        self.timebase = timebase
        self.source_labels = sources
        self.electrode_labels = gauge.electrode_labels
        self.transfer_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-torso-observation-plan",
                "transfer_id": identifier,
                "transfer": self.transfer.plan_id,
                "gauge": gauge.gauge_id,
                "timebase": timebase.timebase_id,
                "sample_times_ms": array_tree_fingerprint(timebase.sample_times_ms),
            }
        )

    def observe(self, source: ExtracellularSourceDensity, /) -> TorsoPotentialResult:
        if not isinstance(source, ExtracellularSourceDensity):
            raise TypeError("TorsoObservationPlan requires ExtracellularSourceDensity.")
        if not _same_timebase(source.timebase, self.timebase):
            raise ValueError("Extracellular source and torso time bases differ.")
        if source.source_labels != self.source_labels:
            raise ValueError("Extracellular source labels do not match the torso plan.")
        raw = contract("oi,ti->to", self.transfer.matrix, source.values)
        referenced, gauge_evidence = self.gauge.apply(raw)
        time_evidence = _timebase_evidence(self.timebase)
        finite_source = jnp.all(jnp.isfinite(source.values))
        electrode_evidence = _electrode_transfer_evidence(self.transfer)
        successful = (
            finite_source
            & electrode_evidence.successful
            & gauge_evidence.successful
            & time_evidence.successful
        )
        values = jnp.where(successful, referenced, jnp.zeros_like(referenced))
        return TorsoPotentialResult(
            values,
            self.timebase,
            self.electrode_labels,
            TorsoPotentialEvidence(
                time_evidence,
                gauge_evidence,
                electrode_evidence,
                finite_source,
                successful,
            ),
            self.plan_id,
        )


class LeadFieldEvidence(StrictModule):
    """ECG lead reference invariance, reciprocity, filter, and finite evidence."""

    lead_reference_residual: Array
    reciprocity_residual: Array
    reciprocal: Array
    filter: FilterEvidence
    torso_electrodes: ElectrodeTransferEvidence
    timebase: TimeBaseEvidence
    finite_source: Array
    successful: Array


class ECGLeadResult(StrictModule):
    """Filtered ECG lead traces in millivolts."""

    values_mv: Array
    timebase: TimeBase
    lead_labels: tuple[str, ...] = eqx.field(static=True)
    evidence: LeadFieldEvidence
    plan_id: str = eqx.field(static=True)


class ECGLeadFieldPlan(StrictModule, NonTrainableState):
    """Torso-electrode lead map with an independently supplied reciprocal field."""

    torso: TorsoObservationPlan
    lead_response: LinearObservationPlan
    direct_source_response: LinearObservationPlan
    reciprocal_field: Array
    filter: FIRFilterPlan
    timebase: TimeBase
    lead_labels: tuple[str, ...] = eqx.field(static=True)
    reciprocity_tolerance: float = eqx.field(static=True)
    lead_field_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        torso: TorsoObservationPlan,
        lead_matrix: ArrayLike,
        lead_labels: tuple[str, ...],
        reciprocal_field: ArrayLike,
        filter_plan: FIRFilterPlan,
        /,
        *,
        lead_field_id: str,
        reciprocity_tolerance: float = 1.0e-8,
    ):
        labels = _labels(lead_labels, "lead_labels")
        matrix = np.asarray(lead_matrix, dtype=float)
        if matrix.shape != (len(labels), len(torso.electrode_labels)):
            raise ValueError("Lead matrix shape does not match lead/electrode layouts.")
        if np.any(~np.isfinite(matrix)):
            raise ValueError("Lead matrix must be finite.")
        direct = (
            matrix
            @ np.asarray(torso.gauge.response.matrix)
            @ np.asarray(torso.transfer.matrix)
        )
        reciprocal = np.asarray(reciprocal_field, dtype=float)
        if reciprocal.shape != (len(torso.source_labels), len(labels)):
            raise ValueError("Reciprocal field must have shape (sources, leads).")
        if np.any(~np.isfinite(reciprocal)):
            raise ValueError("Reciprocal field must be finite.")
        if not _same_timebase(filter_plan.timebase, torso.timebase):
            raise ValueError("ECG filter and torso time bases must match.")
        tolerance = _finite_scalar(reciprocity_tolerance, "reciprocity_tolerance")
        if tolerance < 0.0:
            raise ValueError("reciprocity_tolerance must be non-negative.")
        identifier = _identifier(lead_field_id, "lead_field_id")
        electrode_layout = CoordinateLayout(torso.electrode_labels)
        lead_layout = CoordinateLayout(labels)
        self.torso = torso
        self.lead_response = LinearObservationPlan(matrix, electrode_layout, lead_layout)
        self.direct_source_response = LinearObservationPlan(
            direct, CoordinateLayout(torso.source_labels), lead_layout
        )
        self.reciprocal_field = jax.lax.stop_gradient(jnp.asarray(reciprocal))
        self.filter = filter_plan
        self.timebase = torso.timebase
        self.lead_labels = labels
        self.reciprocity_tolerance = tolerance
        self.lead_field_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-ecg-lead-field-plan",
                "lead_field_id": identifier,
                "torso": torso.plan_id,
                "lead_response": self.lead_response.plan_id,
                "direct_source_response": self.direct_source_response.plan_id,
                "reciprocal": array_tree_fingerprint(self.reciprocal_field),
                "filter": filter_plan.plan_id,
                "reciprocity_tolerance": tolerance,
            }
        )

    def observe(self, source: ExtracellularSourceDensity, /) -> ECGLeadResult:
        if not isinstance(source, ExtracellularSourceDensity):
            raise TypeError("ECGLeadFieldPlan requires ExtracellularSourceDensity.")
        if not _same_timebase(source.timebase, self.timebase):
            raise ValueError("Extracellular source and ECG time bases differ.")
        if source.source_labels != self.torso.source_labels:
            raise ValueError("Extracellular source labels do not match the lead field.")
        direct = contract("oi,ti->to", self.direct_source_response.matrix, source.values)
        filtered, filter_evidence = self.filter.apply(direct)
        lead_reference_residual = jnp.max(
            jnp.abs(
                self.lead_response.matrix
                @ jnp.ones((len(self.torso.electrode_labels),), dtype=direct.dtype)
            )
        )
        reciprocity_residual = jnp.max(
            jnp.abs(self.direct_source_response.matrix.T - self.reciprocal_field)
        )
        reciprocal = reciprocity_residual <= self.reciprocity_tolerance
        time_evidence = _timebase_evidence(self.timebase)
        electrode_evidence = _electrode_transfer_evidence(self.torso.transfer)
        finite_source = jnp.all(jnp.isfinite(source.values))
        tolerance = 64.0 * jnp.finfo(direct.dtype).eps
        successful = (
            finite_source
            & electrode_evidence.successful
            & filter_evidence.successful
            & time_evidence.successful
            & (lead_reference_residual <= tolerance)
            & reciprocal
        )
        output = jnp.where(successful, filtered, jnp.zeros_like(filtered))
        evidence = LeadFieldEvidence(
            lead_reference_residual,
            reciprocity_residual,
            reciprocal,
            filter_evidence,
            electrode_evidence,
            time_evidence,
            finite_source,
            successful,
        )
        return ECGLeadResult(
            output, self.timebase, self.lead_labels, evidence, self.plan_id
        )


__all__ = [
    "ActionPotentialDurationEvidence",
    "ActionPotentialDurationPlan",
    "ActionPotentialDurationResult",
    "ActivationTimePlan",
    "ActivationTimeResult",
    "ActivationTimingEvidence",
    "ECGLeadFieldPlan",
    "ECGLeadResult",
    "ElectricalGaugeEvidence",
    "ElectrodeTransferEvidence",
    "ElectricalGaugePlan",
    "ElectricalTraceEvidence",
    "ElectricalTraceResult",
    "ElectrogramPlan",
    "ExtracellularSourceDensity",
    "FIRFilterPlan",
    "FilterEvidence",
    "LeadFieldEvidence",
    "TimeBaseEvidence",
    "TorsoObservationPlan",
    "TorsoPotentialEvidence",
    "TorsoPotentialResult",
]
