#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Prepared pressure, volume, flow, and pressure-volume-loop observations."""

from __future__ import annotations

import math

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
from .._quantities import cardiovascular_quantity, CardiovascularQuantitySpec
from ._electrograms import _timebase_evidence, TimeBaseEvidence
from ._metadata import ObservationRecord, TimeBase


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


def _finite_nonnegative(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _matrix(
    values: ArrayLike, target_count: int, source_count: int, name: str, /
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (target_count, source_count) or np.any(~np.isfinite(matrix)):
        raise ValueError(
            f"{name} must be a finite ({target_count}, {source_count}) matrix."
        )
    return matrix


def _trace(values: ArrayLike, timebase: TimeBase, width: int, name: str, /) -> Array:
    result = jnp.asarray(values)
    if result.shape != (timebase.sample_count, width):
        raise ValueError(f"{name} must have shape ({timebase.sample_count}, {width}).")
    return result


def _record_values(
    record: ObservationRecord,
    timebase: TimeBase,
    width: int,
    *,
    modality: str,
    quantity: CardiovascularQuantitySpec,
) -> tuple[Array, Array]:
    if record.modality != modality:
        raise ValueError(f"Observation record modality must be {modality!r}.")
    if record.quantity != quantity.name or record.unit != quantity.kernel_unit:
        raise ValueError(
            f"Observation record must carry {quantity.name!r} in {quantity.kernel_unit!r}."
        )
    if record.timebase_id != timebase.timebase_id:
        raise ValueError("Observation record and plan time bases differ.")
    values = _trace(record.values, timebase, width, "Observation record values")
    valid = jnp.asarray(record.valid_mask, dtype=bool)
    if valid.shape != values.shape:
        raise ValueError("Observation record validity mask must match its values.")
    return values, valid


class HemodynamicObservationEvidence(StrictModule):
    """Fixed response, timebase, mask, and finite-value evidence."""

    timebase: TimeBaseEvidence
    input_valid_fraction: Array
    finite: Array
    successful: Array


class PressureTraceResult(StrictModule):
    """Gauge-referenced pressure traces in kilopascals."""

    pressure_kpa: Array
    timebase: TimeBase
    channel_labels: tuple[str, ...] = eqx.field(static=True)
    reference_configuration: str = eqx.field(static=True)
    evidence: HemodynamicObservationEvidence
    plan_id: str = eqx.field(static=True)


class PressureObservationPlan(StrictModule, NonTrainableState):
    """Fixed labelled pressure response with an explicit target gauge reference."""

    response: LinearObservationPlan
    reference_pressure_kpa: Array
    timebase: TimeBase
    quantity: CardiovascularQuantitySpec = eqx.field(static=True)
    source_labels: tuple[str, ...] = eqx.field(static=True)
    channel_labels: tuple[str, ...] = eqx.field(static=True)
    reference_configuration: str = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        response_matrix: ArrayLike,
        source_labels: tuple[str, ...],
        channel_labels: tuple[str, ...],
        reference_pressure_kpa: ArrayLike,
        timebase: TimeBase,
        /,
        *,
        reference_configuration: str,
        observation_id: str,
    ):
        sources = _labels(source_labels, "source_labels")
        channels = _labels(channel_labels, "channel_labels")
        response = _matrix(
            response_matrix, len(channels), len(sources), "Pressure response"
        )
        reference = np.asarray(reference_pressure_kpa, dtype=float)
        if reference.shape == ():
            reference = np.full((len(channels),), float(reference))
        if reference.shape != (len(channels),) or np.any(~np.isfinite(reference)):
            raise ValueError(
                "Reference pressure must be finite and scalar or per-channel."
            )
        configuration = _identifier(reference_configuration, "reference_configuration")
        identifier = _identifier(observation_id, "observation_id")
        quantity = cardiovascular_quantity("pressure")
        self.response = LinearObservationPlan(
            response, CoordinateLayout(sources), CoordinateLayout(channels)
        )
        self.reference_pressure_kpa = jax.lax.stop_gradient(jnp.asarray(reference))
        self.timebase = timebase
        self.quantity = quantity
        self.source_labels = sources
        self.channel_labels = channels
        self.reference_configuration = configuration
        self.observation_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-pressure-observation-plan",
                "observation_id": identifier,
                "response": self.response.plan_id,
                "reference_pressure_kpa": array_tree_fingerprint(
                    self.reference_pressure_kpa
                ),
                "reference_configuration": configuration,
                "timebase": timebase.timebase_id,
                "sample_times_ms": array_tree_fingerprint(timebase.sample_times_ms),
                "quantity": quantity.quantity_id,
            }
        )

    def observe(self, absolute_pressure_kpa: ArrayLike, /) -> PressureTraceResult:
        values = _trace(
            absolute_pressure_kpa,
            self.timebase,
            len(self.source_labels),
            "Absolute pressure",
        )
        return self._observe(values, jnp.ones_like(values, dtype=bool))

    def from_record(self, record: ObservationRecord, /) -> PressureTraceResult:
        values, valid = _record_values(
            record,
            self.timebase,
            len(self.source_labels),
            modality="pressure",
            quantity=self.quantity,
        )
        return self._observe(values, valid)

    def _observe(self, values: Array, valid: Array, /) -> PressureTraceResult:
        predicted = contract("oi,ti->to", self.response.matrix, values)
        referenced = predicted - self.reference_pressure_kpa[None, :]
        time_evidence = _timebase_evidence(self.timebase)
        finite = jnp.all(jnp.where(valid, jnp.isfinite(values), True)) & jnp.all(
            jnp.isfinite(referenced)
        )
        valid_fraction = jnp.mean(valid)
        successful = finite & jnp.all(valid) & time_evidence.successful
        output = jnp.where(successful, referenced, jnp.zeros_like(referenced))
        return PressureTraceResult(
            output,
            self.timebase,
            self.channel_labels,
            self.reference_configuration,
            HemodynamicObservationEvidence(
                time_evidence, valid_fraction, finite, successful
            ),
            self.plan_id,
        )


class VolumeTraceResult(StrictModule):
    """Observed chamber or control-volume traces in cubic millimetres."""

    volume_mm3: Array
    timebase: TimeBase
    channel_labels: tuple[str, ...] = eqx.field(static=True)
    evidence: HemodynamicObservationEvidence
    plan_id: str = eqx.field(static=True)


class VolumeObservationPlan(StrictModule, NonTrainableState):
    """Fixed labelled linear response for chamber/control-volume observations."""

    response: LinearObservationPlan
    timebase: TimeBase
    quantity: CardiovascularQuantitySpec = eqx.field(static=True)
    source_labels: tuple[str, ...] = eqx.field(static=True)
    channel_labels: tuple[str, ...] = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        response_matrix: ArrayLike,
        source_labels: tuple[str, ...],
        channel_labels: tuple[str, ...],
        timebase: TimeBase,
        /,
        *,
        observation_id: str,
    ):
        sources = _labels(source_labels, "source_labels")
        channels = _labels(channel_labels, "channel_labels")
        response = _matrix(
            response_matrix, len(channels), len(sources), "Volume response"
        )
        identifier = _identifier(observation_id, "observation_id")
        quantity = cardiovascular_quantity("volume")
        self.response = LinearObservationPlan(
            response, CoordinateLayout(sources), CoordinateLayout(channels)
        )
        self.timebase = timebase
        self.quantity = quantity
        self.source_labels = sources
        self.channel_labels = channels
        self.observation_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-volume-observation-plan",
                "observation_id": identifier,
                "response": self.response.plan_id,
                "timebase": timebase.timebase_id,
                "sample_times_ms": array_tree_fingerprint(timebase.sample_times_ms),
                "quantity": quantity.quantity_id,
            }
        )

    def observe(self, volume_mm3: ArrayLike, /) -> VolumeTraceResult:
        values = _trace(volume_mm3, self.timebase, len(self.source_labels), "Volume")
        return self._observe(values, jnp.ones_like(values, dtype=bool))

    def from_record(self, record: ObservationRecord, /) -> VolumeTraceResult:
        values, valid = _record_values(
            record,
            self.timebase,
            len(self.source_labels),
            modality="volume",
            quantity=self.quantity,
        )
        return self._observe(values, valid)

    def _observe(self, values: Array, valid: Array, /) -> VolumeTraceResult:
        predicted = contract("oi,ti->to", self.response.matrix, values)
        time_evidence = _timebase_evidence(self.timebase)
        finite = jnp.all(jnp.where(valid, jnp.isfinite(values), True)) & jnp.all(
            jnp.isfinite(predicted)
        )
        valid_fraction = jnp.mean(valid)
        successful = finite & jnp.all(valid) & time_evidence.successful
        output = jnp.where(successful, predicted, jnp.zeros_like(predicted))
        return VolumeTraceResult(
            output,
            self.timebase,
            self.channel_labels,
            HemodynamicObservationEvidence(
                time_evidence, valid_fraction, finite, successful
            ),
            self.plan_id,
        )


class FlowTraceResult(StrictModule):
    """Oriented volumetric-flow traces in cubic millimetres per millisecond."""

    flow_mm3_per_ms: Array
    timebase: TimeBase
    channel_labels: tuple[str, ...] = eqx.field(static=True)
    positive_directions: tuple[str, ...] = eqx.field(static=True)
    evidence: HemodynamicObservationEvidence
    plan_id: str = eqx.field(static=True)


class FlowObservationPlan(StrictModule, NonTrainableState):
    """Fixed labelled flow response with explicit per-port orientation signs."""

    response: LinearObservationPlan
    orientation_signs: Array
    timebase: TimeBase
    quantity: CardiovascularQuantitySpec = eqx.field(static=True)
    source_labels: tuple[str, ...] = eqx.field(static=True)
    channel_labels: tuple[str, ...] = eqx.field(static=True)
    positive_directions: tuple[str, ...] = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        response_matrix: ArrayLike,
        source_labels: tuple[str, ...],
        channel_labels: tuple[str, ...],
        orientation_signs: ArrayLike,
        positive_directions: tuple[str, ...],
        timebase: TimeBase,
        /,
        *,
        observation_id: str,
    ):
        sources = _labels(source_labels, "source_labels")
        channels = _labels(channel_labels, "channel_labels")
        directions = tuple(
            _identifier(value, "positive direction") for value in positive_directions
        )
        if len(directions) != len(channels):
            raise ValueError("One positive flow direction is required per channel.")
        response = _matrix(response_matrix, len(channels), len(sources), "Flow response")
        signs = np.asarray(orientation_signs, dtype=float)
        if signs.shape != (len(channels),) or np.any(np.abs(signs) != 1.0):
            raise ValueError(
                "Flow orientation signs must be exactly +1 or -1 per channel."
            )
        identifier = _identifier(observation_id, "observation_id")
        quantity = cardiovascular_quantity("volumetric_flow_rate")
        self.response = LinearObservationPlan(
            response, CoordinateLayout(sources), CoordinateLayout(channels)
        )
        self.orientation_signs = jax.lax.stop_gradient(jnp.asarray(signs))
        self.timebase = timebase
        self.quantity = quantity
        self.source_labels = sources
        self.channel_labels = channels
        self.positive_directions = directions
        self.observation_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-flow-observation-plan",
                "observation_id": identifier,
                "response": self.response.plan_id,
                "orientation_signs": array_tree_fingerprint(self.orientation_signs),
                "positive_directions": list(directions),
                "timebase": timebase.timebase_id,
                "sample_times_ms": array_tree_fingerprint(timebase.sample_times_ms),
                "quantity": quantity.quantity_id,
            }
        )

    def observe(self, flow_mm3_per_ms: ArrayLike, /) -> FlowTraceResult:
        values = _trace(flow_mm3_per_ms, self.timebase, len(self.source_labels), "Flow")
        return self._observe(values, jnp.ones_like(values, dtype=bool))

    def from_record(self, record: ObservationRecord, /) -> FlowTraceResult:
        values, valid = _record_values(
            record,
            self.timebase,
            len(self.source_labels),
            modality="flow",
            quantity=self.quantity,
        )
        return self._observe(values, valid)

    def _observe(self, values: Array, valid: Array, /) -> FlowTraceResult:
        predicted = contract("oi,ti->to", self.response.matrix, values)
        oriented = predicted * self.orientation_signs[None, :]
        time_evidence = _timebase_evidence(self.timebase)
        finite = jnp.all(jnp.where(valid, jnp.isfinite(values), True)) & jnp.all(
            jnp.isfinite(oriented)
        )
        valid_fraction = jnp.mean(valid)
        successful = finite & jnp.all(valid) & time_evidence.successful
        output = jnp.where(successful, oriented, jnp.zeros_like(oriented))
        return FlowTraceResult(
            output,
            self.timebase,
            self.channel_labels,
            self.positive_directions,
            HemodynamicObservationEvidence(
                time_evidence, valid_fraction, finite, successful
            ),
            self.plan_id,
        )


class PressureVolumeLoopEvidence(StrictModule):
    """Closure, timing, finiteness, orientation, and reference evidence."""

    timebase: TimeBaseEvidence
    pressure_closure_error_kpa: Array
    volume_closure_error_mm3: Array
    closed: Array
    counterclockwise: Array
    finite: Array
    successful: Array


class PressureVolumeLoopResult(StrictModule):
    """Pressure-volume line integral and positive external stroke work."""

    pressure_relative_kpa: Array
    volume_mm3: Array
    line_integral_kpa_mm3: Array
    external_work_mg_mm2_per_ms2: Array
    external_work_mj: Array
    stroke_volume_mm3: Array
    evidence: PressureVolumeLoopEvidence
    timebase: TimeBase
    plan_id: str = eqx.field(static=True)


class PressureVolumeLoopPlan(StrictModule, NonTrainableState):
    """Closed-loop work with positive sign for work done by the chamber.

    The physiological external-work convention is ``-∮ p dV``.  Numerically,
    ``1 kPa mm³ = 1 mg mm²/ms² = 10⁻³ mJ`` exactly under the cardiovascular
    kernel scale.
    """

    timebase: TimeBase
    pressure_reference_kpa: float = eqx.field(static=True)
    reference_configuration: str = eqx.field(static=True)
    pressure_closure_tolerance_kpa: float = eqx.field(static=True)
    volume_closure_tolerance_mm3: float = eqx.field(static=True)
    loop_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        timebase: TimeBase,
        /,
        *,
        pressure_reference_kpa: float,
        reference_configuration: str,
        pressure_closure_tolerance_kpa: float = 1.0e-6,
        volume_closure_tolerance_mm3: float = 1.0e-3,
        loop_id: str,
    ):
        if timebase.sample_count < 3:
            raise ValueError("A pressure-volume loop requires at least three samples.")
        reference = float(pressure_reference_kpa)
        if not math.isfinite(reference):
            raise ValueError("pressure_reference_kpa must be finite.")
        pressure_tolerance = _finite_nonnegative(
            pressure_closure_tolerance_kpa, "pressure_closure_tolerance_kpa"
        )
        volume_tolerance = _finite_nonnegative(
            volume_closure_tolerance_mm3, "volume_closure_tolerance_mm3"
        )
        configuration = _identifier(reference_configuration, "reference_configuration")
        identifier = _identifier(loop_id, "loop_id")
        self.timebase = timebase
        self.pressure_reference_kpa = reference
        self.reference_configuration = configuration
        self.pressure_closure_tolerance_kpa = pressure_tolerance
        self.volume_closure_tolerance_mm3 = volume_tolerance
        self.loop_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-pressure-volume-loop-plan",
                "loop_id": identifier,
                "timebase": timebase.timebase_id,
                "sample_times_ms": array_tree_fingerprint(timebase.sample_times_ms),
                "pressure_reference_kpa": reference,
                "reference_configuration": configuration,
                "pressure_closure_tolerance_kpa": pressure_tolerance,
                "volume_closure_tolerance_mm3": volume_tolerance,
                "pressure_quantity": cardiovascular_quantity("pressure").quantity_id,
                "volume_quantity": cardiovascular_quantity("volume").quantity_id,
                "energy_quantity": cardiovascular_quantity("energy").quantity_id,
            }
        )

    def evaluate(
        self, pressure_kpa: ArrayLike, volume_mm3: ArrayLike, /
    ) -> PressureVolumeLoopResult:
        pressure = jnp.asarray(pressure_kpa)
        volume = jnp.asarray(volume_mm3, dtype=pressure.dtype)
        expected = (self.timebase.sample_count,)
        if pressure.shape != expected or volume.shape != expected:
            raise ValueError(
                "Pressure and volume must be vectors matching the loop time base."
            )
        relative_pressure = pressure - self.pressure_reference_kpa
        delta_volume = volume[1:] - volume[:-1]
        segment_pressure = 0.5 * (relative_pressure[1:] + relative_pressure[:-1])
        line_integral = jnp.sum(segment_pressure * delta_volume)
        external_work = -line_integral
        pressure_closure_error = jnp.abs(pressure[-1] - pressure[0])
        volume_closure_error = jnp.abs(volume[-1] - volume[0])
        closed = (pressure_closure_error <= self.pressure_closure_tolerance_kpa) & (
            volume_closure_error <= self.volume_closure_tolerance_mm3
        )
        finite = jnp.all(jnp.isfinite(pressure)) & jnp.all(jnp.isfinite(volume))
        time_evidence = _timebase_evidence(self.timebase)
        counterclockwise = line_integral < 0.0
        successful = finite & closed & time_evidence.successful & counterclockwise
        safe_line_integral = jnp.where(successful, line_integral, 0.0)
        safe_external_work = jnp.where(successful, external_work, 0.0)
        stroke_volume = jnp.where(successful, jnp.max(volume) - jnp.min(volume), 0.0)
        evidence = PressureVolumeLoopEvidence(
            time_evidence,
            pressure_closure_error,
            volume_closure_error,
            closed,
            counterclockwise,
            finite,
            successful,
        )
        return PressureVolumeLoopResult(
            jnp.where(successful, relative_pressure, jnp.zeros_like(relative_pressure)),
            jnp.where(successful, volume, jnp.zeros_like(volume)),
            safe_line_integral,
            safe_external_work,
            safe_external_work * 1.0e-3,
            stroke_volume,
            evidence,
            self.timebase,
            self.plan_id,
        )

    def from_records(
        self,
        pressure: ObservationRecord,
        volume: ObservationRecord,
        /,
    ) -> PressureVolumeLoopResult:
        """Evaluate scalar pressure/volume records on the declared time base."""

        if (
            pressure.modality != "pressure"
            or pressure.quantity != "pressure"
            or pressure.unit != "kPa"
        ):
            raise ValueError("PV pressure records must be pressure/pressure/kPa.")
        if (
            volume.modality != "volume"
            or volume.quantity != "volume"
            or volume.unit != "mm3"
        ):
            raise ValueError("PV volume records must be volume/volume/mm3.")
        if (
            pressure.timebase_id != self.timebase.timebase_id
            or volume.timebase_id != self.timebase.timebase_id
        ):
            raise ValueError("PV records and plan time bases must match.")
        if not np.all(pressure.valid_mask) or not np.all(volume.valid_mask):
            raise ValueError("PV loop records require complete valid support.")
        return self.evaluate(pressure.values, volume.values)


__all__ = [
    "FlowObservationPlan",
    "FlowTraceResult",
    "HemodynamicObservationEvidence",
    "PressureObservationPlan",
    "PressureTraceResult",
    "PressureVolumeLoopEvidence",
    "PressureVolumeLoopPlan",
    "PressureVolumeLoopResult",
    "VolumeObservationPlan",
    "VolumeTraceResult",
]
