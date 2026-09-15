#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...units import ENERGY, LENGTH, TIME, UnitDefinition


def _identifier(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


class DetectorConditions(StrictModule, NonTrainableState):
    """Immutable detector geometry, field, material, and calibration conditions."""

    magnetic_field: Array
    electric_field: Array
    momentum_unit: UnitDefinition
    length_unit: UnitDefinition
    time_unit: UnitDefinition
    geometry_id: str = eqx.field(static=True)
    material_id: str = eqx.field(static=True)
    field_id: str = eqx.field(static=True)
    alignment_id: str = eqx.field(static=True)
    calibration_id: str = eqx.field(static=True)
    validity_start: int = eqx.field(static=True)
    validity_end: int = eqx.field(static=True)
    conditions_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        magnetic_field: ArrayLike,
        electric_field: ArrayLike,
        momentum_unit: UnitDefinition,
        length_unit: UnitDefinition,
        time_unit: UnitDefinition,
        geometry_id: str,
        material_id: str,
        field_id: str,
        alignment_id: str,
        calibration_id: str,
        validity_interval: tuple[int, int],
    ):
        magnetic = jnp.asarray(magnetic_field)
        electric = jnp.asarray(electric_field, dtype=magnetic.dtype)
        if magnetic.shape != (3,) or electric.shape != (3,):
            raise ValueError("Constant detector fields must contain three components.")
        if (
            not isinstance(momentum_unit, UnitDefinition)
            or momentum_unit.dimension != ENERGY
        ):
            raise ValueError(
                "momentum_unit must have energy dimension under natural units."
            )
        if not isinstance(length_unit, UnitDefinition) or length_unit.dimension != LENGTH:
            raise ValueError("length_unit must have length dimension.")
        if not isinstance(time_unit, UnitDefinition) or time_unit.dimension != TIME:
            raise ValueError("time_unit must have time dimension.")
        start, end = map(int, validity_interval)
        if end <= start:
            raise ValueError("validity_interval must be increasing.")
        identifiers = tuple(
            _identifier(value, name)
            for value, name in (
                (geometry_id, "Geometry ID"),
                (material_id, "Material ID"),
                (field_id, "Field ID"),
                (alignment_id, "Alignment ID"),
                (calibration_id, "Calibration ID"),
            )
        )
        self.magnetic_field = magnetic
        self.electric_field = electric
        self.momentum_unit = momentum_unit
        self.length_unit = length_unit
        self.time_unit = time_unit
        (
            self.geometry_id,
            self.material_id,
            self.field_id,
            self.alignment_id,
            self.calibration_id,
        ) = identifiers
        self.validity_start = start
        self.validity_end = end
        self.conditions_id = canonical_fingerprint(
            {
                "kind": "detector-conditions",
                "fields": [magnetic.tolist(), electric.tolist()],
                "units": [momentum_unit.unit_id, length_unit.unit_id, time_unit.unit_id],
                "identifiers": list(identifiers),
                "validity_interval": [start, end],
            }
        )

    def valid_for(self, run_number: ArrayLike, /) -> Array:
        value = jnp.asarray(run_number)
        return (value >= self.validity_start) & (value < self.validity_end)


class DetectorResourcePlan(StrictModule, NonTrainableState):
    event_capacity: int = eqx.field(static=True)
    track_capacity: int = eqx.field(static=True)
    step_capacity: int = eqx.field(static=True)
    hit_capacity: int = eqx.field(static=True)
    channel_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        event_capacity: int,
        track_capacity: int,
        step_capacity: int,
        hit_capacity: int,
        channel_capacity: int,
        maximum_storage_bytes: int = 1 << 30,
    ):
        capacities = tuple(
            map(
                int,
                (
                    event_capacity,
                    track_capacity,
                    step_capacity,
                    hit_capacity,
                    channel_capacity,
                ),
            )
        )
        if any(value < 1 for value in capacities):
            raise ValueError("Detector capacities must be positive.")
        estimated = capacities[0] * (
            capacities[1] * 128
            + capacities[2] * 160
            + capacities[3] * 96
            + capacities[4] * 64
        )
        if estimated > int(maximum_storage_bytes):
            raise ValueError("Detector capacities exceed maximum_storage_bytes.")
        (
            self.event_capacity,
            self.track_capacity,
            self.step_capacity,
            self.hit_capacity,
            self.channel_capacity,
        ) = capacities
        self.plan_id = canonical_fingerprint(
            {"kind": "detector-resource-plan", "capacities": list(capacities)}
        )


class TransportTrackBank(StrictModule, NonTrainableState):
    event_ids: Array
    track_ids: Array
    parent_track_ids: Array
    pdg_ids: Array
    positions: Array
    momenta: Array
    rest_energies: Array
    charges: Array
    active: Array
    valid: Array
    conditions_id: str = eqx.field(static=True)
    event_capacity: int = eqx.field(static=True)
    track_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        event_ids: ArrayLike,
        track_ids: ArrayLike,
        parent_track_ids: ArrayLike,
        pdg_ids: ArrayLike,
        positions: ArrayLike,
        momenta: ArrayLike,
        rest_energies: ArrayLike,
        charges: ArrayLike,
        active: ArrayLike,
        conditions_id: str,
    ):
        event_ids_ = jnp.asarray(event_ids)
        tracks = jnp.asarray(track_ids, dtype=jnp.int32)
        parents = jnp.asarray(parent_track_ids, dtype=jnp.int32)
        pdg = jnp.asarray(pdg_ids, dtype=jnp.int32)
        positions_ = jnp.asarray(positions)
        momenta_ = jnp.asarray(momenta, dtype=positions_.dtype)
        masses = jnp.asarray(rest_energies, dtype=positions_.dtype)
        charges_ = jnp.asarray(charges, dtype=positions_.dtype)
        active_ = jnp.asarray(active, dtype=bool)
        if event_ids_.ndim != 1 or tracks.ndim != 2:
            raise ValueError("Track banks require event IDs and an event-by-track grid.")
        expected = tracks.shape
        if (
            parents.shape != expected
            or pdg.shape != expected
            or masses.shape != expected
            or charges_.shape != expected
            or active_.shape != expected
        ):
            raise ValueError("Track scalar fields must share shape (event, track).")
        if positions_.shape != expected + (3,) or momenta_.shape != expected + (3,):
            raise ValueError(
                "Track positions and momenta require a trailing three-vector."
            )
        if event_ids_.shape != (expected[0],):
            raise ValueError("event_ids must align with the track event axis.")
        valid = (
            jnp.all(jnp.isfinite(positions_), axis=-1)
            & jnp.all(jnp.isfinite(momenta_), axis=-1)
            & jnp.isfinite(masses)
            & (masses > 0.0)
            & jnp.isfinite(charges_)
            & (tracks >= 0)
        )
        self.event_ids = event_ids_
        self.track_ids = tracks
        self.parent_track_ids = parents
        self.pdg_ids = pdg
        self.positions = positions_
        self.momenta = momenta_
        self.rest_energies = masses
        self.charges = charges_
        self.active = active_
        self.valid = jnp.where(active_, valid, True)
        self.conditions_id = _identifier(conditions_id, "Conditions ID")
        self.event_capacity, self.track_capacity = map(int, expected)


class TruthStepBank(StrictModule, NonTrainableState):
    event_ids: Array
    track_indices: Array
    detector_element_ids: Array
    start_positions: Array
    end_positions: Array
    start_times: Array
    end_times: Array
    deposited_energy: Array
    active: Array
    valid: Array
    conditions_id: str = eqx.field(static=True)
    step_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        event_ids: ArrayLike,
        track_indices: ArrayLike,
        detector_element_ids: ArrayLike,
        start_positions: ArrayLike,
        end_positions: ArrayLike,
        start_times: ArrayLike,
        end_times: ArrayLike,
        deposited_energy: ArrayLike,
        active: ArrayLike,
        conditions_id: str,
    ):
        event_ids_ = jnp.asarray(event_ids)
        tracks = jnp.asarray(track_indices, dtype=jnp.int32)
        elements = jnp.asarray(detector_element_ids, dtype=jnp.int32)
        starts = jnp.asarray(start_positions)
        ends = jnp.asarray(end_positions, dtype=starts.dtype)
        start_times_ = jnp.asarray(start_times, dtype=starts.dtype)
        end_times_ = jnp.asarray(end_times, dtype=starts.dtype)
        energy = jnp.asarray(deposited_energy, dtype=starts.dtype)
        active_ = jnp.asarray(active, dtype=bool)
        if tracks.ndim != 2:
            raise ValueError("Truth steps require shape (event, step).")
        expected = tracks.shape
        if (
            elements.shape != expected
            or start_times_.shape != expected
            or end_times_.shape != expected
            or energy.shape != expected
            or active_.shape != expected
        ):
            raise ValueError("Truth-step scalar fields must align.")
        if (
            starts.shape != expected + (3,)
            or ends.shape != expected + (3,)
            or event_ids_.shape != (expected[0],)
        ):
            raise ValueError("Truth-step vector or event fields have incompatible shape.")
        valid = (
            jnp.all(jnp.isfinite(starts), axis=-1)
            & jnp.all(jnp.isfinite(ends), axis=-1)
            & jnp.isfinite(start_times_)
            & jnp.isfinite(end_times_)
            & (end_times_ >= start_times_)
            & jnp.isfinite(energy)
            & (energy >= 0.0)
            & (tracks >= 0)
            & (elements >= 0)
        )
        self.event_ids = event_ids_
        self.track_indices = tracks
        self.detector_element_ids = elements
        self.start_positions = starts
        self.end_positions = ends
        self.start_times = start_times_
        self.end_times = end_times_
        self.deposited_energy = energy
        self.active = active_
        self.valid = jnp.where(active_, valid, True)
        self.conditions_id = _identifier(conditions_id, "Conditions ID")
        self.step_capacity = int(expected[1])


class SensitiveHitBank(StrictModule, NonTrainableState):
    event_ids: Array
    hit_ids: Array
    detector_element_ids: Array
    channel_ids: Array
    source_step_indices: Array
    positions: Array
    times: Array
    energies: Array
    active: Array
    valid: Array
    conditions_id: str = eqx.field(static=True)
    hit_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        event_ids: ArrayLike,
        hit_ids: ArrayLike,
        detector_element_ids: ArrayLike,
        channel_ids: ArrayLike,
        source_step_indices: ArrayLike,
        positions: ArrayLike,
        times: ArrayLike,
        energies: ArrayLike,
        active: ArrayLike,
        conditions_id: str,
    ):
        event_ids_ = jnp.asarray(event_ids)
        hit_ids_ = jnp.asarray(hit_ids, dtype=jnp.int32)
        elements = jnp.asarray(detector_element_ids, dtype=jnp.int32)
        channels = jnp.asarray(channel_ids, dtype=jnp.int32)
        sources = jnp.asarray(source_step_indices, dtype=jnp.int32)
        positions_ = jnp.asarray(positions)
        times_ = jnp.asarray(times, dtype=positions_.dtype)
        energies_ = jnp.asarray(energies, dtype=positions_.dtype)
        active_ = jnp.asarray(active, dtype=bool)
        if hit_ids_.ndim != 2:
            raise ValueError("Sensitive hits require shape (event, hit).")
        expected = hit_ids_.shape
        if any(
            value.shape != expected
            for value in (elements, channels, sources, times_, energies_, active_)
        ):
            raise ValueError("Sensitive-hit scalar fields must align.")
        if positions_.shape != expected + (3,) or event_ids_.shape != (expected[0],):
            raise ValueError(
                "Sensitive-hit vector or event fields have incompatible shape."
            )
        valid = (
            jnp.all(jnp.isfinite(positions_), axis=-1)
            & jnp.isfinite(times_)
            & jnp.isfinite(energies_)
            & (energies_ >= 0.0)
            & (elements >= 0)
            & (channels >= 0)
            & (sources >= 0)
        )
        self.event_ids = event_ids_
        self.hit_ids = hit_ids_
        self.detector_element_ids = elements
        self.channel_ids = channels
        self.source_step_indices = sources
        self.positions = positions_
        self.times = times_
        self.energies = energies_
        self.active = active_
        self.valid = jnp.where(active_, valid, True)
        self.conditions_id = _identifier(conditions_id, "Conditions ID")
        self.hit_capacity = int(expected[1])


class DigitBank(StrictModule, NonTrainableState):
    event_ids: Array
    digit_ids: Array
    channel_ids: Array
    signals: Array
    times: Array
    active: Array
    saturated: Array
    valid: Array
    conditions_id: str = eqx.field(static=True)
    channel_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        event_ids: ArrayLike,
        digit_ids: ArrayLike,
        channel_ids: ArrayLike,
        signals: ArrayLike,
        times: ArrayLike,
        active: ArrayLike,
        saturated: ArrayLike,
        conditions_id: str,
    ):
        event_ids_ = jnp.asarray(event_ids)
        digits = jnp.asarray(digit_ids, dtype=jnp.int32)
        channels = jnp.asarray(channel_ids, dtype=jnp.int32)
        signals_ = jnp.asarray(signals)
        times_ = jnp.asarray(times, dtype=signals_.dtype)
        active_ = jnp.asarray(active, dtype=bool)
        saturated_ = jnp.asarray(saturated, dtype=bool)
        if digits.ndim != 2:
            raise ValueError("Digits require shape (event, channel).")
        expected = digits.shape
        if any(
            value.shape != expected
            for value in (channels, signals_, times_, active_, saturated_)
        ) or event_ids_.shape != (expected[0],):
            raise ValueError("Digit fields must align.")
        valid = jnp.isfinite(signals_) & jnp.isfinite(times_) & (channels >= 0)
        self.event_ids = event_ids_
        self.digit_ids = digits
        self.channel_ids = channels
        self.signals = signals_
        self.times = times_
        self.active = active_
        self.saturated = saturated_
        self.valid = jnp.where(active_, valid, True)
        self.conditions_id = _identifier(conditions_id, "Conditions ID")
        self.channel_capacity = int(expected[1])


__all__ = [
    "DetectorConditions",
    "DetectorResourcePlan",
    "DigitBank",
    "SensitiveHitBank",
    "TransportTrackBank",
    "TruthStepBank",
]
