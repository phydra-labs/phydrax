#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity dark-radiation packets and atomic export accounting."""

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


def _scaled_norm(value: Array, /) -> Array:
    """Euclidean norm without avoidable squared-magnitude overflow."""

    scale = jnp.max(jnp.abs(value), axis=-1, initial=0.0)
    safe_scale = jnp.where(jnp.isfinite(scale) & (scale > 0.0), scale, 1.0)
    normalized = value / safe_scale[..., None]
    norm = scale * jnp.sqrt(jnp.sum(normalized * normalized, axis=-1))
    return jnp.where(scale > 0.0, norm, 0.0)


class DarkRadiationStatus(IntEnum):
    """Terminal status for one single-packet export transaction."""

    SUCCESS = 0
    CAPACITY_EXHAUSTED = 1
    INVALID_IDENTITY = 2
    INVALID_KINEMATICS = 3
    FOUR_MOMENTUM_IMBALANCE = 4
    INVALID_LEDGER = 5


class DarkRadiationPacket(StrictModule, NonTrainableState):
    """One emitted packet in physical four-momentum coordinates.

    ``physical_energy`` and ``physical_momentum`` are evaluated at
    ``emission_scale_factor``. ``comoving_position`` is the event position at
    that same time level. Integer identities are stable simulation identities;
    no process-local string registry is involved.
    """

    packet_id: Array
    species_id: Array
    source_event_id: Array
    parent_ids: Array
    physical_energy: Array
    physical_momentum: Array
    comoving_position: Array
    emission_scale_factor: Array

    def __init__(
        self,
        packet_id: ArrayLike,
        species_id: ArrayLike,
        source_event_id: ArrayLike,
        parent_ids: ArrayLike,
        physical_energy: ArrayLike,
        physical_momentum: ArrayLike,
        comoving_position: ArrayLike,
        emission_scale_factor: ArrayLike,
    ):
        energy = jnp.asarray(physical_energy)
        if not jnp.issubdtype(energy.dtype, jnp.floating):
            raise TypeError("Dark-radiation packet energy must use a floating dtype.")
        self.packet_id = jnp.asarray(packet_id, dtype=jnp.int64)
        self.species_id = jnp.asarray(species_id, dtype=jnp.int32)
        self.source_event_id = jnp.asarray(source_event_id, dtype=jnp.int64)
        self.parent_ids = jnp.asarray(parent_ids, dtype=jnp.int64)
        self.physical_energy = energy
        self.physical_momentum = jnp.asarray(physical_momentum, dtype=energy.dtype)
        self.comoving_position = jnp.asarray(comoving_position, dtype=energy.dtype)
        self.emission_scale_factor = jnp.asarray(
            emission_scale_factor, dtype=energy.dtype
        )


class DarkRadiationLedger(StrictModule, NonTrainableState):
    """Fixed-capacity append-only packet storage."""

    packet_ids: Array
    species_ids: Array
    source_event_ids: Array
    parent_ids: Array
    physical_energy: Array
    physical_momentum: Array
    comoving_position: Array
    emission_scale_factors: Array
    active_mask: Array


class DarkRadiationExportEvidence(StrictModule, NonTrainableState):
    """Admission and exact four-momentum evidence for one export."""

    status: Array
    selected_slot: Array
    capacity_available: Array
    source_ledger_valid: Array
    identity_valid: Array
    finite: Array
    positive_energy: Array
    future_directed: Array
    source_future_directed: Array
    retained_future_directed: Array
    four_momentum_defect: Array
    four_momentum_scale: Array
    four_momentum_tolerance: Array
    four_momentum_balanced: Array
    ledger_unchanged_on_failure: Array
    successful: Array


class DarkRadiationExportResult(StrictModule, NonTrainableState):
    packet: DarkRadiationPacket
    candidate_ledger: DarkRadiationLedger
    accepted_ledger: DarkRadiationLedger
    evidence: DarkRadiationExportEvidence
    successful: Array


class DarkRadiationLedgerPlan(StrictModule, NonTrainableState):
    """Bounded export owner for one radiation packet per transaction.

    This is deliberately not a radiation transport model. It records one
    already-resolved packet and refuses the whole transaction when storage or
    four-momentum closure is unavailable. It therefore makes no ``2 -> n``
    transport claim.
    This low-level adapter does not claim particle-state conservation. Coupled
    weighted-particle emission is owned by ``InelasticSIDMPlan.export_radiation``.
    """

    capacity: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    speed_of_light: float = eqx.field(static=True)
    energy_unit: str = eqx.field(static=True)
    momentum_unit: str = eqx.field(static=True)
    position_unit: str = eqx.field(static=True)
    absolute_balance_tolerance: float = eqx.field(static=True)
    relative_balance_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        capacity: int,
        /,
        *,
        dimension: int = 3,
        speed_of_light: float = 1.0,
        energy_unit: str = "physical-energy",
        momentum_unit: str = "physical-mass*physical-length/physical-time",
        position_unit: str = "comoving-length",
        absolute_balance_tolerance: float = 0.0,
        relative_balance_tolerance: float = 64.0 * np.finfo(np.float64).eps,
    ):
        capacity_ = int(capacity)
        dimension_ = int(dimension)
        light_speed = float(speed_of_light)
        absolute = float(absolute_balance_tolerance)
        relative = float(relative_balance_tolerance)
        energy_unit_ = str(energy_unit).strip()
        momentum_unit_ = str(momentum_unit).strip()
        position_unit_ = str(position_unit).strip()
        if capacity_ <= 0:
            raise ValueError("Dark-radiation capacity must be positive.")
        if dimension_ <= 0:
            raise ValueError("Dark-radiation dimension must be positive.")
        if not energy_unit_ or not momentum_unit_ or not position_unit_:
            raise ValueError(
                "Dark-radiation energy, momentum, and position units are required."
            )
        if (
            not np.isfinite(light_speed)
            or light_speed <= 0.0
            or not np.isfinite(absolute)
            or absolute < 0.0
            or not np.isfinite(relative)
            or relative < 0.0
        ):
            raise ValueError(
                "Light speed must be positive and four-momentum tolerances must be finite and nonnegative."
            )
        self.capacity = capacity_
        self.dimension = dimension_
        self.speed_of_light = light_speed
        self.energy_unit = energy_unit_
        self.momentum_unit = momentum_unit_
        self.position_unit = position_unit_
        self.absolute_balance_tolerance = absolute
        self.relative_balance_tolerance = relative
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dark-radiation-ledger",
                "capacity": capacity_,
                "dimension": dimension_,
                "speed_of_light": light_speed,
                "energy_unit": energy_unit_,
                "momentum_unit": momentum_unit_,
                "position_unit": position_unit_,
                "absolute_balance_tolerance": absolute,
                "relative_balance_tolerance": relative,
                "packet_multiplicity": "one-per-transaction",
                "transport_claim": "none",
            }
        )

    def empty(self, *, dtype=jnp.float64) -> DarkRadiationLedger:
        dtype_ = jnp.dtype(dtype)
        if not jnp.issubdtype(dtype_, jnp.floating):
            raise TypeError("Dark-radiation ledgers require a floating dtype.")
        shape = (self.capacity,)
        vectors = (self.capacity, self.dimension)
        return DarkRadiationLedger(
            packet_ids=jnp.full(shape, -1, dtype=jnp.int64),
            species_ids=jnp.full(shape, -1, dtype=jnp.int32),
            source_event_ids=jnp.full(shape, -1, dtype=jnp.int64),
            parent_ids=jnp.full((self.capacity, 2), -1, dtype=jnp.int64),
            physical_energy=jnp.zeros(shape, dtype=dtype_),
            physical_momentum=jnp.zeros(vectors, dtype=dtype_),
            comoving_position=jnp.zeros(vectors, dtype=dtype_),
            emission_scale_factors=jnp.zeros(shape, dtype=dtype_),
            active_mask=jnp.zeros(shape, dtype=jnp.bool_),
        )

    def valid(self, ledger: DarkRadiationLedger, /) -> Array:
        """Return complete identity, capacity, kinematic, and sentinel evidence."""

        self._check_ledger(ledger)
        active = ledger.active_mask
        inactive_id = jnp.iinfo(ledger.packet_ids.dtype).max
        sorted_active_ids = jnp.sort(jnp.where(active, ledger.packet_ids, inactive_id))
        duplicate = jnp.any(
            (sorted_active_ids[1:] == sorted_active_ids[:-1])
            & (sorted_active_ids[1:] != inactive_id)
        )
        active_identity = (
            (ledger.packet_ids >= 0)
            & (ledger.packet_ids < inactive_id)
            & (ledger.species_ids >= 0)
            & (ledger.source_event_ids >= 0)
            & jnp.all(ledger.parent_ids >= 0, axis=-1)
            & (ledger.parent_ids[:, 0] != ledger.parent_ids[:, 1])
        )
        inactive_clean = (
            (ledger.packet_ids == -1)
            & (ledger.species_ids == -1)
            & (ledger.source_event_ids == -1)
            & jnp.all(ledger.parent_ids == -1, axis=-1)
            & (ledger.physical_energy == 0.0)
            & jnp.all(ledger.physical_momentum == 0.0, axis=-1)
            & jnp.all(ledger.comoving_position == 0.0, axis=-1)
            & (ledger.emission_scale_factors == 0.0)
        )
        finite = (
            jnp.all(jnp.isfinite(ledger.physical_energy))
            & jnp.all(jnp.isfinite(ledger.physical_momentum))
            & jnp.all(jnp.isfinite(ledger.comoving_position))
            & jnp.all(jnp.isfinite(ledger.emission_scale_factors))
        )
        radiation_momentum_norm = _scaled_norm(ledger.physical_momentum)
        future = ledger.physical_energy >= self.speed_of_light * radiation_momentum_norm
        active_kinematics = (
            (ledger.physical_energy > 0.0)
            & (ledger.emission_scale_factors > 0.0)
            & future
        )
        return (
            ~duplicate
            & jnp.all(jnp.where(active, active_identity, inactive_clean))
            & jnp.all(~active | active_kinematics)
            & finite
        )

    def packet(self, ledger: DarkRadiationLedger, slot: int, /) -> DarkRadiationPacket:
        """Return one host-selected active packet without changing the ledger."""

        self._check_ledger(ledger)
        index = int(slot)
        if not 0 <= index < self.capacity:
            raise IndexError("Dark-radiation packet slot is outside ledger capacity.")
        if not bool(np.asarray(ledger.active_mask[index])):
            raise ValueError("Dark-radiation packet slot is inactive.")
        return DarkRadiationPacket(
            ledger.packet_ids[index],
            ledger.species_ids[index],
            ledger.source_event_ids[index],
            ledger.parent_ids[index],
            ledger.physical_energy[index],
            ledger.physical_momentum[index],
            ledger.comoving_position[index],
            ledger.emission_scale_factors[index],
        )

    def export(
        self,
        ledger: DarkRadiationLedger,
        packet: DarkRadiationPacket,
        source_four_momentum: ArrayLike,
        retained_four_momentum: ArrayLike,
        /,
    ) -> DarkRadiationExportResult:
        """Atomically append a packet only when its four-vector closes the event.

        Four-vectors use ``(physical_energy, physical_momentum...)`` at the
        packet's emission scale factor. The source and retained vectors must be
        expressed in the identical units and frame.
        This is a ledger adapter over caller-supplied vectors, not a particle
        transaction. It validates causal four-vectors and exact ledger closure.
        """

        self._check_ledger(ledger)
        self._check_packet(packet)
        source_ledger_valid = self.valid(ledger)
        dtype = ledger.physical_energy.dtype
        if packet.physical_energy.dtype != dtype:
            raise TypeError(
                "Dark-radiation packet and ledger must use the identical floating dtype."
            )
        source = jnp.asarray(source_four_momentum, dtype=dtype)
        retained = jnp.asarray(retained_four_momentum, dtype=dtype)
        expected = (self.dimension + 1,)
        if source.shape != expected or retained.shape != expected:
            raise ValueError(f"Dark-radiation four-momenta must have shape {expected}.")

        free = ~ledger.active_mask
        capacity_available = jnp.any(free)
        slot = jnp.argmax(free.astype(jnp.int32))
        duplicate = jnp.any(ledger.active_mask & (ledger.packet_ids == packet.packet_id))
        identity_valid = (
            (packet.packet_id >= 0)
            & (packet.packet_id < jnp.iinfo(packet.packet_id.dtype).max)
            & (packet.species_id >= 0)
            & (packet.source_event_id >= 0)
            & jnp.all(packet.parent_ids >= 0)
            & (packet.parent_ids[0] != packet.parent_ids[1])
            & ~duplicate
        )
        packet_four_momentum = jnp.concatenate(
            (jnp.reshape(packet.physical_energy, (1,)), packet.physical_momentum)
        )
        finite = (
            jnp.all(jnp.isfinite(source))
            & jnp.all(jnp.isfinite(retained))
            & jnp.all(jnp.isfinite(packet_four_momentum))
            & jnp.all(jnp.isfinite(packet.comoving_position))
            & jnp.isfinite(packet.emission_scale_factor)
        )
        positive_energy = packet.physical_energy > 0.0
        packet_momentum_norm = _scaled_norm(packet.physical_momentum)
        future_directed = (
            packet.physical_energy >= self.speed_of_light * packet_momentum_norm
        )
        source_momentum_norm = _scaled_norm(source[1:])
        retained_momentum_norm = _scaled_norm(retained[1:])
        source_future_directed = (source[0] > 0.0) & (
            source[0] >= self.speed_of_light * source_momentum_norm
        )
        retained_future_directed = (retained[0] >= 0.0) & (
            retained[0] >= self.speed_of_light * retained_momentum_norm
        )
        kinematics_valid = (
            finite
            & positive_energy
            & future_directed
            & source_future_directed
            & retained_future_directed
            & (packet.emission_scale_factor > 0.0)
        )
        defect = source - retained - packet_four_momentum
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(source), jnp.abs(retained)),
            jnp.abs(packet_four_momentum),
        )
        runtime_relative_tolerance = jnp.maximum(
            jnp.asarray(self.relative_balance_tolerance, dtype=dtype),
            jnp.asarray(64.0 * jnp.finfo(dtype).eps, dtype=dtype),
        )
        tolerance = (
            jnp.asarray(self.absolute_balance_tolerance, dtype=dtype)
            + runtime_relative_tolerance * scale
        )
        balanced = jnp.all(jnp.abs(defect) <= tolerance)
        successful = (
            source_ledger_valid
            & capacity_available
            & identity_valid
            & kinematics_valid
            & balanced
        )

        candidate = DarkRadiationLedger(
            packet_ids=ledger.packet_ids.at[slot].set(
                jnp.where(capacity_available, packet.packet_id, ledger.packet_ids[slot])
            ),
            species_ids=ledger.species_ids.at[slot].set(
                jnp.where(capacity_available, packet.species_id, ledger.species_ids[slot])
            ),
            source_event_ids=ledger.source_event_ids.at[slot].set(
                jnp.where(
                    capacity_available,
                    packet.source_event_id,
                    ledger.source_event_ids[slot],
                )
            ),
            parent_ids=ledger.parent_ids.at[slot].set(
                jnp.where(capacity_available, packet.parent_ids, ledger.parent_ids[slot])
            ),
            physical_energy=ledger.physical_energy.at[slot].set(
                jnp.where(
                    capacity_available,
                    packet.physical_energy,
                    ledger.physical_energy[slot],
                )
            ),
            physical_momentum=ledger.physical_momentum.at[slot].set(
                jnp.where(
                    capacity_available,
                    packet.physical_momentum,
                    ledger.physical_momentum[slot],
                )
            ),
            comoving_position=ledger.comoving_position.at[slot].set(
                jnp.where(
                    capacity_available,
                    packet.comoving_position,
                    ledger.comoving_position[slot],
                )
            ),
            emission_scale_factors=ledger.emission_scale_factors.at[slot].set(
                jnp.where(
                    capacity_available,
                    packet.emission_scale_factor,
                    ledger.emission_scale_factors[slot],
                )
            ),
            active_mask=ledger.active_mask.at[slot].set(capacity_available),
        )
        accepted = _select_ledger(successful, candidate, ledger)
        unchanged = _ledger_equal(accepted, ledger)
        rollback_valid = successful | unchanged
        status = jnp.asarray(int(DarkRadiationStatus.SUCCESS), dtype=jnp.int32)
        status = jnp.where(
            ~balanced,
            int(DarkRadiationStatus.FOUR_MOMENTUM_IMBALANCE),
            status,
        )
        status = jnp.where(
            ~kinematics_valid,
            int(DarkRadiationStatus.INVALID_KINEMATICS),
            status,
        )
        status = jnp.where(
            ~identity_valid,
            int(DarkRadiationStatus.INVALID_IDENTITY),
            status,
        )
        status = jnp.where(
            ~capacity_available,
            int(DarkRadiationStatus.CAPACITY_EXHAUSTED),
            status,
        )
        status = jnp.where(
            ~source_ledger_valid,
            int(DarkRadiationStatus.INVALID_LEDGER),
            status,
        )
        evidence = DarkRadiationExportEvidence(
            status=status,
            selected_slot=jnp.where(capacity_available, slot, -1).astype(jnp.int32),
            capacity_available=capacity_available,
            source_ledger_valid=source_ledger_valid,
            identity_valid=identity_valid,
            finite=finite,
            positive_energy=positive_energy,
            future_directed=future_directed,
            source_future_directed=source_future_directed,
            retained_future_directed=retained_future_directed,
            four_momentum_defect=defect,
            four_momentum_scale=scale,
            four_momentum_tolerance=tolerance,
            four_momentum_balanced=balanced,
            ledger_unchanged_on_failure=rollback_valid,
            successful=successful,
        )
        return DarkRadiationExportResult(
            packet, candidate, accepted, evidence, successful
        )

    def _check_ledger(self, ledger: DarkRadiationLedger, /) -> None:
        if not isinstance(ledger, DarkRadiationLedger):
            raise TypeError("ledger must be a DarkRadiationLedger.")
        shape = (self.capacity,)
        vectors = (self.capacity, self.dimension)
        if (
            ledger.packet_ids.shape != shape
            or ledger.species_ids.shape != shape
            or ledger.source_event_ids.shape != shape
            or ledger.parent_ids.shape != (self.capacity, 2)
            or ledger.physical_energy.shape != shape
            or ledger.physical_momentum.shape != vectors
            or ledger.comoving_position.shape != vectors
            or ledger.emission_scale_factors.shape != shape
            or ledger.active_mask.shape != shape
        ):
            raise ValueError("Dark-radiation ledger does not match its plan capacity.")
        dtype = ledger.physical_energy.dtype
        if (
            not jnp.issubdtype(dtype, jnp.floating)
            or ledger.physical_momentum.dtype != dtype
            or ledger.comoving_position.dtype != dtype
            or ledger.emission_scale_factors.dtype != dtype
        ):
            raise TypeError(
                "Dark-radiation physical fields require one shared floating dtype."
            )

    def _check_packet(self, packet: DarkRadiationPacket, /) -> None:
        if not isinstance(packet, DarkRadiationPacket):
            raise TypeError("packet must be a DarkRadiationPacket.")
        scalar_fields = (
            packet.packet_id,
            packet.species_id,
            packet.source_event_id,
            packet.physical_energy,
            packet.emission_scale_factor,
        )
        if (
            any(jnp.shape(value) != () for value in scalar_fields)
            or packet.parent_ids.shape != (2,)
            or packet.physical_momentum.shape != (self.dimension,)
            or packet.comoving_position.shape != (self.dimension,)
        ):
            raise ValueError("Dark-radiation packet fields have incompatible shapes.")
        dtype = packet.physical_energy.dtype
        if (
            not jnp.issubdtype(dtype, jnp.floating)
            or packet.physical_momentum.dtype != dtype
            or packet.comoving_position.dtype != dtype
            or packet.emission_scale_factor.dtype != dtype
        ):
            raise TypeError(
                "Dark-radiation packet physical fields require one shared floating dtype."
            )


def _select_ledger(
    predicate: Array,
    candidate: DarkRadiationLedger,
    fallback: DarkRadiationLedger,
    /,
) -> DarkRadiationLedger:
    return DarkRadiationLedger(
        *(
            jnp.where(predicate, proposed, previous)
            for proposed, previous in zip(
                (
                    candidate.packet_ids,
                    candidate.species_ids,
                    candidate.source_event_ids,
                    candidate.parent_ids,
                    candidate.physical_energy,
                    candidate.physical_momentum,
                    candidate.comoving_position,
                    candidate.emission_scale_factors,
                    candidate.active_mask,
                ),
                (
                    fallback.packet_ids,
                    fallback.species_ids,
                    fallback.source_event_ids,
                    fallback.parent_ids,
                    fallback.physical_energy,
                    fallback.physical_momentum,
                    fallback.comoving_position,
                    fallback.emission_scale_factors,
                    fallback.active_mask,
                ),
                strict=True,
            )
        )
    )


def _ledger_equal(left: DarkRadiationLedger, right: DarkRadiationLedger, /) -> Array:
    comparisons = (
        jnp.array_equal(left.packet_ids, right.packet_ids),
        jnp.array_equal(left.species_ids, right.species_ids),
        jnp.array_equal(left.source_event_ids, right.source_event_ids),
        jnp.array_equal(left.parent_ids, right.parent_ids),
        jnp.array_equal(left.physical_energy, right.physical_energy),
        jnp.array_equal(left.physical_momentum, right.physical_momentum),
        jnp.array_equal(left.comoving_position, right.comoving_position),
        jnp.array_equal(left.emission_scale_factors, right.emission_scale_factors),
        jnp.array_equal(left.active_mask, right.active_mask),
    )
    return jnp.all(jnp.stack(comparisons))


__all__ = [
    "DarkRadiationExportEvidence",
    "DarkRadiationExportResult",
    "DarkRadiationLedger",
    "DarkRadiationLedgerPlan",
    "DarkRadiationPacket",
    "DarkRadiationStatus",
]
