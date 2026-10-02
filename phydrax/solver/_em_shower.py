#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Electromagnetic showers from coupled photon and charged-particle transport.

`EMShowerPlan` alternates one `PhotonTransportPlan` generation with one
`ChargedParticleTransportPlan` generation for a fixed number of rounds. Each
generation launches one fixed-capacity `ShowerParticleBatch` per species; the
secondaries every history records into its per-history stack are compacted
into the next generation's batch. Compaction sorts by particle identity, so
the launched batches, every random draw, and every tally are independent of
the order in which primaries were supplied.

Identity is the persistent `(hi, lo)` 64-bit word pair. A secondary created at
event or step `e` by parent `p` receives `p * stride + e + 1`, where
`stride = max(photon events, charged steps) + 1`; with distinct primary
identities this is injective across the whole shower, and a child identity
that would exceed 64 bits refuses the shower. Refusal is atomic: when a
generation's secondaries exceed the batch capacity, or identities are
exhausted, nothing from that generation is launched, later generations run
empty, and the untransported secondary energy is reported as the stack
remainder so the ledger still closes.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations._charged_radiation_interactions import ChargedRadiationParticleKind
from ..typing import (
    Bool,
    checked,
    Dim,
    Float64,
    Identifier,
    Int32,
    parse,
    PRNGKey,
    Scalar,
    Size,
    UInt32,
)
from ._charged_particle_transport import (
    ChargedParticleTransportPlan,
    ChargedParticleTransportResult,
)
from ._photon_transport import PhotonTransportPlan, PhotonTransportResult
from ._secondary_stack import SecondaryParticleStack


ShowerSpecies: TypeAlias = Literal["photon", "charged"]

# Both words all-ones: the reserved identity that marks an absent parent.
_NO_IDENTITY_WORD = np.uint32(0xFFFFFFFF)
_WORD_BITS = 32
_IDENTITY_MAXIMUM = np.uint64(0xFFFFFFFFFFFFFFFF)


class _SlotDim(Dim, minimum=1):
    """Launch slots of one shower batch."""


class _PhotonSlotDim(Dim, minimum=1):
    """Launch slots of the photon batch of one shower."""


class _ChargedSlotDim(Dim, minimum=1):
    """Launch slots of the charged batch of one shower."""


class _GenerationDim(Dim, minimum=1):
    """Transported generation rounds."""


class EMShowerStatus(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    TRANSPORT_FAILURE = 2
    STACK_CAPACITY_EXHAUSTED = 3
    IDENTITY_EXHAUSTED = 4
    NONFINITE = 5


class ShowerParticleBatch(StrictModule):
    """Fixed-capacity launch batch of one species; inactive slots are padding.

    `kinds` holds `ChargedRadiationParticleKind` values for charged batches and
    is unused for photon batches. `requested_count` is the number of particles
    that belonged in the batch; it exceeds the active count only when the
    batch was refused for capacity.
    """

    __strict_contract__ = True

    species: ShowerSpecies = eqx.field(static=True)
    positions: Float64[_SlotDim, Literal[3]]
    directions: Float64[_SlotDim, Literal[3]]
    energies: Float64[_SlotDim]
    kinds: Int32[_SlotDim]
    id_hi: UInt32[_SlotDim]
    id_lo: UInt32[_SlotDim]
    parent_hi: UInt32[_SlotDim]
    parent_lo: UInt32[_SlotDim]
    active: Bool[_SlotDim]
    requested_count: Int32[Scalar]

    @property
    def capacity(self) -> int:
        return self.active.shape[0]

    @property
    def count(self) -> Array:
        return jnp.sum(self.active, dtype=jnp.int32)

    @property
    def energy(self) -> Array:
        return jnp.sum(jnp.where(self.active, self.energies, 0.0))

    @staticmethod
    def photons(
        positions: ArrayLike,
        directions: ArrayLike,
        energies: ArrayLike,
        /,
        *,
        capacity: int,
        identities: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> ShowerParticleBatch:
        """Primary photons padded to `capacity`; identities default to `(0, index)`."""
        return _primaries(
            "photon", positions, directions, energies, None, capacity, identities
        )

    @staticmethod
    def charged(
        positions: ArrayLike,
        directions: ArrayLike,
        energies: ArrayLike,
        kinds: ArrayLike,
        /,
        *,
        capacity: int,
        identities: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> ShowerParticleBatch:
        """Primary electrons/positrons padded to `capacity`."""
        return _primaries(
            "charged", positions, directions, energies, kinds, capacity, identities
        )

    @staticmethod
    def empty(species: ShowerSpecies, capacity: int, /) -> ShowerParticleBatch:
        species_ = parse(species, ShowerSpecies, "species")
        slots = int(capacity)
        if slots < 1:
            raise ValueError("Shower batch capacity must be positive.")
        return _batch(
            species_,
            jnp.zeros((slots, 3), dtype=jnp.float64),
            jnp.zeros((slots, 3), dtype=jnp.float64),
            jnp.zeros((slots,), dtype=jnp.float64),
            jnp.zeros((slots,), dtype=jnp.int32),
            jnp.zeros((slots,), dtype=jnp.uint32),
            jnp.zeros((slots,), dtype=jnp.uint32),
            jnp.full((slots,), _NO_IDENTITY_WORD, dtype=jnp.uint32),
            jnp.full((slots,), _NO_IDENTITY_WORD, dtype=jnp.uint32),
            jnp.zeros((slots,), dtype=jnp.bool_),
            jnp.asarray(0, dtype=jnp.int32),
        )


def _batch(
    species: ShowerSpecies,
    positions: Array,
    directions: Array,
    energies: Array,
    kinds: Array,
    id_hi: Array,
    id_lo: Array,
    parent_hi: Array,
    parent_lo: Array,
    active: Array,
    requested: Array,
    /,
) -> ShowerParticleBatch:
    return ShowerParticleBatch(
        species,
        positions,
        directions,
        energies,
        kinds,
        id_hi,
        id_lo,
        parent_hi,
        parent_lo,
        active,
        requested,
    )


def _primaries(
    species: ShowerSpecies,
    positions: ArrayLike,
    directions: ArrayLike,
    energies: ArrayLike,
    kinds: ArrayLike | None,
    capacity: int,
    identities: tuple[ArrayLike, ArrayLike] | None,
    /,
) -> ShowerParticleBatch:
    slots = int(capacity)
    positions_ = jnp.asarray(positions, dtype=jnp.float64)
    directions_ = jnp.asarray(directions, dtype=jnp.float64)
    energies_ = jnp.asarray(energies, dtype=jnp.float64)
    if (
        positions_.ndim != 2
        or positions_.shape[-1] != 3
        or directions_.shape != positions_.shape
    ):
        raise ValueError("Shower primaries require (count, 3) positions/directions.")
    count = positions_.shape[0]
    if energies_.shape != (count,):
        raise ValueError("Shower primary energies must align with positions.")
    if slots < 1 or count > slots:
        raise ValueError("Shower primaries exceed the batch capacity.")
    if kinds is None:
        kinds_ = jnp.zeros((count,), dtype=jnp.int32)
    else:
        kinds_ = jnp.asarray(kinds)
        if kinds_.shape != (count,) or not jnp.issubdtype(kinds_.dtype, jnp.integer):
            raise ValueError("Shower primary kinds must be an integer (count,) array.")
        kinds_ = kinds_.astype(jnp.int32)
    if identities is None:
        id_hi = jnp.zeros((count,), dtype=jnp.uint32)
        id_lo = jnp.arange(count, dtype=jnp.uint32)
    else:
        words = tuple(jnp.asarray(word) for word in identities)
        if len(words) != 2 or any(
            not jnp.issubdtype(word.dtype, jnp.integer) for word in words
        ):
            raise TypeError("identities must be two integer word arrays (hi, lo).")
        if any(word.shape != (count,) for word in words):
            raise ValueError("Identity words must align with primaries.")
        id_hi, id_lo = words[0].astype(jnp.uint32), words[1].astype(jnp.uint32)
    pad = slots - count

    def padded(value: Array, fill: float | int) -> Array:
        width = ((0, pad),) + ((0, 0),) * (value.ndim - 1)
        return jnp.pad(value, width, constant_values=fill)

    return _batch(
        species,
        padded(positions_, 0.0),
        padded(directions_, 0.0),
        padded(energies_, 0.0),
        padded(kinds_, 0),
        padded(id_hi, 0),
        padded(id_lo, 0),
        jnp.full((slots,), _NO_IDENTITY_WORD, dtype=jnp.uint32),
        jnp.full((slots,), _NO_IDENTITY_WORD, dtype=jnp.uint32),
        jnp.arange(slots) < count,
        jnp.asarray(count, dtype=jnp.int32),
    )


def _pack(hi: Array, lo: Array, /) -> Array:
    return (hi.astype(jnp.uint64) << _WORD_BITS) | lo.astype(jnp.uint64)


def _unpack(identity: Array, /) -> tuple[Array, Array]:
    return (identity >> _WORD_BITS).astype(jnp.uint32), (
        identity & jnp.uint64(0xFFFFFFFF)
    ).astype(jnp.uint32)


def _order(batch: ShowerParticleBatch, /) -> tuple[ShowerParticleBatch, Array]:
    """Sort active slots by identity ahead of padding; flag duplicate identities."""
    identity = _pack(batch.id_hi, batch.id_lo)
    sort_key = jnp.where(batch.active, identity, _IDENTITY_MAXIMUM)
    order = jnp.argsort(sort_key, stable=True)
    sorted_key = sort_key[order]
    duplicate = jnp.any(
        (sorted_key[1:] == sorted_key[:-1]) & (sorted_key[1:] != _IDENTITY_MAXIMUM)
    )
    return _batch(
        batch.species,
        batch.positions[order],
        batch.directions[order],
        batch.energies[order],
        batch.kinds[order],
        batch.id_hi[order],
        batch.id_lo[order],
        batch.parent_hi[order],
        batch.parent_lo[order],
        batch.active[order],
        batch.requested_count,
    ), duplicate


class _Compaction(StrictModule):
    batch: ShowerParticleBatch
    candidate_energy: Array
    candidate_count: Array
    overflow: Array
    exhausted: Array


def _compact(
    species: ShowerSpecies,
    stack: SecondaryParticleStack,
    parents: ShowerParticleBatch,
    capacity: int,
    stride: int,
    kind: int,
    /,
) -> _Compaction:
    """Gather one generation's secondaries into the next launch batch."""
    active = stack.active & parents.active[:, None]
    parent = _pack(parents.id_hi, parents.id_lo)[:, None]
    offset = (stack.creation_index + 1).astype(jnp.uint64)
    stride_ = jnp.asarray(stride, dtype=jnp.uint64)
    exhausted_entry = active & (parent > (_IDENTITY_MAXIMUM - offset) // stride_)
    child = parent * stride_ + offset
    flat_active = active.reshape(-1)
    sort_key = jnp.where(flat_active, child.reshape(-1), _IDENTITY_MAXIMUM)
    order = jnp.argsort(sort_key, stable=True)[:capacity]
    count = jnp.sum(flat_active, dtype=jnp.int32)
    energy = jnp.sum(jnp.where(active, stack.energies, 0.0))
    parent_hi = jnp.broadcast_to(parents.id_hi[:, None], stack.active.shape)
    parent_lo = jnp.broadcast_to(parents.id_lo[:, None], stack.active.shape)
    child_hi, child_lo = _unpack(child.reshape(-1)[order])
    launched = jnp.arange(capacity) < count
    batch = _batch(
        species,
        stack.positions.reshape(-1, 3)[order],
        stack.directions.reshape(-1, 3)[order],
        stack.energies.reshape(-1)[order],
        stack.particle_kind.reshape(-1)[order],
        jnp.where(launched, child_hi, 0).astype(jnp.uint32),
        jnp.where(launched, child_lo, 0).astype(jnp.uint32),
        jnp.where(launched, parent_hi.reshape(-1)[order], _NO_IDENTITY_WORD).astype(
            jnp.uint32
        ),
        jnp.where(launched, parent_lo.reshape(-1)[order], _NO_IDENTITY_WORD).astype(
            jnp.uint32
        ),
        launched,
        count,
    )
    return _Compaction(
        batch,
        energy,
        count,
        count > capacity,
        jnp.any(exhausted_entry),
    )


def _refuse(batch: ShowerParticleBatch, refused: Array, /) -> ShowerParticleBatch:
    """Deactivate every slot when the shower has been refused; counts are kept."""
    return _batch(
        batch.species,
        batch.positions,
        batch.directions,
        batch.energies,
        batch.kinds,
        batch.id_hi,
        batch.id_lo,
        batch.parent_hi,
        batch.parent_lo,
        batch.active & ~refused,
        batch.requested_count,
    )


def _masked_sum(values: Array, active: Array, /) -> Array:
    return jnp.sum(jnp.where(active, values, 0.0))


class EMShowerResult(StrictModule, NonTrainableState):
    """Shower tallies in eV, per-generation evidence, and every transport result.

    `primary_energy` equals `deposited_energy + escaped_energy +
    truncated_energy + stack_remainder_energy` up to `ledger_residual`;
    `annihilation_photon_energy` is positron rest-mass energy released at
    rest and is reported outside that kinetic ledger. `photon_launches` and
    `charged_launches` hold one batch per generation plus the final,
    untransported remainder; `refused_generation` equals the generation count
    when no refusal occurred.
    """

    __strict_contract__ = True

    primary_energy: Float64[Scalar]
    deposited_energy: Float64[Scalar]
    escaped_energy: Float64[Scalar]
    truncated_energy: Float64[Scalar]
    stack_remainder_energy: Float64[Scalar]
    annihilation_photon_energy: Float64[Scalar]
    photon_to_charged_energy: Float64[Scalar]
    charged_to_photon_energy: Float64[Scalar]
    ledger_residual: Float64[Scalar]
    generation_photon_count: Int32[_GenerationDim]
    generation_charged_count: Int32[_GenerationDim]
    generation_photon_energy: Float64[_GenerationDim]
    generation_charged_energy: Float64[_GenerationDim]
    generation_secondary_photon_count: Int32[_GenerationDim]
    generation_secondary_electron_count: Int32[_GenerationDim]
    photon_generations: tuple[PhotonTransportResult, ...]
    charged_generations: tuple[ChargedParticleTransportResult, ...]
    photon_launches: tuple[ShowerParticleBatch, ...]
    charged_launches: tuple[ShowerParticleBatch, ...]
    refused_generation: Int32[Scalar]
    status: Int32[Scalar]
    successful: Bool[Scalar]
    plan_id: Identifier = eqx.field(static=True)


class EMShowerPlan(StrictModule, NonTrainableState):
    """Bounded-generation electromagnetic shower on one voxel geometry."""

    __strict_contract__ = True

    photon_transport: PhotonTransportPlan
    charged_transport: ChargedParticleTransportPlan
    photon_capacity: Size[_PhotonSlotDim] = eqx.field(static=True)
    charged_capacity: Size[_ChargedSlotDim] = eqx.field(static=True)
    maximum_generations: Size[_GenerationDim] = eqx.field(static=True)
    identity_stride: int = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)

    @checked
    def __init__(
        self,
        photon_transport: PhotonTransportPlan,
        charged_transport: ChargedParticleTransportPlan,
        /,
        *,
        photon_capacity: int,
        charged_capacity: int,
        maximum_generations: int,
    ) -> None:
        photons, charged = int(photon_capacity), int(charged_capacity)
        generations = int(maximum_generations)
        if photons < 1 or charged < 1 or generations < 1:
            raise ValueError("Shower capacities and generation count must be positive.")
        electron_stack = photon_transport.electron_stack
        photon_stack = charged_transport.photon_stack
        if electron_stack is None or photon_stack is None:
            raise ValueError(
                "Shower transports must both have a secondary stack attached."
            )
        if (
            photon_transport.geometry.geometry_id
            != charged_transport.geometry.geometry_id
        ):
            raise ValueError("Shower transports must share one voxel geometry.")
        if electron_stack.minimum_energy < charged_transport.cutoff_energy_ev:
            raise ValueError(
                "Photon electron_stack minimum_energy must reach the charged cutoff."
            )
        if photon_stack.minimum_energy < photon_transport.cutoff_energy:
            raise ValueError(
                "Charged photon_stack minimum_energy must reach the photon cutoff."
            )
        self.photon_transport = photon_transport
        self.charged_transport = charged_transport
        self.photon_capacity = photons
        self.charged_capacity = charged
        self.maximum_generations = generations
        self.identity_stride = (
            max(2 * photon_transport.maximum_events, charged_transport.maximum_steps) + 1
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "electromagnetic-shower",
                "photon_transport": photon_transport.plan_id,
                "charged_transport": charged_transport.plan_id,
                "photon_capacity": photons,
                "charged_capacity": charged,
                "maximum_generations": generations,
                "identity_stride": self.identity_stride,
            }
        )

    def _validate_batch(
        self, batch: ShowerParticleBatch | None, species: ShowerSpecies, capacity: int
    ) -> ShowerParticleBatch:
        if batch is None:
            return ShowerParticleBatch.empty(species, capacity)
        if not isinstance(batch, ShowerParticleBatch):
            raise TypeError("Shower primaries must be ShowerParticleBatch values.")
        if batch.species != species or batch.capacity != capacity:
            raise ValueError(
                "Shower primary batch species or capacity does not match the plan."
            )
        return batch

    def simulate(
        self,
        photons: ShowerParticleBatch | None,
        charged: ShowerParticleBatch | None,
        key: PRNGKey,
        /,
    ) -> EMShowerResult:
        root = parse(key, PRNGKey, "key")
        photon_batch, photon_duplicate = _order(
            self._validate_batch(photons, "photon", self.photon_capacity)
        )
        charged_batch, charged_duplicate = _order(
            self._validate_batch(charged, "charged", self.charged_capacity)
        )
        invalid = photon_duplicate | charged_duplicate
        photon_batch = _refuse(photon_batch, invalid)
        charged_batch = _refuse(charged_batch, invalid)
        primary_energy = photon_batch.energy + charged_batch.energy
        generations = self.maximum_generations
        refused = invalid
        refused_generation = jnp.where(invalid, 0, generations).astype(jnp.int32)
        status = jnp.where(
            invalid, int(EMShowerStatus.INVALID_INPUT), int(EMShowerStatus.SUCCESS)
        ).astype(jnp.int32)
        zero = jnp.asarray(0.0, dtype=jnp.float64)
        deposited = escaped = truncated = remainder = annihilation = zero
        photon_to_charged = charged_to_photon = zero
        photon_counts: list[Array] = []
        charged_counts: list[Array] = []
        photon_energies: list[Array] = []
        charged_energies: list[Array] = []
        secondary_photons: list[Array] = []
        secondary_electrons: list[Array] = []
        photon_results: list[PhotonTransportResult] = []
        charged_results: list[ChargedParticleTransportResult] = []
        photon_launches: list[ShowerParticleBatch] = [photon_batch]
        charged_launches: list[ShowerParticleBatch] = [charged_batch]
        for generation in range(generations):
            photon_result = self.photon_transport.simulate(
                photon_batch.positions,
                photon_batch.directions,
                jnp.where(photon_batch.active, photon_batch.energies, 0.0),
                root,
                identities=(photon_batch.id_hi, photon_batch.id_lo),
            )
            charged_result = self.charged_transport.simulate(
                charged_batch.positions,
                charged_batch.directions,
                jnp.where(charged_batch.active, charged_batch.energies, 0.0),
                charged_batch.kinds,
                root,
                identities=(charged_batch.id_hi, charged_batch.id_lo),
            )
            photon_active = photon_batch.active
            charged_active = charged_batch.active
            transport_ok = jnp.all(~photon_active | photon_result.successful) & jnp.all(
                ~charged_active | charged_result.successful
            )
            deposited = (
                deposited
                + _masked_sum(
                    jnp.sum(photon_result.material_kerma, axis=1), photon_active
                )
                + _masked_sum(charged_result.deposited_energy, charged_active)
            )
            escaped = (
                escaped
                + _masked_sum(photon_result.escaped_energy, photon_active)
                + _masked_sum(charged_result.escaped_energy, charged_active)
            )
            truncated = (
                truncated
                + _masked_sum(photon_result.truncated_energy, photon_active)
                + _masked_sum(charged_result.truncated_energy, charged_active)
            )
            annihilation = annihilation + _masked_sum(
                charged_result.annihilation_photon_energy, charged_active
            )
            electron_stack = photon_result.secondary_electrons
            photon_stack = charged_result.secondary_photons
            if electron_stack is None or photon_stack is None:
                raise RuntimeError("Shower transports produced no secondary stacks.")
            next_charged = _compact(
                "charged",
                electron_stack,
                photon_batch,
                self.charged_capacity,
                self.identity_stride,
                int(ChargedRadiationParticleKind.ELECTRON),
            )
            next_photons = _compact(
                "photon",
                photon_stack,
                charged_batch,
                self.photon_capacity,
                self.identity_stride,
                0,
            )
            photon_to_charged = photon_to_charged + next_charged.candidate_energy
            charged_to_photon = charged_to_photon + next_photons.candidate_energy
            overflow = next_charged.overflow | next_photons.overflow
            exhausted = next_charged.exhausted | next_photons.exhausted
            failure = ~transport_ok
            newly = ~refused & (failure | overflow | exhausted)
            status = jnp.where(
                newly,
                jnp.where(
                    failure,
                    int(EMShowerStatus.TRANSPORT_FAILURE),
                    jnp.where(
                        overflow,
                        int(EMShowerStatus.STACK_CAPACITY_EXHAUSTED),
                        int(EMShowerStatus.IDENTITY_EXHAUSTED),
                    ),
                ),
                status,
            ).astype(jnp.int32)
            refused_generation = jnp.where(newly, generation, refused_generation).astype(
                jnp.int32
            )
            refused = refused | newly
            remainder = remainder + jnp.where(
                newly,
                next_charged.candidate_energy + next_photons.candidate_energy,
                0.0,
            )
            photon_counts.append(photon_batch.count)
            charged_counts.append(charged_batch.count)
            photon_energies.append(photon_batch.energy)
            charged_energies.append(charged_batch.energy)
            secondary_photons.append(next_photons.candidate_count)
            secondary_electrons.append(next_charged.candidate_count)
            photon_results.append(photon_result)
            charged_results.append(charged_result)
            photon_batch = _refuse(next_photons.batch, refused)
            charged_batch = _refuse(next_charged.batch, refused)
            photon_launches.append(photon_batch)
            charged_launches.append(charged_batch)
        remainder = remainder + jnp.where(
            refused, 0.0, photon_batch.energy + charged_batch.energy
        )
        residual = primary_energy - deposited - escaped - truncated - remainder
        finite = (
            jnp.isfinite(residual)
            & jnp.isfinite(deposited)
            & jnp.isfinite(escaped)
            & jnp.isfinite(truncated)
            & jnp.isfinite(remainder)
        )
        tolerance = 256.0 * jnp.finfo(jnp.float64).eps * jnp.maximum(primary_energy, 1.0)
        status = jnp.where(finite, status, int(EMShowerStatus.NONFINITE)).astype(
            jnp.int32
        )
        successful = (
            finite
            & (status == int(EMShowerStatus.SUCCESS))
            & (jnp.abs(residual) <= tolerance)
        )
        return EMShowerResult(
            primary_energy,
            deposited,
            escaped,
            truncated,
            remainder,
            annihilation,
            photon_to_charged,
            charged_to_photon,
            residual,
            jnp.stack(photon_counts),
            jnp.stack(charged_counts),
            jnp.stack(photon_energies),
            jnp.stack(charged_energies),
            jnp.stack(secondary_photons),
            jnp.stack(secondary_electrons),
            tuple(photon_results),
            tuple(charged_results),
            tuple(photon_launches),
            tuple(charged_launches),
            refused_generation,
            status,
            successful,
            self.plan_id,
        )


__all__ = [
    "EMShowerPlan",
    "EMShowerResult",
    "EMShowerStatus",
    "ShowerParticleBatch",
    "ShowerSpecies",
]
