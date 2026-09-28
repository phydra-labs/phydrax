#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Identity-addressed particle tracks recorded inside a PIC run.

`PICTrackRecorder` follows declared persistent particle identities (the
``(id_hi, id_lo)`` words of `ParticlePopulationState`), not storage slots: on
every accepted step it locates each tracked identity among the active slots of
its species, so tracks survive slot permutation (migration) and a reused slot
never continues the track of its previous occupant. Identities that do not yet
exist (particles created later by a process) are inactive until they appear;
particles that die become inactive.

Sampling convention. The PIC state holds integer-time positions ``x^k`` and
half-step proper velocities ``u^{k−1/2}``. Sample ``k`` is ``(t_k, x^k, ū^k)``
with the time-centered ``ū^k = (u^{k−1/2} + u^{k+1/2}) / 2``, so the midpoint
velocity of each sampled segment equals the leapfrog drift velocity to second
order. Sample ``k`` is therefore emitted when step ``k → k+1`` is accepted: the
recorded track lags the PIC state by one step. A particle that dies during
step ``k → k+1`` keeps ``u^{k−1/2}`` at its last sample. Reduced 1-D/2-D runs
resolve only the leading position axes; the remaining coordinates start at zero
when a lane first appears and follow the drift ``u/γ`` of the recorded proper
velocity, a reduced-geometry convention rather than resolved positions.
A moving window reports its shifts through `shift_frame`; the accumulated
offset is added to window-local positions, so tracks stay in the fixed frame.

Each lane keeps the charge number and macroparticle mass of its first sighting;
a later change of either (ionization, reweighting) ends the lane with
``property_transition`` evidence instead of silently changing its charge.
Lanes convert to `ChargedTrajectory` with physical particle charge
``sign(q/m) Z e`` and multiplicity ``mass |q/m| / e`` so that their product is
the PIC macrocharge; the charge unit of the PIC run must be the scale's.

Samples are stored in a fixed-capacity ring; the recorder can also fold every
emitted sample into a streaming trajectory-radiation accumulator, with or
without stored tracks. Recorders are diagnostic-only: they own no radiation
energy and never alter the run.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._physical import ElectromagneticScaleContract, RelativityScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_integer
from ...electromagnetics._trajectory_radiation import (
    ChargedTrajectory,
    PreparedTrajectoryRadiation,
    TrajectoryRadiationResult,
    TrajectoryRadiationState,
)
from ...typing import (
    Bool,
    ConvertibleToArray,
    Dim,
    Float64,
    Int32,
    parse,
    Scalar,
    Scope,
    UInt32,
)
from ._charge_state import PICSpeciesPlan, PICSpeciesState
from ._process import AbstractPICRecorder


TrackOverflowPolicy: TypeAlias = Literal["refuse", "keep-latest"]

_NO_IDENTITY_WORD = np.uint32(0xFFFFFFFF)


class _LaneDim(Dim, minimum=1):
    """Tracked identities."""


class _SampleDim(Dim, minimum=1):
    """Ring-buffer rows."""


class PICTrackBuffer(StrictModule):
    """Fixed-capacity sample ring: row ``k mod K`` holds emitted sample ``k``.

    ``slots`` and ``incarnations`` are the storage slot and slot incarnation the
    identity occupied at the sample (``-1`` and ``0`` when absent); they are
    provenance only, the lane identity is what the track follows.
    """

    __strict_contract__ = True

    times: Float64[_SampleDim]
    positions: Float64[_SampleDim, _LaneDim, Literal[3]]
    proper_velocities: Float64[_SampleDim, _LaneDim, Literal[3]]
    active: Bool[_SampleDim, _LaneDim]
    slots: Int32[_SampleDim, _LaneDim]
    incarnations: Int32[_SampleDim, _LaneDim]


class PICTrackRecorderState(StrictModule):
    """Recorder state carried in `ElectromagneticPICState.recorders`.

    ``sample_count`` counts emitted samples; with ring capacity ``K`` the
    ``max(sample_count − K, 0)`` oldest are overwritten. ``pending_*`` is sample
    ``sample_count`` awaiting the next half-step velocity (``pending_proper_velocity``
    is ``u^{k−1/2}``). Per-lane evidence: ``seen`` once the identity was found,
    ``charge_number``/``mass``/``parent_*`` at first sighting, ``first_step`` and
    ``last_step`` of presence (``-1`` if never), ``activations`` counts
    inactive→active transitions (more than one means the identity disappeared and
    reappeared), ``property_transition`` a charge-number or mass change, and
    ``duplicate`` an identity found in more than one active slot (that sample is
    recorded inactive). ``frame_offset`` is the accumulated moving-window shift
    added to runtime positions, so recorded positions stay in the fixed frame.
    """

    __strict_contract__ = True

    buffer: PICTrackBuffer | None
    radiation: TrajectoryRadiationState | None
    sample_count: Int32[Scalar]
    pending_time: Float64[Scalar]
    pending_position: Float64[_LaneDim, Literal[3]]
    pending_proper_velocity: Float64[_LaneDim, Literal[3]]
    pending_active: Bool[_LaneDim]
    pending_slot: Int32[_LaneDim]
    pending_incarnation: Int32[_LaneDim]
    seen: Bool[_LaneDim]
    charge_number: Int32[_LaneDim]
    mass: Float64[_LaneDim]
    parent_hi: UInt32[_LaneDim]
    parent_lo: UInt32[_LaneDim]
    first_step: Int32[_LaneDim]
    last_step: Int32[_LaneDim]
    activations: Int32[_LaneDim]
    property_transition: Bool[_LaneDim]
    duplicate: Bool[_LaneDim]
    frame_offset: Float64[Literal[3]]


class _Observation(NamedTuple):
    present: Array
    duplicate: Array
    slot: Array
    position: Array
    proper_velocity: Array
    charge_number: Array
    mass: Array
    incarnation: Array
    parent_hi: Array
    parent_lo: Array


def _identity_key(hi: Array, lo: Array, /) -> Array:
    return (hi.astype(jnp.uint64) << 32) | lo.astype(jnp.uint64)


def _scale_elementary_charge(
    relativity: RelativityScaleContract, scale: ElectromagneticScaleContract, /
) -> float:
    if not isinstance(scale, ElectromagneticScaleContract):
        raise TypeError("scale must be ElectromagneticScaleContract.")
    if (
        relativity.dimensional_scale.scale_id
        != scale.relativity.dimensional_scale.scale_id
    ):
        raise ValueError(
            "The recorder relativity scale and the electromagnetic scale use "
            "different length, mass, or time units."
        )
    if relativity.speed_of_light != scale.speed_of_light:
        raise ValueError(
            f"The recorder speed of light {relativity.speed_of_light} does not match "
            f"the electromagnetic scale speed of light {scale.speed_of_light}."
        )
    return float(scale.elementary_charge)


class PICTrackRecorder(AbstractPICRecorder, NonTrainableState):
    """Record declared particle identities of a PIC run.

    ``species`` are the run's species plans in run order; lane ``p`` tracks the
    identity ``(identities[0][p], identities[1][p])`` of species
    ``lane_species[p]`` (identities are per species population). ``relativity``
    is the run pusher's relativity scale; the PIC plan refuses a recorder whose
    species plans or relativity scale differ from its own. ``sample_capacity`` rows
    are stored per
    lane (``None`` stores no tracks); under ``overflow="refuse"``
    `to_charged_trajectory` refuses a ring that lost samples, under
    ``"keep-latest"`` it returns the latest ``sample_capacity`` samples.
    ``radiation`` (optional) streams every emitted sample into a prepared
    trajectory-radiation accumulator in the scale of its plan; the Hermite route
    needs accelerations the recorder does not sample and is refused.
    """

    __strict_contract__ = True

    lane_species: Int32[_LaneDim]
    id_hi: UInt32[_LaneDim]
    id_lo: UInt32[_LaneDim]
    lane_specific_charge: Float64[_LaneDim]
    sorted_lanes: Int32[_LaneDim]
    sorted_hi: UInt32[_LaneDim]
    sorted_lo: UInt32[_LaneDim]
    radiation: PreparedTrajectoryRadiation | None
    relativity: RelativityScaleContract = eqx.field(static=True)
    species_ids: tuple[str, ...] = eqx.field(static=True)
    capacities: tuple[int, ...] = eqx.field(static=True)
    species_blocks: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    sample_capacity: int | None = eqx.field(static=True)
    overflow: TrackOverflowPolicy = eqx.field(static=True)
    recorder_id: str = eqx.field(static=True)

    def __init__(
        self,
        species: Sequence[PICSpeciesPlan],
        lane_species: ConvertibleToArray,
        identities: tuple[ConvertibleToArray, ConvertibleToArray],
        /,
        *,
        relativity: RelativityScaleContract,
        sample_capacity: int | None = None,
        overflow: TrackOverflowPolicy = "refuse",
        radiation: PreparedTrajectoryRadiation | None = None,
    ) -> None:
        species_ = tuple(species)
        if not species_ or any(not isinstance(v, PICSpeciesPlan) for v in species_):
            raise TypeError("species must be a nonempty sequence of PICSpeciesPlan.")
        if not isinstance(relativity, RelativityScaleContract):
            raise TypeError("relativity must be RelativityScaleContract.")
        lanes = np.asarray(lane_species)
        hi = np.asarray(identities[0])
        lo = np.asarray(identities[1])
        if not np.issubdtype(lanes.dtype, np.integer):
            raise TypeError("lane_species must be integers.")
        if hi.dtype != np.uint32 or lo.dtype != np.uint32:
            raise TypeError("Tracked identities must be uint32 words.")
        if lanes.ndim != 1 or lanes.size == 0 or hi.shape != lanes.shape:
            raise ValueError("lane_species and identities must be nonempty rank one.")
        if lo.shape != lanes.shape:
            raise ValueError("lane_species and identities must be nonempty rank one.")
        if np.any((lanes < 0) | (lanes >= len(species_))):
            raise ValueError("lane_species references a species outside the run.")
        if np.any((hi == _NO_IDENTITY_WORD) & (lo == _NO_IDENTITY_WORD)):
            raise ValueError("The all-ones identity is reserved and cannot be tracked.")
        key = (hi.astype(np.uint64) << np.uint64(32)) | lo.astype(np.uint64)
        order = np.lexsort((key, lanes))
        ordered_species, ordered_key = lanes[order], key[order]
        if np.any(
            (ordered_species[1:] == ordered_species[:-1])
            & (ordered_key[1:] == ordered_key[:-1])
        ):
            raise ValueError("Tracked identities must be distinct within a species.")
        capacity = (
            None
            if sample_capacity is None
            else positive_integer(sample_capacity, "sample_capacity")
        )
        policy = parse(overflow, TrackOverflowPolicy, "overflow")
        if radiation is not None:
            if not isinstance(radiation, PreparedTrajectoryRadiation):
                raise TypeError("radiation must be PreparedTrajectoryRadiation or None.")
            if radiation.plan.route == "segment-hermite":
                raise ValueError(
                    "segment-hermite radiation needs sampled accelerations; the "
                    "track recorder samples positions and proper velocities only."
                )
            _scale_elementary_charge(relativity, radiation.plan.scale)
        if capacity is None and radiation is None:
            raise ValueError(
                "A track recorder needs sample_capacity, radiation, or both."
            )
        blocks = []
        for index in range(len(species_)):
            members = np.flatnonzero(ordered_species == index)
            if members.size:
                blocks.append((index, int(members[0]), int(members[-1]) + 1))
        specific = np.asarray(
            [value.charge_model.base_specific_charge for value in species_],
            dtype=np.float64,
        )[lanes]
        scope = Scope()
        self.lane_species = parse(
            jnp.asarray(lanes, dtype=jnp.int32),
            Int32[_LaneDim],
            "lane_species",
            scope=scope,
        )
        self.id_hi = parse(
            jnp.asarray(hi), UInt32[_LaneDim], "identities[0]", scope=scope
        )
        self.id_lo = parse(
            jnp.asarray(lo), UInt32[_LaneDim], "identities[1]", scope=scope
        )
        self.lane_specific_charge = parse(
            jnp.asarray(specific), Float64[_LaneDim], "lane_specific_charge", scope=scope
        )
        self.sorted_lanes = parse(
            jnp.asarray(order, dtype=jnp.int32),
            Int32[_LaneDim],
            "sorted_lanes",
            scope=scope,
        )
        self.sorted_hi = parse(
            jnp.asarray(hi[order]), UInt32[_LaneDim], "sorted_hi", scope=scope
        )
        self.sorted_lo = parse(
            jnp.asarray(lo[order]), UInt32[_LaneDim], "sorted_lo", scope=scope
        )
        self.radiation = radiation
        self.relativity = relativity
        self.species_ids = tuple(value.plan_id for value in species_)
        self.capacities = tuple(value.capacity for value in species_)
        self.species_blocks = tuple(blocks)
        self.sample_capacity = capacity
        self.overflow = policy
        self.recorder_id = canonical_fingerprint(
            {
                "kind": "pic-track-recorder",
                "species": [value.plan_id for value in species_],
                "lane_species": lanes.astype(np.int64),
                "id_hi": hi,
                "id_lo": lo,
                "relativity": relativity.scale_id,
                "sample_capacity": capacity,
                "overflow": policy,
                "radiation": None if radiation is None else radiation.plan.plan_id,
            }
        )

    @property
    def lane_count(self) -> int:
        return self.id_hi.shape[0]

    def validate_run(
        self,
        species: tuple[PICSpeciesPlan, ...],
        relativity: RelativityScaleContract,
        /,
    ) -> None:
        """Refuse a run whose species plans or pusher relativity differ from the recorder's."""
        if tuple(value.plan_id for value in species) != self.species_ids:
            raise ValueError(
                f"Track recorder {self.recorder_id[:12]} was declared for other "
                "species plans than the PIC run."
            )
        if relativity.scale_id != self.relativity.scale_id:
            raise ValueError(
                f"Track recorder {self.recorder_id[:12]} relativity scale differs from "
                "the PIC pusher's; positions, times, and proper velocities would be "
                "read in other units."
            )

    def shift_frame(
        self, state: PICTrackRecorderState, axis: int, distance: float, /
    ) -> PICTrackRecorderState:
        """Accumulate a moving-window shift so recorded positions stay in the fixed frame."""
        if not isinstance(state, PICTrackRecorderState):
            raise TypeError("state must be PICTrackRecorderState.")
        return dataclasses.replace(
            state, frame_offset=state.frame_offset.at[axis].add(distance)
        )

    # -- observation -------------------------------------------------------------

    def _observe(self, species: tuple[PICSpeciesState, ...], /) -> _Observation:
        """Locate every tracked identity among the active slots of its species."""
        if len(species) != len(self.capacities) or any(
            value.population.active.shape != (capacity,)
            for value, capacity in zip(species, self.capacities, strict=True)
        ):
            raise ValueError("Species states do not match the recorder's species plans.")
        count = self.lane_count
        matches = jnp.zeros((count,), dtype=jnp.int32)
        slot = jnp.full((count,), -1, dtype=jnp.int32)
        position = jnp.zeros((count, 3), dtype=jnp.float64)
        velocity = jnp.zeros((count, 3), dtype=jnp.float64)
        charge_number = jnp.zeros((count,), dtype=jnp.int32)
        mass = jnp.zeros((count,), dtype=jnp.float64)
        incarnation = jnp.zeros((count,), dtype=jnp.int32)
        parent_hi = jnp.full((count,), _NO_IDENTITY_WORD, dtype=jnp.uint32)
        parent_lo = jnp.full((count,), _NO_IDENTITY_WORD, dtype=jnp.uint32)
        for index, start, stop in self.species_blocks:
            state = species[index]
            population = state.population
            capacity = self.capacities[index]
            keys = _identity_key(self.sorted_hi[start:stop], self.sorted_lo[start:stop])
            slot_keys = _identity_key(population.id_hi, population.id_lo)
            # Tracked keys are sorted per species on the host: each slot finds its
            # candidate lane in O(log L) and lanes collect their slot by scatter.
            where = jnp.clip(jnp.searchsorted(keys, slot_keys), 0, stop - start - 1)
            match = population.active & (keys[where] == slot_keys)
            target = jnp.where(match, self.sorted_lanes[start:stop][where], count)
            matches = matches.at[target].add(1, mode="drop")
            slot = slot.at[target].max(jnp.arange(capacity, dtype=jnp.int32), mode="drop")
            mine = self.lane_species == index
            safe = jnp.clip(slot, 0, capacity - 1)
            resolved = state.particles.position[safe].astype(jnp.float64)
            dimension = resolved.shape[1]
            position = jnp.where(
                mine[:, None], position.at[:, :dimension].set(resolved), position
            )
            velocity = jnp.where(
                mine[:, None],
                state.particles.proper_velocity[safe].astype(jnp.float64),
                velocity,
            )
            charge_number = jnp.where(
                mine, state.charge.charge_number[safe].astype(jnp.int32), charge_number
            )
            mass = jnp.where(mine, population.mass[safe].astype(jnp.float64), mass)
            incarnation = jnp.where(mine, population.incarnation[safe], incarnation)
            parent_hi = jnp.where(mine, population.parent_hi[safe], parent_hi)
            parent_lo = jnp.where(mine, population.parent_lo[safe], parent_lo)
        present = matches == 1
        return _Observation(
            present,
            matches > 1,
            jnp.where(present, slot, -1),
            position,
            velocity,
            charge_number,
            mass,
            jnp.where(present, incarnation, 0),
            parent_hi,
            parent_lo,
        )

    def _drift_velocity(self, proper_velocity: Array, /) -> Array:
        light = float(self.relativity.speed_of_light)
        gamma = jnp.sqrt(
            1.0 + jnp.sum(proper_velocity * proper_velocity, axis=-1) / light**2
        )
        return proper_velocity / gamma[:, None]

    def _lane_charges(
        self, state: PICTrackRecorderState, elementary_charge: float, /
    ) -> tuple[Array, Array]:
        """Physical particle charges and multiplicities whose product is the macrocharge."""
        specific = self.lane_specific_charge
        charges = (
            jnp.sign(specific)
            * state.charge_number.astype(jnp.float64)
            * elementary_charge
        )
        return charges, state.mass * jnp.abs(specific) / elementary_charge

    def _sample_trajectory(
        self,
        state: PICTrackRecorderState,
        times: Array,
        positions: Array,
        proper_velocities: Array,
        active: Array,
        elementary_charge: float,
        /,
    ) -> ChargedTrajectory:
        charges, multiplicities = self._lane_charges(state, elementary_charge)
        return ChargedTrajectory(
            times,
            positions,
            proper_velocities,
            charges,
            multiplicities,
            active,
            (self.id_hi, self.id_lo),
        )

    # -- recorder protocol ---------------------------------------------------------

    def initialize(
        self,
        species: tuple[PICSpeciesState, ...],
        time: Array,
        /,
    ) -> PICTrackRecorderState:
        observed = self._observe(species)
        count = self.lane_count
        present = observed.present
        start = jnp.asarray(time, dtype=jnp.float64).reshape(())
        step = jnp.where(present, 0, -1).astype(jnp.int32)
        buffer = (
            None
            if self.sample_capacity is None
            else PICTrackBuffer(
                jnp.zeros((self.sample_capacity,), dtype=jnp.float64),
                jnp.zeros((self.sample_capacity, count, 3), dtype=jnp.float64),
                jnp.zeros((self.sample_capacity, count, 3), dtype=jnp.float64),
                jnp.zeros((self.sample_capacity, count), dtype=jnp.bool_),
                jnp.full((self.sample_capacity, count), -1, dtype=jnp.int32),
                jnp.zeros((self.sample_capacity, count), dtype=jnp.int32),
            )
        )
        state = PICTrackRecorderState(
            buffer=buffer,
            radiation=None,
            sample_count=jnp.asarray(0, dtype=jnp.int32),
            pending_time=start,
            pending_position=jnp.where(present[:, None], observed.position, 0.0),
            pending_proper_velocity=jnp.where(
                present[:, None], observed.proper_velocity, 0.0
            ),
            pending_active=present,
            pending_slot=observed.slot,
            pending_incarnation=observed.incarnation,
            seen=present,
            charge_number=jnp.where(present, observed.charge_number, 0),
            mass=jnp.where(present, observed.mass, 0.0),
            parent_hi=jnp.where(present, observed.parent_hi, _NO_IDENTITY_WORD),
            parent_lo=jnp.where(present, observed.parent_lo, _NO_IDENTITY_WORD),
            first_step=step,
            last_step=step,
            activations=present.astype(jnp.int32),
            property_transition=jnp.zeros((count,), dtype=jnp.bool_),
            duplicate=observed.duplicate,
            frame_offset=jnp.zeros((3,), dtype=jnp.float64),
        )
        if self.radiation is None:
            return state
        template = self._sample_trajectory(
            state,
            start[None],
            state.pending_position[None],
            state.pending_proper_velocity[None],
            jnp.zeros((1, count), dtype=jnp.bool_),
            float(self.radiation.plan.scale.elementary_charge),
        )
        # Every streamed chunk holds one sample, so the accumulator's static
        # sample capacity is one from the start and the recorder state keeps one
        # PyTree structure across accepted and rejected steps.
        radiation = dataclasses.replace(
            self.radiation.initialize(template), sample_capacity=1
        )
        return dataclasses.replace(state, radiation=radiation)

    def record(
        self,
        state: PICTrackRecorderState,
        species: tuple[PICSpeciesState, ...],
        time: Array,
        step_index: Array,
        /,
    ) -> PICTrackRecorderState:
        if not isinstance(state, PICTrackRecorderState):
            raise TypeError("state must be PICTrackRecorderState.")
        observed = self._observe(species)
        present = observed.present
        now = jnp.asarray(time, dtype=jnp.float64).reshape(())
        step = jnp.asarray(step_index, dtype=jnp.int32).reshape(())
        first = present & ~state.seen
        charge_number = jnp.where(first, observed.charge_number, state.charge_number)
        mass = jnp.where(first, observed.mass, state.mass)
        consistent = (
            present & (observed.charge_number == charge_number) & (observed.mass == mass)
        )
        continuing = state.pending_active & present
        # Sample k: time-centered u^k from u^{k−1/2} and the accepted u^{k+1/2}.
        centered = jnp.where(
            continuing[:, None],
            0.5 * (state.pending_proper_velocity + observed.proper_velocity),
            state.pending_proper_velocity,
        )
        dimension = species[self.species_blocks[0][0]].particles.position.shape[1]
        drifted = state.pending_position + (now - state.pending_time) * (
            self._drift_velocity(observed.proper_velocity)
        )
        unresolved = jnp.where(continuing[:, None], drifted, state.pending_position)[
            :, dimension:
        ]
        position = jnp.where(
            present[:, None],
            jnp.concatenate(
                (
                    observed.position[:, :dimension]
                    + state.frame_offset[None, :dimension],
                    unresolved,
                ),
                axis=1,
            ),
            state.pending_position,
        )
        buffer = state.buffer
        if buffer is not None and self.sample_capacity is not None:
            row = state.sample_count % self.sample_capacity
            buffer = PICTrackBuffer(
                buffer.times.at[row].set(state.pending_time),
                buffer.positions.at[row].set(state.pending_position),
                buffer.proper_velocities.at[row].set(centered),
                buffer.active.at[row].set(state.pending_active),
                buffer.slots.at[row].set(state.pending_slot),
                buffer.incarnations.at[row].set(state.pending_incarnation),
            )
        updated = PICTrackRecorderState(
            buffer=buffer,
            radiation=None,
            sample_count=state.sample_count + 1,
            pending_time=now,
            pending_position=position,
            pending_proper_velocity=jnp.where(
                present[:, None], observed.proper_velocity, state.pending_proper_velocity
            ),
            pending_active=consistent,
            pending_slot=observed.slot,
            pending_incarnation=observed.incarnation,
            seen=state.seen | present,
            charge_number=charge_number,
            mass=mass,
            parent_hi=jnp.where(first, observed.parent_hi, state.parent_hi),
            parent_lo=jnp.where(first, observed.parent_lo, state.parent_lo),
            first_step=jnp.where(first, step, state.first_step),
            last_step=jnp.where(present, step, state.last_step),
            activations=state.activations
            + (consistent & ~state.pending_active).astype(jnp.int32),
            property_transition=state.property_transition | (present & ~consistent),
            duplicate=state.duplicate | observed.duplicate,
            frame_offset=state.frame_offset,
        )
        if self.radiation is None or state.radiation is None:
            return updated
        sample = self._sample_trajectory(
            updated,
            state.pending_time[None],
            state.pending_position[None],
            centered[None],
            state.pending_active[None],
            float(self.radiation.plan.scale.elementary_charge),
        )
        # Lane charges are fixed at first sighting; a lane first seen now has no
        # emitted active sample yet, so updating the accumulator's lane charges
        # before folding the sample changes no earlier contribution.
        accumulator = dataclasses.replace(
            state.radiation, charges=sample.charges, multiplicities=sample.multiplicities
        )
        return dataclasses.replace(
            updated, radiation=self.radiation.accumulate(accumulator, sample)
        )

    # -- consumers -----------------------------------------------------------------

    def dropped_samples(self, state: PICTrackRecorderState, /) -> Array:
        """Emitted samples overwritten in the ring (zero without stored tracks)."""
        if self.sample_capacity is None:
            return jnp.zeros((), dtype=jnp.int32)
        return jnp.maximum(state.sample_count - self.sample_capacity, 0)

    def to_charged_trajectory(
        self, state: PICTrackRecorderState, scale: ElectromagneticScaleContract, /
    ) -> ChargedTrajectory:
        """Stored lanes in chronological order as a `ChargedTrajectory` in ``scale``.

        The trajectory has ``sample_capacity`` rows; rows not yet written are
        inactive and repeat the last written time. ``scale`` must share the
        recorder's dimensional scale and exact speed of light.
        """
        if not isinstance(state, PICTrackRecorderState):
            raise TypeError("state must be PICTrackRecorderState.")
        buffer = state.buffer
        capacity = self.sample_capacity
        if buffer is None or capacity is None:
            raise ValueError("This track recorder stores no tracks.")
        elementary_charge = _scale_elementary_charge(self.relativity, scale)
        written = jnp.minimum(state.sample_count, capacity)
        start = jnp.where(state.sample_count > capacity, state.sample_count % capacity, 0)
        rows = jnp.arange(capacity, dtype=jnp.int32)
        order = (start + rows) % capacity
        valid = rows < written
        times = buffer.times[order]
        times = jnp.where(valid, times, times[jnp.maximum(written - 1, 0)])
        positions = buffer.positions[order]
        if self.overflow == "refuse":
            positions = eqx.error_if(
                positions,
                state.sample_count > capacity,
                "The track ring overflowed and lost its earliest samples; raise "
                "sample_capacity or declare overflow='keep-latest'.",
            )
        return self._sample_trajectory(
            state,
            times,
            positions,
            buffer.proper_velocities[order],
            buffer.active[order] & valid[:, None],
            elementary_charge,
        )

    def finalize_radiation(
        self, state: PICTrackRecorderState, /
    ) -> TrajectoryRadiationResult:
        """Far-field spectrum of every sample emitted so far."""
        if not isinstance(state, PICTrackRecorderState):
            raise TypeError("state must be PICTrackRecorderState.")
        if self.radiation is None or state.radiation is None:
            raise ValueError("This track recorder streams no radiation.")
        return self.radiation.finalize(state.radiation)


__all__ = [
    "PICTrackBuffer",
    "PICTrackRecorder",
    "PICTrackRecorderState",
    "TrackOverflowPolicy",
]
