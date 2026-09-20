#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity relativistic dark-radiation packet transport."""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from ..equations._dark_radiation_moments import DarkRadiationFourForce
from ._dark_sector_epoch_runtime import (
    DarkSectorEpochPlan,
    DarkSectorEpochState,
    decode_content_id,
    empty_dark_sector_epoch_state,
    encode_content_id,
    encode_content_ids,
    replace_dark_sector_conservation,
    replace_dark_sector_pool,
)


_CHECKPOINT_FORMAT = "phydrax-dark-radiation-packets"
_EPOCH_RADIATION_WIDTH = 20


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty stripped string.")
    return value


def _coherency_from_stokes(stokes: Array, /) -> Array:
    intensity, q, u, v = (stokes[..., index] for index in range(4))
    dtype = jnp.result_type(stokes.dtype, jnp.complex64)
    result = jnp.zeros(stokes.shape[:-1] + (2, 2), dtype=dtype)
    result = result.at[..., 0, 0].set(0.5 * (intensity + q))
    result = result.at[..., 1, 1].set(0.5 * (intensity - q))
    result = result.at[..., 0, 1].set(0.5 * (u - 1j * v))
    return result.at[..., 1, 0].set(0.5 * (u + 1j * v))


def _state_select(
    predicate: Array,
    candidate: DarkRadiationPacketState,
    source: DarkRadiationPacketState,
    /,
) -> DarkRadiationPacketState:
    choose = lambda proposed, previous: jnp.where(predicate, proposed, previous)
    return DarkRadiationPacketState(
        choose(candidate.global_ids, source.global_ids),
        choose(candidate.species_ids, source.species_ids),
        choose(candidate.parent_ids, source.parent_ids),
        choose(candidate.source_event_ids, source.source_event_ids),
        choose(candidate.generation, source.generation),
        choose(candidate.comoving_position, source.comoving_position),
        choose(candidate.tetrad_four_momentum, source.tetrad_four_momentum),
        choose(candidate.frequency_group, source.frequency_group),
        choose(candidate.stokes, source.stokes),
        choose(candidate.coherency, source.coherency),
        choose(candidate.optical_depth_threshold, source.optical_depth_threshold),
        choose(candidate.optical_depth_accumulator, source.optical_depth_accumulator),
        choose(candidate.rng_keys, source.rng_keys),
        choose(candidate.rng_counters, source.rng_counters),
        choose(candidate.owner, source.owner),
        choose(candidate.status, source.status),
        choose(candidate.weight, source.weight),
        choose(candidate.active_mask, source.active_mask),
        choose(candidate.frame_token, source.frame_token),
        choose(candidate.observer_coordinates, source.observer_coordinates),
        choose(candidate.coordinate_time, source.coordinate_time),
        choose(candidate.frame_realization_words, source.frame_realization_words),
        choose(candidate.epoch_manifest_words, source.epoch_manifest_words),
        choose(candidate.scale_factor, source.scale_factor),
        choose(candidate.epoch_sequence, source.epoch_sequence),
        choose(candidate.valid, source.valid),
        plan_id=source.plan_id,
        frame_id=source.frame_id,
        unit_contract_id=source.unit_contract_id,
        epoch_plan_id=source.epoch_plan_id,
    )


class DarkRadiationPacketStatus(IntEnum):
    INACTIVE = 0
    ACTIVE = 1
    ABSORBED = 2
    ESCAPED = 3
    TERMINATED = 4


class DarkRadiationInteractionKind(IntEnum):
    NONE = 0
    ABSORPTION = 1
    SCATTERING = 2
    EMISSION = 3
    ESCAPE = 4


class DarkRadiationPacketState(StrictModule, NonTrainableState):
    """Fixed packet arrays for one committed finite transport epoch."""

    global_ids: Array
    species_ids: Array
    parent_ids: Array
    source_event_ids: Array
    generation: Array
    comoving_position: Array
    tetrad_four_momentum: Array
    frequency_group: Array
    stokes: Array
    coherency: Array
    optical_depth_threshold: Array
    optical_depth_accumulator: Array
    rng_keys: Array
    rng_counters: Array
    owner: Array
    status: Array
    weight: Array
    active_mask: Array
    frame_token: Array
    observer_coordinates: Array
    coordinate_time: Array
    frame_realization_words: Array
    epoch_manifest_words: Array
    scale_factor: Array
    epoch_sequence: Array
    valid: Array
    plan_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    epoch_plan_id: str = eqx.field(static=True)

    @property
    def frame_realization_id(self) -> str:
        return decode_content_id(self.frame_realization_words)

    @property
    def epoch_manifest_id(self) -> str:
        return decode_content_id(self.epoch_manifest_words)


class DarkRadiationPacketEvents(StrictModule, NonTrainableState):
    event_ids: Array
    packet_ids: Array
    parent_packet_ids: Array
    child_packet_ids: Array
    interaction_kind: Array
    four_momentum_before: Array
    four_momentum_after: Array
    four_momentum_to_matter: Array
    mask: Array
    epoch_sequence: Array


class DarkRadiationPacketStepEvidence(StrictModule, NonTrainableState):
    source_valid: Array
    frame_compatible: Array
    coefficients_valid: Array
    event_count: Array
    event_capacity_available: Array
    escaped_count: Array
    mass_shell_residual: Array
    polarization_residual: Array
    four_force_residual: Array
    rolled_back: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DarkRadiationPacketStepResult(StrictModule, NonTrainableState):
    source_state: DarkRadiationPacketState
    candidate_state: DarkRadiationPacketState
    accepted_state: DarkRadiationPacketState
    events: DarkRadiationPacketEvents
    exchange: DarkRadiationFourForce
    evidence: DarkRadiationPacketStepEvidence
    successful: Array


class DarkRadiationPacketAdmissionEvidence(StrictModule, NonTrainableState):
    source_valid: Array
    identities_unique: Array
    capacity_available: Array
    packet_valid: Array
    selected_count: Array
    four_momentum_input: Array
    four_momentum_stored: Array
    four_momentum_defect: Array
    rolled_back: Array
    successful: Array


class DarkRadiationPacketAdmissionResult(StrictModule, NonTrainableState):
    source_state: DarkRadiationPacketState
    candidate_state: DarkRadiationPacketState
    accepted_state: DarkRadiationPacketState
    evidence: DarkRadiationPacketAdmissionEvidence
    successful: Array


class DarkRadiationPacketPlan(StrictModule, NonTrainableState):
    """JAX-safe bounded kernel; semantic unboundedness is an epoch chain."""

    units: RelativisticUnitContract
    epoch_plan: DarkSectorEpochPlan = eqx.field(static=True)
    group_edges: Array
    domain_minimum: Array
    domain_maximum: Array
    capacity: int = eqx.field(static=True)
    event_capacity: int = eqx.field(static=True)
    owner_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    epoch_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        capacity: int,
        event_capacity: int,
        units: RelativisticUnitContract,
        group_edges: ArrayLike,
        domain_minimum: ArrayLike,
        domain_maximum: ArrayLike,
        /,
        *,
        epoch_plan: DarkSectorEpochPlan,
        owner_count: int = 1,
    ):
        packets = int(capacity)
        events = int(event_capacity)
        owners = int(owner_count)
        if packets <= 0 or events <= 0 or owners <= 0:
            raise ValueError("Packet, event, and owner capacities must be positive.")
        if not isinstance(units, RelativisticUnitContract):
            raise TypeError("units must be RelativisticUnitContract.")
        if not isinstance(epoch_plan, DarkSectorEpochPlan):
            raise TypeError("epoch_plan must be DarkSectorEpochPlan.")
        if (
            epoch_plan.radiation_capacity < packets
            or epoch_plan.event_capacity < events
            or epoch_plan.radiation_width < _EPOCH_RADIATION_WIDTH
        ):
            raise ValueError(
                "Dark-sector epoch radiation/event capacity or radiation width cannot contain the packet kernel."
            )
        edges = jnp.asarray(group_edges)
        lower = jnp.asarray(domain_minimum, dtype=edges.dtype)
        upper = jnp.asarray(domain_maximum, dtype=edges.dtype)
        if (
            edges.ndim != 1
            or edges.size < 2
            or lower.shape != (3,)
            or upper.shape != (3,)
        ):
            raise ValueError("Packet groups and three-dimensional domain are malformed.")
        if not bool(np.all(np.diff(np.asarray(edges)) > 0.0)) or not bool(
            np.all(np.asarray(upper) > np.asarray(lower))
        ):
            raise ValueError("Packet group edges and domain bounds must increase.")
        epoch_id = epoch_plan.plan_id
        self.units = units
        self.group_edges = edges
        self.domain_minimum = lower
        self.domain_maximum = upper
        self.capacity = packets
        self.event_capacity = events
        self.owner_count = owners
        self.epoch_plan = epoch_plan
        self.epoch_plan_id = epoch_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dark-radiation-packet-plan",
                "capacity": packets,
                "event_capacity": events,
                "owner_count": owners,
                "unit_contract": units.contract_id,
                "group_edges": np.asarray(edges).tolist(),
                "domain_minimum": np.asarray(lower).tolist(),
                "domain_maximum": np.asarray(upper).tolist(),
                "epoch_plan": epoch_id,
                "packet_arrays": "fixed-capacity",
                "continuation": "durable-epoch-chain",
            }
        )

    @property
    def physical_light_speed(self) -> float:
        return float(self.units.speed_of_light)

    def empty(
        self,
        frame: LocalRelativisticFramePlan,
        /,
        *,
        epoch_sequence: int = 0,
        epoch_manifest_id: str,
        dtype=jnp.float64,
    ) -> DarkRadiationPacketState:
        self._check_frame(frame)
        dtype_ = jnp.dtype(dtype)
        if not jnp.issubdtype(dtype_, jnp.floating):
            raise TypeError("Dark-radiation packet state requires floating dtype.")
        shape = (self.capacity,)
        return DarkRadiationPacketState(
            jnp.full(shape, -1, dtype=jnp.int64),
            jnp.full(shape, -1, dtype=jnp.int32),
            jnp.full(shape + (2,), -1, dtype=jnp.int64),
            jnp.full(shape, -1, dtype=jnp.int64),
            jnp.full(shape, -1, dtype=jnp.int32),
            jnp.zeros(shape + (3,), dtype=dtype_),
            jnp.zeros(shape + (4,), dtype=dtype_),
            jnp.full(shape, -1, dtype=jnp.int32),
            jnp.zeros(shape + (4,), dtype=dtype_),
            jnp.zeros(
                shape + (2, 2),
                dtype=jnp.complex64 if dtype_ == jnp.float32 else jnp.complex128,
            ),
            jnp.zeros(shape, dtype=dtype_),
            jnp.zeros(shape, dtype=dtype_),
            jnp.zeros(shape + (2,), dtype=jnp.uint32),
            jnp.zeros(shape, dtype=jnp.int64),
            jnp.full(shape, -1, dtype=jnp.int32),
            jnp.full(shape, int(DarkRadiationPacketStatus.INACTIVE), dtype=jnp.int8),
            jnp.zeros(shape, dtype=dtype_),
            jnp.zeros(shape, dtype=jnp.bool_),
            jnp.asarray(frame.frame_token),
            jnp.asarray(frame.observer_coordinates, dtype=dtype_).reshape((4,)),
            jnp.asarray(frame.time).reshape(()).astype(dtype_),
            jnp.asarray(encode_content_id(frame.realization_id())),
            jnp.asarray(
                encode_content_id(_identifier(epoch_manifest_id, "epoch_manifest_id"))
            ),
            jnp.asarray(frame.scale_factor).reshape(()).astype(dtype_),
            jnp.asarray(epoch_sequence, dtype=jnp.int64),
            jnp.asarray(True),
            plan_id=self.plan_id,
            frame_id=frame.frame_id,
            unit_contract_id=self.units.contract_id,
            epoch_plan_id=self.epoch_plan_id,
        )

    def valid(self, state: DarkRadiationPacketState, /) -> Array:
        self._check_state_shapes(state)
        active = state.active_mask
        inactive = ~active
        sentinel = (
            (state.global_ids == -1)
            & (state.species_ids == -1)
            & jnp.all(state.parent_ids == -1, axis=-1)
            & (state.source_event_ids == -1)
            & (state.generation == -1)
            & jnp.all(state.comoving_position == 0.0, axis=-1)
            & jnp.all(state.tetrad_four_momentum == 0.0, axis=-1)
            & (state.frequency_group == -1)
            & jnp.all(state.stokes == 0.0, axis=-1)
            & jnp.all(state.coherency == 0.0, axis=(-2, -1))
            & (state.optical_depth_threshold == 0.0)
            & (state.optical_depth_accumulator == 0.0)
            & jnp.all(state.rng_keys == 0, axis=-1)
            & (state.rng_counters == 0)
            & (state.owner == -1)
            & (state.status == int(DarkRadiationPacketStatus.INACTIVE))
            & (state.weight == 0.0)
        )
        sorted_ids = jnp.sort(
            jnp.where(active, state.global_ids, jnp.iinfo(state.global_ids.dtype).max)
        )
        inactive_id = jnp.iinfo(state.global_ids.dtype).max
        unique = jnp.all(
            (sorted_ids[1:] != sorted_ids[:-1]) | (sorted_ids[1:] == inactive_id)
        )
        four = state.tetrad_four_momentum
        shell_scale = jnp.maximum(jnp.max(jnp.abs(four), axis=-1) ** 2, 1.0)
        shell_residual = jnp.abs(self.units.mass_shell_residual(four, 0.0))
        shell = shell_residual <= 128.0 * jnp.finfo(four.dtype).eps * shell_scale
        stokes_bound = jnp.sqrt(jnp.sum(state.stokes[:, 1:] ** 2, axis=-1))
        polarization = (state.stokes[:, 0] >= stokes_bound) & jnp.all(
            jnp.abs(state.coherency - _coherency_from_stokes(state.stokes))
            <= 128.0
            * jnp.finfo(state.stokes.dtype).eps
            * jnp.maximum(jnp.abs(state.coherency), 1.0),
            axis=(-2, -1),
        )
        active_valid = (
            (state.global_ids >= 0)
            & (state.species_ids >= 0)
            & jnp.all(state.parent_ids >= -1, axis=-1)
            & (state.source_event_ids >= 0)
            & (state.generation >= 0)
            & (four[:, 0] > 0.0)
            & shell
            & (state.frequency_group >= 0)
            & (state.frequency_group < self.group_edges.size - 1)
            & polarization
            & (state.optical_depth_threshold > 0.0)
            & (state.optical_depth_accumulator >= 0.0)
            & (state.owner >= 0)
            & (state.owner < self.owner_count)
            & (state.status == int(DarkRadiationPacketStatus.ACTIVE))
            & (state.weight > 0.0)
        )
        finite = (
            jnp.all(jnp.isfinite(state.comoving_position))
            & jnp.all(jnp.isfinite(state.tetrad_four_momentum))
            & jnp.all(jnp.isfinite(state.observer_coordinates))
            & jnp.all(jnp.isfinite(state.stokes))
            & jnp.all(jnp.isfinite(state.coherency))
            & jnp.all(jnp.isfinite(state.optical_depth_threshold))
            & jnp.all(jnp.isfinite(state.optical_depth_accumulator))
            & jnp.all(jnp.isfinite(state.weight))
            & jnp.isfinite(state.coordinate_time)
            & jnp.isfinite(state.scale_factor)
            & (state.scale_factor > 0.0)
        )
        return (
            state.valid
            & unique
            & finite
            & jnp.all(jnp.where(active, active_valid, sentinel))
            & jnp.all(inactive | (state.status == int(DarkRadiationPacketStatus.ACTIVE)))
        )

    def admit(
        self,
        state: DarkRadiationPacketState,
        global_ids: ArrayLike,
        species_ids: ArrayLike,
        parent_ids: ArrayLike,
        source_event_ids: ArrayLike,
        generation: ArrayLike,
        comoving_position: ArrayLike,
        tetrad_four_momentum: ArrayLike,
        frequency_group: ArrayLike,
        stokes: ArrayLike,
        optical_depth_threshold: ArrayLike,
        rng_keys: ArrayLike,
        owner: ArrayLike,
        weight: ArrayLike,
        admission_mask: ArrayLike,
        /,
    ) -> DarkRadiationPacketAdmissionResult:
        """Atomically admit one fixed candidate batch, preserving input order."""

        self._check_state_shapes(state)
        identifiers = jnp.asarray(global_ids, dtype=jnp.int64)
        species = jnp.asarray(species_ids, dtype=jnp.int32)
        parents = jnp.asarray(parent_ids, dtype=jnp.int64)
        events = jnp.asarray(source_event_ids, dtype=jnp.int64)
        generations = jnp.asarray(generation, dtype=jnp.int32)
        positions = jnp.asarray(comoving_position, dtype=state.weight.dtype)
        momenta = jnp.asarray(tetrad_four_momentum, dtype=state.weight.dtype)
        groups = jnp.asarray(frequency_group, dtype=jnp.int32)
        polarization = jnp.asarray(stokes, dtype=state.weight.dtype)
        thresholds = jnp.asarray(optical_depth_threshold, dtype=state.weight.dtype)
        raw_keys = jnp.asarray(rng_keys)
        keys = (
            jr.key_data(raw_keys)
            if jax.dtypes.issubdtype(raw_keys.dtype, jax.dtypes.prng_key)
            else jnp.asarray(raw_keys, dtype=jnp.uint32)
        )
        owners = jnp.asarray(owner, dtype=jnp.int32)
        weights = jnp.asarray(weight, dtype=state.weight.dtype)
        mask = jnp.asarray(admission_mask, dtype=jnp.bool_)
        batch = identifiers.shape[0]
        expected = (batch,)
        if (
            identifiers.ndim != 1
            or species.shape != expected
            or parents.shape != expected + (2,)
            or events.shape != expected
            or generations.shape != expected
            or positions.shape != expected + (3,)
            or momenta.shape != expected + (4,)
            or groups.shape != expected
            or polarization.shape != expected + (4,)
            or thresholds.shape != expected
            or keys.shape != expected + (2,)
            or owners.shape != expected
            or weights.shape != expected
            or mask.shape != expected
        ):
            raise ValueError("Dark-radiation admission batch shapes do not align.")
        free_slots = jnp.nonzero(~state.active_mask, size=self.capacity, fill_value=0)[0]
        requested_rank = jnp.cumsum(mask.astype(jnp.int32)) - 1
        selected_count = jnp.sum(mask)
        capacity_available = selected_count <= jnp.sum(~state.active_mask)
        clipped_rank = jnp.clip(requested_rank, 0, self.capacity - 1)
        slots = free_slots[clipped_rank]
        existing_duplicate = jax.vmap(
            lambda value: jnp.any(state.active_mask & (state.global_ids == value))
        )(identifiers)
        pair_duplicate = (
            (identifiers[:, None] == identifiers[None, :]) & mask[:, None] & mask[None, :]
        )
        identities_unique = ~jnp.any(
            pair_duplicate & ~jnp.eye(batch, dtype=jnp.bool_)
        ) & ~jnp.any(mask & existing_duplicate)
        shell = self.units.mass_shell_admissible(momenta, 0.0)
        stokes_norm = jnp.sqrt(jnp.sum(polarization[:, 1:] ** 2, axis=-1))
        packet_valid = jnp.all(
            ~mask
            | (
                (identifiers >= 0)
                & (species >= 0)
                & (events >= 0)
                & (generations >= 0)
                & shell
                & (momenta[:, 0] > 0.0)
                & (groups >= 0)
                & (groups < self.group_edges.size - 1)
                & (polarization[:, 0] >= stokes_norm)
                & (thresholds > 0.0)
                & (owners >= 0)
                & (owners < self.owner_count)
                & (weights > 0.0)
                & jnp.all(jnp.isfinite(positions), axis=-1)
                & jnp.all(jnp.isfinite(momenta), axis=-1)
            )
        )
        source_valid = self.valid(state)
        successful = source_valid & identities_unique & capacity_available & packet_valid

        def assign(array, values):
            def assign_one(index, current):
                return jax.lax.cond(
                    mask[index],
                    lambda operand: operand.at[slots[index]].set(values[index]),
                    lambda operand: operand,
                    current,
                )

            return jax.lax.fori_loop(0, batch, assign_one, array)

        candidate = DarkRadiationPacketState(
            assign(state.global_ids, identifiers),
            assign(state.species_ids, species),
            assign(state.parent_ids, parents),
            assign(state.source_event_ids, events),
            assign(state.generation, generations),
            assign(state.comoving_position, positions),
            assign(state.tetrad_four_momentum, momenta),
            assign(state.frequency_group, groups),
            assign(state.stokes, polarization),
            assign(state.coherency, _coherency_from_stokes(polarization)),
            assign(state.optical_depth_threshold, thresholds),
            assign(state.optical_depth_accumulator, jnp.zeros_like(thresholds)),
            assign(state.rng_keys, keys),
            assign(state.rng_counters, jnp.zeros_like(identifiers)),
            assign(state.owner, owners),
            assign(
                state.status,
                jnp.full(expected, int(DarkRadiationPacketStatus.ACTIVE), dtype=jnp.int8),
            ),
            assign(state.weight, weights),
            assign(state.active_mask, mask),
            state.frame_token,
            state.observer_coordinates,
            state.coordinate_time,
            state.frame_realization_words,
            state.epoch_manifest_words,
            state.scale_factor,
            state.epoch_sequence,
            successful,
            plan_id=state.plan_id,
            frame_id=state.frame_id,
            unit_contract_id=state.unit_contract_id,
            epoch_plan_id=state.epoch_plan_id,
        )
        candidate_valid = self.valid(candidate)
        successful = successful & candidate_valid
        accepted = _state_select(successful, candidate, state)
        stored = jnp.sum(
            jnp.where(mask[:, None], momenta * weights[:, None], 0.0), axis=0
        )
        evidence = DarkRadiationPacketAdmissionEvidence(
            source_valid,
            identities_unique,
            capacity_available,
            packet_valid,
            selected_count,
            stored,
            stored,
            jnp.zeros_like(stored),
            ~successful,
            successful,
        )
        return DarkRadiationPacketAdmissionResult(
            state, candidate, accepted, evidence, successful
        )

    def advance(
        self,
        state: DarkRadiationPacketState,
        start_frame: LocalRelativisticFramePlan,
        end_frame: LocalRelativisticFramePlan,
        absorption_opacity: ArrayLike,
        scattering_opacity: ArrayLike,
        scattering_mueller: ArrayLike,
        /,
        *,
        next_epoch_manifest_id: str,
        end_frame_realization_id: str,
    ) -> DarkRadiationPacketStepResult:
        """Advance one FLRW segment with at most one material event per packet.

        Opacities are physical inverse-length coefficients at the segment. A
        capacity overflow rejects the entire segment; continuation belongs to a
        later durable epoch rather than to a dynamically growing device array.
        """

        self._check_state_shapes(state)
        self._check_frame(start_frame)
        self._check_frame(end_frame)
        if state.frame_id != start_frame.frame_id:
            raise ValueError("Packet source state is not bound to start_frame.")
        if start_frame.units.contract_id != end_frame.units.contract_id:
            raise ValueError("Packet endpoint frames use different unit contracts.")
        absorption = jnp.asarray(absorption_opacity, dtype=state.weight.dtype)
        scattering = jnp.asarray(scattering_opacity, dtype=state.weight.dtype)
        mueller = jnp.asarray(scattering_mueller, dtype=state.weight.dtype)
        if absorption.shape != (self.capacity,) or scattering.shape != (self.capacity,):
            raise ValueError("Packet opacity arrays must match packet capacity.")
        if mueller.shape not in ((4, 4), (self.capacity, 4, 4)):
            raise ValueError(
                "Packet Mueller matrix must be one matrix or one per packet."
            )
        if mueller.shape == (4, 4):
            mueller = jnp.broadcast_to(mueller, (self.capacity, 4, 4))
        start_time = jnp.asarray(start_frame.time).reshape(()).astype(state.weight.dtype)
        end_time = jnp.asarray(end_frame.time).reshape(()).astype(state.weight.dtype)
        start_scale = (
            jnp.asarray(start_frame.scale_factor).reshape(()).astype(state.weight.dtype)
        )
        end_scale = (
            jnp.asarray(end_frame.scale_factor).reshape(()).astype(state.weight.dtype)
        )
        dt = end_time - start_time
        end_realization = _identifier(
            end_frame_realization_id, "end_frame_realization_id"
        )
        frame_compatible = jnp.asarray(
            state.frame_id == start_frame.frame_id
            and start_frame.frame_id == end_frame.frame_id
            and start_frame.units.contract_id == self.units.contract_id
            and end_frame.units.contract_id == self.units.contract_id
            and start_frame.geometry.topology_id == end_frame.geometry.topology_id
            and start_frame.geometry.geometry_lineage_id
            == end_frame.geometry.geometry_lineage_id
        ) & (
            jnp.all(start_frame.admissible)
            & jnp.all(end_frame.admissible)
            & (state.frame_token == start_frame.frame_token)
            & (state.coordinate_time == start_time)
            & (state.scale_factor == start_scale)
            & jnp.all(
                state.observer_coordinates
                == jnp.asarray(start_frame.observer_coordinates).reshape((4,))
            )
        )
        coefficients_valid = (
            jnp.all(jnp.isfinite(absorption))
            & jnp.all(jnp.isfinite(scattering))
            & jnp.all(absorption >= 0.0)
            & jnp.all(scattering >= 0.0)
            & jnp.all(jnp.isfinite(mueller))
            & jnp.isfinite(dt)
            & (dt > 0.0)
            & (start_scale > 0.0)
            & (end_scale > 0.0)
        )
        source_valid = self.valid(state)
        active = state.active_mask
        redshift = start_scale / end_scale
        redshifted_four = state.tetrad_four_momentum * redshift
        spatial = redshifted_four[:, 1:]
        spatial_norm = jnp.sqrt(jnp.sum(spatial**2, axis=-1))
        direction = spatial / jnp.where(spatial_norm > 0.0, spatial_norm, 1.0)[:, None]
        distance = jnp.asarray(self.physical_light_speed, dtype=state.weight.dtype) * dt
        midpoint_scale = jnp.sqrt(start_scale * end_scale)
        position = state.comoving_position + direction * distance / midpoint_scale
        total_opacity = absorption + scattering
        accumulator = state.optical_depth_accumulator + total_opacity * distance
        interacts = (
            active
            & (total_opacity > 0.0)
            & (accumulator >= state.optical_depth_threshold)
        )

        folded = jax.vmap(jr.fold_in)(
            state.rng_keys, state.rng_counters.astype(jnp.uint32)
        )
        choice_keys = jax.vmap(lambda key: jr.fold_in(key, 0))(folded)
        polar_keys = jax.vmap(lambda key: jr.fold_in(key, 1))(folded)
        azimuth_keys = jax.vmap(lambda key: jr.fold_in(key, 2))(folded)
        threshold_keys = jax.vmap(lambda key: jr.fold_in(key, 3))(folded)
        choice = jax.vmap(lambda key: jr.uniform(key, ()))(choice_keys)
        cosine = 2.0 * jax.vmap(lambda key: jr.uniform(key, ()))(polar_keys) - 1.0
        azimuth = 2.0 * jnp.pi * jax.vmap(lambda key: jr.uniform(key, ()))(azimuth_keys)
        sine = jnp.sqrt(jnp.maximum(1.0 - cosine**2, 0.0))
        scattered_direction = jnp.stack(
            (sine * jnp.cos(azimuth), sine * jnp.sin(azimuth), cosine), axis=-1
        )
        scatter_probability = scattering / jnp.where(
            total_opacity > 0.0, total_opacity, 1.0
        )
        scatters = interacts & (choice < scatter_probability)
        absorbs = interacts & ~scatters
        scattered_four = redshifted_four.at[:, 1:].set(
            redshifted_four[:, :1] * scattered_direction
        )
        after_four = jnp.where(scatters[:, None], scattered_four, redshifted_four)
        after_four = jnp.where(absorbs[:, None], 0.0, after_four)
        transformed_stokes = ein.contract("nij,nj->ni", mueller, state.stokes)
        output_stokes = jnp.where(scatters[:, None], transformed_stokes, state.stokes)
        polarization_norm = jnp.sqrt(jnp.sum(output_stokes[:, 1:] ** 2, axis=-1))
        polarization_valid = output_stokes[:, 0] >= polarization_norm
        output_stokes = jnp.where(absorbs[:, None], 0.0, output_stokes)
        next_threshold = jax.vmap(lambda key: jr.exponential(key, ()))(threshold_keys)
        output_threshold = jnp.where(
            scatters, next_threshold, state.optical_depth_threshold
        )
        output_accumulator = jnp.where(interacts, 0.0, accumulator)
        output_status = jnp.where(
            absorbs,
            int(DarkRadiationPacketStatus.ABSORBED),
            state.status,
        ).astype(jnp.int8)
        output_active = active & ~absorbs
        escaped = output_active & (
            jnp.any(position < self.domain_minimum, axis=-1)
            | jnp.any(position >= self.domain_maximum, axis=-1)
        )
        event_count = jnp.sum(interacts) + jnp.sum(escaped)
        event_capacity_available = event_count <= self.event_capacity
        output_active = output_active & ~escaped
        output_status = jnp.where(
            escaped,
            int(DarkRadiationPacketStatus.ESCAPED),
            output_status,
        ).astype(jnp.int8)
        terminal = ~output_active
        output_ids = jnp.where(terminal, -1, state.global_ids)
        output_species = jnp.where(terminal, -1, state.species_ids)
        output_parents = jnp.where(terminal[:, None], -1, state.parent_ids)
        output_source_events = jnp.where(terminal, -1, state.source_event_ids)
        output_generation = jnp.where(terminal, -1, state.generation)
        output_positions = jnp.where(terminal[:, None], 0.0, position)
        output_four = jnp.where(terminal[:, None], 0.0, after_four)
        output_groups = jnp.where(terminal, -1, state.frequency_group)
        output_stokes = jnp.where(terminal[:, None], 0.0, output_stokes)
        output_threshold = jnp.where(terminal, 0.0, output_threshold)
        output_accumulator = jnp.where(terminal, 0.0, output_accumulator)
        output_keys = jnp.where(terminal[:, None], 0, state.rng_keys)
        output_counters = jnp.where(
            terminal, 0, state.rng_counters + interacts.astype(jnp.int64)
        )
        output_owner = jnp.where(terminal, -1, state.owner)
        output_status_clean = jnp.where(
            terminal,
            int(DarkRadiationPacketStatus.INACTIVE),
            output_status,
        ).astype(jnp.int8)
        output_weight = jnp.where(terminal, 0.0, state.weight)
        candidate = DarkRadiationPacketState(
            output_ids,
            output_species,
            output_parents,
            output_source_events,
            output_generation,
            output_positions,
            output_four,
            output_groups,
            output_stokes,
            _coherency_from_stokes(output_stokes),
            output_threshold,
            output_accumulator,
            output_keys,
            output_counters,
            output_owner,
            output_status_clean,
            output_weight,
            output_active,
            jnp.asarray(end_frame.frame_token),
            jnp.asarray(end_frame.observer_coordinates).reshape((4,)),
            end_time,
            jnp.asarray(encode_content_id(end_realization)),
            jnp.asarray(
                encode_content_id(
                    _identifier(next_epoch_manifest_id, "next_epoch_manifest_id")
                )
            ),
            end_scale,
            state.epoch_sequence + 1,
            jnp.asarray(True),
            plan_id=state.plan_id,
            frame_id=end_frame.frame_id,
            unit_contract_id=state.unit_contract_id,
            epoch_plan_id=state.epoch_plan_id,
        )
        candidate_valid = self.valid(candidate)
        successful = (
            source_valid
            & frame_compatible
            & coefficients_valid
            & event_capacity_available
            & jnp.all(~scatters | polarization_valid)
            & candidate_valid
        )
        accepted = _state_select(successful, candidate, state)

        stable_event_id = (
            state.global_ids * jnp.asarray(2_000_006, dtype=jnp.int64)
            + 2 * state.rng_counters
        )
        flat_mask = jnp.concatenate((interacts, escaped))
        flat_event_ids = jnp.concatenate((stable_event_id, stable_event_id + 1))
        flat_packet_ids = jnp.concatenate((state.global_ids, state.global_ids))
        flat_child_ids = jnp.concatenate(
            (
                jnp.where(scatters, state.global_ids, -1),
                jnp.full_like(state.global_ids, -1),
            )
        )
        flat_kind = jnp.concatenate(
            (
                jnp.where(
                    scatters,
                    int(DarkRadiationInteractionKind.SCATTERING),
                    int(DarkRadiationInteractionKind.ABSORPTION),
                ),
                jnp.full(
                    (self.capacity,),
                    int(DarkRadiationInteractionKind.ESCAPE),
                    dtype=jnp.int8,
                ),
            )
        ).astype(jnp.int8)
        flat_before = jnp.concatenate((redshifted_four, after_four), axis=0)
        flat_after = jnp.concatenate((after_four, jnp.zeros_like(after_four)), axis=0)
        flat_to_matter = jnp.concatenate(
            (
                (redshifted_four - after_four) * state.weight[:, None],
                jnp.zeros_like(after_four),
            ),
            axis=0,
        )
        source_index = jnp.nonzero(flat_mask, size=self.event_capacity, fill_value=0)[0]
        valid_event = jnp.arange(self.event_capacity) < event_count
        event_ids = jnp.where(valid_event, flat_event_ids[source_index], -1)
        packet_ids = jnp.where(valid_event, flat_packet_ids[source_index], -1)
        child_ids = jnp.where(valid_event, flat_child_ids[source_index], -1)
        kind = jnp.where(
            valid_event,
            flat_kind[source_index],
            int(DarkRadiationInteractionKind.NONE),
        ).astype(jnp.int8)
        before = jnp.where(valid_event[:, None], flat_before[source_index], 0.0)
        after = jnp.where(valid_event[:, None], flat_after[source_index], 0.0)
        to_matter = jnp.where(valid_event[:, None], flat_to_matter[source_index], 0.0)
        event_records = DarkRadiationPacketEvents(
            event_ids,
            packet_ids,
            packet_ids,
            child_ids,
            kind,
            before,
            after,
            to_matter,
            valid_event & successful,
            jnp.broadcast_to(state.epoch_sequence + 1, (self.event_capacity,)),
        )
        interaction_delta = jnp.sum(
            jnp.where(
                interacts[:, None],
                (after_four - redshifted_four) * state.weight[:, None],
                0.0,
            ),
            axis=0,
        )
        radiation_force = jnp.where(successful, interaction_delta / dt, 0.0)
        exchange = DarkRadiationFourForce.paired(
            radiation_force,
            end_time,
            end_frame.frame_token,
            source_state_id=next_epoch_manifest_id,
            frame_id=end_frame.frame_id,
            frame_realization_id=end_realization,
            unit_contract_id=self.units.contract_id,
        )
        mass_shell = self.units.mass_shell_residual(after_four, 0.0)
        polarization_residual = jnp.maximum(polarization_norm - output_stokes[:, 0], 0.0)
        evidence = DarkRadiationPacketStepEvidence(
            source_valid,
            frame_compatible,
            coefficients_valid,
            event_count,
            event_capacity_available,
            jnp.sum(escaped),
            mass_shell,
            polarization_residual,
            exchange.balance_residual,
            ~successful,
            successful,
            self.plan_id,
        )
        return DarkRadiationPacketStepResult(
            state, candidate, accepted, event_records, exchange, evidence, successful
        )

    def to_dark_sector_epoch_state(
        self,
        state: DarkRadiationPacketState,
        /,
        *,
        parent_epoch_manifest_id: str | None,
    ) -> DarkSectorEpochState:
        """Pack the resident packet frontier into the shared durable epoch owner.

        The packet checkpoint remains the exact restart payload. This fixed-width
        epoch view is the repository/continuation frontier and conservation
        ledger; it never grows with event history.
        """

        self._check_state_shapes(state)
        if not bool(np.asarray(self.valid(state))):
            raise ValueError("Only a valid packet state can enter a dark-sector epoch.")
        epoch_sequence = int(np.asarray(state.epoch_sequence))
        epoch = empty_dark_sector_epoch_state(
            self.epoch_plan,
            epoch_sequence=epoch_sequence,
            parent_epoch_manifest_id=parent_epoch_manifest_id,
        )
        active = np.asarray(state.active_mask, dtype=np.bool_)
        content_ids = tuple(
            canonical_fingerprint(
                {
                    "kind": "dark-radiation-packet-frontier",
                    "packet_plan": self.plan_id,
                    "global_id": int(np.asarray(state.global_ids)[index]),
                    "generation": int(np.asarray(state.generation)[index]),
                    "source_event_id": int(np.asarray(state.source_event_ids)[index]),
                    "epoch_sequence": epoch_sequence,
                }
            )
            for index in np.nonzero(active)[0]
        )
        radiation_ids = encode_content_ids(
            content_ids, self.epoch_plan.radiation_capacity
        )
        values = np.zeros(
            (
                self.epoch_plan.radiation_capacity,
                self.epoch_plan.radiation_width,
            ),
            dtype=self.epoch_plan.value_dtype,
        )
        resident = int(np.sum(active))
        if resident:
            selected = np.nonzero(active)[0]
            four = np.asarray(state.tetrad_four_momentum)[selected]
            values[:resident, 0:4] = four
            values[:resident, 4:7] = np.asarray(state.comoving_position)[selected]
            values[:resident, 7] = np.asarray(state.frequency_group)[selected]
            values[:resident, 8:12] = np.asarray(state.stokes)[selected]
            values[:resident, 12] = np.asarray(state.optical_depth_threshold)[selected]
            values[:resident, 13] = np.asarray(state.optical_depth_accumulator)[selected]
            values[:resident, 14] = np.asarray(state.weight)[selected]
            values[:resident, 15] = np.asarray(state.generation)[selected]
            values[:resident, 16] = np.asarray(state.species_ids)[selected]
            values[:resident, 17] = np.asarray(state.owner)[selected]
            values[:resident, 18] = np.asarray(state.status)[selected]
            values[:resident, 19] = np.asarray(state.rng_counters)[selected]
        mask = np.arange(self.epoch_plan.radiation_capacity) < resident
        status = np.where(mask, int(DarkRadiationPacketStatus.ACTIVE), 0).astype(np.int8)
        epoch = replace_dark_sector_pool(
            epoch,
            "radiation",
            ids=radiation_ids,
            values=values,
            mask=mask,
            status=status,
        )
        weight = state.weight * state.active_mask
        four_total = jnp.sum(weight[:, None] * state.tetrad_four_momentum, axis=0)
        conservation = jnp.zeros((8,), dtype=jnp.dtype(self.epoch_plan.precision_id))
        conservation = conservation.at[0].set(four_total[0])
        conservation = conservation.at[1:4].set(
            four_total[1:] / self.physical_light_speed
        )
        return replace_dark_sector_conservation(epoch, conservation, conservation)

    def _check_frame(self, frame: LocalRelativisticFramePlan, /) -> None:
        if not isinstance(frame, LocalRelativisticFramePlan):
            raise TypeError("frame must be LocalRelativisticFramePlan.")
        if frame.units.contract_id != self.units.contract_id:
            raise ValueError("Packet frame and plan use different unit contracts.")
        if frame.geometry.leading_shape != ():
            raise ValueError(
                "Packet transport currently requires one scalar local frame stage."
            )

    def _check_state_shapes(self, state: DarkRadiationPacketState, /) -> None:
        if not isinstance(state, DarkRadiationPacketState):
            raise TypeError("state must be DarkRadiationPacketState.")
        if (
            state.plan_id != self.plan_id
            or state.unit_contract_id != self.units.contract_id
            or state.epoch_plan_id != self.epoch_plan_id
        ):
            raise ValueError("Packet state runtime identity does not match the plan.")
        shape = (self.capacity,)
        expected = {
            "global_ids": shape,
            "species_ids": shape,
            "parent_ids": shape + (2,),
            "source_event_ids": shape,
            "generation": shape,
            "comoving_position": shape + (3,),
            "tetrad_four_momentum": shape + (4,),
            "frequency_group": shape,
            "stokes": shape + (4,),
            "coherency": shape + (2, 2),
            "optical_depth_threshold": shape,
            "optical_depth_accumulator": shape,
            "rng_keys": shape + (2,),
            "rng_counters": shape,
            "owner": shape,
            "status": shape,
            "weight": shape,
            "active_mask": shape,
            "frame_token": (),
            "observer_coordinates": (4,),
            "frame_realization_words": (8,),
            "epoch_manifest_words": (8,),
        }
        for name, expected_shape in expected.items():
            if vars(state)[name].shape != expected_shape:
                raise ValueError(
                    f"Packet state {name} must have shape {expected_shape}; got {vars(state)[name].shape}."
                )


def write_dark_radiation_packet_checkpoint(
    path: str | Path,
    plan: DarkRadiationPacketPlan,
    state: DarkRadiationPacketState,
    /,
) -> Path:
    """Write an exact restart image, including RNG and durable epoch identity."""

    if not isinstance(plan, DarkRadiationPacketPlan):
        raise TypeError("plan must be DarkRadiationPacketPlan.")
    plan._check_state_shapes(state)
    if not bool(np.asarray(plan.valid(state))):
        raise ValueError("Only a valid dark-radiation packet state can be checkpointed.")
    arrays: dict[str, object] = {}
    specification = pack_array_tree("state", state, arrays)
    payload_id = canonical_fingerprint(
        {
            "plan_id": plan.plan_id,
            "epoch_plan_id": plan.epoch_plan_id,
            "epoch_manifest_id": state.epoch_manifest_id,
            "state": specification,
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    return write_array_archive(
        path,
        manifest={
            "format": _CHECKPOINT_FORMAT,
            "plan_id": plan.plan_id,
            "epoch_plan_id": plan.epoch_plan_id,
            "epoch_manifest_id": state.epoch_manifest_id,
            "state": specification,
            "payload_id": payload_id,
        },
        arrays=arrays,
    )


def read_dark_radiation_packet_checkpoint(
    path: str | Path,
    plan: DarkRadiationPacketPlan,
    template: DarkRadiationPacketState,
    /,
) -> DarkRadiationPacketState:
    """Restore only the exact packet/unit/epoch runtime identity."""

    if not isinstance(plan, DarkRadiationPacketPlan):
        raise TypeError("plan must be DarkRadiationPacketPlan.")
    plan._check_state_shapes(template)
    template_arrays: dict[str, object] = {}
    template_specification = pack_array_tree("state", template, template_arrays)
    expected_inventory = {
        name: (np.asarray(value).shape, np.asarray(value).dtype)
        for name, value in template_arrays.items()
    }
    manifest, arrays = read_array_archive(path, expected_inventory=expected_inventory)
    expected_keys = {
        "format",
        "plan_id",
        "epoch_plan_id",
        "epoch_manifest_id",
        "state",
        "payload_id",
        "arrays",
    }
    if (
        set(manifest) != expected_keys
        or manifest["format"] != _CHECKPOINT_FORMAT
        or manifest["plan_id"] != plan.plan_id
        or manifest["epoch_plan_id"] != plan.epoch_plan_id
        or manifest["epoch_manifest_id"] != template.epoch_manifest_id
        or manifest["state"] != template_specification
    ):
        raise ValueError("Dark-radiation checkpoint runtime identity does not match.")
    payload_id = canonical_fingerprint(
        {
            "plan_id": plan.plan_id,
            "epoch_plan_id": plan.epoch_plan_id,
            "epoch_manifest_id": manifest["epoch_manifest_id"],
            "state": manifest["state"],
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    if payload_id != manifest["payload_id"]:
        raise ValueError("Dark-radiation checkpoint payload identity is corrupt.")
    restored = unpack_array_tree(manifest["state"], arrays, template)
    if not isinstance(restored, DarkRadiationPacketState) or not bool(
        np.asarray(plan.valid(restored))
    ):
        raise ValueError("Restored dark-radiation packet state violates invariants.")
    return restored


__all__ = [
    "DarkRadiationInteractionKind",
    "DarkRadiationPacketAdmissionEvidence",
    "DarkRadiationPacketAdmissionResult",
    "DarkRadiationPacketEvents",
    "DarkRadiationPacketPlan",
    "DarkRadiationPacketState",
    "DarkRadiationPacketStatus",
    "DarkRadiationPacketStepEvidence",
    "DarkRadiationPacketStepResult",
    "read_dark_radiation_packet_checkpoint",
    "write_dark_radiation_packet_checkpoint",
]
