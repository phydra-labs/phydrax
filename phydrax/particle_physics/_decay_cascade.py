#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-epoch execution of semantically unbounded dark decay DAGs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..applications.relativistic_scattering._decays import (
    decay_two_body,
    TwoBodyDecayPlan,
)
from ..applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from ..lifecycle._event_graph_repository import (
    EpochCommitReceipt,
    GlobalEntity,
    GlobalEvent,
    GlobalEventEdge,
    GlobalWorkItem,
)
from ..solver._dark_sector_epoch_runtime import (
    admit_dark_sector_work,
    DarkSectorEpochPlan,
    DarkSectorEpochResult,
    DarkSectorEpochState,
    DarkSectorRunCoordinator,
    decode_content_id,
    encode_content_id,
    finalize_dark_sector_epoch,
)
from ._host_events import HostEventRecord
from ._identity import ParticleRole
from ._species import ParticleSpeciesTable


def _pdg_id(value: object, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    bounds = np.iinfo(np.int32)
    if result < bounds.min or result > bounds.max:
        raise OverflowError(f"{name} must fit signed int32.")
    return result


class DarkDecayTiming(StrEnum):
    PROMPT = "prompt"
    DELAYED = "delayed"


class DarkDecayEpochStatus(IntEnum):
    SUCCESS = 0
    BACKPRESSURED = 1
    INVALID_KINEMATICS = 2


class DarkDecayChannel(StrictModule, NonTrainableState):
    decay: TwoBodyDecayPlan
    channel_id: str = eqx.field(static=True)

    def __init__(self, decay: TwoBodyDecayPlan, /):
        if not isinstance(decay, TwoBodyDecayPlan):
            raise TypeError("decay must be the existing TwoBodyDecayPlan owner.")
        self.decay = decay
        self.channel_id = canonical_fingerprint(
            {"kind": "dark-decay-channel", "two_body_decay": decay.plan_id}
        )


class DarkDecaySpeciesOwner(StrictModule, NonTrainableState):
    channels: tuple[DarkDecayChannel, ...]
    pdg_id: int = eqx.field(static=True)
    owner_id: str = eqx.field(static=True)
    mean_proper_lifetime: float = eqx.field(static=True)
    owner_record_id: str = eqx.field(static=True)

    def __init__(
        self,
        pdg_id: int,
        channels: Sequence[DarkDecayChannel],
        /,
        *,
        owner_id: str,
        mean_proper_lifetime: float,
    ):
        channels_ = tuple(channels)
        pdg_id_ = _pdg_id(pdg_id, "pdg_id")
        owner = str(owner_id).strip()
        lifetime = float(mean_proper_lifetime)
        if not channels_ or any(
            not isinstance(value, DarkDecayChannel) for value in channels_
        ):
            raise TypeError("channels must contain DarkDecayChannel values.")
        if any(value.decay.parent_pdg_id != pdg_id_ for value in channels_):
            raise ValueError(
                "Every owned decay channel must have the owned parent species."
            )
        fractions = tuple(float(value.decay.branching_fraction) for value in channels_)
        if not math.isclose(math.fsum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Owned decay branching fractions must sum to one.")
        if not owner or not math.isfinite(lifetime) or lifetime <= 0.0:
            raise ValueError(
                "Decay owner and mean proper lifetime must be explicit and valid."
            )
        self.channels = channels_
        self.pdg_id = pdg_id_
        self.owner_id = owner
        self.mean_proper_lifetime = lifetime
        self.owner_record_id = canonical_fingerprint(
            {
                "kind": "dark-decay-species-owner",
                "pdg_id": self.pdg_id,
                "owner": owner,
                "mean_proper_lifetime": lifetime,
                "channels": [value.channel_id for value in channels_],
            }
        )


class DarkDecayCascadePlan(StrictModule, NonTrainableState):
    """Physics schema mapped onto the universal durable dark-sector epoch state.

    Frontier values use columns ``(E,cp_x,cp_y,cp_z,remaining_tau,elapsed_tau,
    lab_time,depth)`` and exact species-table slots use ``frontier_status``. No
    dynamic array grows: descendants continue through committed runtime epochs.
    """

    runtime_plan: DarkSectorEpochPlan
    species: ParticleSpeciesTable
    units: RelativisticUnitContract
    frame: LocalRelativisticFramePlan
    frame_realization_id: str = eqx.field(static=True)
    owners: tuple[DarkDecaySpeciesOwner, ...]
    model_id: str = eqx.field(static=True)
    model_revision_id: str = eqx.field(static=True)
    prompt_lifetime_cutoff: float = eqx.field(static=True)
    production_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    support_scope: str = eqx.field(static=True)
    refusal_modes: tuple[str, ...] = eqx.field(static=True)
    differentiation_mode: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime_plan: DarkSectorEpochPlan,
        species: ParticleSpeciesTable,
        units: RelativisticUnitContract,
        frame: LocalRelativisticFramePlan,
        owners: Sequence[DarkDecaySpeciesOwner],
        /,
        *,
        model_id: str,
        model_revision_id: str,
        prompt_lifetime_cutoff: float,
        production_evidence_ids: Sequence[str],
    ):
        if not isinstance(runtime_plan, DarkSectorEpochPlan):
            raise TypeError("runtime_plan must be DarkSectorEpochPlan.")
        if not isinstance(species, ParticleSpeciesTable):
            raise TypeError("species must be ParticleSpeciesTable.")
        if runtime_plan.species_revision_id != species.table_id:
            raise ValueError(
                "runtime_plan species revision must match the decay species table."
            )
        if not isinstance(units, RelativisticUnitContract):
            raise TypeError("units must be RelativisticUnitContract.")
        if species.energy_unit.unit_id != units.energy_unit.unit_id:
            raise ValueError(
                "species and decay cascade must share the exact relativistic energy unit."
            )
        if (
            not isinstance(frame, LocalRelativisticFramePlan)
            or frame.units.contract_id != units.contract_id
        ):
            raise ValueError("frame must use the exact relativistic unit contract.")
        if units.convention.metric_signature != "mostly_minus":
            raise ValueError(
                "The existing two-body decay owner requires the mostly-minus convention."
            )
        owners_ = tuple(owners)
        if not owners_ or any(
            not isinstance(value, DarkDecaySpeciesOwner) for value in owners_
        ):
            raise TypeError("owners must contain DarkDecaySpeciesOwner values.")
        owned_species = tuple(value.pdg_id for value in owners_)
        if len(set(owned_species)) != len(owned_species):
            raise ValueError("Every unstable species must have exactly one decay owner.")
        model = str(model_id).strip()
        revision = str(model_revision_id).strip()
        evidence = tuple(str(value).strip() for value in production_evidence_ids)
        cutoff = float(prompt_lifetime_cutoff)
        if (
            not model
            or not revision
            or not evidence
            or any(not value for value in evidence)
            or len(set(evidence)) != len(evidence)
            or not math.isfinite(cutoff)
            or cutoff < 0.0
        ):
            raise ValueError(
                "Decay model identities, evidence, and prompt cutoff are invalid."
            )
        if len(revision) != 64 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise ValueError("model_revision_id must be a lowercase SHA-256 digest.")
        if (
            runtime_plan.frontier_width < 8
            or runtime_plan.product_width < 4
            or runtime_plan.event_width < 7
        ):
            raise ValueError(
                "Runtime value widths are too small for the decay cascade schema."
            )
        if species.capacity > 127:
            raise ValueError(
                "Decay cascade species tables are limited to 127 exact int8 resident slots."
            )
        ids = np.asarray(species.pdg_ids)
        masses = np.asarray(species.rest_energies)
        charges = np.asarray(species.charges)
        active = np.asarray(species.active)
        by_id = {
            int(identifier): (float(mass), float(charge))
            for identifier, mass, charge, present in zip(
                ids, masses, charges, active, strict=True
            )
            if present
        }
        for owner in owners_:
            if owner.pdg_id not in by_id:
                raise ValueError("Every owned species must exist in the species table.")
            for channel in owner.channels:
                daughters = channel.decay.daughter_pdg_ids
                if any(value not in by_id for value in daughters):
                    raise ValueError(
                        "Every daughter species must exist in the species table."
                    )
                if by_id[owner.pdg_id][0] < sum(by_id[value][0] for value in daughters):
                    raise ValueError(
                        "Decay parent lies below its declared daughter threshold."
                    )
                if not math.isclose(
                    by_id[owner.pdg_id][1],
                    sum(by_id[value][1] for value in daughters),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "Every owned decay channel must conserve declared charge."
                    )
        self.runtime_plan = runtime_plan
        self.species = species
        self.units = units
        self.frame = frame
        self.frame_realization_id = frame.realization_id()
        self.owners = owners_
        self.model_id = model
        self.model_revision_id = revision
        self.prompt_lifetime_cutoff = cutoff
        self.production_evidence_ids = evidence
        self.support_scope = "owned-isotropic-two-body-decay-dag"
        self.refusal_modes = (
            "unowned-unstable-species",
            "non-two-body-channel",
            "nonconserving-channel",
            "non-mostly-minus-event-record",
        )
        self.differentiation_mode = "piecewise-stopped-proper-time-branching"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unbounded-dark-decay-cascade-plan",
                "runtime_plan": runtime_plan.plan_id,
                "species": species.table_id,
                "units": units.contract_id,
                "frame": frame.frame_id,
                "frame_realization": self.frame_realization_id,
                "owners": [value.owner_record_id for value in owners_],
                "model": model,
                "revision": revision,
                "prompt_lifetime_cutoff": cutoff,
                "production_evidence": list(evidence),
                "support_scope": self.support_scope,
                "refusal_modes": list(self.refusal_modes),
                "differentiation": self.differentiation_mode,
            }
        )

    def timing(self, pdg_id: int, /) -> DarkDecayTiming:
        matches = tuple(value for value in self.owners if value.pdg_id == int(pdg_id))
        if len(matches) != 1:
            raise KeyError(f"Species {int(pdg_id)} has no decay owner.")
        return (
            DarkDecayTiming.PROMPT
            if matches[0].mean_proper_lifetime <= self.prompt_lifetime_cutoff
            else DarkDecayTiming.DELAYED
        )


class DarkDecayCascadeEpochEvidence(StrictModule, NonTrainableState):
    runtime_result: DarkSectorEpochResult
    decayed: Array
    prompt: Array
    delayed: Array
    deferred: Array
    channel_indices: Array
    four_momentum_residual: Array
    charge_residual: Array
    status: Array
    plan_id: str = eqx.field(static=True)
    draw_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def _particle_seed_id(
    event: HostEventRecord,
    particle_id: int,
    plan_id: str,
    draw_id: str,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "dark-decay-frontier-seed",
            "source": event.source_id,
            "event": [event.event_id, event.subevent_id],
            "particle_id": int(particle_id),
            "plan": plan_id,
            "draw": draw_id,
        }
    )


def seed_decay_frontier_from_host(
    plan: DarkDecayCascadePlan,
    state: DarkSectorEpochState,
    event: HostEventRecord,
    clock_uniforms: ArrayLike,
    /,
    *,
    momentum_unit_id: str,
    frame_id: str,
    frame_realization_id: str,
) -> DarkSectorEpochResult:
    """Atomically seed owned outgoing particles into the universal frontier."""

    if not isinstance(plan, DarkDecayCascadePlan):
        raise TypeError("plan must be DarkDecayCascadePlan.")
    if not isinstance(state, DarkSectorEpochState):
        raise TypeError("state must be DarkSectorEpochState.")
    if not isinstance(event, HostEventRecord):
        raise TypeError("event must be HostEventRecord.")
    if state.plan.plan_id != plan.runtime_plan.plan_id:
        raise ValueError("state was created for a different dark-sector epoch plan.")
    if (
        str(momentum_unit_id) != plan.units.energy_unit.unit_id
        or str(frame_id) != plan.frame.frame_id
        or str(frame_realization_id) != plan.frame_realization_id
    ):
        raise ValueError(
            "Host event seeding requires exact momentum-unit, local-frame, and frame-realization identities."
        )
    active_species = np.asarray(plan.species.active, dtype=np.bool_)
    species_ids = np.asarray(plan.species.pdg_ids)
    slot_by_id = {
        int(identifier): int(slot)
        for slot, (identifier, active) in enumerate(
            zip(species_ids, active_species, strict=True)
        )
        if active
    }
    owned = {value.pdg_id: value for value in plan.owners}
    particles = tuple(
        value
        for value in event.particles
        if value.role is ParticleRole.OUTGOING and value.pdg_id in owned
    )
    clocks = np.asarray(clock_uniforms, dtype=np.float64)
    if (
        clocks.shape != (len(particles),)
        or np.any(~np.isfinite(clocks))
        or np.any((clocks < 0.0) | (clocks >= 1.0))
    ):
        raise ValueError(
            "clock_uniforms must provide one finite [0,1) value per seeded particle."
        )
    draw_id = canonical_fingerprint(
        {
            "kind": "dark-decay-clock-draws",
            "plan": plan.plan_id,
            "event": [event.event_id, event.subevent_id],
            "algorithm": "caller-supplied-uniforms",
            "uniforms": clocks.tolist(),
        }
    )
    values = np.zeros(
        (len(particles), plan.runtime_plan.work_width),
        dtype=plan.runtime_plan.value_dtype,
    )
    statuses = np.zeros((len(particles),), dtype=np.int8)
    content_ids: list[str] = []
    for index, (particle, uniform) in enumerate(zip(particles, clocks, strict=True)):
        owner = owned[particle.pdg_id]
        lifetime = (
            0.0
            if owner.mean_proper_lifetime <= plan.prompt_lifetime_cutoff
            else -owner.mean_proper_lifetime * math.log1p(-float(uniform))
        )
        content_ids.append(
            _particle_seed_id(event, particle.particle_id, plan.plan_id, draw_id)
        )
        values[index, :4] = particle.momentum
        values[index, 4] = lifetime
        values[index, 5:8] = 0.0
        statuses[index] = slot_by_id[particle.pdg_id]
    admission = admit_dark_sector_work(
        state,
        tuple(content_ids),
        values,
        work_status=statuses,
    )
    return DarkSectorEpochResult(
        admission.state,
        complete=True,
        backpressured=admission.backpressured,
        rolled_back=admission.refused,
        evidence_ids=(plan.plan_id, draw_id, *plan.production_evidence_ids),
    )


def _cascade_tables(plan: DarkDecayCascadePlan):
    species_ids = np.asarray(plan.species.pdg_ids)
    active = np.asarray(plan.species.active)
    charges = np.asarray(plan.species.charges)
    slot_by_id = {
        int(identifier): int(slot)
        for slot, (identifier, present) in enumerate(
            zip(species_ids, active, strict=True)
        )
        if present
    }
    charge_by_id = {
        int(identifier): float(charge)
        for identifier, charge, present in zip(species_ids, charges, active, strict=True)
        if present
    }
    parent_slots = []
    daughter_slots = []
    branching = []
    owner_lifetime = {value.pdg_id: value.mean_proper_lifetime for value in plan.owners}
    decay_plans = []
    parent_ids = []
    daughter_ids = []
    for owner in plan.owners:
        for channel in owner.channels:
            parent_ids.append(owner.pdg_id)
            daughter_ids.append(channel.decay.daughter_pdg_ids)
            parent_slots.append(slot_by_id[owner.pdg_id])
            daughter_slots.append(
                tuple(slot_by_id[value] for value in channel.decay.daughter_pdg_ids)
            )
            branching.append(float(channel.decay.branching_fraction))
            decay_plans.append(channel.decay)
    daughter_lifetime = [
        [owner_lifetime.get(value, math.inf) for value in pair] for pair in daughter_ids
    ]
    daughter_owned = [
        [value in owner_lifetime for value in pair] for pair in daughter_ids
    ]
    return (
        jnp.asarray(parent_slots, dtype=jnp.int8),
        jnp.asarray(daughter_slots, dtype=jnp.int8),
        jnp.asarray(branching),
        jnp.asarray(daughter_lifetime),
        jnp.asarray(daughter_owned, dtype=jnp.bool_),
        jnp.asarray([charge_by_id[value] for value in parent_ids]),
        jnp.asarray([[charge_by_id[value] for value in pair] for pair in daughter_ids]),
        tuple(decay_plans),
    )


def _child_ids(parent_ids, ordinal, epoch_sequence):
    constants = jnp.asarray(
        (
            0x9E3779B9,
            0x85EBCA6B,
            0xC2B2AE35,
            0x27D4EB2F,
            0x165667B1,
            0xD3A2646C,
            0xFD7046C5,
            0xB55A4F09,
        ),
        dtype=jnp.uint32,
    )
    salt = (
        jnp.asarray(epoch_sequence, dtype=jnp.uint32)
        + jnp.asarray(ordinal + 1, dtype=jnp.uint32) * constants
    )
    return (parent_ids * jnp.asarray(0x01000193, dtype=jnp.uint32)) ^ salt


def evolve_decay_cascade_epoch(
    plan: DarkDecayCascadePlan,
    state: DarkSectorEpochState,
    proper_time_step: ArrayLike,
    uniforms: ArrayLike,
    *,
    draw_id: str,
) -> DarkDecayCascadeEpochEvidence:
    """Advance one finite DAG frontier and defer all excess work transactionally."""

    if not isinstance(plan, DarkDecayCascadePlan):
        raise TypeError("plan must be DarkDecayCascadePlan.")
    if not isinstance(state, DarkSectorEpochState):
        raise TypeError("state must be DarkSectorEpochState.")
    if state.plan.plan_id != plan.runtime_plan.plan_id:
        raise ValueError("state was created for a different dark-sector epoch plan.")
    step = jnp.asarray(proper_time_step, dtype=state.work_values.dtype)
    random = jnp.asarray(uniforms, dtype=state.work_values.dtype)
    if random.shape != (plan.runtime_plan.work_capacity, 5):
        raise ValueError("uniforms must have shape (work_capacity, 5).")
    step = eqx.error_if(
        step,
        ~jnp.isfinite(step) | (step <= 0.0),
        "proper_time_step must be finite and positive.",
    )
    random = eqx.error_if(
        random,
        jnp.any(~jnp.isfinite(random) | (random < 0.0) | (random >= 1.0)),
        "uniforms must lie in [0, 1).",
    )
    draw = str(draw_id).strip()
    if not draw:
        raise ValueError("draw_id must be a non-empty RNG stream/draw identity.")
    (
        channel_parents,
        channel_daughters,
        branching,
        daughter_lifetimes,
        daughter_owned,
        parent_charges,
        daughter_charges,
        decay_plans,
    ) = _cascade_tables(plan)
    species_ids = np.asarray(plan.species.pdg_ids)
    active_species = np.asarray(plan.species.active)
    slot_by_id = {
        int(identifier): int(slot)
        for slot, (identifier, active) in enumerate(
            zip(species_ids, active_species, strict=True)
        )
        if active
    }
    owner_slots = jnp.asarray(
        [slot_by_id[value.pdg_id] for value in plan.owners], dtype=jnp.int8
    )
    owner_lifetimes = jnp.asarray([value.mean_proper_lifetime for value in plan.owners])

    frontier_ids = state.frontier_ids
    frontier_values = state.frontier_values
    frontier_mask = state.frontier_mask
    frontier_status = state.frontier_status
    product_ids = jnp.zeros_like(state.product_ids)
    product_values = jnp.zeros_like(state.product_values)
    product_mask = jnp.zeros_like(state.product_mask)
    product_status = jnp.zeros_like(state.product_status)
    event_ids = jnp.zeros_like(state.event_ids)
    event_values = jnp.zeros_like(state.event_values)
    event_mask = jnp.zeros_like(state.event_mask)
    event_status = jnp.zeros_like(state.event_status)
    decayed = jnp.zeros((plan.runtime_plan.work_capacity,), dtype=jnp.bool_)
    prompt = jnp.zeros_like(decayed)
    delayed = jnp.zeros_like(decayed)
    deferred = jnp.zeros_like(decayed)
    channel_indices = jnp.full((plan.runtime_plan.work_capacity,), -1, dtype=jnp.int32)
    momentum_residual = jnp.zeros(
        (plan.runtime_plan.work_capacity, 4), dtype=state.work_values.dtype
    )
    charge_residual = jnp.zeros(
        (plan.runtime_plan.work_capacity,), dtype=state.work_values.dtype
    )

    carry = (
        frontier_ids,
        frontier_values,
        frontier_mask,
        frontier_status,
        product_ids,
        product_values,
        product_mask,
        product_status,
        event_ids,
        event_values,
        event_mask,
        event_status,
        jnp.sum(state.frontier_mask, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
        decayed,
        prompt,
        delayed,
        deferred,
        channel_indices,
        momentum_residual,
        charge_residual,
    )

    def body(index, values):
        (
            next_ids,
            next_values,
            next_mask,
            next_status,
            products_ids,
            products_values,
            products_mask,
            products_status,
            events_ids,
            events_values,
            events_mask,
            events_status,
            next_count,
            product_count,
            event_count,
            backpressured,
            decayed_,
            prompt_,
            delayed_,
            deferred_,
            selected_indices,
            momentum_residual_,
            charge_residual_,
        ) = values
        active = state.work_mask[index]
        species_slot = state.work_status[index]
        parent_momentum = state.work_values[index, :4]
        remaining = state.work_values[index, 4]
        elapsed = state.work_values[index, 5]
        lab_time = state.work_values[index, 6]
        depth = state.work_values[index, 7]
        owner_match = species_slot == owner_slots
        owner_index = jnp.argmax(owner_match).astype(jnp.int32)
        owner_lifetime = owner_lifetimes[owner_index]
        is_prompt = owner_lifetime <= plan.prompt_lifetime_cutoff
        due = active & (is_prompt | (remaining <= step))
        compatible = species_slot == channel_parents
        weights = jnp.where(compatible, branching, 0.0)
        target = random[index, 0] * jnp.sum(weights)
        selected = jnp.argmax(jnp.cumsum(weights) > target).astype(jnp.int32)
        daughter_slots = channel_daughters[selected]
        owned = daughter_owned[selected]
        child_count = jnp.sum(owned, dtype=jnp.int32)
        remaining_active = jnp.sum(
            state.work_mask
            & (jnp.arange(plan.runtime_plan.work_capacity, dtype=jnp.int32) > index),
            dtype=jnp.int32,
        )
        capacity_ok = (
            (product_count + 2 <= plan.runtime_plan.product_capacity)
            & (event_count + 1 <= plan.runtime_plan.event_capacity)
            & (
                next_count + child_count + remaining_active
                <= plan.runtime_plan.frontier_capacity
            )
        )
        publish = due & capacity_ok
        pressure = due & ~capacity_ok
        keep_parent = active & ~publish

        elapsed_increment = jnp.where(due, remaining, step)
        parent_energy = parent_momentum[0]
        parent_mass = jnp.sqrt(
            jnp.maximum(
                plan.units.lorentz_scalar(parent_momentum, parent_momentum), 1e-30
            )
        )
        lab_increment = elapsed_increment * parent_energy / parent_mass
        kept_value = (
            state.work_values[index].at[4].set(jnp.maximum(remaining - step, 0.0))
        )
        kept_value = kept_value.at[5].set(elapsed + elapsed_increment)
        kept_value = kept_value.at[6].set(lab_time + lab_increment)
        safe_next = jnp.minimum(next_count, plan.runtime_plan.frontier_capacity - 1)
        old_next_ids = next_ids[safe_next]
        old_next_values = next_values[safe_next]
        old_next_mask = next_mask[safe_next]
        old_next_status = next_status[safe_next]
        next_ids = next_ids.at[safe_next].set(
            jnp.where(keep_parent, state.work_ids[index], old_next_ids)
        )
        next_values = next_values.at[safe_next].set(
            jnp.where(keep_parent, kept_value, old_next_values)
        )
        next_mask = next_mask.at[safe_next].set(
            jnp.where(keep_parent, True, old_next_mask)
        )
        next_status = next_status.at[safe_next].set(
            jnp.where(keep_parent, species_slot, old_next_status)
        )
        next_count = next_count + keep_parent.astype(jnp.int32)

        branches = tuple(
            (
                lambda coordinates, momentum, decay_plan=decay_plan: (
                    decay_two_body(decay_plan, momentum, coordinates).point.momenta
                )
            )
            for decay_plan in decay_plans
        )
        child_momenta = jax.lax.switch(
            selected, branches, random[index, 1:3], parent_momentum
        )
        child_ids = jnp.stack(
            (
                _child_ids(state.work_ids[index], 0, state.epoch_sequence),
                _child_ids(state.work_ids[index], 1, state.epoch_sequence),
            )
        )
        selected_lifetimes = daughter_lifetimes[selected]
        child_remaining = jnp.where(
            owned & (selected_lifetimes > plan.prompt_lifetime_cutoff),
            -selected_lifetimes * jnp.log1p(-random[index, 3:5]),
            0.0,
        )
        child_values = jnp.zeros(
            (2, plan.runtime_plan.frontier_width), dtype=state.work_values.dtype
        )
        child_values = child_values.at[:, :4].set(child_momenta)
        child_values = child_values.at[:, 4].set(child_remaining)
        child_values = child_values.at[:, 6].set(lab_time + lab_increment)
        child_values = child_values.at[:, 7].set(depth + 1.0)

        for ordinal in range(2):
            product_slot = jnp.minimum(
                product_count + ordinal, plan.runtime_plan.product_capacity - 1
            )
            products_ids = products_ids.at[product_slot].set(
                jnp.where(publish, child_ids[ordinal], products_ids[product_slot])
            )
            products_values = products_values.at[product_slot, :4].set(
                jnp.where(
                    publish, child_momenta[ordinal], products_values[product_slot, :4]
                )
            )
            products_mask = products_mask.at[product_slot].set(
                jnp.where(publish, True, products_mask[product_slot])
            )
            products_status = products_status.at[product_slot].set(
                jnp.where(publish, daughter_slots[ordinal], products_status[product_slot])
            )
            child_frontier_slot = jnp.minimum(
                next_count + jnp.sum(owned[:ordinal], dtype=jnp.int32),
                plan.runtime_plan.frontier_capacity - 1,
            )
            publish_child = publish & owned[ordinal]
            next_ids = next_ids.at[child_frontier_slot].set(
                jnp.where(
                    publish_child, child_ids[ordinal], next_ids[child_frontier_slot]
                )
            )
            next_values = next_values.at[child_frontier_slot].set(
                jnp.where(
                    publish_child, child_values[ordinal], next_values[child_frontier_slot]
                )
            )
            next_mask = next_mask.at[child_frontier_slot].set(
                jnp.where(publish_child, True, next_mask[child_frontier_slot])
            )
            next_status = next_status.at[child_frontier_slot].set(
                jnp.where(
                    publish_child,
                    daughter_slots[ordinal],
                    next_status[child_frontier_slot],
                )
            )

        event_slot = jnp.minimum(event_count, plan.runtime_plan.event_capacity - 1)
        generated_event_id = _child_ids(state.work_ids[index], 2, state.epoch_sequence)
        residual_p = jnp.sum(child_momenta, axis=0) - parent_momentum
        residual_q = jnp.sum(daughter_charges[selected]) - parent_charges[selected]
        event_value = jnp.zeros(
            (plan.runtime_plan.event_width,), dtype=state.event_values.dtype
        )
        event_value = event_value.at[:4].set(residual_p)
        event_value = event_value.at[4].set(residual_q)
        event_value = event_value.at[5].set(is_prompt.astype(event_value.dtype))
        event_value = event_value.at[6].set(depth)
        events_ids = events_ids.at[event_slot].set(
            jnp.where(publish, generated_event_id, events_ids[event_slot])
        )
        events_values = events_values.at[event_slot].set(
            jnp.where(publish, event_value, events_values[event_slot])
        )
        events_mask = events_mask.at[event_slot].set(
            jnp.where(publish, True, events_mask[event_slot])
        )
        events_status = events_status.at[event_slot].set(
            jnp.where(
                publish, int(DarkDecayEpochStatus.SUCCESS), events_status[event_slot]
            )
        )
        next_count = next_count + jnp.where(publish, child_count, 0)
        product_count = product_count + jnp.where(publish, 2, 0)
        event_count = event_count + publish.astype(jnp.int32)
        backpressured = backpressured | pressure
        decayed_ = decayed_.at[index].set(publish)
        prompt_ = prompt_.at[index].set(publish & is_prompt)
        delayed_ = delayed_.at[index].set(publish & ~is_prompt)
        deferred_ = deferred_.at[index].set(pressure)
        selected_indices = selected_indices.at[index].set(
            jnp.where(publish, selected, -1)
        )
        momentum_residual_ = momentum_residual_.at[index].set(
            jnp.where(publish, residual_p, 0.0)
        )
        charge_residual_ = charge_residual_.at[index].set(
            jnp.where(publish, residual_q, 0.0)
        )
        return (
            next_ids,
            next_values,
            next_mask,
            next_status,
            products_ids,
            products_values,
            products_mask,
            products_status,
            events_ids,
            events_values,
            events_mask,
            events_status,
            next_count,
            product_count,
            event_count,
            backpressured,
            decayed_,
            prompt_,
            delayed_,
            deferred_,
            selected_indices,
            momentum_residual_,
            charge_residual_,
        )

    carry = jax.lax.fori_loop(0, plan.runtime_plan.work_capacity, body, carry)
    (
        frontier_ids,
        frontier_values,
        frontier_mask,
        frontier_status,
        product_ids,
        product_values,
        product_mask,
        product_status,
        event_ids,
        event_values,
        event_mask,
        event_status,
        _,
        _,
        _,
        backpressured,
        decayed,
        prompt,
        delayed,
        deferred,
        channel_indices,
        momentum_residual,
        charge_residual,
    ) = carry
    conservation_in = jnp.zeros_like(state.conservation_in)
    conservation_out = jnp.zeros_like(state.conservation_out)
    input_momentum = jnp.sum(
        jnp.where(decayed[:, None], state.work_values[:, :4], 0.0), axis=0
    )
    output_momentum = input_momentum + jnp.sum(momentum_residual, axis=0)
    input_charge = jnp.sum(
        jnp.where(
            decayed,
            parent_charges[jnp.maximum(channel_indices, 0)],
            0.0,
        )
    )
    output_charge = input_charge + jnp.sum(charge_residual)
    conservation_in = conservation_in.at[:4].set(input_momentum)
    conservation_out = conservation_out.at[:4].set(output_momentum)
    conservation_in = conservation_in.at[4].set(input_charge)
    conservation_out = conservation_out.at[4].set(output_charge)
    empty_work_ids = jnp.zeros_like(state.work_ids)
    empty_work_values = jnp.zeros_like(state.work_values)
    empty_work_mask = jnp.zeros_like(state.work_mask)
    empty_work_status = jnp.zeros_like(state.work_status)
    updated = eqx.tree_at(
        lambda value: (
            value.frontier_ids,
            value.frontier_values,
            value.frontier_mask,
            value.frontier_status,
            value.work_ids,
            value.work_values,
            value.work_mask,
            value.work_status,
            value.product_ids,
            value.product_values,
            value.product_mask,
            value.product_status,
            value.event_ids,
            value.event_values,
            value.event_mask,
            value.event_status,
            value.conservation_in,
            value.conservation_out,
        ),
        state,
        (
            frontier_ids,
            frontier_values,
            frontier_mask,
            frontier_status,
            empty_work_ids,
            empty_work_values,
            empty_work_mask,
            empty_work_status,
            product_ids,
            product_values,
            product_mask,
            product_status,
            event_ids,
            event_values,
            event_mask,
            event_status,
            conservation_in,
            conservation_out,
        ),
    )
    runtime_result = finalize_dark_sector_epoch(
        state,
        updated,
        complete=True,
        backpressured=backpressured,
        evidence_ids=(plan.plan_id, *plan.production_evidence_ids),
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "dark-decay-cascade-epoch-evidence",
            "plan": plan.plan_id,
            "runtime_plan": plan.runtime_plan.plan_id,
            "evidence": list(plan.production_evidence_ids),
            "draw": draw,
            "epoch_sequence": state.epoch_sequence,
        }
    )
    status = jnp.where(
        backpressured,
        int(DarkDecayEpochStatus.BACKPRESSURED),
        jnp.where(
            jnp.any(jnp.abs(momentum_residual) > 1e-7)
            | jnp.any(jnp.abs(charge_residual) > 1e-10),
            int(DarkDecayEpochStatus.INVALID_KINEMATICS),
            int(DarkDecayEpochStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    return DarkDecayCascadeEpochEvidence(
        runtime_result,
        decayed,
        prompt,
        delayed,
        deferred,
        channel_indices,
        momentum_residual,
        charge_residual,
        status,
        plan.plan_id,
        draw,
        evidence_id,
    )


@dataclass(frozen=True, slots=True)
class DarkDecayDurableFrontier:
    evidence: DarkDecayCascadeEpochEvidence
    entities: tuple[GlobalEntity, ...]
    work_items: tuple[GlobalWorkItem, ...]


def materialize_decay_frontier_work(
    plan: DarkDecayCascadePlan,
    evidence: DarkDecayCascadeEpochEvidence,
    /,
    *,
    rights_id: str,
    partition_key: str,
) -> DarkDecayDurableFrontier:
    """Replace temporary device lineage IDs with repository-owned work identities."""

    if not isinstance(plan, DarkDecayCascadePlan):
        raise TypeError("plan must be DarkDecayCascadePlan.")
    if not isinstance(evidence, DarkDecayCascadeEpochEvidence):
        raise TypeError("evidence must be DarkDecayCascadeEpochEvidence.")
    result = evidence.runtime_result
    if result.state.plan.plan_id != plan.runtime_plan.plan_id:
        raise ValueError("evidence was produced by a different cascade plan.")
    state = result.state
    mask = np.asarray(state.frontier_mask, dtype=np.bool_)
    values = np.asarray(state.frontier_values)
    slots = np.asarray(state.frontier_status, dtype=np.int8)
    species_ids = np.asarray(plan.species.pdg_ids)
    frontier_ids = np.asarray(state.frontier_ids, dtype=np.uint32).copy()
    entities: list[GlobalEntity] = []
    work_items: list[GlobalWorkItem] = []
    for lane in np.flatnonzero(mask):
        slot = int(slots[lane])
        if slot < 0 or slot >= plan.species.capacity:
            raise ValueError("Frontier species slot is outside the declared table.")
        entity = GlobalEntity(
            "dark-decay-frontier",
            str(int(species_ids[slot])),
            canonical_fingerprint(
                {
                    "kind": "dark-decay-frontier-state",
                    "values": values[lane].tolist(),
                    "species_slot": slot,
                    "cascade_plan": plan.plan_id,
                }
            ),
            frame_id=plan.frame.frame_id,
            frame_realization_id=plan.frame_realization_id,
            unit_contract_id=plan.units.contract_id,
            rights_id=rights_id,
            provenance_ids=plan.production_evidence_ids,
        )
        work = GlobalWorkItem(
            "dark-decay",
            (entity.entity_id,),
            plan.model_revision_id,
            partition_key=partition_key,
            priority=-int(values[lane, 7]),
        )
        frontier_ids[lane] = encode_content_id(work.work_id)
        entities.append(entity)
        work_items.append(work)
    updated_state = eqx.tree_at(
        lambda value: value.frontier_ids,
        state,
        jnp.asarray(frontier_ids),
    )
    updated_result = DarkSectorEpochResult(
        updated_state,
        complete=result.complete,
        backpressured=result.backpressured,
        rolled_back=result.rolled_back,
        evidence_ids=result.evidence_ids,
    )
    updated_evidence = eqx.tree_at(
        lambda value: value.runtime_result,
        evidence,
        updated_result,
    )
    return DarkDecayDurableFrontier(updated_evidence, tuple(entities), tuple(work_items))


def commit_decay_cascade_epoch(
    coordinator: DarkSectorRunCoordinator,
    evidence: DarkDecayCascadeEpochEvidence,
    /,
    *,
    entities: Sequence[GlobalEntity],
    events: Sequence[GlobalEvent],
    edges: Sequence[GlobalEventEdge],
    work_items: Sequence[GlobalWorkItem],
    matrix_element_revision_id: str,
    committed_at: int | None = None,
) -> EpochCommitReceipt:
    """Commit one complete finite epoch and its durable continuation atomically."""

    if not isinstance(coordinator, DarkSectorRunCoordinator):
        raise TypeError("coordinator must be DarkSectorRunCoordinator.")
    if not isinstance(evidence, DarkDecayCascadeEpochEvidence):
        raise TypeError("evidence must be DarkDecayCascadeEpochEvidence.")
    result = evidence.runtime_result
    if result.state.plan.plan_id != coordinator.plan.plan_id:
        raise ValueError("coordinator and cascade result use different runtime plans.")
    active_rows = np.asarray(result.state.frontier_ids)[
        np.asarray(result.state.frontier_mask, dtype=np.bool_)
    ]
    deferred_ids = {decode_content_id(value) for value in active_rows}
    supplied_work = tuple(work_items)
    supplied_ids = {value.work_id for value in supplied_work}
    if deferred_ids != supplied_ids:
        raise ValueError(
            "work_items must exactly materialize every durable frontier identity."
        )
    return coordinator.commit_epoch(
        result,
        entities=tuple(entities),
        events=tuple(events),
        edges=tuple(edges),
        work_items=supplied_work,
        matrix_element_revision_id=matrix_element_revision_id,
        evidence_ids=(evidence.evidence_id,),
        committed_at=committed_at,
    )


__all__ = [
    "DarkDecayCascadeEpochEvidence",
    "DarkDecayCascadePlan",
    "DarkDecayChannel",
    "DarkDecayDurableFrontier",
    "DarkDecayEpochStatus",
    "DarkDecaySpeciesOwner",
    "DarkDecayTiming",
    "commit_decay_cascade_epoch",
    "evolve_decay_cascade_epoch",
    "materialize_decay_frontier_work",
    "seed_decay_frontier_from_host",
]
