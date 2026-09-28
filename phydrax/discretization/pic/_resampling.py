#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Macroparticle resampling (merging and splitting) as PIC population processes.

`ParticleMergePlan` implements the momentum-cell merging of Vranic et al.,
Comput. Phys. Commun. 191, 65 (2015): inside each spatial cell holding more
than ``maximum_per_cell`` macroparticles, particles of one charge state are
grouped into spherical momentum cells (magnitude between the cell's extreme
speeds, polar cosine, azimuth), each momentum cell is cut into packets in
global-identity order, and every packet of total mass ``W``, momentum ``P``,
and kinetic energy ``K`` becomes two particles of mass ``W/2`` with

    γ_t - 1 = K/(W c²),  u_t = c √((γ_t - 1)(γ_t + 1)),  cos ω = |P|/(W u_t),
    u_± = u_t (cos ω ê₁ ± sin ω ê₂),  ê₁ = P/|P|,

which conserves charge, mass, momentum, and energy exactly. ``ê₂`` is the unit
component of the packet's lowest-identity momentum orthogonal to ``ê₁`` (the
coordinate direction least aligned with ``ê₁`` when that is degenerate); both
products sit at the packet's center of mass, so the charge dipole is preserved.

`ParticleSplitPlan` splits the heaviest particles of cells holding fewer than
``minimum_per_cell`` particles into ``2d`` children of equal mass and momentum,
displaced by ``±δ_a`` along each resolved axis: charge, momentum, energy, and
the dipole are conserved and the spatial second moment grows by ``(m/d) δ_a²``.

Both move charge between grid locations while conserving its total, so they
declare ``redistributes_charge``: the PIC runtime Gauss-projects the field onto
the redeposited charge (`PICGaussProjection`). Products and children are
allocated through the run's allocation route (`allocate_particles`) at the
packet's center of mass or the split particle's position and receive fresh
persistent identities in canonical event order with the lowest merged
identity (merge) or the split particle (split) as parent. Grouping, packet
order, reductions, and identity assignment are keyed by cell and global
identity, never by storage slot, so results are invariant to slot order.
Events that do not fit the population's free capacity or the prepared request
width are refused (counted, never truncated silently); a failed allocation or
a conservation defect above tolerance leaves the species unchanged. Every
application reports conservation defects and moment distortion.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from enum import IntFlag
from typing import assert_never, ClassVar, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._physical import RelativityScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import KeyGroupPlan
from ...typing import parse
from ..particle import (
    ParticleAllocationRequest,
    ParticlePopulationPlan,
    ParticleSlotReusePolicy,
)
from ._binning import PICCellBinningPlan, PICCellBins
from ._charge_state import PICChargeState, PICSpeciesPlan, PICSpeciesState
from ._process import (
    AbstractPICProcess,
    allocate_particles,
    PICProcessContext,
    PICProcessLedger,
    PICProcessResult,
    PICProcessStage,
    PICProcessStatePartition,
    RadiationOwnership,
)
from ._types import PICParticleState


ParticleMergeMethod: TypeAlias = Literal["vranic-momentum-cell"]

_TINY = float(np.finfo(np.float64).tiny)


class ParticleResamplingStatus(IntFlag):
    NONE = 0
    CAPACITY_REFUSED = 1
    UNSUPPORTED = 2
    BINNING_OVERFLOW = 4
    ALLOCATION_REFUSED = 8
    CONSERVATION_REFUSED = 16
    NONFINITE = 32


class ParticleResamplingEvidence(StrictModule):
    """Outcome of resampling one species.

    ``events`` counts merged packets or split particles, ``removed``/``created``
    the deactivated and allocated macroparticles, ``refused`` eligible events
    that exceeded free capacity or the prepared request width, and
    ``unsupported`` eligible events outside the binning support. Conservation
    defects are relative: charge to ``Σ|q|``, momentum to ``Σ m|u|``, energy to
    the kinetic energy, dipole ``Σ q x`` to ``Σ|q|·h``. Moment distortions are
    the Frobenius change of ``Σ m u⊗u`` relative to its norm and of
    ``Σ m x⊗x`` relative to ``(Σ m) h²`` (``h`` the largest cell width); the
    center of mass is preserved, so the latter is the central spatial moment.
    ``status`` is a `ParticleResamplingStatus` flag set.
    """

    active_before: Array
    active_after: Array
    events: Array
    removed: Array
    created: Array
    refused: Array
    unsupported: Array
    maximum_cell_count_before: Array
    maximum_cell_count_after: Array
    charge_defect: Array
    momentum_defect: Array
    energy_defect: Array
    dipole_defect: Array
    momentum_second_moment_distortion: Array
    spatial_second_moment_distortion: Array
    status: Array
    successful: Array


class _Moments(NamedTuple):
    count: Array
    mass: Array
    charge: Array
    absolute_charge: Array
    momentum: Array
    momentum_scale: Array
    kinetic: Array
    dipole: Array
    momentum_second: Array
    spatial_second: Array
    finite: Array


class _Proposal(NamedTuple):
    """Canonically ordered replacement of one species' macroparticles.

    ``origin`` is each request row's creating-event position: the merged
    packet's center of mass or the split parent's position.
    """

    removed: Array
    valid: Array
    mass: Array
    parent_hi: Array
    parent_lo: Array
    position: Array
    origin: Array
    proper_velocity: Array
    charge_number: Array
    events: Array
    refused: Array
    unsupported: Array


def _kinetic_per_mass(proper_velocity: Array, light: float, /) -> Array:
    """``(γ - 1) c²`` in the cancellation-free form ``|u|²/(γ + 1)``."""
    squared = jnp.sum(proper_velocity * proper_velocity, axis=-1)
    return squared / (jnp.sqrt(1.0 + squared / light**2) + 1.0)


def _moments(plan: PICSpeciesPlan, state: PICSpeciesState, light: float, /) -> _Moments:
    active = state.population.active
    mass = jnp.where(active, state.population.mass, 0.0).astype(jnp.float64)
    charge = plan.macrocharge(state).astype(jnp.float64)
    velocity = jnp.where(active[:, None], state.particles.proper_velocity, 0.0).astype(
        jnp.float64
    )
    position = jnp.where(active[:, None], state.particles.position, 0.0).astype(
        jnp.float64
    )
    weighted = mass[:, None] * velocity
    finite = (
        jnp.all(jnp.isfinite(mass))
        & jnp.all(jnp.isfinite(velocity))
        & jnp.all(jnp.isfinite(position))
    )
    return _Moments(
        jnp.sum(active, dtype=jnp.int32),
        jnp.sum(mass),
        jnp.sum(charge),
        jnp.sum(jnp.abs(charge)),
        jnp.sum(weighted, axis=0),
        jnp.sum(mass * jnp.sqrt(jnp.sum(velocity * velocity, axis=-1))),
        jnp.sum(mass * _kinetic_per_mass(velocity, light)),
        jnp.sum(charge[:, None] * position, axis=0),
        weighted.T @ velocity,
        (mass[:, None] * position).T @ position,
        finite,
    )


def _identity(state: PICSpeciesState, /) -> Array:
    population = state.population
    return (population.id_hi.astype(jnp.uint64) << 32) | population.id_lo.astype(
        jnp.uint64
    )


def _free_slots(plan: PICSpeciesPlan, state: PICSpeciesState, /) -> Array:
    """Slots `allocate` may fill once the removed particles are deactivated."""
    population = state.population
    free = plan.population.particles.active_mask & ~population.active
    if plan.population.reuse_policy is ParticleSlotReusePolicy.NEVER_REUSE:
        free = free & ~population.ever_occupied & ~population.retired
    return jnp.sum(free, dtype=jnp.int32)


def _canonical_groups(
    keys: Array, valid: Array, stable_ids: Array, key_upper_bound: int, /
) -> tuple[Array, Array, Array, Array, Array]:
    """Items in ``(key, stable id)`` order: order, validity, rank, size, group."""
    capacity = keys.shape[0]
    groups = KeyGroupPlan(capacity, max(capacity, 1), key_upper_bound).build(
        keys, valid, stable_ids=stable_ids
    )
    order = groups.storage_to_logical
    group = jnp.maximum(groups.item_group_slots[order], 0)
    rank = jnp.arange(capacity, dtype=jnp.int32) - groups.group_starts[group]
    return (
        order,
        groups.sorted_item_valid,
        rank,
        groups.group_counts[group],
        group,
    )


def _cell_occupancy(bins: PICCellBins, /) -> Array:
    return jnp.where(bins.binned, bins.cell_counts[jnp.maximum(bins.slot_group, 0)], 0)


def _orthonormal_partner(axis: Array, reference: Array, /) -> Array:
    """Unit vectors orthogonal to ``axis`` in the plane of ``reference``.

    Degenerate references use the coordinate direction least aligned with
    ``axis``; a second Gram-Schmidt pass restores orthogonality to roundoff.
    """
    along = jnp.sum(reference * axis, axis=-1, keepdims=True)
    perpendicular = reference - along * axis
    size = jnp.sqrt(jnp.sum(perpendicular * perpendicular, axis=-1, keepdims=True))
    scale = jnp.sqrt(jnp.sum(reference * reference, axis=-1, keepdims=True))
    coordinate = jax.nn.one_hot(jnp.argmin(jnp.abs(axis), axis=-1), 3, dtype=axis.dtype)
    fallback = coordinate - jnp.sum(coordinate * axis, axis=-1, keepdims=True) * axis
    candidate = jnp.where(size > 1.0e-6 * scale, perpendicular, fallback)
    candidate = candidate / jnp.sqrt(
        jnp.sum(candidate * candidate, axis=-1, keepdims=True)
    )
    candidate = candidate - jnp.sum(candidate * axis, axis=-1, keepdims=True) * axis
    return candidate / jnp.sqrt(jnp.sum(candidate * candidate, axis=-1, keepdims=True))


def _commit(
    context: PICProcessContext,
    plan: PICSpeciesPlan,
    state: PICSpeciesState,
    proposal: _Proposal,
    /,
) -> tuple[PICSpeciesState, Array]:
    """Deactivate removed slots, allocate the canonical request, and scatter it."""
    population = plan.population
    removed = proposal.removed & state.population.active
    deactivated = population.deactivate(state.population, removed).accepted_state
    width = proposal.valid.shape[0]
    particles = state.particles
    allocation = allocate_particles(
        context,
        population,
        deactivated,
        ParticleAllocationRequest(
            jnp.arange(width, dtype=jnp.int32),
            proposal.mass.astype(state.population.mass.dtype),
            proposal.valid,
            parents=(proposal.parent_hi, proposal.parent_lo),
        ),
        proposal.origin.astype(particles.position.dtype),
    )
    capacity = state.population.active.shape[0]
    slots = jnp.where(allocation.allocated, allocation.slots, capacity)
    kept = ~removed
    position = (
        jnp.where(kept[:, None], particles.position, 0.0)
        .at[slots]
        .set(proposal.position.astype(particles.position.dtype), mode="drop")
    )
    velocity = (
        jnp.where(kept[:, None], particles.proper_velocity, 0.0)
        .at[slots]
        .set(
            proposal.proper_velocity.astype(particles.proper_velocity.dtype),
            mode="drop",
        )
    )
    charge = state.charge
    number = (
        jnp.where(kept, charge.charge_number, 0)
        .astype(charge.charge_number.dtype)
        .at[slots]
        .set(proposal.charge_number.astype(charge.charge_number.dtype), mode="drop")
    )
    candidate = PICSpeciesState(
        PICParticleState(position, velocity),
        allocation.candidate_state,
        PICChargeState(
            number,
            charge.transition_count.at[slots].set(0, mode="drop"),
            charge.last_transition_step.at[slots].set(-1, mode="drop"),
        ),
    )
    return candidate, allocation.successful


def _cell_widths(binning: PICCellBinningPlan, /) -> np.ndarray:
    return (np.asarray(binning.upper) - np.asarray(binning.lower)) / np.asarray(
        binning.shape
    )


class _SpeciesOutcome(NamedTuple):
    state: PICSpeciesState
    evidence: ParticleResamplingEvidence
    charge_change: Array
    momentum_change: Array
    energy_change: Array


def _resample_species(
    context: PICProcessContext,
    binning: PICCellBinningPlan,
    plan: PICSpeciesPlan,
    state: PICSpeciesState,
    bins: PICCellBins,
    proposal: _Proposal,
    light: float,
    tolerance: float,
    /,
) -> _SpeciesOutcome:
    """Commit a proposal atomically and measure its conservation and moments."""
    candidate, allocated = _commit(context, plan, state, proposal)
    before = _moments(plan, state, light)
    after = _moments(plan, candidate, light)
    width = float(np.max(_cell_widths(binning)))
    charge_change = after.charge - before.charge
    momentum_change = after.momentum - before.momentum
    energy_change = after.kinetic - before.kinetic
    charge_defect = jnp.abs(charge_change) / jnp.maximum(before.absolute_charge, _TINY)
    momentum_defect = jnp.sqrt(jnp.sum(momentum_change**2)) / jnp.maximum(
        before.momentum_scale, _TINY
    )
    energy_defect = jnp.abs(energy_change) / jnp.maximum(before.kinetic, _TINY)
    dipole_defect = jnp.sqrt(jnp.sum((after.dipole - before.dipole) ** 2)) / jnp.maximum(
        before.absolute_charge * width, _TINY
    )
    conserved = (
        (charge_defect <= tolerance)
        & (momentum_defect <= tolerance)
        & (energy_defect <= tolerance)
        & (dipole_defect <= tolerance)
    )
    finite = before.finite & after.finite
    changed = jnp.any(proposal.valid)
    accept = bins.successful & allocated & conserved & finite & changed
    final = jax.tree.map(lambda new, old: jnp.where(accept, new, old), candidate, state)
    kept = _moments(plan, final, light)
    after_bins = binning.bin(
        final.particles.position,
        final.population.active,
        identity=(final.population.id_hi, final.population.id_lo),
    )
    momentum_norm = jnp.sqrt(jnp.sum(before.momentum_second**2))
    spatial_scale = before.mass * width**2
    status = jnp.asarray(int(ParticleResamplingStatus.NONE), dtype=jnp.int32)
    for flagged, flag in (
        (proposal.refused > 0, ParticleResamplingStatus.CAPACITY_REFUSED),
        (proposal.unsupported > 0, ParticleResamplingStatus.UNSUPPORTED),
        (~bins.successful, ParticleResamplingStatus.BINNING_OVERFLOW),
        (changed & ~allocated, ParticleResamplingStatus.ALLOCATION_REFUSED),
        (changed & ~conserved, ParticleResamplingStatus.CONSERVATION_REFUSED),
        (~finite, ParticleResamplingStatus.NONFINITE),
    ):
        status = jnp.where(flagged, status | int(flag), status)
    events = jnp.where(accept, proposal.events, 0)
    evidence = ParticleResamplingEvidence(
        before.count,
        kept.count,
        events,
        jnp.where(accept, jnp.sum(proposal.removed & state.population.active), 0).astype(
            jnp.int32
        ),
        jnp.where(accept, jnp.sum(proposal.valid), 0).astype(jnp.int32),
        proposal.refused + jnp.where(accept, 0, proposal.events),
        proposal.unsupported,
        bins.maximum_cell_count,
        after_bins.maximum_cell_count,
        jnp.where(accept, charge_defect, 0.0),
        jnp.where(accept, momentum_defect, 0.0),
        jnp.where(accept, energy_defect, 0.0),
        jnp.where(accept, dipole_defect, 0.0),
        jnp.sqrt(jnp.sum((kept.momentum_second - before.momentum_second) ** 2))
        / jnp.maximum(momentum_norm, _TINY),
        jnp.sqrt(jnp.sum((kept.spatial_second - before.spatial_second) ** 2))
        / jnp.maximum(spatial_scale, _TINY),
        status,
        finite,
    )
    return _SpeciesOutcome(
        final,
        evidence,
        kept.charge - before.charge,
        kept.momentum - before.momentum,
        kept.kinetic - before.kinetic,
    )


def _validated_species(species: Sequence[int], /) -> tuple[int, ...]:
    indices = tuple(int(value) for value in species)
    if not indices or any(value < 0 for value in indices):
        raise ValueError("species must be a nonempty sequence of species indices.")
    if len(set(indices)) != len(indices):
        raise ValueError("Resampled species indices must be distinct.")
    return indices


def _validated_positive(name: str, value: int, minimum: int, /) -> int:
    value_ = int(value)
    if value_ < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value_


def _combined_evidence(
    evidence: ParticleResamplingEvidence, /
) -> ParticleResamplingEvidence:
    """One species' evidence from per-device rows (see ``combine``)."""

    def total(value: Array) -> Array:
        return jnp.sum(value, axis=0, dtype=value.dtype)

    def largest(value: Array) -> Array:
        return jnp.max(value, axis=0)

    # Rows are device-sharded: fold the flag sets row by row (a custom
    # reduction cannot be lowered to a cross-device collective).
    status = evidence.status[0]
    for row in range(1, evidence.status.shape[0]):
        status = status | evidence.status[row]

    return ParticleResamplingEvidence(
        total(evidence.active_before),
        total(evidence.active_after),
        total(evidence.events),
        total(evidence.removed),
        total(evidence.created),
        total(evidence.refused),
        total(evidence.unsupported),
        largest(evidence.maximum_cell_count_before),
        largest(evidence.maximum_cell_count_after),
        largest(evidence.charge_defect),
        largest(evidence.momentum_defect),
        largest(evidence.energy_defect),
        largest(evidence.dipole_defect),
        largest(evidence.momentum_second_moment_distortion),
        largest(evidence.spatial_second_moment_distortion),
        status,
        jnp.all(evidence.successful, axis=0),
    )


class _AbstractResamplingProcess(AbstractPICProcess):
    """Shared population-process shell of the resampling plans."""

    binning: eqx.AbstractVar[PICCellBinningPlan]
    relativity: eqx.AbstractVar[RelativityScaleContract]
    speed_of_light: eqx.AbstractVar[float]
    conservation_tolerance: eqx.AbstractVar[float]
    redistributes_charge: ClassVar[bool] = True

    @abc.abstractmethod
    def _request_events(self, plan: PICSpeciesPlan, /) -> int:
        """Static number of events one application can request for ``plan``."""
        raise NotImplementedError

    @abc.abstractmethod
    def _propose(
        self,
        plan: PICSpeciesPlan,
        state: PICSpeciesState,
        bins: PICCellBins,
        /,
    ) -> _Proposal:
        raise NotImplementedError

    def _validate_species(self, plan: PICSpeciesPlan, /) -> int:
        if plan.population.particles.ambient_dimension != len(self.binning.shape):
            raise ValueError("Resampling binning must match the species dimension.")
        events = self._request_events(plan)
        if events <= 0:
            raise ValueError(
                "The species allocation capacity cannot hold one resampling event."
            )
        return events

    def validate_run(
        self,
        species: tuple[PICSpeciesPlan, ...],
        relativity: RelativityScaleContract,
        /,
    ) -> None:
        """Refuse other units than the pusher's and species it cannot resample."""
        if relativity.scale_id != self.relativity.scale_id:
            raise ValueError("Resampling and the PIC pusher use different relativity.")
        for index in self.species_indices:
            self._validate_species(species[index])

    def apply(
        self,
        species: tuple[PICSpeciesPlan, ...],
        context: PICProcessContext,
        /,
    ) -> PICProcessResult:
        values = list(context.species)
        evidence = []
        charge = jnp.zeros((), dtype=jnp.float64)
        momentum = jnp.zeros((), dtype=jnp.float64)
        energy = jnp.zeros((), dtype=jnp.float64)
        events = jnp.zeros((), dtype=jnp.int32)
        successful = jnp.asarray(True)
        for index in self.species_indices:
            plan, state = species[index], values[index]
            self._validate_species(plan)
            bins = self.binning.bin(
                state.particles.position,
                state.population.active,
                identity=(state.population.id_hi, state.population.id_lo),
            )
            outcome = _resample_species(
                context,
                self.binning,
                plan,
                state,
                bins,
                self._propose(plan, state, bins),
                self.speed_of_light,
                self.conservation_tolerance,
            )
            values[index] = outcome.state
            evidence.append(outcome.evidence)
            charge = jnp.maximum(charge, jnp.abs(outcome.charge_change))
            momentum = jnp.maximum(
                momentum, jnp.sqrt(jnp.sum(outcome.momentum_change**2))
            )
            energy = jnp.maximum(energy, jnp.abs(outcome.energy_change))
            events = events + outcome.evidence.events
            successful = successful & outcome.evidence.successful
        return PICProcessResult(
            tuple(values),
            PICProcessLedger(
                events, charge, momentum, energy, successful, self.process_id
            ),
            tuple(evidence),
        )

    # -- distribution ----------------------------------------------------------

    def localize(
        self, species: tuple[PICSpeciesPlan, ...], parts: int, /
    ) -> _AbstractResamplingProcess:
        """The process on one device's slot blocks.

        Request widths and occupancy triggers derive from the species plans it
        is applied with, so the block-capacity plans localize them; the plan
        itself is unchanged once every block can hold one resampling event.
        """
        if parts <= 0:
            raise ValueError("parts must be positive.")
        for index in self.species_indices:
            self._validate_species(species[index])
        return self

    def bank_plans(self) -> tuple[ParticlePopulationPlan, ...]:
        """Resampling owns no particle bank."""
        return ()

    def partition_state(self, state: None, /) -> PICProcessStatePartition:
        """Resampling is stateless: an empty partition."""
        if state is not None:
            raise TypeError("Resampling processes carry no state.")
        return PICProcessStatePartition((), (), (), (), ())

    def assemble_state(self, partition: PICProcessStatePartition, /) -> None:
        """Resampling is stateless."""
        if (
            partition.companions
            or partition.banks
            or partition.totals
            or partition.shared
        ):
            raise ValueError("Resampling processes carry no state.")

    def combine(
        self,
        ledger: PICProcessLedger,
        evidence: tuple[ParticleResamplingEvidence, ...],
        /,
    ) -> tuple[PICProcessLedger, tuple[ParticleResamplingEvidence, ...]]:
        """The run's ledger and per-species evidence from per-device rows.

        Event, particle, refusal and support counts add; the ledger defects
        (largest absolute species change), relative conservation defects,
        moment distortions and maximal cell counts take the largest device
        value; status flags combine by bitwise or and success requires every
        device.
        """
        return PICProcessLedger(
            jnp.sum(ledger.event_count, axis=0, dtype=ledger.event_count.dtype),
            jnp.max(ledger.charge_defect, axis=0),
            jnp.max(ledger.momentum_defect, axis=0),
            jnp.max(ledger.energy_defect, axis=0),
            jnp.all(ledger.successful, axis=0),
            self.process_id,
        ), tuple(_combined_evidence(value) for value in evidence)


class ParticleMergePlan(_AbstractResamplingProcess, NonTrainableState):
    """Vranic momentum-cell merging of crowded cells into exact pairs."""

    binning: PICCellBinningPlan
    relativity: RelativityScaleContract = eqx.field(static=True)
    method: ParticleMergeMethod = eqx.field(static=True)
    maximum_per_cell: int = eqx.field(static=True)
    minimum_occupancy: float = eqx.field(static=True)
    momentum_bins: tuple[int, int, int] = eqx.field(static=True)
    minimum_packet_size: int = eqx.field(static=True)
    maximum_packet_size: int = eqx.field(static=True)
    speed_of_light: float = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    stage: PICProcessStage = eqx.field(static=True)
    stochastic: bool = eqx.field(static=True)
    radiation_ownership: RadiationOwnership | None = eqx.field(static=True)
    species_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        binning: PICCellBinningPlan,
        relativity: RelativityScaleContract,
        /,
        *,
        species: Sequence[int],
        maximum_per_cell: int,
        method: ParticleMergeMethod = "vranic-momentum-cell",
        momentum_bins: tuple[int, int, int] = (4, 4, 8),
        minimum_packet_size: int = 4,
        maximum_packet_size: int = 8,
        conservation_tolerance: float = 1.0e-10,
        minimum_occupancy: float = 0.0,
    ) -> None:
        """Merge species ``species`` in cells holding more than ``maximum_per_cell``.

        ``momentum_bins`` counts (magnitude, polar-cosine, azimuth) momentum
        cells; momentum cells are cut into packets of ``maximum_packet_size``
        in identity order, and a trailing packet merges only when it holds at
        least ``minimum_packet_size`` (≥ 3) particles. Merging is triggered
        only while a species' active fraction of its capacity is at least
        ``minimum_occupancy`` (a QED cascade's merge request), below which the
        species is left unchanged.
        """
        if not isinstance(binning, PICCellBinningPlan):
            raise TypeError("binning must be PICCellBinningPlan.")
        if not isinstance(relativity, RelativityScaleContract):
            raise TypeError("relativity must be a RelativityScaleContract.")
        method_ = parse(method, ParticleMergeMethod, "method")
        bins = tuple(int(value) for value in momentum_bins)
        if len(bins) != 3 or any(value <= 0 for value in bins):
            raise ValueError("momentum_bins must be three positive counts.")
        minimum = _validated_positive("minimum_packet_size", minimum_packet_size, 3)
        maximum = _validated_positive("maximum_packet_size", maximum_packet_size, minimum)
        threshold = _validated_positive("maximum_per_cell", maximum_per_cell, 1)
        occupancy = float(minimum_occupancy)
        if not 0.0 <= occupancy <= 1.0:
            raise ValueError("minimum_occupancy must lie in [0, 1].")
        tolerance = float(conservation_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("conservation_tolerance must be positive and finite.")
        indices = _validated_species(species)
        self.binning = binning
        self.relativity = relativity
        self.method = method_
        self.maximum_per_cell = threshold
        self.minimum_occupancy = occupancy
        self.momentum_bins = (bins[0], bins[1], bins[2])
        self.minimum_packet_size = minimum
        self.maximum_packet_size = maximum
        self.speed_of_light = float(relativity.speed_of_light)
        self.conservation_tolerance = tolerance
        self.stage = "population"
        self.stochastic = False
        self.radiation_ownership = None
        self.species_indices = indices
        self.process_id = canonical_fingerprint(
            {
                "kind": "pic-particle-merge",
                "binning": binning.plan_id,
                "relativity": relativity.scale_id,
                "method": method_,
                "maximum_per_cell": threshold,
                "minimum_occupancy": occupancy,
                "momentum_bins": list(self.momentum_bins),
                "packet_sizes": [minimum, maximum],
                "tolerance": tolerance,
                "species": list(indices),
            }
        )

    def _request_events(self, plan: PICSpeciesPlan, /) -> int:
        return min(
            plan.capacity // self.minimum_packet_size,
            plan.population.allocation_capacity // 2,
        )

    def _momentum_cells(
        self, velocity: Array, bins: PICCellBins, crowded: Array, /
    ) -> Array:
        """Spherical momentum cell; magnitudes span each spatial cell's speeds."""
        radial_count, polar_count, azimuth_count = self.momentum_bins
        speed = jnp.sqrt(jnp.sum(velocity * velocity, axis=-1))
        group = jnp.maximum(bins.slot_group, 0)
        groups = bins.cell_counts.shape[0]
        low = jax.ops.segment_min(
            jnp.where(crowded, speed, jnp.inf), group, num_segments=groups
        )[group]
        high = jax.ops.segment_max(
            jnp.where(crowded, speed, -jnp.inf), group, num_segments=groups
        )[group]
        span = high - low
        spread = crowded & (span > 0.0)
        radial = jnp.where(
            spread, (speed - low) / jnp.where(spread, span, 1.0) * radial_count, 0.0
        )
        moving = speed > 0.0
        polar_cosine = jnp.where(
            moving, velocity[:, 2] / jnp.where(moving, speed, 1.0), 1.0
        )
        azimuth = jnp.arctan2(velocity[:, 1], velocity[:, 0])
        radial_index = jnp.clip(jnp.floor(radial), 0, radial_count - 1)
        polar_index = jnp.clip(
            jnp.floor(0.5 * (1.0 - polar_cosine) * polar_count), 0, polar_count - 1
        )
        azimuth_index = jnp.clip(
            jnp.floor((azimuth + jnp.pi) / (2.0 * jnp.pi) * azimuth_count),
            0,
            azimuth_count - 1,
        )
        return (
            (radial_index * polar_count + polar_index) * azimuth_count + azimuth_index
        ).astype(jnp.int64)

    def _propose(
        self,
        plan: PICSpeciesPlan,
        state: PICSpeciesState,
        bins: PICCellBins,
        /,
    ) -> _Proposal:
        match self.method:
            case "vranic-momentum-cell":
                return self._vranic(plan, state, bins)
            case _:
                assert_never(self.method)

    def _vranic(
        self, plan: PICSpeciesPlan, state: PICSpeciesState, bins: PICCellBins, /
    ) -> _Proposal:
        population = state.population
        capacity = population.active.shape[0]
        light = self.speed_of_light
        velocity = state.particles.proper_velocity.astype(jnp.float64)
        # The occupancy trigger stays traced: below it no cell is crowded.
        triggered = jnp.sum(population.active) / plan.capacity >= self.minimum_occupancy
        crowded = (
            bins.binned & (_cell_occupancy(bins) > self.maximum_per_cell) & triggered
        )
        model = plan.charge_model
        charge_states = model.maximum_charge_number - model.minimum_charge_number + 1
        momentum_cells = int(np.prod(self.momentum_bins))
        charge_index = jnp.clip(
            state.charge.charge_number.astype(jnp.int64) - model.minimum_charge_number,
            0,
            charge_states - 1,
        )
        keys = (
            bins.cell.astype(jnp.int64) * charge_states + charge_index
        ) * momentum_cells + self._momentum_cells(velocity, bins, crowded)
        order, valid, rank, size, _ = _canonical_groups(
            keys,
            crowded & bins.successful,
            _identity(state),
            self.binning.cell_count * charge_states * momentum_cells - 1,
        )
        packet_size, packet_minimum = self.maximum_packet_size, self.minimum_packet_size
        full = size // packet_size
        remainder = size - full * packet_size
        local = rank // packet_size
        merged = valid & ((local < full) | (remainder >= packet_minimum))
        # Packets are numbered in canonical order: each group's packets follow
        # every packet of the groups before it.
        starts = merged & (rank % packet_size == 0)
        packet = jnp.cumsum(starts, dtype=jnp.int32) - 1
        total = jnp.sum(starts, dtype=jnp.int32)
        pairs = self._request_events(plan)
        limit = jnp.asarray(pairs, dtype=jnp.int32)
        if plan.population.reuse_policy is ParticleSlotReusePolicy.NEVER_REUSE:
            limit = jnp.minimum(limit, _free_slots(plan, state) // 2)
        accepted = merged & (packet < limit)
        segment = jnp.where(accepted, packet, pairs)
        mass = population.mass[order].astype(jnp.float64)
        sorted_velocity = velocity[order]
        position = state.particles.position[order].astype(jnp.float64)
        weight = jax.ops.segment_sum(mass, segment, num_segments=pairs)
        momentum = jax.ops.segment_sum(
            mass[:, None] * sorted_velocity, segment, num_segments=pairs
        )
        kinetic = jax.ops.segment_sum(
            mass * _kinetic_per_mass(sorted_velocity, light), segment, num_segments=pairs
        )
        center = jax.ops.segment_sum(
            mass[:, None] * position, segment, num_segments=pairs
        )
        first = (
            jnp.zeros((pairs,), dtype=jnp.int32)
            .at[jnp.where(accepted & starts, packet, pairs)]
            .set(jnp.arange(capacity, dtype=jnp.int32), mode="drop")
        )
        active_pair = jnp.arange(pairs, dtype=jnp.int32) < jnp.minimum(total, limit)
        safe_weight = jnp.where(active_pair, weight, 1.0)
        gamma_minus_one = kinetic / (safe_weight * light**2)
        speed = light * jnp.sqrt(gamma_minus_one * (gamma_minus_one + 2.0))
        norm = jnp.sqrt(jnp.sum(momentum * momentum, axis=-1))
        axis = jnp.where(
            (norm > 0.0)[:, None],
            momentum / jnp.where(norm > 0.0, norm, 1.0)[:, None],
            jnp.asarray([0.0, 0.0, 1.0]),
        )
        cosine = jnp.where(
            speed > 0.0,
            jnp.clip(norm / (safe_weight * jnp.where(speed > 0.0, speed, 1.0)), 0.0, 1.0),
            1.0,
        )
        sine = jnp.sqrt((1.0 - cosine) * (1.0 + cosine))
        partner = _orthonormal_partner(axis, sorted_velocity[first])
        along = (speed * cosine)[:, None] * axis
        across = (speed * sine)[:, None] * partner
        products = jnp.stack((along + across, along - across), axis=1).reshape(
            (2 * pairs, 3)
        )
        center = center / safe_weight[:, None]
        removed = jnp.zeros((capacity,), dtype=jnp.bool_).at[order].set(accepted)
        # Both products sit at, and are created at, the packet's center of mass.
        product_position = jnp.repeat(center, 2, axis=0)
        return _Proposal(
            removed,
            jnp.repeat(active_pair, 2),
            jnp.repeat(0.5 * weight, 2),
            jnp.repeat(population.id_hi[order][first], 2),
            jnp.repeat(population.id_lo[order][first], 2),
            product_position,
            product_position,
            products,
            jnp.repeat(state.charge.charge_number[order][first], 2),
            jnp.minimum(total, limit),
            total - jnp.minimum(total, limit),
            jnp.zeros((), dtype=jnp.int32),
        )


class ParticleSplitPlan(_AbstractResamplingProcess, NonTrainableState):
    """Split the heaviest particles of sparse cells into ``2d`` axis-pair children."""

    binning: PICCellBinningPlan
    relativity: RelativityScaleContract = eqx.field(static=True)
    minimum_per_cell: int = eqx.field(static=True)
    displacement_fraction: float = eqx.field(static=True)
    minimum_child_mass: float = eqx.field(static=True)
    maximum_splits: int = eqx.field(static=True)
    speed_of_light: float = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    stage: PICProcessStage = eqx.field(static=True)
    stochastic: bool = eqx.field(static=True)
    radiation_ownership: RadiationOwnership | None = eqx.field(static=True)
    species_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        binning: PICCellBinningPlan,
        relativity: RelativityScaleContract,
        /,
        *,
        species: Sequence[int],
        minimum_per_cell: int,
        minimum_child_mass: float,
        maximum_splits: int,
        displacement_fraction: float = 0.25,
        conservation_tolerance: float = 1.0e-10,
    ) -> None:
        """Split in cells holding fewer than ``minimum_per_cell`` particles.

        Each cell splits its heaviest particles (identity breaks ties) until
        its count reaches ``minimum_per_cell``; children lighter than
        ``minimum_child_mass`` are never made. ``maximum_splits`` bounds one
        application's request; children sit ``±displacement_fraction`` cell
        widths from the parent along each axis.
        """
        if not isinstance(binning, PICCellBinningPlan):
            raise TypeError("binning must be PICCellBinningPlan.")
        if not isinstance(relativity, RelativityScaleContract):
            raise TypeError("relativity must be a RelativityScaleContract.")
        threshold = _validated_positive("minimum_per_cell", minimum_per_cell, 1)
        splits = _validated_positive("maximum_splits", maximum_splits, 1)
        fraction = float(displacement_fraction)
        child = float(minimum_child_mass)
        tolerance = float(conservation_tolerance)
        if not 0.0 < fraction < 0.5:
            raise ValueError("displacement_fraction must lie in (0, 0.5).")
        if not np.isfinite(child) or child <= 0.0:
            raise ValueError("minimum_child_mass must be positive and finite.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("conservation_tolerance must be positive and finite.")
        indices = _validated_species(species)
        self.binning = binning
        self.relativity = relativity
        self.minimum_per_cell = threshold
        self.displacement_fraction = fraction
        self.minimum_child_mass = child
        self.maximum_splits = splits
        self.speed_of_light = float(relativity.speed_of_light)
        self.conservation_tolerance = tolerance
        self.stage = "population"
        self.stochastic = False
        self.radiation_ownership = None
        self.species_indices = indices
        self.process_id = canonical_fingerprint(
            {
                "kind": "pic-particle-split",
                "binning": binning.plan_id,
                "relativity": relativity.scale_id,
                "minimum_per_cell": threshold,
                "displacement_fraction": fraction,
                "minimum_child_mass": child,
                "maximum_splits": splits,
                "tolerance": tolerance,
                "species": list(indices),
            }
        )

    def _request_events(self, plan: PICSpeciesPlan, /) -> int:
        children = 2 * len(self.binning.shape)
        return min(self.maximum_splits, plan.population.allocation_capacity // children)

    def _propose(
        self,
        plan: PICSpeciesPlan,
        state: PICSpeciesState,
        bins: PICCellBins,
        /,
    ) -> _Proposal:
        population = state.population
        capacity = population.active.shape[0]
        dimension = len(self.binning.shape)
        children = 2 * dimension
        splits = self._request_events(plan)
        delta = self.displacement_fraction * _cell_widths(self.binning)
        position = state.particles.position.astype(jnp.float64)
        mass = population.mass.astype(jnp.float64)
        occupancy = _cell_occupancy(bins)
        candidate = (
            bins.binned
            & bins.successful
            & (occupancy < self.minimum_per_cell)
            & (mass >= children * self.minimum_child_mass)
        )
        inside = jnp.ones((capacity,), dtype=jnp.bool_)
        for axis, periodic in enumerate(self.binning.periodic):
            if not periodic:
                inside = (
                    inside
                    & (position[:, axis] - delta[axis] >= self.binning.lower[axis])
                    & (position[:, axis] + delta[axis] <= self.binning.upper[axis])
                )
        supported = candidate & inside
        # Heaviest first, identity breaking ties: a slot-independent priority.
        priority = jnp.lexsort(
            (population.id_lo, population.id_hi, -jnp.where(supported, mass, 0.0))
        )
        priority_rank = (
            jnp.zeros((capacity,), dtype=jnp.int64)
            .at[priority]
            .set(jnp.arange(capacity, dtype=jnp.int64))
        )
        order, valid, rank, _, _ = _canonical_groups(
            bins.cell.astype(jnp.int64),
            supported,
            priority_rank,
            self.binning.cell_count - 1,
        )
        need = (self.minimum_per_cell - occupancy + children - 2) // (children - 1)
        wanted = valid & (rank < need[order])
        position_in_request = jnp.cumsum(wanted, dtype=jnp.int32) - 1
        total = jnp.sum(wanted, dtype=jnp.int32)
        free = _free_slots(plan, state)
        net = (
            children
            if plan.population.reuse_policy is ParticleSlotReusePolicy.NEVER_REUSE
            else children - 1
        )
        limit = jnp.minimum(jnp.asarray(splits, dtype=jnp.int32), free // net)
        accepted = wanted & (position_in_request < limit)
        parent = (
            jnp.zeros((splits,), dtype=jnp.int32)
            .at[jnp.where(accepted, position_in_request, splits)]
            .set(order, mode="drop")
        )
        active_split = jnp.arange(splits, dtype=jnp.int32) < jnp.minimum(total, limit)
        offsets = np.zeros((children, dimension), dtype=np.float64)
        for axis in range(dimension):
            offsets[2 * axis, axis] = -delta[axis]
            offsets[2 * axis + 1, axis] = delta[axis]
        child_position = position[parent][:, None, :] + jnp.asarray(offsets)[None]
        removed = jnp.zeros((capacity,), dtype=jnp.bool_).at[order].set(accepted)
        return _Proposal(
            removed,
            jnp.repeat(active_split, children),
            jnp.repeat(mass[parent] / children, children),
            jnp.repeat(population.id_hi[parent], children),
            jnp.repeat(population.id_lo[parent], children),
            child_position.reshape((splits * children, dimension)),
            jnp.repeat(position[parent], children, axis=0),
            jnp.repeat(
                state.particles.proper_velocity[parent].astype(jnp.float64),
                children,
                axis=0,
            ),
            jnp.repeat(state.charge.charge_number[parent], children),
            jnp.minimum(total, limit),
            total - jnp.minimum(total, limit),
            jnp.sum(candidate & ~inside, dtype=jnp.int32),
        )


__all__ = [
    "ParticleMergeMethod",
    "ParticleMergePlan",
    "ParticleResamplingEvidence",
    "ParticleResamplingStatus",
    "ParticleSplitPlan",
]
