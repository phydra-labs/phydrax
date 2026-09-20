#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded nonrelativistic ``2 <-> 2`` dark-sector reaction transactions."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._dark_radiation import (
    DarkRadiationExportEvidence,
    DarkRadiationLedger,
    DarkRadiationLedgerPlan,
    DarkRadiationPacket,
)
from ._dark_sector_species import DarkSectorSpeciesPlan
from ._sidm_kernels import (
    angles_from_direction,
    directions_from_angles,
    TwoBodyDifferentialKernelPlan,
)
from ._sidm_weighted import WeightedSIDMPacketState


_CHECKPOINT_FORMAT = "phydrax-inelastic-sidm-checkpoint"


def _segment_samples(nodes: Array, /, *, periodic: bool = False) -> np.ndarray:
    values = np.asarray(nodes, dtype=np.float64)
    if periodic:
        values = values[:-1]
    if values.size == 1:
        return values
    source = np.asarray(nodes, dtype=np.float64)
    midpoints = 0.5 * (source[:-1] + source[1:])
    return np.unique(np.concatenate((values, midpoints)))


def _reciprocity_angle_grid(
    forward: TwoBodyDifferentialKernelPlan,
    reverse: TwoBodyDifferentialKernelPlan,
    /,
) -> tuple[Array, Array, Array, Array]:
    def native_grid(kernel):
        cosines = _segment_samples(kernel.cosines)
        azimuths = (
            np.asarray((0.0,))
            if kernel.azimuths is None
            else _segment_samples(kernel.azimuths, periodic=True)
        )
        mu, phi = np.meshgrid(cosines, azimuths, indexing="ij")
        return mu.reshape((-1,)), phi.reshape((-1,))

    forward_mu, forward_phi = native_grid(forward)
    reverse_mu, reverse_phi = native_grid(reverse)
    forward_relative = jnp.broadcast_to(
        jnp.asarray((1.0, 0.0, 0.0)), (forward_mu.size, 3)
    )
    forward_direction = directions_from_angles(
        forward_relative, jnp.asarray(forward_mu), jnp.asarray(forward_phi)
    )
    mapped_reverse_mu, mapped_reverse_phi = angles_from_direction(
        -forward_direction, -forward_relative
    )
    reverse_relative = jnp.broadcast_to(
        jnp.asarray((1.0, 0.0, 0.0)), (reverse_mu.size, 3)
    )
    reverse_direction = directions_from_angles(
        reverse_relative, jnp.asarray(reverse_mu), jnp.asarray(reverse_phi)
    )
    mapped_forward_mu, mapped_forward_phi = angles_from_direction(
        -reverse_direction, -reverse_relative
    )
    return tuple(
        jax.lax.stop_gradient(value)
        for value in (
            jnp.concatenate((jnp.asarray(forward_mu), mapped_forward_mu)),
            jnp.concatenate((jnp.asarray(forward_phi), mapped_forward_phi)),
            jnp.concatenate((mapped_reverse_mu, jnp.asarray(reverse_mu))),
            jnp.concatenate((mapped_reverse_phi, jnp.asarray(reverse_phi))),
        )
    )


def _packet_lineage_valid(packets: WeightedSIDMPacketState, /) -> Array:
    """Validate stable IDs and depth-ordered active-parent lineage in linear memory."""

    order = jnp.argsort(packets.packet_ids)
    sorted_ids = packets.packet_ids[order]
    unique = jnp.all(sorted_ids[1:] != sorted_ids[:-1])
    insertion = jnp.searchsorted(sorted_ids, packets.parent_packet_ids, side="left")
    safe_insertion = jnp.clip(insertion, 0, sorted_ids.size - 1)
    parent_slots = order[safe_insertion]
    parent_found = (
        (packets.parent_packet_ids >= 0)
        & (insertion < sorted_ids.size)
        & (sorted_ids[safe_insertion] == packets.parent_packet_ids)
    )
    parent_depth = packets.lineage_depth[parent_slots]
    parent_active = packets.active_mask[parent_slots]
    active_valid = ((packets.lineage_depth == 0) & (packets.parent_packet_ids == -1)) | (
        (packets.lineage_depth > 0)
        & parent_found
        & parent_active
        & (packets.parent_packet_ids != packets.packet_ids)
        & (parent_depth >= 0)
        & (parent_depth < packets.lineage_depth)
    )
    inactive_valid = (packets.parent_packet_ids == -1) & (packets.lineage_depth == -1)
    return (
        jnp.all(packets.packet_ids >= 0)
        & unique
        & jnp.all(jnp.where(packets.active_mask, active_valid, inactive_valid))
    )


class DarkReactionStatus(IntEnum):
    """Terminal status for one conditional two-body reaction attempt."""

    SUCCESS = 0
    INVALID_PAIR = 1
    THRESHOLD_CLOSED = 2
    KERNEL_UNSUPPORTED = 3
    RELATIVISTIC_REFUSED = 4
    CAPACITY_EXHAUSTED = 5
    DETAILED_BALANCE_FAILURE = 6
    CONSERVATION_FAILURE = 7
    INVALID_STATE = 8


class DarkTwoBodyReactionPlan(StrictModule, NonTrainableState):
    """One reversible microscopic ``2 <-> 2`` channel.

    Rest-mass energy and species internal energy use the species' explicit
    physical units. The forward and reverse differential kernels are separate
    reference products, but every committed event must satisfy the declared
    microreversibility tolerance at its conjugate incoming/outgoing speeds.
    """

    incoming_species: tuple[DarkSectorSpeciesPlan, DarkSectorSpeciesPlan]
    outgoing_species: tuple[DarkSectorSpeciesPlan, DarkSectorSpeciesPlan]
    forward_kernel: TwoBodyDifferentialKernelPlan
    reverse_kernel: TwoBodyDifferentialKernelPlan
    reciprocity_forward_cosines: Array
    reciprocity_forward_azimuths: Array
    reciprocity_reverse_cosines: Array
    reciprocity_reverse_azimuths: Array
    conserved_charge_names: tuple[str, ...] = eqx.field(static=True)
    charge_residual: Array
    incoming_degeneracy: int = eqx.field(static=True)
    outgoing_degeneracy: int = eqx.field(static=True)
    incoming_pair_symmetry: int = eqx.field(static=True)
    outgoing_pair_symmetry: int = eqx.field(static=True)
    forward_outgoing_convention: str = eqx.field(static=True)
    reverse_outgoing_convention: str = eqx.field(static=True)
    incoming_reduced_mass: float = eqx.field(static=True)
    outgoing_reduced_mass: float = eqx.field(static=True)
    energy_change: float = eqx.field(static=True)
    mass_unit: str = eqx.field(static=True)
    energy_unit: str = eqx.field(static=True)
    speed_unit: str = eqx.field(static=True)
    cross_section_unit: str = eqx.field(static=True)
    momentum_unit: str = eqx.field(static=True)
    position_unit: str = eqx.field(static=True)
    speed_of_light: float = eqx.field(static=True)
    maximum_speed_fraction: float = eqx.field(static=True)
    detailed_balance_tolerance: float = eqx.field(static=True)
    channel_id: str = eqx.field(static=True)

    def __init__(
        self,
        incoming_species: tuple[DarkSectorSpeciesPlan, DarkSectorSpeciesPlan],
        outgoing_species: tuple[DarkSectorSpeciesPlan, DarkSectorSpeciesPlan],
        forward_kernel: TwoBodyDifferentialKernelPlan,
        reverse_kernel: TwoBodyDifferentialKernelPlan,
        /,
        *,
        conserved_charge_names: Sequence[str] | None = None,
        forward_outgoing_convention: str | None = None,
        reverse_outgoing_convention: str | None = None,
        speed_of_light: float = 1.0,
        maximum_speed_fraction: float = 0.1,
        charge_tolerance: float = 0.0,
        detailed_balance_tolerance: float = 1.0e-10,
    ):
        if (
            not isinstance(incoming_species, tuple)
            or len(incoming_species) != 2
            or not all(
                isinstance(item, DarkSectorSpeciesPlan) for item in incoming_species
            )
        ):
            raise TypeError(
                "incoming_species must be a pair of DarkSectorSpeciesPlan objects."
            )
        if (
            not isinstance(outgoing_species, tuple)
            or len(outgoing_species) != 2
            or not all(
                isinstance(item, DarkSectorSpeciesPlan) for item in outgoing_species
            )
        ):
            raise TypeError(
                "outgoing_species must be a pair of DarkSectorSpeciesPlan objects."
            )
        if not isinstance(
            forward_kernel, TwoBodyDifferentialKernelPlan
        ) or not isinstance(reverse_kernel, TwoBodyDifferentialKernelPlan):
            raise TypeError(
                "Reaction kernels must be TwoBodyDifferentialKernelPlan objects."
            )
        incoming_ids = tuple(item.species_plan_id for item in incoming_species)
        outgoing_ids = tuple(item.species_plan_id for item in outgoing_species)
        if (
            forward_kernel.first_species.species_plan_id,
            forward_kernel.second_species.species_plan_id,
        ) != incoming_ids:
            raise ValueError(
                "Forward kernel species do not match the incoming pair order."
            )
        if (
            reverse_kernel.first_species.species_plan_id,
            reverse_kernel.second_species.species_plan_id,
        ) != outgoing_ids:
            raise ValueError(
                "Reverse kernel species do not match the outgoing pair order."
            )
        species = (*incoming_species, *outgoing_species)
        mass_units = {item.mass_unit for item in species}
        energy_units = {item.energy_unit for item in species}
        if len(mass_units) != 1 or len(energy_units) != 1:
            raise ValueError(
                "All reaction species must share explicit mass and energy units."
            )
        if (
            forward_kernel.speed_unit != reverse_kernel.speed_unit
            or forward_kernel.cross_section_unit != reverse_kernel.cross_section_unit
        ):
            raise ValueError("Forward and reverse kernels must share physical units.")
        forward_outgoing_identical = outgoing_ids[0] == outgoing_ids[1]
        reverse_outgoing_identical = incoming_ids[0] == incoming_ids[1]

        def outgoing_convention(value, identical, direction):
            if value is None:
                if identical:
                    raise ValueError(
                        f"{direction}_outgoing_convention is required for an identical outgoing pair."
                    )
                return "distinguishable-full-sphere"
            convention = str(value)
            if identical and convention not in (
                "labeled-full-sphere",
                "exchange-quotient",
            ):
                raise ValueError(
                    f"{direction} identical outgoing species require labeled-full-"
                    "sphere or exchange-quotient normalization."
                )
            if not identical and convention != "distinguishable-full-sphere":
                raise ValueError(
                    f"{direction} distinct outgoing species require distinguishable-full-sphere normalization."
                )
            return convention

        forward_outgoing = outgoing_convention(
            forward_outgoing_convention,
            forward_outgoing_identical,
            "forward",
        )
        reverse_outgoing = outgoing_convention(
            reverse_outgoing_convention,
            reverse_outgoing_identical,
            "reverse",
        )
        if (
            forward_outgoing != reverse_kernel.identical_particle_convention
            or reverse_outgoing != forward_kernel.identical_particle_convention
        ):
            raise ValueError(
                "Outgoing angular normalization must match the time-reversed kernel's incoming-pair convention."
            )
        mass_unit = species[0].mass_unit
        energy_unit = species[0].energy_unit
        speed_unit = forward_kernel.speed_unit
        cross_section_unit = forward_kernel.cross_section_unit
        momentum_unit = f"{mass_unit}*{speed_unit}"
        (
            reciprocity_forward_cosines,
            reciprocity_forward_azimuths,
            reciprocity_reverse_cosines,
            reciprocity_reverse_azimuths,
        ) = _reciprocity_angle_grid(forward_kernel, reverse_kernel)
        forward_support = (reciprocity_forward_cosines >= forward_kernel.cosines[0]) & (
            reciprocity_forward_cosines <= forward_kernel.cosines[-1]
        )
        reverse_support = (reciprocity_reverse_cosines >= reverse_kernel.cosines[0]) & (
            reciprocity_reverse_cosines <= reverse_kernel.cosines[-1]
        )
        if not bool(np.asarray(jnp.all(forward_support & reverse_support))):
            raise ValueError(
                "Forward/reverse angular conventions do not map to common support."
            )
        position_unit = "comoving-length"
        light_speed = float(speed_of_light)
        maximum_fraction = float(maximum_speed_fraction)
        charge_tolerance_ = float(charge_tolerance)
        balance_tolerance = float(detailed_balance_tolerance)
        if not np.isfinite(light_speed) or light_speed <= 0.0:
            raise ValueError("speed_of_light must be finite and positive.")
        if (
            not np.isfinite(maximum_fraction)
            or maximum_fraction <= 0.0
            or maximum_fraction >= 1.0
        ):
            raise ValueError(
                "maximum_speed_fraction must lie strictly between zero and one."
            )
        if not np.isfinite(charge_tolerance_) or charge_tolerance_ < 0.0:
            raise ValueError("charge_tolerance must be finite and nonnegative.")
        if not np.isfinite(balance_tolerance) or balance_tolerance <= 0.0:
            raise ValueError("detailed_balance_tolerance must be finite and positive.")

        declared_charge_names = incoming_species[0].charge_names
        if any(item.charge_names != declared_charge_names for item in species):
            raise ValueError(
                "All reaction species must share one ordered conserved-charge basis."
            )
        if conserved_charge_names is None:
            names = incoming_species[0].charge_names
        else:
            names = tuple(str(name).strip() for name in conserved_charge_names)
        if names != declared_charge_names:
            raise ValueError(
                "conserved_charge_names must include the complete species charge basis."
            )
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Conserved charge names must be non-empty and unique.")
        incoming_charge = np.asarray(
            [
                sum(float(np.asarray(item.charge(name))) for item in incoming_species)
                for name in names
            ]
        )
        outgoing_charge = np.asarray(
            [
                sum(float(np.asarray(item.charge(name))) for item in outgoing_species)
                for name in names
            ]
        )
        charge_residual = outgoing_charge - incoming_charge
        if np.any(np.abs(charge_residual) > charge_tolerance_):
            raise ValueError("The declared two-body reaction does not conserve charge.")

        incoming_mass = tuple(item.mass for item in incoming_species)
        outgoing_mass = tuple(item.mass for item in outgoing_species)
        incoming_mu = incoming_mass[0] * incoming_mass[1] / sum(incoming_mass)
        outgoing_mu = outgoing_mass[0] * outgoing_mass[1] / sum(outgoing_mass)
        incoming_energy = sum(
            item.mass * light_speed * light_speed + item.internal_energy
            for item in incoming_species
        )
        outgoing_energy = sum(
            item.mass * light_speed * light_speed + item.internal_energy
            for item in outgoing_species
        )
        incoming_degeneracy = int(
            incoming_species[0].degeneracy * incoming_species[1].degeneracy
        )
        outgoing_degeneracy = int(
            outgoing_species[0].degeneracy * outgoing_species[1].degeneracy
        )
        incoming_symmetry = 2 if incoming_ids[0] == incoming_ids[1] else 1
        outgoing_symmetry = 2 if outgoing_ids[0] == outgoing_ids[1] else 1
        self.incoming_species = incoming_species
        self.outgoing_species = outgoing_species
        self.forward_kernel = forward_kernel
        self.reverse_kernel = reverse_kernel
        self.conserved_charge_names = names
        self.charge_residual = jax.lax.stop_gradient(jnp.asarray(charge_residual))
        self.incoming_degeneracy = incoming_degeneracy
        self.outgoing_degeneracy = outgoing_degeneracy
        self.incoming_pair_symmetry = incoming_symmetry
        self.outgoing_pair_symmetry = outgoing_symmetry
        self.forward_outgoing_convention = forward_outgoing
        self.reciprocity_forward_cosines = reciprocity_forward_cosines
        self.reciprocity_forward_azimuths = reciprocity_forward_azimuths
        self.reciprocity_reverse_cosines = reciprocity_reverse_cosines
        self.reciprocity_reverse_azimuths = reciprocity_reverse_azimuths
        self.reverse_outgoing_convention = reverse_outgoing
        self.incoming_reduced_mass = incoming_mu
        self.outgoing_reduced_mass = outgoing_mu
        self.energy_change = outgoing_energy - incoming_energy
        self.mass_unit = mass_unit
        self.energy_unit = energy_unit
        self.speed_unit = speed_unit
        self.cross_section_unit = cross_section_unit
        self.momentum_unit = momentum_unit
        self.position_unit = position_unit
        self.speed_of_light = light_speed
        self.maximum_speed_fraction = maximum_fraction
        self.detailed_balance_tolerance = balance_tolerance
        self.channel_id = canonical_fingerprint(
            {
                "kind": "reversible-dark-two-body-reaction",
                "incoming": list(incoming_ids),
                "outgoing": list(outgoing_ids),
                "forward_kernel": forward_kernel.kernel_id,
                "reverse_kernel": reverse_kernel.kernel_id,
                "conserved_charges": list(names),
                "charge_residual": array_tree_fingerprint(charge_residual),
                "incoming_degeneracy": incoming_degeneracy,
                "outgoing_degeneracy": outgoing_degeneracy,
                "incoming_pair_symmetry": incoming_symmetry,
                "outgoing_pair_symmetry": outgoing_symmetry,
                "forward_outgoing_convention": forward_outgoing,
                "reverse_outgoing_convention": reverse_outgoing,
                "mass_unit": mass_unit,
                "energy_unit": energy_unit,
                "speed_unit": speed_unit,
                "cross_section_unit": cross_section_unit,
                "momentum_unit": momentum_unit,
                "position_unit": position_unit,
                "reciprocity_angles": array_tree_fingerprint(
                    (
                        reciprocity_forward_cosines,
                        reciprocity_forward_azimuths,
                        reciprocity_reverse_cosines,
                        reciprocity_reverse_azimuths,
                    )
                ),
                "speed_of_light": light_speed,
                "maximum_speed_fraction": maximum_fraction,
                "detailed_balance_tolerance": balance_tolerance,
            }
        )

    @property
    def threshold_energy(self) -> float:
        return max(self.energy_change, 0.0)

    @property
    def released_energy(self) -> float:
        return max(-self.energy_change, 0.0)

    def thermal_equilibrium_ratio(self, thermal_energy: ArrayLike, /) -> Array:
        """Return ``n_out0*n_out1/(n_in0*n_in1)`` for dilute MB species."""

        temperature = jnp.asarray(thermal_energy)
        if temperature.shape != ():
            raise ValueError("thermal_energy must be scalar.")
        temperature = eqx.error_if(
            temperature,
            ~jnp.isfinite(temperature) | (temperature <= 0.0),
            "thermal_energy must be finite and positive.",
        )
        mass_ratio = (
            self.outgoing_species[0].mass
            * self.outgoing_species[1].mass
            / (self.incoming_species[0].mass * self.incoming_species[1].mass)
        )
        phase_ratio = self.outgoing_degeneracy / self.incoming_degeneracy
        return phase_ratio * mass_ratio**1.5 * jnp.exp(-self.energy_change / temperature)


class InelasticSIDMState(StrictModule, NonTrainableState):
    """Weighted packets, numeric species marks, radiation, and accepted epoch."""

    packets: WeightedSIDMPacketState
    species_indices: Array
    radiation: DarkRadiationLedger
    reaction_epoch: Array


class DarkReactionPartialRates(StrictModule, NonTrainableState):
    partial_rates: Array
    matched: Array
    threshold_open: Array
    input_nonrelativistic: Array
    output_nonrelativistic: Array
    kernel_supported: Array
    outgoing_relative_speeds: Array
    input_relative_speed: Array
    center_of_mass_velocity: Array
    pair_valid: Array
    state_valid: Array
    total_rate: Array
    successful: Array


class DarkReactionEvidence(StrictModule, NonTrainableState):
    status: Array
    selected_direction: Array
    selected_channel: Array
    forward: Array
    partial_rates: Array
    channel_probabilities: Array
    total_rate: Array
    threshold_energy: Array
    incoming_relative_kinetic_energy: Array
    outgoing_relative_kinetic_energy: Array
    outgoing_relative_speed: Array
    outgoing_reduced_mass: Array
    angular_cosine: Array
    angular_azimuth: Array
    forward_reciprocity_cosine: Array
    forward_reciprocity_azimuth: Array
    reverse_reciprocity_cosine: Array
    reverse_reciprocity_azimuth: Array
    angular_normalization_residual: Array
    kernel_supported: Array
    threshold_open: Array
    nonrelativistic: Array
    relativistic_refusal: Array
    integrated_balance_forward: Array
    integrated_balance_reverse: Array
    integrated_balance_residual: Array
    integrated_balance_valid: Array
    detailed_balance_forward: Array
    detailed_balance_reverse: Array
    detailed_balance_residual: Array
    detailed_balance_valid: Array
    reacted_weight: Array
    split_required: Array
    child_slot: Array
    capacity_available: Array
    charge_defect: Array
    rest_mass_energy_change: Array
    internal_energy_change: Array
    kinetic_energy_change: Array
    radiation_energy_change: Array
    total_energy_defect: Array
    physical_momentum_defect: Array
    canonical_momentum_defect: Array
    gravitational_mass_change: Array
    packet_microscopic_mass_change: Array
    packet_gravitational_mass_change: Array
    dynamic_pm_mass_required: Array
    pm_mass_consistent: Array
    stable_ids_preserved: Array
    lineage_valid: Array
    conservation_valid: Array
    finite: Array
    state_valid: Array
    discrete_nondifferentiable: Array
    rolled_back: Array
    successful: Array


class DarkReactionResult(StrictModule, NonTrainableState):
    candidate_state: InelasticSIDMState
    accepted_state: InelasticSIDMState
    evidence: DarkReactionEvidence
    successful: Array


class DarkRadiationTransactionEvidence(StrictModule, NonTrainableState):
    export: DarkRadiationExportEvidence
    source_state_valid: Array
    retained_state_valid: Array
    candidate_state_valid: Array
    support_preserved: Array
    time_level_preserved: Array
    source_ledger_preserved: Array
    source_epoch_preserved: Array
    parent_indices_valid: Array
    parents_active: Array
    parent_positions_coincident: Array
    positions_preserved: Array
    charge_preserved: Array
    nonparents_unchanged: Array
    packet_microscopic_mass_change: Array
    packet_gravitational_mass_change: Array
    dynamic_pm_mass_required: Array
    source_particle_four_momentum: Array
    retained_particle_four_momentum: Array
    four_momentum_defect: Array
    conservation_valid: Array
    rolled_back: Array
    successful: Array


class DarkRadiationStateResult(StrictModule, NonTrainableState):
    packet: DarkRadiationPacket
    candidate_state: InelasticSIDMState
    accepted_state: InelasticSIDMState
    evidence: DarkRadiationTransactionEvidence
    successful: Array


class InelasticSIDMPlan(StrictModule, NonTrainableState):
    """Fixed typed tuple of reversible channels over weighted packet state."""

    channels: tuple[DarkTwoBodyReactionPlan, ...]
    radiation: DarkRadiationLedgerPlan
    species: tuple[DarkSectorSpeciesPlan, ...]
    species_plan_ids: tuple[str, ...] = eqx.field(static=True)
    incoming_indices: Array
    outgoing_indices: Array
    forward_directions: Array
    direction_channels: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        channels: tuple[DarkTwoBodyReactionPlan, ...],
        radiation: DarkRadiationLedgerPlan,
        /,
    ):
        if (
            not isinstance(channels, tuple)
            or not channels
            or not all(isinstance(item, DarkTwoBodyReactionPlan) for item in channels)
        ):
            raise TypeError("channels must be a non-empty typed tuple of reaction plans.")
        if not isinstance(radiation, DarkRadiationLedgerPlan):
            raise TypeError("radiation must be a DarkRadiationLedgerPlan.")
        first = channels[0]
        unit_contract = (
            first.mass_unit,
            first.energy_unit,
            first.speed_unit,
            first.cross_section_unit,
            first.momentum_unit,
            first.position_unit,
        )
        if any(
            (
                item.speed_of_light != first.speed_of_light
                or item.maximum_speed_fraction != first.maximum_speed_fraction
                or item.conserved_charge_names != first.conserved_charge_names
                or (
                    item.mass_unit,
                    item.energy_unit,
                    item.speed_unit,
                    item.cross_section_unit,
                    item.momentum_unit,
                    item.position_unit,
                )
                != unit_contract
            )
            for item in channels
        ):
            raise ValueError(
                "All reaction channels must share charge, validity, and unit contracts."
            )
        if (
            radiation.dimension != 3
            or radiation.speed_of_light != first.speed_of_light
            or radiation.energy_unit != first.energy_unit
            or radiation.momentum_unit != first.momentum_unit
            or radiation.position_unit != first.position_unit
        ):
            raise ValueError(
                "Reaction and radiation plans must share dimensions, light speed, and energy/momentum/position units."
            )
        if len({item.channel_id for item in channels}) != len(channels):
            raise ValueError("Reaction channel identities must be unique.")

        unique: list[DarkSectorSpeciesPlan] = []
        index_by_id: dict[str, int] = {}
        for channel in channels:
            for species in (*channel.incoming_species, *channel.outgoing_species):
                if species.species_plan_id not in index_by_id:
                    index_by_id[species.species_plan_id] = len(unique)
                    unique.append(species)
        incoming = []
        outgoing = []
        for channel in channels:
            incoming.append(
                tuple(
                    index_by_id[item.species_plan_id] for item in channel.incoming_species
                )
            )
            outgoing.append(
                tuple(
                    index_by_id[item.species_plan_id] for item in channel.outgoing_species
                )
            )
        direction_incoming = np.asarray((*incoming, *outgoing), dtype=np.int32)
        direction_outgoing = np.asarray((*outgoing, *incoming), dtype=np.int32)
        direction_forward = np.asarray(
            (True,) * len(channels) + (False,) * len(channels), dtype=np.bool_
        )
        direction_channels = np.asarray(
            (*range(len(channels)), *range(len(channels))), dtype=np.int32
        )
        self.channels = channels
        self.radiation = radiation
        self.species = tuple(unique)
        self.species_plan_ids = tuple(item.species_plan_id for item in unique)
        self.incoming_indices = jax.lax.stop_gradient(jnp.asarray(direction_incoming))
        self.outgoing_indices = jax.lax.stop_gradient(jnp.asarray(direction_outgoing))
        self.forward_directions = jax.lax.stop_gradient(jnp.asarray(direction_forward))
        self.direction_channels = jax.lax.stop_gradient(jnp.asarray(direction_channels))
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-channel-inelastic-sidm",
                "channels": [item.channel_id for item in channels],
                "species": list(self.species_plan_ids),
                "radiation": radiation.plan_id,
                "reaction_profile": "nonrelativistic-reversible-2-to-2",
                "radiation_transport_claim": "none",
            }
        )

    def initialize(
        self,
        packets: WeightedSIDMPacketState,
        species_indices: ArrayLike,
        /,
        *,
        reaction_epoch: ArrayLike = 0,
    ) -> InelasticSIDMState:
        if not isinstance(packets, WeightedSIDMPacketState):
            raise TypeError("packets must be WeightedSIDMPacketState.")
        indices = jnp.asarray(species_indices, dtype=jnp.int32)
        if indices.shape != packets.active_mask.shape:
            raise ValueError("species_indices must match weighted packet capacity.")
        epoch = jnp.asarray(reaction_epoch, dtype=jnp.int64)
        if epoch.shape != ():
            raise ValueError("reaction_epoch must be scalar.")
        state = InelasticSIDMState(
            packets,
            indices,
            self.radiation.empty(dtype=packets.positions.dtype),
            epoch,
        )
        valid = self.state_valid(state)
        indices = eqx.error_if(
            indices,
            ~valid,
            "Weighted reaction state violates species, mass, identity, or scale invariants.",
        )
        return InelasticSIDMState(state.packets, indices, state.radiation, epoch)

    def state_valid(self, state: InelasticSIDMState, /) -> Array:
        self._check_state_shapes(state)
        packets = state.packets
        active = packets.active_mask
        safe_species = jnp.clip(state.species_indices, 0, len(self.species) - 1)
        mass_table = jnp.asarray(
            tuple(item.mass for item in self.species), dtype=packets.positions.dtype
        )
        expected_mass = mass_table[safe_species]
        shapes_valid = (
            jnp.all(jnp.isfinite(packets.positions))
            & jnp.all(jnp.isfinite(packets.microscopic_masses))
            & jnp.all(jnp.isfinite(packets.weights))
            & jnp.all(jnp.isfinite(packets.gravitational_masses))
            & jnp.all(jnp.isfinite(packets.canonical_momenta))
            & jnp.isfinite(packets.scale_factor)
            & (packets.scale_factor > 0.0)
            & jnp.isfinite(state.reaction_epoch)
            & (state.reaction_epoch >= 0)
        )
        species_valid = jnp.all(
            jnp.where(
                active,
                (state.species_indices >= 0)
                & (state.species_indices < len(self.species)),
                state.species_indices == -1,
            )
        )
        masses_valid = jnp.all(
            ~active
            | (
                (packets.microscopic_masses == expected_mass)
                & (packets.weights > 0.0)
                & (
                    packets.gravitational_masses
                    == packets.microscopic_masses * packets.weights
                )
                & (packets.gravitational_masses > 0.0)
            )
        )
        identity_valid = _packet_lineage_valid(packets)
        return (
            shapes_valid
            & species_valid
            & masses_valid
            & identity_valid
            & self.radiation.valid(state.radiation)
        )

    def partial_rates(
        self,
        state: InelasticSIDMState,
        left_index: ArrayLike,
        right_index: ArrayLike,
        /,
    ) -> DarkReactionPartialRates:
        self._check_state_shapes(state)
        packets = state.packets
        capacity = packets.active_mask.size
        left_raw = jnp.asarray(left_index, dtype=jnp.int32).reshape(())
        right_raw = jnp.asarray(right_index, dtype=jnp.int32).reshape(())
        indices_valid = (
            (left_raw >= 0)
            & (left_raw < capacity)
            & (right_raw >= 0)
            & (right_raw < capacity)
            & (left_raw != right_raw)
        )
        left = jnp.clip(left_raw, 0, capacity - 1)
        right = jnp.clip(right_raw, 0, capacity - 1)
        active_pair = packets.active_mask[left] & packets.active_mask[right]
        state_valid = self.state_valid(state)
        pair_valid = indices_valid & active_pair & state_valid
        scale = packets.scale_factor
        left_mass = jnp.where(
            packets.gravitational_masses[left] > 0.0,
            packets.gravitational_masses[left],
            1.0,
        )
        right_mass = jnp.where(
            packets.gravitational_masses[right] > 0.0,
            packets.gravitational_masses[right],
            1.0,
        )
        left_velocity = packets.canonical_momenta[left] / (left_mass * scale)
        right_velocity = packets.canonical_momenta[right] / (right_mass * scale)
        relative_velocity = left_velocity - right_velocity
        relative_speed = jnp.sqrt(
            ein.contract("i,i->", relative_velocity, relative_velocity)
        )
        microscopic_left = packets.microscopic_masses[left]
        microscopic_right = packets.microscopic_masses[right]
        center = (
            microscopic_left * left_velocity + microscopic_right * right_velocity
        ) / jnp.where(
            microscopic_left + microscopic_right > 0.0,
            microscopic_left + microscopic_right,
            1.0,
        )
        input_absolute_speed = jnp.maximum(
            jnp.sqrt(ein.contract("i,i->", left_velocity, left_velocity)),
            jnp.sqrt(ein.contract("i,i->", right_velocity, right_velocity)),
        )
        current = jnp.asarray(
            (state.species_indices[left], state.species_indices[right]), dtype=jnp.int32
        )
        direct = jnp.all(self.incoming_indices == current[None, :], axis=-1)
        swapped = jnp.all(self.incoming_indices == current[::-1][None, :], axis=-1)
        matched = direct | swapped

        direction_indices = (*range(len(self.channels)), *range(len(self.channels)))
        moment_values = []
        moment_supported = []
        for direction, channel_index in enumerate(direction_indices):
            channel = self.channels[int(channel_index)]
            kernel = (
                channel.forward_kernel
                if direction < len(self.channels)
                else channel.reverse_kernel
            )
            moments = kernel.moments(relative_speed)
            moment_values.append(moments.total)
            moment_supported.append(moments.supported)
        cross_sections = jnp.stack(moment_values)
        kernel_supported = jnp.stack(moment_supported)

        outgoing_mass = jnp.asarray(
            tuple(
                self.channels[int(index)].outgoing_reduced_mass
                if direction < len(self.channels)
                else self.channels[int(index)].incoming_reduced_mass
                for direction, index in enumerate(direction_indices)
            ),
            dtype=packets.positions.dtype,
        )
        energy_change = jnp.asarray(
            tuple(
                self.channels[int(index)].energy_change
                if direction < len(self.channels)
                else -self.channels[int(index)].energy_change
                for direction, index in enumerate(direction_indices)
            ),
            dtype=packets.positions.dtype,
        )
        outgoing_pair_masses = jnp.asarray(
            tuple(
                (
                    channel.outgoing_species[0].mass,
                    channel.outgoing_species[1].mass,
                )
                if direction < len(self.channels)
                else (
                    channel.incoming_species[0].mass,
                    channel.incoming_species[1].mass,
                )
                for direction, channel in (
                    (direction, self.channels[int(index)])
                    for direction, index in enumerate(direction_indices)
                )
            ),
            dtype=packets.positions.dtype,
        )
        microscopic_momentum = (
            microscopic_left * left_velocity + microscopic_right * right_velocity
        )
        microscopic_momentum_squared = ein.contract(
            "i,i->", microscopic_momentum, microscopic_momentum
        )
        incoming_kinetic = 0.5 * (
            microscopic_left * ein.contract("i,i->", left_velocity, left_velocity)
            + microscopic_right * ein.contract("i,i->", right_velocity, right_velocity)
        )
        outgoing_total_mass = jnp.sum(outgoing_pair_masses, axis=-1)
        outgoing_center_kinetic = 0.5 * microscopic_momentum_squared / outgoing_total_mass
        outgoing_kinetic = incoming_kinetic - outgoing_center_kinetic - energy_change
        threshold_open = outgoing_kinetic >= 0.0
        safe_outgoing_kinetic = jnp.maximum(outgoing_kinetic, 0.0)
        outgoing_speed = jnp.sqrt(2.0 * safe_outgoing_kinetic / outgoing_mass)
        light_speed = self.channels[0].speed_of_light
        speed_limit = self.channels[0].maximum_speed_fraction * light_speed
        input_nonrelativistic = input_absolute_speed < speed_limit
        outgoing_fraction = jnp.max(
            outgoing_pair_masses[:, ::-1] / outgoing_total_mass[:, None],
            axis=-1,
        )
        outgoing_center_speed = (
            jnp.sqrt(microscopic_momentum_squared) / outgoing_total_mass
        )
        output_nonrelativistic = (
            outgoing_center_speed + outgoing_fraction * outgoing_speed
        ) < speed_limit
        admissible = (
            pair_valid
            & matched
            & threshold_open
            & input_nonrelativistic
            & output_nonrelativistic
            & kernel_supported
            & jnp.isfinite(cross_sections)
            & (cross_sections >= 0.0)
        )
        partial = jnp.where(admissible, cross_sections * relative_speed, 0.0)
        total = jnp.sum(partial)
        successful = pair_valid & jnp.isfinite(total) & (total > 0.0)
        return DarkReactionPartialRates(
            partial,
            matched,
            threshold_open,
            jnp.broadcast_to(input_nonrelativistic, partial.shape),
            output_nonrelativistic,
            kernel_supported,
            outgoing_speed,
            relative_speed,
            center,
            pair_valid,
            state_valid,
            total,
            successful,
        )

    def react(
        self,
        state: InelasticSIDMState,
        left_index: ArrayLike,
        right_index: ArrayLike,
        key: Array,
        /,
    ) -> DarkReactionResult:
        """Select one open partial rate and commit the complete packet transaction."""

        rates = self.partial_rates(state, left_index, right_index)
        direction_indices = (*range(len(self.channels)), *range(len(self.channels)))
        packets = state.packets
        capacity = packets.active_mask.size
        left = jnp.clip(
            jnp.asarray(left_index, dtype=jnp.int32).reshape(()), 0, capacity - 1
        )
        right = jnp.clip(
            jnp.asarray(right_index, dtype=jnp.int32).reshape(()), 0, capacity - 1
        )
        selection_key, angular_key = jr.split(key)
        logits = jnp.where(
            rates.partial_rates > 0.0, jnp.log(rates.partial_rates), -jnp.inf
        )
        selected = jax.lax.stop_gradient(
            jr.categorical(selection_key, logits).astype(jnp.int32)
        )
        selected_channel = self.direction_channels[selected]
        forward = self.forward_directions[selected]
        probability = jnp.where(
            rates.total_rate > 0.0, rates.partial_rates / rates.total_rate, 0.0
        )

        keys = jr.split(angular_key, self.incoming_indices.shape[0])
        angular_samples = []
        angular_tolerances = []
        for direction, channel_index in enumerate(direction_indices):
            channel = self.channels[int(channel_index)]
            kernel = (
                channel.forward_kernel
                if direction < len(self.channels)
                else channel.reverse_kernel
            )
            angular_samples.append(
                kernel.sample_angles(keys[direction], rates.input_relative_speed)
            )
            angular_tolerances.append(kernel.normalization_tolerance)
        cosines = jnp.stack(tuple(item.cosine for item in angular_samples))
        azimuths = jnp.stack(tuple(item.azimuth for item in angular_samples))
        angular_supported = jnp.stack(tuple(item.supported for item in angular_samples))
        angular_residuals = jnp.stack(
            tuple(item.normalization_residual for item in angular_samples)
        )
        angular_tolerance = jnp.asarray(angular_tolerances, dtype=packets.positions.dtype)
        angular_valid = angular_supported[selected] & (
            angular_residuals[selected] <= angular_tolerance[selected]
        )
        cosine = cosines[selected]
        azimuth = azimuths[selected]

        current = jnp.asarray(
            (state.species_indices[left], state.species_indices[right]), dtype=jnp.int32
        )
        selected_incoming = self.incoming_indices[selected]
        direct = jnp.all(selected_incoming == current)
        selected_outgoing = self.outgoing_indices[selected]
        left_species = jnp.where(direct, selected_outgoing[0], selected_outgoing[1])
        right_species = jnp.where(direct, selected_outgoing[1], selected_outgoing[0])
        mass_table = jnp.asarray(
            tuple(item.mass for item in self.species), dtype=packets.positions.dtype
        )
        left_output_mass = mass_table[left_species]
        right_output_mass = mass_table[right_species]

        left_weight = packets.weights[left]
        right_weight = packets.weights[right]
        reacted_weight = jnp.minimum(left_weight, right_weight)
        left_excess = left_weight - reacted_weight
        right_excess = right_weight - reacted_weight
        split_required = (left_excess > 0.0) | (right_excess > 0.0)
        free = ~packets.active_mask
        free_available = jnp.any(free)
        child = jnp.argmax(free.astype(jnp.int32))
        capacity_available = ~split_required | free_available
        child_source = jnp.where(left_excess > 0.0, left, right)
        child_excess = jnp.maximum(left_excess, right_excess)

        scale = packets.scale_factor
        safe_left_macro = jnp.where(
            packets.gravitational_masses[left] > 0.0,
            packets.gravitational_masses[left],
            1.0,
        )
        safe_right_macro = jnp.where(
            packets.gravitational_masses[right] > 0.0,
            packets.gravitational_masses[right],
            1.0,
        )
        left_velocity = packets.canonical_momenta[left] / (safe_left_macro * scale)
        right_velocity = packets.canonical_momenta[right] / (safe_right_macro * scale)
        relative = left_velocity - right_velocity
        incoming_axis = directions_from_angles(
            relative,
            jnp.asarray(1.0, dtype=packets.positions.dtype),
            jnp.asarray(0.0, dtype=packets.positions.dtype),
        )
        outgoing_axis = directions_from_angles(relative, cosine, azimuth)
        mapped_cosine, mapped_azimuth = angles_from_direction(
            -outgoing_axis, -incoming_axis
        )
        forward_cosine = jnp.where(forward, cosine, mapped_cosine)
        forward_azimuth = jnp.where(forward, azimuth, mapped_azimuth)
        reverse_cosine = jnp.where(forward, mapped_cosine, cosine)
        reverse_azimuth = jnp.where(forward, mapped_azimuth, azimuth)
        outgoing_speed = rates.outgoing_relative_speeds[selected]
        left_input_mass = packets.microscopic_masses[left]
        right_input_mass = packets.microscopic_masses[right]
        microscopic_momentum = (
            left_input_mass * left_velocity + right_input_mass * right_velocity
        )
        outgoing_total_mass = left_output_mass + right_output_mass
        center = microscopic_momentum / outgoing_total_mass
        left_output_velocity = (
            center
            + (right_output_mass / outgoing_total_mass) * outgoing_speed * outgoing_axis
        )
        right_output_velocity = (
            center
            - (left_output_mass / outgoing_total_mass) * outgoing_speed * outgoing_axis
        )
        left_macro_mass = left_output_mass * reacted_weight
        right_macro_mass = right_output_mass * reacted_weight
        left_momentum = left_macro_mass * scale * left_output_velocity
        right_momentum = right_macro_mass * scale * right_output_velocity
        child_fraction = child_excess / jnp.where(
            packets.weights[child_source] > 0.0,
            packets.weights[child_source],
            1.0,
        )
        child_momentum = packets.canonical_momenta[child_source] * child_fraction
        spawn_child = split_required & free_available

        positions = packets.positions
        microscopic = packets.microscopic_masses.at[left].set(left_output_mass)
        microscopic = microscopic.at[right].set(right_output_mass)
        weights = packets.weights.at[left].set(reacted_weight)
        weights = weights.at[right].set(reacted_weight)
        gravitational = packets.gravitational_masses.at[left].set(left_macro_mass)
        gravitational = gravitational.at[right].set(right_macro_mass)
        momenta = packets.canonical_momenta.at[left].set(left_momentum)
        momenta = momenta.at[right].set(right_momentum)
        active = packets.active_mask
        parent_ids = packets.parent_packet_ids
        lineage = packets.lineage_depth
        species_indices = state.species_indices.at[left].set(left_species)
        species_indices = species_indices.at[right].set(right_species)
        positions = positions.at[child].set(
            jnp.where(spawn_child, packets.positions[child_source], positions[child])
        )
        microscopic = microscopic.at[child].set(
            jnp.where(
                spawn_child,
                packets.microscopic_masses[child_source],
                microscopic[child],
            )
        )
        weights = weights.at[child].set(
            jnp.where(spawn_child, child_excess, weights[child])
        )
        gravitational = gravitational.at[child].set(
            jnp.where(
                spawn_child,
                packets.microscopic_masses[child_source] * child_excess,
                gravitational[child],
            )
        )
        momenta = momenta.at[child].set(
            jnp.where(spawn_child, child_momentum, momenta[child])
        )
        active = active.at[child].set(active[child] | spawn_child)
        parent_ids = parent_ids.at[child].set(
            jnp.where(spawn_child, packets.packet_ids[child_source], parent_ids[child])
        )
        lineage = lineage.at[child].set(
            jnp.where(
                spawn_child, packets.lineage_depth[child_source] + 1, lineage[child]
            )
        )
        species_indices = species_indices.at[child].set(
            jnp.where(
                spawn_child, state.species_indices[child_source], species_indices[child]
            )
        )
        candidate_packets = WeightedSIDMPacketState(
            positions,
            microscopic,
            weights,
            gravitational,
            momenta,
            active,
            packets.packet_ids,
            parent_ids,
            lineage,
            packets.scale_factor,
        )
        candidate = InelasticSIDMState(
            candidate_packets,
            species_indices,
            state.radiation,
            state.reaction_epoch + 1,
        )

        changed_mask = (
            (jnp.arange(capacity) == left)
            | (jnp.arange(capacity) == right)
            | (spawn_child & (jnp.arange(capacity) == child))
        )
        before = self._totals(state, changed_mask)
        after = self._totals(candidate, changed_mask)
        charge_defect = after[0] - before[0]
        rest_change = after[1] - before[1]
        internal_change = after[2] - before[2]
        kinetic_change = after[3] - before[3]
        radiation_change = after[4] - before[4]
        total_energy_defect = (
            rest_change + internal_change + kinetic_change + radiation_change
        )
        physical_momentum_defect = (after[5] + after[6]) - (before[5] + before[6])
        canonical_momentum_defect = after[7] - before[7]
        gravitational_mass_change = after[8] - before[8]
        packet_microscopic_mass_change = (
            candidate_packets.microscopic_masses - packets.microscopic_masses
        )
        packet_mass_change = (
            candidate_packets.gravitational_masses - packets.gravitational_masses
        )
        mass_relation = jnp.all(
            ~candidate_packets.active_mask
            | (
                candidate_packets.gravitational_masses
                == candidate_packets.microscopic_masses * candidate_packets.weights
            )
        )
        stable_ids = jnp.array_equal(candidate_packets.packet_ids, packets.packet_ids)
        lineage_valid = _packet_lineage_valid(candidate_packets) & jnp.where(
            spawn_child,
            candidate_packets.parent_packet_ids[child]
            == packets.packet_ids[child_source],
            True,
        )

        dtype = packets.positions.dtype
        epsilon = jnp.finfo(dtype).eps
        tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
        packet_mass_scale = jnp.maximum(
            jnp.maximum(
                jnp.abs(candidate_packets.gravitational_masses),
                jnp.abs(packets.gravitational_masses),
            ),
            tiny,
        )
        dynamic_pm_mass_required = jnp.any(
            jnp.abs(packet_mass_change) > 64.0 * epsilon * packet_mass_scale
        ) | jnp.any(candidate_packets.active_mask != packets.active_mask)
        energy_scale = jnp.max(
            jnp.concatenate(
                (
                    jnp.abs(jnp.asarray(before[1:5], dtype=dtype)),
                    jnp.abs(jnp.asarray(after[1:5], dtype=dtype)),
                    jnp.reshape(tiny, (1,)),
                )
            )
        )
        before_physical_scale = jnp.sum(
            jnp.abs(
                jnp.where(
                    (packets.active_mask & changed_mask)[:, None],
                    packets.canonical_momenta / packets.scale_factor,
                    0.0,
                )
            )
        )
        after_physical_scale = jnp.sum(
            jnp.abs(
                jnp.where(
                    (candidate_packets.active_mask & changed_mask)[:, None],
                    candidate_packets.canonical_momenta / candidate_packets.scale_factor,
                    0.0,
                )
            )
        )
        momentum_scale = jnp.maximum(
            jnp.maximum(before_physical_scale, after_physical_scale), tiny
        )
        canonical_scale = jnp.maximum(
            jnp.maximum(
                jnp.sum(
                    jnp.abs(
                        jnp.where(
                            (packets.active_mask & changed_mask)[:, None],
                            packets.canonical_momenta,
                            0.0,
                        )
                    )
                ),
                jnp.sum(
                    jnp.abs(
                        jnp.where(
                            (candidate_packets.active_mask & changed_mask)[:, None],
                            candidate_packets.canonical_momenta,
                            0.0,
                        )
                    )
                ),
            ),
            tiny,
        )
        charge_values = jnp.concatenate(
            (
                jnp.ravel(jnp.abs(before[0])),
                jnp.ravel(jnp.abs(after[0])),
                jnp.reshape(tiny, (1,)),
            )
        )
        charge_scale = jnp.max(charge_values)
        conservation_valid = (
            jnp.all(jnp.abs(charge_defect) <= 256.0 * epsilon * charge_scale)
            & (jnp.abs(total_energy_defect) <= 512.0 * epsilon * energy_scale)
            & jnp.all(
                jnp.abs(physical_momentum_defect) <= 512.0 * epsilon * momentum_scale
            )
            & jnp.all(
                jnp.abs(canonical_momentum_defect) <= 512.0 * epsilon * canonical_scale
            )
        )

        forward_speed = jnp.where(forward, rates.input_relative_speed, outgoing_speed)
        reverse_speed = jnp.where(forward, outgoing_speed, rates.input_relative_speed)
        integrated_forward_values = []
        integrated_reverse_values = []
        integrated_valid_values = []
        differential_forward_values = []
        differential_reverse_values = []
        differential_residual_values = []
        differential_valid_values = []
        for channel in self.channels:
            forward_moment = channel.forward_kernel.moments(forward_speed)
            reverse_moment = channel.reverse_kernel.moments(reverse_speed)
            forward_differential = channel.forward_kernel.evaluate(
                forward_speed,
                channel.reciprocity_forward_cosines,
                channel.reciprocity_forward_azimuths,
            )
            reverse_differential = channel.reverse_kernel.evaluate(
                reverse_speed,
                channel.reciprocity_reverse_cosines,
                channel.reciprocity_reverse_azimuths,
            )
            incoming_momentum = channel.incoming_reduced_mass * forward_speed
            outgoing_momentum = channel.outgoing_reduced_mass * reverse_speed
            forward_phase = (
                channel.incoming_degeneracy
                / channel.incoming_pair_symmetry
                * incoming_momentum
                * incoming_momentum
            )
            reverse_phase = (
                channel.outgoing_degeneracy
                / channel.outgoing_pair_symmetry
                * outgoing_momentum
                * outgoing_momentum
            )
            integrated_forward = forward_phase * forward_moment.total
            integrated_reverse = reverse_phase * reverse_moment.total
            integrated_residual = jnp.abs(
                integrated_forward - integrated_reverse
            ) / jnp.maximum(
                jnp.maximum(jnp.abs(integrated_forward), jnp.abs(integrated_reverse)),
                jnp.finfo(dtype).tiny,
            )
            differential_forward = (
                forward_phase * forward_differential.differential_cross_section
            )
            differential_reverse = (
                reverse_phase * reverse_differential.differential_cross_section
            )
            differential_residual = jnp.abs(
                differential_forward - differential_reverse
            ) / jnp.maximum(
                jnp.maximum(jnp.abs(differential_forward), jnp.abs(differential_reverse)),
                jnp.finfo(dtype).tiny,
            )
            integrated_forward_values.append(integrated_forward)
            integrated_reverse_values.append(integrated_reverse)
            integrated_valid_values.append(
                forward_moment.supported
                & reverse_moment.supported
                & (integrated_residual <= channel.detailed_balance_tolerance)
            )
            differential_forward_values.append(jnp.max(differential_forward))
            differential_reverse_values.append(jnp.max(differential_reverse))
            differential_residual_values.append(jnp.max(differential_residual))
            differential_valid_values.append(
                jnp.all(forward_differential.supported)
                & jnp.all(reverse_differential.supported)
                & jnp.all(differential_residual <= channel.detailed_balance_tolerance)
            )
        integrated_balance_forward = jnp.stack(integrated_forward_values)[
            selected_channel
        ]
        integrated_balance_reverse = jnp.stack(integrated_reverse_values)[
            selected_channel
        ]
        integrated_balance_residual = jnp.abs(
            integrated_balance_forward - integrated_balance_reverse
        ) / jnp.maximum(
            jnp.maximum(
                jnp.abs(integrated_balance_forward),
                jnp.abs(integrated_balance_reverse),
            ),
            jnp.finfo(dtype).tiny,
        )
        integrated_balance_valid = jnp.stack(integrated_valid_values)[selected_channel]
        balance_forward = jnp.stack(differential_forward_values)[selected_channel]
        balance_reverse = jnp.stack(differential_reverse_values)[selected_channel]
        balance_residual = jnp.stack(differential_residual_values)[selected_channel]
        differential_balance_valid = jnp.stack(differential_valid_values)[
            selected_channel
        ]
        detailed_balance_valid = integrated_balance_valid & differential_balance_valid

        finite = jnp.all(
            jnp.stack(
                (
                    jnp.all(jnp.isfinite(candidate_packets.positions)),
                    jnp.all(jnp.isfinite(candidate_packets.microscopic_masses)),
                    jnp.all(jnp.isfinite(candidate_packets.weights)),
                    jnp.all(jnp.isfinite(candidate_packets.gravitational_masses)),
                    jnp.all(jnp.isfinite(candidate_packets.canonical_momenta)),
                    jnp.isfinite(total_energy_defect),
                    jnp.all(jnp.isfinite(physical_momentum_defect)),
                )
            )
        )
        candidate_state_valid = self.state_valid(candidate)
        selected_threshold = rates.threshold_open[selected]
        selected_nonrelativistic = (
            rates.input_nonrelativistic[selected] & rates.output_nonrelativistic[selected]
        )
        selected_kernel_supported = rates.kernel_supported[selected] & angular_valid
        successful = (
            rates.successful
            & selected_threshold
            & selected_nonrelativistic
            & selected_kernel_supported
            & capacity_available
            & detailed_balance_valid
            & mass_relation
            & stable_ids
            & lineage_valid
            & conservation_valid
            & finite
            & candidate_state_valid
        )
        accepted = _select_state(successful, candidate, state)
        rolled_back = ~successful & _state_equal(accepted, state)

        any_matched = jnp.any(rates.matched)
        any_threshold = jnp.any(rates.matched & rates.threshold_open)
        any_supported = jnp.any(
            rates.matched & rates.threshold_open & rates.kernel_supported
        )
        any_nonrelativistic = jnp.any(
            rates.matched
            & rates.threshold_open
            & rates.kernel_supported
            & rates.input_nonrelativistic
            & rates.output_nonrelativistic
        )
        status = jnp.asarray(int(DarkReactionStatus.SUCCESS), dtype=jnp.int32)
        status = jnp.where(
            ~conservation_valid,
            int(DarkReactionStatus.CONSERVATION_FAILURE),
            status,
        )
        status = jnp.where(
            ~detailed_balance_valid,
            int(DarkReactionStatus.DETAILED_BALANCE_FAILURE),
            status,
        )
        status = jnp.where(
            ~capacity_available,
            int(DarkReactionStatus.CAPACITY_EXHAUSTED),
            status,
        )
        status = jnp.where(
            any_supported & ~any_nonrelativistic,
            int(DarkReactionStatus.RELATIVISTIC_REFUSED),
            status,
        )
        status = jnp.where(
            any_threshold & ~any_supported,
            int(DarkReactionStatus.KERNEL_UNSUPPORTED),
            status,
        )
        status = jnp.where(
            any_matched & ~any_threshold,
            int(DarkReactionStatus.THRESHOLD_CLOSED),
            status,
        )
        status = jnp.where(
            ~rates.pair_valid | ~any_matched,
            int(DarkReactionStatus.INVALID_PAIR),
            status,
        )
        status = jnp.where(
            ~rates.state_valid | ~candidate_state_valid | ~finite,
            int(DarkReactionStatus.INVALID_STATE),
            status,
        )
        threshold = jnp.asarray(
            tuple(
                max(
                    self.channels[int(index)].energy_change
                    if direction < len(self.channels)
                    else -self.channels[int(index)].energy_change,
                    0.0,
                )
                for direction, index in enumerate(direction_indices)
            ),
            dtype=dtype,
        )[selected]
        incoming_mu = jnp.asarray(
            tuple(
                self.channels[int(index)].incoming_reduced_mass
                if direction < len(self.channels)
                else self.channels[int(index)].outgoing_reduced_mass
                for direction, index in enumerate(direction_indices)
            ),
            dtype=dtype,
        )[selected]
        outgoing_mu = jnp.asarray(
            tuple(
                self.channels[int(index)].outgoing_reduced_mass
                if direction < len(self.channels)
                else self.channels[int(index)].incoming_reduced_mass
                for direction, index in enumerate(direction_indices)
            ),
            dtype=dtype,
        )[selected]
        incoming_kinetic = 0.5 * incoming_mu * rates.input_relative_speed**2
        outgoing_kinetic = 0.5 * outgoing_mu * outgoing_speed**2
        evidence = DarkReactionEvidence(
            status,
            selected,
            selected_channel,
            forward,
            rates.partial_rates,
            probability,
            rates.total_rate,
            threshold,
            incoming_kinetic,
            outgoing_kinetic,
            outgoing_speed,
            outgoing_mu,
            cosine,
            azimuth,
            forward_cosine,
            forward_azimuth,
            reverse_cosine,
            reverse_azimuth,
            angular_residuals[selected],
            selected_kernel_supported,
            selected_threshold,
            selected_nonrelativistic,
            any_supported & ~any_nonrelativistic,
            integrated_balance_forward,
            integrated_balance_reverse,
            integrated_balance_residual,
            integrated_balance_valid,
            balance_forward,
            balance_reverse,
            balance_residual,
            detailed_balance_valid,
            reacted_weight,
            split_required,
            jnp.where(spawn_child, child, -1).astype(jnp.int32),
            capacity_available,
            charge_defect,
            rest_change,
            internal_change,
            kinetic_change,
            radiation_change,
            total_energy_defect,
            physical_momentum_defect,
            canonical_momentum_defect,
            gravitational_mass_change,
            packet_microscopic_mass_change,
            packet_mass_change,
            dynamic_pm_mass_required,
            mass_relation,
            stable_ids,
            lineage_valid,
            conservation_valid,
            finite,
            rates.state_valid & candidate_state_valid,
            jnp.asarray(True),
            rolled_back,
            successful,
        )
        return DarkReactionResult(candidate, accepted, evidence, successful)

    def export_radiation(
        self,
        state: InelasticSIDMState,
        retained_state: InelasticSIDMState,
        packet: DarkRadiationPacket,
        parent_indices: ArrayLike,
        /,
    ) -> DarkRadiationStateResult:
        """Commit a local neutral-radiation emission as one atomic transaction.

        Parent identities, coincident event position, emission scale factor, and
        both particle four-momenta are derived from the source/retained states.
        Parent positions cannot move, nonparents cannot change, and dark charge
        must remain in the particles because this first radiation profile is
        explicitly neutral. Any failure rolls back every state and ledger leaf.
        """

        self._check_state_shapes(state)
        self._check_state_shapes(retained_state)
        source_valid = self.state_valid(state)
        retained_input_valid = self.state_valid(retained_state)
        capacity = state.packets.active_mask.size
        parents_raw = jnp.asarray(parent_indices, dtype=jnp.int32)
        if parents_raw.shape != (2,):
            raise ValueError("parent_indices must have shape (2,).")
        parent_indices_valid = (
            jnp.all(parents_raw >= 0)
            & jnp.all(parents_raw < capacity)
            & (parents_raw[0] != parents_raw[1])
        )
        parents = jnp.clip(parents_raw, 0, capacity - 1)
        parent_mask = (jnp.arange(capacity) == parents[0]) | (
            jnp.arange(capacity) == parents[1]
        )
        parents_active = jnp.all(state.packets.active_mask[parents])
        parent_positions_coincident = jnp.array_equal(
            state.packets.positions[parents[0]],
            state.packets.positions[parents[1]],
        )
        positions_preserved = jnp.array_equal(
            retained_state.packets.positions, state.packets.positions
        )
        support_preserved = jnp.array_equal(
            retained_state.packets.packet_ids, state.packets.packet_ids
        )
        time_level_preserved = (
            retained_state.packets.scale_factor == state.packets.scale_factor
        )
        source_ledger_preserved = _tree_equal_arrays(
            retained_state.radiation, state.radiation
        )
        source_epoch_preserved = retained_state.reaction_epoch == state.reaction_epoch
        nonparents_unchanged = (
            jnp.all(
                parent_mask
                | (
                    retained_state.packets.microscopic_masses
                    == state.packets.microscopic_masses
                )
            )
            & jnp.all(
                parent_mask | (retained_state.packets.weights == state.packets.weights)
            )
            & jnp.all(
                parent_mask
                | (
                    retained_state.packets.gravitational_masses
                    == state.packets.gravitational_masses
                )
            )
            & jnp.all(
                parent_mask[:, None]
                | (
                    retained_state.packets.canonical_momenta
                    == state.packets.canonical_momenta
                )
            )
            & jnp.all(
                parent_mask
                | (retained_state.packets.active_mask == state.packets.active_mask)
            )
            & jnp.all(
                parent_mask
                | (
                    retained_state.packets.parent_packet_ids
                    == state.packets.parent_packet_ids
                )
            )
            & jnp.all(
                parent_mask
                | (retained_state.packets.lineage_depth == state.packets.lineage_depth)
            )
            & jnp.all(
                parent_mask | (retained_state.species_indices == state.species_indices)
            )
        )
        retained = InelasticSIDMState(
            retained_state.packets,
            retained_state.species_indices,
            state.radiation,
            state.reaction_epoch,
        )
        retained_valid = self.state_valid(retained)
        source_totals = self._totals(state, parent_mask)
        retained_totals = self._totals(retained, parent_mask)
        dtype = state.packets.positions.dtype
        charge_scale = jnp.maximum(
            jnp.maximum(
                jnp.max(jnp.abs(source_totals[0]), initial=0.0),
                jnp.max(jnp.abs(retained_totals[0]), initial=0.0),
            ),
            jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype),
        )
        charge_preserved = jnp.all(
            jnp.abs(retained_totals[0] - source_totals[0])
            <= 256.0 * jnp.finfo(dtype).eps * charge_scale
        )
        source_four_momentum = jnp.concatenate(
            (
                jnp.reshape(
                    source_totals[1] + source_totals[2] + source_totals[3],
                    (1,),
                ),
                source_totals[5],
            )
        )
        retained_four_momentum = jnp.concatenate(
            (
                jnp.reshape(
                    retained_totals[1] + retained_totals[2] + retained_totals[3],
                    (1,),
                ),
                retained_totals[5],
            )
        )
        bound_packet = DarkRadiationPacket(
            packet.packet_id,
            packet.species_id,
            packet.source_event_id,
            state.packets.packet_ids[parents],
            packet.physical_energy,
            packet.physical_momentum,
            state.packets.positions[parents[0]],
            state.packets.scale_factor,
        )
        exported = self.radiation.export(
            state.radiation,
            bound_packet,
            source_four_momentum,
            retained_four_momentum,
        )
        packet_microscopic_mass_change = (
            retained.packets.microscopic_masses - state.packets.microscopic_masses
        )
        packet_mass_change = (
            retained.packets.gravitational_masses - state.packets.gravitational_masses
        )
        mass_scale = jnp.maximum(
            jnp.maximum(
                jnp.abs(retained.packets.gravitational_masses),
                jnp.abs(state.packets.gravitational_masses),
            ),
            jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype),
        )
        dynamic_pm_mass_required = jnp.any(
            jnp.abs(packet_mass_change) > 64.0 * jnp.finfo(dtype).eps * mass_scale
        ) | jnp.any(retained.packets.active_mask != state.packets.active_mask)
        contract_valid = (
            source_valid
            & retained_input_valid
            & retained_valid
            & support_preserved
            & time_level_preserved
            & source_ledger_preserved
            & source_epoch_preserved
            & parent_indices_valid
            & parents_active
            & parent_positions_coincident
            & positions_preserved
            & charge_preserved
            & nonparents_unchanged
        )
        candidate = InelasticSIDMState(
            retained.packets,
            retained.species_indices,
            exported.candidate_ledger,
            state.reaction_epoch + 1,
        )
        candidate_valid = self.state_valid(candidate)
        successful = contract_valid & exported.successful & candidate_valid
        accepted = _select_state(successful, candidate, state)
        rolled_back = ~successful & _state_equal(accepted, state)
        evidence = DarkRadiationTransactionEvidence(
            exported.evidence,
            source_valid,
            retained_valid & retained_input_valid,
            candidate_valid,
            support_preserved,
            time_level_preserved,
            source_ledger_preserved,
            source_epoch_preserved,
            parent_indices_valid,
            parents_active,
            parent_positions_coincident,
            positions_preserved,
            charge_preserved,
            nonparents_unchanged,
            packet_microscopic_mass_change,
            packet_mass_change,
            dynamic_pm_mass_required,
            source_four_momentum,
            retained_four_momentum,
            exported.evidence.four_momentum_defect,
            exported.evidence.four_momentum_balanced,
            rolled_back,
            successful,
        )
        return DarkRadiationStateResult(
            bound_packet, candidate, accepted, evidence, successful
        )

    def _totals(self, state: InelasticSIDMState, packet_mask: Array | None = None, /):
        packets = state.packets
        active = packets.active_mask
        if packet_mask is not None:
            if packet_mask.shape != active.shape:
                raise ValueError("packet_mask must match packet capacity.")
            active = active & packet_mask
        safe_species = jnp.clip(state.species_indices, 0, len(self.species) - 1)
        dtype = packets.positions.dtype
        masses = jnp.asarray(tuple(item.mass for item in self.species), dtype=dtype)
        internal = jnp.asarray(
            tuple(item.internal_energy for item in self.species), dtype=dtype
        )
        charge_names = self.channels[0].conserved_charge_names
        if charge_names:
            charges = jnp.stack(
                tuple(
                    jnp.stack(tuple(item.charge(name) for name in charge_names))
                    for item in self.species
                )
            ).astype(dtype)
        else:
            charges = jnp.zeros((len(self.species), 0), dtype=dtype)
        weights = jnp.where(active, packets.weights, 0.0)
        charge_total = jnp.sum(weights[:, None] * charges[safe_species], axis=0)
        rest_energy = jnp.sum(
            weights
            * masses[safe_species]
            * self.channels[0].speed_of_light
            * self.channels[0].speed_of_light
        )
        internal_energy = jnp.sum(weights * internal[safe_species])
        safe_macro = jnp.where(
            packets.gravitational_masses > 0.0,
            packets.gravitational_masses,
            1.0,
        )
        velocity = packets.canonical_momenta / (
            safe_macro[:, None] * packets.scale_factor
        )
        kinetic = jnp.sum(
            jnp.where(
                active,
                0.5
                * packets.gravitational_masses
                * ein.contract("...i,...i->...", velocity, velocity),
                0.0,
            )
        )
        radiation_active = (
            state.radiation.active_mask
            if packet_mask is None
            else jnp.zeros_like(state.radiation.active_mask)
        )
        radiation_energy = jnp.sum(
            jnp.where(radiation_active, state.radiation.physical_energy, 0.0)
        )
        particle_physical_momentum = jnp.sum(
            jnp.where(
                active[:, None],
                packets.canonical_momenta / packets.scale_factor,
                0.0,
            ),
            axis=0,
        )
        radiation_momentum = jnp.sum(
            jnp.where(
                radiation_active[:, None],
                state.radiation.physical_momentum,
                0.0,
            ),
            axis=0,
        )
        canonical_momentum = jnp.sum(
            jnp.where(active[:, None], packets.canonical_momenta, 0.0), axis=0
        )
        gravitational_mass = jnp.sum(jnp.where(active, packets.gravitational_masses, 0.0))
        return (
            charge_total,
            rest_energy,
            internal_energy,
            kinetic,
            radiation_energy,
            particle_physical_momentum,
            radiation_momentum,
            canonical_momentum,
            gravitational_mass,
        )

    def _check_state_shapes(self, state: InelasticSIDMState, /) -> None:
        if not isinstance(state, InelasticSIDMState):
            raise TypeError("state must be InelasticSIDMState.")
        if not isinstance(state.packets, WeightedSIDMPacketState):
            raise TypeError("state packets must be WeightedSIDMPacketState.")
        packets = state.packets
        if packets.positions.ndim != 2 or packets.active_mask.ndim != 1:
            raise ValueError("Inelastic SIDM positions/mask must be fixed-rank arrays.")
        capacity = packets.active_mask.size
        dimension = packets.positions.shape[1]
        if capacity < 2 or dimension != 3:
            raise ValueError(
                "Inelastic SIDM requires at least two packets in three dimensions."
            )
        if (
            packets.positions.shape != (capacity, dimension)
            or packets.microscopic_masses.shape != (capacity,)
            or packets.weights.shape != (capacity,)
            or packets.gravitational_masses.shape != (capacity,)
            or packets.canonical_momenta.shape != (capacity, dimension)
            or packets.packet_ids.shape != (capacity,)
            or packets.parent_packet_ids.shape != (capacity,)
            or packets.lineage_depth.shape != (capacity,)
            or state.species_indices.shape != (capacity,)
            or packets.scale_factor.shape != ()
            or state.reaction_epoch.shape != ()
        ):
            raise ValueError("Inelastic SIDM packet state has incompatible fixed shapes.")
        self.radiation._check_ledger(state.radiation)


def _select_packets(
    predicate: Array,
    candidate: WeightedSIDMPacketState,
    fallback: WeightedSIDMPacketState,
    /,
) -> WeightedSIDMPacketState:
    return WeightedSIDMPacketState(
        *(
            jnp.where(predicate, proposed, previous)
            for proposed, previous in zip(
                jax.tree_util.tree_leaves(candidate),
                jax.tree_util.tree_leaves(fallback),
                strict=True,
            )
        )
    )


def _select_state(
    predicate: Array,
    candidate: InelasticSIDMState,
    fallback: InelasticSIDMState,
    /,
) -> InelasticSIDMState:
    return InelasticSIDMState(
        _select_packets(predicate, candidate.packets, fallback.packets),
        jnp.where(predicate, candidate.species_indices, fallback.species_indices),
        type(candidate.radiation)(
            *(
                jnp.where(predicate, proposed, previous)
                for proposed, previous in zip(
                    jax.tree_util.tree_leaves(candidate.radiation),
                    jax.tree_util.tree_leaves(fallback.radiation),
                    strict=True,
                )
            )
        ),
        jnp.where(predicate, candidate.reaction_epoch, fallback.reaction_epoch),
    )


def _tree_equal_arrays(left, right, /) -> Array:
    comparisons = tuple(
        jnp.array_equal(first, second)
        for first, second in zip(
            jax.tree_util.tree_leaves(left),
            jax.tree_util.tree_leaves(right),
            strict=True,
        )
    )
    return jnp.all(jnp.stack(comparisons))


def _state_equal(left: InelasticSIDMState, right: InelasticSIDMState, /) -> Array:
    return _tree_equal_arrays(left, right)


def write_inelastic_sidm_checkpoint(
    path: str | Path,
    plan: InelasticSIDMPlan,
    state: InelasticSIDMState,
    /,
) -> Path:
    """Write one accepted inelastic state with exact runtime identity binding."""

    if not isinstance(plan, InelasticSIDMPlan):
        raise TypeError("plan must be InelasticSIDMPlan.")
    plan._check_state_shapes(state)
    if not bool(np.asarray(plan.state_valid(state))):
        raise ValueError(
            "Only a valid accepted inelastic SIDM state can be checkpointed."
        )
    arrays: dict[str, object] = {}
    specification = pack_array_tree("state", state, arrays)
    payload_id = canonical_fingerprint(
        {
            "plan_id": plan.plan_id,
            "state": specification,
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    return write_array_archive(
        path,
        manifest={
            "format": _CHECKPOINT_FORMAT,
            "plan_id": plan.plan_id,
            "state": specification,
            "payload_id": payload_id,
        },
        arrays=arrays,
    )


def read_inelastic_sidm_checkpoint(
    path: str | Path,
    plan: InelasticSIDMPlan,
    template: InelasticSIDMState,
    /,
) -> InelasticSIDMState:
    """Restore only against the exact channel, species, and capacity contract."""

    if not isinstance(plan, InelasticSIDMPlan):
        raise TypeError("plan must be InelasticSIDMPlan.")
    plan._check_state_shapes(template)
    template_arrays: dict[str, object] = {}
    template_specification = pack_array_tree("state", template, template_arrays)
    expected_inventory = {
        name: (np.asarray(value).shape, np.asarray(value).dtype)
        for name, value in template_arrays.items()
    }
    manifest, arrays = read_array_archive(path, expected_inventory=expected_inventory)
    if (
        set(manifest) != {"format", "plan_id", "state", "payload_id", "arrays"}
        or manifest["format"] != _CHECKPOINT_FORMAT
        or manifest["plan_id"] != plan.plan_id
        or manifest["state"] != template_specification
    ):
        raise ValueError("Inelastic SIDM checkpoint runtime identity does not match.")
    payload_id = canonical_fingerprint(
        {
            "plan_id": plan.plan_id,
            "state": manifest["state"],
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    if payload_id != manifest["payload_id"]:
        raise ValueError("Inelastic SIDM checkpoint payload identity is corrupt.")
    restored = unpack_array_tree(manifest["state"], arrays, template)
    if not np.array_equal(
        np.asarray(restored.packets.packet_ids),
        np.asarray(template.packets.packet_ids),
    ):
        raise ValueError("Inelastic SIDM checkpoint particle support does not match.")
    if not isinstance(restored, InelasticSIDMState) or not bool(
        np.asarray(plan.state_valid(restored))
    ):
        raise ValueError("Restored inelastic SIDM state violates runtime invariants.")
    return restored


__all__ = [
    "DarkRadiationStateResult",
    "DarkRadiationTransactionEvidence",
    "DarkReactionEvidence",
    "DarkReactionPartialRates",
    "DarkReactionResult",
    "DarkReactionStatus",
    "DarkTwoBodyReactionPlan",
    "InelasticSIDMPlan",
    "InelasticSIDMState",
    "read_inelastic_sidm_checkpoint",
    "write_inelastic_sidm_checkpoint",
]
