#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from itertools import pairwise

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import (
    AtomisticSystemPlan,
    AtomisticUnitSystem,
    MolecularTopologyPlan,
    PolymerChainLayoutPlan,
)
from ...discretization import PeriodicCell


class PolymerChainSpec(StrictModule, NonTrainableState):
    chain_id: str = eqx.field(static=True)
    bead_type_indices: Array
    ring: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        chain_id: str,
        bead_type_indices: ArrayLike,
        /,
        *,
        ring: bool = False,
    ):
        identifier = str(chain_id).strip()
        sequence = np.asarray(bead_type_indices, dtype=np.int32)
        ring_ = bool(ring)
        if (
            not identifier
            or sequence.ndim != 1
            or sequence.size == 0
            or np.any(sequence < 0)
            or (ring_ and sequence.size < 3)
        ):
            raise ValueError("Polymer chain specification is invalid.")
        self.chain_id = identifier
        self.bead_type_indices = jnp.asarray(sequence)
        self.ring = ring_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polymer-chain-spec",
                "chain_id": identifier,
                "bead_type_indices": sequence.tolist(),
                "ring": ring_,
            }
        )


class PolymerConnectionPortPlan(StrictModule, NonTrainableState):
    port_id: str = eqx.field(static=True)
    chain_id: str = eqx.field(static=True)
    bead_offset: int = eqx.field(static=True)
    compatibility_class: str = eqx.field(static=True)
    maximum_uses: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        port_id: str,
        chain_id: str,
        bead_offset: int,
        compatibility_class: str,
        /,
        *,
        maximum_uses: int = 1,
    ):
        identifier = str(port_id).strip()
        chain = str(chain_id).strip()
        compatibility = str(compatibility_class).strip()
        offset = int(bead_offset)
        uses = int(maximum_uses)
        if not identifier or not chain or not compatibility or offset < 0 or uses <= 0:
            raise ValueError("Polymer connection-port definition is invalid.")
        self.port_id = identifier
        self.chain_id = chain
        self.bead_offset = offset
        self.compatibility_class = compatibility
        self.maximum_uses = uses
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polymer-connection-port-plan",
                "port_id": identifier,
                "chain_id": chain,
                "bead_offset": offset,
                "compatibility_class": compatibility,
                "maximum_uses": uses,
            }
        )


class PolymerMaterialRecipePlan(StrictModule, NonTrainableState):
    material_id: str = eqx.field(static=True)
    bead_type_ids: tuple[str, ...] = eqx.field(static=True)
    bead_masses: Array
    bead_charges: Array
    chains: tuple[PolymerChainSpec, ...]
    ports: tuple[PolymerConnectionPortPlan, ...]
    units: AtomisticUnitSystem
    cell: PeriodicCell | None
    maximum_particles: int = eqx.field(static=True)
    particle_id_start: int = eqx.field(static=True)
    bonded_lennard_jones_scale: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        material_id: str,
        bead_type_ids: tuple[str, ...],
        bead_masses: ArrayLike,
        chains: tuple[PolymerChainSpec, ...],
        units: AtomisticUnitSystem,
        /,
        *,
        bead_charges: ArrayLike | None = None,
        ports: tuple[PolymerConnectionPortPlan, ...] = (),
        cell: PeriodicCell | None = None,
        maximum_particles: int | None = None,
        particle_id_start: int = 1,
        bonded_lennard_jones_scale: float = 1.0,
    ):
        identifier = str(material_id).strip()
        type_ids = tuple(str(value).strip() for value in bead_type_ids)
        masses = np.asarray(bead_masses, dtype=float)
        chain_values = tuple(chains)
        port_values = tuple(ports)
        charges = (
            np.zeros_like(masses)
            if bead_charges is None
            else np.asarray(bead_charges, dtype=float)
        )
        active_count = sum(int(chain.bead_type_indices.size) for chain in chain_values)
        capacity = active_count if maximum_particles is None else int(maximum_particles)
        start = int(particle_id_start)
        bonded_scale = float(bonded_lennard_jones_scale)
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        if cell is not None and not isinstance(cell, PeriodicCell):
            raise TypeError("cell must be PeriodicCell or None.")
        if (
            not identifier
            or not type_ids
            or any(not value for value in type_ids)
            or len(set(type_ids)) != len(type_ids)
            or masses.shape != (len(type_ids),)
            or charges.shape != masses.shape
            or np.any(~np.isfinite(masses))
            or np.any(masses <= 0.0)
            or np.any(~np.isfinite(charges))
            or not chain_values
            or any(not isinstance(chain, PolymerChainSpec) for chain in chain_values)
            or len({chain.chain_id for chain in chain_values}) != len(chain_values)
            or any(
                int(jnp.max(chain.bead_type_indices)) >= len(type_ids)
                for chain in chain_values
            )
            or any(
                not isinstance(port, PolymerConnectionPortPlan) for port in port_values
            )
            or len({port.port_id for port in port_values}) != len(port_values)
            or capacity < active_count
            or start < 0
            or not math.isfinite(bonded_scale)
            or bonded_scale < 0.0
        ):
            raise ValueError("Polymer material recipe is invalid.")
        chain_by_id = {chain.chain_id: chain for chain in chain_values}
        if any(
            port.chain_id not in chain_by_id
            or port.bead_offset >= chain_by_id[port.chain_id].bead_type_indices.size
            for port in port_values
        ):
            raise ValueError("Polymer connection port references an absent chain bead.")
        self.material_id = identifier
        self.bead_type_ids = type_ids
        self.bead_masses = jnp.asarray(masses)
        self.bead_charges = jnp.asarray(charges)
        self.chains = chain_values
        self.ports = port_values
        self.units = units
        self.cell = cell
        self.maximum_particles = capacity
        self.particle_id_start = start
        self.bonded_lennard_jones_scale = bonded_scale
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polymer-material-recipe-plan",
                "material_id": identifier,
                "bead_type_ids": list(type_ids),
                "bead_masses": array_tree_fingerprint(masses),
                "bead_charges": array_tree_fingerprint(charges),
                "chains": [chain.plan_id for chain in chain_values],
                "ports": [port.plan_id for port in port_values],
                "units": units.unit_system_id,
                "cell": None if cell is None else cell.cell_id,
                "maximum_particles": capacity,
                "particle_id_start": start,
                "bonded_lennard_jones_scale": bonded_scale,
            }
        )


class RealizedPolymerConnectionPort(StrictModule, NonTrainableState):
    port_id: str = eqx.field(static=True)
    particle_id: int = eqx.field(static=True)
    compatibility_class: str = eqx.field(static=True)
    maximum_uses: int = eqx.field(static=True)
    uses: int = eqx.field(static=True)

    def __init__(
        self,
        port_id: str,
        particle_id: int,
        compatibility_class: str,
        maximum_uses: int,
        uses: int = 0,
        /,
    ):
        uses_ = int(uses)
        maximum = int(maximum_uses)
        if uses_ < 0 or maximum <= 0 or uses_ > maximum:
            raise ValueError("Realized connection-port use count is invalid.")
        self.port_id = str(port_id)
        self.particle_id = int(particle_id)
        self.compatibility_class = str(compatibility_class)
        self.maximum_uses = maximum
        self.uses = uses_

    @property
    def available(self) -> bool:
        return self.uses < self.maximum_uses


class PolymerLoweringRecord(StrictModule, NonTrainableState):
    recipe_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    chain_ids: tuple[str, ...] = eqx.field(static=True)
    chain_particle_ids: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        recipe_id: str,
        system_id: str,
        topology_id: str,
        chain_ids: tuple[str, ...],
        chain_particle_ids: tuple[tuple[int, ...], ...],
        /,
    ):
        self.recipe_id = recipe_id
        self.system_id = system_id
        self.topology_id = topology_id
        self.chain_ids = chain_ids
        self.chain_particle_ids = chain_particle_ids
        self.record_id = canonical_fingerprint(
            {
                "kind": "polymer-lowering-record",
                "recipe": recipe_id,
                "system": system_id,
                "topology": topology_id,
                "chain_ids": list(chain_ids),
                "chain_particle_ids": [list(value) for value in chain_particle_ids],
            }
        )


class PolymerEnsembleStatistics(StrictModule):
    chain_lengths: Array
    number_average_length: Array
    weight_average_length: Array
    dispersity: Array
    bead_type_fractions: Array


class PolymerConstructionResult(StrictModule, NonTrainableState):
    system: AtomisticSystemPlan
    topology: MolecularTopologyPlan
    chain_layout: PolymerChainLayoutPlan
    ports: tuple[RealizedPolymerConnectionPort, ...]
    lowering: PolymerLoweringRecord
    statistics: PolymerEnsembleStatistics
    successful: bool = eqx.field(static=True)
    recipe_id: str = eqx.field(static=True)


def lower_polymer_recipe(
    recipe: PolymerMaterialRecipePlan, /
) -> PolymerConstructionResult:
    if not isinstance(recipe, PolymerMaterialRecipePlan):
        raise TypeError("recipe must be PolymerMaterialRecipePlan.")
    capacity = recipe.maximum_particles
    active_count = sum(int(chain.bead_type_indices.size) for chain in recipe.chains)
    particle_ids = np.arange(
        recipe.particle_id_start,
        recipe.particle_id_start + capacity,
        dtype=np.int64,
    )
    active = np.arange(capacity) < active_count
    atom_types = np.zeros((capacity,), dtype=np.int32)
    masses = np.ones((capacity,), dtype=float)
    charges = np.zeros((capacity,), dtype=float)
    molecule_ids = np.zeros((capacity,), dtype=np.int32)
    chain_particle_ids: list[tuple[int, ...]] = []
    bonds: list[tuple[int, int]] = []
    angles: list[tuple[int, int, int]] = []
    cursor = 0
    maximum_chain_length = max(
        int(chain.bead_type_indices.size) for chain in recipe.chains
    )
    layout_indices = np.zeros((len(recipe.chains), maximum_chain_length), dtype=np.int32)
    layout_mask = np.zeros_like(layout_indices, dtype=bool)
    slot_by_chain: dict[str, np.ndarray] = {}
    for chain_index, chain in enumerate(recipe.chains):
        sequence = np.asarray(chain.bead_type_indices, dtype=np.int32)
        length = sequence.size
        slots = np.arange(cursor, cursor + length, dtype=np.int32)
        stable = particle_ids[slots]
        slot_by_chain[chain.chain_id] = slots
        chain_particle_ids.append(tuple(int(value) for value in stable))
        atom_types[slots] = sequence
        masses[slots] = np.asarray(recipe.bead_masses)[sequence]
        charges[slots] = np.asarray(recipe.bead_charges)[sequence]
        molecule_ids[slots] = chain_index
        layout_indices[chain_index, :length] = slots
        layout_mask[chain_index, :length] = True
        bonds.extend((int(left), int(right)) for left, right in pairwise(stable))
        angles.extend(
            (int(left), int(center), int(right))
            for left, center, right in zip(
                stable[:-2], stable[1:-1], stable[2:], strict=True
            )
        )
        if chain.ring:
            bonds.append((int(stable[-1]), int(stable[0])))
            for offset in range(length):
                angles.append(
                    (
                        int(stable[(offset - 1) % length]),
                        int(stable[offset]),
                        int(stable[(offset + 1) % length]),
                    )
                )
        cursor += length
    exception_pairs = bonds if recipe.bonded_lennard_jones_scale != 1.0 else []
    topology = MolecularTopologyPlan(
        bonds=None if not bonds else bonds,
        angles=None if not angles else sorted(set(angles)),
        pair_exceptions=None if not exception_pairs else exception_pairs,
        lennard_jones_scales=(
            np.full((len(exception_pairs),), recipe.bonded_lennard_jones_scale)
            if exception_pairs
            else None
        ),
    )
    system = AtomisticSystemPlan(
        particle_ids,
        np.zeros((capacity,), dtype=np.int32),
        masses,
        recipe.units,
        atom_type_ids=atom_types,
        element_mask=np.zeros((capacity,), dtype=bool),
        charges=charges,
        active_mask=active,
        mobile_mask=active,
        molecule_ids=molecule_ids,
        topology=topology,
        cell=recipe.cell,
        name=recipe.material_id,
    )
    chain_layout = PolymerChainLayoutPlan(
        layout_indices,
        layout_mask,
        maximum_frames=1,
    )
    realized_ports = tuple(
        RealizedPolymerConnectionPort(
            port.port_id,
            int(particle_ids[slot_by_chain[port.chain_id][port.bead_offset]]),
            port.compatibility_class,
            port.maximum_uses,
        )
        for port in recipe.ports
    )
    lengths = np.asarray(
        [chain.bead_type_indices.size for chain in recipe.chains], dtype=float
    )
    number_average = np.mean(lengths)
    weight_average = np.sum(lengths * lengths) / np.sum(lengths)
    type_counts = np.bincount(
        atom_types[:active_count], minlength=len(recipe.bead_type_ids)
    )
    statistics = PolymerEnsembleStatistics(
        jnp.asarray(lengths, dtype=jnp.int32),
        jnp.asarray(number_average),
        jnp.asarray(weight_average),
        jnp.asarray(weight_average / number_average),
        jnp.asarray(type_counts / active_count),
    )
    lowering = PolymerLoweringRecord(
        recipe.plan_id,
        system.system_id,
        topology.plan_id,
        tuple(chain.chain_id for chain in recipe.chains),
        tuple(chain_particle_ids),
    )
    return PolymerConstructionResult(
        system,
        topology,
        chain_layout,
        realized_ports,
        lowering,
        statistics,
        True,
        recipe.plan_id,
    )


__all__ = [
    "PolymerChainSpec",
    "PolymerConnectionPortPlan",
    "PolymerConstructionResult",
    "PolymerEnsembleStatistics",
    "PolymerLoweringRecord",
    "PolymerMaterialRecipePlan",
    "RealizedPolymerConnectionPort",
    "lower_polymer_recipe",
]
