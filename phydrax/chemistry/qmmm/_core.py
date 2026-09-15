#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite fixed-region ONIOM and electrostatic-embedding QM/MM surfaces."""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomisticSystemPlan, AtomisticUnitSystem
from ...units import BOHR, conversion_factor, DALTON, ELEMENTARY_CHARGE, HARTREE
from .._context import ElectrostaticEmbeddingState
from .._state import MolecularElectronicSectorPlan
from .._surface import (
    AbstractPreparedPotentialEnergySurface,
    PotentialEnergySurfaceCapabilities,
    PotentialEnergySurfaceEvaluation,
)
from ..electronic_structure._hartree_fock import NativeRHFPlan


class QuantumRegionPlan(StrictModule, NonTrainableState):
    """Stable-ID QM region and explicit QM/MM boundary bonds."""

    system: AtomisticSystemPlan
    particle_ids: tuple[int, ...] = eqx.field(static=True)
    boundary_bonds: tuple[tuple[int, int], ...] = eqx.field(static=True)
    link_ratio: float = eqx.field(static=True)
    total_charge: int = eqx.field(static=True)
    spin_multiplicity: int = eqx.field(static=True)
    region_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        particle_ids: Sequence[int],
        /,
        *,
        boundary_bonds: Sequence[tuple[int, int]] = (),
        link_ratio: float = 0.72,
        total_charge: int = 0,
        spin_multiplicity: int = 1,
    ):
        if not isinstance(system, AtomisticSystemPlan):
            raise TypeError("system must be AtomisticSystemPlan.")
        ids = tuple(int(value) for value in particle_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("QM particle IDs must be non-empty and unique.")
        active_ids = {
            int(value)
            for value, active in zip(
                np.asarray(system.particle_ids),
                np.asarray(system.active_mask),
                strict=True,
            )
            if bool(active)
        }
        if any(value not in active_ids for value in ids):
            raise ValueError("QM region references an inactive or unknown particle ID.")
        boundaries = tuple((int(qm), int(mm)) for qm, mm in boundary_bonds)
        if len(set(boundaries)) != len(boundaries):
            raise ValueError("QM/MM boundary bonds must be unique.")
        for qm, mm in boundaries:
            if qm not in ids or mm in ids or mm not in active_ids:
                raise ValueError(
                    "Each boundary bond must point from a QM atom to an active MM atom."
                )
        ratio = float(link_ratio)
        if not isfinite(ratio) or ratio <= 0.0 or ratio >= 1.0:
            raise ValueError("link_ratio must lie strictly between zero and one.")
        charge = int(total_charge)
        multiplicity = int(spin_multiplicity)
        if multiplicity <= 0:
            raise ValueError("spin_multiplicity must be positive.")
        self.system = system
        self.particle_ids = ids
        self.boundary_bonds = boundaries
        self.link_ratio = ratio
        self.total_charge = charge
        self.spin_multiplicity = multiplicity
        self.region_id = canonical_fingerprint(
            {
                "kind": "quantum-region-plan",
                "system": system.system_id,
                "particle_ids": list(ids),
                "boundary_bonds": [list(value) for value in boundaries],
                "link_ratio": ratio,
                "total_charge": charge,
                "spin_multiplicity": multiplicity,
            }
        )

    def prepare(self) -> PreparedQuantumRegion:
        return PreparedQuantumRegion(self)


class PreparedQuantumRegion(StrictModule, NonTrainableState):
    plan: QuantumRegionPlan
    region_system: AtomisticSystemPlan
    quantum_indices: tuple[int, ...] = eqx.field(static=True)
    boundary_indices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    link_particle_ids: tuple[int, ...] = eqx.field(static=True)
    mm_indices: tuple[int, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: QuantumRegionPlan, /):
        system = plan.system
        id_to_index = {
            int(value): index
            for index, value in enumerate(np.asarray(system.particle_ids))
        }
        quantum_indices = tuple(id_to_index[value] for value in plan.particle_ids)
        boundary_indices = tuple(
            (id_to_index[qm], id_to_index[mm]) for qm, mm in plan.boundary_bonds
        )
        active = np.asarray(system.active_mask, dtype=bool)
        quantum_set = set(quantum_indices)
        mm_indices = tuple(
            int(index)
            for index in np.flatnonzero(active)
            if int(index) not in quantum_set
        )
        maximum_id = int(np.max(np.asarray(system.particle_ids), initial=-1))
        link_ids = tuple(maximum_id + 1 + index for index in range(len(boundary_indices)))
        numbers = np.concatenate(
            (
                np.asarray(system.atomic_numbers)[list(quantum_indices)],
                np.ones((len(boundary_indices),), dtype=np.int32),
            )
        )
        hydrogen_mass = 1.00784 * float(conversion_factor(DALTON, system.units.mass_unit))
        masses = np.concatenate(
            (
                np.asarray(system.masses)[list(quantum_indices)],
                np.full((len(boundary_indices),), hydrogen_mass),
            )
        )
        region_ids = (*plan.particle_ids, *link_ids)
        region_system = AtomisticSystemPlan(
            region_ids,
            numbers,
            masses,
            system.units,
            atom_type_ids=np.arange(len(region_ids), dtype=np.int32),
            molecule_ids=np.zeros((len(region_ids),), dtype=np.int32),
            region_ids=np.zeros((len(region_ids),), dtype=np.int32),
            coordinate_dtype=system.coordinate_dtype,
            name=f"{system.name}:qm-region",
        )
        MolecularElectronicSectorPlan(plan.total_charge, plan.spin_multiplicity).prepare(
            region_system
        )
        self.plan = plan
        self.region_system = region_system
        self.quantum_indices = quantum_indices
        self.boundary_indices = boundary_indices
        self.link_particle_ids = link_ids
        self.mm_indices = mm_indices
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-quantum-region",
                "plan": plan.region_id,
                "region_system": region_system.system_id,
                "quantum_indices": list(quantum_indices),
                "boundary_indices": [list(value) for value in boundary_indices],
                "mm_indices": list(mm_indices),
            }
        )

    def realize(self, positions: ArrayLike, /) -> Array:
        coordinate = jnp.asarray(positions)
        expected = (int(self.plan.system.particle_ids.shape[0]), 3)
        if coordinate.shape != expected:
            raise ValueError(f"positions must have shape {expected}.")
        quantum = coordinate[jnp.asarray(self.quantum_indices)]
        if not self.boundary_indices:
            return quantum
        links = []
        for qm, mm in self.boundary_indices:
            links.append(
                (1.0 - self.plan.link_ratio) * coordinate[qm]
                + self.plan.link_ratio * coordinate[mm]
            )
        return jnp.concatenate((quantum, jnp.stack(tuple(links))), axis=0)

    def pullback(self, region_forces: ArrayLike, /) -> Array:
        force = jnp.asarray(region_forces)
        expected = (len(self.quantum_indices) + len(self.boundary_indices), 3)
        if force.shape != expected:
            raise ValueError(f"region_forces must have shape {expected}.")
        full = jnp.zeros(
            (int(self.plan.system.particle_ids.shape[0]), 3), dtype=force.dtype
        )
        full = full.at[jnp.asarray(self.quantum_indices)].add(
            force[: len(self.quantum_indices)]
        )
        for link_index, (qm, mm) in enumerate(self.boundary_indices):
            link_force = force[len(self.quantum_indices) + link_index]
            full = full.at[qm].add((1.0 - self.plan.link_ratio) * link_force)
            full = full.at[mm].add(self.plan.link_ratio * link_force)
        return full

    def embedding(self, positions: ArrayLike, /) -> ElectrostaticEmbeddingState:
        coordinate = jnp.asarray(positions)
        indices = jnp.asarray(self.mm_indices)
        return ElectrostaticEmbeddingState(
            jnp.asarray(self.plan.system.particle_ids)[indices],
            coordinate[indices],
            jnp.asarray(self.plan.system.charges)[indices],
            self.plan.system.units,
        )


class QMMMEvaluation(StrictModule, NonTrainableState):
    total_energy: Array
    classical_full_energy: Array
    quantum_model_energy: Array
    classical_model_energy: Array
    forces: Array
    point_charge_forces: Array | None
    successful: Array
    component_result_ids: tuple[str, ...] = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        total_energy: ArrayLike,
        classical_full_energy: ArrayLike,
        quantum_model_energy: ArrayLike,
        classical_model_energy: ArrayLike,
        forces: ArrayLike,
        successful: ArrayLike,
        component_result_ids: tuple[str, ...],
        /,
        *,
        point_charge_forces: ArrayLike | None = None,
    ):
        total = jnp.asarray(total_energy).reshape(())
        dtype = total.dtype
        classical_full = jnp.asarray(classical_full_energy, dtype=dtype).reshape(())
        quantum = jnp.asarray(quantum_model_energy, dtype=dtype).reshape(())
        classical_model = jnp.asarray(classical_model_energy, dtype=dtype).reshape(())
        force = jnp.asarray(forces, dtype=dtype)
        point_force = (
            None
            if point_charge_forces is None
            else jnp.asarray(point_charge_forces, dtype=dtype)
        )
        self.total_energy = total
        self.classical_full_energy = classical_full
        self.quantum_model_energy = quantum
        self.classical_model_energy = classical_model
        self.forces = force
        self.point_charge_forces = point_force
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.component_result_ids = component_result_ids
        self.result_id = canonical_fingerprint(
            {
                "kind": "qmmm-evaluation",
                "components": list(component_result_ids),
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "total_energy": np.asarray(total),
                        "classical_full_energy": np.asarray(classical_full),
                        "quantum_model_energy": np.asarray(quantum),
                        "classical_model_energy": np.asarray(classical_model),
                        "forces": np.asarray(force),
                        "point_charge_forces": (
                            None if point_force is None else np.asarray(point_force)
                        ),
                    }
                ),
            }
        )


class SubtractiveQMMMSurface(AbstractPreparedPotentialEnergySurface):
    region: PreparedQuantumRegion
    classical_full: AbstractPreparedPotentialEnergySurface
    quantum_model: AbstractPreparedPotentialEnergySurface
    classical_model: AbstractPreparedPotentialEnergySurface
    system_id: str = eqx.field(static=True)
    units: AtomisticUnitSystem
    provider_id: str = eqx.field(static=True)
    surface_id: str = eqx.field(static=True)
    capabilities: PotentialEnergySurfaceCapabilities

    def __init__(
        self,
        region: PreparedQuantumRegion,
        classical_full: AbstractPreparedPotentialEnergySurface,
        quantum_model: AbstractPreparedPotentialEnergySurface,
        classical_model: AbstractPreparedPotentialEnergySurface,
        /,
    ):
        if not isinstance(region, PreparedQuantumRegion):
            raise TypeError("region must be PreparedQuantumRegion.")
        surfaces = (classical_full, quantum_model, classical_model)
        if any(
            not isinstance(value, AbstractPreparedPotentialEnergySurface)
            for value in surfaces
        ):
            raise TypeError(
                "QM/MM components must be prepared potential-energy surfaces."
            )
        if classical_full.system_id != region.plan.system.system_id:
            raise ValueError("Full classical surface belongs to another system.")
        if quantum_model.system_id != region.region_system.system_id or (
            classical_model.system_id != region.region_system.system_id
        ):
            raise ValueError(
                "Model surfaces must use the prepared quantum-region system."
            )
        units = region.plan.system.units
        if any(value.units.unit_system_id != units.unit_system_id for value in surfaces):
            raise ValueError("QM/MM component unit systems differ.")
        capabilities = PotentialEnergySurfaceCapabilities(
            forces=True,
            conservative=all(value.capabilities.conservative for value in surfaces),
            differentiable=False,
        )
        self.region = region
        self.classical_full = classical_full
        self.quantum_model = quantum_model
        self.classical_model = classical_model
        self.system_id = region.plan.system.system_id
        self.units = units
        self.capabilities = capabilities
        self.provider_id = canonical_fingerprint(
            {"kind": "subtractive-qmmm-provider", "region": region.prepared_id}
        )
        self.surface_id = canonical_fingerprint(
            {
                "kind": "subtractive-qmmm-surface",
                "region": region.prepared_id,
                "components": [value.surface_id for value in surfaces],
            }
        )

    def evaluate_components(
        self, positions: ArrayLike, cell_vectors: ArrayLike | None = None, /
    ) -> QMMMEvaluation:
        full = self.classical_full.evaluate(positions, cell_vectors)
        region_positions = self.region.realize(positions)
        quantum = self.quantum_model.evaluate(region_positions, None)
        model = self.classical_model.evaluate(region_positions, None)
        correction_force = self.region.pullback(quantum.forces - model.forces)
        total_energy = full.energy + quantum.energy - model.energy
        forces = full.forces + correction_force
        successful = full.successful & quantum.successful & model.successful
        return QMMMEvaluation(
            total_energy,
            full.energy,
            quantum.energy,
            model.energy,
            forces,
            successful,
            (full.source_result_id, quantum.source_result_id, model.source_result_id),
        )

    def evaluate(
        self, positions: ArrayLike, cell_vectors: ArrayLike | None = None, /
    ) -> PotentialEnergySurfaceEvaluation:
        result = self.evaluate_components(positions, cell_vectors)
        return PotentialEnergySurfaceEvaluation(
            result.total_energy,
            result.forces,
            None,
            result.successful,
            provider_id=self.provider_id,
            source_result_id=result.result_id,
        )


class EmbeddedRegionEvaluation(StrictModule, NonTrainableState):
    energy: Array
    region_forces: Array
    point_charge_forces: Array
    successful: Array
    provider_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy: ArrayLike,
        region_forces: ArrayLike,
        point_charge_forces: ArrayLike,
        successful: ArrayLike,
        provider_id: str,
        /,
    ):
        energy_ = jnp.asarray(energy).reshape(())
        region = jnp.asarray(region_forces, dtype=energy_.dtype)
        points = jnp.asarray(point_charge_forces, dtype=energy_.dtype)
        if (
            region.ndim != 2
            or region.shape[1] != 3
            or points.ndim != 2
            or points.shape[1] != 3
        ):
            raise ValueError("Embedded forces must have shape (site, 3).")
        provider = str(provider_id).strip()
        if not provider:
            raise ValueError("provider_id must be non-empty.")
        self.energy = energy_
        self.region_forces = region
        self.point_charge_forces = points
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.provider_id = provider
        self.result_id = canonical_fingerprint(
            {
                "kind": "embedded-region-evaluation",
                "provider": provider,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "energy": np.asarray(energy_),
                        "region_forces": np.asarray(region),
                        "point_charge_forces": np.asarray(points),
                    }
                ),
            }
        )


class AbstractEmbeddedRegionProvider(StrictModule, NonTrainableState):
    provider_id: AbstractAttribute[str]
    conservative: AbstractAttribute[bool]
    region_system_id: AbstractAttribute[str]
    unit_system_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(
        self,
        region_positions: ArrayLike,
        embedding: ElectrostaticEmbeddingState,
        /,
    ) -> EmbeddedRegionEvaluation:
        raise NotImplementedError


EmbeddedEvaluator = Callable[
    [ArrayLike, ElectrostaticEmbeddingState], EmbeddedRegionEvaluation
]


class CallableEmbeddedRegionProvider(AbstractEmbeddedRegionProvider):
    evaluator: EmbeddedEvaluator
    declared_provider_id: str = eqx.field(static=True)
    region_system_id: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)

    def __init__(
        self,
        evaluator: EmbeddedEvaluator,
        provider_id: str,
        region_system_id: str,
        units: AtomisticUnitSystem,
        /,
        *,
        conservative: bool = True,
    ):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        provider = str(provider_id).strip()
        if not provider:
            raise ValueError("provider_id must be non-empty.")
        region_system = str(region_system_id).strip()
        if not region_system:
            raise ValueError("region_system_id must be non-empty.")
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        self.evaluator = evaluator
        self.declared_provider_id = provider
        self.region_system_id = region_system
        self.unit_system_id = units.unit_system_id
        self.provider_id = canonical_fingerprint(
            {
                "kind": "callable-embedded-region-provider",
                "declared_provider": provider,
                "region_system": region_system,
                "unit_system": units.unit_system_id,
                "conservative": bool(conservative),
            }
        )
        self.conservative = bool(conservative)

    def evaluate(
        self,
        region_positions: ArrayLike,
        embedding: ElectrostaticEmbeddingState,
        /,
    ) -> EmbeddedRegionEvaluation:
        result = self.evaluator(region_positions, embedding)
        if not isinstance(result, EmbeddedRegionEvaluation):
            raise TypeError("Embedded evaluator must return EmbeddedRegionEvaluation.")
        if result.provider_id != self.declared_provider_id:
            raise ValueError("Embedded evaluator changed declared provider identity.")
        return EmbeddedRegionEvaluation(
            result.energy,
            result.region_forces,
            result.point_charge_forces,
            result.successful,
            self.provider_id,
        )


class NativeRHFEmbeddedRegionProvider(AbstractEmbeddedRegionProvider):
    plan: NativeRHFPlan
    provider_id: str = eqx.field(static=True)
    region_system_id: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)

    def __init__(self, plan: NativeRHFPlan, /):
        if not isinstance(plan, NativeRHFPlan):
            raise TypeError("plan must be NativeRHFPlan.")
        self.plan = plan
        self.region_system_id = plan.system.system_id
        self.unit_system_id = plan.system.units.unit_system_id
        self.provider_id = canonical_fingerprint(
            {
                "kind": "native-rhf-embedded-region-provider",
                "plan": plan.plan_id,
                "region_system": self.region_system_id,
                "unit_system": self.unit_system_id,
            }
        )
        self.conservative = True

    def evaluate(
        self,
        region_positions: ArrayLike,
        embedding: ElectrostaticEmbeddingState,
        /,
    ) -> EmbeddedRegionEvaluation:
        if embedding.units.unit_system_id != self.plan.system.units.unit_system_id:
            raise ValueError("Embedding and native RHF unit systems differ.")
        region = jnp.asarray(region_positions)
        expected = (int(self.plan.system.particle_ids.shape[0]), 3)
        if region.shape != expected:
            raise ValueError(f"region_positions must have shape {expected}.")
        active = np.asarray(embedding.active_mask, dtype=bool)
        point_positions = jnp.asarray(embedding.positions)[active]
        charge_factor = float(
            conversion_factor(
                embedding.units.charge_unit,
                ELEMENTARY_CHARGE,
            )
        )
        point_charges = jnp.asarray(embedding.charges)[active] * charge_factor
        length_to_bohr = float(
            conversion_factor(
                self.plan.system.units.scale.length_unit,
                BOHR,
            )
        )
        energy_factor = float(
            conversion_factor(
                HARTREE,
                self.plan.system.units.scale.energy_unit,
            )
        )

        def solve(nuclei, points):
            return self.plan.solve_atomic_units(
                nuclei * length_to_bohr,
                embedding_positions_bohr=points * length_to_bohr,
                embedding_charges=point_charges,
            )

        central = solve(region, point_positions)
        displacement = self.plan.force_displacement
        region_forces = jnp.zeros_like(region)
        point_forces = jnp.zeros_like(point_positions)
        successful = bool(central.converged)
        for atom in range(int(region.shape[0])):
            for component in range(3):
                shift = jnp.zeros_like(region).at[atom, component].set(displacement)
                plus = solve(region + shift, point_positions)
                minus = solve(region - shift, point_positions)
                successful = successful and bool(plus.converged) and bool(minus.converged)
                region_forces = region_forces.at[atom, component].set(
                    -(plus.total_energy - minus.total_energy)
                    * energy_factor
                    / (2.0 * displacement)
                )
        for point in range(int(point_positions.shape[0])):
            for component in range(3):
                shift = (
                    jnp.zeros_like(point_positions).at[point, component].set(displacement)
                )
                plus = solve(region, point_positions + shift)
                minus = solve(region, point_positions - shift)
                successful = successful and bool(plus.converged) and bool(minus.converged)
                point_forces = point_forces.at[point, component].set(
                    -(plus.total_energy - minus.total_energy)
                    * energy_factor
                    / (2.0 * displacement)
                )
        return EmbeddedRegionEvaluation(
            central.total_energy * energy_factor,
            region_forces,
            point_forces,
            successful,
            self.provider_id,
        )


class ElectrostaticEmbeddingQMMMSurface(AbstractPreparedPotentialEnergySurface):
    region: PreparedQuantumRegion
    classical_partition: AbstractPreparedPotentialEnergySurface
    quantum_provider: AbstractEmbeddedRegionProvider
    classical_partition_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    units: AtomisticUnitSystem
    provider_id: str = eqx.field(static=True)
    surface_id: str = eqx.field(static=True)
    capabilities: PotentialEnergySurfaceCapabilities

    def __init__(
        self,
        region: PreparedQuantumRegion,
        classical_partition: AbstractPreparedPotentialEnergySurface,
        quantum_provider: AbstractEmbeddedRegionProvider,
        classical_partition_id: str,
        /,
    ):
        if not isinstance(region, PreparedQuantumRegion):
            raise TypeError("region must be PreparedQuantumRegion.")
        if not isinstance(classical_partition, AbstractPreparedPotentialEnergySurface):
            raise TypeError(
                "classical_partition must be a prepared potential-energy surface."
            )
        if not isinstance(quantum_provider, AbstractEmbeddedRegionProvider):
            raise TypeError(
                "quantum_provider must implement AbstractEmbeddedRegionProvider."
            )
        if quantum_provider.region_system_id != region.region_system.system_id:
            raise ValueError(
                "Embedded provider belongs to another quantum-region system."
            )
        if quantum_provider.unit_system_id != region.plan.system.units.unit_system_id:
            raise ValueError("Embedded provider and QM/MM surface unit systems differ.")
        if classical_partition.system_id != region.plan.system.system_id:
            raise ValueError("Classical partition belongs to another full system.")
        if (
            classical_partition.units.unit_system_id
            != region.plan.system.units.unit_system_id
        ):
            raise ValueError("Classical partition and QM region units differ.")
        partition_id = str(classical_partition_id).strip()
        if not partition_id:
            raise ValueError("classical_partition_id must be non-empty.")
        self.region = region
        self.classical_partition = classical_partition
        self.quantum_provider = quantum_provider
        self.classical_partition_id = partition_id
        self.system_id = region.plan.system.system_id
        self.units = region.plan.system.units
        self.provider_id = canonical_fingerprint(
            {
                "kind": "electrostatic-qmmm-provider",
                "quantum": quantum_provider.provider_id,
                "classical_partition": partition_id,
            }
        )
        self.capabilities = PotentialEnergySurfaceCapabilities(
            forces=True,
            conservative=(
                classical_partition.capabilities.conservative
                and quantum_provider.conservative
            ),
        )
        self.surface_id = canonical_fingerprint(
            {
                "kind": "electrostatic-embedding-qmmm-surface",
                "region": region.prepared_id,
                "classical": classical_partition.surface_id,
                "classical_partition": partition_id,
                "quantum": quantum_provider.provider_id,
            }
        )

    def evaluate_components(
        self, positions: ArrayLike, cell_vectors: ArrayLike | None = None, /
    ) -> QMMMEvaluation:
        classical = self.classical_partition.evaluate(positions, cell_vectors)
        region_positions = self.region.realize(positions)
        embedding = self.region.embedding(positions)
        quantum = self.quantum_provider.evaluate(region_positions, embedding)
        if quantum.point_charge_forces.shape != (
            len(self.region.mm_indices),
            3,
        ):
            raise ValueError(
                "Embedded provider point-charge forces do not align with MM sites."
            )
        full_quantum_force = self.region.pullback(quantum.region_forces)
        point_forces = jnp.zeros_like(classical.forces)
        point_forces = point_forces.at[jnp.asarray(self.region.mm_indices)].add(
            quantum.point_charge_forces
        )
        forces = classical.forces + full_quantum_force + point_forces
        energy = classical.energy + quantum.energy
        successful = classical.successful & quantum.successful
        return QMMMEvaluation(
            energy,
            classical.energy,
            quantum.energy,
            jnp.asarray(0.0, dtype=energy.dtype),
            forces,
            successful,
            (classical.source_result_id, quantum.result_id),
            point_charge_forces=quantum.point_charge_forces,
        )

    def evaluate(
        self, positions: ArrayLike, cell_vectors: ArrayLike | None = None, /
    ) -> PotentialEnergySurfaceEvaluation:
        result = self.evaluate_components(positions, cell_vectors)
        return PotentialEnergySurfaceEvaluation(
            result.total_energy,
            result.forces,
            None,
            result.successful,
            provider_id=self.provider_id,
            source_result_id=result.result_id,
        )


__all__ = [
    "AbstractEmbeddedRegionProvider",
    "CallableEmbeddedRegionProvider",
    "ElectrostaticEmbeddingQMMMSurface",
    "EmbeddedRegionEvaluation",
    "PreparedQuantumRegion",
    "QMMMEvaluation",
    "QuantumRegionPlan",
    "SubtractiveQMMMSurface",
    "NativeRHFEmbeddedRegionProvider",
]
