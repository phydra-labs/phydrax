#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    ParticleDiscretization,
    ParticlePairKeySpace,
    ParticleSetPlan,
    PeriodicCell,
)
from ._sites import AtomisticCoordinateMapPlan, PreparedAtomisticCoordinateMap
from ._topology import MolecularTopologyPlan, PreparedMolecularTopology
from ._types import AtomicStructure
from ._units import AtomisticUnitSystem


class AtomisticSystemPlan(StrictModule, NonTrainableState):
    """Position-independent atomistic identity, topology, and fixed-cell policy."""

    particle_ids: Array
    atomic_numbers: Array
    element_mask: Array
    atom_type_ids: Array
    masses: Array
    charges: Array
    active_mask: Array
    mobile_mask: Array
    molecule_ids: Array
    region_ids: Array
    units: AtomisticUnitSystem
    topology: MolecularTopologyPlan
    cell: PeriodicCell | None
    coordinate_map: AtomisticCoordinateMapPlan
    name: str = eqx.field(static=True)
    coordinate_dtype: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        particle_ids: ArrayLike,
        atomic_numbers: ArrayLike,
        masses: ArrayLike,
        units: AtomisticUnitSystem,
        /,
        *,
        atom_type_ids: ArrayLike | None = None,
        element_mask: ArrayLike | None = None,
        charges: ArrayLike | None = None,
        active_mask: ArrayLike | None = None,
        mobile_mask: ArrayLike | None = None,
        molecule_ids: ArrayLike | None = None,
        region_ids: ArrayLike | None = None,
        topology: MolecularTopologyPlan | None = None,
        cell: PeriodicCell | None = None,
        coordinate_map: AtomisticCoordinateMapPlan | None = None,
        name: str = "atomistic-system",
        coordinate_dtype: Any = "float64",
        system_id: str | None = None,
    ):
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be an AtomisticUnitSystem.")
        ids = np.asarray(particle_ids)
        numbers = np.asarray(atomic_numbers)
        mass = np.asarray(masses)
        if ids.ndim != 1 or ids.size == 0:
            raise ValueError("particle_ids must be a non-empty vector.")
        if numbers.shape != ids.shape or mass.shape != ids.shape:
            raise ValueError("Atomic numbers and masses must match particle_ids.")
        if not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(
            numbers.dtype, np.integer
        ):
            raise TypeError("Particle IDs and atomic numbers must be integers.")
        ids = ids.astype(np.int64, copy=False)
        numbers = numbers.astype(np.int32, copy=False)
        if np.unique(ids).size != ids.size:
            raise ValueError("Atomistic systems require unique stable particle IDs.")
        dtype = np.dtype(coordinate_dtype)
        if dtype.kind != "f":
            raise TypeError("coordinate_dtype must be a real floating dtype.")
        mass = mass.astype(dtype, copy=False)
        active = (
            np.ones(ids.shape, dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if active.shape != ids.shape or not np.any(active):
            raise ValueError("active_mask must select at least one particle.")
        elements = (
            active.copy()
            if element_mask is None
            else np.asarray(element_mask, dtype=bool)
        )
        if elements.shape != ids.shape or np.any(elements & ~active):
            raise ValueError("element_mask must be a subset of active_mask.")
        if np.any(numbers[elements] <= 0) or np.any(numbers[~elements] != 0):
            raise ValueError(
                "Element particles require positive atomic numbers; other particles use zero."
            )
        if np.any(~np.isfinite(mass[active])) or np.any(mass[active] <= 0.0):
            raise ValueError("Active masses must be finite and positive.")
        mobile = (
            active.copy() if mobile_mask is None else np.asarray(mobile_mask, dtype=bool)
        )
        if mobile.shape != ids.shape or np.any(mobile & ~active):
            raise ValueError("mobile_mask must be a subset of active_mask.")
        atom_types = (
            numbers.copy() if atom_type_ids is None else np.asarray(atom_type_ids)
        )
        if atom_types.shape != ids.shape or not np.issubdtype(
            atom_types.dtype, np.integer
        ):
            raise TypeError(
                "atom_type_ids must be an integer vector matching particle_ids."
            )
        atom_types = atom_types.astype(np.int32, copy=False)
        if np.any(atom_types[active] < 0):
            raise ValueError("Active atom type IDs must be non-negative.")
        charge = (
            np.zeros(ids.shape, dtype=dtype)
            if charges is None
            else np.asarray(charges, dtype=dtype)
        )
        if charge.shape != ids.shape or np.any(~np.isfinite(charge[active])):
            raise ValueError("charges must be finite and match particle_ids.")
        mass = np.where(active, mass, 1.0)
        atom_types = np.where(active, atom_types, 0)
        charge = np.where(active, charge, 0.0)

        def labels(name_: str, value: ArrayLike | None) -> np.ndarray:
            result = (
                np.zeros(ids.shape, dtype=np.int32)
                if value is None
                else np.asarray(value)
            )
            if result.shape != ids.shape or not np.issubdtype(result.dtype, np.integer):
                raise TypeError(
                    f"{name_} must be an integer vector matching particle_ids."
                )
            return result.astype(np.int32, copy=False)

        molecules = np.where(
            active,
            np.arange(ids.size, dtype=np.int32)
            if molecule_ids is None
            else labels("molecule_ids", molecule_ids),
            0,
        )
        regions = np.where(active, labels("region_ids", region_ids), 0)
        topology_ = MolecularTopologyPlan.empty() if topology is None else topology
        if not isinstance(topology_, MolecularTopologyPlan):
            raise TypeError("topology must be a MolecularTopologyPlan or None.")
        if cell is not None:
            if not isinstance(cell, PeriodicCell):
                raise TypeError("cell must be a PeriodicCell or None.")
            if cell.ambient_dimension != 3:
                raise ValueError("Atomistic dynamics requires a three-dimensional cell.")
        coordinate_map_ = (
            AtomisticCoordinateMapPlan.identity(
                ids,
                numbers,
                atom_types,
                charge,
                element_mask=elements,
                active_mask=active,
            )
            if coordinate_map is None
            else coordinate_map
        )
        if not isinstance(coordinate_map_, AtomisticCoordinateMapPlan):
            raise TypeError("coordinate_map must be AtomisticCoordinateMapPlan or None.")
        if not np.array_equal(np.asarray(coordinate_map_.dof_particle_ids), ids):
            raise ValueError("Coordinate-map DOF identity differs from particle_ids.")
        identifier_name = str(name).strip()
        if not identifier_name:
            raise ValueError("name must be non-empty.")
        arrays = {
            "particle_ids": ids,
            "atomic_numbers": numbers,
            "element_mask": elements,
            "atom_type_ids": atom_types,
            "masses": mass,
            "charges": charge,
            "active_mask": active,
            "mobile_mask": mobile,
            "molecule_ids": molecules,
            "region_ids": regions,
        }
        generated = canonical_fingerprint(
            {
                "kind": "atomistic-system-plan",
                "name": identifier_name,
                "units": units.unit_system_id,
                "topology": topology_.plan_id,
                "cell": None if cell is None else cell.cell_id,
                "coordinate_map": coordinate_map_.plan_id,
                "arrays": array_tree_fingerprint(arrays),
                "coordinate_dtype": dtype.name,
            }
        )
        resolved = generated if system_id is None else str(system_id)
        if not resolved:
            raise ValueError("system_id must be non-empty.")
        for field, value in arrays.items():
            setattr(self, field, jnp.asarray(value))
        self.units = units
        self.topology = topology_
        self.cell = cell
        self.coordinate_map = coordinate_map_
        self.name = identifier_name
        self.coordinate_dtype = dtype.name
        self.system_id = resolved

    @classmethod
    def from_structure(
        cls,
        structure: AtomicStructure,
        units: AtomisticUnitSystem,
        /,
        **kwargs: Any,
    ) -> "AtomisticSystemPlan":
        if not isinstance(structure, AtomicStructure):
            raise TypeError("structure must be an AtomicStructure.")
        if structure.scale.scale_id != units.scale.scale_id:
            raise ValueError("Structure and dynamics units must share one exact scale.")
        cell = kwargs.pop("cell", None)
        if cell is None and structure.cell is not None:
            axes = (
                (False, False, False)
                if structure.periodic_axes is None
                else tuple(bool(value) for value in np.asarray(structure.periodic_axes))
            )
            cell = PeriodicCell(np.asarray(structure.cell), periodic_axes=axes)
        return cls(
            structure.particle_ids,
            structure.atomic_numbers,
            structure.masses,
            units,
            active_mask=structure.active_mask,
            cell=cell,
            name=structure.name,
            coordinate_dtype=structure.positions.dtype,
            **kwargs,
        )

    def prepare(self, /, *, numeric_version: str = "0") -> "PreparedAtomisticSystem":
        return PreparedAtomisticSystem(self, numeric_version=numeric_version)


class PreparedAtomisticSystem(StrictModule, NonTrainableState):
    """Prepared static atomistic support and topology."""

    plan: AtomisticSystemPlan
    particles: ParticleDiscretization
    topology: PreparedMolecularTopology
    pair_key_space: ParticlePairKeySpace
    coordinate_map: PreparedAtomisticCoordinateMap
    inverse_masses: Array
    active_mask: Array
    mobile_mask: Array
    molecule_labels: tuple[int, ...] = eqx.field(static=True)
    degrees_of_freedom: int = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: AtomisticSystemPlan, /, *, numeric_version: str = "0"):
        if not isinstance(plan, AtomisticSystemPlan):
            raise TypeError("plan must be an AtomisticSystemPlan.")
        particles = ParticleSetPlan(
            plan.particle_ids,
            plan.masses,
            ambient_dimension=3,
            active_mask=plan.active_mask,
            name=f"{plan.name}-atoms",
            domain_labels=("atom",),
            coordinate_dtype=plan.coordinate_dtype,
        ).prepare(numeric_version=numeric_version)
        topology = plan.topology.prepare(particles)
        pair_key_space = ParticlePairKeySpace(particles)
        coordinate_map = plan.coordinate_map.prepare(particles)
        inverse = jnp.where(plan.active_mask, 1.0 / plan.masses, 0.0)
        mobile_count = int(np.count_nonzero(np.asarray(plan.mobile_mask)))
        molecule_labels = tuple(
            int(value)
            for value in np.unique(
                np.asarray(plan.molecule_ids)[np.asarray(plan.active_mask)]
            )
        )
        degrees = 3 * mobile_count - topology.constraint_count
        if degrees <= 0:
            raise ValueError(
                "Atomistic system has no unconstrained mobile degrees of freedom."
            )
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        self.plan = plan
        self.particles = particles
        self.topology = topology
        self.pair_key_space = pair_key_space
        self.coordinate_map = coordinate_map
        self.inverse_masses = inverse
        self.active_mask = plan.active_mask
        self.mobile_mask = plan.mobile_mask
        self.molecule_labels = molecule_labels
        self.degrees_of_freedom = degrees
        self.numeric_version = version
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-atomistic-system",
                "plan": plan.system_id,
                "particles": particles.prepared_id,
                "topology": topology.topology_id,
                "units": plan.units.unit_system_id,
                "coordinate_map": coordinate_map.prepared_id,
                "molecule_labels": list(molecule_labels),
                "numeric_version": version,
            }
        )

    @property
    def capacity(self) -> int:
        return self.particles.capacity

    @property
    def cell(self) -> PeriodicCell | None:
        return self.plan.cell


__all__ = ["AtomisticSystemPlan", "PreparedAtomisticSystem"]
