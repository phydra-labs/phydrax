#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    ParticleDiscretization,
    ParticlePairKeySpace,
    ParticleSetPlan,
    PeriodicCell,
)
from ..typing import checked
from ._execution_binding import atomistic_case_address_words
from ._sites import AtomisticCoordinateMapPlan, PreparedAtomisticCoordinateMap
from ._topology import MolecularTopologyPlan, PreparedMolecularTopology
from ._types import AtomicStructure
from ._units import AtomisticUnitSystem


def _admit_particle_identity(
    particle_ids: ArrayLike,
    atomic_numbers: ArrayLike,
    masses: ArrayLike,
    coordinate_dtype: Any,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.dtype]:
    """Stable particle identity, atomic numbers and coordinate-dtype masses.

    Particle IDs are a unique non-empty ``int64`` vector, atomic numbers the
    matching ``int32`` vector, and masses are cast to the real floating
    ``coordinate_dtype``; their values are admitted against the occupancy masks.
    """
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
    return ids, numbers, mass, dtype


def _admit_occupancy_masks(
    ids: np.ndarray,
    numbers: np.ndarray,
    mass: np.ndarray,
    active_mask: ArrayLike | None,
    element_mask: ArrayLike | None,
    mobile_mask: ArrayLike | None,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Active, element and mobile masks and the chemistry they condition.

    The active set is non-empty; element and mobile particles are subsets of
    it.  Element particles carry positive atomic numbers and every other
    particle zero, and active masses are finite and positive.
    """
    active = (
        np.ones(ids.shape, dtype=np.bool_)
        if active_mask is None
        else np.asarray(active_mask, dtype=np.bool_)
    )
    if active.shape != ids.shape or not np.any(active):
        raise ValueError("active_mask must select at least one particle.")
    elements = (
        active.copy()
        if element_mask is None
        else np.asarray(element_mask, dtype=np.bool_)
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
        active.copy() if mobile_mask is None else np.asarray(mobile_mask, dtype=np.bool_)
    )
    if mobile.shape != ids.shape or np.any(mobile & ~active):
        raise ValueError("mobile_mask must be a subset of active_mask.")
    return active, elements, mobile


def _admit_particle_attributes(
    ids: np.ndarray,
    numbers: np.ndarray,
    mass: np.ndarray,
    active: np.ndarray,
    dtype: np.dtype,
    atom_type_ids: ArrayLike | None,
    charges: ArrayLike | None,
    molecule_ids: ArrayLike | None,
    region_ids: ArrayLike | None,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Canonical masses, atom types, charges and molecule/region labels.

    Active atom types are non-negative and active charges finite.  Inactive
    particles are canonicalized to unit mass, type zero, zero charge and zero
    labels, so padding content never enters the material digest.
    """
    atom_types = numbers.copy() if atom_type_ids is None else np.asarray(atom_type_ids)
    if atom_types.shape != ids.shape or not np.issubdtype(atom_types.dtype, np.integer):
        raise TypeError("atom_type_ids must be an integer vector matching particle_ids.")
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
            np.zeros(ids.shape, dtype=np.int32) if value is None else np.asarray(value)
        )
        if result.shape != ids.shape or not np.issubdtype(result.dtype, np.integer):
            raise TypeError(f"{name_} must be an integer vector matching particle_ids.")
        return result.astype(np.int32, copy=False)

    molecules = np.where(
        active,
        np.arange(ids.size, dtype=np.int32)
        if molecule_ids is None
        else labels("molecule_ids", molecule_ids),
        0,
    )
    regions = np.where(active, labels("region_ids", region_ids), 0)
    return mass, atom_types, charge, molecules, regions


class AtomisticSystemPlan(StrictModule, NonTrainableState):
    """Position-independent atomistic identity, topology, and fixed-cell policy.

    ``name`` and ``coordinate_dtype`` are declared executable configuration.  The
    canonical material digest (name, units, topology, cell, coordinate map and
    every material array) is the dynamic ``system_identity``; ``system_id``
    decodes the exact digest on the host.  The reference cell is resource
    layout: it fixes search grids and image stencils.
    """

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
    system_identity: IdentityWords
    name: str = eqx.field(static=True)
    coordinate_dtype: str = eqx.field(static=True)

    @checked
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
    ) -> None:
        ids, numbers, mass, dtype = _admit_particle_identity(
            particle_ids, atomic_numbers, masses, coordinate_dtype
        )
        active, elements, mobile = _admit_occupancy_masks(
            ids, numbers, mass, active_mask, element_mask, mobile_mask
        )
        mass, atom_types, charge, molecules, regions = _admit_particle_attributes(
            ids,
            numbers,
            mass,
            active,
            dtype,
            atom_type_ids,
            charges,
            molecule_ids,
            region_ids,
        )
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
        for field, value in arrays.items():
            setattr(self, field, jnp.asarray(value))
        self.units = units
        self.topology = topology_
        self.cell = cell
        self.coordinate_map = coordinate_map_
        self.name = identifier_name
        self.coordinate_dtype = dtype.name
        self.system_identity = IdentityWords.of(
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

    @property
    def system_id(self) -> str:
        """Host decoding of the canonical material digest."""
        return self.system_identity.fingerprint()

    @classmethod
    @checked
    def from_structure(
        cls,
        structure: AtomicStructure,
        units: AtomisticUnitSystem,
        /,
        **kwargs: Any,
    ) -> "AtomisticSystemPlan":
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

    @checked
    def with_cell(self, cell: PeriodicCell, /) -> "AtomisticSystemPlan":
        """Return the same particle identity, topology, and site map in another cell.

        The cell enters the material digest, so the result carries a regenerated
        ``system_id``.
        """
        return AtomisticSystemPlan(
            self.particle_ids,
            self.atomic_numbers,
            self.masses,
            self.units,
            atom_type_ids=self.atom_type_ids,
            element_mask=self.element_mask,
            charges=self.charges,
            active_mask=self.active_mask,
            mobile_mask=self.mobile_mask,
            molecule_ids=self.molecule_ids,
            region_ids=self.region_ids,
            topology=self.topology,
            cell=cell,
            coordinate_map=self.coordinate_map,
            name=self.name,
            coordinate_dtype=self.coordinate_dtype,
        )

    def prepare(self, /, *, numeric_version: str = "0") -> "PreparedAtomisticSystem":
        return PreparedAtomisticSystem(self, numeric_version=numeric_version)


def _particle_plan(plan: AtomisticSystemPlan, /) -> ParticleSetPlan:
    return ParticleSetPlan(
        plan.particle_ids,
        plan.masses,
        ambient_dimension=3,
        active_mask=plan.active_mask,
        name=f"{plan.name}-atoms",
        domain_labels=("atom",),
        coordinate_dtype=plan.coordinate_dtype,
    )


class PreparedAtomisticSystem(StrictModule, NonTrainableState):
    """Prepared atomistic support and topology as a layout-only executable.

    Static fields are layout: ``capacity``, ``numeric_version`` (a declared
    preparation label) and ``support_schema_id``, the indexed-support schema of
    this layout's particle slots that resource relations and the pair key space
    index.  The schema is never a material identity: the canonical prepared
    digest (material plan, prepared particles, topology, units, coordinate map,
    molecule labels, version) is the dynamic ``prepared_identity``, and
    ``prepared_id`` decodes it on the host.  Material-derived counts are
    dynamic: ``degrees_of_freedom`` (int32 scalar), ``molecule_count`` (int32
    scalar) and ``molecule_labels`` (the sorted unique active molecule labels
    in its first ``molecule_count`` entries, padded to ``capacity`` with the
    int32 maximum so the padded vector stays sorted).  ``case_address_words``
    is the ``(1, 8)`` random address of this prepared system
    (`atomistic_case_address_words` of ``prepared_id``), carried dynamically
    so stochastic graph addressing is unchanged.  Particle discretizations
    carry content-derived static identities and are never retained;
    `particle_discretization` prepares the support on the host for resource
    preparation.
    """

    plan: AtomisticSystemPlan
    topology: PreparedMolecularTopology
    pair_key_space: ParticlePairKeySpace
    coordinate_map: PreparedAtomisticCoordinateMap
    inverse_masses: Array
    active_mask: Array
    mobile_mask: Array
    molecule_labels: Array
    molecule_count: Array
    degrees_of_freedom: Array
    prepared_identity: IdentityWords
    case_address_words: Array
    capacity: int = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    support_schema_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, plan: AtomisticSystemPlan, /, *, numeric_version: str = "0"
    ) -> None:
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        particles = _particle_plan(plan).prepare(numeric_version=version)
        topology = plan.topology.prepare(particles)
        coordinate_map = plan.coordinate_map.prepare(particles)
        capacity = particles.capacity
        support_schema = canonical_fingerprint(
            {
                "kind": "atomistic-particle-support-schema",
                "entity": "atom",
                "capacity": capacity,
                "ambient_dimension": 3,
                "coordinate_dtype": plan.coordinate_dtype,
            }
        )
        inverse = jnp.where(plan.active_mask, 1.0 / plan.masses, 0.0)
        mobile_count = int(np.count_nonzero(np.asarray(plan.mobile_mask)))
        molecule_label_values = np.unique(
            np.asarray(plan.molecule_ids)[np.asarray(plan.active_mask)]
        )
        degrees = 3 * mobile_count - topology.constraint_count
        if degrees <= 0:
            raise ValueError(
                "Atomistic system has no unconstrained mobile degrees of freedom."
            )
        padded_labels = np.full((capacity,), np.iinfo(np.int32).max, dtype=np.int32)
        padded_labels[: molecule_label_values.size] = molecule_label_values
        self.plan = plan
        self.topology = topology
        self.pair_key_space = ParticlePairKeySpace(
            particle_ids=plan.particle_ids, support_id=support_schema
        )
        self.coordinate_map = coordinate_map
        self.inverse_masses = inverse
        self.active_mask = plan.active_mask
        self.mobile_mask = plan.mobile_mask
        self.molecule_labels = jnp.asarray(padded_labels)
        self.molecule_count = jnp.asarray(molecule_label_values.size, dtype=jnp.int32)
        self.degrees_of_freedom = jnp.asarray(degrees, dtype=jnp.int32)
        prepared_digest = canonical_fingerprint(
            {
                "kind": "prepared-atomistic-system",
                "plan": plan.system_id,
                "particles": particles.prepared_id,
                "topology": topology.topology_id,
                "units": plan.units.unit_system_id,
                "coordinate_map": coordinate_map.prepared_id,
                # The digest keeps the plan-dtype label scalars exactly as before.
                "molecule_labels": list(molecule_label_values),
                "numeric_version": version,
            }
        )
        self.prepared_identity = IdentityWords.from_fingerprint(prepared_digest)
        self.case_address_words = atomistic_case_address_words((prepared_digest,))
        self.capacity = capacity
        self.numeric_version = version
        self.support_schema_id = support_schema

    @property
    def prepared_id(self) -> str:
        """Host decoding of the canonical prepared-system digest."""
        return self.prepared_identity.fingerprint()

    @property
    def layout_id(self) -> str:
        """Content-free executable layout of this prepared system."""
        return canonical_fingerprint(
            {
                "kind": "atomistic-system-layout",
                "name": self.plan.name,
                "support": self.support_schema_id,
                "numeric_version": self.numeric_version,
                "units": self.plan.units.unit_system_id,
                "cell": None if self.cell is None else self.cell.cell_id,
                "topology_counts": [
                    list(self.topology.bond_indices.shape),
                    list(self.topology.angle_indices.shape),
                    list(self.topology.torsion_indices.shape),
                    list(self.topology.improper_indices.shape),
                    list(self.topology.constraint_indices.shape),
                    list(self.topology.exception_keys.shape),
                ],
                "sites": self.coordinate_map.plan.sites.capacity,
                "virtual_rules": [
                    [rule.kind.value, int(rule.parent_ids.shape[0])]
                    for rule in self.coordinate_map.plan.virtual_rules
                ],
            }
        )

    def particle_discretization(self) -> ParticleDiscretization:
        """Host preparation of this system's material particle support."""
        return _particle_plan(self.plan).prepare(numeric_version=self.numeric_version)

    def site_discretization(self) -> ParticleDiscretization:
        """Host preparation of this system's interaction-site particle support."""
        return self.coordinate_map.site_discretization()

    @property
    def cell(self) -> PeriodicCell | None:
        return self.plan.cell


__all__ = ["AtomisticSystemPlan", "PreparedAtomisticSystem"]
