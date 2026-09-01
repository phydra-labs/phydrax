#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Iterable

import numpy as np

from ..._fingerprint import canonical_fingerprint
from ._types import BondOrder, ConnectionKind, EntityKind, PolymerKind


def _text(value: str, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


@dataclass(frozen=True, slots=True)
class EntityRecord:
    """One author-independent mmCIF entity and its declared polymer sequence."""

    entity_id: str
    kind: EntityKind
    description: str = ""
    polymer_kind: PolymerKind = PolymerKind.NONE
    sequence_components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _text(self.entity_id, "entity_id"))
        object.__setattr__(self, "kind", EntityKind(self.kind))
        object.__setattr__(self, "description", str(self.description).strip())
        object.__setattr__(self, "polymer_kind", PolymerKind(self.polymer_kind))
        components = tuple(
            _text(value, "sequence component") for value in self.sequence_components
        )
        object.__setattr__(self, "sequence_components", components)
        if (
            self.kind is not EntityKind.POLYMER
            and self.polymer_kind is not PolymerKind.NONE
        ):
            raise ValueError("Only polymer entities may declare a polymer_kind.")


@dataclass(frozen=True, slots=True)
class ChainRecord:
    """One label/auth asymmetric unit identifier pair."""

    label_asym_id: str
    auth_asym_id: str
    entity_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "label_asym_id", _text(self.label_asym_id, "label_asym_id")
        )
        object.__setattr__(self, "auth_asym_id", _text(self.auth_asym_id, "auth_asym_id"))
        object.__setattr__(self, "entity_id", _text(self.entity_id, "entity_id"))


@dataclass(frozen=True, slots=True)
class ResidueRecord:
    """One chemical-component occurrence with both label and author numbering."""

    chain_index: int
    label_comp_id: str
    auth_comp_id: str
    label_seq_id: int | None
    auth_seq_id: int | None
    insertion_code: str | None = None
    entity_sequence_index: int | None = None
    hetero: bool = False

    def __post_init__(self) -> None:
        if int(self.chain_index) < 0:
            raise ValueError("chain_index must be non-negative.")
        object.__setattr__(self, "chain_index", int(self.chain_index))
        object.__setattr__(
            self, "label_comp_id", _text(self.label_comp_id, "label_comp_id")
        )
        object.__setattr__(self, "auth_comp_id", _text(self.auth_comp_id, "auth_comp_id"))
        object.__setattr__(
            self,
            "label_seq_id",
            None if self.label_seq_id is None else int(self.label_seq_id),
        )
        object.__setattr__(
            self,
            "auth_seq_id",
            None if self.auth_seq_id is None else int(self.auth_seq_id),
        )
        object.__setattr__(
            self,
            "entity_sequence_index",
            None
            if self.entity_sequence_index is None
            else int(self.entity_sequence_index),
        )
        object.__setattr__(self, "insertion_code", _optional_text(self.insertion_code))
        object.__setattr__(self, "hetero", bool(self.hetero))


@dataclass(frozen=True, slots=True)
class AtomRecord:
    """One coordinate observation, including model and alternate-location identity."""

    atom_site_id: str
    residue_index: int
    model_number: int
    label_atom_id: str
    auth_atom_id: str
    element: str
    atomic_number: int
    position: tuple[float, float, float]
    occupancy: float = 1.0
    b_factor: float = 0.0
    altloc_id: str | None = None
    formal_charge: int = 0
    present: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "atom_site_id", _text(self.atom_site_id, "atom_site_id"))
        if int(self.residue_index) < 0 or int(self.model_number) <= 0:
            raise ValueError(
                "residue_index must be non-negative and model_number positive."
            )
        object.__setattr__(self, "residue_index", int(self.residue_index))
        object.__setattr__(self, "model_number", int(self.model_number))
        object.__setattr__(
            self, "label_atom_id", _text(self.label_atom_id, "label_atom_id")
        )
        object.__setattr__(self, "auth_atom_id", _text(self.auth_atom_id, "auth_atom_id"))
        element = _text(self.element, "element").upper()
        object.__setattr__(self, "element", element)
        number = int(self.atomic_number)
        if number < 0:
            raise ValueError(
                "atomic_number must be non-negative; zero means unresolved chemistry."
            )
        object.__setattr__(self, "atomic_number", number)
        position = tuple(float(value) for value in self.position)
        if len(position) != 3 or (
            self.present and not all(isfinite(value) for value in position)
        ):
            raise ValueError("Present atom positions must be finite three-vectors.")
        object.__setattr__(self, "position", position)
        occupancy = float(self.occupancy)
        b_factor = float(self.b_factor)
        if not isfinite(occupancy) or occupancy < 0.0 or occupancy > 1.0:
            raise ValueError("occupancy must be finite and lie in [0, 1].")
        if not isfinite(b_factor) or b_factor < 0.0:
            raise ValueError("b_factor must be finite and non-negative.")
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "b_factor", b_factor)
        object.__setattr__(self, "altloc_id", _optional_text(self.altloc_id))
        object.__setattr__(self, "formal_charge", int(self.formal_charge))
        object.__setattr__(self, "present", bool(self.present))

    @property
    def identity_key(self) -> tuple[int, str, str | None]:
        """Identity shared by the same atom across coordinate models."""

        return (self.residue_index, self.label_atom_id, self.altloc_id)


@dataclass(frozen=True, slots=True)
class ChemicalComponentAtom:
    atom_id: str
    element: str
    atomic_number: int
    formal_charge: int = 0
    aromatic: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "atom_id", _text(self.atom_id, "atom_id"))
        object.__setattr__(self, "element", _text(self.element, "element").upper())
        if int(self.atomic_number) < 0:
            raise ValueError("atomic_number must be non-negative.")
        object.__setattr__(self, "atomic_number", int(self.atomic_number))
        object.__setattr__(self, "formal_charge", int(self.formal_charge))
        object.__setattr__(self, "aromatic", bool(self.aromatic))


@dataclass(frozen=True, slots=True)
class ChemicalComponentBond:
    atom_id_1: str
    atom_id_2: str
    order: BondOrder = BondOrder.UNKNOWN
    aromatic: bool = False

    def __post_init__(self) -> None:
        first = _text(self.atom_id_1, "atom_id_1")
        second = _text(self.atom_id_2, "atom_id_2")
        if first == second:
            raise ValueError("A component bond requires two distinct atoms.")
        object.__setattr__(self, "atom_id_1", first)
        object.__setattr__(self, "atom_id_2", second)
        object.__setattr__(self, "order", BondOrder(self.order))
        object.__setattr__(self, "aromatic", bool(self.aromatic))


@dataclass(frozen=True, slots=True)
class ChemicalComponent:
    """Local chemical-component dictionary entry; no external dictionary is implied."""

    component_id: str
    name: str
    component_type: str
    atoms: tuple[ChemicalComponentAtom, ...] = ()
    bonds: tuple[ChemicalComponentBond, ...] = ()
    parent_component_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_id", _text(self.component_id, "component_id"))
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "component_type", str(self.component_type).strip())
        atoms = tuple(self.atoms)
        bonds = tuple(self.bonds)
        names = [atom.atom_id for atom in atoms]
        if len(set(names)) != len(names):
            raise ValueError("Chemical-component atom identifiers must be unique.")
        known = set(names)
        for bond in bonds:
            if bond.atom_id_1 not in known or bond.atom_id_2 not in known:
                raise ValueError(
                    "Chemical-component bonds must reference declared atoms."
                )
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "bonds", bonds)
        object.__setattr__(
            self, "parent_component_id", _optional_text(self.parent_component_id)
        )


@dataclass(frozen=True, slots=True)
class BondRecord:
    """Resolved covalent or coordination relation between coordinate observations."""

    atom_index_1: int
    atom_index_2: int
    order: BondOrder = BondOrder.UNKNOWN
    aromatic: bool = False
    connection_kind: ConnectionKind = ConnectionKind.COVALENT
    connection_id: str | None = None

    def __post_init__(self) -> None:
        first = int(self.atom_index_1)
        second = int(self.atom_index_2)
        if first < 0 or second < 0 or first == second:
            raise ValueError("Bond atom indices must be distinct and non-negative.")
        object.__setattr__(self, "atom_index_1", first)
        object.__setattr__(self, "atom_index_2", second)
        object.__setattr__(self, "order", BondOrder(self.order))
        object.__setattr__(self, "aromatic", bool(self.aromatic))
        object.__setattr__(self, "connection_kind", ConnectionKind(self.connection_kind))
        object.__setattr__(self, "connection_id", _optional_text(self.connection_id))


@dataclass(frozen=True, slots=True)
class MissingResidueRecord:
    chain_index: int
    label_comp_id: str
    label_seq_id: int | None
    auth_seq_id: int | None
    insertion_code: str | None = None
    model_number: int | None = None

    def __post_init__(self) -> None:
        if int(self.chain_index) < 0:
            raise ValueError("chain_index must be non-negative.")
        object.__setattr__(self, "chain_index", int(self.chain_index))
        object.__setattr__(
            self, "label_comp_id", _text(self.label_comp_id, "label_comp_id")
        )
        object.__setattr__(
            self,
            "label_seq_id",
            None if self.label_seq_id is None else int(self.label_seq_id),
        )
        object.__setattr__(
            self,
            "auth_seq_id",
            None if self.auth_seq_id is None else int(self.auth_seq_id),
        )
        object.__setattr__(
            self,
            "model_number",
            None if self.model_number is None else int(self.model_number),
        )
        object.__setattr__(self, "insertion_code", _optional_text(self.insertion_code))


@dataclass(frozen=True, slots=True)
class MissingAtomRecord:
    residue_index: int
    label_atom_id: str
    model_number: int | None = None

    def __post_init__(self) -> None:
        if int(self.residue_index) < 0:
            raise ValueError("residue_index must be non-negative.")
        object.__setattr__(self, "residue_index", int(self.residue_index))
        object.__setattr__(
            self, "label_atom_id", _text(self.label_atom_id, "label_atom_id")
        )
        object.__setattr__(
            self,
            "model_number",
            None if self.model_number is None else int(self.model_number),
        )


@dataclass(frozen=True, slots=True)
class AssemblyOperation:
    operation_id: str
    rotation: tuple[tuple[float, float, float], ...]
    translation: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _text(self.operation_id, "operation_id"))
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError(
                "Assembly operations require a (3, 3) rotation and 3-vector translation."
            )
        if np.any(~np.isfinite(rotation)) or np.any(~np.isfinite(translation)):
            raise ValueError("Assembly operations must be finite.")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=1e-6):
            raise ValueError("Assembly rotations must be orthogonal.")
        determinant = float(np.linalg.det(rotation))
        if not np.isclose(determinant, 1.0, atol=1e-6, rtol=1e-6):
            raise ValueError("Assembly rotations must be proper (determinant +1).")
        object.__setattr__(
            self, "rotation", tuple(tuple(float(v) for v in row) for row in rotation)
        )
        object.__setattr__(self, "translation", tuple(float(v) for v in translation))


@dataclass(frozen=True, slots=True)
class AssemblyGenerator:
    assembly_id: str
    operation_ids: tuple[str, ...]
    chain_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "assembly_id", _text(self.assembly_id, "assembly_id"))
        operations = tuple(_text(value, "operation_id") for value in self.operation_ids)
        chains = tuple(int(value) for value in self.chain_indices)
        if not operations or not chains or any(value < 0 for value in chains):
            raise ValueError(
                "Assembly generators require operations and non-negative chains."
            )
        object.__setattr__(self, "operation_ids", operations)
        object.__setattr__(self, "chain_indices", chains)


@dataclass(frozen=True, slots=True)
class MacromolecularRecord:
    """Lossless host-side macromolecular identity, chemistry, and coordinate record."""

    data_block: str
    entities: tuple[EntityRecord, ...]
    chains: tuple[ChainRecord, ...]
    residues: tuple[ResidueRecord, ...]
    atoms: tuple[AtomRecord, ...]
    chemical_components: tuple[ChemicalComponent, ...] = ()
    bonds: tuple[BondRecord, ...] = ()
    missing_residues: tuple[MissingResidueRecord, ...] = ()
    missing_atoms: tuple[MissingAtomRecord, ...] = ()
    assembly_operations: tuple[AssemblyOperation, ...] = ()
    assembly_generators: tuple[AssemblyGenerator, ...] = ()
    experimental_method: str | None = None
    resolution_angstrom: float | None = None
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_block", _text(self.data_block, "data_block"))
        object.__setattr__(self, "entities", tuple(self.entities))
        object.__setattr__(self, "chains", tuple(self.chains))
        object.__setattr__(self, "residues", tuple(self.residues))
        object.__setattr__(self, "atoms", tuple(self.atoms))
        object.__setattr__(self, "chemical_components", tuple(self.chemical_components))
        object.__setattr__(self, "bonds", tuple(self.bonds))
        object.__setattr__(self, "missing_residues", tuple(self.missing_residues))
        object.__setattr__(self, "missing_atoms", tuple(self.missing_atoms))
        object.__setattr__(self, "assembly_operations", tuple(self.assembly_operations))
        object.__setattr__(self, "assembly_generators", tuple(self.assembly_generators))
        if not self.atoms:
            raise ValueError(
                "A macromolecular record requires at least one atom observation."
            )
        entity_ids = [value.entity_id for value in self.entities]
        chain_ids = [value.label_asym_id for value in self.chains]
        component_ids = [value.component_id for value in self.chemical_components]
        operation_ids = [value.operation_id for value in self.assembly_operations]
        for name, values in (
            ("entity", entity_ids),
            ("label chain", chain_ids),
            ("chemical component", component_ids),
            ("assembly operation", operation_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} identifiers must be unique.")
        entity_set = set(entity_ids)
        if any(chain.entity_id not in entity_set for chain in self.chains):
            raise ValueError("Every chain must reference a declared entity.")
        if any(residue.chain_index >= len(self.chains) for residue in self.residues):
            raise ValueError("Every residue must reference a declared chain.")
        if any(atom.residue_index >= len(self.residues) for atom in self.atoms):
            raise ValueError("Every atom must reference a declared residue.")
        atom_count = len(self.atoms)
        if any(
            max(bond.atom_index_1, bond.atom_index_2) >= atom_count for bond in self.bonds
        ):
            raise ValueError("Every bond must reference a declared atom observation.")
        if any(value.residue_index >= len(self.residues) for value in self.missing_atoms):
            raise ValueError("Missing-atom records must reference declared residues.")
        if any(value.chain_index >= len(self.chains) for value in self.missing_residues):
            raise ValueError("Missing-residue records must reference declared chains.")
        operation_set = set(operation_ids)
        for generator in self.assembly_generators:
            if any(value not in operation_set for value in generator.operation_ids):
                raise ValueError(
                    "Assembly generators must reference declared operations."
                )
            if any(value >= len(self.chains) for value in generator.chain_indices):
                raise ValueError("Assembly generators must reference declared chains.")
        site_keys = [(atom.model_number, atom.atom_site_id) for atom in self.atoms]
        if len(site_keys) != len(set(site_keys)):
            raise ValueError("atom_site_id must be unique within each model.")
        identity_models = [(atom.model_number, atom.identity_key) for atom in self.atoms]
        if len(identity_models) != len(set(identity_models)):
            raise ValueError("An atom/altloc identity may occur at most once per model.")
        object.__setattr__(
            self, "experimental_method", _optional_text(self.experimental_method)
        )
        if self.resolution_angstrom is not None:
            resolution = float(self.resolution_angstrom)
            if not isfinite(resolution) or resolution <= 0.0:
                raise ValueError("resolution_angstrom must be finite and positive.")
            object.__setattr__(self, "resolution_angstrom", resolution)
        payload = {
            "kind": "macromolecular-record",
            "data_block": self.data_block,
            "entities": [
                (
                    v.entity_id,
                    v.kind.value,
                    v.description,
                    v.polymer_kind.value,
                    v.sequence_components,
                )
                for v in self.entities
            ],
            "chains": [
                (v.label_asym_id, v.auth_asym_id, v.entity_id) for v in self.chains
            ],
            "residues": [
                (
                    v.chain_index,
                    v.label_comp_id,
                    v.auth_comp_id,
                    v.label_seq_id,
                    v.auth_seq_id,
                    v.insertion_code,
                    v.entity_sequence_index,
                    v.hetero,
                )
                for v in self.residues
            ],
            "atoms": [
                (
                    v.atom_site_id,
                    v.residue_index,
                    v.model_number,
                    v.label_atom_id,
                    v.auth_atom_id,
                    v.element,
                    v.atomic_number,
                    v.position,
                    v.occupancy,
                    v.b_factor,
                    v.altloc_id,
                    v.formal_charge,
                    v.present,
                )
                for v in self.atoms
            ],
            "components": [
                (
                    v.component_id,
                    v.name,
                    v.component_type,
                    [
                        (
                            a.atom_id,
                            a.element,
                            a.atomic_number,
                            a.formal_charge,
                            a.aromatic,
                        )
                        for a in v.atoms
                    ],
                    [
                        (b.atom_id_1, b.atom_id_2, int(b.order), b.aromatic)
                        for b in v.bonds
                    ],
                    v.parent_component_id,
                )
                for v in self.chemical_components
            ],
            "bonds": [
                (
                    v.atom_index_1,
                    v.atom_index_2,
                    int(v.order),
                    v.aromatic,
                    v.connection_kind.value,
                    v.connection_id,
                )
                for v in self.bonds
            ],
            "missing_residues": [
                (
                    v.chain_index,
                    v.label_comp_id,
                    v.label_seq_id,
                    v.auth_seq_id,
                    v.insertion_code,
                    v.model_number,
                )
                for v in self.missing_residues
            ],
            "missing_atoms": [
                (v.residue_index, v.label_atom_id, v.model_number)
                for v in self.missing_atoms
            ],
            "operations": [
                (v.operation_id, v.rotation, v.translation)
                for v in self.assembly_operations
            ],
            "generators": [
                (v.assembly_id, v.operation_ids, v.chain_indices)
                for v in self.assembly_generators
            ],
            "experimental_method": self.experimental_method,
            "resolution": self.resolution_angstrom,
        }
        object.__setattr__(self, "record_id", canonical_fingerprint(payload))

    @property
    def model_numbers(self) -> tuple[int, ...]:
        return tuple(sorted({atom.model_number for atom in self.atoms}))

    def atoms_for_model(self, model_number: int) -> tuple[AtomRecord, ...]:
        return tuple(
            atom for atom in self.atoms if atom.model_number == int(model_number)
        )

    def component(self, component_id: str) -> ChemicalComponent | None:
        key = str(component_id)
        for component in self.chemical_components:
            if component.component_id == key:
                return component
        return None

    def altloc_ids(self, residue_index: int, model_number: int) -> tuple[str, ...]:
        values = {
            atom.altloc_id
            for atom in self.atoms
            if atom.residue_index == int(residue_index)
            and atom.model_number == int(model_number)
            and atom.altloc_id is not None
        }
        return tuple(sorted(value for value in values if value is not None))


def tuple_records(values: Iterable[object]) -> tuple[object, ...]:
    """Materialize a one-pass host record iterable explicitly."""

    return tuple(values)


__all__ = [
    "AssemblyGenerator",
    "AssemblyOperation",
    "AtomRecord",
    "BondRecord",
    "ChainRecord",
    "ChemicalComponent",
    "ChemicalComponentAtom",
    "ChemicalComponentBond",
    "EntityRecord",
    "MacromolecularRecord",
    "MissingAtomRecord",
    "MissingResidueRecord",
    "ResidueRecord",
]
