#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic._topology import MolecularTopologyPlan
from ...atomistic._types import AtomicStructure, AtomisticScaleContract
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._record import MacromolecularRecord
from ._topology import MacromolecularStructure
from ._types import ConnectionKind, StructureStatus


# IUPAC conventional atomic weights (or bracketed mass number for unstable elements)
# needed by biological structures. This is chemistry metadata, not a force field.
_ATOMIC_MASSES: dict[int, float] = {
    1: 1.008,
    2: 4.002602,
    3: 6.94,
    4: 9.0121831,
    5: 10.81,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    9: 18.998403163,
    10: 20.1797,
    11: 22.98976928,
    12: 24.305,
    13: 26.9815385,
    14: 28.085,
    15: 30.973761998,
    16: 32.06,
    17: 35.45,
    18: 39.948,
    19: 39.0983,
    20: 40.078,
    21: 44.955908,
    22: 47.867,
    23: 50.9415,
    24: 51.9961,
    25: 54.938044,
    26: 55.845,
    27: 58.933194,
    28: 58.6934,
    29: 63.546,
    30: 65.38,
    31: 69.723,
    32: 72.630,
    33: 74.921595,
    34: 78.971,
    35: 79.904,
    36: 83.798,
    37: 85.4678,
    38: 87.62,
    39: 88.90584,
    40: 91.224,
    41: 92.90637,
    42: 95.95,
    43: 98.0,
    44: 101.07,
    45: 102.90550,
    46: 106.42,
    47: 107.8682,
    48: 112.414,
    49: 114.818,
    50: 118.710,
    51: 121.760,
    52: 127.60,
    53: 126.90447,
    54: 131.293,
    55: 132.90545196,
    56: 137.327,
    57: 138.90547,
    58: 140.116,
    59: 140.90766,
    60: 144.242,
    62: 150.36,
    64: 157.25,
    78: 195.084,
    79: 196.966569,
    80: 200.592,
    82: 207.2,
    92: 238.02891,
}


class StructureLoweringPlan(StrictModule, NonTrainableState):
    """Explicit capacities and chemistry policy for host-to-numeric lowering."""

    atom_capacity: int = eqx.field(static=True)
    residue_capacity: int = eqx.field(static=True)
    chain_capacity: int = eqx.field(static=True)
    model_capacity: int = eqx.field(static=True)
    bond_capacity: int = eqx.field(static=True)
    assembly_application_capacity: int = eqx.field(static=True)
    missing_residue_capacity: int = eqx.field(static=True)
    missing_atom_capacity: int = eqx.field(static=True)
    strict_component_chemistry: bool = eqx.field(static=True)
    coordinate_dtype: str = eqx.field(static=True)
    method_contract: BioinformaticsMethodContract
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        atom_capacity: int,
        residue_capacity: int,
        chain_capacity: int,
        model_capacity: int,
        bond_capacity: int,
        assembly_application_capacity: int = 0,
        missing_residue_capacity: int = 0,
        missing_atom_capacity: int = 0,
        strict_component_chemistry: bool = True,
        coordinate_dtype: str = "float64",
    ):
        capacities = {
            "atom": int(atom_capacity),
            "residue": int(residue_capacity),
            "chain": int(chain_capacity),
            "model": int(model_capacity),
            "bond": int(bond_capacity),
            "assembly_application": int(assembly_application_capacity),
            "missing_residue": int(missing_residue_capacity),
            "missing_atom": int(missing_atom_capacity),
        }
        if any(value < 0 for value in capacities.values()) or any(
            capacities[name] == 0 for name in ("atom", "residue", "chain", "model")
        ):
            raise ValueError(
                "Core capacities must be positive and other capacities non-negative."
            )
        dtype = np.dtype(coordinate_dtype)
        if not np.issubdtype(dtype, np.inexact):
            raise TypeError("coordinate_dtype must be an inexact numeric dtype.")
        for name, value in capacities.items():
            setattr(self, f"{name}_capacity", value)
        self.strict_component_chemistry = bool(strict_component_chemistry)
        self.coordinate_dtype = dtype.name
        self.method_contract = BioinformaticsMethodContract(
            "macromolecular-record-lowering",
            MethodKind.EXACT_MODEL,
            ExecutionKind.FLOATING_POINT_DIRECT,
            DifferentiationKind.NONE,
            OutputKind.STRUCTURED,
            conditioning_statement=(
                "Host identity resolution is exact; coordinate values are copied "
                "in the declared dtype."
            ),
            truncation_statement="No truncation is permitted.",
            capacity_semantics=(
                "All atom, residue, chain, model, bond, missingness, and assembly "
                "counts are preflighted."
            ),
            assumptions=(
                "Atomic numbers and component chemistry are explicitly resolved.",
            ),
            nondifferentiable_outputs=("topology", "status", "evidence"),
            input_dtype="host-record",
            compute_dtype=dtype.name,
            output_dtype=dtype.name,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "structure-lowering-plan",
                "capacities": capacities,
                "strict_component_chemistry": self.strict_component_chemistry,
                "coordinate_dtype": self.coordinate_dtype,
                "contract": self.method_contract.contract_id,
            }
        )

    @classmethod
    def for_record(
        cls,
        record: MacromolecularRecord,
        /,
        *,
        strict_component_chemistry: bool = True,
        coordinate_dtype: str = "float64",
    ) -> "StructureLoweringPlan":
        identities = {atom.identity_key for atom in record.atoms}
        assembly_count = sum(
            len(generator.operation_ids) * len(generator.chain_indices)
            for generator in record.assembly_generators
        )
        # Component bonds expand over compatible alternate locations. Preflight
        # an upper bound before numeric allocation; the exact unique count is
        # checked after chemistry resolution.
        component_by_id = {
            component.component_id: component for component in record.chemical_components
        }
        identities_by_residue_name: dict[tuple[int, str], int] = {}
        for residue_index, atom_name, _ in identities:
            key = (residue_index, atom_name)
            identities_by_residue_name[key] = identities_by_residue_name.get(key, 0) + 1
        component_bound = 0
        for residue_index, residue in enumerate(record.residues):
            component = component_by_id.get(residue.label_comp_id)
            if component is None:
                continue
            for bond in component.bonds:
                component_bound += identities_by_residue_name.get(
                    (residue_index, bond.atom_id_1), 0
                ) * identities_by_residue_name.get((residue_index, bond.atom_id_2), 0)
        return cls(
            atom_capacity=len(identities),
            residue_capacity=len(record.residues),
            chain_capacity=len(record.chains),
            model_capacity=len(record.model_numbers),
            bond_capacity=len(record.bonds) + component_bound,
            assembly_application_capacity=assembly_count,
            missing_residue_capacity=len(record.missing_residues),
            missing_atom_capacity=len(record.missing_atoms),
            strict_component_chemistry=strict_component_chemistry,
            coordinate_dtype=coordinate_dtype,
        )


class StructureLoweringResult(StrictModule):
    """Audited all-or-nothing structure and atomistic lowering result."""

    structure: MacromolecularStructure | None
    atomistic_structure: AtomicStructure | None
    atomistic_topology: MolecularTopologyPlan | None
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    evidence_labels: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        structure: MacromolecularStructure | None,
        atomistic_structure: AtomicStructure | None,
        atomistic_topology: MolecularTopologyPlan | None,
        valid: bool,
        status: StructureStatus,
        evidence: np.ndarray,
        method_contract: BioinformaticsMethodContract,
    ):
        self.structure = structure
        self.atomistic_structure = atomistic_structure
        self.atomistic_topology = atomistic_topology
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(int(status), dtype=jnp.int32)
        self.evidence = jnp.asarray(evidence, dtype=jnp.int32)
        self.method_contract = method_contract
        self.evidence_labels = (
            "required_atoms",
            "required_residues",
            "required_chains",
            "required_models",
            "required_bonds",
            "required_assembly_applications",
            "required_missing_residues",
            "required_missing_atoms",
            "unresolved_atomic_numbers",
            "unresolved_component_atoms",
            "unresolved_bond_references",
        )


def _failure(
    plan: StructureLoweringPlan, status: StructureStatus, evidence: np.ndarray
) -> StructureLoweringResult:
    return StructureLoweringResult(
        None, None, None, False, status, evidence, plan.method_contract
    )


def lower_macromolecular_record(
    record: MacromolecularRecord,
    plan: StructureLoweringPlan | None = None,
    /,
    *,
    model_number: int | None = None,
    atomic_masses: Mapping[int, float] | None = None,
) -> StructureLoweringResult:
    """Strictly lower one host record; unresolved chemistry is never inferred."""

    if not isinstance(record, MacromolecularRecord):
        raise TypeError("record must be a MacromolecularRecord.")
    resolved_plan = StructureLoweringPlan.for_record(record) if plan is None else plan
    if not isinstance(resolved_plan, StructureLoweringPlan):
        raise TypeError("plan must be a StructureLoweringPlan.")
    identity_keys = sorted(
        {atom.identity_key for atom in record.atoms},
        key=lambda value: (value[0], value[1], "" if value[2] is None else value[2]),
    )
    model_numbers = record.model_numbers
    assembly_count = sum(
        len(generator.operation_ids) * len(generator.chain_indices)
        for generator in record.assembly_generators
    )
    required = np.asarray(
        [
            len(identity_keys),
            len(record.residues),
            len(record.chains),
            len(model_numbers),
            0,
            assembly_count,
            len(record.missing_residues),
            len(record.missing_atoms),
            0,
            0,
            0,
        ],
        dtype=np.int32,
    )
    capacities = np.asarray(
        [
            resolved_plan.atom_capacity,
            resolved_plan.residue_capacity,
            resolved_plan.chain_capacity,
            resolved_plan.model_capacity,
            resolved_plan.bond_capacity,
            resolved_plan.assembly_application_capacity,
            resolved_plan.missing_residue_capacity,
            resolved_plan.missing_atom_capacity,
        ],
        dtype=np.int32,
    )
    if np.any(required[:8] > capacities):
        return _failure(resolved_plan, StructureStatus.CAPACITY_EXCEEDED, required)

    atom_slot = {key: index for index, key in enumerate(identity_keys)}
    model_slot = {number: index for index, number in enumerate(model_numbers)}
    atom_count = len(identity_keys)
    model_count = len(model_numbers)
    dtype = np.dtype(resolved_plan.coordinate_dtype)
    positions = np.zeros((model_count, atom_count, 3), dtype=dtype)
    occupancies = np.zeros((model_count, atom_count), dtype=dtype)
    b_factors = np.zeros((model_count, atom_count), dtype=dtype)
    present = np.zeros((model_count, atom_count), dtype=bool)
    atomic_numbers = np.zeros((atom_count,), dtype=np.int32)
    formal_charges = np.zeros((atom_count,), dtype=np.int32)
    atom_to_residue = np.zeros((atom_count,), dtype=np.int32)

    atom_by_model_identity: dict[tuple[int, tuple[int, str, str | None]], int] = {}
    for record_index, atom in enumerate(record.atoms):
        slot = atom_slot[atom.identity_key]
        mslot = model_slot[atom.model_number]
        atom_by_model_identity[(atom.model_number, atom.identity_key)] = record_index
        if atomic_numbers[slot] not in (0, atom.atomic_number) or (
            atomic_numbers[slot] != 0 and formal_charges[slot] != atom.formal_charge
        ):
            required[8] += 1
            continue
        atomic_numbers[slot] = atom.atomic_number
        formal_charges[slot] = atom.formal_charge
        atom_to_residue[slot] = atom.residue_index
        positions[mslot, slot] = np.asarray(atom.position, dtype=dtype)
        occupancies[mslot, slot] = atom.occupancy
        b_factors[mslot, slot] = atom.b_factor
        present[mslot, slot] = atom.present
    unresolved_number = atomic_numbers <= 0
    required[8] += int(np.count_nonzero(unresolved_number))

    components = {
        component.component_id: component for component in record.chemical_components
    }
    for slot, (residue_index, atom_name, _) in enumerate(identity_keys):
        component = components.get(record.residues[residue_index].label_comp_id)
        if component is None or not component.atoms:
            continue
        matches = [atom for atom in component.atoms if atom.atom_id == atom_name]
        if (
            len(matches) != 1
            or matches[0].atomic_number <= 0
            or matches[0].atomic_number != atomic_numbers[slot]
        ):
            required[9] += 1
    if required[8] or (resolved_plan.strict_component_chemistry and required[9]):
        return _failure(resolved_plan, StructureStatus.UNRESOLVED_CHEMISTRY, required)

    residue_to_chain = np.asarray(
        [residue.chain_index for residue in record.residues], dtype=np.int32
    )
    entity_code = {
        identifier: index
        for index, identifier in enumerate(
            sorted(entity.entity_id for entity in record.entities)
        )
    }
    chain_to_entity = np.asarray(
        [entity_code[chain.entity_id] for chain in record.chains], dtype=np.int32
    )
    atom_names = sorted(
        {key[1] for key in identity_keys}
        | {value.label_atom_id for value in record.missing_atoms}
    )
    atom_name_code = {name: index for index, name in enumerate(atom_names)}
    atom_name_codes = np.asarray(
        [atom_name_code[key[1]] for key in identity_keys], dtype=np.int32
    )
    component_names = sorted({residue.label_comp_id for residue in record.residues})
    component_code = {name: index for index, name in enumerate(component_names)}
    residue_component_codes = np.asarray(
        [component_code[residue.label_comp_id] for residue in record.residues],
        dtype=np.int32,
    )

    alt_pairs = sorted(
        {(key[0], key[2]) for key in identity_keys if key[2] is not None},
        key=lambda value: (value[0], value[1]),
    )
    alt_choice_code = {pair: index + 1 for index, pair in enumerate(alt_pairs)}
    atom_altloc_choice = np.asarray(
        [
            0 if key[2] is None else alt_choice_code[(key[0], key[2])]
            for key in identity_keys
        ],
        dtype=np.int32,
    )
    altloc_choice_residue = np.asarray([value[0] for value in alt_pairs], dtype=np.int32)

    residue_anchor_atoms = np.full((len(record.residues),), -1, dtype=np.int32)
    anchor_priority = ("P", "C4'", "C4*", "CA", "C1'", "C1*")
    for residue_index in range(len(record.residues)):
        slots = [
            index for index, key in enumerate(identity_keys) if key[0] == residue_index
        ]
        shared = [index for index in slots if identity_keys[index][2] is None]
        candidates = shared or slots
        by_name = {identity_keys[index][1]: index for index in candidates}
        for name in anchor_priority:
            if name in by_name:
                residue_anchor_atoms[residue_index] = by_name[name]
                break
        if residue_anchor_atoms[residue_index] < 0 and candidates:
            residue_anchor_atoms[residue_index] = candidates[0]

    bond_map: dict[tuple[int, int], tuple[int, bool, int]] = {}

    def add_bond(first: int, second: int, order: int, aromatic: bool, kind: int) -> None:
        if first == second:
            return
        key = (min(first, second), max(first, second))
        value = (int(order), bool(aromatic), int(kind))
        if key in bond_map and bond_map[key] != value:
            required[10] += 1
        else:
            bond_map[key] = value

    connection_codes = {kind: index for index, kind in enumerate(ConnectionKind)}
    for bond in record.bonds:
        first_record = record.atoms[bond.atom_index_1]
        second_record = record.atoms[bond.atom_index_2]
        first = atom_slot[first_record.identity_key]
        second = atom_slot[second_record.identity_key]
        if (
            first_record.altloc_id is not None
            and second_record.altloc_id is not None
            and first_record.altloc_id != second_record.altloc_id
        ):
            required[10] += 1
            continue
        add_bond(
            first,
            second,
            int(bond.order),
            bond.aromatic,
            connection_codes[bond.connection_kind],
        )

    for residue_index, residue in enumerate(record.residues):
        component = components.get(residue.label_comp_id)
        if component is None:
            continue
        local = [key for key in identity_keys if key[0] == residue_index]
        by_name: dict[str, list[tuple[int, str, str | None]]] = {}
        for key in local:
            by_name.setdefault(key[1], []).append(key)
        for bond in component.bonds:
            first_keys = by_name.get(bond.atom_id_1, [])
            second_keys = by_name.get(bond.atom_id_2, [])
            for first_key in first_keys:
                for second_key in second_keys:
                    first_alt = first_key[2]
                    second_alt = second_key[2]
                    if (
                        first_alt is not None
                        and second_alt is not None
                        and first_alt != second_alt
                    ):
                        continue
                    add_bond(
                        atom_slot[first_key],
                        atom_slot[second_key],
                        int(bond.order),
                        bond.aromatic,
                        connection_codes[ConnectionKind.COVALENT],
                    )
    bonds = sorted(bond_map)
    required[4] = len(bonds)
    if required[10]:
        return _failure(resolved_plan, StructureStatus.UNRESOLVED_REFERENCE, required)
    if len(bonds) > resolved_plan.bond_capacity:
        return _failure(resolved_plan, StructureStatus.CAPACITY_EXCEEDED, required)
    bond_indices = np.asarray(bonds, dtype=np.int32).reshape((-1, 2))
    bond_orders = np.asarray([bond_map[key][0] for key in bonds], dtype=np.int32)
    bond_aromatic = np.asarray([bond_map[key][1] for key in bonds], dtype=bool)
    connection_kinds = np.asarray([bond_map[key][2] for key in bonds], dtype=np.int32)

    operation_index = {
        operation.operation_id: index
        for index, operation in enumerate(record.assembly_operations)
    }
    assembly_name_code = {
        name: index
        for index, name in enumerate(
            sorted({value.assembly_id for value in record.assembly_generators})
        )
    }
    assembly_ids: list[int] = []
    assembly_operations: list[int] = []
    assembly_chains: list[int] = []
    for generator in record.assembly_generators:
        for operation_id in generator.operation_ids:
            for chain_index in generator.chain_indices:
                assembly_ids.append(assembly_name_code[generator.assembly_id])
                assembly_operations.append(operation_index[operation_id])
                assembly_chains.append(chain_index)
    rotations = np.asarray(
        [operation.rotation for operation in record.assembly_operations], dtype=dtype
    ).reshape((-1, 3, 3))
    translations = np.asarray(
        [operation.translation for operation in record.assembly_operations], dtype=dtype
    ).reshape((-1, 3))

    missing_residue_chain = np.asarray(
        [value.chain_index for value in record.missing_residues], dtype=np.int32
    )
    missing_residue_label = np.asarray(
        [
            -1 if value.label_seq_id is None else value.label_seq_id
            for value in record.missing_residues
        ],
        dtype=np.int32,
    )
    missing_residue_auth = np.asarray(
        [
            -1 if value.auth_seq_id is None else value.auth_seq_id
            for value in record.missing_residues
        ],
        dtype=np.int32,
    )
    missing_residue_model = np.asarray(
        [
            0 if value.model_number is None else value.model_number
            for value in record.missing_residues
        ],
        dtype=np.int32,
    )
    missing_atom_residue = np.asarray(
        [value.residue_index for value in record.missing_atoms], dtype=np.int32
    )
    missing_atom_name = np.asarray(
        [atom_name_code[value.label_atom_id] for value in record.missing_atoms],
        dtype=np.int32,
    )
    missing_atom_model = np.asarray(
        [
            0 if value.model_number is None else value.model_number
            for value in record.missing_atoms
        ],
        dtype=np.int32,
    )

    structure = MacromolecularStructure(
        atomic_numbers,
        positions,
        present,
        atom_to_residue,
        residue_to_chain,
        chain_to_entity,
        formal_charges=formal_charges,
        occupancies=occupancies,
        b_factors=b_factors,
        model_numbers=np.asarray(model_numbers, dtype=np.int32),
        atom_name_codes=atom_name_codes,
        atom_altloc_choice=atom_altloc_choice,
        residue_component_codes=residue_component_codes,
        residue_anchor_atoms=residue_anchor_atoms,
        bond_indices=bond_indices,
        bond_orders=bond_orders,
        bond_aromatic=bond_aromatic,
        connection_kinds=connection_kinds,
        altloc_choice_residue=altloc_choice_residue,
        assembly_ids=np.asarray(assembly_ids, dtype=np.int32),
        assembly_operation_indices=np.asarray(assembly_operations, dtype=np.int32),
        assembly_chain_indices=np.asarray(assembly_chains, dtype=np.int32),
        assembly_rotations=rotations,
        assembly_translations=translations,
        missing_residue_chain_indices=missing_residue_chain,
        missing_residue_label_seq_ids=missing_residue_label,
        missing_residue_auth_seq_ids=missing_residue_auth,
        missing_residue_model_numbers=missing_residue_model,
        missing_atom_residue_indices=missing_atom_residue,
        missing_atom_name_codes=missing_atom_name,
        missing_atom_model_numbers=missing_atom_model,
        source_record_id=record.record_id,
    )

    selected_model_number = (
        model_numbers[0] if model_number is None else int(model_number)
    )
    if selected_model_number not in model_slot:
        return _failure(resolved_plan, StructureStatus.NO_VALID_MODEL, required)
    selected_model = model_slot[selected_model_number]
    selected = np.asarray(structure.altloc_mask(selected_model), dtype=bool)
    if not np.any(selected):
        return _failure(resolved_plan, StructureStatus.NO_VALID_MODEL, required)
    mass_table = dict(_ATOMIC_MASSES)
    if atomic_masses is not None:
        for number, mass in atomic_masses.items():
            number_ = int(number)
            mass_ = float(mass)
            if number_ <= 0 or not np.isfinite(mass_) or mass_ <= 0.0:
                raise ValueError(
                    "atomic_masses must map positive atomic numbers to positive finite masses."
                )
            mass_table[number_] = mass_
    unresolved_mass = sorted(
        {int(value) for value in atomic_numbers[selected] if int(value) not in mass_table}
    )
    if unresolved_mass:
        required[8] += len(unresolved_mass)
        return _failure(resolved_plan, StructureStatus.UNRESOLVED_CHEMISTRY, required)
    active_numbers = np.where(selected, atomic_numbers, 0)
    masses = np.asarray(
        [
            mass_table[int(number)] if active else 0.0
            for number, active in zip(atomic_numbers, selected, strict=True)
        ],
        dtype=dtype,
    )
    particle_ids = np.arange(atom_count, dtype=np.int64)
    atomistic = AtomicStructure(
        active_numbers,
        positions[selected_model],
        masses,
        AtomisticScaleContract("angstrom", "kilojoule_per_mole"),
        particle_ids=particle_ids,
        active_mask=selected,
        name=record.data_block,
        coordinate_dtype=dtype,
    )
    selected_bonds = (
        bond_indices[selected[bond_indices[:, 0]] & selected[bond_indices[:, 1]]]
        if bond_indices.size
        else bond_indices
    )
    topology = MolecularTopologyPlan(
        bonds=particle_ids[selected_bonds] if selected_bonds.size else selected_bonds
    )
    return StructureLoweringResult(
        structure,
        atomistic,
        topology,
        True,
        StructureStatus.SUCCESS,
        required,
        resolved_plan.method_contract,
    )


__all__ = [
    "StructureLoweringPlan",
    "StructureLoweringResult",
    "lower_macromolecular_record",
]
