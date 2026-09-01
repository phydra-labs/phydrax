#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.bioinformatics.interchange._mmcif import dumps_mmcif, parse_mmcif
from phydrax.bioinformatics.rna._constraints import RNAConstraints, RNAFoldStatus
from phydrax.bioinformatics.rna._energy_model import nussinov_energy_model
from phydrax.bioinformatics.rna._mfe import minimum_free_energy
from phydrax.bioinformatics.rna._partition import partition_function
from phydrax.bioinformatics.rna._pseudoknot import (
    restricted_pseudoknot_fold,
    RestrictedPseudoknotPlan,
)
from phydrax.bioinformatics.rna._tertiary import (
    lower_tertiary_restraints,
    RNATertiaryRestraints,
)
from phydrax.bioinformatics.structure._alignment import align_coordinates
from phydrax.bioinformatics.structure._ensemble import analyze_structure_ensemble
from phydrax.bioinformatics.structure._interfaces import analyze_chain_interfaces
from phydrax.bioinformatics.structure._lowering import (
    lower_macromolecular_record,
    StructureLoweringPlan,
)
from phydrax.bioinformatics.structure._record import (
    AssemblyGenerator,
    AssemblyOperation,
    AtomRecord,
    BondRecord,
    ChainRecord,
    ChemicalComponent,
    ChemicalComponentAtom,
    ChemicalComponentBond,
    EntityRecord,
    MacromolecularRecord,
    MissingAtomRecord,
    MissingResidueRecord,
    ResidueRecord,
)
from phydrax.bioinformatics.structure._secondary import (
    ContactAnalysisPlan,
    residue_contacts,
)
from phydrax.bioinformatics.structure._types import (
    BondOrder,
    ConnectionKind,
    EntityKind,
    PolymerKind,
    StructureStatus,
)


def _altloc_record() -> MacromolecularRecord:
    entity = EntityRecord(
        "1",
        EntityKind.POLYMER,
        polymer_kind=PolymerKind.PROTEIN_L,
        sequence_components=("ALA",),
    )
    chain = ChainRecord("A", "X", "1")
    residue = ResidueRecord(0, "ALA", "ALA", 1, 7)
    atoms = (
        AtomRecord("1", 0, 1, "CA", "CA", "C", 6, (0.0, 0.0, 0.0)),
        AtomRecord("2", 0, 1, "CB", "CB", "C", 6, (1.0, 0.0, 0.0), 0.7, altloc_id="A"),
        AtomRecord("3", 0, 1, "CG", "CG", "C", 6, (2.0, 0.0, 0.0), 0.2, altloc_id="A"),
        AtomRecord("4", 0, 1, "CB", "CB", "C", 6, (0.0, 1.0, 0.0), 0.6, altloc_id="B"),
        AtomRecord("5", 0, 1, "CG", "CG", "C", 6, (0.0, 2.0, 0.0), 0.6, altloc_id="B"),
    )
    return MacromolecularRecord("alt", (entity,), (chain,), (residue,), atoms)


def _chemistry_record() -> MacromolecularRecord:
    entities = (
        EntityRecord(
            "1",
            EntityKind.POLYMER,
            polymer_kind=PolymerKind.PROTEIN_L,
            sequence_components=("MSE",),
        ),
        EntityRecord("2", EntityKind.NON_POLYMER),
    )
    chains = (ChainRecord("A", "AUTH", "1"), ChainRecord("L", "Z", "2"))
    residues = (
        ResidueRecord(0, "MSE", "MSE", 1, 10),
        ResidueRecord(1, "ZN", "ZN", None, 501, hetero=True),
    )
    components = (
        ChemicalComponent(
            "MSE",
            "selenomethionine",
            "peptide linking",
            (
                ChemicalComponentAtom("CA", "C", 6),
                ChemicalComponentAtom("SE", "SE", 34),
            ),
            (ChemicalComponentBond("CA", "SE", BondOrder.SINGLE),),
            "MET",
        ),
        ChemicalComponent(
            "ZN", "zinc", "non-polymer", (ChemicalComponentAtom("ZN", "ZN", 30, 2),)
        ),
    )
    atoms = (
        AtomRecord("1", 0, 1, "CA", "CA", "C", 6, (0.0, 0.0, 0.0), b_factor=11.0),
        AtomRecord("2", 0, 1, "SE", "SE", "SE", 34, (1.9, 0.0, 0.0), b_factor=12.0),
        AtomRecord("3", 1, 1, "ZN", "ZN", "ZN", 30, (3.8, 1.0, 0.0), formal_charge=2),
        AtomRecord("1", 0, 2, "CA", "CA", "C", 6, (1.0, 2.0, 3.0), b_factor=13.0),
        AtomRecord("2", 0, 2, "SE", "SE", "SE", 34, (2.9, 2.0, 3.0), b_factor=14.0),
        AtomRecord("3", 1, 2, "ZN", "ZN", "ZN", 30, (4.8, 3.0, 3.0), formal_charge=2),
    )
    bonds = (
        BondRecord(
            1,
            2,
            BondOrder.SINGLE,
            connection_kind=ConnectionKind.METAL_COORDINATION,
            connection_id="metal1",
        ),
    )
    operation = AssemblyOperation(
        "1", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (10.0, 0.0, 0.0)
    )
    return MacromolecularRecord(
        "chem",
        entities,
        chains,
        residues,
        atoms,
        components,
        bonds,
        (MissingResidueRecord(0, "GLY", 2, 11, model_number=1),),
        (MissingAtomRecord(0, "N", 1),),
        (operation,),
        (AssemblyGenerator("bio1", ("1",), (0, 1)),),
        "X-RAY DIFFRACTION",
        1.8,
    )


def test_altloc_selection_is_residue_coupled_not_atomwise() -> None:
    lowered = lower_macromolecular_record(_altloc_record())
    assert bool(lowered.valid)
    structure = lowered.structure
    assert structure is not None
    mask = np.asarray(structure.altloc_mask())
    # Shared CA plus conformer B: mean B occupancy 0.6 beats A mean 0.45,
    # even though CB-A individually has the largest occupancy.
    np.testing.assert_array_equal(mask, np.asarray([True, False, True, False, True]))


def test_lowering_preserves_modified_ligand_metal_links_models_missingness_assemblies() -> (
    None
):
    record = _chemistry_record()
    plan = StructureLoweringPlan.for_record(record)
    result = lower_macromolecular_record(record, plan)
    assert bool(result.valid)
    structure = result.structure
    assert structure is not None
    np.testing.assert_array_equal(structure.atomic_numbers, np.asarray([6, 34, 30]))
    assert structure.model_capacity == 2
    assert structure.bond_indices.shape == (
        2,
        2,
    )  # component bond plus struct_conn metal coordination
    assert structure.missing_residue_chain_indices.shape == (1,)
    assert structure.missing_atom_residue_indices.shape == (1,)
    assert structure.assembly_application_capacity == 2
    transformed, mask = structure.assembly_application(0)
    np.testing.assert_allclose(
        np.asarray(transformed[mask])[:, 0],
        np.asarray(structure.positions[0][mask])[:, 0] + 10.0,
    )
    assert result.atomistic_structure is not None
    assert result.atomistic_topology is not None


def test_lowering_preflights_capacity_and_rejects_unresolved_chemistry() -> None:
    record = _chemistry_record()
    undersized = StructureLoweringPlan(
        atom_capacity=2,
        residue_capacity=2,
        chain_capacity=2,
        model_capacity=2,
        bond_capacity=4,
        assembly_application_capacity=2,
        missing_residue_capacity=1,
        missing_atom_capacity=1,
    )
    overflow = lower_macromolecular_record(record, undersized)
    assert not bool(overflow.valid)
    assert int(overflow.status) == int(StructureStatus.CAPACITY_EXCEEDED)
    atom = AtomRecord("x", 0, 1, "X", "X", "X", 0, (0.0, 0.0, 0.0))
    unresolved = MacromolecularRecord(
        "unresolved",
        (EntityRecord("1", EntityKind.NON_POLYMER),),
        (ChainRecord("A", "A", "1"),),
        (ResidueRecord(0, "UNK", "UNK", None, 1, hetero=True),),
        (atom,),
    )
    rejected = lower_macromolecular_record(unresolved)
    assert int(rejected.status) == int(StructureStatus.UNRESOLVED_CHEMISTRY)


def test_mmcif_roundtrip_preserves_auth_label_altloc_models_links_and_assemblies() -> (
    None
):
    record = _chemistry_record()
    restored = parse_mmcif(dumps_mmcif(record))
    assert restored.chains[0].label_asym_id == "A"
    assert restored.chains[0].auth_asym_id == "AUTH"
    assert restored.residues[0].auth_seq_id == 10
    assert restored.model_numbers == (1, 2)
    assert restored.bonds[0].connection_kind is ConnectionKind.METAL_COORDINATION
    assert restored.chemical_components[0].parent_component_id == "MET"
    assert restored.assembly_generators[0].chain_indices == (0, 1)


def test_rigid_invariance_and_equivariance_of_structure_analyses() -> None:
    mobile = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    rotation = jnp.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = jnp.asarray([3.0, -2.0, 5.0])
    reference = mobile @ rotation.T + translation
    alignment = align_coordinates(mobile, reference)
    assert bool(alignment.valid)
    np.testing.assert_allclose(alignment.aligned, reference, atol=1e-6)
    np.testing.assert_allclose(alignment.rotation, rotation, atol=1e-6)
    np.testing.assert_allclose(alignment.rmsd, 0.0, atol=1e-6)

    structure = lower_macromolecular_record(_chemistry_record()).structure
    assert structure is not None
    contacts = residue_contacts(structure, ContactAnalysisPlan(2.2))
    interface = analyze_chain_interfaces(structure, cutoff=2.2)
    ensemble = analyze_structure_ensemble(structure)
    assert bool(contacts.valid)
    assert bool(interface.valid)
    assert bool(ensemble.valid)
    displacement = np.asarray(interface.centroid_displacements[0, 1])
    np.testing.assert_allclose(
        np.linalg.norm(displacement),
        np.asarray(interface.minimum_distances[0, 1]) + 0.05,
        atol=2.0,
    )


def _enumerate_structures(
    sequence: tuple[int, ...], allowed: np.ndarray, minimum_loop: int
) -> list[tuple[tuple[int, int], ...]]:
    def recurse(indices: tuple[int, ...]) -> list[tuple[tuple[int, int], ...]]:
        if not indices:
            return [()]
        first = indices[0]
        results = recurse(indices[1:])
        for offset, partner in enumerate(indices[1:], start=1):
            if (
                partner - first <= minimum_loop
                or not allowed[sequence[first], sequence[partner]]
            ):
                continue
            left = indices[1:offset]
            right = indices[offset + 1 :]
            for left_pairs, right_pairs in itertools.product(
                recurse(left), recurse(right)
            ):
                results.append(((first, partner), *left_pairs, *right_pairs))
        return results

    return recurse(tuple(range(len(sequence))))


def test_exact_rna_mfe_partition_and_inside_outside_match_tiny_enumeration() -> None:
    model = nussinov_energy_model(
        pair_energy=-1.3,
        wobble_energy=-0.4,
        unpaired_energy=0.2,
        minimum_hairpin_length=0,
    )
    sequence = (0, 3, 2, 1)  # A U G C
    structures = _enumerate_structures(sequence, np.asarray(model.allowed_pairs), 0)
    energies = []
    for pairs in structures:
        paired_bases = {index for pair in pairs for index in pair}
        energies.append(
            sum(float(model.pair_energies[sequence[i], sequence[j]]) for i, j in pairs)
            + sum(
                float(model.unpaired_energies[sequence[i]])
                for i in range(len(sequence))
                if i not in paired_bases
            )
        )
    mfe = minimum_free_energy(jnp.asarray(sequence), model)
    partition = partition_function(jnp.asarray(sequence), model)
    thermal = float(model.thermal_energy)
    weights = np.exp(-np.asarray(energies) / thermal)
    np.testing.assert_allclose(mfe.energy, np.min(energies), rtol=1e-6)
    np.testing.assert_allclose(
        partition.log_partition, np.log(np.sum(weights)), rtol=1e-6
    )
    for first in range(len(sequence)):
        for second in range(first + 1, len(sequence)):
            expected = sum(
                weight
                for pairs, weight in zip(structures, weights, strict=True)
                if (first, second) in pairs
            ) / np.sum(weights)
            np.testing.assert_allclose(
                partition.pair_marginals[first, second], expected, rtol=2e-6, atol=2e-7
            )
    np.testing.assert_allclose(
        partition.unpaired_marginals + jnp.sum(partition.pair_marginals, axis=1),
        1.0,
        atol=2e-6,
    )


def test_log_partition_gradient_is_pair_marginal_identity() -> None:
    model = nussinov_energy_model(pair_energy=-0.8, minimum_hairpin_length=0)
    sequence = jnp.asarray([0, 3, 0, 3])
    constraints = RNAConstraints.unconstrained(4)

    def objective(offsets: jax.Array) -> jax.Array:
        modified = eqx.tree_at(
            lambda value: value.pair_energy_offsets, constraints, offsets
        )
        return partition_function(sequence, model, modified).log_partition

    gradient = jax.grad(objective)(constraints.pair_energy_offsets)
    result = partition_function(sequence, model, constraints)
    expected = -jnp.triu(result.pair_marginals, 1) / model.thermal_energy
    np.testing.assert_allclose(gradient, expected, rtol=2e-5, atol=2e-6)


def test_temperature_constraint_fingerprints_and_unsupported_pseudoknots() -> None:
    model = nussinov_energy_model(minimum_hairpin_length=0)
    assert model.model_id != model.with_temperature(300.0).model_id
    free = RNAConstraints.unconstrained(7)
    required = np.full((7,), -2, dtype=np.int32)
    required[0] = 4
    required[4] = 0
    required[2] = 6
    required[6] = 2
    crossing = RNAConstraints(required)
    assert crossing.constraint_id != free.constraint_id
    folded = minimum_free_energy(jnp.asarray([0, 3, 0, 3, 3, 0, 3]), model, crossing)
    assert not bool(folded.valid)
    assert int(folded.status) == int(RNAFoldStatus.UNSUPPORTED_PSEUDOKNOT)
    heuristic = restricted_pseudoknot_fold(
        jnp.asarray([0, 3, 0, 3, 3, 0, 3]),
        model,
        RestrictedPseudoknotPlan(max_candidates=32, max_pairs=4, max_crossings=2),
        crossing,
    )
    assert bool(heuristic.valid)
    assert int(heuristic.status) == int(RNAFoldStatus.HEURISTIC_RESULT)


def test_rna_tertiary_restraints_lower_to_native_atomistic_constraints() -> None:
    structure = lower_macromolecular_record(_chemistry_record()).structure
    assert structure is not None
    restraints = RNATertiaryRestraints(
        np.asarray([[0, 1]], dtype=np.int32), np.asarray([4.0])
    )
    lowered = lower_tertiary_restraints(restraints, structure)
    assert bool(lowered.valid)
    assert lowered.topology is not None
    np.testing.assert_array_equal(lowered.topology.constraints, lowered.atom_pairs)
    np.testing.assert_allclose(lowered.topology.constraint_distances, np.asarray([4.0]))
