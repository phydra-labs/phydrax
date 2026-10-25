#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Molecular crystal input identity, stereo, atom types and correspondence.

Controls: exact automorphism orbits against the pinned Toolkit's documented
ethanol types and ``for_connectivity`` example, hand-derived stereo parity
transport for chiral, meso and C2 diastereomers, Kekulé-resolved valence
admission, spin-parity refusal, bounded correspondence statuses, mirror
correspondences of enantiomers, topology-rule contact reductions and
formula-unit volume/contact bookkeeping.
"""

from math import cos, radians, sin

import numpy as np
import pytest

from phydrax.atomistic.crystal._atom_types import (
    atom_type_assignment,
    map_connectivity_types,
    molecular_correspondences,
)
from phydrax.atomistic.crystal._input import (
    AROMATIC_BOND,
    ConformerPool,
    ContactPolicy,
    ElementTable,
    MolecularComponent,
    MolecularConformer,
    MolecularCrystalInput,
    stereo_realization,
    VolumePolicy,
)


MASSES = ElementTable(
    "mass",
    {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 17: 35.45, 35: 79.904},
    provenance="CIAAW 2021 conventional standard atomic weights",
)
BONDI = ElementTable(
    "contact-radius",
    {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 17: 1.75, 35: 1.85},
    provenance="Bondi, J. Phys. Chem. 68 (1964) 441",
)

ETHANOL_NUMBERS = (6, 1, 1, 1, 6, 1, 1, 8, 1)
ETHANOL_BONDS = ((0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 7), (7, 8))


def ethanol() -> MolecularComponent:
    return MolecularComponent(
        ETHANOL_NUMBERS,
        ETHANOL_BONDS,
        bond_orders=(1,) * 8,
        masses=MASSES,
        spin_multiplicity=1,
        name="ethanol",
    )


def halomethane(sign: int = 1) -> MolecularComponent:
    """CHFClBr: C0 bonded to H1, F2, Cl3, Br4."""
    return MolecularComponent(
        (6, 1, 9, 17, 35),
        ((0, 1), (0, 2), (0, 3), (0, 4)),
        bond_orders=(1, 1, 1, 1),
        masses=MASSES,
        spin_multiplicity=1,
        tetrahedral_stereo=((0, 1, 2, 3, sign),),
        name="bromochlorofluoromethane",
    )


def halomethane_positions() -> np.ndarray:
    """H on +z; F, Cl, Br anticlockwise seen from H, so det(H, F, Cl) > 0."""
    theta = radians(109.4712206)
    lengths = (1.35, 1.77, 1.94)
    positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.09)]
    for length, phi in zip(lengths, (0.0, 120.0, 240.0), strict=True):
        positions.append(
            (
                length * sin(theta) * cos(radians(phi)),
                length * sin(theta) * sin(radians(phi)),
                length * cos(theta),
            )
        )
    return np.asarray(positions)


def dihalo_ethane(first: int, second: int) -> MolecularComponent:
    """CHFCl-CHFCl: C0(H2, F3, Cl4), C1(H5, F6, Cl7)."""
    return MolecularComponent(
        (6, 6, 1, 9, 17, 1, 9, 17),
        ((0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)),
        bond_orders=(1,) * 7,
        masses=MASSES,
        spin_multiplicity=1,
        tetrahedral_stereo=((0, 1, 2, 3, first), (1, 0, 5, 6, second)),
    )


def dichloroethene(sign: int) -> MolecularComponent:
    """ClHC=CHCl: C0(Cl2, H3)=C1(Cl4, H5)."""
    return MolecularComponent(
        (6, 6, 17, 1, 17, 1),
        ((0, 1), (0, 2), (0, 3), (1, 4), (1, 5)),
        bond_orders=(2, 1, 1, 1, 1),
        masses=MASSES,
        spin_multiplicity=1,
        double_bond_stereo=((2, 0, 1, 4, sign),),
    )


def benzene() -> MolecularComponent:
    ring = tuple((index, (index + 1) % 6) for index in range(6))
    hydrogens = tuple((index, index + 6) for index in range(6))
    return MolecularComponent(
        (6,) * 6 + (1,) * 6,
        ring + hydrogens,
        bond_orders=(4,) * 6 + (1,) * 6,
        masses=MASSES,
        spin_multiplicity=1,
        name="benzene",
    )


def pyrrole(*, with_nh: bool) -> MolecularComponent:
    ring = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0))
    hydrogens = ((1, 5), (2, 6), (3, 7), (4, 8)) + (((0, 9),) if with_nh else ())
    return MolecularComponent(
        (7, 6, 6, 6, 6) + (1,) * len(hydrogens),
        ring + hydrogens,
        bond_orders=(4,) * 5 + (1,) * len(hydrogens),
        masses=MASSES,
        # Removing N-H leaves an odd-electron neutral candidate; keep spin
        # parity admissible so the negative case reaches aromatic valence.
        spin_multiplicity=1 if with_nh else 2,
    )


def water() -> MolecularComponent:
    return MolecularComponent(
        (8, 1, 1),
        ((0, 1), (0, 2)),
        bond_orders=(1, 1),
        masses=MASSES,
        spin_multiplicity=1,
        name="water",
    )


def water_positions() -> np.ndarray:
    angle = radians(104.52)
    return np.asarray(
        [
            (0.0, 0.0, 0.0),
            (0.9572, 0.0, 0.0),
            (0.9572 * cos(angle), 0.9572 * sin(angle), 0.0),
        ]
    )


def test_ethanol_orbits_match_the_toolkit_documented_types() -> None:
    component = ethanol()
    assert component.atom_classes == (0, 1, 1, 1, 2, 3, 3, 4, 5)
    assert component.automorphisms_complete
    assert len(component.automorphisms) == 3 * 2 * 1 * 2
    assert component.automorphisms[0] == tuple(range(9))
    for permutation in component.automorphisms:
        mapped = {
            tuple(sorted((permutation[a], permutation[b]))) for a, b in ETHANOL_BONDS
        }
        assert mapped == set(ETHANOL_BONDS)
    assert not component.stereo_chiral
    assert component.mirror_automorphism == tuple(range(9))


def test_component_identity_ignores_display_name_and_stereo_spelling() -> None:
    reference = halomethane()
    # Odd permutation of the declared neighbors with the opposite sign is the
    # same configuration.
    respelled = MolecularComponent(
        (6, 1, 9, 17, 35),
        ((0, 1), (0, 2), (0, 3), (0, 4)),
        bond_orders=(1, 1, 1, 1),
        masses=MASSES,
        spin_multiplicity=1,
        tetrahedral_stereo=((0, 2, 1, 3, -1),),
        name="renamed",
    )
    assert respelled.component_id == reference.component_id
    assert respelled.tetrahedral_stereo == reference.tetrahedral_stereo
    # Replacing the third listed neighbor by the implied fourth is an odd
    # permutation of the four-neighbor tuple.
    implied = MolecularComponent(
        (6, 1, 9, 17, 35),
        ((0, 1), (0, 2), (0, 3), (0, 4)),
        bond_orders=(1, 1, 1, 1),
        masses=MASSES,
        spin_multiplicity=1,
        tetrahedral_stereo=((0, 1, 2, 4, -1),),
    )
    assert implied.component_id == reference.component_id
    assert halomethane(-1).component_id != reference.component_id


def test_undeclared_stereocenter_is_refused_or_explicitly_unspecified() -> None:
    numbers, bonds = (6, 1, 9, 17, 35), ((0, 1), (0, 2), (0, 3), (0, 4))
    with pytest.raises(ValueError, match="Undeclared stereo"):
        MolecularComponent(
            numbers, bonds, bond_orders=(1, 1, 1, 1), masses=MASSES, spin_multiplicity=1
        )
    unspecified = MolecularComponent(
        numbers,
        bonds,
        bond_orders=(1, 1, 1, 1),
        masses=MASSES,
        spin_multiplicity=1,
        undeclared_stereo="unspecified",
    )
    assert unspecified.unspecified_tetrahedral == (0,)
    assert not unspecified.stereo_chiral
    assert unspecified.component_id != halomethane().component_id


def test_chiral_component_has_a_distinct_enantiomer_related_by_mirror_only() -> None:
    right = halomethane(1)
    left = right.enantiomer()
    assert right.stereo_chiral and left.stereo_chiral
    assert right.mirror_automorphism is None
    assert left.component_id == halomethane(-1).component_id
    proper = molecular_correspondences(right, left, max_mappings=8, max_nodes=10_000)
    assert proper.status == "incompatible" and proper.complete
    improper = molecular_correspondences(
        right, left, max_mappings=8, max_nodes=10_000, mirror=True
    )
    assert improper.status == "complete"
    assert improper.mappings == (tuple(range(5)),)
    chemical = molecular_correspondences(
        right, left, max_mappings=8, max_nodes=10_000, mode="chemical"
    )
    assert chemical.mappings == (tuple(range(5)),)
    with pytest.raises(ValueError, match="achiral"):
        ethanol().enantiomer()


def test_meso_and_c2_diastereomers_follow_exact_parity_transport() -> None:
    swap = (1, 0, 5, 6, 7, 2, 3, 4)
    meso = dihalo_ethane(1, -1)
    assert not meso.stereo_chiral
    assert meso.mirror_automorphism == swap
    assert meso.automorphisms == (tuple(range(8)),)
    chiral = dihalo_ethane(1, 1)
    assert chiral.stereo_chiral
    assert chiral.mirror_automorphism is None
    assert chiral.automorphisms == (tuple(range(8)), swap)
    assert chiral.atom_classes == (0, 0, 1, 2, 3, 1, 2, 3)
    assert meso.atom_classes == tuple(range(8))


def test_double_bond_stereo_is_identity_and_refused_when_undeclared() -> None:
    trans, cis = dichloroethene(-1), dichloroethene(1)
    assert trans.component_id != cis.component_id
    assert trans.double_bond_stereo == ((2, 0, 1, 4, -1),)
    assert (1, 0, 4, 5, 2, 3) in trans.automorphisms
    assert not trans.stereo_chiral
    # The same configuration declared through the other substituents.
    respelled = MolecularComponent(
        (6, 6, 17, 1, 17, 1),
        ((0, 1), (0, 2), (0, 3), (1, 4), (1, 5)),
        bond_orders=(2, 1, 1, 1, 1),
        masses=MASSES,
        spin_multiplicity=1,
        double_bond_stereo=((3, 0, 1, 4, 1),),
    )
    assert respelled.component_id == trans.component_id
    with pytest.raises(ValueError, match="Undeclared stereo"):
        MolecularComponent(
            (6, 6, 17, 1, 17, 1),
            ((0, 1), (0, 2), (0, 3), (1, 4), (1, 5)),
            bond_orders=(2, 1, 1, 1, 1),
            masses=MASSES,
            spin_multiplicity=1,
        )


def test_aromatic_valence_requires_a_kekule_structure() -> None:
    ring = benzene()
    aromatic_orders = [
        resolved
        for declared, resolved in zip(
            ring.bond_orders, ring.kekule_bond_orders, strict=True
        )
        if declared == AROMATIC_BOND
    ]
    assert sorted(aromatic_orders) == [1, 1, 1, 2, 2, 2]
    assert len(ring.automorphisms) == 12
    assert ring.atom_classes == (0,) * 6 + (1,) * 6
    assert pyrrole(with_nh=True).kekule_bond_orders.count(2) == 2
    with pytest.raises(ValueError):
        pyrrole(with_nh=False)


def test_missing_hydrogens_and_inconsistent_spin_are_refused() -> None:
    with pytest.raises(ValueError, match="valence"):
        MolecularComponent(
            (6, 1, 1, 1),
            ((0, 1), (0, 2), (0, 3)),
            bond_orders=(1, 1, 1),
            masses=MASSES,
            spin_multiplicity=2,
        )
    triplet = MolecularComponent(
        (8, 8), ((0, 1),), bond_orders=(2,), masses=MASSES, spin_multiplicity=3
    )
    assert triplet.spin_multiplicity == 3
    with pytest.raises(ValueError, match="Spin multiplicity"):
        MolecularComponent(
            (8, 8), ((0, 1),), bond_orders=(2,), masses=MASSES, spin_multiplicity=2
        )
    with pytest.raises(ValueError, match="connected"):
        MolecularComponent(
            (8, 1, 1, 8, 1, 1),
            ((0, 1), (0, 2), (3, 4), (3, 5)),
            bond_orders=(1, 1, 1, 1),
            masses=MASSES,
            spin_multiplicity=1,
        )
    with pytest.raises(ValueError, match="no data"):
        MolecularComponent(
            (14, 1, 1, 1, 1),
            ((0, 1), (0, 2), (0, 3), (0, 4)),
            bond_orders=(1, 1, 1, 1),
            masses=MASSES,
            spin_multiplicity=1,
        )


def test_bounded_correspondence_statuses() -> None:
    reference = ethanol()
    order = (7, 8, 4, 5, 6, 0, 1, 2, 3)
    new_index = {old: new for new, old in enumerate(order)}
    permuted = MolecularComponent(
        tuple(ETHANOL_NUMBERS[old] for old in order),
        tuple((new_index[a], new_index[b]) for a, b in ETHANOL_BONDS),
        bond_orders=(1,) * 8,
        masses=MASSES,
        spin_multiplicity=1,
    )
    complete = molecular_correspondences(
        reference, permuted, max_mappings=64, max_nodes=100_000
    )
    assert complete.status == "complete" and complete.complete
    assert len(complete.mappings) == 12
    assert list(complete.mappings) == sorted(complete.mappings)
    for mapping in complete.mappings:
        assert mapping[7] == 0 and mapping[8] == 1 and mapping[4] == 2
        assert all(
            permuted.atomic_numbers[mapping[atom]] == ETHANOL_NUMBERS[atom]
            for atom in range(9)
        )
    limited = molecular_correspondences(
        reference, permuted, max_mappings=1, max_nodes=100_000
    )
    assert limited.status == "mapping-limit" and not limited.complete
    assert len(limited.mappings) == 1
    exhausted = molecular_correspondences(
        reference, permuted, max_mappings=64, max_nodes=2
    )
    assert exhausted.status == "node-limit" and not exhausted.complete
    ether = MolecularComponent(
        (6, 8, 6, 1, 1, 1, 1, 1, 1),
        ((0, 1), (1, 2), (0, 3), (0, 4), (0, 5), (2, 6), (2, 7), (2, 8)),
        bond_orders=(1,) * 8,
        masses=MASSES,
        spin_multiplicity=1,
    )
    distinct = molecular_correspondences(
        reference, ether, max_mappings=4, max_nodes=100_000
    )
    assert distinct.status == "incompatible" and distinct.mappings == ()
    with pytest.raises(ValueError, match="Mirror"):
        molecular_correspondences(
            reference,
            permuted,
            max_mappings=1,
            max_nodes=10,
            mode="chemical",
            mirror=True,
        )


def test_formula_unit_atom_types_respect_the_declared_mode() -> None:
    right = halomethane(1)
    left = right.enantiomer()
    stereo = atom_type_assignment((right, left), (1, 1), mode="stereo")
    assert stereo.type_count == 10
    chemical = atom_type_assignment((right, left), (1, 1), mode="chemical")
    assert chemical.type_count == 5
    assert chemical.component_type_ids == ((0, 1, 2, 3, 4), (0, 1, 2, 3, 4))
    repeated = atom_type_assignment((ethanol(),), (2,), mode="connectivity")
    assert repeated.type_ids == (0, 1, 1, 1, 2, 3, 3, 4, 5) * 2
    assert repeated.representatives == ((0, 0), (0, 1), (0, 4), (0, 5), (0, 7), (0, 8))


def test_connectivity_types_map_a_reordered_graph_like_the_toolkit_example() -> None:
    reference = MolecularComponent(
        (6, 6, 8),
        ((0, 1), (1, 2)),
        bond_orders=(1, 1),
        masses=MASSES,
        spin_multiplicity=1,
        valence="unchecked",
    )
    assignment = atom_type_assignment((reference,), (1,), mode="connectivity")
    assert map_connectivity_types(
        assignment, (reference,), (8, 6, 6), ((0, 1), (1, 2))
    ) == (
        2,
        1,
        0,
    )
    with pytest.raises(ValueError, match="no isomorphic"):
        map_connectivity_types(assignment, (reference,), (8, 8, 6), ((0, 1), (1, 2)))


def test_input_conformer_is_admitted_in_a_proper_principal_frame() -> None:
    component = halomethane()
    positions = halomethane_positions() + np.asarray([3.0, -2.0, 1.0])
    conformer = MolecularConformer(
        component, positions, provenance="hand-built tetrahedron"
    )
    local = np.asarray(conformer.positions)
    masses = np.asarray(component.masses)
    np.testing.assert_allclose(masses @ local, 0.0, atol=1e-10)
    inertia = (
        np.eye(3) * np.sum(masses * np.sum(local**2, axis=1)) - (local.T * masses) @ local
    )
    np.testing.assert_allclose(
        inertia, np.diag(np.asarray(conformer.principal_moments)), atol=1e-8
    )
    assert np.all(np.diff(np.asarray(conformer.principal_moments)) >= 0.0)

    def distances(values: np.ndarray) -> np.ndarray:
        return np.linalg.norm(values[:, None] - values[None, :], axis=-1)

    np.testing.assert_allclose(distances(local), distances(positions), atol=1e-10)
    assert stereo_realization(component, local).tetrahedral_volumes[0] > 0.0
    mirrored = positions * np.asarray([-1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="tetrahedral stereo"):
        MolecularConformer(component, mirrored, provenance="mirror image")
    collapsed = positions.copy()
    collapsed[1] = collapsed[0] + 0.1
    with pytest.raises(ValueError, match="minimum_distance"):
        MolecularConformer(component, collapsed, provenance="collapsed")


def test_contact_rules_and_formula_unit_bookkeeping() -> None:
    component = water()
    conformer = MolecularConformer(
        component, water_positions(), provenance="gas-phase water"
    )
    pool = ConformerPool(component, (conformer,))
    reduced = MolecularCrystalInput(
        (pool,),
        contact=ContactPolicy(BONDI, rule="donor-acceptor-reduced"),
        volume=VolumePolicy(
            atomic_volumes=ElementTable(
                "atomic-volume", {1: 5.08, 8: 8.71}, provenance="test-declared volumes"
            )
        ),
        formula_counts=(2,),
    )
    assert reduced.formula_atomic_numbers == (8, 1, 1, 8, 1, 1)
    assert reduced.formula_atom_sources == (
        (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2)
    )  # fmt: skip
    assert reduced.formula_unit_volume == pytest.approx(2 * (8.71 + 2 * 5.08))
    contacts = np.asarray(reduced.contact_distances)
    # Hydroxyl H donor 0.55 plus neutral divalent O acceptor 0.15.
    assert contacts[1, 3] == pytest.approx(1.20 + 1.52 - 0.70)
    assert contacts[0, 3] == pytest.approx(2 * 1.52)
    assert contacts[1, 4] == pytest.approx(2 * 1.20)
    np.testing.assert_array_equal(contacts, contacts.T)
    plain = MolecularCrystalInput(
        (pool,),
        contact=ContactPolicy(BONDI),
        volume=VolumePolicy(formula_unit_volume=30.0),
    )
    assert np.asarray(plain.contact_distances)[0, 1] == pytest.approx(2.72)
    assert plain.total_charge == 0
    with pytest.raises(ValueError, match="symmetric"):
        MolecularCrystalInput(
            (pool,),
            contact=np.asarray([[3.0, 2.0, 2.0], [2.5, 2.4, 2.4], [2.0, 2.4, 2.4]]),
            volume=VolumePolicy(formula_unit_volume=30.0),
        )
    with pytest.raises(ValueError, match="no data"):
        ContactPolicy(
            ElementTable("contact-radius", {1: 1.2}, provenance="hydrogen only")
        ).distances((component,))
    with pytest.raises(ValueError, match="exactly one"):
        VolumePolicy()
