#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native distance-geometry conformers: geometry, stereo, rings, rank and addressing.

Controls: generated bond lengths against the admitted radius table, realized
signed chiral volumes and double-bond cosines, aromatic ring planarity by an
independent SVD plane fit, triangle-infeasible ring/stereo declarations,
metric-embedding rank and indefiniteness on exactly known distance matrices,
ordinal-addressed reproducibility across split and resumed ranges, complete
failed-attempt evidence, energetic cleanup by a native potential and
symmetry-aware RMSD pruning.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.atomistic.crystal._conformers import (
    conformer_rmsd,
    ConformerEnergyCleanup,
    ConformerGenerationPlan,
    ConformerGeometryTable,
    ConformerQualification,
    metric_embedding,
    prepare_conformer_constraints,
    prune_conformers,
    qualify_geometry,
)
from phydrax.atomistic.crystal._input import (
    ElementTable,
    MolecularComponent,
    MolecularConformer,
    stereo_realization,
)
from phydrax.optim import FIRE, OptimizationStatus, OptimizationTermination


MASSES = ElementTable(
    "mass",
    {1: 1.008, 6: 12.011, 8: 15.999, 9: 18.998, 17: 35.45, 35: 79.904},
    provenance="CIAAW 2021 conventional standard atomic weights",
)
PYYKKO = "Pyykkö and Atsumi, Chem. Eur. J. 15 (2009) 186 and 12770"
TABLE = ConformerGeometryTable(
    ElementTable(
        "covalent-radius",
        {1: 0.32, 6: 0.75, 8: 0.63, 9: 0.64, 17: 0.99, 35: 1.14},
        provenance=PYYKKO,
    ),
    ElementTable(
        "contact-radius",
        {1: 1.20, 6: 1.70, 8: 1.52, 9: 1.47, 17: 1.75, 35: 1.85},
        provenance="Bondi, J. Phys. Chem. 68 (1964) 441",
    ),
    double_radii=ElementTable(
        "double-bond-radius", {6: 0.67, 8: 0.57}, provenance=PYYKKO
    ),
)


def ethanol() -> MolecularComponent:
    return MolecularComponent(
        (6, 1, 1, 1, 6, 1, 1, 8, 1),
        ((0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 7), (7, 8)),
        bond_orders=(1,) * 8,
        masses=MASSES,
        spin_multiplicity=1,
    )


def halomethane(sign: int = 1) -> MolecularComponent:
    return MolecularComponent(
        (6, 1, 9, 17, 35),
        ((0, 1), (0, 2), (0, 3), (0, 4)),
        bond_orders=(1, 1, 1, 1),
        masses=MASSES,
        spin_multiplicity=1,
        tetrahedral_stereo=((0, 1, 2, 3, sign),),
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
    )


def dichloroethene(sign: int) -> MolecularComponent:
    return MolecularComponent(
        (6, 6, 17, 1, 17, 1),
        ((0, 1), (0, 2), (0, 3), (1, 4), (1, 5)),
        bond_orders=(2, 1, 1, 1, 1),
        masses=MASSES,
        spin_multiplicity=1,
        double_bond_stereo=((2, 0, 1, 4, sign),),
    )


def cyclohexene(sign: int) -> MolecularComponent:
    """C0=C1 in a six-membered ring; ``sign`` relates ring neighbors C5 and C2."""
    ring = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0))
    hydrogens = (
        (0, 6), (1, 7), (2, 8), (2, 9), (3, 10), (3, 11), (4, 12), (4, 13), (5, 14), (5, 15)
    )  # fmt: skip
    return MolecularComponent(
        (6,) * 6 + (1,) * 10,
        ring + hydrogens,
        bond_orders=(2,) + (1,) * 5 + (1,) * 10,
        masses=MASSES,
        spin_multiplicity=1,
        double_bond_stereo=((5, 0, 1, 2, sign),),
    )


def plan(component: MolecularComponent) -> ConformerGenerationPlan:
    return ConformerGenerationPlan(
        component, prepare_conformer_constraints(component, TABLE), max_attempts=16
    )


def bond_lengths(component: MolecularComponent, positions: np.ndarray) -> np.ndarray:
    bonds = np.asarray(component.bonds)
    return np.linalg.norm(positions[bonds[:, 0]] - positions[bonds[:, 1]], axis=-1)


def test_constraints_are_consistent_and_carry_stereo_and_planarity() -> None:
    constraints = prepare_conformer_constraints(benzene(), TABLE)
    assert constraints.consistent
    assert constraints.local_geometries == ("trigonal",) * 6 + ("terminal",) * 6
    # Six trigonal centers plus four (substituent, end) quads per ring bond.
    assert constraints.planar_atoms.shape == (6 + 6 * 4, 4)
    assert constraints.required_rank == 2
    lower, upper = np.asarray(constraints.lower), np.asarray(constraints.upper)
    assert np.all(lower <= upper + 1e-9)
    np.testing.assert_allclose(lower, lower.T)
    aromatic = 0.5 * ((0.75 + 0.75) + (0.67 + 0.67))
    assert aromatic - TABLE.bond_tolerance - 1e-12 <= lower[0, 1]
    assert lower[0, 1] <= upper[0, 1] <= aromatic + TABLE.bond_tolerance + 1e-12
    chiral = prepare_conformer_constraints(halomethane(), TABLE)
    assert chiral.chiral_atoms.shape == (1, 4)
    assert chiral.required_rank == 3


def test_trans_double_bond_in_a_small_ring_is_triangle_infeasible() -> None:
    assert prepare_conformer_constraints(cyclohexene(1), TABLE).consistent
    infeasible = prepare_conformer_constraints(cyclohexene(-1), TABLE)
    assert not infeasible.consistent
    assert infeasible.maximum_inconsistency > 0.1
    assert infeasible.inconsistent_pair is not None
    component = cyclohexene(-1)
    result = ConformerGenerationPlan(component, infeasible).generate(
        jax.random.key(0), count=2
    )
    assert result.status == "bounds-inconsistent"
    assert result.conformers == () and result.attempts == ()
    assert result.failed_ordinals == (0, 1)
    with pytest.raises(ValueError, match="No conformer"):
        result.pool(component)


def test_local_geometry_overrides_must_admit_the_atom() -> None:
    with pytest.raises(ValueError, match="does not admit"):
        prepare_conformer_constraints(
            halomethane(), TABLE, local_geometries={0: "trigonal"}
        )
    with pytest.raises(ValueError, match="bond of order 2"):
        prepare_conformer_constraints(
            dichloroethene(-1),
            ConformerGeometryTable(TABLE.single_radii, TABLE.nonbonded_radii),
        )


def test_metric_embedding_reports_rank_and_indefiniteness() -> None:
    line = np.abs(np.arange(4.0)[:, None] - np.arange(4.0)[None, :])
    collinear = metric_embedding(line, dimension=3, required_rank=3)
    assert collinear.positive_rank == 1
    assert collinear.rank_deficient
    assert collinear.exact
    coordinates = np.asarray(collinear.coordinates)
    np.testing.assert_allclose(
        np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1),
        line,
        atol=1e-9,
    )
    violating = np.asarray([[0.0, 1.0, 3.0], [1.0, 0.0, 1.0], [3.0, 1.0, 0.0]])
    indefinite = metric_embedding(violating, dimension=3, required_rank=2)
    assert indefinite.negative_mass > 0.0
    assert not indefinite.exact


def test_generated_conformers_have_admitted_bonds_and_declared_stereo() -> None:
    component = halomethane()
    generation = plan(component)
    result = generation.generate(jax.random.key(7), count=3)
    assert result.status == "complete"
    assert result.accepted_ordinals == (0, 1, 2)
    assert all(attempt.ordinal in (0, 1, 2) for attempt in result.attempts)
    expected = np.asarray(generation.constraints.bond_lengths)
    for conformer in result.conformers:
        positions = np.asarray(conformer.positions)
        assert conformer.source == "native-distance-geometry"
        np.testing.assert_allclose(
            bond_lengths(component, positions), expected, atol=0.1 + TABLE.bond_tolerance
        )
        assert stereo_realization(component, positions).tetrahedral_volumes[0] > 0.1
    enantiomer = halomethane(-1)
    mirrored = plan(enantiomer).generate(jax.random.key(7), count=1)
    positions = np.asarray(mirrored.conformers[0].positions)
    assert stereo_realization(enantiomer, positions).tetrahedral_volumes[0] > 0.1


def test_aromatic_ring_is_planar_and_double_bond_stereo_is_realized() -> None:
    ring = benzene()
    result = plan(ring).generate(jax.random.key(3), count=1)
    assert result.status == "complete"
    positions = np.asarray(result.conformers[0].positions)
    centered = positions - positions.mean(axis=0)
    thickness = np.linalg.svd(centered, compute_uv=False)[-1] / np.sqrt(
        positions.shape[0]
    )
    assert thickness < 0.05
    for configuration in (-1, 1):
        component = dichloroethene(configuration)
        generated = plan(component).generate(jax.random.key(11), count=1)
        assert generated.status == "complete"
        cosine = stereo_realization(
            component, np.asarray(generated.conformers[0].positions)
        ).double_bond_cosines[0]
        assert cosine > 0.5


def test_ordinal_addressing_is_independent_of_the_requested_range() -> None:
    generation = plan(ethanol())
    key = jax.random.key(42)
    whole = generation.generate(key, count=3)
    head = generation.generate(key, count=2)
    tail = generation.generate(key, count=1, start_ordinal=2)
    assert head.next_ordinal == 2 and tail.next_ordinal == 3
    resumed = head.conformers + tail.conformers
    assert [item.conformer_id for item in resumed] == [
        item.conformer_id for item in whole.conformers
    ]
    for left, right in zip(resumed, whole.conformers, strict=True):
        np.testing.assert_array_equal(
            np.asarray(left.positions), np.asarray(right.positions)
        )
    other = generation.generate(jax.random.key(43), count=1)
    assert other.conformers[0].conformer_id != whole.conformers[0].conformer_id
    empty = generation.generate(key, count=0)
    assert empty.status == "empty" and empty.attempts == () and empty.next_ordinal == 0
    with pytest.raises(TypeError, match="typed JAX key"):
        generation.generate(jax.random.PRNGKey(0), count=1)


def test_failed_attempts_are_bounded_and_retained() -> None:
    component = halomethane()
    constraints = prepare_conformer_constraints(component, TABLE)
    impossible = ConformerGenerationPlan(
        component,
        constraints,
        max_attempts=2,
        qualification=ConformerQualification(minimum_distance=5.0),
    )
    result = impossible.generate(jax.random.key(1), count=2)
    assert result.status == "attempt-budget-exhausted"
    assert result.conformers == ()
    assert result.failed_ordinals == (0, 1)
    assert [(item.ordinal, item.attempt) for item in result.attempts] == [
        (0, 0), (0, 1), (1, 0), (1, 1)
    ]  # fmt: skip
    assert {item.status for item in result.attempts} == {"minimum-distance"}
    assert all(item.embedding_positive_rank >= 3 for item in result.attempts)
    assert all(np.isfinite(item.minimum_pair_distance) for item in result.attempts)


def test_energy_cleanup_uses_the_supplied_native_potential() -> None:
    component = ethanol()
    constraints = prepare_conformer_constraints(component, TABLE)
    bonds = jnp.asarray(component.bonds)
    targets = 1.05 * jnp.asarray(constraints.bond_lengths)

    def energy(positions: Array) -> Array:
        lengths = jnp.linalg.norm(
            positions[bonds[:, 0]] - positions[bonds[:, 1]], axis=-1
        )
        return jnp.sum((lengths - targets) ** 2)

    cleanup = ConformerEnergyCleanup(
        energy,
        method=FIRE(),
        termination=OptimizationTermination(maximum_steps=2000),
        energy_unit="test-harmonic",
        model_id="test-harmonic-bonds",
    )
    result = ConformerGenerationPlan(component, constraints, cleanup=cleanup).generate(
        jax.random.key(5), count=1
    )
    assert result.status == "complete"
    accepted = result.attempts[-1]
    assert accepted.status == "accepted"
    assert accepted.cleanup_status in (
        int(OptimizationStatus.SUCCESS),
        int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
    )
    assert np.isfinite(accepted.cleanup_energy)
    positions = np.asarray(result.conformers[0].positions)
    np.testing.assert_allclose(bond_lengths(component, positions), targets, atol=0.05)
    assert "cleanup=test-harmonic-bonds" in result.conformers[0].provenance


def test_rmsd_is_symmetry_aware_and_pruning_is_ordered() -> None:
    component = ethanol()
    generated = plan(component).generate(jax.random.key(9), count=1)
    conformer = generated.conformers[0]
    positions = np.asarray(conformer.positions)
    angle = 0.7
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0, 0, 1],
        ]
    )
    swapped = positions @ rotation.T
    swapped[[1, 2]] = swapped[[2, 1]]
    copy = MolecularConformer(component, swapped, provenance="rotated relabeled copy")
    rmsd = conformer_rmsd(component, conformer, copy, include_hydrogens=True)
    assert rmsd.rmsd < 1e-6
    assert rmsd.complete
    kept, removed = prune_conformers(
        component, (conformer, copy), rmsd_threshold=0.01, include_hydrogens=True
    )
    assert [item.conformer_id for item in kept] == [conformer.conformer_id]
    assert removed == (1,)
    evidence = qualify_geometry(
        component, prepare_conformer_constraints(component, TABLE), positions
    )
    assert evidence.maximum_bound_violation <= 0.1
