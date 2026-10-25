#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certified crystal equivalence: equivalent, distinct and undetermined outcomes.

Independent controls: structures built by explicit unimodular basis changes,
Cartesian rotations, origin shifts, atom permutations and supercells of a
reference must be certified equivalent, and every certificate is re-verified
here from its returned exact bases, transform, shift and bijection; a 0.5 A
displacement is proven distinct; a displacement between the tolerance and
twice the tolerance, or an exhausted budget, is undetermined; the mirror
image of a chiral arrangement is distinct under proper-only matching yet
certified by an improper isometry when admitted; a racemate's mirrored copies
carry the independently declared opposite stereoisomer, realize its declared
tetrahedral sign, certify against a relabelled copy and survive persistence.
"""

from fractions import Fraction
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.atomistic.crystal import (
    AsymmetricUnitPlan,
    AsymmetricUnitSite,
    AsymmetricUnitState,
    certify_equivalence,
    ConformerPool,
    create_dedup_bank,
    crystal_structure_from_expansion,
    CrystalStructure,
    DedupBankError,
    DedupBankLimits,
    DedupCandidate,
    DedupPolicy,
    EnantiomorphPolicy,
    EquivalencePolicy,
    EquivalenceResult,
    EquivalenceStatus,
    expand_asymmetric_unit,
    metric_from_cell_parameters,
    MolecularComponent,
    open_dedup_bank,
    site_of_point,
    standard_space_group_setting,
    stereo_realization,
    submit_to_dedup_bank,
    SupercellPolicy,
)
from tests._support.crystal_geometry_molecules import (
    chlorofluorobromomethane,
    trans_dichloroethene,
)


_TOLERANCE = 0.05


def _policy(
    *,
    supercell: SupercellPolicy = "primitive-reduction",
    enantiomorph: EnantiomorphPolicy = "proper-only",
    maximum_lattice_vectors: int = 200_000,
    maximum_assignments: int = 20_000,
) -> EquivalencePolicy:
    return EquivalencePolicy(
        position_tolerance=_TOLERANCE,
        lattice_tolerance=1.0e-3,
        supercell=supercell,
        enantiomorph=enantiomorph,
        maximum_atoms=256,
        maximum_translation_order=8,
        maximum_lattice_vectors=maximum_lattice_vectors,
        maximum_basis_checks=2_000_000,
        maximum_bases=512,
        maximum_assignments=maximum_assignments,
        maximum_mappings=64,
        maximum_nodes=100_000,
    )


def _cell(
    lengths: tuple[float, float, float], angles: tuple[float, float, float]
) -> np.ndarray:
    a, b, c = lengths
    alpha, beta, gamma = np.deg2rad(angles)
    vectors = np.zeros((3, 3))
    vectors[0] = (a, 0.0, 0.0)
    vectors[1] = (b * np.cos(gamma), b * np.sin(gamma), 0.0)
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    vectors[2] = (cx, cy, np.sqrt(c * c - cx * cx - cy * cy))
    return vectors


_REFERENCE_CELL = _cell((5.3, 6.1, 7.4), (81.0, 97.0, 104.0))
_REFERENCE_FRACTIONAL = np.asarray(
    [(0.11, 0.23, 0.37), (0.61, 0.17, 0.82), (0.29, 0.71, 0.44), (0.83, 0.52, 0.09)]
)
_REFERENCE_SPECIES = (8, 14, 26, 29)


def _reference(structure_id: str = "reference") -> CrystalStructure:
    return CrystalStructure(
        _REFERENCE_CELL,
        _REFERENCE_FRACTIONAL,
        _REFERENCE_SPECIES,
        structure_id=structure_id,
    )


def _rotation(seed: int) -> np.ndarray:
    matrix, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return matrix * np.sign(np.linalg.det(matrix))


def _transformed(
    transform: np.ndarray,
    shift: np.ndarray,
    permutation: np.ndarray,
    *,
    rotation: np.ndarray,
    fractional: np.ndarray = _REFERENCE_FRACTIONAL,
    structure_id: str,
) -> CrystalStructure:
    vectors = transform.T @ _REFERENCE_CELL @ rotation.T
    moved = fractional @ np.linalg.inv(transform).T + shift
    return CrystalStructure(
        vectors,
        moved[permutation],
        np.asarray(_REFERENCE_SPECIES)[permutation],
        structure_id=structure_id,
    )


def _verify_certificate(
    left: CrystalStructure, right: CrystalStructure, result: EquivalenceResult
) -> None:
    certificate = result.certificate
    assert certificate is not None
    left_basis = np.asarray([float(v) for v in certificate.left_basis]).reshape(3, 3)
    right_basis = np.asarray([float(v) for v in certificate.right_basis]).reshape(3, 3)
    transform = np.asarray(certificate.transform, dtype=float).reshape(3, 3)
    left_compared = left.fractional[certificate.left_atoms] @ np.linalg.inv(left_basis).T
    right_compared = (
        right.fractional[certificate.right_atoms] @ np.linalg.inv(right_basis).T
    )
    predicted = left_compared @ np.linalg.inv(transform).T + certificate.origin_shift
    difference = right_compared[certificate.mapping] - predicted
    difference -= np.round(difference)
    right_vectors = right_basis.T @ right.vectors
    residual = np.linalg.norm(difference @ right_vectors, axis=1)
    assert np.max(residual) <= _TOLERANCE + 1e-9
    assert np.max(residual) == pytest.approx(certificate.max_residual, abs=1e-9)
    left_species = left.species[certificate.left_atoms]
    right_species = right.species[certificate.right_atoms]
    assert np.array_equal(left_species, right_species[certificate.mapping])


def test_basis_origin_rotation_and_permutation_are_certified_equivalent() -> None:
    left = _reference()
    transform = np.asarray([[1, 1, 0], [0, 1, 0], [-1, 0, 1]])
    noise = 0.01 * np.random.default_rng(2).normal(size=_REFERENCE_FRACTIONAL.shape) / 7.0
    right = _transformed(
        transform,
        np.asarray((0.3, -0.2, 0.45)),
        np.asarray([2, 0, 3, 1]),
        rotation=_rotation(4),
        fractional=_REFERENCE_FRACTIONAL + noise,
        structure_id="moved",
    )
    result = certify_equivalence(left, right, _policy())
    assert result.status is EquivalenceStatus.EQUIVALENT
    assert result.left_id == "reference" and result.right_id == "moved"
    assert result.left_content_id != result.right_content_id
    assert result.certificate is not None and not result.certificate.improper
    _verify_certificate(left, right, result)


def test_supercell_equivalence_depends_on_the_declared_supercell_policy() -> None:
    left = _reference()
    doubled = np.concatenate(
        (
            _REFERENCE_FRACTIONAL * (0.5, 1.0, 1.0),
            _REFERENCE_FRACTIONAL * (0.5, 1.0, 1.0) + (0.5, 0, 0),
        )
    )
    right = CrystalStructure(
        np.diag((2.0, 1.0, 1.0)) @ _REFERENCE_CELL,
        doubled,
        _REFERENCE_SPECIES * 2,
        structure_id="supercell",
    )
    primitive = certify_equivalence(left, right, _policy())
    assert primitive.status is EquivalenceStatus.EQUIVALENT
    _verify_certificate(left, right, primitive)
    identical = certify_equivalence(left, right, _policy(supercell="identical-lattice"))
    assert identical.status is EquivalenceStatus.DISTINCT


def test_large_displacement_is_proven_distinct() -> None:
    left = _reference()
    fractional = _REFERENCE_FRACTIONAL.copy()
    fractional[1] += np.asarray((0.5, 0.0, 0.0)) @ np.linalg.inv(_REFERENCE_CELL)
    right = CrystalStructure(
        _REFERENCE_CELL, fractional, _REFERENCE_SPECIES, structure_id="displaced"
    )
    result = certify_equivalence(left, right, _policy())
    assert result.status is EquivalenceStatus.DISTINCT
    assert result.exhausted == () and result.ambiguous == 0
    assert result.certificate is None


def test_tolerance_boundary_and_exhausted_budget_are_undetermined() -> None:
    left = _reference()
    fractional = _REFERENCE_FRACTIONAL.copy()
    fractional[2] += np.asarray((0.08, 0.0, 0.0)) @ np.linalg.inv(_REFERENCE_CELL)
    near = CrystalStructure(
        _REFERENCE_CELL, fractional, _REFERENCE_SPECIES, structure_id="near"
    )
    boundary = certify_equivalence(left, near, _policy())
    assert boundary.status is EquivalenceStatus.UNDETERMINED
    assert boundary.ambiguous >= 1
    budget = certify_equivalence(left, _reference("copy"), _policy(maximum_assignments=1))
    assert budget.status is EquivalenceStatus.UNDETERMINED
    assert "assignments" in budget.exhausted
    lattice = certify_equivalence(
        left, _reference("copy"), _policy(maximum_lattice_vectors=3)
    )
    assert lattice.status is EquivalenceStatus.UNDETERMINED
    assert "lattice-vectors" in lattice.exhausted


def test_mirror_image_is_an_explicit_enantiomorph() -> None:
    left = _reference()
    mirror = CrystalStructure(
        _REFERENCE_CELL, -_REFERENCE_FRACTIONAL, _REFERENCE_SPECIES, structure_id="mirror"
    )
    proper = certify_equivalence(left, mirror, _policy())
    assert proper.status is EquivalenceStatus.DISTINCT
    assert proper.enantiomorphic is True
    improper = certify_equivalence(left, mirror, _policy(enantiomorph="allow-improper"))
    assert improper.status is EquivalenceStatus.EQUIVALENT
    assert improper.certificate is not None and improper.certificate.improper
    _verify_certificate(left, mirror, improper)


def _molecular_structure(
    pool: ConformerPool,
    setting_number: int,
    centre: tuple[float, float, float],
    quaternion: np.ndarray,
    structure_id: str,
) -> CrystalStructure:
    setting = standard_space_group_setting(setting_number)
    site = site_of_point(setting, (Fraction(1, 5), Fraction(1, 3), Fraction(1, 7)))
    plan = AsymmetricUnitPlan(
        setting, [AsymmetricUnitSite(pool, site)], chirality="enantiopure"
    )
    metric = metric_from_cell_parameters(
        jnp.asarray((6.2, 7.4, 8.9), dtype=jnp.float64),
        jnp.asarray((90.0, 101.0, 90.0), dtype=jnp.float64),
    )
    state = AsymmetricUnitState(
        jnp.asarray([centre]),
        jnp.asarray([quaternion]),
        plan.chart.coefficients(metric),
        jnp.zeros((1,), dtype=jnp.int32),
    )
    return crystal_structure_from_expansion(
        plan, expand_asymmetric_unit(plan, state), structure_id=structure_id
    )


def _molecules(
    structure: CrystalStructure, component: MolecularComponent | None = None
) -> list[tuple[MolecularComponent, list[int]]]:
    result: list[tuple[MolecularComponent, list[int]]] = []
    for m, own in enumerate(structure.molecule_components):
        atoms = np.flatnonzero(structure.molecule_index == m)
        order = atoms[np.argsort(structure.molecule_local[atoms])]
        result.append((own if component is None else component, order.tolist()))
    return result


def test_molecular_structures_require_topology_correspondence_and_handedness() -> None:
    quaternion = np.asarray((0.7, 0.1, -0.4, 0.3))
    quaternion /= np.linalg.norm(quaternion)
    planar = trans_dichloroethene()
    first = _molecular_structure(planar, 14, (0.21, 0.13, 0.36), quaternion, "planar-a")
    second = CrystalStructure(
        np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, -1]]) @ first.vectors,
        first.fractional[:, (1, 0, 2)] * (1, 1, -1) + (0.25, 0.5, 0.125),
        first.species,
        structure_id="planar-b",
        molecules=_molecules(first),
    )
    result = certify_equivalence(first, second, _policy())
    assert result.status is EquivalenceStatus.EQUIVALENT
    _verify_certificate(first, second, result)
    chiral = chlorofluorobromomethane()
    left = _molecular_structure(chiral, 4, (0.21, 0.13, 0.36), quaternion, "chiral")
    # The mirror image carries the opposite declared tetrahedral configuration.
    enantiomer = chlorofluorobromomethane(-1).component
    right = CrystalStructure(
        left.vectors,
        -left.fractional,
        left.species,
        structure_id="enantiomer",
        molecules=_molecules(left, enantiomer),
    )
    assert enantiomer.component_id != chiral.component.component_id
    proper = certify_equivalence(left, right, _policy())
    assert proper.status is EquivalenceStatus.DISTINCT
    assert proper.enantiomorphic is True
    improper = certify_equivalence(left, right, _policy(enantiomorph="allow-improper"))
    assert improper.status is EquivalenceStatus.EQUIVALENT
    assert improper.certificate is not None and improper.certificate.improper


def test_racemate_mirrored_copies_carry_the_opposite_stereoisomer(
    tmp_path: Path,
) -> None:
    chiral = chlorofluorobromomethane()
    # Independent declaration of the mirror stereoisomer (inverted parity).
    mirror = chlorofluorobromomethane(-1).component
    setting = standard_space_group_setting(2)
    site = site_of_point(setting, (Fraction(1, 5), Fraction(1, 3), Fraction(1, 7)))
    plan = AsymmetricUnitPlan(
        setting, [AsymmetricUnitSite(chiral, site)], chirality="racemic"
    )
    assert sorted(plan.copy_enantiomer.tolist()) == [False, True]
    quaternion = np.asarray((0.7, 0.1, -0.4, 0.3))
    quaternion /= np.linalg.norm(quaternion)
    metric = metric_from_cell_parameters(
        jnp.asarray((6.2, 7.4, 8.9), dtype=jnp.float64),
        jnp.asarray((90.0, 101.0, 90.0), dtype=jnp.float64),
    )
    state = AsymmetricUnitState(
        jnp.asarray([(0.21, 0.13, 0.36)]),
        jnp.asarray([quaternion]),
        plan.chart.coefficients(metric),
        jnp.zeros((1,), dtype=jnp.int32),
    )
    expansion = expand_asymmetric_unit(plan, state)
    structure = crystal_structure_from_expansion(plan, expansion, structure_id="racemate")
    positions = np.asarray(expansion.positions, dtype=np.float64)
    declared = []
    members = []
    for m, component in enumerate(structure.molecule_components):
        expected = mirror if bool(plan.copy_enantiomer[m]) else chiral.component
        assert component.component_id == expected.component_id
        atoms = np.flatnonzero(structure.molecule_index == m)
        order = atoms[np.argsort(structure.molecule_local[atoms])]
        # Each copy realizes its own declaration; the mirrored copy realizes the
        # opposite sign of the source declaration.
        assert np.all(
            stereo_realization(expected, positions[order]).tetrahedral_volumes > 0
        )
        source = stereo_realization(chiral.component, positions[order])
        assert np.all((source.tetrahedral_volumes > 0) != bool(plan.copy_enantiomer[m]))
        declared.append(expected)
        members.append(order)
    # A relabelled, origin-shifted copy declared from the independent components.
    count = structure.atom_count
    permutation = np.arange(count)[::-1]
    inverse = np.argsort(permutation)
    relabelled = CrystalStructure(
        structure.vectors,
        structure.fractional[permutation] + (0.25, 0.5, 0.125),
        structure.species[permutation],
        structure_id="racemate-relabelled",
        molecules=[
            (component, inverse[order].tolist())
            for component, order in zip(declared, members, strict=True)
        ],
    )
    result = certify_equivalence(structure, relabelled, _policy())
    assert result.status is EquivalenceStatus.EQUIVALENT
    _verify_certificate(structure, relabelled, result)
    # The same coordinates declared with one handedness for every copy describe
    # a different (stereo-inconsistent) structure and are not certified.
    uniform = CrystalStructure(
        relabelled.vectors,
        relabelled.fractional,
        relabelled.species,
        structure_id="racemate-uniform",
        molecules=[(chiral.component, inverse[order].tolist()) for order in members],
    )
    assert (
        certify_equivalence(structure, uniform, _policy()).status
        is not EquivalenceStatus.EQUIVALENT
    )
    # Persistence keeps the mirrored component identity; restore needs both.
    limits = DedupBankLimits(
        max_commit_bytes=4_000_000, max_commits=8, max_entries=16, max_stored_atoms=1_000
    )
    root = tmp_path / "racemate-bank"
    bank = create_dedup_bank(
        root,
        name="racemate",
        policy=DedupPolicy(
            mode="greedy", priority_order="ascending", equivalence=_policy()
        ),
        limits=limits,
    )
    submission = submit_to_dedup_bank(
        bank, [DedupCandidate(structure, priority=0.0, ordinal=0)]
    )
    assert submission.published
    reopened = open_dedup_bank(root, limits=limits, components=(chiral.component, mirror))
    (stored,) = reopened.structures
    assert stored.content_id == structure.content_id
    assert [c.component_id for c in stored.molecule_components] == [
        c.component_id for c in declared
    ]
    with pytest.raises(DedupBankError, match="molecular component"):
        open_dedup_bank(root, limits=limits, components=(chiral.component,))
