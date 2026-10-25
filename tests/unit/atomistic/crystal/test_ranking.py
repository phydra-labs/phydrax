#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Radial screening, certified dedup composition, ranking and relaxation handoff.

Independent controls: a host brute-force radial oracle that enumerates a
fixed generous image box (checked to be exhaustive) and compares per-species
sorted lists padded with the cutoff; exact dyadic simple-cubic scores for the
inclusive threshold and greedy non-transitivity; an explicitly stripped
structure for hydrogen exclusion; the persistent bank's greedy and component
decisions on an X ~ Y ~ Z chain whose ends are not certified; hand-computed
energies per formula unit and chemical-potential grand energies; a host
periodic Lennard-Jones energy (sigma = 1 angstrom) over every image inside the
cutoff of every relaxed structure, which also certifies the angstrom/nanometer
handoff conversion and that the relaxed supercell is the original primitive
lattice; and explicit refusal of uncalibrated reduced or unconvertible length
units.
"""

import itertools
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.atomistic._neighborhood_bundle import (
    AtomisticCellListImageSearch,
    AtomisticNeighborhoodBundlePlan,
    AtomisticNeighborhoodSource,
    AtomisticRoleCapacity,
    AtomisticRoleCapacityLadder,
)
from phydrax.atomistic._potential_program import PreparedAtomisticPotentialProgram
from phydrax.atomistic._relaxation import (
    AtomisticDeformationChart,
    AtomisticRelaxationConvergence,
    AtomisticSymmetryChart,
)
from phydrax.atomistic._sites import AtomisticSiteDomain
from phydrax.atomistic.crystal import (
    AsymmetricUnitPlan,
    AsymmetricUnitSite,
    AsymmetricUnitState,
    AtomicSymmetryPlan,
    create_dedup_bank,
    crystal_structure_from_expansion,
    CrystalStructure,
    DedupBankLimits,
    DedupMode,
    DedupPolicy,
    EquivalencePolicy,
    expand_asymmetric_unit,
    metric_from_cell_parameters,
    site_of_point,
    standard_space_group_setting,
)
from phydrax.atomistic.crystal._ranking import (
    ChemicalPotentialReference,
    CrystalRankingCandidate,
    CrystalRankingPolicy,
    CrystalRelaxationChartKind,
    CrystalRelaxationEvidence,
    CrystalRelaxationInput,
    radial_fingerprints,
    radial_scores,
    RadialScreenCapacityError,
    RadialScreenMode,
    RadialScreenPolicy,
    rank_crystal_candidates,
    relax_crystal_candidates,
    screen_dedup_rank,
    screen_radial,
)
from phydrax.discretization import (
    ParticleImageCapacity,
    ParticleImageCapacityLadder,
    PeriodicCell,
)
from phydrax.optim import TrustScaledLBFGS
from tests._support.crystal_geometry_molecules import argon


# -- radial oracle --------------------------------------------------------------------


_OracleRow = tuple[int, list[tuple[int, float]]]


def _oracle_rows(
    structure: CrystalStructure,
    cutoff: float,
    *,
    include_hydrogens: bool,
    typed: bool,
    reach: int = 4,
) -> list[_OracleRow]:
    keep = [
        a
        for a in range(structure.atom_count)
        if include_hydrogens or structure.species[a] != 1
    ]
    vectors = np.asarray(structure.vectors)
    cartesian = np.asarray(structure.fractional)[keep] @ vectors
    species = np.asarray(structure.species)[keep]
    rows: list[_OracleRow] = []
    for i in range(len(keep)):
        entries: list[tuple[int, float]] = []
        boundary = np.inf
        for j in range(len(keep)):
            for shift in itertools.product(range(-reach, reach + 1), repeat=3):
                if j == i and shift == (0, 0, 0):
                    continue
                distance = float(
                    np.linalg.norm(
                        cartesian[j] + np.asarray(shift) @ vectors - cartesian[i]
                    )
                )
                if max(abs(value) for value in shift) == reach:
                    boundary = min(boundary, distance)
                if distance < cutoff:
                    entries.append((int(species[j]), distance))
        # The fixed box is exhaustive only if its outer shell lies beyond the cutoff.
        assert boundary >= cutoff
        entries.sort(key=(lambda e: (e[0], e[1])) if typed else (lambda e: e[1]))
        rows.append((int(species[i]), entries))
    return rows


def _padded_mismatch(
    left: Sequence[float], right: Sequence[float], cutoff: float
) -> float:
    width = max(len(left), len(right))
    left = list(left) + [cutoff] * (width - len(left))
    right = list(right) + [cutoff] * (width - len(right))
    return max((max(a, b) / min(a, b) for a, b in zip(left, right)), default=1.0)


def _row_mismatch(
    left: Sequence[tuple[int, float]],
    right: Sequence[tuple[int, float]],
    cutoff: float,
    typed: bool,
) -> float:
    if not typed:
        return _padded_mismatch([d for _, d in left], [d for _, d in right], cutoff)
    kinds = sorted({t for t, _ in left} | {t for t, _ in right})
    return max(
        (
            _padded_mismatch(
                [d for t, d in left if t == kind],
                [d for t, d in right if t == kind],
                cutoff,
            )
            for kind in kinds
        ),
        default=1.0,
    )


def _oracle_score(
    first: CrystalStructure, second: CrystalStructure, policy: RadialScreenPolicy
) -> float:
    typed = policy.mode == "neighbor-typed"
    rows = [
        _oracle_rows(
            structure,
            policy.cutoff,
            include_hydrogens=policy.include_hydrogens,
            typed=typed,
        )
        for structure in (first, second)
    ]

    def directed(source: Sequence[_OracleRow], target: Sequence[_OracleRow]) -> float:
        worst = 1.0
        for center, row in source:
            best = np.inf
            for other_center, other_row in target:
                if policy.mode != "untyped" and center != other_center:
                    continue
                best = min(best, _row_mismatch(row, other_row, policy.cutoff, typed))
            worst = max(worst, best)
        return worst

    return max(directed(rows[0], rows[1]), directed(rows[1], rows[0])) - 1.0


def _policy(
    mode: RadialScreenMode = "untyped",
    *,
    cutoff: float = 3.2,
    threshold: float = 0.1,
    hydrogens: bool = True,
    max_atoms: int = 4,
    max_images: int = 343,
    max_neighbors: int = 96,
) -> RadialScreenPolicy:
    return RadialScreenPolicy(
        cutoff=cutoff,
        mode=mode,
        include_hydrogens=hydrogens,
        threshold=threshold,
        max_atoms=max_atoms,
        max_images=max_images,
        max_neighbors=max_neighbors,
    )


_SKEW = np.asarray([[2.1, 0.0, 0.0], [0.7, 2.3, 0.0], [0.4, 0.5, 2.6]])
_SKEW_FRACTIONAL = np.asarray(
    [(0.11, 0.21, 0.32), (0.58, 0.47, 0.13), (0.36, 0.83, 0.71)]
)


def _skew(
    species: tuple[int, ...],
    structure_id: str,
    *,
    displaced: float = 0.0,
    strain: float = 0.0,
) -> CrystalStructure:
    vectors = _SKEW @ (
        np.eye(3)
        + strain * np.asarray([[0.0, 0.3, 0.1], [0.3, 0.2, 0.0], [0.1, 0.0, -0.1]])
    )
    fractional = _SKEW_FRACTIONAL + displaced * np.asarray(
        [(0.02, -0.01, 0.03), (-0.03, 0.02, 0.01), (0.01, 0.03, -0.02)]
    )
    return CrystalStructure(vectors, fractional, species, structure_id=structure_id)


def _cubic(lattice: float, structure_id: str) -> CrystalStructure:
    return CrystalStructure(
        lattice * np.eye(3), np.zeros((1, 3)), [18], structure_id=structure_id
    )


# -- radial scores --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["untyped", "center-typed", "neighbor-typed"])
def test_radial_score_matches_the_brute_force_oracle_in_a_skew_cell_with_self_images(
    mode: RadialScreenMode,
) -> None:
    first = _skew((8, 8, 14), "first")
    second = _skew((8, 8, 14), "second", displaced=1.0, strain=0.05)
    policy = _policy(mode)
    # Every cell height is below the cutoff: atoms see their own periodic images.
    fingerprints = radial_fingerprints([first, second], policy, chunk_size=2)
    assert np.all(fingerprints.image_bounds >= 2)
    scores = radial_scores(
        fingerprints,
        fingerprints,
        np.asarray([(0, 1), (1, 0), (0, 0)]),
        pair_chunk_size=2,
    )
    expected = _oracle_score(first, second, policy)
    assert expected > 0.0
    np.testing.assert_allclose(scores[:2], expected, rtol=1e-10)
    assert scores[2] == 0.0
    counts = [
        len(row)
        for _, row in _oracle_rows(
            first, policy.cutoff, include_hydrogens=True, typed=False
        )
    ]
    np.testing.assert_array_equal(fingerprints.neighbor_counts[0, :3], counts)


def test_typed_modes_separate_species_environments() -> None:
    first = _skew((8, 8, 14), "first")
    swapped = _skew((8, 14, 8), "swapped")
    uniform = _skew((8, 8, 8), "uniform")
    scores = {}
    for mode in ("untyped", "center-typed", "neighbor-typed"):
        policy = _policy(mode)
        fingerprints = radial_fingerprints(
            [first, swapped, uniform], policy, chunk_size=3
        )
        values = radial_scores(
            fingerprints, fingerprints, np.asarray([(0, 1), (0, 2)]), pair_chunk_size=4
        )
        np.testing.assert_allclose(
            values[0], _oracle_score(first, swapped, policy), rtol=1e-10
        )
        np.testing.assert_allclose(
            values[1], _oracle_score(first, uniform, policy), rtol=1e-10
        )
        scores[mode] = values
    # Identical geometry with permuted species is indistinguishable untyped.
    assert scores["untyped"][0] == 0.0 and scores["untyped"][1] == 0.0
    assert scores["center-typed"][0] > 0.0
    # A centre species absent from the other structure has no compatible row.
    assert np.isinf(scores["center-typed"][1]) and np.isinf(scores["neighbor-typed"][1])


def test_hydrogen_exclusion_removes_centres_and_neighbours() -> None:
    vectors = np.asarray([[4.2, 0.0, 0.0], [0.9, 4.6, 0.0], [0.3, -0.6, 5.1]])
    full = CrystalStructure(
        vectors,
        np.asarray(
            [
                (0.10, 0.20, 0.30),
                (0.18, 0.25, 0.33),
                (0.55, 0.60, 0.70),
                (0.60, 0.52, 0.74),
            ]
        ),
        [6, 1, 8, 1],
        structure_id="full",
    )
    stripped = CrystalStructure(
        vectors,
        np.asarray([(0.10, 0.20, 0.30), (0.55, 0.60, 0.70)]),
        [6, 8],
        structure_id="stripped",
    )
    excluded = radial_fingerprints(
        [full], _policy("neighbor-typed", hydrogens=False), chunk_size=1
    )
    reference = radial_fingerprints([stripped], _policy("neighbor-typed"), chunk_size=1)
    np.testing.assert_array_equal(excluded.atom_mask, reference.atom_mask)
    np.testing.assert_array_equal(excluded.center_types, reference.center_types)
    np.testing.assert_array_equal(excluded.neighbor_types, reference.neighbor_types)
    np.testing.assert_allclose(
        excluded.distances, reference.distances, rtol=1e-14, atol=0.0
    )
    np.testing.assert_array_equal(excluded.source_atoms[0, :2], [0, 2])
    included = radial_fingerprints(
        [full, stripped], _policy("neighbor-typed"), chunk_size=2
    )
    assert (
        radial_scores(included, included, np.asarray([(0, 1)]), pair_chunk_size=1)[0]
        > 0.0
    )


def test_inclusive_threshold_boundary() -> None:
    pool = [_cubic(1.0, "small"), _cubic(1.25, "large")]
    for threshold, role in ((0.25, "duplicate"), (0.25 - 1e-9, "representative")):
        policy = _policy(cutoff=1.3, threshold=threshold, max_images=125, max_neighbors=8)
        fingerprints = radial_fingerprints(pool, policy, chunk_size=2)
        # Six neighbours at a and pads at the cutoff: the score is exactly 1.25 - 1.
        assert (
            radial_scores(
                fingerprints, fingerprints, np.asarray([(0, 1)]), pair_chunk_size=1
            )[0]
            == 0.25
        )
        result = screen_radial(fingerprints, pair_chunk_size=1)
        assert result.decisions[1].role == role
        assert result.decisions[1].score == 0.25


def test_capacities_refuse_truncated_searches() -> None:
    structure = _skew((8, 8, 14), "skew")
    for policy, capacity in (
        (_policy(max_atoms=2), "atoms"),
        (_policy(max_images=64), "images"),
        (_policy(max_neighbors=4), "neighbors"),
    ):
        with pytest.raises(RadialScreenCapacityError) as refused:
            radial_fingerprints([structure], policy, chunk_size=1)
        assert refused.value.capacity == capacity
        assert refused.value.structure_id == "skew"
    coincident = CrystalStructure(
        _SKEW,
        np.asarray([(0.1, 0.2, 0.3), (0.1, 0.2, 0.3)]),
        [8, 14],
        structure_id="coincident",
    )
    with pytest.raises(ValueError, match="coincident"):
        radial_fingerprints([coincident], _policy(), chunk_size=1)


def test_greedy_screen_is_order_dependent_non_transitive_and_chunk_invariant() -> None:
    a, b, c = _cubic(1.0, "a"), _cubic(1.05, "b"), _cubic(1.10, "c")
    policy = _policy(cutoff=1.2, threshold=0.06, max_images=125, max_neighbors=8)
    ordered = radial_fingerprints([a, b, c], policy, chunk_size=1)
    chunked = radial_fingerprints([a, b, c], policy, chunk_size=2)
    np.testing.assert_allclose(ordered.distances, chunked.distances, rtol=1e-15, atol=0.0)
    np.testing.assert_array_equal(ordered.neighbor_counts, chunked.neighbor_counts)
    scores = radial_scores(
        ordered, ordered, np.asarray([(0, 1), (1, 2), (0, 2)]), pair_chunk_size=3
    )
    assert scores[0] <= 0.06 and scores[1] <= 0.06 and scores[2] > 0.06
    results = [screen_radial(ordered, pair_chunk_size=size) for size in (1, 2, 5)]
    for result in results:
        assert [(d.role, d.representative_id) for d in result.decisions] == [
            ("representative", "a"),
            ("duplicate", "a"),
            ("representative", "c"),
        ]
        assert [d.score for d in result.decisions] == [
            d.score for d in results[0].decisions
        ]
        assert result.representative_ids == ("a", "c")
    reordered = screen_radial(
        radial_fingerprints([b, a, c], policy, chunk_size=3), pair_chunk_size=2
    )
    assert [(d.role, d.representative_id) for d in reordered.decisions] == [
        ("representative", "b"),
        ("duplicate", "b"),
        ("duplicate", "b"),
    ]


# -- ranking ----------------------------------------------------------------------------


_CELL = np.asarray([[5.3, 0.0, 0.0], [-1.4, 5.9, 0.0], [-0.9, 0.7, 7.3]])
_FRACTIONAL = np.asarray(
    [(0.11, 0.23, 0.37), (0.61, 0.17, 0.82), (0.29, 0.71, 0.44), (0.83, 0.52, 0.09)]
)
_SPECIES = (8, 14, 26, 29)


def _structure(
    structure_id: str, *, displaced: float = 0.0, atom: int = 2
) -> CrystalStructure:
    fractional = _FRACTIONAL.copy()
    fractional[atom] += np.asarray((displaced, 0.0, 0.0)) @ np.linalg.inv(_CELL)
    return CrystalStructure(_CELL, fractional, _SPECIES, structure_id=structure_id)


def _doubled(structure_id: str) -> CrystalStructure:
    vectors = _CELL.copy()
    vectors[0] *= 2.0
    fractional = np.concatenate(
        (_FRACTIONAL * (0.5, 1.0, 1.0), (_FRACTIONAL + (1.0, 0.0, 0.0)) * (0.5, 1.0, 1.0))
    )
    return CrystalStructure(vectors, fractional, _SPECIES * 2, structure_id=structure_id)


def _evidence(
    status: str = "converged", *, full: bool = True
) -> CrystalRelaxationEvidence:
    return CrystalRelaxationEvidence(
        status,
        status == "converged",
        full,
        True,
        1e-7,
        1e-7,
        1e-6,
        12,
        "variable-cell",
        "relaxation-definition",
        "setting",
        0.0,
        None,
    )


def _candidate(
    structure: CrystalStructure,
    energy: float | None,
    ordinal: int,
    *,
    energy_unit: str = "reduced-energy",
    formula_units: int = 1,
    hamiltonian_id: str = "atomistic-hamiltonian:test",
    numeric_revision_id: str = "revision-1",
    relaxation: CrystalRelaxationEvidence = _evidence(),
) -> CrystalRankingCandidate:
    return CrystalRankingCandidate(
        structure,
        energy=energy,
        ordinal=ordinal,
        energy_unit=energy_unit,
        formula_units=formula_units,
        hamiltonian_id=hamiltonian_id,
        numeric_revision_id=numeric_revision_id,
        program_prepared_id="prepared-1",
        relaxation=relaxation,
    )


def test_ranking_orders_energy_per_formula_unit_and_records_exclusions() -> None:
    candidates = [
        _candidate(_structure("one"), -10.0, 0),
        _candidate(_doubled("two"), -21.0, 1, formula_units=2),
        _candidate(_structure("three", displaced=0.4), -9.8, 2),
        _candidate(_structure("tie", displaced=0.6), -9.8, 3),
        _candidate(
            _structure("stalled", displaced=0.2),
            -30.0,
            4,
            relaxation=_evidence("evaluation_limit"),
        ),
        _candidate(_structure("missing", displaced=0.3), None, 5),
        _candidate(
            _structure("projected", displaced=0.5),
            -40.0,
            6,
            relaxation=_evidence(full=False),
        ),
    ]
    ranking = rank_crystal_candidates(
        candidates, policy=CrystalRankingPolicy(require_full_minimum=True)
    )
    assert [entry.structure_id for entry in ranking.entries] == [
        "two",
        "one",
        "three",
        "tie",
    ]
    np.testing.assert_allclose(
        [entry.value for entry in ranking.entries], [-10.5, -10.0, -9.8, -9.8]
    )
    np.testing.assert_allclose(
        [entry.relative for entry in ranking.entries], [0.0, 0.5, 0.7, 0.7]
    )
    assert [entry.rank for entry in ranking.entries] == [1, 2, 3, 4]
    assert {(item.structure_id, item.reason) for item in ranking.excluded} == {
        ("stalled", "not-converged:evaluation_limit"),
        ("missing", "no-energy"),
        ("projected", "projected-minimum-only"),
    }
    assert ranking.basis == "energy-per-formula-unit"
    assert ranking.formula == tuple((number, 1) for number in _SPECIES)
    assert "not finite-temperature polymorph stability" in ranking.nonclaim
    with pytest.raises(ValueError, match="formula_units"):
        _candidate(_structure("odd"), -1.0, 9, formula_units=3)


def test_ranking_refuses_mixed_definitions_and_needs_a_reference_for_compositions() -> (
    None
):
    base = _candidate(_structure("base"), -10.0, 0)
    policy = CrystalRankingPolicy()
    displaced = _structure("other", displaced=0.4)
    for other, field in (
        (
            _candidate(displaced, -9.0, 1, hamiltonian_id="atomistic-hamiltonian:other"),
            "hamiltonian_id",
        ),
        (
            _candidate(displaced, -9.0, 1, numeric_revision_id="revision-2"),
            "numeric_revision_id",
        ),
        (_candidate(displaced, -9.0, 1, energy_unit="electronvolt"), "energy_unit"),
    ):
        with pytest.raises(ValueError, match=field):
            rank_crystal_candidates([base, other], policy=policy)
    richer = _candidate(
        CrystalStructure(
            _CELL,
            np.concatenate((_FRACTIONAL, [(0.45, 0.05, 0.62)])),
            (8, 14, 26, 29, 8),
            structure_id="richer",
        ),
        -12.0,
        2,
    )
    with pytest.raises(ValueError, match="ChemicalPotentialReference"):
        rank_crystal_candidates([base, richer], policy=policy)
    potentials = {8: -1.5, 14: -2.0, 26: -3.0, 29: -2.5}
    reference = ChemicalPotentialReference(
        potentials, energy_unit="reduced-energy", name="test"
    )
    ranking = rank_crystal_candidates(
        [base, richer], policy=CrystalRankingPolicy(chemical_potentials=reference)
    )
    assert ranking.basis == "grand-energy-per-atom" and ranking.formula is None
    assert [entry.structure_id for entry in ranking.entries] == ["richer", "base"]
    np.testing.assert_allclose([entry.value for entry in ranking.entries], [-0.3, -0.25])
    with pytest.raises(ValueError, match="energy unit"):
        rank_crystal_candidates(
            [base, richer],
            policy=CrystalRankingPolicy(
                chemical_potentials=ChemicalPotentialReference(
                    potentials, energy_unit="electronvolt", name="test"
                )
            ),
        )
    with pytest.raises(ValueError, match="lacks elements"):
        rank_crystal_candidates(
            [base, richer],
            policy=CrystalRankingPolicy(
                chemical_potentials=ChemicalPotentialReference(
                    {8: -1.5, 14: -2.0, 26: -3.0},
                    energy_unit="reduced-energy",
                    name="partial",
                )
            ),
        )


# -- composition through the persistent bank ----------------------------------------------


_LIMITS = DedupBankLimits(
    max_commit_bytes=4_000_000, max_commits=64, max_entries=256, max_stored_atoms=10_000
)


def _dedup_policy(mode: DedupMode) -> DedupPolicy:
    return DedupPolicy(
        mode=mode,
        priority_order="ascending",
        equivalence=EquivalencePolicy(
            position_tolerance=0.05,
            lattice_tolerance=1.0e-3,
            supercell="primitive-reduction",
            enantiomorph="proper-only",
            maximum_atoms=64,
            maximum_translation_order=4,
            maximum_lattice_vectors=100_000,
            maximum_basis_checks=1_000_000,
            maximum_bases=256,
            maximum_assignments=10_000,
            maximum_mappings=16,
            maximum_nodes=10_000,
        ),
    )


@pytest.mark.parametrize("mode", ["greedy", "components"])
def test_screen_dedup_rank_separates_greedy_and_component_semantics(
    tmp_path: Path, mode: DedupMode
) -> None:
    candidates = [
        _candidate(_structure("z", displaced=0.08), -9.0, 2),
        _candidate(_structure("x"), -10.0, 0),
        _candidate(_structure("y", displaced=0.04), -9.5, 1),
        _candidate(
            _structure("w", displaced=0.5),
            -20.0,
            3,
            relaxation=_evidence("evaluation_limit"),
        ),
    ]
    bank = create_dedup_bank(
        tmp_path / mode, name=mode, policy=_dedup_policy(mode), limits=_LIMITS
    )
    outcome = screen_dedup_rank(
        candidates,
        bank,
        screen=_policy(
            "neighbor-typed", cutoff=4.0, threshold=0.5, max_images=27, max_neighbors=32
        ),
        ranking=CrystalRankingPolicy(),
        fingerprint_chunk_size=2,
        pair_chunk_size=2,
    )
    # Screening and certification follow the declared priority (energy, ordinal, id).
    assert [d.structure_id for d in outcome.screen.decisions] == ["x", "y", "z"]
    assert [(d.role, d.representative_id) for d in outcome.screen.decisions] == [
        ("representative", "x"),
        ("duplicate", "x"),
        ("duplicate", "x"),
    ]
    assert {d.structure_id for d in outcome.submission.decisions} == {"x", "y", "z"}
    excluded = {(item.structure_id, item.reason) for item in outcome.ranking.excluded}
    assert ("w", "not-converged:evaluation_limit") in excluded
    records = {record.structure_id: record for record in outcome.records}
    assert outcome.certified_comparisons == sum(
        record.comparisons for record in outcome.records
    )
    if mode == "greedy":
        # y ~ x is certified; z is not certified against x and greedy never revisits y.
        assert [entry.structure_id for entry in outcome.ranking.entries] == ["x", "z"]
        np.testing.assert_allclose(
            [entry.relative for entry in outcome.ranking.entries], [0.0, 1.0]
        )
        assert ("y", "duplicate-of:x") in excluded
        assert records["z"].certified_role == "representative"
        assert outcome.unconfirmed_proposals == (("z", "x"),)
    else:
        # The component closes x ~ y ~ z without certifying x ~ z.
        assert [entry.structure_id for entry in outcome.ranking.entries] == ["x"]
        assert {("y", "duplicate-of:x"), ("z", "duplicate-of:x")} <= excluded
        assert records["z"].certified_role == "member"
        assert outcome.unconfirmed_proposals == ()
    assert outcome.screen_misses == ()


# -- relaxation handoff ---------------------------------------------------------------------


_LJ_CUTOFF = 2.0
# Each primitive P-1 argon lattice (two atoms in a ~1.27 angstrom cell) is
# represented by its explicit 4 x 4 x 4 supercell, whose shortest translation
# (~5 angstrom) admits the classical minimum-image cutoff of 2 angstrom.
_SUPERCELL = 4
_ARGON_SITES = _SUPERCELL**3
_ARGON_ATOMS = 2 * _ARGON_SITES
_ARGON_LATTICES: tuple[
    tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    ...,
] = (
    ((1.27, 1.29, 1.26), (89.0, 91.0, 90.5), (0.24, 0.26, 0.25)),
    ((1.30, 1.25, 1.28), (91.0, 88.0, 92.0), (0.23, 0.27, 0.26)),
)


def _lj_energy(structure: CrystalStructure) -> float:
    """Shifted Lennard-Jones energy summed over every periodic image inside the cutoff.

    Wrapped fractional differences lie in ``(-1, 1)``, and a displacement ``v``
    has fractional component ``|g_k| <= |v| |b*_k|`` with ``b*_k`` the
    reciprocal (inverse-cell column) vectors, so integer translations with
    ``|n_k| <= floor(cutoff |b*_k|) + 1`` contain every image inside the cutoff.
    """
    vectors = np.asarray(structure.vectors, dtype=np.float64)
    fractional = np.asarray(structure.fractional, dtype=np.float64)
    cartesian = (fractional - np.floor(fractional)) @ vectors
    reach = (
        np.floor(_LJ_CUTOFF * np.linalg.norm(np.linalg.inv(vectors), axis=0)).astype(
            np.int64
        )
        + 1
    )
    shift = 4.0 * (_LJ_CUTOFF**-12 - _LJ_CUTOFF**-6)
    # displacement[i, j] = x_j - x_i in the central cell.
    displacement = cartesian[None, :, :] - cartesian[:, None, :]
    central = np.eye(structure.atom_count, dtype=np.bool_)
    total = 0.0
    for image in itertools.product(*(range(-int(n), int(n) + 1) for n in reach)):
        translation = np.asarray(image, dtype=np.float64) @ vectors
        distance = np.linalg.norm(displacement + translation, axis=-1)
        within = distance < _LJ_CUTOFF
        if not any(image):
            within &= ~central
        contact = distance[within]
        total += 0.5 * float(np.sum(4.0 * (contact**-12 - contact**-6) - shift))
    return total


def _argon_plan(sites: int) -> AsymmetricUnitPlan:
    """Declare ``sites`` general-position argon ASU sites in P-1."""
    setting = standard_space_group_setting(2)
    general = site_of_point(setting, (Fraction(1, 5), Fraction(1, 3), Fraction(1, 7)))
    return AsymmetricUnitPlan(
        setting,
        [AsymmetricUnitSite(argon(), general) for _ in range(sites)],
        chirality="enantiopure",
    )


def _argon_state(
    plan: AsymmetricUnitPlan,
    lengths: tuple[float, float, float],
    angles: tuple[float, float, float],
    centers: np.ndarray,
) -> AsymmetricUnitState:
    metric = metric_from_cell_parameters(
        jnp.asarray(lengths, dtype=jnp.float64), jnp.asarray(angles, dtype=jnp.float64)
    )
    count = centers.shape[0]
    return AsymmetricUnitState(
        jnp.asarray(centers, dtype=jnp.float64),
        jnp.tile(jnp.asarray([(1.0, 0.0, 0.0, 0.0)], dtype=jnp.float64), (count, 1)),
        plan.chart.coefficients(metric),
        jnp.zeros((count,), dtype=jnp.int32),
    )


def _primitive_argon(ordinal: int) -> CrystalStructure:
    """The original two-atom primitive cell of one argon lattice."""
    lengths, angles, center = _ARGON_LATTICES[ordinal]
    plan = _argon_plan(1)
    state = _argon_state(plan, lengths, angles, np.asarray([center], dtype=np.float64))
    return crystal_structure_from_expansion(
        plan, expand_asymmetric_unit(plan, state), structure_id=f"primitive-{ordinal}"
    )


_NANOMETER = phx.units.UnitDefinition(
    "nm", phx.units.LENGTH, phx.units.SI_REFERENCE_SYSTEM_ID, "1e-9"
)


def _unit_system(length: str) -> phx.atomistic.AtomisticUnitSystem:
    if length == "angstrom":
        return (
            phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
        )
    if length == "reduced":
        return phx.atomistic.AtomisticUnitSystem.reduced()
    return phx.atomistic.AtomisticUnitSystem(
        phx.atomistic.AtomisticScaleContract(_NANOMETER, phx.units.ELECTRONVOLT),
        mass_unit=phx.units.DALTON,
        time_unit=phx.units.FEMTOSECOND,
        charge_unit=phx.units.ELEMENTARY_CHARGE,
        temperature_unit=phx.units.KELVIN,
        constant_set_id="codata-2018",
    )


def _lj_program(
    cell: np.ndarray, *, atom_count: int, length: str = "angstrom"
) -> PreparedAtomisticPotentialProgram:
    """Lennard-Jones argon with ``sigma = 1`` angstrom and ``epsilon = 1`` eV.

    The same physical parameters are declared in the program's own length
    unit (``nm`` scales every length by 0.1); ``cell`` is in angstrom.
    """
    scale = 0.1 if length == "nanometer" else 1.0
    system = phx.atomistic.AtomisticSystemPlan(
        np.arange(1, atom_count + 1),
        np.full((atom_count,), 18, dtype=np.int32),
        np.ones((atom_count,), dtype=np.float64),
        _unit_system(length),
        atom_type_ids=np.zeros((atom_count,), dtype=np.int32),
        cell=PeriodicCell(scale * np.asarray(cell, dtype=np.float64)),
    ).prepare()
    plan = phx.atomistic.AtomisticPotentialProgram(
        [
            phx.atomistic.LennardJonesPotential(
                np.asarray([1.0]),
                np.asarray([scale]),
                _LJ_CUTOFF * scale,
                shift_energy_at_cutoff=True,
            )
        ]
    )
    (requirement,) = plan.requirements.neighborhoods
    source = AtomisticNeighborhoodSource(
        "pairs",
        AtomisticSiteDomain.PHYSICAL_ATOMS,
        (requirement,),
        AtomisticCellListImageSearch(
            maximum_candidate_slots=10**6, deformation_margin=1.0
        ),
        # Complete directed routes inside cutoff + skin: 58 per atom in the
        # start lattices (7424), with headroom for compression and shear.
        ParticleImageCapacityLadder(
            (
                ParticleImageCapacity(
                    maximum_particles_per_cell=32,
                    maximum_edges=12288,
                    maximum_degree=1024,
                    maximum_images=4096,
                    maximum_case_edges=12288,
                ),
            )
        ),
        skin=0.3 * scale,
    )
    # Directed routes inside the cutoff: 26 per atom in the start lattices and
    # 42 per atom (5376) in a close-packed relaxed lattice.
    ladder = AtomisticRoleCapacityLadder(
        requirement, (AtomisticRoleCapacity(maximum_routes=8192, maximum_degree=256),)
    )
    bundle = AtomisticNeighborhoodBundlePlan((requirement,), (source,), (ladder,))
    return plan.prepare(system, neighborhood_bundle=bundle)


def _argon_inputs() -> tuple[AsymmetricUnitPlan, list[CrystalRelaxationInput]]:
    plan = _argon_plan(_ARGON_SITES)
    translations = np.asarray(
        list(itertools.product(range(_SUPERCELL), repeat=3)), dtype=np.float64
    )
    inputs = []
    for ordinal, (lengths, angles, center) in enumerate(_ARGON_LATTICES):
        # Centers (n + c) / 4 and lengths 4 L: the same lattice and inversion pairs.
        centers = (translations + np.asarray(center, dtype=np.float64)) / _SUPERCELL
        state = _argon_state(
            plan,
            (
                _SUPERCELL * lengths[0],
                _SUPERCELL * lengths[1],
                _SUPERCELL * lengths[2],
            ),
            angles,
            centers,
        )
        expansion = expand_asymmetric_unit(plan, state)
        inputs.append(
            CrystalRelaxationInput(
                state,
                expansion,
                structure_id=f"packed-{ordinal}",
                lineage_id=f"campaign/packed-{ordinal}",
                ordinal=ordinal,
            )
        )
    return plan, inputs


def _relax_argon(
    plan: AsymmetricUnitPlan,
    inputs: Sequence[CrystalRelaxationInput],
    kind: CrystalRelaxationChartKind,
    *,
    length: str = "angstrom",
) -> tuple[CrystalRankingCandidate, ...]:
    # The nanometer program sees every length scaled by ``scale``.  Atom
    # coordinates are lengths while strain coordinates are ``cell_factor``
    # times a dimensionless strain, so ``cell_factor = scale * atom_count``
    # (the angstrom default of one per mobile atom) scales the whole chart
    # uniformly: chart steps scale by ``scale`` and chart gradients by
    # ``1 / scale``.  The step cap and projected tolerance are then the same
    # physical 0.02 angstrom and 1e-6 eV/angstrom in the program's unit.
    scale = 0.1 if length == "nanometer" else 1.0
    program = _lj_program(
        np.asarray(inputs[0].expansion.cell.vectors),
        atom_count=plan.atom_count,
        length=length,
    )
    atoms = np.arange(plan.atom_count)
    chart = (
        AtomisticDeformationChart(cell_factor=scale * plan.atom_count)
        if kind == "variable-cell"
        else AtomisticSymmetryChart(AtomicSymmetryPlan(plan), atoms)
    )
    return relax_crystal_candidates(
        plan,
        inputs,
        program,
        chart,
        system_atoms=atoms,
        method=TrustScaledLBFGS(maximum_step=0.02 * scale),
        convergence=AtomisticRelaxationConvergence(
            projected_tolerance=1e-6 / scale, maximum_evaluations=3000
        ),
        maximum_rounds=3000,
    )


@pytest.mark.parametrize("kind", ["variable-cell", "symmetry-atomic"])
def test_relaxation_handoff_ranks_relaxed_p1bar_argon_crystals(
    kind: CrystalRelaxationChartKind,
) -> None:
    plan, inputs = _argon_inputs()
    assert plan.site_count == _ARGON_SITES and plan.atom_count == _ARGON_ATOMS
    candidates = _relax_argon(plan, inputs, kind)
    assert len(candidates) == 2
    energies: list[float] = []
    for item, candidate in zip(inputs, candidates, strict=True):
        assert candidate.source_structure_id == item.structure_id
        assert candidate.lineage_id == item.lineage_id
        assert candidate.structure_id != item.structure_id
        assert candidate.formula_units == _ARGON_ATOMS
        assert candidate.formula == ((18, 1),)
        evidence = candidate.relaxation
        assert evidence is not None
        assert (
            evidence.chart == kind
            and evidence.converged
            and evidence.evaluation_successful
        )
        assert evidence.setting_id == plan.setting.setting_id
        energy = candidate.energy
        assert energy is not None
        # The same program's accepted energy equals an independent periodic sum.
        np.testing.assert_allclose(energy, _lj_energy(candidate.structure), rtol=1e-9)
        start = crystal_structure_from_expansion(
            plan, item.expansion, structure_id="start"
        )
        start_energy = _lj_energy(start)
        # The supercell is the original interacting primitive lattice.
        assert start_energy < 0.0
        np.testing.assert_allclose(
            start_energy,
            _ARGON_SITES * _lj_energy(_primitive_argon(item.ordinal)),
            rtol=1e-11,
        )
        assert energy <= start_energy + 1e-12
        if kind == "symmetry-atomic":
            assert evidence.symmetry_deviation < 1e-9
            assert evidence.symmetry_projected_norm is not None
        else:
            assert evidence.symmetry_projected_norm is None
            assert np.isfinite(evidence.symmetry_deviation)
        energies.append(energy)
    ranking = rank_crystal_candidates(candidates, policy=CrystalRankingPolicy())
    assert len(ranking.entries) == 2 and not ranking.excluded
    values = [energy / _ARGON_ATOMS for energy in energies]
    assert [entry.value for entry in ranking.entries] == sorted(values)
    assert ranking.hamiltonian_id == candidates[0].hamiltonian_id
    assert ranking.energy_unit == candidates[0].energy_unit


def test_relaxation_handoff_converts_angstrom_geometry_for_nanometer_units() -> None:
    plan, inputs = _argon_inputs()
    reference = _relax_argon(plan, inputs, "variable-cell")
    converted = _relax_argon(plan, inputs, "variable-cell", length="nanometer")
    for item, left, right in zip(inputs, reference, converted, strict=True):
        evidence = right.relaxation
        assert evidence is not None and evidence.converged
        assert evidence.evaluation_successful
        assert left.relaxation is not None
        assert evidence.relaxation_id != left.relaxation.relaxation_id
        energy = right.energy
        assert energy is not None and left.energy is not None
        # The returned structure is angstrom-valued: the angstrom-defined oracle
        # (sigma = 1 angstrom) reproduces the nanometer program's energy.  Read
        # as nanometers, the ~1.1 angstrom start contacts would lie far outside
        # the 0.2 nm cutoff.
        np.testing.assert_allclose(energy, _lj_energy(right.structure), rtol=1e-9)
        start = crystal_structure_from_expansion(
            plan, item.expansion, structure_id="start"
        )
        assert energy <= _lj_energy(start) + 1e-12
        # The same physical minimum as the angstrom program from the same start.
        np.testing.assert_allclose(energy, left.energy, rtol=1e-6)
        assert right.energy_unit == left.energy_unit


@pytest.mark.parametrize(
    ("length", "kind", "message"),
    [
        ("reduced", "variable-cell", "no calibration to the angstrom"),
        ("reduced", "symmetry-atomic", "no calibration to the angstrom"),
        ("nanometer", "symmetry-atomic", "must measure length in angstrom"),
    ],
)
def test_relaxation_handoff_refuses_uncalibrated_or_unconvertible_length_units(
    length: str, kind: CrystalRelaxationChartKind, message: str
) -> None:
    plan, inputs = _argon_inputs()
    with pytest.raises(ValueError, match=message):
        _relax_argon(plan, inputs, kind, length=length)
