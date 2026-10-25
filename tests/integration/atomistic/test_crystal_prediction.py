#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native molecular crystal prediction, end to end.

molecular input -> native distance-geometry conformers -> general-position
packing in P1 -> ASU/expanded structure-store records -> expansion ->
symmetry-constrained rigid-molecule physical relaxation with a native
Lennard-Jones program -> relaxed records with source lineage -> radial
screen and certified persistent dedup -> same-definition ranking.

The Lennard-Jones parameters are a declared test Hamiltonian, not a
chemically qualified force field; the scenario proves the native mechanics,
identities and evidence of the workflow, not predictive accuracy.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax.atomistic._generation import AtomisticStructureStoreGenerationSink
from phydrax.atomistic._neighborhood_bundle import (
    AtomisticDenseImageSearch,
    AtomisticNeighborhoodBundlePlan,
    AtomisticNeighborhoodSource,
    AtomisticRoleCapacity,
    AtomisticRoleCapacityLadder,
)
from phydrax.atomistic._potential_program import PreparedAtomisticPotentialProgram
from phydrax.atomistic._relaxation import (
    AtomisticRelaxationConvergence,
    AtomisticSymmetryChart,
)
from phydrax.atomistic._sites import AtomisticSiteDomain
from phydrax.atomistic._structure_store import (
    AtomisticStoreLayout,
    AtomisticStoreLimits,
    AtomisticStructureSource,
    AtomisticStructureStoreWriter,
    open_atomistic_structure_store,
)
from phydrax.atomistic.crystal import (
    ConformerGenerationPlan,
    ConformerGeometryTable,
    ContactPolicy,
    create_dedup_bank,
    DedupBankLimits,
    DedupPolicy,
    ElementTable,
    EquivalencePolicy,
    MolecularComponent,
    MolecularCrystalInput,
    prepare_conformer_constraints,
    RigidSymmetryLayout,
    standard_space_group_setting,
    VolumePolicy,
)
from phydrax.atomistic.crystal._packing import (
    crystal_structure_record,
    crystal_structure_schema,
    CrystalCellPolicy,
    CrystalContactResources,
    CrystalPackingPlan,
    CrystalReliefPolicy,
    CrystalSpaceGroupPolicy,
    native_conformer_pools,
)
from phydrax.atomistic.crystal._ranking import (
    CrystalRankingPolicy,
    CrystalRelaxationInput,
    RadialScreenPolicy,
    relax_crystal_candidates,
    screen_dedup_rank,
)
from phydrax.discretization import (
    ParticleImageCapacity,
    ParticleImageCapacityLadder,
    PeriodicCell,
)
from phydrax.optim import TrustScaledLBFGS


PYYKKO = "Pyykkö and Atsumi, Chem. Eur. J. 15 (2009) 186 and 12770"
BONDI = "Bondi, J. Phys. Chem. 68 (1964) 441"
MASSES = ElementTable(
    "mass",
    {1: 1.008, 6: 12.011, 9: 18.998, 17: 35.45, 35: 79.904},
    provenance="CIAAW 2021 conventional standard atomic weights",
)
CONTACT_RADII = ElementTable(
    "contact-radius", {1: 1.20, 6: 1.70, 9: 1.47, 17: 1.75, 35: 1.85}, provenance=BONDI
)
TABLE = ConformerGeometryTable(
    ElementTable(
        "covalent-radius",
        {1: 0.32, 6: 0.75, 9: 0.64, 17: 0.99, 35: 1.14},
        provenance=PYYKKO,
    ),
    CONTACT_RADII,
)


def _component() -> MolecularComponent:
    return MolecularComponent(
        (6, 1, 9, 17, 35),
        ((0, 1), (0, 2), (0, 3), (0, 4)),
        bond_orders=(1, 1, 1, 1),
        masses=MASSES,
        spin_multiplicity=1,
        tetrahedral_stereo=((0, 1, 2, 3, 1),),
        name="chlorofluorobromomethane",
    )


def _contact_resources(lanes: int) -> CrystalContactResources:
    width = 8192
    return CrystalContactResources(
        AtomisticDenseImageSearch(maximum_dense_routes=lanes * 500_000),
        search_capacities=(
            ParticleImageCapacity(
                maximum_particles_per_cell=64,
                maximum_edges=lanes * width,
                maximum_degree=width,
                maximum_images=4096,
                maximum_case_edges=width,
            ),
        ),
        role_capacities=(
            AtomisticRoleCapacity(maximum_routes=lanes * width, maximum_degree=width),
        ),
        skin=0.5,
    )


def _store_limits() -> AtomisticStoreLimits:
    return AtomisticStoreLimits(
        max_commit_bytes=1 << 21,
        max_member_bytes=1 << 22,
        max_shard_bytes=1 << 24,
        max_shard_members=8192,
        max_chunk_bytes=1 << 20,
        max_index_bytes=1 << 21,
        max_decoded_bytes=1 << 24,
        max_read_bytes=1 << 24,
        max_records=1000,
    )


def _lj_program(
    atom_count: int,
    numbers: Sequence[int],
    masses: Sequence[float],
    cell: np.ndarray,
    bonds: Sequence[tuple[int, int]],
) -> PreparedAtomisticPotentialProgram:
    system = phx.atomistic.AtomisticSystemPlan(
        np.arange(1, atom_count + 1),
        np.asarray(numbers, dtype=np.int32),
        np.asarray(masses, dtype=np.float64),
        phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond(),
        atom_type_ids=np.zeros((atom_count,), dtype=np.int32),
        topology=phx.atomistic.MolecularTopologyPlan(
            bonds=[[a + 1, b + 1] for a, b in bonds]
        ),
        cell=PeriodicCell(np.asarray(cell)),
    ).prepare()
    plan = phx.atomistic.AtomisticPotentialProgram(
        [
            phx.atomistic.LennardJonesPotential(
                np.asarray([0.01]), np.asarray([2.0]), 3.2, switch_distance=2.9
            )
        ]
    )
    (requirement,) = plan.requirements.neighborhoods
    source = AtomisticNeighborhoodSource(
        "pairs",
        AtomisticSiteDomain.PHYSICAL_ATOMS,
        (requirement,),
        AtomisticDenseImageSearch(maximum_dense_routes=2_000_000, deformation_margin=1.0),
        ParticleImageCapacityLadder(
            (
                ParticleImageCapacity(
                    maximum_particles_per_cell=64,
                    maximum_edges=16384,
                    maximum_degree=16384,
                    maximum_images=4096,
                    maximum_case_edges=16384,
                ),
            )
        ),
        skin=0.5,
    )
    ladder = AtomisticRoleCapacityLadder(
        requirement, (AtomisticRoleCapacity(maximum_routes=16384, maximum_degree=4096),)
    )
    bundle = AtomisticNeighborhoodBundlePlan((requirement,), (source,), (ladder,))
    return plan.prepare(system, neighborhood_bundle=bundle)


# One end-to-end scientific lifecycle: the same admitted molecules flow through
# conformers, durable packing, physical relaxation, deduplication and ranking.
# Intermediate evidence stays in this scenario so a later success cannot hide loss.
def test_molecular_input_to_ranked_crystals(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    component = _component()
    # Native distance-geometry conformers (no external embedding engine).
    conformer_plan = ConformerGenerationPlan(
        component, prepare_conformer_constraints(component, TABLE), max_attempts=16
    )
    pools, conformer_results = native_conformer_pools(
        [conformer_plan], jr.key(0), count=2
    )
    assert conformer_results[0].accepted_ordinals
    molecular = MolecularCrystalInput(
        pools,
        contact=ContactPolicy(CONTACT_RADII),
        volume=VolumePolicy(formula_unit_volume=180.0),
    )
    lanes = 2
    # Every cell height stays at least 2 x 3.2 A (the Lennard-Jones cutoff) plus
    # margin, so the program's minimum-image neighborhoods remain valid.  With
    # Z = 2 and V_ref = (1.0..1.3) x 2 x 180 A^3, h1 h2 h3 <= V bounds the shapes
    # meeting that floor to nearly cubic cells (axis ratio <= ~1.8, angles
    # near 90 degrees); the cell prior draws from that admissible region.
    # Relief compression is isotropic and its step is refused at the height
    # floor, so an oversampled start would pin the cell above V_ref; the dilute
    # reference volume itself is the starting volume and relief removes the
    # random-placement overlaps at it.
    packer = CrystalPackingPlan(
        molecular,
        CrystalSpaceGroupPolicy.fixed(standard_space_group_setting(1)),
        z_prime=2,
        chirality="enantiopure",
        resources=_contact_resources(lanes),
        cell=CrystalCellPolicy(
            volume_range=(1.0, 1.3),
            oversample=1.0,
            maximum_axis_ratio=1.5,
            angle_range=(75.0, 105.0),
            minimum_height=6.5,
            minimum_height_floor=6.45,
        ),
        relief=CrystalReliefPolicy(
            maximum_steps=400, check_interval=10, overlap_tolerance=0.1
        ),
    ).prepare(lanes=lanes)

    # Packing publishes every attempt (accepted and rejected) to the store.
    root = tmp_path / "crystals"
    limits = _store_limits()
    writer = AtomisticStructureStoreWriter(
        root,
        layout=AtomisticStoreLayout(
            records_per_shard=8, target_chunk_bytes=4096, compression="none"
        ),
        limits=limits,
    )
    schema = crystal_structure_schema()
    writer.create(schema, name="crystal-prediction")
    sink = AtomisticStructureStoreGenerationSink(
        root, writer, packer.record_builder, limits=limits
    )
    packed, _ = packer.pack(
        jr.key(1),
        campaign_id="crystal-prediction",
        target=3,
        attempt_budget=12,
        sink=sink,
    )
    accepted = packed.accepted
    assert accepted, (
        f"no accepted packing: {[attempt.status for attempt in packed.attempts]}"
    )
    stored = open_atomistic_structure_store(root, limits=limits)
    assert stored.record_ids == tuple(attempt.proposal_id for attempt in packed.attempts)

    # Re-expansion from the stored ASU reproduces the published geometry.
    plan = accepted[0].plan
    batch = stored.read(np.arange(len(packed.attempts)))
    offsets = batch.offsets("atom")
    for index, attempt in enumerate(packed.attempts):
        rows = slice(int(offsets[index]), int(offsets[index + 1]))
        np.testing.assert_allclose(
            batch.values["positions"][rows],
            np.asarray(attempt.expansion.positions),
            atol=1e-12,
        )

    # Rigid-molecule symmetry-constrained relaxation with one native program.
    numbers = [component.atomic_numbers[int(local)] for local in plan.atom_local]
    masses = [component.masses[int(local)] for local in plan.atom_local]
    bonds = [
        (
            int(np.flatnonzero(plan.atom_copy == copy)[a]),
            int(np.flatnonzero(plan.atom_copy == copy)[b]),
        )
        for copy in range(plan.copy_count)
        for a, b in component.bonds
    ]
    program = _lj_program(
        plan.atom_count,
        numbers,
        masses,
        np.asarray(accepted[0].expansion.cell.vectors),
        bonds,
    )
    atoms = np.arange(plan.atom_count)
    inputs = [
        CrystalRelaxationInput(
            attempt.state,
            attempt.expansion,
            structure_id=attempt.proposal_id,
            lineage_id=attempt.proposal_id,
            ordinal=attempt.ordinal,
        )
        for attempt in accepted
    ]
    candidates = relax_crystal_candidates(
        plan,
        inputs,
        program,
        AtomisticSymmetryChart(RigidSymmetryLayout(plan), atoms),
        system_atoms=atoms,
        method=TrustScaledLBFGS(maximum_step=0.02),
        convergence=AtomisticRelaxationConvergence(
            projected_tolerance=1e-4, maximum_evaluations=5000
        ),
        maximum_rounds=5000,
    )
    assert [candidate.source_structure_id for candidate in candidates] == [
        attempt.proposal_id for attempt in accepted
    ]
    for candidate, item in zip(candidates, inputs, strict=True):
        evidence = candidate.relaxation
        assert evidence is not None
        assert evidence.converged and evidence.evaluation_successful
        assert evidence.chart == "symmetry-rigid"
        assert evidence.setting_id == plan.setting.setting_id
        assert candidate.energy is not None and np.isfinite(candidate.energy)
        assert candidate.lineage_id == item.lineage_id

    # Relaxed records keep their packed source as lineage.
    relaxed = []
    for candidate in candidates:
        source = candidate.source_structure_id
        assert source is not None
        relaxed.append(
            crystal_structure_record(
                schema,
                record_id=candidate.structure_id,
                lineage_id=source,
                plan=plan,
                positions=np.asarray(candidate.structure.fractional)
                @ np.asarray(candidate.structure.vectors),
                cell=np.asarray(candidate.structure.vectors),
                stage="relaxed",
                status=0,
                ordinal=candidate.ordinal,
                z_prime=2,
                energy=candidate.energy,
            )
        )
    writer.append(sink.head, AtomisticStructureSource(schema, relaxed))
    final = open_atomistic_structure_store(root, limits=limits)
    lineage = dict(zip(final.record_ids, final.lineage_ids, strict=True))
    for candidate in candidates:
        assert lineage[candidate.structure_id] == candidate.source_structure_id

    # Radial screen, certified persistent dedup and same-definition ranking.
    bank = create_dedup_bank(
        tmp_path / "bank",
        name="crystal-prediction",
        policy=DedupPolicy(
            mode="greedy",
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
        ),
        limits=DedupBankLimits(
            max_commit_bytes=4_000_000,
            max_commits=64,
            max_entries=256,
            max_stored_atoms=10_000,
        ),
    )
    outcome = screen_dedup_rank(
        candidates,
        bank,
        screen=RadialScreenPolicy(
            cutoff=5.0,
            mode="neighbor-typed",
            include_hydrogens=True,
            threshold=0.05,
            max_atoms=64,
            max_images=512,
            max_neighbors=256,
        ),
        ranking=CrystalRankingPolicy(require_converged=True),
        fingerprint_chunk_size=2,
        pair_chunk_size=2,
    )
    ranked = [entry.structure_id for entry in outcome.ranking.entries]
    excluded = {item.structure_id for item in outcome.ranking.excluded}
    assert set(ranked) | excluded == {candidate.structure_id for candidate in candidates}
    assert ranked, "same-definition ranking produced no entry"
    values = [entry.value for entry in outcome.ranking.entries]
    assert values == sorted(values)
    assert outcome.ranking.hamiltonian_id == candidates[0].hamiltonian_id
