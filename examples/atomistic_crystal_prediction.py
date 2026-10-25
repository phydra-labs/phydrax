# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native molecular crystal prediction from a molecular graph.

The workflow runs entirely on native Phydrax owners:

1. a declared CHFClBr graph with explicit stereo, masses and admitted radii;
2. native distance-geometry conformers;
3. general-position packing in P1 and P2_1 (Sohncke settings, so the declared
   enantiomer is never mirrored) under a declared uniform prior, published as
   a persisted packing recipe and re-admitted from it (the packer and
   molecule below come from the admitted recipe), with every attempt
   published to a structure store;
4. rigid-molecule, symmetry-constrained relaxation of two independent molecules
   per ASU with a bounded minimum-image Lennard-Jones test Hamiltonian;
5. radial screening, certified persistent deduplication and same-definition
   ranking.

The Lennard-Jones parameters and the uniform setting prior are explicit
demonstration inputs, not an admitted force field or CSD prior: the printed
ranking shows the mechanics and evidence, not chemical accuracy or
finite-temperature polymorph stability.

Run: ``python examples/atomistic_crystal_prediction.py [--output DIR]``.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import jax.random as jr
import numpy as np
from jax import Array

import phydrax as phx
from phydrax.atomistic import (
    AtomisticDenseImageSearch,
    AtomisticNeighborhoodBundlePlan,
    AtomisticNeighborhoodSource,
    AtomisticRelaxationConvergence,
    AtomisticRoleCapacity,
    AtomisticRoleCapacityLadder,
    AtomisticSiteDomain,
    AtomisticStoreLayout,
    AtomisticStoreLimits,
    AtomisticStructureStoreGenerationSink,
    AtomisticStructureStoreWriter,
    AtomisticSymmetryChart,
    open_atomistic_structure_store,
    PreparedAtomisticPotentialProgram,
)
from phydrax.atomistic.crystal import (
    AsymmetricUnitPlan,
    ConformerGenerationPlan,
    ConformerGeometryTable,
    ContactPolicy,
    create_dedup_bank,
    crystal_structure_schema,
    CrystalCellPolicy,
    CrystalContactResources,
    CrystalPackingPlan,
    CrystalPackingRecipe,
    CrystalRankingPolicy,
    CrystalRelaxationInput,
    CrystalReliefPolicy,
    CrystalSpaceGroupPolicy,
    DedupBankLimits,
    DedupPolicy,
    ElementTable,
    EquivalencePolicy,
    MolecularComponent,
    MolecularCrystalInput,
    native_conformer_pools,
    prepare_conformer_constraints,
    PreparedCrystalPacker,
    RadialScreenPolicy,
    read_crystal_packing_recipe,
    relax_crystal_candidates,
    RigidSymmetryLayout,
    screen_dedup_rank,
    standard_space_group_setting,
    VolumePolicy,
    write_crystal_packing_recipe,
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
FORMULA_UNIT_VOLUME = 180.0
LJ_CUTOFF = 3.2


def _capacity(width: int, *, cases: int = 1) -> ParticleImageCapacity:
    return ParticleImageCapacity(
        maximum_particles_per_cell=64,
        maximum_edges=cases * width,
        maximum_degree=width,
        maximum_images=4096,
        maximum_case_edges=width,
    )


def _packer(lanes: int) -> PreparedCrystalPacker:
    component = MolecularComponent(
        (6, 1, 9, 17, 35),
        ((0, 1), (0, 2), (0, 3), (0, 4)),
        bond_orders=(1, 1, 1, 1),
        masses=MASSES,
        spin_multiplicity=1,
        tetrahedral_stereo=((0, 1, 2, 3, 1),),
        name="chlorofluorobromomethane",
    )
    table = ConformerGeometryTable(
        ElementTable(
            "covalent-radius",
            {1: 0.32, 6: 0.75, 9: 0.64, 17: 0.99, 35: 1.14},
            provenance=PYYKKO,
        ),
        CONTACT_RADII,
    )
    conformers = ConformerGenerationPlan(
        component, prepare_conformer_constraints(component, table), max_attempts=16
    )
    pools, _ = native_conformer_pools([conformers], jr.key(0), count=3)
    molecular = MolecularCrystalInput(
        pools,
        contact=ContactPolicy(CONTACT_RADII),
        volume=VolumePolicy(formula_unit_volume=FORMULA_UNIT_VOLUME),
    )
    width = 8192
    groups = CrystalSpaceGroupPolicy(
        [standard_space_group_setting(number) for number in (1, 4)],
        weights=(1.0, 1.0),
        provenance="uniform demonstration prior (not a CSD frequency table)",
    ).filter(sohncke_only=True)
    plan = CrystalPackingPlan(
        molecular,
        groups,
        z_prime=2,
        chirality="enantiopure",
        resources=CrystalContactResources(
            AtomisticDenseImageSearch(maximum_dense_routes=lanes * 1_000_000),
            search_capacities=(_capacity(width, cases=lanes),),
            role_capacities=(
                AtomisticRoleCapacity(maximum_routes=lanes * width, maximum_degree=width),
            ),
            skin=0.5,
        ),
        cell=CrystalCellPolicy(
            volume_range=(1.0, 1.3),
            # This minimum-image cutoff leaves little room for anisotropy or
            # isotropic compression at the declared volume per formula unit.
            oversample=1.0,
            maximum_axis_ratio=1.5,
            angle_range=(75.0, 105.0),
            minimum_height=2.0 * LJ_CUTOFF + 0.1,
            minimum_height_floor=2.0 * LJ_CUTOFF + 0.05,
        ),
        relief=CrystalReliefPolicy(
            maximum_steps=400, check_interval=10, overlap_tolerance=0.1
        ),
    )
    return plan.prepare(lanes=lanes)


def _program(
    plan: AsymmetricUnitPlan, component: MolecularComponent, cell: Array
) -> PreparedAtomisticPotentialProgram:
    count = plan.atom_count
    bonds = [
        (
            int(np.flatnonzero(plan.atom_copy == copy)[a]) + 1,
            int(np.flatnonzero(plan.atom_copy == copy)[b]) + 1,
        )
        for copy in range(plan.copy_count)
        for a, b in component.bonds
    ]
    system = phx.atomistic.AtomisticSystemPlan(
        np.arange(1, count + 1),
        np.asarray(
            [component.atomic_numbers[int(i)] for i in plan.atom_local], dtype=np.int32
        ),
        np.asarray([component.masses[int(i)] for i in plan.atom_local]),
        phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond(),
        atom_type_ids=np.zeros((count,), dtype=np.int32),
        topology=phx.atomistic.MolecularTopologyPlan(
            bonds=[list(bond) for bond in bonds]
        ),
        cell=PeriodicCell(np.asarray(cell)),
    ).prepare()
    program = phx.atomistic.AtomisticPotentialProgram(
        [
            phx.atomistic.LennardJonesPotential(
                np.asarray([0.01]),
                np.asarray([2.0]),
                LJ_CUTOFF,
                switch_distance=2.9,
            )
        ]
    )
    (requirement,) = program.requirements.neighborhoods
    source = AtomisticNeighborhoodSource(
        "pairs",
        AtomisticSiteDomain.PHYSICAL_ATOMS,
        (requirement,),
        AtomisticDenseImageSearch(maximum_dense_routes=4_000_000, deformation_margin=1.0),
        ParticleImageCapacityLadder((_capacity(32768),)),
        skin=0.5,
    )
    bundle = AtomisticNeighborhoodBundlePlan(
        (requirement,),
        (source,),
        (
            AtomisticRoleCapacityLadder(
                requirement,
                (AtomisticRoleCapacity(maximum_routes=32768, maximum_degree=4096),),
            ),
        ),
    )
    return program.prepare(system, neighborhood_bundle=bundle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--attempts", type=int, default=16)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp()) if args.output is None else args.output
    root.mkdir(parents=True, exist_ok=True)
    written = write_crystal_packing_recipe(
        root / "packing.recipe",
        CrystalPackingRecipe(_packer(lanes=2), campaign_id="crystal-prediction"),
    )
    recipe, admitted = read_crystal_packing_recipe(root / "packing.recipe")
    if admitted.content_id != written.content_id:
        raise RuntimeError("The admitted recipe differs from the published one.")
    print(
        f"recipe: {admitted.recipe_id[:12]} plan {admitted.plan_id[:12]} "
        f"content {admitted.content_id[:12]}"
    )
    packer = recipe.packer
    component = recipe.plan.molecular_input.pools[0].component
    limits = AtomisticStoreLimits(
        max_commit_bytes=1 << 21,
        max_member_bytes=1 << 22,
        max_shard_bytes=1 << 24,
        max_shard_members=8192,
        max_chunk_bytes=1 << 20,
        max_index_bytes=1 << 21,
        max_decoded_bytes=1 << 24,
        max_read_bytes=1 << 24,
        max_records=10_000,
    )
    writer = AtomisticStructureStoreWriter(
        root / "store",
        layout=AtomisticStoreLayout(
            records_per_shard=16, target_chunk_bytes=8192, compression="none"
        ),
        limits=limits,
    )
    writer.create(crystal_structure_schema(), name="crystal-prediction")
    sink = AtomisticStructureStoreGenerationSink(
        root / "store", writer, packer.record_builder, limits=limits
    )
    result, _ = packer.pack(
        jr.key(1),
        campaign_id=recipe.campaign_id,
        target=4,
        attempt_budget=args.attempts,
        sink=sink,
    )
    print(
        f"packing: {len(result.attempts)} attempts, {len(result.accepted)} accepted, "
        f"stop={result.stop_reason}, shortfall={result.shortfall}"
    )
    print(
        f"store: {len(open_atomistic_structure_store(root / 'store', limits=limits).record_ids)} records"
    )
    bank = create_dedup_bank(
        root / "bank",
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
            max_commit_bytes=8_000_000,
            max_commits=64,
            max_entries=1024,
            max_stored_atoms=100_000,
        ),
    )
    candidates = []
    for group in sorted({attempt.group_index for attempt in result.accepted}):
        members = [attempt for attempt in result.accepted if attempt.group_index == group]
        plan = members[0].plan
        program = _program(plan, component, members[0].expansion.cell.vectors)
        atoms = np.arange(plan.atom_count)
        candidates.extend(
            relax_crystal_candidates(
                plan,
                [
                    CrystalRelaxationInput(
                        attempt.state,
                        attempt.expansion,
                        structure_id=attempt.proposal_id,
                        lineage_id=attempt.proposal_id,
                        ordinal=attempt.ordinal,
                    )
                    for attempt in members
                ],
                program,
                AtomisticSymmetryChart(RigidSymmetryLayout(plan), atoms),
                system_atoms=atoms,
                method=TrustScaledLBFGS(maximum_step=0.02),
                convergence=AtomisticRelaxationConvergence(
                    projected_tolerance=1e-4,
                    maximum_evaluations=5000,
                ),
                maximum_rounds=5000,
            )
        )
    if not candidates:
        raise RuntimeError("The declared packing budget produced no candidate to relax.")
    for candidate in candidates:
        evidence = candidate.relaxation
        if evidence is None:
            raise RuntimeError("A relaxed candidate lost its native solver evidence.")
        print(
            f"relaxation {candidate.ordinal}: {evidence.status}, "
            f"evaluations={evidence.evaluations}, "
            f"projected_residual={evidence.projected_residual:.6g}, "
            f"evaluation_successful={evidence.evaluation_successful}"
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
        fingerprint_chunk_size=4,
        pair_chunk_size=4,
    )
    for entry in outcome.ranking.entries:
        print(
            f"rank {entry.rank}: {entry.structure_id[:12]}  {entry.value:.6f} eV/formula-unit  (+{entry.relative:.6f})"
        )
    for item in outcome.ranking.excluded:
        print(f"excluded {item.structure_id[:12]}: {item.reason}")
    print(outcome.ranking.nonclaim)
    if not outcome.ranking.entries:
        raise RuntimeError(
            "No candidate met the screening, deduplication and converged-ranking contract."
        )


if __name__ == "__main__":
    main()
