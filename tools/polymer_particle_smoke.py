from __future__ import annotations

import json

import jax
import jax.numpy as jnp

import phydrax as phx


def main() -> int:
    cell = phx.discretization.PeriodicCell(jnp.eye(3) * 8.0)
    topology = phx.atomistic.MolecularTopologyPlan(
        bonds=[[10, 20], [20, 30]], bond_type_ids=[0, 0]
    )
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [10, 20, 30],
        # ty: ignore[invalid-argument-type]
        [0, 0, 0],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0, 1.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0, 0],
        # ty: ignore[invalid-argument-type]
        element_mask=[False, False, False],
        # ty: ignore[invalid-argument-type]
        molecule_ids=[0, 0, 0],
        topology=topology,
        cell=cell,
    ).prepare()
    cutoff = 2.0 ** (1.0 / 6.0)
    program = phx.atomistic.AtomisticPotentialProgram(
        [
            # ty: ignore[invalid-argument-type]
            phx.atomistic.FiniteExtensibleNonlinearElasticBondPotential([30.0], [1.5]),
            phx.atomistic.LennardJonesPotential(
                # ty: ignore[invalid-argument-type]
                [1.0],
                # ty: ignore[invalid-argument-type]
                [1.0],
                cutoff,
                shift_energy_at_cutoff=True,
            ),
        ]
    )
    (pairs,) = program.requirements.neighborhoods
    bundle = phx.atomistic.AtomisticNeighborhoodBundlePlan(
        (pairs,),
        (
            phx.atomistic.AtomisticNeighborhoodSource(
                "wca-pairs",
                phx.atomistic.AtomisticSiteDomain.PHYSICAL_ATOMS,
                (pairs,),
                phx.atomistic.AtomisticDenseImageSearch(maximum_dense_routes=243),
                phx.discretization.ParticleImageCapacityLadder(
                    (
                        phx.discretization.ParticleImageCapacity(
                            maximum_particles_per_cell=3,
                            maximum_edges=6,
                            maximum_degree=2,
                            maximum_images=27,
                            maximum_case_edges=6,
                        ),
                    )
                ),
                skin=0.3,
            ),
        ),
        (
            phx.atomistic.AtomisticRoleCapacityLadder(
                pairs,
                (
                    phx.atomistic.AtomisticRoleCapacity(
                        maximum_routes=3, maximum_degree=2
                    ),
                ),
            ),
        ),
    )
    potential = program.prepare(system, neighborhood_bundle=bundle)
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        phx.atomistic.BAOABLangevinPlan(1.0e-3, 1.0, realization_id=4),
    ).prepare()
    thermo = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system),
        ensemble="nvt",
        temperature=1.0,
    ).prepare(dynamics)
    layout = phx.atomistic.PolymerChainLayoutPlan(
        # ty: ignore[invalid-argument-type]
        [[0, 1, 2]],
        # ty: ignore[invalid-argument-type]
        [[True, True, True]],
        maximum_frames=2,
    )
    profile = phx.applications.polymer_liquids.KremerGrestProfilePlan(
        production_steps=2, maximum_particles=3, maximum_chains=1, minimum_fene_margin=0.1
    ).prepare(dynamics, layout)
    positions = jnp.asarray([[1.0, 1.0, 1.0], [1.96, 1.0, 1.0], [2.92, 1.0, 1.0]])
    state = dynamics.initialize_state(
        positions, thermo, velocity=jnp.zeros_like(positions), key=jax.random.key(4)
    )
    evidence = phx.applications.polymer_liquids.kremer_grest_evidence(
        profile, state, thermo
    )
    stepped = dynamics.step_detailed(state, thermo)
    successful = bool(evidence.successful & stepped.successful)
    print(
        json.dumps(
            {
                "successful": successful,
                "maximum_bond_fraction": float(evidence.maximum_bond_fraction),
            }
        )
    )
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
