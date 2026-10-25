import jax
import jax.numpy as jnp

import phydrax as phx


units = phx.atomistic.AtomisticUnitSystem.reduced()
system_plan = phx.atomistic.AtomisticSystemPlan(
    # ty: ignore[invalid-argument-type]
    [10, 20, 30],
    # ty: ignore[invalid-argument-type]
    [1, 1, 1],
    # ty: ignore[invalid-argument-type]
    [1.0, 1.0, 1.0],
    units,
    # ty: ignore[invalid-argument-type]
    atom_type_ids=[0, 0, 1],
    # ty: ignore[invalid-argument-type]
    charges=[1.0, -1.0, 0.0],
    # ty: ignore[invalid-argument-type]
    region_ids=[1, 1, 0],
)
program = phx.atomistic.AtomisticPotentialProgram(
    [
        # ty: ignore[invalid-argument-type]
        phx.atomistic.LennardJonesPotential([0.4, 0.8], [1.0, 1.2], 3.0),
        phx.atomistic.DirectCoulombPotential(),
    ]
)
# Bounded steric pairs come from a dense image search in an explicit finite
# frame; direct Coulomb is the complete all-pairs relation (3 atoms, 3 pairs).
(sterics,) = tuple(
    value for value in program.requirements.neighborhoods if not value.complete
)
(coulomb,) = tuple(
    value for value in program.requirements.neighborhoods if value.complete
)
bundle = phx.atomistic.AtomisticNeighborhoodBundlePlan(
    (sterics, coulomb),
    (
        phx.atomistic.AtomisticNeighborhoodSource(
            "steric-pairs",
            phx.atomistic.AtomisticSiteDomain.PHYSICAL_ATOMS,
            (sterics,),
            phx.atomistic.AtomisticDenseImageSearch(maximum_dense_routes=9),
            phx.discretization.ParticleImageCapacityLadder(
                (
                    phx.discretization.ParticleImageCapacity(
                        maximum_particles_per_cell=3,
                        maximum_edges=6,
                        maximum_degree=2,
                        maximum_images=1,
                        maximum_case_edges=6,
                    ),
                )
            ),
            skin=0.3,
        ),
        phx.atomistic.AtomisticCompletePairSource(
            "coulomb-pairs",
            phx.atomistic.AtomisticSiteDomain.PHYSICAL_ATOMS,
            (coulomb,),
            maximum_pairs=3,
        ),
    ),
    (
        phx.atomistic.AtomisticRoleCapacityLadder(
            sterics,
            (phx.atomistic.AtomisticRoleCapacity(maximum_routes=3, maximum_degree=2),),
        ),
    ),
    frame=phx.discretization.PeriodicCell(
        16.0 * jnp.eye(3), origin=jnp.full((3,), -8.0), periodic_axes=(False,) * 3
    ),
)
force_field = phx.atomistic.AtomisticForceFieldPlan(
    system_plan,
    program,
    phx.atomistic.AtomisticNonbondedPolicy(3.0, electrostatics="direct"),
    phx.atomistic.AtomisticForceFieldProvenance(
        "native",
        ("example-parameters",),
        "example",
        "controlled-free-energy",
    ),
).prepare(neighborhood_bundle=bundle)
schedule = phx.atomistic.AlchemicalControlSchedulePlan(
    ("coupled", "intermediate", "decoupled"),
    ("solute-sterics", "solute-electrostatics"),
    (
        phx.atomistic.AlchemicalControlKind.STERICS,
        phx.atomistic.AlchemicalControlKind.ELECTROSTATICS,
    ),
    jnp.asarray([[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]]),
)
partition = phx.atomistic.AlchemicalInteractionPartitionPlan(
    schedule.control_ids,
    # ty: ignore[invalid-argument-type]
    ([10, 20], [10, 20]),
    # ty: ignore[invalid-argument-type]
    mapped_particle_ids=[[10, 10], [20, 20]],
)
hamiltonian = phx.atomistic.ControlledHamiltonianPlan(
    force_field,
    schedule,
    partition,
).prepare()
# Canonical NVT lanes need a canonical thermostat; velocity Verlet is NVE only.
dynamics = phx.atomistic.AtomisticDynamicsPlan(
    force_field.system,
    hamiltonian,
    phx.atomistic.BAOABLangevinPlan(1.0e-5, 1.0),
).prepare()
measure = phx.atomistic.AtomisticPhaseSpaceMeasurePlan(force_field.system)
state_plans = tuple(
    phx.atomistic.AtomisticThermodynamicStatePlan(
        measure,
        ensemble="nvt",
        temperature=1.0,
        controls=schedule.controls[index],
        control_ids=schedule.control_ids,
        state_id=state_id,
    )
    for index, state_id in enumerate(schedule.state_ids)
)
thermodynamic = phx.atomistic.PreparedThermodynamicStateTable(dynamics, state_plans)
positions = jnp.asarray([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.2, 0.2, 0.0]])
lanes = tuple(
    dynamics.initialize_state(
        positions,
        thermodynamic,
        state_index=index,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(10 + index),
    )
    for index in range(thermodynamic.state_count)
)
runtime = phx.atomistic.sampling.AtomisticMultistatePlan(
    thermodynamic,
    # ty: ignore[invalid-argument-type]
    [100, 200, 300],
    qualification=phx.atomistic.sampling.AtomisticCanonicalSamplingQualification(
        dynamics,
        thermodynamic,
        "controlled-free-energy-demonstration-qualification",
        sampling_exact=False,
        sampling_bias_bound=1.0,
    ),
    exchange=phx.atomistic.sampling.AtomisticReplicaExchangePlan(1),
    run_id="controlled-free-energy-example",
).prepare(dynamics)
# ty: ignore[invalid-argument-type]
initial = runtime.initialize(lanes, [0, 1, 2], jax.random.key(21))
segment = phx.atomistic.sampling.AtomisticMultistateSegmentPlan(
    runtime,
    8,
    0,
    0,
    runtime.initial_continuation_id,
).run(initial)
dataset = phx.uq.reduced_potential_dataset_from_multistate(segment)
result = phx.uq.multistate_bennett_acceptance_ratio(
    dataset,
    phx.uq.FreeEnergySelectionPlan(block_length=1),
)
if not bool(segment.successful & jnp.all(jnp.isfinite(result.free_energies))):
    raise RuntimeError("controlled multistate free-energy example failed numerically")
if bool(result.successful):
    raise RuntimeError("finite-step demonstration must retain approximate status")
print(
    {
        "states": result.state_ids,
        "free_energies": result.free_energies.tolist(),
        "standard_errors": result.standard_errors.tolist(),
        "minimum_overlap": float(jnp.min(result.overlap)),
    }
)
