import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


system = phx.atomistic.AtomisticSystemPlan(
    np.asarray([0, 1], dtype=np.int64),
    np.asarray([1, 1], dtype=np.int32),
    np.asarray([1.0, 1.0], dtype=np.float64),
    phx.atomistic.AtomisticUnitSystem.reduced(),
    atom_type_ids=np.asarray([0, 0], dtype=np.int32),
).prepare()
positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
program = phx.atomistic.AtomisticPotentialProgram(
    (phx.atomistic.LennardJonesPotential(np.asarray([0.2]), np.asarray([1.0]), 2.5),)
)
(pairs,) = program.requirements.neighborhoods
# Finite system: a bounded dense image source searches an explicit
# nonperiodic frame; one unordered pair needs one half-counted route.
bundle = phx.atomistic.AtomisticNeighborhoodBundlePlan(
    (pairs,),
    (
        phx.atomistic.AtomisticNeighborhoodSource(
            "lennard-jones-pairs",
            phx.atomistic.AtomisticSiteDomain.PHYSICAL_ATOMS,
            (pairs,),
            phx.atomistic.AtomisticDenseImageSearch(maximum_dense_routes=4),
            phx.discretization.ParticleImageCapacityLadder(
                (
                    phx.discretization.ParticleImageCapacity(
                        maximum_particles_per_cell=2,
                        maximum_edges=2,
                        maximum_degree=1,
                        maximum_images=1,
                        maximum_case_edges=2,
                    ),
                )
            ),
            skin=0.3,
        ),
    ),
    (
        phx.atomistic.AtomisticRoleCapacityLadder(
            pairs,
            (phx.atomistic.AtomisticRoleCapacity(maximum_routes=1, maximum_degree=1),),
        ),
    ),
    frame=phx.discretization.PeriodicCell(
        16.0 * jnp.eye(3), origin=jnp.full((3,), -8.0), periodic_axes=(False,) * 3
    ),
)
potential = program.prepare(system, neighborhood_bundle=bundle)
# Canonical NVT replicas need a canonical thermostat; velocity Verlet is NVE only.
dynamics = phx.atomistic.AtomisticDynamicsPlan(
    system, potential, phx.atomistic.BAOABLangevinPlan(1.0e-3, 1.0)
).prepare()
measure = phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system)
thermodynamic = phx.atomistic.PreparedThermodynamicStateTable(
    dynamics,
    (
        phx.atomistic.AtomisticThermodynamicStatePlan(
            measure, ensemble="nvt", temperature=1.0, state_id="sampling-low"
        ),
        phx.atomistic.AtomisticThermodynamicStatePlan(
            measure, ensemble="nvt", temperature=2.0, state_id="sampling-high"
        ),
    ),
)
state = dynamics.initialize_state(
    positions,
    thermodynamic,
    state_index=0,
    velocity=jnp.zeros_like(positions),
    key=jax.random.key(0),
)
cv = phx.atomistic.sampling.CollectiveVariablePlan(
    phx.atomistic.sampling.CollectiveVariableKind.DISTANCE, [0, 1]
).prepare(system)
program = phx.atomistic.sampling.CollectiveVariableProgram((cv,))
bias = phx.atomistic.sampling.AtomisticBiasTerm(
    program,
    phx.atomistic.sampling.HarmonicBiasForm(jnp.asarray([1.0]), 2.0),
    name="distance-umbrella",
    force_group=1,
).prepare(system)
bias_energy, bias_cvs, bias_successful = bias.bias(
    phx.atomistic.sampling.CollectiveVariableGeometry(positions),
    bias.initial_term_state(positions.dtype),
)
replica_plan = phx.atomistic.sampling.AtomisticMultistatePlan(
    thermodynamic,
    np.asarray([0, 1], dtype=np.int64),
    qualification=phx.atomistic.sampling.AtomisticCanonicalSamplingQualification(
        dynamics,
        thermodynamic,
        "atomistic-sampling-demonstration-qualification",
        sampling_exact=False,
        sampling_bias_bound=1.0,
    ),
    exchange=phx.atomistic.sampling.AtomisticReplicaExchangePlan(1),
    run_id="atomistic-sampling-example",
).prepare(dynamics)
replicas = replica_plan.initialize(
    (
        state,
        dynamics.initialize_state(
            positions,
            thermodynamic,
            state_index=1,
            velocity=jnp.zeros_like(positions),
            key=jax.random.key(1),
        ),
    ),
    np.asarray([0, 1], dtype=np.int32),
    jax.random.key(2),
)
segment = phx.atomistic.sampling.AtomisticMultistateSegmentPlan(
    replica_plan,
    4,
    0,
    0,
    replica_plan.initial_continuation_id,
).run(replicas)
dataset = phx.uq.reduced_potential_dataset_from_multistate(segment)
estimate = phx.uq.multistate_bennett_acceptance_ratio(
    dataset,
    phx.uq.FreeEnergySelectionPlan(block_length=1),
)
if not bool(
    bias_successful & segment.successful & jnp.all(jnp.isfinite(estimate.free_energies))
):
    raise RuntimeError("enhanced-sampling example failed numerically")
if bool(estimate.successful):
    raise RuntimeError("finite-step demonstration must retain approximate status")
print("distance", float(bias_cvs.values[0]))
print("bias energy", float(bias_energy))
print("free-energy difference", float(estimate.free_energies[1]))
