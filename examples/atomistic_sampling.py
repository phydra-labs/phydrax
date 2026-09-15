import jax
import jax.numpy as jnp

import phydrax as phx


system = phx.atomistic.AtomisticSystemPlan(
    [0, 1],
    [1, 1],
    [1.0, 1.0],
    phx.atomistic.AtomisticUnitSystem.reduced(),
    atom_type_ids=[0, 0],
).prepare()
positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(2).prepare(
    system.particles
)
potential = phx.atomistic.AtomisticPotentialProgram(
    [phx.atomistic.LennardJonesPotential([0.2], [1.0], 2.5)]
).prepare(system)
dynamics = phx.atomistic.AtomisticDynamicsPlan(
    system,
    potential,
    neighborhood,
    phx.atomistic.VelocityVerletPlan(1.0e-3),
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
bias = phx.atomistic.sampling.PreparedAtomisticBias(
    phx.atomistic.sampling.AtomisticBiasPlan(
        phx.atomistic.sampling.BiasKind.HARMONIC,
        program,
        center=[1.0],
        stiffness=[2.0],
    ),
    dynamics,
)
bias_evaluation = bias.evaluate(
    positions, bias.plan.initialize(positions.dtype), state.time
)
replica_plan = phx.atomistic.sampling.AtomisticMultistatePlan(
    thermodynamic,
    [0, 1],
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
    [0, 1],
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
    bias_evaluation.successful
    & segment.successful
    & jnp.all(jnp.isfinite(estimate.free_energies))
):
    raise RuntimeError("enhanced-sampling example failed numerically")
if bool(estimate.successful):
    raise RuntimeError("finite-step demonstration must retain approximate status")
print("distance", float(bias_evaluation.variables[0]))
print("bias energy", float(bias_evaluation.energy))
print("free-energy difference", float(estimate.free_energies[1]))
