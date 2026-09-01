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
state = dynamics.initialize_state(
    positions, velocity=jnp.zeros_like(positions), key=jax.random.key(0)
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
replica_plan = phx.atomistic.sampling.AtomisticReplicaEnsemblePlan([1.0, 2.0])
replicas = phx.atomistic.sampling.initialize_replica_state(
    replica_plan,
    jnp.stack((positions, positions)),
    jnp.stack((state.kinematics.momenta, state.kinematics.momenta)),
    [[0.0, 1.0], [1.0, 0.0]],
    jax.random.key(1),
)
exchange = phx.atomistic.sampling.replica_exchange_step(replica_plan, replicas, 1.0)
estimate = phx.uq.free_energy_perturbation([0.0, 0.1, -0.1, 0.0])
if not bool(bias_evaluation.successful & exchange.successful & estimate.converged):
    raise RuntimeError("enhanced-sampling example failed")
print("distance", float(bias_evaluation.variables[0]))
print("bias energy", float(bias_evaluation.energy))
print("free-energy difference", float(estimate.free_energies[1]))
