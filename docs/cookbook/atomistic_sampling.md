# Distance umbrella and replica exchange

Build the collective variable once, then reuse its wrapped metric in the bias and replica
labels.

```python
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
neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1).prepare(
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
initial_thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
    measure,
    ensemble="nvt",
    temperature=1.0,
).prepare(dynamics)
state = dynamics.initialize_state(
    positions,
    initial_thermodynamic,
    velocity=jnp.zeros_like(positions),
    key=jax.random.key(0),
)

cv = phx.atomistic.sampling.CollectiveVariablePlan(
    phx.atomistic.sampling.CollectiveVariableKind.DISTANCE, [0, 1]
).prepare(system)
cv_program = phx.atomistic.sampling.CollectiveVariableProgram((cv,))
bias = phx.atomistic.sampling.PreparedAtomisticBias(
    phx.atomistic.sampling.AtomisticBiasPlan(
        phx.atomistic.sampling.BiasKind.HARMONIC,
        cv_program,
        center=[1.0],
        stiffness=[2.0],
    ),
    dynamics,
)
bias_state = bias.plan.initialize()
evaluation = bias.evaluate(state.kinematics.positions, bias_state, state.time)
assert bool(evaluation.successful)

thermodynamic = phx.atomistic.PreparedThermodynamicStateTable(
    dynamics,
    (
        phx.atomistic.AtomisticThermodynamicStatePlan(
            measure, ensemble="nvt", temperature=1.0, state_id="low"
        ),
        phx.atomistic.AtomisticThermodynamicStatePlan(
            measure, ensemble="nvt", temperature=2.0, state_id="high"
        ),
    ),
)
lanes = tuple(
    dynamics.initialize_state(
        state.kinematics.positions,
        thermodynamic,
        state_index=index,
        momentum=state.kinematics.momenta,
        key=jax.random.key(4 + index),
    )
    for index in range(2)
)
runtime = phx.atomistic.sampling.AtomisticMultistatePlan(
    thermodynamic,
    [10, 20],
    qualification=phx.atomistic.sampling.AtomisticCanonicalSamplingQualification(
        dynamics,
        thermodynamic,
        "finite-step-cookbook-qualification",
        sampling_exact=False,
        sampling_bias_bound=1.0e-3,
    ),
    exchange=phx.atomistic.sampling.AtomisticReplicaExchangePlan(1),
    run_id="umbrella-replica-example",
).prepare(dynamics)
replicas = runtime.initialize(lanes, [0, 1], jax.random.key(8))
segment = phx.atomistic.sampling.AtomisticMultistateSegmentPlan(
    runtime, 4, 0, 0, runtime.initial_continuation_id
).run(replicas)
dataset = phx.uq.reduced_potential_dataset_from_multistate(segment)
estimate = phx.uq.multistate_bennett_acceptance_ratio(
    dataset, phx.uq.FreeEnergySelectionPlan(block_length=1)
)
assert bool(segment.successful & jnp.all(jnp.isfinite(estimate.free_energies)))
assert not bool(estimate.successful)  # The declared finite-step kernel is approximate.
```

The reduced-potential matrix must contain both cross evaluations. An exchange result with a
missing or nonfinite cross energy is unsuccessful and must not be silently accepted.
