# Distance umbrella and replica exchange

Build the collective variable once, then reuse its wrapped metric in the bias and replica
labels.

```python
import jax
import jax.numpy as jnp
import phydrax as phx

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

replica_plan = phx.atomistic.sampling.AtomisticReplicaEnsemblePlan([1.0, 2.0])
replicas = phx.atomistic.sampling.initialize_replica_state(
    replica_plan,
    jnp.stack((state.kinematics.positions, state.kinematics.positions)),
    jnp.stack((state.kinematics.momenta, state.kinematics.momenta)),
    [[0.0, 1.0], [1.0, 0.0]],
    jax.random.key(4),
)
replicas = phx.atomistic.sampling.replica_exchange_step(replica_plan, replicas, 1.0)
```

The reduced-potential matrix must contain both cross evaluations. An exchange result with a
missing or nonfinite cross energy is unsuccessful and must not be silently accepted.
