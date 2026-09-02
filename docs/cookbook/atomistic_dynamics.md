# Run a bounded periodic atomistic trajectory

Prepare one explicit reduced unit system, fixed periodic cell, atomistic support, and
short-range potential program:

```python
import jax.random as jr
import jax.numpy as jnp
import phydrax as phx

units = phx.atomistic.AtomisticUnitSystem.reduced()
cell = phx.discretization.PeriodicCell(4.0 * jnp.eye(3))
system = phx.atomistic.AtomisticSystemPlan(
    [100, 101, 102, 103],
    [1, 1, 1, 1],
    [1.0, 1.0, 1.0, 1.0],
    units,
    atom_type_ids=[0, 0, 0, 0],
    cell=cell,
).prepare()
potential = phx.atomistic.AtomisticPotentialProgram(
    [phx.atomistic.LennardJonesPotential([0.2], [0.8], 1.5)]
).prepare(system)
```

Declare the maximum cell and pair resources. Verlet owns a candidate radius equal to the
interaction radius plus skin and rebuilds only after its displacement certificate expires:

```python
base = phx.discretization.MetricCellListParticleNeighborhoodPlan(
    1.7, 4, 6, cell
)
neighborhood = phx.discretization.VerletParticleNeighborhoodPlan(
    base, 1.5, 0.2
).prepare(system.particles)
dynamics = phx.atomistic.AtomisticDynamicsPlan(
    system,
    potential,
    neighborhood,
    phx.atomistic.BAOABLangevinPlan(2e-4, 1.0, 0.2),
).prepare()
```

Initialize with exactly one of velocity or momentum, then run a fixed-capacity trajectory:

```python
positions = cell.cartesian(
    jnp.asarray(
        [[0.10, 0.10, 0.10], [0.35, 0.10, 0.10],
         [0.10, 0.35, 0.10], [0.35, 0.35, 0.10]]
    )
)
state = dynamics.initialize_state(
    positions, velocity=jnp.zeros_like(positions), key=jr.key(0)
)
rollout = phx.atomistic.AtomisticRolloutPlan(
    dynamics,
    phx.atomistic.AtomisticTrajectoryPlan(100, sample_stride=10),
    replay=phx.atomistic.AtomisticReplayPolicy("step"),
)
result = rollout.rollout(state)
if not bool(result.successful):
    raise RuntimeError("atomistic rollout rejected a step")
```

The trajectory capacity is resolved before execution. `result.replay` records accepted and
rejected counts plus route, image, and stochastic digests. Persist exact continuation state
with `AtomisticCheckpointPlan`, `write_atomistic_checkpoint`, and
`read_atomistic_checkpoint`.

For trained PaiNN or NequIP dynamics, wrap the checkpointed model in
`LearnedGraphPotentialTerm` and supply a particle `AtomisticGraphExecutionPlan` while
preparing the potential program. `allow_periodic=True` only enables periodic graph geometry;
it does not certify a fitted model's rollout stability.
