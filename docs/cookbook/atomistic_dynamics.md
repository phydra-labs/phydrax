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
    phx.atomistic.BAOABLangevinPlan(2e-4, 0.2),
).prepare()
measure = phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system)
thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
    measure,
    ensemble="nvt",
    temperature=1.0,
).prepare(dynamics)
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
    positions,
    thermodynamic,
    velocity=jnp.zeros_like(positions),
    key=jr.key(0),
)
rollout = phx.atomistic.AtomisticRolloutPlan(
    dynamics,
    thermodynamic,
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

For trained PaiNN, NequIP, or native MACE dynamics, wrap the trained model in
`LearnedGraphPotentialTerm` and supply a particle `AtomisticGraphExecutionPlan` while
preparing the potential program. `allow_periodic=True` only enables periodic graph geometry;
it does not certify a fitted model's rollout stability. A cutoff beyond the unique-image
radius needs an image-aware neighborhood, as in the recipe below.

## Periodic native MACE dynamics

This recipe follows `examples/periodic_mace_dynamics.py`. Two water-like molecules sit in
a triclinic cell whose unique-image radius is smaller than the 3 Å cutoff, so a
pair-once minimum-image neighborhood would be refused. The image-aware graph enumerates
every translation within `cutoff + skin`, including repeated images of one pair when the
geometry places them inside the cutoff. The model is a random native initialization,
not a fitted or released potential.

```python
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import phydrax as phx

units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
cell = np.array([[4.6, 0.0, 0.0], [0.9, 4.4, 0.0], [0.5, -0.6, 4.8]])
positions = jnp.asarray(
    [
        [0.2, 0.3, 0.1],
        [1.16, 0.3, 0.1],
        [-0.04, 1.23, 0.1],
        [2.5, 2.4, 2.6],
        [3.46, 2.4, 2.6],
        [2.26, 3.33, 2.6],
    ]
)
numbers = jnp.asarray([8, 1, 1, 8, 1, 1])
masses = jnp.asarray([15.999, 1.008, 1.008, 15.999, 1.008, 1.008])

architecture = phx.nn.atomistic.MACEArchitecture(
    species=(1, 8),
    cutoff=3.0,
    radial_basis_count=4,
    cutoff_power=5,
    channel_count=4,
    hidden_degree=1,
    edge_degree=2,
    interactions=("real-agnostic", "real-agnostic-residual"),
    correlations=(2, 2),
    radial_widths=(8,),
    readout_width=4,
    average_neighbor_count=6.0,
)
model = phx.nn.atomistic.MACEPotential(
    units.scale, architecture, atomic_energies=[[-1.0, -2.0]], key=jax.random.key(0)
)
```

`NativeAtomisticProviderPlan` prepares the model as one potential program. For a
periodic system, it also prepares an image-aware Verlet neighborhood over a cell-list
image search of radius `cutoff + skin`. `ParticleImageCapacity` charges cell
occupancy, stored edges, receiver degree, and image translations separately. The same
program and neighborhood drive fixed-cell NVE dynamics:

```python
def prepare(model, image_capacity):
    structure = phx.atomistic.AtomicStructure(
        numbers,
        positions,
        masses,
        units.scale,
        cell=jnp.asarray(cell),
        periodic_axes=jnp.asarray([True, True, True]),
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(structure, units).prepare()
    execution = phx.atomistic.AtomisticGraphExecutionPlan(
        64, backend="particle", image_capacity=image_capacity
    )
    provider = phx.atomistic.NativeAtomisticProviderPlan(
        model,
        execution,
        finite_neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(64),
        skin=0.4,
    ).prepare(system)
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        provider.program,
        provider.neighborhood,
        phx.atomistic.VelocityVerletPlan(0.25),
    ).prepare()
    state_plan = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system), ensemble="nve"
    )
    return provider, dynamics, state_plan, state_plan.prepare(dynamics)


capacity = phx.discretization.ParticleImageCapacity(
    maximum_particles_per_cell=16,
    maximum_edges=2048,
    maximum_degree=96,
    maximum_images=343,
)
provider, dynamics, state_plan, thermodynamic = prepare(model, capacity)

evaluation = provider.evaluate_state(positions, None, None).evaluation
energy = evaluation.energy  # eV
stress = evaluation.stress  # eV / Å^3, tension positive, H' = H @ F.T

velocity = 0.002 * jax.random.normal(jax.random.key(1), positions.shape)
state = dynamics.initialize_state(
    positions, thermodynamic, velocity=velocity, key=jax.random.key(2)
)
for _ in range(20):
    state = dynamics.step(state, thermodynamic)
```

The provider reports stress because the cell is fully periodic 3D and the learned term
owns a cell derivative. Any self-image routes contribute to the stress even though their
net positional force cancels. `state.energy.cumulative_balance_residual` tracks the NVE
energy balance. `BAOABLangevinPlan` with an `ensemble="nvt"` state supplies fixed-cell
NVT. No NPT or other dynamic-cell method is admitted for learned graph terms.

A portable restart bundles the pickle-free native model artifact with the complete
dynamics state. A fresh process restores the model and rebuilds its dynamics over it,
then resumes. An altered model, system, integrator, thermodynamic table, or graph
preparation is refused, so the rebuilt preparation must match the one that wrote the
restart, image capacity included:

```python
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "restart"
    plan = phx.atomistic.AtomisticCheckpointPlan(dynamics, thermodynamic)
    phx.atomistic.write_atomistic_restart(path, plan, state, model=model)

    restored_model = phx.atomistic.read_atomistic_restart_model(path).model
    _, restored_dynamics, _, restored_thermodynamic = prepare(restored_model, capacity)
    template = restored_dynamics.initialize_state(
        positions,
        restored_thermodynamic,
        velocity=jnp.zeros_like(velocity),
        key=jax.random.key(2),
    )
    resumed = phx.atomistic.read_atomistic_restart(
        path,
        phx.atomistic.AtomisticCheckpointPlan(restored_dynamics, restored_thermodynamic),
        template,
    ).state
    for _ in range(10):
        resumed = restored_dynamics.step(resumed, restored_thermodynamic)
```

The example compares this continuation with the uninterrupted trajectory.

When a step can outgrow the declared image capacity, declare a finite capacity ladder.
`retry_atomistic_step_with_capacity` grows capacity on the host only after a
capacity-only rejection. It retries the same physical attempt from the retained
accepted coordinates, RNG address, thermostat state, and force cache. Scientific or
geometric failures are returned unchanged:

```python
ladder = phx.discretization.ParticleImageCapacityLadder(
    (
        capacity,
        phx.discretization.ParticleImageCapacity(
            maximum_particles_per_cell=32,
            maximum_edges=4096,
            maximum_degree=192,
            maximum_images=729,
        ),
    )
)
dynamics, thermodynamic, step = phx.atomistic.retry_atomistic_step_with_capacity(
    dynamics, state, thermodynamic, (state_plan,), ladder
)
if not bool(step.successful):
    raise RuntimeError(f"step rejected: {int(step.rejection_reasons)}")
state = step.accepted_state
```

A restart written after a capacity replacement binds the rebound preparation, and
`read_atomistic_restart` refuses a rebuilt preparation that does not match it. Native
MACE support envelopes are unreleased candidates. A random-initialized model
demonstrates the runtime contract, not material accuracy or MD stability.
