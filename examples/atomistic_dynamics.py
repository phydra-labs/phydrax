#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservative periodic Lennard-Jones dynamics with a bounded trajectory."""

import jax
import jax.numpy as jnp

import phydrax as phx


units = phx.atomistic.AtomisticUnitSystem.reduced()
cell = phx.discretization.ParticleCell(4.0 * jnp.eye(3))
system = phx.atomistic.AtomisticSystemPlan(
    [100, 101, 102, 103],
    [1, 1, 1, 1],
    [1.0, 1.0, 1.0, 1.0],
    units,
    atom_type_ids=[0, 0, 0, 0],
    cell=cell,
).prepare()
base_neighborhood = phx.discretization.MetricCellListParticleNeighborhoodPlan(
    1.7, 4, 6, cell
)
neighborhood = phx.discretization.VerletParticleNeighborhoodPlan(
    base_neighborhood, 1.5, 0.2
).prepare(system.particles)
potential = phx.atomistic.AtomisticPotentialProgram(
    [phx.atomistic.LennardJonesPotential([0.2], [0.8], 1.5, switch_distance=1.3)]
).prepare(system)
dynamics = phx.atomistic.AtomisticDynamicsPlan(
    system,
    potential,
    neighborhood,
    phx.atomistic.BAOABLangevinPlan(2.0e-4, 1.0, 0.2),
).prepare()
positions = cell.cartesian(
    jnp.asarray(
        [
            [0.10, 0.10, 0.10],
            [0.35, 0.10, 0.10],
            [0.10, 0.35, 0.10],
            [0.35, 0.35, 0.10],
        ]
    )
)
initial = dynamics.initialize_state(
    positions,
    velocity=jnp.zeros_like(positions),
    key=jax.random.key(0),
)
result = phx.atomistic.AtomisticRolloutPlan(
    dynamics,
    phx.atomistic.AtomisticTrajectoryPlan(20, sample_stride=5),
    replay=phx.atomistic.AtomisticReplayPolicy("step"),
).rollout(initial)
if not bool(result.successful):
    raise RuntimeError("Atomistic dynamics rollout failed")
diagnostics = dynamics.diagnostics(result.final_state)
print(
    {
        "samples": int(result.trajectory.count),
        "accepted_steps": int(result.replay.accepted_steps),
        "total_energy": float(diagnostics.total_energy),
        "temperature": float(diagnostics.temperature),
        "neighbor_rebuilds": int(diagnostics.neighborhood_rebuild_count),
    }
)
