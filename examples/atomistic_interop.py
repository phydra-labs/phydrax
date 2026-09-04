import tempfile
from pathlib import Path

import jax.numpy as jnp

import phydrax as phx


system = phx.atomistic.AtomisticSystemPlan(
    [10, 20],
    [1, 1],
    [1.0, 1.0],
    phx.atomistic.AtomisticUnitSystem.reduced(),
    atom_type_ids=[0, 0],
).prepare()
potential = phx.atomistic.AtomisticPotentialProgram(
    [phx.atomistic.LennardJonesPotential([0.2], [1.0], 2.5)]
).prepare(system)
neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(2).prepare(
    system.particles
)
frame = phx.atomistic.AtomisticFrame(
    0.0,
    0,
    jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
    system.plan.particle_ids,
    system_id=system.plan.system_id,
    topology_id=system.topology.topology_id,
    units=system.plan.units,
    source_id="example-frame",
)
with tempfile.TemporaryDirectory(prefix="phydrax-example-") as directory:
    trajectory = phx.atomistic.interchange.H5MDTrajectoryPlan(
        Path(directory) / "trajectory.h5"
    )
    with trajectory.open(append=False) as writer:
        writer.write(frame)
    with trajectory.open() as reader:
        recovered = tuple(reader)
    if len(recovered) != 1 or not bool(
        jnp.array_equal(recovered[0].positions, frame.positions)
    ):
        raise RuntimeError("H5MD roundtrip failed")
    rerun = phx.atomistic.AtomisticRerunPlan(
        trajectory, potential, neighborhood, lambda_values=(0.0, 1.0)
    ).run()
    if not bool(rerun.successful) or rerun.reduction.frame_count != 1:
        raise RuntimeError("H5MD rerun failed")
    print(rerun.reduction.mean_energies)
