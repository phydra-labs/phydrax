import tempfile
from pathlib import Path

import jax.numpy as jnp

import phydrax as phx


def finite_resources(
    program: phx.atomistic.AtomisticPotentialProgram,
) -> phx.atomistic.AtomisticNeighborhoodBundlePlan:
    """Explicit resources of one finite pair program in a nonperiodic frame."""
    requirements = program.requirements.neighborhoods
    return phx.atomistic.AtomisticNeighborhoodBundlePlan(
        requirements,
        (
            phx.atomistic.AtomisticNeighborhoodSource(
                "pairs",
                phx.atomistic.AtomisticSiteDomain.PHYSICAL_ATOMS,
                requirements,
                phx.atomistic.AtomisticCellListImageSearch(maximum_candidate_slots=4096),
                phx.discretization.ParticleImageCapacityLadder(
                    (
                        phx.discretization.ParticleImageCapacity(
                            maximum_particles_per_cell=2,
                            maximum_edges=8,
                            maximum_degree=2,
                            maximum_images=27,
                            maximum_case_edges=8,
                        ),
                    )
                ),
                skin=0.5,
            ),
        ),
        tuple(
            phx.atomistic.AtomisticRoleCapacityLadder(
                requirement,
                (
                    phx.atomistic.AtomisticRoleCapacity(
                        maximum_routes=4, maximum_degree=4
                    ),
                ),
            )
            for requirement in requirements
        ),
        frame=phx.discretization.PeriodicCell(
            20.0 * jnp.eye(3), origin=jnp.full((3,), -10.0), periodic_axes=(False,) * 3
        ),
    )


system = phx.atomistic.AtomisticSystemPlan(
    # ty: ignore[invalid-argument-type]
    [10, 20],
    # ty: ignore[invalid-argument-type]
    [1, 1],
    # ty: ignore[invalid-argument-type]
    [1.0, 1.0],
    phx.atomistic.AtomisticUnitSystem.reduced(),
    # ty: ignore[invalid-argument-type]
    atom_type_ids=[0, 0],
).prepare()
program = phx.atomistic.AtomisticPotentialProgram(
    # ty: ignore[invalid-argument-type]
    [phx.atomistic.LennardJonesPotential([0.2], [1.0], 2.5)]
)
potential = program.prepare(system, neighborhood_bundle=finite_resources(program))
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
    # Each frame binds its own certified resources from the potential's bundle.
    rerun = phx.atomistic.AtomisticRerunPlan(trajectory, potential).run()
    if not bool(rerun.successful) or rerun.reduction.frame_count != 1:
        raise RuntimeError("H5MD rerun failed")
    print(rerun.reduction.mean_energies)
