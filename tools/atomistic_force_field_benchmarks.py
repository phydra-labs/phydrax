import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


units = phx.atomistic.AtomisticUnitSystem.reduced()
system = phx.atomistic.AtomisticSystemPlan(
    np.asarray([0, 1, 2]),
    np.asarray([1, 1, 1]),
    np.asarray([1.0, 1.0, 1.0]),
    units,
    atom_type_ids=np.asarray([0, 0, 0]),
).prepare()
positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]])
program = phx.atomistic.AtomisticPotentialProgram(
    (
        phx.atomistic.MorsePotential(
            np.asarray([[0.2]]), np.asarray([[2.0]]), np.asarray([[1.0]]), 2.5
        ),
        phx.atomistic.BuckinghamPotential(
            np.asarray([[1.0]]), np.asarray([[2.0]]), np.asarray([[0.1]]), 2.5
        ),
    )
)
# Both interaction-site pair terms are served by one declared cell-list image
# source of their entity inside an explicit nonperiodic search frame; each
# term's view has its own role ladder.
requirements = program.requirements.neighborhoods
(entity,) = {requirement.entity for requirement in requirements}
resources = phx.atomistic.AtomisticNeighborhoodBundlePlan(
    requirements,
    (
        phx.atomistic.AtomisticNeighborhoodSource(
            "pair-terms",
            entity,
            requirements,
            phx.atomistic.AtomisticCellListImageSearch(maximum_candidate_slots=4096),
            phx.discretization.ParticleImageCapacityLadder(
                (
                    phx.discretization.ParticleImageCapacity(
                        maximum_particles_per_cell=3,
                        maximum_edges=12,
                        maximum_degree=4,
                        maximum_images=27,
                        maximum_case_edges=12,
                    ),
                )
            ),
            skin=0.5,
        ),
    ),
    tuple(
        phx.atomistic.AtomisticRoleCapacityLadder(
            requirement,
            (phx.atomistic.AtomisticRoleCapacity(maximum_routes=6, maximum_degree=4),),
        )
        for requirement in requirements
    ),
    frame=phx.discretization.PeriodicCell(
        20.0 * jnp.eye(3), origin=jnp.full((3,), -10.0), periodic_axes=(False,) * 3
    ),
)
potential = program.prepare(system, neighborhood_bundle=resources)
# Resource binding stays outside the timed numerical evaluation.
bundle = potential.neighborhood_bundle
neighborhoods = bundle.initialize(
    phx.atomistic.bind_atomistic_bundle_geometry(bundle, system, positions)
)
if not bool(neighborhoods.successful[0]):
    raise RuntimeError("neighborhood bundle binding failed")
compiled = eqx.filter_jit(potential.evaluate)
started = time.perf_counter()
first = compiled(positions, neighborhoods)
jax.block_until_ready(first.energy)
compile_seconds = time.perf_counter() - started
started = time.perf_counter()
for _ in range(20):
    result = compiled(positions, neighborhoods)
jax.block_until_ready(result.energy)
steady_seconds = (time.perf_counter() - started) / 20
print(
    json.dumps(
        {
            "compile_seconds": compile_seconds,
            "steady_seconds": steady_seconds,
            "energy": float(result.energy),
            "force_norm": float(jnp.sqrt(jnp.sum(result.forces**2))),
            "successful": bool(result.successful),
            "program_id": potential.prepared_id,
            "neighborhood_bundle_id": bundle.prepared_id,
        },
        indent=2,
        sort_keys=True,
    )
)
