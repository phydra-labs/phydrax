import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


units = phx.atomistic.AtomisticUnitSystem.reduced()
system = phx.atomistic.AtomisticSystemPlan(
    [0, 1, 2], [1, 1, 1], [1.0, 1.0, 1.0], units, atom_type_ids=[0, 0, 0]
).prepare()
positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]])
neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(3).prepare(
    system.particles
)
relation = neighborhood.build(positions)
program = phx.atomistic.AtomisticPotentialProgram(
    [
        phx.atomistic.MorsePotential([[0.2]], [[2.0]], [[1.0]], 2.5),
        phx.atomistic.BuckinghamPotential([[1.0]], [[2.0]], [[0.1]], 2.5),
    ]
).prepare(system)
compiled = eqx.filter_jit(program.evaluate)
started = time.perf_counter()
first = compiled(positions, relation)
jax.block_until_ready(first.energy)
compile_seconds = time.perf_counter() - started
started = time.perf_counter()
for _ in range(20):
    result = compiled(positions, relation)
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
            "program_id": program.prepared_id,
        },
        indent=2,
        sort_keys=True,
    )
)
