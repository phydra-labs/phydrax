import jax.numpy as jnp

import phydrax as phx


system_plan = phx.atomistic.AtomisticSystemPlan(
    [0, 1, 2],
    [1, 1, 1],
    [1.0, 1.0, 1.0],
    phx.atomistic.AtomisticUnitSystem.reduced(),
    atom_type_ids=[0, 0, 0],
)
system = system_plan.prepare()
force_field = phx.atomistic.AtomisticForceFieldPlan(
    system_plan,
    phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.MorsePotential([[0.2]], [[2.0]], [[1.0]], 2.5)]
    ),
    phx.atomistic.AtomisticNonbondedPolicy(2.5),
    phx.atomistic.AtomisticForceFieldProvenance(
        "native", ("example",), "custom", "explicit"
    ),
).prepare()
positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]])
neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(3).prepare(
    system.particles
)
result = force_field.potential.evaluate(positions, neighborhood.build(positions))
if not bool(result.successful):
    raise RuntimeError("force-field evaluation failed")
print(float(result.energy), result.forces)
