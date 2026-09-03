import jax
import jax.numpy as jnp

import phydrax as phx


units = phx.atomistic.AtomisticUnitSystem.reduced()
fine_system = phx.atomistic.AtomisticSystemPlan(
    [10, 20, 30, 40],
    [1, 1, 1, 1],
    [1.0, 3.0, 2.0, 2.0],
    units,
    atom_type_ids=[1, 1, 1, 1],
    molecule_ids=[0, 0, 1, 1],
).prepare()
mapping = phx.atomistic.MolecularCoarseMapPlan([100, 200], [0, 1], [0, 0, 1, 1]).prepare(
    fine_system
)
positions = jnp.asarray(
    [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 5.0, 0.0]]]
)
forces = jnp.asarray(
    [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]]
)
fine_batch = phx.atomistic.AtomisticBatch(
    jnp.ones((1, 4), dtype=jnp.int32),
    positions,
    jnp.asarray([[1.0, 3.0, 2.0, 2.0]]),
    units.scale,
    particle_ids=jnp.asarray([[10, 20, 30, 40]]),
    atom_type_ids=jnp.ones((1, 4), dtype=jnp.int32),
    structure_ids=("fine-example",),
)
execution = phx.atomistic.AtomisticGraphExecutionPlan(
    1, backend="dense", maximum_dense_atoms=2
)
problem = phx.atomistic.CoarseForceMatchingProblem(mapping, fine_batch, forces, execution)
potential = phx.nn.atomistic.PaiNNPotential(
    units.scale,
    cutoff=6.0,
    feature_count=4,
    interaction_count=1,
    radial_basis_count=3,
    maximum_species_id=1,
    species_kind=phx.atomistic.AtomisticSpeciesKind.ATOM_TYPE_ID,
    key=jax.random.key(0),
)
result = phx.atomistic.fit_coarse_potential(
    potential,
    problem,
    phx.atomistic.AtomisticTrainingPolicy(maximum_steps=0, energy_weight=0.0),
    jax.random.key(1),
)
if not bool(result.valid):
    raise RuntimeError("Coarse force-matching example did not qualify.")
print(mapping.coarse_system.plan.masses, float(result.projected_force_rms))
