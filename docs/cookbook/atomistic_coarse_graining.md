# Fit a fixed-map coarse molecular potential

Use a strict center-of-mass partition and a type-ID potential. Beads are active
particles with no chemical-element identity.

```python
import jax
import phydrax as phx

mapping = phx.atomistic.MolecularCoarseMapPlan(
    bead_particle_ids,
    bead_type_ids,
    fine_particle_to_bead,
    topology=coarse_topology,
).prepare(fine_system)
problem = phx.atomistic.CoarseForceMatchingProblem(
    mapping,
    fine_training_batch,
    fine_training_forces,
    dense_coarse_graph,
    validation_batch=fine_validation_batch,
    validation_fine_forces=fine_validation_forces,
)
model = phx.nn.atomistic.PaiNNPotential(
    scale,
    cutoff=cutoff,
    maximum_species_id=maximum_bead_type,
    species_kind=phx.atomistic.AtomisticSpeciesKind.ATOM_TYPE_ID,
    key=jax.random.key(0),
)
result = phx.atomistic.fit_coarse_potential(
    model,
    problem,
    phx.atomistic.AtomisticTrainingPolicy(
        maximum_steps=1000,
        energy_weight=0.0,
        force_weight=1.0,
    ),
    jax.random.key(1),
)
if not bool(result.valid):
    raise RuntimeError("Coarse force match did not qualify")
```

When an analytic prior is used, pass its force contribution and stable ID to the
problem and include the same prior in the runtime potential program. Force-matching
validation supports an equilibrium PMF claim only; establish coarse kinetics
separately.
