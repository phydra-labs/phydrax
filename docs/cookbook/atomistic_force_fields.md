# Native classical force fields

This minimal reduced-unit system combines two pair terms and records its origin.

```python
import jax.numpy as jnp
import phydrax as phx

system_plan = phx.atomistic.AtomisticSystemPlan(
    [10, 20, 30], [1, 1, 1], [1.0, 1.0, 1.0],
    phx.atomistic.AtomisticUnitSystem.reduced(), atom_type_ids=[0, 0, 0],
)
system = system_plan.prepare()
program_plan = phx.atomistic.AtomisticPotentialProgram([
    phx.atomistic.MorsePotential([[0.2]], [[2.0]], [[1.0]], 2.5),
    phx.atomistic.BuckinghamPotential([[1.0]], [[2.0]], [[0.1]], 2.5),
])
force_field = phx.atomistic.AtomisticForceFieldPlan(
    system_plan,
    program_plan,
    phx.atomistic.AtomisticNonbondedPolicy(2.5),
    phx.atomistic.AtomisticForceFieldProvenance(
        "native", ("project-parameters",), "custom", "explicit"
    ),
).prepare()
positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]])
neighbors = phx.discretization.DenseParticleNeighborhoodPlan(3).prepare(system.particles)
result = force_field.potential.evaluate(positions, neighbors.build(positions))
assert bool(result.successful)
```

For physical calculations, replace reduced units and demonstration parameters with one
qualified parameter source. Persist `force_field.preparation` and
`force_field.plan.provenance` with outputs.
