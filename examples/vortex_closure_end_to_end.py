#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


position = jnp.asarray(((-0.6, -0.2), (-0.2, 0.3), (0.3, -0.4), (0.7, 0.2)))
source = phx.discretization.VortexSourceState(
    position,
    jnp.asarray((0.5, -0.3, 0.8, -0.4)),
    core_radius=jnp.full((4,), 0.08),
    volume=jnp.full((4,), 0.25),
)
target = phx.discretization.VortexTargetState(jnp.asarray(((0.9, 0.9), (-0.9, -0.9))))
fmm = phx.operators.VortexFMMPlan(
    position,
    (-1.0, -1.0),
    (1.0, 1.0),
    depth=2,
    expansion_order=1,
    leaf_capacity=4,
).prepare(
    source_capacity=4,
    target_capacity=2,
    target_topology="arbitrary-targets",
)
field = fmm.evaluate(source, target)

population_plan = phx.discretization.VortexPopulationPlan(6, 2)
population, journal = population_plan.initialize(
    jnp.pad(position, ((0, 2), (0, 0))),
    jnp.pad(source.strength, (0, 2)),
    jnp.pad(source.core_radius, (0, 2), constant_values=1.0),
    jnp.pad(source.volume, (0, 2), constant_values=1.0),
    active_mask=jnp.asarray((True, True, True, True, False, False)),
)
split = population_plan.split(population, journal, 0, (0.02, 0.0))

print("field", field.velocity)
print("FMM tail bound", field.diagnostics.backend_diagnostics.geometric_tail_bound)
print("population count", split.evidence.active_count_after)
print("strength defect", split.evidence.strength_residual)
print("successful", bool(field.successful & split.successful))
