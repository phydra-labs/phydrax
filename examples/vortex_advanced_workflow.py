#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


position = jnp.asarray(((-0.5, -0.2), (-0.1, 0.3), (0.4, -0.25), (0.7, 0.2)))
circulation = jnp.asarray((0.6, -0.2, 0.4, -0.8))
core = jnp.full((4,), 0.1)
probe_position = jnp.asarray(((2.0, 1.5), (-2.0, -1.5)))
accelerated = phx.operators.BarnesHutVortexPlan2D(
    position,
    leaf_size=2,
    opening_angle=0.6,
).evaluate(position, circulation, core, probe_position)

trainer = lambda samples, weights, args: jnp.mean(weights)
evaluator = lambda model, targets: jnp.full((targets.shape[0],), model)
reconstruction = lambda vorticity, targets, args: (
    vorticity[:, None] * jnp.stack((-targets[:, 1], targets[:, 0]), axis=-1)
)
learned = phx.applications.vortex_flow.LearnedVorticityWorkflow(
    trainer,
    evaluator,
    reconstruction,
    workflow_id="example-learned-vorticity",
).fit_and_reconstruct(position, circulation, probe_position)

print("accelerated velocity", accelerated.velocity)
print("tree error bound", accelerated.diagnostics.backend_diagnostics.truncation_bound)
print("learned velocity", learned.velocity)
print("successful", bool(accelerated.successful & learned.finite))
