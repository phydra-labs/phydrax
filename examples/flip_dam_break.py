#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


grid = phx.discretization.TensorGridPlan(
    (
        phx.discretization.UniformCellAxisSpec(20),
        phx.discretization.UniformCellAxisSpec(16),
    ),
    axis_names=("x", "y"),
).prepare(jnp.asarray([[0.0, 0.0], [1.0, 0.8]]))
finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
mac = phx.discretization.MACOperatorPlan(finite_volume).prepare()
boundaries = phx.discretization.MACBoundaryPlan(mac).prepare()
projection = phx.solver.MACFreeSurfaceProjectionPlan(
    mac, boundaries=boundaries, tolerance=1.0e-7
)

x = jnp.linspace(0.15, 0.42, 7)
y = jnp.linspace(0.15, 0.62, 9)
xx, yy = jnp.meshgrid(x, y, indexing="ij")
position = jnp.stack((xx.reshape((-1,)), yy.reshape((-1,))), axis=-1)
particles = phx.discretization.ParticleSetPlan(
    jnp.arange(position.shape[0]),
    jnp.full((position.shape[0],), 1000.0 * 0.004),
    ambient_dimension=2,
).prepare()
transfer = phx.discretization.flip.FLIPParticleTransferPlan(mac).prepare(particles)
problem = phx.equations.FLIPProblemIR("dam-break", 1000.0, jnp.asarray([0.0, -9.81]))
method = phx.discretization.flip.FLIPMethodPlan(
    0.05, liquid_fraction_threshold=0.02, extrapolation_layers=3
)
compiled = phx.equations.compile_flip_problem(problem, transfer, projection, method)
state = compiled.initialize_state(position, jnp.zeros_like(position))
result = compiled.step_detailed(state, 5.0e-4)
_, accepted_inspection = phx.equations.flip_inspection_frames(
    compiled,
    result,
    result_id="flip-dam-break-step",
)

print(
    {
        "successful": bool(result.successful),
        "liquid_cells": int(result.diagnostics.liquid_count),
        "divergence": float(result.diagnostics.divergence_norm),
        "volume_defect": float(result.diagnostics.mass_balance_defect),
        "inspection_fields": tuple(
            field.name for field in accepted_inspection.frame.fields
        ),
    }
)
