#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


fd = phx.applications.solid_mechanics
structure = fd.ForceDensityStructure.from_edges(
    jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
    3,
    2,
    fixed_nodes=(0, 2),
)
reference = jnp.asarray(((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)))
loads = jnp.asarray(((0.0, 0.0), (0.0, -1.0), (0.0, 0.0)))
prescribed = structure.prescribed_values(reference)
sample = fd.ForceDensityInputs(jnp.full((2,), 2.0), prescribed, loads)
equilibrium = fd.ForceDensityProblem(structure, sign_mode="tension")
plan = fd.plan_force_density(equilibrium, sample)


def decode(magnitude, _):
    return fd.ForceDensityInputs(jnp.repeat(magnitude.reshape(()), 2), prescribed, loads)


design = fd.ForceDensityDesignProblem(
    plan,
    decode,
    lambda state, magnitude, _: (state.positions[1, 1] + 0.3) ** 2,
    design_bounds=phx.optim.Bounds(0.2, 8.0),
)
compiled = fd.compile_structured_force_density_design(
    design,
    jnp.asarray(2.0),
)
result = fd.solve_structured_force_density_design(
    compiled,
    method=phx.optim.PrimalDualInteriorPoint(mode="dense-filter"),
    termination=phx.optim.OptimizationTermination(maximum_steps=100),
)
print("status", int(result.state_design.optimization.optimization.status))
print("design", result.state_design.design)
print("center", result.equilibrium.state.positions[1])
print("equilibrium", result.equilibrium.diagnostics.free_residual_norm)
