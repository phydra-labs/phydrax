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
base_prescribed = structure.prescribed_values(reference)
base_loads = jnp.zeros((3, 2)).at[1, 1].set(-1.0)
sample = fd.ForceDensityInputs(jnp.ones((2,)), base_prescribed, base_loads)
equilibrium = fd.ForceDensityProblem(structure, sign_mode="tension")
plan = fd.plan_force_density(equilibrium, sample)


def decode(design, _):
    force_density, support_height, center_load = design
    prescribed = base_prescribed.at[1].set(support_height)
    loads = base_loads.at[1, 1].set(center_load)
    return fd.ForceDensityInputs(jnp.repeat(force_density, 2), prescribed, loads)


def objective(state, design, _):
    target = jnp.asarray((0.0, -0.3))
    point = fd.scaled_target_residual(state.positions[1], target, 1.0)
    reaction_balance = state.support_reactions[0, 1] - state.support_reactions[2, 1]
    return jnp.sum(point**2) + 0.05 * reaction_balance**2


design = fd.ForceDensityDesignProblem(
    plan,
    decode,
    objective,
    design_bounds=phx.optim.Bounds(
        jnp.asarray((0.1, -0.5, -2.0)),
        jnp.asarray((5.0, 0.5, -0.1)),
    ),
)
result = fd.solve_force_density_design(
    design,
    jnp.asarray((1.0, 0.0, -1.0)),
    termination=phx.optim.OptimizationTermination(
        absolute_optimality=1.0e-6,
        relative_optimality=0.0,
        maximum_steps=400,
    ),
)
print("status", int(result.state_design.status))
print("design [q, support-y, load-y]", result.state_design.design)
print("center", result.equilibrium.state.positions[1])
print("reactions", result.equilibrium.state.support_reactions)
