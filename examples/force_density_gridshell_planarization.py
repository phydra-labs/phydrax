#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


fd = phx.applications.solid_mechanics
connectivity = phx.discretization.polygonal_connectivity(
    None,
    jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
    4,
)
structure = fd.ForceDensityStructure.from_edges(
    connectivity.edges,
    4,
    3,
    fixed_nodes=(0, 1, 2, 3),
    surface_connectivity=connectivity,
)
base = jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)))
loads = jnp.zeros((4, 3))
sample_positions = base.at[2, 2].set(0.2)
sample = fd.ForceDensityInputs(
    jnp.ones((structure.member_count,)),
    structure.prescribed_values(sample_positions),
    loads,
)
equilibrium = fd.ForceDensityProblem(structure, sign_mode="tension")
plan = fd.plan_force_density(equilibrium, sample)


def decode(support_height, _):
    positions = base.at[2, 2].set(support_height.reshape(()))
    return fd.ForceDensityInputs(
        sample.force_densities,
        structure.prescribed_values(positions),
        loads,
    )


def objective(state, support_height, _):
    planarity = fd.surface_planarity_residual(structure, state.positions, 1.0)
    rectangularity = fd.surface_rectangularity_residual(structure, state.positions, 1.0)
    return jnp.sum(planarity**2) + 0.1 * jnp.sum(rectangularity**2)


design = fd.ForceDensityDesignProblem(
    plan,
    decode,
    objective,
    design_bounds=phx.optim.Bounds(-0.5, 0.5),
)
result = fd.solve_force_density_design(
    design,
    jnp.asarray(0.2),
    termination=phx.optim.OptimizationTermination(
        absolute_optimality=1.0e-7,
        relative_optimality=0.0,
        maximum_steps=200,
    ),
)
print("status", int(result.state_design.status))
print("support height", result.state_design.design)
print(
    "planarity",
    fd.surface_planarity_residual(structure, result.equilibrium.state.positions, 1.0),
)
