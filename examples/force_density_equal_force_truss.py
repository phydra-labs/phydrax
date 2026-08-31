#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


fd = phx.applications.solid_mechanics
structure = fd.ForceDensityStructure.from_edges(
    jnp.asarray(((0, 1), (1, 2), (2, 3)), dtype=jnp.int32),
    4,
    2,
    fixed_nodes=(0, 3),
)
reference = jnp.asarray(((-1.5, 0.0), (-0.5, 0.0), (0.5, 0.0), (1.5, 0.0)))
loads = jnp.asarray(((0.0, 0.0), (0.0, -0.8), (0.2, -1.2), (0.0, 0.0)))
sample = fd.ForceDensityInputs(
    jnp.asarray((1.0, 1.5, 0.8)),
    structure.prescribed_values(reference),
    loads,
)
equilibrium = fd.ForceDensityProblem(structure, sign_mode="tension")
plan = fd.plan_force_density(equilibrium, sample)


def decode(force_densities, _):
    return fd.ForceDensityInputs(force_densities, sample.prescribed_values, loads)


def objective(state, force_densities, _):
    uniformity = fd.scaled_uniformity_residual(
        state.axial_forces,
        1.0,
        mask=state.member_valid,
    )
    target_height = fd.scaled_target_residual(
        state.positions[1:3, 1], jnp.asarray((-0.6, -0.6)), 1.0
    )
    return jnp.mean(uniformity**2) + 0.1 * jnp.mean(target_height**2)


design = fd.ForceDensityDesignProblem(
    plan,
    decode,
    objective,
    design_bounds=phx.optim.Bounds(0.1, 10.0),
)
result = fd.solve_force_density_design(
    design,
    sample.force_densities,
    termination=phx.optim.OptimizationTermination(
        absolute_optimality=1.0e-2,
        relative_optimality=0.0,
        maximum_steps=500,
    ),
)
print("status", int(result.state_design.status))
print("force densities", result.state_design.design)
print("axial forces", result.equilibrium.state.axial_forces)
