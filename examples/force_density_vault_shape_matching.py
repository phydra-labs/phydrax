#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


fd = phx.applications.solid_mechanics
node_count = 24
span = 10.0
x = jnp.linspace(-span / 2.0, span / 2.0, node_count)
reference = jnp.stack((x, jnp.zeros_like(x)), axis=-1)
edges = jnp.stack(
    (jnp.arange(node_count - 1), jnp.arange(1, node_count)), axis=-1
).astype(jnp.int32)
structure = fd.ForceDensityStructure.from_edges(
    edges, node_count, 2, fixed_nodes=(0, node_count - 1)
)
loads = jnp.zeros((node_count, 2)).at[:, 1].set(-span / node_count)
sample = fd.ForceDensityInputs(
    jnp.full((node_count - 1,), -10.0),
    structure.prescribed_values(reference),
    loads,
)
equilibrium = fd.ForceDensityProblem(structure, sign_mode="compression")
plan = fd.plan_force_density(equilibrium, sample)
target_inputs = fd.ForceDensityInputs(
    jnp.full((node_count - 1,), -8.0),
    sample.prescribed_values,
    loads,
)
target = fd.force_density_equilibrium(equilibrium, target_inputs).state.positions


def decode(magnitude, _):
    return fd.ForceDensityInputs(
        jnp.full((node_count - 1,), -magnitude.reshape(())),
        sample.prescribed_values,
        loads,
    )


def objective(state, magnitude, _):
    target_residual = fd.point_target_residual(state, target, span)
    return jnp.mean(target_residual**2) + 0.1 * (magnitude - 8.0) ** 2


design = fd.ForceDensityDesignProblem(
    plan,
    decode,
    objective,
    design_bounds=phx.optim.Bounds(1.0e-3, 1.0e3),
)
result = fd.solve_force_density_design(
    design,
    jnp.asarray(10.0),
    termination=phx.optim.OptimizationTermination(
        absolute_optimality=1.0e-5,
        relative_optimality=0.0,
        maximum_steps=400,
    ),
)
print("status", int(result.state_design.status))
print("force-density magnitude", result.state_design.design)
print(
    "maximum target error", jnp.max(jnp.abs(result.equilibrium.state.positions - target))
)
