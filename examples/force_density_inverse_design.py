#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from math import sqrt

import jax.numpy as jnp

import phydrax as phx


fd = phx.applications.solid_mechanics

span = 10.0
load_density = 1.0
num_nodes = 40
coordinates = jnp.stack(
    (
        jnp.linspace(-span / 2.0, span / 2.0, num_nodes),
        jnp.zeros((num_nodes,)),
    ),
    axis=-1,
)
edges = jnp.stack((jnp.arange(num_nodes - 1), jnp.arange(1, num_nodes)), axis=-1).astype(
    jnp.int32
)
structure = fd.ForceDensityStructure.from_edges(
    edges,
    num_nodes,
    2,
    fixed_nodes=(0, num_nodes - 1),
)
loads = jnp.zeros((num_nodes, 2)).at[:, 1].set(-(load_density * span) / num_nodes)
prescribed = structure.prescribed_values(coordinates)
sample = fd.ForceDensityInputs(
    jnp.full((num_nodes - 1,), -10.0),
    prescribed,
    loads,
)
equilibrium = fd.ForceDensityProblem(
    structure,
    sign_mode="compression",
    problem_id="minimum-load-path-arch",
)
plan = fd.plan_force_density(equilibrium, sample)


def decode(magnitude, _):
    return fd.ForceDensityInputs(
        jnp.full((num_nodes - 1,), -magnitude.reshape(())),
        prescribed,
        loads,
    )


design_problem = fd.ForceDensityDesignProblem(
    plan,
    decode,
    lambda state, design, _: fd.force_density_load_path(state),
    design_bounds=phx.optim.Bounds(1.0e-3, 1.0e3),
    problem_id="minimum-load-path-arch-design",
)
result = fd.solve_force_density_design(
    design_problem,
    jnp.asarray(10.0),
    termination=phx.optim.OptimizationTermination(
        absolute_optimality=1.0e-6,
        relative_optimality=0.0,
        maximum_steps=500,
    ),
)

rise = jnp.max(jnp.abs(result.equilibrium.state.positions[:, 1]))
load_path = fd.force_density_load_path(result.equilibrium.state)
print("status", int(result.state_design.status))
print("optimized force-density magnitude", result.state_design.design)
print("rise", rise, "analytical", sqrt(3.0) * span / 4.0)
print(
    "load path",
    load_path,
    "analytical",
    load_density * span**2 / sqrt(3.0),
)
print("equilibrium residual", result.equilibrium.diagnostics.free_residual_norm)
