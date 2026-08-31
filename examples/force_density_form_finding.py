#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


fd = phx.applications.solid_mechanics

edges = jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32)
structure = fd.ForceDensityStructure.from_edges(
    edges,
    3,
    2,
    fixed_nodes=(0, 2),
)
reference = jnp.asarray(((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)))
loads = jnp.asarray(((0.0, 0.0), (0.0, -1.0), (0.0, 0.0)))
prescribed = structure.prescribed_values(reference)


def solve(sign_mode, force_density):
    problem = fd.ForceDensityProblem(
        structure,
        sign_mode=sign_mode,
        problem_id=f"three-node-{sign_mode}",
    )
    inputs = fd.ForceDensityInputs(
        jnp.full((2,), force_density),
        prescribed,
        loads,
    )
    return fd.force_density_equilibrium(problem, inputs)


tension = solve("tension", 1.0)
compression = solve("compression", -1.0)

print("tension positions", tension.state.positions)
print("tension forces", tension.state.axial_forces)
print("tension reactions", tension.state.support_reactions)
print("tension residual", tension.diagnostics.free_residual_norm)
print("compression positions", compression.state.positions)
print("compression forces", compression.state.axial_forces)
print("compression reactions", compression.state.support_reactions)
print("compression residual", compression.diagnostics.free_residual_norm)
