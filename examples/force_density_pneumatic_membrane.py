#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


fd = phx.applications.solid_mechanics
positions = jnp.asarray(
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
)
connectivity = phx.discretization.polygonal_connectivity(
    jnp.asarray(((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)), dtype=jnp.int32),
    None,
    4,
)
structure = fd.ForceDensityStructure.from_edges(
    connectivity.edges,
    4,
    3,
    fixed_nodes=(0, 1, 2),
    surface_connectivity=connectivity,
)
volume = fd.enclosed_surface_volume(structure, positions)
load_model = fd.PneumaticPressureLoadModel(
    "ideal-gas", reference_volume=float(volume), exponent=1.4
)
problem = fd.ForceDensityProblem(
    structure,
    load_model=load_model,
    sign_mode="tension",
)
inputs = fd.ForceDensityInputs(
    jnp.full((structure.member_count,), 20.0),
    structure.prescribed_values(positions),
    jnp.asarray(0.02),
)
result = fd.force_density_equilibrium(
    problem,
    inputs,
    initial_positions=positions,
)
print("status", int(result.status), result.message)
print("volume", fd.enclosed_surface_volume(structure, result.state.positions))
print("top", result.state.positions[3])
print("load components", result.state.load_state.component_ids)
print("nonlinear iterations", result.nonlinear_result.diagnostics.iterations)
