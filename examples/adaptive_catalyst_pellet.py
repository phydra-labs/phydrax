#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Adapt one two-dimensional catalyst-pellet mesh conservatively."""

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def grid(nx, ny):
    vertices = np.asarray(
        [(2.0 * i / nx, j / ny) for j in range(ny + 1) for i in range(nx + 1)]
    )
    cells = []
    for j in range(ny):
        for i in range(nx):
            lower = j * (nx + 1) + i
            cells.append((lower, lower + 1, lower + nx + 2, lower + nx + 1))
    return phx.discretization.UnstructuredFiniteVolumePlan(
        vertices, quadrilaterals=np.asarray(cells)
    ).prepare()


coarse = grid(2, 1)
fine = grid(4, 2)
parent = np.asarray((0, 0, 1, 1, 0, 0, 1, 1), dtype=np.int32)
prolongation = phx.discretization.UnstructuredConservativeRemapPlan(
    coarse,
    fine,
    np.arange(fine.cell_count + 1, dtype=np.int32),
    parent,
    fine.cell_volumes,
    method="pellet-prolongation",
    provenance="analytic-refinement",
)
restriction = phx.discretization.UnstructuredConservativeRemapPlan(
    fine,
    coarse,
    np.asarray((0, 4, 8), dtype=np.int32),
    np.asarray((0, 1, 4, 5, 2, 3, 6, 7), dtype=np.int32),
    np.asarray((0.25,) * 8),
    method="pellet-restriction",
    provenance="analytic-refinement",
)
hierarchy = phx.discretization.UnstructuredAMRHierarchyPlan(
    coarse, fine, prolongation, restriction, maximum_refined_cells=2
)
state = phx.discretization.initialize_particle_internal_amr(
    hierarchy,
    jnp.asarray([[2.0, 4.0]]),
    jnp.asarray([[[1.0], [3.0]]]),
    jnp.asarray([[0.2, 0.4]]),
    jnp.asarray([[2.0, 3.0]]),
    jnp.asarray([[[0.2], [0.8]]]),
    jnp.asarray([1.0]),
    jnp.asarray([True]),
)
result = phx.discretization.adapt_particle_internal_mesh(
    hierarchy,
    phx.discretization.ParticleInternalAdaptationPolicy(1.0, 0.5),
    state,
    jnp.asarray([[2.0, 0.0]]),
)
print(f"successful={bool(result.successful)}")
print(f"refined_cells={int(jnp.sum(result.accepted_state.coarse_refined))}")
print(f"active_fine_cells={int(jnp.sum(result.accepted_state.fine_active))}")
print(f"energy_residual={float(jnp.max(jnp.abs(result.evidence.energy_residual))):.6e}")
print(f"species_residual={float(jnp.max(jnp.abs(result.evidence.species_residual))):.6e}")
