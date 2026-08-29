#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from time import perf_counter

import equinox as eqx
import jax.numpy as jnp

import phydrax as phx


def run() -> dict[str, float]:
    vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = jnp.asarray([[0, 1, 3], [1, 2, 3]], dtype=jnp.int32)
    mesh = phx.discretization.CellMesh.from_triangles(vertices, cells)
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.discontinuous_element("triangle", 1)
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    form = phx.equations.fem.sipg_poisson_form(
        "u",
        1.0,
        phx.equations.fem.SIPGPenaltyPolicy(12.0),
        discretization.cell_domain,
        discretization.interior_facet_domain,
        (),
    )
    compiled = phx.equations.compile_finite_element_problem(form, discretization)
    state = jnp.arange(6.0)
    action = eqx.filter_jit(compiled.full_residual)
    action(state).block_until_ready()
    start = perf_counter()
    iterations = 25
    for _ in range(iterations):
        action(state).block_until_ready()
    elapsed = (perf_counter() - start) / iterations

    adapt_start = perf_counter()
    refined, adaptation, transfer = phx.discretization.refine_triangles_local(
        mesh, jnp.asarray([0])
    )
    adapt_elapsed = perf_counter() - adapt_start
    return {
        "sipg_apply_seconds": elapsed,
        "local_adaptation_seconds": adapt_elapsed,
        "refined_cells": float(refined.blocks[0].cell_count),
        "transfer_rows": float(transfer.primal.shape[0]),
        "adaptation_identity_length": float(len(adaptation.adaptation_id)),
    }


if __name__ == "__main__":
    print(run())
