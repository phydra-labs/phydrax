#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
cells = jnp.asarray([[0, 1, 3], [1, 2, 3]], dtype=jnp.int32)
mesh = phx.discretization.CellMesh.from_triangles(
    vertices,
    cells,
    cell_global_ids=jnp.asarray([10, 20]),
)
dg_field = phx.discretization.FiniteElementFieldSpec(
    "u", phx.discretization.discontinuous_element("triangle", 1)
)
dg = phx.discretization.FiniteElementPlan(mesh, dg_field).prepare()
boundary_data = phx.equations.coefficient(
    lambda points, context: points[..., 0] + points[..., 1],
    coefficient_id="full-stack-affine-boundary",
)
sipg = phx.equations.fem.sipg_poisson_form(
    "u",
    1.0,
    phx.equations.fem.SIPGPenaltyPolicy(12.0),
    dg.cell_domain,
    dg.interior_facet_domain,
    (phx.equations.fem.sipg_dirichlet(dg.exterior_facet_domain, boundary_data),),
)
compiled_sipg = phx.equations.compile_finite_element_problem(sipg, dg)
affine_state = dg.project("u", lambda points, args: points[..., 0] + points[..., 1])
sipg_defect = jnp.linalg.norm(compiled_sipg.full_residual(affine_state))

refined, adaptation, transfer = phx.discretization.refine_triangles_local(
    mesh, jnp.asarray([10])
)
transfer_defect = jnp.max(jnp.abs(transfer.primal @ jnp.ones((4,)) - 1.0))

cg_field = phx.discretization.FiniteElementFieldSpec(
    "eta", phx.discretization.lagrange_element("triangle", 1)
)
cg = phx.discretization.FiniteElementPlan(mesh, cg_field).prepare()
phase_result = phx.applications.phase_field.solve_allen_cahn_step(
    cg,
    "eta",
    jnp.full((4,), 0.2),
    0.01,
    phx.applications.phase_field.AllenCahnParameters(
        1.0,
        phx.equations.BinaryThermodynamicParameters(1.0, 0.02),
    ),
)

if (
    float(sipg_defect) > 1.0e-11
    or float(transfer_defect) > 1.0e-12
    or not bool(phase_result.successful)
    or not bool(phase_result.energy_after < phase_result.energy_before)
):
    raise RuntimeError("Finite-element full-stack smoke failed.")

print(
    {
        "sipg_affine_defect": float(sipg_defect),
        "refined_cells": refined.blocks[0].cell_count,
        "adaptation_id": adaptation.adaptation_id,
        "transfer_constant_defect": float(transfer_defect),
        "allen_cahn_energy_before": float(phase_result.energy_before),
        "allen_cahn_energy_after": float(phase_result.energy_after),
    }
)
