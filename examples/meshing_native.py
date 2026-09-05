#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Certify, refine, transfer, and solve on a native mesh without external providers."""

import json

import jax.numpy as jnp
import numpy as np

import phydrax as phx


mesh = phx.discretization.CellMesh.from_triangles(
    np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5))),
    np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int32),
)
certified = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
transition, _ = phx.meshing.refine_triangle_mesh(
    certified.mesh,
    np.asarray(certified.mesh.blocks[0].global_ids)[:1],
    certified.coordinate_contract,
)
stencil = transition.vertex_stencil
assert stencil is not None
transferred = stencil.apply(
    mesh.vertex_global_ids, mesh.coordinates[:, 0] + mesh.coordinates[:, 1]
)
target = transition.target.mesh
expected = target.coordinates[:, 0] + target.coordinates[:, 1]
transfer_error = float(jnp.max(jnp.abs(transferred - expected)))
field = phx.discretization.FiniteElementFieldSpec(
    "u", phx.discretization.lagrange_element("triangle", 1)
)
space = phx.discretization.FiniteElementPlan(target, field).prepare()
constraint = phx.discretization.dirichlet_constraint(space, "u")
form = phx.equations.FiniteElementForm(
    "meshing-affine-poisson", "u", (phx.equations.DiffusionAction("u", 1.0),)
)
problem = phx.equations.compile_finite_element_problem(
    form,
    space,
    constraint=constraint,
    dirichlet_values=lambda points: points[..., 0] + points[..., 1],
)
operator, rhs = problem.linear_system()
solved = phx.linalg.solve(operator, rhs)
error = float(jnp.max(jnp.abs(problem.expand(solved.value) - expected)))
if not bool(jnp.all(solved.successful)) or error > 1e-10 or transfer_error > 1e-12:
    raise RuntimeError("Native meshing workflow failed affine transfer or PDE exactness.")
print(
    json.dumps(
        {
            "source_cells": mesh.blocks[0].cell_count,
            "target_cells": target.blocks[0].cell_count,
            "transfer_error": transfer_error,
            "solution_error": error,
            "audit_passed": transition.target.audit.passed,
        },
        indent=2,
    )
)
