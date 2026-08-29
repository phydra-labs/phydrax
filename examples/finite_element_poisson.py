#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
cells = jnp.asarray(
    [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
    dtype=jnp.int32,
)
mesh = phx.discretization.CellMesh.from_triangles(vertices, cells)
element = phx.discretization.lagrange_element("triangle", 1)
field = phx.discretization.FiniteElementFieldSpec("u", element)
discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
constraint = phx.discretization.dirichlet_constraint(discretization, "u")
form = phx.equations.FiniteElementForm(
    "affine-poisson",
    "u",
    (
        phx.equations.DiffusionAction("u", 1.0),
        phx.equations.SourceAction("u", 0.0),
    ),
)
compiled = phx.equations.compile_finite_element_problem(
    form,
    discretization,
    constraint=constraint,
    dirichlet_values=lambda points: points[..., 0] + points[..., 1],
)
system, right_hand_side = compiled.linear_system()
result = phx.linalg.solve(system, right_hand_side)
solution = compiled.expand(result.value)
expected = vertices[:, 0] + vertices[:, 1]
error = jnp.max(jnp.abs(solution - expected))

if not bool(jnp.all(result.successful)) or float(error) > 1.0e-10:
    raise RuntimeError(
        "Finite-element Poisson example failed its affine exactness check."
    )

print(
    {
        "successful": bool(jnp.all(result.successful)),
        "maximum_error": float(error),
        "prepared_id": discretization.prepared_id,
        "compilation_id": compiled.compilation_id,
    }
)
