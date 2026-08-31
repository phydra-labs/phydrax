"""Solve a polynomial Poisson problem with conforming virtual elements."""

import jax.numpy as jnp

import phydrax as phx


coordinates = jnp.asarray(
    (
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
        (1.0, 0.5),
        (0.0, 1.0),
        (0.5, 1.0),
        (1.0, 1.0),
    )
)
polygons = (
    (0, 1, 4, 3),
    (1, 2, 5, 4),
    (3, 4, 7, 6),
    (4, 5, 8, 7),
)
mesh = phx.discretization.CellMesh.from_polygons(coordinates, polygons)
element = phx.discretization.conforming_h1_virtual_element(2)
field = phx.discretization.VirtualElementFieldSpec("u", element)
space = phx.discretization.VirtualElementPlan(mesh, field).prepare()
constraint = phx.discretization.virtual_element_dirichlet_constraint(space, "u")
form = phx.equations.VirtualElementForm(
    "linear-poisson",
    "u",
    (phx.equations.DiffusionAction("u", 1.0),),
)
compiled = phx.equations.compile_virtual_element_problem(
    form,
    space,
    constraint=constraint,
    dirichlet_values=lambda points: points[:, 0] + points[:, 1],
    execution_policy=phx.equations.VirtualElementExecutionPolicy(realization="sparse"),
)
problem, right_hand_side = compiled.linear_system()
solution = phx.linalg.solve(problem, right_hand_side)
full_solution = compiled.expand(solution.value)
residual = compiled.residual(solution.value)
reconstruction = phx.equations.project_virtual_element_field(space, full_solution)
centroid_points = tuple(
    geometry.centroids[:, None, :] for geometry in space.default_runtime.geometries
)
centroid_values = tuple(
    phx.equations.evaluate_virtual_element_reconstruction(
        reconstruction,
        space,
        block,
        points,
    )[0][:, 0]
    for block, points in enumerate(centroid_points)
)
print(
    {
        "residual_norm": float(jnp.sqrt(jnp.sum(residual * residual))),
        "center_value": float(full_solution[4]),
        "projected_centroids": [value.tolist() for value in centroid_values],
        "status": str(solution.status),
    }
)
