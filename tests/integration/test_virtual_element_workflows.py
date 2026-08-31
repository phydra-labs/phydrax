#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math

import jax.numpy as jnp

import phydrax as phx


def _structured_space(count, degree=1):
    axis = jnp.linspace(0.0, 1.0, count + 1)
    x, y = jnp.meshgrid(axis, axis, indexing="xy")
    coordinates = jnp.stack((x.reshape((-1,)), y.reshape((-1,))), axis=-1)
    polygons = []
    for row in range(count):
        for column in range(count):
            lower = row * (count + 1) + column
            polygons.append(
                (
                    lower,
                    lower + 1,
                    lower + count + 2,
                    lower + count + 1,
                )
            )
    mesh = phx.discretization.CellMesh.from_polygons(coordinates, tuple(polygons))
    field = phx.discretization.VirtualElementFieldSpec(
        "u", phx.discretization.conforming_h1_virtual_element(degree)
    )
    return phx.discretization.VirtualElementPlan(mesh, field).prepare()


def _poisson_error(count):
    space = _structured_space(count)
    constraint = phx.discretization.virtual_element_dirichlet_constraint(space, "u")
    source = phx.equations.coefficient(
        lambda points, _args: (
            2.0
            * math.pi**2
            * jnp.sin(math.pi * points[..., 0])
            * jnp.sin(math.pi * points[..., 1])
        ),
        coefficient_id="manufactured-sine-source",
    )
    form = phx.equations.VirtualElementForm(
        "manufactured-poisson",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction("u", source),
        ),
    )
    compiled = phx.equations.compile_virtual_element_problem(
        form,
        space,
        constraint=constraint,
        dirichlet_values=0.0,
    )
    problem, rhs = compiled.linear_system()
    solution = phx.linalg.solve(problem, rhs)
    full = compiled.expand(solution.value)
    reconstruction = phx.equations.project_virtual_element_field(space, full)
    errors = []
    for block, geometry in enumerate(space.default_runtime.geometries):
        points = geometry.centroids[:, None, :]
        value, _ = phx.equations.evaluate_virtual_element_reconstruction(
            reconstruction, space, block, points
        )
        exact = jnp.sin(math.pi * points[:, 0, 0]) * jnp.sin(math.pi * points[:, 0, 1])
        errors.append(jnp.sum(geometry.areas * (value[:, 0] - exact) ** 2))
    return jnp.sqrt(sum(errors))


def test_virtual_element_poisson_error_decreases_under_refinement():
    coarse = _poisson_error(2)
    fine = _poisson_error(4)
    assert fine < 0.75 * coarse


def test_virtual_element_heat_dae_and_eigen_operators_are_executable():
    space = _structured_space(2)
    constraint = phx.discretization.virtual_element_dirichlet_constraint(space, "u")
    form = phx.equations.VirtualElementForm(
        "diffusion",
        "u",
        (phx.equations.DiffusionAction("u", 1.0),),
    )
    compiled = phx.equations.compile_virtual_element_problem(
        form,
        space,
        constraint=constraint,
        dirichlet_values=0.0,
    )
    dae = compiled.as_dae_system()
    state = jnp.zeros(compiled.state_space.shape)
    residual = dae.evaluate(0.0, state, jnp.zeros_like(state))
    eigenproblem = compiled.as_generalized_eigenproblem()
    vector = jnp.ones(compiled.state_space.shape)

    assert jnp.allclose(residual, 0.0)
    assert jnp.all(jnp.isfinite(eigenproblem.operator.mv(vector)))
    assert jnp.all(jnp.isfinite(eigenproblem.metric_operator.mv(vector)))
