#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import opt_einsum as oe

import phydrax as phx


def _structured_space(count, *, component_shape=()):
    coordinates = jnp.asarray(
        tuple((i / count, j / count) for j in range(count + 1) for i in range(count + 1))
    )
    cells = tuple(
        (
            j * (count + 1) + i,
            j * (count + 1) + i + 1,
            (j + 1) * (count + 1) + i + 1,
            (j + 1) * (count + 1) + i,
        )
        for j in range(count)
        for i in range(count)
    )
    mesh = phx.discretization.CellMesh.from_polygons(coordinates, cells)
    field = phx.discretization.ExplicitPolygonH1FieldSpec(
        "u", component_shape=component_shape
    )
    return phx.discretization.ExplicitPolygonH1Plan(mesh, field).prepare()


def _poisson_error(count):
    space = _structured_space(count)
    constraint = phx.discretization.explicit_polygon_h1_dirichlet_constraint(space, "u")

    def source(points, _args):
        x = points[..., 0]
        y = points[..., 1]
        return 2.0 * (x * (1.0 - x) + y * (1.0 - y))

    form = phx.equations.FiniteElementForm(
        "explicit-polygon-poisson",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    source, coefficient_id="explicit-polygon-poisson-source"
                ),
            ),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form, space, constraint=constraint, dirichlet_values=0.0
    )
    problem, right_hand_side = compiled.linear_system()
    result = phx.linalg.solve(problem, right_hand_side)
    solution = compiled.expand(result.value)
    block = space.default_runtime.bases[0]
    local = solution[space.dof_map.cell_dofs[0][:, : block.arity]]
    values = oe.contract("cqi,ci->cq", block.basis_values[..., : block.arity], local)
    points = block.physical_points
    exact = (
        points[..., 0] * (1.0 - points[..., 0]) * points[..., 1] * (1.0 - points[..., 1])
    )
    error = jnp.sqrt(jnp.sum(block.physical_weights * (values - exact) ** 2))
    return error, result.successful


def test_explicit_polygon_poisson_error_decreases_under_refinement():
    coarse, coarse_success = _poisson_error(2)
    fine, fine_success = _poisson_error(4)
    assert jnp.all(coarse_success) & jnp.all(fine_success)
    assert fine < 0.45 * coarse


def test_explicit_polygon_linear_elasticity_has_exact_rigid_rotation_mode():
    space = _structured_space(2, component_shape=(2,))
    coordinates = space.mesh.coordinates
    state = jnp.stack((-coordinates[:, 1], coordinates[:, 0]), axis=-1)
    form = phx.equations.fem.linear_elasticity_form("u", 2.0, 1.5)
    compiled = phx.equations.compile_finite_element_problem(form, space)
    residual = compiled.full_residual(state)
    assert jnp.allclose(residual, 0.0, atol=1e-10)


def test_explicit_polygon_neo_hookean_energy_residual_and_tangent_are_consistent():
    space = _structured_space(2, component_shape=(2,))
    parameters = phx.applications.solid_mechanics.NeoHookeanParameters(1.0, 2.0)
    functional = phx.applications.solid_mechanics.neo_hookean_functional("u", parameters)
    compiled = phx.equations.compile_finite_element_functional(
        functional,
        space,
        fields={"u": "u"},
        regions={"body": None},
    )
    coordinates = space.mesh.coordinates
    state = 0.03 * jnp.stack((coordinates[:, 0], -0.5 * coordinates[:, 1]), axis=-1)
    direction = jnp.stack((coordinates[:, 1], coordinates[:, 0]), axis=-1)
    value, residual = compiled.value_and_residual(state)
    tangent_direction = compiled.linearization_operator(state).mv(direction)
    epsilon = 1.0e-5
    perturbed_value, perturbed_residual = compiled.value_and_residual(
        state + epsilon * direction
    )
    first_order_value = value + epsilon * jnp.vdot(residual, direction).real
    first_order_residual = residual + epsilon * tangent_direction

    assert jnp.isfinite(value) & jnp.isfinite(perturbed_value)
    assert jnp.all(jnp.isfinite(residual))
    assert jnp.all(jnp.isfinite(tangent_direction))
    assert jnp.abs(perturbed_value - first_order_value) < 1.0e-8
    assert jnp.linalg.norm(perturbed_residual - first_order_residual) < 1.0e-8

    def geometry_response(coordinates_):
        runtime = space.prepare_runtime(coordinates_, numeric_version="differentiated")
        context = phx.equations.FiniteElementExecutionContext(runtime)
        return compiled.full_potential(state, context)

    geometry_gradient = jax.grad(geometry_response)(coordinates)
    assert jnp.all(jnp.isfinite(geometry_gradient))
