from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


fd = phx.applications.solid_mechanics


def _edge_structure(*, all_fixed: bool = True):
    return fd.ForceDensityStructure.from_edges(
        jnp.asarray(((0, 1),), dtype=jnp.int32),
        2,
        2,
        fixed_nodes=(0, 1) if all_fixed else (0,),
    )


def test_fixed_and_reference_edge_loads_conserve_total_force():
    structure = _edge_structure()
    positions = jnp.asarray(((0.0, 0.0), (2.0, 0.0)))
    fixed = fd.FixedNodalLoadModel()
    fixed_parameters = jnp.asarray(((1.0, -2.0), (3.0, 4.0)))
    assert jnp.array_equal(
        fixed.nodal_loads(structure, positions, fixed_parameters), fixed_parameters
    )

    line = fd.EdgeLineLoadModel(
        measure="reference", reference_lengths=jnp.asarray((2.0,))
    )
    parameters = jnp.asarray(((0.0, -3.0),))
    nodal = line.nodal_loads(structure, positions, parameters)
    assert jnp.allclose(nodal, jnp.asarray(((0.0, -3.0), (0.0, -3.0))))
    assert jnp.allclose(jnp.sum(nodal, axis=0), jnp.asarray((0.0, -6.0)))
    assert bool(line.valid(structure, positions, parameters))


def test_current_edge_load_tracks_length_and_orientation_without_changing_total():
    forward = _edge_structure()
    reverse = fd.ForceDensityStructure.from_edges(
        jnp.asarray(((1, 0),), dtype=jnp.int32),
        2,
        2,
        fixed_nodes=(0, 1),
    )
    positions = jnp.asarray(((0.0, 0.0), (3.0, 4.0)))
    parameters = jnp.asarray(((2.0, -1.0),))
    model = fd.EdgeLineLoadModel(measure="current")
    expected = jnp.asarray(((5.0, -2.5), (5.0, -2.5)))
    assert jnp.allclose(model.nodal_loads(forward, positions, parameters), expected)
    assert jnp.allclose(model.nodal_loads(reverse, positions, parameters), expected)
    assert bool(model.valid(forward, positions, parameters))


def test_surface_pressure_integrates_oriented_triangle_and_quadrilateral():
    triangle_connectivity = phx.discretization.polygonal_connectivity(
        jnp.asarray(((0, 1, 2),), dtype=jnp.int32),
        None,
        3,
    )
    triangle_structure = fd.ForceDensityStructure.from_edges(
        triangle_connectivity.edges,
        3,
        3,
        fixed_nodes=(0, 1, 2),
        surface_connectivity=triangle_connectivity,
    )
    triangle_points = jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    model = fd.SurfacePressureLoadModel()
    triangle_loads = model.nodal_loads(
        triangle_structure, triangle_points, jnp.asarray((2.0,))
    )
    assert jnp.allclose(
        triangle_loads,
        jnp.asarray(((0.0, 0.0, 1.0 / 3.0),) * 3),
    )
    assert bool(model.valid(triangle_structure, triangle_points, jnp.asarray((0.0,))))

    quad_connectivity = phx.discretization.polygonal_connectivity(
        None,
        jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
        4,
    )
    quad_structure = fd.ForceDensityStructure.from_edges(
        quad_connectivity.edges,
        4,
        3,
        fixed_nodes=(0, 1, 2, 3),
        surface_connectivity=quad_connectivity,
    )
    quad_points = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )
    quad_loads = model.nodal_loads(quad_structure, quad_points, jnp.asarray((3.0,)))
    assert jnp.allclose(jnp.sum(quad_loads, axis=0), jnp.asarray((0.0, 0.0, 3.0)))
    assert jnp.allclose(quad_loads[:, 2], 0.75)


def test_surface_pressure_orientation_reversal_flips_resultant():
    forward = phx.discretization.polygonal_connectivity(
        jnp.asarray(((0, 1, 2),), dtype=jnp.int32), None, 3
    )
    reverse = phx.discretization.polygonal_connectivity(
        jnp.asarray(((0, 2, 1),), dtype=jnp.int32), None, 3
    )
    points = jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    model = fd.SurfacePressureLoadModel()

    def resultant(connectivity):
        structure = fd.ForceDensityStructure.from_edges(
            connectivity.edges,
            3,
            3,
            fixed_nodes=(0, 1, 2),
            surface_connectivity=connectivity,
        )
        return jnp.sum(model.nodal_loads(structure, points, jnp.asarray((1.0,))), axis=0)

    assert jnp.allclose(resultant(reverse), -resultant(forward))


def _current_line_problem(line_load: float):
    structure = fd.ForceDensityStructure.from_edges(
        jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
        3,
        2,
        fixed_nodes=(0, 2),
    )
    initial = jnp.asarray(((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)))
    problem = fd.ForceDensityProblem(
        structure,
        load_model=fd.EdgeLineLoadModel(measure="current"),
        sign_mode="tension",
        problem_id="current-line-cable",
    )
    inputs = fd.ForceDensityInputs(
        jnp.asarray((10.0, 10.0)),
        structure.prescribed_values(initial),
        jnp.asarray(((0.0, line_load), (0.0, line_load))),
    )
    return problem, inputs, initial


def test_position_dependent_edge_load_uses_nonlinear_root_and_certifies_residual():
    problem, inputs, initial = _current_line_problem(-0.1)
    result = fd.force_density_equilibrium(
        problem,
        inputs,
        initial_positions=initial,
        nonlinear_termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=1.0e-11,
            relative_residual=0.0,
            maximum_steps=30,
        ),
    )
    assert result.successful
    assert result.nonlinear_result is not None
    assert result.linear_result is None
    assert result.state.positions[1, 1] < 0.0
    assert result.diagnostics.free_residual_norm <= 1.0e-9
    assert result.nonlinear_result.diagnostics.residual_evaluations >= 1


def test_position_dependent_solve_has_implicit_load_derivative():
    problem, sample, initial = _current_line_problem(-0.1)
    plan = fd.plan_force_density(problem, sample, initial_positions=initial)

    def center_height(line_load):
        inputs = fd.ForceDensityInputs(
            sample.force_densities,
            sample.prescribed_values,
            jnp.full((2, 2), 0.0).at[:, 1].set(line_load),
        )
        result = fd.solve_force_density(
            fd.prepare_force_density(plan, inputs, initial_positions=initial)
        )
        return result.state.positions[1, 1]

    derivative = jax.grad(center_height)(jnp.asarray(-0.1))
    epsilon = 1.0e-4
    finite_difference = (
        center_height(-0.1 + epsilon) - center_height(-0.1 - epsilon)
    ) / (2.0 * epsilon)
    assert jnp.isfinite(derivative)
    assert derivative == pytest.approx(finite_difference, rel=2.0e-3, abs=2.0e-5)


def test_composite_load_model_sums_children_and_preserves_dependency():
    structure = _edge_structure()
    positions = jnp.asarray(((0.0, 0.0), (2.0, 0.0)))
    model = fd.CompositeForceDensityLoadModel(
        (fd.FixedNodalLoadModel(), fd.EdgeLineLoadModel(measure="current"))
    )
    parameters = (
        jnp.asarray(((1.0, 0.0), (0.0, 0.0))),
        jnp.asarray(((0.0, -2.0),)),
    )
    expected = jnp.asarray(((1.0, -2.0), (0.0, -2.0)))
    assert model.depends_on_positions
    assert jnp.allclose(model.nodal_loads(structure, positions, parameters), expected)
    assert bool(model.valid(structure, positions, parameters))
