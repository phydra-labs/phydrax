#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _context(grid, fields, args=None, *, time=0.0):
    return phx.discretization.BoundaryStageContext(
        time,
        fields,
        args,
        grid.axis_names,
        grid.primary_entity_layout.coordinates_by_axis,
        stage_id="boundary-program-test",
    )


def _diffusion_problem(lower_target, upper_target):
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0))
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    u = phx.equations.PDEExpression.field("u")
    return phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        equations=(
            phx.equations.PDEEquation(
                "diffusion",
                u.derivative("t"),
                u.derivative("x", order=2),
            ),
        ),
        conditions=(
            phx.equations.PDECondition(
                "lower",
                "boundary",
                u,
                target=lower_target,
                region="lower",
                coordinate="x",
            ),
            phx.equations.PDECondition(
                "upper",
                "boundary",
                u,
                target=upper_target,
                region="upper",
                coordinate="x",
            ),
        ),
        regions=(
            phx.equations.PDERegion(
                "lower",
                "boundary",
                ("x",),
                component="lower",
            ),
            phx.equations.PDERegion(
                "upper",
                "boundary",
                ("x",),
                component="upper",
            ),
        ),
    )


def test_arbitrary_depth_cell_ghosts_extend_linear_field_exactly():
    runtime = phx.discretization.CellGhostBoundary(
        0,
        "dirichlet",
        "neumann",
        1.0,
        lower_width=2,
        upper_width=2,
    )

    padded = runtime.fill(jnp.asarray([0.5, 1.5, 2.5]), 0.0, 1.0)

    np.testing.assert_allclose(
        padded,
        jnp.asarray([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]),
        rtol=0.0,
        atol=0.0,
    )


def test_stage_boundary_program_evaluates_time_parameter_and_tangential_coordinate():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4),
            phx.discretization.UniformCellAxisSpec(3),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, -1.0], [1.0, 2.0]]))
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0))
    y = phx.equations.PDECoordinate("y", "space", bounds=(-1.0, 2.0))
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    u_field = phx.equations.PDEField("u", coordinates=("x", "y", "t"))
    u = phx.equations.PDEExpression.field("u")
    target = phx.equations.PDEExpression.coordinate_value(
        "t"
    ).sin() + phx.equations.PDEExpression.parameter(
        "scale"
    ) * phx.equations.PDEExpression.coordinate_value("y")
    problem = phx.equations.PDEProblemIR(
        coordinates=(x, y, t),
        fields=(u_field,),
        conditions=(
            phx.equations.PDECondition(
                "dynamic",
                "boundary",
                u,
                target=target,
                region="x-boundary",
                coordinate="x",
            ),
        ),
        regions=(phx.equations.PDERegion("x-boundary", "boundary", ("x",)),),
    )
    bindings = phx.equations.lower_fd_boundaries(problem, grid)
    program = phx.equations.prepare_fd_boundary_program(
        grid,
        bindings,
        ("u",),
        ghost_widths={"x": (2, 2)},
    )
    values = jnp.zeros(grid.shape)
    context = _context(grid, {"u": values}, {"scale": 2.0}, time=0.3)

    workspace = program.workspace("u", values, context)
    lower, upper = workspace.target_values("x")
    expected = jnp.sin(0.3) + 2.0 * grid.axes[1].nodes

    np.testing.assert_allclose(lower, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(upper, expected, rtol=2e-12, atol=2e-12)
    assert workspace.for_axis("x").shape == (8, 3)


def test_corner_policy_requires_explicit_tensor_product_realization():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4),
            phx.discretization.UniformCellAxisSpec(5),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    u = phx.equations.PDEExpression.field("u")
    conditions = []
    regions = []
    for axis in ("x", "y"):
        regions.append(phx.equations.PDERegion(axis, "boundary", (axis,)))
        conditions.append(
            phx.equations.PDECondition(
                axis,
                "boundary",
                u,
                target=phx.equations.PDEExpression.constant(0.0),
                region=axis,
                coordinate=axis,
            )
        )
    problem = phx.equations.PDEProblemIR(
        coordinates=(
            phx.equations.PDECoordinate("x", "space"),
            phx.equations.PDECoordinate("y", "space"),
        ),
        fields=(phx.equations.PDEField("u", coordinates=("x", "y")),),
        conditions=tuple(conditions),
        regions=tuple(regions),
    )
    bindings = phx.equations.lower_fd_boundaries(problem, grid)
    values = jnp.ones(grid.shape)
    context = _context(grid, {"u": values})
    separable = phx.equations.prepare_fd_boundary_program(
        grid,
        bindings,
        ("u",),
    )
    tensor = phx.equations.prepare_fd_boundary_program(
        grid,
        bindings,
        ("u",),
        corner_policy="tensor_product",
    )

    with pytest.raises(ValueError, match="tensor_product"):
        separable.workspace("u", values, context, require_tensor=True)
    workspace = tensor.workspace("u", values, context, require_tensor=True)

    assert workspace.tensor_values is not None
    assert workspace.tensor_values.shape == (6, 7)


def test_conforming_interface_runtime_enforces_field_and_outward_flux_jumps():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(5),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    u = phx.equations.PDEExpression.field("u")
    problem = phx.equations.PDEProblemIR(
        coordinates=(phx.equations.PDECoordinate("x", "space"),),
        fields=(phx.equations.PDEField("u", coordinates=("x",)),),
        conditions=(
            phx.equations.PDECondition(
                "field-jump",
                "interface",
                u,
                target=phx.equations.PDEExpression.parameter("field_jump"),
                region="middle",
                coordinate="x",
            ),
            phx.equations.PDECondition(
                "flux-jump",
                "interface",
                u.derivative("x"),
                target=phx.equations.PDEExpression.parameter("flux_jump"),
                region="middle",
                coordinate="x",
            ),
        ),
        regions=(phx.equations.PDERegion("middle", "interface", ("x",)),),
    )
    prepared = phx.equations.prepare_fd_interfaces(
        grid,
        phx.equations.lower_fd_interfaces(problem, grid),
    )[0]
    context = _context(
        grid,
        {"u": jnp.zeros(grid.shape)},
        {"field_jump": 2.0, "flux_jump": 3.0},
    )

    left, right, left_flux, right_flux = prepared.couple(
        jnp.asarray(1.0),
        jnp.asarray(5.0),
        jnp.asarray(4.0),
        jnp.asarray(-2.0),
        context,
    )

    np.testing.assert_allclose(right - left, 2.0)
    np.testing.assert_allclose(left_flux + right_flux, 3.0)


def test_native_cell_compiler_uses_boundary_ghosts_for_second_derivative():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(32),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    compiled = phx.equations.compile_finite_difference_pde(
        _diffusion_problem(
            phx.equations.PDEExpression.constant(0.0),
            phx.equations.PDEExpression.constant(1.0),
        ),
        grid,
    )
    state = grid.axes[0].nodes

    derivative = compiled(jnp.asarray(0.0), state, None)

    np.testing.assert_allclose(derivative, 0.0, rtol=0.0, atol=2e-11)


def test_native_nodal_compiler_differentiates_time_dependent_dirichlet_data():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(33),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    time_target = phx.equations.PDEExpression.coordinate_value("t")
    compiled = phx.equations.compile_finite_difference_pde(
        _diffusion_problem(time_target, time_target),
        grid,
    )
    state = jnp.full(grid.shape, 0.3)

    derivative = compiled(jnp.asarray(0.3), state, None)

    np.testing.assert_allclose(derivative[0], 1.0, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(derivative[-1], 1.0, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(derivative[1:-1], 0.0, rtol=0.0, atol=2e-12)
