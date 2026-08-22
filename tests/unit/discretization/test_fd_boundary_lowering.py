#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _grid():
    return phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))


def _context(grid, state, args=None, *, time=0.0):
    return phx.discretization.BoundaryStageContext(
        time,
        {"u": state},
        args,
        grid.axis_names,
        grid.primary_entity_layout.coordinates_by_axis,
        stage_id="test-stage",
    )


def _problem(expression, target, *, component=None):
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0))
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    region = phx.equations.PDERegion(
        "boundary",
        "boundary",
        ("x",),
        component=component,
    )
    return phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        conditions=(
            phx.equations.PDECondition(
                "condition",
                "boundary",
                expression,
                target=target,
                region="boundary",
                coordinate="x",
            ),
        ),
        regions=(region,),
    )


def test_dirichlet_condition_lowers_to_executable_cell_ghost_pair():
    u = phx.equations.PDEExpression.field("u")
    problem = _problem(u, phx.equations.PDEExpression.constant(2.0))

    grid = _grid()
    bindings = phx.equations.lower_fd_boundaries(problem, grid)
    runtime = phx.equations.prepare_fd_boundary_runtime(grid, bindings, "u")
    values = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    result = runtime[0].apply(values, _context(grid, values))

    assert len(bindings) == 2
    assert all(binding.kind == "dirichlet" for binding in bindings)
    assert jnp.allclose(result, jnp.asarray([3.0, 1.0, 2.0, 3.0, 4.0, 0.0]))


def test_neumann_and_robin_forms_preserve_coefficients_and_runtime_parameters():
    u = phx.equations.PDEExpression.field("u")
    neumann = _problem(
        u.derivative("x"),
        phx.equations.PDEExpression.constant(3.0),
    )
    robin = _problem(
        2.0 * u + 3.0 * u.derivative("x"),
        phx.equations.PDEExpression.parameter("g"),
    )

    grid = _grid()
    neumann_bindings = phx.equations.lower_fd_boundaries(neumann, grid)
    robin_bindings = phx.equations.lower_fd_boundaries(robin, grid)
    runtime = phx.equations.prepare_fd_boundary_runtime(
        grid,
        robin_bindings,
        "u",
    )[0]
    values = jnp.ones((4,))
    result = runtime.apply(values, _context(grid, values, {"g": 5.0}))

    assert all(binding.kind == "neumann" for binding in neumann_bindings)
    assert all(binding.kind == "robin" for binding in robin_bindings)
    assert all(binding.alpha == 2.0 and binding.beta == 3.0 for binding in robin_bindings)
    assert jnp.all(jnp.isfinite(result))


def test_one_sided_region_cannot_silently_invent_missing_boundary_condition():
    u = phx.equations.PDEExpression.field("u")
    problem = _problem(
        u,
        phx.equations.PDEExpression.constant(0.0),
        component="left",
    )
    bindings = phx.equations.lower_fd_boundaries(problem, _grid())

    with pytest.raises(ValueError, match="one condition per side"):
        phx.equations.prepare_fd_boundary_runtime(_grid(), bindings, "u")
