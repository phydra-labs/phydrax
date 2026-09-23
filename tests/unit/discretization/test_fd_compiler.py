#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _heat_problem(*, periodic, boundary=False):
    x = phx.equations.PDECoordinate(
        "x",
        "space",
        bounds=(0.0, 1.0),
        periodic=periodic,
    )
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    kappa = phx.equations.PDEParameter("kappa", value=0.2)
    u = phx.equations.PDEExpression.field("u")
    regions = ()
    conditions = ()
    if boundary:
        regions = (phx.equations.PDERegion("boundary", "boundary", ("x",)),)
        conditions = (
            phx.equations.PDECondition(
                "dirichlet",
                "boundary",
                u,
                target=phx.equations.PDEExpression.constant(0.0),
                region="boundary",
                coordinate="x",
            ),
        )
    return phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        parameters=(kappa,),
        equations=(
            phx.equations.PDEEquation(
                "heat",
                u.derivative("t"),
                phx.equations.PDEExpression.parameter("kappa") * u.laplacian("x"),
            ),
        ),
        conditions=conditions,
        regions=regions,
    )


def test_compile_semidiscrete_pde_dispatches_prepared_grid_to_native_fd():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(64, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    compiled = phx.equations.compile_semidiscrete_pde(
        _heat_problem(periodic=True),
        grid,
    )
    values = jnp.sin(2.0 * jnp.pi * grid.axes[0].nodes)
    state = compiled.layout.pack({"u": values})

    drift = compiled.drift(jnp.asarray(0.0), state, {"kappa": 0.2})
    expected = 0.2 * compiled.spatial_discretization.operator("d_x_2").mv(values)

    assert isinstance(compiled, phx.equations.CompiledFiniteDifferenceDynamics)
    assert jnp.allclose(drift[..., 0], expected)
    assert (
        compiled.discretization_bundle.record(
            compiled.spatial_discretization.key
        ).artifact_id
        == compiled.spatial_discretization.prepared_id
    )


def test_native_fd_nodal_dirichlet_boundary_constrains_state_and_derivative():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(33),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    compiled = phx.equations.compile_finite_difference_pde(
        _heat_problem(periodic=False, boundary=True),
        grid,
        policy=phx.equations.FiniteDifferenceCompilationPolicy(accuracy_order=4),
    )
    values = jnp.sin(jnp.pi * grid.axes[0].nodes).at[0].set(5.0).at[-1].set(-3.0)
    state = compiled.layout.pack({"u": values})

    drift = compiled.drift(jnp.asarray(0.0), state, {"kappa": 0.2})[..., 0]

    assert jnp.allclose(drift[jnp.asarray([0, -1])], 0.0)
    assert (
        jnp.max(
            jnp.abs(
                drift[2:-2] + 0.2 * jnp.pi**2 * jnp.sin(jnp.pi * grid.axes[0].nodes[2:-2])
            )
        )
        < 5e-4
    )


def test_native_fd_compiler_handles_pointwise_reaction_with_runtime_parameter():
    problem = _heat_problem(periodic=True)
    equation = problem.equations[0]
    u = phx.equations.PDEExpression.field("u")
    reaction_problem = phx.equations.PDEProblemIR(
        coordinates=problem.coordinates,
        fields=problem.fields,
        parameters=problem.parameters,
        equations=(
            phx.equations.PDEEquation(
                equation.name,
                equation.lhs,
                equation.rhs + u * (1.0 - u),
            ),
        ),
    )
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(32, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    compiled = phx.equations.compile_finite_difference_pde(reaction_problem, grid)
    values = jnp.full(grid.shape, 0.25)

    drift = compiled.drift(
        jnp.asarray(0.0),
        compiled.layout.pack({"u": values}),
        {"kappa": 0.0},
    )[..., 0]

    assert jnp.allclose(drift, 0.25 * 0.75)


def test_native_fd_round_trip_folds_every_associative_operand():
    problem = _heat_problem(periodic=True)
    equation = problem.equations[0]
    u = phx.equations.PDEExpression.field("u")
    expanded = phx.equations.PDEProblemIR(
        coordinates=problem.coordinates,
        fields=problem.fields,
        parameters=problem.parameters,
        equations=(
            phx.equations.PDEEquation(
                equation.name,
                equation.lhs,
                equation.rhs + u + (2.0 * u) * 3.0,
            ),
        ),
    )
    round_tripped = phx.equations.pde_ir_from_json(phx.equations.pde_ir_to_json(expanded))
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(16, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    compiled = phx.equations.compile_finite_difference_pde(round_tripped, grid)
    values = jnp.full(grid.shape, 0.25)

    drift = compiled.drift(
        jnp.asarray(0.0),
        compiled.layout.pack({"u": values}),
        {"kappa": 0.0},
    )[..., 0]

    assert jnp.allclose(drift, 7.0 * values)


def test_native_fd_rejects_component_valued_field_layouts():
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0), periodic=True)
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField(
        "u",
        representation="vector",
        components=2,
        coordinates=("x", "t"),
    )
    u = phx.equations.PDEExpression.field("u")
    problem = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        equations=(
            phx.equations.PDEEquation(
                "vector-heat",
                u.derivative("t"),
                u.laplacian("x"),
            ),
        ),
    )
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))

    with pytest.raises(ValueError, match="only scalar fields"):
        phx.equations.compile_finite_difference_pde(problem, grid)


def test_native_fd_validates_ir_before_lowering():
    problem = _heat_problem(periodic=True)
    equation = problem.equations[0]
    malformed = phx.equations.PDEProblemIR(
        coordinates=problem.coordinates,
        fields=problem.fields,
        parameters=problem.parameters,
        equations=(
            phx.equations.PDEEquation(
                equation.name,
                equation.lhs,
                phx.equations.PDEExpression.field("missing").laplacian("x"),
            ),
        ),
    )
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))

    with pytest.raises(ValueError, match="malformed field reference"):
        phx.equations.compile_finite_difference_pde(malformed, grid)


def test_fixed_order_pde_operations_reject_derivative_order_metadata():
    u = phx.equations.PDEExpression.field("u")

    with pytest.raises(ValueError, match="does not accept derivative-order"):
        phx.equations.PDEExpression(
            "gradient",
            (u,),
            coordinate="x",
            order=2,
        )
    with pytest.raises(TypeError, match="must be an integer"):
        phx.equations.PDEExpression(
            "derivative",
            (u,),
            coordinate="x",
            order=1.5,
        )
