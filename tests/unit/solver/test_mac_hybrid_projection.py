#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.linalg._transform_line import TransformLineNullspacePolicy


def _operators(
    *,
    uniform_line=False,
    dimension=3,
    periodic_line=False,
    nonuniform_transverse=False,
):
    x = (
        phx.discretization.NonuniformCellAxisSpec(
            jnp.asarray([0.0, 0.12, 0.36, 0.68, 1.0]), periodic=True
        )
        if nonuniform_transverse
        else phx.discretization.UniformCellAxisSpec(4, periodic=True)
    )
    y = (
        phx.discretization.UniformCellAxisSpec(5, periodic=periodic_line)
        if uniform_line or periodic_line
        else phx.discretization.NonuniformCellAxisSpec(
            jnp.asarray([0.0, 0.08, 0.24, 0.52, 0.78, 1.0])
        )
    )
    axes = (x, y)
    names = ("x", "y")
    lower = (0.0, 0.0)
    upper = (2.0 * jnp.pi, 1.0)
    if dimension == 3:
        axes = axes + (phx.discretization.UniformCellAxisSpec(4, periodic=True),)
        names = names + ("z",)
        lower = lower + (0.0,)
        upper = upper + (2.0 * jnp.pi,)
    grid = phx.discretization.TensorGridPlan(axes, axis_names=names).prepare(
        jnp.asarray((lower, upper))
    )
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    return grid, operators


def _pressure_probe(operators):
    shape = operators.discretization.cell_shape
    pressure = jnp.sin(
        0.23 * jnp.arange(int(np.prod(shape)), dtype=operators.pressure_space.dtype)
    ).reshape(shape)
    return operators.gauge_project(pressure)


def _assert_projection_evidence(result, operators, tolerance):
    volumes = operators.discretization.cell_volumes
    assert result.converged
    assert result.closure.successful
    assert result.closure.kind == "neumann"
    assert result.closure.gauge == "zero-mean"
    assert result.closure.compatibility == "projected"
    assert jnp.abs(jnp.sum(volumes * result.pressure)) < tolerance
    assert jnp.sqrt(jnp.sum(volumes * result.divergence_after**2)) < tolerance
    assert result.closure.mass_defect < tolerance
    assert jnp.sqrt(jnp.sum(volumes * result.pressure_residual**2)) < tolerance


def test_pressure_hybrid_stretched_periodic_channel_and_global_mode_evidence():
    _, operators = _operators()
    plan = phx.solver.MACPressureProjectionPlan(
        operators,
        solve_method="hybrid",
        hybrid_line_axis=1,
        tolerance=1e-9,
        maximum_resource_bytes=2_000_000,
    )
    prepared = plan.hybrid_plan
    factors = prepared.factors
    global_mode = prepared.solve(jnp.ones(operators.discretization.cell_shape))

    assert plan.constant_route == "hybrid"
    assert factors.zero_mode_index == 0
    assert factors.pin_row == 0
    assert factors.right_null is not None
    assert factors.left_null is not None
    assert factors.nullspace_policy_id == prepared.plan.nullspace.policy_id
    assert global_mode.converged
    np.testing.assert_allclose(global_mode.compatible_rhs, 0.0, atol=1e-12)
    np.testing.assert_allclose(global_mode.value, 0.0, atol=1e-12)
    assert global_mode.compatibility_defect < 1e-12
    assert global_mode.gauge_defect < 1e-12
    assert global_mode.resources.total_bytes <= 2_000_000


def test_uniform_pressure_hybrid_matches_full_transform_and_auto_prefers_full():
    _, operators = _operators(uniform_line=True)
    pressure = _pressure_probe(operators)
    velocity = operators.gradient(pressure)
    hybrid = phx.solver.MACPressureProjectionPlan(
        operators,
        solve_method="hybrid",
        hybrid_line_axis=1,
        tolerance=1e-9,
    ).project(velocity, 0.2)
    transform = phx.solver.MACPressureProjectionPlan(
        operators, solve_method="transform", tolerance=1e-9
    ).project(velocity, 0.2)
    automatic = phx.solver.MACPressureProjectionPlan(
        operators,
        solve_method="auto",
        hybrid_line_axis=1,
        tolerance=1e-9,
    ).project(velocity, 0.2)

    assert hybrid.solve_method == "hybrid"
    assert hybrid.hybrid is not None and hybrid.transform is None
    assert transform.solve_method == "transform"
    assert automatic.solve_method == "transform"
    for hybrid_component, transform_component in zip(
        hybrid.velocity, transform.velocity, strict=True
    ):
        np.testing.assert_allclose(
            hybrid_component, transform_component, rtol=2e-8, atol=2e-8
        )


def test_stretched_pressure_hybrid_matches_iterative_projection_and_rate():
    _, operators = _operators()
    pressure = _pressure_probe(operators)
    velocity = operators.gradient(pressure)
    hybrid_plan = phx.solver.MACPressureProjectionPlan(
        operators,
        solve_method="auto",
        hybrid_line_axis=1,
        tolerance=1e-9,
    )
    iterative_plan = phx.solver.MACPressureProjectionPlan(
        operators, solve_method="iterative", tolerance=1e-9
    )
    hybrid = hybrid_plan.project(velocity, 0.2)
    iterative = iterative_plan.project(velocity, 0.2)
    rate = hybrid_plan.project_rate(velocity)

    assert hybrid.solve_method == "hybrid"
    assert hybrid.hybrid is not None
    assert hybrid.linear is None and hybrid.transform is None
    assert rate.solve_method == "hybrid"
    assert rate.hybrid is not None
    assert rate.hybrid_line_axis == 1
    assert hybrid.hybrid_action_defect < 2e-8
    assert rate.hybrid_action_defect == hybrid.hybrid_action_defect
    assert hybrid.maximum_resource_bytes == hybrid_plan.maximum_resource_bytes
    _assert_projection_evidence(hybrid, operators, 2e-8)
    _assert_projection_evidence(rate, operators, 2e-8)
    for hybrid_component, iterative_component in zip(
        hybrid.velocity, iterative.velocity, strict=True
    ):
        np.testing.assert_allclose(
            hybrid_component, iterative_component, rtol=2e-7, atol=2e-7
        )


def test_pressure_hybrid_rhs_jvp_and_vjp_obey_the_adjoint_identity():
    _, operators = _operators()
    prepared = phx.solver.MACPressureProjectionPlan(
        operators,
        solve_method="hybrid",
        hybrid_line_axis=1,
        tolerance=1e-9,
    ).hybrid_plan
    shape = operators.discretization.cell_shape
    count = int(np.prod(shape))
    rhs = jnp.sin(0.17 * jnp.arange(count)).reshape(shape)
    tangent = jnp.cos(0.11 * jnp.arange(count)).reshape(shape)
    cotangent = jnp.sin(0.07 * jnp.arange(count) + 0.3).reshape(shape)

    def solve_value(value):
        return prepared.solve(value).value

    value, jvp = jax.jvp(solve_value, (rhs,), (tangent,))
    _, pullback = jax.vjp(solve_value, rhs)
    vjp = pullback(cotangent)[0]
    solved = prepared.solve(rhs)

    assert jnp.all(jnp.isfinite(value))
    assert jnp.all(jnp.isfinite(jvp))
    assert jnp.all(jnp.isfinite(vjp))
    np.testing.assert_allclose(
        solved.residual,
        prepared.plan.representation.apply(solved.candidate) - solved.compatible_rhs,
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        jnp.vdot(cotangent, jvp),
        jnp.vdot(vjp, tangent),
        rtol=2e-8,
        atol=2e-8,
    )


def test_execution_supplied_line_coefficient_retains_certified_hybrid_route():
    _, operators = _operators()
    plan = phx.solver.MACPressureProjectionPlan(
        operators,
        solve_method="hybrid",
        hybrid_line_axis=1,
        tolerance=1e-9,
    )
    pressure = _pressure_probe(operators)
    beta = (
        0.15
        + 0.05 * jnp.arange(operators.discretization.cell_shape[1], dtype=pressure.dtype)
    ).reshape((1, -1, 1))
    beta = jnp.broadcast_to(beta, operators.discretization.cell_shape)
    face_beta = operators.interpolate_inverse_momentum(beta)
    velocity = tuple(
        coefficient * derivative
        for coefficient, derivative in zip(
            face_beta, operators.gradient(pressure), strict=True
        )
    )
    result = plan.project(
        velocity,
        0.2,
        inverse_momentum_diagonal=beta,
    )

    assert result.solve_method == "hybrid"
    assert result.linear is None
    assert result.hybrid is not None and result.transform is None
    assert result.hybrid_action_defect < 2e-8
    assert result.converged


def test_pressure_hybrid_rejects_every_uncertified_preparation_predicate():
    _, operators = _operators()
    with pytest.raises(ValueError, match="explicit hybrid_line_axis"):
        phx.solver.MACPressureProjectionPlan(operators, solve_method="hybrid")

    _, two_dimensional = _operators(dimension=2)
    with pytest.raises(ValueError, match="three-dimensional"):
        phx.solver.MACPressureProjectionPlan(
            two_dimensional, solve_method="hybrid", hybrid_line_axis=1
        )

    _, periodic_line = _operators(uniform_line=True, periodic_line=True)
    with pytest.raises(ValueError, match="nonperiodic line"):
        phx.solver.MACPressureProjectionPlan(
            periodic_line, solve_method="hybrid", hybrid_line_axis=1
        )

    _, bad_transverse = _operators(nonuniform_transverse=True)
    with pytest.raises(ValueError, match="transform-compatible transverse"):
        phx.solver.MACPressureProjectionPlan(
            bad_transverse, solve_method="hybrid", hybrid_line_axis=1
        )

    boundaries = phx.discretization.MACBoundaryPlan(
        operators,
        (
            phx.discretization.MACBoundarySide("y", "lower", "no-slip"),
            phx.discretization.MACBoundarySide("y", "upper", "pressure-outlet"),
        ),
    ).prepare()
    with pytest.raises(ValueError, match="all-Neumann"):
        phx.solver.MACPressureProjectionPlan(
            operators,
            boundaries=boundaries,
            solve_method="hybrid",
            hybrid_line_axis=1,
        )

    with pytest.raises(ValueError, match="resources"):
        phx.solver.MACPressureProjectionPlan(
            operators,
            solve_method="hybrid",
            hybrid_line_axis=1,
            maximum_resource_bytes=1,
        )


def test_transform_line_nullspace_preparation_rejects_ambiguous_or_false_data():
    line_lower = -jnp.ones(2)
    line_diagonal = jnp.asarray([1.0, 2.0, 1.0])
    line_upper = -jnp.ones(2)
    duplicate_zero = phx.linalg.TransformLineRepresentation(
        (phx.linalg.FFTLinearTransform(4),),
        1,
        line_lower,
        line_diagonal,
        line_upper,
        jnp.asarray([0.0, 0.0, 4.0, 2.0]),
    )
    with pytest.raises(ValueError, match="one declared all-zero"):
        phx.linalg.TransformLineSolvePlan(
            duplicate_zero,
            nullspace=TransformLineNullspacePolicy(jnp.ones(3)),
        ).prepare()

    false_left_null = phx.linalg.TransformLineRepresentation(
        (phx.linalg.FFTLinearTransform(4),),
        1,
        line_lower,
        line_diagonal,
        line_upper,
        jnp.asarray([0.0, 2.0, 4.0, 2.0]),
    )
    with pytest.raises(ValueError, match="right/left constant null"):
        phx.linalg.TransformLineSolvePlan(
            false_left_null,
            nullspace=TransformLineNullspacePolicy(jnp.asarray([1.0, 2.0, 1.0])),
        ).prepare()
