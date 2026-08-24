#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _problem():
    return phx.control.LinearQuadraticControlProblem(
        jnp.ones((1, 1, 1)),
        jnp.ones((1, 1, 1)),
        jnp.asarray([1.0]),
        jnp.ones((1, 1, 1)),
        jnp.ones((1, 1, 1)),
        jnp.ones((1, 1)),
    )


def _control_norm_constraint(limit=0.5):
    return phx.control.StageSecondOrderConstraint(
        jnp.zeros((1, 1, 1)),
        jnp.ones((1, 1, 1)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1)),
        jnp.asarray([limit]),
        label="control-norm",
    )


def test_linear_conic_compiler_preserves_soc_block_and_decision_layout():
    compilation = phx.control.compile_linear_conic_control(
        _problem(),
        stage_constraints=(_control_norm_constraint(),),
    )
    conic = compilation.conic_program
    decision = compilation.decision_layout
    rows = compilation.stage_soc_slices[0][0]
    candidate = decision.encode(jnp.asarray([[1.0], [0.75]]), jnp.asarray([[-0.25]]))
    slack = conic.constraint_rhs[rows] - conic.constraint_matrix[rows] @ candidate

    assert conic.cone.cones[-1].contains(slack, tolerance=1e-12)
    np.testing.assert_allclose(slack, [0.5, -0.25], atol=1e-12)
    assert compilation.quadratic_compilation.qp.num_user_inequalities == 0


def test_stage_soc_constraint_shape_mismatch_is_rejected():
    invalid = phx.control.StageSecondOrderConstraint(
        jnp.zeros((2, 1, 1)),
        jnp.zeros((2, 1, 1)),
        jnp.zeros((2, 1)),
        jnp.zeros((2, 1)),
        jnp.zeros((2, 1)),
        jnp.zeros(2),
    )
    with pytest.raises(ValueError, match="stage left_state"):
        phx.control.compile_linear_conic_control(
            _problem(),
            stage_constraints=(invalid,),
        )


def test_clarabel_solves_control_socp_when_installed():
    pytest.importorskip("clarabel")
    policy = phx.optim.ConvexSolvePolicy(
        phx.optim.ClarabelInteriorPoint(presolve=False),
        termination=phx.optim.ConvexTermination(absolute=1e-8, relative=1e-8),
    )
    result = phx.control.solve_linear_conic_control(
        _problem(),
        policy,
        stage_constraints=(_control_norm_constraint(),),
    )

    assert result.status == phx.optim.ConvexProgramStatus.OPTIMAL
    assert jnp.abs(result.controls[0, 0]) <= 0.5 + 1e-7
    assert result.conic_result.kkt_residual_norm < 1e-7


def test_active_soc_control_limit_has_regular_canonical_sensitivity():
    pytest.importorskip("clarabel")
    policy = phx.optim.ConvexSolvePolicy(
        phx.optim.ClarabelInteriorPoint(presolve=False),
        termination=phx.optim.ConvexTermination(absolute=1e-9, relative=1e-9),
    )
    limit = 0.4
    compilation = phx.control.compile_linear_conic_control(
        _problem(),
        stage_constraints=(_control_norm_constraint(limit),),
    )
    prepared = phx.optim.prepare_convex_program(compilation.conic_program, policy)
    execution = phx.optim.solve_convex_program(prepared)
    sensitivity = phx.optim.prepare_conic_sensitivity(prepared, execution)
    zero = phx.optim.ConicProgramData.zeros_like(compilation.conic_program)
    rows = compilation.stage_soc_slices[0][0]
    tangent = phx.optim.ConicProgramData(
        zero.quadratic,
        zero.linear,
        zero.constraint_matrix,
        zero.constraint_rhs.at[rows.start].set(1.0),
        zero.lower_bounds,
        zero.upper_bounds,
    )

    derivative = phx.optim.conic_primal_jvp(sensitivity, tangent)
    _, control_tangent = compilation.decode(derivative.value)
    assert derivative.regular

    step = 1e-4
    lower = phx.control.solve_linear_conic_control(
        _problem(),
        policy,
        stage_constraints=(_control_norm_constraint(limit - step),),
    )
    upper = phx.control.solve_linear_conic_control(
        _problem(),
        policy,
        stage_constraints=(_control_norm_constraint(limit + step),),
    )
    finite_difference = (upper.controls - lower.controls) / (2.0 * step)
    np.testing.assert_allclose(
        control_tangent,
        finite_difference,
        atol=2e-4,
        rtol=2e-4,
    )
