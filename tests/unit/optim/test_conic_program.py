#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


pytest.importorskip("clarabel")


def _policy():
    return phx.optim.ConvexSolvePolicy(
        phx.optim.ClarabelInteriorPoint(presolve=False),
        termination=phx.optim.ConvexTermination(
            absolute=1e-8,
            relative=1e-8,
            maximum_steps=200,
        ),
    )


def test_clarabel_solves_active_second_order_cone_program():
    problem = phx.optim.ConicProgram(
        jnp.eye(2),
        jnp.asarray([-2.0, 0.0]),
        jnp.asarray([[0.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]),
        jnp.asarray([1.0, 0.0, 0.0]),
        phx.optim.SecondOrderCone(3),
        problem_id="active-soc",
    )
    result = phx.optim.solve_conic_program(problem, policy=_policy())

    np.testing.assert_allclose(result.primal, [1.0, 0.0], atol=2e-6)
    assert result.status == phx.optim.ConvexProgramStatus.OPTIMAL
    assert result.kkt_residual_norm < 1e-7
    assert result.cone_slack.shape == (3,)
    assert result.provenance.backend == "clarabel"


def test_clarabel_projects_advanced_cones_in_native_coordinates():
    psd = phx.optim.PositiveSemidefiniteCone(2)
    cases = (
        (psd, psd.pack(jnp.asarray([[1.0, 2.0], [2.0, -1.0]])), 1e-7),
        (phx.optim.ExponentialCone(), jnp.asarray([1.0, 2.0, 3.0]), 2e-6),
        (phx.optim.PowerCone(0.4), jnp.asarray([-1.0, 2.0, 1.0]), 2e-5),
    )
    for index, (cone, value, tolerance) in enumerate(cases):
        problem = phx.optim.ConicProgram(
            jnp.eye(cone.dimension),
            -value,
            -jnp.eye(cone.dimension),
            jnp.zeros(cone.dimension),
            cone,
            problem_id=f"advanced-cone-projection-{index}",
        )
        result = phx.optim.solve_conic_program(problem, policy=_policy())
        expected = cone.project(value)

        assert result.status == phx.optim.ConvexProgramStatus.OPTIMAL
        np.testing.assert_allclose(
            result.primal, expected, atol=tolerance, rtol=tolerance
        )
        np.testing.assert_allclose(
            result.cone_slack, expected, atol=tolerance, rtol=tolerance
        )
        assert cone.contains(result.cone_slack, tolerance=tolerance)
        assert cone.contains_dual(result.cone_dual, tolerance=tolerance)
        assert result.kkt_residual_norm < 5e-7


def test_clarabel_preserves_qp_user_constraint_axes():
    problem = phx.optim.QuadraticProgram(
        jnp.eye(2),
        jnp.zeros(2),
        equality_matrix=jnp.asarray([[1.0, 1.0]]),
        equality_rhs=jnp.asarray([1.0]),
        inequality_matrix=-jnp.eye(2),
        inequality_rhs=jnp.zeros(2),
        problem_id="clarabel-qp-axes",
    )
    result = phx.optim.solve_convex_program(problem, policy=_policy()).result

    np.testing.assert_allclose(result.primal, [0.5, 0.5], atol=2e-6)
    assert result.equality_dual.shape == (1,)
    assert result.inequality_dual.shape == (2,)
    assert result.inequality_slack.shape == (2,)
    np.testing.assert_allclose(result.equality_residual, 0.0, atol=1e-8)
    np.testing.assert_allclose(result.inequality_residual, 0.0, atol=1e-8)
    assert result.status == phx.optim.ConvexProgramStatus.OPTIMAL


def test_clarabel_prepared_refresh_reuses_provider_structure():
    def problem(linear):
        return phx.optim.ConicProgram(
            jnp.eye(2),
            jnp.asarray(linear),
            jnp.asarray([[0.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]),
            jnp.asarray([1.0, 0.0, 0.0]),
            phx.optim.SecondOrderCone(3),
            problem_id="prepared-soc",
        )

    prepared = phx.optim.prepare_convex_program(problem([-2.0, 0.0]), _policy())
    refreshed = phx.optim.refresh_convex_program(
        prepared,
        problem([-1.5, 0.0]),
    )
    execution = phx.optim.solve_convex_program(refreshed)

    assert refreshed.template is prepared.template
    assert refreshed.template.symbolic_state is prepared.template.symbolic_state
    assert execution.result.status == phx.optim.ConvexProgramStatus.OPTIMAL
    assert execution.result.provenance.numeric_version == 1


def test_fixed_bound_roles_participate_in_conic_structure_identity():
    def problem(lower, upper):
        return phx.optim.ConicProgram(
            jnp.ones((1, 1)),
            jnp.zeros(1),
            jnp.empty((0, 1)),
            jnp.empty((0,)),
            phx.optim.ProductCone(()),
            bounds=phx.optim.Bounds(lower, upper),
            problem_id="bound-role-identity",
        )

    fixed = problem(1.0, 1.0)
    interval = problem(0.0, 2.0)
    assert fixed.structure_id != interval.structure_id

    prepared = phx.optim.prepare_convex_program(fixed, _policy())
    with pytest.raises(ValueError, match="structure"):
        phx.optim.refresh_convex_program(prepared, interval)


def test_clarabel_maps_rotated_cone_and_native_bounds():
    problem = phx.optim.ConicProgram(
        jnp.eye(1),
        jnp.asarray([-3.0]),
        jnp.asarray([[0.0], [0.0], [-1.0]]),
        jnp.asarray([1.0, 1.0, 0.0]),
        phx.optim.RotatedSecondOrderCone(3),
        bounds=phx.optim.Bounds(0.0, 1.0),
        problem_id="rotated-bound",
    )
    result = phx.optim.solve_conic_program(problem, policy=_policy())

    assert result.status == phx.optim.ConvexProgramStatus.OPTIMAL
    assert 0.0 <= result.primal[0] <= 1.0 + 1e-7
    assert result.upper_bound_dual[0] > 0.0
    assert result.kkt_residual_norm < 1e-7
    expected_gap = problem.cone.complementarity(
        result.cone_slack,
        result.cone_dual,
    )
    expected_gap += result.primal[0] * result.lower_bound_dual[0]
    expected_gap += (1.0 - result.primal[0]) * result.upper_bound_dual[0]
    np.testing.assert_allclose(result.complementarity_gap, expected_gap, atol=1e-12)


def test_clarabel_infeasibility_requires_independent_dual_ray():
    problem = phx.optim.ConicProgram(
        None,
        jnp.zeros(1),
        jnp.asarray([[1.0], [1.0]]),
        jnp.asarray([0.0, 1.0]),
        phx.optim.ZeroCone(2),
        problem_id="infeasible-zero-cone",
    )
    result = phx.optim.solve_conic_program(problem, policy=_policy())

    assert result.status == phx.optim.ConvexProgramStatus.PRIMAL_INFEASIBLE
    assert result.certificate.dual_ray_valid
    assert result.certificate.dual_ray_residual_norm < 1e-7


def test_clarabel_preserves_program_batches():
    problem = phx.optim.ConicProgram(
        jnp.broadcast_to(jnp.eye(1), (2, 1, 1)),
        jnp.asarray([[-1.0], [-2.0]]),
        jnp.empty((0, 1)),
        jnp.empty((0,)),
        phx.optim.ProductCone(()),
        bounds=phx.optim.Bounds(0.0, 3.0),
        problem_id="batched-box-qp",
    )
    result = phx.optim.solve_conic_program(problem, policy=_policy())

    np.testing.assert_allclose(result.primal[:, 0], [1.0, 2.0], atol=2e-6)
    assert jnp.all(result.status == int(phx.optim.ConvexProgramStatus.OPTIMAL))
