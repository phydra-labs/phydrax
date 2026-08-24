#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.optim._programming._clarabel import _audit_result


def _policy(*, regularization=0.0):
    return phx.optim.ConvexSolvePolicy(
        phx.optim.ClarabelInteriorPoint(presolve=False),
        termination=phx.optim.ConvexTermination(
            absolute=1e-9,
            relative=1e-9,
            maximum_steps=500,
        ),
        regularization=regularization,
    )


def _prepare(
    program,
    *,
    solution=None,
    regularization=0.0,
    linear=None,
    regularity_tolerance=1e-7,
):
    policy = _policy(regularization=regularization)
    if solution is None:
        pytest.importorskip("clarabel")
        prepared = phx.optim.prepare_convex_program(program, policy)
        execution = phx.optim.solve_convex_program(prepared)
    else:
        plan = phx.optim.plan_convex_program(program, policy)
        prepared = phx.optim.bind_convex_numeric(
            phx.optim.ConvexProgramTemplate(plan),
            program,
        )
        primal, slack, dual, lower_dual, upper_dual = solution
        result = _audit_result(
            program,
            jnp.asarray(primal),
            jnp.asarray(slack),
            jnp.asarray(dual),
            jnp.asarray(lower_dual),
            jnp.asarray(upper_dual),
            jnp.ones(program.batch_shape, dtype=bool),
            jnp.zeros(program.batch_shape, dtype=jnp.int32),
            policy,
            "analytic-test",
        )
        provenance = phx.optim.ConvexProgramProvenance(
            numeric_version=prepared.numeric_version,
            problem_id=program.problem_id,
            structure_id=program.structure_id,
            policy_id=policy.policy_id,
            method_id=policy.method.method_id,
            backend=result.provenance.backend,
            backend_version=result.provenance.backend_version,
            convexity_evidence=program.convexity_evidence,
            regularization=policy.regularization,
            numeric_binding_id=prepared.numeric_binding_id,
        )
        result = eqx.tree_at(lambda value: value.provenance, result, provenance)
        execution = phx.optim.ConvexProgramExecution(
            result,
            numeric_version=prepared.numeric_version,
            plan_id=prepared.plan.plan_id,
            numeric_binding_id=prepared.numeric_binding_id,
        )
    sensitivity = phx.optim.prepare_conic_sensitivity(
        prepared,
        execution,
        linear=linear,
        regularity_tolerance=regularity_tolerance,
    )
    return prepared, execution, sensitivity


def _tangent(
    program,
    *,
    quadratic=None,
    linear=None,
    matrix=None,
    rhs=None,
    lower=None,
    upper=None,
):
    zero = phx.optim.ConicProgramData.zeros_like(program)
    return phx.optim.ConicProgramData(
        zero.quadratic if quadratic is None else quadratic,
        zero.linear if linear is None else linear,
        zero.constraint_matrix if matrix is None else matrix,
        zero.constraint_rhs if rhs is None else rhs,
        zero.lower_bounds if lower is None else lower,
        zero.upper_bounds if upper is None else upper,
    )


def _data_pairing(left, right):
    value = jnp.vdot(left.linear, right.linear)
    value += jnp.vdot(left.constraint_matrix, right.constraint_matrix)
    value += jnp.vdot(left.constraint_rhs, right.constraint_rhs)
    value += jnp.vdot(left.lower_bounds, right.lower_bounds)
    value += jnp.vdot(left.upper_bounds, right.upper_bounds)
    if left.quadratic is not None and right.quadratic is not None:
        value += jnp.vdot(left.quadratic, right.quadratic)
    return value


def test_cone_dual_projection_smoothness_margins_locate_kinks():
    zero = phx.optim.ZeroCone(2)
    nonnegative = phx.optim.NonnegativeCone(3)
    soc = phx.optim.SecondOrderCone(3)
    rotated = phx.optim.RotatedSecondOrderCone(3)
    product = phx.optim.ProductCone((nonnegative, soc))

    assert jnp.isinf(zero.dual_projection_smoothness_margin(jnp.ones(2)))
    np.testing.assert_allclose(
        nonnegative.dual_projection_smoothness_margin(jnp.asarray([2.0, -3.0, 0.5])),
        0.5,
    )
    assert (
        nonnegative.dual_projection_smoothness_margin(jnp.asarray([1.0, 0.0, -1.0])) == 0
    )
    assert soc.dual_projection_smoothness_margin(jnp.asarray([2.0, 0.5, 0.0])) > 0
    assert soc.dual_projection_smoothness_margin(jnp.asarray([1.0, 1.0, 0.0])) == 0
    assert soc.dual_projection_smoothness_margin(jnp.zeros(3)) == 0
    rotated_point = jnp.asarray([2.0, 1.0, 0.25])
    np.testing.assert_allclose(
        rotated.dual_projection_smoothness_margin(rotated_point),
        rotated._soc.dual_projection_smoothness_margin(rotated._to_soc(rotated_point)),
    )
    blocks = jnp.asarray([2.0, -3.0, 0.5, 2.0, 0.5, 0.0])
    np.testing.assert_allclose(product.dual_projection_smoothness_margin(blocks), 0.5)
    empty = phx.optim.ProductCone(())
    assert jnp.isinf(empty.dual_projection_smoothness_margin(jnp.empty((0,))))


def test_active_orthant_primal_jvp_is_linear_at_small_scale():
    problem = phx.optim.ConicProgram(
        jnp.ones((1, 1)),
        jnp.asarray([-2.0]),
        jnp.ones((1, 1)),
        jnp.asarray([1.0]),
        phx.optim.NonnegativeCone(1),
        problem_id="active-orthant-sensitivity",
    )
    _, execution, sensitivity = _prepare(
        problem,
        solution=(
            jnp.ones(1),
            jnp.zeros(1),
            jnp.ones(1),
            jnp.zeros(1),
            jnp.zeros(1),
        ),
    )
    assert execution.result.successful
    tangent = _tangent(problem, rhs=jnp.ones(1))

    derivative = phx.optim.conic_primal_jvp(sensitivity, tangent)
    np.testing.assert_allclose(derivative.value, jnp.ones(1), atol=2e-6)
    assert derivative.regular
    assert derivative.root_residual_norm < 1e-7

    alpha = jnp.asarray(1e-9, dtype=problem.linear.dtype)
    scaled = _tangent(problem, rhs=alpha * jnp.ones(1))
    scaled_derivative = phx.optim.conic_primal_jvp(sensitivity, scaled)
    np.testing.assert_allclose(
        scaled_derivative.value,
        alpha * derivative.value,
        rtol=2e-4,
        atol=1e-12,
    )


def test_soc_projection_qcp_matches_analytic_jvp_and_adjoint():
    cone = phx.optim.SecondOrderCone(2)
    center = jnp.asarray([0.0, 2.0])
    direction = jnp.asarray([0.3, -0.2])
    problem = phx.optim.ConicProgram(
        jnp.eye(2),
        -center,
        -jnp.eye(2),
        jnp.zeros(2),
        cone,
        problem_id="soc-projection-sensitivity",
    )
    primal = cone.project(center)
    _, execution, sensitivity = _prepare(
        problem,
        solution=(
            primal,
            primal,
            primal - center,
            jnp.zeros(2),
            jnp.zeros(2),
        ),
    )
    np.testing.assert_allclose(execution.result.primal, cone.project(center), atol=2e-6)
    tangent = _tangent(problem, linear=-direction)

    derivative = phx.optim.conic_primal_jvp(sensitivity, tangent)
    expected = jax.jvp(cone.project, (center,), (direction,))[1]
    np.testing.assert_allclose(derivative.value, expected, atol=2e-6, rtol=2e-6)
    assert derivative.regular

    cotangent = jnp.asarray([0.7, -0.4])
    adjoint = phx.optim.conic_primal_vjp(sensitivity, cotangent)
    np.testing.assert_allclose(
        jnp.vdot(cotangent, derivative.value),
        _data_pairing(adjoint.value, tangent),
        atol=2e-7,
        rtol=2e-7,
    )
    assert adjoint.regular


def test_batched_soc_sensitivity_maps_common_residual_over_array_leaves():
    cone = phx.optim.SecondOrderCone(2)
    centers = jnp.asarray([[0.0, 2.0], [0.0, 3.0]])
    directions = jnp.asarray([[0.3, -0.2], [-0.4, 0.1]])
    problem = phx.optim.ConicProgram(
        jnp.eye(2),
        -centers,
        -jnp.eye(2),
        jnp.zeros(2),
        cone,
        problem_id="batched-soc-projection-sensitivity",
    )
    primal = jax.vmap(cone.project)(centers)
    _, _, sensitivity = _prepare(
        problem,
        solution=(
            primal,
            primal,
            primal - centers,
            jnp.zeros_like(centers),
            jnp.zeros_like(centers),
        ),
    )
    tangent = _tangent(problem, linear=-directions)

    derivative = phx.optim.conic_primal_jvp(sensitivity, tangent)
    expected = jax.vmap(
        lambda point, direction: jax.jvp(cone.project, (point,), (direction,))[1]
    )(centers, directions)
    np.testing.assert_allclose(derivative.value, expected, atol=3e-6, rtol=3e-6)
    np.testing.assert_array_equal(derivative.regular, jnp.asarray([True, True]))
    assert derivative.value.shape == centers.shape
    assert derivative.linear_status.shape == (2,)
    cotangent = jnp.asarray([[0.7, -0.4], [-0.2, 0.5]])
    adjoint = phx.optim.conic_primal_vjp(sensitivity, cotangent)
    np.testing.assert_allclose(
        jnp.vdot(cotangent, derivative.value),
        _data_pairing(adjoint.value, tangent),
        atol=3e-7,
        rtol=3e-7,
    )
    np.testing.assert_array_equal(adjoint.regular, jnp.asarray([True, True]))


def test_fixed_bound_sensitivity_uses_diagonal_tangent_and_symmetric_pullback():
    problem = phx.optim.ConicProgram(
        jnp.ones((1, 1)),
        jnp.zeros(1),
        jnp.empty((0, 1)),
        jnp.empty((0,)),
        phx.optim.ProductCone(()),
        bounds=phx.optim.Bounds(2.0, 2.0),
        problem_id="fixed-bound-sensitivity",
    )
    _, execution, sensitivity = _prepare(
        problem,
        solution=(
            jnp.asarray([2.0]),
            jnp.empty((0,)),
            jnp.empty((0,)),
            jnp.asarray([2.0]),
            jnp.zeros(1),
        ),
    )
    np.testing.assert_allclose(execution.result.primal, jnp.asarray([2.0]), atol=2e-6)
    tangent = _tangent(problem, lower=jnp.ones(1), upper=jnp.ones(1))

    derivative = phx.optim.conic_primal_jvp(sensitivity, tangent)
    np.testing.assert_allclose(derivative.value, jnp.ones(1), atol=2e-6)
    adjoint = phx.optim.conic_primal_vjp(sensitivity, jnp.ones(1))
    np.testing.assert_allclose(adjoint.value.lower_bounds, jnp.asarray([0.5]), atol=2e-6)
    np.testing.assert_allclose(adjoint.value.upper_bounds, jnp.asarray([0.5]), atol=2e-6)
    np.testing.assert_allclose(_data_pairing(adjoint.value, tangent), 1.0, atol=2e-6)

    invalid = _tangent(problem, lower=jnp.ones(1), upper=jnp.zeros(1))
    with pytest.raises((ValueError, RuntimeError), match="preserve fixed"):
        jax.block_until_ready(phx.optim.conic_primal_jvp(sensitivity, invalid).value)


def test_weak_complementarity_returns_nonregular_nan_sensitivity():
    problem = phx.optim.ConicProgram(
        jnp.ones((1, 1)),
        jnp.zeros(1),
        jnp.ones((1, 1)),
        jnp.zeros(1),
        phx.optim.NonnegativeCone(1),
        problem_id="weak-complementarity-sensitivity",
    )
    _, execution, sensitivity = _prepare(
        problem,
        solution=(
            jnp.zeros(1),
            jnp.zeros(1),
            jnp.zeros(1),
            jnp.zeros(1),
            jnp.zeros(1),
        ),
        regularity_tolerance=1e-5,
    )
    assert execution.result.successful
    derivative = phx.optim.conic_primal_jvp(
        sensitivity,
        _tangent(problem, linear=jnp.ones(1)),
    )

    assert not derivative.projection_regular
    assert not derivative.regular
    assert jnp.isnan(derivative.value[0])


def test_regularized_linear_conic_objective_differentiates_executed_map():
    problem = phx.optim.ConicProgram(
        None,
        jnp.asarray([-1.0]),
        jnp.empty((0, 1)),
        jnp.empty((0,)),
        phx.optim.ProductCone(()),
        problem_id="regularized-linear-conic-sensitivity",
    )
    _, execution, sensitivity = _prepare(
        problem,
        solution=(
            jnp.asarray([0.5]),
            jnp.empty((0,)),
            jnp.empty((0,)),
            jnp.zeros(1),
            jnp.zeros(1),
        ),
        regularization=2.0,
    )
    np.testing.assert_allclose(execution.result.primal, jnp.asarray([0.5]), atol=2e-7)

    derivative = phx.optim.conic_primal_jvp(
        sensitivity,
        _tangent(problem, linear=jnp.ones(1)),
    )
    np.testing.assert_allclose(derivative.value, jnp.asarray([-0.5]), atol=2e-7)
    adjoint = phx.optim.conic_primal_vjp(sensitivity, jnp.ones(1))
    assert adjoint.value.quadratic is None
    np.testing.assert_allclose(adjoint.value.linear, jnp.asarray([-0.5]), atol=2e-7)


def test_prepared_sensitivity_rejects_stale_numeric_execution():
    def problem(rhs):
        return phx.optim.ConicProgram(
            jnp.ones((1, 1)),
            jnp.asarray([-2.0]),
            jnp.ones((1, 1)),
            jnp.asarray([rhs]),
            phx.optim.NonnegativeCone(1),
            problem_id="stale-conic-sensitivity",
        )

    prepared, execution, _ = _prepare(
        problem(1.0),
        solution=(
            jnp.ones(1),
            jnp.zeros(1),
            jnp.ones(1),
            jnp.zeros(1),
            jnp.zeros(1),
        ),
    )
    refreshed = phx.optim.refresh_convex_program(prepared, problem(1.5))

    with pytest.raises(ValueError, match="numeric version"):
        phx.optim.prepare_conic_sensitivity(refreshed, execution)


def test_prepared_sensitivity_rejects_independent_same_version_binding():
    solution = (
        jnp.ones(1),
        jnp.zeros(1),
        jnp.ones(1),
        jnp.zeros(1),
        jnp.zeros(1),
    )
    problem = phx.optim.ConicProgram(
        jnp.ones((1, 1)),
        jnp.asarray([-2.0]),
        jnp.ones((1, 1)),
        jnp.ones(1),
        phx.optim.NonnegativeCone(1),
        problem_id="independent-conic-binding",
    )
    first, execution, _ = _prepare(problem, solution=solution)
    second, _, _ = _prepare(problem, solution=solution)
    assert first.numeric_binding_id != second.numeric_binding_id

    with pytest.raises(ValueError, match="numeric binding"):
        phx.optim.prepare_conic_sensitivity(second, execution)


def test_projection_regularity_uses_the_differentiated_residual_point():
    problem = phx.optim.ConicProgram(
        jnp.ones((1, 1)),
        jnp.asarray([-2.0]),
        jnp.ones((1, 1)),
        jnp.ones(1),
        phx.optim.NonnegativeCone(1),
        problem_id="projection-point-consistency",
    )
    prepared, execution, _ = _prepare(
        problem,
        solution=(
            jnp.ones(1),
            jnp.zeros(1),
            jnp.ones(1),
            jnp.zeros(1),
            jnp.zeros(1),
        ),
    )
    inconsistent = eqx.tree_at(
        lambda value: value.result.cone_slack,
        execution,
        jnp.ones(1),
    )

    sensitivity = phx.optim.prepare_conic_sensitivity(prepared, inconsistent)
    np.testing.assert_allclose(sensitivity.projection_margin, jnp.ones(1))
    np.testing.assert_array_equal(sensitivity.projection_regular, jnp.ones(1, dtype=bool))


def test_damped_derivative_svd_is_rejected():
    problem = phx.optim.ConicProgram(
        jnp.ones((1, 1)),
        jnp.asarray([-2.0]),
        jnp.ones((1, 1)),
        jnp.ones(1),
        phx.optim.NonnegativeCone(1),
        problem_id="damped-derivative-rejection",
    )
    damped = phx.linalg.LinearSolvePolicy(phx.linalg.DenseSVD(damping=1e-3))

    with pytest.raises(ValueError, match="zero derivative-solver damping"):
        _prepare(
            problem,
            solution=(
                jnp.ones(1),
                jnp.zeros(1),
                jnp.ones(1),
                jnp.zeros(1),
                jnp.zeros(1),
            ),
            linear=damped,
        )
