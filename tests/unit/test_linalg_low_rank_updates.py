#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg


def _positive_definite_properties():
    return la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
            "positive_semidefinite": "construction",
        },
    )


def _low_rank_data():
    base = jnp.diag(jnp.asarray([3.0, 4.0, 5.0]))
    left = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    right = jnp.asarray([[0.5, 0.0], [0.0, 0.25], [0.1, 0.2]])
    core = jnp.asarray([[2.0, 0.1], [0.1, 1.0]])
    return base, left, right, core


def _status_policy(*, condition_limit=1e12, resources=None):
    return la.LowRankSolvePolicy(
        la.LinearSolvePolicy(la.DenseLU()),
        condition_limit=condition_limit,
        base_nonsingularity="asserted",
        failure=la.FailurePolicy("status"),
        resources=resources,
    )


def test_base_plus_low_rank_operator_actions_match_its_dense_matrix():
    base, left, right, core = _low_rank_data()
    operator = la.BasePlusLowRankLinearOperator(
        la.DenseLinearOperator(base),
        left,
        right,
        core,
    )
    matrix = base + left @ core @ right.T
    vector = jnp.asarray([1.0, -2.0, 0.5])

    assert operator.rank == 2
    assert jnp.allclose(operator.mv(vector), matrix @ vector)
    assert jnp.allclose(operator.transpose_mv(vector), matrix.T @ vector)
    assert jnp.allclose(operator.adjoint_mv(vector), matrix.T @ vector)
    assert jnp.allclose(
        la.materialize(operator, la.MaterializationPolicy(max_entries=9)),
        matrix,
    )
    cost = la.estimate_operator_action_cost(operator)
    assert cost.exact
    assert cost.operation_class == "base-plus-low-rank-action"


def test_low_rank_solve_is_jittable_refreshable_and_supports_rhs_layouts():
    base, left, right, core = _low_rank_data()
    operator = la.BasePlusLowRankLinearOperator(
        la.DenseLinearOperator(base, operator_id="woodbury-base"),
        left,
        right,
        core,
        operator_id="woodbury-system",
    )
    policy = _status_policy()
    plan = la.plan_low_rank_solve(operator, policy)
    prepared = la.prepare_low_rank_solve(operator, plan)
    rhs = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layout = la.RHSLayout((2,))
    expected_matrix = base + left @ core @ right.T
    expected = jnp.linalg.solve(expected_matrix, rhs)

    result = la.solve_low_rank(prepared, rhs, rhs_layout=layout)
    compiled = jax.jit(la.solve_low_rank)(prepared, rhs, rhs_layout=layout)

    assert jnp.all(result.successful)
    assert jnp.allclose(result.value, expected, rtol=1e-11, atol=1e-12)
    assert jnp.allclose(compiled.value, expected, rtol=1e-11, atol=1e-12)
    assert jnp.all(result.diagnostics.residual_norm < 1e-12)
    assert result.diagnostics.rank == 2
    assert result.provenance.plan_id == plan.plan_id
    assert result.provenance.base_nonsingularity == "asserted"

    updated_base = jnp.diag(jnp.asarray([4.0, 5.0, 6.0]))
    updated_core = core + jnp.asarray([[0.2, -0.05], [0.0, 0.1]])
    updated = la.BasePlusLowRankLinearOperator(
        la.DenseLinearOperator(updated_base, operator_id="woodbury-base"),
        left,
        right,
        updated_core,
        operator_id="woodbury-system",
    )
    refreshed = la.refresh_low_rank_solve(prepared, updated)
    refreshed_result = la.solve_low_rank(refreshed, rhs, rhs_layout=layout)
    refreshed_expected = jnp.linalg.solve(
        updated_base + left @ updated_core @ right.T,
        rhs,
    )

    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert refreshed.numeric_version == 1
    assert refreshed.base_prepared.numeric_version == 1
    assert jnp.allclose(
        refreshed_result.value,
        refreshed_expected,
        rtol=1e-11,
        atol=1e-12,
    )


def test_low_rank_solve_uses_an_arbitrary_matrix_free_base():
    base, left, right, core = _low_rank_data()
    space = la.ArraySpace((3,), dtype=jnp.float64)
    base_operator = la.FunctionLinearOperator(
        lambda vector: base @ vector,
        source=space,
        target=space,
        transpose_action=lambda vector: base.T @ vector,
        properties=_positive_definite_properties(),
        operator_id="matrix-free-low-rank-base",
    )
    operator = la.BasePlusLowRankLinearOperator(
        base_operator,
        left,
        right,
        core,
        operator_id="matrix-free-low-rank-system",
    )
    policy = la.LowRankSolvePolicy(
        la.LinearSolvePolicy(
            la.PCG(),
            tolerance=la.TolerancePolicy(relative=1e-12, absolute=1e-12, max_steps=20),
        ),
        failure=la.FailurePolicy("status"),
    )
    rhs = jnp.asarray([1.0, 3.0, 5.0])

    prepared = la.prepare_low_rank_solve(operator, policy)
    result = la.solve_low_rank(prepared, rhs)
    expected = jnp.linalg.solve(base + left @ core @ right.T, rhs)

    assert result.successful
    assert prepared.base_prepared.plan.backend == "native-krylov"
    assert jnp.allclose(result.value, expected, rtol=1e-10, atol=1e-11)


def test_low_rank_one_shot_derivative_matches_the_dense_solve_derivative():
    base, left, right, core = _low_rank_data()
    rhs = jnp.asarray([1.0, 3.0, 5.0])
    policy = la.LowRankSolvePolicy(
        la.LinearSolvePolicy(
            la.DenseLU(),
            differentiation=la.DifferentiationPolicy("mathematical"),
        ),
        base_nonsingularity="asserted",
        failure=la.FailurePolicy("status"),
    )

    def specialized(candidate_core):
        operator = la.BasePlusLowRankLinearOperator(
            la.DenseLinearOperator(base),
            left,
            right,
            candidate_core,
        )
        return jnp.sum(la.solve_low_rank(operator, rhs, policy).value)

    def dense(candidate_core):
        matrix = base + left @ candidate_core @ right.T
        return jnp.sum(jnp.linalg.solve(matrix, rhs))

    actual = jax.jit(jax.grad(specialized))(core)
    expected = jax.grad(dense)(core)

    assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-11)


def test_low_rank_planning_requires_evidence_and_enforces_resource_bounds():
    base, left, right, core = _low_rank_data()
    operator = la.BasePlusLowRankLinearOperator(
        la.DenseLinearOperator(base),
        left,
        right,
        core,
    )

    with pytest.raises(ValueError, match="lacks a full-rank"):
        la.plan_low_rank_solve(operator)
    with pytest.raises(ValueError, match="rank exceeds max_rank"):
        la.plan_low_rank_solve(
            operator,
            _status_policy(resources=la.LowRankResourcePolicy(max_rank=1)),
        )


def test_low_rank_status_exposes_an_ill_conditioned_correction():
    base, left, right, core = _low_rank_data()
    operator = la.BasePlusLowRankLinearOperator(
        la.DenseLinearOperator(base),
        left,
        right,
        core,
    )
    prepared = la.prepare_low_rank_solve(
        operator,
        _status_policy(condition_limit=1.01),
    )
    result = la.solve_low_rank(prepared, jnp.asarray([1.0, 3.0, 5.0]))

    assert prepared.correction_condition > 1.01
    assert result.status == int(la.LowRankSolveStatus.CORRECTION_ILL_CONDITIONED)
    assert jnp.all(jnp.isfinite(result.value))
