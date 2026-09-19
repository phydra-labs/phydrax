#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.linalg._low_rank_updates import (
    accept_low_rank_update,
    column_low_rank_update,
    dense_low_rank_update,
    LowRankDeterminantStatus,
    prepare_low_rank_sequence,
    propose_low_rank_update,
    rebase_low_rank_sequence,
    refresh_low_rank_sequence,
    row_low_rank_update,
    skew_row_column_low_rank_update,
    solve_low_rank_sequence,
)


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


@pytest.mark.parametrize(
    ("complex_data", "rank"),
    ((False, 1), (False, 2), (True, 1), (True, 2)),
)
def test_low_rank_determinant_ratio_and_sequence_solve_match_dense_recomputation(
    complex_data,
    rank,
):
    base = jnp.asarray(
        [[3.0, 0.2, -0.1], [0.1, 2.5, 0.3], [0.0, -0.2, 4.0]],
        dtype=jnp.complex128 if complex_data else jnp.float64,
    )
    left = jnp.asarray(
        [[0.3, -0.2], [0.1, 0.4], [-0.5, 0.2]],
        dtype=base.dtype,
    )
    right = jnp.asarray(
        [[0.2, 0.1], [-0.3, 0.2], [0.4, -0.1]],
        dtype=base.dtype,
    )
    if complex_data:
        base = base + 1j * jnp.asarray(
            [[0.1, -0.2, 0.0], [0.3, 0.0, 0.1], [-0.1, 0.2, -0.1]]
        )
        left = left + 1j * jnp.asarray([[0.1, 0.0], [-0.2, 0.3], [0.1, -0.1]])
        right = right + 1j * jnp.asarray([[-0.1, 0.2], [0.1, 0.0], [0.2, -0.3]])
    left, right = left[:, :rank], right[:, :rank]
    base_factorization = la.factorize(
        la.DenseLinearOperator(base, operator_id="determinant-sequence-base"),
        la.FactorizationPolicy("lu"),
    )
    sequence = la.prepare_factorized_low_rank_sequence(
        base_factorization,
        4,
        _status_policy(),
    )
    proposal = propose_low_rank_update(
        sequence,
        dense_low_rank_update(left, right),
    )
    updated = base + left @ right.T
    expected_ratio = jnp.linalg.det(updated) / jnp.linalg.det(base)

    assert proposal.successful
    assert jnp.allclose(proposal.value, expected_ratio, rtol=1e-11, atol=1e-12)
    assert jnp.allclose(
        proposal.sign * jnp.exp(proposal.log_abs),
        expected_ratio,
        rtol=1e-11,
        atol=1e-12,
    )
    assert proposal.provenance.route == "dense"

    accepted = accept_low_rank_update(sequence, proposal)
    expected_sign, expected_log_abs = jnp.linalg.slogdet(updated)
    assert sequence.base_determinant_available
    assert jnp.allclose(accepted.absolute_determinant_sign, expected_sign)
    assert jnp.allclose(accepted.absolute_log_abs_determinant, expected_log_abs)
    rhs = jnp.asarray([0.5, -1.0, 2.0], dtype=base.dtype)
    solved = solve_low_rank_sequence(accepted, rhs)
    method_solved = accepted.solve(rhs)
    expected_solution = jnp.linalg.solve(updated, rhs)
    assert jnp.allclose(solved.value, expected_solution, rtol=1e-11, atol=1e-12)
    assert jnp.allclose(method_solved.value, expected_solution, rtol=1e-11, atol=1e-12)
    assert accepted.prepared.base_prepared.numeric_version == (
        sequence.prepared.base_prepared.numeric_version
    )


@pytest.mark.parametrize("route", ("row", "column"))
def test_indexed_low_rank_updates_match_dense_without_retained_one_hot(route):
    base = jnp.diag(jnp.asarray([2.0, 3.0, 4.0, 5.0]))
    indices = jnp.asarray([1, 3])
    sequence = prepare_low_rank_sequence(
        la.DenseLinearOperator(base),
        2,
        _status_policy(),
    )
    if route == "row":
        values = jnp.asarray([[0.2, -0.1, 0.3, 0.0], [0.1, 0.2, 0.0, -0.2]])
        update = row_low_rank_update(indices, values)
        expected = base.at[indices, :].add(values)
    else:
        values = jnp.asarray([[0.2, 0.1], [-0.1, 0.2], [0.3, 0.0], [0.0, -0.2]])
        update = column_low_rank_update(indices, values)
        expected = base.at[:, indices].add(values)

    proposal = propose_low_rank_update(sequence, update)
    accepted = accept_low_rank_update(sequence, proposal)
    rhs = jnp.asarray([1.0, -0.5, 0.2, 2.0])

    assert proposal.successful
    assert proposal.provenance.route == f"{route}-indexed"
    assert jnp.allclose(
        proposal.value,
        jnp.linalg.det(expected) / jnp.linalg.det(base),
        rtol=1e-11,
        atol=1e-12,
    )
    assert jnp.allclose(
        solve_low_rank_sequence(accepted, rhs).value,
        jnp.linalg.solve(expected, rhs),
        rtol=1e-11,
        atol=1e-12,
    )


def test_skew_row_column_update_is_one_joint_rank_two_ratio():
    base = jnp.asarray(
        [
            [0.0, 1.2, -0.7, 0.3],
            [-1.2, 0.0, 0.4, -0.2],
            [0.7, -0.4, 0.0, 1.1],
            [-0.3, 0.2, -1.1, 0.0],
        ]
    )
    row_delta = jnp.asarray([0.25, -0.1, 0.0, 0.2])
    index = 2
    update = skew_row_column_low_rank_update(index, row_delta)
    sequence = prepare_low_rank_sequence(
        la.DenseLinearOperator(base),
        2,
        _status_policy(),
    )
    proposal = propose_low_rank_update(sequence, update)
    expected = base.at[index, :].add(row_delta).at[:, index].add(-row_delta)

    assert update.route == "skew-row-column-indexed"
    assert proposal.successful
    assert jnp.allclose(
        proposal.value,
        jnp.linalg.det(expected) / jnp.linalg.det(base),
        rtol=1e-11,
        atol=1e-12,
    )


def test_skew_row_column_pfaffian_ratio_reuses_native_solve_state():
    base = jnp.asarray(
        [
            [0.0, 1.2, -0.7, 0.3],
            [-1.2, 0.0, 0.4, -0.2],
            [0.7, -0.4, 0.0, 1.1],
            [-0.3, 0.2, -1.1, 0.0],
        ]
    )
    index = 2
    row_delta = jnp.asarray([0.25, -0.1, 0.0, 0.2])
    sequence = la.prepare_low_rank_sequence(
        la.DenseLinearOperator(base),
        2,
        _status_policy(),
    )
    update = la.skew_row_column_low_rank_update(index, row_delta)
    proposal = la.propose_pfaffian_update(sequence, update)
    compiled = jax.jit(la.propose_pfaffian_update)(sequence, update)
    updated = base.at[index, :].add(row_delta).at[:, index].add(-row_delta)
    expected = la.evaluate_pfaffian(updated).value / la.evaluate_pfaffian(base).value

    assert proposal.successful
    assert compiled.successful
    assert jnp.allclose(proposal.value, expected, rtol=1e-11, atol=1e-12)
    assert jnp.allclose(compiled.value, expected, rtol=1e-11, atol=1e-12)
    assert proposal.determinant_identity_residual < 1e-11
    accepted = la.accept_low_rank_update(sequence, proposal.determinant)
    rhs = jnp.asarray([0.3, -0.4, 0.7, 0.2])
    assert jnp.allclose(
        la.solve_low_rank_sequence(accepted, rhs).value,
        jnp.linalg.solve(updated, rhs),
        rtol=1e-11,
        atol=1e-12,
    )


def test_sequence_selection_composes_signed_logs_and_reuses_candidate_factorization():
    base = jnp.diag(jnp.asarray([2.0, 3.0, 4.0]))
    sequence = prepare_low_rank_sequence(
        la.DenseLinearOperator(base),
        2,
        _status_policy(),
    )
    first = dense_low_rank_update(
        jnp.asarray([[0.2], [0.1], [-0.1]]),
        jnp.asarray([[0.3], [-0.2], [0.1]]),
    )
    first_proposal = propose_low_rank_update(sequence, first)
    rejected = accept_low_rank_update(sequence, first_proposal, accepted=False)
    accepted = accept_low_rank_update(sequence, first_proposal, accepted=True)
    compiled_accepted = jax.jit(accept_low_rank_update)(
        sequence,
        first_proposal,
        accepted=jnp.asarray(True),
    )

    assert rejected.active_rank == 0
    assert rejected.accepted_count == 0
    assert accepted.active_rank == 1
    assert accepted.accepted_count == 1
    assert compiled_accepted.active_rank == 1
    assert compiled_accepted.accepted_count == 1

    second = column_low_rank_update(
        jnp.asarray([1]),
        jnp.asarray([[0.1], [0.2], [-0.1]]),
    )
    second_proposal = propose_low_rank_update(accepted, second)
    composed = accept_low_rank_update(accepted, second_proposal)
    first_matrix = base + first.left_factor @ first.right_factor.T
    second_matrix = first_matrix.at[:, 1].add(second.left_factor[:, 0])
    expected_ratio = jnp.linalg.det(second_matrix) / jnp.linalg.det(base)

    assert jnp.allclose(
        composed.determinant_sign * jnp.exp(composed.log_abs_determinant_ratio),
        expected_ratio,
        rtol=1e-11,
        atol=1e-12,
    )

    overflow = propose_low_rank_update(composed, first)
    assert overflow.status == int(LowRankDeterminantStatus.CAPACITY_EXCEEDED)
    assert overflow.requires_rebase
    unchanged = accept_low_rank_update(composed, overflow)
    assert unchanged.active_rank == composed.active_rank
    assert unchanged.accepted_count == composed.accepted_count


def test_low_rank_proposal_cannot_cross_numeric_base_lineage():
    first_base = jnp.diag(jnp.asarray([2.0, 3.0]))
    second_base = jnp.diag(jnp.asarray([4.0, 5.0]))
    first_sequence = prepare_low_rank_sequence(
        la.DenseLinearOperator(first_base, operator_id="shared-base-identity"),
        1,
        _status_policy(),
    )
    second_sequence = prepare_low_rank_sequence(
        la.DenseLinearOperator(second_base, operator_id="shared-base-identity"),
        1,
        _status_policy(),
    )
    update = dense_low_rank_update(
        jnp.asarray([[0.2], [-0.1]]),
        jnp.asarray([[0.3], [0.4]]),
    )
    proposal = propose_low_rank_update(first_sequence, update)

    unchanged = accept_low_rank_update(second_sequence, proposal)

    assert unchanged.active_rank == 0
    assert unchanged.accepted_count == 0
    rhs = jnp.asarray([1.0, -0.5])
    assert jnp.allclose(
        solve_low_rank_sequence(unchanged, rhs).value,
        jnp.linalg.solve(second_base, rhs),
    )


def test_ill_conditioned_proposal_is_reported_and_not_accepted():
    base = jnp.eye(3)
    sequence = prepare_low_rank_sequence(
        la.DenseLinearOperator(base),
        1,
        _status_policy(condition_limit=1e6),
    )
    left = jnp.asarray([[1.0], [0.0], [0.0]])
    right = -left
    proposal = propose_low_rank_update(sequence, dense_low_rank_update(left, right))
    unchanged = accept_low_rank_update(sequence, proposal)

    assert proposal.status == int(
        LowRankDeterminantStatus.PROPOSAL_CORRECTION_ILL_CONDITIONED
    )
    assert proposal.requires_rebase
    assert not proposal.valid
    assert unchanged.active_rank == 0


def test_low_rank_proposals_support_jit_vmap_refresh_and_explicit_rebase():
    base = jnp.diag(jnp.asarray([2.0, 3.0, 4.0]))
    base_operator = la.DenseLinearOperator(base, operator_id="refreshable-sequence-base")
    sequence = prepare_low_rank_sequence(base_operator, 2, _status_policy())
    right = jnp.asarray([[0.2], [-0.1], [0.3]])
    left_batch = jnp.asarray([[[0.1], [0.0], [-0.2]], [[-0.2], [0.3], [0.1]]])

    def ratio(left):
        return propose_low_rank_update(
            sequence,
            dense_low_rank_update(left, right),
        ).value

    vmapped = jax.jit(jax.vmap(ratio))(left_batch)
    expected = jax.vmap(
        lambda left: jnp.linalg.det(base + left @ right.T) / jnp.linalg.det(base)
    )(left_batch)
    assert jnp.allclose(vmapped, expected, rtol=1e-11, atol=1e-12)

    update = dense_low_rank_update(left_batch[0], right)
    accepted = accept_low_rank_update(sequence, propose_low_rank_update(sequence, update))
    changed_base = jnp.diag(jnp.asarray([2.5, 3.5, 4.5]))
    refreshed = refresh_low_rank_sequence(
        accepted,
        la.DenseLinearOperator(
            changed_base,
            operator_id="refreshable-sequence-base",
        ),
    )
    current = changed_base + update.left_factor @ update.right_factor.T
    rhs = jnp.asarray([0.2, -0.5, 1.0])
    assert refreshed.active_rank == 1
    assert jnp.allclose(
        refreshed.solve(rhs).value,
        jnp.linalg.solve(current, rhs),
        rtol=1e-11,
        atol=1e-12,
    )

    rebased = rebase_low_rank_sequence(
        refreshed,
        la.DenseLinearOperator(current, operator_id="folded-sequence-base"),
    )
    assert rebased.active_rank == 0
    assert rebased.accepted_count == 0
    assert rebased.determinant_sign == 1
    assert rebased.log_abs_determinant_ratio == 0
