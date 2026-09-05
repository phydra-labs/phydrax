#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg
svd = la.svd


def test_dense_svd_is_jittable_refreshable_and_reports_rank():
    matrix = jnp.asarray([[3.0, 1.0], [0.0, 2.0], [1.0, 0.0]])
    operator = la.DenseLinearOperator(matrix, operator_id="refreshable-svd")
    problem = svd.SVDProblem(operator, problem_id="svd-problem")
    policy = svd.SVDSolvePolicy(count=2)
    plan = svd.plan_svd(problem, policy)
    prepared = svd.prepare_svd(problem, plan)
    result = svd.svd(prepared)
    compiled = jax.jit(svd.svd)(prepared)

    assert bool(result.successful)
    assert jnp.allclose(
        result.singular_values,
        jnp.linalg.svd(matrix, compute_uv=False),
    )
    assert jnp.allclose(compiled.singular_values, result.singular_values)
    assert int(result.numerical_rank) == 2
    assert jnp.max(result.diagnostics.left_residual_norms) < 1e-12
    assert jnp.max(result.diagnostics.right_residual_norms) < 1e-12

    changed_operator = la.DenseLinearOperator(
        2.0 * matrix,
        operator_id=operator.operator_id,
    )
    refreshed = svd.refresh_svd(
        prepared,
        svd.SVDProblem(changed_operator, problem_id=problem.problem_id),
    )
    refreshed_result = svd.svd(refreshed)
    assert int(refreshed.numeric_version) == 1
    assert jnp.allclose(
        refreshed_result.singular_values,
        2.0 * result.singular_values,
    )


@pytest.mark.parametrize("entry, expected_rank", [(1.0, 1), (0.0, 0)])
@pytest.mark.parametrize("which", ["largest", "smallest"])
def test_svd_certifies_rank_deficient_and_zero_operators(entry, expected_rank, which):
    matrix = jnp.full((2, 2), entry)
    result = svd.svd(
        svd.SVDProblem(la.DenseLinearOperator(matrix)),
        policy=svd.SVDSolvePolicy(count=2, which=which),
    )
    expected_values = jnp.asarray([2 * entry, 0.0])
    if which == "smallest":
        expected_values = expected_values[::-1]

    assert bool(result.successful)
    assert int(result.numerical_rank) == expected_rank
    assert jnp.allclose(result.singular_values, expected_values, atol=1e-12)
    assert jnp.allclose(
        (result.left_vectors * result.singular_values) @ result.right_vectors.T,
        matrix,
        atol=1e-12,
    )


def test_svd_refuses_stale_decomposition_state():
    matrix = jnp.asarray([[3.0, 1.0], [0.0, 2.0]])
    operator = la.DenseLinearOperator(matrix, operator_id="stale-svd")
    problem = svd.SVDProblem(operator, problem_id="stale-svd-problem")
    prepared = svd.prepare_svd(problem, svd.SVDSolvePolicy(count=2))
    changed = svd.refresh_svd(
        prepared,
        svd.SVDProblem(
            la.DenseLinearOperator(2 * matrix, operator_id=operator.operator_id),
            problem_id=problem.problem_id,
        ),
    )
    stale = svd.PreparedSVDSolve(problem, prepared.plan, changed.state)
    result = svd.svd(stale)

    assert not bool(result.successful)
    assert result.status == int(svd.SVDSolveStatus.RESIDUAL_TOLERANCE_NOT_MET)


def test_svd_honors_source_and_target_pairings_and_smallest_target():
    source_weights = jnp.asarray([2.0, 5.0])
    target_weights = jnp.asarray([3.0, 4.0, 6.0])
    source = la.ArraySpace(
        (2,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(source_weights),
    )
    target = la.ArraySpace(
        (3,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(target_weights),
    )
    matrix = jnp.asarray([[2.0, 1.0], [-1.0, 3.0], [0.5, 2.0]])
    operator = la.DenseLinearOperator(
        matrix,
        source=source,
        target=target,
    )
    result = svd.svd(
        svd.SVDProblem(operator),
        policy=svd.SVDSolvePolicy(count=2, which="smallest"),
    )
    transformed = (
        jnp.sqrt(target_weights)[:, None] * matrix / jnp.sqrt(source_weights)[None, :]
    )
    expected = jnp.linalg.svd(transformed, compute_uv=False)[::-1]
    left = jnp.asarray(result.left_vectors)
    right = jnp.asarray(result.right_vectors)

    assert bool(result.successful)
    assert jnp.allclose(result.singular_values, expected)
    assert jnp.allclose(
        left.T @ (target_weights[:, None] * left),
        jnp.eye(2),
        atol=1e-12,
    )
    assert jnp.allclose(
        right.T @ (source_weights[:, None] * right),
        jnp.eye(2),
        atol=1e-12,
    )


def test_singular_value_derivatives_require_nonzero_isolated_values():
    policy = svd.SVDSolvePolicy(
        count=2,
        differentiation="singular-values",
    )

    def nuclear_norm(matrix):
        return jnp.sum(
            svd.svd(
                svd.SVDProblem(la.DenseLinearOperator(matrix)),
                policy=policy,
            ).singular_values
        )

    matrix = jnp.asarray([[3.0, 0.5], [0.0, 2.0], [1.0, 0.0]])
    gradient = jax.jit(jax.grad(nuclear_norm))(matrix)
    left, _, right_adjoint = jnp.linalg.svd(matrix, full_matrices=False)
    assert jnp.allclose(gradient, left @ right_adjoint, atol=1e-10)

    repeated = svd.svd(
        svd.SVDProblem(la.DenseLinearOperator(jnp.eye(2))),
        policy=policy,
    )
    assert repeated.status == int(svd.SVDSolveStatus.DIFFERENTIATION_REJECTED)

    rank_deficient = svd.svd(
        svd.SVDProblem(la.DenseLinearOperator(jnp.asarray([[1.0, 0.0], [0.0, 0.0]]))),
        policy=svd.SVDSolvePolicy(
            count=2,
            rank=la.RankPolicy(require_full_rank=True),
        ),
    )
    assert rank_deficient.status == int(svd.SVDSolveStatus.RANK_DEFICIENT)
    assert int(rank_deficient.numerical_rank) == 1

    with pytest.raises(ValueError, match="materialization limit"):
        svd.plan_svd(
            svd.SVDProblem(la.DenseLinearOperator(matrix)),
            svd.SVDSolvePolicy(
                count=2,
                materialization=la.MaterializationPolicy(
                    max_entries=5,
                    max_bytes=1024,
                ),
            ),
        )


def test_matrix_free_singular_value_gradient_supports_closure_converted_operator():
    policy = svd.SVDSolvePolicy(
        count=2,
        differentiation="singular-values",
    )

    def nuclear_norm(coefficient):
        diagonal = jnp.stack((coefficient, jnp.asarray(3.0)))
        space = la.ArraySpace((2,), dtype=diagonal.dtype)
        operator = la.FunctionLinearOperator(
            lambda vector: diagonal * vector,
            source=space,
            target=space,
        )
        return jnp.sum(
            svd.svd(
                svd.SVDProblem(operator),
                policy=policy,
            ).singular_values
        )

    gradient = jax.jit(jax.grad(nuclear_norm))(jnp.asarray(1.25))

    assert jnp.allclose(gradient, 1.0, atol=1e-8)
