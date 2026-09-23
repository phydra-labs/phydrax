import jax.numpy as jnp
import pytest

import phydrax.tensor_train as tt


def test_two_site_cross_reports_pivots_evaluations_and_held_out_estimator():
    modes = (4, 4, 4)

    def evaluator(indices):
        values = indices.astype(jnp.float32)
        return (values[:, 0] + 1) * (values[:, 1] + 2) * (values[:, 2] + 1)

    plan = tt.TTCrossPlan(
        modes,
        max_rank=2,
        sweeps=4,
        evaluation_budget=56,
        holdout_count=12,
        max_local_unknowns=64,
        regularization=1e-7,
        relative_tolerance=2e-3,
    )
    result = tt.tensor_train_cross(evaluator, plan)

    assert result.evidence.evaluations_used == 56
    assert result.evidence.holdout_count == 12
    assert result.evidence.pivot_indices.shape == (4 * 2 * (len(modes) - 1), len(modes))
    assert result.evidence.estimator_is_guarantee is False
    assert result.evidence.holdout_relative_error_estimator < 2e-3


def test_rank_one_cross_preserves_nonunit_anchor_scale():
    factors = (
        jnp.asarray([2.0, 3.0], dtype=jnp.float32),
        jnp.asarray([5.0, 7.0], dtype=jnp.float32),
        jnp.asarray([11.0, 13.0], dtype=jnp.float32),
    )

    def evaluator(indices):
        return (
            factors[0][indices[:, 0]]
            * factors[1][indices[:, 1]]
            * factors[2][indices[:, 2]]
        )

    plan = tt.TTCrossPlan(
        (2, 2, 2),
        max_rank=1,
        sweeps=1,
        evaluation_budget=8,
        holdout_count=2,
        max_local_unknowns=4,
        regularization=1e-7,
        relative_tolerance=1e-6,
    )
    result = tt.tensor_train_cross(evaluator, plan)
    all_indices = jnp.stack(
        jnp.meshgrid(jnp.arange(2), jnp.arange(2), jnp.arange(2), indexing="ij"),
        axis=-1,
    ).reshape((-1, 3))
    assert result.converged
    assert jnp.allclose(result.tensor.evaluate(all_indices), evaluator(all_indices))


def test_cross_and_completion_reject_nonfinite_observations_and_policy():
    with pytest.raises(ValueError, match="regularization"):
        tt.TTCrossPlan(
            (2, 2),
            max_rank=1,
            sweeps=1,
            evaluation_budget=4,
            holdout_count=1,
            max_local_unknowns=4,
            regularization=jnp.nan,
            relative_tolerance=0.0,
        )

    cross_plan = tt.TTCrossPlan(
        (2, 2),
        max_rank=1,
        sweeps=1,
        evaluation_budget=4,
        holdout_count=1,
        max_local_unknowns=4,
        regularization=1e-6,
        relative_tolerance=0.0,
    )
    with pytest.raises(Exception, match="finite"):
        tt.tensor_train_cross(
            lambda indices: jnp.full((indices.shape[0],), jnp.nan),
            cross_plan,
        )

    completion_plan = tt.TensorCompletionPlan(
        (2, 2),
        max_rank=1,
        sweeps=1,
        relative_tolerance=0.0,
        regularization=1e-6,
        max_local_unknowns=4,
    )
    indices = jnp.asarray([[0, 0], [1, 1]], dtype=jnp.int32)
    with pytest.raises(ValueError, match="finite"):
        tt.weighted_tensor_completion(
            completion_plan,
            indices,
            jnp.asarray([1.0, jnp.nan]),
            jnp.ones((2,)),
            indices,
            jnp.ones((2,)),
            jnp.ones((2,)),
        )


def test_amen_like_poisson_solve_uses_true_global_residual():
    size = 6
    modes = (size,)
    operator = tt.laplacian_operator(
        modes,
        spacing=1.0 / (size + 1),
        boundary=tt.BoundaryPolicy("dirichlet"),
    )
    coordinates = jnp.arange(1, size + 1, dtype=jnp.float32) / (size + 1)
    manufactured = jnp.sin(jnp.pi * coordinates)
    matrix = operator.to_matrix(max_entries=size * size)
    right = matrix @ manufactured
    right_train = tt.TensorTrain((right[None, :, None],))
    initial = tt.TensorTrain((jnp.zeros((1, size, 1), dtype=right.dtype),))
    plan = tt.plan_amen(
        operator,
        max_rank=1,
        enrichment_rank=1,
        sweeps=2,
        relative_tolerance=2e-4,
        local_regularization=1e-8,
        max_dense_entries=size * size,
        max_local_unknowns=size,
    )
    prepared = tt.prepare_tensor_train_solve(plan, operator, right_train, initial)
    refreshed = tt.refresh_tensor_train_solve(
        prepared,
        operator=operator,
        right_hand_side=right_train,
        initial=initial,
    )
    result = tt.solve_tensor_train(refreshed)
    assert refreshed.numeric_version == 1

    recovered = result.solution.to_dense(max_entries=size)
    assert jnp.allclose(recovered, manufactured, rtol=3e-4, atol=3e-4)
    assert result.evidence.true_global_residual_norms.shape == (plan.sweeps + 1,)
    assert (
        result.evidence.true_global_residual_norms[-1]
        < result.evidence.true_global_residual_norms[0]
    )
    assert result.converged


def test_weighted_completion_reports_independent_holdout_error():
    dense = (jnp.arange(4, dtype=jnp.float32) + 1)[:, None] * (
        jnp.arange(4, dtype=jnp.float32) + 2
    )[None, :]
    all_indices = jnp.stack(
        jnp.meshgrid(jnp.arange(4), jnp.arange(4), indexing="ij"), axis=-1
    ).reshape((-1, 2))
    holdout_mask = jnp.asarray(
        [
            False,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            False,
        ]
    )
    observed_indices = all_indices[~holdout_mask]
    holdout_indices = all_indices[holdout_mask]
    observed_values = dense[observed_indices[:, 0], observed_indices[:, 1]]
    holdout_values = dense[holdout_indices[:, 0], holdout_indices[:, 1]]
    plan = tt.TensorCompletionPlan(
        (4, 4),
        max_rank=1,
        sweeps=4,
        relative_tolerance=2e-3,
        regularization=1e-7,
        max_local_unknowns=4,
    )
    result = tt.weighted_tensor_completion(
        plan,
        observed_indices,
        observed_values,
        jnp.ones_like(observed_values),
        holdout_indices,
        holdout_values,
        jnp.ones_like(holdout_values),
    )

    assert result.evidence.observed_count == 12
    assert result.evidence.holdout_count == 4
    assert result.evidence.estimator_is_guarantee is False
    assert result.evidence.holdout_relative_error_estimator < 2e-3


def test_block_smallest_eigen_solver_returns_orthogonal_tt_block():
    diagonal = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    diagonal_train = tt.tt_svd(diagonal, max_ranks=2, relative_tolerance=0.0).tensor
    operator = tt.diagonal_operator(diagonal_train)
    plan = tt.BlockTensorTrainEigenPlan(
        (2, 2),
        block_size=2,
        iterations=18,
        max_rank=2,
        compression_relative_tolerance=0.0,
        residual_relative_tolerance=2e-3,
        orthogonality_tolerance=2e-4,
        inverse_shift=0.1,
        max_dense_entries=16,
    )
    result = tt.smallest_eigenpairs(operator, plan)

    assert jnp.allclose(result.eigenvalues, jnp.asarray([1.0, 2.0]), atol=3e-3)
    assert result.evidence.orthogonality_error < 2e-4
    assert jnp.max(result.evidence.relative_residual_norms) < 2e-3
    assert result.converged
