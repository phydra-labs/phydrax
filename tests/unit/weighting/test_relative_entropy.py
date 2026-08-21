#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


TIGHT_TERMINATION = phx.optim.OptimizationTermination(
    absolute_optimality=1e-10,
    relative_optimality=0.0,
    maximum_steps=100,
)


def _problem(features, target, prior, *, mask=None):
    return phx.weighting.MomentCalibrationProblem(
        features,
        target,
        prior_log_weights=jnp.log(prior),
        mask=mask,
    )


def test_binary_exact_calibration_has_analytic_weights_dual_and_kl():
    features = jnp.array([[0.0], [1.0]])
    prior = jnp.array([0.25, 0.75])
    problem = _problem(
        features,
        phx.weighting.ExactMoments(jnp.array([0.5])),
        prior,
    )

    result = phx.weighting.calibrate_moments(
        problem,
        termination=TIGHT_TERMINATION,
    )

    assert result.successful
    assert jnp.allclose(result.weights, jnp.array([0.5, 0.5]), atol=1e-10)
    assert jnp.allclose(result.achieved_moments, jnp.array([0.5]), atol=1e-10)
    assert jnp.allclose(result.dual_variables, -jnp.log(3.0), atol=1e-9)
    expected_kl = 0.5 * jnp.log(0.5 / 0.25) + 0.5 * jnp.log(0.5 / 0.75)
    assert jnp.allclose(result.diagnostics.relative_entropy, expected_kl, atol=1e-10)
    assert int(result.diagnostics.numerical_affine_rank) == 1
    assert result.diagnostics.maximum_scaled_residual < 1e-9


def test_prior_solution_mask_and_extreme_logits_remain_exact_and_finite():
    features = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    mask = jnp.array([True, True, False, True])
    logits = jnp.array([1000.0, 999.0, -jnp.inf, 998.0])
    prior = jax.nn.softmax(jnp.array([0.0, -1.0, -2.0]))
    target = jnp.array([prior @ jnp.array([0.0, 1.0, 3.0])])
    problem = phx.weighting.MomentCalibrationProblem(
        features,
        phx.weighting.ExactMoments(target),
        prior_log_weights=logits,
        mask=mask,
    )

    result = phx.weighting.calibrate_moments(
        problem,
        termination=TIGHT_TERMINATION,
    )

    assert result.successful
    assert jnp.all(jnp.isfinite(result.weights))
    assert result.weights[2] == 0.0
    assert result.log_weights[2] == -jnp.inf
    assert jnp.allclose(result.dual_variables, 0.0, atol=1e-9)
    assert jnp.allclose(result.diagnostics.relative_entropy, 0.0, atol=1e-11)
    assert jnp.isclose(jnp.sum(result.weights), 1.0)


def test_exact_calibration_recovers_known_exponential_tilt_and_is_permutation_invariant():
    features = jnp.array(
        [
            [-1.0, 1.0],
            [-0.2, 0.04],
            [0.4, 0.16],
            [1.3, 1.69],
            [2.0, 4.0],
        ]
    )
    prior = jnp.array([0.1, 0.2, 0.25, 0.3, 0.15])
    known_dual = jnp.array([0.45, -0.2])
    expected = jax.nn.softmax(jnp.log(prior) + features @ known_dual)
    target = features.T @ expected
    problem = _problem(features, phx.weighting.ExactMoments(target), prior)

    result = phx.weighting.calibrate_moments(
        problem,
        termination=TIGHT_TERMINATION,
    )
    permutation = jnp.array([3, 0, 4, 1, 2])
    permuted = _problem(
        features[permutation],
        phx.weighting.ExactMoments(target),
        prior[permutation],
    )
    permuted_result = phx.weighting.calibrate_moments(
        permuted,
        termination=TIGHT_TERMINATION,
    )

    assert result.successful
    assert permuted_result.successful
    assert jnp.allclose(result.weights, expected, atol=2e-9)
    assert jnp.allclose(result.dual_variables, known_dual, atol=2e-8)
    assert jnp.allclose(
        permuted_result.weights,
        result.weights[permutation],
        atol=2e-9,
    )


def test_redundant_and_constant_moments_are_rank_reduced_without_changing_solution():
    x = jnp.linspace(-1.0, 1.0, 9)
    features = jnp.stack((x, 2.0 * x, jnp.ones_like(x)), axis=1)
    prior = jnp.ones((9,)) / 9.0
    target = jnp.array([0.25, 0.5, 1.0])
    problem = _problem(features, phx.weighting.ExactMoments(target), prior)

    result = phx.weighting.calibrate_moments(
        problem,
        termination=TIGHT_TERMINATION,
    )

    assert result.successful
    assert int(result.diagnostics.numerical_affine_rank) == 1
    assert result.diagnostics.affine_residual_norm < 1e-10
    assert jnp.allclose(result.achieved_moments, target, atol=1e-9)
    assert result.diagnostics.minimum_final_eigenvalue > 0.0


def test_affine_inconsistency_and_boundary_targets_never_report_success():
    x = jnp.array([0.0, 1.0, 2.0])
    features = jnp.stack((x, 2.0 * x, jnp.ones_like(x)), axis=1)
    prior = jnp.ones((3,)) / 3.0
    inconsistent = _problem(
        features,
        phx.weighting.ExactMoments(jnp.array([0.5, 1.2, 1.0])),
        prior,
    )
    boundary = _problem(
        x[:, None],
        phx.weighting.ExactMoments(jnp.array([0.0])),
        prior,
    )

    inconsistent_result = phx.weighting.calibrate_moments(inconsistent)
    boundary_result = phx.weighting.calibrate_moments(
        boundary,
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=1e-10,
            relative_optimality=0.0,
            maximum_steps=150,
        ),
    )

    assert inconsistent_result.status == int(
        phx.weighting.MomentCalibrationStatus.AFFINE_TARGET_INCONSISTENT
    )
    assert not inconsistent_result.successful
    assert not boundary_result.successful
    assert boundary_result.status in (
        int(phx.weighting.MomentCalibrationStatus.REGULARITY_NOT_CERTIFIED),
        int(phx.weighting.MomentCalibrationStatus.TARGET_RESIDUAL_NOT_MET),
        int(phx.weighting.MomentCalibrationStatus.OPTIMIZATION_FAILED),
    )


def test_quadratic_calibration_recovers_known_soft_stationary_point():
    features = jnp.array([[-1.0, 0.5], [0.0, -0.2], [0.7, 0.4], [1.5, 1.2]])
    prior = jnp.array([0.15, 0.25, 0.4, 0.2])
    scale = jnp.array([0.3, 0.5])
    known_dual = jnp.array([0.4, -0.25])
    expected = jax.nn.softmax(jnp.log(prior) + features @ known_dual)
    achieved = features.T @ expected
    target = achieved + scale**2 * known_dual
    problem = _problem(
        features,
        phx.weighting.QuadraticMoments(target, scale=scale),
        prior,
    )

    result = phx.weighting.calibrate_moments(
        problem,
        termination=TIGHT_TERMINATION,
    )

    assert result.successful
    assert jnp.allclose(result.weights, expected, atol=2e-9)
    assert jnp.allclose(result.dual_variables, known_dual, atol=2e-8)
    assert result.diagnostics.dual_gradient_norm < 1e-8


def test_quadratic_scale_controls_target_fit_and_prior_shrinkage():
    features = jnp.array([[0.0], [1.0], [2.0]])
    prior = jnp.array([0.2, 0.5, 0.3])

    def solve(scale):
        problem = _problem(
            features,
            phx.weighting.QuadraticMoments(jnp.array([-1.0]), scale=scale),
            prior,
        )
        return phx.weighting.calibrate_moments(
            problem,
            termination=TIGHT_TERMINATION,
        )

    tight = solve(0.1)
    loose = solve(2.0)

    assert tight.successful and loose.successful
    assert jnp.abs(tight.achieved_moments[0] + 1.0) < jnp.abs(
        loose.achieved_moments[0] + 1.0
    )
    assert tight.diagnostics.relative_entropy > loose.diagnostics.relative_entropy
    assert jnp.linalg.norm(loose.weights - prior) < jnp.linalg.norm(tight.weights - prior)


def test_dense_sparse_and_function_operator_calibrations_agree():
    features = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 2.0], [2.0, 1.0]])
    prior = jnp.array([0.1, 0.2, 0.3, 0.4])
    target = jnp.array([1.1, 1.05])
    source_indices = jnp.tile(jnp.arange(4, dtype=jnp.int32), 2)
    target_indices = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 4)
    relation = phx.sparse.EdgeRelation(
        source_indices,
        target_indices,
        source_size=4,
        target_size=2,
    )
    sparse_map = phx.sparse.SparseLinearMap(relation, features.T.reshape((-1,)))
    source_space = phx.linalg.ArraySpace((4,), dtype=features.dtype)
    target_space = phx.linalg.ArraySpace((2,), dtype=features.dtype)
    function_map = phx.linalg.FunctionLinearOperator(
        lambda weights: features.T @ weights,
        source=source_space,
        target=target_space,
        transpose_action=lambda dual: features @ dual,
    )
    operator_termination = phx.optim.OptimizationTermination(
        absolute_optimality=1e-8,
        relative_optimality=0.0,
        maximum_steps=100,
    )

    dense_result = phx.weighting.calibrate_moments(
        _problem(features, phx.weighting.ExactMoments(target), prior),
        termination=operator_termination,
    )
    sparse_result = phx.weighting.calibrate_moments(
        phx.weighting.MomentCalibrationProblem(
            sparse_map,
            phx.weighting.ExactMoments(target),
            prior_log_weights=jnp.log(prior),
        ),
        termination=operator_termination,
    )
    function_result = phx.weighting.calibrate_moments(
        phx.weighting.MomentCalibrationProblem(
            function_map,
            phx.weighting.ExactMoments(target),
            prior_log_weights=jnp.log(prior),
        ),
        termination=operator_termination,
    )

    assert dense_result.successful
    assert sparse_result.successful
    assert function_result.successful
    assert jnp.allclose(sparse_result.weights, dense_result.weights, atol=2e-9)
    assert jnp.allclose(function_result.weights, dense_result.weights, atol=2e-9)
    assert sparse_result.provenance.execution == "operator"
    assert function_result.provenance.execution == "operator"


def test_calibration_is_filter_jittable_and_warm_start_reuses_dual():
    x = jnp.linspace(-2.0, 2.0, 31)
    features = jnp.stack((x, x**2), axis=1)
    prior = jax.nn.softmax(-0.3 * x**2)
    first_problem = _problem(
        features,
        phx.weighting.ExactMoments(jnp.array([0.15, 0.9])),
        prior,
    )
    nearby_problem = _problem(
        features,
        phx.weighting.ExactMoments(jnp.array([0.16, 0.91])),
        prior,
    )
    compiled = eqx.filter_jit(
        lambda problem: phx.weighting.calibrate_moments(
            problem,
            termination=TIGHT_TERMINATION,
        )
    )

    first = compiled(first_problem)
    eager_nearby = phx.weighting.calibrate_moments(
        nearby_problem,
        termination=TIGHT_TERMINATION,
    )
    warm_nearby = phx.weighting.calibrate_moments(
        nearby_problem,
        termination=TIGHT_TERMINATION,
        initial_dual=first.dual_variables,
    )
    compiled_nearby = compiled(nearby_problem)

    assert first.successful and eager_nearby.successful and warm_nearby.successful
    assert compiled_nearby.successful
    assert jnp.allclose(compiled_nearby.weights, eager_nearby.weights, atol=1e-9)
    assert warm_nearby.diagnostics.optimization.iterations <= (
        eager_nearby.diagnostics.optimization.iterations
    )


def test_implicit_exact_derivatives_match_analytic_and_finite_difference_results():
    features = jnp.array([[0.0], [1.0]])
    prior_logits = jnp.log(jnp.array([0.25, 0.75]))

    def solve_target(target):
        problem = phx.weighting.MomentCalibrationProblem(
            features,
            phx.weighting.ExactMoments(jnp.reshape(target, (1,))),
            prior_log_weights=prior_logits,
        )
        return phx.weighting.implicit_calibrate_moments(
            problem,
            termination=TIGHT_TERMINATION,
        )

    forward = jax.jacfwd(solve_target)(jnp.array(0.5))
    reverse = jax.jacrev(solve_target)(jnp.array(0.5))

    assert jnp.allclose(forward, jnp.array([-1.0, 1.0]), atol=1e-8)
    assert jnp.allclose(reverse, forward, atol=1e-8)

    three_features = jnp.array([[0.0], [1.0], [2.0]])
    direction = jnp.array([0.3, -0.2, 0.1])

    def solve_prior(logits):
        problem = phx.weighting.MomentCalibrationProblem(
            three_features,
            phx.weighting.ExactMoments(jnp.array([0.8])),
            prior_log_weights=logits,
        )
        return phx.weighting.implicit_calibrate_moments(
            problem,
            termination=phx.optim.OptimizationTermination(
                absolute_optimality=1e-8,
                relative_optimality=0.0,
                maximum_steps=100,
            ),
        )

    logits = jnp.log(jnp.array([0.2, 0.5, 0.3]))
    _, tangent = jax.jvp(solve_prior, (logits,), (direction,))
    step = 2e-5
    finite_difference = (
        solve_prior(logits + step * direction) - solve_prior(logits - step * direction)
    ) / (2.0 * step)
    assert jnp.allclose(tangent, finite_difference, atol=2e-6, rtol=2e-5)


def test_problem_validation_precision_and_convergence_guard_contracts():
    with pytest.raises(ValueError, match="Target moments"):
        phx.weighting.MomentCalibrationProblem(
            jnp.ones((3, 2)),
            phx.weighting.ExactMoments(jnp.ones((3,))),
        )
    with pytest.raises(ValueError, match="finite"):
        phx.weighting.MomentCalibrationProblem(
            jnp.array([[0.0], [jnp.nan]]),
            phx.weighting.ExactMoments(jnp.array([0.0])),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        phx.weighting.QuadraticMoments(jnp.array([0.0]), scale=0.0)

    for dtype in (jnp.float32, jnp.float64):
        features = jnp.array([[0.0], [1.0]], dtype=dtype)
        problem = phx.weighting.MomentCalibrationProblem(
            features,
            phx.weighting.ExactMoments(jnp.array([0.5], dtype=dtype)),
        )
        result = phx.weighting.calibrate_moments(
            problem,
            termination=TIGHT_TERMINATION,
        )
        assert result.weights.dtype == dtype
        assert result.dual_variables.dtype == dtype

    failed_problem = _problem(
        jnp.array([[0.0], [1.0]]),
        phx.weighting.ExactMoments(jnp.array([2.0])),
        jnp.array([0.5, 0.5]),
    )
    failed = phx.weighting.calibrate_moments(failed_problem)
    with pytest.raises(eqx.EquinoxRuntimeError, match="did not converge"):
        phx.weighting.require_converged(failed)
    compiled_guard = eqx.filter_jit(phx.weighting.require_converged)
    with pytest.raises(Exception, match="did not converge"):
        jax.block_until_ready(compiled_guard(failed).weights)
