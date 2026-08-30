#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pytest

import phydrax as phx


def _dense_gp(kernel, train_times, train_values, query_times, mask, noise_scale):
    active_times = train_times[mask]
    active_values = train_values[mask]
    query_covariance = kernel.matrix(query_times, query_times)
    if active_times.size == 0:
        return jnp.asarray(0.0), jnp.zeros_like(query_times), jnp.diag(query_covariance)
    covariance = kernel.matrix(active_times, active_times)
    covariance = covariance + noise_scale**2 * jnp.eye(
        active_times.size, dtype=covariance.dtype
    )
    factor = jnp.linalg.cholesky(covariance)
    solved = jsp.linalg.solve_triangular(factor, active_values, lower=True)
    alpha = jsp.linalg.solve_triangular(factor.T, solved, lower=False)
    cross = kernel.matrix(query_times, active_times)
    projected = jsp.linalg.solve_triangular(factor, cross.T, lower=True)
    mean = cross @ alpha
    variance = jnp.diag(query_covariance) - jnp.sum(projected**2, axis=0)
    log_marginal_likelihood = -0.5 * (
        active_values @ alpha
        + 2.0 * jnp.sum(jnp.log(jnp.diag(factor)))
        + active_times.size * jnp.log(2.0 * jnp.pi)
    )
    return log_marginal_likelihood, mean, variance


@pytest.mark.parametrize(
    ("kernel", "dimension", "tolerance"),
    [
        (phx.kernels.Matern32Kernel(length_scale=0.7), 2, 2e-6),
        (phx.kernels.Matern52Kernel(length_scale=1.1), 3, 5e-6),
        (
            phx.kernels.ScaleKernel(
                phx.kernels.Matern52Kernel(length_scale=0.45), 1.7
            ),
            3,
            8e-6,
        ),
    ],
)
def test_compiled_state_covariance_and_factors_match_matern(
    kernel, dimension, tolerance
):
    plan = phx.uq.compile_state_space_kernel(
        kernel,
        jnp.asarray([-0.4, 0.2]),
        jnp.asarray([1.3]),
    )
    lags = jnp.asarray([0.0, 0.03, 0.4, 1.8])

    def state_covariance(lag):
        transition = jsp.linalg.expm(plan.drift_matrix * lag)
        return (
            plan.observation_map
            @ plan.stationary_covariance
            @ transition.T
            @ plan.observation_map.T
        ).reshape(())

    expected = jax.vmap(lambda lag: kernel.pairwise(jnp.asarray([0.0]), lag[None]))(
        lags
    )
    actual = jax.vmap(state_covariance)(lags)

    assert plan.state_dimension == dimension
    assert jnp.allclose(actual, expected, rtol=tolerance, atol=tolerance)
    assert jnp.allclose(
        plan.stationary_factor @ plan.stationary_factor.T,
        plan.stationary_covariance,
        rtol=tolerance,
        atol=tolerance,
    )
    assert jnp.allclose(
        plan.process_noise_factor @ plan.process_noise_factor.T,
        plan.process_noise,
        rtol=tolerance,
        atol=tolerance,
    )
    assert jnp.max(jnp.abs(plan.lyapunov_residual)) <= tolerance
    reconstructed_residual = (
        plan.drift_matrix @ plan.stationary_covariance
        + plan.stationary_covariance @ plan.drift_matrix.T
        + plan.process_noise
    )
    assert jnp.allclose(reconstructed_residual, plan.lyapunov_residual)
    problem = plan._problem_template
    first_parameters = problem.model.transition.parameters(
        problem.initial_time,
        plan.schedule_times[0],
        problem.step_context(0, 0),
    )
    first_factor = phx.uq.gaussian_factor_from_covariance(
        first_parameters.covariance
    )
    assert jnp.any(first_factor.factor != 0.0)
    assert jnp.allclose(
        first_factor.factor @ first_factor.factor.T,
        first_parameters.covariance,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize(
    "kernel",
    [
        phx.kernels.Matern32Kernel(length_scale=0.6),
        phx.kernels.ScaleKernel(
            phx.kernels.Matern52Kernel(length_scale=0.9), 1.4
        ),
    ],
)
def test_likelihood_and_posterior_match_independent_dense_gp(kernel):
    train_times = jnp.asarray([1.1, -0.2, 0.45, 2.0, 0.1, 0.8])
    train_values = jnp.sin(1.7 * train_times) + 0.1 * train_times
    train_mask = jnp.asarray([True, False, True, True, True, False])
    query_times = jnp.asarray([2.8, -1.0, 0.45, 1.5, -0.4, 0.45])
    noise_scale = jnp.asarray(0.08)
    plan = phx.uq.compile_state_space_kernel(
        kernel,
        train_times,
        query_times,
        train_mask=train_mask,
    )

    result = phx.uq.fit_state_space_gaussian_process(
        plan,
        train_values,
        noise_scale=noise_scale,
    )
    dense_likelihood, dense_mean, dense_variance = _dense_gp(
        kernel,
        train_times,
        train_values,
        query_times,
        train_mask,
        noise_scale,
    )
    masked_values_changed = phx.uq.fit_state_space_gaussian_process(
        plan,
        jnp.where(train_mask, train_values, 12345.0),
        noise_scale=noise_scale,
    )

    assert bool(result.successful)
    assert result.active_observation_count == jnp.sum(train_mask)
    assert jnp.array_equal(result.posterior_times, query_times)
    assert jnp.allclose(
        result.log_marginal_likelihood,
        dense_likelihood,
        rtol=2e-5,
        atol=2e-5,
    )
    assert jnp.allclose(result.posterior_mean, dense_mean, rtol=3e-5, atol=3e-5)
    assert jnp.allclose(
        result.posterior_variance,
        dense_variance,
        rtol=5e-5,
        atol=5e-5,
    )
    assert jnp.allclose(
        result.predictive_variance,
        dense_variance + noise_scale**2,
        rtol=5e-5,
        atol=5e-5,
    )
    assert jnp.array_equal(
        masked_values_changed.log_marginal_likelihood,
        result.log_marginal_likelihood,
    )
    assert jnp.array_equal(masked_values_changed.posterior_mean, result.posterior_mean)
    assert jnp.array_equal(
        masked_values_changed.posterior_variance,
        result.posterior_variance,
    )
    assert result.posterior_mean[2] == result.posterior_mean[5]
    assert result.posterior_variance[2] == result.posterior_variance[5]
    assert result.filter_result.covariance_form == "square_root"
    assert result.smoother_result.covariance_form == "square_root"
    assert result.method_id == "exact-matern-sequential-square-root-kalman-rts"
    assert result.kernel_content_id
    assert result.schedule_id
    assert result.precision_evidence.evidence_id


@pytest.mark.parametrize(
    "kernel",
    [
        phx.kernels.Matern32Kernel(length_scale=0.2),
        phx.kernels.Matern52Kernel(length_scale=0.2),
    ],
)
def test_widely_separated_schedule_matches_dense_without_van_loan_overflow(kernel):
    train_times = jnp.asarray([0.0, 12.0, 25.0])
    train_values = jnp.asarray([0.3, -0.7, 0.2])
    query_times = jnp.asarray([-8.0, 6.0, 18.0, 40.0])
    noise_scale = jnp.asarray(0.05)
    mask = jnp.ones(train_times.shape, dtype=bool)
    plan = phx.uq.compile_state_space_kernel(kernel, train_times, query_times)
    result = phx.uq.fit_state_space_gaussian_process(
        plan,
        train_values,
        noise_scale=noise_scale,
    )
    dense = _dense_gp(
        kernel,
        train_times,
        train_values,
        query_times,
        mask,
        noise_scale,
    )

    assert bool(result.successful)
    assert jnp.all(jnp.isfinite(result.posterior_mean))
    assert jnp.all(jnp.isfinite(result.posterior_variance))
    assert jnp.allclose(result.log_marginal_likelihood, dense[0], rtol=2e-7, atol=2e-7)
    assert jnp.allclose(result.posterior_mean, dense[1], rtol=2e-7, atol=2e-7)
    assert jnp.allclose(result.posterior_variance, dense[2], rtol=2e-7, atol=2e-7)


def test_schedule_is_stable_sorted_masked_and_invertible():
    train_times = jnp.asarray([3.0, -1.0, 0.5, 2.0])
    query_times = jnp.asarray([2.0, -2.0, 2.0, 4.0, 0.25])
    mask = jnp.asarray([True, False, True, True])
    plan = phx.uq.compile_state_space_kernel(
        phx.kernels.Matern32Kernel(length_scale=0.4),
        train_times,
        query_times,
        train_mask=mask,
    )

    assert jnp.array_equal(
        plan.schedule_times, jnp.asarray([-2.0, -1.0, 0.25, 0.5, 2.0, 3.0, 4.0])
    )
    assert jnp.array_equal(plan.schedule_times[plan.train_schedule_indices], train_times)
    assert jnp.array_equal(plan.schedule_times[plan.query_schedule_indices], query_times)
    assert jnp.array_equal(
        plan.train_sort_indices[plan.train_inverse_permutation],
        jnp.arange(train_times.size),
    )
    assert jnp.array_equal(
        plan.query_sort_indices[plan.query_inverse_permutation],
        jnp.arange(query_times.size),
    )
    assert jnp.array_equal(
        plan.schedule_observation_mask,
        jnp.asarray([False, False, False, True, True, True, False]),
    )
    assert "training times must be unique" in plan.repeated_time_policy
    assert "share one latent schedule state" in plan.repeated_time_policy
    changed_kernel = phx.uq.compile_state_space_kernel(
        phx.kernels.Matern32Kernel(length_scale=0.6),
        train_times,
        query_times,
        train_mask=mask,
    )
    changed_mask = phx.uq.compile_state_space_kernel(
        phx.kernels.Matern32Kernel(length_scale=0.4),
        train_times,
        query_times,
        train_mask=jnp.ones(mask.shape, dtype=bool),
    )
    assert changed_kernel.kernel_content_id != plan.kernel_content_id
    assert changed_mask.schedule_id != plan.schedule_id


def test_repeated_training_times_are_rejected_but_repeated_queries_are_shared():
    kernel = phx.kernels.Matern32Kernel(length_scale=1.0)
    with pytest.raises(ValueError, match="Repeated training times"):
        phx.uq.compile_state_space_kernel(
            kernel,
            jnp.asarray([0.0, 1.0, 0.0]),
            jnp.asarray([0.5]),
        )

    plan = phx.uq.compile_state_space_kernel(
        kernel,
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([0.5, 0.5, 1.0]),
    )
    assert plan.query_schedule_indices[0] == plan.query_schedule_indices[1]
    assert plan.query_schedule_indices[2] == plan.train_schedule_indices[1]


def test_empty_active_observations_return_stationary_prior_and_zero_likelihood():
    kernel = phx.kernels.ScaleKernel(
        phx.kernels.Matern52Kernel(length_scale=0.8), 2.5
    )
    train_times = jnp.asarray([0.0, 0.7, 1.2])
    query_times = jnp.asarray([-1.0, 0.4, 2.5])
    plan = phx.uq.compile_state_space_kernel(
        kernel,
        train_times,
        query_times,
        train_mask=jnp.zeros(train_times.shape, dtype=bool),
    )
    result = phx.uq.fit_state_space_gaussian_process(
        plan,
        jnp.zeros(train_times.shape),
        noise_scale=0.0,
    )

    assert bool(result.successful)
    assert result.active_observation_count == 0
    assert result.log_marginal_likelihood == 0.0
    assert jnp.allclose(result.posterior_mean, 0.0)
    assert jnp.allclose(result.posterior_variance, 2.5, rtol=2e-5, atol=2e-5)
    assert not bool(jnp.any(result.schedule_observation_mask))
    assert jnp.all(result.filter_result.incremental_log_likelihood == 0.0)


@pytest.mark.parametrize("noise_scale", [0.0, 1e-7])
def test_zero_and_tiny_observation_noise_match_dense_algebra(noise_scale):
    kernel = phx.kernels.Matern32Kernel(length_scale=0.55)
    train_times = jnp.asarray([-0.7, -0.1, 0.4, 1.3])
    train_values = jnp.asarray([0.2, -0.4, 0.8, 0.1])
    query_times = jnp.asarray([-1.4, 0.0, 0.9, 2.0])
    mask = jnp.ones(train_times.shape, dtype=bool)
    plan = phx.uq.compile_state_space_kernel(kernel, train_times, query_times)
    result = phx.uq.fit_state_space_gaussian_process(
        plan, train_values, noise_scale=noise_scale
    )
    dense_likelihood, dense_mean, dense_variance = _dense_gp(
        kernel,
        train_times,
        train_values,
        query_times,
        mask,
        jnp.asarray(noise_scale),
    )

    assert bool(result.successful)
    assert jnp.allclose(result.log_marginal_likelihood, dense_likelihood, rtol=3e-5)
    assert jnp.allclose(result.posterior_mean, dense_mean, rtol=3e-5, atol=3e-5)
    assert jnp.allclose(
        result.posterior_variance, dense_variance, rtol=8e-5, atol=8e-5
    )


def test_degenerate_zero_signal_and_zero_noise_reports_failure_without_repair():
    kernel = phx.kernels.ScaleKernel(
        phx.kernels.Matern32Kernel(length_scale=0.7), 0.0
    )
    plan = phx.uq.compile_state_space_kernel(
        kernel,
        jnp.asarray([0.0]),
        jnp.asarray([0.5]),
    )
    result = phx.uq.fit_state_space_gaussian_process(
        plan,
        jnp.asarray([0.0]),
        noise_scale=0.0,
    )

    assert not bool(result.successful)
    assert int(result.status) != phx.uq.KALMAN_SUCCESS
    assert result.filter_result.covariance_regularization == 0.0


@pytest.mark.parametrize(
    ("x64", "expected_dtype", "tolerance"),
    [
        (False, jnp.float32, 5e-5),
        (True, jnp.float64, 2e-10),
    ],
)
def test_float32_and_float64_precision_is_retained(x64, expected_dtype, tolerance):
    with jax.enable_x64(x64):
        kernel = phx.kernels.Matern52Kernel(length_scale=0.65)
        train_times = jnp.asarray([0.0, 0.3, 1.0], dtype=expected_dtype)
        query_times = jnp.asarray([-0.2, 0.7, 1.8], dtype=expected_dtype)
        plan = phx.uq.compile_state_space_kernel(kernel, train_times, query_times)
        result = phx.uq.fit_state_space_gaussian_process(
            plan,
            jnp.asarray([0.1, -0.2, 0.5], dtype=expected_dtype),
            noise_scale=jnp.asarray(0.04, dtype=expected_dtype),
        )
        dense = _dense_gp(
            kernel,
            train_times,
            jnp.asarray([0.1, -0.2, 0.5], dtype=expected_dtype),
            query_times,
            jnp.ones((3,), dtype=bool),
            jnp.asarray(0.04, dtype=expected_dtype),
        )

        assert result.posterior_mean.dtype == expected_dtype
        assert result.posterior_variance.dtype == expected_dtype
        assert dict(result.precision_evidence.observed)["output"] == expected_dtype.__name__
        assert jnp.allclose(result.posterior_mean, dense[1], rtol=tolerance, atol=tolerance)


def test_mixed_kernel_and_schedule_dtypes_are_rejected_explicitly():
    with jax.enable_x64(False):
        kernel = phx.kernels.Matern32Kernel(length_scale=0.5)
    with jax.enable_x64(True):
        with pytest.raises(TypeError, match="one identical compute dtype"):
            phx.uq.compile_state_space_kernel(
                kernel,
                jnp.asarray([0.0, 1.0], dtype=jnp.float64),
                jnp.asarray([0.5], dtype=jnp.float64),
            )


def test_fit_is_jittable_and_matches_dense_kernel_parameter_gradients():
    train_times = jnp.asarray([-0.3, 0.2, 0.9, 1.5])
    query_times = jnp.asarray([-0.8, 0.5, 2.0])
    plan = phx.uq.compile_state_space_kernel(
        phx.kernels.ScaleKernel(
            phx.kernels.Matern32Kernel(length_scale=0.75), 1.3
        ),
        train_times,
        query_times,
    )
    values = jnp.asarray([0.4, -0.1, 0.7, 0.2])
    compiled = eqx.filter_jit(phx.uq.fit_state_space_gaussian_process)(
        plan,
        values,
        noise_scale=0.06,
    )

    def candidate(length_scale, scale):
        return eqx.tree_at(
            lambda node: (
                node.kernel.kernel.length_scale,
                node.kernel.scale,
            ),
            plan,
            (length_scale, scale),
        )

    def state_space_objective(length_scale, scale):
        return phx.uq.fit_state_space_gaussian_process(
            candidate(length_scale, scale),
            values,
            noise_scale=0.06,
        ).log_marginal_likelihood

    def dense_objective(length_scale, scale):
        return _dense_gp(
            candidate(length_scale, scale).kernel,
            train_times,
            values,
            query_times,
            jnp.ones(train_times.shape, dtype=bool),
            jnp.asarray(0.06),
        )[0]

    arguments = (plan.kernel.kernel.length_scale, plan.kernel.scale)
    gradient = jax.grad(state_space_objective, argnums=(0, 1))(*arguments)
    dense_gradient = jax.grad(dense_objective, argnums=(0, 1))(*arguments)

    assert bool(compiled.successful)
    assert compiled.posterior_mean.shape == (3,)
    assert jnp.all(jnp.isfinite(jnp.stack(gradient)))
    assert jnp.allclose(
        jnp.stack(gradient),
        jnp.stack(dense_gradient),
        rtol=2e-5,
        atol=2e-5,
    )


def test_mutated_kernel_reports_evaluated_identity_and_exported_parameters(tmp_path):
    train_times = jnp.asarray([0.0, 0.4, 1.0])
    query_times = jnp.asarray([-0.3, 0.7, 1.6])
    plan = phx.uq.compile_state_space_kernel(
        phx.kernels.Matern52Kernel(length_scale=0.5),
        train_times,
        query_times,
    )
    mutated = eqx.tree_at(
        lambda node: node.kernel.length_scale,
        plan,
        jnp.asarray(0.9),
    )
    result = phx.uq.fit_state_space_gaussian_process(
        mutated,
        jnp.asarray([0.2, -0.1, 0.4]),
        noise_scale=0.03,
    )
    expected = phx.uq.compile_state_space_kernel(
        mutated.kernel,
        train_times,
        query_times,
    )

    assert result.kernel_content_id == expected.kernel_content_id
    assert result.kernel_content_id != plan.kernel_content_id
    assert result.prepared_kernel_content_id == plan.kernel_content_id
    assert result.evaluated_length_scale == 0.9
    path = phx.uq.export_result(result, tmp_path / "mutated-state-space-gp")
    archive = phx.uq.read_result_archive(path)
    assert archive.metadata["kernel_content_id"] == expected.kernel_content_id
    assert archive.metadata["prepared_kernel_content_id"] == plan.kernel_content_id
    assert archive.array("evaluated_length_scale") == 0.9


def test_smoother_invalidity_has_a_non_success_gp_status():
    from phydrax.uq._state_space_gp import _state_space_gp_status

    status = _state_space_gp_status(
        jnp.asarray(True),
        jnp.asarray([phx.uq.KALMAN_SUCCESS], dtype=jnp.int32),
        jnp.asarray([True, False]),
    )

    assert int(status) == phx.uq.STATE_SPACE_GP_SMOOTHER_FAILURE
    assert phx.uq.state_space_gaussian_process_status_name(int(status)) == (
        "smoother_failure"
    )


def test_benchmark_storage_counts_complete_unique_retained_arrays():
    from tools.state_space_gp_benchmarks import _retained_storage

    shared = jnp.ones((3,))
    other = jnp.ones((2, 2), dtype=jnp.float32)
    elements, bytes_ = _retained_storage((shared, {"alias": shared, "other": other}))
    assert elements == shared.size + other.size
    assert bytes_ == shared.nbytes + other.nbytes

    plan = phx.uq.compile_state_space_kernel(
        phx.kernels.Matern32Kernel(length_scale=0.6),
        jnp.asarray([0.0, 0.5, 1.0]),
        jnp.asarray([-0.2, 0.8, 1.4]),
    )
    result = phx.uq.fit_state_space_gaussian_process(
        plan,
        jnp.asarray([0.1, -0.2, 0.3]),
        noise_scale=0.04,
    )
    retained_elements, retained_bytes = _retained_storage(result)
    old_partial_count = (
        result.smoother_result.means.size
        + result.smoother_result.covariances.size
        + plan.schedule_size * plan.state_dimension**2
    )
    assert retained_elements > old_partial_count
    assert retained_bytes > old_partial_count * result.posterior_mean.dtype.itemsize


def test_preparation_and_fit_reject_unsupported_or_invalid_inputs():
    train_times = jnp.asarray([0.0, 1.0])
    query_times = jnp.asarray([0.5])
    invalid_length_scale = eqx.tree_at(
        lambda node: node.length_scale,
        phx.kernels.Matern32Kernel(length_scale=0.5),
        jnp.asarray(-0.5),
    )
    with pytest.raises(ValueError, match="finite and positive"):
        phx.uq.compile_state_space_kernel(
            invalid_length_scale,
            train_times,
            query_times,
        )
    with pytest.raises(ValueError, match="length_scale must be scalar"):
        phx.uq.compile_state_space_kernel(
            phx.kernels.Matern32Kernel(length_scale=jnp.asarray([0.5])),
            train_times,
            query_times,
        )
    with pytest.raises(TypeError, match="Matern32Kernel or Matern52Kernel"):
        phx.uq.compile_state_space_kernel(
            phx.kernels.SquaredExponentialKernel(length_scale=0.5),
            train_times,
            query_times,
        )
    with pytest.raises(ValueError, match="max_schedule_size"):
        phx.uq.compile_state_space_kernel(
            phx.kernels.Matern52Kernel(length_scale=0.5),
            train_times,
            query_times,
            max_schedule_size=2,
        )
    with pytest.raises(ValueError, match="At least one"):
        phx.uq.compile_state_space_kernel(
            phx.kernels.Matern52Kernel(length_scale=0.5),
            jnp.empty((0,)),
            jnp.empty((0,)),
        )

    plan = phx.uq.compile_state_space_kernel(
        phx.kernels.Matern52Kernel(length_scale=0.5),
        train_times,
        query_times,
    )
    with pytest.raises(ValueError, match="train_values"):
        phx.uq.fit_state_space_gaussian_process(plan, jnp.asarray([0.0]))
    with pytest.raises(ValueError, match="noise_scale"):
        phx.uq.fit_state_space_gaussian_process(
            plan,
            jnp.asarray([0.0, 0.1]),
            noise_scale=jnp.asarray([0.1, 0.2]),
        )


def test_state_space_gp_result_uses_portable_result_export(tmp_path):
    plan = phx.uq.compile_state_space_kernel(
        phx.kernels.Matern32Kernel(length_scale=0.8),
        jnp.asarray([0.0, 0.5, 1.0]),
        jnp.asarray([-0.5, 0.25, 1.5]),
        train_mask=jnp.asarray([True, False, True]),
    )
    result = phx.uq.fit_state_space_gaussian_process(
        plan,
        jnp.asarray([0.2, 0.0, -0.1]),
        noise_scale=0.03,
    )
    path = phx.uq.export_result(result, tmp_path / "state-space-gp")
    archive = phx.uq.read_result_archive(path)

    assert archive.kind == "state_space_gaussian_process"
    assert archive.metadata["kernel_content_id"] == result.kernel_content_id
    assert archive.metadata["prepared_kernel_content_id"] == plan.kernel_content_id
    assert archive.array("evaluated_length_scale") == 0.8
    assert archive.array("evaluated_scale") == 1.0
    assert archive.metadata["schedule_id"] == result.schedule_id
    assert archive.metadata["method_id"] == result.method_id
    assert archive.metadata["precision_evidence"]["evidence_id"]
    assert np.array_equal(archive.array("posterior_times"), np.asarray(result.posterior_times))
    assert np.allclose(archive.array("posterior_mean"), np.asarray(result.posterior_mean))
    assert set(archive.excluded) == {"filter_result", "smoother_result"}
