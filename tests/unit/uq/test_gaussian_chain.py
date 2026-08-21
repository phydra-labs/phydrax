import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


_ARRAY_FIELDS = (
    "predicted_means",
    "predicted_covariances",
    "filtered_means",
    "filtered_covariances",
    "transition_matrices",
    "innovations",
    "innovation_covariances",
    "normalized_innovation_squared",
    "incremental_log_likelihood",
    "cumulative_log_likelihood",
    "observed_counts",
    "step_valid",
    "valid",
    "status",
)


def _problem(*, num_steps=9, failed=False):
    base_times = jnp.cumsum(jnp.linspace(0.05, 0.2, num_steps))
    times = jnp.stack((base_times, base_times * 1.13))
    valid_count = max(num_steps - 3, 1)
    step_valid = jnp.stack(
        (
            jnp.ones(num_steps, dtype=bool),
            jnp.arange(num_steps) < valid_count,
        )
    )
    times = times.at[1, valid_count:].set(times[1, valid_count - 1])
    values = jnp.stack(
        (
            jnp.sin(base_times),
            -0.3 + jnp.cos(base_times),
        )
    )[..., None]
    mask = jnp.broadcast_to(step_valid[..., None], values.shape)
    if num_steps >= 4:
        mask = mask.at[0, 2, 0].set(False)
        mask = mask.at[1, 1, 0].set(False)
    observations = phx.stochastic.ObservationSequence(
        times,
        values,
        case_axes=("case",),
        case_shape=(2,),
        step_valid=step_valid,
        observation_mask=mask,
        case_ids=("first", "second"),
        sequence_id=f"gaussian-chain-{num_steps}-{failed}",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([[0.0, 0.2], [-0.1, 0.3]]),
        jnp.asarray([[0.8, 0.1], [0.1, 0.5]]),
        state_shape=(2,),
        prior_id="chain-prior",
    )
    dynamics = phx.stochastic.LinearGaussianDynamics(
        jnp.asarray([[-0.2, 0.7], [0.0, -0.4]]),
        jnp.asarray([[0.3], [0.2]]),
        state_shape=(2,),
        offset=jnp.asarray([0.05, -0.02]),
        dynamics_id="chain-dynamics",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(dynamics)
    observation_covariance = -jnp.ones((1, 1)) if failed else jnp.asarray([[0.15]])
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0, -0.25]]),
        observation_covariance,
        state_shape=(2,),
        observation_shape=(1,),
        offset=jnp.asarray([0.1]),
    )
    model = phx.stochastic.StateSpaceModel(
        prior, transition, observation, model_id=f"chain-model-{failed}"
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        observations,
        initial_time=jnp.asarray([0.0, 0.0]),
        problem_id=f"chain-problem-{num_steps}-{failed}",
    )


def _assert_filter_equivalent(sequential, parallel, *, tolerance=2e-9):
    for field in _ARRAY_FIELDS:
        assert jnp.allclose(
            getattr(sequential, field),
            getattr(parallel, field),
            rtol=tolerance,
            atol=tolerance,
            equal_nan=True,
        ), field
    assert jnp.allclose(
        sequential.final_state.mean,
        parallel.final_state.mean,
        rtol=tolerance,
        atol=tolerance,
    )
    assert jnp.allclose(
        sequential.final_state.covariance,
        parallel.final_state.covariance,
        rtol=tolerance,
        atol=tolerance,
    )
    assert jnp.allclose(
        sequential.final_state.log_likelihood,
        parallel.final_state.log_likelihood,
        rtol=tolerance,
        atol=tolerance,
    )


def test_parallel_filter_matches_irregular_masked_padded_batch():
    problem = _problem()
    sequential = phx.uq.kalman_filter(problem, method="sequential")
    parallel = phx.uq.kalman_filter(problem, method="parallel")
    _assert_filter_equivalent(sequential, parallel)
    assert sequential.execution_method == "sequential"
    assert parallel.execution_method == "parallel"
    assert jnp.array_equal(parallel.filtered_means[1, -1], parallel.filtered_means[1, -2])


def test_parallel_filter_preserves_multidimensional_case_batches():
    case_shape = (2, 2)
    num_steps = 5
    observations = phx.stochastic.ObservationSequence(
        jnp.linspace(0.1, 0.5, num_steps),
        jnp.arange(20, dtype=float).reshape(case_shape + (num_steps, 1)) / 10.0,
        case_axes=("row", "column"),
        case_shape=case_shape,
        case_ids=("00", "01", "10", "11"),
    )
    covariance = jnp.asarray([[[[0.4]], [[0.5]]], [[[0.6]], [[0.7]]]])
    prior = phx.stochastic.GaussianStatePrior(
        jnp.zeros(case_shape + (1,)),
        covariance,
        state_shape=(1,),
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[1.0]]), jnp.asarray([[0.05]]), state_shape=(1,)
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    problem = phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            prior, transition, observation, model_id="matrix-case-model"
        ),
        observations,
        initial_time=jnp.zeros(case_shape),
        problem_id="matrix-case-problem",
    )
    sequential = phx.uq.kalman_filter(problem, method="sequential")
    parallel = phx.uq.kalman_filter(problem, method="parallel")
    _assert_filter_equivalent(sequential, parallel)
    assert parallel.filtered_means.shape == case_shape + (num_steps, 1)


def test_parallel_filter_freezes_failed_cases_exactly():
    problem = _problem(failed=True)
    sequential = phx.uq.kalman_filter(problem, method="sequential")
    parallel = phx.uq.kalman_filter(problem, method="parallel")
    _assert_filter_equivalent(sequential, parallel)
    assert not jnp.any(parallel.successful)
    assert jnp.array_equal(parallel.filtered_means, sequential.filtered_means)
    assert jnp.array_equal(parallel.status, sequential.status)


@pytest.mark.parametrize("method", ("sequential", "parallel", "auto"))
@pytest.mark.parametrize("regularization", (-1.0, jnp.nan))
def test_filter_methods_share_regularization_validation(method, regularization):
    with pytest.raises(ValueError, match="must be finite and nonnegative"):
        phx.uq.kalman_filter(
            _problem(),
            method=method,
            covariance_regularization=regularization,
        )


def test_parallel_smoother_and_coherent_samples_are_equivalent_and_prefix_stable():
    problem = _problem()
    filtered = phx.uq.kalman_filter(problem, method="parallel")
    sequential = phx.uq.rts_smoother(filtered, method="sequential")
    parallel = phx.uq.rts_smoother(filtered, method="parallel")
    assert parallel.execution_method == "parallel"
    assert jnp.allclose(parallel.means, sequential.means, rtol=2e-9, atol=2e-9)
    assert jnp.allclose(
        parallel.covariances, sequential.covariances, rtol=2e-9, atol=2e-9
    )
    assert jnp.allclose(parallel.gains, sequential.gains, rtol=2e-9, atol=2e-9)

    key = jr.key(19)
    sequential_paths = phx.uq.sample_kalman_smoother_paths(
        key, sequential, sample_shape=(6,), method="sequential"
    )
    parallel_paths = phx.uq.sample_kalman_smoother_paths(
        key, parallel, sample_shape=(2, 3), method="parallel"
    ).reshape(sequential_paths.shape)
    prefix = phx.uq.sample_kalman_smoother_paths(
        key, parallel, sample_shape=(3,), method="parallel"
    )
    assert jnp.allclose(parallel_paths, sequential_paths, rtol=2e-9, atol=2e-9)
    assert jnp.array_equal(prefix, parallel_paths[:3])
    compiled_paths = eqx.filter_jit(
        lambda draw_key, result: phx.uq.sample_kalman_smoother_paths(
            draw_key,
            result,
            sample_shape=(2,),
            method="parallel",
        )
    )(key, parallel)
    assert jnp.allclose(compiled_paths, sequential_paths[:2], rtol=2e-9, atol=2e-9)
    increments = parallel_paths[:, 0, 1:, 0] - parallel_paths[:, 0, :-1, 0]
    assert jnp.any(jnp.abs(increments) > 0.0)


def test_auto_policy_is_conservative_and_execution_is_archived(tmp_path):
    short = phx.uq.kalman_filter(_problem(num_steps=9), method="auto")
    long_problem = _problem(num_steps=80)
    long = phx.uq.kalman_filter(long_problem, method="auto")
    long_sequential = phx.uq.kalman_filter(long_problem, method="sequential")
    assert short.execution_method == "sequential"
    assert long.execution_method == "parallel"
    _assert_filter_equivalent(long_sequential, long, tolerance=2e-8)

    smoother = phx.uq.rts_smoother(long, method="parallel")
    filter_path = phx.uq.export_result(long, tmp_path / "filter.phxresult")
    smoother_path = phx.uq.export_result(smoother, tmp_path / "smoother.phxresult")
    filter_archive = phx.uq.read_result_archive(filter_path)
    smoother_archive = phx.uq.read_result_archive(smoother_path)
    assert filter_archive.metadata["execution_method"] == "parallel"
    assert smoother_archive.metadata["execution_method"] == "parallel"
    assert smoother_archive.metadata["smoother_execution_method"] == "parallel"
    assert np.array_equal(
        filter_archive.array("filtered_means"), np.asarray(long.filtered_means)
    )


def test_parallel_filter_matches_float32_execution():
    with jax.enable_x64(False):
        problem = _problem(num_steps=12)
        sequential = phx.uq.kalman_filter(problem, method="sequential")
        parallel = phx.uq.kalman_filter(problem, method="parallel")
        _assert_filter_equivalent(sequential, parallel, tolerance=4e-5)
        assert parallel.filtered_means.dtype == jnp.float32


def test_exact_state_space_threads_temporal_execution_separately():
    problem = _problem()
    likelihood = phx.uq.exact_state_space_log_likelihood(
        problem, method="kalman", temporal_method="parallel"
    )
    assert likelihood.method == "kalman"
    assert likelihood.temporal_method == "parallel"
    assert likelihood.backend.execution_method == "parallel"


def _markov_marginals(*, node_count=6, padded=False):
    mean = jnp.asarray([0.25, -0.4])
    covariance = jnp.asarray([[0.7, 0.12], [0.12, 0.45]])
    means = [mean]
    covariances = [covariance]
    cross_covariances = []
    for index in range(node_count - 1):
        transition = jnp.asarray(
            [[0.72 + 0.01 * index, -0.23], [0.31, 0.58 - 0.01 * index]]
        )
        offset = jnp.asarray([0.04 * (index + 1), -0.025])
        process_covariance = jnp.asarray([[0.18 + 0.01 * index, 0.025], [0.025, 0.14]])
        cross_covariances.append(covariance @ transition.T)
        mean = transition @ mean + offset
        covariance = transition @ covariance @ transition.T + process_covariance
        means.append(mean)
        covariances.append(covariance)
    node_valid = jnp.ones((node_count,), dtype=bool)
    if padded:
        node_valid = node_valid.at[-2:].set(False)
    return phx.uq.gaussian_markov_moments_from_marginals(
        jnp.stack(means),
        jnp.stack(covariances),
        jnp.stack(cross_covariances),
        node_valid=node_valid,
        moments_id="chain-reference",
    )


def _dense_information_moments(information):
    node_count = information.num_nodes
    state_size = information.state_size
    precision = np.zeros((node_count * state_size, node_count * state_size))
    vector = np.asarray(information.information_vector).reshape((-1,))
    for index in range(node_count):
        block = slice(index * state_size, (index + 1) * state_size)
        precision[block, block] = np.asarray(information.diagonal_precision[index])
    for index in range(node_count - 1):
        left = slice(index * state_size, (index + 1) * state_size)
        right = slice((index + 1) * state_size, (index + 2) * state_size)
        precision[left, right] = np.asarray(information.transition_precision[index])
        precision[right, left] = np.asarray(information.transition_precision[index].T)
    covariance = np.linalg.inv(precision)
    mean = covariance @ vector
    sign, log_determinant = np.linalg.slogdet(precision)
    assert sign > 0
    log_normalizer = (
        0.5 * vector @ mean
        + 0.5 * precision.shape[0] * np.log(2.0 * np.pi)
        - 0.5 * log_determinant
    )
    return mean, covariance, log_normalizer


def test_information_chain_round_trip_matches_dense_and_parallel_algebra():
    reference = _markov_marginals()
    information = phx.uq.gaussian_markov_information_from_moments(reference)
    sequential = phx.uq.gaussian_markov_moments(information, method="sequential")
    parallel = phx.uq.gaussian_markov_moments(information, method="parallel")
    dense_mean, dense_covariance, dense_log_normalizer = _dense_information_moments(
        information
    )
    state_size = information.state_size
    node_count = information.num_nodes
    dense_mean = dense_mean.reshape((node_count, state_size))
    dense_marginals = np.stack(
        tuple(
            dense_covariance[
                index * state_size : (index + 1) * state_size,
                index * state_size : (index + 1) * state_size,
            ]
            for index in range(node_count)
        )
    )
    dense_cross = np.stack(
        tuple(
            dense_covariance[
                index * state_size : (index + 1) * state_size,
                (index + 1) * state_size : (index + 2) * state_size,
            ]
            for index in range(node_count - 1)
        )
    )
    assert sequential.successful
    assert parallel.successful
    assert sequential.execution_method == "sequential"
    assert parallel.execution_method == "parallel"
    assert jnp.allclose(sequential.means, reference.means, rtol=2e-9, atol=2e-9)
    assert jnp.allclose(
        sequential.covariances, reference.covariances, rtol=2e-9, atol=2e-9
    )
    assert jnp.allclose(
        sequential.transition_cross_covariances,
        reference.transition_cross_covariances,
        rtol=2e-9,
        atol=2e-9,
    )
    assert jnp.allclose(parallel.means, dense_mean, rtol=2e-9, atol=2e-9)
    assert jnp.allclose(parallel.covariances, dense_marginals, rtol=2e-9, atol=2e-9)
    assert jnp.allclose(
        parallel.transition_cross_covariances,
        dense_cross,
        rtol=2e-9,
        atol=2e-9,
    )
    assert jnp.allclose(
        parallel.log_normalizer, dense_log_normalizer, rtol=2e-9, atol=2e-9
    )
    assert jnp.allclose(
        parallel.second_moments,
        sequential.second_moments,
        rtol=2e-9,
        atol=2e-9,
    )


def test_information_chain_source_target_orientation_and_padding_are_explicit():
    reference = _markov_marginals(padded=True)
    information = phx.uq.gaussian_markov_information_from_moments(reference)
    recovered = phx.uq.gaussian_markov_moments(information, method="parallel")
    active = reference.node_valid
    assert recovered.successful
    assert jnp.allclose(
        recovered.means[active], reference.means[active], rtol=2e-9, atol=2e-9
    )
    assert jnp.allclose(
        recovered.covariances[active],
        reference.covariances[active],
        rtol=2e-9,
        atol=2e-9,
    )
    assert jnp.allclose(
        recovered.transition_cross_covariances[:3],
        reference.transition_cross_covariances[:3],
        rtol=2e-9,
        atol=2e-9,
    )
    assert jnp.array_equal(recovered.means[~active], jnp.zeros((2, 2)))
    assert jnp.array_equal(
        recovered.transition_cross_covariances[3:], jnp.zeros((2, 2, 2))
    )
    assert not jnp.allclose(
        reference.transition_cross_covariances[0],
        reference.transition_cross_covariances[0].T,
    )


def test_information_chain_statuses_never_repair_invalid_inputs():
    reference = _markov_marginals(node_count=3)
    information = phx.uq.gaussian_markov_information_from_moments(reference)
    nonhermitian = phx.uq.GaussianMarkovInformation(
        information.diagonal_precision.at[0, 0, 1].add(0.2),
        information.transition_precision,
        information.information_vector,
    )
    nonhermitian_result = phx.uq.gaussian_markov_log_normalizer(nonhermitian)
    assert not nonhermitian_result.valid
    assert nonhermitian_result.status == phx.uq.GAUSSIAN_MARKOV_NON_HERMITIAN
    indefinite = phx.uq.GaussianMarkovInformation(
        -jnp.eye(2)[None, ...],
        jnp.zeros((0, 2, 2)),
        jnp.zeros((1, 2)),
    )
    indefinite_result = phx.uq.gaussian_markov_log_normalizer(indefinite)
    assert not indefinite_result.valid
    assert indefinite_result.status == phx.uq.GAUSSIAN_MARKOV_NOT_POSITIVE_DEFINITE
    invalid_mask = phx.uq.GaussianMarkovInformation(
        information.diagonal_precision,
        information.transition_precision,
        information.information_vector,
        node_valid=jnp.asarray([True, False, True]),
    )
    invalid_mask_result = phx.uq.gaussian_markov_log_normalizer(invalid_mask)
    assert not invalid_mask_result.valid
    assert invalid_mask_result.status == phx.uq.GAUSSIAN_MARKOV_INVALID_NODE_MASK


def test_information_chain_is_jittable_and_coherent_samples_recover_lag_moments():
    reference = _markov_marginals(node_count=4)
    information = phx.uq.gaussian_markov_information_from_moments(reference)
    recovered = jax.jit(
        lambda value: phx.uq.gaussian_markov_moments(value, method="parallel")
    )(information)
    draws = phx.uq.sample_gaussian_markov(jr.key(912), recovered, sample_shape=(20_000,))
    empirical_mean = jnp.mean(draws, axis=0)
    centered = draws - empirical_mean
    empirical_covariance = jnp.einsum("sni,snj->nij", centered, centered) / draws.shape[0]
    empirical_cross = (
        jnp.einsum("sni,snj->nij", centered[:, :-1], centered[:, 1:]) / draws.shape[0]
    )
    assert jnp.allclose(empirical_mean, recovered.means, rtol=0.0, atol=2.5e-2)
    assert jnp.allclose(
        empirical_covariance, recovered.covariances, rtol=0.0, atol=3.5e-2
    )
    assert jnp.allclose(
        empirical_cross,
        recovered.transition_cross_covariances,
        rtol=0.0,
        atol=3.5e-2,
    )
    short = phx.uq.sample_gaussian_markov(jr.key(44), recovered, sample_shape=(3,))
    long = phx.uq.sample_gaussian_markov(jr.key(44), recovered, sample_shape=(8,))
    assert jnp.array_equal(short, long[:3])
