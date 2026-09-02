#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.stochastic.path_sampling import (
    autocorrelation_uncertainty,
    block_mean_uncertainty,
    CommittorFitPlan,
    cross_evaluate_path_potentials,
    DeterministicPathAction,
    DynamicsKernelCapabilities,
    DynamicsStep,
    estimate_reactive_flux,
    factorize_tis_rate,
    FirstPassagePathEnsemble,
    fit_committor,
    FixedPathEnsemble,
    FunctionalDynamicsKernel,
    IdentityShootingModifier,
    integrated_autocorrelation_time,
    InterfacePathEnsemble,
    make_incremental_path_target,
    moving_block_bootstrap_uncertainty,
    NormalizedStochasticPathAction,
    path_fep_work,
    PATH_PROPAGATION_KERNEL_FAILURE,
    PATH_PROPAGATION_NONFINITE,
    PATH_PROPAGATION_OVERFLOW,
    PathBuffer,
    predict_committor,
    prepare_tps,
    propose_one_way_shooting,
    propose_path_reversal,
    propose_path_shift,
    propose_replica_exchange,
    propose_two_way_shooting,
    ReducedPathPotential,
    StateRegionPlan,
    SurrogatePathAction,
    TPSPlan,
    UniformShootingSelector,
    WeightedShootingSelector,
)


def _deterministic_kernel(*, fail: bool = False) -> FunctionalDynamicsKernel:
    def step(key, state, direction):
        del key
        value = state + direction.astype(state.dtype)
        valid = jnp.asarray(not fail)
        status = jnp.where(valid, 0, PATH_PROPAGATION_KERNEL_FAILURE)
        return DynamicsStep(value, jnp.asarray(0.0), valid, status)

    def transition(source, destination, direction):
        expected = source + direction.astype(source.dtype)
        return jnp.where(jnp.all(jnp.abs(destination - expected) < 1.0e-6), 0.0, -jnp.inf)

    capabilities = DynamicsKernelCapabilities(
        stochastic=False,
        reversible=True,
        supports_backward=True,
        normalized_transition_density=False,
    )
    return FunctionalDynamicsKernel(
        step,
        transition,
        capabilities,
        time_step=1.0,
        kernel_id="toy-deterministic-failing" if fail else "toy-deterministic",
    )


def _gaussian_kernel(scale: float = 0.5) -> FunctionalDynamicsKernel:
    def log_density(source, destination, direction):
        residual = (destination - source - 0.25 * direction.astype(source.dtype)) / scale
        return -0.5 * jnp.sum(residual**2) - source.size * jnp.log(
            scale * jnp.sqrt(2.0 * jnp.pi)
        )

    def step(key, state, direction):
        proposed = (
            state
            + 0.25 * direction.astype(state.dtype)
            + scale * jax.random.normal(key, state.shape)
        )
        return DynamicsStep(
            proposed,
            log_density(state, proposed, direction),
            jnp.asarray(True),
            jnp.asarray(0, jnp.int32),
        )

    capabilities = DynamicsKernelCapabilities(
        stochastic=True,
        reversible=False,
        supports_backward=True,
        normalized_transition_density=True,
    )
    return FunctionalDynamicsKernel(
        step,
        log_density,
        capabilities,
        time_step=1.0,
        kernel_id="toy-normalized-gaussian",
    )


def _path(capacity: int = 7) -> PathBuffer:
    values = jnp.arange(5.0).reshape((5, 1))
    return PathBuffer.from_trajectory(values, jnp.arange(5.0), capacity=capacity)


def test_state_regions_are_boolean_and_half_open() -> None:
    left = StateRegionPlan.half_open(jnp.asarray([0.0]), jnp.asarray([1.0]))
    right = StateRegionPlan.half_open(jnp.asarray([1.0]), jnp.asarray([2.0]))
    values = jnp.asarray([[-0.1], [0.0], [0.999], [1.0], [1.999], [2.0]])
    np.testing.assert_array_equal(
        left.contains(values), [False, True, True, False, False, False]
    )
    np.testing.assert_array_equal(
        (left | right).contains(values), [False, True, True, True, True, False]
    )
    np.testing.assert_array_equal((left & ~right).contains(values), left.contains(values))
    np.testing.assert_array_equal(
        (left ^ right).contains(values), (left | right).contains(values)
    )


def test_path_buffer_mask_lineage_and_time_reversal_are_exact() -> None:
    path = _path()
    np.testing.assert_array_equal(path.mask, [True, True, True, True, True, False, False])
    np.testing.assert_array_equal(path.lineage, [0, 1, 2, 3, 4, -1, -1])
    reversed_path = path.time_reversed()
    np.testing.assert_allclose(reversed_path.positions[:5, 0], [4, 3, 2, 1, 0])
    np.testing.assert_allclose(reversed_path.times[:5], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(reversed_path.lineage[:5], [4, 3, 2, 1, 0])
    assert int(reversed_path.direction) == -1
    np.testing.assert_allclose(reversed_path.time_reversed().positions, path.positions)
    irregular = PathBuffer.from_trajectory(
        jnp.arange(5.0).reshape((5, 1)),
        jnp.asarray([0.0, 0.1, 0.4, 1.25, 2.0]),
        capacity=7,
    )
    restored = irregular.time_reversed().time_reversed()
    np.testing.assert_array_equal(restored.positions, irregular.positions)
    np.testing.assert_array_equal(restored.times, irregular.times)
    invalid_padding = PathBuffer(
        path.positions.at[-1].set(7.0),
        path.times,
        path.length,
        path.mask,
        path.direction,
        path.lineage,
    )
    assert not bool(invalid_padding.valid())


def test_path_reversal_move_is_symmetric_for_reversible_dynamics() -> None:
    kernel = _deterministic_kernel()
    result = propose_path_reversal(
        FixedPathEnsemble(5),
        DeterministicPathAction(kernel),
        kernel,
        _path(),
        jax.random.key(50),
    )
    assert bool(result.accepted)
    np.testing.assert_allclose(result.evaluation.log_acceptance_ratio, 0.0)
    assert int(result.committed.direction) == -1
    shot = propose_two_way_shooting(
        FixedPathEnsemble(5),
        DeterministicPathAction(kernel),
        kernel,
        UniformShootingSelector(),
        IdentityShootingModifier(),
        result.committed,
        jax.random.key(54),
    )
    assert bool(shot.evaluation.proposal_valid)
    np.testing.assert_allclose(shot.proposed.positions, result.committed.positions)


def test_asymmetric_selector_separates_length_normalization() -> None:
    short = PathBuffer.from_trajectory(
        jnp.arange(4.0).reshape((4, 1)), jnp.arange(4.0), capacity=7
    )
    long = _path()
    uniform = UniformShootingSelector(endpoint_margin=1)
    log_ratio = uniform.log_probability(short, jnp.asarray(1)) - uniform.log_probability(
        long, jnp.asarray(1)
    )
    np.testing.assert_allclose(log_ratio, jnp.log(3.0 / 2.0))
    weighted = WeightedShootingSelector(
        lambda points: 0.5 * points[:, 0],
        endpoint_margin=1,
        selector_id="biased-selector",
    )
    forward = weighted.log_probability(long, jnp.asarray(1))
    reverse = weighted.log_probability(long, jnp.asarray(3))
    assert float(reverse - forward) > 0.0
    invalid = WeightedShootingSelector(
        lambda points: jnp.where(jnp.arange(points.shape[0]) == 1, jnp.inf, 0.0),
        endpoint_margin=1,
        selector_id="invalid-infinite-selector",
    ).select(jax.random.key(55), long)
    assert not bool(invalid.valid)
    assert int(invalid.eligible_count) == 3


def test_first_passage_ensemble_rejects_a_premature_target_visit() -> None:
    initial = StateRegionPlan.half_open(jnp.asarray([-0.5]), jnp.asarray([0.5]))
    final = StateRegionPlan.half_open(jnp.asarray([1.5]), jnp.asarray([2.5]))
    ensemble = FirstPassagePathEnsemble(initial, final)
    valid = PathBuffer.from_trajectory(
        jnp.asarray([[0.0], [1.0], [2.0]]),
        jnp.arange(3.0),
        capacity=5,
    )
    premature = PathBuffer.from_trajectory(
        jnp.asarray([[0.0], [2.0], [1.0], [2.0]]),
        jnp.arange(4.0),
        capacity=5,
    )
    revisiting_initial = PathBuffer.from_trajectory(
        jnp.asarray([[0.0], [0.0], [1.0], [2.0]]),
        jnp.arange(4.0),
        capacity=5,
    )
    assert bool(ensemble.contains(valid))
    assert bool(ensemble.contains(revisiting_initial))
    assert not bool(ensemble.contains(premature))


def test_interface_ensemble_matches_first_terminal_hit_support() -> None:
    initial = StateRegionPlan.half_open(jnp.asarray([-0.5]), jnp.asarray([0.5]))
    final = StateRegionPlan.half_open(jnp.asarray([1.5]), jnp.asarray([2.5]))
    path = PathBuffer.from_trajectory(
        jnp.asarray([[0.0], [1.0], [0.0], [2.0]]),
        jnp.arange(4.0),
        capacity=6,
    )
    interface = InterfacePathEnsemble(
        initial,
        final,
        lambda states: states[..., 0],
        0.75,
        coordinate_id="support-coordinate",
    )
    assert not bool(interface.contains(path))
    nan_interface = InterfacePathEnsemble(
        initial,
        final,
        lambda states: jnp.where(states[..., 0] == 1.0, jnp.nan, states[..., 0]),
        0.75,
        coordinate_id="nan-coordinate",
    )
    direct = PathBuffer.from_trajectory(
        jnp.asarray([[0.0], [1.0], [2.0]]),
        jnp.arange(3.0),
        capacity=6,
    )
    assert not bool(nan_interface.contains(direct))


def test_deterministic_two_way_shooting_obeys_detailed_balance() -> None:
    kernel = _deterministic_kernel()
    action = DeterministicPathAction(kernel)
    ensemble = FixedPathEnsemble(5)
    result = propose_two_way_shooting(
        ensemble,
        action,
        kernel,
        UniformShootingSelector(),
        IdentityShootingModifier(),
        _path(),
        jax.random.key(1),
    )
    assert bool(result.evaluation.proposal_valid)
    np.testing.assert_allclose(result.evaluation.log_acceptance_ratio, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(result.committed.positions, result.current.positions)


def test_variable_length_shooting_reports_separate_length_correction() -> None:
    def step(key, state, direction):
        del key
        return DynamicsStep(
            state + direction.astype(state.dtype),
            jnp.asarray(0.0),
            jnp.asarray(True),
            jnp.asarray(0, jnp.int32),
        )

    kernel = FunctionalDynamicsKernel(
        step,
        lambda source, destination, direction: jnp.asarray(0.0),
        DynamicsKernelCapabilities(
            stochastic=False,
            reversible=True,
            supports_backward=True,
            normalized_transition_density=False,
        ),
        time_step=1.0,
        kernel_id="variable-length-toy",
    )
    initial = StateRegionPlan.half_open(jnp.asarray([-0.5]), jnp.asarray([0.5]))
    final = StateRegionPlan.half_open(jnp.asarray([1.5]), jnp.asarray([2.5]))
    ensemble = FirstPassagePathEnsemble(initial, final)
    path = PathBuffer.from_trajectory(
        jnp.asarray([[0.0], [1.0], [1.0], [2.0]]),
        jnp.arange(4.0),
        capacity=5,
    )
    results = tuple(
        propose_one_way_shooting(
            ensemble,
            DeterministicPathAction(kernel),
            kernel,
            UniformShootingSelector(),
            IdentityShootingModifier(),
            path,
            jax.random.key(index),
        )
        for index in range(12)
    )
    changed = tuple(
        result for result in results if int(result.proposed.length) != int(path.length)
    )
    assert changed
    for result in changed:
        np.testing.assert_allclose(result.evaluation.length_log_ratio, jnp.log(2.0))
        np.testing.assert_allclose(result.evaluation.selector_log_ratio, 0.0)


def test_normalized_gaussian_action_and_propagation_have_exact_mh_sum() -> None:
    kernel = _gaussian_kernel()
    action = NormalizedStochasticPathAction(
        kernel,
        lambda state: -0.5 * jnp.sum(state**2) - 0.5 * state.size * jnp.log(2.0 * jnp.pi),
        initial_density_id="standard-normal",
    )
    path = PathBuffer.from_trajectory(
        jnp.asarray([[0.0], [0.1], [0.4], [0.7], [1.0]]),
        jnp.arange(5.0),
        capacity=7,
    )
    result = propose_one_way_shooting(
        FixedPathEnsemble(5),
        action,
        kernel,
        UniformShootingSelector(),
        IdentityShootingModifier(),
        path,
        jax.random.key(4),
    )
    evidence = result.evaluation
    summed = (
        evidence.target_log_ratio
        + evidence.selector_log_ratio
        + evidence.modifier_log_ratio
        + evidence.propagation_log_ratio
        + evidence.length_log_ratio
        + evidence.exchange_log_ratio
    )
    np.testing.assert_allclose(evidence.log_acceptance_ratio, summed, rtol=1.0e-6)
    assert bool(evidence.proposal_valid)


def test_failed_propagation_rejects_once_without_retry() -> None:
    kernel = _deterministic_kernel(fail=True)
    result = propose_one_way_shooting(
        FixedPathEnsemble(5),
        DeterministicPathAction(_deterministic_kernel()),
        kernel,
        UniformShootingSelector(),
        IdentityShootingModifier(),
        _path(),
        jax.random.key(5),
    )
    assert not bool(result.accepted)
    assert not bool(result.evaluation.propagation_valid)
    assert int(result.evaluation.propagation_status) == PATH_PROPAGATION_KERNEL_FAILURE
    np.testing.assert_allclose(result.committed.positions, result.current.positions)


def test_nonfinite_propagation_is_classified_and_rejected() -> None:
    def step(key, state, direction):
        del key, direction
        return DynamicsStep(
            jnp.full_like(state, jnp.nan),
            jnp.asarray(jnp.nan),
            jnp.asarray(True),
            jnp.asarray(0, jnp.int32),
        )

    def transition(source, destination, direction):
        del source, destination, direction
        return jnp.asarray(0.0)

    good = _deterministic_kernel()
    kernel = FunctionalDynamicsKernel(
        step,
        transition,
        DynamicsKernelCapabilities(
            stochastic=False,
            reversible=True,
            supports_backward=True,
            normalized_transition_density=False,
        ),
        time_step=1.0,
        kernel_id="nonfinite-propagation",
    )
    result = propose_one_way_shooting(
        FixedPathEnsemble(5),
        DeterministicPathAction(good),
        kernel,
        UniformShootingSelector(),
        IdentityShootingModifier(),
        _path(),
        jax.random.key(52),
    )
    assert not bool(result.accepted)
    assert int(result.evaluation.propagation_status) == PATH_PROPAGATION_NONFINITE


def test_capacity_overflow_rejects_without_extending_path_shape() -> None:
    def step(key, state, direction):
        del key
        return DynamicsStep(
            state + direction.astype(state.dtype),
            jnp.asarray(0.0),
            jnp.asarray(True),
            jnp.asarray(0, jnp.int32),
        )

    def transition(source, destination, direction):
        del source, destination, direction
        return jnp.asarray(0.0)

    kernel = FunctionalDynamicsKernel(
        step,
        transition,
        DynamicsKernelCapabilities(
            stochastic=False,
            reversible=True,
            supports_backward=True,
            normalized_transition_density=False,
        ),
        time_step=1.0,
        kernel_id="overflow-propagation",
    )
    initial = StateRegionPlan.half_open(jnp.asarray([-0.5]), jnp.asarray([0.5]))
    final = StateRegionPlan.half_open(jnp.asarray([99.5]), jnp.asarray([100.5]))
    path = PathBuffer.from_trajectory(
        jnp.asarray([[0.0], [100.0]]),
        jnp.arange(2.0),
        capacity=4,
    )
    result = propose_one_way_shooting(
        FirstPassagePathEnsemble(initial, final),
        DeterministicPathAction(kernel),
        kernel,
        UniformShootingSelector(endpoint_margin=0),
        IdentityShootingModifier(),
        path,
        jax.random.key(53),
    )
    assert not bool(result.accepted)
    assert int(result.evaluation.propagation_status) == PATH_PROPAGATION_OVERFLOW
    assert result.proposed.positions.shape == path.positions.shape


def test_fixed_path_shifting_preserves_length_and_dynamics() -> None:
    kernel = _deterministic_kernel()
    result = propose_path_shift(
        FixedPathEnsemble(5),
        DeterministicPathAction(kernel),
        kernel,
        _path(),
        jax.random.key(51),
        maximum_shift=2,
    )
    assert bool(result.evaluation.proposal_valid)
    assert int(result.proposed.length) == 5
    np.testing.assert_allclose(jnp.diff(result.proposed.positions[:5, 0]), 1.0)


def test_regrowth_requires_fixed_uniform_time_contract() -> None:
    kernel = _deterministic_kernel()
    plan = TPSPlan(
        FixedPathEnsemble(5),
        kernel,
        DeterministicPathAction(kernel),
    )
    nonuniform = PathBuffer.from_trajectory(
        jnp.arange(5.0).reshape((5, 1)),
        jnp.asarray([0.0, 1.0, 2.5, 3.5, 4.5]),
        capacity=7,
    )
    with pytest.raises(ValueError, match="kernel.time_step"):
        prepare_tps(plan, nonuniform)

    variable_step = FunctionalDynamicsKernel(
        kernel.step_fn,
        kernel.transition_log_density_fn,
        DynamicsKernelCapabilities(
            stochastic=False,
            reversible=True,
            supports_backward=True,
            normalized_transition_density=False,
            fixed_step=False,
        ),
        time_step=1.0,
        kernel_id="variable-step-contract",
    )
    with pytest.raises(ValueError, match="fixed-step"):
        TPSPlan(
            FixedPathEnsemble(5),
            variable_step,
            DeterministicPathAction(variable_step),
        )


def test_path_target_uses_incremental_mh_contract() -> None:
    kernel = _deterministic_kernel()
    path = _path()
    target = make_incremental_path_target(
        FixedPathEnsemble(5),
        DeterministicPathAction(kernel),
    )
    state = target.initialize(path)
    proposal = target.propose(state, path, ())
    np.testing.assert_allclose(proposal.log_ratio, 0.0)
    assert bool(proposal.valid)


def test_tis_rate_factorization_is_flux_times_crossing_factors() -> None:
    flux = estimate_reactive_flux(jnp.asarray([True, False, True]), 4.0)
    result = factorize_tis_rate(flux, jnp.asarray([0.5, 0.25, 0.2]))
    np.testing.assert_allclose(result.rate, 0.0125)
    np.testing.assert_allclose(jnp.exp(result.log_rate), result.rate)
    assert bool(result.valid)
    zero = factorize_tis_rate(flux, jnp.asarray([0.5, 0.0]))
    np.testing.assert_allclose(zero.rate, 0.0)
    assert bool(jnp.isneginf(zero.log_rate))
    np.testing.assert_allclose(jnp.exp(zero.log_rate), zero.rate)
    with pytest.raises(ValueError, match="binary indicators"):
        estimate_reactive_flux(jnp.asarray([1.0, jnp.nan]), 2.0)


def test_committor_fit_preserves_probability_ordering() -> None:
    features = jnp.linspace(-2.0, 2.0, 64).reshape((-1, 1))
    outcomes = (features[:, 0] > 0.0).astype(features.dtype)
    result = fit_committor(
        CommittorFitPlan(
            1,
            maximum_iterations=2000,
            learning_rate=0.1,
            l2_regularization=1.0e-4,
        ),
        features,
        outcomes,
    )
    probability = predict_committor(result, jnp.asarray([[-1.0], [1.0]]))
    assert bool(result.valid)
    assert float(probability[0]) < 0.5 < float(probability[1])
    assert bool(result.converged) == bool(result.gradient_norm <= 1.0e-7)


def test_replica_exchange_is_symmetric_for_equal_targets() -> None:
    kernel = _deterministic_kernel()
    action = DeterministicPathAction(kernel)
    ensemble = FixedPathEnsemble(5)
    path = _path()
    exchange = propose_replica_exchange(
        ensemble, ensemble, action, path, path, jax.random.key(6)
    )
    reverse = propose_replica_exchange(
        ensemble, ensemble, action, exchange.left, exchange.right, jax.random.key(7)
    )
    np.testing.assert_allclose(exchange.evaluation.exchange_log_ratio, 0.0)
    np.testing.assert_allclose(reverse.evaluation.exchange_log_ratio, 0.0)
    assert bool(exchange.accepted)


def test_correlated_uncertainty_exceeds_naive_independent_error() -> None:
    innovations = jax.random.normal(jax.random.key(8), (512,))
    correlated = jax.lax.scan(
        lambda value, noise: (0.95 * value + noise, 0.95 * value + noise),
        0.0,
        innovations,
    )[1]
    block = block_mean_uncertainty(correlated, block_size=32)
    autocorrelation = autocorrelation_uncertainty(correlated, maximum_lag=64)
    bootstrap = moving_block_bootstrap_uncertainty(
        jax.random.key(9),
        correlated,
        block_length=32,
        resamples=128,
    )
    naive = jnp.std(correlated, ddof=1) / jnp.sqrt(correlated.size)
    assert float(block.standard_error) > float(naive)
    assert float(autocorrelation.standard_error) > float(naive)
    assert float(autocorrelation.integrated_autocorrelation_time) > 1.0
    assert float(bootstrap.standard_error) > float(naive)
    with pytest.raises(ValueError, match="finite samples"):
        integrated_autocorrelation_time(
            jnp.asarray([0.0, jnp.nan, 1.0]),
            maximum_lag=1,
        )


def test_path_reweighting_fails_closed_for_surrogates_and_crosses_normalized_actions() -> (
    None
):
    kernel = _gaussian_kernel()
    action = NormalizedStochasticPathAction(
        kernel,
        lambda state: -0.5 * jnp.sum(state**2) - 0.5 * jnp.log(2.0 * jnp.pi),
        initial_density_id="normal-a",
    )
    ensemble = FixedPathEnsemble(5)
    potential = ReducedPathPotential(ensemble, action)
    evaluation = cross_evaluate_path_potentials(
        (potential, potential),
        (_path(), _path()),
        jnp.asarray([0, 1]),
    )
    assert evaluation.samples.values.shape == (2, 2)
    np.testing.assert_allclose(path_fep_work(evaluation, 0, 1), jnp.asarray([0.0]))
    with pytest.raises(ValueError, match="normalized stochastic"):
        ReducedPathPotential(
            ensemble,
            SurrogatePathAction(lambda path: jnp.asarray(0.0), action_id="surrogate"),
        )
