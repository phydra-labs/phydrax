import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax._numerics import (
    log_normalize,
    LogWeightedAccumulator,
    weighted_diagnostics,
)


def test_log_normalize_reduces_multiple_axes_per_retained_slice():
    log_weights = jnp.log(
        jnp.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 1.0], [1.0, 2.0]],
            ]
        )
    )
    mask = jnp.asarray(
        [
            [[True, True], [True, False]],
            [[False, False], [False, False]],
        ]
    )

    normalized, log_sum, valid = log_normalize(
        log_weights,
        axes=(1, 2),
        mask=mask,
    )

    assert normalized.shape == log_weights.shape
    assert jnp.allclose(jnp.sum(normalized, axis=(1, 2)), jnp.asarray([1.0, 0.0]))
    assert jnp.allclose(log_sum[0], jnp.log(6.0))
    assert jnp.isneginf(log_sum[1])
    assert jnp.array_equal(valid, jnp.asarray([True, False]))


def test_log_normalize_ignores_masked_nan_but_rejects_included_infinity():
    masked_nan = jnp.asarray([[0.0, jnp.nan], [0.0, 0.0]])
    mask = jnp.asarray([[True, False], [True, True]])
    _, _, masked_valid = log_normalize(masked_nan, axes=1, mask=mask)
    _, _, infinite_valid = log_normalize(
        jnp.asarray([[0.0, jnp.inf], [0.0, 0.0]]),
        axes=1,
    )

    assert jnp.array_equal(masked_valid, jnp.asarray([True, True]))
    assert jnp.array_equal(infinite_valid, jnp.asarray([False, True]))


def test_weighted_accumulator_preserves_batches_and_ignores_zero_weight_nan():
    values = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [jnp.nan, jnp.nan]],
            [[9.0, 10.0], [7.0, 8.0]],
        ]
    )
    log_weights = jnp.asarray(
        [
            [0.0, 0.0],
            [0.0, -jnp.inf],
            [0.0, 0.0],
        ]
    )
    accumulator = LogWeightedAccumulator.from_values(values, log_weights)
    diagnostics = weighted_diagnostics(accumulator, log_weights)

    assert accumulator.normalized_mean.shape == (2, 2)
    assert jnp.allclose(
        accumulator.normalized_mean,
        jnp.asarray([[5.0, 6.0], [5.0, 6.0]]),
    )
    assert jnp.array_equal(diagnostics.finite_count, jnp.asarray([3, 2]))
    assert jnp.all(jnp.isfinite(accumulator.normalized_mean))


def test_weighted_accumulator_merge_handles_an_empty_chunk():
    values = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    log_weights = jnp.asarray([[0.0, -1000.0], [0.0, -1001.0]])
    empty = LogWeightedAccumulator.from_values(
        values,
        log_weights,
        mask=jnp.zeros_like(log_weights, dtype=bool),
    )
    full = LogWeightedAccumulator.from_values(values, log_weights)

    merged = empty.merge(full)

    assert jnp.allclose(merged.normalized_mean, full.normalized_mean)
    assert jnp.allclose(merged.raw_mean, full.raw_mean)


def test_named_weighted_measure_reduces_multiple_sample_axes():
    samples = cx.Field(
        jnp.arange(2 * 2 * 3 * 2, dtype=float).reshape((2, 2, 3, 2)),
        dims=("case", "chain", "draw", "state"),
    )
    log_weights = cx.Field(
        jnp.zeros((2, 2, 3)),
        dims=("case", "chain", "draw"),
    )
    target = phx.integration.weighted(
        samples,
        log_weights,
        sample_axes=("chain", "draw"),
    )

    estimate = phx.integration.integrate(lambda values: values, target)

    assert estimate.value.dims == ("case", "state")
    assert jnp.allclose(estimate.value.data, jnp.mean(samples.data, axis=(1, 2)))
    assert jnp.array_equal(estimate.status, jnp.zeros((2,), dtype=jnp.int32))
    assert jnp.array_equal(estimate.num_evaluations, jnp.asarray([6, 6]))


def test_weighted_measure_reports_empty_mask_per_retained_slice():
    samples = cx.Field(
        jnp.arange(6.0).reshape((2, 3)),
        dims=("case", "particle"),
    )
    log_weights = cx.Field(jnp.zeros((2, 3)), dims=("case", "particle"))
    mask = cx.Field(
        jnp.asarray([[False, False, False], [True, False, True]]),
        dims=("case", "particle"),
    )
    target = phx.integration.weighted(
        samples,
        log_weights,
        mask=mask,
        sample_axes="particle",
    )

    estimate = phx.integration.integrate(lambda values: values, target)

    assert estimate.status[0] == int(phx.integration.IntegrationStatus.NO_VALID_SAMPLES)
    assert estimate.status[1] == int(phx.integration.IntegrationStatus.CONVERGED)
    assert jnp.isnan(estimate.value.data[0])
    assert jnp.allclose(estimate.value.data[1], 4.0)


def test_one_independent_weighted_sample_does_not_claim_uncertainty():
    target = phx.integration.weighted(
        jnp.asarray([2.0]),
        jnp.asarray([0.0]),
        independent=True,
    )

    estimate = phx.integration.integrate(lambda values: values, target)

    assert estimate.successful
    assert estimate.error_estimate is None
    assert estimate.diagnostics.standard_error is None
    assert estimate.diagnostics.normalizer_standard_error is None


def test_external_measures_reject_plans_and_random_keys():
    target = phx.integration.weighted(jnp.ones((2,)), jnp.zeros((2,)))

    with pytest.raises(TypeError, match="do not take an integration plan"):
        phx.integration.integrate(1.0, target, phx.integration.FixedQuadraturePlan(3))
    with pytest.raises(ValueError, match="do not consume a random key"):
        phx.integration.integrate(1.0, target, key=jr.key(0))


def test_discrete_measure_preserves_retained_axes_and_fixed_diagnostics():
    points = cx.Field(
        jnp.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]),
        dims=("case", "node"),
    )
    weights = cx.Field(
        jnp.asarray([[1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]),
        dims=("case", "node"),
    )
    target = phx.integration.discrete(points, weights, axes="node")

    estimate = phx.integration.integrate(lambda values: values, target)

    assert estimate.value.dims == ("case",)
    assert jnp.allclose(estimate.value.data, jnp.asarray([4.0, 9.0]))
    assert jnp.array_equal(estimate.num_evaluations, jnp.asarray([3, 3]))
    assert isinstance(estimate.diagnostics, phx.integration.FixedQuadratureDiagnostics)
    assert estimate.error_estimate is None


def test_separable_discrete_measure_avoids_a_second_weight_convention():
    x = cx.Field(jnp.asarray([0.0, 1.0]), dims=("x",))
    y = cx.Field(jnp.asarray([0.0, 2.0, 4.0]), dims=("y",))
    points = cx.Field(
        x.data[:, None] + y.data[None, :],
        dims=("x", "y"),
    )
    target = phx.integration.discrete(
        points,
        {
            "x": cx.Field(jnp.asarray([0.5, 0.5]), dims=("x",)),
            "y": cx.Field(jnp.asarray([1.0, 2.0, 1.0]), dims=("y",)),
        },
        axes=("x", "y"),
    )

    estimate = phx.integration.integrate(lambda values: values, target)

    assert estimate.successful
    assert estimate.value.dims == ()
    assert jnp.allclose(estimate.value.data, 10.0)


def test_weighted_measure_preserves_design_metadata_and_support_status():
    samples = cx.Field(
        jnp.arange(6.0).reshape((2, 3)),
        dims=("case", "particle"),
    )
    log_weights = cx.Field(jnp.zeros((2, 3)), dims=("case", "particle"))
    ancestry = cx.Field(
        jnp.asarray([[0, 0, 1], [2, 1, 0]]),
        dims=("case", "particle"),
    )
    target = phx.integration.weighted(
        samples,
        log_weights,
        sample_axes="particle",
        support_valid=jnp.asarray([True, False]),
        stratum_ids=jnp.asarray([0, 0, 1]),
        pair_ids=jnp.asarray([0, 0, 1]),
        replicate_ids=jnp.asarray([0, 1, 2]),
        ancestry=ancestry,
    )

    estimate = phx.integration.integrate(lambda values: values, target)

    assert estimate.status[0] == int(phx.integration.IntegrationStatus.CONVERGED)
    assert estimate.status[1] == int(
        phx.integration.IntegrationStatus.PROPOSAL_SUPPORT_FAILURE
    )
    assert jnp.array_equal(estimate.diagnostics.stratum_ids, jnp.asarray([0, 0, 1]))
    assert jnp.array_equal(estimate.diagnostics.pair_ids, jnp.asarray([0, 0, 1]))
    assert jnp.array_equal(
        estimate.diagnostics.replicate_ids,
        jnp.asarray([0, 1, 2]),
    )
    assert jnp.array_equal(estimate.diagnostics.ancestry_ids, ancestry.data)
