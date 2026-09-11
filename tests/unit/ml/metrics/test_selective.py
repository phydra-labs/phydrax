#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

from phydrax.ml import metrics


def _points(result, value):
    return value[result.point_mask]


def test_weighted_selective_curve_uses_attainable_tie_endpoints_and_mass_aurc():
    loss = jnp.array([1.0, 4.0, 2.0, jnp.nan])
    score = jnp.array([3.0, 1.0, 2.0, jnp.nan])
    weight = jnp.array([1.0, 2.0, 1.0, 20.0])
    mask = jnp.array([True, True, True, False])

    result = metrics.selective_risk_curve(loss, score, sample_weight=weight, mask=mask)
    oracle = metrics.selective_risk_curve(loss, loss, sample_weight=weight, mask=mask)

    assert bool(result.valid)
    assert int(result.status) == metrics.METRIC_SUCCESS
    assert jnp.allclose(
        _points(result, result.score_threshold), jnp.array([1.0, 2.0, 3.0])
    )
    assert jnp.allclose(
        _points(result, result.retained_weight), jnp.array([2.0, 3.0, 4.0])
    )
    assert jnp.allclose(_points(result, result.coverage), jnp.array([0.5, 0.75, 1.0]))
    assert jnp.allclose(
        _points(result, result.retained_risk),
        jnp.array([4.0, 10.0 / 3.0, 11.0 / 4.0]),
    )
    assert jnp.allclose(result.aurc, 169.0 / 48.0)
    assert jnp.allclose(result.effective_weight, 4.0)
    assert jnp.allclose(oracle.aurc, 2.0)
    assert float(oracle.aurc) < float(result.aurc)


def test_complete_score_ties_are_indivisible_and_permutation_invariant():
    loss = jnp.array([1.0, 4.0, 2.0, 7.0])
    score = jnp.array([1.0, 1.0, 2.0, 2.0])
    weight = jnp.array([1.0, 2.0, 1.0, 3.0])
    permutation = jnp.array([3, 1, 0, 2])

    original = metrics.selective_risk_curve(loss, score, sample_weight=weight)
    permuted = metrics.selective_risk_curve(
        loss[permutation], score[permutation], sample_weight=weight[permutation]
    )

    assert int(jnp.sum(original.point_mask)) == 2
    assert jnp.allclose(
        _points(original, original.score_threshold), jnp.array([1.0, 2.0])
    )
    assert jnp.allclose(_points(original, original.coverage), jnp.array([3.0 / 7.0, 1.0]))
    assert jnp.allclose(original.aurc, permuted.aurc)
    assert jnp.allclose(
        _points(original, original.retained_risk),
        _points(permuted, permuted.retained_risk),
    )
    assert jnp.allclose(
        _points(original, original.coverage), _points(permuted, permuted.coverage)
    )

    complete_tie = metrics.selective_risk_curve(loss, jnp.ones_like(score))
    assert int(jnp.sum(complete_tie.point_mask)) == 1
    assert jnp.allclose(_points(complete_tie, complete_tie.coverage), 1.0)
    assert jnp.allclose(complete_tie.aurc, jnp.mean(loss))


def test_selective_and_spearman_masks_follow_metric_status_semantics():
    loss = jnp.array([1.0, 2.0, jnp.nan])
    score = jnp.array([1.0, 2.0, jnp.nan])
    mask = jnp.array([True, True, False])

    masked = metrics.selective_risk_curve(loss, score, mask=mask)
    invalid = metrics.selective_risk_curve(loss, score)
    empty = metrics.selective_risk_curve(loss, score, mask=jnp.zeros(3, dtype=bool))
    bad_weight = metrics.selective_risk_curve(
        jnp.ones(3), jnp.arange(3.0), sample_weight=jnp.array([1.0, -1.0, 1.0])
    )
    constant_rank = metrics.spearman_rank_correlation(jnp.ones(3), jnp.arange(3.0))
    masked_rank = metrics.spearman_rank_correlation(loss, score, mask=mask)
    invalid_rank = metrics.spearman_rank_correlation(loss, score)

    assert int(masked.status) == metrics.METRIC_SUCCESS
    assert int(invalid.status) == metrics.METRIC_INVALID_INPUT
    assert int(empty.status) == metrics.METRIC_EMPTY
    assert int(bad_weight.status) == metrics.METRIC_INVALID_INPUT
    assert int(constant_rank.status) == metrics.METRIC_ZERO_DENOMINATOR
    assert int(masked_rank.status) == metrics.METRIC_SUCCESS
    assert int(invalid_rank.status) == metrics.METRIC_INVALID_INPUT
    assert bool(masked.valid)
    assert not bool(invalid.valid)
    assert not bool(empty.valid)
    assert not bool(constant_rank.valid)
    assert bool(masked_rank.valid)
    assert not bool(invalid_rank.valid)
    negative = metrics.selective_risk_curve(jnp.array([1.0, -1.0]), jnp.array([0.0, 1.0]))
    assert int(negative.status) == metrics.METRIC_INVALID_INPUT
    for values in (
        masked.score_threshold,
        masked.coverage,
        masked.retained_risk,
        masked.retained_weight,
    ):
        assert jnp.all(jnp.isnan(values[~masked.point_mask]))


def test_weighted_spearman_midranks_match_literal_integer_replication():
    first = jnp.array([0.0, 1.0, 2.0])
    second = jnp.array([2.0, 0.0, 1.0])
    frequency = jnp.array([1.0, 2.0, 3.0])
    replicated_first = jnp.array([0.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    replicated_second = jnp.array([2.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    weighted = metrics.spearman_rank_correlation(first, second, sample_weight=frequency)
    replicated = metrics.spearman_rank_correlation(replicated_first, replicated_second)

    assert bool(weighted.valid)
    assert jnp.allclose(weighted.value, replicated.value)
    assert jnp.allclose(weighted.effective_weight, 6.0)


def test_selective_metrics_are_jittable_case_batched_and_hard_ranked():
    loss = jnp.array([[1.0, 4.0, 2.0], [3.0, 1.0, 5.0]])
    score = jnp.array([[3.0, 1.0, 2.0], [2.0, 3.0, 1.0]])

    direct = metrics.selective_risk_curve(loss, score)
    mapped = jax.vmap(
        lambda case_loss, case_score: (
            metrics.selective_risk_curve(case_loss, case_score).aurc
        )
    )(loss, score)
    compiled = jax.jit(metrics.selective_risk_curve)(loss, score)
    score_gradient = jax.grad(
        lambda values: metrics.selective_risk_curve(loss[0], values).aurc
    )(score[0])
    loss_gradient = jax.grad(
        lambda values: metrics.selective_risk_curve(values, score[0]).aurc
    )(loss[0])
    rank_gradient = jax.grad(
        lambda values: metrics.spearman_rank_correlation(loss[0], values).value
    )(score[0])

    assert isinstance(compiled, metrics.SelectiveRiskCurveResult)
    assert jnp.allclose(direct.aurc, mapped)
    assert jnp.allclose(compiled.aurc, direct.aurc)
    assert jnp.allclose(score_gradient, 0.0)
    assert jnp.allclose(rank_gradient, 0.0)
    assert jnp.all(jnp.isfinite(loss_gradient))
    assert jnp.any(jnp.abs(loss_gradient) > 0.0)


def test_identical_paired_losses_have_exact_zero_effect_and_bounds():
    plan = metrics.PairedLossComparisonPlan(
        confidence=0.8, resamples=32, noninferiority_margin=0.0
    )
    loss = jnp.array([1.0, 4.0, 2.0, 3.0])

    result = metrics.compare_paired_losses(loss, loss, key=jax.random.key(4), plan=plan)

    assert bool(result.valid)
    assert int(result.status) == metrics.METRIC_SUCCESS
    assert jnp.allclose(result.loss_difference, 0.0)
    assert jnp.allclose(result.effect, 0.0)
    assert jnp.allclose(result.bootstrap_effects, 0.0)
    assert jnp.allclose(result.interval_lower, 0.0)
    assert jnp.allclose(result.interval_upper, 0.0)
    assert jnp.allclose(result.noninferiority_upper_bound, 0.0)
    assert bool(result.noninferior)


def test_grouped_paired_bootstrap_resamples_whole_groups_and_replays():
    reference = jnp.array([1_000.0, 2_000.0, 3_000.0, 4_000.0])
    candidate = reference + jnp.array([0.0, 2.0, 3.0, 5.0])
    weight = jnp.array([1.0, 3.0, 2.0, 2.0])
    groups = jnp.array([10, 10, 20, 20])
    plan = metrics.PairedLossComparisonPlan(confidence=0.8, resamples=64)
    key = jax.random.key(17)

    first = metrics.compare_paired_losses(
        reference,
        candidate,
        key=key,
        plan=plan,
        sample_weight=weight,
        groups=groups,
    )
    replay = metrics.compare_paired_losses(
        reference,
        candidate,
        key=key,
        plan=plan,
        sample_weight=weight,
        groups=groups,
    )

    possible_group_draw_effects = jnp.array([1.5, 2.75, 4.0])
    matches_group_draw = jnp.any(
        jnp.isclose(
            first.bootstrap_effects[:, None], possible_group_draw_effects[None, :]
        ),
        axis=-1,
    )
    assert first.grouped is True
    assert int(first.independent_unit_count) == 2
    assert jnp.all(matches_group_draw)
    assert jnp.array_equal(first.bootstrap_effects, replay.bootstrap_effects)
    assert jnp.allclose(first.effect, 2.75)


def test_paired_central_and_one_sided_quantiles_are_distinct_contracts():
    reference = jnp.zeros(8)
    candidate = jnp.arange(8.0)
    plan = metrics.PairedLossComparisonPlan(
        confidence=0.8, resamples=257, noninferiority_margin=3.0
    )
    result = metrics.compare_paired_losses(
        reference, candidate, key=jax.random.key(23), plan=plan
    )

    assert jnp.allclose(
        result.interval_lower, jnp.quantile(result.bootstrap_effects, 0.1)
    )
    assert jnp.allclose(
        result.interval_upper, jnp.quantile(result.bootstrap_effects, 0.9)
    )
    assert jnp.allclose(
        result.noninferiority_upper_bound,
        jnp.quantile(result.bootstrap_effects, 0.8),
    )
    assert not jnp.isclose(result.noninferiority_upper_bound, result.interval_upper)
    assert bool(result.noninferior) == bool(
        result.noninferiority_upper_bound <= plan.noninferiority_margin
    )


def test_paired_comparison_reports_independent_unit_insufficiency():
    plan = metrics.PairedLossComparisonPlan(resamples=8)
    one_case = metrics.compare_paired_losses(
        jnp.array([1.0]),
        jnp.array([0.5]),
        key=jax.random.key(0),
        plan=plan,
    )
    one_group = metrics.compare_paired_losses(
        jnp.array([1.0, 2.0]),
        jnp.array([0.5, 1.5]),
        key=jax.random.key(0),
        plan=plan,
        groups=jnp.array([7, 7]),
    )
    empty = metrics.compare_paired_losses(
        jnp.ones(2),
        jnp.ones(2),
        key=jax.random.key(0),
        plan=plan,
        mask=jnp.zeros(2, dtype=bool),
    )
    invalid = metrics.compare_paired_losses(
        jnp.array([1.0, jnp.nan]),
        jnp.ones(2),
        key=jax.random.key(0),
        plan=plan,
    )

    assert int(one_case.status) == metrics.METRIC_UNDEFINED
    assert int(one_group.status) == metrics.METRIC_UNDEFINED
    assert int(empty.status) == metrics.METRIC_EMPTY
    assert int(invalid.status) == metrics.METRIC_INVALID_INPUT
    assert not bool(one_case.valid)
    assert not bool(one_group.valid)
    assert jnp.isnan(one_case.effect)
    assert jnp.all(jnp.isnan(one_group.bootstrap_effects))
    negative = metrics.compare_paired_losses(
        jnp.array([1.0, -1.0]),
        jnp.ones(2),
        key=jax.random.key(0),
        plan=plan,
    )
    assert int(negative.status) == metrics.METRIC_INVALID_INPUT
