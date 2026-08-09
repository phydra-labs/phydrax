#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import metrics


def test_exact_ranking_metrics_use_hard_order_and_weighted_masks():
    relevance = jnp.array([1.0, 0.0, 1.0, 1.0])
    score = jnp.array([4.0, 3.0, 2.0, 100.0])
    mask = jnp.array([True, True, True, False])
    weight = jnp.array([1.0, 1.0, 1.0, 9.0])

    precision = metrics.precision_at_k(
        relevance, score, k=2, sample_weight=weight, mask=mask
    )
    recall = metrics.recall_at_k(relevance, score, k=2, sample_weight=weight, mask=mask)
    reciprocal = metrics.reciprocal_rank(
        relevance, score, sample_weight=weight, mask=mask
    )
    average_precision = metrics.average_precision_score(
        relevance, score, sample_weight=weight, mask=mask
    )

    assert jnp.allclose(precision.value, 0.5)
    assert jnp.allclose(recall.value, 0.5)
    assert jnp.allclose(reciprocal.value, 1.0)
    assert jnp.allclose(average_precision.value, 5.0 / 6.0)
    assert jnp.allclose(precision.effective_weight, 3.0)


def test_dcg_ndcg_exact_and_smooth_rank_semantics():
    relevance = jnp.array([3.0, 2.0, 0.0])
    ideal_score = jnp.array([3.0, 2.0, 0.0])
    reversed_score = -ideal_score

    ideal = metrics.ndcg_score(relevance, ideal_score)
    reversed_result = metrics.ndcg_score(relevance, reversed_score)
    dcg = metrics.discounted_cumulative_gain(relevance, ideal_score, gain="identity")

    assert jnp.allclose(ideal.value, 1.0)
    assert float(reversed_result.value) < float(ideal.value)
    assert jnp.allclose(dcg.value, 3.0 + 2.0 / jnp.log2(3.0))

    exact_gradient = jax.grad(lambda values: metrics.ndcg_score(relevance, values).value)(
        ideal_score
    )
    smooth_gradient = jax.grad(
        lambda values: metrics.smooth_ndcg_score(relevance, values, temperature=0.5).value
    )(ideal_score)
    assert jnp.allclose(exact_gradient, 0.0)
    assert jnp.any(jnp.abs(smooth_gradient) > 0.0)
    assert jnp.all(jnp.isfinite(smooth_gradient))


def test_ranking_zero_relevance_has_explicit_denominator_status():
    result = metrics.average_precision_score(jnp.zeros(3), jnp.arange(3.0))
    assert not bool(result.valid)
    assert int(result.status) == metrics.METRIC_ZERO_DENOMINATOR


def test_hard_clustering_scores_match_separated_partition():
    features = jnp.array([[0.0], [1.0], [10.0], [11.0]])
    labels = jnp.array([0, 0, 1, 1])

    silhouette = metrics.silhouette_score(features, labels, num_clusters=2)
    davies_bouldin = metrics.davies_bouldin_score(features, labels, num_clusters=2)
    calinski = metrics.calinski_harabasz_score(features, labels, num_clusters=2)
    adjusted_rand = metrics.adjusted_rand_score(
        labels,
        labels,
        num_true_clusters=2,
        num_pred_clusters=2,
    )
    mutual_information = metrics.normalized_mutual_info_score(
        labels,
        labels,
        num_true_clusters=2,
        num_pred_clusters=2,
    )

    assert bool(silhouette.valid)
    assert float(silhouette.value) > 0.85
    assert jnp.allclose(davies_bouldin.value, 0.1)
    assert float(calinski.value) > 100.0
    assert jnp.allclose(adjusted_rand.value, 1.0)
    assert jnp.allclose(mutual_information.value, 1.0)


def test_soft_clustering_scores_are_differentiable_in_membership_logits():
    features = jnp.array([[0.0], [1.0], [10.0], [11.0]])
    logits = jnp.array([[5.0, -5.0], [4.0, -4.0], [-4.0, 4.0], [-5.0, 5.0]])

    silhouette_gradient = jax.grad(
        lambda values: (
            metrics.smooth_silhouette_score(
                features,
                values,
                from_logits=True,
                temperature=0.5,
            ).value
        )
    )(logits)
    calinski_gradient = jax.grad(
        lambda values: (
            metrics.smooth_calinski_harabasz_score(
                features, values, from_logits=True
            ).value
        )
    )(logits)

    assert jnp.all(jnp.isfinite(silhouette_gradient))
    assert jnp.any(jnp.abs(silhouette_gradient) > 0.0)
    assert jnp.all(jnp.isfinite(calinski_gradient))
    assert jnp.any(jnp.abs(calinski_gradient) > 0.0)


def test_clustering_single_cluster_and_complex_feature_policies():
    real_features = jnp.arange(4.0)[:, None]
    single = metrics.silhouette_score(
        real_features, jnp.zeros(4, dtype=jnp.int32), num_clusters=2
    )
    assert not bool(single.valid)
    assert int(single.status) == metrics.METRIC_SINGLE_CLASS

    complex_features = jnp.array(
        [[0.0 + 0.0j], [1.0 + 1.0j], [10.0 - 1.0j], [11.0 + 0.0j]]
    )
    complex_result = metrics.silhouette_score(
        complex_features, jnp.array([0, 0, 1, 1]), num_clusters=2
    )
    assert bool(complex_result.valid)
    assert jnp.isfinite(complex_result.value)


def test_ranking_ties_case_axes_jit_and_vmap_are_deterministic():
    tied = metrics.discounted_cumulative_gain(
        jnp.array([3.0, 1.0, 2.0]),
        jnp.zeros(3),
        gain="identity",
    )
    assert jnp.allclose(
        tied.value,
        3.0 + 1.0 / jnp.log2(3.0) + 2.0 / jnp.log2(4.0),
    )

    relevance = jnp.array([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    score = jnp.array([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    weight = jnp.array([1.0, 2.0, 1.0])
    batched = metrics.ndcg_score(relevance, score, sample_weight=weight, sample_axis=-1)
    mapped = jax.vmap(
        lambda rel, values: metrics.ndcg_score(rel, values, sample_weight=weight).value
    )(relevance, score)
    compiled = jax.jit(
        lambda values: metrics.ndcg_score(relevance, values, sample_weight=weight).value
    )(score)

    assert batched.value.shape == (2,)
    assert jnp.allclose(batched.value, mapped)
    assert jnp.allclose(compiled, mapped)


def test_all_smooth_ranking_surrogates_have_prediction_gradients():
    relevance = jnp.array([1.0, 0.2, 0.8, 0.0])
    score = jnp.array([1.2, 0.4, 0.7, -0.3])

    smooth_values = (
        lambda values: (
            metrics.smooth_ndcg_score(relevance, values, temperature=0.5).value
        ),
        lambda values: (
            metrics.smooth_precision_at_k(
                relevance,
                values,
                k=2,
                temperature=0.5,
                relevance_temperature=0.4,
                relevance_threshold=0.5,
            ).value
        ),
        lambda values: (
            metrics.smooth_recall_at_k(
                relevance,
                values,
                k=2,
                temperature=0.5,
                relevance_temperature=0.4,
                relevance_threshold=0.5,
            ).value
        ),
        lambda values: (
            metrics.smooth_reciprocal_rank(
                relevance,
                values,
                temperature=0.5,
                relevance_temperature=0.4,
                relevance_threshold=0.5,
            ).value
        ),
        lambda values: (
            metrics.smooth_average_precision_score(
                relevance,
                values,
                temperature=0.5,
                relevance_temperature=0.4,
                relevance_threshold=0.5,
            ).value
        ),
    )

    for value_function in smooth_values:
        value = jax.jit(value_function)(score)
        gradient = jax.grad(value_function)(score)
        assert jnp.isfinite(value)
        assert jnp.all(jnp.isfinite(gradient))
        assert jnp.any(jnp.abs(gradient) > 0.0)


def test_ranking_invalid_empty_and_zero_denominator_states():
    relevance = jnp.array([1.0, 0.0, 0.0])
    score = jnp.array([1.0, 3.0, 2.0])

    empty = metrics.discounted_cumulative_gain(
        relevance, score, mask=jnp.zeros(3, dtype=bool)
    )
    invalid = metrics.ndcg_score(jnp.array([1.0, -1.0, 0.0]), score)
    zero_ndcg = metrics.ndcg_score(jnp.zeros(3), score)
    zero_recall = metrics.recall_at_k(jnp.zeros(3), score, k=2)
    zero_reciprocal = metrics.reciprocal_rank(jnp.zeros(3), score)

    assert int(empty.status) == metrics.METRIC_EMPTY
    assert int(invalid.status) == metrics.METRIC_INVALID_INPUT
    assert int(zero_ndcg.status) == metrics.METRIC_ZERO_DENOMINATOR
    assert int(zero_recall.status) == metrics.METRIC_ZERO_DENOMINATOR
    assert int(zero_reciprocal.status) == metrics.METRIC_ZERO_DENOMINATOR

    with pytest.raises(TypeError, match="complex"):
        metrics.ndcg_score(
            relevance.astype(jnp.complex64),
            score.astype(jnp.complex64),
        )
    with pytest.raises(ValueError, match="positive"):
        metrics.precision_at_k(relevance, score, k=0)


def test_clustering_weights_masks_case_axes_jit_and_vmap():
    features = jnp.array([[0.0], [1.0], [10.0], [11.0], [jnp.nan]])
    labels = jnp.array([0, 0, 1, 1, 0])
    weight = jnp.array([1.0, 2.0, 1.0, 2.0, 9.0])
    mask = jnp.array([True, True, True, True, False])
    masked = metrics.silhouette_score(
        features,
        labels,
        num_clusters=2,
        sample_weight=weight,
        mask=mask,
    )
    clean = metrics.silhouette_score(
        features[:4],
        labels[:4],
        num_clusters=2,
        sample_weight=weight[:4],
    )
    assert bool(masked.valid)
    assert jnp.allclose(masked.value, clean.value)
    assert jnp.allclose(masked.effective_weight, 6.0)

    feature_cases = jnp.stack((features[:4], features[:4] + 3.0))
    label_cases = jnp.stack((labels[:4], labels[:4]))
    batched = metrics.silhouette_score(
        feature_cases, label_cases, num_clusters=2, sample_axis=-2
    )
    mapped = jax.vmap(
        lambda values, groups: (
            metrics.silhouette_score(values, groups, num_clusters=2).value
        )
    )(feature_cases, label_cases)
    compiled = jax.jit(
        lambda values: metrics.silhouette_score(values, label_cases, num_clusters=2).value
    )(feature_cases)

    assert batched.value.shape == (2,)
    assert jnp.allclose(batched.value, mapped)
    assert jnp.allclose(compiled, mapped)


def test_soft_pair_clustering_metrics_values_and_membership_gradients():
    hard_membership = jax.nn.one_hot(jnp.array([0, 0, 1, 1]), 2)
    exact_rand = metrics.smooth_rand_score(hard_membership, hard_membership)
    exact_nmi = metrics.smooth_normalized_mutual_info_score(
        hard_membership, hard_membership
    )
    assert jnp.allclose(exact_rand.value, 1.0)
    assert jnp.allclose(exact_nmi.value, 1.0)

    membership_true = jnp.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    membership_pred = jnp.array([[0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8]])

    def combined(values):
        return (
            metrics.smooth_rand_score(membership_true, values).value
            + metrics.smooth_normalized_mutual_info_score(membership_true, values).value
        )

    compiled = jax.jit(combined)(membership_pred)
    gradient = jax.grad(combined)(membership_pred)
    assert jnp.isfinite(compiled)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(jnp.abs(gradient) > 0.0)


def test_clustering_zero_denominator_empty_and_invalid_states():
    coincident = jnp.zeros((4, 1))
    labels = jnp.array([0, 0, 1, 1])
    davies = metrics.davies_bouldin_score(coincident, labels, num_clusters=2)
    calinski = metrics.calinski_harabasz_score(coincident, labels, num_clusters=2)
    adjusted = metrics.adjusted_rand_score(
        jnp.array([0]),
        jnp.array([0]),
        num_true_clusters=1,
        num_pred_clusters=1,
    )
    mutual_information = metrics.normalized_mutual_info_score(
        jnp.zeros(3, dtype=jnp.int32),
        jnp.zeros(3, dtype=jnp.int32),
        num_true_clusters=1,
        num_pred_clusters=1,
    )
    smooth_pair = metrics.smooth_rand_score(
        jnp.array([[0.5, 0.5]]), jnp.array([[0.4, 0.6]])
    )
    empty = metrics.silhouette_score(
        jnp.arange(4.0)[:, None],
        labels,
        num_clusters=2,
        mask=jnp.zeros(4, dtype=bool),
    )
    invalid_label = metrics.silhouette_score(
        jnp.arange(4.0)[:, None],
        jnp.array([0, 0, 1, 2]),
        num_clusters=2,
    )
    invalid_membership = metrics.smooth_rand_score(
        jnp.array([[0.7, 0.7], [0.2, 0.8]]),
        jnp.array([[0.5, 0.5], [0.5, 0.5]]),
    )

    assert int(davies.status) == metrics.METRIC_ZERO_DENOMINATOR
    assert int(calinski.status) == metrics.METRIC_ZERO_DENOMINATOR
    assert int(adjusted.status) == metrics.METRIC_ZERO_DENOMINATOR
    assert int(mutual_information.status) == metrics.METRIC_ZERO_DENOMINATOR
    assert int(smooth_pair.status) == metrics.METRIC_ZERO_DENOMINATOR
    assert int(empty.status) == metrics.METRIC_EMPTY
    assert int(invalid_label.status) == metrics.METRIC_INVALID_INPUT
    assert int(invalid_membership.status) == metrics.METRIC_INVALID_INPUT

    with pytest.raises(TypeError, match="complex"):
        metrics.smooth_rand_score(
            jnp.array([[0.5 + 0.0j, 0.5]]),
            jnp.array([[0.5 + 0.0j, 0.5]]),
        )
