#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import metrics


def test_regression_preserves_case_sample_output_axes_weights_and_masks():
    target = jnp.array(
        [
            [[1.0, 2.0], [2.0, 4.0], [jnp.nan, jnp.nan]],
            [[5.0, -1.0], [6.0, 1.0], [jnp.nan, jnp.nan]],
        ]
    )
    prediction = jnp.array(
        [
            [[0.0, 2.0], [4.0, 1.0], [jnp.nan, jnp.nan]],
            [[4.0, -1.0], [8.0, -2.0], [jnp.nan, jnp.nan]],
        ]
    )
    weight = jnp.array([1.0, 3.0, 9.0])
    sample_mask = jnp.array([True, True, False])

    raw = metrics.mean_squared_error(
        target,
        prediction,
        sample_weight=weight,
        mask=sample_mask,
        sample_axis=-2,
        output_reduction="raw_values",
    )
    reduced = metrics.mean_squared_error(
        target,
        prediction,
        sample_weight=weight,
        mask=sample_mask,
        sample_axis=-2,
    )
    absolute = metrics.mean_absolute_error(
        target,
        prediction,
        sample_weight=weight,
        mask=sample_mask,
        sample_axis=-2,
        output_reduction="raw_values",
    )

    assert raw.value.shape == (2, 2)
    assert jnp.all(raw.valid)
    assert jnp.all(raw.status == metrics.METRIC_SUCCESS)
    assert jnp.allclose(raw.value, jnp.array([[3.25, 6.75], [3.25, 6.75]]))
    assert jnp.allclose(raw.effective_weight, 4.0)
    assert jnp.allclose(reduced.value, jnp.array([5.0, 5.0]))
    assert jnp.allclose(absolute.value, jnp.array([[1.75, 2.25], [1.75, 2.25]]))


def test_regression_definitions_and_explicit_edge_statuses():
    target = jnp.array([1.0, 2.0, 3.0])
    perfect = metrics.r2_score(target, target)
    explained = metrics.explained_variance_score(target, target + 2.0)
    median = metrics.pinball_loss(target, jnp.array([0.0, 2.0, 4.0]))

    assert jnp.allclose(perfect.value, 1.0)
    assert jnp.allclose(explained.value, 1.0)
    assert jnp.allclose(median.value, 1.0 / 3.0)

    empty = metrics.mean_squared_error(target, target, mask=jnp.zeros(3, dtype=bool))
    invalid = metrics.mean_squared_error(
        target, target, sample_weight=jnp.array([1.0, -1.0, 1.0])
    )
    zero_denominator = metrics.r2_score(jnp.ones(3), jnp.ones(3))

    assert int(empty.status) == metrics.METRIC_EMPTY
    assert int(invalid.status) == metrics.METRIC_INVALID_INPUT
    assert int(zero_denominator.status) == metrics.METRIC_ZERO_DENOMINATOR
    assert not bool(empty.valid)
    assert not bool(invalid.valid)
    assert not bool(zero_denominator.valid)


def test_classification_exact_weighted_catalog_and_confusion_orientation():
    target = jnp.array([0, 1, 1, 0])
    prediction = jnp.array([0, 1, 0, 0])
    weight = jnp.array([1.0, 2.0, 3.0, 4.0])

    accuracy = metrics.accuracy_score(target, prediction, sample_weight=weight)
    balanced = metrics.balanced_accuracy_score(
        target, prediction, num_classes=2, sample_weight=weight
    )
    confusion = metrics.confusion_matrix(
        target, prediction, num_classes=2, sample_weight=weight
    )
    report = metrics.precision_recall_fscore(
        target,
        prediction,
        num_classes=2,
        average="binary",
        sample_weight=weight,
    )

    assert jnp.allclose(accuracy.value, 0.7)
    assert jnp.allclose(confusion.value, jnp.array([[5.0, 0.0], [3.0, 2.0]]))
    assert jnp.allclose(balanced.value, 0.7)
    assert jnp.allclose(report.precision, 1.0)
    assert jnp.allclose(report.recall, 0.4)
    assert jnp.allclose(report.fscore, 4.0 / 7.0)
    assert jnp.allclose(report.support, 5.0)


def test_classification_probability_scores_and_auc_hard_sorting():
    target = jnp.array([0, 0, 1, 1])
    score = jnp.array([0.1, 0.4, 0.35, 0.8])
    probability = jnp.stack((1.0 - score, score), axis=-1)

    log_score = metrics.log_loss(target, probability)
    brier = metrics.brier_score(target, score)
    roc = metrics.roc_auc_score(target, score)
    pr = metrics.pr_auc_score(target, score)
    tie_roc = metrics.roc_auc_score(target, jnp.ones_like(score))
    tie_pr = metrics.pr_auc_score(target, jnp.ones_like(score))
    impossible_log = metrics.log_loss(
        jnp.array([0, 1]),
        jnp.array([[0.0, 1.0], [0.2, 0.8]]),
    )

    expected_log = -jnp.mean(jnp.log(jnp.array([0.9, 0.6, 0.35, 0.8])))
    expected_brier = jnp.mean((score - target) ** 2)
    assert jnp.allclose(log_score.value, expected_log)
    assert jnp.allclose(brier.value, expected_brier)
    assert jnp.allclose(roc.value, 0.75)
    assert jnp.allclose(pr.value, 19.0 / 24.0)
    assert jnp.allclose(tie_roc.value, 0.5)
    assert jnp.allclose(tie_pr.value, 0.75)
    assert bool(impossible_log.valid)
    assert jnp.isinf(impossible_log.value)


def test_hard_and_smooth_classification_semantics_are_distinct():
    target = jnp.array([0, 1, 1])
    probability = jnp.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])
    hard = metrics.accuracy_score(target, jnp.argmax(probability, axis=-1))
    smooth = metrics.smooth_accuracy_score(target, probability)
    smooth_f = metrics.smooth_f1_score(target, probability, average="binary")

    assert jnp.allclose(hard.value, 1.0)
    assert jnp.allclose(smooth.value, 0.7)
    assert 0.0 < float(smooth_f.value) < 1.0

    hard_gradient = jax.grad(lambda values: metrics.roc_auc_score(target, values).value)(
        probability[:, 1]
    )
    smooth_gradient = jax.grad(
        lambda values: (
            metrics.smooth_roc_auc_score(target, values, temperature=0.25).value
        )
    )(probability[:, 1])
    assert jnp.allclose(hard_gradient, 0.0)
    assert jnp.any(jnp.abs(smooth_gradient) > 0.0)
    assert jnp.all(jnp.isfinite(smooth_gradient))


def test_classification_single_class_and_complex_policies_fail_closed():
    single = metrics.roc_auc_score(jnp.zeros(3, dtype=jnp.int32), jnp.arange(3.0))
    assert int(single.status) == metrics.METRIC_SINGLE_CLASS
    assert not bool(single.valid)

    complex_target = jnp.array([1.0 + 1.0j, 2.0 - 1.0j])
    complex_prediction = jnp.array([0.0 + 1.0j, 2.0 + 1.0j])
    squared = metrics.mean_squared_error(complex_target, complex_prediction)
    assert jnp.allclose(squared.value, 2.5)

    with pytest.raises(TypeError, match="complex"):
        metrics.pinball_loss(complex_target, complex_prediction)
    with pytest.raises(TypeError, match="complex"):
        metrics.log_loss(
            jnp.array([0, 1]),
            jnp.array([[0.5 + 0.0j, 0.5], [0.2, 0.8]]),
        )


def test_regression_catalog_output_reductions_and_gradients():
    target = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 4.0]])
    prediction = jnp.array([[0.0, 1.0], [2.0, 1.0], [0.0, 5.0]])

    squared = metrics.mean_squared_error(
        target, prediction, sample_axis=0, output_reduction="raw_values"
    )
    root = metrics.root_mean_squared_error(
        target, prediction, sample_axis=0, output_reduction="raw_values"
    )
    absolute = metrics.mean_absolute_error(
        target, prediction, sample_axis=0, output_reduction="raw_values"
    )
    pinball = metrics.pinball_loss(
        target,
        prediction,
        quantile=jnp.array([0.25, 0.75]),
        sample_axis=0,
        output_reduction="raw_values",
    )

    assert jnp.allclose(squared.value, jnp.array([5.0 / 3.0, 2.0 / 3.0]))
    assert jnp.allclose(root.value, jnp.sqrt(squared.value))
    assert jnp.allclose(absolute.value, jnp.array([1.0, 2.0 / 3.0]))
    assert jnp.allclose(pinball.value, jnp.array([5.0 / 12.0, 1.0 / 6.0]))

    target_variance = jnp.var(target, axis=0)
    raw_r2 = metrics.r2_score(
        target, prediction, sample_axis=0, output_reduction="raw_values"
    )
    uniform_r2 = metrics.r2_score(target, prediction, sample_axis=0)
    weighted_r2 = metrics.r2_score(
        target, prediction, sample_axis=0, output_reduction="variance_weighted"
    )
    raw_explained = metrics.explained_variance_score(
        target, prediction, sample_axis=0, output_reduction="raw_values"
    )
    weighted_explained = metrics.explained_variance_score(
        target, prediction, sample_axis=0, output_reduction="variance_weighted"
    )

    assert jnp.allclose(uniform_r2.value, jnp.mean(raw_r2.value))
    assert jnp.allclose(
        weighted_r2.value,
        jnp.sum(target_variance * raw_r2.value) / jnp.sum(target_variance),
    )
    assert jnp.allclose(
        weighted_explained.value,
        jnp.sum(target_variance * raw_explained.value) / jnp.sum(target_variance),
    )

    gradient = jax.grad(
        lambda values: (
            metrics.mean_squared_error(target, values, sample_axis=0).value
            + metrics.root_mean_squared_error(target, values, sample_axis=0).value
            + metrics.mean_absolute_error(target, values, sample_axis=0).value
            + metrics.pinball_loss(target, values, sample_axis=0).value
        )
    )(prediction)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(jnp.abs(gradient) > 0.0)

    invalid_quantile = metrics.pinball_loss(
        target, prediction, quantile=1.0, sample_axis=0
    )
    assert int(invalid_quantile.status) == metrics.METRIC_INVALID_INPUT
    with pytest.raises(ValueError, match="variance_weighted"):
        metrics.mean_squared_error(
            target,
            prediction,
            sample_axis=0,
            output_reduction="variance_weighted",
        )


def test_classification_averaging_wrappers_and_denominator_states():
    target = jnp.array([0, 1, 2, 2, 1, 0])
    prediction = jnp.array([0, 2, 2, 1, 1, 0])

    per_class = metrics.precision_recall_fscore(
        target, prediction, num_classes=3, average="none"
    )
    macro = metrics.precision_recall_fscore(
        target, prediction, num_classes=3, average="macro"
    )
    weighted = metrics.precision_recall_fscore(
        target, prediction, num_classes=3, average="weighted"
    )
    micro = metrics.precision_recall_fscore(
        target, prediction, num_classes=3, average="micro"
    )

    expected_by_class = jnp.array([1.0, 0.5, 0.5])
    assert per_class.average == "none"
    assert jnp.allclose(per_class.precision, expected_by_class)
    assert jnp.allclose(per_class.recall, expected_by_class)
    assert jnp.allclose(per_class.fscore, expected_by_class)
    assert jnp.allclose(per_class.support, jnp.full(3, 2.0))
    assert jnp.allclose(macro.fscore, 2.0 / 3.0)
    assert jnp.allclose(weighted.fscore, 2.0 / 3.0)
    assert jnp.allclose(micro.fscore, 2.0 / 3.0)

    binary_target = jnp.array([0, 1, 1, 0])
    binary_prediction = jnp.array([0, 1, 0, 0])
    sample_weight = jnp.array([1.0, 2.0, 3.0, 4.0])
    kwargs = {
        "num_classes": 2,
        "average": "binary",
        "sample_weight": sample_weight,
    }
    precision = metrics.precision_score(binary_target, binary_prediction, **kwargs)
    recall = metrics.recall_score(binary_target, binary_prediction, **kwargs)
    f1 = metrics.f1_score(binary_target, binary_prediction, **kwargs)
    fbeta = metrics.fbeta_score(binary_target, binary_prediction, beta=2.0, **kwargs)
    assert jnp.allclose(precision.value, 1.0)
    assert jnp.allclose(recall.value, 0.4)
    assert jnp.allclose(f1.value, 4.0 / 7.0)
    assert jnp.allclose(fbeta.value, 5.0 / 11.0)

    absent_positive = metrics.precision_recall_fscore(
        jnp.zeros(3, dtype=jnp.int32),
        jnp.zeros(3, dtype=jnp.int32),
        num_classes=2,
        average="binary",
    )
    missing_row = metrics.confusion_matrix(
        jnp.zeros(3, dtype=jnp.int32),
        jnp.zeros(3, dtype=jnp.int32),
        num_classes=2,
        normalize="true",
    )
    empty = metrics.accuracy_score(
        binary_target,
        binary_prediction,
        mask=jnp.zeros(4, dtype=bool),
    )
    invalid = metrics.brier_score(binary_target, jnp.array([0.1, 1.2, 0.8, 0.2]))
    invalid_label = metrics.smooth_accuracy_score(
        jnp.array([0, 2]),
        jnp.array([[0.8, 0.2], [0.2, 0.8]]),
    )

    assert int(absent_positive.status) == metrics.METRIC_ZERO_DENOMINATOR
    assert int(missing_row.status) == metrics.METRIC_ZERO_DENOMINATOR
    assert int(empty.status) == metrics.METRIC_EMPTY
    assert int(invalid.status) == metrics.METRIC_INVALID_INPUT
    assert int(invalid_label.status) == metrics.METRIC_INVALID_INPUT


def test_smooth_classification_catalog_matches_expected_count_wrappers():
    target = jnp.array([0, 1, 1])
    probability = jnp.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])

    confusion = metrics.smooth_confusion_matrix(target, probability)
    balanced = metrics.smooth_balanced_accuracy_score(target, probability)
    report = metrics.smooth_precision_recall_fscore(target, probability, average="binary")
    precision = metrics.smooth_precision_score(target, probability, average="binary")
    recall = metrics.smooth_recall_score(target, probability, average="binary")
    f1 = metrics.smooth_f1_score(target, probability, average="binary")
    fbeta = metrics.smooth_fbeta_score(target, probability, average="binary", beta=2.0)

    assert jnp.allclose(confusion.value, jnp.array([[0.8, 0.2], [0.7, 1.3]]))
    assert jnp.allclose(balanced.value, 0.725)
    assert jnp.allclose(precision.value, report.precision)
    assert jnp.allclose(recall.value, report.recall)
    assert jnp.allclose(f1.value, report.fscore)
    assert jnp.allclose(report.precision, 13.0 / 15.0)
    assert jnp.allclose(report.recall, 0.65)
    assert bool(fbeta.valid)

    logits = jnp.log(probability)
    probability_log_loss = metrics.log_loss(target, probability)
    logits_log_loss = metrics.log_loss(target, logits, from_logits=True)
    multiclass_brier = metrics.brier_score(target, probability)
    expected_brier = jnp.mean(
        jnp.sum(
            (probability - jax.nn.one_hot(target, 2, dtype=probability.dtype)) ** 2,
            axis=-1,
        )
    )
    assert jnp.allclose(probability_log_loss.value, logits_log_loss.value)
    assert jnp.allclose(multiclass_brier.value, expected_brier)

    smooth_values = (
        lambda values: metrics.smooth_accuracy_score(target, values).value,
        lambda values: metrics.smooth_balanced_accuracy_score(target, values).value,
        lambda values: jnp.trace(metrics.smooth_confusion_matrix(target, values).value),
        lambda values: (
            metrics.smooth_precision_score(target, values, average="binary").value
        ),
        lambda values: (
            metrics.smooth_recall_score(target, values, average="binary").value
        ),
        lambda values: (
            metrics.smooth_fbeta_score(target, values, average="binary", beta=2.0).value
        ),
        lambda values: metrics.smooth_f1_score(target, values, average="binary").value,
    )
    for value_function in smooth_values:
        gradient = jax.grad(value_function)(probability)
        assert jnp.all(jnp.isfinite(gradient))
        assert jnp.any(jnp.abs(gradient) > 0.0)

    score = probability[:, 1]
    pr_gradient = jax.grad(
        lambda values: metrics.smooth_pr_auc_score(target, values, temperature=0.25).value
    )(score)
    assert jnp.all(jnp.isfinite(pr_gradient))
    assert jnp.any(jnp.abs(pr_gradient) > 0.0)


def test_classification_case_axes_vmap_and_jit_from_logits():
    target = jnp.array([[0, 1, 1], [1, 0, 0]])
    logits = jnp.array(
        [
            [[2.0, -1.0], [-1.0, 2.0], [0.0, 1.0]],
            [[-1.0, 2.0], [2.0, -1.0], [1.0, 0.0]],
        ]
    )
    weight = jnp.array([1.0, 2.0, 1.0])
    mask = jnp.array([[True, True, False], [True, True, True]])

    batched = metrics.smooth_accuracy_score(
        target,
        logits,
        sample_weight=weight,
        mask=mask,
        sample_axis=-1,
        from_logits=True,
    )
    mapped = jax.vmap(
        lambda labels, values, included: (
            metrics.smooth_accuracy_score(
                labels,
                values,
                sample_weight=weight,
                mask=included,
                from_logits=True,
            ).value
        )
    )(target, logits, mask)
    compiled_loss = jax.jit(
        lambda values: (
            metrics.log_loss(
                target,
                values,
                sample_weight=weight,
                mask=mask,
                from_logits=True,
            ).value
        )
    )(logits)

    assert batched.value.shape == (2,)
    assert jnp.allclose(batched.value, mapped)
    assert compiled_loss.shape == (2,)
    assert jnp.all(jnp.isfinite(compiled_loss))
