#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import metrics


def test_metric_results_are_jittable_pytrees_with_stable_status_arrays():
    target = jnp.array([1.0, 2.0, 3.0])
    prediction = jnp.array([0.0, 2.0, 4.0])
    compiled = jax.jit(metrics.mean_squared_error)(target, prediction)

    assert isinstance(compiled, metrics.MetricResult)
    assert jnp.allclose(compiled.value, 2.0 / 3.0)
    assert bool(compiled.valid)
    assert compiled.status.dtype == jnp.int32
    assert int(compiled.status) == metrics.METRIC_SUCCESS


def test_vmap_and_grad_preserve_metric_case_contracts():
    target = jnp.array([1.0, 2.0, 3.0])
    predictions = jnp.array([[1.0, 2.0, 3.0], [0.0, 2.0, 4.0], [2.0, 2.0, 2.0]])
    mapped = jax.vmap(lambda value: metrics.mean_squared_error(target, value).value)(
        predictions
    )
    direct = metrics.mean_squared_error(
        jnp.broadcast_to(target, predictions.shape),
        predictions,
        sample_axis=-1,
    )
    gradient = jax.grad(lambda value: metrics.mean_squared_error(target, value).value)(
        predictions[1]
    )

    assert jnp.allclose(mapped, jnp.array([0.0, 2.0 / 3.0, 2.0 / 3.0]))
    assert jnp.allclose(direct.value, mapped)
    assert jnp.allclose(gradient, 2.0 * (predictions[1] - target) / 3.0)


def test_float_precision_is_preserved_by_weighted_metrics():
    target32 = jnp.array([1.0, 2.0], dtype=jnp.float32)
    target64 = jnp.array([1.0, 2.0], dtype=jnp.float64)
    result32 = metrics.mean_squared_error(target32, target32 + 1.0)
    result64 = metrics.mean_squared_error(target64, target64 + 1.0)

    assert result32.value.dtype == jnp.float32
    assert result64.value.dtype == jnp.float64


def test_function_scorer_has_explicit_direction_and_prediction_first_order():
    scorer = metrics.FunctionScorer(
        metrics.mean_squared_error,
        name="negative_mse",
        greater_is_better=False,
    )
    prediction = jnp.array([0.0, 2.0, 4.0])
    target = jnp.array([1.0, 2.0, 3.0])
    result = scorer.score(
        prediction,
        target,
        sample_weight=jnp.array([1.0, 2.0, 1.0]),
        mask=jnp.array([True, True, True]),
    )

    assert scorer.name == "negative_mse"
    assert scorer.greater_is_better is False
    assert scorer.requires_probabilities is False
    assert isinstance(result, metrics.MetricResult)
    assert jnp.allclose(result.value, 0.5)
    with pytest.raises(AttributeError):
        scorer.greater_is_better = True


def test_function_scorer_preserves_arbitrary_structured_metric_output():
    scorer = metrics.FunctionScorer(
        metrics.expected_calibration_error,
        name="ece",
        greater_is_better=False,
        requires_probabilities=True,
        metric_kwargs={"num_bins": 2},
    )
    probability = jnp.array([0.1, 0.2, 0.8, 0.9])
    target = jnp.array([0, 0, 1, 1])
    result = scorer(probability, target)

    assert scorer.requires_probabilities is True
    assert isinstance(result, metrics.CalibrationResult)
    assert jnp.allclose(result.value, 0.15)
    assert jnp.allclose(result.bin_weight, jnp.array([2.0, 2.0]))
    assert int(result.status) == metrics.METRIC_SUCCESS


def test_function_scorer_is_usable_inside_jit_without_string_dispatch():
    scorer = metrics.FunctionScorer(
        metrics.mean_absolute_error,
        name="mae",
        greater_is_better=False,
    )
    target = jnp.array([1.0, 2.0, 3.0])
    compiled_value = jax.jit(lambda prediction: scorer.score(prediction, target).value)(
        jnp.array([0.0, 2.0, 5.0])
    )
    assert jnp.allclose(compiled_value, 1.0)


def test_all_public_edge_statuses_are_jax_integer_scalars():
    success = metrics.mean_absolute_error(jnp.ones(2), jnp.ones(2))
    empty = metrics.mean_absolute_error(
        jnp.ones(2), jnp.ones(2), mask=jnp.zeros(2, dtype=bool)
    )
    invalid = metrics.mean_absolute_error(
        jnp.ones(2), jnp.ones(2), sample_weight=jnp.array([1.0, -1.0])
    )
    zero_denominator = metrics.r2_score(jnp.ones(2), jnp.ones(2))
    single_class = metrics.roc_auc_score(jnp.zeros(2, dtype=jnp.int32), jnp.arange(2.0))
    undefined = metrics.MetricResult(
        jnp.nan,
        valid=False,
        status=metrics.METRIC_UNDEFINED,
        effective_weight=1.0,
    )

    observed = jnp.stack(
        (
            success.status,
            empty.status,
            invalid.status,
            zero_denominator.status,
            single_class.status,
            undefined.status,
        )
    )
    assert observed.dtype == jnp.int32
    assert jnp.array_equal(
        observed,
        jnp.array(
            [
                metrics.METRIC_SUCCESS,
                metrics.METRIC_EMPTY,
                metrics.METRIC_INVALID_INPUT,
                metrics.METRIC_ZERO_DENOMINATOR,
                metrics.METRIC_SINGLE_CLASS,
                metrics.METRIC_UNDEFINED,
            ],
            dtype=jnp.int32,
        ),
    )


def test_abstract_scorer_protocol_is_abstract_and_call_delegates_to_score():
    with pytest.raises(TypeError, match="abstract"):
        metrics.AbstractScorer()

    scorer = metrics.FunctionScorer(
        metrics.accuracy_score,
        name="accuracy",
        greater_is_better=True,
    )
    target = jnp.array([0, 1, 1])
    correct = scorer(jnp.array([0, 1, 1]), target)
    incorrect = scorer.score(jnp.array([1, 0, 0]), target)

    assert isinstance(scorer, metrics.AbstractScorer)
    assert scorer.greater_is_better is True
    assert jnp.allclose(correct.value, 1.0)
    assert jnp.allclose(incorrect.value, 0.0)


def test_scorer_direction_is_metadata_not_an_implicit_sign_change():
    scorer = metrics.FunctionScorer(
        metrics.mean_squared_error,
        name="mse",
        greater_is_better=False,
    )
    target = jnp.array([0.0, 1.0])
    perfect = scorer.score(target, target)
    worse = scorer.score(jnp.array([2.0, 1.0]), target)

    assert scorer.greater_is_better is False
    assert jnp.allclose(perfect.value, 0.0)
    assert jnp.allclose(worse.value, 2.0)
    assert float(worse.value) > float(perfect.value)


def test_function_scorer_validates_and_freezes_metric_configuration():
    scorer = metrics.FunctionScorer(
        metrics.pinball_loss,
        name="lower_quartile",
        greater_is_better=False,
        metric_kwargs={"quantile": 0.25},
    )
    target = jnp.array([0.0, 2.0])
    prediction = jnp.array([1.0, 0.0])
    result = scorer(prediction, target)

    assert scorer.metric_kwargs == (("quantile", 0.25),)
    assert jnp.allclose(result.value, 0.625)
    with pytest.raises(TypeError, match="callable"):
        metrics.FunctionScorer(1, greater_is_better=False)
    with pytest.raises(TypeError, match="keys"):
        metrics.FunctionScorer(
            metrics.mean_squared_error,
            greater_is_better=False,
            metric_kwargs={1: 2},
        )
    with pytest.raises(TypeError, match="immutable"):
        metrics.FunctionScorer(
            metrics.mean_squared_error,
            greater_is_better=False,
            metric_kwargs={"configuration": [1, 2]},
        )
    with pytest.raises(AttributeError):
        scorer.metric_kwargs = ()


def test_structured_precision_recall_result_is_jittable():
    target = jnp.array([0, 1, 1])
    probability = jnp.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])
    compiled = jax.jit(
        lambda values: metrics.smooth_precision_recall_fscore(
            target, values, average="binary"
        )
    )(probability)

    assert isinstance(compiled, metrics.PrecisionRecallFScoreResult)
    assert compiled.average == "binary"
    assert bool(compiled.valid)
    assert int(compiled.status) == metrics.METRIC_SUCCESS
    assert jnp.allclose(compiled.support, 2.0)
