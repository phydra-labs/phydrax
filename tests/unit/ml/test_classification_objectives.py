#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._classification import (
    binary_focal_risk_from_logits,
    binary_log_prob_from_logits,
    categorical_focal_risk_from_logits,
    categorical_log_prob_from_logits,
    classification_probabilities,
    ordinal_log_prob_from_location,
    pointwise_classification_loss,
    soft_categorical_cross_entropy_from_logits,
)


def test_hard_classification_kernels_preserve_likelihood_contracts():
    binary_logits = jnp.asarray([-10_000.0, 10_000.0])
    binary_target = jnp.asarray([0, 1])
    assert jnp.all(
        jnp.isfinite(binary_log_prob_from_logits(binary_logits, binary_target))
    )

    logits = jnp.asarray([[3.0, -1.0, 0.2], [-0.5, 1.4, 0.1]])
    labels = jnp.asarray([0, 1])
    expected = jax.nn.log_softmax(logits, axis=-1)[jnp.arange(2), labels]
    np.testing.assert_allclose(
        categorical_log_prob_from_logits(logits, labels), expected, atol=2e-15
    )
    np.testing.assert_allclose(
        categorical_log_prob_from_logits(logits + 19.0, labels),
        expected,
        atol=2e-15,
    )
    invalid = categorical_log_prob_from_logits(logits, jnp.asarray([-1.0, 3.0]))
    assert jnp.all(jnp.isneginf(invalid))


def test_soft_categorical_cross_entropy_validates_simplex_without_renormalizing():
    logits = jnp.asarray([[2.0, -1.0, 0.3]])
    target = jnp.asarray([[0.25, 0.5, 0.25]])
    expected = -jnp.sum(target * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    np.testing.assert_allclose(
        soft_categorical_cross_entropy_from_logits(logits, target), expected
    )
    assert jnp.isinf(
        soft_categorical_cross_entropy_from_logits(
            logits, jnp.asarray([[0.2, 0.2, 0.2]])
        )[0]
    )


def test_focal_gamma_zero_matches_nll_in_value_and_gradient():
    binary_logits = jnp.asarray([-2.0, 0.3, 1.7])
    binary_target = jnp.asarray([0, 1, 1])
    np.testing.assert_allclose(
        binary_focal_risk_from_logits(binary_logits, binary_target, gamma=0.0),
        -binary_log_prob_from_logits(binary_logits, binary_target),
    )
    binary_focal_grad = jax.grad(
        lambda values: jnp.sum(
            binary_focal_risk_from_logits(values, binary_target, gamma=0.0)
        )
    )(binary_logits)
    binary_nll_grad = jax.grad(
        lambda values: -jnp.sum(binary_log_prob_from_logits(values, binary_target))
    )(binary_logits)
    np.testing.assert_allclose(binary_focal_grad, binary_nll_grad)

    logits = jnp.asarray([[1.2, -0.4, 0.1], [-0.7, 0.3, 1.6]])
    labels = jnp.asarray([0, 2])
    np.testing.assert_allclose(
        categorical_focal_risk_from_logits(logits, labels, gamma=0.0),
        -categorical_log_prob_from_logits(logits, labels),
    )


def test_target_masks_sanitize_invalid_inactive_values_before_scoring():
    logits = jnp.asarray([[2.0, -1.0], [jnp.nan, jnp.nan]])
    labels = jnp.asarray([0, -99])
    mask = jnp.asarray([True, False])
    loss = pointwise_classification_loss(
        logits,
        labels,
        kind="multiclass",
        objective="nll",
        class_count=2,
        target_mask=mask,
    )
    assert jnp.isfinite(loss[0])
    assert loss[1] == 0.0
    gradient = jax.grad(
        lambda values: jnp.sum(
            pointwise_classification_loss(
                values,
                labels,
                kind="multiclass",
                class_count=2,
                target_mask=mask,
            )
        )
    )(logits)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.all(gradient[1] == 0.0)


def test_ordinal_probabilities_and_log_prob_are_ordered_and_normalized():
    thresholds = jnp.asarray([-1.0, 0.5, 2.0])
    location = jnp.asarray([-2.0, 0.0, 3.0])
    probabilities = classification_probabilities(
        location,
        kind="ordinal",
        thresholds=thresholds,
    )
    np.testing.assert_allclose(jnp.sum(probabilities, axis=-1), jnp.ones((3,)))
    assert jnp.all(probabilities >= 0.0)
    exceedance = jax.nn.sigmoid(location[..., None] - thresholds)
    assert jnp.all(exceedance[1:] >= exceedance[:-1])
    labels = jnp.asarray([0, 1, 3])
    selected = jnp.take_along_axis(probabilities, labels[..., None], axis=-1)[..., 0]
    np.testing.assert_allclose(
        ordinal_log_prob_from_location(location, labels, thresholds),
        jnp.log(selected),
        atol=2e-14,
    )
    np.testing.assert_allclose(
        classification_probabilities(
            location + 7.0,
            kind="ordinal",
            thresholds=thresholds + 7.0,
        ),
        probabilities,
        atol=2e-15,
    )


def test_classification_objective_is_canonical_and_json_safe():
    focal = phx.ml.ClassificationObjective.focal(
        gamma=1.5,
        alpha=(1.0, 2.0, 3.0),
    )
    assert focal.to_dict() == {
        "kind": "focal",
        "gamma": 1.5,
        "alpha": (1.0, 2.0, 3.0),
        "thresholds": None,
    }
    ordinal = phx.ml.ClassificationObjective.nll(thresholds=(-1.0, 0.0, 2.0))
    assert ordinal.thresholds == (-1.0, 0.0, 2.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        phx.ml.ClassificationObjective.nll(thresholds=(0.0, 0.0))


def test_target_schema_distinguishes_multilabel_ordinal_and_ranking():
    multilabel = phx.ml.TargetSchema("multilabel", names=("wet", "warm"))
    assert multilabel.num_labels == 2
    ordinal = phx.ml.TargetSchema("ordinal", class_labels=("none", "mild", "severe"))
    assert ordinal.num_classes == 3
    with pytest.raises(ValueError, match="named label"):
        phx.ml.TargetSchema("multilabel")
    with pytest.raises(ValueError, match="at least three"):
        phx.ml.TargetSchema("ordinal", class_labels=("low", "high"))
