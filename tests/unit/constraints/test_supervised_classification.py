#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_binary_classification_matches_weighted_bernoulli_nll_and_metrics():
    logits = jnp.asarray([-2.0, -0.4, 0.6, 2.2])
    domain = phx.domain.DatasetDomain(logits[:, None])
    targets = jnp.asarray([0, 0, 1, 1], dtype=jnp.int32)
    weights = jnp.asarray([1.0, 2.0, 3.0, 4.0])

    @domain.Function("data")
    def field(row):
        return row[0]

    term = phx.terms.SupervisedClassificationTerm(
        "logit",
        domain.component(),
        targets,
        phx.ml.TargetSchema(
            "binary",
            names=("stability",),
            class_labels=("unstable", "stable"),
        ),
        sampling=phx.domain.PointSampling(4, design="uniform"),
        sample_weight=weights,
    )
    batch = term.observed_batch()
    expected_per_case = jax.nn.softplus(logits) - targets * logits
    expected = jnp.sum(weights * expected_per_case) / jnp.sum(weights)
    metrics = term.data_metrics({"logit": field}, batch=batch)

    np.testing.assert_allclose(term.loss({"logit": field}, batch=batch), expected)
    np.testing.assert_allclose(metrics["data_negative_log_likelihood"], expected)
    np.testing.assert_allclose(metrics["data_accuracy"], 1.0)
    np.testing.assert_allclose(metrics["data_effective_weight"], jnp.sum(weights))
    assert bool(metrics["data_valid"])
    assert int(metrics["data_status"]) == phx.ml.metrics.METRIC_SUCCESS
    assert 0.0 <= float(metrics["data_brier_score"]) < 0.25


def test_multiclass_classification_matches_log_softmax_and_is_shift_invariant():
    logits = jnp.asarray(
        [
            [3.0, 0.2, -1.0],
            [-0.5, 2.0, 0.1],
            [0.2, -0.7, 2.4],
            [1.5, 0.8, -0.3],
        ]
    )
    domain = phx.domain.DatasetDomain(logits)
    targets = jnp.asarray([0, 1, 2, 0], dtype=jnp.int32)

    @domain.Function("data")
    def field(row):
        return row

    @domain.Function("data")
    def shifted_field(row):
        return row + 17.0

    term = phx.terms.SupervisedClassificationTerm(
        "phase_logits",
        domain.component(),
        targets,
        phx.ml.TargetSchema(
            "multiclass",
            names=("phase",),
            class_labels=("solid", "liquid", "gas"),
        ),
        sampling=phx.domain.PointSampling(4, design="uniform"),
    )
    batch = term.observed_batch()
    expected = -jnp.mean(
        jax.nn.log_softmax(logits, axis=-1)[jnp.arange(targets.size), targets]
    )
    metrics = term.data_metrics({"phase_logits": field}, batch=batch)

    np.testing.assert_allclose(term.loss({"phase_logits": field}, batch=batch), expected)
    np.testing.assert_allclose(
        term.loss({"phase_logits": shifted_field}, batch=batch), expected, atol=2e-15
    )
    np.testing.assert_allclose(metrics["data_negative_log_likelihood"], expected)
    np.testing.assert_allclose(metrics["data_accuracy"], 1.0)
    assert bool(metrics["data_valid"])


def test_zero_weight_short_circuits_nonfinite_nll_and_gradient():
    domain = phx.domain.DatasetDomain(jnp.asarray([[1.0], [2.0]]))
    term = phx.terms.SupervisedClassificationTerm(
        "u",
        domain.component(),
        jnp.asarray([0, 1], dtype=jnp.int32),
        phx.ml.TargetSchema("binary", class_labels=(0, 1)),
        sampling=phx.domain.PointSampling(2, design="uniform"),
        weight=0.0,
    )
    batch = term.observed_batch()

    def objective(coefficient):
        field = domain.Function("data")(
            lambda row: coefficient * row[0] / jnp.asarray(0.0)
        )
        return term.loss({"u": field}, batch=batch)

    value, gradient = jax.value_and_grad(objective)(jnp.asarray(1.0))

    assert value == 0.0
    assert gradient == 0.0
    assert jnp.isfinite(gradient)


def test_classification_validates_schema_targets_masks_and_output_width():
    domain = phx.domain.DatasetDomain(jnp.asarray([[2.0], [-1.0], [0.5]]))
    sampling = phx.domain.PointSampling(3, design="uniform")
    binary = phx.ml.TargetSchema("binary", class_labels=(0, 1))

    with pytest.raises(ValueError, match="binary and multiclass"):
        phx.terms.SupervisedClassificationTerm(
            "u",
            domain.component(),
            jnp.asarray([0, 1, 0]),
            phx.ml.TargetSchema("continuous"),
            sampling=sampling,
        )
    with pytest.raises(ValueError, match="class_labels"):
        phx.terms.SupervisedClassificationTerm(
            "u",
            domain.component(),
            jnp.asarray([0, 1, 0]),
            phx.ml.TargetSchema("multiclass"),
            sampling=sampling,
        )
    with pytest.raises(TypeError, match="integer or Boolean"):
        phx.terms.SupervisedClassificationTerm(
            "u",
            domain.component(),
            jnp.asarray([0.0, 1.0, 0.0]),
            binary,
            sampling=sampling,
        )
    with pytest.raises(ValueError, match=r"within \[0, 2\)"):
        phx.terms.SupervisedClassificationTerm(
            "u",
            domain.component(),
            jnp.asarray([0, 2, 0]),
            binary,
            sampling=sampling,
        )

    masked = phx.terms.SupervisedClassificationTerm(
        "u",
        domain.component(),
        jnp.asarray([0, -99, 1]),
        binary,
        sampling=sampling,
        sample_mask=jnp.asarray([True, False, True]),
    )
    assert jnp.array_equal(masked.observed_batch().target, jnp.asarray([0, 1]))

    @domain.Function("data")
    def wrong_width(row):
        return jnp.asarray([row[0], -row[0], 0.0])

    with pytest.raises(ValueError, match="incompatible"):
        masked.loss({"u": wrong_width}, batch=masked.observed_batch())
