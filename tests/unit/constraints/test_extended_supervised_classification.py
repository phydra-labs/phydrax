#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_multilabel_term_masks_factors_and_excludes_empty_rows_from_case_mass():
    logits = jnp.asarray(
        [[2.0, -1.0, jnp.nan], [jnp.nan, jnp.nan, jnp.nan], [-0.5, 1.2, 2.0]]
    )
    target = jnp.asarray([[1, 0, -99], [-99, -99, -99], [0, 1, 1]])
    target_mask = jnp.asarray(
        [[True, True, False], [False, False, False], [True, True, True]]
    )
    domain = phx.domain.DatasetDomain(jnp.arange(3.0)[:, None])

    @domain.Function("data")
    def field(row):
        return logits[row[0].astype(jnp.int32)]

    term = phx.terms.SupervisedClassificationTerm(
        "labels",
        domain.component(),
        target,
        phx.ml.TargetSchema("multilabel", names=("a", "b", "c")),
        sampling=phx.domain.PointSampling(2, design="uniform"),
        target_mask=target_mask,
    )
    batch = term.observed_batch()
    expected_indices = jnp.asarray([0, 2])
    expected = (
        jax.nn.softplus(logits[expected_indices])
        - target[expected_indices] * logits[expected_indices]
    )
    expected = jnp.where(target_mask[expected_indices], expected, 0.0).sum(axis=-1)

    assert jnp.array_equal(batch.indices, expected_indices)
    np.testing.assert_allclose(
        term.loss({"labels": field}, batch=batch), jnp.mean(expected)
    )
    metrics = term.data_metrics({"labels": field}, batch=batch)
    assert metrics["data_observed_label_count"] == 5.0
    assert metrics["data_effective_label_weight"] == 5.0
    assert bool(metrics["data_valid"])

    fixed = phx.uq.FixedSupervisedLikelihood(term, lambda _: {"labels": field})
    np.testing.assert_allclose(fixed.per_case_log_prob(None), -expected)


def test_soft_classification_matches_expected_log_score_and_is_not_a_likelihood():
    logits = jnp.asarray([[2.0, -1.0, 0.0], [-0.5, 1.0, 0.3]])
    target = jnp.asarray([[0.7, 0.2, 0.1], [0.0, 0.4, 0.6]])
    domain = phx.domain.DatasetDomain(logits)

    @domain.Function("data")
    def field(row):
        return row

    term = phx.terms.SupervisedSoftClassificationTerm(
        "soft",
        domain.component(),
        target,
        phx.ml.TargetSchema("multiclass", class_labels=("a", "b", "c")),
        sampling=phx.domain.PointSampling(2, design="uniform"),
    )
    expected = -jnp.mean(jnp.sum(target * jax.nn.log_softmax(logits, axis=-1), axis=-1))
    batch = term.observed_batch()
    np.testing.assert_allclose(term.loss({"soft": field}, batch=batch), expected)
    metrics = term.data_metrics({"soft": field}, batch=batch)
    assert "data_accuracy" not in metrics
    assert bool(metrics["data_valid"])
    with pytest.raises(TypeError, match="supervised likelihood"):
        phx.uq.FixedSupervisedLikelihood(term, lambda _: {"soft": field})


def test_focal_gamma_zero_matches_hard_classification_term():
    logits = jnp.asarray([[2.0, -1.0], [-0.5, 1.0], [0.2, 0.7]])
    labels = jnp.asarray([0, 1, 1])
    domain = phx.domain.DatasetDomain(logits)

    @domain.Function("data")
    def field(row):
        return row

    schema = phx.ml.TargetSchema("multiclass", class_labels=("left", "right"))
    hard = phx.terms.SupervisedClassificationTerm(
        "u",
        domain.component(),
        labels,
        schema,
        sampling=phx.domain.PointSampling(3, design="uniform"),
    )
    focal = phx.terms.SupervisedFocalClassificationTerm(
        "u",
        domain.component(),
        labels,
        schema,
        gamma=0.0,
        sampling=phx.domain.PointSampling(3, design="uniform"),
    )
    batch = hard.observed_batch()
    np.testing.assert_allclose(
        focal.loss({"u": field}, batch=batch), hard.loss({"u": field}, batch=batch)
    )
    assert "data_focal_risk" in focal.data_metrics({"u": field}, batch=batch)


def test_ordinal_term_matches_cumulative_link_likelihood_and_metrics():
    location = jnp.asarray([-2.0, 0.0, 1.0, 3.0])
    labels = jnp.asarray([0, 1, 2, 3])
    thresholds = jnp.asarray([-1.0, 0.5, 2.0])
    domain = phx.domain.DatasetDomain(location[:, None])

    @domain.Function("data")
    def field(row):
        return row[0]

    term = phx.terms.SupervisedOrdinalClassificationTerm(
        "severity",
        domain.component(),
        labels,
        phx.ml.TargetSchema(
            "ordinal", class_labels=("none", "mild", "moderate", "severe")
        ),
        thresholds=thresholds,
        sampling=phx.domain.PointSampling(4, design="uniform"),
    )
    likelihood = phx.uq.OrdinalCumulativeLinkLikelihood(thresholds)
    expected = -jnp.mean(likelihood.log_prob(location, labels))
    batch = term.observed_batch()
    np.testing.assert_allclose(term.loss({"severity": field}, batch=batch), expected)
    metrics = term.data_metrics({"severity": field}, batch=batch)
    assert "data_expected_rank" in metrics
    assert "data_rank_mean_absolute_error" in metrics
    assert bool(metrics["data_valid"])


def test_zero_weight_focal_short_circuits_nonfinite_logits_and_gradient():
    domain = phx.domain.DatasetDomain(jnp.asarray([[1.0], [2.0]]))
    term = phx.terms.SupervisedFocalClassificationTerm(
        "u",
        domain.component(),
        jnp.asarray([0, 1]),
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
