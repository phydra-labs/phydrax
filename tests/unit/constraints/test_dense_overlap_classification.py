#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax.discretization import UniformAxisSpec
from phydrax.ml._classification import ClassificationObjective
from phydrax.ml._overlap import (
    dice_score,
    jaccard_score,
    OverlapScoreConfig,
    reduce_overlap_score,
    tversky_score,
)
from phydrax.terms._dense_classification import (
    _categorical_statistics,
    DenseOverlapClassificationTerm,
)


def test_overlap_kernels_match_weighted_sufficient_statistics():
    intersection = jnp.asarray([3.0, 2.0])
    prediction = jnp.asarray([5.0, 4.0])
    target = jnp.asarray([6.0, 3.0])

    assert jnp.allclose(
        dice_score(intersection, prediction, target),
        2.0 * intersection / (prediction + target),
    )
    assert jnp.allclose(
        jaccard_score(intersection, prediction, target),
        intersection / (prediction + target - intersection),
    )
    assert jnp.allclose(
        tversky_score(intersection, prediction, target, alpha=0.25, beta=0.75),
        intersection
        / (
            intersection
            + 0.25 * (prediction - intersection)
            + 0.75 * (target - intersection)
        ),
    )


def test_micro_macro_and_support_weighted_reductions_diverge():
    intersection = jnp.asarray([[9.0, 1.0]])
    prediction = jnp.asarray([[10.0, 10.0]])
    target = jnp.asarray([[10.0, 2.0]])
    micro = reduce_overlap_score(
        intersection,
        prediction,
        target,
        OverlapScoreConfig("dice", class_reduction="micro"),
    )
    macro = reduce_overlap_score(
        intersection,
        prediction,
        target,
        OverlapScoreConfig("dice", class_reduction="macro"),
    )
    weighted = reduce_overlap_score(
        intersection,
        prediction,
        target,
        OverlapScoreConfig("dice", class_reduction="support_weighted"),
    )

    assert jnp.allclose(micro, 20.0 / 32.0)
    assert jnp.allclose(macro, (0.9 + 1.0 / 6.0) / 2.0)
    assert jnp.allclose(weighted, (10.0 * 0.9 + 2.0 / 6.0) / 12.0)
    assert not jnp.allclose(micro, macro)


def test_overlap_empty_policies_cover_both_empty_and_one_sided_empty():
    zero = jnp.asarray([0.0])
    one = jnp.asarray([1.0])

    assert jnp.isnan(dice_score(zero, zero, zero, empty="nan"))[0]
    assert dice_score(zero, zero, zero, empty="one")[0] == 1.0
    assert dice_score(zero, zero, zero, empty="zero")[0] == 0.0
    assert dice_score(zero, one, zero, empty="one")[0] == 0.0
    assert dice_score(zero, zero, one, empty="one")[0] == 0.0

    ignored = reduce_overlap_score(
        jnp.asarray([[0.0, 1.0]]),
        jnp.asarray([[0.0, 1.0]]),
        jnp.asarray([[0.0, 1.0]]),
        OverlapScoreConfig("dice", class_reduction="macro", empty="ignore"),
    )
    assert jnp.allclose(ignored, 1.0)
    all_ignored = reduce_overlap_score(
        jnp.zeros((1, 2)),
        jnp.zeros((1, 2)),
        jnp.zeros((1, 2)),
        OverlapScoreConfig("dice", class_reduction="macro", empty="ignore"),
    )
    assert jnp.isnan(all_ignored)[0]


def _multiclass_overlap_problem(*, soft=False, sample_weight=None, target_mask=None):
    rows = jnp.asarray([0.0, 1.0])
    data = phx.domain.DatasetDomain(rows)
    domain = data @ phx.domain.Interval1d(0.0, 1.0)
    nodes = jnp.linspace(0.0, 1.0, 4)
    hard = jnp.asarray([[0, 0, 1, 2], [2, 1, 1, 0]], dtype=jnp.int32)
    target = jax.nn.one_hot(hard, 3) if soft else hard

    @domain.Function("data", "x")
    def logits(row, x):
        return jnp.stack((1.0 - x[0], row + x[0], x[0] - row))

    term = DenseOverlapClassificationTerm(
        "u",
        domain.component(),
        target,
        phx.ml.TargetSchema("multiclass", class_labels=(0, 1, 2)),
        OverlapScoreConfig("dice", class_reduction="macro", empty="zero"),
        sampling=phx.domain.GridSampling(
            {"x": UniformAxisSpec(4)},
            dense=phx.domain.PointSampling(2, design="uniform"),
        ),
        objective=ClassificationObjective.nll(),
        support_measure="physical",
        target_mask=target_mask,
        sample_weight=sample_weight,
    )
    return term, logits, nodes


def test_dense_overlap_hard_gather_and_soft_targets_share_support_contract():
    hard_term, logits, _ = _multiclass_overlap_problem()
    soft_term, _, _ = _multiclass_overlap_problem(soft=True)
    hard_score = hard_term.per_case_score({"u": logits}, hard_term.observed_batch())
    soft_score = soft_term.per_case_score({"u": logits}, soft_term.observed_batch())

    assert hard_score.shape == (2,)
    assert soft_score.shape == (2,)
    assert jnp.all(jnp.isfinite(hard_score))
    assert jnp.all(jnp.isfinite(soft_score))
    hard_jaxpr = str(
        jax.make_jaxpr(
            lambda values, labels, weights: _categorical_statistics(
                values, labels, weights
            )
        )(
            jnp.full((1, 4, 3), 1.0 / 3.0),
            jnp.asarray([[0, 1, 2, 1]]),
            jnp.ones((1, 4)),
        )
    )
    assert "one_hot" not in hard_jaxpr


def test_dense_overlap_case_weights_follow_ratio_not_support_pooling():
    case_weight = jnp.asarray([1.0, 4.0])
    term, logits, _ = _multiclass_overlap_problem(sample_weight=case_weight)
    batch = term.observed_batch()
    score = term.per_case_score({"u": logits}, batch)
    expected = jnp.sum(case_weight * (1.0 - score)) / jnp.sum(case_weight)

    assert jnp.allclose(term.loss({"u": logits}, batch=batch), expected)


def test_dense_overlap_masking_and_refinement_are_geometry_owned():
    mask = jnp.ones((2, 4), dtype=bool).at[:, 0].set(False)
    term, logits, _ = _multiclass_overlap_problem(target_mask=mask)
    score = term.per_case_score({"u": logits}, term.observed_batch())

    assert jnp.all(jnp.isfinite(score))
    assert jnp.all((score >= 0.0) & (score <= 1.0))

    data = phx.domain.DatasetDomain(jnp.asarray([0.0]))
    domain = data @ phx.domain.Interval1d(0.0, 2.0)

    @domain.Function("data", "x")
    def constant_logits(row, x):
        del row, x
        return 0.4

    def refined_score(count):
        refined = DenseOverlapClassificationTerm(
            "u",
            domain.component(),
            jnp.ones((1, count)),
            phx.ml.TargetSchema("binary", class_labels=(0, 1)),
            OverlapScoreConfig("dice", class_reduction="micro"),
            sampling=phx.domain.GridSampling(
                {"x": UniformAxisSpec(count)},
                dense=phx.domain.PointSampling(1, design="uniform"),
            ),
            support_measure="physical",
        )
        return refined.per_case_score({"u": constant_logits}, refined.observed_batch())

    assert jnp.allclose(refined_score(5), refined_score(17), rtol=1e-5)


def test_dense_overlap_jit_gradient_and_ordinal_probabilities():
    rows = jnp.asarray([0.0])
    data = phx.domain.DatasetDomain(rows)
    domain = data @ phx.domain.Interval1d(0.0, 1.0)
    binary = DenseOverlapClassificationTerm(
        "u",
        domain.component(),
        jnp.asarray([[0.0, 0.0, 1.0, 1.0]]),
        phx.ml.TargetSchema("binary", class_labels=(0, 1)),
        OverlapScoreConfig("jaccard", class_reduction="macro"),
        sampling=phx.domain.GridSampling(
            {"x": UniformAxisSpec(4)},
            dense=phx.domain.PointSampling(1, design="uniform"),
        ),
    )
    batch = binary.observed_batch()

    def loss(scale):
        @domain.Function("data", "x")
        def logits(row, x):
            del row
            return scale * (x[0] - 0.5)

        return binary.loss({"u": logits}, batch=batch)

    assert jnp.isfinite(jax.jit(jax.grad(loss))(jnp.asarray(1.0)))

    ordinal = DenseOverlapClassificationTerm(
        "u",
        domain.component(),
        jnp.asarray([[0, 1, 1, 2]]),
        phx.ml.TargetSchema("ordinal", class_labels=("low", "mid", "high")),
        OverlapScoreConfig("dice", class_reduction="micro"),
        sampling=phx.domain.GridSampling(
            {"x": UniformAxisSpec(4)},
            dense=phx.domain.PointSampling(1, design="uniform"),
        ),
        objective=ClassificationObjective.nll(thresholds=(-0.25, 0.25)),
    )

    @domain.Function("data", "x")
    def location(row, x):
        del row
        return x[0] - 0.5

    ordinal_score = ordinal.per_case_score({"u": location}, ordinal.observed_batch())
    assert ordinal_score.shape == (1,)
    assert jnp.all(jnp.isfinite(ordinal_score))


def test_overlap_config_is_final_json_safe_and_composes_as_separate_scalar_term():
    config = OverlapScoreConfig(
        "tversky", class_reduction="support_weighted", alpha=0.3, beta=0.7
    )
    term, logits, _ = _multiclass_overlap_problem()
    overlap = term.loss({"u": logits}, batch=term.observed_batch())
    nll = jnp.asarray(0.4)

    assert config.to_dict() == {
        "kind": "tversky",
        "class_reduction": "support_weighted",
        "empty": "zero",
        "smooth": 0.0,
        "alpha": 0.3,
        "beta": 0.7,
    }
    assert (nll + overlap).shape == ()
