#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.discretization import UniformAxisSpec
from phydrax.ml._classification import ClassificationObjective
from phydrax.terms._dense_classification import DenseSiteClassificationTerm


def _binary_problem(*, target_mask=None, sample_weight=None, where=None):
    cases = jnp.asarray([-1.0, 0.5, 1.5])
    data = phx.domain.DatasetDomain(cases)
    domain = data @ phx.domain.Interval1d(0.0, 1.0)
    component = domain.component(where={} if where is None else {"x": where})
    nodes = jnp.linspace(0.0, 1.0, 5)
    logits = cases[:, None] + nodes[None, :]
    targets = (logits >= 0.75).astype(jnp.int32)

    @domain.Function("data", "x")
    def field(row, x):
        return row + x[0]

    term = DenseSiteClassificationTerm(
        "u",
        component,
        targets,
        phx.ml.TargetSchema("binary", class_labels=(0, 1)),
        sampling=phx.domain.GridSampling(
            {"x": UniformAxisSpec(5)},
            dense=phx.domain.PointSampling(3, design="uniform"),
        ),
        objective=ClassificationObjective.nll(),
        target_mask=target_mask,
        sample_weight=sample_weight,
        site_reduction="mean",
    )
    return term, field, logits, targets


def test_dense_site_named_axes_and_exact_binary_nll():
    term, field, logits, targets = _binary_problem()
    batch = term.observed_batch(key=jr.key(1))
    per_case = term.per_case_loss({"u": field}, batch)
    expected = jnp.mean((jax.nn.softplus(logits) - targets * logits)[:, 1:-1], axis=1)

    assert batch.case_axis == "__phydra_blk__data"
    assert batch.site_axes == ("__phydra_sep__x__0",)
    assert batch.target.shape == (3, 5)
    assert jnp.allclose(per_case, expected)


def test_dense_masks_sanitize_poisoned_targets_before_geometry_weighting():
    target_mask = jnp.ones((3, 5), dtype=bool).at[:, 0].set(False)
    term, field, logits, targets = _binary_problem(
        target_mask=target_mask,
        where=lambda x: x[0] <= 0.75,
    )
    term = eqx.tree_at(lambda item: item.values, term, term.values.at[:, 0].set(-99))
    batch = term.observed_batch()
    per_case = term.per_case_loss({"u": field}, batch)
    active = jnp.asarray([False, True, True, True, False])
    pointwise = jax.nn.softplus(logits) - targets * logits

    assert jnp.all(jnp.isfinite(per_case))
    assert jnp.allclose(per_case, jnp.mean(pointwise[:, active], axis=1))


def test_dense_case_weights_are_separate_from_support_reduction():
    case_weight = jnp.asarray([1.0, 2.0, 5.0])
    term, field, _, _ = _binary_problem(sample_weight=case_weight)
    batch = term.observed_batch()
    per_case = term.per_case_loss({"u": field}, batch)

    assert jnp.allclose(
        term.loss({"u": field}, batch=batch),
        jnp.sum(case_weight * per_case) / jnp.sum(case_weight),
    )


def test_dense_hard_soft_and_multilabel_target_shapes():
    rows = jnp.asarray([0.0, 1.0])
    data = phx.domain.DatasetDomain(rows)
    domain = data @ phx.domain.Interval1d(0.0, 1.0)
    component = domain.component()
    sampling = phx.domain.GridSampling(
        {"x": UniformAxisSpec(4)},
        dense=phx.domain.PointSampling(2, design="uniform"),
    )
    hard = jnp.asarray([[0, 1, 2, 1], [2, 1, 0, 1]])
    soft = jax.nn.one_hot(hard, 3) * 0.8 + 0.2 / 3.0

    @domain.Function("data", "x")
    def categorical(row, x):
        return jnp.stack((row - x[0], x[0], 1.0 - row + x[0]))

    schema = phx.ml.TargetSchema("multiclass", class_labels=("a", "b", "c"))
    hard_term = DenseSiteClassificationTerm(
        "u",
        component,
        hard,
        schema,
        sampling=sampling,
        objective=ClassificationObjective.nll(),
    )
    soft_term = DenseSiteClassificationTerm(
        "u",
        component,
        soft,
        schema,
        sampling=sampling,
        objective=ClassificationObjective.soft_cross_entropy(),
    )
    multilabel_term = DenseSiteClassificationTerm(
        "u",
        component,
        (soft > 0.25).astype(jnp.int32),
        phx.ml.TargetSchema("multilabel", names=("a", "b", "c")),
        sampling=sampling,
        objective=ClassificationObjective.nll(),
    )

    assert hard_term.per_case_loss(
        {"u": categorical}, hard_term.observed_batch()
    ).shape == (2,)
    assert soft_term.per_case_loss(
        {"u": categorical}, soft_term.observed_batch()
    ).shape == (2,)
    assert multilabel_term.per_case_loss(
        {"u": categorical}, multilabel_term.observed_batch()
    ).shape == (2,)


def test_dense_ordinal_active_invalid_label_remains_infinite():
    rows = jnp.asarray([0.0, 1.0])
    data = phx.domain.DatasetDomain(rows)
    domain = data @ phx.domain.Interval1d(0.0, 1.0)

    @domain.Function("data", "x")
    def location(row, x):
        return row + x[0]

    term = DenseSiteClassificationTerm(
        "u",
        domain.component(),
        jnp.asarray([[0, 1, 2], [2, 7, 0]]),
        phx.ml.TargetSchema("ordinal", class_labels=("low", "mid", "high")),
        sampling=phx.domain.GridSampling(
            {"x": UniformAxisSpec(3)},
            dense=phx.domain.PointSampling(2, design="uniform"),
        ),
        objective=ClassificationObjective.nll(thresholds=(-0.5, 0.5)),
    )
    per_case = term.per_case_loss({"u": location}, term.observed_batch())

    assert jnp.isfinite(per_case[0])
    assert jnp.isinf(per_case[1])


def test_dense_integral_refinement_and_jit_gradient():
    data = phx.domain.DatasetDomain(jnp.asarray([0.0]))
    domain = data @ phx.domain.Interval1d(0.0, 2.0)

    def make_term(count):
        return DenseSiteClassificationTerm(
            "u",
            domain.component(),
            jnp.ones((1, count), dtype=jnp.int32),
            phx.ml.TargetSchema("binary", class_labels=(0, 1)),
            sampling=phx.domain.GridSampling(
                {"x": UniformAxisSpec(count)},
                dense=phx.domain.PointSampling(1, design="uniform"),
            ),
            site_reduction="integral",
        )

    def evaluate(scale, term, batch):
        @domain.Function("data", "x")
        def field(row, x):
            del row, x
            return scale

        return term.loss({"u": field}, batch=batch)

    coarse = make_term(5)
    fine = make_term(17)
    coarse_batch = coarse.observed_batch()
    coarse_value = evaluate(0.3, coarse, coarse_batch)
    fine_value = evaluate(0.3, fine, fine.observed_batch())
    exact_integral = 2.0 * jax.nn.softplus(-0.3)
    assert jnp.abs(fine_value - exact_integral) < jnp.abs(coarse_value - exact_integral)
    gradient = jax.jit(jax.grad(lambda scale: evaluate(scale, coarse, coarse_batch)))(0.3)
    assert jnp.isfinite(gradient)


def test_dense_rejects_resampled_site_coordinates_and_shape_mismatch():
    data = phx.domain.DatasetDomain(jnp.asarray([0.0, 1.0]))
    component = (data @ phx.domain.Interval1d(0.0, 1.0)).component()
    schema = phx.ml.TargetSchema("binary", class_labels=(0, 1))

    with pytest.raises(ValueError, match="fixed explicit axis"):
        DenseSiteClassificationTerm(
            "u",
            component,
            jnp.zeros((2, 4), dtype=jnp.int32),
            schema,
            sampling=phx.domain.GridSampling(
                {"x": 4}, dense=phx.domain.PointSampling(2, design="uniform")
            ),
        )
    with pytest.raises(ValueError, match="align exactly"):
        DenseSiteClassificationTerm(
            "u",
            component,
            jnp.zeros((2, 3), dtype=jnp.int32),
            schema,
            sampling=phx.domain.GridSampling(
                {"x": UniformAxisSpec(4)},
                dense=phx.domain.PointSampling(2, design="uniform"),
            ),
        )


def test_dense_classification_composes_with_physics_residual_in_solver():
    term, field, _, _ = _binary_problem()
    residual = phx.conditions.Residual("u", term.component, lambda candidate: candidate)
    residual_term = phx.terms.ResidualPenalty(
        residual,
        phx.integration.per_step(
            phx.integration.mean_over(term.component),
            phx.domain.PointSampling(
                8,
                layout=phx.domain.SampleLayout((("data", "x"),)),
                design="uniform",
            ),
        ),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": field},
        terms=(term, residual_term),
    )

    assert jnp.isfinite(solver.loss(key=jr.key(12)))


def test_dense_zero_weight_skips_nonfinite_predictions():
    term, _, _, _ = _binary_problem()
    disabled = eqx.tree_at(lambda value: value.weight, term, jnp.asarray(0.0))

    @term.component.domain.Function("data", "x")
    def poisoned(row, x):
        return row + x[0] + jnp.nan

    assert disabled.loss({"u": poisoned}, batch=disabled.observed_batch()) == 0.0
