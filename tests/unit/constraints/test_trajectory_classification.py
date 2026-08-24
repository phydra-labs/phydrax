#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.domain import (
    IrregularTrajectoryDatasetDomain,
    SampleLayout,
    TrajectoryDatasetDomain,
)
from phydrax.ml import ClassificationObjective, TargetSchema
from phydrax.operators.differential import partial_t
from phydrax.terms import (
    RaggedTimeSeriesClassificationTerm,
    RaggedTimeSeriesDataTerm,
    TrajectoryCaseClassificationTerm,
    TrajectoryCaseDataTerm,
)
from phydrax.terms._trajectory_classification import TrajectoryCaseClassificationBatch


def _regular_domain(*, measure="case_time_probability"):
    return TrajectoryDatasetDomain(
        jnp.asarray([[0.0], [1.0], [2.0]]),
        jnp.asarray([2, 4, 3]),
        dt=0.5,
        measure=measure,
    )


def _irregular_domain(*, measure="case_time_probability"):
    return IrregularTrajectoryDatasetDomain(
        jnp.asarray([[0.0], [1.0], [2.0]]),
        jnp.asarray(
            [
                [0.0, 0.2, 0.7, 99.0],
                [0.1, 0.4, 1.2, 1.8],
                [-0.2, 0.3, 99.0, 99.0],
            ]
        ),
        jnp.asarray([3, 4, 2]),
        measure=measure,
    )


def _paired_sampling(count=24):
    return phx.domain.PointSampling(
        count,
        layout=SampleLayout((("data", "t"),)),
        design="uniform",
    )


def _binary_case_term(domain, *, case_time="start", **kwargs):
    return TrajectoryCaseClassificationTerm(
        "classify",
        domain.component(),
        jnp.asarray([0, 1, 0], dtype=jnp.int32),
        TargetSchema("binary", class_labels=("no", "yes")),
        sampling=phx.domain.PointSampling(18, design="uniform"),
        case_time=case_time,
        **kwargs,
    )


@pytest.mark.parametrize("make_domain", [_regular_domain, _irregular_domain])
def test_case_classification_start_and_end_times(make_domain):
    domain = make_domain()
    start_batch = _binary_case_term(domain, case_time="start").sample(key=jr.key(0))
    end_batch = _binary_case_term(domain, case_time="end").sample(key=jr.key(1))

    expected_start = (
        domain.start_times[start_batch.case_indices]
        if isinstance(domain, IrregularTrajectoryDatasetDomain)
        else jnp.full(start_batch.times.shape, domain.start)
    )
    assert jnp.allclose(start_batch.times, expected_start)
    assert jnp.allclose(end_batch.times, domain.end_times[end_batch.case_indices])
    assert jnp.issubdtype(start_batch.target.dtype, jnp.integer)


@pytest.mark.parametrize(
    ("make_domain", "fixed_time"),
    [(_regular_domain, 0.5), (_irregular_domain, 0.25)],
)
def test_case_classification_fixed_time_samples_only_valid_cases(make_domain, fixed_time):
    domain = make_domain()
    batch = _binary_case_term(domain, case_time=fixed_time).sample(key=jr.key(2))

    assert jnp.allclose(batch.times, fixed_time)
    starts = (
        domain.start_times[batch.case_indices]
        if isinstance(domain, IrregularTrajectoryDatasetDomain)
        else jnp.full(batch.times.shape, domain.start)
    )
    assert jnp.all(starts <= batch.times)
    assert jnp.all(batch.times <= domain.end_times[batch.case_indices])


@pytest.mark.parametrize("make_domain", [_regular_domain, _irregular_domain])
def test_hard_ragged_nearest_lookup_preserves_labels_and_skips_padding(make_domain):
    domain = make_domain()
    targets = jnp.full((domain.size, domain.max_length), 99, dtype=jnp.int32)
    valid = jnp.arange(domain.max_length)[None, :] < domain.lengths[:, None]
    labels = jnp.arange(domain.max_length)[None, :] % 2
    targets = jnp.where(valid, labels, targets)
    term = RaggedTimeSeriesClassificationTerm(
        "classify",
        domain.component(),
        targets,
        TargetSchema("binary", class_labels=(0, 1)),
        sampling=_paired_sampling(64),
        selection="observation_uniform",
        interpolation="nearest",
    )

    batch = term.sample(key=jr.key(3))
    assert jnp.issubdtype(batch.target.dtype, jnp.integer)
    assert batch.target.shape == batch.time_indices.shape
    assert jnp.all(batch.time_indices < domain.lengths[batch.case_indices])
    assert jnp.array_equal(batch.target, targets[batch.case_indices, batch.time_indices])
    assert not bool(jnp.any(batch.target == 99))


def test_shared_trajectory_data_batches_preserve_discrete_target_dtype():
    domain = _regular_domain()
    case_term = TrajectoryCaseDataTerm(
        "value",
        domain.component(),
        jnp.asarray([0, 1, 0], dtype=jnp.int32),
        sampling=phx.domain.PointSampling(8, design="uniform"),
    )
    ragged_term = RaggedTimeSeriesDataTerm(
        "value",
        domain.component(),
        jnp.zeros((domain.size, domain.max_length), dtype=bool),
        sampling=_paired_sampling(8),
        selection="observation_uniform",
    )

    assert jnp.issubdtype(case_term.sample(key=jr.key(12)).target.dtype, jnp.integer)
    assert ragged_term.sample(key=jr.key(13)).target.dtype == jnp.bool_


@pytest.mark.parametrize(
    "selection", ["observation_uniform", "case_uniform", "case_time_uniform"]
)
def test_ragged_selection_policies_respect_case_subset(selection):
    domain = _regular_domain()
    targets = jnp.zeros((domain.size, domain.max_length), dtype=bool)
    term = RaggedTimeSeriesClassificationTerm(
        "classify",
        domain.component(),
        targets,
        TargetSchema("binary", class_labels=(0, 1)),
        sampling=_paired_sampling(32),
        selection=selection,
        case_indices=jnp.asarray([2], dtype=jnp.int32),
    )

    batch = term.sample(key=jr.key(4))
    assert jnp.all(batch.case_indices == 2)
    assert jnp.all(batch.time_indices < domain.lengths[batch.case_indices])


def test_hard_targets_reject_linear_interpolation():
    domain = _regular_domain()
    targets = jnp.zeros((domain.size, domain.max_length), dtype=jnp.int32)
    with pytest.raises(ValueError, match="Hard.*nearest"):
        RaggedTimeSeriesClassificationTerm(
            "classify",
            domain.component(),
            targets,
            TargetSchema("binary", class_labels=(0, 1)),
            sampling=_paired_sampling(),
            objective=ClassificationObjective.soft_cross_entropy(),
            selection="case_time_uniform",
            interpolation="linear",
        )


def test_soft_multiclass_linear_interpolation_stays_on_simplex():
    domain = _regular_domain()
    time = jnp.arange(domain.max_length, dtype=float)
    probability = jnp.broadcast_to(
        jnp.stack((0.2 + 0.1 * time, 0.8 - 0.1 * time), axis=-1)[None, ...],
        (domain.size, domain.max_length, 2),
    )
    term = RaggedTimeSeriesClassificationTerm(
        "classify",
        domain.component(),
        probability,
        TargetSchema("multiclass", class_labels=("left", "right")),
        sampling=_paired_sampling(48),
        objective=ClassificationObjective.soft_cross_entropy(),
        selection="case_time_uniform",
        interpolation="linear",
    )

    batch = term.sample(key=jr.key(5))
    assert batch.target.shape == (48, 2)
    assert jnp.all(batch.target >= 0.0)
    assert jnp.allclose(jnp.sum(batch.target, axis=-1), 1.0, atol=1e-6)


def test_soft_multiclass_linear_rejects_invalid_active_simplex():
    domain = _regular_domain()
    probability = jnp.full((domain.size, domain.max_length, 3), 1.0 / 3.0)
    probability = probability.at[1, 1].set(jnp.asarray([0.8, 0.8, -0.6]))
    with pytest.raises(ValueError, match="probability simplex"):
        RaggedTimeSeriesClassificationTerm(
            "classify",
            domain.component(),
            probability,
            TargetSchema("multiclass", class_labels=(0, 1, 2)),
            sampling=_paired_sampling(),
            objective=ClassificationObjective.soft_cross_entropy(),
            selection="case_time_uniform",
            interpolation="linear",
        )


def test_multilabel_case_time_grid_retains_case_time_and_label_axes():
    domain = _regular_domain()
    targets = jnp.zeros((domain.size, domain.max_length, 3), dtype=bool)
    targets = targets.at[..., 0].set(True)
    term = RaggedTimeSeriesClassificationTerm(
        "classify",
        domain.component(),
        targets,
        TargetSchema("multilabel", names=("hot", "wet", "fast")),
        sampling=phx.domain.PointSampling(
            (2, 5),
            layout=SampleLayout((("data",), ("t",))),
            design="uniform",
        ),
        selection="case_uniform",
    )

    batch = term.sample(key=jr.key(6))
    assert batch.times.shape == (2, 5)
    assert batch.target.shape == (2, 5, 3)
    assert batch.target.dtype == jnp.bool_
    assert batch.sample_weight.shape == (2, 5)
    assert batch.geometry_weight.shape == (2, 5)

    @domain.Function("data", "t")
    def logits(data, time):
        del data, time
        return jnp.zeros((3,))

    loss = term.loss({"classify": logits}, batch=batch)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def _all_case_batch(term):
    domain = term.domain
    case_indices = jnp.arange(domain.size, dtype=jnp.int32)
    times = jnp.full((domain.size,), domain.start)
    time_indices = jnp.zeros((domain.size,), dtype=jnp.int32)
    layout = term.sampling.layout
    assert layout is not None
    points = domain.points_from_case_time(
        case_indices,
        times,
        structure=layout,
        time_indices=time_indices,
    )
    return TrajectoryCaseClassificationBatch(
        points=points,
        target=term.values,
        target_mask=term.target_mask,
        sample_weight=term.sample_weight,
        geometry_weight=jnp.ones((domain.size,)),
        case_indices=case_indices,
        times=times,
    )


def test_multilabel_mean_sums_observed_labels_before_averaging_cases():
    domain = _regular_domain()
    targets = jnp.zeros((domain.size, 3), dtype=jnp.int32)
    target_mask = jnp.asarray(
        [[True, False, False], [True, True, False], [False, False, False]]
    )
    term = TrajectoryCaseClassificationTerm(
        "classify",
        domain.component(),
        targets,
        TargetSchema("multilabel", names=("hot", "wet", "fast")),
        sampling=phx.domain.PointSampling(3, design="uniform"),
        target_mask=target_mask,
    )
    batch = _all_case_batch(term)

    @domain.Function("data", "t")
    def zero_logits(data, time):
        del data, time
        return jnp.zeros((3,))

    loss = term.loss({"classify": zero_logits}, batch=batch)
    assert jnp.allclose(loss, 1.5 * jnp.log(2.0), atol=1e-7)


def test_target_mask_and_case_weights_define_statistical_mean_and_sum():
    domain = _regular_domain()
    common = dict(
        target_mask=jnp.asarray([True, False, True]),
        sample_weight=jnp.asarray([1.0, 2.0, 3.0]),
    )
    mean_term = _binary_case_term(domain, reduction="mean", **common)
    sum_term = _binary_case_term(domain, reduction="sum", **common)
    batch = _all_case_batch(mean_term)

    @domain.Function("data", "t")
    def zero_logit(data, time):
        del data, time
        return 0.0

    mean_loss = mean_term.loss({"classify": zero_logit}, batch=batch)
    sum_loss = sum_term.loss({"classify": zero_logit}, batch=batch)
    assert jnp.allclose(mean_loss, jnp.log(2.0), atol=1e-7)
    assert jnp.allclose(sum_loss, 4.0 * jnp.log(2.0), atol=1e-7)


def test_masked_invalid_hard_label_is_inert_but_active_invalid_label_is_infinite():
    domain = _regular_domain()
    targets = jnp.asarray([0, 99, 1], dtype=jnp.int32)
    schema = TargetSchema("multiclass", class_labels=(0, 1, 2))
    masked = TrajectoryCaseClassificationTerm(
        "classify",
        domain.component(),
        targets,
        schema,
        sampling=phx.domain.PointSampling(3, design="uniform"),
        target_mask=jnp.asarray([True, False, True]),
    )
    active = TrajectoryCaseClassificationTerm(
        "classify",
        domain.component(),
        targets,
        schema,
        sampling=phx.domain.PointSampling(3, design="uniform"),
        target_mask=jnp.asarray([True, True, True]),
    )
    masked_batch = _all_case_batch(masked)
    active_batch = TrajectoryCaseClassificationBatch(
        points=masked_batch.points,
        target=active.values,
        target_mask=active.target_mask,
        sample_weight=active.sample_weight,
        geometry_weight=masked_batch.geometry_weight,
        case_indices=masked_batch.case_indices,
        times=masked_batch.times,
    )

    @domain.Function("data", "t")
    def logits(data, time):
        del data, time
        return jnp.zeros((3,))

    assert jnp.isfinite(masked.loss({"classify": logits}, batch=masked_batch))
    assert jnp.isinf(active.loss({"classify": logits}, batch=active_batch))


def test_focal_objective_preserves_case_shape_and_returns_scalar():
    domain = _regular_domain()
    term = TrajectoryCaseClassificationTerm(
        "classify",
        domain.component(),
        jnp.asarray([0, 1, 0]),
        TargetSchema("binary", class_labels=(0, 1)),
        sampling=phx.domain.PointSampling(9, design="uniform"),
        objective=ClassificationObjective.focal(gamma=1.5, alpha=0.25),
    )

    @domain.Function("data", "t")
    def logits(data, time):
        del time
        return data[0] - 1.0

    batch = term.sample(key=jr.key(11))
    loss = term.loss({"classify": logits}, batch=batch)
    assert batch.target.shape == batch.case_indices.shape
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_ordinal_case_classification_uses_scalar_latent_and_ordered_thresholds():
    domain = _regular_domain()
    term = TrajectoryCaseClassificationTerm(
        "classify",
        domain.component(),
        jnp.asarray([0, 1, 2]),
        TargetSchema("ordinal", class_labels=("low", "medium", "high")),
        sampling=phx.domain.PointSampling(12, design="uniform"),
        objective=ClassificationObjective.nll(thresholds=(-0.5, 0.75)),
    )

    @domain.Function("data", "t")
    def latent(data, time):
        del time
        return data[0] - 1.0

    loss = term.loss({"classify": latent}, key=jr.key(7))
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_physical_measure_requires_sum_and_preserves_trajectory_mass():
    domain = TrajectoryDatasetDomain(
        jnp.asarray([[0.0], [1.0], [2.0]]),
        jnp.asarray([3, 3, 3]),
        dt=0.5,
        measure="time_integral_sum",
    )
    targets = jnp.zeros((domain.size, domain.max_length), dtype=bool)
    with pytest.raises(ValueError, match="Physical.*sum"):
        RaggedTimeSeriesClassificationTerm(
            "classify",
            domain.component(),
            targets,
            TargetSchema("binary", class_labels=(0, 1)),
            sampling=_paired_sampling(),
            selection="case_time_uniform",
            measure="physical",
            reduction="mean",
        )

    term = RaggedTimeSeriesClassificationTerm(
        "classify",
        domain.component(),
        targets,
        TargetSchema("binary", class_labels=(0, 1)),
        sampling=_paired_sampling(32),
        selection="case_time_uniform",
        measure="physical",
        reduction="sum",
    )
    batch = term.sample(key=jr.key(8))
    assert jnp.allclose(jnp.sum(batch.geometry_weight), 3.0)


def test_classification_and_physics_terms_share_one_trajectory_function_mapping():
    domain = _regular_domain()
    classification = _binary_case_term(domain)

    @domain.Function("data", "t")
    def classify(data, time):
        del time
        return data[0] - 0.5

    @domain.Function("data", "t")
    def state(data, time):
        return data[0] + time

    condition = phx.conditions.Residual(
        ("state",),
        domain.component(),
        lambda state_fn: partial_t(state_fn, var="t") - 1.0,
    )
    source = phx.integration.per_step(
        phx.integration.mean_over(domain.component()),
        _paired_sampling(12),
    )
    physics = phx.terms.ResidualPenalty(condition, source)
    functions = {"classify": classify, "state": state}

    classification_loss = classification.loss(functions, key=jr.key(9))
    physics_loss = physics.loss(functions, key=jr.key(10))
    assert jnp.isfinite(classification_loss)
    assert jnp.allclose(physics_loss, 0.0, atol=1e-12)
    assert (classification_loss + physics_loss).shape == ()


def test_trajectory_zero_weight_skips_nonfinite_predictions():
    domain = _regular_domain()
    term = _binary_case_term(domain, weight=0.0)

    @domain.Function("data", "t")
    def poisoned(row, time):
        del row, time
        return jnp.nan

    assert term.loss({"classify": poisoned}, key=jr.key(0)) == 0.0
