#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.constraints import RaggedSeriesSupervisedConstraint
from phydrax.domain import ProductStructure, RaggedSeriesDatasetDomain


def _domain_and_targets():
    static = jnp.asarray([[1.0, 0.5], [2.0, -1.0], [3.0, 2.0]])
    series = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]],
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            [[11.0, 12.0], [99.0, 99.0], [99.0, 99.0]],
        ]
    )
    lengths = jnp.asarray([2, 3, 1], dtype=jnp.int32)
    domain = RaggedSeriesDatasetDomain(series, lengths, static=static, dt=0.25)
    valid_sum = jnp.asarray([4.0, 21.0, 11.0])
    targets = jnp.stack(
        (
            static[:, 0] + valid_sum,
            static[:, 1] - valid_sum,
        ),
        axis=-1,
    )
    return domain, targets


def test_ragged_series_supervised_constraint_matches_exact_vector_targets():
    domain, targets = _domain_and_targets()

    def exact(payload, *, key=None):
        del key
        series0 = payload.series[..., 0]
        valid_sum = jnp.sum(series0 * payload.mask.astype(series0.dtype), axis=1)
        return jnp.stack(
            (
                payload.static[:, 0] + valid_sum,
                payload.static[:, 1] - valid_sum,
            ),
            axis=-1,
        )

    u = domain.Function("data")(phx.nn.RaggedSeriesModel(exact))
    constraint = RaggedSeriesSupervisedConstraint(
        "u",
        domain.component(),
        targets,
        num_cases=16,
        structure=ProductStructure((("data",),)),
    )

    loss = constraint.loss({"u": u}, key=jr.key(0))
    metrics = constraint.data_metrics({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0, atol=1e-12)
    assert jnp.allclose(metrics["data_accuracy"], 1.0)
    assert jnp.allclose(metrics["data_relative_l2_error"], 0.0)


def test_ragged_series_supervised_constraint_samples_index_subset():
    domain, targets = _domain_and_targets()
    allowed = jnp.asarray([0, 2], dtype=jnp.int32)
    constraint = RaggedSeriesSupervisedConstraint(
        "u",
        domain.component(),
        targets,
        num_cases=20,
        indices=allowed,
    )

    batch = constraint.sample(key=jr.key(1))
    assert jnp.all(jnp.isin(batch.indices, allowed))
    assert jnp.allclose(batch.target, targets[batch.indices])


def test_ragged_series_supervised_constraint_samples_fixed_width_series_points():
    domain, targets = _domain_and_targets()
    constraint = RaggedSeriesSupervisedConstraint(
        "u",
        domain.component(),
        targets,
        num_cases=6,
        series_sampling="window_uniform",
        num_series_points=2,
    )

    batch = constraint.sample(key=jr.key(5))
    assert batch.points["data"]["series"].data.shape == (6, 2, 2)
    assert batch.points["data"]["mask"].data.shape == (6, 2)
    assert batch.points["data"]["sample_index"].data.shape == (6, 2)
    assert batch.target.shape == (6, 2)


def test_ragged_series_supervised_constraint_loss_uses_sampled_series_payload():
    domain, targets = _domain_and_targets()

    def sampled_model(payload, *, key=None):
        del key
        series0 = payload.series[..., 0]
        valid_sum = jnp.sum(series0 * payload.mask.astype(series0.dtype), axis=1)
        return jnp.stack((valid_sum, -valid_sum), axis=-1)

    u = domain.Function("data")(phx.nn.RaggedSeriesModel(sampled_model))
    constraint = RaggedSeriesSupervisedConstraint(
        "u",
        domain.component(),
        targets,
        num_cases=8,
        series_sampling="prefix",
        num_series_points=2,
    )

    loss = constraint.loss({"u": u}, key=jr.key(6))
    assert jnp.isfinite(loss)


def test_ragged_series_supervised_constraint_bucketed_covers_cases_once():
    domain, targets = _domain_and_targets()
    constraints = RaggedSeriesSupervisedConstraint.bucketed(
        "u",
        domain.component(),
        targets,
        num_cases=4,
        num_buckets=2,
        structure=ProductStructure((("data",),)),
        label="train",
    )

    assert len(constraints) == 2
    assert tuple(c.series_sampling for c in constraints) == ("prefix", "prefix")
    assert tuple(c.num_series_points for c in constraints) == (2, 3)
    assert tuple(c.num_points for c in constraints) == (3, 1)
    assert sum(c.num_points for c in constraints) == 4
    assert tuple(c.label for c in constraints) == ("train_bucket_1", "train_bucket_2")
    assert jnp.allclose(
        jnp.stack([c.weight for c in constraints]),
        jnp.asarray([2.0 / 3.0, 1.0 / 3.0]),
    )

    covered = jnp.sort(jnp.concatenate([c.indices for c in constraints]))
    assert jnp.array_equal(covered, jnp.asarray([0, 1, 2], dtype=jnp.int32))

    for constraint in constraints:
        batch = constraint.sample(key=jr.key(9))
        width = int(constraint.num_series_points)
        assert batch.points["data"]["series"].data.shape == (
            constraint.num_points,
            width,
            2,
        )
        assert jnp.all(jnp.isin(batch.indices, constraint.indices))
        assert jnp.all(domain.lengths[batch.indices] <= width)


def test_ragged_series_supervised_constraint_bucketed_accepts_length_edges():
    domain, targets = _domain_and_targets()
    constraints = RaggedSeriesSupervisedConstraint.bucketed(
        "u",
        domain.component(),
        targets,
        num_cases=3,
        length_bucket_edges=jnp.asarray([1, 2, 3]),
        indices=jnp.asarray([0, 2], dtype=jnp.int32),
    )

    assert len(constraints) == 2
    assert tuple(c.num_series_points for c in constraints) == (1, 2)
    assert tuple(c.num_points for c in constraints) == (2, 1)
    assert jnp.array_equal(constraints[0].indices, jnp.asarray([2], dtype=jnp.int32))
    assert jnp.array_equal(constraints[1].indices, jnp.asarray([0], dtype=jnp.int32))


def test_ragged_series_supervised_constraint_bucketed_scales_sum_reduction():
    domain, targets = _domain_and_targets()
    constraints = RaggedSeriesSupervisedConstraint.bucketed(
        "u",
        domain.component(),
        targets,
        num_cases=4,
        num_buckets=2,
        reduction="sum",
    )

    assert tuple(c.num_points for c in constraints) == (3, 1)
    assert jnp.allclose(
        jnp.stack([c.weight for c in constraints]),
        jnp.asarray([8.0 / 9.0, 4.0 / 3.0]),
    )


def test_ragged_series_supervised_constraint_bucketed_requires_case_per_bucket():
    domain, targets = _domain_and_targets()

    with pytest.raises(ValueError, match="number of non-empty length buckets"):
        RaggedSeriesSupervisedConstraint.bucketed(
            "u",
            domain.component(),
            targets,
            num_cases=1,
            length_bucket_edges=jnp.asarray([1, 2, 3]),
        )


def test_ragged_series_supervised_constraint_bucketed_avoids_global_padding_width():
    static = jnp.zeros((4, 1))
    series = jnp.zeros((4, 10_000, 1))
    lengths = jnp.asarray([2, 10_000, 5, 9_999], dtype=jnp.int32)
    domain = RaggedSeriesDatasetDomain(series, lengths, static=static)
    targets = jnp.zeros((4, 1))
    constraints = RaggedSeriesSupervisedConstraint.bucketed(
        "u",
        domain.component(),
        targets,
        num_cases=2,
        num_buckets=2,
    )

    short_constraint, _long_constraint = constraints
    batch = short_constraint.sample(key=jr.key(11))

    assert int(short_constraint.num_series_points) == 5
    assert batch.points["data"]["series"].data.shape == (1, 5, 1)
    assert batch.points["data"]["mask"].data.shape == (1, 5)


def test_ragged_series_supervised_constraint_requires_points_for_sampled_modes():
    domain, targets = _domain_and_targets()

    with pytest.raises(ValueError, match="num_series_points"):
        RaggedSeriesSupervisedConstraint(
            "u",
            domain.component(),
            targets,
            num_cases=4,
            series_sampling="points_uniform",
        )


def test_ragged_series_supervised_constraint_validates_domain_and_targets():
    data_domain = phx.domain.DatasetDomain(jnp.zeros((3, 2)))
    domain, targets = _domain_and_targets()

    with pytest.raises(TypeError, match="RaggedSeriesDatasetDomain"):
        RaggedSeriesSupervisedConstraint(
            "u",
            data_domain.component(),
            targets,
            num_cases=4,
        )

    with pytest.raises(ValueError, match="leading axis"):
        RaggedSeriesSupervisedConstraint(
            "u",
            domain.component(),
            jnp.zeros((4, 2)),
            num_cases=4,
        )
