#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.domain import SampleLayout, TrajectoryDatasetDomain
from phydrax.terms import RaggedTimeSeriesDataTerm


def _make_domain_and_values():
    inputs = jnp.asarray([[0.0], [1.0], [2.0]])
    lengths = jnp.asarray([2, 4, 3])
    domain = TrajectoryDatasetDomain(inputs, lengths, dt=0.5)
    times = domain.start + domain.dt * jnp.arange(domain.max_length)
    values = inputs[:, 0, None] + times[None, :]
    return domain, values


def test_ragged_time_series_data_constraint_matches_exact_observations():
    domain, values = _make_domain_and_values()

    @domain.Function("data", "t")
    def exact(data, t):
        return data[0] + t

    constraint = RaggedTimeSeriesDataTerm(
        "u",
        domain.component(),
        values,
        sampling=phx.domain.PointSampling(
            12, layout=SampleLayout((("data", "t"),)), design="uniform"
        ),
        selection="observation_uniform",
        label="trajectory_data",
    )

    loss = constraint.loss({"u": exact}, key=jr.key(0))
    metrics = constraint.data_metrics({"u": exact}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0, atol=1e-12)
    assert jnp.allclose(metrics["data_accuracy"], 1.0)
    assert jnp.allclose(metrics["data_relative_l2_error"], 0.0)
    assert jnp.allclose(metrics["data_rmse"], 0.0)


def test_ragged_time_series_data_constraint_linear_interpolation():
    domain, values = _make_domain_and_values()

    @domain.Function("data", "t")
    def exact(data, t):
        return data[0] + t

    constraint = RaggedTimeSeriesDataTerm(
        "u",
        domain.component(),
        values,
        sampling=phx.domain.PointSampling(
            16, layout=SampleLayout((("data", "t"),)), design="uniform"
        ),
        selection="case_time_uniform",
        interpolation="linear",
    )

    loss = constraint.loss({"u": exact}, key=jr.key(1))
    assert jnp.allclose(loss, 0.0, atol=1e-12)


def test_ragged_time_series_data_constraint_vector_targets():
    domain, scalar_values = _make_domain_and_values()
    values = jnp.stack((scalar_values, 2.0 * scalar_values), axis=-1)

    @domain.Function("data", "t")
    def exact(data, t):
        y = data[0] + t
        return jnp.asarray([y, 2.0 * y])

    constraint = RaggedTimeSeriesDataTerm(
        "u",
        domain.component(),
        values,
        sampling=phx.domain.PointSampling(12, design="uniform"),
        selection="observation_uniform",
    )

    loss = constraint.loss({"u": exact}, key=jr.key(2))
    metrics = constraint.data_metrics({"u": exact}, key=jr.key(2))
    assert jnp.allclose(loss, 0.0, atol=1e-12)
    assert jnp.allclose(metrics["data_accuracy"], 1.0)


def test_ragged_time_series_data_constraint_samples_only_case_subset():
    domain, values = _make_domain_and_values()
    allowed = jnp.asarray([1, 2], dtype=jnp.int32)
    constraint = RaggedTimeSeriesDataTerm(
        "u",
        domain.component(),
        values,
        sampling=phx.domain.PointSampling(24, design="uniform"),
        selection="observation_uniform",
        case_indices=allowed,
    )

    batch = constraint.sample(key=jr.key(9))
    assert jnp.all(jnp.isin(batch.case_indices, allowed))


def test_ragged_time_series_case_uniform_samples_only_case_subset():
    domain, values = _make_domain_and_values()
    allowed = jnp.asarray([2], dtype=jnp.int32)
    constraint = RaggedTimeSeriesDataTerm(
        "u",
        domain.component(),
        values,
        sampling=phx.domain.PointSampling(10, design="uniform"),
        selection="case_uniform",
        case_indices=allowed,
    )

    batch = constraint.sample(key=jr.key(10))
    assert jnp.all(batch.case_indices == 2)


def test_ragged_time_series_data_constraint_penalizes_wrong_function():
    domain, values = _make_domain_and_values()

    @domain.Function("data", "t")
    def wrong(data, t):
        del data, t
        return 0.0

    constraint = RaggedTimeSeriesDataTerm(
        "u",
        domain.component(),
        values,
        sampling=phx.domain.PointSampling(12, design="uniform"),
        selection="observation_uniform",
    )

    loss = constraint.loss({"u": wrong}, key=jr.key(3))
    metrics = constraint.data_metrics({"u": wrong}, key=jr.key(3))
    assert loss > 0.0
    assert metrics["data_relative_l2_error"] > 0.0
