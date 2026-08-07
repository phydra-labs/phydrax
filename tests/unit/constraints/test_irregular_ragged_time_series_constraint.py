#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.constraints import RaggedTimeSeriesDataConstraint, TrajectorySignal
from phydrax.domain import IrregularTrajectoryDatasetDomain, SampleLayout
from phydrax.operators.differential import partial_t


def _make_domain_and_values(*, sampling="observation_uniform"):
    inputs = jnp.asarray([[0.0], [1.0], [2.0]])
    times = jnp.asarray(
        [
            [0.0, 0.2, 0.7, 0.0],
            [0.1, 0.4, 1.2, 1.8],
            [-0.2, 0.3, 0.0, 0.0],
        ]
    )
    lengths = jnp.asarray([3, 4, 2])
    domain = IrregularTrajectoryDatasetDomain(
        inputs,
        times,
        lengths,
        sampling=sampling,
    )
    scalar = inputs[:, 0, None] + times
    values = jnp.stack((scalar, 2.0 * scalar), axis=-1)
    return domain, values


def test_irregular_ragged_time_series_matches_exact_observations():
    domain, values = _make_domain_and_values(sampling="observation_uniform")

    @domain.Function("data", "t")
    def exact(data, t):
        y = data[0] + t
        return jnp.asarray([y, 2.0 * y])

    constraint = RaggedTimeSeriesDataConstraint(
        "u",
        domain.component(),
        values,
        sampling=phx.domain.PointSampling(
            16, layout=SampleLayout((("data", "t"),)), design="uniform"
        ),
        selection="observation_uniform",
        label="irregular_trajectory_data",
    )

    loss = constraint.loss({"u": exact}, key=jr.key(0))
    metrics = constraint.data_metrics({"u": exact}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0, atol=1e-12)
    assert jnp.allclose(metrics["data_accuracy"], 1.0)


def test_irregular_ragged_time_series_linear_interpolation():
    domain, values = _make_domain_and_values(sampling="case_time_uniform")

    @domain.Function("data", "t")
    def exact(data, t):
        y = data[0] + t
        return jnp.asarray([y, 2.0 * y])

    constraint = RaggedTimeSeriesDataConstraint(
        "u",
        domain.component(),
        values,
        sampling=phx.domain.PointSampling(
            20, layout=SampleLayout((("data", "t"),)), design="uniform"
        ),
        selection="case_time_uniform",
        interpolation="linear",
    )

    loss = constraint.loss({"u": exact}, key=jr.key(1))
    assert jnp.allclose(loss, 0.0, atol=1e-12)


def test_irregular_trajectory_signal_linear_value_and_time_derivative():
    domain, values = _make_domain_and_values(sampling="case_time_uniform")
    signal = TrajectorySignal(domain, values, interpolation="linear")
    batch = domain.component().sample(
        phx.domain.PointSampling(20, layout=SampleLayout((("data", "t"),))), key=jr.key(2)
    )

    out = signal(batch)
    ds_dt = partial_t(signal, var="t")(batch)
    case_indices = jnp.asarray(
        batch["__phydrax_trajectory_case_index"].data,
        dtype=jnp.int32,
    )
    t = jnp.asarray(batch["t"].data)
    y = domain.inputs[case_indices, 0] + t
    expected = jnp.stack((y, 2.0 * y), axis=-1)

    assert jnp.allclose(out.data, expected, atol=1e-12)
    assert jnp.allclose(ds_dt.data, jnp.asarray([1.0, 2.0]), atol=1e-12)
