#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.domain import SampleLayout, TrajectoryDatasetDomain
from phydrax.operators.differential import partial_n, partial_t
from phydrax.terms import (
    TrajectoryCaseDataTerm,
    TrajectorySignal,
)


def _make_problem():
    inputs = jnp.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])
    lengths = jnp.asarray([2, 4, 3])
    domain = TrajectoryDatasetDomain(inputs, lengths, dt=0.5)
    times = domain.start + domain.dt * jnp.arange(domain.max_length)
    slopes = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    offsets = jnp.asarray([0.0, 10.0, 20.0, 30.0])
    values = inputs[:, 0, None, None] + times[None, :, None] * slopes + offsets
    structure = SampleLayout((("data", "t"),))
    return domain, values, slopes, structure


def _all_observation_batch(domain, structure):
    case_indices = domain.flat_case_indices
    time_indices = domain.flat_time_indices
    times = domain.observation_times(case_indices, time_indices)
    return domain.points_from_case_time(
        case_indices,
        times,
        structure=structure,
        time_indices=time_indices,
    )


def test_trajectory_signal_matches_vector_observed_nodes():
    domain, values, _slopes, structure = _make_problem()
    signal = TrajectorySignal(domain, values, interpolation="linear")

    batch = _all_observation_batch(domain, structure)
    pred = jnp.asarray(signal(batch, key=jr.key(0)).data)
    target = values[domain.flat_case_indices, domain.flat_time_indices]
    assert jnp.allclose(pred, target, atol=1e-12)


def test_trajectory_signal_linear_interpolates_and_differentiates():
    domain, values, slopes, structure = _make_problem()
    signal = TrajectorySignal(domain, values, interpolation="linear")
    dt_signal = partial_t(signal, var="t")

    case_indices = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    times = jnp.asarray([0.25, 0.75, 0.5])
    batch = domain.points_from_case_time(
        case_indices,
        times,
        structure=structure,
        time_indices=jnp.asarray([0, 1, 1], dtype=jnp.int32),
    )

    pred = jnp.asarray(signal(batch, key=jr.key(1)).data)
    expected = values[case_indices, 0] + times[:, None] * slopes
    assert jnp.allclose(pred, expected, atol=1e-12)

    deriv = jnp.asarray(dt_signal(batch, key=jr.key(2)).data)
    assert jnp.allclose(deriv, jnp.broadcast_to(slopes, deriv.shape), atol=1e-12)


def test_trajectory_signal_nearest_rejects_time_derivative():
    domain, values, _slopes, _structure = _make_problem()
    signal = TrajectorySignal(domain, values, interpolation="nearest")

    with pytest.raises(ValueError, match="nearest.*not differentiable"):
        partial_t(signal, var="t")


def test_trajectory_signal_cubic_hermite_supports_second_time_derivative():
    domain, values, _slopes, structure = _make_problem()
    signal = TrajectorySignal(domain, values, interpolation="cubic_hermite")
    d2_signal = partial_n(signal, var="t", order=2)

    batch = domain.component().sample(
        phx.domain.PointSampling(12, layout=structure), key=jr.key(6)
    )
    pred = jnp.asarray(d2_signal(batch, key=jr.key(7)).data)
    assert jnp.allclose(pred, jnp.zeros_like(pred), atol=1e-10)


def test_trajectory_case_data_term_supervises_case_only_vector_target():
    domain, _values, _slopes, _structure = _make_problem()
    inputs = domain.inputs
    targets = jnp.stack(
        (inputs[:, 0] + inputs[:, 1], inputs[:, 0] - inputs[:, 1]), axis=-1
    )

    @domain.Function("data")
    def theta(data):
        return jnp.asarray([data[0] + data[1], data[0] - data[1]])

    term = TrajectoryCaseDataTerm(
        "theta",
        domain.component(),
        targets,
        sampling=phx.domain.PointSampling(12, design="uniform"),
        label="case_target",
    )

    loss = term.loss({"theta": theta}, key=jr.key(3))
    metrics = term.data_metrics({"theta": theta}, key=jr.key(3))
    assert jnp.allclose(loss, 0.0, atol=1e-12)
    assert jnp.allclose(metrics["data_accuracy"], 1.0)
    assert jnp.allclose(metrics["data_relative_l2_error"], 0.0)


def test_trajectory_case_data_term_can_evaluate_at_case_end():
    domain, _values, _slopes, _structure = _make_problem()
    inputs = domain.inputs
    targets = inputs[:, 0] + domain.end_times

    @domain.Function("data", "t")
    def final_value(data, t):
        return data[0] + t

    term = TrajectoryCaseDataTerm(
        "theta",
        domain.component(),
        targets,
        sampling=phx.domain.PointSampling(12, design="uniform"),
        case_time="end",
    )

    loss = term.loss({"theta": final_value}, key=jr.key(4))
    assert jnp.allclose(loss, 0.0, atol=1e-12)


def test_trajectory_case_data_term_samples_only_case_subset():
    domain, _values, _slopes, _structure = _make_problem()
    targets = domain.inputs[:, 0]
    allowed = jnp.asarray([0, 2], dtype=jnp.int32)

    term = TrajectoryCaseDataTerm(
        "theta",
        domain.component(),
        targets,
        sampling=phx.domain.PointSampling(16, design="uniform"),
        case_indices=allowed,
    )

    batch = term.sample(key=jr.key(8))
    assert jnp.all(jnp.isin(batch.case_indices, allowed))
    assert jnp.allclose(batch.target, targets[batch.case_indices])


def test_physics_residual_can_use_fixed_trajectory_signal():
    inputs = jnp.asarray([[0.0], [1.0], [2.0]])
    lengths = jnp.asarray([2, 4, 3])
    domain = TrajectoryDatasetDomain(inputs, lengths, dt=0.25)
    structure = SampleLayout((("data", "t"),))
    times = domain.start + domain.dt * jnp.arange(domain.max_length)
    values = inputs[:, 0, None] + 2.0 * times[None, :]
    signal = TrajectorySignal(domain, values, interpolation="linear")

    @domain.Function("data", "t")
    def u(data, t):
        return data[0] + 2.0 * t

    component = domain.component()
    condition = phx.conditions.Residual(
        ("u", "s"),
        component,
        lambda u_fn, s_fn: partial_t(u_fn, var="t") - partial_t(s_fn, var="t"),
    )
    source = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.domain.PointSampling(16, layout=structure),
    )
    term = phx.terms.ResidualPenalty(condition, source)

    loss = term.loss({"u": u, "s": signal}, key=jr.key(5))
    assert jnp.allclose(loss, 0.0, atol=1e-12)
