#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax._frozendict import frozendict
from phydrax.domain import PointBatch, SampleLayout, TrajectoryDatasetDomain
from phydrax.operators.differential import partial_n, partial_t


def _make_linear_problem():
    inputs = jnp.asarray([[0.0], [1.0], [2.0]])
    lengths = jnp.asarray([2, 4, 3])
    domain = TrajectoryDatasetDomain(inputs, lengths, dt=0.25)
    times = domain.start + domain.dt * jnp.arange(domain.max_length)
    y0 = inputs[:, 0, None] + 2.0 * times[None, :]
    y1 = 1.0 - inputs[:, 0, None] + 3.0 * times[None, :]
    values = jnp.stack((y0, y1), axis=-1)
    structure = SampleLayout((("data", "t"),))
    return domain, values, structure


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


def test_enforce_ragged_time_series_matches_all_observed_nodes_exactly():
    domain, values, structure = _make_linear_problem()

    @domain.Function("data", "t")
    def free(data, t):
        del data, t
        return jnp.asarray([10.0, -10.0])

    hard = phx.enforcement.enforce_ragged_time_series(free, domain, values)
    batch = _all_observation_batch(domain, structure)
    pred = jnp.asarray(hard(batch, key=jr.key(0)).data)
    case_indices = domain.flat_case_indices
    time_indices = domain.flat_time_indices
    target = values[case_indices, time_indices]
    assert jnp.allclose(pred, target, atol=1e-12)


def test_enforce_ragged_time_series_supports_first_time_derivative():
    domain, values, structure = _make_linear_problem()

    @domain.Function("data", "t")
    def free(data, t):
        return jnp.asarray([data[0] + 2.0 * t, 1.0 - data[0] + 3.0 * t])

    hard = phx.enforcement.enforce_ragged_time_series(free, domain, values)
    dt_hard = partial_t(hard, var="t")
    batch = domain.component().sample(phx.domain.PointSampling(12, layout=structure), key=jr.key(1))
    pred = jnp.asarray(dt_hard(batch, key=jr.key(2)).data)
    expected = jnp.broadcast_to(jnp.asarray([2.0, 3.0]), pred.shape)
    assert jnp.allclose(pred, expected, atol=1e-12)


def test_enforce_ragged_time_series_supports_cubic_hermite_second_time_derivative():
    domain, values, structure = _make_linear_problem()

    @domain.Function("data", "t")
    def free(data, t):
        return jnp.asarray([data[0] + 2.0 * t, 1.0 - data[0] + 3.0 * t])

    hard = phx.enforcement.enforce_ragged_time_series(
        free,
        domain,
        values,
        interpolation="cubic_hermite",
        gate="sin4",
    )
    d2_hard = partial_n(hard, var="t", order=2)
    batch = domain.component().sample(phx.domain.PointSampling(12, layout=structure), key=jr.key(5))
    pred = jnp.asarray(d2_hard(batch, key=jr.key(6)).data)
    assert jnp.allclose(pred, jnp.zeros_like(pred), atol=1e-10)


def test_enforce_ragged_time_series_rejects_linear_second_time_derivative():
    domain, values, _structure = _make_linear_problem()

    @domain.Function("data", "t")
    def free(data, t):
        return jnp.asarray([data[0] + 2.0 * t, 1.0 - data[0] + 3.0 * t])

    hard = phx.enforcement.enforce_ragged_time_series(
        free, domain, values, interpolation="linear"
    )

    with pytest.raises(ValueError, match="linear.*order 1"):
        partial_n(hard, var="t", order=2)


def test_enforce_ragged_time_series_can_enforce_selected_components():
    domain, values, structure = _make_linear_problem()

    @domain.Function("data", "t")
    def free(data, t):
        del data, t
        return jnp.asarray([10.0, -5.0])

    hard = phx.enforcement.enforce_ragged_time_series(
        free, domain, values, components=[0]
    )
    batch = _all_observation_batch(domain, structure)
    pred = jnp.asarray(hard(batch, key=jr.key(7)).data)
    case_indices = domain.flat_case_indices
    time_indices = domain.flat_time_indices
    target = values[case_indices, time_indices]
    assert jnp.allclose(pred[:, 0], target[:, 0], atol=1e-12)
    assert jnp.allclose(pred[:, 1], jnp.full_like(pred[:, 1], -5.0), atol=1e-12)


def test_enforce_ragged_time_series_requires_trajectory_batch_indices():
    domain, values, structure = _make_linear_problem()

    @domain.Function("data", "t")
    def free(data, t):
        return jnp.asarray([data[0] + 2.0 * t, 1.0 - data[0] + 3.0 * t])

    hard = phx.enforcement.enforce_ragged_time_series(free, domain, values)
    batch = _all_observation_batch(domain, structure)
    stripped = PointBatch(
        points=frozendict({"data": batch["data"], "t": batch["t"]}),
        structure=batch.structure,
    )

    with pytest.raises(ValueError, match="internal field"):
        hard(stripped, key=jr.key(4))


def test_physics_only_constraint_can_use_hard_enforced_ragged_data():
    domain, values, structure = _make_linear_problem()

    @domain.Function("data", "t")
    def free(data, t):
        return jnp.asarray([data[0] + 2.0 * t, 1.0 - data[0] + 3.0 * t])

    @domain.Function()
    def rhs():
        return jnp.asarray([2.0, 3.0])

    hard = phx.enforcement.enforce_ragged_time_series(free, domain, values)
    component = domain.component()
    condition = phx.conditions.Residual(
        "u", component, lambda u: partial_t(u, var="t") - rhs
    )
    source = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.domain.PointSampling(16, layout=structure),
    )
    term = phx.terms.ResidualPenalty(condition, source)

    loss = term.loss({"u": hard}, key=jr.key(3))
    assert jnp.allclose(loss, 0.0, atol=1e-12)


def test_physics_only_constraint_can_use_second_derivative_hard_ragged_data():
    domain, values, structure = _make_linear_problem()

    @domain.Function("data", "t")
    def free(data, t):
        return jnp.asarray([data[0] + 2.0 * t, 1.0 - data[0] + 3.0 * t])

    hard = phx.enforcement.enforce_ragged_time_series(
        free,
        domain,
        values,
        interpolation="cubic_hermite",
        gate="sin4",
    )
    component = domain.component()
    condition = phx.conditions.Residual(
        "u", component, lambda u: partial_n(u, var="t", order=2)
    )
    source = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.domain.PointSampling(16, layout=structure),
    )
    term = phx.terms.ResidualPenalty(condition, source)

    loss = term.loss({"u": hard}, key=jr.key(8))
    assert jnp.allclose(loss, 0.0, atol=1e-10)
