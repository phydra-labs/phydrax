#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.conditions import Initial, Observation, Residual
from phydrax.domain import FixedStart, Interval1d, TimeInterval
from phydrax.terms import ObservationPenalty, ResidualPenalty


def _jit_loss(term, functions):
    loss_fn = eqx.filter_jit(lambda k: term.loss(functions, key=k))
    return loss_fn(jr.key(0))


def _fixed_source(component, points):
    realization = phx.integration.from_samples(
        phx.integration.mean_over(component),
        component.points(points),
    )
    return phx.integration.fixed(realization)


def test_continuous_pointwise_interior_constraint_zero():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()

    @geom.Function("x")
    def u(x):
        return 0.0

    condition = Residual("u", component, lambda field: field)
    source = phx.integration.per_step(
        phx.integration.mean_over(condition.on),
        phx.integration.MonteCarloPlan(8),
    )
    term = ResidualPenalty(condition, source)
    assert _jit_loss(term, {"u": u}) < 1e-6


def test_continuous_initial_function_constraint_zero():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component({"t": FixedStart()})

    @domain.Function("x", "t")
    def u(x, t):
        return t**2

    source = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.integration.MonteCarloPlan(8),
    )
    first = ResidualPenalty(
        Initial("u", component, target=0.0, order=1),
        source,
    )
    assert _jit_loss(first, {"u": u}) < 1e-6

    second = ResidualPenalty(
        Initial("u", component, target=2.0, order=2, backend="jet"),
        source,
    )
    assert _jit_loss(second, {"u": u}) < 1e-6


def test_discrete_initial_constraint_zero():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component({"t": FixedStart()})

    @domain.Function("x", "t")
    def u(x, t):
        return t**2

    points = {"x": jnp.array([[0.25], [0.75]], dtype=float)}
    values = jnp.array([0.0, 0.0], dtype=float)
    source = _fixed_source(component, points)
    term = ResidualPenalty(
        Initial("u", component, target=values[0]),
        source,
    )
    assert _jit_loss(term, {"u": u}) < 1e-6

    values2 = jnp.array([2.0, 2.0], dtype=float)
    term2 = ResidualPenalty(
        Initial("u", component, target=values2[0], order=2, backend="jet"),
        source,
    )
    assert _jit_loss(term2, {"u": u}) < 1e-6


def test_discrete_interior_data_constraint_points_zero():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()

    @geom.Function("x")
    def u(x):
        return x[0]

    points = {"x": jnp.array([[0.25], [0.75]], dtype=float)}
    values = jnp.array([0.25, 0.75], dtype=float)

    @geom.Function("x")
    def target(x):
        return x[0]

    assert jnp.allclose(values, jnp.asarray([0.25, 0.75]))
    condition = Observation("u", component, target)
    term = ObservationPenalty(condition, _fixed_source(component, points))
    assert _jit_loss(term, {"u": u}) < 1e-6


def test_discrete_interior_data_constraint_sensor_tracks_zero():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component()

    @domain.Function("x", "t")
    def u(x, t):
        return 1.0

    sensors = jnp.array([[0.2], [0.8]], dtype=float)
    times = jnp.array([0.25, 0.75], dtype=float)
    sensor_values = jnp.ones((2, 2), dtype=float)
    assert sensors.shape == (2, 1)
    assert times.shape == (2,)

    target = domain.Function()(sensor_values[0, 0])
    condition = Observation("u", component, target)
    source = phx.integration.per_step(
        phx.integration.mean_over(condition.on),
        phx.integration.MonteCarloPlan(16),
    )
    term = ObservationPenalty(condition, source)
    assert _jit_loss(term, {"u": u}) < 1e-6
