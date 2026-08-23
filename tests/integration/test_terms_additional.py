#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.conditions import Initial, Moment, Observation, Residual
from phydrax.domain import (
    ComponentSum,
    FixedStart,
    Interval1d,
    SampleLayout,
    TimeInterval,
)
from phydrax.operators.differential import div_diag_k_grad, dt, laplacian
from phydrax.terms import (
    MomentPenalty,
    ObservationPenalty,
    RandomizedMomentPenalty,
    ResidualPenalty,
)


def _jit_loss(term, functions):
    loss_fn = eqx.filter_jit(lambda k: term.loss(functions, key=k))
    return loss_fn(jr.key(0))


def _per_step(condition, count, *, moment=False):
    target = (
        phx.integration.over(condition.on)
        if moment
        else phx.integration.mean_over(condition.on)
    )
    return phx.integration.per_step(target, phx.integration.MonteCarloPlan(count))


def _fixed_source(target, batch):
    return phx.integration.fixed(phx.integration.from_samples(target, batch))


def test_continuous_initial_coord_separable_spatial():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component({"t": FixedStart()})

    @domain.Function("x", "t")
    def u(x, t):
        return 0.0

    condition = Initial("u", component, target=0.0)
    batch = component.sample(phx.domain.GridSampling({"x": 4}), key=jr.key(1))
    term = ResidualPenalty(
        condition,
        _fixed_source(phx.integration.mean_over(condition.on), batch),
    )
    assert _jit_loss(term, {"u": u}) < 1e-6


def test_integral_constraint_coord_separable_constant():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component()

    @domain.Function("x", "t")
    def u(x, t):
        return 1.0

    condition = Moment("u", component, lambda field: field, target=1.0)
    batch = component.sample(
        phx.domain.GridSampling(
            {"x": 5},
            dense=phx.domain.PointSampling(6),
        ),
        key=jr.key(2),
    )
    term = MomentPenalty(
        condition,
        _fixed_source(phx.integration.over(condition.on), batch),
    )
    assert _jit_loss(term, {"u": u}) < 1e-6


def test_integral_constraint_over_axis_constant():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component()
    num_x = 8
    num_t = 6

    @domain.Function("x", "t")
    def u(x, t):
        return 1.0

    expected = 1.0
    condition = Moment("u", component, lambda field: field, target=expected)
    term = RandomizedMomentPenalty(
        condition,
        _per_step(condition, num_x * num_t, moment=True),
    )
    assert _jit_loss(term, {"u": u}) < 1e-6


def test_residual_penalty_union_zero_loss():
    geom = Interval1d(0.0, 1.0)
    c1 = geom.component(where={"x": lambda p: p[0] < 0.5})
    c2 = geom.component(where={"x": lambda p: p[0] >= 0.5})
    union = ComponentSum((c1, c2), assume_disjoint=True)

    @geom.Function("x")
    def u(x):
        return 0.0

    condition = Residual("u", union, lambda field: field)
    target = phx.integration.mean_over(condition.on)
    source = _fixed_source(
        target,
        (
            c1.points({"x": jnp.array([0.1, 0.4])}),
            c2.points({"x": jnp.array([0.6, 0.9])}),
        ),
    )
    term = ResidualPenalty(condition, source)
    assert _jit_loss(term, {"u": u}) < 1e-6


def test_where_all_masks_interior_constraint():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return jnp.where(x[..., 0] < 0.5, 0.0, 1.0)

    @geom.Function("x")
    def mask(x):
        return jnp.where(x[..., 0] < 0.5, 1.0, 0.0)

    component = geom.component(where_all=mask)
    condition = Residual("u", component, lambda field: field)
    term = ResidualPenalty(condition, _per_step(condition, 16))
    assert _jit_loss(term, {"u": u}) < 1e-6


def test_discrete_interior_sensor_track_custom_weights_zero():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component()

    @domain.Function("x", "t")
    def u(x, t):
        return 1.0

    sensors = jnp.array([[0.2], [0.8]], dtype=float)
    times = jnp.array([0.25, 0.5, 0.75], dtype=float)
    sensor_values = jnp.ones((2, 3), dtype=float)
    lengthscales = {"x": 0.3}
    assert sensors.shape == (2, 1)
    assert times.shape == (3,)
    assert lengthscales["x"] == 0.3

    condition = Observation("u", component, domain.Function()(sensor_values[0, 0]))
    term = ObservationPenalty(condition, _per_step(condition, 12))
    assert _jit_loss(term, {"u": u}) < 1e-6


def test_pointset_constraint_weighted_sum():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()

    @geom.Function("x")
    def u(x):
        return 1.0

    points = {"x": jnp.array([[0.1], [0.4], [0.9]], dtype=float)}
    condition = Residual("u", component, lambda field: field)
    source = _fixed_source(
        phx.integration.mean_over(condition.on),
        component.points(points),
    )
    term = ResidualPenalty(condition, source, scale=6.0)
    loss = _jit_loss(term, {"u": u})
    assert jnp.allclose(loss, 6.0)


def test_discrete_initial_constraint_forward_mode():
    time = TimeInterval(0.0, 1.0).relabel("tau")
    component = time.component({"tau": FixedStart()})

    @time.Function("tau")
    def u(tau):
        return tau**2

    condition = Initial(
        "u",
        component,
        evolution_var="tau",
        target=0.0,
        order=1,
        mode="forward",
    )
    source = _fixed_source(
        phx.integration.mean_over(condition.on),
        component.points({}),
    )
    term = ResidualPenalty(condition, source)
    assert _jit_loss(term, {"u": u}) < 1e-6


def test_ode_constraints_relabel_nonuniform_times():
    time = TimeInterval(0.0, 1.0).relabel("tau")
    component = time.component()

    @time.Function("tau")
    def u(tau):
        return tau**2

    @time.Function("tau")
    def target(tau):
        return 2.0 * tau

    def operator(field):
        return dt(field, var="tau") - target

    condition = Residual("u", component, operator)
    continuous = ResidualPenalty(condition, _per_step(condition, 32))
    assert _jit_loss(continuous, {"u": u}) < 1e-6

    times = jnp.array([0.0, 0.1, 0.4, 1.0], dtype=float)
    discrete = ResidualPenalty(
        condition,
        _fixed_source(
            phx.integration.mean_over(condition.on),
            component.points({"tau": times}),
        ),
    )
    assert _jit_loss(discrete, {"u": u}) < 1e-6


def test_integral_constraint_union_zero_loss():
    geom = Interval1d(0.0, 1.0)
    left = geom.component(where={"x": lambda p: p[0] < 0.5})
    right = geom.component(where={"x": lambda p: p[0] >= 0.5})
    union = ComponentSum((left, right), assume_disjoint=True)

    @geom.Function("x")
    def u(x):
        return 0.0

    condition = Moment("u", union, lambda field: field, target=0.0)
    target = phx.integration.over(condition.on)
    source = _fixed_source(
        target,
        (
            left.points({"x": jnp.array([0.1, 0.4])}),
            right.points({"x": jnp.array([0.6, 0.9])}),
        ),
    )
    term = MomentPenalty(condition, source)
    assert _jit_loss(term, {"u": u}) < 1e-6


def test_integral_constraint_where_zero_mask():
    geom = Interval1d(0.0, 1.0)
    component = geom.component(where={"x": lambda p: p * 0.0})

    @geom.Function("x")
    def u(x):
        return 1.0

    condition = Moment("u", component, lambda field: field, target=0.0)
    points = component.points({"x": jnp.linspace(0.0, 1.0, 16)})
    term = MomentPenalty(
        condition,
        _fixed_source(phx.integration.over(condition.on), points),
    )
    assert _jit_loss(term, {"u": u}) < 1e-6


def test_discrete_interior_sensor_track_coord_separable_multilabel():
    x_dom = Interval1d(0.0, 1.0)
    y_dom = Interval1d(0.0, 1.0).relabel("y")
    time = TimeInterval(0.0, 1.0)
    domain = x_dom @ y_dom @ time
    component = domain.component()

    @domain.Function("x", "y", "t")
    def u(x, y, t):
        return 1.0

    sensors = {
        "x": jnp.array([[0.2], [0.8]], dtype=float),
        "y": jnp.array([[0.3], [0.7]], dtype=float),
    }
    times = jnp.array([0.25, 0.75], dtype=float)
    sensor_values = jnp.ones((2, 2), dtype=float)
    assert sensors["x"].shape == sensors["y"].shape
    assert times.shape == (2,)

    condition = Observation("u", component, domain.Function()(sensor_values[0, 0]))
    batch = component.sample(
        phx.domain.GridSampling(
            {"x": 4},
            dense=phx.domain.PointSampling(
                12,
                layout=SampleLayout((("y", "t"),)),
            ),
        ),
        key=jr.key(3),
    )
    term = ObservationPenalty(
        condition,
        _fixed_source(phx.integration.mean_over(condition.on), batch),
    )
    assert _jit_loss(term, {"u": u}) < 1e-6


def test_coord_separable_laplacian_jet_zero():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()

    @geom.Function("x")
    def u(x):
        return 1.0

    condition = Residual(
        "u",
        component,
        lambda field: laplacian(field, var="x", backend="jet"),
    )
    batch = component.sample(phx.domain.GridSampling({"x": 8}), key=jr.key(4))
    term = ResidualPenalty(
        condition,
        _fixed_source(phx.integration.mean_over(condition.on), batch),
    )
    assert _jit_loss(term, {"u": u}) < 1e-6


def test_coord_separable_div_diag_k_grad_jet_zero():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()

    @geom.Function("x")
    def u(x):
        return 1.0

    @geom.Function("x")
    def k_vec(x):
        return jnp.array([1.0], dtype=float)

    condition = Residual(
        "u",
        component,
        lambda field: div_diag_k_grad(field, k_vec, var="x", backend="jet"),
    )
    batch = component.sample(phx.domain.GridSampling({"x": 8}), key=jr.key(5))
    term = ResidualPenalty(
        condition,
        _fixed_source(phx.integration.mean_over(condition.on), batch),
    )
    assert _jit_loss(term, {"u": u}) < 1e-6
