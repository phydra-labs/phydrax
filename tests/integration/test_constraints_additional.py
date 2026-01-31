#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import cast

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from phydrax.constraints import (
    ContinuousInitialConstraint,
    ContinuousODEConstraint,
    DiscreteODEConstraint,
    FunctionalConstraint,
    IntegralEqualityConstraint,
    PointSetConstraint,
)
from phydrax.constraints._continuous_interior import ContinuousPointwiseInteriorConstraint
from phydrax.constraints._discrete_interior import DiscreteInteriorDataConstraint
from phydrax.constraints._functional_initial import DiscreteInitialConstraint
from phydrax.domain import (
    DomainComponentUnion,
    FixedStart,
    Interval1d,
    ProductStructure,
    TimeInterval,
)
from phydrax.operators.differential import div_diag_k_grad, dt, laplacian


def _jit_loss(constraint, functions):
    loss_fn = eqx.filter_jit(lambda k: constraint.loss(functions, key=k))
    return loss_fn(jr.key(0))


def test_continuous_initial_coord_separable_spatial():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component({"t": FixedStart()})
    structure = ProductStructure((("x",),))

    @domain.Function("x", "t")
    def u(x, t):
        return 0.0

    constraint = ContinuousInitialConstraint(
        "u",
        component,
        func=0.0,
        num_points=(),
        structure=structure,
        coord_separable={"x": 4},
    )
    assert _jit_loss(constraint, {"u": u}) < 1e-6


def test_integral_constraint_coord_separable_constant():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component()
    structure = ProductStructure((("x", "t"),))

    @domain.Function("x", "t")
    def u(x, t):
        return 1.0

    constraint = IntegralEqualityConstraint.from_operator(
        component=component,
        operator=lambda f: f,
        constraint_vars="u",
        num_points=6,
        structure=structure,
        coord_separable={"x": 5},
        equal_to=1.0,
    )
    assert _jit_loss(constraint, {"u": u}) < 1e-6


def test_integral_constraint_over_axis_constant():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component()
    structure = ProductStructure((("x",), ("t",)))
    num_x = 8
    num_t = 6

    @domain.Function("x", "t")
    def u(x, t):
        return 1.0

    expected = 1.0 / float(num_t)
    constraint = IntegralEqualityConstraint.from_operator(
        component=component,
        operator=lambda f: f,
        constraint_vars="u",
        num_points=(num_x, num_t),
        structure=structure,
        over="x",
        equal_to=expected,
    )
    assert _jit_loss(constraint, {"u": u}) < 1e-6


def test_functional_constraint_union_zero_loss():
    geom = Interval1d(0.0, 1.0)
    c1 = geom.component(where={"x": lambda p: p[0] < 0.5})
    c2 = geom.component(where={"x": lambda p: p[0] >= 0.5})
    union = DomainComponentUnion((c1, c2))
    structure = ProductStructure((("x",),))

    @geom.Function("x")
    def u(x):
        return 0.0

    constraint = FunctionalConstraint.from_operator(
        component=union,
        operator=lambda f: f,
        constraint_vars="u",
        num_points=8,
        structure=structure,
    )
    assert _jit_loss(constraint, {"u": u}) < 1e-6


def test_where_all_masks_interior_constraint():
    geom = Interval1d(0.0, 1.0)
    structure = ProductStructure((("x",),))

    @geom.Function("x")
    def u(x):
        return jnp.where(x[..., 0] < 0.5, 0.0, 1.0)

    @geom.Function("x")
    def mask(x):
        return jnp.where(x[..., 0] < 0.5, 1.0, 0.0)

    constraint = ContinuousPointwiseInteriorConstraint(
        "u",
        geom,
        operator=lambda f: f,
        num_points=16,
        structure=structure,
        where_all=mask,
    )
    assert _jit_loss(constraint, {"u": u}) < 1e-6


def test_discrete_interior_sensor_track_custom_weights_zero():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    structure = ProductStructure((("x", "t"),))

    @domain.Function("x", "t")
    def u(x, t):
        return 1.0

    sensors = jnp.array([[0.2], [0.8]], dtype=float)
    times = jnp.array([0.25, 0.5, 0.75], dtype=float)
    sensor_values = jnp.ones((2, 3), dtype=float)

    constraint = DiscreteInteriorDataConstraint(
        "u",
        domain,
        sensors=sensors,
        times=times,
        sensor_values=sensor_values,
        num_points=12,
        structure=structure,
        idw_exponent=3.0,
        eps_snap=1e-6,
        lengthscales={"x": 0.3},
    )
    assert _jit_loss(constraint, {"u": u}) < 1e-6


def test_pointset_constraint_weighted_sum():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()

    @geom.Function("x")
    def u(x):
        return 1.0

    points = {"x": jnp.array([[0.1], [0.4], [0.9]], dtype=float)}

    constraint = PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=lambda fns: fns["u"],
        weight=2.0,
        reduction="sum",
    )
    loss = _jit_loss(constraint, {"u": u})
    assert jnp.allclose(loss, 6.0)


def test_discrete_initial_constraint_forward_mode():
    time = TimeInterval(0.0, 1.0).relabel("tau")
    component = time.component({"tau": FixedStart()})

    @time.Function("tau")
    def u(tau):
        return tau**2

    constraint = DiscreteInitialConstraint(
        "u",
        component,
        evolution_var="tau",
        points={},
        values=0.0,
        time_derivative_order=1,
        mode="forward",
    )
    assert _jit_loss(constraint, {"u": u}) < 1e-6


def test_ode_constraints_relabel_nonuniform_times():
    time = TimeInterval(0.0, 1.0).relabel("tau")
    structure = ProductStructure((("tau",),))

    @time.Function("tau")
    def u(tau):
        return tau**2

    @time.Function("tau")
    def target(tau):
        return 2.0 * tau

    def operator(f):
        return dt(f, var="tau") - target

    continuous = ContinuousODEConstraint(
        "u",
        cast(TimeInterval, time),
        operator,
        num_points=32,
        structure=structure,
    )
    assert _jit_loss(continuous, {"u": u}) < 1e-6

    times = jnp.array([0.0, 0.1, 0.4, 1.0], dtype=float)
    discrete = DiscreteODEConstraint(
        "u",
        cast(TimeInterval, time),
        operator,
        times=times,
    )
    assert _jit_loss(discrete, {"u": u}) < 1e-6


def test_integral_constraint_union_zero_loss():
    geom = Interval1d(0.0, 1.0)
    left = geom.component(where={"x": lambda p: p[0] < 0.5})
    right = geom.component(where={"x": lambda p: p[0] >= 0.5})
    union = DomainComponentUnion((left, right))
    structure = ProductStructure((("x",),))

    @geom.Function("x")
    def u(x):
        return 0.0

    constraint = IntegralEqualityConstraint.from_operator(
        component=union,
        operator=lambda f: f,
        constraint_vars="u",
        num_points=8,
        structure=structure,
        equal_to=0.0,
    )
    assert _jit_loss(constraint, {"u": u}) < 1e-6


def test_integral_constraint_where_zero_mask():
    geom = Interval1d(0.0, 1.0)
    component = geom.component(where={"x": lambda p: p * 0.0})
    structure = ProductStructure((("x",),))

    @geom.Function("x")
    def u(x):
        return 1.0

    constraint = IntegralEqualityConstraint.from_operator(
        component=component,
        operator=lambda f: f,
        constraint_vars="u",
        num_points=16,
        structure=structure,
        equal_to=0.0,
    )
    assert _jit_loss(constraint, {"u": u}) < 1e-6


def test_discrete_interior_sensor_track_coord_separable_multilabel():
    x_dom = Interval1d(0.0, 1.0)
    y_dom = Interval1d(0.0, 1.0).relabel("y")
    time = TimeInterval(0.0, 1.0)
    domain = x_dom @ y_dom @ time
    structure = ProductStructure((("x", "y", "t"),))

    @domain.Function("x", "y", "t")
    def u(x, y, t):
        return 1.0

    sensors = {
        "x": jnp.array([[0.2], [0.8]], dtype=float),
        "y": jnp.array([[0.3], [0.7]], dtype=float),
    }
    times = jnp.array([0.25, 0.75], dtype=float)
    sensor_values = jnp.ones((2, 2), dtype=float)

    constraint = DiscreteInteriorDataConstraint(
        "u",
        domain,
        sensors=sensors,
        times=times,
        sensor_values=sensor_values,
        num_points=12,
        structure=structure,
        coord_separable={"x": 4},
    )
    assert _jit_loss(constraint, {"u": u}) < 1e-6


def test_coord_separable_laplacian_jet_zero():
    geom = Interval1d(0.0, 1.0)
    structure = ProductStructure((("x",),))

    @geom.Function("x")
    def u(x):
        return 1.0

    constraint = ContinuousPointwiseInteriorConstraint(
        "u",
        geom,
        operator=lambda f: laplacian(f, var="x", backend="jet"),
        num_points=16,
        structure=structure,
        coord_separable={"x": 8},
    )
    assert _jit_loss(constraint, {"u": u}) < 1e-6


def test_coord_separable_div_diag_k_grad_jet_zero():
    geom = Interval1d(0.0, 1.0)
    structure = ProductStructure((("x",),))

    @geom.Function("x")
    def u(x):
        return 1.0

    @geom.Function("x")
    def k_vec(x):
        return jnp.array([1.0], dtype=float)

    constraint = ContinuousPointwiseInteriorConstraint(
        "u",
        geom,
        operator=lambda f: div_diag_k_grad(f, k_vec, var="x", backend="jet"),
        num_points=16,
        structure=structure,
        coord_separable={"x": 8},
    )
    assert _jit_loss(constraint, {"u": u}) < 1e-6
