#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from phydrax.constraints._continuous_interior import (
    ContinuousInitialFunctionConstraint,
    ContinuousPointwiseInteriorConstraint,
)
from phydrax.constraints._discrete_interior import DiscreteInteriorDataConstraint
from phydrax.constraints._functional_initial import DiscreteInitialConstraint
from phydrax.domain import (
    Boundary,
    FixedStart,
    Interval1d,
    ProductStructure,
    TimeInterval,
)


def _jit_loss(constraint, functions):
    loss_fn = eqx.filter_jit(lambda k: constraint.loss(functions, key=k))
    return loss_fn(jr.key(0))


def test_continuous_pointwise_interior_constraint_zero():
    geom = Interval1d(0.0, 1.0)
    structure = ProductStructure((("x",),))

    @geom.Function("x")
    def u(x):
        return 0.0

    constraint = ContinuousPointwiseInteriorConstraint(
        "u",
        geom,
        operator=lambda f: f,
        num_points=8,
        structure=structure,
    )
    assert _jit_loss(constraint, {"u": u}) < 1e-6


def test_continuous_initial_function_constraint_zero():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    structure = ProductStructure((("x",),))

    @domain.Function("x", "t")
    def u(x, t):
        return t**2

    constraint = ContinuousInitialFunctionConstraint(
        "u",
        domain,
        func=0.0,
        time_derivative_order=1,
        num_points=8,
        structure=structure,
    )
    assert _jit_loss(constraint, {"u": u}) < 1e-6

    constraint2 = ContinuousInitialFunctionConstraint(
        "u",
        domain,
        func=2.0,
        time_derivative_order=2,
        time_derivative_backend="jet",
        num_points=8,
        structure=structure,
    )
    assert _jit_loss(constraint2, {"u": u}) < 1e-6


def test_discrete_initial_constraint_zero():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component({"t": FixedStart()})

    @domain.Function("x", "t")
    def u(x, t):
        return t**2

    points = {
        "x": jnp.array([[0.25], [0.75]], dtype=float),
        "t": jnp.array([0.0, 0.0], dtype=float),
    }
    values = jnp.array([0.0, 0.0], dtype=float)

    constraint = DiscreteInitialConstraint(
        "u",
        component,
        points=points,
        values=values,
    )
    assert _jit_loss(constraint, {"u": u}) < 1e-6

    values2 = jnp.array([2.0, 2.0], dtype=float)
    constraint2 = DiscreteInitialConstraint(
        "u",
        component,
        points=points,
        values=values2,
        time_derivative_order=2,
        time_derivative_backend="jet",
    )
    assert _jit_loss(constraint2, {"u": u}) < 1e-6


def test_discrete_interior_data_constraint_points_zero():
    geom = Interval1d(0.0, 1.0)
    component = geom.component({"x": Boundary()})

    @geom.Function("x")
    def u(x):
        return x[0]

    points = {"x": jnp.array([[0.25], [0.75]], dtype=float)}
    values = jnp.array([0.25, 0.75], dtype=float)

    constraint = DiscreteInteriorDataConstraint(
        "u",
        geom,
        points=points,
        values=values,
        num_points=8,
        structure=ProductStructure((("x",),)),
    )
    assert _jit_loss(constraint, {"u": u}) < 1e-6


def test_discrete_interior_data_constraint_sensor_tracks_zero():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    structure = ProductStructure((("x", "t"),))

    @domain.Function("x", "t")
    def u(x, t):
        return 1.0

    sensors = jnp.array([[0.2], [0.8]], dtype=float)
    times = jnp.array([0.25, 0.75], dtype=float)
    sensor_values = jnp.ones((2, 2), dtype=float)

    constraint = DiscreteInteriorDataConstraint(
        "u",
        domain,
        sensors=sensors,
        times=times,
        sensor_values=sensor_values,
        num_points=16,
        structure=structure,
    )
    assert _jit_loss(constraint, {"u": u}) < 1e-6
