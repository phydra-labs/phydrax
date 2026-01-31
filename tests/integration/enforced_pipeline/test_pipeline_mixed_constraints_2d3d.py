#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.constraints import enforce_dirichlet, FunctionalConstraint
from phydrax.domain import (
    Boundary,
    FixedStart,
    PointsBatch,
    ProductStructure,
    TimeInterval,
)
from phydrax.domain.geometry2d import Square
from phydrax.domain.geometry3d import Cube
from phydrax.operators.differential import grad
from phydrax.solver import (
    EnforcedConstraintPipelines,
    EnforcedInteriorData,
    SingleFieldEnforcedConstraint,
)


def _paired_batch(domain, xs, ts):
    structure = ProductStructure((("x", "t"),)).canonicalize(domain.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    points = frozendict(
        {
            "x": cx.Field(jnp.asarray(xs, dtype=float), dims=(axis, None)),
            "t": cx.Field(jnp.asarray(ts, dtype=float).reshape((-1,)), dims=(axis,)),
        }
    )
    return PointsBatch(points=points, structure=structure)


def _eval(domain, u_enforced, xs, ts):
    batch = _paired_batch(domain, xs=xs, ts=ts)
    return jnp.asarray(u_enforced(batch).data).reshape((-1,))


def test_mixed_constraints_2d_transient():
    geom = Square(center=(0.0, 0.0), side=2.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return x[0] + 2.0 * x[1] + t

    left = domain.component({"x": Boundary()}, where={"x": lambda p: p[0] < -0.9})
    right = domain.component({"x": Boundary()}, where={"x": lambda p: p[0] > 0.9})
    initial = domain.component({"t": FixedStart()})

    constraints = [
        SingleFieldEnforcedConstraint(
            "u",
            left,
            lambda f: enforce_dirichlet(f, left, var="x", target=1.0),
        ),
        SingleFieldEnforcedConstraint(
            "u",
            right,
            lambda f: enforce_dirichlet(f, right, var="x", target=2.0),
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: enforce_dirichlet(f, initial, var="t", target=3.0),
            time_derivative_order=0,
            initial_target=3.0,
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: enforce_dirichlet(f, initial, var="t", target=3.0),
            time_derivative_order=1,
            initial_target=1.0,
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: enforce_dirichlet(f, initial, var="t", target=3.0),
            time_derivative_order=2,
            initial_target=0.0,
        ),
    ]

    sensors = jnp.array([[0.2, 0.1], [0.4, -0.2]], dtype=float)
    times = jnp.array([0.25, 0.75], dtype=float)
    sensor_values = jnp.array([[4.0, 5.0], [6.0, 7.0]], dtype=float)
    interior = EnforcedInteriorData(
        "u",
        sensors=sensors,
        times=times,
        sensor_values=sensor_values,
    )

    pipelines = EnforcedConstraintPipelines.build(
        functions={"u": u},
        constraints=constraints,
        interior_data=[interior],
        num_reference=256,
    )
    u_enforced = pipelines.apply({"u": u})["u"]

    out = _eval(
        domain,
        u_enforced,
        xs=jnp.array([[-1.0, 0.0], [1.0, 0.0]]),
        ts=jnp.array([0.5, 0.5]),
    )
    assert jnp.allclose(out[0], 1.0, atol=5e-2)
    assert jnp.allclose(out[1], 2.0, atol=5e-2)

    out = _eval(domain, u_enforced, xs=jnp.array([[0.0, 0.0]]), ts=jnp.array([0.0]))
    assert jnp.allclose(out, 3.0, atol=2e-2)

    xs = jnp.repeat(sensors, times.shape[0], axis=0)
    ts = jnp.tile(times, sensors.shape[0])
    expected = sensor_values.reshape((-1,))
    out = _eval(domain, u_enforced, xs=xs, ts=ts)
    assert jnp.allclose(out, expected, atol=1e-3)

    soft = FunctionalConstraint.from_operator(
        component=domain.component({"x": Boundary()}),
        operator=lambda f: grad(f, var="x"),
        constraint_vars="u",
        num_points=16,
        structure=ProductStructure((("x", "t"),)),
        reduction="mean",
    )
    loss = eqx.filter_jit(lambda: soft.loss({"u": u_enforced}))()
    assert jnp.isfinite(loss)


def test_mixed_constraints_3d_transient():
    geom = Cube(center=(0.0, 0.0, 0.0), side=2.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return x[0] + x[1] + x[2] + t

    left = domain.component({"x": Boundary()}, where={"x": lambda p: p[0] < -0.9})
    right = domain.component({"x": Boundary()}, where={"x": lambda p: p[0] > 0.9})
    initial = domain.component({"t": FixedStart()})

    constraints = [
        SingleFieldEnforcedConstraint(
            "u",
            left,
            lambda f: enforce_dirichlet(f, left, var="x", target=1.0),
        ),
        SingleFieldEnforcedConstraint(
            "u",
            right,
            lambda f: enforce_dirichlet(f, right, var="x", target=2.0),
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: enforce_dirichlet(f, initial, var="t", target=3.0),
            time_derivative_order=0,
            initial_target=3.0,
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: enforce_dirichlet(f, initial, var="t", target=3.0),
            time_derivative_order=1,
            initial_target=1.0,
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: enforce_dirichlet(f, initial, var="t", target=3.0),
            time_derivative_order=2,
            initial_target=0.0,
        ),
    ]

    sensors = jnp.array([[0.2, 0.1, -0.1], [0.4, -0.2, 0.3]], dtype=float)
    times = jnp.array([0.25, 0.75], dtype=float)
    sensor_values = jnp.array([[4.0, 5.0], [6.0, 7.0]], dtype=float)
    interior = EnforcedInteriorData(
        "u",
        sensors=sensors,
        times=times,
        sensor_values=sensor_values,
    )

    pipelines = EnforcedConstraintPipelines.build(
        functions={"u": u},
        constraints=constraints,
        interior_data=[interior],
        num_reference=256,
    )
    u_enforced = pipelines.apply({"u": u})["u"]

    out = _eval(
        domain,
        u_enforced,
        xs=jnp.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        ts=jnp.array([0.5, 0.5]),
    )
    assert jnp.allclose(out[0], 1.0, atol=5e-2)
    assert jnp.allclose(out[1], 2.0, atol=5e-2)

    out = _eval(domain, u_enforced, xs=jnp.array([[0.0, 0.0, 0.0]]), ts=jnp.array([0.0]))
    assert jnp.allclose(out, 3.0, atol=2e-2)

    xs = jnp.repeat(sensors, times.shape[0], axis=0)
    ts = jnp.tile(times, sensors.shape[0])
    expected = sensor_values.reshape((-1,))
    out = _eval(domain, u_enforced, xs=xs, ts=ts)
    assert jnp.allclose(out, expected, atol=1e-3)

    soft = FunctionalConstraint.from_operator(
        component=domain.component({"x": Boundary()}),
        operator=lambda f: grad(f, var="x"),
        constraint_vars="u",
        num_points=16,
        structure=ProductStructure((("x", "t"),)),
        reduction="mean",
    )
    loss = eqx.filter_jit(lambda: soft.loss({"u": u_enforced}))()
    assert jnp.isfinite(loss)
