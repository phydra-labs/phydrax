#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax.numpy as jnp

import phydrax as phx
from phydrax._frozendict import frozendict
from phydrax.domain import (
    Boundary,
    FixedStart,
    PointBatch,
    SampleLayout,
    TimeInterval,
)
from phydrax.enforcement import (
    EnforcementProgram,
    EnforcementSpec,
    InteriorAnchors,
)


def _paired_batch(domain, xs, ts):
    structure = SampleLayout((("x", "t"),)).canonicalize(domain.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    points = frozendict(
        {
            "x": cx.Field(jnp.asarray(xs, dtype=float), dims=(axis, None)),
            "t": cx.Field(jnp.asarray(ts, dtype=float).reshape((-1,)), dims=(axis,)),
        }
    )
    return PointBatch(points=points, structure=structure)


def _eval(domain, u_enforced, xs, ts):
    batch = _paired_batch(domain, xs=xs, ts=ts)
    return jnp.asarray(u_enforced(batch).data).reshape((-1,))


def test_mixed_constraints_2d_transient():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return x[0] + 2.0 * x[1] + t

    left = domain.component({"x": Boundary()}, where={"x": lambda p: p[0] < -0.9})
    right = domain.component({"x": Boundary()}, where={"x": lambda p: p[0] > 0.9})
    full_boundary = domain.component({"x": Boundary()})
    initial = domain.component({"t": FixedStart()})

    specs = [
        EnforcementSpec(phx.conditions.Dirichlet("u", left, target=1.0)),
        EnforcementSpec(phx.conditions.Dirichlet("u", right, target=2.0)),
        EnforcementSpec(phx.conditions.Initial("u", initial, target=3.0, order=0)),
        EnforcementSpec(phx.conditions.Initial("u", initial, target=1.0, order=1)),
        EnforcementSpec(phx.conditions.Initial("u", initial, target=0.0, order=2)),
    ]

    sensors = jnp.array([[0.2, 0.1], [0.4, -0.2]], dtype=float)
    times = jnp.array([0.25, 0.75], dtype=float)
    sensor_values = jnp.array([[4.0, 5.0], [6.0, 7.0]], dtype=float)
    interior = InteriorAnchors(
        "u",
        sensors=sensors,
        times=times,
        sensor_values=sensor_values,
    )

    program = EnforcementProgram.build(
        functions={"u": u},
        specs=specs,
        interior=[interior],
        num_reference=256,
    )
    u_enforced = program.apply({"u": u})["u"]

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

    soft_condition = phx.conditions.Neumann(
        "u",
        domain.component({"x": Boundary()}),
        var="x",
        target=0.0,
    )
    soft = phx.terms.ResidualPenalty(
        soft_condition,
        phx.integration.per_step(
            phx.integration.mean_over(soft_condition.on),
            phx.integration.MonteCarloPlan(16),
        ),
    )
    loss = eqx.filter_jit(lambda: soft.loss({"u": u_enforced}))()
    assert jnp.isfinite(loss)


def test_mixed_constraints_3d_transient():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Cube(center=(0.0, 0.0, 0.0), side=2.0).compile()
    )
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return x[0] + x[1] + x[2] + t

    left = domain.component({"x": Boundary()}, where={"x": lambda p: p[0] < -0.9})
    right = domain.component({"x": Boundary()}, where={"x": lambda p: p[0] > 0.9})
    full_boundary = domain.component({"x": Boundary()})
    initial = domain.component({"t": FixedStart()})

    specs = [
        EnforcementSpec(phx.conditions.Dirichlet("u", left, target=1.0)),
        EnforcementSpec(phx.conditions.Dirichlet("u", right, target=2.0)),
        EnforcementSpec(phx.conditions.Initial("u", initial, target=3.0, order=0)),
        EnforcementSpec(phx.conditions.Initial("u", initial, target=1.0, order=1)),
        EnforcementSpec(phx.conditions.Initial("u", initial, target=0.0, order=2)),
    ]

    sensors = jnp.array([[0.2, 0.1, -0.1], [0.4, -0.2, 0.3]], dtype=float)
    times = jnp.array([0.25, 0.75], dtype=float)
    sensor_values = jnp.array([[4.0, 5.0], [6.0, 7.0]], dtype=float)
    interior = InteriorAnchors(
        "u",
        sensors=sensors,
        times=times,
        sensor_values=sensor_values,
    )

    program = EnforcementProgram.build(
        functions={"u": u},
        specs=specs,
        interior=[interior],
        num_reference=256,
    )
    u_enforced = program.apply({"u": u})["u"]

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

    soft_condition = phx.conditions.Neumann(
        "u",
        domain.component({"x": Boundary()}),
        var="x",
        target=0.0,
    )
    soft = phx.terms.ResidualPenalty(
        soft_condition,
        phx.integration.per_step(
            phx.integration.mean_over(soft_condition.on),
            phx.integration.MonteCarloPlan(16),
        ),
    )
    loss = eqx.filter_jit(lambda: soft.loss({"u": u_enforced}))()
    assert jnp.isfinite(loss)
