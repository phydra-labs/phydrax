#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.constraints import enforce_dirichlet
from phydrax.domain import (
    Boundary,
    FixedStart,
    Interval1d,
    PointsBatch,
    ProductStructure,
    TimeInterval,
)
from phydrax.operators.differential import partial_x
from phydrax.solver import (
    EnforcedConstraintPipelines,
    EnforcedInteriorData,
    SingleFieldEnforcedConstraint,
)


def _paired_batch_xp(domain, xs, ps):
    structure = ProductStructure((("x", "p"),)).canonicalize(domain.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    points = frozendict(
        {
            "x": cx.Field(
                jnp.asarray(xs, dtype=float).reshape((-1, 1)), dims=(axis, None)
            ),
            "p": cx.Field(
                jnp.asarray(ps, dtype=float).reshape((-1, 1)), dims=(axis, None)
            ),
        }
    )
    return PointsBatch(points=points, structure=structure)


def _paired_batch_xpt(domain, xs, ps, ts):
    structure = ProductStructure((("x", "p", "t"),)).canonicalize(domain.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    points = frozendict(
        {
            "x": cx.Field(
                jnp.asarray(xs, dtype=float).reshape((-1, 1)), dims=(axis, None)
            ),
            "p": cx.Field(
                jnp.asarray(ps, dtype=float).reshape((-1, 1)), dims=(axis, None)
            ),
            "t": cx.Field(jnp.asarray(ts, dtype=float).reshape((-1,)), dims=(axis,)),
        }
    )
    return PointsBatch(points=points, structure=structure)


def test_xp_steady_state_explicit_anchors():
    geom = Interval1d(0.0, 1.0)
    momentum = Interval1d(-1.0, 1.0).relabel("p")
    domain = geom @ momentum

    @domain.Function("x", "p")
    def u(x, p):
        return x[0] + 2.0 * p[0]

    left = domain.component({"x": Boundary()}, where={"x": lambda p: p[0] < 0.5})
    right = domain.component({"x": Boundary()}, where={"x": lambda p: p[0] > 0.5})

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
    ]

    anchors = {
        "x": jnp.array([[0.25], [0.75]], dtype=float),
        "p": jnp.array([[-0.5], [0.5]], dtype=float),
    }
    values = jnp.array([3.0, 4.0], dtype=float)
    interior = EnforcedInteriorData("u", points=anchors, values=values)

    pipelines = EnforcedConstraintPipelines.build(
        functions={"u": u},
        constraints=constraints,
        interior_data=[interior],
        num_reference=128,
    )
    u_enforced = pipelines.apply({"u": u})["u"]
    eval_jit = eqx.filter_jit(lambda b: u_enforced(b).data)

    batch = _paired_batch_xp(
        domain,
        xs=jnp.array([[0.0], [1.0]], dtype=float),
        ps=jnp.array([[0.0], [0.0]], dtype=float),
    )
    out = eval_jit(batch).reshape((-1,))
    assert jnp.allclose(out, jnp.array([1.0, 2.0]), atol=5e-2)

    batch = _paired_batch_xp(domain, xs=anchors["x"], ps=anchors["p"])
    out = eval_jit(batch).reshape((-1,))
    assert jnp.allclose(out, values, atol=1e-3)


def test_xp_coord_separable_partials():
    geom = Interval1d(0.0, 1.0)
    momentum = Interval1d(-1.0, 1.0).relabel("p")
    domain = geom @ momentum

    @domain.Function("x", "p")
    def u(x, p):
        return x[0] ** 2 + 3.0 * p[0]

    component = domain.component()
    sep = component.sample_coord_separable({"x": 6, "p": 5}, num_points=())

    du_dx = partial_x(u, var="x")
    du_dp = partial_x(u, var="p")
    eval_jit = eqx.filter_jit(lambda f, b: f(b).data)

    dx = eval_jit(du_dx, sep)
    dp = eval_jit(du_dp, sep)

    x_axis = sep.points["x"][0].data
    p_axis = sep.points["p"][0].data
    xx, pp = jnp.meshgrid(x_axis, p_axis, indexing="ij")
    assert jnp.allclose(dx, 2.0 * xx, atol=1e-5)
    assert jnp.allclose(dp, 3.0 * jnp.ones_like(pp), atol=1e-5)


def test_xpt_transient_explicit_anchors():
    geom = Interval1d(0.0, 1.0)
    momentum = Interval1d(-1.0, 1.0).relabel("p")
    time = TimeInterval(0.0, 1.0)
    domain = geom @ momentum @ time

    @domain.Function("x", "p", "t")
    def u(x, p, t):
        return x[0] + p[0] + t

    left = domain.component({"x": Boundary()}, where={"x": lambda p: p[0] < 0.5})
    right = domain.component({"x": Boundary()}, where={"x": lambda p: p[0] > 0.5})
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

    anchors = {
        "x": jnp.array([[0.25], [0.75]], dtype=float),
        "p": jnp.array([[-0.5], [0.5]], dtype=float),
        "t": jnp.array([0.4, 0.6], dtype=float),
    }
    values = jnp.array([4.0, 5.0], dtype=float)
    interior = EnforcedInteriorData("u", points=anchors, values=values)

    pipelines = EnforcedConstraintPipelines.build(
        functions={"u": u},
        constraints=constraints,
        interior_data=[interior],
        num_reference=128,
    )
    u_enforced = pipelines.apply({"u": u})["u"]
    eval_jit = eqx.filter_jit(lambda b: u_enforced(b).data)

    batch = _paired_batch_xpt(
        domain,
        xs=jnp.array([[0.0], [1.0]], dtype=float),
        ps=jnp.array([[0.0], [0.0]], dtype=float),
        ts=jnp.array([0.5, 0.5], dtype=float),
    )
    out = eval_jit(batch).reshape((-1,))
    assert jnp.allclose(out, jnp.array([1.0, 2.0]), atol=5e-2)

    batch = _paired_batch_xpt(
        domain,
        xs=jnp.array([[0.5]], dtype=float),
        ps=jnp.array([[0.0]], dtype=float),
        ts=jnp.array([0.0], dtype=float),
    )
    out = eval_jit(batch).reshape((-1,))
    assert jnp.allclose(out, 3.0, atol=2e-2)

    batch = _paired_batch_xpt(domain, xs=anchors["x"], ps=anchors["p"], ts=anchors["t"])
    out = eval_jit(batch).reshape((-1,))
    assert jnp.allclose(out, values, atol=1e-3)


def test_xpt_coord_separable_partials():
    geom = Interval1d(0.0, 1.0)
    momentum = Interval1d(-1.0, 1.0).relabel("p")
    time = TimeInterval(0.0, 1.0)
    domain = geom @ momentum @ time

    @domain.Function("x", "p", "t")
    def u(x, p, t):
        return x[0] ** 2 + 3.0 * p[0] + t

    component = domain.component()
    structure = ProductStructure((("t",),))
    sep = component.sample_coord_separable(
        {"x": 5, "p": 4},
        num_points=6,
        dense_structure=structure,
    )

    du_dx = partial_x(u, var="x")
    du_dp = partial_x(u, var="p")
    eval_jit = eqx.filter_jit(lambda f, b: f(b).data)

    dx = eval_jit(du_dx, sep)
    dp = eval_jit(du_dp, sep)

    t_axis = sep.points["t"].data
    x_axis = sep.points["x"][0].data
    p_axis = sep.points["p"][0].data
    tt, xx, pp = jnp.meshgrid(t_axis, x_axis, p_axis, indexing="ij")
    assert jnp.allclose(dx, 2.0 * xx, atol=1e-5)
    assert jnp.allclose(dp, 3.0 * jnp.ones_like(pp), atol=1e-5)
