#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax.numpy as jnp

import phydrax as phx
from phydrax._frozendict import frozendict
from phydrax.domain import Boundary, Interval1d, PointBatch, SampleLayout
from phydrax.enforcement import (
    enforce_dirichlet,
    EnforcementProgram,
    EnforcementSpec,
    InteriorAnchors,
)
from phydrax.operators.differential import grad


def _batch(domain, xs):
    structure = SampleLayout((("x",),)).canonicalize(domain.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    points = frozendict(
        {"x": cx.Field(jnp.asarray(xs, dtype=float).reshape((-1, 1)), dims=(axis, None))}
    )
    return PointBatch(points=points, structure=structure)


def test_mixed_constraints_steady_state():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 2

    left_component = geom.component({"x": Boundary()}, where={"x": lambda p: p[0] < 0.5})
    right_component = geom.component(
        {"x": Boundary()}, where={"x": lambda p: p[0] >= 0.5}
    )
    full_boundary = geom.component({"x": Boundary()})

    left_constraint = EnforcementSpec(
        phx.conditions.Dirichlet("u", left_component, target=1.0),
        kind="custom",
        transform=lambda f, _: enforce_dirichlet(f, full_boundary, var="x", target=1.0),
    )
    right_constraint = EnforcementSpec(
        phx.conditions.Dirichlet("u", right_component, target=2.0),
        kind="custom",
        transform=lambda f, _: enforce_dirichlet(f, full_boundary, var="x", target=2.0),
    )

    anchors = {"x": jnp.array([[0.25], [0.75]], dtype=float)}
    anchor_values = jnp.array([3.0, 4.0], dtype=float)
    interior = InteriorAnchors("u", points=anchors, values=anchor_values)

    pipelines = EnforcementProgram.build(
        functions={"u": u},
        specs=[left_constraint, right_constraint],
        interior=[interior],
        num_reference=256,
    )
    u_enforced = pipelines.apply({"u": u})["u"]

    batch = _batch(geom, xs=jnp.array([0.0, 1.0], dtype=float))
    out = jnp.asarray(u_enforced(batch).data).reshape((-1,))
    assert jnp.allclose(out[0], 1.0, atol=1e-3)
    assert jnp.allclose(out[1], 2.0, atol=1e-3)

    batch = _batch(geom, xs=jnp.array([0.25, 0.75], dtype=float))
    out = jnp.asarray(u_enforced(batch).data).reshape((-1,))
    assert jnp.allclose(out, anchor_values, atol=1e-3)

    condition = phx.conditions.Residual(
        "u",
        geom.component(),
        lambda f: grad(f, var="x"),
    )
    term = phx.terms.ResidualPenalty(
        condition,
        phx.integration.per_step(
            phx.integration.mean_over(condition.on),
            phx.integration.MonteCarloPlan(16),
        ),
    )
    loss_fn = eqx.filter_jit(lambda: term.loss({"u": u_enforced}))
    loss = loss_fn()
    assert jnp.isfinite(loss)
