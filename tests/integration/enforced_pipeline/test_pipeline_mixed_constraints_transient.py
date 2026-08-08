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
    Interval1d,
    PointBatch,
    SampleLayout,
    TimeInterval,
)
from phydrax.enforcement import (
    EnforcementProgram,
    EnforcementSpec,
    InteriorAnchors,
)
from phydrax.operators.differential import dt


def _paired_batch(domain, xs, ts):
    structure = SampleLayout((("x", "t"),)).canonicalize(domain.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    points = frozendict(
        {
            "x": cx.Field(
                jnp.asarray(xs, dtype=float).reshape((-1, 1)), dims=(axis, None)
            ),
            "t": cx.Field(jnp.asarray(ts, dtype=float).reshape((-1,)), dims=(axis,)),
        }
    )
    return PointBatch(points=points, structure=structure)


def test_mixed_constraints_transient():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return x[0] + t

    boundary_component = domain.component({"x": Boundary()})
    initial_component = domain.component({"t": FixedStart()})

    boundary_constraint = EnforcementSpec(
        phx.conditions.Dirichlet("u", boundary_component, target=5.0)
    )
    initial_constraint = EnforcementSpec(
        phx.conditions.Initial("u", initial_component, target=2.0)
    )

    anchors = {
        "x": jnp.array([[0.25], [0.75]], dtype=float),
        "t": jnp.array([0.6, 0.4], dtype=float),
    }
    anchor_values = jnp.array([3.0, 4.0], dtype=float)
    interior = InteriorAnchors("u", points=anchors, values=anchor_values)

    pipelines = EnforcementProgram.build(
        functions={"u": u},
        specs=[boundary_constraint, initial_constraint],
        interior=[interior],
        num_reference=256,
    )
    u_enforced = pipelines.apply({"u": u})["u"]

    batch = _paired_batch(domain, xs=jnp.array([0.0, 1.0]), ts=jnp.array([0.5, 0.5]))
    out = jnp.asarray(u_enforced(batch).data).reshape((-1,))
    assert jnp.allclose(out, 5.0, atol=1e-3)

    batch = _paired_batch(domain, xs=jnp.array([0.5, 0.5]), ts=jnp.array([0.0, 0.0]))
    out = jnp.asarray(u_enforced(batch).data).reshape((-1,))
    assert jnp.allclose(out, 2.0, atol=1e-2)

    batch = _paired_batch(domain, xs=jnp.array([0.25, 0.75]), ts=jnp.array([0.6, 0.4]))
    out = jnp.asarray(u_enforced(batch).data).reshape((-1,))
    assert jnp.allclose(out, anchor_values, atol=1e-3)

    condition = phx.conditions.Residual(
        "u",
        domain.component(),
        lambda f: dt(f, var="t"),
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
