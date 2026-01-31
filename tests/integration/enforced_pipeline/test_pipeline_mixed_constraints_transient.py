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
    Interval1d,
    PointsBatch,
    ProductStructure,
    TimeInterval,
)
from phydrax.operators.differential import dt
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
            "x": cx.Field(
                jnp.asarray(xs, dtype=float).reshape((-1, 1)), dims=(axis, None)
            ),
            "t": cx.Field(jnp.asarray(ts, dtype=float).reshape((-1,)), dims=(axis,)),
        }
    )
    return PointsBatch(points=points, structure=structure)


def test_mixed_constraints_transient():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return x[0] + t

    boundary_component = domain.component({"x": Boundary()})
    initial_component = domain.component({"t": FixedStart()})

    boundary_constraint = SingleFieldEnforcedConstraint(
        "u",
        boundary_component,
        lambda f: enforce_dirichlet(f, boundary_component, var="x", target=5.0),
    )
    initial_constraint = SingleFieldEnforcedConstraint(
        "u",
        initial_component,
        lambda f: enforce_dirichlet(f, initial_component, var="t", target=2.0),
    )

    anchors = {
        "x": jnp.array([[0.25], [0.75]], dtype=float),
        "t": jnp.array([0.6, 0.4], dtype=float),
    }
    anchor_values = jnp.array([3.0, 4.0], dtype=float)
    interior = EnforcedInteriorData("u", points=anchors, values=anchor_values)

    pipelines = EnforcedConstraintPipelines.build(
        functions={"u": u},
        constraints=[boundary_constraint, initial_constraint],
        interior_data=[interior],
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

    constraint = FunctionalConstraint.from_operator(
        component=domain.component(),
        operator=lambda f: dt(f, var="t"),
        constraint_vars="u",
        num_points=16,
        structure=ProductStructure((("x", "t"),)),
        reduction="mean",
    )
    loss_fn = eqx.filter_jit(lambda: constraint.loss({"u": u_enforced}))
    loss = loss_fn()
    assert jnp.isfinite(loss)
