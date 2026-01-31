#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.constraints import enforce_dirichlet, FunctionalConstraint
from phydrax.domain import Boundary, Interval1d, PointsBatch, ProductStructure
from phydrax.operators.differential import grad
from phydrax.solver import (
    EnforcedConstraintPipelines,
    EnforcedInteriorData,
    SingleFieldEnforcedConstraint,
)


def _batch(domain, xs):
    structure = ProductStructure((("x",),)).canonicalize(domain.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    points = frozendict(
        {"x": cx.Field(jnp.asarray(xs, dtype=float).reshape((-1, 1)), dims=(axis, None))}
    )
    return PointsBatch(points=points, structure=structure)


def test_mixed_constraints_steady_state():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 2

    left_component = geom.component({"x": Boundary()}, where={"x": lambda p: p[0] < 0.5})
    right_component = geom.component(
        {"x": Boundary()}, where={"x": lambda p: p[0] >= 0.5}
    )

    left_constraint = SingleFieldEnforcedConstraint(
        "u",
        left_component,
        lambda f: enforce_dirichlet(f, left_component, var="x", target=1.0),
    )
    right_constraint = SingleFieldEnforcedConstraint(
        "u",
        right_component,
        lambda f: enforce_dirichlet(f, right_component, var="x", target=2.0),
    )

    anchors = {"x": jnp.array([[0.25], [0.75]], dtype=float)}
    anchor_values = jnp.array([3.0, 4.0], dtype=float)
    interior = EnforcedInteriorData("u", points=anchors, values=anchor_values)

    pipelines = EnforcedConstraintPipelines.build(
        functions={"u": u},
        constraints=[left_constraint, right_constraint],
        interior_data=[interior],
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

    constraint = FunctionalConstraint.from_operator(
        component=geom.component(),
        operator=lambda f: grad(f, var="x"),
        constraint_vars="u",
        num_points=16,
        structure=ProductStructure((("x",),)),
        reduction="mean",
    )
    loss_fn = eqx.filter_jit(lambda: constraint.loss({"u": u_enforced}))
    loss = loss_fn()
    assert jnp.isfinite(loss)
