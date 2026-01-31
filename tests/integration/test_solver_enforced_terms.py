#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from phydrax._frozendict import frozendict
from phydrax.constraints import ContinuousInitialConstraint, enforce_dirichlet
from phydrax.domain import (
    Boundary,
    FixedStart,
    Interval1d,
    PointsBatch,
    ProductStructure,
    TimeInterval,
)
from phydrax.domain._structure import CoordSeparableBatch
from phydrax.solver import (
    EnforcedInteriorData,
    FunctionalSolver,
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


def test_functional_solver_builds_enforced_pipeline_terms():
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
    interior = EnforcedInteriorData(
        "u",
        points=anchors,
        values=anchor_values,
        eps_snap=1e-12,
    )

    solver = FunctionalSolver(
        functions={"u": u},
        constraints=(),
        constraint_terms=[boundary_constraint, initial_constraint],
        interior_data_terms=[interior],
        boundary_weight_num_reference=256,
    )
    u_enforced = solver.ansatz_functions()["u"]
    eval_jit = eqx.filter_jit(lambda b: u_enforced(b).data)

    out = eval_jit(
        _paired_batch(domain, xs=jnp.array([0.0, 1.0]), ts=jnp.array([0.5, 0.5]))
    )
    assert jnp.allclose(out.reshape((-1,)), 5.0, atol=1e-3)

    out = eval_jit(
        _paired_batch(domain, xs=jnp.array([0.5, 0.5]), ts=jnp.array([0.0, 0.0]))
    )
    assert jnp.allclose(out.reshape((-1,)), 2.0, atol=1e-3)

    out = eval_jit(
        _paired_batch(domain, xs=jnp.array([0.25, 0.75]), ts=jnp.array([0.6, 0.4]))
    )
    assert jnp.allclose(out.reshape((-1,)), anchor_values, atol=1e-3)


def test_initial_constraint_coord_separable_spatial():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return 0.0

    initial_component = domain.component({"t": FixedStart()})
    structure = ProductStructure((("x",),))

    constraint = ContinuousInitialConstraint(
        "u",
        initial_component,
        func=0.0,
        num_points=(),
        structure=structure,
        coord_separable={"x": 5},
    )

    batch = constraint.sample(key=jr.key(0))
    assert isinstance(batch, CoordSeparableBatch)
    assert isinstance(batch.points["x"], tuple)
    assert len(batch.points["x"]) == 1

    loss = constraint.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0, atol=1e-6)
