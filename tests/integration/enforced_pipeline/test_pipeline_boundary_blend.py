#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.constraints import enforce_dirichlet
from phydrax.domain import Boundary, Interval1d, PointsBatch, ProductStructure
from phydrax.solver import EnforcedConstraintPipelines, SingleFieldEnforcedConstraint


def _line_batch(domain, xs):
    structure = ProductStructure((("x",),)).canonicalize(domain.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    points = frozendict(
        {"x": cx.Field(jnp.asarray(xs, dtype=float).reshape((-1, 1)), dims=(axis, None))}
    )
    return PointsBatch(points=points, structure=structure)


def test_boundary_subset_blend_matches_pieces():
    geom = Interval1d(0.0, 1.0)

    def left_where(x):
        return x[0] < 0.5

    def right_where(x):
        return x[0] >= 0.5

    @geom.Function("x")
    def u(x):
        return x[0] * 0.0

    left_component = geom.component({"x": Boundary()}, where={"x": left_where})
    right_component = geom.component({"x": Boundary()}, where={"x": right_where})

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

    pipelines = EnforcedConstraintPipelines.build(
        functions={"u": u},
        constraints=[left_constraint, right_constraint],
        num_reference=256,
    )
    u_enforced = pipelines.apply({"u": u})["u"]

    batch = _line_batch(geom, xs=jnp.array([0.0, 1.0]))
    out = jnp.asarray(u_enforced(batch).data).reshape((-1,))
    assert jnp.allclose(out[0], 1.0, atol=1e-3)
    assert jnp.allclose(out[1], 2.0, atol=1e-3)
