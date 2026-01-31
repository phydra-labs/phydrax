#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import pytest

from phydrax._frozendict import frozendict
from phydrax.domain import Boundary, Interval1d, PointsBatch, ProductStructure
from phydrax.solver import (
    EnforcedConstraintPipelines,
    MultiFieldEnforcedConstraint,
    SingleFieldEnforcedConstraint,
)


def _line_batch(domain, xs):
    structure = ProductStructure((("x",),)).canonicalize(domain.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    points = frozendict(
        {"x": cx.Field(jnp.asarray(xs, dtype=float).reshape((-1, 1)), dims=(axis, None))}
    )
    return PointsBatch(points=points, structure=structure)


def test_multifield_pipeline_uses_enforced_covars():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] * 0.0

    @geom.Function("x")
    def v(x):
        return x[0]

    boundary_component = geom.component({"x": Boundary()})

    v_constraint = SingleFieldEnforcedConstraint(
        "v",
        boundary_component,
        lambda f: f + 2.0,
    )

    u_constraint = MultiFieldEnforcedConstraint(
        "u",
        boundary_component,
        ("v",),
        lambda _u, get_field: get_field("v"),
    )

    pipelines = EnforcedConstraintPipelines.build(
        functions={"u": u, "v": v},
        constraints=[u_constraint, v_constraint],
    )
    enforced = pipelines.apply({"u": u, "v": v})

    batch = _line_batch(geom, xs=jnp.array([0.3, 0.7]))
    out_u = jnp.asarray(enforced["u"](batch).data).reshape((-1,))
    out_v = jnp.asarray(enforced["v"](batch).data).reshape((-1,))
    assert jnp.allclose(out_u, out_v, atol=1e-6)
    assert jnp.allclose(out_v, jnp.array([2.3, 2.7]), atol=1e-6)


def test_multifield_pipeline_cycle_error():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0]

    @geom.Function("x")
    def v(x):
        return x[0] * 2.0

    boundary_component = geom.component({"x": Boundary()})

    u_constraint = MultiFieldEnforcedConstraint(
        "u",
        boundary_component,
        ("v",),
        lambda _u, get_field: get_field("v"),
    )
    v_constraint = MultiFieldEnforcedConstraint(
        "v",
        boundary_component,
        ("u",),
        lambda _v, get_field: get_field("u"),
    )

    with pytest.raises(ValueError, match="dependency cycle"):
        EnforcedConstraintPipelines.build(
            functions={"u": u, "v": v},
            constraints=[u_constraint, v_constraint],
        )
