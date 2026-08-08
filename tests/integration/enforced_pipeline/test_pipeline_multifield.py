#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax._frozendict import frozendict
from phydrax.domain import Boundary, Interval1d, PointBatch, SampleLayout
from phydrax.enforcement import EnforcementProgram, EnforcementSpec


def _line_batch(domain, xs):
    structure = SampleLayout((("x",),)).canonicalize(domain.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    points = frozendict(
        {"x": cx.Field(jnp.asarray(xs, dtype=float).reshape((-1, 1)), dims=(axis, None))}
    )
    return PointBatch(points=points, structure=structure)


def test_multifield_pipeline_uses_enforced_covars():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] * 0.0

    @geom.Function("x")
    def v(x):
        return x[0]

    boundary_component = geom.component({"x": Boundary()})

    v_condition = phx.conditions.Dirichlet(
        "v",
        boundary_component,
        target=lambda x: x[0] + 2.0,
    )
    v_spec = EnforcementSpec(
        v_condition,
        kind="custom",
        transform=lambda value, _get_field: value + 2.0,
    )
    u_condition = phx.conditions.Residual(
        ("u", "v"),
        boundary_component,
        lambda first, second: first - second,
    )
    u_spec = EnforcementSpec(
        u_condition,
        field="u",
        kind="custom",
        transform=lambda _value, get_field: get_field("v"),
    )

    pipelines = EnforcementProgram.build(functions={"u": u, "v": v}, specs=[u_spec, v_spec], )
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

    u_condition = phx.conditions.Residual(
        ("u", "v"),
        boundary_component,
        lambda first, second: first - second,
    )
    v_condition = phx.conditions.Residual(
        ("v", "u"),
        boundary_component,
        lambda first, second: first - second,
    )
    u_spec = EnforcementSpec(
        u_condition,
        field="u",
        kind="custom",
        transform=lambda _value, get_field: get_field("v"),
    )
    v_spec = EnforcementSpec(
        v_condition,
        field="v",
        kind="custom",
        transform=lambda _value, get_field: get_field("u"),
    )

    with pytest.raises(ValueError, match="dependency cycle"):
        EnforcementProgram.build(functions={"u": u, "v": v}, specs=[u_spec, v_spec], )
