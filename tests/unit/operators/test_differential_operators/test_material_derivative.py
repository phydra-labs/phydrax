#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

import phydrax as phx
from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, TimeInterval
from phydrax.operators.differential import material_derivative


def test_material_derivative_scalar_point():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    dom = geom @ TimeInterval(0.0, 1.0)

    @dom.Function("x", "t")
    def u(x, t):
        return x[0] ** 2 + x[1] ** 2 + t

    v = geom.Function("x")(lambda x: jnp.array([x[1], -x[0]]))
    DuDt = material_derivative(u, v)

    pts = frozendict(
        {
            "x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,)),
            "t": cx.Field(jnp.array(1.0), dims=()),
        }
    )
    out = jnp.asarray(DuDt(pts).data)
    assert jnp.allclose(out, 1.0)


def test_material_derivative_vector_point():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    dom = geom @ TimeInterval(0.0, 1.0)

    @dom.Function("x")
    def u(x):
        return jnp.array([x[0] ** 2, x[1] ** 2])

    v = geom.Function("x")(lambda x: jnp.array([x[1], -x[0]]))
    DuDt = material_derivative(u, v)

    pts = frozendict(
        {
            "x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,)),
            "t": cx.Field(jnp.array(0.5), dims=()),
        }
    )
    out = jnp.asarray(DuDt(pts).data)
    assert jnp.allclose(out, jnp.array([12.0, -12.0]))


def test_material_derivative_preserves_metadata():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    dom = geom @ TimeInterval(0.0, 1.0)

    u = dom.Function("x", "t")(lambda x, t: x[0] + t).with_metadata(**{"tag": 1})
    v = geom.Function("x")(lambda x: jnp.array([0.0 * x[0], 0.0 * x[0]]))
    assert material_derivative(u, v).metadata == u.metadata


def test_material_derivative_ad_engine_jvp_matches_default():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    dom = geom @ TimeInterval(0.0, 1.0)

    @dom.Function("x", "t")
    def u(x, t):
        return x[0] ** 2 + x[1] ** 2 + t**3

    v = DomainFunction(domain=geom, deps=(), func=jnp.array([1.0, -0.5]))

    pts = frozendict(
        {
            "x": cx.Field(jnp.array([0.25, -0.75]), dims=(None,)),
            "t": cx.Field(jnp.array(0.4), dims=()),
        }
    )
    out_ref = jnp.asarray(material_derivative(u, v)(pts).data)
    out_jvp = jnp.asarray(material_derivative(u, v, ad_engine="jvp")(pts).data)
    assert jnp.allclose(out_jvp, out_ref, atol=1e-6)
