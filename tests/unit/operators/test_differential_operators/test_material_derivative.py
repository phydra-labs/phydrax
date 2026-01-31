#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, Square, TimeInterval
from phydrax.operators.differential import material_derivative


def test_material_derivative_scalar_point():
    geom = Square(center=(0.0, 0.0), side=2.0)
    dom = geom @ TimeInterval(0.0, 1.0)

    @dom.Function("x", "t")
    def u(x, t):
        return x[0] ** 2 + x[1] ** 2 + t

    v = DomainFunction(domain=geom, deps=("x",), func=lambda x: jnp.array([x[1], -x[0]]))
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
    geom = Square(center=(0.0, 0.0), side=2.0)
    dom = geom @ TimeInterval(0.0, 1.0)

    @dom.Function("x")
    def u(x):
        return jnp.array([x[0] ** 2, x[1] ** 2])

    v = DomainFunction(domain=geom, deps=("x",), func=lambda x: jnp.array([x[1], -x[0]]))
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
    geom = Square(center=(0.0, 0.0), side=2.0)
    dom = geom @ TimeInterval(0.0, 1.0)

    u = DomainFunction(
        domain=dom, deps=("x", "t"), func=lambda x, t: x[0] + t, metadata={"tag": 1}
    )
    v = DomainFunction(
        domain=geom, deps=("x",), func=lambda x: jnp.array([0.0 * x[0], 0.0 * x[0]])
    )
    assert material_derivative(u, v).metadata == u.metadata
