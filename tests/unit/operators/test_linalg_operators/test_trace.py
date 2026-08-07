#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

import phydrax as phx
from phydrax._frozendict import frozendict
from phydrax.domain import TimeInterval
from phydrax.operators.linalg import trace


def test_trace_simple_matrix_function():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def u(x):
        return jnp.array([[x[0], 0], [0, x[1]]])

    trace_u = trace(u)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    result = jnp.asarray(trace_u(pts).data)

    expected = 5.0
    assert jnp.allclose(result, expected)


def test_trace_time_dependent_matrix_function():
    dom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    ) @ TimeInterval(0.0, 1.0)

    @dom.Function("x", "t")
    def u(x, t):
        return jnp.array([[x[0] * t, 0.0], [0.0, x[1] * t]])

    trace_u = trace(u)
    pts = frozendict(
        {
            "x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,)),
            "t": cx.Field(jnp.array(0.5), dims=()),
        }
    )
    result = jnp.asarray(trace_u(pts).data)

    expected = 2.5  # 0.5*(2+3)
    assert jnp.allclose(result, expected)


def test_trace_complex_function():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def u(x):
        return jnp.array([[x[0], 0], [0, 1j * x[1]]])

    trace_u = trace(u)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    result = jnp.asarray(trace_u(pts).data)

    expected = 2.0 + 3.0j
    assert jnp.allclose(result, expected)


def test_trace_preserves_metadata():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    u = geom.Function("x")(lambda x: jnp.eye(2)).with_metadata(**{"tag": "keep"})
    tr = trace(u)
    assert tr.metadata == u.metadata
