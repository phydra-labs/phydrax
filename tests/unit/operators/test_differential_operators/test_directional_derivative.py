#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, TimeInterval
from phydrax.operators.differential import directional_derivative


def test_directional_derivative_scalar_point():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    v = DomainFunction(domain=geom, deps=(), func=jnp.array([1.0, 0.0]))
    dd = directional_derivative(f, v)

    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(dd(pts).data)
    assert jnp.allclose(out, 4.0)


def test_directional_derivative_vector_point():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def f(x):
        return jnp.array([x[0] ** 2, x[1] ** 2])

    v = DomainFunction(domain=geom, deps=(), func=jnp.array([1.0, 1.0]))
    dd = directional_derivative(f, v)

    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(dd(pts).data)
    assert jnp.allclose(out, jnp.array([4.0, 6.0]))


def test_directional_derivative_direction_is_function():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    v = geom.Function("x")(lambda x: x)
    dd = directional_derivative(f, v)

    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(dd(pts).data)
    assert jnp.allclose(out, 26.0)


def test_directional_derivative_spacetime_broadcasts_over_t(sample_batch):
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    dom = geom @ TimeInterval(0.0, 1.0)

    @dom.Function("x", "t")
    def f(x, t):
        return x[0] ** 2 + x[1] ** 2 + t

    v = DomainFunction(domain=geom, deps=(), func=jnp.array([1.0, 0.0]))
    dd = directional_derivative(f, v, var="x")

    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(3, 4), key=0)
    out = jnp.asarray(dd(batch).data)
    assert out.shape == (3, 4)
    x = jnp.asarray(batch.points["x"].data)
    assert jnp.allclose(out, 2.0 * x[..., 0:1])


def test_directional_derivative_coord_separable(sample_grid):
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    component = geom.component()
    batch = sample_grid(component, {"x": (5, 4)}, dense_blocks=(), key=0)

    @geom.Function("x")
    def f(x):
        x, y = x
        return x**2 + y**2

    v = DomainFunction(domain=geom, deps=(), func=jnp.array([1.0, 0.0]))
    dd = directional_derivative(f, v)
    out = jnp.asarray(dd(batch).data)

    xs = jnp.asarray(batch.points["x"][0].data)
    ys = jnp.asarray(batch.points["x"][1].data)
    X, _ = jnp.meshgrid(xs, ys, indexing="ij")
    assert jnp.allclose(out, 2.0 * X, atol=1e-6)


def test_directional_derivative_preserves_metadata():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    u = geom.Function("x")(lambda x: x[0] ** 2).with_metadata(**{"scale": 7})
    v = DomainFunction(domain=geom, deps=(), func=jnp.array([1.0, 0.0]))
    assert directional_derivative(u, v).metadata == u.metadata


def test_directional_derivative_ad_engine_jvp_matches_default():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def f(x):
        return x[0] ** 2 + x[1] ** 2 + x[0] * x[1]

    v = DomainFunction(domain=geom, deps=(), func=jnp.array([1.0, -0.25]))
    pts = frozendict({"x": cx.Field(jnp.array([0.2, -0.4]), dims=(None,))})
    out_ref = jnp.asarray(directional_derivative(f, v, backend="ad")(pts).data)
    out_jvp = jnp.asarray(
        directional_derivative(f, v, backend="ad", ad_engine="jvp")(pts).data
    )
    assert jnp.allclose(out_jvp, out_ref, atol=1e-6)


def test_directional_derivative_ad_engine_requires_ad_backend():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    v = DomainFunction(domain=geom, deps=(), func=jnp.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="backend='ad'"):
        directional_derivative(f, v, backend="fd", ad_engine="jvp")
