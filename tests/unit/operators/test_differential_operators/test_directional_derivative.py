#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, Square, TimeInterval
from phydrax.operators.differential import directional_derivative


def test_directional_derivative_scalar_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    v = DomainFunction(domain=geom, deps=(), func=jnp.array([1.0, 0.0]))
    dd = directional_derivative(f, v)

    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(dd(pts).data)
    assert jnp.allclose(out, 4.0)


def test_directional_derivative_vector_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def f(x):
        return jnp.array([x[0] ** 2, x[1] ** 2])

    v = DomainFunction(domain=geom, deps=(), func=jnp.array([1.0, 1.0]))
    dd = directional_derivative(f, v)

    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(dd(pts).data)
    assert jnp.allclose(out, jnp.array([4.0, 6.0]))


def test_directional_derivative_direction_is_function():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    v = DomainFunction(domain=geom, deps=("x",), func=lambda x: x)
    dd = directional_derivative(f, v)

    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(dd(pts).data)
    assert jnp.allclose(out, 26.0)


def test_directional_derivative_spacetime_broadcasts_over_t(sample_batch):
    geom = Square(center=(0.0, 0.0), side=2.0)
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


def test_directional_derivative_coord_separable(sample_coord_separable):
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component()
    batch = sample_coord_separable(component, {"x": (5, 4)}, dense_blocks=(), key=0)

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
    geom = Square(center=(0.0, 0.0), side=2.0)
    u = DomainFunction(
        domain=geom, deps=("x",), func=lambda x: x[0] ** 2, metadata={"scale": 7}
    )
    v = DomainFunction(domain=geom, deps=(), func=jnp.array([1.0, 0.0]))
    assert directional_derivative(u, v).metadata == u.metadata
