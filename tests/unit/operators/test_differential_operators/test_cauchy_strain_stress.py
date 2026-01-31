#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, Square, TimeInterval
from phydrax.operators.differential import cauchy_strain, cauchy_stress


def test_cauchy_strain_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([x[0] ** 2, x[1] ** 2])

    strain_u = cauchy_strain(u)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(strain_u(pts).data)
    assert out.shape == (2, 2)
    assert jnp.allclose(out, jnp.array([[4.0, 0.0], [0.0, 6.0]]))


def test_cauchy_strain_spacetime_depends_on_t(sample_batch):
    geom = Square(center=(0.0, 0.0), side=2.0)
    dom = geom @ TimeInterval(0.0, 1.0)

    @dom.Function("x", "t")
    def u(x, t):
        return jnp.array([x[0] * t, x[1] * t])

    strain_u = cauchy_strain(u, var="x")
    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(3, 4), key=0)
    out = jnp.asarray(strain_u(batch).data)
    assert out.shape == (3, 4, 2, 2)
    t = jnp.asarray(batch.points["t"].data)
    expected = jnp.broadcast_to(t[None, :, None, None] * jnp.eye(2), out.shape)
    assert jnp.allclose(out, expected)


def test_cauchy_strain_coord_separable(sample_coord_separable):
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component()
    batch = sample_coord_separable(component, {"x": (4, 5)}, dense_blocks=(), key=0)

    @geom.Function("x")
    def u(x):
        x, y = x
        return jnp.stack([2.0 * x, 3.0 * y], axis=-1)

    out = jnp.asarray(cauchy_strain(u)(batch).data)
    expected = jnp.broadcast_to(jnp.array([[2.0, 0.0], [0.0, 3.0]]), out.shape)
    assert jnp.allclose(out, expected, atol=1e-6)


def test_cauchy_stress_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([x[0], x[1]])

    sigma = cauchy_stress(u, lambda_=1.0, mu=2.0)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(sigma(pts).data)
    assert jnp.allclose(out, jnp.array([[6.0, 0.0], [0.0, 6.0]]))


def test_cauchy_stress_time_dependent(sample_batch):
    geom = Square(center=(0.0, 0.0), side=2.0)
    dom = geom @ TimeInterval(0.0, 1.0)

    @dom.Function("x", "t")
    def u(x, t):
        return jnp.array([x[0] * t, x[1] * t])

    sigma = cauchy_stress(u, lambda_=1.0, mu=2.0, var="x")
    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(2, 3), key=1)
    out = jnp.asarray(sigma(batch).data)
    assert out.shape == (2, 3, 2, 2)
    t = jnp.asarray(batch.points["t"].data)
    expected = jnp.broadcast_to((6.0 * t)[None, :, None, None] * jnp.eye(2), out.shape)
    assert jnp.allclose(out, expected)


def test_cauchy_stress_complex_valued_u():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([x[0], 1j * x[1]])

    sigma = cauchy_stress(u, lambda_=1.0, mu=2.0)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(sigma(pts).data)
    expected = jnp.array([[5.0 + 1j, 0.0], [0.0, 1.0 + 5.0j]])
    assert jnp.allclose(out, expected)


def test_cauchy_strain_preserves_metadata():
    geom = Square(center=(0.0, 0.0), side=2.0)
    u = DomainFunction(
        domain=geom,
        deps=("x",),
        func=lambda x: jnp.array([x[0], x[1]]),
        metadata={"tag": 1},
    )
    assert cauchy_strain(u).metadata == u.metadata
