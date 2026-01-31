#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, Square, TimeInterval
from phydrax.operators.differential import div_cauchy_stress


def test_div_cauchy_stress_zero_for_linear_u():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([x[0], x[1]])

    op = div_cauchy_stress(u, lambda_=1.0, mu=2.0)
    pts = frozendict({"x": cx.Field(jnp.array([1.0, -2.0]), dims=(None,))})
    out = jnp.asarray(op(pts).data)
    assert jnp.allclose(out, jnp.array([0.0, 0.0]))


def test_div_cauchy_stress_quadratic_u_constant():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([0.5 * x[0] ** 2, 0.0])

    op = div_cauchy_stress(u, lambda_=1.0, mu=2.0)
    pts = frozendict({"x": cx.Field(jnp.array([1.2, -0.7]), dims=(None,))})
    out = jnp.asarray(op(pts).data)
    assert jnp.allclose(out, jnp.array([5.0, 0.0]))


def test_div_cauchy_stress_coord_separable(sample_coord_separable):
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component()
    batch = sample_coord_separable(component, {"x": (6, 5)}, dense_blocks=(), key=0)

    @geom.Function("x")
    def u(x):
        x, y = x
        return jnp.stack([x, y], axis=-1)

    op = div_cauchy_stress(u, lambda_=1.0, mu=2.0)
    out = jnp.asarray(op(batch).data)
    assert jnp.allclose(out, jnp.zeros_like(jnp.asarray(out)), atol=1e-6)


def test_div_cauchy_stress_spacetime_broadcasts(sample_batch):
    geom = Square(center=(0.0, 0.0), side=2.0)
    dom = geom @ TimeInterval(0.0, 1.0)

    @dom.Function("x", "t")
    def u(x, t):
        return jnp.array([x[0] * t, x[1] * t])

    op = div_cauchy_stress(u, lambda_=1.0, mu=2.0, var="x")
    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(3, 4), key=1)
    out = jnp.asarray(op(batch).data)
    assert out.shape == (3, 4, 2)
    assert jnp.allclose(out, jnp.zeros_like(jnp.asarray(out)))


def test_div_cauchy_stress_preserves_metadata():
    geom = Square(center=(0.0, 0.0), side=2.0)
    u = DomainFunction(
        domain=geom,
        deps=("x",),
        func=lambda x: jnp.array([x[0], x[1]]),
        metadata={"tag": 3},
    )
    assert div_cauchy_stress(u, lambda_=1.0, mu=2.0).metadata == u.metadata
