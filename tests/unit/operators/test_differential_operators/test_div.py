#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, Square, TimeInterval
from phydrax.operators.differential import div


def test_div_vector_field_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([x[0], x[1]])

    d = div(u)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(d(pts).data)
    assert jnp.allclose(out, jnp.array(2.0))


def test_div_spacetime_var_x_ignores_t(sample_batch):
    dom = Square(center=(0.0, 0.0), side=2.0) @ TimeInterval(0.0, 1.0)

    @dom.Function("x")
    def u(x):
        return jnp.array([x[0], x[1]])

    d = div(u, var="x")
    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(4, 3), key=0)
    out = jnp.asarray(d(batch).data)
    assert out.shape == (4, 3)
    assert jnp.allclose(out, 2.0)


def test_div_coord_separable_constant(sample_coord_separable):
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component()
    batch = sample_coord_separable(component, {"x": (7, 6)}, dense_blocks=(), key=1)

    @geom.Function("x")
    def u(x):
        x, y = x
        return jnp.stack([2.0 * x, 3.0 * y], axis=-1)

    d = div(u)
    out = jnp.asarray(d(batch).data)
    assert jnp.allclose(out, 5.0, atol=1e-6)


def test_div_preserves_metadata():
    geom = Square(center=(0.0, 0.0), side=2.0)
    u = DomainFunction(
        domain=geom,
        deps=("x",),
        func=lambda x: jnp.array([x[0], x[1]]),
        metadata={"tag": 1},
    )
    out = div(u)
    assert out.metadata == u.metadata
