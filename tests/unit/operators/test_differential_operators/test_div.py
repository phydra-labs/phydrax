#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import pytest

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


def test_div_ad_engine_jvp_matches_default():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([x[0] ** 2 + x[1], x[1] ** 2 + x[0]])

    pts = frozendict({"x": cx.Field(jnp.array([0.3, -0.7]), dims=(None,))})
    out_ref = jnp.asarray(div(u, backend="ad")(pts).data)
    out_jvp = jnp.asarray(div(u, backend="ad", ad_engine="jvp")(pts).data)
    assert jnp.allclose(out_jvp, out_ref, atol=1e-6)


def test_div_ad_engine_requires_ad_backend():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([x[0], x[1]])

    with pytest.raises(ValueError, match="backend='ad'"):
        div(u, backend="fd", ad_engine="jvp")
