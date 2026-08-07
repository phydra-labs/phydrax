#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax._frozendict import frozendict
from phydrax.domain import Interval1d, TimeInterval
from phydrax.operators.differential import div, div_tensor


def test_div_vector_field_point():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def u(x):
        return jnp.array([x[0], x[1]])

    d = div(u)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(d(pts).data)
    assert jnp.allclose(out, jnp.array(2.0))


def test_div_spacetime_var_x_ignores_t(sample_batch):
    dom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    ) @ TimeInterval(0.0, 1.0)

    @dom.Function("x")
    def u(x):
        return jnp.array([x[0], x[1]])

    d = div(u, var="x")
    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(4, 3), key=0)
    out = jnp.asarray(d(batch).data)
    assert out.shape == (4, 3)
    assert jnp.allclose(out, 2.0)


def test_div_coord_separable_constant(sample_grid):
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    component = geom.component()
    batch = sample_grid(component, {"x": (7, 6)}, dense_blocks=(), key=1)

    @geom.Function("x")
    def u(x):
        x, y = x
        return jnp.stack([2.0 * x, 3.0 * y], axis=-1)

    d = div(u)
    out = jnp.asarray(d(batch).data)
    assert jnp.allclose(out, 5.0, atol=1e-6)


def test_div_preserves_metadata():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    u = geom.Function("x")(lambda x: jnp.array([x[0], x[1]])).with_metadata(**{"tag": 1})
    out = div(u)
    assert out.metadata == u.metadata


def test_nested_divergence_drops_stale_optimized_derivative_hooks():
    geom = Interval1d(-2.0, 2.0)
    density = geom.Function("x")(lambda x: x[0] ** 2)
    drift = geom.Function("x")(lambda x: jnp.asarray([2.0 * x[0]]))
    covariance = geom.Function("x")(lambda x: jnp.asarray([[3.0 * x[0] ** 2]]))
    adjoint = -div(drift * density, var="x") + 0.5 * div(
        div_tensor(covariance * density, var="x"),
        var="x",
    )
    point = frozendict({"x": cx.Field(jnp.asarray([0.4]), dims=(None,))})

    assert jnp.allclose(adjoint(point).data, 12.0 * 0.4**2)


def test_div_ad_engine_jvp_matches_default():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def u(x):
        return jnp.array([x[0] ** 2 + x[1], x[1] ** 2 + x[0]])

    pts = frozendict({"x": cx.Field(jnp.array([0.3, -0.7]), dims=(None,))})
    out_ref = jnp.asarray(div(u, backend="ad")(pts).data)
    out_jvp = jnp.asarray(div(u, backend="ad", ad_engine="jvp")(pts).data)
    assert jnp.allclose(out_jvp, out_ref, atol=1e-6)


def test_div_ad_engine_requires_ad_backend():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def u(x):
        return jnp.array([x[0], x[1]])

    with pytest.raises(ValueError, match="backend='ad'"):
        div(u, backend="fd", ad_engine="jvp")
