#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax._frozendict import frozendict
from phydrax.domain import (
    DomainFunction,
    Interval1d,
    ProductStructure,
    Square,
    TimeInterval,
)
from phydrax.operators.differential import dt_n, partial_t


def test_partial_t_time_only_scalar():
    dom = TimeInterval(0.0, 1.0)

    @dom.Function("t")
    def f(t):
        return t**2

    t = jnp.linspace(0.0, 1.0, 7)
    out = jnp.asarray(partial_t(f)(frozendict({"t": cx.Field(t, dims=("t",))})).data)
    assert out.shape == (t.shape[0],)
    assert jnp.allclose(out, 2.0 * t)


def test_partial_t_time_only_vector():
    dom = TimeInterval(0.0, 1.0)

    @dom.Function("t")
    def f(t):
        return jnp.stack([t**2, t**3], axis=-1)

    t = jnp.linspace(0.0, 1.0, 5)
    out = jnp.asarray(partial_t(f)(frozendict({"t": cx.Field(t, dims=("t",))})).data)
    expected = jnp.stack([2.0 * t, 3.0 * t**2], axis=-1)
    assert out.shape == expected.shape
    assert jnp.allclose(out, expected)


def test_partial_t_spacetime_broadcasts_over_space(sample_batch):
    dom = Square(center=(0.0, 0.0), side=2.0) @ TimeInterval(0.0, 1.0)

    @dom.Function("t")
    def u(t):
        return jnp.sin(t)

    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(4, 6), key=0)
    out = jnp.asarray(partial_t(u)(batch).data)

    assert out.shape == (4, 6)
    t = jnp.asarray(batch.points["t"].data)
    assert jnp.allclose(out, jnp.cos(t)[None, :])


def test_partial_t_preserves_metadata():
    dom = TimeInterval(0.0, 1.0)
    u = DomainFunction(
        domain=dom, deps=("t",), func=lambda t: t**2, metadata={"scale": 3}
    )
    assert partial_t(u).metadata == u.metadata


def test_partial_t_product_skips_space_only_factor():
    dom = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)

    @dom.Function("x")
    def fx(x):
        return x[0] + 2.0

    @dom.Function("t")
    def gt(t):
        return t**3

    u = fx * gt
    component = dom.component()
    batch = component.sample(
        num_points=(5, 7),
        structure=ProductStructure((("x",), ("t",))),
        key=jr.key(0),
    )
    x_vals = jnp.asarray(batch.points["x"].data[:, 0])
    t_vals = jnp.asarray(batch.points["t"].data)
    expected = (x_vals[:, None] + 2.0) * (3.0 * t_vals**2)[None, :]
    out = jnp.asarray(partial_t(u, var="t")(batch).data)
    assert jnp.allclose(out, expected, atol=1e-6)


def test_dt_n_quotient_with_time_independent_denominator():
    dom = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)

    @dom.Function("x")
    def fx(x):
        return x[0] + 2.0

    @dom.Function("t")
    def gt(t):
        return t**3

    u = gt / fx
    component = dom.component()
    batch = component.sample(
        num_points=(6, 8),
        structure=ProductStructure((("x",), ("t",))),
        key=jr.key(1),
    )
    x_vals = jnp.asarray(batch.points["x"].data[:, 0])
    t_vals = jnp.asarray(batch.points["t"].data)
    expected = (6.0 * t_vals)[None, :] / (x_vals[:, None] + 2.0)
    out = jnp.asarray(dt_n(u, var="t", order=2)(batch).data)
    assert jnp.allclose(out, expected, atol=1e-6)


def test_partial_t_quotient_with_time_independent_numerator():
    dom = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)

    @dom.Function("x")
    def fx(x):
        return x[0] + 2.0

    @dom.Function("t")
    def gt(t):
        return 1.0 + t

    u = fx / gt
    component = dom.component()
    batch = component.sample(
        num_points=(4, 9),
        structure=ProductStructure((("x",), ("t",))),
        key=jr.key(2),
    )
    x_vals = jnp.asarray(batch.points["x"].data[:, 0])
    t_vals = jnp.asarray(batch.points["t"].data)
    expected = -(x_vals[:, None] + 2.0) / ((1.0 + t_vals)[None, :] ** 2)
    out = jnp.asarray(partial_t(u, var="t")(batch).data)
    assert jnp.allclose(out, expected, atol=1e-6)


def test_partial_t_ad_engine_jvp_matches_default():
    dom = TimeInterval(0.0, 1.0)

    @dom.Function("t")
    def f(t):
        return jnp.sin(t) + t**3

    t = jnp.linspace(0.0, 1.0, 11)
    batch = frozendict({"t": cx.Field(t, dims=("t",))})
    out_ref = jnp.asarray(partial_t(f)(batch).data)
    out_jvp = jnp.asarray(partial_t(f, ad_engine="jvp")(batch).data)
    assert jnp.allclose(out_jvp, out_ref, atol=1e-6)


def test_dt_n_ad_engine_jvp_matches_default():
    dom = TimeInterval(0.0, 1.0)

    @dom.Function("t")
    def f(t):
        return t**4 + 2.0 * t

    t = jnp.linspace(0.0, 1.0, 9)
    batch = frozendict({"t": cx.Field(t, dims=("t",))})
    out_ref = jnp.asarray(dt_n(f, order=2, backend="ad")(batch).data)
    out_jvp = jnp.asarray(dt_n(f, order=2, backend="ad", ad_engine="jvp")(batch).data)
    assert jnp.allclose(out_jvp, out_ref, atol=1e-6)


def test_dt_n_ad_engine_requires_ad_backend():
    dom = TimeInterval(0.0, 1.0)

    @dom.Function("t")
    def f(t):
        return t**3

    with pytest.raises(ValueError, match="backend='ad'"):
        dt_n(f, order=2, backend="jet", ad_engine="jvp")
