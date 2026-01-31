#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.domain import DomainFunction, Interval1d, Square, TimeInterval
from phydrax.operators.functional import (
    spatial_inner_product,
    spatial_l2_norm,
    spatial_mean,
)


def test_spatial_mean_constant_matches_constant(sample_batch):
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    batch = sample_batch(component, blocks=(("x",),), num_points=2048, key=0)

    u = DomainFunction(domain=geom, deps=(), func=2.0)
    val = jnp.asarray(spatial_mean(u, batch, component=component).data)
    assert jnp.allclose(val, 2.0, atol=1e-12, rtol=0.0)


def test_spatial_mean_time_broadcasts_over_space(sample_batch):
    dom = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(3, 4), key=1)

    @dom.Function("t")
    def u(t):
        return t

    out = jnp.asarray(spatial_mean(u, batch, component=component, over="x").data)
    t = jnp.asarray(batch.points["t"].data)
    assert out.shape == t.shape
    assert jnp.allclose(out, t, atol=1e-12, rtol=0.0)


def test_spatial_inner_product_constant(sample_batch):
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    batch = sample_batch(component, blocks=(("x",),), num_points=1024, key=2)

    u = DomainFunction(domain=geom, deps=(), func=2.0)
    v = DomainFunction(domain=geom, deps=(), func=3.0)
    val = jnp.asarray(spatial_inner_product(u, v, batch, component=component).data)
    assert jnp.allclose(val, 6.0, atol=1e-12, rtol=0.0)


def test_spatial_l2_norm_constant_matches_closed_form(sample_batch):
    geom = Square(center=(0.0, 0.0), side=2.0)  # area=4
    component = geom.component()
    batch = sample_batch(component, blocks=(("x",),), num_points=2048, key=3)

    u = DomainFunction(domain=geom, deps=(), func=2.0)
    val = jnp.asarray(spatial_l2_norm(u, batch, component=component).data)
    assert jnp.allclose(val, 4.0, atol=1e-12, rtol=0.0)
