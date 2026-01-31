#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import Boundary, Square
from phydrax.operators.differential import (
    laplace_beltrami,
    surface_div,
    surface_grad,
    tangential_component,
)


def test_surface_grad_scalar_flat_edge_projection():
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component({"x": Boundary()})

    x = jnp.linspace(-1.0, 1.0, 11)
    x_inner = x[1:-1]
    pts = jnp.stack([x_inner, -jnp.ones_like(x_inner)], axis=-1)  # bottom edge y=-1

    @geom.Function("x")
    def u(p):
        return p[0] ** 2 + p[1] ** 3

    sg = surface_grad(u, component)
    val = jnp.asarray(sg(frozendict({"x": cx.Field(pts, dims=("n", None))})).data)

    expected = jnp.stack([2.0 * x_inner, jnp.zeros_like(x_inner)], axis=-1)
    assert jnp.allclose(val, expected, atol=1e-6)


def test_surface_div_vector_flat_edge():
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component({"x": Boundary()})

    x = jnp.linspace(-1.0, 1.0, 7)
    pts = jnp.stack([x, -jnp.ones_like(x)], axis=-1)

    @geom.Function("x")
    def v(p):
        return p

    sd = surface_div(v, component)
    val = jnp.asarray(sd(frozendict({"x": cx.Field(pts, dims=("n", None))})).data)
    assert jnp.allclose(val, jnp.ones_like(x), atol=1e-6)


def test_laplace_beltrami_scalar_flat_edge():
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component({"x": Boundary()})

    x = jnp.linspace(-1.0, 1.0, 7)
    pts = jnp.stack([x, -jnp.ones_like(x)], axis=-1)

    @geom.Function("x")
    def u(p):
        return p[0] ** 2 + p[1] ** 2

    lb = laplace_beltrami(u, component)
    val = jnp.asarray(lb(frozendict({"x": cx.Field(pts, dims=("n", None))})).data)
    assert jnp.allclose(val, 2.0 * jnp.ones_like(x), atol=1e-5)

    lb2 = laplace_beltrami(u, component, curvature_aware=True)
    val2 = jnp.asarray(lb2(frozendict({"x": cx.Field(pts, dims=("n", None))})).data)
    assert jnp.allclose(val2, 2.0 * jnp.ones_like(x), atol=1e-5)


def test_tangential_component_projection():
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component({"x": Boundary()})

    pts = jnp.array([[0.0, -1.0]])

    @geom.Function("x")
    def w(_):
        return jnp.array([1.0, 2.0])

    wt = tangential_component(w, component)
    val = jnp.asarray(wt(frozendict({"x": cx.Field(pts, dims=("n", None))})).data)
    assert jnp.allclose(val, jnp.array([[1.0, 0.0]]), atol=1e-6)
