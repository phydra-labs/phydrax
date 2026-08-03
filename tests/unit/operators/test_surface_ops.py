#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import Boundary, Cube, Square
from phydrax.operators.differential import (
    ambient_surface_hessian_trace,
    surface_curl_scalar,
    surface_curl_vector,
    surface_div,
    surface_grad,
    tangential_component,
)


class _RadialNormalComponent:
    def __init__(self, domain):
        self.domain = domain

    def normal(self, *, var):
        @self.domain.Function(var)
        def radial(point):
            return point / jnp.linalg.norm(point)

        return radial


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


def test_ambient_surface_hessian_trace_flat_edge():
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component({"x": Boundary()})

    x = jnp.linspace(-1.0, 1.0, 7)
    pts = jnp.stack([x, -jnp.ones_like(x)], axis=-1)

    @geom.Function("x")
    def u(p):
        return p[0] ** 2 + p[1] ** 2

    trace = ambient_surface_hessian_trace(u, component)
    val = jnp.asarray(trace(frozendict({"x": cx.Field(pts, dims=("n", None))})).data)
    assert jnp.allclose(val, 2.0 * jnp.ones_like(x), atol=1e-5)


def test_surface_curl_scalar_on_flat_face():
    geom = Cube(center=(0.0, 0.0, 0.0), side=2.0)
    component = geom.component({"x": Boundary()})
    points = jnp.array([[0.2, 0.3, 1.0], [-0.4, 0.1, 1.0]])

    @geom.Function("x")
    def scalar(point):
        return point[0] ** 2 + point[1]

    result = surface_curl_scalar(scalar, component)
    values = jnp.asarray(
        result(frozendict({"x": cx.Field(points, dims=("n", None))})).data
    )
    expected = jnp.stack(
        (
            -jnp.ones((points.shape[0],)),
            2.0 * points[:, 0],
            jnp.zeros((points.shape[0],)),
        ),
        axis=-1,
    )

    assert jnp.allclose(values, expected, atol=1e-6)


def test_surface_curl_vector_on_flat_face():
    geom = Cube(center=(0.0, 0.0, 0.0), side=2.0)
    component = geom.component({"x": Boundary()})
    points = jnp.array([[0.2, 0.3, 1.0], [-0.4, 0.1, 1.0]])

    @geom.Function("x")
    def vector(point):
        return jnp.array([-0.5 * point[1], 0.5 * point[0], 0.0])

    result = surface_curl_vector(vector, component)
    values = jnp.asarray(
        result(frozendict({"x": cx.Field(points, dims=("n", None))})).data
    )

    assert jnp.allclose(values, jnp.ones((points.shape[0],)), atol=1e-6)


def test_surface_div_grad_uses_differentiable_normal_provider():
    geom = Cube(center=(0.0, 0.0, 0.0), side=2.0)
    component = _RadialNormalComponent(geom)
    points = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.6, 0.8, 0.0],
            [-0.5, 0.5, jnp.sqrt(0.5)],
        ]
    )

    @geom.Function("x")
    def scalar(point):
        return point[0]

    result = surface_div(surface_grad(scalar, component), component)
    values = jnp.asarray(
        result(frozendict({"x": cx.Field(points, dims=("n", None))})).data
    )

    assert jnp.allclose(values, -2.0 * points[:, 0], atol=1e-6)


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
