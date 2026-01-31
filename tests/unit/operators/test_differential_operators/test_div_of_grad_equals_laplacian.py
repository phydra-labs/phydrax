#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import Square, TimeInterval
from phydrax.operators.differential import div, grad, laplacian


def test_div_grad_equals_laplacian_scalar_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out_divgrad = jnp.asarray(div(grad(f))(pts).data)
    out_lap = jnp.asarray(laplacian(f)(pts).data)
    assert jnp.allclose(out_divgrad, out_lap)


def test_div_grad_equals_laplacian_vector_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def f(x):
        return jnp.array([x[0] ** 2, x[1] ** 2])

    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out_divgrad = jnp.asarray(div(grad(f))(pts).data)
    out_lap = jnp.asarray(laplacian(f)(pts).data)
    assert jnp.allclose(out_divgrad, out_lap)


def test_div_grad_equals_laplacian_spacetime(sample_batch):
    geom = Square(center=(0.0, 0.0), side=2.0)
    dom = geom @ TimeInterval(0.0, 1.0)

    @dom.Function("x", "t")
    def f(x, t):
        return x[0] ** 2 + x[1] ** 2 + t

    divgrad = div(grad(f, var="x"), var="x")
    lap = laplacian(f, var="x")

    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(3, 4), key=0)
    out_divgrad = jnp.asarray(divgrad(batch).data)
    out_lap = jnp.asarray(lap(batch).data)
    assert jnp.allclose(out_divgrad, out_lap)
