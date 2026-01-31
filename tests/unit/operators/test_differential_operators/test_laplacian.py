#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, Square, TimeInterval
from phydrax.operators.differential import laplacian


def test_laplacian_scalar_function_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    L = laplacian(f)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(L(pts).data)
    assert jnp.allclose(out, 4.0)


def test_laplacian_vector_function_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def f(x):
        return jnp.array([x[0] ** 2, x[1] ** 2])

    L = laplacian(f)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(L(pts).data)
    assert out.shape == (2,)
    assert jnp.allclose(out, jnp.array([2.0, 2.0]))


def test_laplacian_spacetime_var_x_ignores_t(sample_batch):
    dom = Square(center=(0.0, 0.0), side=2.0) @ TimeInterval(0.0, 1.0)

    @dom.Function("x", "t")
    def f(x, t):
        return x[0] ** 2 + x[1] ** 2 + t

    L = laplacian(f, var="x")
    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(3, 4), key=0)
    out = jnp.asarray(L(batch).data)
    assert out.shape == (3, 4)
    assert jnp.allclose(out, 4.0)


def test_laplacian_coord_separable_constant(sample_coord_separable):
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component()
    batch = sample_coord_separable(component, {"x": (6, 5)}, dense_blocks=(), key=0)

    @geom.Function("x")
    def f(x):
        x, y = x
        return x**2 + y**2

    L = laplacian(f)
    out = jnp.asarray(L(batch).data)
    assert jnp.allclose(out, 4.0, atol=1e-6)


def test_laplacian_complex_output_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def f(x):
        return x[0] ** 2 + 1j * x[1] ** 2

    L = laplacian(f)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(L(pts).data)
    assert jnp.allclose(out, 2.0 + 2.0j)


def test_laplacian_preserves_metadata():
    geom = Square(center=(0.0, 0.0), side=2.0)
    u = DomainFunction(
        domain=geom, deps=("x",), func=lambda x: x[0] ** 2, metadata={"tag": 1}
    )
    out = laplacian(u)
    assert out.metadata == u.metadata
