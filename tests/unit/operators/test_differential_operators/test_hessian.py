#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, Square, TimeInterval
from phydrax.operators.differential import hessian


def test_hessian_scalar_function_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    H = hessian(f)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(H(pts).data)
    assert out.shape == (2, 2)
    assert jnp.allclose(out, 2.0 * jnp.eye(2))


def test_hessian_vector_function_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def f(x):
        return jnp.array([x[0] ** 2, x[0] * x[1]])

    H = hessian(f)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(H(pts).data)
    assert out.shape == (2, 2, 2)
    expected = jnp.array([[[2.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]])
    assert jnp.allclose(out, expected)


def test_hessian_spacetime_var_x_ignores_t(sample_batch):
    dom = Square(center=(0.0, 0.0), side=2.0) @ TimeInterval(0.0, 1.0)

    @dom.Function("x", "t")
    def f(x, t):
        return x[0] ** 2 + x[1] ** 2 + t

    H = hessian(f, var="x")
    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(3, 4), key=0)
    out = jnp.asarray(H(batch).data)
    assert out.shape == (3, 4, 2, 2)
    assert jnp.allclose(out, 2.0 * jnp.eye(2)[None, None, :, :])


def test_hessian_coord_separable_constant(sample_coord_separable):
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component()
    batch = sample_coord_separable(component, {"x": (6, 5)}, dense_blocks=(), key=0)

    @geom.Function("x")
    def f(x):
        x, y = x
        return x**2 + y**2

    H = hessian(f)
    out = jnp.asarray(H(batch).data)
    assert out.shape == (6, 5, 2, 2)
    assert jnp.allclose(out, 2.0 * jnp.eye(2)[None, None, :, :], atol=1e-6)


def test_hessian_complex_scalar_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def f(x):
        return x[0] ** 2 + 1j * x[1] ** 2

    H = hessian(f)
    pts = frozendict({"x": cx.Field(jnp.array([0.3, -1.1]), dims=(None,))})
    out = jnp.asarray(H(pts).data)
    expected = jnp.array([[2.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 2.0j]])
    assert jnp.allclose(out, expected)


def test_hessian_time_only_scalar():
    dom = TimeInterval(0.0, 1.0)

    @dom.Function("t")
    def f(t):
        return t**3

    t = jnp.linspace(0.0, 1.0, 7)
    out = jnp.asarray(hessian(f)(frozendict({"t": cx.Field(t, dims=("t",))})).data)
    assert out.shape == (t.shape[0],)
    assert jnp.allclose(out, 6.0 * t)


def test_hessian_preserves_metadata():
    geom = Square(center=(0.0, 0.0), side=2.0)
    u = DomainFunction(
        domain=geom, deps=("x",), func=lambda x: x[0] ** 2, metadata={"tag": 1}
    )
    out = hessian(u)
    assert out.metadata == u.metadata
