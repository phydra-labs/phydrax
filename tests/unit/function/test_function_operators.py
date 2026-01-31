#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, Interval1d, ProductStructure, TimeInterval


@pytest.fixture
def interval():
    return Interval1d(0.0, 1.0)


@pytest.fixture
def sample_batch(interval):
    component = interval.component()
    structure = ProductStructure((("x",),))
    return component.sample(8, structure=structure, key=jr.key(0))


def test_add(sample_batch, interval):
    @interval.Function("x")
    def f(x):
        return x[0] + 1.0

    @interval.Function("x")
    def g(x):
        return 2.0 * x[0]

    h = f + g
    assert jnp.allclose(h(sample_batch).data, f(sample_batch).data + g(sample_batch).data)


def test_radd(sample_batch, interval):
    @interval.Function("x")
    def f(x):
        return x[0] + 1.0

    h = 3.0 + f
    assert jnp.allclose(h(sample_batch).data, 3.0 + f(sample_batch).data)


def test_sub(sample_batch, interval):
    @interval.Function("x")
    def f(x):
        return x[0] + 1.0

    @interval.Function("x")
    def g(x):
        return 2.0 * x[0]

    h = f - g
    assert jnp.allclose(h(sample_batch).data, f(sample_batch).data - g(sample_batch).data)


def test_rsub(sample_batch, interval):
    @interval.Function("x")
    def f(x):
        return x[0] + 1.0

    h = 3.0 - f
    assert jnp.allclose(h(sample_batch).data, 3.0 - f(sample_batch).data)


def test_mul(sample_batch, interval):
    @interval.Function("x")
    def f(x):
        return x[0] + 1.0

    @interval.Function("x")
    def g(x):
        return 2.0 * x[0]

    h = f * g
    assert jnp.allclose(h(sample_batch).data, f(sample_batch).data * g(sample_batch).data)


def test_rmul(sample_batch, interval):
    @interval.Function("x")
    def f(x):
        return x[0] + 1.0

    h = 3.0 * f
    assert jnp.allclose(h(sample_batch).data, 3.0 * f(sample_batch).data)


def test_truediv(sample_batch, interval):
    @interval.Function("x")
    def f(x):
        return x[0] + 1.0

    @interval.Function("x")
    def g(x):
        return 2.0 * x[0] + 1.0

    h = f / g
    assert jnp.allclose(h(sample_batch).data, f(sample_batch).data / g(sample_batch).data)


def test_rtruediv(sample_batch, interval):
    @interval.Function("x")
    def f(x):
        return x[0] + 1.0

    h = 3.0 / f
    assert jnp.allclose(h(sample_batch).data, 3.0 / f(sample_batch).data)


def test_pow(sample_batch, interval):
    @interval.Function("x")
    def f(x):
        return x[0] + 1.0

    h = f**2.0
    assert jnp.allclose(h(sample_batch).data, f(sample_batch).data ** 2.0)


def test_rpow(sample_batch, interval):
    @interval.Function("x")
    def f(x):
        return x[0]

    h = 3.0**f
    assert jnp.allclose(h(sample_batch).data, 3.0 ** f(sample_batch).data)


def test_abs(sample_batch, interval):
    @interval.Function("x")
    def f(x):
        return -x[0]

    h = abs(f)
    assert jnp.allclose(h(sample_batch).data, jnp.abs(f(sample_batch).data))


def test_neg(sample_batch, interval):
    @interval.Function("x")
    def f(x):
        return x[0]

    h = -f
    assert jnp.allclose(h(sample_batch).data, -f(sample_batch).data)


def test_transpose(sample_batch, interval):
    @interval.Function("x")
    def f(x):
        del x
        return jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    out = f(sample_batch)
    out_t = f.T(sample_batch)
    assert out.data.shape == (8, 2, 3)
    assert out_t.data.shape == (8, 3, 2)
    assert jnp.allclose(out_t.data[0], out.data[0].T)


def test_domain_join_and_broadcast():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    dom = geom @ time

    @geom.Function("x")
    def fx(x):
        return 2.0 * x[0]

    @time.Function("t")
    def gt(t):
        return t + 1.0

    h = fx + gt
    assert isinstance(h, DomainFunction)
    assert h.domain.labels == dom.labels

    component = dom.component()
    structure = ProductStructure((("x",), ("t",)))
    batch = component.sample((4, 5), structure=structure, key=jr.key(0))
    out = h(batch)

    axis_x = batch.structure.axis_for("x")
    axis_t = batch.structure.axis_for("t")
    assert axis_x is not None and axis_t is not None
    assert axis_x in out.named_dims
    assert axis_t in out.named_dims


def test_metadata_merge_rules(interval):
    f = DomainFunction(domain=interval, deps=("x",), func=lambda x: x[0]).with_metadata(
        m=1
    )
    g = DomainFunction(
        domain=interval, deps=("x",), func=lambda x: 2.0 * x[0]
    ).with_metadata(m=1)
    h = f + g
    assert h.metadata == f.metadata

    k = f + g.with_metadata(m=2)
    assert k.metadata == frozendict({})
