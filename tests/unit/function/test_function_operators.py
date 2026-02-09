#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import operator

import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax._frozendict import frozendict
from phydrax.domain import (
    DomainFunction,
    FourierAxisSpec,
    Interval1d,
    ProductStructure,
    TimeInterval,
)
from phydrax.domain._function import _rank1_leading_broadcast_op
from phydrax.operators.differential import partial_n
from phydrax.operators.differential._hooks import blend_with_gate


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


def test_constant_with_dependencies_participates_in_arithmetic(sample_batch, interval):
    const = interval.Function("x")(2.0)

    @interval.Function("x")
    def f(x):
        return x[0] + 1.0

    out = (f - const)(sample_batch)
    expected = jnp.asarray(f(sample_batch).data) - 2.0
    assert jnp.allclose(jnp.asarray(out.data), expected)


def test_constant_with_dependencies_works_on_coord_separable_batch():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    dom = geom @ time

    @dom.Function("x", "t")
    def u(x, t):
        return x[0] + t

    const = dom.Function("x", "t")(1.0)
    h = u - const

    component = dom.component()
    batch = component.sample_coord_separable(
        {"x": FourierAxisSpec(8)},
        num_points=5,
        dense_structure=ProductStructure((("t",),)),
        key=jr.key(0),
    )
    out = h(batch)
    expected = jnp.asarray(u(batch).data) - 1.0
    assert jnp.allclose(jnp.asarray(out.data), expected)


def test_rank1_leading_broadcast_op_mul():
    w = jnp.array([1.0, 2.0, 3.0], dtype=float)
    u = jnp.arange(12.0, dtype=float).reshape((3, 4))
    out = _rank1_leading_broadcast_op(operator.mul, w, u)
    expected = w[:, None] * u
    assert jnp.allclose(out, expected)


def test_rank1_leading_broadcast_op_div():
    w = jnp.array([1.0, 2.0, 4.0], dtype=float)
    u = jnp.arange(12.0, dtype=float).reshape((3, 4)) + 1.0
    out = _rank1_leading_broadcast_op(operator.truediv, u, w)
    expected = u / w[:, None]
    assert jnp.allclose(out, expected)


def test_rank1_leading_broadcast_op_add():
    w = jnp.array([1.0, 2.0, 3.0], dtype=float)
    u = jnp.arange(12.0, dtype=float).reshape((3, 4))
    out = _rank1_leading_broadcast_op(operator.add, u, w)
    expected = u + w[:, None]
    assert jnp.allclose(out, expected)


def test_rank1_leading_broadcast_op_sub():
    w = jnp.array([1.0, 2.0, 3.0], dtype=float)
    u = jnp.arange(12.0, dtype=float).reshape((3, 4))
    out = _rank1_leading_broadcast_op(operator.sub, u, w)
    expected = u - w[:, None]
    assert jnp.allclose(out, expected)


def test_rank1_leading_broadcast_op_mul_outer_for_mismatched_rank1():
    x = jnp.array([1.0, 2.0, 3.0], dtype=float)
    t = jnp.array([4.0, 5.0], dtype=float)
    out = _rank1_leading_broadcast_op(operator.mul, x, t)
    expected = x[:, None] * t[None, :]
    assert jnp.allclose(out, expected)


def test_blend_with_gate_matches_manual_expression():
    dom = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)

    @dom.Function("x", "t")
    def base(x, t):
        return x[0] + 2.0 * t

    @dom.Function("x", "t")
    def overlay(x, t):
        return (x[0] ** 2) + (t**2)

    @dom.Function("x")
    def gate(x):
        return x[0]

    blended = blend_with_gate(base, overlay, gate)
    manual = base + gate * (overlay - base)

    batch = dom.component().sample_coord_separable(
        {"x": FourierAxisSpec(9)},
        num_points=7,
        dense_structure=ProductStructure((("t",),)),
        key=jr.key(11),
    )

    val_blended = jnp.asarray(blended(batch).data)
    val_manual = jnp.asarray(manual(batch).data)
    assert jnp.allclose(val_blended, val_manual, atol=1e-6)

    dt_blended = partial_n(blended, var="t", order=1, backend="ad")
    dt_manual = partial_n(manual, var="t", order=1, backend="ad")
    assert jnp.allclose(
        jnp.asarray(dt_blended(batch).data),
        jnp.asarray(dt_manual(batch).data),
        atol=1e-6,
    )

    dx_blended = partial_n(blended, var="x", axis=0, order=1, backend="ad")
    dx_manual = partial_n(manual, var="x", axis=0, order=1, backend="ad")
    assert jnp.allclose(
        jnp.asarray(dx_blended(batch).data),
        jnp.asarray(dx_manual(batch).data),
        atol=1e-6,
    )


def test_blend_with_gate_evaluates_inputs_once_per_call():
    dom = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    calls = {"base": 0, "overlay": 0, "gate": 0}

    @dom.Function("x", "t")
    def base(x, t):
        calls["base"] += 1
        return x[0] + t

    @dom.Function("x", "t")
    def overlay(x, t):
        calls["overlay"] += 1
        return (x[0] ** 2) + t

    @dom.Function("x")
    def gate(x):
        calls["gate"] += 1
        return x[0]

    blended = blend_with_gate(base, overlay, gate)
    out = blended.func(jnp.asarray([0.2]), jnp.asarray(0.3), key=jr.key(0))
    expected = (0.2 + 0.3) + 0.2 * (((0.2**2) + 0.3) - (0.2 + 0.3))
    assert jnp.allclose(jnp.asarray(out), jnp.asarray(expected))
    assert calls["base"] == 1
    assert calls["overlay"] == 1
    assert calls["gate"] == 1
