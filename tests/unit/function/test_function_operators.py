#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import operator
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax._frozendict import frozendict
from phydrax.discretization import FourierAxisSpec
from phydrax.domain import (
    DomainFunction,
    Interval1d,
    PointBatch,
    SampleLayout,
    TimeInterval,
)
from phydrax.domain._function import _rank1_leading_broadcast_op
from phydrax.operators.differential import partial_n
from phydrax.operators.differential._hooks import blend_with_gate


@pytest.fixture
def interval() -> Interval1d:
    return Interval1d(0.0, 1.0)


@pytest.fixture
def sample_batch(interval: Interval1d) -> PointBatch:
    component = interval.component()
    structure = SampleLayout((("x",),))
    batch = component.sample(
        phx.domain.PointSampling(8, layout=structure),
        key=jr.key(0),
    )
    assert isinstance(batch, PointBatch)
    return batch


def test_function_binding_passes_only_explicit_runtime_arguments(
    sample_batch: PointBatch,
    interval: Interval1d,
) -> None:
    @interval.Function("x")
    def implicit_key(x: jax.Array, *, key: jax.Array | None = None) -> jax.Array:
        del x
        return jnp.asarray(key is None, dtype=jnp.float64)

    @interval.Function("x", binding=phx.domain.FunctionBinding(pass_key=True))
    def keyed(x: jax.Array, *, key: jax.Array) -> jax.Array:
        del x
        return jr.uniform(key)

    @interval.Function("x", binding=phx.domain.FunctionBinding(pass_iter=True))
    def iterated(x: jax.Array, *, iter_: int) -> jax.Array:
        return iter_ * x[0]

    assert isinstance(implicit_key.func, phx.domain.PointwiseEvaluator)
    assert jnp.all(implicit_key(sample_batch, key=jr.key(1)).data == 1.0)
    key = jr.key(2)
    assert jnp.allclose(jnp.asarray(keyed(sample_batch, key=key).data), jr.uniform(key))
    expected = 3.0 * sample_batch.points["x"].data[..., 0]
    assert jnp.allclose(jnp.asarray(iterated(sample_batch, iter_=3).data), expected)


def test_function_arithmetic_matches_the_corresponding_array_operations(
    sample_batch: PointBatch,
    interval: Interval1d,
) -> None:
    @interval.Function("x")
    def first(x: jax.Array) -> jax.Array:
        return x[0] + 1.0

    @interval.Function("x")
    def second(x: jax.Array) -> jax.Array:
        return 2.0 * x[0] + 1.0

    first_values = jnp.asarray(first(sample_batch).data)
    second_values = jnp.asarray(second(sample_batch).data)
    cases = (
        ("add", first + second, first_values + second_values),
        ("radd", 3.0 + first, 3.0 + first_values),
        ("sub", first - second, first_values - second_values),
        ("rsub", 3.0 - first, 3.0 - first_values),
        ("mul", first * second, first_values * second_values),
        ("rmul", 3.0 * first, 3.0 * first_values),
        ("div", first / second, first_values / second_values),
        ("rdiv", 3.0 / first, 3.0 / first_values),
        ("pow", first**2.0, first_values**2.0),
        ("rpow", 3.0**first, 3.0**first_values),
        ("abs", abs(-first), jnp.abs(-first_values)),
        ("neg", -first, -first_values),
    )
    for case_id, function, expected in cases:
        actual = jnp.asarray(function(sample_batch).data)
        assert jnp.allclose(actual, expected), case_id


def test_transpose(sample_batch: Any, interval: Any) -> None:
    @interval.Function("x")
    def f(x: Any) -> Any:
        del x
        return jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    out = f(sample_batch)
    out_t = f.T(sample_batch)
    assert out.data.shape == (8, 2, 3)
    assert out_t.data.shape == (8, 3, 2)
    assert jnp.allclose(out_t.data[0], out.data[0].T)


def test_domain_join_and_broadcast() -> None:
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    dom = geom @ time

    @geom.Function("x")
    def fx(x: Any) -> Any:
        return 2.0 * x[0]

    @time.Function("t")
    def gt(t: Any) -> Any:
        return t + 1.0

    h = fx + gt
    assert isinstance(h, DomainFunction)
    assert h.domain.labels == dom.labels

    component = dom.component()
    structure = SampleLayout((("x",), ("t",)))
    batch = component.sample(
        phx.domain.PointSampling((4, 5), layout=structure), key=jr.key(0)
    )
    out = h(batch)

    axis_x = batch.structure.axis_for("x")
    axis_t = batch.structure.axis_for("t")
    assert axis_x is not None and axis_t is not None
    assert axis_x in out.named_dims
    assert axis_t in out.named_dims


def test_metadata_merge_rules(interval: Any) -> None:
    f = interval.Function("x")(lambda x: x[0]).with_metadata(m=1)
    g = interval.Function("x")(lambda x: 2.0 * x[0]).with_metadata(m=1)
    h = f + g
    assert h.metadata == f.metadata

    k = f + g.with_metadata(m=2)
    assert k.metadata == frozendict({})


def test_constant_with_dependencies_participates_in_arithmetic(
    sample_batch: Any, interval: Any
) -> None:
    const = interval.Function("x")(2.0)

    @interval.Function("x")
    def f(x: Any) -> Any:
        return x[0] + 1.0

    out = (f - const)(sample_batch)
    expected = jnp.asarray(f(sample_batch).data) - 2.0
    assert jnp.allclose(jnp.asarray(out.data), expected)


def test_constant_with_dependencies_works_on_coord_separable_batch() -> None:
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    dom = geom @ time

    @dom.Function("x", "t")
    def u(x: Any, t: Any) -> Any:
        return x[0] + t

    const = dom.Function("x", "t")(1.0)
    h = u - const

    component = dom.component()
    batch = component.sample(
        phx.domain.GridSampling(
            {"x": FourierAxisSpec(8)},
            dense=phx.domain.PointSampling(5, layout=SampleLayout((("t",),))),
        ),
        key=jr.key(0),
    )
    out = h(batch)
    expected = jnp.asarray(u(batch).data) - 1.0
    assert jnp.allclose(jnp.asarray(out.data), expected)


def test_rank1_broadcast_operations_preserve_leading_axis_semantics() -> None:
    weights = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    values = jnp.arange(12.0, dtype=jnp.float64).reshape((3, 4))
    division_weights = jnp.asarray([1.0, 2.0, 4.0], dtype=jnp.float64)
    positive_values = values + 1.0
    cases = (
        (
            "mul",
            _rank1_leading_broadcast_op(operator.mul, weights, values),
            weights[:, None] * values,
        ),
        (
            "div",
            _rank1_leading_broadcast_op(
                operator.truediv,
                positive_values,
                division_weights,
            ),
            positive_values / division_weights[:, None],
        ),
        (
            "add",
            _rank1_leading_broadcast_op(operator.add, values, weights),
            values + weights[:, None],
        ),
        (
            "sub",
            _rank1_leading_broadcast_op(operator.sub, values, weights),
            values - weights[:, None],
        ),
    )
    for case_id, actual, expected in cases:
        assert jnp.allclose(actual, expected), case_id

    time_values = jnp.asarray([4.0, 5.0], dtype=jnp.float64)
    outer = _rank1_leading_broadcast_op(operator.mul, weights, time_values)
    assert jnp.allclose(outer, weights[:, None] * time_values[None, :])


def test_blend_with_gate_matches_manual_expression() -> None:
    dom = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)

    @dom.Function("x", "t")
    def base(x: Any, t: Any) -> Any:
        return x[0] + 2.0 * t

    @dom.Function("x", "t")
    def overlay(x: Any, t: Any) -> Any:
        return (x[0] ** 2) + (t**2)

    @dom.Function("x")
    def gate(x: Any) -> Any:
        return x[0]

    blended = blend_with_gate(base, overlay, gate)
    manual = base + gate * (overlay - base)

    batch = dom.component().sample(
        phx.domain.GridSampling(
            {"x": FourierAxisSpec(9)},
            dense=phx.domain.PointSampling(7, layout=SampleLayout((("t",),))),
        ),
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


def test_blend_with_gate_evaluates_inputs_once_per_call() -> None:
    dom = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    calls = {"base": 0, "overlay": 0, "gate": 0}

    @dom.Function("x", "t")
    def base(x: Any, t: Any) -> Any:
        calls["base"] += 1
        return x[0] + t

    @dom.Function("x", "t")
    def overlay(x: Any, t: Any) -> Any:
        calls["overlay"] += 1
        return (x[0] ** 2) + t

    @dom.Function("x")
    def gate(x: Any) -> Any:
        calls["gate"] += 1
        return x[0]

    blended = blend_with_gate(base, overlay, gate)
    out = blended.func(jnp.asarray([0.2]), jnp.asarray(0.3), key=jr.key(0))
    expected = (0.2 + 0.3) + 0.2 * (((0.2**2) + 0.3) - (0.2 + 0.3))
    assert jnp.allclose(jnp.asarray(out), jnp.asarray(expected))
    assert calls["base"] == 1
    assert calls["overlay"] == 1
    assert calls["gate"] == 1
