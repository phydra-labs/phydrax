from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
import phydrax.axes as cx
from phydrax._frozendict import frozendict
from phydrax.domain import Interval1d, SampleLayout, TimeInterval
from phydrax.operators.differential import dt_n, partial_t


def test_temporal_derivative_values_shapes_broadcasting_and_metadata(
    sample_batch: Any,
) -> None:
    time = TimeInterval(0.0, 1.0)

    @time.Function("t")
    def scalar(value: jax.Array) -> jax.Array:
        return value**2

    @time.Function("t")
    def vector(value: jax.Array) -> jax.Array:
        return jnp.stack([value**2, value**3], axis=-1)

    scalar_times = jnp.linspace(0.0, 1.0, 7)
    scalar_batch = frozendict({"t": cx.AxisArray(scalar_times, dims=("t",))})
    scalar_result = jnp.asarray(partial_t(scalar)(scalar_batch).data)
    assert scalar_result.shape == scalar_times.shape
    assert jnp.allclose(scalar_result, 2.0 * scalar_times)

    vector_times = jnp.linspace(0.0, 1.0, 5)
    vector_batch = frozendict({"t": cx.AxisArray(vector_times, dims=("t",))})
    vector_result = jnp.asarray(partial_t(vector)(vector_batch).data)
    vector_expected = jnp.stack(
        [2.0 * vector_times, 3.0 * vector_times**2],
        axis=-1,
    )
    assert vector_result.shape == vector_expected.shape
    assert jnp.allclose(vector_result, vector_expected)

    annotated = scalar.with_metadata(scale=3)
    assert partial_t(annotated).metadata == annotated.metadata

    space = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    spacetime = space @ time

    @spacetime.Function("t")
    def time_only(value: jax.Array) -> jax.Array:
        return jnp.sin(value)

    batch = sample_batch(
        spacetime.component(),
        blocks=(("x",), ("t",)),
        num_points=(4, 6),
        key=0,
    )
    broadcast = jnp.asarray(partial_t(time_only)(batch).data)
    sampled_times = jnp.asarray(batch.points["t"].data)
    assert broadcast.shape == (4, 6)
    assert jnp.allclose(broadcast, jnp.cos(sampled_times)[None, :])


def test_temporal_product_and_quotient_rules_skip_time_independent_factors() -> None:
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)

    @domain.Function("x")
    def spatial(x: jax.Array) -> jax.Array:
        return x[0] + 2.0

    @domain.Function("t")
    def cubic(time: jax.Array) -> jax.Array:
        return time**3

    @domain.Function("t")
    def affine(time: jax.Array) -> jax.Array:
        return 1.0 + time

    component = domain.component()
    product_batch = component.sample(
        phx.domain.PointSampling(
            (5, 7),
            layout=SampleLayout((("x",), ("t",))),
        ),
        key=jr.key(0),
    )
    product_x = jnp.asarray(product_batch.points["x"].data[:, 0])
    product_t = jnp.asarray(product_batch.points["t"].data)
    product_expected = (product_x[:, None] + 2.0) * (3.0 * product_t**2)[None, :]
    assert jnp.allclose(
        jnp.asarray(partial_t(spatial * cubic, var="t")(product_batch).data),
        product_expected,
        atol=1e-6,
    )

    quotient_batch = component.sample(
        phx.domain.PointSampling(
            (6, 8),
            layout=SampleLayout((("x",), ("t",))),
        ),
        key=jr.key(1),
    )
    quotient_x = jnp.asarray(quotient_batch.points["x"].data[:, 0])
    quotient_t = jnp.asarray(quotient_batch.points["t"].data)
    second_expected = (6.0 * quotient_t)[None, :] / (quotient_x[:, None] + 2.0)
    assert jnp.allclose(
        jnp.asarray(dt_n(cubic / spatial, var="t", order=2)(quotient_batch).data),
        second_expected,
        atol=1e-6,
    )

    reciprocal_batch = component.sample(
        phx.domain.PointSampling(
            (4, 9),
            layout=SampleLayout((("x",), ("t",))),
        ),
        key=jr.key(2),
    )
    reciprocal_x = jnp.asarray(reciprocal_batch.points["x"].data[:, 0])
    reciprocal_t = jnp.asarray(reciprocal_batch.points["t"].data)
    first_expected = -(reciprocal_x[:, None] + 2.0) / ((1.0 + reciprocal_t)[None, :] ** 2)
    assert jnp.allclose(
        jnp.asarray(partial_t(spatial / affine, var="t")(reciprocal_batch).data),
        first_expected,
        atol=1e-6,
    )


def test_temporal_jvp_engine_matches_default_and_requires_ad_backend() -> None:
    domain = TimeInterval(0.0, 1.0)

    @domain.Function("t")
    def first_order(time: jax.Array) -> jax.Array:
        return jnp.sin(time) + time**3

    @domain.Function("t")
    def second_order(time: jax.Array) -> jax.Array:
        return time**4 + 2.0 * time

    first_times = jnp.linspace(0.0, 1.0, 11)
    first_batch = frozendict({"t": cx.AxisArray(first_times, dims=("t",))})
    first_reference = jnp.asarray(partial_t(first_order)(first_batch).data)
    first_jvp = jnp.asarray(partial_t(first_order, ad_engine="jvp")(first_batch).data)
    assert jnp.allclose(first_jvp, first_reference, atol=1e-6)

    second_times = jnp.linspace(0.0, 1.0, 9)
    second_batch = frozendict({"t": cx.AxisArray(second_times, dims=("t",))})
    second_reference = jnp.asarray(
        dt_n(second_order, order=2, backend="ad")(second_batch).data
    )
    second_jvp = jnp.asarray(
        dt_n(second_order, order=2, backend="ad", ad_engine="jvp")(second_batch).data
    )
    assert jnp.allclose(second_jvp, second_reference, atol=1e-6)

    with pytest.raises(ValueError, match="backend='ad'"):
        dt_n(second_order, order=2, backend="jet", ad_engine="jvp")
