from typing import Any

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
import phydrax.axes as cx
from phydrax._frozendict import frozendict
from phydrax.domain import Interval1d, TimeInterval
from phydrax.operators.differential import div, div_tensor


pytestmark = [pytest.mark.strict_jax, pytest.mark.filterwarnings("error")]


def _square() -> phx.domain.GeometryDomain:
    return phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )


def test_divergence_point_metadata_and_nested_composition_contracts() -> None:
    geometry = _square()

    @geometry.Function("x")
    def vector(x: jax.Array) -> jax.Array:
        return jnp.asarray([x[0], x[1]])

    points = frozendict({"x": cx.AxisArray(jnp.asarray([2.0, 3.0]), dims=(None,))})
    assert jnp.allclose(jnp.asarray(div(vector)(points).data), 2.0)
    annotated = vector.with_metadata(tag=1)
    assert div(annotated).metadata == annotated.metadata

    interval = Interval1d(-2.0, 2.0)
    density = interval.Function("x")(lambda x: x[0] ** 2)
    drift = interval.Function("x")(lambda x: jnp.asarray([2.0 * x[0]]))
    covariance = interval.Function("x")(lambda x: jnp.asarray([[3.0 * x[0] ** 2]]))
    adjoint = -div(drift * density, var="x") + 0.5 * div(
        div_tensor(covariance * density, var="x"),
        var="x",
    )
    point = frozendict({"x": cx.AxisArray(jnp.asarray([0.4]), dims=(None,))})
    assert jnp.allclose(jnp.asarray(adjoint(point).data), 12.0 * 0.4**2)


def test_divergence_respects_spacetime_and_coordinate_separable_layouts(
    sample_batch: Any,
    sample_grid: Any,
) -> None:
    geometry = _square()
    spacetime = geometry @ TimeInterval(0.0, 1.0)

    @spacetime.Function("x")
    def time_independent(x: jax.Array) -> jax.Array:
        return jnp.asarray([x[0], x[1]])

    point_batch = sample_batch(
        spacetime.component(),
        blocks=(("x",), ("t",)),
        num_points=(4, 3),
        key=0,
    )
    point_result = jnp.asarray(div(time_independent, var="x")(point_batch).data)
    assert point_result.shape == (4, 3)
    assert jnp.allclose(point_result, 2.0)

    grid_batch = sample_grid(geometry.component(), {"x": (7, 6)}, dense_blocks=(), key=1)

    @geometry.Function("x")
    def separable(x: jax.Array) -> jax.Array:
        x0, x1 = x
        return jnp.stack([2.0 * x0, 3.0 * x1], axis=-1)

    assert jnp.allclose(
        jnp.asarray(div(separable)(grid_batch).data),
        5.0,
        atol=1e-6,
    )


def test_divergence_jvp_engine_matches_default_and_requires_ad_backend() -> None:
    geometry = _square()

    @geometry.Function("x")
    def vector(x: jax.Array) -> jax.Array:
        return jnp.asarray([x[0] ** 2 + x[1], x[1] ** 2 + x[0]])

    points = frozendict({"x": cx.AxisArray(jnp.asarray([0.3, -0.7]), dims=(None,))})
    reference = jnp.asarray(div(vector, backend="ad")(points).data)
    jvp = jnp.asarray(div(vector, backend="ad", ad_engine="jvp")(points).data)
    assert jnp.allclose(jvp, reference, atol=1e-6)
    with pytest.raises(ValueError, match="backend='ad'"):
        div(vector, backend="fd", ad_engine="jvp")
