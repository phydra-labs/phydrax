from typing import Any

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
import phydrax.axes as cx
from phydrax._frozendict import frozendict
from phydrax.domain import TimeInterval
from phydrax.operators.differential import laplacian


def _square() -> phx.domain.GeometryDomain:
    return phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )


def test_laplacian_point_values_shapes_dtypes_and_metadata() -> None:
    geometry = _square()
    points = frozendict({"x": cx.AxisArray(jnp.asarray([2.0, 3.0]), dims=(None,))})

    @geometry.Function("x")
    def scalar(x: jax.Array) -> jax.Array:
        return x[0] ** 2 + x[1] ** 2

    @geometry.Function("x")
    def vector(x: jax.Array) -> jax.Array:
        return jnp.asarray([x[0] ** 2, x[1] ** 2])

    @geometry.Function("x")
    def complex_value(x: jax.Array) -> jax.Array:
        return x[0] ** 2 + 1j * x[1] ** 2

    cases = (
        ("scalar", scalar, jnp.asarray(4.0), ()),
        ("vector", vector, jnp.asarray([2.0, 2.0]), (2,)),
        ("complex", complex_value, jnp.asarray(2.0 + 2.0j), ()),
    )
    for case_id, function, expected, expected_shape in cases:
        result = jnp.asarray(laplacian(function)(points).data)
        assert result.shape == expected_shape, case_id
        assert jnp.allclose(result, expected), case_id

    annotated = geometry.Function("x")(lambda x: x[0] ** 2).with_metadata(tag=1)
    assert laplacian(annotated).metadata == annotated.metadata


def test_laplacian_respects_spacetime_and_coordinate_separable_layouts(
    sample_batch: Any,
    sample_grid: Any,
) -> None:
    geometry = _square()
    spacetime = geometry @ TimeInterval(0.0, 1.0)

    @spacetime.Function("x", "t")
    def time_dependent(x: jax.Array, time: jax.Array) -> jax.Array:
        return x[0] ** 2 + x[1] ** 2 + time

    point_batch = sample_batch(
        spacetime.component(),
        blocks=(("x",), ("t",)),
        num_points=(3, 4),
        key=0,
    )
    point_result = jnp.asarray(laplacian(time_dependent, var="x")(point_batch).data)
    assert point_result.shape == (3, 4)
    assert jnp.allclose(point_result, 4.0)

    grid_batch = sample_grid(geometry.component(), {"x": (6, 5)}, dense_blocks=(), key=0)

    @geometry.Function("x")
    def separable(x: jax.Array) -> jax.Array:
        x0, x1 = x
        return x0**2 + x1**2

    assert jnp.allclose(
        jnp.asarray(laplacian(separable)(grid_batch).data),
        4.0,
        atol=1e-6,
    )


def test_laplacian_jvp_engine_matches_default_for_point_and_grid_layouts(
    sample_grid: Any,
) -> None:
    geometry = _square()

    @geometry.Function("x")
    def function(x: jax.Array) -> jax.Array:
        return x[0] ** 2 + 3.0 * x[1] ** 2 + x[0] * x[1]

    points = frozendict({"x": cx.AxisArray(jnp.asarray([0.7, -0.2]), dims=(None,))})
    grid = sample_grid(geometry.component(), {"x": (7, 6)}, dense_blocks=(), key=3)
    for case_id, batch in (("point", points), ("grid", grid)):
        reference = jnp.asarray(laplacian(function, backend="ad")(batch).data)
        jvp = jnp.asarray(laplacian(function, backend="ad", ad_engine="jvp")(batch).data)
        assert jnp.allclose(jvp, reference, atol=1e-6), case_id


def test_laplacian_jvp_engine_requires_the_ad_backend() -> None:
    geometry = _square()

    @geometry.Function("x")
    def function(x: jax.Array) -> jax.Array:
        return x[0] ** 2 + x[1] ** 2

    with pytest.raises(ValueError, match="backend='ad'"):
        laplacian(function, backend="jet", ad_engine="jvp")
