from typing import Any

import jax
import jax.numpy as jnp

import phydrax as phx
import phydrax.axes as cx
from phydrax._frozendict import frozendict
from phydrax.domain import TimeInterval
from phydrax.operators.differential import hessian


def _square() -> phx.domain.GeometryDomain:
    return phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )


def test_hessian_point_values_shapes_dtypes_and_metadata() -> None:
    geometry = _square()
    points = frozendict({"x": cx.AxisArray(jnp.asarray([2.0, 3.0]), dims=(None,))})

    @geometry.Function("x")
    def scalar(x: jax.Array) -> jax.Array:
        return x[0] ** 2 + x[1] ** 2

    @geometry.Function("x")
    def vector(x: jax.Array) -> jax.Array:
        return jnp.asarray([x[0] ** 2, x[0] * x[1]])

    @geometry.Function("x")
    def complex_value(x: jax.Array) -> jax.Array:
        return x[0] ** 2 + 1j * x[1] ** 2

    expected_vector = jnp.asarray([[[2.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]])
    expected_complex = jnp.asarray([[2.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 2.0j]])
    cases = (
        ("scalar", scalar, 2.0 * jnp.eye(2), (2, 2)),
        ("vector", vector, expected_vector, (2, 2, 2)),
        ("complex", complex_value, expected_complex, (2, 2)),
    )
    for case_id, function, expected, expected_shape in cases:
        result = jnp.asarray(hessian(function)(points).data)
        assert result.shape == expected_shape, case_id
        assert jnp.allclose(result, expected), case_id

    annotated = geometry.Function("x")(lambda x: x[0] ** 2).with_metadata(tag=1)
    assert hessian(annotated).metadata == annotated.metadata


def test_hessian_respects_spacetime_and_coordinate_separable_layouts(
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
    point_result = jnp.asarray(hessian(time_dependent, var="x")(point_batch).data)
    assert point_result.shape == (3, 4, 2, 2)
    assert jnp.allclose(point_result, 2.0 * jnp.eye(2)[None, None, :, :])

    grid_batch = sample_grid(geometry.component(), {"x": (6, 5)}, dense_blocks=(), key=0)

    @geometry.Function("x")
    def separable(x: jax.Array) -> jax.Array:
        x0, x1 = x
        return x0**2 + x1**2

    grid_result = jnp.asarray(hessian(separable)(grid_batch).data)
    assert grid_result.shape == (6, 5, 2, 2)
    assert jnp.allclose(
        grid_result,
        2.0 * jnp.eye(2)[None, None, :, :],
        atol=1e-6,
    )


def test_hessian_time_only_scalar_preserves_the_time_axis() -> None:
    domain = TimeInterval(0.0, 1.0)

    @domain.Function("t")
    def function(time: jax.Array) -> jax.Array:
        return time**3

    times = jnp.linspace(0.0, 1.0, 7)
    batch = frozendict({"t": cx.AxisArray(times, dims=("t",))})
    result = jnp.asarray(hessian(function)(batch).data)
    assert result.shape == times.shape
    assert jnp.allclose(result, 6.0 * times)
