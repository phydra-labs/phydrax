from typing import Any

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
import phydrax.axes as cx
from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, TimeInterval
from phydrax.operators.differential import directional_derivative


def _square() -> phx.domain.GeometryDomain:
    return phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )


def test_directional_derivative_point_values_direction_fields_and_metadata() -> None:
    geometry = _square()

    @geometry.Function("x")
    def scalar(x: jax.Array) -> jax.Array:
        return x[0] ** 2 + x[1] ** 2

    @geometry.Function("x")
    def vector(x: jax.Array) -> jax.Array:
        return jnp.asarray([x[0] ** 2, x[1] ** 2])

    constant_x = DomainFunction(domain=geometry, deps=(), func=jnp.asarray([1.0, 0.0]))
    constant_xy = DomainFunction(domain=geometry, deps=(), func=jnp.asarray([1.0, 1.0]))
    radial = geometry.Function("x")(lambda x: x)
    points = frozendict({"x": cx.AxisArray(jnp.asarray([2.0, 3.0]), dims=(None,))})
    cases = (
        ("scalar", scalar, constant_x, jnp.asarray(4.0)),
        ("vector", vector, constant_xy, jnp.asarray([4.0, 6.0])),
        ("field-direction", scalar, radial, jnp.asarray(26.0)),
    )
    for case_id, function, direction, expected in cases:
        result = jnp.asarray(directional_derivative(function, direction)(points).data)
        assert jnp.allclose(result, expected), case_id

    annotated = scalar.with_metadata(scale=7)
    assert directional_derivative(annotated, constant_x).metadata == annotated.metadata


def test_directional_derivative_respects_spacetime_and_separable_layouts(
    sample_batch: Any,
    sample_grid: Any,
) -> None:
    geometry = _square()
    spacetime = geometry @ TimeInterval(0.0, 1.0)
    direction = DomainFunction(domain=geometry, deps=(), func=jnp.asarray([1.0, 0.0]))

    @spacetime.Function("x", "t")
    def time_dependent(x: jax.Array, time: jax.Array) -> jax.Array:
        return x[0] ** 2 + x[1] ** 2 + time

    point_batch = sample_batch(
        spacetime.component(),
        blocks=(("x",), ("t",)),
        num_points=(3, 4),
        key=0,
    )
    point_result = jnp.asarray(
        directional_derivative(time_dependent, direction, var="x")(point_batch).data
    )
    coordinates = jnp.asarray(point_batch.points["x"].data)
    assert point_result.shape == (3, 4)
    assert jnp.allclose(point_result, 2.0 * coordinates[..., 0:1])

    grid_batch = sample_grid(geometry.component(), {"x": (5, 4)}, dense_blocks=(), key=0)

    @geometry.Function("x")
    def separable(x: jax.Array) -> jax.Array:
        x0, x1 = x
        return x0**2 + x1**2

    grid_result = jnp.asarray(
        directional_derivative(separable, direction)(grid_batch).data
    )
    x0 = jnp.asarray(grid_batch.points["x"][0].data)
    x1 = jnp.asarray(grid_batch.points["x"][1].data)
    mesh_x, _ = jnp.meshgrid(x0, x1, indexing="ij")
    assert jnp.allclose(grid_result, 2.0 * mesh_x, atol=1e-6)


def test_directional_derivative_jvp_matches_default_and_requires_ad_backend() -> None:
    geometry = _square()

    @geometry.Function("x")
    def function(x: jax.Array) -> jax.Array:
        return x[0] ** 2 + x[1] ** 2 + x[0] * x[1]

    direction = DomainFunction(
        domain=geometry,
        deps=(),
        func=jnp.asarray([1.0, -0.25]),
    )
    points = frozendict({"x": cx.AxisArray(jnp.asarray([0.2, -0.4]), dims=(None,))})
    reference = jnp.asarray(
        directional_derivative(function, direction, backend="ad")(points).data
    )
    jvp = jnp.asarray(
        directional_derivative(
            function,
            direction,
            backend="ad",
            ad_engine="jvp",
        )(points).data
    )
    assert jnp.allclose(jvp, reference, atol=1e-6)
    with pytest.raises(ValueError, match="backend='ad'"):
        directional_derivative(function, direction, backend="fd", ad_engine="jvp")
