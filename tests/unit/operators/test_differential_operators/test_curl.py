from typing import Any

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
import phydrax.axes as cx
from phydrax._frozendict import frozendict
from phydrax.domain import TimeInterval
from phydrax.operators.differential import curl, scalar_curl_2d, vector_curl_2d


def test_curl_point_values_planar_shapes_and_metadata(box3d: Any) -> None:
    @box3d.Function("x")
    def vector3d(x: jax.Array) -> jax.Array:
        return jnp.asarray([x[1], -x[0], 0.0])

    points3d = frozendict({"x": cx.AxisArray(jnp.asarray([1.0, 2.0, 3.0]), dims=(None,))})
    result3d = jnp.asarray(curl(vector3d, var="x")(points3d).data)
    assert jnp.allclose(result3d, jnp.asarray([0.0, 0.0, -2.0]))
    annotated = vector3d.with_metadata(k=1)
    assert curl(annotated, var="x").metadata == annotated.metadata

    box2d = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @box2d.Function("x")
    def vector2d(x: jax.Array) -> jax.Array:
        return jnp.asarray((-x[1], x[0]))

    @box2d.Function("x")
    def scalar2d(x: jax.Array) -> jax.Array:
        return x[0] ** 2 + x[1] ** 2

    points2d = frozendict({"x": cx.AxisArray(jnp.asarray((0.5, -0.25)), dims=(None,))})
    assert jnp.allclose(jnp.asarray(vector_curl_2d(vector2d)(points2d).data), 2.0)
    assert jnp.allclose(
        jnp.asarray(scalar_curl_2d(scalar2d)(points2d).data),
        jnp.asarray((-0.5, -1.0)),
    )


def test_curl_respects_spacetime_and_coordinate_separable_layouts(
    sample_batch: Any,
    sample_grid: Any,
    box3d: Any,
) -> None:
    spacetime = box3d @ TimeInterval(0.0, 1.0)

    @spacetime.Function("x", "t")
    def time_dependent(x: jax.Array, time: jax.Array) -> jax.Array:
        return jnp.asarray([x[1] * time, -x[0] * time, 0.0])

    operator = curl(time_dependent, var="x")
    points = frozendict(
        {
            "x": cx.AxisArray(jnp.asarray([1.0, 2.0, 3.0]), dims=(None,)),
            "t": cx.AxisArray(jnp.asarray(0.5), dims=()),
        }
    )
    assert jnp.allclose(
        jnp.asarray(operator(points).data),
        jnp.asarray([0.0, 0.0, -1.0]),
    )

    point_batch = sample_batch(
        spacetime.component(),
        blocks=(("x",), ("t",)),
        num_points=(4, 5),
        key=0,
    )
    point_result = jnp.asarray(operator(point_batch).data)
    time = jnp.asarray(point_batch.points["t"].data)
    point_z = -2.0 * jnp.broadcast_to(time[None, :], point_result.shape[:2])
    assert point_result.shape == (4, 5, 3)
    assert jnp.allclose(
        point_result,
        jnp.stack(
            [jnp.zeros_like(point_z), jnp.zeros_like(point_z), point_z],
            axis=-1,
        ),
    )

    grid_batch = sample_grid(
        box3d.component(),
        {"x": (3, 4, 2)},
        dense_blocks=(),
        key=0,
    )

    @box3d.Function("x")
    def separable(x: jax.Array) -> jax.Array:
        x0, x1, x2 = x
        return jnp.stack([x1, -x0, jnp.zeros_like(x2)], axis=-1)

    grid_result = jnp.asarray(curl(separable, var="x")(grid_batch).data)
    grid_z = -2.0 * jnp.ones_like(grid_result[..., 0])
    assert grid_result.shape[-1] == 3
    assert jnp.allclose(
        grid_result,
        jnp.stack(
            [jnp.zeros_like(grid_z), jnp.zeros_like(grid_z), grid_z],
            axis=-1,
        ),
        atol=1e-6,
    )


def test_curl_jvp_engine_matches_default_and_requires_ad_backend(box3d: Any) -> None:
    @box3d.Function("x")
    def function(x: jax.Array) -> jax.Array:
        return jnp.asarray([x[1] + x[2], -x[0], x[0] * x[1]])

    points = frozendict({"x": cx.AxisArray(jnp.asarray([0.3, -0.4, 0.2]), dims=(None,))})
    reference = jnp.asarray(curl(function, var="x", backend="ad")(points).data)
    jvp = jnp.asarray(curl(function, var="x", backend="ad", ad_engine="jvp")(points).data)
    assert jnp.allclose(jvp, reference, atol=1e-6)
    with pytest.raises(ValueError, match="backend='ad'"):
        curl(function, var="x", backend="fd", ad_engine="jvp")
