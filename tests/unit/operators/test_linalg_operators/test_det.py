import jax
import jax.numpy as jnp

import phydrax as phx
import phydrax.axes as cx
from phydrax._frozendict import frozendict
from phydrax.domain import TimeInterval
from phydrax.operators.linalg import det


def test_determinant_matches_exact_real_complex_spacetime_and_rank3_references() -> None:
    square = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    cube = phx.domain.GeometryDomain(
        phx.geometry.Cube(center=(0.0, 0.0, 0.0), side=2.0).compile()
    )
    spacetime = square @ TimeInterval(0.0, 1.0)

    @square.Function("x")
    def diagonal(x: jax.Array) -> jax.Array:
        return jnp.asarray([[x[0], 0.0], [0.0, x[1]]])

    @square.Function("x")
    def complex_diagonal(x: jax.Array) -> jax.Array:
        return jnp.asarray([[x[0], 0.0], [0.0, 1j * x[1]]])

    @square.Function("x")
    def dense(x: jax.Array) -> jax.Array:
        return jnp.asarray([[x[0], x[1]], [x[1], x[0]]])

    @spacetime.Function("x", "t")
    def time_dependent(x: jax.Array, time: jax.Array) -> jax.Array:
        return jnp.asarray([[x[0] * time, 0.0], [0.0, x[1] * time]])

    @cube.Function("x")
    def rank3(x: jax.Array) -> jax.Array:
        return jnp.asarray([[x[0], x[1], x[2]], [x[2], x[0], x[1]], [x[1], x[2], x[0]]])

    xy = frozendict({"x": cx.AxisArray(jnp.asarray([2.0, 3.0]), dims=(None,))})
    cases = (
        ("diagonal", diagonal, xy, 6.0),
        ("complex", complex_diagonal, xy, 6.0j),
        (
            "dense",
            dense,
            frozendict({"x": cx.AxisArray(jnp.asarray([3.0, 2.0]), dims=(None,))}),
            5.0,
        ),
        (
            "spacetime",
            time_dependent,
            frozendict(
                {
                    "x": cx.AxisArray(jnp.asarray([2.0, 3.0]), dims=(None,)),
                    "t": cx.AxisArray(jnp.asarray(0.5), dims=()),
                }
            ),
            1.5,
        ),
        (
            "rank3",
            rank3,
            frozendict({"x": cx.AxisArray(jnp.asarray([1.0, 2.0, 3.0]), dims=(None,))}),
            18.0,
        ),
    )
    for case_id, function, points, expected in cases:
        result = jnp.asarray(det(function)(points).data)
        assert jnp.allclose(result, expected), case_id


def test_determinant_preserves_metadata() -> None:
    square = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    function = square.Function("x")(lambda x: jnp.eye(2)).with_metadata(tag="keep")
    assert det(function).metadata == function.metadata
