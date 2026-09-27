import jax
import jax.numpy as jnp

import phydrax as phx
import phydrax.axes as cx
from phydrax._frozendict import frozendict
from phydrax.domain import TimeInterval
from phydrax.operators.linalg import norm


def test_norm_matches_real_complex_ordered_spacetime_references_and_metadata() -> None:
    geometry = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    spacetime = geometry @ TimeInterval(0.0, 2.0)

    @geometry.Function("x")
    def vector(x: jax.Array) -> jax.Array:
        return jnp.asarray([x[0], x[1]])

    @geometry.Function("x")
    def complex_vector(x: jax.Array) -> jax.Array:
        return jnp.asarray([x[0], 1j * x[1]])

    @spacetime.Function("x", "t")
    def time_dependent(x: jax.Array, time: jax.Array) -> jax.Array:
        return jnp.asarray([x[0] * time, x[1] * time])

    positive = frozendict({"x": cx.AxisArray(jnp.asarray([3.0, 4.0]), dims=(None,))})
    negative = frozendict({"x": cx.AxisArray(jnp.asarray([3.0, -4.0]), dims=(None,))})
    cases = (
        ("euclidean", norm(vector), positive, 5.0),
        ("l1", norm(vector, order=1), negative, 7.0),
        ("complex", norm(complex_vector), positive, 5.0),
        (
            "spacetime",
            norm(time_dependent),
            frozendict(
                {
                    "x": cx.AxisArray(jnp.asarray([3.0, 4.0]), dims=(None,)),
                    "t": cx.AxisArray(jnp.asarray(2.0), dims=()),
                }
            ),
            10.0,
        ),
    )
    for case_id, function, points, expected in cases:
        assert jnp.allclose(jnp.asarray(function(points).data), expected), case_id

    annotated = geometry.Function("x")(lambda x: jnp.asarray([1.0, 2.0])).with_metadata(
        tag=1
    )
    assert norm(annotated).metadata == annotated.metadata
