import jax
import jax.numpy as jnp

import phydrax as phx
import phydrax.axes as cx
from phydrax._frozendict import frozendict
from phydrax.domain import TimeInterval
from phydrax.operators.linalg import trace


def test_trace_matches_real_complex_spacetime_references_and_metadata() -> None:
    geometry = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    spacetime = geometry @ TimeInterval(0.0, 1.0)

    @geometry.Function("x")
    def matrix(x: jax.Array) -> jax.Array:
        return jnp.asarray([[x[0], 0.0], [0.0, x[1]]])

    @geometry.Function("x")
    def complex_matrix(x: jax.Array) -> jax.Array:
        return jnp.asarray([[x[0], 0.0], [0.0, 1j * x[1]]])

    @spacetime.Function("x", "t")
    def time_dependent(x: jax.Array, time: jax.Array) -> jax.Array:
        return jnp.asarray([[x[0] * time, 0.0], [0.0, x[1] * time]])

    points = frozendict({"x": cx.AxisArray(jnp.asarray([2.0, 3.0]), dims=(None,))})
    cases = (
        ("real", trace(matrix), points, 5.0),
        ("complex", trace(complex_matrix), points, 2.0 + 3.0j),
        (
            "spacetime",
            trace(time_dependent),
            frozendict(
                {
                    "x": cx.AxisArray(jnp.asarray([2.0, 3.0]), dims=(None,)),
                    "t": cx.AxisArray(jnp.asarray(0.5), dims=()),
                }
            ),
            2.5,
        ),
    )
    for case_id, function, batch, expected in cases:
        assert jnp.allclose(jnp.asarray(function(batch).data), expected), case_id

    annotated = geometry.Function("x")(lambda x: jnp.eye(2)).with_metadata(tag="keep")
    assert trace(annotated).metadata == annotated.metadata
