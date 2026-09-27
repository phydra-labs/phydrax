import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
import phydrax.axes as cx
from phydrax._frozendict import frozendict
from phydrax.operators.linalg import einsum


def test_domain_contractions_match_dot_outer_matrix_vector_and_constant_references() -> (
    None
):
    geometry = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geometry.Function("x")
    def first(x: jax.Array) -> jax.Array:
        return jnp.asarray([x[0], x[1]])

    @geometry.Function("x")
    def second(x: jax.Array) -> jax.Array:
        return jnp.asarray([x[1], x[0]])

    @geometry.Function("x")
    def matrix(x: jax.Array) -> jax.Array:
        return jnp.asarray([[x[0], 0.0], [0.0, x[1]]])

    @geometry.Function("x")
    def triple(x: jax.Array) -> jax.Array:
        return jnp.asarray([x[0], x[1], x[0] - x[1]])

    constant = jnp.asarray([[3.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 3.0]])
    points = frozendict({"x": cx.AxisArray(jnp.asarray([2.0, 3.0]), dims=(None,))})
    cases = (
        ("dot", einsum("i,i->", first, second), jnp.asarray(12.0), ()),
        (
            "outer",
            einsum("i,j->ij", first, second),
            jnp.asarray([[6.0, 4.0], [9.0, 6.0]]),
            (2, 2),
        ),
        (
            "matrix-vector",
            einsum("ij,j->i", matrix, second),
            jnp.asarray([6.0, 6.0]),
            (2,),
        ),
        (
            "constant-matrix",
            einsum("ij,...j->...i", constant, triple),
            constant @ jnp.asarray([2.0, 3.0, -1.0]),
            (3,),
        ),
    )
    for case_id, function, expected, expected_shape in cases:
        result = jnp.asarray(function(points).data)
        assert result.shape == expected_shape, case_id
        assert jnp.allclose(result, expected), case_id


def test_domain_contraction_contracts() -> None:
    geometry = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    first = geometry.Function("x")(lambda x: jnp.asarray([x[0], x[1]])).with_metadata(m=1)
    second = geometry.Function("x")(lambda x: jnp.asarray([x[1], x[0]])).with_metadata(
        m=2
    )
    assert einsum("i,i->", first, second).metadata == {}
    with pytest.raises(ValueError, match="at least one DomainFunction operand"):
        einsum("ij,j->i", jnp.eye(2), jnp.asarray([1.0, 2.0]))
