from typing import Any

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
import phydrax.axes as cx
from phydrax._frozendict import frozendict
from phydrax.operators.differential import div_K_grad, div_k_grad


def _square() -> phx.domain.GeometryDomain:
    return phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )


def test_scalar_coefficient_divergence_matches_scalar_vector_and_metadata_contracts() -> (
    None
):
    geometry = _square()

    @geometry.Function("x")
    def scalar(x: jax.Array) -> jax.Array:
        return x[0] ** 2 + x[1] ** 2

    @geometry.Function("x")
    def vector(x: jax.Array) -> jax.Array:
        return jnp.asarray([x[0] ** 2, x[1] ** 2])

    @geometry.Function("x")
    def coefficient(x: jax.Array) -> jax.Array:
        return x[0] + 2.0 * x[1]

    points = frozendict({"x": cx.AxisArray(jnp.asarray([2.0, -1.5]), dims=(None,))})
    cases = (
        ("scalar", scalar, jnp.asarray(6.0 * 2.0 + 12.0 * -1.5)),
        (
            "vector",
            vector,
            jnp.asarray([4.0 * 2.0 + 4.0 * -1.5, 2.0 * 2.0 + 8.0 * -1.5]),
        ),
    )
    for case_id, function, expected in cases:
        result = jnp.asarray(div_k_grad(function, coefficient)(points).data)
        assert jnp.allclose(result, expected), case_id

    annotated = geometry.Function("x")(lambda x: x[0] ** 2).with_metadata(scale=1)
    assert div_k_grad(annotated, 1.0).metadata == annotated.metadata


def test_tensor_coefficient_divergence_matches_scalar_and_vector_contracts() -> None:
    geometry = _square()

    @geometry.Function("x")
    def scalar(x: jax.Array) -> jax.Array:
        return x[0] ** 2 + x[1] ** 2

    @geometry.Function("x")
    def vector(x: jax.Array) -> jax.Array:
        return jnp.asarray([x[0] ** 2, x[1] ** 2])

    @geometry.Function("x")
    def coefficient(x: jax.Array) -> jax.Array:
        return jnp.asarray([[x[0], x[1]], [x[1], x[0]]])

    points = frozendict({"x": cx.AxisArray(jnp.asarray([2.0, -1.5]), dims=(None,))})
    assert jnp.allclose(
        jnp.asarray(div_K_grad(scalar, coefficient)(points).data),
        16.0,
    )
    assert jnp.allclose(
        jnp.asarray(div_K_grad(vector, coefficient)(points).data),
        jnp.asarray([12.0, 4.0]),
    )


def test_variable_coefficient_divergence_matches_coordinate_separable_references(
    sample_grid: Any,
) -> None:
    geometry = _square()

    @geometry.Function("x")
    def scalar(x: jax.Array) -> jax.Array:
        x0, x1 = x
        return x0**2 + x1**2

    @geometry.Function("x")
    def scalar_coefficient(x: jax.Array) -> jax.Array:
        x0, x1 = x
        return x0 + 2.0 * x1

    scalar_batch = sample_grid(
        geometry.component(),
        {"x": (6, 5)},
        dense_blocks=(),
        key=0,
    )
    x0 = jnp.asarray(scalar_batch.points["x"][0].data)
    x1 = jnp.asarray(scalar_batch.points["x"][1].data)
    mesh_x, mesh_y = jnp.meshgrid(x0, x1, indexing="ij")
    assert jnp.allclose(
        jnp.asarray(div_k_grad(scalar, scalar_coefficient)(scalar_batch).data),
        6.0 * mesh_x + 12.0 * mesh_y,
        atol=1e-6,
    )

    @geometry.Function("x")
    def tensor_coefficient(x: jax.Array) -> jax.Array:
        x0, x1 = x
        row0 = jnp.stack([x0, x1], axis=-1)
        row1 = jnp.stack([x1, x0], axis=-1)
        return jnp.stack([row0, row1], axis=-2)

    tensor_batch = sample_grid(
        geometry.component(),
        {"x": (5, 4)},
        dense_blocks=(),
        key=0,
    )
    tensor_x = jnp.asarray(tensor_batch.points["x"][0].data)
    tensor_y = jnp.asarray(tensor_batch.points["x"][1].data)
    tensor_mesh_x, _ = jnp.meshgrid(tensor_x, tensor_y, indexing="ij")
    assert jnp.allclose(
        jnp.asarray(div_K_grad(scalar, tensor_coefficient)(tensor_batch).data),
        8.0 * tensor_mesh_x,
        atol=1e-6,
    )


def test_variable_coefficient_jvp_engines_match_defaults_and_require_ad_backend() -> None:
    geometry = _square()

    @geometry.Function("x")
    def scalar(x: jax.Array) -> jax.Array:
        return x[0] ** 2 + x[1] ** 2

    @geometry.Function("x")
    def scalar_coefficient(x: jax.Array) -> jax.Array:
        return 1.0 + x[0] - 0.5 * x[1]

    @geometry.Function("x")
    def tensor_coefficient(x: jax.Array) -> jax.Array:
        return jnp.asarray([[1.0 + x[0], x[1]], [x[1], 2.0 + x[0]]])

    cases = (
        (
            "scalar",
            div_k_grad,
            scalar_coefficient,
            frozendict({"x": cx.AxisArray(jnp.asarray([0.3, -0.7]), dims=(None,))}),
        ),
        (
            "tensor",
            div_K_grad,
            tensor_coefficient,
            frozendict({"x": cx.AxisArray(jnp.asarray([0.2, -0.4]), dims=(None,))}),
        ),
    )
    for case_id, operator, coefficient, points in cases:
        reference = jnp.asarray(operator(scalar, coefficient, backend="ad")(points).data)
        jvp = jnp.asarray(
            operator(
                scalar,
                coefficient,
                backend="ad",
                ad_engine="jvp",
            )(points).data
        )
        assert jnp.allclose(jvp, reference, atol=1e-6), case_id

    with pytest.raises(ValueError, match="backend='ad'"):
        div_k_grad(scalar, 1.0, backend="fd", ad_engine="jvp")
