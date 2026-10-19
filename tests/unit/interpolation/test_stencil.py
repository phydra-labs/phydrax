#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._interpolation import apply_gather_stencil, GatherStencil
from phydrax._interpolation._types import MaskMode


def _linear_stencil() -> GatherStencil:
    return GatherStencil(
        indices=jnp.asarray([[0, 1], [1, 2]], dtype=jnp.int32),
        weights=jnp.asarray([[0.75, 0.25], [0.25, 0.75]]),
        source_size=3,
    )


def test_gather_stencil_contracts() -> None:
    values = jnp.asarray(
        [[1.0 + 2.0j, 3.0 - 1.0j], [1.0 + 2.0j, 3.0 - 1.0j], [1.0 + 2.0j, 3.0 - 1.0j]]
    )

    result = jax.jit(lambda x: apply_gather_stencil(x, _linear_stencil()).values)(values)

    assert result.shape == (2, 2)
    assert jnp.allclose(result, values[:2])
    assert jnp.issubdtype(result.dtype, jnp.complexfloating)
    stencil = _linear_stencil()
    values = jnp.asarray([2.0, 6.0, 10.0])
    source_mask = jnp.asarray([True, False, False])

    renormalized = apply_gather_stencil(
        values,
        stencil,
        source_mask=source_mask,
        mask_mode="renormalize",
    )
    strict = apply_gather_stencil(
        values,
        stencil,
        source_mask=source_mask,
        mask_mode="strict",
    )

    assert jnp.allclose(renormalized.values, jnp.asarray([2.0, 0.0]))
    assert jnp.array_equal(renormalized.support, jnp.asarray([True, False]))
    assert jnp.allclose(strict.values, 0.0)
    assert not jnp.any(strict.support)


def test_gather_stencil_value_gradient_is_the_declared_linear_map() -> None:
    stencil = _linear_stencil()

    def total(values: Any) -> Any:
        return jnp.sum(apply_gather_stencil(values, stencil).values)

    gradient = jax.grad(total)(jnp.asarray([1.0, 2.0, 3.0]))
    assert jnp.allclose(gradient, jnp.asarray([0.75, 0.5, 0.75]))


@pytest.mark.parametrize("dtype", (np.dtype(np.float32), np.dtype(np.float64)))
@pytest.mark.parametrize("query_shape", ((0,), (0, 8)))
@pytest.mark.parametrize("mode", ("strict", "renormalize", "reject"))
def test_empty_gather_preserves_payload_dtype_and_full_adjoint(
    dtype: np.dtype, query_shape: tuple[int, ...], mode: MaskMode
) -> None:
    stencil = GatherStencil(
        indices=jnp.empty(query_shape + (2,), dtype=jnp.int32),
        weights=jnp.empty(query_shape + (2,), dtype=dtype),
        source_size=3,
    )
    payload_dtype = np.result_type(dtype, np.complex64)
    source = jnp.asarray(
        ((1.0 + 2.0j, 3.0), (-0.5, 2.0 - 1.0j), (0.7, -0.2)),
        dtype=payload_dtype,
    )
    source_mask = jnp.ones((3,), dtype=jnp.bool_)

    def apply(values: jax.Array) -> jax.Array:
        return apply_gather_stencil(
            values, stencil, source_mask=source_mask, mask_mode=mode
        ).values

    direction = jnp.ones_like(source)
    cotangent = jnp.empty(query_shape + (2,), dtype=payload_dtype)
    (values, tangent), gradient = jax.jit(
        lambda value: (
            jax.jvp(apply, (value,), (direction,)),
            jax.vjp(apply, value)[1](cotangent)[0],
        )
    )(source)
    assert values.shape == tangent.shape == query_shape + (2,)
    assert values.dtype == tangent.dtype == source.dtype
    np.testing.assert_array_equal(gradient, np.zeros(source.shape, dtype=payload_dtype))
    result = apply_gather_stencil(
        source, stencil, source_mask=source_mask, mask_mode=mode
    )
    assert result.support.shape == query_shape
    assert result.support.dtype == jnp.bool_


@pytest.mark.parametrize("dtype", (np.dtype(np.float32), np.dtype(np.float64)))
def test_nonempty_gather_retains_values_dtype_and_nonfinite_signal(
    dtype: np.dtype,
) -> None:
    source = jnp.asarray((2.0, 6.0, 10.0), dtype=dtype)
    stencil = GatherStencil(
        indices=jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
        weights=jnp.asarray(((0.75, 0.25), (0.25, 0.75)), dtype=dtype),
        source_size=3,
    )
    result = jax.jit(apply_gather_stencil)(source, stencil)
    assert result.values.dtype == source.dtype
    np.testing.assert_array_equal(result.values, np.asarray((3.0, 9.0), dtype=dtype))
    np.testing.assert_array_equal(
        result.support, np.asarray((True, True), dtype=np.bool_)
    )
    invalid = GatherStencil(
        indices=jnp.asarray(((0, 1),), dtype=jnp.int32),
        weights=jnp.asarray(((jnp.nan, 0.5),), dtype=dtype),
        source_size=3,
    )
    refused_values = jax.jit(apply_gather_stencil)(source, invalid).values
    assert not bool(jnp.all(jnp.isfinite(refused_values)))
