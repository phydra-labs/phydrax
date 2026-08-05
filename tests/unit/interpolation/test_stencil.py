#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

from phydrax._interpolation import apply_gather_stencil, GatherStencil


def _linear_stencil() -> GatherStencil:
    return GatherStencil(
        indices=jnp.asarray([[0, 1], [1, 2]], dtype=jnp.int32),
        weights=jnp.asarray([[0.75, 0.25], [0.25, 0.75]]),
        source_size=3,
    )


def test_gather_stencil_preserves_payload_shape_constants_and_complex_values():
    values = jnp.asarray(
        [[1.0 + 2.0j, 3.0 - 1.0j], [1.0 + 2.0j, 3.0 - 1.0j], [1.0 + 2.0j, 3.0 - 1.0j]]
    )

    result = jax.jit(lambda x: apply_gather_stencil(x, _linear_stencil()).values)(values)

    assert result.shape == (2, 2)
    assert jnp.allclose(result, values[:2])
    assert jnp.issubdtype(result.dtype, jnp.complexfloating)


def test_gather_stencil_renormalizes_valid_sources_and_reports_support():
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


def test_gather_stencil_value_gradient_is_the_declared_linear_map():
    stencil = _linear_stencil()

    def total(values):
        return jnp.sum(apply_gather_stencil(values, stencil).values)

    gradient = jax.grad(total)(jnp.asarray([1.0, 2.0, 3.0]))
    assert jnp.allclose(gradient, jnp.asarray([0.75, 0.5, 0.75]))
