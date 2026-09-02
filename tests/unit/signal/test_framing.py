#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.signal import frame, overlap_add


def test_frame_and_overlap_add_have_explicit_adjacent_axis_layout():
    values = jnp.arange(2 * 10 * 3, dtype=float).reshape((2, 10, 3))

    framed = frame(values, 4, 2, axis=1)
    restored_sum = overlap_add(framed, 2, frame_axis=1, sample_axis=2)

    assert framed.shape == (2, 4, 4, 3)
    assert restored_sum.shape == values.shape
    coverage = jnp.asarray((1, 1, 2, 2, 2, 2, 2, 2, 1, 1))
    assert jnp.allclose(restored_sum, values * coverage[None, :, None])


def test_framing_drops_only_incomplete_trailing_samples_and_supports_gaps():
    values = jnp.arange(11.0)

    framed = frame(values, 4, 3)
    gapped = overlap_add(frame(values[:10], 2, 4), 4)

    assert framed.shape == (3, 4)
    assert jnp.array_equal(framed[-1], values[6:10])
    assert jnp.array_equal(
        gapped,
        jnp.asarray((0.0, 1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 8.0, 9.0)),
    )


def test_framing_is_jittable_vmappable_and_differentiable():
    framed = jax.jit(lambda x: frame(x, 3, 2))(jnp.arange(7.0))
    batched = jax.vmap(lambda x: frame(x, 3, 2))(jnp.arange(14.0).reshape(2, 7))
    gradient = jax.grad(lambda x: jnp.sum(frame(x, 3, 2)))(jnp.arange(7.0))

    assert framed.shape == (3, 3)
    assert batched.shape == (2, 3, 3)
    assert np.allclose(gradient, np.asarray((1, 1, 2, 1, 2, 1, 1)))


def test_framing_validation_rejects_invalid_shapes_and_axes():
    with pytest.raises(ValueError, match="at least frame_length"):
        frame(jnp.ones((3,)), 4, 1)
    with pytest.raises(ValueError, match="distinct"):
        overlap_add(jnp.ones((2, 3)), 1, frame_axis=1, sample_axis=1)
    with pytest.raises(ValueError, match="positive"):
        frame(jnp.ones((3,)), 2, 0)
