#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.signal import (
    kaiser_sinc_resampling_filter,
    RationalResamplingPlan,
    upfirdn,
)


def _active_values(result):
    return np.asarray(result.values)[np.asarray(result.active)]


def test_streaming_chunks_and_flush_equal_causal_raw_upfirdn():
    values = jnp.linspace(-1.0, 1.0, 11)
    prototype = kaiser_sinc_resampling_filter(3, 2, half_width=3)
    plan = RationalResamplingPlan(3, 2, prototype.size, 4)
    state = plan.initial_state((4,), dtype=jnp.float64)
    outputs = []
    offsets = []
    chunks = (
        (values[:4], 4),
        (values[4:8], 4),
        (jnp.pad(values[8:], (0, 1)), 3),
    )
    for chunk, valid_length in chunks:
        state, result = plan.step(state, chunk, prototype, valid_length=valid_length)
        outputs.append(_active_values(result))
        offsets.append(int(result.sample_offset))
    reset, tail = plan.flush(state, prototype)
    outputs.append(_active_values(tail))

    actual = np.concatenate(outputs)
    expected = np.asarray(upfirdn(values, prototype * 3, up=3, down=2))
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert offsets == [0, 6, 12]
    assert int(state.input_count) == values.size
    assert int(state.output_count) == 17
    assert int(reset.input_count) == 0
    assert int(reset.output_count) == 0


def test_zero_valid_resampling_chunk_is_a_state_preserving_noop():
    prototype = kaiser_sinc_resampling_filter(3, 2, half_width=2)
    plan = RationalResamplingPlan(3, 2, prototype.size, 4)
    state = plan.initial_state((4,), dtype=jnp.float64)

    next_state, result = plan.step(
        state,
        jnp.arange(4.0),
        prototype,
        valid_length=0,
    )

    assert jnp.array_equal(next_state.history, state.history)
    assert int(next_state.input_count) == 0
    assert int(next_state.output_count) == 0
    assert not bool(jnp.any(result.active))
    assert jnp.allclose(result.values, 0.0)


def test_streaming_resampling_is_jittable_vmappable_and_differentiable():
    prototype = kaiser_sinc_resampling_filter(3, 2, half_width=2)
    plan = RationalResamplingPlan(3, 2, prototype.size, 4)
    state = plan.initial_state((4,), dtype=jnp.float64)
    chunks = jnp.arange(8.0).reshape((2, 4))

    @eqx.filter_jit
    def two_steps(initial_state, values, taps):
        def body(carried, chunk):
            carried, result = plan.step(carried, chunk, taps)
            return carried, result.values

        return jax.lax.scan(body, initial_state, values)

    final_state, outputs = two_steps(state, chunks, prototype)
    tap_gradient = jax.grad(lambda taps: jnp.sum(two_steps(state, chunks, taps)[1] ** 2))(
        prototype
    )
    batched = jax.vmap(lambda chunk: plan.step(state, chunk, prototype)[1].values)(
        jnp.stack((chunks[0], chunks[0] + 1.0))
    )

    assert outputs.shape == (2, plan.output_capacity)
    assert int(final_state.input_count) == 8
    assert jnp.all(jnp.isfinite(tap_gradient))
    assert batched.shape == (2, plan.output_capacity)


def test_resampling_plan_rejects_incompatible_topology_and_chunk_shape():
    with pytest.raises(ValueError, match="divisible"):
        RationalResamplingPlan(3, 2, 7, 3)
    plan = RationalResamplingPlan(3, 2, 7, 4)
    state = plan.initial_state((4,), dtype=jnp.float64)
    with pytest.raises(ValueError, match="length 4"):
        plan.step(state, jnp.ones((5,)), jnp.ones((7,)))
