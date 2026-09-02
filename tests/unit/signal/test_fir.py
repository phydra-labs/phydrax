#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import signal as scipy_signal

from phydrax.signal import fir_filter, FIRFilterPlan


def _active_values(result):
    return np.asarray(result.values)[np.asarray(result.active)]


def test_zero_state_fir_matches_scipy_and_preserves_middle_sample_axis():
    values = jnp.arange(2 * 11 * 3, dtype=float).reshape((2, 11, 3))
    taps = jnp.asarray((0.25, 0.5, 0.25))

    output = fir_filter(values, taps, axis=1)

    assert output.shape == values.shape
    assert np.allclose(
        output[1, :, 2],
        scipy_signal.lfilter(np.asarray(taps), (1.0,), np.asarray(values[1, :, 2])),
    )


def test_chunk_partitions_and_flush_equal_full_causal_convolution():
    values = jnp.linspace(-1.0, 1.0, 11)
    taps = jnp.asarray((0.2, 0.5, -0.1, 0.3))
    plan = FIRFilterPlan(taps.size)
    state = plan.initial_state((4,), dtype=jnp.float64)
    outputs = []
    chunks = (
        (values[:4], 4),
        (values[4:8], 4),
        (jnp.pad(values[8:], (0, 1)), 3),
    )
    for chunk, valid_length in chunks:
        state, result = plan.step(state, chunk, taps, valid_length=valid_length)
        outputs.append(_active_values(result))
    reset, tail = plan.flush(state, taps)
    outputs.append(_active_values(tail))

    actual = np.concatenate(outputs)
    expected = np.convolve(np.asarray(values), np.asarray(taps), mode="full")
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert int(state.sample_count) == values.size
    assert int(reset.sample_count) == 0
    assert np.allclose(reset.history, 0.0)


def test_zero_valid_chunk_is_a_state_preserving_noop():
    plan = FIRFilterPlan(3)
    taps = jnp.asarray((1.0, -0.5, 0.25))
    state = plan.initial_state((5,), dtype=jnp.float64)

    next_state, result = plan.step(
        state,
        jnp.arange(5.0),
        taps,
        valid_length=0,
    )

    assert jnp.array_equal(next_state.history, state.history)
    assert int(next_state.sample_count) == 0
    assert not bool(jnp.any(result.active))
    assert jnp.allclose(result.values, 0.0)


def test_fir_state_and_taps_remain_differentiable_through_jit():
    plan = FIRFilterPlan(3)
    state = plan.initial_state((6,), dtype=jnp.float64)
    values = jnp.arange(6.0)
    taps = jnp.asarray((0.2, 0.5, 0.3))

    @eqx.filter_jit
    def loss(history, coefficients):
        carried = eqx.tree_at(lambda item: item.history, state, history)
        _, result = plan.step(carried, values, coefficients)
        return jnp.sum(result.values**2)

    history_gradient, tap_gradient = jax.grad(loss, argnums=(0, 1))(
        state.history,
        taps,
    )

    assert jnp.all(jnp.isfinite(history_gradient))
    assert jnp.all(jnp.isfinite(tap_gradient))


def test_fir_rejects_state_from_an_incompatible_plan():
    state = FIRFilterPlan(3).initial_state((4,), dtype=jnp.float64)
    with pytest.raises(ValueError, match="different filter plan"):
        FIRFilterPlan(4).step(state, jnp.ones((4,)), jnp.ones((4,)))
