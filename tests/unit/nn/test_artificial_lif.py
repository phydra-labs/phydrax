# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from math import log

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.nn.layers import (
    ArtificialLIFCell,
    RecurrentBatch,
    run_recurrent,
    StackedRecurrentCell,
)
from phydrax.nn.models import RecurrentSequenceModel


def _cell(**kwargs):
    cell = ArtificialLIFCell(
        1, 1, time_constant_ms=1.0, dt_ms=log(2.0), use_bias=False, **kwargs
    )
    return eqx.tree_at(
        lambda current: (current.weight_ih, current.weight_hh),
        cell,
        (jnp.ones((1, 1)), jnp.full((1, 1), 0.6)),
    )


@pytest.mark.parametrize("reset_mode", ("subtract", "hard"))
def test_lif_previous_spike_drive_and_single_spike_overflow(reset_mode):
    cell = _cell(reset_mode=reset_mode)
    state, spikes = cell.step((jnp.array([0.2]), jnp.array([1.0])), jnp.array([8.0]))
    # Half-life charging: .5*.2 + .5*(8 + .6*1) = 4.4. There is one
    # binary spike, not four events; subtractive reset retains excess charge.
    np.testing.assert_array_equal(spikes, [1.0])
    np.testing.assert_allclose(state[0], [3.4 if reset_mode == "subtract" else 0.0])
    unchanged, duplicate_spikes = cell.step_with_context(
        state, jnp.array([100.0]), time=jnp.array(2.0), interval=jnp.array(0.0)
    )
    np.testing.assert_array_equal(duplicate_spikes, [0.0])
    for actual, expected in zip(unchanged, state, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("family", ("fast_sigmoid", "triangular"))
def test_spike_tangent_matches_declared_estimator_not_binary_finite_difference(family):
    width = 0.5
    cell = _cell(surrogate=family, surrogate_width=width)
    charged = jnp.array([-0.5, 0.75, 1.0, 1.25, 2.5])[:, None]
    state = cell.initial_state((5,), dtype=jnp.float32)
    function = lambda inputs: cell.step(state, inputs)[1]
    primal, tangent = jax.jvp(function, (2.0 * charged,), (jnp.ones_like(charged),))
    scaled_margin = (np.asarray(charged) - 1.0) / width
    slope = (
        0.5 / (width * (1.0 + np.abs(scaled_margin)) ** 2)
        if family == "fast_sigmoid"
        else np.maximum(1.0 - np.abs(scaled_margin), 0.0) / width
    )
    np.testing.assert_array_equal(primal, np.asarray(charged >= 1.0, dtype=np.float32))
    np.testing.assert_allclose(tangent, 0.5 * slope, atol=1e-7)
    np.testing.assert_allclose(
        jax.grad(lambda inputs: jnp.sum(function(inputs)))(2.0 * charged),
        0.5 * slope,
        atol=1e-7,
    )
    _, zero_tangent = jax.jvp(function, (2.0 * charged,), (jnp.zeros_like(charged),))
    np.testing.assert_array_equal(zero_tangent, jnp.zeros_like(charged))


@pytest.mark.parametrize("reset_mode", ("subtract", "hard"))
@pytest.mark.parametrize("detach", (False, True))
def test_detached_reset_blocks_only_reset_spike_tangent(reset_mode, detach):
    cell = _cell(reset_mode=reset_mode, detach_reset=detach, surrogate_width=0.5)
    state = cell.initial_state((), dtype=jnp.float32)
    (next_state, spike), (state_tangent, spike_tangent) = jax.jvp(
        lambda inputs: cell.step(state, inputs),
        (jnp.array([2.4]),),
        (jnp.ones((1,)),),
    )
    charged = 1.2
    slope = 1.0 / (2.0 * 0.5 * (1.0 + abs((charged - 1.0) / 0.5)) ** 2)
    reset_slope = 0.0 if detach else slope
    membrane_tangent = (
        -0.5 * charged * reset_slope
        if reset_mode == "hard"
        else 0.5 * (1.0 - reset_slope)
    )
    np.testing.assert_array_equal(spike, [1.0])
    np.testing.assert_allclose(
        next_state[0], [0.0 if reset_mode == "hard" else 0.2], atol=1e-7
    )
    np.testing.assert_allclose(state_tangent[0], [membrane_tangent], atol=1e-7)
    np.testing.assert_allclose(spike_tangent, [0.5 * slope], atol=1e-7)
    np.testing.assert_allclose(state_tangent[1], spike_tangent, atol=1e-7)


@pytest.mark.parametrize("detach", (False, True))
def test_reset_detachment_retains_previous_spike_recurrent_gradient(detach):
    cell = _cell(reset_mode="hard", detach_reset=detach, surrogate_width=0.5)

    def second_spike(first_input):
        state, _ = cell.step(cell.initial_state((), dtype=jnp.float32), first_input)
        return cell.step(state, jnp.array([1.8]))[1]

    spike, tangent = jax.jvp(second_spike, (jnp.array([2.4]),), (jnp.ones((1,)),))
    # Both charged voltages are 1.2. With detached hard reset only the
    # previous-spike recurrent path survives; with propagated reset the
    # negative membrane-reset path also contributes to the second spike.
    slope = 1.0 / (2.0 * 0.5 * (1.0 + 0.2 / 0.5) ** 2)
    expected = (0.15 if detach else -0.15) * slope**2
    np.testing.assert_array_equal(spike, [1.0])
    np.testing.assert_allclose(tangent, [expected], atol=1e-7)


def test_physical_time_streaming_preserves_padding_resets_and_boundary_tangents():
    cell = _cell()
    values = jnp.array(
        [
            [9.0, 5.0, 99.0, 0.2, 0.0, 9.0, 2.5, 0.0],
            [9.0, 1.0, 2.0, 4.0, 9.0, 3.0, 0.5, 0.0],
        ]
    )[..., None]
    valid = jnp.array(
        [
            [True, True, True, True, False, True, True, False],
            [True, True, True, True, True, True, True, False],
        ]
    )
    reset = jnp.array(
        [
            [False, False, False, False, False, True, False, False],
            [False, False, False, False, True, False, False, False],
        ]
    )
    time = jnp.array(
        [
            [4.0, 4.5, 4.5, 5.5, jnp.nan, 10.0, 12.0, jnp.nan],
            [0.0, 0.25, 0.75, 1.0, 0.0, 0.5, 1.0, jnp.nan],
        ]
    )

    def full(inputs):
        return run_recurrent(cell, RecurrentBatch(inputs, valid, reset=reset, time=time))

    def chunked(inputs):
        first = run_recurrent(
            cell,
            RecurrentBatch(
                inputs[:, :2], valid[:, :2], reset=reset[:, :2], time=time[:, :2]
            ),
        )
        second = run_recurrent(
            cell,
            RecurrentBatch(
                inputs[:, 2:], valid[:, 2:], reset=reset[:, 2:], time=time[:, 2:]
            ),
            initial_state=first.final_state,
            initial_context=first.final_context,
        )
        return jnp.concatenate((first.outputs, second.outputs), axis=1), second

    result = eqx.filter_jit(full)(values)
    output, last = eqx.filter_jit(chunked)(values)
    expected_membrane = np.zeros((2, 8, 1))
    expected_spike_state = np.zeros((2, 8, 1))
    expected_output = np.zeros((2, 8, 1))
    for case in range(2):
        membrane = previous_spike = 0.0
        previous_time = None
        for index in range(8):
            if bool(valid[case, index]):
                if bool(reset[case, index]):
                    membrane = previous_spike = 0.0
                    previous_time = None
                node = float(time[case, index])
                elapsed = 0.0 if previous_time is None else node - previous_time
                if elapsed > 0.0:
                    fraction = -np.expm1(-elapsed)
                    drive = float(values[case, index, 0]) + 0.6 * previous_spike
                    charged = membrane + fraction * (drive - membrane)
                    previous_spike = float(charged >= 1.0)
                    membrane = charged - previous_spike
                    expected_output[case, index, 0] = previous_spike
                previous_time = node
            expected_membrane[case, index, 0] = membrane
            expected_spike_state[case, index, 0] = previous_spike
    np.testing.assert_allclose(result.states[0], expected_membrane, atol=3e-7)
    np.testing.assert_array_equal(result.states[1], expected_spike_state)
    np.testing.assert_array_equal(result.outputs, expected_output)
    np.testing.assert_array_equal(output, result.outputs)
    for actual, expected in zip(last.final_state, result.final_state, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-7)
    np.testing.assert_array_equal(last.final_context.time, result.final_context.time)
    full_gradient = jax.grad(
        lambda inputs: (
            jnp.sum(full(inputs).outputs) + 0.25 * jnp.sum(full(inputs).final_state[0])
        )
    )(values)

    def chunk_loss(inputs):
        spikes, continuation = chunked(inputs)
        return jnp.sum(spikes) + 0.25 * jnp.sum(continuation.final_state[0])

    np.testing.assert_allclose(jax.grad(chunk_loss)(values), full_gradient, atol=2e-7)
    np.testing.assert_array_equal(
        full_gradient[~valid], jnp.zeros_like(full_gradient[~valid])
    )


def test_stacked_sequence_model_keeps_physical_intervals():
    cell = _cell()
    stacked = StackedRecurrentCell((cell,))
    batch = RecurrentBatch(
        jnp.array([[100.0], [4.0], [100.0], [0.0]]),
        jnp.ones((4,), dtype=bool),
        time=jnp.array([0.0, log(2.0), log(2.0), 2.0 * log(2.0)]),
    )
    np.testing.assert_array_equal(
        RecurrentSequenceModel(stacked)(batch), [[0.0], [1.0], [0.0], [0.0]]
    )
    final = RecurrentSequenceModel(stacked, return_mode="final")(batch)
    np.testing.assert_array_equal(final, [0.0])
