#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


layers = phx.nn.layers
nl = phx.nonlinear


def _config(*, steps=24, failure_policy="raise"):
    return layers.CausalRecurrentConfig(
        method=nl.CausalNewton(),
        termination=nl.NonlinearTermination(
            absolute_residual=1e-10,
            relative_residual=1e-10,
            maximum_steps=steps,
        ),
        failure_policy=failure_policy,
    )


@pytest.mark.parametrize(
    "cell",
    (
        layers.RNNCell(3, 4, dtype=jnp.float64, key=jax.random.key(1)),
        layers.GRUCell(3, 4, dtype=jnp.float64, key=jax.random.key(2)),
        layers.LSTMCell(3, 4, dtype=jnp.float64, key=jax.random.key(3)),
    ),
)
def test_causal_recurrent_matches_serial_masks_resets_and_gradients(cell):
    inputs = jax.random.normal(jax.random.key(4), (2, 10, 3), dtype=jnp.float64)
    valid = jnp.asarray([[True] * 8 + [False] * 2, [True] * 10])
    reset = jnp.zeros_like(valid).at[0, 4].set(True).at[1, 6].set(True)
    batch = layers.RecurrentBatch(inputs, valid, reset=reset)

    serial = layers.run_recurrent(cell, batch)
    causal = jax.jit(
        lambda current: layers.run_causal_recurrent(
            current,
            batch,
            config=_config(),
        )
    )(cell)

    assert jnp.all(causal.successful)
    assert jax.tree.all(
        jax.tree.map(
            lambda left, right: jnp.allclose(left, right, atol=1e-9, rtol=1e-9),
            causal.states,
            serial.states,
        )
    )
    assert jax.tree.all(
        jax.tree.map(
            lambda left, right: jnp.allclose(left, right, atol=1e-9, rtol=1e-9),
            causal.outputs,
            serial.outputs,
        )
    )
    assert jax.tree.all(
        jax.tree.map(
            lambda left, right: jnp.allclose(left, right, atol=1e-9, rtol=1e-9),
            causal.final_state,
            serial.final_state,
        )
    )

    serial_gradient = jax.grad(
        lambda current: sum(
            jnp.sum(leaf)
            for leaf in jax.tree.leaves(layers.run_recurrent(current, batch).outputs)
        )
    )(cell)
    causal_gradient = jax.grad(
        lambda current: sum(
            jnp.sum(leaf)
            for leaf in jax.tree.leaves(
                layers.run_causal_recurrent(current, batch, config=_config()).outputs
            )
        )
    )(cell)
    assert jax.tree.all(
        jax.tree.map(
            lambda left, right: jnp.allclose(left, right, atol=2e-8, rtol=2e-8),
            causal_gradient,
            serial_gradient,
        )
    )


def test_causal_recurrent_supports_multidimensional_cases_and_explicit_states():
    cell = layers.RNNCell(2, 3, dtype=jnp.float64, key=jax.random.key(5))
    inputs = jax.random.normal(jax.random.key(6), (2, 3, 7, 2), dtype=jnp.float64)
    valid = jnp.ones((2, 3, 7), dtype=bool)
    reset = jnp.zeros_like(valid).at[:, :, 4].set(True)
    batch = layers.RecurrentBatch(inputs, valid, reset=reset)
    initial = jnp.full((2, 3, 3), 0.1)
    restart = jnp.full((2, 3, 3), -0.2)

    serial = layers.run_recurrent(
        cell,
        batch,
        initial_state=initial,
        reset_state=restart,
    )
    causal = layers.run_causal_recurrent(
        cell,
        batch,
        initial_state=initial,
        reset_state=restart,
        config=_config(),
    )

    assert causal.states.shape == (2, 3, 7, 3)
    assert jnp.allclose(causal.states, serial.states, atol=1e-9)
    assert jnp.allclose(causal.final_output, serial.final_output, atol=1e-9)


def test_causal_recurrent_explicit_serial_fallback_is_recorded():
    cell = layers.RNNCell(1, 1, key=jax.random.key(7))
    batch = layers.RecurrentBatch(jnp.ones((6, 1)), jnp.ones((6,), dtype=bool))
    serial = layers.run_recurrent(cell, batch)
    causal = layers.run_causal_recurrent(
        cell,
        batch,
        config=layers.CausalRecurrentConfig(
            method=nl.CausalNewton(),
            termination=nl.NonlinearTermination(
                absolute_residual=0.0,
                relative_residual=0.0,
                maximum_steps=1,
            ),
            failure_policy="serial",
        ),
    )

    assert bool(causal.diagnostics.fallback_used)
    assert jnp.array_equal(causal.states, serial.states)
    assert jnp.array_equal(causal.outputs, serial.outputs)


def test_hutchinson_recurrent_requires_explicit_probe_key():
    cell = layers.RNNCell(1, 1, key=jax.random.key(8))
    batch = layers.RecurrentBatch(jnp.ones((4, 1)), jnp.ones((4,), dtype=bool))
    config = layers.CausalRecurrentConfig(
        method=nl.CausalNewton(
            linearization=nl.CausalLinearizationPolicy("diagonal-hutchinson")
        )
    )

    with pytest.raises(ValueError, match="probe_key"):
        layers.run_causal_recurrent(cell, batch, config=config)
