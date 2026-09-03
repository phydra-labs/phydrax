#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_public_trainable_fir_and_streaming_rate_conversion_workflow():
    time = jnp.arange(12, dtype=float)
    signal = jnp.cos(2.0 * jnp.pi * 0.08 * time) + 0.25j * jnp.sin(
        2.0 * jnp.pi * 0.17 * time
    )
    fir_taps = jnp.asarray((0.2, 0.6, 0.2))
    filtered = phx.signal.fir_filter(signal, fir_taps)
    prototype = phx.signal.kaiser_sinc_resampling_filter(3, 2, half_width=3)
    plan = phx.signal.RationalResamplingPlan(3, 2, prototype.size, 4)
    state = plan.initial_state((4,), dtype=filtered.dtype)
    outputs = []

    @eqx.filter_jit
    def process(carried, chunk, taps):
        return plan.step(carried, chunk, taps)

    for chunk in filtered.reshape((3, 4)):
        state, result = process(state, chunk, prototype)
        outputs.append(np.asarray(result.values)[np.asarray(result.active)])
    _, tail = plan.flush(state, prototype)
    outputs.append(np.asarray(tail.values)[np.asarray(tail.active)])

    actual = np.concatenate(outputs)
    expected = np.asarray(phx.signal.upfirdn(filtered, prototype * 3, up=3, down=2))
    tap_gradient = jax.grad(
        lambda taps: jnp.sum(
            jnp.abs(
                phx.signal.upfirdn(
                    phx.signal.fir_filter(signal, taps), prototype * 3, up=3, down=2
                )
            )
            ** 2
        )
    )(fir_taps)

    assert np.allclose(actual, expected, rtol=1e-11, atol=1e-11)
    assert jnp.all(jnp.isfinite(tap_gradient))
