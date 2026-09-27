#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.signal import convolve


def test_convolution_scenario_1() -> None:
    for mode in ("full", "same", "valid"):
        for tap_count in (3, 4):
            values = np.linspace(-1.0, 1.0, 9)
            taps = np.linspace(0.2, 0.8, tap_count)
            full = np.convolve(values, taps, mode="full")
            if mode == "full":
                expected = full
            elif mode == "same":
                start = (tap_count - 1) // 2
                expected = full[start : start + values.size]
            else:
                expected = full[tap_count - 1 : values.size]

            direct = convolve(values, taps, mode=mode, method="direct")
            transformed = convolve(values, taps, mode=mode, method="fft")

            assert np.allclose(direct, expected, rtol=1e-12, atol=1e-12)
            assert np.allclose(transformed, expected, rtol=1e-12, atol=1e-12)
    values = jnp.arange(2 * 7 * 3, dtype="float64").reshape((2, 7, 3))
    taps = jnp.asarray((1.0 + 0.5j, -0.25j))

    output = convolve(values, taps, axis=1, mode="same", method="fft")

    assert output.shape == values.shape
    assert jnp.issubdtype(output.dtype, jnp.complexfloating)
    assert jnp.allclose(
        output[1, :, 2],
        convolve(values[1, :, 2], taps, mode="same", method="direct"),
    )
    values = jnp.arange(8.0)
    taps = jnp.asarray((0.2, 0.5, -0.1))
    compiled = jax.jit(lambda x, h: convolve(x, h, mode="same", method="fft"))
    batched = jax.vmap(lambda x: convolve(x, taps, mode="same"))(
        jnp.stack((values, values + 1.0))
    )
    value_gradient, tap_gradient = jax.grad(
        lambda x, h: jnp.sum(jnp.abs(compiled(x, h)) ** 2),
        argnums=(0, 1),
    )(values, taps)

    assert batched.shape == (2, values.size)
    assert jnp.all(jnp.isfinite(value_gradient))
    assert jnp.all(jnp.isfinite(tap_gradient))
    with pytest.raises(ValueError, match="signal length"):
        convolve(jnp.ones((2,)), jnp.ones((3,)), mode="valid")
    with pytest.raises(ValueError, match="method"):
        # ty: ignore[invalid-argument-type]
        convolve(jnp.ones((3,)), jnp.ones((2,)), method="auto")
    with pytest.raises(ValueError, match="taps"):
        convolve(jnp.ones((3,)), jnp.ones((2, 1)))
