#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.signal import windows as scipy_windows

from phydrax.signal import (
    blackman_window,
    hamming_window,
    hann_window,
    kaiser_window,
    tukey_window,
)


def test_windows_scenario_1() -> None:
    for length in (1, 2, 7, 8):
        for periodic in (False, True):
            symmetric = not periodic
            cases = (
                (hann_window, scipy_windows.hann, ()),
                (hamming_window, scipy_windows.hamming, ()),
                (blackman_window, scipy_windows.blackman, ()),
                (kaiser_window, scipy_windows.kaiser, (5.0,)),
                (tukey_window, scipy_windows.tukey, (0.35,)),
            )
            for actual_fn, expected_fn, parameters in cases:
                # ty: ignore[missing-argument, too-many-positional-arguments]
                actual = actual_fn(
                    length, *parameters, periodic=periodic, dtype=jnp.float64
                )
                # ty: ignore[no-matching-overload]
                expected = expected_fn(length, *parameters, sym=symmetric)
                assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
    kaiser_energy = jax.jit(lambda beta: jnp.sum(kaiser_window(9, beta) ** 2))
    tukey_energy = jax.jit(lambda alpha: jnp.sum(tukey_window(9, alpha) ** 2))

    assert jnp.isfinite(jax.grad(kaiser_energy)(jnp.asarray(4.0)))
    assert jnp.isfinite(jax.grad(tukey_energy)(jnp.asarray(0.4)))
    with pytest.raises(ValueError, match="length"):
        hann_window(0)
    with pytest.raises(TypeError, match="floating dtype"):
        hann_window(4, dtype=jnp.int32)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="beta"):
        kaiser_window(4, -1.0)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="alpha"):
        tukey_window(4, -0.1)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="alpha"):
        tukey_window(4, 1.1)
