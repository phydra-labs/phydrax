#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import capture_environment, logical_array_bytes, measure_repeated

from phydrax import signal


def _profile(
    operation: Callable[[], Any],
    /,
    *,
    repeats: int,
) -> dict[str, Any]:
    result, timing = measure_repeated(operation, warmup=1, repeats=repeats)
    leaves = jax.tree.leaves(result)
    checksum = float(
        sum(
            np.asarray(jnp.sum(jnp.abs(leaf)))
            for leaf in leaves
            if isinstance(leaf, (jax.Array, np.ndarray))
        )
    )
    return {
        "timing": timing.to_milliseconds_dict(),
        "logical_output_bytes": logical_array_bytes(result),
        "checksum": checksum,
    }


def run(*, repeats: int) -> dict[str, Any]:
    key = jax.random.key(19)
    real = jax.random.normal(key, (16, 4096), dtype=jnp.float64)
    complex_values = real + 1j * jax.random.normal(
        jax.random.fold_in(key, 1), real.shape, dtype=jnp.float64
    )
    short_taps = signal.kaiser_sinc_resampling_filter(3, 2, half_width=10)
    long_taps = signal.kaiser_sinc_resampling_filter(5, 7, half_width=20)
    fir_taps = jnp.hanning(257)
    fir_taps = fir_taps / jnp.sum(fir_taps)
    resampling_plan = signal.RationalResamplingPlan(
        3,
        2,
        short_taps.size,
        256,
    )
    resampling_state = resampling_plan.initial_state(
        (16, 256),
        dtype=jnp.float64,
    )
    stream_chunk = real[:, :256]

    operations = {
        "fourier_resize_real": jax.jit(
            lambda: signal.fourier_resample(real, (6144,), axes=(1,))
        ),
        "fourier_resize_complex": jax.jit(
            lambda: signal.fourier_resample(complex_values, (3072,), axes=(1,))
        ),
        "convolution_direct": jax.jit(
            lambda: signal.convolve(real, fir_taps, axis=1, mode="same")
        ),
        "convolution_fft": jax.jit(
            lambda: signal.convolve(
                real,
                fir_taps,
                axis=1,
                mode="same",
                method="fft",
            )
        ),
        "fir_value_and_grad": jax.jit(
            lambda: jax.value_and_grad(
                lambda taps: jnp.sum(signal.fir_filter(real, taps, axis=1) ** 2)
            )(fir_taps)
        ),
        "upfirdn_3_2": jax.jit(
            lambda: signal.upfirdn(real, short_taps, up=3, down=2, axis=1)
        ),
        "upfirdn_5_7_complex": jax.jit(
            lambda: signal.upfirdn(
                complex_values,
                long_taps,
                up=5,
                down=7,
                axis=1,
            )
        ),
        "resample_poly_147_160": jax.jit(
            lambda: signal.resample_poly(real, 147, 160, axis=1)
        ),
        "streaming_resample_step": jax.jit(
            lambda: resampling_plan.step(
                resampling_state,
                stream_chunk,
                short_taps,
            )
        ),
    }
    return {
        "environment": capture_environment().to_dict(),
        "inputs": {
            "real_bytes": logical_array_bytes(real),
            "complex_bytes": logical_array_bytes(complex_values),
            "stream_state_bytes": logical_array_bytes(resampling_state),
        },
        "profiles": {
            name: _profile(operation, repeats=repeats)
            for name, operation in operations.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    report = run(repeats=arguments.repeats)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
