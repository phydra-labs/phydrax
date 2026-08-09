#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def stable_exponential_phi(z: Array, /) -> tuple[Array, Array]:
    """Evaluate the first two exponential phi functions near zero stably."""
    values = jnp.asarray(z)
    real_dtype = jnp.real(values).dtype
    if not jnp.issubdtype(real_dtype, jnp.floating):
        raise TypeError("z must have a real or complex floating dtype.")
    threshold = jnp.sqrt(jnp.asarray(jnp.finfo(real_dtype).eps, dtype=real_dtype))
    small = jnp.abs(values) < threshold
    safe_z = jnp.where(small, jnp.ones_like(values), values)
    phi_one_quotient = jnp.expm1(safe_z) / safe_z
    phi_two_quotient = (phi_one_quotient - 1.0) / safe_z
    series_z = jnp.where(small, values, jnp.zeros_like(values))
    z2 = series_z * series_z
    z3 = z2 * series_z
    z4 = z3 * series_z
    phi_one_series = 1.0 + series_z / 2.0 + z2 / 6.0 + z3 / 24.0 + z4 / 120.0
    phi_two_series = 0.5 + series_z / 6.0 + z2 / 24.0 + z3 / 120.0 + z4 / 720.0
    return (
        jnp.where(small, phi_one_series, phi_one_quotient),
        jnp.where(small, phi_two_series, phi_two_quotient),
    )


def normalize_physical_schedule(
    times: ArrayLike,
    /,
    *,
    case_shape: tuple[int, ...],
    sequence_length: int,
    mask: ArrayLike | None,
    reset: ArrayLike | None,
    dtype: Any,
    require_prefix: bool,
) -> tuple[Array, Array, Array, Array]:
    """Normalize one physical schedule and validate its segment boundaries."""
    length = int(sequence_length)
    if length <= 0:
        raise ValueError("sequence_length must be positive.")
    expected_shape = tuple(case_shape) + (length,)
    time_values = jnp.asarray(times)
    if jnp.issubdtype(time_values.dtype, jnp.complexfloating):
        raise TypeError("times must be real-valued.")
    if time_values.shape == (length,):
        time_values = jnp.broadcast_to(time_values, expected_shape)
    elif time_values.shape != expected_shape:
        raise ValueError("times must be shared or have shape case_shape + (length,).")

    if mask is None:
        valid = jnp.ones(expected_shape, dtype=bool)
    else:
        valid = jnp.asarray(mask, dtype=bool)
        if valid.shape == (length,):
            valid = jnp.broadcast_to(valid, expected_shape)
        elif valid.shape != expected_shape:
            raise ValueError("mask must be shared or have shape case_shape + (length,).")

    if reset is None:
        resets = jnp.zeros(expected_shape, dtype=bool)
    else:
        resets = jnp.asarray(reset, dtype=bool)
        if resets.shape == (length,):
            resets = jnp.broadcast_to(resets, expected_shape)
        elif resets.shape != expected_shape:
            raise ValueError("reset must be shared or have shape case_shape + (length,).")

    time_values = time_values.astype(dtype)
    time_values = eqx.error_if(
        time_values,
        jnp.any(valid & ~jnp.isfinite(time_values)),
        "Valid time nodes must be finite.",
    )
    time_values = eqx.error_if(
        time_values,
        jnp.any(resets & ~valid),
        "reset=True requires a valid physical sample.",
    )
    if length == 1:
        continuation = jnp.zeros(expected_shape[:-1] + (0,), dtype=bool)
        return time_values, valid, resets, continuation

    valid_after_padding = valid[..., 1:] & ~valid[..., :-1]
    if require_prefix:
        time_values = eqx.error_if(
            time_values,
            jnp.any(valid_after_padding),
            "mask must describe a valid prefix of each ragged schedule.",
        )
        continuation = valid[..., :-1] & valid[..., 1:]
    else:
        time_values = eqx.error_if(
            time_values,
            jnp.any(valid_after_padding & ~resets[..., 1:]),
            "A valid node after padding must declare reset=True.",
        )
        continuation = valid[..., :-1] & valid[..., 1:] & ~resets[..., 1:]
    time_values = eqx.error_if(
        time_values,
        jnp.any(continuation & (time_values[..., 1:] < time_values[..., :-1])),
        "Continuation times must be non-decreasing within each segment.",
    )
    return time_values, valid, resets, continuation


__all__ = ["normalize_physical_schedule", "stable_exponential_phi"]
