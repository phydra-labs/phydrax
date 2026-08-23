#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


ComplexParameterMode: TypeAlias = Literal["real", "holomorphic", "nonholomorphic"]


class LogAmplitude(StrictModule):
    """Stable log magnitude and unit phase of a real- or complex-valued amplitude."""

    log_abs: Array
    phase: Array
    valid: Array
    nonzero: Array

    def __init__(
        self,
        log_abs: ArrayLike,
        phase: ArrayLike = 1.0 + 0.0j,
        /,
        *,
        valid: ArrayLike | None = None,
    ):
        magnitude = jnp.asarray(log_abs)
        if jnp.iscomplexobj(magnitude):
            raise TypeError("log_abs must be real-valued.")
        phase_value = jnp.asarray(phase)
        if phase_value.shape != magnitude.shape:
            phase_value = jnp.broadcast_to(phase_value, magnitude.shape)
        if not jnp.iscomplexobj(phase_value):
            phase_value = phase_value.astype(jnp.result_type(phase_value, 1.0j))
        admissible_magnitude = ~jnp.isnan(magnitude) & ~jnp.isposinf(magnitude)
        finite_phase = jnp.isfinite(phase_value)
        unit_phase = jnp.isclose(jnp.abs(phase_value), 1.0, rtol=1e-6, atol=1e-7)
        resolved_valid = admissible_magnitude & finite_phase & unit_phase
        if valid is not None:
            declared = jnp.asarray(valid, dtype=bool)
            if declared.shape != magnitude.shape:
                declared = jnp.broadcast_to(declared, magnitude.shape)
            resolved_valid = resolved_valid & declared
        self.log_abs = magnitude
        self.phase = phase_value
        self.valid = resolved_valid
        self.nonzero = resolved_valid & jnp.isfinite(magnitude)


class AmplitudeRatio(StrictModule):
    """Amplitude ratio with explicit validity at nodes and nonfinite values."""

    value: Array
    valid: Array


def sampling_log_weight(amplitude: LogAmplitude, /) -> Array:
    """Return the real log target ``2 log |psi|`` with zero mass at invalid states."""
    if not isinstance(amplitude, LogAmplitude):
        raise TypeError("amplitude must be a LogAmplitude.")
    return jnp.where(amplitude.valid, 2.0 * amplitude.log_abs, -jnp.inf)


def amplitude_ratio(
    proposed: LogAmplitude,
    current: LogAmplitude,
    /,
) -> AmplitudeRatio:
    """Return ``psi(proposed) / psi(current)`` without materializing either amplitude."""
    if not isinstance(proposed, LogAmplitude) or not isinstance(current, LogAmplitude):
        raise TypeError("proposed and current must be LogAmplitude values.")
    if proposed.log_abs.shape != current.log_abs.shape:
        raise ValueError("Amplitude ratio inputs must have identical shapes.")
    exponent = jnp.where(
        proposed.nonzero,
        proposed.log_abs - current.log_abs,
        0.0,
    )
    magnitude = jnp.where(proposed.nonzero, jnp.exp(exponent), 0.0)
    value = magnitude * proposed.phase * jnp.conj(current.phase)
    valid = proposed.valid & current.valid & current.nonzero & jnp.isfinite(value)
    return AmplitudeRatio(value=jnp.where(valid, value, 0.0j), valid=valid)


__all__ = [
    "AmplitudeRatio",
    "ComplexParameterMode",
    "LogAmplitude",
    "amplitude_ratio",
    "sampling_log_weight",
]
