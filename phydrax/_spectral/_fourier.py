#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def resize_fourier_axis(
    coefficients: Array,
    axis: int,
    target_size: int,
    /,
) -> Array:
    """Resize one unshifted DFT axis with exact even-Nyquist handling."""
    source_size = int(coefficients.shape[axis])
    target = int(target_size)
    if target <= 0:
        raise ValueError("Fourier target sizes must be positive.")
    if source_size == target:
        return coefficients

    target_minimum = -(target // 2)
    target_maximum = (target - 1) // 2
    source_maximum = (source_size - 1) // 2
    primary_indices: list[int] = []
    primary_weights: list[float] = []
    secondary_indices: list[int] = []
    secondary_weights: list[float] = []

    for source_index in range(source_size):
        frequency = (
            source_index if source_index <= source_maximum else source_index - source_size
        )
        if (
            target > source_size
            and source_size % 2 == 0
            and source_index == source_size // 2
        ):
            mappings = (
                (-source_size // 2, 0.5),
                (source_size // 2, 0.5),
            )
        else:
            canonical = (
                -target // 2
                if target % 2 == 0 and frequency == target // 2
                else frequency
            )
            mappings = ((canonical, 1.0),)

        valid = tuple(
            (mapped_frequency % target, weight)
            for mapped_frequency, weight in mappings
            if target_minimum <= mapped_frequency <= target_maximum
        )
        primary_indices.append(valid[0][0] if valid else 0)
        primary_weights.append(valid[0][1] if valid else 0.0)
        secondary_indices.append(valid[1][0] if len(valid) > 1 else 0)
        secondary_weights.append(valid[1][1] if len(valid) > 1 else 0.0)

    moved = jnp.moveaxis(coefficients, axis, 0)
    weight_shape = (source_size,) + (1,) * (moved.ndim - 1)
    output = jnp.zeros((target, *moved.shape[1:]), dtype=coefficients.dtype)
    output = output.at[jnp.asarray(primary_indices)].add(
        moved * jnp.asarray(primary_weights, dtype=moved.real.dtype).reshape(weight_shape)
    )
    output = output.at[jnp.asarray(secondary_indices)].add(
        moved
        * jnp.asarray(secondary_weights, dtype=moved.real.dtype).reshape(weight_shape)
    )
    return jnp.moveaxis(output, 0, axis)


def _resolve_fourier_axes(
    array: Array,
    output_shape: tuple[int, ...],
    axes: Sequence[int] | None,
    /,
) -> tuple[int, ...]:
    if axes is None:
        if array.ndim < len(output_shape) + 1:
            raise ValueError(
                "Fourier resampling expects trailing spatial axes before a payload axis."
            )
        return tuple(range(array.ndim - len(output_shape) - 1, array.ndim - 1))
    resolved_input = tuple(axes)
    if array.ndim == 0 or len(resolved_input) != len(output_shape):
        raise ValueError("axes must provide one array axis per output size.")
    resolved = tuple(int(axis) % array.ndim for axis in resolved_input)
    if len(set(resolved)) != len(resolved):
        raise ValueError("Fourier resampling axes must be unique.")
    return resolved


def phase_shift_fourier_coefficients(
    coefficients: Array,
    axes: tuple[int, ...],
    source_shape: tuple[int, ...],
    offsets: Sequence[ArrayLike],
    /,
) -> Array:
    """Apply fractional-period shifts without ambiguous even Nyquist modes."""
    offsets_value = tuple(offsets)
    if len(offsets_value) != len(axes):
        raise ValueError("phase_offsets must provide one scalar per Fourier axis.")
    result = coefficients
    for axis, size in zip(axes, source_shape, strict=True):
        if size % 2 == 0:
            result = resize_fourier_axis(result, axis, size + 1)
    for local_axis, (axis, offset) in enumerate(zip(axes, offsets_value, strict=True)):
        raw_offset = jnp.asarray(offset)
        if raw_offset.ndim != 0:
            raise ValueError(
                f"phase_offsets[{local_axis}] must be scalar; got {raw_offset.shape}."
            )
        if jnp.issubdtype(raw_offset.dtype, jnp.complexfloating):
            raise TypeError("Fourier phase offsets must be real-valued.")
        offset_value = jnp.asarray(raw_offset, dtype=result.real.dtype)
        offset_value = eqx.error_if(
            offset_value,
            ~jnp.isfinite(offset_value),
            "Fourier phase offsets must be finite.",
        )
        size = int(result.shape[axis])
        modes = jnp.fft.fftfreq(size).astype(result.real.dtype) * size
        angle = 2.0 * jnp.asarray(jnp.pi, dtype=result.real.dtype) * modes * offset_value
        phase = jnp.exp(1j * angle).astype(result.dtype)
        shape = [1] * result.ndim
        shape[axis] = size
        result = result * phase.reshape(tuple(shape))
    return result


def fourier_resample(
    values: ArrayLike,
    output_shape: Sequence[int],
    /,
    *,
    axes: Sequence[int] | None = None,
    phase_offsets: Sequence[ArrayLike] | None = None,
) -> Array:
    """Band-limited periodic resampling with parity-correct Nyquist transfer."""
    array = jnp.asarray(values)
    shape = tuple(int(size) for size in output_shape)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("output_shape must contain positive spatial sizes.")
    resolved_axes = _resolve_fourier_axes(array, shape, axes)
    source_shape = tuple(int(array.shape[axis]) for axis in resolved_axes)
    if source_shape == shape and phase_offsets is None:
        return array
    coefficients = jnp.fft.fftn(array, axes=resolved_axes, norm="forward")
    if phase_offsets is not None:
        coefficients = phase_shift_fourier_coefficients(
            coefficients,
            resolved_axes,
            source_shape,
            phase_offsets,
        )
    for axis, size in zip(resolved_axes, shape, strict=True):
        coefficients = resize_fourier_axis(coefficients, axis, size)
    result = jnp.fft.ifftn(coefficients, axes=resolved_axes, norm="forward")
    return result if jnp.issubdtype(array.dtype, jnp.complexfloating) else result.real


__all__ = [
    "fourier_resample",
    "phase_shift_fourier_coefficients",
    "resize_fourier_axis",
]
