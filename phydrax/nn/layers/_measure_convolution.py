# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule


Padding = Literal["SAME", "VALID"] | tuple[tuple[int, int], ...]


def _sizes(value: int | Sequence[int], ndim: int, /, *, owner: str) -> tuple[int, ...]:
    if isinstance(value, int):
        result = (int(value),) * ndim
    else:
        result = tuple(int(size) for size in value)
        if len(result) != ndim:
            raise ValueError(f"{owner} must contain {ndim} entries.")
    if any(size <= 0 for size in result):
        raise ValueError(f"{owner} entries must be positive.")
    return result


def _padding(value: str | Sequence[tuple[int, int]], ndim: int, /) -> Padding:
    if isinstance(value, str):
        canonical = value.upper()
        if canonical not in ("SAME", "VALID"):
            raise ValueError("padding must be 'SAME', 'VALID', or explicit pairs.")
        return canonical
    result = tuple((int(left), int(right)) for left, right in value)
    if len(result) != ndim or any(left < 0 or right < 0 for left, right in result):
        raise ValueError("Explicit padding must contain one non-negative pair per axis.")
    return result


def _dimension_numbers(ndim: int, /) -> tuple[str, str, str]:
    if ndim == 1:
        return "NWC", "WIO", "NWC"
    if ndim == 2:
        return "NHWC", "HWIO", "NHWC"
    if ndim == 3:
        return "NDHWC", "DHWIO", "NDHWC"
    raise ValueError("MeasureNormalizedConvND supports one, two, or three dimensions.")


def _broadcast_sample_field(
    value: ArrayLike,
    case_shape: tuple[int, ...],
    sample_shape: tuple[int, ...],
    /,
    *,
    owner: str,
    dtype: Any,
) -> Array:
    array = jnp.asarray(value, dtype=dtype)
    expected = case_shape + sample_shape
    if array.shape == sample_shape:
        return jnp.broadcast_to(array, expected)
    if array.shape != expected:
        raise ValueError(
            f"{owner} must have shared shape {sample_shape} or full shape {expected}; "
            f"got {array.shape}."
        )
    return array


class _AbstractMeasureNormalizedConvND(StrictModule):
    r"""Channels-last convolution normalized by observed physical measure.

    The numerator convolves ``where(source_mask, values, 0) * quadrature`` with
    the learned kernel. The denominator is the mean non-negative observed
    quadrature over the in-domain geometric stencil, computed with a separate
    all-ones kernel. Learned signed weights never enter the denominator.
    Consequently, uniform full support exactly reproduces ordinary convolution,
    physical missingness is renormalized, and boundary padding retains ordinary
    convolution semantics.
    """

    weight: Array
    bias: Array | None
    spatial_ndim: int
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, ...]
    strides: tuple[int, ...]
    dilation: tuple[int, ...]
    padding: Padding
    epsilon: float

    def __init__(
        self,
        *,
        spatial_ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        strides: int | Sequence[int] = 1,
        dilation: int | Sequence[int] = 1,
        padding: str | Sequence[tuple[int, int]] = "SAME",
        use_bias: bool = True,
        epsilon: float = 1e-12,
        dtype: Any = jnp.float32,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        ndim = int(spatial_ndim)
        _dimension_numbers(ndim)
        in_count = int(in_channels)
        out_count = int(out_channels)
        if in_count <= 0 or out_count <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        kernel = _sizes(kernel_size, ndim, owner="kernel_size")
        stride = _sizes(strides, ndim, owner="strides")
        dilation_value = _sizes(dilation, ndim, owner="dilation")
        padding_value = _padding(padding, ndim)
        epsilon_value = float(epsilon)
        if not math.isfinite(epsilon_value) or epsilon_value <= 0.0:
            raise ValueError("epsilon must be finite and positive.")
        real_dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(real_dtype, jnp.floating):
            raise TypeError("dtype must be a real floating dtype.")

        receptive_size = math.prod(kernel)
        fan_in = receptive_size * in_count
        fan_out = receptive_size * out_count
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        weight_shape = kernel + (in_count, out_count)
        self.weight = jr.uniform(
            key,
            weight_shape,
            minval=-limit,
            maxval=limit,
            dtype=real_dtype,
        )
        self.bias = jnp.zeros((out_count,), dtype=real_dtype) if use_bias else None
        self.spatial_ndim = ndim
        self.in_channels = in_count
        self.out_channels = out_count
        self.kernel_size = kernel
        self.strides = stride
        self.dilation = dilation_value
        self.padding = padding_value
        self.epsilon = epsilon_value

    def _convolve(self, values: Array, kernel: Array, /) -> Array:
        return jax.lax.conv_general_dilated(
            values,
            kernel,
            window_strides=self.strides,
            padding=self.padding,
            rhs_dilation=self.dilation,
            dimension_numbers=_dimension_numbers(self.spatial_ndim),
        )

    def __call__(
        self,
        values: Array,
        /,
        *,
        source_mask: ArrayLike | None = None,
        target_mask: ArrayLike | None = None,
        quadrature: ArrayLike | None = None,
    ) -> Array:
        inputs = jnp.asarray(values)
        minimum_rank = self.spatial_ndim + 1
        if inputs.ndim < minimum_rank:
            raise ValueError(
                "values must have case axes followed by spatial axes and channels."
            )
        if int(inputs.shape[-1]) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {inputs.shape[-1]}."
            )
        sample_shape = tuple(
            int(size) for size in inputs.shape[-(self.spatial_ndim + 1) : -1]
        )
        case_shape = tuple(int(size) for size in inputs.shape[: -(self.spatial_ndim + 1)])
        compute_dtype = jnp.result_type(inputs.dtype, self.weight.dtype)
        inputs = inputs.astype(compute_dtype)
        source_valid = (
            jnp.ones(case_shape + sample_shape, dtype=bool)
            if source_mask is None
            else _broadcast_sample_field(
                source_mask,
                case_shape,
                sample_shape,
                owner="source_mask",
                dtype=bool,
            )
        )
        measure = (
            jnp.ones(case_shape + sample_shape, dtype=compute_dtype)
            if quadrature is None
            else _broadcast_sample_field(
                quadrature,
                case_shape,
                sample_shape,
                owner="quadrature",
                dtype=compute_dtype,
            )
        )
        measure = eqx.error_if(
            measure,
            jnp.any(~jnp.isfinite(measure) | (measure < 0.0)),
            "quadrature must be finite and non-negative.",
        )

        case_count = math.prod(case_shape) if case_shape else 1
        flat_shape = (case_count,) + sample_shape
        flat_inputs = jnp.reshape(inputs, flat_shape + (self.in_channels,))
        flat_valid = jnp.reshape(source_valid, flat_shape)
        flat_measure = jnp.reshape(measure, flat_shape)
        measured_inputs = (
            jnp.where(flat_valid[..., None], flat_inputs, jnp.zeros_like(flat_inputs))
            * flat_measure[..., None]
        )
        numerator = self._convolve(measured_inputs, self.weight.astype(compute_dtype))

        stencil = jnp.ones(self.kernel_size + (1, 1), dtype=compute_dtype)
        measured_support = jnp.where(
            flat_valid, flat_measure, jnp.zeros_like(flat_measure)
        )
        support_sum = self._convolve(measured_support[..., None], stencil)
        domain_count = self._convolve(
            jnp.ones(flat_shape + (1,), dtype=compute_dtype), stencil
        )
        mean_measure = support_sum / domain_count
        has_support = support_sum > 0.0
        output = numerator / jnp.maximum(mean_measure, self.epsilon)
        if self.bias is not None:
            output = output + self.bias.astype(compute_dtype)
        output = jnp.where(has_support, output, jnp.zeros_like(output))

        output_sample_shape = tuple(int(size) for size in output.shape[1:-1])
        output = jnp.reshape(
            output, case_shape + output_sample_shape + (self.out_channels,)
        )
        if target_mask is not None:
            target_valid = _broadcast_sample_field(
                target_mask,
                case_shape,
                output_sample_shape,
                owner="target_mask",
                dtype=bool,
            )
            output = jnp.where(target_valid[..., None], output, jnp.zeros_like(output))
        return output


class MeasureNormalizedConvND(_AbstractMeasureNormalizedConvND):
    """Public measure-normalized convolution layer."""


__all__ = ["MeasureNormalizedConvND"]
