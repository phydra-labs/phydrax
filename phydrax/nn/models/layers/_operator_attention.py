#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ...._strict import StrictModule
from ..core._operator import FunctionSamples, OperatorAxis
from ._linear import Linear
from ._measure_attention import (
    AttentionExecution,
    AttentionKernel,
    MeasureAwareAttention,
)


def _sample_values(
    values: Array,
    samples: FunctionSamples,
    channels: int,
    /,
) -> tuple[Array, tuple[int, ...]]:
    array = jnp.asarray(values)
    shape = samples.sample_shape
    ndim = len(shape)
    if not shape:
        raise ValueError("Operator attention requires a non-empty sample geometry.")
    has_channel_axis = (
        array.ndim > ndim
        and tuple(array.shape[-ndim - 1 : -1]) == shape
        and int(array.shape[-1]) == channels
    )
    if not has_channel_axis:
        if tuple(array.shape[-ndim:]) != shape or channels != 1:
            raise ValueError(
                f"Attention values do not contain sample shape {shape} with "
                f"{channels} channels; got {array.shape}."
            )
        array = array[..., None]
    case_shape = tuple(int(size) for size in array.shape[: -ndim - 1])
    return array.reshape(
        (prod(case_shape) if case_shape else 1, prod(shape), channels)
    ), case_shape


_AttentionCore = MeasureAwareAttention


class OperatorAttention(StrictModule):
    """Quadrature-aware self/cross attention between function sample sets."""

    core: _AttentionCore
    source_channels: int
    query_channels: int
    out_channels: int

    def __init__(
        self,
        *,
        source_channels: int,
        query_channels: int | None = None,
        out_channels: int | None = None,
        num_heads: int = 4,
        head_dim: int = 16,
        kernel: AttentionKernel = "softmax",
        execution: AttentionExecution = "auto",
        block_size: int = 256,
        accumulation_dtype: str = "input",
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.source_channels = int(source_channels)
        self.query_channels = (
            self.source_channels if query_channels is None else int(query_channels)
        )
        self.out_channels = (
            self.query_channels if out_channels is None else int(out_channels)
        )
        self.core = _AttentionCore(
            source_channels=self.source_channels,
            query_channels=self.query_channels,
            out_channels=self.out_channels,
            num_heads=num_heads,
            head_dim=head_dim,
            kernel=kernel,
            execution=execution,
            block_size=block_size,
            accumulation_dtype=accumulation_dtype,
            key=key,
        )

    def cross(
        self,
        source_values: Array,
        query_values: Array,
        source: FunctionSamples,
        query: FunctionSamples,
        /,
    ) -> Array:
        source_flat, source_cases = _sample_values(
            source_values, source, self.source_channels
        )
        query_flat, query_cases = _sample_values(query_values, query, self.query_channels)
        if source_cases != query_cases:
            raise ValueError("Source and query attention case shapes must match.")
        weights = source.weights(case_shape=source_cases).reshape(source_flat.shape[:2])
        output = self.core(source_flat, query_flat, weights)
        query_mask = query.mask_array(case_shape=query_cases).reshape(
            query_flat.shape[:2] + (1,)
        )
        output = output * query_mask
        return output.reshape(query_cases + query.sample_shape + (self.out_channels,))

    def __call__(
        self,
        values: Array,
        samples: FunctionSamples,
        /,
    ) -> Array:
        if self.source_channels != self.query_channels:
            raise ValueError("Self-attention requires equal source and query channels.")
        return self.cross(values, values, samples, samples)


class SliceAttention(StrictModule):
    """Transolver-style learned physical slices with latent self-attention."""

    assignment: Linear
    attention: _AttentionCore
    projection: Linear
    channels: int
    out_channels: int
    num_slices: int

    def __init__(
        self,
        *,
        channels: int,
        num_slices: int,
        out_channels: int | None = None,
        num_heads: int = 4,
        head_dim: int = 16,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.channels = int(channels)
        self.out_channels = self.channels if out_channels is None else int(out_channels)
        self.num_slices = int(num_slices)
        if self.channels <= 0 or self.out_channels <= 0 or self.num_slices <= 0:
            raise ValueError("channels, out_channels, and num_slices must be positive.")
        assignment_key, attention_key, projection_key = jr.split(key, 3)
        self.assignment = Linear(
            in_size=self.channels,
            out_size=self.num_slices,
            activation=None,
            key=assignment_key,
        )
        self.attention = _AttentionCore(
            source_channels=self.channels,
            query_channels=self.channels,
            out_channels=self.channels,
            num_heads=num_heads,
            head_dim=head_dim,
            key=attention_key,
        )
        self.projection = Linear(
            in_size=self.channels,
            out_size=self.out_channels,
            activation=None,
            key=projection_key,
        )

    def __call__(
        self,
        values: Array,
        samples: FunctionSamples,
        /,
    ) -> Array:
        flattened, case_shape = _sample_values(values, samples, self.channels)
        logits = self.assignment(flattened)
        measure = samples.weights(case_shape=case_shape).reshape(
            (flattened.shape[0], flattened.shape[1], 1)
        )
        measure_logits = jnp.where(
            measure > 0.0,
            jnp.log(jnp.maximum(measure, jnp.finfo(logits.dtype).tiny)),
            jnp.asarray(-1e30, dtype=logits.dtype),
        )
        point_to_slice = jnn.softmax(logits + measure_logits, axis=1)
        tokens = jnp.einsum("bns,bnc->bsc", point_to_slice, flattened)
        tokens = self.attention(
            tokens,
            tokens,
            jnp.ones((flattened.shape[0], self.num_slices), dtype=float),
        )
        slice_to_point = jnn.softmax(logits, axis=-1)
        decoded = jnp.einsum("bns,bsc->bnc", slice_to_point, tokens)
        output = self.projection(decoded)
        output = output * samples.mask_array(case_shape=case_shape).reshape(
            flattened.shape[:2] + (1,)
        )
        return output.reshape(case_shape + samples.sample_shape + (self.out_channels,))


class CodomainAttention(StrictModule):
    """Self-attention over a variable set of physical fields at every sample."""

    core: _AttentionCore
    channels: int
    out_channels: int

    def __init__(
        self,
        *,
        channels: int,
        out_channels: int | None = None,
        num_heads: int = 4,
        head_dim: int = 16,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.channels = int(channels)
        self.out_channels = self.channels if out_channels is None else int(out_channels)
        self.core = _AttentionCore(
            source_channels=self.channels,
            query_channels=self.channels,
            out_channels=self.out_channels,
            num_heads=num_heads,
            head_dim=head_dim,
            key=key,
        )

    def __call__(self, values: Array, field_mask: Array | None = None, /) -> Array:
        array = jnp.asarray(values)
        if array.ndim < 2 or int(array.shape[-1]) != self.channels:
            raise ValueError("CodomainAttention expects (..., fields, channels).")
        field_count = int(array.shape[-2])
        leading = tuple(int(size) for size in array.shape[:-2])
        flattened = array.reshape(
            (prod(leading) if leading else 1, field_count, self.channels)
        )
        if field_mask is None:
            flattened_mask = jnp.ones(flattened.shape[:2], dtype=bool)
        else:
            mask = jnp.asarray(field_mask, dtype=bool)
            if mask.shape == (field_count,):
                flattened_mask = jnp.broadcast_to(mask, flattened.shape[:2])
            elif mask.shape == leading + (field_count,):
                flattened_mask = mask.reshape(flattened.shape[:2])
            else:
                raise ValueError(
                    "Codomain field_mask must have shape (fields,) or "
                    "values.shape[:-1]."
                )
        output = self.core(
            flattened,
            flattened,
            flattened_mask.astype(array.dtype),
            source_mask=flattened_mask,
            query_mask=flattened_mask,
        )
        return output.reshape(leading + (field_count, self.out_channels))


class AxialOperatorAttention(StrictModule):
    """Factorized quadrature-aware attention along tensor-product axes."""

    core: _AttentionCore
    channels: int

    def __init__(
        self,
        *,
        channels: int,
        num_heads: int = 4,
        head_dim: int = 16,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.channels = int(channels)
        self.core = _AttentionCore(
            source_channels=self.channels,
            query_channels=self.channels,
            out_channels=self.channels,
            num_heads=num_heads,
            head_dim=head_dim,
            key=key,
        )

    def __call__(self, values: Array, axes: tuple[OperatorAxis, ...], /) -> Array:
        output = jnp.asarray(values)
        ndim = len(axes)
        if ndim == 0 or output.ndim < ndim + 1:
            raise ValueError("Axial attention requires tensor-product spatial axes.")
        if int(output.shape[-1]) != self.channels:
            raise ValueError(f"Expected {self.channels} channels.")
        spatial_start = output.ndim - ndim - 1
        if tuple(int(size) for size in output.shape[spatial_start:-1]) != tuple(
            axis.size for axis in axes
        ):
            raise ValueError("Axial attention values do not match OperatorAxis sizes.")

        for index, axis in enumerate(axes):
            array_axis = spatial_start + index
            moved = jnp.moveaxis(output, array_axis, -2)
            leading = moved.shape[:-2]
            flattened = moved.reshape((-1, axis.size, self.channels))
            weights = (
                jnp.ones((axis.size,), dtype=float)
                if axis.quadrature_weights is None
                else axis.quadrature_weights
            )
            attended = self.core(flattened, flattened, weights)
            moved = attended.reshape(leading + (axis.size, self.channels))
            output = jnp.moveaxis(moved, -2, array_axis)
        return output


__all__ = [
    "AxialOperatorAttention",
    "CodomainAttention",
    "OperatorAttention",
    "SliceAttention",
]
