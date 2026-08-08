#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from ._linear import Linear


AttentionKernel = Literal["softmax", "kernel_linear", "galerkin", "identity"]
AttentionExecution = Literal["auto", "dense", "xla", "cudnn", "blockwise"]


_ATTENTION_KERNELS = ("softmax", "kernel_linear", "galerkin", "identity")
_ATTENTION_EXECUTIONS = ("auto", "dense", "xla", "cudnn", "blockwise")


def _attention_scale(query: Array, /) -> float:
    return float(query.shape[-1]) ** -0.5


def _softmax_measure(
    measure: Array,
    /,
) -> tuple[Array, Array, Array]:
    positive = measure > 0.0
    any_source = jnp.any(positive, axis=-1)
    first_source = jnp.arange(measure.shape[-1]) == 0
    safe_mask = positive | ((~any_source)[:, None] & first_source[None, :])
    log_measure = jnp.log(jnp.where(positive, measure, 1.0))
    return safe_mask, log_measure, any_source


def _effective_measure(
    source_weights: Array,
    source_mask: Array | None,
    /,
    *,
    batch: int,
    source_count: int,
    dtype: jnp.dtype,
) -> Array:
    measure = jnp.broadcast_to(
        jnp.asarray(source_weights, dtype=dtype),
        (batch, source_count),
    )
    if source_mask is not None:
        measure = measure * jnp.broadcast_to(
            jnp.asarray(source_mask, dtype=bool),
            (batch, source_count),
        ).astype(dtype)
    return jnp.where(
        (measure > 0.0) & jnp.isfinite(measure),
        measure,
        0.0,
    )


def _dense_softmax_attention(
    query: Array,
    key: Array,
    value: Array,
    measure: Array,
    /,
) -> Array:
    safe_mask, log_measure, any_source = _softmax_measure(measure)
    logits = oe.contract("bqhd,bshd->bhqs", query, key) * _attention_scale(query)
    logits = jnp.where(
        safe_mask[:, None, None, :],
        logits + log_measure[:, None, None, :],
        -jnp.inf,
    )
    attention = jnn.softmax(logits, axis=-1)
    attended = oe.contract("bhqs,bshd->bqhd", attention, value)
    return jnp.where(any_source[:, None, None, None], attended, 0.0)


def _blockwise_softmax_attention(
    query: Array,
    key: Array,
    value: Array,
    measure: Array,
    /,
    *,
    block_size: int,
) -> Array:
    batch, query_count, heads, head_dim = query.shape
    source_count = int(key.shape[1])
    q = jnp.transpose(query, (0, 2, 1, 3))
    running_max = jnp.full(
        (batch, heads, query_count),
        -jnp.inf,
        dtype=query.dtype,
    )
    running_sum = jnp.zeros_like(running_max)
    accumulator = jnp.zeros(
        (batch, heads, query_count, head_dim),
        dtype=query.dtype,
    )
    scale = _attention_scale(query)
    for start in range(0, source_count, int(block_size)):
        stop = min(source_count, start + int(block_size))
        block_key = key[:, start:stop]
        block_value = value[:, start:stop]
        block_measure = measure[:, start:stop]
        valid = block_measure > 0.0
        logits = oe.contract("bhqd,bshd->bhqs", q, block_key) * scale
        logits = jnp.where(
            valid[:, None, None, :],
            logits + jnp.log(jnp.where(valid, block_measure, 1.0))[:, None, None, :],
            -jnp.inf,
        )
        block_max = jnp.max(logits, axis=-1)
        updated_max = jnp.maximum(running_max, block_max)
        previous_scale = jnp.where(
            jnp.isfinite(running_max) & jnp.isfinite(updated_max),
            jnp.exp(running_max - updated_max),
            0.0,
        )
        shifted = jnp.where(
            valid[:, None, None, :] & jnp.isfinite(updated_max[..., None]),
            logits - updated_max[..., None],
            -jnp.inf,
        )
        exponential = jnp.exp(shifted)
        running_sum = running_sum * previous_scale + jnp.sum(exponential, axis=-1)
        accumulator = accumulator * previous_scale[..., None] + oe.contract(
            "bhqs,bshd->bhqd", exponential, block_value
        )
        running_max = updated_max
    normalized = jnp.where(
        running_sum[..., None] > 0.0,
        accumulator
        / jnp.maximum(running_sum[..., None], jnp.finfo(accumulator.dtype).tiny),
        0.0,
    )
    return jnp.transpose(normalized, (0, 2, 1, 3))


def _fused_softmax_attention(
    query: Array,
    key: Array,
    value: Array,
    measure: Array,
    /,
    *,
    implementation: Literal["xla", "cudnn"],
) -> Array:
    safe_mask, log_measure, any_source = _softmax_measure(measure)
    bias = log_measure[:, None, None, :]
    mask = safe_mask[:, None, None, :]
    if implementation == "cudnn":
        fused_shape = (
            query.shape[0],
            1,
            query.shape[1],
            key.shape[1],
        )
        bias = jnp.broadcast_to(bias, fused_shape)
        mask = jnp.broadcast_to(mask, fused_shape)
    result = jnn.dot_product_attention(
        query,
        key,
        value,
        bias=bias,
        mask=mask,
        scale=_attention_scale(query),
        implementation=implementation,
    )
    return jnp.where(any_source[:, None, None, None], result, 0.0)


def _linear_attention(
    query: Array,
    key: Array,
    value: Array,
    measure: Array,
    /,
    *,
    normalize: bool,
) -> Array:
    q = jnn.elu(query) + 1.0
    k = jnn.elu(key) + 1.0
    weighted_key = k * measure[:, :, None, None]
    covariance = oe.contract("bshd,bshv->bhdv", weighted_key, value)
    attended = oe.contract("bqhd,bhdv->bqhv", q, covariance)
    if normalize:
        normalizer = oe.contract("bqhd,bhd->bqh", q, jnp.sum(weighted_key, axis=1))
        positive_normalizer = normalizer > 0.0
        attended = attended / jnp.where(
            positive_normalizer[..., None],
            normalizer[..., None],
            1.0,
        )
        attended = jnp.where(positive_normalizer[..., None], attended, 0.0)
    any_source = jnp.any(measure > 0.0, axis=-1)
    return jnp.where(any_source[:, None, None, None], attended, 0.0)


class MeasureAwareAttention(StrictModule):
    """Projected attention with physical source measure and scalable execution."""

    query: Linear
    key: Linear
    value: Linear
    output: Linear
    num_heads: int
    head_dim: int
    out_channels: int
    kernel: AttentionKernel
    execution: AttentionExecution
    block_size: int
    accumulation_dtype: str

    def __init__(
        self,
        *,
        source_channels: int,
        query_channels: int,
        out_channels: int,
        num_heads: int,
        head_dim: int,
        kernel: AttentionKernel = "softmax",
        execution: AttentionExecution = "auto",
        block_size: int = 256,
        accumulation_dtype: str = "input",
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.out_channels = int(out_channels)
        self.kernel = kernel
        self.execution = execution
        self.block_size = int(block_size)
        self.accumulation_dtype = str(accumulation_dtype)
        source_channels = int(source_channels)
        query_channels = int(query_channels)
        if source_channels <= 0 or query_channels <= 0:
            raise ValueError("source_channels and query_channels must be positive.")
        if self.kernel not in _ATTENTION_KERNELS:
            raise ValueError(
                f"kernel must be one of {_ATTENTION_KERNELS}; got {self.kernel!r}."
            )
        if self.execution not in _ATTENTION_EXECUTIONS:
            raise ValueError(
                "execution must be one of "
                f"{_ATTENTION_EXECUTIONS}; got {self.execution!r}."
            )
        if self.num_heads <= 0 or self.head_dim <= 0 or self.out_channels <= 0:
            raise ValueError("num_heads, head_dim, and out_channels must be positive.")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive.")
        if self.kernel != "softmax" and self.execution not in ("auto", "dense"):
            raise ValueError(
                "Non-softmax attention kernels accept only 'auto' or 'dense' execution."
            )
        if self.accumulation_dtype not in ("input", "float32", "float64"):
            raise ValueError(
                "accumulation_dtype must be 'input', 'float32', or 'float64'."
            )
        keys = jr.split(key, 4)
        hidden = self.num_heads * self.head_dim
        self.query = Linear(
            in_size=query_channels,
            out_size=hidden,
            activation=None,
            key=keys[0],
        )
        self.key = Linear(
            in_size=source_channels,
            out_size=hidden,
            activation=None,
            key=keys[1],
        )
        self.value = Linear(
            in_size=source_channels,
            out_size=hidden,
            activation=None,
            key=keys[2],
        )
        self.output = Linear(
            in_size=hidden,
            out_size=self.out_channels,
            activation=None,
            key=keys[3],
        )

    def project_query(self, query: Array, /) -> Array:
        projected = self.query(query)
        return projected.reshape(query.shape[:-1] + (self.num_heads, self.head_dim))

    def project_source(self, source: Array, /) -> tuple[Array, Array]:
        shape = source.shape[:-1] + (self.num_heads, self.head_dim)
        return self.key(source).reshape(shape), self.value(source).reshape(shape)

    def attend_projected(
        self,
        projected_query: Array,
        projected_key: Array,
        projected_value: Array,
        source_weights: Array,
        /,
        *,
        source_mask: Array | None = None,
        query_mask: Array | None = None,
    ) -> Array:
        q = jnp.asarray(projected_query)
        k = jnp.asarray(projected_key)
        v = jnp.asarray(projected_value)
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("Projected attention arrays must have rank four.")
        if k.shape != v.shape or q.shape[0] != k.shape[0]:
            raise ValueError("Projected query/key/value batch and source shapes differ.")
        expected_heads = (self.num_heads, self.head_dim)
        if q.shape[-2:] != expected_heads:
            raise ValueError("Projected query has incompatible head dimensions.")
        if k.shape[-2:] != expected_heads:
            raise ValueError("Projected key/value have incompatible head dimensions.")
        if int(k.shape[1]) == 0:
            raise ValueError("Projected attention requires at least one source.")
        if not all(jnp.issubdtype(array.dtype, jnp.floating) for array in (q, k, v)):
            raise ValueError("Projected attention arrays must have floating dtype.")
        dtype = (
            jnp.result_type(q.dtype, k.dtype, v.dtype)
            if self.accumulation_dtype == "input"
            else jnp.dtype(self.accumulation_dtype)
        )
        q_acc = q.astype(dtype)
        k_acc = k.astype(dtype)
        v_acc = v.astype(dtype)
        measure = _effective_measure(
            source_weights,
            source_mask,
            batch=int(k.shape[0]),
            source_count=int(k.shape[1]),
            dtype=dtype,
        )
        source_support = measure > 0.0
        output_mask = jnp.broadcast_to(
            jnp.any(source_support, axis=-1)[:, None],
            q.shape[:2],
        )
        if self.kernel == "identity":
            if q.shape[1] != k.shape[1]:
                raise ValueError("Identity attention requires equal source/query counts.")
            attended = jnp.where(
                source_support[:, :, None, None],
                v_acc,
                0.0,
            )
            output_mask = source_support
        elif self.kernel == "kernel_linear":
            attended = _linear_attention(
                q_acc,
                k_acc,
                v_acc,
                measure,
                normalize=True,
            )
        elif self.kernel == "galerkin":
            attended = _linear_attention(
                q_acc,
                k_acc,
                v_acc,
                measure,
                normalize=False,
            )
        elif self.execution == "blockwise":
            attended = _blockwise_softmax_attention(
                q_acc,
                k_acc,
                v_acc,
                measure,
                block_size=self.block_size,
            )
        elif self.execution == "cudnn":
            attended = _fused_softmax_attention(
                q_acc,
                k_acc,
                v_acc,
                measure,
                implementation="cudnn",
            )
        elif self.execution == "xla" or (
            self.execution == "auto" and q_acc.dtype == jnp.float32
        ):
            attended = _fused_softmax_attention(
                q_acc,
                k_acc,
                v_acc,
                measure,
                implementation="xla",
            )
        elif self.execution in ("auto", "dense"):
            attended = _dense_softmax_attention(q_acc, k_acc, v_acc, measure)
        else:
            raise ValueError(f"Unknown attention execution {self.execution!r}.")
        if query_mask is not None:
            mask = jnp.broadcast_to(
                jnp.asarray(query_mask, dtype=bool),
                q.shape[:2],
            )
            output_mask = output_mask & mask
        flattened = attended.astype(q.dtype).reshape(
            attended.shape[:-2] + (self.num_heads * self.head_dim,)
        )
        output = self.output(flattened)
        return jnp.where(output_mask[..., None], output, 0.0)

    def __call__(
        self,
        source: Array,
        query: Array,
        source_weights: Array,
        /,
        *,
        source_mask: Array | None = None,
        query_mask: Array | None = None,
    ) -> Array:
        projected_query = self.project_query(query)
        projected_key, projected_value = self.project_source(source)
        return self.attend_projected(
            projected_query,
            projected_key,
            projected_value,
            source_weights,
            source_mask=source_mask,
            query_mask=query_mask,
        )


__all__ = [
    "AttentionExecution",
    "AttentionKernel",
    "MeasureAwareAttention",
]
