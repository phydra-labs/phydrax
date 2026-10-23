#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from functools import partial
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.typing import DTypeLike

import phydrax.ein as ein

from ..._doc import DOC_KEY0
from ..._dtype_names import real_precision_dtype_name
from ..._strict import StrictModule
from ..._trainable import ParameterOwner
from ...typing import parse, PRNGKey
from ._linear import Linear


AttentionKernel: TypeAlias = Literal["softmax", "kernel_linear", "galerkin", "identity"]
AttentionExecution: TypeAlias = Literal["auto", "dense", "xla", "cudnn", "blockwise"]


def _attention_scale(query: Array, /) -> float:
    return float(query.shape[-1]) ** -0.5


def _fill(value: float, like: Array, /) -> Array:
    """Scalar `jnp.where` fill typed like `like`, so no wider constant is traced."""
    return jnp.asarray(value, dtype=like.dtype)


def _softmax_measure(
    measure: Array,
    /,
) -> tuple[Array, Array, Array]:
    positive = measure > 0.0
    any_source = jnp.any(positive, axis=-1)
    first_source = jnp.arange(measure.shape[-1]) == 0
    safe_mask = positive | ((~any_source)[:, None] & first_source[None, :])
    log_measure = jnp.log(jnp.where(positive, measure, _fill(1.0, measure)))
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
            jnp.asarray(source_mask, dtype=jnp.bool_),
            (batch, source_count),
        ).astype(dtype)
    return jnp.where(
        (measure > 0.0) & jnp.isfinite(measure),
        measure,
        _fill(0.0, measure),
    )


def _dense_softmax_attention(
    query: Array,
    key: Array,
    value: Array,
    measure: Array,
    /,
) -> Array:
    safe_mask, log_measure, any_source = _softmax_measure(measure)
    logits = ein.contract("bqhd,bshd->bhqs", query, key) * _attention_scale(query)
    logits = jnp.where(
        safe_mask[:, None, None, :],
        logits + log_measure[:, None, None, :],
        _fill(-jnp.inf, logits),
    )
    # Differentiate through the max shift rather than using a softmax custom
    # JVP or a stop-gradient shift: the dominant shifted logit is then exactly
    # constant, so saturated attention keeps accurate higher derivatives in
    # float32 instead of cancelling O(|d logits|^2) terms.
    shifted = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
    attention = shifted / jnp.sum(shifted, axis=-1, keepdims=True)
    attended = ein.contract("bhqs,bshd->bqhd", attention, value)
    return jnp.where(any_source[:, None, None, None], attended, _fill(0.0, attended))


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
    source_count = key.shape[1]
    if block_size >= source_count:
        return _dense_softmax_attention(query, key, value, measure)
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
    tile = min(block_size, source_count)
    tile_indices = jnp.arange(tile, dtype=jnp.int32)

    def step(
        index: Array, carry: tuple[Array, Array, Array]
    ) -> tuple[Array, Array, Array]:
        running_max, running_sum, accumulator = carry
        requested_start = index * tile
        start = jnp.minimum(requested_start, source_count - tile)
        block_key = jax.lax.dynamic_slice_in_dim(key, start, tile, axis=1)
        block_value = jax.lax.dynamic_slice_in_dim(value, start, tile, axis=1)
        block_measure = jax.lax.dynamic_slice_in_dim(measure, start, tile, axis=1)
        # The final clamped slice overlaps its predecessor; exclude that prefix.
        valid = (block_measure > 0.0) & (
            (start + tile_indices >= requested_start)[None, :]
        )
        logits = ein.contract("bhqd,bshd->bhqs", q, block_key) * scale
        logits = jnp.where(
            valid[:, None, None, :],
            logits
            + jnp.log(jnp.where(valid, block_measure, _fill(1.0, block_measure)))[
                :, None, None, :
            ],
            _fill(-jnp.inf, logits),
        )
        block_max = jnp.max(logits, axis=-1)
        updated_max = jnp.maximum(running_max, block_max)
        # Guard both sides of each infinite difference: a NaN in an unselected
        # branch is harmless to the value and first derivative but poisons
        # higher derivatives of empty-support rows through 0 * NaN.
        finite_max = jnp.isfinite(updated_max)
        rescale = jnp.isfinite(running_max) & finite_max
        previous_scale = jnp.where(
            rescale,
            jnp.exp(
                jnp.where(rescale, running_max - updated_max, _fill(0.0, running_max))
            ),
            _fill(0.0, running_max),
        )
        safe_max = jnp.where(finite_max, updated_max, _fill(0.0, updated_max))
        shifted = jnp.where(
            valid[:, None, None, :] & finite_max[..., None],
            logits - safe_max[..., None],
            _fill(-jnp.inf, logits),
        )
        exponential = jnp.exp(shifted)
        running_sum = running_sum * previous_scale + jnp.sum(exponential, axis=-1)
        accumulator = accumulator * previous_scale[..., None] + ein.contract(
            "bhqs,bshd->bhqd", exponential, block_value
        )
        return updated_max, running_sum, accumulator

    running_max, running_sum, accumulator = jax.lax.fori_loop(
        0,
        (source_count + tile - 1) // tile,
        step,
        (running_max, running_sum, accumulator),
    )
    supported = running_sum[..., None] > 0.0
    # A unit denominator on empty rows keeps every derivative order finite.
    denominator = jnp.where(
        supported,
        jnp.maximum(running_sum[..., None], jnp.finfo(accumulator.dtype).tiny),
        _fill(1.0, running_sum),
    )
    normalized = jnp.where(supported, accumulator / denominator, _fill(0.0, accumulator))
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
    return jnp.where(any_source[:, None, None, None], result, _fill(0.0, result))


@partial(jax.custom_jvp, nondiff_argnums=(4,))
def _xla_softmax_attention(
    query: Array,
    key: Array,
    value: Array,
    measure: Array,
    reference_block_size: int,
    /,
) -> Array:
    """Fused XLA primal whose derivatives follow the bounded reference route.

    Upstream XLA attention differentiates a float32 softmax custom JVP whose
    saturated higher derivatives cancel ``O(|d logits|^2)`` terms. Tangents
    instead come from the source-blockwise reference (direct dense when the
    tile covers every source), so every derivative order keeps the reference
    accuracy, measure and mask semantics, and workspace bound.
    """
    return _fused_softmax_attention(query, key, value, measure, implementation="xla")


@_xla_softmax_attention.defjvp
def _xla_softmax_attention_jvp(
    reference_block_size: int,
    primals: tuple[Array, Array, Array, Array],
    tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    output = _xla_softmax_attention(*primals, reference_block_size)
    _, tangent = jax.jvp(
        partial(_blockwise_softmax_attention, block_size=reference_block_size),
        primals,
        tangents,
    )
    return output, tangent


def _positive_feature(x: Array, /) -> Array:
    """``elu(x) + 1`` with typed constants.

    Same operations as `jax.nn.elu`, whose jitted body traces its scalar fill
    at the widest float dtype.
    """
    positive = x > 0.0
    return jnp.where(positive, x, jnp.expm1(jnp.where(positive, _fill(0.0, x), x))) + 1.0


def _linear_attention(
    query: Array,
    key: Array,
    value: Array,
    measure: Array,
    /,
    *,
    normalize: bool,
) -> Array:
    q = _positive_feature(query)
    k = _positive_feature(key)
    weighted_key = k * measure[:, :, None, None]
    covariance = ein.contract("bshd,bshv->bhdv", weighted_key, value)
    attended = ein.contract("bqhd,bhdv->bqhv", q, covariance)
    if normalize:
        normalizer = ein.contract("bqhd,bhd->bqh", q, jnp.sum(weighted_key, axis=1))
        positive_normalizer = normalizer > 0.0
        attended = attended / jnp.where(
            positive_normalizer[..., None],
            normalizer[..., None],
            _fill(1.0, normalizer),
        )
        attended = jnp.where(
            positive_normalizer[..., None], attended, _fill(0.0, attended)
        )
    any_source = jnp.any(measure > 0.0, axis=-1)
    return jnp.where(any_source[:, None, None, None], attended, _fill(0.0, attended))


class MeasureAwareAttention(StrictModule, ParameterOwner):
    """Projected attention with physical source measure and scalable execution.

    `dtype` selects the real floating dtype of the query, key, value and output
    projection parameters under the active JAX canonicalization policy. It is a
    construction choice, not an input cast, and is independent of
    `accumulation_dtype`. `None` keeps JAX's default floating dtype.
    """

    query: Linear
    key: Linear
    value: Linear
    output: Linear
    num_heads: int
    head_dim: int
    out_channels: int
    kernel: AttentionKernel
    execution: AttentionExecution
    block_size: int | None
    attention_workspace_bytes: int = eqx.field(static=True)
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
        block_size: int | None = None,
        attention_workspace_bytes: int = 8 * 1024 * 1024,
        accumulation_dtype: str = "input",
        dtype: DTypeLike | None = None,
        key: PRNGKey = DOC_KEY0,
    ) -> None:
        # Resolve an explicit parameter dtype before any allocation.
        param_dtype = None if dtype is None else real_precision_dtype_name(dtype)
        if isinstance(attention_workspace_bytes, bool) or not isinstance(
            attention_workspace_bytes, int
        ):
            raise TypeError("attention_workspace_bytes must be an integer.")
        if attention_workspace_bytes <= 0:
            raise ValueError("attention_workspace_bytes must be positive.")
        resolved_block_size = None if block_size is None else int(block_size)
        if resolved_block_size is not None and resolved_block_size <= 0:
            raise ValueError("block_size must be positive.")
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.out_channels = int(out_channels)
        self.block_size = resolved_block_size
        self.attention_workspace_bytes = attention_workspace_bytes
        self.accumulation_dtype = str(accumulation_dtype)
        source_channels = int(source_channels)
        query_channels = int(query_channels)
        if source_channels <= 0 or query_channels <= 0:
            raise ValueError("source_channels and query_channels must be positive.")
        self.kernel = parse(kernel, AttentionKernel, "kernel")
        self.execution = parse(execution, AttentionExecution, "execution")
        if self.num_heads <= 0 or self.head_dim <= 0 or self.out_channels <= 0:
            raise ValueError("num_heads, head_dim, and out_channels must be positive.")
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
            dtype=param_dtype,
            key=keys[0],
        )
        self.key = Linear(
            in_size=source_channels,
            out_size=hidden,
            activation=None,
            dtype=param_dtype,
            key=keys[1],
        )
        self.value = Linear(
            in_size=source_channels,
            out_size=hidden,
            activation=None,
            dtype=param_dtype,
            key=keys[2],
        )
        self.output = Linear(
            in_size=hidden,
            out_size=self.out_channels,
            activation=None,
            dtype=param_dtype,
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
        if k.shape[1] == 0:
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
            batch=k.shape[0],
            source_count=k.shape[1],
            dtype=dtype,
        )
        source_support = measure > 0.0
        output_mask = jnp.broadcast_to(
            jnp.any(source_support, axis=-1)[:, None],
            q.shape[:2],
        )
        match self.kernel:
            case "identity":
                if q.shape[1] != k.shape[1]:
                    raise ValueError(
                        "Identity attention requires equal source/query counts."
                    )
                attended = jnp.where(
                    source_support[:, :, None, None],
                    v_acc,
                    _fill(0.0, v_acc),
                )
                output_mask = source_support
            case "kernel_linear":
                attended = _linear_attention(
                    q_acc,
                    k_acc,
                    v_acc,
                    measure,
                    normalize=True,
                )
            case "galerkin":
                attended = _linear_attention(
                    q_acc,
                    k_acc,
                    v_acc,
                    measure,
                    normalize=False,
                )
            case "softmax":
                # Bound logical scores, with one source column as the floor.
                column_bytes = (
                    q_acc.shape[0] * q_acc.shape[1] * self.num_heads * dtype.itemsize
                )
                tile = (
                    self.block_size
                    if self.block_size is not None
                    else max(1, self.attention_workspace_bytes // max(1, column_bytes))
                )
                match self.execution:
                    case "cudnn":
                        attended = _fused_softmax_attention(
                            q_acc,
                            k_acc,
                            v_acc,
                            measure,
                            implementation="cudnn",
                        )
                    case "xla" | "auto" if self.execution == "xla" or (
                        q_acc.dtype == jnp.float32 and self.block_size is None
                    ):
                        attended = _xla_softmax_attention(
                            q_acc, k_acc, v_acc, measure, tile
                        )
                    case "auto" | "blockwise":
                        attended = _blockwise_softmax_attention(
                            q_acc, k_acc, v_acc, measure, block_size=tile
                        )
                    case "dense":
                        attended = _dense_softmax_attention(q_acc, k_acc, v_acc, measure)
                    case _:
                        assert_never(self.execution)
            case _:
                assert_never(self.kernel)
        if query_mask is not None:
            mask = jnp.broadcast_to(
                jnp.asarray(query_mask, dtype=jnp.bool_),
                q.shape[:2],
            )
            output_mask = output_mask & mask
        flattened = attended.astype(q.dtype).reshape(
            attended.shape[:-2] + (self.num_heads * self.head_dim,)
        )
        output = self.output(flattened)
        return jnp.where(output_mask[..., None], output, _fill(0.0, output))

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
