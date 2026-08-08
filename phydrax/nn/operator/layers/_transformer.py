#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, log, prod

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ..._keys import EvalKey, split_eval_key
from ...layers._dropout import Dropout
from ...layers._linear import Linear


def _apply_feature_norm(norm: eqx.nn.RMSNorm, values: Array, /) -> Array:
    values = jnp.asarray(values)
    flattened = values.reshape((-1, int(values.shape[-1])))
    return jax.vmap(norm)(flattened).reshape(values.shape)


def _patchify_grid(
    values: Array,
    spatial_shape: tuple[int, ...],
    patch_shape: tuple[int, ...],
    /,
) -> Array:
    values = jnp.asarray(values)
    spatial_ndim = len(spatial_shape)
    if values.ndim < spatial_ndim + 1:
        raise ValueError("Patch values must contain spatial and channel axes.")
    if tuple(int(size) for size in values.shape[-spatial_ndim - 1 : -1]) != spatial_shape:
        raise ValueError("Patch values do not match the configured latent shape.")
    case_shape = tuple(int(size) for size in values.shape[: -spatial_ndim - 1])
    channels = int(values.shape[-1])
    patch_grid = tuple(
        size // patch for size, patch in zip(spatial_shape, patch_shape, strict=True)
    )
    interleaved = tuple(
        size
        for grid_size, patch_size in zip(patch_grid, patch_shape, strict=True)
        for size in (grid_size, patch_size)
    )
    reshaped = values.reshape(case_shape + interleaved + (channels,))
    case_ndim = len(case_shape)
    spatial_grid_axes = tuple(case_ndim + 2 * index for index in range(spatial_ndim))
    patch_axes = tuple(case_ndim + 2 * index + 1 for index in range(spatial_ndim))
    permutation = (
        tuple(range(case_ndim)) + spatial_grid_axes + patch_axes + (reshaped.ndim - 1,)
    )
    arranged = jnp.transpose(reshaped, permutation)
    return arranged.reshape(case_shape + (prod(patch_grid), prod(patch_shape) * channels))


def _unpatchify_grid(
    tokens: Array,
    case_shape: tuple[int, ...],
    spatial_shape: tuple[int, ...],
    patch_shape: tuple[int, ...],
    channels: int,
    /,
) -> Array:
    tokens = jnp.asarray(tokens)
    spatial_ndim = len(spatial_shape)
    patch_grid = tuple(
        size // patch for size, patch in zip(spatial_shape, patch_shape, strict=True)
    )
    expected = case_shape + (prod(patch_grid), prod(patch_shape) * int(channels))
    if tokens.shape != expected:
        raise ValueError(f"Patch token shape must be {expected}; got {tokens.shape}.")
    arranged = tokens.reshape(case_shape + patch_grid + patch_shape + (int(channels),))
    case_ndim = len(case_shape)
    permutation = (
        tuple(range(case_ndim))
        + tuple(
            axis
            for index in range(spatial_ndim)
            for axis in (case_ndim + index, case_ndim + spatial_ndim + index)
        )
        + (arranged.ndim - 1,)
    )
    interleaved = jnp.transpose(arranged, permutation)
    return interleaved.reshape(case_shape + spatial_shape + (int(channels),))


def _sinusoidal_positions(positions: Array, width: int, /) -> Array:
    coord_dim = int(positions.shape[-1])
    frequency_count = max(1, ceil(int(width) / (2 * coord_dim)))
    frequencies = jnp.exp(
        -log(10000.0)
        * jnp.arange(frequency_count, dtype=positions.dtype)
        / float(frequency_count)
    )
    phase = positions[..., None] * frequencies
    embedding = jnp.concatenate((jnp.sin(phase), jnp.cos(phase)), axis=-1)
    embedding = embedding.reshape(positions.shape[:-1] + (-1,))
    return embedding[..., : int(width)]


class _SelfAttention(eqx.Module):
    query: Linear
    key: Linear
    value: Linear
    output: Linear
    dropout: Dropout
    heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(
        self,
        width: int,
        heads: int,
        /,
        *,
        dropout: float,
        key: Key[Array, ""],
    ):
        if int(width) % int(heads) != 0:
            raise ValueError("Transformer width must be divisible by heads.")
        keys = jr.split(key, 4)
        self.query = Linear(
            in_size=width,
            out_size=width,
            activation=None,
            use_bias=False,
            key=keys[0],
        )
        self.key = Linear(
            in_size=width,
            out_size=width,
            activation=None,
            use_bias=False,
            key=keys[1],
        )
        self.value = Linear(
            in_size=width,
            out_size=width,
            activation=None,
            use_bias=False,
            key=keys[2],
        )
        self.output = Linear(
            in_size=width,
            out_size=width,
            activation=None,
            use_bias=False,
            key=keys[3],
        )
        self.dropout = Dropout("scalar", p=float(dropout), mode="elementwise")
        self.heads = int(heads)
        self.head_dim = int(width) // int(heads)

    def __call__(
        self,
        values: Array,
        token_measure: Array,
        token_mask: Array,
        /,
        *,
        position_encoding: Array | None = None,
        key: EvalKey = None,
    ) -> Array:
        values = jnp.asarray(values)
        case_shape = tuple(int(size) for size in values.shape[:-2])
        token_count = int(values.shape[-2])
        width = self.heads * self.head_dim
        cases = prod(case_shape) if case_shape else 1
        flattened = values.reshape((cases, token_count, width))
        if position_encoding is None:
            query_key_values = flattened
        else:
            encoding = jnp.asarray(position_encoding, dtype=values.dtype)
            if encoding.shape != values.shape:
                raise ValueError("Position encoding must match transformer token values.")
            query_key_values = flattened + encoding.reshape(flattened.shape)
        q = self.query(query_key_values).reshape(
            (cases, token_count, self.heads, self.head_dim)
        )
        k = self.key(query_key_values).reshape(
            (cases, token_count, self.heads, self.head_dim)
        )
        v = self.value(flattened).reshape((cases, token_count, self.heads, self.head_dim))
        logits = oe.contract("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(float(self.head_dim))
        measure = jnp.asarray(token_measure, dtype=logits.dtype).reshape(
            (cases, token_count)
        )
        mask = jnp.asarray(token_mask, dtype=bool).reshape((cases, token_count))
        log_measure = jnp.where(
            mask,
            jnp.log(jnp.maximum(measure, jnp.finfo(logits.dtype).tiny)),
            jnp.asarray(-1e30, dtype=logits.dtype),
        )
        attention = jnn.softmax(logits + log_measure[:, None, None, :], axis=-1)
        attention = self.dropout(attention, key=key)
        attended = oe.contract("bhqk,bkhd->bqhd", attention, v).reshape(
            (cases, token_count, width)
        )
        output = self.output(attended)
        output = output * mask[..., None].astype(output.dtype)
        return output.reshape(values.shape)


class _SwiGLU(eqx.Module):
    gate: Linear
    value: Linear
    output: Linear
    dropout: Dropout

    def __init__(
        self,
        width: int,
        hidden_width: int,
        /,
        *,
        dropout: float,
        key: Key[Array, ""],
    ):
        keys = jr.split(key, 3)
        self.gate = Linear(
            in_size=width,
            out_size=hidden_width,
            activation=None,
            use_bias=False,
            key=keys[0],
        )
        self.value = Linear(
            in_size=width,
            out_size=hidden_width,
            activation=None,
            use_bias=False,
            key=keys[1],
        )
        self.output = Linear(
            in_size=hidden_width,
            out_size=width,
            activation=None,
            use_bias=False,
            key=keys[2],
        )
        self.dropout = Dropout("scalar", p=float(dropout), mode="elementwise")

    def __call__(self, values: Array, /, *, key: EvalKey = None) -> Array:
        output = self.output(jnn.silu(self.gate(values)) * self.value(values))
        return self.dropout(output, key=key)


class _TransformerBlock(eqx.Module):
    attention: _SelfAttention
    feed_forward: _SwiGLU
    attention_norm: eqx.nn.RMSNorm
    feed_forward_norm: eqx.nn.RMSNorm
    skip_projection: Linear | None

    def __init__(
        self,
        width: int,
        heads: int,
        hidden_width: int,
        /,
        *,
        attention_dropout: float,
        feed_forward_dropout: float,
        skip_connection: bool,
        norm_eps: float,
        key: Key[Array, ""],
    ):
        attention_key, feed_forward_key, skip_key = jr.split(key, 3)
        self.attention = _SelfAttention(
            width,
            heads,
            dropout=attention_dropout,
            key=attention_key,
        )
        self.feed_forward = _SwiGLU(
            width,
            hidden_width,
            dropout=feed_forward_dropout,
            key=feed_forward_key,
        )
        self.attention_norm = eqx.nn.RMSNorm(
            width,
            eps=float(norm_eps),
            use_bias=False,
        )
        self.feed_forward_norm = eqx.nn.RMSNorm(
            width,
            eps=float(norm_eps),
            use_bias=False,
        )
        self.skip_projection = (
            Linear(
                in_size=2 * width,
                out_size=width,
                activation=None,
                key=skip_key,
            )
            if skip_connection
            else None
        )

    def __call__(
        self,
        values: Array,
        token_measure: Array,
        token_mask: Array,
        /,
        *,
        position_encoding: Array | None = None,
        skip: Array | None = None,
        key: EvalKey = None,
    ) -> Array:
        if self.skip_projection is not None:
            if skip is None:
                raise ValueError("A decoder transformer block requires its long skip.")
            values = self.skip_projection(jnp.concatenate((values, skip), axis=-1))
        elif skip is not None:
            raise ValueError("This transformer block does not accept a long skip.")
        attention_key, feed_forward_key = split_eval_key(key, 2)
        attended = self.attention(
            _apply_feature_norm(self.attention_norm, values),
            token_measure,
            token_mask,
            position_encoding=position_encoding,
            key=attention_key,
        )
        values = values + attended
        values = values + self.feed_forward(
            _apply_feature_norm(self.feed_forward_norm, values),
            key=feed_forward_key,
        )
        return values * token_mask[..., None].astype(values.dtype)


class OperatorTransformerProcessor(eqx.Module):
    """Patchwise U-shaped operator transformer on a tensor latent grid.

    Patch tokens retain all cell channels, receive sinusoidal encodings of
    measure-weighted patch-center coordinates, and interact through
    quadrature-aware self-attention. The output is unpatchified exactly back to
    the original channels-last latent grid.
    """

    input_projection: Linear
    output_projection: Linear
    encoder_blocks: tuple[_TransformerBlock, ...]
    middle_block: _TransformerBlock | None
    decoder_blocks: tuple[_TransformerBlock, ...]
    latent_shape: tuple[int, ...] = eqx.field(static=True)
    patch_shape: tuple[int, ...] = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    model_width: int = eqx.field(static=True)
    coord_dim: int = eqx.field(static=True)

    def __init__(
        self,
        latent_shape: Sequence[int],
        channels: int,
        /,
        *,
        patch_shape: int | Sequence[int] = 2,
        model_width: int = 128,
        depth: int = 3,
        heads: int = 8,
        feed_forward_multiplier: float = 4.0,
        attention_dropout: float = 0.0,
        feed_forward_dropout: float = 0.0,
        long_range_skip: bool = True,
        norm_eps: float = 1e-6,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        shape = tuple(int(size) for size in latent_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("latent_shape must contain positive dimensions.")
        patches = (
            (int(patch_shape),) * len(shape)
            if isinstance(patch_shape, int)
            else tuple(int(size) for size in patch_shape)
        )
        if len(patches) != len(shape) or any(size <= 0 for size in patches):
            raise ValueError("patch_shape must contain one positive size per dimension.")
        if any(size % patch != 0 for size, patch in zip(shape, patches, strict=True)):
            raise ValueError(
                "Every latent dimension must be divisible by its patch size."
            )
        if int(channels) <= 0 or int(model_width) <= 0:
            raise ValueError("channels and model_width must be positive.")
        if int(depth) <= 0 or int(heads) <= 0:
            raise ValueError("depth and heads must be positive.")
        if int(model_width) % int(heads) != 0:
            raise ValueError("model_width must be divisible by heads.")
        hidden_width = round(float(feed_forward_multiplier) * int(model_width))
        if hidden_width <= 0:
            raise ValueError("feed_forward_multiplier must produce a positive width.")
        if float(norm_eps) <= 0.0:
            raise ValueError("norm_eps must be positive.")

        encoder_count = int(depth) // 2
        decoder_count = int(depth) // 2
        has_middle = int(depth) % 2 == 1
        keys = iter(jr.split(key, int(depth) + 2))
        token_width = prod(patches) * int(channels)
        self.input_projection = Linear(
            in_size=token_width,
            out_size=int(model_width),
            activation=None,
            key=next(keys),
        )
        self.output_projection = Linear(
            in_size=int(model_width),
            out_size=token_width,
            activation=None,
            key=next(keys),
        )
        self.encoder_blocks = tuple(
            _TransformerBlock(
                int(model_width),
                int(heads),
                hidden_width,
                attention_dropout=attention_dropout,
                feed_forward_dropout=feed_forward_dropout,
                skip_connection=False,
                norm_eps=norm_eps,
                key=next(keys),
            )
            for _ in range(encoder_count)
        )
        self.middle_block = (
            _TransformerBlock(
                int(model_width),
                int(heads),
                hidden_width,
                attention_dropout=attention_dropout,
                feed_forward_dropout=feed_forward_dropout,
                skip_connection=False,
                norm_eps=norm_eps,
                key=next(keys),
            )
            if has_middle
            else None
        )
        self.decoder_blocks = tuple(
            _TransformerBlock(
                int(model_width),
                int(heads),
                hidden_width,
                attention_dropout=attention_dropout,
                feed_forward_dropout=feed_forward_dropout,
                skip_connection=bool(long_range_skip),
                norm_eps=norm_eps,
                key=next(keys),
            )
            for _ in range(decoder_count)
        )
        self.latent_shape = shape
        self.patch_shape = patches
        self.channels = int(channels)
        self.model_width = int(model_width)
        self.coord_dim = len(shape)

    @property
    def patch_count(self) -> int:
        return prod(
            size // patch
            for size, patch in zip(self.latent_shape, self.patch_shape, strict=True)
        )

    def patchify(self, values: Array, /) -> Array:
        """Convert flattened latent point values into ordered patch tokens."""
        values = jnp.asarray(values)
        if values.ndim < 2 or int(values.shape[-2]) != prod(self.latent_shape):
            raise ValueError("Latent values do not match the configured point count.")
        if int(values.shape[-1]) != self.channels:
            raise ValueError(f"Latent values must have {self.channels} channels.")
        case_shape = tuple(int(size) for size in values.shape[:-2])
        grid = values.reshape(case_shape + self.latent_shape + (self.channels,))
        return _patchify_grid(grid, self.latent_shape, self.patch_shape)

    def unpatchify(self, tokens: Array, case_shape: Sequence[int], /) -> Array:
        """Invert :meth:`patchify` without interpolation or averaging."""
        cases = tuple(int(size) for size in case_shape)
        grid = _unpatchify_grid(
            tokens,
            cases,
            self.latent_shape,
            self.patch_shape,
            self.channels,
        )
        return grid.reshape(cases + (prod(self.latent_shape), self.channels))

    def __call__(
        self,
        values: Array,
        coordinates: Array,
        measure: Array,
        mask: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        values = jnp.asarray(values)
        coordinates = jnp.asarray(coordinates, dtype=float)
        measure = jnp.asarray(measure, dtype=float)
        mask = jnp.asarray(mask, dtype=bool)
        point_count = prod(self.latent_shape)
        if values.ndim < 2 or values.shape[-2:] != (point_count, self.channels):
            raise ValueError(
                f"Transformer values must end in {(point_count, self.channels)}."
            )
        case_shape = tuple(int(size) for size in values.shape[:-2])
        if coordinates.shape != case_shape + (point_count, self.coord_dim):
            raise ValueError("Transformer coordinates do not match its latent grid.")
        if measure.shape != case_shape + (point_count,):
            raise ValueError("Transformer measure does not match its latent grid.")
        if mask.shape != case_shape + (point_count,):
            raise ValueError("Transformer mask does not match its latent grid.")

        masked_values = values * mask[..., None].astype(values.dtype)
        tokens = self.patchify(masked_values)
        coordinate_cells = _patchify_grid(
            coordinates.reshape(case_shape + self.latent_shape + (self.coord_dim,)),
            self.latent_shape,
            self.patch_shape,
        ).reshape(case_shape + (self.patch_count, prod(self.patch_shape), self.coord_dim))
        measure_cells = _patchify_grid(
            measure.reshape(case_shape + self.latent_shape + (1,)),
            self.latent_shape,
            self.patch_shape,
        ).reshape(case_shape + (self.patch_count, prod(self.patch_shape)))
        mask_cells = (
            _patchify_grid(
                mask[..., None].reshape(case_shape + self.latent_shape + (1,)),
                self.latent_shape,
                self.patch_shape,
            )
            .reshape(case_shape + (self.patch_count, prod(self.patch_shape)))
            .astype(bool)
        )
        weighted_measure = measure_cells * mask_cells.astype(measure_cells.dtype)
        token_measure = jnp.sum(weighted_measure, axis=-1)
        token_mask = token_measure > 0.0
        token_measure = eqx.error_if(
            token_measure,
            jnp.any(jnp.sum(token_mask, axis=-1) == 0),
            "OperatorTransformerProcessor requires one non-empty patch per case.",
        )
        patch_centers = jnp.sum(
            coordinate_cells * weighted_measure[..., None],
            axis=-2,
        ) / jnp.maximum(token_measure[..., None], jnp.finfo(token_measure.dtype).tiny)
        lower = jnp.min(
            jnp.where(token_mask[..., None], patch_centers, jnp.inf),
            axis=-2,
            keepdims=True,
        )
        upper = jnp.max(
            jnp.where(token_mask[..., None], patch_centers, -jnp.inf),
            axis=-2,
            keepdims=True,
        )
        span = jnp.where(upper > lower, upper - lower, 1.0)
        normalized_positions = (patch_centers - lower) / span

        transformed = self.input_projection(tokens)
        transformed = transformed * token_mask[..., None].astype(transformed.dtype)
        position_encoding = _sinusoidal_positions(
            normalized_positions,
            self.model_width,
        ) * token_mask[..., None].astype(transformed.dtype)
        block_keys = split_eval_key(
            key,
            len(self.encoder_blocks)
            + (1 if self.middle_block is not None else 0)
            + len(self.decoder_blocks),
        )
        key_index = 0
        skips = []
        for block in self.encoder_blocks:
            transformed = block(
                transformed,
                token_measure,
                token_mask,
                position_encoding=position_encoding,
                key=block_keys[key_index],
            )
            key_index += 1
            skips.append(transformed)
        if self.middle_block is not None:
            transformed = self.middle_block(
                transformed,
                token_measure,
                token_mask,
                position_encoding=position_encoding,
                key=block_keys[key_index],
            )
            key_index += 1
        for block in self.decoder_blocks:
            transformed = block(
                transformed,
                token_measure,
                token_mask,
                position_encoding=position_encoding,
                skip=skips.pop() if block.skip_projection is not None else None,
                key=block_keys[key_index],
            )
            key_index += 1
        output_tokens = self.output_projection(transformed)
        output = self.unpatchify(output_tokens, case_shape)
        return output * mask[..., None].astype(output.dtype)


__all__ = ["OperatorTransformerProcessor"]
