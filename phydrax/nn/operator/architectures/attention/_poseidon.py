#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod, sqrt
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

import phydrax.ein as ein
from phydrax._doc import DOC_KEY0
from phydrax._strict import StrictModule
from phydrax.nn._keys import EvalKey
from phydrax.nn._utils import _get_size
from phydrax.nn.layers._linear import Linear
from phydrax.nn.operator.data import FunctionSamples, OperatorAxis, OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel


def _image_shape(shape: int | Sequence[int], /) -> tuple[int, int]:
    if isinstance(shape, int):
        result = (int(shape), int(shape))
    else:
        result = tuple(int(size) for size in shape)
    if len(result) != 2 or any(size <= 0 for size in result):
        raise ValueError("image_shape must contain two positive dimensions.")
    return result


def _patch_shape(shape: int | Sequence[int], /) -> tuple[int, int]:
    if isinstance(shape, int):
        result = (int(shape), int(shape))
    else:
        result = tuple(int(size) for size in shape)
    if len(result) != 2 or any(size <= 0 for size in result):
        raise ValueError("patch_size must contain two positive dimensions.")
    return result


def _single_array(samples: FunctionSamples, name: str, /) -> Array:
    if samples.values is None:
        raise ValueError(f"{name} requires sampled values.")
    return samples.values


def _source(
    batch: OperatorBatch,
    source_key: str | None,
    excluded_key: str | None,
    /,
) -> FunctionSamples:
    if source_key is not None:
        return batch.input(source_key)
    candidates = tuple(
        samples
        for name, samples in batch.inputs.items()
        if excluded_key is None or name != excluded_key
    )
    if len(candidates) != 1:
        raise ValueError(
            "source_key is required when multiple source fields are present."
        )
    return candidates[0]


def _prepare_field(
    samples: FunctionSamples,
    case_shape: tuple[int, ...],
    channels: int,
    image_shape: tuple[int, int],
    /,
) -> Array:
    if len(samples.axes) != 2:
        raise ValueError(
            "Poseidon requires a two-dimensional tensor-product source grid."
        )
    if samples.sample_shape != image_shape:
        raise ValueError(
            f"Poseidon was configured for source shape {image_shape}; "
            f"got {samples.sample_shape}."
        )
    values = _single_array(samples, "Poseidon")
    scalar_shape = case_shape + image_shape
    channel_shape = scalar_shape + (channels,)
    if values.shape == scalar_shape and channels == 1:
        values = values[..., None]
    elif values.shape != channel_shape:
        raise ValueError(f"Poseidon source values must have shape {channel_shape}.")
    mask = samples.mask_array(case_shape=case_shape)
    return values * mask[..., None].astype(values.dtype)


def _prepare_time(
    batch: OperatorBatch,
    time_input_name: str | None,
    /,
) -> Array | None:
    if time_input_name is None:
        return None
    values = _single_array(batch.input(time_input_name), "Poseidon time conditioning")
    expected = batch.case_shape
    if values.shape == expected:
        return values.reshape((prod(expected) if expected else 1,))
    if values.shape == expected + (1,):
        return values.reshape((prod(expected) if expected else 1,))
    raise ValueError(
        "Poseidon time conditioning must contain one scalar per operator case."
    )


def _patchify(values: Array, patch_size: tuple[int, int], /) -> Array:
    batch, height, width, channels = values.shape
    patch_height, patch_width = patch_size
    if height % patch_height or width % patch_width:
        raise ValueError("Image dimensions must be divisible by patch_size.")
    return (
        values.reshape(
            batch,
            height // patch_height,
            patch_height,
            width // patch_width,
            patch_width,
            channels,
        )
        .transpose((0, 1, 3, 2, 4, 5))
        .reshape(
            batch,
            height // patch_height,
            width // patch_width,
            patch_height * patch_width * channels,
        )
    )


def _unpatchify(
    patches: Array,
    patch_size: tuple[int, int],
    channels: int,
    /,
) -> Array:
    batch, height, width, _ = patches.shape
    patch_height, patch_width = patch_size
    return (
        patches.reshape(
            batch,
            height,
            width,
            patch_height,
            patch_width,
            channels,
        )
        .transpose((0, 1, 3, 2, 4, 5))
        .reshape(
            batch,
            height * patch_height,
            width * patch_width,
            channels,
        )
    )


def _apply_layer_norm(norm: eqx.nn.LayerNorm, values: Array, /) -> Array:
    flattened = values.reshape((-1, int(values.shape[-1])))
    return jax.vmap(norm)(flattened).reshape(values.shape)


class _ConditionedLayerNorm(StrictModule):
    norm: eqx.nn.LayerNorm
    scale_weight: Array
    scale_bias: Array
    shift_weight: Array
    shift_bias: Array
    conditioned: bool

    def __init__(
        self,
        width: int,
        /,
        *,
        conditioned: bool,
        eps: float,
        key: Key[Array, ""],
    ):
        self.conditioned = bool(conditioned)
        self.norm = eqx.nn.LayerNorm(
            int(width),
            eps=float(eps),
            use_weight=not self.conditioned,
            use_bias=not self.conditioned,
        )
        scale_key, shift_key = jr.split(key)
        modulation_scale = 1.0 / sqrt(float(width))
        self.scale_weight = modulation_scale * jr.normal(scale_key, (int(width),))
        self.scale_bias = jnp.ones((int(width),), dtype=float)
        self.shift_weight = modulation_scale * jr.normal(shift_key, (int(width),))
        self.shift_bias = jnp.zeros((int(width),), dtype=float)

    def __call__(self, values: Array, time: Array | None, /) -> Array:
        normalized = _apply_layer_norm(self.norm, values)
        if not self.conditioned:
            return normalized
        if time is None:
            raise ValueError("This Poseidon model requires one time value per case.")
        if values.ndim < 2 or int(values.shape[0]) != int(time.shape[0]):
            raise ValueError("Time conditioning and latent case dimensions differ.")
        broadcast = (
            (int(values.shape[0]),) + (1,) * (values.ndim - 2) + (int(values.shape[-1]),)
        )
        scalar_time = time.reshape((int(values.shape[0]),) + (1,) * (values.ndim - 1))
        scale = scalar_time * self.scale_weight + self.scale_bias
        shift = scalar_time * self.shift_weight + self.shift_bias
        return normalized * scale.reshape(broadcast) + shift.reshape(broadcast)


class _WindowAttention2D(StrictModule):
    query: Linear
    key: Linear
    value: Linear
    output: Linear
    relative_bias: Array
    width: int
    num_heads: int
    head_dim: int
    window_size: int
    shift_size: int

    def __init__(
        self,
        *,
        width: int,
        num_heads: int,
        window_size: int,
        shifted: bool,
        key: Key[Array, ""],
    ):
        self.width = int(width)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.shift_size = self.window_size // 2 if shifted else 0
        if self.width <= 0 or self.num_heads <= 0 or self.window_size <= 0:
            raise ValueError("width, num_heads, and window_size must be positive.")
        if self.width % self.num_heads:
            raise ValueError("Every Poseidon stage width must be divisible by num_heads.")
        self.head_dim = self.width // self.num_heads
        keys = jr.split(key, 5)
        self.query = Linear(
            in_size=self.width,
            out_size=self.width,
            activation=None,
            rwf=False,
            key=keys[0],
        )
        self.key = Linear(
            in_size=self.width,
            out_size=self.width,
            activation=None,
            rwf=False,
            key=keys[1],
        )
        self.value = Linear(
            in_size=self.width,
            out_size=self.width,
            activation=None,
            rwf=False,
            key=keys[2],
        )
        self.output = Linear(
            in_size=self.width,
            out_size=self.width,
            activation=None,
            rwf=False,
            key=keys[3],
        )
        side = 2 * self.window_size - 1
        self.relative_bias = 0.02 * jr.normal(keys[4], (self.num_heads, side, side))

    def _partition(self, values: Array, /) -> tuple[Array, Array, Array, int, int]:
        batch, height, width, channels = values.shape
        window = self.window_size
        pad_height = (-height) % window
        pad_width = (-width) % window
        padded = jnp.pad(values, ((0, 0), (0, pad_height), (0, pad_width), (0, 0)))
        valid = jnp.pad(
            jnp.ones((batch, height, width), dtype=bool),
            ((0, 0), (0, pad_height), (0, pad_width)),
        )
        rows, columns = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
        coordinates = jnp.stack((rows, columns), axis=-1)
        coordinates = jnp.pad(
            coordinates,
            ((0, pad_height), (0, pad_width), (0, 0)),
            constant_values=-2 * max(height, width, window),
        )
        coordinates = jnp.broadcast_to(coordinates, padded.shape[:3] + (2,))
        if self.shift_size:
            shift = (-self.shift_size, -self.shift_size)
            padded = jnp.roll(padded, shift, axis=(1, 2))
            valid = jnp.roll(valid, shift, axis=(1, 2))
            coordinates = jnp.roll(coordinates, shift, axis=(1, 2))
        padded_height, padded_width = padded.shape[1:3]
        row_windows = padded_height // window
        column_windows = padded_width // window
        windows = (
            padded.reshape(
                batch,
                row_windows,
                window,
                column_windows,
                window,
                channels,
            )
            .transpose((0, 1, 3, 2, 4, 5))
            .reshape((-1, window * window, channels))
        )
        valid_windows = (
            valid.reshape(batch, row_windows, window, column_windows, window)
            .transpose((0, 1, 3, 2, 4))
            .reshape((-1, window * window))
        )
        coordinate_windows = (
            coordinates.reshape(
                batch,
                row_windows,
                window,
                column_windows,
                window,
                2,
            )
            .transpose((0, 1, 3, 2, 4, 5))
            .reshape((-1, window * window, 2))
        )
        return windows, valid_windows, coordinate_windows, padded_height, padded_width

    def _unpartition(
        self,
        windows: Array,
        /,
        *,
        batch: int,
        height: int,
        width: int,
        padded_height: int,
        padded_width: int,
    ) -> Array:
        window = self.window_size
        values = (
            windows.reshape(
                batch,
                padded_height // window,
                padded_width // window,
                window,
                window,
                self.width,
            )
            .transpose((0, 1, 3, 2, 4, 5))
            .reshape((batch, padded_height, padded_width, self.width))
        )
        if self.shift_size:
            values = jnp.roll(
                values,
                (self.shift_size, self.shift_size),
                axis=(1, 2),
            )
        return values[:, :height, :width, :]

    def _relative_position_bias(self, /) -> Array:
        window = self.window_size
        rows, columns = jnp.meshgrid(
            jnp.arange(window), jnp.arange(window), indexing="ij"
        )
        coordinates = jnp.stack((rows, columns), axis=-1).reshape((-1, 2))
        relative = coordinates[:, None, :] - coordinates[None, :, :] + window - 1
        return self.relative_bias[:, relative[..., 0], relative[..., 1]]

    def __call__(self, values: Array, /) -> Array:
        batch, height, width, _ = values.shape
        windows, valid, coordinates, padded_height, padded_width = self._partition(values)
        token_count = self.window_size * self.window_size
        shape = (int(windows.shape[0]), token_count, self.num_heads, self.head_dim)
        query = self.query(windows).reshape(shape)
        key = self.key(windows).reshape(shape)
        value = self.value(windows).reshape(shape)
        logits = ein.contract("bqhd,bkhd->bhqk", query, key) / sqrt(float(self.head_dim))
        logits = logits + self._relative_position_bias()[None, ...]
        separation = jnp.abs(coordinates[:, :, None, :] - coordinates[:, None, :, :])
        same_region = jnp.all(separation < self.window_size, axis=-1)
        pair_mask = valid[:, :, None] & valid[:, None, :] & same_region
        logits = jnp.where(pair_mask[:, None, :, :], logits, -1e30)
        weights = jax.nn.softmax(logits, axis=-1)
        attended = ein.contract("bhqk,bkhd->bqhd", weights, value)
        attended = attended * valid[:, :, None, None].astype(attended.dtype)
        attended = self.output(attended.reshape(windows.shape))
        return self._unpartition(
            attended,
            batch=batch,
            height=height,
            width=width,
            padded_height=padded_height,
            padded_width=padded_width,
        )


class _PoseidonBlock(StrictModule):
    attention: _WindowAttention2D
    attention_norm: _ConditionedLayerNorm
    feed_forward_norm: _ConditionedLayerNorm
    expand: Linear
    contract: Linear

    def __init__(
        self,
        *,
        width: int,
        num_heads: int,
        window_size: int,
        shifted: bool,
        mlp_ratio: float,
        conditioned: bool,
        norm_eps: float,
        key: Key[Array, ""],
    ):
        hidden = int(round(float(mlp_ratio) * int(width)))
        if hidden <= 0:
            raise ValueError("mlp_ratio must produce a positive hidden width.")
        keys = jr.split(key, 5)
        self.attention = _WindowAttention2D(
            width=width,
            num_heads=num_heads,
            window_size=window_size,
            shifted=shifted,
            key=keys[0],
        )
        self.attention_norm = _ConditionedLayerNorm(
            width,
            conditioned=conditioned,
            eps=norm_eps,
            key=keys[1],
        )
        self.feed_forward_norm = _ConditionedLayerNorm(
            width,
            conditioned=conditioned,
            eps=norm_eps,
            key=keys[2],
        )
        self.expand = Linear(
            in_size=width,
            out_size=hidden,
            activation=jax.nn.gelu,
            rwf=False,
            key=keys[3],
        )
        self.contract = Linear(
            in_size=hidden,
            out_size=width,
            activation=None,
            rwf=False,
            key=keys[4],
        )

    def __call__(self, values: Array, time: Array | None, /) -> Array:
        attended = self.attention(values)
        hidden = values + self.attention_norm(attended, time)
        feed_forward = self.contract(self.expand(hidden))
        return hidden + self.feed_forward_norm(feed_forward, time)


class _PoseidonStage(StrictModule):
    blocks: tuple[_PoseidonBlock, ...]

    def __init__(
        self,
        *,
        width: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        conditioned: bool,
        norm_eps: float,
        key: Key[Array, ""],
    ):
        self.blocks = tuple(
            _PoseidonBlock(
                width=width,
                num_heads=num_heads,
                window_size=window_size,
                shifted=bool(index % 2),
                mlp_ratio=mlp_ratio,
                conditioned=conditioned,
                norm_eps=norm_eps,
                key=block_key,
            )
            for index, block_key in enumerate(jr.split(key, int(depth)))
        )

    def __call__(self, values: Array, time: Array | None, /) -> Array:
        for block in self.blocks:
            values = block(values, time)
        return values


class _PatchMerge(StrictModule):
    reduction: Linear
    norm: _ConditionedLayerNorm
    in_width: int

    def __init__(
        self,
        in_width: int,
        /,
        *,
        conditioned: bool,
        norm_eps: float,
        key: Key[Array, ""],
    ):
        reduction_key, norm_key = jr.split(key)
        self.in_width = int(in_width)
        self.reduction = Linear(
            in_size=4 * self.in_width,
            out_size=2 * self.in_width,
            activation=None,
            use_bias=False,
            rwf=False,
            key=reduction_key,
        )
        self.norm = _ConditionedLayerNorm(
            2 * self.in_width,
            conditioned=conditioned,
            eps=norm_eps,
            key=norm_key,
        )

    def __call__(self, values: Array, time: Array | None, /) -> Array:
        batch, height, width, channels = values.shape
        if height % 2 or width % 2:
            raise ValueError("Poseidon latent dimensions must remain divisible by two.")
        merged = (
            values.reshape(batch, height // 2, 2, width // 2, 2, channels)
            .transpose((0, 1, 3, 2, 4, 5))
            .reshape(batch, height // 2, width // 2, 4 * channels)
        )
        return self.norm(self.reduction(merged), time)


class _PatchUnmerge(StrictModule):
    expansion: Linear
    mix: Linear
    norm: _ConditionedLayerNorm
    in_width: int

    def __init__(
        self,
        in_width: int,
        /,
        *,
        conditioned: bool,
        norm_eps: float,
        key: Key[Array, ""],
    ):
        expansion_key, mix_key, norm_key = jr.split(key, 3)
        self.in_width = int(in_width)
        out_width = self.in_width // 2
        self.expansion = Linear(
            in_size=self.in_width,
            out_size=4 * out_width,
            activation=None,
            use_bias=False,
            rwf=False,
            key=expansion_key,
        )
        self.mix = Linear(
            in_size=out_width,
            out_size=out_width,
            activation=None,
            use_bias=False,
            rwf=False,
            key=mix_key,
        )
        self.norm = _ConditionedLayerNorm(
            out_width,
            conditioned=conditioned,
            eps=norm_eps,
            key=norm_key,
        )

    def __call__(self, values: Array, time: Array | None, /) -> Array:
        batch, height, width, _ = values.shape
        out_width = self.in_width // 2
        expanded = self.expansion(values).reshape(batch, height, width, 2, 2, out_width)
        expanded = expanded.transpose((0, 1, 3, 2, 4, 5)).reshape(
            batch, 2 * height, 2 * width, out_width
        )
        return self.mix(self.norm(expanded, time))


class _ConvNeXtSkipBlock(StrictModule):
    depthwise_weight: Array
    norm: _ConditionedLayerNorm
    expand: Linear
    contract: Linear
    layer_scale: Array
    width: int

    def __init__(
        self,
        width: int,
        /,
        *,
        conditioned: bool,
        norm_eps: float,
        key: Key[Array, ""],
    ):
        keys = jr.split(key, 4)
        self.width = int(width)
        self.depthwise_weight = jr.normal(keys[0], (7, 7, 1, self.width)) / 7.0
        self.norm = _ConditionedLayerNorm(
            self.width,
            conditioned=conditioned,
            eps=norm_eps,
            key=keys[1],
        )
        self.expand = Linear(
            in_size=self.width,
            out_size=4 * self.width,
            activation=jax.nn.gelu,
            rwf=False,
            key=keys[2],
        )
        self.contract = Linear(
            in_size=4 * self.width,
            out_size=self.width,
            activation=None,
            rwf=False,
            key=keys[3],
        )
        self.layer_scale = jnp.full((self.width,), 1e-6)

    def __call__(self, values: Array, time: Array | None, /) -> Array:
        filtered = jax.lax.conv_general_dilated(
            values,
            self.depthwise_weight,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=self.width,
        )
        update = self.contract(self.expand(self.norm(filtered, time)))
        return values + self.layer_scale * update


class Poseidon(AbstractOperatorModel):
    """Native scOT-style multiscale operator transformer used by Poseidon.

    The model consumes a two-dimensional tensor-product field, patchifies it,
    applies a shifted-window transformer U-Net with ConvNeXt skip processing,
    and reconstructs a coincident output grid. Supplying ``time_input_name``
    enables the continuous-time affine layer-normalization used by Poseidon.
    """

    operator_architecture = "Poseidon"

    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    image_shape: tuple[int, int]
    patch_size: tuple[int, int]
    widths: tuple[int, ...]
    source_key: str | None
    time_input_name: str | None
    learn_residual: bool
    patch_embedding: Linear
    encoder_stages: tuple[_PoseidonStage, ...]
    mergers: tuple[_PatchMerge, ...]
    skip_processors: tuple[tuple[_ConvNeXtSkipBlock, ...], ...]
    decoder_stages: tuple[_PoseidonStage, ...]
    unmergers: tuple[_PatchUnmerge, ...]
    patch_recovery: Linear
    output_mix_weight: Array

    def __init__(
        self,
        *,
        image_shape: int | Sequence[int],
        patch_size: int | Sequence[int] = 4,
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        skip_depths: Sequence[int] | None = None,
        time_input_name: str | None = None,
        source_key: str | None = None,
        learn_residual: bool = False,
        norm_eps: float = 1e-5,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_size = in_channels
        self.out_size = out_channels
        self.image_shape = _image_shape(image_shape)
        self.patch_size = _patch_shape(patch_size)
        self.source_key = source_key
        self.time_input_name = time_input_name
        self.learn_residual = bool(learn_residual)
        depths_ = tuple(int(depth) for depth in depths)
        heads_ = tuple(int(heads) for heads in num_heads)
        if not depths_ or len(depths_) != len(heads_):
            raise ValueError(
                "depths and num_heads must be non-empty and have equal length."
            )
        if any(depth <= 0 for depth in depths_) or any(heads <= 0 for heads in heads_):
            raise ValueError("Poseidon depths and head counts must be positive.")
        if int(embed_dim) <= 0 or int(window_size) <= 0 or float(mlp_ratio) <= 0.0:
            raise ValueError("embed_dim, window_size, and mlp_ratio must be positive.")
        latent_shape = tuple(
            size // patch for size, patch in zip(self.image_shape, self.patch_size)
        )
        divisor = 2 ** (len(depths_) - 1)
        if any(
            size % patch or latent % divisor
            for size, patch, latent in zip(
                self.image_shape, self.patch_size, latent_shape
            )
        ):
            raise ValueError(
                "image_shape must be divisible by patch_size and every multiscale merge."
            )
        self.widths = tuple(int(embed_dim) * 2**level for level in range(len(depths_)))
        if any(width % heads for width, heads in zip(self.widths, heads_)):
            raise ValueError(
                "Each Poseidon stage width must be divisible by its head count."
            )
        if self.learn_residual and _get_size(in_channels) != _get_size(out_channels):
            raise ValueError("learn_residual requires equal input and output channels.")
        if skip_depths is None:
            skip_depths_ = (1,) * (len(depths_) - 1)
        else:
            skip_depths_ = tuple(int(depth) for depth in skip_depths)
        if len(skip_depths_) != len(depths_) - 1 or any(
            depth < 0 for depth in skip_depths_
        ):
            raise ValueError(
                "skip_depths must provide one non-negative depth per skip level."
            )

        conditioned = self.time_input_name is not None
        key_count = 1 + 4 * len(depths_) + sum(skip_depths_)
        keys = iter(jr.split(key, key_count))
        patch_features = self.patch_size[0] * self.patch_size[1] * _get_size(in_channels)
        self.patch_embedding = Linear(
            in_size=patch_features,
            out_size=self.widths[0],
            activation=None,
            rwf=False,
            key=next(keys),
        )
        self.encoder_stages = tuple(
            _PoseidonStage(
                width=width,
                depth=depth,
                num_heads=heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                conditioned=conditioned,
                norm_eps=norm_eps,
                key=next(keys),
            )
            for width, depth, heads in zip(self.widths, depths_, heads_)
        )
        self.mergers = tuple(
            _PatchMerge(
                width,
                conditioned=conditioned,
                norm_eps=norm_eps,
                key=next(keys),
            )
            for width in self.widths[:-1]
        )
        self.skip_processors = tuple(
            tuple(
                _ConvNeXtSkipBlock(
                    self.widths[level],
                    conditioned=conditioned,
                    norm_eps=norm_eps,
                    key=next(keys),
                )
                for _ in range(skip_depths_[level])
            )
            for level in range(len(skip_depths_))
        )
        self.decoder_stages = tuple(
            _PoseidonStage(
                width=width,
                depth=depth,
                num_heads=heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                conditioned=conditioned,
                norm_eps=norm_eps,
                key=next(keys),
            )
            for width, depth, heads in zip(self.widths, depths_, heads_)
        )
        self.unmergers = tuple(
            _PatchUnmerge(
                self.widths[level + 1],
                conditioned=conditioned,
                norm_eps=norm_eps,
                key=next(keys),
            )
            for level in range(len(self.widths) - 1)
        )
        recovered_features = (
            self.patch_size[0] * self.patch_size[1] * _get_size(out_channels)
        )
        self.patch_recovery = Linear(
            in_size=self.widths[0],
            out_size=recovered_features,
            activation=None,
            rwf=False,
            key=next(keys),
        )
        output_channels_ = _get_size(out_channels)
        self.output_mix_weight = jr.normal(
            next(keys), (5, 5, output_channels_, output_channels_)
        ) / sqrt(float(25 * output_channels_))

    def _evaluate(self, values: Array, time: Array | None, /) -> Array:
        batch = int(values.shape[0])
        hidden = self.patch_embedding(_patchify(values, self.patch_size))
        skips: list[Array] = []
        for level, stage in enumerate(self.encoder_stages):
            hidden = stage(hidden, time)
            skips.append(hidden)
            if level < len(self.mergers):
                hidden = self.mergers[level](hidden, time)
        for level, processors in enumerate(self.skip_processors):
            for processor in processors:
                skips[level] = processor(skips[level], time)

        hidden = skips[-1]
        for level in reversed(range(len(self.decoder_stages))):
            if level < len(self.decoder_stages) - 1:
                hidden = hidden + skips[level]
            hidden = self.decoder_stages[level](hidden, time)
            if level > 0:
                hidden = self.unmergers[level - 1](hidden, time)

        patches = self.patch_recovery(hidden)
        output = _unpatchify(patches, self.patch_size, _get_size(self.out_size))
        output = jax.lax.conv_general_dilated(
            output,
            self.output_mix_weight,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        if self.learn_residual:
            output = output + values
        if int(output.shape[0]) != batch:
            raise RuntimeError("Poseidon changed the flattened operator case count.")
        return output

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        del key
        source = _source(batch, self.source_key, self.time_input_name)
        if (
            batch.require_single_query().sample_shape != self.image_shape
            or len(batch.require_single_query().axes) != 2
        ):
            raise ValueError("Poseidon requires a coincident two-dimensional query grid.")
        values = _prepare_field(
            source,
            batch.case_shape,
            _get_size(self.in_size),
            self.image_shape,
        )
        flat_batch = prod(batch.case_shape) if batch.case_shape else 1
        values = values.reshape(
            (flat_batch,) + self.image_shape + (_get_size(self.in_size),)
        )
        time = _prepare_time(batch, self.time_input_name)
        output = self._evaluate(values, time)
        output = output.reshape(
            batch.case_shape + self.image_shape + (_get_size(self.out_size),)
        )
        output = output * batch.require_single_query().mask_array(
            case_shape=batch.case_shape
        )[..., None].astype(output.dtype)
        if self.out_size == "scalar":
            return output[..., 0]
        return output

    def __call__(
        self,
        x: Array | tuple[Array, ...] | OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        del key
        if isinstance(x, OperatorBatch):
            return self.__call_operator_batch__(x)
        expected = 4 if self.time_input_name is not None else 3
        if not isinstance(x, tuple) or len(x) != expected:
            suffix = ", time" if self.time_input_name is not None else ""
            raise ValueError(f"Poseidon requires (values, axis_0, axis_1{suffix}).")
        axes = (
            OperatorAxis("axis_0", x[1]),
            OperatorAxis("axis_1", x[2]),
        )
        values = jnp.asarray(x[0])
        scalar_suffix = self.image_shape
        channel_suffix = self.image_shape + (_get_size(self.in_size),)
        if self.in_size == "scalar" and values.shape[-2:] == scalar_suffix:
            case_shape = values.shape[:-2]
        elif values.shape[-3:] == channel_suffix:
            case_shape = values.shape[:-3]
        else:
            raise ValueError(
                "Poseidon values must end in its image shape and optional channel axis."
            )
        samples = FunctionSamples(values=values, axes=axes)
        inputs = {self.source_key or "input": samples}
        if self.time_input_name is not None:
            inputs[self.time_input_name] = FunctionSamples(values=jnp.asarray(x[3]))
        batch = OperatorBatch(
            inputs=inputs,
            queries={"query": FunctionSamples(values=None, axes=axes)},
            case_axes=tuple(f"case_{index}" for index in range(len(case_shape))),
            case_shape=case_shape,
        )
        return self.__call_operator_batch__(batch)


__all__ = ["Poseidon"]
