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
import opt_einsum as oe
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax._strict import StrictModule
from phydrax.nn._keys import EvalKey
from phydrax.nn._utils import _get_size
from phydrax.nn.layers._linear import Linear
from phydrax.nn.operator.data import FunctionSamples, OperatorAxis, OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel


def _pair(shape: int | Sequence[int], name: str, /) -> tuple[int, int]:
    if isinstance(shape, int):
        result = (int(shape), int(shape))
    else:
        result = tuple(int(size) for size in shape)
    if len(result) != 2 or any(size <= 0 for size in result):
        raise ValueError(f"{name} must contain two positive dimensions.")
    return result


def _single_array(samples: FunctionSamples, name: str, /) -> Array:
    if samples.values is None:
        raise ValueError(f"{name} requires sampled values.")
    return samples.values


def _source(batch: OperatorBatch, source_key: str | None, /) -> FunctionSamples:
    if source_key is not None:
        return batch.input(source_key)
    if len(batch.inputs) != 1:
        raise ValueError("source_key is required when OperatorBatch has multiple inputs.")
    return next(iter(batch.inputs.values()))


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


def _group_norm(norm: eqx.nn.GroupNorm, values: Array, /) -> Array:
    channel_first = values.transpose((0, 3, 1, 2))
    return jax.vmap(norm)(channel_first).transpose((0, 2, 3, 1))


def dpot_corrupt_history(
    values: Array,
    /,
    *,
    noise_scale: float = 1e-3,
    key: Key[Array, ""],
    mask: Array | None = None,
    channel_axis: int | None = -1,
) -> Array:
    """Apply DPOT's scale-relative Gaussian history corruption.

    Array-valued fields end in ``(height, width, history, channels)``. Pass
    ``channel_axis=None`` for scalar fields ending in ``(height, width, history)``.
    The L2 norm over the three sample axes independently scales every case/channel
    noise field, matching the official auto-regressive denoising recipe.
    """

    array = jnp.asarray(values)
    if float(noise_scale) < 0.0:
        raise ValueError("noise_scale must be non-negative.")
    if channel_axis is None:
        if array.ndim < 3:
            raise ValueError("Scalar DPOT histories must end in three sample axes.")
        sample_axes = (-3, -2, -1)
        expected_mask_shape = array.shape
        mask_suffix = ()
    else:
        axis = int(channel_axis) % array.ndim
        if array.ndim < 4 or axis != array.ndim - 1:
            raise ValueError("DPOT channel_axis must identify a final channel axis.")
        sample_axes = (-4, -3, -2)
        expected_mask_shape = array.shape[:-1]
        mask_suffix = (1,)
    sample_mask = None
    if mask is not None:
        sample_mask = jnp.asarray(mask, dtype=bool)
        if sample_mask.shape != expected_mask_shape:
            raise ValueError("mask must match the DPOT case and sample axes.")
        expanded_mask = sample_mask.reshape(sample_mask.shape + mask_suffix)
        array_for_norm = array * expanded_mask.astype(array.dtype)
    else:
        array_for_norm = array
    norm = jnp.sqrt(jnp.sum(jnp.square(array_for_norm), axis=sample_axes, keepdims=True))
    corrupted = array + float(noise_scale) * norm * jr.normal(key, array.shape)
    if sample_mask is not None:
        corrupted = corrupted * expanded_mask.astype(corrupted.dtype)
    return corrupted


class _AFNO2D(StrictModule):
    first_real: Array
    first_imag: Array
    first_real_bias: Array
    first_imag_bias: Array
    second_real: Array
    second_imag: Array
    second_real_bias: Array
    second_imag_bias: Array
    width: int
    num_blocks: int
    block_size: int
    modes: tuple[int, int]

    def __init__(
        self,
        *,
        width: int,
        num_blocks: int,
        modes: int | Sequence[int],
        key: Key[Array, ""],
    ):
        self.width = int(width)
        self.num_blocks = int(num_blocks)
        self.modes = _pair(modes, "modes")
        if self.width <= 0 or self.num_blocks <= 0:
            raise ValueError("width and num_blocks must be positive.")
        if self.width % self.num_blocks:
            raise ValueError("DPOT embed_dim must be divisible by num_blocks.")
        self.block_size = self.width // self.num_blocks
        scale = 1.0 / float(self.block_size * self.block_size)
        keys = jr.split(key, 8)
        matrix_shape = (self.num_blocks, self.block_size, self.block_size)
        bias_shape = (self.num_blocks, self.block_size)
        self.first_real = scale * jr.normal(keys[0], matrix_shape)
        self.first_imag = scale * jr.normal(keys[1], matrix_shape)
        self.first_real_bias = scale * jr.normal(keys[2], bias_shape)
        self.first_imag_bias = scale * jr.normal(keys[3], bias_shape)
        self.second_real = scale * jr.normal(keys[4], matrix_shape)
        self.second_imag = scale * jr.normal(keys[5], matrix_shape)
        self.second_real_bias = scale * jr.normal(keys[6], bias_shape)
        self.second_imag_bias = scale * jr.normal(keys[7], bias_shape)

    def __call__(self, values: Array, /) -> Array:
        batch, height, width, _ = values.shape
        spectrum = jnp.fft.rfft2(values, axes=(1, 2), norm="ortho")
        frequency_width = int(spectrum.shape[2])
        blocks = spectrum.reshape(
            batch,
            height,
            frequency_width,
            self.num_blocks,
            self.block_size,
        )
        modes_height = min(self.modes[0], height)
        modes_width = min(self.modes[1], frequency_width)
        active = blocks[:, :modes_height, :modes_width]
        first_real = jax.nn.gelu(
            oe.contract("...bi,bio->...bo", active.real, self.first_real)
            - oe.contract("...bi,bio->...bo", active.imag, self.first_imag)
            + self.first_real_bias
        )
        first_imag = jax.nn.gelu(
            oe.contract("...bi,bio->...bo", active.imag, self.first_real)
            + oe.contract("...bi,bio->...bo", active.real, self.first_imag)
            + self.first_imag_bias
        )
        second_real = (
            oe.contract("...bi,bio->...bo", first_real, self.second_real)
            - oe.contract("...bi,bio->...bo", first_imag, self.second_imag)
            + self.second_real_bias
        )
        second_imag = (
            oe.contract("...bi,bio->...bo", first_imag, self.second_real)
            + oe.contract("...bi,bio->...bo", first_real, self.second_imag)
            + self.second_imag_bias
        )
        transformed = (
            jnp.zeros_like(blocks)
            .at[:, :modes_height, :modes_width]
            .set(second_real + 1j * second_imag)
        )
        transformed = transformed.reshape(spectrum.shape)
        filtered = jnp.fft.irfft2(
            transformed,
            s=(height, width),
            axes=(1, 2),
            norm="ortho",
        )
        return filtered + values


class _DPOTBlock(StrictModule):
    first_norm: eqx.nn.GroupNorm
    filter: _AFNO2D
    second_norm: eqx.nn.GroupNorm
    expand: Linear
    contract: Linear
    double_skip: bool

    def __init__(
        self,
        *,
        width: int,
        num_blocks: int,
        modes: int | Sequence[int],
        mlp_ratio: float,
        groups: int,
        double_skip: bool,
        key: Key[Array, ""],
    ):
        hidden = int(round(float(mlp_ratio) * int(width)))
        if hidden <= 0:
            raise ValueError("mlp_ratio must produce a positive DPOT hidden width.")
        if int(width) % int(groups):
            raise ValueError("normalization groups must divide embed_dim.")
        filter_key, expand_key, contract_key = jr.split(key, 3)
        self.first_norm = eqx.nn.GroupNorm(int(groups), int(width))
        self.filter = _AFNO2D(
            width=width,
            num_blocks=num_blocks,
            modes=modes,
            key=filter_key,
        )
        self.second_norm = eqx.nn.GroupNorm(int(groups), int(width))
        self.expand = Linear(
            in_size=width,
            out_size=hidden,
            activation=jax.nn.gelu,
            rwf=False,
            key=expand_key,
        )
        self.contract = Linear(
            in_size=hidden,
            out_size=width,
            activation=None,
            rwf=False,
            key=contract_key,
        )
        self.double_skip = bool(double_skip)

    def __call__(self, values: Array, /) -> Array:
        residual = values
        hidden = self.filter(_group_norm(self.first_norm, values))
        if self.double_skip:
            hidden = hidden + residual
            residual = hidden
        hidden = self.contract(self.expand(_group_norm(self.second_norm, hidden)))
        return hidden + residual


class _TemporalAggregator(StrictModule):
    weight: Array
    frequencies: Array
    history_steps: int
    width: int
    exponential_embedding: bool

    def __init__(
        self,
        *,
        history_steps: int,
        width: int,
        exponential_embedding: bool,
        key: Key[Array, ""],
    ):
        self.history_steps = int(history_steps)
        self.width = int(width)
        self.exponential_embedding = bool(exponential_embedding)
        scale = 1.0 / float(self.history_steps * sqrt(float(self.width)))
        self.weight = scale * jr.normal(key, (self.history_steps, self.width, self.width))
        self.frequencies = 2.0 ** jnp.linspace(-10.0, 10.0, self.width)[None, :]

    def __call__(self, values: Array, /) -> Array:
        if int(values.shape[-2]) != self.history_steps:
            raise ValueError("DPOT latent history length differs from history_steps.")
        if self.exponential_embedding:
            time = jnp.linspace(0.0, 1.0, self.history_steps)[:, None]
            values = values * jnp.cos(time * self.frequencies)
        return oe.contract("tij,bhwti->bhwj", self.weight, values)


class DPOT(AbstractOperatorModel):
    """Auto-regressive Denoising Operator Transformer for temporal PDE fields.

    Inputs use explicit ``(x, y, history_time)`` sample axes. Outputs use an
    explicit ``(x, y, forecast_time)`` query, preserving operator metadata rather
    than flattening temporal steps into channels. The architecture implements
    DPOT's patch encoder, exponential temporal aggregation, AFNO block-diagonal
    Fourier mixing, and patch decoder. Use :func:`dpot_corrupt_history` for the
    official scale-relative denoising pretraining corruption.
    """

    operator_architecture = "DPOT"

    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    image_shape: tuple[int, int]
    patch_size: tuple[int, int]
    history_steps: int
    forecast_steps: int
    embed_dim: int
    out_layer_dim: int
    source_key: str | None
    normalize: bool
    patch_lift: Linear
    patch_projection: Linear
    position_embedding: Array
    temporal_aggregation: _TemporalAggregator
    scale_features_mean: Linear | None
    scale_features_std: Linear | None
    blocks: tuple[_DPOTBlock, ...]
    patch_recovery: Linear
    output_hidden: Linear
    output_projection: Linear

    def __init__(
        self,
        *,
        image_shape: int | Sequence[int],
        history_steps: int,
        forecast_steps: int = 1,
        patch_size: int | Sequence[int] = 4,
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        embed_dim: int = 128,
        depth: int = 8,
        modes: int | Sequence[int] = 16,
        num_blocks: int = 8,
        mlp_ratio: float = 1.0,
        out_layer_dim: int = 32,
        normalization_groups: int = 8,
        normalize: bool = False,
        exponential_time_embedding: bool = True,
        source_key: str | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_size = in_channels
        self.out_size = out_channels
        self.image_shape = _pair(image_shape, "image_shape")
        self.patch_size = _pair(patch_size, "patch_size")
        self.history_steps = int(history_steps)
        self.forecast_steps = int(forecast_steps)
        self.embed_dim = int(embed_dim)
        self.out_layer_dim = int(out_layer_dim)
        self.source_key = source_key
        self.normalize = bool(normalize)
        if self.history_steps <= 0 or self.forecast_steps <= 0:
            raise ValueError("history_steps and forecast_steps must be positive.")
        if self.embed_dim <= 0 or int(depth) <= 0 or self.out_layer_dim <= 0:
            raise ValueError("embed_dim, depth, and out_layer_dim must be positive.")
        if any(size % patch for size, patch in zip(self.image_shape, self.patch_size)):
            raise ValueError("image_shape must be divisible by patch_size.")
        in_channels_ = _get_size(in_channels)
        out_channels_ = _get_size(out_channels)
        if self.normalize and in_channels_ != out_channels_:
            raise ValueError("normalize=True requires equal input and output channels.")
        keys = iter(jr.split(key, int(depth) + 9))
        patch_area = self.patch_size[0] * self.patch_size[1]
        patch_hidden = out_channels_ * max(self.patch_size) + 3
        self.patch_lift = Linear(
            in_size=patch_area * (in_channels_ + 3),
            out_size=patch_hidden,
            activation=jax.nn.gelu,
            rwf=False,
            key=next(keys),
        )
        self.patch_projection = Linear(
            in_size=patch_hidden,
            out_size=self.embed_dim,
            activation=None,
            rwf=False,
            key=next(keys),
        )
        latent_shape = tuple(
            size // patch for size, patch in zip(self.image_shape, self.patch_size)
        )
        self.position_embedding = 0.02 * jr.normal(
            next(keys), latent_shape + (self.embed_dim,)
        )
        self.temporal_aggregation = _TemporalAggregator(
            history_steps=self.history_steps,
            width=self.embed_dim,
            exponential_embedding=exponential_time_embedding,
            key=next(keys),
        )
        if self.normalize:
            self.scale_features_mean = Linear(
                in_size=2 * in_channels_,
                out_size=self.embed_dim,
                activation=None,
                rwf=False,
                key=next(keys),
            )
            self.scale_features_std = Linear(
                in_size=2 * in_channels_,
                out_size=self.embed_dim,
                activation=None,
                rwf=False,
                key=next(keys),
            )
        else:
            self.scale_features_mean = None
            self.scale_features_std = None
            next(keys)
            next(keys)
        self.blocks = tuple(
            _DPOTBlock(
                width=self.embed_dim,
                num_blocks=num_blocks,
                modes=modes,
                mlp_ratio=mlp_ratio,
                groups=normalization_groups,
                double_skip=False,
                key=next(keys),
            )
            for _ in range(int(depth))
        )
        self.patch_recovery = Linear(
            in_size=self.embed_dim,
            out_size=patch_area * self.out_layer_dim,
            activation=jax.nn.gelu,
            rwf=False,
            key=next(keys),
        )
        self.output_hidden = Linear(
            in_size=self.out_layer_dim,
            out_size=self.out_layer_dim,
            activation=jax.nn.gelu,
            rwf=False,
            key=next(keys),
        )
        self.output_projection = Linear(
            in_size=self.out_layer_dim,
            out_size=self.forecast_steps * out_channels_,
            activation=None,
            rwf=False,
            key=next(keys),
        )

    def corrupt_batch(
        self,
        batch: OperatorBatch,
        /,
        *,
        noise_scale: float = 1e-3,
        key: Key[Array, ""],
    ) -> OperatorBatch:
        """Return an operator batch with DPOT denoising corruption on its history."""

        source = _source(batch, self.source_key)
        values = _single_array(source, "DPOT")
        scalar_layout = self.in_size == "scalar" and values.ndim == len(
            batch.case_shape
        ) + len(source.sample_shape)
        corrupted = dpot_corrupt_history(
            values,
            noise_scale=noise_scale,
            key=key,
            mask=source.mask_array(case_shape=batch.case_shape),
            channel_axis=None if scalar_layout else -1,
        )
        source_name = (
            self.source_key if self.source_key is not None else next(iter(batch.inputs))
        )
        inputs = dict(batch.inputs)
        inputs[source_name] = FunctionSamples(
            values=corrupted,
            axes=source.axes,
            coordinates=source.coordinates,
            quadrature_weights=source.quadrature_weights,
            mask=source.mask,
            topology=source.topology,
        )
        return OperatorBatch(
            inputs=inputs,
            queries=batch.queries,
            case_axes=batch.case_axes,
            case_shape=batch.case_shape,
        )

    def _prepare_history(
        self,
        samples: FunctionSamples,
        case_shape: tuple[int, ...],
        /,
    ) -> tuple[Array, Array]:
        expected_sample_shape = self.image_shape + (self.history_steps,)
        if len(samples.axes) != 3 or samples.sample_shape != expected_sample_shape:
            raise ValueError(
                "DPOT source axes must have sample shape "
                f"{expected_sample_shape}; got {samples.sample_shape}."
            )
        values = _single_array(samples, "DPOT")
        channels = _get_size(self.in_size)
        scalar_shape = case_shape + expected_sample_shape
        channel_shape = scalar_shape + (channels,)
        if values.shape == scalar_shape and channels == 1:
            values = values[..., None]
        elif values.shape != channel_shape:
            raise ValueError(f"DPOT source values must have shape {channel_shape}.")
        mask = samples.mask_array(case_shape=case_shape)
        values = values * mask[..., None].astype(values.dtype)
        return values, mask

    def _normalize_history(
        self,
        values: Array,
        mask: Array,
        /,
    ) -> tuple[Array, Array | None, Array | None]:
        if not self.normalize:
            return values, None, None
        weights = mask[..., None].astype(values.dtype)
        count = jnp.maximum(jnp.sum(weights, axis=(1, 2, 3), keepdims=True), 1.0)
        mean = jnp.sum(values, axis=(1, 2, 3), keepdims=True) / count
        variance = (
            jnp.sum(jnp.square(values - mean) * weights, axis=(1, 2, 3), keepdims=True)
            / count
        )
        std = jnp.sqrt(variance + 1e-6)
        normalized = (values - mean) / std
        return normalized * weights, mean, std

    def _coordinate_grid(self, batch: int, /) -> Array:
        height, width = self.image_shape
        x_coordinate, y_coordinate, time_coordinate = jnp.meshgrid(
            jnp.linspace(0.0, 1.0, height),
            jnp.linspace(0.0, 1.0, width),
            jnp.linspace(0.0, 1.0, self.history_steps),
            indexing="ij",
        )
        grid = jnp.stack((x_coordinate, y_coordinate, time_coordinate), axis=-1)
        return jnp.broadcast_to(grid, (batch,) + grid.shape)

    def _evaluate(self, history: Array, mask: Array, /) -> Array:
        batch = int(history.shape[0])
        history, mean, std = self._normalize_history(history, mask)
        history = jnp.concatenate((history, self._coordinate_grid(batch)), axis=-1)
        per_time = history.transpose((0, 3, 1, 2, 4)).reshape(
            batch * self.history_steps,
            self.image_shape[0],
            self.image_shape[1],
            int(history.shape[-1]),
        )
        embedded = self.patch_projection(
            self.patch_lift(_patchify(per_time, self.patch_size))
        )
        latent_height, latent_width = embedded.shape[1:3]
        embedded = embedded + self.position_embedding
        embedded = embedded.reshape(
            batch,
            self.history_steps,
            latent_height,
            latent_width,
            self.embed_dim,
        ).transpose((0, 2, 3, 1, 4))
        hidden = self.temporal_aggregation(embedded)
        if self.normalize:
            assert mean is not None
            assert std is not None
            statistics = jnp.concatenate((mean, std), axis=-1).reshape(
                batch, 2 * _get_size(self.in_size)
            )
            assert self.scale_features_mean is not None
            assert self.scale_features_std is not None
            hidden = (
                self.scale_features_std(statistics)[:, None, None, :] * hidden
                + self.scale_features_mean(statistics)[:, None, None, :]
            )
        for block in self.blocks:
            hidden = block(hidden)
        patches = self.patch_recovery(hidden)
        decoded = _unpatchify(patches, self.patch_size, self.out_layer_dim)
        output = self.output_projection(self.output_hidden(decoded))
        output = output.reshape(
            (batch,) + self.image_shape + (self.forecast_steps, _get_size(self.out_size))
        )
        if self.normalize:
            assert mean is not None
            assert std is not None
            output = output * std + mean
        return output

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        del key
        expected_query_shape = self.image_shape + (self.forecast_steps,)
        if (
            len(batch.require_single_query().axes) != 3
            or batch.require_single_query().sample_shape != expected_query_shape
        ):
            raise ValueError(
                "DPOT query axes must have sample shape "
                f"{expected_query_shape}; got {batch.require_single_query().sample_shape}."
            )
        history, mask = self._prepare_history(
            _source(batch, self.source_key), batch.case_shape
        )
        flat_batch = prod(batch.case_shape) if batch.case_shape else 1
        history = history.reshape(
            (flat_batch,)
            + self.image_shape
            + (self.history_steps, _get_size(self.in_size))
        )
        mask = mask.reshape((flat_batch,) + self.image_shape + (self.history_steps,))
        output = self._evaluate(history, mask)
        output = output.reshape(
            batch.case_shape
            + self.image_shape
            + (self.forecast_steps, _get_size(self.out_size))
        )
        output = output * batch.require_single_query().mask_array(
            case_shape=batch.case_shape
        )[..., None].astype(output.dtype)
        if self.out_size == "scalar":
            return output[..., 0]
        return output

    def __call__(
        self,
        x: tuple[Array, Array, Array, Array, Array] | OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        del key
        if isinstance(x, OperatorBatch):
            return self.__call_operator_batch__(x)
        if not isinstance(x, tuple) or len(x) != 5:
            raise ValueError(
                "DPOT requires (history, axis_0, axis_1, history_time, forecast_time)."
            )
        spatial_axes = (
            OperatorAxis("axis_0", x[1]),
            OperatorAxis("axis_1", x[2]),
        )
        source_axes = spatial_axes + (OperatorAxis("history_time", x[3]),)
        query_axes = spatial_axes + (OperatorAxis("forecast_time", x[4]),)
        history = jnp.asarray(x[0])
        sample_suffix = self.image_shape + (self.history_steps,)
        channel_suffix = sample_suffix + (_get_size(self.in_size),)
        if self.in_size == "scalar" and history.shape[-3:] == sample_suffix:
            case_shape = history.shape[:-3]
        elif history.shape[-4:] == channel_suffix:
            case_shape = history.shape[:-4]
        else:
            raise ValueError(
                "DPOT history must end in its spatial, history, and optional channel axes."
            )
        name = self.source_key or "input"
        batch = OperatorBatch(
            inputs={name: FunctionSamples(values=history, axes=source_axes)},
            queries={"query": FunctionSamples(values=None, axes=query_axes)},
            case_axes=tuple(f"case_{index}" for index in range(len(case_shape))),
            case_shape=case_shape,
        )
        return self.__call_operator_batch__(batch)


__all__ = ["DPOT", "dpot_corrupt_history"]
