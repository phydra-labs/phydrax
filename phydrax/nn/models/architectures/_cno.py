#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ...._strict import StrictModule
from ..._utils import _get_size
from ..core._base import _AbstractOperatorModel
from ..core._keys import EvalKey
from ..core._operator import OperatorAxis, OperatorBatch
from ..layers._linear import Linear
from ._fno import spectral_resample


CNOActivation = Literal["gelu", "silu", "tanh"]


def _activate(name: CNOActivation, values: Array, /) -> Array:
    if name == "gelu":
        return jax.nn.gelu(values)
    if name == "silu":
        return jax.nn.silu(values)
    if name == "tanh":
        return jnp.tanh(values)
    raise ValueError("activation must be 'gelu', 'silu', or 'tanh'.")


def _prepare_grid_values(
    values: Array,
    axes: tuple[OperatorAxis, ...],
    channels: int,
    /,
) -> tuple[Array, tuple[int, ...]]:
    array = jnp.asarray(values)
    shape = tuple(axis.size for axis in axes)
    ndim = len(shape)
    if array.ndim >= ndim and tuple(array.shape[-ndim:]) == shape:
        if channels != 1:
            raise ValueError(f"Expected {channels} input channels, got scalar values.")
        array = array[..., None]
    elif array.ndim <= ndim or tuple(array.shape[-ndim - 1 : -1]) != shape:
        raise ValueError(
            f"Grid values must contain spatial shape {shape}; got {array.shape}."
        )
    if int(array.shape[-1]) != channels:
        raise ValueError(f"Expected {channels} input channels, got {array.shape[-1]}.")
    case_shape = tuple(int(size) for size in array.shape[: -ndim - 1])
    return array, case_shape


def _coordinate_features(
    axes: tuple[OperatorAxis, ...],
    case_shape: tuple[int, ...],
    /,
) -> Array:
    normalized = []
    for axis in axes:
        span = axis.nodes[-1] - axis.nodes[0]
        normalized.append(2.0 * (axis.nodes - axis.nodes[0]) / span - 1.0)
    grids = jnp.meshgrid(*normalized, indexing="ij")
    coordinates = jnp.stack(grids, axis=-1)
    return jnp.broadcast_to(coordinates, case_shape + coordinates.shape)


def _operator_source(batch: OperatorBatch, source_key: str | None, /):
    if source_key is not None:
        return batch.input(source_key)
    if len(batch.inputs) != 1:
        raise ValueError("source_key is required for multiple operator inputs.")
    return next(iter(batch.inputs.values()))


class AntiAliasedConvND(StrictModule):
    """Channels-last N-D convolution with alias-free oversampled activation."""

    weight: Array
    bias: Array
    in_channels: int
    out_channels: int
    spatial_ndim: int
    kernel_size: tuple[int, ...]
    activation: CNOActivation | None
    oversample_factor: int

    def __init__(
        self,
        *,
        spatial_ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        activation: CNOActivation | None = "gelu",
        oversample_factor: int = 2,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.spatial_ndim = int(spatial_ndim)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if isinstance(kernel_size, int):
            self.kernel_size = (int(kernel_size),) * self.spatial_ndim
        else:
            self.kernel_size = tuple(int(size) for size in kernel_size)
        self.activation = activation
        self.oversample_factor = int(oversample_factor)
        if self.spatial_ndim not in (1, 2, 3):
            raise ValueError("AntiAliasedConvND supports one, two, or three dimensions.")
        if len(self.kernel_size) != self.spatial_ndim or any(
            size <= 0 or size % 2 == 0 for size in self.kernel_size
        ):
            raise ValueError("kernel_size must contain positive odd sizes per axis.")
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if self.oversample_factor < 1:
            raise ValueError("oversample_factor must be at least one.")
        weight_key, _ = jr.split(key)
        scale = 1.0 / jnp.sqrt(float(prod(self.kernel_size) * self.in_channels))
        self.weight = scale * jr.normal(
            weight_key,
            shape=self.kernel_size + (self.in_channels, self.out_channels),
        )
        self.bias = jnp.zeros((self.out_channels,), dtype=float)

    def _convolve(self, values: Array, /) -> Array:
        spatial = {1: "W", 2: "HW", 3: "DHW"}[self.spatial_ndim]
        array = jnp.asarray(values)
        spatial_shape = tuple(
            int(size) for size in array.shape[-self.spatial_ndim - 1 : -1]
        )
        case_shape = tuple(int(size) for size in array.shape[: -self.spatial_ndim - 1])
        batched = array.reshape(
            (prod(case_shape) if case_shape else 1,) + spatial_shape + (self.in_channels,)
        )
        output = jax.lax.conv_general_dilated(
            batched,
            self.weight,
            window_strides=(1,) * self.spatial_ndim,
            padding="SAME",
            dimension_numbers=(f"N{spatial}C", f"{spatial}IO", f"N{spatial}C"),
        )
        return (output + self.bias).reshape(
            case_shape + spatial_shape + (self.out_channels,)
        )

    def __call__(self, values: Array, /) -> Array:
        output = self._convolve(values)
        if self.activation is None:
            return output
        shape = tuple(int(size) for size in output.shape[-self.spatial_ndim - 1 : -1])
        if self.oversample_factor == 1:
            return _activate(self.activation, output)
        fine_shape = tuple(self.oversample_factor * size for size in shape)
        fine = spectral_resample(output, fine_shape)
        activated = _activate(self.activation, fine)
        return spectral_resample(activated, shape)


class _CNOBlock(StrictModule):
    first: AntiAliasedConvND
    second: AntiAliasedConvND
    skip: Linear | None

    def __init__(
        self,
        *,
        spatial_ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        activation: CNOActivation,
        oversample_factor: int,
        key: Key[Array, ""],
    ):
        first_key, second_key, skip_key = jr.split(key, 3)
        self.first = AntiAliasedConvND(
            spatial_ndim=spatial_ndim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation,
            oversample_factor=oversample_factor,
            key=first_key,
        )
        self.second = AntiAliasedConvND(
            spatial_ndim=spatial_ndim,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=None,
            oversample_factor=oversample_factor,
            key=second_key,
        )
        self.skip = (
            None
            if in_channels == out_channels
            else Linear(
                in_size=in_channels,
                out_size=out_channels,
                activation=None,
                key=skip_key,
            )
        )

    def __call__(self, values: Array, /) -> Array:
        residual = values if self.skip is None else self.skip(values)
        return (residual + self.second(self.first(values))) / jnp.sqrt(2.0)


class CNO(_AbstractOperatorModel):
    """N-dimensional anti-aliased Convolutional Neural Operator."""

    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    spatial_ndim: int
    width: int
    coordinate_embedding: bool
    source_key: str | None
    lift: Linear
    blocks: tuple[_CNOBlock, ...]
    projection: Linear

    def __init__(
        self,
        *,
        spatial_ndim: int,
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        width: int = 32,
        depth: int = 4,
        kernel_size: int | Sequence[int] = 3,
        activation: CNOActivation = "gelu",
        oversample_factor: int = 2,
        coordinate_embedding: bool = True,
        source_key: str | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_size = in_channels
        self.out_size = out_channels
        self.spatial_ndim = int(spatial_ndim)
        self.width = int(width)
        self.coordinate_embedding = bool(coordinate_embedding)
        self.source_key = source_key
        if self.spatial_ndim not in (1, 2, 3) or self.width <= 0 or int(depth) <= 0:
            raise ValueError("spatial_ndim must be 1-3 and width/depth must be positive.")
        keys = jr.split(key, int(depth) + 2)
        lifted = _get_size(in_channels) + (
            self.spatial_ndim if coordinate_embedding else 0
        )
        self.lift = Linear(
            in_size=lifted, out_size=self.width, activation=None, key=keys[0]
        )
        self.blocks = tuple(
            _CNOBlock(
                spatial_ndim=self.spatial_ndim,
                in_channels=self.width,
                out_channels=self.width,
                kernel_size=kernel_size,
                activation=activation,
                oversample_factor=oversample_factor,
                key=block_key,
            )
            for block_key in keys[1:-1]
        )
        self.projection = Linear(
            in_size=self.width,
            out_size=_get_size(out_channels),
            activation=None,
            key=keys[-1],
        )

    def _evaluate(self, values: Array, axes: tuple[OperatorAxis, ...], /) -> Array:
        if len(axes) != self.spatial_ndim:
            raise ValueError(f"CNO expects {self.spatial_ndim} spatial axes.")
        array, case_shape = _prepare_grid_values(values, axes, _get_size(self.in_size))
        if self.coordinate_embedding:
            array = jnp.concatenate(
                (array, _coordinate_features(axes, case_shape)), axis=-1
            )
        hidden = self.lift(array)
        for block in self.blocks:
            hidden = block(hidden)
        output = self.projection(hidden)
        if self.out_size == "scalar":
            return output[..., 0]
        return output

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        del key
        source = _operator_source(batch, self.source_key)
        axes = source.axes or batch.require_single_query().axes
        if not axes or source.values is None:
            raise ValueError("CNO requires tensor-grid source values and query axes.")
        if source.axes and source.sample_shape != batch.require_single_query().sample_shape:
            raise ValueError("CNO requires coincident source and query grids.")
        return self._evaluate(jnp.asarray(source.values), axes)

    def __call__(
        self,
        x: Array | tuple[Array, ...] | OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        del key
        if isinstance(x, OperatorBatch):
            return self.__call_operator_batch__(x)
        if not isinstance(x, tuple) or len(x) != self.spatial_ndim + 1:
            raise ValueError("CNO requires (values, axis_0, ..., axis_d).")
        axes = tuple(
            OperatorAxis(f"axis_{index}", nodes) for index, nodes in enumerate(x[1:])
        )
        return self._evaluate(jnp.asarray(x[0]), axes)


class UNO(_AbstractOperatorModel):
    """U-shaped anti-aliased neural operator with band-limited resampling."""

    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    spatial_ndim: int
    widths: tuple[int, ...]
    source_key: str | None
    coordinate_embedding: bool
    lift: Linear
    encoders: tuple[_CNOBlock, ...]
    mergers: tuple[Linear, ...]
    decoders: tuple[_CNOBlock, ...]
    projection: Linear

    def __init__(
        self,
        *,
        spatial_ndim: int,
        widths: Sequence[int] = (16, 32, 64),
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        kernel_size: int | Sequence[int] = 3,
        activation: CNOActivation = "gelu",
        oversample_factor: int = 2,
        coordinate_embedding: bool = True,
        source_key: str | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_size = in_channels
        self.out_size = out_channels
        self.spatial_ndim = int(spatial_ndim)
        self.widths = tuple(int(width) for width in widths)
        self.source_key = source_key
        self.coordinate_embedding = bool(coordinate_embedding)
        if self.spatial_ndim not in (1, 2, 3):
            raise ValueError("UNO supports one, two, or three spatial dimensions.")
        if len(self.widths) < 2 or any(width <= 0 for width in self.widths):
            raise ValueError("UNO widths must contain at least two positive levels.")
        keys = jr.split(key, 3 * len(self.widths))
        lifted = _get_size(in_channels) + (
            self.spatial_ndim if coordinate_embedding else 0
        )
        self.lift = Linear(
            in_size=lifted,
            out_size=self.widths[0],
            activation=None,
            key=keys[0],
        )
        encoders = []
        current = self.widths[0]
        for index, width in enumerate(self.widths):
            encoders.append(
                _CNOBlock(
                    spatial_ndim=self.spatial_ndim,
                    in_channels=current,
                    out_channels=width,
                    kernel_size=kernel_size,
                    activation=activation,
                    oversample_factor=oversample_factor,
                    key=keys[1 + index],
                )
            )
            current = width
        self.encoders = tuple(encoders)

        mergers = []
        decoders = []
        key_offset = 1 + len(self.widths)
        current = self.widths[-1]
        for index, width in enumerate(reversed(self.widths[:-1])):
            mergers.append(
                Linear(
                    in_size=current + width,
                    out_size=width,
                    activation=None,
                    key=keys[key_offset + 2 * index],
                )
            )
            decoders.append(
                _CNOBlock(
                    spatial_ndim=self.spatial_ndim,
                    in_channels=width,
                    out_channels=width,
                    kernel_size=kernel_size,
                    activation=activation,
                    oversample_factor=oversample_factor,
                    key=keys[key_offset + 2 * index + 1],
                )
            )
            current = width
        self.mergers = tuple(mergers)
        self.decoders = tuple(decoders)
        self.projection = Linear(
            in_size=self.widths[0],
            out_size=_get_size(out_channels),
            activation=None,
            key=keys[-1],
        )

    def _evaluate(self, values: Array, axes: tuple[OperatorAxis, ...], /) -> Array:
        if len(axes) != self.spatial_ndim:
            raise ValueError(f"UNO expects {self.spatial_ndim} spatial axes.")
        array, case_shape = _prepare_grid_values(values, axes, _get_size(self.in_size))
        if self.coordinate_embedding:
            array = jnp.concatenate(
                (array, _coordinate_features(axes, case_shape)), axis=-1
            )
        hidden = self.lift(array)
        skips = []
        shapes = []
        for index, encoder in enumerate(self.encoders):
            hidden = encoder(hidden)
            skips.append(hidden)
            shape = tuple(int(size) for size in hidden.shape[-self.spatial_ndim - 1 : -1])
            shapes.append(shape)
            if index < len(self.encoders) - 1:
                hidden = spectral_resample(
                    hidden,
                    tuple(max(2, (size + 1) // 2) for size in shape),
                )

        for decoder, merger, skip, shape in zip(
            self.decoders,
            self.mergers,
            reversed(skips[:-1]),
            reversed(shapes[:-1]),
            strict=True,
        ):
            hidden = spectral_resample(hidden, shape)
            hidden = merger(jnp.concatenate((hidden, skip), axis=-1))
            hidden = decoder(hidden)
        output = self.projection(hidden)
        if self.out_size == "scalar":
            return output[..., 0]
        return output

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        del key
        source = _operator_source(batch, self.source_key)
        axes = source.axes or batch.require_single_query().axes
        if not axes or source.values is None:
            raise ValueError("UNO requires tensor-grid source values and query axes.")
        if source.axes and source.sample_shape != batch.require_single_query().sample_shape:
            raise ValueError("UNO requires coincident source and query grids.")
        return self._evaluate(jnp.asarray(source.values), axes)

    def __call__(
        self,
        x: Array | tuple[Array, ...] | OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        del key
        if isinstance(x, OperatorBatch):
            return self.__call_operator_batch__(x)
        if not isinstance(x, tuple) or len(x) != self.spatial_ndim + 1:
            raise ValueError("UNO requires (values, axis_0, ..., axis_d).")
        axes = tuple(
            OperatorAxis(f"axis_{index}", nodes) for index, nodes in enumerate(x[1:])
        )
        return self._evaluate(jnp.asarray(x[0]), axes)


__all__ = ["AntiAliasedConvND", "CNO", "UNO"]
