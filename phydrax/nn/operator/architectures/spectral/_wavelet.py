#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import sqrt
from typing import Literal

import equinox as eqx
import jax.nn as jnn
import jax.random as jr
import numpy as np
import opt_einsum as oe
from jax import core as jax_core
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax._spectral import (
    DiscreteWaveletTransform,
    MultiresolutionCoefficients,
    WaveletBoundary,
)
from phydrax._spectral._multiwavelet import AlpertMultiwaveletTransform
from phydrax._strict import StrictModule
from phydrax.nn._keys import EvalKey, fold_in_eval_key
from phydrax.nn._utils import _get_size
from phydrax.nn.layers._linear import Linear
from phydrax.nn.operator.data import FunctionSamples, OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel


class _WaveletSubbandMixerND(StrictModule):
    """Learned channel maps over one fixed tensor-wavelet coefficient topology."""

    scaling_weight: Array
    detail_weights: tuple[tuple[Array, ...], ...]
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    transform_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        transform: DiscreteWaveletTransform,
        /,
        *,
        in_channels: int,
        out_channels: int,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        in_size = int(in_channels)
        out_size = int(out_channels)
        if min(in_size, out_size) <= 0:
            raise ValueError("Wavelet mixer channels must be positive.")
        keys = iter(jr.split(key, 1 + transform.levels * transform.detail_count))
        scale = 1.0 / sqrt(float(in_size))
        self.scaling_weight = scale * jr.normal(next(keys), (out_size, in_size))
        self.detail_weights = tuple(
            tuple(
                scale * jr.normal(next(keys), (out_size, in_size))
                for _ in range(transform.detail_count)
            )
            for _ in range(transform.levels)
        )
        self.in_channels = in_size
        self.out_channels = out_size
        self.transform_fingerprint = transform.fingerprint

    @staticmethod
    def _mix(weight: Array, values: Array, /) -> Array:
        return oe.contract("oi,...i->...o", weight, values)

    def __call__(
        self, coefficients: MultiresolutionCoefficients, /
    ) -> MultiresolutionCoefficients:
        if coefficients.transform_fingerprint != self.transform_fingerprint:
            raise ValueError("Wavelet mixer and coefficient transforms do not match.")
        if int(coefficients.scaling.shape[-1]) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} coefficient channels; "
                f"got {coefficients.scaling.shape[-1]}."
            )
        scaling = self._mix(self.scaling_weight, coefficients.scaling)
        details = tuple(
            tuple(
                self._mix(weight, band)
                for weight, band in zip(weights, bands, strict=True)
            )
            for weights, bands in zip(
                self.detail_weights, coefficients.details, strict=True
            )
        )
        return coefficients.with_bands(scaling, details)


class _MultiwaveletSubbandMixer1D(StrictModule):
    """Learned polynomial/channel maps over one Alpert coefficient topology."""

    scaling_weight: Array
    detail_weights: tuple[Array, ...]
    order: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    transform_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        transform: AlpertMultiwaveletTransform,
        /,
        *,
        in_channels: int,
        out_channels: int,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        in_size = int(in_channels)
        out_size = int(out_channels)
        if min(in_size, out_size) <= 0:
            raise ValueError("Multiwavelet mixer channels must be positive.")
        keys = jr.split(key, transform.levels + 1)
        input_width = transform.order * in_size
        output_width = transform.order * out_size
        scale = 1.0 / sqrt(float(input_width))
        self.scaling_weight = scale * jr.normal(
            keys[0], (output_width, input_width)
        )
        self.detail_weights = tuple(
            scale * jr.normal(keys[index + 1], (output_width, input_width))
            for index in range(transform.levels)
        )
        self.order = transform.order
        self.in_channels = in_size
        self.out_channels = out_size
        self.transform_fingerprint = transform.fingerprint

    def _mix(self, weight: Array, values: Array, /) -> Array:
        flattened = values.reshape(
            values.shape[:-2] + (self.order * self.in_channels,)
        )
        mixed = oe.contract("oi,...i->...o", weight, flattened)
        return mixed.reshape(mixed.shape[:-1] + (self.order, self.out_channels))

    def __call__(
        self, coefficients: MultiresolutionCoefficients, /
    ) -> MultiresolutionCoefficients:
        if coefficients.transform_fingerprint != self.transform_fingerprint:
            raise ValueError("Multiwavelet mixer and coefficient transforms do not match.")
        scaling = self._mix(self.scaling_weight, coefficients.scaling)
        details = tuple(
            (self._mix(weight, bands[0]),)
            for weight, bands in zip(
                self.detail_weights, coefficients.details, strict=True
            )
        )
        return coefficients.with_bands(scaling, details)


def _grid_values(
    samples: FunctionSamples,
    case_shape: tuple[int, ...],
    channels: int,
    spatial_shape: tuple[int, ...],
    /,
) -> Array:
    if samples.values is None:
        raise ValueError("Grid operators require one array-valued source field.")
    values = samples.values
    scalar_shape = case_shape + spatial_shape
    if tuple(int(size) for size in values.shape) == scalar_shape and channels == 1:
        return values[..., None]
    expected = scalar_shape + (channels,)
    if tuple(int(size) for size in values.shape) != expected:
        raise ValueError(f"Grid source values must have shape {expected}; got {values.shape}.")
    return values


def _validate_uniform_axis(nodes: Array, /) -> None:
    if int(nodes.shape[0]) < 2:
        raise ValueError("Wavelet axes require at least two nodes.")
    if isinstance(nodes, jax_core.Tracer):
        return
    spacing = np.diff(np.asarray(nodes))
    if not np.allclose(spacing, spacing[0], rtol=1e-5, atol=1e-8):
        raise ValueError("Wavelet operators require uniformly spaced axes.")


def _validate_tensor_grid(
    source: FunctionSamples,
    query: FunctionSamples,
    spatial_ndim: int,
    boundaries: Sequence[WaveletBoundary],
    /,
) -> tuple[int, ...]:
    if not source.axes or not query.axes:
        raise ValueError(
            "Wavelet operators require tensor-product source and query axes."
        )
    if len(source.axes) != spatial_ndim or len(query.axes) != spatial_ndim:
        raise ValueError(f"Wavelet operators require {spatial_ndim} spatial axes.")
    spatial_shape = tuple(int(size) for size in source.sample_shape)
    if query.sample_shape != spatial_shape:
        raise ValueError("Wavelet source and query grids must have the same shape.")
    if source.axis_names != query.axis_names:
        raise ValueError("Wavelet source and query axis names must match.")
    for source_axis, query_axis, boundary in zip(
        source.axes, query.axes, boundaries, strict=True
    ):
        _validate_uniform_axis(source_axis.nodes)
        _validate_uniform_axis(query_axis.nodes)
        if source_axis.nodes.shape != query_axis.nodes.shape:
            raise ValueError("Wavelet source and query axis nodes must align.")
        if (
            not isinstance(source_axis.nodes, jax_core.Tracer)
            and not isinstance(query_axis.nodes, jax_core.Tracer)
            and not np.array_equal(
                np.asarray(source_axis.nodes), np.asarray(query_axis.nodes)
            )
        ):
            raise ValueError("Wavelet source and query grids must use identical nodes.")
        if boundary == "periodization" and not (
            source_axis.periodic and query_axis.periodic
        ):
            raise ValueError("Periodization requires periodic source and query axes.")
    return spatial_shape


class WaveletNeuralOperator(AbstractOperatorModel):
    """Resolution-variable WNO with exact separable wavelet reconstruction."""

    operator_architecture = "WaveletNeuralOperator"

    transform: DiscreteWaveletTransform
    lift: Linear
    wavelet_mixers: tuple[_WaveletSubbandMixerND, ...]
    pointwise_layers: tuple[Linear, ...]
    projection: Linear
    activation: Callable[[Array], Array]
    source_key: str | None
    spatial_ndim: int
    in_channels: int
    out_channels: int
    width: int
    depth: int
    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]

    def __init__(
        self,
        spatial_ndim: int,
        /,
        *,
        in_channels: int | Literal["scalar"],
        out_channels: int | Literal["scalar"],
        levels: int,
        wavelet: str | Sequence[str] = "haar",
        boundary: WaveletBoundary | Sequence[WaveletBoundary] = "periodization",
        width: int = 64,
        depth: int = 4,
        source_key: str | None = None,
        activation: Callable[[Array], Array] = jnn.gelu,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        dimension = int(spatial_ndim)
        if dimension not in (1, 2, 3):
            raise ValueError("WaveletNeuralOperator spatial_ndim must be 1, 2, or 3.")
        self.transform = DiscreteWaveletTransform(
            tuple(range(-dimension - 1, -1)),
            levels=levels,
            wavelet=wavelet,
            boundary=boundary,
        )
        self.spatial_ndim = dimension
        self.in_channels = _get_size(in_channels)
        self.out_channels = _get_size(out_channels)
        self.width = int(width)
        self.depth = int(depth)
        self.source_key = source_key
        self.activation = activation
        self.in_size = in_channels
        self.out_size = out_channels
        if min(self.in_channels, self.out_channels, self.width, self.depth) <= 0:
            raise ValueError("Wavelet operator dimensions must be positive.")
        keys = jr.split(key, 2 * self.depth + 2)
        self.lift = Linear(
            in_size=self.in_channels,
            out_size=self.width,
            activation=None,
            key=keys[0],
        )
        self.wavelet_mixers = tuple(
            _WaveletSubbandMixerND(
                self.transform,
                in_channels=self.width,
                out_channels=self.width,
                key=keys[1 + index],
            )
            for index in range(self.depth)
        )
        self.pointwise_layers = tuple(
            Linear(
                in_size=self.width,
                out_size=self.width,
                activation=None,
                key=keys[1 + self.depth + index],
            )
            for index in range(self.depth)
        )
        self.projection = Linear(
            in_size=self.width,
            out_size=self.out_channels,
            activation=None,
            key=keys[-1],
        )

    def _source(self, batch: OperatorBatch, /) -> FunctionSamples:
        if self.source_key is not None:
            return batch.input(self.source_key)
        if len(batch.inputs) != 1:
            raise ValueError(
                "WaveletNeuralOperator requires source_key for multiple inputs."
            )
        return next(iter(batch.inputs.values()))

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        source = self._source(batch)
        query = batch.require_single_query()
        spatial_shape = _validate_tensor_grid(
            source,
            query,
            self.spatial_ndim,
            self.transform.boundaries,
        )
        values = _grid_values(
            source, batch.case_shape, self.in_channels, spatial_shape
        )
        source_mask = source.mask_array(case_shape=batch.case_shape)
        hidden = self.lift(values * source_mask[..., None], key=fold_in_eval_key(key, 0))
        for index, (mixer, pointwise) in enumerate(
            zip(self.wavelet_mixers, self.pointwise_layers, strict=True)
        ):
            coefficients = self.transform.analysis(hidden)
            wavelet_update = self.transform.synthesis(mixer(coefficients))
            update = wavelet_update + pointwise(
                hidden, key=fold_in_eval_key(key, 2 * index + 1)
            )
            hidden = self.activation(hidden + update)
        output = self.projection(hidden, key=fold_in_eval_key(key, 2 * self.depth + 1))
        query_mask = query.mask_array(case_shape=batch.case_shape)
        output = output * query_mask[..., None]
        return output[..., 0] if self.out_size == "scalar" else output

    def __call__(
        self,
        x: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("WaveletNeuralOperator requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


class MultiwaveletOperator(AbstractOperatorModel):
    """Resolution-variable one-dimensional polynomial multiwavelet operator."""

    operator_architecture = "MultiwaveletOperator"

    transform: AlpertMultiwaveletTransform
    lift: Linear
    multiwavelet_mixers: tuple[_MultiwaveletSubbandMixer1D, ...]
    pointwise_layers: tuple[Linear, ...]
    projection: Linear
    activation: Callable[[Array], Array]
    source_key: str | None
    in_channels: int
    out_channels: int
    width: int
    depth: int
    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]

    def __init__(
        self,
        *,
        in_channels: int | Literal["scalar"],
        out_channels: int | Literal["scalar"],
        order: int = 3,
        levels: int = 3,
        boundary: WaveletBoundary = "periodization",
        width: int = 64,
        depth: int = 4,
        source_key: str | None = None,
        activation: Callable[[Array], Array] = jnn.gelu,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.transform = AlpertMultiwaveletTransform(
            order=order, levels=levels, boundary=boundary
        )
        self.in_channels = _get_size(in_channels)
        self.out_channels = _get_size(out_channels)
        self.width = int(width)
        self.depth = int(depth)
        self.source_key = source_key
        self.activation = activation
        self.in_size = in_channels
        self.out_size = out_channels
        if min(self.in_channels, self.out_channels, self.width, self.depth) <= 0:
            raise ValueError("Multiwavelet operator dimensions must be positive.")
        keys = jr.split(key, 2 * self.depth + 2)
        self.lift = Linear(
            in_size=self.in_channels,
            out_size=self.width,
            activation=None,
            key=keys[0],
        )
        self.multiwavelet_mixers = tuple(
            _MultiwaveletSubbandMixer1D(
                self.transform,
                in_channels=self.width,
                out_channels=self.width,
                key=keys[1 + index],
            )
            for index in range(self.depth)
        )
        self.pointwise_layers = tuple(
            Linear(
                in_size=self.width,
                out_size=self.width,
                activation=None,
                key=keys[1 + self.depth + index],
            )
            for index in range(self.depth)
        )
        self.projection = Linear(
            in_size=self.width,
            out_size=self.out_channels,
            activation=None,
            key=keys[-1],
        )

    def _source(self, batch: OperatorBatch, /) -> FunctionSamples:
        if self.source_key is not None:
            return batch.input(self.source_key)
        if len(batch.inputs) != 1:
            raise ValueError(
                "MultiwaveletOperator requires source_key for multiple inputs."
            )
        return next(iter(batch.inputs.values()))

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        source = self._source(batch)
        query = batch.require_single_query()
        spatial_shape = _validate_tensor_grid(
            source,
            query,
            1,
            (self.transform.boundary,),
        )
        values = _grid_values(
            source, batch.case_shape, self.in_channels, spatial_shape
        )
        source_mask = source.mask_array(case_shape=batch.case_shape)
        hidden = self.lift(values * source_mask[..., None], key=fold_in_eval_key(key, 0))
        for index, (mixer, pointwise) in enumerate(
            zip(self.multiwavelet_mixers, self.pointwise_layers, strict=True)
        ):
            coefficients = self.transform.analysis(hidden)
            wavelet_update = self.transform.synthesis(mixer(coefficients))
            update = wavelet_update + pointwise(
                hidden, key=fold_in_eval_key(key, 2 * index + 1)
            )
            hidden = self.activation(hidden + update)
        output = self.projection(hidden, key=fold_in_eval_key(key, 2 * self.depth + 1))
        query_mask = query.mask_array(case_shape=batch.case_shape)
        output = output * query_mask[..., None]
        return output[..., 0] if self.out_size == "scalar" else output

    def __call__(
        self,
        x: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("MultiwaveletOperator requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


__all__ = ["MultiwaveletOperator", "WaveletNeuralOperator"]
