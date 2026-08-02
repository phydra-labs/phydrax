#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product
from math import sqrt
from typing import Literal

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import opt_einsum as oe
from jax import core as jax_core
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ...._trainable import NonTrainableState
from ..._utils import _get_size
from ..core._base import _AbstractOperatorModel
from ..core._keys import EvalKey, fold_in_eval_key
from ..core._operator import FunctionSamples, OperatorBatch
from ..layers._linear import Linear


WaveletBasis = Literal["haar", "db2", "db4"]
WaveletBoundary = Literal["periodic", "symmetric", "zero"]


def _wavelet_filter(name: WaveletBasis, /) -> np.ndarray:
    if name == "haar":
        return np.asarray((1.0, 1.0), dtype=float) / sqrt(2.0)
    if name == "db2":
        root_three = sqrt(3.0)
        denominator = 4.0 * sqrt(2.0)
        return np.asarray(
            (
                (1.0 + root_three) / denominator,
                (3.0 + root_three) / denominator,
                (3.0 - root_three) / denominator,
                (1.0 - root_three) / denominator,
            ),
            dtype=float,
        )
    if name == "db4":
        return np.asarray(
            (
                0.2303778133088964,
                0.7148465705529154,
                0.6308807679298587,
                -0.02798376941685985,
                -0.18703481171888114,
                0.030841381835560764,
                0.0328830116668852,
                -0.010597401785069032,
            ),
            dtype=float,
        )
    raise ValueError(f"Unknown wavelet basis {name!r}.")


def _symmetric_index(index: int, size: int, /) -> int:
    value = int(index)
    while value < 0 or value >= size:
        value = -value - 1 if value < 0 else 2 * size - value - 1
    return value


def _analysis_matrix(
    size: int,
    wavelet: WaveletBasis,
    boundary: WaveletBoundary,
    /,
) -> tuple[np.ndarray, np.ndarray, int]:
    if int(size) <= 0 or int(size) % 2:
        raise ValueError("Wavelet analysis matrices require a positive even size.")
    low_filter = _wavelet_filter(wavelet)
    high_filter = np.asarray(
        [(-1.0) ** index * low_filter[::-1][index] for index in range(len(low_filter))]
    )
    best: tuple[float, int, np.ndarray] | None = None
    for offset in range(len(low_filter)):
        matrix = np.zeros((size, size), dtype=float)
        for output_index in range(size // 2):
            for tap, (low_value, high_value) in enumerate(
                zip(low_filter, high_filter, strict=True)
            ):
                input_index = 2 * output_index + tap - offset
                if boundary == "periodic":
                    input_index %= size
                elif boundary == "symmetric":
                    input_index = _symmetric_index(input_index, size)
                elif input_index < 0 or input_index >= size:
                    continue
                matrix[output_index, input_index] += low_value
                matrix[size // 2 + output_index, input_index] += high_value
        if np.linalg.matrix_rank(matrix) != size:
            continue
        condition = float(np.linalg.cond(matrix))
        if best is None or condition < best[0]:
            best = (condition, offset, matrix)
    if best is None:
        raise ValueError(
            f"Could not construct an invertible {wavelet} transform of size {size} "
            f"with {boundary} boundaries."
        )
    _, offset, analysis = best
    return analysis, np.linalg.inv(analysis), offset


def _pad_right(values: Array, spatial_axis: int, boundary: WaveletBoundary, /) -> Array:
    pads = [(0, 0)] * values.ndim
    pads[spatial_axis] = (0, 1)
    if boundary == "periodic":
        return jnp.pad(values, pads, mode="wrap")
    if boundary == "symmetric":
        return jnp.pad(values, pads, mode="symmetric")
    return jnp.pad(values, pads, mode="constant")


def _apply_axis_matrix(values: Array, matrix: Array, spatial_axis: int, /) -> Array:
    moved = jnp.moveaxis(values, spatial_axis, -2)
    transformed = oe.contract("ij,...jc->...ic", matrix, moved)
    return jnp.moveaxis(transformed, -2, spatial_axis)


class WaveletLevelPlan(eqx.Module, NonTrainableState):
    """Fixed invertible filter-bank maps for one multiresolution level."""

    analysis_matrices: tuple[Array, ...]
    synthesis_matrices: tuple[Array, ...]
    input_shape: tuple[int, ...] = eqx.field(static=True)
    padded_shape: tuple[int, ...] = eqx.field(static=True)
    offsets: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        input_shape: Sequence[int],
        /,
        *,
        wavelet: WaveletBasis,
        boundary: WaveletBoundary,
    ):
        shape = tuple(int(size) for size in input_shape)
        padded = tuple(size + size % 2 for size in shape)
        matrices = tuple(_analysis_matrix(size, wavelet, boundary) for size in padded)
        self.analysis_matrices = tuple(jnp.asarray(item[0]) for item in matrices)
        self.synthesis_matrices = tuple(jnp.asarray(item[1]) for item in matrices)
        self.input_shape = shape
        self.padded_shape = padded
        self.offsets = tuple(item[2] for item in matrices)

    @property
    def subband_shape(self) -> tuple[int, ...]:
        return tuple(size // 2 for size in self.padded_shape)


class WaveletCoefficients(eqx.Module):
    """Coarsest scaling coefficients and fine-to-coarse detail subbands."""

    low: Array
    details: tuple[tuple[Array, ...], ...]


class MultiresolutionTransform(eqx.Module, NonTrainableState):
    """Invertible separable N-D orthogonal-wavelet transform with explicit edges."""

    level_plans: tuple[WaveletLevelPlan, ...]
    spatial_shape: tuple[int, ...] = eqx.field(static=True)
    wavelet: WaveletBasis = eqx.field(static=True)
    boundary: WaveletBoundary = eqx.field(static=True)
    levels: int = eqx.field(static=True)

    def __init__(
        self,
        spatial_shape: Sequence[int],
        /,
        *,
        levels: int,
        wavelet: WaveletBasis = "haar",
        boundary: WaveletBoundary = "periodic",
    ):
        shape = tuple(int(size) for size in spatial_shape)
        if not shape or any(size <= 1 for size in shape):
            raise ValueError("Wavelet spatial dimensions must each exceed one.")
        if int(levels) <= 0:
            raise ValueError("Wavelet levels must be positive.")
        if wavelet not in ("haar", "db2", "db4"):
            raise ValueError("wavelet must be 'haar', 'db2', or 'db4'.")
        if boundary not in ("periodic", "symmetric", "zero"):
            raise ValueError("boundary must be 'periodic', 'symmetric', or 'zero'.")
        plans: list[WaveletLevelPlan] = []
        current = shape
        for _ in range(int(levels)):
            if any(size <= 1 for size in current):
                raise ValueError("Too many wavelet levels for the configured shape.")
            plan = WaveletLevelPlan(current, wavelet=wavelet, boundary=boundary)
            plans.append(plan)
            current = plan.subband_shape
        self.level_plans = tuple(plans)
        self.spatial_shape = shape
        self.wavelet = wavelet
        self.boundary = boundary
        self.levels = int(levels)

    @property
    def spatial_ndim(self) -> int:
        return len(self.spatial_shape)

    @property
    def detail_count(self) -> int:
        return 2**self.spatial_ndim - 1

    @property
    def subband_labels(self) -> tuple[tuple[int, ...], ...]:
        zero = (0,) * self.spatial_ndim
        return tuple(
            label for label in product((0, 1), repeat=self.spatial_ndim) if label != zero
        )

    def analysis(self, values: Array, /) -> WaveletCoefficients:
        low = jnp.asarray(values)
        if low.ndim < self.spatial_ndim + 1:
            raise ValueError("Wavelet values require spatial axes and a channel axis.")
        if (
            tuple(int(size) for size in low.shape[-self.spatial_ndim - 1 : -1])
            != self.spatial_shape
        ):
            raise ValueError(
                f"Wavelet values must end in spatial/channel shape {self.spatial_shape} + "
                f"(channels,); got {low.shape}."
            )
        detail_levels: list[tuple[Array, ...]] = []
        zero = (0,) * self.spatial_ndim
        for plan in self.level_plans:
            spatial_start = low.ndim - self.spatial_ndim - 1
            for axis, (input_size, padded_size) in enumerate(
                zip(plan.input_shape, plan.padded_shape, strict=True)
            ):
                if padded_size != input_size:
                    low = _pad_right(low, spatial_start + axis, self.boundary)
            transformed = low
            for axis, matrix in enumerate(plan.analysis_matrices):
                transformed = _apply_axis_matrix(
                    transformed, matrix, spatial_start + axis
                )
            bands: dict[tuple[int, ...], Array] = {}
            for label in product((0, 1), repeat=self.spatial_ndim):
                selection: list[slice] = [slice(None)] * transformed.ndim
                for axis, high in enumerate(label):
                    half = plan.padded_shape[axis] // 2
                    selection[spatial_start + axis] = (
                        slice(half, 2 * half) if high else slice(0, half)
                    )
                bands[label] = transformed[tuple(selection)]
            low = bands[zero]
            detail_levels.append(tuple(bands[label] for label in self.subband_labels))
        return WaveletCoefficients(low=low, details=tuple(detail_levels))

    def synthesis(self, coefficients: WaveletCoefficients, /) -> Array:
        if len(coefficients.details) != self.levels:
            raise ValueError("Wavelet coefficient depth does not match this transform.")
        low = jnp.asarray(coefficients.low)
        labels = self.subband_labels
        zero = (0,) * self.spatial_ndim
        for plan, detail in zip(
            reversed(self.level_plans), reversed(coefficients.details), strict=True
        ):
            if len(detail) != self.detail_count:
                raise ValueError("Wavelet detail count does not match spatial rank.")
            bands = {zero: low}
            bands.update(zip(labels, detail, strict=True))
            spatial_start = low.ndim - self.spatial_ndim - 1

            def combine(axis: int, prefix: tuple[int, ...]) -> Array:
                if axis == self.spatial_ndim:
                    return bands[prefix]
                lower = combine(axis + 1, prefix + (0,))
                upper = combine(axis + 1, prefix + (1,))
                return jnp.concatenate((lower, upper), axis=spatial_start + axis)

            reconstructed = combine(0, ())
            for axis in reversed(range(self.spatial_ndim)):
                reconstructed = _apply_axis_matrix(
                    reconstructed,
                    plan.synthesis_matrices[axis],
                    spatial_start + axis,
                )
            selection = [slice(None)] * reconstructed.ndim
            for axis, size in enumerate(plan.input_shape):
                selection[spatial_start + axis] = slice(0, size)
            low = reconstructed[tuple(selection)]
        return low

    def __call__(self, values: Array, /) -> WaveletCoefficients:
        return self.analysis(values)


class WaveletSpectralConvND(eqx.Module):
    """Learned channel maps applied independently to every wavelet subband."""

    transform: MultiresolutionTransform
    low_weight: Array
    detail_weights: tuple[tuple[Array, ...], ...]
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    def __init__(
        self,
        transform: MultiresolutionTransform,
        /,
        *,
        in_channels: int,
        out_channels: int,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.transform = transform
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if min(self.in_channels, self.out_channels) <= 0:
            raise ValueError("Wavelet convolution channels must be positive.")
        count = 1 + transform.levels * transform.detail_count
        keys = iter(jr.split(key, count))
        scale = 1.0 / sqrt(float(self.in_channels))
        self.low_weight = scale * jr.normal(
            next(keys), (self.out_channels, self.in_channels)
        )
        self.detail_weights = tuple(
            tuple(
                scale * jr.normal(next(keys), (self.out_channels, self.in_channels))
                for _ in range(transform.detail_count)
            )
            for _ in range(transform.levels)
        )

    @staticmethod
    def _mix(weight: Array, values: Array, /) -> Array:
        return oe.contract("oi,...i->...o", weight, values)

    def __call__(self, values: Array, /) -> Array:
        array = jnp.asarray(values)
        if int(array.shape[-1]) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} wavelet input channels; "
                f"got {array.shape[-1]}."
            )
        coefficients = self.transform.analysis(array)
        low = self._mix(self.low_weight, coefficients.low)
        details = tuple(
            tuple(
                self._mix(weight, band)
                for weight, band in zip(weights, bands, strict=True)
            )
            for weights, bands in zip(
                self.detail_weights, coefficients.details, strict=True
            )
        )
        return self.transform.synthesis(WaveletCoefficients(low=low, details=details))


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
        raise ValueError(
            f"Grid source values must have shape {expected}; got {values.shape}."
        )
    return values


def _validate_tensor_grid(
    source: FunctionSamples,
    query: FunctionSamples,
    spatial_shape: tuple[int, ...],
    /,
) -> None:
    if not source.axes or not query.axes:
        raise ValueError(
            "Wavelet operators require tensor-product source and query axes."
        )
    if source.sample_shape != spatial_shape or query.sample_shape != spatial_shape:
        raise ValueError(
            f"Wavelet source/query grids must both have shape {spatial_shape}."
        )
    if source.axis_names != query.axis_names:
        raise ValueError("Wavelet source and query axis names must match.")
    for source_axis, query_axis in zip(source.axes, query.axes, strict=True):
        if source_axis.nodes.shape != query_axis.nodes.shape:
            raise ValueError("Wavelet source and query axis nodes must align.")
        if not isinstance(source_axis.nodes, jax_core.Tracer) and not bool(
            jnp.array_equal(source_axis.nodes, query_axis.nodes)
        ):
            raise ValueError("Wavelet source and query grids must use identical nodes.")


class WaveletNeuralOperator(_AbstractOperatorModel):
    """WNO on a fixed tensor grid with exact multiresolution reconstruction."""

    transform: MultiresolutionTransform
    lift: Linear
    wavelet_layers: tuple[WaveletSpectralConvND, ...]
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
        spatial_shape: Sequence[int],
        /,
        *,
        in_channels: int | Literal["scalar"],
        out_channels: int | Literal["scalar"],
        levels: int,
        wavelet: WaveletBasis = "haar",
        boundary: WaveletBoundary = "periodic",
        width: int = 64,
        depth: int = 4,
        source_key: str | None = None,
        activation: Callable[[Array], Array] = jnn.gelu,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.transform = MultiresolutionTransform(
            spatial_shape, levels=levels, wavelet=wavelet, boundary=boundary
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
            raise ValueError("Wavelet operator dimensions must be positive.")
        keys = jr.split(key, 2 * self.depth + 2)
        self.lift = Linear(
            in_size=self.in_channels,
            out_size=self.width,
            activation=None,
            key=keys[0],
        )
        self.wavelet_layers = tuple(
            WaveletSpectralConvND(
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
        _validate_tensor_grid(
            source, batch.require_single_query(), self.transform.spatial_shape
        )
        values = _grid_values(
            source,
            batch.case_shape,
            self.in_channels,
            self.transform.spatial_shape,
        )
        source_mask = source.mask_array(case_shape=batch.case_shape)
        hidden = self.lift(values * source_mask[..., None], key=fold_in_eval_key(key, 0))
        for index, (wavelet_layer, pointwise_layer) in enumerate(
            zip(self.wavelet_layers, self.pointwise_layers, strict=True)
        ):
            update = wavelet_layer(hidden) + pointwise_layer(
                hidden, key=fold_in_eval_key(key, 2 * index + 1)
            )
            hidden = self.activation(hidden + update)
        output = self.projection(hidden, key=fold_in_eval_key(key, 2 * self.depth + 1))
        query_mask = batch.require_single_query().mask_array(case_shape=batch.case_shape)
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


def _canonical_columns(matrix: np.ndarray, /) -> np.ndarray:
    result = np.asarray(matrix, dtype=float).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def _discrete_legendre_analysis(order: int, /) -> np.ndarray:
    nodes = (np.arange(order, dtype=float) + 0.5) / float(order)
    vandermonde = np.polynomial.legendre.legvander(2.0 * nodes - 1.0, order - 1)
    orthogonal, _ = np.linalg.qr(vandermonde)
    return _canonical_columns(orthogonal).T


def _alpert_analysis(order: int, /) -> np.ndarray:
    quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(
        max(16, 4 * order)
    )
    low_rows = np.zeros((order, 2 * order), dtype=float)
    for branch in range(2):
        lower = 0.5 * branch
        points = lower + 0.25 * (quadrature_nodes + 1.0)
        weights = 0.25 * quadrature_weights
        coarse_values = np.stack(
            [
                sqrt(2 * degree + 1)
                * np.polynomial.legendre.legval(
                    2.0 * points - 1.0,
                    [0.0] * degree + [1.0],
                )
                for degree in range(order)
            ]
        )
        local_coordinate = 4.0 * points - (1.0 if branch == 0 else 3.0)
        fine_values = np.stack(
            [
                sqrt(2.0)
                * sqrt(2 * degree + 1)
                * np.polynomial.legendre.legval(
                    local_coordinate,
                    [0.0] * degree + [1.0],
                )
                for degree in range(order)
            ]
        )
        low_rows[:, branch * order : (branch + 1) * order] = (
            coarse_values * weights[None, :]
        ) @ fine_values.T
    left, _, right = np.linalg.svd(low_rows, full_matrices=True)
    low_rows = left @ right[:order]
    high_rows = right[order:]
    analysis = np.concatenate((low_rows, high_rows), axis=0)
    for row in range(analysis.shape[0]):
        pivot = int(np.argmax(np.abs(analysis[row])))
        if analysis[row, pivot] < 0.0:
            analysis[row] *= -1.0
    return analysis


class MultiwaveletCoefficients(eqx.Module):
    """Alpert scaling coefficients and one polynomial detail bank per level."""

    low: Array
    details: tuple[Array, ...]


class AlpertMultiwaveletTransform(eqx.Module, NonTrainableState):
    """Exact 1-D Alpert-style polynomial multiwavelet transform."""

    base_analysis: Array
    base_synthesis: Array
    level_analysis: Array
    level_synthesis: Array
    num_points: int = eqx.field(static=True)
    padded_points: int = eqx.field(static=True)
    order: int = eqx.field(static=True)
    levels: int = eqx.field(static=True)
    boundary: WaveletBoundary = eqx.field(static=True)

    def __init__(
        self,
        num_points: int,
        /,
        *,
        order: int = 3,
        levels: int = 3,
        boundary: WaveletBoundary = "periodic",
    ):
        self.num_points = int(num_points)
        self.order = int(order)
        self.levels = int(levels)
        self.boundary = boundary
        if min(self.num_points, self.order, self.levels) <= 0:
            raise ValueError("Multiwavelet dimensions must be positive.")
        if boundary not in ("periodic", "symmetric", "zero"):
            raise ValueError("boundary must be 'periodic', 'symmetric', or 'zero'.")
        multiple = self.order * 2**self.levels
        self.padded_points = ((self.num_points + multiple - 1) // multiple) * multiple
        base = _discrete_legendre_analysis(self.order)
        level = _alpert_analysis(self.order)
        self.base_analysis = jnp.asarray(base)
        self.base_synthesis = jnp.asarray(base.T)
        self.level_analysis = jnp.asarray(level)
        self.level_synthesis = jnp.asarray(level.T)

    def _pad(self, values: Array, /) -> Array:
        amount = self.padded_points - self.num_points
        if amount == 0:
            return values
        pads = [(0, 0)] * values.ndim
        pads[-2] = (0, amount)
        if self.boundary == "periodic":
            return jnp.pad(values, pads, mode="wrap")
        if self.boundary == "symmetric":
            return jnp.pad(values, pads, mode="symmetric")
        return jnp.pad(values, pads, mode="constant")

    def analysis(self, values: Array, /) -> MultiwaveletCoefficients:
        array = jnp.asarray(values)
        if array.ndim < 2 or int(array.shape[-2]) != self.num_points:
            raise ValueError(
                f"Multiwavelet values must end in ({self.num_points}, channels)."
            )
        padded = self._pad(array)
        cells = self.padded_points // self.order
        samples = padded.reshape(
            padded.shape[:-2] + (cells, self.order, padded.shape[-1])
        )
        low = oe.contract("mp,...cpi->...cmi", self.base_analysis, samples)
        details: list[Array] = []
        for _ in range(self.levels):
            cells = int(low.shape[-3])
            paired = low.reshape(
                low.shape[:-3] + (cells // 2, 2 * self.order, low.shape[-1])
            )
            transformed = oe.contract("mn,...pni->...pmi", self.level_analysis, paired)
            low = transformed[..., : self.order, :]
            details.append(transformed[..., self.order :, :])
        return MultiwaveletCoefficients(low=low, details=tuple(details))

    def synthesis(self, coefficients: MultiwaveletCoefficients, /) -> Array:
        if len(coefficients.details) != self.levels:
            raise ValueError("Multiwavelet coefficient depth does not match transform.")
        low = jnp.asarray(coefficients.low)
        for detail in reversed(coefficients.details):
            merged = jnp.concatenate((low, detail), axis=-2)
            fine = oe.contract("nm,...pmi->...pni", self.level_synthesis, merged)
            low = fine.reshape(
                fine.shape[:-3]
                + (int(fine.shape[-3]) * 2, self.order, int(fine.shape[-1]))
            )
        samples = oe.contract("pm,...cmi->...cpi", self.base_synthesis, low)
        output = samples.reshape(
            samples.shape[:-3] + (self.padded_points, samples.shape[-1])
        )
        return output[..., : self.num_points, :]

    def __call__(self, values: Array, /) -> MultiwaveletCoefficients:
        return self.analysis(values)


class MultiwaveletSpectralConv1D(eqx.Module):
    """Learned full polynomial/channel maps at every Alpert resolution."""

    transform: AlpertMultiwaveletTransform
    low_weight: Array
    detail_weights: tuple[Array, ...]
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    def __init__(
        self,
        transform: AlpertMultiwaveletTransform,
        /,
        *,
        in_channels: int,
        out_channels: int,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.transform = transform
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if min(self.in_channels, self.out_channels) <= 0:
            raise ValueError("Multiwavelet convolution channels must be positive.")
        keys = jr.split(key, transform.levels + 1)
        input_width = transform.order * self.in_channels
        output_width = transform.order * self.out_channels
        scale = 1.0 / sqrt(float(input_width))
        self.low_weight = scale * jr.normal(keys[0], (output_width, input_width))
        self.detail_weights = tuple(
            scale * jr.normal(keys[index + 1], (output_width, input_width))
            for index in range(transform.levels)
        )

    def _mix(self, weight: Array, values: Array, /) -> Array:
        flattened = values.reshape(
            values.shape[:-2] + (self.transform.order * self.in_channels,)
        )
        mixed = oe.contract("oi,...i->...o", weight, flattened)
        return mixed.reshape(mixed.shape[:-1] + (self.transform.order, self.out_channels))

    def __call__(self, values: Array, /) -> Array:
        array = jnp.asarray(values)
        if int(array.shape[-1]) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} multiwavelet input channels; "
                f"got {array.shape[-1]}."
            )
        coefficients = self.transform.analysis(array)
        low = self._mix(self.low_weight, coefficients.low)
        details = tuple(
            self._mix(weight, detail)
            for weight, detail in zip(
                self.detail_weights, coefficients.details, strict=True
            )
        )
        return self.transform.synthesis(
            MultiwaveletCoefficients(low=low, details=details)
        )


class MultiwaveletOperator(_AbstractOperatorModel):
    """Polynomial MWT neural operator on a fixed one-dimensional grid."""

    transform: AlpertMultiwaveletTransform
    lift: Linear
    multiwavelet_layers: tuple[MultiwaveletSpectralConv1D, ...]
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
        num_points: int,
        /,
        *,
        in_channels: int | Literal["scalar"],
        out_channels: int | Literal["scalar"],
        order: int = 3,
        levels: int = 3,
        boundary: WaveletBoundary = "periodic",
        width: int = 64,
        depth: int = 4,
        source_key: str | None = None,
        activation: Callable[[Array], Array] = jnn.gelu,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.transform = AlpertMultiwaveletTransform(
            num_points, order=order, levels=levels, boundary=boundary
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
        self.multiwavelet_layers = tuple(
            MultiwaveletSpectralConv1D(
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
        spatial_shape = (self.transform.num_points,)
        _validate_tensor_grid(source, batch.require_single_query(), spatial_shape)
        values = _grid_values(source, batch.case_shape, self.in_channels, spatial_shape)
        source_mask = source.mask_array(case_shape=batch.case_shape)
        hidden = self.lift(values * source_mask[..., None], key=fold_in_eval_key(key, 0))
        for index, (multiwavelet, pointwise) in enumerate(
            zip(self.multiwavelet_layers, self.pointwise_layers, strict=True)
        ):
            update = multiwavelet(hidden) + pointwise(
                hidden, key=fold_in_eval_key(key, 2 * index + 1)
            )
            hidden = self.activation(hidden + update)
        output = self.projection(hidden, key=fold_in_eval_key(key, 2 * self.depth + 1))
        query_mask = batch.require_single_query().mask_array(case_shape=batch.case_shape)
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


__all__ = [
    "AlpertMultiwaveletTransform",
    "MultiresolutionTransform",
    "MultiwaveletCoefficients",
    "MultiwaveletOperator",
    "MultiwaveletSpectralConv1D",
    "WaveletCoefficients",
    "WaveletLevelPlan",
    "WaveletNeuralOperator",
    "WaveletSpectralConvND",
]
