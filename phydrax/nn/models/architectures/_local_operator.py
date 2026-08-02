#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Literal

import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from ....graph._query_batch import query_neighbors
from ..._utils import _get_size
from ..core._base import _AbstractBaseModel, _AbstractOperatorModel
from ..core._keys import EvalKey, split_eval_key
from ..core._operator import FunctionSamples, OperatorBatch
from ..layers._linear import Linear


LocalGlobalFusion = Literal["sum", "concat"]


def _coordinates(
    samples: FunctionSamples,
    case_shape: tuple[int, ...],
    /,
) -> Array:
    if not samples.axes and samples.coordinates is None:
        raise ValueError("Local operators require source and query coordinates.")
    coordinates = samples.coordinates_array(case_shape=case_shape, flatten=True)
    cases = prod(case_shape) if case_shape else 1
    return coordinates.reshape(
        (cases, prod(samples.sample_shape), int(coordinates.shape[-1]))
    )


def _source_values(
    samples: FunctionSamples,
    /,
    *,
    case_ndim: int,
    channels: int,
) -> tuple[Array, tuple[int, ...]]:
    if samples.values is None:
        raise ValueError("Local operator source values cannot be None.")
    values = jnp.asarray(samples.values)
    sample_shape = samples.sample_shape
    if not sample_shape:
        raise ValueError("Local operator source samples must define a sample geometry.")
    sample_ndim = len(sample_shape)
    if tuple(values.shape[case_ndim : case_ndim + sample_ndim]) != sample_shape:
        raise ValueError(
            "Source values do not contain the source sample shape after case axes."
        )
    case_shape = tuple(int(size) for size in values.shape[:case_ndim])
    trailing = tuple(int(size) for size in values.shape[case_ndim + sample_ndim :])
    if not trailing:
        values = values[..., None]
    elif trailing != (int(channels),):
        raise ValueError(f"Expected {channels} source channels, got {trailing}.")
    return values.reshape(
        (prod(case_shape) if case_shape else 1, -1, channels)
    ), case_shape


def _apply_rows(model: _AbstractBaseModel, values: Array, key: EvalKey, /) -> Array:
    shape = values.shape[:-1]
    flattened = values.reshape((-1, int(values.shape[-1])))
    output = jax.vmap(lambda row: model(row, key=key))(flattened)
    return jnp.asarray(output).reshape(shape + (_get_size(model.out_size),))


def _query_mask(
    query: FunctionSamples,
    case_shape: tuple[int, ...],
    /,
) -> Array:
    cases = prod(case_shape) if case_shape else 1
    return query.mask_array(case_shape=case_shape).reshape(
        (cases, prod(query.sample_shape))
    )


def _neighbor_data(
    source_values: Array,
    source_coordinates: Array,
    source_weights: Array,
    query: Array,
    /,
    *,
    radius: float | None,
    max_neighbors: int | None,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    cases, query_count = int(query.shape[0]), int(query.shape[1])
    source_count = int(source_coordinates.shape[1])
    neighbor_count = (
        source_count if max_neighbors is None else min(int(max_neighbors), source_count)
    )
    neighborhood = query_neighbors(
        source_coordinates,
        query,
        source_mask=source_weights > 0.0,
        max_neighbors=neighbor_count,
        radius=radius,
    )
    indices = neighborhood.indices
    expanded_values = jnp.broadcast_to(
        source_values[:, None, :, :],
        (cases, query_count, source_count, source_values.shape[-1]),
    )
    expanded_coordinates = jnp.broadcast_to(
        source_coordinates[:, None, :, :],
        (cases, query_count, source_count, source_coordinates.shape[-1]),
    )
    expanded_weights = jnp.broadcast_to(
        source_weights[:, None, :],
        (cases, query_count, source_count),
    )
    source_data = jnp.take_along_axis(
        expanded_values,
        indices[..., None],
        axis=2,
    )
    source_position = jnp.take_along_axis(
        expanded_coordinates,
        indices[..., None],
        axis=2,
    )
    selected_weights = jnp.take_along_axis(expanded_weights, indices, axis=2)
    query_position = jnp.broadcast_to(
        query[:, :, None, :],
        source_position.shape,
    )
    pair_weights = selected_weights * neighborhood.mask.astype(selected_weights.dtype)
    return (
        source_data,
        source_position,
        query_position,
        -neighborhood.relative,
        pair_weights,
        neighborhood.distance,
    )


class LocalIntegralOperator(_AbstractOperatorModel):
    """Coordinate-kernel integral operator with physical-radius neighborhoods."""

    kernel_model: _AbstractBaseModel
    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    coord_dim: int
    radius: float | None
    normalize: bool
    source_key: str | None
    query_chunk_size: int
    max_neighbors: int | None

    def __init__(
        self,
        *,
        kernel_model: _AbstractBaseModel,
        coord_dim: int,
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        radius: float | None = None,
        normalize: bool = False,
        source_key: str | None = None,
        query_chunk_size: int = 256,
        max_neighbors: int | None = None,
    ):
        self.kernel_model = kernel_model
        self.coord_dim = int(coord_dim)
        self.in_size = in_channels
        self.out_size = out_channels
        self.radius = None if radius is None else float(radius)
        self.normalize = bool(normalize)
        self.source_key = source_key
        self.query_chunk_size = int(query_chunk_size)
        self.max_neighbors = None if max_neighbors is None else int(max_neighbors)
        expected_input = _get_size(in_channels) + 3 * self.coord_dim
        if _get_size(kernel_model.in_size) != expected_input:
            raise ValueError(
                f"Integral kernel model input size must be {expected_input}: source "
                "values, source coordinates, query coordinates, and displacement."
            )
        if _get_size(kernel_model.out_size) != _get_size(out_channels):
            raise ValueError("Integral kernel model output size must match out_channels.")
        if self.coord_dim <= 0 or self.query_chunk_size <= 0:
            raise ValueError("coord_dim and query_chunk_size must be positive.")
        if self.radius is not None and self.radius <= 0.0:
            raise ValueError("radius must be positive when supplied.")
        if self.max_neighbors is not None and self.max_neighbors <= 0:
            raise ValueError("max_neighbors must be positive when supplied.")

    def _source(self, batch: OperatorBatch, /) -> FunctionSamples:
        if self.source_key is not None:
            return batch.input(self.source_key)
        if len(batch.inputs) != 1:
            raise ValueError("source_key is required for multiple operator inputs.")
        return next(iter(batch.inputs.values()))

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        source = self._source(batch)
        source_coordinates = _coordinates(source, batch.case_shape)
        query_coordinates = _coordinates(batch.require_single_query(), batch.case_shape)
        if (
            int(source_coordinates.shape[-1]) != self.coord_dim
            or int(query_coordinates.shape[-1]) != self.coord_dim
        ):
            raise ValueError(
                "Source/query coordinate dimension does not match coord_dim."
            )
        source_values, case_shape = _source_values(
            source,
            case_ndim=len(batch.case_axes),
            channels=_get_size(self.in_size),
        )
        source_weights = source.weights(case_shape=case_shape).reshape(
            source_values.shape[:2]
        )
        mask = _query_mask(batch.require_single_query(), case_shape)
        chunks = []
        for start in range(0, int(query_coordinates.shape[1]), self.query_chunk_size):
            query = query_coordinates[:, start : start + self.query_chunk_size, :]
            (
                source_data,
                source_position,
                query_position,
                displacement,
                pair_weights,
                _,
            ) = _neighbor_data(
                source_values,
                source_coordinates,
                source_weights,
                query,
                radius=self.radius,
                max_neighbors=self.max_neighbors,
            )
            if self.normalize:
                total = jnp.sum(pair_weights, axis=-1, keepdims=True)
                pair_weights = jnp.where(
                    total > 0.0,
                    pair_weights / total,
                    jnp.zeros_like(pair_weights),
                )
            features = jnp.concatenate(
                (source_data, source_position, query_position, displacement),
                axis=-1,
            )
            messages = _apply_rows(self.kernel_model, features, key)
            chunk = jnp.sum(messages * pair_weights[..., None], axis=2)
            chunks.append(chunk)

        output = jnp.concatenate(chunks, axis=1)
        output = output * mask[..., None]
        output = output.reshape(
            case_shape
            + batch.require_single_query().sample_shape
            + (_get_size(self.out_size),)
        )
        if self.out_size == "scalar":
            return output[..., 0]
        return output

    def __call__(
        self,
        x: Array | tuple[Array, ...] | OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("LocalIntegralOperator requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


class LocalDifferentialOperator(_AbstractOperatorModel):
    """Constant-preserving localized nonlocal differential kernel."""

    kernel_model: _AbstractBaseModel
    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    coord_dim: int
    radius: float
    differential_order: float
    source_key: str | None
    query_chunk_size: int
    max_neighbors: int | None

    def __init__(
        self,
        *,
        kernel_model: _AbstractBaseModel,
        coord_dim: int,
        radius: float,
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        differential_order: float = 1.0,
        source_key: str | None = None,
        query_chunk_size: int = 256,
        max_neighbors: int | None = None,
    ):
        self.kernel_model = kernel_model
        self.coord_dim = int(coord_dim)
        self.radius = float(radius)
        self.in_size = in_channels
        self.out_size = out_channels
        self.differential_order = float(differential_order)
        self.source_key = source_key
        self.query_chunk_size = int(query_chunk_size)
        self.max_neighbors = None if max_neighbors is None else int(max_neighbors)
        if self.coord_dim <= 0 or self.radius <= 0.0 or self.query_chunk_size <= 0:
            raise ValueError("coord_dim, radius, and query_chunk_size must be positive.")
        if self.max_neighbors is not None and self.max_neighbors <= 0:
            raise ValueError("max_neighbors must be positive when supplied.")
        if self.differential_order < 0.0:
            raise ValueError("differential_order must be non-negative.")
        if _get_size(kernel_model.in_size) != self.coord_dim + 1:
            raise ValueError(
                "Differential kernel input must contain normalized displacement and distance."
            )
        expected_output = _get_size(in_channels) * _get_size(out_channels)
        if _get_size(kernel_model.out_size) != expected_output:
            raise ValueError(
                f"Differential kernel output size must be {expected_output}."
            )

    def _source(self, batch: OperatorBatch, /) -> FunctionSamples:
        if self.source_key is not None:
            return batch.input(self.source_key)
        if len(batch.inputs) != 1:
            raise ValueError("source_key is required for multiple operator inputs.")
        return next(iter(batch.inputs.values()))

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        source = self._source(batch)
        source_coordinates = _coordinates(source, batch.case_shape)
        query_coordinates = _coordinates(batch.require_single_query(), batch.case_shape)
        if (
            int(source_coordinates.shape[-1]) != self.coord_dim
            or int(query_coordinates.shape[-1]) != self.coord_dim
        ):
            raise ValueError(
                "Source/query coordinate dimension does not match coord_dim."
            )
        source_values, case_shape = _source_values(
            source,
            case_ndim=len(batch.case_axes),
            channels=_get_size(self.in_size),
        )
        source_weights = source.weights(case_shape=case_shape).reshape(
            source_values.shape[:2]
        )
        query_mask = _query_mask(batch.require_single_query(), case_shape)
        chunks = []
        for start in range(0, int(query_coordinates.shape[1]), self.query_chunk_size):
            query = query_coordinates[:, start : start + self.query_chunk_size, :]
            (
                source_data,
                _source_position,
                _query_position,
                displacement,
                pair_weights,
                distance,
            ) = _neighbor_data(
                source_values,
                source_coordinates,
                source_weights,
                query,
                radius=self.radius,
                max_neighbors=self.max_neighbors,
            )
            denominator = jnp.sum(pair_weights, axis=-1, keepdims=True)
            normalized_weights = jnp.where(
                denominator > 0.0,
                pair_weights / denominator,
                jnp.zeros_like(pair_weights),
            )
            center = oe.contract("cqs,cqsi->cqi", normalized_weights, source_data)
            differences = source_data - center[:, :, None, :]
            kernel_inputs = jnp.concatenate(
                (
                    displacement / self.radius,
                    (distance / self.radius)[..., None],
                ),
                axis=-1,
            )
            kernels = _apply_rows(self.kernel_model, kernel_inputs, key)
            kernels = kernels.reshape(
                (
                    int(query.shape[0]),
                    int(query.shape[1]),
                    int(source_data.shape[2]),
                    _get_size(self.out_size),
                    _get_size(self.in_size),
                )
            )
            chunk = oe.contract(
                "cqsoi,cqsi,cqs->cqo",
                kernels,
                differences,
                normalized_weights,
            )
            chunk = chunk / self.radius**self.differential_order
            chunks.append(chunk)

        output = jnp.concatenate(chunks, axis=1)
        output = output * query_mask[..., None]
        output = output.reshape(
            case_shape
            + batch.require_single_query().sample_shape
            + (_get_size(self.out_size),)
        )
        if self.out_size == "scalar":
            return output[..., 0]
        return output

    def __call__(
        self,
        x: Array | tuple[Array, ...] | OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("LocalDifferentialOperator requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


class LocalGlobalOperator(_AbstractOperatorModel):
    """Compose global and localized operator paths over one OperatorBatch."""

    global_operator: _AbstractOperatorModel
    local_operator: _AbstractOperatorModel
    mixer: Linear | None
    fusion: LocalGlobalFusion
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]

    def __init__(
        self,
        *,
        global_operator: _AbstractOperatorModel,
        local_operator: _AbstractOperatorModel,
        fusion: LocalGlobalFusion = "sum",
        mixer: Linear | None = None,
    ):
        self.global_operator = global_operator
        self.local_operator = local_operator
        self.fusion = fusion
        self.mixer = mixer
        self.in_size = global_operator.in_size
        self.out_size = global_operator.out_size
        if global_operator.out_size != local_operator.out_size:
            raise ValueError("Local and global operator output sizes must match.")
        if fusion not in ("sum", "concat"):
            raise ValueError("fusion must be 'sum' or 'concat'.")
        channels = _get_size(self.out_size)
        if fusion == "concat":
            if mixer is None:
                raise ValueError("concat fusion requires a mixer Linear layer.")
            if (
                _get_size(mixer.in_size) != 2 * channels
                or _get_size(mixer.out_size) != channels
            ):
                raise ValueError("concat mixer must map 2*out_size to out_size.")
        elif mixer is not None:
            raise ValueError("mixer is only used by concat fusion.")

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        global_key, local_key = split_eval_key(key, 2)
        global_output = jnp.asarray(
            self.global_operator.__call_operator_batch__(batch, key=global_key)
        )
        local_output = jnp.asarray(
            self.local_operator.__call_operator_batch__(batch, key=local_key)
        )
        if self.fusion == "sum":
            return (global_output + local_output) / jnp.sqrt(2.0)
        assert self.mixer is not None
        if self.out_size == "scalar":
            global_output = global_output[..., None]
            local_output = local_output[..., None]
        output = self.mixer(jnp.concatenate((global_output, local_output), axis=-1))
        if self.out_size == "scalar":
            return output[..., 0]
        return output

    def __call__(
        self,
        x: Array | tuple[Array, ...] | OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("LocalGlobalOperator requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


__all__ = [
    "LocalDifferentialOperator",
    "LocalGlobalOperator",
    "LocalIntegralOperator",
]
