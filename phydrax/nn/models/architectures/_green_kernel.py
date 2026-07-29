#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ..._utils import _get_size
from ..core._base import _AbstractBaseModel, _AbstractOperatorModel
from ..core._keys import EvalKey, split_eval_key
from ..core._operator import FunctionSamples, OperatorBatch
from ._mlp import MLP


def _coordinates(
    samples: FunctionSamples,
    case_shape: tuple[int, ...],
    name: str,
    /,
) -> Array:
    if not samples.axes and samples.coordinates is None:
        raise ValueError(f"{name} must define coordinate geometry.")
    coordinates = samples.coordinates_array(case_shape=case_shape, flatten=True)
    return coordinates.reshape(
        (
            prod(case_shape) if case_shape else 1,
            prod(samples.sample_shape),
            int(coordinates.shape[-1]),
        )
    )


def _values(
    samples: FunctionSamples,
    case_shape: tuple[int, ...],
    channels: int,
    name: str,
    /,
) -> Array:
    if samples.values is None:
        raise ValueError(f"{name} must contain sampled values.")
    values = samples.values
    sample_shape = samples.sample_shape
    if not sample_shape:
        raise ValueError(f"{name} must define a non-empty sample geometry.")
    prefix = case_shape + sample_shape
    if tuple(int(size) for size in values.shape[: len(prefix)]) != prefix:
        raise ValueError(
            f"{name} values must start with case/sample shape {prefix}; "
            f"got {values.shape}."
        )
    trailing = tuple(int(size) for size in values.shape[len(prefix) :])
    if not trailing and channels == 1:
        values = values[..., None]
    elif trailing != (channels,):
        raise ValueError(
            f"{name} values must be scalar or channel-last with {channels} "
            f"channels; got trailing shape {trailing}."
        )
    return values.reshape(
        (
            prod(case_shape) if case_shape else 1,
            prod(sample_shape),
            channels,
        )
    )


def _physical_weights(
    samples: FunctionSamples,
    case_shape: tuple[int, ...],
    name: str,
    /,
) -> Array:
    has_measure = samples.quadrature_weights is not None or (
        bool(samples.axes)
        and all(axis.quadrature_weights is not None for axis in samples.axes)
    )
    if not has_measure:
        raise ValueError(
            f"{name} requires physical quadrature weights; unit-counting measures "
            "are not used by GreenKernelOperator."
        )
    return samples.weights(case_shape=case_shape, normalized=False).reshape(
        (
            prod(case_shape) if case_shape else 1,
            prod(samples.sample_shape),
        )
    )


def _apply_rows(model: _AbstractBaseModel, values: Array, key: EvalKey, /) -> Array:
    shape = values.shape[:-1]
    flattened = values.reshape((-1, int(values.shape[-1])))
    output = jax.vmap(lambda row: model(row, key=key))(flattened)
    return jnp.asarray(output).reshape(shape + (_get_size(model.out_size),))


class GreenKernelOperator(_AbstractOperatorModel):
    r"""Learned volume/boundary Green-kernel prototype.

    The model approximates a solution operator by two distinct, measure-aware
    integrals: one over interior forcing samples and one over boundary samples.
    The kernels are learned from source/query geometry and are not tied to a
    particular PDE. Consequently this prototype does not impose, or claim,
    exact PDE or boundary-condition satisfaction.

    Boundary-condition descriptors or geometric features such as normals can
    be appended as channels of the boundary field and included in
    ``boundary_channels``. Source coordinates, query coordinates, displacement,
    and distance are always supplied to both learned kernels.
    """

    forcing_kernel: MLP
    boundary_kernel: MLP
    head: MLP
    in_size: tuple[int, int]
    out_size: int | Literal["scalar"]
    coord_dim: int
    forcing_channels: int
    boundary_channels: int
    latent_channels: int
    forcing_key: str
    boundary_key: str
    query_chunk_size: int

    def __init__(
        self,
        *,
        coord_dim: int,
        forcing_channels: int | Literal["scalar"] = "scalar",
        boundary_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        width: int = 64,
        depth: int = 2,
        kernel_width: int | None = None,
        kernel_depth: int = 2,
        forcing_key: str = "forcing",
        boundary_key: str = "boundary",
        query_chunk_size: int = 256,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        coordinate_dimension = int(coord_dim)
        forcing_count = _get_size(forcing_channels)
        boundary_count = _get_size(boundary_channels)
        output_count = _get_size(out_channels)
        latent_count = int(width)
        width_ = int(width)
        depth_ = int(depth)
        kernel_width_ = width_ if kernel_width is None else int(kernel_width)
        kernel_depth_ = int(kernel_depth)
        chunk_size = int(query_chunk_size)
        if coordinate_dimension <= 0:
            raise ValueError("coord_dim must be positive.")
        if width_ <= 0:
            raise ValueError("width must be positive.")
        if kernel_width_ <= 0:
            raise ValueError("kernel_width must be positive when supplied.")
        if depth_ < 0 or kernel_depth_ < 0:
            raise ValueError("depth and kernel_depth must be non-negative.")
        if chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive.")
        if not forcing_key or not boundary_key:
            raise ValueError("forcing_key and boundary_key must be non-empty.")
        if forcing_key == boundary_key:
            raise ValueError("forcing_key and boundary_key must be distinct.")

        forcing_init, boundary_init, head_init = jr.split(key, 3)
        pair_features = 3 * coordinate_dimension + 1
        self.forcing_kernel = MLP(
            in_size=pair_features,
            out_size=latent_count * forcing_count,
            width_size=kernel_width_,
            depth=kernel_depth_,
            activation=jax.nn.gelu,
            key=forcing_init,
        )
        self.boundary_kernel = MLP(
            in_size=pair_features,
            out_size=latent_count * boundary_count,
            width_size=kernel_width_,
            depth=kernel_depth_,
            activation=jax.nn.gelu,
            key=boundary_init,
        )
        self.head = MLP(
            in_size=2 * latent_count + coordinate_dimension,
            out_size=output_count,
            width_size=width_,
            depth=depth_,
            activation=jax.nn.gelu,
            key=head_init,
        )
        self.in_size = (forcing_count, boundary_count)
        self.out_size = out_channels
        self.coord_dim = coordinate_dimension
        self.forcing_channels = forcing_count
        self.boundary_channels = boundary_count
        self.latent_channels = latent_count
        self.forcing_key = str(forcing_key)
        self.boundary_key = str(boundary_key)
        self.query_chunk_size = chunk_size

    def _branch_integral(
        self,
        kernel: _AbstractBaseModel,
        source_values: Array,
        source_coordinates: Array,
        source_weights: Array,
        query_coordinates: Array,
        channels: int,
        key: EvalKey,
        /,
    ) -> Array:
        source_position = source_coordinates[:, None, :, :]
        query_position = query_coordinates[:, :, None, :]
        displacement = query_position - source_position
        distance = jnp.linalg.norm(displacement, axis=-1, keepdims=True)
        source_position = jnp.broadcast_to(source_position, displacement.shape)
        query_position = jnp.broadcast_to(query_position, displacement.shape)
        features = jnp.concatenate(
            (source_position, query_position, displacement, distance), axis=-1
        )
        learned_kernel = _apply_rows(kernel, features, key).reshape(
            features.shape[:-1] + (self.latent_channels, channels)
        )
        messages = jnp.einsum("bqslc,bsc->bqsl", learned_kernel, source_values)
        return jnp.sum(messages * source_weights[:, None, :, None], axis=2)

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        forcing = batch.input(self.forcing_key)
        boundary = batch.input(self.boundary_key)
        forcing_coordinates = _coordinates(
            forcing, batch.case_shape, "Interior forcing"
        )
        boundary_coordinates = _coordinates(
            boundary, batch.case_shape, "Boundary data"
        )
        query_coordinates = _coordinates(batch.require_single_query(), batch.case_shape, "Query")
        if (
            int(forcing_coordinates.shape[-1]) != self.coord_dim
            or int(boundary_coordinates.shape[-1]) != self.coord_dim
            or int(query_coordinates.shape[-1]) != self.coord_dim
        ):
            raise ValueError(
                "Forcing, boundary, and query coordinate dimensions must match "
                "coord_dim."
            )

        forcing_values = _values(
            forcing,
            batch.case_shape,
            self.forcing_channels,
            "Interior forcing",
        )
        boundary_values = _values(
            boundary,
            batch.case_shape,
            self.boundary_channels,
            "Boundary data",
        )
        forcing_weights = _physical_weights(
            forcing, batch.case_shape, "Interior forcing"
        )
        boundary_weights = _physical_weights(
            boundary, batch.case_shape, "Boundary data"
        )
        query_mask = batch.require_single_query().mask_array(case_shape=batch.case_shape).reshape(
            (
                prod(batch.case_shape) if batch.case_shape else 1,
                prod(batch.require_single_query().sample_shape),
            )
        )
        forcing_eval_key, boundary_eval_key, head_eval_key = split_eval_key(key, 3)

        chunks: list[Array] = []
        for start in range(0, int(query_coordinates.shape[1]), self.query_chunk_size):
            query_chunk = query_coordinates[:, start : start + self.query_chunk_size]
            forcing_state = self._branch_integral(
                self.forcing_kernel,
                forcing_values,
                forcing_coordinates,
                forcing_weights,
                query_chunk,
                self.forcing_channels,
                forcing_eval_key,
            )
            boundary_state = self._branch_integral(
                self.boundary_kernel,
                boundary_values,
                boundary_coordinates,
                boundary_weights,
                query_chunk,
                self.boundary_channels,
                boundary_eval_key,
            )
            head_features = jnp.concatenate(
                (forcing_state, boundary_state, query_chunk), axis=-1
            )
            chunks.append(_apply_rows(self.head, head_features, head_eval_key))

        output = jnp.concatenate(chunks, axis=1)
        output = output * query_mask[..., None]
        output = output.reshape(
            batch.case_shape
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
            raise TypeError("GreenKernelOperator requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


__all__ = ["GreenKernelOperator"]
