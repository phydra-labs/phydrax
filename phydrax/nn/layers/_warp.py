#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import cast, Literal, TypeAlias

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from .._keys import EvalKey
from ._linear import Linear
from ._warp_geometry import (
    normalized_lattice_from_nodes,
    RectilinearWarpDiagnostics,
    sample_rectilinear_grid,
    warp_jacobian,
    WarpMaskMode,
)


WarpBoundaryMode: TypeAlias = Literal["periodic", "reflect", "clamp", "constant"]
_VALID_BOUNDARY_MODES = frozenset(("periodic", "reflect", "clamp", "constant"))


def _boundary_modes(
    boundary: WarpBoundaryMode | Sequence[WarpBoundaryMode],
    spatial_ndim: int,
    /,
) -> tuple[WarpBoundaryMode, ...]:
    modes = (boundary,) * spatial_ndim if isinstance(boundary, str) else tuple(boundary)
    if len(modes) != spatial_ndim:
        raise ValueError(
            f"boundary must provide one mode per spatial axis; expected "
            f"{spatial_ndim}, got {len(modes)}."
        )
    invalid = tuple(mode for mode in modes if mode not in _VALID_BOUNDARY_MODES)
    if invalid:
        raise ValueError(
            "boundary modes must be 'periodic', 'reflect', 'clamp', or "
            f"'constant'; got {invalid}."
        )
    return cast(tuple[WarpBoundaryMode, ...], modes)


def _normalized_lattice(
    spatial_shape: tuple[int, ...],
    boundary: tuple[WarpBoundaryMode, ...],
    /,
    *,
    dtype: jnp.dtype,
) -> Array:
    coordinates = []
    for size, mode in zip(spatial_shape, boundary, strict=True):
        if mode == "periodic":
            coordinate = -1.0 + 2.0 * jnp.arange(size, dtype=dtype) / float(size)
        else:
            coordinate = jnp.linspace(-1.0, 1.0, size, dtype=dtype)
        coordinates.append(coordinate)
    return jnp.stack(jnp.meshgrid(*coordinates, indexing="ij"), axis=-1)


def _sample_regular_grid_linear(
    values: Array,
    coordinates: Array,
    /,
    *,
    spatial_ndim: int,
    boundary: tuple[WarpBoundaryMode, ...],
    fill_value: float,
) -> Array:
    """Sample a channel-last regular grid at normalized query coordinates."""

    array = jnp.asarray(values)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(
            "Regular-grid warping currently supports real-valued arrays only."
        )
    if array.ndim < spatial_ndim + 1:
        raise ValueError(
            "values must end in spatial dimensions followed by one channel axis."
        )
    batch_shape = tuple(int(size) for size in array.shape[: -spatial_ndim - 1])
    spatial_shape = tuple(int(size) for size in array.shape[-spatial_ndim - 1 : -1])
    if any(size < 2 for size in spatial_shape):
        raise ValueError("Every warped spatial axis must contain at least two nodes.")

    coordinate_dtype = jnp.result_type(array.dtype, float)
    query = jnp.asarray(coordinates, dtype=coordinate_dtype)
    if query.ndim < len(batch_shape) + 2 or int(query.shape[-1]) != spatial_ndim:
        raise ValueError(
            "coordinates must have shape batch_shape + query_shape + "
            f"({spatial_ndim},); got {query.shape}."
        )
    if tuple(int(size) for size in query.shape[: len(batch_shape)]) != batch_shape:
        raise ValueError(
            f"Coordinate batch shape must be {batch_shape}; got "
            f"{query.shape[: len(batch_shape)]}."
        )
    query_shape = tuple(int(size) for size in query.shape[len(batch_shape) : -1])
    if not query_shape or any(size <= 0 for size in query_shape):
        raise ValueError("Regular-grid queries must contain at least one sample.")

    lower_indices: list[Array] = []
    upper_indices: list[Array] = []
    fractions: list[Array] = []
    outside = jnp.zeros(batch_shape + query_shape, dtype=bool)

    for axis, (size, mode) in enumerate(zip(spatial_shape, boundary, strict=True)):
        normalized = query[..., axis]
        if mode == "periodic":
            continuous = jnp.mod(0.5 * (normalized + 1.0) * float(size), float(size))
            lower_float = jnp.floor(continuous)
            lower = lower_float.astype(jnp.int32)
            upper = jnp.mod(lower + 1, size)
        else:
            continuous = 0.5 * (normalized + 1.0) * float(size - 1)
            if mode == "reflect":
                period = float(2 * (size - 1))
                reflected = jnp.mod(continuous, period)
                continuous = jnp.where(
                    reflected <= float(size - 1), reflected, period - reflected
                )
            elif mode == "clamp":
                continuous = jnp.clip(continuous, 0.0, float(size - 1))
            else:
                outside = outside | (continuous < 0.0) | (continuous > float(size - 1))
                continuous = jnp.clip(continuous, 0.0, float(size - 1))
            lower_float = jnp.floor(continuous)
            lower = lower_float.astype(jnp.int32)
            upper = jnp.minimum(lower + 1, size - 1)
        lower_indices.append(lower)
        upper_indices.append(upper)
        fractions.append(continuous - lower_float)

    batch_count = prod(batch_shape) if batch_shape else 1
    query_count = prod(query_shape)
    channels = int(array.shape[-1])
    flat_values = array.reshape((batch_count, prod(spatial_shape), channels))
    output = jnp.zeros(batch_shape + query_shape + (channels,), dtype=array.dtype)

    for corner in range(1 << spatial_ndim):
        indices = []
        weight = jnp.ones(batch_shape + query_shape, dtype=coordinate_dtype)
        for axis in range(spatial_ndim):
            upper = bool(corner & (1 << axis))
            indices.append(upper_indices[axis] if upper else lower_indices[axis])
            fraction = fractions[axis]
            weight = weight * (fraction if upper else 1.0 - fraction)
        linear_index = indices[0]
        for axis in range(1, spatial_ndim):
            linear_index = linear_index * spatial_shape[axis] + indices[axis]
        flat_index = linear_index.reshape((batch_count, query_count))
        gathered = jnp.take_along_axis(
            flat_values,
            flat_index[..., None],
            axis=1,
        ).reshape(batch_shape + query_shape + (channels,))
        output = output + gathered * weight[..., None].astype(array.dtype)

    if "constant" in boundary:
        output = jnp.where(
            outside[..., None],
            jnp.asarray(fill_value, dtype=array.dtype),
            output,
        )
    return output


class MultiheadWarp(StrictModule):
    """Adaptive multihead pullback on a regular channel-last grid.

    Displacements are predicted in domain-normalized coordinates. Periodic axes
    use the half-open interval ``[-1, 1)``; nonperiodic axes use ``[-1, 1]``.
    """

    value_projection: Linear
    displacement_hidden: Linear
    displacement_condition: Linear | None
    displacement_output: Linear
    spatial_ndim: int
    in_channels: int
    out_channels: int
    num_heads: int
    conditioning_size: int
    boundary: tuple[WarpBoundaryMode, ...]
    fill_value: float
    mask_mode: WarpMaskMode

    def __init__(
        self,
        *,
        spatial_ndim: int,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        boundary: WarpBoundaryMode | Sequence[WarpBoundaryMode],
        conditioning_size: int = 0,
        mask_mode: WarpMaskMode = "reject",
        displacement_width: int | None = None,
        fill_value: float = 0.0,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.spatial_ndim = int(spatial_ndim)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_heads = int(num_heads)
        self.conditioning_size = int(conditioning_size)
        self.fill_value = float(fill_value)
        self.mask_mode = mask_mode
        if self.spatial_ndim not in (1, 2, 3):
            raise ValueError("MultiheadWarp supports one, two, or three dimensions.")
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if self.conditioning_size < 0:
            raise ValueError("conditioning_size must be non-negative.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.out_channels % self.num_heads != 0:
            raise ValueError("out_channels must be divisible by num_heads.")
        hidden_width = (
            self.out_channels if displacement_width is None else int(displacement_width)
        )
        if hidden_width <= 0:
            raise ValueError("displacement_width must be positive.")
        if self.mask_mode not in ("reject", "renormalize", "strict"):
            raise ValueError("mask_mode must be 'reject', 'renormalize', or 'strict'.")
        self.boundary = _boundary_modes(boundary, self.spatial_ndim)

        value_key, hidden_key, output_key = jr.split(key, 3)
        self.value_projection = Linear(
            in_size=self.in_channels,
            out_size=self.out_channels,
            activation=None,
            rwf=False,
            bias_init_lim=0.0,
            key=value_key,
        )
        self.displacement_hidden = Linear(
            in_size=self.in_channels,
            out_size=hidden_width,
            activation=None,
            rwf=False,
            bias_init_lim=0.0,
            key=hidden_key,
        )
        self.displacement_condition = (
            None
            if self.conditioning_size == 0
            else Linear(
                in_size=self.conditioning_size,
                out_size=hidden_width,
                activation=None,
                rwf=False,
                use_bias=False,
                key=jr.fold_in(key, 3),
            )
        )
        self.displacement_output = Linear(
            in_size=hidden_width,
            out_size=self.num_heads * self.spatial_ndim,
            activation=None,
            rwf=False,
            bias_init_lim=0.0,
            key=output_key,
        )

    def _layout(
        self,
        values: Array,
        /,
    ) -> tuple[Array, tuple[int, ...], tuple[int, ...], int]:
        array = jnp.asarray(values)
        if jnp.issubdtype(array.dtype, jnp.complexfloating):
            raise TypeError("MultiheadWarp currently supports real-valued arrays only.")
        if array.ndim < self.spatial_ndim + 1:
            raise ValueError(
                "MultiheadWarp input must end in spatial dimensions and channels."
            )
        if int(array.shape[-1]) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {array.shape[-1]}."
            )
        spatial_shape = tuple(
            int(size) for size in array.shape[-self.spatial_ndim - 1 : -1]
        )
        if any(size < 2 for size in spatial_shape):
            raise ValueError("Every warped spatial axis must contain at least two nodes.")
        case_shape = tuple(int(size) for size in array.shape[: -self.spatial_ndim - 1])
        return (
            array,
            case_shape,
            spatial_shape,
            self.out_channels // self.num_heads,
        )

    def displacement_features(
        self,
        values: Array,
        /,
        *,
        condition: Array | None = None,
    ) -> Array:
        """Return the hidden features used to predict per-head displacements."""

        array, case_shape, _, _ = self._layout(values)
        hidden = self.displacement_hidden(array)
        if self.displacement_condition is None:
            if condition is not None:
                raise ValueError("condition must be None when conditioning_size is zero.")
            return hidden
        if condition is None:
            raise ValueError(
                "MultiheadWarp requires condition when conditioning_size is positive."
            )
        condition_array = jnp.asarray(condition)
        if jnp.issubdtype(condition_array.dtype, jnp.complexfloating):
            raise TypeError(
                "MultiheadWarp currently supports real-valued conditions only."
            )
        expected_condition_shape = case_shape + (self.conditioning_size,)
        if condition_array.shape != expected_condition_shape:
            raise ValueError(
                f"MultiheadWarp condition must have shape "
                f"{expected_condition_shape}; got {condition_array.shape}."
            )
        condition_hidden = self.displacement_condition(
            condition_array.astype(array.dtype)
        )
        condition_hidden = condition_hidden.reshape(
            case_shape + (1,) * self.spatial_ndim + (int(condition_hidden.shape[-1]),)
        )
        return hidden + condition_hidden

    def displacement(
        self,
        values: Array,
        /,
        *,
        condition: Array | None = None,
    ) -> Array:
        """Predict domain-normalized displacements for every spatial head."""

        _, case_shape, spatial_shape, _ = self._layout(values)
        hidden = self.displacement_features(values, condition=condition)
        return self.displacement_output(jax.nn.gelu(hidden)).reshape(
            case_shape + spatial_shape + (self.num_heads, self.spatial_ndim)
        )

    def _route_geometry(
        self,
        displacement: Array,
        spatial_shape: tuple[int, ...],
        case_shape: tuple[int, ...],
        axis_nodes: Sequence[Array] | None,
        /,
    ) -> tuple[Array, Array]:
        if axis_nodes is None:
            lattice = _normalized_lattice(
                spatial_shape,
                self.boundary,
                dtype=jnp.result_type(displacement.dtype, float),
            )
        else:
            lattice = normalized_lattice_from_nodes(axis_nodes)
            if lattice.shape != spatial_shape + (self.spatial_ndim,):
                raise ValueError(
                    "axis_nodes do not match the warped spatial shape "
                    f"{spatial_shape}; got lattice shape {lattice.shape}."
                )
        coordinates = displacement + lattice[..., None, :]
        routed_displacement = jnp.moveaxis(
            displacement,
            len(case_shape) + self.spatial_ndim,
            len(case_shape),
        )
        routed_coordinates = jnp.moveaxis(
            coordinates,
            len(case_shape) + self.spatial_ndim,
            len(case_shape),
        )
        return routed_displacement, routed_coordinates

    def _source_mask(
        self,
        source_mask: Array | None,
        case_shape: tuple[int, ...],
        spatial_shape: tuple[int, ...],
        /,
    ) -> Array | None:
        if source_mask is None:
            return None
        mask = jnp.asarray(source_mask, dtype=bool)
        if mask.shape == spatial_shape:
            mask = jnp.broadcast_to(mask, case_shape + spatial_shape)
        elif mask.shape != case_shape + spatial_shape:
            raise ValueError(
                f"MultiheadWarp source_mask must have shape {spatial_shape} or "
                f"{case_shape + spatial_shape}; got {mask.shape}."
            )
        return jnp.broadcast_to(
            jnp.expand_dims(mask, axis=len(case_shape)),
            case_shape + (self.num_heads,) + spatial_shape,
        )

    def transport(
        self,
        values: Array,
        displacement: Array,
        /,
        *,
        axis_nodes: Sequence[Array] | None = None,
        source_mask: Array | None = None,
    ) -> Array:
        """Project and sample values along an explicitly supplied displacement."""

        array, case_shape, spatial_shape, head_channels = self._layout(values)
        expected = case_shape + spatial_shape + (self.num_heads, self.spatial_ndim)
        if displacement.shape != expected:
            raise ValueError(
                f"MultiheadWarp displacement must have shape {expected}; "
                f"got {displacement.shape}."
            )
        projected = self.value_projection(array).reshape(
            case_shape + spatial_shape + (self.num_heads, head_channels)
        )
        projected = jnp.moveaxis(
            projected,
            len(case_shape) + self.spatial_ndim,
            len(case_shape),
        )
        _, coordinates = self._route_geometry(
            displacement,
            spatial_shape,
            case_shape,
            axis_nodes,
        )
        routed_mask = self._source_mask(source_mask, case_shape, spatial_shape)
        batch_count = (prod(case_shape) if case_shape else 1) * self.num_heads
        routed_values = projected.reshape(
            (batch_count,) + spatial_shape + (head_channels,)
        )
        routed_coordinates = coordinates.reshape(
            (batch_count,) + spatial_shape + (self.spatial_ndim,)
        )
        if axis_nodes is None and routed_mask is None:
            sampled = _sample_regular_grid_linear(
                routed_values,
                routed_coordinates,
                spatial_ndim=self.spatial_ndim,
                boundary=self.boundary,
                fill_value=self.fill_value,
            )
        else:
            flattened_mask = (
                None
                if routed_mask is None
                else routed_mask.reshape((batch_count,) + spatial_shape)
            )
            sampled = sample_rectilinear_grid(
                routed_values,
                routed_coordinates,
                spatial_ndim=self.spatial_ndim,
                boundary=self.boundary,
                axis_nodes=axis_nodes,
                source_mask=flattened_mask,
                mask_mode=self.mask_mode,
                fill_value=self.fill_value,
            )
        if isinstance(sampled, tuple):
            raise RuntimeError("Warp transport unexpectedly returned support.")
        sampled = sampled.reshape(
            case_shape + (self.num_heads,) + spatial_shape + (head_channels,)
        )
        sampled = jnp.moveaxis(
            sampled,
            len(case_shape),
            len(case_shape) + self.spatial_ndim,
        )
        return sampled.reshape(case_shape + spatial_shape + (self.out_channels,))

    def diagnostics_from_displacement(
        self,
        values: Array,
        displacement: Array,
        /,
        *,
        axis_nodes: Sequence[Array] | None = None,
        source_mask: Array | None = None,
        route_scale: Array | None = None,
    ) -> RectilinearWarpDiagnostics:
        """Evaluate geometry for an explicitly supplied displacement route."""

        array, case_shape, spatial_shape, head_channels = self._layout(values)
        routed_displacement, coordinates = self._route_geometry(
            jnp.asarray(displacement),
            spatial_shape,
            case_shape,
            axis_nodes,
        )
        projected = self.value_projection(array).reshape(
            case_shape + spatial_shape + (self.num_heads, head_channels)
        )
        projected = jnp.moveaxis(
            projected,
            len(case_shape) + self.spatial_ndim,
            len(case_shape),
        )
        routed_mask = self._source_mask(source_mask, case_shape, spatial_shape)
        batch_count = (prod(case_shape) if case_shape else 1) * self.num_heads
        sampling_result = sample_rectilinear_grid(
            projected.reshape((batch_count,) + spatial_shape + (head_channels,)),
            coordinates.reshape((batch_count,) + spatial_shape + (self.spatial_ndim,)),
            spatial_ndim=self.spatial_ndim,
            boundary=self.boundary,
            axis_nodes=axis_nodes,
            source_mask=(
                None
                if routed_mask is None
                else routed_mask.reshape((batch_count,) + spatial_shape)
            ),
            mask_mode=self.mask_mode,
            fill_value=self.fill_value,
            return_support=True,
        )
        if not isinstance(sampling_result, tuple):
            raise RuntimeError("Warp interpolation support was not returned.")
        _, support = sampling_result
        jacobian = warp_jacobian(
            routed_displacement,
            boundary=self.boundary,
            axis_nodes=axis_nodes,
        )
        head_axis = len(case_shape)
        target_axis = len(case_shape) + self.spatial_ndim
        return RectilinearWarpDiagnostics(
            displacement=displacement,
            coordinates=jnp.moveaxis(coordinates, head_axis, target_axis),
            jacobian=jnp.moveaxis(jacobian, head_axis, target_axis),
            determinant=jnp.moveaxis(
                jnp.linalg.det(jacobian),
                head_axis,
                target_axis,
            ),
            interpolation_support=jnp.moveaxis(
                support.reshape(case_shape + (self.num_heads,) + spatial_shape),
                head_axis,
                target_axis,
            ),
            route_scale=route_scale,
        )

    def diagnostics(
        self,
        values: Array,
        /,
        *,
        condition: Array | None = None,
        axis_nodes: Sequence[Array] | None = None,
        source_mask: Array | None = None,
        key: EvalKey = None,
    ) -> RectilinearWarpDiagnostics:
        """Evaluate deterministic route geometry without changing inference."""

        del key
        displacement = self.displacement(values, condition=condition)
        return self.diagnostics_from_displacement(
            values,
            displacement,
            axis_nodes=axis_nodes,
            source_mask=source_mask,
        )

    def __call__(
        self,
        values: Array,
        /,
        *,
        condition: Array | None = None,
        axis_nodes: Sequence[Array] | None = None,
        source_mask: Array | None = None,
        key: EvalKey = None,
    ) -> Array:
        del key
        displacement = self.displacement(values, condition=condition)
        return self.transport(
            values,
            displacement,
            axis_nodes=axis_nodes,
            source_mask=source_mask,
        )


__all__ = ["MultiheadWarp", "WarpBoundaryMode"]
