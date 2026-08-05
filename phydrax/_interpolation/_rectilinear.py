#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._stencil import GatherStencil
from ._types import InterpolationCapabilities


RectilinearBoundaryMode: TypeAlias = Literal[
    "periodic",
    "reflect",
    "clamp",
    "constant",
]
AxisBound: TypeAlias = tuple[float, float] | None


RECTILINEAR_CAPABILITIES = InterpolationCapabilities(
    partition_of_unity=True,
    nonnegative_value_weights=True,
    local_support=True,
    mask_renormalizable=True,
    tensor_product_composable=True,
    maximum_explicit_derivative_order=1,
)


def _axis_metadata(
    nodes: ArrayLike,
    /,
    *,
    mode: RectilinearBoundaryMode,
    period: float | Array | None,
    axis_bound: AxisBound,
    dtype: jnp.dtype,
) -> tuple[Array, Array, Array, Array | None]:
    raw_values = jnp.asarray(nodes)
    if jnp.issubdtype(raw_values.dtype, jnp.complexfloating):
        raise TypeError("Rectilinear coordinates must be real-valued.")
    values = raw_values.astype(dtype).reshape((-1,))
    if int(values.size) < 2:
        raise ValueError("Every rectilinear axis must contain at least two nodes.")
    spacing = jnp.diff(values)
    values = eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)) | jnp.any(spacing <= 0.0),
        "Rectilinear nodes must be finite and strictly increasing.",
    )

    period_value: Array | None = None
    if mode == "periodic":
        if period is None:
            values = eqx.error_if(
                values,
                jnp.logical_not(
                    jnp.allclose(
                        spacing,
                        jnp.mean(spacing),
                        rtol=1e-5,
                        atol=1e-8,
                    )
                ),
                "A nonuniform periodic axis requires an explicit period.",
            )
            period_value = values[-1] - values[0] + jnp.mean(spacing)
        else:
            period_value = jnp.asarray(period, dtype=dtype)
        values = eqx.error_if(
            values,
            ~jnp.isfinite(period_value) | (period_value <= values[-1] - values[0]),
            "A periodic axis period must exceed its sampled span.",
        )

    if axis_bound is None:
        lower = values[0]
        upper = values[-1] if period_value is None else values[0] + period_value
    else:
        lower = jnp.asarray(axis_bound[0], dtype=dtype)
        upper = jnp.asarray(axis_bound[1], dtype=dtype)
        values = eqx.error_if(
            values,
            ~jnp.isfinite(lower)
            | ~jnp.isfinite(upper)
            | (upper <= lower)
            | (values[0] < lower)
            | (values[-1] > upper),
            "Rectilinear axis bounds must be finite, ordered, and contain the nodes.",
        )
        if period_value is not None:
            values = eqx.error_if(
                values,
                jnp.logical_not(
                    jnp.isclose(
                        upper - lower,
                        period_value,
                        rtol=1e-6,
                        atol=1e-8,
                    )
                ),
                "Periodic axis bounds must span exactly one period.",
            )
    return values, lower, upper, period_value


def rectilinear_stencil(
    axis_nodes: Sequence[ArrayLike],
    coordinates: ArrayLike,
    /,
    *,
    boundary: Sequence[RectilinearBoundaryMode],
    batch_shape: Sequence[int] = (),
    periods: Sequence[float | Array | None] | None = None,
    axis_bounds: Sequence[AxisBound] | None = None,
) -> GatherStencil:
    """Build a multilinear gather map for a batch of rectilinear grids."""
    nodes_input = tuple(axis_nodes)
    modes = tuple(boundary)
    dimensions = len(nodes_input)
    if dimensions <= 0 or len(modes) != dimensions:
        raise ValueError("axis_nodes and boundary must contain the same nonzero axes.")
    invalid_modes = tuple(
        mode for mode in modes if mode not in ("periodic", "reflect", "clamp", "constant")
    )
    if invalid_modes:
        raise ValueError(f"Unsupported rectilinear boundary modes: {invalid_modes}.")

    batch = tuple(int(size) for size in batch_shape)
    if any(size <= 0 for size in batch):
        raise ValueError("batch_shape dimensions must be positive.")
    query = jnp.asarray(coordinates)
    if jnp.issubdtype(query.dtype, jnp.complexfloating):
        raise TypeError("Rectilinear query coordinates must be real-valued.")
    dtype = jnp.result_type(query.dtype, float)
    query = query.astype(dtype)
    if query.ndim < len(batch) + 1 or int(query.shape[-1]) != dimensions:
        raise ValueError(
            "coordinates must have shape batch_shape + query_shape + "
            f"({dimensions},); got {query.shape}."
        )
    if tuple(int(size) for size in query.shape[: len(batch)]) != batch:
        raise ValueError(
            f"Coordinate batch shape must be {batch}; got {query.shape[: len(batch)]}."
        )
    query = eqx.error_if(
        query,
        jnp.any(~jnp.isfinite(query)),
        "Rectilinear query coordinates must be finite.",
    )
    query_shape = tuple(int(size) for size in query.shape[len(batch) : -1])

    period_values = (None,) * dimensions if periods is None else tuple(periods)
    bounds = (None,) * dimensions if axis_bounds is None else tuple(axis_bounds)
    if len(period_values) != dimensions or len(bounds) != dimensions:
        raise ValueError("periods and axis_bounds must provide one entry per axis.")

    nodes: list[Array] = []
    domain_lowers: list[Array] = []
    domain_uppers: list[Array] = []
    resolved_periods: list[Array | None] = []
    for values, mode, period, axis_bound in zip(
        nodes_input,
        modes,
        period_values,
        bounds,
        strict=True,
    ):
        node, lower, upper, resolved_period = _axis_metadata(
            values,
            mode=mode,
            period=period,
            axis_bound=axis_bound,
            dtype=dtype,
        )
        nodes.append(node)
        domain_lowers.append(lower)
        domain_uppers.append(upper)
        resolved_periods.append(resolved_period)

    spatial_shape = tuple(int(values.size) for values in nodes)
    spatial_count = prod(spatial_shape)
    index_dtype = jnp.int64 if bool(jax.config.read("jax_enable_x64")) else jnp.int32
    if spatial_count > jnp.iinfo(index_dtype).max:
        raise ValueError("Rectilinear grid is too large for JAX gather indices.")

    lower_indices: list[Array] = []
    upper_indices: list[Array] = []
    fractions: list[Array] = []
    outside = jnp.zeros(batch + query_shape, dtype=bool)
    for axis, (size, mode, values, domain_lower, domain_upper, period) in enumerate(
        zip(
            spatial_shape,
            modes,
            nodes,
            domain_lowers,
            domain_uppers,
            resolved_periods,
            strict=True,
        )
    ):
        coordinate = query[..., axis]
        if mode == "periodic":
            assert period is not None
            coordinate = jnp.mod(coordinate - domain_lower, period) + domain_lower
            upper_raw = jnp.searchsorted(values, coordinate, side="right")
            lower_index = jnp.mod(upper_raw - 1, size)
            upper_index = jnp.mod(upper_raw, size)
            lower_coordinate = values[lower_index] - jnp.where(
                upper_raw == 0,
                period,
                0.0,
            )
            upper_coordinate = values[upper_index] + jnp.where(
                upper_raw == size,
                period,
                0.0,
            )
        else:
            if mode == "reflect":
                extent = domain_upper - domain_lower
                phase = jnp.mod(coordinate - domain_lower, 2.0 * extent)
                coordinate = jnp.where(
                    phase <= extent,
                    domain_lower + phase,
                    domain_upper - (phase - extent),
                )
            elif mode == "clamp":
                coordinate = jnp.clip(coordinate, domain_lower, domain_upper)
            else:
                outside = (
                    outside | (coordinate < domain_lower) | (coordinate > domain_upper)
                )
                coordinate = jnp.clip(coordinate, domain_lower, domain_upper)
            upper_raw = jnp.searchsorted(values, coordinate, side="right")
            upper_index = jnp.clip(upper_raw, 1, size - 1)
            lower_index = upper_index - 1
            lower_coordinate = values[lower_index]
            upper_coordinate = values[upper_index]

        denominator = jnp.maximum(
            upper_coordinate - lower_coordinate,
            jnp.finfo(dtype).eps,
        )
        fraction = jnp.clip(
            (coordinate - lower_coordinate) / denominator,
            0.0,
            1.0,
        )
        lower_indices.append(lower_index.astype(index_dtype))
        upper_indices.append(upper_index.astype(index_dtype))
        fractions.append(fraction)

    corner_indices: list[Array] = []
    corner_weights: list[Array] = []
    for corner in range(1 << dimensions):
        axis_indices: list[Array] = []
        weight = jnp.ones(batch + query_shape, dtype=dtype)
        for axis in range(dimensions):
            upper_corner = bool(corner & (1 << axis))
            axis_indices.append(
                upper_indices[axis] if upper_corner else lower_indices[axis]
            )
            fraction = fractions[axis]
            weight = weight * (fraction if upper_corner else 1.0 - fraction)
        linear_index = axis_indices[0]
        for axis in range(1, dimensions):
            linear_index = linear_index * spatial_shape[axis] + axis_indices[axis]
        corner_indices.append(linear_index)
        corner_weights.append(weight)

    batch_count = prod(batch) if batch else 1
    batch_index = jnp.arange(batch_count, dtype=index_dtype).reshape(
        batch + (1,) * len(query_shape)
    )
    offset = batch_index * spatial_count
    stacked_indices = jnp.stack(corner_indices, axis=-1) + offset[..., None]
    weights = jnp.stack(corner_weights, axis=-1)
    return GatherStencil(
        indices=stacked_indices,
        weights=weights,
        source_size=batch_count * spatial_count,
        support=~outside,
    )


__all__ = [
    "AxisBound",
    "RECTILINEAR_CAPABILITIES",
    "RectilinearBoundaryMode",
    "rectilinear_stencil",
]
