#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._interpolation import (
    fourier_interpolate,
    FourierEvaluationMethod,
)


def sample_fourier_grid(
    values: ArrayLike,
    coordinates: ArrayLike,
    /,
    *,
    spatial_ndim: int,
    axis_nodes: Sequence[ArrayLike] | None = None,
    periods: Sequence[ArrayLike] | None = None,
    method: FourierEvaluationMethod = "direct",
    tolerance: float | None = None,
    query_chunk_size: int | None = None,
    return_support: bool = False,
) -> Array | tuple[Array, Array]:
    """Evaluate a channel-last periodic tensor grid at paired coordinates.

    Values have shape ``batch_shape + spatial_shape + (channels,)`` and queries
    have shape ``batch_shape + query_shape + (spatial_ndim,)``. Without explicit
    axis nodes, every source axis is the endpoint-excluded uniform grid on
    ``[-1, 1)`` with period two. The direct method is roundoff-accurate;
    ``method="nufft"`` requires an explicit approximation ``tolerance``.
    """
    array = jnp.asarray(values)
    dimensions = int(spatial_ndim)
    if dimensions <= 0:
        raise ValueError("spatial_ndim must be positive.")
    if array.ndim < dimensions + 1:
        raise ValueError("values must end in spatial dimensions and one channel axis.")

    resolved_nodes = axis_nodes
    resolved_periods = periods
    if axis_nodes is None:
        spatial_shape = tuple(int(size) for size in array.shape[-dimensions - 1 : -1])
        real_dtype = (
            array.real.dtype if jnp.issubdtype(array.dtype, jnp.inexact) else float
        )
        resolved_nodes = tuple(
            -1.0 + 2.0 * jnp.arange(size, dtype=real_dtype) / float(size)
            for size in spatial_shape
        )
        if periods is None:
            resolved_periods = (2.0,) * dimensions

    interpolation = fourier_interpolate(
        array,
        coordinates,
        spatial_ndim=dimensions,
        payload_ndim=1,
        axis_nodes=resolved_nodes,
        periods=resolved_periods,
        method=method,
        tolerance=tolerance,
        query_chunk_size=query_chunk_size,
    )
    if return_support:
        return interpolation.values, interpolation.support
    return interpolation.values


__all__ = ["FourierEvaluationMethod", "sample_fourier_grid"]
