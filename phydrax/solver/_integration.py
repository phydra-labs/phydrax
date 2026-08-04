#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import coordax as cx
import jax.numpy as jnp

from ..integration import DiscreteMeasureTarget
from ._spatial import AbstractSpatialDiscretization, TensorGridDiscretization


def _spatial_dims(
    discretization: AbstractSpatialDiscretization,
    dims: str | Sequence[str] | None,
    /,
) -> tuple[str, ...]:
    rank = len(discretization.state_shape)
    if dims is None:
        resolved = (
            ("space",) if rank == 1 else tuple(f"space_{index}" for index in range(rank))
        )
    elif isinstance(dims, str):
        resolved = (dims,)
    else:
        resolved = tuple(str(dim) for dim in dims)
    if len(resolved) != rank:
        raise ValueError(f"spatial_dims must contain exactly {rank} names.")
    if any(not dim for dim in resolved) or len(set(resolved)) != len(resolved):
        raise ValueError("spatial_dims must contain distinct non-empty names.")
    return resolved


def _spatial_points(
    discretization: AbstractSpatialDiscretization,
    dims: tuple[str, ...],
    coordinate_dim: str,
    /,
) -> Any:
    points = discretization.points
    if points is None:
        return None
    data = jnp.asarray(points)
    if data.shape[: len(dims)] != discretization.state_shape:
        if data.ndim < 1 or int(data.shape[0]) != discretization.num_points:
            raise ValueError(
                "Spatial points do not match the discretization state shape."
            )
        data = data.reshape(discretization.state_shape + data.shape[1:])
    output_rank = data.ndim - len(dims)
    output_dims = (coordinate_dim,) if output_rank == 1 else (None,) * output_rank
    return cx.Field(data, dims=dims + output_dims)


def spatial_measure(
    discretization: AbstractSpatialDiscretization,
    /,
    *,
    spatial_dims: str | Sequence[str] | None = None,
    coordinate_dim: str = "coordinate",
    mask: Any | None = None,
    normalized: bool = False,
) -> DiscreteMeasureTarget:
    """Expose deterministic spatial quadrature as a named discrete measure."""
    if not isinstance(discretization, AbstractSpatialDiscretization):
        raise TypeError("discretization must be an AbstractSpatialDiscretization.")
    dims = _spatial_dims(discretization, spatial_dims)
    coordinate_dim = str(coordinate_dim)
    if not coordinate_dim or coordinate_dim in dims:
        raise ValueError(
            "coordinate_dim must be non-empty and distinct from spatial_dims."
        )
    if isinstance(discretization, TensorGridDiscretization):
        weights = {
            dim: cx.Field(jnp.asarray(axis.quad_weights), dims=(dim,))
            for dim, axis in zip(dims, discretization.axes, strict=True)
        }
    else:
        weights = cx.Field(discretization.quadrature_weights, dims=dims)
    if mask is None or isinstance(mask, cx.Field):
        resolved_mask = mask
    else:
        mask_data = jnp.asarray(mask, dtype=bool)
        if mask_data.shape != discretization.state_shape:
            raise ValueError(
                "Spatial masks must have the discretization state shape "
                f"{discretization.state_shape}; got {mask_data.shape}."
            )
        resolved_mask = cx.Field(mask_data, dims=dims)
    return DiscreteMeasureTarget(
        _spatial_points(discretization, dims, coordinate_dim),
        weights,
        axes=dims,
        mask=resolved_mask,
        normalized=normalized,
        provenance=f"spatial-discretization:{discretization.discretization_id}",
    )


__all__ = ["spatial_measure"]
