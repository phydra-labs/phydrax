#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import coordax as cx
import jax.numpy as jnp

from ..discretization._tensor import AbstractStrongFormDiscretization
from ..discretization.spectral import TensorSpectralDiscretization
from ._targets import DiscreteMeasureTarget


def _spatial_shape(
    discretization: AbstractStrongFormDiscretization | TensorSpectralDiscretization,
    /,
) -> tuple[int, ...]:
    return (
        discretization.physical_shape
        if isinstance(discretization, TensorSpectralDiscretization)
        else discretization.state_shape
    )


def _spatial_dims(
    discretization: AbstractStrongFormDiscretization | TensorSpectralDiscretization,
    dims: str | Sequence[str] | None,
    /,
) -> tuple[str, ...]:
    rank = len(_spatial_shape(discretization))
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
    discretization: AbstractStrongFormDiscretization | TensorSpectralDiscretization,
    dims: tuple[str, ...],
    coordinate_dim: str,
    /,
) -> Any:
    points = discretization.points
    if points is None:
        return None
    data = jnp.asarray(points)
    shape = _spatial_shape(discretization)
    if data.shape[: len(dims)] != shape:
        if data.ndim < 1 or int(data.shape[0]) != discretization.num_points:
            raise ValueError(
                "Spatial points do not match the discretization physical shape."
            )
        data = data.reshape(shape + data.shape[1:])
    output_rank = data.ndim - len(dims)
    output_dims = (coordinate_dim,) if output_rank == 1 else (None,) * output_rank
    return cx.Field(data, dims=dims + output_dims)


def spatial_measure(
    discretization: AbstractStrongFormDiscretization | TensorSpectralDiscretization,
    /,
    *,
    spatial_dims: str | Sequence[str] | None = None,
    coordinate_dim: str = "coordinate",
    mask: Any | None = None,
    normalized: bool = False,
) -> DiscreteMeasureTarget:
    """Expose deterministic spatial quadrature as a named discrete measure."""
    if not isinstance(
        discretization,
        (AbstractStrongFormDiscretization, TensorSpectralDiscretization),
    ):
        raise TypeError(
            "discretization must provide a prepared strong-form or tensor spectral "
            "spatial measure."
        )
    dims = _spatial_dims(discretization, spatial_dims)
    coordinate_dim = str(coordinate_dim)
    if not coordinate_dim or coordinate_dim in dims:
        raise ValueError(
            "coordinate_dim must be non-empty and distinct from spatial_dims."
        )
    if isinstance(discretization, TensorSpectralDiscretization):
        weights = {
            dim: cx.Field(axis.quadrature_weights, dims=(dim,))
            for dim, axis in zip(dims, discretization.axes, strict=True)
        }
    else:
        weights = cx.Field(discretization.quadrature_weights, dims=dims)
    if mask is None or isinstance(mask, cx.Field):
        resolved_mask = mask
    else:
        mask_data = jnp.asarray(mask, dtype=bool)
        expected_shape = _spatial_shape(discretization)
        if mask_data.shape != expected_shape:
            raise ValueError(
                "Spatial masks must have the discretization physical shape "
                f"{expected_shape}; got {mask_data.shape}."
            )
        resolved_mask = cx.Field(mask_data, dims=dims)
    return DiscreteMeasureTarget(
        _spatial_points(discretization, dims, coordinate_dim),
        weights,
        axes=dims,
        mask=resolved_mask,
        normalized=normalized,
        provenance=f"spatial-discretization:{discretization.prepared_id}",
    )


__all__ = ["spatial_measure"]
