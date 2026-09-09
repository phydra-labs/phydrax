#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Differentiable nonperiodic image sampling and warping."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._interpolation import apply_gather_stencil, rectilinear_stencil
from .._strict import StrictModule
from ._plane import ImagePlaneSupport


class ImageSample2D(StrictModule):
    """Sampled scalar or channel-valued image data and support validity."""

    values: Array
    valid: Array

    def __init__(self, values: ArrayLike, valid: ArrayLike, /):
        values_ = jnp.asarray(values)
        valid_ = jnp.asarray(valid, dtype=bool)
        if values_.shape[: valid_.ndim] != valid_.shape:
            raise ValueError("valid must match the leading sampled value shape.")
        self.values = values_
        self.valid = valid_


def image_coordinates(support_or_shape: ImagePlaneSupport | Sequence[int], /) -> Array:
    """Return pixel-index coordinates as row-down and column-right."""
    if isinstance(support_or_shape, ImagePlaneSupport):
        shape = support_or_shape.image_shape
    else:
        shape = tuple(int(item) for item in support_or_shape)
        if len(shape) != 2 or any(item < 1 for item in shape):
            raise ValueError("Image shape must contain two positive dimensions.")
    rows, columns = jnp.meshgrid(
        jnp.arange(shape[0], dtype=float),
        jnp.arange(shape[1], dtype=float),
        indexing="ij",
    )
    return jnp.stack((rows, columns), axis=-1)


def bilinear_sample(
    image: ArrayLike,
    coordinates_rc: ArrayLike,
    /,
    *,
    valid_mask: ArrayLike | None = None,
    fill_value: ArrayLike | float = 0.0,
) -> ImageSample2D:
    """Bilinearly sample through the strict nonperiodic rectilinear map."""
    values = jnp.asarray(image)
    coordinates = jnp.asarray(coordinates_rc, dtype=float)
    if values.ndim < 2:
        raise ValueError("image must have at least two dimensions.")
    if coordinates.ndim < 1 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates_rc must have shape (..., 2).")
    source_mask = (
        jnp.ones(values.shape[:2], dtype=bool)
        if valid_mask is None
        else jnp.asarray(valid_mask, dtype=bool)
    )
    if source_mask.shape != values.shape[:2]:
        raise ValueError("valid_mask must match the first two image dimensions.")
    source_shape = values.shape[:2]
    finite_coordinates = jnp.all(jnp.isfinite(coordinates), axis=-1)
    safe_coordinates = jnp.where(finite_coordinates[..., None], coordinates, 0.0)
    working_values = values
    working_mask = source_mask
    if source_shape[0] == 1:
        working_values = jnp.repeat(working_values, 2, axis=0)
        working_mask = jnp.repeat(working_mask, 2, axis=0)
    if source_shape[1] == 1:
        working_values = jnp.repeat(working_values, 2, axis=1)
        working_mask = jnp.repeat(working_mask, 2, axis=1)
    nodes = (
        jnp.arange(working_values.shape[0], dtype=coordinates.dtype),
        jnp.arange(working_values.shape[1], dtype=coordinates.dtype),
    )
    stencil = rectilinear_stencil(
        nodes,
        safe_coordinates,
        boundary=("constant", "constant"),
    )
    payload_shape = values.shape[2:]
    channels = 1
    for size in payload_shape:
        channels *= int(size)
    flat_values = working_values.reshape(
        (working_values.shape[0] * working_values.shape[1], channels)
    )
    interpolation = apply_gather_stencil(
        flat_values,
        stencil,
        source_mask=working_mask.reshape((-1,)),
        mask_mode="strict",
    )
    support = (
        interpolation.support
        & finite_coordinates
        & (coordinates[..., 0] >= 0.0)
        & (coordinates[..., 0] <= source_shape[0] - 1)
        & (coordinates[..., 1] >= 0.0)
        & (coordinates[..., 1] <= source_shape[1] - 1)
    )
    sampled = interpolation.values.reshape(coordinates.shape[:-1] + payload_shape)
    expanded_support = support.reshape(support.shape + (1,) * len(payload_shape))
    output = jnp.where(
        expanded_support,
        sampled,
        jnp.asarray(fill_value, dtype=sampled.dtype),
    )
    return ImageSample2D(output, support)


def backward_warp(
    image: ArrayLike,
    displacement_rc: ArrayLike,
    /,
    *,
    valid_mask: ArrayLike | None = None,
    fill_value: ArrayLike | float = 0.0,
) -> ImageSample2D:
    """Backward warp so output[r,c] samples input[r-dr,c-dc]."""
    displacement = jnp.asarray(displacement_rc, dtype=float)
    if displacement.ndim != 3 or displacement.shape[-1] != 2:
        raise ValueError("displacement_rc must have shape (rows, columns, 2).")
    if tuple(displacement.shape[:2]) != tuple(jnp.shape(image)[:2]):
        raise ValueError("displacement_rc must match the image spatial shape.")
    coordinates = image_coordinates(displacement.shape[:2]) - displacement
    return bilinear_sample(
        image,
        coordinates,
        valid_mask=valid_mask,
        fill_value=fill_value,
    )


__all__ = [
    "ImageSample2D",
    "backward_warp",
    "bilinear_sample",
    "image_coordinates",
]
