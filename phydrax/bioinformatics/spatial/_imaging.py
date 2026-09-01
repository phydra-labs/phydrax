#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal, Sequence

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._frame import SpatialUnit


AxisType = Literal["space", "channel", "time"]


@dataclass(frozen=True, slots=True)
class ImageAxis:
    name: str
    axis_type: AxisType
    unit: SpatialUnit | str | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Image axis names must be non-empty.")
        if self.axis_type not in ("space", "channel", "time"):
            raise ValueError("axis_type must be 'space', 'channel', or 'time'.")
        if (
            self.axis_type == "space"
            and self.unit is not None
            and not isinstance(self.unit, SpatialUnit)
        ):
            raise TypeError("Spatial image axis units must be SpatialUnit or None.")
        if (
            self.axis_type == "time"
            and self.unit is not None
            and (not isinstance(self.unit, str) or not self.unit.strip())
        ):
            raise TypeError(
                "Temporal image axis units must be non-empty strings or None."
            )
        if self.axis_type == "channel" and self.unit is not None:
            raise ValueError("Channel axes cannot carry physical units.")


@dataclass(frozen=True, slots=True)
class ImagePyramidLevel:
    path: str
    shape: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    scale: tuple[float, ...]
    translation: tuple[float, ...]

    def __init__(
        self,
        path: str,
        shape: Sequence[int],
        chunk_shape: Sequence[int],
        scale: Sequence[float],
        translation: Sequence[float],
        /,
    ):
        if not isinstance(path, str) or not path.strip():
            raise ValueError("Pyramid level path must be non-empty.")
        shape_ = tuple(int(value) for value in shape)
        chunks_ = tuple(int(value) for value in chunk_shape)
        scale_ = tuple(float(value) for value in scale)
        translation_ = tuple(float(value) for value in translation)
        if not shape_ or any(value <= 0 for value in shape_):
            raise ValueError("Pyramid level dimensions must be positive.")
        if len(chunks_) != len(shape_) or any(value <= 0 for value in chunks_):
            raise ValueError("chunk_shape must align with shape and be positive.")
        if len(scale_) != len(shape_) or any(
            not np.isfinite(value) or value <= 0.0 for value in scale_
        ):
            raise ValueError("scale must align with shape and be finite and positive.")
        if len(translation_) != len(shape_) or any(
            not np.isfinite(value) for value in translation_
        ):
            raise ValueError("translation must align with shape and be finite.")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "shape", shape_)
        object.__setattr__(self, "chunk_shape", chunks_)
        object.__setattr__(self, "scale", scale_)
        object.__setattr__(self, "translation", translation_)

    @property
    def chunk_grid_shape(self) -> tuple[int, ...]:
        return tuple(
            (size + chunk - 1) // chunk
            for size, chunk in zip(self.shape, self.chunk_shape, strict=True)
        )


@dataclass(frozen=True, slots=True)
class ImagePyramidMetadata:
    """Host OME-NGFF image-pyramid metadata after strict semantic validation."""

    name: str
    axes: tuple[ImageAxis, ...]
    levels: tuple[ImagePyramidLevel, ...]
    ngff_version: str

    def __init__(
        self,
        name: str,
        axes: Sequence[ImageAxis],
        levels: Sequence[ImagePyramidLevel],
        /,
        *,
        ngff_version: str,
    ):
        axes_ = tuple(axes)
        levels_ = tuple(levels)
        if not isinstance(name, str):
            raise TypeError("Pyramid name must be a string.")
        if not axes_ or any(not isinstance(axis, ImageAxis) for axis in axes_):
            raise TypeError("axes must be a non-empty sequence of ImageAxis.")
        if len({axis.name for axis in axes_}) != len(axes_):
            raise ValueError("Image pyramid axis names must be unique.")
        if not levels_ or any(
            not isinstance(level, ImagePyramidLevel) for level in levels_
        ):
            raise TypeError("levels must be a non-empty sequence of ImagePyramidLevel.")
        if len({level.path for level in levels_}) != len(levels_):
            raise ValueError("Image pyramid level paths must be unique.")
        if any(len(level.shape) != len(axes_) for level in levels_):
            raise ValueError("Every pyramid level must align with the declared axes.")
        if not isinstance(ngff_version, str) or not ngff_version.strip():
            raise ValueError("ngff_version must be non-empty.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "axes", axes_)
        object.__setattr__(self, "levels", levels_)
        object.__setattr__(self, "ngff_version", ngff_version)


@dataclass(frozen=True, slots=True)
class ImageTileMetadata:
    """Host identity and clipped array extent of one pyramid chunk/tile."""

    level_path: str
    tile_index: tuple[int, ...]
    start: tuple[int, ...]
    stop: tuple[int, ...]
    touches_array_boundary: tuple[bool, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(
            end - begin for begin, end in zip(self.start, self.stop, strict=True)
        )


@dataclass(frozen=True, slots=True)
class BoundedImagePatchPlan:
    """Static patch allocation independent of a dynamic origin and request size."""

    capacity_shape: tuple[int, ...]

    def __init__(self, capacity_shape: Sequence[int], /):
        shape = tuple(int(value) for value in capacity_shape)
        if not shape or any(value <= 0 for value in shape):
            raise ValueError("capacity_shape entries must be positive.")
        object.__setattr__(self, "capacity_shape", shape)


class ImagePatchStatus(IntEnum):
    OK = 0
    OUT_OF_BOUNDS = 1
    CAPACITY_EXCEEDED = 2
    INVALID_REQUEST = 3


class ImagePatchEvidence(StrictModule):
    start: Array
    requested_shape: Array
    image_shape: Array
    capacity_shape: Array
    valid_element_count: Array
    touched_chunk_count: Array
    crosses_chunk_boundary: Array


_PATCH_CONTRACT = BioinformaticsMethodContract(
    "bounded_image_patch_extraction",
    MethodKind.EXACT_MODEL,
    ExecutionKind.FLOATING_POINT_DIRECT,
    DifferentiationKind.EXACT_AD,
    OutputKind.STRUCTURED,
    conditioning_statement=(
        "Array axes and chunk_shape are interpreted in the declared pyramid-level order."
    ),
    truncation_statement=(
        "Requests exceeding capacity return no valid elements; boundary overlap is padded "
        "and explicitly marked invalid rather than presented as a complete patch."
    ),
    capacity_semantics=(
        "The returned patch always has BoundedImagePatchPlan.capacity_shape."
    ),
    nondifferentiable_outputs=("valid_mask", "status", "evidence"),
)


class BoundedImagePatch(StrictModule):
    values: Array
    valid_mask: Array
    valid: Array
    status: Array
    evidence: ImagePatchEvidence
    method_contract: BioinformaticsMethodContract


def extract_bounded_image_patch(
    image: Any,
    start: Any,
    requested_shape: Any,
    plan: BoundedImagePatchPlan,
    /,
    *,
    chunk_shape: Sequence[int] | None = None,
    fill_value: float = 0.0,
) -> BoundedImagePatch:
    """Gather a fixed allocation across arbitrary chunk/tile boundaries."""
    if not isinstance(plan, BoundedImagePatchPlan):
        raise TypeError("plan must be a BoundedImagePatchPlan.")
    values = jnp.asarray(image)
    dimension = values.ndim
    if dimension != len(plan.capacity_shape):
        raise ValueError("Image rank must match patch capacity rank.")
    start_ = jnp.asarray(start, dtype=jnp.int32)
    requested = jnp.asarray(requested_shape, dtype=jnp.int32)
    if start_.shape != (dimension,) or requested.shape != (dimension,):
        raise ValueError("start and requested_shape must have one entry per image axis.")
    chunks = tuple(
        int(value) for value in (values.shape if chunk_shape is None else chunk_shape)
    )
    if len(chunks) != dimension or any(value <= 0 for value in chunks):
        raise ValueError("chunk_shape must align with image rank and be positive.")

    image_shape = jnp.asarray(values.shape, dtype=jnp.int32)
    capacity = jnp.asarray(plan.capacity_shape, dtype=jnp.int32)
    request_valid = jnp.all(requested > 0)
    overflow = jnp.any(requested > capacity)
    grid = jnp.meshgrid(
        *(jnp.arange(size, dtype=jnp.int32) for size in plan.capacity_shape),
        indexing="ij",
    )
    global_indices = tuple(start_[axis] + grid[axis] for axis in range(dimension))
    safe_indices = tuple(
        jnp.clip(global_indices[axis], 0, int(values.shape[axis]) - 1)
        for axis in range(dimension)
    )
    gathered = values[safe_indices]
    inside_request = jnp.ones(plan.capacity_shape, dtype=bool)
    inside_image = jnp.ones(plan.capacity_shape, dtype=bool)
    for axis in range(dimension):
        inside_request = inside_request & (grid[axis] < requested[axis])
        inside_image = (
            inside_image
            & (global_indices[axis] >= 0)
            & (global_indices[axis] < image_shape[axis])
        )
    valid_mask = inside_request & inside_image & ~overflow & request_valid
    patch = jnp.where(valid_mask, gathered, jnp.asarray(fill_value, dtype=values.dtype))
    end = start_ + requested
    wholly_inside = request_valid & jnp.all(start_ >= 0) & jnp.all(end <= image_shape)

    overlap_start = jnp.maximum(start_, 0)
    overlap_end = jnp.minimum(end, image_shape)
    overlap = jnp.all(overlap_end > overlap_start) & ~overflow & request_valid
    chunk_array = jnp.asarray(chunks, dtype=jnp.int32)
    first_chunk = overlap_start // chunk_array
    last_chunk = (jnp.maximum(overlap_end - 1, overlap_start)) // chunk_array
    chunk_span = jnp.where(overlap, last_chunk - first_chunk + 1, 0)
    touched_chunks = jnp.prod(chunk_span, dtype=jnp.int32)
    crosses = jnp.any(chunk_span > 1) & overlap
    status = jnp.where(
        ~request_valid,
        int(ImagePatchStatus.INVALID_REQUEST),
        jnp.where(
            overflow,
            int(ImagePatchStatus.CAPACITY_EXCEEDED),
            jnp.where(
                wholly_inside,
                int(ImagePatchStatus.OK),
                int(ImagePatchStatus.OUT_OF_BOUNDS),
            ),
        ),
    ).astype(jnp.int32)
    evidence = ImagePatchEvidence(
        start=start_,
        requested_shape=requested,
        image_shape=image_shape,
        capacity_shape=capacity,
        valid_element_count=jnp.sum(valid_mask, dtype=jnp.int32),
        touched_chunk_count=touched_chunks,
        crosses_chunk_boundary=crosses,
    )
    return BoundedImagePatch(
        values=patch,
        valid_mask=valid_mask,
        valid=(request_valid & ~overflow & wholly_inside),
        status=status,
        evidence=evidence,
        method_contract=_PATCH_CONTRACT,
    )


def pyramid_level_tile_bounds(
    level: ImagePyramidLevel,
    tile_index: Sequence[int],
    /,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return clipped half-open array bounds for one declared chunk/tile."""
    if not isinstance(level, ImagePyramidLevel):
        raise TypeError("level must be an ImagePyramidLevel.")
    index = tuple(int(value) for value in tile_index)
    if len(index) != len(level.shape):
        raise ValueError("tile_index must align with the pyramid level rank.")
    if any(
        value < 0 or value >= grid
        for value, grid in zip(index, level.chunk_grid_shape, strict=True)
    ):
        raise ValueError("tile_index lies outside the chunk grid.")
    start = tuple(
        value * chunk for value, chunk in zip(index, level.chunk_shape, strict=True)
    )
    stop = tuple(
        min(begin + chunk, size)
        for begin, chunk, size in zip(start, level.chunk_shape, level.shape, strict=True)
    )
    return start, stop


def image_tile_metadata(
    level: ImagePyramidLevel,
    tile_index: Sequence[int],
    /,
) -> ImageTileMetadata:
    """Return host metadata for one exact pyramid tile, including edge clipping."""
    index = tuple(int(value) for value in tile_index)
    start, stop = pyramid_level_tile_bounds(level, index)
    touches = tuple(
        begin == 0 or end == size
        for begin, end, size in zip(start, stop, level.shape, strict=True)
    )
    return ImageTileMetadata(level.path, index, start, stop, touches)


__all__ = [
    "BoundedImagePatch",
    "BoundedImagePatchPlan",
    "ImageAxis",
    "ImagePatchEvidence",
    "ImagePatchStatus",
    "ImagePyramidLevel",
    "ImageTileMetadata",
    "ImagePyramidMetadata",
    "extract_bounded_image_patch",
    "image_tile_metadata",
    "pyramid_level_tile_bounds",
]
