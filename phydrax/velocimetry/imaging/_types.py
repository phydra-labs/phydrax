#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


_IMAGE_COORDINATE_CONVENTION = "row-down-column-right"


def _shape2(value: Sequence[int], /, *, name: str) -> tuple[int, int]:
    shape = tuple(int(item) for item in value)
    if len(shape) != 2 or any(item < 1 for item in shape):
        raise ValueError(f"{name} must contain two positive dimensions.")
    return shape


def _vector2(value: Array | Sequence[float], /, *, name: str) -> Array:
    result = jnp.asarray(value, dtype=float)
    if result.shape != (2,):
        raise ValueError(f"{name} must have shape (2,).")
    return result


class ImageGeometry2D(StrictModule, NonTrainableState):
    """Immutable geometry for image components ordered as ``(row, column)``."""

    image_shape: tuple[int, int] = eqx.field(static=True)
    pixel_origin_rc: Array
    pixel_spacing_rc: Array
    coordinate_convention: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        image_shape: Sequence[int],
        /,
        *,
        pixel_origin_rc: Array | Sequence[float] = (0.0, 0.0),
        pixel_spacing_rc: Array | Sequence[float] = (1.0, 1.0),
        geometry_id: str | None = None,
    ):
        shape = _shape2(image_shape, name="image_shape")
        origin = _vector2(pixel_origin_rc, name="pixel_origin_rc")
        spacing = _vector2(pixel_spacing_rc, name="pixel_spacing_rc")
        if not bool(jnp.all(jnp.isfinite(origin))):
            raise ValueError("pixel_origin_rc must be finite.")
        if not bool(jnp.all(jnp.isfinite(spacing) & (spacing > 0.0))):
            raise ValueError("pixel_spacing_rc must be finite and positive.")
        resolved_id = geometry_id or canonical_fingerprint(
            {
                "kind": "image-geometry-2d",
                "image_shape": list(shape),
                "pixel_origin_rc": [float(value) for value in origin],
                "pixel_spacing_rc": [float(value) for value in spacing],
                "coordinate_convention": _IMAGE_COORDINATE_CONVENTION,
            }
        )
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("geometry_id must be a non-empty string.")
        self.image_shape = shape
        self.pixel_origin_rc = origin
        self.pixel_spacing_rc = spacing
        self.coordinate_convention = _IMAGE_COORDINATE_CONVENTION
        self.geometry_id = resolved_id


class ImagePair2D(StrictModule, NonTrainableState):
    """Two scalar images, validity masks, elapsed time, and retained provenance."""

    first: Array
    second: Array
    geometry: ImageGeometry2D
    first_mask: Array
    second_mask: Array
    delta_t: Array
    pair_id: str = eqx.field(static=True)
    provenance: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        first: Array,
        second: Array,
        geometry: ImageGeometry2D,
        /,
        *,
        first_mask: Array | None = None,
        second_mask: Array | None = None,
        delta_t: Array | float = 1.0,
        pair_id: str | None = None,
        provenance: Sequence[str] = (),
    ):
        if not isinstance(geometry, ImageGeometry2D):
            raise TypeError("geometry must be an ImageGeometry2D.")
        first_ = jnp.asarray(first)
        second_ = jnp.asarray(second)
        if first_.shape != geometry.image_shape or second_.shape != geometry.image_shape:
            raise ValueError("Both images must match geometry.image_shape.")
        if (
            not jnp.issubdtype(first_.dtype, jnp.number)
            or not jnp.issubdtype(second_.dtype, jnp.number)
            or jnp.issubdtype(first_.dtype, jnp.complexfloating)
            or jnp.issubdtype(second_.dtype, jnp.complexfloating)
        ):
            raise TypeError("Images must have real numeric dtypes.")
        first_mask_ = (
            jnp.ones(geometry.image_shape, dtype=bool)
            if first_mask is None
            else jnp.asarray(first_mask, dtype=bool)
        )
        second_mask_ = (
            jnp.ones(geometry.image_shape, dtype=bool)
            if second_mask is None
            else jnp.asarray(second_mask, dtype=bool)
        )
        if (
            first_mask_.shape != geometry.image_shape
            or second_mask_.shape != geometry.image_shape
        ):
            raise ValueError("Image masks must match geometry.image_shape.")
        delta_t_ = jnp.asarray(delta_t, dtype=float)
        if delta_t_.shape != ():
            raise ValueError("delta_t must be scalar.")
        provenance_ = tuple(str(item) for item in provenance)
        if any(not item for item in provenance_):
            raise ValueError("provenance entries must be non-empty strings.")
        resolved_id = pair_id or canonical_fingerprint(
            {
                "kind": "image-pair-2d",
                "geometry_id": geometry.geometry_id,
                "image_shape": list(geometry.image_shape),
                "image_dtypes": [str(first_.dtype), str(second_.dtype)],
                "mask_dtypes": [str(first_mask_.dtype), str(second_mask_.dtype)],
                "provenance": list(provenance_),
            }
        )
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("pair_id must be a non-empty string.")
        self.first = first_
        self.second = second_
        self.geometry = geometry
        self.first_mask = first_mask_ & jnp.isfinite(first_)
        self.second_mask = second_mask_ & jnp.isfinite(second_)
        self.delta_t = delta_t_
        self.pair_id = resolved_id
        self.provenance = provenance_


class DenseDisplacementField2D(StrictModule, NonTrainableState):
    """A fixed-shape displacement field in ``(row_down, column_right)`` pixels."""

    positions_rc: Array
    displacement_rc: Array
    valid: Array
    geometry_id: str = eqx.field(static=True)
    field_id: str = eqx.field(static=True)
    provenance: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        positions_rc: Array,
        displacement_rc: Array,
        valid: Array,
        /,
        *,
        geometry_id: str,
        field_id: str | None = None,
        provenance: Sequence[str] = (),
    ):
        positions = jnp.asarray(positions_rc)
        displacement = jnp.asarray(displacement_rc)
        if jnp.issubdtype(positions.dtype, jnp.complexfloating) or jnp.issubdtype(
            displacement.dtype, jnp.complexfloating
        ):
            raise TypeError("Displacement fields must contain real values.")
        if not jnp.issubdtype(positions.dtype, jnp.inexact):
            positions = positions.astype(float)
        if not jnp.issubdtype(displacement.dtype, jnp.inexact):
            displacement = displacement.astype(float)
        valid_ = jnp.asarray(valid, dtype=bool)
        if positions.ndim < 2 or positions.shape[-1] != 2:
            raise ValueError("positions_rc must have shape (..., 2).")
        if displacement.shape != positions.shape:
            raise ValueError("displacement_rc must match positions_rc.")
        if valid_.shape != positions.shape[:-1]:
            raise ValueError("valid must match the field grid shape.")
        if not isinstance(geometry_id, str) or not geometry_id:
            raise ValueError("geometry_id must be a non-empty string.")
        provenance_ = tuple(str(item) for item in provenance)
        if any(not item for item in provenance_):
            raise ValueError("provenance entries must be non-empty strings.")
        resolved_id = field_id or canonical_fingerprint(
            {
                "kind": "dense-displacement-field-2d",
                "geometry_id": geometry_id,
                "grid_shape": list(valid_.shape),
                "position_dtype": str(positions.dtype),
                "displacement_dtype": str(displacement.dtype),
                "provenance": list(provenance_),
            }
        )
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("field_id must be a non-empty string.")
        finite = jnp.all(jnp.isfinite(positions) & jnp.isfinite(displacement), axis=-1)
        self.positions_rc = positions
        self.displacement_rc = displacement
        self.valid = valid_ & finite
        self.geometry_id = geometry_id
        self.field_id = resolved_id
        self.provenance = provenance_

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return self.valid.shape


class ImageSample2D(StrictModule):
    """Sampled scalar or channel-valued image data and geometric support validity."""

    values: Array
    valid: Array

    def __init__(self, values: Array, valid: Array, /):
        values_ = jnp.asarray(values)
        valid_ = jnp.asarray(valid, dtype=bool)
        if values_.shape[: valid_.ndim] != valid_.shape:
            raise ValueError("valid must match the leading sampled value shape.")
        self.values = values_
        self.valid = valid_


__all__ = ["DenseDisplacementField2D", "ImageGeometry2D", "ImagePair2D", "ImageSample2D"]
