#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generic two-dimensional image-plane support."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


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


class ImagePlaneSupport(StrictModule, NonTrainableState):
    """Image samples ordered as row-down and column-right."""

    image_shape: tuple[int, int] = eqx.field(static=True)
    sample_shape: tuple[int, int] = eqx.field(static=True)
    pixel_origin_rc: Array
    pixel_spacing_rc: Array
    coordinate_convention: str = eqx.field(static=True)
    detector_frame_id: str | None = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        image_shape: Sequence[int],
        /,
        *,
        pixel_origin_rc: Array | Sequence[float] = (0.0, 0.0),
        pixel_spacing_rc: Array | Sequence[float] = (1.0, 1.0),
        detector_frame_id: str | None = None,
        support_id: str | None = None,
    ):
        shape = _shape2(image_shape, name="image_shape")
        origin = _vector2(pixel_origin_rc, name="pixel_origin_rc")
        spacing = _vector2(pixel_spacing_rc, name="pixel_spacing_rc")
        if not bool(jnp.all(jnp.isfinite(origin))):
            raise ValueError("pixel_origin_rc must be finite.")
        if not bool(jnp.all(jnp.isfinite(spacing) & (spacing > 0.0))):
            raise ValueError("pixel_spacing_rc must be finite and positive.")
        frame = detector_frame_id
        if frame is not None and (
            not isinstance(frame, str) or not frame or frame.strip() != frame
        ):
            raise ValueError("detector_frame_id must be non-empty stripped text or None.")
        resolved_id = support_id or canonical_fingerprint(
            {
                "kind": "image-plane-support",
                "image_shape": list(shape),
                "pixel_origin_rc": [float(value) for value in origin],
                "pixel_spacing_rc": [float(value) for value in spacing],
                "coordinate_convention": _IMAGE_COORDINATE_CONVENTION,
                "detector_frame": frame,
            }
        )
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("support_id must be a non-empty string.")
        self.image_shape = shape
        self.sample_shape = shape
        self.pixel_origin_rc = origin
        self.pixel_spacing_rc = spacing
        self.coordinate_convention = _IMAGE_COORDINATE_CONVENTION
        self.detector_frame_id = frame
        self.support_id = resolved_id

    @property
    def geometry_id(self) -> str:
        """Image-geometry identity used by displacement and PIV contracts."""
        return self.support_id


__all__ = ["ImagePlaneSupport"]
