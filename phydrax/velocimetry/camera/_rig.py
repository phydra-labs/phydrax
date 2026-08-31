#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._model import CameraModel


class CameraRig(StrictModule):
    """A fixed-capacity ordered collection of cameras with an explicit mask."""

    cameras: tuple[CameraModel, ...]
    camera_valid: Array
    capacity: int = eqx.field(static=True)
    rig_id: str = eqx.field(static=True)

    def __init__(
        self,
        cameras: Sequence[CameraModel],
        *,
        camera_valid: ArrayLike | None = None,
    ):
        cameras_ = tuple(cameras)
        if not cameras_:
            raise ValueError("A camera rig must have positive capacity.")
        if not all(isinstance(camera, CameraModel) for camera in cameras_):
            raise TypeError("Every rig entry must be a CameraModel.")
        capacity = len(cameras_)
        if camera_valid is None:
            valid_host = np.ones((capacity,), dtype=bool)
        else:
            valid_host = np.asarray(camera_valid, dtype=bool)
            if valid_host.shape != (capacity,):
                raise ValueError("camera_valid must have shape (capacity,).")
        if not np.any(valid_host):
            raise ValueError("A camera rig must contain at least one active camera.")
        self.cameras = cameras_
        self.camera_valid = jnp.asarray(valid_host)
        self.capacity = capacity
        self.rig_id = canonical_fingerprint(
            {
                "kind": "fixed-capacity-camera-rig",
                "capacity": capacity,
                "camera_valid": valid_host.tolist(),
                "image_shapes": [camera.intrinsics.image_shape for camera in cameras_],
                "refractive_stacks": [
                    None if camera.refraction is None else camera.refraction.stack_id
                    for camera in cameras_
                ],
            }
        )


__all__ = ["CameraRig"]
