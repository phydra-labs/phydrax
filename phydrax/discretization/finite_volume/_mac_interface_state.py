#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array

from ..._strict import StrictModule


class MACFreeSurfaceGeometryState(StrictModule):
    signed_distance: Array
    liquid_mask: Array
    valid_band: Array
    cell_fraction: Array
    face_fraction: tuple[Array, ...]
    ghost_fraction: tuple[Array, ...]
    interface_faces: tuple[Array, ...]
    normal: Array
    curvature: Array
    minimum_ghost_fraction: Array
    clamped_face_count: Array
    finite: Array
    successful: Array
    geometry_id: str = eqx.field(static=True)


__all__ = ["MACFreeSurfaceGeometryState"]
