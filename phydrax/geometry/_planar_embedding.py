#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Engine-neutral affine embedding of planar coordinates in one world frame."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._fingerprint import canonical_fingerprint


_FRAME_TOLERANCE = 128.0 * np.finfo(np.float64).eps


def _vector3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite three-vector.")
    return (float(array[0]), float(array[1]), float(array[2]))


@dataclass(frozen=True, slots=True, init=False)
class PlanarEmbedding:
    """Exact affine embedding of planar coordinates in one world frame."""

    origin: tuple[float, float, float]
    x_axis: tuple[float, float, float]
    y_axis: tuple[float, float, float]
    normal: tuple[float, float, float]
    embedding_id: str

    def __init__(
        self,
        origin: Sequence[float],
        x_axis: Sequence[float],
        y_axis: Sequence[float],
        normal: Sequence[float],
        /,
    ) -> None:
        origin_ = _vector3(origin, "origin")
        x_axis_ = _vector3(x_axis, "x_axis")
        y_axis_ = _vector3(y_axis, "y_axis")
        normal_ = _vector3(normal, "normal")
        frame = np.stack((x_axis_, y_axis_, normal_))
        if not np.allclose(
            frame @ frame.T,
            np.eye(3),
            rtol=0.0,
            atol=_FRAME_TOLERANCE,
        ):
            raise ValueError("Planar embedding axes must be orthonormal unit vectors.")
        if not np.allclose(
            np.cross(frame[0], frame[1]),
            frame[2],
            rtol=0.0,
            atol=_FRAME_TOLERANCE,
        ):
            raise ValueError("Planar embedding axes must form a right-handed frame.")
        embedding_id = canonical_fingerprint(
            {
                "kind": "planar-embedding",
                "origin": origin_,
                "x_axis": x_axis_,
                "y_axis": y_axis_,
                "normal": normal_,
            }
        )
        object.__setattr__(self, "origin", origin_)
        object.__setattr__(self, "x_axis", x_axis_)
        object.__setattr__(self, "y_axis", y_axis_)
        object.__setattr__(self, "normal", normal_)
        object.__setattr__(self, "embedding_id", embedding_id)

    def to_world(self, coordinates: Any, /) -> np.ndarray:
        """Map coordinates with trailing dimension two into the world frame."""

        planar = np.asarray(coordinates, dtype=np.float64)
        if planar.ndim == 0 or planar.shape[-1] != 2:
            raise ValueError("Planar coordinates must have trailing dimension two.")
        if not np.all(np.isfinite(planar)):
            raise ValueError("Planar coordinates must be finite.")
        basis = np.stack((self.x_axis, self.y_axis), axis=0)
        return np.asarray(self.origin) + planar @ basis

    def to_planar(self, points: Any, /) -> np.ndarray:
        """Return the exact frame coordinates of world points on the plane."""

        world = np.asarray(points, dtype=np.float64)
        if world.ndim == 0 or world.shape[-1] != 3:
            raise ValueError("World points must have trailing dimension three.")
        if not np.all(np.isfinite(world)):
            raise ValueError("World points must be finite.")
        basis = np.stack((self.x_axis, self.y_axis), axis=1)
        return (world - np.asarray(self.origin)) @ basis

    def plane_residual(self, points: Any, /) -> np.ndarray:
        """Return the signed world-frame residual along the plane normal."""

        world = np.asarray(points, dtype=np.float64)
        if world.ndim == 0 or world.shape[-1] != 3:
            raise ValueError("World points must have trailing dimension three.")
        if not np.all(np.isfinite(world)):
            raise ValueError("World points must be finite.")
        return (world - np.asarray(self.origin)) @ np.asarray(self.normal)


__all__ = ["PlanarEmbedding"]
