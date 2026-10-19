#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Engine-neutral affine embedding of planar coordinates in one world frame."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint


_FRAME_TOLERANCE = 128.0 * np.finfo(np.float64).eps


def _vector3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        field = f"PlanarEmbedding.{name}"
        raise ValueError(
            f"{field} must be a finite three-vector; got shape={array.shape}"
        )
    return (float(array[0]), float(array[1]), float(array[2]))


@dataclass(frozen=True, slots=True, init=False)
class PlanarEmbedding:
    """Embed planar coordinates in a finite right-handed orthonormal frame."""

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
            np.eye(3, dtype=np.float64),
            rtol=0.0,
            atol=_FRAME_TOLERANCE,
        ):
            raise ValueError(
                "PlanarEmbedding.axes must be orthonormal unit vectors; got nonorthonormal frame"
            )
        if not np.allclose(
            np.cross(frame[0], frame[1]),
            frame[2],
            rtol=0.0,
            atol=_FRAME_TOLERANCE,
        ):
            raise ValueError(
                "PlanarEmbedding.axes must form a right-handed frame; got reversed normal"
            )
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

    def to_world(self, coordinates: ArrayLike, /) -> np.ndarray:
        """Map coordinates with trailing dimension two into the world frame."""

        planar = np.asarray(coordinates, dtype=np.float64)
        if planar.ndim == 0 or planar.shape[-1] != 2:
            raise ValueError(
                f"PlanarEmbedding.coordinates must have trailing dimension two; got shape={planar.shape}"
            )
        if not np.all(np.isfinite(planar)):
            raise ValueError(
                "PlanarEmbedding.coordinates must be finite; got nonfinite values"
            )
        basis = np.stack((self.x_axis, self.y_axis), axis=0)
        return np.asarray(self.origin, dtype=np.float64) + planar @ basis

    def to_planar(self, points: ArrayLike, /) -> np.ndarray:
        """Return world points projected onto the two planar frame axes."""

        world = np.asarray(points, dtype=np.float64)
        if world.ndim == 0 or world.shape[-1] != 3:
            raise ValueError(
                f"PlanarEmbedding.points must have trailing dimension three; got shape={world.shape}"
            )
        if not np.all(np.isfinite(world)):
            raise ValueError(
                "PlanarEmbedding.points must be finite; got nonfinite values"
            )
        basis = np.stack((self.x_axis, self.y_axis), axis=1)
        return (world - np.asarray(self.origin, dtype=np.float64)) @ basis

    def plane_residual(self, points: ArrayLike, /) -> np.ndarray:
        """Return the signed world-frame residual along the plane normal."""

        world = np.asarray(points, dtype=np.float64)
        if world.ndim == 0 or world.shape[-1] != 3:
            raise ValueError(
                f"PlanarEmbedding.points must have trailing dimension three; got shape={world.shape}"
            )
        if not np.all(np.isfinite(world)):
            raise ValueError(
                "PlanarEmbedding.points must be finite; got nonfinite values"
            )
        return (world - np.asarray(self.origin, dtype=np.float64)) @ np.asarray(
            self.normal, dtype=np.float64
        )


__all__ = ["PlanarEmbedding"]
