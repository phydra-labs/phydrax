#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Rigid placement and family vocabulary of native CAD carriers.

External formats locate shared carriers and sub-shapes with rigid transforms
(OCCT locations, IGES transformation matrices, STEP placements). These helpers
validate such transforms as proper rigid motions and move every native carrier
family exactly, preserving its parameterization, so that same-parameter
p-curves stay valid. Non-rigid transforms are refused, never approximated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.brep._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConePatch,
    CylinderPatch,
    EllipseCurve,
    ExtrusionSurface,
    HyperbolaCurve,
    LineCurve,
    OffsetCurve,
    OffsetSurface,
    ParabolaCurve,
    PlanePatch,
    RevolutionSurface,
    SpherePatch,
    SurfaceIsoparametricCurve,
    TorusPatch,
)
from ._cad import refuse


_FRAME_TOLERANCE = 1.0e-9


@dataclass(frozen=True, slots=True)
class RigidPlacement:
    """Proper rigid motion ``x -> rotation @ x + translation``."""

    rotation: np.ndarray
    translation: np.ndarray

    @classmethod
    def identity(cls) -> RigidPlacement:
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_matrix(
        cls, matrix: np.ndarray, label: str, /, *, reflection: bool = False
    ) -> RigidPlacement:
        """Validate a ``(3, 4)`` affine matrix as an orthogonal motion.

        Reflections (determinant -1) are admitted only when ``reflection`` is
        requested: carriers map exactly under them, but sub-shape placements
        would silently invert topological orientation.
        """
        values = np.asarray(matrix, dtype=np.float64)
        if values.shape != (3, 4) or not np.all(np.isfinite(values)):
            raise refuse("malformed", label, "A placement must be a finite 3x4 matrix.")
        rotation = values[:, :3]
        if np.max(np.abs(rotation.T @ rotation - np.eye(3))) > _FRAME_TOLERANCE:
            raise refuse(
                "unsupported-entity",
                label,
                "Scaled or sheared placements have no rigid native representation.",
            )
        if np.linalg.det(rotation) <= 0.0 and not reflection:
            raise refuse(
                "unsupported-entity",
                label,
                "Mirroring placements of sub-shapes are not admitted.",
            )
        return cls(rotation.copy(), values[:, 3].copy())

    @property
    def proper(self) -> bool:
        return bool(np.linalg.det(self.rotation) > 0.0)

    @property
    def is_identity(self) -> bool:
        return bool(
            np.array_equal(self.rotation, np.eye(3))
            and np.array_equal(self.translation, np.zeros(3))
        )

    def matrix(self) -> np.ndarray:
        return np.concatenate((self.rotation, self.translation[:, None]), axis=1)

    def compose(self, inner: RigidPlacement, /) -> RigidPlacement:
        """``self`` after ``inner``: ``x -> self(inner(x))``."""
        return RigidPlacement(
            self.rotation @ inner.rotation,
            self.rotation @ inner.translation + self.translation,
        )

    def inverse(self) -> RigidPlacement:
        return RigidPlacement(self.rotation.T, -(self.rotation.T @ self.translation))

    def power(self, exponent: int, /) -> RigidPlacement:
        base = self if exponent >= 0 else self.inverse()
        result = RigidPlacement.identity()
        for _ in range(abs(exponent)):
            result = base.compose(result)
        return result

    def point(self, value: object, /) -> np.ndarray:
        return self.rotation @ np.asarray(value, dtype=np.float64) + self.translation

    def vector(self, value: object, /) -> np.ndarray:
        return self.rotation @ np.asarray(value, dtype=np.float64)

    def key(self) -> bytes:
        """Exact binary location identity; nearby distinct locations never merge."""
        values = self.matrix()
        # Signed zero is the same location, unlike any nonzero displacement.
        values[values == 0.0] = 0.0
        return values.tobytes()


def _host(value: object, /) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def place_curve(
    curve: AbstractCurve, placement: RigidPlacement, label: str, /
) -> AbstractCurve:
    """Rigid image of a 3D curve carrier (same parameterization)."""
    if placement.is_identity:
        return curve
    match curve:
        case LineCurve():
            return LineCurve(
                placement.point(curve.origin), placement.vector(curve.direction)
            )
        case CircleCurve():
            return CircleCurve(
                placement.point(curve.center),
                placement.vector(curve.first_axis),
                placement.vector(curve.second_axis),
                curve.radius,
            )
        case EllipseCurve():
            return EllipseCurve(
                placement.point(curve.center),
                placement.vector(curve.first_axis),
                placement.vector(curve.second_axis),
                curve.first_radius,
                curve.second_radius,
            )
        case ParabolaCurve():
            return ParabolaCurve(
                placement.point(curve.vertex),
                placement.vector(curve.first_axis),
                placement.vector(curve.second_axis),
                curve.focal_length,
            )
        case HyperbolaCurve():
            return HyperbolaCurve(
                placement.point(curve.center),
                placement.vector(curve.first_axis),
                placement.vector(curve.second_axis),
                curve.first_radius,
                curve.second_radius,
            )
        case OffsetCurve():
            direction = (
                None if curve.direction is None else placement.vector(curve.direction)
            )
            return OffsetCurve(
                place_curve(curve.base, placement, label),
                curve.distance if placement.proper else -curve.distance,
                direction,
            )
        case BSplineCurve():
            points = _host(curve.control_points) @ placement.rotation.T
            return BSplineCurve(
                points + placement.translation, curve.weights, curve.knots, curve.degree
            )
        case SurfaceIsoparametricCurve():
            return SurfaceIsoparametricCurve(
                place_surface(curve.surface, placement, label),
                curve.fixed_axis,
                curve.fixed_value,
                parameter_range=curve.parameter_range,
            )
        case _:
            raise refuse(
                "unsupported-entity",
                label,
                f"No rigid placement rule for curve family {type(curve).__name__}.",
            )


def place_surface(
    patch: AbstractSurfacePatch, placement: RigidPlacement, label: str, /
) -> AbstractSurfacePatch:
    """Rigid image of a surface carrier (same parameterization)."""
    if placement.is_identity:
        return patch
    point, vector = placement.point, placement.vector
    match patch:
        case PlanePatch():
            return PlanePatch(
                point(patch.origin), vector(patch.first_axis), vector(patch.second_axis)
            )
        case CylinderPatch():
            return CylinderPatch(
                point(patch.origin),
                vector(patch.first_axis),
                vector(patch.second_axis),
                vector(patch.axis),
                patch.radius,
            )
        case ConePatch():
            return ConePatch(
                point(patch.origin),
                vector(patch.first_axis),
                vector(patch.second_axis),
                vector(patch.axis),
                patch.reference_radius,
                patch.semi_angle,
            )
        case SpherePatch():
            return SpherePatch(
                point(patch.center),
                vector(patch.first_axis),
                vector(patch.second_axis),
                vector(patch.axis),
                patch.radius,
            )
        case TorusPatch():
            return TorusPatch(
                point(patch.center),
                vector(patch.first_axis),
                vector(patch.second_axis),
                vector(patch.axis),
                patch.major_radius,
                patch.minor_radius,
            )
        case BSplineSurfacePatch():
            points = _host(patch.control_points) @ placement.rotation.T
            return BSplineSurfacePatch(
                points + placement.translation,
                patch.weights,
                patch.u_knots,
                patch.v_knots,
                patch.u_degree,
                patch.v_degree,
            )
        case ExtrusionSurface():
            return ExtrusionSurface(
                place_curve(patch.curve, placement, label), vector(patch.direction)
            )
        case RevolutionSurface():
            # A reflection reverses the rotation sense: M R(a, u) = R(-M a, u) M.
            sense = 1.0 if placement.proper else -1.0
            return RevolutionSurface(
                place_curve(patch.curve, placement, label),
                point(patch.axis_origin),
                sense * vector(patch.axis_direction),
            )
        case OffsetSurface():
            # Reflection reverses the oriented base normal; reverse distance too.
            sense = 1.0 if placement.proper else -1.0
            return OffsetSurface(
                place_surface(patch.base, placement, label),
                sense * float(patch.distance),
            )
        case _:
            raise refuse(
                "unsupported-entity",
                label,
                f"No rigid placement rule for surface family {type(patch).__name__}.",
            )


def surface_tag(patch: AbstractSurfacePatch, /) -> str:
    """Native physical tag vocabulary of a surface family."""
    match patch:
        case PlanePatch():
            return "plane"
        case CylinderPatch():
            return "cylinder"
        case ConePatch():
            return "cone"
        case SpherePatch():
            return "sphere"
        case TorusPatch():
            return "torus"
        case BSplineSurfacePatch():
            return "bspline"
        case ExtrusionSurface():
            return "extrusion"
        case RevolutionSurface():
            return "revolution"
        case OffsetSurface():
            return "offset"
        case _:
            raise TypeError(f"Surface family {type(patch).__name__} has no tag.")


def orthonormal_frame(
    first: object, second: object, label: str, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate an orthonormal pair (either handedness); return it with its normal."""
    first_, second_ = _host(first), _host(second)
    if (
        abs(np.linalg.norm(first_) - 1.0) > _FRAME_TOLERANCE
        or abs(np.linalg.norm(second_) - 1.0) > _FRAME_TOLERANCE
        or abs(first_ @ second_) > _FRAME_TOLERANCE
    ):
        raise refuse("inexact-export", label, "The carrier frame is not orthonormal.")
    normal = np.cross(first_, second_) if first_.size == 3 else np.zeros(0)
    return first_, second_, normal


__all__ = [
    "RigidPlacement",
    "orthonormal_frame",
    "place_curve",
    "place_surface",
    "surface_tag",
]
