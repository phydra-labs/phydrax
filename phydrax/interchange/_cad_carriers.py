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

from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np

from .._strict import StrictModule
from ..discretization._exact_algebraic_geometry import ExactScalar
from ..geometry._rational import fraction_up
from ..geometry.brep._exact import exact_determinant, gram_deviation_bound
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
    RuledSurface,
    SpherePatch,
    SurfaceIsoparametricCurve,
    TorusPatch,
)
from ..geometry.brep._tspline import TSplineSurfacePatch
from ..typing import checked, ConvertibleToArray
from ._cad import refuse
from ._cad_placement_frame import (
    compose_enclosures,
    inverse_enclosure,
    pose_deviation_bounds,
    source_frame,
)


_FRAME_TOLERANCE = 1.0e-9


@final
class RigidPlacement(StrictModule):
    """Proper placement with a rational polar-factor deviation bound."""

    rotation: np.ndarray
    translation: np.ndarray
    rotation_deviation_bound: float = eqx.field(static=True, default=0.0)
    translation_deviation_bound: float = eqx.field(static=True, default=0.0)
    source_enclosure: np.ndarray | None = None

    @classmethod
    def identity(cls) -> RigidPlacement:
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_matrix(
        cls, matrix: ConvertibleToArray, label: str, /, *, reflection: bool = False
    ) -> RigidPlacement:
        """Validate a ``(3, 4)`` affine matrix as an orthogonal motion.

        Reflections (determinant -1) are admitted only when ``reflection`` is
        requested: carriers map exactly under them, but sub-shape placements
        would silently invert topological orientation.
        """
        values = np.asarray(matrix, dtype=np.float64)
        if values.shape != (3, 4) or not np.all(np.isfinite(values)):
            raise refuse(
                "malformed",
                label,
                f"RigidPlacement.from_matrix.matrix must be finite with shape (3, 4); got shape={values.shape}, values={values!r}",
            )
        rotation = values[:, :3]
        orthogonality_deviation = np.max(np.abs(rotation.T @ rotation - np.eye(3)))
        if orthogonality_deviation > _FRAME_TOLERANCE:
            raise refuse(
                "unsupported-entity",
                label,
                f"RigidPlacement.rotation must be orthogonal within tolerance={_FRAME_TOLERANCE}; got deviation={orthogonality_deviation}",
            )
        determinant = np.linalg.det(rotation)
        if determinant <= 0.0 and not reflection:
            raise refuse(
                "unsupported-entity",
                label,
                f"RigidPlacement.rotation must have positive determinant unless reflection is requested; got determinant={determinant}, reflection={reflection}",
            )
        return cls(rotation.copy(), values[:, 3].copy())

    @classmethod
    def from_certified_matrix(
        cls,
        matrix: ConvertibleToArray,
        label: str,
        /,
        *,
        maximum_deviation: float = 0.0,
        model_extent: float = 1.0,
    ) -> RigidPlacement:
        """Certify a proper source frame within its physical deviation request."""
        values = _certification_inputs(matrix, label, maximum_deviation, model_extent)
        rotation = values[:, :3]
        bound = gram_deviation_bound(rotation)
        enclosure = None
        if bound > 0.0 and maximum_deviation > 0.0:
            _, enclosure = source_frame(
                rotation[:, 0],
                rotation[:, 2],
                values[:, 3],
                label,
                second_axis=rotation[:, 1],
            )
            if np.any(values < enclosure[0]) or np.any(values > enclosure[1]):
                raise refuse(
                    "unsupported-entity",
                    label,
                    "RigidPlacement.matrix must lie in its source-rotation enclosure; got scale or shear.",
                )
            source_bound, _ = pose_deviation_bounds(values, enclosure)
            bound = max(bound, source_bound)
        if (
            exact_determinant(rotation) <= 0
            or (bound > 0.0 and maximum_deviation == 0.0)
            or fraction_up(Fraction(bound) * Fraction(model_extent)) > maximum_deviation
        ):
            raise refuse(
                "unsupported-entity",
                label,
                "RigidPlacement.rotation must be proper within its declared point-deviation limit; got unsupported placement.",
            )
        return cls(rotation.copy(), values[:, 3].copy(), bound, 0.0, enclosure)

    @classmethod
    def from_source_directions(
        cls,
        first_axis: np.ndarray,
        third_axis: np.ndarray,
        translation: np.ndarray,
        label: str,
        /,
        *,
        second_axis: np.ndarray | None = None,
    ) -> RigidPlacement:
        """Certify the proper rotation intended by source direction ratios."""
        matrix, enclosure = source_frame(
            first_axis,
            third_axis,
            translation,
            label,
            second_axis=second_axis,
        )
        if exact_determinant(matrix[:, :3]) <= 0:
            raise refuse(
                "unsupported-entity",
                label,
                "RigidPlacement.rotation must be certified proper; got an unresolved represented frame.",
            )
        rotation_bound, translation_bound = pose_deviation_bounds(matrix, enclosure)
        return cls(
            matrix[:, :3],
            matrix[:, 3],
            max(rotation_bound, gram_deviation_bound(matrix[:, :3])),
            translation_bound,
            enclosure,
        )

    @property
    def proper(self) -> bool:
        return bool(np.linalg.det(self.rotation) > 0.0)

    @property
    def is_identity(self) -> bool:
        if self.source_enclosure is not None and not np.array_equal(
            self.source_enclosure, np.stack((self.matrix(), self.matrix()))
        ):
            return False
        return bool(
            np.array_equal(self.rotation, np.eye(3))
            and np.array_equal(self.translation, np.zeros(3))
        )

    def matrix(self) -> np.ndarray:
        return np.concatenate((self.rotation, self.translation[:, None]), axis=1)

    @checked
    def compose(self, inner: RigidPlacement, /) -> RigidPlacement:
        """Compose authored poses and retain their source-intent enclosures."""
        rotation = self.rotation @ inner.rotation
        translation = self.rotation @ inner.translation + self.translation
        if self.source_enclosure is None and inner.source_enclosure is None:
            return RigidPlacement(rotation, translation)
        outer_box = (
            np.stack((self.matrix(), self.matrix()))
            if self.source_enclosure is None
            else self.source_enclosure
        )
        inner_box = (
            np.stack((inner.matrix(), inner.matrix()))
            if inner.source_enclosure is None
            else inner.source_enclosure
        )
        enclosure = compose_enclosures(outer_box, inner_box)
        matrix = np.concatenate((rotation, translation[:, None]), axis=1)
        rotation_bound, translation_bound = pose_deviation_bounds(matrix, enclosure)
        return RigidPlacement(
            rotation,
            translation,
            max(rotation_bound, gram_deviation_bound(rotation)),
            translation_bound,
            enclosure,
        )

    def inverse(self) -> RigidPlacement:
        """Invert a proper pose and retain certified source-direction intent."""
        rotation, translation = self.rotation.T, -(self.rotation.T @ self.translation)
        if self.source_enclosure is None:
            return RigidPlacement(rotation, translation)
        enclosure = inverse_enclosure(self.source_enclosure)
        matrix = np.concatenate((rotation, translation[:, None]), axis=1)
        rotation_bound, translation_bound = pose_deviation_bounds(matrix, enclosure)
        return RigidPlacement(
            rotation,
            translation,
            max(rotation_bound, gram_deviation_bound(rotation)),
            translation_bound,
            enclosure,
        )

    def power(self, exponent: int, /) -> RigidPlacement:
        base = self if exponent >= 0 else self.inverse()
        result = RigidPlacement.identity()
        remaining = abs(exponent)
        while remaining:
            if remaining & 1:
                result = base.compose(result)
            remaining >>= 1
            if remaining:
                base = base.compose(base)
        return result

    def point(self, value: ConvertibleToArray, /) -> np.ndarray:
        return self.rotation @ np.asarray(value, dtype=np.float64) + self.translation

    def vector(self, value: ConvertibleToArray, /) -> np.ndarray:
        return self.rotation @ np.asarray(value, dtype=np.float64)

    def key(self) -> bytes:
        """Encode the binary rigid-location matrix with signed zeros normalized."""
        values = self.matrix()
        # Signed zero is the same location, unlike any nonzero displacement.
        values[values == 0.0] = 0.0
        return values.tobytes()


def _certification_inputs(
    matrix: ConvertibleToArray,
    label: str,
    maximum_deviation: float,
    model_extent: float,
    /,
) -> np.ndarray:
    """Validate finite affine coefficients and nonnegative physical declarations."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (3, 4) or not np.all(np.isfinite(values)):
        raise refuse(
            "malformed",
            label,
            f"RigidPlacement.matrix must be finite with shape (3, 4); got shape={values.shape}",
        )
    if not np.isfinite(maximum_deviation) or maximum_deviation < 0.0:
        raise ValueError(
            "RigidPlacement.maximum_deviation must be finite and nonnegative; got an invalid limit."
        )
    if not np.isfinite(model_extent) or model_extent < 0.0:
        raise ValueError(
            "RigidPlacement.model_extent must be finite and nonnegative; got an invalid extent."
        )
    return values


def _host(value: ConvertibleToArray, /) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _exact_placement_coefficients(
    placement: RigidPlacement, /
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]]:
    return (
        tuple(
            tuple(Fraction.from_float(float(value)) for value in row)
            for row in placement.rotation
        ),
        tuple(Fraction.from_float(float(value)) for value in placement.translation),
    )


def _place_exact_vector(
    values: tuple[ExactScalar, ...],
    rotation: tuple[tuple[Fraction, ...], ...],
    translation: tuple[Fraction, ...] | None = None,
    /,
) -> tuple[ExactScalar, ...]:
    return tuple(
        sum(
            (rotation[row][column] * values[column] for column in range(3)),
            Fraction(0) if translation is None else translation[row],
        )
        for row in range(3)
    )


@checked
def place_curve(
    curve: AbstractCurve, placement: RigidPlacement, label: str, /
) -> AbstractCurve:
    """Rigid image of a 3D curve carrier (same parameterization)."""
    if placement.is_identity:
        return curve
    match curve:
        case LineCurve():
            rotation, translation = _exact_placement_coefficients(placement)
            origin_exact = _place_exact_vector(curve.origin_exact, rotation, translation)
            direction_exact = _place_exact_vector(curve.direction_exact, rotation)
            return LineCurve(
                tuple(float(value) for value in origin_exact),
                tuple(float(value) for value in direction_exact),
                origin_exact=origin_exact,
                direction_exact=direction_exact,
            )
        case CircleCurve():
            rotation, translation = _exact_placement_coefficients(placement)
            center_exact = _place_exact_vector(curve.center_exact, rotation, translation)
            first_axis_exact = _place_exact_vector(curve.first_axis_exact, rotation)
            second_axis_exact = _place_exact_vector(curve.second_axis_exact, rotation)
            return CircleCurve(
                tuple(float(value) for value in center_exact),
                tuple(float(value) for value in first_axis_exact),
                tuple(float(value) for value in second_axis_exact),
                curve.radius,
                center_exact=center_exact,
                first_axis_exact=first_axis_exact,
                second_axis_exact=second_axis_exact,
                radius_exact=curve.radius_exact,
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
                curve.distance_exact if placement.proper else -curve.distance_exact,
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
                f"place_curve.curve must have a supported rigid placement rule; got {type(curve).__name__}",
            )


@checked
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
            rotation_exact, translation_exact = _exact_placement_coefficients(placement)
            origin_exact = _place_exact_vector(
                patch.origin_exact, rotation_exact, translation_exact
            )
            first_axis_exact = _place_exact_vector(patch.first_axis_exact, rotation_exact)
            second_axis_exact = _place_exact_vector(
                patch.second_axis_exact, rotation_exact
            )
            axis_exact = _place_exact_vector(patch.axis_exact, rotation_exact)
            return CylinderPatch(
                tuple(float(value) for value in origin_exact),
                tuple(float(value) for value in first_axis_exact),
                tuple(float(value) for value in second_axis_exact),
                tuple(float(value) for value in axis_exact),
                patch.radius,
                origin_exact=origin_exact,
                first_axis_exact=first_axis_exact,
                second_axis_exact=second_axis_exact,
                axis_exact=axis_exact,
                radius_exact=patch.radius_exact,
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
            rotation_exact, translation_exact = _exact_placement_coefficients(placement)
            center_exact = _place_exact_vector(
                patch.center_exact, rotation_exact, translation_exact
            )
            first_axis_exact = _place_exact_vector(patch.first_axis_exact, rotation_exact)
            second_axis_exact = _place_exact_vector(
                patch.second_axis_exact, rotation_exact
            )
            axis_exact = _place_exact_vector(patch.axis_exact, rotation_exact)
            return TorusPatch(
                tuple(float(value) for value in center_exact),
                tuple(float(value) for value in first_axis_exact),
                tuple(float(value) for value in second_axis_exact),
                tuple(float(value) for value in axis_exact),
                patch.major_radius,
                patch.minor_radius,
                center_exact=center_exact,
                first_axis_exact=first_axis_exact,
                second_axis_exact=second_axis_exact,
                axis_exact=axis_exact,
                major_radius_exact=patch.major_radius_exact,
                minor_radius_exact=patch.minor_radius_exact,
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
                # Placement moves poles; it never changes their source identity.
                control_point_ids=patch.control_point_ids,
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
        case RuledSurface():
            return RuledSurface(
                place_curve(patch.first, placement, label),
                place_curve(patch.second, placement, label),
            )
        case _:
            raise refuse(
                "unsupported-entity",
                label,
                f"place_surface.patch must have a supported rigid placement rule; got {type(patch).__name__}",
            )


@checked
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
        case TSplineSurfacePatch():
            return "tspline"
        case ExtrusionSurface():
            return "extrusion"
        case RevolutionSurface():
            return "revolution"
        case OffsetSurface():
            return "offset"
        case _:
            raise TypeError(
                f"surface_tag.patch must have a declared surface-family tag; got {type(patch).__name__}"
            )


def orthonormal_frame(
    first: ConvertibleToArray, second: ConvertibleToArray, label: str, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate an orthonormal pair (either handedness); return it with its normal."""
    first_, second_ = _host(first), _host(second)
    first_norm = np.linalg.norm(first_)
    if abs(first_norm - 1.0) > _FRAME_TOLERANCE:
        raise refuse(
            "inexact-export",
            label,
            f"orthonormal_frame.first must have unit length within tolerance={_FRAME_TOLERANCE}; got norm={first_norm}",
        )
    second_norm = np.linalg.norm(second_)
    if abs(second_norm - 1.0) > _FRAME_TOLERANCE:
        raise refuse(
            "inexact-export",
            label,
            f"orthonormal_frame.second must have unit length within tolerance={_FRAME_TOLERANCE}; got norm={second_norm}",
        )
    inner_product = first_ @ second_
    if abs(inner_product) > _FRAME_TOLERANCE:
        raise refuse(
            "inexact-export",
            label,
            f"orthonormal_frame.first must be orthogonal to second within tolerance={_FRAME_TOLERANCE}; got inner_product={inner_product}",
        )
    normal = np.cross(first_, second_) if first_.size == 3 else np.zeros(0)
    return first_, second_, normal


__all__ = [
    "RigidPlacement",
    "orthonormal_frame",
    "place_curve",
    "place_surface",
    "surface_tag",
]
