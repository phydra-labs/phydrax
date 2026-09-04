#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._state_geometry import AbstractStateGeometry


QuaternionConvention: TypeAlias = Literal["body", "spatial"]


def _convention(value: QuaternionConvention, /) -> QuaternionConvention:
    if value == "body":
        return "body"
    if value == "spatial":
        return "spatial"
    raise ValueError("convention must be 'body' or 'spatial'.")


def _tolerance(value: float, /) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    return resolved


def _real_vector(value: ArrayLike, size: int, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},); got {array.shape}.")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must have a real floating dtype.")
    return array


def _quaternion(value: ArrayLike, name: str, /) -> Array:
    return _real_vector(value, 4, name)


def _normalize_quaternion(value: ArrayLike, name: str, /) -> Array:
    quaternion = _quaternion(value, name)
    norm = jnp.linalg.norm(quaternion)
    quaternion = eqx.error_if(
        quaternion,
        (~jnp.isfinite(norm)) | (norm <= jnp.finfo(quaternion.dtype).tiny),
        f"{name} must have a finite nonzero norm.",
    )
    return quaternion / norm


def _conjugate(quaternion: Array, /) -> Array:
    return quaternion.at[1:].multiply(-1.0)


def _multiply(left: Array, right: Array, /) -> Array:
    left_scalar = left[0]
    left_vector = left[1:]
    right_scalar = right[0]
    right_vector = right[1:]
    return jnp.concatenate(
        (
            (left_scalar * right_scalar - jnp.dot(left_vector, right_vector))[None],
            left_scalar * right_vector
            + right_scalar * left_vector
            + jnp.cross(left_vector, right_vector),
        )
    )


def _rotate(quaternion: Array, vector: Array, /) -> Array:
    imaginary = quaternion[1:]
    doubled_cross = 2.0 * jnp.cross(imaginary, vector)
    return vector + quaternion[0] * doubled_cross + jnp.cross(imaginary, doubled_cross)


def _rotation_vector_quaternion(rotation_vector: Array, /) -> Array:
    angle_squared = jnp.dot(rotation_vector, rotation_vector)
    near_zero = angle_squared < 1.0e-12
    safe_angle = jnp.sqrt(jnp.where(near_zero, 1.0, angle_squared))
    angle_fourth = angle_squared * angle_squared
    scalar = jnp.where(
        near_zero,
        1.0 - angle_squared / 8.0 + angle_fourth / 384.0,
        jnp.cos(0.5 * safe_angle),
    )
    vector_scale = jnp.where(
        near_zero,
        0.5 - angle_squared / 48.0 + angle_fourth / 3840.0,
        jnp.sin(0.5 * safe_angle) / safe_angle,
    )
    return jnp.concatenate((scalar[None], vector_scale * rotation_vector))


def _rotation_vector(quaternion: Array, /) -> Array:
    vector = quaternion[1:]
    vector_norm_squared = jnp.dot(vector, vector)
    near_zero = vector_norm_squared < 1.0e-14
    safe_norm = jnp.sqrt(jnp.where(near_zero, 1.0, vector_norm_squared))
    safe_angle = 2.0 * jnp.arctan2(safe_norm, quaternion[0])
    scale = jnp.where(
        near_zero,
        2.0
        + vector_norm_squared / 3.0
        + 3.0 * vector_norm_squared * vector_norm_squared / 20.0,
        safe_angle / safe_norm,
    )
    return scale * vector


def _align_to_anchor(anchor: Array, target: Array, tolerance: float, /) -> Array:
    overlap = jnp.dot(anchor, target)
    target = eqx.error_if(
        target,
        jnp.abs(overlap) <= tolerance,
        "Quaternion chart reaches the rotation-by-pi cut locus.",
    )
    sign = jnp.where(overlap < 0.0, -1.0, 1.0).astype(target.dtype)
    return sign * target


def _validate_chart_step(rotation_vector: Array, tolerance: float, /) -> Array:
    return eqx.error_if(
        rotation_vector,
        jnp.linalg.norm(rotation_vector) >= jnp.pi - tolerance,
        "Quaternion retraction step reaches the rotation-by-pi cut locus.",
    )


def _hat(vector: Array, /) -> Array:
    x, y, z = vector
    zero = jnp.zeros((), dtype=vector.dtype)
    return jnp.stack(
        (
            jnp.stack((zero, -z, y)),
            jnp.stack((z, zero, -x)),
            jnp.stack((-y, x, zero)),
        )
    )


def _left_jacobian(rotation_vector: Array, /) -> Array:
    angle_squared = jnp.dot(rotation_vector, rotation_vector)
    near_zero = angle_squared < 1.0e-10
    safe_squared = jnp.where(near_zero, 1.0, angle_squared)
    safe_angle = jnp.sqrt(safe_squared)
    first = jnp.where(
        near_zero,
        0.5 - angle_squared / 24.0 + angle_squared * angle_squared / 720.0,
        (1.0 - jnp.cos(safe_angle)) / safe_squared,
    )
    second = jnp.where(
        near_zero,
        1.0 / 6.0 - angle_squared / 120.0 + angle_squared * angle_squared / 5040.0,
        (safe_angle - jnp.sin(safe_angle)) / (safe_squared * safe_angle),
    )
    skew = _hat(rotation_vector)
    return jnp.eye(3, dtype=rotation_vector.dtype) + first * skew + second * (skew @ skew)


def _left_jacobian_inverse(rotation_vector: Array, /) -> Array:
    angle_squared = jnp.dot(rotation_vector, rotation_vector)
    near_zero = angle_squared < 1.0e-8
    safe_squared = jnp.where(near_zero, 1.0, angle_squared)
    safe_angle = jnp.sqrt(safe_squared)
    half_angle = 0.5 * safe_angle
    coefficient = jnp.where(
        near_zero,
        1.0 / 12.0 + angle_squared / 720.0 + angle_squared * angle_squared / 30240.0,
        (1.0 - half_angle * jnp.cos(half_angle) / jnp.sin(half_angle)) / safe_squared,
    )
    skew = _hat(rotation_vector)
    return (
        jnp.eye(3, dtype=rotation_vector.dtype) - 0.5 * skew + coefficient * (skew @ skew)
    )


def _quaternion_velocity(
    point: Array, point_tangent: Array, convention: QuaternionConvention, /
) -> Array:
    if convention == "body":
        product = _multiply(_conjugate(point), point_tangent)
    else:
        product = _multiply(point_tangent, _conjugate(point))
    return 2.0 * product[1:]


def _quaternion_tangent(
    point: Array, angular_velocity: Array, convention: QuaternionConvention, /
) -> Array:
    pure = jnp.concatenate((jnp.zeros((1,), dtype=point.dtype), angular_velocity))
    if convention == "body":
        return 0.5 * _multiply(point, pure)
    return 0.5 * _multiply(pure, point)


class ScalarFirstQuaternionStateGeometry(AbstractStateGeometry):
    """Unit-quaternion rotation geometry with 4/3/3/3 storage and coordinates."""

    convention: QuaternionConvention = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_inverse: bool = eqx.field(static=True)
    supports_exact_differential: bool = eqx.field(static=True)
    supports_transport: bool = eqx.field(static=True)
    supports_isometric_transport: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        convention: QuaternionConvention = "body",
        tolerance: float = 1.0e-9,
    ):
        convention_ = _convention(convention)
        tolerance_ = _tolerance(tolerance)
        self.convention = convention_
        self.tolerance = tolerance_
        self.geometry_id = f"state-geometry:unit-quaternion:scalar-first:{convention_}"
        self.retraction_method = f"quaternion-exponential:{convention_}"
        self.trivial = False
        self.supports_exact_inverse = True
        self.supports_exact_differential = True
        self.supports_transport = True
        self.supports_isometric_transport = True
        self.supports_commutator_free = True

    def contains(self, state: ArrayLike, /) -> Array:
        point = _quaternion(state, "Quaternion state")
        norm = jnp.linalg.norm(point)
        tolerance = jnp.maximum(
            jnp.asarray(self.tolerance, dtype=point.dtype),
            jnp.asarray(500.0 * jnp.finfo(point.dtype).eps, dtype=point.dtype),
        )
        return jnp.all(jnp.isfinite(point)) & (jnp.abs(norm - 1.0) <= tolerance)

    def cut_locus_margin(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        anchor = _normalize_quaternion(state, "Quaternion chart anchor")
        target = _normalize_quaternion(point, "Quaternion chart point")
        return jnp.abs(jnp.dot(anchor, target))

    def project_tangent(self, state: ArrayLike, vector: ArrayLike, /) -> Array:
        point = _normalize_quaternion(state, "Quaternion state")
        ambient = _quaternion(vector, "Ambient quaternion tangent")
        return _quaternion_velocity(point, ambient, self.convention)

    def retract(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        point = _normalize_quaternion(state, "Quaternion state")
        local = _validate_chart_step(
            _real_vector(local_tangent, 3, "Quaternion local tangent"), self.tolerance
        )
        increment = _rotation_vector_quaternion(local)
        candidate = (
            _multiply(point, increment)
            if self.convention == "body"
            else _multiply(increment, point)
        )
        return _align_to_anchor(
            point,
            _normalize_quaternion(candidate, "Retracted quaternion"),
            self.tolerance,
        )

    def inverse_retract(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        anchor = _normalize_quaternion(state, "Quaternion chart anchor")
        target = _align_to_anchor(
            anchor,
            _normalize_quaternion(point, "Quaternion chart point"),
            self.tolerance,
        )
        relative = (
            _multiply(_conjugate(anchor), target)
            if self.convention == "body"
            else _multiply(target, _conjugate(anchor))
        )
        return _rotation_vector(_normalize_quaternion(relative, "Relative quaternion"))

    def retraction_jvp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        local_tangent_direction: ArrayLike,
        /,
    ) -> Array:
        anchor = _normalize_quaternion(state, "Quaternion state")
        local = _real_vector(local_tangent, 3, "Quaternion local tangent")
        direction = _real_vector(
            local_tangent_direction, 3, "Quaternion local tangent direction"
        )
        target, point_tangent = jax.jvp(
            lambda value: self.retract(anchor, value),
            (local,),
            (direction,),
        )
        return _quaternion_velocity(target, point_tangent, self.convention)

    def retraction_inverse_jvp(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        anchor = _normalize_quaternion(state, "Quaternion chart anchor")
        target = _normalize_quaternion(point, "Quaternion chart point")
        velocity = _real_vector(tangent, 3, "Quaternion physical tangent")
        point_tangent = _quaternion_tangent(target, velocity, self.convention)
        return jax.jvp(
            lambda value: self.inverse_retract(anchor, value),
            (target,),
            (point_tangent,),
        )[1]

    def retraction_vjp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        anchor = _normalize_quaternion(state, "Quaternion state")
        local = _real_vector(local_tangent, 3, "Quaternion local tangent")
        target_cotangent = _real_vector(cotangent, 3, "Quaternion physical cotangent")
        return jax.linear_transpose(
            lambda direction: self.retraction_jvp(anchor, local, direction),
            jnp.zeros_like(local),
        )(target_cotangent)[0]

    def transport_tangent(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        _normalize_quaternion(state, "Transport source quaternion")
        _normalize_quaternion(point, "Transport target quaternion")
        return _real_vector(tangent, 3, "Quaternion physical tangent")

    def transport_cotangent_pullback(
        self,
        state: ArrayLike,
        point: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        _normalize_quaternion(state, "Transport source quaternion")
        _normalize_quaternion(point, "Transport target quaternion")
        return _real_vector(cotangent, 3, "Quaternion physical cotangent")


class QuaternionPoseStateGeometry(AbstractStateGeometry):
    """Quaternion-position SE(3) geometry with 7/6/6/6 storage and coordinates."""

    convention: QuaternionConvention = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_inverse: bool = eqx.field(static=True)
    supports_exact_differential: bool = eqx.field(static=True)
    supports_transport: bool = eqx.field(static=True)
    supports_isometric_transport: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        convention: QuaternionConvention = "body",
        tolerance: float = 1.0e-9,
    ):
        convention_ = _convention(convention)
        tolerance_ = _tolerance(tolerance)
        self.convention = convention_
        self.tolerance = tolerance_
        self.geometry_id = f"state-geometry:se3:quaternion-position:{convention_}"
        self.retraction_method = f"se3-quaternion-exponential:{convention_}"
        self.trivial = False
        self.supports_exact_inverse = True
        self.supports_exact_differential = True
        self.supports_transport = True
        self.supports_isometric_transport = True
        self.supports_commutator_free = True

    def _point(self, value: ArrayLike, name: str, /) -> tuple[Array, Array]:
        point = _real_vector(value, 7, name)
        return _normalize_quaternion(point[:4], f"{name} quaternion"), point[4:]

    def contains(self, state: ArrayLike, /) -> Array:
        point = _real_vector(state, 7, "Quaternion pose state")
        quaternion = point[:4]
        tolerance = jnp.maximum(
            jnp.asarray(self.tolerance, dtype=point.dtype),
            jnp.asarray(500.0 * jnp.finfo(point.dtype).eps, dtype=point.dtype),
        )
        return jnp.all(jnp.isfinite(point)) & (
            jnp.abs(jnp.linalg.norm(quaternion) - 1.0) <= tolerance
        )

    def cut_locus_margin(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        anchor_quaternion, _ = self._point(state, "Quaternion pose chart anchor")
        target_quaternion, _ = self._point(point, "Quaternion pose chart point")
        return jnp.abs(jnp.dot(anchor_quaternion, target_quaternion))

    def project_tangent(self, state: ArrayLike, vector: ArrayLike, /) -> Array:
        quaternion, position = self._point(state, "Quaternion pose state")
        ambient = _real_vector(vector, 7, "Ambient quaternion-pose tangent")
        angular = _quaternion_velocity(quaternion, ambient[:4], self.convention)
        if self.convention == "body":
            linear = _rotate(_conjugate(quaternion), ambient[4:])
        else:
            linear = ambient[4:] - jnp.cross(angular, position)
        return jnp.concatenate((linear, angular))

    def retract(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        quaternion, position = self._point(state, "Quaternion pose state")
        local = _real_vector(local_tangent, 6, "Quaternion-pose local tangent")
        rotation = _validate_chart_step(local[3:], self.tolerance)
        increment_quaternion = _rotation_vector_quaternion(rotation)
        increment_translation = _left_jacobian(rotation) @ local[:3]
        if self.convention == "body":
            target_quaternion = _multiply(quaternion, increment_quaternion)
            target_position = position + _rotate(quaternion, increment_translation)
        else:
            target_quaternion = _multiply(increment_quaternion, quaternion)
            target_position = increment_translation + _rotate(
                increment_quaternion, position
            )
        target_quaternion = _align_to_anchor(
            quaternion,
            _normalize_quaternion(target_quaternion, "Retracted pose quaternion"),
            self.tolerance,
        )
        return jnp.concatenate((target_quaternion, target_position))

    def inverse_retract(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        anchor_quaternion, anchor_position = self._point(
            state, "Quaternion pose chart anchor"
        )
        target_quaternion, target_position = self._point(
            point, "Quaternion pose chart point"
        )
        target_quaternion = _align_to_anchor(
            anchor_quaternion, target_quaternion, self.tolerance
        )
        if self.convention == "body":
            relative_quaternion = _multiply(
                _conjugate(anchor_quaternion), target_quaternion
            )
            relative_translation = _rotate(
                _conjugate(anchor_quaternion), target_position - anchor_position
            )
        else:
            relative_quaternion = _multiply(
                target_quaternion, _conjugate(anchor_quaternion)
            )
            relative_translation = target_position - _rotate(
                relative_quaternion, anchor_position
            )
        rotation = _rotation_vector(
            _normalize_quaternion(relative_quaternion, "Relative pose quaternion")
        )
        translation = _left_jacobian_inverse(rotation) @ relative_translation
        return jnp.concatenate((translation, rotation))

    def retraction_jvp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        local_tangent_direction: ArrayLike,
        /,
    ) -> Array:
        anchor = _real_vector(state, 7, "Quaternion pose state")
        local = _real_vector(local_tangent, 6, "Quaternion-pose local tangent")
        direction = _real_vector(
            local_tangent_direction, 6, "Quaternion-pose local tangent direction"
        )
        target, point_tangent = jax.jvp(
            lambda value: self.retract(anchor, value),
            (local,),
            (direction,),
        )
        return self.project_tangent(target, point_tangent)

    def retraction_inverse_jvp(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        anchor = _real_vector(state, 7, "Quaternion pose chart anchor")
        target_quaternion, target_position = self._point(
            point, "Quaternion pose chart point"
        )
        velocity = _real_vector(tangent, 6, "Quaternion-pose physical tangent")
        linear = velocity[:3]
        angular = velocity[3:]
        quaternion_tangent = _quaternion_tangent(
            target_quaternion, angular, self.convention
        )
        if self.convention == "body":
            position_tangent = _rotate(target_quaternion, linear)
        else:
            position_tangent = linear + jnp.cross(angular, target_position)
        target = jnp.concatenate((target_quaternion, target_position))
        point_tangent = jnp.concatenate((quaternion_tangent, position_tangent))
        return jax.jvp(
            lambda value: self.inverse_retract(anchor, value),
            (target,),
            (point_tangent,),
        )[1]

    def retraction_vjp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        anchor = _real_vector(state, 7, "Quaternion pose state")
        local = _real_vector(local_tangent, 6, "Quaternion-pose local tangent")
        target_cotangent = _real_vector(
            cotangent, 6, "Quaternion-pose physical cotangent"
        )
        return jax.linear_transpose(
            lambda direction: self.retraction_jvp(anchor, local, direction),
            jnp.zeros_like(local),
        )(target_cotangent)[0]

    def transport_tangent(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        self._point(state, "Transport source pose")
        self._point(point, "Transport target pose")
        return _real_vector(tangent, 6, "Quaternion-pose physical tangent")

    def transport_cotangent_pullback(
        self,
        state: ArrayLike,
        point: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        self._point(state, "Transport source pose")
        self._point(point, "Transport target pose")
        return _real_vector(cotangent, 6, "Quaternion-pose physical cotangent")


__all__ = [
    "QuaternionConvention",
    "QuaternionPoseStateGeometry",
    "ScalarFirstQuaternionStateGeometry",
]
