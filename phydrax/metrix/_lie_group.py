#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from .._strict import AbstractAttribute, StrictModule
from ._matrix_manifold import SpecialOrthogonalManifold
from ._state_geometry import AbstractStateGeometry


def _transpose(matrix: Array, /) -> Array:
    return jnp.swapaxes(matrix, -1, -2)


def _skew(matrix: Array, /) -> Array:
    return 0.5 * (matrix - _transpose(matrix))


def _matrix_shape(value: ArrayLike, shape: tuple[int, int], name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.shape[-2:] != shape:
        raise ValueError(f"{name} must have trailing shape {shape}; got {array.shape}.")
    return array


def _so3_hat(vector: Array, /) -> Array:
    x = vector[..., 0]
    y = vector[..., 1]
    z = vector[..., 2]
    zero = jnp.zeros_like(x)
    return jnp.stack(
        (
            jnp.stack((zero, -z, y), axis=-1),
            jnp.stack((z, zero, -x), axis=-1),
            jnp.stack((-y, x, zero), axis=-1),
        ),
        axis=-2,
    )


def _so3_vee(matrix: Array, /) -> Array:
    return jnp.stack(
        (matrix[..., 2, 1], matrix[..., 0, 2], matrix[..., 1, 0]),
        axis=-1,
    )


def _rotation_log(rotation: Array, dimension: int, /) -> Array:
    if dimension == 2:
        angle = jnp.arctan2(rotation[..., 1, 0], rotation[..., 0, 0])
        generator = jnp.asarray(((0.0, -1.0), (1.0, 0.0)), dtype=rotation.dtype)
        return angle[..., None, None] * generator
    cosine = jnp.clip(
        (jnp.trace(rotation, axis1=-2, axis2=-1) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    angle = jnp.arccos(cosine)
    sine = jnp.sin(angle)
    near_zero = jnp.abs(angle) < 1e-5
    near_pi = jnp.abs(jnp.pi - angle) < 1e-5
    rotation = eqx.error_if(
        rotation,
        jnp.any(near_pi),
        "The principal SO(3) logarithm is ill-conditioned at rotations by pi.",
    )
    safe_sine = jnp.where(near_zero, 1.0, sine)
    factor = jnp.where(
        near_zero,
        0.5 + angle**2 / 12.0 + 7.0 * angle**4 / 720.0,
        angle / (2.0 * safe_sine),
    )
    return factor[..., None, None] * (rotation - _transpose(rotation))


class AbstractLieGroup(StrictModule):
    """Minimal group, Lie-algebra, and trivialization contract."""

    group_id: AbstractAttribute[str]
    point_shape: AbstractAttribute[tuple[int, ...]]

    @abstractmethod
    def identity(self, *, dtype: Any = jnp.float64) -> Array:
        raise NotImplementedError

    @abstractmethod
    def contains(self, point: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def compose(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def inverse(self, point: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def project_algebra(self, value: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def exp(self, algebra: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def log(self, point: ArrayLike, /) -> Array:
        raise NotImplementedError

    def lie_bracket(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_array = _matrix_shape(left, self.point_shape, "Left Lie-algebra element")
        right_array = _matrix_shape(right, self.point_shape, "Right Lie-algebra element")
        return left_array @ right_array - right_array @ left_array

    def left_trivialize(self, point: ArrayLike, tangent: ArrayLike, /) -> Array:
        point_array = _matrix_shape(point, self.point_shape, "Lie-group point")
        tangent_array = _matrix_shape(tangent, self.point_shape, "Lie-group tangent")
        return self.project_algebra(self.inverse(point_array) @ tangent_array)

    def left_untrivialize(self, point: ArrayLike, algebra: ArrayLike, /) -> Array:
        point_array = _matrix_shape(point, self.point_shape, "Lie-group point")
        algebra_array = self.project_algebra(algebra)
        return point_array @ algebra_array

    def adjoint(self, point: ArrayLike, algebra: ArrayLike, /) -> Array:
        point_array = _matrix_shape(point, self.point_shape, "Lie-group point")
        algebra_array = self.project_algebra(algebra)
        return point_array @ algebra_array @ self.inverse(point_array)


class SpecialOrthogonalGroup(AbstractLieGroup):
    """SO(2) or SO(3) with matrix-valued Lie-algebra elements."""

    dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    manifold: SpecialOrthogonalManifold
    group_id: str = eqx.field(static=True)
    point_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, dimension: int, /, *, tolerance: float = 1e-6):
        dimension_value = int(dimension)
        if dimension_value not in (2, 3):
            raise ValueError("SpecialOrthogonalGroup currently supports SO(2) and SO(3).")
        self.dimension = dimension_value
        self.tolerance = float(tolerance)
        self.manifold = SpecialOrthogonalManifold(
            dimension_value,
            retraction="exponential",
            tolerance=tolerance,
        )
        self.group_id = f"lie-group:so:{dimension_value}"
        self.point_shape = (dimension_value, dimension_value)

    def identity(self, *, dtype: Any = jnp.float64) -> Array:
        return jnp.eye(self.dimension, dtype=dtype)

    def contains(self, point: ArrayLike, /) -> Array:
        return self.manifold.contains(point)

    def compose(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return _matrix_shape(left, self.point_shape, "SO point") @ _matrix_shape(
            right, self.point_shape, "SO point"
        )

    def inverse(self, point: ArrayLike, /) -> Array:
        return _transpose(_matrix_shape(point, self.point_shape, "SO point"))

    def project_algebra(self, value: ArrayLike, /) -> Array:
        return _skew(_matrix_shape(value, self.point_shape, "so algebra element"))

    def exp(self, algebra: ArrayLike, /) -> Array:
        return jsp.linalg.expm(self.project_algebra(algebra))

    def log(self, point: ArrayLike, /) -> Array:
        rotation = _matrix_shape(point, self.point_shape, "SO point")
        rotation = eqx.error_if(
            rotation,
            ~self.contains(rotation),
            "SO logarithm requires an orthogonal matrix with determinant one.",
        )
        return _rotation_log(rotation, self.dimension)

    def hat(self, coordinates: ArrayLike, /) -> Array:
        values = jnp.asarray(coordinates)
        expected = (1,) if self.dimension == 2 else (3,)
        if values.shape[-1:] != expected:
            raise ValueError(
                f"SO algebra coordinates must have trailing shape {expected}."
            )
        if self.dimension == 2:
            angle = values[..., 0]
            generator = jnp.asarray(
                ((0.0, -1.0), (1.0, 0.0)),
                dtype=values.dtype,
            )
            return angle[..., None, None] * generator
        return _so3_hat(values)

    def vee(self, algebra: ArrayLike, /) -> Array:
        matrix = self.project_algebra(algebra)
        if self.dimension == 2:
            return matrix[..., 1, 0][..., None]
        return _so3_vee(matrix)


class SpecialEuclideanGroup(AbstractLieGroup):
    """SE(2) or SE(3) in homogeneous-matrix representation."""

    spatial_dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    rotation_group: SpecialOrthogonalGroup
    group_id: str = eqx.field(static=True)
    point_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, spatial_dimension: int, /, *, tolerance: float = 1e-6):
        dimension = int(spatial_dimension)
        if dimension not in (2, 3):
            raise ValueError("SpecialEuclideanGroup currently supports SE(2) and SE(3).")
        self.spatial_dimension = dimension
        self.tolerance = float(tolerance)
        self.rotation_group = SpecialOrthogonalGroup(dimension, tolerance=tolerance)
        self.group_id = f"lie-group:se:{dimension}"
        self.point_shape = (dimension + 1, dimension + 1)

    def identity(self, *, dtype: Any = jnp.float64) -> Array:
        return jnp.eye(self.spatial_dimension + 1, dtype=dtype)

    def contains(self, point: ArrayLike, /) -> Array:
        matrix = _matrix_shape(point, self.point_shape, "SE point")
        bottom = matrix[..., -1, :]
        expected = jnp.concatenate(
            (
                jnp.zeros((self.spatial_dimension,), dtype=matrix.dtype),
                jnp.ones((1,), dtype=matrix.dtype),
            )
        )
        return self.rotation_group.contains(
            matrix[..., : self.spatial_dimension, : self.spatial_dimension]
        ) & jnp.all(jnp.abs(bottom - expected) <= self.tolerance)

    def compose(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return _matrix_shape(left, self.point_shape, "SE point") @ _matrix_shape(
            right, self.point_shape, "SE point"
        )

    def inverse(self, point: ArrayLike, /) -> Array:
        matrix = _matrix_shape(point, self.point_shape, "SE point")
        rotation = matrix[..., : self.spatial_dimension, : self.spatial_dimension]
        translation = matrix[..., : self.spatial_dimension, -1]
        result = jnp.broadcast_to(
            self.identity(dtype=matrix.dtype),
            matrix.shape,
        )
        result = result.at[..., : self.spatial_dimension, : self.spatial_dimension].set(
            _transpose(rotation)
        )
        return result.at[..., : self.spatial_dimension, -1].set(
            -(_transpose(rotation) @ translation[..., None])[..., 0]
        )

    def project_algebra(self, value: ArrayLike, /) -> Array:
        matrix = _matrix_shape(value, self.point_shape, "se algebra element")
        result = jnp.zeros_like(matrix)
        result = result.at[..., : self.spatial_dimension, : self.spatial_dimension].set(
            _skew(matrix[..., : self.spatial_dimension, : self.spatial_dimension])
        )
        return result.at[..., : self.spatial_dimension, -1].set(
            matrix[..., : self.spatial_dimension, -1]
        )

    def exp(self, algebra: ArrayLike, /) -> Array:
        return jsp.linalg.expm(self.project_algebra(algebra))

    def log(self, point: ArrayLike, /) -> Array:
        matrix = _matrix_shape(point, self.point_shape, "SE point")
        matrix = eqx.error_if(
            matrix,
            ~self.contains(matrix),
            "SE logarithm requires a valid homogeneous rigid transformation.",
        )
        rotation = matrix[..., : self.spatial_dimension, : self.spatial_dimension]
        translation = matrix[..., : self.spatial_dimension, -1]
        omega = _rotation_log(rotation, self.spatial_dimension)
        if self.spatial_dimension == 2:
            angle = omega[..., 1, 0]
        else:
            angle = jnp.linalg.norm(_so3_vee(omega), axis=-1)
        near_zero = jnp.abs(angle) < 1e-5
        angle_squared = angle**2
        safe_angle_squared = jnp.where(near_zero, 1.0, angle_squared)
        safe_angle = jnp.where(near_zero, 1.0, angle)
        first_coefficient = jnp.where(
            near_zero,
            0.5 - angle_squared / 24.0,
            (1.0 - jnp.cos(angle)) / safe_angle_squared,
        )
        second_coefficient = jnp.where(
            near_zero,
            1.0 / 6.0 - angle_squared / 120.0,
            (angle - jnp.sin(angle)) / (safe_angle_squared * safe_angle),
        )
        identity = jnp.broadcast_to(
            jnp.eye(self.spatial_dimension, dtype=matrix.dtype),
            omega.shape,
        )
        left_jacobian = (
            identity
            + first_coefficient[..., None, None] * omega
            + second_coefficient[..., None, None] * (omega @ omega)
        )
        velocity = jnp.linalg.solve(left_jacobian, translation[..., None])[..., 0]
        result = jnp.zeros_like(matrix)
        result = result.at[..., : self.spatial_dimension, : self.spatial_dimension].set(
            omega
        )
        return result.at[..., : self.spatial_dimension, -1].set(velocity)

    def hat(self, coordinates: ArrayLike, /) -> Array:
        values = jnp.asarray(coordinates)
        rotation_dimension = 1 if self.spatial_dimension == 2 else 3
        expected = (self.spatial_dimension + rotation_dimension,)
        if values.shape[-1:] != expected:
            raise ValueError(
                f"SE algebra coordinates must have trailing shape {expected}."
            )
        result = jnp.zeros(values.shape[:-1] + self.point_shape, dtype=values.dtype)
        rotation = self.rotation_group.hat(values[..., self.spatial_dimension :])
        result = result.at[..., : self.spatial_dimension, : self.spatial_dimension].set(
            rotation
        )
        return result.at[..., : self.spatial_dimension, -1].set(
            values[..., : self.spatial_dimension]
        )

    def vee(self, algebra: ArrayLike, /) -> Array:
        matrix = self.project_algebra(algebra)
        translation = matrix[..., : self.spatial_dimension, -1]
        rotation = self.rotation_group.vee(
            matrix[..., : self.spatial_dimension, : self.spatial_dimension]
        )
        return jnp.concatenate((translation, rotation), axis=-1)


class LieGroupStateGeometry(AbstractStateGeometry):
    """State-space adapter using left-trivialized matrix Lie-group increments."""

    group: AbstractLieGroup
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_pullback: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(self, group: AbstractLieGroup, /):
        if not isinstance(group, AbstractLieGroup):
            raise TypeError("LieGroupStateGeometry requires an AbstractLieGroup.")
        self.group = group
        self.geometry_id = f"state-geometry:{group.group_id}:left"
        self.retraction_method = "group-exponential"
        self.trivial = False
        self.supports_exact_pullback = False
        self.supports_commutator_free = True

    def contains(self, state: ArrayLike, /) -> Array:
        return self.group.contains(state)

    def project_tangent(self, state: ArrayLike, vector: ArrayLike, /) -> Array:
        return self.from_local(state, self.to_local(state, vector))

    def to_local(self, state: ArrayLike, tangent: ArrayLike, /) -> Array:
        return self.group.left_trivialize(state, tangent)

    def from_local(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        return self.group.left_untrivialize(state, local_tangent)

    def retract(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        return self.group.compose(state, self.group.exp(local_tangent))

    def inverse_retract(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        relative = self.group.compose(self.group.inverse(state), point)
        return self.group.log(relative)

    def pullback(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        del state, local_tangent, tangent
        raise ValueError(
            "LieGroupStateGeometry does not claim an exact exponential pullback."
        )


__all__ = [
    "AbstractLieGroup",
    "LieGroupStateGeometry",
    "SpecialEuclideanGroup",
    "SpecialOrthogonalGroup",
]
