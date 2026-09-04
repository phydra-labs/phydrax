#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
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


def _coordinate_shape(
    value: ArrayLike,
    shape: tuple[int, ...],
    name: str,
    /,
) -> Array:
    array = jnp.asarray(value)
    rank = len(shape)
    if array.ndim < rank or (rank and array.shape[-rank:] != shape):
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
    skew = _skew(rotation)
    sine_vector = _so3_vee(skew)
    sine_squared = jnp.sum(sine_vector * sine_vector, axis=-1)
    near_zero = sine_squared < 1.0e-10
    safe_sine = jnp.sqrt(jnp.where(near_zero, 1.0, sine_squared))
    cosine = jnp.clip(
        (jnp.trace(rotation, axis1=-2, axis2=-1) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    rotation = eqx.error_if(
        rotation,
        jnp.any(near_zero & (cosine < 0.0)),
        "The principal SO(3) logarithm is ill-conditioned at rotations by pi.",
    )
    safe_angle = jnp.arctan2(safe_sine, cosine)
    factor = jnp.where(
        near_zero,
        1.0 + sine_squared / 6.0 + 3.0 * sine_squared * sine_squared / 40.0,
        safe_angle / safe_sine,
    )
    return factor[..., None, None] * skew


class AbstractLieGroup(StrictModule):
    """Minimal group, Lie-algebra, and trivialization contract."""

    group_id: AbstractAttribute[str]
    point_shape: AbstractAttribute[tuple[int, int]]
    algebra_shape: AbstractAttribute[tuple[int, ...]]

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

    @abstractmethod
    def hat(self, coordinates: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def vee(self, algebra: ArrayLike, /) -> Array:
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

    def right_trivialize(self, point: ArrayLike, tangent: ArrayLike, /) -> Array:
        point_array = _matrix_shape(point, self.point_shape, "Lie-group point")
        tangent_array = _matrix_shape(tangent, self.point_shape, "Lie-group tangent")
        return self.project_algebra(tangent_array @ self.inverse(point_array))

    def right_untrivialize(self, point: ArrayLike, algebra: ArrayLike, /) -> Array:
        point_array = _matrix_shape(point, self.point_shape, "Lie-group point")
        algebra_array = self.project_algebra(algebra)
        return algebra_array @ point_array

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
    point_shape: tuple[int, int] = eqx.field(static=True)
    algebra_shape: tuple[int, ...] = eqx.field(static=True)

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
        self.algebra_shape = (1,) if dimension_value == 2 else (3,)

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
    point_shape: tuple[int, int] = eqx.field(static=True)
    algebra_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, spatial_dimension: int, /, *, tolerance: float = 1e-6):
        dimension = int(spatial_dimension)
        if dimension not in (2, 3):
            raise ValueError("SpecialEuclideanGroup currently supports SE(2) and SE(3).")
        self.spatial_dimension = dimension
        self.tolerance = float(tolerance)
        self.rotation_group = SpecialOrthogonalGroup(dimension, tolerance=tolerance)
        self.group_id = f"lie-group:se:{dimension}"
        self.point_shape = (dimension + 1, dimension + 1)
        self.algebra_shape = (3,) if dimension == 2 else (6,)

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
            angle_squared = omega[..., 1, 0] ** 2
        else:
            rotation_coordinates = _so3_vee(omega)
            angle_squared = jnp.sum(rotation_coordinates * rotation_coordinates, axis=-1)
        near_zero = angle_squared < 1.0e-10
        safe_angle_squared = jnp.where(near_zero, 1.0, angle_squared)
        safe_angle = jnp.sqrt(safe_angle_squared)
        first_coefficient = jnp.where(
            near_zero,
            0.5 - angle_squared / 24.0 + angle_squared * angle_squared / 720.0,
            (1.0 - jnp.cos(safe_angle)) / safe_angle_squared,
        )
        second_coefficient = jnp.where(
            near_zero,
            1.0 / 6.0 - angle_squared / 120.0 + angle_squared * angle_squared / 5040.0,
            (safe_angle - jnp.sin(safe_angle)) / (safe_angle_squared * safe_angle),
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


def _lie_cut_locus_margin(
    group: AbstractLieGroup,
    local: ArrayLike,
    /,
) -> Array:
    coordinates = jnp.asarray(local)
    if isinstance(group, SpecialOrthogonalGroup):
        rotation = coordinates
    elif isinstance(group, SpecialEuclideanGroup):
        rotation = coordinates[..., group.spatial_dimension :]
    else:
        return jnp.asarray(1.0, dtype=coordinates.dtype)
    return jnp.maximum(
        jnp.asarray(jnp.pi, dtype=coordinates.dtype) - jnp.linalg.norm(rotation, axis=-1),
        jnp.asarray(0.0, dtype=coordinates.dtype),
    )


class LieGroupStateGeometry(AbstractStateGeometry):
    """Matrix Lie-group geometry with left increments and body velocities."""

    group: AbstractLieGroup
    convention: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_inverse: bool = eqx.field(static=True)
    supports_exact_differential: bool = eqx.field(static=True)
    supports_transport: bool = eqx.field(static=True)
    supports_isometric_transport: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(self, group: AbstractLieGroup, /):
        if not isinstance(group, AbstractLieGroup):
            raise TypeError("LieGroupStateGeometry requires an AbstractLieGroup.")
        self.group = group
        self.convention = "body"
        self.geometry_id = f"state-geometry:{group.group_id}:left:body"
        self.retraction_method = "left-group-exponential"
        self.trivial = False
        self.supports_exact_inverse = True
        self.supports_exact_differential = True
        self.supports_transport = True
        self.supports_isometric_transport = True
        self.supports_commutator_free = True

    def contains(self, state: ArrayLike, /) -> Array:
        return self.group.contains(state)

    def project_tangent(self, state: ArrayLike, vector: ArrayLike, /) -> Array:
        point = _matrix_shape(state, self.group.point_shape, "Lie-group point")
        ambient = _matrix_shape(
            vector, self.group.point_shape, "Ambient Lie-group tangent"
        )
        if ambient.shape != point.shape:
            raise ValueError("Ambient Lie-group tangent must match point shape.")
        return self.group.vee(self.group.left_trivialize(point, ambient))

    def retract(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        point = _matrix_shape(state, self.group.point_shape, "Lie-group point")
        local = _coordinate_shape(
            local_tangent, self.group.algebra_shape, "Lie-group local tangent"
        )
        return self.group.compose(point, self.group.exp(self.group.hat(local)))

    def inverse_retract(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        anchor = _matrix_shape(state, self.group.point_shape, "Lie-group chart anchor")
        target = _matrix_shape(point, self.group.point_shape, "Lie-group chart point")
        relative = self.group.compose(self.group.inverse(anchor), target)
        return self.group.vee(self.group.log(relative))

    def retraction_jvp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        local_velocity: ArrayLike,
        /,
    ) -> Array:
        point = _matrix_shape(state, self.group.point_shape, "Lie-group point")
        local = _coordinate_shape(
            local_tangent, self.group.algebra_shape, "Lie-group local tangent"
        )
        direction = _coordinate_shape(
            local_velocity, self.group.algebra_shape, "Lie-group local velocity"
        )
        target, ambient_tangent = jax.jvp(
            lambda value: self.retract(point, value),
            (local,),
            (direction,),
        )
        return self.project_tangent(target, ambient_tangent)

    def retraction_inverse_jvp(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        anchor = _matrix_shape(state, self.group.point_shape, "Lie-group chart anchor")
        target = _matrix_shape(point, self.group.point_shape, "Lie-group chart point")
        velocity = _coordinate_shape(
            tangent, self.group.algebra_shape, "Lie-group physical tangent"
        )
        ambient_tangent = self.group.left_untrivialize(target, self.group.hat(velocity))
        return jax.jvp(
            lambda value: self.inverse_retract(anchor, value),
            (target,),
            (ambient_tangent,),
        )[1]

    def retraction_vjp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        point = _matrix_shape(state, self.group.point_shape, "Lie-group point")
        local = _coordinate_shape(
            local_tangent, self.group.algebra_shape, "Lie-group local tangent"
        )
        target_cotangent = _coordinate_shape(
            cotangent, self.group.algebra_shape, "Lie-group physical cotangent"
        )
        return jax.linear_transpose(
            lambda direction: self.retraction_jvp(point, local, direction),
            jnp.zeros_like(local),
        )(target_cotangent)[0]

    def transport_tangent(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        _matrix_shape(state, self.group.point_shape, "Transport source point")
        _matrix_shape(point, self.group.point_shape, "Transport target point")
        return _coordinate_shape(
            tangent, self.group.algebra_shape, "Body physical tangent"
        )

    def transport_cotangent_pullback(
        self,
        state: ArrayLike,
        point: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        _matrix_shape(state, self.group.point_shape, "Transport source point")
        _matrix_shape(point, self.group.point_shape, "Transport target point")
        return _coordinate_shape(
            cotangent, self.group.algebra_shape, "Body physical cotangent"
        )

    def cut_locus_margin(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        return _lie_cut_locus_margin(
            self.group,
            self.inverse_retract(state, point),
        )


class RightLieGroupStateGeometry(AbstractStateGeometry):
    """Matrix Lie-group geometry with right increments and spatial velocities."""

    group: AbstractLieGroup
    convention: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_inverse: bool = eqx.field(static=True)
    supports_exact_differential: bool = eqx.field(static=True)
    supports_transport: bool = eqx.field(static=True)
    supports_isometric_transport: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(self, group: AbstractLieGroup, /):
        if not isinstance(group, AbstractLieGroup):
            raise TypeError("RightLieGroupStateGeometry requires an AbstractLieGroup.")
        self.group = group
        self.convention = "spatial"
        self.geometry_id = f"state-geometry:{group.group_id}:right:spatial"
        self.retraction_method = "right-group-exponential"
        self.trivial = False
        self.supports_exact_inverse = True
        self.supports_exact_differential = True
        self.supports_transport = True
        self.supports_isometric_transport = True
        self.supports_commutator_free = True

    def contains(self, state: ArrayLike, /) -> Array:
        return self.group.contains(state)

    def project_tangent(self, state: ArrayLike, vector: ArrayLike, /) -> Array:
        point = _matrix_shape(state, self.group.point_shape, "Lie-group point")
        ambient = _matrix_shape(
            vector, self.group.point_shape, "Ambient Lie-group tangent"
        )
        if ambient.shape != point.shape:
            raise ValueError("Ambient Lie-group tangent must match point shape.")
        return self.group.vee(self.group.right_trivialize(point, ambient))

    def retract(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        point = _matrix_shape(state, self.group.point_shape, "Lie-group point")
        local = _coordinate_shape(
            local_tangent, self.group.algebra_shape, "Lie-group local tangent"
        )
        return self.group.compose(self.group.exp(self.group.hat(local)), point)

    def inverse_retract(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        anchor = _matrix_shape(state, self.group.point_shape, "Lie-group chart anchor")
        target = _matrix_shape(point, self.group.point_shape, "Lie-group chart point")
        relative = self.group.compose(target, self.group.inverse(anchor))
        return self.group.vee(self.group.log(relative))

    def retraction_jvp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        local_velocity: ArrayLike,
        /,
    ) -> Array:
        point = _matrix_shape(state, self.group.point_shape, "Lie-group point")
        local = _coordinate_shape(
            local_tangent, self.group.algebra_shape, "Lie-group local tangent"
        )
        direction = _coordinate_shape(
            local_velocity, self.group.algebra_shape, "Lie-group local velocity"
        )
        target, ambient_tangent = jax.jvp(
            lambda value: self.retract(point, value),
            (local,),
            (direction,),
        )
        return self.project_tangent(target, ambient_tangent)

    def retraction_inverse_jvp(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        anchor = _matrix_shape(state, self.group.point_shape, "Lie-group chart anchor")
        target = _matrix_shape(point, self.group.point_shape, "Lie-group chart point")
        velocity = _coordinate_shape(
            tangent, self.group.algebra_shape, "Lie-group physical tangent"
        )
        ambient_tangent = self.group.right_untrivialize(target, self.group.hat(velocity))
        return jax.jvp(
            lambda value: self.inverse_retract(anchor, value),
            (target,),
            (ambient_tangent,),
        )[1]

    def retraction_vjp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        point = _matrix_shape(state, self.group.point_shape, "Lie-group point")
        local = _coordinate_shape(
            local_tangent, self.group.algebra_shape, "Lie-group local tangent"
        )
        target_cotangent = _coordinate_shape(
            cotangent, self.group.algebra_shape, "Lie-group physical cotangent"
        )
        return jax.linear_transpose(
            lambda direction: self.retraction_jvp(point, local, direction),
            jnp.zeros_like(local),
        )(target_cotangent)[0]

    def transport_tangent(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        _matrix_shape(state, self.group.point_shape, "Transport source point")
        _matrix_shape(point, self.group.point_shape, "Transport target point")
        return _coordinate_shape(
            tangent, self.group.algebra_shape, "Spatial physical tangent"
        )

    def transport_cotangent_pullback(
        self,
        state: ArrayLike,
        point: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        _matrix_shape(state, self.group.point_shape, "Transport source point")
        _matrix_shape(point, self.group.point_shape, "Transport target point")
        return _coordinate_shape(
            cotangent, self.group.algebra_shape, "Spatial physical cotangent"
        )

    def cut_locus_margin(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        return _lie_cut_locus_margin(
            self.group,
            self.inverse_retract(state, point),
        )


__all__ = [
    "AbstractLieGroup",
    "LieGroupStateGeometry",
    "RightLieGroupStateGeometry",
    "SpecialEuclideanGroup",
    "SpecialOrthogonalGroup",
]
