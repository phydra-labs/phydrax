#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._guarantee import ContactGuaranteeLevel


class ContactTrajectoryEvaluation(StrictModule):
    positions: Array
    lower_bound: Array
    upper_bound: Array
    guarantee_level: Array
    finite: Array
    successful: Array
    trajectory_id: str = eqx.field(static=True)


class ContactPathEnclosure(StrictModule):
    """Per-vertex axis-aligned enclosure of one trajectory over a time interval.

    ``lower``/``upper`` contain every vertex position of the declared path for
    all normalized times in ``[lower_time, upper_time]``. ``guarantee_level`` is
    declared by the trajectory owner, never selected by a caller. ``revision``
    is the dynamic numerical revision of the bound path data; together with the
    static ``trajectory_id`` it binds search coverage to one actual trajectory.
    """

    lower: Array
    upper: Array
    lower_time: Array
    upper_time: Array
    guarantee_level: Array
    revision: Array
    finite: Array
    successful: Array
    trajectory_id: str = eqx.field(static=True)


class ContactPathMidpointBound(StrictModule):
    """Midpoint positions and Hausdorff displacement bounds over an interval.

    For every vertex ``i`` and time ``s`` in ``[lower_time, upper_time]``,
    ``|x_i(s) - positions[i]| <= displacement[i]``. Any convex core feature
    spanned by such vertices moves by at most the maximum displacement of its
    vertices, so a primitive distance at the midpoint minus both features'
    displacement bounds is a lower bound of the distance over the interval.
    """

    midpoint_time: Array
    positions: Array
    displacement: Array
    lower_time: Array
    upper_time: Array
    guarantee_level: Array
    revision: Array
    finite: Array
    successful: Array
    trajectory_id: str = eqx.field(static=True)


def _interval_times(
    lower_time: ArrayLike, upper_time: ArrayLike, dtype: jnp.dtype, /
) -> tuple[Array, Array, Array]:
    lower = jnp.asarray(lower_time, dtype=dtype)
    upper = jnp.asarray(upper_time, dtype=dtype)
    if lower.shape != () or upper.shape != ():
        raise ValueError("Trajectory interval endpoints must be scalars.")
    valid = (
        jnp.isfinite(lower)
        & jnp.isfinite(upper)
        & (lower >= 0.0)
        & (lower <= upper)
        & (upper <= 1.0)
    )
    return lower, upper, valid


def _revision(value: ArrayLike, /) -> Array:
    revision = jnp.asarray(value)
    if revision.shape != () or not jnp.issubdtype(revision.dtype, jnp.integer):
        raise TypeError("Trajectory revision must be one integer scalar.")
    return revision.astype(jnp.int64)


def _box_displacement(positions: Array, lower: Array, upper: Array, /) -> Array:
    reach = jnp.maximum(jnp.abs(upper - positions), jnp.abs(positions - lower))
    return jnp.sqrt(jnp.sum(reach * reach, axis=-1))


class AbstractContactTrajectory(StrictModule, NonTrainableState):
    """A declared vertex path over normalized time ``s`` in ``[0, 1]``.

    Implementations own their path definition and the strength of their
    enclosure guarantee. Endpoints alone never define a nonlinear path.
    """

    @property
    @abc.abstractmethod
    def trajectory_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def guarantee_level(self) -> ContactGuaranteeLevel:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def revision(self) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, time: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def bounds(
        self, lower_time: ArrayLike, upper_time: ArrayLike, /
    ) -> tuple[Array, Array]:
        raise NotImplementedError

    def enclosure(
        self, lower_time: ArrayLike = 0.0, upper_time: ArrayLike = 1.0, /
    ) -> ContactPathEnclosure:
        lower, upper = self.bounds(lower_time, upper_time)
        lower_time_, upper_time_, interval_valid = _interval_times(
            lower_time, upper_time, lower.dtype
        )
        finite = jnp.all(jnp.isfinite(lower)) & jnp.all(jnp.isfinite(upper))
        ordered = jnp.all(lower <= upper)
        return ContactPathEnclosure(
            lower,
            upper,
            lower_time_,
            upper_time_,
            jnp.asarray(int(self.guarantee_level), dtype=jnp.int32),
            self.revision,
            finite,
            finite & ordered & interval_valid,
            self.trajectory_id,
        )

    def midpoint_bound(
        self, lower_time: ArrayLike = 0.0, upper_time: ArrayLike = 1.0, /
    ) -> ContactPathMidpointBound:
        enclosure = self.enclosure(lower_time, upper_time)
        midpoint = 0.5 * (enclosure.lower_time + enclosure.upper_time)
        positions = self.evaluate(midpoint)
        inside = jnp.all(positions >= enclosure.lower) & jnp.all(
            positions <= enclosure.upper
        )
        displacement = _box_displacement(positions, enclosure.lower, enclosure.upper)
        finite = (
            enclosure.finite
            & jnp.all(jnp.isfinite(positions))
            & jnp.all(jnp.isfinite(displacement))
        )
        return ContactPathMidpointBound(
            midpoint,
            positions,
            displacement,
            enclosure.lower_time,
            enclosure.upper_time,
            enclosure.guarantee_level,
            enclosure.revision,
            finite,
            enclosure.successful & finite & inside,
            self.trajectory_id,
        )

    def interval_bounds(
        self, lower_time: ArrayLike = 0.0, upper_time: ArrayLike = 1.0, /
    ) -> tuple[ContactPathEnclosure, ContactPathMidpointBound]:
        """Return the box enclosure and midpoint bound of one interval."""
        return (
            self.enclosure(lower_time, upper_time),
            self.midpoint_bound(lower_time, upper_time),
        )

    def evaluate_with_bounds(
        self,
        time: ArrayLike,
        /,
        *,
        lower_time: ArrayLike = 0.0,
        upper_time: ArrayLike = 1.0,
    ) -> ContactTrajectoryEvaluation:
        positions = self.evaluate(time)
        enclosure = self.enclosure(lower_time, upper_time)
        finite = enclosure.finite & jnp.all(jnp.isfinite(positions))
        return ContactTrajectoryEvaluation(
            positions,
            enclosure.lower,
            enclosure.upper,
            enclosure.guarantee_level,
            finite,
            finite & enclosure.successful,
            self.trajectory_id,
        )


class LinearContactTrajectory(AbstractContactTrajectory):
    """Vertexwise straight-line motion ``x(s) = x0 + s (x1 - x0)``."""

    start: Array
    end: Array
    path_revision: Array
    _trajectory_id: str = eqx.field(static=True)

    def __init__(
        self, start: ArrayLike, end: ArrayLike, /, *, revision: ArrayLike = 0
    ) -> None:
        start_ = jnp.asarray(start)
        end_ = jnp.asarray(end, dtype=start_.dtype)
        if start_.shape != end_.shape or start_.ndim != 2:
            raise ValueError(
                "Linear contact trajectory endpoints must be matching matrices."
            )
        self.start = start_
        self.end = end_
        self.path_revision = _revision(revision)
        self._trajectory_id = canonical_fingerprint(
            {
                "kind": "linear-contact-trajectory",
                "shape": tuple(start_.shape),
                "dtype": str(start_.dtype),
            }
        )

    @property
    def trajectory_id(self) -> str:
        return self._trajectory_id

    @property
    def guarantee_level(self) -> ContactGuaranteeLevel:
        return ContactGuaranteeLevel.ANALYTIC_CONSERVATIVE

    @property
    def revision(self) -> Array:
        return self.path_revision

    def evaluate(self, time: ArrayLike, /) -> Array:
        time_ = jnp.asarray(time, dtype=self.start.dtype)
        return self.start + time_ * (self.end - self.start)

    def bounds(
        self, lower_time: ArrayLike, upper_time: ArrayLike, /
    ) -> tuple[Array, Array]:
        lower_point = self.evaluate(lower_time)
        upper_point = self.evaluate(upper_time)
        return jnp.minimum(lower_point, upper_point), jnp.maximum(
            lower_point, upper_point
        )


def _bernstein_blossom(controls: Array, parameters: tuple[Array, Array, Array], /) -> Array:
    values = controls
    for parameter in parameters:
        values = (1.0 - parameter) * values[:-1] + parameter * values[1:]
    return values[0]


class CubicHermiteContactTrajectory(AbstractContactTrajectory):
    """Cubic Hermite trajectory with exact Bernstein restriction bounds.

    Subinterval bounds use the exact Bernstein control polygon of the cubic
    restricted to ``[lower_time, upper_time]`` (de Casteljau blossoming); its
    convex hull contains the restricted curve and contracts with the interval.
    """

    start: Array
    start_tangent: Array
    end: Array
    end_tangent: Array
    controls: Array
    path_revision: Array
    _trajectory_id: str = eqx.field(static=True)

    def __init__(
        self,
        start: ArrayLike,
        start_tangent: ArrayLike,
        end: ArrayLike,
        end_tangent: ArrayLike,
        /,
        *,
        revision: ArrayLike = 0,
    ) -> None:
        start_ = jnp.asarray(start)
        start_tangent_ = jnp.asarray(start_tangent, dtype=start_.dtype)
        end_ = jnp.asarray(end, dtype=start_.dtype)
        end_tangent_ = jnp.asarray(end_tangent, dtype=start_.dtype)
        if (
            start_.shape != end_.shape
            or start_.shape != start_tangent_.shape
            or start_.shape != end_tangent_.shape
            or start_.ndim != 2
        ):
            raise ValueError("Cubic contact trajectory values must be matching matrices.")
        controls = jnp.stack(
            (
                start_,
                start_ + start_tangent_ / 3.0,
                end_ - end_tangent_ / 3.0,
                end_,
            ),
            axis=0,
        )
        self.start = start_
        self.start_tangent = start_tangent_
        self.end = end_
        self.end_tangent = end_tangent_
        self.controls = controls
        self.path_revision = _revision(revision)
        self._trajectory_id = canonical_fingerprint(
            {
                "kind": "cubic-hermite-contact-trajectory",
                "shape": tuple(start_.shape),
                "dtype": str(start_.dtype),
            }
        )

    @property
    def trajectory_id(self) -> str:
        return self._trajectory_id

    @property
    def guarantee_level(self) -> ContactGuaranteeLevel:
        return ContactGuaranteeLevel.ANALYTIC_CONSERVATIVE

    @property
    def revision(self) -> Array:
        return self.path_revision

    def evaluate(self, time: ArrayLike, /) -> Array:
        time_ = jnp.asarray(time, dtype=self.start.dtype)
        return _bernstein_blossom(self.controls, (time_, time_, time_))

    def restricted_controls(
        self, lower_time: ArrayLike, upper_time: ArrayLike, /
    ) -> Array:
        """Return the exact Bernstein controls of the cubic on a subinterval."""
        lower = jnp.asarray(lower_time, dtype=self.start.dtype)
        upper = jnp.asarray(upper_time, dtype=self.start.dtype)
        return jnp.stack(
            (
                _bernstein_blossom(self.controls, (lower, lower, lower)),
                _bernstein_blossom(self.controls, (lower, lower, upper)),
                _bernstein_blossom(self.controls, (lower, upper, upper)),
                _bernstein_blossom(self.controls, (upper, upper, upper)),
            ),
            axis=0,
        )

    def bounds(
        self, lower_time: ArrayLike, upper_time: ArrayLike, /
    ) -> tuple[Array, Array]:
        controls = self.restricted_controls(lower_time, upper_time)
        return jnp.min(controls, axis=0), jnp.max(controls, axis=0)


def _skew(vector: Array, /) -> Array:
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    zero = jnp.zeros_like(x)
    return jnp.stack((zero, -z, y, z, zero, -x, -y, x, zero), axis=-1).reshape(
        vector.shape[:-1] + (3, 3)
    )


def _rotation_exponential(rotation_vector: Array, /) -> Array:
    if rotation_vector.shape[-1] == 1:
        angle = rotation_vector[..., 0]
        cosine = jnp.cos(angle)
        sine = jnp.sin(angle)
        return jnp.stack((cosine, -sine, sine, cosine), axis=-1).reshape(
            rotation_vector.shape[:-1] + (2, 2)
        )
    squared = jnp.sum(rotation_vector * rotation_vector, axis=-1)
    small = squared <= jnp.finfo(rotation_vector.dtype).eps
    safe_squared = jnp.where(small, 1.0, squared)
    angle = jnp.sqrt(safe_squared)
    first = jnp.where(
        small,
        1.0 - squared / 6.0 + squared * squared / 120.0,
        jnp.sin(angle) / angle,
    )
    second = jnp.where(
        small,
        0.5 - squared / 24.0 + squared * squared / 720.0,
        (1.0 - jnp.cos(angle)) / safe_squared,
    )
    skew = _skew(rotation_vector)
    identity = jnp.broadcast_to(jnp.eye(3, dtype=rotation_vector.dtype), skew.shape)
    return identity + first[..., None, None] * skew + second[..., None, None] * (skew @ skew)


class RigidSweepContactTrajectory(AbstractContactTrajectory):
    """Rigid vertex motion with an explicit exponential rotation path.

    ``x(s) = c0 + s (c1 - c0) + exp(s [w]) R0 l`` where ``w`` is the declared
    world-frame rotation increment (one angle in two dimensions). The rotation
    increment is part of the path: endpoint rotations alone do not determine
    full turns. Bounds combine the rotation-independent sphere around the
    center path with the midpoint displacement ball.
    """

    local_vertices: Array
    start_center: Array
    end_center: Array
    start_rotation: Array
    rotation_increment: Array
    vertex_radius: Array
    path_revision: Array
    _trajectory_id: str = eqx.field(static=True)

    def __init__(
        self,
        local_vertices: ArrayLike,
        start_center: ArrayLike,
        end_center: ArrayLike,
        start_rotation: ArrayLike,
        rotation_increment: ArrayLike,
        /,
        *,
        revision: ArrayLike = 0,
    ) -> None:
        vertices = jnp.asarray(local_vertices)
        start_center_ = jnp.asarray(start_center, dtype=vertices.dtype)
        end_center_ = jnp.asarray(end_center, dtype=vertices.dtype)
        start_rotation_ = jnp.asarray(start_rotation, dtype=vertices.dtype)
        increment = jnp.asarray(rotation_increment, dtype=vertices.dtype)
        dimension = vertices.shape[-1]
        if vertices.ndim != 2 or dimension not in (2, 3):
            raise ValueError("Rigid contact vertices require dimension two or three.")
        if start_center_.shape != (dimension,) or end_center_.shape != (dimension,):
            raise ValueError("Rigid contact centers have invalid shape.")
        if start_rotation_.shape != (dimension, dimension):
            raise ValueError("Rigid contact start rotation has invalid shape.")
        if increment.shape != ((1,) if dimension == 2 else (3,)):
            raise ValueError(
                "Rigid rotation increment must be one angle in 2D or one rotation vector in 3D."
            )
        self.local_vertices = vertices
        self.start_center = start_center_
        self.end_center = end_center_
        self.start_rotation = start_rotation_
        self.rotation_increment = increment
        self.vertex_radius = jnp.sqrt(jnp.sum(vertices * vertices, axis=-1))
        self.path_revision = _revision(revision)
        self._trajectory_id = canonical_fingerprint(
            {
                "kind": "rigid-sweep-contact-trajectory",
                "shape": tuple(vertices.shape),
                "dtype": str(vertices.dtype),
            }
        )

    @property
    def trajectory_id(self) -> str:
        return self._trajectory_id

    @property
    def guarantee_level(self) -> ContactGuaranteeLevel:
        return ContactGuaranteeLevel.ANALYTIC_CONSERVATIVE

    @property
    def revision(self) -> Array:
        return self.path_revision

    def _center(self, time: Array, /) -> Array:
        return self.start_center + time * (self.end_center - self.start_center)

    def evaluate(self, time: ArrayLike, /) -> Array:
        time_ = jnp.asarray(time, dtype=self.local_vertices.dtype)
        rotation = _rotation_exponential(time_ * self.rotation_increment)
        world = rotation @ self.start_rotation
        return self.local_vertices @ world.T + self._center(time_)

    def _speed_bound(self, /) -> Array:
        translation = self.end_center - self.start_center
        linear = jnp.sqrt(jnp.sum(translation * translation))
        angular = jnp.sqrt(jnp.sum(self.rotation_increment * self.rotation_increment))
        speed = linear + angular * self.vertex_radius
        return speed * (1.0 + 16.0 * jnp.finfo(speed.dtype).eps)

    def midpoint_bound(
        self, lower_time: ArrayLike = 0.0, upper_time: ArrayLike = 1.0, /
    ) -> ContactPathMidpointBound:
        lower, upper, valid = _interval_times(
            lower_time, upper_time, self.local_vertices.dtype
        )
        midpoint = 0.5 * (lower + upper)
        positions = self.evaluate(midpoint)
        scale = jnp.max(jnp.abs(positions), axis=-1) + self.vertex_radius
        allowance = 64.0 * jnp.finfo(positions.dtype).eps * scale
        displacement = 0.5 * (upper - lower) * self._speed_bound() + allowance
        finite = jnp.all(jnp.isfinite(positions)) & jnp.all(jnp.isfinite(displacement))
        return ContactPathMidpointBound(
            midpoint,
            positions,
            displacement,
            lower,
            upper,
            jnp.asarray(int(self.guarantee_level), dtype=jnp.int32),
            self.path_revision,
            finite,
            finite & valid,
            self.trajectory_id,
        )

    def bounds(
        self, lower_time: ArrayLike, upper_time: ArrayLike, /
    ) -> tuple[Array, Array]:
        lower_time_ = jnp.asarray(lower_time, dtype=self.start_center.dtype)
        upper_time_ = jnp.asarray(upper_time, dtype=self.start_center.dtype)
        lower_center = self._center(lower_time_)
        upper_center = self._center(upper_time_)
        center_min = jnp.minimum(lower_center, upper_center)
        center_max = jnp.maximum(lower_center, upper_center)
        radius = self.vertex_radius[:, None]
        sphere_lower = center_min[None, :] - radius
        sphere_upper = center_max[None, :] + radius
        ball = self.midpoint_bound(lower_time_, upper_time_)
        ball_lower = ball.positions - ball.displacement[:, None]
        ball_upper = ball.positions + ball.displacement[:, None]
        return jnp.maximum(sphere_lower, ball_lower), jnp.minimum(
            sphere_upper, ball_upper
        )


__all__ = [
    "AbstractContactTrajectory",
    "ContactPathEnclosure",
    "ContactPathMidpointBound",
    "ContactTrajectoryEvaluation",
    "CubicHermiteContactTrajectory",
    "LinearContactTrajectory",
    "RigidSweepContactTrajectory",
]
