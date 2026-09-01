#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

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


class AbstractContactTrajectory(StrictModule, NonTrainableState):
    @property
    @abc.abstractmethod
    def trajectory_id(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, time: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def bounds(
        self, lower_time: ArrayLike, upper_time: ArrayLike, /
    ) -> tuple[Array, Array]:
        raise NotImplementedError

    def evaluate_with_bounds(
        self,
        time: ArrayLike,
        /,
        *,
        lower_time: ArrayLike = 0.0,
        upper_time: ArrayLike = 1.0,
        guarantee_level: ContactGuaranteeLevel,
    ) -> ContactTrajectoryEvaluation:
        positions = self.evaluate(time)
        lower, upper = self.bounds(lower_time, upper_time)
        finite = (
            jnp.all(jnp.isfinite(positions))
            & jnp.all(jnp.isfinite(lower))
            & jnp.all(jnp.isfinite(upper))
            & jnp.all(lower <= upper)
        )
        return ContactTrajectoryEvaluation(
            positions,
            lower,
            upper,
            jnp.asarray(int(guarantee_level), dtype=jnp.int32),
            finite,
            finite,
            self.trajectory_id,
        )


class LinearContactTrajectory(AbstractContactTrajectory):
    start: Array
    end: Array
    _trajectory_id: str = eqx.field(static=True)

    def __init__(self, start: ArrayLike, end: ArrayLike, /):
        start_ = jnp.asarray(start)
        end_ = jnp.asarray(end, dtype=start_.dtype)
        if start_.shape != end_.shape or start_.ndim != 2:
            raise ValueError(
                "Linear contact trajectory endpoints must be matching matrices."
            )
        self.start = start_
        self.end = end_
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


class CubicHermiteContactTrajectory(AbstractContactTrajectory):
    """Cubic trajectory with exact Bernstein convex-hull bounds."""

    start: Array
    start_tangent: Array
    end: Array
    end_tangent: Array
    controls: Array
    _trajectory_id: str = eqx.field(static=True)

    def __init__(
        self,
        start: ArrayLike,
        start_tangent: ArrayLike,
        end: ArrayLike,
        end_tangent: ArrayLike,
        /,
    ):
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

    def evaluate(self, time: ArrayLike, /) -> Array:
        time_ = jnp.asarray(time, dtype=self.start.dtype)
        one_minus = 1.0 - time_
        weights = jnp.stack(
            (
                one_minus**3,
                3.0 * one_minus * one_minus * time_,
                3.0 * one_minus * time_ * time_,
                time_**3,
            )
        )
        return jnp.sum(weights[:, None, None] * self.controls, axis=0)

    def bounds(
        self, lower_time: ArrayLike, upper_time: ArrayLike, /
    ) -> tuple[Array, Array]:
        lower_time_ = jnp.asarray(lower_time, dtype=self.start.dtype)
        upper_time_ = jnp.asarray(upper_time, dtype=self.start.dtype)
        if bool((lower_time_ == 0.0) & (upper_time_ == 1.0)):
            controls = self.controls
        else:
            sample_times = jnp.linspace(lower_time_, upper_time_, 5)
            samples = jnp.stack(tuple(self.evaluate(value) for value in sample_times))
            speed_bound = jnp.maximum(
                jnp.sqrt(jnp.sum(self.start_tangent**2, axis=-1)),
                jnp.sqrt(jnp.sum(self.end_tangent**2, axis=-1)),
            )
            inflation = (upper_time_ - lower_time_) * speed_bound[:, None] / 4.0
            return jnp.min(samples, axis=0) - inflation, jnp.max(
                samples, axis=0
            ) + inflation
        return jnp.min(controls, axis=0), jnp.max(controls, axis=0)


class RigidSweepContactTrajectory(AbstractContactTrajectory):
    """Rigid vertex motion with a rotation-independent spherical enclosure."""

    local_vertices: Array
    start_center: Array
    end_center: Array
    start_rotation: Array
    end_rotation: Array
    radius: Array
    _trajectory_id: str = eqx.field(static=True)

    def __init__(
        self,
        local_vertices: ArrayLike,
        start_center: ArrayLike,
        end_center: ArrayLike,
        start_rotation: ArrayLike,
        end_rotation: ArrayLike,
        /,
    ):
        vertices = jnp.asarray(local_vertices)
        start_center_ = jnp.asarray(start_center, dtype=vertices.dtype)
        end_center_ = jnp.asarray(end_center, dtype=vertices.dtype)
        start_rotation_ = jnp.asarray(start_rotation, dtype=vertices.dtype)
        end_rotation_ = jnp.asarray(end_rotation, dtype=vertices.dtype)
        dimension = vertices.shape[-1]
        if vertices.ndim != 2 or dimension not in (2, 3):
            raise ValueError("Rigid contact vertices require dimension two or three.")
        if start_center_.shape != (dimension,) or end_center_.shape != (dimension,):
            raise ValueError("Rigid contact centers have invalid shape.")
        expected_rotation = (dimension, dimension)
        if (
            start_rotation_.shape != expected_rotation
            or end_rotation_.shape != expected_rotation
        ):
            raise ValueError("Rigid contact rotation matrices have invalid shape.")
        radius = jnp.max(jnp.sqrt(jnp.sum(vertices * vertices, axis=-1)), initial=0.0)
        self.local_vertices = vertices
        self.start_center = start_center_
        self.end_center = end_center_
        self.start_rotation = start_rotation_
        self.end_rotation = end_rotation_
        self.radius = radius
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

    def evaluate(self, time: ArrayLike, /) -> Array:
        time_ = jnp.asarray(time, dtype=self.local_vertices.dtype)
        center = self.start_center + time_ * (self.end_center - self.start_center)
        rotation = (1.0 - time_) * self.start_rotation + time_ * self.end_rotation
        first_axis = rotation[:, 0]
        first_axis = first_axis / jnp.maximum(
            jnp.sqrt(jnp.sum(first_axis * first_axis)),
            jnp.finfo(rotation.dtype).eps,
        )
        if self.local_vertices.shape[1] == 2:
            second_axis = jnp.stack((-first_axis[1], first_axis[0]))
            projected = jnp.stack((first_axis, second_axis), axis=1)
        else:
            second_axis = (
                rotation[:, 1] - jnp.sum(rotation[:, 1] * first_axis) * first_axis
            )
            second_axis = second_axis / jnp.maximum(
                jnp.sqrt(jnp.sum(second_axis * second_axis)),
                jnp.finfo(rotation.dtype).eps,
            )
            third_axis = jnp.cross(first_axis, second_axis)
            projected = jnp.stack((first_axis, second_axis, third_axis), axis=1)
        return self.local_vertices @ projected.T + center

    def bounds(
        self, lower_time: ArrayLike, upper_time: ArrayLike, /
    ) -> tuple[Array, Array]:
        lower_center = self.start_center + jnp.asarray(
            lower_time, dtype=self.start_center.dtype
        ) * (self.end_center - self.start_center)
        upper_center = self.start_center + jnp.asarray(
            upper_time, dtype=self.start_center.dtype
        ) * (self.end_center - self.start_center)
        center_min = jnp.minimum(lower_center, upper_center)
        center_max = jnp.maximum(lower_center, upper_center)
        lower = jnp.broadcast_to(center_min - self.radius, self.local_vertices.shape)
        upper = jnp.broadcast_to(center_max + self.radius, self.local_vertices.shape)
        return lower, upper


__all__ = [
    "AbstractContactTrajectory",
    "ContactTrajectoryEvaluation",
    "CubicHermiteContactTrajectory",
    "LinearContactTrajectory",
    "RigidSweepContactTrajectory",
]
