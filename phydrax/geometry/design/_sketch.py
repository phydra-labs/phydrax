#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Literal, TypeAlias
from uuid import uuid4

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule


@jax.custom_jvp
def _finite_norm(value: Array) -> Array:
    return jnp.sqrt(jnp.sum(value * value, axis=-1))


@_finite_norm.defjvp
def _finite_norm_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    norm = _finite_norm(value)
    derivative = jnp.sum(value * tangent, axis=-1) / jnp.where(norm > 0.0, norm, 1.0)
    return norm, jnp.where(norm > 0.0, derivative, 0.0)


def _cross_2d(first: Array, second: Array) -> Array:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _line_direction(points: Array, lines: Array, index: int) -> Array:
    endpoints = points[lines[index]]
    return endpoints[1] - endpoints[0]


class AbstractSketchConstraint(StrictModule):
    """A dimensionless or physically scaled residual over sketch variables."""

    weight: Array

    def __init__(self, weight: float = 1.0):
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("constraint weight must be finite and positive.")
        self.weight = jnp.asarray(weight, dtype=float).reshape(())

    @abstractmethod
    def residual(
        self,
        points: Array,
        lines: Array,
        circle_centers: Array,
        circle_radii: Array,
        /,
    ) -> Array:
        raise NotImplementedError

    def _weighted(self, residual: Array) -> Array:
        return jnp.sqrt(self.weight) * jnp.atleast_1d(residual)


class Coincident(AbstractSketchConstraint):
    first_point: int = eqx.field(static=True)
    second_point: int = eqx.field(static=True)

    def __init__(self, first_point: int, second_point: int, *, weight: float = 1.0):
        super().__init__(weight)
        self.first_point = int(first_point)
        self.second_point = int(second_point)

    def residual(self, points, lines, circle_centers, circle_radii, /):
        del lines, circle_centers, circle_radii
        return self._weighted(points[self.first_point] - points[self.second_point])


class FixedPoint(AbstractSketchConstraint):
    point: int = eqx.field(static=True)
    target: Array

    def __init__(self, point: int, target: Array, *, weight: float = 1.0):
        super().__init__(weight)
        target_ = jnp.asarray(target, dtype=float)
        if target_.shape != (2,):
            raise ValueError("FixedPoint target must have shape (2,).")
        self.point = int(point)
        self.target = target_

    def residual(self, points, lines, circle_centers, circle_radii, /):
        del lines, circle_centers, circle_radii
        return self._weighted(points[self.point] - self.target)


class PointDistance(AbstractSketchConstraint):
    first_point: int = eqx.field(static=True)
    second_point: int = eqx.field(static=True)
    distance: Array

    def __init__(
        self,
        first_point: int,
        second_point: int,
        distance: float,
        *,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        if not np.isfinite(distance) or distance < 0.0:
            raise ValueError("distance must be finite and non-negative.")
        self.first_point = int(first_point)
        self.second_point = int(second_point)
        self.distance = jnp.asarray(distance, dtype=float).reshape(())

    def residual(self, points, lines, circle_centers, circle_radii, /):
        del lines, circle_centers, circle_radii
        value = _finite_norm(points[self.second_point] - points[self.first_point])
        return self._weighted(value - self.distance)


class Horizontal(AbstractSketchConstraint):
    line: int = eqx.field(static=True)

    def __init__(self, line: int, *, weight: float = 1.0):
        super().__init__(weight)
        self.line = int(line)

    def residual(self, points, lines, circle_centers, circle_radii, /):
        del circle_centers, circle_radii
        return self._weighted(_line_direction(points, lines, self.line)[1])


class Vertical(AbstractSketchConstraint):
    line: int = eqx.field(static=True)

    def __init__(self, line: int, *, weight: float = 1.0):
        super().__init__(weight)
        self.line = int(line)

    def residual(self, points, lines, circle_centers, circle_radii, /):
        del circle_centers, circle_radii
        return self._weighted(_line_direction(points, lines, self.line)[0])


class Parallel(AbstractSketchConstraint):
    first_line: int = eqx.field(static=True)
    second_line: int = eqx.field(static=True)

    def __init__(self, first_line: int, second_line: int, *, weight: float = 1.0):
        super().__init__(weight)
        self.first_line = int(first_line)
        self.second_line = int(second_line)

    def residual(self, points, lines, circle_centers, circle_radii, /):
        del circle_centers, circle_radii
        first = _line_direction(points, lines, self.first_line)
        second = _line_direction(points, lines, self.second_line)
        scale = _finite_norm(first) * _finite_norm(second)
        return self._weighted(_cross_2d(first, second) / jnp.maximum(scale, 1e-15))


class Perpendicular(AbstractSketchConstraint):
    first_line: int = eqx.field(static=True)
    second_line: int = eqx.field(static=True)

    def __init__(self, first_line: int, second_line: int, *, weight: float = 1.0):
        super().__init__(weight)
        self.first_line = int(first_line)
        self.second_line = int(second_line)

    def residual(self, points, lines, circle_centers, circle_radii, /):
        del circle_centers, circle_radii
        first = _line_direction(points, lines, self.first_line)
        second = _line_direction(points, lines, self.second_line)
        scale = _finite_norm(first) * _finite_norm(second)
        return self._weighted(jnp.dot(first, second) / jnp.maximum(scale, 1e-15))


class EqualLength(AbstractSketchConstraint):
    first_line: int = eqx.field(static=True)
    second_line: int = eqx.field(static=True)

    def __init__(self, first_line: int, second_line: int, *, weight: float = 1.0):
        super().__init__(weight)
        self.first_line = int(first_line)
        self.second_line = int(second_line)

    def residual(self, points, lines, circle_centers, circle_radii, /):
        del circle_centers, circle_radii
        first = _finite_norm(_line_direction(points, lines, self.first_line))
        second = _finite_norm(_line_direction(points, lines, self.second_line))
        return self._weighted(first - second)


class LineAngle(AbstractSketchConstraint):
    first_line: int = eqx.field(static=True)
    second_line: int = eqx.field(static=True)
    angle: Array

    def __init__(
        self,
        first_line: int,
        second_line: int,
        angle: float,
        *,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        if not np.isfinite(angle):
            raise ValueError("angle must be finite.")
        self.first_line = int(first_line)
        self.second_line = int(second_line)
        self.angle = jnp.asarray(angle, dtype=float).reshape(())

    def residual(self, points, lines, circle_centers, circle_radii, /):
        del circle_centers, circle_radii
        first = _line_direction(points, lines, self.first_line)
        second = _line_direction(points, lines, self.second_line)
        angle = jnp.arctan2(_cross_2d(first, second), jnp.dot(first, second))
        difference = angle - self.angle
        return self._weighted(jnp.arctan2(jnp.sin(difference), jnp.cos(difference)))


class Midpoint(AbstractSketchConstraint):
    point: int = eqx.field(static=True)
    line: int = eqx.field(static=True)

    def __init__(self, point: int, line: int, *, weight: float = 1.0):
        super().__init__(weight)
        self.point = int(point)
        self.line = int(line)

    def residual(self, points, lines, circle_centers, circle_radii, /):
        del circle_centers, circle_radii
        endpoints = points[lines[self.line]]
        return self._weighted(points[self.point] - 0.5 * jnp.sum(endpoints, axis=0))


class PointOnLine(AbstractSketchConstraint):
    point: int = eqx.field(static=True)
    line: int = eqx.field(static=True)

    def __init__(self, point: int, line: int, *, weight: float = 1.0):
        super().__init__(weight)
        self.point = int(point)
        self.line = int(line)

    def residual(self, points, lines, circle_centers, circle_radii, /):
        del circle_centers, circle_radii
        endpoints = points[lines[self.line]]
        direction = endpoints[1] - endpoints[0]
        residual = _cross_2d(direction, points[self.point] - endpoints[0])
        return self._weighted(residual / jnp.maximum(_finite_norm(direction), 1e-15))


class Radius(AbstractSketchConstraint):
    circle: int = eqx.field(static=True)
    radius: Array

    def __init__(self, circle: int, radius: float, *, weight: float = 1.0):
        super().__init__(weight)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be finite and positive.")
        self.circle = int(circle)
        self.radius = jnp.asarray(radius, dtype=float).reshape(())

    def residual(self, points, lines, circle_centers, circle_radii, /):
        del points, lines, circle_centers
        return self._weighted(circle_radii[self.circle] - self.radius)


class TangentLineCircle(AbstractSketchConstraint):
    line: int = eqx.field(static=True)
    circle: int = eqx.field(static=True)
    side: int = eqx.field(static=True)

    def __init__(
        self,
        line: int,
        circle: int,
        *,
        side: Literal[-1, 1] = 1,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        if side not in (-1, 1):
            raise ValueError("side must be -1 or +1.")
        self.line = int(line)
        self.circle = int(circle)
        self.side = int(side)

    def residual(self, points, lines, circle_centers, circle_radii, /):
        endpoints = points[lines[self.line]]
        direction = endpoints[1] - endpoints[0]
        center = points[circle_centers[self.circle]]
        signed_distance = _cross_2d(direction, center - endpoints[0]) / jnp.maximum(
            _finite_norm(direction), 1e-15
        )
        return self._weighted(signed_distance - self.side * circle_radii[self.circle])


class TangentCircles(AbstractSketchConstraint):
    first_circle: int = eqx.field(static=True)
    second_circle: int = eqx.field(static=True)
    internal: bool = eqx.field(static=True)

    def __init__(
        self,
        first_circle: int,
        second_circle: int,
        *,
        internal: bool = False,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        self.first_circle = int(first_circle)
        self.second_circle = int(second_circle)
        self.internal = bool(internal)

    def residual(self, points, lines, circle_centers, circle_radii, /):
        del lines
        first_center = points[circle_centers[self.first_circle]]
        second_center = points[circle_centers[self.second_circle]]
        center_distance = _finite_norm(second_center - first_center)
        if self.internal:
            target = jnp.abs(
                circle_radii[self.first_circle] - circle_radii[self.second_circle]
            )
        else:
            target = circle_radii[self.first_circle] + circle_radii[self.second_circle]
        return self._weighted(center_distance - target)


SketchConstraint: TypeAlias = (
    Coincident
    | FixedPoint
    | PointDistance
    | Horizontal
    | Vertical
    | Parallel
    | Perpendicular
    | EqualLength
    | LineAngle
    | Midpoint
    | PointOnLine
    | Radius
    | TangentLineCircle
    | TangentCircles
)


class SketchSolution(StrictModule):
    points: Array
    circle_radii: Array
    residual: Array
    residual_norm: Array
    converged: Array
    iterations: Array

    def __init__(
        self,
        *,
        points,
        circle_radii,
        residual,
        residual_norm,
        converged,
        iterations,
    ):
        self.points = jnp.asarray(points, dtype=float)
        self.circle_radii = jnp.asarray(circle_radii, dtype=float)
        self.residual = jnp.asarray(residual, dtype=float)
        self.residual_norm = jnp.asarray(residual_norm, dtype=float).reshape(())
        self.converged = jnp.asarray(converged, dtype=bool).reshape(())
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())


class Sketch(StrictModule):
    """A pure-JAX 2D line/circle sketch with declarative geometric constraints."""

    points: Array
    lines: Array
    circle_centers: Array
    circle_radii: Array
    constraints: tuple[AbstractSketchConstraint, ...]
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        points: Array,
        *,
        lines: Array | None = None,
        circle_centers: Array | None = None,
        circle_radii: Array | None = None,
        constraints: Sequence[AbstractSketchConstraint] = (),
        feature_id: str | None = None,
    ):
        points_host = np.asarray(points, dtype=float)
        lines_host = (
            np.empty((0, 2), dtype=np.int32)
            if lines is None
            else np.asarray(lines, dtype=np.int32)
        )
        centers_host = (
            np.empty((0,), dtype=np.int32)
            if circle_centers is None
            else np.asarray(circle_centers, dtype=np.int32).reshape((-1,))
        )
        radii_host = (
            np.empty((0,), dtype=float)
            if circle_radii is None
            else np.asarray(circle_radii, dtype=float).reshape((-1,))
        )
        if points_host.ndim != 2 or points_host.shape[1] != 2:
            raise ValueError("points must have shape (num_points, 2).")
        if lines_host.ndim != 2 or lines_host.shape[1] != 2:
            raise ValueError("lines must have shape (num_lines, 2).")
        if np.any(lines_host < 0) or np.any(lines_host >= points_host.shape[0]):
            raise ValueError("lines reference an absent point.")
        if np.any(lines_host[:, 0] == lines_host[:, 1]):
            raise ValueError("Every line must have distinct endpoints.")
        if centers_host.shape != radii_host.shape:
            raise ValueError("circle_centers and circle_radii must align.")
        if np.any(centers_host < 0) or np.any(centers_host >= points_host.shape[0]):
            raise ValueError("circle_centers reference an absent point.")
        if np.any(~np.isfinite(radii_host)) or np.any(radii_host <= 0.0):
            raise ValueError("circle_radii must be finite and positive.")
        constraints_ = tuple(constraints)
        if any(not isinstance(item, AbstractSketchConstraint) for item in constraints_):
            raise TypeError("constraints must contain sketch constraint objects.")
        self._validate_constraint_indices(
            constraints_, points_host.shape[0], lines_host.shape[0], radii_host.shape[0]
        )
        self.points = jnp.asarray(points_host, dtype=float)
        self.lines = jnp.asarray(lines_host, dtype=jnp.int32)
        self.circle_centers = jnp.asarray(centers_host, dtype=jnp.int32)
        self.circle_radii = jnp.asarray(radii_host, dtype=float)
        self.constraints = constraints_
        self.feature_id = feature_id or f"sketch-{uuid4().hex}"

    @staticmethod
    def _validate_constraint_indices(constraints, num_points, num_lines, num_circles):
        def require(indices, size, kind):
            if any(index < 0 or index >= size for index in indices):
                raise ValueError(f"Constraint references an absent {kind}.")

        for constraint in constraints:
            point_indices: tuple[int, ...] = ()
            line_indices: tuple[int, ...] = ()
            circle_indices: tuple[int, ...] = ()
            if isinstance(constraint, (Coincident, PointDistance)):
                point_indices = (constraint.first_point, constraint.second_point)
            elif isinstance(constraint, FixedPoint):
                point_indices = (constraint.point,)
            elif isinstance(constraint, (Midpoint, PointOnLine)):
                point_indices = (constraint.point,)
                line_indices = (constraint.line,)
            elif isinstance(constraint, (Horizontal, Vertical)):
                line_indices = (constraint.line,)
            elif isinstance(
                constraint, (Parallel, Perpendicular, EqualLength, LineAngle)
            ):
                line_indices = (constraint.first_line, constraint.second_line)
            elif isinstance(constraint, Radius):
                circle_indices = (constraint.circle,)
            elif isinstance(constraint, TangentLineCircle):
                line_indices = (constraint.line,)
                circle_indices = (constraint.circle,)
            elif isinstance(constraint, TangentCircles):
                circle_indices = (
                    constraint.first_circle,
                    constraint.second_circle,
                )
            require(point_indices, num_points, "point")
            require(line_indices, num_lines, "line")
            require(circle_indices, num_circles, "circle")

    def residual(self, points: Array, circle_radii: Array, /) -> Array:
        points_ = jnp.asarray(points, dtype=self.points.dtype).reshape(self.points.shape)
        radii_ = jnp.asarray(circle_radii, dtype=self.circle_radii.dtype).reshape(
            self.circle_radii.shape
        )
        values = tuple(
            constraint.residual(points_, self.lines, self.circle_centers, radii_).reshape(
                (-1,)
            )
            for constraint in self.constraints
        )
        if not values:
            return jnp.empty((0,), dtype=self.points.dtype)
        return jnp.concatenate(values)

    def solve(
        self,
        *,
        max_iterations: int = 32,
        tolerance: float = 1e-9,
        damping: float = 1e-8,
    ) -> SketchSolution:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if tolerance <= 0.0 or damping <= 0.0:
            raise ValueError("tolerance and damping must be positive.")
        point_size = self.points.size
        initial = jnp.concatenate((self.points.reshape((-1,)), self.circle_radii))

        def unpack(vector):
            points = vector[:point_size].reshape(self.points.shape)
            radii = vector[point_size:]
            return points, radii

        def residual(vector):
            return self.residual(*unpack(vector))

        initial_norm = jnp.linalg.norm(residual(initial))
        loop_state = (
            initial,
            initial_norm <= tolerance,
            jnp.asarray(0, dtype=jnp.int32),
        )

        def iteration(_, state):
            vector, converged, count = state

            def update(current):
                values = residual(current)
                jacobian = jax.jacfwd(residual)(current)
                normal = jacobian.T @ jacobian + damping * jnp.eye(
                    current.shape[0], dtype=current.dtype
                )
                step = jnp.linalg.solve(normal, -(jacobian.T @ values))
                candidate = current + step
                candidate = candidate.at[point_size:].set(
                    jnp.maximum(candidate[point_size:], jnp.finfo(current.dtype).eps)
                )
                return candidate

            candidate = jax.lax.cond(converged, lambda value: value, update, vector)
            candidate_norm = jnp.linalg.norm(residual(candidate))
            active = ~converged
            return (
                candidate,
                converged | (candidate_norm <= tolerance),
                count + active.astype(jnp.int32),
            )

        vector, converged, iterations = jax.lax.fori_loop(
            0, max_iterations, iteration, loop_state
        )
        points, radii = unpack(vector)
        residual_value = residual(vector)
        residual_norm = jnp.linalg.norm(residual_value)
        return SketchSolution(
            points=points,
            circle_radii=radii,
            residual=residual_value,
            residual_norm=residual_norm,
            converged=converged,
            iterations=iterations,
        )

    def to_source(self, solution: SketchSolution | None = None):
        """Lower one closed line loop or one circle to a geometry source."""

        points = np.asarray(self.points if solution is None else solution.points)
        radii = np.asarray(
            self.circle_radii if solution is None else solution.circle_radii
        )
        if self.lines.shape[0] == 0 and self.circle_centers.shape[0] == 1:
            from ..analytic import Circle

            center = points[int(np.asarray(self.circle_centers)[0])]
            return Circle(center, float(radii[0]), feature_id=self.feature_id)
        if self.circle_centers.shape[0] != 0:
            raise ValueError("Mixed line/circle profile lowering is not yet defined.")
        lines = np.asarray(self.lines, dtype=np.int32)
        if lines.shape[0] < 3:
            raise ValueError("A planar region sketch requires at least three lines.")
        adjacency: dict[int, list[int]] = {}
        for start, end in lines.tolist():
            adjacency.setdefault(start, []).append(end)
            adjacency.setdefault(end, []).append(start)
        if len(adjacency) != lines.shape[0] or any(
            len(neighbors) != 2 for neighbors in adjacency.values()
        ):
            raise ValueError("Line profiles must form one simple closed cycle.")
        start = min(adjacency)
        order = [start]
        previous = -1
        current = start
        while len(order) < len(adjacency):
            neighbors = adjacency[current]
            following = neighbors[0] if neighbors[0] != previous else neighbors[1]
            if following == start:
                break
            order.append(following)
            previous, current = current, following
        if len(order) != len(adjacency) or start not in adjacency[current]:
            raise ValueError("Line profiles must form one connected cycle.")
        polygon = points[np.asarray(order, dtype=np.int32)]
        signed_area = 0.5 * np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - np.roll(polygon[:, 0], -1) * polygon[:, 1]
        )
        if signed_area < 0.0:
            polygon = polygon[::-1]
        loop = np.arange(polygon.shape[0], dtype=np.int32)
        from ..simplicial import PlanarMeshRegion

        return PlanarMeshRegion(polygon, (loop,), feature_id=self.feature_id)


__all__ = [
    "AbstractSketchConstraint",
    "Coincident",
    "EqualLength",
    "FixedPoint",
    "Horizontal",
    "LineAngle",
    "Midpoint",
    "Parallel",
    "Perpendicular",
    "PointDistance",
    "PointOnLine",
    "Radius",
    "Sketch",
    "SketchConstraint",
    "SketchSolution",
    "TangentCircles",
    "TangentLineCircle",
    "Vertical",
]
