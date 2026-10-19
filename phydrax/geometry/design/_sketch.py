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
from jax import Array

import phydrax.geometry.analytic as analytic_owner
import phydrax.geometry.brep._constructors as constructors_owner
import phydrax.geometry.simplicial as simplicial_owner

from ..._fingerprint import tree_content_equal
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._validation import positive_integer, positive_real_scalar
from ...nonlinear import SensitivityEvidence
from ...optim import LeastSquaresResult
from ...typing import checked, ConvertibleToArray, parse
from ..brep._constructors import BRepTessellationPolicy, ProfilePlane
from ..brep._model import BRepModel
from ._solve import attach_sketch_sensitivity, DesignTrialPolicy, solve_design_residual


SketchTangentSide: TypeAlias = Literal[-1, 1]


@jax.custom_jvp
def _finite_norm(value: Array) -> Array:
    return jnp.sqrt(jnp.sum(value * value, axis=-1))


@_finite_norm.defjvp
def _finite_norm_jvp(
    primals: tuple[Array], tangents: tuple[Array]
) -> tuple[Array, Array]:
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

    def __init__(self, weight: float = 1.0) -> None:
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"AbstractSketchConstraint.weight must be finite and positive; got {weight}"
            )
        self.weight = jnp.asarray(weight, dtype=jnp.float64).reshape(())

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

    def __init__(
        self, first_point: int, second_point: int, *, weight: float = 1.0
    ) -> None:
        super().__init__(weight)
        self.first_point = int(first_point)
        self.second_point = int(second_point)

    def residual(
        self,
        points: Array,
        lines: Array,
        circle_centers: Array,
        circle_radii: Array,
        /,
    ) -> Array:
        del lines, circle_centers, circle_radii
        return self._weighted(points[self.first_point] - points[self.second_point])


class FixedPoint(AbstractSketchConstraint):
    point: int = eqx.field(static=True)
    target: Array

    def __init__(
        self, point: int, target: ConvertibleToArray, *, weight: float = 1.0
    ) -> None:
        super().__init__(weight)
        target_ = jnp.asarray(target, dtype=jnp.float64)
        if target_.shape != (2,):
            raise ValueError(
                f"FixedPoint.target must have shape (2,); got {target_.shape}"
            )
        self.point = int(point)
        self.target = target_

    def residual(
        self,
        points: Array,
        lines: Array,
        circle_centers: Array,
        circle_radii: Array,
        /,
    ) -> Array:
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
    ) -> None:
        super().__init__(weight)
        if not np.isfinite(distance) or distance < 0.0:
            raise ValueError(
                f"PointDistance.distance must be finite and nonnegative; got {distance}"
            )
        self.first_point = int(first_point)
        self.second_point = int(second_point)
        self.distance = jnp.asarray(distance, dtype=jnp.float64).reshape(())

    def residual(
        self,
        points: Array,
        lines: Array,
        circle_centers: Array,
        circle_radii: Array,
        /,
    ) -> Array:
        del lines, circle_centers, circle_radii
        value = _finite_norm(points[self.second_point] - points[self.first_point])
        return self._weighted(value - self.distance)


class Horizontal(AbstractSketchConstraint):
    line: int = eqx.field(static=True)

    def __init__(self, line: int, *, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.line = int(line)

    def residual(
        self,
        points: Array,
        lines: Array,
        circle_centers: Array,
        circle_radii: Array,
        /,
    ) -> Array:
        del circle_centers, circle_radii
        return self._weighted(_line_direction(points, lines, self.line)[1])


class Vertical(AbstractSketchConstraint):
    line: int = eqx.field(static=True)

    def __init__(self, line: int, *, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.line = int(line)

    def residual(
        self,
        points: Array,
        lines: Array,
        circle_centers: Array,
        circle_radii: Array,
        /,
    ) -> Array:
        del circle_centers, circle_radii
        return self._weighted(_line_direction(points, lines, self.line)[0])


class Parallel(AbstractSketchConstraint):
    first_line: int = eqx.field(static=True)
    second_line: int = eqx.field(static=True)

    def __init__(self, first_line: int, second_line: int, *, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.first_line = int(first_line)
        self.second_line = int(second_line)

    def residual(
        self,
        points: Array,
        lines: Array,
        circle_centers: Array,
        circle_radii: Array,
        /,
    ) -> Array:
        del circle_centers, circle_radii
        first = _line_direction(points, lines, self.first_line)
        second = _line_direction(points, lines, self.second_line)
        scale = _finite_norm(first) * _finite_norm(second)
        return self._weighted(_cross_2d(first, second) / jnp.maximum(scale, 1e-15))


class Perpendicular(AbstractSketchConstraint):
    first_line: int = eqx.field(static=True)
    second_line: int = eqx.field(static=True)

    def __init__(self, first_line: int, second_line: int, *, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.first_line = int(first_line)
        self.second_line = int(second_line)

    def residual(
        self,
        points: Array,
        lines: Array,
        circle_centers: Array,
        circle_radii: Array,
        /,
    ) -> Array:
        del circle_centers, circle_radii
        first = _line_direction(points, lines, self.first_line)
        second = _line_direction(points, lines, self.second_line)
        scale = _finite_norm(first) * _finite_norm(second)
        return self._weighted(jnp.dot(first, second) / jnp.maximum(scale, 1e-15))


class EqualLength(AbstractSketchConstraint):
    first_line: int = eqx.field(static=True)
    second_line: int = eqx.field(static=True)

    def __init__(self, first_line: int, second_line: int, *, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.first_line = int(first_line)
        self.second_line = int(second_line)

    def residual(
        self,
        points: Array,
        lines: Array,
        circle_centers: Array,
        circle_radii: Array,
        /,
    ) -> Array:
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
    ) -> None:
        super().__init__(weight)
        if not np.isfinite(angle):
            raise ValueError(f"LineAngle.angle must be finite; got {angle}")
        self.first_line = int(first_line)
        self.second_line = int(second_line)
        self.angle = jnp.asarray(angle, dtype=jnp.float64).reshape(())

    def residual(
        self,
        points: Array,
        lines: Array,
        circle_centers: Array,
        circle_radii: Array,
        /,
    ) -> Array:
        del circle_centers, circle_radii
        first = _line_direction(points, lines, self.first_line)
        second = _line_direction(points, lines, self.second_line)
        angle = jnp.arctan2(_cross_2d(first, second), jnp.dot(first, second))
        difference = angle - self.angle
        return self._weighted(jnp.arctan2(jnp.sin(difference), jnp.cos(difference)))


class Midpoint(AbstractSketchConstraint):
    point: int = eqx.field(static=True)
    line: int = eqx.field(static=True)

    def __init__(self, point: int, line: int, *, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.point = int(point)
        self.line = int(line)

    def residual(
        self,
        points: Array,
        lines: Array,
        circle_centers: Array,
        circle_radii: Array,
        /,
    ) -> Array:
        del circle_centers, circle_radii
        endpoints = points[lines[self.line]]
        return self._weighted(points[self.point] - 0.5 * jnp.sum(endpoints, axis=0))


class PointOnLine(AbstractSketchConstraint):
    point: int = eqx.field(static=True)
    line: int = eqx.field(static=True)

    def __init__(self, point: int, line: int, *, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.point = int(point)
        self.line = int(line)

    def residual(
        self,
        points: Array,
        lines: Array,
        circle_centers: Array,
        circle_radii: Array,
        /,
    ) -> Array:
        del circle_centers, circle_radii
        endpoints = points[lines[self.line]]
        direction = endpoints[1] - endpoints[0]
        residual = _cross_2d(direction, points[self.point] - endpoints[0])
        return self._weighted(residual / jnp.maximum(_finite_norm(direction), 1e-15))


class Radius(AbstractSketchConstraint):
    circle: int = eqx.field(static=True)
    radius: Array

    def __init__(self, circle: int, radius: float, *, weight: float = 1.0) -> None:
        super().__init__(weight)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"Radius.radius must be finite and positive; got {radius}")
        self.circle = int(circle)
        self.radius = jnp.asarray(radius, dtype=jnp.float64).reshape(())

    def residual(
        self,
        points: Array,
        lines: Array,
        circle_centers: Array,
        circle_radii: Array,
        /,
    ) -> Array:
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
        side: SketchTangentSide = 1,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        side = parse(side, SketchTangentSide, "TangentLineCircle.side")
        self.line = int(line)
        self.circle = int(circle)
        self.side = int(side)

    def residual(
        self,
        points: Array,
        lines: Array,
        circle_centers: Array,
        circle_radii: Array,
        /,
    ) -> Array:
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
    ) -> None:
        super().__init__(weight)
        self.first_circle = int(first_circle)
        self.second_circle = int(second_circle)
        self.internal = bool(internal)

    def residual(
        self,
        points: Array,
        lines: Array,
        circle_centers: Array,
        circle_radii: Array,
        /,
    ) -> Array:
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


def _constraint_metadata(
    count: int,
    constraint_ids: Sequence[str] | None,
    constraint_priorities: Sequence[int] | None,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Own the static identity/priority alignment of the authored constraint sequence."""
    identities = (
        tuple(f"constraint:{index}" for index in range(count))
        if constraint_ids is None
        else tuple(constraint_ids)
    )
    priorities = (
        (0,) * count if constraint_priorities is None else tuple(constraint_priorities)
    )
    if (
        len(identities) != count
        or any(
            not isinstance(identity, str) or not identity or identity.startswith("arc:")
            for identity in identities
        )
        or len(set(identities)) != len(identities)
    ):
        raise ValueError(
            f"Sketch.constraint_ids must be unique nonempty names aligned with constraints outside reserved arc: namespace; got {identities!r} for {count} constraints"
        )
    if len(priorities) != count or any(
        isinstance(priority, bool) or not isinstance(priority, int) or priority < 0
        for priority in priorities
    ):
        raise ValueError(
            f"Sketch.constraint_priorities must be nonnegative integer tiers aligned with constraints; got {priorities!r} for {count} constraints"
        )
    return identities, priorities


class _SketchSnapshot(StrictModule):
    points: np.ndarray
    circle_radii: np.ndarray
    lines: np.ndarray
    arcs: np.ndarray
    circle_centers: np.ndarray
    residual_bound: float = eqx.field(static=True)


class Sketch(StrictModule):
    """A pure-JAX 2D line/arc/circle sketch with declarative geometric constraints.

    ``arcs`` rows ``(circle, start, end)`` are counterclockwise arcs of a sketch
    circle from point ``start`` to point ``end``; incidence of both end points
    on that circle is an intrinsic residual. Circles carrying an arc are its
    construction circles; every other circle is a closed profile curve.
    """

    points: Array
    lines: Array
    circle_centers: Array
    circle_radii: Array
    arcs: Array
    constraints: tuple[AbstractSketchConstraint, ...]
    feature_id: str = eqx.field(static=True)
    constraint_ids: tuple[str, ...] = eqx.field(static=True)
    constraint_priorities: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        points: ConvertibleToArray,
        *,
        lines: ConvertibleToArray | None = None,
        circle_centers: ConvertibleToArray | None = None,
        circle_radii: ConvertibleToArray | None = None,
        arcs: ConvertibleToArray | None = None,
        constraints: Sequence[AbstractSketchConstraint] = (),
        feature_id: str | None = None,
        constraint_ids: Sequence[str] | None = None,
        constraint_priorities: Sequence[int] | None = None,
    ) -> None:
        points, lines, circle_centers, circle_radii, arcs = jax.device_get(
            (points, lines, circle_centers, circle_radii, arcs)
        )
        points_host = np.asarray(points, dtype=np.float64)
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
            np.empty((0,), dtype=np.float64)
            if circle_radii is None
            else np.asarray(circle_radii, dtype=np.float64).reshape((-1,))
        )
        arcs_host = (
            np.empty((0, 3), dtype=np.int32)
            if arcs is None
            else np.asarray(arcs, dtype=np.int32)
        )
        if points_host.ndim != 2 or points_host.shape[1] != 2:
            raise ValueError(
                f"Sketch.points must have shape (num_points, 2); got {points_host.shape}"
            )
        if lines_host.ndim != 2 or lines_host.shape[1] != 2:
            raise ValueError(
                f"Sketch.lines must have shape (num_lines, 2); got {lines_host.shape}"
            )
        if np.any(lines_host < 0) or np.any(lines_host >= points_host.shape[0]):
            raise ValueError(
                f"Sketch.lines must reference present points; got {lines_host}"
            )
        if np.any(lines_host[:, 0] == lines_host[:, 1]):
            raise ValueError(
                f"Sketch.lines must have distinct endpoints; got {lines_host}"
            )
        if centers_host.shape != radii_host.shape:
            raise ValueError(
                f"Sketch.circle_radii must align with circle_centers; got {radii_host.shape} and {centers_host.shape}"
            )
        if np.any(centers_host < 0) or np.any(centers_host >= points_host.shape[0]):
            raise ValueError(
                f"Sketch.circle_centers must reference present points; got {centers_host}"
            )
        if np.any(~np.isfinite(radii_host)) or np.any(radii_host <= 0.0):
            raise ValueError(
                f"Sketch.circle_radii must be finite and positive; got {radii_host}"
            )
        if arcs_host.ndim != 2 or arcs_host.shape[1] != 3:
            raise ValueError(
                f"Sketch.arcs must have shape (num_arcs, 3); got {arcs_host.shape}"
            )
        if np.any(arcs_host[:, 0] < 0) or np.any(
            arcs_host[:, 0] >= centers_host.shape[0]
        ):
            raise ValueError(
                f"Sketch.arcs.circle must reference present circles; got {arcs_host[:, 0]}"
            )
        if np.any(arcs_host[:, 1:] < 0) or np.any(
            arcs_host[:, 1:] >= points_host.shape[0]
        ):
            raise ValueError(
                f"Sketch.arcs.endpoints must reference present points; got {arcs_host[:, 1:]}"
            )
        if np.any(arcs_host[:, 1] == arcs_host[:, 2]) or np.any(
            arcs_host[:, 1:] == centers_host[arcs_host[:, :1]]
        ):
            raise ValueError(
                f"Sketch.arcs.endpoints must be distinct and off their center; got {arcs_host}"
            )
        constraints_ = tuple(constraints)
        identities, priorities = _constraint_metadata(
            len(constraints_), constraint_ids, constraint_priorities
        )
        if any(not isinstance(item, AbstractSketchConstraint) for item in constraints_):
            raise TypeError(
                "Sketch.constraints must contain sketch constraints; got unsupported member"
            )
        self._validate_constraint_indices(
            constraints_, points_host.shape[0], lines_host.shape[0], radii_host.shape[0]
        )
        self.points = jnp.asarray(points_host, dtype=jnp.float64)
        self.lines = jnp.asarray(lines_host, dtype=jnp.int32)
        self.circle_centers = jnp.asarray(centers_host, dtype=jnp.int32)
        self.circle_radii = jnp.asarray(radii_host, dtype=jnp.float64)
        self.arcs = jnp.asarray(arcs_host, dtype=jnp.int32)
        self.constraints = constraints_
        self.feature_id = feature_id or f"sketch-{uuid4().hex}"
        self.constraint_ids = identities
        self.constraint_priorities = priorities

    @staticmethod
    def _validate_constraint_indices(
        constraints: tuple[AbstractSketchConstraint, ...],
        num_points: int,
        num_lines: int,
        num_circles: int,
    ) -> None:
        def require(indices: tuple[int, ...], size: int, kind: str) -> None:
            if any(index < 0 or index >= size for index in indices):
                raise ValueError(
                    f"Sketch.constraints.indices must reference {kind} indices in [0, {size}); got {indices}"
                )

        for constraint in constraints:
            point_indices: tuple[int, ...] = ()
            line_indices: tuple[int, ...] = ()
            circle_indices: tuple[int, ...] = ()
            match constraint:
                case Coincident() | PointDistance():
                    point_indices = (constraint.first_point, constraint.second_point)
                case FixedPoint():
                    point_indices = (constraint.point,)
                case Midpoint() | PointOnLine():
                    point_indices = (constraint.point,)
                    line_indices = (constraint.line,)
                case Horizontal() | Vertical():
                    line_indices = (constraint.line,)
                case Parallel() | Perpendicular() | EqualLength() | LineAngle():
                    line_indices = (constraint.first_line, constraint.second_line)
                case Radius():
                    circle_indices = (constraint.circle,)
                case TangentLineCircle():
                    line_indices = (constraint.line,)
                    circle_indices = (constraint.circle,)
                case TangentCircles():
                    circle_indices = (constraint.first_circle, constraint.second_circle)
                case AbstractSketchConstraint():
                    pass
                case _:
                    raise TypeError(
                        "Sketch.constraints must contain sketch constraints; "
                        f"got {type(constraint).__name__}"
                    )
            require(point_indices, num_points, "point")
            require(line_indices, num_lines, "line")
            require(circle_indices, num_circles, "circle")

    def residual(self, points: Array, circle_radii: Array, /) -> Array:
        """Constraint residuals followed by intrinsic arc end-point incidence."""
        points_ = jnp.asarray(points, dtype=self.points.dtype).reshape(self.points.shape)
        radii_ = jnp.asarray(circle_radii, dtype=self.circle_radii.dtype).reshape(
            self.circle_radii.shape
        )
        values = [
            constraint.residual(points_, self.lines, self.circle_centers, radii_).reshape(
                (-1,)
            )
            for constraint in self.constraints
        ]
        if self.arcs.shape[0]:
            centers = points_[self.circle_centers[self.arcs[:, 0]]]
            radii = radii_[self.arcs[:, 0]]
            values.extend(
                _finite_norm(points_[self.arcs[:, column]] - centers) - radii
                for column in (1, 2)
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
        """Solve sketch residuals using native LM with admissible decreasing trials.

        Nonpositive radii refuse. Converged regular endpoints carry native IFT
        sensitivities to constraint data, independently of the initial seed.
        """
        max_iterations = positive_integer(max_iterations, "Sketch.solve.max_iterations")
        tolerance = positive_real_scalar(tolerance, "Sketch.solve.tolerance")
        damping = positive_real_scalar(damping, "Sketch.solve.damping")
        initial = jnp.concatenate((self.points.reshape((-1,)), self.circle_radii))
        initial = eqx.error_if(
            initial,
            jnp.any(self.circle_radii <= 0.0),
            "Sketch.circle_radii must be positive; got nonpositive radius",
        )
        lower = jnp.full_like(initial, -jnp.inf).at[self.points.size :].set(0.0)
        strict_lower = (
            jnp.zeros(initial.shape, dtype=jnp.bool_).at[self.points.size :].set(True)
        )
        residual = _SketchResidual(self)
        result = solve_design_residual(
            residual,
            initial,
            DesignTrialPolicy(
                lower, jnp.full_like(initial, jnp.inf), strict_lower, tolerance
            ),
            max_iterations=max_iterations,
            damping=damping,
            maximum_trials=12,
            problem_id=f"{self.feature_id}/sketch",
        )
        vector, derivative_evidence, derivative_rank = attach_sketch_sensitivity(
            residual, result, tolerance
        )
        return SketchSolution(
            points=vector[: self.points.size].reshape(self.points.shape),
            circle_radii=vector[self.points.size :],
            solver_result=result,
            sketch=self,
            residual_bound=tolerance,
            derivative_evidence=derivative_evidence,
            derivative_rank=derivative_rank,
        )

    def _snapshot(self, solution: SketchSolution, /) -> _SketchSnapshot:
        if (
            solution.points.shape != self.points.shape
            or solution.circle_radii.shape != self.circle_radii.shape
        ):
            raise ValueError(
                "Sketch.solution.variables must match sketch shapes; "
                f"got {solution.points.shape}, {solution.circle_radii.shape}"
            )
        points, radii, host_sketch, host_binding, converged, status, bound, norm = (
            jax.device_get(
                (
                    solution.points,
                    solution.circle_radii,
                    self,
                    solution.sketch,
                    solution.converged & solution.solver_result.successful,
                    solution.status,
                    solution.residual_bound,
                    jnp.linalg.norm(
                        self.residual(solution.points, solution.circle_radii)
                    ),
                )
            )
        )
        if not tree_content_equal(host_sketch, host_binding):
            raise ValueError(
                "Sketch.solution.sketch must match the stored geometry and constraints; "
                "got a different sketch binding"
            )
        if (
            not bool(converged)
            or not np.isfinite(bound)
            or bound <= 0.0
            or not np.isfinite(norm)
            or norm > bound
        ):
            raise ValueError(
                "Sketch.solution must be converged within its residual bound; "
                f"got status={int(status)}, residual={norm}, bound={bound}"
            )
        if (
            not np.all(np.isfinite(points))
            or not np.all(np.isfinite(radii))
            or np.any(radii <= 0.0)
        ):
            raise ValueError(
                "Sketch.solution.variables must contain finite points and positive finite radii; "
                f"got points={points}, circle_radii={radii}"
            )
        return _SketchSnapshot(
            np.asarray(points),
            np.asarray(radii),
            np.asarray(host_sketch.lines),
            np.asarray(host_sketch.arcs),
            np.asarray(host_sketch.circle_centers),
            float(bound),
        )

    @checked
    def to_source(
        self, solution: SketchSolution
    ) -> analytic_owner.Circle | simplicial_owner.PlanarMeshRegion:
        """Lower one closed line loop or one circle to a geometry source."""

        snapshot = self._snapshot(solution)
        points, radii = snapshot.points, snapshot.circle_radii
        if self.lines.shape[0] == 0 and self.circle_centers.shape[0] == 1:
            center = points[snapshot.circle_centers[0]]
            return analytic_owner.Circle(
                center, float(radii[0]), feature_id=self.feature_id
            )
        if self.circle_centers.shape[0] != 0:
            raise ValueError(
                "Sketch.to_source must contain one line loop or one circle; "
                "got mixed curves requiring to_profile or to_model"
            )
        lines = snapshot.lines
        polygon = self._line_profile_vertices(points, lines)
        loop = np.arange(polygon.shape[0], dtype=np.int32)

        return simplicial_owner.PlanarMeshRegion(
            polygon, (loop,), feature_id=self.feature_id
        )

    @staticmethod
    def _line_profile_vertices(points: np.ndarray, lines: np.ndarray, /) -> np.ndarray:
        if lines.shape[0] < 3:
            raise ValueError(
                f"Sketch.lines must contain at least three lines; got {lines.shape[0]}"
            )
        adjacency: dict[int, list[int]] = {}
        for start, end in lines.tolist():
            adjacency.setdefault(start, []).append(end)
            adjacency.setdefault(end, []).append(start)
        if len(adjacency) != lines.shape[0] or any(
            len(neighbors) != 2 for neighbors in adjacency.values()
        ):
            raise ValueError(
                "Sketch.lines must form one simple closed cycle; got irregular incidence"
            )
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
            raise ValueError(
                "Sketch.lines must form one connected cycle; got disconnected incidence"
            )
        polygon = points[np.asarray(order, dtype=np.int32)]
        signed_area = 0.5 * np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - np.roll(polygon[:, 0], -1) * polygon[:, 1]
        )
        if signed_area < 0.0:
            polygon = polygon[::-1]
        return polygon

    @staticmethod
    def _profile_cycles(
        endpoints: np.ndarray, /
    ) -> tuple[tuple[tuple[int, bool], ...], ...]:
        """Closed cycles of ``(curve, forward)`` uses; every end point has degree two."""
        incidence: dict[int, list[int]] = {}
        for curve, (start, end) in enumerate(endpoints.tolist()):
            incidence.setdefault(start, []).append(curve)
            incidence.setdefault(end, []).append(curve)
        if any(len(curves) != 2 for curves in incidence.values()):
            raise ValueError(
                "Sketch.profile.curves must form disjoint simple closed cycles; got irregular incidence"
            )
        used = [False] * endpoints.shape[0]
        cycles = []
        for first in range(endpoints.shape[0]):
            if used[first]:
                continue
            cycle = []
            curve, point = first, int(endpoints[first, 0])
            while not used[curve]:
                used[curve] = True
                start, end = (int(value) for value in endpoints[curve])
                forward = start == point
                cycle.append((curve, forward))
                point = end if forward else start
                a, b = incidence[point]
                curve = b if a == curve else a
            cycles.append(tuple(cycle))
        return tuple(cycles)

    @checked
    def to_profile(
        self,
        solution: SketchSolution,
        *,
        plane: constructors_owner.ProfilePlane | None = None,
    ) -> constructors_owner.PlanarProfile:
        """Lower converged lines, arcs and circles to a certified planar profile.

        The profile retains the weighted residual bound, not a spatial error
        enclosure. Its topology is certified independently from the snapshot.
        Line/arc cycles use rational conic carriers; circles without arcs become
        analytic loops. The largest enclosing loop is the outer boundary.
        """

        snapshot = self._snapshot(solution)
        points, radii = snapshot.points, snapshot.circle_radii
        lines, arcs, centers = snapshot.lines, snapshot.arcs, snapshot.circle_centers

        def vertex(index: int) -> tuple[float, float]:
            return float(points[index, 0]), float(points[index, 1])

        def arc_use(
            arc: int, forward: bool
        ) -> tuple[constructors_owner.ProfileSegment, float]:
            """Traversal-ordered conic carrier and its signed area beyond the chord."""
            circle, start, end = (int(value) for value in arcs[arc])
            center, radius = vertex(int(centers[circle])), float(radii[circle])
            spline = constructors_owner.rational_circular_arc(
                vertex(start), vertex(end), center, radius
            )
            sweep = (
                np.arctan2(points[end, 1] - center[1], points[end, 0] - center[0])
                - np.arctan2(points[start, 1] - center[1], points[start, 0] - center[0])
            ) % (2.0 * np.pi)
            area = 0.5 * radius**2 * float(sweep - np.sin(sweep))
            return (
                (spline, area)
                if forward
                else (constructors_owner.reversed_bspline(spline), -area)
            )

        loops: list[constructors_owner.ProfileLoop] = []
        areas: list[float] = []
        for cycle in self._profile_cycles(np.concatenate((lines, arcs[:, 1:]))):
            vertices, segments, area = [], [], 0.0
            for curve, forward in cycle:
                ends = (
                    lines[curve]
                    if curve < lines.shape[0]
                    else arcs[curve - lines.shape[0], 1:]
                )
                start, end = (
                    (int(ends[0]), int(ends[1]))
                    if forward
                    else (int(ends[1]), int(ends[0]))
                )
                vertices.append(vertex(start))
                area += 0.5 * float(
                    points[start, 0] * points[end, 1] - points[end, 0] * points[start, 1]
                )
                segment: constructors_owner.ProfileSegment = (
                    constructors_owner.ProfileLine()
                )
                if curve >= lines.shape[0]:
                    segment, bulge = arc_use(curve - lines.shape[0], forward)
                    area += bulge
                segments.append(segment)
            loops.append(constructors_owner.ProfileLoop(tuple(vertices), tuple(segments)))
            areas.append(abs(area))
        arc_circles = set(arcs[:, 0].tolist())
        for circle in range(centers.shape[0]):
            if circle not in arc_circles:
                loops.append(
                    constructors_owner.ProfileLoop.circle(
                        vertex(int(centers[circle])), float(radii[circle])
                    )
                )
                areas.append(np.pi * float(radii[circle]) ** 2)
        if not loops:
            raise ValueError(
                "Sketch.profile.loops must contain at least one closed curve; got empty loops"
            )
        outer = int(np.argmax(areas))
        return constructors_owner.PlanarProfile(
            constructors_owner.ProfilePlane() if plane is None else plane,
            loops[outer],
            tuple(loop for index, loop in enumerate(loops) if index != outer),
            residual_bound=snapshot.residual_bound,
        )

    @checked
    def to_model(
        self,
        solution: SketchSolution,
        *,
        coordinate_contract: SpatialCoordinateContract,
        plane: ProfilePlane | None = None,
        tessellation: BRepTessellationPolicy | None = None,
    ) -> BRepModel:
        """Lower a converged sketch snapshot to a native planar CAD face.

        The sketch feature identity becomes the CAD source identity. The input
        profile retains the weighted solve residual bound.
        """

        return constructors_owner.brep_planar_face(
            self.to_profile(solution, plane=plane),
            coordinate_contract=coordinate_contract,
            tessellation=tessellation,
            source_id=self.feature_id,
        )


class SketchSolution(StrictModule):
    """Return sketch variables with native solver status and residual evidence."""

    points: Array
    circle_radii: Array
    residual: Array
    residual_norm: Array
    residual_bound: Array
    converged: Array
    iterations: Array
    status: Array
    solver_result: LeastSquaresResult
    sketch: Sketch
    derivative_evidence: SensitivityEvidence
    derivative_rank: Array

    @checked
    def __init__(
        self,
        *,
        points: ConvertibleToArray,
        circle_radii: ConvertibleToArray,
        solver_result: LeastSquaresResult,
        sketch: Sketch,
        residual_bound: float,
        derivative_evidence: SensitivityEvidence,
        derivative_rank: Array,
    ) -> None:
        residual_bound = positive_real_scalar(
            residual_bound, "SketchSolution.residual_bound"
        )
        self.points = jnp.asarray(points, dtype=jnp.float64)
        self.circle_radii = jnp.asarray(circle_radii, dtype=jnp.float64)
        self.residual = jnp.asarray(solver_result.residual, dtype=jnp.float64)
        self.residual_norm = jnp.linalg.norm(self.residual)
        self.residual_bound = jnp.asarray(residual_bound, dtype=jnp.float64)
        self.converged = solver_result.successful & (
            self.residual_norm <= self.residual_bound
        )
        self.iterations = solver_result.diagnostics.iterations
        self.status = solver_result.status
        self.solver_result = solver_result
        self.sketch = sketch
        self.derivative_evidence = derivative_evidence
        self.derivative_rank = derivative_rank


class _SketchResidual(StrictModule):
    sketch: Sketch

    def __call__(self, vector: Array, args: None, /) -> Array:
        del args
        return self.sketch.residual(
            vector[: self.sketch.points.size].reshape(self.sketch.points.shape),
            vector[self.sketch.points.size :],
        )


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
