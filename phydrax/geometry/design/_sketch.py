#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import fields
from fractions import Fraction
from typing import Literal, TYPE_CHECKING, TypeAlias
from uuid import uuid4

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._validation import normalized_identifier
from ._schema import DesignState, ParameterId, ParameterSchema
from ._sketch_certificate import (
    LineCircleConstruction,
    RootDOFEvidence,
    TwoCircleConstruction,
)
from ._sketch_dimensions import (
    compile_sketch_dimensions,
    require_dimension_state,
    SketchDimensionAttribute,
    SketchDimensionBinding,
    SketchDimensionSensitivity,
)
from ._solve import (
    prepare_sketch_fixed_coordinates,
    seed_sketch_circle_intersections,
    solve_prepared_sketch_residual,
)


if TYPE_CHECKING:
    from ..._physical import SpatialCoordinateContract
    from ...optim import DifferentialEvolutionSearch
    from ...typing import PRNGKey
    from ..analytic import Circle
    from ._reduced import _SketchDistanceRigidityWitness
    from ..brep._constructors import PlanarProfile, ProfilePlane
    from ..brep._model import BRepModel
    from ..simplicial import PlanarMeshRegion
    from ._search_result import SketchSearchResult


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
            raise ValueError("constraint weight must be finite and positive.")
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

    def __init__(self, point: int, target: ArrayLike, *, weight: float = 1.0) -> None:
        super().__init__(weight)
        target_ = jnp.asarray(target, dtype=jnp.float64)
        if target_.shape != (2,):
            raise ValueError("FixedPoint target must have shape (2,).")
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
    parameter_id: ParameterId | None = eqx.field(static=True)

    def __init__(
        self,
        first_point: int,
        second_point: int,
        distance: float,
        *,
        parameter_id: ParameterId | None = None,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        if not np.isfinite(distance) or distance < 0.0:
            raise ValueError("distance must be finite and non-negative.")
        if parameter_id is not None and not isinstance(parameter_id, ParameterId):
            raise TypeError("parameter_id must be ParameterId or None.")
        self.first_point = int(first_point)
        self.second_point = int(second_point)
        self.distance = jnp.asarray(distance, dtype=jnp.float64).reshape(())
        self.parameter_id = parameter_id

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
    parameter_id: ParameterId | None = eqx.field(static=True)

    def __init__(
        self,
        first_line: int,
        second_line: int,
        angle: float,
        *,
        parameter_id: ParameterId | None = None,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        if not np.isfinite(angle):
            raise ValueError("angle must be finite.")
        if parameter_id is not None and not isinstance(parameter_id, ParameterId):
            raise TypeError("parameter_id must be ParameterId or None.")
        self.first_line = int(first_line)
        self.second_line = int(second_line)
        self.angle = jnp.asarray(angle, dtype=jnp.float64).reshape(())
        self.parameter_id = parameter_id

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
    parameter_id: ParameterId | None = eqx.field(static=True)

    def __init__(
        self,
        circle: int,
        radius: float,
        *,
        parameter_id: ParameterId | None = None,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be finite and positive.")
        if parameter_id is not None and not isinstance(parameter_id, ParameterId):
            raise TypeError("parameter_id must be ParameterId or None.")
        self.circle = int(circle)
        self.radius = jnp.asarray(radius, dtype=jnp.float64).reshape(())
        self.parameter_id = parameter_id

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
        side: Literal[-1, 1] = 1,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        if side not in (-1, 1):
            raise ValueError("side must be -1 or +1.")
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


class SketchPreparationEvidence(StrictModule):
    """Inputs and algebraic witnesses of one prepared root attempt.

    This records eligibility and initialization separately from convergence,
    numerical rank, and uniqueness. Stored witnesses describe the original
    represented inputs; a consumer must replay their premises before reuse.
    ``reduced_root_accepted`` is an execution observation, not a certificate
    that the recorded path was executed.
    """

    problem_id: str = eqx.field(static=True)
    reference_coordinates: Array
    seed_coordinates: Array
    pinned_indices: tuple[int, ...] = eqx.field(static=True)
    pinned_values: Array
    constructions: tuple[
        tuple[int, TwoCircleConstruction | LineCircleConstruction], ...
    ] = eqx.field(static=True)
    rigid_clusters: tuple[_SketchDistanceRigidityWitness, ...] = eqx.field(static=True)
    reduced_root_accepted: bool = eqx.field(static=True)

    def __init__(
        self,
        problem_id: str,
        reference_coordinates: ArrayLike,
        seed_coordinates: ArrayLike,
        pinned_indices: tuple[int, ...],
        pinned_values: ArrayLike,
        constructions: tuple[
            tuple[int, TwoCircleConstruction | LineCircleConstruction], ...
        ],
        rigid_clusters: tuple[_SketchDistanceRigidityWitness, ...],
        reduced_root_accepted: bool,
        /,
    ) -> None:
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError("problem_id must be a non-empty string.")
        reference = jnp.asarray(reference_coordinates, dtype=jnp.float64).reshape((-1,))
        seed = jnp.asarray(seed_coordinates, dtype=jnp.float64).reshape((-1,))
        values = jnp.asarray(pinned_values, dtype=jnp.float64).reshape((-1,))
        indices = tuple(pinned_indices)
        if reference.shape != seed.shape or values.size != len(indices):
            raise ValueError("Preparation coordinates and fixed values must align.")
        if len(set(indices)) != len(indices) or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= reference.size
            for index in indices
        ):
            raise ValueError("Preparation fixed indices must be distinct and in range.")
        if any(
            not isinstance(point, int)
            or isinstance(point, bool)
            or point < 0
            or 2 * point + 1 >= reference.size
            or not isinstance(witness, (LineCircleConstruction, TwoCircleConstruction))
            for point, witness in constructions
        ):
            raise ValueError(
                "Preparation constructions must identify valid point witnesses."
            )
        if rigid_clusters:
            from ._reduced import _SketchDistanceRigidityWitness

            if any(
                not isinstance(item, _SketchDistanceRigidityWitness)
                for item in rigid_clusters
            ):
                raise TypeError("rigid_clusters must contain exact distance witnesses.")
        if reduced_root_accepted and not (indices or constructions):
            raise ValueError("A reduced root requires an admitted construction.")
        if not isinstance(reduced_root_accepted, bool):
            raise TypeError("reduced_root_accepted must be bool.")
        self.problem_id = problem_id
        self.reference_coordinates = reference
        self.seed_coordinates = seed
        self.pinned_indices = indices
        self.pinned_values = values
        self.constructions = tuple(constructions)
        self.rigid_clusters = tuple(rigid_clusters)
        self.reduced_root_accepted = reduced_root_accepted

    def verify(
        self, sketch: Sketch, dimensions: DesignState, /, *, tolerance: float
    ) -> None:
        """Replay every retained preparation premise against the supplied sketch."""
        if not isinstance(sketch, Sketch) or self.problem_id != sketch._problem_id():
            raise ValueError("Preparation belongs to a different sketch.")
        if (
            not isinstance(dimensions, DesignState)
            or dimensions.schema != sketch.dimension_schema
        ):
            raise ValueError("Preparation dimensions do not match the sketch.")
        if (
            self.reference_coordinates.size
            != sketch.points.size + sketch.circle_radii.size
        ):
            raise ValueError("Preparation reference has the wrong coordinate extent.")
        indices, values, seed, constructions, rigid_clusters = sketch._prepare_components(
            self.reference_coordinates, dimensions, tolerance
        )
        if (
            indices != self.pinned_indices
            or not np.array_equal(np.asarray(values), np.asarray(self.pinned_values))
            or not np.array_equal(np.asarray(seed), np.asarray(self.seed_coordinates))
            or constructions != self.constructions
            or rigid_clusters != self.rigid_clusters
        ):
            raise ValueError("Preparation witnesses do not match the supplied sketch.")


class SketchSolution(StrictModule):
    points: Array
    circle_radii: Array
    residual: Array
    residual_norm: Array
    converged: Array
    iterations: Array
    status: Array
    root_evidence: RootDOFEvidence
    preparation: SketchPreparationEvidence
    dimensions: DesignState

    def __init__(
        self,
        *,
        points: ArrayLike,
        circle_radii: ArrayLike,
        residual: ArrayLike,
        residual_norm: ArrayLike,
        converged: ArrayLike,
        iterations: ArrayLike,
        status: ArrayLike,
        root_evidence: RootDOFEvidence,
        preparation: SketchPreparationEvidence,
        dimensions: DesignState,
    ) -> None:
        if not isinstance(root_evidence, RootDOFEvidence):
            raise TypeError("root_evidence must be RootDOFEvidence.")
        if not isinstance(preparation, SketchPreparationEvidence):
            raise TypeError("preparation must be SketchPreparationEvidence.")
        if preparation.problem_id != root_evidence.problem_id:
            raise ValueError(
                "Preparation and root evidence must identify the same problem."
            )
        if not isinstance(dimensions, DesignState):
            raise TypeError("dimensions must be a DesignState.")
        self.points = jnp.asarray(points, dtype=jnp.float64)
        self.circle_radii = jnp.asarray(circle_radii, dtype=jnp.float64)
        self.residual = jnp.asarray(residual, dtype=jnp.float64)
        self.residual_norm = jnp.asarray(residual_norm, dtype=jnp.float64).reshape(())
        self.converged = jnp.asarray(converged, dtype=jnp.bool_).reshape(())
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())
        self.status = jnp.asarray(status, dtype=jnp.int32).reshape(())
        self.root_evidence = root_evidence
        self.preparation = preparation
        self.dimensions = dimensions


class SketchProfileResult(StrictModule):
    """Exact native planar profile plus the sketch root and source-lineage scope."""

    profile: PlanarProfile
    root_evidence: RootDOFEvidence
    recipe_name: str = eqx.field(static=True)
    lineage_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    derivative_scope: str = eqx.field(static=True)

    def __init__(
        self,
        profile: PlanarProfile,
        root_evidence: RootDOFEvidence,
        recipe_name: str,
        lineage_id: str,
        source_id: str,
    ) -> None:
        if not isinstance(root_evidence, RootDOFEvidence):
            raise TypeError("root_evidence must be RootDOFEvidence.")
        if not isinstance(recipe_name, str):
            raise TypeError("recipe_name must be a string.")
        name = recipe_name.strip()
        if not name:
            raise ValueError("recipe_name must be non-empty.")
        for identifier, field_name in (
            (lineage_id, "lineage_id"),
            (source_id, "source_id"),
        ):
            if (
                not isinstance(identifier, str)
                or len(identifier) != 64
                or identifier != identifier.lower()
                or any(character not in "0123456789abcdef" for character in identifier)
            ):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 fingerprint.")
        self.profile = profile
        self.root_evidence = root_evidence
        self.recipe_name = name
        self.lineage_id = lineage_id
        self.source_id = source_id
        self.derivative_scope = (
            "value-only-profile-lowering; named local root sensitivities are separate"
        )


class SketchModelResult(StrictModule):
    """Native B-Rep face model kept distinct from its source-profile result."""

    model: BRepModel
    source_profile: SketchProfileResult
    root_evidence: RootDOFEvidence
    recipe_name: str = eqx.field(static=True)
    source_lineage_id: str = eqx.field(static=True)
    model_revision_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    derivative_scope: str = eqx.field(static=True)

    def __init__(
        self,
        model: BRepModel,
        source_profile: SketchProfileResult,
    ) -> None:
        from ..brep._model import BRepModel

        if not isinstance(model, BRepModel):
            raise TypeError("model must be a BRepModel.")
        if not isinstance(source_profile, SketchProfileResult):
            raise TypeError("source_profile must be a SketchProfileResult.")
        self.model = model
        self.source_profile = source_profile
        self.root_evidence = source_profile.root_evidence
        self.recipe_name = source_profile.recipe_name
        self.source_lineage_id = source_profile.lineage_id
        self.model_revision_id = model.source_revision
        self.source_id = source_profile.source_id
        self.derivative_scope = (
            "value-only-native-face-construction; no automatic sketch derivative"
        )


class Sketch(StrictModule):
    """A pure-JAX 2D line, circle and directed-arc sketch.

    Each arc is specified by ``arc_endpoints[i]`` (start and end point indices),
    ``arc_centers[i]`` (a center point index), and the boolean
    ``arc_counterclockwise[i]``. Arc incidence adds an equal-endpoint-radius
    residual; its center and endpoints remain ordinary sketch point variables.
    Standalone circles retain their separate radius variables.

    ``to_source`` remains the line-loop/circle geometry-source conversion.
    ``to_profile`` additionally lowers a single connected line/arc cycle to
    native circular segments. Endpoint radii must agree exactly in represented
    coordinates, in addition to native profile geometry validation.
    """

    points: Array
    lines: Array
    circle_centers: Array
    circle_radii: Array
    arc_endpoints: Array
    arc_centers: Array
    arc_counterclockwise: Array
    constraints: tuple[AbstractSketchConstraint, ...]
    constraint_ids: tuple[str, ...] = eqx.field(static=True)
    constraint_priorities: tuple[int, ...] = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)
    dimension_schema: ParameterSchema = eqx.field(static=True)
    dimensions: DesignState
    dimension_bindings: tuple[SketchDimensionBinding, ...] = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        *,
        lines: ArrayLike | None = None,
        circle_centers: ArrayLike | None = None,
        circle_radii: ArrayLike | None = None,
        arc_endpoints: ArrayLike | None = None,
        arc_centers: ArrayLike | None = None,
        arc_counterclockwise: ArrayLike | None = None,
        constraints: Sequence[AbstractSketchConstraint] = (),
        feature_id: str | None = None,
        constraint_ids: Sequence[str] | None = None,
        constraint_priorities: Sequence[int] | None = None,
    ) -> None:
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
        endpoints_host = (
            np.empty((0, 2), dtype=np.int32)
            if arc_endpoints is None
            else np.asarray(arc_endpoints)
        )
        arc_centers_host = (
            np.empty((0,), dtype=np.int32)
            if arc_centers is None
            else np.asarray(arc_centers)
        )
        arc_directions_host = (
            np.empty((0,), dtype=np.bool_)
            if arc_counterclockwise is None
            else np.asarray(arc_counterclockwise)
        )
        if points_host.ndim != 2 or points_host.shape[1] != 2:
            raise ValueError("points must have shape (num_points, 2).")
        if np.any(~np.isfinite(points_host)):
            raise ValueError("points must contain only finite coordinates.")
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
        if endpoints_host.ndim != 2 or endpoints_host.shape[1] != 2:
            raise ValueError("arc_endpoints must have shape (num_arcs, 2).")
        if arc_centers_host.shape != (endpoints_host.shape[0],):
            raise ValueError("arc_centers must contain one point index per arc.")
        if arc_directions_host.shape != (endpoints_host.shape[0],):
            raise ValueError("arc_counterclockwise must contain one boolean per arc.")
        if not np.issubdtype(endpoints_host.dtype, np.integer) or not np.issubdtype(
            arc_centers_host.dtype, np.integer
        ):
            raise TypeError("Arc endpoints and centers must contain integer indices.")
        if arc_directions_host.dtype != np.bool_:
            raise TypeError("arc_counterclockwise must contain booleans.")
        if (
            np.any(endpoints_host < 0)
            or np.any(endpoints_host >= points_host.shape[0])
            or np.any(arc_centers_host < 0)
            or np.any(arc_centers_host >= points_host.shape[0])
        ):
            raise ValueError("Arcs reference an absent point.")
        if np.any(endpoints_host[:, 0] == endpoints_host[:, 1]):
            raise ValueError("Every arc must have distinct endpoints.")
        if np.any(endpoints_host == arc_centers_host[:, None]):
            raise ValueError("An arc center must be distinct from its endpoints.")
        constraints_ = tuple(constraints)
        if any(not isinstance(item, AbstractSketchConstraint) for item in constraints_):
            raise TypeError("constraints must contain sketch constraint objects.")
        self._validate_constraint_indices(
            constraints_, points_host.shape[0], lines_host.shape[0], radii_host.shape[0]
        )
        if feature_id is not None and not isinstance(feature_id, str):
            raise TypeError("feature_id must be a string or None.")
        feature_id_ = f"sketch-{uuid4().hex}" if feature_id is None else feature_id
        if not feature_id_:
            raise ValueError("feature_id must be non-empty.")
        if constraint_ids is None:
            constraint_ids_ = tuple(
                f"{feature_id_}.constraint.{index}" for index in range(len(constraints_))
            )
        else:
            if not isinstance(constraint_ids, Sequence) or isinstance(
                constraint_ids, str
            ):
                raise TypeError("constraint_ids must be a sequence of strings.")
            constraint_ids_ = tuple(
                normalized_identifier(value, "constraint_id") for value in constraint_ids
            )
        if len(constraint_ids_) != len(constraints_) or len(set(constraint_ids_)) != len(
            constraint_ids_
        ):
            raise ValueError("constraint_ids must be unique, one per sketch constraint.")
        if constraint_priorities is None:
            constraint_priorities_ = (0,) * len(constraints_)
        else:
            if not isinstance(constraint_priorities, Sequence) or isinstance(
                constraint_priorities, str
            ):
                raise TypeError("constraint_priorities must be a sequence of integers.")
            if any(
                isinstance(priority, bool) or not isinstance(priority, int)
                for priority in constraint_priorities
            ):
                raise TypeError("Every constraint priority must be an integer.")
            constraint_priorities_ = tuple(constraint_priorities)
            if len(constraint_priorities_) != len(constraints_):
                raise ValueError(
                    "constraint_priorities must contain one value per sketch constraint."
                )
            if any(priority < 0 for priority in constraint_priorities_):
                raise ValueError("constraint_priorities must be non-negative.")
        dimension_entries: list[
            tuple[
                int,
                ParameterId,
                ArrayLike,
                SketchDimensionAttribute,
                tuple[float | None, float | None],
            ]
        ] = []
        for constraint_index, constraint in enumerate(constraints_):
            if (
                isinstance(constraint, PointDistance)
                and constraint.parameter_id is not None
            ):
                dimension_entries.append(
                    (
                        constraint_index,
                        constraint.parameter_id,
                        constraint.distance,
                        "distance",
                        (0.0, None),
                    )
                )
            elif (
                isinstance(constraint, LineAngle) and constraint.parameter_id is not None
            ):
                dimension_entries.append(
                    (
                        constraint_index,
                        constraint.parameter_id,
                        constraint.angle,
                        "angle",
                        (None, None),
                    )
                )
            elif isinstance(constraint, Radius) and constraint.parameter_id is not None:
                dimension_entries.append(
                    (
                        constraint_index,
                        constraint.parameter_id,
                        constraint.radius,
                        "radius",
                        (0.0, None),
                    )
                )
        dimension_schema, dimensions, dimension_bindings = compile_sketch_dimensions(
            dimension_entries, feature_id_
        )
        self.points = jnp.asarray(points_host, dtype=jnp.float64)
        self.lines = jnp.asarray(lines_host, dtype=jnp.int32)
        self.circle_centers = jnp.asarray(centers_host, dtype=jnp.int32)
        self.circle_radii = jnp.asarray(radii_host, dtype=jnp.float64)
        self.arc_endpoints = jnp.asarray(endpoints_host, dtype=jnp.int32)
        self.arc_centers = jnp.asarray(arc_centers_host, dtype=jnp.int32)
        self.arc_counterclockwise = jnp.asarray(arc_directions_host, dtype=jnp.bool_)
        self.constraints = constraints_
        self.constraint_ids = constraint_ids_
        self.constraint_priorities = constraint_priorities_
        self.feature_id = feature_id_
        self.dimension_schema = dimension_schema
        self.dimensions = dimensions
        self.dimension_bindings = dimension_bindings

    @staticmethod
    def _validate_constraint_indices(
        constraints: tuple[AbstractSketchConstraint, ...],
        num_points: int,
        num_lines: int,
        num_circles: int,
    ) -> None:
        def require(indices: tuple[int, ...], size: int, kind: str) -> None:
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

    def _residual_components(
        self,
        points: Array,
        circle_radii: Array,
        /,
        *,
        dimensions: DesignState | None = None,
    ) -> tuple[Array, ...]:
        points_ = jnp.asarray(points, dtype=self.points.dtype).reshape(self.points.shape)
        radii_ = jnp.asarray(circle_radii, dtype=self.circle_radii.dtype).reshape(
            self.circle_radii.shape
        )
        dimension_state = self.dimensions if dimensions is None else dimensions
        if not isinstance(dimension_state, DesignState):
            raise TypeError("dimensions must be a DesignState.")
        if dimension_state.schema != self.dimension_schema:
            raise ValueError(
                "Sketch dimensions do not match the sketch dimension schema."
            )
        binding_by_constraint = {
            binding.constraint_index: binding for binding in self.dimension_bindings
        }
        values: list[Array] = []
        for index, constraint in enumerate(self.constraints):
            binding = binding_by_constraint.get(index)
            if binding is not None:
                value = dimension_state.values[binding.state_index]
                match binding.attribute:
                    case "distance":
                        constraint = eqx.tree_at(
                            lambda item: item.distance, constraint, value
                        )
                    case "angle":
                        constraint = eqx.tree_at(
                            lambda item: item.angle, constraint, value
                        )
                    case "radius":
                        constraint = eqx.tree_at(
                            lambda item: item.radius, constraint, value
                        )
                    case _:
                        raise RuntimeError("Unknown sketch dimension attribute.")
            values.append(
                constraint.residual(
                    points_, self.lines, self.circle_centers, radii_
                ).reshape((-1,))
            )
        return tuple(values)

    def residual(
        self,
        points: ArrayLike,
        circle_radii: ArrayLike,
        /,
        *,
        dimensions: DesignState | None = None,
    ) -> Array:
        points_ = jnp.asarray(points, dtype=self.points.dtype).reshape(self.points.shape)
        radii_ = jnp.asarray(circle_radii, dtype=self.circle_radii.dtype).reshape(
            self.circle_radii.shape
        )
        values = self._residual_components(points_, radii_, dimensions=dimensions)
        if self.arc_endpoints.shape[0]:
            vectors = points_[self.arc_endpoints] - points_[self.arc_centers, None]
            incidence = _finite_norm(vectors[:, 0]) - _finite_norm(vectors[:, 1])
            values = (*values, incidence)
        if not values:
            return jnp.empty((0,), dtype=self.points.dtype)
        return jnp.concatenate(values)

    def _problem_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "line-circle-sketch",
                "feature_id": self.feature_id,
                "lines": np.asarray(self.lines).tolist(),
                "circle_centers": np.asarray(self.circle_centers).tolist(),
                "constraints": [
                    {
                        "type": f"{type(value).__module__}.{type(value).__qualname__}",
                        "static": {
                            item.name: (
                                str(field_value)
                                if isinstance(field_value, ParameterId)
                                else field_value
                            )
                            for item in fields(value)
                            if item.metadata.get("static", False)
                            for field_value in (getattr(value, item.name),)
                        },
                    }
                    for value in self.constraints
                ],
                "constraint_ids": list(self.constraint_ids),
                "constraint_priorities": list(self.constraint_priorities),
                "dimension_ids": [
                    str(value) for value in self.dimension_schema.parameter_ids
                ],
                "arc_endpoints": np.asarray(self.arc_endpoints).tolist(),
                "arc_centers": np.asarray(self.arc_centers).tolist(),
                "arc_counterclockwise": np.asarray(self.arc_counterclockwise).tolist(),
            }
        )

    def _prepare_components(
        self, initial: Array, dimension_state: DesignState, tolerance: float, /
    ) -> tuple[
        tuple[int, ...],
        Array,
        Array,
        tuple[tuple[int, TwoCircleConstruction | LineCircleConstruction], ...],
        tuple[_SketchDistanceRigidityWitness, ...],
    ]:
        pinned_indices, pinned_values = prepare_sketch_fixed_coordinates(
            self, dimension_state
        )
        cluster_seed = initial
        rigid_clusters: tuple[_SketchDistanceRigidityWitness, ...] = ()
        if any(isinstance(item, PointDistance) for item in self.constraints):
            from ._reduced import _prepare_sketch_rigid_clusters

            point_size = self.points.size
            cluster = _prepare_sketch_rigid_clusters(
                self,
                initial[:point_size].reshape(self.points.shape),
                dimensions=dimension_state,
                tolerance=tolerance,
            )
            cluster_seed = initial.at[:point_size].set(cluster.seed_points.reshape((-1,)))
            rigid_clusters = tuple(
                item.exact_distance_witness
                for item in cluster.clusters
                if item.exact_distance_witness is not None
            )
        prepared_seed, constructions = seed_sketch_circle_intersections(
            self,
            dimension_state,
            initial,
            pinned_indices,
            pinned_values,
            initial_seed=cluster_seed,
        )
        return pinned_indices, pinned_values, prepared_seed, constructions, rigid_clusters

    def solve(
        self,
        *,
        initial_coordinates: ArrayLike | None = None,
        dimensions: DesignState | None = None,
        max_iterations: int = 32,
        tolerance: float = 1e-9,
        damping: float = 1e-8,
    ) -> SketchSolution:
        dimension_state = (
            self.dimensions
            if dimensions is None
            else require_dimension_state(dimensions, self.dimension_schema)
        )
        point_size = self.points.size
        initial_default = jnp.concatenate((self.points.reshape((-1,)), self.circle_radii))
        initial = (
            initial_default
            if initial_coordinates is None
            else jnp.asarray(initial_coordinates, dtype=jnp.float64).reshape((-1,))
        )
        if initial.shape != initial_default.shape:
            raise ValueError(
                f"initial_coordinates must have shape {initial_default.shape}."
            )

        def unpack(vector: Array) -> tuple[Array, Array]:
            points = vector[:point_size].reshape(self.points.shape)
            radii = vector[point_size:]
            return points, radii

        def residual(vector: Array) -> Array:
            return self.residual(*unpack(vector), dimensions=dimension_state)

        bounds = None
        if self.circle_radii.size:
            lower = jnp.concatenate(
                (
                    jnp.full((point_size,), -jnp.inf, dtype=initial.dtype),
                    jnp.full(
                        self.circle_radii.shape,
                        np.finfo(np.float64).tiny,
                        dtype=initial.dtype,
                    ),
                )
            )
            upper = jnp.full(initial.shape, jnp.inf, dtype=initial.dtype)
            bounds = (lower, upper)
        problem_id = self._problem_id()
        pinned_indices, pinned_values, prepared_seed, constructions, rigid_clusters = (
            self._prepare_components(initial, dimension_state, tolerance)
        )
        solved, reduced_root_accepted = solve_prepared_sketch_residual(
            residual,
            initial,
            prepared_seed,
            pinned_indices,
            pinned_values,
            constructions,
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=damping,
            problem_id=problem_id,
            bounds=bounds,
        )
        points, radii = unpack(solved.coordinates)
        jacobian = jax.jacfwd(residual)(solved.coordinates)
        root_evidence = RootDOFEvidence(
            problem_id,
            solved.coordinates,
            solved.residual,
            jacobian,
            residual_tolerance=tolerance,
        )
        preparation = SketchPreparationEvidence(
            problem_id,
            initial,
            prepared_seed,
            pinned_indices,
            pinned_values,
            constructions,
            rigid_clusters,
            reduced_root_accepted,
        )
        return SketchSolution(
            points=points,
            circle_radii=radii,
            residual=solved.residual,
            residual_norm=solved.residual_norm,
            converged=solved.converged,
            iterations=solved.iterations,
            status=solved.status,
            root_evidence=root_evidence,
            preparation=preparation,
            dimensions=dimension_state,
        )

    def search(
        self,
        search: DifferentialEvolutionSearch,
        /,
        *,
        key: PRNGKey,
        bounds: tuple[ArrayLike, ArrayLike],
        max_iterations: int = 32,
        tolerance: float = 1e-9,
        damping: float = 1e-8,
    ) -> SketchSearchResult:
        """Search a finite coordinate box and return separate search/root evidence."""
        from ._search import search_sketch

        return search_sketch(
            self,
            search,
            key=key,
            bounds=bounds,
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=damping,
        )

    def _validate_solution_root(self, solution: SketchSolution, /) -> None:
        """Replay numerical root evidence before deriving a scientific result."""
        if not isinstance(solution, SketchSolution):
            raise TypeError("solution must be a SketchSolution.")
        if solution.root_evidence.problem_id != self._problem_id():
            raise ValueError("Sketch solution belongs to a different sketch definition.")
        if solution.dimensions.schema != self.dimension_schema:
            raise ValueError("Sketch solution dimensions do not match the sketch.")
        if (
            solution.points.shape != self.points.shape
            or solution.circle_radii.shape != self.circle_radii.shape
        ):
            raise ValueError("Sketch solution layout does not match the sketch.")
        if not bool(np.asarray(solution.converged)):
            raise ValueError("A converged sketch root is required.")
        solution.preparation.verify(
            self, solution.dimensions, tolerance=solution.root_evidence.residual_tolerance
        )
        coordinates = jnp.concatenate(
            (solution.points.reshape((-1,)), solution.circle_radii)
        )

        def residual(vector: Array) -> Array:
            return self.residual(
                vector[: self.points.size].reshape(self.points.shape),
                vector[self.points.size :],
                dimensions=solution.dimensions,
            )

        values = residual(coordinates)
        evidence = RootDOFEvidence(
            self._problem_id(),
            coordinates,
            values,
            jax.jacfwd(residual)(coordinates),
            residual_tolerance=solution.root_evidence.residual_tolerance,
        )
        if (
            not evidence.root_converged
            or evidence.evidence_id != solution.root_evidence.evidence_id
            or not np.array_equal(np.asarray(values), np.asarray(solution.residual))
        ):
            raise ValueError("Sketch root evidence does not match the supplied solution.")

    def dimension_sensitivity(
        self,
        solution: SketchSolution,
        parameter_id: ParameterId,
        /,
    ) -> SketchDimensionSensitivity:
        """Differentiate a regular converged sketch root with respect to one named dimension."""
        if not isinstance(parameter_id, ParameterId):
            raise TypeError("parameter_id must be ParameterId.")
        self._validate_solution_root(solution)
        if not solution.root_evidence.regular_root_indicated:
            raise NotImplementedError(
                "Dimension sensitivity requires a full-column-rank local root."
            )
        dimension_index = self.dimension_schema.index(parameter_id)
        binding = next(
            (
                value
                for value in self.dimension_bindings
                if value.parameter_id == parameter_id
            ),
            None,
        )
        if binding is None:
            raise KeyError(f"Unknown sketch dimension {parameter_id}.")
        dimension_value = float(
            np.asarray(solution.dimensions.values[dimension_index], dtype=np.float64)
        )
        if binding.attribute == "distance" and dimension_value == 0.0:
            raise NotImplementedError(
                "A zero point-distance dimension is not classically differentiable."
            )
        points_host = np.asarray(solution.points, dtype=np.float64)
        lines_host = np.asarray(self.lines, dtype=np.int32)
        if lines_host.size:
            vectors = points_host[lines_host[:, 1]] - points_host[lines_host[:, 0]]
            if np.any(np.linalg.norm(vectors, axis=1) == 0.0):
                raise NotImplementedError(
                    "Dimension sensitivity is unsupported at a collapsed sketch line."
                )
        if self.arc_endpoints.shape[0]:
            arc_vectors = (
                points_host[np.asarray(self.arc_endpoints, dtype=np.int32)]
                - points_host[np.asarray(self.arc_centers, dtype=np.int32), None]
            )
            if np.any(np.linalg.norm(arc_vectors, axis=2) == 0.0):
                raise NotImplementedError(
                    "Dimension sensitivity is unsupported at a collapsed sketch arc."
                )
        centers = points_host[np.asarray(self.circle_centers, dtype=np.int32)]
        radii_host = np.asarray(solution.circle_radii, dtype=np.float64)
        for constraint in self.constraints:
            if (
                isinstance(constraint, PointDistance)
                and np.linalg.norm(
                    points_host[constraint.second_point]
                    - points_host[constraint.first_point]
                )
                == 0.0
            ):
                raise NotImplementedError(
                    "Dimension sensitivity is unsupported at coincident distance points."
                )
            if isinstance(constraint, TangentCircles):
                first_center = centers[constraint.first_circle]
                second_center = centers[constraint.second_circle]
                if np.linalg.norm(second_center - first_center) == 0.0:
                    raise NotImplementedError(
                        "Dimension sensitivity is unsupported at coincident circle centers."
                    )
                if constraint.internal and (
                    radii_host[constraint.first_circle]
                    == radii_host[constraint.second_circle]
                ):
                    raise NotImplementedError(
                        "Internal circle tangency is nonsmooth at equal radii."
                    )
        coordinates = jnp.concatenate(
            (solution.points.reshape((-1,)), solution.circle_radii)
        )
        point_size = self.points.size

        def unpack(vector: Array) -> tuple[Array, Array]:
            return (
                vector[:point_size].reshape(self.points.shape),
                vector[point_size:],
            )

        def residual_coordinates(vector: Array) -> Array:
            return self.residual(
                *unpack(vector),
                dimensions=solution.dimensions,
            )

        def residual_dimension(value: Array) -> Array:
            dimensions = solution.dimensions.replace_at(dimension_index, value)
            return self.residual(*unpack(coordinates), dimensions=dimensions)

        coordinate_jacobian = jax.jacfwd(residual_coordinates)(coordinates)
        dimension_tangent = jax.jacfwd(residual_dimension)(
            solution.dimensions.values[dimension_index]
        )
        from ...linalg import (
            DenseLinearOperator,
            DenseSVD,
            LeastSquaresProblem,
            LinearSolvePolicy,
            solve as solve_linear,
        )

        linear = solve_linear(
            LeastSquaresProblem(DenseLinearOperator(coordinate_jacobian)),
            -dimension_tangent,
            policy=LinearSolvePolicy(DenseSVD()),
        )
        if not bool(np.all(np.asarray(linear.successful))):
            raise NotImplementedError(
                "The native linear solver could not resolve the sketch dimension sensitivity."
            )
        tangent = jnp.asarray(linear.value, dtype=coordinates.dtype).reshape((-1,))
        residual_tangent = coordinate_jacobian @ tangent + dimension_tangent
        residual_tangent_norm = float(
            np.linalg.norm(np.asarray(residual_tangent, dtype=np.float64))
        )
        scale = max(
            1.0,
            float(np.linalg.norm(np.asarray(dimension_tangent, dtype=np.float64))),
        )
        if residual_tangent_norm > solution.root_evidence.residual_tolerance * scale:
            raise NotImplementedError(
                "The dimension perturbation is inconsistent with the retained sketch constraints."
            )
        return SketchDimensionSensitivity(
            parameter_id,
            tangent,
            residual_tangent,
            solution.root_evidence.evidence_id,
        )

    def to_source(
        self, solution: SketchSolution | None = None
    ) -> Circle | PlanarMeshRegion:
        """Lower one closed line loop or one circle to a geometry source."""

        if self.arc_endpoints.shape[0]:
            raise NotImplementedError(
                "Arc profiles are unsupported by Sketch.to_source; use to_profile."
            )
        points = np.asarray(self.points if solution is None else solution.points)
        radii = np.asarray(
            self.circle_radii if solution is None else solution.circle_radii
        )
        if self.lines.shape[0] == 0 and self.circle_centers.shape[0] == 1:
            from ..analytic import Circle

            center = points[int(np.asarray(self.circle_centers)[0])]
            return Circle(center, float(radii[0]), feature_id=self.feature_id)
        if self.circle_centers.shape[0] != 0:
            raise NotImplementedError(
                "Mixed line/circle profiles are unsupported by Sketch.to_source."
            )
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

    def to_profile(
        self,
        solution: SketchSolution,
        /,
        *,
        name: str,
        plane: ProfilePlane | None = None,
    ) -> SketchProfileResult:
        """Lower one converged sketch root to an exact native planar profile."""
        self._validate_solution_root(solution)
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        recipe_name = name.strip()
        if not recipe_name:
            raise ValueError("name must be non-empty.")
        from ..analytic import Circle
        from ..brep._constructors import (
            PlanarProfile,
            ProfileArc,
            ProfileLine,
            ProfileLoop,
            ProfilePlane,
        )
        from ..simplicial import PlanarMeshRegion

        plane_ = ProfilePlane() if plane is None else plane
        if not isinstance(plane_, ProfilePlane):
            raise TypeError("plane must be a ProfilePlane or None.")
        if self.arc_endpoints.shape[0]:
            if self.circle_centers.shape[0]:
                raise NotImplementedError(
                    "Mixed standalone circles and arc profiles are unsupported."
                )
            points = np.asarray(solution.points, dtype=np.float64)
            lines = np.asarray(self.lines, dtype=np.int32)
            arcs = np.asarray(self.arc_endpoints, dtype=np.int32)
            centers = np.asarray(self.arc_centers, dtype=np.int32)
            directions = np.asarray(self.arc_counterclockwise, dtype=np.bool_)
            edges = np.concatenate((lines, arcs), axis=0)
            adjacency: dict[int, list[int]] = {}
            for index, (start, end) in enumerate(edges.tolist()):
                adjacency.setdefault(start, []).append(index)
                adjacency.setdefault(end, []).append(index)
            if any(len(incident) != 2 for incident in adjacency.values()):
                raise ValueError("Line and arc profiles must form one closed cycle.")
            start = min(adjacency)
            current = start
            used: set[int] = set()
            vertices: list[tuple[float, float]] = []
            segments: list[ProfileLine | ProfileArc] = []
            geometry_segments: list[dict[str, object]] = []
            while True:
                candidates = [index for index in adjacency[current] if index not in used]
                if not candidates:
                    break
                index = candidates[0]
                used.add(index)
                first, last = edges[index]
                forward = int(first) == current
                following = int(last if forward else first)
                vertices.append((float(points[current, 0]), float(points[current, 1])))
                if index < len(lines):
                    segments.append(ProfileLine())
                    geometry_segments.append({"kind": "line"})
                else:
                    arc_index = index - len(lines)
                    center = (
                        float(points[centers[arc_index], 0]),
                        float(points[centers[arc_index], 1]),
                    )
                    exact_center = tuple(Fraction.from_float(value) for value in center)
                    exact_start = tuple(
                        Fraction.from_float(float(value)) for value in points[current]
                    )
                    exact_end = tuple(
                        Fraction.from_float(float(value)) for value in points[following]
                    )
                    start_radius_squared = sum(
                        (value - origin) ** 2
                        for value, origin in zip(exact_start, exact_center, strict=True)
                    )
                    end_radius_squared = sum(
                        (value - origin) ** 2
                        for value, origin in zip(exact_end, exact_center, strict=True)
                    )
                    if (
                        start_radius_squared == 0
                        or start_radius_squared != end_radius_squared
                    ):
                        raise ValueError(
                            "Exact arc lowering requires both endpoints on the same "
                            "positive-radius circle in represented coordinates."
                        )
                    direction = (
                        bool(directions[arc_index])
                        if forward
                        else not bool(directions[arc_index])
                    )
                    segments.append(ProfileArc(center, direction))
                    geometry_segments.append(
                        {"kind": "arc", "center": center, "counterclockwise": direction}
                    )
                current = following
                if current == start:
                    break
            if current != start or len(used) != len(edges):
                raise ValueError("Line and arc profiles must form one connected cycle.")
            outer = ProfileLoop(tuple(vertices), tuple(segments))
            profile_geometry = {
                "kind": "line-arc-loop",
                "vertices": vertices,
                "segments": geometry_segments,
            }
        else:
            source = self.to_source(solution)
            if isinstance(source, Circle):
                outer = ProfileLoop.circle(
                    np.asarray(source.center, dtype=np.float64),
                    float(source.radius),
                )
                profile_geometry = {
                    "kind": "circle",
                    "center": np.asarray(source.center, dtype=np.float64).tolist(),
                    "radius": float(source.radius),
                }
            elif isinstance(source, PlanarMeshRegion):
                vertices = np.asarray(source.vertices, dtype=np.float64)
                outer = ProfileLoop.polygon(vertices)
                profile_geometry = {
                    "kind": "polygon",
                    "vertices": vertices.tolist(),
                }
            else:
                raise TypeError("Sketch.to_source returned an unsupported source type.")
        profile = PlanarProfile(
            plane_,
            outer,
        )
        lineage_id = canonical_fingerprint(
            {
                "kind": "sketch-planar-profile-lineage",
                "feature_id": self.feature_id,
                "lines": np.asarray(self.lines, dtype=np.int32).tolist(),
                "circle_centers": np.asarray(
                    self.circle_centers, dtype=np.int32
                ).tolist(),
                "arc_endpoints": np.asarray(self.arc_endpoints).tolist(),
                "arc_centers": np.asarray(self.arc_centers).tolist(),
                "arc_counterclockwise": np.asarray(self.arc_counterclockwise).tolist(),
                "plane": {
                    "origin": np.asarray(plane_.origin, dtype=np.float64).tolist(),
                    "x_axis": np.asarray(plane_.x_axis, dtype=np.float64).tolist(),
                    "y_axis": np.asarray(plane_.y_axis, dtype=np.float64).tolist(),
                },
                "geometry": profile_geometry,
            }
        )
        source_id = canonical_fingerprint(
            {
                "kind": "native-sketch-profile-source",
                "lineage": lineage_id,
            }
        )
        return SketchProfileResult(
            profile,
            solution.root_evidence,
            recipe_name,
            lineage_id,
            source_id,
        )

    def to_model(
        self,
        source_profile: SketchProfileResult,
        /,
        *,
        coordinate_contract: SpatialCoordinateContract,
    ) -> SketchModelResult:
        """Construct the native B-Rep face model from one explicit profile result."""
        if not isinstance(source_profile, SketchProfileResult):
            raise TypeError("source_profile must be a SketchProfileResult.")
        if source_profile.root_evidence.problem_id != self._problem_id():
            raise ValueError("Source profile belongs to a different sketch definition.")
        from ..._physical import SpatialCoordinateContract
        from ..brep._constructors import brep_planar_face

        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        model = brep_planar_face(
            source_profile.profile,
            coordinate_contract=coordinate_contract,
            source_id=source_profile.source_id,
        )
        return SketchModelResult(model, source_profile)


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
    "SketchPreparationEvidence",
    "SketchSolution",
    "TangentCircles",
    "TangentLineCircle",
    "SketchDimensionBinding",
    "SketchDimensionSensitivity",
    "SketchModelResult",
    "SketchProfileResult",
    "Vertical",
]
