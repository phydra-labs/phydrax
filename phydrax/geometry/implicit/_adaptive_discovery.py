#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from enum import IntEnum
from fractions import Fraction
from typing import final, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import checked
from .._certificate import FieldRegularity, SignReliability, ZeroSetAccuracy
from .._certified_implicit import CertifiedImplicitTopology
from .._contracts import CompiledGeometry, GeometryKind
from .._interval_enclosure import (
    IntervalExtensionError,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from .._source_boundary_records import (
    SourceBoundaryDistance,
    SourceBoundaryProjectionResult,
    SourceBoundarySamples,
)
from ..simplicial import TriangleMesh
from ._policy import (
    AdaptiveImplicitBoxIssue,
    AdaptiveImplicitSurfacePolicy,
    AdaptiveImplicitSurfaceStatus,
    ImplicitDiscoveryAccuracy,
    ImplicitDiscoveryEnclosure,
)


_INTERVAL_BATCH_SIZE = 64
_GridIndex: TypeAlias = tuple[int, int, int]
_GridEdge: TypeAlias = tuple[_GridIndex, _GridIndex]
_GridFaceKey: TypeAlias = tuple[_GridIndex, ...]


def _evaluate_interval_points(
    program: PreparedIntervalFunction, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lower_parts: list[np.ndarray] = []
    upper_parts: list[np.ndarray] = []
    for start in range(0, points.shape[0], _INTERVAL_BATCH_SIZE):
        stop = min(start + _INTERVAL_BATCH_SIZE, points.shape[0])
        lower, upper = program.evaluate(points[start:stop], points[start:stop])
        lower_parts.append(np.asarray(lower, dtype=np.float64))
        upper_parts.append(np.asarray(upper, dtype=np.float64))
    if not lower_parts:
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty.copy()
    return np.concatenate(lower_parts, axis=0), np.concatenate(upper_parts, axis=0)


def _evaluate_interval_boxes(
    program: PreparedIntervalFunction, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lower_parts: list[np.ndarray] = []
    upper_parts: list[np.ndarray] = []
    for start in range(0, lower.shape[0], _INTERVAL_BATCH_SIZE):
        stop = min(start + _INTERVAL_BATCH_SIZE, lower.shape[0])
        bound_lower, bound_upper = program.evaluate(lower[start:stop], upper[start:stop])
        lower_parts.append(np.asarray(bound_lower, dtype=np.float64))
        upper_parts.append(np.asarray(bound_upper, dtype=np.float64))
    if not lower_parts:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    return np.concatenate(lower_parts, axis=0), np.concatenate(upper_parts, axis=0)


_DIMENSION = 3
_CORNER_OFFSETS = np.asarray(
    (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ),
    dtype=np.int64,
)
_CUBE_EDGES = np.asarray(
    (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ),
    dtype=np.int64,
)
_CUBE_FACES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)
_CORNER_LOOKUP = {
    tuple(point.tolist()): index for index, point in enumerate(_CORNER_OFFSETS)
}
_TETRAHEDRA = tuple(
    tuple(
        _CORNER_LOOKUP[tuple(point.tolist())]
        for point in (
            np.zeros(3, dtype=np.int64),
            np.eye(3, dtype=np.int64)[permutation[0]],
            np.eye(3, dtype=np.int64)[permutation[0]]
            + np.eye(3, dtype=np.int64)[permutation[1]],
            np.ones(3, dtype=np.int64),
        )
    )
    for permutation in itertools.permutations(range(3))
)
_TETRA_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


class ImplicitVolumeClass(IntEnum):
    """Inside, outside, or unresolved classification of a source query."""

    OUTSIDE = 0
    INSIDE = 1
    UNKNOWN = 2


@dataclass(frozen=True, slots=True)
class ImplicitVolumeClassification:
    classes: np.ndarray
    value_lower: np.ndarray
    value_upper: np.ndarray
    certified: bool


@dataclass(frozen=True, slots=True)
class _BoxBounds:
    value_lower: np.ndarray
    value_upper: np.ndarray
    gradient_lower: np.ndarray
    gradient_upper: np.ndarray


@dataclass(frozen=True, slots=True)
class _Cell:
    level: int
    lower_index: _GridIndex
    lower: np.ndarray
    upper: np.ndarray
    value_lower: float
    value_upper: float
    gradient_lower: np.ndarray
    gradient_upper: np.ndarray
    issue: int = 0


class _FieldEnclosures:
    """Value and gradient bounds selected by the adaptive discovery policy."""

    lipschitz: float | None
    error: float | None
    _values: PreparedIntervalFunction | None
    _gradients: PreparedIntervalFunction | None

    def __init__(
        self, geometry: CompiledGeometry, selector: ImplicitDiscoveryEnclosure
    ) -> None:
        self.geometry = geometry
        self.selector = selector
        self.certificate = geometry.field_certificate
        self.value_rigorous = selector != "sampled"
        self.gradient_rigorous = selector == "interval"
        self.continuity_rigorous = selector != "sampled"
        self.lipschitz = self.certificate.lipschitz_upper_bound
        self.error = self.certificate.evaluation_error
        if selector == "interval":
            kernel, state = geometry.kernel, geometry.state

            def value_function(points: jax.Array) -> jax.Array:
                return kernel.boundary_field(state, points)

            def gradient_function(points: jax.Array) -> jax.Array:
                def scalar(point: jax.Array) -> jax.Array:
                    return kernel.boundary_field(state, point[None, :])[0]

                return jax.grad(scalar)(points)

            self._values = prepare_interval_function(value_function, _DIMENSION)
            self._gradients = None
            try:
                self._gradients = prepare_interval_function(gradient_function, _DIMENSION)
            except IntervalExtensionError:
                self.gradient_rigorous = False
        elif selector == "lipschitz":
            if self.lipschitz is None or self.error is None:
                raise ValueError(
                    "Lipschitz discovery requires declared field Lipschitz and evaluation-error bounds."
                )
            self._values = self._gradients = None
        else:
            self._values = self._gradients = None

    def field_values(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(
            self.geometry.kernel.boundary_field(self.geometry.state, jnp.asarray(points)),
            dtype=np.float64,
        ).reshape((-1,))
        if not np.all(np.isfinite(values)):
            raise ValueError("The implicit field produced a nonfinite sample.")
        return values

    def point_values(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != _DIMENSION
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("point queries must be finite with shape (N, 3).")
        if values.shape[0] == 0:
            return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
        if self.selector == "interval":
            program = self._values
            if program is None:
                raise RuntimeError("Interval value function was not prepared.")
            lower, upper = _evaluate_interval_points(program, values)
            return np.asarray(lower, dtype=np.float64).reshape((-1,)), np.asarray(
                upper, dtype=np.float64
            ).reshape((-1,))
        sampled = self.field_values(values)
        if self.selector == "sampled":
            return sampled, sampled
        error_bound = self.error
        if error_bound is None:
            raise ValueError(
                "Lipschitz discovery requires declared field Lipschitz and evaluation-error bounds."
            )
        error = float(error_bound)
        return (
            np.nextafter(sampled - error, -np.inf),
            np.nextafter(sampled + error, np.inf),
        )

    def boxes(self, lower: np.ndarray, upper: np.ndarray) -> _BoxBounds:
        low = np.asarray(lower, dtype=np.float64)
        high = np.asarray(upper, dtype=np.float64)
        if low.ndim != 2 or high.shape != low.shape or low.shape[1] != 3:
            raise ValueError(
                "box queries require lower and upper arrays of shape (N, 3)."
            )
        if (
            not np.all(np.isfinite(low))
            or not np.all(np.isfinite(high))
            or np.any(low > high)
        ):
            raise ValueError("box queries require finite ordered bounds.")
        if low.shape[0] == 0:
            return _BoxBounds(
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )
        if self.selector == "interval":
            program = self._values
            if program is None:
                raise RuntimeError("Interval value function was not prepared.")
            value_lower, value_upper = _evaluate_interval_boxes(program, low, high)
            gradient_program = self._gradients
            if gradient_program is None:
                gradient_lower = np.full((low.shape[0], 3), -np.inf, dtype=np.float64)
                gradient_upper = np.full((low.shape[0], 3), np.inf, dtype=np.float64)
            else:
                try:
                    gradient_lower, gradient_upper = _evaluate_interval_boxes(
                        gradient_program, low, high
                    )
                except IntervalExtensionError:
                    self._gradients = None
                    self.gradient_rigorous = False
                    gradient_lower = np.full((low.shape[0], 3), -np.inf, dtype=np.float64)
                    gradient_upper = np.full((low.shape[0], 3), np.inf, dtype=np.float64)
            return _BoxBounds(
                np.asarray(value_lower, dtype=np.float64).reshape((-1,)),
                np.asarray(value_upper, dtype=np.float64).reshape((-1,)),
                np.asarray(gradient_lower, dtype=np.float64).reshape((-1, 3)),
                np.asarray(gradient_upper, dtype=np.float64).reshape((-1, 3)),
            )
        centers = low + 0.5 * (high - low)
        samples = [self.field_values(centers)]
        if self.selector == "sampled":
            for corner in _CORNER_OFFSETS:
                point = np.where(corner[None, :] == 0, low, high)
                samples.append(self.field_values(point))
            values = np.stack(samples, axis=0)
            return _BoxBounds(
                np.min(values, axis=0),
                np.max(values, axis=0),
                np.full((low.shape[0], 3), -np.inf, dtype=np.float64),
                np.full((low.shape[0], 3), np.inf, dtype=np.float64),
            )
        displacement = np.nextafter(0.5 * (high - low), np.inf)
        squared = np.nextafter(displacement * displacement, np.inf)
        sum_upper = np.zeros((low.shape[0],), dtype=np.float64)
        for axis in range(3):
            sum_upper = np.nextafter(sum_upper + squared[:, axis], np.inf)
        radius = np.nextafter(np.sqrt(np.nextafter(sum_upper, np.inf)), np.inf)
        lipschitz = self.lipschitz
        error_bound = self.error
        if lipschitz is None or error_bound is None:
            raise ValueError(
                "Lipschitz discovery requires declared field Lipschitz and evaluation-error bounds."
            )
        lipschitz_value = float(lipschitz)
        error_value = float(error_bound)
        variation = np.nextafter(lipschitz_value * radius, np.inf)
        sampled = samples[0]
        lower_values = np.nextafter(
            np.nextafter(sampled - variation, -np.inf) - error_value, -np.inf
        )
        upper_values = np.nextafter(
            np.nextafter(sampled + variation, np.inf) + error_value, np.inf
        )
        gradient_lower = np.full((low.shape[0], 3), -lipschitz_value, dtype=np.float64)
        gradient_upper = np.full((low.shape[0], 3), lipschitz_value, dtype=np.float64)
        return _BoxBounds(lower_values, upper_values, gradient_lower, gradient_upper)


class AdaptiveImplicitCover(StrictModule, NonTrainableState):
    """Exhaustive domain cover retaining value enclosures of every final leaf."""

    boxes: np.ndarray
    value_lower: np.ndarray
    value_upper: np.ndarray
    gradient_lower: np.ndarray
    gradient_upper: np.ndarray
    domain: np.ndarray
    source_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)
    enclosure: ImplicitDiscoveryEnclosure = eqx.field(static=True)
    complete: bool = eqx.field(static=True)
    certified: bool = eqx.field(static=True)
    cover_id: str = eqx.field(static=True)

    def __init__(
        self,
        boxes: np.ndarray,
        value_lower: np.ndarray,
        value_upper: np.ndarray,
        gradient_lower: np.ndarray,
        gradient_upper: np.ndarray,
        domain: np.ndarray,
        source_id: str,
        state_id: str,
        enclosure: ImplicitDiscoveryEnclosure,
        complete: bool,
        certified: bool,
        cover_id: str,
    ) -> None:
        boxes_ = np.array(boxes, dtype=np.float64, copy=True).reshape((-1, 2, 3))
        lower = np.array(value_lower, dtype=np.float64, copy=True).reshape((-1,))
        upper = np.array(value_upper, dtype=np.float64, copy=True).reshape((-1,))
        gradient_lower_ = np.array(gradient_lower, dtype=np.float64, copy=True).reshape(
            (-1, 3)
        )
        gradient_upper_ = np.array(gradient_upper, dtype=np.float64, copy=True).reshape(
            (-1, 3)
        )
        domain_ = np.array(domain, dtype=np.float64, copy=True)
        if (
            lower.shape != (boxes_.shape[0],)
            or upper.shape != lower.shape
            or gradient_lower_.shape != (boxes_.shape[0], 3)
            or gradient_upper_.shape != gradient_lower_.shape
            or domain_.shape != (2, 3)
            or not np.all(np.isfinite(boxes_))
            or not np.all(np.isfinite(domain_))
            or np.any(np.isnan(lower))
            or np.any(np.isnan(upper))
            or np.any(np.isnan(gradient_lower_))
            or np.any(np.isnan(gradient_upper_))
            or np.any(boxes_[:, 0] > boxes_[:, 1])
            or np.any(lower > upper)
            or np.any(domain_[0] >= domain_[1])
        ):
            raise ValueError(
                "Adaptive cover arrays must retain aligned ordered enclosures."
            )
        for values in (boxes_, lower, upper, gradient_lower_, gradient_upper_, domain_):
            values.setflags(write=False)
        self.boxes = boxes_
        self.value_lower = lower
        self.value_upper = upper
        self.gradient_lower = gradient_lower_
        self.gradient_upper = gradient_upper_
        self.domain = domain_
        self.source_id = source_id
        self.state_id = state_id
        self.enclosure = enclosure
        self.complete = bool(complete)
        self.certified = bool(certified)
        self.cover_id = cover_id

    @property
    def established(self) -> bool:
        return self.complete and self.certified

    def require_bound(self, source_id: str, state_id: str, /) -> None:
        if source_id != self.source_id or state_id != self.state_id:
            raise ValueError(
                "The adaptive cover is bound to a different source revision."
            )

    @property
    def possible_zero_mask(self) -> np.ndarray:
        return (self.value_lower <= 0.0) & (self.value_upper >= 0.0)


class ImplicitVolumeQuery(StrictModule, NonTrainableState):
    """Point and box classification against the retained implicit source."""

    geometry: CompiledGeometry
    bounds: _FieldEnclosures = eqx.field(static=True)
    domain: np.ndarray
    source_id: str = eqx.field(static=True)
    certified: bool = eqx.field(static=True)
    query_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: CompiledGeometry,
        bounds: _FieldEnclosures,
        domain: np.ndarray,
        source_id: str,
        certified: bool,
        query_id: str,
    ) -> None:
        domain_ = np.array(domain, dtype=np.float64, copy=True)
        if (
            domain_.shape != (2, 3)
            or not np.all(np.isfinite(domain_))
            or np.any(domain_[0] >= domain_[1])
        ):
            raise ValueError("Implicit volume query domain must be finite and ordered.")
        domain_.setflags(write=False)
        self.geometry = geometry
        self.bounds = bounds
        self.domain = domain_
        self.source_id = source_id
        self.certified = bool(certified)
        self.query_id = query_id

    def classify_points(self, points: ArrayLike, /) -> ImplicitVolumeClassification:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("points must be finite with shape (N, 3).")
        lower, upper = self.bounds.point_values(values)
        classes = _classes(lower, upper)
        within = np.all((values >= self.domain[0]) & (values <= self.domain[1]), axis=1)
        classes[~within] = int(ImplicitVolumeClass.UNKNOWN)
        return ImplicitVolumeClassification(classes, lower, upper, self.certified)

    def classify_boxes(
        self, lower: ArrayLike, upper: ArrayLike, /
    ) -> ImplicitVolumeClassification:
        low = np.asarray(lower, dtype=np.float64)
        high = np.asarray(upper, dtype=np.float64)
        if low.ndim != 2 or high.shape != low.shape or low.shape[1] != 3:
            raise ValueError(
                "box queries require lower and upper arrays of shape (N, 3)."
            )
        if (
            not np.all(np.isfinite(low))
            or not np.all(np.isfinite(high))
            or np.any(low > high)
        ):
            raise ValueError("box queries require finite ordered bounds.")
        result = self.bounds.boxes(low, high)
        classes = _classes(result.value_lower, result.value_upper)
        within = np.all((low >= self.domain[0]) & (high <= self.domain[1]), axis=1)
        classes[~within] = int(ImplicitVolumeClass.UNKNOWN)
        return ImplicitVolumeClassification(
            classes, result.value_lower, result.value_upper, self.certified
        )


def _classes(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    result = np.full(lower.shape, int(ImplicitVolumeClass.UNKNOWN), dtype=np.int8)
    result[upper < 0.0] = int(ImplicitVolumeClass.INSIDE)
    result[lower > 0.0] = int(ImplicitVolumeClass.OUTSIDE)
    return result


class AdaptiveImplicitSurfaceEvidence(StrictModule, NonTrainableState):
    """Adaptive-discovery issues and numerical termination evidence."""

    unresolved_boxes: np.ndarray
    unresolved_issues: np.ndarray
    accuracy: ImplicitDiscoveryAccuracy = eqx.field(static=True)
    enclosure: ImplicitDiscoveryEnclosure = eqx.field(static=True)
    status: int = eqx.field(static=True)
    leaf_count: int = eqx.field(static=True)
    surface_leaf_count: int = eqx.field(static=True)
    maximum_level_reached: int = eqx.field(static=True)
    maximum_surface_box_diagonal: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        unresolved_boxes: np.ndarray,
        unresolved_issues: np.ndarray,
        accuracy: ImplicitDiscoveryAccuracy,
        enclosure: ImplicitDiscoveryEnclosure,
        status: int,
        leaf_count: int,
        surface_leaf_count: int,
        maximum_level_reached: int,
        maximum_surface_box_diagonal: float,
        evidence_id: str,
    ) -> None:
        boxes = np.array(unresolved_boxes, dtype=np.float64, copy=True).reshape(
            (-1, 2, 3)
        )
        issues = np.array(unresolved_issues, dtype=np.int64, copy=True).reshape((-1,))
        if (
            boxes.shape[0] != issues.shape[0]
            or not np.all(np.isfinite(boxes))
            or np.any(boxes[:, 0] > boxes[:, 1])
            or np.any(issues == 0)
        ):
            raise ValueError(
                "Adaptive evidence requires aligned unresolved boxes and nonzero issues."
            )
        boxes.setflags(write=False)
        issues.setflags(write=False)
        self.unresolved_boxes = boxes
        self.unresolved_issues = issues
        self.accuracy = accuracy
        self.enclosure = enclosure
        self.status = int(status)
        self.leaf_count = int(leaf_count)
        self.surface_leaf_count = int(surface_leaf_count)
        self.maximum_level_reached = int(maximum_level_reached)
        self.maximum_surface_box_diagonal = float(maximum_surface_box_diagonal)
        self.evidence_id = evidence_id

    @property
    def certified(self) -> bool:
        return self.accuracy == "certified"

    @property
    def unresolved_count(self) -> int:
        return int(self.unresolved_issues.shape[0])

    @property
    def status_flags(self) -> AdaptiveImplicitSurfaceStatus:
        return AdaptiveImplicitSurfaceStatus(self.status)


class AdaptiveImplicitScaffold(StrictModule, NonTrainableState):
    """Retained boundary-band P1 scaffold and source-root ancestry."""

    domain: np.ndarray
    cell_indices: np.ndarray
    cell_boxes: np.ndarray
    cell_issues: np.ndarray
    cell_value_lower: np.ndarray
    cell_value_upper: np.ndarray
    cell_gradient_lower: np.ndarray
    cell_gradient_upper: np.ndarray
    cell_corner_ids: np.ndarray
    corner_ids: np.ndarray
    corner_coordinates: np.ndarray
    corner_value_lower: np.ndarray
    corner_value_upper: np.ndarray
    corner_sampled_values: np.ndarray
    corner_signs: np.ndarray
    tetrahedra: np.ndarray
    tetrahedron_cell_rows: np.ndarray
    root_edge_ids: np.ndarray
    root_brackets: np.ndarray
    root_points: np.ndarray
    root_errors: np.ndarray
    root_issues: np.ndarray
    root_endpoint_value_lower: np.ndarray
    root_endpoint_value_upper: np.ndarray
    triangle_cell_rows: np.ndarray
    triangle_root_edge_rows: np.ndarray
    target_level: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        domain: np.ndarray,
        cell_indices: np.ndarray,
        cell_boxes: np.ndarray,
        cell_issues: np.ndarray,
        cell_value_lower: np.ndarray,
        cell_value_upper: np.ndarray,
        cell_gradient_lower: np.ndarray,
        cell_gradient_upper: np.ndarray,
        cell_corner_ids: np.ndarray,
        corner_ids: np.ndarray,
        corner_coordinates: np.ndarray,
        corner_value_lower: np.ndarray,
        corner_value_upper: np.ndarray,
        corner_sampled_values: np.ndarray,
        corner_signs: np.ndarray,
        tetrahedra: np.ndarray,
        tetrahedron_cell_rows: np.ndarray,
        root_edge_ids: np.ndarray,
        root_brackets: np.ndarray,
        root_points: np.ndarray,
        root_errors: np.ndarray,
        root_issues: np.ndarray,
        root_endpoint_value_lower: np.ndarray,
        root_endpoint_value_upper: np.ndarray,
        triangle_cell_rows: np.ndarray,
        triangle_root_edge_rows: np.ndarray,
        target_level: int,
    ) -> None:
        arrays = {
            "domain": np.array(domain, dtype=np.float64, copy=True).reshape((2, 3)),
            "cell_indices": np.array(cell_indices, dtype=np.int64, copy=True).reshape(
                (-1, 4)
            ),
            "cell_boxes": np.array(cell_boxes, dtype=np.float64, copy=True).reshape(
                (-1, 2, 3)
            ),
            "cell_issues": np.array(cell_issues, dtype=np.int64, copy=True).reshape(
                (-1,)
            ),
            "cell_value_lower": np.array(
                cell_value_lower, dtype=np.float64, copy=True
            ).reshape((-1,)),
            "cell_value_upper": np.array(
                cell_value_upper, dtype=np.float64, copy=True
            ).reshape((-1,)),
            "cell_gradient_lower": np.array(
                cell_gradient_lower, dtype=np.float64, copy=True
            ).reshape((-1, 3)),
            "cell_gradient_upper": np.array(
                cell_gradient_upper, dtype=np.float64, copy=True
            ).reshape((-1, 3)),
            "cell_corner_ids": np.array(
                cell_corner_ids, dtype=np.int64, copy=True
            ).reshape((-1, 8, 3)),
            "corner_ids": np.array(corner_ids, dtype=np.int64, copy=True).reshape(
                (-1, 3)
            ),
            "corner_coordinates": np.array(
                corner_coordinates, dtype=np.float64, copy=True
            ).reshape((-1, 3)),
            "corner_value_lower": np.array(
                corner_value_lower, dtype=np.float64, copy=True
            ).reshape((-1,)),
            "corner_value_upper": np.array(
                corner_value_upper, dtype=np.float64, copy=True
            ).reshape((-1,)),
            "corner_sampled_values": np.array(
                corner_sampled_values, dtype=np.float64, copy=True
            ).reshape((-1,)),
            "corner_signs": np.array(corner_signs, dtype=np.int8, copy=True).reshape(
                (-1,)
            ),
            "tetrahedra": np.array(tetrahedra, dtype=np.int64, copy=True).reshape(
                (-1, 4)
            ),
            "tetrahedron_cell_rows": np.array(
                tetrahedron_cell_rows, dtype=np.int64, copy=True
            ).reshape((-1,)),
            "root_edge_ids": np.array(root_edge_ids, dtype=np.int64, copy=True).reshape(
                (-1, 2, 3)
            ),
            "root_brackets": np.array(root_brackets, dtype=np.float64, copy=True).reshape(
                (-1, 2, 3)
            ),
            "root_points": np.array(root_points, dtype=np.float64, copy=True).reshape(
                (-1, 3)
            ),
            "root_errors": np.array(root_errors, dtype=np.float64, copy=True).reshape(
                (-1,)
            ),
            "root_issues": np.array(root_issues, dtype=np.int64, copy=True).reshape(
                (-1,)
            ),
            "root_endpoint_value_lower": np.array(
                root_endpoint_value_lower, dtype=np.float64, copy=True
            ).reshape((-1, 2)),
            "root_endpoint_value_upper": np.array(
                root_endpoint_value_upper, dtype=np.float64, copy=True
            ).reshape((-1, 2)),
            "triangle_cell_rows": np.array(
                triangle_cell_rows, dtype=np.int64, copy=True
            ).reshape((-1,)),
            "triangle_root_edge_rows": np.array(
                triangle_root_edge_rows, dtype=np.int64, copy=True
            ).reshape((-1, 3)),
        }
        cell_count = arrays["cell_indices"].shape[0]
        corner_count = arrays["corner_ids"].shape[0]
        root_count = arrays["root_edge_ids"].shape[0]
        triangle_count = arrays["triangle_cell_rows"].shape[0]
        if (
            not np.all(np.isfinite(arrays["domain"]))
            or np.any(arrays["domain"][0] >= arrays["domain"][1])
            or arrays["cell_boxes"].shape[0] != cell_count
            or arrays["cell_issues"].shape != (cell_count,)
            or arrays["cell_value_lower"].shape != (cell_count,)
            or arrays["cell_value_upper"].shape != (cell_count,)
            or arrays["cell_gradient_lower"].shape != (cell_count, 3)
            or arrays["cell_gradient_upper"].shape != (cell_count, 3)
            or arrays["cell_corner_ids"].shape != (cell_count, 8, 3)
            or arrays["corner_coordinates"].shape != (corner_count, 3)
            or arrays["corner_value_lower"].shape != (corner_count,)
            or arrays["corner_value_upper"].shape != (corner_count,)
            or arrays["corner_sampled_values"].shape != (corner_count,)
            or arrays["corner_signs"].shape != (corner_count,)
            or arrays["tetrahedron_cell_rows"].shape != arrays["tetrahedra"].shape[:1]
            or arrays["root_brackets"].shape != (root_count, 2, 3)
            or arrays["root_points"].shape != (root_count, 3)
            or arrays["root_errors"].shape != (root_count,)
            or arrays["root_issues"].shape != (root_count,)
            or arrays["root_endpoint_value_lower"].shape != (root_count, 2)
            or arrays["root_endpoint_value_upper"].shape != (root_count, 2)
            or arrays["triangle_cell_rows"].shape != (triangle_count,)
            or arrays["triangle_root_edge_rows"].shape != (triangle_count, 3)
            or (
                arrays["tetrahedron_cell_rows"].size
                and (
                    np.min(arrays["tetrahedron_cell_rows"]) < 0
                    or np.max(arrays["tetrahedron_cell_rows"]) >= cell_count
                )
            )
            or (
                arrays["triangle_cell_rows"].size
                and (
                    np.min(arrays["triangle_cell_rows"]) < 0
                    or np.max(arrays["triangle_cell_rows"]) >= cell_count
                )
            )
            or (
                arrays["triangle_root_edge_rows"].size
                and (
                    np.min(arrays["triangle_root_edge_rows"]) < 0
                    or np.max(arrays["triangle_root_edge_rows"]) >= root_count
                )
            )
        ):
            raise ValueError(
                "The implicit boundary scaffold requires aligned IDs, intervals and ancestry."
            )
        for name, values in arrays.items():
            values.setflags(write=False)
            setattr(self, name, values)
        self.target_level = int(target_level)


@dataclass(frozen=True, slots=True)
class AdaptiveImplicitSurfaceArchiveRecipe:
    """Portable source inputs for recomputing an adaptive discovery result."""

    geometry: CompiledGeometry
    domain: np.ndarray
    policy: AdaptiveImplicitSurfacePolicy
    source_id: str

    def __post_init__(self) -> None:
        domain = np.array(self.domain, dtype=np.float64, copy=True)
        if (
            domain.shape != (2, 3)
            or not np.all(np.isfinite(domain))
            or np.any(domain[0] >= domain[1])
        ):
            raise ValueError("Adaptive archive domain must be finite and ordered.")
        domain.setflags(write=False)
        object.__setattr__(self, "domain", domain)


class AdaptiveImplicitSurface(StrictModule, NonTrainableState):
    """Adaptive discovery product: source, boundary scaffold and enclosure cover."""

    mesh: TriangleMesh | None
    evidence: AdaptiveImplicitSurfaceEvidence
    cover: AdaptiveImplicitCover | None
    topology: CertifiedImplicitTopology | None
    volume: ImplicitVolumeQuery
    scaffold: AdaptiveImplicitScaffold
    feature_edges: np.ndarray
    boundary_witness_boxes: np.ndarray
    boundary_witnesses: np.ndarray
    boundary_errors: np.ndarray
    geometry: CompiledGeometry
    policy: AdaptiveImplicitSurfacePolicy = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: TriangleMesh | None,
        evidence: AdaptiveImplicitSurfaceEvidence,
        cover: AdaptiveImplicitCover | None,
        topology: CertifiedImplicitTopology | None,
        volume: ImplicitVolumeQuery,
        scaffold: AdaptiveImplicitScaffold,
        feature_edges: np.ndarray,
        boundary_witness_boxes: np.ndarray,
        boundary_witnesses: np.ndarray,
        boundary_errors: np.ndarray,
        geometry: CompiledGeometry,
        policy: AdaptiveImplicitSurfacePolicy,
        source_id: str,
    ) -> None:
        feature_edges_ = np.array(feature_edges, dtype=np.int64, copy=True).reshape(
            (-1, 2)
        )
        witness_boxes = np.array(
            boundary_witness_boxes, dtype=np.float64, copy=True
        ).reshape((-1, 2, 3))
        witnesses = np.array(boundary_witnesses, dtype=np.float64, copy=True).reshape(
            (-1, 3)
        )
        errors = np.array(boundary_errors, dtype=np.float64, copy=True).reshape((-1,))
        if (
            witness_boxes.shape[0] != witnesses.shape[0]
            or errors.shape != (witnesses.shape[0],)
            or not np.all(np.isfinite(witness_boxes))
            or not np.all(np.isfinite(witnesses))
            or not np.all(np.isfinite(errors))
            or np.any(witness_boxes[:, 0] > witness_boxes[:, 1])
            or np.any(errors < 0.0)
        ):
            raise ValueError(
                "Adaptive surface root witnesses and their boxes must remain aligned."
            )
        for values in (feature_edges_, witness_boxes, witnesses, errors):
            values.setflags(write=False)
        self.mesh = mesh
        self.evidence = evidence
        self.cover = cover
        self.topology = topology
        self.volume = volume
        self.scaffold = scaffold
        self.feature_edges = feature_edges_
        self.boundary_witness_boxes = witness_boxes
        self.boundary_witnesses = witnesses
        self.boundary_errors = errors
        self.geometry = geometry
        self.policy = policy
        self.source_id = source_id

    @property
    def region_ids(self) -> tuple[str, ...]:
        """Expose a volume stratum only when the bounded source and cover exist."""
        if (
            self.geometry.kind is not GeometryKind.REGION
            or not self.volume.certified
            or bool(
                self.evidence.status_flags
                & AdaptiveImplicitSurfaceStatus.DOMAIN_BOUNDARY_CROSSING
            )
            or self.cover is None
            or not self.cover.established
            or self.evidence.unresolved_count
            or self.mesh is None
            or not self.mesh.topology.watertight
            or self.scaffold.triangle_cell_rows.shape[0] != self.mesh.faces.shape[0]
            or np.any(self.scaffold.corner_signs == 0)
            or np.any(self.scaffold.root_issues != 0)
            or np.any(self.scaffold.cell_issues != 0)
        ):
            return ()
        return ("interior",)

    @property
    def domain_id(self) -> str:
        """Identify the source region independently of discovery scheduling."""
        if not self.region_ids:
            raise ValueError(
                "An unestablished implicit source has no region-domain identity."
            )
        return canonical_fingerprint(
            {
                "kind": "adaptive-implicit-region-domain",
                "source_id": self.source_id,
                "geometry": array_tree_fingerprint(self.geometry),
                "bounded_domain": array_tree_fingerprint(self.volume.domain),
                "region_ids": self.region_ids,
            }
        )

    @property
    def archive_recipe(self) -> AdaptiveImplicitSurfaceArchiveRecipe:
        """Return source-only inputs; derived interval programs are intentionally omitted."""
        return AdaptiveImplicitSurfaceArchiveRecipe(
            self.geometry,
            self.volume.domain,
            self.policy,
            self.source_id,
        )

    @property
    def boundary_source(self) -> AdaptiveImplicitBoundarySource:
        return AdaptiveImplicitBoundarySource(self)


@final
class AdaptiveImplicitBoundarySource(StrictModule, NonTrainableState):
    """SourceBoundaryQuery backed by exhaustive possible-zero boxes and root witnesses."""

    surface: AdaptiveImplicitSurface
    source_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)

    def __init__(
        self, surface: AdaptiveImplicitSurface, frame_id: str | None = None
    ) -> None:
        if not isinstance(surface, AdaptiveImplicitSurface):
            raise TypeError("surface must be an AdaptiveImplicitSurface.")
        if frame_id is not None and (not isinstance(frame_id, str) or not frame_id):
            raise ValueError(
                "frame_id must be a non-empty source coordinate-frame identity."
            )
        self.surface = surface
        self.source_id = surface.source_id
        self.frame_id = "" if frame_id is None else frame_id
        self.source_revision = canonical_fingerprint(
            {
                "kind": "adaptive-implicit-boundary-source-revision",
                "source": surface.source_id,
                "geometry": array_tree_fingerprint(surface.geometry),
                "domain": array_tree_fingerprint(surface.volume.domain),
            }
        )

    @property
    def geometry(self) -> CompiledGeometry:
        return self.surface.geometry

    @property
    def ambient_dimension(self) -> int:
        return _DIMENSION

    def require_current(self) -> None:
        """Recompute the source binding against the retained adaptive discovery."""
        if (
            not isinstance(self.surface, AdaptiveImplicitSurface)
            or self.source_id != self.surface.source_id
            or self.ambient_dimension != _DIMENSION
        ):
            raise ValueError(
                "Adaptive implicit boundary query changed its discovery source."
            )
        fresh = AdaptiveImplicitBoundarySource(
            self.surface, None if not self.frame_id else self.frame_id
        )
        if fresh.source_revision != self.source_revision:
            raise ValueError(
                "Adaptive implicit boundary query no longer matches its discovery cover."
            )

    def project_boundary(
        self,
        points: ArrayLike,
        /,
        *,
        entity_ids: tuple[str, ...] | None = None,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
        frame_id: str | None = None,
    ) -> SourceBoundaryProjectionResult:
        """Return scoped interval bounds, never a false global closest-point claim."""
        from ._source_charts import project_adaptive_implicit_boundary

        selected_frame = self.frame_id if frame_id is None else frame_id
        if self.frame_id and frame_id is not None and frame_id != self.frame_id:
            raise ValueError(
                "Adaptive implicit projection frame differs from its bound source frame."
            )
        if not selected_frame:
            raise ValueError(
                "Adaptive implicit projection requires an explicitly bound source frame."
            )
        return project_adaptive_implicit_boundary(
            self,
            points,
            frame_id=selected_frame,
            entity_ids=entity_ids,
            occurrence_paths=occurrence_paths,
        )

    def _possible_boxes(self) -> np.ndarray:
        cover = self.surface.cover
        if cover is None:
            return np.empty((0, 2, 3), dtype=np.float64)
        return cover.boxes[cover.possible_zero_mask]

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("points must be finite with shape (N, 3).")
        boxes = self._possible_boxes()
        witnesses = self.surface.boundary_witnesses
        errors = self.surface.boundary_errors
        lower = np.full(values.shape[0], np.inf, dtype=np.float64)
        upper = np.full(values.shape[0], np.inf, dtype=np.float64)
        for box in boxes:
            first_gap = np.nextafter(box[0] - values, -np.inf)
            second_gap = np.nextafter(values - box[1], -np.inf)
            gap = np.maximum(np.maximum(first_gap, second_gap), 0.0)
            sum_lower = np.zeros((values.shape[0],), dtype=np.float64)
            for axis in range(3):
                squared_lower = np.nextafter(gap[:, axis] * gap[:, axis], 0.0)
                sum_lower = np.maximum(
                    np.nextafter(sum_lower + squared_lower, -np.inf),
                    0.0,
                )
            distance_lower = np.nextafter(np.sqrt(sum_lower), 0.0)
            lower = np.minimum(lower, distance_lower)
        if witnesses.shape[0]:
            for index, witness in enumerate(witnesses):
                delta = np.nextafter(np.abs(values - witness), np.inf)
                squared_upper = np.nextafter(delta * delta, np.inf)
                sum_upper = np.zeros((values.shape[0],), dtype=np.float64)
                for axis in range(3):
                    sum_upper = np.nextafter(sum_upper + squared_upper[:, axis], np.inf)
                distance_upper = np.nextafter(
                    np.sqrt(np.nextafter(sum_upper, np.inf)), np.inf
                )
                upper = np.minimum(
                    upper, np.nextafter(distance_upper + errors[index], np.inf)
                )
        semantics = (
            "certified"
            if self.surface.cover is not None
            and self.surface.cover.certified
            and self.surface.cover.complete
            else "sampled"
        )
        return SourceBoundaryDistance(lower, upper, semantics)

    def boundary_cover(self, maximum_radius: float, /) -> SourceBoundarySamples:
        """Grow the retained source cover until every physical radius is proved."""
        if isinstance(maximum_radius, (bool, np.bool_)):
            raise TypeError("maximum_radius must be a finite positive physical length.")
        requested = float(maximum_radius)
        if not math.isfinite(requested) or requested <= 0.0:
            raise ValueError("maximum_radius must be finite and positive.")
        current_samples = self._cover_from(self.surface, requested)
        if current_samples.complete:
            return current_samples
        recipe = self.surface.archive_recipe
        domain = np.asarray(recipe.domain, dtype=np.float64)
        diameter = _norm_upper(domain[1] - domain[0])
        if not math.isfinite(diameter) or diameter <= 0.0:
            raise ValueError(
                "Adaptive source domain must have a finite positive diameter."
            )
        target_relative = requested / diameter
        if not math.isfinite(target_relative) or target_relative <= 0.0:
            return self._cover_from(self.surface, requested)

        current = self.surface
        policy = current.policy
        relative = min(policy.relative_cell_diameter, target_relative)
        root_tolerance = policy.root_tolerance
        if relative >= policy.relative_cell_diameter:
            relative = policy.relative_cell_diameter * 0.5
        if relative <= 0.0 or relative >= policy.relative_cell_diameter:
            return self._cover_from(current, requested)

        while True:
            candidate_policy = replace(
                policy,
                relative_cell_diameter=relative,
                root_tolerance=root_tolerance,
            )
            candidate = discover_adaptive_implicit_surface(
                recipe.geometry,
                domain=recipe.domain,
                policy=candidate_policy,
                source_id=recipe.source_id,
            )
            samples = self._cover_from(candidate, requested)
            if samples.complete:
                return samples
            if self._cover_refinement_terminal(candidate):
                return samples
            current = candidate
            policy = candidate.policy
            next_relative = min(policy.relative_cell_diameter * 0.5, target_relative)
            next_tolerance = policy.root_tolerance * 0.5
            if (
                not math.isfinite(next_relative)
                or not math.isfinite(next_tolerance)
                or next_relative <= 0.0
                or next_tolerance <= 0.0
                or next_relative >= policy.relative_cell_diameter
                or next_tolerance >= policy.root_tolerance
            ):
                return samples
            relative = next_relative
            root_tolerance = next_tolerance

    @staticmethod
    def _cover_from(
        surface: AdaptiveImplicitSurface, requested: float, /
    ) -> SourceBoundarySamples:
        cover = surface.cover
        boxes = (
            np.empty((0, 2, 3), dtype=np.float64)
            if cover is None
            else cover.boxes[cover.possible_zero_mask]
        )
        root_points = np.asarray(surface.boundary_witnesses, dtype=np.float64)
        root_errors = np.asarray(surface.boundary_errors, dtype=np.float64)
        root_boxes = np.asarray(surface.boundary_witness_boxes, dtype=np.float64)
        root_count = root_points.shape[0]
        if (
            root_points.shape != (root_count, 3)
            or root_errors.shape != (root_count,)
            or root_boxes.shape != (root_count, 2, 3)
        ):
            raise ValueError(
                "Adaptive source witnesses and their physical boxes are incompatible."
            )
        if root_count:
            extents = np.nextafter(root_boxes[:, 1] - root_boxes[:, 0], np.inf)
            squared = np.nextafter(extents * extents, np.inf)
            root_radii = np.nextafter(
                np.sqrt(np.nextafter(np.sum(squared, axis=1), np.inf)),
                np.inf,
            )
        else:
            root_radii = np.empty((0,), dtype=np.float64)

        witness_by_box = {
            tuple(float(value) for value in box.reshape((-1,))): row
            for row, box in enumerate(root_boxes)
        }
        if len(witness_by_box) != root_count:
            raise ValueError(
                "Adaptive source witness boxes must uniquely identify their roots."
            )
        assigned_roots = np.full((boxes.shape[0],), -1, dtype=np.int64)
        covering_radius = np.full((boxes.shape[0],), np.inf, dtype=np.float64)
        if root_count and boxes.shape[0]:
            point_order = np.argsort(root_points[:, 0], kind="stable")
            ordered_x = root_points[point_order, 0]
            for box_row, box in enumerate(boxes):
                key = tuple(float(value) for value in box.reshape((-1,)))
                direct_row = witness_by_box.get(key)
                if direct_row is not None:
                    assigned_roots[box_row] = direct_row
                    covering_radius[box_row] = root_radii[direct_row]
                    continue
                first = float(np.nextafter(box[0, 0] - requested, -np.inf))
                second = float(np.nextafter(box[1, 0] + requested, np.inf))
                start = int(np.searchsorted(ordered_x, first, side="left"))
                stop = int(np.searchsorted(ordered_x, second, side="right"))
                candidate_rows = point_order[start:stop]
                if not candidate_rows.size:
                    continue
                candidate_points = root_points[candidate_rows]
                first_delta = np.nextafter(np.abs(candidate_points - box[0]), np.inf)
                second_delta = np.nextafter(np.abs(candidate_points - box[1]), np.inf)
                delta = np.maximum(first_delta, second_delta)
                squared_upper = np.nextafter(delta * delta, np.inf)
                distance_squared = np.zeros((candidate_rows.size,), dtype=np.float64)
                for axis in range(3):
                    distance_squared = np.nextafter(
                        distance_squared + squared_upper[:, axis], np.inf
                    )
                distance_upper = np.nextafter(
                    np.sqrt(np.nextafter(distance_squared, np.inf)), np.inf
                )
                best = int(np.argmin(distance_upper))
                if distance_upper[best] <= requested:
                    assigned_roots[box_row] = int(candidate_rows[best])
                    covering_radius[box_row] = float(distance_upper[best])

        covered = assigned_roots >= 0
        complete = bool(
            cover is not None
            and cover.certified
            and cover.complete
            and np.all(covered)
            and np.all(covering_radius <= requested)
        )
        semantics = "certified" if complete else "sampled"
        selected_roots = assigned_roots[covered]
        return SourceBoundarySamples(
            root_points[selected_roots],
            covering_radius[covered],
            root_errors[selected_roots],
            semantics,
            complete=complete,
        )

    @staticmethod
    def _cover_refinement_terminal(surface: AdaptiveImplicitSurface, /) -> bool:
        terminal = (
            AdaptiveImplicitSurfaceStatus.REPRESENTABILITY_LIMIT_REACHED
            | AdaptiveImplicitSurfaceStatus.NO_SURFACE
            | AdaptiveImplicitSurfaceStatus.DOMAIN_BOUNDARY_CROSSING
        )
        if surface.evidence.status_flags & terminal:
            return True
        unresolved = np.asarray(surface.evidence.unresolved_issues, dtype=np.int64)
        terminal_issues = int(
            AdaptiveImplicitBoxIssue.DOMAIN_BOUNDARY
            | AdaptiveImplicitBoxIssue.REPRESENTABILITY
            | AdaptiveImplicitBoxIssue.SINGULAR
        )
        if np.any(unresolved & terminal_issues):
            return True
        refinement_issues = int(AdaptiveImplicitBoxIssue.EDGE) | int(
            AdaptiveImplicitBoxIssue.MULTIPLE_SHEETS
        )
        refinement_mask = (unresolved & refinement_issues) != 0
        if not np.any(refinement_mask):
            return False
        boxes = np.asarray(surface.evidence.unresolved_boxes)[refinement_mask]
        return (
            max(
                (_cell_diagonal(box[0], box[1]) for box in boxes),
                default=0.0,
            )
            <= surface.policy.root_tolerance
        )


def _state_identity(geometry: CompiledGeometry) -> str:
    return canonical_fingerprint(
        {
            "kind": "adaptive-implicit-state",
            "kernel": type(geometry.kernel).__module__
            + "."
            + type(geometry.kernel).__qualname__,
            "state": array_tree_fingerprint(geometry.state),
        }
    )


def _norm_upper(vector: np.ndarray) -> float:
    raw = np.abs(np.asarray(vector, dtype=np.float64).reshape((-1,)))
    if not np.any(raw):
        return 0.0
    widened = np.nextafter(raw, np.inf)
    value = math.hypot(*widened.tolist())
    return float(np.nextafter(value, np.inf))


def _grid_coordinate_fraction(
    index: int, level: int, lower: float, upper: float
) -> Fraction:
    denominator = 1 << level
    if index <= 0:
        return Fraction(lower)
    if index >= denominator:
        return Fraction(upper)
    position = Fraction(index, denominator)
    return (1 - position) * Fraction(lower) + position * Fraction(upper)


def _grid_coordinate(index: int, level: int, lower: float, upper: float) -> float:
    return float(_grid_coordinate_fraction(index, level, lower, upper))


def _outward_coordinate(value: Fraction, *, lower: bool) -> float:
    rounded = float(value)
    if lower and Fraction(rounded) > value:
        return float(np.nextafter(rounded, -np.inf))
    if not lower and Fraction(rounded) < value:
        return float(np.nextafter(rounded, np.inf))
    return rounded


def _domain_boundary_faces(
    cell: _Cell, domain: np.ndarray, /
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Closed faces of ``cell`` that lie on the discovery domain boundary."""
    faces: list[tuple[np.ndarray, np.ndarray]] = []
    for axis in range(3):
        for side, value in ((0, cell.lower[axis]), (1, cell.upper[axis])):
            if value != domain[side, axis]:
                continue
            lower = np.array(cell.lower, dtype=np.float64, copy=True)
            upper = np.array(cell.upper, dtype=np.float64, copy=True)
            lower[axis] = upper[axis] = value
            faces.append((lower, upper))
    return tuple(faces)


def _faces_exclude_zero(
    bounds: _FieldEnclosures, faces: tuple[tuple[np.ndarray, np.ndarray], ...], /
) -> bool:
    """Whether a rigorous value enclosure separates every face from the zero set.

    A sign change inside a cell only crosses the domain boundary when the zero
    set meets one of its boundary faces; a face enclosure of one strict sign
    proves it does not.
    """
    if not bounds.value_rigorous:
        return False
    face_bounds = bounds.boxes(
        np.stack([lower for lower, _ in faces]), np.stack([upper for _, upper in faces])
    )
    lower = np.asarray(face_bounds.value_lower, dtype=np.float64)
    upper = np.asarray(face_bounds.value_upper, dtype=np.float64)
    return bool(np.all((lower > 0.0) | (upper < 0.0)))


def _cell_box(
    prefix: _GridIndex, level: int, domain: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lower_values: list[float] = []
    upper_values: list[float] = []
    for axis in range(3):
        exact_lower = _grid_coordinate_fraction(
            prefix[axis], level, domain[0, axis], domain[1, axis]
        )
        exact_upper = _grid_coordinate_fraction(
            prefix[axis] + 1, level, domain[0, axis], domain[1, axis]
        )
        lower_values.append(_outward_coordinate(exact_lower, lower=True))
        upper_values.append(_outward_coordinate(exact_upper, lower=False))
    return np.asarray(lower_values, dtype=np.float64), np.asarray(
        upper_values, dtype=np.float64
    )


def _cell_diagonal(lower: np.ndarray, upper: np.ndarray) -> float:
    return _norm_upper(upper - lower)


def _target_level(domain: np.ndarray, relative_diameter: float) -> int:
    diagonal = _norm_upper(domain[1] - domain[0])
    if not math.isfinite(diagonal):
        raise ValueError(
            "The discovery-domain diagonal is not representable in binary64."
        )
    target = relative_diameter * diagonal
    level = max(0, math.ceil(-math.log2(relative_diameter)))
    if target == 0.0:
        return level
    while float(np.nextafter(math.ldexp(diagonal, -level), np.inf)) > target:
        level += 1
    return level


def _box_corner_indices(prefix: _GridIndex) -> tuple[_GridIndex, ...]:
    return tuple(
        (
            prefix[0] + int(offset[0]),
            prefix[1] + int(offset[1]),
            prefix[2] + int(offset[2]),
        )
        for offset in _CORNER_OFFSETS
    )


def _edge_key(first: _GridIndex, second: _GridIndex) -> _GridEdge:
    return (first, second) if first <= second else (second, first)


def _root_on_edge(
    bounds: _FieldEnclosures,
    first: np.ndarray,
    second: np.ndarray,
    first_sign: int,
    second_sign: int,
    tolerance: float,
    /,
) -> tuple[np.ndarray, float, int, np.ndarray, np.ndarray, int, int]:
    endpoint_bounds = np.stack((first, second))[None, :, :]
    endpoint_signs = np.asarray(((first_sign, second_sign),), dtype=np.int8)
    return _root_on_edges(bounds, endpoint_bounds, endpoint_signs, tolerance)[0]


def _root_on_edges(
    bounds: _FieldEnclosures,
    endpoint_bounds: np.ndarray,
    endpoint_signs: np.ndarray,
    tolerance: float,
    /,
) -> tuple[tuple[np.ndarray, float, int, np.ndarray, np.ndarray, int, int], ...]:
    endpoints = np.asarray(endpoint_bounds, dtype=np.float64)
    signs = np.asarray(endpoint_signs, dtype=np.int8)
    if (
        endpoints.ndim != 3
        or endpoints.shape[1:] != (2, 3)
        or signs.shape != endpoints.shape[:2]
    ):
        raise ValueError(
            "Root-edge endpoint arrays must have shapes (N, 2, 3) and (N, 2)."
        )
    count = endpoints.shape[0]
    sign_low = signs[:, 0].copy()
    sign_high = signs[:, 1].copy()
    if np.any(sign_low == sign_high):
        raise ValueError("A root bracket must have opposite certified endpoint signs.")

    low = np.array(endpoints[:, 0], copy=True)
    high = np.array(endpoints[:, 1], copy=True)
    root_points = np.empty((count, 3), dtype=np.float64)
    root_errors = np.empty((count,), dtype=np.float64)
    root_issues = np.zeros((count,), dtype=np.int64)
    root_brackets = np.empty((count, 2, 3), dtype=np.float64)
    root_signs = np.empty((count, 2), dtype=np.int8)
    active = np.ones((count,), dtype=np.bool_)

    def record(
        rows: np.ndarray,
        points: np.ndarray,
        half_lengths: np.ndarray,
        issue: int,
        /,
    ) -> None:
        root_points[rows] = points
        root_errors[rows] = half_lengths
        root_issues[rows] = issue
        root_brackets[rows, 0] = low[rows]
        root_brackets[rows, 1] = high[rows]
        root_signs[rows, 0] = sign_low[rows]
        root_signs[rows, 1] = sign_high[rows]

    while np.any(active):
        rows = np.flatnonzero(active)
        half_lengths = np.asarray(
            [_norm_upper(0.5 * (high[row] - low[row])) for row in rows],
            dtype=np.float64,
        )
        complete = half_lengths <= tolerance
        completed_rows = rows[complete]
        if completed_rows.size:
            midpoint = low[completed_rows] + 0.5 * (
                high[completed_rows] - low[completed_rows]
            )
            record(
                completed_rows,
                midpoint,
                half_lengths[complete],
                0,
            )
            active[completed_rows] = False
        rows = rows[~complete]
        half_lengths = half_lengths[~complete]
        if not rows.size:
            continue

        midpoints = low[rows] + 0.5 * (high[rows] - low[rows])
        unrepresentable = np.all(midpoints == low[rows], axis=1) | np.all(
            midpoints == high[rows], axis=1
        )
        unrepresentable_rows = rows[unrepresentable]
        if unrepresentable_rows.size:
            record(
                unrepresentable_rows,
                midpoints[unrepresentable],
                half_lengths[unrepresentable],
                int(AdaptiveImplicitBoxIssue.REPRESENTABILITY),
            )
            active[unrepresentable_rows] = False
        rows = rows[~unrepresentable]
        half_lengths = half_lengths[~unrepresentable]
        midpoints = midpoints[~unrepresentable]
        if not rows.size:
            continue

        selected_points = np.empty((rows.size, 3), dtype=np.float64)
        selected_signs = np.zeros((rows.size,), dtype=np.int8)
        for fraction in (0.5, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 4.0, 3.0 / 4.0):
            pending = np.flatnonzero(selected_signs == 0)
            if not pending.size:
                break
            pending_rows = rows[pending]
            candidates = low[pending_rows] + fraction * (
                high[pending_rows] - low[pending_rows]
            )
            interior = np.any(candidates != low[pending_rows], axis=1) & np.any(
                candidates != high[pending_rows], axis=1
            )
            query_rows = pending[interior]
            if not query_rows.size:
                continue
            candidate_points = candidates[interior]
            value_lower, value_upper = bounds.point_values(candidate_points)
            candidate_signs = np.zeros((query_rows.size,), dtype=np.int8)
            candidate_signs[value_upper < 0.0] = -1
            candidate_signs[value_lower > 0.0] = 1
            selected = candidate_signs != 0
            selected_rows = query_rows[selected]
            if selected_rows.size:
                selected_points[selected_rows] = candidate_points[selected]
                selected_signs[selected_rows] = candidate_signs[selected]

        resolved = selected_signs != 0
        resolved_rows = rows[resolved]
        if resolved_rows.size:
            same_sign = selected_signs[resolved] == sign_low[resolved_rows]
            low_rows = resolved_rows[same_sign]
            high_rows = resolved_rows[~same_sign]
            low[low_rows] = selected_points[resolved][same_sign]
            sign_low[low_rows] = selected_signs[resolved][same_sign]
            high[high_rows] = selected_points[resolved][~same_sign]
            sign_high[high_rows] = selected_signs[resolved][~same_sign]
        unresolved_rows = rows[~resolved]
        if unresolved_rows.size:
            record(
                unresolved_rows,
                midpoints[~resolved],
                half_lengths[~resolved],
                int(AdaptiveImplicitBoxIssue.PHYSICAL_RESOLUTION),
            )
            active[unresolved_rows] = False

    return tuple(
        (
            root_points[row],
            float(root_errors[row]),
            int(root_issues[row]),
            root_brackets[row, 0],
            root_brackets[row, 1],
            int(root_signs[row, 0]),
            int(root_signs[row, 1]),
        )
        for row in range(count)
    )


def _grid_point(index: _GridIndex, level: int, domain: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            _grid_coordinate(index[axis], level, domain[0, axis], domain[1, axis])
            for axis in range(3)
        ],
        dtype=np.float64,
    )


def _triangle_area(
    vertices: list[np.ndarray], triangle: tuple[int, int, int], /
) -> float:
    first, second, third = (vertices[index] for index in triangle)
    return 0.5 * float(np.linalg.norm(np.cross(second - first, third - first)))


def _triangulate_polygon(
    vertices: list[np.ndarray], cycle: list[int], /
) -> tuple[tuple[int, int, int], ...] | None:
    if len(cycle) == 3:
        return ((cycle[0], cycle[1], cycle[2]),)
    if len(cycle) == 4:
        first_split = (
            (cycle[0], cycle[1], cycle[2]),
            (cycle[0], cycle[2], cycle[3]),
        )
        second_split = (
            (cycle[0], cycle[1], cycle[3]),
            (cycle[1], cycle[2], cycle[3]),
        )
        first_minimum = min(_triangle_area(vertices, row) for row in first_split)
        second_minimum = min(_triangle_area(vertices, row) for row in second_split)
        return first_split if first_minimum >= second_minimum else second_split

    points = np.asarray([vertices[index] for index in cycle], dtype=np.float64)
    center = np.mean(points, axis=0)
    centered = points - center
    normal = np.sum(np.cross(centered, np.roll(centered, -1, axis=0)), axis=0)
    normal_length = float(np.linalg.norm(normal))
    if not math.isfinite(normal_length) or normal_length == 0.0:
        return None
    normal /= normal_length
    reference = centered[0] - normal * float(np.dot(centered[0], normal))
    reference_length = float(np.linalg.norm(reference))
    if reference_length == 0.0:
        axis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(normal)))]
        reference = np.cross(normal, axis)
        reference_length = float(np.linalg.norm(reference))
    if not math.isfinite(reference_length) or reference_length == 0.0:
        return None
    first_axis = reference / reference_length
    second_axis = np.cross(normal, first_axis)
    planar = np.stack((centered @ first_axis, centered @ second_axis), axis=1)

    def orient(first: int, second: int, third: int, /) -> float:
        a, b, c = planar[[first, second, third]]
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    signed_area_twice = sum(
        planar[index, 0] * planar[(index + 1) % len(cycle), 1]
        - planar[index, 1] * planar[(index + 1) % len(cycle), 0]
        for index in range(len(cycle))
    )
    if not math.isfinite(float(signed_area_twice)) or signed_area_twice == 0.0:
        return None
    orientation = 1.0 if signed_area_twice > 0.0 else -1.0
    remaining = list(range(len(cycle)))
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        ears: list[tuple[float, int, tuple[int, int, int]]] = []
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following = remaining[(position + 1) % len(remaining)]
            if orientation * orient(previous, current, following) <= 0.0:
                continue
            contains_vertex = False
            for candidate in remaining:
                if candidate in (previous, current, following):
                    continue
                first_side = orientation * orient(previous, current, candidate)
                second_side = orientation * orient(current, following, candidate)
                third_side = orientation * orient(following, previous, candidate)
                if min(first_side, second_side, third_side) >= 0.0:
                    contains_vertex = True
                    break
            if contains_vertex:
                continue
            triangle = (
                cycle[previous],
                cycle[current],
                cycle[following],
            )
            ears.append((_triangle_area(vertices, triangle), position, triangle))
        if not ears:
            return None
        _area, position, triangle = max(
            ears,
            key=lambda row: (row[0], cycle[remaining[row[1]]]),
        )
        triangles.append(triangle)
        del remaining[position]
    triangles.append(
        (
            cycle[remaining[0]],
            cycle[remaining[1]],
            cycle[remaining[2]],
        )
    )
    return tuple(triangles)


def _triangulate_cell_band(
    cell: _Cell,
    corner_indices: tuple[_GridIndex, ...],
    signs: dict[_GridIndex, int],
    point_values: dict[_GridIndex, float],
    face_center_signs: dict[_GridFaceKey, int],
    root_cache: dict[
        _GridEdge, tuple[np.ndarray, float, int, np.ndarray, np.ndarray, int, int]
    ],
    vertices: list[np.ndarray],
    vertex_errors: list[float],
    vertex_keys: dict[_GridEdge, int],
    faces: list[tuple[int, int, int]],
    edge_keys_by_vertex: list[_GridEdge],
    triangle_cell_rows: list[int],
    cell_row: int,
    policy: AdaptiveImplicitSurfacePolicy,
) -> tuple[int, np.ndarray, float]:
    issue = 0
    cell_signs = np.asarray([signs[index] for index in corner_indices], dtype=np.int8)
    cell_values = np.asarray(
        [point_values[index] for index in corner_indices], dtype=np.float64
    )
    first_witness: np.ndarray | None = None
    first_error = 0.0

    def root_vertex(first_corner: int, second_corner: int, /) -> int:
        nonlocal issue, first_witness, first_error
        first_key, second_key = (
            corner_indices[first_corner],
            corner_indices[second_corner],
        )
        key = _edge_key(first_key, second_key)
        root = root_cache.get(key)
        if root is None:
            raise RuntimeError(
                "Every crossing cube edge must have a batched root record."
            )
        point, error, root_issue = root[:3]
        issue |= root_issue
        if first_witness is None:
            first_witness, first_error = point, error
        vertex = vertex_keys.get(key)
        if vertex is None:
            vertex = len(vertices)
            vertex_keys[key] = vertex
            edge_keys_by_vertex.append(key)
            vertices.append(point)
            vertex_errors.append(error)
        return vertex

    segments: list[tuple[int, int]] = []
    segment_keys: set[tuple[int, int]] = set()

    def connect(first_vertex: int, second_vertex: int, /) -> None:
        nonlocal issue
        if first_vertex == second_vertex:
            issue |= int(AdaptiveImplicitBoxIssue.MULTIPLE_SHEETS)
            return
        key = (
            (first_vertex, second_vertex)
            if first_vertex < second_vertex
            else (second_vertex, first_vertex)
        )
        if key not in segment_keys:
            segment_keys.add(key)
            segments.append(key)

    for face in _CUBE_FACES:
        face_roots: dict[int, int] = {}
        for edge_position in range(4):
            first_corner = face[edge_position]
            second_corner = face[(edge_position + 1) % 4]
            if cell_signs[first_corner] != cell_signs[second_corner]:
                face_roots[edge_position] = root_vertex(first_corner, second_corner)
        crossing_count = len(face_roots)
        if crossing_count == 0:
            continue
        if crossing_count == 2:
            first_position, second_position = sorted(face_roots)
            connect(face_roots[first_position], face_roots[second_position])
            continue
        if crossing_count != 4:
            issue |= int(AdaptiveImplicitBoxIssue.MULTIPLE_SHEETS)
            continue
        face_key = tuple(sorted(corner_indices[corner] for corner in face))
        center_sign = face_center_signs.get(face_key, 0)
        if center_sign not in (-1, 1):
            issue |= int(AdaptiveImplicitBoxIssue.EDGE)
            continue
        for corner_position, corner in enumerate(face):
            if int(cell_signs[corner]) == center_sign:
                continue
            previous_edge = (corner_position - 1) % 4
            next_edge = corner_position
            connect(face_roots[previous_edge], face_roots[next_edge])

    if not segments:
        issue |= int(AdaptiveImplicitBoxIssue.EDGE)
        witness = (
            np.zeros(3, dtype=np.float64) if first_witness is None else first_witness
        )
        return issue, witness, first_error

    adjacency: dict[int, list[int]] = {}
    for first_vertex, second_vertex in segments:
        adjacency.setdefault(first_vertex, []).append(second_vertex)
        adjacency.setdefault(second_vertex, []).append(first_vertex)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        issue |= int(AdaptiveImplicitBoxIssue.MULTIPLE_SHEETS)
        witness = (
            np.zeros(3, dtype=np.float64) if first_witness is None else first_witness
        )
        return issue, witness, first_error

    def vertex_order(vertex: int, /) -> _GridEdge:
        return edge_keys_by_vertex[vertex]

    unvisited = set(segments)
    cycles: list[list[int]] = []
    while unvisited:
        first_edge = min(
            unvisited,
            key=lambda edge: (
                vertex_order(edge[0]),
                vertex_order(edge[1]),
            ),
        )
        start = min(first_edge, key=vertex_order)
        current = first_edge[1] if first_edge[0] == start else first_edge[0]
        unvisited.remove(first_edge)
        cycle = [start, current]
        previous = start
        while current != start:
            neighbors = adjacency[current]
            next_vertices = [vertex for vertex in neighbors if vertex != previous]
            if len(next_vertices) != 1:
                issue |= int(AdaptiveImplicitBoxIssue.MULTIPLE_SHEETS)
                break
            next_vertex = next_vertices[0]
            edge = (
                (current, next_vertex)
                if current < next_vertex
                else (next_vertex, current)
            )
            if edge not in unvisited:
                issue |= int(AdaptiveImplicitBoxIssue.MULTIPLE_SHEETS)
                break
            unvisited.remove(edge)
            if next_vertex != start:
                cycle.append(next_vertex)
            previous, current = current, next_vertex
        if issue & int(AdaptiveImplicitBoxIssue.MULTIPLE_SHEETS):
            break
        if len(cycle) < 3:
            issue |= int(AdaptiveImplicitBoxIssue.EDGE)
            break
        cycles.append(cycle)

    if issue & int(AdaptiveImplicitBoxIssue.MULTIPLE_SHEETS):
        witness = (
            np.zeros(3, dtype=np.float64) if first_witness is None else first_witness
        )
        return issue, witness, first_error
    if len(cycles) != 1:
        issue |= int(AdaptiveImplicitBoxIssue.MULTIPLE_SHEETS)
        witness = (
            np.zeros(3, dtype=np.float64) if first_witness is None else first_witness
        )
        return issue, witness, first_error

    gradient = np.empty((3,), dtype=np.float64)
    for axis in range(3):
        lower_mean = float(np.mean(cell_values[_CORNER_OFFSETS[:, axis] == 0]))
        upper_mean = float(np.mean(cell_values[_CORNER_OFFSETS[:, axis] == 1]))
        gradient[axis] = (upper_mean - lower_mean) / (cell.upper[axis] - cell.lower[axis])
    if not np.all(np.isfinite(gradient)) or float(np.linalg.norm(gradient)) == 0.0:
        issue |= int(AdaptiveImplicitBoxIssue.SINGULAR)
        witness = (
            np.zeros(3, dtype=np.float64) if first_witness is None else first_witness
        )
        return issue, witness, first_error

    cycle = cycles[0]
    cycle_points = np.asarray([vertices[vertex] for vertex in cycle])
    cycle_center = np.mean(cycle_points, axis=0)
    centered_points = cycle_points - cycle_center
    cycle_normal = np.sum(
        np.cross(centered_points, np.roll(centered_points, -1, axis=0)),
        axis=0,
    )
    orientation = float(np.dot(cycle_normal, gradient))
    if not math.isfinite(orientation) or orientation == 0.0:
        issue |= int(AdaptiveImplicitBoxIssue.SINGULAR)
        witness = (
            np.zeros(3, dtype=np.float64) if first_witness is None else first_witness
        )
        return issue, witness, first_error
    if orientation < 0.0:
        cycle.reverse()
    triangles = _triangulate_polygon(vertices, cycle)
    if triangles is None:
        issue |= int(AdaptiveImplicitBoxIssue.MULTIPLE_SHEETS)
        witness = (
            np.zeros(3, dtype=np.float64) if first_witness is None else first_witness
        )
        return issue, witness, first_error
    for triangle in triangles:
        first_point, second_point, third_point = (vertices[vertex] for vertex in triangle)
        normal = np.cross(second_point - first_point, third_point - first_point)
        area = 0.5 * float(np.linalg.norm(normal))
        if not math.isfinite(area) or area <= policy.minimum_face_area:
            issue |= int(AdaptiveImplicitBoxIssue.FACE)
            continue
        if float(np.dot(normal, gradient)) < 0.0:
            triangle = (triangle[0], triangle[2], triangle[1])
        faces.append(triangle)
        triangle_cell_rows.append(cell_row)
    witness = np.zeros(3, dtype=np.float64) if first_witness is None else first_witness
    return issue, witness, first_error


def _cover_identity(
    boxes: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    domain: np.ndarray,
    source_id: str,
    state_id: str,
    enclosure: str,
    complete: bool,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "adaptive-implicit-cover",
            "boxes": array_tree_fingerprint(boxes),
            "lower": array_tree_fingerprint(lower),
            "upper": array_tree_fingerprint(upper),
            "domain": array_tree_fingerprint(domain),
            "source": source_id,
            "state": state_id,
            "enclosure": enclosure,
            "complete": complete,
        }
    )


@checked
def _discover_adaptive_implicit_surface_once(
    geometry: CompiledGeometry,
    /,
    *,
    domain: ArrayLike,
    policy: AdaptiveImplicitSurfacePolicy = AdaptiveImplicitSurfacePolicy(),
    source_id: str,
) -> AdaptiveImplicitSurface:
    """Exhaustively enclose and polygonize a region's implicit zero set.

    Possible-zero leaves are resolved to the requested physical cell diameter.
    Pure edge-ambiguous leaves trigger finer dyadic discovery until their roots
    are witnessed or their diameter reaches the requested root tolerance.
    Singular and otherwise unresolved leaves remain explicit.
    """
    if not isinstance(geometry, CompiledGeometry):
        raise TypeError("geometry must be a CompiledGeometry.")
    if not isinstance(policy, AdaptiveImplicitSurfacePolicy):
        raise TypeError("policy must be an AdaptiveImplicitSurfacePolicy.")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be a non-empty string.")
    if geometry.ambient_dimension != 3 or geometry.kind is not GeometryKind.REGION:
        raise ValueError(
            "Adaptive implicit surface discovery requires a three-dimensional region."
        )
    if not bool(np.asarray(geometry.validity().accepted)):
        raise ValueError("geometry must have accepted validity evidence.")
    certificate = geometry.field_certificate
    if certificate.sign_reliability is SignReliability.UNRELIABLE:
        raise ValueError("Adaptive discovery requires reliable or heuristic field signs.")
    if (
        certificate.zero_set_accuracy is ZeroSetAccuracy.APPROXIMATE
        and not policy.allow_approximate_zero_set
    ):
        raise ValueError("The policy must explicitly admit an approximate zero set.")
    if (
        certificate.regularity is FieldRegularity.NONSMOOTH
        and not policy.allow_nonsmooth_field
    ):
        raise ValueError("The policy must explicitly admit a nonsmooth field.")
    if (
        policy.enclosure != "sampled"
        and certificate.sign_reliability is not SignReliability.RELIABLE
    ):
        raise ValueError("A rigorous enclosure requires reliable field signs.")
    if policy.enclosure == "lipschitz" and (
        certificate.lipschitz_upper_bound is None or certificate.evaluation_error is None
    ):
        raise ValueError(
            "Lipschitz enclosure requires declared Lipschitz and evaluation-error bounds."
        )
    domain_ = np.asarray(domain, dtype=np.float64)
    if (
        domain_.shape != (2, 3)
        or not np.all(np.isfinite(domain_))
        or np.any(domain_[0] >= domain_[1])
    ):
        raise ValueError(
            "domain must contain finite lower and upper corners with shape (2, 3)."
        )
    if not np.all(np.isfinite(domain_[1] - domain_[0])):
        raise ValueError("domain extents must be representable in binary64.")
    bounds = _FieldEnclosures(geometry, policy.enclosure)
    state_id = _state_identity(geometry)
    geometry_bounds = np.asarray(geometry.bounds, dtype=np.float64)
    source_in_domain = (
        geometry_bounds.shape == (2, 3)
        and np.all(np.isfinite(geometry_bounds))
        and np.all(domain_[0] <= geometry_bounds[0])
        and np.all(domain_[1] >= geometry_bounds[1])
    )
    target_level = _target_level(domain_, policy.relative_cell_diameter)
    frontier: list[_GridIndex] = [(0, 0, 0)]
    level = 0
    cover_cells: list[_Cell] = []
    surface_cells: list[_Cell] = []
    deepest = 0
    status = 0
    while frontier:
        deepest = max(deepest, level)
        next_frontier: list[_GridIndex] = []
        for start in range(0, len(frontier), _INTERVAL_BATCH_SIZE):
            prefixes = frontier[start : start + _INTERVAL_BATCH_SIZE]
            valid_prefixes: list[_GridIndex] = []
            lower_rows: list[np.ndarray] = []
            upper_rows: list[np.ndarray] = []
            for prefix in prefixes:
                lower, upper = _cell_box(prefix, level, domain_)
                if np.any(lower >= upper):
                    cover_cells.append(
                        _Cell(
                            level,
                            prefix,
                            lower,
                            upper,
                            -math.inf,
                            math.inf,
                            np.full(3, -math.inf),
                            np.full(3, math.inf),
                            int(AdaptiveImplicitBoxIssue.REPRESENTABILITY),
                        )
                    )
                    continue
                valid_prefixes.append(prefix)
                lower_rows.append(lower)
                upper_rows.append(upper)
            if not valid_prefixes:
                continue
            box_bounds = bounds.boxes(np.stack(lower_rows), np.stack(upper_rows))
            for row, prefix in enumerate(valid_prefixes):
                lower, upper = lower_rows[row], upper_rows[row]
                lo = float(box_bounds.value_lower[row])
                hi = float(box_bounds.value_upper[row])
                gradients_low = box_bounds.gradient_lower[row]
                gradients_high = box_bounds.gradient_upper[row]
                cell = _Cell(
                    level,
                    prefix,
                    lower,
                    upper,
                    lo,
                    hi,
                    gradients_low,
                    gradients_high,
                )
                if hi < 0.0 or lo > 0.0:
                    cover_cells.append(cell)
                    continue
                if level < target_level:
                    lower_sample = np.asarray(
                        [
                            _grid_coordinate(
                                prefix[axis],
                                level,
                                domain_[0, axis],
                                domain_[1, axis],
                            )
                            for axis in range(3)
                        ],
                        dtype=np.float64,
                    )
                    upper_sample = np.asarray(
                        [
                            _grid_coordinate(
                                prefix[axis] + 1,
                                level,
                                domain_[0, axis],
                                domain_[1, axis],
                            )
                            for axis in range(3)
                        ],
                        dtype=np.float64,
                    )
                    middle_sample = np.asarray(
                        [
                            _grid_coordinate(
                                2 * prefix[axis] + 1,
                                level + 1,
                                domain_[0, axis],
                                domain_[1, axis],
                            )
                            for axis in range(3)
                        ],
                        dtype=np.float64,
                    )
                    if np.any(middle_sample <= lower_sample) or np.any(
                        middle_sample >= upper_sample
                    ):
                        issue = int(AdaptiveImplicitBoxIssue.REPRESENTABILITY)
                        unresolved_cell = _Cell(
                            level,
                            prefix,
                            lower,
                            upper,
                            lo,
                            hi,
                            gradients_low,
                            gradients_high,
                            issue,
                        )
                        cover_cells.append(unresolved_cell)
                        surface_cells.append(unresolved_cell)
                        status |= int(
                            AdaptiveImplicitSurfaceStatus.REPRESENTABILITY_LIMIT_REACHED
                        )
                        continue
                    for bits in itertools.product((0, 1), repeat=3):
                        next_frontier.append(
                            (
                                2 * prefix[0] + bits[0],
                                2 * prefix[1] + bits[1],
                                2 * prefix[2] + bits[2],
                            )
                        )
                    continue
                cover_cells.append(cell)
                surface_cells.append(cell)
        frontier = next_frontier
        level += 1

    unresolved: dict[tuple[int, _GridIndex], int] = {
        (cell.level, cell.lower_index): cell.issue for cell in surface_cells if cell.issue
    }
    mesh_cells = [
        cell for cell in surface_cells if not cell.issue and cell.level == target_level
    ]
    corner_indices_by_cell = [
        _box_corner_indices(cell.lower_index) for cell in mesh_cells
    ]
    unique_corner_indices = sorted(
        {point for cell_indices in corner_indices_by_cell for point in cell_indices}
    )
    corner_points = np.asarray(
        [_grid_point(index, target_level, domain_) for index in unique_corner_indices],
        dtype=np.float64,
    )
    point_lower, point_upper = (
        bounds.point_values(corner_points)
        if corner_points.size
        else (
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    )
    point_signs: dict[_GridIndex, int] = {}
    point_values: dict[_GridIndex, float] = {}
    for index, lower_value, upper_value in zip(
        unique_corner_indices, point_lower, point_upper, strict=True
    ):
        if upper_value < 0.0:
            point_signs[index] = -1
        elif lower_value > 0.0:
            point_signs[index] = 1
        else:
            point_signs[index] = 0
    if unique_corner_indices:
        sampled_values = bounds.field_values(corner_points)
        point_values.update(
            zip(unique_corner_indices, sampled_values.tolist(), strict=True)
        )
    corner_row_by_index = {index: row for row, index in enumerate(unique_corner_indices)}
    face_center_points: dict[_GridFaceKey, np.ndarray] = {}
    for corners in corner_indices_by_cell:
        corner_signs = tuple(point_signs[index] for index in corners)
        if any(sign == 0 for sign in corner_signs):
            continue
        for face in _CUBE_FACES:
            crossing_count = sum(
                corner_signs[face[position]] != corner_signs[face[(position + 1) % 4]]
                for position in range(4)
            )
            if crossing_count != 4:
                continue
            key = tuple(sorted(corners[position] for position in face))
            if key not in face_center_points:
                face_rows = [corner_row_by_index[corners[position]] for position in face]
                face_center_points[key] = np.mean(corner_points[face_rows], axis=0)
    face_center_keys = tuple(sorted(face_center_points))
    if face_center_keys:
        face_center_coordinates = np.asarray(
            [face_center_points[key] for key in face_center_keys],
            dtype=np.float64,
        )
        face_lower, face_upper = bounds.point_values(face_center_coordinates)
        face_center_signs = {
            key: -1 if upper < 0.0 else 1 if lower > 0.0 else 0
            for key, lower, upper in zip(
                face_center_keys, face_lower, face_upper, strict=True
            )
        }
    else:
        face_center_signs = {}

    # A positive interval box is only a possible zero. Its corner signs supply
    # an independent root witness; absence of one is retained as unresolved.
    for cell, corners in zip(mesh_cells, corner_indices_by_cell, strict=True):
        key = (cell.level, cell.lower_index)
        signs = [point_signs[index] for index in corners]
        if any(sign == 0 for sign in signs):
            unresolved[key] = int(AdaptiveImplicitBoxIssue.SINGULAR)
            continue
        if not (any(sign < 0 for sign in signs) and any(sign > 0 for sign in signs)):
            gradient_has_component = np.any(
                (cell.gradient_lower > 0.0) | (cell.gradient_upper < 0.0)
            )
            unresolved[key] = int(
                AdaptiveImplicitBoxIssue.EDGE
                if gradient_has_component
                else AdaptiveImplicitBoxIssue.SINGULAR
            )
            continue
        boundary_faces = _domain_boundary_faces(cell, domain_)
        if boundary_faces and not _faces_exclude_zero(bounds, boundary_faces):
            unresolved[key] = int(AdaptiveImplicitBoxIssue.DOMAIN_BOUNDARY)
            status |= int(AdaptiveImplicitSurfaceStatus.DOMAIN_BOUNDARY_CROSSING)

    root_cache: dict[
        _GridEdge, tuple[np.ndarray, float, int, np.ndarray, np.ndarray, int, int]
    ] = {}
    vertices: list[np.ndarray] = []
    vertex_errors: list[float] = []
    vertex_keys: dict[_GridEdge, int] = {}
    edge_keys_by_vertex: list[_GridEdge] = []
    triangle_cell_rows: list[int] = []
    faces: list[tuple[int, int, int]] = []
    boundary_witnesses: dict[
        tuple[int, _GridIndex], tuple[np.ndarray, float, np.ndarray]
    ] = {}
    required_edges = sorted(
        {
            _edge_key(corners[first], corners[second])
            for cell, corners in zip(mesh_cells, corner_indices_by_cell, strict=True)
            if (cell.level, cell.lower_index) not in unresolved
            for first, second in _CUBE_EDGES.tolist()
            if point_signs[corners[first]] != point_signs[corners[second]]
        }
    )
    if required_edges:
        endpoints = np.asarray(
            [
                (
                    _grid_point(edge[0], target_level, domain_),
                    _grid_point(edge[1], target_level, domain_),
                )
                for edge in required_edges
            ],
            dtype=np.float64,
        )
        endpoint_signs = np.asarray(
            [(point_signs[edge[0]], point_signs[edge[1]]) for edge in required_edges],
            dtype=np.int8,
        )
        root_records = _root_on_edges(
            bounds,
            endpoints,
            endpoint_signs,
            policy.root_tolerance,
        )
        root_cache.update(
            {edge: root for edge, root in zip(required_edges, root_records, strict=True)}
        )
    for cell_row, (cell, corners) in enumerate(
        zip(mesh_cells, corner_indices_by_cell, strict=True)
    ):
        cell_key = (cell.level, cell.lower_index)
        if cell_key in unresolved:
            continue
        cube_signs = [point_signs[index] for index in corners]
        crossing_edges = [
            (int(first), int(second))
            for first, second in _CUBE_EDGES
            if cube_signs[first] != cube_signs[second]
        ]
        roots = [
            root_cache[_edge_key(corners[first], corners[second])]
            for first, second in crossing_edges
        ]
        if not roots:
            unresolved[cell_key] = int(AdaptiveImplicitBoxIssue.EDGE)
            continue
        first_root = roots[0]
        boundary_witnesses[cell_key] = (
            first_root[0],
            first_root[1],
            np.stack((cell.lower, cell.upper)),
        )
        issue, _, _ = _triangulate_cell_band(
            cell,
            corners,
            point_signs,
            point_values,
            face_center_signs,
            root_cache,
            vertices,
            vertex_errors,
            vertex_keys,
            faces,
            edge_keys_by_vertex,
            triangle_cell_rows,
            cell_row,
            policy,
        )
        if issue:
            unresolved[cell_key] = issue

    unresolved_lookup = set(unresolved)
    witness_rows = [boundary_witnesses[key] for key in sorted(boundary_witnesses)]
    if witness_rows:
        witness_values = np.asarray(
            [row[0] for row in witness_rows], dtype=np.float64
        ).reshape((-1, 3))
        witness_errors = np.asarray([row[1] for row in witness_rows], dtype=np.float64)
        witness_boxes = np.asarray(
            [row[2] for row in witness_rows], dtype=np.float64
        ).reshape((-1, 2, 3))
    else:
        witness_values = np.empty((0, 3), dtype=np.float64)
        witness_errors = np.empty((0,), dtype=np.float64)
        witness_boxes = np.empty((0, 2, 3), dtype=np.float64)

    unresolved_boxes = np.asarray(
        [
            np.stack((cell.lower, cell.upper))
            for cell in surface_cells
            if (cell.level, cell.lower_index) in unresolved_lookup
        ],
        dtype=np.float64,
    ).reshape((-1, 2, 3))
    unresolved_issues = np.asarray(
        [
            unresolved[(cell.level, cell.lower_index)]
            for cell in surface_cells
            if (cell.level, cell.lower_index) in unresolved_lookup
        ],
        dtype=np.int64,
    )
    if unresolved_issues.size:
        status |= int(AdaptiveImplicitSurfaceStatus.UNRESOLVED_BOXES)
        if np.any(unresolved_issues & int(AdaptiveImplicitBoxIssue.REPRESENTABILITY)):
            status |= int(AdaptiveImplicitSurfaceStatus.REPRESENTABILITY_LIMIT_REACHED)
        physical_issues = int(
            AdaptiveImplicitBoxIssue.PHYSICAL_RESOLUTION
            | AdaptiveImplicitBoxIssue.SINGULAR
            | AdaptiveImplicitBoxIssue.EDGE
            | AdaptiveImplicitBoxIssue.FACE
            | AdaptiveImplicitBoxIssue.MULTIPLE_SHEETS
        )
        if np.any(unresolved_issues & physical_issues):
            status |= int(AdaptiveImplicitSurfaceStatus.PHYSICAL_RESOLUTION_REACHED)
    if not surface_cells and source_in_domain:
        status |= int(AdaptiveImplicitSurfaceStatus.NO_SURFACE)
    if not source_in_domain:
        status |= int(AdaptiveImplicitSurfaceStatus.DOMAIN_BOUNDARY_CROSSING)

    mesh: TriangleMesh | None = None
    if faces and not unresolved_issues.size and source_in_domain:
        if len(vertices) > np.iinfo(np.int32).max:
            status |= int(AdaptiveImplicitSurfaceStatus.DEGENERATE_EXTRACTION)
        else:
            try:
                candidate_mesh = TriangleMesh(
                    np.asarray(vertices, dtype=np.float64),
                    np.asarray(faces, dtype=np.int32),
                    source_id=source_id,
                )
                if candidate_mesh.topology.watertight:
                    mesh = candidate_mesh
                else:
                    status |= int(AdaptiveImplicitSurfaceStatus.NONMANIFOLD_EXTRACTION)
            except (ValueError, OverflowError):
                status |= int(AdaptiveImplicitSurfaceStatus.DEGENERATE_EXTRACTION)

    cell_corner_rows = np.asarray(
        [
            [corner_row_by_index[index] for index in corners]
            for corners in corner_indices_by_cell
        ],
        dtype=np.int64,
    ).reshape((-1, 8))
    tetrahedra = np.asarray(
        [
            [cell_corner_rows[cell_row, local_corner] for local_corner in tetra]
            for cell_row in range(len(mesh_cells))
            for tetra in _TETRAHEDRA
        ],
        dtype=np.int64,
    ).reshape((-1, 4))
    tetrahedron_cell_rows = np.repeat(
        np.arange(len(mesh_cells), dtype=np.int64), len(_TETRAHEDRA)
    )
    root_edge_ids = sorted(root_cache)
    root_row_by_edge = {edge: row for row, edge in enumerate(root_edge_ids)}
    root_rows = [root_cache[edge] for edge in root_edge_ids]
    triangle_root_edge_rows = np.asarray(
        [
            [root_row_by_edge[edge_keys_by_vertex[vertex]] for vertex in triangle]
            for triangle in faces
        ],
        dtype=np.int64,
    ).reshape((-1, 3))
    scaffold = AdaptiveImplicitScaffold(
        domain=domain_,
        cell_indices=np.asarray(
            [(cell.level, *cell.lower_index) for cell in mesh_cells], dtype=np.int64
        ).reshape((-1, 4)),
        cell_boxes=np.asarray(
            [np.stack((cell.lower, cell.upper)) for cell in mesh_cells], dtype=np.float64
        ).reshape((-1, 2, 3)),
        cell_issues=np.asarray(
            [unresolved.get((cell.level, cell.lower_index), 0) for cell in mesh_cells],
            dtype=np.int64,
        ),
        cell_value_lower=np.asarray(
            [cell.value_lower for cell in mesh_cells], dtype=np.float64
        ),
        cell_value_upper=np.asarray(
            [cell.value_upper for cell in mesh_cells], dtype=np.float64
        ),
        cell_gradient_lower=np.asarray(
            [cell.gradient_lower for cell in mesh_cells], dtype=np.float64
        ).reshape((-1, 3)),
        cell_gradient_upper=np.asarray(
            [cell.gradient_upper for cell in mesh_cells], dtype=np.float64
        ).reshape((-1, 3)),
        cell_corner_ids=np.asarray(corner_indices_by_cell, dtype=np.int64).reshape(
            (-1, 8, 3)
        ),
        corner_ids=np.asarray(unique_corner_indices, dtype=np.int64).reshape((-1, 3)),
        corner_coordinates=corner_points,
        corner_value_lower=point_lower,
        corner_value_upper=point_upper,
        corner_sampled_values=np.asarray(
            [point_values[index] for index in unique_corner_indices], dtype=np.float64
        ),
        corner_signs=np.asarray(
            [point_signs[index] for index in unique_corner_indices], dtype=np.int8
        ),
        tetrahedra=tetrahedra,
        tetrahedron_cell_rows=tetrahedron_cell_rows,
        root_edge_ids=np.asarray(root_edge_ids, dtype=np.int64).reshape((-1, 2, 3)),
        root_brackets=np.asarray(
            [[root[3], root[4]] for root in root_rows], dtype=np.float64
        ).reshape((-1, 2, 3)),
        root_points=np.asarray([root[0] for root in root_rows], dtype=np.float64).reshape(
            (-1, 3)
        ),
        root_errors=np.asarray([root[1] for root in root_rows], dtype=np.float64),
        root_issues=np.asarray([root[2] for root in root_rows], dtype=np.int64),
        root_endpoint_value_lower=np.asarray(
            [
                [point_lower[corner_row_by_index[index]] for index in edge]
                for edge in root_edge_ids
            ],
            dtype=np.float64,
        ).reshape((-1, 2)),
        root_endpoint_value_upper=np.asarray(
            [
                [point_upper[corner_row_by_index[index]] for index in edge]
                for edge in root_edge_ids
            ],
            dtype=np.float64,
        ).reshape((-1, 2)),
        triangle_cell_rows=np.asarray(triangle_cell_rows, dtype=np.int64),
        triangle_root_edge_rows=triangle_root_edge_rows,
        target_level=target_level,
    )
    cover_boxes = np.asarray(
        [np.stack((cell.lower, cell.upper)) for cell in cover_cells], dtype=np.float64
    ).reshape((-1, 2, 3))
    cover_lower = np.asarray([cell.value_lower for cell in cover_cells], dtype=np.float64)
    cover_upper = np.asarray([cell.value_upper for cell in cover_cells], dtype=np.float64)
    cover_gradient_lower = np.asarray(
        [cell.gradient_lower for cell in cover_cells], dtype=np.float64
    ).reshape((-1, 3))
    cover_gradient_upper = np.asarray(
        [cell.gradient_upper for cell in cover_cells], dtype=np.float64
    ).reshape((-1, 3))
    cover_complete = bool(source_in_domain)
    cover = None
    if bounds.value_rigorous:
        cover_id = _cover_identity(
            cover_boxes,
            cover_lower,
            cover_upper,
            domain_,
            source_id,
            state_id,
            policy.enclosure,
            cover_complete,
        )
        cover = AdaptiveImplicitCover(
            cover_boxes,
            cover_lower,
            cover_upper,
            cover_gradient_lower,
            cover_gradient_upper,
            domain_.copy(),
            source_id,
            state_id,
            policy.enclosure,
            cover_complete,
            True,
            cover_id,
        )
    accuracy: ImplicitDiscoveryAccuracy = (
        "sampled" if not bounds.value_rigorous else "enclosed"
    )
    if (
        bounds.value_rigorous
        and not unresolved_issues.size
        and cover_complete
        and mesh is not None
    ):
        # Value enclosures and a watertight polygonization establish coverage;
        # the stronger source-isotopy claim remains deliberately withheld.
        accuracy = "enclosed"
    max_surface_diagonal = max(
        (_cell_diagonal(cell.lower, cell.upper) for cell in surface_cells),
        default=0.0,
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "adaptive-implicit-evidence",
            "accuracy": accuracy,
            "enclosure": policy.enclosure,
            "status": status,
            "unresolved_boxes": array_tree_fingerprint(unresolved_boxes),
            "unresolved_issues": array_tree_fingerprint(unresolved_issues),
            "cover": None if cover is None else cover.cover_id,
            "leaf_count": len(cover_cells),
        }
    )
    evidence = AdaptiveImplicitSurfaceEvidence(
        unresolved_boxes,
        unresolved_issues,
        accuracy,
        policy.enclosure,
        status,
        len(cover_cells),
        len(surface_cells),
        deepest,
        max_surface_diagonal,
        evidence_id,
    )
    volume = ImplicitVolumeQuery(
        geometry,
        bounds,
        domain_.copy(),
        source_id,
        bounds.value_rigorous,
        canonical_fingerprint(
            {
                "kind": "implicit-volume-query",
                "state": state_id,
                "domain": array_tree_fingerprint(domain_),
                "enclosure": policy.enclosure,
            }
        ),
    )
    return AdaptiveImplicitSurface(
        mesh,
        evidence,
        cover,
        None,
        volume,
        scaffold,
        np.empty((0, 2), dtype=np.int64),
        witness_boxes,
        witness_values,
        witness_errors,
        geometry,
        policy,
        source_id,
    )


@checked
def discover_adaptive_implicit_surface(
    geometry: CompiledGeometry,
    /,
    *,
    domain: ArrayLike,
    policy: AdaptiveImplicitSurfacePolicy = AdaptiveImplicitSurfacePolicy(),
    source_id: str,
) -> AdaptiveImplicitSurface:
    """Discover every possible zero, refining unresolved cell contours as needed."""
    requested_policy = policy
    effective_policy = policy
    selected_domain = domain
    while True:
        surface = _discover_adaptive_implicit_surface_once(
            geometry,
            domain=selected_domain,
            policy=effective_policy,
            source_id=source_id,
        )
        issues = np.asarray(surface.evidence.unresolved_issues, dtype=np.int64)
        refinable_issues = int(
            AdaptiveImplicitBoxIssue.EDGE | AdaptiveImplicitBoxIssue.MULTIPLE_SHEETS
        )
        if issues.size == 0 or np.any(issues & ~refinable_issues):
            break
        retryable_status = int(
            AdaptiveImplicitSurfaceStatus.UNRESOLVED_BOXES
            | AdaptiveImplicitSurfaceStatus.PHYSICAL_RESOLUTION_REACHED
        )
        if int(surface.evidence.status_flags) & ~retryable_status:
            break
        unresolved_boxes = np.asarray(surface.evidence.unresolved_boxes)
        unresolved_diameters = tuple(
            _cell_diagonal(box[0], box[1]) for box in unresolved_boxes
        )
        if max(unresolved_diameters, default=0.0) <= effective_policy.root_tolerance:
            break
        next_relative = effective_policy.relative_cell_diameter * 0.5
        if (
            not math.isfinite(next_relative)
            or next_relative <= 0.0
            or next_relative >= effective_policy.relative_cell_diameter
        ):
            break
        next_level = _target_level(surface.scaffold.domain, next_relative)
        if next_level <= surface.scaffold.target_level:
            break
        selected_domain = surface.scaffold.domain
        effective_policy = replace(
            effective_policy,
            relative_cell_diameter=next_relative,
        )
    if effective_policy is requested_policy:
        return surface
    return AdaptiveImplicitSurface(
        surface.mesh,
        surface.evidence,
        surface.cover,
        surface.topology,
        surface.volume,
        surface.scaffold,
        surface.feature_edges,
        surface.boundary_witness_boxes,
        surface.boundary_witnesses,
        surface.boundary_errors,
        surface.geometry,
        requested_policy,
        surface.source_id,
    )


__all__ = [
    "AdaptiveImplicitBoundarySource",
    "AdaptiveImplicitCover",
    "AdaptiveImplicitSurface",
    "AdaptiveImplicitSurfaceArchiveRecipe",
    "AdaptiveImplicitScaffold",
    "AdaptiveImplicitSurfaceEvidence",
    "ImplicitVolumeClass",
    "ImplicitVolumeClassification",
    "ImplicitVolumeQuery",
    "discover_adaptive_implicit_surface",
]
