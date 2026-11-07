#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from enum import IntEnum
from fractions import Fraction
from typing import final, Protocol, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

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
    _SOURCE_BOUNDARY_SAMPLES_OWNERSHIP,
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


class _PolygonVertices(Protocol):
    """Appendable indexed vertex storage used by scalar contouring."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int, /) -> np.ndarray: ...
    def append(self, point: np.ndarray, /) -> None: ...


class _PolygonRootEdges(Protocol):
    """Indexed root-edge identities, whether materialized or viewed."""

    def __getitem__(self, index: int, /) -> _GridEdge: ...
    def append(self, edge: _GridEdge, /) -> None: ...


class _PolygonVertexIndex(Protocol):
    """Canonical root-edge to output-vertex assignment."""

    def get(self, edge: _GridEdge, /) -> int | None: ...
    def __setitem__(self, edge: _GridEdge, vertex: int, /) -> None: ...


class _PolygonErrors(Protocol):
    """Append-only error storage for newly assigned vertices."""

    def append(self, error: float, /) -> None: ...


def _evaluate_interval_points(
    program: PreparedIntervalFunction, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lower_parts: list[np.ndarray] = []
    upper_parts: list[np.ndarray] = []
    for start in range(0, points.shape[0], program.batch_capacity):
        stop = min(start + program.batch_capacity, points.shape[0])
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

    def _root_point_enclosures(
        self, batch_capacity: int, /
    ) -> _FieldEnclosures | _RootPointEnclosures:
        """Prepare only the value program at a root-query scheduling capacity."""
        if self.selector != "interval":
            return self
        return _RootPointEnclosures(self, batch_capacity)

    def point_values(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._point_values(points, self._values)

    def _point_values(
        self, points: np.ndarray, program: PreparedIntervalFunction | None, /
    ) -> tuple[np.ndarray, np.ndarray]:
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


class _RootPointEnclosures:
    """A root-only point evaluator bound to the source field and its state."""

    def __init__(self, bounds: _FieldEnclosures, batch_capacity: int, /) -> None:
        self._bounds = bounds
        geometry = bounds.geometry

        def value_function(points: jax.Array) -> jax.Array:
            return geometry.kernel.boundary_field(geometry.state, points)

        self._values = prepare_interval_function(
            value_function, _DIMENSION, batch_capacity=batch_capacity
        )

    def point_values(self, points: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
        return self._bounds._point_values(points, self._values)


_PACKED_OWNERSHIP = object()
_IMMUTABLE_ARCHIVE_OWNERSHIP = object()


def _immutable_archive_array(values: object) -> bool:
    """Accept only arrays ultimately backed by immutable Python bytes."""
    if type(values) is not np.ndarray or not values.flags.c_contiguous:
        return False
    base = values
    while isinstance(base, np.ndarray):
        if type(base) is not np.ndarray or base.flags.writeable:
            return False
        base = base.base
    if isinstance(base, memoryview):
        base = base.obj
    return isinstance(base, bytes)


def _discovery_array(
    values: np.ndarray,
    dtype: DTypeLike,
    shape: tuple[int, ...],
    ownership: object,
) -> np.ndarray:
    """Adopt internal sealed or immutable-backed arrays; copy public inputs."""
    if ownership is _PACKED_OWNERSHIP:
        if (
            not isinstance(values, np.ndarray)
            or values.dtype != np.dtype(dtype)
            or not values.flags.owndata
            or not values.flags.c_contiguous
            or values.flags.writeable
        ):
            raise ValueError("Packed discovery buffers must be sealed owning arrays.")
        array = values
    elif ownership is _IMMUTABLE_ARCHIVE_OWNERSHIP:
        if not _immutable_archive_array(values) or values.dtype != np.dtype(dtype):
            raise ValueError("Archive discovery buffers must have immutable backing.")
        array = values
    else:
        array = np.array(values, dtype=dtype, copy=True)
    return array.reshape(shape)


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
        *,
        _ownership: object = None,
    ) -> None:
        boxes_ = _discovery_array(boxes, np.float64, (-1, 2, 3), _ownership)
        lower = _discovery_array(value_lower, np.float64, (-1,), _ownership)
        upper = _discovery_array(value_upper, np.float64, (-1,), _ownership)
        gradient_lower_ = _discovery_array(
            gradient_lower, np.float64, (-1, 3), _ownership
        )
        gradient_upper_ = _discovery_array(
            gradient_upper, np.float64, (-1, 3), _ownership
        )
        domain_ = np.array(domain, dtype=np.float64, copy=True)
        if (
            lower.shape != (boxes_.shape[0],)
            or upper.shape != lower.shape
            or gradient_lower_.shape != (boxes_.shape[0], 3)
            or gradient_upper_.shape != gradient_lower_.shape
            or domain_.shape != (2, 3)
            or not np.all(np.isfinite(domain_))
            or np.any(domain_[0] >= domain_[1])
        ):
            raise ValueError(
                "Adaptive cover arrays must retain aligned ordered enclosures."
            )
        for start in range(0, len(lower), 16384):
            stop = min(start + 16384, len(lower))
            box_batch = boxes_[start:stop]
            low_batch, high_batch = lower[start:stop], upper[start:stop]
            if (
                not np.all(np.isfinite(box_batch))
                or np.any(np.isnan(low_batch))
                or np.any(np.isnan(high_batch))
                or np.any(np.isnan(gradient_lower_[start:stop]))
                or np.any(np.isnan(gradient_upper_[start:stop]))
                or np.any(box_batch[:, 0] > box_batch[:, 1])
                or np.any(low_batch > high_batch)
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

    def volume_enclosure(self) -> ImplicitCoverVolume:
        """Enclose the region's volume within ``domain`` from the final leaf cover.

        Leaves partition the domain on an exact dyadic lattice; their stored
        corners are outward binary64 roundings, so each leaf is shrunk by one
        ulp per side before its volume is rounded down. Leaves with a negative
        value enclosure lie inside and positive ones outside; every other leaf
        is unresolved volume. The result is independent of any mesh.
        """
        if not self.certified:
            raise ValueError("A volume enclosure needs a value-rigorous discovery cover.")
        inner_lower = np.nextafter(self.boxes[:, 0], np.inf)
        inner_upper = np.nextafter(self.boxes[:, 1], -np.inf)
        widths = np.maximum(np.nextafter(inner_upper - inner_lower, -np.inf), 0.0)
        volumes = np.nextafter(widths[:, 0] * widths[:, 1], -np.inf)
        volumes = np.maximum(np.nextafter(volumes * widths[:, 2], -np.inf), 0.0)

        def lower_sum(mask: np.ndarray) -> Fraction:
            total = math.fsum(volumes[mask].tolist())
            return Fraction(max(float(np.nextafter(total, -np.inf)), 0.0))

        domain_volume = math.prod(
            Fraction(float(self.domain[1, axis])) - Fraction(float(self.domain[0, axis]))
            for axis in range(3)
        )
        inside = lower_sum(self.value_upper < 0.0)
        outside = lower_sum(self.value_lower > 0.0)
        return ImplicitCoverVolume(
            inside_lower=_fraction_down(inside),
            outside_lower=_fraction_down(outside),
            unresolved_upper=_fraction_up(
                max(domain_volume - inside - outside, Fraction(0))
            ),
            domain_volume=float(domain_volume),
            fraction_lower=_fraction_down(inside / domain_volume),
            fraction_upper=_fraction_up(
                min((domain_volume - outside) / domain_volume, Fraction(1))
            ),
            cover_id=self.cover_id,
        )


def _fraction_down(value: Fraction, /) -> float:
    nearest = float(value)
    return float(np.nextafter(nearest, -np.inf)) if Fraction(nearest) > value else nearest


def _fraction_up(value: Fraction, /) -> float:
    nearest = float(value)
    return float(np.nextafter(nearest, np.inf)) if Fraction(nearest) < value else nearest


class ImplicitCoverVolume(StrictModule):
    """Certified, mesh-independent volume of a source region within a cover domain.

    ``inside_lower`` and ``outside_lower`` bound the region's and its
    complement's measure from below, ``unresolved_upper`` bounds the volume of
    leaves the cover could not classify, and ``[fraction_lower,
    fraction_upper]`` encloses the region's volume fraction of the domain
    (relative density; porosity is its complement).
    """

    inside_lower: float = eqx.field(static=True)
    outside_lower: float = eqx.field(static=True)
    unresolved_upper: float = eqx.field(static=True)
    domain_volume: float = eqx.field(static=True)
    fraction_lower: float = eqx.field(static=True)
    fraction_upper: float = eqx.field(static=True)
    cover_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        inside_lower: float,
        outside_lower: float,
        unresolved_upper: float,
        domain_volume: float,
        fraction_lower: float,
        fraction_upper: float,
        cover_id: str,
    ) -> None:
        if not 0.0 <= fraction_lower <= fraction_upper <= 1.0:
            raise ValueError(
                "A cover volume fraction enclosure must be ordered in [0, 1]."
            )
        self.inside_lower = inside_lower
        self.outside_lower = outside_lower
        self.unresolved_upper = unresolved_upper
        self.domain_volume = domain_volume
        self.fraction_lower = fraction_lower
        self.fraction_upper = fraction_upper
        self.cover_id = cover_id


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
    """Adaptive-discovery issues and numerical termination evidence.

    ``lattice_anchor`` names the discovery lattice: ``0`` subdivides the declared
    domain, ``k > 0`` the ``k``-th deterministic re-anchored lattice, tried because
    the previous lattice placed a vertex on a sign crossing of the zero set.
    ``declared_domain`` is the caller's box.
    """

    unresolved_boxes: np.ndarray
    unresolved_issues: np.ndarray
    declared_domain: np.ndarray
    accuracy: ImplicitDiscoveryAccuracy = eqx.field(static=True)
    enclosure: ImplicitDiscoveryEnclosure = eqx.field(static=True)
    status: int = eqx.field(static=True)
    leaf_count: int = eqx.field(static=True)
    surface_leaf_count: int = eqx.field(static=True)
    maximum_level_reached: int = eqx.field(static=True)
    maximum_surface_box_diagonal: float = eqx.field(static=True)
    lattice_anchor: int = eqx.field(static=True)
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
        *,
        declared_domain: np.ndarray,
        lattice_anchor: int,
        _ownership: object = None,
    ) -> None:
        boxes = _discovery_array(unresolved_boxes, np.float64, (-1, 2, 3), _ownership)
        issues = _discovery_array(unresolved_issues, np.int64, (-1,), _ownership)
        declared = np.array(declared_domain, dtype=np.float64, copy=True)
        if (
            boxes.shape[0] != issues.shape[0]
            or not np.all(np.isfinite(boxes))
            or np.any(boxes[:, 0] > boxes[:, 1])
            or np.any(issues == 0)
        ):
            raise ValueError(
                "Adaptive evidence requires aligned unresolved boxes and nonzero issues."
            )
        if (
            declared.shape != (2, 3)
            or not np.all(np.isfinite(declared))
            or np.any(declared[0] >= declared[1])
            or not 0 <= lattice_anchor <= len(_LATTICE_ANCHORS)
        ):
            raise ValueError(
                "Adaptive evidence requires a finite declared domain and a known lattice anchor."
            )
        boxes.setflags(write=False)
        issues.setflags(write=False)
        declared.setflags(write=False)
        self.unresolved_boxes = boxes
        self.unresolved_issues = issues
        self.declared_domain = declared
        self.accuracy = accuracy
        self.enclosure = enclosure
        self.status = int(status)
        self.leaf_count = int(leaf_count)
        self.surface_leaf_count = int(surface_leaf_count)
        self.maximum_level_reached = int(maximum_level_reached)
        self.maximum_surface_box_diagonal = float(maximum_surface_box_diagonal)
        self.lattice_anchor = int(lattice_anchor)
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
        _ownership: object = None,
    ) -> None:
        arrays = {
            "domain": np.array(domain, dtype=np.float64, copy=True).reshape((2, 3)),
            "cell_indices": _discovery_array(cell_indices, np.int64, (-1, 4), _ownership),
            "cell_boxes": _discovery_array(
                cell_boxes, np.float64, (-1, 2, 3), _ownership
            ),
            "cell_issues": _discovery_array(cell_issues, np.int64, (-1,), _ownership),
            "cell_value_lower": _discovery_array(
                cell_value_lower, np.float64, (-1,), _ownership
            ),
            "cell_value_upper": _discovery_array(
                cell_value_upper, np.float64, (-1,), _ownership
            ),
            "cell_gradient_lower": _discovery_array(
                cell_gradient_lower, np.float64, (-1, 3), _ownership
            ),
            "cell_gradient_upper": _discovery_array(
                cell_gradient_upper, np.float64, (-1, 3), _ownership
            ),
            "cell_corner_ids": _discovery_array(
                cell_corner_ids, np.int64, (-1, 8, 3), _ownership
            ),
            "corner_ids": _discovery_array(corner_ids, np.int64, (-1, 3), _ownership),
            "corner_coordinates": _discovery_array(
                corner_coordinates, np.float64, (-1, 3), _ownership
            ),
            "corner_value_lower": _discovery_array(
                corner_value_lower, np.float64, (-1,), _ownership
            ),
            "corner_value_upper": _discovery_array(
                corner_value_upper, np.float64, (-1,), _ownership
            ),
            "corner_sampled_values": _discovery_array(
                corner_sampled_values, np.float64, (-1,), _ownership
            ),
            "corner_signs": _discovery_array(corner_signs, np.int8, (-1,), _ownership),
            "tetrahedra": _discovery_array(tetrahedra, np.int64, (-1, 4), _ownership),
            "tetrahedron_cell_rows": _discovery_array(
                tetrahedron_cell_rows, np.int64, (-1,), _ownership
            ),
            "root_edge_ids": _discovery_array(
                root_edge_ids, np.int64, (-1, 2, 3), _ownership
            ),
            "root_brackets": _discovery_array(
                root_brackets, np.float64, (-1, 2, 3), _ownership
            ),
            "root_points": _discovery_array(root_points, np.float64, (-1, 3), _ownership),
            "root_errors": _discovery_array(root_errors, np.float64, (-1,), _ownership),
            "root_issues": _discovery_array(root_issues, np.int64, (-1,), _ownership),
            "root_endpoint_value_lower": _discovery_array(
                root_endpoint_value_lower, np.float64, (-1, 2), _ownership
            ),
            "root_endpoint_value_upper": _discovery_array(
                root_endpoint_value_upper, np.float64, (-1, 2), _ownership
            ),
            "triangle_cell_rows": _discovery_array(
                triangle_cell_rows, np.int64, (-1,), _ownership
            ),
            "triangle_root_edge_rows": _discovery_array(
                triangle_root_edge_rows, np.int64, (-1, 3), _ownership
            ),
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
    """Actual accepted discovery state and its original source inputs."""

    geometry: CompiledGeometry
    domain: np.ndarray
    policy: AdaptiveImplicitSurfacePolicy
    source_id: str
    mesh_arrays: tuple[Array | np.ndarray, Array | np.ndarray]
    mesh_source_id: str
    cover_arrays: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]
    cover_metadata: tuple[str, str, ImplicitDiscoveryEnclosure, bool, bool, str]
    evidence_arrays: tuple[np.ndarray, np.ndarray, np.ndarray]
    evidence_metadata: tuple[
        ImplicitDiscoveryAccuracy,
        ImplicitDiscoveryEnclosure,
        int,
        int,
        int,
        int,
        float,
        int,
        str,
    ]
    scaffold_arrays: tuple[np.ndarray, ...]
    scaffold_target_level: int
    boundary_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

    def __post_init__(self) -> None:
        domain = np.array(self.domain, dtype=np.float64, copy=True)
        if (
            domain.shape != (2, 3)
            or not np.all(np.isfinite(domain))
            or np.any(domain[0] >= domain[1])
        ):
            raise ValueError("Adaptive archive domain must be finite and ordered.")
        if not np.all(np.isfinite(domain[1] - domain[0])):
            raise ValueError("Adaptive archive domain extents must be representable.")
        domain.setflags(write=False)
        object.__setattr__(self, "domain", domain)


# Accepted numerical fields remain archive/PyTree leaves.
jax.tree_util.register_dataclass(
    AdaptiveImplicitSurfaceArchiveRecipe,
    data_fields=(
        "geometry",
        "domain",
        "mesh_arrays",
        "cover_arrays",
        "evidence_arrays",
        "scaffold_arrays",
        "boundary_arrays",
    ),
    meta_fields=(
        "policy",
        "source_id",
        "mesh_source_id",
        "cover_metadata",
        "evidence_metadata",
        "scaffold_target_level",
    ),
)


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
        *,
        _ownership: object = None,
    ) -> None:
        feature_edges_ = _discovery_array(feature_edges, np.int64, (-1, 2), _ownership)
        witness_boxes = _discovery_array(
            boundary_witness_boxes, np.float64, (-1, 2, 3), _ownership
        )
        witnesses = _discovery_array(boundary_witnesses, np.float64, (-1, 3), _ownership)
        errors = _discovery_array(boundary_errors, np.float64, (-1,), _ownership)
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
        """Retain actual discovery arrays rather than only a request to rediscover."""
        if self.mesh is None or self.cover is None or self.topology is not None:
            raise ValueError("Only a complete adaptive discovery can be archived.")
        cover = self.cover
        evidence = self.evidence
        scaffold = self.scaffold
        return AdaptiveImplicitSurfaceArchiveRecipe(
            self.geometry,
            evidence.declared_domain,
            self.policy,
            self.source_id,
            (self.mesh.vertices, self.mesh.faces),
            self.mesh.source_id,
            (
                cover.boxes,
                cover.value_lower,
                cover.value_upper,
                cover.gradient_lower,
                cover.gradient_upper,
                cover.domain,
            ),
            (
                cover.source_id,
                cover.state_id,
                cover.enclosure,
                cover.complete,
                cover.certified,
                cover.cover_id,
            ),
            (
                evidence.unresolved_boxes,
                evidence.unresolved_issues,
                evidence.declared_domain,
            ),
            (
                evidence.accuracy,
                evidence.enclosure,
                evidence.status,
                evidence.leaf_count,
                evidence.surface_leaf_count,
                evidence.maximum_level_reached,
                evidence.maximum_surface_box_diagonal,
                evidence.lattice_anchor,
                evidence.evidence_id,
            ),
            tuple(
                getattr(scaffold, name)
                for name in AdaptiveImplicitScaffold.__annotations__
                if name != "target_level"
            ),
            scaffold.target_level,
            (
                self.feature_edges,
                self.boundary_witness_boxes,
                self.boundary_witnesses,
                self.boundary_errors,
            ),
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
        domain = np.asarray(self.surface.evidence.declared_domain, dtype=np.float64)
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
                self.surface.geometry,
                domain=domain,
                policy=candidate_policy,
                source_id=self.surface.source_id,
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

        # Exact box keys replace a Python tuple/dictionary per witness. The
        # stable permutation is numeric and preserves float equality (including
        # signed zero); all comparisons and query scratch are chunked.
        batch_size = 16384
        key_dtype = np.dtype([(f"coordinate_{axis}", np.float64) for axis in range(6)])

        def box_keys(values: np.ndarray) -> np.ndarray:
            rows = np.ascontiguousarray(values, dtype=np.float64).reshape((-1, 6))
            return rows.view(key_dtype).reshape((-1,))

        root_keys = box_keys(root_boxes)
        order = np.argsort(root_keys, kind="stable")
        for start in range(0, root_count, batch_size):
            stop = min(start + batch_size, root_count)
            sorted_keys = root_keys[order[start:stop]]
            if (start and root_keys[order[start - 1]] == sorted_keys[0]) or np.any(
                sorted_keys[1:] == sorted_keys[:-1]
            ):
                raise ValueError(
                    "Adaptive source witness boxes must uniquely identify their roots."
                )
        root_radii = np.empty((root_count,), dtype=np.float64)
        for start in range(0, root_count, batch_size):
            stop = min(start + batch_size, root_count)
            extents = np.nextafter(
                root_boxes[start:stop, 1] - root_boxes[start:stop, 0], np.inf
            )
            squared = np.nextafter(extents * extents, np.inf)
            root_radii[start:stop] = np.nextafter(
                np.sqrt(np.nextafter(np.sum(squared, axis=1), np.inf)), np.inf
            )

        possible_count = 0
        if cover is not None:
            for start in range(0, len(cover.value_lower), batch_size):
                stop = min(start + batch_size, len(cover.value_lower))
                possible_count += int(
                    np.count_nonzero(
                        (cover.value_lower[start:stop] <= 0.0)
                        & (cover.value_upper[start:stop] >= 0.0)
                    )
                )
        assigned_roots = np.full((possible_count,), -1, dtype=np.int64)
        covering_radius = np.full((possible_count,), np.inf, dtype=np.float64)
        point_order = None
        ordered_x = None
        offset = 0
        if cover is not None and root_count:
            for start in range(0, len(cover.value_lower), batch_size):
                stop = min(start + batch_size, len(cover.value_lower))
                possible = (cover.value_lower[start:stop] <= 0.0) & (
                    cover.value_upper[start:stop] >= 0.0
                )
                boxes = np.asarray(cover.boxes[start:stop][possible], dtype=np.float64)
                if not boxes.size:
                    continue
                keys = box_keys(boxes)
                positions = np.searchsorted(root_keys, keys, side="left", sorter=order)
                candidates = np.flatnonzero(positions < root_count)
                matches = np.zeros((len(boxes),), dtype=np.bool_)
                if candidates.size:
                    root_rows = order[positions[candidates]]
                    matched = root_keys[root_rows] == keys[candidates]
                    matched_positions = candidates[matched]
                    matched_rows = root_rows[matched]
                    matches[matched_positions] = True
                    assigned_roots[offset + matched_positions] = matched_rows
                    covering_radius[offset + matched_positions] = root_radii[matched_rows]
                if not np.all(matches):
                    if point_order is None:
                        point_order = np.argsort(root_points[:, 0], kind="stable")
                        ordered_x = root_points[point_order, 0]
                    if ordered_x is None:
                        raise RuntimeError(
                            "Implicit root ordering has no ordered coordinates."
                        )
                    for row in np.flatnonzero(~matches):
                        box = boxes[row]
                        first = float(np.nextafter(box[0, 0] - requested, -np.inf))
                        second = float(np.nextafter(box[1, 0] + requested, np.inf))
                        begin = int(np.searchsorted(ordered_x, first, side="left"))
                        end = int(np.searchsorted(ordered_x, second, side="right"))
                        candidate_rows = point_order[begin:end]
                        if not candidate_rows.size:
                            continue
                        candidate_points = root_points[candidate_rows]
                        first_delta = np.nextafter(
                            np.abs(candidate_points - box[0]), np.inf
                        )
                        second_delta = np.nextafter(
                            np.abs(candidate_points - box[1]), np.inf
                        )
                        delta = np.maximum(first_delta, second_delta)
                        squared_upper = np.nextafter(delta * delta, np.inf)
                        distance_squared = np.zeros(
                            (candidate_rows.size,), dtype=np.float64
                        )
                        for axis in range(3):
                            distance_squared = np.nextafter(
                                distance_squared + squared_upper[:, axis], np.inf
                            )
                        distance_upper = np.nextafter(
                            np.sqrt(np.nextafter(distance_squared, np.inf)), np.inf
                        )
                        best = int(np.argmin(distance_upper))
                        if distance_upper[best] <= requested:
                            assigned_roots[offset + row] = int(candidate_rows[best])
                            covering_radius[offset + row] = float(distance_upper[best])
                offset += len(boxes)

        covered = assigned_roots >= 0
        all_covered = bool(np.all(covered))
        complete = bool(
            cover is not None
            and cover.certified
            and cover.complete
            and all_covered
            and np.all(covering_radius <= requested)
        )
        semantics = "certified" if complete else "sampled"
        del root_radii, root_keys, order, point_order, ordered_x
        selected_roots = assigned_roots if all_covered else assigned_roots[covered]
        radii = covering_radius if all_covered else covering_radius[covered]
        del assigned_roots, covering_radius, covered
        points = root_points[selected_roots]
        errors = root_errors[selected_roots]
        del selected_roots
        for values in (points, radii, errors):
            values.setflags(write=False)
        return SourceBoundarySamples(
            points,
            radii,
            errors,
            semantics,
            complete=complete,
            _ownership=_SOURCE_BOUNDARY_SAMPLES_OWNERSHIP,
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


def _norm_upper_rows(vectors: np.ndarray, /) -> np.ndarray:
    """`_norm_upper` of every row: the same widening, ``math.hypot`` and rounding."""
    raw = np.abs(np.asarray(vectors, dtype=np.float64))
    # Grid-edge brackets are axis aligned. For a normal single nonzero
    # component, widening the zero components to the least subnormal changes
    # its hypot by far less than half an ulp. Preserve the scalar route for
    # subnormal components and general vectors.
    nonzero = np.count_nonzero(raw, axis=1)
    magnitude = np.max(raw, axis=1, initial=0.0)
    if np.all(
        (nonzero <= 1) & ((magnitude == 0.0) | (magnitude >= np.finfo(np.float64).tiny))
    ):
        widened = np.nextafter(magnitude, np.inf)
        return np.where(magnitude == 0.0, 0.0, np.nextafter(widened, np.inf))
    widened = np.nextafter(raw, np.inf).tolist()
    norms = np.nextafter(
        np.asarray([math.hypot(*row) for row in widened], dtype=np.float64), np.inf
    )
    return np.where(np.any(raw, axis=1), norms, 0.0)


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
    try:
        face_bounds = bounds.boxes(
            np.stack([lower for lower, _ in faces]),
            np.stack([upper for _, upper in faces]),
        )
    except IntervalExtensionError:
        # No interval extension over a face is no proof: the crossing stays flagged.
        return False
    lower = np.asarray(face_bounds.value_lower, dtype=np.float64)
    upper = np.asarray(face_bounds.value_upper, dtype=np.float64)
    return bool(np.all((lower > 0.0) | (upper < 0.0)))


def _prove_constant_signs(
    bounds: _FieldEnclosures,
    lower: np.ndarray,
    upper: np.ndarray,
    signs: np.ndarray,
    gradient_lower: np.ndarray,
    gradient_upper: np.ndarray,
    /,
) -> np.ndarray:
    """Prove that the field keeps the corners' strict sign on each closed box.

    A band box whose eight corners share a proved sign ``s`` misses the zero
    set exactly when ``s F`` stays positive on it. Along an axis where the
    gradient enclosure proves monotonicity, the minimum of ``s F`` lies on one
    face, so the box restricts to that face. Remaining pieces are bisected
    with interval value enclosures until each is proved, a point enclosure of
    the opposite sign proves the zero set meets the box, or a piece cannot be
    bisected in binary64. A box is proved only when all its pieces are.
    """
    count = int(lower.shape[0])
    if not count or not bounds.value_rigorous:
        return np.zeros(count, dtype=np.bool_)
    failed = np.zeros(count, dtype=np.bool_)
    owner = np.arange(count, dtype=np.int64)
    low = np.array(lower, dtype=np.float64, copy=True)
    high = np.array(upper, dtype=np.float64, copy=True)
    sign = np.asarray(signs, dtype=np.int64)
    slope_lower = np.asarray(gradient_lower, dtype=np.float64)
    slope_upper = np.asarray(gradient_upper, dtype=np.float64)
    while owner.size:
        positive = (sign[owner] > 0)[:, None]
        increasing = slope_lower > 0.0
        decreasing = slope_upper < 0.0
        at_low = (positive & increasing) | (~positive & decreasing)
        at_high = (positive & decreasing) | (~positive & increasing)
        high = np.where(at_low, low, high)
        low = np.where(at_high, high, low)
        try:
            evidence = bounds.boxes(low, high)
        except IntervalExtensionError:
            # No interval extension over a piece is no proof.
            return np.zeros(count, dtype=np.bool_)
        value_lower = np.asarray(evidence.value_lower, dtype=np.float64)
        value_upper = np.asarray(evidence.value_upper, dtype=np.float64)
        positive = positive[:, 0]
        settled = np.where(positive, value_lower > 0.0, value_upper < 0.0)
        middle = 0.5 * low + 0.5 * high
        open_rows = np.flatnonzero(~settled)
        point_lower, point_upper = bounds.point_values(middle[open_rows])
        opposite = np.where(positive[open_rows], point_upper < 0.0, point_lower > 0.0)
        failed[owner[open_rows[opposite]]] = True
        splittable = (middle > low) & (middle < high)
        divisible = np.any(splittable, axis=1)
        failed[owner[open_rows[~divisible[open_rows]]]] = True
        keep = ~settled & divisible & ~failed[owner]
        owner, low, high, middle = owner[keep], low[keep], high[keep], middle[keep]
        splittable = splittable[keep]
        slope_lower = np.asarray(evidence.gradient_lower, dtype=np.float64)[keep]
        slope_upper = np.asarray(evidence.gradient_upper, dtype=np.float64)[keep]
        axis = np.argmax(np.where(splittable, high - low, -1.0), axis=1)
        rows = np.arange(owner.size)
        left_high = high.copy()
        left_high[rows, axis] = middle[rows, axis]
        right_low = low.copy()
        right_low[rows, axis] = middle[rows, axis]
        owner = np.concatenate((owner, owner))
        low = np.concatenate((low, right_low))
        high = np.concatenate((left_high, high))
        slope_lower = np.concatenate((slope_lower, slope_lower))
        slope_upper = np.concatenate((slope_upper, slope_upper))
    return ~failed


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


# Per-axis lattice shifts, in target-level cells, of the re-anchored lattices
# tried when a lattice vertex lies on a regular crossing of the zero set. The
# shifts are the fractional parts of multiples of the golden ratio, sqrt(2) and
# sqrt(3): no low-denominator dyadic coincidence with authored coordinates, and
# the zero set meets any one shifted vertex family only on a measure-zero set.
_LATTICE_ANCHORS: tuple[tuple[float, float, float], ...] = (
    (0.6180339887498949, 0.41421356237309515, 0.7320508075688772),
    (0.2360679774997898, 0.8284271247461903, 0.4641016151377544),
    (0.8541019662496847, 0.2426406871192854, 0.1961524227066316),
)

# Neighbour stencil of the lattice-independent vertex-root probe, in units of
# ``_ROOT_PROBE_SCALE`` target-level cells.
_ROOT_PROBE_OFFSETS = np.asarray(
    [offset for offset in itertools.product((-1.0, 0.0, 1.0), repeat=3) if any(offset)],
    dtype=np.float64,
)
_ROOT_PROBE_SCALE = 2.0**-8


def _anchored_domain(
    declared: np.ndarray, relative_diameter: float, anchor: int, /
) -> np.ndarray:
    """Discovery box of lattice ``anchor``; anchor ``0`` is the declared box.

    A re-anchored box extends the declared box by one target-level cell per axis,
    split by the anchor's shift, so it still covers the declared box.
    """
    if anchor == 0:
        return declared
    cell = (declared[1] - declared[0]) / 2.0 ** _target_level(declared, relative_diameter)
    shift = np.asarray(_LATTICE_ANCHORS[anchor - 1], dtype=np.float64)
    anchored = np.stack((declared[0] - shift * cell, declared[1] + (1.0 - shift) * cell))
    if (
        not np.all(np.isfinite(anchored))
        or np.any(anchored[0] >= declared[0])
        or np.any(anchored[1] <= declared[1])
    ):
        raise ValueError(
            "A re-anchored discovery lattice must cover the declared domain."
        )
    return anchored


def _anchored_relative_diameter(
    declared: np.ndarray, anchored: np.ndarray, relative_diameter: float, /
) -> float:
    """Relative cell diameter keeping the declared physical resolution on ``anchored``.

    Every anchored axis is at most ``1 / ratio`` times its declared extent, so the
    anchored diagonal times the returned value never exceeds the declared target.
    """
    ratio = float(np.min((declared[1] - declared[0]) / (anchored[1] - anchored[0])))
    return float(np.nextafter(np.nextafter(relative_diameter * ratio, 0.0), 0.0))


def _vertex_roots_are_lattice_coincidences(
    geometry: CompiledGeometry,
    surface: AdaptiveImplicitSurface,
    enclosure: ImplicitDiscoveryEnclosure,
    /,
) -> bool:
    """Whether every undecided lattice vertex lies on a sign crossing of the field.

    A vertex whose point enclosure contains zero is a lattice coincidence when the
    field takes both certified signs on its neighbour stencil a small fraction of
    a cell away: the zero set separates the stencil there, so a shifted lattice
    brackets it by corner signs. A vertex without such a crossing (a touching or
    degenerate zero, or a field too uncertain to sign its neighbourhood) is
    singular for the field independently of the lattice; re-anchoring cannot
    resolve it, so the current lattice's refusal stands.
    """
    scaffold = surface.scaffold
    undecided = np.asarray(scaffold.corner_signs) == 0
    if not np.any(undecided):
        return False
    vertices = np.asarray(scaffold.corner_coordinates, dtype=np.float64)[undecided]
    domain = np.asarray(scaffold.domain, dtype=np.float64)
    step = _ROOT_PROBE_SCALE * (domain[1] - domain[0]) / 2.0**scaffold.target_level
    probes = vertices[:, None, :] + _ROOT_PROBE_OFFSETS[None, :, :] * step
    lower, upper = _FieldEnclosures(geometry, enclosure).point_values(
        probes.reshape((-1, _DIMENSION))
    )
    lower = lower.reshape(probes.shape[:2])
    upper = upper.reshape(probes.shape[:2])
    crossing = np.any(upper < 0.0, axis=1) & np.any(lower > 0.0, axis=1)
    return bool(np.all(crossing))


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
    bounds: _FieldEnclosures | _RootPointEnclosures,
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
        half_lengths = _norm_upper_rows(0.5 * (high[rows] - low[rows]))
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
    vertices: _PolygonVertices, triangle: tuple[int, int, int], /
) -> float:
    first, second, third = (vertices[index] for index in triangle)
    return 0.5 * float(np.linalg.norm(np.cross(second - first, third - first)))


def _certified_triangle_orientations(
    points: np.ndarray, errors: np.ndarray, gradients: np.ndarray, /
) -> np.ndarray:
    """Whether each contour triangle's orientation is certified by its root bounds.

    ``points`` are the computed roots ``(n, 3, 3)``, ``errors`` bound the
    Euclidean distance of each computed root from its exact root ``(n, 3)``, and
    ``gradients`` are the orientation references ``(n, 3)``. For exact roots
    ``p_i + d_i`` with ``|d_i| <= e_i`` the exact normal differs from the computed
    ``a x b`` by at most ``|a|(e0+e2) + |b|(e0+e1) + (e0+e1)(e0+e2)``. The errors
    are widened by the rounding of the coordinate differences and the bound by
    the rounding of the cross and dot products. When ``|n . g|`` exceeds that
    bound times ``|g|``, every admissible exact triangle has a nonzero normal on
    the computed side of ``g``: it is nondegenerate with the computed
    orientation. No absolute area enters, so the decision does not depend on the
    length unit or the lattice resolution.
    """
    eps = float(np.finfo(np.float64).eps)
    corners = np.asarray(points, dtype=np.float64)
    bounds = np.asarray(errors, dtype=np.float64)
    references = np.asarray(gradients, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        widened = bounds + 4.0 * eps * np.max(np.abs(corners), axis=2)
        first = corners[:, 1] - corners[:, 0]
        second = corners[:, 2] - corners[:, 0]
        normals = np.cross(first, second)
        first_length = np.linalg.norm(first, axis=1)
        second_length = np.linalg.norm(second, axis=1)
        first_error = widened[:, 0] + widened[:, 1]
        second_error = widened[:, 0] + widened[:, 2]
        perturbation = (
            first_length * second_error
            + second_length * first_error
            + first_error * second_error
        )
        bound = (perturbation + 64.0 * eps * first_length * second_length) * (
            1.0 + 64.0 * eps
        )
        reference_length = np.linalg.norm(references, axis=1)
        orientation = np.abs(np.sum(normals * references, axis=1))
        return (
            np.isfinite(orientation)
            & np.isfinite(bound)
            & np.isfinite(reference_length)
            & (reference_length > 0.0)
            & (orientation > bound * reference_length)
        )


def _triangulate_polygon(
    vertices: _PolygonVertices, cycle: list[int], /
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
    # Python binary64 rows: the same IEEE operations as NumPy scalars, without
    # a fancy-indexed array per predicate.
    rows = planar.tolist()

    def orient(first: int, second: int, third: int, /) -> float:
        a, b, c = rows[first], rows[second], rows[third]
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

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
    vertices: _PolygonVertices,
    vertex_errors: _PolygonErrors,
    vertex_keys: _PolygonVertexIndex,
    faces: list[tuple[int, int, int]],
    edge_keys_by_vertex: _PolygonRootEdges,
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
        corners = np.stack([vertices[vertex] for vertex in triangle])
        bounds = np.asarray(
            [[root_cache[edge_keys_by_vertex[vertex]][1] for vertex in triangle]],
            dtype=np.float64,
        )
        if not bool(
            _certified_triangle_orientations(corners[None], bounds, gradient[None])[0]
        ):
            issue |= int(AdaptiveImplicitBoxIssue.FACE)
            continue
        first_point, second_point, third_point = corners
        normal = np.cross(second_point - first_point, third_point - first_point)
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


def _validate_discovery_inputs(
    geometry: CompiledGeometry, policy: AdaptiveImplicitSurfacePolicy, source_id: str, /
) -> None:
    """Recheck the same field and policy premises for authoring and restoration."""
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


@checked
def _discover_adaptive_implicit_surface_once(
    geometry: CompiledGeometry,
    /,
    *,
    domain: ArrayLike,
    policy: AdaptiveImplicitSurfacePolicy = AdaptiveImplicitSurfacePolicy(),
    source_id: str,
    declared_domain: ArrayLike,
    lattice_anchor: int,
) -> AdaptiveImplicitSurface:
    """Exhaustively enclose and polygonize a region's implicit zero set.

    Possible-zero leaves are resolved to the requested physical cell diameter.
    Pure edge-ambiguous leaves trigger finer dyadic discovery until their roots
    are witnessed or their diameter reaches the requested root tolerance.
    Singular and otherwise unresolved leaves remain explicit.
    """
    _validate_discovery_inputs(geometry, policy, source_id)
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
    from ._adaptive_packed_frontier import discover_leaves, PackedPolygonLeaves
    from ._adaptive_packed_polygon import polygonize

    leaves = discover_leaves(bounds, domain_, target_level)
    leaf_count = len(leaves.level)
    surface_count = len(leaves.surface_rows)
    deepest_level = leaves.deepest_level
    status = leaves.status
    cover_arrays = (
        leaves.boxes,
        leaves.value_lower,
        leaves.value_upper,
        leaves.gradient_lower,
        leaves.gradient_upper,
    )
    polygon_leaves = PackedPolygonLeaves.from_leaves(leaves)
    del leaves
    polygon = polygonize(
        bounds, domain_, target_level, policy, polygon_leaves, source_id, source_in_domain
    )
    del polygon_leaves
    unresolved_boxes = polygon.unresolved_boxes
    unresolved_issues = polygon.unresolved_issues
    status |= polygon.status
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
    if not surface_count and source_in_domain:
        status |= int(AdaptiveImplicitSurfaceStatus.NO_SURFACE)
    if not source_in_domain:
        status |= int(AdaptiveImplicitSurfaceStatus.DOMAIN_BOUNDARY_CROSSING)
    feature_edges = np.empty((0, 2), dtype=np.int64)
    for values in (
        unresolved_boxes,
        unresolved_issues,
        polygon.witness_boxes,
        polygon.witness_values,
        polygon.witness_errors,
        feature_edges,
    ):
        if not values.flags.owndata or not values.flags.c_contiguous:
            raise ValueError("Packed discovery evidence must own its contiguous buffers.")
        values.setflags(write=False)

    cover_complete = bool(source_in_domain)
    cover = None
    if bounds.value_rigorous:
        cover_id = _cover_identity(
            cover_arrays[0],
            cover_arrays[1],
            cover_arrays[2],
            domain_,
            source_id,
            state_id,
            policy.enclosure,
            cover_complete,
        )
        cover = AdaptiveImplicitCover(
            *cover_arrays,
            domain_,
            source_id,
            state_id,
            policy.enclosure,
            cover_complete,
            True,
            cover_id,
            _ownership=_PACKED_OWNERSHIP,
        )
    del cover_arrays
    # These buffers are freshly produced internally and have not escaped to a
    # caller. Seal their owning allocations before transferring them to the
    # validating immutable scaffold constructor.
    for values in polygon.scaffold_arrays.values():
        if not values.flags.owndata or not values.flags.c_contiguous:
            raise ValueError("Packed scaffold must own its contiguous buffers.")
        values.setflags(write=False)
    scaffold = AdaptiveImplicitScaffold(
        domain=domain_,
        target_level=target_level,
        _ownership=_PACKED_OWNERSHIP,
        **polygon.scaffold_arrays,
    )
    accuracy: ImplicitDiscoveryAccuracy = (
        "sampled" if not bounds.value_rigorous else "enclosed"
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
            "leaf_count": leaf_count,
            **(
                {}
                if lattice_anchor == 0
                else {
                    "lattice_anchor": lattice_anchor,
                    "declared_domain": array_tree_fingerprint(
                        np.asarray(declared_domain, dtype=np.float64)
                    ),
                }
            ),
        }
    )
    evidence = AdaptiveImplicitSurfaceEvidence(
        unresolved_boxes,
        unresolved_issues,
        accuracy,
        policy.enclosure,
        status,
        leaf_count,
        surface_count,
        deepest_level,
        polygon.max_surface_diagonal,
        evidence_id,
        declared_domain=np.asarray(declared_domain, dtype=np.float64),
        lattice_anchor=lattice_anchor,
        _ownership=_PACKED_OWNERSHIP,
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
        polygon.mesh,
        evidence,
        cover,
        None,
        volume,
        scaffold,
        feature_edges,
        polygon.witness_boxes,
        polygon.witness_values,
        polygon.witness_errors,
        geometry,
        policy,
        source_id,
        _ownership=_PACKED_OWNERSHIP,
    )


def _next_discovery_resolution(
    surface: AdaptiveImplicitSurface, policy: AdaptiveImplicitSurfacePolicy, /
) -> float | None:
    """Return the next dyadic resolution only for a retryable discovery."""
    issues = np.asarray(surface.evidence.unresolved_issues, dtype=np.int64)
    refinable = int(
        AdaptiveImplicitBoxIssue.EDGE | AdaptiveImplicitBoxIssue.MULTIPLE_SHEETS
    )
    if not issues.size or np.any(issues & ~refinable):
        return None
    retryable = int(
        AdaptiveImplicitSurfaceStatus.UNRESOLVED_BOXES
        | AdaptiveImplicitSurfaceStatus.PHYSICAL_RESOLUTION_REACHED
    )
    if int(surface.evidence.status_flags) & ~retryable:
        return None
    boxes = np.asarray(surface.evidence.unresolved_boxes)
    if (
        max((_cell_diagonal(box[0], box[1]) for box in boxes), default=0.0)
        <= policy.root_tolerance
    ):
        return None
    next_relative = policy.relative_cell_diameter * 0.5
    if (
        not math.isfinite(next_relative)
        or next_relative <= 0.0
        or next_relative >= policy.relative_cell_diameter
        or _target_level(surface.scaffold.domain, next_relative)
        <= surface.scaffold.target_level
    ):
        return None
    return next_relative


def _discover_on_lattice(
    geometry: CompiledGeometry,
    declared_domain: np.ndarray,
    lattice_anchor: int,
    policy: AdaptiveImplicitSurfacePolicy,
    source_id: str,
    /,
) -> AdaptiveImplicitSurface:
    """Discover on one lattice, refining edge-ambiguous cell contours as needed."""
    selected_domain = _anchored_domain(
        declared_domain, policy.relative_cell_diameter, lattice_anchor
    )
    effective_policy = (
        policy
        if lattice_anchor == 0
        else replace(
            policy,
            relative_cell_diameter=_anchored_relative_diameter(
                declared_domain, selected_domain, policy.relative_cell_diameter
            ),
        )
    )
    while True:
        surface = _discover_adaptive_implicit_surface_once(
            geometry,
            domain=selected_domain,
            policy=effective_policy,
            source_id=source_id,
            declared_domain=declared_domain,
            lattice_anchor=lattice_anchor,
        )
        next_relative = _next_discovery_resolution(surface, effective_policy)
        if next_relative is None:
            break
        selected_domain = surface.scaffold.domain
        effective_policy = replace(
            effective_policy,
            relative_cell_diameter=next_relative,
        )
        # The prior, rejected carrier is not an input to the next discovery.
        del surface
    if effective_policy is policy:
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
        policy,
        surface.source_id,
    )


def verify_discovery_history(surface: AdaptiveImplicitSurface, /) -> bool:
    """Cold-replay prior retries and bind a validated result to its policy.

    The caller independently validates retained leaf and polygon evidence.
    Rejected attempts are replayed in bounded chunks without constructing
    another discovery, mesh, or scaffold.
    """
    from ._adaptive_polygon_replay import replay_history_attempt

    declared = np.asarray(surface.evidence.declared_domain, dtype=np.float64)
    policy = surface.policy
    anchor = surface.evidence.lattice_anchor
    if not 0 <= anchor <= len(_LATTICE_ANCHORS):
        return False
    for previous_anchor in range(anchor):
        previous_domain = _anchored_domain(
            declared, policy.relative_cell_diameter, previous_anchor
        )
        previous_policy = (
            policy
            if previous_anchor == 0
            else replace(
                policy,
                relative_cell_diameter=_anchored_relative_diameter(
                    declared, previous_domain, policy.relative_cell_diameter
                ),
            )
        )
        while True:
            retry, coincident = replay_history_attempt(
                surface.geometry, previous_domain, previous_policy
            )
            if not retry:
                break
            previous_policy = replace(
                previous_policy,
                relative_cell_diameter=previous_policy.relative_cell_diameter * 0.5,
            )
        if not coincident:
            return False

    selected_domain = _anchored_domain(declared, policy.relative_cell_diameter, anchor)
    if not np.array_equal(selected_domain, surface.scaffold.domain):
        return False
    effective_policy = (
        policy
        if anchor == 0
        else replace(
            policy,
            relative_cell_diameter=_anchored_relative_diameter(
                declared, selected_domain, policy.relative_cell_diameter
            ),
        )
    )
    target_level = surface.scaffold.target_level
    while (
        _target_level(selected_domain, effective_policy.relative_cell_diameter)
        < target_level
    ):
        retry, _ = replay_history_attempt(
            surface.geometry, selected_domain, effective_policy
        )
        if not retry:
            return False
        effective_policy = replace(
            effective_policy,
            relative_cell_diameter=effective_policy.relative_cell_diameter * 0.5,
        )
    if (
        _target_level(selected_domain, effective_policy.relative_cell_diameter)
        != target_level
    ):
        return False
    if _next_discovery_resolution(surface, effective_policy) is not None:
        return False
    return anchor == len(_LATTICE_ANCHORS) or not _vertex_roots_are_lattice_coincidences(
        surface.geometry, surface, policy.enclosure
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
    """Discover every possible zero, refining unresolved cell contours as needed.

    The discovery lattice is internal. A lattice vertex whose point enclosure
    contains zero while the field takes both certified signs on its small
    neighbour stencil is a coincidence of where the lattice falls on a crossing of
    the zero set, so discovery restarts on the next deterministic re-anchored
    lattice covering the declared domain at the declared physical resolution. Any
    other singular cell (a touching or degenerate vertex zero, or a possible zero
    without a corner sign change or a nonvanishing gradient component) is refused
    on the current lattice without re-anchoring. When every anchor still places a
    vertex on a crossing, the last lattice's singular cells remain the refusal.
    """
    declared = np.array(domain, dtype=np.float64, copy=True)
    anchor = 0
    while True:
        surface = _discover_on_lattice(geometry, declared, anchor, policy, source_id)
        if anchor == len(_LATTICE_ANCHORS) or not _vertex_roots_are_lattice_coincidences(
            geometry, surface, policy.enclosure
        ):
            return surface
        anchor += 1
        del surface


def restore_adaptive_implicit_surface(
    recipe: AdaptiveImplicitSurfaceArchiveRecipe,
    geometry: CompiledGeometry,
    policy: AdaptiveImplicitSurfacePolicy,
    /,
) -> AdaptiveImplicitSurface:
    """Reconstruct and independently reprove the retained discovery state."""
    from .._implicit_source_coverage import (
        _check_discovery_metadata,
        _check_leaf_replay,
        _check_scaffold_lattice,
    )
    from ._adaptive_polygon_replay import replay_polygon

    if type(recipe) is not AdaptiveImplicitSurfaceArchiveRecipe:
        raise TypeError("Expected an adaptive implicit surface archive recipe.")
    _validate_discovery_inputs(geometry, policy, recipe.source_id)
    if (
        array_tree_fingerprint(recipe.geometry) != array_tree_fingerprint(geometry)
        or recipe.policy != policy
        or type(recipe.source_id) is not str
        or not recipe.source_id
        or recipe.mesh_source_id != recipe.source_id
    ):
        raise ValueError("Adaptive discovery archive source binding is invalid.")
    scaffold_names = tuple(
        name
        for name in AdaptiveImplicitScaffold.__annotations__
        if name != "target_level"
    )
    groups = (
        (recipe.mesh_arrays, (np.float64, np.int32)),
        (recipe.cover_arrays, (np.float64,) * 6),
        (recipe.evidence_arrays, (np.float64, np.int64, np.float64)),
        (
            recipe.scaffold_arrays,
            (
                np.float64,
                np.int64,
                np.float64,
                np.int64,
                np.float64,
                np.float64,
                np.float64,
                np.float64,
                np.int64,
                np.int64,
                np.float64,
                np.float64,
                np.float64,
                np.float64,
                np.int8,
                np.int64,
                np.int64,
                np.int64,
                np.float64,
                np.float64,
                np.float64,
                np.int64,
                np.float64,
                np.float64,
                np.int64,
                np.int64,
            ),
        ),
        (recipe.boundary_arrays, (np.int64, np.float64, np.float64, np.float64)),
    )
    for arrays, dtypes in groups:
        if type(arrays) is not tuple or len(arrays) != len(dtypes):
            raise ValueError("Adaptive discovery archive has an invalid field set.")
        if any(
            np.asarray(array).dtype != np.dtype(dtype)
            for array, dtype in zip(arrays, dtypes, strict=True)
        ):
            raise ValueError("Adaptive discovery archive has an invalid array dtype.")
    if not np.array_equal(recipe.domain, recipe.evidence_arrays[2]):
        raise ValueError("Adaptive discovery archive declared domain differs.")
    if (
        type(recipe.cover_metadata) is not tuple
        or tuple(map(type, recipe.cover_metadata)) != (str, str, str, bool, bool, str)
        or type(recipe.evidence_metadata) is not tuple
        or tuple(map(type, recipe.evidence_metadata))
        != (str, str, int, int, int, int, float, int, str)
        or type(recipe.scaffold_target_level) is not int
    ):
        raise ValueError("Adaptive discovery archive has invalid metadata.")

    def ownership(arrays: tuple[np.ndarray, ...]) -> object:
        # A read-only flag on an owning ndarray is reversible by its caller.
        # Only immutable backing can safely be retained without copying.
        if all(_immutable_archive_array(a) for a in arrays):
            return _IMMUTABLE_ARCHIVE_OWNERSHIP
        return None

    mesh = TriangleMesh(*recipe.mesh_arrays, source_id=recipe.mesh_source_id)
    cover = AdaptiveImplicitCover(
        *recipe.cover_arrays,
        *recipe.cover_metadata,
        _ownership=ownership(recipe.cover_arrays[:5]),
    )
    (
        accuracy,
        enclosure,
        status,
        leaf_count,
        surface_leaf_count,
        maximum_level,
        maximum_diagonal,
        lattice_anchor,
        evidence_id,
    ) = recipe.evidence_metadata
    evidence = AdaptiveImplicitSurfaceEvidence(
        recipe.evidence_arrays[0],
        recipe.evidence_arrays[1],
        accuracy,
        enclosure,
        status,
        leaf_count,
        surface_leaf_count,
        maximum_level,
        maximum_diagonal,
        evidence_id,
        declared_domain=recipe.evidence_arrays[2],
        lattice_anchor=lattice_anchor,
        _ownership=ownership(recipe.evidence_arrays[:2]),
    )
    scaffold = AdaptiveImplicitScaffold(
        **dict(zip(scaffold_names, recipe.scaffold_arrays, strict=True)),
        target_level=recipe.scaffold_target_level,
        _ownership=ownership(recipe.scaffold_arrays[1:]),
    )
    bounds = _FieldEnclosures(geometry, policy.enclosure)
    state_id = _state_identity(geometry)
    volume = ImplicitVolumeQuery(
        geometry,
        bounds,
        cover.domain,
        recipe.source_id,
        bounds.value_rigorous,
        canonical_fingerprint(
            {
                "kind": "implicit-volume-query",
                "state": state_id,
                "domain": array_tree_fingerprint(cover.domain),
                "enclosure": policy.enclosure,
            }
        ),
    )
    surface = AdaptiveImplicitSurface(
        mesh,
        evidence,
        cover,
        None,
        volume,
        scaffold,
        *recipe.boundary_arrays,
        geometry,
        policy,
        recipe.source_id,
        _ownership=ownership(recipe.boundary_arrays),
    )
    reconstructed = (
        (mesh.vertices, mesh.faces),
        (
            cover.boxes,
            cover.value_lower,
            cover.value_upper,
            cover.gradient_lower,
            cover.gradient_upper,
            cover.domain,
        ),
        (evidence.unresolved_boxes, evidence.unresolved_issues, evidence.declared_domain),
        tuple(getattr(scaffold, name) for name in scaffold_names),
        (
            surface.feature_edges,
            surface.boundary_witness_boxes,
            surface.boundary_witnesses,
            surface.boundary_errors,
        ),
    )
    for (stored, _), restored in zip(groups, reconstructed, strict=True):
        if any(np.shape(a) != np.shape(b) for a, b in zip(stored, restored, strict=True)):
            raise ValueError("Adaptive discovery archive has an invalid array shape.")
    if (
        not bounds.value_rigorous
        or not bounds.gradient_rigorous
        or evidence.unresolved_count
        or evidence.status != 0
        or not mesh.topology.watertight
        or np.any(scaffold.cell_issues != 0)
        or np.any(scaffold.root_issues != 0)
        or np.any(scaffold.corner_signs == 0)
    ):
        raise ValueError("Adaptive discovery archive is not a complete resolved source.")
    _check_leaf_replay(surface, bounds)
    _check_scaffold_lattice(surface, bounds)
    _check_discovery_metadata(surface, bounds)
    if not replay_polygon(surface, bounds) or not verify_discovery_history(surface):
        raise ValueError("Adaptive discovery archive failed independent replay.")
    return surface


__all__ = [
    "AdaptiveImplicitBoundarySource",
    "AdaptiveImplicitCover",
    "AdaptiveImplicitSurface",
    "AdaptiveImplicitSurfaceArchiveRecipe",
    "AdaptiveImplicitScaffold",
    "AdaptiveImplicitSurfaceEvidence",
    "ImplicitCoverVolume",
    "ImplicitVolumeClass",
    "ImplicitVolumeClassification",
    "ImplicitVolumeQuery",
    "discover_adaptive_implicit_surface",
]
