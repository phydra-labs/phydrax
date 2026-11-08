#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
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
    # The program schedules its own blocks; block sizes never change results.
    if points.shape[0] == 0:
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty.copy()
    lower, upper = program.evaluate(points, points)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _evaluate_interval_boxes(
    program: PreparedIntervalFunction, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if lower.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    bound_lower, bound_upper = program.evaluate(lower, upper)
    return np.asarray(bound_lower, dtype=np.float64), np.asarray(
        bound_upper, dtype=np.float64
    )


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
    _branches: tuple[_BranchNode | None] | None

    def __init__(
        self, geometry: CompiledGeometry, selector: ImplicitDiscoveryEnclosure
    ) -> None:
        self.geometry = geometry
        self.selector = selector
        self.certificate = geometry.field_certificate
        self.value_rigorous = selector != "sampled"
        self.gradient_rigorous = selector == "interval"
        self.continuity_rigorous = selector != "sampled"
        self._branches = None
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

    def branch_tree(self) -> _BranchNode | None:
        """The field's sharp min/max structure (`_branch_node`), prepared once.

        ``None`` without interval programs, or for a field that is not a
        sharp CSG combination of interval-evaluable children.
        """
        if self._branches is None:
            node = None
            if self.selector == "interval" and self.gradient_rigorous:
                node = _branch_node(self.geometry.kernel, self.geometry.state)
                if node is not None and node.operation == "leaf":
                    node = None
            self._branches = (node,)
        return self._branches[0]

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
    domain, ``k > 0`` the ``k``-th deterministic re-anchored lattice, reached
    because an earlier lattice placed a vertex on a sign crossing of the zero set
    or a face tangent to it (see `discover_adaptive_implicit_surface`).
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
    """Retained boundary-band P1 scaffold and source-root ancestry.

    ``local_resolutions`` rows ``(cell row, group lower index, group upper
    index, axis)`` record each constant-sign band cell decided locally. A
    lattice-face tangency (`_resolve_tangencies`) names its target-level cell
    group ``[lower, upper)``, at least two cells, and the gradient component
    proved strict on it. A sharp CSG crease cell (`_resolve_creases`) names
    itself, ``[index, index + 1)``, and the axis along which its branches
    prove the field strictly monotone. Such a cell has no issue and no
    contour of its own.
    """

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
    local_resolutions: np.ndarray
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
        local_resolutions: np.ndarray,
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
            "local_resolutions": _discovery_array(
                local_resolutions, np.int64, (-1, 2 * _DIMENSION + 2), _ownership
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
                arrays["local_resolutions"].size
                and (
                    np.any(np.diff(arrays["local_resolutions"][:, 0]) <= 0)
                    or np.min(arrays["local_resolutions"][:, 0]) < 0
                    or np.max(arrays["local_resolutions"][:, 0]) >= cell_count
                    or np.any(
                        arrays["cell_issues"][arrays["local_resolutions"][:, 0]] != 0
                    )
                )
            )
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
        # Finer lattices shrink edge-ambiguous, multiple-sheet and tangency
        # boxes toward their witnessed neighbours, so they stay refinable for
        # the cover radius even though a tangency never drives discovery.
        refinement_issues = int(
            AdaptiveImplicitBoxIssue.EDGE
            | AdaptiveImplicitBoxIssue.MULTIPLE_SHEETS
            | AdaptiveImplicitBoxIssue.TANGENCY
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
    return _constant_sign_search(
        bounds, lower, upper, signs, gradient_lower, gradient_upper
    )[0]


def _constant_sign_search(
    bounds: _FieldEnclosures,
    lower: np.ndarray,
    upper: np.ndarray,
    signs: np.ndarray,
    gradient_lower: np.ndarray,
    gradient_upper: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`_prove_constant_signs` and, per box, where its search stopped.

    Also returns the first piece midpoint, in piece order, whose point
    enclosure has the sign opposite to the corners', and the first midpoint
    of a piece binary64 cannot bisect further when no such point was found
    (``NaN`` rows where there is none). Each box's pieces evolve independently
    of the others, so all three results are row-wise.
    """
    count = int(lower.shape[0])
    points = np.full((count, _DIMENSION), np.nan, dtype=np.float64)
    stalls = np.full((count, _DIMENSION), np.nan, dtype=np.float64)
    if not count or not bounds.value_rigorous:
        return np.zeros(count, dtype=np.bool_), points, stalls
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
            return np.zeros(count, dtype=np.bool_), points * np.nan, stalls * np.nan
        value_lower = np.asarray(evidence.value_lower, dtype=np.float64)
        value_upper = np.asarray(evidence.value_upper, dtype=np.float64)
        positive = positive[:, 0]
        settled = np.where(positive, value_lower > 0.0, value_upper < 0.0)
        middle = 0.5 * low + 0.5 * high
        open_rows = np.flatnonzero(~settled)
        point_lower, point_upper = bounds.point_values(middle[open_rows])
        opposite = np.where(positive[open_rows], point_upper < 0.0, point_lower > 0.0)
        hits = open_rows[opposite]
        _, first = np.unique(owner[hits], return_index=True)
        points[owner[hits[first]]] = middle[hits[first]]
        failed[owner[hits]] = True
        splittable = (middle > low) & (middle < high)
        divisible = np.any(splittable, axis=1)
        stalled = open_rows[~divisible[open_rows]]
        stalled = stalled[~failed[owner[stalled]]]
        _, first = np.unique(owner[stalled], return_index=True)
        stalls[owner[stalled[first]]] = middle[stalled[first]]
        failed[owner[stalled]] = True
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
    return ~failed, points, stalls


# Width, as a fraction of a box's extent, of the face slab that confines the
# possible zeros of a lattice-face tangency (``AdaptiveImplicitBoxIssue.TANGENCY``).
_TANGENCY_SLAB = 2.0**-8


def _tangency_faces(
    bounds: _FieldEnclosures,
    lower: np.ndarray,
    upper: np.ndarray,
    signs: np.ndarray,
    gradient_lower: np.ndarray,
    gradient_upper: np.ndarray,
    /,
) -> np.ndarray:
    """Faces of each box whose thin slab holds every possible zero of the box.

    Column ``2 * axis + side`` of row ``i`` is true when ``signs[i] F`` is proved
    positive (`_prove_constant_signs`) on box ``i`` with the slab of that face
    (``side`` 0 the lower, 1 the upper face), of width ``_TANGENCY_SLAB`` times
    the box extent along ``axis``, removed. Every possible zero of the box then
    lies in that slab. A slab bound that rounds onto a box face proves nothing.
    """
    count = int(lower.shape[0])
    faces = np.zeros((count, 6), dtype=np.bool_)
    if not count:
        return faces
    width = (upper - lower) * _TANGENCY_SLAB
    low = np.repeat(lower[:, None, :], 6, axis=1)
    high = np.repeat(upper[:, None, :], 6, axis=1)
    for axis in range(_DIMENSION):
        low[:, 2 * axis, axis] = lower[:, axis] + width[:, axis]
        high[:, 2 * axis + 1, axis] = upper[:, axis] - width[:, axis]
    valid = np.all(low < high, axis=2)
    for axis in range(_DIMENSION):
        valid[:, 2 * axis] &= low[:, 2 * axis, axis] > lower[:, axis]
        valid[:, 2 * axis + 1] &= high[:, 2 * axis + 1, axis] < upper[:, axis]
    rows, columns = np.nonzero(valid)
    faces[rows, columns] = _prove_constant_signs(
        bounds,
        low[rows, columns],
        high[rows, columns],
        np.asarray(signs)[rows],
        np.asarray(gradient_lower)[rows],
        np.asarray(gradient_upper)[rows],
    )
    return faces


def _tangency_slabs(
    bounds: _FieldEnclosures, boxes: np.ndarray, /
) -> tuple[tuple[int, float, float], ...]:
    """Closed ``(axis, low, high)`` neighbourhoods of every tangency face of ``boxes``.

    Each box is a ``TANGENCY`` box: corners of one strict sign and possible
    zeros confined to the slab of one or more faces (`_tangency_faces`, with
    the box's own value and gradient enclosures). The neighbourhood of a face
    at coordinate ``p`` with slab width ``w`` is ``[p - w, p + w]`` widened
    outward by one rounding step; a lattice with no plane in it separates the
    face. A box whose lower corner has no strict sign proves no face.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape((-1, 2, _DIMENSION))
    if not boxes.shape[0]:
        return ()
    lower = np.ascontiguousarray(boxes[:, 0])
    upper = np.ascontiguousarray(boxes[:, 1])
    evidence = bounds.boxes(lower, upper)
    corner_lower, corner_upper = bounds.point_values(lower)
    signs = np.where(corner_lower > 0.0, 1, np.where(corner_upper < 0.0, -1, 0))
    strict = signs != 0
    faces = np.zeros((boxes.shape[0], 6), dtype=np.bool_)
    faces[strict] = _tangency_faces(
        bounds,
        lower[strict],
        upper[strict],
        signs[strict],
        np.asarray(evidence.gradient_lower, dtype=np.float64)[strict],
        np.asarray(evidence.gradient_upper, dtype=np.float64)[strict],
    )
    slabs: list[tuple[int, float, float]] = []
    for row, column in zip(*np.nonzero(faces), strict=True):
        axis = int(column) // 2
        plane = float(boxes[row, int(column) % 2, axis])
        width = float((upper[row, axis] - lower[row, axis]) * _TANGENCY_SLAB)
        slabs.append(
            (
                axis,
                math.nextafter(plane - width, -math.inf),
                math.nextafter(plane + width, math.inf),
            )
        )
    return tuple(slabs)


def _hug_groups(
    axis: int, side: int, /
) -> tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]:
    """Cell groups that may resolve a tangency on face ``(axis, side)``.

    Each group is an index box ``[lower, upper)`` relative to the tangency cell:
    the pair across the face, then the 2x2 blocks around each edge of the face,
    then the 2x2x2 blocks around each of its vertices, in a fixed order.
    """
    tangential = tuple(other for other in range(_DIMENSION) if other != axis)

    def group(sides: dict[int, int]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        lower = [0, 0, 0]
        upper = [1, 1, 1]
        for direction, toward in sides.items():
            lower[direction], upper[direction] = (-1, 1) if toward == 0 else (0, 2)
        return (lower[0], lower[1], lower[2]), (upper[0], upper[1], upper[2])

    groups = [group({axis: side})]
    for other in tangential:
        for toward in range(2):
            groups.append(group({axis: side, other: toward}))
    for first, second in itertools.product(range(2), repeat=2):
        groups.append(group({axis: side, tangential[0]: first, tangential[1]: second}))
    return tuple(groups)


def _hug_certificate(
    bounds: _FieldEnclosures,
    domain: np.ndarray,
    level: int,
    lower: tuple[int, int, int],
    upper: tuple[int, int, int],
    /,
) -> int | None:
    """Graph axis proving the target-level cell group ``[lower, upper)``, if any.

    The group ``G`` is the box of lattice cells between the binary64 lattice
    planes, the partition the polygonizer contours. It is proved when

    1. one gradient component ``c`` is strict on ``G`` (one box enclosure), so
       every component of ``Z ∩ G`` is a graph along ``c`` that reaches ``∂G``;
    2. every lattice edge on ``∂G`` holds exactly the zeros its corner signs
       name: monotone along it, or zero-free without a sign change;
    3. every lattice face on ``∂G`` has 0 or 2 crossing edges and holds exactly
       one arc between them, or none: a strict tangential gradient component
       excludes closed curves, or a strict value excludes every zero;
    4. those arcs form at most one loop ``L``, so ``Z ∩ G`` is one disk spanning
       ``L``, or empty;
    5. the cell contours of ``G``, whose cycles are fixed by the corner signs
       (no face with four crossings), form one connected complex of Euler
       characteristic 1 spanning ``L``, a disk, or nothing when ``L`` is empty.

    The contoured surface then agrees with the zero set on ``G`` up to a
    homeomorphism fixing ``∂G``. Corner signs are evaluated at the binary64
    lattice points, as everywhere else in discovery.
    """
    span = 1 << level
    counts = tuple(upper[axis] - lower[axis] for axis in range(_DIMENSION))
    if any(lower[axis] < 0 or upper[axis] > span for axis in range(_DIMENSION)):
        return None
    planes = tuple(
        np.asarray(
            [
                _grid_coordinate(
                    lower[axis] + offset, level, domain[0, axis], domain[1, axis]
                )
                for offset in range(counts[axis] + 1)
            ],
            dtype=np.float64,
        )
        for axis in range(_DIMENSION)
    )
    grid = np.stack(np.meshgrid(*planes, indexing="ij"), axis=-1)
    try:
        point_lower, point_upper = bounds.point_values(grid.reshape(-1, _DIMENSION))
        hull = bounds.boxes(grid[0, 0, 0][None, :], grid[-1, -1, -1][None, :])
    except IntervalExtensionError:
        return None
    signs = np.where(point_lower > 0.0, 1, np.where(point_upper < 0.0, -1, 0)).reshape(
        grid.shape[:3]
    )
    strict = (np.asarray(hull.gradient_lower)[0] > 0.0) | (
        np.asarray(hull.gradient_upper)[0] < 0.0
    )
    if np.any(signs == 0) or not np.any(strict):
        return None

    def step(point: _GridIndex, axis: int, /) -> _GridIndex:
        moved = [point[0], point[1], point[2]]
        moved[axis] += 1
        return (moved[0], moved[1], moved[2])

    def crossing(edge: tuple[int, _GridIndex], /) -> bool:
        axis, start = edge
        return bool(signs[start] != signs[step(start, axis)])

    def face_edges(
        axis: int, corner: _GridIndex, /
    ) -> tuple[tuple[int, _GridIndex], ...]:
        first, second = (other for other in range(_DIMENSION) if other != axis)
        return (
            (first, corner),
            (first, step(corner, second)),
            (second, corner),
            (second, step(corner, first)),
        )

    def components(links: Sequence[tuple[object, object]], /) -> int:
        parent: dict[object, object] = {}

        def find(node: object, /) -> object:
            parent.setdefault(node, node)
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for first, second in links:
            parent[find(first)] = find(second)
        return len({find(node) for node in list(parent)})

    def cyclic(links: Sequence[tuple[object, object]], /) -> bool:
        """Whether ``links`` form disjoint cycles: every endpoint used twice."""
        degree: dict[object, int] = {}
        for first, second in links:
            degree[first] = degree.get(first, 0) + 1
            degree[second] = degree.get(second, 0) + 1
        return all(value == 2 for value in degree.values())

    # Boundary edges and faces the hull does not prove are decided with one
    # batch of their own enclosures.
    open_edges: list[tuple[int, _GridIndex]] = []
    for axis in range(_DIMENSION):
        others = tuple(other for other in range(_DIMENSION) if other != axis)
        for start in itertools.product(
            *(range(counts[other] + (other != axis)) for other in range(_DIMENSION))
        ):
            point = (start[0], start[1], start[2])
            if not strict[axis] and any(
                point[other] in (0, counts[other]) for other in others
            ):
                open_edges.append((axis, point))
    open_faces: list[tuple[int, _GridIndex, bool]] = []
    arcs: list[tuple[tuple[int, _GridIndex], tuple[int, _GridIndex]]] = []
    for axis in range(_DIMENSION):
        first, second = (other for other in range(_DIMENSION) if other != axis)
        for position in sorted({0, counts[axis]}):
            for a, b in itertools.product(range(counts[first]), range(counts[second])):
                start = [0, 0, 0]
                start[axis], start[first], start[second] = position, a, b
                corner = (start[0], start[1], start[2])
                crossed = tuple(
                    edge for edge in face_edges(axis, corner) if crossing(edge)
                )
                if len(crossed) not in (0, 2):
                    return None
                if crossed:
                    arcs.append((crossed[0], crossed[1]))
                if not (strict[first] or strict[second]):
                    open_faces.append((axis, corner, bool(crossed)))
    if open_edges or open_faces:
        low = [grid[start] for _, start in open_edges] + [
            grid[corner] for _, corner, _ in open_faces
        ]
        high = [grid[step(start, axis)] for axis, start in open_edges] + [
            grid[step(step(corner, (axis + 1) % 3), (axis + 2) % 3)]
            for axis, corner, _ in open_faces
        ]
        try:
            evidence = bounds.boxes(np.stack(low), np.stack(high))
        except IntervalExtensionError:
            return None
        free = (np.asarray(evidence.value_lower) > 0.0) | (
            np.asarray(evidence.value_upper) < 0.0
        )
        monotone = (np.asarray(evidence.gradient_lower) > 0.0) | (
            np.asarray(evidence.gradient_upper) < 0.0
        )
        for row, edge in enumerate(open_edges):
            if not (monotone[row, edge[0]] or (free[row] and not crossing(edge))):
                return None
        for row, (axis, _, crossed) in enumerate(open_faces, len(open_edges)):
            tangential = [other for other in range(_DIMENSION) if other != axis]
            if not (np.any(monotone[row, tangential]) or (free[row] and not crossed)):
                return None
    if not cyclic(arcs) or components(arcs) > 1:
        return None

    # Contour cycles of the group's cells, fixed by their corner signs.
    segments: set[tuple[tuple[int, _GridIndex], tuple[int, _GridIndex]]] = set()
    cycles = 0
    for cell in itertools.product(*(range(count) for count in counts)):
        origin = (cell[0], cell[1], cell[2])
        local: list[tuple[tuple[int, _GridIndex], tuple[int, _GridIndex]]] = []
        for axis, side in itertools.product(range(_DIMENSION), range(2)):
            corner = step(origin, axis) if side else origin
            crossed = tuple(edge for edge in face_edges(axis, corner) if crossing(edge))
            if len(crossed) == 4:
                return None
            if crossed:
                local.append((min(crossed), max(crossed)))
        if not local:
            continue
        if not cyclic(local) or components(local) != 1:
            return None
        cycles += 1
        segments.update(local)
    vertices = {vertex for segment in segments for vertex in segment}
    if not arcs:
        return int(np.flatnonzero(strict)[0]) if not cycles else None
    if components(sorted(segments)) != 1 or len(vertices) - len(segments) + cycles != 1:
        return None
    return int(np.flatnonzero(strict)[0])


# Half-lengths of the axis segments bracketing a zero next to a stalled point
# of a tangency cell: 2**-k of the cell extent for k = _WITNESS_STEPS..0.
_WITNESS_STEPS = 44


def _tangency_witness(
    bounds: _FieldEnclosures,
    corner: np.ndarray,
    extent: np.ndarray,
    sign: int,
    opposite: np.ndarray,
    stall: np.ndarray,
    tolerance: float,
    /,
) -> tuple[np.ndarray, float] | None:
    """A point of a tangency cell and a certified bound on its distance to ``Z``.

    With a point of the opposite sign in the cell, the witness is the root
    bracketed between it and the cell's lattice corner ``corner``
    (`_root_on_edges`): a zero in the cell, with its root error. A zero set
    that only touches the cell (a pole on a lattice face) has no such point;
    then the witness is the stalled point ``stall`` of the cell's search, and
    its bound is the half-length of the shortest axis segment centred there
    whose endpoints have strict opposite signs, which therefore holds a zero.
    ``None`` when neither is available.
    """
    if np.all(np.isfinite(opposite)):
        point, error, issue = _root_on_edges(
            bounds,
            np.stack((opposite, corner))[None, :, :],
            np.asarray([[-sign, sign]], dtype=np.int8),
            tolerance,
        )[0][:3]
        return None if issue else (np.asarray(point, dtype=np.float64), float(error))
    if not np.all(np.isfinite(stall)):
        return None
    steps = [
        (axis, 2.0 ** (-power) * float(extent[axis]))
        for axis in range(_DIMENSION)
        for power in range(_WITNESS_STEPS, -1, -1)
    ]
    ends = np.empty((len(steps), 2, _DIMENSION), dtype=np.float64)
    for row, (axis, length) in enumerate(steps):
        ends[row] = stall
        ends[row, 0, axis] = stall[axis] - length
        ends[row, 1, axis] = stall[axis] + length
    value_lower, value_upper = bounds.point_values(ends.reshape(-1, _DIMENSION))
    signs = np.where(value_lower > 0.0, 1, np.where(value_upper < 0.0, -1, 0)).reshape(
        -1, 2
    )
    bracketed = np.flatnonzero(signs[:, 0] * signs[:, 1] == -1)
    if not len(bracketed):
        return None
    row = int(bracketed[0])
    axis = steps[row][0]
    error = float(
        np.nextafter(
            max(stall[axis] - ends[row, 0, axis], ends[row, 1, axis] - stall[axis]),
            np.inf,
        )
    )
    return np.array(stall, dtype=np.float64), error


def _resolve_tangencies(
    bounds: _FieldEnclosures,
    domain: np.ndarray,
    level: int,
    indices: np.ndarray,
    faces: np.ndarray,
    signs: np.ndarray,
    opposite: np.ndarray,
    stalls: np.ndarray,
    tolerance: float,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decide each tangency cell locally on a group of lattice cells around it.

    Row ``i`` is a target-level cell index whose zeros lie in the slabs of the
    faces marked in ``faces[i]`` (`_tangency_faces`), whose corners have sign
    ``signs[i]``, and where ``opposite[i]`` and ``stalls[i]`` are where its
    constant-sign search stopped (`_constant_sign_search`). The cell first
    needs a witness, a point of the cell within a certified distance of the
    zero set like every other band cell's root (`_tangency_witness`). The
    candidate groups of each marked face (`_hug_groups`) are then tried in
    order, and the first proved one (`_hug_certificate`) resolves the cell:
    its own contour is empty, because the contours of its group already span
    the zero set there.

    Returns the group's absolute index box ``[lower, upper)`` and graph axis
    per row (``-1`` for a refusal), and the witness points and distance
    bounds (``NaN`` for a refusal).
    """
    count = len(indices)
    result = np.full((count, 2 * _DIMENSION + 1), -1, dtype=np.int64)
    witness_points = np.full((count, _DIMENSION), np.nan, dtype=np.float64)
    witness_errors = np.full(count, np.nan, dtype=np.float64)
    if not (bounds.value_rigorous and bounds.gradient_rigorous):
        return result, witness_points, witness_errors
    for row in range(count):
        cell = tuple(int(value) for value in indices[row])
        corner, far = _lattice_cell_corners(domain, level, cell)
        witness = _tangency_witness(
            bounds,
            corner,
            far - corner,
            int(signs[row]),
            opposite[row],
            stalls[row],
            tolerance,
        )
        if witness is None:
            continue
        for column in np.flatnonzero(faces[row]).tolist():
            for offset_lower, offset_upper in _hug_groups(column // 2, column % 2):
                lower = (
                    cell[0] + offset_lower[0],
                    cell[1] + offset_lower[1],
                    cell[2] + offset_lower[2],
                )
                upper = (
                    cell[0] + offset_upper[0],
                    cell[1] + offset_upper[1],
                    cell[2] + offset_upper[2],
                )
                axis = _hug_certificate(bounds, domain, level, lower, upper)
                if axis is not None:
                    result[row] = (*lower, *upper, axis)
                    witness_points[row], witness_errors[row] = witness
                    break
            if result[row, -1] >= 0:
                break
    return result, witness_points, witness_errors


@dataclass(frozen=True, slots=True)
class _BranchNode:
    """One node of a sharp CSG field: a leaf field, or a min/max of signed children.

    A ``min`` (sharp union) or ``max`` (sharp intersection, or a difference
    with its subtrahends negated) node's value is that extremum of
    ``signs[k]`` times child ``k``'s value. A ``leaf`` carries interval
    programs of its own value and gradient.
    """

    operation: str
    signs: tuple[int, ...]
    children: tuple[_BranchNode, ...]
    values: PreparedIntervalFunction | None
    gradients: PreparedIntervalFunction | None


def _ledger_crease_cells(
    bounds: _FieldEnclosures, lower: np.ndarray, upper: np.ndarray, /
) -> np.ndarray:
    """Boxes of a non-smooth field that its branch ledger does not prove smooth.

    One evaluation of the discovery's interval value program decides each
    box (`PreparedIntervalFunction.evaluate_branches`). Only such a box may
    lack a strict regular-gradient enclosure in a root scaffold: there the
    branches of a sharp min or max may switch, and its faces take the root
    witness bound. A box of a smooth field, or one the ledger proves smooth,
    is never a crease box.
    """
    count = int(np.asarray(lower).shape[0])
    values = bounds._values
    if (
        not count
        or bounds.certificate.regularity is FieldRegularity.SMOOTH
        or bounds.selector != "interval"
        or values is None
    ):
        return np.zeros(count, dtype=np.bool_)
    _, _, smooth, _ = values.evaluate_branches(
        np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    )
    return ~np.asarray(smooth, dtype=np.bool_).reshape((-1,))


def _branch_node(kernel: object, state: object, /) -> _BranchNode | None:
    """The sharp min/max tree of a kernel's field, or ``None`` without programs."""
    from ..analytic._expressions import _UnionKernel
    from ..analytic._operations import _SharpCSGKernel

    if isinstance(kernel, (_UnionKernel, _SharpCSGKernel)):
        count = len(kernel.children)
        if isinstance(kernel, _UnionKernel):
            operation, signs = "min", (1,) * count
        elif kernel.operation == "intersection":
            operation, signs = "max", (1,) * count
        else:
            operation, signs = "max", (1, *((-1,) * (count - 1)))
        children = tuple(_branch_node(child, state) for child in kernel.children)
        if any(child is None for child in children):
            return None
        return _BranchNode(
            operation,
            signs,
            tuple(child for child in children if child is not None),
            None,
            None,
        )
    boundary_field = getattr(kernel, "boundary_field", None)
    if boundary_field is None:
        return None

    def value_function(points: jax.Array) -> jax.Array:
        return boundary_field(state, points)

    def gradient_function(points: jax.Array) -> jax.Array:
        def scalar(point: jax.Array) -> jax.Array:
            return boundary_field(state, point[None, :])[0]

        return jax.grad(scalar)(points)

    try:
        return _BranchNode(
            "leaf",
            (),
            (),
            prepare_interval_function(value_function, _DIMENSION),
            prepare_interval_function(gradient_function, _DIMENSION),
        )
    except IntervalExtensionError:
        return None


def _branch_bounds(
    node: _BranchNode, lower: np.ndarray, upper: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Value enclosure and proved strict monotonicity of a branch tree per box.

    Returns value lower and upper bounds and, per box and axis, ``+1`` or
    ``-1`` where the field is proved strictly increasing or decreasing along
    that axis on the box (``0`` otherwise). A leaf is monotone where its
    interval program's branch ledger proves it smooth on the box, so its
    gradient program encloses the true gradient there, and that enclosure
    has the sign. A min or max node drops children proved never extremal on
    the box (a child of a min whose lower bound exceeds another's upper
    bound) and is monotone along an axis where every remaining signed child
    is, in one direction: a min or max of strictly increasing functions is
    strictly increasing, with no derivative of the extremum itself.
    """
    if node.operation == "leaf":
        values, gradients = node.values, node.gradients
        if values is None or gradients is None:
            raise RuntimeError("A branch leaf has no interval programs.")
        value_lower, value_upper, smooth, _ = values.evaluate_branches(lower, upper)
        slope_lower, slope_upper = _evaluate_interval_boxes(gradients, lower, upper)
        smooth = np.asarray(smooth, dtype=np.bool_)[:, None]
        return (
            np.asarray(value_lower, dtype=np.float64).reshape((-1,)),
            np.asarray(value_upper, dtype=np.float64).reshape((-1,)),
            np.where(
                smooth & (slope_lower > 0.0),
                1,
                np.where(smooth & (slope_upper < 0.0), -1, 0),
            ).astype(np.int8),
        )
    lows, highs, slopes = [], [], []
    for sign, child in zip(node.signs, node.children, strict=True):
        low, high, slope = _branch_bounds(child, lower, upper)
        if sign < 0:
            low, high, slope = -high, -low, -slope
        lows.append(low)
        highs.append(high)
        slopes.append(slope)
    low, high, slope = np.stack(lows), np.stack(highs), np.stack(slopes)
    if node.operation == "min":
        active = low <= np.min(high, axis=0)
        value = (np.min(low, axis=0), np.min(high, axis=0))
    else:
        active = high >= np.max(low, axis=0)
        value = (np.max(low, axis=0), np.max(high, axis=0))
    increasing = np.all(~active[:, :, None] | (slope > 0), axis=0)
    decreasing = np.all(~active[:, :, None] | (slope < 0), axis=0)
    return (
        value[0],
        value[1],
        np.where(increasing, 1, np.where(decreasing, -1, 0)).astype(np.int8),
    )


def _lattice_cell_corners(
    domain: np.ndarray, level: int, cell: tuple[int, ...], /
) -> tuple[np.ndarray, np.ndarray]:
    """Binary64 lattice points of a target-level cell's lower and upper corners."""

    def point(offset: int, /) -> np.ndarray:
        return np.asarray(
            [
                _grid_coordinate(
                    cell[axis] + offset, level, domain[0, axis], domain[1, axis]
                )
                for axis in range(_DIMENSION)
            ],
            dtype=np.float64,
        )

    return point(0), point(1)


def _resolve_creases(
    bounds: _FieldEnclosures,
    domain: np.ndarray,
    level: int,
    indices: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    signs: np.ndarray,
    opposite: np.ndarray,
    stalls: np.ndarray,
    tolerance: float,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decide constant-sign cells without a strict gradient on a sharp CSG crease.

    The field's own gradient enclosure cannot be strict where the branches of
    a sharp min or max switch. Cell ``i`` (corners of sign ``signs[i]``,
    search results ``opposite[i]`` and ``stalls[i]`` from
    `_constant_sign_search`) is resolved when

    1. the branch tree (`_FieldEnclosures.branch_tree`) proves the field
       strictly monotone along an axis ``c`` on the cell (`_branch_bounds`):
       every component of the zero set in the cell is then a graph along
       ``c`` that reaches the cell boundary, so none is hidden inside it;
    2. a crease is certified in the cell: two of its corners have different
       top-level branches strictly extremal there, so the switching set of
       those branches meets the cell; and
    3. the cell has a witness (`_tangency_witness`).

    The cell is then contoured by its corner signs, with no triangle, as any
    crease cell is: its zeros lie in the cell, within its witness's reach.
    Returns, per row, the cell's own index box ``[index, index + 1)`` and
    ``c`` (``-1`` for a refusal), and the witness points and bounds.
    """
    count = len(indices)
    result = np.full((count, 2 * _DIMENSION + 1), -1, dtype=np.int64)
    witness_points = np.full((count, _DIMENSION), np.nan, dtype=np.float64)
    witness_errors = np.full(count, np.nan, dtype=np.float64)
    tree = bounds.branch_tree() if bounds.value_rigorous else None
    if tree is None or not count:
        return result, witness_points, witness_errors
    try:
        _, _, slopes = _branch_bounds(tree, lower, upper)
        corners = np.where(
            _CORNER_OFFSETS[None, :, :] == 0, lower[:, None, :], upper[:, None, :]
        ).reshape(-1, _DIMENSION)
        corner_values = []
        for sign, child in zip(tree.signs, tree.children, strict=True):
            low, high, _ = _branch_bounds(child, corners, corners)
            corner_values.append((-high, -low) if sign < 0 else (low, high))
    except IntervalExtensionError:
        return result, witness_points, witness_errors
    low = np.stack([value[0] for value in corner_values]).reshape(-1, count, 8)
    high = np.stack([value[1] for value in corner_values]).reshape(-1, count, 8)
    extremal = np.full((count, 8), -1, dtype=np.int64)
    for branch in range(low.shape[0]):
        others = np.delete(np.arange(low.shape[0]), branch)
        if tree.operation == "min":
            strict = np.all(high[branch][None] < low[others], axis=0)
        else:
            strict = np.all(low[branch][None] > high[others], axis=0)
        extremal[strict] = branch
    for row in range(count):
        axes = np.flatnonzero(slopes[row])
        decided = extremal[row][extremal[row] >= 0]
        if not len(axes) or len(np.unique(decided)) < 2:
            continue
        cell = tuple(int(value) for value in indices[row])
        corner, far = _lattice_cell_corners(domain, level, cell)
        witness = _tangency_witness(
            bounds,
            corner,
            far - corner,
            int(signs[row]),
            opposite[row],
            stalls[row],
            tolerance,
        )
        if witness is None:
            continue
        result[row] = (*cell, *(value + 1 for value in cell), int(axes[0]))
        witness_points[row], witness_errors[row] = witness
    return result, witness_points, witness_errors


# Probe distances, in cell extents along each axis, of the witness of a cell
# proved free of the zero set: 2**k for k = 0.._EMPTY_WITNESS_STEPS - 1.
_EMPTY_WITNESS_STEPS = 4


def _empty_cell_witnesses(
    bounds: _FieldEnclosures,
    domain: np.ndarray,
    level: int,
    indices: np.ndarray,
    signs: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Witnesses of possible-zero cells proved free of the zero set.

    The cell's enclosure admitted a zero that `_constant_sign_search`
    refuted, so no root lies in it; its witness is its centre ``c``, of the
    corners' sign, with the distance to the zero set bounded by the first
    probe ``c ± 2**k h_a e_a`` (``k`` first, then axis, then ``-`` before
    ``+``) whose point enclosure has the opposite sign: the segment between
    them holds a zero. Rows with no such probe keep ``NaN``.
    """
    count = len(indices)
    points = np.full((count, _DIMENSION), np.nan, dtype=np.float64)
    errors = np.full(count, np.nan, dtype=np.float64)
    if not count:
        return points, errors
    corners = [
        _lattice_cell_corners(domain, level, tuple(int(value) for value in row))
        for row in indices
    ]
    lower = np.stack([corner for corner, _ in corners])
    upper = np.stack([far for _, far in corners])
    centres = 0.5 * lower + 0.5 * upper
    steps = [
        (2.0**power, axis, side)
        for power in range(_EMPTY_WITNESS_STEPS)
        for axis in range(_DIMENSION)
        for side in (-1.0, 1.0)
    ]
    probes = np.repeat(centres[:, None, :], len(steps), axis=1)
    for column, (scale, axis, side) in enumerate(steps):
        probes[:, column, axis] = centres[:, axis] + side * scale * (
            upper[:, axis] - lower[:, axis]
        )
    value_lower, value_upper = bounds.point_values(probes.reshape(-1, _DIMENSION))
    opposite = np.where(
        (np.asarray(signs) > 0)[:, None],
        value_upper.reshape(count, -1) < 0.0,
        value_lower.reshape(count, -1) > 0.0,
    )
    found = np.flatnonzero(np.any(opposite, axis=1))
    first = np.argmax(opposite[found], axis=1)
    axes = np.asarray([steps[column][1] for column in first], dtype=np.int64)
    points[found] = centres[found]
    errors[found] = np.nextafter(
        np.abs(probes[found, first, axes] - centres[found, axes]), np.inf
    )
    return points, errors


def _classify_constant_sign_cells(
    bounds: _FieldEnclosures,
    domain: np.ndarray,
    level: int,
    indices: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    signs: np.ndarray,
    gradient_lower: np.ndarray,
    gradient_upper: np.ndarray,
    tolerance: float,
    /,
    *,
    resolve: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Issues of band cells whose eight corners share one strict sign.

    The one classification of the polygonizer, `replay_polygon` and the history
    replay. A cell proved free of the zero set (`_prove_constant_signs`) has no
    issue; otherwise a cell with no strict gradient component is ``SINGULAR``,
    one whose zeros lie in a face slab (`_tangency_faces`) is ``TANGENCY``, and
    any other is ``EDGE``. With ``resolve`` a tangency proved on a group of
    cells around it (`_resolve_tangencies`), and a singular cell proved to be
    a monotone sharp CSG crease cell (`_resolve_creases`), roots within
    ``tolerance``, have no issue either. Also returns the resolution rows
    (``-1`` where none), and, with ``resolve``, the witness points and
    distance bounds of the resolved cells and of the cells proved free of the
    zero set (`_empty_cell_witnesses`; ``NaN`` where none).
    """
    count = int(lower.shape[0])
    issues = np.zeros(count, dtype=np.int64)
    resolutions = np.full((count, 2 * _DIMENSION + 1), -1, dtype=np.int64)
    witness_points = np.full((count, _DIMENSION), np.nan, dtype=np.float64)
    witness_errors = np.full(count, np.nan, dtype=np.float64)
    if not count:
        return issues, resolutions, witness_points, witness_errors
    gradient = np.any((gradient_lower > 0) | (gradient_upper < 0), axis=1)
    proved, opposite, stalls = _constant_sign_search(
        bounds, lower, upper, signs, gradient_lower, gradient_upper
    )
    issues[~proved & gradient] = int(AdaptiveImplicitBoxIssue.EDGE)
    issues[~proved & ~gradient] = int(AdaptiveImplicitBoxIssue.SINGULAR)
    empty = np.flatnonzero(proved)
    if resolve and len(empty):
        witness_points[empty], witness_errors[empty] = _empty_cell_witnesses(
            bounds, domain, level, indices[empty], signs[empty]
        )
    singular = np.flatnonzero(~proved & ~gradient)
    if resolve and len(singular):
        groups, points, errors = _resolve_creases(
            bounds,
            domain,
            level,
            indices[singular],
            lower[singular],
            upper[singular],
            signs[singular],
            opposite[singular],
            stalls[singular],
            tolerance,
        )
        resolutions[singular] = groups
        witness_points[singular] = points
        witness_errors[singular] = errors
        issues[singular[groups[:, -1] >= 0]] = 0
    edge = np.flatnonzero(~proved & gradient)
    if len(edge):
        faces = _tangency_faces(
            bounds,
            lower[edge],
            upper[edge],
            signs[edge],
            gradient_lower[edge],
            gradient_upper[edge],
        )
        tangent = np.any(faces, axis=1)
        rows = edge[tangent]
        issues[rows] = int(AdaptiveImplicitBoxIssue.TANGENCY)
        if resolve and len(rows):
            groups, points, errors = _resolve_tangencies(
                bounds,
                domain,
                level,
                indices[rows],
                faces[tangent],
                signs[rows],
                opposite[rows],
                stalls[rows],
                tolerance,
            )
            resolutions[rows] = groups
            witness_points[rows] = points
            witness_errors[rows] = errors
            issues[rows[groups[:, -1] >= 0]] = 0
    return issues, resolutions, witness_points, witness_errors


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
# tried when a lattice vertex lies on a regular crossing of the zero set or a
# lattice face is tangent to it. The shifts are the binary64 (dyadic) values of
# the fractional parts of multiples of the golden ratio, sqrt(2) and sqrt(3): no
# low-denominator dyadic coincidence with authored coordinates, and the zero set
# meets any one shifted vertex family only on a measure-zero set.
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


def _exact_binary64_integers(values: list[float], /) -> list[int] | None:
    """Finite binary64 values as integers over one common power-of-two denominator.

    Sums, differences and products of the integers are exact multiples of the
    values' (positive) scale, so every sign and comparison they decide is the
    exact decision on the represented binary64 inputs, on any platform.
    """
    if not all(math.isfinite(value) for value in values):
        return None
    ratios = [float(value).as_integer_ratio() for value in values]
    denominator = max(den for _, den in ratios)
    return [numerator * (denominator // den) for numerator, den in ratios]


def _integer_cross(
    first: Sequence[int], second: Sequence[int], /
) -> tuple[int, int, int]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _integer_difference(
    first: Sequence[int], second: Sequence[int], /
) -> tuple[int, int, int]:
    return (first[0] - second[0], first[1] - second[1], first[2] - second[2])


def _integer_dot(first: Sequence[int], second: Sequence[int], /) -> int:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


# A floating-point squared edge length is within a few units in the last place
# of the exact one, so a longest edge this far ahead of the next, above the
# subnormal range, is the exact longest edge; any other triangle is decided
# exactly.
_WIDEST_CORNER_GAP = 2.0**-40
_WIDEST_CORNER_FLOOR = 2.0**-1000


def _widest_corner_offsets(corners: np.ndarray, /) -> np.ndarray:
    """Rotation offset putting first each triangle's corner opposite its longest edge.

    ``corners`` is ``(n, 3, 3)``. By the law of sines the corner opposite the
    longest edge has the largest angle sine, so the corner-0 Jacobian columns
    of the rotated triangle (which cell validity and finite-element geometry
    normalize and measure) are its best conditioned; a cyclic rotation keeps
    the triangle and its orientation. The longest edge is decided exactly on
    the binary64 corners, and equal longest edges keep the earliest corner.
    """
    points = np.asarray(corners, dtype=np.float64)
    offsets = np.zeros(points.shape[0], dtype=np.int64)
    if points.shape[0] == 0:
        return offsets
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        # The edge opposite corner k joins corners k + 1 and k + 2.
        opposite = np.roll(points, -2, axis=1) - np.roll(points, -1, axis=1)
        lengths = np.sum(opposite * opposite, axis=2)
        ordered = np.sort(lengths, axis=1)
        decided = (
            np.all(np.isfinite(lengths), axis=1)
            & (ordered[:, 2] > _WIDEST_CORNER_FLOOR)
            & (ordered[:, 2] - ordered[:, 1] > _WIDEST_CORNER_GAP * ordered[:, 2])
        )
    offsets[decided] = np.argmax(lengths[decided], axis=1)
    for row in np.flatnonzero(~decided).tolist():
        values = _exact_binary64_integers(points[row].reshape(-1).tolist())
        if values is None:
            continue
        exact = [values[3 * corner : 3 * corner + 3] for corner in range(3)]
        squared = [
            sum(
                (exact[(corner + 2) % 3][axis] - exact[(corner + 1) % 3][axis]) ** 2
                for axis in range(3)
            )
            for corner in range(3)
        ]
        offsets[row] = squared.index(max(squared))
    return offsets


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
    """Triangulate one contour cycle by exact predicates on its binary64 points.

    A quadrilateral takes the diagonal whose smaller triangle is larger (the
    first diagonal on a tie). A longer cycle clips, among its ears, the one of
    largest area, then of largest middle vertex. Convexity and containment
    are orientations about the cycle's exact Newell normal ``N`` (the sign of
    ``N . (b - a) x (c - a)``), areas compare as exact squared doubled areas:
    no rounded norm, projection or reduction order decides any choice, so
    every platform triangulates a cycle the same way.
    """
    if len(cycle) == 3:
        return ((cycle[0], cycle[1], cycle[2]),)
    values = _exact_binary64_integers(
        [float(value) for index in cycle for value in vertices[index]]
    )
    if values is None:
        return None
    points = [tuple(values[3 * row : 3 * row + 3]) for row in range(len(cycle))]

    def area(first: int, second: int, third: int, /) -> int:
        normal = _integer_cross(
            _integer_difference(points[second], points[first]),
            _integer_difference(points[third], points[first]),
        )
        return _integer_dot(normal, normal)

    if len(cycle) == 4:
        first_split = (
            (cycle[0], cycle[1], cycle[2]),
            (cycle[0], cycle[2], cycle[3]),
        )
        second_split = (
            (cycle[0], cycle[1], cycle[3]),
            (cycle[1], cycle[2], cycle[3]),
        )
        first_minimum = min(area(0, 1, 2), area(0, 2, 3))
        second_minimum = min(area(0, 1, 3), area(1, 2, 3))
        return first_split if first_minimum >= second_minimum else second_split

    normal = (0, 0, 0)
    for row in range(len(cycle)):
        term = _integer_cross(points[row], points[(row + 1) % len(cycle)])
        normal = (normal[0] + term[0], normal[1] + term[1], normal[2] + term[2])
    # A cycle winds positively about its own Newell normal (its signed area
    # about N is |N| / 2), so orientations about N need no separate sign.
    if normal == (0, 0, 0):
        return None

    def orient(first: int, second: int, third: int, /) -> int:
        return _integer_dot(
            normal,
            _integer_cross(
                _integer_difference(points[second], points[first]),
                _integer_difference(points[third], points[first]),
            ),
        )

    remaining = list(range(len(cycle)))
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        ears: list[tuple[int, int, int, tuple[int, int, int]]] = []
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following = remaining[(position + 1) % len(remaining)]
            if orient(previous, current, following) <= 0:
                continue
            contains_vertex = False
            for candidate in remaining:
                if candidate in (previous, current, following):
                    continue
                if (
                    min(
                        orient(previous, current, candidate),
                        orient(current, following, candidate),
                        orient(following, previous, candidate),
                    )
                    >= 0
                ):
                    contains_vertex = True
                    break
            if contains_vertex:
                continue
            triangle = (
                cycle[previous],
                cycle[current],
                cycle[following],
            )
            ears.append(
                (area(previous, current, following), cycle[current], position, triangle)
            )
        if not ears:
            return None
        _area, _vertex, position, triangle = max(ears, key=lambda row: row[:2])
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
    # Both scaled by one positive power of two: the sign of the integer dot
    # product is the exact sign of the Newell normal against the gradient.
    exact = _exact_binary64_integers(
        [float(value) for vertex in cycle for value in vertices[vertex]]
        + [float(value) for value in gradient]
    )
    orientation = 0
    if exact is not None:
        cycle_normal = (0, 0, 0)
        for row in range(len(cycle)):
            term = _integer_cross(
                exact[3 * row : 3 * row + 3],
                exact[3 * ((row + 1) % len(cycle)) : 3 * ((row + 1) % len(cycle)) + 3],
            )
            cycle_normal = (
                cycle_normal[0] + term[0],
                cycle_normal[1] + term[1],
                cycle_normal[2] + term[2],
            )
        orientation = _integer_dot(cycle_normal, exact[3 * len(cycle) :])
    if orientation == 0:
        issue |= int(AdaptiveImplicitBoxIssue.SINGULAR)
        witness = (
            np.zeros(3, dtype=np.float64) if first_witness is None else first_witness
        )
        return issue, witness, first_error
    if orientation < 0:
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
        offset = int(
            _widest_corner_offsets(
                np.stack([vertices[vertex] for vertex in triangle])[None]
            )[0]
        )
        faces.append(
            (
                triangle[offset],
                triangle[(offset + 1) % 3],
                triangle[(offset + 2) % 3],
            )
        )
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
    A lattice-face tangency proved on a group of cells around it
    (`_resolve_tangencies`), and a constant-sign cell on a monotone sharp CSG
    crease (`_resolve_creases`), are resolved there and recorded in the
    scaffold's ``local_resolutions``; other tangencies, singular and otherwise
    unresolved leaves remain explicit.
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
            | AdaptiveImplicitBoxIssue.TANGENCY
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
    """Return the next dyadic resolution only for a retryable discovery.

    ``TANGENCY`` boxes do not drive refinement: every finer lattice keeps the
    face they hug. The remaining unresolved boxes must all be edge-ambiguous
    or multiple-sheet boxes wider than the root tolerance.
    """
    issues = np.asarray(surface.evidence.unresolved_issues, dtype=np.int64)
    refinable = int(
        AdaptiveImplicitBoxIssue.EDGE | AdaptiveImplicitBoxIssue.MULTIPLE_SHEETS
    )
    others = issues != int(AdaptiveImplicitBoxIssue.TANGENCY)
    if not np.any(others) or np.any(issues[others] & ~refinable):
        return None
    retryable = int(
        AdaptiveImplicitSurfaceStatus.UNRESOLVED_BOXES
        | AdaptiveImplicitSurfaceStatus.PHYSICAL_RESOLUTION_REACHED
    )
    if int(surface.evidence.status_flags) & ~retryable:
        return None
    boxes = np.asarray(surface.evidence.unresolved_boxes)[others]
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


def _anchored_lattice(
    declared: np.ndarray, policy: AdaptiveImplicitSurfacePolicy, anchor: int, /
) -> tuple[np.ndarray, AdaptiveImplicitSurfacePolicy]:
    """Discovery box and first-attempt policy of lattice ``anchor``."""
    domain = _anchored_domain(declared, policy.relative_cell_diameter, anchor)
    if anchor == 0:
        return domain, policy
    return domain, replace(
        policy,
        relative_cell_diameter=_anchored_relative_diameter(
            declared, domain, policy.relative_cell_diameter
        ),
    )


def _lattice_separates(
    domain: np.ndarray, level: int, slabs: tuple[tuple[int, float, float], ...], /
) -> bool:
    """Whether no exact level-``level`` plane of ``domain`` meets any slab."""
    denominator = 1 << level
    for axis, low, high in slabs:
        start = Fraction(float(domain[0, axis]))
        span = Fraction(float(domain[1, axis])) - start
        first = max(0, math.ceil((Fraction(low) - start) * denominator / span))
        if first <= denominator and start + span * Fraction(
            first, denominator
        ) <= Fraction(high):
            return False
    return True


def _separating_anchor(
    bounds: _FieldEnclosures,
    tangencies: np.ndarray,
    declared: np.ndarray,
    policy: AdaptiveImplicitSurfacePolicy,
    anchor: int,
    /,
) -> int | None:
    """First later lattice anchor whose first lattice separates every tangency face.

    ``tangencies`` are the ``TANGENCY`` boxes of the lattice being left and of
    every lattice left before it. A candidate lattice separates them when none
    of its exact target-level planes lies in the neighbourhood of a tangency
    face (`_tangency_slabs`), so its cells meet each touching or hugging zero
    set away from their faces. Without tangencies the next anchor is taken.
    ``None`` when no later anchor of the fixed sequence separates them.
    """
    slabs = _tangency_slabs(bounds, tangencies)
    for candidate in range(anchor + 1, len(_LATTICE_ANCHORS) + 1):
        domain, candidate_policy = _anchored_lattice(declared, policy, candidate)
        level = _target_level(domain, candidate_policy.relative_cell_diameter)
        if _lattice_separates(domain, level, slabs):
            return candidate
    return None


def _tangency_boxes(surface: AdaptiveImplicitSurface, /) -> np.ndarray:
    issues = np.asarray(surface.evidence.unresolved_issues, dtype=np.int64)
    return np.asarray(surface.evidence.unresolved_boxes, dtype=np.float64)[
        issues == int(AdaptiveImplicitBoxIssue.TANGENCY)
    ]


def _discover_on_lattice(
    geometry: CompiledGeometry,
    declared_domain: np.ndarray,
    lattice_anchor: int,
    policy: AdaptiveImplicitSurfacePolicy,
    source_id: str,
    known_tangencies: np.ndarray,
    /,
) -> tuple[AdaptiveImplicitSurface, int | None, np.ndarray]:
    """Discover on one lattice and name the next lattice anchor, if any.

    Each attempt first leaves the lattice when it has ``TANGENCY`` boxes and
    a later anchor separates them and ``known_tangencies``, those of the
    lattices already left (`_separating_anchor`); otherwise it refines
    edge-ambiguous cell contours (`_next_discovery_resolution`). The final
    attempt leaves for the next separating anchor only when its undecided
    vertices are lattice coincidences. Also returns every tangency box known
    on leaving.
    """
    bounds = _FieldEnclosures(geometry, policy.enclosure)
    selected_domain, effective_policy = _anchored_lattice(
        declared_domain, policy, lattice_anchor
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
        tangencies = _tangency_boxes(surface)
        known = np.concatenate((known_tangencies, tangencies))
        if tangencies.shape[0]:
            following = _separating_anchor(
                bounds, known, declared_domain, policy, lattice_anchor
            )
            if following is not None:
                return surface, following, known
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
    following = (
        _separating_anchor(bounds, known, declared_domain, policy, lattice_anchor)
        if _vertex_roots_are_lattice_coincidences(geometry, surface, policy.enclosure)
        else None
    )
    if effective_policy is not policy:
        surface = AdaptiveImplicitSurface(
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
    return surface, following, known


def verify_discovery_history(surface: AdaptiveImplicitSurface, /) -> bool:
    """Cold-replay prior retries and bind a validated result to its policy.

    The caller independently validates retained leaf and polygon evidence.
    Rejected attempts, on earlier lattices and at coarser resolutions, are
    replayed in bounded chunks without constructing another discovery, mesh,
    or scaffold, and must reproduce the lattice schedule of
    `_discover_on_lattice` that ends at the retained lattice and resolution.
    """
    from ._adaptive_polygon_replay import replay_history_attempt

    declared = np.asarray(surface.evidence.declared_domain, dtype=np.float64)
    policy = surface.policy
    final_anchor = surface.evidence.lattice_anchor
    if not 0 <= final_anchor <= len(_LATTICE_ANCHORS):
        return False
    bounds = _FieldEnclosures(surface.geometry, policy.enclosure)
    known = np.empty((0, 2, _DIMENSION), dtype=np.float64)
    anchor = 0
    while anchor != final_anchor:
        if anchor > final_anchor:
            return False
        previous_domain, previous_policy = _anchored_lattice(declared, policy, anchor)
        while True:
            retry, coincident, tangencies = replay_history_attempt(
                surface.geometry, previous_domain, previous_policy
            )
            attempt_known = np.concatenate((known, tangencies))
            following = (
                _separating_anchor(bounds, attempt_known, declared, policy, anchor)
                if tangencies.shape[0]
                else None
            )
            if following is not None:
                break
            if not retry:
                following = (
                    _separating_anchor(bounds, attempt_known, declared, policy, anchor)
                    if coincident
                    else None
                )
                break
            previous_policy = replace(
                previous_policy,
                relative_cell_diameter=previous_policy.relative_cell_diameter * 0.5,
            )
        if following is None:
            return False
        known = attempt_known
        anchor = following

    selected_domain, effective_policy = _anchored_lattice(declared, policy, anchor)
    if not np.array_equal(selected_domain, surface.scaffold.domain):
        return False
    target_level = surface.scaffold.target_level
    while (
        _target_level(selected_domain, effective_policy.relative_cell_diameter)
        < target_level
    ):
        retry, _, tangencies = replay_history_attempt(
            surface.geometry, selected_domain, effective_policy
        )
        if not retry or (
            tangencies.shape[0]
            and _separating_anchor(
                bounds, np.concatenate((known, tangencies)), declared, policy, anchor
            )
            is not None
        ):
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
    tangencies = _tangency_boxes(surface)
    final_known = np.concatenate((known, tangencies))
    if (
        tangencies.shape[0]
        and _separating_anchor(bounds, final_known, declared, policy, anchor) is not None
    ):
        return False
    if _next_discovery_resolution(surface, effective_policy) is not None:
        return False
    return (
        not _vertex_roots_are_lattice_coincidences(
            surface.geometry, surface, policy.enclosure
        )
        or _separating_anchor(bounds, final_known, declared, policy, anchor) is None
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

    The discovery lattice is internal and two of its defects depend only on
    where the lattice falls:

    - a lattice vertex whose point enclosure contains zero while the field
      takes both certified signs on its small neighbour stencil lies on a
      crossing of the zero set;
    - a ``TANGENCY`` box has corners of one sign and possible zeros proved to
      lie in a slab of ``2**-8`` of its width along one face: the zero set
      touches or hugs that lattice face, which every finer lattice keeps.

    A tangency is first decided locally, on the lattice it occurs on: on the
    pair of cells across the touched face, or the block of cells around one of
    that face's edges or vertices, a gradient component strict on the whole
    group, proved boundary arcs and the group's cell contours forming one disk
    prove that those contours already span the zero set there
    (`_hug_certificate`). Such a tangency is resolved without moving or
    refining the lattice. Either defect left undecided restarts discovery on
    the first later lattice of a fixed sequence of deterministic, exactly
    computed re-anchored lattices (covering the declared domain at the
    declared physical resolution) whose target-level planes are proved to
    avoid the neighbourhood of every unresolved tangency face. A tangency
    defect leaves a lattice at once; refinement never targets it. Any other
    singular cell (a touching or degenerate vertex zero, or a possible zero
    without a corner sign change or a nonvanishing gradient component) is
    refused on the current lattice without re-anchoring. When no later anchor
    separates the current tangencies, or the sequence is exhausted, the
    current lattice's ``TANGENCY`` and singular cells remain the refusal: a
    tangency that no cell group proves and that persists on every lattice
    tried is an unresolved tangency event.

    Termination: anchors strictly increase through the finite sequence, and
    on one lattice each refinement strictly deepens the target level of
    nonempty edge-ambiguous boxes wider than the root tolerance, which binary64
    representability bounds.
    """
    declared = np.array(domain, dtype=np.float64, copy=True)
    known = np.empty((0, 2, _DIMENSION), dtype=np.float64)
    anchor = 0
    while True:
        surface, following, known = _discover_on_lattice(
            geometry, declared, anchor, policy, source_id, known
        )
        if following is None:
            return surface
        anchor = following
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
