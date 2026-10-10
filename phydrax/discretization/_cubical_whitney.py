#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Prepared cubical spline-Whitney reconstruction and exact chain moments."""

from __future__ import annotations

from itertools import product
from math import comb, factorial
from typing import assert_never, final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..exterior._chains import (
    AbstractChainIntegrationKernel,
    PreparedChainQuery,
    SegmentWeight,
)
from ..exterior._form_type import FormProxy
from ..typing import (
    checked,
    Dim,
    Float64,
    HostFloat64,
    HostInt32,
    Identifier,
    parse,
    Size,
)
from ._structured_cochain import StructuredCochainBridge
from ._tensor_entities import StructuredAxis
from ._tensor_support import TensorEntityLayout
from .splatting import (
    AbstractStructuredSplatAssignment,
    MultilinearSplatAssignment,
    TensorBSplineSplatAssignment,
)


type PICShapeOrder = Literal[1, 2, 3]


class _CubicalKnotDim(Dim, minimum=2):
    """Cell-facet coordinates on one admitted axis."""


@final
class _CubicalWhitneyAxis(StrictModule):
    """One physical knot axis with an independently bound extent."""

    __strict_contract__ = True

    knots: Float64[_CubicalKnotDim]
    knot_count: Size[_CubicalKnotDim] = eqx.field(static=True)
    axis: int = eqx.field(static=True)

    def __init__(self, knots: np.ndarray, /, *, axis: int) -> None:
        if knots.dtype != np.dtype(np.float64):
            raise TypeError("Cubical Whitney knots must have float64 dtype.")
        if knots.ndim != 1 or knots.size < 2 or axis < 0:
            raise ValueError("Cubical Whitney axes require at least two knot points.")
        if np.any(~np.isfinite(knots)) or np.any(np.diff(knots) <= 0):
            raise ValueError("Cubical Whitney knots must be finite and increasing.")
        self.knots = jnp.asarray(knots)
        self.knot_count = knots.size
        self.axis = axis


class _CubicalAxisDim(Dim, minimum=1):
    """Physical coordinate axes of an admitted cubical grid."""


class _CubicalEdgeSlotDim(Dim, minimum=0):
    """Prepared electric path contributions."""


class _CubicalNodeSlotDim(Dim, minimum=0):
    """Prepared nodal path contributions."""


class _CubicalPathSegmentDim(Dim, minimum=1):
    """Host-prepared nonzero knot intervals."""


class _CubicalPolynomialDim(Dim, minimum=1):
    """Ascending local polynomial moments."""


def _cell_facets(axis: StructuredAxis, /) -> np.ndarray:
    """Cell facets of one structured axis, with the closing facet of periodic axes.

    Interval-primary facets end exactly on the declared bounds: the vertex
    coordinates accumulate width roundoff, and points on the closed domain box
    must stay inside the facet span.
    """
    points = np.asarray(axis.point_coordinates, dtype=np.float64)
    bounds = np.asarray(axis.bounds, dtype=np.float64)
    match axis.primary_entity:
        case "interval":
            return np.concatenate((points[: axis.interval_centers.size], bounds[1:]))
        case "point":
            if axis.periodic:
                return np.concatenate((points, points[:1] + (bounds[1] - bounds[0])))
            return points
        case _:
            assert_never(axis.primary_entity)


def phase_moments(theta: Array, count: int = 4, /) -> Array:
    """Analytic moments ∫₀¹ tᵐ exp(+i theta t) dt, including theta=0.

    Taylor evaluation below the order-dependent switch avoids unstable upward
    recurrence. The series termination exceeds double-precision accuracy.
    """
    z = 1j * theta.astype(jnp.complex128)
    small = jnp.abs(theta) < max(1.0, float(count))
    safe = jnp.where(small, 1.0 + 0j, z)
    exponential = jnp.exp(z)
    recurrence = [jnp.expm1(z) / safe]
    for order in range(1, count):
        recurrence.append((exponential - order * recurrence[-1]) / safe)
    orders = jnp.arange(count, dtype=jnp.float64)
    series_z = jnp.where(small, z, 0.0 + 0j)

    def accumulate(index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        term, total = carry
        return (
            term * series_z / (index + 1),
            total + term[..., None] / (index + orders + 1),
        )

    _, series = jax.lax.fori_loop(
        0,
        80,
        accumulate,
        (jnp.ones_like(z), jnp.zeros((*z.shape, count), dtype=z.dtype)),
    )
    return jnp.where(small[..., None], series, jnp.stack(tuple(recurrence), axis=-1))


def _spline_polynomial(order: int, alpha: Array, beta: Array, /) -> Array:
    """Ascending coefficients on one knot interval, selected by its midpoint."""
    midpoint = alpha + 0.5 * beta
    if order == 0:
        return ((midpoint >= -0.5) & (midpoint < 0.5)).astype(alpha.dtype)[..., None]
    coefficients = []
    for power in range(order + 1):
        value = jnp.zeros_like(alpha)
        for knot in range(order + 2):
            offset = 0.5 * (order + 1) - knot
            positive = midpoint + offset > 0.0
            value = value + jnp.where(
                positive,
                (-1) ** knot
                * comb(order + 1, knot)
                * comb(order, power)
                * (alpha + offset) ** (order - power)
                * beta**power
                / factorial(order),
                0.0,
            )
        coefficients.append(value)
    return jnp.stack(tuple(coefficients), axis=-1)


def _multiply(left: Array, right: Array, /) -> Array:
    coefficients = []
    for power in range(left.shape[-1] + right.shape[-1] - 1):
        value = jnp.zeros_like(left[..., 0])
        for index in range(left.shape[-1]):
            other = power - index
            if 0 <= other < right.shape[-1]:
                value = value + left[..., index] * right[..., other]
        coefficients.append(value)
    return jnp.stack(tuple(coefficients), axis=-1)


@final
class CubicalSplineWhitneyKernel(AbstractChainIntegrationKernel):
    """Exact cardinal spline chains, with nonuniform lowest-order geometry.

    Slots are segment-major, orientation-major, then lexicographic support.
    Capacity exhaustion is reported, never replaced by uniform subdivisions.
    Higher orders require uniform axes; order one uses actual cell widths.
    """

    __strict_contract__ = True

    bridge: StructuredCochainBridge
    shape_order: PICShapeOrder = eqx.field(static=True)
    dof_counts: tuple[int, ...] = eqx.field(static=True)
    dof_offsets: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    component_shapes: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    kernel_id: Identifier = eqx.field(static=True)
    axes: tuple[_CubicalWhitneyAxis, ...]
    periodic: tuple[bool, ...] = eqx.field(static=True)
    uniform: bool = eqx.field(static=True)

    @checked
    def __init__(
        self, bridge: StructuredCochainBridge, shape_order: PICShapeOrder = 1, /
    ) -> None:
        order = parse(shape_order, PICShapeOrder, "shape_order")
        axes = bridge.grid.structured_axes
        widths = tuple(
            np.asarray(axis.interval_widths, dtype=np.float64) for axis in axes
        )
        uniform = all(
            np.allclose(value, value[0], rtol=1e-12, atol=1e-14) for value in widths
        )
        if order > 1 and not uniform:
            raise ValueError("Higher-order cardinal spline chains require uniform axes.")
        prepared_axes = tuple(
            _CubicalWhitneyAxis(_cell_facets(axis), axis=index)
            for index, axis in enumerate(axes)
        )
        self.bridge = bridge
        self.shape_order = order
        self.dof_counts = bridge.cochain.cell_counts
        self.dof_offsets = bridge.orientation_offsets
        self.component_shapes = bridge.orientation_shapes
        self.kernel_id = canonical_fingerprint(
            {
                "kind": "cubical-spline-whitney",
                "bridge": bridge.bridge_id,
                "shape_order": order,
            }
        )
        self.axes = prepared_axes
        self.periodic = tuple(bool(axis.periodic) for axis in axes)
        self.uniform = uniform

    def entity_offsets(
        self, degree: int, /, *, proxy: FormProxy = "components"
    ) -> tuple[tuple[float, ...], ...]:
        proxy = parse(proxy, FormProxy, "proxy")
        orientations = self.bridge.orientations[degree]
        if proxy == "flux":
            if self.bridge.dimension != 3 or degree != 2:
                raise ValueError(
                    "Cartesian flux offsets require degree two in three dimensions."
                )
            orientations = ((1, 2), (0, 2), (0, 1))
        elif proxy == "circulation":
            if degree != 1:
                raise ValueError("Circulation offsets require degree one.")
        elif proxy != "components":
            raise ValueError("Unknown form proxy.")
        return tuple(
            tuple(
                0.5 if axis in orientation else 0.0
                for axis in range(self.bridge.dimension)
            )
            for orientation in orientations
        )

    def assignment(
        self, layout: TensorEntityLayout, /
    ) -> AbstractStructuredSplatAssignment:
        """The owning spline-Whitney assignment for existing prepared splats."""
        if self.shape_order == 1 and all(
            kind == "point" for kind in layout.axis_entities
        ):
            return MultilinearSplatAssignment()
        if not self.uniform:
            raise ValueError("Nonuniform mixed entities use prepared chain queries.")
        return TensorBSplineSplatAssignment(
            tuple(
                self.shape_order if kind == "point" else self.shape_order - 1
                for kind in layout.axis_entities
            )
        )

    def _points(self, points: ArrayLike, /) -> Array:
        value = jnp.asarray(points)
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("Chain coordinates must be real.")
        if value.ndim != 2 or value.shape[1] != self.bridge.dimension:
            raise ValueError("Points must have shape (particles, dimension).")
        if not jnp.issubdtype(value.dtype, jnp.floating):
            value = value.astype(jnp.float64)
        return value

    def _cell(
        self, coordinate: Array, axis: int, direction: Array, /
    ) -> tuple[Array, Array, Array]:
        edges = self.axes[axis].knots.astype(coordinate.dtype)
        length = edges[-1] - edges[0]
        epsilon = 32.0 * jnp.finfo(coordinate.dtype).eps * jnp.maximum(length, 1.0)
        turn = (
            jnp.floor((coordinate + epsilon * jnp.sign(direction) - edges[0]) / length)
            if self.periodic[axis]
            else jnp.zeros_like(coordinate)
        )
        wrapped = coordinate - turn * length
        cell = jnp.clip(
            jnp.searchsorted(edges, wrapped + epsilon * jnp.sign(direction), side="right")
            - 1,
            0,
            edges.size - 2,
        )
        lower = edges[cell] + turn * length
        width = edges[cell + 1] - edges[cell]
        return cell, lower, width

    def _split(
        self, start: Array, end: Array, capacity: int, /
    ) -> tuple[Array, Array, Array]:
        delta = end - start
        intervals = jnp.zeros((start.shape[0], capacity, 2), dtype=start.dtype)
        valid = jnp.zeros((start.shape[0], capacity), dtype=jnp.bool_)
        epsilon = 64.0 * jnp.finfo(start.dtype).eps
        shift = 0.5 * ((self.shape_order - 1) % 2)

        def step(
            slot: int, carry: tuple[Array, Array, Array]
        ) -> tuple[Array, Array, Array]:
            time, result, mask = carry
            position = start + time[:, None] * delta
            crossings = []
            for axis in range(self.bridge.dimension):
                speed = delta[:, axis]
                if self.shape_order > 1:
                    edges = self.axes[axis].knots
                    width = edges[1] - edges[0]
                    q = (position[:, axis] - edges[0]) / width - shift
                    cell = jnp.floor(q + epsilon * jnp.sign(speed))
                    boundary = (
                        edges[0]
                        + (cell + shift + jnp.where(speed > 0.0, 1.0, 0.0)) * width
                    )
                else:
                    _, lower, width = self._cell(position[:, axis], axis, speed)
                    boundary = lower + jnp.where(speed > 0.0, width, 0.0)
                crossing = (boundary - start[:, axis]) / jnp.where(
                    speed != 0.0, speed, 1.0
                )
                # A crossing within epsilon of the path end belongs to the final
                # segment; a masked sliver would still carry a nonzero moment.
                crossings.append(
                    jnp.where(
                        (speed != 0.0)
                        & (crossing > time + epsilon)
                        & (crossing < 1.0 - epsilon),
                        crossing,
                        1.0,
                    )
                )
            stop = jnp.minimum(
                jnp.min(jnp.stack(tuple(crossings), axis=-1), axis=-1), 1.0
            )
            result = result.at[:, slot, 0].set(time).at[:, slot, 1].set(stop)
            mask = mask.at[:, slot].set(stop > time + epsilon)
            return stop, result, mask

        time, intervals, valid = jax.lax.fori_loop(
            0,
            capacity,
            step,
            (jnp.zeros((start.shape[0],), dtype=start.dtype), intervals, valid),
        )
        return intervals, valid, time < 1.0 - epsilon

    def _routes(
        self, start: Array, end: Array, degree: int, /
    ) -> tuple[Array, Array, Array]:
        """Polynomial coefficients per orientation and local support on each piece."""
        middle = 0.5 * (start + end)
        displacement = end - start
        indices, polynomials, masks = [], [], []
        for orientation_index, orientation in enumerate(self.bridge.orientations[degree]):
            axis_indices, axis_polynomials = [], []
            for axis in range(self.bridge.dimension):
                cell, lower, width = self._cell(
                    middle[..., axis], axis, displacement[..., axis]
                )
                along = axis in orientation
                if self.shape_order == 1:
                    if along:
                        axis_indices.append(cell[..., None])
                        axis_polynomials.append(
                            jnp.ones((*cell.shape, 1, 1), dtype=start.dtype)
                            / width[..., None, None]
                        )
                    else:
                        alpha = (start[..., axis] - lower) / width
                        beta = displacement[..., axis] / width
                        axis_indices.append(
                            cell[..., None] + jnp.arange(2, dtype=jnp.int32)
                        )
                        axis_polynomials.append(
                            jnp.stack(
                                (
                                    jnp.stack((1.0 - alpha, -beta), axis=-1),
                                    jnp.stack((alpha, beta), axis=-1),
                                ),
                                axis=-2,
                            )
                        )
                else:
                    edges = self.axes[axis].knots
                    spacing = edges[1] - edges[0]
                    order = self.shape_order - int(along)
                    offset = 0.5 if along else 0.0
                    qmid = (middle[..., axis] - edges[0]) / spacing - offset
                    base = jnp.floor(qmid - 0.5 * (order - 1)).astype(jnp.int32)
                    nodes = base[..., None] + jnp.arange(order + 1, dtype=jnp.int32)
                    alpha = ((start[..., axis] - edges[0]) / spacing - offset)[
                        ..., None
                    ] - nodes
                    beta = jnp.broadcast_to(
                        (displacement[..., axis] / spacing)[..., None], alpha.shape
                    )
                    polynomial = _spline_polynomial(order, alpha, beta)
                    axis_indices.append(nodes)
                    axis_polynomials.append(polynomial / spacing if along else polynomial)
            shape = self.component_shapes[degree][orientation_index]
            for support in product(*(range(value.shape[-1]) for value in axis_indices)):
                flat = jnp.zeros(start.shape[:-1], dtype=jnp.int32)
                inside = jnp.ones(start.shape[:-1], dtype=jnp.bool_)
                polynomial = jnp.ones((*start.shape[:-1], 1), dtype=start.dtype)
                for axis, slot in enumerate(support):
                    index = axis_indices[axis][..., slot]
                    if self.periodic[axis]:
                        index = index % shape[axis]
                    else:
                        inside = inside & (index >= 0) & (index < shape[axis])
                        edges = self.axes[axis].knots
                        inside = (
                            inside
                            & (middle[..., axis] >= edges[0])
                            & (middle[..., axis] <= edges[-1])
                        )
                        index = jnp.clip(index, 0, shape[axis] - 1)
                    flat = flat * shape[axis] + index
                    polynomial = _multiply(
                        polynomial, axis_polynomials[axis][..., slot, :]
                    )
                indices.append(flat + self.dof_offsets[degree][orientation_index])
                polynomials.append(polynomial)
                masks.append(inside)
        return (
            jnp.stack(tuple(indices), axis=-1),
            jnp.stack(tuple(polynomials), axis=-2),
            jnp.stack(tuple(masks), axis=-1),
        )

    def evaluate(self, points: ArrayLike, degree: int, /) -> PreparedChainQuery:
        value = self._points(points)
        if not 0 <= degree <= self.bridge.dimension:
            raise ValueError("Form degree is outside the complex.")
        indices, coefficients, valid = self._routes(value, value, degree)
        count = len(self.bridge.orientations[degree])
        routes = indices.shape[1] // count
        components = jnp.repeat(jnp.eye(count, dtype=value.dtype), routes, axis=0)
        coefficients = coefficients[..., 0, None] * components[None]
        return self._query(
            indices,
            coefficients,
            valid,
            degree,
            jnp.zeros((value.shape[0],), dtype=jnp.bool_),
        )

    def integrate_points(self, points: ArrayLike, /) -> PreparedChainQuery:
        value = self._points(points)
        indices, coefficients, valid = self._routes(value, value, 0)
        return self._query(
            indices,
            coefficients[..., 0],
            valid,
            0,
            jnp.zeros((value.shape[0],), dtype=jnp.bool_),
        )

    def integrate_segments(
        self,
        start: ArrayLike,
        end: ArrayLike,
        /,
        *,
        weight: SegmentWeight = "uniform",
        phase_rate: ArrayLike | None = None,
        maximum_segments: int | None = None,
    ) -> PreparedChainQuery:
        first, last = self._points(start), self._points(end)
        if first.shape != last.shape:
            raise ValueError("Segment endpoints must have equal shape.")
        weight = parse(weight, SegmentWeight, "weight")
        capacity = (
            sum(axis.interval_centers.size for axis in self.bridge.grid.structured_axes)
            + 1
            if maximum_segments is None
            else maximum_segments
        )
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise TypeError("maximum_segments must be an integer.")
        if capacity < 1:
            raise ValueError("maximum_segments must be positive.")
        intervals, segment_valid, overflow = self._split(first, last, capacity)
        delta = last - first
        a = first[:, None] + intervals[..., 0, None] * delta[:, None]
        b = first[:, None] + intervals[..., 1, None] * delta[:, None]
        indices, polynomial, valid = self._routes(a, b, 1)
        components = len(self.bridge.orientations[1])
        routes = indices.shape[-1] // components
        velocities = jnp.repeat(delta, routes, axis=-1)
        length = intervals[..., 1] - intervals[..., 0]
        if weight == "uniform":
            if phase_rate is not None:
                raise ValueError("phase_rate is only accepted for phase weights.")
            moments = 1.0 / jnp.arange(1, polynomial.shape[-1] + 1, dtype=first.dtype)
            integral = jnp.sum(polynomial * moments, axis=-1)
        else:
            if phase_rate is None:
                raise ValueError("phase weights require phase_rate.")
            raw_rate = jnp.asarray(phase_rate)
            if jnp.issubdtype(raw_rate.dtype, jnp.complexfloating):
                raise TypeError("phase_rate must be real.")
            rate = jnp.broadcast_to(raw_rate.astype(first.dtype), (first.shape[0],))
            moments = phase_moments(rate[:, None] * length, polynomial.shape[-1])
            integral = jnp.sum(polynomial * moments[..., None, :], axis=-1) * jnp.exp(
                1j * rate[:, None, None] * intervals[..., 0, None]
            )
        coefficients = integral * length[..., None] * velocities[:, None]
        valid = valid & segment_valid[..., None]
        return self._query(
            indices.reshape((first.shape[0], -1)),
            coefficients.reshape((first.shape[0], -1)),
            valid.reshape((first.shape[0], -1)),
            1,
            overflow,
        )

    def _query(
        self,
        indices: Array,
        coefficients: Array,
        valid: Array,
        degree: int,
        overflow: Array,
        /,
    ) -> PreparedChainQuery:
        nonzero = (
            jnp.any(coefficients != 0.0, axis=tuple(range(2, coefficients.ndim)))
            if coefficients.ndim > 2
            else coefficients != 0.0
        )
        truncated = jnp.any(nonzero & ~valid, axis=1)
        finite = jnp.all(
            jnp.isfinite(coefficients), axis=tuple(range(1, coefficients.ndim))
        )
        return PreparedChainQuery(
            indices,
            coefficients,
            valid,
            finite & jnp.any(valid, axis=1) & ~truncated & ~overflow,
            overflow,
            dof_count=self.dof_counts[degree],
            degree=degree,
            kernel_id=self.kernel_id,
        )

    def prepare_path(
        self, origin: np.ndarray, direction: np.ndarray, breakpoints: np.ndarray, /
    ) -> PreparedCubicalWhitneyPath:
        """Prepare reusable exact polynomial loads on an admitted host path."""
        if self.shape_order != 1:
            raise ValueError("Host polynomial path preparation requires lowest order.")
        return PreparedCubicalWhitneyPath(
            CubicalGridGeometry(self.bridge), origin, direction, breakpoints
        )

    def phase_integral(
        self, coefficients: Array, starts: Array, lengths: Array, phase_rate: Array, /
    ) -> Array:
        """Integrate prepared local polynomials with physical phase offsets."""
        moments = phase_moments(phase_rate * lengths, coefficients.shape[-1])
        return (
            lengths
            * jnp.exp(1j * phase_rate * starts)
            * jnp.sum(coefficients * moments, axis=-1)
        )


__all__ = ["CubicalSplineWhitneyKernel"]


@final
class CubicalGridGeometry(StrictModule):
    """Host view of a structured bridge: nodes, widths, periodicity, indexing."""

    __strict_contract__ = True

    dimension: Size[_CubicalAxisDim] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    widths: tuple[np.ndarray, ...]
    boundaries: tuple[np.ndarray, ...]
    lower: HostFloat64[_CubicalAxisDim]
    upper: HostFloat64[_CubicalAxisDim]
    length: HostFloat64[_CubicalAxisDim]
    node_shape: tuple[int, ...] = eqx.field(static=True)
    edge_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    edge_offsets: tuple[int, ...] = eqx.field(static=True)

    @checked
    def __init__(self, bridge: StructuredCochainBridge, /) -> None:
        axes = bridge.grid.structured_axes
        self.dimension = bridge.dimension
        self.periodic = tuple(bool(axis.periodic) for axis in axes)
        self.widths = tuple(
            np.asarray(axis.interval_widths, dtype=np.float64) for axis in axes
        )
        # Cell boundaries including the closing node of periodic axes.
        self.boundaries = tuple(_cell_facets(axis) for axis in axes)
        self.lower = np.asarray([edges[0] for edges in self.boundaries])
        self.upper = np.asarray([edges[-1] for edges in self.boundaries])
        self.length = self.upper - self.lower
        self.node_shape = bridge.orientation_shapes[0][0]
        self.edge_shapes = bridge.orientation_shapes[1]
        self.edge_offsets = bridge.orientation_offsets[1]

    def cell(self, axis: int, coordinate: float, /) -> tuple[int, float]:
        """Cell index and lower corner containing ``coordinate`` (wrapped if periodic)."""
        value = coordinate
        if self.periodic[axis]:
            value = self.lower[axis] + (coordinate - self.lower[axis]) % self.length[axis]
        edges = self.boundaries[axis]
        index = int(
            np.clip(np.searchsorted(edges, value, side="right") - 1, 0, edges.size - 2)
        )
        return index, value - edges[index]

    def node(self, axis: int, index: int, /) -> int:
        return index % self.node_shape[axis] if self.periodic[axis] else index

    def edge_index(self, axis: int, index: tuple[int, ...], /) -> int:
        return self.edge_offsets[axis] + int(
            np.ravel_multi_index(index, self.edge_shapes[axis])
        )

    def node_index(self, index: tuple[int, ...], /) -> int:
        return int(np.ravel_multi_index(index, self.node_shape))


def _linear_factor(alpha: float, beta: float, high: bool, /) -> np.ndarray:
    """Coefficients (ascending in τ) of ``ξ`` or ``1 − ξ`` for ``ξ = α + βτ``."""
    return np.asarray([alpha, beta]) if high else np.asarray([1.0 - alpha, -beta])


def _padded(polynomial: np.ndarray, /) -> np.ndarray:
    return np.pad(polynomial, (0, 4 - polynomial.size))


@final
class PreparedCubicalWhitneyPath(StrictModule):
    """Host Whitney edge/node contributions of the straight path."""

    __strict_contract__ = True

    edge_rows: HostInt32[_CubicalEdgeSlotDim]
    edge_coefficients: HostFloat64[_CubicalEdgeSlotDim, _CubicalPolynomialDim]
    edge_segments: HostInt32[_CubicalEdgeSlotDim]
    node_rows: HostInt32[_CubicalNodeSlotDim]
    node_coefficients: HostFloat64[_CubicalNodeSlotDim, _CubicalPolynomialDim]
    node_segments: HostInt32[_CubicalNodeSlotDim]
    starts: HostFloat64[_CubicalPathSegmentDim]
    lengths: HostFloat64[_CubicalPathSegmentDim]
    first_cell: tuple[int, ...] = eqx.field(static=True)
    last_cell: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        grid: CubicalGridGeometry,
        origin: np.ndarray,
        direction: np.ndarray,
        breakpoints: np.ndarray,
        /,
    ) -> None:
        dimension = grid.dimension
        if origin.shape != (dimension,) or direction.shape != origin.shape:
            raise ValueError("Origin and direction must have the grid dimension.")
        if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction)):
            raise ValueError("Path origin and direction must be finite.")
        if breakpoints.ndim != 1 or breakpoints.size < 2:
            raise ValueError("Path breakpoints must contain at least two parameters.")
        if not np.all(np.isfinite(breakpoints)) or np.any(np.diff(breakpoints) <= 0.0):
            raise ValueError("Path breakpoints must be finite and strictly increasing.")
        edge_rows: list[int] = []
        edge_coefficients: list[np.ndarray] = []
        edge_segments: list[int] = []
        node_rows: list[int] = []
        node_coefficients: list[np.ndarray] = []
        node_segments: list[int] = []
        starts: list[float] = []
        lengths: list[float] = []
        cells: list[tuple[int, ...]] = []
        scale = max(float(np.max(np.abs(breakpoints))), 1.0)
        for start, stop in zip(breakpoints[:-1], breakpoints[1:], strict=True):
            length = float(stop - start)
            if length <= 64.0 * np.finfo(np.float64).eps * scale:
                continue
            middle = origin + 0.5 * (start + stop) * direction
            index: list[int] = []
            alpha: list[float] = []
            beta: list[float] = []
            for axis in range(dimension):
                cell, offset = grid.cell(axis, float(middle[axis]))
                width = float(grid.widths[axis][cell])
                index.append(cell)
                slope = direction[axis] * length / width
                alpha.append(offset / width - 0.5 * slope)
                beta.append(slope)
            segment = len(starts)
            starts.append(float(start))
            lengths.append(length)
            cells.append(tuple(index))
            for axis in range(dimension):
                if abs(direction[axis]) <= 1e-12:
                    continue
                others = tuple(other for other in range(dimension) if other != axis)
                weight = direction[axis] / float(grid.widths[axis][index[axis]])
                for sides in np.ndindex(*((2,) * len(others))):
                    polynomial = np.asarray([weight])
                    node = list(index)
                    for other, side in zip(others, sides, strict=True):
                        polynomial = np.polynomial.polynomial.polymul(
                            polynomial,
                            _linear_factor(alpha[other], beta[other], bool(side)),
                        )
                        node[other] = grid.node(other, index[other] + side)
                    edge_rows.append(grid.edge_index(axis, tuple(node)))
                    edge_coefficients.append(_padded(polynomial))
                    edge_segments.append(segment)
            for sides in np.ndindex(*((2,) * dimension)):
                polynomial = np.asarray([1.0])
                node = []
                for axis, side in enumerate(sides):
                    polynomial = np.polynomial.polynomial.polymul(
                        polynomial, _linear_factor(alpha[axis], beta[axis], bool(side))
                    )
                    node.append(grid.node(axis, index[axis] + side))
                node_rows.append(grid.node_index(tuple(node)))
                node_coefficients.append(_padded(polynomial))
                node_segments.append(segment)
        if not starts:
            raise ValueError("The charge path has no segment inside the domain.")
        self.edge_rows = np.asarray(edge_rows, dtype=np.int32)
        self.edge_coefficients = np.stack(edge_coefficients)
        self.edge_segments = np.asarray(edge_segments, dtype=np.int32)
        self.node_rows = np.asarray(node_rows, dtype=np.int32)
        self.node_coefficients = np.stack(node_coefficients)
        self.node_segments = np.asarray(node_segments, dtype=np.int32)
        self.starts = np.asarray(starts)
        self.lengths = np.asarray(lengths)
        self.first_cell = cells[0]
        self.last_cell = cells[-1]
