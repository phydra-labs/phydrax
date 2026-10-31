#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Revision-bound source view of trimmed parametric surface domains."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import cast, final, Literal, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._interpolation._bspline_extraction import (
    _rational_bernstein_surface_span,
    rational_bspline_surface_spans,
    RationalBSplineSurfaceHessianEnclosure,
    RationalBSplineSurfaceSpan,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..discretization._coordinate_enclosure import ExactScalar
from ..geometry._rational import fraction_endpoint, rational_vector
from ..linalg._exact import determinant_exact
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ..typing import checked, ConvertibleToArray, parse
from ._atlas import AbstractBoundaryMap, BoundaryAtlas
from ._interval_enclosure import prepare_interval_function
from ._source_identity import source_entity_id as canonical_source_entity_id
from .brep._assembly import BRepAffinePlacement, BRepAssemblyOccurrence
from .brep._model import BRepGeometry, BRepModel
from .brep._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineSurfacePatch,
    surface_differential,
    SurfaceMappedCurve,
)


if TYPE_CHECKING:
    from ._curve_boundary import ParametricCurveBoundarySource
    from .multiregion_surface import MultiRegionSurfaceTopology

type MeshingSourceAccuracy = Literal[
    "exact_represented",
    "bounded_approximation",
    "certified_enclosure",
    "sampled_interpretation",
    "repaired_envelope",
]
type _ExactUVChart = tuple[tuple[Fraction, Fraction], ...]
type _ExactChartInput = Sequence[
    Sequence[
        Sequence[Fraction | int | float | np.integer | np.floating | bool | np.bool_]
    ]
]
type _MeshingChartInput = ConvertibleToArray | _ExactChartInput


def _outward_surface_parameter_box(
    surface: AbstractSurfacePatch,
    lower: ConvertibleToArray,
    upper: ConvertibleToArray,
    /,
) -> np.ndarray:
    lower_ = np.asarray(lower, dtype=np.float64)
    upper_ = np.asarray(upper, dtype=np.float64)
    if (
        lower_.shape != (2,)
        or upper_.shape != (2,)
        or not np.all(np.isfinite(lower_))
        or not np.all(np.isfinite(upper_))
        or np.any(lower_ > upper_)
    ):
        raise ValueError("Surface parameter bounds must be finite ordered two-vectors.")
    lower_ = np.nextafter(lower_, -np.inf)
    upper_ = np.nextafter(upper_, np.inf)
    domain = surface.parameter_domain
    if domain is not None:
        domain_ = np.asarray(domain, dtype=np.float64)
        if (
            domain_.shape != (2, 2)
            or not np.all(np.isfinite(domain_))
            or np.any(domain_[0] > domain_[1])
        ):
            raise ValueError("Surface parameter domain must be finite and ordered.")
        lower_ = np.maximum(lower_, domain_[0])
        upper_ = np.minimum(upper_, domain_[1])
    if (
        not np.all(np.isfinite(lower_))
        or not np.all(np.isfinite(upper_))
        or np.any(lower_ >= upper_)
    ):
        raise ValueError("Surface parameter point has no strict in-domain enclosure box.")
    return np.stack((lower_, upper_))


type PatchBoundaryUse = PatchCurveUse | PatchPoleUse


class PatchCurveUse(StrictModule, NonTrainableState):
    """One traversal of a declared curve along a source surface trim."""

    curve: int = eqx.field(static=True)
    pcurve: AbstractCurve
    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)
    sense: int = eqx.field(static=True)

    def __init__(
        self,
        curve: int,
        pcurve: AbstractCurve,
        first: float,
        last: float,
        /,
        *,
        sense: int = 1,
    ) -> None:
        if not isinstance(pcurve, AbstractCurve) or pcurve.ambient_dimension != 2:
            raise TypeError(
                "PatchCurveUse.pcurve must be a two-dimensional source curve."
            )
        if isinstance(curve, (bool, np.bool_)) or not isinstance(
            curve, (int, np.integer)
        ):
            raise TypeError(
                "PatchCurveUse.curve must be an integer source stratum index."
            )
        curve_ = int(curve)
        first_ = float(first)
        last_ = float(last)
        if (
            curve_ < 0
            or not math.isfinite(first_)
            or not math.isfinite(last_)
            or first_ == last_
        ):
            raise ValueError(
                "PatchCurveUse requires a source curve and distinct finite parameters."
            )
        if sense not in (-1, 1):
            raise ValueError("PatchCurveUse.sense must be +1 or -1.")
        self.curve = curve_
        self.pcurve = pcurve
        self.first = first_
        self.last = last_
        self.sense = sense

    def charts(self, parameters: ConvertibleToArray, /) -> np.ndarray:
        values = np.asarray(parameters, dtype=np.float64).reshape((-1,))
        return np.asarray(
            self.pcurve.evaluate(jnp.asarray(values)), dtype=np.float64
        ).reshape((-1, 2))

    def traversal_endpoints(self) -> np.ndarray:
        parameters = (
            (self.first, self.last) if self.sense > 0 else (self.last, self.first)
        )
        return self.charts(parameters)

    def closes_native_turn(self) -> bool:
        """Whether this use spans exactly one turn of a ``2 pi``-periodic p-curve.

        Such a use returns to its start chart point exactly; its end is evaluated
        at the start parameter instead of through the rounded binary64 turn.
        """
        period = self.pcurve.period
        return (
            self.pcurve.native_period_symbol() == "two_pi"
            and period is not None
            and self.last == self.first + period
        )

    def closes_chart_loop(self) -> bool:
        """Whether this use returns exactly to its start chart point.

        One whole native turn does by construction; a nonperiodic closed
        p-curve does when its binary64 end evaluations coincide. Either way the
        end shares the start's chart point and records the start parameter.
        """
        if self.closes_native_turn():
            return True
        ends = self.charts((self.first, self.last))
        return bool(np.array_equal(ends[0], ends[1]))


class PatchPoleUse(StrictModule, NonTrainableState):
    """A source trim side collapsed to a declared physical corner."""

    corner: int = eqx.field(static=True)
    pcurve: AbstractCurve
    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)
    sense: int = eqx.field(static=True)

    def __init__(
        self,
        corner: int,
        pcurve: AbstractCurve,
        first: float,
        last: float,
        /,
        *,
        sense: int = 1,
    ) -> None:
        if not isinstance(pcurve, AbstractCurve) or pcurve.ambient_dimension != 2:
            raise TypeError("PatchPoleUse.pcurve must be a two-dimensional source curve.")
        first_ = float(first)
        last_ = float(last)
        if (
            isinstance(corner, (bool, np.bool_))
            or not isinstance(corner, (int, np.integer))
            or int(corner) < 0
            or not math.isfinite(first_)
            or not math.isfinite(last_)
            or first_ == last_
        ):
            raise ValueError(
                "PatchPoleUse requires one corner and a finite parameter interval."
            )
        if sense not in (-1, 1):
            raise ValueError("PatchPoleUse.sense must be +1 or -1.")
        self.corner = int(corner)
        self.pcurve = pcurve
        self.first = first_
        self.last = last_
        self.sense = sense

    def traversal_endpoints(self) -> np.ndarray:
        parameters = (
            (self.first, self.last) if self.sense > 0 else (self.last, self.first)
        )
        return np.asarray(
            self.pcurve.evaluate(jnp.asarray(parameters)), dtype=np.float64
        ).reshape((-1, 2))


def use_endpoints(use: PatchBoundaryUse, /) -> np.ndarray:
    return use.traversal_endpoints()


def _lift_native_period_loop(
    surface: AbstractSurfacePatch, uses: tuple[PatchBoundaryUse, ...], /
) -> tuple[PatchBoundaryUse, ...]:
    """Choose a continuous native-period gauge for an authored BRep loop."""
    from .brep._boolean_core import PeriodicPCurve

    symbols = surface.native_period_symbols()
    if not uses or not any(symbol == "two_pi" for symbol in symbols):
        return uses
    lifted: list[PatchBoundaryUse] = [uses[0]]
    period_shadow = Fraction.from_float(math.tau)
    for use in uses[1:]:
        previous = use_endpoints(lifted[-1])[1]
        start = use_endpoints(use)[0]
        shifts = [0, 0]
        for axis, symbol in enumerate(symbols):
            if symbol != "two_pi":
                continue
            turns = (
                Fraction.from_float(float(previous[axis]))
                - Fraction.from_float(float(start[axis]))
            ) / period_shadow
            # A declared exact period supplies the gauge. Do not glue unrelated
            # endpoints by nearest-integer rounding or a numerical tolerance.
            if turns.denominator == 1:
                shifts[axis] = turns.numerator
        if any(shifts):
            pcurve = PeriodicPCurve(
                use.pcurve, surface, period_shifts=(shifts[0], shifts[1])
            )
            if isinstance(use, PatchCurveUse):
                use = PatchCurveUse(
                    use.curve, pcurve, use.first, use.last, sense=use.sense
                )
            else:
                use = PatchPoleUse(
                    use.corner, pcurve, use.first, use.last, sense=use.sense
                )
        lifted.append(use)
    return tuple(lifted)


class MeshingSurfacePatch(StrictModule, NonTrainableState):
    """One source surface and its authored oriented p-curve loops."""

    surface: AbstractSurfacePatch
    loops: tuple[tuple[PatchBoundaryUse, ...], ...]
    orientation: int = eqx.field(static=True)

    def __init__(
        self,
        surface: AbstractSurfacePatch,
        loops: tuple[tuple[PatchBoundaryUse, ...], ...],
        /,
        *,
        orientation: int = 1,
    ) -> None:
        if not isinstance(surface, AbstractSurfacePatch):
            raise TypeError(
                "MeshingSurfacePatch.surface must be an AbstractSurfacePatch."
            )
        loops_ = tuple(tuple(loop) for loop in loops)
        if not loops_ or any(not loop for loop in loops_):
            raise ValueError("MeshingSurfacePatch requires nonempty trim loops.")
        if any(
            not isinstance(use, (PatchCurveUse, PatchPoleUse))
            for loop in loops_
            for use in loop
        ):
            raise TypeError("MeshingSurfacePatch loops must contain curve or pole uses.")
        if orientation not in (-1, 1):
            raise ValueError("MeshingSurfacePatch.orientation must be +1 or -1.")
        self.surface = surface
        self.loops = loops_
        self.orientation = orientation


class MeshingDomainCurve(StrictModule, NonTrainableState):
    """A declared curve stratum joining two source corners."""

    start: int = eqx.field(static=True)
    end: int = eqx.field(static=True)

    def __init__(self, start: int, end: int, /) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (start, end)
        ):
            raise ValueError(
                "MeshingDomainCurve endpoints must be nonnegative corner indices."
            )
        self.start, self.end = start, end


class MeshingDomainRegion(StrictModule, NonTrainableState):
    """A material region bounded by oriented source patches."""

    name: str = eqx.field(static=True)
    boundary: tuple[tuple[int, int], ...] = eqx.field(static=True)

    def __init__(self, name: str, boundary: tuple[tuple[int, int], ...], /) -> None:
        name_ = canonical_identifier(name, "region_name")
        rows = tuple((int(patch), int(orientation)) for patch, orientation in boundary)
        if not rows or any(
            patch < 0 or orientation not in (-1, 1) for patch, orientation in rows
        ):
            raise ValueError(
                "MeshingDomainRegion requires valid patch/orientation incidences."
            )
        if len({patch for patch, _ in rows}) != len(rows):
            raise ValueError("MeshingDomainRegion cannot repeat one patch boundary.")
        self.name, self.boundary = name_, rows


class MeshingDomainProjection(StrictModule, NonTrainableState):
    """Batched closest-point representatives with per-query convergence status."""

    points: Array
    parameters: Array
    distances: Array
    converged: Array

    def __init__(
        self,
        points: ArrayLike,
        parameters: ArrayLike,
        distances: ArrayLike,
        converged: ArrayLike,
        /,
    ) -> None:
        self.points = jnp.asarray(points, dtype=jnp.float64)
        self.parameters = jnp.asarray(parameters, dtype=jnp.float64)
        self.distances = jnp.asarray(distances, dtype=jnp.float64)
        self.converged = jnp.asarray(converged, dtype=jnp.bool_)


class DomainCurveMap(AbstractBoundaryMap):
    """Physical source curves evaluated from their owning p-curves and surfaces."""

    surfaces: tuple[AbstractSurfacePatch, ...]
    pcurves: tuple[AbstractCurve, ...]
    ranges: tuple[tuple[float, float], ...]

    def __init__(
        self,
        surfaces: tuple[AbstractSurfacePatch, ...],
        pcurves: tuple[AbstractCurve, ...],
        ranges: tuple[tuple[float, float], ...],
        /,
    ) -> None:
        if not (len(surfaces) == len(pcurves) == len(ranges)):
            raise ValueError(
                "DomainCurveMap banks must have one surface, p-curve, and range per chart."
            )
        if any(curve.ambient_dimension != 2 for curve in pcurves):
            raise ValueError("DomainCurveMap p-curves must be two-dimensional.")
        self.surfaces, self.pcurves, self.ranges = surfaces, pcurves, ranges

    @property
    def num_charts(self) -> int:
        return len(self.surfaces)

    @property
    def reference_dimension(self) -> int:
        return 1

    @property
    def ambient_dimension(self) -> int:
        return 3

    def _branch(self, index: int, /) -> Callable[[Array], Array]:
        surface, pcurve = self.surfaces[index], self.pcurves[index]
        first, last = self.ranges[index]

        def evaluate(reference: Array, /) -> Array:
            parameter = first + reference[0] * (last - first)
            chart = pcurve.evaluate(parameter)
            return surface.evaluate(chart)

        return evaluate

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        indices_ = jnp.asarray(chart_indices, dtype=jnp.int32)
        reference_ = jnp.asarray(reference, dtype=jnp.float64)
        if reference_.shape[:-1] != indices_.shape or reference_.shape[-1] != 1:
            raise ValueError("DomainCurveMap chart indices and references must align.")
        leading = indices_.shape
        indices = indices_.reshape((-1,))
        values = reference_.reshape((-1, 1))
        if not indices.size:
            return jnp.empty((*leading, self.ambient_dimension), dtype=jnp.float64)
        branches = tuple(self._branch(index) for index in range(self.num_charts))
        result = jax.vmap(lambda chart, point: jax.lax.switch(chart, branches, point))(
            indices, values
        )
        return result.reshape((*leading, self.ambient_dimension))

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        indices_ = jnp.asarray(chart_indices, dtype=jnp.int32)
        reference_ = jnp.asarray(reference, dtype=jnp.float64)
        if reference_.shape[:-1] != indices_.shape or reference_.shape[-1] != 1:
            raise ValueError("DomainCurveMap chart indices and references must align.")
        leading = indices_.shape
        indices = indices_.reshape((-1,))
        values = reference_.reshape((-1, 1))
        if not indices.size:
            return jnp.empty((*leading, self.ambient_dimension, 1), dtype=jnp.float64)
        branches = tuple(self._branch(index) for index in range(self.num_charts))
        result = jax.vmap(
            lambda chart, point: jax.jacfwd(
                lambda value: jax.lax.switch(chart, branches, value)
            )(point)
        )(indices, values)
        return result.reshape((*leading, self.ambient_dimension, 1))


@final
class _PlacedSurfacePatch(AbstractSurfacePatch):
    """JAX query realization retaining the exact source placement separately."""

    source: AbstractSurfacePatch
    placement: BRepAffinePlacement = eqx.field(static=True)
    linear: Array
    offset: Array
    orientation_sign: int = eqx.field(static=True)

    def __init__(
        self, source: AbstractSurfacePatch, placement: BRepAffinePlacement, /
    ) -> None:
        matrix = placement.matrix
        linear_exact = tuple(
            tuple(matrix[row][column] for column in range(3)) for row in range(3)
        )
        orientation = determinant_exact(linear_exact)
        numeric = np.asarray(
            [[float(value) for value in row] for row in linear_exact], dtype=np.float64
        )
        offset = np.asarray([float(matrix[row][3]) for row in range(3)], dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or not np.all(np.isfinite(offset)):
            raise ValueError(
                "Placed source patch must have a finite binary64 query realization."
            )
        numeric_exact = tuple(
            tuple(Fraction.from_float(float(value)) for value in row) for row in numeric
        )
        numeric_orientation = determinant_exact(numeric_exact)
        if numeric_orientation == 0:
            raise ValueError(
                "Placed source patch binary64 realization must be nonsingular."
            )
        if (numeric_orientation > 0) != (orientation > 0):
            raise ValueError(
                "Placed source patch binary64 realization changes exact orientation."
            )
        self.source, self.placement = source, placement
        self.linear = jnp.asarray(numeric, dtype=jnp.float64)
        self.offset = jnp.asarray(offset, dtype=jnp.float64)
        self.orientation_sign = 1 if orientation > 0 else -1

    @property
    def parameter_domain(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        return self.source.parameter_domain

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.source.periods

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        return self.source.native_period_symbols()

    def evaluate(self, parameters: Array, /) -> Array:
        points = self.source.evaluate(parameters)
        return points @ self.linear.T + self.offset

    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        """Place the source's own enclosure through the exact affine placement."""
        from ..discretization._exact_algebraic_geometry import coefficient_interval

        source_box = np.asarray(self.source.bounding_box(parameter_box), np.float64)
        if source_box.shape != (2, 3) or not np.all(np.isfinite(source_box)):
            raise ValueError("Placed source patch needs a finite source enclosure.")
        source_bounds = tuple(
            (
                Fraction.from_float(float(source_box[0, axis])),
                Fraction.from_float(float(source_box[1, axis])),
            )
            for axis in range(3)
        )
        matrix = self.placement.matrix
        lower: list[float] = []
        upper: list[float] = []
        for row in range(3):
            row_lower, row_upper = coefficient_interval(matrix[row][3])
            for column in range(3):
                entry = coefficient_interval(matrix[row][column])
                products = tuple(
                    factor * bound for factor in entry for bound in source_bounds[column]
                )
                row_lower += min(products)
                row_upper += max(products)
            lower.append(fraction_endpoint(row_lower, upper=False))
            upper.append(fraction_endpoint(row_upper, upper=True))
        return np.asarray((lower, upper), dtype=np.float64)


def _placed_homogeneous_surface_coefficients(
    coefficients: tuple[tuple[tuple[ExactScalar, ...], ...], ...],
    placement: BRepAffinePlacement,
    /,
) -> tuple[tuple[tuple[ExactScalar, ...], ...], ...]:
    matrix = placement.matrix
    if any(len(point) != 4 for row in coefficients for point in row):
        raise ValueError(
            "Placed rational surface spans require three homogeneous coordinates."
        )
    return tuple(
        tuple(
            tuple(
                sum(
                    (
                        matrix[axis][component] * point[component]
                        for component in range(3)
                    ),
                    Fraction(0),
                )
                + matrix[axis][3] * point[3]
                for axis in range(3)
            )
            + (point[3],)
            for point in row
        )
        for row in coefficients
    )


def _placed_rational_surface_span(
    span: RationalBSplineSurfaceSpan,
    placement: BRepAffinePlacement,
    /,
) -> RationalBSplineSurfaceSpan:
    return _rational_bernstein_surface_span(
        canonical_fingerprint(
            {
                "kind": "placed-rational-bernstein-surface-span",
                "source": span.source_id,
                "placement": placement.identity_payload(),
            }
        ),
        span.span_indices,
        span.degrees,
        span.coefficient_rows,
        span.parameter_bounds,
        _placed_homogeneous_surface_coefficients(
            span.homogeneous_bernstein_coefficients, placement
        ),
        include_hessian=span.has_hessian,
    )


def _exact_rational_surface_spans(
    surface: AbstractSurfacePatch, /
) -> tuple[RationalBSplineSurfaceSpan, ...] | None:
    placements: list[BRepAffinePlacement] = []
    source: AbstractSurfacePatch = surface
    while isinstance(source, _PlacedSurfacePatch):
        placements.append(source.placement)
        source = source.source
    if isinstance(source, BSplineSurfacePatch):
        spans = rational_bspline_surface_spans(
            np.asarray(source.control_points, dtype=np.float64).tolist(),
            np.asarray(source.weights, dtype=np.float64).tolist(),
            np.asarray(source.u_knots, dtype=np.float64).tolist(),
            np.asarray(source.v_knots, dtype=np.float64).tolist(),
            source.u_degree,
            source.v_degree,
            exact_control_points=source.exact_control_points,
            exact_weights=source.exact_weights,
            exact_u_knots=source.exact_u_knots,
            exact_v_knots=source.exact_v_knots,
            include_hessian=True,
        )
    else:
        from .brep._tspline import TSplineSurfacePatch

        if not isinstance(source, TSplineSurfacePatch):
            return None
        source.validate_source_bank()
        source.validate_exact_transfer_chain()
        realization = source.realization
        spans_list: list[RationalBSplineSurfaceSpan] = []
        for extraction in realization.extractions:
            homogeneous_controls = realization.exact_homogeneous_piece(
                extraction.piece_id, source.exact_target_homogeneous
            )
            if len(homogeneous_controls) != 16:
                raise ValueError(
                    "A bicubic T-spline extraction must retain sixteen homogeneous controls."
                )
            homogeneous_net = tuple(
                tuple(homogeneous_controls[u_index * 4 + v_index] for v_index in range(4))
                for u_index in range(4)
            )
            u0, u1, v0, v1 = extraction.parameter_bounds
            spans_list.append(
                _rational_bernstein_surface_span(
                    source.source_definition_id,
                    (extraction.cell_id, extraction.piece_id),
                    (3, 3),
                    extraction.coefficient_row_ids,
                    ((u0, u1), (v0, v1)),
                    homogeneous_net,
                    include_hessian=True,
                )
            )
        spans = tuple(spans_list)
    for placement in reversed(placements):
        spans = tuple(_placed_rational_surface_span(span, placement) for span in spans)
    return spans


def _clip_parameter_polygon(
    polygon: tuple[tuple[Fraction, Fraction], ...],
    axis: int,
    bound: Fraction,
    keep_greater: bool,
    /,
) -> tuple[tuple[Fraction, Fraction], ...]:
    if not polygon:
        return ()
    output: list[tuple[Fraction, Fraction]] = []
    for first, last in zip(polygon, (*polygon[1:], polygon[0]), strict=True):
        first_inside = first[axis] >= bound if keep_greater else first[axis] <= bound
        last_inside = last[axis] >= bound if keep_greater else last[axis] <= bound
        if first_inside and last_inside:
            output.append(last)
        elif first_inside != last_inside:
            direction = last[axis] - first[axis]
            ratio = (bound - first[axis]) / direction
            intersection = (
                first[0] + ratio * (last[0] - first[0]),
                first[1] + ratio * (last[1] - first[1]),
            )
            output.append(intersection)
            if last_inside:
                output.append(last)
    compact: list[tuple[Fraction, Fraction]] = []
    for point in output:
        if not compact or point != compact[-1]:
            compact.append(point)
    if len(compact) > 1 and compact[0] == compact[-1]:
        compact.pop()
    return tuple(compact)


def _exact_uv_chart_bank(
    charts: _MeshingChartInput,
    cells: np.ndarray,
    /,
) -> tuple[_ExactUVChart, ...]:
    if isinstance(charts, Sequence):
        raw_rows = cast(_ExactChartInput, charts)
    elif isinstance(charts, np.ndarray) and charts.dtype.kind == "O":
        raw_rows = cast(_ExactChartInput, charts.tolist())
    else:
        raw_rows = tuple(
            tuple(tuple(float(value) for value in point) for point in cell)
            for cell in cells
        )
    if len(raw_rows) != cells.shape[0]:
        raise ValueError("Exact UV chart rows must match their binary64 realization.")
    result: list[_ExactUVChart] = []
    for row, raw_cell in enumerate(raw_rows):
        if len(raw_cell) != cells.shape[1]:
            raise ValueError(
                "Exact UV chart vertices must match their binary64 realization."
            )
        exact_vertices: list[tuple[Fraction, Fraction]] = []
        for vertex, raw_point in enumerate(raw_cell):
            if len(raw_point) != 2:
                raise ValueError("Exact source-chart vertices must have two parameters.")
            exact_coordinates: list[Fraction] = []
            for axis, raw_value in enumerate(raw_point):
                if isinstance(raw_value, (bool, np.bool_)):
                    raise TypeError("Exact source-chart parameters cannot be boolean.")
                if isinstance(raw_value, Fraction):
                    exact_value = raw_value
                elif isinstance(raw_value, (int, np.integer)):
                    exact_value = Fraction(int(raw_value))
                elif isinstance(raw_value, (float, np.floating)):
                    exact_value = Fraction(float(raw_value))
                else:
                    raise TypeError(
                        "Exact source-chart coordinates must be rational or binary64 values."
                    )
                if float(exact_value) != float(cells[row, vertex, axis]):
                    raise ValueError(
                        "Exact source-chart coordinates must realize their binary64 chart."
                    )
                exact_coordinates.append(exact_value)
            exact_vertices.append((exact_coordinates[0], exact_coordinates[1]))
        result.append(tuple(exact_vertices))
    return tuple(result)


def _span_enclosures_for_chart(
    spans: tuple[RationalBSplineSurfaceSpan, ...],
    chart: _ExactUVChart,
    /,
) -> tuple[RationalBSplineSurfaceHessianEnclosure, ...]:
    polygon = chart
    rational_span_bounds: list[
        tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
    ] = []
    for span in spans:
        raw_u_bounds, raw_v_bounds = span.parameter_bounds
        lower_u, upper_u = raw_u_bounds
        lower_v, upper_v = raw_v_bounds
        if (
            not isinstance(lower_u, Fraction)
            or not isinstance(upper_u, Fraction)
            or not isinstance(lower_v, Fraction)
            or not isinstance(upper_v, Fraction)
        ):
            raise TypeError(
                "MeshingDomain spline parameter span bounds must be exact Fractions."
            )
        rational_span_bounds.append(((lower_u, upper_u), (lower_v, upper_v)))
    if not rational_span_bounds:
        return ()
    active_lower = (
        min(bounds[0][0] for bounds in rational_span_bounds),
        min(bounds[1][0] for bounds in rational_span_bounds),
    )
    active_upper = (
        max(bounds[0][1] for bounds in rational_span_bounds),
        max(bounds[1][1] for bounds in rational_span_bounds),
    )
    if any(
        point[axis] < active_lower[axis] or point[axis] > active_upper[axis]
        for point in polygon
        for axis in range(2)
    ):
        return ()
    enclosures: list[RationalBSplineSurfaceHessianEnclosure] = []
    for span, ((lower_u, upper_u), (lower_v, upper_v)) in zip(
        spans, rational_span_bounds, strict=True
    ):
        clipped = polygon
        for axis, bound, keep_greater in (
            (0, lower_u, True),
            (0, upper_u, False),
            (1, lower_v, True),
            (1, upper_v, False),
        ):
            clipped = _clip_parameter_polygon(clipped, axis, bound, keep_greater)
            if not clipped:
                break
        if not clipped:
            continue
        sub_bounds = (
            (
                min(point[0] for point in clipped),
                max(point[0] for point in clipped),
            ),
            (
                min(point[1] for point in clipped),
                max(point[1] for point in clipped),
            ),
        )
        enclosure = span.enclose_hessian(sub_bounds)
        if not enclosure.positive_denominator:
            raise ValueError(
                "Every spline extraction intersecting a chart must have a positive denominator enclosure."
            )
        enclosures.append(enclosure)
    return tuple(enclosures)


def _spline_interpolation_bound_from_enclosures(
    chart: _ExactUVChart,
    enclosures: tuple[RationalBSplineSurfaceHessianEnclosure, ...],
    /,
) -> float:
    """Affine interpolation bound of one chart cell from its span Hessian enclosures."""
    if not enclosures:
        return math.inf
    width_u = max(point[0] for point in chart) - min(point[0] for point in chart)
    width_v = max(point[1] for point in chart) - min(point[1] for point in chart)
    component_bounds: list[Fraction] = []
    for component in range(len(enclosures[0].duu_bounds)):
        maximum_uu = Fraction(0)
        maximum_uv = Fraction(0)
        maximum_vv = Fraction(0)
        for enclosure in enclosures:
            bounds_uu = enclosure.duu_bounds[component]
            bounds_uv = enclosure.duv_bounds[component]
            bounds_vv = enclosure.dvv_bounds[component]
            maximum_uu = max(maximum_uu, abs(bounds_uu[0]), abs(bounds_uu[1]))
            maximum_uv = max(maximum_uv, abs(bounds_uv[0]), abs(bounds_uv[1]))
            maximum_vv = max(maximum_vv, abs(bounds_vv[0]), abs(bounds_vv[1]))
        component_bounds.append(
            (
                maximum_uu * width_u * width_u
                + 2 * maximum_uv * width_u * width_v
                + maximum_vv * width_v * width_v
            )
            / 2
        )
    squared = sum((value * value for value in component_bounds), Fraction(0))
    return fraction_endpoint(
        fraction_sqrt_interval(squared)[1], upper=True, saturate=True
    )


def _exact_spline_interpolation_bounds(
    cells: np.ndarray,
    spans: tuple[RationalBSplineSurfaceSpan, ...],
    exact_charts: tuple[_ExactUVChart, ...],
    /,
) -> np.ndarray:
    result = np.full((cells.shape[0],), np.inf, dtype=np.float64)
    for row, chart in enumerate(exact_charts):
        result[row] = _spline_interpolation_bound_from_enclosures(
            chart, _span_enclosures_for_chart(spans, chart)
        )
    return result


def _exact_spline_normal_turn_bounds(
    cells: np.ndarray,
    spans: tuple[RationalBSplineSurfaceSpan, ...],
    exact_charts: tuple[_ExactUVChart, ...],
    /,
) -> np.ndarray:
    result = np.full((cells.shape[0],), np.inf, dtype=np.float64)
    pi_upper = Fraction(float(np.nextafter(math.pi, math.inf)))
    for row, chart in enumerate(exact_charts):
        enclosures = _span_enclosures_for_chart(spans, chart)
        if not enclosures:
            continue
        jacobian_intervals: list[list[list[tuple[Fraction, Fraction]]]] = [
            [[], []] for _ in range(3)
        ]
        hessian_magnitudes = [[Fraction(0), Fraction(0), Fraction(0)] for _ in range(3)]
        for enclosure in enclosures:
            if (
                len(enclosure.du_bounds) != 3
                or len(enclosure.dv_bounds) != 3
                or len(enclosure.duu_bounds) != 3
                or len(enclosure.duv_bounds) != 3
                or len(enclosure.dvv_bounds) != 3
            ):
                raise ValueError(
                    "MeshingDomain surface spline spans must map into three-space."
                )
            for physical in range(3):
                jacobian_intervals[physical][0].append(enclosure.du_bounds[physical])
                jacobian_intervals[physical][1].append(enclosure.dv_bounds[physical])
                for index, bounds in enumerate(
                    (
                        enclosure.duu_bounds[physical],
                        enclosure.duv_bounds[physical],
                        enclosure.dvv_bounds[physical],
                    )
                ):
                    hessian_magnitudes[physical][index] = max(
                        hessian_magnitudes[physical][index],
                        abs(bounds[0]),
                        abs(bounds[1]),
                    )
        center: list[list[Fraction]] = [[Fraction(0), Fraction(0)] for _ in range(3)]
        radius_squared = Fraction(0)
        for physical in range(3):
            for parameter in range(2):
                intervals = jacobian_intervals[physical][parameter]
                interval_lower = min(bounds[0] for bounds in intervals)
                interval_upper = max(bounds[1] for bounds in intervals)
                nominal = (interval_lower + interval_upper) / 2
                center[physical][parameter] = nominal
                radius = max(abs(interval_lower - nominal), abs(interval_upper - nominal))
                radius_squared += radius * radius
        exact_center = tuple(tuple(row) for row in center)
        radius = fraction_sqrt_interval(radius_squared)[1]
        g00 = sum((entry[0] * entry[0] for entry in exact_center), Fraction(0))
        g11 = sum((entry[1] * entry[1] for entry in exact_center), Fraction(0))
        g01 = sum((entry[0] * entry[1] for entry in exact_center), Fraction(0))
        gram_determinant = g00 * g11 - g01 * g01
        trace = g00 + g11
        if gram_determinant <= 0 or trace <= 0:
            continue
        center_singular_lower = fraction_sqrt_interval(gram_determinant / trace)[0]
        singular_lower = center_singular_lower - radius
        if singular_lower <= 0:
            continue
        jacobian_center_norm = fraction_sqrt_interval(
            sum(
                (value * value for line in exact_center for value in line),
                Fraction(0),
            )
        )[1]
        jacobian_norm = jacobian_center_norm + radius
        hessian_squared = sum(
            (
                values[0] * values[0] + 2 * values[1] * values[1] + values[2] * values[2]
                for values in hessian_magnitudes
            ),
            Fraction(0),
        )
        hessian_norm = fraction_sqrt_interval(hessian_squared)[1]
        exact_vertices = chart
        diameter_squared = max(
            (
                sum(
                    (
                        (first_value - second_value) ** 2
                        for first_value, second_value in zip(
                            first_point, second_point, strict=True
                        )
                    ),
                    Fraction(0),
                )
                for first_point in exact_vertices
                for second_point in exact_vertices
            ),
            default=Fraction(0),
        )
        diameter = fraction_sqrt_interval(diameter_squared)[1]
        chord = min(
            Fraction(2),
            2
            * hessian_norm
            * jacobian_norm
            * diameter
            / (singular_lower * singular_lower),
        )
        angle = min(pi_upper, pi_upper * chord / 2)
        result[row] = fraction_endpoint(angle, upper=True, saturate=True)
    return result


type _Barycentric = tuple[Fraction, Fraction, Fraction]
type _BarycentricTriangle = tuple[_Barycentric, _Barycentric, _Barycentric]
# Leaf: (barycentric triangle, Hessian term, leaf upper bound, vertex resolution).
type _InterpolationLeaf = tuple[
    _BarycentricTriangle, Fraction | None, Fraction | None, Fraction
]
_UNIT_BARYCENTRICS: tuple[_Barycentric, _Barycentric, _Barycentric] = (
    (Fraction(1), Fraction(0), Fraction(0)),
    (Fraction(0), Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(0), Fraction(1)),
)


def _barycentric_midpoint(first: _Barycentric, second: _Barycentric, /) -> _Barycentric:
    return (
        (first[0] + second[0]) / 2,
        (first[1] + second[1]) / 2,
        (first[2] + second[2]) / 2,
    )


def _fraction_magnitude(lower: float, upper: float, /) -> Fraction:
    return max(abs(Fraction(lower)), abs(Fraction(upper)))


def _adaptive_triangle_interpolation_bounds(
    surface: AbstractSurfacePatch,
    charts: Sequence[_ExactUVChart],
    budgets: Sequence[Fraction],
    /,
) -> tuple[Fraction | None, ...]:
    """Bound ``|S - I_T S|`` on exact UV triangles ``T``, refining each until decided.

    On a leaf ``L`` of a midpoint refinement of ``T``,
    ``S - I_T S = (S - I_L S) + (I_L S - I_T S)``. The first term is bounded by the
    leaf's interval Hessian, the second is affine on ``L`` and so bounded by its
    largest norm at the leaf vertices, where ``S`` is enclosed pointwise and
    ``I_T S`` is the barycentric combination of the enclosed corner values. A row
    refines its leaves above its budget until its bound meets the budget, a vertex
    lower bound exceeds it, or a leaf above it reaches binary64 enclosure resolution.
    ``None`` marks a row whose Hessian enclosure is nonfinite.
    """
    if len(charts) != len(budgets):
        raise ValueError("Every interpolation chart needs exactly one budget.")
    if any(len(chart) != 3 for chart in charts):
        raise ValueError("Adaptive interpolation bounds require triangular UV charts.")
    if not charts:
        return ()
    point_program = prepare_interval_function(surface.evaluate, 2)
    hessian_program = prepare_interval_function(
        jax.jacfwd(jax.jacfwd(surface.evaluate)), 2
    )

    def chart_point(row: int, weights: _Barycentric, /) -> tuple[Fraction, Fraction]:
        chart = charts[row]
        return (
            weights[0] * chart[0][0]
            + weights[1] * chart[1][0]
            + weights[2] * chart[2][0],
            weights[0] * chart[0][1]
            + weights[1] * chart[1][1]
            + weights[2] * chart[2][1],
        )

    def point_enclosures(
        points: Sequence[tuple[Fraction, Fraction]], /
    ) -> tuple[np.ndarray, np.ndarray]:
        boxes = np.asarray(
            [
                _outward_surface_parameter_box(
                    surface,
                    (float(point[0]), float(point[1])),
                    (float(point[0]), float(point[1])),
                )
                for point in points
            ],
            dtype=np.float64,
        )
        lower, upper = point_program.evaluate(boxes[:, 0], boxes[:, 1])
        lower_ = np.asarray(lower, dtype=np.float64).reshape(len(points), -1)
        upper_ = np.asarray(upper, dtype=np.float64).reshape(len(points), -1)
        if not np.all(np.isfinite(lower_)) or not np.all(np.isfinite(upper_)):
            raise ValueError("Source surface point enclosures must be finite.")
        return lower_, upper_

    corner_lower, corner_upper = point_enclosures(
        [point for chart in charts for point in chart]
    )
    corner_values = tuple(
        tuple(
            tuple(
                (
                    Fraction(float(corner_lower[3 * row + corner, axis])),
                    Fraction(float(corner_upper[3 * row + corner, axis])),
                )
                for axis in range(corner_lower.shape[1])
            )
            for corner in range(3)
        )
        for row in range(len(charts))
    )
    resolution_floors = tuple(
        max(
            (
                max(abs(bounds[0]), abs(bounds[1]))
                for corner in values
                for bounds in corner
            ),
            default=Fraction(0),
        )
        * Fraction(1, 2**48)
        + Fraction(1, 2**1022)
        for values in corner_values
    )
    # (upper, lower) of |S(q) - I_T S(q)| at each refinement vertex; zero at corners.
    vertex_terms: dict[tuple[int, _Barycentric], tuple[Fraction, Fraction]] = {
        (row, weights): (Fraction(0), Fraction(0))
        for row in range(len(charts))
        for weights in _UNIT_BARYCENTRICS
    }

    def evaluate_vertices(keys: Sequence[tuple[int, _Barycentric]], /) -> None:
        missing = tuple(dict.fromkeys(key for key in keys if key not in vertex_terms))
        if not missing:
            return
        lower, upper = point_enclosures(
            [chart_point(row, weights) for row, weights in missing]
        )
        for index, (row, weights) in enumerate(missing):
            upper_squared = Fraction(0)
            lower_squared = Fraction(0)
            for axis, corner_axis in enumerate(zip(*corner_values[row], strict=True)):
                interpolated_lower = sum(
                    (
                        weight * bounds[0]
                        for weight, bounds in zip(weights, corner_axis, strict=True)
                    ),
                    Fraction(0),
                )
                interpolated_upper = sum(
                    (
                        weight * bounds[1]
                        for weight, bounds in zip(weights, corner_axis, strict=True)
                    ),
                    Fraction(0),
                )
                difference_lower = (
                    Fraction(float(lower[index, axis])) - interpolated_upper
                )
                difference_upper = (
                    Fraction(float(upper[index, axis])) - interpolated_lower
                )
                magnitude = max(abs(difference_lower), abs(difference_upper))
                upper_squared += magnitude * magnitude
                if difference_lower > 0:
                    lower_squared += difference_lower * difference_lower
                elif difference_upper < 0:
                    lower_squared += difference_upper * difference_upper
            vertex_terms[(row, weights)] = (
                fraction_sqrt_interval(upper_squared)[1],
                fraction_sqrt_interval(lower_squared)[0],
            )

    def hessian_terms(
        leaves: Sequence[tuple[int, _BarycentricTriangle]], /
    ) -> list[Fraction | None]:
        boxes = []
        for row, triangle in leaves:
            points = [chart_point(row, weights) for weights in triangle]
            boxes.append(
                _outward_surface_parameter_box(
                    surface,
                    tuple(
                        float(min(point[axis] for point in points)) for axis in range(2)
                    ),
                    tuple(
                        float(max(point[axis] for point in points)) for axis in range(2)
                    ),
                )
            )
        boxes_ = np.asarray(boxes, dtype=np.float64)
        lower, upper = hessian_program.evaluate(boxes_[:, 0], boxes_[:, 1])
        lower_ = np.asarray(lower, dtype=np.float64)
        upper_ = np.asarray(upper, dtype=np.float64)
        terms: list[Fraction | None] = []
        for index in range(len(leaves)):
            low = lower_[index]
            high = upper_[index]
            if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
                terms.append(None)
                continue
            widths = tuple(
                Fraction(float(boxes_[index, 1, axis]))
                - Fraction(float(boxes_[index, 0, axis]))
                for axis in range(2)
            )
            squared = Fraction(0)
            for physical in range(low.shape[0]):
                component = (
                    sum(
                        (
                            _fraction_magnitude(
                                float(low[physical, first, second]),
                                float(high[physical, first, second]),
                            )
                            * widths[first]
                            * widths[second]
                            for first in range(2)
                            for second in range(2)
                        ),
                        Fraction(0),
                    )
                    / 2
                )
                squared += component * component
            terms.append(fraction_sqrt_interval(squared)[1])
        return terms

    def make_leaves(
        pending: Sequence[tuple[int, _BarycentricTriangle]], /
    ) -> list[tuple[int, _InterpolationLeaf]]:
        evaluate_vertices(
            [(row, weights) for row, triangle in pending for weights in triangle]
        )
        result: list[tuple[int, _InterpolationLeaf]] = []
        for (row, triangle), hessian in zip(pending, hessian_terms(pending), strict=True):
            terms = [vertex_terms[(row, weights)] for weights in triangle]
            vertex_upper = max(term[0] for term in terms)
            resolution = max(term[0] - term[1] for term in terms) + resolution_floors[row]
            result.append(
                (
                    row,
                    (
                        triangle,
                        hessian,
                        None if hessian is None else hessian + vertex_upper,
                        resolution,
                    ),
                )
            )
        return result

    leaves: list[list[_InterpolationLeaf]] = [[] for _ in charts]
    for row, leaf in make_leaves(
        [(row, _UNIT_BARYCENTRICS) for row in range(len(charts))]
    ):
        leaves[row].append(leaf)
    row_lower = [Fraction(0) for _ in charts]
    active = set(range(len(charts)))
    while active:
        pending: list[tuple[int, _BarycentricTriangle]] = []
        for row in sorted(active):
            budget = budgets[row]
            above = [leaf for leaf in leaves[row] if leaf[2] is None or leaf[2] > budget]
            if (
                not above
                or row_lower[row] > budget
                or any(leaf[1] is None or leaf[1] <= leaf[3] for leaf in above)
            ):
                active.discard(row)
                continue
            leaves[row] = [
                leaf for leaf in leaves[row] if not (leaf[2] is None or leaf[2] > budget)
            ]
            for triangle, _, _, _ in above:
                first, second, third = triangle
                first_second = _barycentric_midpoint(first, second)
                second_third = _barycentric_midpoint(second, third)
                third_first = _barycentric_midpoint(third, first)
                pending.extend(
                    (
                        (row, (first, first_second, third_first)),
                        (row, (first_second, second, second_third)),
                        (row, (third_first, second_third, third)),
                        (row, (first_second, second_third, third_first)),
                    )
                )
        if not pending:
            break
        for row, leaf in make_leaves(pending):
            leaves[row].append(leaf)
            row_lower[row] = max(
                row_lower[row],
                *(vertex_terms[(row, weights)][1] for weights in leaf[0]),
            )
    result: list[Fraction | None] = []
    for row_leaves in leaves:
        bounds = [leaf[2] for leaf in row_leaves]
        result.append(
            None
            if any(bound is None for bound in bounds)
            else max(
                (bound for bound in bounds if bound is not None), default=Fraction(0)
            )
        )
    return tuple(result)


def _source_tree_payload(value: object, /) -> dict[str, object]:
    identity = getattr(value, "identity_payload", None)
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "structure": str(jax.tree_util.tree_structure(value)),
        "arrays": array_tree_fingerprint(value),
        "source": identity() if callable(identity) else None,
    }


def _use_payload(use: PatchBoundaryUse, /) -> tuple[object, ...]:
    if isinstance(use, PatchCurveUse):
        return (
            "curve",
            use.curve,
            use.first,
            use.last,
            use.sense,
            _source_tree_payload(use.pcurve),
        )
    return (
        "pole",
        use.corner,
        use.first,
        use.last,
        use.sense,
        _source_tree_payload(use.pcurve),
    )


def _patch_payload(patch: MeshingSurfacePatch, /) -> tuple[object, ...]:
    placement = (
        patch.surface.placement.identity_payload()
        if isinstance(patch.surface, _PlacedSurfacePatch)
        else None
    )
    return (
        _source_tree_payload(patch.surface),
        patch.surface.parameter_domain,
        placement,
        patch.orientation,
        tuple(tuple(_use_payload(use) for use in loop) for loop in patch.loops),
    )


type _SourceIndexBanks = tuple[tuple[int, ...], ...]
type _SourceOccurrenceBanks = tuple[tuple[tuple[str, ...], ...], ...]


def _brep_corner_realization(
    authority: BRepModel,
    vertex: int,
    path: tuple[str, ...],
    placement: BRepAffinePlacement | None = None,
    /,
) -> np.ndarray:
    """Place exact vertices exactly and retain bounded roots' numeric realization."""
    geometry = authority.geometry
    exact = geometry.vertex_exact_points[vertex]
    if exact is None:
        geometry.source_vertex_point_bounds(vertex)
        source_point = rational_vector(
            np.asarray(geometry.vertex_points[vertex], dtype=np.float64)
        )
    else:
        source_point = exact
    if path:
        if placement is None:
            raise ValueError("Placed BRep corners require their owning occurrence.")
        source_point = placement.apply(source_point)
    elif placement is not None:
        raise ValueError("Unplaced BRep corners cannot have an occurrence placement.")
    return np.asarray(tuple(float(value) for value in source_point), dtype=np.float64)


def _placed_copy_keys(
    occurrences: tuple[BRepAssemblyOccurrence, ...], /
) -> dict[tuple[str, ...], tuple[tuple[tuple[Fraction, ...], ...], int]]:
    """Key each occurrence by its exact placement and reuse rank.

    Distinct solids at one exact placement form one placed copy and share their
    common faces, edges, and corners; a repeated use of the same solid at the
    same placement starts a new copy, so reused definitions stay separate.
    """
    uses: dict[tuple[int, tuple[tuple[Fraction, ...], ...]], int] = {}
    keys: dict[tuple[str, ...], tuple[tuple[tuple[Fraction, ...], ...], int]] = {}
    for occurrence in occurrences:
        matrix = occurrence.placement.matrix
        rank = uses.get((occurrence.solid_index, matrix), 0)
        uses[(occurrence.solid_index, matrix)] = rank + 1
        keys[occurrence.path] = (matrix, rank)
    return keys


def _same_placed_copy(
    copies: dict[tuple[str, ...], tuple[tuple[tuple[Fraction, ...], ...], int]],
    first: tuple[str, ...],
    second: tuple[str, ...],
    /,
) -> bool:
    """Whether two occurrence paths realize one exactly placed copy of the source."""
    if first == second:
        return True
    first_copy = copies.get(first)
    return first_copy is not None and first_copy == copies.get(second)


def _require_brep_source_scope(
    authority: BRepModel,
    indices: _SourceIndexBanks,
    paths: _SourceOccurrenceBanks,
    source_kinds: tuple[str, ...],
    occurrences_by_path: dict[tuple[str, ...], BRepAssemblyOccurrence],
    /,
) -> None:
    if source_kinds != ("vertex", "edge", "face"):
        raise ValueError(
            "BRep-backed domains must use vertex, edge, and face source kinds."
        )
    geometry = authority.geometry
    topology = geometry.topology
    counts = (topology.num_vertices, topology.num_edges, topology.num_faces)
    incidence: dict[int, tuple[frozenset[int], ...]] = {}
    for occurrence in occurrences_by_path.values():
        solid = occurrence.solid_index
        if solid in incidence:
            continue
        faces = frozenset(topology.solid_faces[solid])
        edges = frozenset(edge for face in faces for edge in topology.face_edges[face])
        vertices = frozenset(
            vertex for edge in edges for vertex in geometry.edge_vertices[edge]
        )
        incidence[solid] = (vertices, edges, faces)
    for dimension, (source_indices, source_paths, count) in enumerate(
        zip(indices, paths, counts, strict=True)
    ):
        for index, path in zip(source_indices, source_paths, strict=True):
            if index >= count:
                raise ValueError(
                    "Domain source key names an absent authoritative BRep entity."
                )
            if occurrences_by_path:
                occurrence = occurrences_by_path.get(path)
                if (
                    occurrence is None
                    or index not in incidence[occurrence.solid_index][dimension]
                ):
                    raise ValueError(
                        "Domain source key is not incident to its authoritative "
                        "BRep occurrence solid."
                    )
            elif path:
                raise ValueError("Unplaced BRep source key cannot name an occurrence.")


def _require_brep_region_bindings(
    authority: BRepModel,
    regions: tuple[MeshingDomainRegion, ...],
    source_indices: _SourceIndexBanks,
    source_occurrences: _SourceOccurrenceBanks,
    region_indices: tuple[int, ...],
    region_occurrences: tuple[tuple[str, ...], ...],
    occurrences_by_path: dict[tuple[str, ...], BRepAssemblyOccurrence],
    copies: dict[tuple[str, ...], tuple[tuple[tuple[Fraction, ...], ...], int]],
    /,
) -> None:
    topology = authority.topology
    expected_regions = (
        {
            (occurrence.solid_index, occurrence.path)
            for occurrence in occurrences_by_path.values()
        }
        if occurrences_by_path
        else {(solid, ()) for solid in range(topology.num_solids)}
    )
    observed_regions = set(zip(region_indices, region_occurrences, strict=True))
    if observed_regions != expected_regions:
        raise ValueError(
            "Domain BRep regions must cover every authoritative solid occurrence."
        )
    for region, solid, path in zip(
        regions, region_indices, region_occurrences, strict=True
    ):
        if solid >= topology.num_solids:
            raise ValueError("Domain region names an absent authoritative BRep solid.")
        if occurrences_by_path:
            occurrence = occurrences_by_path.get(path)
            if occurrence is None or occurrence.solid_index != solid:
                raise ValueError(
                    "Domain region changes its authoritative BRep solid occurrence."
                )
        elif path:
            raise ValueError("Unplaced BRep regions cannot name an occurrence path.")
        expected = tuple(
            zip(
                topology.solid_faces[solid],
                topology.solid_face_orientations[solid],
                strict=True,
            )
        )
        observed: list[tuple[int, int]] = []
        for patch, sign in region.boundary:
            if not _same_placed_copy(copies, source_occurrences[2][patch], path):
                raise ValueError(
                    "Domain region boundary changes its authoritative BRep occurrence."
                )
            observed.append((source_indices[2][patch], sign))
        if frozenset(observed) != frozenset(expected):
            raise ValueError(
                "Domain region boundary changes its complete oriented "
                "BRep solid boundary."
            )


def _require_brep_corner_bindings(
    authority: BRepModel,
    curves: tuple[MeshingDomainCurve, ...],
    points: np.ndarray,
    indices: _SourceIndexBanks,
    paths: _SourceOccurrenceBanks,
    placements: dict[tuple[str, ...], BRepAffinePlacement],
    copies: dict[tuple[str, ...], tuple[tuple[tuple[Fraction, ...], ...], int]],
    /,
) -> None:
    geometry = authority.geometry
    for corner, (vertex, path) in enumerate(zip(indices[0], paths[0], strict=True)):
        if vertex >= geometry.topology.num_vertices:
            raise ValueError("Domain corner names an absent authoritative BRep vertex.")
        expected = _brep_corner_realization(authority, vertex, path, placements.get(path))
        if not np.array_equal(points[corner], expected):
            raise ValueError("Domain corner changes its authoritative BRep realization.")
    for row, curve in enumerate(curves):
        edge = indices[1][row]
        path = paths[1][row]
        if edge >= len(geometry.edge_vertices) or (
            (indices[0][curve.start], indices[0][curve.end])
            != geometry.edge_vertices[edge]
            or not _same_placed_copy(copies, paths[0][curve.start], path)
            or not _same_placed_copy(copies, paths[0][curve.end], path)
        ):
            raise ValueError(
                "Domain curve changes its authoritative BRep vertex incidence."
            )


def _require_brep_use_binding(
    use: PatchBoundaryUse,
    coedge: int,
    geometry: BRepGeometry,
    ranges: np.ndarray,
    surface: AbstractSurfacePatch,
    indices: _SourceIndexBanks,
    paths: _SourceOccurrenceBanks,
    path: tuple[str, ...],
    placements: dict[tuple[str, ...], BRepAffinePlacement],
    copies: dict[tuple[str, ...], tuple[tuple[tuple[Fraction, ...], ...], int]],
    /,
) -> None:
    from ._mesh_certificates import _same_tree_content
    from .brep._boolean_core import pcurve_surface_period_shifts

    edge = geometry.coedge_edges[coedge]
    first, last = ranges[edge]
    if (
        use.first != first
        or use.last != last
        or use.sense != geometry.coedge_senses[coedge]
    ):
        raise ValueError("Domain trim use changes its authoritative BRep traversal.")
    if isinstance(use, PatchPoleUse):
        source_vertex = geometry.edge_vertices[edge][0]
        if (
            geometry.edge_curves[edge] >= 0
            or indices[0][use.corner] != source_vertex
            or not _same_placed_copy(copies, paths[0][use.corner], path)
        ):
            raise ValueError("Domain pole changes its authoritative BRep corner.")
        from .brep._correspondence import _exact_brep_pole_collapse

        if not _exact_brep_pole_collapse(
            None,
            surface,
            geometry.pcurves[coedge],
            float(first),
            float(last),
            geometry.vertex_exact_points[source_vertex],
            geometry.vertex_roots[source_vertex],
            geometry.coedge_endpoint_roots[coedge],
        ):
            raise ValueError(
                "Domain pole lacks an exact full-interval BRep collapse proof."
            )
    elif (
        geometry.edge_curves[edge] < 0
        or indices[1][use.curve] != edge
        or not _same_placed_copy(copies, paths[1][use.curve], path)
    ):
        raise ValueError("Domain trim use changes its authoritative BRep edge.")
    original = geometry.pcurves[coedge]
    if not _same_tree_content(original, use.pcurve):
        identity = (
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        )
        if (
            pcurve_surface_period_shifts(
                original, use.pcurve, identity, (Fraction(0), Fraction(0)), surface
            )
            is None
        ):
            raise ValueError("Domain p-curve is not an exact native-period source lift.")


def _same_brep_trim_endpoint(
    authority: BRepModel,
    face: int,
    first_coedge: int,
    second_coedge: int,
    first_use: PatchBoundaryUse,
    second_use: PatchBoundaryUse,
    /,
) -> bool:
    """Replay the shared source vertex's UV roots, not trigonometric shadows."""
    geometry = authority.geometry
    first_end = 1 if first_use.sense > 0 else 0
    second_start = 0 if second_use.sense > 0 else 1
    vertex = geometry.edge_vertices[geometry.coedge_edges[first_coedge]][first_end]
    next_vertex = geometry.edge_vertices[geometry.coedge_edges[second_coedge]][
        second_start
    ]
    if vertex != next_vertex:
        return False
    from .brep._intersection import closes_native_turn

    end_root = geometry.coedge_endpoint_roots[first_coedge][first_end]
    start_root = geometry.coedge_endpoint_roots[second_coedge][second_start]
    if first_coedge == second_coedge and closes_native_turn(end_root, start_root):
        return True
    root = geometry.vertex_roots[vertex]
    return root is not None and root.same_uv_endpoint(
        authority.patches[face],
        first_use.pcurve,
        end_root,
        second_use.pcurve,
        start_root,
    )


def _require_brep_domain_binding(
    authority: BRepModel,
    patches: tuple[MeshingSurfacePatch, ...],
    curves: tuple[MeshingDomainCurve, ...],
    points: np.ndarray,
    regions: tuple[MeshingDomainRegion, ...],
    indices: _SourceIndexBanks,
    paths: _SourceOccurrenceBanks,
    region_indices: tuple[int, ...],
    region_paths: tuple[tuple[str, ...], ...],
    source_kinds: tuple[str, ...],
    source_id: str,
    source_revision: str,
    authority_id: str,
    tolerance: float,
    /,
) -> None:
    authority.require_current()
    from ._mesh_certificates import _same_tree_content

    if (
        source_id != authority.source_id
        or source_revision != authority.source_revision
        or authority_id != authority.model_id
    ):
        raise ValueError("Domain source binding changes its authoritative BRep identity.")
    geometry = authority.geometry
    occurrences = tuple(geometry.assembly_graph.iter_occurrences())
    occurrences_by_path = {occurrence.path: occurrence for occurrence in occurrences}
    if len(occurrences_by_path) != len(occurrences):
        raise ValueError("BRep source occurrence paths must identify unique occurrences.")
    placements = {occurrence.path: occurrence.placement for occurrence in occurrences}
    copies = _placed_copy_keys(occurrences)
    _require_brep_source_scope(
        authority, indices, paths, source_kinds, occurrences_by_path
    )
    _require_brep_corner_bindings(
        authority, curves, points, indices, paths, placements, copies
    )
    ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    orientations = np.asarray(authority.orientation)
    for row, patch in enumerate(patches):
        face = indices[2][row]
        path = paths[2][row]
        source = authority.patches[face]
        expected: AbstractSurfacePatch = source
        orientation = int(orientations[face])
        if path in placements:
            placed = _PlacedSurfacePatch(source, placements[path])
            expected = placed
            orientation *= placed.orientation_sign
        if (
            not _same_tree_content(patch.surface, expected)
            or patch.orientation != orientation
        ):
            raise ValueError("Domain patch changes its authoritative placed BRep source.")
        loops = geometry.face_loops[face]
        if len(patch.loops) != len(loops):
            raise ValueError("Domain patch changes its authoritative BRep loops.")
        for uses, coedges in zip(patch.loops, loops, strict=True):
            if len(uses) != len(coedges):
                raise ValueError("Domain loop changes its authoritative BRep uses.")
            for slot, (use, coedge) in enumerate(zip(uses, coedges, strict=True)):
                _require_brep_use_binding(
                    use,
                    coedge,
                    geometry,
                    ranges,
                    source,
                    indices,
                    paths,
                    path,
                    placements,
                    copies,
                )
                next_slot = (slot + 1) % len(uses)
                next_use = uses[next_slot]
                if float(
                    np.linalg.norm(use_endpoints(use)[1] - use_endpoints(next_use)[0])
                ) > tolerance and not _same_brep_trim_endpoint(
                    authority, face, coedge, coedges[next_slot], use, next_use
                ):
                    raise ValueError(
                        "MeshingDomain trim loop uses do not connect in source parameter space."
                    )
    _require_brep_region_bindings(
        authority,
        regions,
        indices,
        paths,
        region_indices,
        region_paths,
        occurrences_by_path,
        copies,
    )


def _source_entity_slot_index(
    keys: tuple[str, ...], /
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Order one stratum's canonical keys for bisection back to their slots."""
    order = sorted(range(len(keys)), key=lambda slot: (keys[slot], slot))
    return tuple(keys[slot] for slot in order), tuple(order)


@final
class MeshingDomain(StrictModule, NonTrainableState):
    """Revision/frame-bound view of actual surface, curve, corner, and region sources."""

    patches: tuple[MeshingSurfacePatch, ...]
    curves: tuple[MeshingDomainCurve, ...]
    regions: tuple[MeshingDomainRegion, ...]
    corner_points: np.ndarray
    corner_points_authored: bool = eqx.field(static=True)
    curve_owners: np.ndarray
    patch_regions: tuple[tuple[int, ...], ...]
    curve_atlas: BoundaryAtlas
    brep_authority: BRepModel | None
    source_indices: tuple[tuple[int, ...], ...]
    source_occurrences: tuple[tuple[tuple[str, ...], ...], ...]
    region_source_indices: tuple[int, ...]
    region_source_occurrences: tuple[tuple[str, ...], ...]
    source_kinds: tuple[str, ...] = eqx.field(static=True)
    authority_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    accuracy: MeshingSourceAccuracy = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    scope_entity_id_banks: tuple[np.ndarray, ...]
    scope_entity_key_banks: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]
    scope_entity_set_ids: tuple[str, ...] = eqx.field(static=True)
    source_entity_slot_index: tuple[tuple[tuple[str, ...], tuple[int, ...]], ...]

    @checked
    def __init__(
        self,
        patches: tuple[MeshingSurfacePatch, ...],
        curves: tuple[MeshingDomainCurve, ...],
        corner_count: int,
        /,
        *,
        source_id: str,
        source_revision: str,
        regions: tuple[MeshingDomainRegion, ...] = (),
        tolerance: float = 0.0,
        accuracy: MeshingSourceAccuracy = "exact_represented",
        source_indices: tuple[tuple[int, ...], ...] | None = None,
        source_occurrences: tuple[tuple[tuple[str, ...], ...], ...] | None = None,
        region_source_indices: tuple[int, ...] | None = None,
        region_source_occurrences: tuple[tuple[str, ...], ...] | None = None,
        source_kinds: tuple[str, ...] = ("corner", "curve", "surface"),
        authority_id: str = "",
        brep_authority: BRepGeometry | BRepModel | None = None,
        corner_points: ArrayLike | None = None,
    ) -> None:
        """Validate one complete source graph before publishing its fields.

        The sequential admission preserves deterministic error ordering across
        geometry, incidence, identity, and region scopes without a second request
        schema. Exact BRep proofs are delegated; assignments remain atomic.
        """
        patches_ = tuple(patches)
        curves_ = tuple(curves)
        regions_ = tuple(regions)
        if not patches_ or not all(
            isinstance(value, MeshingSurfacePatch) for value in patches_
        ):
            raise TypeError(
                "MeshingDomain.patches must be a nonempty tuple of MeshingSurfacePatch values."
            )
        if not all(isinstance(value, MeshingDomainCurve) for value in curves_):
            raise TypeError(
                "MeshingDomain.curves must contain MeshingDomainCurve values."
            )
        if not all(isinstance(value, MeshingDomainRegion) for value in regions_):
            raise TypeError(
                "MeshingDomain.regions must contain MeshingDomainRegion values."
            )
        if isinstance(brep_authority, BRepGeometry):
            raise TypeError(
                "Bare BRepGeometry cannot authorize an exact meshing-domain binding."
            )
        if brep_authority is not None and not isinstance(brep_authority, BRepModel):
            raise TypeError("brep_authority must be a BRepModel.")
        if len({region.name for region in regions_}) != len(regions_):
            raise ValueError("MeshingDomain region identities must be unique.")
        if (
            isinstance(corner_count, bool)
            or not isinstance(corner_count, (int, np.integer))
            or int(corner_count) < 1
        ):
            raise ValueError("MeshingDomain.corner_count must be a positive integer.")
        corners = int(corner_count)
        if any(max(curve.start, curve.end) >= corners for curve in curves_):
            raise ValueError("MeshingDomain.curves must reference declared corners.")
        tolerance_ = float(tolerance)
        if not math.isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("MeshingDomain.tolerance must be finite and nonnegative.")
        source_id_ = canonical_identifier(source_id, "source_id")
        revision_ = canonical_identifier(source_revision, "source_revision")
        accuracy_ = parse(accuracy, MeshingSourceAccuracy, "accuracy")
        owners: list[tuple[int, int, int]] = [(-1, -1, -1) for _ in curves_]
        endpoint_images: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(corners)]
        chart_curves: list[tuple[int, int, int, PatchCurveUse]] = []
        for patch_index, patch in enumerate(patches_):
            for loop_index, loop in enumerate(patch.loops):
                for use_index, use in enumerate(loop):
                    uv_endpoints = use_endpoints(use)
                    if uv_endpoints.shape != (2, 2) or not np.all(
                        np.isfinite(uv_endpoints)
                    ):
                        raise ValueError(
                            "MeshingDomain trim endpoints must be finite UV points."
                        )
                    next_use = loop[(use_index + 1) % len(loop)]
                    next_start = use_endpoints(next_use)[0]
                    if (
                        not isinstance(brep_authority, BRepModel)
                        and float(np.linalg.norm(uv_endpoints[1] - next_start))
                        > tolerance_
                    ):
                        raise ValueError(
                            "MeshingDomain trim loop uses do not connect in source parameter space."
                        )
                    if isinstance(use, PatchPoleUse):
                        for uv in uv_endpoints:
                            point = np.asarray(
                                patch.surface.evaluate(jnp.asarray(uv)), dtype=np.float64
                            )
                            endpoint_images[use.corner].append((patch_index, point))
                        continue
                    if use.curve >= len(curves_):
                        raise ValueError(
                            "PatchCurveUse references an absent domain curve."
                        )
                    if owners[use.curve] == (-1, -1, -1):
                        owners[use.curve] = (patch_index, loop_index, use_index)
                        chart_curves.append((use.curve, patch_index, loop_index, use))
                    source_curve = curves_[use.curve]
                    physical_start = (
                        source_curve.start if use.sense > 0 else source_curve.end
                    )
                    physical_end = (
                        source_curve.end if use.sense > 0 else source_curve.start
                    )
                    for corner, uv in (
                        (physical_start, uv_endpoints[0]),
                        (physical_end, uv_endpoints[1]),
                    ):
                        point = np.asarray(
                            patch.surface.evaluate(jnp.asarray(uv)), dtype=np.float64
                        )
                        endpoint_images[corner].append((patch_index, point))
        for region in regions_:
            if any(patch >= len(patches_) for patch, _ in region.boundary):
                raise ValueError(
                    "MeshingDomainRegion references an absent surface patch."
                )
        if corner_points is None:
            points = np.full((corners, 3), np.nan, dtype=np.float64)
            for corner, images in enumerate(endpoint_images):
                if not images:
                    raise ValueError(
                        f"MeshingDomain corner {corner} has no source patch endpoint."
                    )
                points[corner] = images[0][1]
        else:
            points = np.asarray(corner_points, dtype=np.float64)
            if points.shape != (corners, 3) or not np.all(np.isfinite(points)):
                raise ValueError(
                    "MeshingDomain.corner_points must be finite with shape (corner_count, 3)."
                )
        if not np.all(np.isfinite(points)):
            raise ValueError("MeshingDomain corner source points must be finite.")
        scale_ = max(float(np.max(np.abs(points), initial=0.0)), 1.0)
        allowance = tolerance_ * scale_
        raw_indices = (
            (
                tuple(range(corners)),
                tuple(range(len(curves_))),
                tuple(range(len(patches_))),
            )
            if source_indices is None
            else tuple(tuple(row) for row in source_indices)
        )
        if len(raw_indices) != 3 or tuple(map(len, raw_indices)) != (
            corners,
            len(curves_),
            len(patches_),
        ):
            raise ValueError(
                "MeshingDomain.source_indices must align with each declared stratum."
            )
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            for row in raw_indices
            for value in row
        ):
            raise ValueError("MeshingDomain source indices must be nonnegative integers.")
        indices = tuple(tuple(int(value) for value in row) for row in raw_indices)
        raw_paths = (
            tuple(
                tuple(() for _ in range(count))
                for count in (corners, len(curves_), len(patches_))
            )
            if source_occurrences is None
            else tuple(tuple(tuple(path) for path in row) for row in source_occurrences)
        )
        if len(raw_paths) != 3 or tuple(map(len, raw_paths)) != (
            corners,
            len(curves_),
            len(patches_),
        ):
            raise ValueError(
                "MeshingDomain.source_occurrences must align with each source stratum."
            )
        if any(
            not isinstance(name, str) or not name
            for row in raw_paths
            for path in row
            for name in path
        ):
            raise ValueError(
                "MeshingDomain occurrence paths must contain nonempty names."
            )
        paths = tuple(tuple(tuple(path) for path in row) for row in raw_paths)
        if not isinstance(brep_authority, BRepModel):
            for corner, images in enumerate(endpoint_images):
                for _, point in images:
                    if np.linalg.norm(point - points[corner]) > allowance:
                        raise ValueError(
                            f"MeshingDomain source patch endpoints disagree at corner {corner}."
                        )
        for ids, occurrence_paths in zip(indices, paths, strict=True):
            if len(set(zip(ids, occurrence_paths, strict=True))) != len(ids):
                raise ValueError(
                    "MeshingDomain source indices and occurrence paths must be unique."
                )
        raw_region_indices = (
            tuple(range(len(regions_)))
            if region_source_indices is None
            else tuple(region_source_indices)
        )
        raw_region_paths = (
            tuple(() for _ in regions_)
            if region_source_occurrences is None
            else tuple(tuple(path) for path in region_source_occurrences)
        )
        if len(raw_region_indices) != len(regions_) or len(raw_region_paths) != len(
            regions_
        ):
            raise ValueError(
                "MeshingDomain region source strata must align with regions."
            )
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            for value in raw_region_indices
        ):
            raise ValueError(
                "MeshingDomain region source indices must be nonnegative integers."
            )
        if any(
            not isinstance(name, str) or not name
            for path in raw_region_paths
            for name in path
        ):
            raise ValueError(
                "MeshingDomain region occurrence paths must contain nonempty names."
            )
        region_indices = tuple(int(value) for value in raw_region_indices)
        region_paths = tuple(tuple(path) for path in raw_region_paths)
        if len(set(zip(region_indices, region_paths, strict=True))) != len(
            region_indices
        ):
            raise ValueError(
                "MeshingDomain region source indices and occurrence paths must be unique."
            )
        if isinstance(brep_authority, BRepModel):
            _require_brep_domain_binding(
                brep_authority,
                patches_,
                curves_,
                points,
                regions_,
                indices,
                paths,
                region_indices,
                region_paths,
                source_kinds,
                source_id_,
                revision_,
                authority_id,
                tolerance_,
            )
        patch_regions = tuple(
            tuple(
                index
                for index, region in enumerate(regions_)
                if any(patch == patch_index for patch, _ in region.boundary)
            )
            for patch_index in range(len(patches_))
        )
        regular_curves = tuple(chart_curves)
        curve_map = DomainCurveMap(
            tuple(patches_[patch].surface for _, patch, _, _ in regular_curves),
            tuple(use.pcurve for _, _, _, use in regular_curves),
            tuple((use.first, use.last) for _, _, _, use in regular_curves),
        )
        atlas = BoundaryAtlas(
            curve_map,
            source_entity_ids=jnp.arange(len(regular_curves), dtype=jnp.int32),
            source_id=source_id_,
            physical_tags=tuple(
                f"curve:{source_indices[1][curve]}" for curve, _, _, _ in regular_curves
            )
            if source_indices is not None
            else tuple(f"curve:{curve}" for curve, _, _, _ in regular_curves),
        )
        self.patches = patches_
        self.curves = curves_
        self.regions = regions_
        self.corner_points = np.array(points, dtype=np.float64, copy=True)
        self.corner_points_authored = corner_points is not None
        self.corner_points.setflags(write=False)
        self.curve_owners = np.asarray(owners, dtype=np.int64).reshape((-1, 3))
        self.curve_owners.setflags(write=False)
        self.patch_regions = patch_regions
        self.curve_atlas = atlas
        self.brep_authority = brep_authority
        self.source_indices = tuple(tuple(int(value) for value in row) for row in indices)
        self.source_occurrences = tuple(
            tuple(tuple(path) for path in row) for row in paths
        )
        self.region_source_indices = tuple(int(value) for value in region_indices)
        self.region_source_occurrences = tuple(tuple(path) for path in region_paths)
        self.source_kinds = source_kinds
        self.authority_id = str(authority_id)
        self.source_id = source_id_
        self.source_revision = revision_
        self.accuracy = accuracy_
        self.tolerance = tolerance_
        self.scale = scale_
        stratum_keys = tuple(self.stratum_entity_ids(dimension) for dimension in range(4))
        (
            self.scope_entity_id_banks,
            self.scope_entity_key_banks,
            self.scope_entity_set_ids,
        ) = self._prepare_scope_entity_banks(stratum_keys)
        self.source_entity_slot_index = tuple(
            _source_entity_slot_index(keys) for keys in stratum_keys
        )
        self.domain_id = self.content_domain_id()

    @classmethod
    @checked
    def from_brep(cls, model: BRepModel, /) -> MeshingDomain:
        if not isinstance(model, BRepModel):
            raise TypeError("MeshingDomain.from_brep requires a BRepModel.")
        geometry = model.geometry
        occurrences = tuple(geometry.assembly_graph.iter_occurrences())
        if occurrences:
            return _placed_brep_domain(cls, model, occurrences)
        return cls.from_brep_geometry(
            geometry,
            model.patches,
            np.asarray(model.orientation),
            source_id=model.source_id,
            source_revision=model.source_revision,
            _authority=model,
            corner_points=None,
        )

    @classmethod
    @checked
    def from_brep_geometry(
        cls,
        geometry: BRepGeometry,
        surfaces: tuple[AbstractSurfacePatch, ...],
        orientation: np.ndarray,
        /,
        *,
        source_id: str,
        source_revision: str,
        _authority: BRepModel | None = None,
        corner_points: ArrayLike | None = None,
    ) -> MeshingDomain:
        """Build a generic domain from supplied geometry, or bind its owning model."""
        if len(surfaces) != len(geometry.face_loops):
            raise ValueError("BRep surfaces must align with source face loops.")
        orientations = np.asarray(orientation, dtype=np.float64).reshape((-1,))
        if orientations.shape != (len(surfaces),) or not np.all(
            np.isin(orientations, (-1.0, 1.0))
        ):
            raise ValueError("BRep face orientations must align with source surfaces.")
        if _authority is not None and not isinstance(_authority, BRepModel):
            raise TypeError("_authority must be the owning BRepModel.")
        curves = tuple(
            MeshingDomainCurve(*endpoints) for endpoints in geometry.edge_vertices
        )
        patches: list[MeshingSurfacePatch] = []
        for face, loops in enumerate(geometry.face_loops):
            patch_loops: list[tuple[PatchBoundaryUse, ...]] = []
            for loop in loops:
                uses: list[PatchBoundaryUse] = []
                for coedge in loop:
                    edge = geometry.coedge_edges[coedge]
                    first, last = (
                        float(value) for value in np.asarray(geometry.edge_ranges[edge])
                    )
                    sense = geometry.coedge_senses[coedge]
                    if geometry.edge_curves[edge] < 0:
                        vertex = geometry.edge_vertices[edge][0]
                        if _authority is None:
                            from .brep._correspondence import (
                                _exact_brep_pole_collapse,
                            )

                            if not _exact_brep_pole_collapse(
                                None,
                                surfaces[face],
                                geometry.pcurves[coedge],
                                first,
                                last,
                                geometry.vertex_exact_points[vertex],
                                geometry.vertex_roots[vertex],
                                geometry.coedge_endpoint_roots[coedge],
                            ):
                                raise ValueError(
                                    "A bare BRepGeometry pole requires an exact "
                                    "full-interval source proof."
                                )
                        uses.append(
                            PatchPoleUse(
                                vertex, geometry.pcurves[coedge], first, last, sense=sense
                            )
                        )
                    else:
                        uses.append(
                            PatchCurveUse(
                                edge, geometry.pcurves[coedge], first, last, sense=sense
                            )
                        )
                patch_loops.append(_lift_native_period_loop(surfaces[face], tuple(uses)))
            patches.append(
                MeshingSurfacePatch(
                    surfaces[face],
                    tuple(patch_loops),
                    orientation=int(orientations[face]),
                )
            )
        regions = tuple(
            MeshingDomainRegion(
                f"solid_{solid}",
                tuple(
                    (face, orientation)
                    for face, orientation in zip(faces, signs, strict=True)
                ),
            )
            for solid, (faces, signs) in enumerate(
                zip(
                    geometry.topology.solid_faces,
                    geometry.topology.solid_face_orientations,
                    strict=True,
                )
            )
        )
        authority = _authority
        if corner_points is None and authority is not None:
            corner_points = np.asarray(
                tuple(
                    _brep_corner_realization(authority, vertex, ())
                    for vertex in range(geometry.topology.num_vertices)
                ),
                dtype=np.float64,
            )
        source_kinds = (
            ("vertex", "edge", "face")
            if authority is not None
            else ("corner", "curve", "surface")
        )
        authority_id_ = "" if authority is None else authority.model_id
        return cls(
            tuple(patches),
            curves,
            geometry.vertex_points.shape[0],
            source_id=source_id,
            source_revision=source_revision,
            regions=regions,
            source_indices=(
                tuple(range(geometry.vertex_points.shape[0])),
                tuple(range(len(geometry.edge_curves))),
                tuple(range(len(surfaces))),
            ),
            region_source_indices=tuple(range(len(regions))),
            source_kinds=source_kinds,
            authority_id=authority_id_,
            brep_authority=authority,
            corner_points=corner_points,
        )

    def content_domain_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "meshing-domain",
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "accuracy": self.accuracy,
                "tolerance": self.tolerance,
                "source_indices": self.source_indices,
                "source_occurrences": self.source_occurrences,
                "region_source_indices": self.region_source_indices,
                "region_source_occurrences": self.region_source_occurrences,
                "source_kinds": self.source_kinds,
                "patches": tuple(_patch_payload(patch) for patch in self.patches),
                "curves": tuple((curve.start, curve.end) for curve in self.curves),
                "regions": tuple(
                    (region.name, region.boundary) for region in self.regions
                ),
                "corners": array_tree_fingerprint(self.corner_points),
                "corner_points_authored": self.corner_points_authored,
                "scope_entity_id_banks": array_tree_fingerprint(
                    self.scope_entity_id_banks
                ),
                "scope_entity_key_banks": self.scope_entity_key_banks,
                "scope_entity_set_ids": self.scope_entity_set_ids,
            }
        )

    def require_current(
        self, source_id: str | None = None, source_revision: str | None = None, /
    ) -> None:
        if source_id is not None and source_id != self.source_id:
            raise ValueError("MeshingDomain source_id changed.")
        if source_revision is not None and source_revision != self.source_revision:
            raise ValueError("MeshingDomain source_revision changed.")
        authority = self.brep_authority
        if authority is not None:
            authority.require_current()
        if self.content_domain_id() != self.domain_id:
            raise ValueError("MeshingDomain content changed after construction.")

    @property
    def corner_count(self) -> int:
        return self.corner_points.shape[0]

    @property
    def region_ids(self) -> tuple[str, ...]:
        return tuple(region.name for region in self.regions)

    @property
    def physical_extent(self) -> float:
        return float(np.max(np.abs(self.corner_points), initial=0.0))

    def _source_entity_table(
        self, dimension: int, /
    ) -> tuple[str, tuple[int, ...], tuple[tuple[str, ...], ...]]:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError("MeshingDomain entity dimension must be an integer.")
        if dimension not in (0, 1, 2, 3):
            raise ValueError("MeshingDomain entity dimension must be in 0..3.")
        if dimension == 3:
            kind = (
                "solid" if self.source_kinds == ("vertex", "edge", "face") else "region"
            )
            return kind, self.region_source_indices, self.region_source_occurrences
        return (
            self.source_kinds[dimension],
            self.source_indices[dimension],
            self.source_occurrences[dimension],
        )

    def entity_id(self, dimension: int, index: int, /) -> str:
        kind, source_indices, source_paths = self._source_entity_table(dimension)
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("MeshingDomain entity index must be an integer slot.")
        if index < 0 or index >= len(source_indices):
            raise ValueError("MeshingDomain entity index is outside its source stratum.")
        return canonical_source_entity_id(
            self.source_id,
            self.source_revision,
            kind,
            source_indices[index],
            occurrence_path=source_paths[index],
        )

    def stratum_entity_ids(self, dimension: int, /) -> tuple[str, ...]:
        """Return every canonical source key of one stratum in slot order.

        Loops over many slots should call this once: each `entity_id` call reads
        the stratum's source-entity table again.
        """
        kind, source_indices, source_paths = self._source_entity_table(dimension)
        source_id, source_revision = self.source_id, self.source_revision
        return tuple(
            canonical_source_entity_id(
                source_id, source_revision, kind, index, occurrence_path=path
            )
            for index, path in zip(source_indices, source_paths, strict=True)
        )

    def resolve_entity_id(
        self,
        dimension: int,
        source_entity_id: str,
        /,
        *,
        occurrence_path: tuple[str, ...] | None = None,
    ) -> int:
        """Resolve a canonical source key through the actual source stratum."""
        _, _, source_paths = self._source_entity_table(dimension)
        if not isinstance(source_entity_id, str) or not source_entity_id:
            raise TypeError("source_entity_id must be a canonical identity string.")
        if occurrence_path is not None:
            if isinstance(occurrence_path, (str, bytes)):
                raise TypeError("occurrence_path must contain occurrence identifiers.")
            occurrence_path = tuple(
                canonical_identifier(value, "occurrence_id") for value in occurrence_path
            )
        keys, slots = self.source_entity_slot_index[dimension]
        first = bisect_left(keys, source_entity_id)
        last = bisect_right(keys, source_entity_id, lo=first)
        matches = tuple(
            slots[position]
            for position in range(first, last)
            if occurrence_path is None or source_paths[slots[position]] == occurrence_path
        )
        if len(matches) != 1:
            raise ValueError(
                "MeshingDomain source entity must resolve uniquely through its source table; "
                f"found {len(matches)} matching strata."
            )
        return matches[0]

    def entity_ids(
        self, dimensions: np.ndarray, indices: np.ndarray, /
    ) -> tuple[str, ...]:
        dimensions_ = np.asarray(dimensions)
        indices_ = np.asarray(indices)
        if dimensions_.ndim != 1 or indices_.shape != dimensions_.shape:
            raise ValueError("entity_ids requires aligned one-dimensional arrays.")
        if not np.issubdtype(dimensions_.dtype, np.integer) or not np.issubdtype(
            indices_.dtype, np.integer
        ):
            raise TypeError("entity_ids inputs must be integer arrays.")
        keys: dict[int, tuple[str, ...]] = {}
        result = []
        for dimension, index in zip(dimensions_.tolist(), indices_.tolist(), strict=True):
            stratum = keys.get(dimension)
            if stratum is None:
                stratum = keys[dimension] = self.stratum_entity_ids(dimension)
            if index < 0 or index >= len(stratum):
                raise ValueError(
                    "MeshingDomain entity index is outside its source stratum."
                )
            result.append(stratum[index])
        return tuple(result)

    def entity_set_id(self, dimension: int, /) -> str:
        kind, source_indices, source_paths = self._source_entity_table(dimension)
        return canonical_fingerprint(
            {
                "kind": "meshing-domain-entity-set",
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "entity_kind": kind,
                "entities": tuple(
                    (source_id, path)
                    for source_id, path in zip(source_indices, source_paths, strict=True)
                ),
            }
        )

    def scope_indices(self, dimension: int, /) -> tuple[int, ...]:
        _, source_indices, source_paths = self._source_entity_table(dimension)
        return tuple(range(len(source_indices))) if any(source_paths) else source_indices

    def _prepare_scope_entity_banks(
        self, stratum_keys: tuple[tuple[str, ...], ...], /
    ) -> tuple[
        tuple[np.ndarray, ...],
        tuple[tuple[tuple[str, tuple[str, ...]], ...], ...],
        tuple[str, ...],
    ]:
        id_banks: list[np.ndarray] = []
        key_banks: list[tuple[tuple[str, tuple[str, ...]], ...]] = []
        set_ids: list[str] = []
        maximum = (1 << 63) - 1
        source_id, source_revision = self.source_id, self.source_revision
        for dimension in range(4):
            _, source_ids, occurrence_paths = self._source_entity_table(dimension)
            entity_keys = stratum_keys[dimension]
            records = []
            for source_index, path, entity_key in zip(
                source_ids, occurrence_paths, entity_keys, strict=True
            ):
                numeric_id = (
                    int(
                        canonical_fingerprint(
                            {
                                "kind": "meshing-domain-scope-entity",
                                "source_id": source_id,
                                "source_revision": source_revision,
                                "dimension": dimension,
                                "source_index": source_index,
                                "source_entity_id": entity_key,
                                "occurrence_path": path,
                            }
                        )[:16],
                        16,
                    )
                    & maximum
                )
                records.append((numeric_id, entity_key, path))
            if len({record[0] for record in records}) != len(records):
                raise ValueError(
                    f"MeshingDomain dimension-{dimension} scope IDs collide in int64."
                )
            records.sort(key=lambda record: record[0])
            identifiers = np.asarray([record[0] for record in records], dtype=np.int64)
            identifiers.setflags(write=False)
            keys = tuple((entity_key, path) for _, entity_key, path in records)
            id_banks.append(identifiers)
            key_banks.append(keys)
            set_ids.append(
                canonical_fingerprint(
                    {
                        "kind": "meshing-domain-scope-set",
                        "source_id": self.source_id,
                        "source_revision": self.source_revision,
                        "dimension": dimension,
                        "entities": keys,
                    }
                )
            )
        return tuple(id_banks), tuple(key_banks), tuple(set_ids)

    @staticmethod
    def _scope_dimension(dimension: int, /) -> int:
        if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
            raise TypeError("dimension must be an integer source stratum.")
        index = int(dimension)
        if index not in (0, 1, 2, 3):
            raise ValueError("MeshingDomain scope dimensions must lie in 0..3.")
        return index

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        """Return stable numeric source IDs, separate from source row positions."""
        return self.scope_entity_id_banks[self._scope_dimension(dimension)]

    def scope_entity_set_id(self, dimension: int, /) -> str:
        """Return the revision-bound identity of a source stratum bank."""
        return self.scope_entity_set_ids[self._scope_dimension(dimension)]

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        """Resolve a stable scope ID to its canonical source key and occurrence."""
        index = self._scope_dimension(dimension)
        if isinstance(entity_id, bool) or not isinstance(entity_id, (int, np.integer)):
            raise TypeError("entity_id must be an integer source scope ID.")
        identifiers = self.scope_entity_id_banks[index]
        position = int(np.searchsorted(identifiers, int(entity_id)))
        if position >= identifiers.size or int(identifiers[position]) != int(entity_id):
            raise ValueError("entity_id is absent from this MeshingDomain stratum.")
        return self.scope_entity_key_banks[index][position]

    def source_corner_point_bounds(
        self,
        row: int,
        /,
    ) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
        """Return exact or outward-bounded coordinates of one authored source corner."""
        if isinstance(row, bool) or not isinstance(row, (int, np.integer)):
            raise TypeError("row must be an integer MeshingDomain corner row.")
        index = int(row)
        if index < 0 or index >= len(self.corner_points):
            raise ValueError("row is absent from the MeshingDomain corner table.")
        self.require_current(self.source_id, self.source_revision)
        authority = self.brep_authority
        if authority is not None:
            source_vertex = int(self.source_indices[0][index])
            occurrence_path = self.source_occurrences[0][index]
            return authority.source_vertex_point_bounds(
                source_vertex, occurrence_path=occurrence_path
            )

        endpoint_records: list[tuple[np.ndarray, np.ndarray]] = []
        for patch in self.patches:
            for loop in patch.loops:
                for use in loop:
                    if isinstance(use, PatchPoleUse):
                        endpoint_corners = (use.corner, use.corner)
                    else:
                        curve = self.curves[use.curve]
                        endpoint_corners = (
                            (curve.start, curve.end)
                            if use.sense > 0
                            else (curve.end, curve.start)
                        )
                    for endpoint, corner in enumerate(endpoint_corners):
                        if int(corner) != index:
                            continue
                        parameters = (use.first, use.last)
                        parameter = (
                            parameters[endpoint]
                            if use.sense > 0
                            else parameters[1 - endpoint]
                        )
                        try:
                            pcurve_bounds = np.asarray(
                                use.pcurve.bounding_box(parameter, parameter),
                                dtype=np.float64,
                            )
                            if (
                                pcurve_bounds.shape != (2, 2)
                                or not np.all(np.isfinite(pcurve_bounds))
                                or np.any(pcurve_bounds[0] > pcurve_bounds[1])
                            ):
                                raise ValueError(
                                    "Source p-curve endpoint has no finite ordered UV box."
                                )
                            uv_bounds = _outward_surface_parameter_box(
                                patch.surface,
                                pcurve_bounds[0],
                                pcurve_bounds[1],
                            )
                            point_bounds = np.asarray(
                                patch.surface.bounding_box(uv_bounds),
                                dtype=np.float64,
                            )
                            if (
                                point_bounds.shape != (2, 3)
                                or not np.all(np.isfinite(point_bounds))
                                or np.any(point_bounds[0] > point_bounds[1])
                            ):
                                raise ValueError(
                                    "Source trim endpoint has no finite ordered physical point box."
                                )
                        except (
                            ArithmeticError,
                            FloatingPointError,
                            NotImplementedError,
                            TypeError,
                            ValueError,
                        ) as error:
                            raise ValueError(
                                "Source trim endpoint has no finite continuous point enclosure."
                            ) from error
                        low = point_bounds[0]
                        high = point_bounds[1]
                        endpoint_records.append((low, high))
        if not endpoint_records:
            raise ValueError(
                "MeshingDomain corner has no source point enclosure authority."
            )
        lower_float = np.max(np.stack([record[0] for record in endpoint_records]), axis=0)
        upper_float = np.min(np.stack([record[1] for record in endpoint_records]), axis=0)
        if np.any(lower_float > upper_float):
            raise ValueError("Source trim endpoint enclosures disagree at this corner.")
        if self.corner_points_authored:
            authored = tuple(
                Fraction.from_float(float(value)) for value in self.corner_points[index]
            )
            if any(
                value < Fraction.from_float(float(low))
                or value > Fraction.from_float(float(high))
                for value, low, high in zip(
                    authored, lower_float, upper_float, strict=True
                )
            ):
                raise ValueError(
                    "Authored source corner lies outside its continuous trim enclosure."
                )
            return authored, authored
        return (
            tuple(Fraction.from_float(float(value)) for value in lower_float),
            tuple(Fraction.from_float(float(value)) for value in upper_float),
        )

    def curve_uses(self, curve: int, /) -> tuple[tuple[int, PatchCurveUse], ...]:
        return tuple(
            (patch_index, use)
            for patch_index, patch in enumerate(self.patches)
            for loop in patch.loops
            for use in loop
            if isinstance(use, PatchCurveUse) and use.curve == curve
        )

    def source_curve_chart_uses(
        self, entity_id: int, /
    ) -> tuple[tuple[str, int, int, int, PatchCurveUse], ...]:
        """Return every authored trim chart for one source edge identity."""
        if isinstance(entity_id, bool) or not isinstance(entity_id, (int, np.integer)):
            raise TypeError("entity_id must be an integer source curve scope ID.")
        source_key = self.source_entity_key_for_scope(1, int(entity_id))
        curve_row = self.resolve_entity_id(
            1, source_key[0], occurrence_path=source_key[1]
        )
        result = []
        patch_keys = self.stratum_entity_ids(2)
        for patch_row, patch in enumerate(self.patches):
            patch_key = (
                patch_keys[patch_row],
                self.source_occurrences[2][patch_row],
            )
            for loop_row, loop in enumerate(patch.loops):
                for use_row, use in enumerate(loop):
                    if not isinstance(use, PatchCurveUse) or use.curve != curve_row:
                        continue
                    chart_id = canonical_fingerprint(
                        {
                            "kind": "meshing-domain-source-curve-use",
                            "source_id": self.source_id,
                            "source_revision": self.source_revision,
                            "curve": source_key,
                            "patch": patch_key,
                            "loop": loop_row,
                            "use": use_row,
                            "range": (use.first, use.last),
                            "sense": use.sense,
                        }
                    )
                    result.append((chart_id, patch_row, loop_row, use_row, use))
        if not result:
            raise ValueError(
                "MeshingDomain source curve has no authored surface trim chart."
            )
        return tuple(result)

    def validate_curve_boundary_query(
        self,
        query: ParametricCurveBoundarySource,
        requested_entity_ids: ArrayLike,
        /,
    ) -> None:
        from ._curve_boundary import ParametricCurveBoundarySource

        if not isinstance(query, ParametricCurveBoundarySource):
            raise TypeError(
                "MeshingDomain curve validation requires ParametricCurveBoundarySource."
            )
        if (query.source_id, query.source_revision) != (
            self.source_id,
            self.source_revision,
        ):
            raise ValueError("MeshingDomain curve query must bind this source revision.")
        query.require_current()
        raw_ids = np.asarray(requested_entity_ids)
        if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "iu":
            raise TypeError("requested_entity_ids must be an integer vector.")
        if (
            raw_ids.dtype.kind == "u"
            and raw_ids.size
            and np.any(raw_ids > np.iinfo(np.int64).max)
        ):
            raise ValueError("requested_entity_ids must fit int64.")
        identifiers = np.asarray(raw_ids, dtype=np.int64)
        if identifiers.size == 0 or np.unique(identifiers).size != identifiers.size:
            raise ValueError(
                "requested_entity_ids must be a nonempty set of unique curve IDs."
            )
        expected: dict[
            str,
            tuple[
                str, tuple[str, ...], tuple[float, float], int, int, int, PatchCurveUse
            ],
        ] = {}
        for identifier in identifiers.tolist():
            entity_id, occurrence = self.source_entity_key_for_scope(1, int(identifier))
            for (
                chart_id,
                patch_row,
                loop_row,
                use_row,
                use,
            ) in self.source_curve_chart_uses(int(identifier)):
                if chart_id in expected:
                    raise ValueError(
                        "MeshingDomain source curve chart identities are not unique."
                    )
                expected[chart_id] = (
                    entity_id,
                    occurrence,
                    (float(use.first), float(use.last)),
                    patch_row,
                    loop_row,
                    use_row,
                    use,
                )
        if not expected:
            raise ValueError(
                "Requested MeshingDomain curve scope has no authored trim charts."
            )
        actual: dict[str, tuple[str, tuple[str, ...], tuple[float, float]]] = {}
        for entity_id, occurrence, chart_id, parameter_range in zip(
            query.curve_entity_ids,
            query.occurrence_paths,
            query.chart_ids,
            query.parameter_ranges,
            strict=True,
        ):
            if chart_id in actual:
                raise ValueError("MeshingDomain curve query repeats a chart identity.")
            parameter_values = tuple(float(value) for value in parameter_range)
            if len(parameter_values) != 2:
                raise ValueError(
                    "MeshingDomain curve parameter ranges must contain two endpoints."
                )
            actual[chart_id] = (
                entity_id,
                tuple(occurrence),
                (parameter_values[0], parameter_values[1]),
            )
        expected_records = {
            chart_id: (entity_id, occurrence, parameter_range)
            for chart_id, (
                entity_id,
                occurrence,
                parameter_range,
                *_rest,
            ) in expected.items()
        }
        if actual != expected_records:
            raise ValueError(
                "MeshingDomain curve query must cover every actual trim chart and full range."
            )

        expected_point_bounds: dict[
            tuple[str, tuple[str, ...]], tuple[np.ndarray, np.ndarray]
        ] = {}
        corner_keys = self.stratum_entity_ids(0)
        for query_row, chart_id in enumerate(query.chart_ids):
            _, _, (first, last), patch_row, _, _, use = expected[chart_id]
            curve = query.curves[query_row]
            patch = self.patches[patch_row]
            if (
                not isinstance(curve, SurfaceMappedCurve)
                or curve.surface is not patch.surface
                or curve.pcurve is not use.pcurve
            ):
                raise ValueError(
                    "MeshingDomain curve query changed its surface/pcurve source map."
                )
            source_curve = self.curves[use.curve]
            # Coedge sense reverses loop traversal, not the chart's parameter order.
            corners = (source_curve.start, source_curve.end)
            expected_endpoint_keys = tuple(
                (corner_keys[corner], self.source_occurrences[0][corner])
                for corner in corners
            )
            if query.endpoint_source_keys[query_row] != expected_endpoint_keys:
                raise ValueError(
                    "MeshingDomain curve query changed its trim endpoint source identities."
                )
            if query.parameter_ranges[query_row] != (first, last):
                raise ValueError(
                    "MeshingDomain curve query changed its trim parameter range."
                )
            supplied_bounds = query.endpoint_point_bounds[query_row]
            for side, (key, corner, parameter, supplied) in enumerate(
                zip(
                    expected_endpoint_keys,
                    corners,
                    (first, last),
                    supplied_bounds,
                    strict=True,
                )
            ):
                source_lower, source_upper = self.source_corner_point_bounds(corner)
                if supplied is None:
                    raise ValueError(
                        "MeshingDomain curve query omits an authored trim point enclosure."
                    )
                for value, bound in zip(supplied[0], source_lower, strict=True):
                    if Fraction.from_float(float(value)) > bound:
                        raise ValueError(
                            "MeshingDomain curve endpoint lower bound excludes its source corner."
                        )
                for value, bound in zip(supplied[1], source_upper, strict=True):
                    if Fraction.from_float(float(value)) < bound:
                        raise ValueError(
                            "MeshingDomain curve endpoint upper bound excludes its source corner."
                        )
                lower, upper = query.chart_point_bounds(chart_id, parameter)
                lower = np.maximum(lower, np.asarray(supplied[0], dtype=np.float64))
                upper = np.minimum(upper, np.asarray(supplied[1], dtype=np.float64))
                if np.any(lower > upper):
                    raise ValueError(
                        "MeshingDomain trim endpoint enclosures have empty intersection."
                    )
                previous = expected_point_bounds.get(key)
                if previous is not None:
                    lower = np.maximum(lower, previous[0])
                    upper = np.minimum(upper, previous[1])
                    if np.any(lower > upper):
                        raise ValueError(
                            "Repeated MeshingDomain point identities have disjoint enclosures."
                        )
                expected_point_bounds[key] = (lower, upper)
        expected_keys = tuple(sorted(expected_point_bounds))
        actual_keys = tuple(
            zip(query.point_entity_ids, query.point_occurrence_paths, strict=True)
        )
        if actual_keys != expected_keys:
            raise ValueError(
                "MeshingDomain curve query point bank differs from its authored trim endpoints."
            )
        for key, (expected_lower, expected_upper) in expected_point_bounds.items():
            actual_lower, actual_upper = query.source_point_bounds(*key)
            if not np.array_equal(actual_lower, expected_lower) or not np.array_equal(
                actual_upper, expected_upper
            ):
                raise ValueError(
                    "MeshingDomain curve query changed its continuous point enclosures."
                )

    def validate_point_boundary_query(
        self,
        query: ParametricCurveBoundarySource,
        requested_point_ids: ArrayLike,
        /,
    ) -> None:
        from ._curve_boundary import ParametricCurveBoundarySource

        if not isinstance(query, ParametricCurveBoundarySource):
            raise TypeError(
                "MeshingDomain point validation requires ParametricCurveBoundarySource."
            )
        raw_ids = np.asarray(requested_point_ids)
        if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "iu":
            raise TypeError("requested_point_ids must be an integer vector.")
        if (
            raw_ids.dtype.kind == "u"
            and raw_ids.size
            and np.any(raw_ids > np.iinfo(np.int64).max)
        ):
            raise ValueError("requested_point_ids must fit int64.")
        identifiers = np.asarray(raw_ids, dtype=np.int64)
        if identifiers.size == 0 or np.unique(identifiers).size != identifiers.size:
            raise ValueError(
                "requested_point_ids must be a nonempty set of unique corner IDs."
            )
        curve_ids_by_key = {
            self.source_entity_key_for_scope(1, int(identifier)): int(identifier)
            for identifier in self.scope_entity_ids(1).tolist()
        }
        try:
            query_curve_ids = np.asarray(
                sorted(
                    {
                        curve_ids_by_key[(entity_id, tuple(occurrence))]
                        for entity_id, occurrence in zip(
                            query.curve_entity_ids, query.occurrence_paths, strict=True
                        )
                    }
                ),
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                "MeshingDomain point query names an unauthored source curve."
            ) from error
        self.validate_curve_boundary_query(query, query_curve_ids)
        query_point_keys = set(
            zip(query.point_entity_ids, query.point_occurrence_paths, strict=True)
        )
        requested_keys = tuple(
            self.source_entity_key_for_scope(0, int(identifier))
            for identifier in identifiers.tolist()
        )
        if any(key not in query_point_keys for key in requested_keys):
            raise ValueError(
                "Requested corner scope is absent from the complete trim endpoint query."
            )

    def patch_curves(self, patch: int, /) -> np.ndarray:
        return np.asarray(
            sorted(
                {
                    use.curve
                    for loop in self.patches[patch].loops
                    for use in loop
                    if isinstance(use, PatchCurveUse)
                }
            ),
            dtype=np.int64,
        )

    def _batch(
        self, patches: ArrayLike, charts: ArrayLike, /
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(patches, dtype=np.int64).reshape((-1,))
        points = np.asarray(charts, dtype=np.float64).reshape((-1, 2))
        if rows.shape != (points.shape[0],) or not np.all(np.isfinite(points)):
            raise ValueError(
                "MeshingDomain patch indices and finite UV charts must align."
            )
        if np.any(rows < 0) or np.any(rows >= len(self.patches)):
            raise ValueError("MeshingDomain query references an absent patch.")
        return rows, points

    def evaluate(self, patches: ArrayLike, charts: ArrayLike, /) -> np.ndarray:
        rows, charts_ = self._batch(patches, charts)
        result = np.empty((rows.size, 3), dtype=np.float64)
        for patch in np.unique(rows):
            selected = rows == patch
            result[selected] = np.asarray(
                self.patches[int(patch)].surface.evaluate(jnp.asarray(charts_[selected])),
                dtype=np.float64,
            )
        return result

    def oriented_normals(
        self, patches: ArrayLike, charts: ArrayLike, /
    ) -> tuple[np.ndarray, np.ndarray]:
        rows, charts_ = self._batch(patches, charts)
        normals = np.zeros((rows.size, 3), dtype=np.float64)
        regular = np.zeros((rows.size,), dtype=np.bool_)
        for patch_index in np.unique(rows):
            selected = rows == patch_index
            patch = self.patches[int(patch_index)]
            differential = np.asarray(
                surface_differential(patch.surface, jnp.asarray(charts_[selected])),
                dtype=np.float64,
            )
            raw = patch.orientation * np.cross(
                differential[..., :, 0], differential[..., :, 1]
            )
            magnitude = np.linalg.norm(raw, axis=-1)
            speed = np.linalg.norm(differential, axis=-2)
            bound = 64.0 * np.finfo(np.float64).eps * np.prod(speed, axis=-1)
            valid = np.isfinite(magnitude) & (magnitude > bound)
            normals[selected] = np.where(
                valid[:, None], raw / np.where(valid, magnitude, 1.0)[:, None], 0.0
            )
            regular[selected] = valid
        return normals, regular

    def interpolation_bounds(
        self, patch: int, charts: _MeshingChartInput, /
    ) -> np.ndarray:
        """Bound surface-to-affine interpolation over complete UV chart cells."""
        if (
            isinstance(patch, bool)
            or not isinstance(patch, int)
            or patch < 0
            or patch >= len(self.patches)
        ):
            raise ValueError("interpolation_bounds.patch must name a source surface.")
        cells = np.asarray(charts, dtype=np.float64)
        if (
            cells.ndim != 3
            or cells.shape[1] < 3
            or cells.shape[2] != 2
            or not np.all(np.isfinite(cells))
        ):
            raise ValueError(
                "interpolation_bounds.charts must contain finite UV cell corners."
            )
        if cells.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)
        surface = self.patches[patch].surface
        spans = _exact_rational_surface_spans(surface)
        if spans is not None:
            exact_charts = _exact_uv_chart_bank(charts, cells)
            return _exact_spline_interpolation_bounds(cells, spans, exact_charts)
        lower = np.min(cells, axis=1)
        upper = np.max(cells, axis=1)
        second = jax.jacfwd(jax.jacfwd(surface.evaluate))
        prepared = prepare_interval_function(second, 2)
        hessian_lower, hessian_upper = prepared.evaluate(lower, upper)
        result = np.full((cells.shape[0],), np.inf, dtype=np.float64)
        for row in range(cells.shape[0]):
            low = hessian_lower[row]
            high = hessian_upper[row]
            if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
                continue
            widths = tuple(
                Fraction(float(upper[row, axis])) - Fraction(float(lower[row, axis]))
                for axis in range(2)
            )
            components = []
            for physical in range(low.shape[0]):
                value = (
                    sum(
                        (
                            max(
                                abs(Fraction(float(low[physical, first, second_axis]))),
                                abs(Fraction(float(high[physical, first, second_axis]))),
                            )
                            * widths[first]
                            * widths[second_axis]
                            for first in range(2)
                            for second_axis in range(2)
                        ),
                        Fraction(0),
                    )
                    / 2
                )
                components.append(value)
            squared = sum((value * value for value in components), Fraction(0))
            result[row] = fraction_endpoint(
                fraction_sqrt_interval(squared)[1], upper=True, saturate=True
            )
        return result

    def adaptive_interpolation_bounds(
        self,
        patch: int,
        charts: Sequence[_ExactUVChart],
        budgets: Sequence[Fraction],
        /,
    ) -> tuple[Fraction | None, ...]:
        """Bound surface-to-affine interpolation over exact UV triangles to a budget.

        Exact spline carriers keep their span Hessian bound. Other carriers refine
        each triangle adaptively until its bound meets ``budgets[row]``, a vertex
        lower bound exceeds it, or binary64 enclosure resolution is reached.
        ``None`` marks a row without a finite bound.
        """
        if (
            isinstance(patch, bool)
            or not isinstance(patch, int)
            or patch < 0
            or patch >= len(self.patches)
        ):
            raise ValueError(
                "adaptive_interpolation_bounds.patch must name a source surface."
            )
        exact_charts = tuple(
            tuple((Fraction(point[0]), Fraction(point[1])) for point in chart)
            for chart in charts
        )
        surface = self.patches[patch].surface
        spans = _exact_rational_surface_spans(surface)
        if spans is None:
            return _adaptive_triangle_interpolation_bounds(
                surface, exact_charts, tuple(Fraction(value) for value in budgets)
            )
        cells = np.asarray(
            [
                [[float(value) for value in point] for point in chart]
                for chart in exact_charts
            ],
            dtype=np.float64,
        ).reshape(len(exact_charts), -1, 2)
        bounds = _exact_spline_interpolation_bounds(cells, spans, exact_charts)
        return tuple(
            Fraction(float(value)) if math.isfinite(float(value)) else None
            for value in bounds
        )

    def normal_turn_bounds(self, patch: int, charts: _MeshingChartInput, /) -> np.ndarray:
        """Bound normal rotation in radians over complete UV chart cells."""
        if (
            isinstance(patch, bool)
            or not isinstance(patch, int)
            or patch < 0
            or patch >= len(self.patches)
        ):
            raise ValueError("normal_turn_bounds.patch must name a source surface.")
        cells = np.asarray(charts, dtype=np.float64)
        if (
            cells.ndim != 3
            or cells.shape[1] < 3
            or cells.shape[2] != 2
            or not np.all(np.isfinite(cells))
        ):
            raise ValueError(
                "normal_turn_bounds.charts must contain finite UV cell corners."
            )
        if cells.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)
        surface = self.patches[patch].surface
        spans = _exact_rational_surface_spans(surface)
        if spans is not None:
            exact_charts = _exact_uv_chart_bank(charts, cells)
            return _exact_spline_normal_turn_bounds(cells, spans, exact_charts)
        lower = np.min(cells, axis=1)
        upper = np.max(cells, axis=1)
        first = prepare_interval_function(jax.jacfwd(surface.evaluate), 2)
        second = prepare_interval_function(jax.jacfwd(jax.jacfwd(surface.evaluate)), 2)
        jacobian_lower, jacobian_upper = first.evaluate(lower, upper)
        hessian_lower, hessian_upper = second.evaluate(lower, upper)
        centers = np.mean(cells, axis=1)
        center_jacobians = np.asarray(
            surface_differential(surface, jnp.asarray(centers)), dtype=np.float64
        )
        result = np.full((cells.shape[0],), np.inf, dtype=np.float64)
        pi_upper = Fraction(float(np.nextafter(math.pi, math.inf)))
        for row in range(cells.shape[0]):
            center = center_jacobians[row]
            low_j = jacobian_lower[row]
            high_j = jacobian_upper[row]
            low_h = hessian_lower[row]
            high_h = hessian_upper[row]
            if not all(
                np.all(np.isfinite(value))
                for value in (center, low_j, high_j, low_h, high_h)
            ):
                continue
            exact_center = tuple(
                tuple(
                    Fraction(float(center[physical, parameter])) for parameter in range(2)
                )
                for physical in range(3)
            )
            radius_squared = Fraction(0)
            for physical in range(3):
                for parameter in range(2):
                    nominal = exact_center[physical][parameter]
                    interval_lower = Fraction(float(low_j[physical, parameter]))
                    interval_upper = Fraction(float(high_j[physical, parameter]))
                    radius = max(
                        abs(interval_lower - nominal), abs(interval_upper - nominal)
                    )
                    radius_squared += radius * radius
            radius = fraction_sqrt_interval(radius_squared)[1]
            g00 = sum((entry[0] * entry[0] for entry in exact_center), Fraction(0))
            g11 = sum((entry[1] * entry[1] for entry in exact_center), Fraction(0))
            g01 = sum((entry[0] * entry[1] for entry in exact_center), Fraction(0))
            gram_determinant = g00 * g11 - g01 * g01
            trace = g00 + g11
            if gram_determinant <= 0 or trace <= 0:
                continue
            center_singular_lower = fraction_sqrt_interval(gram_determinant / trace)[0]
            singular_lower = center_singular_lower - radius
            if singular_lower <= 0:
                continue
            jacobian_center_norm = fraction_sqrt_interval(
                sum(
                    (value * value for line in exact_center for value in line),
                    Fraction(0),
                )
            )[1]
            jacobian_norm = jacobian_center_norm + radius
            hessian_squared = Fraction(0)
            for low_value, high_value in zip(
                low_h.reshape((-1,)), high_h.reshape((-1,)), strict=True
            ):
                absolute = max(
                    abs(Fraction(float(low_value))), abs(Fraction(float(high_value)))
                )
                hessian_squared += absolute * absolute
            hessian_norm = fraction_sqrt_interval(hessian_squared)[1]
            exact_vertices = tuple(rational_vector(point) for point in cells[row])
            diameter_squared = max(
                (
                    sum(
                        (
                            (first_value - second_value) ** 2
                            for first_value, second_value in zip(
                                first_point, second_point, strict=True
                            )
                        ),
                        Fraction(0),
                    )
                    for first_point in exact_vertices
                    for second_point in exact_vertices
                ),
                default=Fraction(0),
            )
            diameter = fraction_sqrt_interval(diameter_squared)[1]
            chord = min(
                Fraction(2),
                2
                * hessian_norm
                * jacobian_norm
                * diameter
                / (singular_lower * singular_lower),
            )
            angle = min(pi_upper, pi_upper * chord / 2)
            result[row] = fraction_endpoint(angle, upper=True, saturate=True)
        return result

    def multiregion_surface(
        self,
        coordinates: ArrayLike,
        triangles: ArrayLike,
        triangle_patches: ArrayLike,
        /,
        *,
        vertex_global_ids: ArrayLike,
        cell_global_ids: ArrayLike,
    ) -> MultiRegionSurfaceTopology | None:
        """Label a patch triangulation by the regions on both sides of each patch.

        Triangles follow their patch's oriented normal. A region whose boundary
        uses the patch with sign ``+1`` lies behind that normal (``left``), one
        with sign ``-1`` in front (``right``); an unclaimed side is the exterior
        boundary label. Returns ``None`` when no patch bounds two regions, where
        every region boundary is an ordinary manifold surface.
        """
        from .multiregion_surface import MultiRegionSurfaceSeed

        sides: dict[int, list[tuple[int, int]]] = {}
        for index, region in enumerate(self.regions):
            for patch, sign in region.boundary:
                sides.setdefault(patch, []).append((index, sign))
        if all(len(uses) < 2 for uses in sides.values()):
            return None
        exterior = len(self.regions)
        patch_labels = np.full((len(self.patches), 2), -1, dtype=np.int64)
        for patch, uses in sides.items():
            left = tuple(index for index, sign in uses if sign > 0)
            right = tuple(index for index, sign in uses if sign < 0)
            if len(left) > 1 or len(right) > 1:
                raise ValueError(
                    "A multiregion patch must separate at most one region per side."
                )
            patch_labels[patch] = (
                left[0] if left else exterior,
                right[0] if right else exterior,
            )
        rows = np.asarray(triangle_patches, dtype=np.int64)
        labels = patch_labels[rows]
        if np.any(labels < 0):
            raise ValueError("Every multiregion surface patch must bound a region.")
        seed = MultiRegionSurfaceSeed(
            coordinates,
            triangles,
            labels,
            (*(region.name for region in self.regions), "exterior"),
            (*("finite",) * len(self.regions), "boundary"),
            source=self.domain_id,
            vertex_global_ids=vertex_global_ids,
            face_global_ids=cell_global_ids,
        )
        return seed.topology(seed.capacity_plan(resource_id=self.domain_id))

    def project(
        self,
        points: ArrayLike,
        patches: ArrayLike,
        seeds: ArrayLike,
        /,
    ) -> MeshingDomainProjection:
        points_ = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        rows, seeds_ = self._batch(patches, seeds)
        if points_.shape[0] != rows.size or not np.all(np.isfinite(points_)):
            raise ValueError(
                "MeshingDomain projection points must be finite and align with seeds."
            )
        from ..nonlinear import VectorLocalRootPlan

        plan = VectorLocalRootPlan(
            2, maximum_steps=32, tolerance=1.0e-12, plan_id="meshing-domain-projection"
        )
        parameters = np.array(seeds_, copy=True)
        converged = np.zeros((rows.size,), dtype=np.bool_)
        for patch_index in np.unique(rows):
            selected = rows == patch_index
            surface = self.patches[int(patch_index)].surface

            def solve(point: Array, seed: Array) -> tuple[Array, Array]:
                def residual(uv: Array) -> Array:
                    jacobian = jax.jacfwd(surface.evaluate)(uv)
                    return (
                        jacobian.T
                        @ (surface.evaluate(uv) - point)
                        / (self.scale * self.scale)
                    )

                root, diagnostics = plan.solve_with_diagnostics(residual, seed)
                return root, diagnostics.converged

            roots, status = jax.vmap(solve)(
                jnp.asarray(points_[selected]), jnp.asarray(seeds_[selected])
            )
            parameters[selected] = np.asarray(roots, dtype=np.float64)
            converged[selected] = np.asarray(status, dtype=np.bool_)
        projected = self.evaluate(rows, parameters)
        distances = np.linalg.norm(projected - points_, axis=-1)
        return MeshingDomainProjection(projected, parameters, distances, converged)

    @property
    def curve_count(self) -> int:
        return len(self.curves)


def _placed_brep_domain(
    domain_type: type[MeshingDomain],
    model: BRepModel,
    occurrences: tuple[BRepAssemblyOccurrence, ...],
    /,
) -> MeshingDomain:
    """Expand source-defined solid occurrences while preserving their exact paths."""
    geometry = model.geometry
    topology = geometry.topology
    patches: list[MeshingSurfacePatch] = []
    curves: list[MeshingDomainCurve] = []
    regions: list[MeshingDomainRegion] = []
    corner_points: list[tuple[float, float, float]] = []
    source_vertices: list[int] = []
    source_vertex_paths: list[tuple[str, ...]] = []
    source_edges: list[int] = []
    source_edge_paths: list[tuple[str, ...]] = []
    source_faces: list[int] = []
    source_face_paths: list[tuple[str, ...]] = []
    region_sources: list[int] = []
    region_paths: list[tuple[str, ...]] = []
    # Occurrences with the same exact placement realize one placed copy of each
    # shared source entity: a face, edge, or corner incident to several solids of
    # that copy is one domain slot, recorded under its first occurrence path, and
    # each region references the shared patch with its own shell sign.
    vertex_slots: dict[tuple[int, tuple[tuple[tuple[Fraction, ...], ...], int]], int] = {}
    edge_slots: dict[tuple[int, tuple[tuple[tuple[Fraction, ...], ...], int]], int] = {}
    face_slots: dict[tuple[int, tuple[tuple[tuple[Fraction, ...], ...], int]], int] = {}
    copies = _placed_copy_keys(occurrences)
    for occurrence in occurrences:
        placement = copies[occurrence.path]
        solid = occurrence.solid_index
        face_ids = topology.solid_faces[solid]
        edge_ids = sorted(
            {
                geometry.coedge_edges[coedge]
                for face in face_ids
                for loop in geometry.face_loops[face]
                for coedge in loop
            }
        )
        vertex_ids = sorted(
            {vertex for edge in edge_ids for vertex in geometry.edge_vertices[edge]}
        )
        vertex_map: dict[int, int] = {}
        for vertex in vertex_ids:
            if (vertex, placement) in vertex_slots:
                vertex_map[vertex] = vertex_slots[(vertex, placement)]
                continue
            vertex_map[vertex] = vertex_slots[(vertex, placement)] = len(corner_points)
            point = _brep_corner_realization(
                model, vertex, occurrence.path, occurrence.placement
            )
            corner_points.append((float(point[0]), float(point[1]), float(point[2])))
            source_vertices.append(vertex)
            source_vertex_paths.append(occurrence.path)
        edge_map: dict[int, int] = {}
        for edge in edge_ids:
            if (edge, placement) in edge_slots:
                edge_map[edge] = edge_slots[(edge, placement)]
                continue
            edge_map[edge] = edge_slots[(edge, placement)] = len(curves)
            start, end = geometry.edge_vertices[edge]
            curves.append(MeshingDomainCurve(vertex_map[start], vertex_map[end]))
            source_edges.append(edge)
            source_edge_paths.append(occurrence.path)
        local_face_map: dict[int, int] = {}
        for face in face_ids:
            if (face, placement) in face_slots:
                local_face_map[face] = face_slots[(face, placement)]
                continue
            local_face_map[face] = face_slots[(face, placement)] = len(patches)
            loops: list[tuple[PatchBoundaryUse, ...]] = []
            for loop in geometry.face_loops[face]:
                uses: list[PatchBoundaryUse] = []
                for coedge in loop:
                    edge = geometry.coedge_edges[coedge]
                    first, last = (
                        float(value) for value in np.asarray(geometry.edge_ranges[edge])
                    )
                    sense = geometry.coedge_senses[coedge]
                    pcurve = geometry.pcurves[coedge]
                    if geometry.edge_curves[edge] < 0:
                        uses.append(
                            PatchPoleUse(
                                vertex_map[geometry.edge_vertices[edge][0]],
                                pcurve,
                                first,
                                last,
                                sense=sense,
                            )
                        )
                    else:
                        uses.append(
                            PatchCurveUse(
                                edge_map[edge], pcurve, first, last, sense=sense
                            )
                        )
                loops.append(_lift_native_period_loop(model.patches[face], tuple(uses)))
            surface = _PlacedSurfacePatch(model.patches[face], occurrence.placement)
            face_orientation = (
                int(np.asarray(model.orientation)[face]) * surface.orientation_sign
            )
            patches.append(
                MeshingSurfacePatch(surface, tuple(loops), orientation=face_orientation)
            )
            source_faces.append(face)
            source_face_paths.append(occurrence.path)
        signs = topology.solid_face_orientations[solid]
        regions.append(
            MeshingDomainRegion(
                f"solid_{solid}_occurrence_{len(regions)}",
                tuple(
                    (local_face_map[face], sign)
                    for face, sign in zip(face_ids, signs, strict=True)
                ),
            )
        )
        region_sources.append(solid)
        region_paths.append(occurrence.path)
    return domain_type(
        tuple(patches),
        tuple(curves),
        len(corner_points),
        source_id=model.source_id,
        source_revision=model.source_revision,
        regions=tuple(regions),
        source_indices=(tuple(source_vertices), tuple(source_edges), tuple(source_faces)),
        source_occurrences=(
            tuple(source_vertex_paths),
            tuple(source_edge_paths),
            tuple(source_face_paths),
        ),
        region_source_indices=tuple(region_sources),
        region_source_occurrences=tuple(region_paths),
        source_kinds=("vertex", "edge", "face"),
        authority_id=model.model_id,
        brep_authority=model,
        corner_points=np.asarray(corner_points, dtype=np.float64),
    )


__all__ = [
    "DomainCurveMap",
    "MeshingDomain",
    "MeshingDomainCurve",
    "MeshingDomainProjection",
    "MeshingDomainRegion",
    "MeshingSurfacePatch",
    "MeshingSourceAccuracy",
    "PatchCurveUse",
    "PatchPoleUse",
    "PatchBoundaryUse",
    "use_endpoints",
]
