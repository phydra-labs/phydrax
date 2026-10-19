from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from fractions import Fraction
from typing import final

import numpy as np

import phydrax.geometry._domain_numerics as domain_numerics_owner
import phydrax.geometry.brep._query as query_owner
from phydrax.geometry._planar_primitives import rational_points, reserve_fraction_work

from .._strict import StrictModule
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    RationalEnclosureError,
)
from ..discretization._exact_algebraic_geometry import rational_coefficient
from ..typing import checked
from ._interval_enclosure import (
    interval_add,
    interval_multiply,
    interval_subtract,
    interval_tan,
)
from .brep._exact import square_root_interval
from .brep._patches import (
    AbstractSurfacePatch,
    BernsteinRestrictionBank,
    bspline_span_jet_bounds,
    BSplineCurve,
    ConePatch,
    ExtrusionSurface,
    interval_cross,
    interval_norm,
    meridian_second_jets,
    MeridianOverlaps,
    OffsetSurface,
    prepare_bspline_curve_spans,
    sphere_source_equivalence,
    SpherePatch,
    TorusPatch,
)
from .brep._placed import PlacedSurface, source_transform_bounds
from .brep._request_preparations import TrimRequestPreparations


class SourceInterpolationFailure(ValueError):
    """Report a source-stratum failure to enclose its affine chord."""

    def __init__(
        self, check: str, row: int, spans: int, visits: int, queries: int
    ) -> None:
        self.check, self.row, self.spans = check, row, spans
        self.visits, self.queries = visits, queries
        super().__init__(
            f"MeshingDomain.interpolation_source must prove {check}; got "
            f"source row {row}, spans {spans}, visits {visits}, queries {queries}"
        )


def norm_upper(vector: np.ndarray, *, budget: CoordinateEnclosureBudget) -> float:
    """Outward norm without overflowing or underflowing the squared vector."""
    if not np.all(np.isfinite(vector)):
        return np.inf
    exact = tuple(
        rational_coefficient(value, role="source_interpolation.norm_coordinate")
        for value in rational_points(vector[None], budget=budget)[0]
    )
    scale = max((abs(value) for value in exact), default=Fraction())
    if scale == 0:
        return 0.0
    reserve_fraction_work(
        (exact, (scale,)), 3 * len(exact) + 8, 1, 4 * len(exact) + 4, budget=budget
    )
    squared = sum(((value / scale) ** 2 for value in exact), Fraction())
    upper = Fraction(square_root_interval(squared)[1]) * scale
    if upper > Fraction(float(np.finfo(np.float64).max)):
        return np.inf
    result = float(upper)
    return result if Fraction(result) >= upper else float(np.nextafter(result, np.inf))


def extrusion_interpolation_bounds(
    surface: AbstractSurfacePatch,
    charts: np.ndarray,
    reserve_queries: Callable[[int], None] | None,
    *,
    budget: CoordinateEnclosureBudget,
    curve: BSplineCurve,
) -> np.ndarray:
    """whole native-knot strip enclosure of an extrusion operation tree.

    The affine sweep cancels against the source triangle's barycentric chord.
    Each footprint is intersected with the profile spans. One span
    uses its closed one-sided second jet. Multiple C0 spans use the whole
    first-jet hull: after subtracting its midpoint slope, the source is Lipschitz
    with constant half the hull diameter; barycentric mean absolute deviation is
    at most half the native-u width. No global C1 or finite Hessian is asserted.

    A convex footprint's vertical sections are positive on its open ``u``
    range, so a span strip meets it in positive measure exactly when the
    span's closed ``u`` overlap has positive length, and the strips cover its
    measure exactly when those disjoint overlaps cover ``[first, last]``.
    Coverage is therefore proved on the ``u`` overlaps. Jets of one
    span overlap share one second-jet norm per call.
    """
    rotation = (
        np.asarray(surface.rotation) if isinstance(surface, PlacedSurface) else None
    )
    queries, visits = 0, 0

    def query() -> None:
        nonlocal queries
        if reserve_queries is not None:
            reserve_queries(1)
        queries += 1

    query()
    pieces = prepare_bspline_curve_spans(curve, budget=budget)
    budget.admit_work_bound(charts.shape[0])
    budget.reserve(0, charts.shape[0] * np.dtype(np.float64).itemsize)
    result = np.empty((charts.shape[0],), dtype=np.float64)
    # Scalar span-overlap norms outlive their row scopes; jet arrays do not.
    norms: dict[tuple[int, float, float], float] = {}

    def jet(
        row: int, span: int, low: float, high: float, order: int
    ) -> tuple[np.ndarray, np.ndarray]:
        query()
        try:
            with budget.measurement():
                lower, upper = bspline_span_jet_bounds(
                    pieces[span], low, high, order, budget=budget
                )
        except RationalEnclosureError as error:
            raise SourceInterpolationFailure(
                "source_knot_denominator", row, len(pieces), visits, queries
            ) from error
        if rotation is not None:
            return source_transform_bounds(rotation, lower, upper)
        return lower, upper

    for row, triangle in enumerate(charts):
        budget.reserve(1)
        with budget.temporary_scope():
            footprint = rational_points(triangle, budget=budget)
            first = min(point[0] for point in footprint)
            last = max(point[0] for point in footprint)
            if first == last:
                result[row] = 0.0
                continue
            covered = Fraction()
            owners = []
            for span, piece in enumerate(pieces):
                budget.reserve(1)
                visits += 1
                ((a, b),) = piece.parameter_bounds
                low, high = max(first, Fraction(a)), min(last, Fraction(b))
                if low >= high:
                    continue
                covered += high - low
                owners.append((span, float(low), float(high)))
            width = last - first
            if covered != width or not owners:
                raise SourceInterpolationFailure(
                    "source_knot_coverage", row, len(pieces), visits, queries
                )
            reserve_fraction_work(((width,),), 4, 1, 4, budget=budget)
            if len(owners) == 1:
                span, low, high = owners[0]
                norm = norms.get((span, low, high))
                if norm is None:
                    lower, upper = jet(row, span, low, high, 2)
                    norm = norm_upper(
                        np.maximum(np.abs(lower), np.abs(upper)), budget=budget
                    )
                    norms[span, low, high] = norm
                if not np.isfinite(norm):
                    raise SourceInterpolationFailure(
                        "source_knot_second_jet", row, len(pieces), visits, queries
                    )
                bound = Fraction(norm) * width * width / 8
            else:
                first_jets = [jet(row, span, low, high, 1) for span, low, high in owners]
                lower = np.min([value[0] for value in first_jets], axis=0)
                upper = np.max([value[1] for value in first_jets], axis=0)
                diameter = np.asarray(
                    interval_subtract((upper, upper), (lower, lower))[1]
                )
                norm = norm_upper(diameter, budget=budget)
                if not np.isfinite(norm):
                    raise SourceInterpolationFailure(
                        "source_knot_first_jet", row, len(pieces), visits, queries
                    )
                bound = Fraction(norm) * width / 4
            if bound > Fraction(float(np.finfo(np.float64).max)):
                raise SourceInterpolationFailure(
                    "source_knot_chord_range", row, len(pieces), visits, queries
                )
            value = float(bound)
            result[row] = (
                value if Fraction(value) >= bound else np.nextafter(value, np.inf)
            )
    return np.nextafter(result, np.inf)


def revolution_interpolation_bounds(
    surface: AbstractSurfacePatch,
    charts: np.ndarray,
    placement: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    meridian_jets: list[MeridianOverlaps] | None = None,
) -> np.ndarray | None:
    """Chart-to-affine interpolation bounds of a swept B-spline meridian.

    For ``x = o + R(u)(w(v) - o)`` about an exactly unit axis ``k``, with
    ``w`` a revolution's meridian or an offset's parallel meridian, the
    rotation is orthogonal, so ``|x_uu| = |k x (w - o)|``, ``|x_uv| = |k x w'|``
    and ``|x_vv| = |w''|`` at every chart point. The Taylor-variance bound
    ``(Huu du² + 2 Huv du dv + Hvv dv²) / 8`` of ``interpolation_bounds`` takes
    these row norms, scaled by the placement's operator norm, instead of
    componentwise rotated interval jets, which lose the rotation's
    cancellation. NaN rows meet a junction without a C1 sweep proof and need
    first-jet bounds; None, a surface without this batched meridian structure.
    """
    widths = np.ptp(charts, axis=1)
    boxes = np.stack((np.min(charts, axis=1), np.max(charts, axis=1)), axis=1)
    swept = meridian_second_jets(surface, boxes, budget=budget)
    if swept is None:
        return None
    (relative, first, second), axis, smooth = swept
    unit = (axis, axis)
    huu = interval_norm(interval_cross(unit, relative))[1]
    huv = interval_norm(interval_cross(unit, first))[1]
    hvv = interval_norm(second)[1]
    du, dv = widths.T
    with np.errstate(invalid="ignore", over="ignore"):
        bound = placement * (huu * du * du + 2 * huv * du * dv + hvv * dv * dv) / 8
    bound = np.nextafter(bound * (1 + 64 * np.finfo(np.float64).eps), np.inf)
    return np.where(smooth, bound, np.nan)


@final
class _InterpolationCharts(StrictModule):
    """Host cell charts and their parameter widths after rank validation."""

    coordinates: np.ndarray
    widths: np.ndarray


@final
class _InterpolationJets(StrictModule):
    """Second jets and first-jet fallbacks over closed source-knot cells."""

    lower: np.ndarray
    upper: np.ndarray
    first: dict[int, tuple[np.ndarray, np.ndarray]]


def _prepare_interpolation_charts(charts: np.ndarray) -> _InterpolationCharts:
    """Validate cell/node/chart axes and compute their parameter widths."""
    coordinates = np.asarray(charts, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[2] != 2:
        raise ValueError(
            f"MeshingDomain.interpolation_charts must have shape (cell, node, 2); got {coordinates.shape}"
        )
    widths_raw = np.ptp(coordinates, axis=1)
    expected_widths = (coordinates.shape[0], 2)
    widths = np.empty(expected_widths, dtype=np.float64)
    widths[...] = widths_raw
    return _InterpolationCharts(coordinates, widths)


def _cone_interpolation_bounds(
    surface: AbstractSurfacePatch,
    definition: ConePatch,
    prepared: _InterpolationCharts,
) -> np.ndarray:
    """Bound the cone's affine axial factor before enclosing radial derivatives."""
    coordinates, widths = prepared.coordinates, prepared.widths
    # A cone has no global axial Hessian bound, but its radial factor
    # is affine on this triangle's box. Bound that factor before taking
    # the radial frame norm, rather than losing the cone cancellation
    # in componentwise interval automatic differentiation.
    axial = (
        np.min(coordinates[..., 1], axis=1),
        np.max(coordinates[..., 1], axis=1),
    )
    angle = np.asarray(definition.semi_angle)
    slope = interval_tan((angle, angle))
    reference = np.asarray(definition.reference_radius)
    radius = interval_add((reference, reference), interval_multiply(axial, slope))
    radial = domain_numerics_owner.matrix_norm_upper(
        np.column_stack(
            (
                np.asarray(definition.first_axis),
                np.asarray(definition.second_axis),
            )
        )
    )
    if isinstance(surface, PlacedSurface):
        radial = float(
            np.nextafter(
                radial
                * domain_numerics_owner.matrix_norm_upper(np.asarray(surface.rotation)),
                np.inf,
            )
        )
    huu = np.nextafter(np.maximum(np.abs(radius[0]), np.abs(radius[1])) * radial, np.inf)
    huv = np.nextafter(np.maximum(np.abs(slope[0]), np.abs(slope[1])) * radial, np.inf)
    du, dv = widths.T
    bound = (huu * du * du + 2 * huv * du * dv) / 8
    return np.nextafter(bound * (1 + 64 * np.finfo(np.float64).eps), np.inf)


def _prepare_interpolation_jets(
    surface: AbstractSurfacePatch,
    boxes: np.ndarray,
    reserve_queries: Callable[[int], None] | None,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None,
    restriction_bank: BernsteinRestrictionBank | None,
    meridian_jets: list[MeridianOverlaps] | None,
) -> _InterpolationJets:
    """Obtain second jets, then query first jets only for continuous C0 rows."""
    lower, upper = surface.derivative_bounds_batch(
        boxes,
        order=2,
        budget=budget,
        preparations=preparations,
        restriction_bank=restriction_bank,
        meridian_jets=meridian_jets,
    )
    continuous_spline = domain_numerics_owner.continuous_spline_carrier(
        surface, budget=budget
    )
    c0_rows = np.asarray(
        [
            row
            for row, box in enumerate(boxes)
            if continuous_spline
            and not surface.is_c1_on(box, budget=budget, preparations=preparations)
        ],
        dtype=np.int64,
    )
    first_jets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if c0_rows.size:
        if reserve_queries is not None:
            reserve_queries(int(c0_rows.size))
        domain_numerics_owner.consume_source_queries(int(c0_rows.size), budget=budget)
        first_lower_raw, first_upper_raw = surface.derivative_bounds_batch(
            boxes[c0_rows],
            order=1,
            budget=budget,
            preparations=preparations,
            restriction_bank=restriction_bank,
            meridian_jets=meridian_jets,
        )
        expected = (c0_rows.size, 3, 2)
        if np.shape(first_lower_raw) != expected or np.shape(first_upper_raw) != expected:
            raise ValueError(
                f"MeshingDomain.first_jet_bounds must have shape {expected}; got {np.shape(first_lower_raw)} and {np.shape(first_upper_raw)}"
            )
        first_lower = np.empty(expected, dtype=np.float64)
        first_upper = np.empty(expected, dtype=np.float64)
        first_lower[...] = first_lower_raw
        first_upper[...] = first_upper_raw
        first_jets = {
            int(row): (first_lower[index], first_upper[index])
            for index, row in enumerate(c0_rows)
        }
    return _InterpolationJets(lower, upper, first_jets)


def _swept_interpolation_bounds(
    surface: AbstractSurfacePatch,
    definition: AbstractSurfacePatch,
    prepared: _InterpolationCharts,
    boxes: np.ndarray,
    reserve_queries: Callable[[int], None] | None,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None,
    restriction_bank: BernsteinRestrictionBank | None,
    meridian_jets: list[MeridianOverlaps] | None,
) -> np.ndarray | None:
    """Bound supported sweep cells and replace rough rows with first-jet remainders."""
    coordinates, widths = prepared.coordinates, prepared.widths
    swept = revolution_interpolation_bounds(
        definition,
        coordinates,
        domain_numerics_owner.matrix_norm_upper(np.asarray(surface.rotation))
        if isinstance(surface, PlacedSurface)
        else 1.0,
        budget=budget,
    )
    if swept is not None:
        rough = np.flatnonzero(np.isnan(swept))
        if rough.size:
            if reserve_queries is not None:
                reserve_queries(int(rough.size))
            domain_numerics_owner.consume_source_queries(int(rough.size), budget=budget)
            first_lower, first_upper = surface.derivative_bounds_batch(
                boxes[rough],
                order=1,
                budget=budget,
                preparations=preparations,
                restriction_bank=restriction_bank,
                meridian_jets=meridian_jets,
            )
            swept[rough] = np.nextafter(
                np.asarray(
                    [
                        domain_numerics_owner.first_jet_chord_remainder(
                            low, high, widths[row], budget=budget
                        )
                        for low, high, row in zip(
                            first_lower, first_upper, rough, strict=True
                        )
                    ]
                )
                * (1 + 64 * np.finfo(np.float64).eps),
                np.inf,
            )
        return swept
    return None


def _piecewise_interpolation_bounds(
    surface: AbstractSurfacePatch,
    definition: AbstractSurfacePatch,
    prepared: _InterpolationCharts,
    reserve_queries: Callable[[int], None] | None,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None,
    restriction_bank: BernsteinRestrictionBank | None,
    meridian_jets: list[MeridianOverlaps] | None,
) -> np.ndarray:
    """Combine closed-source Taylor bounds with first-jet bounds at C0 knots."""
    coordinates, widths = prepared.coordinates, prepared.widths
    bounds = []
    boxes = np.stack((np.min(coordinates, axis=1), np.max(coordinates, axis=1)), axis=1)
    swept = _swept_interpolation_bounds(
        surface,
        definition,
        prepared,
        boxes,
        reserve_queries,
        budget=budget,
        preparations=preparations,
        restriction_bank=restriction_bank,
        meridian_jets=meridian_jets,
    )
    if swept is not None:
        return swept
    jets = _prepare_interpolation_jets(
        surface,
        boxes,
        reserve_queries,
        budget=budget,
        preparations=preparations,
        restriction_bank=restriction_bank,
        meridian_jets=meridian_jets,
    )
    lower, upper, first_jets = jets.lower, jets.upper, jets.first
    for row, (low, high, width) in enumerate(zip(lower, upper, widths, strict=True)):
        if row in first_jets:
            first_lower_row, first_upper_row = first_jets[row]
            bounds.append(
                domain_numerics_owner.first_jet_chord_remainder(
                    first_lower_row, first_upper_row, width, budget=budget
                )
            )
            continue
        absolute = np.maximum(np.abs(low), np.abs(high))
        h = np.linalg.norm(absolute, axis=0)
        if np.any(~np.isfinite(h)) and np.any(width > 0):
            bounds.append(np.inf)
        else:
            bounds.append(float(width @ h @ width / 8))
    return np.nextafter(np.asarray(bounds) * (1 + 64 * np.finfo(np.float64).eps), np.inf)


def _analytic_interpolation_bounds(
    surface: AbstractSurfacePatch,
    prepared: _InterpolationCharts,
    hessian: tuple[float, float, float],
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Tighten analytic Hessian factors over each chart before Taylor contraction."""
    coordinates, widths = prepared.coordinates, prepared.widths
    du, dv = widths.T
    huu = np.full((coordinates.shape[0],), hessian[0])
    huv = np.full((coordinates.shape[0],), hessian[1])
    definition = surface.definition if isinstance(surface, PlacedSurface) else surface
    sphere = sphere_source_equivalence(definition, budget=budget)
    if sphere is not None:
        definition = sphere[0]
    else:
        match definition:
            case OffsetSurface():
                equivalent = definition.analytic_equivalent(budget=budget)
                if equivalent is not None:
                    definition = equivalent
            case _:
                pass
    lower = np.min(coordinates[..., 1], axis=1)
    upper = np.max(coordinates[..., 1], axis=1)
    match definition:
        case SpherePatch():
            closest = np.where(
                (lower <= 0) & (upper >= 0),
                0.0,
                np.minimum(np.abs(lower), np.abs(upper)),
            )
            huu *= np.maximum(np.cos(closest), 0.0)
        case TorusPatch() if float(definition.major_radius) > float(
            definition.minor_radius
        ):
            major, minor = (
                float(definition.major_radius),
                float(definition.minor_radius),
            )
            cosine = np.where(
                np.ceil(lower / (2 * np.pi)) <= np.floor(upper / (2 * np.pi)),
                1.0,
                np.maximum(np.cos(lower), np.cos(upper)),
            )
            sine = np.where(
                np.ceil((lower - 0.5 * np.pi) / np.pi)
                <= np.floor((upper - 0.5 * np.pi) / np.pi),
                1.0,
                np.maximum(np.abs(np.sin(lower)), np.abs(np.sin(upper))),
            )
            huu *= np.clip((major + minor * cosine) / (major + minor), 0.0, 1.0)
            huv *= np.clip(sine, 0.0, 1.0)
        case _:
            pass
    bound = (huu * du * du + 2 * huv * du * dv + hessian[2] * dv * dv) / 8
    return np.nextafter(bound * (1 + 64 * np.finfo(np.float64).eps), np.inf)


@checked
def surface_interpolation_bounds(
    surface: AbstractSurfacePatch,
    charts: np.ndarray,
    reserve_queries: Callable[[int], None] | None,
    *,
    standalone: bool,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None,
    restriction_bank: BernsteinRestrictionBank | None,
    meridian_jets: list[MeridianOverlaps] | None,
) -> np.ndarray:
    """Enclose continuous source-to-affine cell interpolation by carrier family."""
    prepared = _prepare_interpolation_charts(charts)
    definition = surface.definition if isinstance(surface, PlacedSurface) else surface
    match definition:
        case ExtrusionSurface(curve=BSplineCurve() as curve):
            stage = nullcontext()
            if standalone:
                policy = query_owner.BRepQueryPolicy()
                stage = budget.bound_stage(
                    policy.maximum_operations, policy.maximum_scratch_bytes
                )
            with budget.activate(), stage:
                enclosed = extrusion_interpolation_bounds(
                    surface,
                    prepared.coordinates,
                    reserve_queries,
                    curve=curve,
                    budget=budget,
                )

            return enclosed
        case ConePatch():
            return _cone_interpolation_bounds(surface, definition, prepared)
        case _:
            hessian = domain_numerics_owner.hessian_bounds(surface, budget=budget)
            if hessian is None:
                return _piecewise_interpolation_bounds(
                    surface,
                    definition,
                    prepared,
                    reserve_queries,
                    budget=budget,
                    preparations=preparations,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
            return _analytic_interpolation_bounds(
                surface, prepared, hessian, budget=budget
            )
