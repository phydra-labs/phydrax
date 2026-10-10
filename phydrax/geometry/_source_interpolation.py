from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from ..discretization._coordinate_enclosure import RationalEnclosureError
from ._interval_enclosure import interval_subtract
from ._planar_coverage import _reserve_fraction_work, rational_points
from .brep._patches import (
    _bspline_span_jet_bounds,
    _interval_cross,
    _interval_norm,
    _meridian_second_jets,
    _prepare_bspline_curve_spans,
    AbstractSurfacePatch,
    BSplineCurve,
    ExtrusionSurface,
)
from .brep._placed import PlacedSurface, source_transform_bounds
from .brep._sphere_membership import _square_root_interval


if TYPE_CHECKING:
    from ..discretization._coordinate_enclosure import CoordinateEnclosureBudget


class SourceInterpolationFailure(ValueError):
    """An actual source stratum could not prove the original chord enclosure."""

    def __init__(
        self, check: str, row: int, spans: int, visits: int, queries: int
    ) -> None:
        self.check, self.row, self.spans = check, row, spans
        self.visits, self.queries = visits, queries
        super().__init__(
            f"{check}: source row {row}, spans {spans}, visits {visits}, queries {queries}"
        )


def _norm_upper(vector: np.ndarray) -> float:
    """Outward norm without overflowing or underflowing the squared vector."""
    if not np.all(np.isfinite(vector)):
        return np.inf
    exact = rational_points(vector[None])[0]
    scale = max((abs(value) for value in exact), default=Fraction())
    if scale == 0:
        return 0.0
    _reserve_fraction_work((exact, (scale,)), 3 * len(exact) + 8, 1, 4 * len(exact) + 4)
    squared = sum(((value / scale) ** 2 for value in exact), Fraction())
    upper = Fraction(_square_root_interval(squared)[1]) * scale
    if upper > Fraction(float(np.finfo(np.float64).max)):
        return np.inf
    result = float(upper)
    return result if Fraction(result) >= upper else float(np.nextafter(result, np.inf))


def extrusion_interpolation_bounds(
    surface: AbstractSurfacePatch,
    charts: np.ndarray,
    budget: CoordinateEnclosureBudget,
    reserve_queries: Callable[[int], None] | None,
) -> np.ndarray | None:
    """Complete native-knot strip enclosure of an actual extrusion operation tree.

    The affine sweep cancels against the original triangle's barycentric chord.
    Each exact footprint is intersected with the actual profile spans. One span
    uses its closed one-sided second jet. Multiple C0 spans use the complete
    first-jet hull: after subtracting its midpoint slope, the source is Lipschitz
    with constant half the hull diameter; barycentric mean absolute deviation is
    at most half the native-u width. No global C1 or finite Hessian is asserted.

    A convex footprint's vertical sections are positive on its open ``u``
    range, so a span strip meets it in positive measure exactly when the
    span's closed ``u`` overlap has positive length, and the strips cover its
    measure exactly when those disjoint overlaps cover ``[first, last]``.
    Coverage is therefore proved on the exact ``u`` overlaps. Jets of one
    exact span overlap share one second-jet norm per call.
    """
    definition = surface.definition if isinstance(surface, PlacedSurface) else surface
    if not isinstance(definition, ExtrusionSurface) or not isinstance(
        definition.curve, BSplineCurve
    ):
        return None
    curve = definition.curve
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
    pieces = _prepare_bspline_curve_spans(curve, budget)
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
            lower, upper = _bspline_span_jet_bounds(
                pieces[span], low, high, order, budget
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
            footprint = rational_points(triangle)
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
            _reserve_fraction_work(((width,),), 4, 1, 4)
            if len(owners) == 1:
                span, low, high = owners[0]
                norm = norms.get((span, low, high))
                if norm is None:
                    lower, upper = jet(row, span, low, high, 2)
                    norm = _norm_upper(np.maximum(np.abs(lower), np.abs(upper)))
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
                diameter = interval_subtract((upper, upper), (lower, lower))[1]
                norm = _norm_upper(diameter)
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
    surface: AbstractSurfacePatch, charts: np.ndarray, placement: float, /
) -> np.ndarray | None:
    """Chart-to-affine interpolation bounds of a swept B-spline meridian.

    For ``x = o + R(u)(w(v) - o)`` about an exactly unit axis ``k``, with
    ``w`` a revolution's meridian or an offset's exact parallel meridian, the
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
    swept = _meridian_second_jets(surface, boxes)
    if swept is None:
        return None
    (relative, first, second), axis, smooth = swept
    unit = (axis, axis)
    huu = _interval_norm(_interval_cross(unit, relative))[1]
    huv = _interval_norm(_interval_cross(unit, first))[1]
    hvv = _interval_norm(second)[1]
    du, dv = widths.T
    with np.errstate(invalid="ignore", over="ignore"):
        bound = placement * (huu * du * du + 2 * huv * du * dv + hvv * dv * dv) / 8
    bound = np.nextafter(bound * (1 + 64 * np.finfo(np.float64).eps), np.inf)
    return np.where(smooth, bound, np.nan)
