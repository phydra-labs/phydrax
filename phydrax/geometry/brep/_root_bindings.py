#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-root vertex definitions and exact B-Rep endpoint lineage."""

from __future__ import annotations

from fractions import Fraction
from typing import cast, TYPE_CHECKING, TypeAlias, Union

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...discretization._exact_algebraic_geometry import (
    coefficient_interval,
    coefficient_payload,
    ExactScalar,
    RadicalElement,
    rational_coefficient,
)
from ...typing import checked
from .._interval_enclosure import (
    Interval,
    interval_add,
    interval_multiply,
    prepare_interval_function,
)
from ._assembly import BRepAffinePlacement
from ._intersection import (
    _root_definition_payload,
    _same_parameter_source,
    BranchRootEndpoint,
    ConicParameterEndpoint,
    CurveSurfaceIntersectionRoot,
    endpoint_root_payload,
    IntersectionCurvePointRoot,
    NativePeriodEndpoint,
    RootEndpoint,
    TrimIntersectionRoot,
    TrimRootEndpoint,
    TripleSurfaceIntersectionRoot,
)
from ._intersection_curve import (
    CurveTrimSegment,
    IntersectionCurve,
    IntersectionPCurve,
    SurfaceRegion,
)
from ._patches import AbstractCurve, AbstractSurfacePatch, ConicParameterRoot
from ._placed import PlacedSurface, validate_pose
from ._trim_helpers import period_offset, period_scalar_bounds, PeriodOffset


if TYPE_CHECKING:
    from ..._interpolation import PreparedSourceSpans, RationalBSplineSurfaceSpan


BRepPrimaryVertexRoot: TypeAlias = Union[
    "BRepRootSupport",
    "BRepConicVertex",
    CurveSurfaceIntersectionRoot,
    TripleSurfaceIntersectionRoot,
    IntersectionCurvePointRoot,
    "BRepPlacedVertex",
]
_UVSourceRoot: TypeAlias = (
    TrimIntersectionRoot
    | CurveSurfaceIntersectionRoot
    | TripleSurfaceIntersectionRoot
    | IntersectionCurvePointRoot
)


def _source_payload(source: object, /) -> dict[str, object]:
    identity = getattr(source, "identity_payload", None)
    return (
        identity()
        if callable(identity)
        else {
            "type": f"{type(source).__module__}.{type(source).__qualname__}",
            "arrays": array_tree_fingerprint(source),
        }
    )


def _same_source(first: object, second: object, /) -> bool:
    return type(first) is type(second) and canonical_fingerprint(
        _source_payload(first)
    ) == canonical_fingerprint(_source_payload(second))


def _source_coefficients_are_binary(source: object, /) -> bool:
    for name in (
        "origin_exact",
        "center_exact",
        "first_axis_exact",
        "second_axis_exact",
        "axis_exact",
        "major_radius_exact",
        "minor_radius_exact",
        "first_radius_exact",
        "second_radius_exact",
        "first_exact",
        "last_exact",
        "distance_exact",
        "exact_control_points",
        "exact_weights",
        "exact_knots",
        "exact_u_knots",
        "exact_v_knots",
    ):
        value = getattr(source, name, None)
        if value is None:
            continue
        stack = list(value) if isinstance(value, (tuple, list)) else [value]
        while stack:
            component = stack.pop()
            if isinstance(component, (tuple, list)):
                stack.extend(component)
            elif isinstance(component, (Fraction, RadicalElement)):
                try:
                    if Fraction.from_float(float(component)) != component:
                        return False
                except (OverflowError, ValueError):
                    return False
    return all(
        child is None or _source_coefficients_are_binary(child)
        for child in (
            getattr(source, "base", None),
            getattr(source, "curve", None),
            getattr(source, "pcurve", None),
            getattr(source, "surface", None),
            getattr(source, "definition", None),
        )
    )


def _exact_cylinder_patch_box(
    patch: object, lower: np.ndarray, upper: np.ndarray, /
) -> np.ndarray:
    from .._interval_enclosure import interval_cos, interval_sin
    from .._rational import fraction_down, fraction_up
    from ._patches import CylinderPatch

    if not isinstance(patch, CylinderPatch):
        raise TypeError("Exact cylinder bounds require a CylinderPatch source.")
    angle = (np.asarray(lower[0]), np.asarray(upper[0]))
    cosine_float = interval_cos(angle)
    sine_float = interval_sin(angle)
    cosine = (
        Fraction.from_float(float(cosine_float[0])),
        Fraction.from_float(float(cosine_float[1])),
    )
    sine = (
        Fraction.from_float(float(sine_float[0])),
        Fraction.from_float(float(sine_float[1])),
    )
    height = (
        Fraction.from_float(float(lower[1])),
        Fraction.from_float(float(upper[1])),
    )

    def add(
        first: tuple[Fraction, Fraction],
        second: tuple[Fraction, Fraction],
        /,
    ) -> tuple[Fraction, Fraction]:
        return first[0] + second[0], first[1] + second[1]

    def multiply(
        first: tuple[Fraction, Fraction],
        second: tuple[Fraction, Fraction],
        /,
    ) -> tuple[Fraction, Fraction]:
        products = tuple(a * b for a in first for b in second)
        return min(products), max(products)

    cosine_coefficients = tuple(
        coefficient_interval(patch.radius_exact * value)
        for value in patch.first_axis_exact
    )
    sine_coefficients = tuple(
        coefficient_interval(patch.radius_exact * value)
        for value in patch.second_axis_exact
    )
    axis_coefficients = tuple(coefficient_interval(value) for value in patch.axis_exact)
    origin_coefficients = tuple(
        coefficient_interval(value) for value in patch.origin_exact
    )
    lower_values: list[float] = []
    upper_values: list[float] = []
    for index in range(3):
        radial = add(
            multiply(cosine_coefficients[index], cosine),
            multiply(sine_coefficients[index], sine),
        )
        axial = multiply(axis_coefficients[index], height)
        enclosure = add(add(origin_coefficients[index], radial), axial)
        lower_values.append(fraction_down(enclosure[0]))
        upper_values.append(fraction_up(enclosure[1]))
    return np.asarray((lower_values, upper_values), dtype=np.float64)


def _exact_torus_patch_box(
    patch: object, lower: np.ndarray, upper: np.ndarray, /
) -> np.ndarray:
    from .._interval_enclosure import interval_cos, interval_sin
    from .._rational import fraction_down, fraction_up
    from ._patches import TorusPatch

    if not isinstance(patch, TorusPatch):
        raise TypeError("Exact torus bounds require a TorusPatch source.")

    def add(
        first: tuple[Fraction, Fraction],
        second: tuple[Fraction, Fraction],
        /,
    ) -> tuple[Fraction, Fraction]:
        return first[0] + second[0], first[1] + second[1]

    def multiply(
        first: tuple[Fraction, Fraction],
        second: tuple[Fraction, Fraction],
        /,
    ) -> tuple[Fraction, Fraction]:
        products = tuple(a * b for a in first for b in second)
        return min(products), max(products)

    def exact_interval(value: ExactScalar, /) -> tuple[Fraction, Fraction]:
        return coefficient_interval(value, bits=160)

    def float_interval(values: Interval, /) -> tuple[Fraction, Fraction]:
        lower_value = np.asarray(values[0], dtype=np.float64)
        upper_value = np.asarray(values[1], dtype=np.float64)
        if lower_value.shape != () or upper_value.shape != ():
            raise ValueError("Torus trigonometric intervals must have scalar endpoints.")
        return Fraction.from_float(float(lower_value)), Fraction.from_float(
            float(upper_value)
        )

    longitude = (np.asarray(lower[0]), np.asarray(upper[0]))
    tube = (np.asarray(lower[1]), np.asarray(upper[1]))
    cosine_u = float_interval(interval_cos(longitude))
    sine_u = float_interval(interval_sin(longitude))
    cosine_v = float_interval(interval_cos(tube))
    sine_v = float_interval(interval_sin(tube))
    center = tuple(exact_interval(value) for value in patch.center_exact)
    first_axis = tuple(exact_interval(value) for value in patch.first_axis_exact)
    second_axis = tuple(exact_interval(value) for value in patch.second_axis_exact)
    axis = tuple(exact_interval(value) for value in patch.axis_exact)
    major = exact_interval(patch.major_radius_exact)
    minor = exact_interval(patch.minor_radius_exact)
    ring_radius = add(major, multiply(minor, cosine_v))
    lower_values: list[float] = []
    upper_values: list[float] = []
    for index in range(3):
        radial = add(
            multiply(cosine_u, first_axis[index]),
            multiply(sine_u, second_axis[index]),
        )
        height = multiply(minor, multiply(sine_v, axis[index]))
        enclosure = add(
            add(center[index], multiply(ring_radius, radial)),
            height,
        )
        lower_values.append(fraction_down(enclosure[0]))
        upper_values.append(fraction_up(enclosure[1]))
    return np.asarray((lower_values, upper_values), dtype=np.float64)


def _exact_span_surface_box(
    spans: PreparedSourceSpans[RationalBSplineSurfaceSpan],
    lower: np.ndarray,
    upper: np.ndarray,
    /,
) -> np.ndarray:
    """Union of exact rational Bernstein span value bounds over a UV box."""
    from .._rational import fraction_down, fraction_up

    query = tuple(
        (
            Fraction.from_float(float(lower[axis])),
            Fraction.from_float(float(upper[axis])),
        )
        for axis in range(2)
    )
    component_bounds: list[list[tuple[Fraction, Fraction]]] = [[], [], []]
    for span in spans.spans:
        restricted = tuple(
            (
                max(query[axis][0], span.parameter_bounds[axis][0]),
                min(query[axis][1], span.parameter_bounds[axis][1]),
            )
            for axis in range(2)
        )
        if any(first > last for first, last in restricted):
            continue
        enclosure = span.enclose(restricted)
        if not enclosure.positive_denominator:
            raise ValueError(
                "A rational B-spline surface span has no positive denominator."
            )
        if len(enclosure.value_bounds) != 3:
            raise ValueError("A B-spline surface source box must have three coordinates.")
        for axis, bounds in enumerate(enclosure.value_bounds):
            component_bounds[axis].append(bounds)
    if any(not bounds for bounds in component_bounds):
        raise ValueError(
            "The source box does not intersect the active B-spline surface domain."
        )
    lower_values = tuple(
        fraction_down(min(bounds[0] for bounds in component))
        for component in component_bounds
    )
    upper_values = tuple(
        fraction_up(max(bounds[1] for bounds in component))
        for component in component_bounds
    )
    return np.asarray((lower_values, upper_values), dtype=np.float64)


def _exact_line_extrusion_box(
    patch: AbstractSurfacePatch, lower: np.ndarray, upper: np.ndarray, /
) -> np.ndarray | None:
    """Enclose an extruded exact line, an affine map of ``(u, v)``.

    Rational coordinates (such as a box side ``lower + extent`` that no float
    represents) are evaluated exactly at the four parameter corners, where an
    affine map takes its extremes, and rounded outward.
    """
    from .._rational import fraction_down, fraction_up
    from ._patches import ExactLineCurve, ExtrusionSurface, LineCurve

    if not isinstance(patch, ExtrusionSurface):
        return None
    curve = patch.curve
    if isinstance(curve, LineCurve):
        origin, direction = curve.origin_exact, curve.direction_exact
    elif isinstance(curve, ExactLineCurve):
        origin = curve.first_exact
        direction = tuple(
            last - first
            for first, last in zip(curve.first_exact, curve.last_exact, strict=True)
        )
    else:
        return None
    if not all(isinstance(value, Fraction) for value in (*origin, *direction)):
        return None
    generator = tuple(
        Fraction.from_float(float(value)) for value in np.asarray(patch.direction)
    )
    corners = [
        (Fraction.from_float(float(u)), Fraction.from_float(float(v)))
        for u in (lower[0], upper[0])
        for v in (lower[1], upper[1])
    ]
    lower_values: list[float] = []
    upper_values: list[float] = []
    for index in range(3):
        values = [
            cast(Fraction, origin[index])
            + u * cast(Fraction, direction[index])
            + v * generator[index]
            for u, v in corners
        ]
        lower_values.append(fraction_down(min(values)))
        upper_values.append(fraction_up(max(values)))
    return np.asarray((lower_values, upper_values), dtype=np.float64)


def _patch_box(
    patch: AbstractSurfacePatch, lower: np.ndarray, upper: np.ndarray, /
) -> np.ndarray:
    """Enclose ``patch`` on a UV box.

    Spline carriers extract their exact spans for this one query; consumers
    of many queries restrict spans they prepared once (`_exact_span_surface_box`).
    """
    from ._patches import (
        BSplineSurfacePatch,
        CircleCurve,
        CylinderPatch,
        ExtrusionSurface,
        RevolutionSurface,
        RuledSurface,
        TorusPatch,
    )
    from ._tspline import TSplineSurfacePatch

    if lower.shape != (2,) or upper.shape != (2,):
        raise ValueError("A patch source box requires two ordered UV coordinates.")
    if isinstance(patch, TorusPatch):
        return _exact_torus_patch_box(patch, lower, upper)
    if isinstance(patch, RuledSurface) or (
        isinstance(patch, RevolutionSurface) and isinstance(patch.curve, CircleCurve)
    ):
        return np.asarray(patch.bounding_box(np.stack((lower, upper))), dtype=np.float64)
    if isinstance(patch, (BSplineSurfacePatch, TSplineSurfacePatch)):
        return _exact_span_surface_box(patch.prepare_spans(), lower, upper)
    if isinstance(patch, CylinderPatch) and not _source_coefficients_are_binary(patch):
        return _exact_cylinder_patch_box(patch, lower, upper)
    if isinstance(patch, ExtrusionSurface) and not _source_coefficients_are_binary(patch):
        linear_box = _exact_line_extrusion_box(patch, lower, upper)
        if linear_box is not None:
            return linear_box
    from ._model import _bspline_revolution_box

    source_box = _bspline_revolution_box(
        patch,
        (float(lower[0]), float(lower[1])),
        (float(upper[0]), float(upper[1])),
    )
    if source_box is not None:
        return np.stack(source_box)
    if not _source_coefficients_are_binary(patch):
        raise NotImplementedError(
            "Exact patch coefficients have no registered algebraic interval realization."
        )
    evaluator = prepare_interval_function(patch.evaluate, 2)
    lo, hi = evaluator.evaluate(
        np.asarray(lower, dtype=np.float64)[None],
        np.asarray(upper, dtype=np.float64)[None],
    )
    return np.stack((lo[0], hi[0]))


def _curve_box(curve: AbstractCurve, lower: float, upper: float, /) -> np.ndarray:
    if isinstance(curve, (IntersectionPCurve, CurveTrimSegment)):
        return curve.bounding_box(lower, upper)
    bounding_box = getattr(curve, "bounding_box", None)
    if callable(bounding_box):
        return np.asarray(bounding_box(lower, upper), dtype=np.float64)
    if not _source_coefficients_are_binary(curve):
        raise NotImplementedError(
            "Exact curve coefficients have no registered algebraic interval realization."
        )
    evaluator = prepare_interval_function(curve.evaluate, 1)
    lo, hi = evaluator.evaluate(
        np.asarray([[lower]], dtype=np.float64), np.asarray([[upper]], dtype=np.float64)
    )
    return np.stack((lo[0], hi[0]))


def _exact_source_domain_boundary_root_box(
    patch: AbstractSurfacePatch,
    root: TrimIntersectionRoot,
    curve_box: np.ndarray,
    /,
) -> np.ndarray | None:
    """Prove an isolated exact line root lies on the closed analytic source boundary."""
    from .._rational import fraction_down, fraction_up
    from ._patches import RevolutionSurface, RuledSurface

    if not isinstance(patch, (RevolutionSurface, RuledSurface)) or (
        isinstance(patch, RevolutionSurface)
        and not _source_coefficients_are_binary(patch)
    ):
        return None
    domain = patch.parameter_domain
    point = _exact_trim_root_chart_point(root, patch)
    if point is None:
        return None
    if not isinstance(point[0], (Fraction, RadicalElement)) or not isinstance(
        point[1], (Fraction, RadicalElement)
    ):
        return None
    point_exact: tuple[ExactScalar, ExactScalar] = (point[0], point[1])
    exact_domain: tuple[
        tuple[Fraction | None, Fraction | None], tuple[Fraction | None, Fraction | None]
    ]
    if domain is None:
        if not isinstance(patch, RuledSurface):
            return None
        exact_domain = ((None, Fraction(0)), (None, Fraction(1)))
    else:
        if any(not np.isfinite(value) for row in domain for value in row):
            return None
        exact_domain = (
            (Fraction.from_float(domain[0][0]), Fraction.from_float(domain[0][1])),
            (Fraction.from_float(domain[1][0]), Fraction.from_float(domain[1][1])),
        )
    on_boundary = False
    for axis in range(2):
        domain_lower, domain_upper = exact_domain[0][axis], exact_domain[1][axis]
        if (domain_lower is not None and point_exact[axis] < domain_lower) or (
            domain_upper is not None and point_exact[axis] > domain_upper
        ):
            return None
        on_boundary |= (
            domain_lower is not None and point_exact[axis] == domain_lower
        ) or (domain_upper is not None and point_exact[axis] == domain_upper)
    if not on_boundary:
        return None

    def parameter_bounds(value: ExactScalar, /) -> tuple[float, float]:
        lower, upper = coefficient_interval(value, bits=160)
        return fraction_down(lower), fraction_up(upper)

    intervals = tuple(parameter_bounds(value) for value in point_exact)
    lower = np.asarray(tuple(interval[0] for interval in intervals), dtype=np.float64)
    upper = np.asarray(tuple(interval[1] for interval in intervals), dtype=np.float64)
    if (
        curve_box.shape != (2, 2)
        or np.any(curve_box[0] > lower)
        or np.any(curve_box[1] < upper)
    ):
        return None
    for axis in range(2):
        if lower[axis] < upper[axis]:
            continue
        domain_lower, domain_upper = exact_domain[0][axis], exact_domain[1][axis]
        if domain_lower is not None and point_exact[axis] == domain_lower:
            lower[axis] = float(domain_lower)
            upper[axis] = np.nextafter(lower[axis], np.inf)
        elif domain_upper is not None and point_exact[axis] == domain_upper:
            upper[axis] = float(domain_upper)
            lower[axis] = np.nextafter(upper[axis], -np.inf)
        else:
            represented = float(point_exact[axis])
            lower[axis] = np.nextafter(represented, -np.inf)
            upper[axis] = np.nextafter(represented, np.inf)
            if domain_lower is not None:
                lower[axis] = max(float(domain_lower), lower[axis])
            if domain_upper is not None:
                upper[axis] = min(float(domain_upper), upper[axis])
        if (
            not np.isfinite(lower[axis])
            or not np.isfinite(upper[axis])
            or not lower[axis] < upper[axis]
            or Fraction.from_float(float(lower[axis])) > point_exact[axis]
            or Fraction.from_float(float(upper[axis])) < point_exact[axis]
        ):
            return None
    if np.any(lower >= upper):
        return None
    return np.stack((lower, upper))


def _root_support_parameter_box(
    patch: AbstractSurfacePatch, root: TrimIntersectionRoot, /
) -> np.ndarray:
    witness = root.endpoint_witness
    if witness is not None and _same_parameter_source(patch, witness.source_region.patch):
        return np.asarray(witness.uv_enclosure, dtype=np.float64)
    lower, upper = root.parameter_enclosure()
    curve_box = _curve_box(root.first, float(lower[0]), float(upper[0]))
    try:
        return np.asarray(patch.validate_parameter_box(curve_box), dtype=np.float64)
    except ValueError:
        boundary_box = _exact_source_domain_boundary_root_box(patch, root, curve_box)
        if boundary_box is not None:
            return boundary_box
        if (
            curve_box.shape != (2, 2)
            or not np.all(np.isfinite(curve_box))
            or np.any(curve_box[0] > curve_box[1])
            or not np.any(curve_box[0] == curve_box[1])
        ):
            raise
        # A constant UV coordinate still needs a nonempty surface query box.
        # Widen only that exact singleton, retaining the entire certified image.
        expanded = curve_box.copy()
        domain = patch.parameter_domain
        for axis in range(2):
            value = curve_box[0, axis]
            if value != curve_box[1, axis]:
                continue
            lower = np.nextafter(value, -np.inf)
            upper = np.nextafter(value, np.inf)
            if domain is not None and domain[0][axis] <= value <= domain[1][axis]:
                lower = max(lower, domain[0][axis])
                upper = min(upper, domain[1][axis])
            expanded[:, axis] = (lower, upper)
        return np.asarray(patch.validate_parameter_box(expanded), dtype=np.float64)


class BRepRootSupport(StrictModule):
    """Map one isolated planar trim root through an exact supporting surface."""

    patch: AbstractSurfacePatch
    root: TrimIntersectionRoot
    root_id: str = eqx.field(static=True)

    @checked
    def __init__(self, patch: AbstractSurfacePatch, root: TrimIntersectionRoot) -> None:
        self.patch, self.root = patch, root
        self.root_id = canonical_fingerprint(
            {
                "kind": "brep-root-support",
                "patch": _source_payload(patch),
                "root": _root_definition_payload(root),
            }
        )
        _root_support_parameter_box(patch, root)

    def point_enclosure(self) -> np.ndarray:
        uv = _root_support_parameter_box(self.patch, self.root)
        return _patch_box(self.patch, uv[0], uv[1])


class BRepCurveSurfaceLift(StrictModule):
    """Whole-range exact correspondence between a 3D edge and its face p-curve."""

    curve: AbstractCurve
    surface: AbstractSurfacePatch
    pcurve: AbstractCurve
    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)
    parameter_box: tuple[tuple[float, float], tuple[float, float]] = eqx.field(
        static=True
    )
    root_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        curve: AbstractCurve,
        surface: AbstractSurfacePatch,
        pcurve: AbstractCurve,
        first: float,
        last: float,
        /,
        *,
        source_region: SurfaceRegion,
    ) -> None:
        from ._correspondence import exact_source_lift

        if not _same_parameter_source(source_region.patch, surface):
            raise ValueError(
                "BRepCurveSurfaceLift.source_region must retain its exact surface source."
            )
        if not (np.isfinite(first) and np.isfinite(last) and first < last):
            raise ValueError(
                "BRepCurveSurfaceLift requires a finite increasing source interval."
            )
        region_box = np.asarray(source_region.parameter_box, dtype=np.float64)
        if not exact_source_lift(curve, pcurve, source_region, float(first), float(last)):
            raise ValueError(
                "BRepCurveSurfaceLift requires a proved source correspondence."
            )
        self.curve, self.surface, self.pcurve = curve, surface, pcurve
        self.first, self.last = float(first), float(last)
        self.parameter_box = (
            (float(region_box[0, 0]), float(region_box[0, 1])),
            (float(region_box[1, 0]), float(region_box[1, 1])),
        )
        self.root_id = canonical_fingerprint(self.identity_payload())

    def identity_payload(self, /) -> dict[str, object]:
        return {
            "kind": "brep-curve-surface-lift",
            "curve": _source_payload(self.curve),
            "surface": _source_payload(self.surface),
            "pcurve": _source_payload(self.pcurve),
            "parameter_box": tuple(
                tuple(float(value).hex() for value in row) for row in self.parameter_box
            ),
            "range": (float(self.first).hex(), float(self.last).hex()),
        }


type _LiftChartScalar = ExactScalar | PeriodOffset
type _LiftChartPoint = tuple[_LiftChartScalar, _LiftChartScalar]
type _ExactLineChart = tuple[_LiftChartPoint, tuple[ExactScalar, ExactScalar]]


def _exact_line_chart(
    curve: AbstractCurve, patch: AbstractSurfacePatch, /
) -> _ExactLineChart | None:
    from ._boolean_core import PeriodicPCurve
    from ._patches import ExactLineCurve, LineCurve

    source = curve
    shifts = [0, 0]
    symbols = patch.native_period_symbols()
    while isinstance(source, PeriodicPCurve):
        if not _same_parameter_source(source.patch, patch):
            return None
        for axis, shift in enumerate(source.period_shifts):
            if shift and symbols[axis] != "two_pi":
                return None
            shifts[axis] += shift
        source = source.source_curve
    if isinstance(source, LineCurve):
        origin, direction = source.origin_exact, source.direction_exact
    elif isinstance(source, ExactLineCurve):
        origin = source.first_exact
        direction = tuple(
            last - first
            for first, last in zip(source.first_exact, source.last_exact, strict=True)
        )
    else:
        return None
    if len(origin) != 2 or len(direction) != 2:
        return None
    origin_values: list[_LiftChartScalar] = list(origin)
    for axis, shift in enumerate(shifts):
        if shift:
            origin_value = origin_values[axis]
            if not isinstance(origin_value, Fraction):
                return None
            origin_values[axis] = period_offset(origin_value, Fraction(shift))
    return (
        (origin_values[0], origin_values[1]),
        (direction[0], direction[1]),
    )


def _chart_add(
    first: _LiftChartScalar, second: _LiftChartScalar, /
) -> _LiftChartScalar | None:
    if isinstance(first, PeriodOffset):
        return first + second if isinstance(second, (Fraction, PeriodOffset)) else None
    if isinstance(second, PeriodOffset):
        return first + second if isinstance(first, Fraction) else None
    return first + second


def _chart_subtract(
    first: _LiftChartScalar, second: _LiftChartScalar, /
) -> _LiftChartScalar | None:
    if isinstance(first, PeriodOffset):
        return first - second if isinstance(second, (Fraction, PeriodOffset)) else None
    if isinstance(second, PeriodOffset):
        return first - second if isinstance(first, Fraction) else None
    return first - second


def _chart_multiply(
    first: ExactScalar, second: _LiftChartScalar, /
) -> _LiftChartScalar | None:
    if isinstance(second, PeriodOffset):
        return second * first if isinstance(first, Fraction) else None
    return first * second


def _chart_divide(
    numerator: _LiftChartScalar, denominator: ExactScalar, /
) -> _LiftChartScalar | None:
    if denominator == 0:
        return None
    if isinstance(numerator, PeriodOffset):
        if not isinstance(denominator, Fraction):
            return None
        return period_offset(
            numerator.rational / denominator, numerator.turns / denominator
        )
    return numerator / denominator


def _chart_product(
    first: _LiftChartScalar, second: _LiftChartScalar, /
) -> _LiftChartScalar | None:
    if isinstance(first, PeriodOffset):
        return first * second if isinstance(second, Fraction) else None
    if isinstance(second, PeriodOffset):
        return second * first if isinstance(first, Fraction) else None
    return first * second


def _chart_cross(
    first: _LiftChartPoint, second: _LiftChartPoint, /
) -> _LiftChartScalar | None:
    left = _chart_product(first[0], second[1])
    right = _chart_product(first[1], second[0])
    return None if left is None or right is None else _chart_subtract(left, right)


def _exact_line_chart_image(
    line: _ExactLineChart, parameter: _LiftChartScalar, /
) -> _LiftChartPoint | None:
    origin, direction = line
    values: list[_LiftChartScalar] = []
    for base, slope in zip(origin, direction, strict=True):
        delta = _chart_multiply(slope, parameter)
        if delta is None:
            return None
        value = _chart_add(base, delta)
        if value is None:
            return None
        values.append(value)
    return values[0], values[1]


def _exact_trim_root_chart_point(
    root: TrimIntersectionRoot, patch: AbstractSurfacePatch, /
) -> _LiftChartPoint | None:
    if not np.isfinite(root.contraction) or root.contraction >= 1.0:
        return None
    first = _exact_line_chart(root.first, patch)
    second = _exact_line_chart(root.second, patch)
    if first is None or second is None:
        return None
    first_origin, first_direction = first
    second_origin, second_direction = second
    determinant = _chart_cross(first_direction, second_direction)
    if determinant is None or isinstance(determinant, PeriodOffset) or determinant == 0:
        return None
    delta = tuple(
        _chart_subtract(second_origin[axis], first_origin[axis]) for axis in range(2)
    )
    if delta[0] is None or delta[1] is None:
        return None
    delta_point: _LiftChartPoint = (delta[0], delta[1])
    first_numerator = _chart_cross(delta_point, second_direction)
    second_numerator = _chart_cross(delta_point, first_direction)
    if first_numerator is None or second_numerator is None:
        return None
    first_parameter = _chart_divide(first_numerator, determinant)
    second_parameter = _chart_divide(second_numerator, determinant)
    if first_parameter is None or second_parameter is None:
        return None

    lower = np.asarray(root.parameter_lower, dtype=np.float64)
    upper = np.asarray(root.parameter_upper, dtype=np.float64)

    def isolated(parameter: _LiftChartScalar, axis: int, /) -> bool:
        lower_bound = Fraction.from_float(float(lower[axis]))
        upper_bound = Fraction.from_float(float(upper[axis]))
        if isinstance(parameter, PeriodOffset):
            period_lower, period_upper = period_scalar_bounds(parameter)
            exact_lower = Fraction.from_float(float(period_lower))
            exact_upper = Fraction.from_float(float(period_upper))
        else:
            exact_lower, exact_upper = coefficient_interval(parameter, bits=160)
        return lower_bound <= exact_lower and exact_upper <= upper_bound

    if not isolated(first_parameter, 0) or not isolated(second_parameter, 1):
        return None
    first_point = _exact_line_chart_image(first, first_parameter)
    second_point = _exact_line_chart_image(second, second_parameter)
    return (
        first_point if first_point is not None and first_point == second_point else None
    )


def _line_parameter_for_chart_point(
    curve: AbstractCurve,
    patch: AbstractSurfacePatch,
    point: _LiftChartPoint,
    /,
) -> _LiftChartScalar | None:
    line = _exact_line_chart(curve, patch)
    if line is None:
        return None
    origin, direction = line
    for axis, slope in enumerate(direction):
        if slope == 0:
            if point[axis] != origin[axis]:
                return None
            continue
        difference = _chart_subtract(point[axis], origin[axis])
        if difference is None:
            return None
        parameter = _chart_divide(difference, slope)
        if parameter is None:
            return None
        if _exact_line_chart_image(line, parameter) == point:
            return parameter
    return None


def _lift_endpoint_parameters(
    lift: BRepCurveSurfaceLift, /
) -> tuple[_LiftChartScalar, _LiftChartScalar]:
    first = Fraction.from_float(float(lift.first))
    last = Fraction.from_float(float(lift.last))
    period = lift.curve.period
    if (
        lift.curve.native_period_symbol() == "two_pi"
        and period is not None
        and last - first == Fraction.from_float(float(period))
    ):
        return first, period_offset(first, Fraction(1))
    return first, last


def _same_lift_parameter(
    first: _LiftChartScalar,
    second: _LiftChartScalar,
    first_lift: BRepCurveSurfaceLift,
    second_lift: BRepCurveSurfaceLift,
    /,
) -> bool:
    if first == second:
        return True
    if (
        first_lift.curve.native_period_symbol() != "two_pi"
        or second_lift.curve.native_period_symbol() != "two_pi"
        or first_lift.curve.period is None
        or second_lift.curve.period is None
        or first_lift.surface.native_period_symbols()[0] != "two_pi"
        or second_lift.surface.native_period_symbols()[0] != "two_pi"
    ):
        return False

    def coefficients(value: _LiftChartScalar, /) -> tuple[ExactScalar, Fraction]:
        if isinstance(value, PeriodOffset):
            return value.rational, value.turns
        return value, Fraction(0)

    first_rational, first_turns = coefficients(first)
    second_rational, second_turns = coefficients(second)
    turn_difference = first_turns - second_turns
    return first_rational == second_rational and turn_difference.denominator == 1


def _isolated_axis_line_lift_endpoint(
    support: BRepRootSupport, lift: BRepCurveSurfaceLift, /
) -> Fraction | None:
    root = support.root
    if not np.isfinite(root.contraction) or not 0.0 <= root.contraction < 1.0:
        return None
    lower = np.asarray(root.parameter_lower, dtype=np.float64)
    upper = np.asarray(root.parameter_upper, dtype=np.float64)
    for source_axis, source_curve, other_curve in (
        (0, root.first, root.second),
        (1, root.second, root.first),
    ):
        if not _same_parameter_source(source_curve, lift.pcurve):
            continue
        line = _exact_line_chart(other_curve, support.patch)
        if line is None:
            continue
        origin, direction = line
        fixed_axes = tuple(axis for axis in range(2) if direction[axis] == 0)
        if len(fixed_axes) != 1:
            continue
        fixed_axis = fixed_axes[0]
        varying_axis = 1 - fixed_axis
        varying_origin = origin[varying_axis]
        if isinstance(varying_origin, PeriodOffset):
            continue
        for parameter in _lift_endpoint_parameters(lift):
            if not isinstance(parameter, Fraction):
                continue
            if not (
                Fraction.from_float(float(lower[source_axis]))
                <= parameter
                <= Fraction.from_float(float(upper[source_axis]))
            ):
                continue
            image = _curve_box(source_curve, float(parameter), float(parameter))
            if image.shape != (2, 2) or not np.all(np.isfinite(image)):
                continue
            if np.any(image[0] > image[1]):
                continue
            # Singleton equality proves incidence with the whole affine line;
            # overlap alone cannot identify a shared endpoint.
            if (
                image[0, fixed_axis] != image[1, fixed_axis]
                or _chart_subtract(
                    Fraction.from_float(float(image[0, fixed_axis])),
                    origin[fixed_axis],
                )
                != 0
            ):
                continue
            other_parameters = tuple(
                (Fraction.from_float(float(value)) - varying_origin)
                / direction[varying_axis]
                for value in image[:, varying_axis]
            )
            other_lower = coefficient_interval(min(other_parameters), bits=160)[0]
            other_upper = coefficient_interval(max(other_parameters), bits=160)[1]
            other_axis = 1 - source_axis
            if Fraction.from_float(
                float(lower[other_axis])
            ) <= other_lower and other_upper <= Fraction.from_float(
                float(upper[other_axis])
            ):
                # The entire endpoint image is in this isolated intersection,
                # whose uniqueness identifies the source root.
                return parameter
    return None


def _support_lift_parameter(
    support: BRepRootSupport, lift: BRepCurveSurfaceLift, /
) -> _LiftChartScalar | None:
    if not isinstance(support.root, TrimIntersectionRoot) or not _same_parameter_source(
        support.patch, lift.surface
    ):
        return None
    witness = support.root.endpoint_witness
    if witness is not None:
        from ._boolean_core import pcurve_affine_correspondence

        parameter = witness.parameter_exact
        first, last = witness.source_interval
        try:
            chart_transport = pcurve_affine_correspondence(
                witness.source_pcurve, lift.pcurve
            )
        except (ArithmeticError, TypeError, ValueError, NotImplementedError):
            chart_transport = None
        if (
            _same_parameter_source(support.patch, witness.source_region.patch)
            and _same_parameter_source(lift.curve, witness.source_curve)
            and chart_transport is not None
            and parameter in (first, last)
            and Fraction.from_float(lift.first) <= first
            and last <= Fraction.from_float(lift.last)
        ):
            return parameter
    point = _exact_trim_root_chart_point(support.root, support.patch)
    if point is None:
        return _isolated_axis_line_lift_endpoint(support, lift)
    parameter = _line_parameter_for_chart_point(lift.pcurve, lift.surface, point)
    return parameter if parameter in _lift_endpoint_parameters(lift) else None


def _shared_support_lift_pairs(
    first: BRepRootSupport,
    second: BRepRootSupport,
    lifts: tuple[BRepCurveSurfaceLift, ...],
    /,
) -> tuple[tuple[BRepCurveSurfaceLift, BRepCurveSurfaceLift], ...]:
    pairs = []
    for first_lift in lifts:
        if not _same_parameter_source(first_lift.surface, first.patch):
            continue
        for second_lift in lifts:
            if (
                _same_parameter_source(second_lift.surface, second.patch)
                and _same_parameter_source(first_lift.curve, second_lift.curve)
                and first_lift.first == second_lift.first
                and first_lift.last == second_lift.last
                and first_lift.curve.native_period_symbol()
                == second_lift.curve.native_period_symbol()
            ):
                pairs.append((first_lift, second_lift))
    return tuple(pairs)


def _support_roots_share_lifted_endpoint(
    first: BRepRootSupport,
    second: BRepRootSupport,
    lifts: tuple[BRepCurveSurfaceLift, ...],
    /,
) -> bool:
    for first_lift, second_lift in _shared_support_lift_pairs(first, second, lifts):
        first_parameter = _support_lift_parameter(first, first_lift)
        second_parameter = _support_lift_parameter(second, second_lift)
        if (
            first_parameter is not None
            and second_parameter is not None
            and _same_lift_parameter(
                first_parameter, second_parameter, first_lift, second_lift
            )
        ):
            return True
    return False


def _has_conic_support_lift(
    primary: BRepConicVertex,
    support: BRepRootSupport,
    lifts: tuple[BRepCurveSurfaceLift, ...],
    /,
) -> bool:
    return any(
        _same_parameter_source(lift.surface, support.patch)
        and _same_parameter_source(lift.curve, primary.root.conic)
        for lift in lifts
    )


def _conic_root_period_parameter(
    root: ConicParameterRoot, /
) -> Fraction | PeriodOffset | None:
    if root.conic.native_period_symbol() != "two_pi":
        return None
    if root.cosine_exact == 1 and root.sine_exact == 0:
        quarter_turns = Fraction(0)
    elif root.cosine_exact == 0 and root.sine_exact == 1:
        quarter_turns = Fraction(1, 4)
    elif root.cosine_exact == -1 and root.sine_exact == 0:
        quarter_turns = Fraction(1, 2)
    elif root.cosine_exact == 0 and root.sine_exact == -1:
        quarter_turns = Fraction(3, 4)
    else:
        return None
    return period_offset(Fraction(0), Fraction(root.turns) + quarter_turns)


def _conic_support_shares_lifted_endpoint(
    primary: BRepConicVertex,
    support: BRepRootSupport,
    lifts: tuple[BRepCurveSurfaceLift, ...],
    /,
) -> bool:
    from ._intersection_curve import exact_curve_parameter_point

    parameter = _conic_root_period_parameter(primary.root)
    if parameter is None:
        return False
    for lift in lifts:
        if (
            not _same_parameter_source(lift.surface, support.patch)
            or not _same_parameter_source(lift.curve, primary.root.conic)
            or parameter not in _lift_endpoint_parameters(lift)
        ):
            continue
        support_parameter = _support_lift_parameter(support, lift)
        if support_parameter is None or not _same_lift_parameter(
            parameter, support_parameter, lift, lift
        ):
            continue
        source_point = exact_curve_parameter_point(lift.curve, parameter)
        if source_point is None:
            continue
        if len(source_point) != 3:
            continue
        point = (
            source_point[0] + primary.offset_exact[0],
            source_point[1] + primary.offset_exact[1],
            source_point[2] + primary.offset_exact[2],
        )
        if point == primary.point_exact:
            return True
    return False


class BRepPlacedVertex(StrictModule):
    """Apply an authored occurrence pose to a literal point or a rooted vertex."""

    source_point: Array
    rotation: Array
    affine: BRepAffinePlacement = eqx.field(static=True)
    translation: Array
    source_root: BRepVertexRoot | None
    source_entity_id: str = eqx.field(static=True)
    root_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source_point: ArrayLike,
        rotation: ArrayLike,
        translation: ArrayLike,
        source_entity_id: str,
        /,
        *,
        source_root: BRepVertexRoot | None = None,
        affine: BRepAffinePlacement | None = None,
    ) -> None:
        point = np.asarray(source_point, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("BRepPlacedVertex.source_point must be a finite 3D vector.")
        if not isinstance(source_entity_id, str) or not source_entity_id:
            raise ValueError(
                "BRepPlacedVertex.source_entity_id must be a nonempty source identity."
            )
        if source_root is not None and not isinstance(source_root, BRepVertexRoot):
            raise TypeError("BRepPlacedVertex.source_root must be a BRepVertexRoot.")
        self.rotation, self.translation = validate_pose(rotation, translation)
        from ._placed import _exact_affine

        self.affine = _exact_affine(self.rotation, self.translation, affine)
        self.source_point = jnp.asarray(point, dtype=jnp.float64)
        self.source_root, self.source_entity_id = source_root, source_entity_id
        self.root_id = canonical_fingerprint(
            {
                "kind": "brep-placed-vertex",
                "source_entity_id": source_entity_id,
                "source_root": None if source_root is None else source_root.root_id,
                "source_point": array_tree_fingerprint(self.source_point),
                "rotation": array_tree_fingerprint(self.rotation),
                "translation": array_tree_fingerprint(self.translation),
                "affine": self.affine.identity_payload(),
            }
        )

    def point_enclosure(self) -> np.ndarray:
        if self.source_root is None:
            source = np.asarray(self.source_point, dtype=np.float64)
            box = np.stack((source, source))
        else:
            box = self.source_root.point_enclosure()
        from .._rational import fraction_down, fraction_up

        matrix = self.affine.matrix
        low = np.zeros((3,), dtype=np.float64)
        high = np.zeros((3,), dtype=np.float64)
        for row in range(3):
            total: Interval = (np.asarray(0.0), np.asarray(0.0))
            for column in range(3):
                coefficient: Interval = (
                    np.asarray(float(fraction_down(matrix[row][column]))),
                    np.asarray(float(fraction_up(matrix[row][column]))),
                )
                source_interval: Interval = (
                    np.asarray(box[0, column]),
                    np.asarray(box[1, column]),
                )
                total = interval_add(
                    total, interval_multiply(coefficient, source_interval)
                )
            translation: Interval = (
                np.asarray(float(fraction_down(matrix[row][3]))),
                np.asarray(float(fraction_up(matrix[row][3]))),
            )
            placed = interval_add(total, translation)
            low[row] = float(placed[0])
            high[row] = float(placed[1])
        return np.stack((low, high))

    def matches_pose(self, patch: PlacedSurface, /) -> bool:
        return patch.affine.matrix == self.affine.matrix


class BRepConicVertex(StrictModule):
    """Exact 3D vertex selected by a conic root and optional source offset."""

    root: ConicParameterRoot
    offset_exact: tuple[ExactScalar, ExactScalar, ExactScalar] = eqx.field(static=True)
    root_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        root: ConicParameterRoot,
        /,
        *,
        offset_exact: tuple[ExactScalar, ExactScalar, ExactScalar] = (
            Fraction(0),
            Fraction(0),
            Fraction(0),
        ),
    ) -> None:
        if not isinstance(root, ConicParameterRoot) or root.conic.ambient_dimension != 3:
            raise TypeError("BRepConicVertex.root must define a 3D conic parameter root.")
        if len(offset_exact) != 3 or any(
            not isinstance(value, (Fraction, RadicalElement)) for value in offset_exact
        ):
            raise TypeError(
                "BRepConicVertex.offset_exact must contain three exact algebraic coordinates."
            )
        self.root = root
        self.offset_exact = tuple(offset_exact)
        self.root_id = canonical_fingerprint(self.identity_payload())

    @property
    def point_exact(self) -> tuple[ExactScalar, ExactScalar, ExactScalar]:
        point = tuple(
            value + offset
            for value, offset in zip(
                self.root.exact_point(), self.offset_exact, strict=True
            )
        )
        return point[0], point[1], point[2]

    def identity_payload(self, /) -> dict[str, object]:

        return {
            "kind": "brep-conic-vertex",
            "root": self.root.identity_payload(),
            "offset": tuple(coefficient_payload(value) for value in self.offset_exact),
        }

    def point_enclosure_exact(
        self, /, *, bits: int = 96
    ) -> tuple[tuple[Fraction, Fraction, Fraction], tuple[Fraction, Fraction, Fraction]]:
        first, second, third = self.point_exact
        bounds = (
            coefficient_interval(first, bits=bits),
            coefficient_interval(second, bits=bits),
            coefficient_interval(third, bits=bits),
        )
        return (
            (bounds[0][0], bounds[1][0], bounds[2][0]),
            (bounds[0][1], bounds[1][1], bounds[2][1]),
        )

    def point_enclosure(self, /) -> np.ndarray:
        from .._rational import fraction_down, fraction_up

        lower, upper = self.point_enclosure_exact()
        return np.asarray(
            (
                tuple(fraction_down(value) for value in lower),
                tuple(fraction_up(value) for value in upper),
            ),
            dtype=np.float64,
        )


def revolution_offset_conic_frame(
    surface: AbstractSurfacePatch,
    parameter: ExactScalar | PeriodOffset,
    /,
) -> (
    tuple[
        tuple[ExactScalar, ExactScalar, ExactScalar],
        tuple[ExactScalar, ExactScalar, ExactScalar],
        tuple[ExactScalar, ExactScalar, ExactScalar],
        tuple[ExactScalar, ExactScalar, ExactScalar],
        ExactScalar,
    ]
    | None
):
    """Return the exact offset point and circular frame at a meridian parameter."""
    from ..._interpolation import rational_bspline_curve_spans
    from .._rational import rational_vector
    from ._patches import (
        BSplineCurve,
        CircleCurve,
        EllipseCurve,
        OffsetSurface,
        RevolutionSurface,
    )

    if (
        not isinstance(surface, OffsetSurface)
        or surface.frame_sign not in (-1, 1)
        or not isinstance(surface.base, RevolutionSurface)
    ):
        return None
    profile = surface.base.curve
    if isinstance(profile, (CircleCurve, EllipseCurve)):
        if (
            len(profile.center_exact) != 3
            or len(profile.first_axis_exact) != 3
            or len(profile.second_axis_exact) != 3
        ):
            return None
        if isinstance(parameter, PeriodOffset):
            if parameter.rational != 0:
                return None
            quarter_turns = 4 * parameter.turns
            if quarter_turns.denominator != 1:
                return None
            phase = int(quarter_turns) % 4
        else:
            try:
                exact_parameter = rational_coefficient(
                    parameter, role="conic profile parameter"
                )
            except (TypeError, ValueError):
                return None
            if exact_parameter != 0:
                return None
            phase = 0
        cosine, sine = (
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
            (Fraction(-1), Fraction(0)),
            (Fraction(0), Fraction(-1)),
        )[phase]
        if isinstance(profile, CircleCurve):
            first_radius = second_radius = profile.radius_exact
        else:
            first_radius = profile.first_radius_exact
            second_radius = profile.second_radius_exact
        profile_point = tuple(
            profile.center_exact[index]
            + first_radius * cosine * profile.first_axis_exact[index]
            + second_radius * sine * profile.second_axis_exact[index]
            for index in range(3)
        )
        profile_tangent = tuple(
            -first_radius * sine * profile.first_axis_exact[index]
            + second_radius * cosine * profile.second_axis_exact[index]
            for index in range(3)
        )
    elif isinstance(profile, BSplineCurve):
        if not isinstance(parameter, (Fraction, RadicalElement)):
            return None
        knot_domain = (
            profile.exact_knots[profile.degree],
            profile.exact_knots[len(profile.exact_control_points)],
        )
        if parameter not in knot_domain:
            return None
        try:
            spans = rational_bspline_curve_spans(
                profile.exact_control_points,
                profile.exact_weights,
                profile.exact_knots,
                profile.degree,
            )
        except (ArithmeticError, TypeError, ValueError, ZeroDivisionError):
            return None
        if not spans:
            return None
        span = spans[0] if parameter == knot_domain[0] else spans[-1]
        lower, upper = span.parameter_bounds
        coefficients = span.homogeneous_bernstein_coefficients
        if len(coefficients) <= 1 or upper <= lower:
            return None
        degree = span.degree
        if parameter == knot_domain[0]:
            point_h = coefficients[0]
            derivative_h = tuple(
                Fraction(degree) * (second - first) / (upper - lower)
                for first, second in zip(coefficients[0], coefficients[1], strict=True)
            )
        else:
            point_h = coefficients[-1]
            derivative_h = tuple(
                Fraction(degree) * (last - previous) / (upper - lower)
                for previous, last in zip(coefficients[-2], coefficients[-1], strict=True)
            )
        weight = point_h[-1]
        weight_derivative = derivative_h[-1]
        if weight <= 0:
            return None
        profile_point = tuple(value / weight for value in point_h[:-1])
        profile_tangent = tuple(
            (derivative * weight - value * weight_derivative) / (weight * weight)
            for value, derivative in zip(point_h[:-1], derivative_h[:-1], strict=True)
        )
    else:
        return None
    if (
        len(profile_point) != 3
        or len(profile_tangent) != 3
        or not any(profile_tangent)
        or any(
            not isinstance(value, (Fraction, RadicalElement))
            for value in (*profile_point, *profile_tangent)
        )
    ):
        return None
    profile_point_3: tuple[ExactScalar, ExactScalar, ExactScalar] = (
        profile_point[0],
        profile_point[1],
        profile_point[2],
    )
    profile_tangent_3: tuple[ExactScalar, ExactScalar, ExactScalar] = (
        profile_tangent[0],
        profile_tangent[1],
        profile_tangent[2],
    )
    axis_origin = rational_vector(np.asarray(surface.base.axis_origin))
    axis = rational_vector(np.asarray(surface.base.axis_direction))
    if len(axis_origin) != 3 or len(axis) != 3:
        return None
    axis_origin_3: tuple[Fraction, Fraction, Fraction] = (
        axis_origin[0],
        axis_origin[1],
        axis_origin[2],
    )
    axis_3: tuple[Fraction, Fraction, Fraction] = (axis[0], axis[1], axis[2])
    if sum((value * value for value in axis_3), Fraction(0)) != 1:
        return None
    relative = (
        profile_point_3[0] - axis_origin_3[0],
        profile_point_3[1] - axis_origin_3[1],
        profile_point_3[2] - axis_origin_3[2],
    )
    cross_axis_relative = (
        axis_3[1] * relative[2] - axis_3[2] * relative[1],
        axis_3[2] * relative[0] - axis_3[0] * relative[2],
        axis_3[0] * relative[1] - axis_3[1] * relative[0],
    )
    normal = (
        cross_axis_relative[1] * profile_tangent_3[2]
        - cross_axis_relative[2] * profile_tangent_3[1],
        cross_axis_relative[2] * profile_tangent_3[0]
        - cross_axis_relative[0] * profile_tangent_3[2],
        cross_axis_relative[0] * profile_tangent_3[1]
        - cross_axis_relative[1] * profile_tangent_3[0],
    )
    norm_squared = sum((value * value for value in normal), Fraction(0))
    try:
        rational_norm_squared = rational_coefficient(
            norm_squared, role="revolution normal squared norm"
        )
    except (TypeError, ValueError):
        return None
    if rational_norm_squared <= 0:
        return None
    normal_length = RadicalElement.sqrt(rational_norm_squared)
    point_values = tuple(
        value
        + surface.frame_sign * surface.distance_exact * normal_component / normal_length
        for value, normal_component in zip(profile_point_3, normal, strict=True)
    )
    point_exact: tuple[ExactScalar, ExactScalar, ExactScalar] = (
        point_values[0],
        point_values[1],
        point_values[2],
    )
    axial = sum(
        (
            (value - origin) * direction
            for value, origin, direction in zip(
                point_exact, axis_origin_3, axis_3, strict=True
            )
        ),
        Fraction(0),
    )
    center_exact: tuple[ExactScalar, ExactScalar, ExactScalar] = (
        axis_origin_3[0] + axial * axis_3[0],
        axis_origin_3[1] + axial * axis_3[1],
        axis_origin_3[2] + axial * axis_3[2],
    )
    radial: tuple[ExactScalar, ExactScalar, ExactScalar] = (
        point_exact[0] - center_exact[0],
        point_exact[1] - center_exact[1],
        point_exact[2] - center_exact[2],
    )
    radius_squared = sum((value * value for value in radial), Fraction(0))
    try:
        rational_radius_squared = rational_coefficient(
            radius_squared, role="revolution offset radius squared"
        )
    except (TypeError, ValueError):
        return None
    if rational_radius_squared <= 0:
        return None
    radius: ExactScalar = RadicalElement.sqrt(rational_radius_squared)
    first_axis: tuple[ExactScalar, ExactScalar, ExactScalar] = (
        radial[0] / radius,
        radial[1] / radius,
        radial[2] / radius,
    )
    second_axis: tuple[ExactScalar, ExactScalar, ExactScalar] = (
        axis_3[1] * first_axis[2] - axis_3[2] * first_axis[1],
        axis_3[2] * first_axis[0] - axis_3[0] * first_axis[2],
        axis_3[0] * first_axis[1] - axis_3[1] * first_axis[0],
    )
    return point_exact, center_exact, first_axis, second_axis, radius


def _curve_surface_root_shares_trim_support(
    spatial: CurveSurfaceIntersectionRoot,
    support: BRepRootSupport,
    lifts: tuple[BRepCurveSurfaceLift, ...],
    /,
) -> bool:
    from ._boolean_core import affine_root_endpoint

    root = support.root
    if not isinstance(root, TrimIntersectionRoot):
        return False
    if root.endpoint_witness is not None and spatial.endpoint_witness is not None:
        witness = root.endpoint_witness
        if (
            witness.identity_payload() != spatial.endpoint_witness.identity_payload()
            or not witness.binds_curves(root.first, root.second)
            or not _same_parameter_source(support.patch, witness.source_region.patch)
        ):
            return False
        witness.require_current()
        point_box = spatial.endpoint_witness.curve_surface_parameter_box(
            spatial.curve, spatial.surface
        )
        return bool(
            np.all(point_box[0] >= np.asarray(spatial.parameter_lower))
            and np.all(point_box[1] <= np.asarray(spatial.parameter_upper))
        )
    for source_operand in range(2):
        source_trim = (root.first, root.second)[source_operand]
        branch_trim = (root.second, root.first)[source_operand]
        if not isinstance(source_trim, CurveTrimSegment) or not isinstance(
            branch_trim, IntersectionPCurve
        ):
            continue
        branch = branch_trim.curve
        if not branch.fully_certified:
            continue
        lift = next(
            (
                item
                for item in lifts
                if _same_parameter_source(item.curve, spatial.curve)
                and _same_parameter_source(item.surface, support.patch)
                and _same_parameter_source(item.pcurve, source_trim.curve)
            ),
            None,
        )
        if lift is None:
            continue
        if _same_parameter_source(branch.first.patch, support.patch):
            if not _same_parameter_source(branch.second.patch, spatial.surface.patch):
                continue
            chart_columns = slice(2, 4)
        elif _same_parameter_source(branch.second.patch, support.patch):
            if not _same_parameter_source(branch.first.patch, spatial.surface.patch):
                continue
            chart_columns = slice(0, 2)
        else:
            continue
        source_endpoint = TrimRootEndpoint(
            root, "first" if source_operand == 0 else "second"
        )
        source_scale = source_trim.last - source_trim.first
        source_offset = source_trim.first
        if source_trim.reversed:
            source_scale = -source_scale
            source_offset = source_trim.last
        edge_endpoint = affine_root_endpoint(source_endpoint, source_scale, source_offset)
        edge_lower, edge_upper = edge_endpoint.parameter_enclosure()
        if edge_lower < lift.first or edge_upper > lift.last:
            continue
        branch_endpoint = TrimRootEndpoint(
            root, "second" if source_operand == 0 else "first"
        )
        branch_lower, branch_upper = branch_endpoint.parameter_enclosure()
        if not (0.0 <= branch_lower <= branch_upper <= branch.num_charts):
            continue
        try:
            coupled = branch.parameter_enclosures(branch_lower, branch_upper)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            continue
        chart_lower = np.min(coupled[:, 0, chart_columns], axis=0)
        chart_upper = np.max(coupled[:, 1, chart_columns], axis=0)
        reference = np.asarray(spatial.parameters[1:], dtype=np.float64)
        for axis, period in enumerate(spatial.surface.patch.periods):
            if (
                period is None
                or spatial.surface.patch.native_period_symbols()[axis] != "two_pi"
            ):
                continue
            shift = round(
                (reference[axis] - 0.5 * (chart_lower[axis] + chart_upper[axis]))
                / float(period)
            )
            if shift:
                period_lower, period_upper = period_scalar_bounds(
                    PeriodOffset(Fraction(0), Fraction(shift))
                )
                chart_lower[axis], chart_upper[axis] = interval_add(
                    (np.asarray(chart_lower[axis]), np.asarray(chart_upper[axis])),
                    (period_lower, period_upper),
                )
        lower = np.minimum(
            np.asarray(spatial.parameter_lower, dtype=np.float64),
            np.asarray((edge_lower, *chart_lower), dtype=np.float64),
        )
        upper = np.maximum(
            np.asarray(spatial.parameter_upper, dtype=np.float64),
            np.asarray((edge_upper, *chart_upper), dtype=np.float64),
        )
        margin = (
            64.0
            * np.finfo(np.float64).eps
            * (1.0 + np.maximum(np.abs(lower), np.abs(upper)))
        )
        lower, upper = (
            np.nextafter(lower - margin, -np.inf),
            np.nextafter(upper + margin, np.inf),
        )
        try:
            CurveSurfaceIntersectionRoot(
                spatial.curve,
                spatial.surface,
                parameter_lower=lower,
                parameter_upper=upper,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        return True
    return False


class BRepBranchNodeRoots(StrictModule):
    """Source-edge roots each certified as one isolated intersection-branch node.

    Where periodic seams of both branch faces cross on a branch, each seam edge
    has its own root against the opposite face; all of them are the branch
    node. Every endpoint carries the exact chart and deck evidence that
    certifies its root as that node, and the constructor checks it.
    """

    node: IntersectionCurvePointRoot
    endpoints: tuple[BranchRootEndpoint, ...]
    root_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        node: IntersectionCurvePointRoot,
        endpoints: tuple[BranchRootEndpoint, ...],
        /,
    ) -> None:
        if (
            not isinstance(node, IntersectionCurvePointRoot)
            or not float(node.parameter).is_integer()
        ):
            raise TypeError(
                "BRepBranchNodeRoots.node must be an integer intersection-branch node."
            )
        if (
            not isinstance(endpoints, tuple)
            or len(endpoints) < 2
            or any(not isinstance(value, BranchRootEndpoint) for value in endpoints)
        ):
            raise TypeError(
                "BRepBranchNodeRoots.endpoints must hold at least two branch endpoints."
            )
        root_ids = tuple(endpoint.root.root_id for endpoint in endpoints)
        if len(set(root_ids)) != len(root_ids):
            raise ValueError("BRepBranchNodeRoots endpoints must name distinct roots.")
        for endpoint in endpoints:
            if endpoint.curve.branch_id != node.curve.branch_id or not (
                endpoint.certifies_point_root(node)
            ):
                raise ValueError(
                    "BRepBranchNodeRoots endpoints must each certify the branch node."
                )
        self.node, self.endpoints = node, endpoints
        self.root_id = canonical_fingerprint(branch_node_roots_payload(self))

    def binds(self, root: object, /) -> bool:
        return any(
            _same_parameter_source(endpoint.root, root) for endpoint in self.endpoints
        )


def branch_node_roots_payload(roots: BRepBranchNodeRoots, /) -> dict[str, object]:
    return {
        "kind": "brep-branch-node-roots",
        "node": _root_definition_payload(roots.node),
        "endpoints": tuple(
            endpoint_root_payload(endpoint) for endpoint in roots.endpoints
        ),
    }


def _branch_node_binds_vertex(
    branch_node: object,
    primary: BRepPrimaryVertexRoot,
    spatial_root: CurveSurfaceIntersectionRoot | None,
    source_edge_lifts: tuple[BRepCurveSurfaceLift, ...],
    /,
) -> bool:
    """Whether a branch-node witness names the vertex's own source root.

    A source-root vertex is bound when its primary root is a witnessed root. A
    joint vertex keeps a witness only through its spatial root: the witness
    certifies that root as the branch node and the joint root certifies the
    same root, so both name one point.
    """
    if not isinstance(branch_node, BRepBranchNodeRoots):
        return False
    if isinstance(primary, CurveSurfaceIntersectionRoot):
        return branch_node.binds(primary)
    return (
        isinstance(primary, TripleSurfaceIntersectionRoot)
        and spatial_root is not None
        and branch_node.binds(spatial_root)
        and primary.certifies_root(spatial_root, source_edge_lifts=source_edge_lifts)
    )


class BRepVertexRoot(StrictModule):
    """One exact source-root vertex and its alternative face/edge witnesses."""

    primary: BRepPrimaryVertexRoot
    aliases: tuple[BRepRootSupport, ...]
    spatial_root: CurveSurfaceIntersectionRoot | None
    joint_root: TripleSurfaceIntersectionRoot | None
    source_edge_lifts: tuple[BRepCurveSurfaceLift, ...]
    branch_node: BRepBranchNodeRoots | None
    root_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        primary: BRepPrimaryVertexRoot,
        /,
        *,
        aliases: tuple[BRepRootSupport, ...] = (),
        spatial_root: CurveSurfaceIntersectionRoot | None = None,
        joint_root: TripleSurfaceIntersectionRoot | None = None,
        source_edge_lifts: tuple[BRepCurveSurfaceLift, ...] = (),
        branch_node: BRepBranchNodeRoots | None = None,
    ) -> None:
        if not isinstance(
            primary,
            (
                BRepRootSupport,
                BRepConicVertex,
                CurveSurfaceIntersectionRoot,
                TripleSurfaceIntersectionRoot,
                IntersectionCurvePointRoot,
                BRepPlacedVertex,
            ),
        ):
            raise TypeError(
                "BRepVertexRoot.primary must be a source-root or placed-root record."
            )
        if not isinstance(aliases, tuple) or any(
            not isinstance(value, BRepRootSupport) for value in aliases
        ):
            raise TypeError("BRepVertexRoot.aliases must contain BRepRootSupport values.")
        if not isinstance(source_edge_lifts, tuple) or any(
            not isinstance(value, BRepCurveSurfaceLift) for value in source_edge_lifts
        ):
            raise TypeError(
                "BRepVertexRoot.source_edge_lifts must contain BRepCurveSurfaceLift values."
            )
        for alias in aliases:
            if isinstance(primary, BRepRootSupport):
                has_shared_lifts = bool(
                    _shared_support_lift_pairs(primary, alias, source_edge_lifts)
                )
                if has_shared_lifts and not _support_roots_share_lifted_endpoint(
                    primary, alias, source_edge_lifts
                ):
                    raise ValueError(
                        "BRepRootSupport aliases need an exact shared-lift endpoint proof."
                    )
            elif (
                isinstance(primary, BRepConicVertex)
                and _has_conic_support_lift(primary, alias, source_edge_lifts)
                and not _conic_support_shares_lifted_endpoint(
                    primary, alias, source_edge_lifts
                )
            ):
                raise ValueError(
                    "A conic vertex support alias needs an exact shared-lift endpoint proof."
                )
            elif isinstance(primary, CurveSurfaceIntersectionRoot) and not (
                (
                    joint_root is not None
                    and joint_root.certifies_root(
                        alias.root, source_edge_lifts=source_edge_lifts
                    )
                )
                or _curve_surface_root_shares_trim_support(
                    primary, alias, source_edge_lifts
                )
            ):
                raise ValueError(
                    "CurveSurfaceIntersectionRoot aliases need an exact source-lift root proof."
                )
        if (
            joint_root is not None
            and not isinstance(primary, CurveSurfaceIntersectionRoot)
            and any(
                not joint_root.certifies_root(
                    alias.root, source_edge_lifts=source_edge_lifts
                )
                for alias in aliases
            )
        ):
            raise ValueError(
                "TripleSurfaceIntersectionRoot aliases need an exact joint-root proof."
            )
        if branch_node is not None and not _branch_node_binds_vertex(
            branch_node, primary, spatial_root, source_edge_lifts
        ):
            raise ValueError(
                "BRepVertexRoot.branch_node must certify the vertex source root as its node."
            )
        self.primary, self.aliases = primary, aliases
        self.spatial_root, self.joint_root = spatial_root, joint_root
        self.source_edge_lifts = source_edge_lifts
        self.branch_node = branch_node
        self.root_id = canonical_fingerprint(vertex_root_payload(self))

    @classmethod
    def from_curve_point(
        cls, curve: IntersectionCurve, parameter: float = 0.0, /
    ) -> BRepVertexRoot:
        return cls(IntersectionCurvePointRoot(curve, parameter))

    def point_enclosure(self) -> np.ndarray:
        return _primary_point_enclosure(self.primary)

    def supports_uv(
        self, patch: AbstractSurfacePatch, root: TrimIntersectionRoot, /
    ) -> bool:
        return any(
            _same_source(alias.patch, patch) and alias.root.root_id == root.root_id
            for alias in self.aliases
        ) or (
            isinstance(self.primary, BRepRootSupport)
            and _same_source(self.primary.patch, patch)
            and self.primary.root.root_id == root.root_id
        )

    def supports_endpoint(self, endpoint: RootEndpoint, /) -> bool:
        if isinstance(endpoint, NativePeriodEndpoint):
            return False
        if isinstance(endpoint, ConicParameterEndpoint):
            if isinstance(self.primary, BRepConicVertex):
                return endpoint.point_exact == self.primary.point_exact
            if (
                isinstance(self.primary, BRepPlacedVertex)
                and self.primary.source_root is not None
            ):

                def source_point(
                    root: BRepVertexRoot, /
                ) -> tuple[ExactScalar, ExactScalar, ExactScalar] | None:
                    if isinstance(root.primary, BRepConicVertex):
                        return root.primary.point_exact
                    if (
                        isinstance(root.primary, BRepPlacedVertex)
                        and root.primary.source_root is not None
                    ):
                        inner = source_point(root.primary.source_root)
                        return None if inner is None else root.primary.affine.apply(inner)
                    return None

                source = source_point(self.primary.source_root)
                return (
                    source is not None
                    and self.primary.affine.apply(source) == endpoint.point_exact
                )
            return False
        if (
            isinstance(self.primary, BRepPlacedVertex)
            and self.primary.source_root is not None
            and self.primary.source_root.supports_endpoint(endpoint)
        ):
            return True
        if (
            isinstance(endpoint, TrimRootEndpoint)
            and isinstance(endpoint.root, TrimIntersectionRoot)
            and endpoint.source_surface is not None
        ):
            joint = self.joint_root
            if joint is None and isinstance(self.primary, TripleSurfaceIntersectionRoot):
                joint = self.primary
            if (
                joint is not None
                and any(
                    _same_parameter_source(region.patch, endpoint.source_surface.patch)
                    for region in (joint.first, joint.second, joint.third)
                )
                and joint.certifies_root(
                    endpoint.root, source_edge_lifts=self.source_edge_lifts
                )
            ):
                return True
        endpoint_root = getattr(endpoint, "root", None)
        if endpoint_root is None:
            return False
        for root in (self.primary, self.spatial_root, self.joint_root):
            if root is None:
                continue
            root_id = (
                root.root_id
                if not isinstance(root, BRepRootSupport)
                else root.root.root_id
            )
            if endpoint_root.root_id == root_id:
                return True
        if self.branch_node is not None and self.branch_node.binds(endpoint_root):
            return True
        return any(alias.root.root_id == endpoint_root.root_id for alias in self.aliases)

    def _trim_endpoint_uv_box(
        self,
        patch: AbstractSurfacePatch,
        curve: AbstractCurve,
        endpoint: TrimRootEndpoint,
        root: TrimIntersectionRoot,
        /,
    ) -> np.ndarray | None:
        from ._boolean_core import pcurve_affine_correspondence
        from ._intersection import _affine_parameter_interval

        operand = root.first if endpoint.operand == "first" else root.second
        lower, upper = TrimRootEndpoint(root, endpoint.operand).parameter_enclosure()
        while isinstance(operand, CurveTrimSegment):
            first = Fraction.from_float(operand.first)
            last = Fraction.from_float(operand.last)
            scale, offset = (
                (first - last, last) if operand.reversed else (last - first, first)
            )
            lower, upper = _affine_parameter_interval(lower, upper, scale, offset)
            operand = operand.curve
        box = _root_support_parameter_box(patch, root)
        if _same_parameter_source(operand, curve):
            return box
        correspondence = pcurve_affine_correspondence(operand, curve)
        if correspondence is not None:
            matrix, offsets = correspondence
            if matrix != ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))):
                return None
            shifted = box.copy()
            for axis, offset in enumerate(offsets):
                if offset == 0:
                    continue
                if (
                    not isinstance(offset, PeriodOffset)
                    or offset.rational != 0
                    or offset.turns.denominator != 1
                    or patch.native_period_symbols()[axis] != "two_pi"
                ):
                    return None
                shift_lower, shift_upper = period_scalar_bounds(offset)
                shifted[0, axis], shifted[1, axis] = interval_add(
                    (box[0, axis], box[1, axis]), (shift_lower, shift_upper)
                )
            return shifted
        source_pcurve = endpoint.source_pcurve
        if source_pcurve is None or not _same_parameter_source(operand, source_pcurve):
            return None
        if any(
            _same_parameter_source(lift.curve, curve)
            and _same_parameter_source(lift.surface, patch)
            and _same_parameter_source(lift.pcurve, source_pcurve)
            and lift.first <= lower <= upper <= lift.last
            for lift in self.source_edge_lifts
        ):
            return box
        return None

    def _endpoint_uv_binding(
        self,
        patch: AbstractSurfacePatch,
        curve: AbstractCurve,
        endpoint: RootEndpoint | None,
        /,
    ) -> tuple[_UVSourceRoot, np.ndarray] | None:
        while isinstance(curve, CurveTrimSegment):
            curve = curve.curve
        if isinstance(endpoint, BranchRootEndpoint):
            source_side = endpoint.source_side
            if source_side not in ("first", "second"):
                return None
            # Bind the branch side carried by this patch. The endpoint root may
            # be isolated on the opposite face's edge; its source image still
            # encloses that root in this face's own parameters. A patch carrying
            # neither or both branch sides cannot select a chart.
            sides = tuple(
                candidate
                for candidate, region in (
                    ("first", endpoint.curve.first),
                    ("second", endpoint.curve.second),
                )
                if _same_parameter_source(region.patch, patch)
            )
            if len(sides) != 1:
                return None
            side = sides[0]
            if isinstance(curve, IntersectionPCurve):
                if (
                    curve.curve.branch_id != endpoint.curve.branch_id
                    or curve.side != side
                    or curve.reversed
                ):
                    return None
            elif (
                side != source_side
                or endpoint.source_pcurve is None
                or not _same_parameter_source(endpoint.source_pcurve, curve)
            ):
                return None
            try:
                image = endpoint.source_image()
            except (TypeError, ValueError, NotImplementedError):
                return None
            offset = 0 if side == "first" else 2
            return endpoint.root, image[:, offset : offset + 2]
        if not isinstance(endpoint, TrimRootEndpoint):
            return None
        region = endpoint.source_surface
        if region is not None and not _same_parameter_source(region.patch, patch):
            return None
        root = endpoint.root
        if region is None and not (
            any(
                _same_parameter_source(alias.patch, patch)
                and _same_parameter_source(alias.root, root)
                for alias in self.aliases
            )
            or (
                isinstance(self.primary, BRepRootSupport)
                and _same_parameter_source(self.primary.patch, patch)
                and _same_parameter_source(self.primary.root, root)
            )
        ):
            return None
        if isinstance(root, TrimIntersectionRoot):
            image = self._trim_endpoint_uv_box(patch, curve, endpoint, root)
            return None if image is None else (root, image)
        if (
            endpoint.source_pcurve is None
            or region is None
            or not _same_parameter_source(endpoint.source_pcurve, curve)
        ):
            return None
        from ._correspondence import exact_source_lift

        # Arc-local affine transports do not change the root's source UV frame.
        if isinstance(root, CurveSurfaceIntersectionRoot):
            source_interval = root.parameter_enclosure()
            lower, upper = float(source_interval[0, 0]), float(source_interval[1, 0])
        else:
            lower = upper = root.parameter
        if not exact_source_lift(root.curve, curve, region, lower, upper):
            return None
        return root, _curve_box(curve, lower, upper)

    def _exact_source_endpoint_uv(
        self,
        patch: AbstractSurfacePatch,
        curve: AbstractCurve,
        endpoint: RootEndpoint | None,
        /,
    ) -> tuple[object, ...] | None:
        while isinstance(curve, CurveTrimSegment):
            curve = curve.curve
        if isinstance(endpoint, ConicParameterEndpoint):
            if not self.supports_endpoint(endpoint):
                return None
            return (
                _exact_conic_uv_endpoint(patch, curve, endpoint)
                or _exact_bspline_lift_uv_endpoint(self, patch, curve, endpoint)
                or _exact_revolution_pole_uv_endpoint(self, patch, curve, endpoint)
            )
        if isinstance(endpoint, NativePeriodEndpoint):
            return (
                _exact_native_period_uv_endpoint(self, patch, curve, endpoint)
                or _exact_native_cylinder_uv_endpoint(self, patch, curve, endpoint)
                or _exact_revolution_pole_uv_endpoint(self, patch, curve, endpoint)
            )
        return None

    def source_endpoint_uv_box(
        self,
        patch: AbstractSurfacePatch,
        curve: AbstractCurve,
        endpoint: RootEndpoint | None,
        /,
    ) -> np.ndarray | None:
        """Outward UV enclosure of an endpoint proved against this vertex source."""
        from .._rational import fraction_down, fraction_up

        chart = self._exact_source_endpoint_uv(patch, curve, endpoint)
        if chart is not None and len(chart) == 3 and chart[0] in ("plane", "bspline"):
            first, second = chart[1], chart[2]
            if not isinstance(first, (Fraction, RadicalElement)) or not isinstance(
                second, (Fraction, RadicalElement)
            ):
                return None
            bounds = (
                coefficient_interval(first, bits=128),
                coefficient_interval(second, bits=128),
            )
            return np.asarray(
                (
                    tuple(fraction_down(bound[0]) for bound in bounds),
                    tuple(fraction_up(bound[1]) for bound in bounds),
                ),
                dtype=np.float64,
            )
        if chart is not None and len(chart) == 5 and chart[0] == "revolution-pole":
            from ._patches import _conic_parameter_root_bounds

            longitude, cosine, sine, turns = chart[1:]
            if (
                not isinstance(longitude, (Fraction, PeriodOffset))
                or not isinstance(cosine, (Fraction, RadicalElement))
                or not isinstance(sine, (Fraction, RadicalElement))
                or type(turns) is not int
            ):
                return None
            longitude_low, longitude_high = period_scalar_bounds(longitude)
            latitude_low, latitude_high = _conic_parameter_root_bounds(
                cosine, sine, turns
            )
            return np.asarray(
                ((longitude_low, latitude_low), (longitude_high, latitude_high)),
                dtype=np.float64,
            )
        if chart is not None and chart[0] in ("cylinder", "torus"):
            from ._patches import _conic_parameter_root_bounds

            def angle_bounds(
                cosine: object, sine: object, turns: object
            ) -> tuple[float, float] | None:
                if (
                    not isinstance(cosine, (Fraction, RadicalElement))
                    or not isinstance(sine, (Fraction, RadicalElement))
                    or type(turns) is not int
                ):
                    return None
                return _conic_parameter_root_bounds(cosine, sine, turns)

            if chart[0] == "cylinder" and len(chart) == 5:
                first_bounds = angle_bounds(chart[1], chart[2], chart[3])
                height = chart[4]
                if first_bounds is None or not isinstance(
                    height, (Fraction, RadicalElement)
                ):
                    return None
                height_bounds = coefficient_interval(height, bits=128)
                return np.asarray(
                    (
                        (first_bounds[0], fraction_down(height_bounds[0])),
                        (first_bounds[1], fraction_up(height_bounds[1])),
                    ),
                    dtype=np.float64,
                )
            if chart[0] == "torus" and len(chart) == 7:
                first_bounds = angle_bounds(chart[1], chart[2], chart[3])
                second_bounds = angle_bounds(chart[4], chart[5], chart[6])
                if first_bounds is None or second_bounds is None:
                    return None
                return np.asarray(
                    (
                        (first_bounds[0], second_bounds[0]),
                        (first_bounds[1], second_bounds[1]),
                    ),
                    dtype=np.float64,
                )
        if endpoint is None or not self.supports_endpoint(endpoint):
            return None
        binding = self._endpoint_uv_binding(patch, curve, endpoint)
        return None if binding is None else binding[1]

    def same_uv_endpoint(
        self,
        patch: AbstractSurfacePatch,
        first_curve: AbstractCurve,
        first_endpoint: RootEndpoint | None,
        second_curve: AbstractCurve,
        second_endpoint: RootEndpoint | None,
        /,
    ) -> bool:
        """Prove two endpoint witnesses name the same source UV root on a face."""
        if isinstance(
            first_endpoint, (ConicParameterEndpoint, NativePeriodEndpoint)
        ) or isinstance(second_endpoint, (ConicParameterEndpoint, NativePeriodEndpoint)):
            first_uv = self._exact_source_endpoint_uv(patch, first_curve, first_endpoint)
            second_uv = self._exact_source_endpoint_uv(
                patch, second_curve, second_endpoint
            )
            return first_uv is not None and first_uv == second_uv

        def root_is_bound(root: _UVSourceRoot, /) -> bool:
            if isinstance(root, TrimIntersectionRoot):
                if any(
                    _same_parameter_source(alias.patch, patch)
                    and _same_parameter_source(alias.root, root)
                    for alias in self.aliases
                ) or (
                    isinstance(self.primary, BRepRootSupport)
                    and _same_parameter_source(self.primary.patch, patch)
                    and _same_parameter_source(self.primary.root, root)
                ):
                    return True
                joint = self.joint_root
                if joint is None and isinstance(
                    self.primary, TripleSurfaceIntersectionRoot
                ):
                    joint = self.primary
                return (
                    joint is not None
                    and any(
                        _same_parameter_source(region.patch, patch)
                        for region in (joint.first, joint.second, joint.third)
                    )
                    and joint.certifies_root(
                        root, source_edge_lifts=self.source_edge_lifts
                    )
                )
            return any(
                candidate is not None and _same_parameter_source(candidate, root)
                for candidate in (self.primary, self.spatial_root, self.joint_root)
            ) or (self.branch_node is not None and self.branch_node.binds(root))

        first_binding = self._endpoint_uv_binding(patch, first_curve, first_endpoint)
        second_binding = self._endpoint_uv_binding(patch, second_curve, second_endpoint)
        if first_binding is None or second_binding is None:
            return False
        first_root, first_uv = first_binding
        second_root, second_uv = second_binding
        if not root_is_bound(first_root) or not root_is_bound(second_root):
            return False
        # Root proofs below establish incidence. These bounds only distinguish
        # its authored chart representative from other periodic sheets.
        for axis, symbol in enumerate(patch.native_period_symbols()):
            lower = Fraction.from_float(float(first_uv[0, axis])) - Fraction.from_float(
                float(second_uv[1, axis])
            )
            upper = Fraction.from_float(float(first_uv[1, axis])) - Fraction.from_float(
                float(second_uv[0, axis])
            )
            if not lower <= 0 <= upper:
                return False
            if symbol == "two_pi":
                period_lower, _ = period_scalar_bounds(
                    PeriodOffset(Fraction(0), Fraction(1))
                )
                period = Fraction.from_float(float(period_lower))
                if lower <= -period or upper >= period:
                    return False
        joint = self.joint_root
        if joint is None and isinstance(self.primary, TripleSurfaceIntersectionRoot):
            joint = self.primary
        if (
            joint is not None
            and isinstance(
                first_root,
                (
                    TrimIntersectionRoot,
                    CurveSurfaceIntersectionRoot,
                    TripleSurfaceIntersectionRoot,
                ),
            )
            and isinstance(
                second_root,
                (
                    TrimIntersectionRoot,
                    CurveSurfaceIntersectionRoot,
                    TripleSurfaceIntersectionRoot,
                ),
            )
        ):
            if joint.certifies_root(
                first_root, source_edge_lifts=self.source_edge_lifts
            ) and joint.certifies_root(
                second_root, source_edge_lifts=self.source_edge_lifts
            ):
                return True
        spatial = self.spatial_root
        if spatial is None and isinstance(self.primary, CurveSurfaceIntersectionRoot):
            spatial = self.primary
        if spatial is not None:
            spatial_root = spatial

            def binds_spatial_root(root: _UVSourceRoot, /) -> bool:
                if _same_parameter_source(root, spatial_root):
                    return True
                if not isinstance(root, TrimIntersectionRoot):
                    return False
                return any(
                    _same_parameter_source(alias.root, root)
                    and _curve_surface_root_shares_trim_support(
                        spatial_root, alias, self.source_edge_lifts
                    )
                    for alias in self.aliases
                )

            if binds_spatial_root(first_root) and binds_spatial_root(second_root):
                return True
        # Roots certified as one branch node name one point; the sheet bounds
        # above already place both on the same chart representative.
        if (
            self.branch_node is not None
            and self.branch_node.binds(first_root)
            and self.branch_node.binds(second_root)
        ):
            return True
        return joint is None and _same_parameter_source(first_root, second_root)


def _exact_revolution_pole_uv_endpoint(
    vertex: BRepVertexRoot,
    patch: AbstractSurfacePatch,
    curve: AbstractCurve,
    endpoint: ConicParameterEndpoint | NativePeriodEndpoint,
    /,
) -> tuple[object, ...] | None:
    """Replay the source-rooted pole chart and retain its native longitude sheet."""
    from .._rational import rational_vector
    from ._boolean_core import AffinePCurve, line_chart_coefficients
    from ._correspondence import _exact_vertex_root_source
    from ._patches import _validate_source_shadows, ConicRootLinePCurve, RevolutionSurface

    if not isinstance(patch, RevolutionSurface):
        return None
    point, root = _exact_vertex_root_source(vertex)
    if (
        point is None
        or root is None
        or not _same_parameter_source(root.conic, patch.curve)
        or root.exact_point() != point
    ):
        return None
    axis_origin = rational_vector(np.asarray(patch.axis_origin))
    axis = rational_vector(np.asarray(patch.axis_direction))
    relative = tuple(point[i] - axis_origin[i] for i in range(3))
    if (
        relative[1] * axis[2] - relative[2] * axis[1] != 0
        or relative[2] * axis[0] - relative[0] * axis[2] != 0
        or relative[0] * axis[1] - relative[1] * axis[0] != 0
    ):
        return None
    try:
        _validate_source_shadows(curve, set())
    except (TypeError, ValueError):
        return None
    if isinstance(endpoint, NativePeriodEndpoint):
        if (
            not isinstance(curve, AffinePCurve)
            or not isinstance(curve.curve, ConicRootLinePCurve)
            or curve.matrix != ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(0)))
            or curve.offset != (Fraction(0), Fraction(0))
            or curve.curve.phase_shift_exact != 0
            or curve.curve.height_origin_exact != 0
            or curve.curve.height_direction_exact != 1
            or curve.curve.root.identity_payload() != root.identity_payload()
            or endpoint.patch is None
            or not _same_parameter_source(endpoint.patch, patch)
            or endpoint.axis != 0
            or not _same_parameter_source(endpoint.curve, curve)
            or endpoint.rational != 0
            or endpoint.turns not in (0, 1)
        ):
            return None
        longitude = endpoint.exact_parameter
    else:
        chart = line_chart_coefficients(curve)
        if chart is None:
            return None
        origin, direction = chart
        if (
            not _same_parameter_source(endpoint.carrier, curve)
            or endpoint.parameter_exact is not None
            or not _same_parameter_source(endpoint.root.conic, patch.curve)
            or endpoint.point_exact != point
            or endpoint.root.exact_point() != point
            or direction != (Fraction(0), Fraction(1))
            or origin[1] != 0
        ):
            return None
        longitude = origin[0]
        root = endpoint.root
    return (
        "revolution-pole",
        longitude,
        root.cosine_exact,
        root.sine_exact,
        root.turns,
    )


def _exact_native_cylinder_uv_endpoint(
    vertex: BRepVertexRoot,
    patch: AbstractSurfacePatch,
    curve: AbstractCurve,
    endpoint: NativePeriodEndpoint,
    /,
) -> tuple[object, ...] | None:
    """Compose a physical native parameter with the cylinder's exact chart phase."""
    from ._correspondence import _exact_vertex_root_source, _periodic_pcurve_source
    from ._patches import (
        _combine_conic_parameter_roots,
        _validate_source_shadows,
        ConicParameterRoot,
        CylinderPatch,
        PlaneCylinderSectionPCurve,
    )

    if (
        not isinstance(patch, CylinderPatch)
        or not _same_parameter_source(endpoint.curve, curve)
        or endpoint.patch is None
        or not _same_parameter_source(endpoint.patch, patch)
        or endpoint.axis != 0
        or endpoint.rational != 0
    ):
        return None
    quarters = endpoint.turns * 4
    if quarters.denominator != 1:
        return None
    periodic = _periodic_pcurve_source(curve, patch)
    if periodic is None:
        return None
    source, shifts = periodic
    if (
        not isinstance(source, PlaneCylinderSectionPCurve)
        or shifts[1] != 0
        or source.offset_exact[0] != 0
        or source.linear_exact != (Fraction(1), Fraction(0))
        or source.cosine_exact[0] != 0
        or source.sine_exact[0] != 0
    ):
        return None
    try:
        _validate_source_shadows(curve, set())
    except (TypeError, ValueError):
        return None
    point, vertex_root = _exact_vertex_root_source(vertex)
    if point is None or vertex_root is None:
        return None
    cosine, sine = ((1, 0), (0, 1), (-1, 0), (0, -1))[quarters.numerator % 4]
    phase = source.angle_offset_root
    try:
        parameter_root = ConicParameterRoot(
            vertex_root.conic,
            Fraction(cosine),
            Fraction(sine),
            turns=quarters.numerator // 4,
        )
    except (TypeError, ValueError, OverflowError):
        return None
    if phase is None:
        chart_root = parameter_root
    elif phase.cosine_exact == 1 and phase.sine_exact == 0:
        chart_root = ConicParameterRoot(
            vertex_root.conic,
            parameter_root.cosine_exact,
            parameter_root.sine_exact,
            turns=parameter_root.turns + phase.turns,
        )
    else:
        chart_root = _combine_conic_parameter_roots(parameter_root, phase)
    if chart_root is None:
        return None
    height = (
        source.offset_exact[1]
        + source.cosine_exact[1] * cosine
        + source.sine_exact[1] * sine
    )
    actual = tuple(
        patch.origin_exact[i]
        + patch.radius_exact
        * (
            chart_root.cosine_exact * patch.first_axis_exact[i]
            + chart_root.sine_exact * patch.second_axis_exact[i]
        )
        + height * patch.axis_exact[i]
        for i in range(3)
    )
    if actual != point:
        return None
    return (
        "cylinder",
        chart_root.cosine_exact,
        chart_root.sine_exact,
        chart_root.turns + shifts[0],
        height,
    )


def _exact_native_period_uv_endpoint(
    vertex: BRepVertexRoot,
    patch: AbstractSurfacePatch,
    curve: AbstractCurve,
    endpoint: NativePeriodEndpoint,
    /,
) -> tuple[object, ...] | None:
    """Bind a cardinal native-period parameter to its exact planar source point."""
    from .._rational import rational_vector
    from ._correspondence import _exact_vertex_root_source
    from ._patches import (
        _validate_source_shadows,
        CircleCurve,
        PlaneCylinderSectionPCurve,
        PlanePatch,
    )

    if (
        not isinstance(patch, PlanePatch)
        or not _same_parameter_source(endpoint.curve, curve)
        or endpoint.rational != 0
    ):
        return None
    if endpoint.patch is not None and not _same_parameter_source(endpoint.patch, patch):
        return None
    quarters = endpoint.turns * 4
    if quarters.denominator != 1:
        return None
    try:
        _validate_source_shadows(curve, set())
    except (TypeError, ValueError):
        return None
    cosine, sine = ((1, 0), (0, 1), (-1, 0), (0, -1))[quarters.numerator % 4]
    if isinstance(curve, PlaneCylinderSectionPCurve):
        if (
            curve.linear_exact != (Fraction(0), Fraction(0))
            or curve.angle_offset_root is not None
        ):
            return None
        uv = tuple(
            curve.offset_exact[i]
            + cosine * curve.cosine_exact[i]
            + sine * curve.sine_exact[i]
            for i in range(2)
        )
    elif isinstance(curve, CircleCurve) and curve.ambient_dimension == 2:
        uv = tuple(
            curve.center_exact[i]
            + curve.radius_exact
            * (cosine * curve.first_axis_exact[i] + sine * curve.second_axis_exact[i])
            for i in range(2)
        )
    else:
        return None
    point, _ = _exact_vertex_root_source(vertex)
    if point is None:
        return None
    origin = rational_vector(np.asarray(patch.origin))
    first = rational_vector(np.asarray(patch.first_axis))
    second = rational_vector(np.asarray(patch.second_axis))
    if tuple(origin[i] + uv[0] * first[i] + uv[1] * second[i] for i in range(3)) != point:
        return None
    return ("plane", uv[0], uv[1])


def _torus_conic_uv_endpoint(
    patch: AbstractSurfacePatch,
    source: AbstractCurve,
    endpoint: ConicParameterEndpoint,
    shifts: tuple[int, int],
    /,
) -> tuple[object, ...] | None:
    """Identify a torus endpoint by exact angular directions and chart sheets."""
    from ._patches import (
        _validate_source_shadows,
        LineCurve,
        TorusPatch,
        TorusRootedCirclePCurve,
    )

    if not isinstance(patch, TorusPatch) or patch.native_period_symbols() != (
        "two_pi",
        "two_pi",
    ):
        return None
    try:
        _validate_source_shadows(source, set())
    except (TypeError, ValueError):
        return None
    if endpoint.root.exact_point() != endpoint.point_exact:
        return None

    def phase(
        cosine: ExactScalar,
        sine: ExactScalar,
        turns: int,
        orientation: int,
        quarter_turns: int,
        /,
    ) -> tuple[ExactScalar, ExactScalar, int]:
        # The principal conic parameter is in [0, 2π). Determine its
        # quadrant and boundary exactly, including axes, before rotating.
        if sine == 0 and cosine == 1:
            quadrant, boundary = 0, True
        elif sine > 0 and cosine > 0:
            quadrant, boundary = 0, False
        elif sine > 0 and cosine <= 0:
            quadrant, boundary = 1, cosine == 0
        elif sine <= 0 and cosine < 0:
            quadrant, boundary = 2, sine == 0
        else:
            quadrant, boundary = 3, cosine == 0
        if orientation == 1:
            sheet = turns + (quadrant + quarter_turns) // 4
        else:
            sheet = -turns + (quarter_turns - quadrant - (not boundary)) // 4
            sine = -sine
        a, b = ((1, 0), (0, 1), (-1, 0), (0, -1))[quarter_turns % 4]
        return a * cosine - b * sine, b * cosine + a * sine, sheet

    if endpoint.parameter_exact is None:
        parameter = (
            endpoint.root.cosine_exact,
            endpoint.root.sine_exact,
            endpoint.root.turns,
        )
    elif endpoint.parameter_exact == 0:
        parameter = (Fraction(1), Fraction(0), 0)
    else:
        # A nonzero rational radian is not an exact conic phase.
        return None

    if isinstance(source, TorusRootedCirclePCurve):
        quarters = source.turns * 4
        if quarters.denominator != 1:
            return None
        longitude = phase(*parameter, source.orientation, quarters.numerator)
        root = source.root_line.root
        latitude = (root.cosine_exact, root.sine_exact, root.turns)
    elif isinstance(source, LineCurve):
        if source.origin_exact != (Fraction(0), Fraction(0)) or (
            source.direction_exact
            not in ((Fraction(0), Fraction(1)), (Fraction(0), Fraction(-1)))
        ):
            return None
        longitude = (Fraction(1), Fraction(0), 0)
        latitude = phase(*parameter, 1 if source.direction_exact[1] == 1 else -1, 0)
    else:
        return None

    uc, us, ut = longitude
    vc, vs, vt = latitude
    center, first, second, axis = (
        patch.center_exact,
        patch.first_axis_exact,
        patch.second_axis_exact,
        patch.axis_exact,
    )
    radius = patch.major_radius_exact + patch.minor_radius_exact * vc
    point = tuple(
        center[i]
        + radius * (uc * first[i] + us * second[i])
        + patch.minor_radius_exact * vs * axis[i]
        for i in range(3)
    )
    if point != endpoint.point_exact:
        return None
    return ("torus", uc, us, ut + shifts[0], vc, vs, vt + shifts[1])


def _exact_bspline_lift_uv_endpoint(
    vertex: BRepVertexRoot,
    patch: AbstractSurfacePatch,
    curve: AbstractCurve,
    endpoint: ConicParameterEndpoint,
    /,
) -> tuple[object, ...] | None:
    """Bind a conic vertex to a spline UV endpoint through an exact source lift."""
    from ._intersection_curve import exact_curve_parameter_point
    from ._patches import BSplineSurfacePatch

    parameter = endpoint.parameter_exact
    if (
        not isinstance(patch, BSplineSurfacePatch)
        or parameter is None
        or not _same_parameter_source(endpoint.carrier, curve)
        or endpoint.root.exact_point() != endpoint.point_exact
        or not vertex.supports_endpoint(endpoint)
    ):
        return None
    uv = exact_curve_parameter_point(curve, parameter)
    if uv is None or len(uv) != 2:
        return None
    for lift in vertex.source_edge_lifts:
        if (
            _same_parameter_source(lift.surface, patch)
            and _same_parameter_source(lift.pcurve, curve)
            and Fraction.from_float(lift.first)
            <= parameter
            <= Fraction.from_float(lift.last)
            and exact_curve_parameter_point(lift.curve, parameter) == endpoint.point_exact
        ):
            return ("bspline", uv[0], uv[1])
    return None


def _exact_conic_uv_endpoint(
    patch: AbstractSurfacePatch,
    curve: AbstractCurve,
    endpoint: ConicParameterEndpoint,
    /,
) -> tuple[object, ...] | None:
    """Evaluate one bound conic endpoint in an exact plane/cylinder UV chart."""
    from .._rational import rational_vector
    from ._boolean_core import PeriodicPCurve
    from ._correspondence import _exact_plane_cylinder_section_endpoint_chart
    from ._intersection_curve import exact_curve_parameter_point
    from ._patches import (
        CircleCurve,
        ConicRootLinePCurve,
        CylinderPatch,
        EllipseCurve,
        ExactLineCurve,
        LineCurve,
        PlaneCylinderSectionPCurve,
        PlanePatch,
        TorusPatch,
    )

    if isinstance(curve, CurveTrimSegment) or not _same_parameter_source(
        endpoint.carrier, curve
    ):
        return None
    source = curve
    shifts = [0, 0]
    symbols = patch.native_period_symbols()
    while isinstance(source, PeriodicPCurve):
        if not _same_parameter_source(source.patch, patch):
            return None
        for axis, shift in enumerate(source.period_shifts):
            if shift and symbols[axis] != "two_pi":
                return None
            shifts[axis] += shift
        source = source.source_curve

    parameter = endpoint.parameter_exact
    if parameter is None:
        parameter = Fraction.from_float(endpoint.root.parameter)
    domain = source.parameter_domain
    if domain is not None and not domain[0] <= float(parameter) <= domain[1]:
        return None
    if isinstance(patch, TorusPatch):
        from ._patches import _validate_source_shadows

        try:
            _validate_source_shadows(curve, set())
        except (TypeError, ValueError):
            return None
        return _torus_conic_uv_endpoint(patch, source, endpoint, (shifts[0], shifts[1]))

    if isinstance(patch, PlanePatch):
        if shifts != [0, 0]:
            return None
        if isinstance(source, PlaneCylinderSectionPCurve):
            if endpoint.parameter_exact is not None and (
                endpoint.parameter_exact != 0
                or endpoint.root.cosine_exact != 1
                or endpoint.root.sine_exact != 0
                or endpoint.root.turns != 0
            ):
                return None
            chart = _exact_plane_cylinder_section_endpoint_chart(
                endpoint.root.conic, curve, endpoint, patch
            )
            if chart is None or len(chart) != 2:
                return None
            uv = chart
        elif isinstance(source, (CircleCurve, EllipseCurve)):
            if (
                endpoint.parameter_exact is not None
                or source.ambient_dimension != 2
                or source.native_period_symbol() != "two_pi"
                or endpoint.root.exact_point() != endpoint.point_exact
            ):
                return None
            if isinstance(source, CircleCurve):
                first_radius = second_radius = source.radius_exact
            else:
                first_radius = source.first_radius_exact
                second_radius = source.second_radius_exact
            cosine, sine = (
                endpoint.root.cosine_exact,
                endpoint.root.sine_exact,
            )
            uv = tuple(
                source.center_exact[axis]
                + first_radius * cosine * source.first_axis_exact[axis]
                + second_radius * sine * source.second_axis_exact[axis]
                for axis in range(2)
            )
        elif isinstance(source, (LineCurve, ExactLineCurve)):
            if (
                endpoint.parameter_exact is None
                or endpoint.root.exact_point() != endpoint.point_exact
            ):
                return None
            uv = exact_curve_parameter_point(source, endpoint.parameter_exact)
            if uv is None or len(uv) != 2:
                return None
        else:
            return None
        origin = rational_vector(np.asarray(patch.origin))
        first_axis = rational_vector(np.asarray(patch.first_axis))
        second_axis = rational_vector(np.asarray(patch.second_axis))
        point = tuple(
            origin[axis] + uv[0] * first_axis[axis] + uv[1] * second_axis[axis]
            for axis in range(3)
        )
        if point != endpoint.point_exact:
            return None
        return ("plane", uv[0], uv[1])

    if not isinstance(patch, CylinderPatch) or symbols[0] != "two_pi" or shifts[1] != 0:
        return None
    if isinstance(source, PlaneCylinderSectionPCurve):
        chart = _exact_plane_cylinder_section_endpoint_chart(
            endpoint.root.conic, curve, endpoint, patch
        )
        if chart is None or len(chart) != 4:
            return None
        angle_cosine, angle_sine, turns, height = chart
    elif isinstance(source, ConicRootLinePCurve):
        if (
            endpoint.parameter_exact is None
            or endpoint.root.exact_point() != endpoint.point_exact
            or source.phase_shift_exact != 0
        ):
            return None
        angular_root = source.root
        circle = angular_root.conic
        if (
            not isinstance(circle, CircleCurve)
            or circle.first_axis_exact != patch.first_axis_exact
            or circle.second_axis_exact != patch.second_axis_exact
            or (
                (circle.center_exact, circle.radius_exact)
                not in (
                    ((Fraction(0), Fraction(0), Fraction(0)), Fraction(1)),
                    (patch.origin_exact, patch.radius_exact),
                )
            )
            or circle.native_period_symbol() != "two_pi"
            or angular_root.cosine_exact * angular_root.cosine_exact
            + angular_root.sine_exact * angular_root.sine_exact
            != 1
        ):
            return None
        angle_cosine, angle_sine = (
            angular_root.cosine_exact,
            angular_root.sine_exact,
        )
        turns = angular_root.turns + shifts[0]
        height = (
            source.height_origin_exact
            + source.height_direction_exact * endpoint.parameter_exact
        )
    else:
        return None

    point = tuple(
        patch.origin_exact[axis]
        + patch.radius_exact
        * (
            angle_cosine * patch.first_axis_exact[axis]
            + angle_sine * patch.second_axis_exact[axis]
        )
        + height * patch.axis_exact[axis]
        for axis in range(3)
    )
    if point != endpoint.point_exact:
        return None
    return ("cylinder", angle_cosine, angle_sine, turns, height)


def _primary_payload(root: BRepPrimaryVertexRoot, /) -> dict[str, object]:
    if isinstance(root, BRepConicVertex):
        return root.identity_payload()
    if isinstance(root, BRepRootSupport):
        return {
            "kind": "support",
            "patch": _source_payload(root.patch),
            "root": _root_definition_payload(root.root),
        }
    if isinstance(root, BRepPlacedVertex):
        return {
            "kind": "placed",
            "source_entity_id": root.source_entity_id,
            "source_point": array_tree_fingerprint(root.source_point),
            "source_root": None
            if root.source_root is None
            else vertex_root_payload(root.source_root),
            "rotation": array_tree_fingerprint(root.rotation),
            "translation": array_tree_fingerprint(root.translation),
            "affine": root.affine.identity_payload(),
        }
    return _root_definition_payload(root)


def vertex_root_payload(root: BRepVertexRoot, /) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": "brep-vertex-root",
        "primary": _primary_payload(root.primary),
        "aliases": tuple(
            {
                "patch": _source_payload(alias.patch),
                "root": _root_definition_payload(alias.root),
            }
            for alias in root.aliases
        ),
        "spatial_root": None
        if root.spatial_root is None
        else _root_definition_payload(root.spatial_root),
        "joint_root": None
        if root.joint_root is None
        else _root_definition_payload(root.joint_root),
        "source_edge_lifts": tuple(
            {
                "curve": _source_payload(lift.curve),
                "surface": _source_payload(lift.surface),
                "pcurve": _source_payload(lift.pcurve),
                "parameter_box": tuple(
                    tuple(float(value).hex() for value in row)
                    for row in lift.parameter_box
                ),
                "first": float(lift.first).hex(),
                "last": float(lift.last).hex(),
            }
            for lift in root.source_edge_lifts
        ),
    }
    if root.branch_node is not None:
        # Absent for every vertex without one, so existing identities are kept.
        payload["branch_node"] = branch_node_roots_payload(root.branch_node)
    return payload


def _primary_point_enclosure(root: BRepPrimaryVertexRoot, /) -> np.ndarray:
    if isinstance(root, BRepRootSupport):
        return root.point_enclosure()
    if isinstance(root, BRepConicVertex):
        return root.point_enclosure()
    if isinstance(root, BRepPlacedVertex):
        return root.point_enclosure()
    if isinstance(root, IntersectionCurvePointRoot):
        parameter = float(root.parameter)
        chart = min(int(np.floor(parameter)), root.curve.num_charts - 1)
        box = np.stack(
            (
                np.asarray(root.curve.chart_lower[chart]),
                np.asarray(root.curve.chart_upper[chart]),
            )
        )
        first = _patch_box(root.curve.first.patch, box[0, :2], box[1, :2])
        second = _patch_box(root.curve.second.patch, box[0, 2:], box[1, 2:])
        return np.stack(
            (np.maximum(first[0], second[0]), np.minimum(first[1], second[1]))
        )
    if isinstance(root, CurveSurfaceIntersectionRoot):
        lower, upper = np.asarray(root.parameter_lower), np.asarray(root.parameter_upper)
        if isinstance(root.curve, IntersectionCurve):
            chart = min(
                int(np.floor(float(root.parameters[0]))), root.curve.num_charts - 1
            )
            box = np.stack(
                (
                    np.asarray(root.curve.chart_lower[chart]),
                    np.asarray(root.curve.chart_upper[chart]),
                )
            )
            first = _patch_box(root.curve.first.patch, box[0, :2], box[1, :2])
            second = _patch_box(root.curve.second.patch, box[0, 2:], box[1, 2:])
            return np.stack(
                (np.maximum(first[0], second[0]), np.minimum(first[1], second[1]))
            )
        curve_box = _curve_box(root.curve, float(lower[0]), float(upper[0]))
        surface_box = _patch_box(root.surface.patch, lower[1:], upper[1:])
        return np.stack(
            (
                np.maximum(curve_box[0], surface_box[0]),
                np.minimum(curve_box[1], surface_box[1]),
            )
        )
    if isinstance(root, TripleSurfaceIntersectionRoot):
        lower, upper = np.asarray(root.parameter_lower), np.asarray(root.parameter_upper)
        boxes = tuple(
            _patch_box(region.patch, lower[start : start + 2], upper[start : start + 2])
            for region, start in ((root.first, 0), (root.second, 2), (root.third, 4))
        )
        low = np.maximum(np.maximum(boxes[0][0], boxes[1][0]), boxes[2][0])
        high = np.minimum(np.minimum(boxes[0][1], boxes[1][1]), boxes[2][1])
        return np.stack((low, high))
    raise TypeError(f"Unsupported BRep vertex root: {type(root)!r}.")


__all__ = [
    "BRepConicVertex",
    "BRepCurveSurfaceLift",
    "BRepPlacedVertex",
    "BRepRootSupport",
    "BRepVertexRoot",
    "vertex_root_payload",
]
