#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact bounds for correspondences between analytic trimmed faces.

These are mathematical primitives, not an incidence or face-region recognizer.
The caller must derive endpoint and boundary labels from actual source coedges,
including deck shifts and native-period endpoint witnesses. Bijection bounds
require proof that both faces are the stated complete strips or annuli.
Enclosure bounds instead require a certified inner/outer sandwich for the
explicitly stated source interpretation and a proved complete target face.
Topological adjacency or geometric gap tolerance alone does not establish
closure or a gap-filling region. A rounded numerical shadow of an algebraic
angle is not an endpoint witness.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ...discretization._exact_algebraic_geometry import (
    coefficient_interval,
    ExactScalar,
    RadicalElement,
)
from ._patches import CircleCurve, ConicParameterRoot, LineCurve
from ._trim_helpers import period_offset, period_scalar_bounds


Interval = tuple[Fraction, Fraction]
ExactVector = tuple[ExactScalar, ...]
ExactFrame = tuple[ExactVector, ...]


def scalar_interval(value: object, /) -> Interval:
    """Enclose an exact rational or supported algebraic coefficient."""
    if not isinstance(value, (Fraction, RadicalElement)):
        raise ValueError("An exact algebraic coefficient is required.")
    try:
        lo, hi = coefficient_interval(value, bits=128)
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("An exact algebraic coefficient is required.") from error
    return Fraction(lo), Fraction(hi)


def phase_interval(value: object, /, *, turns: Fraction = Fraction(0)) -> Interval:
    """Enclose an exact coefficient or isolated conic angle plus native turns.

    A root's isolating interval, rather than its rounded realization, carries
    the proof. A rational coefficient is the actual authored chart value, not
    a rounding promoted to an exact transcendental phase. Periods are enclosed
    by the existing directed period arithmetic.
    """
    if not isinstance(turns, Fraction):
        raise TypeError("Native-period turns must be rational.")
    if isinstance(value, ConicParameterRoot):
        lo = Fraction.from_float(value.parameter_lower)
        hi = Fraction.from_float(value.parameter_upper)
    else:
        lo, hi = scalar_interval(value)
    period_lo, period_hi = period_scalar_bounds(period_offset(Fraction(0), turns))
    return lo + Fraction.from_float(float(period_lo)), hi + Fraction.from_float(
        float(period_hi)
    )


def _interval(value: Interval, /) -> Interval:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or not all(isinstance(item, Fraction) for item in value)
        or value[0] > value[1]
    ):
        raise ValueError("A closed rational interval is required.")
    return value


def _difference(first: Interval, second: Interval, /) -> Fraction:
    a, b = _interval(first)
    c, d = _interval(second)
    return max(abs(a - d), abs(b - c))


def _magnitude(value: object, /) -> Fraction:
    lo, hi = scalar_interval(value)
    return max(abs(lo), abs(hi))


def _positive(value: object, /) -> Fraction:
    lo, hi = scalar_interval(value)
    if lo <= 0:
        raise ValueError("A strictly positive exact radius is required.")
    return hi


def periodic_partition_bound(
    source_phases: tuple[Interval, ...],
    target_phases: tuple[Interval, ...],
    source_edges: tuple[int, ...],
    target_edges: tuple[int, ...],
    /,
) -> Fraction:
    """Bound an occurrence-preserving piecewise-affine circle correspondence.

    Phases name successive vertices on one unwrapped period, one per edge;
    the last endpoint is the first phase plus 2π. The caller proves the
    phases and labels describe complete consecutive source coedge arcs. The
    corresponding piecewise-affine degree-one circle map is strictly
    increasing and its displacement is bounded by its endpoint differences.
    """
    count = len(source_phases)
    if (
        not count
        or len(target_phases) != count
        or len(source_edges) != count
        or source_edges != target_edges
        or len(set(source_edges)) != count
    ):
        raise ValueError("Circle occurrence partition does not correspond.")
    period_lower, _ = period_scalar_bounds(period_offset(Fraction(0), Fraction(1)))
    period = Fraction.from_float(float(period_lower))
    for phases in (source_phases, target_phases):
        for phase in phases:
            _interval(phase)
        if any(a[1] >= b[0] for a, b in zip(phases, phases[1:])):
            raise ValueError("Circle vertex phases are not strictly ordered.")
        if phases[-1][1] - phases[0][0] >= period:
            raise ValueError("Circle occurrence partition is not within one period.")
    return max(
        _difference(a, b) for a, b in zip(source_phases, target_phases, strict=True)
    )


def _equal_frame(first: ExactFrame, second: ExactFrame, /) -> None:
    if len(first) != 3 or len(second) != 3 or first != second:
        raise ValueError("A common exact orthonormal frame is required.")
    if any(len(axis) != 3 for axis in first):
        raise ValueError("A three-dimensional frame is required.")
    if any(
        not isinstance(value, (Fraction, RadicalElement))
        for frame in (first, second)
        for axis in frame
        for value in axis
    ):
        raise ValueError("Frame coordinates must be exact algebraic coefficients.")
    for i in range(3):
        for j in range(3):
            if sum((first[i][k] * first[j][k] for k in range(3)), Fraction(0)) != int(
                i == j
            ):
                raise ValueError("The frame is not exactly orthonormal.")


def _center_difference(first: ExactVector, second: ExactVector, /) -> Fraction:
    if len(first) != 3 or len(second) != 3:
        raise ValueError("Three-dimensional centers are required.")
    # L1 dominates Euclidean norm, without an additional square-root enclosure.
    return sum(
        (_magnitude(a - b) for a, b in zip(first, second, strict=True)), Fraction(0)
    )


@dataclass(frozen=True, slots=True)
class StripBoundary:
    """A coedge-derived pair of ordered strip limits and its seam phase.

    ``lower_edges`` and ``upper_edges`` are the complete occurrence-independent
    edge identities on each limit, after the caller has proved the pieces join
    into one closed circle.  The seam identifies the two oppositely oriented
    occurrences of the same edge.  Labels must be shared by source and target.
    ``seam_phase`` is on a common unwrapped sheet in both models.
    """

    lower: Interval
    upper: Interval
    seam_phase: Interval
    lower_edges: frozenset[int]
    upper_edges: frozenset[int]
    seam_edge: int

    def validate(self, *, periodic_profile: bool) -> None:
        low, high = _interval(self.lower), _interval(self.upper)
        _interval(self.seam_phase)
        if (
            not self.lower_edges
            or not self.upper_edges
            or self.lower_edges & self.upper_edges
        ):
            raise ValueError("Strip boundary incidence is ambiguous.")
        if self.seam_edge in self.lower_edges or self.seam_edge in self.upper_edges:
            raise ValueError("A strip seam cannot also be a boundary edge.")
        if low[1] >= high[0]:
            raise ValueError("Strip endpoint order is not proved.")
        if periodic_profile:
            period_lower, _ = period_scalar_bounds(
                period_offset(Fraction(0), Fraction(1))
            )
            if high[1] - low[0] >= Fraction.from_float(float(period_lower)):
                raise ValueError("Torus profile must occupy less than one exact period.")


def _match_strips(
    first: StripBoundary, second: StripBoundary, /, *, periodic_profile: bool
) -> tuple[Fraction, Fraction]:
    first.validate(periodic_profile=periodic_profile)
    second.validate(periodic_profile=periodic_profile)
    if (first.lower_edges, first.upper_edges, first.seam_edge) != (
        second.lower_edges,
        second.upper_edges,
        second.seam_edge,
    ):
        raise ValueError("Source and repaired strip incidences do not correspond.")
    profile = max(
        _difference(first.lower, second.lower), _difference(first.upper, second.upper)
    )
    longitude = _difference(first.seam_phase, second.seam_phase)
    return profile, longitude


def cylinder_strip_bound(
    source_center: ExactVector,
    target_center: ExactVector,
    source_frame: ExactFrame,
    target_frame: ExactFrame,
    source_radius: ExactScalar,
    target_radius: ExactScalar,
    source_trim: StripBoundary,
    target_trim: StripBoundary,
    /,
) -> Fraction:
    """Bound every point under the affine-height, seam-preserving strip map.

    The seam phases move by the same amount throughout the circle.  The endpoint
    interpolant is strictly increasing, so it is a bijection of the two strips.
    """
    _equal_frame(source_frame, target_frame)
    radius = _positive(source_radius)
    _positive(target_radius)
    height, longitude = _match_strips(source_trim, target_trim, periodic_profile=False)
    return (
        _center_difference(source_center, target_center)
        + _magnitude(source_radius - target_radius)
        + height
        + radius * longitude
    )


def torus_strip_bound(
    source_center: ExactVector,
    target_center: ExactVector,
    source_frame: ExactFrame,
    target_frame: ExactFrame,
    source_major: ExactScalar,
    target_major: ExactScalar,
    source_minor: ExactScalar,
    target_minor: ExactScalar,
    source_trim: StripBoundary,
    target_trim: StripBoundary,
    /,
) -> Fraction:
    """Bound the whole torus strip under a seam-preserving bijection.

    Exact ring-torus inequalities guarantee an injective parametrization on
    S1 times the profile interval.  The affine profile interpolant maps both
    boundary circles and never folds.  Unit-circle chord length is at most the
    absolute angular change, yielding a continuous bound without sampling.
    """
    _equal_frame(source_frame, target_frame)
    major = _positive(source_major)
    minor = _positive(source_minor)
    _positive(target_major)
    _positive(target_minor)
    if (
        scalar_interval(source_major - source_minor)[0] <= 0
        or scalar_interval(target_major - target_minor)[0] <= 0
    ):
        raise ValueError("Both supports must be regular ring tori.")
    profile, longitude = _match_strips(source_trim, target_trim, periodic_profile=True)
    return (
        _center_difference(source_center, target_center)
        + _magnitude(source_major - target_major)
        + _magnitude(source_minor - target_minor)
        + minor * profile
        + (major + minor) * longitude
    )


@dataclass(frozen=True, slots=True)
class AnnulusBoundary:
    """Exact concentric annulus radii and complete boundary edge identities."""

    inner: ExactScalar
    outer: ExactScalar
    inner_edges: frozenset[int]
    outer_edges: frozenset[int]

    def validate(self) -> None:
        _positive(self.inner)
        _positive(self.outer)
        if scalar_interval(self.outer - self.inner)[0] <= 0:
            raise ValueError("Annulus radial order is not proved.")
        if (
            not self.inner_edges
            or not self.outer_edges
            or self.inner_edges & self.outer_edges
        ):
            raise ValueError("Annulus boundary incidence is ambiguous.")


def annulus_bound(
    source_center: ExactVector,
    target_center: ExactVector,
    source_frame: ExactFrame,
    target_frame: ExactFrame,
    source: AnnulusBoundary,
    target: AnnulusBoundary,
    /,
) -> Fraction:
    """Bound an affine-radial bijection between exact planar annuli.

    The caller proves each labeled circle is complete and concentric in its
    plane chart.  Interpolation between positive, strictly ordered radii is a
    bijection, and its displacement attains its maximum at a boundary radius.
    """
    _equal_frame(source_frame, target_frame)
    source.validate()
    target.validate()
    if (source.inner_edges, source.outer_edges) != (
        target.inner_edges,
        target.outer_edges,
    ):
        raise ValueError("Source and repaired annulus incidences do not correspond.")
    return _center_difference(source_center, target_center) + max(
        _magnitude(source.inner - target.inner), _magnitude(source.outer - target.outer)
    )


@dataclass(frozen=True, slots=True)
class ProfileEnclosure:
    """Conditional inner and outer full-circle profile limits.

    The caller certifies that the source face image contains the complete
    inner strip and is contained in the complete outer strip. The intervals
    enclose the four actual limits on one common unwrapped profile sheet. This
    is an uncertainty statement, not a claim that gapped source coedges close
    or that either enclosure is the authored face.
    """

    outer_lower: Interval
    inner_lower: Interval
    inner_upper: Interval
    outer_upper: Interval

    def validate(self, *, periodic: bool = False, radial: bool = False) -> None:
        outer_lower = _interval(self.outer_lower)
        inner_lower = _interval(self.inner_lower)
        inner_upper = _interval(self.inner_upper)
        outer_upper = _interval(self.outer_upper)
        if (
            outer_lower[1] > inner_lower[0]
            or inner_lower[1] >= inner_upper[0]
            or inner_upper[1] > outer_upper[0]
        ):
            raise ValueError("Nested nonempty profile enclosures are not proved.")
        if radial and outer_lower[0] < 0:
            raise ValueError("Annular enclosure radii must be nonnegative.")
        if periodic:
            period_lower, _ = period_scalar_bounds(
                period_offset(Fraction(0), Fraction(1))
            )
            if outer_upper[1] - outer_lower[0] >= Fraction.from_float(
                float(period_lower)
            ):
                raise ValueError(
                    "Periodic profile enclosure must occupy less than one turn."
                )


def completed_profile_enclosure(
    lower: Interval,
    upper: Interval,
    /,
    *,
    periodic: bool = False,
    radial: bool = False,
) -> ProfileEnclosure:
    """Enclose an explicitly completed full-circle strip or annulus.

    The completion is the closed product of the longitude circle with the
    interval between the designated exact limits, isolated by the supplied
    rational intervals. Positive width, and (when relevant) positive radius
    or less than one native period, make the completed region an annulus. Its
    boundary circles have opposite winding and the periodic seam is a chart cut.

    This defines the interpretation, not its source-incidence evidence. The
    caller must separately prove each authored occurrence's correspondence and
    bound every replacement or connector against its original carrier and
    declared neighborhood.
    """
    low, high = _interval(lower), _interval(upper)
    result = ProfileEnclosure(
        (low[0], low[0]),
        (low[1], low[1]),
        (high[0], high[0]),
        (high[1], high[1]),
    )
    result.validate(periodic=periodic, radial=radial)
    if radial and low[0] <= 0:
        raise ValueError("A completed annulus must have a positive inner radius.")
    return result


def chart_connector_bound(
    start: tuple[Interval, Interval],
    end: tuple[Interval, Interval],
    first_lipschitz: object,
    second_lipschitz: object,
    /,
) -> Fraction:
    """Bound a straight chart connector's whole image from its start image.

    The caller supplies certified nonnegative bounds on the support's two
    partial derivative norms over the connector, and binds both interval
    endpoints to the authored occurrence and declared completion. Convexity of
    the chart segment and the fundamental theorem of calculus bound every
    connector point by the endpoint coordinate differences. This proves a
    neighborhood bound; it does not identify either endpoint with the other.
    """
    if len(start) != 2 or len(end) != 2:
        raise ValueError("A chart connector requires two-dimensional endpoints.")
    first = scalar_interval(first_lipschitz)
    second = scalar_interval(second_lipschitz)
    if first[0] < 0 or second[0] < 0:
        raise ValueError("Connector derivative bounds must be nonnegative.")
    return first[1] * _difference(start[0], end[0]) + second[1] * _difference(
        start[1], end[1]
    )


@dataclass(frozen=True, slots=True)
class CoedgeCompletionBounds:
    """Occurrence replacement and its two endpoint-connector bounds.

    ``occurrence`` bounds both directed deviations between the two complete
    parameter intervals under their affine endpoint correspondence. Each
    ``connectors`` entry bounds the straight connector between that source
    endpoint and the corresponding completed endpoint from either endpoint.
    """

    occurrence: Fraction
    connectors: tuple[Fraction, Fraction]


def _ordered_endpoints(endpoints: tuple[Interval, Interval], /) -> None:
    if len(endpoints) != 2:
        raise ValueError("A coedge requires two endpoint intervals.")
    first, last = _interval(endpoints[0]), _interval(endpoints[1])
    if first[1] >= last[0] and last[1] >= first[0]:
        raise ValueError("Coedge endpoint ordering is not proved.")


def _line_coordinate_interval(
    origin: object, direction: object, parameter: Interval, /
) -> Interval:
    a, b = scalar_interval(origin), scalar_interval(direction)
    t = _interval(parameter)
    products = (b[0] * t[0], b[0] * t[1], b[1] * t[0], b[1] * t[1])
    return a[0] + min(products), a[1] + max(products)


def line_coedge_completion_bounds(
    source: LineCurve,
    completed: LineCurve,
    source_parameters: tuple[Interval, Interval],
    completed_parameters: tuple[Interval, Interval],
    /,
) -> CoedgeCompletionBounds:
    """Certify affine line-occurrence replacement and straight connectors.

    The caller binds the parameter intervals to the authored occurrence and
    chosen completion endpoints, in occurrence order. The affine map between
    those intervals is a bijection. The Euclidean displacement is bounded by
    its endpoint L1 bounds everywhere, including the straight connectors.
    """
    if type(source) is not LineCurve or type(completed) is not LineCurve:
        raise TypeError("Line completion requires native line curves.")
    _ordered_endpoints(source_parameters)
    _ordered_endpoints(completed_parameters)
    if len(source.origin_exact) != len(completed.origin_exact):
        raise ValueError("Line ambient dimensions must agree.")
    bounds = []
    for first, second in zip(source_parameters, completed_parameters, strict=True):
        bounds.append(
            sum(
                (
                    _difference(
                        _line_coordinate_interval(a, b, first),
                        _line_coordinate_interval(c, d, second),
                    )
                    for a, b, c, d in zip(
                        source.origin_exact,
                        source.direction_exact,
                        completed.origin_exact,
                        completed.direction_exact,
                        strict=True,
                    )
                ),
                Fraction(0),
            )
        )
    return CoedgeCompletionBounds(max(bounds), (bounds[0], bounds[1]))


def circle_coedge_completion_bounds(
    source: CircleCurve,
    completed: CircleCurve,
    source_phases: tuple[Interval, Interval],
    completed_phases: tuple[Interval, Interval],
    /,
) -> CoedgeCompletionBounds:
    """Certify circular occurrence replacement and straight connectors.

    Phases must enclose actual native endpoints on corresponding unwrapped
    sheets, not merely nearby floating shadows. The exact frames must agree.
    The source-to-completion phase interpolation is bijective; its phase
    displacement is bounded by its endpoint differences. The same bound
    covers straight endpoint connectors. Occurrence identity, native turn or
    cardinal interpretation, and any surface lift are separate caller proofs.
    """
    from ._repair_bounds import _frame_norm_upper, circle_displacement_bound

    _ordered_endpoints(source_phases)
    _ordered_endpoints(completed_phases)
    support = circle_displacement_bound(source, completed)
    speed = _positive(source.radius_exact) * _frame_norm_upper(
        (source.first_axis_exact, source.second_axis_exact)
    )
    bounds = tuple(
        support + speed * _difference(first, second)
        for first, second in zip(source_phases, completed_phases, strict=True)
    )
    return CoedgeCompletionBounds(max(bounds), (bounds[0], bounds[1]))


@dataclass(frozen=True, slots=True)
class DirectedFaceBounds:
    """Euclidean source-to-target and target-to-source distance upper bounds."""

    source_to_target: Fraction
    target_to_source: Fraction


def _enclosed_profile_offsets(
    source: ProfileEnclosure, target_lower: Interval, target_upper: Interval, /
) -> tuple[Fraction, Fraction]:
    lower, upper = _interval(target_lower), _interval(target_upper)
    if lower[1] >= upper[0]:
        raise ValueError("Target profile order is not proved.")
    # Clamp a point of the outer source strip to the target strip, and a point
    # of the target strip to the inner source strip, respectively. Every
    # longitude is present, so no seam correspondence is needed.
    return (
        max(
            Fraction(0),
            lower[1] - source.outer_lower[0],
            source.outer_upper[1] - upper[0],
        ),
        max(
            Fraction(0),
            source.inner_lower[1] - lower[0],
            upper[1] - source.inner_upper[0],
        ),
    )


def cylinder_enclosure_bounds(
    source_center: ExactVector,
    target_center: ExactVector,
    source_frame: ExactFrame,
    target_frame: ExactFrame,
    source_radius: ExactScalar,
    target_radius: ExactScalar,
    source: ProfileEnclosure,
    target: StripBoundary,
    /,
) -> DirectedFaceBounds:
    """Bound both directed deviations under a certified strip sandwich.

    The source and target charts use the common exact frame, and the caller
    proves the target is the whole complete strip described by ``target``.
    Neither a source closure nor a source-to-target bijection is assumed.
    """
    _equal_frame(source_frame, target_frame)
    _positive(source_radius)
    _positive(target_radius)
    source.validate()
    target.validate(periodic_profile=False)
    outward, inward = _enclosed_profile_offsets(source, target.lower, target.upper)
    support = _center_difference(source_center, target_center) + _magnitude(
        source_radius - target_radius
    )
    return DirectedFaceBounds(support + outward, support + inward)


def torus_enclosure_bounds(
    source_center: ExactVector,
    target_center: ExactVector,
    source_frame: ExactFrame,
    target_frame: ExactFrame,
    source_major: ExactScalar,
    target_major: ExactScalar,
    source_minor: ExactScalar,
    target_minor: ExactScalar,
    source: ProfileEnclosure,
    target: StripBoundary,
    /,
) -> DirectedFaceBounds:
    """Bound both directed deviations under a certified ring-torus sandwich."""
    _equal_frame(source_frame, target_frame)
    _positive(source_major)
    _positive(target_major)
    minor = _positive(source_minor)
    _positive(target_minor)
    if (
        scalar_interval(source_major - source_minor)[0] <= 0
        or scalar_interval(target_major - target_minor)[0] <= 0
    ):
        raise ValueError("Both supports must be regular ring tori.")
    source.validate(periodic=True)
    target.validate(periodic_profile=True)
    outward, inward = _enclosed_profile_offsets(source, target.lower, target.upper)
    support = (
        _center_difference(source_center, target_center)
        + _magnitude(source_major - target_major)
        + _magnitude(source_minor - target_minor)
    )
    return DirectedFaceBounds(support + minor * outward, support + minor * inward)


def annulus_enclosure_bounds(
    source_center: ExactVector,
    target_center: ExactVector,
    source_frame: ExactFrame,
    target_frame: ExactFrame,
    source: ProfileEnclosure,
    target: AnnulusBoundary,
    /,
) -> DirectedFaceBounds:
    """Bound both directed deviations under a certified annulus sandwich."""
    _equal_frame(source_frame, target_frame)
    source.validate(radial=True)
    target.validate()
    outward, inward = _enclosed_profile_offsets(
        source, scalar_interval(target.inner), scalar_interval(target.outer)
    )
    support = _center_difference(source_center, target_center)
    return DirectedFaceBounds(support + outward, support + inward)
